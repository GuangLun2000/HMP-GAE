# fed_resume.py
# Per-round checkpoint + resume for the FL training loop in main.run_experiment.
#
# Purpose
# -------
# Colab sessions can be killed at any time (idle timeout, runtime restart,
# network drop).  Without resume, a 50-round run that dies on round 37 wastes
# ~3 hours of A100 compute.  This module persists a compact snapshot after
# each round so a re-launched run can pick up where it left off, losing at
# most the in-flight round.
#
# What gets saved
# ---------------
#   * round_num            : index of the NEXT round to run (== rounds completed)
#   * global_model_flat    : server.global_model.get_flat_params() on CPU
#                            (LoRA-only when use_lora=True, so usually <10 MB)
#   * server_history       : server.history (clean_acc, local_accuracies, ...)
#   * server_log_data      : server.log_data (per-round aggregation logs)
#   * progressive_metrics  : the lightweight metrics dict from run_experiment
#   * defense_state        : defense.state_dict()  (HMP-GAE: encoder/decoder
#                            weights + Adam state + z_hist EMA; FedAvg: {})
#   * rng                  : torch CPU / CUDA / numpy / python random states
#   * fingerprint          : config fields that must match for resume to be
#                            safe (including the complete defense_config)
#
# Why flat_params instead of full state_dict
# ------------------------------------------
# The base model is re-loaded from HuggingFace at setup_experiment() time
# (same model_name -> same weights).  Only the trainable surface (LoRA
# adapters or full FT params) changes across rounds, and that surface is
# exactly what get_flat_params() / set_flat_params() operate on.  Storing
# the flat tensor is the smallest correct representation and matches what
# the FL aggregation already uses.
#
# Atomicity
# ---------
# Each save writes to checkpoint_last.pt.tmp and then os.replace()-s into
# checkpoint_last.pt.  On POSIX (Colab Linux) this is atomic, so a process
# killed mid-write leaves the previous good checkpoint intact.
#
# Reproducibility note
# --------------------
# We restore the global torch / numpy / python RNG, but client-owned private
# RNGs (e.g. HallucinationAttackerClient._round_rng) are not snapshotted.
# Their *seeds* are derived from (client_id, round_num) so the flip pattern
# per round is identical to a non-resumed run; only the scalar flip_ratio
# drawn from hallu_flip_ratio_range can differ slightly post-resume.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


CHECKPOINT_FILE = "checkpoint_last.pt"

# Config fields that must match exactly between the saved checkpoint and the
# currently-running config for resume to be safe.  In particular, the complete
# defense_config is included so switching V4/V5/V6/V7 cannot reuse an older
# round checkpoint just because experiment_name was accidentally reused.
_FINGERPRINT_KEYS = (
    "experiment_name",
    "num_clients",
    "num_rounds",
    "num_attackers",
    "model_name",
    "defense_method",
    "use_lora",
    "lora_r",
    "seed",
    "dataset",
    "defense_config",
)


def _fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: config.get(k) for k in _FINGERPRINT_KEYS}


def checkpoint_path(results_dir: Path, subdir: str = "round_checkpoint") -> Path:
    return Path(results_dir) / subdir / CHECKPOINT_FILE


def _collect_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as e:  # noqa: BLE001 — CUDA RNG restore is best-effort
            print(f"  [resume] Warning: could not restore CUDA RNG state: {e}")


def save_round_checkpoint(
    server,
    progressive_metrics: Dict[str, Any],
    config: Dict[str, Any],
    results_dir: Path,
    next_round: int,
    subdir: str = "round_checkpoint",
) -> Optional[Path]:
    """
    Persist a resumable snapshot of the FL state after completing a round.

    Args:
        next_round: 0-indexed round to start from on resume (i.e. number of
                    completed rounds).  E.g. after run_round(0) finishes,
                    next_round=1.
    """
    if not config.get("save_round_checkpoint", True):
        return None

    ckpt_dir = Path(results_dir) / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / CHECKPOINT_FILE
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    payload: Dict[str, Any] = {
        "round_num": int(next_round),
        "global_model_flat": server.global_model.get_flat_params().detach().cpu(),
        "server_history": server.history,
        "server_log_data": server.log_data,
        "progressive_metrics": progressive_metrics,
        "defense_state": server.defense.state_dict(),
        "rng": _collect_rng_state(),
        "fingerprint": _fingerprint(config),
    }

    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def load_round_checkpoint(
    config: Dict[str, Any],
    results_dir: Path,
    subdir: str = "round_checkpoint",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try to load a previously-saved round checkpoint.

    Returns:
        (payload, reason)
        payload is None when no usable checkpoint exists; reason is a short
        human-readable string explaining why (printed by the caller).  When
        payload is not None, reason describes the resume point.
    """
    if not config.get("resume_from_checkpoint", True):
        return None, "resume disabled by config"

    path = checkpoint_path(Path(results_dir), subdir)
    if not path.is_file():
        return None, f"no checkpoint at {path}"

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return None, f"failed to load {path}: {type(e).__name__}: {e}"

    # Fingerprint check — refuse to resume if the run identity changed.
    saved_fp = payload.get("fingerprint") or {}
    cur_fp = _fingerprint(config)
    mismatches = [
        f"{k}: ckpt={saved_fp.get(k)!r} vs cfg={cur_fp.get(k)!r}"
        for k in _FINGERPRINT_KEYS
        if saved_fp.get(k) != cur_fp.get(k)
    ]
    if mismatches:
        return None, "fingerprint mismatch (" + "; ".join(mismatches) + ")"

    next_round = int(payload.get("round_num", 0))
    total = int(config.get("num_rounds", 0))
    if next_round >= total:
        return payload, f"checkpoint already at round {next_round}/{total} (training complete)"
    return payload, f"resuming from round {next_round + 1}/{total}"


def apply_round_checkpoint(
    server,
    progressive_metrics: Dict[str, Any],
    payload: Dict[str, Any],
) -> int:
    """
    Restore server / metrics state from a loaded checkpoint payload.

    Returns:
        next_round (0-indexed) the caller should pass into the round loop.
    """
    # 1) global model parameters (LoRA-only when use_lora=True)
    flat = payload["global_model_flat"]
    # set_flat_params handles dtype/device internally (param.data.copy_).
    server.global_model.set_flat_params(flat.to(
        next(server.global_model.parameters()).device
    ))

    # 2) server histories and per-round logs
    server.history = payload["server_history"]
    server.log_data = payload["server_log_data"]

    # 3) progressive_metrics (mutated in place so the caller's reference stays valid)
    pm_saved = payload["progressive_metrics"]
    progressive_metrics.clear()
    progressive_metrics.update(pm_saved)

    # 4) defense state (HMP-GAE runtime: encoder/decoder + optim + z_hist)
    defense_state = payload.get("defense_state") or {}
    server.defense.load_state_dict(defense_state)

    # 5) RNG
    _restore_rng_state(payload.get("rng") or {})

    return int(payload["round_num"])
