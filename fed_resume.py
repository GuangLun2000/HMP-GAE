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
# We restore the global torch / numpy / python RNG. Hallucination label masks
# and scalar flip ratios are both deterministic functions of
# (client flip_seed, round_num), so no client-private RNG state is required and
# a resumed round is identical to the corresponding uninterrupted round.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


CHECKPOINT_FILE = "checkpoint_last.pt"
FINGERPRINT_SCHEMA = 3

# Config fields that can change the FL trajectory and therefore must match
# exactly between the saved checkpoint and the current run. Output-only and
# post-FL evaluation keys are intentionally excluded. The experiment name still
# must be unique per arm so result artifacts remain unambiguous.
_FINGERPRINT_KEYS = (
    "experiment_name",
    "seed",
    "num_clients",
    "num_rounds",
    "num_attackers",
    "client_lr",
    "server_lr",
    "batch_size",
    "test_batch_size",
    "local_epochs",
    "grad_clip_norm",
    "alpha",
    "dataset",
    "num_labels",
    "max_length",
    "data_distribution",
    "dirichlet_alpha",
    "dataset_size_limit",
    "server_reference_size",
    "server_reference_seed",
    "server_reference_stratified",
    "model_name",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "attack_method",
    "attack_start_round",
    "hallu_flip_ratio",
    "hallu_flip_mode",
    "hallu_flip_map",
    "hallu_target_class",
    "hallu_attack_start_round",
    "hallu_per_round_reseed",
    "hallu_flip_ratio_range",
    "sign_flip_scale",
    "sign_flip_attack_start_round",
    "gaussian_std_scale",
    "gaussian_attack_start_round",
    "alie_z_max",
    "alie_attack_start_round",
    "defense_method",
    "defense_config",
    "semantic_probe_size",
)


def _fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_schema": FINGERPRINT_SCHEMA,
        **{k: config.get(k) for k in _FINGERPRINT_KEYS},
    }


def _fingerprint_mismatches(
    saved_fp: Dict[str, Any], config: Dict[str, Any]
) -> list[str]:
    """Compare a checkpoint fingerprint without stranding legacy snapshots.

    Schema-3 checkpoints require every trajectory key. Older checkpoints are
    compared on the keys they actually recorded (which already included the
    experiment, seed, model, dataset and complete defense config); new saves
    immediately gain the stricter contract.
    """
    cur_fp = _fingerprint(config)
    saved_schema = int(saved_fp.get("_schema", 1))
    if saved_schema >= FINGERPRINT_SCHEMA:
        keys = _FINGERPRINT_KEYS
    else:
        keys = tuple(k for k in saved_fp if k != "_schema")
    mismatches = [
        f"{k}: ckpt={saved_fp.get(k)!r} vs cfg={cur_fp.get(k)!r}"
        for k in keys
        if saved_fp.get(k) != cur_fp.get(k)
    ]
    # Schema 1/2 checkpoints predate the disjoint server-reference protocol.
    # They may have used the official test loader inside aggregation, so a run
    # that now requests a reference holdout must never resume from them even
    # when all of their older keys happen to match.
    if saved_schema < FINGERPRINT_SCHEMA and int(
        config.get("server_reference_size") or 0
    ) > 0:
        mismatches.append(
            "fingerprint schema: checkpoint predates server-reference isolation"
        )
    return mismatches


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
    mismatches = _fingerprint_mismatches(saved_fp, config)
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
