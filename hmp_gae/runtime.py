# hmp_gae/runtime.py
# HMPGAERuntime: the stateful engine that performs one round of HMP-GAE
# self-supervised training + trust scoring + weighted aggregation.
#
# Called from defense.HMPGAEDefense. Keeps:
#   - a fixed random projection (buffer, not trained)
#   - a NodeFeatureEncoder (trained jointly)
#   - an HMPEncoder (trained jointly)
#   - a HyperedgeDecoder (trained jointly)
#   - an EMA cache Z_hist of previous-round embeddings (detached)
#
# The whole runtime defaults to CPU because N is small and running on CPU
# avoids frequent host<->device transfers of the aggregated update.

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .node_features import (
    FixedRandomProjection,
    NodeFeatureEncoder,
    compute_node_features,
    CONTEXT_DIM,
)
from .hypergraph import knn_hypergraph
from .encoder import HMPEncoder
from .decoder import inner_product_decoder, HyperedgeDecoder
from .losses import total_loss
from .trust_scorer import (
    compute_trust_weights,
    weighted_aggregate,
    reject_then_weighted,
    reject_soft_weighted,
    gate_diagnostics,
    v4_cse_reject_weights,
    v5_cse_reject_weights,
    v6_cse_reject_geo_weights,
    v7_cse_reject_corrob_weights,
)


class HMPGAERuntime:
    def __init__(
        self,
        num_clients: int,
        flat_update_dim: int,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.num_clients = int(num_clients)
        self.flat_update_dim = int(flat_update_dim)
        self.cfg = dict(config or {})
        self.device = torch.device(self.cfg.get("device", device))

        # ---- Hyperparameters with sane defaults ---- #
        self.proj_dim = int(self.cfg.get("proj_dim", 64))
        self.eta_dim = int(self.cfg.get("eta_dim", 64))
        self.hidden_dim = int(self.cfg.get("hidden_dim", 64))
        self.latent_dim = int(self.cfg.get("latent_dim", 32))
        self.num_hmp_layers = int(self.cfg.get("num_hmp_layers", 2))
        self.knn_k = int(self.cfg.get("knn_k", 3))

        self.train_steps_per_round = int(self.cfg.get("train_steps_per_round", 5))
        self.train_lr = float(self.cfg.get("train_lr", 1e-3))
        self.weight_decay = float(self.cfg.get("weight_decay", 1e-5))
        self.lambda_H = float(self.cfg.get("lambda_H", 1.0))
        self.lambda_A = float(self.cfg.get("lambda_A", 1.0))
        self.lambda_hist = float(self.cfg.get("lambda_hist", 0.5))

        # Trust score signal weights.
        # Defaults: graph-structural signal dominates, with a small amount of
        # decoder-residual refinement. Historical deviation is disabled by
        # default (benign clients drift more than attackers during real
        # learning, which can invert the signal).
        self.graph_weight = float(self.cfg.get("graph_weight", 1.0))
        self.residual_weight_alpha = float(self.cfg.get("residual_weight_alpha", 0.3))
        self.hist_weight_beta = float(self.cfg.get("hist_weight_beta", 0.0))
        # Round-dependent phase gating: if set, hist signal is only enabled for
        # round_num < hist_warmup_rounds; otherwise β_eff = 0.  None = always
        # on (backward compatible with all V1/V2 runs including Y2 and Y5).
        #
        # Rationale: Y5 (2026-05-22) showed hist_dev signal direction is
        # correct in Phase 1 (R1-R11, 100% atk_hist_dev > bgn_hist_dev) but
        # inverts in Phase 3 (R26+, only 28% correct).  Gating hist to the
        # warmup window exploits the good phase and avoids the steady-state
        # inversion.
        _hwr = self.cfg.get("hist_warmup_rounds", None)
        self.hist_warmup_rounds: Optional[int] = (
            None if _hwr is None else int(_hwr)
        )
        # Per-sample semantic-divergence signal weight. Off by default (=0.0)
        # so existing experiments reproduce bit-for-bit; enable with
        # defense_config.semantic_weight > 0 once the server is forwarding a
        # probe_distributions tensor.
        self.semantic_weight = float(self.cfg.get("semantic_weight", 0.0))
        self.softmax_tau = float(self.cfg.get("softmax_tau", 0.1))

        # ---- Robust trust scoring (all defaults = pre-existing behavior) ----
        # zscore_mode: 'std' (classic mean/std, breaks down as attacker
        #   fraction grows) or 'mad' (median/MAD, robust to <50% attackers).
        # zscore_clip: symmetric bound on |z| so a single extreme outlier
        #   cannot dominate the weighted signal sum.
        self.zscore_mode = str(self.cfg.get("zscore_mode", "std"))
        _zc = self.cfg.get("zscore_clip", None)
        self.zscore_clip: Optional[float] = None if _zc is None else float(_zc)
        # gate_rezscore: True keeps the historical second z-score on the
        # combined gate (forces every round onto a +-sigma scale -> an
        # all-benign round always gates its most extreme client). False gates
        # on -trust.s directly; retune reject_z_threshold (~2.5) with it.
        self.gate_rezscore = bool(self.cfg.get("gate_rezscore", True))
        # sus_ema_beta: cross-round EMA on the suspicion score. 0 = off.
        # Benign clients take turns being the round's extreme value, so their
        # EMA reverts to ~0; attackers are suspicious every round, so their
        # EMA stays high. Adds ~1/(1-beta) rounds of detection lag.
        self.sus_ema_beta = float(self.cfg.get("sus_ema_beta", 0.0))
        # Semantic-divergence reference: 'pairwise' (historical) or 'median'
        # (robust per-sample consensus; recommended under non-IID).
        self.semantic_reference = str(self.cfg.get("semantic_reference", "pairwise"))
        self.semantic_confidence_weight = bool(
            self.cfg.get("semantic_confidence_weight", False)
        )

        # Trust-to-weight mapping:
        #   'reject_then_fedavg' (default, recommended for V1): use trust
        #     signals to flag attackers (graph_residual_z > threshold), then
        #     aggregate the non-rejected with their natural FedAvg weights.
        #     Preserves collaborative learning benefit among benigns.
        #   'softmax': pure softmax of trust logits. Simpler but tends to
        #     concentrate weight on 1-2 benign clients when their residuals
        #     are nearly tied.
        # Trust-to-weight mapping mode:
        #   'soft_reject_fedavg' (default, recommended): sigmoid gate on
        #     graph_residual_z, scales suspicious clients down smoothly;
        #     robust to threshold miscalibration across different N values.
        #   'reject_then_fedavg': hard binary rejection via z-score threshold,
        #     then data-size FedAvg among kept clients.  Fragile near threshold.
        #   'softmax': pure softmax on trust logits; tends to concentrate
        #     weight on 1-2 benign clients when residuals are tied.
        #   'v4_cse_reject' (V4, 2026-07-28): rejection driven by the ABSOLUTE
        #     per-client full-test CSE, pool-median normalised and rank-capped
        #     (see trust_scorer.v4_cse_reject_weights). The four geometry
        #     channels stay computed + logged as diagnostics but no longer
        #     drive rejection. Requires the server to evaluate local CSE
        #     BEFORE aggregation and pass it via aggregate(local_cse=...).
        #   'v5_cse_reject' (V5, 2026-08-06): V4's flag decision byte-identical,
        #     but the flagged-client multiplier is a linear ramp in the CSE
        #     ratio (trust_scorer.v5_cse_reject_weights): mild just past tau,
        #     v5_m_floor at ratio >= v5_r_hard. Same local-CSE requirement and
        #     server eval timing as V4.
        #   'v6_cse_reject_geo' (V6, 2026-08-07): V5's ramp times a ONE-SIDED
        #     read-out of the geometry gate on flagged clients only
        #     (trust_scorer.v6_cse_reject_geo_weights). Restores the hypergraph
        #     signal to alpha — V4/V5 compute it and multiply it by zero —
        #     without letting it soften any CSE-driven penalty. v6_geo_floor=1.0
        #     reproduces V5 exactly. Same local-CSE requirement / eval timing /
        #     crafts_update incompatibility as V4.
        #   'v7_cse_reject_corrob' (V7, 2026-08-08): V6 byte-identical, plus a
        #     Tier-2 flag armed ONLY inside the CSE cold-start window
        #     (v7_round_min..v7_round_max, 1-indexed): a client whose CSE
        #     ratio is elevated but sub-1.85 (r > v7_tau_lo) AND whose RAW
        #     hypergraph isolation crosses v7_iso_min gets the constant
        #     v7_corrob_mult. Iso-corroborated threshold discount — the
        #     hypergraph is what makes a sub-1.85 threshold safe; it never
        #     flags alone and never displaces a CSE flag
        #     (trust_scorer.v7_cse_reject_corrob_weights). v7_round_max=0
        #     reproduces V6 exactly. ⚠ Constants provisional until the
        #     replay calibration (replay_v7_calibration.py) passes — do NOT
        #     launch a V7 training run before that (docs/DECISION.md "V7").
        self.trust_mode = str(
            self.cfg.get("trust_mode", "soft_reject_fedavg")
        ).strip().lower()
        # Unknown values used to silently fall into the 'softmax' catch-all
        # branch — a typo'd mode (e.g. 'v4_cse_rejects') would quietly run a
        # different aggregation rule for 50 rounds. Fail loudly instead.
        _known_modes = {
            "soft_reject_fedavg", "reject_then_fedavg", "softmax",
            "v4_cse_reject", "v5_cse_reject", "v6_cse_reject_geo",
            "v7_cse_reject_corrob",
        }
        if self.trust_mode not in _known_modes:
            raise ValueError(
                f"Unknown trust_mode={self.trust_mode!r}; expected one of "
                f"{sorted(_known_modes)}"
            )
        # --- V4 knobs (inert unless trust_mode == 'v4_cse_reject') ---
        #   v4_tau_ratio  : pre-registered 1.85 (zero-FP plateau [1.785, 1.90]
        #                   over 51 archived runs); do NOT re-tune post hoc.
        #   k_cap         : REUSES defense_config.num_byzantine — no new
        #                   hyperparameter. Rule is sound only for
        #                   #attackers <= k_cap < N/2 (validated below).
        #   v4_reject_mult: rejection multiplier, default 0.10 (soft). 0.0 =
        #                   HARD REMOVAL — a flagged client contributes nothing
        #                   to that round's aggregate. Legal since 2026-08-07 as
        #                   an explicit pre-registered ablation arm (docs/
        #                   DECISION.md "V4-remove"); as a DEFAULT it stays
        #                   rejected (FoolsGold's mechanism, the archive's
        #                   worst Qwen-Yahoo PPL). Sweep set {0.10, 0.05,
        #                   0.02, 0.0}.
        self.v4_tau_ratio = float(self.cfg.get("v4_tau_ratio", 1.85))
        self.v4_reject_mult = float(self.cfg.get("v4_reject_mult", 0.10))
        self.v4_k_cap = int(self.cfg.get("num_byzantine", 2))
        # --- V5 knobs (inert unless trust_mode == 'v5_cse_reject') ---
        #   v5_m_floor: multiplier floor — plays v4_reject_mult's role at
        #               ratio >= v5_r_hard and inherits its rules (never 0.0;
        #               the pre-authorized {0.05, 0.02} sweep knob).
        #   v5_r_hard : ratio where the ramp saturates. Pre-registered 2.5
        #               from the archived V4 runs' steady-state attacker
        #               ratio minima (2.38 Llama-Yahoo / 3.72 Llama-AG /
        #               4.09 Qwen-AG): steady-state attackers saturate to
        #               m_floor (≈V4 behavior, CSE risk bounded) while
        #               borderline flags near tau stay mild.
        self.v5_m_floor = float(self.cfg.get("v5_m_floor", 0.10))
        self.v5_r_hard = float(self.cfg.get("v5_r_hard", 2.5))
        # --- V6 knob (inert unless trust_mode == 'v6_cse_reject_geo') ---
        #   v6_geo_floor: the strongest geometric discount a flagged client can
        #               receive on top of V5's CSE ramp, i.e. the flagged
        #               multiplier is m_cse * (geo_floor + (1-geo_floor)*gate).
        #               1.0 = geometry disabled = V5 element-for-element (the
        #               Run-0 regression guard); 0.5 = PRE-REGISTERED default
        #               (2026-08-07). Never re-tune after seeing a run: if the
        #               logged geo_mult sits at ~1.0 all round, the honest
        #               report is "geometry did not act", not a lower floor.
        #               V6 reuses v5_m_floor / v5_r_hard for Stage 2 unchanged.
        self.v6_geo_floor = float(self.cfg.get("v6_geo_floor", 0.5))
        # --- V7 knobs (inert unless trust_mode == 'v7_cse_reject_corrob') ---
        # ALL PROVISIONAL until replay_v7_calibration.py picks them from the
        # pre-committed grids on the archived logs (docs/DECISION.md "V7");
        # after that they are pre-registered and never re-tuned post hoc.
        #   v7_tau_lo     : Tier-2 CSE floor, 1 < tau_lo < v4_tau_ratio.
        #                   Candidates {1.30..1.80 step 0.05}; provisional 1.40.
        #                   NOTE tau_lo < 1.7833 (archived clean benign max
        #                   ratio) — the iso conjunct is what keeps clean
        #                   federations exact, and that premise is a replay
        #                   PASS criterion, not an assumption.
        #   v7_iso_min    : absolute floor on RAW graph_residual, at an
        #                   inter-level midpoint of the quantized channel
        #                   (levels are multiples of 1/(N-1)): 7/12 = only
        #                   reach<=2 at N=7/k=2; candidate 5/12 also admits
        #                   reach=3. Never place it ON a level.
        #   v7_corrob_mult: constant Tier-2 multiplier in (0,1) — never 0
        #                   (weaker evidence tier than a full CSE flag), never
        #                   gate-modulated (archived attacker gate ~0.766
        #                   would shrink the penalty into seed noise).
        #   v7_round_min/max: cold window on the 1-INDEXED logged round
        #                   (same units as the archives' 'round' channel).
        #                   min=3 — R1-R2 have no recall (attacker r~1) and
        #                   the noisiest signals, so they carry pure FP risk.
        #                   max=0 disables the window = V6 bit-identical (the
        #                   Run-0 wiring regression arm). Candidates {5,8,10}.
        self.v7_tau_lo = float(self.cfg.get("v7_tau_lo", 1.40))
        self.v7_iso_min = float(self.cfg.get("v7_iso_min", 7.0 / 12.0))
        self.v7_corrob_mult = float(self.cfg.get("v7_corrob_mult", 0.5))
        self.v7_round_min = int(self.cfg.get("v7_round_min", 3))
        self.v7_round_max = int(self.cfg.get("v7_round_max", 10))
        if self.trust_mode in (
            "v4_cse_reject", "v5_cse_reject", "v6_cse_reject_geo",
            "v7_cse_reject_corrob",
        ):
            # The pool median must be benign-controlled: majority-poisoned
            # federations invert the rule. Raise (not assert) so a bad config
            # fails loudly at construction — _lazy_init runs OUTSIDE the
            # FedAvg-fallback try/except in HMPGAEDefense.aggregate.
            if not (0 < self.v4_k_cap and 2 * self.v4_k_cap < self.num_clients):
                raise ValueError(
                    f"trust_mode='{self.trust_mode}' requires "
                    f"0 < num_byzantine < N/2; got num_byzantine={self.v4_k_cap} "
                    f"with N={self.num_clients}"
                )
            if not (self.v4_tau_ratio > 1.0):
                raise ValueError(
                    f"v4_tau_ratio must be > 1.0, got {self.v4_tau_ratio}"
                )
        if self.trust_mode == "v4_cse_reject":
            # Exactly 0.0 (hard removal) became legal on 2026-08-07 as a
            # pre-registered ablation arm — the detect-then-remove paper
            # story (docs/DECISION.md "V4-remove"). The bound is now closed
            # at 0: negatives would flip a flagged client's update sign,
            # which is an attack, not a penalty. v5_m_floor / v6_geo_floor
            # below keep their OPEN lower bounds — widening those is a
            # separate decision this entry deliberately does not make.
            if not (0.0 <= self.v4_reject_mult < 1.0):
                raise ValueError(
                    "v4_reject_mult must be in [0, 1) — 0.0 = hard removal "
                    "(pre-registered ablation arm, DECISION 2026-08-07); "
                    f"got {self.v4_reject_mult}"
                )
        if self.trust_mode in (
            "v5_cse_reject", "v6_cse_reject_geo", "v7_cse_reject_corrob"
        ):
            # V6 reuses V5's Stage-2 ramp verbatim, so it inherits both
            # guards; V7 embeds V6 as its Stage A and inherits them again.
            if not (0.0 < self.v5_m_floor < 1.0):
                raise ValueError(
                    "v5_m_floor must be in (0, 1) — 0.0 is FoolsGold-style "
                    f"hard zeroing (rejected); got {self.v5_m_floor}"
                )
            if not (self.v5_r_hard > self.v4_tau_ratio):
                raise ValueError(
                    "v5_r_hard must be > v4_tau_ratio (the ramp divides by "
                    f"their difference); got v5_r_hard={self.v5_r_hard}, "
                    f"v4_tau_ratio={self.v4_tau_ratio}"
                )
        if self.trust_mode in ("v6_cse_reject_geo", "v7_cse_reject_corrob"):
            # Upper bound is CLOSED: geo_floor == 1.0 is the V5-equivalence
            # point and must stay legal (it is the Run-0 regression guard).
            # Validated here, in __init__, and NOT inside aggregate(): a
            # ValueError raised from aggregate() is swallowed by
            # HMPGAEDefense.aggregate's FedAvg safety net, which would turn a
            # config typo into 50 silent rounds of plain FedAvg.
            # V7 embeds V6's Stage 3 unchanged, so the guard applies there too.
            if not (0.0 < self.v6_geo_floor <= 1.0):
                raise ValueError(
                    "v6_geo_floor must be in (0, 1] — 1.0 disables the "
                    "geometry term (= V5 exactly), 0.0 would let the gate "
                    f"zero a client outright (rejected); got {self.v6_geo_floor}"
                )
        if self.trust_mode == "v7_cse_reject_corrob":
            # Same __init__-not-aggregate() rationale as the V6 guard above.
            # The weights function re-validates (it is also called directly in
            # tests/replay), but a config typo must die HERE, loudly, before
            # the FedAvg safety net can eat it.
            if not (1.0 < self.v7_tau_lo < self.v4_tau_ratio):
                raise ValueError(
                    "v7_tau_lo must satisfy 1 < tau_lo < v4_tau_ratio; got "
                    f"v7_tau_lo={self.v7_tau_lo} v4_tau_ratio={self.v4_tau_ratio}"
                )
            if not (0.0 < self.v7_corrob_mult < 1.0):
                raise ValueError(
                    "v7_corrob_mult must be in (0, 1) — 0.0 is hard zeroing "
                    "on the weaker (corroborated) evidence tier, rejected a "
                    f"fortiori; got {self.v7_corrob_mult}"
                )
            if not (0.0 < self.v7_iso_min < 1.0):
                raise ValueError(
                    f"v7_iso_min must be in (0, 1); got {self.v7_iso_min}"
                )
            if self.v7_round_min < 1:
                raise ValueError(
                    "v7_round_min is 1-indexed (logged-round units) and must "
                    f"be >= 1; got {self.v7_round_min}"
                )
            if self.v7_round_max != 0 and self.v7_round_max < self.v7_round_min:
                raise ValueError(
                    "v7_round_max must be 0 (window disabled = V6 exactly) or "
                    f">= v7_round_min; got v7_round_max={self.v7_round_max} "
                    f"v7_round_min={self.v7_round_min}"
                )
        # reject_z_threshold: midpoint of the sigmoid gate for soft_reject_fedavg,
        # or the hard cutoff for reject_then_fedavg.  Same scale (gr_z units)
        # for both modes so switching is a single config-key change.
        self.reject_z_threshold = float(self.cfg.get("reject_z_threshold", 0.75))
        # soft_reject_k: sigmoid steepness for soft_reject_fedavg.
        # k=1 → very smooth; k=2 → recommended; k=3 → near-binary.
        self.soft_reject_k = float(self.cfg.get("soft_reject_k", 2.0))
        self.keep_min = int(self.cfg.get("keep_min", 1))
        # gate_signal: which suspicion signal drives the rejection gate.
        #   'graph'    -> graph_residual_z only (backward compatible).
        #   'combined' -> z-score(-trust.s), folds in all enabled signals.
        # Auto-promoted to 'combined' when semantic_weight > 0 unless the user
        # has explicitly set gate_signal in the config.
        cfg_gate = self.cfg.get("gate_signal", None)
        if cfg_gate is None:
            self.gate_signal = "combined" if self.semantic_weight > 0.0 else "graph"
        else:
            self.gate_signal = str(cfg_gate)
        self.hist_ema_beta = float(self.cfg.get("hist_ema_beta", 0.9))
        # graph_min_distinct: zero the graph channel (and drop its weight from
        # weight_norm) in rounds where graph_residual resolves fewer than this
        # many distinct values across clients. With knn_k=2 and N=7 the channel
        # takes only 4-5 discrete levels (multiples of 1/6) and its MAD is
        # exactly 0 in most Yahoo rounds. 0 = off (legacy behavior).
        self.graph_min_distinct = int(self.cfg.get("graph_min_distinct", 0))
        self.proj_seed = int(self.cfg.get("random_proj_seed", 42))
        # Cold-start policy: Signal 1 (graph_residual from k-NN hypergraph)
        # is computed from raw projected updates and does NOT need Z_hist.
        # hist_weight_beta defaults to 0.0 so hist_dev has zero weight anyway.
        # Defaulting to False: HMP-GAE detects from round 0 using the graph
        # signal, which is available immediately.  Set True only if you observe
        # spurious rejections on round 0 with very small N.
        self.cold_start_fallback = bool(self.cfg.get("cold_start_fallback", False))
        self.min_history_for_trust = int(self.cfg.get("min_history_for_trust", 1))

        # ---- Modules ---- #
        self.projection = FixedRandomProjection(
            d_in=self.flat_update_dim, d_out=self.proj_dim, seed=self.proj_seed
        )
        self.node_encoder = NodeFeatureEncoder(
            proj_dim=self.proj_dim, hist_dim=self.latent_dim, eta_dim=self.eta_dim
        ).to(self.device)
        self.hmp_encoder = HMPEncoder(
            eta_dim=self.eta_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_hmp_layers,
        ).to(self.device)
        # M = N: one hyperedge per client (center-node construction).
        self.hyperedge_decoder = HyperedgeDecoder(
            latent_dim=self.latent_dim, num_hyperedges=self.num_clients
        ).to(self.device)

        params = (
            list(self.node_encoder.parameters())
            + list(self.hmp_encoder.parameters())
            + list(self.hyperedge_decoder.parameters())
        )
        self.optim = torch.optim.Adam(
            params, lr=self.train_lr, weight_decay=0.0  # L2 handled in loss
        )

        # ---- State ---- #
        # z_hist is a per-client EMA buffer of the latent embedding.
        self.z_hist: Dict[int, torch.Tensor] = {}
        # sus_ema is a per-client EMA of the suspicion score driving the gate
        # (only maintained when sus_ema_beta > 0).
        self.sus_ema: Dict[int, float] = {}

    # --------------------------------------------------------------------- #
    # Helper: pack updates into a tensor aligned with self.num_clients order #
    # --------------------------------------------------------------------- #

    def _stack_updates(self, updates: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([u.detach() for u in updates]).to(
            device=self.device, dtype=torch.float32
        )
        return stacked

    def _history_matrix(
        self, client_ids: List[int]
    ) -> Tuple[torch.Tensor, bool]:
        """
        Build (N, latent_dim) history matrix indexed by the given client_ids.
        Returns (matrix, has_any_history).

        Cold-start clients contribute zero rows.
        """
        n = len(client_ids)
        out = torch.zeros(n, self.latent_dim, device=self.device, dtype=torch.float32)
        any_hist = False
        for i, cid in enumerate(client_ids):
            h = self.z_hist.get(int(cid))
            if h is not None:
                out[i] = h.to(device=self.device, dtype=torch.float32)
                any_hist = True
        return out, any_hist

    def _smooth_suspicion(
        self, client_ids: List[int], sus_raw: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-round EMA of the per-client suspicion score.

        Returns sus_raw unchanged when sus_ema_beta <= 0. Otherwise updates
        self.sus_ema in place (first observation initializes the EMA) and
        returns the smoothed vector aligned with client_ids.
        """
        beta = self.sus_ema_beta
        if beta <= 0.0:
            return sus_raw
        out = sus_raw.clone()
        for i, cid in enumerate(client_ids):
            key = int(cid)
            cur = float(sus_raw[i].item())
            prev = self.sus_ema.get(key)
            smoothed = cur if prev is None else beta * prev + (1.0 - beta) * cur
            self.sus_ema[key] = smoothed
            out[i] = smoothed
        return out

    def _update_history(self, client_ids: List[int], Z_new: torch.Tensor) -> None:
        Z_detached = Z_new.detach()
        beta = self.hist_ema_beta
        for i, cid in enumerate(client_ids):
            key = int(cid)
            prev = self.z_hist.get(key)
            cur = Z_detached[i].clone().cpu()
            if prev is None:
                self.z_hist[key] = cur
            else:
                self.z_hist[key] = beta * prev + (1.0 - beta) * cur

    # --------------------------------------------------------------------- #
    # Main entry                                                            #
    # --------------------------------------------------------------------- #

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        probe_distributions: "torch.Tensor | None" = None,
        local_cse: "List[float] | None" = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t0 = time.perf_counter()

        N = len(updates)
        assert N == len(client_ids) == len(data_sizes)

        # ---- 1) pack updates on device ---- #
        updates_stack = self._stack_updates(updates)   # (N, d_update)
        hist_mat, has_hist = self._history_matrix(client_ids)
        Z_hist_arg = hist_mat if has_hist else None

        # Probe distributions (N, K, C) for the semantic-divergence signal.
        # When the server has not provided one, the trust scorer simply leaves
        # sem_div = 0 and the combination falls back to graph + recon signals.
        if probe_distributions is not None:
            probe_arg = probe_distributions.detach().to(
                device=self.device, dtype=torch.float32
            )
            if probe_arg.dim() != 3 or probe_arg.shape[0] != N:
                raise ValueError(
                    "probe_distributions must be (N, K, C) with matching N; "
                    f"got {tuple(probe_arg.shape)} for N={N}"
                )
        else:
            probe_arg = None

        # ---- 2) self-supervised training steps ---- #
        self.node_encoder.train()
        self.hmp_encoder.train()
        self.hyperedge_decoder.train()
        last_loss_bundle = None
        for step in range(self.train_steps_per_round):
            self.optim.zero_grad(set_to_none=True)

            eta = compute_node_features(
                updates=updates_stack,
                projection=self.projection,
                encoder=self.node_encoder,
                history=hist_mat if has_hist else None,
            )
            H, D_V_inv, D_E_inv = knn_hypergraph(eta, k=self.knn_k)
            Z = self.hmp_encoder(eta, H, D_V_inv, D_E_inv)

            _, A_probs = inner_product_decoder(Z)
            H_hat_logits, _ = self.hyperedge_decoder(Z)

            bundle = total_loss(
                H=H,
                H_hat_logits=H_hat_logits,
                A_hat=A_probs,
                Z=Z,
                Z_hist=Z_hist_arg,
                lambda_H=self.lambda_H,
                lambda_A=self.lambda_A,
                lambda_hist=self.lambda_hist,
                weight_decay=self.weight_decay,
                params=list(self.node_encoder.parameters())
                    + list(self.hmp_encoder.parameters())
                    + list(self.hyperedge_decoder.parameters()),
            )
            bundle.total.backward()
            # Mild gradient clipping for stability when N is small.
            torch.nn.utils.clip_grad_norm_(
                list(self.node_encoder.parameters())
                + list(self.hmp_encoder.parameters())
                + list(self.hyperedge_decoder.parameters()),
                max_norm=5.0,
            )
            self.optim.step()
            last_loss_bundle = bundle

        # ---- 3) eval mode forward for trust scoring ---- #
        self.node_encoder.eval()
        self.hmp_encoder.eval()
        self.hyperedge_decoder.eval()
        with torch.no_grad():
            eta = compute_node_features(
                updates=updates_stack,
                projection=self.projection,
                encoder=self.node_encoder,
                history=hist_mat if has_hist else None,
            )
            H, D_V_inv, D_E_inv = knn_hypergraph(eta, k=self.knn_k)
            Z = self.hmp_encoder(eta, H, D_V_inv, D_E_inv)
            _, A_probs = inner_product_decoder(Z)

            # Phase-gated β: hist signal only active in warmup window.
            # hist_warmup_rounds is None  -> always on (backward compatible)
            # hist_warmup_rounds is int N -> active for round_num < N (0-indexed)
            if (self.hist_warmup_rounds is not None
                    and int(round_num) >= int(self.hist_warmup_rounds)):
                hist_weight_beta_eff = 0.0
            else:
                hist_weight_beta_eff = self.hist_weight_beta

            trust = compute_trust_weights(
                A_hat=A_probs,
                Z=Z,
                Z_hist=Z_hist_arg,
                H=H,
                graph_weight=self.graph_weight,
                residual_weight_alpha=self.residual_weight_alpha,
                hist_weight_beta=hist_weight_beta_eff,
                softmax_tau=self.softmax_tau,
                probe_distributions=probe_arg,
                semantic_weight=self.semantic_weight,
                zscore_mode=self.zscore_mode,
                zscore_clip=self.zscore_clip,
                semantic_reference=self.semantic_reference,
                semantic_confidence_weight=self.semantic_confidence_weight,
                graph_min_distinct=self.graph_min_distinct,
            )

        # ---- 3b) Suspicion score + cross-round EMA smoothing ---- #
        # sus_raw is this round's suspicion (per gate_signal / gate_rezscore);
        # sus_used is what actually drives the gate: EMA-smoothed when
        # sus_ema_beta > 0, identical to sus_raw otherwise. Computed before
        # the trust_mode branch so diagnostics exist for every mode.
        sus_raw, _ = gate_diagnostics(
            trust, self.reject_z_threshold, self.soft_reject_k,
            self.gate_signal,
            gate_rezscore=self.gate_rezscore,
            zscore_mode=self.zscore_mode,
            zscore_clip=self.zscore_clip,
        )
        sus_used = self._smooth_suspicion(client_ids, sus_raw)

        # Geometry gate, from the same gate_diagnostics() that
        # reject_soft_weighted uses (same sus_override), so the logged
        # sus_z/gate match the production aggregation exactly. sus_z is the
        # gating value (EMA-smoothed when sus_ema_beta > 0); sus_raw in stats
        # keeps this round's unsmoothed one. Computed HERE rather than after
        # aggregation because trust_mode='v6_cse_reject_geo' consumes `gate`
        # as an input; it is a pure function of `trust`/`sus_used`, so hoisting
        # it changes nothing for the other modes.
        diag_sus_z, diag_gate = gate_diagnostics(
            trust, self.reject_z_threshold, self.soft_reject_k, self.gate_signal,
            sus_override=sus_used,
        )

        # ---- 3c) Map trust signals to aggregation weights ---- #
        ds_tensor = torch.tensor(
            data_sizes, dtype=torch.float32, device=self.device
        )
        ds_total = ds_tensor.sum()
        if ds_total.item() > 0:
            alpha_cold = ds_tensor / ds_total
        else:
            alpha_cold = torch.ones(N, device=self.device) / N

        # Cold-start: graph_residual (Signal 1) works from round 0 because it
        # only needs the k-NN hypergraph of raw projected updates (no Z_hist).
        # hist_weight_beta defaults to 0.0, so hist_dev has no influence anyway.
        # We only fall back when cold_start_fallback=True is explicitly set.
        use_cold_start_fallback = (
            self.cold_start_fallback and (not has_hist)
        )
        v4_diag: "Dict[str, Any] | None" = None
        v4_cse_t: "torch.Tensor | None" = None
        if use_cold_start_fallback:
            used_alpha = alpha_cold
            used_mode = "cold_start_fedavg"
        elif self.trust_mode in (
            "v4_cse_reject", "v5_cse_reject", "v6_cse_reject_geo",
            "v7_cse_reject_corrob",
        ):
            # V4/V5/V6/V7: rejection driven by the absolute per-client full-test CSE.
            # local_cse is required — do NOT silently fall back (the defense
            # facade also validates this BEFORE its FedAvg-fallback net).
            # Deliberately routed AROUND _zscore: pool-relative scoring has no
            # absolute floor and scapegoats the most heterogeneous benign
            # client in a clean federation.
            if local_cse is None:
                raise ValueError(
                    f"trust_mode='{self.trust_mode}' requires per-client "
                    "local_cse; the server must evaluate local CSE BEFORE "
                    "aggregation (see Server._needs_local_cse)."
                )
            v4_cse_t = torch.as_tensor(
                list(local_cse), dtype=torch.float32, device=self.device
            )
            if v4_cse_t.numel() != N:
                raise ValueError(
                    f"local_cse must have length N={N}, got {v4_cse_t.numel()}"
                )
            if self.trust_mode == "v4_cse_reject":
                used_alpha, v4_diag = v4_cse_reject_weights(
                    local_cse=v4_cse_t,
                    data_sizes=ds_tensor,
                    tau_ratio=self.v4_tau_ratio,
                    k_cap=self.v4_k_cap,
                    reject_mult=self.v4_reject_mult,
                    keep_min=self.keep_min,
                )
            elif self.trust_mode == "v5_cse_reject":
                # V5: same flag decision, graded multiplier (linear ramp in
                # the CSE ratio between tau and v5_r_hard).
                used_alpha, v4_diag = v5_cse_reject_weights(
                    local_cse=v4_cse_t,
                    data_sizes=ds_tensor,
                    tau_ratio=self.v4_tau_ratio,
                    k_cap=self.v4_k_cap,
                    m_floor=self.v5_m_floor,
                    r_hard=self.v5_r_hard,
                    keep_min=self.keep_min,
                )
            elif self.trust_mode == "v6_cse_reject_geo":
                # V6: V5's ramp, tightened (never loosened) on flagged clients
                # by the geometry gate. `diag_gate` is the SAME tensor the
                # soft_reject_fedavg path gates on — sigmoid(-k*(sus - thr))
                # over the EMA-smoothed suspicion, not the raw one — so V6's
                # geometry read-out is V3's, only conjunctive.
                used_alpha, v4_diag = v6_cse_reject_geo_weights(
                    local_cse=v4_cse_t,
                    data_sizes=ds_tensor,
                    gate=diag_gate,
                    tau_ratio=self.v4_tau_ratio,
                    k_cap=self.v4_k_cap,
                    m_floor=self.v5_m_floor,
                    r_hard=self.v5_r_hard,
                    geo_floor=self.v6_geo_floor,
                    keep_min=self.keep_min,
                )
            else:
                # V7: V6 plus the iso-corroborated cold-window tier. The
                # Tier-2 conjunct is the RAW graph_residual — deliberately
                # NOT diag_gate, whose suspicion input is pool-relative
                # z-scores (C2 bans those from flag decisions; the gate keeps
                # its V6 penalty-magnitude role). When the graph channel is
                # resolution-gated this round (graph_min_distinct), Tier 2
                # abstains: quantization noise must not flag.
                used_alpha, v4_diag = v7_cse_reject_corrob_weights(
                    local_cse=v4_cse_t,
                    data_sizes=ds_tensor,
                    gate=diag_gate,
                    iso=None if trust.graph_gated else trust.graph_residual,
                    # 1-indexed, matching the archived logs' 'round' channel
                    # (server logs round_num + 1) so replay reads the window
                    # identically to the live run.
                    round_logged=int(round_num) + 1,
                    tau_ratio=self.v4_tau_ratio,
                    k_cap=self.v4_k_cap,
                    m_floor=self.v5_m_floor,
                    r_hard=self.v5_r_hard,
                    geo_floor=self.v6_geo_floor,
                    tau_lo=self.v7_tau_lo,
                    iso_min=self.v7_iso_min,
                    corrob_mult=self.v7_corrob_mult,
                    round_min=self.v7_round_min,
                    round_max=self.v7_round_max,
                    keep_min=self.keep_min,
                )
            used_mode = self.trust_mode
        elif self.trust_mode == "soft_reject_fedavg":
            # Soft sigmoid gate on the suspicion z-score selected by gate_signal,
            # then data-size FedAvg among the (continuously) trusted clients.
            # Robust to threshold miscalibration: suspicious clients are down-
            # weighted, not zeroed.
            used_alpha = reject_soft_weighted(
                trust=trust,
                data_sizes=ds_tensor,
                reject_z_threshold=self.reject_z_threshold,
                soft_reject_k=self.soft_reject_k,
                keep_min=self.keep_min,
                gate_signal=self.gate_signal,
                sus_override=sus_used,
            )
            used_mode = f"soft_reject_fedavg[{self.gate_signal}]"
        elif self.trust_mode == "reject_then_fedavg":
            # Hard binary rejection via z-score threshold, then data-size FedAvg.
            used_alpha = reject_then_weighted(
                trust=trust,
                data_sizes=ds_tensor,
                reject_z_threshold=self.reject_z_threshold,
                keep_min=self.keep_min,
                gate_signal=self.gate_signal,
                sus_override=sus_used,
            )
            used_mode = f"reject_then_fedavg[{self.gate_signal}]"
        else:
            # Pure softmax over trust logits.
            used_alpha = trust.alpha
            used_mode = "softmax"

        # ---- 4) weighted aggregation ---- #
        aggregated = weighted_aggregate(updates_stack, used_alpha)

        # ---- 4b) gate diagnostics (sus_z + gate, pre keep_min fallback) ---- #
        # (diag_sus_z / diag_gate were computed in 3b — V6 needs the gate as an
        # aggregation input, not just as a log line.)

        # ---- 5) update EMA history ---- #
        self._update_history(client_ids, Z)

        # ---- 6) stats dict ---- #
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats: Dict[str, Any] = {
            # `alpha` = weights actually used for aggregation this round.
            "alpha": used_alpha.detach().cpu().tolist(),
            # `alpha_hmp` = what HMP-GAE trust scoring would give (for
            # diagnostics even when cold-start fallback overrides it).
            "alpha_hmp": trust.alpha.detach().cpu().tolist(),
            # Kept as 'residual' (backward-compat field name in logs /
            # visualization) -- this is the graph-structural signal, the
            # primary driver of trust in V1.
            "residual": trust.graph_residual.detach().cpu().tolist(),
            "recon_residual": trust.recon_residual.detach().cpu().tolist(),
            "sem_div": trust.sem_div.detach().cpu().tolist(),
            "sem_div_z": trust.sem_div_z.detach().cpu().tolist(),
            "graph_residual_z": trust.graph_residual_z.detach().cpu().tolist(),
            "recon_residual_z": trust.recon_residual_z.detach().cpu().tolist(),
            "hist_dev_z": trust.hist_dev_z.detach().cpu().tolist(),
            # Trust-weight formula decomposition (NEW 2026-05-23, Issue 1):
            #   s       = -(w_g·z_g + alpha·z_r + w_s·z_s + beta·z_h)  (trust logit)
            #   sus_z   = combined-gate suspicion z-score (= zscore(-s) when
            #             gate_signal='combined'); the SECOND z-score that washes
            #             out absolute weight magnitudes (Bug 1 evidence).
            #   gate    = sigmoid(-k·(sus_z - threshold)), raw multiplicative
            #             weight before keep_min fallback.
            # graph_weight / residual_weight_alpha emitted so the notebook can
            # reconstruct the per-term weighted contributions w_k·z_k.
            "s": trust.s.detach().cpu().tolist(),
            "sus_z": diag_sus_z.detach().cpu().tolist(),
            "sus_raw": sus_raw.detach().cpu().tolist(),
            "gate": diag_gate.detach().cpu().tolist(),
            "graph_weight": float(self.graph_weight),
            "residual_weight_alpha": float(self.residual_weight_alpha),
            "semantic_weight": float(self.semantic_weight),
            "gate_signal": str(self.gate_signal),
            # Robust-scoring knobs actually in effect this round (for
            # post-hoc filtering of mixed-config result sets).
            "zscore_mode": str(self.zscore_mode),
            "gate_rezscore": bool(self.gate_rezscore),
            "sus_ema_beta": float(self.sus_ema_beta),
            "semantic_reference": str(self.semantic_reference),
            # Phase-gating diagnostics (NEW 2026-05-23):
            #   _configured = what main.py set (static, same every round)
            #   _effective  = what was actually applied this round (=0 once
            #                 round_num >= hist_warmup_rounds)
            # When hist_warmup_rounds is None, _effective == _configured.
            "hist_weight_beta_configured": float(self.hist_weight_beta),
            "hist_weight_beta_effective": float(hist_weight_beta_eff),
            "hist_warmup_rounds": (
                None if self.hist_warmup_rounds is None
                else int(self.hist_warmup_rounds)
            ),
            "hist_dev": trust.hist_dev.detach().cpu().tolist(),
            "has_history": bool(has_hist),
            "cold_start_fallback_used": bool(use_cold_start_fallback),
            "trust_mode_used": used_mode,
            # C1 diagnostics: whether the coarsely-quantized graph channel was
            # zeroed this round (resolution below graph_min_distinct).
            "graph_channel_gated": bool(trust.graph_gated),
            "graph_min_distinct": int(self.graph_min_distinct),
            "defense_time_ms": float(elapsed_ms),
        }
        # V4/V5/V6/V7 per-round diagnostics (trust_mode 'v4_cse_reject',
        # 'v5_cse_reject', 'v6_cse_reject_geo' or 'v7_cse_reject_corrob').
        # The "v4_" prefix names the shared CSE-reject diagnostic channel
        # family — V5/V6/V7 reuse it so the archive-side CSV tooling works
        # unchanged; version-only extras carry a "v5_" / "v6_" / "v7_"
        # prefix. NOTE: this must stay a FOUR-way branch — an `if/else`
        # collapse would silently mislabel one mode's rows as another's.
        if v4_diag is not None and v4_cse_t is not None:
            stats["v4_cse"] = v4_cse_t.detach().cpu().tolist()
            stats["v4_ratio"] = v4_diag["ratio"].detach().cpu().tolist()
            stats["v4_flagged"] = [
                int(b) for b in v4_diag["flagged"].detach().cpu().tolist()
            ]
            stats["v4_multiplier"] = v4_diag["multiplier"].detach().cpu().tolist()
            stats["v4_median_cse"] = float(v4_diag["median"])
            stats["v4_tau_ratio"] = float(self.v4_tau_ratio)
            stats["v4_k_cap"] = int(self.v4_k_cap)
            if self.trust_mode == "v4_cse_reject":
                stats["v4_reject_mult"] = float(self.v4_reject_mult)
            elif self.trust_mode == "v5_cse_reject":
                stats["v5_m_floor"] = float(self.v5_m_floor)
                stats["v5_r_hard"] = float(self.v5_r_hard)
                stats["v5_ramp_t"] = v4_diag["ramp_t"].detach().cpu().tolist()
            elif self.trust_mode == "v6_cse_reject_geo":
                # V6: V5's ramp channels (the Stage-2 knobs are literally the
                # v5_* ones) plus the geometry read-out. v6_geo_mult and
                # v6_m_cse are emitted for all N but only ACT where
                # v4_flagged is 1 — v4_multiplier is the applied value. So the
                # falsification statistic for V6 is v6_geo_mult restricted to
                # the flagged set: ≈1.0 there in every round means the geometry
                # never acted. Report that; do not tune v6_geo_floor to move it.
                stats["v5_m_floor"] = float(self.v5_m_floor)
                stats["v5_r_hard"] = float(self.v5_r_hard)
                stats["v5_ramp_t"] = v4_diag["ramp_t"].detach().cpu().tolist()
                stats["v6_geo_floor"] = float(self.v6_geo_floor)
                stats["v6_geo_gate"] = v4_diag["geo_gate"].detach().cpu().tolist()
                stats["v6_geo_mult"] = v4_diag["geo_mult"].detach().cpu().tolist()
                stats["v6_m_cse"] = v4_diag["m_cse"].detach().cpu().tolist()
            else:
                # V7: the full V6 channel family (Stage A is V6 verbatim)
                # plus the Tier-2 read-out. The falsification statistic for
                # V7 is the per-run count of v7_corrob_flagged rounds: zero
                # everywhere means the cold-window tier never acted and V7 is
                # V6 renamed — report that honestly; do not widen the window
                # or lower v7_tau_lo / v7_iso_min to make it move.
                stats["v5_m_floor"] = float(self.v5_m_floor)
                stats["v5_r_hard"] = float(self.v5_r_hard)
                stats["v5_ramp_t"] = v4_diag["ramp_t"].detach().cpu().tolist()
                stats["v6_geo_floor"] = float(self.v6_geo_floor)
                stats["v6_geo_gate"] = v4_diag["geo_gate"].detach().cpu().tolist()
                stats["v6_geo_mult"] = v4_diag["geo_mult"].detach().cpu().tolist()
                stats["v6_m_cse"] = v4_diag["m_cse"].detach().cpu().tolist()
                stats["v7_corrob_flagged"] = [
                    int(b) for b in
                    v4_diag["corrob_flagged"].detach().cpu().tolist()
                ]
                stats["v7_iso"] = v4_diag["iso"].detach().cpu().tolist()
                stats["v7_geo_resolved"] = bool(v4_diag["geo_resolved"])
                stats["v7_tau_lo"] = float(self.v7_tau_lo)
                stats["v7_iso_min"] = float(self.v7_iso_min)
                stats["v7_corrob_mult"] = float(self.v7_corrob_mult)
                stats["v7_round_min"] = int(self.v7_round_min)
                stats["v7_round_max"] = int(self.v7_round_max)
        # Probe-entropy diagnostic (V4 brief, decision (d)): the mean
        # per-sample entropy on the K-sample probe is the "free" pre-agg
        # statistic under option (ii). Logged whenever the probe exists so a
        # single run shows whether it agrees with the full-test local CSE
        # (v4_cse) at the tau threshold. Diagnostic only — never drives
        # rejection.
        if probe_arg is not None:
            Pp = probe_arg.clamp(min=1e-8)
            Pp = Pp / Pp.sum(dim=-1, keepdim=True)
            probe_cse = -(Pp * Pp.log()).sum(dim=-1).mean(dim=1)
            stats["probe_cse"] = probe_cse.detach().cpu().tolist()
        if last_loss_bundle is not None:
            stats["L_rec"] = float(last_loss_bundle.L_rec_H.item())
            stats["L_smooth"] = float(last_loss_bundle.L_smooth.item())
            stats["L_hist"] = float(last_loss_bundle.L_hist.item())
        # Keep Z around so the caller can persist it for visualization,
        # but do not let it leak into the standard JSON log (defense
        # package strips this key before logging).
        stats["Z"] = Z.detach().cpu().numpy()
        return aggregated.detach().cpu(), stats

    # --------------------------------------------------------------------- #
    # Checkpoint helpers (for resumable FL runs)                            #
    # --------------------------------------------------------------------- #
    # Serialize / restore all state that changes across rounds: the three
    # trained sub-modules (node_encoder, hmp_encoder, hyperedge_decoder),
    # the Adam optimizer, and the EMA latent cache z_hist.  The fixed random
    # projection and all hyperparameters are reconstructed deterministically
    # from config + random_proj_seed at __init__, so they are not stored.
    # The V4/V5/V6 rules ('v4_cse_reject' / 'v5_cse_reject' /
    # 'v6_cse_reject_geo') are deliberately STATELESS across rounds (per-round
    # median ratio; V5's ramp and V6's geometry factor are pure per-round
    # functions of that ratio and of the gate — whose own cross-round state,
    # the suspicion EMA, is already serialized below as `sus_ema`). They add
    # nothing here; if any of them ever grows cross-round state (e.g. sticky
    # flags), serialize it alongside sus_ema or resumed runs will silently
    # restart it.

    def state_dict(self) -> Dict[str, Any]:
        # Deep-copy the module/optimizer payloads: torch's .state_dict()
        # returns LIVE tensor references, and Optimizer.load_state_dict's
        # .to(dtype/device) is a no-op for same-dtype CPU tensors — so an
        # in-memory snapshot/restore used to end up SHARING the Adam moment
        # tensors (exp_avg / exp_avg_sq / step) with the live optimizer.
        # Two runtimes then double-updated the shared moments and a "resumed"
        # runtime silently diverged from the uninterrupted one (caught by
        # test_runtime_ema_and_state_roundtrip). The on-disk path
        # (torch.save -> torch.load in fed_resume) broke the aliasing by
        # serialization, so Colab resumes were unaffected — this makes the
        # in-memory contract match, and keeps a held snapshot immutable
        # instead of being polluted by later training steps.
        return {
            "node_encoder": copy.deepcopy(self.node_encoder.state_dict()),
            "hmp_encoder": copy.deepcopy(self.hmp_encoder.state_dict()),
            "hyperedge_decoder": copy.deepcopy(self.hyperedge_decoder.state_dict()),
            "optim": copy.deepcopy(self.optim.state_dict()),
            "z_hist": {int(k): v.detach().cpu().clone()
                       for k, v in self.z_hist.items()},
            "sus_ema": {int(k): float(v) for k, v in self.sus_ema.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.node_encoder.load_state_dict(state["node_encoder"])
        self.hmp_encoder.load_state_dict(state["hmp_encoder"])
        self.hyperedge_decoder.load_state_dict(state["hyperedge_decoder"])
        self.optim.load_state_dict(state["optim"])
        self.z_hist = {int(k): v.detach().clone()
                       for k, v in (state.get("z_hist") or {}).items()}
        # Older checkpoints predate sus_ema; default to empty (EMA restarts).
        self.sus_ema = {int(k): float(v)
                        for k, v in (state.get("sus_ema") or {}).items()}
