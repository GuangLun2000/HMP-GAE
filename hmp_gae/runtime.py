# hmp_gae/runtime.py
# HMPGAERuntime: the stateful engine that performs one round of trust-scored
# weighted aggregation.
#
# Called from defense.HMPGAEDefense. Three trust modes survive the 2026-08-11
# pruning (V6/V7 and the V1-V3 geometry stack live only in git history and
# docs/DECISION.md):
#
#   'v4_cse_reject' (V4, 2026-07-28): rejection driven by the ABSOLUTE
#     per-client full-test CSE, pool-median normalised and rank-capped
#     (trust_scorer.v4_cse_reject_weights); flagged clients take the constant
#     v4_reject_mult. Requires the server to evaluate local CSE BEFORE
#     aggregation and pass it via aggregate(local_cse=...). Pure decision
#     rule — no GAE, no learned state. Kept as V8's detect-then-suppress
#     ablation arm.
#
#   'v5_cse_reject' (V5, 2026-08-06): V4's flag decision byte-identical, but
#     the flagged-client multiplier is a linear ramp in the CSE ratio
#     (trust_scorer.v5_cse_reject_weights): mild just past tau, v5_m_floor at
#     ratio >= v5_r_hard. Same local-CSE requirement as V4; also stateless.
#     V5 is V8's Stage A and its matched-run safety baseline — a
#     hypergraph-attributable claim requires matched V5/V8 runs.
#
#   'v8_hmp_cse_propagation' (V8, 2026-08-09, current mechanism): V5 remains
#     the high-confidence CSE decision layer.  Its flags seed risk on a fixed
#     dual-view hypergraph: raw-update and label-free probe behavior must
#     agree on mutual neighbors.  The HMP-GAE (the only learned component in
#     the runtime) trains on the fixed update topology and its signed
#     affinity attenuates node-edge-node propagation.  Only directionally
#     elevated-CSE peers can consume unused rank-cap budget; no seed /
#     reliable edge / budget returns V5 exactly.
#
# Only V8 keeps cross-round state: the three GAE sub-modules, their Adam
# optimizer, and the EMA cache z_hist of previous-round embeddings (detached).
# The whole runtime defaults to CPU because N is small and running on CPU
# avoids frequent host<->device transfers of the aggregated update.

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .node_features import (
    FixedRandomProjection,
    NodeFeatureEncoder,
    compute_node_features,
)
from .hypergraph import (
    knn_hypergraph,
    semantic_js_similarity,
    knn_hypergraph_from_similarity,
    mutual_neighbor_adjacency,
    consensus_propagation_hypergraph,
    hypergraph_propagation_matrix,
)
from .encoder import HMPEncoder
from .decoder import normalized_cosine_decoder, HyperedgeDecoder
from .losses import total_loss_v8, per_node_adjacency_recon_error
from .trust_scorer import (
    weighted_aggregate,
    v4_cse_reject_weights,
    v5_cse_reject_weights,
    v8_hmp_cse_propagation_weights,
)

_KNOWN_MODES = ("v4_cse_reject", "v5_cse_reject", "v8_hmp_cse_propagation")


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

        # trust_mode is REQUIRED: a missing key silently running some default
        # aggregation rule for 50 rounds is exactly the failure mode the
        # loud-crash contract exists to prevent.
        raw_mode = self.cfg.get("trust_mode")
        if raw_mode is None:
            raise ValueError(
                f"defense_config.trust_mode is required; expected one of "
                f"{sorted(_KNOWN_MODES)}"
            )
        self.trust_mode = str(raw_mode).strip().lower()
        if self.trust_mode not in _KNOWN_MODES:
            raise ValueError(
                f"Unknown trust_mode={self.trust_mode!r}; expected one of "
                f"{sorted(_KNOWN_MODES)}. V6/V7 and the pre-V4 geometry modes "
                "were removed 2026-08-11 (docs/DECISION.md); reproduce them "
                "from git history."
            )
        self.is_v8 = self.trust_mode == "v8_hmp_cse_propagation"

        # semantic_weight > 0 is the server-side switch for the per-client
        # probe forward. V8 hard-requires it: the probe distributions are its
        # independent behavior view.
        self.semantic_weight = float(self.cfg.get("semantic_weight", 0.0))
        if self.is_v8 and self.semantic_weight <= 0.0:
            raise ValueError(
                "trust_mode='v8_hmp_cse_propagation' requires "
                "semantic_weight > 0 so the independent probe-behavior "
                "hypergraph is available"
            )

        # --- Shared CSE-decision knobs (all three modes) ---
        #   v4_tau_ratio  : pre-registered 1.85 (zero-FP plateau [1.785, 1.90]
        #                   over 51 archived runs); do NOT re-tune post hoc.
        #   k_cap         : REUSES defense_config.num_byzantine — no new
        #                   hyperparameter. Rule is sound only for
        #                   #attackers <= k_cap < N/2 (validated below).
        self.v4_tau_ratio = float(self.cfg.get("v4_tau_ratio", 1.85))
        self.v4_k_cap = int(self.cfg.get("num_byzantine", 2))
        self.keep_min = int(self.cfg.get("keep_min", 1))
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

        # --- V4 knob (inert unless trust_mode == 'v4_cse_reject') ---
        #   v4_reject_mult: rejection multiplier, default 0.10 (soft). 0.0 =
        #                   HARD REMOVAL — a flagged client contributes nothing
        #                   to that round's aggregate. Legal since 2026-08-07 as
        #                   an explicit pre-registered ablation arm (docs/
        #                   DECISION.md "V4-remove"); as a DEFAULT it stays
        #                   rejected (FoolsGold's mechanism, the archive's
        #                   worst Qwen-Yahoo PPL). Sweep set {0.10, 0.05,
        #                   0.02, 0.0}.
        self.v4_reject_mult = float(self.cfg.get("v4_reject_mult", 0.10))
        if self.trust_mode == "v4_cse_reject":
            # The bound is closed at 0: negatives would flip a flagged
            # client's update sign, which is an attack, not a penalty.
            if not (0.0 <= self.v4_reject_mult < 1.0):
                raise ValueError(
                    "v4_reject_mult must be in [0, 1) — 0.0 = hard removal "
                    "(pre-registered ablation arm, DECISION 2026-08-07); "
                    f"got {self.v4_reject_mult}"
                )

        # --- V5 knobs (V5 mode, and V8's Stage A) ---
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
        if self.trust_mode in ("v5_cse_reject", "v8_hmp_cse_propagation"):
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

        # --- V8 GAE hyperparameters (constructed only under V8) ---
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
        self.hist_ema_beta = float(self.cfg.get("hist_ema_beta", 0.9))
        self.proj_seed = int(self.cfg.get("random_proj_seed", 42))

        # ---- Modules + state (V8 only; V4/V5 are stateless rules) ---- #
        # z_hist is a per-client EMA buffer of the latent embedding. It feeds
        # the node-feature history input and L_hist.
        self.z_hist: Dict[int, torch.Tensor] = {}
        self.projection: Optional[FixedRandomProjection] = None
        self.node_encoder: Optional[NodeFeatureEncoder] = None
        self.hmp_encoder: Optional[HMPEncoder] = None
        self.hyperedge_decoder: Optional[HyperedgeDecoder] = None
        self.optim: Optional[torch.optim.Adam] = None
        if self.is_v8:
            self.projection = FixedRandomProjection(
                d_in=self.flat_update_dim, d_out=self.proj_dim,
                seed=self.proj_seed,
            )
            self.node_encoder = NodeFeatureEncoder(
                proj_dim=self.proj_dim, hist_dim=self.latent_dim,
                eta_dim=self.eta_dim,
            ).to(self.device)
            self.hmp_encoder = HMPEncoder(
                eta_dim=self.eta_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_hmp_layers,
                residual=True,
                signed_output=True,
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
                params, lr=self.train_lr, weight_decay=0.0  # L2 in the loss
            )

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
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

        updates_stack = self._stack_updates(updates)   # (N, d_update)

        # Probe distributions (N, K, C): V8's independent behavior view. For
        # V4/V5 they only feed the probe_cse diagnostic below.
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

        # All three modes reject on the absolute per-client full-test CSE.
        # local_cse is required — do NOT silently fall back (the defense
        # facade also validates this BEFORE its FedAvg-fallback net).
        # Deliberately an absolute statistic: pool-relative scoring has no
        # absolute floor and scapegoats the most heterogeneous benign client
        # in a clean federation.
        if local_cse is None:
            raise ValueError(
                f"trust_mode='{self.trust_mode}' requires per-client "
                "local_cse; the server must evaluate local CSE BEFORE "
                "aggregation (see Server._needs_local_cse)."
            )
        cse_t = torch.as_tensor(
            list(local_cse), dtype=torch.float32, device=self.device
        )
        if cse_t.numel() != N:
            raise ValueError(
                f"local_cse must have length N={N}, got {cse_t.numel()}"
            )

        ds_tensor = torch.tensor(
            data_sizes, dtype=torch.float32, device=self.device
        )

        # ---- V8: fixed dual-view topology + GAE training + propagation ---- #
        # The update view uses the fixed JL projection (stable basis, no
        # optimizer feedback); the behavior view uses label-free per-sample
        # probe distributions.  Both graphs are built ONCE per round and
        # detached; the learned GAE only denoises the propagation operator.
        v8_update_mutual = None
        v8_behavior_mutual = None
        v8_consensus_mutual = None
        v8_propagation = None
        v8_recon_error = None
        last_loss_bundle = None
        Z = None
        if self.is_v8:
            if probe_arg is None:
                raise ValueError(
                    "trust_mode='v8_hmp_cse_propagation' requires "
                    "probe_distributions; the server must collect the shared "
                    "label-free semantic probe before aggregation"
                )
            hist_mat, has_hist = self._history_matrix(client_ids)
            Z_hist_arg = hist_mat if has_hist else None
            with torch.no_grad():
                stable_update_view = self.projection(updates_stack).detach()
                fixed_H, fixed_D_V_inv, fixed_D_E_inv = knn_hypergraph(
                    stable_update_view, k=self.knn_k
                )
                behavior_sim = semantic_js_similarity(probe_arg)
                behavior_H, _, _ = knn_hypergraph_from_similarity(
                    behavior_sim, k=self.knn_k
                )
                v8_update_mutual = mutual_neighbor_adjacency(fixed_H)
                v8_behavior_mutual = mutual_neighbor_adjacency(behavior_H)
                (
                    v8_propagation_H,
                    _,
                    _,
                    v8_consensus_mutual,
                ) = consensus_propagation_hypergraph(fixed_H, behavior_H)
                v8_adjacency_target = v8_update_mutual.to(
                    dtype=updates_stack.dtype
                )

            # Self-supervised training steps on the FIXED topology.
            self.node_encoder.train()
            self.hmp_encoder.train()
            self.hyperedge_decoder.train()
            loss_params = (
                list(self.node_encoder.parameters())
                + list(self.hmp_encoder.parameters())
                + list(self.hyperedge_decoder.parameters())
            )
            for _step in range(self.train_steps_per_round):
                self.optim.zero_grad(set_to_none=True)
                eta = compute_node_features(
                    updates=updates_stack,
                    projection=self.projection,
                    encoder=self.node_encoder,
                    history=hist_mat if has_hist else None,
                )
                Z = self.hmp_encoder(eta, fixed_H, fixed_D_V_inv, fixed_D_E_inv)
                A_logits, A_probs = normalized_cosine_decoder(Z)
                H_hat_logits, _ = self.hyperedge_decoder(Z)
                bundle = total_loss_v8(
                    H=fixed_H,
                    H_hat_logits=H_hat_logits,
                    adjacency_target=v8_adjacency_target,
                    adjacency_logits=A_logits,
                    Z=Z,
                    Z_hist=Z_hist_arg,
                    lambda_H=self.lambda_H,
                    lambda_A=self.lambda_A,
                    lambda_hist=self.lambda_hist,
                    weight_decay=self.weight_decay,
                    params=loss_params,
                )
                bundle.total.backward()
                # Mild gradient clipping for stability when N is small.
                torch.nn.utils.clip_grad_norm_(loss_params, max_norm=5.0)
                self.optim.step()
                last_loss_bundle = bundle

            # Eval-mode forward for the propagation operator.
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
                Z = self.hmp_encoder(eta, fixed_H, fixed_D_V_inv, fixed_D_E_inv)
                A_logits, A_probs = normalized_cosine_decoder(Z)
                v8_recon_error = per_node_adjacency_recon_error(
                    v8_adjacency_target, A_logits
                )
                v8_propagation = hypergraph_propagation_matrix(
                    v8_propagation_H, pair_affinity=A_probs
                )

        # ---- CSE decision -> aggregation weights ---- #
        if self.trust_mode == "v4_cse_reject":
            used_alpha, diag = v4_cse_reject_weights(
                local_cse=cse_t,
                data_sizes=ds_tensor,
                tau_ratio=self.v4_tau_ratio,
                k_cap=self.v4_k_cap,
                reject_mult=self.v4_reject_mult,
                keep_min=self.keep_min,
            )
        elif self.trust_mode == "v5_cse_reject":
            used_alpha, diag = v5_cse_reject_weights(
                local_cse=cse_t,
                data_sizes=ds_tensor,
                tau_ratio=self.v4_tau_ratio,
                k_cap=self.v4_k_cap,
                m_floor=self.v5_m_floor,
                r_hard=self.v5_r_hard,
                keep_min=self.keep_min,
            )
        else:
            # V8: V5 CSE flags are immutable high-confidence seeds.  The
            # fixed cross-view hypergraph may use only the unused rank-cap
            # budget to softly penalize a connected, elevated-CSE peer.
            used_alpha, diag = v8_hmp_cse_propagation_weights(
                local_cse=cse_t,
                data_sizes=ds_tensor,
                propagation_matrix=v8_propagation,
                tau_ratio=self.v4_tau_ratio,
                k_cap=self.v4_k_cap,
                m_floor=self.v5_m_floor,
                r_hard=self.v5_r_hard,
                keep_min=self.keep_min,
            )

        # ---- weighted aggregation ---- #
        aggregated = weighted_aggregate(updates_stack, used_alpha)

        # ---- update EMA history (V8's cross-round state) ---- #
        if self.is_v8:
            self._update_history(client_ids, Z)

        # ---- stats dict ---- #
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats: Dict[str, Any] = {
            # `alpha` = weights actually used for aggregation this round.
            "alpha": used_alpha.detach().cpu().tolist(),
            "trust_mode_used": self.trust_mode,
            "defense_time_ms": float(elapsed_ms),
        }
        # V4/V5/V8 per-round diagnostics.
        # The "v4_" prefix names the shared CSE-reject diagnostic channel
        # family — V5/V8 reuse it so archive-side CSV tooling works unchanged;
        # version-only extras carry their own prefix. NOTE: this must stay an
        # explicit mode branch — an early `else` collapse would silently
        # mislabel one mode's rows as another's.
        stats["v4_cse"] = cse_t.detach().cpu().tolist()
        stats["v4_ratio"] = diag["ratio"].detach().cpu().tolist()
        stats["v4_flagged"] = [
            int(b) for b in diag["flagged"].detach().cpu().tolist()
        ]
        stats["v4_multiplier"] = diag["multiplier"].detach().cpu().tolist()
        stats["v4_median_cse"] = float(diag["median"])
        stats["v4_tau_ratio"] = float(self.v4_tau_ratio)
        stats["v4_k_cap"] = int(self.v4_k_cap)
        if self.trust_mode == "v4_cse_reject":
            stats["v4_reject_mult"] = float(self.v4_reject_mult)
        elif self.trust_mode == "v5_cse_reject":
            stats["v5_m_floor"] = float(self.v5_m_floor)
            stats["v5_r_hard"] = float(self.v5_r_hard)
            stats["v5_ramp_t"] = diag["ramp_t"].detach().cpu().tolist()
        else:
            # V8 Stage A is V5. Emit every input and intermediate needed to
            # falsify the mechanism from an archived result: no seeds, no
            # seed-connected consensus path, no elevated-CSE peer, or no
            # propagated flags each explain exact V5 behavior.
            stats["v5_m_floor"] = float(self.v5_m_floor)
            stats["v5_r_hard"] = float(self.v5_r_hard)
            stats["v5_ramp_t"] = diag["ramp_t"].detach().cpu().tolist()
            stats["v8_seed"] = diag["seed"].detach().cpu().tolist()
            stats["v8_propagated_risk"] = (
                diag["propagated_risk"].detach().cpu().tolist()
            )
            stats["v8_cse_evidence"] = (
                diag["cse_evidence"].detach().cpu().tolist()
            )
            stats["v8_joint_evidence"] = (
                diag["joint_evidence"].detach().cpu().tolist()
            )
            stats["v8_propagated_flagged"] = [
                int(b) for b in
                diag["propagated_flagged"].detach().cpu().tolist()
            ]
            stats["v8_propagated_multiplier"] = (
                diag["propagated_multiplier"].detach().cpu().tolist()
            )
            stats["v8_propagation_matrix"] = (
                v8_propagation.detach().cpu().tolist()
            )
            stats["v8_update_mutual"] = (
                v8_update_mutual.to(dtype=torch.int8).cpu().tolist()
            )
            stats["v8_behavior_mutual"] = (
                v8_behavior_mutual.to(dtype=torch.int8).cpu().tolist()
            )
            stats["v8_consensus_mutual"] = (
                v8_consensus_mutual.to(dtype=torch.int8).cpu().tolist()
            )
            stats["v8_update_edge_count"] = int(
                v8_update_mutual.sum().item() // 2
            )
            stats["v8_behavior_edge_count"] = int(
                v8_behavior_mutual.sum().item() // 2
            )
            stats["v8_consensus_edge_count"] = int(
                v8_consensus_mutual.sum().item() // 2
            )
            stats["v8_recon_error"] = (
                v8_recon_error.detach().cpu().tolist()
            )
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
            stats["L_struct"] = float(last_loss_bundle.L_smooth.item())
            stats["L_hist"] = float(last_loss_bundle.L_hist.item())
        if Z is not None:
            # Keep Z around so the caller can persist it for visualization,
            # but do not let it leak into the standard JSON log (defense
            # package strips this key before logging).
            stats["Z"] = Z.detach().cpu().numpy()
        return aggregated.detach().cpu(), stats

    # --------------------------------------------------------------------- #
    # Checkpoint helpers (for resumable FL runs)                            #
    # --------------------------------------------------------------------- #
    # Serialize / restore all state that changes across rounds — which since
    # the 2026-08-11 pruning exists only under V8: the three trained
    # sub-modules, the Adam optimizer, and the EMA latent cache z_hist. The
    # fixed random projection and all hyperparameters are reconstructed
    # deterministically from config + random_proj_seed at __init__, so they
    # are not stored. V4/V5 decisions are stateless — their state_dict is
    # empty. If any rule ever grows cross-round state (e.g. sticky flags),
    # serialize it here or resumed runs will silently restart it.

    def state_dict(self) -> Dict[str, Any]:
        if not self.is_v8:
            return {}
        # Deep-copy the module/optimizer payloads: torch's .state_dict()
        # returns LIVE tensor references, and Optimizer.load_state_dict's
        # .to(dtype/device) is a no-op for same-dtype CPU tensors — so an
        # in-memory snapshot/restore used to end up SHARING the Adam moment
        # tensors (exp_avg / exp_avg_sq / step) with the live optimizer.
        # Two runtimes then double-updated the shared moments and a "resumed"
        # runtime silently diverged from the uninterrupted one (caught by
        # test_runtime_v8_state_roundtrip). The on-disk path
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
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not self.is_v8 or not state:
            # V4/V5 are stateless; also tolerate legacy checkpoints written
            # by removed modes (their GAE state is meaningless here).
            return
        self.node_encoder.load_state_dict(state["node_encoder"])
        self.hmp_encoder.load_state_dict(state["hmp_encoder"])
        self.hyperedge_decoder.load_state_dict(state["hyperedge_decoder"])
        self.optim.load_state_dict(state["optim"])
        self.z_hist = {int(k): v.detach().clone()
                       for k, v in (state.get("z_hist") or {}).items()}
