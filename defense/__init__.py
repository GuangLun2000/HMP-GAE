# defense/__init__.py
# Pluggable defense strategies for federated aggregation.
#
# Provides a unified interface so that Server.aggregate_updates can swap between
# standard FedAvg and robust/immunization methods (e.g., HMP-GAE) purely via
# config, without changing the FL orchestration.
#
# Exports:
#   - Defense           : abstract base class
#   - FedAvgDefense     : faithful migration of the original FedAvg logic
#   - HMPGAEDefense     : hypergraph message-passing GAE immunization (this paper)
#   - KrumDefense       : Krum                  (Blanchard et al., NeurIPS '17)
#   - MultiKrumDefense  : Multi-Krum            (Blanchard et al., NeurIPS '17)
#   - CoordMedianDefense: Coord-wise Median     (Yin et al., ICML '18)
#   - FLTrustDefense    : FLTrust (medroot var.)(Cao et al., NDSS '21)
#   - FoolsGoldDefense  : FoolsGold             (Fung et al., RAID '20)
#
# Each baseline lives in its own peer module (defense/krum.py, etc.) and is
# re-exported here. Wire new methods into ``build_defense`` below.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import torch


class Defense(ABC):
    """
    Strategy interface for server-side aggregation.

    Subclasses implement `aggregate` which returns the aggregated update
    (before server_lr scaling) along with a stats dict used for logging
    and visualization.
    """

    name: str = "abstract"

    @abstractmethod
    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the aggregated update Delta_global from per-client updates.

        Args:
            updates: List of N flat-parameter update tensors (all same shape).
            client_ids: List of N client identifiers (parallel to updates).
            data_sizes: List of N raw aggregation weights (before normalization).
                        Defenses may ignore this (e.g. HMP-GAE uses trust scores).
            round_num: 0-indexed round counter; lets defenses track history.
            device: Target torch device for aggregation.
            probe_distributions: Optional (N, K, C) tensor of per-client softmax
                outputs on a fixed K-sample probe subset. Used by HMP-GAE as a
                semantic-divergence trust signal; ignored by FedAvg-style
                defenses. None when the server has not provided one.

        Returns:
            aggregated_update: 1-D tensor, same shape as each element of `updates`.
            stats: dict with at minimum key 'alpha' (list of length N, floats
                   summing to ~1.0) and 'defense_name'. May include extra fields
                   like 'residual', 'hist_dev', 'L_rec', 'Z' for HMP-GAE.
        """


class FedAvgDefense(Defense):
    """
    Standard FedAvg weighted aggregation (data-size-weighted).

    This is a faithful re-implementation of the original Server.aggregate_updates
    weighting logic, preserved bit-for-bit so that the `defense_method='fedavg'`
    path produces identical results to the pre-plugin codebase.
    """

    name = "fedavg"

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # FedAvg ignores semantic probe signals.
        del probe_distributions
        if len(updates) == 0:
            raise ValueError("FedAvgDefense.aggregate received 0 updates")

        dtype = updates[0].dtype
        stacked = torch.stack(updates).to(device)
        weight_tensor = torch.tensor(data_sizes, device=device, dtype=dtype)
        total = weight_tensor.sum()
        if total.item() <= 0:
            weight_tensor = torch.ones_like(weight_tensor) / len(data_sizes)
        else:
            weight_tensor = weight_tensor / total
        aggregated_update = (stacked * weight_tensor.view(-1, 1)).sum(dim=0)
        del stacked

        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": weight_tensor.detach().cpu().tolist(),
            "raw_weights": list(map(float, data_sizes)),
        }
        return aggregated_update, stats

    # FedAvg has no cross-round state; checkpoint hooks are no-ops so the
    # resume layer can call them uniformly across defenses.
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None


# --------------------------------------------------------------------------- #
# HMP-GAE defense                                                             #
# --------------------------------------------------------------------------- #
# HMPGAEDefense is implemented as a thin facade over the `hmp_gae` sub-package
# (node features, hypergraph construction, L-layer HMP encoder, GAE decoder,
# losses, trust scoring). For V1 simplicity we keep the whole pipeline on CPU
# since N (number of clients) is small and the latent dims are modest.


class HMPGAEDefense(Defense):
    """
    Hypergraph Message-Passing Graph AutoEncoder immunization (this paper).

    Per round:
      1. Extract node features eta_i from each client's flat update (+ history).
      2. Build a k-NN hypergraph H over eta (M = N, one hyperedge per node).
      3. Run an L-layer HMP encoder (node -> hyperedge -> node) to obtain Z.
      4. Decode pairwise adjacency A_hat and hyperedge incidence H_hat.
      5. Self-supervised training for a handful of Adam steps:
             L = lambda_H * BCE(H, H_hat_logits)
               + lambda_A * smoothness(Z, A_hat)
               + lambda_hist * || Z - Z_hist ||^2
      6. Closed-form trust score s_i from graph residual + historical deviation.
      7. alpha_i = softmax(s_i / tau), aggregate Delta = sum alpha_i * Delta_i.
      8. Update EMA historical embedding cache z_hist.

    Degenerate cases (N <= 2 or numerical issue) fall back to FedAvg weights.
    """

    name = "hmp_gae"

    def __init__(
        self,
        num_clients: int,
        config: Optional[Dict[str, Any]] = None,
        flat_update_dim: Optional[int] = None,
    ):
        self.num_clients = int(num_clients)
        self.cfg: Dict[str, Any] = dict(config or {})
        self.flat_update_dim = flat_update_dim
        self._initialized = False
        self._hmp_runtime = None
        self._fallback = FedAvgDefense()
        # Optional cached state from a checkpoint load that happened before
        # the first aggregate() call (runtime not yet constructed).  Applied
        # by _lazy_init once the runtime exists.
        self._pending_state: Optional[Dict[str, Any]] = None

    def _lazy_init(self, flat_update_dim: int, device: torch.device) -> None:
        # Import lazily so that importing this package stays cheap when only
        # FedAvgDefense is used (e.g. baselines).
        from hmp_gae.runtime import HMPGAERuntime

        self._hmp_runtime = HMPGAERuntime(
            num_clients=self.num_clients,
            flat_update_dim=flat_update_dim,
            config=self.cfg,
            device=device,
        )
        self._initialized = True
        if self._pending_state is not None:
            self._hmp_runtime.load_state_dict(self._pending_state)
            self._pending_state = None

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if len(updates) == 0:
            raise ValueError("HMPGAEDefense.aggregate received 0 updates")

        # Fallback for degenerate N — HMP message passing is ill-defined with
        # fewer than 3 nodes and offers no benefit.
        if len(updates) <= 2:
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = f"N={len(updates)} <= 2"
            return agg, stats

        if not self._initialized:
            self._lazy_init(int(updates[0].numel()), torch.device("cpu"))

        try:
            agg_cpu, stats = self._hmp_runtime.aggregate(
                updates=updates,
                client_ids=client_ids,
                data_sizes=data_sizes,
                round_num=round_num,
                probe_distributions=probe_distributions,
            )
        except Exception as e:  # noqa: BLE001 - runtime safety net
            # Numerical / shape issues: fall back silently to FedAvg so the FL
            # run does not crash. The failure is reported in stats.
            print(
                f"  [HMP-GAE] runtime error at round {round_num}: {type(e).__name__}: {e}. "
                "Falling back to FedAvg for this round."
            )
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = f"{type(e).__name__}: {e}"
            return agg, stats

        # Move aggregated update to the server's device for downstream use.
        agg = agg_cpu.to(device=device, dtype=updates[0].dtype)
        stats["defense_name"] = self.name
        return agg, stats

    def state_dict(self) -> Dict[str, Any]:
        # Pre-aggregation, the runtime hasn't been built yet — nothing to save.
        if not self._initialized or self._hmp_runtime is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "runtime": self._hmp_runtime.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state or not state.get("initialized"):
            return
        # We need flat_update_dim to build the runtime.  Defer the actual
        # parameter load until the runtime exists; in practice load is called
        # right after construction and before the first aggregate(), so the
        # flat dim isn't known yet.  Cache the payload and let _lazy_init
        # consume it on the first aggregate() call.
        self._pending_state = state["runtime"]
        if self._initialized and self._hmp_runtime is not None:
            self._hmp_runtime.load_state_dict(self._pending_state)
            self._pending_state = None


def build_defense(
    method: str,
    num_clients: int,
    defense_config: Optional[Dict[str, Any]] = None,
    flat_update_dim: Optional[int] = None,
) -> Defense:
    """
    Factory: instantiate a Defense from a config-facing method string.

    Supported methods (case-insensitive, hyphens/underscores normalized):
      - 'fedavg' / 'none'                : FedAvg (data-size weighted)
      - 'hmp_gae'                        : HMP-GAE (this paper)
      - 'krum'                           : Krum                (NeurIPS '17)
      - 'multi_krum' / 'multikrum'       : Multi-Krum          (NeurIPS '17)
      - 'coord_median' / 'median'        : Coord-wise Median   (ICML '18)
      - 'fltrust'                        : FLTrust medroot var.(NDSS '21)
      - 'foolsgold'                      : FoolsGold           (RAID '20)
    """
    m = (method or "fedavg").strip().lower()
    if m in {"fedavg", "fed_avg", "none", ""}:
        return FedAvgDefense()
    if m in {"hmp_gae", "hmpgae", "hmp-gae"}:
        return HMPGAEDefense(
            num_clients=num_clients,
            config=defense_config or {},
            flat_update_dim=flat_update_dim,
        )
    # Lazy imports keep the fedavg / hmp_gae paths zero-cost and avoid any
    # partial-init issues at package load time.
    if m in {"krum"}:
        from defense.krum import KrumDefense
        return KrumDefense(num_clients=num_clients, config=defense_config)
    if m in {"multi_krum", "multikrum", "multi-krum"}:
        from defense.krum import MultiKrumDefense
        return MultiKrumDefense(num_clients=num_clients, config=defense_config)
    if m in {"coord_median", "median", "coordmedian", "coord-median"}:
        from defense.median import CoordMedianDefense
        return CoordMedianDefense(num_clients=num_clients, config=defense_config)
    if m in {"fltrust", "fl_trust", "fl-trust"}:
        from defense.fltrust import FLTrustDefense
        return FLTrustDefense(num_clients=num_clients, config=defense_config)
    if m in {"foolsgold", "fools_gold", "fools-gold"}:
        from defense.foolsgold import FoolsGoldDefense
        return FoolsGoldDefense(num_clients=num_clients, config=defense_config)
    raise ValueError(
        f"Unknown defense_method={method!r}. Supported: "
        f"'fedavg', 'hmp_gae', 'krum', 'multi_krum', 'coord_median', 'fltrust', 'foolsgold'."
    )


# Re-export baseline classes at the package level so callers can do
# `from defense import KrumDefense` (mirrors how FedAvgDefense / HMPGAEDefense
# are surfaced here). Lazy attribute access avoids the modest import cost when
# only the factory is used.
def __getattr__(name: str):
    if name in {"KrumDefense", "MultiKrumDefense"}:
        from defense.krum import KrumDefense, MultiKrumDefense
        return {"KrumDefense": KrumDefense, "MultiKrumDefense": MultiKrumDefense}[name]
    if name == "CoordMedianDefense":
        from defense.median import CoordMedianDefense
        return CoordMedianDefense
    if name == "FLTrustDefense":
        from defense.fltrust import FLTrustDefense
        return FLTrustDefense
    if name == "FoolsGoldDefense":
        from defense.foolsgold import FoolsGoldDefense
        return FoolsGoldDefense
    raise AttributeError(f"module 'defense' has no attribute {name!r}")
