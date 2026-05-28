"""Krum and Multi-Krum (Blanchard et al., NeurIPS '17).

- Krum       : select the single update with the smallest Krum score
               (sum of the n-f-2 nearest non-self squared distances).
               Aggregated update = the selected client's update;
               ``alpha`` is one-hot.
- Multi-Krum : select the m updates with the smallest Krum scores and
               average them. Default m = n - f (canonical choice in the
               original paper); override via ``defense_config['m']``.

Both defenses require 2f + 2 < n. When that fails for a given round
(e.g., N drops below threshold under client dropout), they fall back to
data-size-weighted FedAvg and report it in ``stats['fallback_reason']``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from defense import Defense, FedAvgDefense
from defense._utils import pairwise_sq_l2, krum_scores


def _default_f(num_clients: int) -> int:
    """Default Byzantine count satisfying 2f + 2 < n. Returns 0 for n < 5."""
    return max(0, (num_clients - 3) // 2)


class KrumDefense(Defense):
    """Krum (NeurIPS '17): pick the single most-central update."""

    name = "krum"

    def __init__(self, num_clients: int, config: Optional[Dict[str, Any]] = None):
        self.num_clients = int(num_clients)
        cfg = dict(config or {})
        self.num_byzantine = int(
            cfg.get("num_byzantine", _default_f(self.num_clients))
        )
        self._fallback = FedAvgDefense()

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del probe_distributions
        n = len(updates)
        if n == 0:
            raise ValueError("KrumDefense.aggregate received 0 updates")

        if 2 * self.num_byzantine + 2 >= n:
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = (
                f"2f+2 >= n (f={self.num_byzantine}, n={n})"
            )
            return agg, stats

        stacked = torch.stack(updates).to(device)
        sq_d = pairwise_sq_l2(stacked)
        scores = krum_scores(sq_d, self.num_byzantine)
        selected = int(scores.argmin().item())

        alpha = [0.0] * n
        alpha[selected] = 1.0
        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": alpha,
            "krum_scores": scores.detach().cpu().tolist(),
            "selected_idx": selected,
            "selected_client_id": int(client_ids[selected]),
            "num_byzantine_assumed": self.num_byzantine,
        }
        return stacked[selected].clone(), stats

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None


class MultiKrumDefense(Defense):
    """Multi-Krum (NeurIPS '17): average the m lowest-score updates."""

    name = "multi_krum"

    def __init__(self, num_clients: int, config: Optional[Dict[str, Any]] = None):
        self.num_clients = int(num_clients)
        cfg = dict(config or {})
        self.num_byzantine = int(
            cfg.get("num_byzantine", _default_f(self.num_clients))
        )
        # Canonical Multi-Krum keeps m = n - f.
        default_m = max(1, self.num_clients - self.num_byzantine)
        self.m = int(cfg.get("m", default_m))
        self._fallback = FedAvgDefense()

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del probe_distributions
        n = len(updates)
        if n == 0:
            raise ValueError("MultiKrumDefense.aggregate received 0 updates")

        if 2 * self.num_byzantine + 2 >= n:
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = (
                f"2f+2 >= n (f={self.num_byzantine}, n={n})"
            )
            return agg, stats

        stacked = torch.stack(updates).to(device)
        sq_d = pairwise_sq_l2(stacked)
        scores = krum_scores(sq_d, self.num_byzantine)

        m_eff = max(1, min(self.m, n))
        selected_idx = torch.topk(scores, m_eff, largest=False).indices.tolist()
        alpha = [0.0] * n
        for i in selected_idx:
            alpha[i] = 1.0 / m_eff
        agg = stacked[selected_idx].mean(dim=0)

        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": alpha,
            "krum_scores": scores.detach().cpu().tolist(),
            "selected_idxs": list(selected_idx),
            "selected_client_ids": [int(client_ids[i]) for i in selected_idx],
            "m": m_eff,
            "num_byzantine_assumed": self.num_byzantine,
        }
        return agg, stats

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None
