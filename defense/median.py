"""Coordinate-wise median aggregation (Yin et al., ICML '18).

Each output coordinate is independently the median of that coordinate
across all N client updates. Hyperparameter-free.

Because the median is taken per-coordinate, no single client has a
well-defined contribution weight; we report uniform alpha = 1/N for
diagnostic interface consistency with FedAvg/HMP-GAE/etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from defense import Defense


class CoordMedianDefense(Defense):
    """Per-coordinate median (Yin et al., ICML '18)."""

    name = "coord_median"

    def __init__(self, num_clients: int, config: Optional[Dict[str, Any]] = None):
        self.num_clients = int(num_clients)
        del config  # no hyperparameters

    def aggregate(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: List[float],
        round_num: int,
        device: torch.device,
        probe_distributions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del probe_distributions, data_sizes
        n = len(updates)
        if n == 0:
            raise ValueError("CoordMedianDefense.aggregate received 0 updates")
        stacked = torch.stack(updates).to(device)
        med = stacked.median(dim=0).values
        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": [1.0 / n] * n,
            "note": "coord-wise median; alpha reported uniformly for interface consistency",
        }
        return med, stats

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None
