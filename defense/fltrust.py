"""FLTrust (Cao et al., NDSS '21).

Server-side trust bootstrapping via cosine similarity to a reference
("anchor") update. Each client's contribution is:
    1. cosine similarity to the anchor  ->  raw trust score
    2. ReLU clipping (clients pointing away from the anchor are dropped)
    3. normalize trust scores to sum to 1  ->  alpha
    4. magnitude normalization: rescale each client's update to the
       anchor's norm before weighting (removes magnitude-based attacks).

Anchor source:
- Canonical FLTrust trains the SERVER's own copy one step on a small
  clean root dataset to obtain the anchor. The current Defense interface
  does not expose the global model or a server-side training step.
- We therefore default to ``anchor='median'``: the coordinate-wise median
  of client updates. This is a documented FLTrust variant for when
  server-side root data is unavailable; it preserves the cosine + ReLU +
  magnitude-norm pipeline and only substitutes the source of the
  reference direction.
- An ``anchor='external'`` mode is exposed via ``set_root_update()`` so
  canonical FLTrust can be wired in later by a small server-side change
  (run one SGD step on a held-out clean set, inject the result).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from defense import Defense, FedAvgDefense


class FLTrustDefense(Defense):
    """FLTrust (NDSS '21)."""

    name = "fltrust"

    def __init__(self, num_clients: int, config: Optional[Dict[str, Any]] = None):
        self.num_clients = int(num_clients)
        cfg = dict(config or {})
        self.anchor = str(cfg.get("anchor", "median")).lower()
        if self.anchor not in {"median", "mean", "external"}:
            raise ValueError(
                f"FLTrust anchor must be one of 'median', 'mean', 'external'; got {self.anchor!r}"
            )
        self._root_update: Optional[torch.Tensor] = None
        self._fallback = FedAvgDefense()

    def set_root_update(self, root_update: torch.Tensor) -> None:
        """Inject a server-trained anchor update (for ``anchor='external'``).

        Should be called by the server each round, before ``aggregate``.
        """
        self._root_update = root_update.detach().clone()

    def _compute_anchor(self, stacked: torch.Tensor) -> torch.Tensor:
        if self.anchor == "external":
            if self._root_update is None:
                raise RuntimeError(
                    "FLTrust anchor='external' but no root_update injected. "
                    "Call set_root_update() before aggregate()."
                )
            return self._root_update.to(stacked.device, dtype=stacked.dtype)
        if self.anchor == "mean":
            return stacked.mean(dim=0)
        return stacked.median(dim=0).values

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
            raise ValueError("FLTrustDefense.aggregate received 0 updates")
        if n < 2:
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = "n < 2"
            return agg, stats

        stacked = torch.stack(updates).to(device)
        anchor = self._compute_anchor(stacked)
        anchor_norm = torch.norm(anchor).clamp_min(1e-12)

        cos = F.cosine_similarity(
            stacked, anchor.unsqueeze(0).expand_as(stacked), dim=1
        )
        ts = torch.clamp(cos, min=0.0)
        ts_sum = ts.sum()
        if ts_sum.item() <= 1e-9:
            # Pathological round: no client aligned with anchor. Uniform fallback
            # so the FL run does not stall.
            alpha_t = torch.ones_like(ts) / n
            degenerate = True
        else:
            alpha_t = ts / ts_sum
            degenerate = False

        # Magnitude normalization: rescale each client to the anchor's norm.
        client_norms = torch.norm(stacked, dim=1).clamp_min(1e-12)
        normed = stacked * (anchor_norm / client_norms).unsqueeze(1)

        agg = (normed * alpha_t.unsqueeze(1)).sum(dim=0)
        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": alpha_t.detach().cpu().tolist(),
            "raw_cosine": cos.detach().cpu().tolist(),
            "anchor": self.anchor,
            "anchor_norm": float(anchor_norm.item()),
            "degenerate_round": degenerate,
        }
        return agg, stats

    def state_dict(self) -> Dict[str, Any]:
        # No cross-round state; external anchor is injected per-round.
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None
