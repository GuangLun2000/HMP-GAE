"""FoolsGold (Fung et al., RAID '20).

Detects coordinated / sybil attackers via cosine similarity of
historically *accumulated* per-client updates. The intuition:
- Sybil clients pursue a shared adversarial objective, so their
  accumulated updates point in similar directions across rounds.
- Benign clients on heterogeneous data have accumulated updates that
  spread out in direction.

Pipeline (per round):
  1. Per-client clip-to-unit-norm of this round's update.
  2. Add the clipped update into a running per-client sum.
  3. Compute pairwise cosine similarity of the accumulated sums.
  4. Pardoning: if client i's max similarity is smaller than client j's,
     scale cos[i][j] down by max_cs[i] / max_cs[j].
  5. wv[i] = 1 - max_j cos[i][j], clip to [0, 1].
  6. Normalize, then logit transform (FoolsGold paper).
  7. Final clipping to [0, 1]; normalize to sum 1 -> alpha.

Notes:
- The FLPoison/FoolsGold reference does NOT normalize wv to sum to 1
  (it leaves a "shrunken aggregation" when sybils dominate). Our Defense
  interface requires alpha to sum to ~1.0, so we renormalize. This
  changes the effective server learning rate under heavy sybil pressure
  but is the correct adapter to our orchestration.
- The reference also masks updates to "indicative features" (top-k of
  the last classifier layer). We omit this mask because the Defense
  interface receives a flat update vector without layer structure; on
  the LoRA-fine-tuned models used here, the full-vector cosine is the
  standard substitute used in subsequent FoolsGold reimplementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from defense import Defense, FedAvgDefense


class FoolsGoldDefense(Defense):
    """FoolsGold (RAID '20)."""

    name = "foolsgold"

    def __init__(self, num_clients: int, config: Optional[Dict[str, Any]] = None):
        self.num_clients = int(num_clients)
        cfg = dict(config or {})
        self.epsilon = float(cfg.get("epsilon", 1e-6))
        # Per-client accumulated (clipped) update sum, shape (num_clients, D).
        # Built lazily on first aggregate when D is known.
        self._history: Optional[torch.Tensor] = None
        self._fallback = FedAvgDefense()

    def _ensure_history(self, d: int, dtype: torch.dtype, device: torch.device) -> None:
        if (
            self._history is None
            or self._history.shape != (self.num_clients, d)
            or self._history.device != device
            or self._history.dtype != dtype
        ):
            self._history = torch.zeros(
                self.num_clients, d, dtype=dtype, device=device
            )

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
            raise ValueError("FoolsGoldDefense.aggregate received 0 updates")
        if n < 2:
            agg, stats = self._fallback.aggregate(
                updates, client_ids, data_sizes, round_num, device
            )
            stats["defense_name"] = self.name
            stats["fallback_reason"] = "n < 2"
            return agg, stats

        stacked = torch.stack(updates).to(device)
        d = stacked.shape[1]
        self._ensure_history(d, stacked.dtype, stacked.device)

        # 1) Per-client norm clip to unit ball (only if > 1).
        norms = torch.norm(stacked, dim=1, keepdim=True).clamp_min(1.0)
        clipped = stacked / norms

        # 2) Accumulate this round's clipped updates into per-client history.
        for row, cid in enumerate(client_ids):
            if 0 <= cid < self.num_clients:
                self._history[cid] = self._history[cid] + clipped[row]
        hist_round = self._history[client_ids]  # (n, D)

        # 3) Pairwise cosine similarity of accumulated updates; zero the diag.
        cos = F.cosine_similarity(
            hist_round.unsqueeze(1), hist_round.unsqueeze(0), dim=2
        )
        eye = torch.eye(n, dtype=cos.dtype, device=cos.device)
        cos = cos - eye

        # 4) Pardoning: scale cos[i][j] by min(1, max_cs[i] / max_cs[j]).
        max_cs = cos.max(dim=1).values + self.epsilon  # (n,)
        ratio = (max_cs.unsqueeze(1) / max_cs.unsqueeze(0)).clamp(max=1.0)
        # Leave the diagonal at 1.0 so we don't accidentally scale self-cells.
        ratio = ratio - torch.diag(torch.diag(ratio)) + eye
        pardoned = cos * ratio

        # 5) Weight: low max-similarity -> high weight.
        wv = 1.0 - pardoned.max(dim=1).values
        wv = wv.clamp(min=0.0, max=1.0)
        wv_max = wv.max().clamp_min(self.epsilon)
        wv = (wv / wv_max).clamp(max=0.99)

        # 6) Logit transform (FoolsGold paper) and final clamp to [0, 1].
        wv = torch.log(wv / (1.0 - wv) + self.epsilon) + 0.5
        wv = wv.clamp(min=0.0, max=1.0)

        # 7) Normalize to sum 1 (Defense-interface contract).
        wv_sum = wv.sum().clamp_min(self.epsilon)
        alpha = wv / wv_sum
        agg = (stacked * alpha.unsqueeze(1)).sum(dim=0)

        stats: Dict[str, Any] = {
            "defense_name": self.name,
            "alpha": alpha.detach().cpu().tolist(),
            "pre_logit_wv": (wv / wv_sum).detach().cpu().tolist(),
            "pardoned_max_cos": pardoned.max(dim=1).values.detach().cpu().tolist(),
        }
        return agg, stats

    def state_dict(self) -> Dict[str, Any]:
        if self._history is None:
            return {}
        return {"history": self._history.detach().cpu()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        h = state.get("history") if state else None
        if h is not None:
            self._history = h.clone()
