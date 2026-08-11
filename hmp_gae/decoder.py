# hmp_gae/decoder.py
# GAE decoder for the V8 HMP-GAE.
#
# Two outputs:
#   - A_hat_ij = sigmoid(gamma * cos(z_i, z_j))     (pairwise, signed cosine)
#   - H_hat    = sigmoid(Z W_dec^T) in [0,1]^{N,M}  (hyperedge incidence)
#
# We expose logits and probabilities separately so that the BCE reconstruction
# loss can be computed with numerically stable
# binary_cross_entropy_with_logits. The pre-V8 inner-product decoder
# (sigmoid(Z Z^T) after a final ReLU) was removed with the legacy modes on
# 2026-08-11 — it could never emit a negative logit.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_cosine_decoder(
    Z: torch.Tensor,
    scale: float = 4.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V8 adjacency decoder with a signed, normalized similarity logit.

    The legacy encoder ends in ReLU and ``Z @ Z.T`` can therefore never emit a
    negative logit.  V8 uses a signed final latent and cosine normalization, so
    non-neighbors can reconstruct below probability 0.5 while the fixed scale
    avoids a new learned calibration parameter on the N=7 online problem.
    """
    Z_n = F.normalize(Z, p=2, dim=1, eps=eps)
    logits = float(scale) * (Z_n @ Z_n.t())
    return logits, torch.sigmoid(logits)


class HyperedgeDecoder(nn.Module):
    """
    Per-node projection to hyperedge-incidence logits.

    H_hat_{i,e} = sigmoid( z_i^T  w_dec_e )

    With M = num_hyperedges fixed (we use M = N, one hyperedge per node,
    consistent with the k-NN construction).
    """

    def __init__(self, latent_dim: int, num_hyperedges: int):
        super().__init__()
        self.proj = nn.Linear(int(latent_dim), int(num_hyperedges), bias=False)

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(Z)  # (N, M)
        probs = torch.sigmoid(logits)
        return logits, probs
