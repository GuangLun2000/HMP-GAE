# hmp_gae/decoder.py
# GAE decoder for HMP-GAE.
#
# Two outputs per the paper:
#   - A_hat_ij = sigmoid(z_i^T z_j)                 (eq. 17, pairwise)
#   - H_hat    = sigmoid(Z W_dec^T) in [0,1]^{N,M}  (eq. 18, hyperedge incidence)
#
# We expose logits and probabilities separately so that the BCE reconstruction
# loss can be computed with numerically stable
# binary_cross_entropy_with_logits.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def inner_product_decoder(Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute A_hat logits and probabilities.

    Args:
        Z: (N, latent_dim)

    Returns:
        A_hat_logits: (N, N), Z Z^T (un-sigmoid)
        A_hat_probs:  (N, N), sigmoid(Z Z^T)
    """
    logits = Z @ Z.t()
    probs = torch.sigmoid(logits)
    return logits, probs


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
