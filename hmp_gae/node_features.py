# hmp_gae/node_features.py
# Node feature extraction for HMP-GAE.
#
# Implements the paper's eta_i = f_enc(omega_i, c_i, h_i^{t-1}) with three
# concrete signal sources combined by a small MLP:
#
#   1. Random-projection signature of the flat update omega_i (JL embedding,
#      preserves approximate geometry of high-dimensional updates).
#   2. Context statistics c_i (magnitude, mean, std, cos-to-mean) -- cheap
#      summary in the spirit of Safe-FedLLM's behavioral-feature idea.
#   3. Previous-round embedding h_i^{t-1} (zero when unavailable).

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Context statistics (c_i)                                                    #
# --------------------------------------------------------------------------- #

CONTEXT_DIM = 4  # [norm, mean, std, cos_to_mean]


def context_stats(updates: torch.Tensor) -> torch.Tensor:
    """
    Compute a small statistical context vector per update.

    Args:
        updates: tensor of shape (N, d_update)

    Returns:
        (N, CONTEXT_DIM) tensor of [||u_i||, mean(u_i), std(u_i), cos(u_i, u_bar)].
        The cos-to-mean term is defined as cos(u_i, mean_j u_j); when only one
        update exists it defaults to 1.0.
    """
    n, d = updates.shape
    norms = updates.norm(dim=1, keepdim=True)               # (N, 1)
    means = updates.mean(dim=1, keepdim=True)               # (N, 1)
    stds = updates.std(dim=1, keepdim=True, unbiased=False) # (N, 1)

    if n > 1:
        u_bar = updates.mean(dim=0, keepdim=True)           # (1, d)
        cos = F.cosine_similarity(updates, u_bar.expand_as(updates), dim=1, eps=1e-12)
        cos = cos.unsqueeze(1)                              # (N, 1)
    else:
        cos = torch.ones(n, 1, device=updates.device, dtype=updates.dtype)

    # Clip/standardize to avoid exploding features.
    eps = 1e-8
    norms_log = torch.log1p(norms.clamp(min=0) + eps)
    stats = torch.cat([norms_log, means, stds, cos], dim=1)
    return stats


# --------------------------------------------------------------------------- #
# Random projection (JL)                                                      #
# --------------------------------------------------------------------------- #

class FixedRandomProjection:
    """
    Deterministic random projection R^{d_in} -> R^{d_out} with a fixed seed.

    Kept outside of nn.Module on purpose: the projection matrix is not trained
    and not updated across rounds -- holding it as a buffer ensures the feature
    geometry stays stable over time (otherwise the historical embeddings would
    live in a drifting basis).
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 42, dtype=torch.float32):
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.seed = int(seed)
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        # Scale by 1/sqrt(d_out) to keep output magnitude stable.
        self.W = torch.randn(
            self.d_in, self.d_out, generator=gen, dtype=dtype
        ) / (self.d_out ** 0.5)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Support (N, d_in) input
        if x.dim() != 2:
            raise ValueError(f"FixedRandomProjection expected 2D input, got {x.shape}")
        if x.shape[1] != self.d_in:
            raise ValueError(
                f"FixedRandomProjection: input dim {x.shape[1]} != d_in {self.d_in}"
            )
        W = self.W.to(device=x.device, dtype=x.dtype)
        return x @ W


# --------------------------------------------------------------------------- #
# Node encoder f_enc                                                          #
# --------------------------------------------------------------------------- #

class NodeFeatureEncoder(nn.Module):
    """
    f_enc: [ projected_update ; context_stats ; historical_embedding ] -> R^{eta_dim}

    The concatenated raw feature is:
        in_dim = proj_dim + CONTEXT_DIM + hist_dim
    We feed it through a small 2-layer MLP (Linear-ReLU-Linear, no dropout —
    scoring must be deterministic given the inputs) to a fixed `eta_dim`
    output (default 64).
    """

    def __init__(self, proj_dim: int, hist_dim: int, eta_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.hist_dim = int(hist_dim)
        self.eta_dim = int(eta_dim)
        in_dim = self.proj_dim + CONTEXT_DIM + self.hist_dim
        hidden = int(hidden_dim or max(self.eta_dim, 64))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.eta_dim),
        )

    def forward(
        self,
        projected: torch.Tensor,
        context: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        feats = torch.cat([projected, context, history], dim=1)
        return self.mlp(feats)


# --------------------------------------------------------------------------- #
# High-level helper                                                           #
# --------------------------------------------------------------------------- #

def compute_node_features(
    updates: torch.Tensor,
    projection: FixedRandomProjection,
    encoder: NodeFeatureEncoder,
    history: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute eta_i for all N clients.

    Args:
        updates: (N, d_update) flat LoRA updates
        projection: FixedRandomProjection (d_update -> proj_dim)
        encoder: NodeFeatureEncoder (proj_dim + CONTEXT_DIM + hist_dim -> eta_dim)
        history: (N, hist_dim) previous-round embeddings; if None -> zeros

    Returns:
        eta of shape (N, eta_dim)
    """
    n = updates.shape[0]
    projected = projection(updates)                     # (N, proj_dim)
    ctx = context_stats(updates)                        # (N, CONTEXT_DIM)
    if history is None:
        hist = torch.zeros(
            n, encoder.hist_dim, device=updates.device, dtype=updates.dtype
        )
    else:
        if history.shape != (n, encoder.hist_dim):
            raise ValueError(
                f"history shape {history.shape} != expected ({n}, {encoder.hist_dim})"
            )
        hist = history.to(device=updates.device, dtype=updates.dtype)
    return encoder(projected, ctx, hist)
