"""Shared helpers for baseline robust aggregators.

Kept in its own module so each defense file is a thin, readable port
of the algorithm without the linear-algebra plumbing.
"""

from __future__ import annotations

import torch


def pairwise_sq_l2(stacked: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared L2 distances between rows of ``stacked``.

    Args:
        stacked: (N, D) tensor.

    Returns:
        (N, N) tensor where entry (i, j) = ||u_i - u_j||^2.
    """
    d = torch.cdist(stacked, stacked, p=2.0)
    return d * d


def krum_scores(sq_distances: torch.Tensor, num_byzantine: int) -> torch.Tensor:
    """
    Per-client Krum score (Blanchard et al., NeurIPS '17).

    Score s(i) = sum over the n - f - 2 closest *other* clients of
    ||V_i - V_j||^2. Self-distance is 0 and excluded.

    Args:
        sq_distances: (N, N) squared-distance matrix.
        num_byzantine: f, number of assumed Byzantine clients.

    Returns:
        (N,) tensor of Krum scores; lower = more central / more trusted.
    """
    n = sq_distances.shape[0]
    k = max(1, n - num_byzantine - 2)  # n - f - 2 closest non-self
    sorted_d, _ = sq_distances.sort(dim=1)
    # Index 0 along each row is the self-distance (0); take indices 1..k inclusive.
    return sorted_d[:, 1: 1 + k].sum(dim=1)
