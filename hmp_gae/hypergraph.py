# hmp_gae/hypergraph.py
# k-NN hypergraph construction for HMP-GAE.
#
# For each node i we create one hyperedge epsilon_i = {i} U top-k nearest
# neighbors of i (by cosine similarity in the eta feature space). Setting
# M = N (one hyperedge per node centered at that node) makes the incidence
# matrix square and keeps the decoder dimension stable across rounds.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def knn_hypergraph(
    eta: torch.Tensor,
    k: int,
    include_self: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a k-NN hypergraph from node features.

    Args:
        eta: (N, d) node feature matrix (dense).
        k:   number of neighbors per hyperedge (excluding self). Effective k
             is clipped to [1, N-1]. When include_self=True, each hyperedge
             contains k+1 nodes (self + k neighbors).
        include_self: if True, node i is always in hyperedge epsilon_i.
        eps: numerical guard for degree inversion.

    Returns:
        H     : (N, N) incidence matrix in {0, 1}. H[i, e] = 1 means node i
                belongs to hyperedge e. Column e is centered on node e.
        D_V_inv : (N,) 1/degree(node i). Used as diag(D_V^{-1}) (kept as a
                  vector to avoid constructing a dense diag matrix).
        D_E_inv : (N,) 1/degree(hyperedge e). Same.
    """
    if eta.dim() != 2:
        raise ValueError(f"knn_hypergraph expects 2D eta, got {eta.shape}")
    N, _ = eta.shape
    if N == 0:
        raise ValueError("knn_hypergraph received empty eta (N=0)")

    # Effective neighborhood size.
    k_eff = max(1, min(int(k), N - 1))

    # Pairwise cosine similarity matrix (N, N). Diagonal becomes 1.
    eta_n = F.normalize(eta, p=2, dim=1, eps=eps)
    sim = eta_n @ eta_n.t()

    # Mask out self-similarities so they don't dominate the top-k selection.
    sim_for_knn = sim.clone()
    sim_for_knn.fill_diagonal_(float("-inf"))

    # top-k neighbors per node (as columns of the hyperedge centered at node i).
    _, nbrs = torch.topk(sim_for_knn, k=k_eff, dim=1)  # (N, k_eff)

    H = torch.zeros(N, N, device=eta.device, dtype=eta.dtype)
    rows = torch.arange(N, device=eta.device).view(-1, 1).expand_as(nbrs)
    # H[ nbrs[i, j], i ] = 1 : neighbor membership in hyperedge i
    # i.e. column i is filled with the neighbors of center node i.
    H[nbrs, rows] = 1.0
    if include_self:
        # Also include the center node itself in its hyperedge.
        diag_idx = torch.arange(N, device=eta.device)
        H[diag_idx, diag_idx] = 1.0

    # Degree vectors (node degree = #hyperedges node belongs to; hyperedge
    # degree = #nodes in that hyperedge).
    d_v = H.sum(dim=1)  # (N,)
    d_e = H.sum(dim=0)  # (N,) -- equal to k_eff + (1 if include_self else 0)

    D_V_inv = 1.0 / d_v.clamp(min=eps)
    D_E_inv = 1.0 / d_e.clamp(min=eps)

    return H, D_V_inv, D_E_inv


def semantic_js_similarity(
    probe_distributions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a bounded client-by-client behavior similarity matrix.

    ``probe_distributions`` has shape ``(N, K, C)``.  For each client pair we
    average Jensen-Shannon divergence over the same K probe examples, then map
    it to similarity with ``1 - JS / log(2)``.  The result lies in ``[0, 1]``
    (up to float dust), is symmetric, and does not use probe labels.

    V8 deliberately builds this behavior view independently from the update
    view.  A relation is trusted for risk propagation only when both views
    agree, so neither update geometry nor semantic behavior can certify a
    client on its own.
    """
    if probe_distributions.dim() != 3:
        raise ValueError(
            "semantic_js_similarity expects (N, K, C), got "
            f"{tuple(probe_distributions.shape)}"
        )
    P = probe_distributions.clamp(min=eps)
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=eps)
    Pi = P[:, None, :, :]  # (N, 1, K, C)
    Pj = P[None, :, :, :]  # (1, N, K, C)
    M = 0.5 * (Pi + Pj)
    js = 0.5 * (
        (Pi * (Pi.log() - M.log())).sum(dim=-1)
        + (Pj * (Pj.log() - M.log())).sum(dim=-1)
    ).mean(dim=-1)
    sim = 1.0 - js / float(torch.log(torch.tensor(2.0)))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    sim = sim.clamp(min=0.0, max=1.0)
    sim.fill_diagonal_(1.0)
    return sim


def knn_hypergraph_from_similarity(
    similarity: torch.Tensor,
    k: int,
    include_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the centered binary k-NN incidence matrix from a similarity.

    This is the V8 counterpart of :func:`knn_hypergraph` for views whose
    pairwise similarity is computed directly (the semantic-probe view).  It
    keeps the same orientation: column ``e`` is the hyperedge centered on
    client ``e`` and contains the neighbors selected by row ``e``.
    """
    if similarity.dim() != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            "knn_hypergraph_from_similarity expects square (N, N), got "
            f"{tuple(similarity.shape)}"
        )
    N = int(similarity.shape[0])
    if N == 0:
        raise ValueError("knn_hypergraph_from_similarity received N=0")
    k_eff = max(1, min(int(k), N - 1))
    sim = torch.nan_to_num(
        similarity, nan=float("-inf"), posinf=1.0, neginf=float("-inf")
    ).clone()
    sim.fill_diagonal_(float("-inf"))
    nbrs = torch.topk(sim, k=k_eff, dim=1).indices

    H = torch.zeros(N, N, device=similarity.device, dtype=similarity.dtype)
    centers = torch.arange(N, device=similarity.device).view(-1, 1).expand_as(nbrs)
    H[nbrs, centers] = 1.0
    if include_self:
        idx = torch.arange(N, device=similarity.device)
        H[idx, idx] = 1.0
    d_v = H.sum(dim=1)
    d_e = H.sum(dim=0)
    eps = torch.finfo(H.dtype).eps
    return H, 1.0 / d_v.clamp(min=eps), 1.0 / d_e.clamp(min=eps)


def mutual_neighbor_adjacency(H: torch.Tensor) -> torch.Tensor:
    """Return the symmetric direct-mutual-neighbor adjacency induced by H.

    A pair is present only if each endpoint selected the other.  This is
    intentionally stricter than ``(H @ H.T) > 0``: merely co-occurring through
    a third center is not enough to carry V8 risk.
    """
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            f"mutual_neighbor_adjacency expects square H, got {tuple(H.shape)}"
        )
    directed = H.t() > 0  # directed[center, selected_client]
    directed = directed.clone()
    directed.fill_diagonal_(False)
    mutual = directed & directed.t()
    return mutual


def consensus_propagation_hypergraph(
    update_H: torch.Tensor,
    behavior_H: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build V8's conservative cross-view propagation hypergraph.

    Only client pairs that are mutual k-NN neighbors in BOTH the raw-update
    and probe-behavior views survive.  Each surviving pair appears in both
    endpoint-centered hyperedges; self membership keeps the incidence shape
    stable at ``(N, N)``.  Isolated singleton columns do not propagate risk
    because :func:`hypergraph_propagation_matrix` removes self transitions.

    Returns ``(H, D_V_inv, D_E_inv, consensus_adjacency)``.
    """
    if update_H.shape != behavior_H.shape:
        raise ValueError(
            f"update_H shape {tuple(update_H.shape)} != behavior_H "
            f"shape {tuple(behavior_H.shape)}"
        )
    mutual = mutual_neighbor_adjacency(update_H) & mutual_neighbor_adjacency(behavior_H)
    N = int(update_H.shape[0])
    H = mutual.to(dtype=update_H.dtype).t().contiguous()
    idx = torch.arange(N, device=H.device)
    H[idx, idx] = 1.0
    d_v = H.sum(dim=1)
    d_e = H.sum(dim=0)
    eps = torch.finfo(H.dtype).eps
    return H, 1.0 / d_v.clamp(min=eps), 1.0 / d_e.clamp(min=eps), mutual


def hypergraph_propagation_matrix(
    H: torch.Tensor,
    pair_affinity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an off-diagonal node-to-edge-to-node risk operator.

    The base operator is ``D_V^-1 H D_E^-1 H.T``.  V8 removes the diagonal so
    a CSE seed cannot simply feed itself and row-normalizes the remaining mass.
    It then optionally attenuates surviving pairs with the learned GAE
    affinity WITHOUT normalizing that mass back to one.  The result is
    therefore sub-stochastic: a weakly reconstructed relation carries weak
    risk even when it is a node's only relation.  Zero rows abstain exactly.
    """
    if H.dim() != 2:
        raise ValueError(f"hypergraph_propagation_matrix expects 2D H, got {H.shape}")
    d_v = H.sum(dim=1)
    d_e = H.sum(dim=0)
    eps = torch.finfo(H.dtype).eps
    T = (H * (1.0 / d_v.clamp(min=eps)).unsqueeze(1)) @ (
        H.t() * (1.0 / d_e.clamp(min=eps)).unsqueeze(1)
    )
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if T.shape[0] == T.shape[1]:
        T.fill_diagonal_(0.0)
    row_sum = T.sum(dim=1, keepdim=True)
    T = torch.where(row_sum > eps, T / row_sum.clamp(min=eps), torch.zeros_like(T))
    if pair_affinity is not None:
        if pair_affinity.shape != T.shape:
            raise ValueError(
                f"pair_affinity shape {tuple(pair_affinity.shape)} != "
                f"propagation shape {tuple(T.shape)}"
            )
        affinity = torch.nan_to_num(
            pair_affinity.to(device=T.device, dtype=T.dtype),
            nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp(min=0.0, max=1.0)
        T = T * affinity
    return T


def apply_diag_inv(D_inv_vec: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficient equivalent of diag(D_inv_vec) @ X via broadcasting.
    """
    if D_inv_vec.dim() != 1:
        raise ValueError(f"Expected 1D D_inv_vec, got {D_inv_vec.shape}")
    return D_inv_vec.unsqueeze(1) * X
