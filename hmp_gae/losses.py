# hmp_gae/losses.py
# Self-supervised reconstruction and regularization losses for HMP-GAE.
#
# Corresponds to the paper's reconstruction loss (eq. 21):
#
#     L_rec = lambda_H * BCE(H, H_hat)  +  lambda_A * sum_ij A_hat_ij ||z_i - z_j||^2
#
# plus the historical-consistency regularizer introduced in the algorithm:
#
#     L_hist = sum_i || z_i - z_hist_i ||^2
#
# We rename the original L_B(Z) term to "smoothness" because the paper's
# definition is actually a Laplacian-style smoothness term (not a BCE).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossBundle:
    """Structured return from total_loss for logging clarity."""
    total: torch.Tensor
    L_rec_H: torch.Tensor        # BCE(H, H_hat_logits)
    L_smooth: torch.Tensor       # sum_ij A_hat_ij ||z_i - z_j||^2 (mean over N^2)
    L_hist: torch.Tensor         # mean squared distance to Z_hist
    L_reg: torch.Tensor          # L2 on trainable params (optional)


def recon_loss_H(H: torch.Tensor, H_hat_logits: torch.Tensor, pos_weight_cap: float = 10.0) -> torch.Tensor:
    """
    Binary cross-entropy reconstruction loss for the hyperedge incidence matrix.

    Uses pos_weight to counter the class imbalance between edge entries (1s)
    and non-edge entries (0s), capped for numerical stability.
    """
    num_ones = H.sum()
    num_zeros = H.numel() - num_ones
    pos_weight = torch.tensor(1.0, device=H.device, dtype=H.dtype)
    if num_ones > 0:
        ratio = (num_zeros / num_ones).clamp(min=1.0, max=pos_weight_cap)
        pos_weight = ratio.to(dtype=H.dtype)
    return F.binary_cross_entropy_with_logits(
        H_hat_logits, H, pos_weight=pos_weight
    )


def smoothness_loss(A_hat: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Laplacian-style smoothness: sum_ij A_hat_ij ||z_i - z_j||^2 / N^2.

    Computed via the identity
        sum_ij A_hat_ij ||z_i - z_j||^2 = 2 * tr(Z^T L Z)
    where L = D - A_hat is the graph Laplacian of A_hat. We compute it
    directly for clarity with N small.
    """
    N = Z.shape[0]
    if N == 0:
        return torch.zeros((), device=Z.device, dtype=Z.dtype)
    # Pairwise squared distances (N, N)
    diff_sq = torch.cdist(Z, Z, p=2.0) ** 2
    return (A_hat * diff_sq).sum() / (N * N)


def adjacency_recon_loss(
    target: torch.Tensor,
    logits: torch.Tensor,
    pos_weight_cap: float = 10.0,
) -> torch.Tensor:
    """Off-diagonal BCE for V8's fixed direct-neighbor target.

    Unlike :func:`smoothness_loss`, the target is constructed once from the
    detached raw-update topology and is independent of the decoder output.
    The diagonal is excluded because every node is trivially identical to
    itself and would otherwise dominate the tiny N=7 loss.
    """
    if target.shape != logits.shape or target.dim() != 2:
        raise ValueError(
            f"adjacency target/logits mismatch: {target.shape} vs {logits.shape}"
        )
    N = int(target.shape[0])
    mask = ~torch.eye(N, device=target.device, dtype=torch.bool)
    y = target.to(dtype=logits.dtype)[mask]
    z = logits[mask]
    if y.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    positives = y.sum()
    negatives = y.numel() - positives
    pos_weight = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    if float(positives) > 0.0:
        pos_weight = (negatives / positives).clamp(
            min=1.0, max=float(pos_weight_cap)
        ).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)


def per_node_adjacency_recon_error(
    target: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Mean held-structure BCE per client (the V8 recon residual)."""
    if target.shape != logits.shape or target.dim() != 2:
        raise ValueError(
            f"per-node adjacency target/logits mismatch: {target.shape} vs {logits.shape}"
        )
    N = int(target.shape[0])
    y = target.to(dtype=logits.dtype)
    elem = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = 1.0 - torch.eye(N, device=logits.device, dtype=logits.dtype)
    return (elem * mask).sum(dim=1) / max(1, N - 1)


def hist_loss(Z: torch.Tensor, Z_hist: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Historical consistency: mean squared deviation of Z from EMA history.

    Returns zero when Z_hist is None (cold start).
    """
    if Z_hist is None:
        return torch.zeros((), device=Z.device, dtype=Z.dtype)
    if Z_hist.shape != Z.shape:
        raise ValueError(f"Z_hist shape {Z_hist.shape} != Z shape {Z.shape}")
    Z_hist_d = Z_hist.detach().to(device=Z.device, dtype=Z.dtype)
    return ((Z - Z_hist_d) ** 2).mean()


def param_l2(params) -> torch.Tensor:
    """Sum of squared L2 norms over given parameters (for weight_decay logging)."""
    total = None
    for p in params:
        if p is None or not p.requires_grad:
            continue
        t = p.pow(2).sum()
        total = t if total is None else total + t
    if total is None:
        # Return a 0-dim tensor on a reasonable device.
        return torch.zeros(())
    return total


def total_loss(
    H: torch.Tensor,
    H_hat_logits: torch.Tensor,
    A_hat: torch.Tensor,
    Z: torch.Tensor,
    Z_hist: Optional[torch.Tensor],
    lambda_H: float = 1.0,
    lambda_A: float = 1.0,
    lambda_hist: float = 0.5,
    weight_decay: float = 0.0,
    params=None,
) -> LossBundle:
    L_H = recon_loss_H(H, H_hat_logits)
    L_S = smoothness_loss(A_hat, Z)
    L_Hi = hist_loss(Z, Z_hist)
    if weight_decay > 0 and params is not None:
        L_reg = weight_decay * param_l2(params).to(L_H.device)
    else:
        L_reg = torch.zeros((), device=L_H.device, dtype=L_H.dtype)
    total = lambda_H * L_H + lambda_A * L_S + lambda_hist * L_Hi + L_reg
    return LossBundle(total=total, L_rec_H=L_H, L_smooth=L_S, L_hist=L_Hi, L_reg=L_reg)


def total_loss_v8(
    H: torch.Tensor,
    H_hat_logits: torch.Tensor,
    adjacency_target: torch.Tensor,
    adjacency_logits: torch.Tensor,
    Z: torch.Tensor,
    Z_hist: Optional[torch.Tensor],
    lambda_H: float = 1.0,
    lambda_A: float = 1.0,
    lambda_hist: float = 0.5,
    weight_decay: float = 0.0,
    params=None,
) -> LossBundle:
    """V8 objective: reconstruct a fixed topology instead of self-smoothing.

    ``LossBundle.L_smooth`` carries the structural reconstruction BCE for
    backward-compatible logging; the runtime additionally labels it
    ``L_struct`` in V8 diagnostics.
    """
    L_H = recon_loss_H(H, H_hat_logits)
    L_A = adjacency_recon_loss(adjacency_target, adjacency_logits)
    L_Hi = hist_loss(Z, Z_hist)
    if weight_decay > 0 and params is not None:
        L_reg = weight_decay * param_l2(params).to(L_H.device)
    else:
        L_reg = torch.zeros((), device=L_H.device, dtype=L_H.dtype)
    total = lambda_H * L_H + lambda_A * L_A + lambda_hist * L_Hi + L_reg
    return LossBundle(total=total, L_rec_H=L_H, L_smooth=L_A, L_hist=L_Hi, L_reg=L_reg)
