# hmp_gae/trust_scorer.py
# Closed-form trust scoring for HMP-GAE.
#
# Trust score s_i combines four structural signals (each z-scored):
#   1. graph_residual_i : how "off-cluster" node i is in the k-NN hypergraph
#                         incidence H. Attackers that fail to share hyperedges
#                         with benign clients have high residual.
#   2. recon_residual_i : 1 - mean_{j != i} A_hat_ij from the GAE-reconstructed
#                         adjacency. Refines (1) once the encoder has trained.
#   3. sem_div_i        : per-sample symmetric KL divergence of the client's
#                         softmax outputs (on a fixed probe set) to its peers,
#                         averaged. Catches "geometrically stealthy" attackers
#                         whose updates pass cosine/L2 checks but whose local
#                         model still produces semantically inverted predictions.
#   4. hist_dev_i       : ||z_i - z_hist_i||_2 vs EMA latent history. Off by
#                         default (benign drift > attacker drift in real runs).
#
# Combined:
#     s_i = - ( graph_weight        * z(graph_residual_i)
#             + residual_weight     * z(recon_residual_i)
#             + semantic_weight     * z(sem_div_i)
#             + hist_weight         * z(hist_dev_i) )
#     alpha_i = softmax( s_i / tau )
#
# Rationale:
#   - graph + recon = pure update-geometry signal (cheap, but a stealth
#     attacker with cosine/norm projection can mimic benign geometry).
#   - sem_div = output-behavior signal (orthogonal to update geometry; an
#     attacker has to *both* match update statistics *and* produce benign-like
#     per-sample probabilities, which is incompatible with hallucination).
#   - tau -> 0 = Krum-like hard selection; tau in [0.05, 0.5] = soft rejection.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class TrustResult:
    alpha: torch.Tensor          # (N,) non-negative weights summing to 1
    s: torch.Tensor              # (N,) trust logits
    # Signal 1: graph-structural residual from the hypergraph incidence H.
    # High residual = this node is included in few hyperedges (isolated).
    graph_residual: torch.Tensor        # (N,) in [0, 1]
    graph_residual_z: torch.Tensor      # z-scored graph_residual
    # Signal 2: decoder-based residual from the reconstructed A_hat.
    # High residual = low average similarity to other nodes in learned
    # latent space. Noisy until the encoder is sufficiently trained.
    recon_residual: torch.Tensor        # (N,)
    recon_residual_z: torch.Tensor      # z-scored recon_residual
    # Signal 3: per-sample semantic divergence on a fixed probe subset.
    # All-zero when probe_distributions is None.
    sem_div: torch.Tensor               # (N,)
    sem_div_z: torch.Tensor             # z-scored sem_div
    # Signal 4: historical deviation (disabled by default in V1 because
    # benign clients drift more than attackers during real learning).
    hist_dev: torch.Tensor              # (N,)
    hist_dev_z: torch.Tensor            # z-scored hist_dev
    # L2 norm of the ACTIVE signal weights, sqrt(sum_k w_k^2) over signals
    # that actually contributed to s. Used by the gate_rezscore=False path to
    # express the suspicion threshold in per-signal z units, invariant to
    # rescaling / enabling additional signals (otherwise adding a signal
    # would silently tighten the same numeric threshold).
    weight_norm: float = 1.0


def _zscore(
    x: torch.Tensor,
    eps: float = 1e-6,
    mode: str = "std",
    clip: Optional[float] = None,
) -> torch.Tensor:
    """
    Standardize a 1-D signal across clients.

    mode='std' (default): classic (x - mean) / std. Breaks down as the
        attacker fraction grows -- attackers pull mean/std toward themselves
        and their own z-scores back into the benign range.
    mode='mad': robust (x - median) / (1.4826 * MAD). The median/MAD are
        computed from the (benign) majority, so attacker z-scores stay large
        up to a ~50% attacker fraction. When MAD degenerates to 0 (e.g. the
        coarsely-quantized graph_residual where the benign majority ties
        exactly), fall back to the std scale rather than dividing by eps --
        otherwise a single one-quantum deviation would explode to z ~ 1e5.

    clip: optional symmetric bound on |z|. Keeps a genuinely extreme outlier
        from dominating the weighted signal sum by orders of magnitude.
    """
    if x.numel() == 0:
        return x
    m = (mode or "std").lower()
    if m == "mad":
        med = x.median()
        mad = (x - med).abs().median()
        scale = 1.4826 * mad
        if float(scale) < eps:
            scale = x.std(unbiased=False)
        z = (x - med) / scale.clamp(min=eps)
    elif m == "std":
        mean = x.mean()
        std = x.std(unbiased=False).clamp(min=eps)
        z = (x - mean) / std
    else:
        raise ValueError(f"Unknown zscore mode={mode!r}; expected 'std' or 'mad'")
    if clip is not None:
        c = abs(float(clip))
        z = z.clamp(min=-c, max=c)
    return z


def _semantic_divergence_signal(
    probe_dists: torch.Tensor,
    reference: str = "pairwise",
    confidence_weight: bool = False,
) -> torch.Tensor:
    """
    Per-client semantic divergence on a fixed probe subset.

    reference='pairwise' (default): for each probe sample k and each ordered
        client pair (i, j), compute KL(p_i^k || p_j^k), symmetrize, average
        over peers j != i and over the K samples. Weakness under non-IID:
        legitimately heterogeneous benign clients diverge from peers and get
        penalized, and every benign score is inflated by its distance to the
        attackers (compressing attacker/benign contrast).
    reference='median': compare each client to the per-sample per-class
        median distribution across clients (renormalized). Attackers are a
        minority so they cannot move the median; a benign score reduces to
        its own heterogeneity bias while an attacker score measures its
        systematic wrongness -- larger contrast, robust to <50% attackers.

    confidence_weight: weight each sample's divergence by the client's own
        max softmax prob. "Confidently wrong" (attacker) counts in full;
        "unconfidently different" (typical non-IID benign) is discounted.

    Honest clients agree per-sample on the correct class -> low divergence.
    Hallucination attackers invert per-sample predictions vs honest peers
    -> high divergence, even when their flat update is geometrically stealthy.

    Args:
        probe_dists: (N, K, C) softmax probabilities. Must be non-negative.

    Returns:
        (N,) mean per-sample symmetric KL to the chosen reference.
    """
    if probe_dists.dim() != 3:
        raise ValueError(
            f"probe_dists must be (N, K, C), got {tuple(probe_dists.shape)}"
        )
    eps = 1e-8
    P = probe_dists.clamp(min=eps)
    P = P / P.sum(dim=-1, keepdim=True)
    logP = P.log()
    N, K, _ = P.shape
    device, dtype = P.device, P.dtype
    if N <= 1 or K == 0:
        return torch.zeros(N, device=device, dtype=dtype)

    ref = (reference or "pairwise").lower()
    if ref == "median":
        R = P.median(dim=0).values                       # (K, C)
        R = R.clamp(min=eps)
        R = R / R.sum(dim=-1, keepdim=True)
        logR = R.log()
        # symKL(P_i^k || R^k), per (i, k)
        kl_pr = (P * (logP - logR.unsqueeze(0))).sum(dim=-1)               # (N, K)
        kl_rp = (R.unsqueeze(0) * (logR.unsqueeze(0) - logP)).sum(dim=-1)  # (N, K)
        div_ik = 0.5 * (kl_pr + kl_rp)                                     # (N, K)
    elif ref == "pairwise":
        # H_ik = sum_c P[i,k,c] * logP[i,k,c]                       (N, K)
        # X_ijk = sum_c P[i,k,c] * logP[j,k,c]                      (N, N, K)
        # KL_ijk = H_ik - X_ijk                                     (N, N, K)
        H_ik = (P * logP).sum(dim=-1)
        X = torch.einsum("ikc,jkc->ijk", P, logP)
        KL = H_ik.unsqueeze(1) - X
        sym_KL = 0.5 * (KL + KL.transpose(0, 1))
        mask = 1.0 - torch.eye(N, device=device, dtype=dtype)
        if not confidence_weight:
            # Keep the original single-expression reduction bit-for-bit so
            # pre-existing experiments reproduce exactly.
            return (sym_KL * mask.unsqueeze(-1)).sum(dim=(1, 2)) / float((N - 1) * K)
        div_ik = (sym_KL * mask.unsqueeze(-1)).sum(dim=1) / float(N - 1)   # (N, K)
    else:
        raise ValueError(
            f"Unknown semantic reference={reference!r}; expected 'pairwise' or 'median'"
        )

    if confidence_weight:
        w = P.max(dim=-1).values                                           # (N, K)
        return (div_ik * w).sum(dim=1) / w.sum(dim=1).clamp(min=eps)
    return div_ik.mean(dim=1)


def compute_trust_weights(
    A_hat: torch.Tensor,
    Z: torch.Tensor,
    Z_hist: Optional[torch.Tensor],
    H: Optional[torch.Tensor] = None,
    graph_weight: float = 1.0,
    residual_weight_alpha: float = 0.3,
    hist_weight_beta: float = 0.0,
    softmax_tau: float = 0.1,
    min_alpha_clip: float = 1e-6,
    probe_distributions: Optional[torch.Tensor] = None,
    semantic_weight: float = 0.0,
    zscore_mode: str = "std",
    zscore_clip: Optional[float] = None,
    semantic_reference: str = "pairwise",
    semantic_confidence_weight: bool = False,
) -> TrustResult:
    """
    Compute closed-form trust weights for N clients.

    Combines three signals (each z-scored for scale invariance):

        s_i = - ( graph_weight           * z(graph_residual_i)
                + residual_weight_alpha  * z(recon_residual_i)
                + hist_weight_beta       * z(hist_dev_i) )

    graph_residual uses only the deterministic k-NN hypergraph incidence H
    (so it is robust even when the HMP encoder is only partially trained).
    recon_residual uses the learned A_hat (informative once encoder has
    converged). hist_dev is included for completeness but defaults to weight
    zero -- benign clients learning from data drift more than attackers
    trapped on a fixed mislabel manifold, which can invert the signal.

    Args:
        A_hat:  (N, N) reconstructed adjacency in [0, 1].
        Z:      (N, d) latent embeddings from the HMP encoder.
        Z_hist: (N, d) EMA history embeddings (None on cold start).
        H:      (N, M) incidence matrix (optional; required for graph signal).

    Returns:
        TrustResult with alpha (N,) and diagnostic tensors.
    """
    N = A_hat.shape[0]
    device = A_hat.device
    dtype = A_hat.dtype

    if N == 0:
        empty = torch.zeros(0, device=device, dtype=dtype)
        return TrustResult(
            alpha=empty, s=empty,
            graph_residual=empty, graph_residual_z=empty,
            recon_residual=empty, recon_residual_z=empty,
            sem_div=empty, sem_div_z=empty,
            hist_dev=empty, hist_dev_z=empty,
        )

    # ---- Signal 1: graph residual from hypergraph incidence H ---- #
    # A node with low "reach" across hyperedges is isolated/anomalous.
    # Specifically, we measure how many other nodes share at least one
    # hyperedge with node i: reach_i = (H H^T)[i, :] count > 0.
    # Normalized to [0, 1]: graph_residual_i = 1 - reach_i / (N - 1).
    if H is not None and N > 1:
        # co_membership[i, j] = #hyperedges shared between i and j.
        co = (H @ H.t())                           # (N, N)
        co.fill_diagonal_(0.0)
        reach = (co > 0).to(dtype).sum(dim=1)      # (N,) # peers touched
        graph_residual = 1.0 - reach / max(1, N - 1)
    else:
        graph_residual = torch.zeros(N, device=device, dtype=dtype)

    # ---- Signal 2: reconstructed adjacency residual ---- #
    off_mask = 1.0 - torch.eye(N, device=device, dtype=dtype)
    if N > 1:
        recon_residual = 1.0 - (A_hat * off_mask).sum(dim=1) / (N - 1)
    else:
        recon_residual = torch.zeros(N, device=device, dtype=dtype)

    # ---- Signal 3: per-sample semantic divergence ---- #
    if probe_distributions is None or semantic_weight == 0.0:
        sem_div = torch.zeros(N, device=device, dtype=dtype)
        use_sem = False
    else:
        sem_div = _semantic_divergence_signal(
            probe_distributions.to(device=device, dtype=dtype),
            reference=semantic_reference,
            confidence_weight=semantic_confidence_weight,
        )
        use_sem = True

    # ---- Signal 4: historical deviation ---- #
    if Z_hist is None:
        hist_dev = torch.zeros(N, device=device, dtype=dtype)
        use_hist = False
    else:
        Z_hist_d = Z_hist.detach().to(device=device, dtype=dtype)
        hist_dev = (Z - Z_hist_d).norm(dim=1)
        use_hist = True

    graph_residual_z = _zscore(graph_residual, mode=zscore_mode, clip=zscore_clip)
    recon_residual_z = _zscore(recon_residual, mode=zscore_mode, clip=zscore_clip)
    sem_div_z = (
        _zscore(sem_div, mode=zscore_mode, clip=zscore_clip)
        if use_sem else torch.zeros_like(sem_div)
    )
    hist_dev_z = (
        _zscore(hist_dev, mode=zscore_mode, clip=zscore_clip)
        if use_hist else torch.zeros_like(hist_dev)
    )

    s = -(
        graph_weight * graph_residual_z
        + residual_weight_alpha * recon_residual_z
        + semantic_weight * sem_div_z
        + hist_weight_beta * hist_dev_z
    )

    # L2 norm of the weights of the signals that actually contributed to s.
    # The gate_rezscore=False path divides the suspicion score by this so the
    # rejection threshold is expressed in per-signal z units and stays valid
    # when signal weights are rescaled or extra signals are switched on.
    active_sq = float(graph_weight) ** 2 + float(residual_weight_alpha) ** 2
    if use_sem:
        active_sq += float(semantic_weight) ** 2
    if use_hist:
        active_sq += float(hist_weight_beta) ** 2
    weight_norm = active_sq ** 0.5 if active_sq > 0.0 else 1.0

    tau = max(float(softmax_tau), 1e-4)
    alpha = torch.softmax(s / tau, dim=0)
    if min_alpha_clip > 0:
        alpha = alpha.clamp(min=min_alpha_clip)
        alpha = alpha / alpha.sum()

    return TrustResult(
        alpha=alpha, s=s,
        graph_residual=graph_residual,
        graph_residual_z=graph_residual_z,
        recon_residual=recon_residual,
        recon_residual_z=recon_residual_z,
        sem_div=sem_div,
        sem_div_z=sem_div_z,
        hist_dev=hist_dev,
        hist_dev_z=hist_dev_z,
        weight_norm=weight_norm,
    )


def weighted_aggregate(updates, alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute sum_i alpha_i * update_i with shape-robust accumulation.

    Works with either a list of 1-D tensors or a (N, D) stacked tensor.
    """
    if isinstance(updates, list):
        stacked = torch.stack(updates)
    else:
        stacked = updates
    stacked = stacked.to(device=alpha.device, dtype=alpha.dtype)
    return (stacked * alpha.view(-1, 1)).sum(dim=0)


def _suspicion_signal(
    trust: "TrustResult",
    source: str,
    gate_rezscore: bool = True,
    zscore_mode: str = "std",
    zscore_clip: Optional[float] = None,
) -> torch.Tensor:
    """
    Pick which suspicion signal drives the rejection gate.

      'graph'    : use graph_residual_z only (backward-compatible).
                   Robust at cold start; ignores recon and sem_div.
      'combined' : the full trust logit (-trust.s, since trust.s is built so
                   that high s = trustworthy). Lets all enabled signals
                   (graph + recon + semantic + hist) drive the gate.

    gate_rezscore controls whether combined mode re-z-scores -trust.s:
      True  (default, backward-compatible): the historical double-z-score.
            It forces the round's suspicion values onto a +-sigma scale, so
            even an all-benign round always pushes its most extreme client
            past a fixed threshold -- the "scapegoat" failure mode.
      False (recommended): use -trust.s / trust.weight_norm. Each component
            of s is already z-scored, so the weighted sum carries an absolute
            scale: an all-benign round stays near 0 and no one gets gated,
            while attackers land at |sus| >> threshold. Dividing by the L2
            norm of the active signal weights expresses the threshold in
            per-signal z units, so it survives weight rescaling and signal
            on/off toggles. Retune reject_z_threshold (~2.5 in those units)
            when switching this off.

    Combined mode is the right choice once any of recon/semantic/hist
    weights are non-zero, because graph-only gating would silently discard
    those signals.
    """
    src = (source or "graph").lower()
    if src == "combined":
        sus = (-trust.s).detach()
        if gate_rezscore:
            return _zscore(sus, mode=zscore_mode, clip=zscore_clip)
        norm = float(getattr(trust, "weight_norm", 1.0))
        return sus / max(norm, 1e-6)
    if src != "graph":
        raise ValueError(
            f"Unknown gate_signal={source!r}; expected 'graph' or 'combined'"
        )
    return trust.graph_residual_z.detach().clone()


def gate_diagnostics(
    trust: "TrustResult",
    reject_z_threshold: float,
    soft_reject_k: float,
    gate_signal: str,
    gate_rezscore: bool = True,
    zscore_mode: str = "std",
    zscore_clip: Optional[float] = None,
    sus_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single source of truth for the soft-rejection gate.

    Returns (sus_z, gate) where:
      sus_z = suspicion score selected by gate_signal (see _suspicion_signal;
              with gate_rezscore=True and 'combined' this is the historical
              SECOND z-score _zscore(-trust.s)).
      gate  = sigmoid(-k * (sus_z - threshold)), the raw multiplicative weight
              BEFORE the keep_min safety fallback.

    sus_override: when given, gate on this precomputed suspicion vector
    instead of recomputing it from `trust`. The runtime uses this to gate on
    the cross-round EMA-smoothed suspicion (see HMPGAERuntime) while keeping
    this function the single place the sigmoid expression lives.

    `reject_soft_weighted` calls this so the production aggregation path and the
    diagnostic both compute sus_z/gate from the exact same expression -- no
    drift.
    """
    if sus_override is not None:
        sus_z = sus_override
    else:
        sus_z = _suspicion_signal(
            trust, gate_signal, gate_rezscore, zscore_mode, zscore_clip
        )
    gate = torch.sigmoid(-soft_reject_k * (sus_z - float(reject_z_threshold)))
    return sus_z, gate


def reject_then_weighted(
    trust: "TrustResult",
    data_sizes: torch.Tensor,
    reject_z_threshold: float = 1.0,
    keep_min: int = 1,
    gate_signal: str = "graph",
    gate_rezscore: bool = True,
    zscore_mode: str = "std",
    zscore_clip: Optional[float] = None,
    sus_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hybrid aggregation: use HMP-GAE trust signals to *detect* attackers,
    then fall back to data-size-weighted FedAvg among trusted clients.

    Rationale: the softmax on trust logits is great at flagging outliers
    but tends to concentrate weight on 1-2 benign clients when benign
    graph_residual values are nearly tied -- this wastes the collaborative
    learning benefit. Splitting detection from weighting gives both:
      1. attacker contributions are zeroed out,
      2. benign contributions aggregate at their natural data-size weights.

    Detection rule: a client i is rejected when its suspicion z-score (see
    `_suspicion_signal`) exceeds `reject_z_threshold` (default 1.0, > 1 sigma).
    `keep_min` guarantees at least k clients are kept even in degenerate cases.
    """
    device = trust.alpha.device
    dtype = trust.alpha.dtype
    N = trust.alpha.numel()

    if sus_override is not None:
        gr_z = sus_override
    else:
        gr_z = _suspicion_signal(
            trust, gate_signal, gate_rezscore, zscore_mode, zscore_clip
        )
    mask = gr_z <= float(reject_z_threshold)

    if int(mask.sum().item()) < max(1, keep_min):
        # Too aggressive: keep keep_min most-trusted by lowest gr_z.
        k = max(1, keep_min)
        idx = torch.topk(-gr_z, k=min(k, N)).indices
        mask = torch.zeros(N, device=device, dtype=torch.bool)
        mask[idx] = True

    ds = data_sizes.to(device=device, dtype=dtype) * mask.to(dtype)
    total = ds.sum()
    if total.item() <= 0:
        # Fallback: uniform over kept clients.
        uniform = mask.to(dtype)
        ds = uniform
        total = ds.sum().clamp(min=1.0)
    return ds / total


def reject_soft_weighted(
    trust: "TrustResult",
    data_sizes: torch.Tensor,
    reject_z_threshold: float = 0.75,
    soft_reject_k: float = 2.0,
    keep_min: int = 1,
    gate_signal: str = "graph",
    gate_rezscore: bool = True,
    zscore_mode: str = "std",
    zscore_clip: Optional[float] = None,
    sus_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft-rejection variant of reject_then_weighted.

    Instead of a binary mask (sus > threshold → weight=0), applies a sigmoid
    gate that smoothly reduces weight for suspicious clients:

        gate_i = sigmoid( -k * (sus_z_i - threshold) )

    where sus_z_i is the suspicion z-score selected by `gate_signal`:
        'graph'    -> trust.graph_residual_z (backward compatible)
        'combined' -> z-score(-trust.s), folding in all enabled signals
                      (graph + recon + semantic + hist)

    Interpretation:
        sus_z_i << threshold  →  gate ≈ 1.0  (clearly benign, full weight)
        sus_z_i == threshold  →  gate = 0.5  (at decision boundary, halved)
        sus_z_i >> threshold  →  gate ≈ 0.0  (clearly attacker, near-zero)

    Final weight = data_size_i * gate_i / sum_j(data_size_j * gate_j)

    Advantages over hard rejection:
    - No cliff at a single threshold value; miscalibration degrades gracefully.
    - Works for any N without re-tuning the threshold as a hard cutoff.
    - The steepness k controls how "hard" the boundary is:
        k=1  very smooth, k=3  near-binary, k=2  recommended default.
    - The threshold parameter controls the midpoint (same scale as before,
      but semantics shift from "reject above" to "sigmoid centre").

    Args:
        trust:               TrustResult from compute_trust_weights.
        data_sizes:          (N,) raw data-size weights (for FedAvg scaling).
        reject_z_threshold:  sigmoid midpoint on the suspicion z-score scale.
        soft_reject_k:       sigmoid steepness (higher = closer to hard reject).
        keep_min:            if all gates fall below 0.1, force top-k by sus_z.
        gate_signal:         which signal drives the gate; see _suspicion_signal.
    """
    device = trust.alpha.device
    dtype = trust.alpha.dtype
    N = trust.alpha.numel()

    gr_z, gate = gate_diagnostics(
        trust, reject_z_threshold, soft_reject_k, gate_signal,
        gate_rezscore=gate_rezscore,
        zscore_mode=zscore_mode,
        zscore_clip=zscore_clip,
        sus_override=sus_override,
    )

    # Safety: if every client's gate is tiny (all look suspicious), fall back
    # to keeping the keep_min least-isolated clients with uniform weight.
    if int((gate > 0.1).sum().item()) < max(1, keep_min):
        k = max(1, keep_min)
        idx = torch.topk(-gr_z, k=min(k, N)).indices
        gate = torch.zeros(N, device=device, dtype=dtype)
        gate[idx] = 1.0

    ds = data_sizes.to(device=device, dtype=dtype) * gate
    total = ds.sum()
    if total.item() <= 0:
        ds = gate
        total = ds.sum().clamp(min=1.0)
    return ds / total
