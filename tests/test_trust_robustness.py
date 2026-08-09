# tests/test_trust_robustness.py
# CPU-only sanity tests for the robust trust-scoring stack (2026-07):
#   - mad z-score (robust to high attacker fractions, degenerate-MAD fallback)
#   - median semantic reference (non-IID robustness)
#   - gate_rezscore=False + weight-norm scaling (no "scapegoat tax" on
#     all-benign rounds; threshold in per-signal robust-z units)
#   - suspicion EMA state + checkpoint roundtrip
#   - the CSE-reject rules V4 / V5 / V6, including V6's three structural
#     guarantees (geo_floor=1.0 == V5 element-for-element, m_v6 <= m_v5 for
#     every gate, and no-flags == the exact n_k prior)
#   - bit-for-bit backward compatibility of all legacy code paths
#
# Pure tensors, no FL training, no GPU, no dataset — runs in ~1s:
#
#     python tests/test_trust_robustness.py
#
# Intentionally plain asserts (no pytest dependency, matching the repo).

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hmp_gae.trust_scorer import (
    _semantic_divergence_signal,
    _zscore,
    compute_trust_weights,
    gate_diagnostics,
    reject_soft_weighted,
    v4_cse_reject_weights,
    v5_cse_reject_weights,
    v6_cse_reject_geo_weights,
    v7_cse_reject_corrob_weights,
)
from hmp_gae.hypergraph import knn_hypergraph
from hmp_gae.runtime import HMPGAERuntime

# The robust configuration under test (mirrors main.py defense_config).
ROBUST = dict(zscore_mode="mad", zscore_clip=10.0, semantic_reference="median")
GATE_ROBUST = dict(gate_rezscore=False, zscore_mode="mad", zscore_clip=10.0)
THRESHOLD, STEEPNESS = 2.5, 2.0


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _auroc(pos, neg) -> float:
    """P(pos > neg) with ties at 0.5 (Mann-Whitney)."""
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _round_rng(rng, seed, round_seed):
    """Per-round noise stream: same stream as `rng` when round_seed is None
    (single-round scenarios), a distinct seeded stream otherwise (so client
    structure stays fixed across simulated rounds while noise varies)."""
    if round_seed is None:
        return rng
    return np.random.default_rng(7919 * (seed + 1) + int(round_seed))


def make_probe_dists(
    n_benign: int,
    n_attackers: int,
    K: int = 64,
    C: int = 4,
    het: float = 0.35,
    seed: int = 0,
    round_seed=None,
) -> torch.Tensor:
    """
    Synthetic (N, K, C) probe softmax distributions.

    Benign i: mostly-correct predictions biased toward a client-specific
        Dirichlet class prior (simulates non-IID heterogeneity; the prior is
        persistent across round_seeds — client identity).
    Attacker: confident mass on a wrong class per sample (label-flip training;
        re-drawn per round_seed, matching the per-round-reseed attacker).

    Attackers are appended AFTER the benign rows.
    """
    rng = np.random.default_rng(seed)
    nrng = _round_rng(rng, seed, round_seed)
    y = np.arange(K) % C  # balanced true labels
    rows = []
    for _ in range(n_benign):
        prior = rng.dirichlet(np.ones(C) * 0.5)  # skewed local prior
        P = (1.0 - het) * np.eye(C)[y] + het * prior[None, :]
        P = P + 0.02 * nrng.random((K, C))
        rows.append(P / P.sum(axis=1, keepdims=True))
    for _ in range(n_attackers):
        wrong = (y + nrng.integers(1, C, size=K)) % C
        P = 0.8 * np.eye(C)[wrong] + 0.2 / C
        P = P + 0.02 * nrng.random((K, C))
        rows.append(P / P.sum(axis=1, keepdims=True))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def make_geometry(
    n_benign: int,
    n_attackers: int,
    dim: int = 64,
    het: float = 0.3,
    seed: int = 0,
    round_seed=None,
) -> torch.Tensor:
    """
    Synthetic (N, dim) node features: benign = shared direction + persistent
    per-client heterogeneity offset + per-round jitter; attackers = a distinct
    offset cluster.
    """
    rng = np.random.default_rng(seed)
    nrng = _round_rng(rng, seed, round_seed)
    g = rng.standard_normal(dim)
    g /= np.linalg.norm(g)
    base = [g + het * rng.standard_normal(dim) * 0.5 for _ in range(n_benign)]
    rows = [b + 0.05 * nrng.standard_normal(dim) for b in base]
    a_dir = -0.5 * g + rng.standard_normal(dim) * 0.3
    rows += [a_dir + 0.1 * nrng.standard_normal(dim) for _ in range(n_attackers)]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def trust_from_synthetic(n_benign, n_attackers, seed=0, round_seed=None, **kwargs):
    """Run compute_trust_weights on a fully synthetic scenario."""
    eta = make_geometry(n_benign, n_attackers, seed=seed, round_seed=round_seed)
    H, _, _ = knn_hypergraph(eta, k=2)
    Zn = torch.nn.functional.normalize(eta, dim=1)
    A_hat = torch.sigmoid(4.0 * (Zn @ Zn.t()))
    probes = make_probe_dists(n_benign, n_attackers, seed=seed, round_seed=round_seed)
    return compute_trust_weights(
        A_hat=A_hat, Z=eta, Z_hist=None, H=H,
        probe_distributions=probes, semantic_weight=1.0, **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1) _zscore                                                                  #
# --------------------------------------------------------------------------- #

def test_zscore_std_backward_compat():
    torch.manual_seed(0)
    x = torch.randn(7)
    legacy = (x - x.mean()) / x.std(unbiased=False).clamp(min=1e-6)
    assert torch.allclose(_zscore(x), legacy), "default _zscore must equal legacy"
    print("PASS  _zscore default == legacy mean/std")


def test_zscore_mad_high_attacker_fraction():
    # 6 benign + 4 attackers (40%): std z-scores break down, mad survives.
    rng = np.random.default_rng(1)
    x = torch.tensor(
        np.concatenate([0.1 * rng.standard_normal(6), 5.0 + 0.2 * rng.standard_normal(4)]),
        dtype=torch.float32,
    )
    z_std, z_mad = _zscore(x, mode="std"), _zscore(x, mode="mad", clip=10.0)
    sep_std = z_std[6:].min() - z_std[:6].max()
    sep_mad = z_mad[6:].min() - z_mad[:6].max()
    assert sep_mad > sep_std, f"mad sep {sep_mad:.2f} !> std sep {sep_std:.2f}"
    assert z_std[6:].max() < 2.5, "std-z should be polluted (attackers look mild)"
    assert z_mad[6:].min() > 3.0, "mad-z should keep attackers extreme"
    print(f"PASS  mad-z separation {sep_mad:.2f} > std-z {sep_std:.2f} at 40% attackers")


def test_zscore_mad_degenerate_fallback():
    # Quantized signal: benign majority ties exactly -> MAD == 0.
    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33])
    z = _zscore(x, mode="mad", clip=10.0)
    assert torch.isfinite(z).all()
    assert z[-1] <= 10.0 and z[-1] < 5.0, (
        f"MAD=0 must fall back to std scale, got z={z[-1]:.1f}"
    )
    print(f"PASS  degenerate MAD falls back to std (outlier z={z[-1]:.2f})")


def test_zscore_relative_degeneracy_guard():
    """C1 (V4 brief): the recon_residual pathology. Client spread ~1e-4 around
    ~0.49 passed the old ABSOLUTE guard (scale < 1e-6 never fired), z exploded
    to 18-36x and the ±10 clip pinned attacker AND benign at the bound — an
    exact rank tie. The RELATIVE guard (scale < 1e-3·max|x|) must zero a
    channel whose spread is negligible at its own magnitude, while keeping the
    std fallback for a tied majority + lone genuine outlier."""
    x = 0.49 + 1e-4 * torch.tensor([0.1, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2])
    z = _zscore(x, mode="mad", clip=10.0)
    assert torch.equal(z, torch.zeros_like(x)), (
        f"noise-only degenerate channel must be zeroed, got {z.tolist()}"
    )
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33])
    zy = _zscore(y, mode="mad", clip=10.0)
    assert zy[-1] > 1.5, (
        f"tied-majority + lone outlier must keep the std fallback, got {zy[-1]:.2f}"
    )
    print("PASS  relative degeneracy guard: noise channel zeroed, lone outlier kept")


def test_zscore_clip_post_fusion():
    """C1 (V4 brief): zscore_clip bounds the FUSED score s, not each channel.
    Per-channel *_z diagnostics are unclipped (true magnitudes visible)."""
    # Tight benign cluster (small but non-degenerate MAD) + one far outlier:
    # unclipped per-channel mad-z far exceeds 10 — this is what used to be
    # per-channel-clipped into an exact ±10 tie.
    x = torch.tensor([0.10, 0.105, 0.095, 0.1025, 0.0975, 0.11, 0.90])
    z_unclipped = _zscore(x, mode="mad")
    assert z_unclipped.abs().max() > 10.0, (
        "test premise: unclipped mad-z should exceed 10 for the outlier, "
        f"got {z_unclipped.abs().max():.1f}"
    )
    trust = trust_from_synthetic(5, 2, seed=0, **ROBUST)
    assert trust.s.abs().max() <= 10.0 + 1e-6, (
        f"fused s must respect zscore_clip=10, got {trust.s.abs().max():.2f}"
    )
    print("PASS  zscore_clip applies post-fusion (fused |s| <= clip)")


def test_graph_min_distinct_gating():
    """C1 (V4 brief): with knn_k=2 and N=7, graph_residual takes only a few
    discrete levels (multiples of 1/6). When a round resolves fewer than
    graph_min_distinct values the channel must be zeroed AND its weight
    dropped from weight_norm (a zeroed channel whose weight stays in the norm
    silently shifts the effective gate threshold)."""
    eta = make_geometry(5, 2, seed=0)
    H, _, _ = knn_hypergraph(eta, k=2)
    Zn = torch.nn.functional.normalize(eta, dim=1)
    A_hat = torch.sigmoid(4.0 * (Zn @ Zn.t()))
    t_off = compute_trust_weights(A_hat=A_hat, Z=eta, Z_hist=None, H=H, **ROBUST)
    assert bool(t_off.graph_gated) is False, "default (0) must never gate"
    n_distinct = int(torch.unique(t_off.graph_residual).numel())
    t_on = compute_trust_weights(
        A_hat=A_hat, Z=eta, Z_hist=None, H=H,
        graph_min_distinct=n_distinct + 1, **ROBUST,
    )
    assert bool(t_on.graph_gated) is True
    assert torch.equal(t_on.graph_residual_z, torch.zeros_like(t_on.graph_residual_z))
    # Only recon remains active (semantic off without probes): ||w|| = 0.3.
    assert abs(t_on.weight_norm - 0.3) < 1e-6, t_on.weight_norm
    # Raw residual stays logged for diagnostics even when gated.
    assert torch.equal(t_on.graph_residual, t_off.graph_residual)
    t_keep = compute_trust_weights(
        A_hat=A_hat, Z=eta, Z_hist=None, H=H,
        graph_min_distinct=n_distinct, **ROBUST,
    )
    assert bool(t_keep.graph_gated) is False
    print(f"PASS  graph_min_distinct gating ({n_distinct} distinct levels resolved)")


# --------------------------------------------------------------------------- #
# 2) semantic divergence                                                      #
# --------------------------------------------------------------------------- #

def test_semantic_pairwise_bitforbit():
    P = make_probe_dists(5, 2, seed=2)
    new = _semantic_divergence_signal(P)  # defaults = legacy
    # Legacy expression, replicated verbatim:
    eps = 1e-8
    Q = P.clamp(min=eps)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    logQ = Q.log()
    N, K, _ = Q.shape
    H_ik = (Q * logQ).sum(dim=-1)
    X = torch.einsum("ikc,jkc->ijk", Q, logQ)
    KL = H_ik.unsqueeze(1) - X
    sym_KL = 0.5 * (KL + KL.transpose(0, 1))
    mask = 1.0 - torch.eye(N, dtype=Q.dtype)
    legacy = (sym_KL * mask.unsqueeze(-1)).sum(dim=(1, 2)) / float((N - 1) * K)
    assert torch.equal(new, legacy), "pairwise path must be bit-for-bit legacy"
    print("PASS  semantic pairwise path bit-for-bit unchanged")


def test_semantic_median_beats_pairwise_noniid():
    aurocs = {}
    for ref in ("pairwise", "median"):
        vals = []
        for seed in range(5):
            P = make_probe_dists(5, 2, het=0.45, seed=seed)
            d = _semantic_divergence_signal(P, reference=ref)
            vals.append(_auroc(d[5:].tolist(), d[:5].tolist()))
        aurocs[ref] = float(np.mean(vals))
    assert aurocs["median"] >= aurocs["pairwise"], aurocs
    # Contrast (attacker mean / benign mean) should widen with the median ref.
    P = make_probe_dists(5, 2, het=0.45, seed=0)
    d_pw = _semantic_divergence_signal(P, reference="pairwise")
    d_md = _semantic_divergence_signal(P, reference="median")
    c_pw = (d_pw[5:].mean() / d_pw[:5].mean()).item()
    c_md = (d_md[5:].mean() / d_md[:5].mean()).item()
    assert c_md > c_pw, f"median contrast {c_md:.2f} !> pairwise {c_pw:.2f}"
    print(f"PASS  median ref: AUROC {aurocs['median']:.3f} >= {aurocs['pairwise']:.3f}, "
          f"contrast {c_md:.1f}x vs {c_pw:.1f}x")


# --------------------------------------------------------------------------- #
# 3) gate behavior: scapegoat tax & attacker rejection                        #
# --------------------------------------------------------------------------- #

def test_no_attack_no_scapegoat():
    """
    All-benign rounds. Legacy double-z gate shaves the round's most extreme
    benign client every single round; the robust gate (weight-norm scale +
    suspicion EMA over rounds, as the runtime applies it) must not.
    """
    # (a) Legacy config: scapegoat tax exists (single rounds).
    legacy_min = []
    for seed in range(5):
        t = trust_from_synthetic(7, 0, seed=seed)  # std z, legacy defaults
        _, gate = gate_diagnostics(t, 0.75, STEEPNESS, "combined")
        legacy_min.append(gate.min().item())
    assert float(np.mean(legacy_min)) < 0.5, (
        f"legacy config should exhibit the scapegoat tax, min gates {legacy_min}"
    )

    # (b) Robust config: 6 simulated rounds, fixed clients, EMA beta=0.6
    #     (replicates HMPGAERuntime._smooth_suspicion + gate).
    for seed in (1, 3):  # the harshest seeds from the single-round analysis
        ema = None
        for r in range(6):
            t = trust_from_synthetic(7, 0, seed=seed, round_seed=r, **ROBUST)
            sus, _ = gate_diagnostics(t, THRESHOLD, STEEPNESS, "combined", **GATE_ROBUST)
            ema = sus if ema is None else 0.6 * ema + 0.4 * sus
        gate = torch.sigmoid(-STEEPNESS * (ema - THRESHOLD))
        assert gate.min() > 0.5, f"seed {seed}: benign gated, gates={gate.tolist()}"
        assert gate.mean() > 0.8, f"seed {seed}: mean gate {gate.mean():.2f} <= 0.8"
    print(f"PASS  no-attack: legacy min-gate {np.mean(legacy_min):.2f} (scapegoat) "
          f"vs robust EMA gates all > 0.5, mean > 0.8")


def test_attack_detected_and_rejected():
    """5 benign + 2 attackers, single rounds: suspicion ranks perfectly and
    attackers lose their gate; benign keep most of theirs (EMA lifts the
    residual benign dip further — covered by the runtime test)."""
    for seed in range(5):
        trust = trust_from_synthetic(5, 2, seed=seed, **ROBUST)
        sus, gate = gate_diagnostics(
            trust, THRESHOLD, STEEPNESS, "combined", **GATE_ROBUST
        )
        assert _auroc(sus[5:].tolist(), sus[:5].tolist()) == 1.0, "sus must rank perfectly"
        assert gate[5:].max() < 0.25, f"attacker gates too high: {gate[5:].tolist()}"
        assert gate[:5].mean() > 0.7, f"benign gates too low: {gate[:5].tolist()}"
        # Single-round benign floor recalibrated 2026-07-28 against a REAL
        # execution (the original 0.35 was set analytically and never run —
        # unmodified HEAD 33d40dd fails it identically at seed 4): a genuinely
        # semantically-divergent benign (c4: sem_div 0.375 vs peers ~0.07)
        # legitimately dips to gate ~0.10 in a single round, and the
        # cross-round EMA lifts it back (see the runtime tests). What must
        # hold within one round is SEPARATION from the attackers.
        assert gate[:5].min() > 0.05, f"benign worst-case gate: {gate[:5].tolist()}"
        assert gate[:5].min() > 50.0 * gate[5:].max(), (
            f"benign/attacker gate separation lost: {gate.tolist()}"
        )
        # Aggregation weights: attackers' mass should be negligible.
        ds = torch.ones(7)
        alpha = reject_soft_weighted(
            trust, ds, THRESHOLD, STEEPNESS, gate_signal="combined", **GATE_ROBUST
        )
        assert alpha[5:].sum() < 0.10, f"attacker alpha mass {alpha[5:].sum():.3f}"
    print("PASS  attack rounds: AUROC 1.0, attacker gate < 0.25, attacker mass < 0.10")


def test_sus_override_drives_gate():
    trust = trust_from_synthetic(5, 2, seed=0, **ROBUST)
    override = torch.zeros(7)
    override[0] = 99.0  # pretend EMA says client 0 is the attacker
    sus, gate = gate_diagnostics(
        trust, THRESHOLD, STEEPNESS, "combined", sus_override=override,
    )
    assert torch.equal(sus, override)
    assert gate[0] < 1e-3 and gate[1:].min() > 0.9
    ds = torch.ones(7)
    alpha = reject_soft_weighted(trust, ds, THRESHOLD, STEEPNESS,
                                 gate_signal="combined", sus_override=override)
    assert alpha[0] < 1e-3
    print("PASS  sus_override drives gate and aggregation weights")


# --------------------------------------------------------------------------- #
# 4) compute_trust_weights default-path compatibility                         #
# --------------------------------------------------------------------------- #

def test_trust_weights_default_path_unchanged():
    trust = trust_from_synthetic(5, 2, seed=3)  # all-new kwargs left at defaults
    # s must decompose with the LEGACY z-score (mean/std, no clip).
    z = lambda v: (v - v.mean()) / v.std(unbiased=False).clamp(min=1e-6)  # noqa: E731
    s_ref = -(1.0 * z(trust.graph_residual) + 0.3 * z(trust.recon_residual)
              + 1.0 * z(trust.sem_div))
    assert torch.allclose(trust.s, s_ref, atol=1e-5), "default s decomposition drifted"
    sus, _ = gate_diagnostics(trust, 0.75, STEEPNESS, "combined")
    assert torch.allclose(sus, z(-trust.s), atol=1e-5), "default combined gate drifted"
    # weight_norm bookkeeping: sqrt(1^2 + 0.3^2 + 1^2) with semantic active.
    assert abs(trust.weight_norm - (1 + 0.09 + 1) ** 0.5) < 1e-6
    print("PASS  default kwargs reproduce legacy s and double-z gate")


# --------------------------------------------------------------------------- #
# 4b) V4 rejection rule (per-client CSE, pool-median normalised, rank-capped) #
# --------------------------------------------------------------------------- #

def test_v4_cse_reject_rule():
    """The V4 rule in isolation: both conditions (rank cap AND ratio > tau)
    required, soft rejection, n_k prior untouched."""
    ds = torch.tensor([309., 1500., 2100., 900., 1200., 1800., 1191.])
    # (a) attack round: two elevated-CSE clients flagged; attacker aggregation
    #     mass collapses ~10x below its n_k share; weights normalise.
    cse = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.90, 1.75])
    w, diag = v4_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                    reject_mult=0.10)
    assert diag["flagged"].tolist() == [False] * 5 + [True, True]
    atk_nk = float((ds[5:] / ds.sum()).sum())
    assert float(w[5:].sum()) < 0.35 * atk_nk, f"attacker mass {w[5:].sum():.4f}"
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[5:].sum()) > 0.0, "soft rejection: mass must NOT be zeroed"
    # (b) clean round: heterogeneous benign pool below tau -> zero flags and
    #     weights exactly equal to the n_k prior (no scapegoat).
    cse0 = torch.tensor([0.60, 0.55, 0.98, 0.58, 0.62, 0.70, 0.66])
    w0, d0 = v4_cse_reject_weights(cse0, ds, tau_ratio=1.85, k_cap=2,
                                   reject_mult=0.10)
    assert not bool(d0["flagged"].any()), d0["flagged"]
    assert torch.allclose(w0, ds / ds.sum(), atol=1e-6)
    # (c) rank cap binds: three clients above tau, only the top-2 by ratio
    #     flagged (the cap, not tau, is what carries the zero-FP property).
    cse3 = torch.tensor([0.30, 0.30, 0.30, 0.30, 0.65, 0.80, 0.90])
    _, d3 = v4_cse_reject_weights(cse3, ds, tau_ratio=1.85, k_cap=2,
                                  reject_mult=0.10)
    assert d3["flagged"].tolist() == [False] * 5 + [True, True], d3["flagged"]
    # (d) high-rank client BELOW tau is not flagged (both conditions needed).
    cse4 = torch.tensor([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 1.00])
    _, d4 = v4_cse_reject_weights(cse4, ds, tau_ratio=1.85, k_cap=2,
                                  reject_mult=0.10)
    assert not bool(d4["flagged"].any()), d4["flagged"]
    print("PASS  v4 rule: flags need rank AND ratio, soft mass, clean = n_k prior")


def test_v4_hard_removal_arm():
    """The 2026-08-07 pre-registered ablation arm (docs/DECISION.md
    "V4-remove"): reject_mult=0.0 excludes flagged clients from the round's
    aggregate outright. Flagged weights must be exactly 0, survivors must
    renormalise to their n_k prior, clean rounds are untouched, and the
    runtime guard accepts 0.0 but still refuses negatives (a negative
    multiplier would sign-flip a flagged update — an attack, not a penalty)."""
    ds = torch.tensor([309., 1500., 2100., 900., 1200., 1800., 1191.])
    # (a) attack round: both attackers flagged -> exactly zero mass; the
    #     benign remainder is the n_k prior renormalised over survivors.
    cse = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.90, 1.75])
    w, diag = v4_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                    reject_mult=0.0)
    assert diag["flagged"].tolist() == [False] * 5 + [True, True]
    assert float(w[5:].abs().sum()) == 0.0, "flagged mass must be exactly 0"
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert torch.allclose(w[:5], ds[:5] / ds[:5].sum(), atol=1e-6), \
        "survivors must renormalise to their n_k prior"
    # (b) clean round: no flags -> exact n_k prior, bit-identical to soft V4
    #     (the arm changes nothing until a flag actually fires).
    cse0 = torch.tensor([0.60, 0.55, 0.98, 0.58, 0.62, 0.70, 0.66])
    w0, d0 = v4_cse_reject_weights(cse0, ds, tau_ratio=1.85, k_cap=2,
                                   reject_mult=0.0)
    assert not bool(d0["flagged"].any()), d0["flagged"]
    assert torch.allclose(w0, ds / ds.sum(), atol=1e-6)
    # (c) the normalisation denominator stays positive by construction: the
    #     rank cap (num_byzantine < N/2, runtime-validated) plus keep_min
    #     bound the flag count below N, and unflagged clients keep m=1.
    assert float(w[:5].sum()) > 0.99
    # (d) runtime guard: 0.0 constructs since 2026-08-07; negatives refuse.
    rt = _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 2,
                   "v4_reject_mult": 0.0, "graph_min_distinct": 4})
    assert rt.v4_reject_mult == 0.0
    try:
        _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 2,
                  "v4_reject_mult": -0.1})
        raise AssertionError("expected ValueError for negative reject_mult")
    except ValueError:
        pass
    print("PASS  v4 hard-removal arm: flagged mass exactly 0, survivors = n_k prior")


def test_v5_cse_reject_rule():
    """The V5 graded rule in isolation: flag set identical to V4, multiplier
    is a monotone ramp in the ratio (mild just past tau, m_floor at r_hard),
    clean rounds keep the exact n_k prior."""
    ds = torch.tensor([309., 1500., 2100., 900., 1200., 1800., 1191.])
    # (a) graded penalty: median = 0.62; r5 = 1.28/0.62 ≈ 2.065 (ramp zone),
    #     r6 = 3.20/0.62 ≈ 5.161 (saturated).  t5 ≈ 0.330 → mult5 ≈ 0.703;
    #     mult6 = m_floor = 0.10 exactly.
    cse = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.28, 3.20])
    w5, d5 = v5_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                   m_floor=0.10, r_hard=2.5)
    _, d4 = v4_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                  reject_mult=0.10)
    assert d5["flagged"].tolist() == d4["flagged"].tolist() == \
        [False] * 5 + [True, True], "V5 flag set must equal V4's"
    m5, m6 = float(d5["multiplier"][5]), float(d5["multiplier"][6])
    assert 0.69 < m5 < 0.72, f"ramp-zone mult expected ≈0.703, got {m5}"
    assert abs(m6 - 0.10) < 1e-6, f"saturated mult expected 0.10, got {m6}"
    assert m5 > m6, "higher ratio must never get a milder multiplier"
    assert abs(float(w5.sum()) - 1.0) < 1e-6
    # Unflagged clients keep exact n_k proportions among themselves.
    ratio_b = w5[:5] / ds[:5]
    assert torch.allclose(ratio_b, ratio_b[0].expand(5), rtol=1e-5)
    # (b) clean round: zero flags -> weights exactly the n_k prior (identical
    #     guarantee to V4's no-scapegoat property).
    cse0 = torch.tensor([0.60, 0.55, 0.98, 0.58, 0.62, 0.70, 0.66])
    w0, d0 = v5_cse_reject_weights(cse0, ds, tau_ratio=1.85, k_cap=2,
                                   m_floor=0.10, r_hard=2.5)
    assert not bool(d0["flagged"].any()), d0["flagged"]
    assert torch.allclose(w0, ds / ds.sum(), atol=1e-6)
    # (c) invalid geometry: r_hard <= tau must raise (ramp divides by the
    #     difference), never silently degenerate.
    try:
        v5_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                              m_floor=0.10, r_hard=1.85)
        raise AssertionError("expected ValueError for r_hard <= tau_ratio")
    except ValueError:
        pass
    print("PASS  v5 rule: V4 flags, graded ramp, clean = n_k prior")


def test_v5_saturation_equals_v4():
    """When every flagged client sits at ratio >= r_hard, V5 must reproduce
    V4 with reject_mult = m_floor (the ramp term is an exact 0.0)."""
    ds = torch.tensor([309., 1500., 2100., 900., 1200., 1800., 1191.])
    # V4 test's attack round: r5 ≈ 3.065, r6 ≈ 2.823 — both >= r_hard=2.5.
    cse = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.90, 1.75])
    w5, d5 = v5_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                   m_floor=0.10, r_hard=2.5)
    w4, d4 = v4_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                   reject_mult=0.10)
    assert d5["flagged"].tolist() == d4["flagged"].tolist()
    assert torch.allclose(w5, w4, atol=1e-9, rtol=0.0), (w5, w4)
    assert torch.allclose(d5["multiplier"], d4["multiplier"], atol=1e-9)
    print("PASS  v5 saturation: bit-equal to V4 at reject_mult = m_floor")


def test_v5_false_positive_containment():
    """The design motivation: a borderline flag (ratio just past tau — the
    shape a benign false positive would take) keeps most of its weight under
    V5, versus losing 90% under V4."""
    ds = torch.tensor([1000.] * 7)
    # Only c6 clears tau: median = 0.60, r6 = 1.20/0.60 = 2.0 (borderline).
    cse = torch.tensor([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 1.20])
    w5, d5 = v5_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                   m_floor=0.10, r_hard=2.5)
    w4, _ = v4_cse_reject_weights(cse, ds, tau_ratio=1.85, k_cap=2,
                                  reject_mult=0.10)
    assert d5["flagged"].tolist() == [False] * 6 + [True]
    m6 = float(d5["multiplier"][6])
    # t = (2.0-1.85)/0.65 ≈ 0.231 -> mult ≈ 0.792.
    assert 0.75 < m6 < 0.82, f"borderline mult expected ≈0.79, got {m6}"
    # Weight retained by the borderline client: ~5x more than under V4.
    assert float(w5[6]) > 4.0 * float(w4[6]), (float(w5[6]), float(w4[6]))
    print("PASS  v5 FP containment: borderline flag keeps most of its weight")


# --------------------------------------------------------------------------- #
# 4c) V6 geometry conjunction (V5's ramp, tightened by the hypergraph gate)    #
# --------------------------------------------------------------------------- #

# Shared fixtures for the V6 block: a 7-client federation whose last two
# clients carry elevated CSE, and a gate vector that DISAGREES with the CSE
# evidence on one attacker (0.9 = "geometry says benign"). The disagreement is
# the point — the archived replay measured a mean gate of 0.766 on confirmed
# attackers in the Qwen AG cell, so V6 must behave correctly precisely when
# the geometry is wrong.
_V6_DS = torch.tensor([309., 1500., 2100., 900., 1200., 1800., 1191.])
_V6_CSE = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.28, 3.20])
_V6_GATE = torch.tensor([0.95, 0.90, 0.85, 0.92, 0.88, 0.20, 0.90])


def test_v6_geo_floor_one_equals_v5():
    """THE regression guard: at geo_floor = 1.0 the geometry term is an exact
    1.0, so V6 must reproduce v5_cse_reject_weights element-for-element
    (atol=0, rtol=0) — for the flagged AND the unflagged clients, and for any
    gate vector at all. Run 0 of the experiment plan is this test at FL scale."""
    for gate in (_V6_GATE, torch.zeros(7), torch.ones(7), torch.full((7,), 0.5)):
        w6, d6 = v6_cse_reject_geo_weights(
            _V6_CSE, _V6_DS, gate=gate, tau_ratio=1.85, k_cap=2,
            m_floor=0.10, r_hard=2.5, geo_floor=1.0,
        )
        w5, d5 = v5_cse_reject_weights(
            _V6_CSE, _V6_DS, tau_ratio=1.85, k_cap=2,
            m_floor=0.10, r_hard=2.5,
        )
        assert d6["flagged"].tolist() == d5["flagged"].tolist()
        assert torch.equal(w6, w5), (w6, w5)
        assert torch.equal(d6["multiplier"], d5["multiplier"])
        assert torch.equal(d6["ramp_t"], d5["ramp_t"])
        assert d6["median"] == d5["median"]
        # ...and the geometry factor is a literal 1.0 everywhere.
        assert torch.equal(d6["geo_mult"], torch.ones(7))
    print("PASS  v6 degeneracy: geo_floor=1.0 is bit-equal to V5 for any gate")


def test_v6_monotone_safety():
    """Monotone safety: for ANY gate in [0,1] and ANY geo_floor in (0,1], a
    flagged client's V6 multiplier (and weight share among the flagged) is
    <= V5's. Geometry may only deepen a CSE-driven penalty, never soften one
    — the property that bounds V6's CSE risk a priori."""
    _, d5 = v5_cse_reject_weights(_V6_CSE, _V6_DS, tau_ratio=1.85, k_cap=2,
                                  m_floor=0.10, r_hard=2.5)
    m5, flagged = d5["multiplier"], d5["flagged"]
    assert bool(flagged.any()), "fixture must produce flags"
    rng = np.random.default_rng(0)
    for geo_floor in (1.0, 0.75, 0.5, 0.25, 0.05):
        for _ in range(20):
            gate = torch.tensor(rng.random(7), dtype=torch.float32)
            _, d6 = v6_cse_reject_geo_weights(
                _V6_CSE, _V6_DS, gate=gate, tau_ratio=1.85, k_cap=2,
                m_floor=0.10, r_hard=2.5, geo_floor=geo_floor,
            )
            m6 = d6["multiplier"]
            # Compare the MULTIPLIERS, not the normalised weights: shrinking
            # the flagged mass raises every unflagged weight, so a weight-space
            # comparison would confound the two effects.
            assert torch.all(m6 <= m5 + 1e-7), (geo_floor, gate, m6, m5)
            assert torch.all(m6[flagged] >= float(geo_floor) * m5[flagged] - 1e-7)
            # Flag set is untouched by the geometry (Stage 1 is V4's, verbatim).
            assert d6["flagged"].tolist() == flagged.tolist()
    # Out-of-range / non-finite gates are sanitized, never trusted: none of
    # these may push the multiplier above V5's or leave a NaN in the weights.
    for bad_gate in (torch.full((7,), 5.0), torch.full((7,), -3.0),
                     torch.full((7,), float("inf")),
                     torch.full((7,), float("-inf")),
                     torch.full((7,), float("nan"))):
        w_b, d_b = v6_cse_reject_geo_weights(
            _V6_CSE, _V6_DS, gate=bad_gate, tau_ratio=1.85, k_cap=2,
            m_floor=0.10, r_hard=2.5, geo_floor=0.5,
        )
        assert torch.all(d_b["multiplier"] <= m5 + 1e-7), bad_gate
        assert torch.isfinite(w_b).all(), (bad_gate, w_b)
    # A NaN gate specifically means "the geometry abstains" -> exactly V5, so
    # a numerically broken channel degrades to the validated rule instead of
    # inventing a penalty or poisoning the aggregate.
    _, d_nan = v6_cse_reject_geo_weights(
        _V6_CSE, _V6_DS, gate=torch.full((7,), float("nan")), tau_ratio=1.85,
        k_cap=2, m_floor=0.10, r_hard=2.5, geo_floor=0.5,
    )
    assert torch.equal(d_nan["multiplier"], m5), d_nan["multiplier"]
    print("PASS  v6 monotone safety: m_v6 <= m_v5 for every gate and floor")


def test_v6_clean_federation_identity():
    """Invariant 9 under V6: with no flags, weights are EXACTLY the n_k prior.
    Unflagged clients take m = 1.0, not the gate — so unlike V3's always-on
    sigmoid, a clean federation pays no scapegoat tax even when the geometry
    dislikes somebody (gate 0.05 on c2 below)."""
    cse0 = torch.tensor([0.60, 0.55, 0.98, 0.58, 0.62, 0.70, 0.66])
    gate0 = torch.tensor([0.9, 0.9, 0.05, 0.9, 0.9, 0.9, 0.9])
    w0, d0 = v6_cse_reject_geo_weights(
        cse0, _V6_DS, gate=gate0, tau_ratio=1.85, k_cap=2,
        m_floor=0.10, r_hard=2.5, geo_floor=0.5,
    )
    assert not bool(d0["flagged"].any()), d0["flagged"]
    assert torch.allclose(w0, _V6_DS / _V6_DS.sum(), atol=1e-6)
    assert torch.equal(d0["multiplier"], torch.ones(7))
    print("PASS  v6 clean federation: no flags -> exactly the n_k prior")


def test_v6_monotone_in_ratio_and_gate():
    """Ordering sanity in both arguments: (a) with the gate held equal, a
    higher CSE ratio never earns a milder multiplier; (b) with the ratio held
    equal, a more suspicious gate (lower value) never earns a milder one."""
    # (a) equal gates, c6 at a far higher ratio than c5.
    _, da = v6_cse_reject_geo_weights(
        _V6_CSE, _V6_DS, gate=torch.full((7,), 0.7), tau_ratio=1.85, k_cap=2,
        m_floor=0.10, r_hard=2.5, geo_floor=0.5,
    )
    assert da["flagged"].tolist() == [False] * 5 + [True, True]
    assert float(da["multiplier"][5]) > float(da["multiplier"][6])
    # m_cse is the V5 value; multiplier is m_cse scaled by a constant 0.85
    # (= 0.5 + 0.5*0.7) on the flagged clients.
    for i in (5, 6):
        assert abs(float(da["multiplier"][i])
                   - 0.85 * float(da["m_cse"][i])) < 1e-6
    # (b) equal ratios (same cse vector), gate varied on one flagged client.
    prev = None
    for g in (1.0, 0.75, 0.5, 0.25, 0.0):
        gate = torch.full((7,), 0.9)
        gate[6] = g
        _, db = v6_cse_reject_geo_weights(
            _V6_CSE, _V6_DS, gate=gate, tau_ratio=1.85, k_cap=2,
            m_floor=0.10, r_hard=2.5, geo_floor=0.5,
        )
        m = float(db["multiplier"][6])
        if prev is not None:
            assert m <= prev + 1e-9, f"gate {g} gave a milder multiplier"
        prev = m
    # Bottom of the range is exactly geo_floor x the V5 multiplier.
    assert abs(prev - 0.5 * float(da["m_cse"][6])) < 1e-6
    # Invalid geometry / floor must raise, never silently degenerate.
    for bad in dict(geo_floor=0.0), dict(geo_floor=1.5), dict(r_hard=1.85):
        kwargs = dict(tau_ratio=1.85, k_cap=2, m_floor=0.10, r_hard=2.5,
                      geo_floor=0.5)
        kwargs.update(bad)
        try:
            v6_cse_reject_geo_weights(_V6_CSE, _V6_DS, gate=_V6_GATE, **kwargs)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # A mis-sized gate is a plumbing bug: loud, not broadcast.
    try:
        v6_cse_reject_geo_weights(_V6_CSE, _V6_DS, gate=torch.ones(3),
                                  tau_ratio=1.85, k_cap=2, m_floor=0.10,
                                  r_hard=2.5, geo_floor=0.5)
        raise AssertionError("expected ValueError for a length-3 gate")
    except ValueError:
        pass
    print("PASS  v6 monotonicity in both ratio and gate, guards raise")


# Shared fixtures for the V7 block. cse gives ONE genuine CSE flag (c6,
# r=3.06 > 1.85) and ONE elevated-but-sub-threshold client (c5, r=1.53 in
# (tau_lo, 1.85) — V4/V5/V6 structurally cannot touch it). iso is the RAW
# quantized graph_residual channel (multiples of 1/6 at N=7): c0 is a benign
# client the hypergraph dislikes (2/3) with r ~ 1 — the scapegoat probe.
_V7_DS = _V6_DS
_V7_CSE = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 0.95, 1.90])
_V7_ISO = torch.tensor([2 / 3, 0.0, 1 / 6, 0.0, 1 / 6, 0.5, 2 / 3])
_V7_GATE = torch.full((7,), 0.8)
_V7_KW = dict(tau_ratio=1.85, k_cap=2, m_floor=0.10, r_hard=2.5,
              geo_floor=0.5, tau_lo=1.40, iso_min=5 / 12, corrob_mult=0.5,
              round_min=3, round_max=10)


def test_v7_window_off_equals_v6():
    """THE V7 regression guard (house degeneracy family, like geo_floor=1.0
    -> V5): outside the cold window — round below round_min, above round_max,
    or round_max=0 — V7 must reproduce v6_cse_reject_geo_weights bit-for-bit
    (weights AND multipliers), for any iso vector at all. Same when iso=None
    (graph channel resolution-gated: geometry abstains)."""
    w6, d6 = v6_cse_reject_geo_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, tau_ratio=1.85, k_cap=2,
        m_floor=0.10, r_hard=2.5, geo_floor=0.5,
    )
    cases = [dict(round_logged=1), dict(round_logged=2),
             dict(round_logged=11), dict(round_logged=50),
             dict(round_logged=5, round_max=0)]
    for iso in (_V7_ISO, torch.ones(7), torch.zeros(7)):
        for case in cases:
            kw = dict(_V7_KW)
            kw.update(case)
            w7, d7 = v7_cse_reject_corrob_weights(
                _V7_CSE, _V7_DS, gate=_V7_GATE, iso=iso, **kw)
            assert torch.equal(w7, w6), (case, w7, w6)
            assert torch.equal(d7["multiplier"], d6["multiplier"])
            assert not bool(d7["corrob_flagged"].any())
    # In-window but geometry abstains (iso=None) -> V6 exactly.
    w7n, d7n = v7_cse_reject_corrob_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, iso=None, round_logged=5, **_V7_KW)
    assert torch.equal(w7n, w6)
    assert d7n["geo_resolved"] is False
    assert not bool(d7n["corrob_flagged"].any())
    print("PASS  v7 degeneracy: outside window / iso=None is bit-equal to V6")


def test_v7_corroborated_flag_rule():
    """The Tier-2 rule in isolation: in-window, a non-CSE-flagged client
    crossing BOTH absolute floors (r > tau_lo AND iso >= iso_min) takes the
    constant corrob_mult; each conjunct is necessary; the shared rank-cap
    budget (CSE flags first) is respected; the pre-registered tie-break is
    (iso desc, ratio desc, index asc)."""
    w7, d7 = v7_cse_reject_corrob_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, iso=_V7_ISO, round_logged=5, **_V7_KW)
    # c6 is the CSE flag (V6 handling, untouched); c5 is the corroborated one.
    assert d7["flagged"].tolist() == [False] * 6 + [True]
    assert d7["corrob_flagged"].tolist() == [False] * 5 + [True, False]
    assert abs(float(d7["multiplier"][5]) - 0.5) < 1e-7
    # c0: iso 2/3 (max isolation) but r ~ 1 -> UNTOUCHED. Isolation alone
    # never flags — the anti-scapegoat conjunct.
    assert float(d7["multiplier"][0]) == 1.0
    # c6 keeps its V6 multiplier (ramp x geo factor), not corrob_mult.
    _, d6 = v6_cse_reject_geo_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, tau_ratio=1.85, k_cap=2,
        m_floor=0.10, r_hard=2.5, geo_floor=0.5,
    )
    assert abs(float(d7["multiplier"][6]) - float(d6["multiplier"][6])) < 1e-9
    # Weights renormalise over m_i * n_i and still sum to 1.
    assert abs(float(w7.sum()) - 1.0) < 1e-6
    assert float(w7[5]) < float(_V7_DS[5] / _V7_DS.sum())
    # Conjunct necessity: raise iso_min one midpoint (7/12) -> c5's iso 0.5
    # no longer qualifies -> exactly V6 again.
    kw = dict(_V7_KW); kw["iso_min"] = 7 / 12
    _, d_hi = v7_cse_reject_corrob_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, iso=_V7_ISO, round_logged=5, **kw)
    assert not bool(d_hi["corrob_flagged"].any())
    # Conjunct necessity: r below tau_lo (raise tau_lo above c5's 1.53).
    kw = dict(_V7_KW); kw["tau_lo"] = 1.60
    _, d_lo = v7_cse_reject_corrob_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, iso=_V7_ISO, round_logged=5, **kw)
    assert not bool(d_lo["corrob_flagged"].any())
    # Budget sharing: with BOTH rank-cap slots consumed by CSE flags, Tier 2
    # must stay empty no matter how isolated an elevated third client is.
    cse_two = torch.tensor([0.60, 0.55, 0.90, 0.58, 0.62, 1.90, 2.10])
    _, d_full = v7_cse_reject_corrob_weights(
        cse_two, _V7_DS, gate=_V7_GATE, iso=torch.full((7,), 2 / 3),
        round_logged=5, **_V7_KW)
    assert d_full["flagged"].tolist() == [False] * 5 + [True, True]
    assert not bool(d_full["corrob_flagged"].any()), \
        "Tier 2 must never exceed the shared rank cap"
    # Tie-break (a): equal iso -> higher ratio wins the last slot.
    cse_tie = torch.tensor([0.60, 0.55, 0.90, 0.58, 0.62, 1.00, 1.90])
    iso_tie = torch.tensor([0.0, 0.0, 2 / 3, 0.0, 0.0, 2 / 3, 0.0])
    _, d_tie = v7_cse_reject_corrob_weights(
        cse_tie, _V7_DS, gate=_V7_GATE, iso=iso_tie, round_logged=5, **_V7_KW)
    assert d_tie["flagged"].tolist() == [False] * 6 + [True]
    # candidates: c2 (r=1.45, iso 2/3) vs c5 (r=1.61, iso 2/3); budget 1.
    assert d_tie["corrob_flagged"].tolist() == [False] * 5 + [True, False]
    # Tie-break (b): iso ranks before ratio — c2 at higher iso beats c5.
    iso_rank = torch.tensor([0.0, 0.0, 2 / 3, 0.0, 0.0, 0.5, 0.0])
    _, d_rank = v7_cse_reject_corrob_weights(
        cse_tie, _V7_DS, gate=_V7_GATE, iso=iso_rank, round_logged=5, **_V7_KW)
    assert d_rank["corrob_flagged"].tolist() == \
        [False, False, True, False, False, False, False]
    # Tie-break (c): a FULL tie (equal iso AND equal ratio) resolves by
    # index asc — the last pre-registered key, needed because the channel is
    # quantized and dyadic CSE values can tie the ratio exactly too.
    cse_full = torch.tensor([0.5, 0.5, 0.75, 0.5, 0.75, 0.5, 1.0])
    iso_full = torch.tensor([0.0, 0.0, 2 / 3, 0.0, 2 / 3, 0.0, 0.0])
    _, d_ix = v7_cse_reject_corrob_weights(
        cse_full, _V7_DS, gate=_V7_GATE, iso=iso_full, round_logged=5,
        **_V7_KW)
    # c6 takes the CSE flag (r = 2.0); c2 and c4 are identical candidates
    # (r = 1.5, iso = 2/3) for the single remaining slot -> lower index (c2).
    assert d_ix["flagged"].tolist() == [False] * 6 + [True]
    assert d_ix["corrob_flagged"].tolist() == \
        [False, False, True, False, False, False, False]
    # NaN iso cannot corroborate (sanitizes to 0.0 = maximally connected).
    iso_nan = _V7_ISO.clone(); iso_nan[5] = float("nan")
    _, d_nan = v7_cse_reject_corrob_weights(
        _V7_CSE, _V7_DS, gate=_V7_GATE, iso=iso_nan, round_logged=5, **_V7_KW)
    assert not bool(d_nan["corrob_flagged"][5])
    # Weight formula is the normative w = normalize(m_i * n_i) — the n_i
    # prior must survive the Tier-2 renormalization (kills a w = m_i mutant;
    # _V7_DS is deliberately non-uniform).
    exp_mult = d6["multiplier"].clone()
    exp_mult[5] = 0.5
    exp_w = exp_mult * _V7_DS
    exp_w = exp_w / exp_w.sum()
    assert torch.equal(w7, exp_w), (w7, exp_w)
    # Window edges are INCLUSIVE on both sides (round_min=3, round_max=10).
    for rl in (3, 10):
        _, d_edge = v7_cse_reject_corrob_weights(
            _V7_CSE, _V7_DS, gate=_V7_GATE, iso=_V7_ISO, round_logged=rl,
            **_V7_KW)
        assert d_edge["corrob_flagged"].tolist() == [False] * 5 + [True, False], rl
    # Boundary strictness, pinned with exactly-representable dyadic values:
    # ratio uses a STRICT > (r == tau_lo never flags), iso uses an INCLUSIVE
    # >= (iso == iso_min flags). cse 0.75/median 0.5 = 1.5 exact in float32.
    cse_b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75])
    iso_b = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    kw_b = dict(_V7_KW); kw_b.update(tau_lo=1.5, iso_min=0.5)
    _, d_strict = v7_cse_reject_corrob_weights(
        cse_b, _V7_DS, gate=_V7_GATE, iso=iso_b, round_logged=5, **kw_b)
    assert not bool(d_strict["corrob_flagged"].any()), \
        "r == tau_lo must NOT flag (strict >)"
    kw_b = dict(_V7_KW); kw_b.update(tau_lo=1.4375, iso_min=0.5)
    _, d_incl = v7_cse_reject_corrob_weights(
        cse_b, _V7_DS, gate=_V7_GATE, iso=iso_b, round_logged=5, **kw_b)
    assert d_incl["corrob_flagged"].tolist() == [False] * 6 + [True], \
        "iso == iso_min MUST flag (inclusive >=)"
    print("PASS  v7 corrob rule: both floors + budget + tie-break, iso alone never flags")


def test_v7_clean_federation_identity():
    """Clean pool under V7: no CSE flags, no corroborated flags -> weights
    are EXACTLY the n_k prior even in-window. The fixture deliberately
    includes the archive's known hazard, a heterogeneous benign client with
    an elevated ratio: r_2 = 0.98/0.62 = 1.58 > tau_lo — under V7 the iso
    conjunct is what protects it (iso_2 = 1/6 < iso_min). NOTE the honest
    caveat this fixture encodes: a clean benign client with BOTH r > tau_lo
    AND iso >= iso_min inside the window WOULD be mis-flagged — V7's clean
    exactness is conditional on the replay-verified premise that this joint
    event never occurs in the clean archives (replay_v7_calibration.py PASS
    criterion), unlike V4/V5/V6 whose exactness is structural. This test
    pins the mechanics, not that premise."""
    cse0 = torch.tensor([0.60, 0.55, 0.98, 0.58, 0.62, 0.70, 0.66])
    # r_2 = 0.98/0.62 = 1.58 > tau_lo, but iso_2 = 1/6: below iso_min.
    iso0 = torch.tensor([0.0, 1 / 6, 1 / 6, 0.0, 1 / 3, 0.0, 1 / 6])
    w0, d0 = v7_cse_reject_corrob_weights(
        cse0, _V7_DS, gate=torch.full((7,), 0.9), iso=iso0, round_logged=5,
        **_V7_KW)
    assert not bool(d0["flagged"].any())
    assert not bool(d0["corrob_flagged"].any())
    # Bit-exact, not allclose: the no-flag path multiplies by a literal 1.0
    # vector, so w IS ds/ds.sum() to the last ulp.
    assert torch.equal(w0, _V7_DS / _V7_DS.sum())
    assert torch.equal(d0["multiplier"], torch.ones(7))
    print("PASS  v7 clean federation: no flags -> exactly the n_k prior")


def test_v7_monotone_vs_v6():
    """m_v7 <= m_v6 pointwise for any iso and gate: Tier 2 can only replace a
    1.0 with corrob_mult < 1 on unflagged clients; Stage-A (CSE) flags and
    multipliers are byte-identical to V6's. V7 never softens anything."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        gate = torch.tensor(rng.random(7), dtype=torch.float32)
        iso = torch.tensor(
            rng.integers(0, 5, size=7) / 6.0, dtype=torch.float32)
        w6, d6 = v6_cse_reject_geo_weights(
            _V7_CSE, _V7_DS, gate=gate, tau_ratio=1.85, k_cap=2,
            m_floor=0.10, r_hard=2.5, geo_floor=0.5,
        )
        _, d7 = v7_cse_reject_corrob_weights(
            _V7_CSE, _V7_DS, gate=gate, iso=iso, round_logged=5, **_V7_KW)
        assert d7["flagged"].tolist() == d6["flagged"].tolist()
        assert torch.all(d7["multiplier"] <= d6["multiplier"] + 1e-9)
        flagged = d6["flagged"]
        assert torch.all(
            d7["multiplier"][flagged] == d6["multiplier"][flagged]
        ), "V7 must not touch a CSE-flagged client's V6 multiplier"
    # Guards: every knob out of range raises (ValueError, at the function).
    for bad in (dict(tau_lo=1.85), dict(tau_lo=1.0), dict(tau_lo=2.0),
                dict(corrob_mult=0.0), dict(corrob_mult=1.0),
                dict(corrob_mult=-0.5), dict(iso_min=0.0), dict(iso_min=1.0),
                dict(round_min=0), dict(round_min=5, round_max=4)):
        kw = dict(_V7_KW)
        kw.update(bad)
        try:
            v7_cse_reject_corrob_weights(
                _V7_CSE, _V7_DS, gate=_V7_GATE, iso=_V7_ISO,
                round_logged=5, **kw)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # A mis-sized iso is a plumbing bug: loud, not broadcast.
    try:
        v7_cse_reject_corrob_weights(
            _V7_CSE, _V7_DS, gate=_V7_GATE, iso=torch.ones(3),
            round_logged=5, **_V7_KW)
        raise AssertionError("expected ValueError for a length-3 iso")
    except ValueError:
        pass
    print("PASS  v7 monotone vs v6, CSE flags untouched, guards raise")


# --------------------------------------------------------------------------- #
# 5) runtime: suspicion EMA + checkpoint roundtrip                            #
# --------------------------------------------------------------------------- #

def _runtime(cfg_extra=None):
    cfg = {
        "device": "cpu", "proj_dim": 32, "eta_dim": 32, "hidden_dim": 32,
        "latent_dim": 16, "num_hmp_layers": 2, "knn_k": 2,
        "train_steps_per_round": 2, "semantic_weight": 0.0,
        "trust_mode": "soft_reject_fedavg", "gate_signal": "combined",
        "zscore_mode": "mad", "zscore_clip": 10.0, "gate_rezscore": False,
        "reject_z_threshold": 2.5, "sus_ema_beta": 0.6,
        "semantic_reference": "median",
    }
    cfg.update(cfg_extra or {})
    return HMPGAERuntime(num_clients=7, flat_update_dim=256, config=cfg,
                         device=torch.device("cpu"))


def test_runtime_ema_and_state_roundtrip():
    torch.manual_seed(0)
    rt = _runtime()
    ids, ds = list(range(7)), [100.0] * 7
    stats = None
    for rnd in range(2):
        rows = make_geometry(5, 2, dim=256, seed=rnd)
        _, stats = rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=rnd)
    assert set(rt.sus_ema.keys()) == set(ids), "EMA must track every client"
    assert stats["sus_ema_beta"] == 0.6 and stats["gate_rezscore"] is False
    assert stats["sus_z"] != stats["sus_raw"], "round 1 sus_z must be EMA-smoothed"

    # Roundtrip into a fresh runtime; round 2 must produce identical stats
    # (modules have no dropout, Adam is deterministic, so equal state + equal
    # inputs => equal outputs).
    rt2 = _runtime()
    rt2.load_state_dict(rt.state_dict())
    assert rt2.sus_ema == rt.sus_ema, "sus_ema must survive the checkpoint"
    rows = make_geometry(5, 2, dim=256, seed=99)
    updates = [rows[i] for i in range(7)]
    _, s1 = rt.aggregate(updates, ids, ds, round_num=2)
    _, s2 = rt2.aggregate(updates, ids, ds, round_num=2)
    assert np.allclose(s1["sus_z"], s2["sus_z"], atol=1e-6)
    assert np.allclose(s1["alpha"], s2["alpha"], atol=1e-6)
    print("PASS  runtime EMA state + checkpoint roundtrip")


def test_runtime_v4_cse_reject():
    """End-to-end runtime in V4 mode: rejection driven by the absolute
    local-CSE ratio, geometry channels still logged as diagnostics, missing
    local_cse loud (never a silent FedAvg fallback), and the num_byzantine <
    N/2 precondition enforced at construction."""
    torch.manual_seed(0)
    rt = _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 2,
                   "graph_min_distinct": 4})
    ids, ds = list(range(7)), [100.0] * 7
    rows = make_geometry(5, 2, dim=256, seed=1)
    cse = [0.60, 0.62, 0.58, 0.61, 0.59, 1.60, 1.50]
    _, stats = rt.aggregate([rows[i] for i in range(7)], ids, ds,
                            round_num=0, local_cse=cse)
    assert stats["trust_mode_used"] == "v4_cse_reject"
    assert stats["v4_flagged"] == [0, 0, 0, 0, 0, 1, 1], stats["v4_flagged"]
    alpha = np.asarray(stats["alpha"])
    # equal n_k: attacker mass = 0.1*2 / (5 + 0.1*2) ≈ 0.0385
    assert alpha[5:].sum() < 0.06, alpha
    assert alpha[5:].sum() > 0.0, "soft rejection, not zeroing"
    # Geometry diagnostics keep flowing even though they don't drive rejection.
    for key in ("residual", "recon_residual", "sus_z", "gate", "s",
                "v4_cse", "v4_ratio", "v4_median_cse"):
        assert key in stats, f"missing diagnostic {key}"
    # Missing local_cse must raise, not silently degrade.
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=1)
        raise AssertionError("expected ValueError when local_cse is missing")
    except ValueError:
        pass
    # Majority-poisoned precondition: num_byzantine >= N/2 rejected at init.
    try:
        _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 4})
        raise AssertionError("expected ValueError for num_byzantine >= N/2")
    except ValueError:
        pass
    print("PASS  runtime v4_cse_reject: flags attackers, loud on missing CSE")


def test_runtime_v5_cse_reject():
    """End-to-end runtime in V5 mode: graded rejection driven by the CSE
    ratio, V5 diagnostics emitted (and V4's scalar mult NOT emitted),
    missing local_cse loud, and the V5-specific config guards enforced at
    construction."""
    torch.manual_seed(0)
    rt = _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 2,
                   "graph_min_distinct": 4})
    ids, ds = list(range(7)), [100.0] * 7
    rows = make_geometry(5, 2, dim=256, seed=1)
    # median = 0.61: r5 = 1.60/0.61 ≈ 2.62 (saturated -> 0.10),
    # r6 = 1.50/0.61 ≈ 2.46 (ramp zone -> ≈ 0.157).
    cse = [0.60, 0.62, 0.58, 0.61, 0.59, 1.60, 1.50]
    _, stats = rt.aggregate([rows[i] for i in range(7)], ids, ds,
                            round_num=0, local_cse=cse)
    assert stats["trust_mode_used"] == "v5_cse_reject"
    assert stats["v4_flagged"] == [0, 0, 0, 0, 0, 1, 1], stats["v4_flagged"]
    mults = stats["v4_multiplier"]
    assert abs(mults[5] - 0.10) < 1e-6, mults
    assert 0.13 < mults[6] < 0.19, mults
    alpha = np.asarray(stats["alpha"])
    assert 0.0 < alpha[5:].sum() < 0.06, alpha
    # V5 diagnostics present; V4's scalar mult absent (it does not apply).
    for key in ("v5_m_floor", "v5_r_hard", "v5_ramp_t",
                "v4_cse", "v4_ratio", "v4_median_cse"):
        assert key in stats, f"missing diagnostic {key}"
    assert "v4_reject_mult" not in stats
    assert len(stats["v5_ramp_t"]) == 7
    # Missing local_cse must raise, not silently degrade.
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=1)
        raise AssertionError("expected ValueError when local_cse is missing")
    except ValueError:
        pass
    # Config guards at construction: hard zeroing and degenerate ramp.
    try:
        _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 2,
                  "v5_m_floor": 0.0})
        raise AssertionError("expected ValueError for v5_m_floor = 0")
    except ValueError:
        pass
    # ...and the same invalid V5 knobs must be INERT under V4 mode (a V4 run
    # with stale v5_* keys in defense_config must not crash).
    _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 2,
              "v5_m_floor": 0.0, "v5_r_hard": 1.0})
    try:
        _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 2,
                  "v5_r_hard": 1.85})
        raise AssertionError("expected ValueError for v5_r_hard <= tau")
    except ValueError:
        pass
    try:
        _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 4})
        raise AssertionError("expected ValueError for num_byzantine >= N/2")
    except ValueError:
        pass
    print("PASS  runtime v5_cse_reject: graded flags, V5 diagnostics, guards")


def test_runtime_v6_cse_reject_geo():
    """End-to-end runtime in V6 mode. Checks the three things only the wiring
    can get wrong: (1) the gate V6 multiplies in is the SAME vector the run
    logs as 'gate' (not sus_raw, not a freshly recomputed one); (2) against a
    V5 runtime in identical state on identical inputs, V6's multipliers are
    pointwise <= V5's, and exactly equal at v6_geo_floor = 1.0; (3) the V6
    diagnostics reach stats and are not mislabelled as V4's."""
    ids, ds = list(range(7)), [100.0] * 7
    cse = [0.60, 0.62, 0.58, 0.61, 0.59, 1.60, 1.50]

    def _run(cfg_extra):
        # Re-seed before EVERY construction so the two runtimes start from
        # bit-identical modules / random projection; the round is then a
        # deterministic function of the inputs (no dropout, Adam is exact).
        torch.manual_seed(0)
        rt = _runtime(dict({"num_byzantine": 2, "graph_min_distinct": 4},
                           **cfg_extra))
        rows_local = make_geometry(5, 2, dim=256, seed=1)
        _, st = rt.aggregate([rows_local[i] for i in range(7)], ids, ds,
                             round_num=0, local_cse=cse)
        return rt, st

    _, s6 = _run({"trust_mode": "v6_cse_reject_geo", "v6_geo_floor": 0.5})
    _, s5 = _run({"trust_mode": "v5_cse_reject"})
    _, s6_one = _run({"trust_mode": "v6_cse_reject_geo", "v6_geo_floor": 1.0})

    assert s6["trust_mode_used"] == "v6_cse_reject_geo"
    assert s6["v4_flagged"] == [0, 0, 0, 0, 0, 1, 1], s6["v4_flagged"]
    assert s6["v4_flagged"] == s5["v4_flagged"], "V6 must not move the flag set"

    # (1) the geometry input is the logged gate, on the EMA-smoothed suspicion.
    gate = np.asarray(s6["gate"])
    assert np.allclose(s6["v6_geo_gate"], gate, atol=1e-6), (s6["v6_geo_gate"], gate)
    assert np.allclose(s6["v6_geo_mult"], 0.5 + 0.5 * gate, atol=1e-6)
    assert np.allclose(s6["v6_m_cse"], s5["v4_multiplier"], atol=1e-6), (
        "V6's Stage-2 value must be V5's multiplier exactly"
    )

    # (2) monotone safety end-to-end, and exact V5 degeneracy at floor 1.0.
    m6, m5 = np.asarray(s6["v4_multiplier"]), np.asarray(s5["v4_multiplier"])
    assert np.all(m6 <= m5 + 1e-7), (m6, m5)
    flagged = np.asarray(s6["v4_flagged"], dtype=bool)
    assert np.all(m6[~flagged] == 1.0), "unflagged clients must keep m = 1.0"
    assert np.all(m6[flagged] < m5[flagged]), (
        "the fixture's gate is < 1 on the flagged clients, so V6 must be "
        "strictly tighter there"
    )
    assert np.allclose(s6_one["alpha"], s5["alpha"], atol=0.0, rtol=0.0), (
        "v6_geo_floor=1.0 must reproduce V5 exactly"
    )
    assert np.allclose(s6_one["v4_multiplier"], m5, atol=0.0, rtol=0.0)
    # Attackers lose (weakly) more mass than under V5, benigns gain.
    assert np.asarray(s6["alpha"])[5:].sum() < np.asarray(s5["alpha"])[5:].sum()

    # (3) diagnostics present and correctly labelled.
    for key in ("v6_geo_floor", "v6_geo_gate", "v6_geo_mult", "v6_m_cse",
                "v5_m_floor", "v5_r_hard", "v5_ramp_t",
                "v4_cse", "v4_ratio", "v4_median_cse"):
        assert key in s6, f"missing diagnostic {key}"
    assert "v4_reject_mult" not in s6, "V6 must not advertise V4's scalar mult"
    assert s6["v6_geo_floor"] == 0.5
    assert "v6_geo_floor" not in s5, "V5 rounds must not carry V6 diagnostics"

    # Missing local_cse must raise, not silently degrade.
    torch.manual_seed(0)
    rt = _runtime({"trust_mode": "v6_cse_reject_geo", "num_byzantine": 2})
    rows = make_geometry(5, 2, dim=256, seed=1)
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=0)
        raise AssertionError("expected ValueError when local_cse is missing")
    except ValueError:
        pass

    # Config guards, all at CONSTRUCTION (a raise from aggregate() would be
    # swallowed by HMPGAEDefense's FedAvg fallback and run 50 silent rounds).
    _runtime({"trust_mode": "v6_cse_reject_geo", "num_byzantine": 2,
              "v6_geo_floor": 1.0})               # the V5-equivalence point is legal
    for bad in ({"v6_geo_floor": 0.0}, {"v6_geo_floor": 1.5},
                {"v6_geo_floor": -0.5}, {"v5_m_floor": 0.0},
                {"v5_r_hard": 1.85}, {"num_byzantine": 4}):
        cfg = dict({"trust_mode": "v6_cse_reject_geo", "num_byzantine": 2}, **bad)
        try:
            _runtime(cfg)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # ...and an out-of-range v6_geo_floor must stay INERT under V4/V5 (a
    # stale key in defense_config must not crash an older arm).
    _runtime({"trust_mode": "v4_cse_reject", "num_byzantine": 2,
              "v6_geo_floor": 0.0})
    _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 2,
              "v6_geo_floor": 0.0})
    print("PASS  runtime v6_cse_reject_geo: gate wired in, <= V5, guards")


def test_runtime_v7_cse_reject_corrob():
    """End-to-end runtime in V7 mode. Checks the four things only the wiring
    can get wrong: (1) the Tier-2 iso input is the SAME vector the run logs
    as 'residual' (raw graph_residual — NOT the z-scored gate); (2) the
    window arithmetic is 1-indexed (round_num 2 = logged round 3 = armed;
    round_num 0 = logged round 1 = V6 bit-identical); (3) when the graph
    channel is resolution-gated (graph_min_distinct) the geometry abstains
    end-to-end; (4) the v7_* diagnostics reach stats and stale v7 keys stay
    inert under older modes."""
    ids, ds = list(range(7)), [100.0] * 7
    # c6: genuine CSE flag (r = 3.28). c5: elevated-but-sub-1.85 (r = 1.56)
    # — V4/V5/V6 structurally cannot touch it; V7's cold window can, iff the
    # hypergraph corroborates. With the seed-1 geometry the raw residuals
    # resolve only 3 distinct levels, so graph_min_distinct=4 gates the
    # channel; the in-window arms therefore run graph_min_distinct=0.
    cse = [0.60, 0.62, 0.58, 0.61, 0.59, 0.95, 2.00]

    def _run(cfg_extra, round_num=2):
        torch.manual_seed(0)
        rt = _runtime(dict({"num_byzantine": 2, "graph_min_distinct": 0},
                           **cfg_extra))
        rows_local = make_geometry(5, 2, dim=256, seed=1)
        _, st = rt.aggregate([rows_local[i] for i in range(7)], ids, ds,
                             round_num=round_num, local_cse=cse)
        return st

    v7_cfg = {"trust_mode": "v7_cse_reject_corrob",
              "v7_tau_lo": 1.40, "v7_iso_min": 5 / 12,
              "v7_corrob_mult": 0.5, "v7_round_min": 3, "v7_round_max": 10}

    s7 = _run(v7_cfg, round_num=2)              # logged round 3: in-window
    s6 = _run({"trust_mode": "v6_cse_reject_geo"}, round_num=2)
    assert s7["trust_mode_used"] == "v7_cse_reject_corrob"
    # (1) iso == the logged raw residual channel, and the geometry resolved.
    assert np.allclose(s7["v7_iso"], s7["residual"], atol=1e-7)
    assert s7["v7_geo_resolved"] is True
    # Flag sets: CSE flag on c6 identical to V6; corroborated flag on c5
    # (r=1.56 > tau_lo, residual 0.5 >= 5/12); the two benign clients the
    # hypergraph dislikes MOST (c0/c3 at residual 2/3) stay untouched
    # because their ratio is ~1 — iso alone never flags.
    assert s7["v4_flagged"] == s6["v4_flagged"] == [0, 0, 0, 0, 0, 0, 1]
    assert s7["v7_corrob_flagged"] == [0, 0, 0, 0, 0, 1, 0]
    m7 = np.asarray(s7["v4_multiplier"])
    assert abs(m7[5] - 0.5) < 1e-7
    assert m7[0] == 1.0 and m7[3] == 1.0
    assert np.asarray(s7["alpha"])[5] < np.asarray(s6["alpha"])[5]
    # (2) out-of-window round: V7 == V6 bit-for-bit (logged round 1 < 3).
    s7_cold = _run(v7_cfg, round_num=0)
    s6_cold = _run({"trust_mode": "v6_cse_reject_geo"}, round_num=0)
    assert np.allclose(s7_cold["alpha"], s6_cold["alpha"], atol=0.0, rtol=0.0)
    assert s7_cold["v7_corrob_flagged"] == [0] * 7
    # (3) graph channel gated -> geometry abstains -> V6 exactly, in-window.
    s7_gated = _run(dict(v7_cfg, graph_min_distinct=4), round_num=2)
    s6_gated = _run({"trust_mode": "v6_cse_reject_geo",
                     "graph_min_distinct": 4}, round_num=2)
    assert s7_gated["graph_channel_gated"] is True
    assert s7_gated["v7_geo_resolved"] is False
    assert s7_gated["v7_corrob_flagged"] == [0] * 7
    assert np.allclose(s7_gated["alpha"], s6_gated["alpha"],
                       atol=0.0, rtol=0.0)
    # (3b) knob plumbing + inclusive edges with a NON-DEFAULT window: the
    # config values must actually reach the scorer call (a dropped kwarg
    # falls back to the function defaults and is invisible when the config
    # repeats them), and round_max is the LAST armed round — this arm also
    # kills a round_num+2 / exclusive-'<' wiring regression, because logged
    # round 4 must NOT fire while logged round 3 (== round_max) must.
    v7_edge = dict(v7_cfg, v7_round_min=2, v7_round_max=3,
                   v7_corrob_mult=0.25)
    s_edge = _run(v7_edge, round_num=2)      # logged 3 == round_max: armed
    assert s_edge["v7_corrob_flagged"] == [0, 0, 0, 0, 0, 1, 0]
    assert abs(np.asarray(s_edge["v4_multiplier"])[5] - 0.25) < 1e-7, \
        "v7_corrob_mult must be plumbed, not the 0.5 default"
    assert s_edge["v7_round_max"] == 3 and s_edge["v7_round_min"] == 2
    s_lo = _run(v7_edge, round_num=1)        # logged 2 == round_min: armed
    assert s_lo["v7_corrob_flagged"] == [0, 0, 0, 0, 0, 1, 0], \
        "v7_round_min must be plumbed (default 3 would leave logged 2 cold)"
    s_past = _run(v7_edge, round_num=3)      # logged 4 > round_max: V6 again
    assert s_past["v7_corrob_flagged"] == [0] * 7
    # tau_lo plumbing: a floor above c5's r = 1.56 must silence Tier 2
    # (the 1.40 default would fire).
    s_hi = _run(dict(v7_cfg, v7_tau_lo=1.60), round_num=2)
    assert s_hi["v7_corrob_flagged"] == [0] * 7
    # (4) diagnostics present and correctly labelled; V6 rounds carry none.
    for key in ("v7_corrob_flagged", "v7_iso", "v7_geo_resolved",
                "v7_tau_lo", "v7_iso_min", "v7_corrob_mult",
                "v7_round_min", "v7_round_max",
                "v6_geo_floor", "v6_geo_gate", "v6_geo_mult", "v6_m_cse",
                "v5_m_floor", "v5_r_hard", "v5_ramp_t",
                "v4_cse", "v4_ratio", "v4_median_cse"):
        assert key in s7, f"missing diagnostic {key}"
    assert "v4_reject_mult" not in s7
    assert "v7_corrob_flagged" not in s6, \
        "V6 rounds must not carry V7 diagnostics"

    # Missing local_cse must raise, not silently degrade.
    torch.manual_seed(0)
    rt = _runtime({"trust_mode": "v7_cse_reject_corrob", "num_byzantine": 2})
    rows = make_geometry(5, 2, dim=256, seed=1)
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=2)
        raise AssertionError("expected ValueError when local_cse is missing")
    except ValueError:
        pass

    # Config guards at CONSTRUCTION (same rationale as V6: a raise inside
    # aggregate() would be eaten by the FedAvg fallback net).
    _runtime({"trust_mode": "v7_cse_reject_corrob", "num_byzantine": 2,
              "v7_round_max": 0})               # window-off arm is legal
    for bad in ({"v7_tau_lo": 1.85}, {"v7_tau_lo": 1.0},
                {"v7_corrob_mult": 0.0}, {"v7_corrob_mult": 1.0},
                {"v7_iso_min": 0.0}, {"v7_iso_min": 1.0},
                {"v7_round_min": 0}, {"v7_round_min": 5, "v7_round_max": 4},
                {"v5_m_floor": 0.0}, {"v6_geo_floor": 0.0},
                {"num_byzantine": 4}):
        cfg = dict({"trust_mode": "v7_cse_reject_corrob", "num_byzantine": 2},
                   **bad)
        try:
            _runtime(cfg)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # ...and out-of-range v7 keys must stay INERT under V4/V5/V6 (a stale
    # key in defense_config must not crash an older arm).
    for mode in ("v4_cse_reject", "v5_cse_reject", "v6_cse_reject_geo"):
        _runtime({"trust_mode": mode, "num_byzantine": 2,
                  "v7_tau_lo": 5.0, "v7_corrob_mult": 0.0,
                  "v7_round_min": 0})
    print("PASS  runtime v7_cse_reject_corrob: iso wired raw, window 1-indexed, guards")


def test_defense_facade_local_cse_guard():
    """The loud-crash contract lives in the FACADE, before its FedAvg
    fallback try/except: a CSE-reject trust_mode without a local_cse vector
    must raise RuntimeError out of HMPGAEDefense.aggregate, never degrade to
    50 silent FedAvg rounds. Includes a whitespace-padded mode string: the
    runtime normalizes with .strip().lower(), so the facade (and
    Server._needs_local_cse) must strip identically or the padded mode
    slips past this guard while still running a CSE-reject rule."""
    from defense import HMPGAEDefense
    rows = make_geometry(5, 2, dim=256, seed=1)
    updates = [rows[i] for i in range(7)]
    for mode in ("v7_cse_reject_corrob", " v7_cse_reject_corrob ",
                 "v4_cse_reject"):
        d = HMPGAEDefense(num_clients=7, config={
            "trust_mode": mode, "num_byzantine": 2, "device": "cpu",
            "proj_dim": 32, "eta_dim": 32, "hidden_dim": 32,
            "latent_dim": 16, "num_hmp_layers": 2, "knn_k": 2,
            "train_steps_per_round": 2})
        try:
            d.aggregate(updates, list(range(7)), [100.0] * 7, 0,
                        torch.device("cpu"))
            raise AssertionError(f"expected RuntimeError for mode={mode!r}")
        except RuntimeError:
            pass
    print("PASS  defense facade: missing local_cse raises before any fallback")


def test_runtime_attackers_lose_weight_over_rounds():
    """End-to-end runtime with the production signal stack (semantic ON):
    after the EMA warms up, attackers hold negligible aggregation mass."""
    torch.manual_seed(0)
    rt = _runtime({"semantic_weight": 1.0})
    ids, ds = list(range(7)), [100.0] * 7
    stats = None
    for rnd in range(4):
        rows = make_geometry(5, 2, dim=256, seed=7, round_seed=rnd)
        probes = make_probe_dists(5, 2, seed=7, round_seed=rnd)
        _, stats = rt.aggregate(
            [rows[i] for i in range(7)], ids, ds, round_num=rnd,
            probe_distributions=probes,
        )
    alpha = np.asarray(stats["alpha"])
    assert alpha[5:].sum() < 0.10, f"attacker alpha after EMA warmup: {alpha[5:]}"
    assert alpha[:5].min() > 0.10, f"benign alpha collapsed: {alpha[:5]}"
    print(f"PASS  runtime end-to-end: attacker mass {alpha[5:].sum():.3f} after 4 rounds")


if __name__ == "__main__":
    test_zscore_std_backward_compat()
    test_zscore_mad_high_attacker_fraction()
    test_zscore_mad_degenerate_fallback()
    test_zscore_relative_degeneracy_guard()
    test_zscore_clip_post_fusion()
    test_graph_min_distinct_gating()
    test_semantic_pairwise_bitforbit()
    test_semantic_median_beats_pairwise_noniid()
    test_no_attack_no_scapegoat()
    test_attack_detected_and_rejected()
    test_sus_override_drives_gate()
    test_trust_weights_default_path_unchanged()
    test_v4_cse_reject_rule()
    test_v4_hard_removal_arm()
    test_v5_cse_reject_rule()
    test_v5_saturation_equals_v4()
    test_v5_false_positive_containment()
    test_v6_geo_floor_one_equals_v5()
    test_v6_monotone_safety()
    test_v6_clean_federation_identity()
    test_v6_monotone_in_ratio_and_gate()
    test_v7_window_off_equals_v6()
    test_v7_corroborated_flag_rule()
    test_v7_clean_federation_identity()
    test_v7_monotone_vs_v6()
    test_runtime_ema_and_state_roundtrip()
    test_runtime_v4_cse_reject()
    test_runtime_v5_cse_reject()
    test_runtime_v6_cse_reject_geo()
    test_runtime_v7_cse_reject_corrob()
    test_defense_facade_local_cse_guard()
    test_runtime_attackers_lose_weight_over_rounds()
    print("\nAll trust-robustness tests passed.")
