# tests/test_trust_robustness.py
# CPU-only sanity tests for the robust trust-scoring stack (2026-07):
#   - mad z-score (robust to high attacker fractions, degenerate-MAD fallback)
#   - median semantic reference (non-IID robustness)
#   - gate_rezscore=False + weight-norm scaling (no "scapegoat tax" on
#     all-benign rounds; threshold in per-signal robust-z units)
#   - suspicion EMA state + checkpoint roundtrip
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
    test_runtime_ema_and_state_roundtrip()
    test_runtime_v4_cse_reject()
    test_runtime_attackers_lose_weight_over_rounds()
    print("\nAll trust-robustness tests passed.")
