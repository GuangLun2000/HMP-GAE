# tests/test_trust_robustness.py
# CPU-only sanity tests for the CSE-reject trust stack (V4 / V5 / V8):
#   - the V4 rule (rank cap AND ratio floor, soft + hard-removal arms)
#   - the V5 graded ramp (identical flag set, saturation == V4, FP containment)
#   - V8's safe degradation (no seed / no edge / no budget == V5 exactly),
#     joint-evidence propagation, and dual-view consensus topology
#   - runtime wiring for all three modes, incl. the loud-crash contracts
#     (missing local_cse / probe, removed legacy modes, num_byzantine < N/2)
#   - V8 GAE state + checkpoint roundtrip (Adam-aliasing regression)
#   - hallucination flip-ratio schedule + resume fingerprint invariants
#
# The V1-V3 geometry stack and the V6/V7 arms were removed 2026-08-11
# (docs/DECISION.md); their tests went with them and live in git history.
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

from attack.hallucination import deterministic_round_flip_ratio
from fed_resume import _fingerprint, _fingerprint_mismatches
from hmp_gae.trust_scorer import (
    v4_cse_reject_weights,
    v5_cse_reject_weights,
    v8_hmp_cse_propagation_weights,
)
from hmp_gae.hypergraph import (
    consensus_propagation_hypergraph,
    hypergraph_propagation_matrix,
)
from hmp_gae.runtime import HMPGAERuntime


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

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
    Synthetic (N, dim) client updates: benign = shared direction + persistent
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


def _runtime(cfg_extra=None):
    """Small-dimension runtime for wiring tests. trust_mode must be supplied
    by the caller — the runtime deliberately has no default."""
    cfg = {
        "device": "cpu", "proj_dim": 32, "eta_dim": 32, "hidden_dim": 32,
        "latent_dim": 16, "num_hmp_layers": 2, "knn_k": 2,
        "train_steps_per_round": 2, "num_byzantine": 2,
    }
    cfg.update(cfg_extra or {})
    return HMPGAERuntime(num_clients=7, flat_update_dim=256, config=cfg,
                         device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# 1) V4 rejection rule (per-client CSE, pool-median normalised, rank-capped)  #
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
    rt = _runtime({"trust_mode": "v4_cse_reject", "v4_reject_mult": 0.0})
    assert rt.v4_reject_mult == 0.0
    try:
        _runtime({"trust_mode": "v4_cse_reject", "v4_reject_mult": -0.1})
        raise AssertionError("expected ValueError for negative reject_mult")
    except ValueError:
        pass
    print("PASS  v4 hard-removal arm: flagged mass exactly 0, survivors = n_k prior")


# --------------------------------------------------------------------------- #
# 2) V5 graded rejection                                                      #
# --------------------------------------------------------------------------- #

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
# 3) V8: CSE-seeded dual-view HMP propagation                                 #
# --------------------------------------------------------------------------- #

def test_v8_safe_degradation_to_v5():
    """No seed, no edge, or exhausted cap must return V5 exactly."""
    ds = torch.tensor([100., 120., 90., 110., 105., 95., 80.])
    dense_T = torch.ones(7, 7) - torch.eye(7)

    # Clean federation: no V5 seed, so even a dense graph has no authority.
    clean = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 0.72, 0.66])
    w5, _ = v5_cse_reject_weights(clean, ds)
    w8, d8 = v8_hmp_cse_propagation_weights(clean, ds, dense_T)
    assert torch.equal(w8, w5), "no-seed V8 must return V5 exactly"
    assert not bool(d8["propagated_flagged"].any())

    # One seed but no reliable cross-view edge: exact V5 again.
    one_seed = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 0.90, 1.50])
    w5e, _ = v5_cse_reject_weights(one_seed, ds)
    w8e, d8e = v8_hmp_cse_propagation_weights(
        one_seed, ds, torch.zeros(7, 7)
    )
    assert torch.equal(w8e, w5e), "zero-edge V8 must return V5 exactly"
    assert float(d8e["propagated_risk"].sum()) == 0.0

    # Two V5 seeds exhaust k_cap=2; propagation cannot displace either seed.
    two_seeds = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 1.40, 1.50])
    w5c, d5c = v5_cse_reject_weights(two_seeds, ds, k_cap=2)
    w8c, d8c = v8_hmp_cse_propagation_weights(
        two_seeds, ds, dense_T, k_cap=2
    )
    assert int(d5c["flagged"].sum()) == 2
    assert torch.equal(w8c, w5c), "exhausted rank cap must return V5 exactly"
    assert not bool(d8c["propagated_flagged"].any())
    print("PASS  v8 safe degradation: no seed/edge/budget == V5 exactly")


def test_v8_propagates_only_joint_evidence():
    """A V5 seed can softly expose one connected, moderate-CSE peer."""
    ds = torch.ones(7)
    cse = torch.tensor([0.60, 0.55, 0.70, 0.58, 0.62, 0.90, 1.50])
    # median=.60: c6 ratio=2.5 is a V5 seed; c5 ratio=1.5 is below tau.
    T = torch.zeros(7, 7)
    T[5, 6] = 0.8
    w5, d5 = v5_cse_reject_weights(cse, ds, k_cap=2)
    w8, d8 = v8_hmp_cse_propagation_weights(cse, ds, T, k_cap=2)
    assert d5["flagged"].tolist() == [False] * 6 + [True]
    assert d8["flagged"].tolist() == d5["flagged"].tolist(), \
        "V8 must not change the high-confidence CSE seed set"
    assert d8["propagated_flagged"].tolist() == \
        [False, False, False, False, False, True, False]
    assert 0.0 < float(d8["joint_evidence"][5]) < 1.0
    assert float(d8["multiplier"][5]) < 1.0
    assert float(w8[5]) < float(w5[5])
    assert int(d8["flagged"].sum() + d8["propagated_flagged"].sum()) <= 2
    # A connected client at/below the benign-controlled median has zero CSE
    # evidence and therefore cannot consume the remaining budget.
    low = cse.clone()
    low[5] = 0.60
    w8_low, d8_low = v8_hmp_cse_propagation_weights(low, ds, T, k_cap=2)
    w5_low, _ = v5_cse_reject_weights(low, ds, k_cap=2)
    assert torch.equal(w8_low, w5_low)
    assert not bool(d8_low["propagated_flagged"].any())
    print("PASS  v8 joint evidence: seed + HMP relation + elevated CSE required")


def test_v8_dual_view_consensus_and_affinity_mass():
    """Only mutual edges shared by both views propagate, with GAE attenuation."""
    update_H = torch.eye(4)
    behavior_H = torch.eye(4)
    # Shared mutual pair 0<->1.
    update_H[1, 0] = update_H[0, 1] = 1.0
    behavior_H[1, 0] = behavior_H[0, 1] = 1.0
    # View-specific pairs must not survive the intersection.
    update_H[2, 1] = update_H[1, 2] = 1.0
    behavior_H[3, 2] = behavior_H[2, 3] = 1.0
    prop_H, _, _, consensus = consensus_propagation_hypergraph(
        update_H, behavior_H
    )
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0, 1] = expected[1, 0] = True
    assert torch.equal(consensus, expected)

    affinity = torch.zeros(4, 4)
    affinity[0, 1] = affinity[1, 0] = 0.25
    T = hypergraph_propagation_matrix(prop_H, pair_affinity=affinity)
    assert abs(float(T[0, 1]) - 0.25) < 1e-6
    assert abs(float(T[1, 0]) - 0.25) < 1e-6
    assert float(T[2:].abs().sum()) == 0.0
    assert float(T[:, 2:].abs().sum()) == 0.0
    # Load-bearing: do not renormalize the only weak edge back to 1.
    assert float(T.max()) < 0.3
    print("PASS  v8 topology: dual-view mutual consensus + sub-stochastic affinity")


# --------------------------------------------------------------------------- #
# 4) attack schedule + resume fingerprint                                     #
# --------------------------------------------------------------------------- #

def test_hallucination_round_ratio_resume_exact():
    """Stateless round lookup must reproduce the old sequential RNG stream."""
    bounds = (0.3, 0.8)
    for seed in (5, 6):
        sequential = np.random.default_rng(seed).uniform(*bounds, size=50)
        replayed = np.asarray([
            deterministic_round_flip_ratio(seed, rnd, bounds)
            for rnd in range(50)
        ])
        assert np.array_equal(replayed, sequential)
        # Directly asking for a late resumed round cannot depend on prior calls.
        assert deterministic_round_flip_ratio(seed, 43, bounds) == sequential[43]
    try:
        deterministic_round_flip_ratio(5, -1, bounds)
        raise AssertionError("negative round_num must fail")
    except ValueError:
        pass
    try:
        deterministic_round_flip_ratio(5, 0, (0.8, 0.3))
        raise AssertionError("invalid flip-ratio bounds must fail")
    except ValueError:
        pass
    print("PASS  hallucination flip-ratio schedule is checkpoint-resume exact")


def test_resume_fingerprint_covers_trajectory_and_legacy():
    """New snapshots guard attack knobs; old snapshots retain subset safety."""
    cfg = {
        "experiment_name": "arm-a",
        "seed": 42,
        "dataset": "ag_news",
        "model_name": "model",
        "hallu_flip_ratio_range": [0.3, 0.8],
        "defense_config": {"trust_mode": "v8_hmp_cse_propagation"},
    }
    current = _fingerprint(cfg)
    changed = dict(cfg, hallu_flip_ratio_range=[0.6, 1.0])
    assert any(
        "hallu_flip_ratio_range" in m
        for m in _fingerprint_mismatches(current, changed)
    )

    # Schema-1 snapshots are checked on their recorded safety subset instead
    # of being rejected merely because newer keys did not exist yet.
    legacy = {
        "experiment_name": cfg["experiment_name"],
        "seed": cfg["seed"],
        "dataset": cfg["dataset"],
        "model_name": cfg["model_name"],
        "defense_config": cfg["defense_config"],
    }
    assert _fingerprint_mismatches(legacy, cfg) == []
    assert any(
        "dataset" in m
        for m in _fingerprint_mismatches(legacy, dict(cfg, dataset="yahoo_answers"))
    )
    print("PASS  resume fingerprint covers trajectory keys and legacy snapshots")


# --------------------------------------------------------------------------- #
# 5) runtime wiring                                                           #
# --------------------------------------------------------------------------- #

def test_removed_modes_raise():
    """The 2026-08-11 pruning contract: a legacy trust_mode in an old config
    must fail LOUDLY at construction (outside the FedAvg fallback net), never
    silently run a different rule. Missing trust_mode is equally loud, and
    stale legacy knobs alone must stay inert."""
    for mode in ("soft_reject_fedavg", "reject_then_fedavg", "softmax",
                 "v6_cse_reject_geo", "v7_cse_reject_corrob", "typo_mode"):
        try:
            _runtime({"trust_mode": mode})
            raise AssertionError(f"expected ValueError for mode={mode!r}")
        except ValueError:
            pass
    try:
        _runtime({})
        raise AssertionError("expected ValueError for missing trust_mode")
    except ValueError:
        pass
    # Stale keys from archived pre-pruning configs are ignored, not fatal —
    # re-running an old arm's config (with its mode updated) must not crash.
    rt = _runtime({"trust_mode": "v5_cse_reject",
                   "v6_geo_floor": 0.0, "v7_tau_lo": 5.0,
                   "zscore_mode": "mad", "gate_rezscore": False,
                   "sus_ema_beta": 0.6, "graph_min_distinct": 4})
    assert rt.trust_mode == "v5_cse_reject"
    print("PASS  removed modes raise; stale legacy knobs stay inert")


def test_runtime_v4_cse_reject():
    """End-to-end runtime in V4 mode: rejection driven by the absolute
    local-CSE ratio, missing local_cse loud (never a silent FedAvg fallback),
    the num_byzantine < N/2 precondition enforced at construction, and no
    legacy geometry channels in the stats schema."""
    torch.manual_seed(0)
    rt = _runtime({"trust_mode": "v4_cse_reject"})
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
    for key in ("v4_cse", "v4_ratio", "v4_median_cse", "v4_reject_mult"):
        assert key in stats, f"missing diagnostic {key}"
    # The legacy geometry stack is gone — its channels must not reappear.
    for gone in ("residual", "sus_z", "gate", "s", "sem_div", "alpha_hmp"):
        assert gone not in stats, f"legacy channel {gone} resurfaced"
    # V4 runs no GAE: no Z, no losses.
    assert "Z" not in stats and "L_rec" not in stats
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
    rt = _runtime({"trust_mode": "v5_cse_reject"})
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
        _runtime({"trust_mode": "v5_cse_reject", "v5_m_floor": 0.0})
        raise AssertionError("expected ValueError for v5_m_floor = 0")
    except ValueError:
        pass
    # ...and the same invalid V5 knobs must be INERT under V4 mode (a V4 run
    # with stale v5_* keys in defense_config must not crash).
    _runtime({"trust_mode": "v4_cse_reject",
              "v5_m_floor": 0.0, "v5_r_hard": 1.0})
    try:
        _runtime({"trust_mode": "v5_cse_reject", "v5_r_hard": 1.85})
        raise AssertionError("expected ValueError for v5_r_hard <= tau")
    except ValueError:
        pass
    try:
        _runtime({"trust_mode": "v5_cse_reject", "num_byzantine": 4})
        raise AssertionError("expected ValueError for num_byzantine >= N/2")
    except ValueError:
        pass
    print("PASS  runtime v5_cse_reject: graded flags, V5 diagnostics, guards")


def test_runtime_v8_hmp_cse_propagation():
    """V8 end-to-end: fixed dual-view topology, learned affinity, and CSE
    seed propagation all reach the applied weights and archived diagnostics."""
    torch.manual_seed(0)
    rt = _runtime({
        "trust_mode": "v8_hmp_cse_propagation",
        "semantic_weight": 1.0,
    })
    ids, ds = list(range(7)), [100.0] * 7
    rows = make_geometry(5, 2, dim=256, seed=19)
    # Make the two suspicious clients exact update-view neighbors.
    rows[6] = rows[5].clone()
    probes = make_probe_dists(5, 2, seed=19)
    # And exact behavior-view neighbors; labels never enter this graph.
    probes[6] = probes[5].clone()
    cse = [0.60, 0.55, 0.70, 0.58, 0.62, 0.90, 1.50]
    _, stats = rt.aggregate(
        [rows[i] for i in range(7)], ids, ds, round_num=0,
        probe_distributions=probes, local_cse=cse,
    )
    assert stats["trust_mode_used"] == "v8_hmp_cse_propagation"
    assert stats["v4_flagged"] == [0, 0, 0, 0, 0, 0, 1]
    assert stats["v8_propagated_flagged"] == [0, 0, 0, 0, 0, 1, 0]
    assert stats["v8_consensus_mutual"][5][6] == 1
    assert stats["v8_consensus_edge_count"] >= 1
    assert 0.0 < stats["v8_propagated_risk"][5] <= 1.0
    assert stats["v4_multiplier"][5] < 1.0
    assert "L_struct" in stats and "v8_recon_error" in stats
    assert abs(sum(stats["alpha"]) - 1.0) < 1e-6

    # Both required inputs fail loudly in the runtime itself.
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=1,
                     local_cse=cse)
        raise AssertionError("expected ValueError for missing V8 probe")
    except ValueError:
        pass
    try:
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=1,
                     probe_distributions=probes)
        raise AssertionError("expected ValueError for missing V8 local_cse")
    except ValueError:
        pass
    # A V8 config cannot disable the independent behavior view.
    try:
        _runtime({"trust_mode": "v8_hmp_cse_propagation",
                  "semantic_weight": 0.0})
        raise AssertionError("expected ValueError for semantic_weight=0 in V8")
    except ValueError:
        pass
    print("PASS  runtime v8: dual-view HMP propagates one CSE seed to its peer")


def test_runtime_v8_state_roundtrip():
    """V8's cross-round state (GAE modules, Adam moments, z_hist EMA) must
    survive an in-memory checkpoint roundtrip: a restored runtime given the
    same inputs must reproduce the uninterrupted one exactly. Guards the
    Adam-aliasing regression (a snapshot sharing live moment tensors made
    'resumed' runs silently diverge)."""
    def _fresh():
        torch.manual_seed(0)
        return _runtime({"trust_mode": "v8_hmp_cse_propagation",
                         "semantic_weight": 1.0})

    ids, ds = list(range(7)), [100.0] * 7
    cse = [0.60, 0.55, 0.70, 0.58, 0.62, 0.90, 1.50]

    rt = _fresh()
    for rnd in range(2):
        rows = make_geometry(5, 2, dim=256, seed=7, round_seed=rnd)
        probes = make_probe_dists(5, 2, seed=7, round_seed=rnd)
        rt.aggregate([rows[i] for i in range(7)], ids, ds, round_num=rnd,
                     probe_distributions=probes, local_cse=cse)
    assert set(rt.z_hist.keys()) == set(ids), "z_hist must track every client"

    snapshot = rt.state_dict()
    rt2 = _fresh()
    rt2.load_state_dict(snapshot)

    rows = make_geometry(5, 2, dim=256, seed=7, round_seed=99)
    probes = make_probe_dists(5, 2, seed=7, round_seed=99)
    updates = [rows[i] for i in range(7)]
    _, s1 = rt.aggregate(updates, ids, ds, round_num=2,
                         probe_distributions=probes, local_cse=cse)
    _, s2 = rt2.aggregate(updates, ids, ds, round_num=2,
                          probe_distributions=probes, local_cse=cse)
    assert np.allclose(s1["alpha"], s2["alpha"], atol=1e-6)
    assert np.allclose(s1["v8_propagated_risk"], s2["v8_propagated_risk"],
                       atol=1e-6)
    assert np.allclose(s1["v8_recon_error"], s2["v8_recon_error"], atol=1e-6)

    # V4/V5 are stateless: empty state_dict, load is a tolerated no-op (also
    # covers loading a legacy checkpoint written by a removed mode).
    rt4 = _runtime({"trust_mode": "v4_cse_reject"})
    assert rt4.state_dict() == {}
    rt4.load_state_dict({"node_encoder": {}, "sus_ema": {0: 1.0}})
    print("PASS  runtime v8 state + checkpoint roundtrip")


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
    for mode in ("v4_cse_reject", "v5_cse_reject",
                 "v8_hmp_cse_propagation", " v8_hmp_cse_propagation "):
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
    # V8's missing probe is also a pre-fallback plumbing error. Supply CSE so
    # the probe guard itself is the one being exercised.
    d8 = HMPGAEDefense(num_clients=7, config={
        "trust_mode": "v8_hmp_cse_propagation", "num_byzantine": 2,
        "semantic_weight": 1.0, "device": "cpu",
        "proj_dim": 32, "eta_dim": 32, "hidden_dim": 32,
        "latent_dim": 16, "num_hmp_layers": 2, "knn_k": 2,
        "train_steps_per_round": 2,
    })
    try:
        d8.aggregate(updates, list(range(7)), [100.0] * 7, 0,
                     torch.device("cpu"), local_cse=[0.6] * 7)
        raise AssertionError("expected RuntimeError for missing V8 probe")
    except RuntimeError:
        pass
    print("PASS  defense facade: missing local_cse raises before any fallback")


def test_detection_summary_reads_live_diagnostics():
    """main.compute_detection_summary scores the run from the aggregation log,
    so every key it reads must (a) still be emitted by the runtime and (b) be
    on server.py's persistence whitelist.

    Regression: the 2026-08-11 pruning deleted the geometry 'gate'/'sus_z'
    channels, but the summary still keyed its whole per-round loop on 'gate'.
    Every round hit `continue`, so detection_summary silently became null and
    the V8 propagation recall/FPR/precision — the numbers that falsify the
    mechanism — vanished from the results JSON. Parsed statically (ast) so
    this needs no transformers/datasets import.
    """
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "main.py"), encoding="utf-8") as f:
        main_src = f.read()
    with open(os.path.join(root, "server.py"), encoding="utf-8") as f:
        server_src = f.read()

    fn = next(
        n for n in ast.walk(ast.parse(main_src))
        if isinstance(n, ast.FunctionDef)
        and n.name == "compute_detection_summary"
    )
    wanted = {
        node.args[0].value
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "agg"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert wanted, "no agg.get('...') reads found — the ast walk drifted"

    for key in sorted(wanted):
        assert f"'{key}'" in server_src, (
            f"compute_detection_summary reads '{key}' but server.py never "
            f"writes it into the aggregation log"
        )

    # The runtime must still emit them. V8 is the superset of the three modes'
    # diagnostics, so one V8 round covers every key the summary can ask for.
    torch.manual_seed(0)
    rt = _runtime({
        "trust_mode": "v8_hmp_cse_propagation",
        "semantic_weight": 1.0,
    })
    rows = make_geometry(5, 2, dim=256, seed=19)
    rows[6] = rows[5].clone()
    probes = make_probe_dists(5, 2, seed=19)
    probes[6] = probes[5].clone()
    _, stats = rt.aggregate(
        [rows[i] for i in range(7)], list(range(7)), [100.0] * 7, round_num=0,
        probe_distributions=probes,
        local_cse=[0.60, 0.55, 0.70, 0.58, 0.62, 0.90, 1.50],
    )
    # 'accepted_clients' is the server's own bookkeeping, not a defense stat.
    for key in sorted(wanted - {"accepted_clients"}):
        assert key in stats, (
            f"compute_detection_summary reads '{key}' but the V8 runtime no "
            f"longer emits it"
        )
    print("PASS  detection summary reads only live, persisted diagnostics")


if __name__ == "__main__":
    test_v4_cse_reject_rule()
    test_v4_hard_removal_arm()
    test_v5_cse_reject_rule()
    test_v5_saturation_equals_v4()
    test_v5_false_positive_containment()
    test_v8_safe_degradation_to_v5()
    test_v8_propagates_only_joint_evidence()
    test_v8_dual_view_consensus_and_affinity_mass()
    test_hallucination_round_ratio_resume_exact()
    test_resume_fingerprint_covers_trajectory_and_legacy()
    test_removed_modes_raise()
    test_runtime_v4_cse_reject()
    test_runtime_v5_cse_reject()
    test_runtime_v8_hmp_cse_propagation()
    test_runtime_v8_state_roundtrip()
    test_defense_facade_local_cse_guard()
    test_detection_summary_reads_live_diagnostics()
    print("\nAll trust-robustness tests passed.")
