# hmp_gae/trust_scorer.py
# CSE-reject decision rules for HMP-GAE.
#
# All three surviving trust modes share one detection statistic: the ABSOLUTE
# per-client full-test Classification Semantic Entropy (CSE), pool-median
# normalised into a ratio r_i and rank-capped by defense_config.num_byzantine.
# The pool median must be benign-controlled (num_byzantine < N/2, validated by
# the runtime) or the rule inverts.
#
# V4 (2026-07-28): flag = top-k_cap by r AND r > tau_ratio; flagged clients
# take the constant multiplier v4_reject_mult (v4_cse_reject_weights).
# tau_ratio = 1.85 is pre-registered (zero-FP plateau [1.785, 1.90] over 51
# archived runs); do not re-tune post hoc.
#
# V5 (2026-08-06): V4's flag decision byte-identical, but the flagged-client
# multiplier is a linear ramp in r (v5_cse_reject_weights): evidence just past
# tau -> mild penalty, r >= r_hard -> v5_m_floor. Motivation is
# false-positive cost containment (archived benign max ratios reach 1.89-2.73
# in the AG cells — only the rank cap keeps them unflagged; a borderline
# mis-flag under V4 costs 90% of that client's weight, under V5 nearly
# nothing). V5 is V8's decision layer and its matched-run safety baseline.
#
# V8 (2026-08-09): v8_hmp_cse_propagation_weights. Stage A is
# v5_cse_reject_weights byte-for-byte; its flags become immutable risk seeds
# on a fixed dual-view consensus hypergraph (raw-update view AND label-free
# probe-behavior view must agree on mutual neighbors), denoised by the learned
# HMP-GAE affinity. A non-seed client can be softly penalized only with a
# seed, positive propagated risk, directionally elevated CSE (r > 1), and
# unused rank-cap budget. No seed / no reliable edge / no budget returns V5's
# weight tensor exactly.
#
# Removed 2026-08-11 (history in docs/DECISION.md; code in git history):
# the V1-V3 geometry trust stack (four-signal fusion, robust z-scores,
# suspicion EMA, sigmoid gate) and the V6/V7 arms built on it. V6 could only
# tighten clients CSE already flagged and V7's scalar-isolation conjunct was
# never calibrated; V8 asks the hypergraph a relational question instead.

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def v4_cse_reject_weights(
    local_cse: torch.Tensor,
    data_sizes: torch.Tensor,
    tau_ratio: float = 1.85,
    k_cap: int = 2,
    reject_mult: float = 0.10,
    keep_min: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    V4 rejection rule (2026-07-28): per-client CSE, pool-median normalised,
    rank-capped.

        r_i     = local_cse_i / max(median(local_cse), eps)
        flagged = { top-k_cap clients by r }  INTERSECT  { r > tau_ratio }
        m_i     = reject_mult if flagged else 1.0
        w       = normalize(m_i * n_i)

    Both conditions are required: the rank cap alone is what makes the rule
    zero-false-positive on the archived no-attacker baselines (at tau=1.85
    with NO cap there are 36 benign false-flags; with k_cap=2, zero), and the
    ratio floor alone is what keeps a clean federation from always flagging
    its top-k. Validated by replay over 51 archived runs / 17,850 client-round
    decisions: ~89-90% exact-detection rounds, 0 false positives including all
    5 no-attacker baselines. Residual errors are cold-start false negatives
    (78% in rounds <= 5).

    Design constraints honoured here (rationale: docs/DECISION.md "V4"):
      * local_cse must be the ABSOLUTE per-client statistic (full-test CSE) —
        do NOT pool-relative z-score it: pool-relative scoring has no absolute
        floor and scapegoats the most heterogeneous benign client in a clean
        federation.
      * reject_mult DEFAULTS soft (0.10). Exactly 0.0 — hard removal: a
        flagged client is excluded from the round's aggregate outright — is
        legal since 2026-08-07 as a pre-registered ablation arm for the
        detect-then-remove paper story (docs/DECISION.md "V4-remove"). As a
        default it stays rejected: hard zeroing is FoolsGold's mechanism and
        carries the archive's worst PPL for Qwen Yahoo. Removal is per-round
        (flags re-evaluated each round, no sticky state), and the rank cap
        (< N/2) plus keep_min guarantee the unflagged remainder always
        carries positive mass, so w still sums to 1.
      * tau_ratio=1.85 is pre-registered (zero-FP plateau [1.785, 1.90]); do
        not re-tune it after seeing a confirmatory run.
      * k_cap reuses defense_config.num_byzantine; the rule is sound only for
        #attackers <= k_cap < N/2 (the pool median must be benign-controlled;
        majority-poisoned federations invert it). The runtime validates this
        at construction.

    Args:
        local_cse:  (N,) per-client CSE (absolute scale, NOT z-scored).
        data_sizes: (N,) raw data-size prior n_i (unchanged, uncapped —
                    changing n_i would alter the rule for every run and break
                    FedAvg comparability).
        keep_min:   defensive floor on unflagged clients (structurally
                    guaranteed anyway while k_cap < N/2).

    Returns:
        (weights, diag) where weights is (N,) summing to 1 and diag carries
        'ratio' (N,), 'flagged' (N,) bool, 'multiplier' (N,), 'median' float.
    """
    x = local_cse.detach().to(dtype=torch.float32)
    N = int(x.numel())
    if N == 0:
        raise ValueError("v4_cse_reject_weights received an empty local_cse")
    ds = data_sizes.detach().to(device=x.device, dtype=torch.float32)
    if int(ds.numel()) != N:
        raise ValueError(
            f"data_sizes length {int(ds.numel())} != local_cse length {N}"
        )

    med = x.median()
    ratio = x / med.clamp(min=eps)

    order = torch.argsort(ratio, descending=True)
    max_flags = min(max(0, int(k_cap)), max(0, N - max(1, int(keep_min))))
    flagged = torch.zeros(N, dtype=torch.bool, device=x.device)
    for j in order[:max_flags].tolist():
        if float(ratio[j]) > float(tau_ratio):
            flagged[j] = True

    mult = torch.where(
        flagged,
        torch.full_like(x, float(reject_mult)),
        torch.ones_like(x),
    )
    w = mult * ds
    total = w.sum()
    if float(total) <= 0.0:
        w = mult.clone()
        total = w.sum().clamp(min=1.0)
    w = w / total

    diag: Dict[str, Any] = {
        "ratio": ratio,
        "flagged": flagged,
        "multiplier": mult,
        "median": float(med),
    }
    return w, diag


def v5_cse_reject_weights(
    local_cse: torch.Tensor,
    data_sizes: torch.Tensor,
    tau_ratio: float = 1.85,
    k_cap: int = 2,
    m_floor: float = 0.10,
    r_hard: float = 2.5,
    keep_min: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    V5 graded-rejection rule (2026-08-06): V4's flag decision, byte-identical
    (top-k_cap by ratio AND ratio > tau_ratio), but the flagged-client
    multiplier is a linear ramp in the CSE ratio instead of a constant:

        r_i     = local_cse_i / max(median(local_cse), eps)
        flagged = { top-k_cap clients by r }  INTERSECT  { r > tau_ratio }
        t_i     = clamp((r_i - tau_ratio) / (r_hard - tau_ratio), 0, 1)
        m_i     = m_floor + (1 - m_floor) * (1 - t_i)   if flagged else 1.0
        w       = normalize(m_i * n_i)

    Rationale (see docs/DECISION.md, V5 entry):
      * Flag decision unchanged -> V4's zero-false-positive replay record and
        the tau=1.85 pre-registration carry over untouched.
      * Graded penalty: a client flagged at r just past tau (ambiguous
        evidence) keeps most of its weight; a client at r >= r_hard (clear
        evidence) gets exactly m_floor — for inputs where every flagged
        client sits at r >= r_hard, V5 output equals V4 output with
        reject_mult = m_floor (float-exact; the ramp term is multiplied by
        an exact 0.0).
      * Primary motivation is false-positive cost containment: archived
        benign max ratios reach 1.89 (Llama AG) / 2.73 (Qwen AG, seed 42069)
        — above tau, shielded only by the rank cap. A borderline mis-flag
        costs ~90% of the client's weight under V4 but almost nothing under
        V5's ramp.
      * Calibration (archived V4 runs, steady-state rounds > 5): attacker
        ratio minima are 2.38/2.43 (Llama Yahoo), 3.72 (Llama AG), 4.09
        (Qwen AG), 2.02 (Qwen Yahoo seed 42) — with the pre-registered
        r_hard = 2.5, steady-state attackers overwhelmingly saturate to
        m_floor, so V5's admitted attacker mass stays ~V4-equal by
        construction (CSE risk bounded); only genuinely ambiguous flags
        (cold-start entry rounds, borderline benign) are treated mildly.
      * m_floor plays v4_reject_mult's role and inherits its rules: the
        runtime rejects m_floor <= 0 (hard zeroing is FoolsGold's mechanism,
        rejected) and m_floor is the pre-authorized sweep knob ({0.05, 0.02})
        — under V5 the sweep deepens the penalty ONLY for high-ratio
        (clearly guilty) attackers, a strictly better risk profile than
        sweeping V4's uniform constant.

    Args:
        local_cse:  (N,) per-client full-test CSE (absolute scale, NOT
                    z-scored — same contract as V4).
        data_sizes: (N,) raw data-size prior n_i (unchanged, uncapped).
        tau_ratio:  flag threshold on r (pre-registered 1.85; NOT re-tuned).
        k_cap:      rank cap, reuses defense_config.num_byzantine (< N/2).
        m_floor:    multiplier floor in (0, 1); validated by the runtime.
        r_hard:     ratio at which the ramp saturates to m_floor; must be
                    > tau_ratio (validated here — the ramp divides by
                    r_hard - tau_ratio).
        keep_min:   defensive floor on unflagged clients (as in V4).

    Returns:
        (weights, diag) where weights is (N,) summing to 1 and diag carries
        'ratio' (N,), 'flagged' (N,) bool, 'multiplier' (N,), 'ramp_t' (N,),
        'median' float.
    """
    if not (float(r_hard) > float(tau_ratio)):
        raise ValueError(
            f"v5_cse_reject_weights requires r_hard > tau_ratio; got "
            f"r_hard={r_hard} tau_ratio={tau_ratio}"
        )
    x = local_cse.detach().to(dtype=torch.float32)
    N = int(x.numel())
    if N == 0:
        raise ValueError("v5_cse_reject_weights received an empty local_cse")
    ds = data_sizes.detach().to(device=x.device, dtype=torch.float32)
    if int(ds.numel()) != N:
        raise ValueError(
            f"data_sizes length {int(ds.numel())} != local_cse length {N}"
        )

    # --- Flag decision: byte-identical to v4_cse_reject_weights ---------- #
    med = x.median()
    ratio = x / med.clamp(min=eps)

    order = torch.argsort(ratio, descending=True)
    max_flags = min(max(0, int(k_cap)), max(0, N - max(1, int(keep_min))))
    flagged = torch.zeros(N, dtype=torch.bool, device=x.device)
    for j in order[:max_flags].tolist():
        if float(ratio[j]) > float(tau_ratio):
            flagged[j] = True

    # --- V5 delta: graded multiplier instead of a constant --------------- #
    ramp_t = torch.clamp(
        (ratio - float(tau_ratio)) / (float(r_hard) - float(tau_ratio)),
        min=0.0, max=1.0,
    )
    graded = float(m_floor) + (1.0 - float(m_floor)) * (1.0 - ramp_t)
    mult = torch.where(flagged, graded, torch.ones_like(x))

    w = mult * ds
    total = w.sum()
    if float(total) <= 0.0:
        w = mult.clone()
        total = w.sum().clamp(min=1.0)
    w = w / total

    diag: Dict[str, Any] = {
        "ratio": ratio,
        "flagged": flagged,
        "multiplier": mult,
        "ramp_t": ramp_t,
        "median": float(med),
    }
    return w, diag


def v8_hmp_cse_propagation_weights(
    local_cse: torch.Tensor,
    data_sizes: torch.Tensor,
    propagation_matrix: torch.Tensor,
    tau_ratio: float = 1.85,
    k_cap: int = 2,
    m_floor: float = 0.10,
    r_hard: float = 2.5,
    keep_min: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """V8: V5-safe CSE seeds plus conservative hypergraph risk propagation.

    Stage A is :func:`v5_cse_reject_weights` byte-for-byte.  Its high-
    confidence CSE flags become binary risk seeds.  ``propagation_matrix`` is
    the sub-stochastic, off-diagonal node->hyperedge->node operator built only
    from client pairs that are mutual neighbors in BOTH the raw-update and
    probe-behavior views, then denoised by V8's learned adjacency.

    A non-seed client can be additionally down-weighted only when all of the
    following hold:

      * at least one V5 CSE seed exists;
      * cross-view HMP propagation gives it positive risk;
      * its own pool-median CSE ratio is above 1 (directionally suspicious);
      * the shared rank-cap has budget left after all V5 flags.

    The strength has no new tuned threshold:

        e_cse_i = clip((ratio_i - 1) / (tau_ratio - 1), 0, 1)
        joint_i = propagated_risk_i * e_cse_i
        m_prop_i = 1 - (1 - m_floor) * joint_i

    Thus a weakly elevated CSE or a weak relation causes only a mild penalty;
    both must be strong to approach ``m_floor``.  No seed, no reliable edge,
    or no remaining rank budget returns V5's weight tensor unchanged, exactly.
    This is deliberately NOT neighborhood-median CSE: the benign-controlled
    pool median remains the only denominator and HMP propagates only detached,
    already-confirmed anomaly evidence.
    """
    w5, diag = v5_cse_reject_weights(
        local_cse=local_cse,
        data_sizes=data_sizes,
        tau_ratio=tau_ratio,
        k_cap=k_cap,
        m_floor=m_floor,
        r_hard=r_hard,
        keep_min=keep_min,
        eps=eps,
    )
    x = local_cse.detach().to(dtype=torch.float32)
    N = int(x.numel())
    T = propagation_matrix.detach().to(device=x.device, dtype=torch.float32)
    if T.shape != (N, N):
        raise ValueError(
            f"propagation_matrix shape {tuple(T.shape)} != expected ({N}, {N})"
        )
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    # Re-impose the runtime contract defensively: no self-propagation and at
    # most unit outgoing mass.  Do NOT normalize a sub-unit row back to one:
    # that mass is the learned GAE confidence, and restoring it would make the
    # decoder irrelevant whenever a client has only one surviving relation.
    T = T.clone()
    T.fill_diagonal_(0.0)
    row_sum = T.sum(dim=1, keepdim=True)
    T = torch.where(
        row_sum > 1.0,
        T / row_sum.clamp(min=eps),
        T,
    )

    flagged = diag["flagged"]
    seeds = flagged.to(dtype=torch.float32)
    propagated = (T @ seeds).clamp(min=0.0, max=1.0)
    ratio = diag["ratio"]
    cse_evidence = torch.clamp(
        (ratio - 1.0) / max(float(tau_ratio) - 1.0, eps),
        min=0.0,
        max=1.0,
    )
    joint = (propagated * cse_evidence).clamp(min=0.0, max=1.0)
    joint = torch.where(flagged, torch.zeros_like(joint), joint)

    max_flags = min(max(0, int(k_cap)), max(0, N - max(1, int(keep_min))))
    budget = max_flags - int(flagged.sum())
    propagated_flagged = torch.zeros(N, dtype=torch.bool, device=x.device)
    if budget > 0 and bool(joint.gt(0).any()):
        # Deterministic order: joint evidence desc, CSE ratio desc, index asc.
        order = sorted(
            range(N),
            key=lambda i: (-float(joint[i]), -float(ratio[i]), i),
        )
        chosen = [i for i in order if float(joint[i]) > 0.0][:budget]
        if chosen:
            propagated_flagged[chosen] = True

    diag["seed"] = seeds
    diag["propagated_risk"] = propagated
    diag["cse_evidence"] = cse_evidence
    diag["joint_evidence"] = joint
    diag["propagated_flagged"] = propagated_flagged
    prop_mult = 1.0 - (1.0 - float(m_floor)) * joint
    diag["propagated_multiplier"] = prop_mult

    if not bool(propagated_flagged.any()):
        # Load-bearing regression invariant: V8 with no actionable propagated
        # evidence is V5 element-for-element, not merely numerically close.
        return w5, diag

    mult = torch.where(propagated_flagged, prop_mult, diag["multiplier"])
    ds = data_sizes.detach().to(device=x.device, dtype=torch.float32)
    w = mult * ds
    total = w.sum()
    if float(total) <= 0.0:
        w = mult.clone()
        total = w.sum().clamp(min=1.0)
    w = w / total
    diag["multiplier"] = mult
    return w, diag


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
