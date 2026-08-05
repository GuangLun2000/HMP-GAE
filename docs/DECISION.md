# DECISION.md — settled design decisions (do not re-litigate without new evidence)

> Created in-repo 2026-07-28 during the V4 change-set. The V4 coding brief
> ([HMP-GAE-V4-coding-brief.md](../HMP-GAE-V4-coding-brief.md)) references an
> earlier DECISION.md kept alongside the results archive (outside this repo);
> the entries below are the ones that constrain the code in this repo. Add new
> entries here when an alternative is tested and rejected, so it is not retried.

## V4 rejection signal (2026-07-28)

**Adopted:** `trust_mode='v4_cse_reject'` — per-client **full-test local CSE**,
pool-median normalised (`r_i = cse_i / median(cse)`), flag the top-`num_byzantine`
clients by `r` with `r > v4_tau_ratio`, soft-reject (`× v4_reject_mult`), then
data-size FedAvg. Implemented in `hmp_gae/trust_scorer.py::v4_cse_reject_weights`.

- `v4_tau_ratio = 1.85` is **pre-registered** (zero-FP plateau [1.785, 1.90]
  over 51 archived runs / 17,850 client-round decisions, including all 5
  no-attacker baselines). Do **not** re-select it after seeing a confirmatory
  run. Headroom is thin (max clean-run ratio 1.7833) and Qwen-only; a Llama
  no-attack baseline is the outstanding validation (brief §5, Run 0).
- The **rank cap is load-bearing**: at tau=1.85 with no cap there are 36 benign
  false-flags; with `k_cap = num_byzantine = 2`, zero. The cap reuses
  `defense_config.num_byzantine` and the rule is sound only for
  `#attackers ≤ num_byzantine < N/2` (validated at runtime construction).

### Rejected: local CSE through the unchanged MAD-z pipeline

Feeding local CSE into the existing pool-relative stack
(MAD-z → EMA(0.6) → sigmoid at 2.5 → × n_k) **fails the no-attack baselines**:
MAD-z has no absolute floor, so in an attack-free federation the most
heterogeneous benign client is driven to the clip — `sus_z` exceeds 2.5 in
31/50 and 34/50 clean rounds with a benign gate reaching 0.000, failing
`tests/test_trust_robustness.py::test_no_attack_no_scapegoat`. The
ratio-to-median form has an absolute floor and produced 0 false flags across
all 5 clean baselines. **Do not route the V4 signal through `_zscore`.**

### Rejected: hard zeroing (`v4_reject_mult = 0.0`)

Hard discard is FoolsGold's mechanism; FoolsGold on Qwen Yahoo non-IID has the
archive's worst PPL (1549.30) and worst per-class spread. Since lowering PPL is
half the goal, rejection is soft (0.10). `0.10` itself was not separately
validated — it is the first knob to sweep if a confirmatory run shows residual
attacker mass.

### Design decision (d): which CSE statistic feeds the rule

**Chosen: option (i)** — the exact statistic validated on the 51-run archive:
`Server.evaluate_local_metrics` over the **full test_loader**. The claimed ~2×
per-round eval cost does not materialise: aggregation only mutates
`server.global_model`, never `client.model`, so the per-client local eval is
**hoisted before** `aggregate_updates` and its values reused for the round log
(one eval per round, same as before, when `eval_local_every_n_rounds == 1`).
The probe-tensor alternative (option (ii)) is a *different* statistic with
10 samples/class on Yahoo — the same sampling noise that killed `sem_div` —
and nobody has shown it agrees at tau=1.85. The runtime still logs the probe
entropy (`probe_cse`) alongside `v4_cse` whenever the probe exists, so any V4
run doubles as the (i)-vs-(ii) agreement check.

### Known scope limits (stated, not hidden)

- V4 raises `RuntimeError` when combined with update-forging attackers
  (`crafts_update`, i.e. AugMP): local CSE evaluates `client.model`, which
  such attackers leave looking benign.
- Circularity: the trust signal and the reported global CSE share
  `test_loader`. Mitigation candidates (disjoint server pool,
  update-direction self-angle) are future work — see brief §4.
- When V4 is enabled, per-client local eval runs **every round** regardless of
  `eval_local_every_n_rounds` (the rule needs it pre-aggregation).

## V5 graded rejection (2026-08-06)

**Adopted:** `trust_mode='v5_cse_reject'` — V4's flag decision **byte-identical**
(top-`num_byzantine` by ratio AND `r > v4_tau_ratio`), but the flagged-client
multiplier is a **linear ramp in the CSE ratio** instead of the constant
`v4_reject_mult`. Implemented in `hmp_gae/trust_scorer.py::v5_cse_reject_weights`:

```text
t    = clamp((r - tau) / (v5_r_hard - tau), 0, 1)
mult = v5_m_floor + (1 - v5_m_floor) * (1 - t)
```

- **Primary motivation is false-positive cost containment**, not per-cell
  tuning: archived benign max ratios reach 1.89 (Llama AG) and 2.73 (Qwen AG,
  seed 42069) — above tau, shielded only by the rank cap. A borderline
  mis-flag costs ~90% of the client's weight under V4 but ~5-20% under the
  ramp. Restores V3's graded-response virtue on V4's absolute-scale signal.
- `v5_r_hard = 2.5` is **pre-registered** (2026-08-06, before any V5 run),
  calibrated from the archived V4 runs' steady-state (rounds > 5) attacker
  ratio minima: 2.38/2.43 (Llama Yahoo 2atk/1atk), 3.72 (Llama AG), 4.09
  (Qwen AG), 2.02 (Qwen Yahoo s42). Steady-state attackers therefore
  saturate to `m_floor` — for those rounds V5 is float-exactly V4 with
  `reject_mult = m_floor` (tested: `test_v5_saturation_equals_v4`) — so the
  admitted attacker mass stays ≈V4-equal by construction and the CSE risk of
  the softer ramp is bounded. Do **not** re-tune after a confirmatory run.
- `v5_m_floor = 0.10` inherits `v4_reject_mult`'s role and rules: never 0.0
  (hard zeroing rejected, see V4 entry), and it remains the pre-authorized
  sweep knob ({0.05, 0.02}) — under V5 the sweep deepens the penalty ONLY
  for high-ratio (clearly guilty) attackers, a strictly better risk profile
  than sweeping V4's uniform constant.
- Everything else carries over from V4 unchanged: tau=1.85 pre-registration,
  rank cap semantics (`num_byzantine < N/2`, construction-validated), the
  no-`_zscore` rule, pre-aggregation full-test local CSE (`_needs_local_cse`),
  AugMP incompatibility, and the every-round local eval behavior.

### Rejected: alternative ramp shapes

Exponential decay (`exp(-beta(r-tau))`) and rational (`(tau/r)^p`) forms were
considered: both need a shape parameter with no natural calibration anchor and
neither reproduces V4 exactly in the saturated regime (the equivalence that
makes V5's CSE risk arguable a priori). The linear ramp has two interpretable,
replay-calibratable parameters and exact V4 saturation equivalence.

### What V5 deliberately does NOT do

- No coverage-aware reweighting and no per-class/head-row trust: the Y18
  regression (Qwen Yahoo V4 acc −2.1pp tail-mean below V3, PPL above floor
  AND ceiling) is a **label-coverage** problem orthogonal to penalty
  softness — Qwen-Yahoo attackers sit at ratio ≈3.5 median, NOT near tau, so
  the ramp barely touches them. Y18 gets its queued diagnosis first; any
  coverage mechanism is a separate, new decision.
- No sticky flags / hysteresis (variance lever, needs its own replay
  validation of a `tau_hold`), no cold-start holdback (new prereg + clean
  ceiling reruns). Both remain future work.

## C1 z-score hygiene (2026-07-28)

- `_zscore` MAD degeneracy guard is **relative** (`scale < 1e-3·max|x|`), with
  std fallback, and zeros the channel when std is also degenerate. The old
  absolute guard (`< 1e-6`) never fired on recon_residual (spread ~1e-4), so
  the ±10 clip pinned attacker AND benign at the bound in 100% of saturated
  rounds — an exact rank tie.
- `zscore_clip` applies **post-fusion** (to `s`), per-channel `*_z` diagnostics
  are unclipped.
- `graph_min_distinct` (config, default 0 = off; canonical config 4) zeroes the
  graph channel in rounds where it resolves too few distinct values (knn_k=2,
  N=7 → 4-5 discrete levels; MAD exactly 0 in 33-44/50 Yahoo rounds). The
  gated channel's weight is **dropped from `weight_norm`** so the effective
  gate threshold does not silently shift. Do **not** re-tune `knn_k` instead.

## Banned knobs (carried over from the archive DECISION.md, per the V4 brief)

- `hist_weight_beta` tuning and `semantic_weight = 2.0`: proven-null Yahoo
  fixes. (`hist_dev` is identically zero in all 350 client-rounds of all
  archived runs — the "4-channel" stack is effectively 3 channels.)
- `gate_rezscore = true`: on Yahoo the top-1 suspect is a **benign** client in
  45/50 (Llama) / 34/50 (Qwen) rounds; re-z-scoring ejects the most
  heterogeneous benign client.
- `server_lr < 1`, norm clipping, LoRA r/alpha reduction, early stopping as
  PPL levers: shrinkage/under-training artifacts that confound the
  attacker-mass measurement.
- Capping / sqrt-ing the `n_k` data-size prior: attacker mass reaches ~0
  without it, and changing it breaks FedAvg comparability for every run.
- Adaptive per-channel weighting: improves Llama Yahoo but degrades Llama AG
  (−79% → −45% attacker mass), the paper's strongest block. Deferred.
