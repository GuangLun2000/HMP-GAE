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

> **Scope narrowed 2026-08-07** — see "V4-remove ablation arm" below: 0.0 is
> now legal as an explicit, pre-registered ablation arm on Qwen AG News. As a
> *default*, and on Yahoo, it remains rejected exactly as stated here.

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

## V6 geometry conjunction (2026-08-07)

**Adopted:** `trust_mode='v6_cse_reject_geo'` — V5's flag decision **and** V5's
CSE ramp, both byte-identical (V6 reuses `v5_m_floor` / `v5_r_hard` directly),
times a **one-sided** read-out of the HMP-GAE geometry gate, applied to flagged
clients only. Implemented in
`hmp_gae/trust_scorer.py::v6_cse_reject_geo_weights`:

```text
geo_i = v6_geo_floor + (1 - v6_geo_floor) * gate_i     # gate = V3's sigmoid gate
m_i   = m_cse_i * geo_i     if flagged      else 1.0
```

**Problem it fixes:** the paper (Eq. 21, Algorithm 1 lines 16–17) claims
`alpha = softmax(f_trust(z))`, but V4/V5 execute `alpha = normalize(m_i · n_i)`
with `m_i` from local CSE alone — the hypergraph/VGAE channels are computed,
logged, and multiplied by zero. The archive proves it: trust separation is
bit-identical across BACKBONES (Qwen AG V4 = Llama AG V4 = Qwen AG V5 =
16.1289; Qwen Yahoo V4 = Llama Yahoo V4 = 8.9906), i.e. `alpha` is a
deterministic function of (partition, seed) with no model-dependent — and
therefore no geometric — content.

- **The conjunction is one-sided on purpose.** `geo_i ∈ [v6_geo_floor, 1]`, so
  `m_i ≤ m_cse_i` pointwise and V6's weight for a flagged client is always
  ≤ V5's: CSE cannot regress by construction. This is not stylistic. Offline
  replay of the archived CSVs (per-round flag sets recomputed under the V4/V5
  rule, benign false positives = 0 in all 5 replayed runs, then read the
  same round's discarded `sigmoid_gate`) measures the geometry at a mean of
  **0.766 on confirmed attackers in Qwen AG non-IID V4** (0.649 under V5,
  0.606 Llama Yahoo V4, 0.286 Llama AG V4) — i.e. in the very cell where V4/V5
  beat the clean ceiling, the geometry votes "give this attacker 77% weight".
  Any two-sided fusion hands that weight back.
- `v6_geo_floor = 0.5` is **pre-registered** (2026-08-07, before any V6 run).
  Estimated effect on flagged-attacker weight vs V5, from the archived gate
  means at steady-state `m_cse ≈ m_floor`: −12% (Qwen AG), −20% (Llama Yahoo),
  −36% (Llama AG). Do **not** re-tune after seeing a run. If the logged
  `v6_geo_mult` sits at ≈1.0 every round, the honest report is "the geometry
  did not act, V6 = V5 renamed" — that is a publishable negative result, not a
  reason to lower the floor.
- `v6_geo_floor = 1.0` is **legal and is the regression guard**: the
  `(1 - geo_floor)` factor is a float-exact 0.0, so V6 reproduces
  `v5_cse_reject_weights` element-for-element (tested:
  `test_v6_geo_floor_one_equals_v5`). Run 0 of the experiment plan is this at
  FL scale — `trust_weights.csv` must match the archived V5 companion bit for
  bit before Run 1 (floor 0.5) is worth spending.
- **Unflagged clients keep `m_i = 1.0`, not `gate_i`.** A federation with no
  flags therefore aggregates at exactly `n_k/Σn` (invariant 9,
  `test_no_attack_no_scapegoat`). V3's always-on gate could not hold this: its
  sigmoid never equals 1, so it taxed the most heterogeneous benign client
  even with no attacker present.
- **Side effect to keep in mind:** `graph_min_distinct`, `reject_z_threshold`,
  `soft_reject_k` and `semantic_weight` are diagnostics-only under V4/V5 but
  become live α-affecting knobs under V6. Keep `reject_z_threshold = 2.5` /
  `soft_reject_k = 2.0` so V6's gate is the same object the replay measured.
- A NaN gate maps to 1.0 ("the geometry abstains") = exactly V5 for that
  client. Raising instead would be worse: `HMPGAEDefense.aggregate` catches it
  and drops the whole round to plain FedAvg, losing the CSE rejection too.

### Rejected: reverting to V3, or feeding CSE into V3's gate

Both were measured on the archive before V6 was designed, and both fail:

- **V3 is not "the accurate one".** Over 69 archived runs / 6 cells, plain
  FedAvg-under-attack ranks **1st** by mean accuracy in Qwen AG non-IID and
  Llama Yahoo non-IID, and the no-attack clean ceiling ranks *below* an
  attacked undefended run in two cells. V3 never places 1st in any of its 4
  cells. Median seed-to-seed |Δmean acc| over the archive's 6 reproduction
  pairs is 0.0207, and 42 of 49 adjacent rank gaps (86%) are smaller than that
  — **accuracy cannot order defenses on this benchmark** and is a floor check
  only (extends the 2026-07-29 "The Attack Has No Accuracy Cost" entry from 2
  cells to 4). Meanwhile V3 costs +37.8%/+73.0%/+10.9%/+63.3% mean CSE vs V4,
  and its trust separation is **< 1 in both Yahoo cells** (0.958 / 0.994) —
  attackers outweighing the benign mean for 40 consecutive rounds, i.e. the
  defense pointing backwards.
- **CSE as a fifth V3 channel tops out.** Injecting `z_cse = log(r)/log(tau)`
  into the trust logit and sweeping `cse_weight` peaks at ≈2.0 and then
  *declines* (Qwen AG separation 1.16 → 2.04 → 1.07 at weights 0/2/8). The
  cause is structural, not a tuning miss: `weight_norm = sqrt(Σ w_k²)` grows
  with the new channel and `sus = -s/weight_norm` divides the added signal
  back out, while `reject_z_threshold` is denominated in per-signal robust-z
  units that a single dominant channel can barely exceed. Optimally tuned,
  V3's gate reaches ~2.0x separation where V4/V5 reach 16.13x. **The
  bottleneck is V3's gate/normalisation architecture, not the signal set** —
  do not retry `graph_weight` / `semantic_weight` / `hist_weight_beta` /
  `cse_weight` tuning to rescue it (and see the standing rejections of
  `semantic_weight=2.0` and `hist_weight_beta` tuning).

### What V6 deliberately does NOT do

- No change to `v4_tau_ratio` (1.85), `v5_r_hard` (2.5), or the never-zero
  `m_floor` rule — all pre-registered, all inherited unchanged.
- No coverage-aware reweighting. The one genuinely unsolved failure is Yahoo
  **PPL** (Qwen Yahoo V4: CSE 0.6378 beats the 0.6551 ceiling, but PPL 1209.10
  is worse than both the attack floor 1092.07 and the ceiling 1109.56; Llama
  Yahoo V4: 620.23 vs ceiling 431.07). Under 10-class Dirichlet-0.5, damping
  2 of 7 clients to 0.1× costs **label coverage**, not hallucination — no
  trust architecture fixes it, and V6 (which only ever tightens) will make it
  marginally worse. Needs its own decision entry; still deferred, as in V5.
- No sticky flags / hysteresis and no cold-start holdback (deferred in V5, and
  both would add cross-round state that `HMPGAERuntime.state_dict` would have
  to serialize).
- The **test-set leakage** issue (`evaluate_local_metrics` iterates
  `self.test_loader`, the same set the reported global CSE comes from) is
  acknowledged and **not** addressed here — it is a separate change, gated on
  first archiving `probe_cse` to check whether the 100-sample probe entropy
  discriminates as well as the 1500-sample full-test CSE.

## V4-remove ablation arm: `v4_reject_mult = 0.0` legalized (2026-08-07)

**Adopted (user decision, paper-story motivation):** exactly `0.0` is now a
legal value for `v4_reject_mult` under `trust_mode='v4_cse_reject'` — a
flagged client is excluded from that round's aggregate outright
("detect-then-remove"). This narrows, but does not overturn, the V4 entry's
"Rejected: hard zeroing": 0.0 as a **default** stays rejected, the runtime
default stays 0.10, and the guard still refuses negatives. The authorized
sweep set is now {0.10, 0.05, 0.02, 0.0}.

- **Motivation is narrative, not metrics.** "Detected attackers are removed"
  is a cleaner paper story than "detected attackers keep a 0.10 multiplier".
  The arm exists to test whether that story is free — NOT because a
  measurable CSE gain is expected.
- **Pre-registered expectation (written before the run):** V4 leaves
  attackers ~2.4% of aggregate weight in the Qwen AG cell, and ~half the
  mean-CSE mass accrues in R1–R10 before detection fires (R1–R2 alone ~17%,
  untouched by any multiplier). Extrapolating the measured V6 dose-response
  (−0.41pp attacker share → −0.0008 mean CSE), full removal moves mean/final
  CSE by at most ~0.005 — inside the seed-noise band (median |Δmean CSE|
  13.1% over the 6 archived seed pairs). A within-noise delta in EITHER
  direction is a tie. Success for the story = no metric regresses beyond
  seed noise; the insight either way is what the residual 2% of attacker
  mass is actually worth.
- **Noise-band discipline for PPL (V6 Run-1 lesson):** same-cell seed-pair
  PPL deltas measure +6%/+38%/+65%, so only a PPL / ppl_class_std regression
  beyond that band counts as harm. The V6 criterion of +2% was calibrated on
  the V5-vs-V4 delta (+0.3%) — two nearly-identical trajectories, not two
  independent draws — and is retired as miscalibrated.
- **Scope: Qwen AG News non-IID only.** Extending remove to Yahoo requires a
  new entry first: the 2026-07-29 coverage mechanism (10-class
  Dirichlet-0.5; m=0.10 already pushed Qwen-Yahoo PPL past the attack floor)
  predicts hard removal makes Yahoo PPL strictly worse.
- **Removal is per-round.** Flags are re-evaluated every round; an unflagged
  round re-admits the client. Sticky flags / permanent blacklisting remain
  deferred (V5 entry) — with archived flag stability (97/100
  attacker-rounds, 0 benign) per-round removal approximates permanent
  exclusion without new cross-round state or checkpoint changes.
- **Unchanged:** `v5_m_floor` and `v6_geo_floor` keep their open lower
  bounds (no 0.0) — under the V5 ramp a floor of 0 zeroes only saturated
  (r ≥ r_hard) clients, a different mechanism that gets its own decision if
  ever wanted. `tau=1.85` pre-registration, rank-cap semantics, the
  no-`_zscore` rule, and AugMP incompatibility all carry over.
- **Companion & controlled delta:** archived Qwen AG non-IID V4 run
  20260729 (seed 42); the ONLY moving axis is `v4_reject_mult` 0.10 → 0.0.
- Tested: `test_v4_hard_removal_arm` (flagged mass exactly 0, survivors
  renormalise to the n_k prior, clean rounds bit-identical to soft V4,
  guard accepts 0.0 / refuses negatives).

## V7 `v7_cse_reject_corrob`: iso-corroborated cold-window rejection (2026-08-08)

**Adopted (design + implementation; NOT yet runnable):** V7 re-admits the
hypergraph to the *flag decision* — the one place V6 structurally cannot
reach — but only inside the CSE cold-start window and only as a
corroborator. `trust_mode='v7_cse_reject_corrob'`
(`trust_scorer.v7_cse_reject_corrob_weights`):

- **Stage A = V6 byte-identical** (V4 flag rule → V5 ramp → V6 geometry
  factor). Outside the window, or whenever Tier 2 stays empty, V7 *returns
  V6's output tensor unchanged* — bit-identity by construction, tested.
- **Tier 2 (the V7 delta), armed only for logged rounds
  `v7_round_min..v7_round_max` (1-indexed, same units as the archives'
  `round` channel):** a client NOT CSE-flagged is corroboration-flagged iff
  `r > v7_tau_lo` **AND** `raw graph_residual >= v7_iso_min`, subject to the
  shared rank-cap budget (`num_byzantine − |CSE flags|`; CSE flags always
  take precedence and can never be displaced or re-ranked — the set form is
  normative; a seemingly-equivalent "rescaled statistic" form was shown to
  re-order CSE flags and is rejected). Candidate ranking is pre-registered:
  iso desc, then ratio desc, then index asc (the channel is quantized at
  N=7, ties are near-certain). Flagged multiplier = constant
  `v7_corrob_mult`; everyone else keeps the literal 1.0.

**Honest mechanism statement (wording is load-bearing):** V7 is an
*iso-corroborated threshold discount*, not "the hypergraph detects in the
cold window". Geometry never flags alone; R1–R2 stay untouched by V7's own
physics (attacker r ≈ 1 there fails `tau_lo`) — which is why the window
opens at round 3, where the archive puts 78% of V4's residual false
negatives (rounds ≤ 5) and ~half the mean-CSE mass (R1–R10). The
hypergraph's necessity claim is: **without the iso conjunct, no threshold
below 1.85 survives zero-FP replay** (archived clean-run benign ratio max
1.7833 > any useful `tau_lo`), so the geometry is what makes a sub-1.85
threshold safe. Every Tier-2 flag with `r < 1.785` (bottom of the archived
V4 plateau) is a detection no legal CSE threshold could have made —
replayable, countable, hypergraph-attributable.

Design choices and their reasons:

- **The conjunct is the RAW `graph_residual`, never the sigmoid gate.** The
  gate's suspicion input is a weighted sum of `_zscore`d (pool-relative)
  channels; giving it flag authority is exactly what the V4 no-`_zscore`
  rule forbids. The raw channel is bounded [0,1], absolute-scaled, quantized
  to multiples of 1/(N−1), and archived per round as `residual`. The gate
  keeps its V6 role: penalty *magnitude* on Stage-A-flagged clients.
- **`v7_iso_min` sits at an inter-level midpoint** (7/12 = only reach ≤ 2
  flags at N=7/knn_k=2; candidate 5/12 also admits reach = 3) — never ON a
  quantization level, where float dust decides flags.
- **Constant penalty, not gate-modulated:** the archived gate averages 0.766
  on confirmed Qwen-AG attackers, so a gate-modulated Tier-2 penalty would
  trim ~12% and vanish inside seed noise ("detection without penalty").
  `v7_corrob_mult` ∈ (0,1), never 0 — this tier's evidence is weaker than a
  full CSE flag, so hard removal is rejected *a fortiori*.
- **Resolution abstention:** when `graph_min_distinct` gates the graph
  channel (quantization-noise round), Tier 2 abstains entirely (`iso=None`
  → V6 exactly). A degenerate channel must degrade to the validated rule.
- **Honest demotion of the clean-federation property:** because
  `tau_lo < 1.7833`, clean exactness is no longer purely structural as in
  V4–V6. Unflagged clients still take the literal 1.0, but "no clean flags"
  now rests on the replay-verified premise that no clean benign client-round
  jointly crosses `(tau_lo, iso_min)` inside the window. That premise is a
  pre-registered PASS criterion of the replay, with margins reported per
  cell — and it is honestly weaker evidence than V4's: the armed window
  holds only ~a few hundred clean client-round decisions, not 17,850.
- **`graph_residual` is computed in the TRAINED node-encoder embedding**,
  not raw update geometry — its cross-seed/backbone/round stability is a
  replay PASS criterion (round-stratified histograms), not an assumption.

**Pre-committed calibration ([replay_v7_calibration.py](../replay_v7_calibration.py),
stdlib-only, runs on the archived `*_results.json`):** grids `tau_lo` ∈
{1.30..1.80 step 0.05}, `iso_min` ∈ {5/12, 7/12}, `round_max` ∈ {5, 8, 10},
`round_min` = 3. PASS = a ≥ 2-point contiguous zero-FP `tau_lo` plateau
(cap-free AND budgeted counts both 0 over all admitted benign client-round
decisions — the budget is an exposure bound here, NOT V4's structural
shield, since Tier 2 ranks by iso in the regime where attacker
iso-dominance is the unproven hypothesis). Selection is fully deterministic
(windows NEST — the Tier-2 flag set at a smaller `round_max` is a subset of
a larger one's, so zero-FP at the largest passing window implies zero-FP at
every smaller candidate): (1) the most conservative `iso_min` with a
passing plateau at any `round_max`; (2) the CERTIFIED plateau is the one at
the largest passing `round_max`; (3) `tau_lo` = that plateau's midpoint
rounded down to the grid (lower-middle element); (4) `round_max` = the
smallest candidate covering ≥ 80% of the certified recall at the selected
`(tau_lo, iso_min)`. An always-on arm (no window) is replayed
for documentation: its expected steady-state FP failures justify the
window's existence (the archive's top-1 geometry suspect is BENIGN in 45/50
Llama-Yahoo and 34/50 Qwen-Yahoo steady-state rounds). Kill criteria:
empty/point plateau, zero recall everywhere, or benign iso distributions
shifting ≥ one quantization level across cells at matched rounds → V7 is
recorded here as a negative result; the grids are NOT widened.

**2026-08-08 post-implementation audit amendments (pre-data — nothing had
been replayed when these were written; they tighten the calibration
contract, they do not react to results):**

- *The replay now mirrors the resolution abstention.* The initial script
  counted Tier-2 decisions in rounds where `graph_min_distinct` gated the
  graph channel — rounds where the live rule abstains (`iso=None`), and the
  archive's MAJORITY case (the channel degenerates in 33–44 of 50 Yahoo
  rounds). Gated rounds are now excluded from every statistic (FP, recall,
  plateau, always-on arm) and the per-run excluded count is reported in
  Phase 0, keeping the admitted-decision denominator auditable. Archives
  predating the `graph_channel_gated` key reconstruct gating from the
  persisted residual levels with the archived `graph_min_distinct`
  (default 4).
- *Pre-V4 archives are admitted by exact reconstruction.* Rounds lacking
  as-run `v4_ratio`/`v4_flagged` (written before the V4 commit,
  2026-07-28 — which includes the 5 no-attacker baselines behind the
  zero-FP claim) recompute `ratio_i = cse_i / max(median_lower(cse), 1e-6)`
  from the per-round `local_cse` dict, where `median_lower` is
  torch.median's lower-middle convention (NOT `statistics.median`, which
  averages at even N), and re-derive the as-run V4 flag set by the V4 rule
  (rank by ratio desc / index asc, cap `min(k_cap, N − max(1, keep_min))`,
  flag only `r > 1.85`). This is a float64 reconstruction of a float32
  rule; the ~1e-6 relative error is four orders of magnitude below the
  0.05 `tau_lo` grid step. Rounds without a complete `local_cse` dict are
  excluded and counted.
- *The replayed budget now carries the `keep_min` clamp*
  (`min(k_cap, N − max(1, keep_min)) − |CSE flags|`), matching
  `trust_scorer` exactly instead of the bare `k_cap`.
- *Selection was made fully deterministic* (the nested-window rule above —
  the original "plateau midpoint" left WHICH `round_max`'s plateau
  unspecified when they differ).
- *Exit codes are now load-bearing*: 0 = pass with a printed selection,
  2 = no usable clean baseline (zero-FP premise uncertifiable), 3 =
  negative result — so a notebook cell keying on exit status cannot
  misread a kill as success. Phase 5 additionally emits the
  round-stratified benign iso medians per cell that back the stability
  kill criterion (previously stated here but not computed anywhere).

**Current knob values in `main()` (`v7_tau_lo=1.40`, `v7_iso_min≈7/12`,
`v7_corrob_mult=0.5`, `v7_round_min=3`, `v7_round_max=10`) are PROVISIONAL
placeholders — no V7 training run may launch before the replay passes and
this entry is amended with the selected constants.** `v7_round_max=0` is
the wiring-regression arm (V6 bit-identical; the geo_floor=1.0 / W=0
degeneracy family) and must be Run 0 of any V7 experiment plan.

**Threat-model scope, stated up front:** fixed attack-from-round-1,
non-adaptive label-flip. Known counterplay recorded now, not after review:
(a) delayed-onset attackers re-open the cold hole past the window (the
archive cannot test this — all archived attacks start at R1); (b) the
roadmapped cosine+norm stealth attacker nulls the iso channel; (c)
update-forging (AugMP) stays incompatible with the whole CSE family; (d)
*mutual-selection reach inflation*: two colluding attackers that select
each other with DISTINCT benign second picks reach 3 peers (residual 1/2)
— under the preferred 7/12 floor, admitted only by the 5/12 candidate.
`reach` counts ALL co-members (including the other attacker), so "isolated
from the benign majority" is an interpretation, not the statistic; Phase 3
of the replay reports the recovered attackers' in-window residual levels,
so which regime the archived label-flip attackers actually occupy is
measured, not assumed. Two upstream channel edge cases are recorded but
deliberately left unfixed mid-campaign (changing the hypergraph
construction now would desynchronize live runs from the archived channel
the calibration replays): a NaN client update inverts the signal
(`torch.topk` ranks NaN similarities on top, so every node selects the NaN
client and its residual collapses to 0.0 — but a NaN-update client has a
degenerate local model, which the CSE tier flags), and exact-duplicate
updates fall back to topk's index order (`graph_min_distinct` gates such
degenerate rounds). The
"geometry owns update-forging / no-server-data" division-of-labor design
(geometry EMA with independent flag authority) is the natural V8 direction
but requires a *log-only phase first* (a raw-projection isolation statistic
is not in the archive) and its own entry — explicitly deferred.

**Rejected within this design:**

- *Gate-as-flag-authority* — C2 violation (pool-relative z inside), plus
  config-heterogeneity makes the archived gate uncalibratable across runs.
- *Neighborhood-median CSE* (hypergraph inside the statistic) — at N=7,
  knn_k=2 a hyperedge has 3 members; {A1, A2, benign} is attacker-majority
  and the local median loses the benign-controlled-denominator guarantee.
  One-sided fixes fail both ways (min() kills the 1.7833/1.85 headroom,
  max() helps colluders). Pool median stays.
- *Gate-modulated Tier-2 penalty* — see above; ~0.88 effective multiplier is
  outcome noise.
- *Sticky flags / permanent blacklisting* — still deferred (V5 entry).

Falsification statistic: the per-run count of `v7_corrob_flagged`
client-rounds. Zero everywhere = the cold-window tier never acted and V7 is
V6 renamed — report it; do not widen the window or lower the floors.

Tested (`tests/test_trust_robustness.py`): `test_v7_window_off_equals_v6`
(bit-identity outside the window / at `round_max=0` / on iso=None),
`test_v7_corroborated_flag_rule` (both floors necessary, budget sharing,
CSE-flag priority, tie-break, NaN abstention),
`test_v7_clean_federation_identity` (n_k prior exactly; the fixture encodes
the elevated-ratio benign client protected by the iso floor),
`test_v7_monotone_vs_v6` (m_V7 ≤ m_V6 pointwise; CSE flags untouched;
guards), `test_runtime_v7_cse_reject_corrob` (iso wired from the logged raw
`residual`, 1-indexed window arithmetic, resolution abstention end-to-end,
construction-time guards, stale-key inertness under V4/V5/V6, and — added
by the 2026-08-08 audit — non-default-knob plumbing with both window edges
inclusive, killing `round_num+2` / exclusive-bound wiring mutants),
`test_defense_facade_local_cse_guard` (the loud-crash contract at the
facade, including a whitespace-padded `trust_mode` — pins the
`.strip().lower()` normalization being identical across runtime, facade
and `Server._needs_local_cse`). The pure-function tests additionally pin,
per the same audit: the normative `w = normalize(m_i · n_i)` against a
`w = m_i` mutant (non-uniform `n_k` fixture), strict `>` at `r == tau_lo`
vs inclusive `>=` at `iso == iso_min` (dyadic exactly-representable
fixtures), and the index-ascending final tie-break key under a full
`(iso, ratio)` tie.

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
