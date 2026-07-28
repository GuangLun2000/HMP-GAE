# HMP-GAE V4 — Code Modification Brief

**Repo:** `/Users/lancecai/Documents/GitHub/HMP-GNN` (branch `main`, HEAD `33d40dd`, clean tree, remote `github.com/GuangLun2000/HMP-GNN`)
**Goal:** lower global CSE **and** PPL on Yahoo Answers non-IID without regressing AG News.
**Evidence base:** all claims below are replayed from 51–55 archived runs (17,850 client-round decisions) in the results archive, and every number was independently re-verified by a second adversarial pass. Where a claim is *not* verified, it says so explicitly.

> **Environment note:** `import torch` fails on this Mac. Runnable locally: `python3 check_docs.py`, `python3 -m compileall -q .`. `tests/test_trust_robustness.py` needs torch → Colab/CI only. Repo AGENTS.md: never run `main.py` locally; training is Colab A100 only. Do not `git add`/`commit` — the user commits.

---

## 1. Diagnosis (verified)

### 1.1 The trust signal is destroyed in two places on Yahoo

Reconstructed pipeline algebra (fitted from archived CSVs to max abs err ~1e-4 — use this as ground truth):

```
z_c           = clip( (x - median_7(x)) / (MAD_7(x)/0.6745 if MAD>0 else std_7(x)), ±10 )
comb          = semantic_weight*sem_div_z + graph_weight*graph_residual_z
                + residual_weight_alpha*recon_residual_z + hist_weight_beta*hist_dev_z
trust_logit_s = -comb                                     # UN-smoothed
sus_z         = 0.69171 * EMA_{β=0.6}(comb)               # only sus_z is smoothed
sigmoid_gate  = sigmoid( soft_reject_k * (reject_z_threshold - sus_z) )
trust_weights = normalize( n_k * sigmoid_gate )           # n_k = constant client data share
```

**Loss #1 — `sem_div` dies on 10 classes.** Per-round rank AUC: Llama AG 0.972 → Llama Yahoo **0.576**, Qwen Yahoo **0.546**. Attacker sem_div falls 1.68→1.06 while benign rises 0.273→0.974 (+257%): the attacker signature sits inside the benign heterogeneity envelope. On Llama Yahoo, benign C2's mean sem_div (1.663) is **higher than both attackers** (1.008 / 1.120).
Mechanistic root: `semantic_probe_size=100` is stratified over `num_labels`, i.e. 25/class on AG but only **10/class on Yahoo**. Gap/temporal-noise ratio collapses 4.28 → 0.32 / 0.06 while per-client temporal sd is unchanged.
MAD z-scoring is **not** the culprit — it is rank-preserving within a round (raw and `_z` AUCs identical to 3 dp).

**Loss #2 — the gate + `n_k` prior invert the decision (fatal).** `sus_z` still carries AUC 0.716/0.648, but final `trust_weights` AUC is **0.389 / 0.434 — below chance**. Causes: (a) mean attacker `sus_z` is 1.29/0.87 vs a fixed `reject_z_threshold=2.5`, so the gate's total attenuation authority is only 0.947/0.965; (b) Dirichlet(0.5) handed the two attackers **30.77%** of the data (prior odds 1.111). 1.111 × 0.947 = 1.046 > 1. Net attacker aggregation mass reduction vs plain FedAvg: **−4.1% (Llama Yahoo), −1.4% (Qwen Yahoo)** vs **−78.0% (Llama AG)**.

### 1.2 Two corrections to earlier internal analysis — do not chase the wrong client

- **Benign C0 is NOT the trust misfire.** C0 has the lowest final weight (0.0349) but the **highest gate (0.9992)** — its low weight is entirely the `n_k` prior (0.0309, only 309 samples). The genuine misfire is **benign C2**: gate 0.5517, the lowest of all 7 clients, *below both attackers* (0.8681/0.8435). On Llama Yahoo C2 is the #1 suspect in 44/50 rounds.
- **V3 did improve Yahoo over V2** (Qwen Yahoo final CSE 0.6915 vs V2's 0.8491–1.1217); it regressed AG (0.0441→0.0607). Do not write V3 up as a Yahoo failure — it is an A/B with trade-offs.

### 1.3 PPL: what it actually measures

Found in the run logs (`result.md`, "V2 M7: Perplexity evaluation (backbone transfer to CausalLM)"): the trained backbone is transplanted into a fresh `AutoModelForCausalLM` built from the **original pretrained base**, and NLL is computed with the **untouched pretrained `lm_head`**. So **PPL measures backbone drift from the pretrained model and carries no label-semantic information.**

Consequences (verified):
- Attack sensitivity is weak and unreliable: Qwen AG non-IID clean→attacked ppl +14.2% (CSE +65%); Qwen Yahoo non-IID **−1.6%** (CSE +212%). On the group that matters most, PPL is *not* an immunity metric.
- Per-class PPL profile is a **dataset+model fixed effect**, not a lever: rank-1 fit explains 61–83% of log-PPL variance; class 9 is max in 10/10 Qwen Yahoo non-IID runs; Spearman between runs +0.939. "Reduce class spread" is **not** a causal lever (`ppl_class_std` correlates r=+0.822 with ppl_mean only because the whole curve slides together).
- **The one real, legitimate PPL lever is reducing admitted attacker weight.** Within the 5-run Qwen Yahoo HMP family, r(ppl_mean, steady attacker weight share) = **+0.802**, slope 749 PPL per unit share. Cross-defense: Llama AG FedAvg 0.286→72.12 vs HMP 0.030→51.73 (−28.3%). Among keep-all aggregators (excluding Krum/FLTrust), r(ppl, mean_cse) = **+0.74 to +0.86** in 4 of 5 groups — CSE and PPL move *together*. The single exception is Qwen Yahoo non-IID (−0.54), precisely the group where trust separation fails.

**→ Fixing the trust signal is simultaneously the CSE fix and the PPL fix. There is no second PPL lever worth taking.**

### 1.4 Replicate noise floor

Four near-identical Qwen Yahoo HMP-v2 runs span 1082.49–1153.74 → **PPL differences under ~7% within a group are not effects.**

---

## 2. The change-set

Implement in this order. C1 is independent and can ship immediately; C2 is the main change.

### C0 (prerequisite, no code logic) — restore the config arm

`main.py` config dict (≈1240–1660) is currently `defense_method='foolsgold'`, `num_clients=6`, `num_attackers=1`. A run launched from this tree would look valid but be un-comparable to every archived 7-client/2-attacker HMP-GAE run.

Restore: `defense_method='hmp_gae'`, `num_clients=7`, `num_attackers=2`, matching `experiment_name`. Route the V4 delta through `COLAB_CONFIG_OVERRIDES` / `run_suite()` rather than editing notebook cells (per repo AGENTS.md).
Also confirm with the user **which repo Colab pulls**: archived runs used `github.com/sileneer/HMP-GNN`; the local clone and its notebook point at `github.com/GuangLun2000/HMP-GNN`.

### C1 (ship independently) — fix the `recon_residual` degeneracy

**File:** `hmp_gae/trust_scorer.py`, `_zscore` (lines 71–112).

`recon_residual` is near-degenerate (all clients within [0.491, 0.494], spread ~1e-4), so its MAD is ~1e-4 and the guard `if float(scale) < eps` with `eps=1e-6` never fires. Unclipped z would be 17.9/36.4; the ±10 clip is binding by 2–4×. **In 100% of rounds where an attacker is pinned at +10, a benign client is also pinned at +10** — the ordering becomes an exact tie. Saturation rate: 63/350 cells (Llama Yahoo), 75/350 (Qwen Yahoo). Each saturated cell injects 0.3×10 = 3.0 into `comb`, comparable to the entire attacker/benign signal. On Qwen AG v3 this channel's AUC is **0.115** (strongly anti-discriminative).

Changes:
1. Replace the absolute degeneracy guard with a **relative** one: fall back to std (or zero the channel) when `scale < 1e-3 * max(|x|)`.
2. Move `zscore_clip` to **post-fusion** instead of per-channel.
3. Same class of problem in `graph_residual`: with `knn_k=2` and 7 clients it takes only 4–5 discrete values (multiples of 1/6); `MAD_7 == 0` in 33/50 (Llama Yahoo) and 44/50 (Qwen Yahoo) rounds, silently falling back to std and dropping the 0.6745 factor. It resolves ~2.2–2.5 distinct levels among 7 clients yet still carries `graph_weight=1.0` and ~25% of fused variance. **Gate the channel out when its within-round resolution is degenerate** (e.g. fewer than 4 distinct values), rather than re-tuning `knn_k` blindly.

This is cheap, independent of C2, touches no banned knob, and improves the diagnostic figures the paper already ships.

### C2 (main change) — V4 rejection rule: per-client CSE, pool-median normalised, rank-capped

**Replaces** the four geometry channels *as the rejection signal*. Keep computing and logging them as diagnostics and as the adaptive-attacker fallback.

**Why this form and not the obvious alternative** (this was tested and rejected — record it in DECISION.md so it is not retried): feeding local CSE through the *existing unchanged* pipeline (MAD-z → EMA 0.6 → sigmoid at 2.5 → ×`n_k`) **fails the no-attack baselines** — MAD-z has no absolute floor, so in an attack-free federation the most heterogeneous benign client is driven to the clip: `sus_z` exceeds 2.5 in 31/50 and 34/50 clean rounds, with a benign gate reaching **0.000**. It would fail the existing regression test `tests/test_trust_robustness.py:225 test_no_attack_no_scapegoat`. The ratio-to-median form below has an absolute floor and produces **0 false flags across all 5 clean baselines**.

```python
# ---- constants ----
TAU        = 1.85            # ratio threshold; zero-FP plateau [1.785, 1.90]
K_CAP      = num_byzantine   # = 2. REUSE the existing key; do NOT add a hyperparameter.
KEEP_MIN   = 1               # existing key
REJECT_MULT = 0.10           # NOT 0.0 — see "soft vs hard" below
EPS        = 1e-6

# ---- per round ----
cse = [per_client_semantic_entropy(model_i, probe) for i in range(n)]
med = median(cse)
r   = [cse[i] / max(med, EPS) for i in range(n)]

order   = argsort(r, descending=True)
flagged = {j for j in order[:K_CAP] if r[j] > TAU}     # BOTH conditions required

m = [REJECT_MULT if i in flagged else 1.0 for i in range(n)]
w = normalize([m[i] * client_sizes[i] for i in range(n)])
```

**Validated performance** (unchanged constants, no per-run tuning, replayed over every archived run with a `local_cse.csv`):
- 2193–2219 / 2460 exact-detection rounds (~89–90%)
- **0 false positives** in 17,850 client-round decisions across 51 runs, including all 5 no-attacker baselines (250/250 clean rounds correct)
- All residual errors are false negatives, concentrated in cold start: round 1 fails in essentially every run, 78% of failures in rounds ≤5, 97.7% in rounds ≤10
- Transfers across aggregators (FedAvg/Krum/Multi-Krum/FoolsGold/FLTrust/coord-median), both models, IID and non-IID, 6 and 7 clients, 1 and 2 attackers, and both flip ranges

**Simulated attacker aggregation mass** (steady state, using each run's own `n_k`):

| Run | now | V4 | cumulative admitted poison (50 rounds) |
|---|---|---|---|
| Llama Yahoo | 0.2949 | ~0.011 | 14.747 → **0.615** |
| Qwen Yahoo | 0.3032 | ~0.026 | 15.159 → **1.591** |
| Llama AG | 0.0421 | ~0.006 | 2.105 → **0.191** |
| Qwen AG v3 | 0.1679 | ~0.006 | 8.397 → **0.327** |

Quote the **cumulative** figures, not just steady state — they are the evidence that this does *not* reproduce FoolsGold's warmup-leakage pattern.

#### C2 design decisions the coding agent must honour

**(a) Soft, not hard rejection.** Use `REJECT_MULT ≈ 0.10`, not `0.0`. Rationale: hard zeroing is exactly FoolsGold's mechanism, and per-run class-profile distortion (RMS log residual from the group rank-1 fit) shows hard-discarding aggregators bend the PPL profile — FoolsGold on Qwen Yahoo non-IID scores 0.2813 with `ppl_class_std` 755.94 and `ppl_mean` **1549.30 (worst in the archive for that group)**, versus every HMP-GAE run at 0.0431–0.0908 / 344.41 / 1061.74. Since the goal is to lower PPL as well as CSE, do not adopt the mechanism that inflates it. `REJECT_MULT=0.10` was **not** separately validated — treat it as the first thing to sweep if the confirmatory run shows residual attacker mass.

**(b) The rank cap is load-bearing and assumption-bearing.** At `TAU=1.85` with **no** cap there are 36 benign false-flags; with `K_CAP=2`, zero; with `K_CAP=3`, the 36 return and detection drops. So the zero-FP result is a property of the cap, not of tau. Add an explicit `assert num_byzantine < n/2` and a comment stating the rule is sound only when `#attackers ≤ num_byzantine < n/2`. Note this cuts against the usual Byzantine convention of setting `f` *above* the expected count. Consider a **soft rank decay** instead of a hard top-k so that a third attacker is attenuated rather than ignored, and a third benign outlier is attenuated rather than zeroed.

**(c) tau's headroom is thin and Qwen-only.** Max ratio ever observed in a no-attack federation is **1.7833** (Qwen AG non-IID) → headroom to 1.85 is only **3.7%**. All five clean baselines are Qwen; **there is no Llama clean baseline anywhere in the archive**. Also, three Qwen Yahoo IID runs have steady-state attacker ratios *below* tau (1.7068/1.7188/1.7509). Pre-register `TAU=1.85`; do not re-select it after seeing the confirmatory run.

**(d) ⚠️ The "zero extra compute" claim is FALSE — resolve this before implementing.** The statistic validated on 51 runs is `local_cse` from `Server.evaluate_local_metrics` (`server.py:483-497`), which iterates the **full `test_loader`** and is called at `server.py:897`, i.e. **after** `aggregate_updates` at `server.py:872`. The genuinely free, pre-aggregation quantity is `evaluate_local_probe_distribution` (`server.py:586-625`), which returns only a `(K, C)` softmax tensor with `K ≤ semantic_probe_size = 100` → **10 samples/class on Yahoo** — the same sampling noise that killed `sem_div` (§1.1), against a 3.7% headroom. **These are different statistics and nobody has shown they agree at tau=1.85.** Choose one:
- **(i)** Reorder `run_round` so `evaluate_local_metrics` runs before `aggregate_updates` (accept ~2× per-round eval cost), using exactly the validated statistic; or
- **(ii)** Use the probe tensor (zero extra cost, edits confined to `trust_scorer.py`) **and** raise `semantic_probe_size` to `25 * num_labels` (250 on Yahoo) — under option (ii) this is a *prerequisite*, not an optional improvement — **and** log both quantities for one run to show the flags agree.

**(e) Implementation anchors** (`/Users/lancecai/Documents/GitHub/HMP-GNN`):
- `hmp_gae/trust_scorer.py` — `compute_trust_weights` (197–341): add the new signal after the Signal-3 block (274–284) and **route it around `_zscore`** (every existing channel is pool-relative; passing the new one through `_zscore` reproduces the exact failure being fixed). Extend the `TrustResult` dataclass (42–68).
- **⚠️ `active_sq` / `weight_norm` (317–322) must be updated for any new weighted term.** With `gate_rezscore=False` the gate is `sus = -s/‖w‖₂`; adding a fifth weight to `s` without adding `w₅²` to `active_sq` silently *lowers* the effective threshold and starts gating benign clients. This is the easiest way to break the run.
- `hmp_gae/runtime.py` — parse new `defense_config` keys in `__init__` (51–162, the only place `self.cfg.get(...)` is read); pass them through in `aggregate` (368–383); emit new per-client vectors in the `stats` dict (466–532) or they never reach `result.md`/CSVs; add any cross-round state to `state_dict`/`load_state_dict` (543–563) alongside `sus_ema` or resumed runs silently restart it.
- `server.py` — `aggregate_updates` (436–446): the whitelist `for k in ('residual','recon_residual','sem_div',...)` gatekeeps which `defense_stats` keys reach the JSON log. **A new key not added to this tuple is dropped silently.** Console printers at 393–404 are what `nb_to_resultmd.py` → `extract_csvs.py` turn into archive CSVs.
- `_needs_probe` (`server.py:92-98`) is `defense_method in ('hmp_gae',...) and semantic_weight > 0`. If a new signal needs the probe but someone sets `semantic_weight=0` for an ablation, the probe path switches off and the signal silently becomes `None`. Guard this.
- **Keep the new signal default-off** behind a config flag so `test_trust_weights_default_path_unchanged` stays green.
- `HMPGAEDefense.aggregate` wraps the runtime in a bare `except` that silently degrades to FedAvg. **Grep the log for `[HMP-GAE] runtime error` before trusting any result.**

### C3 (instrumentation, small, high value) — make every defense write per-round weights

`trust_weights.csv` exists only for `hmp_gae` runs, so FoolsGold's and Multi-Krum's effective attacker shares are *inferred from algorithm definitions*, and that inferred value (0.05) is the low-`w` anchor of the entire CSE-vs-attacker-weight regression. Have every defense emit a per-round per-client weight CSV. This converts the key regression from assumed to measured.

---

## 3. Explicitly rejected — do NOT include

| Proposal | Why rejected |
|---|---|
| Local CSE through the **unchanged** MAD-z pipeline | Drives a benign gate to 0.000 in a no-attacker federation (31/50 and 34/50 clean rounds over threshold); fails `test_no_attack_no_scapegoat`. |
| `gate_rezscore = true` | On Yahoo the top-1 suspect is **benign** in 45/50 (Llama) and 34/50 (Qwen) rounds, so re-z-scoring would systematically eject the most heterogeneous benign client. |
| Decoupling / capping the `n_k` prior | The diagnosis is right, but with a rejected client's multiplier near 0 the prior cannot rescue it — attacker mass reaches ~0 without touching `n_k`. Changing `n_k` alters the rule for *every* run incl. AG News and breaks FedAvg comparability, for no measured benefit. Treat the attacker data-share asymmetry (0.3077 Yahoo vs 0.1913 Llama AG) as a **disclosed confound** instead. |
| Adaptive per-round channel weighting | Motivating evidence is real, but there is no label-free selection rule, and the one tested variant (recon→1.0, sem→0.3) improves Llama Yahoo (−4%→−65% attacker mass) while **degrading Llama AG (−79%→−45%)** — the paper's strongest block. Defer. |
| `server_lr < 1` / FLTrust-style norm clipping as a PPL lever | Shrinkage/under-training. FLTrust has the worst or 2nd-worst final CSE in **all six** groups (3.7–4.9× the group best) and its PPL benefit fails outright on Qwen Yahoo IID (rank 9/10). Do **not** vary `server_lr` in the confirmatory run — it would confound the one measurement that matters. |
| Reducing LoRA `r`/`alpha` for PPL | Zero archive evidence (constant across all 55 runs). Targets the same knob that controls task learning → most likely another under-training artifact. Would destroy the measurement if bundled. |
| Early stopping / fewer rounds for PPL | 10-round run: PPL −22% but CSE +44% (0.8539→1.2306). Worthless. |
| `hist_weight_beta` tuning, `semantic_weight=2.0` | Banned by `docs/DECISION.md` (proven null/worse). Note `hist` is already identically zero in all 350 client-rounds of all runs — the "4-channel" stack is really 3 channels. |
| `corrupt_i` as the de-circularisation answer | Untested and it is an **accuracy-family** statistic; `DECISION.md` already records "Local Accuracy Is Not A Trust Signal Under Non-IID". Its only recoverable proxy scores 2/50 on both Yahoo runs. Test **update-direction self-angle** instead (the other `DECISION.md`-endorsed absolute signal) — it is geometric, shares no data with the test set, and is the only genuinely non-circular candidate. |

---

## 4. Known limitations to state, not hide

1. **Circularity is deeper than "same formula".** The probe set is a **snapshot of `self.test_loader`** (`server.py:508-531`), so the trust signal and the reported global CSE are computed on the **same data**, not merely the same functional. Every non-entropy substitute available in the archive fails on Yahoo (probe accuracy raw / normalised / delta / vs pool max: all 2/50, vs 37/50 and 30/50 on AG). Mitigations: draw the probe from a server pool **disjoint** from the reported test set; report accuracy and PPL alongside; and test update-direction self-angle as the genuinely independent layer. If the probe is resampled per round for adaptive-attacker hardening, **re-run the EMA sweep** — the β=0 recommendation was validated on a *fixed* probe, and per-round resampling injects fresh noise into a 100-sample estimate against 3.7% headroom.
2. **Adaptive-attacker margin is thin on Yahoo.** Evasion budget (CSE reduction needed to fall under tau): AG 39–78%, **Yahoo 14–18% at p5**. Hallu v2 is non-adaptive, so this is speculative hardening, not a measured weakness.
3. **The CSE predictions are an observational regression**, not a simulation — 3–8 points per group, replayed against artifacts produced under a *different* aggregator, with the low-`w` anchor assumed rather than measured. Predicted: Qwen Yahoo final CSE 0.6915 → ~0.29, Llama Yahoo 0.4042 → ~0.19. The independent anchor (FoolsGold Qwen Yahoo at 0.3040/0.6978) agrees to within 4%/2% — but **that same run is the archive's worst PPL (1549.30)**, so the anchor is also the cautionary tale.
4. **Majority-poisoned federations invert the rule** (the pool median is the reference). Safe only for `num_byzantine < n/2`.
5. **Yahoo class-index → label mapping is not recorded anywhere** (`per_class_ppl.csv`'s label column is empty on Yahoo). Verify in `data_loader.py` before naming any class in the paper.

---

## 5. Validation plan

**Run 0 (do first, cheapest, highest value): Llama no-attacker baseline.** It is the only thing that validates `TAU` on the model family carrying the paper's strongest block, and it simultaneously closes the PPL attack-sensitivity gap (currently measurable on Qwen only). Success check: max ratio-to-pool-median stays under ~1.7. Add this check to the ingestion workflow.

**Run 1 (confirmatory): Qwen Yahoo non-IID, V3 config, `trust_mode` → V4 only.** Reproduce `.../2-Yahoo-Answer-datasets/non-iid实验/2-non-iid-20260706-汉霖v3` with one knob changed.

Success criteria — **all four**, not CSE alone (`DECISION.md`: never rank by a single scalar):
- final CSE ≈ 0.29–0.35 (now 0.6915); mean CSE ≈ 0.68–0.75 (now 1.1034)
- **PPL does not drift toward FoolsGold's 1549** — target ≤ ~1060 (now 1061.74; remember <7% is noise)
- **`ppl_class_std` does not drift toward FoolsGold's 756** — target ≤ ~350 (now 344.41)
- accuracy holds near 0.6533

**Run 2 (regression guard): Llama AG News.** This is the block with all four columns currently best and the one the change-set most endangers — V4 buys it ~nothing (predicted CSE gain ≈ 0) while changing its aggregation semantics. Must not regress.

**Not covered by any archived run** (flag as future work): ≥3 attackers, colluding attackers parked just under tau, and federations with three heterogeneous benign outliers — all three break the rank cap.
