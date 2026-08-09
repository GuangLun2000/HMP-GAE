# HMP-GNN

- Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing.
- [Hanlin Cai](https://caihanlin.com/)

## File Structure

```
.
├── .gitignore
├── LICENSE
├── README.md                          # This documentation
├── AGENTS.md / CLAUDE.md              # Agent working conventions (CLAUDE.md imports AGENTS.md)
├── MATH_LOGIC.md                      # Algorithm symbols and derivations
├── docs/DECISION.md                   # Settled design decisions + pre-registered constants
├── check_docs.py                      # Doc↔code consistency guard (run after editing any .md)
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry: configure and run federated learning
├── client.py                          # Client base + BenignClient (FedProx local training)
├── server.py                          # Aggregation, evaluation, round orchestration
├── models.py                          # NewsClassifierModel (SeqCLS + optional LoRA)
├── data_loader.py                     # DataManager / datasets (AG News, Yahoo Answers, IMDB, DBpedia)
├── fed_checkpoint.py                  # Save global model + metadata after FL
├── fed_resume.py                      # Per-round checkpoint + fingerprint resume (Colab resilience)
├── replay_v7_calibration.py           # V7 pre-run replay calibration over archived runs (stdlib)
├── decoder_adapters.py                # SeqCLS backbone → CausalLM transfer adapters
├── run_downstream_generation.py       # CLI: checkpoint + probes → JSONL (Task 2)
├── visualization.py                   # Experiment figures / plots
├── attack/                            # Attack baselines (label-flip + classical model poisoning)
│   ├── __init__.py                    # Re-exports attacker client classes
│   ├── hallucination.py               # Hallucination attack (V1, main)
│   ├── sign_flipping.py               # Sign-flipping (ICML ’18)
│   ├── gaussian.py                    # Gaussian (USENIX Security ’20)
│   └── alie.py                        # ALIE (NeurIPS ’19)
├── defense/                           # Server-side defense wiring
│   ├── __init__.py                    # build_defense: fedavg / hmp_gae / krum / multi_krum
│   │                                  #   / coord_median / fltrust / foolsgold
│   └── baselines/                     # (reserved)
│       └── __init__.py
├── evaluation_hallucination.py        # V2 M7: end-of-FL PPL (backbone transfer to CausalLM)
├── hmp_gae/                           # HMP-GAE defense sub-package (this paper)
│   ├── node_features.py               #   eta_i = f_enc(Delta_i, stats, history)
│   ├── hypergraph.py                  #   k-NN + V8 update/behavior consensus hypergraphs
│   ├── encoder.py                     #   L-layer HMP encoder (V8 residual + signed output)
│   ├── decoder.py                     #   GAE decoder: A_hat, H_hat (V8 cosine logits)
│   ├── losses.py                      #   legacy smoothness + V8 fixed-topology reconstruction
│   ├── trust_scorer.py                #   closed-form trust -> alpha_i
│   └── runtime.py                     #   end-to-end HMPGAERuntime
├── data/                              # Local CSV caches (AG News + Yahoo Answers)
│   ├── ag_news/                       # train.csv, test.csv (label,title,text — no header)
│   └── yahoo_answers/                 # train.csv, test.csv (label,text — no header; 1-based labels)
├── tests/
│   └── test_trust_robustness.py       # Trust-scoring CPU regression (V3–V8, legacy bit-for-bit)
└── HMP_GAE_Colab.ipynb                # Colab: main experiment + full inline results; then disconnect GPU
```

**AG News** and **Yahoo Answers** read CSVs under **`data/ag_news/`** and **`data/yahoo_answers/`** respectively. If either split is missing, the loader downloads and caches it there (see [`data_loader.py`](data_loader.py)). **IMDB** and **DBpedia** still load directly from Hugging Face `datasets` and do not use those folders.

**Task 2** requires a probe list JSON path you provide (`--probes` / `downstream_probes`).

## Supported Models

- Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`
- Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B` (gated: accept the HF license and provide `HF_TOKEN`; fp32 needs an A100)
- Configure in `main.py` via `model_name`.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default). CSVs: `data/ag_news/train.csv`, `data/ag_news/test.csv`.
- **Yahoo Answers** (yassiracharki/Yahoo_Answers_10_categories_for_NLP): `dataset='yahoo_answers'`, `num_labels=10`, `max_length=256` standalone — the cross-dataset comparison arms keep 128 (see `main.py`). 10 topic classes, 1.4M train / 60K test. CSVs: `data/yahoo_answers/train.csv`, `data/yahoo_answers/test.csv`.
- **IMDB** (stanfordnlp/imdb): `dataset='imdb'`, `num_labels=2`, `max_length=512` (or 256 for lower memory)
- **DBpedia 14** (fancyzhx/dbpedia_14): `dataset='dbpedia'`, `num_labels=14`, `max_length=512` (14 topic classes, 560K train / 70K test)
- Configure in `main.py` via `dataset`, `num_labels`, and `max_length`.

<br>

## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

### Local Execution

```bash
python main.py
```

### Google Colab Execution (or other Cloud AI platforms)

**Recommended: run the notebook.** Open [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb), enable **T4 GPU**, then **Run all**. It runs **`main.main()`** only — with **no overrides of any kind**: [`main.py`](main.py)'s `config` dict is the single source of truth, so what the notebook runs is exactly what that file says. It then prints the full `*_results.json` / PPL / per-round tables inline. The last cell calls **`google.colab.runtime.unassign()`** to release the GPU. Wall-clock time follows `main.py` (the canonical 50-round Qwen2.5 arm is ~3–4 h on a T4).

**Alternative: pure shell (same entry as local).**

```bash
git clone https://github.com/GuangLun2000/HMP-GNN.git
cd HMP-GNN
pip install -r requirements.txt
python main.py
```

<br>

---

### Checkpoints and Task 2 (downstream generation)

In [`main.py`](main.py) → `config`, turn on **`save_global_checkpoint`** and optionally **`global_checkpoint_subdir`** (under `results/`). You get `global_model.pt`, `checkpoint_metadata.json`, and with LoRA a **`peft_adapter/`** folder. Train with a causal **`model_name`** that matches **`num_labels`** / **`dataset`** (e.g. AG News + Pythia or Qwen2.5 as in **Supported Models**).

**Task 2** classifies each probe with the saved SeqCLS head, copies the backbone into **`AutoModelForCausalLM`** (no LM fine-tuning), and decodes a short explanation. AG News labels: 0–3 → World, Sports, Business, Sci/Tech. Backbone wiring lives in [`decoder_adapters.py`](decoder_adapters.py). Prepare your own probe JSON (list of objects with at least `news_text`; optional `id`, `question`, label fields as in the script’s `load_probes`).

To chain after FL, set **`run_downstream_after_fl`**: `True` and a non-None **`downstream_probes`** path (plus `downstream_output`, `downstream_cli_args`, …). Or run the CLI:

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --probes /path/to/your_probes.json \
  --output results/downstream_gen.jsonl \
  --stable
```

`--stable` is a conservative greedy preset; use **`--help`** for decoding flags. Each output line is JSONL (labels + text); compare predictions to ground-truth categories and read the rationale fields to study poisoning.

**Other decoder families:** implement `DecoderAdapter` (`matches`, `transfer_backbone`), append to **`ADAPTER_REGISTRY`** in [`decoder_adapters.py`](decoder_adapters.py), then point Task 2 at checkpoints with the same **`model_name`**.

<br>

---

## HMP-GAE Immunization (V1)

V1 ships the paper's core immunization pipeline end-to-end:

- **Attack**: `HallucinationAttackerClient` — the client trains on (partially) label-flipped data. No nested optimization loop, same wall-clock as benign clients.
- **Defense**: `HMPGAEDefense` — server-side hypergraph message-passing graph autoencoder that self-supervises on each round's updates, outputs per-client trust weights, and aggregates accordingly.

### Configure via `main.py::main()`

All knobs live in the single authoritative `config` dict in `main.py::main()` — there is
no CLI / notebook override path. This README deliberately does **not** restate live
values (they change per experiment arm; read the dict). The knob groups:

- **Attack** — `attack_method`: `'NoAttack' | 'Hallucination' | 'SignFlipping' |
  'Gaussian' | 'ALIE'`, plus the `hallu_*` knobs (flip mode / ratio / per-round
  randomization).
- **Defense** — `defense_method`: `'fedavg' | 'hmp_gae' | 'krum' | 'multi_krum' |
  'coord_median' | 'fltrust' | 'foolsgold'`. The `defense_config` block holds the
  hypergraph geometry (`knn_k`, encoder dims, loss weights), trust-signal fusion
  (`graph_weight` / `residual_weight_alpha` / `semantic_weight` / `hist_weight_beta`),
  and the robust suspicion scale (`zscore_mode`, `sus_ema_beta`, `reject_z_threshold`,
  `graph_min_distinct`, …).
- **`trust_mode`** — the trust→weight mapping: V3 `'soft_reject_fedavg'` (sigmoid gate +
  FedAvg) and the CSE-reject family `'v4_cse_reject'` / `'v5_cse_reject'` /
  `'v6_cse_reject_geo'` / `'v7_cse_reject_corrob'` /
  `'v8_hmp_cse_propagation'`. V7 remains a frozen calibration-dependent arm;
  V8 combines V5 CSE seeds with a fixed update/probe consensus hypergraph and
  learned HMP propagation. Pre-registered constants and design rationale:
  [docs/DECISION.md](docs/DECISION.md).

### V8: CSE-seeded dual-view hypergraph propagation

V8 addresses the observed V7 degeneracy directly: a scalar isolation floor can
remain inactive even though the hypergraph has useful relational structure.
For each round V8 builds two independent centered k-NN hypergraphs: one from the
fixed JL projection of client updates, and one from label-free softmax behavior
on the shared probe set. A relation can carry risk only when it is mutual in
both views. The HMP-GAE encoder trains against this round-fixed update topology,
uses residual layers and a signed cosine decoder, and attenuates node→edge→node
risk with its reconstructed affinity.

The decision authority remains conservative. V5's full-test CSE flags are the
only seeds and always have priority. A non-seed can use only unused
`num_byzantine` rank-cap budget, and only when it receives propagated seed risk
and its own pool-median CSE ratio is above 1. Its multiplier varies continuously
with the product of those two signals. No seed, no cross-view path to a seed,
no directionally elevated peer, or no remaining budget returns V5's weights
exactly; weak learned affinity instead yields a proportionally mild penalty.
This is the mechanism to test; it is not yet a claimed accuracy improvement.
The per-round `v8_*` diagnostics make a null result identifiable rather than
silently calling an inactive graph successful.

**Robust trust scoring (2026-07).** Four config-gated fixes targeting the two
failure modes that cost clean accuracy: (1) `semantic_reference='median'`
compares each client's probe softmax to the per-sample **median** consensus
instead of pairwise peers, so non-IID benign heterogeneity is no longer
penalized and attackers (a minority) cannot pollute the reference;
(2) `zscore_mode='mad'` swaps mean/std for median/MAD z-scores, which stay
meaningful up to ~50% attackers; (3) `gate_rezscore=False` removes the double
z-score on the combined gate — the legacy path forced every round onto a ±σ
scale so an all-benign round always down-weighted its most extreme client
(the "scapegoat tax"); the suspicion score is instead `-s / ‖w‖₂`, putting
`reject_z_threshold` in per-signal robust-z units; (4) `sus_ema_beta=0.6`
smooths suspicion across rounds, so one-off benign extremes recover while
persistent attackers stay gated. Setting the seven legacy values listed in
`main.py`'s robust-suspicion comment reproduces pre-2026-07 runs bit-for-bit.
Detection quality (attacker/benign
gate means + suspicion AUROC) is written to `detection_summary` in the
results JSON. Sanity tests: `python tests/test_trust_robustness.py` (CPU,
~1s, no dataset).

### Representative results (example regime)

Historical V1 snapshot (2026-06: N=10 clients, 2 attackers, short rounds, AG News subset, DistilBERT + LoRA, V3-era trust scoring). The current canonical arm is N=7 / 50 rounds / Qwen2.5-0.5B — see `config` in [`main.py`](main.py); archived results live outside this repo.

| Setting | Final Clean Acc (3-seed mean ± std) |
|---|---|
| Hallu + FedAvg   | 0.5667 ± 0.0661 |
| Hallu + HMP-GAE  | 0.6361 ± 0.0474 |
| **Delta (HMP-GAE improvement)** | **+0.0694** |

The trust-weight evolution in logged metrics / custom plots shows the two attackers (when configured) driven toward low aggregation mass while benign clients retain most of the weight.

### V2 M7: Hallucination Evaluation Metrics (no text generation)

Two additional metrics are computed without generating any text -- consistent with the paper's promise of reporting **task accuracy, semantic entropy, and perplexity** on the same benchmark.

- **Classification Semantic Entropy (CSE)** -- the mean Shannon entropy `H(p(y|x))` of the SeqCLS softmax distribution over the test set. Under a hallucination-inducing attack the classifier becomes less confident, driving `H` up; HMP-GAE filtering should bring `H` back down. **Every round**, essentially free (shares the test-set forward pass with accuracy/loss). Implemented in [server.py::evaluate_with_loss](server.py); also see the Farquhar-style cluster interpretation in [evaluation_hallucination.py](evaluation_hallucination.py).
- **Perplexity (PPL)** -- after FL finishes, the LoRA-fine-tuned backbone is transferred to an `AutoModelForCausalLM` via [decoder_adapters.py::resolve_adapter](decoder_adapters.py) and per-token negative log-likelihood is measured on a **stratified test subset** (default 200 samples, balanced across classes). No generation required. Available only for decoder-style backbones (Qwen, Pythia, OPT, GPT-2, LLaMA-family); encoder-only backbones such as DistilBERT/BERT report `skipped: true` cleanly.

Config knobs (already in [main.py](main.py)):

```python
'eval_classification_semantic_entropy': True,   # per-round, always on
'eval_perplexity': True,                         # end-of-FL, needs checkpoint
'ppl_num_samples': 200,                          # balanced across classes
'ppl_seed': 42,
'ppl_max_length': None,                          # None -> reuse config['max_length']
```

Output files per run (the `results/` folder is gitignored; paths below are produced by **`python main.py`** or the Colab notebook calling `main.main`):

- `results/<exp>_results.json` — config, round logs, `progressive_metrics` (including per-round CSE when enabled).
- `results/<exp>_eval_ppl.json` — end-of-FL PPL summary when `eval_perplexity` applies.
- `results/<exp>_figure1.png` … **`_figure5.png`** — publication-style plots from [`visualization.py`](visualization.py) (`ExperimentVisualizer.generate_all_figures`).

### V1 / V2 limitations and roadmap

- Baseline defenses (Krum / Multi-Krum / Coord-Median / FLTrust / FoolsGold) are implemented in `defense/` and selectable via `defense_method`; FLDetector / Safe-FedLLM remain future work.
- PPL currently evaluates a decoder-only backbone; when `model_name` is encoder-only, PPL is skipped with a reason string in the JSON.
- Single modality (text) -- the paper's multimodal formulation is simulated via LoRA-only updates; true multimodal encoders are later work.
- The canonical regime is N=7 (5 benign + 2 attackers). For `num_clients <= 2` the defense auto-falls back to FedAvg (the hard threshold in `defense/__init__.py::HMPGAEDefense.aggregate`; hypergraph signals are simply weak at small N); for very heterogeneous (`dirichlet_alpha << 0.3`) data, `reject_z_threshold` may need to be raised.

<br>
