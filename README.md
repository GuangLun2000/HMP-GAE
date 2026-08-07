# HMP-GNN

- Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing.
- [Hanlin Cai](https://caihanlin.com/)

## File Structure

```
.
├── .gitignore
├── LICENSE
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry: configure and run federated learning
├── client.py                          # Client base + BenignClient (FedProx local training)
├── server.py                          # Aggregation, evaluation, round orchestration
├── models.py                          # NewsClassifierModel (SeqCLS + optional LoRA)
├── data_loader.py                     # DataManager / datasets (AG News, Yahoo Answers, IMDB, DBpedia)
├── fed_checkpoint.py                  # Save global model + metadata after FL
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
│   ├── __init__.py                    # FedAvg / HMP-GAE + build_defense (was root defense.py)
│   └── baselines/                     # Placeholder for future defense baselines
│       └── __init__.py
├── evaluation_hallucination.py        # V2 M7: end-of-FL PPL (backbone transfer to CausalLM)
├── hmp_gae/                           # HMP-GAE defense sub-package (this paper)
│   ├── node_features.py               #   eta_i = f_enc(Delta_i, stats, history)
│   ├── hypergraph.py                  #   k-NN hypergraph H, D_V, D_E
│   ├── encoder.py                     #   L-layer HMP encoder (node↔hyperedge)
│   ├── decoder.py                     #   GAE decoder: A_hat, H_hat
│   ├── losses.py                      #   BCE(H,H_hat) + smoothness + hist
│   ├── trust_scorer.py                #   closed-form trust -> alpha_i
│   └── runtime.py                     #   end-to-end HMPGAERuntime
├── data/                              # Local CSV caches (AG News + Yahoo Answers)
│   ├── ag_news/                       # train.csv, test.csv (label,title,text — no header)
│   └── yahoo_answers/                 # train.csv, test.csv (label,text — no header; 1-based labels)
└── HMP_GAE_Colab.ipynb                # Colab: main experiment + full inline results; then disconnect GPU
```

**AG News** and **Yahoo Answers** read CSVs under **`data/ag_news/`** and **`data/yahoo_answers/`** respectively. If either split is missing, the loader downloads and caches it there (see [`data_loader.py`](data_loader.py)). **IMDB** and **DBpedia** still load directly from Hugging Face `datasets` and do not use those folders.

**Task 2** requires a probe list JSON path you provide (`--probes` / `downstream_probes`).

## Supported Models

- Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`
- Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`
- Configure in `main.py` via `model_name`.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default). CSVs: `data/ag_news/train.csv`, `data/ag_news/test.csv`.
- **Yahoo Answers** (yassiracharki/Yahoo_Answers_10_categories_for_NLP): `dataset='yahoo_answers'`, `num_labels=10`, `max_length=256` (10 topic classes, 1.4M train / 60K test). CSVs: `data/yahoo_answers/train.csv`, `data/yahoo_answers/test.csv`.
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

**Recommended: run the notebook.** Open [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb), enable **T4 GPU**, then **Run all**. It runs **`main.main()`** only — with **no overrides of any kind**: [`main.py`](main.py)'s `config` dict is the single source of truth, so what the notebook runs is exactly what that file says. It then prints the full `*_results.json` / PPL / per-round tables inline. The last cell calls **`google.colab.runtime.unassign()`** to release the GPU. Wall-clock time follows `main.py` (e.g. Qwen2.5 + 10 rounds is long).

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

```python
# Attack
'attack_method': 'Hallucination',
'hallu_flip_ratio': 1.0,               # 0..1, fraction of samples flipped
'hallu_flip_mode': 'pairwise',         # 'pairwise' | 'targeted' | 'random'
'hallu_flip_map': {0: 1, 1: 0, 2: 3, 3: 2},   # AG News: World<->Sports, Business<->Sci/Tech

# Defense
'defense_method': 'hmp_gae',           # or 'fedavg' for the baseline
'defense_config': {
    'knn_k': 3, 'hidden_dim': 64, 'latent_dim': 32, 'num_hmp_layers': 2,
    'train_steps_per_round': 5, 'train_lr': 1e-3,
    'lambda_H': 1.0, 'lambda_A': 1.0, 'lambda_hist': 0.5,
    'graph_weight': 1.0, 'residual_weight_alpha': 0.3, 'hist_weight_beta': 0.0,
    'semantic_weight': 1.0,
    # --- Robust trust scoring (2026-07; legacy values in comments) ---
    'zscore_mode': 'mad',                # 'std' = legacy mean/std z-scores
    'zscore_clip': 10.0,                 # cap |z| per signal
    'gate_rezscore': False,              # True = legacy double z-score gate
    'sus_ema_beta': 0.6,                 # 0.0 = no cross-round suspicion EMA
    'semantic_reference': 'median',      # 'pairwise' = legacy peer-consensus KL
    'semantic_confidence_weight': False, # ablation knob
    'semantic_probe_stratified': True,   # False = head-of-test_loader probes
    'trust_mode': 'soft_reject_fedavg',
    'reject_z_threshold': 2.5,           # per-signal robust-z units; use 0.75
                                         # with gate_rezscore=True (legacy)
    'soft_reject_k': 2.0,
    'softmax_tau': 0.1, 'hist_ema_beta': 0.9,
    'cold_start_fallback': False,
    'device': 'cpu', 'random_proj_seed': 42,
},
```

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
persistent attackers stay gated. Setting the six legacy values noted above
reproduces pre-2026-07 runs bit-for-bit. Detection quality (attacker/benign
gate means + suspicion AUROC) is written to `detection_summary` in the
results JSON. Sanity tests: `python tests/test_trust_robustness.py` (CPU,
~1s, no dataset).

### Representative results (example regime)

Runs below use **`python main.py`** with comparable settings (e.g. N=10 clients, 2 attackers, short rounds, AG News subset, DistilBERT + LoRA); tune `config` in [`main.py`](main.py) to reproduce.

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

- V1 still omits comparison baselines (Krum / Median / FLTrust / FLDetector / Safe-FedLLM). Planned for the next V2 milestone.
- PPL currently evaluates a decoder-only backbone; when `model_name` is encoder-only, PPL is skipped with a reason string in the JSON.
- Single modality (text) -- the paper's multimodal formulation is simulated via LoRA-only updates; true multimodal encoders are later work.
- Tuning presets above are calibrated for the N=10 / 2-attackers / AG News regime. For `num_clients <= 2` the defense auto-falls back to FedAvg (the hard threshold in `defense/__init__.py::HMPGAEDefense.aggregate`; hypergraph signals are simply weak at small N); for very heterogeneous (`dirichlet_alpha << 0.3`) data, `reject_z_threshold` may need to be raised.

<br>
