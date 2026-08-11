# HMP-GAE

Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing.

[Hanlin Cai](https://caihanlin.com/) · Research prototype, work in progress, not submitted.

HMP-GAE studies label-flip hallucination attacks in federated language-model
classification. The server combines conservative CSE decisions with
hypergraph message passing to assign client aggregation weights. The active
experiment is always defined by the `config` dictionary inside `main()` in
[`main.py`](main.py); values written in old logs or commits are not defaults.

## Documentation

| Document | Purpose |
|---|---|
| This README | Installation, execution, outputs, and a short method overview |
| [Documentation map](docs/README.md) | Source-of-truth rules and maintenance checklist |
| [MATH_LOGIC.md](MATH_LOGIC.md) | Current equations, symbols, and code mapping |
| [docs/DECISION.md](docs/DECISION.md) | Design history, rejected alternatives, and experiment contracts |
| [AGENTS.md](AGENTS.md) | Repository-specific working and verification rules for coding agents |

## Method at a glance

Each client trains locally with FedProx. Hallucination attackers use their own
data but randomly flip a configured fraction of labels. The current V8 defense
then performs:

```text
client updates + probe behavior
            ↓
 update-view and behavior-view hypergraphs
            ↓
 mutual cross-view relations + learned HMP affinity
            ↓
 CSE seeds → bounded risk propagation → data-size weighted aggregation
```

CSE supplies the high-confidence seeds; the hypergraph may only propagate risk
through relations supported by both views and within the remaining Byzantine
rank budget. If no valid propagation exists, V8 reduces exactly to its V5 CSE
baseline. This is a safety property, not evidence of improved accuracy. A
hypergraph-attributable claim requires matched V5/V8 runs and nonzero,
correctly targeted propagation; see the
[experiment contract](docs/DECISION.md#falsification-and-experiment-contract).

## Repository map

| Path | Responsibility |
|---|---|
| [`main.py`](main.py) | Sole experiment configuration and entry point |
| [`server.py`](server.py) / [`client.py`](client.py) | FL orchestration, evaluation, and local training |
| [`attack/`](attack/) | Hallucination and classical Byzantine attacks |
| [`defense/`](defense/) | Defense selection and aggregation facade |
| [`hmp_gae/`](hmp_gae/) | Node features, hypergraphs, HMP encoder/decoder, losses, and trust scoring |
| [`data_loader.py`](data_loader.py) | Dataset loading, tokenization, and local caches |
| [`fed_resume.py`](fed_resume.py) / [`fed_checkpoint.py`](fed_checkpoint.py) | Round resume and final checkpoint persistence |
| [`evaluation_hallucination.py`](evaluation_hallucination.py) | End-of-FL perplexity evaluation |
| [`run_downstream_generation.py`](run_downstream_generation.py) | Optional checkpoint-to-generation analysis |
| [`tests/`](tests/) | CPU regression and invariant tests |
| [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb) | The only maintained Colab notebook |

AG News and Yahoo Answers use CSV caches under `data/ag_news/` and
`data/yahoo_answers/`. Missing splits are downloaded automatically. IMDB and
DBpedia load through Hugging Face `datasets`.

## Supported backbones and datasets

- Encoder-only: DistilBERT, BERT, RoBERTa, and DeBERTa-v3.
- Decoder-only: GPT-2, Pythia, OPT, Qwen2.5, and Llama 3.2.
- Datasets: AG News, Yahoo Answers, IMDB, and DBpedia 14.

Use `model_name`, `dataset`, `num_labels`, and `max_length` in `main()` to
select a compatible arm. Llama is gated: accept its Hugging Face license and
provide `HF_TOKEN`. GPU memory requirements depend on the selected model,
precision, sequence length, and batch size; larger fp32 decoder arms may require
an A100-class GPU.

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

For full experiments, use [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb), choose
a suitable GPU, and run all cells. The notebook calls `main()` without config
overrides, prints the generated results, and releases the runtime at the end.
All experiment changes belong in `main.py`; do not create notebook-specific
configuration paths.

The repository can be cloned directly in a cloud runtime:

```bash
git clone https://github.com/GuangLun2000/HMP-GAE.git
cd HMP-GAE
pip install -r requirements.txt
python main.py
```

Local macOS development is intended for editing and static checks, not full FL
training.

## Configuration and controlled experiments

`main()` in [`main.py`](main.py) is the single source of truth. Important key
groups are:

- Experiment: `experiment_name`, model, dataset, partition, seeds, clients,
  attackers, and rounds.
- Attack: `attack_method` and the `hallu_*` label-flip settings.
- Defense: `defense_method`, `trust_mode`, and `defense_config`.
- Evaluation: classification entropy, perplexity, checkpoints, and downstream
  generation.

For every experimental arm:

1. Change only the intended variables.
2. Assign a unique `experiment_name` and checkpoint subdirectory.
3. Archive the generated config with the result.
4. Compare V5 and V8 only when model, data, attack, seed, and training settings
   are otherwise identical.

The resume fingerprint rejects incompatible checkpoints, but distinct names
remain required to prevent accidental continuation across arms.

## Outputs and evaluation

`results/` is gitignored. A run may produce:

- `results/<experiment>_results.json`: config, round metrics, trust signals,
  and detection diagnostics.
- `results/<experiment>_eval_ppl.json`: end-of-FL perplexity summary when the
  active backbone supports causal language modeling.
- `results/<experiment>_figure1.png` through `_figure5.png`: generated plots.
- A final global checkpoint and optional PEFT adapter when checkpoint saving is
  enabled.

Reported metrics include clean classification accuracy, loss, Classification
Semantic Entropy (CSE), and end-of-FL perplexity (PPL). CSE is a label-cluster
entropy proxy, not free-form semantic entropy. PPL is evaluated once after FL
and is skipped with an explicit reason for encoder-only backbones.

For V8, inspect `v8_propagated_flagged`, `v8_joint_evidence`, and
`v8_consensus_edge_count`. All-zero propagation means the hypergraph did not
change V5's decision in that run; any performance difference must not be
attributed to hypergraph propagation.

## Checkpoints and downstream generation

Enable `save_global_checkpoint` and set `global_checkpoint_subdir` in
`main()`. A checkpoint contains `global_model.pt`, metadata, and—when LoRA is
active—a `peft_adapter/` directory.

Provide a probe JSON list containing at least `news_text`, then run:

```bash
python run_downstream_generation.py \
  --checkpoint results/YOUR_GLOBAL_CHECKPOINT_SUBDIR \
  --probes /path/to/your_probes.json \
  --output results/downstream_gen.jsonl \
  --stable
```

`--stable` uses conservative greedy decoding. This analysis transfers the
classification backbone into a causal-LM wrapper; it does not perform language
model fine-tuning.

## Current limitations

- The implementation is text-only; LoRA updates stand in for a future true
  multimodal encoder.
- V8 requires server-side probe distributions and per-client local CSE. The
  current local-CSE rule evaluates on the server test loader, so claims must
  disclose this assumption and should later be validated with a disjoint
  server-held set.
- HMP-GAE falls back to FedAvg when `num_clients <= 2`; small federations can
  still have weak relational evidence above that hard threshold.
- Update-forging attackers whose local model remains benign are incompatible
  with CSE-reject modes; the facade raises instead of silently producing an
  invalid result.
- Convergence and performance require Colab experiments. Static tests validate
  wiring and invariants only.
- Paper-ready evidence still needs matched baselines, multiple seeds, and
  component ablations.

## Lightweight verification

After documentation edits:

```bash
python check_docs.py
```

After Python edits, also run `python -m compileall -q .`. Changes under
`hmp_gae/` additionally require `python tests/test_trust_robustness.py` in an
environment with PyTorch.
