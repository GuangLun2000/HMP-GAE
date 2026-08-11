# HMP-GAE

Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing.

[Hanlin Cai](https://caihanlin.com/) · Research prototype, work in progress, not submitted.

A federated-learning testbed for label-flip hallucination attacks on LLM text
classification. Clients fine-tune a shared backbone locally; the server scores
each client before aggregation and reports clean accuracy, Classification
Semantic Entropy (CSE), and end-of-FL perplexity. A single run is fully
described by the `config` dictionary inside `main()` in [`main.py`](main.py).

## Repository map

| Path | Contents |
|---|---|
| [`main.py`](main.py) | The one authoritative `config` dictionary and the run entry point |
| [`server.py`](server.py) | Round orchestration, evaluation, probe forward, aggregation calls |
| [`client.py`](client.py) | Benign client and FedProx local training |
| [`models.py`](models.py) | Backbone loading and LoRA wiring |
| [`data_loader.py`](data_loader.py) | Dataset download, tokenization, IID/Dirichlet partitioning, local caches |
| [`attack/`](attack/) | Hallucination label-flip attack plus SignFlipping, Gaussian, and ALIE baselines |
| [`defense/`](defense/) | Defense facade and the FedAvg, Krum, median, FLTrust, FoolsGold baselines |
| [`hmp_gae/`](hmp_gae/) | Node features, hypergraph construction, HMP encoder/decoder, losses, trust scoring |
| [`fed_resume.py`](fed_resume.py) | Per-round resume snapshots and trajectory fingerprints |
| [`fed_checkpoint.py`](fed_checkpoint.py) | Final global-model checkpoint saving |
| [`decoder_adapters.py`](decoder_adapters.py) | Backbone transfer from the classifier into a causal-LM wrapper |
| [`evaluation_hallucination.py`](evaluation_hallucination.py) | End-of-FL perplexity evaluation |
| [`run_downstream_generation.py`](run_downstream_generation.py) | Optional checkpoint-to-generation analysis |
| [`visualization.py`](visualization.py) | Result figures |
| [`tests/`](tests/) | CPU regression tests (no GPU or dataset needed) |
| [`check_docs.py`](check_docs.py) | Documentation/code consistency guard |
| [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb) | The only maintained Colab notebook |
| [`docs/`](docs/) | Method equations, design history, and the documentation map |
| [`data/`](data/) | CSV caches for AG News and Yahoo Answers (downloaded on demand) |

`results/` is created at runtime and is gitignored.

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

`python main.py` runs the experiment exactly as configured; there are no
command-line flags, environment overrides, or notebook override hooks. A local
macOS machine is fine for editing and for the checks in
[Verification](#verification), but full runs need a GPU.

To change the experiment, edit the `config` dictionary inside `main()` in
[`main.py`](main.py). Each key is commented in place with its accepted values.
Give every run its own `experiment_name` and checkpoint subdirectories so that
resume never picks up another run's state.

## Run on Colab

Open [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb), select a GPU runtime, and run
all cells. Its steps are: fetch the repository, install `requirements.txt`,
check the GPU and Hugging Face login, call `main()` without overriding anything,
render the metrics and per-client figures inline, print the numeric tables, zip
the `results/` artifacts for download, and release the runtime. Run the last
cell when you are done so the GPU is not held.

To use a plain Colab or any other cloud shell instead:

```bash
git clone https://github.com/GuangLun2000/HMP-GAE.git
cd HMP-GAE
pip install -r requirements.txt
python main.py
```

Gated backbones such as Llama 3.2 require accepting the model license on Hugging
Face and providing `HF_TOKEN` (in Colab, add it under the sidebar's Secrets tab;
the notebook logs in automatically). GPU memory depends on the selected backbone,
precision, sequence length, and batch size; larger fp32 decoder runs need an
A100-class GPU.

## Outputs

Written to `results/`, named after `experiment_name`:

- `<experiment>_results.json` — the full config, per-round metrics, and defense
  diagnostics.
- `<experiment>_eval_ppl.json` — perplexity, when the backbone supports causal
  language modeling.
- `<experiment>_figure1.png` … `_figure5.png` — generated plots.
- A global checkpoint directory, plus a `peft_adapter/` when LoRA is enabled, if
  checkpoint saving is on. This checkpoint is what perplexity and
  `run_downstream_generation.py` consume.

Per-round resume snapshots are stored separately so an interrupted Colab session
can continue from the last completed round.

## Verification

No training required; these are the checks CI also runs
([`.github/workflows/verify.yml`](.github/workflows/verify.yml)):

```bash
python check_docs.py            # after editing any Markdown
python -m compileall -q .       # after editing any Python
python tests/test_trust_robustness.py   # after editing hmp_gae/ (needs PyTorch)
```

## Documentation

| Document | Purpose |
|---|---|
| [docs/README.md](docs/README.md) | Documentation map and maintenance rules |
| [docs/MATH_LOGIC.md](docs/MATH_LOGIC.md) | Equations, symbols, and their code locations |
| [docs/DECISION.md](docs/DECISION.md) | Design history, rejected alternatives, experiment contracts |
| [AGENTS.md](AGENTS.md) | Working and verification rules for coding agents |
