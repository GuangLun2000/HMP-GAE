# HMP-GAE

## Repository map

| Path | Contents |
|---|---|
| [`main.py`](main.py) | The one authoritative `config` dictionary and the run entry point |
| [`server.py`](server.py) | Round orchestration, evaluation, probe forward, aggregation calls |
| [`client.py`](client.py) | Benign client and FedProx local training |
| [`models.py`](models.py) | Backbone loading and LoRA wiring |
| [`data_loader.py`](data_loader.py) | Dataset download, tokenization, IID/Dirichlet partitioning, local caches |
| [`attack/`](attack/) | The hallucination attack, plus the SignFlipping, Gaussian, and ALIE baselines |
| [`defense/`](defense/) | Defense facade and the FedAvg, Krum, median, FLTrust, FoolsGold baselines |
| [`hmp_gae/`](hmp_gae/) | Node features, hypergraph construction, HMP encoder/decoder, losses, trust scoring |
| [`fed_resume.py`](fed_resume.py) | Per-round resume snapshots and trajectory fingerprints |
| [`fed_checkpoint.py`](fed_checkpoint.py) | Final global-model checkpoint saving |
| [`decoder_adapters.py`](decoder_adapters.py) | Backbone transfer from the classifier into a causal-LM wrapper |
| [`evaluation_hallucination.py`](evaluation_hallucination.py) | End-of-FL perplexity evaluation |
| [`run_downstream_generation.py`](run_downstream_generation.py) | Optional checkpoint-to-generation analysis |
| [`visualization.py`](visualization.py) | Result figures |
| [`HMP_GAE_Colab.ipynb`](HMP_GAE_Colab.ipynb) | The only maintained Colab notebook |
| [`data/`](data/) | CSV caches for AG News and Yahoo Answers (downloaded on demand) |
