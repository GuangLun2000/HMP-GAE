# main.py — FL experiment entry point: label-flip Hallucination attack vs HMP-GAE defense.
# The config dict in main() is the single source of truth (conventions: AGENTS.md).

import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Sequence

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient
from server import Server
from visualization import ExperimentVisualizer
from fed_checkpoint import save_global_model_checkpoint
from fed_resume import (
    apply_round_checkpoint,
    load_round_checkpoint,
    save_round_checkpoint,
)

warnings.filterwarnings('ignore')


def _preflight_hf_auth(model_name):
    """Fail fast when the configured backbone is a gated HF repo (e.g.
    meta-llama/*) and the current session cannot access it, instead of
    401-ing deep inside AutoTokenizer.from_pretrained after setup starts.
    Network/offline errors are ignored — the normal download path decides."""
    try:
        from huggingface_hub import auth_check, get_token, login
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        return

    # Colab fallback: pull HF_TOKEN from Colab Secrets if notebook Step 2 didn't run.
    colab_secret_err = None
    if get_token() is None:
        try:
            from google.colab import userdata
        except ImportError:
            pass
        else:
            try:
                login(token=userdata.get("HF_TOKEN"))
                print("HF login OK（自动从 Colab Secrets 读取 HF_TOKEN）")
            except Exception as err:
                colab_secret_err = f"{type(err).__name__}: {err}"

    try:
        auth_check(model_name)
    except GatedRepoError as e:
        if get_token() is None:
            hint = ("当前会话没有 HF token。\n"
                    "  1) 在 https://huggingface.co/{m} 接受许可\n"
                    "  2) 在 https://huggingface.co/settings/tokens 创建 Read token\n"
                    "  3) **Colab** 左侧边栏 🔑 Secrets（不是 GitHub 的 Secrets）添加名为\n"
                    "     HF_TOKEN 的 secret（全大写），并打开 'Notebook access' 开关\n"
                    "  4) 重新运行本 cell")
            if colab_secret_err:
                hint += f"\n  [从 Colab Secrets 读取失败，原因: {colab_secret_err}]"
        else:
            hint = ("已有 HF token 但无权访问该仓库：请用同一账号在\n"
                    "  https://huggingface.co/{m} 接受许可（或等待审核通过）后重试；\n"
                    "  若是 fine-grained token，确认已勾选 gated repo 读取权限")
        raise RuntimeError(
            f"'{model_name}' 是 gated 仓库，当前无法访问。\n" + hint.format(m=model_name)
        ) from e
    except Exception:
        return


def setup_experiment(config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Setting up Experiment: {config['experiment_name']}")
    print("=" * 50)

    _preflight_hf_auth(config.get('model_name', 'distilbert-base-uncased'))
    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'],
        test_seed=config['seed'],
        dataset_size_limit=config['dataset_size_limit'],
        batch_size=config['batch_size'],
        test_batch_size=config['test_batch_size'],
        model_name=config.get('model_name', 'distilbert-base-uncased'),
        max_length=config.get('max_length', 128),
        dataset=config.get('dataset', 'ag_news')
    )

    # Partition data among clients (IID or Dirichlet non-IID).
    data_distribution = config.get('data_distribution', 'non-iid').lower()
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    num_labels = config.get('num_labels', 4)
    num_clients = config['num_clients']
    num_attackers = config.get('num_attackers', 0)
    num_benign = num_clients - num_attackers
    
    rng = np.random.default_rng(config['seed'])
    
    client_indices = {i: [] for i in range(num_clients)}
    
    if data_distribution == 'iid':
        print("\nPartitioning data (IID distribution)...")
        
        all_indices = indices.copy()
        rng.shuffle(all_indices)
        
        total_samples = len(all_indices)
        base_samples = total_samples // num_clients
        remainder = total_samples % num_clients
        
        start_idx = 0
        for client_id in range(num_clients):
            extra = 1 if client_id < remainder else 0
            end_idx = start_idx + base_samples + extra
            client_indices[client_id] = all_indices[start_idx:end_idx].tolist()
            start_idx = end_idx
        
        print(f"  IID distribution (uniform random partition)")
        for client_id in range(num_clients):
            client_labels = [labels[idx] for idx in client_indices[client_id]]
            label_counts = {l: client_labels.count(l) for l in range(num_labels)}
            total = len(client_indices[client_id])
            if total > 0:
                dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): {total} samples ({dist_str})")
            else:
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): 0 samples WARNING: No data assigned!")

    else:
        print("\nPartitioning data (Non-IID distribution)...")
        
        dirichlet_alpha = config['dirichlet_alpha']
        
        label_indices = {label: [] for label in range(num_labels)}
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        for label in range(num_labels):
            label_list = np.array(label_indices[label])
            rng.shuffle(label_list)
            
            # Lower alpha = more heterogeneous.
            proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
            proportions = np.cumsum(proportions)
            proportions[-1] = 1.0  # Ensure last is exactly 1.0
            
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = int(len(label_list) * proportions[client_id])
                client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
                start_idx = end_idx
        
        for client_id in range(num_clients):
            client_list = np.array(client_indices[client_id])
            rng.shuffle(client_list)
            client_indices[client_id] = client_list.tolist()
        
        print(f"  Non-IID distribution (Dirichlet alpha={dirichlet_alpha})")
        for client_id in range(num_clients):
            client_labels = [labels[idx] for idx in client_indices[client_id]]
            label_counts = {l: client_labels.count(l) for l in range(num_labels)}
            total = len(client_indices[client_id])
            if total > 0:
                dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): {total} samples ({dist_str})")
            else:
                client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
                print(f"    Client {client_id} ({client_type}): 0 samples WARNING: No data assigned!")

    # Attacker data semantics depend on attack_method (see AGENTS.md): Hallucination
    # attackers train on their assigned local data with flipped labels; the classical
    # baselines forge updates and use assigned data mainly as claimed size.
    if num_benign < num_clients:
        _am = config.get('attack_method', 'Hallucination')
        if _am == 'Hallucination':
            print("\n  [Note] Hallucination attackers USE their assigned local data and flip labels during training.")
        elif _am != 'NoAttack':
            print("\n  [Note] Assigned attacker data mainly defines the claimed update weight; "
                  "actual usage depends on the attack implementation (see attack/).")

    test_loader = data_manager.get_test_loader()

    use_lora = config.get('use_lora', False)
    model_name = config.get('model_name', 'distilbert-base-uncased')
    if use_lora:
        print(f"Initializing global model ({model_name}) with LoRA...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=True,
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            lora_target_modules=config.get('lora_target_modules', None)
        )
    else:
        print(f"Initializing global model ({model_name}) [Full Fine-tuning]...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=False
        )

    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        total_rounds=config['num_rounds'],
        server_lr=config['server_lr'],
        similarity_mode=config.get('server_similarity_mode', 'pairwise'),
        defense_method=config.get('defense_method', 'fedavg'),
        defense_config=config.get('defense_config', None),
        num_clients=config['num_clients'],
        compute_classification_semantic_entropy=config.get(
            'eval_classification_semantic_entropy', True),
        semantic_probe_size=int(config.get('semantic_probe_size', 64)),
        semantic_probe_seed=int(config.get('seed', 42)),
        eval_local_every_n_rounds=int(config.get('eval_local_every_n_rounds', 1)),
    )

    print("\nCreating federated learning clients...")
    num_attackers = config.get('num_attackers', 0)
    attack_method = config.get('attack_method', 'Hallucination')

    # 'NoAttack' forces every client benign even when num_attackers>0.
    if attack_method == 'NoAttack' and num_attackers > 0:
        print(f"  [config] attack_method='NoAttack' overrides num_attackers={num_attackers}: "
              f"all {config['num_clients']} clients will be benign.")
        effective_num_attackers = 0
    else:
        effective_num_attackers = num_attackers

    # The last 'effective_num_attackers' client ids are the attackers.
    for client_id in range(config['num_clients']):
        if client_id < (config['num_clients'] - effective_num_attackers):
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]
            
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer, 
                                  max_length=config.get('max_length', 128))
            client_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            print(f"  Client {client_id}: BENIGN ({len(client_indices[client_id])} samples)")
            
            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config['alpha'],
                data_indices=client_indices[client_id],
                grad_clip_norm=config['grad_clip_norm']
            )
        else:
            # Claimed size = actual assigned size (attackers don't exaggerate weight).
            claimed_data_size = len(client_indices[client_id])

            if attack_method == 'ALIE':
                from attack.alie import ALIEAttackerClient
                print(f"  Client {client_id}: ATTACKER (ALIE Attack)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                
                alie_z_max = config.get('alie_z_max', None)
                alie_attack_start_round = config.get('alie_attack_start_round', None)
                
                client = ALIEAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    num_clients=config['num_clients'],
                    num_attackers=config['num_attackers'],
                    z_max=alie_z_max,
                    attack_start_round=alie_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0)
                )
            elif attack_method == 'SignFlipping':
                from attack.sign_flipping import SignFlippingAttackerClient
                print(f"  Client {client_id}: ATTACKER (Sign-Flipping Attack, ICML '18)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                # Build DataLoader for attacker so it can compute g_own (same as benign client)
                client_texts_sf = [data_manager.train_texts[i] for i in client_indices[client_id]]
                client_labels_sf = [data_manager.train_labels[i] for i in client_indices[client_id]]
                dataset_sf = NewsDataset(client_texts_sf, client_labels_sf, data_manager.tokenizer,
                                         max_length=config.get('max_length', 128))
                client_loader_sf = DataLoader(dataset_sf, batch_size=config['batch_size'], shuffle=True)
                sign_flip_scale = config.get('sign_flip_scale', 10.0)
                sign_flip_attack_start_round = config.get('sign_flip_attack_start_round', None)
                client = SignFlippingAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    data_loader=client_loader_sf,
                    sign_flip_scale=sign_flip_scale,
                    attack_start_round=sign_flip_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0)
                )
            elif attack_method == 'Hallucination':
                # Label-flipping (this paper); per-round randomization in attack/hallucination.py.
                from attack.hallucination import HallucinationAttackerClient
                print(f"  Client {client_id}: ATTACKER (Hallucination Attack - Label Flipping)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                client_texts_h = [data_manager.train_texts[i] for i in client_indices[client_id]]
                client_labels_h = [data_manager.train_labels[i] for i in client_indices[client_id]]
                dataset_h = NewsDataset(client_texts_h, client_labels_h, data_manager.tokenizer,
                                        max_length=config.get('max_length', 128))
                client_loader_h = DataLoader(dataset_h, batch_size=config['batch_size'], shuffle=True)
                hallu_flip_map = config.get('hallu_flip_map', {0: 1, 1: 0, 2: 3, 3: 2})
                # Keys may be strings if config is loaded from JSON; normalize to int.
                hallu_flip_map = {int(k): int(v) for k, v in hallu_flip_map.items()}
                hallu_flip_ratio_range = config.get('hallu_flip_ratio_range', None)
                if hallu_flip_ratio_range is not None:
                    hallu_flip_ratio_range = tuple(float(x) for x in hallu_flip_ratio_range)
                client = HallucinationAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_loader=client_loader_h,
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    data_indices=client_indices[client_id],
                    grad_clip_norm=config.get('grad_clip_norm', 1.0),
                    flip_ratio=float(config.get('hallu_flip_ratio', 1.0)),
                    flip_mode=str(config.get('hallu_flip_mode', 'pairwise')),
                    flip_map=hallu_flip_map,
                    num_labels=config.get('num_labels', 4),
                    target_class=config.get('hallu_target_class', None),
                    attack_start_round=int(config.get('hallu_attack_start_round', 0)),
                    claimed_data_size=claimed_data_size,
                    per_round_reseed=bool(config.get('hallu_per_round_reseed', False)),
                    flip_ratio_range=hallu_flip_ratio_range,
                )
            elif attack_method == 'Gaussian':
                from attack.gaussian import GaussianAttackerClient
                print(f"  Client {client_id}: ATTACKER (Gaussian Attack, USENIX Security '20)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                gaussian_attack_start_round = config.get('gaussian_attack_start_round', None)
                gaussian_std_scale = config.get('gaussian_std_scale', 1.0)
                if gaussian_std_scale != 1.0:
                    print(f"    Gaussian std_scale: {gaussian_std_scale} (noise range expanded for FedAvg)")
                client = GaussianAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    attack_start_round=gaussian_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0),
                    gaussian_std_scale=gaussian_std_scale
                )
            else:
                raise ValueError(
                    f"Unknown attack_method={attack_method!r}. Supported: "
                    "'NoAttack' | 'Hallucination' | 'SignFlipping' | 'Gaussian' | 'ALIE'."
                )

        server.register_client(client)
    
    return server, results_dir


def run_perplexity_eval_if_configured(config: Dict, results_dir: Path) -> None:
    """
    V2 M7: compute end-of-FL perplexity on a balanced test subset via backbone
    transfer into AutoModelForCausalLM. Requires save_global_checkpoint=True.
    Writes results/<experiment_name>_eval_ppl.json. Skips silently if disabled.
    """
    if not config.get("eval_perplexity", False):
        return
    if not config.get("save_global_checkpoint", False):
        print("\n[PPL] Skipped: eval_perplexity=True requires save_global_checkpoint=True.")
        return

    ckpt_dir = results_dir / config.get("global_checkpoint_subdir", "global_checkpoint")
    pt_file = ckpt_dir / "global_model.pt"
    if not pt_file.is_file():
        print(f"\n[PPL] Skipped: checkpoint not found at {pt_file}.")
        return

    try:
        from evaluation_hallucination import compute_test_ppl
    except ImportError as e:
        print(f"\n[PPL] Skipped: cannot import evaluation_hallucination: {e}")
        return

    print("\n" + "=" * 60)
    print("V2 M7: Perplexity evaluation (backbone transfer to CausalLM)")
    print("=" * 60)
    try:
        result = compute_test_ppl(
            checkpoint_dir=ckpt_dir,
            n_samples=int(config.get("ppl_num_samples", 200)),
            seed=int(config.get("ppl_seed", 42)),
            max_length=config.get("ppl_max_length") or config.get("max_length", 128),
            dataset_override=config.get("dataset"),
            num_labels_override=config.get("num_labels"),
            dataset_size_limit=config.get("dataset_size_limit"),
        )
    except Exception as e:
        print(f"[PPL] Evaluation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    out_path = results_dir / f"{config.get('experiment_name', 'experiment')}_eval_ppl.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if result.get("skipped"):
        print(f"[PPL] Skipped: {result.get('skip_reason')}")
    else:
        print(f"[PPL] PPL mean = {result['ppl_mean']:.4f} on {result['n_samples']} samples")
    print(f"[PPL] Wrote {out_path}")


def run_downstream_task2_if_configured(config: Dict, results_dir: Path) -> None:
    """
    Optionally run Task 2 (run_downstream_generation.py) after FL when checkpoint exists.
    Controlled by config['run_downstream_after_fl'].
    """
    if not config.get("run_downstream_after_fl", False):
        return

    ckpt_dir = results_dir / config.get("global_checkpoint_subdir", "global_checkpoint")
    pt_file = ckpt_dir / "global_model.pt"
    if not pt_file.is_file():
        print(
            f"\n⚠️  Task 2 skipped: no checkpoint at {pt_file}. "
            "Set save_global_checkpoint=True and complete training, or run run_downstream_generation.py manually."
        )
        return

    probes_cfg = config.get("downstream_probes")
    if not probes_cfg:
        print(
            "\n⚠️  Task 2 skipped: set config['downstream_probes'] to a probe JSON path "
            "(FL training uses ``data/ag_news/`` or ``data/yahoo_answers/`` for those datasets; see data_loader.py)."
        )
        return
    probes = Path(probes_cfg)
    if not probes.is_file():
        print(f"\n⚠️  Task 2 skipped: probes file not found: {probes}")
        return

    out_raw = config.get("downstream_output")
    if out_raw:
        out_path = Path(out_raw)
        if not out_path.is_absolute():
            out_path = results_dir / out_path
    else:
        out_path = results_dir / f"{config.get('experiment_name', 'experiment')}_downstream_gen.jsonl"

    device = config.get("downstream_device")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extra: Sequence[str] = config.get("downstream_cli_args") or []
    if isinstance(extra, str):
        extra = [extra]

    cmd: List[str] = [
        sys.executable,
        "run_downstream_generation.py",
        "--checkpoint",
        str(ckpt_dir),
        "--probes",
        str(probes),
        "--output",
        str(out_path),
        "--device",
        str(device),
    ]
    cmd.extend(str(x) for x in extra)

    print("\n" + "=" * 60)
    print("Task 2: downstream generation (run_downstream_generation.py)")
    print("=" * 60)
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if proc.returncode != 0:
        print(f"\n⚠️  Task 2 exited with code {proc.returncode}")
    else:
        print(f"\nTask 2 finished; JSONL: {out_path}")


def run_experiment(config):
    server, results_dir = setup_experiment(config)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'acc_diff': [],
        'agg_update_norm': [],
        'cse': [],
    }

    # Resume from a per-round checkpoint if one matches (Colab resilience; fed_resume.py).
    ckpt_subdir = config.get('round_checkpoint_subdir', 'round_checkpoint')
    payload, reason = load_round_checkpoint(config, results_dir, subdir=ckpt_subdir)
    start_round = 0
    if payload is not None:
        start_round = apply_round_checkpoint(server, progressive_metrics, payload)
        print(f"\n[resume] {reason}")
        if start_round >= config['num_rounds']:
            print(f"[resume] All {config['num_rounds']} rounds already completed; skipping FL loop.")
    elif reason:
        print(f"\n[resume] Starting fresh ({reason}).")

    # Initial evaluation (skipped on resume — server.history already has it).
    if start_round == 0:
        print("\nEvaluating initial model...")
        initial_clean = server.evaluate()
        print(f"Initial Performance - Clean Accuracy: {initial_clean:.4f}")

    print("\n" + "=" * 50)
    print("Starting Federated Learning Rounds")
    print("=" * 50)

    try:
        for round_num in range(start_round, config['num_rounds']):
            round_log = server.run_round(round_num)

            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['aggregation'].get('aggregated_update_norm', 0.0))
            progressive_metrics['cse'].append(round_log.get('classification_semantic_entropy'))

            # Atomic write — a kill mid-save leaves the previous checkpoint intact.
            try:
                save_round_checkpoint(
                    server=server,
                    progressive_metrics=progressive_metrics,
                    config=config,
                    results_dir=results_dir,
                    next_round=round_num + 1,
                    subdir=ckpt_subdir,
                )
            except Exception as e:  # noqa: BLE001 — never let checkpointing kill training
                print(f"  [resume] Warning: checkpoint save failed: {type(e).__name__}: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

    attacker_ids = [
        c.client_id for c in server.clients
        if getattr(c, 'is_attacker', False)
    ]
    # Post-hoc detection quality (None for FedAvg / no-attack).
    detection_summary = compute_detection_summary(server.log_data, attacker_ids)
    results_data = {
        'config': config,
        'results': server.log_data,
        'progressive_metrics': progressive_metrics,
        'local_accuracies': server.history['local_accuracies'],
        'local_cse': server.history.get('local_cse', {}),
        'attacker_ids': attacker_ids,
        'detection_summary': detection_summary,
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print_detection_summary(detection_summary)

    save_global_model_checkpoint(server, config, results_dir)

    run_perplexity_eval_if_configured(config, results_dir)

    run_downstream_task2_if_configured(config, results_dir)

    attacker_ids = [client.client_id for client in server.clients 
                   if getattr(client, 'is_attacker', False)]
    print_detailed_statistics(server.log_data, progressive_metrics, 
                            server.history['local_accuracies'], attacker_ids, 
                            config['experiment_name'], results_dir)
    
    print("\n" + "=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)
    
    visualizer = ExperimentVisualizer(results_dir=results_dir)
    
    visualizer.generate_all_figures(
        server_log_data=server.log_data,
        local_accuracies=server.history['local_accuracies'],
        attacker_ids=attacker_ids,
        experiment_name=config['experiment_name'],
        num_rounds=config['num_rounds'],
        attack_start_round=config['attack_start_round'],
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers']
    )
    
    return server.log_data, progressive_metrics

def _rank_auroc(pos_scores: List[float], neg_scores: List[float]) -> Optional[float]:
    """
    AUROC via the Mann-Whitney U statistic: P(pos > neg), ties count 0.5.

    1.0 = suspicion score perfectly ranks every attacker above every benign
    client; 0.5 = chance; <0.5 = signal points the wrong way. O(n*m) pairwise
    comparison — trivially cheap for FL-sized N, no sklearn dependency.
    """
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def compute_detection_summary(server_log_data, attacker_ids) -> Optional[Dict]:
    """
    Post-hoc detection quality, scored against the ground-truth attacker ids.

    Reads only what the aggregator actually applied. All three trust modes log
    the per-client CSE ratio and CSE flag ('v4_ratio' / 'v4_flagged'), so every
    HMP-GAE round contributes CSE-flag recall/FPR plus the AUROC of the CSE
    ratio. V8 rounds also log 'v8_propagated_flagged', which adds the
    hypergraph tier's incremental recall/FPR/precision — the statistic that
    decides whether propagation did anything (docs/DECISION.md). FedAvg,
    baseline-defense, and fallback rounds carry no CSE decision and are
    skipped. Returns None when there are no attackers or no scored round.

    NOTE: this reads the aggregation log, so every key it consumes must stay in
    server.py's persistence whitelist. It previously keyed on the geometry
    'gate'/'sus_z' diagnostics, which the V1-V3 stack owned and which were
    removed with it on 2026-08-11.
    """
    atk = {int(a) for a in (attacker_ids or [])}
    if not atk:
        return None
    per_round = []
    for log in server_log_data:
        agg = log.get('aggregation') or {}
        cids = agg.get('accepted_clients')
        flags = agg.get('v4_flagged')
        if not (isinstance(cids, list) and isinstance(flags, list)
                and len(cids) == len(flags) and len(cids) > 0):
            continue
        is_atk = [int(cid) in atk for cid in cids]
        seed_b = [bool(v) for v in flags]
        entry = {
            'round': log.get('round'),
            # Tier 1 — the CSE flag, shared by V4/V5/V8.
            'seed_atk': sum(s and a for s, a in zip(seed_b, is_atk)),
            'seed_bgn': sum(s and not a for s, a in zip(seed_b, is_atk)),
            'atk_total': sum(is_atk),
            'bgn_total': len(is_atk) - sum(is_atk),
            # Denominators for tier 2: only non-seeds are still reachable.
            'atk_nonseed': sum(a and not s for a, s in zip(is_atk, seed_b)),
            'bgn_nonseed': sum((not a) and not s
                               for a, s in zip(is_atk, seed_b)),
        }
        ratios = agg.get('v4_ratio')
        if isinstance(ratios, list) and len(ratios) == len(cids):
            entry['ratio_auroc'] = _rank_auroc(
                [r for r, a in zip(ratios, is_atk) if a],
                [r for r, a in zip(ratios, is_atk) if not a],
            )
        propagated = agg.get('v8_propagated_flagged')
        if isinstance(propagated, list) and len(propagated) == len(cids):
            # Tier 2 — V8 only; absent under V4/V5.
            prop_b = [bool(v) for v in propagated]
            entry['prop_atk'] = sum(p and a for p, a in zip(prop_b, is_atk))
            entry['prop_bgn'] = sum(p and not a for p, a in zip(prop_b, is_atk))
        per_round.append(entry)
    if not per_round:
        return None

    def _mean_of(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _sum(key, rows):
        return int(sum(r.get(key, 0) for r in rows))

    second_half = per_round[len(per_round) // 2:]
    seed_atk, seed_bgn = _sum('seed_atk', per_round), _sum('seed_bgn', per_round)
    atk_total, bgn_total = _sum('atk_total', per_round), _sum('bgn_total', per_round)
    result = {
        'n_rounds_scored': len(per_round),
        # Client-round counts summed over every scored round.
        'seed_attacker_client_rounds': seed_atk,
        'seed_benign_client_rounds': seed_bgn,
        'seed_attacker_recall': seed_atk / atk_total if atk_total else None,
        'seed_benign_fpr': seed_bgn / bgn_total if bgn_total else None,
        # Ranking quality of the raw statistic, before the flag threshold.
        'cse_ratio_auroc_mean': _mean_of('ratio_auroc', per_round),
        'cse_ratio_auroc_mean_2nd_half': _mean_of('ratio_auroc', second_half),
        'per_round': per_round,
    }
    v8_rows = [r for r in per_round if 'prop_atk' in r]
    if v8_rows:
        prop_atk, prop_bgn = _sum('prop_atk', v8_rows), _sum('prop_bgn', v8_rows)
        atk_nonseed = _sum('atk_nonseed', v8_rows)
        bgn_nonseed = _sum('bgn_nonseed', v8_rows)
        result['v8_decision_summary'] = {
            'n_rounds': len(v8_rows),
            'propagated_attacker_client_rounds': prop_atk,
            'propagated_benign_client_rounds': prop_bgn,
            # Incremental over the seeds: denominators exclude seeded clients.
            'propagated_incremental_recall': (
                prop_atk / atk_nonseed if atk_nonseed else None
            ),
            'propagated_benign_fpr': (
                prop_bgn / bgn_nonseed if bgn_nonseed else None
            ),
            'propagated_precision': (
                prop_atk / (prop_atk + prop_bgn)
                if prop_atk + prop_bgn else None
            ),
            # 0 here means the hypergraph never changed V5's decision.
            'rounds_with_any_propagation': sum(
                (r['prop_atk'] + r['prop_bgn']) > 0 for r in v8_rows
            ),
        }
    return result


def print_detection_summary(summary: Optional[Dict]) -> None:
    if not summary:
        return
    fmt = lambda v: 'n/a' if v is None else f"{v:.3f}"  # noqa: E731
    print("\n" + "-" * 60)
    print("🔍 DETECTION SUMMARY (applied decision vs ground-truth attackers)")
    print("-" * 60)
    print(f"  rounds scored      : {summary['n_rounds_scored']}")
    print(
        f"  CSE flag recall/FPR: {fmt(summary['seed_attacker_recall'])} / "
        f"{fmt(summary['seed_benign_fpr'])}  → want 1.0 / 0.0"
    )
    print(f"  CSE ratio AUROC    : {fmt(summary['cse_ratio_auroc_mean'])}"
          f"  (2nd half {fmt(summary['cse_ratio_auroc_mean_2nd_half'])})"
          f"  [1.0 perfect, 0.5 chance]")
    v8 = summary.get('v8_decision_summary')
    if v8:
        print("  V8 hypergraph tier (client-rounds beyond the CSE flags):")
        print(
            f"    propagated recall/FPR  : "
            f"{fmt(v8['propagated_incremental_recall'])} / "
            f"{fmt(v8['propagated_benign_fpr'])}"
        )
        print(
            f"    propagated precision   : "
            f"{fmt(v8['propagated_precision'])} "
            f"({v8['rounds_with_any_propagation']}/{v8['n_rounds']} rounds active)"
        )


def print_detailed_statistics(server_log_data, progressive_metrics, local_accuracies, attacker_ids,
                             experiment_name='experiment', results_dir=None):
    """Print per-round metric tables and save them as CSVs for multi-run comparison."""
    import csv
    from pathlib import Path
    
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)
    
    print("\n" + "=" * 80)
    print("📊 DETAILED EXPERIMENT STATISTICS FOR DATA COLLECTION")
    print("=" * 80)
    
    rounds = progressive_metrics['rounds']
    if not rounds:
        print("⚠️  No rounds completed.")
        return
    
    all_client_ids = set()
    for log in server_log_data:
        if 'local_accuracies' in log:
            all_client_ids.update(log['local_accuracies'].keys())
        if 'aggregation' in log and 'similarities' in log['aggregation']:
            similarities = log['aggregation'].get('similarities', [])
            accepted = log['aggregation'].get('accepted_clients', [])
            all_client_ids.update(accepted)
    
    if local_accuracies:
        all_client_ids.update(local_accuracies.keys())
    
    all_client_ids = sorted(all_client_ids)
    attacker_ids_set = set(attacker_ids) if attacker_ids else set()
    
    # ========== 1. Global Accuracy Table ==========
    print("\n" + "-" * 80)
    print("1️⃣  GLOBAL ACCURACY (Per Round)")
    print("-" * 80)
    print(f"{'Round':<8} | {'Clean Accuracy':<15} | {'Accuracy Change':<17}")
    print("-" * 80)
    
    clean_acc = progressive_metrics['clean_acc']
    for i, r in enumerate(rounds):
        acc = clean_acc[i] if i < len(clean_acc) else 0.0
        acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
        print(f"{r:<8} | {acc:<15.6f} | {acc_change:>+17.6f}")
    
    print("-" * 80)
    if clean_acc:
        print(f"Summary: Initial={clean_acc[0]:.6f}, Final={clean_acc[-1]:.6f}, "
              f"Best={max(clean_acc):.6f}, Change={clean_acc[-1]-clean_acc[0]:+.6f}")
    
    # ========== 2. Cosine Similarity Table ==========
    print("\n" + "-" * 80)
    print("2️⃣  COSINE SIMILARITY (Per Round, Per Client)")
    print("-" * 80)
    
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        aggregation = log.get('aggregation', {})
        similarities = aggregation.get('similarities', [])
        accepted = aggregation.get('accepted_clients', [])
        
        all_clients_round = sorted(set(accepted))
        sim_map = {}
        if len(similarities) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                sim_map[cid] = similarities[idx]
        
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            sim = sim_map.get(cid, 0.0)
            row += f"{sim:<14.6f} | "
        
        sim_values = [sim_map.get(cid, 0.0) for cid in all_client_ids if cid in sim_map]
        mean_sim = np.mean(sim_values) if sim_values else 0.0
        std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
        
        row += f"{mean_sim:<6.6f} | {std_sim:.6f}"
        print(row)
    
    print("-" * 80)
    
    # ========== 2b. Euclidean Distance Table ==========
    print("\n" + "-" * 80)
    print("2b. EUCLIDEAN DISTANCE (Per Round, Per Client)")
    print("-" * 80)
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    for log in server_log_data:
        round_num = log['round']
        aggregation = log.get('aggregation', {})
        euclidean_distances = aggregation.get('euclidean_distances', [])
        accepted = aggregation.get('accepted_clients', [])
        all_clients_round = sorted(set(accepted))
        dist_map = {}
        if len(euclidean_distances) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                dist_map[cid] = euclidean_distances[idx]
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            d = dist_map.get(cid, 0.0)
            row += f"{d:<14.6f} | "
        dist_values = [dist_map.get(cid, 0.0) for cid in all_client_ids if cid in dist_map]
        mean_d = np.mean(dist_values) if dist_values else 0.0
        std_d = np.std(dist_values) if len(dist_values) > 1 else 0.0
        row += f"{mean_d:<6.6f} | {std_d:.6f}"
        print(row)
    print("-" * 80)
    
    # ========== 2c. Global Loss (Per Round) ==========
    print("\n" + "-" * 80)
    print("2c. GLOBAL LOSS (Per Round)")
    print("-" * 80)
    print(f"{'Round':<8} | {'Global Loss':<15}")
    print("-" * 80)
    for log in server_log_data:
        round_num = log['round']
        global_loss = log.get('global_loss', 0.0)
        print(f"{round_num:<8} | {global_loss:<15.6f}")
    print("-" * 80)
    
    # ========== 3. Local Accuracy Table ==========
    print("\n" + "-" * 80)
    print("3️⃣  LOCAL ACCURACY (Per Round, Per Client)")
    print("-" * 80)
    
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        local_accs_round = log.get('local_accuracies', {})
        
        row = f"{round_num:<6} | "
        acc_values = []
        for cid in all_client_ids:
            acc = local_accs_round.get(cid, 0.0)
            acc_values.append(acc)
            row += f"{acc:<14.6f} | "
        
        mean_acc = np.mean(acc_values) if acc_values else 0.0
        std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
        row += f"{mean_acc:<6.6f} | {std_acc:.6f}"
        print(row)

    print("-" * 80)

    # ========== 4. Aggregate Averages (across ALL rounds) ==========
    print("\n" + "-" * 80)
    print("4️⃣  AGGREGATE AVERAGES (across all rounds)")
    print("-" * 80)

    global_mean = float(np.mean(clean_acc)) if clean_acc else 0.0
    global_std = float(np.std(clean_acc)) if len(clean_acc) > 1 else 0.0

    benign_vals = []
    attacker_vals = []
    for log in server_log_data:
        for cid, acc in log.get('local_accuracies', {}).items():
            if cid in attacker_ids_set:
                attacker_vals.append(acc)
            else:
                benign_vals.append(acc)

    benign_mean = float(np.mean(benign_vals)) if benign_vals else 0.0
    benign_std = float(np.std(benign_vals)) if len(benign_vals) > 1 else 0.0
    attacker_mean = float(np.mean(attacker_vals)) if attacker_vals else 0.0
    attacker_std = float(np.std(attacker_vals)) if len(attacker_vals) > 1 else 0.0

    seen_clients = set(all_client_ids)
    n_attackers = len(attacker_ids_set & seen_clients)
    n_benign = len(seen_clients) - n_attackers
    n_rounds = len(server_log_data)

    print(f"Global model Clean Accuracy        (mean over {len(clean_acc)} rounds): "
          f"{global_mean:.6f}  ± {global_std:.6f}")
    print(f"Benign clients Local Accuracy      (mean over {n_benign} benign × {n_rounds} rounds = {len(benign_vals)} values): "
          f"{benign_mean:.6f}  ± {benign_std:.6f}")
    if n_attackers > 0:
        print(f"Attacker clients Local Accuracy   (mean over {n_attackers} attacker × {n_rounds} rounds = {len(attacker_vals)} values): "
              f"{attacker_mean:.6f}  ± {attacker_std:.6f}")
    else:
        print("Attacker clients Local Accuracy:    N/A (no attackers configured)")
    print("-" * 80)

    # ========== 5. Save to CSV files for easy import ==========
    print("\n" + "-" * 80)
    print("💾 SAVING DATA TO CSV FILES FOR EASY COLLECTION")
    print("-" * 80)
    
    csv_path1 = results_dir / f"{experiment_name}_global_accuracy.csv"
    with open(csv_path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Clean_Accuracy', 'Accuracy_Change'])
        for i, r in enumerate(rounds):
            acc = clean_acc[i] if i < len(clean_acc) else 0.0
            acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
            writer.writerow([r, f"{acc:.6f}", f"{acc_change:.6f}"])
    print(f"✅ Global Accuracy saved to: {csv_path1}")
    
    csv_path2 = results_dir / f"{experiment_name}_cosine_similarity.csv"
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                                           for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            aggregation = log.get('aggregation', {})
            similarities = aggregation.get('similarities', [])
            accepted = aggregation.get('accepted_clients', [])
            
            all_clients_round = sorted(set(accepted))
            sim_map = {}
            if len(similarities) == len(all_clients_round):
                for idx, cid in enumerate(all_clients_round):
                    sim_map[cid] = similarities[idx]
            
            row = [round_num]
            sim_values = []
            for cid in all_client_ids:
                sim = sim_map.get(cid, 0.0)
                sim_values.append(sim)
                row.append(f"{sim:.6f}")
            
            mean_sim = np.mean(sim_values) if sim_values else 0.0
            std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
            row.extend([f"{mean_sim:.6f}", f"{std_sim:.6f}"])
            writer.writerow(row)
    print(f"✅ Cosine Similarity saved to: {csv_path2}")
    
    csv_path3 = results_dir / f"{experiment_name}_local_accuracy.csv"
    with open(csv_path3, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                             for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            local_accs_round = log.get('local_accuracies', {})
            
            row = [round_num]
            acc_values = []
            for cid in all_client_ids:
                acc = local_accs_round.get(cid, 0.0)
                acc_values.append(acc)
                row.append(f"{acc:.6f}")
            
            mean_acc = np.mean(acc_values) if acc_values else 0.0
            std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
            row.extend([f"{mean_acc:.6f}", f"{std_acc:.6f}"])
            writer.writerow(row)
    print(f"✅ Local Accuracy saved to: {csv_path3}")

    csv_path4 = results_dir / f"{experiment_name}_aggregate_averages.csv"
    with open(csv_path4, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std', 'N_values'])
        writer.writerow(['Global_Clean_Accuracy', f"{global_mean:.6f}", f"{global_std:.6f}", len(clean_acc)])
        writer.writerow(['Benign_Local_Accuracy', f"{benign_mean:.6f}", f"{benign_std:.6f}", len(benign_vals)])
        if n_attackers > 0:
            writer.writerow(['Attacker_Local_Accuracy', f"{attacker_mean:.6f}", f"{attacker_std:.6f}", len(attacker_vals)])
        else:
            writer.writerow(['Attacker_Local_Accuracy', 'N/A', 'N/A', 0])
    print(f"✅ Aggregate Averages saved to: {csv_path4}")

    print("\n" + "=" * 80)
    print("✅ All statistics printed and saved to CSV files!")
    print("   You can now easily collect data from multiple runs and compare them.")
    print("=" * 80)

def analyze_results(metrics):
    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    
    rounds = metrics['rounds']
    if not rounds:
        print("No rounds completed.")
        return

    clean = metrics['clean_acc']

    print(f"Total Rounds: {len(rounds)}")
    print(f"Final Clean Accuracy: {clean[-1]:.4f}")
    if len(clean) > 1:
        print(f"Best Clean Accuracy: {max(clean):.4f}")
        print(f"Accuracy Change: {clean[-1] - clean[0]:+.4f}")

def main():
    # SINGLE authoritative config source — no override path exists (config_overrides /
    # COLAB_CONFIG_OVERRIDES / run_suite() were all removed 2026-08-07). To change ANY
    # parameter, including an A/B arm: edit here, run, edit back. Every arm SHOULD get
    # its own experiment_name for readable artifacts; fed_resume also fingerprints the
    # complete defense_config so changing the trust mode cannot silently reuse state.
    #
    # RESUME GOTCHA: fed_resume fingerprints ~43 top-level keys AND the whole
    # defense_config dict, so editing ANY of them — including a key marked
    # "inert" below — invalidates an existing round checkpoint. The run then
    # prints "[resume] Starting fresh" and restarts at round 0 instead of
    # silently mixing trajectories. Mid-run edits are not resumable by design.
    #
    # CURRENT ARM (user-requested): V8 dual-view HMP + CSE-seeded propagation.
    # Ablation arms: 'v4_cse_reject' (detect-then-suppress, constant multiplier)
    # and 'v5_cse_reject' (V8 minus the hypergraph — THE matched-run baseline
    # for any hypergraph-attributable claim, docs/DECISION.md).
    config = {
        # ========== Experiment ==========
        'experiment_name': 'agnews-(non-iid0.5)-hmpgae-v8-hallu(localround=1,seed=42,r50,len128,flip0.3-0.8)-llama3.2-1b',
        'seed': 42,

        # ========== Federated Learning Setup ==========
        'num_clients': 7,    # 5 benign + 2 attackers (canonical arm)
        'num_attackers': 2,  # the LAST client ids are the attackers
        'num_rounds': 50,    # paper regime; 10-round runs are smoke tests

        # ========== Training Hyperparameters ==========
        'client_lr': 5e-5,
        'server_lr': 1.0,
        'batch_size': 32,        # fixed across all runs for comparability
        'test_batch_size': 64,
        'local_epochs': 1,
        'grad_clip_norm': 1.0,   # reduce to 0.5 if NaN
        'alpha': 0.0,            # FedProx μ; 0 = plain FedAvg local step
        
        # ========== Dataset ==========
        # Exactly ONE of the four blocks below stays uncommented. The three keys
        # move together and nothing validates the pairing: a wrong num_labels
        # gives a bare KeyError in the partitioner or a silently mis-sized head.
        # data_loader.py matches 'imdb' / 'dbpedia' / 'yahoo_answers' by name and
        # treats every other string (including 'ag_news') as AG News, so a typo
        # silently loads AG News.
        # WARNING: if you uncomment a second block without commenting this one,
        # Python keeps the LAST assignment of each key — no error, wrong run.
        # Remember to update experiment_name and both checkpoint subdirs too.

        # -- AG News: 4 classes, news topic classification (current arm) --
        'dataset': 'ag_news',
        'num_labels': 4,
        'max_length': 128,

        # -- Yahoo Answers: 10 classes, question topic classification --
        # 128 (not 256) is deliberate: it keeps sequence length constant across
        # datasets so runs stay comparable. 256 is a separate ablation.
        # 'dataset': 'yahoo_answers',
        # 'num_labels': 10,
        # 'max_length': 128,

        # -- IMDB: 2 classes, sentiment; long reviews need the longer window --
        # 'dataset': 'imdb',
        # 'num_labels': 2,
        # 'max_length': 512,

        # -- DBpedia 14: 14 classes, ontology classification --
        # 'dataset': 'dbpedia',
        # 'num_labels': 14,
        # 'max_length': 512,

        # ========== Data Distribution ==========
        # Only 'iid' is matched by name; every other value takes the Dirichlet
        # branch, so 'non-iid' is a label for readers, not a checked keyword.
        'data_distribution': 'non-iid',  # 'iid' | 'non-iid' (Dirichlet)
        'dirichlet_alpha': 0.5,          # lower = more heterogeneous; read only on the non-IID branch
        # Held fixed across datasets for comparability; None = full. NOTE: this
        # also caps the TEST split to 15% of it (data_loader.py, fixed
        # random_state=42), i.e. the full-test CSE that drives every trust mode
        # is computed over ~1.5k rows here.
        'dataset_size_limit': 10000,

        # ========== Model & LoRA ==========
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16,             # keep at 2*r
        'lora_dropout': 0.1,
        'lora_target_modules': None,  # None = auto-resolve per backbone in models.py
                                      # (Llama/Qwen/Mistral → q/k/v/o_proj; OPT → q/k/v/out_proj;
                                      #  GPT-2 → c_attn/c_proj; Pythia → query_key_value/dense_*;
                                      #  DistilBERT → *_lin; BERT/RoBERTa → query/key/value/dense).
                                      # An unrecognised backbone stays None = PEFT's own default.
        # Any HF sequence-classification repo id loads; these are the verified
        # ones. Encoder: 'distilbert-base-uncased' | 'bert-base-uncased' |
        # 'roberta-base' | 'deberta-v3-base'. Decoder: 'gpt2' |
        # 'EleutherAI/pythia-160m' | 'EleutherAI/pythia-1b' | 'facebook/opt-125m' |
        # 'Qwen/Qwen2.5-0.5B' (ungated, fits T4 15GB) | 'meta-llama/Llama-3.2-1B'
        # (GATED: HF license + HF_TOKEN; fp32 needs A100). PPL needs a decoder.
        'model_name': 'meta-llama/Llama-3.2-1B',
        

        # ========== Attack ==========
        # 'NoAttack' | 'Hallucination' (this paper) | 'SignFlipping' | 'Gaussian'
        # | 'ALIE'. Exact strings — this is the one dispatch that is
        # case-sensitive, and it only runs when num_attackers > 0.
        'attack_method': 'Hallucination',
        # INERT: no attacker reads this. Each family has its own start-round key
        # below; the plotter takes this one and never uses it. Kept only because
        # it is part of the resume fingerprint.
        'attack_start_round': None,

        # Hallucination (label-flipping). Canonical strength — do not change without
        # explicit request: mode 'random', per-round reseed, flip_ratio ~ U[0.3, 0.8].
        # Escalation ladder if too weak: range [0.6,1.0] → [0.8,1.0] →
        # num_attackers=3 AND num_byzantine=3 (the rank cap must cover every
        # attacker, or one is structurally unflaggable — see defense_config).
        'hallu_flip_ratio': 0.5,                     # INERT while ratio_range is set
        'hallu_flip_mode': 'random',                 # 'pairwise' | 'targeted' | 'random'
        'hallu_flip_map': {                         # 'pairwise' mode only; inert under current random mode
            0: 1, 1: 0, 2: 3, 3: 2,
        },
        'hallu_target_class': None,                  # 'targeted' mode only
        'hallu_attack_start_round': 0,
        'hallu_per_round_reseed': True,              # False = legacy frozen-flip behaviour
        'hallu_flip_ratio_range': [0.3, 0.8],        # None → scalar hallu_flip_ratio

        # ---- Classical Byzantine baselines; each key is read only when
        # attack_method selects that family. These forge the UPDATE while
        # leaving the local model benign (SignFlipping trains honestly then
        # negates; Gaussian/ALIE never train), so the CSE-reject modes are
        # structurally blind to them — run them against the geometric baseline
        # defenses, not as evidence about V4/V5/V8. ----
        'sign_flip_scale': 10.0,                 # ICML '18: malicious = -scale * g_own
        'sign_flip_attack_start_round': None,
        'gaussian_std_scale': 5.0,               # USENIX Security '20: noise-std multiplier
        'gaussian_attack_start_round': None,
        'alie_z_max': None,                      # NeurIPS '19: None = auto by (num_clients, num_attackers)
        'alie_attack_start_round': None,

        # ========== Defense ==========
        # 'fedavg' | 'hmp_gae' (this paper) | 'krum' | 'multi_krum' | 'coord_median'
        # | 'fltrust' | 'foolsgold'. Matching is case-insensitive and accepts
        # separator variants ('multikrum', 'coord-median', ...); 'none' is an
        # alias for 'fedavg'. Anything else raises in defense.build_defense.
        'defense_method': 'hmp_gae',
        'defense_config': {
            # -- Baseline-defense knobs — inert under hmp_gae, EXCEPT num_byzantine:
            # the CSE-reject family reuses it as its rank cap (must be < N/2).
            'epsilon': 1e-6,       # foolsgold
            'anchor': 'median',    # fltrust
            'num_byzantine': 2,    # krum/multi-krum; ALSO the V4/V5/V8 rank cap.
                                   # It bounds how many clients can be flagged
                                   # per round, so keep it >= num_attackers.

            # -- trust_mode — the aggregation decision rule, and the only knob
            # that selects between the three shipped defense versions. Mechanics:
            # hmp_gae/trust_scorer.py; design history + pre-registered constants:
            # docs/DECISION.md. There is NO default: an unknown or missing value
            # raises. All three share one detection statistic (per-client
            # full-test CSE, pool-median normalised into a ratio, rank-capped by
            # num_byzantine) and all three need that CSE every round before
            # aggregation — the server computes it and crashes loudly if it
            # cannot. All three read: num_byzantine, keep_min, v4_tau_ratio.
            #
            #   'v4_cse_reject'          V4 — flag = top-k by CSE ratio AND ratio
            #                            > v4_tau_ratio; flagged clients take the
            #                            constant v4_reject_mult. Stateless.
            #                            The detect-then-suppress ablation arm.
            #                            Extra knob: v4_reject_mult.
            #
            #   'v5_cse_reject'          V5 — identical flag decision, but the
            #                            penalty is a linear ramp in the ratio
            #                            (mild just past tau, v5_m_floor once
            #                            ratio >= v5_r_hard). Stateless. This is
            #                            V8 minus the hypergraph, i.e. THE
            #                            matched-run baseline for any
            #                            hypergraph-attributable claim.
            #                            Extra knobs: v5_m_floor, v5_r_hard.
            #
            #   'v8_hmp_cse_propagation' V8 (current arm) — V5 runs first and its
            #                            flags become immutable risk seeds on a
            #                            hypergraph built where the raw-update
            #                            view and the probe-behavior view agree,
            #                            denoised by a small GAE trained online.
            #                            A non-seed peer is softly penalised only
            #                            with a seed, positive propagated risk,
            #                            its own ratio > 1, and leftover rank-cap
            #                            budget; otherwise V8 returns V5's
            #                            weights exactly. The only mode with
            #                            cross-round state and the only one that
            #                            requires semantic_weight > 0.
            #                            Extra knobs: v5_*, plus the whole
            #                            hypergraph/GAE block below.
            'trust_mode': 'v8_hmp_cse_propagation',
            'keep_min': 1,               # defensive floor on unflagged clients

            # -- CSE-decision constants — PRE-REGISTERED, do not re-tune
            # (calibration: docs/DECISION.md "V4" / "V5"). v4_tau_ratio applies
            # to all three modes; the other three are per-version, as marked.
            'v4_tau_ratio': 1.85,
            'v4_reject_mult': 0.10,  # V4 only; 0.0 = pre-registered hard-removal arm
            'v5_m_floor': 0.10,      # V5/V8 ramp floor; never 0.0
            'v5_r_hard': 2.5,        # V5/V8 ramp saturation ratio

            # -- V8 hypergraph + GAE (symbols match hmp_gae/*.py and
            # docs/MATH_LOGIC.md; every key below is read only under V8)
            'proj_dim': 64,
            'eta_dim': 64,
            'random_proj_seed': 42,
            'knn_k': 2,                  # k=2 keeps the consensus views sharp at N=7
            'hidden_dim': 64,
            'latent_dim': 32,
            'num_hmp_layers': 2,
            'train_steps_per_round': 5,
            'train_lr': 1e-3,
            'lambda_H': 1.0,
            'lambda_A': 1.0,
            'lambda_hist': 0.5,
            'weight_decay': 1e-5,
            'hist_ema_beta': 0.9,        # z_hist EMA (node-feature history + L_hist)

            # -- Probe (V8's behavior view; probe_cse diagnostic otherwise)
            'semantic_weight': 1.0,      # a SWITCH, not a weight: only "> 0" is
                                         # read. >0 turns on the per-round probe
                                         # forward; V8 rejects <= 0 on the first
                                         # aggregate() call.
            'semantic_probe_stratified': True,   # labels only balance the probe, never score

            'device': 'cpu',             # small N: CPU beats GPU round-trips
        },
        # Semantic-probe size (server-owned; sampling mode = semantic_probe_stratified).
        'semantic_probe_size': 100,
        # Which update-similarity metric the server logs for the figures:
        # 'pairwise' | 'local_vs_global' | 'both' (anything else falls back to
        # 'pairwise'). Diagnostic only — no defense reads it.
        'server_similarity_mode': 'pairwise',

        # ========== Evaluation ==========
        # Global CSE: every round, free — it shares the global test forward.
        # Per-client CSE: N extra full-test forwards every round, and it is the
        # statistic every trust mode decides on, so it is the real per-round cost.
        # PPL: once after FL via backbone → CausalLM transfer
        # (decoder_adapters.py); needs the global checkpoint and a decoder backbone.
        'eval_classification_semantic_entropy': True,  # gates the GLOBAL CSE log only
        'eval_local_every_n_rounds': 1,   # k>1 = sparser per-client eval, but IGNORED
                                          # under hmp_gae: all three trust modes need
                                          # per-client CSE every round
        'eval_perplexity': True,
        'ppl_num_samples': 200,
        'ppl_seed': 42,
        'ppl_max_length': None,           # None → config['max_length']

        # ========== Checkpoints ==========
        'save_global_checkpoint': True,   # needed for PPL / downstream eval
        'global_checkpoint_subdir': 'global_checkpoint_agnews_llama_v8_seed42',
        # Per-round resume snapshot (Colab resilience; fingerprint guard: fed_resume.py)
        'save_round_checkpoint': True,
        'resume_from_checkpoint': True,   # False = force a fresh run
        'round_checkpoint_subdir': 'round_checkpoint_agnews_llama_v8_seed42',
        # ========== Task 2: optional downstream generation after FL ==========
        'run_downstream_after_fl': False,   # subprocess run_downstream_generation.py
        'downstream_probes': None,          # probe JSON path; None skips Task 2
        'downstream_output': None,          # None → results/<experiment_name>_downstream_gen.jsonl
        'downstream_device': None,          # None → cuda if available else cpu
        'downstream_cli_args': [
            '--stable',
        ],

    }

    print("[config] experiment_name:", config['experiment_name'])
    print("[config] defense_method:", config['defense_method'])
    print("[config] trust_mode:", config['defense_config'].get('trust_mode'))

    attack_method = config.get('attack_method', 'Hallucination')
    if config.get('num_attackers', 0) > 0 and attack_method != 'NoAttack':
        if attack_method == 'Hallucination':
            print("Running Hallucination Attack (label-flipping, this paper)...")
        elif attack_method == 'ALIE':
            print("Running ALIE Attack (Model Poisoning Baseline)...")
        elif attack_method == 'SignFlipping':
            print("Running Sign-Flipping Attack (Model Poisoning Baseline)...")
        elif attack_method == 'Gaussian':
            print("Running Gaussian Attack (Random Model Poisoning Baseline)...")
        else:
            print(f"Running attack: {attack_method}")
    else:
        print("Running Baseline Experiment (No Attack)...")
    
    results, metrics = run_experiment(config)
    analyze_results(metrics)


if __name__ == "__main__":
    main()
