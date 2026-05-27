# main.py
# This script sets up and runs a federated learning experiment with a progressive GRMP attack.

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

# Initialize experiment components
def setup_experiment(config):
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Setting up Experiment: {config['experiment_name']}")
    print("=" * 50)

    # 1. Initialize Data Manager
    # dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers' — select dataset; num_labels and max_length must match (see config below)
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

    # 2. Partition data among clients
    # Supports both IID and Non-IID distributions based on config
    data_distribution = config.get('data_distribution', 'non-iid').lower()
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    num_labels = config.get('num_labels', 4)
    num_clients = config['num_clients']
    num_attackers = config.get('num_attackers', 0)
    num_benign = num_clients - num_attackers
    
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    
    client_indices = {i: [] for i in range(num_clients)}
    
    if data_distribution == 'iid':
        # ========== IID Distribution: Uniform Random Partition ==========
        # Each client gets approximately equal number of samples with similar label distribution
        print("\nPartitioning data (IID distribution)...")
        
        # Shuffle all indices
        all_indices = indices.copy()
        rng.shuffle(all_indices)
        
        # Calculate samples per client (approximately equal)
        total_samples = len(all_indices)
        base_samples = total_samples // num_clients
        remainder = total_samples % num_clients
        
        # Assign samples to each client
        start_idx = 0
        for client_id in range(num_clients):
            # First 'remainder' clients get one extra sample
            extra = 1 if client_id < remainder else 0
            end_idx = start_idx + base_samples + extra
            client_indices[client_id] = all_indices[start_idx:end_idx].tolist()
            start_idx = end_idx
        
        # Print distribution statistics
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

        if num_benign < num_clients:
            print("\n  [Note] Attackers are assigned only data *quantities* (sizes) for the experimental setup. "
                  "In reality, attackers do NOT perform local training and do NOT use these local data "
                  "(dataset-free). They also do NOT access other local agents' data.")
    
    else:
        # ========== Non-IID Distribution: Dirichlet-based Partition ==========
        # Per paper: "heterogeneous IoA system" with heterogeneous data distributions
        print("\nPartitioning data (Non-IID distribution)...")
        
        # Use Dirichlet distribution to create heterogeneous data
        # Each client gets data with different label distributions
        dirichlet_alpha = config['dirichlet_alpha']
        
        # Partition data by label first
        label_indices = {label: [] for label in range(num_labels)}
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # Assign samples to clients using Dirichlet distribution for non-IID
        for label in range(num_labels):
            label_list = np.array(label_indices[label])
            rng.shuffle(label_list)
            
            # Generate proportions for each client using Dirichlet distribution
            # Lower dirichlet_alpha creates more heterogeneous (non-IID) distribution
            proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
            proportions = np.cumsum(proportions)
            proportions[-1] = 1.0  # Ensure last is exactly 1.0
            
            # Assign samples based on proportions
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = int(len(label_list) * proportions[client_id])
                client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
                start_idx = end_idx
        
        # Shuffle within each client to mix labels (but distribution remains non-IID)
        for client_id in range(num_clients):
            client_list = np.array(client_indices[client_id])
            rng.shuffle(client_list)
            client_indices[client_id] = client_list.tolist()
        
        # Print distribution statistics
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

        # Clarification: attackers are dataset-free
        if num_benign < num_clients:
            print("\n  [Note] Attackers are assigned only data *quantities* (sizes) following the non-IID distribution, "
                  "for experimental setup. In reality, attackers do NOT perform local training and do NOT use "
                  "these local data (dataset-free). They also do NOT access other local agents' data.")

    # 3. Get global test loader
    test_loader = data_manager.get_test_loader()

    # 4. Initialize Global Model
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

    # 5. Initialize Server
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
        eval_local_every_n_rounds=int(config.get('eval_local_every_n_rounds', 1)),
    )

    # 6. Create Clients
    print("\nCreating federated learning clients...")
    num_attackers = config.get('num_attackers', 0)  # Allow 0 attackers for baseline experiment
    attack_method = config.get('attack_method', 'Hallucination')

    # 'NoAttack' is a first-class no-op: it forces every client to be benign even
    # when num_attackers>0, so the (num_attackers=2, attack_method='NoAttack')
    # combo from notebook overrides doesn't fall through the attacker dispatch.
    if attack_method == 'NoAttack' and num_attackers > 0:
        print(f"  [config] attack_method='NoAttack' overrides num_attackers={num_attackers}: "
              f"all {config['num_clients']} clients will be benign.")
        effective_num_attackers = 0
    else:
        effective_num_attackers = num_attackers

    for client_id in range(config['num_clients']):
        # Determine if benign or attacker
        # Logic: Last 'effective_num_attackers' clients are attackers
        # If effective_num_attackers=0, all clients are benign (baseline experiment)
        if client_id < (config['num_clients'] - effective_num_attackers):
            # --- Benign Client ---
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]
            
            # Create static dataloader for benign client
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
            # --- Attacker Client ---
            # attack_method is resolved once before the loop above.
            # Use the actual assigned data size as claimed size (realistic scenario:
            # attackers do not exaggerate their contribution weight).
            claimed_data_size = len(client_indices[client_id])

            # Create attacker based on attack_method
            if attack_method == 'ALIE':
                # ========== ALIE Attack Client ==========
                from attack.alie import ALIEAttackerClient
                print(f"  Client {client_id}: ATTACKER (ALIE Attack)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                
                # Get ALIE-specific parameters
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
                # ========== Sign-Flipping Attack Client (ICML '18: g^byz = -scale * g_own) ==========
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
                # ========== Hallucination Attack (Label-Flipping, this paper) ==========
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
                # Per-round randomization knobs (None / False values reproduce
                # the original frozen-flip behaviour exactly).
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
                # ========== Gaussian (Random Model Poisoning) Attack - USENIX Security '20 ==========
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


# Run the experiment
def run_experiment(config):
    server, results_dir = setup_experiment(config)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'acc_diff': [],
        'agg_update_norm': [],
        # V2 M7: Classification Semantic Entropy, recorded each round.
        'cse': [],
    }

    # ------------------------------------------------------------------
    # Resume from a previously-saved per-round checkpoint, if available.
    # On Colab the runtime can die at any time; this lets a re-launched
    # run pick up where it left off without re-doing completed rounds.
    # See fed_resume.py for the persisted state and fingerprint guard.
    # ------------------------------------------------------------------
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

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['aggregation'].get('aggregated_update_norm', 0.0))
            progressive_metrics['cse'].append(round_log.get('classification_semantic_entropy'))

            # Persist a resumable snapshot.  Atomic write (.tmp + os.replace)
            # so a kill mid-save leaves the previous good checkpoint intact.
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

            # Memory cleanup after each round
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    results_data = {
        'config': config,
        'results': server.log_data,
        'progressive_metrics': progressive_metrics,
        'local_accuracies': server.history['local_accuracies'],
        'local_cse': server.history.get('local_cse', {}),
        'attacker_ids': [
            c.client_id for c in server.clients
            if getattr(c, 'is_attacker', False)
        ],
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    save_global_model_checkpoint(server, config, results_dir)

    run_perplexity_eval_if_configured(config, results_dir)

    run_downstream_task2_if_configured(config, results_dir)

    # Print detailed statistics for data collection
    attacker_ids = [client.client_id for client in server.clients 
                   if getattr(client, 'is_attacker', False)]
    print_detailed_statistics(server.log_data, progressive_metrics, 
                            server.history['local_accuracies'], attacker_ids, 
                            config['experiment_name'], results_dir)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)
    
    visualizer = ExperimentVisualizer(results_dir=results_dir)
    
    # Generate all figures
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

# Detailed statistics printing for data collection
def print_detailed_statistics(server_log_data, progressive_metrics, local_accuracies, attacker_ids, 
                             experiment_name='experiment', results_dir=None):
    """
    Print detailed statistics for data collection and multi-run comparison.
    Outputs all key metrics in tabular format for easy copying to Excel/CSV.
    
    Args:
        server_log_data: List of round logs from server
        progressive_metrics: Dictionary with progressive metrics
        local_accuracies: Dictionary with local accuracies per client
        attacker_ids: List of attacker client IDs
        experiment_name: Name of the experiment (for file naming)
        results_dir: Path to results directory (default: Path("results"))
    """
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
    
    # Get all client IDs
    all_client_ids = set()
    for log in server_log_data:
        if 'local_accuracies' in log:
            all_client_ids.update(log['local_accuracies'].keys())
        if 'aggregation' in log and 'similarities' in log['aggregation']:
            # Infer client IDs from similarities count (if available)
            similarities = log['aggregation'].get('similarities', [])
            accepted = log['aggregation'].get('accepted_clients', [])
            all_client_ids.update(accepted)
    
    # Also include from local_accuracies history
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
    
    # Prepare header
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
        
        # Create similarity map
        all_clients_round = sorted(set(accepted))
        sim_map = {}
        if len(similarities) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                sim_map[cid] = similarities[idx]
        
        # Print row
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            sim = sim_map.get(cid, 0.0)
            row += f"{sim:<14.6f} | "
        
        # Calculate mean and std for this round
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
    
    # Prepare header
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
        
        # Print row
        row = f"{round_num:<6} | "
        acc_values = []
        for cid in all_client_ids:
            acc = local_accs_round.get(cid, 0.0)
            acc_values.append(acc)
            row += f"{acc:<14.6f} | "
        
        # Calculate mean and std
        mean_acc = np.mean(acc_values) if acc_values else 0.0
        std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
        row += f"{mean_acc:<6.6f} | {std_acc:.6f}"
        print(row)

    print("-" * 80)

    # ========== 4. Aggregate Averages (across ALL rounds) ==========
    # Three headline numbers for cross-run comparison:
    #   - global model Clean Accuracy averaged over all rounds
    #   - benign clients' local accuracy averaged over (round × benign client) pairs
    #   - attacker clients' local accuracy averaged over (round × attacker client) pairs
    # The benign/attacker splits use per-round local_accuracies logged by the server.
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
    
    # Save Global Accuracy
    csv_path1 = results_dir / f"{experiment_name}_global_accuracy.csv"
    with open(csv_path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Clean_Accuracy', 'Accuracy_Change'])
        for i, r in enumerate(rounds):
            acc = clean_acc[i] if i < len(clean_acc) else 0.0
            acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
            writer.writerow([r, f"{acc:.6f}", f"{acc_change:.6f}"])
    print(f"✅ Global Accuracy saved to: {csv_path1}")
    
    # Save Cosine Similarity
    csv_path2 = results_dir / f"{experiment_name}_cosine_similarity.csv"
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
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
    
    # Save Local Accuracy
    csv_path3 = results_dir / f"{experiment_name}_local_accuracy.csv"
    with open(csv_path3, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
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

    # Save Aggregate Averages (the three headline numbers from section 4)
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

# Simple analysis
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

def main(config_overrides: Optional[Dict] = None):
    config = {
        # ========== Experiment Configuration ==========
        # === CURRENT RUN: FedAvg clean-ceiling baseline on AG News (non-IID, no attackers) ===
        # 7 benign clients, no attackers, no defense (plain FedAvg).  Establishes
        # the clean ceiling on non-IID Dirichlet(0.5) AG News that all attack /
        # defense runs on this dataset are compared against.  Because
        # FedAvgDefense ignores trust_scorer entirely, this baseline is
        # unaffected by the ongoing HMP-GAE trust-scoring investigation.
        'experiment_name': 'yahoo-(iid)-hmpgae-defense-attack(localround=1,seed=42,b=0.3,batch=32,64)',
        'seed': 42,  # Random seed for reproducibility

        # ========== Federated Learning Setup ==========
        'num_clients': 7,    # Total clients: 5 benign, 2 attackers (Y2 config)
        'num_attackers': 2,  # SMOKE: 2 attackers (C5/C6), exercises HMP-GAE trust path
        'num_rounds': 50,    # Total federated learning rounds

        # ========== Training Hyperparameters ==========
        'client_lr': 5e-5,   # Learning rate for local client training
        'server_lr': 1.0,    # Server aggregation lr (fixed at 1.0 for standard FedAvg aggregation)
        'batch_size': 32,    # 32 is safe for T4 15GB with Qwen2.5-0.5B + seq_len=128; raise to 64 on A100
        'test_batch_size': 64,   # Inference uses less VRAM; 64 is safe
        'local_epochs': 1,   # 1 epoch per round: 50 rounds × 1 epoch sufficient for LoRA convergence
                             # and keeps total wall-clock time manageable (~3-4 h on T4)
        'grad_clip_norm': 1.0,  # Qwen2.5-0.5B is typically stable at 1.0; reduce to 0.5 if NaN
        'alpha': 0.0,  # FedProx μ: 0 = standard FedAvg local step; >0 penalises local drift from global
        
        # ========== Dataset Configuration ==========
        # Choose dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers' — set num_labels and max_length accordingly
        # Dataset 1: AG News
        # 'dataset': 'ag_news',  # news classification (4 classes)
        # 'num_labels': 4,       # AG News: 4 | IMDB: 2 | DBpedia: 14 | Yahoo Answers: 10
        # 'max_length': 128,     # AG News: 128 | IMDB: 512/256 | DBpedia: 512 | Yahoo Answers: 256
        # -------------------------------------------
        # Dataset 2: IMDB
        # 'dataset': 'imdb',   # sentiment (2 classes)
        # 'num_labels': 2,
        # 'max_length': 512,
        # -------------------------------------------
        # Dataset 3: DBpedia (14 classes, 560K train / 70K test)
        # 'dataset': 'dbpedia',   # topic classification (14 classes)
        # 'num_labels': 14,
        # 'max_length': 512,
        # -------------------------------------------
        # Dataset 4: Yahoo Answers (10 classes, 1.4M train / 60K test)
        'dataset': 'yahoo_answers',   # topic classification (10 classes, yassiracharki/Yahoo_Answers_10_categories_for_NLP)
        'num_labels': 10,       # Yahoo Answers: 10 classes
        'max_length': 128,      # Yahoo Answers: 128, 256 (Q&A text is longer than AG News headlines)
        
        # ========== Data Distribution ==========
        # For V1 first experiment we use IID to isolate the defense effect from data heterogeneity noise.
        # Switch to 'non-iid' with dirichlet_alpha in [0.3, 1.0] once baseline numbers are stable.
        'data_distribution': 'iid',  # 'iid' uniform, 'non-iid' Dirichlet-heterogeneous
        'dirichlet_alpha': 0.5,          # Only used when data_distribution='non-iid'. Lower = more heterogeneous.
        # 'dataset_size_limit': None,  # Full dataset: AG News ~120K; IMDB 25K; DBpedia 560K; Yahoo Answers 1.4M
        'dataset_size_limit': 10000,  # 10K train → ~1428 samples/client (7 clients, IID); test ≤ 1500
                                      # Enough for LoRA convergence on AG News; keeps per-round time ~3-5 min on T4

        # ========== Training Mode Configuration ==========
        'use_lora': True,  # True for LoRA fine-tuning, False for full fine-tuning
        # LoRA parameters (only used when use_lora=True)
        # NOTE: Lower r values = faster training but potentially less capacity
        # Recommended: r=8 for speed, r=16 for better performance (default)
        'lora_r': 8,  # LoRA rank (controls the rank of low-rank matrices). r=8 for speed, r=16/32 for better capacity
        'lora_alpha': 16,  # LoRA alpha (scaling factor, typically 2*r). Must match r: alpha=2*r
        'lora_dropout': 0.1,  # LoRA dropout rate
        'lora_target_modules': None,  # None = use default for DistilBERT (["q_lin", "k_lin", "v_lin", "out_lin"])
        
        # Model configuration
        # Supported models:
        # Encoder-only (BERT-style): 'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'microsoft/deberta-v3-base'
        # 'model_name': 'distilbert-base-uncased',  # distilbert 67M
        # # -------------------------------------------
        # Decoder-only (GPT-style): 'gpt2', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-1b', 'facebook/opt-125m', 'Qwen/Qwen2.5-0.5B'
        # 'model_name': 'gpt2',                      # GPT-2 124M — stable decoder baseline
        # 'model_name': 'EleutherAI/pythia-160m',    # Pythia-160M (may need grad_clip_norm=0.5)
        # 'model_name': 'facebook/opt-125m',         # OPT-125M (Meta)
        'model_name': 'Qwen/Qwen2.5-0.5B',  # Qwen2.5-0.5B ~494M (Alibaba, LLaMA-style arch, Apache 2.0) — use BASE for fine-tuning
        # num_labels and max_length: set above in Dataset Configuration based on chosen dataset
        

        # ========== Attack Configuration ==========
        # Supported: 'NoAttack' | 'Hallucination' (this paper) | 'SignFlipping' | 'Gaussian' | 'ALIE'
        # Current value is 'Hallucination' (paired with num_attackers=2): the
        # proposed per-round randomized label-flipping attack. Switch to
        # 'NoAttack' (with num_attackers=0) for the clean ceiling, or to one
        # of the classical-baseline strings for V2 comparison runs.
        'attack_method': 'Hallucination',
        'attack_start_round': None,  # None = attack active from round 0 (default)

        # ---- Hallucination (label-flipping, this paper's attacker) ----
        # Matches the paper's stealth threat model: ||omega_a - omega'_a|| <= eps is
        # satisfied naturally because the attacker performs standard FedProx local
        # training, only against label-flipped data.
        #
        # The defaults below run the "per-round randomized" variant: each round the
        # attacker resamples (a) which subset of its samples gets flipped and
        # (b) the random target class for each, plus (c) the flip_ratio itself is
        # drawn from hallu_flip_ratio_range.  This produces non-stationary attack
        # gradients, so attacker CSE / local_acc oscillate across rounds instead of
        # smoothly converging.  Set hallu_per_round_reseed=False to recover the
        # original frozen-100%-flip behaviour from the earlier experiments.
        'hallu_flip_ratio': 0.5,                   # used only when hallu_flip_ratio_range is None
        'hallu_flip_mode': 'random',               # 'pairwise' | 'targeted' | 'random'
        'hallu_flip_map': {0: 1, 1: 0, 2: 3, 3: 2},  # only consumed in flip_mode='pairwise'
        'hallu_target_class': None,                # only for flip_mode='targeted'
        'hallu_attack_start_round': 0,
        'hallu_per_round_reseed': True,            # re-sample flipped-label set each round
        'hallu_flip_ratio_range': [0.3, 0.8],      # per-round flip_ratio sampled uniformly here
                                                   # (set to None to use the scalar hallu_flip_ratio)

        # ---- Classical Byzantine baselines (kept for V2 comparison) ----
        'sign_flip_scale': 10.0,                 # ICML '18: malicious = -scale * g_own
        'sign_flip_attack_start_round': None,
        'gaussian_std_scale': 5.0,               # USENIX Security '20: noise-std multiplier
        'gaussian_attack_start_round': None,
        'alie_z_max': None,                      # NeurIPS '19: None = auto by (num_clients, num_attackers)
        'alie_attack_start_round': None,

        # ========== Defense Configuration (V1: fedavg | hmp_gae) ==========
        # defense_method selects the server-side aggregation rule.
        #   'fedavg'  — standard data-size-weighted FedAvg (no-defense baseline)
        #   'hmp_gae' — HMP-GAE immunization (this paper, requires hmp_gae/ subpackage)
        # Current value is 'hmp_gae': the proposed defense, paired with the
        # same Hallucination attack / Yahoo Answers data as the FedAvg
        # no-defense baseline (only this field differs — controlled variable).
        'defense_method': 'hmp_gae',
        'defense_config': {
            # --- Node features (eta_i) ---
            'proj_dim': 64,              # random-projection dim for flat update
            'eta_dim': 64,               # output dim of f_enc MLP
            'random_proj_seed': 42,      # shared across rounds
            # --- Hypergraph (H) ---
            'knn_k': 2,                  # k-NN neighbors; hyperedge size = k+1.
                                         # k=2 for N=7: larger k forces benign nodes to include
                                         # attackers in their hyperedges, diluting the isolation
                                         # signal. k=2 keeps the 2-attacker sub-cluster tighter
                                         # and the graph_residual contrast sharper.
            # --- HMP encoder / decoder ---
            'hidden_dim': 64,
            'latent_dim': 32,
            'num_hmp_layers': 2,         # L
            # --- Self-supervised training (per round) ---
            'train_steps_per_round': 5,
            'train_lr': 1e-3,
            'lambda_H': 1.0,             # BCE(H, H_hat) weight
            'lambda_A': 1.0,             # smoothness: sum A_hat_ij ||z_i - z_j||^2
            'lambda_hist': 0.5,          # ||z_i - z_hist_i||^2 weight
            'weight_decay': 1e-5,
            # --- Trust scoring ---
            # Primary signal: graph-structural residual from hypergraph H
            # (robust at cold start; attackers form tight sub-cluster with
            # low hyperedge reach into the benign majority).
            'graph_weight': 1.0,
            # Secondary signal: learned A_hat residual (kicks in as encoder trains).
            'residual_weight_alpha': 0.3,
            # Tertiary signal: per-sample semantic divergence on a fixed probe
            # subset (Signal 3 in trust_scorer). Off (=0.0) reproduces the
            # original geometry-only HMP-GAE; >0 enables the output-behavior
            # signal that catches geometrically-stealthy hallucination attackers.
            # When >0, the server forwards each client's softmax over a fixed
            # probe set into the runtime; otherwise no probe forward is done.
            # Raised from 1.0 -> 2.0 for the non-IID Yahoo Answers run: graph
            # signal is known to degrade in non-IID, so we give more weight to
            # the output-behavior signal (orthogonal to update geometry).
            # Effective signal share: sem rises 44% -> 61%; graph drops 44% -> 30%.
            'semantic_weight': 1.0,
            # Historical deviation disabled by default: benign clients learning
            # real features drift more than attackers stuck on a fixed mislabel
            # manifold, which can invert the signal. Re-enable with care.
            'hist_weight_beta': 0.3,
            # Round-dependent phase gating for hist signal.
            # None  = always on (backward compatible; matches Y2/Y5 behavior).
            # int N = enable hist for round_num < N (0-indexed), then beta_eff=0.
            #
            # Y5 (β=0.3, no gating) showed hist_dev signal direction is correct
            # in R1-R11 (Phase 1, warmup, 100% atk>bgn) but inverts in R26+
            # (Phase 3, steady state, only 28% atk>bgn). Setting
            # hist_warmup_rounds=10 should capture the good Phase 1 signal
            # without the Phase 3 inversion penalty.
            'hist_warmup_rounds': None,
            'softmax_tau': 0.1,          # only used when trust_mode='softmax'
            # gate_signal: which suspicion signal feeds the rejection gate.
            #   'graph'    -> graph_residual_z only (ignores recon/sem/hist).
            #   'combined' -> z-score(-trust.s); folds in ALL enabled signals.
            # Set explicitly so hist_weight_beta is guaranteed to drive the gate.
            # (runtime.py only auto-promotes to 'combined' when semantic_weight>0,
            #  so a future ablation with semantic_weight=0 + hist_weight_beta>0
            #  would silently fall back to graph-only without this line.)
            'gate_signal': 'combined',
            # Trust-to-weight mapping:
            #   'soft_reject_fedavg' (default): sigmoid gate on the gate_signal,
            #       then data-size FedAvg among continuously-trusted clients.
            #       Robust to threshold miscalibration; works for any N.
            #   'reject_then_fedavg': hard binary rejection (gate_signal > threshold),
            #       then FedAvg.  Calibrated for 8B/2A; fragile on other configs.
            #   'softmax': pure softmax of trust logits (concentrates on 1-2 clients).
            'trust_mode': 'soft_reject_fedavg',
            'reject_z_threshold': 0.75,  # sigmoid midpoint (same scale as hard threshold)
            'soft_reject_k': 2.0,        # sigmoid steepness: 2=recommended, 3=near-binary
            'keep_min': 1,
            # --- Cold start ---
            # False (default): graph_residual works from round 0, no history needed.
            # True: fall back to FedAvg on round 0 when no Z_hist exists yet.
            'cold_start_fallback': False,
            # --- History (EMA) ---
            'hist_ema_beta': 0.9,
            # --- Misc ---
            'device': 'cpu',             # HMP-GAE runs on CPU (N is small)
        },

        # ========== Hallucination Evaluation (V2 M7) ==========
        # CSE: Classification Semantic Entropy, computed every round on the
        # SeqCLS softmax distribution (free -- shares the test-set forward pass
        # with accuracy/loss). Always on.
        #
        # PPL: Perplexity on a balanced test subset computed by transferring
        # the final LoRA-fine-tuned backbone into an AutoModelForCausalLM
        # (see decoder_adapters.py). Runs once at end of FL, requires
        # save_global_checkpoint=True.
        'eval_classification_semantic_entropy': True,   # per round, essentially free
        # Frequency of per-client local accuracy / CSE evaluation (Phase 5 of
        # server.run_round). Default 1 = evaluate every round (current behaviour,
        # gives the densest per-client diagnostic trace). Set to k>1 to evaluate
        # only on round 0, the final round, and every k-th round in between --
        # saves ~75% of local-eval time at k=5, at the cost of a coarser
        # per-client curve. Global Clean Acc / CSE are unaffected (they ride
        # the test-set forward done once per round on the global model).
        'eval_local_every_n_rounds': 1,
        'eval_perplexity': True,                         # end-of-FL, moderate cost
        'ppl_num_samples': 200,                          # stratified across classes
        'ppl_seed': 42,
        'ppl_max_length': None,                          # None -> reuse config['max_length']

        # ========== Global checkpoint (for downstream generation / transfer experiments) ==========
        'save_global_checkpoint': True,  # True: save server.global_model after FL under results_dir/global_checkpoint_subdir
        'global_checkpoint_subdir': 'global_checkpoint',  # Subfolder name under results/ (same run uses results_dir from setup)

        # ========== Per-round checkpoint (Colab resilience) ==========
        # After every round the FL loop atomically writes a compact snapshot
        # (global flat params + server history + defense state + RNG) to
        # results/<round_checkpoint_subdir>/checkpoint_last.pt.  On the next
        # launch run_experiment looks for it and, if the run fingerprint
        # matches, resumes from the next round — so a Colab disconnect costs
        # at most the in-flight round.  Overhead is one torch.save of a few
        # MB per round (LoRA-only), << 1% of a round's wall time.
        # Set resume_from_checkpoint=False to force a fresh run; set
        # save_round_checkpoint=False to disable the periodic save entirely.
        'save_round_checkpoint': True,
        'resume_from_checkpoint': True,
        'round_checkpoint_subdir': 'round_checkpoint',
        # ========== Task 2: optional downstream causal generation (same run as FL) ==========
        # V1 first experiment has PPL + CSE as hallucination metrics already; Task 2 generation is
        # additional explanatory output. Keep off for a faster first run (saves ~2-3 min); switch
        # to True if you want the per-probe JSONL explanations side-by-side.
        'run_downstream_after_fl': False,  # True: subprocess run_downstream_generation.py after checkpoint save
        'downstream_probes': None,  # e.g. Path to probe JSON; None skips Task 2 (no repo `data/` required)
        'downstream_output': None,  # None -> results/<experiment_name>_downstream_gen.jsonl; else path (relative to results/ if not absolute)
        'downstream_device': None,  # None -> cuda if available else cpu
        # Extra CLI tokens for run_downstream_generation.py (SeqCLS classify + CausalLM explain)
        'downstream_cli_args': [
            '--stable',
        ],

    }
    if config_overrides:
        config.update(config_overrides)

    # Run experiment (attack if num_attackers > 0 AND attack_method != 'NoAttack',
    # otherwise baseline). 'NoAttack' overrides num_attackers (see setup_experiment).
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
        

def run_suite(
    suite: List[Dict],
    base_overrides: Optional[Dict] = None,
) -> None:
    """
    Run a list of experiments sequentially, each as a separate main() call.

    Args:
        suite:          List of per-experiment override dicts.  Each dict is
                        merged on top of base_overrides (and on top of main()'s
                        default config via the existing config_overrides path).
                        An empty dict {} means "use base_overrides as-is".
        base_overrides: Shared overrides applied to every experiment before the
                        per-experiment dict.  Useful for Colab-wide settings
                        (e.g. dataset_size_limit) that every run should share.

    Example (Colab notebook):
        run_suite(
            suite=[
                # No-attack ceiling: 7 benign clients, defense is a no-op (use fedavg).
                # This is the "perfect-world" upper bound HMP-GAE is compared against.
                {'experiment_name': 'noattack_n7_r50_qwen',
                 'num_attackers': 0, 'attack_method': 'NoAttack',
                 'defense_method': 'fedavg'},
                # Under attack: FedAvg (no defense) -- shows attack damage.
                {'experiment_name': 'fedavg_hallu_n7_r50_qwen',
                 'defense_method': 'fedavg'},
                # Under attack: HMP-GAE (full, with semantic signal).
                {'experiment_name': 'hmpgae_hallu_n7_r50_qwen',
                 'defense_method': 'hmp_gae'},
            ],
            base_overrides=COLAB_CONFIG_OVERRIDES,  # shared knobs, e.g. num_rounds=5 for a quick test
        )
    """
    n = len(suite)
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT SUITE: {n} run(s) queued")
    print(f"{'=' * 60}\n")

    for idx, exp_overrides in enumerate(suite):
        combined: Dict = {}
        if base_overrides:
            combined.update(base_overrides)
        combined.update(exp_overrides)

        exp_name = combined.get('experiment_name', f'run_{idx + 1}')
        print(f"\n{'=' * 60}")
        print(f"RUN {idx + 1}/{n}: {exp_name}")
        print(f"{'=' * 60}")

        try:
            main(config_overrides=combined if combined else None)
            print(f"\nRUN {idx + 1}/{n} DONE: {exp_name}")
        except KeyboardInterrupt:
            print(f"\nSuite interrupted after run {idx + 1}/{n}.")
            raise
        except Exception as e:
            import traceback
            print(f"\nRUN {idx + 1}/{n} FAILED: {exp_name}")
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  Continuing to next run...\n")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"SUITE COMPLETE: {n} run(s) finished")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()