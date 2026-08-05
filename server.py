# server.py
# This module implements the Server class for federated learning, including model aggregation.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import copy
from client import BenignClient
import torch.nn.functional as F
from defense import Defense, FedAvgDefense, build_defense


class Server:
    """Server class for federated learning with model aggregation"""
    def __init__(self, global_model: nn.Module, test_loader,
                total_rounds=20, server_lr=1.0,
                similarity_mode='pairwise',
                defense_method: str = 'fedavg',
                defense_config: Optional[Dict[str, Any]] = None,
                num_clients: Optional[int] = None,
                compute_classification_semantic_entropy: bool = True,
                semantic_probe_size: int = 64,
                semantic_probe_seed: int = 42,
                eval_local_every_n_rounds: int = 1):
        self.global_model = copy.deepcopy(global_model)
        self.test_loader = test_loader
        self.total_rounds = total_rounds
        # CRITICAL: Use explicit cuda:0 instead of 'cuda' to ensure device consistency
        # This prevents issues where 'cuda' and 'cuda:0' are treated as different devices
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.global_model.to(self.device)
        # Shared GPU-resident model used to evaluate each client's local
        # metrics and probe distribution. We swap the client's flat trainable
        # params into this model (cheap: a few MB for LoRA) instead of moving
        # the entire ~2GB Qwen base back and forth between CPU and GPU per
        # client per round. Frozen base weights here are bit-identical to
        # those held by every client (same HF load, same seed, same arch).
        self._eval_model = copy.deepcopy(global_model)
        self._eval_model.to(self.device)
        self.clients = []
        self.client_dict = {}  # client_id -> client mapping for O(1) lookup
        self.log_data = []

        # Frequency of per-client local accuracy / CSE evaluation. Default 1
        # (every round, current behavior). Set >1 to evaluate only on round 0,
        # the final round, and every n-th round in between -- a sparser
        # diagnostic trace in exchange for ~75% saving on local-eval forwards
        # (LoRA mode: ~10% of total round wall-clock).
        self.eval_local_every_n_rounds = max(1, int(eval_local_every_n_rounds))

        # Server parameters
        self.server_lr = server_lr  # Server learning rate
        # Similarity mode (diagnostics only, consumed by visualization):
        # 'local_vs_global' | 'pairwise' | 'both'
        self.similarity_mode = str(similarity_mode).lower() if similarity_mode else 'pairwise'
        if self.similarity_mode not in ('local_vs_global', 'pairwise', 'both'):
            self.similarity_mode = 'pairwise'

        # Defense strategy (pluggable aggregation rule)
        # 'fedavg' (default, backward-compatible) or 'hmp_gae' (this paper).
        self.defense_method = (defense_method or 'fedavg').lower()
        self.defense_config = defense_config or {}
        self.defense: Defense = build_defense(
            method=self.defense_method,
            num_clients=num_clients if num_clients is not None else 0,
            defense_config=self.defense_config,
        )
        # Track the round currently being aggregated (set in run_round).
        self._current_round = 0
        self.compute_classification_semantic_entropy = bool(
            compute_classification_semantic_entropy)

        # Fixed probe subset for the per-client semantic-divergence trust
        # signal (Signal 3 in hmp_gae.trust_scorer). Built lazily on first
        # request so that defenses that don't need it pay no cost. Identical
        # across rounds. Selection policy (defense_config.semantic_probe_stratified):
        #   False (default) -> deterministic head of test_loader (historical).
        #   True            -> class-stratified deterministic sample, seeded by
        #                      semantic_probe_seed. Labels are used ONLY to
        #                      balance the sample, never in the trust scoring,
        #                      so the semantic signal stays label-free.
        self.semantic_probe_size = int(semantic_probe_size)
        self.semantic_probe_seed = int(semantic_probe_seed)
        self.semantic_probe_stratified = bool(
            (self.defense_config or {}).get('semantic_probe_stratified', False)
        )
        self._probe_batches: Optional[List[Dict[str, torch.Tensor]]] = None
        # Whether the active defense actually consumes probe distributions.
        # HMP-GAE will use them when defense_config.semantic_weight > 0.
        # NOTE (V4/V5): the CSE-reject trust modes ('v4_cse_reject',
        # 'v5_cse_reject') do NOT depend on this probe — their rejection
        # signal is the full-test local CSE (_needs_local_cse below), so a
        # semantic_weight=0 ablation cannot silently disable that signal; it
        # only drops the sem_div/probe_cse diagnostics.
        sem_w = float((self.defense_config or {}).get('semantic_weight', 0.0))
        self._needs_probe = (
            self.defense_method in ('hmp_gae', 'hmpgae', 'hmp-gae')
            and sem_w > 0.0
        )
        # Whether the active defense consumes the per-client local CSE vector
        # BEFORE aggregation (HMP-GAE V4 rejection rule). When True, run_round
        # evaluates local metrics pre-aggregation EVERY round (client models
        # are untouched by aggregation, so the values are identical to the
        # legacy post-aggregation evaluation — computed once and reused for
        # the round log, i.e. no duplicate eval cost when
        # eval_local_every_n_rounds == 1).
        trust_mode_cfg = str(
            (self.defense_config or {}).get('trust_mode', '')
        ).lower()
        self._needs_local_cse = (
            self.defense_method in ('hmp_gae', 'hmpgae', 'hmp-gae')
            and trust_mode_cfg in ('v4_cse_reject', 'v5_cse_reject')
        )

        # Track historical data
        self.history = {
            'clean_acc': [],
            'local_accuracies': {},   # {client_id: [acc_r0, acc_r1, ...]}
            'local_cse': {},          # {client_id: [cse_r0, cse_r1, ...]}
        }

    def register_client(self, client):
        """Register a client to the server."""
        self.clients.append(client)
        # Update client_id -> client mapping for O(1) lookup
        self.client_dict[client.client_id] = client

    def broadcast_model(self):
        """Broadcast the global model to all clients."""
        global_params = self.global_model.get_flat_params()
        # Clone and move to CPU to save GPU memory
        global_params_cpu = global_params.clone().cpu()
        for client in self.clients:
            # set_flat_params works on CPU models
            client.model.set_flat_params(global_params_cpu.clone())
            # Reset optimizer if model is on GPU (rarely needed now)
            if hasattr(client, '_model_on_gpu') and client._model_on_gpu:
                client.reset_optimizer()
            else:
                client.optimizer = None

    def _compute_weighted_average(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute weighted average update (FedAvg style) shared by similarity and distance calculations.
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            weighted_avg: Weighted average update tensor
            weights: List of weights used for each client
        """
        if client_ids is not None and len(client_ids) == len(updates):
            weights = []
            # Use dictionary lookup for O(1) access instead of linear search
            client_dict = getattr(self, 'client_dict', {c.client_id: c for c in self.clients})
            for cid in client_ids:
                client = client_dict.get(cid)
                if client:
                    if getattr(client, 'is_attacker', False):
                        D_i = float(getattr(client, 'claimed_data_size', 1.0))
                    else:
                        D_i = float(len(getattr(client, 'data_indices', [])) or 1.0)
                else:
                    D_i = 1.0
                weights.append(D_i)
            
            total_D = sum(weights) + 1e-12
            weighted_avg = torch.zeros_like(updates[0])
            for update, w in zip(updates, weights):
                weighted_avg += (w / total_D) * update
        else:
            weighted_avg = torch.stack(updates).mean(dim=0)
            weights = [1.0 / len(updates)] * len(updates)
        
        return weighted_avg, weights

    def _compute_similarities(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> np.ndarray:
        """
        Compute cosine similarities between each update and the weighted average update.
        
        CRITICAL: Uses weighted aggregation (FedAvg style) to match attack optimization distance definition.
        
        Definition (consistent with attack optimization):
            sim_i = cosine_similarity(Δ_i, Δ_g)
            where Δ_g = Σ_j (D_j / D_total) * Δ_j (weighted average, FedAvg style)
        
        This matches the distance definition used in _compute_distance_update_space:
            dist = ||Δ_att - Δ_g|| where Δ_g is weighted aggregate
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            numpy array of cosine similarities (one per client)
        """
        n_updates = len(updates)

        print("  📊 Computing cosine similarities (weighted aggregation, matches attack optimization)")

        # Compute weighted average (shared with distance calculation)
        weighted_avg, _ = self._compute_weighted_average(updates, client_ids)
        
        # Compute cosine similarity for all updates at once (batch computation)
        updates_stack = torch.stack(updates)  # (N, D)
        weighted_avg_expanded = weighted_avg.unsqueeze(0).expand_as(updates_stack)  # (N, D)
        similarities = torch.cosine_similarity(updates_stack, weighted_avg_expanded, dim=1).cpu().numpy()

        # Print information
        print(f"  📈 Cosine Similarity - Mean: {similarities.mean():.3f}, "
              f"Std Dev: {similarities.std():.3f}")

        # Display similarity for each client
        # Note: similarities are ordered by updates, which match client_ids order from aggregate_updates
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, sim in enumerate(similarities):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {sim:.3f}")
                else:
                    print(f"    Client {client_id}: {sim:.3f}")
            else:
                print(f"    Update {i}: {sim:.3f}")

        return similarities

    def _compute_euclidean_distances(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> np.ndarray:
        """
        Compute Euclidean distances between each update and the weighted average update.
        
        CRITICAL: Uses weighted aggregation (FedAvg style) to match attack optimization distance definition.
        
        Definition (consistent with attack optimization):
            dist_i = ||Δ_i - Δ_g||
            where Δ_g = Σ_j (D_j / D_total) * Δ_j (weighted average, FedAvg style)
        
        This matches the distance definition used in _compute_distance_update_space:
            dist = ||Δ_att - Δ_g|| where Δ_g is weighted aggregate
        
        Args:
            updates: List of client update tensors
            client_ids: List of client IDs (optional, for weighted aggregation)
            
        Returns:
            numpy array of Euclidean distances (one per client)
        """
        n_updates = len(updates)
        
        print("  📊 Computing Euclidean distances (weighted aggregation, matches attack optimization)")
        
        # Compute weighted average (shared with similarity calculation)
        weighted_avg, _ = self._compute_weighted_average(updates, client_ids)
        
        # Compute Euclidean distance for all updates at once (batch computation)
        updates_stack = torch.stack(updates)  # (N, D)
        weighted_avg_expanded = weighted_avg.unsqueeze(0).expand_as(updates_stack)  # (N, D)
        diff = updates_stack - weighted_avg_expanded  # (N, D)
        distances = torch.norm(diff, dim=1).cpu().numpy()
        
        # Print information
        print(f"  📈 Euclidean Distance - Mean: {distances.mean():.6f}, "
              f"Std Dev: {distances.std():.6f}")
        
        # Display distance for each client
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, dist in enumerate(distances):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {dist:.6f}")
                else:
                    print(f"    Client {client_id}: {dist:.6f}")
            else:
                print(f"    Update {i}: {dist:.6f}")
        
        return distances

    def _compute_similarities_pairwise(self, updates: List[torch.Tensor], client_ids: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise cosine similarities between all client updates (no self, no global).
        S[i,j] = cosine_similarity(Δ_i, Δ_j). Per-client metric: mean similarity to other clients (exclude self).
        
        Returns:
            similarity_matrix: (N, N) numpy array
            similarities_derived: (N,) per-client mean similarity to others (same order as client_ids)
        """
        n = len(updates)
        print("  📊 Computing cosine similarities (pairwise: local vs local, no self)")
        if n == 0:
            return np.array([]).reshape(0, 0), np.array([])
        updates_stack = torch.stack(updates)  # (N, D)
        normalized = F.normalize(updates_stack.float(), p=2, dim=1)  # (N, D)
        similarity_matrix = (normalized @ normalized.T).cpu().numpy()  # (N, N), diagonal = 1
        # Per-client: mean over j != i (exclude self)
        similarities_derived = np.zeros(n)
        if n == 1:
            similarities_derived[0] = 1.0
        else:
            for i in range(n):
                others = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
                similarities_derived[i] = float(np.mean(others))
        print(f"  📈 Cosine Similarity (pairwise mean) - Mean: {similarities_derived.mean():.3f}, Std Dev: {similarities_derived.std():.3f}")
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, sim in enumerate(similarities_derived):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {sim:.3f}")
                else:
                    print(f"    Client {client_id}: {sim:.3f}")
            else:
                print(f"    Update {i}: {sim:.3f}")
        return similarity_matrix, similarities_derived

    def _compute_raw_weights(self, client_ids: List[int]) -> List[float]:
        """
        Data-size-based weights used by FedAvg (and as the default prior for
        defenses that do not override them).
        """
        weights: List[float] = []
        for cid in client_ids:
            client = self.clients[cid]
            if getattr(client, 'is_attacker', False):
                w = float(getattr(client, 'claimed_data_size', 1.0))
            else:
                w = float(len(getattr(client, 'data_indices', [])) or 1.0)
            weights.append(w)
        return weights

    def aggregate_updates(self, updates: List[torch.Tensor],
                          client_ids: List[int],
                          probe_distributions: Optional[torch.Tensor] = None,
                          local_cse: Optional[List[float]] = None) -> Dict:
        # Store client_ids for similarity display
        self._current_client_ids = client_ids
        self._sorted_client_ids = client_ids

        # Raw aggregation weights (data-size-based), passed to the defense as a prior.
        raw_weights = self._compute_raw_weights(client_ids)

        # Delegate to the configured defense strategy.
        # local_cse (per-client full-test CSE, aligned with client_ids) is only
        # computed and only forwarded for the HMP-GAE V4 rule — baseline
        # defenses keep their narrower signature and never see the kwarg.
        defense_kwargs = {}
        if local_cse is not None:
            defense_kwargs['local_cse'] = local_cse
        aggregated_update, defense_stats = self.defense.aggregate(
            updates=updates,
            client_ids=client_ids,
            data_sizes=raw_weights,
            round_num=self._current_round,
            device=self.device,
            probe_distributions=probe_distributions,
            **defense_kwargs,
        )
        # Ensure aggregated update is on the server device with consistent dtype.
        aggregated_update = aggregated_update.to(device=self.device, dtype=updates[0].dtype)
        aggregated_update_norm = torch.norm(aggregated_update).item()

        # Update global model (standard FedAvg update rule: w_{t+1} = w_t + eta * Delta).
        current_params = self.global_model.get_flat_params()
        new_params = current_params + self.server_lr * aggregated_update
        self.global_model.set_flat_params(new_params)

        defense_label = defense_stats.get('defense_name', self.defense_method)
        print(f"  📊 Defense [{defense_label}]: Aggregated {len(updates)}/{len(updates)} updates")
        print(f"  🔧 Server Learning Rate: {self.server_lr}")
        print(f"  📐 Aggregated update norm: {aggregated_update_norm:.6f}")
        alpha_list = defense_stats.get('alpha')
        if isinstance(alpha_list, list) and len(alpha_list) == len(client_ids):
            alpha_summary = ", ".join(
                f"c{cid}={a:.3f}" for cid, a in zip(client_ids, alpha_list)
            )
            print(f"  ⚖️  Trust weights: {alpha_summary}")

        # Per-client historical-deviation signal: ||z_i - z_hist_i||_2.
        # Logged every round regardless of hist_weight_beta so we can study
        # signal direction (attacker high vs benign low) before deciding
        # whether to give it nonzero weight in the trust score.
        hist_dev_list = defense_stats.get('hist_dev')
        if isinstance(hist_dev_list, list) and len(hist_dev_list) == len(client_ids):
            hist_dev_summary = ", ".join(
                f"c{cid}={h:.4f}" for cid, h in zip(client_ids, hist_dev_list)
            )
            print(f"  🕰️  hist_dev:      {hist_dev_summary}")

        # Phase-gating diagnostics (NEW 2026-05-23): show whether hist signal
        # was actually applied this round.  Helps cross-check that
        # hist_warmup_rounds is gating as expected.  Only prints when the
        # runtime exposes these fields (HMP-GAE defense; FedAvg silently skips).
        beta_cfg = defense_stats.get('hist_weight_beta_configured')
        beta_eff = defense_stats.get('hist_weight_beta_effective')
        hwr = defense_stats.get('hist_warmup_rounds')
        if beta_cfg is not None and beta_eff is not None:
            status = "ON" if beta_eff > 0 else "OFF"
            hwr_str = "None" if hwr is None else str(hwr)
            print(
                f"  ⏱️  hist gate:     β_cfg={beta_cfg:.2f}, "
                f"β_eff={beta_eff:.2f}, warmup_rounds={hwr_str}, status={status}"
            )

        # Combined-gate diagnostics (NEW 2026-05-23, Issue 1): the suspicion
        # z-score that actually drives the sigmoid gate, and the resulting gate.
        # High sus_z = suspicious (gate -> 0); compare attacker vs benign to see
        # whether the trust mechanism points the right direction.
        sus_z_list = defense_stats.get('sus_z')
        gate_list = defense_stats.get('gate')
        if isinstance(sus_z_list, list) and len(sus_z_list) == len(client_ids):
            sus_z_summary = ", ".join(
                f"c{cid}={v:.3f}" for cid, v in zip(client_ids, sus_z_list)
            )
            print(f"  🎯 sus_z:        {sus_z_summary}")
        if isinstance(gate_list, list) and len(gate_list) == len(client_ids):
            gate_summary = ", ".join(
                f"c{cid}={v:.3f}" for cid, v in zip(client_ids, gate_list)
            )
            print(f"  🚪 gate:         {gate_summary}")

        # V4/V5 rejection-rule diagnostics (trust_mode 'v4_cse_reject' or
        # 'v5_cse_reject'): per-client CSE/median ratio and which clients were
        # flagged this round. V5 has no scalar mult — show floor + r_hard.
        v4_ratio_list = defense_stats.get('v4_ratio')
        v4_flagged_list = defense_stats.get('v4_flagged')
        if isinstance(v4_ratio_list, list) and len(v4_ratio_list) == len(client_ids):
            ratio_summary = ", ".join(
                f"c{cid}={v:.3f}" for cid, v in zip(client_ids, v4_ratio_list)
            )
            print(f"  🧪 cse/med:      {ratio_summary}")
        if isinstance(v4_flagged_list, list) and len(v4_flagged_list) == len(client_ids):
            flagged_ids = [cid for cid, f in zip(client_ids, v4_flagged_list) if f]
            if 'v5_m_floor' in defense_stats:
                mult_desc = (
                    f"ramp floor={defense_stats.get('v5_m_floor')}, "
                    f"r_hard={defense_stats.get('v5_r_hard')}"
                )
            else:
                mult_desc = f"mult={defense_stats.get('v4_reject_mult')}"
            print(
                f"  ⛔ flagged:      {flagged_ids if flagged_ids else 'none'} "
                f"(tau={defense_stats.get('v4_tau_ratio')}, "
                f"k_cap={defense_stats.get('v4_k_cap')}, "
                f"{mult_desc})"
            )

        # Compute similarity and distance metrics for visualization (unchanged).
        mode = getattr(self, 'similarity_mode', 'local_vs_global')
        if mode == 'local_vs_global':
            similarities = self._compute_similarities(updates, client_ids)
            similarity_matrix = None
            similarities_vs_global = None
        elif mode == 'pairwise':
            similarity_matrix, similarities = self._compute_similarities_pairwise(updates, client_ids)
            similarities_vs_global = None
        else:  # 'both'
            similarities_vs_global = self._compute_similarities(updates, client_ids)
            similarity_matrix, similarities = self._compute_similarities_pairwise(updates, client_ids)
        euclidean_distances = self._compute_euclidean_distances(updates, client_ids) if len(updates) > 0 else np.array([])

        aggregation_log = {
            'similarities': similarities.tolist(),
            'euclidean_distances': euclidean_distances.tolist() if len(euclidean_distances) > 0 else [],
            'accepted_clients': client_ids.copy(),
            'mean_similarity': float(similarities.mean()) if len(similarities) > 0 else 1.0,
            'std_similarity': float(similarities.std()) if len(similarities) > 0 else 0.0,
            'mean_euclidean_distance': euclidean_distances.mean().item() if len(euclidean_distances) > 0 else 0.0,
            'std_euclidean_distance': euclidean_distances.std().item() if len(euclidean_distances) > 0 else 0.0,
            'aggregated_update_norm': aggregated_update_norm,
            'defense_method': defense_label,
            'trust_weights': alpha_list if isinstance(alpha_list, list) else None,
            'raw_weights': raw_weights,
        }
        # Persist extra defense stats (skip bulky numpy blobs like 'Z' from the
        # main JSON log to keep result files lean; HMP runtime writes its own
        # stats file if enabled).
        for k in ('residual', 'recon_residual', 'sem_div',
                  'graph_residual_z', 'recon_residual_z', 'sem_div_z', 'hist_dev_z',
                  'hist_dev', 's', 'sus_z', 'sus_raw', 'gate',
                  'graph_weight', 'residual_weight_alpha',
                  'semantic_weight', 'hist_weight_beta_effective',
                  'zscore_mode', 'gate_rezscore', 'sus_ema_beta',
                  'semantic_reference',
                  'trust_mode_used', 'graph_channel_gated', 'graph_min_distinct',
                  'probe_cse',
                  'v4_cse', 'v4_ratio', 'v4_flagged', 'v4_multiplier',
                  'v4_median_cse', 'v4_tau_ratio', 'v4_k_cap', 'v4_reject_mult',
                  'v5_m_floor', 'v5_r_hard', 'v5_ramp_t',
                  'L_rec', 'L_smooth', 'L_hist',
                  'fallback_reason', 'defense_time_ms'):
            if k in defense_stats:
                aggregation_log[k] = defense_stats[k]
        if similarity_matrix is not None:
            aggregation_log['similarity_matrix'] = similarity_matrix.tolist()
        if similarities_vs_global is not None:
            aggregation_log['similarities_vs_global'] = similarities_vs_global.tolist()
        aggregation_log['similarity_mode'] = mode

        return aggregation_log

    def evaluate_local_metrics(self, client) -> Tuple[float, float]:
        """
        Evaluate a client's local model on the server test set in a single forward pass.

        Returns (accuracy, classification_semantic_entropy).

        In real FL the server never sees client.model directly — it reconstructs
        the local model as w_global + Δ_i.  In this simulation the two are
        equivalent because client.model == w_global + Δ_i after local_train().
        Using the server's public test set is inherent to FedLLMs evaluation.

        Implementation: instead of moving client.model (full ~2GB Qwen) between
        CPU and GPU, we copy only the trainable flat params into the shared
        GPU-resident self._eval_model. In LoRA mode this is a few-MB tensor
        copy; in Full-FT mode it's equivalent to the old .to() call. The
        client's own model object is untouched.
        """
        # client.model lives on CPU between rounds; get_flat_params returns
        # a CPU tensor of just the trainable surface (LoRA-only with use_lora=True).
        flat = client.model.get_flat_params()
        self._eval_model.set_flat_params(flat)
        self._eval_model.eval()

        correct = 0
        total = 0
        total_cse = 0.0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self._eval_model(input_ids, attention_mask)

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                log_probs = F.log_softmax(outputs, dim=1)
                probs = log_probs.exp()
                batch_cse = -(probs * log_probs).sum(dim=1)
                total_cse += batch_cse.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        cse = total_cse / total if total > 0 else 0.0
        return accuracy, cse

    def evaluate_local_accuracy(self, client) -> float:
        """Backward-compatible wrapper; prefer evaluate_local_metrics."""
        acc, _ = self.evaluate_local_metrics(client)
        return acc

    def _collect_local_metrics(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Evaluate every client's (local accuracy, local CSE) on the server test
        set and append to history. Shared by the legacy post-aggregation eval
        and the V4 pre-aggregation eval in run_round — client models are not
        modified by aggregation (only self.global_model is), so the values are
        identical regardless of when in the round this runs.
        """
        local_accs: Dict[int, float] = {}
        local_cses: Dict[int, float] = {}
        for client in self.clients:
            try:
                local_acc, local_cse = self.evaluate_local_metrics(client)
                local_accs[client.client_id] = local_acc
                local_cses[client.client_id] = local_cse

                if client.client_id not in self.history['local_accuracies']:
                    self.history['local_accuracies'][client.client_id] = []
                self.history['local_accuracies'][client.client_id].append(local_acc)

                if client.client_id not in self.history['local_cse']:
                    self.history['local_cse'][client.client_id] = []
                self.history['local_cse'][client.client_id].append(local_cse)
            except Exception as e:
                print(f"  ⚠️  Could not evaluate local metrics for client {client.client_id}: {e}")
        return local_accs, local_cses

    def _ensure_probe_batches(self) -> List[Dict[str, torch.Tensor]]:
        """Lazily snapshot a fixed subset of test_loader for probing clients."""
        if self._probe_batches is not None:
            return self._probe_batches
        target = max(1, self.semantic_probe_size)
        batches: List[Dict[str, torch.Tensor]] = []
        if self.semantic_probe_stratified:
            batches = self._build_stratified_probe_batches(target)
        if not batches:
            # Historical behavior: deterministic head of test_loader. Also the
            # fallback when the dataset does not expose labels for stratification.
            collected = 0
            for batch in self.test_loader:
                # Snapshot tensors on CPU to keep peak GPU memory bounded.
                snapshot = {
                    'input_ids': batch['input_ids'].detach().cpu(),
                    'attention_mask': batch['attention_mask'].detach().cpu(),
                }
                batches.append(snapshot)
                collected += int(snapshot['input_ids'].shape[0])
                if collected >= target:
                    break
        self._probe_batches = batches
        return batches

    def _build_stratified_probe_batches(
        self, target: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Class-stratified deterministic probe selection.

        Draws ~target/num_classes samples per class from test_loader.dataset
        (seeded, without replacement) so the probe set is class-balanced --
        a head-of-loader snapshot can be class-skewed, which biases the
        semantic-divergence signal. Labels are consumed only here, for
        balancing; the snapshot itself carries no labels.

        Returns [] when the dataset does not expose a `labels` attribute,
        in which case the caller falls back to the head-of-loader path.
        """
        dataset = getattr(self.test_loader, 'dataset', None)
        labels = getattr(dataset, 'labels', None)
        if dataset is None or labels is None or len(labels) == 0:
            print("  [Server] semantic_probe_stratified=True but test dataset "
                  "exposes no labels; falling back to head-of-loader probes.")
            return []
        by_class: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            by_class.setdefault(int(lab), []).append(idx)
        classes = sorted(by_class)
        rng = np.random.default_rng(self.semantic_probe_seed)
        quota, rem = divmod(target, len(classes))
        chosen: List[int] = []
        for j, c in enumerate(classes):
            k = min(quota + (1 if j < rem else 0), len(by_class[c]))
            if k <= 0:
                continue
            sel = rng.choice(len(by_class[c]), size=k, replace=False)
            chosen.extend(by_class[c][int(i)] for i in sel)
        if not chosen:
            return []
        chosen.sort()  # deterministic order, independent of class iteration
        bs = int(self.test_loader.batch_size or 32)
        batches: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(chosen), bs):
            items = [dataset[i] for i in chosen[start:start + bs]]
            batches.append({
                'input_ids': torch.stack([it['input_ids'] for it in items]).cpu(),
                'attention_mask': torch.stack(
                    [it['attention_mask'] for it in items]).cpu(),
            })
        per_class = {c: 0 for c in classes}
        for i in chosen:
            per_class[int(labels[i])] += 1
        print(f"  [Server] stratified semantic probe: {len(chosen)} samples, "
              f"per-class counts={per_class} (seed={self.semantic_probe_seed})")
        return batches

    def evaluate_local_probe_distribution(self, client, flat_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward a client's local model over a fixed probe subset and return the
        per-sample softmax probabilities (the behavioral-fingerprint / semantic
        trust signal).

        Args:
            client: the client whose contribution is being probed.
            flat_params: optional explicit trainable-flat params to evaluate. When
                provided, the probe forwards ``w_global + Delta_i`` (the update the
                client actually SUBMITTED) instead of ``client.model``. This is the
                principled definition of the signal -- it measures the behavior of
                each client's contribution. For honest / label-flip attackers the
                two are numerically identical (client.model == w_global + Delta_i
                after local_train, so results are unchanged to float precision).
                For update-forging attackers that do NOT locally train (e.g. AugMP,
                whose client.model stays == w_global while the malice lives only in
                the crafted Delta), ``client.model`` would look perfectly benign;
                passing w_global + Delta_i lets the semantic signal actually see the
                submitted attack. When None, falls back to ``client.model`` (legacy).

        Returns:
            (K, C) tensor on CPU, where K = number of probe samples actually
            taken (<= semantic_probe_size, capped by len(test_loader.dataset))
            and C = num_labels.

        Uses the shared GPU-resident self._eval_model (see evaluate_local_metrics).
        """
        batches = self._ensure_probe_batches()
        flat = client.model.get_flat_params() if flat_params is None else flat_params
        self._eval_model.set_flat_params(flat)
        self._eval_model.eval()
        rows: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in batches:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self._eval_model(input_ids, attention_mask)
                rows.append(F.softmax(logits, dim=-1).detach().cpu())
        return torch.cat(rows, dim=0)
    
    def evaluate(self) -> float:
        """
        Evaluate the global model's performance.

        Returns:
            Clean accuracy (float) on the test set
        """
        accuracy, _, _ = self.evaluate_with_loss()
        return accuracy

    def evaluate_with_loss(self) -> Tuple[float, float, Optional[float]]:
        """
        Evaluate the global model's performance in a single pass and also
        compute the Classification Semantic Entropy (CSE) on the SeqCLS head.

        Returns:
            Tuple of (clean_accuracy, global_loss, classification_semantic_entropy_or_none).
            The third value is ``None`` when ``compute_classification_semantic_entropy`` is False.

        The CSE is the mean Shannon entropy of the softmax class distribution
        p(y|x) over the test set. Lower = more confident predictions; under a
        hallucination-inducing attack the model becomes less confident and CSE
        increases. A principled no-generation surrogate for Farquhar-style
        semantic entropy, using the C class labels as the "semantic clusters".
        """
        self.global_model.eval()

        # Evaluate clean accuracy, loss and CSE in one forward pass.
        correct = 0
        total = 0
        total_loss = 0.0
        total_cse = 0.0
        do_cse = self.compute_classification_semantic_entropy

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.global_model(input_ids, attention_mask)

                # Accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Cross-entropy loss (sum over batch for later averaging).
                loss = F.cross_entropy(outputs, labels, reduction='sum')
                total_loss += loss.item()

                if do_cse:
                    # Classification Semantic Entropy (per-sample Shannon entropy,
                    # summed here and averaged at the end).
                    # Use log_softmax for numerical stability.
                    log_probs = F.log_softmax(outputs, dim=1)
                    probs = log_probs.exp()
                    batch_cse = -(probs * log_probs).sum(dim=1)  # (B,)
                    total_cse += batch_cse.sum().item()

        clean_accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        mean_cse: Optional[float]
        if do_cse:
            mean_cse = total_cse / total if total > 0 else 0.0
        else:
            mean_cse = None

        # Record historical metrics.
        self.history['clean_acc'].append(clean_accuracy)
        if 'cse' not in self.history:
            self.history['cse'] = []
        self.history['cse'].append(mean_cse)

        return clean_accuracy, avg_loss, mean_cse
    
    def evaluate_global_loss(self) -> float:
        """
        Evaluate the global model's loss on the test set.
        For efficiency, use evaluate_with_loss() if you also need accuracy.
        
        Returns:
            Global loss (float) on the test set (cross-entropy loss)
        """
        _, loss, _ = self.evaluate_with_loss()
        return loss

    def adaptive_adjustment(self, round_num: int):
        """Adaptively adjust parameters based on historical performance."""
        # Fixed server_lr (no adaptive change)
        pass

    def run_round(self, round_num: int) -> Dict:
        """Execute one round of federated learning - stable version."""
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # Track the current round so the defense plugin can use it for history.
        self._current_round = int(round_num)

        # Adaptive adjustment
        self.adaptive_adjustment(round_num)

        # Display current parameters
        print(f"Current Parameters: server_lr={self.server_lr:.2f}")
        print(f"{'=' * 60}")

        # Broadcast the model
        print("📡 Broadcasting the global model...")
        self.broadcast_model()

        # Phase 0.5: Constrained-attacker setup (only for update-forging attackers
        # that advertise crafts_update, i.e. AugMP). Hands each such attacker the
        # current global params + constraint bounds so it can build the global-loss
        # proxy F(w'_g) and the distance/cosine-similarity constraints inside
        # camouflage_update. This whole block is SKIPPED for every existing run
        # (benign / hallucination / alie / gaussian / sign_flipping), so their code
        # path is byte-for-byte unchanged. dist_bound / sim_bound_* are read via
        # getattr (default None = auto-derive from benign statistics); main.py sets
        # them on the server only for attack_method='AugMP'.
        needs_constraint_setup = any(getattr(c, 'crafts_update', False) for c in self.clients)
        if needs_constraint_setup:
            global_params = self.global_model.get_flat_params()
            total_data_size = 0.0
            benign_data_sizes: Dict[int, float] = {}
            for client in self.clients:
                if getattr(client, 'is_attacker', False):
                    total_data_size += float(getattr(client, 'claimed_data_size', 1.0))
                else:
                    client_data_size = len(getattr(client, 'data_indices', [])) or 1.0
                    benign_data_sizes[client.client_id] = client_data_size
                    total_data_size += client_data_size
            for client in self.clients:
                if getattr(client, 'is_attacker', False):
                    client.set_global_model_params(global_params)
                    client.set_constraint_params(
                        dist_bound=getattr(self, 'dist_bound', None),
                        sim_bound_low=getattr(self, 'sim_bound_low', None),
                        sim_bound_up=getattr(self, 'sim_bound_up', None),
                        total_data_size=total_data_size,
                        benign_data_sizes=benign_data_sizes,
                    )

        # Phase 1: Preparation
        print("\n🔧 Phase 1: Client Preparation")
        for client in self.clients:
            client.set_round(round_num)
            # Use is_attacker attribute instead of isinstance to support both GRMP and ALIE attackers
            if getattr(client, 'is_attacker', False):
                client.prepare_for_round(round_num)

        # Phase 2: Local Training
        print("\n💪 Phase 2: Local Training")
        initial_updates = {}
        for client in self.clients:
            update = client.local_train()
            initial_updates[client.client_id] = update
            print(f"  ✓ Client {client.client_id} completed training")

        # Phase 3: Attacker Camouflage
        print("\n🎭 Phase 3: Attacker Camouflage")
        benign_updates = []
        benign_client_ids = []
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if not getattr(client, 'is_attacker', False):
                benign_updates.append(update)
                benign_client_ids.append(client_id)
        
        print(f"  Captured {len(benign_updates)} benign updates for camouflage.")
        
        # ===== NEW: Store completed attacker updates for coordinated optimization =====
        completed_attacker_updates = {}  # {client_id: update_tensor}
        completed_attacker_client_ids = []  # Keep order
        completed_attacker_data_sizes = {}  # {client_id: claimed_data_size}
        # ==============================================================================
        
        final_updates = {}
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if getattr(client, 'is_attacker', False):
                print(f"  ⚠️ Triggering camouflage logic for Client {client_id}")
                client.receive_benign_updates(benign_updates, client_ids=benign_client_ids)
                
                # ===== NEW: Pass completed attacker updates to current attacker =====
                if completed_attacker_updates:
                    client.receive_attacker_updates(
                        updates=list(completed_attacker_updates.values()),
                        client_ids=completed_attacker_client_ids,
                        data_sizes=completed_attacker_data_sizes
                    )
                # ====================================================================
                
                final_updates[client_id] = client.camouflage_update(update)
                
                # ===== NEW: Store current attacker's update for subsequent attackers =====
                completed_attacker_updates[client_id] = final_updates[client_id]
                completed_attacker_client_ids.append(client_id)
                completed_attacker_data_sizes[client_id] = float(getattr(client, 'claimed_data_size', 1.0))
                # =========================================================================
            else:
                final_updates[client_id] = update

        # Phase 4: Aggregation
        print("\n📊 Phase 4: Model Aggregation")
        # Ensure deterministic order of keys
        sorted_client_ids = sorted(final_updates.keys())
        final_update_list = [final_updates[cid] for cid in sorted_client_ids]

        # Optional Phase 3.5: per-client probe forward for the semantic-divergence
        # trust signal. Only computed when the active defense actually consumes it.
        probe_tensor: Optional[torch.Tensor] = None
        if self._needs_probe:
            # Semantic signal. For every ordinary client (benign, and attackers that
            # bake their poison into local training such as Hallucination) we probe
            # client.model EXACTLY as before -- byte-for-byte identical to pre-AugMP
            # behaviour. Only for update-forging attackers (crafts_update, i.e. AugMP,
            # whose client.model stays == w_global while the malice lives in the
            # crafted Delta) do we instead probe w_global + Delta_i, so the semantic
            # signal can actually see the submitted attack. self.global_model is still
            # the round-start global here (Phase 4 aggregation happens below).
            global_flat = None
            probe_rows: List[torch.Tensor] = []
            for cid in sorted_client_ids:
                client = self.client_dict.get(cid)
                if client is None:
                    raise KeyError(f"client_id {cid} not registered with server")
                if getattr(client, 'crafts_update', False):
                    if global_flat is None:
                        global_flat = self.global_model.get_flat_params()
                        if global_flat.device.type == "cuda":
                            global_flat = global_flat.cpu()
                    submitted = final_updates[cid]
                    if submitted.device.type == "cuda":
                        submitted = submitted.cpu()
                    probe_rows.append(
                        self.evaluate_local_probe_distribution(
                            client, flat_params=global_flat + submitted
                        )
                    )
                else:
                    probe_rows.append(self.evaluate_local_probe_distribution(client))
            # All rows must have identical shape (K, C) -- same probe set, same head.
            probe_tensor = torch.stack(probe_rows, dim=0)  # (N, K, C)

        # Optional Phase 3.6: pre-aggregation per-client local metrics for the
        # CSE-reject trust rules (trust_mode 'v4_cse_reject' / 'v5_cse_reject').
        # Client models are untouched by aggregation, so these values are
        # identical to the legacy post-aggregation evaluation — computed once
        # here and reused for the round log below (no duplicate eval cost).
        local_accs_this_round: Dict[int, float] = {}
        local_cse_this_round: Dict[int, float] = {}
        local_cse_vector: Optional[List[float]] = None
        if self._needs_local_cse:
            if any(getattr(c, 'crafts_update', False) for c in self.clients):
                raise RuntimeError(
                    "CSE-reject trust modes (v4_cse_reject / v5_cse_reject) "
                    "are not supported with update-forging attackers "
                    "(crafts_update, e.g. AugMP): local CSE evaluates "
                    "client.model, which such attackers leave looking benign "
                    "(the poison lives only in the crafted update)."
                )
            local_accs_this_round, local_cse_this_round = self._collect_local_metrics()
            missing = [cid for cid in sorted_client_ids
                       if cid not in local_cse_this_round]
            if missing:
                raise RuntimeError(
                    "CSE-reject trust mode: local CSE evaluation failed "
                    f"for clients {missing}; cannot aggregate without it."
                )
            local_cse_vector = [float(local_cse_this_round[cid])
                                for cid in sorted_client_ids]

        aggregation_log = self.aggregate_updates(
            final_update_list, sorted_client_ids,
            probe_distributions=probe_tensor,
            local_cse=local_cse_vector,
        )

        # Evaluate the global model (compute accuracy, loss and CSE in one pass).
        clean_acc, global_loss, mean_cse = self.evaluate_with_loss()

        # Evaluate per-client local accuracy and CSE (single forward pass each).
        # When eval_local_every_n_rounds > 1, we only evaluate on round 0, the
        # final round, and every n-th round in between -- a sparser diagnostic
        # trace in exchange for ~75% fewer N-times-test-set forwards.
        n_eval = self.eval_local_every_n_rounds
        is_final_round = (round_num + 1) == self.total_rounds
        do_local_eval = (
            n_eval <= 1
            or round_num == 0
            or is_final_round
            or ((round_num + 1) % n_eval == 0)
        )
        if self._needs_local_cse:
            # V4 path: local metrics were already evaluated pre-aggregation
            # this round (every round — the trust rule needs them even when
            # eval_local_every_n_rounds would skip); reuse those values here.
            pass
        elif do_local_eval:
            local_accs_this_round, local_cse_this_round = self._collect_local_metrics()
        else:
            print(f"  ⏭  Skipping per-client local eval this round "
                  f"(eval_local_every_n_rounds={n_eval}).")

        # Create log for the current round
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'global_loss': global_loss,
            'classification_semantic_entropy': mean_cse,
            'acc_diff': (abs(clean_acc - self.history['clean_acc'][-2])
                         if len(self.history['clean_acc']) > 1 else 0.0),
            'aggregation': aggregation_log,
            'server_lr': self.server_lr,
            'local_accuracies': local_accs_this_round,
            'local_cse': local_cse_this_round,
        }

        self.log_data.append(round_log)

        # Display results
        print(f"\n📊 Round {round_num + 1} Results:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        if len(self.history['clean_acc']) > 1:
            prev_clean = self.history['clean_acc'][-2]
            delta_prev = clean_acc - prev_clean
            best_clean = max(self.history['clean_acc'])
            delta_best = clean_acc - best_clean
            print(f"  ΔClean vs prev: {delta_prev:+.4f}")
            print(f"  ΔClean vs best: {delta_best:+.4f}")
        print(f"  Global Loss: {global_loss:.4f}")
        if mean_cse is not None:
            print(f"  Global CSE: {mean_cse:.4f}")
        else:
            print("  Global CSE: (disabled via config)")

        # Per-client local metrics table (only when evaluated this round).
        if local_accs_this_round:
            print(f"  Per-client local metrics (local model on server test set):")
            attacker_ids = {c.client_id for c in self.clients if getattr(c, 'is_attacker', False)}
            for cid in sorted(local_accs_this_round):
                tag = "ATK" if cid in attacker_ids else "BGN"
                acc_v = local_accs_this_round[cid]
                cse_v = local_cse_this_round.get(cid, float('nan'))
                print(f"    [{tag}] Client {cid}: acc={acc_v:.4f}  cse={cse_v:.4f}")

        return round_log
