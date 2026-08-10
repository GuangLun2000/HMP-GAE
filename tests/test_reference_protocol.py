"""CPU-only invariants for server-reference isolation (no dataset download)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_loader import DataManager
from data_protocol import reference_content_sha256, reference_holdout_indices
from server import Server


class _DictDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = list(tokens)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([self.tokens[index]], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class _AlwaysClassZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        class_zero = torch.ones(batch, device=input_ids.device) + self.offset
        class_one = torch.zeros(batch, device=input_ids.device) + self.offset
        return torch.stack((class_zero, class_one), dim=1)

    def get_flat_params(self):
        return self.offset.detach().reshape(1).clone()

    def set_flat_params(self, flat):
        with torch.no_grad():
            self.offset.copy_(flat.reshape(()).to(self.offset.device))


def test_stratified_holdout_is_exact_and_reproducible():
    labels = np.repeat(np.arange(4), 250)
    first = reference_holdout_indices(labels, 400, 42, stratified=True)
    replay = reference_holdout_indices(labels, 400, 42, stratified=True)
    other_seed = reference_holdout_indices(labels, 400, 43, stratified=True)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other_seed)
    assert len(first) == len(set(first.tolist())) == 400
    assert set(np.bincount(labels[first], minlength=4).tolist()) == {100}
    assert not set(first.tolist()).intersection(
        set(np.setdiff1d(np.arange(len(labels)), first).tolist())
    )
    try:
        reference_holdout_indices(labels, len(labels), 42)
        raise AssertionError("holdout consuming the entire client pool must fail")
    except ValueError:
        pass

    digest = reference_content_sha256(["a", "b"], [0, 1])
    assert digest == reference_content_sha256(["a", "b"], [0, 1])
    assert digest != reference_content_sha256(["a", "c"], [0, 1])
    print("PASS  reference holdout is exact, balanced, disjoint and reproducible")


def test_data_manager_removes_reference_before_partitioning():
    manager = DataManager.__new__(DataManager)
    manager.server_reference_size = 4
    manager.server_reference_seed = 7
    manager.server_reference_stratified = True
    manager.train_texts = [f"train-{i}" for i in range(8)]
    manager.train_labels = [0, 0, 1, 1, 2, 2, 3, 3]
    manager.test_texts = ["official-test"]
    manager.test_labels = [0]
    manager._reserve_server_reference()

    assert len(manager.reference_texts) == 4
    assert len(manager.train_texts) == 4
    assert set(manager.reference_texts).isdisjoint(manager.train_texts)
    assert manager.test_texts == ["official-test"]
    assert manager.server_reference_metadata["per_class_counts"] == {
        "0": 1, "1": 1, "2": 1, "3": 1,
    }
    assert manager.server_reference_metadata["client_train_size"] == 4
    print("PASS  DataManager removes reference rows before client partitioning")


def test_server_routes_defense_metrics_away_from_test():
    reference = DataLoader(_DictDataset([11, 12], [0, 0]), batch_size=2)
    official_test = DataLoader(_DictDataset([91, 92], [1, 1]), batch_size=2)
    model = _AlwaysClassZero()
    server = Server(
        global_model=model,
        test_loader=official_test,
        reference_loader=reference,
        defense_method="fedavg",
        num_clients=1,
        semantic_probe_size=1,
    )
    client = SimpleNamespace(client_id=0, model=_AlwaysClassZero())

    local_accuracy, _ = server.evaluate_local_metrics(client)
    global_accuracy, _, _ = server.evaluate_with_loss()
    assert local_accuracy == 1.0, "local defense metric must use reference labels"
    assert global_accuracy == 0.0, "global reporting must use official test labels"
    probe = server._ensure_probe_batches()
    assert int(probe[0]["input_ids"][0, 0]) == 11
    print("PASS  reference drives local CSE/probes; test drives global reporting")


def test_active_defense_forbids_test_fallback():
    official_test = DataLoader(_DictDataset([91], [1]), batch_size=1)
    try:
        Server(
            global_model=_AlwaysClassZero(),
            test_loader=official_test,
            reference_loader=None,
            defense_method="hmp_gae",
            defense_config={
                "trust_mode": "v8_hmp_cse_propagation",
                "semantic_weight": 1.0,
                "num_byzantine": 0,
                "device": "cpu",
            },
            num_clients=1,
        )
        raise AssertionError("active defense must not fall back to the test loader")
    except RuntimeError as exc:
        assert "test_loader fallback is forbidden" in str(exc)
    print("PASS  active defense fails loudly without a reference loader")


if __name__ == "__main__":
    test_stratified_holdout_is_exact_and_reproducible()
    test_data_manager_removes_reference_before_partitioning()
    test_server_routes_defense_metrics_away_from_test()
    test_active_defense_forbids_test_fallback()
    print("\nAll server-reference protocol tests passed.")
