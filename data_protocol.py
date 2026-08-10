"""Pure helpers for deterministic server-reference data splits.

This module intentionally has no torch/transformers dependency so the protocol
invariants can be tested without loading a model or dataset.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def reference_holdout_indices(
    labels: Sequence[int],
    sample_size: int,
    seed: int,
    *,
    stratified: bool = True,
) -> np.ndarray:
    """Return sorted training-pool indices for a deterministic holdout.

    Stratification allocates an equal quota to each observed class and tops up
    from the unused pool when a class is too small.  ``sample_size`` must leave
    at least one example for client training; silently shrinking the requested
    server set would make matched experiments incomparable.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_items = int(labels_arr.size)
    target = int(sample_size)
    if target <= 0:
        return np.empty(0, dtype=np.int64)
    if target >= n_items:
        raise ValueError(
            "server_reference_size must be smaller than the loaded training "
            f"pool (requested {target}, available {n_items})"
        )

    rng = np.random.default_rng(int(seed))
    if not stratified:
        return np.sort(rng.choice(n_items, size=target, replace=False)).astype(
            np.int64, copy=False
        )

    classes = np.unique(labels_arr)
    if classes.size == 0:
        raise ValueError("cannot build a server reference set from empty labels")

    quota, remainder = divmod(target, int(classes.size))
    chosen: list[int] = []
    for position, class_id in enumerate(classes.tolist()):
        class_indices = np.flatnonzero(labels_arr == class_id)
        take = min(quota + (1 if position < remainder else 0), len(class_indices))
        if take:
            selected = rng.choice(class_indices, size=take, replace=False)
            chosen.extend(int(i) for i in selected)

    if len(chosen) < target:
        selected_mask = np.zeros(n_items, dtype=bool)
        selected_mask[np.asarray(chosen, dtype=np.int64)] = True
        remaining = np.flatnonzero(~selected_mask)
        extra = rng.choice(remaining, size=target - len(chosen), replace=False)
        chosen.extend(int(i) for i in extra)

    return np.asarray(sorted(chosen), dtype=np.int64)


def reference_content_sha256(texts: Sequence[str], labels: Sequence[int]) -> str:
    """Hash a reference set without persisting its raw text in result files."""
    if len(texts) != len(labels):
        raise ValueError("reference texts and labels must have equal length")
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(8, "big", signed=True))
    return digest.hexdigest()
