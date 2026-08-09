# attack/hallucination.py
# Hallucination-inducing attacker via training-time label flipping.
#
# Design principles (V1):
#   - No nested optimization loop -> runs at exactly benign training speed.
#   - Stealth constraint ||omega_a - omega'_a|| <= eps is satisfied naturally
#     because the attacker still performs standard FedProx-style local training
#     (only the labels it is trained against are flipped).
#   - Encodes "false factual associations" (e.g., World <-> Sports) directly
#     into the LoRA update, which manifests as hallucination in downstream
#     generation as discussed in our paper's threat model.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from client import BenignClient
from data_loader import NewsDataset


def deterministic_round_flip_ratio(
    flip_seed: int,
    round_num: int,
    flip_ratio_range: Tuple[float, float],
) -> float:
    """Return the historical sequential RNG draw for one round, statelessly.

    Reconstructing the generator from ``flip_seed`` and selecting draw
    ``round_num`` preserves the exact sequence produced by the old private
    ``_round_rng`` during uninterrupted runs, while making a resumed round
    independent of how many earlier ``prepare_for_round`` calls happened in
    the current Python process.
    """
    r = int(round_num)
    if r < 0:
        raise ValueError(f"round_num must be >= 0, got {round_num}")
    lo, hi = (float(flip_ratio_range[0]), float(flip_ratio_range[1]))
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(
            "flip_ratio_range must satisfy 0 <= lo <= hi <= 1; "
            f"got {flip_ratio_range!r}"
        )
    rng = np.random.default_rng(int(flip_seed))
    # FL runs are short (canonical: 50 rounds), so replaying r+1 scalar draws
    # is clearer and less version-sensitive than manipulating bit-generator
    # offsets directly.
    return float(rng.uniform(lo, hi, size=r + 1)[-1])


# --------------------------------------------------------------------------- #
# Flipped-label dataset                                                       #
# --------------------------------------------------------------------------- #

class FlippedLabelDataset(Dataset):
    """
    Dataset wrapper that overrides labels according to a flip strategy.

    Three modes are supported, matching the flip_mode config:
      - 'pairwise'  : replace label by flip_map[label] (fixed bijection)
      - 'targeted'  : replace every flipped sample with a fixed target_class
      - 'random'    : replace with a uniformly random *other* class

    Flipping is deterministic given `seed` and `flip_ratio`. By default the
    flipped label set is pre-computed once at construction time so that
    (a) labels seen during training are stable across epochs and
    (b) the corruption is reproducible between runs.

    Call ``regenerate(seed, flip_ratio)`` between rounds to draw a fresh
    random flip pattern -- this is what HallucinationAttackerClient does when
    per_round_reseed=True. ``__getitem__`` reads ``self.flipped_labels[idx]``
    dynamically, so the DataLoader picks up the new labels without rebuilding.
    """

    def __init__(
        self,
        base_dataset: NewsDataset,
        flip_ratio: float,
        flip_mode: str,
        flip_map: Optional[Dict[int, int]],
        num_labels: int,
        target_class: Optional[int] = None,
        seed: int = 0,
    ):
        if not isinstance(base_dataset, NewsDataset):
            raise TypeError(
                f"FlippedLabelDataset expected a NewsDataset, got {type(base_dataset).__name__}"
            )
        self.base = base_dataset
        self.flip_ratio = float(flip_ratio)
        self.flip_mode = str(flip_mode).lower()
        self.flip_map = {int(k): int(v) for k, v in (flip_map or {}).items()}
        self.num_labels = int(num_labels)
        self.target_class = None if target_class is None else int(target_class)
        self.seed = int(seed)

        self.flipped_labels: List[int] = self._precompute_flipped_labels()
        self.num_flipped: int = self._count_flipped()

    def _count_flipped(self) -> int:
        return sum(
            1
            for orig, new in zip(self.base.labels, self.flipped_labels)
            if int(orig) != int(new)
        )

    def _precompute_flipped_labels(self) -> List[int]:
        """Sample a fresh flipped-label list using self.seed and self.flip_ratio."""
        n = len(self.base)
        rng = np.random.default_rng(self.seed)
        flip_mask = rng.random(n) < self.flip_ratio
        out: List[int] = []
        for i in range(n):
            orig = int(self.base.labels[i])
            if flip_mask[i]:
                out.append(self._apply_flip(orig, rng))
            else:
                out.append(orig)
        return out

    def regenerate(self, seed: int, flip_ratio: Optional[float] = None) -> None:
        """
        Re-sample the flipped-label set with a new seed (and optionally a new
        flip_ratio). Mutates self.flipped_labels and self.num_flipped in place.
        Safe to call mid-training: the DataLoader's next __getitem__ will see
        the new labels because we don't rebuild the underlying base dataset.
        """
        if flip_ratio is not None:
            self.flip_ratio = float(flip_ratio)
        self.seed = int(seed)
        self.flipped_labels = self._precompute_flipped_labels()
        self.num_flipped = self._count_flipped()

    def _apply_flip(self, orig: int, rng: np.random.Generator) -> int:
        if self.flip_mode == "pairwise":
            return int(self.flip_map.get(orig, orig))
        if self.flip_mode == "targeted":
            if self.target_class is None:
                raise ValueError("flip_mode='targeted' requires a target_class")
            return int(self.target_class)
        if self.flip_mode == "random":
            # Uniform over all classes != orig.
            choices = [c for c in range(self.num_labels) if c != orig]
            if not choices:
                return orig
            return int(rng.choice(choices))
        raise ValueError(f"Unknown flip_mode={self.flip_mode!r}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        item["labels"] = torch.tensor(self.flipped_labels[idx], dtype=torch.long)
        return item


# --------------------------------------------------------------------------- #
# Hallucination attacker client                                               #
# --------------------------------------------------------------------------- #

class HallucinationAttackerClient(BenignClient):
    """
    Hallucination-inducing attacker: trains on (partially) label-flipped data.

    Inherits BenignClient so `local_train` is literally the FedProx local-SGD
    routine -- no attack-time overhead. The only difference is the DataLoader
    we train against.

    Notes on the FL server interface:
      - is_attacker = True (used by Server phases).
      - claimed_data_size is honored by Server._compute_raw_weights so that,
        even with a benign-looking update, the aggregation weight matches
        real local-data size (realistic scenario).
      - The server also calls receive_benign_updates / set_global_model_params /
        set_constraint_params / set_lagrangian_params on every attacker; for
        hallucination attackers these are intentional no-ops.
    """

    def __init__(
        self,
        client_id: int,
        model,
        data_loader: DataLoader,
        lr: float,
        local_epochs: int,
        alpha: float,
        data_indices: Optional[List[int]] = None,
        grad_clip_norm: float = 1.0,
        flip_ratio: float = 1.0,
        flip_mode: str = "pairwise",
        flip_map: Optional[Dict[int, int]] = None,
        num_labels: int = 4,
        target_class: Optional[int] = None,
        attack_start_round: int = 0,
        claimed_data_size: float = 1.0,
        flip_seed: Optional[int] = None,
        per_round_reseed: bool = False,
        flip_ratio_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(
            client_id=client_id,
            model=model,
            data_loader=data_loader,
            lr=lr,
            local_epochs=local_epochs,
            alpha=alpha,
            data_indices=data_indices,
            grad_clip_norm=grad_clip_norm,
        )
        self.is_attacker = True
        self.attack_method = "Hallucination"
        self.claimed_data_size = float(claimed_data_size)
        self.attack_start_round = int(attack_start_round)
        self.flip_ratio = float(flip_ratio)
        self.flip_mode = str(flip_mode).lower()
        self.flip_map = dict(flip_map or {})
        self.num_labels = int(num_labels)
        self.target_class = target_class
        self._flip_seed = int(flip_seed if flip_seed is not None else client_id)

        # ---- Per-round randomization options ---- #
        # When per_round_reseed=True, prepare_for_round() draws a fresh seed
        # for the flip dataset every round, so the set of flipped indices and
        # the random targets vary round-to-round.  Combined with flip_ratio_range
        # this also varies the fraction of flipped samples each round.
        self.per_round_reseed = bool(per_round_reseed)
        if flip_ratio_range is not None:
            lo, hi = flip_ratio_range
            if not (0.0 <= float(lo) <= float(hi) <= 1.0):
                raise ValueError(
                    f"flip_ratio_range must satisfy 0 <= lo <= hi <= 1; "
                    f"got {flip_ratio_range!r}"
                )
            self.flip_ratio_range: Optional[Tuple[float, float]] = (
                float(lo), float(hi)
            )
        else:
            self.flip_ratio_range = None
        # The per-round scalar ratio is derived statelessly from
        # (flip_seed, round_num) in prepare_for_round(). This preserves the old
        # uninterrupted RNG sequence exactly and also makes checkpoint resume
        # exact; no client-private generator state needs to be serialized.

        # Keep a reference to the honest loader for pre-attack rounds.
        self._honest_loader: DataLoader = data_loader

        # Build the flipped loader once (shared across rounds; the dataset
        # itself is mutated in-place when per_round_reseed=True).
        base_dataset = data_loader.dataset
        flipped_dataset = FlippedLabelDataset(
            base_dataset=base_dataset,
            flip_ratio=self.flip_ratio,
            flip_mode=self.flip_mode,
            flip_map=self.flip_map,
            num_labels=self.num_labels,
            target_class=self.target_class,
            seed=self._flip_seed,
        )
        self._flipped_loader: DataLoader = DataLoader(
            flipped_dataset,
            batch_size=data_loader.batch_size,
            shuffle=True,
        )
        if self.per_round_reseed:
            range_str = (
                f"range={self.flip_ratio_range}"
                if self.flip_ratio_range is not None
                else f"fixed={self.flip_ratio:.2f}"
            )
            print(
                f"  [Hallucination Attacker {client_id}] mode={self.flip_mode}, "
                f"per_round_reseed=True, flip_ratio {range_str}, "
                f"init flipped={flipped_dataset.num_flipped}/{len(flipped_dataset)} "
                f"(claimed_data_size={self.claimed_data_size:.0f})"
            )
        else:
            print(
                f"  [Hallucination Attacker {client_id}] mode={self.flip_mode}, "
                f"flip_ratio={self.flip_ratio:.2f}, "
                f"flipped_samples={flipped_dataset.num_flipped}/{len(flipped_dataset)} "
                f"(claimed_data_size={self.claimed_data_size:.0f})"
            )

    # ----------------------------- life cycle ------------------------------ #

    def prepare_for_round(self, round_num: int) -> None:
        """
        Server phase 1 hook. With per_round_reseed=False this is a pure
        bookkeeping update; with per_round_reseed=True we resample the
        flipped-label set (and optionally flip_ratio) so that the attacker's
        gradient direction varies across rounds. This produces non-stationary
        attack metrics (CSE / local_acc oscillate) which better reflects a
        realistic attacker than a frozen wrong-label manifold.
        """
        self.set_round(round_num)
        if not self.per_round_reseed:
            return
        # Sample (or keep) flip_ratio for this round.
        if self.flip_ratio_range is not None:
            ratio = deterministic_round_flip_ratio(
                self._flip_seed, round_num, self.flip_ratio_range
            )
        else:
            ratio = None  # keep current self.flip_ratio
        # Round/client-distinct seed (large prime avoids low-bit collisions
        # between (client_id, round_num) pairs that could pin two attackers
        # to the same flipped-sample subset).
        seed = self._flip_seed * 100_003 + int(round_num)
        dataset = self._flipped_loader.dataset
        dataset.regenerate(seed=seed, flip_ratio=ratio)
        # Reflect the new flip_ratio on the client object for diagnostics.
        if ratio is not None:
            self.flip_ratio = ratio
        print(
            f"  [Hallucination Attacker {self.client_id}] round={round_num} "
            f"re-flipped: flip_ratio={dataset.flip_ratio:.3f}, "
            f"flipped_samples={dataset.num_flipped}/{len(dataset)}"
        )

    def local_train(self, epochs: Optional[int] = None) -> torch.Tensor:
        """
        Run FedProx-style local training (inherited). Before attack_start_round
        the attacker behaves like a benign client (uses the honest loader), so
        experiments can observe a clean ramp-up before the attack kicks in.
        """
        use_flipped = self.current_round >= self.attack_start_round
        active_loader = self._flipped_loader if use_flipped else self._honest_loader
        prev_loader = self.data_loader
        self.data_loader = active_loader
        try:
            update = super().local_train(epochs=epochs)
        finally:
            self.data_loader = prev_loader
        return update

    # ---------------------------- attack hook ------------------------------ #

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        No post-hoc manipulation: the malicious signal is already baked into
        `poisoned_update` via the flipped-label training step. This keeps the
        attack (a) stealthy by construction and (b) cheap.
        """
        return poisoned_update

    # --------------------- server compatibility no-ops --------------------- #

    def receive_benign_updates(
        self,
        updates: List[torch.Tensor],
        client_ids: Optional[List[int]] = None,
    ) -> None:
        # Hallucination attack is data-driven, not update-driven.
        pass

    def receive_attacker_updates(
        self,
        updates: List[torch.Tensor],
        client_ids: List[int],
        data_sizes: Optional[Dict[int, float]] = None,
    ) -> None:
        pass

    def set_global_model_params(self, global_params: torch.Tensor) -> None:
        # Global weights are already synced via Server.broadcast_model.
        pass

    def set_constraint_params(self, **kwargs) -> None:
        pass

    def set_lagrangian_params(self, **kwargs) -> None:
        pass
