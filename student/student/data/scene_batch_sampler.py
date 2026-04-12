"""Batch samplers for multi-scene training."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class SceneBatchSampler(Sampler[list[int]]):
    """Group scene indices into fixed-count training steps.

    The batch count is fixed by ``max_scenes_per_batch`` so DDP ranks stay aligned.
    When ``batch_policy="point_bucket"``, indices are greedily packed into the
    fixed number of batches to reduce per-step point imbalance while still
    respecting the per-batch scene cap.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        scene_sizes: Sequence[int],
        *,
        max_scenes_per_batch: int,
        max_total_points: int | None = None,
        batch_policy: str = "sequential",
        drop_last: bool = False,
    ) -> None:
        self.sampler = sampler
        self.scene_sizes = [max(int(size), 1) for size in scene_sizes]
        self.max_scenes_per_batch = max(int(max_scenes_per_batch), 1)
        self.max_total_points = (
            None if max_total_points is None else max(int(max_total_points), 1)
        )
        self.batch_policy = str(batch_policy)
        self.drop_last = bool(drop_last)
        if self.batch_policy not in {"sequential", "point_bucket"}:
            raise ValueError(
                f"Unknown batch_policy={self.batch_policy!r}; expected 'sequential' or 'point_bucket'"
            )

    def __len__(self) -> int:
        num_indices = len(self.sampler)
        if self.drop_last:
            return num_indices // self.max_scenes_per_batch
        return math.ceil(num_indices / self.max_scenes_per_batch)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def _sequential_batches(self, indices: list[int]) -> list[list[int]]:
        batches = [
            indices[start:start + self.max_scenes_per_batch]
            for start in range(0, len(indices), self.max_scenes_per_batch)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.max_scenes_per_batch:
            batches.pop()
        return batches

    def _point_bucket_batches(self, indices: list[int]) -> list[list[int]]:
        num_batches = len(self)
        if num_batches == 0:
            return []

        batches: list[list[int]] = [[] for _ in range(num_batches)]
        costs = [0 for _ in range(num_batches)]
        ranked_indices = sorted(
            indices,
            key=lambda idx: (self.scene_sizes[idx], idx),
            reverse=True,
        )

        for idx in ranked_indices:
            size = self.scene_sizes[idx]
            eligible = [
                batch_idx
                for batch_idx in range(num_batches)
                if len(batches[batch_idx]) < self.max_scenes_per_batch
            ]
            if not eligible:
                break

            within_budget = eligible
            if self.max_total_points is not None:
                budget_ok = [
                    batch_idx
                    for batch_idx in eligible
                    if costs[batch_idx] + size <= self.max_total_points
                ]
                if budget_ok:
                    within_budget = budget_ok

            chosen = min(
                within_budget,
                key=lambda batch_idx: (
                    costs[batch_idx],
                    len(batches[batch_idx]),
                    batch_idx,
                ),
            )
            batches[chosen].append(idx)
            costs[chosen] += size

        return [batch for batch in batches if batch]

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(iter(self.sampler))
        if self.batch_policy == "sequential":
            batches = self._sequential_batches(indices)
        else:
            batches = self._point_bucket_batches(indices)
        return iter(batches)
