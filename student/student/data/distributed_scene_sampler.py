"""Distributed samplers specialized for multi-scene training."""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class BalancedDistributedSceneSampler(Sampler[int]):
    """Distributed sampler that balances scene counts and point counts.

    The sampler keeps the number of scenes per rank equal so DDP stays aligned,
    then greedily assigns larger scenes to the currently lightest rank to reduce
    straggler effects in one-scene-per-step training.
    """

    def __init__(
        self,
        scene_sizes: Sequence[int],
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package must be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed package must be initialized")
            rank = dist.get_rank()

        self.scene_sizes = [max(int(size), 1) for size in scene_sizes]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.drop_last:
            self.num_samples = len(self.scene_sizes) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.scene_sizes) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.scene_sizes)))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.scene_sizes), generator=generator).tolist()

    def _pad_or_trim(self, indices: list[int]) -> list[int]:
        if self.drop_last:
            return indices[: self.total_size]

        if len(indices) >= self.total_size:
            return indices[: self.total_size]

        padded = list(indices)
        pad_source = sorted(indices, key=lambda idx: self.scene_sizes[idx])
        if not pad_source:
            return padded

        pad_idx = 0
        while len(padded) < self.total_size:
            padded.append(pad_source[pad_idx % len(pad_source)])
            pad_idx += 1
        return padded

    def __iter__(self) -> Iterator[int]:
        indices = self._pad_or_trim(self._ordered_indices())
        ranked_indices = sorted(
            indices,
            key=lambda idx: (self.scene_sizes[idx], idx),
            reverse=True,
        )

        bins: list[list[int]] = [[] for _ in range(self.num_replicas)]
        costs = [0 for _ in range(self.num_replicas)]

        for idx in ranked_indices:
            eligible_ranks = [
                replica
                for replica in range(self.num_replicas)
                if len(bins[replica]) < self.num_samples
            ]
            replica = min(
                eligible_ranks,
                key=lambda replica: (costs[replica], len(bins[replica]), replica),
            )
            bins[replica].append(idx)
            costs[replica] += self.scene_sizes[idx]

        if self.shuffle and bins[self.rank]:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch + 17)
            order = torch.randperm(len(bins[self.rank]), generator=generator).tolist()
            rank_indices = [bins[self.rank][i] for i in order]
        else:
            rank_indices = bins[self.rank]

        return iter(rank_indices)
