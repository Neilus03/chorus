#!/usr/bin/env python3
"""Lightweight DDP smoke test for ``MultiSceneTrainer``.

This script avoids the full LitePT/ScanNet stack and instead exercises:

- distributed process-group bootstrap
- ``DistributedSampler`` scene sharding
- DDP-wrapped training steps
- rank-0 checkpointing
- rank-0-only validation barriers

Examples::

    python scripts/check_multi_scene_ddp.py --device cpu
    torchrun --standalone --nproc_per_node=4 scripts/check_multi_scene_ddp.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.engine.multi_scene_trainer import MultiSceneTrainer
from student.data.batched_scene_batch import split_tensor_by_offsets


log = logging.getLogger("check_multi_scene_ddp")


def _distributed_env() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _configure_logging(rank: int, is_main_process: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _resolve_runtime_device(
    requested_device: str,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[str, str | None]:
    if distributed:
        if requested_device.startswith("cuda") or requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for DDP smoke test, but no CUDA device is available.")
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}", "nccl"
        return requested_device, "gloo"
    return requested_device, None


class ToySceneDataset(Dataset):
    def __init__(
        self,
        *,
        num_scenes: int,
        points_per_scene: int,
        feature_dim: int,
        granularities: tuple[str, ...],
        seed: int,
    ) -> None:
        self._samples: list[dict[str, object]] = []
        self._scene_ids: list[str] = []
        for idx in range(num_scenes):
            gen = torch.Generator().manual_seed(seed + idx)
            points = torch.randn(points_per_scene, 3, generator=gen)
            features = torch.randn(points_per_scene, feature_dim, generator=gen)

            supervision_mask = torch.ones(points_per_scene, dtype=torch.bool)
            labels = (torch.arange(points_per_scene) // max(points_per_scene // 4, 1)).long()
            labels_by_granularity = {g: labels.clone() for g in granularities}
            scene_id = f"toy_scene_{idx:03d}"

            self._scene_ids.append(scene_id)
            self._samples.append(
                {
                    "scene_id": scene_id,
                    "scene_dir": f"/tmp/{scene_id}",
                    "points": points,
                    "features": features,
                    "labels_by_granularity": labels_by_granularity,
                    "valid_points": torch.ones(points_per_scene, dtype=torch.bool),
                    "seen_points": torch.ones(points_per_scene, dtype=torch.bool),
                    "supervision_mask": supervision_mask,
                    "scene_meta": {"toy": True, "index": idx},
                    "granularities": granularities,
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self._samples[idx]
        return {
            "scene_id": sample["scene_id"],
            "scene_dir": sample["scene_dir"],
            "points": sample["points"].clone(),
            "features": sample["features"].clone(),
            "labels_by_granularity": {
                key: value.clone()
                for key, value in dict(sample["labels_by_granularity"]).items()
            },
            "valid_points": sample["valid_points"].clone(),
            "seen_points": sample["seen_points"].clone(),
            "supervision_mask": sample["supervision_mask"].clone(),
            "scene_meta": dict(sample["scene_meta"]),
            "granularities": sample["granularities"],
        }

    @property
    def scene_ids(self) -> list[str]:
        return list(self._scene_ids)

    @property
    def scene_point_counts(self) -> list[int]:
        return [int(sample["points"].shape[0]) for sample in self._samples]


class ToyBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self._cached_voxelization = None
        self.proj = nn.Linear(in_channels + 3, hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        point_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._cached_voxelization = True
        x = torch.cat([points, features], dim=1)
        return torch.tanh(self.proj(x))


class ToyStudentModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        num_queries: int,
        granularities: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.backbone = ToyBackbone(in_channels, hidden_dim)
        self.query_embed = nn.ParameterDict(
            {
                g: nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.05)
                for g in granularities
            }
        )
        self.score_heads = nn.ModuleDict(
            {g: nn.Linear(hidden_dim, 1) for g in granularities}
        )

    def _forward_single(self, point_embed: torch.Tensor) -> dict[str, object]:
        pooled = point_embed.mean(dim=0, keepdim=True)
        out: dict[str, object] = {"point_embed": point_embed, "heads": {}}
        heads = dict(out["heads"])
        for g, query in self.query_embed.items():
            query_feat = query + pooled
            heads[g] = {
                "mask_logits": query @ point_embed.T,
                "score_logits": self.score_heads[g](query_feat).squeeze(-1),
                "query_embed": query_feat,
            }
        out["heads"] = heads
        return out

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        *,
        point_offsets: torch.Tensor | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        point_embed = self.backbone(points, features, point_offsets=point_offsets)
        if point_offsets is None or point_offsets.numel() <= 1:
            return self._forward_single(point_embed)
        return [
            self._forward_single(scene_embed)
            for scene_embed in split_tensor_by_offsets(point_embed, point_offsets)
        ]


class ToyCriterion(nn.Module):
    def __init__(self, granularities: tuple[str, ...]) -> None:
        super().__init__()
        self.granularities = granularities

    def forward(
        self,
        pred: dict[str, object],
        targets_by_gran: dict[str, object],
    ) -> dict[str, object]:
        point_embed = pred["point_embed"]
        heads_out: dict[str, dict[str, torch.Tensor]] = {}
        total_loss = point_embed.sum() * 0.0

        for g in self.granularities:
            head = pred["heads"][g]
            mask_logits = head["mask_logits"]
            score_logits = head["score_logits"]
            targets = targets_by_gran[g]

            gt_masks = targets.gt_masks.to(mask_logits.device).float()
            supervision_mask = targets.supervision_mask.to(mask_logits.device).bool()
            if gt_masks.numel() > 0:
                target_mask = gt_masks.mean(dim=0)
            else:
                target_mask = torch.zeros(mask_logits.shape[1], device=mask_logits.device)

            pred_mask = mask_logits.mean(dim=0)
            pred_mask = pred_mask[supervision_mask]
            target_mask = target_mask[supervision_mask]
            loss_mask = torch.mean((pred_mask - target_mask) ** 2)

            score_target = torch.full_like(
                score_logits,
                1.0 if gt_masks.shape[0] > 0 else 0.0,
            )
            loss_score = torch.mean((score_logits.sigmoid() - score_target) ** 2)
            loss_total = loss_mask + 0.5 * loss_score
            total_loss = total_loss + loss_total

            heads_out[g] = {
                "loss_total": loss_total.detach(),
                "loss_mask_bce": loss_mask.detach(),
                "loss_mask_dice": loss_mask.detach(),
                "loss_score": loss_score.detach(),
            }

        return {"loss_total": total_loss, "heads": heads_out}


def _stub_validate(self: MultiSceneTrainer, epoch: int) -> dict[str, object]:
    aggregate = {
        "loss_mean": max(0.0, 1.0 - 0.05 * epoch),
        "pseudo_AP25_mean": 0.1 * epoch,
        "pseudo_AP50_mean": 0.05 * epoch,
        "real_AP25_mean": 0.02 * epoch,
        "real_AP50_mean": 0.01 * epoch,
        "pseudo_NMI_mean": 0.0,
        "pseudo_ARI_mean": 0.0,
        "real_NMI_mean": 0.0,
        "real_ARI_mean": 0.0,
        "matched_mean_iou_mean": 0.03 * epoch,
    }
    return {"aggregate": aggregate, "per_scene": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP smoke test for MultiSceneTrainer.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--points-per-scene", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = _distributed_env()
    is_main_process = rank == 0
    _configure_logging(rank, is_main_process)

    pg_initialized = False
    try:
        device, backend = _resolve_runtime_device(
            args.device,
            distributed=distributed,
            local_rank=local_rank,
        )
        if distributed:
            dist.init_process_group(backend=backend or "gloo", init_method="env://")
            pg_initialized = True

        torch.manual_seed(args.seed + rank)
        granularities = ("g02", "g05", "g08")
        train_ds = ToySceneDataset(
            num_scenes=args.num_scenes,
            points_per_scene=args.points_per_scene,
            feature_dim=args.feature_dim,
            granularities=granularities,
            seed=args.seed,
        )
        val_ds = ToySceneDataset(
            num_scenes=max(args.num_scenes // 2, 2),
            points_per_scene=args.points_per_scene,
            feature_dim=args.feature_dim,
            granularities=granularities,
            seed=args.seed + 1000,
        )

        model = ToyStudentModel(
            in_channels=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            granularities=granularities,
        )
        criterion = ToyCriterion(granularities)

        base_out_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else (_STUDENT_ROOT / "logs" / "ddp_smoke")
        )
        out_dir = base_out_dir / f"ws{world_size}_{device.replace(':', '_')}"

        trainer = MultiSceneTrainer(
            model=model,
            criterion=criterion,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            max_epochs=args.max_epochs,
            eval_every_epochs=1,
            train_eval_every_epochs=None,
            save_every_epochs=1,
            output_dir=out_dir,
            score_threshold=0.3,
            mask_threshold=0.5,
            min_points_per_proposal=5,
            eval_benchmark="toy",
            min_instance_points=1,
            warmup_epochs=0,
            granularities=granularities,
            config={"toy_smoke": True},
            num_workers=0,
            log_every_steps=1,
            batch_scenes_per_step=2,
            balance_train_by_points=True,
            profile_train_steps=True,
            profile_every_steps=1,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_main_process=is_main_process,
        )
        trainer._validate = MethodType(_stub_validate, trainer)

        result = trainer.train()
        if is_main_process:
            summary = {
                "device": device,
                "distributed": distributed,
                "world_size": world_size,
                "output_dir": str(out_dir),
                "total_training_time_s": result["total_training_time_s"],
                "best_epoch": result["best_epoch"],
                "best_val_metric": result["best_val_metric"],
                "last_exists": (out_dir / "checkpoints" / "last.pt").is_file(),
                "best_exists": (out_dir / "checkpoints" / "best.pt").is_file(),
            }
            print(json.dumps(summary))
    finally:
        if pg_initialized and _dist_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
