"""Epoch-based trainer over multiple scenes with validation.

Replaces :class:`SingleSceneTrainer` for multi-scene experiments.
Reuses existing criterion, evaluator, and metric functions — only
the training loop structure changes.

Supports two decoder modes:
  - ``"multi_head"`` — discrete per-granularity heads (original)
  - ``"continuous"`` — single-head with continuous granularity conditioning
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from student.data.batched_scene_batch import (
    BatchedMultiSceneSample,
    collate_multi_scene_samples,
)
from student.data.distributed_scene_sampler import BalancedDistributedSceneSampler
from student.data.multi_scene_dataset import MultiSceneDataset
from student.data.scene_batch_sampler import SceneBatchSampler
from student.data.target_builder import (
    build_instance_targets,
    build_instance_targets_multi,
)
from student.engine.multi_scene_evaluator import (
    aggregate_multi_scene_results,
    evaluate_multi_scene,
)
from student.losses.mask_set_loss import MultiGranCriterion, SingleGranCriterion
from student.models.continuous_decoder import ContinuousQueryInstanceDecoder

log = logging.getLogger(__name__)

# ── granularity lookup tables ────────────────────────────────────────────

# Keys and float values for stochastic sampling
_GRAN_KEYS = ("g02", "g05", "g08")
_GRAN_VALS = (0.2, 0.5, 0.8)


def sample_granularity_ddp(
    device: torch.device,
    distributed: bool = False,
    granularity_keys: tuple[str, ...] = _GRAN_KEYS,
    granularity_vals: tuple[float, ...] = _GRAN_VALS,
) -> tuple[str, float]:
    """Sample a single granularity, synchronized across all DDP ranks.

    In DDP, all ranks must execute the same forward path to produce matching
    gradient buckets for ``allreduce``.  This function broadcasts the sampled
    index from rank 0 so all ranks agree.

    Parameters
    ----------
    device:
        Torch device for the broadcast tensor.
    distributed:
        Whether DDP is active.
    granularity_keys:
        String keys like ``("g02", "g05", "g08")``.
    granularity_vals:
        Corresponding float values like ``(0.2, 0.5, 0.8)``.

    Returns
    -------
    (key, value) tuple for the sampled granularity.
    """
    n = len(granularity_keys)
    if distributed and dist.is_initialized():
        idx_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if dist.get_rank() == 0:
            idx_tensor[0] = torch.randint(n, (1,)).item()
        dist.broadcast(idx_tensor, src=0)
        idx = int(idx_tensor.item())
    else:
        idx = torch.randint(n, (1,)).item()

    return granularity_keys[idx], granularity_vals[idx]

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


def _wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


class MultiSceneTrainer:
    """Epoch-based trainer over multiple scenes with validation.

    Parameters
    ----------
    model:
        ``StudentInstanceSegModel`` with LitePT backbone and multi-head decoder.
    criterion:
        ``MultiGranCriterion`` instance (shared for all scenes).
    train_dataset:
        ``MultiSceneDataset`` for training scenes.
    val_dataset:
        ``MultiSceneDataset`` for validation scenes.
    device:
        CUDA device string.
    lr:
        Learning rate (uniform for all parameters — training from scratch).
    weight_decay:
        AdamW weight decay.
    grad_clip_norm:
        Maximum gradient norm for clipping.
    max_epochs:
        Total training epochs.
    eval_every_epochs:
        Run validation every N epochs.
    save_every_epochs:
        Save ``last.pt`` checkpoint every N epochs.
    output_dir:
        Directory for checkpoints, logs, and metrics.
    score_threshold / mask_threshold:
        For proposal extraction during evaluation.
    min_points_per_proposal:
        Minimum points for a valid proposal.
    eval_benchmark:
        ScanNet GT evaluation benchmark name.
    min_instance_points:
        Filter pseudo-instances smaller than this when building targets.
    warmup_epochs:
        Linear LR warmup from near-zero to *lr* over this many epochs.
    granularities:
        Dot-free granularity keys.
    config:
        Optional full config dict stored in checkpoints for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: MultiGranCriterion,
        train_dataset: MultiSceneDataset,
        val_dataset: MultiSceneDataset,
        *,
        device: str = "cuda:0",
        lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        max_epochs: int = 50,
        eval_every_epochs: int = 5,
        train_eval_every_epochs: int | None = None,
        train_eval_num_scenes: int = 3,
        train_eval_scene_ids: list[str] | None = None,
        train_eval_selection: str = "first",
        save_every_epochs: int = 10,
        output_dir: Path | str,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        min_points_per_proposal: int = 30,
        eval_benchmark: str = "scannet200",
        eval_benchmarks: str | list[str] | tuple[str, ...] | None = None,
        min_instance_points: int = 10,
        warmup_epochs: int = 5,
        granularities: tuple[str, ...] = ("g02", "g05", "g08"),
        config: dict[str, Any] | None = None,
        num_workers: int = 0,
        log_every_steps: int = 1,
        batch_scenes_per_step: int = 1,
        balance_train_by_points: bool = False,
        drop_last_train: bool = False,
        sync_step_metrics: bool = True,
        log_scene_ids: bool = False,
        profile_train_steps: bool = False,
        profile_every_steps: int = 1,
        max_total_points_per_batch: int | None = None,
        batch_assembly_policy: str = "sequential",
        decoder_loss_mode: str = "scene_local",
        sampler_seed: int = 0,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        is_main_process: bool = True,
        dense_instance_ids: bool = False,
        fragment_merge_eval: bool = False,
        fragment_merge_num: int = 4,
        fragment_merge_point_max: int | None = None,
        fragment_merge_seed: int = 0,
    ) -> None:
        self.device = device
        self.lr = lr
        self.backbone_lr_scale = backbone_lr_scale
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.eval_every_epochs = eval_every_epochs
        self.train_eval_every_epochs = train_eval_every_epochs
        self.train_eval_num_scenes = train_eval_num_scenes
        self.train_eval_scene_ids = train_eval_scene_ids
        self.train_eval_selection = train_eval_selection
        self.save_every_epochs = save_every_epochs
        self.output_dir = Path(output_dir)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_points_per_proposal = min_points_per_proposal
        self.eval_benchmark = eval_benchmark
        self.eval_benchmarks = eval_benchmarks
        self.min_instance_points = min_instance_points
        self.granularities = granularities
        self.config = config
        self.num_workers = max(int(num_workers), 0)
        self.log_every_steps = max(int(log_every_steps), 1)
        self.batch_scenes_per_step = max(int(batch_scenes_per_step), 1)
        self.balance_train_by_points = bool(balance_train_by_points)
        self.drop_last_train = bool(drop_last_train)
        self.sync_step_metrics = bool(sync_step_metrics)
        self.log_scene_ids = bool(log_scene_ids)
        self.profile_train_steps = bool(profile_train_steps)
        self.profile_every_steps = max(int(profile_every_steps), 1)
        self.max_total_points_per_batch = (
            None
            if max_total_points_per_batch is None
            else max(int(max_total_points_per_batch), 1)
        )
        self.batch_assembly_policy = str(batch_assembly_policy)
        self.decoder_loss_mode = str(decoder_loss_mode)
        self.sampler_seed = int(sampler_seed)
        self.distributed = bool(distributed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.is_main_process = bool(is_main_process)
        self.dense_instance_ids = bool(dense_instance_ids)
        self.fragment_merge_eval = bool(fragment_merge_eval)
        self.fragment_merge_num = int(fragment_merge_num)
        self.fragment_merge_point_max = fragment_merge_point_max
        self.fragment_merge_seed = int(fragment_merge_seed)

        # Detect continuous decoder mode
        base_decoder = getattr(model, "decoder", None)
        self._continuous = isinstance(base_decoder, ContinuousQueryInstanceDecoder)

        # Build granularity key→value mapping for DDP-safe sampling
        self._gran_keys = tuple(granularities)
        self._gran_vals = tuple(
            float(g.replace("g0", "0.").replace("g", "0."))
            for g in granularities
        )

        if self.decoder_loss_mode != "scene_local":
            raise ValueError(
                f"Unsupported decoder_loss_mode={self.decoder_loss_mode!r}; only 'scene_local' is implemented"
            )

        base_model = model.to(device)
        if self.distributed:
            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                # Some decoder/head parameters can be legitimately unused on a
                # given scene, so plain DDP reduction can deadlock/fail on the
                # next step without unused-parameter detection enabled.
                "find_unused_parameters": True,
            }
            if str(device).startswith("cuda"):
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.model: nn.Module = DistributedDataParallel(base_model, **ddp_kwargs)
        else:
            self.model = base_model
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        base_model = _unwrap_model(self.model)
        if hasattr(base_model, "parameter_groups"):
            param_groups = [
                {"params": pg["params"], "lr": lr * pg["lr_scale"]}
                for pg in base_model.parameter_groups(backbone_lr_scale)
            ]
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            # Uniform LR — training from scratch, no pretrained backbone weights
            self.optimizer = torch.optim.AdamW(
                base_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

    

        # Linear warmup → MultiStepLR
        if warmup_epochs > 0 and max_epochs > warmup_epochs:
            warmup_sched = LinearLR(
                self.optimizer, start_factor=0.001, total_iters=warmup_epochs,
            )
            # MultiStepLR milestones are relative to the scheduler's own epoch
            # counter. With SequentialLR, the second scheduler starts counting at
            # 0 after warmup, so we offset to keep drops at global epochs 600/800.
            base_milestones = [600, 800]
            adjusted_milestones = sorted({
                max(1, m - warmup_epochs) for m in base_milestones if m > warmup_epochs
            })
            multistep_sched = MultiStepLR(
                self.optimizer,
                milestones=adjusted_milestones,
                gamma=0.1,
            )
            self.scheduler: torch.optim.lr_scheduler.LRScheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, multistep_sched],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=[600, 800],
                gamma=0.1,
            )

        self.train_sampler = (
            BalancedDistributedSceneSampler(
                train_dataset.scene_point_counts,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.sampler_seed,
                drop_last=self.drop_last_train,
            )
            if self.distributed and self.balance_train_by_points
            else DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=self.drop_last_train,
            )
            if self.distributed
            else RandomSampler(train_dataset)
        )
        self.train_batch_sampler = SceneBatchSampler(
            self.train_sampler,
            train_dataset.scene_point_counts,
            max_scenes_per_batch=self.batch_scenes_per_step,
            max_total_points=self.max_total_points_per_batch,
            batch_policy=self.batch_assembly_policy,
            drop_last=self.drop_last_train,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=collate_multi_scene_samples,
            num_workers=self.num_workers,
            pin_memory=str(device).startswith("cuda"),
            persistent_workers=self.num_workers > 0,
        )

        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = -1.0
        self.best_epoch = -1
        self.best_val_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ #

    def _dist_barrier(self) -> None:
        if self.distributed and _dist_ready():
            dist.barrier()

    def _model_module(self) -> nn.Module:
        return _unwrap_model(self.model)

    def _sync_device(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _gather_scene_ids(self, scene_ids: list[str]) -> list[list[str]]:
        if not self.distributed:
            return [scene_ids]
        gathered: list[list[str] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, scene_ids)
        return [list(x or []) for x in gathered]

    def _reduce_mean_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.distributed:
            return metrics
        names = list(metrics.keys())
        values = torch.tensor(
            [metrics[name] for name in names],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {name: values[idx].item() for idx, name in enumerate(names)}

    def _shard_eval_dataset(
        self,
        dataset: MultiSceneDataset | Subset,
    ) -> MultiSceneDataset | Subset:
        """Shard the eval dataset across ranks with point-count balancing.

        Instead of a naive stride ``range(rank, N, world_size)`` which can
        assign all large scenes to a single rank, we greedily assign each
        scene to the lightest-loaded rank (longest-processing-time-first
        heuristic).  This keeps evaluation time similar across ranks and
        avoids NCCL collective timeouts that occur when one rank finishes
        far earlier than another.
        """
        if not self.distributed:
            return dataset
        n = len(dataset)
        if n == 0:
            return Subset(dataset, [])

        # Retrieve per-scene point counts (proxy for eval time).
        base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        if isinstance(base_ds, MultiSceneDataset) and hasattr(base_ds, "scene_point_counts"):
            all_counts = base_ds.scene_point_counts
            if isinstance(dataset, Subset):
                counts = [all_counts[i] for i in dataset.indices]
            else:
                counts = list(all_counts)
        else:
            # Fallback: uniform weight (reverts to round-robin).
            counts = [1] * n

        # Greedy LPT (longest processing time first) assignment.
        indexed = sorted(enumerate(counts), key=lambda x: -x[1])
        rank_loads = [0] * self.world_size
        rank_indices: list[list[int]] = [[] for _ in range(self.world_size)]
        for idx, count in indexed:
            lightest = min(range(self.world_size), key=lambda r: rank_loads[r])
            rank_indices[lightest].append(idx)
            rank_loads[lightest] += count

        my_indices = sorted(rank_indices[self.rank])
        return Subset(dataset, my_indices)

    def _gather_eval_per_scene(
        self,
        per_scene: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not self.distributed:
            return per_scene
        gathered: list[dict[str, dict[str, Any]] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, per_scene)
        if not self.is_main_process:
            return {}

        merged: dict[str, dict[str, Any]] = {}
        for shard in gathered:
            if not shard:
                continue
            overlap = set(merged).intersection(shard)
            if overlap:
                duplicate_scene_ids = ", ".join(sorted(overlap))
                raise RuntimeError(
                    "Distributed evaluation gathered duplicate scene ids: "
                    f"{duplicate_scene_ids}"
                )
            merged.update(shard)

        return {
            scene_id: merged[scene_id]
            for scene_id in sorted(merged)
        }

    def _evaluate_dataset(
        self,
        dataset: MultiSceneDataset | Subset,
        *,
        epoch: int,
        stage_name: str,
    ) -> dict[str, Any]:
        if self.is_main_process:
            if self.distributed:
                log.info(
                    "%s at epoch %d ... (%d total scenes across %d ranks)",
                    stage_name,
                    epoch,
                    len(dataset),
                    self.world_size,
                )
            else:
                log.info("%s at epoch %d ...", stage_name, epoch)

        shard_dataset = self._shard_eval_dataset(dataset)
        local_result = evaluate_multi_scene(
            model=self._model_module(),
            dataset=shard_dataset,  # type: ignore[arg-type]
            criterion=self.criterion,
            device=self.device,
            granularities=self.granularities,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            min_points=self.min_points_per_proposal,
            eval_benchmark=self.eval_benchmark,
            eval_benchmarks=self.eval_benchmarks,
            min_instance_points=self.min_instance_points,
            dense_instance_ids=self.dense_instance_ids,
            fragment_merge_eval=self.fragment_merge_eval,
            fragment_merge_num=self.fragment_merge_num,
            fragment_merge_point_max=self.fragment_merge_point_max,
            fragment_merge_seed=self.fragment_merge_seed,
        )

        self._model_module().eval()
        if not self.distributed:
            return local_result

        merged_per_scene = self._gather_eval_per_scene(local_result["per_scene"])
        if not self.is_main_process:
            return {}

        return {
            "per_scene": merged_per_scene,
            "aggregate": aggregate_multi_scene_results(
                merged_per_scene,
                granularities=self.granularities,
            ),
        }

    def _make_train_eval_subset(self, epoch: int) -> Subset:
        """Create a small training-scene subset for metric evaluation."""
        n_total = len(self.train_dataset)
        if n_total == 0:
            raise ValueError("train_dataset is empty")

        scene_ids = self.train_dataset.scene_ids
        id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

        if self.train_eval_scene_ids:
            chosen_ids = [sid for sid in self.train_eval_scene_ids if sid in id_to_idx]
            if not chosen_ids:
                raise ValueError(
                    "train_eval_scene_ids did not match any training scene ids: "
                    f"{self.train_eval_scene_ids}"
                )
            indices = [id_to_idx[sid] for sid in chosen_ids]
        else:
            k = min(max(self.train_eval_num_scenes, 0), n_total)
            if k == 0:
                raise ValueError("train_eval_num_scenes must be >= 1")

            if self.train_eval_selection == "random":
                rng = random.Random(epoch * 9973 + 1337)
                indices = rng.sample(list(range(n_total)), k=k)
            elif self.train_eval_selection == "first":
                indices = list(range(k))
            else:
                raise ValueError(
                    f"Unknown train_eval_selection={self.train_eval_selection!r} "
                    "(expected 'first' or 'random')"
                )

        return Subset(self.train_dataset, indices)

    def _clear_backbone_cache(self) -> None:
        """Invalidate LitePT voxelization cache before processing a new scene."""
        backbone = getattr(self._model_module(), "backbone", None)
        if backbone is not None and hasattr(backbone, "_cached_voxelization"):
            backbone._cached_voxelization = None

    def _build_targets_for_batch(
        self,
        batch: BatchedMultiSceneSample,
        granularity_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build instance targets for a batch.

        Parameters
        ----------
        batch:
            Batched scene data.
        granularity_key:
            If given (continuous decoder mode), build targets for this single
            granularity only.  The returned dicts map the single key to its
            :class:`InstanceTargets`.
        """
        targets_by_scene: list[dict[str, Any]] = []

        if granularity_key is not None:
            # Continuous decoder: single-granularity targets
            for scene_idx in range(batch.num_scenes):
                labels = batch.labels_by_granularity[granularity_key][scene_idx]
                targets = build_instance_targets(
                    labels,
                    batch.supervision_masks[scene_idx],
                    min_instance_points=self.min_instance_points,
                    ignore_label=-1,
                    dense_instance_ids=self.dense_instance_ids,
                )
                targets_by_scene.append({granularity_key: targets})
        else:
            # Multi-head decoder: all granularities
            for scene_idx in range(batch.num_scenes):
                labels_by_granularity = {
                    g: batch.labels_by_granularity[g][scene_idx]
                    for g in self.granularities
                }
                targets_by_scene.append(build_instance_targets_multi(
                    labels_by_granularity,
                    batch.supervision_masks[scene_idx],
                    min_instance_points=self.min_instance_points,
                    dense_instance_ids=self.dense_instance_ids,
                ))
        return targets_by_scene

    def _aggregate_scene_loss_results(
        self,
        scene_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not scene_results:
            raise ValueError("scene_results must not be empty")

        result: dict[str, Any] = {
            "loss_total": torch.stack([x["loss_total"] for x in scene_results]).mean(),
            "scene_loss_totals": [
                float(x["loss_total"].detach().item()) for x in scene_results
            ],
            "heads": {},
        }
        for g in self.granularities:
            result["heads"][g] = {
                "loss_total": torch.stack([
                    x["heads"][g]["loss_total"].detach() for x in scene_results
                ]).mean(),
                "loss_mask_bce": torch.stack([
                    x["heads"][g]["loss_mask_bce"] for x in scene_results
                ]).mean(),
                "loss_mask_dice": torch.stack([
                    x["heads"][g]["loss_mask_dice"] for x in scene_results
                ]).mean(),
                "loss_score": torch.stack([
                    x["heads"][g]["loss_score"] for x in scene_results
                ]).mean(),
            }

        aux_values = [x["loss_aux"] for x in scene_results if "loss_aux" in x]
        if aux_values:
            result["loss_aux"] = torch.stack(aux_values).mean()
        return result

    def _compute_step_loss(
        self,
        pred: dict[str, Any] | list[dict[str, Any]],
        targets_by_scene: list[dict[str, Any]],
        *,
        sampled_g_key: str | None = None,
    ) -> dict[str, Any]:
        """Compute loss for a training step.

        Parameters
        ----------
        pred:
            Model output (single dict or list of per-scene dicts).
        targets_by_scene:
            List of target dicts, one per scene.
        sampled_g_key:
            If set (continuous decoder mode), each target dict maps this
            single key to its :class:`InstanceTargets`.
        """
        if self._continuous and sampled_g_key is not None:
            return self._compute_step_loss_continuous(pred, targets_by_scene, sampled_g_key)

        # Multi-head path (unchanged)
        if isinstance(pred, list):
            if len(pred) != len(targets_by_scene):
                raise ValueError(
                    f"Got {len(pred)} scene predictions but {len(targets_by_scene)} targets"
                )
            return self._aggregate_scene_loss_results([
                self.criterion(scene_pred, scene_targets)
                for scene_pred, scene_targets in zip(pred, targets_by_scene)
            ])

        if len(targets_by_scene) != 1:
            raise ValueError(
                f"Single-scene prediction requires exactly one target set, got {len(targets_by_scene)}"
            )
        loss_result = self.criterion(pred, targets_by_scene[0])
        loss_result["scene_loss_totals"] = [
            float(loss_result["loss_total"].detach().item())
        ]
        return loss_result

    def _compute_step_loss_continuous(
        self,
        pred: dict[str, Any] | list[dict[str, Any]],
        targets_by_scene: list[dict[str, Any]],
        sampled_g_key: str,
    ) -> dict[str, Any]:
        """Compute loss for continuous decoder (single sampled granularity)."""
        scene_results: list[dict[str, Any]] = []

        preds_list = pred if isinstance(pred, list) else [pred]
        if len(preds_list) != len(targets_by_scene):
            raise ValueError(
                f"Got {len(preds_list)} scene predictions but {len(targets_by_scene)} targets"
            )

        for scene_pred, scene_targets in zip(preds_list, targets_by_scene):
            targets_g = scene_targets[sampled_g_key]
            ctx = f"continuous/g={sampled_g_key}"
            scene_loss = self.criterion(scene_pred, targets_g, context=ctx)
            scene_results.append(scene_loss)

        if len(scene_results) == 1:
            result = scene_results[0]
            result["scene_loss_totals"] = [
                float(result["loss_total"].detach().item())
            ]
            # Wrap in a heads-like structure for logging compatibility
            result["heads"] = {sampled_g_key: {
                "loss_total": result["loss_total"].detach(),
                "loss_mask_bce": result["loss_mask_bce"],
                "loss_mask_dice": result["loss_mask_dice"],
                "loss_score": result["loss_score"],
            }}
            return result

        # Aggregate across scenes
        total = torch.stack([r["loss_total"] for r in scene_results]).mean()
        result: dict[str, Any] = {
            "loss_total": total,
            "scene_loss_totals": [
                float(r["loss_total"].detach().item()) for r in scene_results
            ],
            "heads": {
                sampled_g_key: {
                    "loss_total": torch.stack([
                        r["loss_total"].detach() for r in scene_results
                    ]).mean(),
                    "loss_mask_bce": torch.stack([
                        r["loss_mask_bce"] for r in scene_results
                    ]).mean(),
                    "loss_mask_dice": torch.stack([
                        r["loss_mask_dice"] for r in scene_results
                    ]).mean(),
                    "loss_score": torch.stack([
                        r["loss_score"] for r in scene_results
                    ]).mean(),
                },
            },
        }
        aux_values = [r["loss_aux"] for r in scene_results if "loss_aux" in r]
        if aux_values:
            result["loss_aux"] = torch.stack(aux_values).mean()
        return result

    def _summarize_profile_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            return {}

        def _mean(key: str) -> float:
            values = [float(row[key]) for row in rows]
            return sum(values) / len(values)

        summary: dict[str, Any] = {
            "num_profile_rows": len(rows),
            "data_wait_ms_mean": _mean("data_wait_ms"),
            "target_build_ms_mean": _mean("target_build_ms"),
            "forward_ms_mean": _mean("forward_ms"),
            "loss_ms_mean": _mean("loss_ms"),
            "backward_ms_mean": _mean("backward_ms"),
            "optim_ms_mean": _mean("optim_ms"),
            "metrics_sync_ms_mean": _mean("metrics_sync_ms"),
            "step_total_ms_mean": _mean("step_total_ms"),
            "step_total_ms_median": statistics.median(
                [float(row["step_total_ms"]) for row in rows]
            ),
            "step_total_ms_max": max(float(row["step_total_ms"]) for row in rows),
            "num_points_total_mean": _mean("num_points_total"),
            "num_points_total_median": statistics.median(
                [float(row["num_points_total"]) for row in rows]
            ),
            "num_points_total_max": max(float(row["num_points_total"]) for row in rows),
            "num_points_max_scene_mean": _mean("num_points_max_scene"),
            "num_points_max_scene_median": statistics.median(
                [float(row["num_points_max_scene"]) for row in rows]
            ),
            "num_points_max_scene_max": max(
                float(row["num_points_max_scene"]) for row in rows
            ),
        }

        per_rank_means: dict[int, float] = {}
        for rank in sorted({int(row["rank"]) for row in rows}):
            rank_rows = [row for row in rows if int(row["rank"]) == rank]
            per_rank_means[rank] = sum(
                float(row["step_total_ms"]) for row in rank_rows
            ) / max(len(rank_rows), 1)
        if per_rank_means:
            mean_rank_step_ms = sum(per_rank_means.values()) / len(per_rank_means)
            summary["per_rank_mean_step_ms"] = {
                str(rank): value for rank, value in per_rank_means.items()
            }
            summary["max_rank_step_ms_over_mean"] = (
                max(per_rank_means.values()) / max(mean_rank_step_ms, 1e-8)
            )

        return summary

    def _finalize_train_timing(
        self,
        epoch: int,
        local_rows: list[dict[str, Any]],
    ) -> None:
        if not self.profile_train_steps:
            return

        gathered: list[list[dict[str, Any]] | None]
        if self.distributed:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local_rows)
        else:
            gathered = [local_rows]

        if not self.is_main_process:
            return

        all_rows: list[dict[str, Any]] = []
        for shard in gathered:
            if shard:
                all_rows.extend(shard)
        if not all_rows:
            return

        all_rows.sort(key=lambda row: (int(row["global_step"]), int(row["rank"])))
        for row in all_rows:
            self._log_row({"train_step_profile": True, **row})

        summary = self._summarize_profile_rows(all_rows)
        summary.update({
            "epoch": epoch,
            "train_step_profile_summary": True,
            "world_size": self.world_size,
        })
        self._log_row(summary)
        log.info(
            "  [train profile epoch %d] step_total mean/median/max = %.1f / %.1f / %.1f ms  imbalance=%.3f",
            epoch,
            summary.get("step_total_ms_mean", 0.0),
            summary.get("step_total_ms_median", 0.0),
            summary.get("step_total_ms_max", 0.0),
            summary.get("max_rank_step_ms_over_mean", 1.0),
        )

        if _wandb_active():
            wandb.log({
                "epoch": epoch,
                "global_step": self.global_step,
                "train_profile/step_total_ms_mean": summary.get("step_total_ms_mean", 0.0),
                "train_profile/step_total_ms_median": summary.get("step_total_ms_median", 0.0),
                "train_profile/step_total_ms_max": summary.get("step_total_ms_max", 0.0),
                "train_profile/data_wait_ms_mean": summary.get("data_wait_ms_mean", 0.0),
                "train_profile/target_build_ms_mean": summary.get("target_build_ms_mean", 0.0),
                "train_profile/forward_ms_mean": summary.get("forward_ms_mean", 0.0),
                "train_profile/loss_ms_mean": summary.get("loss_ms_mean", 0.0),
                "train_profile/backward_ms_mean": summary.get("backward_ms_mean", 0.0),
                "train_profile/optim_ms_mean": summary.get("optim_ms_mean", 0.0),
                "train_profile/metrics_sync_ms_mean": summary.get("metrics_sync_ms_mean", 0.0),
                "train_profile/max_rank_step_ms_over_mean": summary.get(
                    "max_rank_step_ms_over_mean", 1.0
                ),
            })

    def _log_row(self, row: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _save_checkpoint(self, tag: str) -> None:
        if not self.is_main_process:
            return
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self._model_module().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_metric": self.best_val_metric,
                "best_epoch": self.best_epoch,
                "config": self.config,
            },
            path,
        )
        log.info("  Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load training state from a checkpoint and continue training."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_state_dict = checkpoint["model_state_dict"]
        if any(key.startswith("module.") for key in model_state_dict):
            model_state_dict = {
                key.removeprefix("module."): value
                for key, value in model_state_dict.items()
            }
        try:
            self._model_module().load_state_dict(model_state_dict)
        except RuntimeError as exc:
            msg = str(exc)
            if "size mismatch" in msg and "stem.conv.weight" in msg:
                saved = checkpoint.get("config") or {}
                saved_data = saved.get("data", {}) if isinstance(saved, dict) else {}
                saved_bb = (
                    (saved.get("model") or {}).get("backbone", {})
                    if isinstance(saved, dict)
                    else {}
                )
                raise RuntimeError(
                    f"{msg}\n"
                    "Hint: first conv input width mismatch — the checkpoint was trained with a "
                    "different point feature layout than the current config (e.g. "
                    f"saved data.use_normals={saved_data.get('use_normals', '?')}, "
                    f"saved backbone.in_channels={saved_bb.get('in_channels', '?')}). "
                    "Match data.use_normals / data.append_xyz_to_features / "
                    "model.backbone.in_channels to the run that produced this checkpoint."
                ) from exc
            raise
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_metric = float(checkpoint.get("best_val_metric", -1.0))
        self.best_epoch = int(checkpoint.get("best_epoch", -1))

        log.info(
            "Resumed from %s (epoch=%d, global_step=%d, best_val=%.4f @ epoch %d)",
            ckpt_path,
            self.current_epoch,
            self.global_step,
            self.best_val_metric,
            self.best_epoch,
        )

    def load_weights_only(self, path: Path | str, *, strict: bool = True) -> None:
        """Load model weights from a checkpoint but reset training state.

        Intended for finetuning: initialize the model from a prior run while
        restarting epoch counters and best-metric tracking. Optimizer/scheduler
        state is intentionally NOT restored.
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise KeyError(
                f"Checkpoint {ckpt_path} missing 'model_state_dict' (keys={sorted(checkpoint.keys())})"
            )
        if any(key.startswith("module.") for key in model_state_dict):
            model_state_dict = {
                key.removeprefix("module."): value
                for key, value in model_state_dict.items()
            }

        self._model_module().load_state_dict(model_state_dict, strict=strict)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = -1.0
        self.best_epoch = -1

        log.info(
            "Loaded weights-only init from %s (strict=%s); reset epoch/global_step/best metrics.",
            ckpt_path,
            strict,
        )

    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """Train on all scenes once.  Returns epoch-level metrics."""
        self.model.train()
        if hasattr(self.train_batch_sampler, "set_epoch"):
            self.train_batch_sampler.set_epoch(epoch)
        elif hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        scene_losses: list[float] = []
        profile_rows: list[dict[str, Any]] = []

        train_iter = iter(self.train_loader)
        while True:
            data_wait_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break

            if not isinstance(batch, BatchedMultiSceneSample):
                raise TypeError(
                    f"Expected BatchedMultiSceneSample from train_loader, got {type(batch)!r}"
                )

            self.global_step += 1
            data_wait_ms = (time.perf_counter() - data_wait_start) * 1000.0

            profile_this_step = (
                self.profile_train_steps
                and self.global_step % self.profile_every_steps == 0
            )
            if profile_this_step:
                self._sync_device()
            step_start = time.perf_counter()

            scene_ids_local = list(batch.scene_ids)
            point_counts = batch.point_counts
            num_points_total = sum(point_counts)
            num_points_max_scene = max(point_counts) if point_counts else 0

            points = batch.points.to(self.device, non_blocking=True)
            features = batch.features.to(self.device, non_blocking=True)
            point_offsets = batch.point_offsets.to(self.device, non_blocking=True)

            # ── continuous decoder: DDP-safe granularity sampling ──
            sampled_g_key: str | None = None
            sampled_g_val: float | None = None
            if self._continuous:
                sampled_g_key, sampled_g_val = sample_granularity_ddp(
                    torch.device(self.device),
                    distributed=self.distributed,
                    granularity_keys=self._gran_keys,
                    granularity_vals=self._gran_vals,
                )

            target_start = time.perf_counter()
            targets_by_scene = self._build_targets_for_batch(
                batch, granularity_key=sampled_g_key,
            )
            target_build_ms = (time.perf_counter() - target_start) * 1000.0

            self._clear_backbone_cache()
            self.optimizer.zero_grad(set_to_none=True)

            if profile_this_step:
                self._sync_device()
            forward_start = time.perf_counter()

            # Forward pass: pass target_g for continuous decoder
            forward_kwargs: dict[str, Any] = {
                "point_offsets": point_offsets,
            }
            if self._continuous and sampled_g_val is not None:
                forward_kwargs["target_g"] = sampled_g_val

            pred = self.model(
                points,
                features,
                **forward_kwargs,
            )
            if profile_this_step:
                self._sync_device()
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            if profile_this_step:
                self._sync_device()
            loss_start = time.perf_counter()
            loss_result = self._compute_step_loss(
                pred, targets_by_scene, sampled_g_key=sampled_g_key,
            )
            if profile_this_step:
                self._sync_device()
            loss_ms = (time.perf_counter() - loss_start) * 1000.0

            if profile_this_step:
                self._sync_device()
            backward_start = time.perf_counter()
            loss_result["loss_total"].backward()
            if profile_this_step:
                self._sync_device()
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            if profile_this_step:
                self._sync_device()
            optim_start = time.perf_counter()
            grad_norm = clip_grad_norm_(
                self._model_module().parameters(), self.grad_clip_norm,
            ).item()
            self.optimizer.step()
            if profile_this_step:
                self._sync_device()
            optim_ms = (time.perf_counter() - optim_start) * 1000.0

            loss_total = float(loss_result["loss_total"].detach().item())
            scene_losses.extend(loss_result["scene_loss_totals"])
            step_total_ms = (time.perf_counter() - step_start) * 1000.0

            metrics_sync_ms = 0.0
            should_log_step = self.global_step % self.log_every_steps == 0
            if should_log_step:
                step_metrics: dict[str, float] = {
                    "loss_total": loss_total,
                    "grad_norm": grad_norm,
                    "step_ms": step_total_ms,
                    "num_scenes": float(len(scene_ids_local)),
                    "num_points_total": float(num_points_total),
                    "num_points_max_scene": float(num_points_max_scene),
                }
                # Collect per-head metrics from loss_result["heads"]
                active_grans = (
                    [sampled_g_key] if sampled_g_key is not None
                    else list(self.granularities)
                )
                for g in active_grans:
                    if g in loss_result.get("heads", {}):
                        ld_g = loss_result["heads"][g]
                        step_metrics[f"loss_{g}"] = ld_g["loss_total"].item()
                        step_metrics[f"loss_mask_bce_{g}"] = ld_g["loss_mask_bce"].item()
                        step_metrics[f"loss_mask_dice_{g}"] = ld_g["loss_mask_dice"].item()
                        step_metrics[f"loss_score_{g}"] = ld_g["loss_score"].item()
                if "loss_aux" in loss_result:
                    step_metrics["loss_aux"] = loss_result["loss_aux"].item()

                if self.distributed and self.sync_step_metrics:
                    sync_start = time.perf_counter()
                    step_metrics = self._reduce_mean_metrics(step_metrics)
                    metrics_sync_ms += (time.perf_counter() - sync_start) * 1000.0

                scene_ids_by_rank = [scene_ids_local]
                if self.distributed and self.log_scene_ids:
                    sync_start = time.perf_counter()
                    scene_ids_by_rank = self._gather_scene_ids(scene_ids_local)
                    metrics_sync_ms += (time.perf_counter() - sync_start) * 1000.0

                if self.is_main_process:
                    per_head_str = "  ".join(
                        f"{g}={step_metrics[f'loss_{g}']:.4f}"
                        for g in active_grans
                        if f"loss_{g}" in step_metrics
                    )
                    if self.distributed and self.log_scene_ids:
                        scene_label = " | ".join(
                            ",".join(rank_scene_ids) for rank_scene_ids in scene_ids_by_rank
                        )
                    else:
                        scene_label = ",".join(scene_ids_local)
                    log.info(
                        "  epoch %d  step %d  scenes=%s  loss=%.4f  [%s]  "
                        "gnorm=%.3f  %.0fms  pts=%d  max_scene_pts=%d",
                        epoch,
                        self.global_step,
                        scene_label,
                        step_metrics["loss_total"],
                        per_head_str,
                        step_metrics["grad_norm"],
                        step_metrics["step_ms"],
                        int(step_metrics["num_points_total"]),
                        int(step_metrics["num_points_max_scene"]),
                    )

                    row: dict[str, Any] = {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "scene_ids": scene_ids_local,
                        "loss_total": step_metrics["loss_total"],
                        "grad_norm": step_metrics["grad_norm"],
                        "step_ms": step_metrics["step_ms"],
                        "num_scenes": int(step_metrics["num_scenes"]),
                        "num_points_total": int(step_metrics["num_points_total"]),
                        "num_points_max_scene": int(step_metrics["num_points_max_scene"]),
                        "metrics_synced": self.distributed and self.sync_step_metrics,
                        "world_size": self.world_size,
                    }
                    if self.distributed and self.log_scene_ids:
                        row["scene_ids_by_rank"] = scene_ids_by_rank
                    for g in active_grans:
                        if f"loss_{g}" in step_metrics:
                            row[f"loss_{g}"] = step_metrics[f"loss_{g}"]
                            row[f"loss_mask_bce_{g}"] = step_metrics[f"loss_mask_bce_{g}"]
                            row[f"loss_mask_dice_{g}"] = step_metrics[f"loss_mask_dice_{g}"]
                            row[f"loss_score_{g}"] = step_metrics[f"loss_score_{g}"]
                    if "loss_aux" in step_metrics:
                        row["loss_aux"] = step_metrics["loss_aux"]
                    self._log_row(row)

                    if _wandb_active():
                        wb: dict[str, Any] = {
                            "global_step": self.global_step,
                            "epoch": epoch,
                            "train_step/loss": step_metrics["loss_total"],
                            "train_step/grad_norm": step_metrics["grad_norm"],
                            "train_step/step_ms": step_metrics["step_ms"],
                            "train_step/num_scenes": step_metrics["num_scenes"],
                            "train_step/num_points_total": step_metrics["num_points_total"],
                            "train_step/num_points_max_scene": step_metrics["num_points_max_scene"],
                            "train_scene/loss": step_metrics["loss_total"],
                            "train_scene/grad_norm": step_metrics["grad_norm"],
                            "train_scene/step_ms": step_metrics["step_ms"],
                        }
                        for g in active_grans:
                            if f"loss_{g}" in step_metrics:
                                wb[f"train_scene/loss_{g}"] = step_metrics[f"loss_{g}"]
                                wb[f"train_scene/loss_mask_bce_{g}"] = step_metrics[f"loss_mask_bce_{g}"]
                                wb[f"train_scene/loss_mask_dice_{g}"] = step_metrics[f"loss_mask_dice_{g}"]
                                wb[f"train_scene/loss_score_{g}"] = step_metrics[f"loss_score_{g}"]
                        if "loss_aux" in step_metrics:
                            wb["train_scene/loss_aux"] = step_metrics["loss_aux"]
                        wandb.log(wb)

            if profile_this_step:
                profile_rows.append({
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "rank": self.rank,
                    "scene_ids": scene_ids_local,
                    "num_scenes": len(scene_ids_local),
                    "num_points_total": num_points_total,
                    "num_points_max_scene": num_points_max_scene,
                    "data_wait_ms": data_wait_ms,
                    "target_build_ms": target_build_ms,
                    "forward_ms": forward_ms,
                    "loss_ms": loss_ms,
                    "backward_ms": backward_ms,
                    "optim_ms": optim_ms,
                    "metrics_sync_ms": metrics_sync_ms,
                    "step_total_ms": step_total_ms,
                })

        self._finalize_train_timing(epoch, profile_rows)

        # ── epoch summary ──
        loss_sum = sum(scene_losses)
        loss_count = len(scene_losses)
        loss_min = min(scene_losses) if scene_losses else 0.0
        loss_max = max(scene_losses) if scene_losses else 0.0
        if self.distributed:
            sum_count = torch.tensor(
                [loss_sum, loss_count],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)
            loss_sum = float(sum_count[0].item())
            loss_count = int(sum_count[1].item())

            min_tensor = torch.tensor(loss_min, device=self.device, dtype=torch.float64)
            max_tensor = torch.tensor(loss_max, device=self.device, dtype=torch.float64)
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            loss_min = float(min_tensor.item())
            loss_max = float(max_tensor.item())

        epoch_metrics = {
            "epoch": epoch,
            "loss_mean": loss_sum / max(loss_count, 1),
            "loss_min": loss_min,
            "loss_max": loss_max,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        if self.is_main_process:
            log.info(
                "Epoch %d/%d  loss_mean=%.4f  loss_range=[%.4f, %.4f]  lr=%.2e",
                epoch, self.max_epochs,
                epoch_metrics["loss_mean"],
                epoch_metrics["loss_min"],
                epoch_metrics["loss_max"],
                epoch_metrics["lr"],
            )
            self._log_row({"epoch_summary": True, **epoch_metrics})

            if _wandb_active():
                wandb.log({
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "train/loss_mean": epoch_metrics["loss_mean"],
                    "train/loss_min": epoch_metrics["loss_min"],
                    "train/loss_max": epoch_metrics["loss_max"],
                    "train/lr": epoch_metrics["lr"],
                })

        return epoch_metrics

    # ------------------------------------------------------------------ #

    def _validate(self, epoch: int) -> dict[str, Any]:
        """Run full evaluation on all validation scenes."""
        val_result = self._evaluate_dataset(
            self.val_dataset,
            epoch=epoch,
            stage_name="Validation",
        )
        if not self.is_main_process:
            return {}

        agg = val_result["aggregate"]
        real_ap50_bits = [
            f"{k.removeprefix('real_AP50_mean_')}={agg[k]:.3f}"
            for k in sorted(agg.keys())
            if k.startswith("real_AP50_mean_") and isinstance(agg.get(k), (int, float))
        ]
        real_ap50_str = ", ".join(real_ap50_bits) if real_ap50_bits else "n/a"
        log.info(
            "  [val epoch %d] loss=%.4f  pseudo_AP50=%.3f  real_AP50=(%s)  mIoU=%.3f",
            epoch,
            agg["loss_mean"],
            agg["pseudo_AP50_mean"],
            real_ap50_str,
            agg["matched_mean_iou_mean"],
        )

        self._log_row({
            "epoch": epoch,
            "validation": True,
            **{f"val_{k}": v for k, v in agg.items() if isinstance(v, (int, float))},
        })

        if _wandb_active():
            wb: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.global_step,
            }
            for k, v in agg.items():
                if isinstance(v, (int, float)):
                    wb[f"val/{k}"] = v

            for scene_id, scene_data in val_result["per_scene"].items():
                eval_data = scene_data.get("eval", {})
                for g in self.granularities:
                    g_eval = eval_data.get(g, {})
                    pseudo = g_eval.get("pseudo_gt", {})
                    if isinstance(pseudo, dict) and "AP25" in pseudo:
                        wb[f"val_scene/{scene_id}/pseudo_AP25_{g}"] = pseudo["AP25"]
                        wb[f"val_scene/{scene_id}/pseudo_AP50_{g}"] = pseudo["AP50"]
                        wb[f"val_scene/{scene_id}/pseudo_NMI_{g}"] = pseudo.get("NMI", 0)
                        wb[f"val_scene/{scene_id}/pseudo_ARI_{g}"] = pseudo.get("ARI", 0)
                    real = g_eval.get("real_gt", {})
                    if isinstance(real, dict) and "AP25" in real:
                        wb[f"val_scene/{scene_id}/real_AP25_{g}"] = real["AP25"]
                        wb[f"val_scene/{scene_id}/real_AP50_{g}"] = real["AP50"]

            wandb.log(wb)

        return val_result

    # ------------------------------------------------------------------ #

    def _train_eval(self, epoch: int) -> dict[str, Any]:
        """Evaluate metrics on a subset of training scenes."""
        train_subset = self._make_train_eval_subset(epoch)
        train_result = self._evaluate_dataset(
            train_subset,
            epoch=epoch,
            stage_name="Train-eval",
        )
        if not self.is_main_process:
            return {}

        agg = train_result["aggregate"]
        real_ap50_bits = [
            f"{k.removeprefix('real_AP50_mean_')}={agg[k]:.3f}"
            for k in sorted(agg.keys())
            if k.startswith("real_AP50_mean_") and isinstance(agg.get(k), (int, float))
        ]
        real_ap50_str = ", ".join(real_ap50_bits) if real_ap50_bits else "n/a"
        log.info(
            "  [train-eval epoch %d] loss=%.4f  "
            "pseudo(AP25=%.3f, AP50=%.3f, NMI=%.4f, ARI=%.4f)  "
            "real(AP50=(%s))  mIoU=%.3f",
            epoch,
            agg["loss_mean"],
            agg["pseudo_AP25_mean"],
            agg["pseudo_AP50_mean"],
            agg["pseudo_NMI_mean"],
            agg["pseudo_ARI_mean"],
            real_ap50_str,
            agg["matched_mean_iou_mean"],
        )

        self._log_row({
            "epoch": epoch,
            "train_eval": True,
            **{f"train_eval_{k}": v for k, v in agg.items() if isinstance(v, (int, float))},
        })

        if _wandb_active():
            wb: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.global_step,
            }
            for k, v in agg.items():
                if isinstance(v, (int, float)):
                    wb[f"train_eval/{k}"] = v

            for scene_id, scene_data in train_result["per_scene"].items():
                eval_data = scene_data.get("eval", {})
                for g in self.granularities:
                    g_eval = eval_data.get(g, {})
                    pseudo = g_eval.get("pseudo_gt", {})
                    if isinstance(pseudo, dict) and "AP25" in pseudo:
                        wb[f"train_eval_scene/{scene_id}/pseudo_AP25_{g}"] = pseudo["AP25"]
                        wb[f"train_eval_scene/{scene_id}/pseudo_AP50_{g}"] = pseudo["AP50"]
                        wb[f"train_eval_scene/{scene_id}/pseudo_NMI_{g}"] = pseudo.get("NMI", 0)
                        wb[f"train_eval_scene/{scene_id}/pseudo_ARI_{g}"] = pseudo.get("ARI", 0)
                    real = g_eval.get("real_gt", {})
                    if isinstance(real, dict) and "AP25" in real:
                        wb[f"train_eval_scene/{scene_id}/real_AP25_{g}"] = real["AP25"]
                        wb[f"train_eval_scene/{scene_id}/real_AP50_{g}"] = real["AP50"]

            wandb.log(wb)

        return train_result

    def train(self) -> dict[str, Any]:
        """Run the full training loop.  Returns final metrics dict."""
        if self.is_main_process:
            log.info(
                "Starting training: %d epochs, %d train scenes, %d val scenes, "
                "%d granularities (%s)",
                self.max_epochs,
                len(self.train_dataset),
                len(self.val_dataset),
                len(self.granularities),
                ", ".join(self.granularities),
            )

        t_start = time.time()
        epoch_times: list[float] = []
        last_val_metrics: dict[str, Any] = {}

        start_epoch = self.current_epoch + 1
        if start_epoch > self.max_epochs:
            if self.is_main_process:
                log.info(
                    "Checkpoint epoch %d already reached max_epochs=%d; nothing to train.",
                    self.current_epoch, self.max_epochs,
                )
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()

            self._train_one_epoch(epoch)
            self.scheduler.step()

            epoch_times.append(time.time() - t_epoch)

            if epoch % self.eval_every_epochs == 0 or epoch == self.max_epochs:
                self._dist_barrier()
                val_metrics = self._validate(epoch)
                if self.is_main_process:
                    last_val_metrics = val_metrics

                    ap50 = val_metrics["aggregate"].get("pseudo_AP50_mean", 0.0)
                    if ap50 > self.best_val_metric:
                        self.best_val_metric = ap50
                        self.best_epoch = epoch
                        self.best_val_metrics = val_metrics
                        self._save_checkpoint("best")
                        log.info(
                            "  New best val pseudo AP50: %.4f at epoch %d",
                            ap50, epoch,
                        )
                self._dist_barrier()

            if (
                self.train_eval_every_epochs is not None
                and (epoch % self.train_eval_every_epochs == 0 or epoch == self.max_epochs)
            ):
                self._dist_barrier()
                self._train_eval(epoch)
                self._dist_barrier()

            if epoch % self.save_every_epochs == 0:
                self._dist_barrier()
                if self.is_main_process:
                    self._save_checkpoint("last")
                self._dist_barrier()

        self._dist_barrier()
        if self.is_main_process:
            self._save_checkpoint("last")
        self._dist_barrier()

        total_time = time.time() - t_start
        if self.is_main_process:
            log.info(
                "Training done: %d epochs in %.1fs (%.1fs/epoch)",
                self.max_epochs, total_time,
                total_time / max(self.max_epochs, 1),
            )

        return {
            "final_val_metrics": last_val_metrics,
            "best_val_metrics": self.best_val_metrics,
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "total_training_time_s": total_time,
            "per_epoch_time_s": epoch_times,
        }
