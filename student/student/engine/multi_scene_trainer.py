"""Epoch-based trainer over multiple scenes with validation.

Replaces :class:`SingleSceneTrainer` for multi-scene experiments.
Reuses existing criterion, evaluator, and metric functions — only
the training loop structure changes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from student.data.multi_scene_dataset import MultiSceneDataset
from student.data.target_builder import build_instance_targets_multi
from student.engine.multi_scene_evaluator import evaluate_multi_scene
from student.losses.mask_set_loss import MultiGranCriterion

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


def _wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


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
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        max_epochs: int = 50,
        eval_every_epochs: int = 5,
        save_every_epochs: int = 10,
        output_dir: Path | str,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        min_points_per_proposal: int = 30,
        eval_benchmark: str = "scannet200",
        min_instance_points: int = 10,
        warmup_epochs: int = 5,
        granularities: tuple[str, ...] = ("g02", "g05", "g08"),
        config: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.lr = lr
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.eval_every_epochs = eval_every_epochs
        self.save_every_epochs = save_every_epochs
        self.output_dir = Path(output_dir)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_points_per_proposal = min_points_per_proposal
        self.eval_benchmark = eval_benchmark
        self.min_instance_points = min_instance_points
        self.granularities = granularities
        self.config = config

        self.model = model.to(device)
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Uniform LR — training from scratch, no pretrained backbone weights
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Linear warmup → cosine annealing
        if warmup_epochs > 0 and max_epochs > warmup_epochs:
            warmup_sched = LinearLR(
                self.optimizer, start_factor=0.001, total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                self.optimizer, T_max=max_epochs - warmup_epochs,
            )
            self.scheduler: torch.optim.lr_scheduler.LRScheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(max_epochs, 1),
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch[0],
            num_workers=0,
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

    def _clear_backbone_cache(self) -> None:
        """Invalidate LitePT voxelization cache before processing a new scene."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "_cached_voxelization"):
            backbone._cached_voxelization = None

    def _log_row(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _save_checkpoint(self, tag: str) -> None:
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_metric": self.best_val_metric,
                "config": self.config,
            },
            path,
        )
        log.info("  Checkpoint saved: %s", path)

    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """Train on all scenes once.  Returns epoch-level metrics."""
        self.model.train()
        scene_losses: list[float] = []

        for sample in self.train_loader:
            self.global_step += 1
            t0 = time.time()

            scene_id = sample["scene_id"]
            points = sample["points"].to(self.device)
            features = sample["features"].to(self.device)

            targets_by_gran = build_instance_targets_multi(
                sample["labels_by_granularity"],
                sample["supervision_mask"],
                min_instance_points=self.min_instance_points,
            )

            self._clear_backbone_cache()
            self.optimizer.zero_grad()

            pred = self.model(points, features)
            loss_result = self.criterion(pred, targets_by_gran)

            loss_result["loss_total"].backward()
            grad_norm = clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm,
            ).item()
            self.optimizer.step()

            loss_total = loss_result["loss_total"].item()
            step_ms = (time.time() - t0) * 1000
            scene_losses.append(loss_total)

            per_head_str = "  ".join(
                f"{g}={loss_result['heads'][g]['loss_total'].item():.4f}"
                for g in self.granularities
            )
            log.info(
                "  epoch %d  step %d  scene=%s  loss=%.4f  [%s]  gnorm=%.3f  %.0fms",
                epoch, self.global_step, scene_id, loss_total,
                per_head_str, grad_norm, step_ms,
            )

            row: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.global_step,
                "scene_id": scene_id,
                "loss_total": loss_total,
                "grad_norm": grad_norm,
                "step_ms": step_ms,
            }
            for g in self.granularities:
                ld_g = loss_result["heads"][g]
                row[f"loss_{g}"] = ld_g["loss_total"].item()
                row[f"loss_mask_bce_{g}"] = ld_g["loss_mask_bce"].item()
                row[f"loss_mask_dice_{g}"] = ld_g["loss_mask_dice"].item()
                row[f"loss_score_{g}"] = ld_g["loss_score"].item()
            self._log_row(row)

            if _wandb_active():
                wb: dict[str, Any] = {
                    "global_step": self.global_step,
                    "epoch": epoch,
                    "train_scene/loss": loss_total,
                    "train_scene/grad_norm": grad_norm,
                    "train_scene/step_ms": step_ms,
                }
                for g in self.granularities:
                    ld_g = loss_result["heads"][g]
                    wb[f"train_scene/loss_{g}"] = ld_g["loss_total"].item()
                    wb[f"train_scene/loss_mask_bce_{g}"] = ld_g["loss_mask_bce"].item()
                    wb[f"train_scene/loss_mask_dice_{g}"] = ld_g["loss_mask_dice"].item()
                    wb[f"train_scene/loss_score_{g}"] = ld_g["loss_score"].item()
                if "loss_aux" in loss_result:
                    wb["train_scene/loss_aux"] = loss_result["loss_aux"].item()
                wandb.log(wb)

        # ── epoch summary ──
        epoch_metrics = {
            "epoch": epoch,
            "loss_mean": sum(scene_losses) / len(scene_losses),
            "loss_min": min(scene_losses),
            "loss_max": max(scene_losses),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
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
        log.info("Validation at epoch %d ...", epoch)

        val_result = evaluate_multi_scene(
            model=self.model,
            dataset=self.val_dataset,
            criterion=self.criterion,
            device=self.device,
            granularities=self.granularities,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            min_points=self.min_points_per_proposal,
            eval_benchmark=self.eval_benchmark,
            min_instance_points=self.min_instance_points,
        )

        self.model.train()

        agg = val_result["aggregate"]
        log.info(
            "  [val epoch %d] loss=%.4f  pseudo_AP50=%.3f  real_AP50=%.3f  mIoU=%.3f",
            epoch,
            agg["loss_mean"],
            agg["pseudo_AP50_mean"],
            agg["real_AP50_mean"],
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

    def train(self) -> dict[str, Any]:
        """Run the full training loop.  Returns final metrics dict."""
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

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()

            self._train_one_epoch(epoch)
            self.scheduler.step()

            epoch_times.append(time.time() - t_epoch)

            if epoch % self.eval_every_epochs == 0 or epoch == self.max_epochs:
                val_metrics = self._validate(epoch)
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

            if epoch % self.save_every_epochs == 0:
                self._save_checkpoint("last")

        self._save_checkpoint("last")

        total_time = time.time() - t_start
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
