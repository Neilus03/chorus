"""Minimal single-scene overfit trainer (multi-granularity).

Answers one question: can the model overfit one scene across all
granularity heads simultaneously?

No dataloader, no batching, no distributed, no evaluator class.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from student.data.target_builder import InstanceTargets
from student.losses.mask_set_loss import MultiGranCriterion
from student.metrics.pseudo_metrics import (
    compute_pseudo_metrics_multi,
    format_pseudo_metrics,
)
from student.engine.evaluator import evaluate_student_predictions_multi

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


def _wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


class SingleSceneTrainer:
    """Train on a single sample repeatedly with multi-granularity heads.

    Parameters
    ----------
    model:
        The full ``StudentInstanceSegModel`` with multi-head decoder.
    criterion:
        ``MultiGranCriterion`` instance.
    sample:
        Dict from ``MultiGranSceneDataset[0]``.
    targets_by_granularity:
        Dict mapping granularity keys to ``InstanceTargets``.
    device:
        CUDA device string.
    lr / weight_decay / grad_clip_norm:
        Optimizer config.
    max_steps:
        Total training iterations.
    log_every / eval_every / save_every:
        Logging, metrics, and checkpoint intervals.
    output_dir:
        Where to write checkpoints, logs, and metrics.
    score_threshold / mask_threshold:
        For pseudo-metrics query counting and mask binarization.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: MultiGranCriterion,
        sample: dict[str, Any],
        targets_by_granularity: dict[str, InstanceTargets],
        *,
        device: str = "cuda:0",
        lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        max_steps: int = 2000,
        log_every: int = 20,
        eval_every: int = 100,
        save_every: int = 200,
        output_dir: Path | str = "student_runs",
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        min_points_per_proposal: int = 30,
        eval_benchmark: str = "scannet200",
        full_eval_every: int | None = None,
    ) -> None:
        self.device = device
        self.max_steps = max_steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.grad_clip_norm = grad_clip_norm
        self.output_dir = Path(output_dir)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_points_per_proposal = min_points_per_proposal
        self.eval_benchmark = eval_benchmark
        self.full_eval_every = full_eval_every or save_every

        self.model = model.to(device)
        self.criterion = criterion
        self.targets_by_granularity = targets_by_granularity
        self.granularities = list(targets_by_granularity.keys())

        self.points = sample["points"].to(device)
        self.features = sample["features"].to(device)
        self.sample = sample

        if hasattr(model, "parameter_groups"):
            param_groups = [
                {
                    "params": pg["params"],
                    "lr": lr * pg["lr_scale"],
                }
                for pg in model.parameter_groups(backbone_lr_scale)
            ]
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=lr, weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        self.lr = lr

        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"

        self.best_iou = -1.0
        self.step = 0

    # ------------------------------------------------------------------ #

    def _log_row(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _save_checkpoint(self, tag: str) -> None:
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_iou": self.best_iou,
            },
            path,
        )
        log.info("  checkpoint saved: %s", path)

    # ------------------------------------------------------------------ #

    def _save_ply_for_head(
        self,
        pred: dict,
        loss_result: dict,
        granularity: str,
        tag: str,
    ) -> None:
        """Save predicted + GT instance-colored mesh PLY for one head."""
        from student.engine.vis import (
            save_prediction_ply, save_gt_ply, _resolve_source_mesh,
        )

        eval_dir = self.output_dir / "eval"
        eval_dir.mkdir(exist_ok=True)

        source_mesh = _resolve_source_mesh(
            self.sample["scene_dir"],
            self.sample.get("scene_meta", {}),
        )

        head_pred = pred["heads"][granularity]
        head_loss = loss_result["heads"][granularity]

        save_prediction_ply(
            mask_logits=head_pred["mask_logits"].detach().cpu(),
            score_logits=head_pred["score_logits"].detach().cpu(),
            matched_pred_idx=head_loss["matched_pred_indices"],
            source_mesh=source_mesh,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            path=eval_dir / f"student_pred_{granularity}_{tag}.ply",
        )

        save_gt_ply(
            targets=self.targets_by_granularity[granularity],
            source_mesh=source_mesh,
            path=eval_dir / f"gt_instances_{granularity}_{tag}.ply",
        )

    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, Any]:
        """Run the full training loop. Returns final metrics dict."""
        log.info(
            "Starting training: %d steps, %d heads (%s)",
            self.max_steps, len(self.granularities),
            ", ".join(self.granularities),
        )
        self.model.train()
        t_start = time.time()

        final_metrics: dict[str, Any] = {}
        last_full_eval: dict[str, Any] | None = None

        for self.step in range(1, self.max_steps + 1):
            t0 = time.time()

            # forward
            self.optimizer.zero_grad()
            pred = self.model(self.points, self.features)
            loss_result = self.criterion(pred, self.targets_by_granularity)

            # backward
            loss_result["loss_total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm,
            ).item()
            self.optimizer.step()

            step_ms = (time.time() - t0) * 1000
            loss_total = loss_result["loss_total"].item()

            # ── wandb: every step ──
            if _wandb_active():
                wb: dict[str, Any] = {
                    "step": self.step,
                    "train/loss_total": loss_total,
                    "train/grad_norm": grad_norm,
                    "train/step_ms": step_ms,
                    "train/lr": self.lr,
                }
                for g in self.granularities:
                    ld_g = loss_result["heads"][g]
                    wb[f"train/loss_{g}"] = ld_g["loss_total"].item()
                    wb[f"train/loss_mask_bce_{g}"] = ld_g["loss_mask_bce"].item()
                    wb[f"train/loss_mask_dice_{g}"] = ld_g["loss_mask_dice"].item()
                    wb[f"train/loss_score_{g}"] = ld_g["loss_score"].item()
                if "loss_aux" in loss_result:
                    wb["train/loss_aux"] = loss_result["loss_aux"].item()
                wandb.log(wb)

            # ── console + jsonl: every N steps ──
            if self.step % self.log_every == 0 or self.step == 1:
                per_head_str = "  ".join(
                    f"{g}={loss_result['heads'][g]['loss_total'].item():.4f}"
                    for g in self.granularities
                )
                log.info(
                    "step %4d/%d  loss=%.4f  [%s]  gnorm=%.3f  %.0fms",
                    self.step, self.max_steps, loss_total,
                    per_head_str, grad_norm, step_ms,
                )
                row: dict[str, Any] = {
                    "step": self.step,
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

            # ── eval metrics ──
            run_pseudo_eval = (self.step % self.eval_every == 0 or self.step == 1)
            is_last_step = (self.step == self.max_steps)
            run_full_eval = (self.step % self.full_eval_every == 0 or is_last_step)

            if run_pseudo_eval or run_full_eval:
                self.model.eval()
                with torch.no_grad():
                    # Hack: Force BatchNorm to compute stats on the fly
                    for module in self.model.modules():
                        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                            module.train()
                    # --------------------------
                    pred_eval = self.model(self.points, self.features)
                    ld_eval = self.criterion(pred_eval, self.targets_by_granularity)
                self.model.train()

            if run_pseudo_eval:
                with torch.no_grad():
                    metrics_by_g = compute_pseudo_metrics_multi(
                        pred_eval, self.targets_by_granularity, ld_eval,
                        score_threshold=self.score_threshold,
                        mask_threshold=self.mask_threshold,
                    )

                for g in self.granularities:
                    log.info(
                        "  [eval %s] %s", g,
                        format_pseudo_metrics(metrics_by_g[g]),
                    )

                eval_row: dict[str, Any] = {"step": self.step, "eval": True}
                for g in self.granularities:
                    for mk, mv in metrics_by_g[g].items():
                        eval_row[f"{mk}_{g}"] = mv
                self._log_row(eval_row)

                if _wandb_active():
                    wb_eval: dict[str, Any] = {"step": self.step}
                    for g in self.granularities:
                        m = metrics_by_g[g]
                        wb_eval[f"eval/matched_mean_iou_{g}"] = m["matched_mean_iou"]
                        wb_eval[f"eval/matched_median_iou_{g}"] = m["matched_median_iou"]
                        wb_eval[f"eval/iou_gt_0.25_{g}"] = m["matched_iou_gt_0.25"]
                        wb_eval[f"eval/iou_gt_0.50_{g}"] = m["matched_iou_gt_0.50"]
                        wb_eval[f"eval/mean_score_matched_{g}"] = m["mean_score_matched"]
                        wb_eval[f"eval/mean_score_unmatched_{g}"] = m["mean_score_unmatched"]
                        wb_eval[f"eval/score_gap_{g}"] = (
                            m["mean_score_matched"] - m["mean_score_unmatched"]
                        )
                        wb_eval[f"eval/num_queries_active_{g}"] = m["num_queries_above_threshold"]
                    wandb.log(wb_eval)

                avg_iou = sum(
                    metrics_by_g[g]["matched_mean_iou"] for g in self.granularities
                ) / len(self.granularities)
                if avg_iou > self.best_iou:
                    self.best_iou = avg_iou
                    self._save_checkpoint("best_pseudo_iou")

                final_metrics = {
                    f"{mk}_{g}": mv
                    for g in self.granularities
                    for mk, mv in metrics_by_g[g].items()
                }

            if run_full_eval:
                try:
                    last_full_eval = evaluate_student_predictions_multi(
                        pred_eval, self.targets_by_granularity,
                        scene_dir=self.sample["scene_dir"],
                        scene_id=self.sample["scene_id"],
                        score_threshold=self.score_threshold,
                        mask_threshold=self.mask_threshold,
                        min_points=self.min_points_per_proposal,
                        eval_benchmark=self.eval_benchmark,
                    )

                    if _wandb_active():
                        wb_full: dict[str, Any] = {"step": self.step}
                        for g, eval_g in last_full_eval.items():
                            pseudo = eval_g.get("pseudo_gt", {})
                            if isinstance(pseudo, dict) and "AP25" in pseudo:
                                wb_full[f"eval/pseudo_AP25_{g}"] = pseudo["AP25"]
                                wb_full[f"eval/pseudo_AP50_{g}"] = pseudo["AP50"]
                                wb_full[f"eval/pseudo_NMI_{g}"] = pseudo.get("NMI", 0)
                                wb_full[f"eval/pseudo_ARI_{g}"] = pseudo.get("ARI", 0)
                            real = eval_g.get("real_gt", {})
                            if isinstance(real, dict) and "AP25" in real:
                                wb_full[f"eval/real_AP25_{g}"] = real["AP25"]
                                wb_full[f"eval/real_AP50_{g}"] = real["AP50"]
                                wb_full[f"eval/real_NMI_{g}"] = real.get("NMI", 0)
                                wb_full[f"eval/real_ARI_{g}"] = real.get("ARI", 0)
                            wb_full[f"eval/num_proposals_{g}"] = eval_g.get("num_proposals", 0)
                        wandb.log(wb_full)

                    self._log_row({"step": self.step, "full_eval": True, **last_full_eval})
                except Exception as e:
                    log.warning("  full eval failed at step %d: %s", self.step, e)

            # ── checkpoint ──
            if self.step % self.save_every == 0:
                self._save_checkpoint("last")

        # ── final save + PLY ──
        self._save_checkpoint("last")

        self.model.eval()
        with torch.no_grad():
            pred_final = self.model(self.points, self.features)
            ld_final = self.criterion(pred_final, self.targets_by_granularity)
        self.model.train()

        for g in self.granularities:
            try:
                self._save_ply_for_head(pred_final, ld_final, g, "final")
                log.info("  PLY files saved for %s to %s/eval/", g, self.output_dir)
            except Exception as e:
                log.warning("  PLY save failed for %s: %s", g, e)

        elapsed = time.time() - t_start
        log.info(
            "Training done: %d steps in %.1fs (%.0f ms/step)",
            self.max_steps, elapsed, elapsed / self.max_steps * 1000,
        )

        final_metrics["total_steps"] = self.max_steps
        final_metrics["best_avg_iou"] = self.best_iou
        final_metrics["training_time_s"] = elapsed
        if last_full_eval is not None:
            final_metrics["evaluation"] = last_full_eval
        return final_metrics
