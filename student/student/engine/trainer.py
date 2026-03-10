"""Minimal single-scene overfit trainer.

Answers one question: can the model overfit one scene?

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
from student.losses.mask_set_loss import MaskSetCriterion
from student.metrics.pseudo_metrics import compute_pseudo_metrics, format_pseudo_metrics
from student.engine.evaluator import evaluate_student_predictions

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


def _wandb_active() -> bool:
    return wandb is not None and wandb.run is not None


class SingleSceneTrainer:
    """Train on a single sample repeatedly.

    Parameters
    ----------
    model:
        The full ``StudentInstanceSegModel``.
    criterion:
        ``MaskSetCriterion`` instance.
    sample:
        Dict from ``SingleSceneTrainingPackDataset[0]``.
    targets:
        ``InstanceTargets`` from ``build_instance_targets``.
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
        criterion: MaskSetCriterion,
        sample: dict[str, Any],
        targets: InstanceTargets,
        *,
        device: str = "cuda:0",
        lr: float = 1e-4,
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
        self.targets = targets

        self.points = sample["points"].to(device)
        self.features = sample["features"].to(device)
        self.sample = sample

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

    def _save_ply(
        self,
        pred: dict[str, torch.Tensor],
        matched_pred_idx: np.ndarray,
        tag: str,
    ) -> None:
        """Save predicted + GT instance-colored mesh PLY files."""
        from student.engine.vis import (
            save_prediction_ply, save_gt_ply, _resolve_source_mesh,
        )

        eval_dir = self.output_dir / "eval"
        eval_dir.mkdir(exist_ok=True)

        source_mesh = _resolve_source_mesh(
            self.sample["scene_dir"],
            self.sample.get("scene_meta", {}),
        )

        save_prediction_ply(
            mask_logits=pred["mask_logits"].detach().cpu(),
            score_logits=pred["score_logits"].detach().cpu(),
            matched_pred_idx=matched_pred_idx,
            source_mesh=source_mesh,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            path=eval_dir / f"student_pred_{tag}.ply",
        )

        save_gt_ply(
            targets=self.targets,
            source_mesh=source_mesh,
            path=eval_dir / f"gt_instances_{tag}.ply",
        )

    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, Any]:
        """Run the full training loop. Returns final metrics dict."""
        log.info("Starting training: %d steps", self.max_steps)
        self.model.train()
        t_start = time.time()

        final_metrics: dict[str, Any] = {}

        for self.step in range(1, self.max_steps + 1):
            t0 = time.time()

            # forward
            self.optimizer.zero_grad()
            pred = self.model(self.points, self.features)
            loss_dict = self.criterion(pred, self.targets)

            # backward
            loss_dict["loss_total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm,
            ).item()
            self.optimizer.step()

            step_ms = (time.time() - t0) * 1000
            loss_total = loss_dict["loss_total"].item()
            loss_bce = loss_dict["loss_mask_bce"].item()
            loss_dice = loss_dict["loss_mask_dice"].item()
            loss_score = loss_dict["loss_score"].item()

            # ── wandb: every step ──
            if _wandb_active():
                wandb.log({
                    "step": self.step,
                    "train/loss_total": loss_total,
                    "train/loss_mask_bce": loss_bce,
                    "train/loss_mask_dice": loss_dice,
                    "train/loss_score": loss_score,
                    "train/grad_norm": grad_norm,
                    "train/step_ms": step_ms,
                    "train/lr": self.lr,
                })

            # ── console + jsonl: every N steps ──
            if self.step % self.log_every == 0 or self.step == 1:
                log.info(
                    "step %4d/%d  loss=%.4f  bce=%.4f  dice=%.4f  score=%.4f  "
                    "gnorm=%.3f  %.0fms",
                    self.step, self.max_steps,
                    loss_total, loss_bce, loss_dice, loss_score,
                    grad_norm, step_ms,
                )
                self._log_row({
                    "step": self.step,
                    "loss_total": loss_total,
                    "loss_mask_bce": loss_bce,
                    "loss_mask_dice": loss_dice,
                    "loss_score": loss_score,
                    "grad_norm": grad_norm,
                    "step_ms": step_ms,
                })

            # ── eval metrics ──
            run_pseudo_eval = (self.step % self.eval_every == 0 or self.step == 1)
            is_last_step = (self.step == self.max_steps)
            run_full_eval = (self.step % self.full_eval_every == 0 or is_last_step)

            if run_pseudo_eval or run_full_eval:
                self.model.eval()
                with torch.no_grad():
                    pred_eval = self.model(self.points, self.features)
                self.model.train()

            if run_pseudo_eval:
                with torch.no_grad():
                    ld_eval = self.criterion(pred_eval, self.targets)
                    metrics = compute_pseudo_metrics(
                        pred_eval, self.targets,
                        ld_eval["matched_pred_indices"],
                        ld_eval["matched_gt_indices"],
                        score_threshold=self.score_threshold,
                        mask_threshold=self.mask_threshold,
                    )

                log.info("  [eval] %s", format_pseudo_metrics(metrics))

                eval_row = {"step": self.step, "eval": True, **metrics}
                self._log_row(eval_row)

                if _wandb_active():
                    wb_metrics = {
                        "eval/matched_mean_iou": metrics["matched_mean_iou"],
                        "eval/matched_median_iou": metrics["matched_median_iou"],
                        "eval/iou_gt_0.25": metrics["matched_iou_gt_0.25"],
                        "eval/iou_gt_0.50": metrics["matched_iou_gt_0.50"],
                        "eval/mean_score_matched": metrics["mean_score_matched"],
                        "eval/mean_score_unmatched": metrics["mean_score_unmatched"],
                        "eval/score_gap": (
                            metrics["mean_score_matched"] - metrics["mean_score_unmatched"]
                        ),
                        "eval/num_queries_active": metrics["num_queries_above_threshold"],
                    }
                    wb_metrics["step"] = self.step
                    wandb.log(wb_metrics)

                if metrics["matched_mean_iou"] > self.best_iou:
                    self.best_iou = metrics["matched_mean_iou"]
                    self._save_checkpoint("best_pseudo_iou")

                final_metrics = metrics

            if run_full_eval:
                try:
                    full_eval = evaluate_student_predictions(
                        pred_eval, self.targets,
                        scene_dir=self.sample["scene_dir"],
                        scene_id=self.sample["scene_id"],
                        score_threshold=self.score_threshold,
                        mask_threshold=self.mask_threshold,
                        min_points=self.min_points_per_proposal,
                        eval_benchmark=self.eval_benchmark,
                    )

                    if _wandb_active():
                        wb_full: dict[str, Any] = {"step": self.step}
                        pseudo = full_eval.get("pseudo_gt", {})
                        if isinstance(pseudo, dict) and "AP25" in pseudo:
                            wb_full["eval/pseudo_AP25"] = pseudo["AP25"]
                            wb_full["eval/pseudo_AP50"] = pseudo["AP50"]
                            wb_full["eval/pseudo_NMI"] = pseudo.get("NMI", 0)
                            wb_full["eval/pseudo_ARI"] = pseudo.get("ARI", 0)
                        real = full_eval.get("real_gt", {})
                        if isinstance(real, dict) and "AP25" in real:
                            wb_full["eval/real_AP25"] = real["AP25"]
                            wb_full["eval/real_AP50"] = real["AP50"]
                            wb_full["eval/real_NMI"] = real.get("NMI", 0)
                            wb_full["eval/real_ARI"] = real.get("ARI", 0)
                        wb_full["eval/num_proposals"] = full_eval.get("num_proposals", 0)
                        wandb.log(wb_full)

                    self._log_row({"step": self.step, "full_eval": True, **full_eval})
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
            ld_final = self.criterion(pred_final, self.targets)
        self.model.train()

        try:
            self._save_ply(pred_final, ld_final["matched_pred_indices"], "final")
            log.info("  PLY files saved to %s/eval/", self.output_dir)
        except Exception as e:
            log.warning("  PLY save failed: %s", e)

        elapsed = time.time() - t_start
        log.info(
            "Training done: %d steps in %.1fs (%.0f ms/step)",
            self.max_steps, elapsed, elapsed / self.max_steps * 1000,
        )

        final_metrics["total_steps"] = self.max_steps
        final_metrics["best_iou"] = self.best_iou
        final_metrics["training_time_s"] = elapsed
        return final_metrics
