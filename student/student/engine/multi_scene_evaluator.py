"""Multi-scene evaluation: iterate scenes, call per-scene evaluators, aggregate.

Wraps existing :func:`evaluate_student_predictions_multi` and
:func:`compute_pseudo_metrics_multi` with cross-scene iteration and
metric aggregation.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn as nn

from student.data.multi_scene_dataset import MultiSceneDataset
from student.data.target_builder import build_instance_targets_multi
from student.engine.evaluator import evaluate_student_predictions_multi
from student.metrics.pseudo_metrics import compute_pseudo_metrics_multi

log = logging.getLogger(__name__)


def _clear_backbone_cache(model: nn.Module) -> None:
    """Invalidate LitePT voxelization cache for multi-scene processing."""
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _safe_mean(values: list[float]) -> float:
    """Mean that filters NaN values; returns 0.0 for empty input."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else 0.0


def evaluate_multi_scene(
    model: nn.Module,
    dataset: MultiSceneDataset,
    criterion: nn.Module,
    *,
    device: str,
    granularities: tuple[str, ...],
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmark: str = "scannet200",
    min_instance_points: int = 10,
) -> dict[str, Any]:
    """Evaluate model on all scenes in a dataset.

    Sets model to eval mode.  Caller is responsible for restoring
    train mode afterward.

    Parameters
    ----------
    model:
        Student model (``StudentInstanceSegModel``).
    dataset:
        ``MultiSceneDataset`` to evaluate on.
    criterion:
        ``MultiGranCriterion`` for computing validation loss.
    device:
        Target CUDA device string.
    granularities:
        Dot-free granularity keys.
    score_threshold / mask_threshold / min_points:
        Proposal extraction thresholds.
    eval_benchmark:
        ScanNet GT evaluation benchmark name.
    min_instance_points:
        Minimum instance size for target building.

    Returns
    -------
    Dict with ``per_scene`` and ``aggregate`` sub-dicts.
    """
    model.eval()

    # Hack: Force BatchNorm to compute stats on the fly
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            module.train()
    # --------------------------

    per_scene: dict[str, dict[str, Any]] = {}

    all_losses: list[float] = []
    pseudo_ap25_all: list[float] = []
    pseudo_ap50_all: list[float] = []
    real_ap25_all: list[float] = []
    real_ap50_all: list[float] = []
    pseudo_nmi_all: list[float] = []
    pseudo_ari_all: list[float] = []
    real_nmi_all: list[float] = []
    real_ari_all: list[float] = []
    matched_iou_by_gran: dict[str, list[float]] = {g: [] for g in granularities}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        scene_id = sample["scene_id"]
        t0 = time.time()

        points = sample["points"].to(device)
        features = sample["features"].to(device)

        targets_by_gran = build_instance_targets_multi(
            sample["labels_by_granularity"],
            sample["supervision_mask"],
            min_instance_points=min_instance_points,
        )

        _clear_backbone_cache(model)

        with torch.no_grad():
            pred = model(points, features)
            loss_result = criterion(pred, targets_by_gran)

        loss_val = loss_result["loss_total"].item()
        all_losses.append(loss_val)

        with torch.no_grad():
            pseudo_metrics = compute_pseudo_metrics_multi(
                pred, targets_by_gran, loss_result,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )

        for g in granularities:
            matched_iou_by_gran[g].append(
                pseudo_metrics[g]["matched_mean_iou"]
            )

        eval_result: dict[str, Any] = {}
        try:
            eval_result = evaluate_student_predictions_multi(
                pred, targets_by_gran,
                scene_dir=sample["scene_dir"],
                scene_id=scene_id,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_points=min_points,
                eval_benchmark=eval_benchmark,
            )

            for g in granularities:
                g_eval = eval_result.get(g, {})
                pseudo = g_eval.get("pseudo_gt", {})
                if isinstance(pseudo, dict) and "AP25" in pseudo:
                    pseudo_ap25_all.append(pseudo["AP25"])
                    pseudo_ap50_all.append(pseudo["AP50"])
                    pseudo_nmi_all.append(pseudo.get("NMI", float("nan")))
                    pseudo_ari_all.append(pseudo.get("ARI", float("nan")))
                real = g_eval.get("real_gt", {})
                if isinstance(real, dict) and "AP25" in real:
                    real_ap25_all.append(real["AP25"])
                    real_ap50_all.append(real["AP50"])
                    real_nmi_all.append(real.get("NMI", float("nan")))
                    real_ari_all.append(real.get("ARI", float("nan")))

        except Exception as e:
            log.warning("  Full eval failed for %s: %s", scene_id, e)

        elapsed = time.time() - t0
        log.info("  [eval %s] loss=%.4f  %.1fs", scene_id, loss_val, elapsed)

        per_scene[scene_id] = {
            "loss": loss_val,
            "pseudo_metrics": pseudo_metrics,
            "eval": eval_result,
        }

    # ── aggregate across scenes ──
    aggregate: dict[str, Any] = {
        "loss_mean": _safe_mean(all_losses),
        "pseudo_AP25_mean": _safe_mean(pseudo_ap25_all),
        "pseudo_AP50_mean": _safe_mean(pseudo_ap50_all),
        "real_AP25_mean": _safe_mean(real_ap25_all),
        "real_AP50_mean": _safe_mean(real_ap50_all),
        "pseudo_NMI_mean": _safe_mean(pseudo_nmi_all),
        "pseudo_ARI_mean": _safe_mean(pseudo_ari_all),
        "real_NMI_mean": _safe_mean(real_nmi_all),
        "real_ARI_mean": _safe_mean(real_ari_all),
    }

    all_iou_values: list[float] = []
    for g in granularities:
        g_mean = _safe_mean(matched_iou_by_gran[g])
        aggregate[f"matched_mean_iou_{g}_mean"] = g_mean
        all_iou_values.extend(matched_iou_by_gran[g])
    aggregate["matched_mean_iou_mean"] = _safe_mean(all_iou_values)

    return {"per_scene": per_scene, "aggregate": aggregate}
