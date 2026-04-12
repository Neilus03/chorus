"""Multi-scene evaluation: iterate scenes, call per-scene evaluators, aggregate.

Wraps existing :func:`evaluate_student_predictions_multi` and
:func:`compute_pseudo_metrics_multi` with cross-scene iteration and
metric aggregation.
"""

from __future__ import annotations

import logging
import math
import time
from numbers import Real
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Subset

from student.data.multi_scene_dataset import MultiSceneDataset
from student.data.target_builder import build_instance_targets_multi
from student.engine.evaluator import evaluate_student_predictions_multi
from student.engine.fragment_merge_eval import evaluate_scene_fragment_merge
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


def _append_real(values: list[float], value: Any) -> None:
    if isinstance(value, Real):
        values.append(float(value))


def aggregate_multi_scene_results(
    per_scene: dict[str, dict[str, Any]],
    *,
    granularities: tuple[str, ...],
) -> dict[str, Any]:
    """Recompute aggregate metrics from per-scene evaluation outputs."""
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

    for scene_data in per_scene.values():
        _append_real(all_losses, scene_data.get("loss"))

        pseudo_metrics = scene_data.get("pseudo_metrics", {})
        if isinstance(pseudo_metrics, dict):
            for g in granularities:
                g_metrics = pseudo_metrics.get(g, {})
                if isinstance(g_metrics, dict):
                    _append_real(
                        matched_iou_by_gran[g],
                        g_metrics.get("matched_mean_iou"),
                    )

        eval_result = scene_data.get("eval", {})
        if not isinstance(eval_result, dict):
            continue

        for g in granularities:
            g_eval = eval_result.get(g, {})
            if not isinstance(g_eval, dict):
                continue

            pseudo = g_eval.get("pseudo_gt", {})
            if isinstance(pseudo, dict) and "AP25" in pseudo and "AP50" in pseudo:
                _append_real(pseudo_ap25_all, pseudo.get("AP25"))
                _append_real(pseudo_ap50_all, pseudo.get("AP50"))
                _append_real(pseudo_nmi_all, pseudo.get("NMI"))
                _append_real(pseudo_ari_all, pseudo.get("ARI"))

            real = g_eval.get("real_gt", {})
            if isinstance(real, dict) and "AP25" in real and "AP50" in real:
                _append_real(real_ap25_all, real.get("AP25"))
                _append_real(real_ap50_all, real.get("AP50"))
                _append_real(real_nmi_all, real.get("NMI"))
                _append_real(real_ari_all, real.get("ARI"))

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

    return aggregate


def _base_dataset_and_index(
    dataset: MultiSceneDataset | Subset,
    idx: int,
) -> tuple[MultiSceneDataset, int]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if not isinstance(base, MultiSceneDataset):
            raise TypeError("Subset base must be MultiSceneDataset for fragment eval")
        return base, int(dataset.indices[idx])
    if not isinstance(dataset, MultiSceneDataset):
        raise TypeError("dataset must be MultiSceneDataset or Subset thereof")
    return dataset, idx


def evaluate_multi_scene(
    model: nn.Module,
    dataset: MultiSceneDataset | Subset,
    criterion: nn.Module,
    *,
    device: str,
    granularities: tuple[str, ...],
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmark: str = "scannet200",
    min_instance_points: int = 10,
    dense_instance_ids: bool = False,
    fragment_merge_eval: bool = False,
    fragment_merge_num: int = 4,
    fragment_merge_point_max: int | None = None,
    fragment_merge_seed: int = 0,
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
    frag_pm = fragment_merge_point_max if fragment_merge_point_max is not None else 50_000

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
            dense_instance_ids=dense_instance_ids,
        )

        _clear_backbone_cache(model)

        with torch.no_grad():
            pred = model(points, features)
            loss_result = criterion(pred, targets_by_gran)

        loss_val = loss_result["loss_total"].item()

        with torch.no_grad():
            pseudo_metrics = compute_pseudo_metrics_multi(
                pred, targets_by_gran, loss_result,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )

        eval_result: dict[str, Any] = {}
        try:
            if fragment_merge_eval:
                base_ds, real_idx = _base_dataset_and_index(dataset, idx)
                eval_result = evaluate_scene_fragment_merge(
                    model,
                    base_ds,
                    real_idx,
                    device=device,
                    granularities=granularities,
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    min_points=min_points,
                    eval_benchmark=eval_benchmark,
                    fragment_num=fragment_merge_num,
                    fragment_point_max=frag_pm,
                    fragment_seed=fragment_merge_seed,
                )
            else:
                vi = sample.get("vertex_indices")
                eval_result = evaluate_student_predictions_multi(
                    pred, targets_by_gran,
                    scene_dir=sample["scene_dir"],
                    scene_id=scene_id,
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    min_points=min_points,
                    eval_benchmark=eval_benchmark,
                    vertex_indices=vi,
                )

        except Exception as e:
            log.warning("  Full eval failed for %s: %s", scene_id, e)

        elapsed = time.time() - t0
        log.info("  [eval %s] loss=%.4f  %.1fs", scene_id, loss_val, elapsed)

        per_scene[scene_id] = {
            "loss": loss_val,
            "pseudo_metrics": pseudo_metrics,
            "eval": eval_result,
        }

    return {
        "per_scene": per_scene,
        "aggregate": aggregate_multi_scene_results(
            per_scene,
            granularities=granularities,
        ),
    }
