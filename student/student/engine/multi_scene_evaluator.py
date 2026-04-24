"""Multi-scene evaluation: iterate scenes, call per-scene evaluators, aggregate.

Wraps existing :func:`evaluate_student_predictions_multi` and
:func:`compute_pseudo_metrics_multi` with cross-scene iteration and
metric aggregation.

Supports both multi-head and continuous decoder models transparently.
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
from student.models.continuous_decoder import ContinuousQueryInstanceDecoder

log = logging.getLogger(__name__)


# ── granularity key → float value ────────────────────────────────────────

_GRAN_KEY_TO_VAL = {
    "g02": 0.2, "g05": 0.5, "g08": 0.8,
    "g01": 0.1, "g03": 0.3, "g04": 0.4,
    "g06": 0.6, "g07": 0.7, "g09": 0.9,
    "g10": 1.0,
}


def _gran_key_to_float(key: str) -> float:
    """Convert granularity key like 'g05' → 0.5."""
    if key in _GRAN_KEY_TO_VAL:
        return _GRAN_KEY_TO_VAL[key]
    return float(key.replace("g0", "0.").replace("g", "0."))


def _is_continuous_model(model: nn.Module) -> bool:
    """Check if model uses a ContinuousQueryInstanceDecoder."""
    # Unwrap DDP / FineTuningWrapper
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "model"):
        m = m.model
    decoder = getattr(m, "decoder", None)
    return isinstance(decoder, ContinuousQueryInstanceDecoder)


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
    pseudo_nmi_all: list[float] = []
    pseudo_ari_all: list[float] = []
    # Real GT metrics can be stored either as a legacy single dict (`real_gt`)
    # or as a dict of benchmark→metrics (`real_gt_by_benchmark`).
    real_ap25_by_bench: dict[str, list[float]] = {}
    real_ap50_by_bench: dict[str, list[float]] = {}
    real_nmi_by_bench: dict[str, list[float]] = {}
    real_ari_by_bench: dict[str, list[float]] = {}
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

            real_by = g_eval.get("real_gt_by_benchmark", None)
            if isinstance(real_by, dict):
                for bench, real in real_by.items():
                    if not isinstance(real, dict):
                        continue
                    if "AP25" in real and "AP50" in real:
                        real_ap25_by_bench.setdefault(str(bench), [])
                        real_ap50_by_bench.setdefault(str(bench), [])
                        real_nmi_by_bench.setdefault(str(bench), [])
                        real_ari_by_bench.setdefault(str(bench), [])
                        _append_real(real_ap25_by_bench[str(bench)], real.get("AP25"))
                        _append_real(real_ap50_by_bench[str(bench)], real.get("AP50"))
                        _append_real(real_nmi_by_bench[str(bench)], real.get("NMI"))
                        _append_real(real_ari_by_bench[str(bench)], real.get("ARI"))
                continue

            # Legacy fallback: single real_gt dict
            real = g_eval.get("real_gt", {})
            if isinstance(real, dict) and "AP25" in real and "AP50" in real:
                bench = str(real.get("eval_benchmark", "unknown"))
                real_ap25_by_bench.setdefault(bench, [])
                real_ap50_by_bench.setdefault(bench, [])
                real_nmi_by_bench.setdefault(bench, [])
                real_ari_by_bench.setdefault(bench, [])
                _append_real(real_ap25_by_bench[bench], real.get("AP25"))
                _append_real(real_ap50_by_bench[bench], real.get("AP50"))
                _append_real(real_nmi_by_bench[bench], real.get("NMI"))
                _append_real(real_ari_by_bench[bench], real.get("ARI"))

    aggregate: dict[str, Any] = {
        "loss_mean": _safe_mean(all_losses),
        "pseudo_AP25_mean": _safe_mean(pseudo_ap25_all),
        "pseudo_AP50_mean": _safe_mean(pseudo_ap50_all),
        "pseudo_NMI_mean": _safe_mean(pseudo_nmi_all),
        "pseudo_ARI_mean": _safe_mean(pseudo_ari_all),
    }

    # Per-benchmark real GT aggregates (no primary)
    for bench in sorted(real_ap25_by_bench):
        aggregate[f"real_AP25_mean_{bench}"] = _safe_mean(real_ap25_by_bench[bench])
        aggregate[f"real_AP50_mean_{bench}"] = _safe_mean(real_ap50_by_bench.get(bench, []))
        aggregate[f"real_NMI_mean_{bench}"] = _safe_mean(real_nmi_by_bench.get(bench, []))
        aggregate[f"real_ARI_mean_{bench}"] = _safe_mean(real_ari_by_bench.get(bench, []))

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


# ── continuous decoder: per-granularity forward + assembly ───────────────


def _evaluate_continuous_scene(
    model: nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    granularities: tuple[str, ...],
    criterion: nn.Module,
    targets_by_gran: dict,
) -> tuple[dict, dict]:
    """Run continuous decoder at each granularity, assemble multi-head output.

    Returns (pred_multihead, loss_result_multihead) in the format expected
    by ``evaluate_student_predictions_multi`` and ``compute_pseudo_metrics_multi``.
    """
    heads_pred: dict[str, dict] = {}
    heads_loss: dict[str, dict] = {}
    loss_total = torch.tensor(0.0, device=points.device)
    point_embed = None

    for g in granularities:
        g_val = _gran_key_to_float(g)
        flat_pred = model(points, features, target_g=g_val)
        heads_pred[g] = {
            "mask_logits": flat_pred["mask_logits"],
            "score_logits": flat_pred["score_logits"],
            "query_embed": flat_pred.get("query_embed"),
        }
        if point_embed is None:
            point_embed = flat_pred.get("point_embed")

        # Compute per-granularity loss
        targets_g = targets_by_gran[g]
        g_loss = criterion(flat_pred, targets_g, context=f"eval/{g}")
        heads_loss[g] = g_loss
        loss_total = loss_total + g_loss["loss_total"]

    pred_multihead = {"heads": heads_pred}
    if point_embed is not None:
        pred_multihead["point_embed"] = point_embed

    loss_multihead = {
        "loss_total": loss_total / len(granularities),
        "heads": heads_loss,
    }

    return pred_multihead, loss_multihead


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
    eval_benchmarks: str | list[str] | tuple[str, ...] | None = None,
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

    Supports both multi-head and continuous decoder models transparently.
    For continuous decoders, the model is run once per granularity to produce
    the same multi-head output structure expected by downstream evaluators.

    Parameters
    ----------
    model:
        Student model (``StudentInstanceSegModel``).
    dataset:
        ``MultiSceneDataset`` to evaluate on.
    criterion:
        ``MultiGranCriterion`` or ``SingleGranCriterion`` for computing
        validation loss.
    device:
        Target CUDA device string.
    granularities:
        Dot-free granularity keys.
    score_threshold / mask_threshold / min_points:
        Proposal extraction thresholds.
    eval_benchmark:
        Backward-compatible ScanNet GT benchmark name (single).
    eval_benchmarks:
        If set, evaluate real GT against all listed benchmarks (e.g. ["scannet20","scannet200"]).
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

    continuous = _is_continuous_model(model)
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
            if continuous:
                # Run model once per granularity, assemble multi-head output
                pred, loss_result = _evaluate_continuous_scene(
                    model, points, features, granularities,
                    criterion, targets_by_gran,
                )
            else:
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
                    eval_benchmarks=(eval_benchmarks if eval_benchmarks is not None else eval_benchmark),
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

