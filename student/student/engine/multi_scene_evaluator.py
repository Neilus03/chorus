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
from student.metrics.official_instance_ap import (
    evaluate_official_and_oracle_ap,
    merge_ap_record_sets,
)
from student.metrics.pseudo_metrics import compute_pseudo_metrics_multi
from student.models.continuous_base import is_continuous_decoder

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
    """Check if model uses a continuous granularity decoder."""
    # Unwrap DDP / FineTuningWrapper
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "model"):
        m = m.model
    decoder = getattr(m, "decoder", None)
    return is_continuous_decoder(decoder)


def _clear_backbone_cache(model: nn.Module) -> None:
    """Invalidate LitePT voxelization cache for multi-scene processing."""
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _safe_mean(values: list[float]) -> float:
    """Mean that filters NaN values; returns 0.0 for empty input."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else 0.0


def _official_mean(values: list[float]) -> float:
    """Mean for official AP values; all-empty/all-NaN remains undefined."""
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, Real) else float("nan")


def _scope_from_values(scopes: list[str]) -> str:
    clean = {str(s) for s in scopes if s}
    if not clean:
        return "full_scene"
    if len(clean) == 1:
        return next(iter(clean))
    return "mixed"


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
    pseudo_legacy_recall25_all: list[float] = []
    pseudo_legacy_recall50_all: list[float] = []
    pseudo_nmi_all: list[float] = []
    pseudo_ari_all: list[float] = []
    # Real GT metrics can be stored either as a legacy single dict (`real_gt`)
    # or as a dict of benchmark→metrics (`real_gt_by_benchmark`).
    real_legacy_recall25_by_bench: dict[str, list[float]] = {}
    real_legacy_recall50_by_bench: dict[str, list[float]] = {}
    real_matched_iou_by_bench: dict[str, list[float]] = {}
    real_nmi_by_bench: dict[str, list[float]] = {}
    real_ari_by_bench: dict[str, list[float]] = {}
    real_class_ap25_by_bench: dict[str, list[float]] = {}
    real_class_ap50_by_bench: dict[str, list[float]] = {}
    real_sem_miou_by_bench: dict[str, list[float]] = {}
    matched_iou_by_gran: dict[str, list[float]] = {g: [] for g in granularities}
    proposal_score_pass_by_gran: dict[str, list[float]] = {g: [] for g in granularities}
    proposal_kept_by_gran: dict[str, list[float]] = {g: [] for g in granularities}
    proposal_removed_by_gran: dict[str, list[float]] = {g: [] for g in granularities}
    pseudo_records_by_gran: dict[str, list[dict[str, Any]]] = {g: [] for g in granularities}
    pseudo_scope_by_gran: dict[str, list[str]] = {g: [] for g in granularities}
    real_records_by_bench_gran: dict[str, dict[str, list[dict[str, Any]]]] = {}
    real_class_records_by_bench_gran: dict[str, dict[str, list[dict[str, Any]]]] = {}
    real_scope_by_bench_gran: dict[str, dict[str, list[str]]] = {}

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
            _append_real(proposal_score_pass_by_gran[g], g_eval.get("num_score_pass"))
            _append_real(proposal_kept_by_gran[g], g_eval.get("num_proposals"))
            _append_real(
                proposal_removed_by_gran[g],
                g_eval.get("num_min_points_removed"),
            )

            pseudo = g_eval.get("pseudo_gt", {})
            if isinstance(pseudo, dict):
                _append_real(
                    pseudo_legacy_recall25_all,
                    pseudo.get("legacy_matched_recall25"),
                )
                _append_real(
                    pseudo_legacy_recall50_all,
                    pseudo.get("legacy_matched_recall50"),
                )
                _append_real(pseudo_nmi_all, pseudo.get("NMI"))
                _append_real(pseudo_ari_all, pseudo.get("ARI"))
                records = pseudo.get("official_records")
                if isinstance(records, dict):
                    pseudo_records_by_gran[g].append(records)
                    pseudo_scope_by_gran[g].append(str(g_eval.get("eval_scope", "full_scene")))

            real_by = g_eval.get("real_gt_by_benchmark", None)
            if isinstance(real_by, dict):
                for bench, real in real_by.items():
                    if not isinstance(real, dict):
                        continue
                    if "legacy_matched_recall25" in real and "legacy_matched_recall50" in real:
                        real_legacy_recall25_by_bench.setdefault(str(bench), [])
                        real_legacy_recall50_by_bench.setdefault(str(bench), [])
                        real_matched_iou_by_bench.setdefault(str(bench), [])
                        real_nmi_by_bench.setdefault(str(bench), [])
                        real_ari_by_bench.setdefault(str(bench), [])
                        _append_real(
                            real_legacy_recall25_by_bench[str(bench)],
                            real.get("legacy_matched_recall25"),
                        )
                        _append_real(
                            real_legacy_recall50_by_bench[str(bench)],
                            real.get("legacy_matched_recall50"),
                        )
                        _append_real(real_matched_iou_by_bench[str(bench)], real.get("matched_mean_iou"))
                        _append_real(real_nmi_by_bench[str(bench)], real.get("NMI"))
                        _append_real(real_ari_by_bench[str(bench)], real.get("ARI"))
                        records = real.get("official_records")
                        if isinstance(records, dict):
                            real_records_by_bench_gran.setdefault(str(bench), {}).setdefault(g, []).append(records)
                            real_scope_by_bench_gran.setdefault(str(bench), {}).setdefault(g, []).append(
                                str(g_eval.get("eval_scope", "full_scene"))
                            )
                        class_aware = real.get("class_aware", {})
                        if isinstance(class_aware, dict):
                            real_class_ap25_by_bench.setdefault(str(bench), [])
                            real_class_ap50_by_bench.setdefault(str(bench), [])
                            real_sem_miou_by_bench.setdefault(str(bench), [])
                            _append_real(real_class_ap25_by_bench[str(bench)], class_aware.get("AP25"))
                            _append_real(real_class_ap50_by_bench[str(bench)], class_aware.get("AP50"))
                            _append_real(
                                real_sem_miou_by_bench[str(bench)],
                                class_aware.get("semantic_mIoU"),
                            )
                            class_records = class_aware.get("official_records")
                            if isinstance(class_records, dict):
                                real_class_records_by_bench_gran.setdefault(str(bench), {}).setdefault(g, []).append(class_records)
                continue

            # Legacy fallback: single real_gt dict
            real = g_eval.get("real_gt", {})
            if isinstance(real, dict) and "legacy_matched_recall25" in real and "legacy_matched_recall50" in real:
                bench = str(real.get("eval_benchmark", "unknown"))
                real_legacy_recall25_by_bench.setdefault(bench, [])
                real_legacy_recall50_by_bench.setdefault(bench, [])
                real_matched_iou_by_bench.setdefault(bench, [])
                real_nmi_by_bench.setdefault(bench, [])
                real_ari_by_bench.setdefault(bench, [])
                _append_real(real_legacy_recall25_by_bench[bench], real.get("legacy_matched_recall25"))
                _append_real(real_legacy_recall50_by_bench[bench], real.get("legacy_matched_recall50"))
                _append_real(real_matched_iou_by_bench[bench], real.get("matched_mean_iou"))
                _append_real(real_nmi_by_bench[bench], real.get("NMI"))
                _append_real(real_ari_by_bench[bench], real.get("ARI"))
                records = real.get("official_records")
                if isinstance(records, dict):
                    real_records_by_bench_gran.setdefault(bench, {}).setdefault(g, []).append(records)
                    real_scope_by_bench_gran.setdefault(bench, {}).setdefault(g, []).append(
                        str(g_eval.get("eval_scope", "full_scene"))
                    )

    aggregate: dict[str, Any] = {
        "loss_mean": _safe_mean(all_losses),
        "legacy_pseudo_matched_recall25_mean": _safe_mean(pseudo_legacy_recall25_all),
        "legacy_pseudo_matched_recall50_mean": _safe_mean(pseudo_legacy_recall50_all),
        "pseudo_NMI_mean": _safe_mean(pseudo_nmi_all),
        "pseudo_ARI_mean": _safe_mean(pseudo_ari_all),
    }
    all_score_pass: list[float] = []
    all_kept: list[float] = []
    all_removed: list[float] = []
    for g in granularities:
        score_pass = proposal_score_pass_by_gran.get(g, [])
        kept = proposal_kept_by_gran.get(g, [])
        removed = proposal_removed_by_gran.get(g, [])
        aggregate[f"proposal_score_pass_{g}"] = _safe_mean(score_pass)
        aggregate[f"proposal_kept_{g}"] = _safe_mean(kept)
        aggregate[f"proposal_removed_min_points_{g}"] = _safe_mean(removed)
        all_score_pass.extend(score_pass)
        all_kept.extend(kept)
        all_removed.extend(removed)
    aggregate["proposal_score_pass_mean"] = _safe_mean(all_score_pass)
    aggregate["proposal_kept_mean"] = _safe_mean(all_kept)
    aggregate["proposal_removed_min_points_mean"] = _safe_mean(all_removed)

    pseudo_ap: list[float] = []
    pseudo_ap50: list[float] = []
    pseudo_ap25: list[float] = []
    pseudo_oracle_ap50: list[float] = []
    pseudo_oracle_ap25: list[float] = []
    for g in granularities:
        record_sets = pseudo_records_by_gran.get(g, [])
        if not record_sets:
            continue
        records = merge_ap_record_sets(record_sets)
        metrics = evaluate_official_and_oracle_ap(records)
        aggregate[f"pseudo_{g}_official_AP"] = _metric_value(metrics, "AP")
        aggregate[f"pseudo_{g}_official_AP50"] = _metric_value(metrics, "AP50")
        aggregate[f"pseudo_{g}_official_AP25"] = _metric_value(metrics, "AP25")
        aggregate[f"pseudo_{g}_oracle_AP50"] = _metric_value(metrics, "oracle_AP50")
        aggregate[f"pseudo_{g}_oracle_AP25"] = _metric_value(metrics, "oracle_AP25")
        aggregate[f"pseudo_{g}_official_total_gt"] = int(metrics.get("total_gt_instances", 0))
        aggregate[f"pseudo_{g}_official_num_predictions"] = int(metrics.get("num_predictions", 0))
        aggregate[f"pseudo_{g}_eval_scope"] = _scope_from_values(pseudo_scope_by_gran.get(g, []))
        pseudo_ap.append(_metric_value(metrics, "AP"))
        pseudo_ap50.append(_metric_value(metrics, "AP50"))
        pseudo_ap25.append(_metric_value(metrics, "AP25"))
        pseudo_oracle_ap50.append(_metric_value(metrics, "oracle_AP50"))
        pseudo_oracle_ap25.append(_metric_value(metrics, "oracle_AP25"))

    aggregate["pseudo_official_AP_mean"] = _official_mean(pseudo_ap)
    aggregate["pseudo_official_AP50_mean"] = _official_mean(pseudo_ap50)
    aggregate["pseudo_official_AP25_mean"] = _official_mean(pseudo_ap25)
    aggregate["pseudo_oracle_AP50_mean"] = _official_mean(pseudo_oracle_ap50)
    aggregate["pseudo_oracle_AP25_mean"] = _official_mean(pseudo_oracle_ap25)

    # Per-benchmark real GT aggregates (no primary)
    for bench in sorted(real_legacy_recall25_by_bench):
        aggregate[f"legacy_matched_recall25_{bench}"] = _safe_mean(real_legacy_recall25_by_bench[bench])
        aggregate[f"legacy_matched_recall50_{bench}"] = _safe_mean(real_legacy_recall50_by_bench.get(bench, []))
        aggregate[f"matched_mean_iou_{bench}"] = _safe_mean(real_matched_iou_by_bench.get(bench, []))
        aggregate[f"real_NMI_mean_{bench}"] = _safe_mean(real_nmi_by_bench.get(bench, []))
        aggregate[f"real_ARI_mean_{bench}"] = _safe_mean(real_ari_by_bench.get(bench, []))
        if real_class_ap25_by_bench.get(bench):
            aggregate[f"real_class_AP25_mean_{bench}"] = _safe_mean(
                real_class_ap25_by_bench[bench]
            )
        if real_class_ap50_by_bench.get(bench):
            aggregate[f"real_class_AP50_mean_{bench}"] = _safe_mean(
                real_class_ap50_by_bench[bench]
            )
        if real_sem_miou_by_bench.get(bench):
            aggregate[f"real_sem_mIoU_mean_{bench}"] = _safe_mean(
                real_sem_miou_by_bench[bench]
            )

    for bench, by_gran in sorted(real_records_by_bench_gran.items()):
        per_g_ap: list[float] = []
        per_g_ap50: list[float] = []
        per_g_ap25: list[float] = []
        per_g_oracle_ap50: list[float] = []
        scopes: list[str] = []
        for g in granularities:
            record_sets = by_gran.get(g, [])
            if not record_sets:
                continue
            records = merge_ap_record_sets(record_sets)
            metrics = evaluate_official_and_oracle_ap(records)
            scope = _scope_from_values(
                real_scope_by_bench_gran.get(bench, {}).get(g, [])
            )
            scopes.append(scope)
            aggregate[f"real_{g}_{scope}_official_AP_{bench}"] = _metric_value(metrics, "AP")
            aggregate[f"real_{g}_{scope}_official_AP50_{bench}"] = _metric_value(metrics, "AP50")
            aggregate[f"real_{g}_{scope}_official_AP25_{bench}"] = _metric_value(metrics, "AP25")
            aggregate[f"real_{g}_{scope}_oracle_AP50_{bench}"] = _metric_value(metrics, "oracle_AP50")
            aggregate[f"real_{g}_{scope}_official_total_gt_{bench}"] = int(metrics.get("total_gt_instances", 0))
            aggregate[f"real_{g}_{scope}_official_num_predictions_{bench}"] = int(metrics.get("num_predictions", 0))
            per_g_ap.append(_metric_value(metrics, "AP"))
            per_g_ap50.append(_metric_value(metrics, "AP50"))
            per_g_ap25.append(_metric_value(metrics, "AP25"))
            per_g_oracle_ap50.append(_metric_value(metrics, "oracle_AP50"))

        scope = _scope_from_values(scopes)
        aggregate[f"real_{scope}_official_AP_{bench}"] = _official_mean(per_g_ap)
        aggregate[f"real_{scope}_official_AP50_{bench}"] = _official_mean(per_g_ap50)
        aggregate[f"real_{scope}_official_AP25_{bench}"] = _official_mean(per_g_ap25)
        aggregate[f"real_{scope}_oracle_AP50_{bench}"] = _official_mean(per_g_oracle_ap50)

    for bench, by_gran in sorted(real_class_records_by_bench_gran.items()):
        per_g_ap: list[float] = []
        per_g_ap50: list[float] = []
        per_g_ap25: list[float] = []
        per_g_oracle_ap50: list[float] = []
        scopes: list[str] = []
        for g in granularities:
            record_sets = by_gran.get(g, [])
            if not record_sets:
                continue
            records = merge_ap_record_sets(record_sets)
            metrics = evaluate_official_and_oracle_ap(records)
            scope = _scope_from_values(
                real_scope_by_bench_gran.get(bench, {}).get(g, [])
            )
            scopes.append(scope)
            per_g_ap.append(_metric_value(metrics, "AP"))
            per_g_ap50.append(_metric_value(metrics, "AP50"))
            per_g_ap25.append(_metric_value(metrics, "AP25"))
            per_g_oracle_ap50.append(_metric_value(metrics, "oracle_AP50"))
            aggregate[f"real_class_{g}_{scope}_official_AP_{bench}"] = _metric_value(metrics, "AP")
            aggregate[f"real_class_{g}_{scope}_official_AP50_{bench}"] = _metric_value(metrics, "AP50")
            aggregate[f"real_class_{g}_{scope}_official_AP25_{bench}"] = _metric_value(metrics, "AP25")
            aggregate[f"real_class_{g}_{scope}_oracle_AP50_{bench}"] = _metric_value(metrics, "oracle_AP50")
        if per_g_ap50:
            scope = _scope_from_values(scopes)
            aggregate[f"real_class_{scope}_official_AP_{bench}"] = _official_mean(per_g_ap)
            aggregate[f"real_class_{scope}_official_AP50_{bench}"] = _official_mean(per_g_ap50)
            aggregate[f"real_class_{scope}_official_AP25_{bench}"] = _official_mean(per_g_ap25)
            aggregate[f"real_class_{scope}_oracle_AP50_{bench}"] = _official_mean(per_g_oracle_ap50)

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
        if "class_logits" in flat_pred:
            heads_pred[g]["class_logits"] = flat_pred["class_logits"]
        if point_embed is None:
            point_embed = flat_pred.get("point_embed")

        # Compute per-granularity loss
        targets_g = targets_by_gran[g]
        g_loss = criterion(
            flat_pred, targets_g, context=f"eval/{g}", granularity_key=g,
        )
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


def _evaluate_prompt_scene(
    model: nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    prompt_key: str,
    criterion: nn.Module,
    targets_by_gran: dict,
) -> tuple[dict, dict]:
    """Run a prompt-tuned continuous model once using its learned ``g_ft``."""
    flat_pred = model(points, features)
    heads_pred = {
        prompt_key: {
            "mask_logits": flat_pred["mask_logits"],
            "score_logits": flat_pred["score_logits"],
            "query_embed": flat_pred.get("query_embed"),
        },
    }
    if "class_logits" in flat_pred:
        heads_pred[prompt_key]["class_logits"] = flat_pred["class_logits"]
    pred_multihead: dict[str, Any] = {"heads": heads_pred}
    if "point_embed" in flat_pred:
        pred_multihead["point_embed"] = flat_pred["point_embed"]

    targets_g = targets_by_gran[prompt_key]
    g_loss = criterion(
        flat_pred,
        targets_g,
        context=f"eval/prompt/{prompt_key}",
        granularity_key=prompt_key,
    )
    loss_multihead = {
        "loss_total": g_loss["loss_total"],
        "heads": {prompt_key: g_loss},
    }
    return pred_multihead, loss_multihead


def evaluate_multi_scene(
    model: nn.Module,
    dataset: MultiSceneDataset | Subset,
    criterion: nn.Module,
    *,
    device: str,
    granularities: tuple[str, ...],
    score_threshold: float | dict[str, float] = 0.3,
    class_score_threshold: float | None = None,
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
    prompt_finetune: bool = False,
    prompt_target_granularity: str | None = None,
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
    prompt_key = prompt_target_granularity or (granularities[0] if granularities else None)
    if prompt_finetune and prompt_key not in granularities:
        raise ValueError(
            f"prompt_target_granularity={prompt_key!r} must be one of {granularities}"
        )
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
            instance_class_maps=sample.get("instance_classes_by_granularity"),
        )

        _clear_backbone_cache(model)

        with torch.no_grad():
            if prompt_finetune:
                if prompt_key is None:
                    raise ValueError("Prompt fine-tuning requires at least one granularity key")
                pred, loss_result = _evaluate_prompt_scene(
                    model, points, features, prompt_key,
                    criterion, targets_by_gran,
                )
            elif continuous:
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
            if fragment_merge_eval and not prompt_finetune:
                base_ds, real_idx = _base_dataset_and_index(dataset, idx)
                eval_result = evaluate_scene_fragment_merge(
                    model,
                    base_ds,
                    real_idx,
                    device=device,
                    granularities=granularities,
                    score_threshold=score_threshold,
                    class_score_threshold=class_score_threshold,
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
                    class_score_threshold=class_score_threshold,
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
