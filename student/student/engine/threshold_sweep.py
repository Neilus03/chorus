"""Eval-only score-threshold sweeps from cached raw validation predictions.

The cache written by this module is explicitly pre-score-threshold: every query
score is stored, and binary masks are packed immediately after ``mask_threshold``
is applied.  ``score_threshold`` and ``min_points`` are applied later when AP is
recomputed offline.
"""

from __future__ import annotations

import math
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from student.data.target_builder import build_instance_targets_multi
from student.engine.evaluator import (
    _build_pseudo_gt_ids,
    _ensure_chorus_importable,
    compute_legacy_best_match_recall,
)
from student.metrics.official_instance_ap import (
    SCANNET_MIN_REGION_SIZE,
    build_instance_ap_records,
    evaluate_official_and_oracle_ap,
    merge_ap_record_sets,
)
from student.metrics.eval_diagnostics import topk_diagnostics
from student.models.continuous_base import is_continuous_decoder
from student.models.finetune_wrapper import FineTuningWrapper

log = logging.getLogger(__name__)

CACHE_VERSION = 1
NORMAL_VALIDATION_OFFICIAL_SCORE_THRESHOLD = 0.0

_GRAN_KEY_TO_VAL = {
    "g01": 0.1,
    "g02": 0.2,
    "g03": 0.3,
    "g04": 0.4,
    "g05": 0.5,
    "g06": 0.6,
    "g07": 0.7,
    "g08": 0.8,
    "g09": 0.9,
    "g10": 1.0,
}


def parse_thresholds(value: str | Iterable[float]) -> list[float]:
    """Parse and de-duplicate score thresholds while preserving order."""
    if isinstance(value, str):
        raw = [float(x.strip()) for x in value.split(",") if x.strip()]
    else:
        raw = [float(x) for x in value]
    out: list[float] = []
    seen: set[float] = set()
    for threshold in raw:
        key = round(float(threshold), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(threshold))
    return out


def normalize_benchmarks(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["scannet200"]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    out = [p for p in parts if p]
    return out or ["scannet200"]


def save_prediction_cache(cache: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_prediction_cache(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError(f"Prediction cache {path} did not contain a dict")
    if int(cache.get("version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Unsupported threshold-sweep cache version {cache.get('version')!r}; "
            f"expected {CACHE_VERSION}"
        )
    return cache


def load_or_create_prediction_cache(
    *,
    cache_path: str | Path,
    reuse_cache: bool,
    write_cache: bool,
    build_cache_fn: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Load an existing cache when requested, otherwise call ``build_cache_fn``.

    Returns ``(cache, reused)``.  This helper deliberately does not build or touch
    a model when ``reuse_cache`` succeeds; tests use that property to guard the
    eval-only reuse path.
    """
    cache_path = Path(cache_path)
    if reuse_cache and cache_path.exists():
        return load_prediction_cache(cache_path), True

    cache = build_cache_fn()
    if write_cache:
        save_prediction_cache(cache, cache_path)
    return cache, False


def set_eval_mode_like_validation(model: nn.Module) -> None:
    """Mirror the existing multi-scene validation mode, including BN behavior."""
    model.eval()
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            module.train()


def _gran_key_to_float(key: str) -> float:
    if key in _GRAN_KEY_TO_VAL:
        return _GRAN_KEY_TO_VAL[key]
    return float(str(key).replace("g0", "0.").replace("g", "0."))


def _is_continuous_model(model: nn.Module) -> bool:
    unwrapped = model
    if isinstance(unwrapped, FineTuningWrapper):
        unwrapped = unwrapped.model
    return is_continuous_decoder(getattr(unwrapped, "decoder", None))


def _clear_backbone_cache(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _pack_masks(masks_binary: np.ndarray) -> np.ndarray:
    masks_bool = np.asarray(masks_binary, dtype=bool)
    if masks_bool.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {masks_bool.shape}")
    return np.packbits(masks_bool, axis=1, bitorder="little")


def _unpack_masks(head_cache: dict[str, Any]) -> np.ndarray:
    packed = np.asarray(head_cache["packed_masks"], dtype=np.uint8)
    num_points = int(head_cache["num_points"])
    return np.unpackbits(
        packed,
        axis=1,
        count=num_points,
        bitorder="little",
    ).astype(bool, copy=False)


def _forward_heads(
    model: nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    granularities: tuple[str, ...],
    prompt_finetune: bool,
    prompt_target_granularity: str | None,
) -> dict[str, dict[str, torch.Tensor]]:
    if prompt_finetune:
        prompt_key = prompt_target_granularity or granularities[0]
        pred = model(points, features)
        if isinstance(pred, list):
            raise RuntimeError("Unexpected batched list output for single-scene prompt eval")
        return {prompt_key: pred}

    if _is_continuous_model(model):
        heads: dict[str, dict[str, torch.Tensor]] = {}
        for granularity in granularities:
            pred = model(points, features, target_g=_gran_key_to_float(granularity))
            if isinstance(pred, list):
                raise RuntimeError("Unexpected batched list output for single-scene continuous eval")
            heads[granularity] = pred
        return heads

    pred = model(points, features)
    if isinstance(pred, list):
        raise RuntimeError("Unexpected batched list output for single-scene multi-head eval")
    return dict(pred["heads"])


def _load_real_gt_for_cache(
    *,
    scene_dir: str | Path,
    scene_id: str,
    benchmark: str,
    vertex_indices: torch.Tensor | None,
    expected_points: int,
) -> np.ndarray:
    _ensure_chorus_importable()
    from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids

    real_gt = load_scannet_gt_instance_ids(Path(scene_dir), scene_id, eval_benchmark=benchmark)
    if vertex_indices is not None:
        vi = vertex_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if vi.shape[0] != expected_points:
            raise ValueError(
                f"vertex_indices length {vi.shape[0]} != model point count {expected_points}"
            )
        if vi.size > 0 and (int(vi.min()) < 0 or int(vi.max()) >= real_gt.shape[0]):
            raise ValueError("vertex_indices out of range for ScanNet GT")
        real_gt = real_gt[vi]
    if real_gt.shape[0] != expected_points:
        raise ValueError(f"GT vertex count {real_gt.shape[0]} != model point count {expected_points}")
    return np.asarray(real_gt, dtype=np.int32)


def build_prediction_cache(
    *,
    model: nn.Module,
    dataset: Any,
    device: str,
    granularities: tuple[str, ...],
    eval_benchmarks: list[str],
    mask_threshold: float,
    min_points: int,
    min_instance_points: int,
    dense_instance_ids: bool,
    prompt_finetune: bool = False,
    prompt_target_granularity: str | None = None,
    max_scenes: int | None = None,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run validation inference once and cache pre-score-threshold predictions."""
    set_eval_mode_like_validation(model)
    scene_limit = len(dataset) if max_scenes is None else min(len(dataset), max(int(max_scenes), 0))
    scenes: list[dict[str, Any]] = []

    for idx in range(scene_limit):
        sample = dataset[idx]
        scene_id = str(sample["scene_id"])
        log.info("[%d/%d] caching pre-score-threshold predictions for %s", idx + 1, scene_limit, scene_id)
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
            heads_pred = _forward_heads(
                model,
                points,
                features,
                granularities=granularities,
                prompt_finetune=prompt_finetune,
                prompt_target_granularity=prompt_target_granularity,
            )

        head_cache: dict[str, dict[str, Any]] = {}
        expected_points: int | None = None
        for granularity in granularities:
            if granularity not in heads_pred:
                continue
            pred_g = heads_pred[granularity]
            mask_logits = pred_g["mask_logits"].detach().cpu()
            score_logits = pred_g["score_logits"].detach().cpu()
            masks_binary = (mask_logits.sigmoid().numpy() >= float(mask_threshold))
            num_points = int(masks_binary.shape[1])
            if expected_points is None:
                expected_points = num_points
            elif expected_points != num_points:
                raise ValueError(
                    f"{scene_id} has inconsistent point counts across heads: "
                    f"{expected_points} vs {num_points}"
                )
            targets_g = targets_by_gran[granularity]
            score_logits_np = score_logits.numpy().astype(np.float32, copy=False)
            score_probs_np = score_logits.sigmoid().numpy().astype(np.float32, copy=False)
            head_cache[granularity] = {
                "num_points": num_points,
                "num_queries": int(masks_binary.shape[0]),
                "score_logits": score_logits_np,
                "score_probs": score_probs_np,
                "query_indices": np.arange(masks_binary.shape[0], dtype=np.int64),
                "mask_point_counts": masks_binary.sum(axis=1).astype(np.int32, copy=False),
                "packed_masks": _pack_masks(masks_binary),
                "packed_bitorder": "little",
                "pseudo_gt_ids": _build_pseudo_gt_ids(targets_g).astype(np.int32, copy=False),
                "pseudo_supervision_mask": targets_g.supervision_mask.detach().cpu().numpy().astype(bool, copy=False),
            }

        if expected_points is None:
            raise RuntimeError(f"No prediction heads cached for scene {scene_id}")

        real_by_benchmark: dict[str, Any] = {}
        for benchmark in eval_benchmarks:
            try:
                real_by_benchmark[benchmark] = _load_real_gt_for_cache(
                    scene_dir=sample["scene_dir"],
                    scene_id=scene_id,
                    benchmark=benchmark,
                    vertex_indices=sample.get("vertex_indices"),
                    expected_points=expected_points,
                )
            except Exception as exc:
                real_by_benchmark[benchmark] = {"error": str(exc)}

        scenes.append(
            {
                "scene_id": scene_id,
                "scene_dir": str(sample["scene_dir"]),
                "num_points": int(expected_points),
                "eval_scope": "crop" if sample.get("vertex_indices") is not None else "full_scene",
                "heads": head_cache,
                "real_gt_by_benchmark": real_by_benchmark,
            }
        )

    return {
        "version": CACHE_VERSION,
        "cache_is_pre_score_threshold": True,
        "mask_representation": "np.packbits(binary_masks_after_mask_threshold_before_score_threshold)",
        "normal_validation_official_score_threshold": NORMAL_VALIDATION_OFFICIAL_SCORE_THRESHOLD,
        "granularities": list(granularities),
        "eval_benchmarks": list(eval_benchmarks),
        "settings": {
            "mask_threshold": float(mask_threshold),
            "min_points_per_proposal": int(min_points),
            "min_instance_points": int(min_instance_points),
            "dense_instance_ids": bool(dense_instance_ids),
            "prompt_finetune": bool(prompt_finetune),
            "prompt_target_granularity": prompt_target_granularity,
            "max_scenes": max_scenes,
            "config": config_path,
            "checkpoint": checkpoint_path,
            **(checkpoint_info or {}),
        },
        "scenes": scenes,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [_finite_or_none(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _scope_from_values(values: Iterable[str]) -> str:
    clean = {str(v) for v in values if v}
    if not clean:
        return "full_scene"
    if len(clean) == 1:
        return next(iter(clean))
    return "mixed"


def _records_metric_row(prefix: str, metrics: dict[str, Any], row: dict[str, Any]) -> None:
    row[f"{prefix}_AP"] = _finite_or_none(metrics.get("AP"))
    row[f"{prefix}_AP50"] = _finite_or_none(metrics.get("AP50"))
    row[f"{prefix}_AP25"] = _finite_or_none(metrics.get("AP25"))
    row[f"{prefix}_oracle_AP"] = _finite_or_none(metrics.get("oracle_AP"))
    row[f"{prefix}_oracle_AP50"] = _finite_or_none(metrics.get("oracle_AP50"))
    row[f"{prefix}_oracle_AP25"] = _finite_or_none(metrics.get("oracle_AP25"))
    row[f"{prefix}_total_gt"] = int(metrics.get("total_gt_instances", 0) or 0)
    row[f"{prefix}_num_predictions"] = int(metrics.get("num_predictions", 0) or 0)


def _topk_metric_row(
    prefix: str,
    records: dict[str, Any],
    row: dict[str, Any],
    *,
    topk_values: Iterable[int] | None,
) -> None:
    if not topk_values:
        return
    topk_by_score, topk_by_oracle = topk_diagnostics(records, topk_values=topk_values)
    for source_name, source_metrics in (
        ("by_score", topk_by_score),
        ("by_oracle", topk_by_oracle),
    ):
        for raw_k, metrics in source_metrics.items():
            k = str(raw_k)
            for metric_name in ("AP50", "AP25", "recall50", "recall25", "num_predictions"):
                value = metrics.get(metric_name)
                key = f"{prefix}_top{k}_{source_name}_{metric_name}"
                row[key] = int(value) if metric_name == "num_predictions" else _finite_or_none(value)


def _proposals_for_threshold(
    head_cache: dict[str, Any],
    *,
    score_threshold: float,
    min_points: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, int]]:
    scores = np.asarray(head_cache["score_probs"], dtype=np.float32)
    mask_counts = np.asarray(head_cache["mask_point_counts"], dtype=np.int64)
    score_pass = scores >= float(score_threshold)
    size_pass = mask_counts >= int(min_points)
    keep_idx = np.flatnonzero(score_pass & size_pass).astype(np.int64, copy=False)

    if keep_idx.size:
        masks_binary = _unpack_masks(head_cache)
        proposals = [masks_binary[int(i)] for i in keep_idx.tolist()]
    else:
        proposals = []
    query_indices_all = np.asarray(head_cache.get("query_indices", np.arange(scores.shape[0])), dtype=np.int64)
    kept_scores = scores[keep_idx].astype(np.float32, copy=False)
    kept_queries = query_indices_all[keep_idx].astype(np.int64, copy=False)
    stats = {
        "num_queries": int(scores.shape[0]),
        "num_score_pass": int(score_pass.sum()),
        "num_min_points_removed": int((score_pass & ~size_pass).sum()),
        "num_proposals": int(keep_idx.shape[0]),
        "min_points_per_proposal": int(min_points),
    }
    return proposals, kept_scores, kept_queries, stats


def evaluate_threshold_from_cache(
    cache: dict[str, Any],
    *,
    score_threshold: float,
    primary_benchmark: str | None = None,
    topk_values: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Recompute pseudo and real official AP for one score threshold."""
    granularities = tuple(str(g) for g in cache.get("granularities", []))
    eval_benchmarks = normalize_benchmarks(cache.get("eval_benchmarks", None))
    if primary_benchmark is None:
        primary_benchmark = eval_benchmarks[0]
    settings = cache.get("settings", {})
    min_points = int(settings.get("min_points_per_proposal", 30))

    pseudo_records_by_gran: dict[str, list[dict[str, Any]]] = {g: [] for g in granularities}
    real_records_by_bench_gran: dict[str, dict[str, list[dict[str, Any]]]] = {}
    pseudo_recall25: list[float] = []
    pseudo_recall50: list[float] = []
    pseudo_iou: list[float] = []
    real_recall25_by_bench: dict[str, list[float]] = {}
    real_recall50_by_bench: dict[str, list[float]] = {}
    real_iou_by_bench: dict[str, list[float]] = {}
    scopes: list[str] = []
    stat_score_pass: list[int] = []
    stat_removed_min_points: list[int] = []
    stat_kept: list[int] = []
    stat_queries: list[int] = []

    for scene in list(cache.get("scenes", []) or []):
        scene_id = str(scene["scene_id"])
        scopes.append(str(scene.get("eval_scope", "full_scene")))
        heads = scene.get("heads", {})
        for granularity in granularities:
            head_cache = heads.get(granularity)
            if not isinstance(head_cache, dict):
                continue
            proposals, scores, query_indices, proposal_stats = _proposals_for_threshold(
                head_cache,
                score_threshold=float(score_threshold),
                min_points=min_points,
            )
            stat_score_pass.append(proposal_stats["num_score_pass"])
            stat_removed_min_points.append(proposal_stats["num_min_points_removed"])
            stat_kept.append(proposal_stats["num_proposals"])
            stat_queries.append(proposal_stats["num_queries"])

            pseudo_gt = np.asarray(head_cache["pseudo_gt_ids"], dtype=np.int64)
            pseudo_records_by_gran[granularity].append(
                build_instance_ap_records(
                    scene_id=scene_id,
                    gt_ids=pseudo_gt,
                    proposals=proposals,
                    scores=scores,
                    query_indices=query_indices,
                    class_agnostic=True,
                    eval_mask=np.asarray(head_cache["pseudo_supervision_mask"], dtype=bool),
                )
            )
            pseudo_legacy = compute_legacy_best_match_recall(pseudo_gt, proposals)
            pseudo_recall25.append(float(pseudo_legacy["legacy_matched_recall25"]))
            pseudo_recall50.append(float(pseudo_legacy["legacy_matched_recall50"]))
            pseudo_iou.append(float(pseudo_legacy["matched_mean_iou"]))

            real_by_benchmark = scene.get("real_gt_by_benchmark", {})
            for benchmark in eval_benchmarks:
                real_gt_raw = real_by_benchmark.get(benchmark)
                if isinstance(real_gt_raw, dict) and "error" in real_gt_raw:
                    continue
                if real_gt_raw is None:
                    continue
                real_gt = np.asarray(real_gt_raw, dtype=np.int64)
                real_records_by_bench_gran.setdefault(benchmark, {}).setdefault(granularity, []).append(
                    build_instance_ap_records(
                        scene_id=scene_id,
                        gt_ids=real_gt,
                        proposals=proposals,
                        scores=scores,
                        query_indices=query_indices,
                        class_agnostic=True,
                        min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                        min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                    )
                )
                real_legacy = compute_legacy_best_match_recall(real_gt, proposals)
                real_recall25_by_bench.setdefault(benchmark, []).append(
                    float(real_legacy["legacy_matched_recall25"])
                )
                real_recall50_by_bench.setdefault(benchmark, []).append(
                    float(real_legacy["legacy_matched_recall50"])
                )
                real_iou_by_bench.setdefault(benchmark, []).append(
                    float(real_legacy["matched_mean_iou"])
                )
    row: dict[str, Any] = {
        "score_threshold": float(score_threshold),
        "eval_scope": _scope_from_values(scopes),
        "num_scene_heads": int(len(stat_kept)),
        "mean_score_pass": _mean(stat_score_pass),
        "mean_removed_min_points": _mean(stat_removed_min_points),
        "mean_kept_after_min_points": _mean(stat_kept),
        "mean_kept_per_scene": _mean(stat_kept),
        "total_kept_after_min_points": int(sum(stat_kept)),
        "total_predictions": int(sum(stat_kept)),
        "mean_num_queries": _mean(stat_queries),
        "min_points_per_proposal": min_points,
        "mask_threshold": float(settings.get("mask_threshold", 0.5)),
        "normal_validation_official_score_threshold": NORMAL_VALIDATION_OFFICIAL_SCORE_THRESHOLD,
    }

    pseudo_ap: list[float] = []
    pseudo_ap50: list[float] = []
    pseudo_ap25: list[float] = []
    pseudo_oracle_ap50: list[float] = []
    pseudo_total_gt: list[int] = []
    pseudo_num_predictions: list[int] = []
    for granularity, record_sets in pseudo_records_by_gran.items():
        if not record_sets:
            continue
        records = merge_ap_record_sets(record_sets)
        metrics = evaluate_official_and_oracle_ap(records)
        _records_metric_row(f"pseudo_{granularity}_official", metrics, row)
        _topk_metric_row(
            f"pseudo_{granularity}_official",
            records,
            row,
            topk_values=topk_values,
        )
        pseudo_ap.append(float(metrics.get("AP", float("nan"))))
        pseudo_ap50.append(float(metrics.get("AP50", float("nan"))))
        pseudo_ap25.append(float(metrics.get("AP25", float("nan"))))
        pseudo_oracle_ap50.append(float(metrics.get("oracle_AP50", float("nan"))))
        pseudo_total_gt.append(int(metrics.get("total_gt_instances", 0) or 0))
        pseudo_num_predictions.append(int(metrics.get("num_predictions", 0) or 0))

    row["pseudo_AP"] = _mean(pseudo_ap)
    row["pseudo_AP50"] = _mean(pseudo_ap50)
    row["pseudo_AP25"] = _mean(pseudo_ap25)
    row["pseudo_oracle_AP50"] = _mean(pseudo_oracle_ap50)
    row["pseudo_official_total_gt"] = pseudo_total_gt[0] if len(pseudo_total_gt) == 1 else int(sum(pseudo_total_gt))
    row["pseudo_official_num_predictions"] = (
        pseudo_num_predictions[0] if len(pseudo_num_predictions) == 1 else int(sum(pseudo_num_predictions))
    )
    row["pseudo_official_AP_mean"] = row["pseudo_AP"]
    row["pseudo_official_AP50_mean"] = row["pseudo_AP50"]
    row["pseudo_official_AP25_mean"] = row["pseudo_AP25"]
    row["pseudo_oracle_AP50_mean"] = row["pseudo_oracle_AP50"]
    row["pseudo_matched_recall25_mean"] = _mean(pseudo_recall25)
    row["pseudo_matched_recall50_mean"] = _mean(pseudo_recall50)
    row["pseudo_matched_mean_iou"] = _mean(pseudo_iou)

    primary_real_metrics: dict[str, Any] | None = None
    for benchmark, by_gran in sorted(real_records_by_bench_gran.items()):
        real_ap: list[float] = []
        real_ap50: list[float] = []
        real_ap25: list[float] = []
        real_oracle_ap50: list[float] = []
        real_total_gt: list[int] = []
        real_num_predictions: list[int] = []
        for granularity in granularities:
            record_sets = by_gran.get(granularity, [])
            if not record_sets:
                continue
            records = merge_ap_record_sets(record_sets)
            metrics = evaluate_official_and_oracle_ap(records)
            prefix = f"real_{granularity}_{row['eval_scope']}_official_{benchmark}"
            _records_metric_row(prefix, metrics, row)
            _topk_metric_row(
                prefix,
                records,
                row,
                topk_values=topk_values,
            )
            row[f"real_{granularity}_{row['eval_scope']}_official_AP_{benchmark}"] = _finite_or_none(
                metrics.get("AP")
            )
            row[f"real_{granularity}_{row['eval_scope']}_official_AP50_{benchmark}"] = _finite_or_none(
                metrics.get("AP50")
            )
            row[f"real_{granularity}_{row['eval_scope']}_official_AP25_{benchmark}"] = _finite_or_none(
                metrics.get("AP25")
            )
            row[f"real_{granularity}_{row['eval_scope']}_oracle_AP50_{benchmark}"] = _finite_or_none(
                metrics.get("oracle_AP50")
            )
            row[f"real_{granularity}_{row['eval_scope']}_official_total_gt_{benchmark}"] = int(
                metrics.get("total_gt_instances", 0) or 0
            )
            row[f"real_{granularity}_{row['eval_scope']}_official_num_predictions_{benchmark}"] = int(
                metrics.get("num_predictions", 0) or 0
            )
            real_ap.append(float(metrics.get("AP", float("nan"))))
            real_ap50.append(float(metrics.get("AP50", float("nan"))))
            real_ap25.append(float(metrics.get("AP25", float("nan"))))
            real_oracle_ap50.append(float(metrics.get("oracle_AP50", float("nan"))))
            real_total_gt.append(int(metrics.get("total_gt_instances", 0) or 0))
            real_num_predictions.append(int(metrics.get("num_predictions", 0) or 0))

        bench_prefix = f"real_{row['eval_scope']}_official_{benchmark}"
        row[f"{bench_prefix}_AP"] = _mean(real_ap)
        row[f"{bench_prefix}_AP50"] = _mean(real_ap50)
        row[f"{bench_prefix}_AP25"] = _mean(real_ap25)
        row[f"real_{row['eval_scope']}_oracle_AP50_{benchmark}"] = _mean(real_oracle_ap50)
        row[f"{bench_prefix}_total_gt"] = real_total_gt[0] if len(real_total_gt) == 1 else int(sum(real_total_gt))
        row[f"{bench_prefix}_num_predictions"] = (
            real_num_predictions[0] if len(real_num_predictions) == 1 else int(sum(real_num_predictions))
        )
        row[f"real_{row['eval_scope']}_official_AP_{benchmark}"] = row[f"{bench_prefix}_AP"]
        row[f"real_{row['eval_scope']}_official_AP50_{benchmark}"] = row[f"{bench_prefix}_AP50"]
        row[f"real_{row['eval_scope']}_official_AP25_{benchmark}"] = row[f"{bench_prefix}_AP25"]
        row[f"real_{row['eval_scope']}_official_total_gt_{benchmark}"] = row[f"{bench_prefix}_total_gt"]
        row[f"real_{row['eval_scope']}_official_num_predictions_{benchmark}"] = row[
            f"{bench_prefix}_num_predictions"
        ]
        row[f"real_matched_recall25_mean_{benchmark}"] = _mean(real_recall25_by_bench.get(benchmark, []))
        row[f"real_matched_recall50_mean_{benchmark}"] = _mean(real_recall50_by_bench.get(benchmark, []))
        row[f"real_matched_mean_iou_{benchmark}"] = _mean(real_iou_by_bench.get(benchmark, []))

        if benchmark == primary_benchmark:
            primary_real_metrics = {
                "AP": row[f"{bench_prefix}_AP"],
                "AP50": row[f"{bench_prefix}_AP50"],
                "AP25": row[f"{bench_prefix}_AP25"],
                "oracle_AP50": row[f"real_{row['eval_scope']}_oracle_AP50_{benchmark}"],
                "total_gt": row[f"{bench_prefix}_total_gt"],
                "num_predictions": row[f"{bench_prefix}_num_predictions"],
                "recall25": row[f"real_matched_recall25_mean_{benchmark}"],
                "recall50": row[f"real_matched_recall50_mean_{benchmark}"],
                "matched_iou": row[f"real_matched_mean_iou_{benchmark}"],
            }

    if primary_real_metrics is not None:
        row["real_AP"] = primary_real_metrics["AP"]
        row["real_AP50"] = primary_real_metrics["AP50"]
        row["real_AP25"] = primary_real_metrics["AP25"]
        row["real_oracle_AP50"] = primary_real_metrics["oracle_AP50"]
        row["real_official_total_gt"] = primary_real_metrics["total_gt"]
        row["real_official_num_predictions"] = primary_real_metrics["num_predictions"]
        row["real_matched_recall25_mean"] = primary_real_metrics["recall25"]
        row["real_matched_recall50_mean"] = primary_real_metrics["recall50"]
        row["real_matched_mean_iou"] = primary_real_metrics["matched_iou"]
        row["official_total_gt"] = primary_real_metrics["total_gt"]
        row["official_num_predictions"] = primary_real_metrics["num_predictions"]
    else:
        row["real_AP"] = None
        row["real_AP50"] = None
        row["real_AP25"] = None
        row["real_oracle_AP50"] = None
        row["real_official_total_gt"] = 0
        row["real_official_num_predictions"] = 0
        row["official_total_gt"] = row["pseudo_official_total_gt"]
        row["official_num_predictions"] = row["pseudo_official_num_predictions"]

    return row


def sweep_thresholds_from_cache(
    cache: dict[str, Any],
    thresholds: Iterable[float],
    *,
    primary_benchmark: str | None = None,
    topk_values: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Recompute all requested thresholds from an existing prediction cache."""
    return [
        evaluate_threshold_from_cache(
            cache,
            score_threshold=float(threshold),
            primary_benchmark=primary_benchmark,
            topk_values=topk_values,
        )
        for threshold in parse_thresholds(thresholds)
    ]
