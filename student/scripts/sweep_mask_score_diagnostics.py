#!/usr/bin/env python3
"""Joint mask-threshold and score-mode diagnostics for one checkpoint.

The script runs model inference once per scene/granularity, then recomputes AP
for multiple mask thresholds, proposal size thresholds, and score definitions.
It is evaluation-only: no optimizer is built and model weights are never
modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.metrics.eval_diagnostics import safe_pearson, safe_spearman
from student.metrics.official_instance_ap import (
    SCANNET_MIN_REGION_SIZE,
    build_instance_ap_records,
    evaluate_official_and_oracle_ap,
    merge_ap_record_sets,
)

log = logging.getLogger("sweep_mask_score_diagnostics")

DEFAULT_MASK_THRESHOLDS = "0.15,0.20,0.30,0.40,0.50"
DEFAULT_SCORE_MODES = "learned,ones,mask_area,mean_mask_prob,oracle"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _parse_csv_floats(value: str | Iterable[float]) -> list[float]:
    raw = [float(x.strip()) for x in str(value).split(",") if x.strip()] if isinstance(value, str) else [float(x) for x in value]
    out: list[float] = []
    seen: set[float] = set()
    for item in raw:
        key = round(float(item), 12)
        if key not in seen:
            seen.add(key)
            out.append(float(item))
    return out


def _parse_csv_ints(value: str | Iterable[int]) -> list[int]:
    raw = [int(x.strip()) for x in str(value).split(",") if x.strip()] if isinstance(value, str) else [int(x) for x in value]
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _to_granularity_key(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("Empty granularity value")
    if value.startswith("g"):
        return value.replace(".", "")
    return f"g{value}".replace(".", "")


def _parse_granularities(value: str | None, available: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return available
    requested = tuple(_to_granularity_key(v) for v in value.split(",") if v.strip())
    missing = [g for g in requested if g not in available]
    if missing:
        raise ValueError(f"Requested granularities {missing} not in config granularities {available}")
    return requested


def _parse_score_modes(value: str) -> tuple[str, ...]:
    allowed = {"learned", "ones", "mask_area", "mean_mask_prob", "oracle"}
    modes = tuple(x.strip() for x in value.split(",") if x.strip())
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unknown score mode(s) {bad}; allowed={sorted(allowed)}")
    return modes


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> float | None:
    vals: list[float] = []
    for value in values:
        f = _finite_or_none(value)
        if f is not None:
            vals.append(f)
    return float(sum(vals) / len(vals)) if vals else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(_jsonable(row), f, sort_keys=True, allow_nan=False)
            f.write("\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "granularity",
        "score_mode",
        "mask_threshold",
        "min_points",
        "num_scene_heads",
        "mean_kept_proposals",
        "mean_removed_min_points",
        "pseudo_AP50",
        "pseudo_oracle_AP50",
        "pseudo_matched_mIoU",
        "real_scannet20_AP50",
        "real_scannet20_oracle_AP50",
        "real_scannet20_matched_mIoU",
        "real_scannet200_AP50",
        "real_scannet200_oracle_AP50",
        "real_scannet200_matched_mIoU",
    ]
    fields: list[str] = []
    seen: set[str] = set()
    for key in preferred + sorted({key for row in rows for key in row}):
        if key not in seen:
            fields.append(key)
            seen.add(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})
    tmp.replace(path)


def _parse_cuda_index(device: str) -> int | None:
    device = device.strip()
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _resolve_runtime_device(requested_device: str) -> str:
    import torch

    device = requested_device
    idx = _parse_cuda_index(device)
    if idx is not None and idx != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested_device}) but CUDA is unavailable")
    return device


def _maybe_apply_known_cluster_fallbacks(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    exp_cfg = cfg.setdefault("experiment", {})
    bb_cfg = model_cfg.setdefault("backbone", {})
    for section, key, fallback in [
        (data_cfg, "scans_root", Path("/cluster/work/igp_psr/nedela/chorus_poc/scans")),
        (bb_cfg, "litept_root", Path("/cluster/work/igp_psr/nedela/LitePT")),
        (exp_cfg, "output_root", Path("/cluster/work/igp_psr/nedela/student_runs")),
    ]:
        current = section.get(key)
        if current and Path(str(current)).exists():
            continue
        if fallback.exists():
            log.warning("%s=%s does not exist; using %s", key, current, fallback)
            section[key] = str(fallback)
    metadata_root = Path("/cluster/work/igp_psr/nedela/LitePT/datasets/preprocessing/scannet/meta_data")
    if "CHORUS_SCANNET_METADATA_ROOT" not in os.environ and (metadata_root / "scannetv2-labels.combined.tsv").exists():
        os.environ["CHORUS_SCANNET_METADATA_ROOT"] = str(metadata_root)
        log.warning("CHORUS_SCANNET_METADATA_ROOT not set; using %s", metadata_root)


def _build_val_dataset(cfg: dict[str, Any], granularities: tuple[str, ...]) -> MultiSceneDataset:
    from student.data.eval_sampling import resolve_eval_sampling_config
    from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list

    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    val_dirs = build_scene_list(_STUDENT_ROOT / data_cfg["val_split"], Path(data_cfg["scans_root"]))
    eval_sampling = resolve_eval_sampling_config(data_cfg, eval_cfg)
    return MultiSceneDataset(
        val_dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=data_cfg.get("preload", True),
        max_points=eval_sampling["max_points"],
        subsampling_mode=eval_sampling["subsampling_mode"],
        sphere_point_max=eval_sampling["sphere_point_max"],
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(data_cfg.get("scannet_gt_supervise_all_points", False)),
    )


def _build_model(cfg: dict[str, Any], granularities: tuple[str, ...], device: str) -> torch.nn.Module:
    from student.models.finetune_wrapper import FineTuningWrapper
    from student.models.student_model import build_student_model

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = model_cfg["backbone"]
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        bb_cfg["in_channels"] = 9 if bool(data_cfg.get("append_xyz_to_features", False)) else 6

    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    model = build_student_model(
        litept_root=bb_cfg["litept_root"],
        in_channels=bb_cfg.get("in_channels", 3),
        grid_size=bb_cfg.get("grid_size", 0.02),
        litept_variant=bb_cfg.get("litept_variant", "litept_s_star"),
        litept_kwargs=bb_cfg.get("litept_kwargs", None),
        hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        num_queries=num_queries,
        num_queries_by_granularity=num_queries_by_granularity,
        granularities=granularities,
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        num_decoder_heads=model_cfg.get("num_decoder_heads", 8),
        query_init=model_cfg.get("query_init", "hybrid"),
        use_positional_guidance=model_cfg.get("use_positional_guidance", True),
        learned_query_ratio=model_cfg.get("learned_query_ratio", 0.25),
        multi_scale=bb_cfg.get("multi_scale", False),
        multi_scale_indices=bb_cfg.get("multi_scale_indices", None),
        decoder_type=str(model_cfg.get("decoder_type", "multi_head")),
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
        continuous_decoder_v2=model_cfg.get("continuous_decoder_v2", None),
    )

    prompt_cfg = train_cfg.get("prompt_finetune", {})
    prompt_enabled = bool(prompt_cfg.get("enabled", False)) if isinstance(prompt_cfg, dict) else bool(prompt_cfg)
    if prompt_enabled:
        prompt_cfg = prompt_cfg if isinstance(prompt_cfg, dict) else {}
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))),
            mode=str(prompt_cfg.get("mode", "learned")),
        )
    model.to(device)
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict[str, Any]:
    import torch

    from student.models.finetune_wrapper import FineTuningWrapper

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} missing model_state_dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if isinstance(model, FineTuningWrapper) and not any(k.startswith("model.") or k == "g_ft_logit" for k in state):
        model.model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return checkpoint


def build_thresholded_proposals(
    mask_probs: np.ndarray,
    *,
    mask_threshold: float,
    min_points: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Return masks surviving the mask threshold and proposal size threshold."""
    probs = np.asarray(mask_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"mask_probs must be [Q, N], got shape {probs.shape}")
    masks_binary = probs >= float(mask_threshold)
    counts = masks_binary.sum(axis=1).astype(np.int64, copy=False)
    keep = counts >= int(min_points)
    keep_idx = np.flatnonzero(keep).astype(np.int64, copy=False)
    proposals = [masks_binary[int(i)].astype(bool, copy=False) for i in keep_idx]
    stats = {
        "num_queries": int(probs.shape[0]),
        "num_points": int(probs.shape[1]),
        "num_score_pass": int(probs.shape[0]),
        "num_min_points_removed": int((~keep).sum()),
        "num_proposals": int(keep_idx.size),
        "mean_mask_area": float(np.mean(counts[keep] / max(probs.shape[1], 1))) if keep_idx.size else 0.0,
        "median_mask_area": float(np.median(counts[keep] / max(probs.shape[1], 1))) if keep_idx.size else 0.0,
        "mean_mask_point_count": float(np.mean(counts[keep])) if keep_idx.size else 0.0,
    }
    return proposals, keep_idx, counts, stats


def scores_for_mode(
    *,
    mode: str,
    learned_scores: np.ndarray,
    mask_probs: np.ndarray,
    keep_indices: np.ndarray,
    mask_counts: np.ndarray,
    proposals: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute proposal scores for target-independent diagnostic modes."""
    mode = str(mode)
    keep = np.asarray(keep_indices, dtype=np.int64)
    if mode == "learned":
        return np.asarray(learned_scores, dtype=np.float64)[keep]
    if mode == "ones":
        return np.ones(keep.shape[0], dtype=np.float64)
    if mode == "mask_area":
        return np.asarray(mask_counts, dtype=np.float64)[keep] / max(int(mask_probs.shape[1]), 1)
    if mode == "mean_mask_prob":
        if proposals is None:
            raise ValueError("mean_mask_prob scoring requires thresholded proposals")
        out: list[float] = []
        for q, support in zip(keep, proposals):
            vals = mask_probs[int(q)][support]
            out.append(float(vals.mean()) if vals.size else 0.0)
        return np.asarray(out, dtype=np.float64)
    if mode == "oracle":
        return np.zeros(keep.shape[0], dtype=np.float64)
    raise ValueError(f"Unknown score mode: {mode}")


def _mean_mask_prob_scores(mask_probs: np.ndarray, proposals: list[np.ndarray], keep_indices: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for mask, q in zip(proposals, keep_indices):
        vals = mask_probs[int(q)][mask]
        values.append(float(vals.mean()) if vals.size else 0.0)
    return np.asarray(values, dtype=np.float64)


def _records_with_oracle_as_score(records: dict[str, Any]) -> dict[str, Any]:
    out = {
        "predictions": [dict(pred, score=float(pred.get("oracle_score", 0.0))) for pred in records.get("predictions", [])],
        "ground_truths": list(records.get("ground_truths", []) or []),
        "num_predictions": int(records.get("num_predictions", 0) or 0),
        "total_gt_instances": int(records.get("total_gt_instances", 0) or 0),
        "granularity": records.get("granularity"),
        "min_valid_gt_points": records.get("min_valid_gt_points"),
        "min_valid_pred_points": records.get("min_valid_pred_points"),
    }
    return out


def _score_oracle_correlation(records: dict[str, Any]) -> dict[str, float | None]:
    predictions = list(records.get("predictions", []) or [])
    scores = [float(p.get("score", 0.0)) for p in predictions]
    oracle = [float(p.get("oracle_score", 0.0)) for p in predictions]
    return {
        "score_oracle_pearson": safe_pearson(scores, oracle),
        "score_oracle_spearman": safe_spearman(scores, oracle),
    }


def _metric_row(prefix: str, metrics: dict[str, Any], row: dict[str, Any]) -> None:
    row[f"{prefix}_AP"] = _finite_or_none(metrics.get("AP"))
    row[f"{prefix}_AP50"] = _finite_or_none(metrics.get("AP50"))
    row[f"{prefix}_AP25"] = _finite_or_none(metrics.get("AP25"))
    row[f"{prefix}_oracle_AP50"] = _finite_or_none(metrics.get("oracle_AP50"))
    row[f"{prefix}_oracle_AP25"] = _finite_or_none(metrics.get("oracle_AP25"))
    row[f"{prefix}_num_predictions"] = int(metrics.get("num_predictions", 0) or 0)
    row[f"{prefix}_total_gt"] = int(metrics.get("total_gt_instances", 0) or 0)


def _make_aggregate_entry() -> dict[str, Any]:
    return {
        "pseudo_records": [],
        "real_records": defaultdict(list),
        "stats": defaultdict(list),
    }


def _append_stats(entry: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            entry["stats"][key].append(float(value))


def _summarize_entry(
    *,
    key: tuple[str, float, int, str],
    entry: dict[str, Any],
    eval_benchmarks: list[str],
) -> dict[str, Any]:
    granularity, mask_threshold, min_points, score_mode = key
    row: dict[str, Any] = {
        "granularity": granularity,
        "mask_threshold": float(mask_threshold),
        "min_points": int(min_points),
        "score_mode": score_mode,
    }
    stats = entry["stats"]
    row["num_scene_heads"] = int(len(stats.get("num_queries", [])))
    row["mean_kept_proposals"] = _mean(stats.get("num_proposals", []))
    row["mean_removed_min_points"] = _mean(stats.get("num_min_points_removed", []))
    row["mean_num_queries"] = _mean(stats.get("num_queries", []))
    row["mean_mask_area"] = _mean(stats.get("mean_mask_area", []))
    row["mean_mask_point_count"] = _mean(stats.get("mean_mask_point_count", []))
    row["pseudo_matched_mIoU"] = _mean(stats.get("pseudo_matched_mIoU", []))
    row["pseudo_matched_recall25"] = _mean(stats.get("pseudo_matched_recall25", []))
    row["pseudo_matched_recall50"] = _mean(stats.get("pseudo_matched_recall50", []))

    pseudo_records = merge_ap_record_sets(entry["pseudo_records"])
    pseudo_metrics = evaluate_official_and_oracle_ap(pseudo_records)
    _metric_row("pseudo", pseudo_metrics, row)
    row.update({f"pseudo_{k}": v for k, v in _score_oracle_correlation(pseudo_records).items()})

    for benchmark in eval_benchmarks:
        real_records = merge_ap_record_sets(entry["real_records"].get(benchmark, []))
        real_metrics = evaluate_official_and_oracle_ap(real_records)
        prefix = f"real_{benchmark}"
        _metric_row(prefix, real_metrics, row)
        row[f"{prefix}_matched_mIoU"] = _mean(stats.get(f"{prefix}_matched_mIoU", []))
        row[f"{prefix}_matched_recall25"] = _mean(stats.get(f"{prefix}_matched_recall25", []))
        row[f"{prefix}_matched_recall50"] = _mean(stats.get(f"{prefix}_matched_recall50", []))
        row.update({f"{prefix}_{k}": v for k, v in _score_oracle_correlation(real_records).items()})
    return row


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from student.data.target_builder import build_instance_targets_multi
    from student.engine.evaluator import _build_pseudo_gt_ids, compute_legacy_best_match_recall
    from student.engine.threshold_sweep import (
        _clear_backbone_cache,
        _forward_heads,
        _load_real_gt_for_cache,
        normalize_benchmarks,
        set_eval_mode_like_validation,
    )

    cfg = load_config(args.config)
    _maybe_apply_known_cluster_fallbacks(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    available_grans = parse_granularities(cfg["data"])
    granularities = _parse_granularities(args.granularities, available_grans)
    mask_thresholds = _parse_csv_floats(args.mask_thresholds)
    min_points_values = _parse_csv_ints(args.min_points)
    score_modes = _parse_score_modes(args.score_modes)
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    eval_benchmarks = normalize_benchmarks(args.benchmarks or eval_cfg.get("scannet_benchmarks", eval_cfg.get("scannet_benchmark", "scannet200")))
    device = _resolve_runtime_device(args.device or train_cfg.get("device", "cuda:0"))

    dataset = _build_val_dataset(cfg, available_grans)
    model = _build_model(cfg, available_grans, device)
    checkpoint = _load_checkpoint(model, Path(args.checkpoint), device)
    set_eval_mode_like_validation(model)

    prompt_cfg = train_cfg.get("prompt_finetune", {})
    prompt_finetune = bool(prompt_cfg.get("enabled", False)) if isinstance(prompt_cfg, dict) else bool(prompt_cfg)
    scene_limit = len(dataset) if args.max_scenes is None else min(len(dataset), max(int(args.max_scenes), 0))
    min_instance_points = int(data_cfg.get("min_instance_points", 10))
    dense_instance_ids = bool(data_cfg.get("dense_instance_ids", False))

    aggregates: dict[tuple[str, float, int, str], dict[str, Any]] = defaultdict(_make_aggregate_entry)
    per_scene_rows: list[dict[str, Any]] = []

    for idx in range(scene_limit):
        sample = dataset[idx]
        scene_id = str(sample["scene_id"])
        log.info("[%d/%d] %s points=%d", idx + 1, scene_limit, scene_id, int(sample["points"].shape[0]))
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
            heads = _forward_heads(
                model,
                points,
                features,
                granularities=granularities,
                prompt_finetune=prompt_finetune,
                prompt_target_granularity=granularities[0] if prompt_finetune else None,
            )

        real_gt_by_benchmark: dict[str, np.ndarray] = {}
        expected_points = int(sample["points"].shape[0])
        for benchmark in eval_benchmarks:
            real_gt_by_benchmark[benchmark] = _load_real_gt_for_cache(
                scene_dir=sample["scene_dir"],
                scene_id=scene_id,
                benchmark=benchmark,
                vertex_indices=sample.get("vertex_indices"),
                expected_points=expected_points,
            ).astype(np.int64, copy=False)

        for granularity in granularities:
            pred = heads[granularity]
            mask_probs = pred["mask_logits"].detach().sigmoid().cpu().numpy().astype(np.float32, copy=False)
            learned_scores = pred["score_logits"].detach().sigmoid().cpu().numpy().astype(np.float64, copy=False)
            targets = targets_by_gran[granularity]
            pseudo_gt = _build_pseudo_gt_ids(targets)
            pseudo_eval_mask = targets.supervision_mask.detach().cpu().numpy().astype(bool, copy=False)

            for mask_threshold in mask_thresholds:
                for min_points in min_points_values:
                    proposals, keep_idx, mask_counts, proposal_stats = build_thresholded_proposals(
                        mask_probs,
                        mask_threshold=mask_threshold,
                        min_points=min_points,
                    )
                    for score_mode in score_modes:
                        if score_mode == "mean_mask_prob":
                            base_scores = _mean_mask_prob_scores(mask_probs, proposals, keep_idx)
                        else:
                            base_scores = scores_for_mode(
                                mode=score_mode,
                                learned_scores=learned_scores,
                                mask_probs=mask_probs,
                                keep_indices=keep_idx,
                                mask_counts=mask_counts,
                            )
                        key = (granularity, float(mask_threshold), int(min_points), score_mode)
                        entry = aggregates[key]
                        stat_payload = dict(proposal_stats)

                        pseudo_records = build_instance_ap_records(
                            scene_id=scene_id,
                            gt_ids=pseudo_gt,
                            proposals=proposals,
                            scores=base_scores,
                            query_indices=keep_idx,
                            granularity=granularity,
                            class_agnostic=True,
                            eval_mask=pseudo_eval_mask,
                        )
                        if score_mode == "oracle":
                            pseudo_records = _records_with_oracle_as_score(pseudo_records)
                        pseudo_metrics = evaluate_official_and_oracle_ap(pseudo_records)
                        pseudo_legacy = compute_legacy_best_match_recall(pseudo_gt, proposals)
                        stat_payload["pseudo_matched_mIoU"] = float(pseudo_legacy["matched_mean_iou"])
                        stat_payload["pseudo_matched_recall25"] = float(pseudo_legacy["legacy_matched_recall25"])
                        stat_payload["pseudo_matched_recall50"] = float(pseudo_legacy["legacy_matched_recall50"])
                        entry["pseudo_records"].append(pseudo_records)

                        scene_row: dict[str, Any] = {
                            "scene_id": scene_id,
                            "granularity": granularity,
                            "mask_threshold": float(mask_threshold),
                            "min_points": int(min_points),
                            "score_mode": score_mode,
                            "num_queries": int(proposal_stats["num_queries"]),
                            "kept_proposals": int(proposal_stats["num_proposals"]),
                            "removed_min_points": int(proposal_stats["num_min_points_removed"]),
                            "mean_mask_area": float(proposal_stats["mean_mask_area"]),
                            "pseudo_AP50": _finite_or_none(pseudo_metrics.get("AP50")),
                            "pseudo_oracle_AP50": _finite_or_none(pseudo_metrics.get("oracle_AP50")),
                            "pseudo_matched_mIoU": float(pseudo_legacy["matched_mean_iou"]),
                        }

                        for benchmark, real_gt in real_gt_by_benchmark.items():
                            real_records = build_instance_ap_records(
                                scene_id=scene_id,
                                gt_ids=real_gt,
                                proposals=proposals,
                                scores=base_scores,
                                query_indices=keep_idx,
                                granularity=granularity,
                                class_agnostic=True,
                                min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                                min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                            )
                            if score_mode == "oracle":
                                real_records = _records_with_oracle_as_score(real_records)
                            real_metrics = evaluate_official_and_oracle_ap(real_records)
                            real_legacy = compute_legacy_best_match_recall(real_gt, proposals)
                            entry["real_records"][benchmark].append(real_records)
                            stat_payload[f"real_{benchmark}_matched_mIoU"] = float(real_legacy["matched_mean_iou"])
                            stat_payload[f"real_{benchmark}_matched_recall25"] = float(real_legacy["legacy_matched_recall25"])
                            stat_payload[f"real_{benchmark}_matched_recall50"] = float(real_legacy["legacy_matched_recall50"])
                            scene_row[f"real_{benchmark}_AP50"] = _finite_or_none(real_metrics.get("AP50"))
                            scene_row[f"real_{benchmark}_oracle_AP50"] = _finite_or_none(real_metrics.get("oracle_AP50"))
                            scene_row[f"real_{benchmark}_matched_mIoU"] = float(real_legacy["matched_mean_iou"])

                        _append_stats(entry, stat_payload)
                        per_scene_rows.append(scene_row)

    aggregate_rows = [
        _summarize_entry(key=key, entry=entry, eval_benchmarks=eval_benchmarks)
        for key, entry in sorted(aggregates.items())
    ]
    summary = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0) or 0),
        "granularities": list(granularities),
        "mask_thresholds": mask_thresholds,
        "min_points": min_points_values,
        "score_modes": list(score_modes),
        "eval_benchmarks": eval_benchmarks,
        "max_scenes": args.max_scenes,
        "num_scenes_evaluated": scene_limit,
        "rows": aggregate_rows,
    }
    return {
        "summary": summary,
        "aggregate_rows": aggregate_rows,
        "per_scene_rows": per_scene_rows,
    }


def _write_markdown(output_dir: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    def best(metric: str) -> dict[str, Any] | None:
        usable = [row for row in aggregate_rows if isinstance(row.get(metric), (int, float))]
        return max(usable, key=lambda row: float(row[metric])) if usable else None

    lines = [
        "# Mask/Score Diagnostic Sweep",
        "",
        "This report is generated by `scripts/sweep_mask_score_diagnostics.py`.",
        "",
    ]
    for metric in ["pseudo_AP50", "pseudo_oracle_AP50", "real_scannet20_AP50", "real_scannet20_oracle_AP50"]:
        row = best(metric)
        if row is None:
            continue
        lines.append(
            f"- Best `{metric}` = `{float(row[metric]):.4f}` at "
            f"`granularity={row['granularity']}`, `mask_threshold={row['mask_threshold']}`, "
            f"`min_points={row['min_points']}`, `score_mode={row['score_mode']}`."
        )
    lines.extend(["", "See `aggregate.csv`, `aggregate.json`, and `per_scene.jsonl` for complete results.", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mask thresholds and score modes for one checkpoint")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--granularities", default=None, type=str)
    parser.add_argument("--benchmarks", default=None, type=str)
    parser.add_argument("--mask-thresholds", default=DEFAULT_MASK_THRESHOLDS, type=str)
    parser.add_argument("--min-points", default="10,30", type=str)
    parser.add_argument("--score-modes", default=DEFAULT_SCORE_MODES, type=str)
    parser.add_argument("--max-scenes", default=None, type=int)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    _configure_logging()
    output_dir = Path(args.output_dir)
    result = run_diagnostics(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", result["summary"])
    _write_json(output_dir / "aggregate.json", result["aggregate_rows"])
    _write_jsonl(output_dir / "per_scene.jsonl", result["per_scene_rows"])
    _write_csv(output_dir / "aggregate.csv", result["aggregate_rows"])
    _write_csv(output_dir / "per_scene.csv", result["per_scene_rows"])
    _write_markdown(output_dir, result["aggregate_rows"])
    log.info("Wrote mask/score diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
