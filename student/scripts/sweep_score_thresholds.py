#!/usr/bin/env python3
"""Sweep score thresholds from one cached validation inference pass.

This is eval-only: it builds no optimizer, calls no training loop, and never
calls ``optimizer.step()``.  Cached masks are pre-score-threshold and packed on
CPU so AP can be recomputed for arbitrary score thresholds without rerunning
fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.engine.threshold_sweep import (
    build_prediction_cache,
    load_or_create_prediction_cache,
    normalize_benchmarks,
    parse_thresholds,
    sweep_thresholds_from_cache,
)
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model

log = logging.getLogger("sweep_score_thresholds")

DEFAULT_THRESHOLDS = "0.0,0.001,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


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
    device = requested_device
    idx = _parse_cuda_index(device)
    if idx is not None and idx != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested_device}) but CUDA is unavailable")
    return device


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _maybe_apply_known_cluster_fallbacks(cfg: dict[str, Any]) -> None:
    """Keep the standalone command usable when sbatch env vars are absent."""
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    exp_cfg = cfg.setdefault("experiment", {})
    bb_cfg = model_cfg.setdefault("backbone", {})

    path_fallbacks = [
        (data_cfg, "scans_root", Path("/cluster/work/igp_psr/nedela/chorus_poc/scans")),
        (bb_cfg, "litept_root", Path("/cluster/work/igp_psr/nedela/LitePT")),
        (exp_cfg, "output_root", Path("/cluster/work/igp_psr/nedela/student_runs")),
    ]
    for section, key, candidate in path_fallbacks:
        current = section.get(key)
        if current and Path(str(current)).exists():
            continue
        if candidate.exists():
            log.warning("%s=%s does not exist; using %s", key, current, candidate)
            section[key] = str(candidate)


def _build_output_dir(cfg: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        root = Path(cfg["experiment"]["output_root"])
        name = cfg["experiment"].get("name", "multi_scene")
        out = root / name / "threshold_sweep"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_val_dataset(cfg: dict[str, Any], granularities: tuple[str, ...]) -> MultiSceneDataset:
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    scans_root = Path(data_cfg["scans_root"])
    val_split = _STUDENT_ROOT / data_cfg["val_split"]
    val_dirs = build_scene_list(val_split, scans_root)
    return MultiSceneDataset(
        val_dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=data_cfg.get("preload", True),
        max_points=data_cfg.get("val_max_points", None),
        subsampling_mode=data_cfg.get("val_subsampling_mode", data_cfg.get("subsampling_mode", "sphere_crop")),
        sphere_point_max=data_cfg.get("val_sphere_point_max", data_cfg.get("sphere_point_max", None)),
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(data_cfg.get("scannet_gt_supervise_all_points", False)),
    )


def _build_model(cfg: dict[str, Any], granularities: tuple[str, ...], device: str) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = model_cfg["backbone"]

    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        append_xyz = bool(data_cfg.get("append_xyz_to_features", False))
        bb_cfg["in_channels"] = 9 if append_xyz else 6
        log.info("use_normals=True: model.backbone.in_channels=%d (auto)", bb_cfg["in_channels"])

    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    decoder_type = str(model_cfg.get("decoder_type", "multi_head"))
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
        decoder_type=decoder_type,
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
        continuous_decoder_v2=model_cfg.get("continuous_decoder_v2", None),
    )

    prompt_ft_cfg = train_cfg.get("prompt_finetune", {})
    if isinstance(prompt_ft_cfg, bool):
        prompt_enabled = prompt_ft_cfg
        prompt_ft_cfg = {"enabled": prompt_enabled}
    elif isinstance(prompt_ft_cfg, dict):
        prompt_enabled = bool(prompt_ft_cfg.get("enabled", False))
    else:
        prompt_enabled = False
    if prompt_enabled:
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_ft_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_ft_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))),
            mode=str(prompt_ft_cfg.get("mode", "learned")),
        )
        log.info("Prompt fine-tuning wrapper enabled")

    model.to(device)
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} missing model_state_dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    loading_base_into_prompt = (
        isinstance(model, FineTuningWrapper)
        and not any(key.startswith("model.") or key == "g_ft_logit" for key in state)
    )
    if loading_base_into_prompt:
        model.model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return checkpoint


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(_jsonable(row), f, sort_keys=True, allow_nan=False)
            f.write("\n")
    tmp_path.replace(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "score_threshold",
        "real_AP",
        "real_AP50",
        "real_AP25",
        "real_oracle_AP50",
        "pseudo_AP",
        "pseudo_AP50",
        "pseudo_AP25",
        "pseudo_oracle_AP50",
        "official_total_gt",
        "official_num_predictions",
        "mean_kept_per_scene",
        "mean_score_pass",
        "mean_removed_min_points",
        "mean_kept_after_min_points",
        "total_predictions",
    ]
    for key in preferred + sorted({k for row in rows for k in row}):
        if key not in seen:
            seen.add(key)
            fields.append(key)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in fields})
    tmp_path.replace(path)


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    usable = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not usable:
        return None
    return max(usable, key=lambda row: float(row[key]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only ScanNet score-threshold sweep")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, type=str)
    parser.add_argument("--topk-values", default="1,5,10,25,50,100,150,250", type=str)
    parser.add_argument("--cache-path", default=None, type=str)
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Write the pre-score-threshold prediction cache after inference.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Use --cache-path if it exists and skip model forward.",
    )
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--output-jsonl", default=None, type=str)
    parser.add_argument("--output-csv", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--max-scenes", default=None, type=int)
    parser.add_argument("--benchmark", default=None, type=str, help="Primary benchmark alias for real_AP columns")
    parser.add_argument("--no-wandb", action="store_true", help="Accepted for parity; this script does not use wandb")
    args = parser.parse_args()

    _configure_logging()
    cfg = load_config(args.config)
    _maybe_apply_known_cluster_fallbacks(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    thresholds = parse_thresholds(args.thresholds)
    topk_values = _parse_csv_ints(args.topk_values)
    granularities = parse_granularities(data_cfg)
    eval_benchmarks = normalize_benchmarks(eval_cfg.get("scannet_benchmarks", eval_cfg.get("scannet_benchmark", "scannet200")))
    primary_benchmark = args.benchmark or eval_benchmarks[0]
    device = _resolve_runtime_device(args.device or train_cfg.get("device", "cuda:0"))
    output_dir = _build_output_dir(cfg, args.output_dir)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "val_predictions_pre_score_threshold.pt"
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else output_dir / "threshold_sweep.jsonl"
    csv_path = Path(args.output_csv) if args.output_csv else output_dir / "threshold_sweep.csv"

    mask_threshold = float(eval_cfg.get("mask_threshold", 0.5))
    min_points = int(eval_cfg.get("min_points_per_proposal", 30))
    prompt_ft_cfg = train_cfg.get("prompt_finetune", {})
    prompt_finetune = bool(prompt_ft_cfg.get("enabled", False)) if isinstance(prompt_ft_cfg, dict) else bool(prompt_ft_cfg)

    def _build_cache() -> dict[str, Any]:
        log.info("Cache miss: running validation inference once from %s", args.checkpoint)
        dataset = _build_val_dataset(cfg, granularities)
        model = _build_model(cfg, granularities, device)
        checkpoint = _load_checkpoint(model, Path(args.checkpoint), device)
        checkpoint_info = {
            "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
            "checkpoint_global_step": int(checkpoint.get("global_step", 0) or 0),
        }
        return build_prediction_cache(
            model=model,
            dataset=dataset,
            device=device,
            granularities=granularities,
            eval_benchmarks=eval_benchmarks,
            mask_threshold=mask_threshold,
            min_points=min_points,
            min_instance_points=int(data_cfg.get("min_instance_points", 10)),
            dense_instance_ids=bool(data_cfg.get("dense_instance_ids", False)),
            prompt_finetune=prompt_finetune,
            prompt_target_granularity=granularities[0] if prompt_finetune else None,
            max_scenes=args.max_scenes,
            config_path=str(Path(args.config)),
            checkpoint_path=str(Path(args.checkpoint)),
            checkpoint_info=checkpoint_info,
        )

    write_cache = bool(args.write_cache or not (args.reuse_cache and cache_path.exists()))
    cache, reused = load_or_create_prediction_cache(
        cache_path=cache_path,
        reuse_cache=bool(args.reuse_cache),
        write_cache=write_cache,
        build_cache_fn=_build_cache,
    )
    if reused:
        log.info("Reused prediction cache: %s", cache_path)
    elif write_cache:
        log.info("Wrote prediction cache: %s", cache_path)

    log.info(
        "Sweeping %d thresholds over %d cached scene(s): %s",
        len(thresholds),
        len(cache.get("scenes", []) or []),
        thresholds,
    )
    rows = sweep_thresholds_from_cache(
        cache,
        thresholds,
        primary_benchmark=primary_benchmark,
        topk_values=topk_values,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows, jsonl_path)
    _write_csv(rows, csv_path)

    summary = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "cache_path": str(cache_path),
        "cache_reused": reused,
        "cache_is_pre_score_threshold": bool(cache.get("cache_is_pre_score_threshold", False)),
        "mask_representation": cache.get("mask_representation"),
        "thresholds": thresholds,
        "topk_values": topk_values,
        "primary_benchmark": primary_benchmark,
        "eval_benchmarks": eval_benchmarks,
        "granularities": list(granularities),
        "config_score_threshold": float(eval_cfg.get("score_threshold", 0.3)),
        "normal_validation_official_score_threshold": float(
            cache.get("normal_validation_official_score_threshold", 0.0)
        ),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "best_real_AP50": _best_row(rows, "real_AP50"),
        "best_real_AP": _best_row(rows, "real_AP"),
        "rows": rows,
    }
    summary_path = output_dir / "threshold_sweep_summary.json"
    tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp_summary.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp_summary.replace(summary_path)

    best_ap50 = summary["best_real_AP50"]
    if best_ap50 is not None:
        log.info(
            "Wrote %s and %s. Best real_AP50 threshold=%s AP50=%.4f",
            jsonl_path,
            csv_path,
            best_ap50.get("score_threshold"),
            float(best_ap50.get("real_AP50") or 0.0),
        )
    else:
        log.info("Wrote %s and %s", jsonl_path, csv_path)


if __name__ == "__main__":
    main()
