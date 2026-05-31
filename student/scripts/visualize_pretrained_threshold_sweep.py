#!/usr/bin/env python3
"""Render prediction visualizations for one checkpoint across score thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDENT_ROOT.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))
chorus_outer = _REPO_ROOT / "chorus"
if chorus_outer.exists() and str(chorus_outer) not in sys.path:
    sys.path.insert(0, str(chorus_outer))

from student.config_utils import load_config, parse_granularities, set_seed
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.data.target_builder import build_instance_targets_multi
from student.engine.vis import _recolor_mesh, _resolve_source_mesh, save_gt_ply

from visualize_checkpoint_comparison import (
    _build_model_for_checkpoint,
    _gt_colors,
    _prediction_colors,
    _predict_multihead,
    _save_topdown_png,
    _source_rgb_colors,
    apply_cli_overrides,
)

log = logging.getLogger("visualize_pretrained_threshold_sweep")


def _threshold_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _granularity_key(value: str) -> str:
    raw = str(value).strip()
    if raw in {"g02", "g05", "g08"}:
        return raw
    try:
        f = float(raw)
    except ValueError:
        return raw
    return f"g{int(round(f * 10)):02d}"


def _score_summary(pred: dict[str, Any], granularity: str) -> dict[str, Any]:
    scores = pred["heads"][granularity]["score_logits"].detach().sigmoid().cpu().numpy()
    return {
        "num_queries": int(scores.shape[0]),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "quantiles": {
            f"{q:.2f}": float(np.quantile(scores, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scannet_full_continuous_ft_gt_scannet20_classagnostic.yaml")
    parser.add_argument(
        "--checkpoint",
        default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/checkpoints/best.pt",
    )
    parser.add_argument("--scenes", nargs="+", default=["scene0488_00", "scene0568_00"])
    parser.add_argument("--granularities", nargs="+", default=None)
    parser.add_argument(
        "--out-dir",
        default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/pretrained_pred_threshold_sweep_g05",
    )
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--png-max-points", type=int, default=200_000)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    set_seed(42)

    cfg = load_config(Path(args.config))
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)

    granularities = parse_granularities(cfg["data"])
    requested_granularities = tuple(_granularity_key(g) for g in args.granularities) if args.granularities else granularities
    missing_granularities = [g for g in requested_granularities if g not in granularities]
    if missing_granularities:
        raise ValueError(f"Requested granularities {missing_granularities} not in config granularities {granularities}")

    device = torch.device(args.device)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    score_thresholds = sorted({float(t) for t in args.score_thresholds})
    mask_threshold = float(eval_cfg.get("mask_threshold", 0.5) if args.mask_threshold is None else args.mask_threshold)
    min_points = int(eval_cfg.get("min_points_per_proposal", 30) if args.min_points is None else args.min_points)

    scans_root = Path(data_cfg["scans_root"])
    val_split = _STUDENT_ROOT / data_cfg["val_split"]
    val_dirs = build_scene_list(val_split, scans_root)
    scene_by_id = {p.name: p for p in val_dirs}
    missing = [s for s in args.scenes if s not in scene_by_id]
    if missing:
        raise ValueError(f"Scene(s) not found in validation split: {missing}")

    ds = MultiSceneDataset(
        [scene_by_id[s] for s in args.scenes],
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=False,
        max_points=None,
        subsampling_mode="none",
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(data_cfg.get("scannet_gt_supervise_all_points", False)),
    )

    log.info("Loading checkpoint: %s", args.checkpoint)
    model = _build_model_for_checkpoint(cfg, granularities, Path(args.checkpoint), device)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "granularities": list(requested_granularities),
        "score_thresholds": score_thresholds,
        "mask_threshold": mask_threshold,
        "min_points": min_points,
        "scenes": {},
    }

    for idx in range(len(ds)):
        sample = ds[idx]
        scene_id = str(sample["scene_id"])
        scene_out = out_root / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        points = sample["points"].to(device)
        features = sample["features"].to(device)
        targets_by_gran = build_instance_targets_multi(
            sample["labels_by_granularity"],
            sample["supervision_mask"],
            min_instance_points=int(data_cfg.get("min_instance_points", 10)),
            dense_instance_ids=bool(data_cfg.get("dense_instance_ids", False)),
            instance_class_maps=sample.get("instance_classes_by_granularity"),
        )
        source_mesh = _resolve_source_mesh(sample["scene_dir"], sample["scene_meta"])
        shutil.copy2(source_mesh, scene_out / "original_rgb.ply")
        points_np = sample["points"].cpu().numpy()
        rgb = _source_rgb_colors(sample)
        manifest["scenes"][scene_id] = {"dir": str(scene_out), "granularities": {}}

        for granularity in requested_granularities:
            gran_out = scene_out / granularity
            gran_out.mkdir(parents=True, exist_ok=True)
            save_gt_ply(targets_by_gran[granularity], source_mesh, path=gran_out / "gt.ply")

            log.info("[%s %s] running checkpoint", scene_id, granularity)
            pred = _predict_multihead(model, points, features, granularity)
            gt = _gt_colors(targets_by_gran[granularity])

            panels: list[tuple[str, np.ndarray]] = [("RGB", rgb), ("GT", gt)]
            scene_summary: dict[str, Any] = {
                "granularity": granularity,
                "score_summary": _score_summary(pred, granularity),
                "kept_proposals": {},
            }
            for threshold in score_thresholds:
                colors, kept = _prediction_colors(
                    pred,
                    granularity,
                    score_threshold=threshold,
                    mask_threshold=mask_threshold,
                    min_points=min_points,
                )
                tag = _threshold_tag(threshold)
                _recolor_mesh(source_mesh, colors, gran_out / f"pred_score_{tag}.ply")
                panels.append((f">= {threshold:g} ({kept})", colors))
                scene_summary["kept_proposals"][f"{threshold:.3f}"] = kept

            _save_topdown_png(gran_out / "topdown_score_sweep.png", points_np, panels, max_points=args.png_max_points)
            with (gran_out / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(scene_summary, f, indent=2)

            manifest["scenes"][scene_id]["granularities"][granularity] = {
                "dir": str(gran_out),
                "topdown_score_sweep": str(gran_out / "topdown_score_sweep.png"),
                "summary": str(gran_out / "summary.json"),
                "predictions": {
                    f"{threshold:.3f}": str(gran_out / f"pred_score_{_threshold_tag(threshold)}.ply")
                    for threshold in score_thresholds
                },
            }
        log.info("[%s] wrote %s", scene_id, scene_out)

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Done. Manifest: %s", out_root / "manifest.json")


if __name__ == "__main__":
    main()
