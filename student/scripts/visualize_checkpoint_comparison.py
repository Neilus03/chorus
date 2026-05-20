#!/usr/bin/env python3
"""Render side-by-side scene visualizations for two student checkpoints.

Outputs, per scene:
  - original_rgb.ply
  - gt_scannet20.ply
  - pretrained_pred.ply
  - finetuned_pred.ply
  - topdown_compare.png
  - metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDENT_ROOT.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))
chorus_outer = _REPO_ROOT / "chorus"
if chorus_outer.exists() and str(chorus_outer) not in sys.path:
    sys.path.insert(0, str(chorus_outer))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.data.target_builder import InstanceTargets, build_instance_targets_multi
from student.engine.evaluator import evaluate_student_predictions_multi
from student.engine.vis import _recolor_mesh, _resolve_source_mesh, save_gt_ply
from student.models.continuous_base import is_continuous_decoder
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model

log = logging.getLogger("visualize_checkpoint_comparison")


def apply_cli_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(value)


def _gran_key_to_float(key: str) -> float:
    if key.startswith("g0") and len(key) == 3:
        return float(f"0.{key[-1]}")
    if key.startswith("g") and len(key) == 3:
        return float(f"{key[1]}.{key[2]}")
    return float(str(key).replace("g", "0."))


def _build_base_model(cfg: dict[str, Any], granularities: tuple[str, ...]) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    bb_cfg = model_cfg["backbone"]
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        bb_cfg["in_channels"] = 9 if bool(data_cfg.get("append_xyz_to_features", False)) else 6
    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    return build_student_model(
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
        decoder_type=model_cfg.get("decoder_type", "multi_head"),
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
        continuous_decoder_v2=model_cfg.get("continuous_decoder_v2", None),
    )


def _state_dict_from_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"{path} does not contain a model_state_dict")
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _build_model_for_checkpoint(
    cfg: dict[str, Any],
    granularities: tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    state = _state_dict_from_checkpoint(checkpoint_path, torch.device("cpu"))
    is_prompt_state = any(k == "g_ft_logit" or k.startswith("model.") for k in state)
    model = _build_base_model(cfg, granularities)
    if is_prompt_state:
        prompt_cfg = cfg.get("train", {}).get("prompt_finetune", {})
        if not isinstance(prompt_cfg, dict):
            prompt_cfg = {}
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_cfg.get("backbone_lr_scale", 0.01)),
            mode=str(prompt_cfg.get("mode", "learned")),
        )
    load = model.load_state_dict(state, strict=True)
    if getattr(load, "missing_keys", None) or getattr(load, "unexpected_keys", None):
        log.warning(
            "Non-empty load result for %s: missing=%s unexpected=%s",
            checkpoint_path,
            getattr(load, "missing_keys", []),
            getattr(load, "unexpected_keys", []),
        )
    model.to(device)
    model.eval()
    return model


def _clear_backbone_cache(model: torch.nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _eval_bn_like_training(model: torch.nn.Module) -> None:
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


def _predict_multihead(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    granularity: str,
) -> dict[str, Any]:
    _clear_backbone_cache(model)
    _eval_bn_like_training(model)
    with torch.no_grad():
        if isinstance(model, FineTuningWrapper):
            flat = model(points, features)
        elif is_continuous_decoder(getattr(model, "decoder", None)):
            flat = model(points, features, target_g=_gran_key_to_float(granularity))
        else:
            pred = model(points, features)
            if isinstance(pred, dict) and "heads" in pred:
                return pred
            raise TypeError("Expected multi-head or continuous student prediction")
    head = {
        "mask_logits": flat["mask_logits"],
        "score_logits": flat["score_logits"],
        "query_embed": flat.get("query_embed"),
    }
    if "class_logits" in flat:
        head["class_logits"] = flat["class_logits"]
    out: dict[str, Any] = {"heads": {granularity: head}}
    if "point_embed" in flat:
        out["point_embed"] = flat["point_embed"]
    return out


def _palette(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 255, size=(n, 3), dtype=np.uint8)


def _prediction_colors(
    pred: dict[str, Any],
    granularity: str,
    *,
    score_threshold: float,
    mask_threshold: float,
    min_points: int,
) -> tuple[np.ndarray, int]:
    head = pred["heads"][granularity]
    mask_logits = head["mask_logits"].detach().cpu()
    score_logits = head["score_logits"].detach().cpu()
    scores = score_logits.sigmoid().numpy()
    masks = (mask_logits.sigmoid().numpy() >= mask_threshold)

    colors = np.full((masks.shape[1], 3), 60, dtype=np.uint8)
    palette = _palette(masks.shape[0])
    kept: list[int] = []
    for q, score in enumerate(scores):
        if score < score_threshold:
            continue
        if int(masks[q].sum()) < min_points:
            continue
        kept.append(q)
    for q in sorted(kept, key=lambda i: float(scores[i])):
        colors[masks[q]] = palette[q]
    return colors, len(kept)


def _gt_colors(targets: InstanceTargets) -> np.ndarray:
    gt_masks = targets.gt_masks.detach().cpu().numpy()
    colors = np.full((targets.supervision_mask.shape[0], 3), 30, dtype=np.uint8)
    palette = _palette(max(targets.num_instances, 1))
    for idx in range(targets.num_instances):
        colors[gt_masks[idx]] = palette[idx]
    return colors


def _save_topdown_png(
    path: Path,
    points: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    *,
    max_points: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable; skipping %s (%s)", path, exc)
        return

    n = points.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        idx = np.arange(n)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, colors) in zip(axes, panels):
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            c=np.asarray(colors[idx], dtype=np.float32) / 255.0,
            s=0.12,
            marker=".",
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _source_rgb_colors(sample: dict[str, Any]) -> np.ndarray:
    scene_dir = Path(sample["scene_dir"])
    colors_path = scene_dir / "colors.npy"
    if colors_path.exists():
        colors = np.load(colors_path)
        if colors.max() <= 1.0:
            colors = colors * 255.0
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.full((sample["points"].shape[0], 3), 180, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scannet_full_continuous_ft_gt_scannet20_classagnostic.yaml")
    parser.add_argument("--pretrained", default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/checkpoints/best.pt")
    parser.add_argument("--finetuned", default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_prompt_ft_gt_scannet20_classagnostic/checkpoints/best.pt")
    parser.add_argument("--scenes", nargs="+", default=["scene0500_01", "scene0575_01"])
    parser.add_argument("--out-dir", default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_prompt_ft_gt_scannet20_classagnostic/visual_compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--png-max-points", type=int, default=160_000)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    set_seed(42)

    cfg = load_config(Path(args.config))
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)
    granularities = parse_granularities(cfg["data"])
    if len(granularities) != 1:
        raise ValueError(f"This comparison script expects one granularity, got {granularities}")
    granularity = granularities[0]

    device = torch.device(args.device)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
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

    log.info("Loading pretrained checkpoint: %s", args.pretrained)
    pretrained = _build_model_for_checkpoint(cfg, granularities, Path(args.pretrained), device)
    log.info("Loading fine-tuned checkpoint: %s", args.finetuned)
    finetuned = _build_model_for_checkpoint(cfg, granularities, Path(args.finetuned), device)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    score_threshold = float(eval_cfg.get("score_threshold", 0.01))
    mask_threshold = float(eval_cfg.get("mask_threshold", 0.5))
    min_points = int(eval_cfg.get("min_points_per_proposal", 30))

    manifest: dict[str, Any] = {"scenes": {}, "pretrained": args.pretrained, "finetuned": args.finetuned}
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
        save_gt_ply(targets_by_gran[granularity], source_mesh, path=scene_out / "gt_scannet20.ply")

        log.info("[%s] running pretrained", scene_id)
        pred_pre = _predict_multihead(pretrained, points, features, granularity)
        colors_pre, kept_pre = _prediction_colors(
            pred_pre,
            granularity,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_points=min_points,
        )
        _recolor_mesh(source_mesh, colors_pre, scene_out / "pretrained_pred.ply")

        log.info("[%s] running fine-tuned", scene_id)
        pred_ft = _predict_multihead(finetuned, points, features, granularity)
        colors_ft, kept_ft = _prediction_colors(
            pred_ft,
            granularity,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_points=min_points,
        )
        _recolor_mesh(source_mesh, colors_ft, scene_out / "finetuned_pred.ply")

        metrics = {
            "pretrained": evaluate_student_predictions_multi(
                pred_pre,
                targets_by_gran,
                scene_dir=sample["scene_dir"],
                scene_id=scene_id,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_points=min_points,
                eval_benchmarks=eval_cfg.get("scannet_benchmarks", eval_cfg.get("scannet_benchmark", "scannet20")),
            ),
            "finetuned": evaluate_student_predictions_multi(
                pred_ft,
                targets_by_gran,
                scene_dir=sample["scene_dir"],
                scene_id=scene_id,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_points=min_points,
                eval_benchmarks=eval_cfg.get("scannet_benchmarks", eval_cfg.get("scannet_benchmark", "scannet20")),
            ),
            "kept_proposals": {
                "pretrained": kept_pre,
                "finetuned": kept_ft,
            },
        }
        with (scene_out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float)

        points_np = sample["points"].cpu().numpy()
        rgb = _source_rgb_colors(sample)
        gt = _gt_colors(targets_by_gran[granularity])
        _save_topdown_png(
            scene_out / "topdown_compare.png",
            points_np,
            [
                ("RGB", rgb),
                ("GT", gt),
                ("Pretrained", colors_pre),
                ("Fine-tuned", colors_ft),
            ],
            max_points=args.png_max_points,
        )
        manifest["scenes"][scene_id] = {
            "dir": str(scene_out),
            "original_rgb": str(scene_out / "original_rgb.ply"),
            "gt": str(scene_out / "gt_scannet20.ply"),
            "pretrained": str(scene_out / "pretrained_pred.ply"),
            "finetuned": str(scene_out / "finetuned_pred.ply"),
            "topdown_png": str(scene_out / "topdown_compare.png"),
            "metrics": str(scene_out / "metrics.json"),
        }
        log.info("[%s] wrote %s", scene_id, scene_out)

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Done. Manifest: %s", out_root / "manifest.json")


if __name__ == "__main__":
    main()
