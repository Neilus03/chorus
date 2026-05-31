#!/usr/bin/env python3
"""PCA-color learned student point features."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model

log = logging.getLogger("visualize_student_point_pca")


def _split_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _STUDENT_ROOT / p


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if isinstance(model, FineTuningWrapper) else model


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _to_uint8(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(rgb)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_point_cloud_ply(
    xyz: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    path: Path,
) -> None:
    xyz_np = _to_numpy(xyz).astype(np.float32)
    rgb_np = _to_uint8(colors)
    if xyz_np.shape[0] != rgb_np.shape[0]:
        raise ValueError(f"xyz/color length mismatch: {xyz_np.shape[0]} vs {rgb_np.shape[0]}")
    verts = np.empty(
        xyz_np.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"] = xyz_np[:, 0]
    verts["y"] = xyz_np[:, 1]
    verts["z"] = xyz_np[:, 2]
    verts["red"] = rgb_np[:, 0]
    verts["green"] = rgb_np[:, 1]
    verts["blue"] = rgb_np[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))


def fit_pca(
    features: torch.Tensor,
    *,
    fit_mask: torch.Tensor | None = None,
    max_fit_points: int | None = 200_000,
    seed: int = 0,
) -> dict[str, torch.Tensor | list[float]]:
    x = features.detach().float().cpu()
    if fit_mask is not None:
        mask = fit_mask.detach().cpu().bool()
        if mask.shape[0] != x.shape[0]:
            raise ValueError(f"fit_mask length {mask.shape[0]} != features length {x.shape[0]}")
        if bool(mask.any()):
            x_fit = x[mask]
        else:
            x_fit = x
    else:
        x_fit = x
    if max_fit_points is not None and x_fit.shape[0] > int(max_fit_points):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(x_fit.shape[0], generator=generator)[: int(max_fit_points)]
        x_fit = x_fit[idx]
    if x_fit.ndim != 2 or min(x_fit.shape) < 3:
        raise ValueError(f"Need at least 3 points/channels for PCA, got {tuple(x_fit.shape)}")
    mean = x_fit.mean(dim=0, keepdim=True)
    centered = x_fit - mean
    _u, s, v = torch.pca_lowrank(centered, q=3, center=False, niter=5)
    for col in range(v.shape[1]):
        pivot = torch.argmax(v[:, col].abs())
        if v[pivot, col] < 0:
            v[:, col] *= -1
    total_var = centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    explained = ((s ** 2) / max(centered.shape[0], 1) / total_var).tolist()
    return {"mean": mean, "components": v, "explained_variance_ratio": explained}


def pca_rgb_from_fit(
    features: torch.Tensor,
    pca: dict[str, torch.Tensor | list[float]],
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> torch.Tensor:
    x = features.detach().float().cpu()
    mean = pca["mean"]
    components = pca["components"]
    if not isinstance(mean, torch.Tensor) or not isinstance(components, torch.Tensor):
        raise TypeError("pca must contain tensor mean and components")
    rgb = (x - mean) @ components[:, :3]
    lo = torch.quantile(rgb, float(q_low), dim=0, keepdim=True)
    hi = torch.quantile(rgb, float(q_high), dim=0, keepdim=True)
    return ((rgb - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def pca_rgb(
    features: torch.Tensor,
    *,
    fit_mask: torch.Tensor | None = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
    max_fit_points: int | None = 200_000,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pca = fit_pca(features, fit_mask=fit_mask, max_fit_points=max_fit_points, seed=seed)
    rgb = pca_rgb_from_fit(features, pca, q_low=q_low, q_high=q_high)
    return rgb, {
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "q_low": q_low,
        "q_high": q_high,
        "max_fit_points": max_fit_points,
    }


def feature_instance_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_points: int = 10,
) -> dict[str, Any]:
    x = F.normalize(features.detach().float().cpu(), dim=1)
    labels_cpu = labels.detach().cpu().long()
    instance_ids = [
        int(v.item())
        for v in labels_cpu.unique(sorted=True)
        if int(v.item()) > 0 and int((labels_cpu == int(v.item())).sum().item()) >= min_points
    ]
    if not instance_ids:
        return {
            "num_instances": 0,
            "mean_intra_instance_cosine_distance": None,
            "mean_nearest_inter_instance_cosine_distance": None,
            "separation_ratio_inter_over_intra": None,
        }
    centroids: list[torch.Tensor] = []
    intra: list[float] = []
    for inst_id in instance_ids:
        mask = labels_cpu == inst_id
        feat_i = x[mask]
        centroid = F.normalize(feat_i.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centroids.append(centroid)
        intra.extend((1.0 - (feat_i * centroid).sum(dim=1)).tolist())
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean: float | None = None
    if len(centroids) >= 2:
        c = torch.stack(centroids, dim=0)
        sim = c @ c.T
        sim.fill_diagonal_(-1.0)
        nearest = 1.0 - sim.max(dim=1).values
        inter_mean = float(nearest.mean().item())
    ratio = None
    if inter_mean is not None and intra_mean > 1e-12:
        ratio = float(inter_mean / intra_mean)
    return {
        "num_instances": len(instance_ids),
        "mean_intra_instance_cosine_distance": intra_mean,
        "mean_nearest_inter_instance_cosine_distance": inter_mean,
        "separation_ratio_inter_over_intra": ratio,
    }


def _build_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = dict(model_cfg["backbone"])
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        bb_cfg["in_channels"] = 9 if bool(data_cfg.get("append_xyz_to_features", False)) else 6
    granularities = parse_granularities(data_cfg)
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
        decoder_type=model_cfg.get("decoder_type", "multi_head"),
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
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))),
            mode=str(prompt_cfg.get("mode", "learned")),
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state is None and isinstance(checkpoint, dict):
        state = checkpoint
    if not isinstance(state, dict):
        raise KeyError(f"{checkpoint_path} does not contain a model state dict")
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    loading_base_into_prompt = (
        isinstance(model, FineTuningWrapper)
        and not any(k.startswith("model.") or k == "g_ft_logit" for k in state)
    )
    (model.model if loading_base_into_prompt else model).load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _scene_dirs_from_args(cfg: dict[str, Any], scenes: list[str]) -> list[Path]:
    data_cfg = cfg["data"]
    all_dirs = build_scene_list(_split_path(data_cfg["val_split"]), Path(data_cfg["scans_root"]))
    by_id = {p.name: p for p in all_dirs}
    missing = [scene for scene in scenes if scene not in by_id]
    if missing:
        raise ValueError(f"Scene(s) not found in validation split: {missing}")
    return [by_id[scene] for scene in scenes]


def _source_rgb(sample: dict[str, Any]) -> np.ndarray:
    colors_path = Path(sample["scene_dir"]) / "colors.npy"
    if colors_path.exists():
        colors = np.load(colors_path)
        if colors.max() <= 1.0:
            colors = colors * 255.0
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.full((sample["points"].shape[0], 3), 180, dtype=np.uint8)


def _instance_rgb(labels: torch.Tensor) -> np.ndarray:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    rgb = np.full((labels_np.shape[0], 3), 190, dtype=np.uint8)
    for inst_id in sorted(int(v) for v in np.unique(labels_np) if int(v) > 0):
        rng = np.random.default_rng(inst_id * 7919)
        rgb[labels_np == inst_id] = rng.integers(30, 240, size=3, dtype=np.uint8)
    return rgb


def _save_topdown_png(path: Path, points: np.ndarray, panels: list[tuple[str, np.ndarray]], max_points: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable; skipping png (%s)", exc)
        return
    n = points.shape[0]
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, rgb) in zip(axes, panels):
        ax.scatter(points[idx, 0], points[idx, 1], c=rgb[idx] / 255.0, s=0.15, marker=".", linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    feature_source: str,
) -> torch.Tensor:
    base = _unwrap_model(model)
    bb = base.backbone(points, features)
    feat = bb.point_feat
    if feature_source == "backbone_point_feat":
        return feat.detach()
    if feature_source == "decoder_mask_feat":
        decoder = getattr(base, "decoder", None)
        point_mask_proj = getattr(decoder, "point_mask_proj", None)
        if point_mask_proj is None:
            raise ValueError("feature-source=decoder_mask_feat requires decoder.point_mask_proj")
        return point_mask_proj(feat).detach()
    raise ValueError(f"Unknown feature source: {feature_source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--feature-source", choices=("backbone_point_feat", "decoder_mask_feat"), default="backbone_point_feat")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-render-points", type=int, default=150_000)
    parser.add_argument("--global-pca", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pca-fit-max-points", type=int, default=200_000)
    parser.add_argument("--save-gt-overlay", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg, Path(args.checkpoint), device)
    granularities = parse_granularities(cfg["data"])
    eval_cfg = cfg.get("eval", {})
    scene_dirs = _scene_dirs_from_args(cfg, args.scenes)
    ds = MultiSceneDataset(
        scene_dirs,
        granularities,
        use_colors=cfg["data"].get("use_colors", True),
        append_xyz=cfg["data"].get("append_xyz_to_features", False),
        use_normals=bool(cfg["data"].get("use_normals", False)),
        preload=False,
        max_points=None,
        subsampling_mode="none",
        train_augmentations=False,
        label_source=cfg["data"].get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(cfg["data"].get("scannet_gt_supervise_all_points", False)),
    )

    scene_payloads: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        sample = ds.get_full_item(idx)
        scene_id = str(sample["scene_id"])
        points = sample["points"].to(device)
        in_features = sample["features"].to(device)
        feat = _extract_features(model, points, in_features, feature_source=args.feature_source).cpu()
        scene_payloads.append({"sample": sample, "features": feat, "scene_id": scene_id})
        log.info("[%s] extracted %s shape=%s", scene_id, args.feature_source, tuple(feat.shape))

    global_fit = None
    if args.global_pca:
        fit_features = torch.cat([p["features"] for p in scene_payloads], dim=0)
        global_fit = fit_pca(fit_features, max_fit_points=args.pca_fit_max_points, seed=args.seed)

    manifest: dict[str, Any] = {"config": args.config, "checkpoint": args.checkpoint, "feature_source": args.feature_source, "scenes": {}}
    for idx, payload in enumerate(scene_payloads):
        sample = payload["sample"]
        scene_id = payload["scene_id"]
        feat = payload["features"]
        scene_out = out_root / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        points_np = sample["points"].cpu().numpy()
        rgb_input = _source_rgb(sample)
        save_point_cloud_ply(points_np, rgb_input, scene_out / "input_rgb_points.ply")

        if global_fit is None:
            fit_mask = sample.get("supervision_mask")
            rgb_float, pca_info = pca_rgb(
                feat,
                fit_mask=fit_mask,
                max_fit_points=args.pca_fit_max_points,
                seed=args.seed + idx,
            )
        else:
            rgb_float = pca_rgb_from_fit(feat, global_fit)
            pca_info = {
                "explained_variance_ratio": global_fit["explained_variance_ratio"],
                "global_pca": True,
                "max_fit_points": args.pca_fit_max_points,
            }
        rgb_pca = _to_uint8(rgb_float)
        stem = "student_backbone_point_feat_pca" if args.feature_source == "backbone_point_feat" else "student_decoder_mask_feat_pca"
        save_point_cloud_ply(points_np, rgb_pca, scene_out / f"{stem}.ply")
        with (scene_out / "pca_explained_variance.json").open("w", encoding="utf-8") as f:
            json.dump(pca_info, f, indent=2)

        labels = sample["labels_by_granularity"][granularities[0]]
        diagnostics = feature_instance_diagnostics(feat, labels, min_points=int(cfg["data"].get("min_instance_points", 10)))
        with (scene_out / "feature_instance_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        panels = [("RGB", rgb_input), ("PCA features", rgb_pca)]
        if args.save_gt_overlay:
            gt_rgb = _instance_rgb(labels)
            save_point_cloud_ply(points_np, gt_rgb, scene_out / "gt_instances_optional.ply")
            panels.append(("GT optional", gt_rgb))
        png_path = scene_out / f"{stem}.png"
        _save_topdown_png(png_path, points_np, panels, max_points=args.max_render_points)
        manifest["scenes"][scene_id] = {
            "num_points": int(feat.shape[0]),
            "feature_dim": int(feat.shape[1]),
            "input_rgb_points": str(scene_out / "input_rgb_points.ply"),
            "pca_points": str(scene_out / f"{stem}.ply"),
            "pca_png": str(png_path),
            "pca_explained_variance": str(scene_out / "pca_explained_variance.json"),
            "feature_instance_diagnostics": str(scene_out / "feature_instance_diagnostics.json"),
        }

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote PCA outputs to %s", out_root)


if __name__ == "__main__":
    main()
