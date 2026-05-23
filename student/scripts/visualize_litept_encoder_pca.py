#!/usr/bin/env python3
"""PCA visualizations for LitePT-S* hierarchical U-Net encoder features.

The script loads a trained student checkpoint, captures LitePT encoder stages
E0..E4, PCA-colors the learned features, and writes point clouds/meshes that
can be opened in MeshLab, CloudCompare, or Open3D.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from plyfile import PlyData, PlyElement

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
from student.engine.vis import _recolor_mesh, _resolve_source_mesh
from student.models.student_model import build_student_model

log = logging.getLogger("visualize_litept_encoder_pca")


@dataclass
class EncoderCapture:
    name: str
    feat: torch.Tensor
    coord: torch.Tensor
    offset: torch.Tensor
    input_voxel_to_stage: torch.Tensor


class LitePTEncoderTap:
    """Temporary forward hooks for LitePT encoder stages.

    LitePT decoder unpooling mutates skip ``Point`` objects, so each hook clones
    tensors immediately while the encoder output still represents that stage.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        self.captures: list[EncoderCapture] = []
        self._voxel_maps: list[torch.Tensor] = []
        self._handles: list[Any] = []

        enc = backbone.model.enc
        for idx in range(len(enc)):
            self._handles.append(enc[idx].register_forward_hook(self._make_hook(idx)))

    def clear(self) -> None:
        self.captures.clear()
        self._voxel_maps.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, stage_idx: int):
        def hook(_module, _inputs, output) -> None:
            feat = output.feat.detach().clone()
            coord = output.coord.detach().clone()
            offset = getattr(
                output,
                "offset",
                torch.tensor([feat.shape[0]], device=feat.device, dtype=torch.long),
            ).detach().clone()

            if stage_idx == 0:
                input_voxel_to_stage = torch.arange(
                    feat.shape[0],
                    device=feat.device,
                    dtype=torch.long,
                )
            else:
                if "pooling_inverse" not in output.keys():
                    raise RuntimeError(
                        f"LitePT encoder stage E{stage_idx} has no pooling_inverse"
                    )
                if len(self._voxel_maps) != stage_idx:
                    raise RuntimeError(
                        f"Unexpected encoder hook order at E{stage_idx}: "
                        f"have {len(self._voxel_maps)} previous maps"
                    )
                parent_to_stage = output.pooling_inverse.detach().long().clone()
                input_voxel_to_stage = parent_to_stage[self._voxel_maps[-1]]

            input_voxel_to_stage = input_voxel_to_stage.detach().clone()
            self._voxel_maps.append(input_voxel_to_stage)
            self.captures.append(
                EncoderCapture(
                    name=f"E{stage_idx}",
                    feat=feat,
                    coord=coord,
                    offset=offset,
                    input_voxel_to_stage=input_voxel_to_stage,
                )
            )

        return hook


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


def _split_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _STUDENT_ROOT / p


def _build_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict")
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items()}
    state.pop("g_ft_logit", None)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _set_bn_mode(model: torch.nn.Module, mode: str) -> None:
    model.eval()
    if mode == "eval":
        return
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


def _stabilize_pca_sign(v: torch.Tensor) -> torch.Tensor:
    v = v.clone()
    for j in range(v.shape[1]):
        idx = torch.argmax(v[:, j].abs())
        if v[idx, j] < 0:
            v[:, j] *= -1
    return v


@torch.no_grad()
def pca_rgb(
    feat: torch.Tensor,
    *,
    style: str,
    q: int,
    niter: int,
    fit_max_points: int | None,
    l2_normalize: bool,
    robust_scale: bool,
    q_low: float,
    q_high: float,
    brightness: float,
) -> torch.Tensor:
    x = feat.detach().float()
    if l2_normalize:
        x = F.normalize(x, dim=1)

    if fit_max_points is not None and x.shape[0] > fit_max_points:
        fit_idx = torch.randperm(x.shape[0], device=x.device)[:fit_max_points]
        x_fit = x[fit_idx]
    else:
        x_fit = x

    q_eff = min(int(q), x_fit.shape[0], x_fit.shape[1])
    if style == "agile3d":
        q_eff = min(3, x_fit.shape[0], x_fit.shape[1])
    if q_eff <= 0:
        raise ValueError(f"Cannot PCA feature tensor with shape {tuple(feat.shape)}")

    _u, _s, v = torch.pca_lowrank(x_fit, q=q_eff, center=True, niter=niter)
    v = _stabilize_pca_sign(v)
    proj = x @ v

    if style == "paper" and q_eff >= 6:
        rgb = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    else:
        rgb = proj[:, : min(3, q_eff)]
        if rgb.shape[1] < 3:
            rgb = F.pad(rgb, (0, 3 - rgb.shape[1]))

    if style == "agile3d":
        # AGILE3D's feature visualization fits a 3-D PCA projection, then
        # rescales all projected values with one global min/max range.
        lo = rgb.min()
        hi = rgb.max()
    elif robust_scale:
        lo = torch.quantile(rgb, q_low, dim=0, keepdim=True)
        hi = torch.quantile(rgb, q_high, dim=0, keepdim=True)
    else:
        lo = rgb.min(dim=0, keepdim=True).values
        hi = rgb.max(dim=0, keepdim=True).values

    rgb = (rgb - lo) / (hi - lo).clamp_min(1e-6)
    return (rgb * brightness).clamp(0.0, 1.0)


def _normalize_concat_block(feat: torch.Tensor, mode: str) -> torch.Tensor:
    feat = feat.float()
    if mode == "none":
        return feat
    if mode == "l2":
        return F.normalize(feat, dim=1)
    if mode == "zscore":
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (feat - mean) / std
    raise ValueError(f"Unknown concat normalization mode: {mode}")


def _to_uint8(colors: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(colors, torch.Tensor):
        arr = colors.detach().cpu().numpy()
    else:
        arr = np.asarray(colors)
    if arr.dtype != np.uint8:
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_point_cloud_ply(xyz: torch.Tensor | np.ndarray, colors: torch.Tensor | np.ndarray, path: Path) -> None:
    xyz_np = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    rgb_np = _to_uint8(colors)
    if xyz_np.shape[0] != rgb_np.shape[0]:
        raise ValueError(
            f"xyz/color length mismatch for {path}: {xyz_np.shape[0]} vs {rgb_np.shape[0]}"
        )
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
    verts["x"] = xyz_np[:, 0].astype(np.float32)
    verts["y"] = xyz_np[:, 1].astype(np.float32)
    verts["z"] = xyz_np[:, 2].astype(np.float32)
    verts["red"] = rgb_np[:, 0]
    verts["green"] = rgb_np[:, 1]
    verts["blue"] = rgb_np[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))


def _source_rgb_colors(sample: dict[str, Any], vertex_indices: torch.Tensor | None = None) -> np.ndarray:
    colors_path = Path(sample["scene_dir"]) / "colors.npy"
    if colors_path.exists():
        colors = np.load(colors_path)
        if vertex_indices is not None:
            colors = colors[vertex_indices.cpu().numpy()]
        if colors.max() <= 1.0:
            colors = colors * 255.0
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.full((sample["points"].shape[0], 3), 180, dtype=np.uint8)


def _maybe_subsample(
    sample: dict[str, Any],
    *,
    max_points: int | None,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    n = int(sample["points"].shape[0])
    if max_points is None or n <= max_points:
        return sample, None

    rng = np.random.default_rng(seed)
    idx_np = np.sort(rng.choice(n, size=int(max_points), replace=False))
    idx = torch.from_numpy(idx_np).long()
    out = dict(sample)
    out["points"] = sample["points"][idx]
    out["features"] = sample["features"][idx]
    if "labels_by_granularity" in sample:
        out["labels_by_granularity"] = {
            g: labels[idx] for g, labels in sample["labels_by_granularity"].items()
        }
    for key in ("valid_points", "seen_points", "supervision_mask"):
        if key in sample:
            out[key] = sample[key][idx]
    out["vertex_indices"] = idx
    return out, idx


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

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, colors) in zip(axes, panels):
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            c=np.asarray(colors[idx], dtype=np.float32) / 255.0,
            s=0.1,
            marker=".",
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _scene_dirs_from_args(cfg: dict[str, Any], split: str, scenes: list[str] | None, num_scenes: int) -> list[Path]:
    data_cfg = cfg["data"]
    scans_root = Path(data_cfg["scans_root"])
    split_key = "train_split" if split == "train" else "val_split"
    all_dirs = build_scene_list(_split_path(data_cfg[split_key]), scans_root)
    if scenes:
        by_id = {p.name: p for p in all_dirs}
        missing = [scene for scene in scenes if scene not in by_id]
        if missing:
            raise ValueError(f"Scene(s) not found in {split} split: {missing}")
        return [by_id[scene] for scene in scenes]
    return all_dirs[:num_scenes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/resolved_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/checkpoints/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_eval150/encoder_pca",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bn-mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--style", choices=("paper", "top3", "agile3d"), default="paper")
    parser.add_argument("--pca-q", type=int, default=6)
    parser.add_argument("--pca-niter", type=int, default=5)
    parser.add_argument("--pca-fit-max-points", type=int, default=200_000)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--robust-scale", action="store_true")
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--brightness", type=float, default=1.15)
    parser.add_argument("--png-max-points", type=int, default=180_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concat-projected-stages",
        action="store_true",
        help=(
            "Project every encoder stage back to input points, concatenate the "
            "multiscale features per point, and PCA-color that concatenated "
            "point feature."
        ),
    )
    parser.add_argument(
        "--concat-stage-normalize",
        choices=("none", "l2", "zscore"),
        default="none",
        help="Optional per-stage normalization before multiscale concatenation.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = _scene_dirs_from_args(cfg, args.split, args.scenes, args.num_scenes)
    granularities = parse_granularities(cfg["data"])
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    ds = MultiSceneDataset(
        scene_dirs,
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
    model = _build_model(cfg, Path(args.checkpoint), device)
    _set_bn_mode(model, args.bn_mode)
    tap = LitePTEncoderTap(model.backbone)

    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "out_dir": str(out_root),
        "pca": {
            "style": args.style,
            "q": args.pca_q,
            "niter": args.pca_niter,
            "fit_max_points": args.pca_fit_max_points,
            "l2_normalize": args.l2_normalize,
            "robust_scale": args.robust_scale,
            "brightness": args.brightness,
            "concat_projected_stages": args.concat_projected_stages,
            "concat_stage_normalize": args.concat_stage_normalize,
        },
        "scenes": {},
    }

    try:
        for ds_idx in range(len(ds)):
            sample_full = ds.get_full_item(ds_idx)
            sample, vertex_indices = _maybe_subsample(
                sample_full,
                max_points=args.max_points,
                seed=args.seed + ds_idx,
            )
            scene_id = str(sample["scene_id"])
            scene_out = out_root / scene_id
            scene_out.mkdir(parents=True, exist_ok=True)

            points = sample["points"].to(device)
            features = sample["features"].to(device)
            log.info("[%s] points=%d features=%d", scene_id, points.shape[0], features.shape[1])

            tap.clear()
            with torch.no_grad():
                bb = model.backbone(points, features)

            if len(tap.captures) == 0:
                raise RuntimeError("No LitePT encoder stages were captured")

            inverse_map = bb.inverse_map.detach().long()
            points_np = sample["points"].cpu().numpy()
            source_rgb = _source_rgb_colors(sample_full, vertex_indices=vertex_indices)
            save_point_cloud_ply(points_np, source_rgb, scene_out / "input_rgb_points.ply")

            source_mesh: Path | None = None
            if vertex_indices is None:
                try:
                    source_mesh = _resolve_source_mesh(sample["scene_dir"], sample["scene_meta"])
                    shutil.copy2(source_mesh, scene_out / "original_rgb_mesh.ply")
                except Exception as exc:
                    log.warning("[%s] source mesh unavailable: %s", scene_id, exc)

            panels: list[tuple[str, np.ndarray]] = [("RGB", source_rgb)]
            scene_manifest: dict[str, Any] = {
                "dir": str(scene_out),
                "input_rgb_points": str(scene_out / "input_rgb_points.ply"),
                "original_rgb_mesh": str(scene_out / "original_rgb_mesh.ply") if source_mesh else None,
                "num_points": int(points.shape[0]),
                "stages": {},
            }
            projected_stage_feats: list[tuple[str, torch.Tensor]] = []

            for cap in tap.captures:
                log.info(
                    "[%s] %s tokens=%d channels=%d",
                    scene_id,
                    cap.name,
                    cap.feat.shape[0],
                    cap.feat.shape[1],
                )
                token_rgb = pca_rgb(
                    cap.feat,
                    style=args.style,
                    q=args.pca_q,
                    niter=args.pca_niter,
                    fit_max_points=args.pca_fit_max_points,
                    l2_normalize=args.l2_normalize,
                    robust_scale=args.robust_scale,
                    q_low=args.q_low,
                    q_high=args.q_high,
                    brightness=args.brightness,
                )
                point_to_stage = cap.input_voxel_to_stage[inverse_map]
                if args.concat_projected_stages:
                    projected_stage_feats.append((cap.name, cap.feat[point_to_stage].detach()))
                point_rgb = token_rgb[point_to_stage]

                native_path = scene_out / f"{cap.name}_native_tokens_pca.ply"
                projected_path = scene_out / f"{cap.name}_projected_points_pca.ply"
                mesh_path = scene_out / f"{cap.name}_projected_mesh_pca.ply"
                save_point_cloud_ply(cap.coord, token_rgb, native_path)
                save_point_cloud_ply(points_np, point_rgb, projected_path)

                mesh_written = False
                if source_mesh is not None:
                    try:
                        _recolor_mesh(source_mesh, _to_uint8(point_rgb), mesh_path)
                        mesh_written = True
                    except Exception as exc:
                        log.warning("[%s] %s mesh recolor failed: %s", scene_id, cap.name, exc)

                point_rgb_u8 = _to_uint8(point_rgb)
                panels.append((f"{cap.name} {cap.feat.shape[1]}d", point_rgb_u8))
                scene_manifest["stages"][cap.name] = {
                    "tokens": int(cap.feat.shape[0]),
                    "channels": int(cap.feat.shape[1]),
                    "native_tokens_ply": str(native_path),
                    "projected_points_ply": str(projected_path),
                    "projected_mesh_ply": str(mesh_path) if mesh_written else None,
                }

            if args.concat_projected_stages:
                concat_name = (
                    "Eall_concat"
                    if args.concat_stage_normalize == "none"
                    else f"Eall_concat_{args.concat_stage_normalize}"
                )
                log.info(
                    "[%s] %s stages=%s normalize=%s",
                    scene_id,
                    concat_name,
                    ",".join(name for name, _feat in projected_stage_feats),
                    args.concat_stage_normalize,
                )
                concat_blocks = [
                    _normalize_concat_block(feat, args.concat_stage_normalize)
                    for _name, feat in projected_stage_feats
                ]
                concat_feat = torch.cat(concat_blocks, dim=1)
                concat_rgb = pca_rgb(
                    concat_feat,
                    style=args.style,
                    q=args.pca_q,
                    niter=args.pca_niter,
                    fit_max_points=args.pca_fit_max_points,
                    l2_normalize=args.l2_normalize,
                    robust_scale=args.robust_scale,
                    q_low=args.q_low,
                    q_high=args.q_high,
                    brightness=args.brightness,
                )
                concat_projected_path = scene_out / f"{concat_name}_projected_points_pca.ply"
                concat_mesh_path = scene_out / f"{concat_name}_projected_mesh_pca.ply"
                save_point_cloud_ply(points_np, concat_rgb, concat_projected_path)

                concat_mesh_written = False
                if source_mesh is not None:
                    try:
                        _recolor_mesh(source_mesh, _to_uint8(concat_rgb), concat_mesh_path)
                        concat_mesh_written = True
                    except Exception as exc:
                        log.warning("[%s] %s mesh recolor failed: %s", scene_id, concat_name, exc)

                concat_rgb_u8 = _to_uint8(concat_rgb)
                panels.append((f"{concat_name} {concat_feat.shape[1]}d", concat_rgb_u8))
                scene_manifest["concat_projected_stages"] = {
                    "name": concat_name,
                    "stages": [name for name, _feat in projected_stage_feats],
                    "channels": int(concat_feat.shape[1]),
                    "normalize": args.concat_stage_normalize,
                    "projected_points_ply": str(concat_projected_path),
                    "projected_mesh_ply": str(concat_mesh_path) if concat_mesh_written else None,
                }
                del concat_blocks, concat_feat, concat_rgb
                torch.cuda.empty_cache()

            topdown_path = scene_out / "encoder_pca_topdown.png"
            _save_topdown_png(topdown_path, points_np, panels, max_points=args.png_max_points)
            scene_manifest["topdown_png"] = str(topdown_path)
            manifest["scenes"][scene_id] = scene_manifest
            log.info("[%s] wrote %s", scene_id, scene_out)

    finally:
        tap.remove()

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Done. Manifest: %s", out_root / "manifest.json")


if __name__ == "__main__":
    main()
