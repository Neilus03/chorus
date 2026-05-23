#!/usr/bin/env python3
"""Render object-centric RGB/PCA pairs in the style of AGILE3D Fig. 14."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32) / 255.0
    return xyz, rgb


def _pick_object_crop(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    size: tuple[float, float, float],
    min_points: int,
    max_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if ids.size == 0:
        center = np.median(xyz, axis=0)
    else:
        n = xyz.shape[0]
        candidates = []
        for inst_id, count in zip(ids, counts):
            if count < min_points or count > max_fraction * n:
                continue
            pts = xyz[labels == inst_id]
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            extent = hi - lo
            if extent[2] < 0.08:
                continue
            if extent[0] * extent[1] > 8.0:
                continue
            candidates.append((inst_id, count, pts.mean(axis=0), extent))

        if not candidates:
            center = np.median(xyz, axis=0)
        else:
            centers = np.stack([c[2] for c in candidates], axis=0)
            half = np.array(size, dtype=np.float32) / 2.0
            best_score = None
            best_center = None
            for c in centers:
                inside_inst = np.all((centers >= c - half) & (centers <= c + half), axis=1)
                point_mask = np.all((xyz >= c - half) & (xyz <= c + half), axis=1)
                z = xyz[point_mask, 2]
                z_span = float(z.max() - z.min()) if z.size else 0.0
                score = (
                    int(inside_inst.sum()),
                    int(point_mask.sum()),
                    z_span,
                    -abs(float(c[2] - np.median(xyz[:, 2]))),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_center = c
            center = best_center if best_center is not None else np.median(xyz, axis=0)

    half = np.array(size, dtype=np.float32) / 2.0
    lo = center - half
    hi = center + half
    lo[2] = max(lo[2], float(xyz[:, 2].min()))
    hi[2] = min(hi[2], float(xyz[:, 2].max()))
    return lo, hi


def _crop(
    xyz: np.ndarray,
    rgb: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
    if idx.size == 0:
        idx = np.arange(xyz.shape[0])
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return xyz[idx], rgb[idx]


def _equal_axes(ax, xyz: np.ndarray) -> None:
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.55, center[2] + radius * 0.55)
    ax.set_box_aspect((1, 1, 0.55))


def _render_panel(
    ax,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    elev: float,
    azim: float,
    point_size: float,
    title: str,
) -> None:
    order = np.argsort(xyz[:, 2])
    xyz = xyz[order]
    rgb = rgb[order]
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=np.clip(rgb, 0.0, 1.0),
        s=point_size,
        marker=".",
        linewidths=0,
        alpha=1.0,
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    _equal_axes(ax, xyz)
    ax.set_axis_off()
    ax.set_title(title, fontfamily="serif", fontsize=16, pad=0)
    try:
        ax.set_proj_type("persp", focal_length=0.7)
    except TypeError:
        ax.set_proj_type("persp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-root", required=True)
    parser.add_argument("--scans-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--feature-ply", default="E3_projected_points_pca.ply")
    parser.add_argument("--granularity", default="g0.5")
    parser.add_argument("--size", nargs=3, type=float, default=(2.4, 2.0, 1.7))
    parser.add_argument("--max-points", type=int, default=180_000)
    parser.add_argument("--point-size", type=float, default=1.4)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--min-instance-points", type=int, default=250)
    parser.add_argument("--max-instance-fraction", type=float, default=0.18)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca_root = Path(args.pca_root)
    scans_root = Path(args.scans_root)
    scenes = list(args.scenes)

    fig = plt.figure(figsize=(4.7 * 2 * len(scenes), 4.2), dpi=230)
    fig.patch.set_facecolor("white")

    for i, scene in enumerate(scenes):
        scene_dir = pca_root / scene
        xyz, rgb = _load_ply(scene_dir / "input_rgb_points.ply")
        feat_xyz, feat_rgb = _load_ply(scene_dir / args.feature_ply)
        labels = np.load(scans_root / scene / "training_pack" / f"labels_{args.granularity}.npy")
        lo, hi = _pick_object_crop(
            xyz,
            labels,
            size=tuple(args.size),
            min_points=args.min_instance_points,
            max_fraction=args.max_instance_fraction,
        )
        rgb_xyz, rgb_color = _crop(
            xyz,
            rgb,
            lo,
            hi,
            max_points=args.max_points,
            seed=i * 2,
        )
        feat_xyz_crop, feat_color = _crop(
            feat_xyz,
            feat_rgb,
            lo,
            hi,
            max_points=args.max_points,
            seed=i * 2 + 1,
        )
        ax_rgb = fig.add_subplot(1, 2 * len(scenes), 2 * i + 1, projection="3d")
        ax_feat = fig.add_subplot(1, 2 * len(scenes), 2 * i + 2, projection="3d")
        _render_panel(
            ax_rgb,
            rgb_xyz,
            rgb_color,
            elev=args.elev,
            azim=args.azim,
            point_size=args.point_size,
            title="RGB",
        )
        _render_panel(
            ax_feat,
            feat_xyz_crop,
            feat_color,
            elev=args.elev,
            azim=args.azim,
            point_size=args.point_size,
            title="Feature PCA",
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.9, wspace=0.0)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
