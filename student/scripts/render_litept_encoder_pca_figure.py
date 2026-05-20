#!/usr/bin/env python3
"""Render a paper-style LitePT encoder PCA figure from native token PLYs."""

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


def _parse_bounds(text: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if text is None:
        return None
    vals = [float(v) for v in text.replace(",", " ").split()]
    if len(vals) != 6:
        raise ValueError("--bounds must be six values: xmin xmax ymin ymax zmin zmax")
    lo = np.array([vals[0], vals[2], vals[4]], dtype=np.float32)
    hi = np.array([vals[1], vals[3], vals[5]], dtype=np.float32)
    return lo, hi


def _auto_bounds(
    xyz: np.ndarray,
    *,
    center: tuple[float, float, float] | None,
    size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if center is None:
        # Prefer a room-interior crop with some vertical/object structure instead
        # of the exact scene center, which is often empty floor in ScanNet.
        z = xyz[:, 2]
        candidate = xyz[z > np.quantile(z, 0.35)]
        if candidate.shape[0] < 1000:
            candidate = xyz
        c = np.median(candidate, axis=0)
    else:
        c = np.array(center, dtype=np.float32)
    half = np.array(size, dtype=np.float32) / 2.0
    lo = c - half
    hi = c + half
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
    mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"Crop is empty: lo={lo.tolist()} hi={hi.tolist()}")
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return xyz[idx], rgb[idx]


def _equal_axes(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.55, center[2] + radius * 0.55)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing E0..E4_native_tokens_pca.ply",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--bounds", default=None, help="xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--center", nargs=3, type=float, default=None)
    parser.add_argument("--size", nargs=3, type=float, default=(3.0, 2.6, 2.4))
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_dir = Path(args.scene_dir)
    stages = []
    for stage in range(5):
        xyz, rgb = _load_ply(scene_dir / f"E{stage}_native_tokens_pca.ply")
        stages.append((xyz, rgb))

    bounds = _parse_bounds(args.bounds)
    if bounds is None:
        bounds = _auto_bounds(
            stages[0][0],
            center=tuple(args.center) if args.center is not None else None,
            size=tuple(args.size),
        )
    lo, hi = bounds

    marker_sizes = [0.7, 2.4, 8.5, 28.0, 72.0]
    max_points_by_stage = [
        args.max_points,
        min(args.max_points, 60_000),
        min(args.max_points, 35_000),
        min(args.max_points, 12_000),
        min(args.max_points, 6_000),
    ]

    fig = plt.figure(figsize=(16.0, 3.85), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    for stage, ((xyz, rgb), point_cap, size) in enumerate(
        zip(stages, max_points_by_stage, marker_sizes)
    ):
        ax = fig.add_subplot(1, 5, stage + 1, projection="3d")
        xyz_crop, rgb_crop = _crop(
            xyz,
            rgb,
            lo,
            hi,
            max_points=point_cap,
            seed=args.seed + stage,
        )
        order = np.argsort(xyz_crop[:, 2])
        xyz_crop = xyz_crop[order]
        rgb_crop = rgb_crop[order]
        ax.scatter(
            xyz_crop[:, 0],
            xyz_crop[:, 1],
            xyz_crop[:, 2],
            c=rgb_crop,
            s=size,
            marker="o",
            linewidths=0,
            alpha=0.96,
            depthshade=True,
        )
        ax.set_title(f"Stage {stage}", fontfamily="serif", fontsize=18, pad=0)
        ax.view_init(elev=args.elev, azim=args.azim)
        _equal_axes(ax, lo, hi)
        ax.set_axis_off()
        try:
            ax.set_proj_type("persp", focal_length=0.65)
        except TypeError:
            ax.set_proj_type("persp")
        ax.set_box_aspect((1, 1, 0.55))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.88, wspace=0.02)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Bounds: {lo[0]:.3f} {hi[0]:.3f} {lo[1]:.3f} {hi[1]:.3f} {lo[2]:.3f} {hi[2]:.3f}")


if __name__ == "__main__":
    main()
