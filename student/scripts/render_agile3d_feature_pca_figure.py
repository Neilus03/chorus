#!/usr/bin/env python3
"""Render AGILE3D-style feature PCA panels from point-cloud PLY exports."""

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


def _auto_bounds(xyz: np.ndarray, size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    z = xyz[:, 2]
    candidate = xyz[z > np.quantile(z, 0.35)]
    if candidate.shape[0] < 1000:
        candidate = xyz
    center = np.median(candidate, axis=0)
    half = np.array(size, dtype=np.float32) / 2.0
    lo = center - half
    hi = center + half
    lo[2] = max(lo[2], float(xyz[:, 2].min()))
    hi[2] = min(hi[2], float(xyz[:, 2].max()))
    return lo, hi


def _parse_bounds(text: str | None, xyz: np.ndarray, size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    if text is None:
        return _auto_bounds(xyz, size)
    vals = [float(v) for v in text.replace(",", " ").split()]
    if len(vals) != 6:
        raise ValueError("--bounds must be six values: xmin xmax ymin ymax zmin zmax")
    lo = np.array([vals[0], vals[2], vals[4]], dtype=np.float32)
    hi = np.array([vals[1], vals[3], vals[5]], dtype=np.float32)
    return lo, hi


def _crop(xyz: np.ndarray, rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_points: int, seed: int):
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
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--plys",
        nargs="+",
        default=[
            "input_rgb_points.ply",
            "E2_projected_points_pca.ply",
            "E3_projected_points_pca.ply",
            "Eall_concat_l2_projected_points_pca.ply",
        ],
        help="PLY filenames relative to --scene-dir, rendered left to right.",
    )
    parser.add_argument("--titles", nargs="+", default=None)
    parser.add_argument("--bounds", default=None, help="xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--size", nargs=3, type=float, default=(3.0, 2.6, 2.4))
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_dir = Path(args.scene_dir)
    plys = [scene_dir / p for p in args.plys]
    titles = args.titles or [Path(p).stem.replace("_projected_points_pca", "").replace("_", " ") for p in args.plys]
    if len(titles) != len(plys):
        raise ValueError("--titles length must match --plys length")

    loaded = [_load_ply(path) for path in plys]
    lo, hi = _parse_bounds(args.bounds, loaded[0][0], tuple(args.size))

    fig = plt.figure(figsize=(4.2 * len(loaded), 3.8), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    for idx, ((xyz, rgb), title) in enumerate(zip(loaded, titles)):
        ax = fig.add_subplot(1, len(loaded), idx + 1, projection="3d")
        xyz_crop, rgb_crop = _crop(xyz, rgb, lo, hi, args.max_points, args.seed + idx)
        order = np.argsort(xyz_crop[:, 2])
        xyz_crop = xyz_crop[order]
        rgb_crop = rgb_crop[order]
        ax.scatter(
            xyz_crop[:, 0],
            xyz_crop[:, 1],
            xyz_crop[:, 2],
            c=rgb_crop,
            s=args.point_size,
            marker=".",
            linewidths=0,
            alpha=0.98,
            depthshade=False,
        )
        ax.set_title(title, fontfamily="serif", fontsize=16, pad=0)
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
