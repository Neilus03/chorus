#!/usr/bin/env python3
"""Quantify how instance-like PCA colorized point clouds are.

The metrics operate on exported ``*_projected_points_pca.ply`` files and
per-point pseudo-instance labels from the ScanNet training pack.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32) / 255.0
    return xyz, rgb


def _subsample(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _instance_stats(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    min_instance_points: int,
) -> dict[str, float]:
    valid = labels >= 0
    rgb = rgb[valid]
    labels = labels[valid]
    ids, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    keep = counts >= min_instance_points
    if keep.sum() < 2:
        return {
            "num_instances": float(keep.sum()),
            "within_instance_rgb_dist": np.nan,
            "between_instance_rgb_dist": np.nan,
            "between_within_ratio": np.nan,
        }

    kept_ids = ids[keep]
    means = []
    within = []
    for inst_id in kept_ids:
        pts = rgb[labels == inst_id]
        mu = pts.mean(axis=0)
        means.append(mu)
        within.append(np.linalg.norm(pts - mu, axis=1).mean())
    means_np = np.stack(means, axis=0)
    within_mean = float(np.mean(within))

    rng = np.random.default_rng(0)
    i = rng.integers(0, len(means_np), size=min(5000, len(means_np) * len(means_np)))
    j = rng.integers(0, len(means_np), size=i.shape[0])
    mask = i != j
    between = float(np.linalg.norm(means_np[i[mask]] - means_np[j[mask]], axis=1).mean())
    return {
        "num_instances": float(len(kept_ids)),
        "within_instance_rgb_dist": within_mean,
        "between_instance_rgb_dist": between,
        "between_within_ratio": between / (within_mean + 1e-8),
    }


def _neighbor_stats(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int,
    seed: int,
    k: int,
) -> dict[str, float]:
    idx = _subsample(xyz.shape[0], max_points, seed)
    xyz = xyz[idx]
    rgb = rgb[idx]
    labels = labels[idx]
    valid = labels >= 0
    xyz = xyz[valid]
    rgb = rgb[valid]
    labels = labels[valid]
    if xyz.shape[0] <= k + 1:
        return {
            "spatial_same_instance_rgb_dist": np.nan,
            "spatial_diff_instance_rgb_dist": np.nan,
            "spatial_boundary_contrast": np.nan,
            "rgb_knn_instance_purity": np.nan,
        }

    spatial_tree = cKDTree(xyz)
    _d, neigh = spatial_tree.query(xyz, k=k + 1)
    neigh = neigh[:, 1:]
    color_d = np.linalg.norm(rgb[:, None, :] - rgb[neigh], axis=2)
    same = labels[:, None] == labels[neigh]
    same_vals = color_d[same]
    diff_vals = color_d[~same]

    rgb_tree = cKDTree(rgb)
    _cd, color_neigh = rgb_tree.query(rgb, k=k + 1)
    color_neigh = color_neigh[:, 1:]
    purity = (labels[:, None] == labels[color_neigh]).mean()

    same_mean = float(same_vals.mean()) if same_vals.size else np.nan
    diff_mean = float(diff_vals.mean()) if diff_vals.size else np.nan
    return {
        "spatial_same_instance_rgb_dist": same_mean,
        "spatial_diff_instance_rgb_dist": diff_mean,
        "spatial_boundary_contrast": diff_mean / (same_mean + 1e-8),
        "rgb_knn_instance_purity": float(purity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-root", required=True, help="Root containing per-scene PCA exports")
    parser.add_argument("--scans-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--granularity", default="g0.5")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument(
        "--ply-names",
        nargs="+",
        default=[
            "E2_projected_points_pca.ply",
            "E3_projected_points_pca.ply",
            "E4_projected_points_pca.ply",
            "Eall_concat_l2_projected_points_pca.ply",
        ],
    )
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--min-instance-points", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    pca_root = Path(args.pca_root)
    scans_root = Path(args.scans_root)
    label_file = f"labels_{args.granularity}.npy"
    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        labels = np.load(scans_root / scene / "training_pack" / label_file)
        for ply_name in args.ply_names:
            ply_path = pca_root / scene / ply_name
            if not ply_path.exists():
                continue
            xyz, rgb = _load_ply(ply_path)
            if xyz.shape[0] != labels.shape[0]:
                raise ValueError(f"{ply_path} has {xyz.shape[0]} points but labels have {labels.shape[0]}")
            inst = _instance_stats(rgb, labels, min_instance_points=args.min_instance_points)
            neigh = _neighbor_stats(
                xyz,
                rgb,
                labels,
                max_points=args.max_points,
                seed=abs(hash((scene, ply_name))) % (2**32),
                k=args.k,
            )
            rows.append({"scene": scene, "ply": ply_name, **inst, **neigh})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["scene", "ply"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = out.with_name(out.stem + "_summary.csv")
    if rows:
        with summary.open("w", newline="") as f:
            numeric = [k for k, v in rows[0].items() if isinstance(v, float)]
            writer = csv.DictWriter(f, fieldnames=["ply", *numeric])
            writer.writeheader()
            for ply in sorted({str(r["ply"]) for r in rows}):
                group = [r for r in rows if r["ply"] == ply]
                writer.writerow(
                    {
                        "ply": ply,
                        **{
                            k: float(np.nanmean([float(r[k]) for r in group]))
                            for k in numeric
                        },
                    }
                )
    print(out)
    print(summary)


if __name__ == "__main__":
    main()
