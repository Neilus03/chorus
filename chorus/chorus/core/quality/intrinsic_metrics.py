from __future__ import annotations

from typing import Any

from chorus.common.types import ClusterOutput


def compute_cluster_intrinsic_metrics(cluster_output: ClusterOutput) -> dict[str, Any]:
    stats = dict(cluster_output.stats)

    num_points = int(stats.get("num_points", 0))
    unseen_points = int(stats.get("unseen_points", 0))
    num_clusters = int(stats.get("num_clusters", 0))
    num_noise_points = int(stats.get("num_noise_points", 0))
    num_2d_masks_total = int(stats.get("num_2d_masks_total", 0))
    used_frames = int(stats.get("used_frames", 0))

    labeled_points = max(num_points - num_noise_points, 0)
    seen_points = max(num_points - unseen_points, 0)

    metrics = {
        "granularity": float(cluster_output.granularity),
        "num_points": num_points,
        "seen_points": seen_points,
        "seen_points_fraction": float(seen_points / max(num_points, 1)),
        "unseen_points": unseen_points,
        "unseen_points_fraction": float(unseen_points / max(num_points, 1)),
        "num_clusters": num_clusters,
        "num_noise_points": num_noise_points,
        "noise_fraction": float(num_noise_points / max(num_points, 1)),
        "num_2d_masks_total": num_2d_masks_total,
        "used_frames": used_frames,
        "masks_per_used_frame": float(num_2d_masks_total / max(used_frames, 1)),
        "labeled_points": labeled_points,
        "labeled_points_fraction": float(labeled_points / max(num_points, 1)),
        "clusters_per_10k_points": float(num_clusters * 10000.0 / max(num_points, 1)),
        "explained_variance_sum": float(stats.get("explained_variance_sum", 0.0)),
        "svd_components": int(stats.get("svd_components", 0)),
    }
    return metrics


def compute_scene_intrinsic_metrics(cluster_outputs: list[ClusterOutput]) -> dict[str, Any]:
    by_granularity = {
        f"g{cluster_output.granularity}": compute_cluster_intrinsic_metrics(cluster_output)
        for cluster_output in cluster_outputs
    }

    if len(cluster_outputs) == 0:
        return {
            "num_granularities": 0,
            "by_granularity": {},
        }

    avg_noise_fraction = sum(
        float(co.stats.get("noise_fraction", 0.0)) for co in cluster_outputs
    ) / len(cluster_outputs)

    avg_unseen_fraction = sum(
        float(co.stats.get("unseen_points_fraction", 0.0)) for co in cluster_outputs
    ) / len(cluster_outputs)

    total_clusters = sum(int(co.stats.get("num_clusters", 0)) for co in cluster_outputs)

    return {
        "num_granularities": len(cluster_outputs),
        "avg_noise_fraction": float(avg_noise_fraction),
        "avg_unseen_fraction": float(avg_unseen_fraction),
        "total_clusters_across_granularities": int(total_clusters),
        "by_granularity": by_granularity,
    }