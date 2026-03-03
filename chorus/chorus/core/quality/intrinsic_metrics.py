from __future__ import annotations

from typing import Any

import numpy as np

from chorus.common.types import ClusterOutput


def compute_cluster_intrinsic_metrics(cluster_output: ClusterOutput) -> dict[str, Any]:
    stats = dict(cluster_output.stats)

    labels = cluster_output.labels
    seen_mask = cluster_output.seen_mask

    num_points = int(labels.shape[0])
    num_seen_points = int(np.sum(seen_mask))
    num_unseen_points = int(num_points - num_seen_points)

    labeled_mask = labels >= 0
    num_labeled_points = int(np.sum(labeled_mask))

    noise_seen_mask = seen_mask & (labels < 0)
    num_noise_points_seen = int(np.sum(noise_seen_mask))

    num_clusters = int(stats.get("num_clusters", 0))
    num_2d_masks_total = int(stats.get("num_2d_masks_total", 0))
    used_frames = int(stats.get("used_frames", 0))

    metrics = {
        "granularity": float(cluster_output.granularity),
        "num_points": num_points,
        "num_seen_points": num_seen_points,
        "seen_points_fraction": float(num_seen_points / max(num_points, 1)),
        "num_unseen_points": num_unseen_points,
        "unseen_points_fraction": float(num_unseen_points / max(num_points, 1)),
        "num_labeled_points": num_labeled_points,
        "labeled_points_fraction": float(num_labeled_points / max(num_points, 1)),
        "labeled_points_fraction_seen": float(num_labeled_points / max(num_seen_points, 1)),
        "num_noise_points_seen": num_noise_points_seen,
        "noise_fraction_seen": float(num_noise_points_seen / max(num_seen_points, 1)),
        "noise_fraction_all_points": float(num_noise_points_seen / max(num_points, 1)),
        "num_clusters": num_clusters,
        "num_2d_masks_total": num_2d_masks_total,
        "used_frames": used_frames,
        "masks_per_used_frame": float(num_2d_masks_total / max(used_frames, 1)),
        "clusters_per_10k_points": float(num_clusters * 10000.0 / max(num_points, 1)),
        "clusters_per_10k_seen_points": float(num_clusters * 10000.0 / max(num_seen_points, 1)),
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

    avg_noise_fraction_seen = sum(
        float(by_granularity[f"g{co.granularity}"]["noise_fraction_seen"])
        for co in cluster_outputs
    ) / len(cluster_outputs)

    avg_unseen_fraction = sum(
        float(by_granularity[f"g{co.granularity}"]["unseen_points_fraction"])
        for co in cluster_outputs
    ) / len(cluster_outputs)

    avg_labeled_fraction_seen = sum(
        float(by_granularity[f"g{co.granularity}"]["labeled_points_fraction_seen"])
        for co in cluster_outputs
    ) / len(cluster_outputs)

    total_clusters = sum(int(co.stats.get("num_clusters", 0)) for co in cluster_outputs)

    return {
        "num_granularities": len(cluster_outputs),
        "avg_noise_fraction_seen": float(avg_noise_fraction_seen),
        "avg_unseen_fraction": float(avg_unseen_fraction),
        "avg_labeled_fraction_seen": float(avg_labeled_fraction_seen),
        "total_clusters_across_granularities": int(total_clusters),
        "by_granularity": by_granularity,
    }