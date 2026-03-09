from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chorus.common.types import ClusterOutput
from chorus.datasets.base import SceneAdapter


def export_training_scene_pack(
    adapter: SceneAdapter,
    cluster_outputs: list[ClusterOutput],
    output_dir: Path | None = None,
    teacher_name: str = "unsamv2",
    projection_type: str = "zbuffer_rgbd",
    embedding_type: str = "truncated_svd",
    clustering_type: str = "hdbscan",
    frame_skip: int | None = None,
    scene_intrinsic_metrics: dict[str, Any] | None = None,
) -> Path:
    if len(cluster_outputs) == 0:
        raise RuntimeError("Cannot export training pack with zero cluster outputs.")

    output_dir = Path(output_dir) if output_dir is not None else (adapter.scene_root / "training_pack")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = adapter.load_geometry_points()
    colors = adapter.load_geometry_colors()
    geometry_record = adapter.get_geometry_record()
    frames = adapter.list_frames()

    np.save(output_dir / "points.npy", points)

    if colors is not None:
        np.save(output_dir / "colors.npy", colors)

    label_file_map: dict[str, str] = {}
    cluster_stats: dict[str, dict] = {}

    valid_stack = []
    seen_stack = []

    for cluster_output in cluster_outputs:
        granularity_key = f"g{cluster_output.granularity}"
        labels_file = output_dir / f"labels_{granularity_key}.npy"
        np.save(labels_file, cluster_output.labels)

        label_file_map[granularity_key] = labels_file.name
        cluster_stats[granularity_key] = cluster_output.stats

        valid_stack.append(cluster_output.labels >= 0)
        seen_stack.append(cluster_output.seen_mask)

    valid_points = np.any(np.stack(valid_stack, axis=0), axis=0)
    seen_points = np.any(np.stack(seen_stack, axis=0), axis=0)

    supervision_mask = valid_points.copy()

    np.save(output_dir / "valid_points.npy", valid_points.astype(np.uint8))
    np.save(output_dir / "seen_points.npy", seen_points.astype(np.uint8))
    np.save(output_dir / "supervision_mask.npy", supervision_mask.astype(np.uint8))

    num_frames_total = len(frames)
    num_frames_used = len(frames[::frame_skip]) if frame_skip is not None and frame_skip > 0 else num_frames_total

    scene_meta = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "geometry_type": geometry_record.geometry_type,
        "geometry_source": geometry_record.geometry_path.name,
        "geometry_path_name": geometry_record.geometry_path.name,
        "num_points": int(points.shape[0]),
        "num_frames_total": int(num_frames_total),
        "num_frames_used": int(num_frames_used),
        "frame_skip": int(frame_skip) if frame_skip is not None else None,
        "granularities": [float(c.granularity) for c in cluster_outputs],
        "label_files": label_file_map,
        "valid_points_file": "valid_points.npy",
        "seen_points_file": "seen_points.npy",
        "supervision_mask_file": "supervision_mask.npy",
        "teacher_name": teacher_name,
        "projection_type": projection_type,
        "embedding_type": embedding_type,
        "clustering_type": clustering_type,
        "cluster_stats": cluster_stats,
        "scene_intrinsic_metrics": scene_intrinsic_metrics if scene_intrinsic_metrics is not None else {},
    }

    with (output_dir / "scene_meta.json").open("w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2)

    print(f"Exported training scene pack: {output_dir}")
    return output_dir
