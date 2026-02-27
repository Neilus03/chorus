from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chorus.common.types import ClusterOutput
from chorus.datasets.base import SceneAdapter


def export_litept_scene_pack(
    adapter: SceneAdapter,
    cluster_outputs: list[ClusterOutput],
    output_dir: Path | None = None,
) -> Path:
    if len(cluster_outputs) == 0:
        raise RuntimeError("Cannot export LitePT pack with zero cluster outputs.")

    output_dir = Path(output_dir) if output_dir is not None else (adapter.scene_root / "litept_pack")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = adapter.load_geometry_points()
    colors = adapter.load_geometry_colors()

    np.save(output_dir / "points.npy", points)

    if colors is not None:
        np.save(output_dir / "colors.npy", colors)

    label_file_map = {}
    cluster_stats = {}

    valid_stack = []
    for cluster_output in cluster_outputs:
        granularity_tag = str(cluster_output.granularity)
        labels_file = output_dir / f"labels_g{granularity_tag}.npy"
        np.save(labels_file, cluster_output.labels)

        label_file_map[granularity_tag] = labels_file.name
        cluster_stats[granularity_tag] = cluster_output.stats
        valid_stack.append(cluster_output.labels >= 0)

    valid_points = np.any(np.stack(valid_stack, axis=0), axis=0)
    np.save(output_dir / "valid_points.npy", valid_points.astype(np.uint8))

    frames = adapter.list_frames()
    scene_meta = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "geometry_type": adapter.get_geometry_record().geometry_type,
        "num_points": int(points.shape[0]),
        "num_frames_total": int(len(frames)),
        "granularities": [float(c.granularity) for c in cluster_outputs],
        "label_files": label_file_map,
        "cluster_stats": cluster_stats,
    }

    with (output_dir / "scene_meta.json").open("w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2)

    print(f"Exported LitePT scene pack: {output_dir}")
    return output_dir