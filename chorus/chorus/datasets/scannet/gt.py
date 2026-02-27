from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from chorus.datasets.scannet.metadata import IGNORE_INSTANCE_CLASSES


def _load_instance_ids_from_aggregation(
    scene_dir: Path,
    scene_name: str,
    n_vertices: int,
) -> np.ndarray | None:
    seg_paths = [
        scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_name}_vh_clean.segs.json",
    ]
    agg_paths = [
        scene_dir / f"{scene_name}.aggregation.json",
        scene_dir / f"{scene_name}_vh_clean.aggregation.json",
    ]

    seg_path = next((p for p in seg_paths if p.exists()), None)
    agg_path = next((p for p in agg_paths if p.exists()), None)
    if seg_path is None or agg_path is None:
        return None

    with seg_path.open("r", encoding="utf-8") as f:
        seg_json = json.load(f)
    with agg_path.open("r", encoding="utf-8") as f:
        agg_json = json.load(f)

    seg_indices = np.asarray(seg_json.get("segIndices", []), dtype=np.int64)
    if seg_indices.shape[0] != n_vertices:
        raise RuntimeError(
            f"segIndices length ({seg_indices.shape[0]}) != num vertices ({n_vertices})"
        )

    seg_to_instance: dict[int, int] = {}
    for group in agg_json.get("segGroups", []):
        label = group.get("label", "").lower()
        if label in IGNORE_INSTANCE_CLASSES:
            continue

        inst_id = int(group.get("objectId", group.get("id", -1)))
        if inst_id < 0:
            continue

        for seg_id in group.get("segments", []):
            seg_to_instance[int(seg_id)] = inst_id

    gt_instance_ids = np.zeros(n_vertices, dtype=np.int64)
    for i, seg_id in enumerate(seg_indices):
        gt_instance_ids[i] = seg_to_instance.get(int(seg_id), 0)

    return gt_instance_ids


def load_scannet_gt_instance_ids(scene_dir: Path, scene_name: str) -> np.ndarray:
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    if not labels_ply.exists():
        raise FileNotFoundError(f"Missing ScanNet labels ply: {labels_ply}")

    plydata = PlyData.read(str(labels_ply))
    n_vertices = len(plydata.elements[0].data)

    gt_instance_ids = _load_instance_ids_from_aggregation(scene_dir, scene_name, n_vertices)
    if gt_instance_ids is not None:
        return gt_instance_ids

    raise RuntimeError("Could not find GT instance ids in aggregation+segments files.")