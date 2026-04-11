from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from chorus.datasets.scannetpp.benchmark import (
    DEFAULT_SCANNETPP_EVAL_BENCHMARK,
    map_scannetpp_instance_label,
    normalize_scannetpp_eval_benchmark,
)


def scannetpp_gt_cache_path(
    scene_dir: Path,
    eval_benchmark: str = DEFAULT_SCANNETPP_EVAL_BENCHMARK,
) -> Path:
    normalized = normalize_scannetpp_eval_benchmark(eval_benchmark)
    if normalized == DEFAULT_SCANNETPP_EVAL_BENCHMARK:
        return Path(scene_dir) / "gt_instance_ids.npy"
    return Path(scene_dir) / f"gt_instance_ids_{normalized}.npy"


def _iter_seg_groups(anno_json: object) -> list[dict]:
    if isinstance(anno_json, dict):
        for key in ("segGroups", "seg_groups"):
            value = anno_json.get(key)
            if isinstance(value, list):
                return [group for group in value if isinstance(group, dict)]
    if isinstance(anno_json, list):
        return [group for group in anno_json if isinstance(group, dict)]
    raise RuntimeError("Could not parse ScanNet++ segment annotations.")


def load_scannetpp_gt_instance_ids(
    scene_dir: Path,
    scene_name: str | None = None,
    eval_benchmark: str = DEFAULT_SCANNETPP_EVAL_BENCHMARK,
) -> np.ndarray:
    normalized_benchmark = normalize_scannetpp_eval_benchmark(eval_benchmark)
    scene_dir = Path(scene_dir)

    mesh_path = scene_dir / "scans" / "mesh_aligned_0.05.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing ScanNet++ geometry file: {mesh_path}")

    segments_path = scene_dir / "scans" / "segments.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing ScanNet++ mesh segment file: {segments_path}")

    annotations_path = scene_dir / "scans" / "segments_anno.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing ScanNet++ annotation file: {annotations_path}")

    plydata = PlyData.read(str(mesh_path))
    if "vertex" not in plydata:
        raise RuntimeError(f"PLY file has no vertex element: {mesh_path}")
    n_vertices = len(plydata["vertex"].data)

    with segments_path.open("r", encoding="utf-8") as f:
        segments_json = json.load(f)
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations_json = json.load(f)

    seg_indices = np.asarray(
        segments_json.get("segIndices", []),
        dtype=np.int64,
    )
    if seg_indices.shape[0] != n_vertices:
        raise RuntimeError(
            f"segIndices length ({seg_indices.shape[0]}) != num vertices ({n_vertices})"
        )

    seg_to_instance: dict[int, int] = {}
    for group in _iter_seg_groups(annotations_json):
        mapped_label = map_scannetpp_instance_label(
            label=str(group.get("label", "")),
            eval_benchmark=normalized_benchmark,
            scene_root=scene_dir,
        )
        if mapped_label is None:
            continue

        raw_instance_id = int(group.get("objectId", group.get("id", -1)))
        if raw_instance_id < 0:
            continue

        instance_id = raw_instance_id + 1
        for seg_id in group.get("segments", []):
            seg_to_instance[int(seg_id)] = instance_id

    gt_instance_ids = np.zeros(n_vertices, dtype=np.int64)
    for vertex_idx, seg_id in enumerate(seg_indices):
        gt_instance_ids[vertex_idx] = seg_to_instance.get(int(seg_id), 0)

    return gt_instance_ids
