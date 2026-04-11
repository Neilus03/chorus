from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chorus.datasets.scannetpp.gt import load_scannetpp_gt_instance_ids


class _FakePlyData:
    def __init__(self, n_vertices: int):
        self._vertex = SimpleNamespace(data=np.zeros(n_vertices, dtype=np.int32))

    def __contains__(self, key: str) -> bool:
        return key == "vertex"

    def __getitem__(self, key: str):
        if key != "vertex":
            raise KeyError(key)
        return self._vertex


def _write_metadata(dataset_root: Path) -> None:
    metadata_root = dataset_root / "metadata"
    benchmark_root = metadata_root / "semantic_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    (metadata_root / "instance_classes.txt").write_text(
        "chair\nmystery\n",
        encoding="utf-8",
    )
    (benchmark_root / "top100_instance.txt").write_text("chair\n", encoding="utf-8")
    (benchmark_root / "map_benchmark.csv").write_text(
        "class,instance_map_to\nchair,chair\nmystery,None\nwall,None\n",
        encoding="utf-8",
    )


def _write_scene(scene_dir: Path) -> None:
    scans_dir = scene_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)
    (scans_dir / "mesh_aligned_0.05.ply").touch()

    with (scans_dir / "segments.json").open("w", encoding="utf-8") as f:
        json.dump({"segIndices": [0, 0, 1, 2]}, f)

    with (scans_dir / "segments_anno.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segGroups": [
                    {"objectId": 0, "label": "chair", "segments": [0]},
                    {"objectId": 1, "label": "wall", "segments": [1]},
                    {"objectId": 2, "label": "mystery", "segments": [2]},
                ]
            },
            f,
        )


def test_load_scannetpp_gt_instance_ids_filters_to_top100(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "scannetpp_data"
    scene_dir = dataset_root / "data" / "abcd1234"
    _write_metadata(dataset_root)
    _write_scene(scene_dir)

    monkeypatch.setattr(
        "chorus.datasets.scannetpp.gt.PlyData.read",
        lambda _: _FakePlyData(n_vertices=4),
    )

    gt_ids = load_scannetpp_gt_instance_ids(
        scene_dir=scene_dir,
        eval_benchmark="top100_instance",
    )

    np.testing.assert_array_equal(gt_ids, np.array([1, 1, 0, 0], dtype=np.int64))


def test_load_scannetpp_gt_instance_ids_all_keeps_nonstuff_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "scannetpp_data"
    scene_dir = dataset_root / "data" / "abcd1234"
    _write_metadata(dataset_root)
    _write_scene(scene_dir)

    monkeypatch.setattr(
        "chorus.datasets.scannetpp.gt.PlyData.read",
        lambda _: _FakePlyData(n_vertices=4),
    )

    gt_ids = load_scannetpp_gt_instance_ids(
        scene_dir=scene_dir,
        eval_benchmark="all",
    )

    np.testing.assert_array_equal(gt_ids, np.array([1, 1, 0, 3], dtype=np.int64))
