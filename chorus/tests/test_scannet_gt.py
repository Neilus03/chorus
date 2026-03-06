from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids


class _FakePlyData:
    def __init__(self, n_vertices: int):
        self.elements = [SimpleNamespace(data=np.zeros(n_vertices, dtype=np.int32))]


def _write_scene_files(scene_dir: Path, scene_name: str) -> None:
    (scene_dir / f"{scene_name}_vh_clean_2.labels.ply").touch()

    with (scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"segIndices": [0, 0, 1, 2]}, f)

    with (scene_dir / f"{scene_name}.aggregation.json").open("w", encoding="utf-8") as f:
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


def test_load_scannet_gt_instance_ids_filters_to_scannet200(monkeypatch, tmp_path: Path) -> None:
    scene_name = "scene0000_00"
    scene_dir = tmp_path / scene_name
    scene_dir.mkdir()
    _write_scene_files(scene_dir, scene_name)

    monkeypatch.setattr(
        "chorus.datasets.scannet.gt.PlyData.read",
        lambda _: _FakePlyData(n_vertices=4),
    )
    monkeypatch.setattr(
        "chorus.datasets.scannet.gt.load_raw_category_label_map",
        lambda: {
            "chair": {"id": 2, "nyu40id": 5},
            "wall": {"id": 1, "nyu40id": 1},
            "mystery": {"id": 5000, "nyu40id": 0},
        },
    )

    gt_ids = load_scannet_gt_instance_ids(
        scene_dir=scene_dir,
        scene_name=scene_name,
        eval_benchmark="scannet200",
    )

    np.testing.assert_array_equal(gt_ids, np.array([1, 1, 0, 0], dtype=np.int64))


def test_load_scannet_gt_instance_ids_all_keeps_nonstuff_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene_name = "scene0000_00"
    scene_dir = tmp_path / scene_name
    scene_dir.mkdir()
    _write_scene_files(scene_dir, scene_name)

    monkeypatch.setattr(
        "chorus.datasets.scannet.gt.PlyData.read",
        lambda _: _FakePlyData(n_vertices=4),
    )

    gt_ids = load_scannet_gt_instance_ids(
        scene_dir=scene_dir,
        scene_name=scene_name,
        eval_benchmark="all",
    )

    np.testing.assert_array_equal(gt_ids, np.array([1, 1, 0, 3], dtype=np.int64))
