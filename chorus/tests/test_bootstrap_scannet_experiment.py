from __future__ import annotations

from pathlib import Path

import numpy as np

from chorus.common.types import ClusterOutput, FrameRecord, GeometryRecord, VisibilityConfig
from chorus.core.experiments.bootstrap_scannet import (
    fuse_stable_core_labels,
    split_frames_for_bootstraps,
)
from chorus.datasets.base import SceneAdapter
from chorus.export.training_pack import export_training_scene_pack


def _frame(frame_id: int, root: Path) -> FrameRecord:
    return FrameRecord(
        frame_id=str(frame_id),
        rgb_path=root / f"{frame_id}.jpg",
        depth_path=root / f"{frame_id}.png",
        pose_path=root / f"{frame_id}.txt",
        intrinsics_path=root / "intrinsic.txt",
    )


def test_split_frames_for_bootstraps_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    frames = [_frame(i, tmp_path) for i in range(10)]

    subsets_a = split_frames_for_bootstraps(
        frames,
        num_bootstraps=4,
        frame_fraction=0.25,
        seed=7,
    )
    subsets_b = split_frames_for_bootstraps(
        frames,
        num_bootstraps=4,
        frame_fraction=0.25,
        seed=7,
    )

    ids_a = [[f.frame_id for f in subset] for subset in subsets_a]
    ids_b = [[f.frame_id for f in subset] for subset in subsets_b]
    assert ids_a == ids_b

    flat = [frame_id for subset in ids_a for frame_id in subset]
    assert len(flat) == len(set(flat))
    assert [len(subset) for subset in subsets_a] == [3, 3, 3, 1]


def test_split_frames_for_bootstraps_honors_max_frames_cap(tmp_path: Path) -> None:
    frames = [_frame(i, tmp_path) for i in range(100)]

    subsets = split_frames_for_bootstraps(
        frames,
        num_bootstraps=4,
        frame_fraction=0.25,
        seed=7,
        max_frames_per_bootstrap=8,
    )

    assert [len(subset) for subset in subsets] == [8, 8, 8, 8]


def test_split_frames_for_bootstraps_all_repeats_full_frame_list(tmp_path: Path) -> None:
    frames = [_frame(i, tmp_path) for i in range(5)]

    subsets = split_frames_for_bootstraps(
        frames,
        num_bootstraps=3,
        frame_fraction=0.25,
        frame_sampling="all",
        seed=7,
    )

    assert [[f.frame_id for f in subset] for subset in subsets] == [
        ["0", "1", "2", "3", "4"],
        ["0", "1", "2", "3", "4"],
        ["0", "1", "2", "3", "4"],
    ]


def test_fuse_stable_core_keeps_repeated_clusters_and_drops_singletons() -> None:
    labels = [
        np.array([0, 0, 0, -1, 1, 1, -1, -1], dtype=np.int32),
        np.array([0, 0, -1, -1, 1, 1, -1, -1], dtype=np.int32),
        np.array([0, 0, 0, -1, 2, 2, -1, 3], dtype=np.int32),
        np.array([5, 5, -1, -1, -1, -1, 6, 6], dtype=np.int32),
    ]

    fused, stats = fuse_stable_core_labels(
        labels,
        support_threshold=0.5,
        cluster_iou_threshold=0.35,
        min_fused_points=2,
    )

    assert stats["required_support"] == 2
    assert stats["num_fused_instances"] == 2
    assert set(np.unique(fused)) == {-1, 0, 1}
    assert np.array_equal(fused[:2], np.array([0, 0], dtype=np.int32))
    assert np.array_equal(fused[4:6], np.array([1, 1], dtype=np.int32))
    assert fused[7] == -1


def test_fuse_stable_core_outputs_contiguous_non_negative_ids() -> None:
    labels = [
        np.array([10, 10, -1, 20, 20, -1], dtype=np.int32),
        np.array([30, 30, -1, 40, 40, -1], dtype=np.int32),
    ]

    fused, _ = fuse_stable_core_labels(
        labels,
        support_threshold=1.0,
        cluster_iou_threshold=0.5,
        min_fused_points=1,
    )

    non_negative = sorted(int(x) for x in np.unique(fused) if x >= 0)
    assert non_negative == list(range(len(non_negative)))
    assert fused[2] == -1
    assert fused[5] == -1


class _TinyAdapter(SceneAdapter):
    @property
    def dataset_name(self) -> str:
        return "tiny"

    def prepare(self) -> None:
        return None

    def list_frames(self) -> list[FrameRecord]:
        return []

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        raise NotImplementedError

    def load_geometry_points(self) -> np.ndarray:
        return np.zeros((6, 3), dtype=np.float32)

    def load_geometry_colors(self) -> np.ndarray | None:
        return np.ones((6, 3), dtype=np.float32)

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self.scene_root / "mesh.ply",
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=0.1,
            z_tolerance_m=0.05,
            depth_scale_to_m=0.001,
        )


def test_synthetic_fused_training_pack_export(tmp_path: Path) -> None:
    adapter = _TinyAdapter(tmp_path / "scene0000_00")
    labels = np.array([0, 0, -1, 1, 1, -1], dtype=np.int32)
    cluster_output = ClusterOutput(
        granularity=0.5,
        labels=labels,
        features=np.zeros((6, 2), dtype=np.float32),
        seen_mask=np.array([True, True, True, True, True, False]),
        ply_path=None,
        labels_path=None,
        stats={"num_clusters": 2},
    )

    pack_dir = export_training_scene_pack(
        adapter=adapter,
        cluster_outputs=[cluster_output],
        output_dir=tmp_path / "experiment" / "fused" / "training_pack",
        teacher_name="test_bootstrap",
    )

    assert (pack_dir / "points.npy").exists()
    assert (pack_dir / "labels_g0.5.npy").exists()
    assert (pack_dir / "scene_meta.json").exists()
    np.testing.assert_array_equal(np.load(pack_dir / "labels_g0.5.npy"), labels)
