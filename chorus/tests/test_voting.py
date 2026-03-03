from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    

from chorus.common.types import (
    ClusterOutput,
    FrameRecord,
    GeometryRecord,
    TeacherOutput,
    VisibilityConfig,
)
from chorus.core.lifting.voting import build_point_mask_matrix
import chorus.core.pipeline.project_cluster_stage as project_cluster_stage
from chorus.datasets.base import SceneAdapter
from chorus.export.litept_pack import export_litept_scene_pack


class DummySceneAdapter(SceneAdapter):
    def __init__(
        self,
        scene_root: Path,
        points: np.ndarray,
        depth_map_m: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        pose_c2w: np.ndarray | None = None,
    ) -> None:
        super().__init__(scene_root=scene_root)
        self._points = np.asarray(points, dtype=np.float32)
        self._depth_map_m = (
            np.asarray(depth_map_m, dtype=np.float32)
            if depth_map_m is not None
            else np.ones((2, 2), dtype=np.float32)
        )
        self._intrinsics = (
            np.asarray(intrinsics, dtype=np.float32)
            if intrinsics is not None
            else np.eye(3, dtype=np.float32)
        )
        self._pose_c2w = (
            np.asarray(pose_c2w, dtype=np.float32)
            if pose_c2w is not None
            else np.eye(4, dtype=np.float32)
        )

        self._frame = FrameRecord(
            frame_id="0",
            rgb_path=self.scene_root / "color" / "0.jpg",
            depth_path=self.scene_root / "depth" / "0.png",
            pose_path=self.scene_root / "pose" / "0.txt",
            intrinsics_path=self.scene_root / "intrinsic" / "intrinsic_color.txt",
        )

    @property
    def dataset_name(self) -> str:
        return "dummy"

    def prepare(self) -> None:
        pass

    def list_frames(self) -> list[FrameRecord]:
        return [self._frame]

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        h, w = self._depth_map_m.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        return self._depth_map_m

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        return self._pose_c2w

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return self._intrinsics

    def load_geometry_points(self) -> np.ndarray:
        return self._points

    def load_geometry_colors(self) -> np.ndarray | None:
        return None

    def get_geometry_record(self) -> GeometryRecord:
        geometry_path = self.scene_root / f"{self.scene_id}.ply"
        geometry_path.touch(exist_ok=True)
        return GeometryRecord(
            geometry_path=geometry_path,
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=0.1,
            z_tolerance_m=0.2,
            depth_scale_to_m=1.0,
            depth_aligned_to_rgb=True,
        )


def _make_cluster_output(
    granularity: float,
    labels: np.ndarray,
    seen_mask: np.ndarray,
) -> ClusterOutput:
    labels = np.asarray(labels, dtype=np.int32)
    seen_mask = np.asarray(seen_mask, dtype=bool)

    return ClusterOutput(
        granularity=granularity,
        labels=labels,
        features=np.zeros((labels.shape[0], 1), dtype=np.float32),
        seen_mask=seen_mask,
        ply_path=None,
        labels_path=None,
        stats={"num_clusters": int(len(np.unique(labels[labels >= 0])))},
    )


def test_build_point_mask_matrix_counts_and_ignores_background() -> None:
    point_assignments = [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
    ]
    mask_assignments = [
        np.array([1, 1, 0, 2], dtype=np.int32),
        np.array([0, 3, 3], dtype=np.int32),
    ]

    matrix, stats = build_point_mask_matrix(
        point_assignments=point_assignments,
        mask_assignments=mask_assignments,
        num_points=5,
    )

    dense = matrix.toarray()

    assert dense.shape == (5, 3)
    assert stats["num_points"] == 5
    assert stats["num_2d_masks_total"] == 3
    assert stats["avg_masks_per_frame"] == pytest.approx(1.5)

    expected = np.array(
        [
            [1, 0, 0],  # point 0 -> frame 1, mask 1
            [1, 0, 0],  # point 1 -> frame 1, mask 1
            [0, 0, 1],  # point 2 -> frame 2, mask 3
            [0, 1, 1],  # point 3 -> frame 1, mask 2 and frame 2, mask 3
            [0, 0, 0],  # point 4 -> unseen / no assignments
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(dense, expected)


def test_build_point_mask_matrix_raises_when_all_assignments_are_background() -> None:
    point_assignments = [np.array([0, 1, 2], dtype=np.int64)]
    mask_assignments = [np.array([0, 0, 0], dtype=np.int32)]

    with pytest.raises(RuntimeError, match="No masks were bridged to 3D"):
        build_point_mask_matrix(
            point_assignments=point_assignments,
            mask_assignments=mask_assignments,
            num_points=3,
        )


def test_run_project_cluster_stage_keeps_zero_vote_points_unseen_and_unlabeled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],   # visible, pixel (0, 0)
            [1.0, 0.0, 1.0],   # visible, pixel (1, 0)
            [10.0, 0.0, 1.0],  # projects out of image bounds, zero votes
        ],
        dtype=np.float32,
    )
    depth_map = np.ones((2, 2), dtype=np.float32)

    adapter = DummySceneAdapter(
        scene_root=tmp_path / "scene_dummy",
        points=points,
        depth_map_m=depth_map,
        intrinsics=np.eye(3, dtype=np.float32),
        pose_c2w=np.eye(4, dtype=np.float32),
    )
    adapter.scene_root.mkdir(parents=True, exist_ok=True)

    mask = np.array(
        [
            [1, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )
    mask_path = adapter.scene_root / "0.npy"
    np.save(mask_path, mask)

    teacher_output = TeacherOutput(
        granularity=0.2,
        frame_mask_paths=[mask_path],
        total_masks=1,
    )

    def fake_compute_svd_features(point_mask_matrix, n_components):
        features = np.array([[1.0], [1.0]], dtype=np.float32)
        stats = {"svd_components": 1, "explained_variance_sum": 1.0}
        return features, stats

    def fake_cluster_features(features, min_cluster_size, min_samples, cluster_selection_epsilon):
        labels = np.array([0, 0], dtype=np.int32)
        stats = {
            "num_clusters": 1,
            "num_noise_points": 0,
            "noise_fraction": 0.0,
        }
        return labels, stats

    monkeypatch.setattr(project_cluster_stage, "compute_svd_features", fake_compute_svd_features)
    monkeypatch.setattr(project_cluster_stage, "cluster_features", fake_cluster_features)

    result = project_cluster_stage.run_project_cluster_stage(
        adapter=adapter,
        teacher_output=teacher_output,
        frame_skip=1,
        svd_components=4,
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        save_outputs=False,
    )

    np.testing.assert_array_equal(result.seen_mask, np.array([True, True, False]))
    np.testing.assert_array_equal(result.labels, np.array([0, 0, -1], dtype=np.int32))

    assert result.stats["num_seen_points"] == 2
    assert result.stats["unseen_points"] == 1
    assert result.stats["num_labeled_points"] == 2
    assert result.stats["num_noise_points_seen"] == 0


def test_export_litept_scene_pack_excludes_unseen_points_from_valid_points(
    tmp_path: Path,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # seen and valid
            [1.0, 0.0, 0.0],  # seen and valid at another granularity
            [2.0, 0.0, 0.0],  # seen but always noise
            [3.0, 0.0, 0.0],  # unseen everywhere
        ],
        dtype=np.float32,
    )
    adapter = DummySceneAdapter(
        scene_root=tmp_path / "scene_export",
        points=points,
    )
    adapter.scene_root.mkdir(parents=True, exist_ok=True)

    cluster_outputs = [
        _make_cluster_output(
            granularity=0.2,
            labels=np.array([0, -1, -1, -1], dtype=np.int32),
            seen_mask=np.array([True, True, True, False]),
        ),
        _make_cluster_output(
            granularity=0.5,
            labels=np.array([-1, 1, -1, -1], dtype=np.int32),
            seen_mask=np.array([True, True, True, False]),
        ),
    ]

    output_dir = export_litept_scene_pack(
        adapter=adapter,
        cluster_outputs=cluster_outputs,
        output_dir=tmp_path / "litept_pack",
        frame_skip=1,
    )

    valid_points = np.load(output_dir / "valid_points.npy").astype(bool)
    seen_points = np.load(output_dir / "seen_points.npy").astype(bool)
    supervision_mask = np.load(output_dir / "supervision_mask.npy").astype(bool)

    np.testing.assert_array_equal(
        valid_points,
        np.array([True, True, False, False]),
    )
    np.testing.assert_array_equal(
        seen_points,
        np.array([True, True, True, False]),
    )
    np.testing.assert_array_equal(supervision_mask, valid_points)

    assert not valid_points[2]
    assert not valid_points[3]
    assert seen_points[2]
    assert not seen_points[3]

if __name__ == "__main__":
    pytest.main([__file__])