from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.common.types import VisibilityConfig
from chorus.core.lifting.project import project_points_to_image
from chorus.core.lifting.visibility import compute_visible_points


def test_project_points_to_image_filters_points_behind_camera() -> None:
    points_3d = np.array(
        [
            [0.0, 0.0, 1.0],   # valid
            [2.0, 2.0, 2.0],   # valid
            [0.0, 0.0, -1.0],  # behind camera
        ],
        dtype=np.float32,
    )
    pose_c2w = np.eye(4, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    u, v, z, valid_indices = project_points_to_image(
        points_3d=points_3d,
        pose_c2w=pose_c2w,
        intrinsics=intrinsics,
    )

    np.testing.assert_array_equal(valid_indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(u, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(v, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(z, np.array([1.0, 2.0], dtype=np.float32), atol=1e-6)


def test_compute_visible_points_respects_image_bounds_and_depth_consistency() -> None:
    points_3d = np.array(
        [
            [0.0, 0.0, 1.0],  # pixel (0, 0), should be visible
            [1.0, 0.0, 1.0],  # pixel (1, 0), should be visible
            [2.0, 0.0, 1.0],  # pixel (2, 0), out of bounds for width=2
            [0.0, 0.0, 2.0],  # same pixel as first point, depth mismatch
        ],
        dtype=np.float32,
    )
    pose_c2w = np.eye(4, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    u, v, z, valid_indices = project_points_to_image(
        points_3d=points_3d,
        pose_c2w=pose_c2w,
        intrinsics=intrinsics,
    )

    depth_map_m = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    visibility_cfg = VisibilityConfig(
        min_depth_m=0.1,
        z_tolerance_m=0.1,
        depth_scale_to_m=1.0,
        depth_aligned_to_rgb=True,
    )

    visible_indices, visible_u, visible_v = compute_visible_points(
        u=u,
        v=v,
        z=z,
        valid_indices=valid_indices,
        depth_map_m=depth_map_m,
        visibility_cfg=visibility_cfg,
    )

    np.testing.assert_array_equal(visible_indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(visible_u, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(visible_v, np.array([0, 0], dtype=np.int32))

if __name__ == "__main__":
    pytest.main([__file__])