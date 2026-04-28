from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chorus.datasets.scannetpp.prepare import _parse_pose_json, _save_pose_and_intrinsics


def test_save_pose_and_intrinsics_supports_per_frame_payload(tmp_path: Path) -> None:
    pose_json_path = tmp_path / "pose_intrinsic_imu.json"
    pose_dir = tmp_path / "pose"
    intrinsic_dir = tmp_path / "intrinsic"

    intrinsic = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    aligned_pose = np.eye(4, dtype=np.float32)
    pose_json_path.write_text(
        json.dumps(
            {
                "frame_000000": {
                    "intrinsic": intrinsic.tolist(),
                    "aligned_pose": aligned_pose.tolist(),
                    "pose": aligned_pose.tolist(),
                    "timestamp": 0.0,
                    "imu": {},
                }
            }
        ),
        encoding="utf-8",
    )

    entries = _parse_pose_json(pose_json_path)
    # Simulate the undistortion step that populates new_k_color and new_k_depth
    entries[0]["new_k_color"] = intrinsic
    entries[0]["new_k_depth"] = np.array(
        [
            [200.0, 0.0, 128.0],
            [0.0, 200.0, 96.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    pose_count = _save_pose_and_intrinsics(entries, pose_dir, intrinsic_dir)

    assert pose_count == 1
    np.testing.assert_allclose(np.loadtxt(pose_dir / "frame_000000.txt"), aligned_pose)
    np.testing.assert_allclose(np.loadtxt(intrinsic_dir / "intrinsic_color.txt"), intrinsic)
    np.testing.assert_allclose(
        np.loadtxt(intrinsic_dir / "intrinsic_depth.txt"),
        entries[0]["new_k_depth"],
    )
    np.testing.assert_allclose(np.loadtxt(intrinsic_dir / "frame_000000.txt"), intrinsic)
