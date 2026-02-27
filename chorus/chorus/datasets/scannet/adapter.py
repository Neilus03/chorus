from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from chorus.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from chorus.datasets.base import SceneAdapter
from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids
from chorus.datasets.scannet.metadata import (
    DEFAULT_DEPTH_SCALE_TO_M,
    DEFAULT_VISIBILITY_MIN_DEPTH_M,
    DEFAULT_VISIBILITY_Z_TOLERANCE_M,
    GEOMETRY_SUFFIX,
)
from chorus.datasets.scannet.prepare import extract_rgbd


class ScanNetSceneAdapter(SceneAdapter):
    @property
    def dataset_name(self) -> str:
        return "scannet"

    def prepare(self) -> None:
        extract_rgbd(self.scene_root)

    def list_frames(self) -> list[FrameRecord]:
        color_dir = self.scene_root / "color"
        depth_dir = self.scene_root / "depth"
        pose_dir = self.scene_root / "pose"
        intrinsics_path = self.scene_root / "intrinsic" / "intrinsic_color.txt"

        frame_ids = sorted(
            [p.stem for p in color_dir.iterdir() if p.suffix == ".jpg"],
            key=lambda x: int(x),
        )

        return [
            FrameRecord(
                frame_id=frame_id,
                rgb_path=color_dir / f"{frame_id}.jpg",
                depth_path=depth_dir / f"{frame_id}.png",
                pose_path=pose_dir / f"{frame_id}.txt",
                intrinsics_path=intrinsics_path,
            )
            for frame_id in frame_ids
        ]

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        return np.array(Image.open(frame.rgb_path).convert("RGB"))

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        depth = np.array(Image.open(frame.depth_path), dtype=np.float32)
        return depth * DEFAULT_DEPTH_SCALE_TO_M

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.pose_path)

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.intrinsics_path)[:3, :3]

    def load_geometry_points(self) -> np.ndarray:
        mesh_path = self.scene_root / f"{self.scene_id}{GEOMETRY_SUFFIX}"
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        return np.asarray(pcd.points)

    def load_geometry_colors(self):
        mesh_path = self.scene_root / f"{self.scene_id}{GEOMETRY_SUFFIX}"
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        colors = np.asarray(pcd.colors)
        return colors if colors.size > 0 else None

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self.scene_root / f"{self.scene_id}{GEOMETRY_SUFFIX}",
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=DEFAULT_VISIBILITY_MIN_DEPTH_M,
            z_tolerance_m=DEFAULT_VISIBILITY_Z_TOLERANCE_M,
            depth_scale_to_m=DEFAULT_DEPTH_SCALE_TO_M,
            depth_aligned_to_rgb=True,
        )

    def load_gt_instance_ids(self) -> np.ndarray | None:
        return load_scannet_gt_instance_ids(self.scene_root, self.scene_id)