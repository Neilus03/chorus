from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import open3d as o3d

from chorus.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from chorus.datasets.base import SceneAdapter
from chorus.datasets.structured3d.prepare import is_prepared, prepare_structured3d_scene


class Structured3DSceneAdapter(SceneAdapter):
    def __init__(self, scene_root: Path, raw_zips_dir: str = "/scratch2/nedela/structured3d_raw"):
        super().__init__(scene_root=scene_root)
        self.raw_zips_dir = raw_zips_dir

    @property
    def dataset_name(self) -> str:
        return "structured3d"

    def prepare(self) -> None:
        if is_prepared(self.scene_root):
            return
        prepare_structured3d_scene(self.scene_id, self.raw_zips_dir, str(self.scene_root.parent))

    def list_frames(self) -> list[FrameRecord]:
        color_dir = self.scene_root / "color"
        depth_dir = self.scene_root / "depth"
        pose_dir = self.scene_root / "pose"
        intrinsic_dir = self.scene_root / "intrinsic"
        default_intrinsics_path = intrinsic_dir / "intrinsic_color.txt"

        if not color_dir.is_dir():
            raise FileNotFoundError(f"Missing color directory at {color_dir}. Call adapter.prepare() first.")
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth directory at {depth_dir}. Call adapter.prepare() first.")
        if not pose_dir.is_dir():
            raise FileNotFoundError(f"Missing pose directory at {pose_dir}. Call adapter.prepare() first.")
        if not default_intrinsics_path.exists():
            raise FileNotFoundError(
                f"Missing intrinsics file at {default_intrinsics_path}. Call adapter.prepare() first."
            )

        color_files = {p.stem: p for p in color_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}}
        frame_ids = sorted(color_files.keys(), key=lambda x: int(x))

        frames = [
            FrameRecord(
                frame_id=frame_id,
                rgb_path=color_files[frame_id],
                depth_path=depth_dir / f"{frame_id}.png",
                pose_path=pose_dir / f"{frame_id}.txt",
                intrinsics_path=(
                    intrinsic_dir / f"{frame_id}.txt"
                    if (intrinsic_dir / f"{frame_id}.txt").exists()
                    else default_intrinsics_path
                ),
            )
            for frame_id in frame_ids
        ]

        return [f for f in frames if f.rgb_path.exists() and f.depth_path.exists() and f.pose_path.exists()]

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {frame.rgb_path}")
        # Important: channel flip produces negative strides; make contiguous for torch/torchvision.
        rgb = bgr[..., ::-1].copy()
        return rgb

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth image: {frame.depth_path}")
        depth_m = depth.astype(np.float32) / 1000.0
        return depth_m

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        pose = np.loadtxt(frame.pose_path)
        if np.isnan(pose).any() or np.isinf(pose).any():
            raise ValueError(f"Invalid pose matrix in {frame.pose_path}")
        return pose

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.intrinsics_path)[:3, :3]

    def _load_point_cloud(self) -> o3d.geometry.PointCloud:
        mesh_path = self.scene_root / f"{self.scene_id}_vh_clean_2.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing Structured3D geometry file: {mesh_path}")

        pcd = o3d.io.read_point_cloud(str(mesh_path))
        if pcd.is_empty():
            raise RuntimeError(f"Loaded empty geometry from {mesh_path}")

        return pcd

    def load_geometry_points(self) -> np.ndarray:
        pcd = self._load_point_cloud()
        return np.asarray(pcd.points, dtype=np.float32)

    def load_geometry_colors(self) -> np.ndarray | None:
        pcd = self._load_point_cloud()
        # Open3D stores colors as float64 in [0,1] if present.
        if not pcd.has_colors():
            return None
        return np.asarray(pcd.colors, dtype=np.float32)

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self.scene_root / f"{self.scene_id}_vh_clean_2.ply",
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            # Structured3D poses/intrinsics can be slightly inconsistent with the fused geometry.
            # A looser tolerance makes the visibility test less brittle.
            min_depth_m=0.01,
            z_tolerance_m=0.8,
            depth_scale_to_m=1.0,  # already converted in load_depth_m
            depth_aligned_to_rgb=True,
        )

    def load_gt_instance_ids(self) -> np.ndarray | None:
        path = self.scene_root / "gt_instance_ids.npy"
        if not path.exists():
            return None
        return np.load(path)

