from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from chorus.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from chorus.datasets.base import SceneAdapter
from chorus.datasets.scannetpp.benchmark import (
    DEFAULT_SCANNETPP_EVAL_BENCHMARK,
    normalize_scannetpp_eval_benchmark,
)
from chorus.datasets.scannetpp.evaluation import ScanNetPPEvaluationHooks
from chorus.datasets.scannetpp.gt import (
    load_scannetpp_gt_instance_ids,
    scannetpp_gt_cache_path,
)
from chorus.datasets.scannetpp.prepare import is_prepared, prepare_scannetpp_scene

DEFAULT_VISIBILITY_MIN_DEPTH_M = 0.05
DEFAULT_VISIBILITY_Z_TOLERANCE_M = 0.15


def _frame_sort_key(frame_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(frame_id))
    if match is not None:
        return int(match.group(1)), str(frame_id)
    return 0, str(frame_id)


class ScanNetPPSceneAdapter(SceneAdapter):
    def __init__(
        self,
        scene_root: Path,
        eval_benchmark: str | None = None,
    ):
        super().__init__(scene_root=scene_root)
        self.eval_benchmark = normalize_scannetpp_eval_benchmark(
            eval_benchmark or DEFAULT_SCANNETPP_EVAL_BENCHMARK
        )

    @property
    def dataset_name(self) -> str:
        return "scannetpp"

    def prepare(self) -> None:
        if is_prepared(self.scene_root):
            print(f"Scene {self.scene_id} is already prepared.")
            return
        prepare_scannetpp_scene(self.scene_root)

    def list_frames(self) -> list[FrameRecord]:
        color_dir = self.scene_root / "color"
        depth_dir = self.scene_root / "depth"
        pose_dir = self.scene_root / "pose"
        intrinsics_path = self.scene_root / "intrinsic" / "intrinsic_color.txt"

        if not color_dir.is_dir():
            raise FileNotFoundError(
                f"Missing color directory at {color_dir}. Call adapter.prepare() first."
            )
        if not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Missing depth directory at {depth_dir}. Call adapter.prepare() first."
            )
        if not pose_dir.is_dir():
            raise FileNotFoundError(
                f"Missing pose directory at {pose_dir}. Call adapter.prepare() first."
            )
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"Missing intrinsics file at {intrinsics_path}. Call adapter.prepare() first."
            )

        color_files = {
            path.stem: path
            for path in color_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".png"}
        }
        frame_ids = sorted(color_files.keys(), key=_frame_sort_key)

        frames = [
            FrameRecord(
                frame_id=frame_id,
                rgb_path=color_files[frame_id],
                depth_path=depth_dir / f"{frame_id}.png",
                pose_path=pose_dir / f"{frame_id}.txt",
                intrinsics_path=intrinsics_path,
            )
            for frame_id in frame_ids
        ]

        return [
            frame
            for frame in frames
            if frame.rgb_path.exists() and frame.depth_path.exists() and frame.pose_path.exists()
        ]

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        return np.array(Image.open(frame.rgb_path).convert("RGB"))

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        depth = np.array(Image.open(frame.depth_path), dtype=np.float32)
        return depth / 1000.0

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        pose = np.loadtxt(frame.pose_path)
        if np.isnan(pose).any() or np.isinf(pose).any():
            raise ValueError(f"Invalid pose matrix in {frame.pose_path}")
        return pose

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.intrinsics_path)[:3, :3]

    def _geometry_path(self) -> Path:
        return self.scene_root / "scans" / "mesh_aligned_0.05.ply"

    def _load_geometry_vertex_data(self):
        mesh_path = self._geometry_path()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing ScanNet++ geometry file: {mesh_path}")

        plydata = PlyData.read(str(mesh_path))
        if "vertex" not in plydata:
            raise RuntimeError(f"PLY file has no vertex element: {mesh_path}")

        vertex_data = plydata["vertex"].data
        if len(vertex_data) == 0:
            raise RuntimeError(f"Loaded empty geometry from {mesh_path}")
        return vertex_data

    def load_geometry_points(self) -> np.ndarray:
        vertex_data = self._load_geometry_vertex_data()
        return np.stack(
            [
                np.asarray(vertex_data["x"], dtype=np.float32),
                np.asarray(vertex_data["y"], dtype=np.float32),
                np.asarray(vertex_data["z"], dtype=np.float32),
            ],
            axis=1,
        )

    def load_geometry_colors(self) -> np.ndarray | None:
        vertex_data = self._load_geometry_vertex_data()
        names = set(vertex_data.dtype.names or [])
        if not {"red", "green", "blue"}.issubset(names):
            return None

        colors = np.stack(
            [
                np.asarray(vertex_data["red"], dtype=np.float32),
                np.asarray(vertex_data["green"], dtype=np.float32),
                np.asarray(vertex_data["blue"], dtype=np.float32),
            ],
            axis=1,
        )
        if colors.max() > 1.0:
            colors = colors / 255.0
        return colors

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self._geometry_path(),
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=DEFAULT_VISIBILITY_MIN_DEPTH_M,
            z_tolerance_m=DEFAULT_VISIBILITY_Z_TOLERANCE_M,
            depth_scale_to_m=1.0,
            depth_aligned_to_rgb=True,
        )

    def load_gt_instance_ids(self) -> np.ndarray | None:
        cache_path = scannetpp_gt_cache_path(self.scene_root, self.eval_benchmark)
        if cache_path.exists():
            return np.load(cache_path)

        gt_instance_ids = load_scannetpp_gt_instance_ids(
            scene_dir=self.scene_root,
            scene_name=self.scene_id,
            eval_benchmark=self.eval_benchmark,
        )
        np.save(cache_path, gt_instance_ids)
        return gt_instance_ids

    def get_evaluation_hooks(self) -> ScanNetPPEvaluationHooks:
        return ScanNetPPEvaluationHooks(self.eval_benchmark)
