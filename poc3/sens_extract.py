import sys
from pathlib import Path

import numpy as np

from config import LITEPT_SCANNET_PAIR


def _load_sensor_data_class():
    sensor_py = LITEPT_SCANNET_PAIR / "SensorData.py"
    if not sensor_py.exists():
        raise FileNotFoundError(f"Missing SensorData.py at {sensor_py}")
    if str(LITEPT_SCANNET_PAIR) not in sys.path:
        sys.path.insert(0, str(LITEPT_SCANNET_PAIR))
    from SensorData import SensorData  # type: ignore

    return SensorData


def _save_mat(matrix: np.ndarray, filename: Path) -> None:
    with filename.open("w", encoding="utf-8") as f:
        for line in matrix:
            np.savetxt(f, line[np.newaxis], fmt="%f")


def extract_rgbd(scene_dir: Path) -> None:
    try:
        import imageio
    except ImportError as exc:
        raise RuntimeError(
            "sens_extract requires 'imageio'. Install it in your active environment."
        ) from exc

    scene_id = scene_dir.name
    sens_path = scene_dir / f"{scene_id}.sens"
    if not sens_path.exists():
        raise FileNotFoundError(f"Missing .sens: {sens_path}")

    SensorData = _load_sensor_data_class()
    sd = SensorData(str(sens_path))

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    for d in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _save_mat(sd.intrinsic_color, intrinsic_dir / "intrinsic_color.txt")
    _save_mat(sd.extrinsic_color, intrinsic_dir / "extrinsic_color.txt")
    _save_mat(sd.intrinsic_depth, intrinsic_dir / "intrinsic_depth.txt")
    _save_mat(sd.extrinsic_depth, intrinsic_dir / "extrinsic_depth.txt")

    for i, frame in enumerate(sd.frames):
        color_path = color_dir / f"{i}.jpg"
        depth_path = depth_dir / f"{i}.png"
        pose_path = pose_dir / f"{i}.txt"

        if not depth_path.exists():
            depth_data = frame.decompress_depth(sd.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                sd.depth_height, sd.depth_width
            )
            imageio.imwrite(depth_path, depth)

        if not color_path.exists():
            color = frame.decompress_color(sd.color_compression_type)
            imageio.imwrite(color_path, color, quality=95)

        if not pose_path.exists():
            _save_mat(frame.camera_to_world, pose_path)

