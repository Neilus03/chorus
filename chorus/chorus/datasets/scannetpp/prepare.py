from __future__ import annotations

import json
import re
import shutil
import subprocess
import zlib
from pathlib import Path

import cv2
import numpy as np

from chorus.datasets.scannetpp.download import (
    expected_scene_asset_paths,
    get_scene_release,
)
from chorus.datasets.scannetpp.gt import (
    load_scannetpp_gt_instance_ids,
    scannetpp_gt_cache_path,
)

try:
    import lz4.block  # type: ignore
except ImportError:  # pragma: no cover - exercised only on lz4-backed depth files
    lz4 = None
else:  # pragma: no cover - exercised only on lz4-backed depth files
    lz4 = lz4.block


def _frame_stem(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}"


def _count_matching_files(directory: Path, suffixes: tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    return len([path for path in directory.iterdir() if path.suffix.lower() in suffixes])


def is_prepared(scene_dir: Path) -> bool:
    scene_dir = Path(scene_dir)
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    if not (scene_dir / ".prepared").exists():
        return False
    if not color_dir.is_dir() or not depth_dir.is_dir() or not pose_dir.is_dir() or not intrinsic_dir.is_dir():
        return False

    required_intrinsics = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
    ]
    for path in required_intrinsics:
        if not path.exists() or path.stat().st_size == 0:
            return False

    return (
        _count_matching_files(color_dir, (".jpg", ".png")) > 0
        and _count_matching_files(depth_dir, (".png",)) > 0
        and _count_matching_files(pose_dir, (".txt",)) > 0
    )


def has_raw_scene_assets(
    scene_dir: Path,
    require_annotations: bool = True,
) -> bool:
    asset_paths = expected_scene_asset_paths(
        scene_root=scene_dir,
        require_annotations=require_annotations,
        include_semantic_mesh=False,
    )
    return all(path.exists() for path in asset_paths.values())


def _validate_raw_scene_assets(scene_dir: Path) -> None:
    missing = [
        path
        for path in expected_scene_asset_paths(
            scene_root=scene_dir,
            require_annotations=False,
            include_semantic_mesh=False,
        ).values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required ScanNet++ raw assets:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def _extract_rgb_ffmpeg(video_path: Path, output_dir: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-start_number",
        "0",
        "-q:v",
        "1",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return True


def _extract_rgb_opencv(video_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open ScanNet++ RGB video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            out_path = output_dir / f"{_frame_stem(frame_idx)}.jpg"
            if not cv2.imwrite(str(out_path), frame_bgr):
                raise RuntimeError(f"Failed to write RGB frame: {out_path}")
            frame_idx += 1
    finally:
        capture.release()


def _extract_rgb_frames(video_path: Path, output_dir: Path) -> int:
    existing = _count_matching_files(output_dir, (".jpg", ".png"))
    if existing > 0:
        return existing

    if not _extract_rgb_ffmpeg(video_path, output_dir):
        _extract_rgb_opencv(video_path, output_dir)

    return _count_matching_files(output_dir, (".jpg", ".png"))


def _decode_depth_payload(data: bytes, height: int, width: int) -> np.ndarray:
    try:
        decoded = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
        depth = np.frombuffer(decoded, dtype=np.float32).reshape(height, width)
        return (depth * 1000.0).astype(np.uint16)
    except Exception:
        if lz4 is None:
            raise RuntimeError(
                "Encountered an lz4-compressed ScanNet++ depth stream, but python-lz4 is not installed."
            )
        decoded = lz4.decompress(data, uncompressed_size=height * width * 2)
        return np.frombuffer(decoded, dtype=np.uint16).reshape(height, width)


def _extract_depth_frames(depth_path: Path, output_dir: Path) -> int:
    existing = _count_matching_files(output_dir, (".png",))
    if existing > 0:
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = 192, 256

    try:
        with depth_path.open("rb") as infile:
            raw = infile.read()
        decoded = zlib.decompress(raw, wbits=-zlib.MAX_WBITS)
        depth_stack = np.frombuffer(decoded, dtype=np.float32).reshape(-1, height, width)
        for frame_idx, depth_m in enumerate(depth_stack):
            depth_mm = (depth_m * 1000.0).astype(np.uint16)
            out_path = output_dir / f"{_frame_stem(frame_idx)}.png"
            if not cv2.imwrite(str(out_path), depth_mm):
                raise RuntimeError(f"Failed to write depth frame: {out_path}")
    except Exception:
        with depth_path.open("rb") as infile:
            frame_idx = 0
            while True:
                size_bytes = infile.read(4)
                if len(size_bytes) == 0:
                    break
                size = int.from_bytes(size_bytes, byteorder="little")
                payload = infile.read(size)
                depth_mm = _decode_depth_payload(payload, height, width)
                out_path = output_dir / f"{_frame_stem(frame_idx)}.png"
                if not cv2.imwrite(str(out_path), depth_mm):
                    raise RuntimeError(f"Failed to write depth frame: {out_path}")
                frame_idx += 1

    return _count_matching_files(output_dir, (".png",))


def _matrix_from_json(value: object, shape: tuple[int, int], field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == shape:
        return array
    if array.size == shape[0] * shape[1]:
        return array.reshape(shape)
    raise RuntimeError(f"Could not reshape ScanNet++ field '{field_name}' to {shape}.")


def _pose_payload_to_entries(payload: object) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Could not parse ScanNet++ pose/intrinsic payload.")

    aligned_poses = payload.get("aligned_poses")
    if isinstance(aligned_poses, list):
        intrinsic = _matrix_from_json(payload.get("intrinsic"), (3, 3), "intrinsic")
        pose_entries = [
            (
                frame_idx,
                _matrix_from_json(pose_value, (4, 4), f"aligned_poses[{frame_idx}]"),
            )
            for frame_idx, pose_value in enumerate(aligned_poses)
        ]
        return intrinsic, pose_entries

    frame_entries: list[tuple[int, str, dict[str, object]]] = []
    for frame_key, frame_payload in payload.items():
        if not isinstance(frame_payload, dict) or not str(frame_key).startswith("frame_"):
            continue
        match = re.search(r"(\d+)$", str(frame_key))
        if match is None:
            continue
        frame_entries.append((int(match.group(1)), str(frame_key), frame_payload))

    if not frame_entries:
        raise RuntimeError(
            "ScanNet++ pose_intrinsic_imu.json is missing 'aligned_poses' and per-frame entries."
        )

    frame_entries.sort(key=lambda item: item[0])
    first_frame_key, first_frame_payload = frame_entries[0][1], frame_entries[0][2]
    intrinsic = _matrix_from_json(
        first_frame_payload.get("intrinsic"),
        (3, 3),
        f"{first_frame_key}.intrinsic",
    )

    pose_entries = []
    for frame_idx, frame_key, frame_payload in frame_entries:
        pose_value = frame_payload.get("aligned_pose", frame_payload.get("pose"))
        pose_entries.append(
            (
                frame_idx,
                _matrix_from_json(
                    pose_value,
                    (4, 4),
                    f"{frame_key}.aligned_pose",
                ),
            )
        )
    return intrinsic, pose_entries


def _save_pose_and_intrinsics(
    pose_intrinsic_imu_path: Path,
    pose_dir: Path,
    intrinsic_dir: Path,
    color_dir: Path,
    depth_dir: Path,
) -> int:
    with pose_intrinsic_imu_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    intrinsic, pose_entries = _pose_payload_to_entries(payload)

    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    color_files = sorted(color_dir.glob("*.jpg"))
    if not color_files:
        color_files = sorted(color_dir.glob("*.png"))
    if not color_files:
        raise RuntimeError("No decoded RGB frames were found while writing ScanNet++ intrinsics.")

    rgb_sample = cv2.imread(str(color_files[0]), cv2.IMREAD_COLOR)
    if rgb_sample is None:
        raise RuntimeError(f"Failed to read RGB sample frame: {color_files[0]}")
    rgb_h, rgb_w = rgb_sample.shape[:2]

    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise RuntimeError("No decoded depth frames were found while writing ScanNet++ intrinsics.")

    depth_sample = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    if depth_sample is None:
        raise RuntimeError(f"Failed to read depth sample frame: {depth_files[0]}")
    depth_h, depth_w = depth_sample.shape[:2]

    intrinsic_color = intrinsic.astype(np.float32)
    intrinsic_depth = intrinsic_color.copy()
    intrinsic_depth[0, 0] *= depth_w / max(rgb_w, 1)
    intrinsic_depth[0, 2] *= depth_w / max(rgb_w, 1)
    intrinsic_depth[1, 1] *= depth_h / max(rgb_h, 1)
    intrinsic_depth[1, 2] *= depth_h / max(rgb_h, 1)

    np.savetxt(intrinsic_dir / "intrinsic_color.txt", intrinsic_color)
    np.savetxt(intrinsic_dir / "intrinsic_depth.txt", intrinsic_depth)

    pose_count = 0
    for frame_idx, pose in pose_entries:
        if not np.isfinite(pose).all():
            continue
        np.savetxt(pose_dir / f"{_frame_stem(frame_idx)}.txt", pose)
        pose_count += 1

    return pose_count


def prepare_scannetpp_scene(
    scene_root: Path,
    cache_gt: bool = True,
) -> None:
    scene_root = Path(scene_root)
    if is_prepared(scene_root):
        return

    _validate_raw_scene_assets(scene_root)
    scene = get_scene_release(scene_root)

    color_dir = scene_root / "color"
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"
    intrinsic_dir = scene_root / "intrinsic"

    rgb_count = _extract_rgb_frames(Path(scene.iphone_video_path), color_dir)
    depth_count = _extract_depth_frames(Path(scene.iphone_depth_path), depth_dir)
    pose_count = _save_pose_and_intrinsics(
        pose_intrinsic_imu_path=Path(scene.iphone_pose_intrinsic_imu_path),
        pose_dir=pose_dir,
        intrinsic_dir=intrinsic_dir,
        color_dir=color_dir,
        depth_dir=depth_dir,
    )

    if rgb_count == 0 or depth_count == 0 or pose_count == 0:
        raise RuntimeError(
            f"ScanNet++ prepare produced empty outputs: rgb={rgb_count}, depth={depth_count}, pose={pose_count}"
        )

    if cache_gt and has_raw_scene_assets(scene_root, require_annotations=True):
        cache_path = scannetpp_gt_cache_path(scene_root)
        if not cache_path.exists():
            gt_instance_ids = load_scannetpp_gt_instance_ids(scene_root)
            np.save(cache_path, gt_instance_ids)

    (scene_root / ".prepared").touch()
    print(
        f"Prepared ScanNet++ scene {scene_root.name}: "
        f"rgb_frames={rgb_count}, depth_frames={depth_count}, poses={pose_count}"
    )
