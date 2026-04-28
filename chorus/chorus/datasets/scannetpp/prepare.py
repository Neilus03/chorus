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
except ImportError:  # pragma: no cover
    lz4 = None
else:  # pragma: no cover
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

    return (
        _count_matching_files(color_dir, (".jpg", ".png")) > 0
        and _count_matching_files(depth_dir, (".png",)) > 0
        and _count_matching_files(pose_dir, (".txt",)) > 0
        and _count_matching_files(intrinsic_dir, (".txt",)) > 0
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


def _matrix_from_json(value: object, shape: tuple[int, int], field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == shape:
        return array
    if array.size == shape[0] * shape[1]:
        return array.reshape(shape)
    raise RuntimeError(f"Could not reshape ScanNet++ field '{field_name}' to {shape}.")


def _parse_pose_json(pose_json_path: Path) -> list[dict]:
    with pose_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frame_entries = []
    for frame_key, frame_payload in payload.items():
        if not isinstance(frame_payload, dict) or not str(frame_key).startswith("frame_"):
            continue
        match = re.search(r"(\d+)$", str(frame_key))
        if match is None:
            continue
        rgb_idx = int(match.group(1))
        frame_entries.append((rgb_idx, str(frame_key), frame_payload))

    if not frame_entries:
        aligned_poses = payload.get("aligned_poses")
        if isinstance(aligned_poses, list):
            intrinsic = _matrix_from_json(payload.get("intrinsic"), (3, 3), "intrinsic")
            distortion = np.zeros(5, dtype=np.float32)
            for rgb_idx, pose_value in enumerate(aligned_poses):
                pose = _matrix_from_json(pose_value, (4, 4), f"aligned_poses[{rgb_idx}]")
                frame_entries.append((rgb_idx, f"frame_{rgb_idx}", {
                    "aligned_pose": pose.tolist(),
                    "intrinsic": intrinsic.tolist(),
                    "distortion": distortion.tolist(),
                }))

    frame_entries.sort(key=lambda item: item[0])
    
    entries = []
    for depth_idx, (rgb_idx, frame_key, frame_payload) in enumerate(frame_entries):
        pose_value = frame_payload.get("aligned_pose", frame_payload.get("pose"))
        if pose_value is None:
            continue
        pose = _matrix_from_json(pose_value, (4, 4), f"{frame_key}.aligned_pose")
        if not np.isfinite(pose).all():
            continue
            
        intrinsic = _matrix_from_json(frame_payload.get("intrinsic"), (3, 3), f"{frame_key}.intrinsic")
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            continue
        
        dist_val = frame_payload.get("distortion")
        if dist_val is not None:
            distortion = np.asarray(dist_val, dtype=np.float32)
            if distortion.size < 5:
                distortion = np.pad(distortion, (0, 5 - distortion.size))
            elif distortion.size > 5:
                distortion = distortion[:5]
        else:
            distortion = np.zeros(5, dtype=np.float32)
            
        if not np.isfinite(distortion).all():
            continue
            
        entries.append({
            "rgb_idx": rgb_idx,
            "depth_idx": depth_idx,
            "pose": pose,
            "intrinsic": intrinsic,
            "distortion": distortion,
            "payload": frame_payload
        })
    return entries


def _extract_and_undistort_rgb(video_path: Path, output_dir: Path, entries: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open ScanNet++ RGB video: {video_path}")
    
    rgb_to_entry = {e["rgb_idx"]: e for e in entries}
    max_rgb_idx = max(rgb_to_entry.keys()) if rgb_to_entry else -1
    
    frame_idx = 0
    try:
        while frame_idx <= max_rgb_idx:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            
            if frame_idx in rgb_to_entry:
                entry = rgb_to_entry[frame_idx]
                h, w = frame_bgr.shape[:2]
                
                raw_k = entry["intrinsic"]
                dist_coeffs = entry["distortion"]
                
                new_k, roi = cv2.getOptimalNewCameraMatrix(raw_k, dist_coeffs, (w, h), 0, (w, h))
                
                undistorted = cv2.undistort(frame_bgr, raw_k, dist_coeffs, None, new_k)
                if undistorted is None or undistorted.size == 0:
                    print(f"Warning: cv2.undistort produced empty image for frame {frame_idx}. Skipping.")
                    continue
                
                out_path = output_dir / f"{_frame_stem(frame_idx)}.jpg"
                if not cv2.imwrite(str(out_path), undistorted):
                    print(f"Warning: Failed to write RGB frame (disk full or invalid array?): {out_path}. Skipping.")
                    continue
                
                entry["new_k_color"] = new_k
                entry["rgb_h"] = h
                entry["rgb_w"] = w
                
            frame_idx += 1
    finally:
        capture.release()


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


def _extract_and_undistort_depth(depth_path: Path, output_dir: Path, entries: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_to_entry = {e["depth_idx"]: e for e in entries}
    
    height, width = 192, 256
    
    def process_depth(depth_idx: int, depth_mm: np.ndarray):
        if depth_idx in depth_to_entry:
            entry = depth_to_entry[depth_idx]
            rgb_idx = entry["rgb_idx"]
            
            depth_intrinsic_val = entry["payload"].get("depth_intrinsic")
            depth_distortion_val = entry["payload"].get("depth_distortion")
            
            if depth_intrinsic_val is not None:
                raw_k = _matrix_from_json(depth_intrinsic_val, (3, 3), "depth_intrinsic")
            else:
                raw_k = entry["intrinsic"].copy()
                rgb_w = entry.get("rgb_w", width * 4) 
                rgb_h = entry.get("rgb_h", height * 4)
                raw_k[0, 0] *= width / max(rgb_w, 1)
                raw_k[0, 2] *= width / max(rgb_w, 1)
                raw_k[1, 1] *= height / max(rgb_h, 1)
                raw_k[1, 2] *= height / max(rgb_h, 1)
                
            if depth_distortion_val is not None:
                dist_coeffs = np.asarray(depth_distortion_val, dtype=np.float32)
                if dist_coeffs.size < 5:
                    dist_coeffs = np.pad(dist_coeffs, (0, 5 - dist_coeffs.size))
                elif dist_coeffs.size > 5:
                    dist_coeffs = dist_coeffs[:5]
            else:
                dist_coeffs = entry["distortion"]
            
            new_k, _ = cv2.getOptimalNewCameraMatrix(raw_k, dist_coeffs, (width, height), 0, (width, height))
            
            map1, map2 = cv2.initUndistortRectifyMap(raw_k, dist_coeffs, None, new_k, (width, height), cv2.CV_32FC1)
            undistorted = cv2.remap(depth_mm, map1, map2, interpolation=cv2.INTER_NEAREST)
            
            out_path = output_dir / f"{_frame_stem(rgb_idx)}.png"
            if not cv2.imwrite(str(out_path), undistorted):
                print(f"Warning: Failed to write depth frame: {out_path}. Skipping.")
                return
            
            entry["new_k_depth"] = new_k

    try:
        with depth_path.open("rb") as infile:
            raw = infile.read()
        decoded = zlib.decompress(raw, wbits=-zlib.MAX_WBITS)
        depth_stack = np.frombuffer(decoded, dtype=np.float32).reshape(-1, height, width)
        for d_idx, depth_m in enumerate(depth_stack):
            depth_mm = (depth_m * 1000.0).astype(np.uint16)
            process_depth(d_idx, depth_mm)
    except Exception:
        with depth_path.open("rb") as infile:
            d_idx = 0
            while True:
                size_bytes = infile.read(4)
                if len(size_bytes) == 0:
                    break
                size = int.from_bytes(size_bytes, byteorder="little")
                payload = infile.read(size)
                depth_mm = _decode_depth_payload(payload, height, width)
                process_depth(d_idx, depth_mm)
                d_idx += 1


def _save_pose_and_intrinsics(entries: list[dict], pose_dir: Path, intrinsic_dir: Path) -> int:
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    
    pose_count = 0
    for entry in entries:
        rgb_idx = entry["rgb_idx"]
        pose = entry["pose"]
        
        np.savetxt(pose_dir / f"{_frame_stem(rgb_idx)}.txt", pose)
        
        if "new_k_color" in entry:
            np.savetxt(intrinsic_dir / f"{_frame_stem(rgb_idx)}.txt", entry["new_k_color"])
            
        pose_count += 1
        
    if entries and "new_k_color" in entries[0]:
        np.savetxt(intrinsic_dir / "intrinsic_color.txt", entries[0]["new_k_color"])
    if entries and "new_k_depth" in entries[0]:
        np.savetxt(intrinsic_dir / "intrinsic_depth.txt", entries[0]["new_k_depth"])
        
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

    entries = _parse_pose_json(Path(scene.iphone_pose_intrinsic_imu_path))

    _extract_and_undistort_rgb(Path(scene.iphone_video_path), color_dir, entries)
    _extract_and_undistort_depth(Path(scene.iphone_depth_path), depth_dir, entries)
    
    pose_count = _save_pose_and_intrinsics(entries, pose_dir, intrinsic_dir)

    rgb_count = _count_matching_files(color_dir, (".jpg", ".png"))
    depth_count = _count_matching_files(depth_dir, (".png",))

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
