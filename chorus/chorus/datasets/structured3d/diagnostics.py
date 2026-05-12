from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RAW_ZIPS_DIR = Path("/cluster/work/igp_psr/nedela/structured3d_raw")
DEFAULT_SCANS_ROOT = Path("/cluster/work/igp_psr/nedela/chorus_poc/structured3d_diag_10")
DEFAULT_REPORT_PATH = DEFAULT_SCANS_ROOT / "_structured3d_diagnostic_report.json"
BBOX_ZIP_NAME = "Structured3D_bbox.zip"
AXIS_CONVERSION = ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))


@dataclass(frozen=True)
class SceneZipAssets:
    scene_id: str
    has_perspective: bool
    has_bbox_3d: bool
    has_instance_png: bool
    perspective_zip: str | None
    num_rgb: int
    num_depth: int
    num_camera: int
    num_instance: int

    @property
    def ok_for_prepare_with_gt(self) -> bool:
        return (
            self.has_perspective
            and self.has_bbox_3d
            and self.has_instance_png
            and self.num_rgb > 0
            and self.num_depth > 0
            and self.num_camera > 0
        )


def scene_ids_from_range(start: int, count: int) -> list[str]:
    if count < 1:
        return []
    return [f"scene_{idx:05d}" for idx in range(start, start + count)]


def _safe_rel(path: Path) -> str:
    return str(path)


def _count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def _read_ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return None
            if line == "end_header":
                return None
    return None


def _list_scene_zip_assets(raw_zips_dir: Path, scene_ids: list[str]) -> dict[str, SceneZipAssets]:
    selected = set(scene_ids)
    perspective_zips = sorted(raw_zips_dir.glob("*perspective_full*.zip"))
    bbox_zip = raw_zips_dir / BBOX_ZIP_NAME
    scene_re = re.compile(r"^Structured3D/(scene_\d{5})/")

    found: dict[str, dict[str, Any]] = {
        scene_id: {
            "scene_id": scene_id,
            "has_perspective": False,
            "has_bbox_3d": False,
            "has_instance_png": False,
            "perspective_zip": None,
            "num_rgb": 0,
            "num_depth": 0,
            "num_camera": 0,
            "num_instance": 0,
        }
        for scene_id in scene_ids
    }

    pending_perspective = set(scene_ids)
    for zip_path in perspective_zips:
        if not pending_perspective:
            break
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                match = scene_re.match(info.filename)
                if match is None:
                    continue
                scene_id = match.group(1)
                if scene_id not in selected:
                    continue
                item = found[scene_id]
                item["has_perspective"] = True
                item["perspective_zip"] = zip_path.name
                if info.filename.endswith("/rgb_rawlight.png"):
                    item["num_rgb"] += 1
                elif info.filename.endswith("/depth.png"):
                    item["num_depth"] += 1
                elif info.filename.endswith("/camera_pose.txt"):
                    item["num_camera"] += 1
                if item["num_rgb"] > 0 and item["num_depth"] > 0 and item["num_camera"] > 0:
                    pending_perspective.discard(scene_id)
                if not pending_perspective:
                    break

    if bbox_zip.exists():
        pending_bbox = set(scene_ids)
        with zipfile.ZipFile(bbox_zip, "r") as zf:
            for info in zf.infolist():
                match = scene_re.match(info.filename)
                if match is None:
                    continue
                scene_id = match.group(1)
                if scene_id not in selected:
                    continue
                item = found[scene_id]
                if info.filename == f"Structured3D/{scene_id}/bbox_3d.json":
                    item["has_bbox_3d"] = True
                elif "/perspective/full/" in info.filename and info.filename.endswith("/instance.png"):
                    item["has_instance_png"] = True
                    item["num_instance"] += 1
                if item["has_bbox_3d"] and item["has_instance_png"]:
                    pending_bbox.discard(scene_id)
                if not pending_bbox:
                    break

    return {scene_id: SceneZipAssets(**values) for scene_id, values in found.items()}


def inspect_raw_zips(
    raw_zips_dir: Path,
    scene_ids: list[str],
    scan_scene_assets: bool = True,
) -> dict[str, Any]:
    raw_zips_dir = raw_zips_dir.resolve()
    perspective_zips = sorted(raw_zips_dir.glob("*perspective_full*.zip"))
    bbox_zip = raw_zips_dir / BBOX_ZIP_NAME
    bbox_candidates = sorted(raw_zips_dir.glob("*bbox*.zip"))

    raw: dict[str, Any] = {
        "raw_zips_dir": _safe_rel(raw_zips_dir),
        "perspective_zip_count": len(perspective_zips),
        "perspective_zip_names": [p.name for p in perspective_zips],
        "bbox_zip_path": _safe_rel(bbox_zip),
        "bbox_zip_exists": bbox_zip.exists(),
        "bbox_zip_size_bytes": bbox_zip.stat().st_size if bbox_zip.exists() else None,
        "bbox_candidates": [_safe_rel(p) for p in bbox_candidates],
        "scene_assets": {},
    }

    if scan_scene_assets and raw_zips_dir.is_dir() and perspective_zips:
        raw["scene_assets"] = {
            scene_id: assets.__dict__
            for scene_id, assets in _list_scene_zip_assets(raw_zips_dir, scene_ids).items()
        }

    return raw


def inspect_prepared_scene(scans_root: Path, scene_id: str) -> dict[str, Any]:
    scene_dir = scans_root / scene_id
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    gt_path = scene_dir / "gt_instance_ids.npy"
    prepared = {
        "scene_id": scene_id,
        "scene_dir": _safe_rel(scene_dir),
        "exists": scene_dir.is_dir(),
        "prepared_marker": (scene_dir / ".prepared").exists(),
        "num_color": _count_files(scene_dir / "color", (".jpg", ".png")),
        "num_depth": _count_files(scene_dir / "depth", (".png",)),
        "num_pose": _count_files(scene_dir / "pose", (".txt",)),
        "has_intrinsic_color": (scene_dir / "intrinsic" / "intrinsic_color.txt").exists(),
        "has_intrinsic_depth": (scene_dir / "intrinsic" / "intrinsic_depth.txt").exists(),
        "geometry_path": _safe_rel(ply_path),
        "geometry_exists": ply_path.exists(),
        "geometry_vertex_count": _read_ply_vertex_count(ply_path),
        "gt_instance_ids_path": _safe_rel(gt_path),
        "gt_instance_ids_exists": gt_path.exists(),
        "gt_instance_count": None,
        "gt_unique_instance_count": None,
    }

    if gt_path.exists():
        try:
            import numpy as np

            gt = np.load(gt_path)
            prepared["gt_instance_count"] = int(gt.shape[0])
            prepared["gt_unique_instance_count"] = int(len(np.unique(gt[gt > 0])))
        except Exception as exc:
            prepared["gt_load_error"] = f"{type(exc).__name__}: {exc}"

    return prepared


def inspect_prepared_scenes(scans_root: Path, scene_ids: list[str]) -> dict[str, Any]:
    scans_root = scans_root.resolve()
    scenes = [inspect_prepared_scene(scans_root, scene_id) for scene_id in scene_ids]
    return {
        "scans_root": _safe_rel(scans_root),
        "scans_root_exists": scans_root.is_dir(),
        "scenes": scenes,
    }


def _pose_with_axis_conversion(pose_c2w: Any, conversion: str):
    import numpy as np

    axis = np.asarray(AXIS_CONVERSION, dtype=np.float32)
    out = np.asarray(pose_c2w, dtype=np.float32).copy()
    if conversion == "current":
        return out
    if conversion == "camera_axis_right_multiply":
        out[:3, :3] = out[:3, :3] @ axis.T
        return out
    if conversion == "camera_axis_left_multiply":
        out[:3, :3] = axis @ out[:3, :3]
        out[:3, 3] = axis @ out[:3, 3]
        return out
    raise ValueError(f"Unknown conversion: {conversion}")


def compute_projection_coverage(
    scans_root: Path,
    scene_id: str,
    sample_points: int,
    max_frames: int,
    z_tolerance_m: float,
) -> dict[str, Any]:
    import numpy as np

    from chorus.common.types import VisibilityConfig
    from chorus.core.lifting.project import project_points_to_image
    from chorus.core.lifting.visibility import compute_visible_points
    from chorus.datasets.structured3d.adapter import Structured3DSceneAdapter

    scene_dir = scans_root / scene_id
    adapter = Structured3DSceneAdapter(scene_dir)
    points = adapter.load_geometry_points()
    frames = adapter.list_frames()

    if len(points) > sample_points:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(points), size=sample_points, replace=False)
        points_used = points[sample_idx]
    else:
        sample_idx = np.arange(len(points), dtype=np.int64)
        points_used = points

    if len(frames) > max_frames:
        frame_positions = np.linspace(0, len(frames) - 1, num=max_frames)
        frame_indices = sorted({int(round(x)) for x in frame_positions})
        frames_used = [frames[i] for i in frame_indices]
    else:
        frames_used = frames

    cfg = VisibilityConfig(
        min_depth_m=0.01,
        z_tolerance_m=float(z_tolerance_m),
        depth_scale_to_m=1.0,
        depth_aligned_to_rgb=True,
    )

    variants: dict[str, Any] = {}
    for variant in ("current", "camera_axis_right_multiply", "camera_axis_left_multiply"):
        seen = np.zeros(points_used.shape[0], dtype=bool)
        per_frame = []
        for frame in frames_used:
            pose_c2w = _pose_with_axis_conversion(adapter.load_pose_c2w(frame), variant)
            intrinsics = adapter.load_intrinsics(frame)
            depth_map_m = adapter.load_depth_m(frame)
            u, v, z, valid_indices = project_points_to_image(
                points_3d=points_used,
                pose_c2w=pose_c2w,
                intrinsics=intrinsics,
            )

            finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
            u_f = u[finite]
            v_f = v[finite]
            valid_f = valid_indices[finite]
            u_i = u_f.astype(np.int32)
            v_i = v_f.astype(np.int32)
            in_image = (u_i >= 0) & (u_i < depth_map_m.shape[1]) & (v_i >= 0) & (v_i < depth_map_m.shape[0])

            visible_indices, _, _ = compute_visible_points(
                u=u,
                v=v,
                z=z,
                valid_indices=valid_indices,
                depth_map_m=depth_map_m,
                visibility_cfg=cfg,
            )
            seen[visible_indices] = True
            per_frame.append(
                {
                    "frame_id": frame.frame_id,
                    "valid_z_points": int(valid_indices.shape[0]),
                    "finite_projected_points": int(valid_f.shape[0]),
                    "in_image_points": int(np.sum(in_image)),
                    "visible_points": int(visible_indices.shape[0]),
                }
            )

        variants[variant] = {
            "scene_seen_fraction_sampled": float(np.mean(seen)) if seen.shape[0] else 0.0,
            "median_valid_z_points": _median([x["valid_z_points"] for x in per_frame]),
            "median_in_image_points": _median([x["in_image_points"] for x in per_frame]),
            "median_visible_points": _median([x["visible_points"] for x in per_frame]),
            "per_frame": per_frame,
        }

    return {
        "scene_id": scene_id,
        "sampled_points": int(points_used.shape[0]),
        "total_points": int(points.shape[0]),
        "sample_seed": 0,
        "frames_used": len(frames_used),
        "total_frames": len(frames),
        "max_frames": int(max_frames),
        "z_tolerance_m": float(z_tolerance_m),
        "variants": variants,
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _collect_projection_reports(
    scans_root: Path,
    scene_ids: list[str],
    sample_points: int,
    max_frames: int,
    z_tolerance_m: float,
) -> list[dict[str, Any]]:
    reports = []
    for scene_id in scene_ids:
        scene_dir = scans_root / scene_id
        if not (scene_dir / f"{scene_id}_vh_clean_2.ply").exists():
            reports.append({"scene_id": scene_id, "skipped_reason": "missing_geometry"})
            continue
        if not (scene_dir / "color").is_dir() or not (scene_dir / "depth").is_dir():
            reports.append({"scene_id": scene_id, "skipped_reason": "missing_rgbd_frames"})
            continue
        try:
            reports.append(
                compute_projection_coverage(
                    scans_root=scans_root,
                    scene_id=scene_id,
                    sample_points=sample_points,
                    max_frames=max_frames,
                    z_tolerance_m=z_tolerance_m,
                )
            )
        except Exception as exc:
            reports.append({"scene_id": scene_id, "error": f"{type(exc).__name__}: {exc}"})
    return reports


def _build_report(args: argparse.Namespace, scene_ids: list[str]) -> dict[str, Any]:
    bbox_exists = (args.raw_zips_dir / BBOX_ZIP_NAME).exists()
    scan_scene_assets = args.scan_scene_assets and (bbox_exists or args.allow_missing_bbox)
    raw = inspect_raw_zips(args.raw_zips_dir, scene_ids, scan_scene_assets=scan_scene_assets)
    prepared = inspect_prepared_scenes(args.scans_root, scene_ids)
    report: dict[str, Any] = {
        "scene_ids": scene_ids,
        "raw": raw,
        "prepared": prepared,
    }

    if args.projection_coverage:
        report["projection_coverage"] = _collect_projection_reports(
            scans_root=args.scans_root,
            scene_ids=scene_ids,
            sample_points=args.sample_points,
            max_frames=args.max_frames,
            z_tolerance_m=args.z_tolerance_m,
        )

    report["ok"] = _report_is_ok(report, require_bbox=not args.allow_missing_bbox)
    report["errors"] = _report_errors(report, require_bbox=not args.allow_missing_bbox)
    return report


def _report_errors(report: dict[str, Any], require_bbox: bool) -> list[str]:
    errors: list[str] = []
    raw = report["raw"]
    if raw["perspective_zip_count"] < 1:
        errors.append(f"No *perspective_full*.zip files found under {raw['raw_zips_dir']}")
    if require_bbox and not raw["bbox_zip_exists"]:
        msg = (
            f"Missing {BBOX_ZIP_NAME} at {raw['bbox_zip_path']}. "
            "Submit the official Structured3D agreement form, download the bbox/instance archive, "
            f"and place or symlink it to {raw['bbox_zip_path']}."
        )
        candidates = raw.get("bbox_candidates") or []
        if candidates:
            msg += f" Found bbox-like candidates: {candidates}"
        errors.append(msg)

    for scene_id, assets in (raw.get("scene_assets") or {}).items():
        if not assets.get("has_perspective"):
            errors.append(f"{scene_id}: missing perspective assets in raw ZIPs")
        if require_bbox and not assets.get("has_bbox_3d"):
            errors.append(f"{scene_id}: missing bbox_3d.json in {BBOX_ZIP_NAME}")
        if require_bbox and not assets.get("has_instance_png"):
            errors.append(f"{scene_id}: missing perspective instance.png files in {BBOX_ZIP_NAME}")

    return errors


def _report_is_ok(report: dict[str, Any], require_bbox: bool) -> bool:
    return len(_report_errors(report, require_bbox=require_bbox)) == 0


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _print_summary(report: dict[str, Any]) -> None:
    print("Structured3D diagnostic report")
    print(f"  scenes: {', '.join(report['scene_ids'])}")
    print(f"  raw_zips_dir: {report['raw']['raw_zips_dir']}")
    print(f"  perspective zips: {report['raw']['perspective_zip_count']}")
    print(f"  bbox zip exists: {report['raw']['bbox_zip_exists']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    else:
        print("  preflight: ok")

    prepared = report.get("prepared", {}).get("scenes", [])
    if prepared:
        print("  prepared scenes:")
        for row in prepared:
            print(
                "    "
                f"{row['scene_id']}: frames={row['num_color']}, "
                f"points={row['geometry_vertex_count']}, "
                f"gt={row['gt_instance_ids_exists']}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured3D raw/prepared preflight and projection diagnostics for CHORUS.",
    )
    parser.add_argument("--raw-zips-dir", type=Path, default=DEFAULT_RAW_ZIPS_DIR)
    parser.add_argument("--scans-root", type=Path, default=DEFAULT_SCANS_ROOT)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--allow-missing-bbox", action="store_true")
    parser.add_argument("--skip-scene-asset-scan", dest="scan_scene_assets", action="store_false")
    parser.set_defaults(scan_scene_assets=True)
    parser.add_argument("--projection-coverage", action="store_true")
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--z-tolerance-m", type=float, default=1.0)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def _read_scene_list(path: Path) -> list[str]:
    scene_ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^(scene_\d{5})", line)
            scene_ids.append(match.group(1) if match else line.split()[0])
    return scene_ids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scene_list_file is not None:
        scene_ids = _read_scene_list(args.scene_list_file)
    else:
        scene_ids = scene_ids_from_range(args.scene_start, args.scene_count)

    if not scene_ids:
        print("No scene ids selected.", file=sys.stderr)
        return 2

    report = _build_report(args, scene_ids)
    if not args.no_write_report:
        _write_json(report, args.report_path)
        print(f"Wrote diagnostic report: {args.report_path}")
    _print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
