import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_teacher_mask_count(scene_dir: Path, frame_skip: int) -> int | None:
    color_dir = scene_dir / "color"
    if not color_dir.is_dir():
        return None
    frames = [p.name for p in color_dir.iterdir() if p.suffix == ".jpg"]
    frames_sorted = sorted(frames, key=lambda name: int(Path(name).stem))
    return len(frames_sorted[::frame_skip])


def has_complete_teacher_outputs(scene_dir: Path, granularity: str, frame_skip: int) -> bool:
    masks_dir = scene_dir / f"unsam_masks_g{granularity}"
    if not masks_dir.is_dir():
        return False
    expected = expected_teacher_mask_count(scene_dir, frame_skip)
    found = len([p for p in masks_dir.iterdir() if p.suffix == ".npy"])
    if expected is None:
        return found > 0
    return found >= expected


def has_cluster_outputs(scene_dir: Path, granularity: str) -> bool:
    labels = scene_dir / f"chorus_instance_labels_g{granularity}.npy"
    result = scene_dir / f"chorus_instance_result_g{granularity}.ply"
    return labels.exists() and result.exists()


def has_oracle_metrics(scene_dir: Path) -> bool:
    return (scene_dir / "oracle_metrics.json").exists()


def has_oracle_outputs(scene_dir: Path) -> bool:
    metrics_file = scene_dir / "oracle_metrics.json"
    pooled_labels_file = scene_dir / "chorus_oracle_best_combined_labels.npy"
    pooled_ply_file = scene_dir / "chorus_oracle_best_combined.ply"
    return metrics_file.exists() and pooled_labels_file.exists() and pooled_ply_file.exists()


def stable_keep_full(scene_id: str, modulo: int) -> bool:
    digest = hashlib.sha1(scene_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % modulo
    return bucket == 0


def verify_final_outputs(scene_dir: Path, granularities: List[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    base_mesh = scene_dir / f"{scene_dir.name}_vh_clean_2.ply"
    if not base_mesh.exists():
        missing.append(base_mesh.name)

    for g in granularities:
        for rel in [f"chorus_instance_labels_g{g}.npy", f"chorus_instance_result_g{g}.ply"]:
            p = scene_dir / rel
            if not p.exists():
                missing.append(rel)
            elif p.stat().st_size == 0:
                missing.append(f"{rel} (empty)")

    metrics = scene_dir / "oracle_metrics.json"
    if not metrics.exists():
        missing.append(metrics.name)
    pooled_labels = scene_dir / "chorus_oracle_best_combined_labels.npy"
    if not pooled_labels.exists():
        missing.append(pooled_labels.name)
    pooled_ply = scene_dir / "chorus_oracle_best_combined.ply"
    if not pooled_ply.exists():
        missing.append(pooled_ply.name)

    return len(missing) == 0, missing


def cleanup_intermediate_data(scene_dir: Path, granularities: List[str]) -> Dict[str, List[str]]:
    deleted: List[str] = []
    skipped: List[str] = []

    # Whole dirs
    for rel in ["color", "depth", "pose", "intrinsic", "sam_masks"]:
        p = scene_dir / rel
        if p.exists():
            shutil.rmtree(p)
            deleted.append(rel + "/")

    for g in granularities:
        p = scene_dir / f"unsam_masks_g{g}"
        if p.exists():
            shutil.rmtree(p)
            deleted.append(f"unsam_masks_g{g}/")

    # Files by suffix/pattern
    for p in scene_dir.iterdir():
        name = p.name
        if p.is_file() and (name.endswith(".sens") or name.endswith(".zip")):
            p.unlink()
            deleted.append(name)
        elif p.is_file() and name.startswith("svd_features_g") and name.endswith(".npy"):
            p.unlink()
            deleted.append(name)

    return {"deleted": deleted, "skipped": skipped}


def write_manifest(scene_dir: Path, manifest: Dict) -> Path:
    out = scene_dir / "poc3_manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return out

