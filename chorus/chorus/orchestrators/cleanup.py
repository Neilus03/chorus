from __future__ import annotations

import shutil
from pathlib import Path


def cleanup_scene_intermediates(
    scene_dir: Path,
    granularities: list[float],
    delete_rgbd: bool = True,
    delete_teacher_masks: bool = True,
    delete_svd_features: bool = True,
    delete_raw_source_files: bool = True,
    raw_source_suffixes: tuple[str, ...] = (".sens", ".zip"),
) -> dict[str, list[str]]:
    scene_dir = Path(scene_dir)

    deleted: list[str] = []
    skipped: list[str] = []

    if delete_rgbd:
        for rel in ["color", "depth", "pose", "intrinsic"]:
            path = scene_dir / rel
            if path.exists():
                shutil.rmtree(path)
                deleted.append(rel + "/")
    else:
        skipped.append("rgbd_dirs")

    if delete_teacher_masks:
        for granularity in granularities:
            path = scene_dir / f"unsam_masks_g{granularity}"
            if path.exists():
                shutil.rmtree(path)
                deleted.append(f"unsam_masks_g{granularity}/")
    else:
        skipped.append("teacher_masks")

    if delete_svd_features:
        for granularity in granularities:
            path = scene_dir / f"svd_features_g{granularity}.npy"
            if path.exists():
                path.unlink()
                deleted.append(path.name)
    else:
        skipped.append("svd_features")

    if delete_raw_source_files:
        for path in scene_dir.iterdir():
            if not path.is_file():
                continue

            name = path.name
            if any(name.endswith(suffix) for suffix in raw_source_suffixes):
                path.unlink()
                deleted.append(name)
    else:
        skipped.append("raw_source_files")

    return {
        "deleted": deleted,
        "skipped": skipped,
    }