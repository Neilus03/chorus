from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _resolve_downloader_path() -> Path:
    user = os.environ.get("USER", "nedela")

    candidates = [
        os.environ.get("CHORUS_SCANNET_DOWNLOADER", ""),
        f"/scratch2/{user}/chorus/tools/download-scannet.py",
        f"/scratch2/{user}/chorus_poc/tools/download-scannet.py",
        "/scratch2/nedela/chorus_poc/tools/download-scannet.py",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find download-scannet.py. "
        "Set CHORUS_SCANNET_DOWNLOADER to the ScanNet downloader script path."
    )


def _load_downloader_module():
    downloader_path = _resolve_downloader_path()

    spec = importlib.util.spec_from_file_location("scannet_dl", downloader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ScanNet downloader module from {downloader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_scene(
    scene_id: str,
    scans_root: Path,
    skip_existing: bool = True,
) -> Path:
    scans_root = Path(scans_root)
    scans_root.mkdir(parents=True, exist_ok=True)

    dl = _load_downloader_module()
    out_dir = scans_root / scene_id

    print(f"Downloading ScanNet scene {scene_id} into {out_dir}")

    dl.download_scan(
        scan_id=scene_id,
        out_dir=str(out_dir),
        file_types=list(dl.FILETYPES),
        use_v1_sens=True,
        skip_existing=skip_existing,
    )

    return out_dir


def load_release_scene_ids() -> list[str]:
    dl = _load_downloader_module()
    release_file = dl.BASE_URL + dl.RELEASE + ".txt"
    scene_ids = dl.get_release_scans(release_file)
    return [str(scene_id).strip() for scene_id in scene_ids if str(scene_id).strip()]