import importlib.util
from pathlib import Path
from typing import Optional

from config import SCANNET_DOWNLOADER


def _load_downloader_module():
    if not SCANNET_DOWNLOADER.exists():
        raise FileNotFoundError(f"Missing ScanNet downloader: {SCANNET_DOWNLOADER}")
    spec = importlib.util.spec_from_file_location("scannet_dl", SCANNET_DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def download_scene(scene_id: str, scans_root: Path, skip_existing: bool = True) -> None:
    dl = _load_downloader_module()
    out_dir = scans_root / scene_id
    dl.download_scan(
        scan_id=scene_id,
        out_dir=str(out_dir),
        file_types=list(dl.FILETYPES),
        use_v1_sens=True,
        skip_existing=skip_existing,
    )


def load_release_scene_ids() -> list[str]:
    dl = _load_downloader_module()
    release_file = dl.BASE_URL + dl.RELEASE + ".txt"
    return dl.get_release_scans(release_file)


def read_scene_ids(scene_list_file: Optional[Path], max_scenes: Optional[int] = None) -> list[str]:
    if scene_list_file is None:
        ids = load_release_scene_ids()
    else:
        ids = [line.strip() for line in scene_list_file.read_text().splitlines() if line.strip()]

    if max_scenes is not None:
        ids = ids[:max_scenes]
    return ids

