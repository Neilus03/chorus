from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from chorus.datasets.scannet.adapter import ScanNetSceneAdapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a ScanNet scene by extracting RGB-D frames")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Path to the ScanNet scene directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    adapter = ScanNetSceneAdapter(scene_root=args.scene_dir)
    adapter.prepare()

    frames = adapter.list_frames()
    print(f"Prepared scene {adapter.scene_id}")
    print(f"Dataset: {adapter.dataset_name}")
    print(f"Frames available: {len(frames)}")
    print(f"Geometry: {adapter.get_geometry_record().geometry_path}")


if __name__ == "__main__":
    main()