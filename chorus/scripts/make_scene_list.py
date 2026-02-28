from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from chorus.datasets.scannet.download import load_release_scene_ids


def _read_scene_ids_from_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_scene_ids_from_dir(path: Path) -> list[str]:
    return sorted(
        [
            p.name
            for p in path.iterdir()
            if p.is_dir() and p.name.startswith("scene")
        ]
    )


def _apply_max_scenes(scene_ids: list[str], max_scenes: int | None) -> list[str]:
    if max_scenes is None:
        return scene_ids
    return scene_ids[:max_scenes]


def _intersect_with_existing(scene_ids: list[str], scans_root: Path) -> list[str]:
    existing = {
        p.name
        for p in scans_root.iterdir()
        if p.is_dir() and p.name.startswith("scene")
    }
    return [scene_id for scene_id in scene_ids if scene_id in existing]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CHORUS scene list")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Read scene ids from an existing directory containing scene folders",
    )
    source.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Read scene ids from a txt file, for example an official ScanNet split file",
    )
    source.add_argument(
        "--use-release-list",
        action="store_true",
        help="Use the full official ScanNet release list from the downloader script",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output txt file, one scene id per line",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap on the number of scenes written",
    )
    parser.add_argument(
        "--intersect-with-existing-root",
        type=Path,
        default=None,
        help="Optionally keep only scene ids whose folders already exist under this root",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the final scene ids before writing. By default file order is preserved.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.from_dir is not None:
        scene_ids = _read_scene_ids_from_dir(args.from_dir)
        source_name = f"directory:{args.from_dir}"
    elif args.from_file is not None:
        scene_ids = _read_scene_ids_from_file(args.from_file)
        source_name = f"file:{args.from_file}"
    else:
        scene_ids = load_release_scene_ids()
        source_name = "official_release_list"

    if args.intersect_with_existing_root is not None:
        scene_ids = _intersect_with_existing(scene_ids, args.intersect_with_existing_root)

    if args.sort:
        scene_ids = sorted(scene_ids)

    scene_ids = _apply_max_scenes(scene_ids, args.max_scenes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{scene_id}\n" for scene_id in scene_ids), encoding="utf-8")

    print(f"Source: {source_name}")
    print(f"Wrote {len(scene_ids)} scene ids to {args.out}")


if __name__ == "__main__":
    main()