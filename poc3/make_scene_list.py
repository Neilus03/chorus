import argparse
from pathlib import Path

from config import DEFAULT_SCENE_LIST_FROM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scene id list file")
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=DEFAULT_SCENE_LIST_FROM_DIR,
        help="Directory containing scene folders (default: processed train split).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output txt file with one scene id per line.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional limit on number of scene ids.",
    )
    args = parser.parse_args()

    scene_ids = sorted(
        [p.name for p in args.from_dir.iterdir() if p.is_dir() and p.name.startswith("scene")]
    )
    if args.max_scenes is not None:
        scene_ids = scene_ids[: args.max_scenes]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(scene_ids)} scene ids to {args.out}")


if __name__ == "__main__":
    main()

