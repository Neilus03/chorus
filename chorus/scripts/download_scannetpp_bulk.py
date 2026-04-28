#!/usr/bin/env python3
"""
Bulk downloader for ScanNet++ using the official release endpoints.

This is intended for the "download everything quickly" case: parallel downloads,
skip-existing by default, and a clear failure summary at the end.

It uses the repo's canonical helpers in `chorus.datasets.scannetpp.download`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.datasets.scannetpp.download import (  # noqa: E402
    SCANNETPP_SPLITS,
    download_scene,
    ensure_metadata,
    read_split_scene_ids,
    resolve_scannetpp_dataset_root,
    resolve_scannetpp_token,
)

DEFAULT_SPLITS = ("nvs_sem_train", "nvs_sem_val", "sem_test")
UNSUPPORTED_FOR_CHORUS = ("nvs_test", "nvs_test_iphone")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk download ScanNet++ scenes.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("CHORUS_SCANNETPP_DATA_ROOT", "/scratch2/nedela/scannetpp_data")),
        help="ScanNet++ dataset root containing metadata/, splits/, and data/.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="ScanNet++ token. If omitted, uses CHORUS_SCANNETPP_TOKEN or SCANNETPP_TOKEN.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help=f"Comma-separated splits to download. Supported: {', '.join(SCANNETPP_SPLITS)}",
    )
    p.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Optional txt file with one scene id per line (overrides --splits).",
    )
    p.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap on number of scenes (after split expansion).",
    )
    p.add_argument(
        "--require-annotations",
        action="store_true",
        default=True,
        help="Also download scan annotations needed for oracle evaluation (default: true).",
    )
    p.add_argument(
        "--no-require-annotations",
        action="store_true",
        help="Disable annotation downloads (faster, smaller; inference-only).",
    )
    p.add_argument(
        "--include-semantic-mesh",
        action="store_true",
        default=False,
        help="Also download semantic mesh if provided by the release.",
    )
    p.add_argument(
        "--overwrite-metadata",
        action="store_true",
        default=False,
        help="Redownload split/meta files even if present.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip assets already present on disk (default: true).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force redownload even if files exist.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download workers (IO-bound; tune to your bandwidth).",
    )
    return p.parse_args()


def _read_scene_ids_from_args(args: argparse.Namespace, *, dataset_root: Path) -> list[str]:
    if args.scene_list_file is not None:
        scene_ids = [line.strip() for line in args.scene_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return scene_ids[: args.max_scenes] if args.max_scenes else scene_ids

    requested_splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    for split in requested_splits:
        if split not in SCANNETPP_SPLITS:
            raise ValueError(f"Unsupported split {split!r}. Supported: {', '.join(SCANNETPP_SPLITS)}")
        if split in UNSUPPORTED_FOR_CHORUS:
            raise ValueError(
                f"Split {split!r} does not contain scan assets required for CHORUS lifting. "
                f"Use one of: {', '.join(DEFAULT_SPLITS)}"
            )

    scene_ids: list[str] = []
    for split in requested_splits:
        scene_ids.extend(read_split_scene_ids(split=split, dataset_root=dataset_root))

    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for sid in scene_ids:
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(sid)

    if args.max_scenes is not None:
        deduped = deduped[: int(args.max_scenes)]
    return deduped


def main() -> None:
    args = _parse_args()
    dataset_root = resolve_scannetpp_dataset_root(dataset_root=args.dataset_root)

    token = resolve_scannetpp_token(args.token)

    require_annotations = bool(args.require_annotations) and not bool(args.no_require_annotations)
    skip_existing = bool(args.skip_existing) and not bool(args.no_skip_existing)

    ensure_metadata(dataset_root=dataset_root, token=token, overwrite=bool(args.overwrite_metadata))

    scene_ids = _read_scene_ids_from_args(args, dataset_root=dataset_root)
    if not scene_ids:
        print("No scenes selected; nothing to download.")
        return

    print(f"dataset_root={dataset_root}")
    print(f"num_scenes={len(scene_ids)}")
    print(f"require_annotations={require_annotations}")
    print(f"include_semantic_mesh={bool(args.include_semantic_mesh)}")
    print(f"skip_existing={skip_existing}")
    print(f"workers={int(args.workers)}")

    started = time.perf_counter()
    status = Counter()
    failures: list[tuple[str, str]] = []

    def _worker(scene_id: str) -> tuple[str, str]:
        try:
            download_scene(
                scene_id=scene_id,
                dataset_root=dataset_root,
                require_annotations=require_annotations,
                skip_existing=skip_existing,
                include_semantic_mesh=bool(args.include_semantic_mesh),
                token=token,
            )
            return scene_id, "ok"
        except Exception as exc:  # noqa: BLE001
            return scene_id, f"failed: {exc!r}"

    with futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {ex.submit(_worker, sid): sid for sid in scene_ids}
        done = 0
        for f in futures.as_completed(futs):
            sid, result = f.result()
            done += 1
            if result == "ok":
                status["ok"] += 1
            else:
                status["failed"] += 1
                failures.append((sid, result))
            if done % 10 == 0 or done == len(scene_ids):
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-9)
                print(f"[{done}/{len(scene_ids)}] ok={status['ok']} failed={status['failed']} rate={rate:.2f} scenes/s")

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 80)
    print(f"Done in {elapsed/3600.0:.2f} h. ok={status['ok']} failed={status['failed']}")
    if failures:
        print("\nFailures (first 50):")
        for sid, msg in failures[:50]:
            print(f"- {sid}: {msg}")


if __name__ == "__main__":
    main()

