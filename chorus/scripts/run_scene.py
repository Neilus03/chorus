from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os

from chorus.core.pipeline.scene_pipeline import run_scene_pipeline
from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
from chorus.datasets.scannet.benchmark import (
    DEFAULT_SCANNET_EVAL_BENCHMARKS,
    parse_scannet_eval_benchmarks,
)
from chorus.datasets.scannet.adapter import ScanNetSceneAdapter


class _Tee(TextIO):
    """Write to multiple streams (e.g. terminal + log file)."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for f in self._streams:
            f.write(s)
            f.flush()
        return len(s)

    def flush(self) -> None:
        for f in self._streams:
            f.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty() if self._streams else False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CHORUS on a single scene (dataset-agnostic)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("CHORUS_DATASET", "scannet"),
        choices=["scannet", "structured3d", "scannetpp"],
        help="Dataset adapter to use",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Path to the scene directory, for example /scratch2/.../scene0000_00",
    )
    parser.add_argument(
        "--granularities",
        type=str,
        default="0.2,0.5,0.8",
        help="Comma-separated granularities",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Use every N-th frame",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CHORUS_DEVICE", "cuda:0"),
        help="Torch device, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=32,
        help="Number of TruncatedSVD components",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="HDBSCAN min_cluster_size",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.1,
        help="HDBSCAN cluster_selection_epsilon",
    )
    parser.add_argument(
        "--debug-first-n-frames",
        type=int,
        default=10,
        help="How many frames should print teacher debug statistics",
    )
    parser.add_argument(
        "--overwrite-teacher",
        action="store_true",
        help="Recompute teacher masks even if they already exist",
    )
    parser.add_argument(
        "--no-oracle-eval",
        action="store_true",
        help="Disable ScanNet oracle evaluation",
    )
    parser.add_argument(
        "--no-export-training-pack",
        action="store_true",
        help="Disable training pack export",
    )
    parser.add_argument(
        "--scannet-eval-benchmark",
        type=str,
        default=os.environ.get(
            "CHORUS_SCANNET_EVAL_BENCHMARK",
            ",".join(DEFAULT_SCANNET_EVAL_BENCHMARKS),
        ),
        help="Comma-separated ScanNet oracle benchmarks to run, for example 'scannet20,scannet200'.",
    )
    parser.add_argument(
        "--structured3d-raw-zips-dir",
        type=str,
        default=os.environ.get("CHORUS_STRUCTURED3D_RAW_ZIPS_DIR", "/scratch2/nedela/structured3d_raw"),
        help="Structured3D raw ZIP directory (only used when --dataset structured3d).",
    )
    parser.add_argument(
        "--scannetpp-eval-benchmark",
        type=str,
        default=os.environ.get("CHORUS_SCANNETPP_EVAL_BENCHMARK", "top100_instance"),
        help="ScanNet++ oracle benchmark filter (only used when --dataset scannetpp).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If set, mirror stdout/stderr to this UTF-8 text file (terminal output unchanged).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_fp: IO[str] | None = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_fp)
        sys.stderr = _Tee(sys.__stderr__, log_fp)

    try:
        _run(args)
    finally:
        if log_fp is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_fp.close()


def _run(args: argparse.Namespace) -> None:
    granularities = [float(g.strip()) for g in args.granularities.split(",") if g.strip()]
    if args.dataset == "scannet":
        scannet_eval_benchmarks = parse_scannet_eval_benchmarks(args.scannet_eval_benchmark)
        adapter = ScanNetSceneAdapter(
            scene_root=args.scene_dir,
            eval_benchmarks=scannet_eval_benchmarks,
        )
        run_oracle_eval = not args.no_oracle_eval
    elif args.dataset == "structured3d":
        # Structured3D integration lives under chorus/datasets/structured3d/.
        # This import is intentionally local to avoid making Structured3D a hard dependency
        # for users who only run ScanNet.
        try:
            from chorus.datasets.structured3d.adapter import Structured3DSceneAdapter
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Structured3D adapter not found. Expected module: chorus.datasets.structured3d.adapter "
                "(see chorus/datasets/structured3D/STRUCTURED3D_PLAN.md)."
            ) from e

        adapter = Structured3DSceneAdapter(
            scene_root=args.scene_dir,
            raw_zips_dir=args.structured3d_raw_zips_dir,
        )
        # Oracle evaluation is ScanNet-specific.
        run_oracle_eval = False
    elif args.dataset == "scannetpp":
        from chorus.datasets.scannetpp.adapter import ScanNetPPSceneAdapter

        adapter = ScanNetPPSceneAdapter(
            scene_root=args.scene_dir,
            eval_benchmark=args.scannetpp_eval_benchmark,
        )
        run_oracle_eval = not args.no_oracle_eval
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite_teacher,
    )

    summary = run_scene_pipeline(
        adapter=adapter,
        teacher=teacher,
        granularities=granularities,
        frame_skip=args.frame_skip,
        svd_components=args.svd_components,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        run_oracle_eval=run_oracle_eval,
        export_training_pack=not args.no_export_training_pack,
    )

    print("\n" + "=" * 80)
    print("CHORUS single-scene summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()