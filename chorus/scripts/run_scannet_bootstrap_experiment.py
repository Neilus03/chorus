#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.core.experiments.bootstrap_scannet import (  # noqa: E402
    BootstrapConfig,
    run_bootstrap_scene_experiment,
)
from chorus.core.quality.diagnostics import save_json  # noqa: E402
from chorus.datasets.scannet.benchmark import (  # noqa: E402
    DEFAULT_SCANNET_EVAL_BENCHMARKS,
    parse_scannet_eval_benchmarks,
)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an independent ScanNet bootstrap stable-core pseudolabel experiment."
    )
    parser.add_argument(
        "--scans-root",
        type=Path,
        default=Path(os.environ.get("CHORUS_SCANS_ROOT", "/scratch2/nedela/chorus_poc/scans")),
    )
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <scans-root>/_chorus_bootstrap_experiments",
    )
    parser.add_argument("--granularities", type=_parse_float_tuple, default=(0.2, 0.5, 0.8))
    parser.add_argument("--num-bootstraps", type=int, default=4)
    parser.add_argument("--frame-fraction", type=float, default=0.25)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--max-frames-per-bootstrap", type=int, default=None)
    parser.add_argument("--frame-sampling", choices=("disjoint", "all"), default="disjoint")
    parser.add_argument("--frame-seed", type=int, default=0)
    parser.add_argument("--hdbscan-max-samples", type=int, default=75_000)
    parser.add_argument("--hdbscan-subsample-seed", type=int, default=0)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-iou-threshold", type=float, default=0.35)
    parser.add_argument("--min-fused-points", type=int, default=30)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--scannet-eval-benchmark",
        type=str,
        default=os.environ.get(
            "CHORUS_SCANNET_EVAL_BENCHMARK",
            ",".join(DEFAULT_SCANNET_EVAL_BENCHMARKS),
        ),
    )
    parser.add_argument("--device", type=str, default=os.environ.get("CHORUS_DEVICE", "cuda:0"))
    parser.add_argument("--debug-first-n-frames", type=int, default=10)
    parser.add_argument("--overwrite-teacher", action="store_true", default=False)
    parser.add_argument("--export-bootstrap-ply", action="store_true", default=False)
    parser.add_argument("--no-oracle-eval", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and obj != obj:
        return None
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return obj


def main() -> None:
    args = _parse_args()
    from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
    from chorus.orchestrators.streaming import read_scene_ids

    scans_root = args.scans_root.resolve()
    output_root = args.output_root or (scans_root / "_chorus_bootstrap_experiments")
    run_id = args.run_id or f"bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = read_scene_ids(
        scans_root=scans_root,
        scene_list_file=args.scene_list_file,
        max_scenes=args.max_scenes,
        use_release_list=False,
    )
    eval_benchmarks = tuple(parse_scannet_eval_benchmarks(args.scannet_eval_benchmark))
    config = BootstrapConfig(
        granularities=tuple(float(g) for g in args.granularities),
        num_bootstraps=int(args.num_bootstraps),
        frame_fraction=float(args.frame_fraction),
        frame_skip=int(args.frame_skip),
        max_frames_per_bootstrap=args.max_frames_per_bootstrap,
        frame_sampling=str(args.frame_sampling),
        hdbscan_max_samples=int(args.hdbscan_max_samples),
        hdbscan_subsample_seed=int(args.hdbscan_subsample_seed),
        support_threshold=float(args.support_threshold),
        cluster_iou_threshold=float(args.cluster_iou_threshold),
        min_fused_points=int(args.min_fused_points),
        svd_components=int(args.svd_components),
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        run_oracle_eval=not args.no_oracle_eval,
        export_bootstrap_ply=bool(args.export_bootstrap_ply),
        overwrite_teacher=bool(args.overwrite_teacher),
        eval_benchmarks=eval_benchmarks,
    )

    print(f"scans_root={scans_root}")
    print(f"run_dir={run_dir}")
    print(f"num_scenes_selected={len(scene_ids)}")
    print(f"granularities={config.granularities}")
    print(f"num_bootstraps={config.num_bootstraps}")
    print(f"frame_fraction={config.frame_fraction}")
    print(f"frame_skip={config.frame_skip}")
    print(f"max_frames_per_bootstrap={config.max_frames_per_bootstrap}")
    print(f"hdbscan_max_samples={config.hdbscan_max_samples}")

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite_teacher,
    )

    run_start = time.perf_counter()
    run_summary: dict[str, Any] = {
        "experiment": "scannet_bootstrap_stable_core",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scans_root": str(scans_root),
        "started_at": datetime.now().isoformat(),
        "scene_ids": scene_ids,
        "config": _json_safe(config.__dict__),
        "done": 0,
        "failed": 0,
        "failed_scenes": [],
        "scene_results": [],
    }

    for idx, scene_id in enumerate(scene_ids, start=1):
        print("\n" + "=" * 90)
        print(f"[{idx}/{len(scene_ids)}] scene={scene_id}")
        print("=" * 90)
        scene_root = scans_root / scene_id
        output_scene_dir = run_dir / scene_id
        started = time.perf_counter()
        try:
            scene_summary = run_bootstrap_scene_experiment(
                scene_root=scene_root,
                output_scene_dir=output_scene_dir,
                teacher=teacher,
                config=config,
                frame_seed=args.frame_seed,
            )
            result = {
                "scene_id": scene_id,
                "status": "done",
                "duration_seconds": float(time.perf_counter() - started),
                "summary_path": scene_summary.get("summary_path"),
                "training_pack_dir": scene_summary.get("training_pack_dir"),
                "timing_seconds": scene_summary.get("timing_seconds"),
                "oracle_ap25_small": scene_summary.get("oracle_ap25_small"),
                "oracle_ap50_small": scene_summary.get("oracle_ap50_small"),
                "oracle_ap25_medium": scene_summary.get("oracle_ap25_medium"),
                "oracle_ap50_medium": scene_summary.get("oracle_ap50_medium"),
                "oracle_ap25_large": scene_summary.get("oracle_ap25_large"),
                "oracle_ap50_large": scene_summary.get("oracle_ap50_large"),
                "oracle_nmi": scene_summary.get("oracle_nmi"),
                "oracle_ari": scene_summary.get("oracle_ari"),
            }
            run_summary["done"] += 1
            print(f"done scene={scene_id} summary={scene_summary.get('summary_path')}")
        except Exception as exc:
            result = {
                "scene_id": scene_id,
                "status": "failed",
                "duration_seconds": float(time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
            }
            run_summary["failed"] += 1
            run_summary["failed_scenes"].append(scene_id)
            print(f"failed scene={scene_id}: {result['error']}")
            if not args.continue_on_error:
                run_summary["scene_results"].append(result)
                break

        run_summary["scene_results"].append(result)
        run_summary["elapsed_seconds"] = float(time.perf_counter() - run_start)
        save_json(_json_safe(run_summary), run_dir / "bootstrap_run_summary.json")

    run_summary["finished_at"] = datetime.now().isoformat()
    run_summary["elapsed_seconds"] = float(time.perf_counter() - run_start)
    report_path = run_dir / "bootstrap_run_summary.json"
    save_json(_json_safe(run_summary), report_path)
    print(f"\nSaved run summary to: {report_path}")
    print(json.dumps(_json_safe(run_summary), indent=2))


if __name__ == "__main__":
    main()
