from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
from datetime import datetime

from chorus.core.quality.diagnostics import save_json
from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
from chorus.datasets.structured3d.evaluation import Structured3DEvaluationHooks
from chorus.orchestrators.streaming import (
    enumerate_structured3d_scene_ids,
    read_structured3d_scene_ids,
    run_streaming_structured3d,
)
from chorus.tracking.local_report import LocalTableReporter
from chorus.tracking.wandb import WandbReporter


class CombinedReporter:
    def __init__(self, reporters):
        self.reporters = [reporter for reporter in reporters if reporter is not None]

    def log_scene(self, result):
        for reporter in self.reporters:
            reporter.log_scene(result)

    def log_summary(self, summary):
        for reporter in self.reporters:
            reporter.log_summary(summary)

    def log_event(self, payload):
        for reporter in self.reporters:
            if hasattr(reporter, "log_event"):
                reporter.log_event(payload)

    def finish(self):
        for reporter in self.reporters:
            reporter.finish()


def _map_scratch2_to_euler(path: Path) -> Path:
    path = Path(path)
    raw = str(path)
    user = os.environ.get("USER", "nedela")
    if raw.startswith(f"/scratch2/{user}/"):
        mapped = Path(raw.replace(f"/scratch2/{user}/", f"/cluster/work/igp_psr/{user}/", 1))
        if mapped.exists():
            return mapped
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CHORUS on many prepared Structured3D scene directories",
    )

    parser.add_argument(
        "--scans-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CHORUS_STRUCTURED3D_SCANS_ROOT",
                "/scratch2/nedela/chorus_poc/structured3d_scans",
            )
        ),
        help="Root containing one folder per scene (e.g. scene_00000/).",
    )
    parser.add_argument(
        "--structured3d-raw-zips-dir",
        type=Path,
        default=Path(os.environ.get("CHORUS_STRUCTURED3D_RAW_ZIPS_DIR", "/scratch2/nedela/structured3d_raw")),
        help="Directory with Structured3D_perspective_full_*.zip (and Structured3D_bbox.zip for GT).",
    )
    parser.add_argument(
        "--structured3d-eval-benchmark",
        type=str,
        default=os.environ.get("CHORUS_STRUCTURED3D_EVAL_BENCHMARK", "structured3d_full"),
        help="Oracle output filename suffix / eval_benchmark key.",
    )
    parser.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Text file with one scene id per line. Overrides directory listing and --enumerate-scenes-start.",
    )
    parser.add_argument(
        "--enumerate-scenes-start",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use consecutive ids scene_NNNNN, scene_NNNNN+1, ... without requiring folders under "
            "--scans-root (streaming creates each scene directory). Requires --max-scenes."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Max scenes: with directory listing, truncate after sort; with --enumerate-scenes-start, number of consecutive ids.",
    )
    parser.add_argument(
        "--granularities",
        type=str,
        default="0.2,0.5,0.8",
        help="Comma-separated granularities.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Use every N-th frame.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CHORUS_DEVICE", "cuda:0"),
        help="Torch device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=32,
        help="Number of TruncatedSVD components.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples.",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.1,
        help="HDBSCAN cluster_selection_epsilon.",
    )
    parser.add_argument(
        "--debug-first-n-frames",
        type=int,
        default=10,
        help="How many frames should print teacher debug statistics.",
    )
    parser.add_argument(
        "--overwrite-teacher",
        action="store_true",
        default=False,
        help="Recompute teacher masks even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Do not skip scenes even if their summary and outputs already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        default=True,
        action="store_true",
        help="Continue processing remaining scenes if one scene fails.",
    )
    parser.add_argument(
        "--no-oracle-eval",
        action="store_true",
        help="Disable oracle evaluation.",
    )
    parser.add_argument(
        "--no-export-training-pack",
        action="store_true",
        help="Disable training pack export.",
    )
    parser.add_argument(
        "--no-auto-download-missing",
        action="store_true",
        help="Structured3D has no HTTP downloader; zips must exist under --structured3d-raw-zips-dir.",
    )
    parser.add_argument(
        "--no-cleanup-after-success",
        action="store_true",
        help="Keep intermediate files after successful processing.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only verify raw zips / scene availability; do not run the CHORUS pipeline.",
    )
    parser.add_argument(
        "--max-download-retries",
        type=int,
        default=3,
        help="Unused for Structured3D (kept for CLI parity with other streaming scripts).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for local CSV/JSON reports (default: <scans-root>/_chorus_reports).",
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action="store_true",
        help="Enable Weights & Biases reporting.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases reporting.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chorus",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", os.environ.get("CHORUS_WANDB_MODE", "online")),
        help="W&B mode: online, offline, disabled.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path(os.environ.get("CHORUS_WANDB_DIR", "/scratch2/nedela/chorus_wandb")),
        help="Directory where W&B run files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    scans_root = args.scans_root.resolve()
    raw_zips_dir = args.structured3d_raw_zips_dir.resolve()
    granularities = [float(g.strip()) for g in args.granularities.split(",") if g.strip()]
    evaluation_hooks = Structured3DEvaluationHooks(args.structured3d_eval_benchmark)

    if args.scene_list_file is not None:
        scene_ids = read_structured3d_scene_ids(
            scans_root=scans_root,
            scene_list_file=args.scene_list_file,
            max_scenes=args.max_scenes,
        )
    elif args.enumerate_scenes_start is not None:
        if args.max_scenes is None:
            raise SystemExit("--enumerate-scenes-start requires --max-scenes (e.g. 10)")
        scene_ids = enumerate_structured3d_scene_ids(
            start_index=args.enumerate_scenes_start,
            count=args.max_scenes,
        )
    else:
        scene_ids = read_structured3d_scene_ids(
            scans_root=scans_root,
            scene_list_file=None,
            max_scenes=args.max_scenes,
        )

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = scans_root / "_chorus_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"scans_root={scans_root}")
    print(f"raw_zips_dir={raw_zips_dir}")
    print(f"num_scenes_selected={len(scene_ids)}")
    print(f"granularities={granularities}")
    print(f"frame_skip={args.frame_skip}")
    print(f"structured3d_eval_benchmark={evaluation_hooks.eval_benchmark}")
    print(f"overwrite_existing={args.overwrite_existing}")
    print(f"continue_on_error={args.continue_on_error}")
    print(f"scene_list_file={args.scene_list_file}")
    print(f"download_only={args.download_only}")
    print(f"report_dir={report_dir}")
    print(f"wandb={args.wandb}")

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite_teacher,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_reporter = LocalTableReporter(
        report_dir=report_dir,
        extra_fieldnames=evaluation_hooks.scene_metric_fieldnames(),
    )

    wandb_dir = _map_scratch2_to_euler(args.wandb_dir)
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb_reporter = WandbReporter(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=f"chorus_streaming_structured3d_{timestamp}",
        wandb_dir=wandb_dir,
        config={
            "scans_root": str(scans_root),
            "raw_zips_dir": str(raw_zips_dir),
            "granularities": granularities,
            "frame_skip": args.frame_skip,
            "eval_benchmark": evaluation_hooks.eval_benchmark,
            "download_only": args.download_only,
        },
        extra_metric_fields=evaluation_hooks.scene_metric_fieldnames(),
    )
    reporter = CombinedReporter([local_reporter, wandb_reporter])

    try:
        run_summary = run_streaming_structured3d(
            scans_root=scans_root,
            raw_zips_dir=raw_zips_dir,
            scene_ids=scene_ids,
            teacher=teacher,
            granularities=granularities,
            eval_benchmark=evaluation_hooks.eval_benchmark,
            frame_skip=args.frame_skip,
            svd_components=args.svd_components,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            run_oracle_eval=not args.no_oracle_eval,
            export_training_pack=not args.no_export_training_pack,
            overwrite_existing=args.overwrite_existing,
            continue_on_error=args.continue_on_error,
            auto_download_missing=not args.no_auto_download_missing,
            cleanup_after_success=not args.no_cleanup_after_success,
            download_only=args.download_only,
            max_download_retries=args.max_download_retries,
            reporter=reporter,
        )

        report_path = report_dir / f"streaming_structured3d_run_summary_{timestamp}.json"
        save_json(run_summary, report_path)

        print("\n" + "=" * 90)
        print("Streaming run summary (Structured3D)")
        print("=" * 90)
        run_timing = run_summary.get("run_timing") or {}
        if run_timing:
            print(
                f"Run wall: {run_timing.get('wall_clock_hours', 0):.3f} h | "
                f"full pipeline scenes: {run_timing.get('num_scenes_fully_processed', 0)} | "
                f"pace (full pipeline time): {run_timing.get('pace_full_pipeline_scenes_per_hour')} sc/h | "
                f"pace (job wall): {run_timing.get('pace_requested_scenes_per_hour_wall')} sc/h"
            )
        print(json.dumps(run_summary, indent=2))
        print(f"\nSaved run summary to: {report_path}")
        print(f"Local scene table CSV: {local_reporter.scene_csv_path}")
        print(f"Latest run summary JSON: {local_reporter.summary_json_path}")
        if not args.no_oracle_eval:
            for line in evaluation_hooks.render_run_summary(run_summary, granularities):
                print(line)
    finally:
        reporter.finish()


if __name__ == "__main__":
    main()
