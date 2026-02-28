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
from chorus.orchestrators.streaming import read_scene_ids, run_streaming_scannet
from chorus.tracking.local_report import LocalTableReporter
from chorus.tracking.wandb import WandbReporter


class CombinedReporter:
    def __init__(self, reporters):
        self.reporters = [r for r in reporters if r is not None]

    def log_scene(self, result):
        for reporter in self.reporters:
            reporter.log_scene(result)

    def log_summary(self, summary):
        for reporter in self.reporters:
            reporter.log_summary(summary)

    def finish(self):
        for reporter in self.reporters:
            reporter.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CHORUS on many ScanNet scenes")

    parser.add_argument(
        "--scans-root",
        type=Path,
        default=Path(os.environ.get("CHORUS_SCANS_ROOT", "/scratch2/nedela/chorus_poc/scans")),
        help="Root directory containing ScanNet scene folders",
    )
    parser.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Optional txt file with one scene id per line",
    )
    parser.add_argument(
        "--use-release-list",
        action="store_true",
        help="Use the full official ScanNet release list from the downloader script",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional maximum number of scenes to process",
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
        "--overwrite-existing",
        action="store_true",
        help="Do not skip scenes even if their summary and outputs already exist",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining scenes if one scene fails",
    )
    parser.add_argument(
        "--no-oracle-eval",
        action="store_true",
        help="Disable ScanNet oracle evaluation",
    )
    parser.add_argument(
        "--no-export-litept",
        action="store_true",
        help="Disable LitePT pack export",
    )
    parser.add_argument(
        "--no-auto-download-missing",
        action="store_true",
        help="Disable automatic ScanNet scene download when data is missing",
    )
    parser.add_argument(
        "--no-cleanup-after-success",
        action="store_true",
        help="Keep raw and intermediate files after successful processing",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only ensure scenes are downloaded locally, do not run CHORUS",
    )
    parser.add_argument(
        "--max-download-retries",
        type=int,
        default=3,
        help="Maximum number of download attempts per scene",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional directory where local reports and run summary JSON will be saved",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases reporting",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chorus",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        help="W&B mode: online, offline, disabled",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    scans_root = args.scans_root.resolve()
    granularities = [float(g.strip()) for g in args.granularities.split(",") if g.strip()]

    scene_ids = read_scene_ids(
        scans_root=scans_root,
        scene_list_file=args.scene_list_file,
        max_scenes=args.max_scenes,
        use_release_list=args.use_release_list,
    )

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = scans_root / "_chorus_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"scans_root={scans_root}")
    print(f"num_scenes_selected={len(scene_ids)}")
    print(f"granularities={granularities}")
    print(f"frame_skip={args.frame_skip}")
    print(f"overwrite_existing={args.overwrite_existing}")
    print(f"continue_on_error={args.continue_on_error}")
    print(f"auto_download_missing={not args.no_auto_download_missing}")
    print(f"cleanup_after_success={not args.no_cleanup_after_success}")
    print(f"use_release_list={args.use_release_list}")
    print(f"scene_list_file={args.scene_list_file}")
    print(f"download_only={args.download_only}")
    print(f"max_download_retries={args.max_download_retries}")
    print(f"report_dir={report_dir}")
    print(f"wandb={args.wandb}")

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite_teacher,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_reporter = LocalTableReporter(report_dir=report_dir)
    wandb_reporter = WandbReporter(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=f"chorus_streaming_{timestamp}",
        config={
            "scans_root": str(scans_root),
            "granularities": granularities,
            "frame_skip": args.frame_skip,
            "download_only": args.download_only,
            "max_download_retries": args.max_download_retries,
        },
    )
    reporter = CombinedReporter([local_reporter, wandb_reporter])

    try:
        run_summary = run_streaming_scannet(
            scans_root=scans_root,
            scene_ids=scene_ids,
            teacher=teacher,
            granularities=granularities,
            frame_skip=args.frame_skip,
            svd_components=args.svd_components,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            run_oracle_eval=not args.no_oracle_eval,
            export_litept=not args.no_export_litept,
            overwrite_existing=args.overwrite_existing,
            continue_on_error=args.continue_on_error,
            auto_download_missing=not args.no_auto_download_missing,
            cleanup_after_success=not args.no_cleanup_after_success,
            download_only=args.download_only,
            max_download_retries=args.max_download_retries,
            reporter=reporter,
        )

        report_path = report_dir / f"streaming_run_summary_{timestamp}.json"
        save_json(run_summary, report_path)

        print("\n" + "=" * 90)
        print("Streaming run summary")
        print("=" * 90)
        print(json.dumps(run_summary, indent=2))
        print(f"\nSaved run summary to: {report_path}")
        print(f"Local scene table CSV: {local_reporter.scene_csv_path}")
        print(f"Latest run summary JSON: {local_reporter.summary_json_path}")

    finally:
        reporter.finish()


if __name__ == "__main__":
    main()