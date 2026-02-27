import argparse
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict

from config import (
    DEFAULT_CONTINUE_ON_ERROR,
    DEFAULT_DELETE_INTERMEDIATE,
    DEFAULT_GRANULARITIES,
    DEFAULT_KEEP_FULL_MODULO,
    REPORTS_ROOT,
    SCANS_ROOT,
    WANDB_CONFIG,
    CONFIG_PATH,
)
from scannet_download import download_scene, read_scene_ids
from chorus_pipeline import process_scene_chorus
from io_utils import verify_final_outputs
from wandb_utils import WandbReporter


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming ScanNet -> CHORUS PoC3 pipeline")
    parser.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Optional file with one scene id per line. If omitted, use full release list.",
    )
    parser.add_argument(
        "--granularities",
        default=",".join(DEFAULT_GRANULARITIES),
        help="Comma-separated granularity list (example: 0.2,0.5,0.8).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap for number of scenes processed.",
    )
    parser.add_argument(
        "--keep-full-modulo",
        type=int,
        default=DEFAULT_KEEP_FULL_MODULO,
        help="Keep full intermediate data for 1/modulo scenes.",
    )
    parser.add_argument(
        "--no-delete-intermediate",
        action="store_true",
        help="Disable cleanup of intermediates after successful processing.",
    )
    parser.add_argument(
        "--delete-intermediate",
        action="store_true",
        help="Force-enable cleanup even if config default disables it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next scene if one scene fails.",
    )
    parser.add_argument(
        "--wandb-off",
        action="store_true",
        help="Disable Weights & Biases reporting for this run.",
    )
    return parser.parse_args()


def _resolve_delete_intermediate(args: argparse.Namespace) -> bool:
    if args.delete_intermediate:
        return True
    if args.no_delete_intermediate:
        return False
    return DEFAULT_DELETE_INTERMEDIATE


def _resolve_continue_on_error(args: argparse.Namespace) -> bool:
    if args.continue_on_error:
        return True
    return DEFAULT_CONTINUE_ON_ERROR


def _run(args: argparse.Namespace, reporter: WandbReporter) -> Dict:
    granularities = [g.strip() for g in args.granularities.split(",") if g.strip()]
    delete_intermediate = _resolve_delete_intermediate(args)
    continue_on_error = _resolve_continue_on_error(args)

    scene_ids = read_scene_ids(args.scene_list_file, max_scenes=args.max_scenes)
    print(f"Selected scenes: {len(scene_ids)}")
    print(f"Granularities: {granularities}")
    print(f"Delete intermediates: {delete_intermediate}")
    print(f"Keep-full modulo: {args.keep_full_modulo}")

    summary = {"done": 0, "failed": 0, "skipped_done": 0, "failed_scenes": []}
    for idx, scene_id in enumerate(scene_ids, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(scene_ids)}] Scene: {scene_id}")
        print("=" * 80)

        scene_dir = SCANS_ROOT / scene_id
        try:
            manifest_path = scene_dir / "poc3_manifest.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest_data = {}
                outputs_ok, _ = verify_final_outputs(scene_dir, granularities)
                if manifest_data.get("status") == "done" and outputs_ok:
                    print("Scene already complete (manifest + outputs verified), skipping.")
                    summary["done"] += 1
                    summary["skipped_done"] += 1
                    reporter.log_scene(
                        scene_idx=idx,
                        scene_id=scene_id,
                        manifest=manifest_data,
                        scene_dir=scene_dir,
                    )
                    continue

            download_scene(scene_id, SCANS_ROOT, skip_existing=True)
            manifest = process_scene_chorus(
                scene_id=scene_id,
                scene_dir=scene_dir,
                granularities=granularities,
                keep_full_modulo=args.keep_full_modulo,
                delete_intermediate=delete_intermediate,
            )
            print(f"Scene status: {manifest['status']}")
            reporter.log_scene(
                scene_idx=idx,
                scene_id=scene_id,
                manifest=manifest,
                scene_dir=scene_dir,
            )
            if manifest["status"] == "done":
                summary["done"] += 1
            else:
                summary["failed"] += 1
                summary["failed_scenes"].append(scene_id)
                if not continue_on_error:
                    break
        except Exception as exc:
            summary["failed"] += 1
            summary["failed_scenes"].append(scene_id)
            print(f"Fatal scene error: {exc}")
            if not continue_on_error:
                break

    print("\n" + "=" * 80)
    print("Run Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    reporter.log_summary(summary)
    return summary


def main() -> None:
    args = _parse_args()
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_ROOT / f"poc3_streaming_run_{ts}.txt"

    run_cfg = {
        "config_path": str(CONFIG_PATH),
        "scans_root": str(SCANS_ROOT),
        "reports_root": str(REPORTS_ROOT),
        "default_granularities": DEFAULT_GRANULARITIES,
        "default_keep_full_modulo": DEFAULT_KEEP_FULL_MODULO,
        "run_name": f"poc3_streaming_{ts}",
        "run_id": f"poc3_{ts}",
    }

    reporter_cfg = dict(WANDB_CONFIG)
    if args.wandb_off:
        reporter_cfg["enabled"] = False
    reporter = WandbReporter(reporter_cfg, run_config=run_cfg, report_path=report_path)

    try:
        with report_path.open("w", encoding="utf-8") as report:
            tee_out = Tee(sys.stdout, report)
            tee_err = Tee(sys.stderr, report)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print(f"Run report: {report_path}")
                _run(args, reporter)
                print(f"Report saved: {report_path}")
    finally:
        reporter.finish()


if __name__ == "__main__":
    main()

