from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.common.progress import phase_timer
from chorus.core.pipeline.project_cluster_stage import run_project_cluster_stage
from chorus.core.pipeline.teacher_stage import run_teacher_stage
from chorus.core.quality.diagnostics import save_json
from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics
from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
from chorus.datasets.scannetpp.adapter import ScanNetPPSceneAdapter
from chorus.datasets.scannetpp.download import resolve_scannetpp_dataset_root
from chorus.orchestrators.streaming import read_scannetpp_scene_ids


def _bucket_slug(bucket_name: object) -> str | None:
    lower = str(bucket_name).strip().lower()
    if lower.startswith("small"):
        return "small"
    if lower.startswith("medium"):
        return "medium"
    if lower.startswith("large"):
        return "large"
    return None


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _flatten_metrics(scene_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_id": scene_summary["scene_id"],
        "status": scene_summary.get("status", "done"),
    }

    by_g = scene_summary.get("scene_intrinsic_metrics", {}).get("by_granularity", {}) or {}
    seen_vals: list[float] = []
    for key, metrics in sorted(by_g.items()):
        safe_key = str(key).replace(".", "_")
        seen = metrics.get("seen_points_fraction")
        row[f"{safe_key}_seen_points_fraction"] = seen
        row[f"{safe_key}_num_clusters"] = metrics.get("num_clusters")
        row[f"{safe_key}_noise_fraction_seen"] = metrics.get("noise_fraction_seen")
        if seen is not None:
            seen_vals.append(float(seen))
    row["mean_seen_points_fraction"] = _mean(seen_vals)

    oracle = scene_summary.get("oracle_summary", {}) or {}
    clustering = oracle.get("clustering_metrics", {}) or {}
    row["NMI"] = clustering.get("NMI")
    row["ARI"] = clustering.get("ARI")

    ap25_vals: list[float] = []
    ap50_vals: list[float] = []
    oracle_results = oracle.get("oracle_results", {}) or {}
    for bucket_name, metrics in oracle_results.items():
        bucket = _bucket_slug(bucket_name)
        if bucket is None or not isinstance(metrics, dict):
            continue
        ap25 = metrics.get("AP25")
        ap50 = metrics.get("AP50")
        count = metrics.get("Count")
        row[f"AP25_{bucket}"] = ap25
        row[f"AP50_{bucket}"] = ap50
        row[f"Count_{bucket}"] = count
        if ap25 is not None:
            ap25_vals.append(float(ap25))
        if ap50 is not None:
            ap50_vals.append(float(ap50))
    row["macro_AP25"] = _mean(ap25_vals)
    row["macro_AP50"] = _mean(ap50_vals)

    map_vals: list[float] = []
    additional = oracle.get("additional_metrics", {}) or {}
    map_by_bucket = additional.get("oracle_mAP_25_95_by_bucket", {}) or {}
    for bucket_name, value in map_by_bucket.items():
        bucket = _bucket_slug(bucket_name)
        if bucket is None:
            continue
        row[f"mAP_25_95_{bucket}"] = value
        if value is not None:
            map_vals.append(float(value))
    row["macro_mAP_25_95"] = _mean(map_vals)
    return row


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "mean_seen_points_fraction",
        "macro_AP25",
        "macro_AP50",
        "macro_mAP_25_95",
        "NMI",
        "ARI",
        "AP25_small",
        "AP25_medium",
        "AP25_large",
        "AP50_small",
        "AP50_medium",
        "AP50_large",
        "mAP_25_95_small",
        "mAP_25_95_medium",
        "mAP_25_95_large",
    ]
    return {
        f"mean_{key}": _mean([row.get(key) for row in rows])
        for key in metric_keys
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patched ScanNet++ CHORUS validation metrics.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="nvs_sem_val")
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--granularities", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--hdbscan-max-samples", type=int, default=200000)
    parser.add_argument("--hdbscan-seed", type=int, default=0)
    parser.add_argument("--eval-benchmark", type=str, default="top100_instance")
    parser.add_argument("--seen-target", type=float, default=0.80)
    parser.add_argument("--overwrite-teacher", action="store_true")
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = resolve_scannetpp_dataset_root(args.dataset_root)
    granularities = [float(g.strip()) for g in args.granularities.split(",") if g.strip()]

    scene_ids = read_scannetpp_scene_ids(
        dataset_root=dataset_root,
        scene_list_file=args.scene_list_file,
        split=args.split if args.scene_list_file is None else None,
        max_scenes=args.max_scenes,
    )
    if not scene_ids:
        raise RuntimeError("No ScanNet++ scenes selected.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=3,
        overwrite=args.overwrite_teacher,
    )

    for idx, scene_id in enumerate(scene_ids, start=1):
        scene_dir = dataset_root / "data" / scene_id
        print("=" * 90, flush=True)
        print(f"[{idx}/{len(scene_ids)}] scene={scene_id}", flush=True)
        adapter = ScanNetPPSceneAdapter(scene_dir, eval_benchmark=args.eval_benchmark)

        with phase_timer(f"Prepare scene {scene_id}"):
            adapter.prepare()

        with phase_timer(f"Teacher scene {scene_id}"):
            teacher_outputs = run_teacher_stage(
                adapter=adapter,
                teacher=teacher,
                granularities=granularities,
                frame_skip=args.frame_skip,
            )

        cluster_outputs = []
        for teacher_output in teacher_outputs:
            with phase_timer(f"Project+Cluster scene {scene_id} g={teacher_output.granularity}"):
                cluster_outputs.append(
                    run_project_cluster_stage(
                        adapter=adapter,
                        teacher_output=teacher_output,
                        frame_skip=args.frame_skip,
                        svd_components=args.svd_components,
                        min_cluster_size=args.min_cluster_size,
                        min_samples=args.min_samples,
                        cluster_selection_epsilon=args.cluster_selection_epsilon,
                        save_outputs=True,
                        hdbscan_max_samples=args.hdbscan_max_samples,
                        hdbscan_subsample_seed=args.hdbscan_seed,
                    )
                )

        scene_intrinsic_metrics = compute_scene_intrinsic_metrics(cluster_outputs)
        evaluation_hooks = adapter.get_evaluation_hooks()
        with phase_timer(f"Oracle eval scene {scene_id}"):
            eval_summary = evaluation_hooks.evaluate_scene(adapter=adapter, cluster_outputs=cluster_outputs)

        scene_summary: dict[str, Any] = {
            "dataset": adapter.dataset_name,
            "scene_id": scene_id,
            "status": "done",
            "granularities": granularities,
            "frame_skip": int(args.frame_skip),
            "hdbscan_max_samples": int(args.hdbscan_max_samples),
            "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
            "teacher_outputs": [
                {
                    "granularity": float(t.granularity),
                    "num_mask_files": len(t.frame_mask_paths),
                    "total_masks": int(t.total_masks),
                    "output_dir": str(t.frame_mask_paths[0].parent) if t.frame_mask_paths else None,
                }
                for t in teacher_outputs
            ],
            "cluster_outputs": [
                {
                    "granularity": float(c.granularity),
                    "labels_path": str(c.labels_path) if c.labels_path is not None else None,
                    "ply_path": str(c.ply_path) if c.ply_path is not None else None,
                    "stats": c.stats,
                }
                for c in cluster_outputs
            ],
            "scene_intrinsic_metrics": scene_intrinsic_metrics,
        }
        if eval_summary is not None:
            scene_summary.update(eval_summary)

        scene_summary_path = scene_dir / "scannetpp_validation_metrics_summary.json"
        save_json(scene_summary, scene_summary_path)
        scene_summary["summary_path"] = str(scene_summary_path)
        scene_summaries.append(scene_summary)

        row = _flatten_metrics(scene_summary)
        rows.append(row)
        seen = row.get("mean_seen_points_fraction")
        print(
            f"scene={scene_id} mean_seen={seen:.4f} target={args.seen_target:.2f} "
            f"AP25={row.get('macro_AP25')} AP50={row.get('macro_AP50')} "
            f"mAP={row.get('macro_mAP_25_95')} NMI={row.get('NMI')} ARI={row.get('ARI')}",
            flush=True,
        )
        if seen is not None and float(seen) < float(args.seen_target):
            print(f"WARNING: scene={scene_id} mean seen fraction below target", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = args.report_dir / "scannetpp_validation_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "config": config | {
            "dataset_root": str(dataset_root),
            "report_dir": str(args.report_dir),
            "granularities": granularities,
        },
        "scene_ids": scene_ids,
        "rows": rows,
        "aggregate": _aggregate_rows(rows),
        "scene_summaries": scene_summaries,
        "csv_path": str(csv_path),
    }
    report_path = args.report_dir / "scannetpp_validation_metrics.json"
    save_json(report, report_path)
    print("=" * 90, flush=True)
    print(f"Saved metrics JSON: {report_path}", flush=True)
    print(f"Saved metrics CSV: {csv_path}", flush=True)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
