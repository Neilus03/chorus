import os
import subprocess
import json
import sys
import time
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from pathlib import Path

# Adjust this to point exactly where your folders are!
BASE_SCANS_DIR = "/scratch2/nedela/chorus_poc/scans"

SCENES = [
    "scene0000_00", "scene0140_00", "scene0263_00",
    "scene0340_01", "scene0381_02", "scene0396_01",
    "scene0399_01", "scene0420_02", "scene0654_00", "scene0662_01"
]

GRANULARITIES = ["0.2", "0.5", "0.8"]
TEACHER_FRAME_SKIP = 10

LITEPT_PYTHON = os.environ.get(
    "LITEPT_PYTHON", "/scratch2/nedela/litept-env/bin/python"
)
RAPIDS_PYTHON = os.environ.get(
    "RAPIDS_PYTHON", "/scratch2/nedela/venvs/chorus-rapids-env/bin/python"
)
USE_RAPIDS_FOR_CLUSTER = os.environ.get("USE_RAPIDS_FOR_CLUSTER", "1").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


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


def _resolve_python(path: str) -> str:
    # Keep venv interpreter paths intact; do not resolve symlinks to system python.
    return str(Path(path).expanduser().absolute())


def _run_script(script_name: str, python_exec: str, extra_env: dict | None = None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"    Executing: {script_name} with {python_exec}")
    subprocess.run([python_exec, script_name], check=True, env=env)


def _cluster_python() -> str:
    rapids_python = _resolve_python(RAPIDS_PYTHON)
    litept_python = _resolve_python(LITEPT_PYTHON)
    if not USE_RAPIDS_FOR_CLUSTER:
        return litept_python
    if Path(rapids_python).exists():
        return rapids_python
    print(
        "    ! RAPIDS_PYTHON not found. Falling back to litept python for clustering."
    )
    return litept_python


def _expected_teacher_mask_count(scene_path: str, frame_skip: int = TEACHER_FRAME_SKIP):
    color_dir = os.path.join(scene_path, "color")
    if not os.path.isdir(color_dir):
        return None

    frames = [name for name in os.listdir(color_dir) if name.endswith(".jpg")]
    # Match teacher logic: numeric sort then subsample with [::FRAME_SKIP].
    frames_sorted = sorted(frames, key=lambda name: int(os.path.splitext(name)[0]))
    return len(frames_sorted[::frame_skip])


def _has_teacher_outputs(scene_path: str, granularity: str) -> bool:
    masks_dir = os.path.join(scene_path, f"unsam_masks_g{granularity}")
    if not os.path.isdir(masks_dir):
        return False

    expected_count = _expected_teacher_mask_count(scene_path)
    mask_files = [name for name in os.listdir(masks_dir) if name.endswith(".npy")]

    # If we cannot infer expected frame count, fall back to "any mask exists".
    if expected_count is None:
        return len(mask_files) > 0
    return len(mask_files) >= expected_count


def _has_cluster_outputs(scene_path: str, granularity: str) -> bool:
    labels_file = os.path.join(scene_path, f"chorus_instance_labels_g{granularity}.npy")
    mesh_file = os.path.join(scene_path, f"chorus_instance_result_g{granularity}.ply")
    return os.path.exists(labels_file) and os.path.exists(mesh_file)


def _has_oracle_outputs(scene_path: str) -> bool:
    metrics_file = os.path.join(scene_path, "oracle_metrics.json")
    pooled_labels_file = os.path.join(scene_path, "chorus_oracle_best_combined_labels.npy")
    pooled_ply_file = os.path.join(scene_path, "chorus_oracle_best_combined.ply")
    return (
        os.path.exists(metrics_file)
        and os.path.exists(pooled_labels_file)
        and os.path.exists(pooled_ply_file)
    )


def _extract_bucket_metrics(scene_data: dict) -> list[tuple[str, dict]]:
    bucket_entries = []
    for key, value in scene_data.items():
        if (
            isinstance(value, dict)
            and "AP25" in value
            and "AP50" in value
            and "Count" in value
        ):
            bucket_entries.append((key, value))
    return bucket_entries


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def _run_pipeline():
    pipeline_start_time = time.time()
    all_scene_metrics = []
    litept_python = _resolve_python(LITEPT_PYTHON)
    cluster_python = _cluster_python()

    print(f"\n{'='*70}")
    print(f"CHORUS PoC2: Multi-Scene Evaluation Pipeline")
    print(f"{'='*70}")
    print(f"Base Directory: {BASE_SCANS_DIR}")
    print(f"Total Scenes: {len(SCENES)}")
    print(f"Granularities: {', '.join(GRANULARITIES)}")
    print(f"Teacher/Eval Python: {litept_python}")
    print(f"Cluster Python: {cluster_python}")
    print(f"Cluster backend default: {'gpu_hdbscan' if cluster_python != litept_python else 'cpu_hdbscan'}")
    print(f"{'='*70}\n")

    for scene_idx, scene in enumerate(SCENES, 1):
        scene_start_time = time.time()
        scene_path = os.path.join(BASE_SCANS_DIR, scene)
        print(f"\n{'='*70}")
        print(f"PROCESSING SCENE {scene_idx}/{len(SCENES)}: {scene}")
        print(f"{'='*70}")
        print(f"Scene Path: {scene_path}")

        # Set the environment variable for the child scripts
        os.environ["SCENE_DIR"] = scene_path

        # 1. Generate masks and clusters for all 3 granularities
        for g_idx, g in enumerate(GRANULARITIES, 1):
            os.environ["GRANULARITY"] = g
            print(f"\n--- Step 1.{g_idx}: Running Teacher & Clustering for g={g} ---")
            if _has_cluster_outputs(scene_path, g):
                print(f"    ✓ Skipping g={g}: clustering outputs already exist")
                continue

            if _has_teacher_outputs(scene_path, g):
                print(f"    ✓ Skipping teacher for g={g}: mask outputs already exist")
            else:
                expected = _expected_teacher_mask_count(scene_path)
                if expected is not None:
                    masks_dir = os.path.join(scene_path, f"unsam_masks_g{g}")
                    found = 0
                    if os.path.isdir(masks_dir):
                        found = len([n for n in os.listdir(masks_dir) if n.endswith(".npy")])
                    print(
                        f"    Incomplete masks for g={g}: found {found}/{expected}. "
                        "Re-running teacher."
                    )
                _run_script("1_run_unsam_teacher.py", litept_python)
                print(f"    ✓ Teacher inference complete for g={g}")

            cluster_env = {}
            if cluster_python != litept_python:
                # Use GPU HDBSCAN automatically when clustering with RAPIDS interpreter.
                cluster_env["CLUSTER_BACKEND"] = os.environ.get(
                    "CLUSTER_BACKEND", "gpu_hdbscan"
                )
            _run_script("2_bridge_and_cluster.py", cluster_python, extra_env=cluster_env)
            print(f"    ✓ Clustering complete for g={g}")

        # 2. Run the Combined Pool Evaluation
        metrics_file = os.path.join(scene_path, "oracle_metrics.json")
        print(f"\n--- Step 2: Running Combined Oracle Evaluation for {scene} ---")
        if _has_oracle_outputs(scene_path):
            print(
                "    ✓ Skipping oracle evaluation: metrics + pooled oracle outputs already exist"
            )
        else:
            _run_script("5_evaluate_combined_oracle.py", litept_python)
            print(f"    ✓ Oracle evaluation complete")

        # 3. Read the saved JSON results
        print(f"\n--- Step 3: Loading metrics from {metrics_file} ---")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                scene_metrics = json.load(f)
                all_scene_metrics.append(scene_metrics)
                print(f"    ✓ Metrics loaded successfully")
                # Print quick summary
                for bucket, metrics in _extract_bucket_metrics(scene_metrics):
                    print(f"      {bucket}: AP@25={metrics['AP25']:.4f}, AP@50={metrics['AP50']:.4f}")
                extras = scene_metrics.get("_extras", {})
                if extras:
                    m_by_bucket = extras.get("oracle_mAP_25_95_by_bucket", {})
                    if m_by_bucket:
                        for bucket, val in m_by_bucket.items():
                            print(f"      {bucket}: mAP@[.25:.95]={val:.4f}")
                    cov25 = extras.get("topk_proposal_coverage", {}).get("iou_0.25", {})
                    cov50 = extras.get("topk_proposal_coverage", {}).get("iou_0.50", {})
                    if cov25:
                        print(
                            "      Top-k cov @0.25: "
                            f"R>=1={cov25.get('R_at_least_1', 0.0):.4f}, "
                            f"R>=3={cov25.get('R_at_least_3', 0.0):.4f}, "
                            f"R>=5={cov25.get('R_at_least_5', 0.0):.4f}"
                        )
                    if cov50:
                        print(
                            "      Top-k cov @0.50: "
                            f"R>=1={cov50.get('R_at_least_1', 0.0):.4f}, "
                            f"R>=3={cov50.get('R_at_least_3', 0.0):.4f}, "
                            f"R>=5={cov50.get('R_at_least_5', 0.0):.4f}"
                        )
        else:
            print(f"    ✗ ERROR: No metrics file found for {scene}")

        scene_elapsed = time.time() - scene_start_time
        print(f"\n{'='*70}")
        print(f"COMPLETED SCENE {scene_idx}/{len(SCENES)}: {scene}")
        print(f"Scene processing time: {_format_duration(scene_elapsed)}")
        print(f"{'='*70}\n")

    # Calculate Grand Averages
    print("\n\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL SCENES")
    print("="*70)
    print(f"Total scenes processed: {len(all_scene_metrics)}")

    # Initialize aggregators
    buckets = ["Small (<318 pts)", "Medium (318-899 pts)", "Large (>899 pts)"]
    avg_results = {b: {"AP25": [], "AP50": []} for b in buckets}
    avg_map_25_95 = {b: [] for b in buckets}
    avg_topk = {
        "iou_0.25": {"R_at_least_1": [], "R_at_least_3": [], "R_at_least_5": []},
        "iou_0.50": {"R_at_least_1": [], "R_at_least_3": [], "R_at_least_5": []},
    }
    avg_winner_share = {f"g{g}": [] for g in GRANULARITIES}
    avg_winner_share["no_match"] = []

    print("\nCollecting metrics by size bucket...")
    for scene_idx, scene_data in enumerate(all_scene_metrics, 1):
        # Note: Bucket names might vary slightly by scene due to dynamic tertiles,
        # so we aggregate by order (0=Small, 1=Medium, 2=Large)
        bucket_entries = _extract_bucket_metrics(scene_data)
        print(f"  Scene {scene_idx}: {len(bucket_entries)} buckets found")
        if len(bucket_entries) == 3:
            for i, (key, metrics) in enumerate(bucket_entries):
                avg_results[buckets[i]]["AP25"].append(scene_data[key]["AP25"])
                avg_results[buckets[i]]["AP50"].append(scene_data[key]["AP50"])
        extras = scene_data.get("_extras", {})
        map_by_bucket = extras.get("oracle_mAP_25_95_by_bucket", {})
        for i, (bucket_name, val) in enumerate(map_by_bucket.items()):
            if i < len(buckets):
                avg_map_25_95[buckets[i]].append(val)

        for iou_key in ("iou_0.25", "iou_0.50"):
            cov = extras.get("topk_proposal_coverage", {}).get(iou_key, {})
            for k in ("R_at_least_1", "R_at_least_3", "R_at_least_5"):
                if k in cov:
                    avg_topk[iou_key][k].append(cov[k])

        winner = extras.get("winner_granularity_share", {})
        for key in avg_winner_share:
            if key in winner:
                avg_winner_share[key].append(winner[key])

    print("\n" + "="*70)
    print("FINAL RESULTS ACROSS 10 SCENES (ScanNet Protocol: No Walls/Floors)")
    print("="*70)
    print(f"{'Size Bucket':<25} | {'Avg AP@25':<12} | {'Avg AP@50':<12}")
    print("-" * 55)

    global_ap25 = []
    global_ap50 = []

    for bucket in buckets:
        ap25_values = avg_results[bucket]["AP25"]
        ap50_values = avg_results[bucket]["AP50"]
        ap25 = np.mean(ap25_values)
        ap50 = np.mean(ap50_values)
        global_ap25.append(ap25)
        global_ap50.append(ap50)
        print(f"{bucket:<25} | {ap25:<12.4f} | {ap50:<12.4f}")
        print(f"  (n={len(ap25_values)} scenes)")

    print("-" * 55)
    print(f"{'GLOBAL AVERAGE':<25} | {np.mean(global_ap25):<12.4f} | {np.mean(global_ap50):<12.4f}")
    print("="*70)

    print("\n" + "="*70)
    print("ADDITIONAL ORACLE METRICS (AVERAGED ACROSS SCENES)")
    print("="*70)
    for bucket in buckets:
        vals = avg_map_25_95[bucket]
        if vals:
            print(f"{bucket:<25} | mAP@[.25:.95]={np.mean(vals):.4f} (n={len(vals)})")
    for iou_key in ("iou_0.25", "iou_0.50"):
        d = avg_topk[iou_key]
        if d["R_at_least_1"]:
            print(
                f"Top-k coverage @{iou_key.split('_')[1]}: "
                f"R>=1={np.mean(d['R_at_least_1']):.4f}, "
                f"R>=3={np.mean(d['R_at_least_3']):.4f}, "
                f"R>=5={np.mean(d['R_at_least_5']):.4f}"
            )
    winner_parts = []
    for key, vals in avg_winner_share.items():
        if vals:
            winner_parts.append(f"{key}={np.mean(vals):.3f}")
    if winner_parts:
        print("Winner granularity share: " + ", ".join(winner_parts))
    print("="*70)

    pipeline_elapsed = time.time() - pipeline_start_time
    print("\n✓ Pipeline complete!")
    print(f"  Total scenes evaluated: {len(all_scene_metrics)}")
    print(f"  Granularities tested: {', '.join(GRANULARITIES)}")
    print(f"  Total execution time: {_format_duration(pipeline_elapsed)}")
    print("="*70 + "\n")

def main():
    main_start_time = time.time()
    report_dir = os.path.join(BASE_SCANS_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"run_10_scenes_{timestamp}.txt")

    # Mirror terminal output into a persistent run report.
    with open(report_path, "w", encoding="utf-8") as report_file:
        tee_out = Tee(sys.stdout, report_file)
        tee_err = Tee(sys.stderr, report_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Writing run report to: {report_path}")
            _run_pipeline()
            print(f"Report saved to: {report_path}")

    main_elapsed = time.time() - main_start_time
    print(f"\nTotal runtime (including report writing): {_format_duration(main_elapsed)}")


if __name__ == "__main__":
    main()