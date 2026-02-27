import os
import subprocess
import json
import sys
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

# Adjust this to point exactly where your folders are!
BASE_SCANS_DIR = "/scratch2/nedela/chorus_poc/scans"

SCENES = [
    "scene0000_00", "scene0140_00", "scene0263_00",
    "scene0340_01", "scene0381_02", "scene0396_01",
    "scene0399_01", "scene0420_02", "scene0654_00", "scene0662_01"
]

GRANULARITIES = ["0.2", "0.5", "0.8"]
TEACHER_FRAME_SKIP = 10


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


def _run_pipeline():
    all_scene_metrics = []

    print(f"\n{'='*70}")
    print(f"CHORUS PoC2: Multi-Scene Evaluation Pipeline")
    print(f"{'='*70}")
    print(f"Base Directory: {BASE_SCANS_DIR}")
    print(f"Total Scenes: {len(SCENES)}")
    print(f"Granularities: {', '.join(GRANULARITIES)}")
    print(f"{'='*70}\n")

    for scene_idx, scene in enumerate(SCENES, 1):
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
                print(f"    Executing: 1_run_unsam_teacher.py")
                subprocess.run(["python", "1_run_unsam_teacher.py"], check=True)
                print(f"    ✓ Teacher inference complete for g={g}")

            print(f"    Executing: 2_bridge_and_cluster.py")
            subprocess.run(["python", "2_bridge_and_cluster.py"], check=True)
            print(f"    ✓ Clustering complete for g={g}")

        # 2. Run the Combined Pool Evaluation
        metrics_file = os.path.join(scene_path, "oracle_metrics.json")
        print(f"\n--- Step 2: Running Combined Oracle Evaluation for {scene} ---")
        if os.path.exists(metrics_file):
            print("    ✓ Skipping oracle evaluation: oracle_metrics.json already exists")
        else:
            print(f"    Executing: 5_evaluate_combined_oracle.py")
            subprocess.run(["python", "5_evaluate_combined_oracle.py"], check=True)
            print(f"    ✓ Oracle evaluation complete")

        # 3. Read the saved JSON results
        print(f"\n--- Step 3: Loading metrics from {metrics_file} ---")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                scene_metrics = json.load(f)
                all_scene_metrics.append(scene_metrics)
                print(f"    ✓ Metrics loaded successfully")
                # Print quick summary
                for bucket, metrics in scene_metrics.items():
                    print(f"      {bucket}: AP@25={metrics['AP25']:.4f}, AP@50={metrics['AP50']:.4f}")
        else:
            print(f"    ✗ ERROR: No metrics file found for {scene}")

        print(f"\n{'='*70}")
        print(f"COMPLETED SCENE {scene_idx}/{len(SCENES)}: {scene}")
        print(f"{'='*70}\n")

    # Calculate Grand Averages
    print("\n\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL SCENES")
    print("="*70)
    print(f"Total scenes processed: {len(all_scene_metrics)}")

    # Initialize aggregators
    buckets = ["Small (<318 pts)", "Medium (318-899 pts)", "Large (>899 pts)"]
    avg_results = {b: {"AP25": [], "AP50": []} for b in buckets}

    print("\nCollecting metrics by size bucket...")
    for scene_idx, scene_data in enumerate(all_scene_metrics, 1):
        # Note: Bucket names might vary slightly by scene due to dynamic tertiles,
        # so we aggregate by order (0=Small, 1=Medium, 2=Large)
        bucket_keys = list(scene_data.keys())
        print(f"  Scene {scene_idx}: {len(bucket_keys)} buckets found")
        if len(bucket_keys) == 3:
            for i, key in enumerate(bucket_keys):
                avg_results[buckets[i]]["AP25"].append(scene_data[key]["AP25"])
                avg_results[buckets[i]]["AP50"].append(scene_data[key]["AP50"])

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
    print("\n✓ Pipeline complete! You can now send this table directly to Yuanwen!")
    print(f"  Total scenes evaluated: {len(all_scene_metrics)}")
    print(f"  Granularities tested: {', '.join(GRANULARITIES)}")
    print("="*70 + "\n")

def main():
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


if __name__ == "__main__":
    main()