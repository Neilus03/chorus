import json
import os
import importlib
from pathlib import Path

import numpy as np
from plyfile import PlyData

# --- CONFIG ---
SCENE_DIR = os.environ.get("SCENE_DIR", "scene0000_00")
GRANULARITIES = [0.2, 0.5, 0.8]

def load_all_proposals(scene_dir, granularities):
    """Loads and merges all clusters into a single pool with unique IDs."""
    all_proposals = []
    current_max_id = 0

    for g in granularities:
        path = os.path.join(scene_dir, f"chorus_instance_labels_g{g}.npy")
        if not os.path.exists(path):
            print(f"  > Warning: {path} not found. Skipping.")
            continue

        labels = np.load(path)
        # Shift IDs so a cluster '5' in g=0.2 is distinct from cluster '5' in g=0.8
        unique_labels = np.unique(labels)
        unique_instances = unique_labels[unique_labels >= 0]

        # Mapping for this specific level to global pool IDs
        shifted_labels = np.where(labels >= 0, labels + current_max_id + 1, -1)

        # We store them as a list of masks to handle the overlap in the pool
        for inst_id in np.unique(shifted_labels):
            if inst_id == -1: continue
            all_proposals.append(shifted_labels == inst_id)

        if len(unique_instances) > 0:
            current_max_id = shifted_labels.max()

    return all_proposals

def evaluate_oracle_ap(gt_ids, proposals, thresholds=(0.25, 0.50)):
    """
    Evaluates the 'Oracle' performance.
    For each GT object, we find the best IoU available in the entire proposal pool.
    """
    eval_script = importlib.import_module("4_evaluate_instances_by_level")

    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]
    gt_areas = {g: np.sum(gt_ids == g) for g in gt_instances}

    # Dynamic Tertiles (Equal Bins) logic from your previous requirement
    gt_areas_list = list(gt_areas.values())

    if len(gt_areas_list) == 0:
        return {}

    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.67)

    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float('inf'))
    }

    results = {}
    print(f"Evaluating pool of {len(proposals)} proposals against {len(gt_instances)} GT objects...")

    # Pre-calculate areas for all proposals
    prop_areas = [np.sum(p) for p in proposals]

    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts = [g for g, area in gt_areas.items() if min_pts <= area < max_pts]
        num_gt = len(valid_gts)
        results[bucket_name] = {"AP25": 0.0, "AP50": 0.0, "Count": num_gt}

        if num_gt == 0: continue

        for th in thresholds:
            matched_this_th = 0
            for g_id in valid_gts:
                g_mask = (gt_ids == g_id)
                best_iou = 0

                # Check every proposal in the pool for this GT
                for i, p_mask in enumerate(proposals):
                    intersection = np.sum(p_mask & g_mask)
                    if intersection == 0: continue

                    union = prop_areas[i] + gt_areas[g_id] - intersection
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= th:
                    matched_this_th += 1

            # For Oracle, AP is simply the recall (as we assume we pick the right one)
            results[bucket_name]["AP25" if th==0.25 else "AP50"] = matched_this_th / num_gt

    return results

def main():
    # Dynamically import the loader from your existing level script
    eval_level = importlib.import_module("4_evaluate_instances_by_level")

    scene_dir = Path(SCENE_DIR)
    scene_name = scene_dir.name

    print(f"--- COMBINED EVALUATION: {scene_name} ---")

    gt_ids = eval_level.load_gt_instance_ids(scene_dir, scene_name)
    proposals = load_all_proposals(SCENE_DIR, GRANULARITIES)

    oracle_results = evaluate_oracle_ap(gt_ids, proposals)

    print("\n" + "="*65)
    print(f"{'Size Bucket (Tertiles)':<25} | {'GT Count':<8} | {'Oracle AP@25':<12} | {'Oracle AP@50':<12}")
    print("-" * 65)
    for bucket, metrics in oracle_results.items():
        print(f"{bucket:<25} | {metrics['Count']:<8} | {metrics['AP25']:<12.4f} | {metrics['AP50']:<12.4f}")
    print("="*65)
    print("Interpretation: This represents the maximum potential of the Teacher pool.")
    print("The Student will learn to pick the best expert for each scale.\n")

    with open(os.path.join(scene_dir, "oracle_metrics.json"), "w") as f:
        json.dump(oracle_results, f)

if __name__ == "__main__":
    main()