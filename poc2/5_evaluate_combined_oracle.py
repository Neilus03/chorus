import json
import os
import importlib
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# --- CONFIG ---
SCENE_DIR = os.environ.get("SCENE_DIR", "scene0000_00")
GRANULARITIES = [0.2, 0.5, 0.8]
ORACLE_MIN_IOU_FOR_PLY = float(os.environ.get("ORACLE_MIN_IOU_FOR_PLY", "0.1"))

def load_all_proposals(scene_dir, granularities, return_sources: bool = False):
    """Loads and merges all clusters into a single pool with unique IDs."""
    all_proposals = []
    proposal_sources = []
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
            proposal_sources.append(float(g))

        if len(unique_instances) > 0:
            current_max_id = shifted_labels.max()

    if return_sources:
        return all_proposals, proposal_sources
    return all_proposals


def _build_size_buckets(gt_ids: np.ndarray):
    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]
    gt_areas = {g: int(np.sum(gt_ids == g)) for g in gt_instances}
    gt_areas_list = list(gt_areas.values())
    if len(gt_areas_list) == 0:
        return {}, gt_areas

    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.67)
    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float("inf")),
    }
    return size_buckets, gt_areas


def _best_iou_and_best_source_per_gt(
    gt_ids: np.ndarray, proposals: list[np.ndarray], proposal_sources: list[float]
):
    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]
    gt_areas = {g: int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    best_iou_by_gt = {}
    best_source_by_gt = {}

    for g_id in gt_instances:
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_source = None

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue
            union = prop_areas[i] + gt_areas[g_id] - intersection
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_source = proposal_sources[i]

        best_iou_by_gt[int(g_id)] = float(best_iou)
        best_source_by_gt[int(g_id)] = best_source

    return best_iou_by_gt, best_source_by_gt


def compute_additional_oracle_metrics(
    gt_ids: np.ndarray, proposals: list[np.ndarray], proposal_sources: list[float]
):
    size_buckets, gt_areas = _build_size_buckets(gt_ids)
    if not size_buckets:
        return {}

    best_iou_by_gt, best_source_by_gt = _best_iou_and_best_source_per_gt(
        gt_ids, proposals, proposal_sources
    )
    thresholds = np.arange(0.25, 1.0, 0.05)

    # 1) COCO-like oracle mAP over IoU thresholds [0.25:0.95].
    map_by_bucket = {}
    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts = [g for g, area in gt_areas.items() if min_pts <= area < max_pts]
        if len(valid_gts) == 0:
            map_by_bucket[bucket_name] = 0.0
            continue
        recalls = []
        for th in thresholds:
            matched = sum(best_iou_by_gt[g] >= float(th) for g in valid_gts)
            recalls.append(matched / len(valid_gts))
        map_by_bucket[bucket_name] = float(np.mean(recalls))

    # 2) Top-k proposal coverage: GTs with at least k proposals above IoU threshold.
    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]
    gt_areas_full = {g: int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    topk_coverage = {}
    for th in (0.25, 0.50):
        counts = {1: 0, 3: 0, 5: 0}
        for g_id in gt_instances:
            g_mask = gt_ids == g_id
            n_above = 0
            for i, p_mask in enumerate(proposals):
                intersection = int(np.sum(p_mask & g_mask))
                if intersection == 0:
                    continue
                union = prop_areas[i] + gt_areas_full[g_id] - intersection
                iou = intersection / max(union, 1)
                if iou >= th:
                    n_above += 1
            for k in counts:
                if n_above >= k:
                    counts[k] += 1

        denom = max(len(gt_instances), 1)
        topk_coverage[f"iou_{th:.2f}"] = {
            "R_at_least_1": counts[1] / denom,
            "R_at_least_3": counts[3] / denom,
            "R_at_least_5": counts[5] / denom,
        }

    # 3) Granularity winner distribution (which granularity wins per GT).
    winner_counts = {f"g{g}": 0 for g in GRANULARITIES}
    no_match = 0
    for g_id in gt_instances:
        src = best_source_by_gt[int(g_id)]
        if src is None:
            no_match += 1
            continue
        winner_counts[f"g{src}"] += 1

    total_gt = max(len(gt_instances), 1)
    winner_share = {k: v / total_gt for k, v in winner_counts.items()}
    winner_share["no_match"] = no_match / total_gt

    return {
        "oracle_mAP_25_95_by_bucket": map_by_bucket,
        "topk_proposal_coverage": topk_coverage,
        "winner_granularity_share": winner_share,
    }

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


def build_oracle_best_labels(
    gt_ids: np.ndarray, proposals: list[np.ndarray], min_iou: float = ORACLE_MIN_IOU_FOR_PLY
) -> np.ndarray:
    """
    Build a pooled-oracle label map:
    for each GT instance, pick the single best proposal from the merged pool.
    """
    oracle_labels = np.full(gt_ids.shape, -1, dtype=np.int32)
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas = {g: int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    for g_id in gt_instances:
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_idx = -1

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue
            union = prop_areas[i] + gt_areas[g_id] - intersection
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= min_iou:
            oracle_labels[proposals[best_idx]] = int(g_id)

    return oracle_labels


def save_oracle_best_ply(scene_dir: Path, scene_name: str, oracle_labels: np.ndarray) -> str:
    mesh_path = scene_dir / f"{scene_name}_vh_clean_2.ply"
    plydata = PlyData.read(str(mesh_path))
    vertices = plydata["vertex"]

    max_id = int(max(0, np.max(oracle_labels)))
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(max_id + 1, 3), dtype=np.uint8)

    # Grey for unmatched/noise points.
    display_ids = np.where(oracle_labels >= 0, oracle_labels, 0)
    vertex_colors = colors[display_ids]
    vertex_colors[oracle_labels < 0] = np.array([80, 80, 80], dtype=np.uint8)

    out_vertices = np.empty(
        len(vertices),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    out_vertices["x"] = vertices["x"]
    out_vertices["y"] = vertices["y"]
    out_vertices["z"] = vertices["z"]
    out_vertices["red"] = vertex_colors[:, 0]
    out_vertices["green"] = vertex_colors[:, 1]
    out_vertices["blue"] = vertex_colors[:, 2]

    out_path = scene_dir / "chorus_oracle_best_combined.ply"
    PlyData([PlyElement.describe(out_vertices, "vertex")]).write(str(out_path))
    return str(out_path)

def main():
    # Dynamically import the loader from your existing level script
    eval_level = importlib.import_module("4_evaluate_instances_by_level")

    scene_dir = Path(SCENE_DIR)
    scene_name = scene_dir.name

    print(f"--- COMBINED EVALUATION: {scene_name} ---")

    gt_ids = eval_level.load_gt_instance_ids(scene_dir, scene_name)
    proposals, proposal_sources = load_all_proposals(
        SCENE_DIR, GRANULARITIES, return_sources=True
    )

    oracle_results = evaluate_oracle_ap(gt_ids, proposals)
    additional_metrics = compute_additional_oracle_metrics(
        gt_ids, proposals, proposal_sources
    )
    oracle_best_labels = build_oracle_best_labels(gt_ids, proposals, ORACLE_MIN_IOU_FOR_PLY)

    print("\n" + "="*65)
    print(f"{'Size Bucket (Tertiles)':<25} | {'GT Count':<8} | {'Oracle AP@25':<12} | {'Oracle AP@50':<12}")
    print("-" * 65)
    for bucket, metrics in oracle_results.items():
        print(f"{bucket:<25} | {metrics['Count']:<8} | {metrics['AP25']:<12.4f} | {metrics['AP50']:<12.4f}")
    print("="*65)
    if additional_metrics:
        print("\nAdditional metrics:")
        for bucket, m in additional_metrics["oracle_mAP_25_95_by_bucket"].items():
            print(f"  mAP@[.25:.95] {bucket}: {m:.4f}")
        cov25 = additional_metrics["topk_proposal_coverage"]["iou_0.25"]
        cov50 = additional_metrics["topk_proposal_coverage"]["iou_0.50"]
        print(
            "  Top-k coverage @IoU0.25: "
            f"R>=1={cov25['R_at_least_1']:.4f}, "
            f"R>=3={cov25['R_at_least_3']:.4f}, "
            f"R>=5={cov25['R_at_least_5']:.4f}"
        )
        print(
            "  Top-k coverage @IoU0.50: "
            f"R>=1={cov50['R_at_least_1']:.4f}, "
            f"R>=3={cov50['R_at_least_3']:.4f}, "
            f"R>=5={cov50['R_at_least_5']:.4f}"
        )
        print(
            "  Winner granularity share: "
            + ", ".join(
                f"{k}={v:.3f}"
                for k, v in additional_metrics["winner_granularity_share"].items()
            )
        )
    print("Interpretation: This represents the maximum potential of the Teacher pool.")
    print("The Student will learn to pick the best expert for each scale.\n")

    results_to_save = dict(oracle_results)
    results_to_save["_extras"] = additional_metrics
    with open(os.path.join(scene_dir, "oracle_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_to_save, f)

    labels_path = scene_dir / "chorus_oracle_best_combined_labels.npy"
    np.save(labels_path, oracle_best_labels)
    ply_path = save_oracle_best_ply(scene_dir, scene_name, oracle_best_labels)
    print(f"Saved oracle pooled labels: {labels_path}")
    print(f"Saved oracle pooled PLY: {ply_path}")

if __name__ == "__main__":
    main()