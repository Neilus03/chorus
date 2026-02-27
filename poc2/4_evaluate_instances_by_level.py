import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# --- CONFIG ---
SCENE_DIR = "scene0000_00"
GRANULARITY = 0.8

PRED_PATH = os.path.join(SCENE_DIR, f"chorus_instance_labels_g{GRANULARITY}.npy")

print(f"Evaluating instances by level for granularity {GRANULARITY}")

def _load_instance_ids_from_ply(labels_ply: Path) -> np.ndarray | None:
    plydata = PlyData.read(str(labels_ply))
    vertex_data = plydata.elements[0].data
    for field in ("objectId", "instance", "instance_id"):
        if field in vertex_data.dtype.names:
            return np.asarray(vertex_data[field]).astype(np.int64)
    return None


def _load_instance_ids_from_aggregation(scene_dir: Path, scene_name: str, n_vertices: int) -> np.ndarray | None:
    seg_paths = [
        scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_name}_vh_clean.segs.json",
    ]
    agg_paths = [
        scene_dir / f"{scene_name}.aggregation.json",
        scene_dir / f"{scene_name}_vh_clean.aggregation.json",
    ]

    seg_path = next((p for p in seg_paths if p.exists()), None)
    agg_path = next((p for p in agg_paths if p.exists()), None)
    if seg_path is None or agg_path is None:
        return None

    with open(seg_path, "r", encoding="utf-8") as f:
        seg_json = json.load(f)
    with open(agg_path, "r", encoding="utf-8") as f:
        agg_json = json.load(f)

    seg_indices = np.asarray(seg_json.get("segIndices", []), dtype=np.int64)
    if seg_indices.shape[0] != n_vertices:
        raise RuntimeError(f"segIndices length ({seg_indices.shape[0]}) != num vertices ({n_vertices})")

    seg_to_instance = {}

    # --- Explicitly ignore "stuff" classes per ScanNet protocol ---
    ignore_classes = {"wall", "floor", "ceiling"}

    for group in agg_json.get("segGroups", []):
        label = group.get("label", "").lower()
        if label in ignore_classes:
            continue  # Skip this, it will remain 0 (background)

        inst_id = int(group.get("objectId", group.get("id", -1)))
        if inst_id < 0:
            continue
        for seg_id in group.get("segments", []):
            seg_to_instance[int(seg_id)] = inst_id

    gt_instance_ids = np.zeros(n_vertices, dtype=np.int64)
    for i, seg_id in enumerate(seg_indices):
        gt_instance_ids[i] = seg_to_instance.get(int(seg_id), 0)

    return gt_instance_ids


def load_gt_instance_ids(scene_dir: Path, scene_name: str) -> np.ndarray:
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    plydata = PlyData.read(str(labels_ply))
    n_vertices = len(plydata.elements[0].data)

    # Force the script to use the aggregation JSON so we can filter by text label!
    gt_instance_ids = _load_instance_ids_from_aggregation(scene_dir, scene_name, n_vertices)
    if gt_instance_ids is not None:
        return gt_instance_ids

    raise RuntimeError("Could not find GT instance ids in aggregation+segments files.")


def evaluate_class_agnostic_ap_and_miou(gt_ids: np.ndarray, pred_ids: np.ndarray, thresholds=(0.10, 0.25, 0.50)):
    """
    Computes Global Class-Agnostic Average Precision (AP) and mean Intersection-over-Union (mIoU).
    """
    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]
    num_gt = len(gt_instances)

    pred_unique, pred_counts = np.unique(pred_ids, return_counts=True)
    valid_pred_mask = pred_unique >= 0
    pred_instances = pred_unique[valid_pred_mask]
    pred_sizes = pred_counts[valid_pred_mask]
    num_pred = len(pred_instances)

    gt_areas = {g: np.sum(gt_ids == g) for g in gt_instances}
    pred_areas = {p: s for p, s in zip(pred_instances, pred_sizes)}

    metrics = {"mIoU": 0.0, "AP10": 0.0, "AP25": 0.0, "AP50": 0.0}

    if num_gt == 0 or num_pred == 0:
        return metrics

    # --- mIoU Calculation ---
    gt_max_ious = []
    for g_id in gt_instances:
        g_mask = (gt_ids == g_id)
        overlapping_preds, overlap_counts = np.unique(pred_ids[g_mask], return_counts=True)

        max_iou = 0.0
        for p_val, intersect_area in zip(overlapping_preds, overlap_counts):
            if p_val < 0:
                continue
            union_area = gt_areas[g_id] + pred_areas[p_val] - intersect_area
            iou = intersect_area / union_area
            if iou > max_iou:
                max_iou = iou
        gt_max_ious.append(max_iou)

    metrics["mIoU"] = float(np.mean(gt_max_ious))

    # --- AP Calculation ---
    sort_idx = np.argsort(-pred_sizes)
    sorted_pred_instances = pred_instances[sort_idx]

    for th in thresholds:
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        gt_matched = set()

        for i, p_id in enumerate(sorted_pred_instances):
            p_mask = (pred_ids == p_id)
            overlapping_gts, overlap_counts = np.unique(gt_ids[p_mask], return_counts=True)

            max_iou = 0.0
            best_gt = -1

            for gt_val, intersect_area in zip(overlapping_gts, overlap_counts):
                if gt_val <= 0:
                    continue
                union_area = pred_areas[p_id] + gt_areas[gt_val] - intersect_area
                iou = intersect_area / union_area
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_val

            if max_iou >= th and best_gt not in gt_matched and best_gt != -1:
                tp[i] = 1.0
                gt_matched.add(best_gt)
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / num_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

        if th == 0.10:
            metrics["AP10"] = ap
        elif th == 0.25:
            metrics["AP25"] = ap
        elif th == 0.50:
            metrics["AP50"] = ap

    return metrics


def evaluate_ap_by_size(gt_ids: np.ndarray, pred_ids: np.ndarray, thresholds=(0.25, 0.50)):
    """
    Computes AP25 and AP50 categorized strictly by Ground Truth object size.
    """
    gt_unique = np.unique(gt_ids)
    gt_instances = gt_unique[gt_unique > 0]

    pred_unique, pred_counts = np.unique(pred_ids, return_counts=True)
    valid_pred_mask = pred_unique >= 0
    pred_instances = pred_unique[valid_pred_mask]
    pred_sizes = pred_counts[valid_pred_mask]

    gt_areas = {g: np.sum(gt_ids == g) for g in gt_instances}
    pred_areas = {p: s for p, s in zip(pred_instances, pred_sizes)}

    # ---------------------------------------------------------
    # The Size Buckets (Adjust these thresholds if necessary)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # The Dynamic Size Buckets (Equal Quantiles / Tertiles)
    # ---------------------------------------------------------
    gt_areas_list = list(gt_areas.values())

    if len(gt_areas_list) == 0:
        return {}

    # Calculate the exact 33.3% and 66.7% percentiles of object sizes in THIS scene
    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.66)

    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float('inf'))
    }

    results = {}
    sort_idx = np.argsort(-pred_sizes)
    sorted_pred_instances = pred_instances[sort_idx]

    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts_for_bucket = {g for g, area in gt_areas.items() if min_pts <= area < max_pts}
        num_gt_in_bucket = len(valid_gts_for_bucket)

        results[bucket_name] = {"AP25": 0.0, "AP50": 0.0, "Count": num_gt_in_bucket}

        if num_gt_in_bucket == 0:
            continue

        for th in thresholds:
            tp = []
            fp = []
            gt_matched = set()

            for p_id in sorted_pred_instances:
                p_mask = (pred_ids == p_id)
                overlapping_gts, overlap_counts = np.unique(gt_ids[p_mask], return_counts=True)

                max_iou = 0.0
                best_gt = -1

                for gt_val, intersect_area in zip(overlapping_gts, overlap_counts):
                    if gt_val <= 0:
                        continue
                    union_area = pred_areas[p_id] + gt_areas[gt_val] - intersect_area
                    iou = intersect_area / union_area
                    if iou > max_iou:
                        max_iou = iou
                        best_gt = gt_val

                if max_iou >= th and best_gt != -1:
                    if best_gt in valid_gts_for_bucket:
                        if best_gt not in gt_matched:
                            tp.append(1.0)
                            fp.append(0.0)
                            gt_matched.add(best_gt)
                        else:
                            tp.append(0.0)
                            fp.append(1.0)
                    else:
                        # IGNORE predictions that correctly hit objects outside this bucket
                        pass
                else:
                    tp.append(0.0)
                    fp.append(1.0)

            if len(tp) == 0:
                continue

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)

            recalls = cum_tp / num_gt_in_bucket
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)

            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))

            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

            if th == 0.25:
                results[bucket_name]["AP25"] = ap
            elif th == 0.50:
                results[bucket_name]["AP50"] = ap

    return results


def main() -> None:
    scene_dir = Path(SCENE_DIR)
    scene_name = scene_dir.name

    print("Loading ground-truth instance IDs...")
    gt_instance_ids = load_gt_instance_ids(scene_dir, scene_name)

    print("Loading CHORUS instance pseudo-labels...")
    pred_instance_ids = np.load(PRED_PATH)

    if pred_instance_ids.shape[0] != gt_instance_ids.shape[0]:
        raise RuntimeError(
            f"Prediction length ({pred_instance_ids.shape[0]}) != GT length ({gt_instance_ids.shape[0]})"
        )

    valid_gt_mask = gt_instance_ids > 0
    gt_valid = gt_instance_ids[valid_gt_mask]
    pred_valid = pred_instance_ids[valid_gt_mask]

    print("\n" + "="*50)
    print("--- 1. STRUCTURAL GEOMETRY (GLOBAL) ---")
    ari_score = adjusted_rand_score(gt_valid, pred_valid)
    nmi_score = normalized_mutual_info_score(gt_valid, pred_valid)

    print(f"Adjusted Rand Index (ARI)       : {ari_score:.4f}")
    print(f"Normalized Mutual Info (NMI)    : {nmi_score:.4f}")

    noise_pts = np.sum(pred_instance_ids == -1)
    print(f"Percentage of noise points      : {noise_pts / len(pred_instance_ids) * 100:.2f}%")

    print("\n" + "="*50)
    print("--- 2. OBJECT DETECTION METRICS (GLOBAL) ---")
    global_metrics = evaluate_class_agnostic_ap_and_miou(gt_instance_ids, pred_instance_ids)
    print(f"mIoU                            : {global_metrics['mIoU']:.4f}")
    print(f"AP@10                           : {global_metrics['AP10']:.4f}")
    print(f"AP@25                           : {global_metrics['AP25']:.4f}")
    print(f"AP@50                           : {global_metrics['AP50']:.4f}")

    print("\n" + "="*50)
    print(f"--- 3. SIZE-AWARE AVERAGE PRECISION (g={GRANULARITY}) ---")
    size_metrics = evaluate_ap_by_size(gt_instance_ids, pred_instance_ids)

    print(f"{'Size Bucket':<18} | {'GT Count':<8} | {'AP@25':<8} | {'AP@50':<8}")
    print("-" * 50)
    for bucket, metrics in size_metrics.items():
        print(f"{bucket:<18} | {metrics['Count']:<8} | {metrics['AP25']:<8.4f} | {metrics['AP50']:<8.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()