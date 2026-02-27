from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from chorus.common.types import ClusterOutput
from chorus.datasets.base import SceneAdapter


def build_proposals_from_cluster_outputs(
    cluster_outputs: list[ClusterOutput],
) -> tuple[list[np.ndarray], list[float]]:
    proposals: list[np.ndarray] = []
    proposal_sources: list[float] = []

    for cluster_output in cluster_outputs:
        labels = cluster_output.labels
        unique_instances = np.unique(labels)
        unique_instances = unique_instances[unique_instances >= 0]

        for inst_id in unique_instances:
            proposals.append(labels == inst_id)
            proposal_sources.append(float(cluster_output.granularity))

    return proposals, proposal_sources


def _build_size_buckets(gt_ids: np.ndarray):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]

    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
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
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    proposal_sources: list[float],
):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]

    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    best_iou_by_gt = {}
    best_source_by_gt = {}

    for g_id in gt_instances:
        g_id_int = int(g_id)
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_source = None

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue

            union = prop_areas[i] + gt_areas[g_id_int] - intersection
            iou = intersection / max(union, 1)

            if iou > best_iou:
                best_iou = iou
                best_source = proposal_sources[i]

        best_iou_by_gt[g_id_int] = float(best_iou)
        best_source_by_gt[g_id_int] = best_source

    return best_iou_by_gt, best_source_by_gt


def compute_additional_oracle_metrics(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    proposal_sources: list[float],
):
    size_buckets, gt_areas = _build_size_buckets(gt_ids)
    if not size_buckets:
        return {}

    best_iou_by_gt, best_source_by_gt = _best_iou_and_best_source_per_gt(
        gt_ids,
        proposals,
        proposal_sources,
    )

    thresholds = np.arange(0.25, 1.0, 0.05)

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

    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas_full = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    topk_coverage = {}
    for th in (0.25, 0.50):
        counts = {1: 0, 3: 0, 5: 0}

        for g_id in gt_instances:
            g_id_int = int(g_id)
            g_mask = gt_ids == g_id
            n_above = 0

            for i, p_mask in enumerate(proposals):
                intersection = int(np.sum(p_mask & g_mask))
                if intersection == 0:
                    continue

                union = prop_areas[i] + gt_areas_full[g_id_int] - intersection
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

    unique_granularities = sorted({float(s) for s in proposal_sources})
    winner_counts = {f"g{g}": 0 for g in unique_granularities}
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


def evaluate_oracle_ap(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    thresholds=(0.25, 0.50),
):
    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}

    gt_areas_list = list(gt_areas.values())
    if len(gt_areas_list) == 0:
        return {}

    p33 = np.percentile(gt_areas_list, 33.33)
    p66 = np.percentile(gt_areas_list, 66.67)
    size_buckets = {
        f"Small (<{p33:.0f} pts)": (0, p33),
        f"Medium ({p33:.0f}-{p66:.0f} pts)": (p33, p66),
        f"Large (>{p66:.0f} pts)": (p66, float("inf")),
    }

    results = {}
    prop_areas = [int(np.sum(p)) for p in proposals]

    print(f"Evaluating {len(proposals)} pooled proposals against {len(gt_instances)} GT instances")

    for bucket_name, (min_pts, max_pts) in size_buckets.items():
        valid_gts = [g for g, area in gt_areas.items() if min_pts <= area < max_pts]
        num_gt = len(valid_gts)

        results[bucket_name] = {"AP25": 0.0, "AP50": 0.0, "Count": num_gt}
        if num_gt == 0:
            continue

        for th in thresholds:
            matched_this_th = 0
            for g_id in valid_gts:
                g_mask = gt_ids == g_id
                best_iou = 0.0

                for i, p_mask in enumerate(proposals):
                    intersection = int(np.sum(p_mask & g_mask))
                    if intersection == 0:
                        continue

                    union = prop_areas[i] + gt_areas[g_id] - intersection
                    iou = intersection / max(union, 1)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= th:
                    matched_this_th += 1

            if th == 0.25:
                results[bucket_name]["AP25"] = matched_this_th / num_gt
            elif th == 0.50:
                results[bucket_name]["AP50"] = matched_this_th / num_gt

    return results


def build_oracle_best_labels(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
    min_iou: float = 0.1,
) -> np.ndarray:
    oracle_labels = np.full(gt_ids.shape, -1, dtype=np.int32)

    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]
    gt_areas = {int(g): int(np.sum(gt_ids == g)) for g in gt_instances}
    prop_areas = [int(np.sum(p)) for p in proposals]

    for g_id in gt_instances:
        g_id_int = int(g_id)
        g_mask = gt_ids == g_id
        best_iou = 0.0
        best_idx = -1

        for i, p_mask in enumerate(proposals):
            intersection = int(np.sum(p_mask & g_mask))
            if intersection == 0:
                continue

            union = prop_areas[i] + gt_areas[g_id_int] - intersection
            iou = intersection / max(union, 1)

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= min_iou:
            oracle_labels[proposals[best_idx]] = g_id_int

    return oracle_labels


def save_oracle_best_ply(
    geometry_path: Path,
    out_path: Path,
    oracle_labels: np.ndarray,
) -> None:
    plydata = PlyData.read(str(geometry_path))
    vertices = plydata["vertex"]

    max_id = int(max(0, np.max(oracle_labels)))
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(max_id + 1, 3), dtype=np.uint8)

    display_ids = np.where(oracle_labels >= 0, oracle_labels, 0)
    vertex_colors = colors[display_ids]
    vertex_colors[oracle_labels < 0] = np.array([80, 80, 80], dtype=np.uint8)

    out_vertices = np.empty(
        len(vertices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    out_vertices["x"] = vertices["x"]
    out_vertices["y"] = vertices["y"]
    out_vertices["z"] = vertices["z"]
    out_vertices["red"] = vertex_colors[:, 0]
    out_vertices["green"] = vertex_colors[:, 1]
    out_vertices["blue"] = vertex_colors[:, 2]

    PlyData([PlyElement.describe(out_vertices, "vertex")]).write(str(out_path))


def evaluate_and_save_scannet_oracle(
    adapter: SceneAdapter,
    cluster_outputs: list[ClusterOutput],
    min_iou_for_ply: float = 0.1,
) -> dict:
    gt_ids = adapter.load_gt_instance_ids()
    if gt_ids is None:
        raise RuntimeError("ScanNet oracle evaluation requires GT instance ids.")

    proposals, proposal_sources = build_proposals_from_cluster_outputs(cluster_outputs)
    if len(proposals) == 0:
        raise RuntimeError("No proposals available for oracle evaluation.")

    oracle_results = evaluate_oracle_ap(gt_ids, proposals)
    additional_metrics = compute_additional_oracle_metrics(
        gt_ids,
        proposals,
        proposal_sources,
    )
    oracle_best_labels = build_oracle_best_labels(
        gt_ids,
        proposals,
        min_iou=min_iou_for_ply,
    )

    scene_root = adapter.scene_root
    metrics_path = scene_root / "oracle_metrics.json"
    labels_path = scene_root / "chorus_oracle_best_combined_labels.npy"
    ply_path = scene_root / "chorus_oracle_best_combined.ply"

    results_to_save = dict(oracle_results)
    results_to_save["_extras"] = additional_metrics

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)

    np.save(labels_path, oracle_best_labels)

    geometry_record = adapter.get_geometry_record()
    save_oracle_best_ply(
        geometry_path=geometry_record.geometry_path,
        out_path=ply_path,
        oracle_labels=oracle_best_labels,
    )

    print(f"Saved oracle metrics: {metrics_path}")
    print(f"Saved oracle pooled labels: {labels_path}")
    print(f"Saved oracle pooled ply: {ply_path}")

    return {
        "metrics_path": metrics_path,
        "labels_path": labels_path,
        "ply_path": ply_path,
        "oracle_results": oracle_results,
        "additional_metrics": additional_metrics,
    }
