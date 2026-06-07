from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


def instance_sizes(labels: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(labels, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) >= 0}


def relabel_contiguous(labels: np.ndarray, *, min_points: int = 1) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    out = np.full(labels.shape, -1, dtype=np.int32)
    next_id = 0
    for label_id, count in instance_sizes(labels).items():
        if count < int(min_points):
            continue
        out[labels == label_id] = next_id
        next_id += 1
    return out


def full_instance_bootstrap_support(
    full_labels: np.ndarray,
    bootstrap_labels: np.ndarray,
    full_id: int,
    full_size: int,
) -> dict[str, Any]:
    full_mask = full_labels == int(full_id)
    overlapping_boot = bootstrap_labels[full_mask]
    overlapping_boot = overlapping_boot[overlapping_boot >= 0]
    if overlapping_boot.size == 0:
        return {
            "best_bootstrap_id": None,
            "intersection": 0,
            "iou": 0.0,
            "full_support": 0.0,
            "bootstrap_containment": 0.0,
            "bootstrap_points": 0,
        }
    ids, counts = np.unique(overlapping_boot, return_counts=True)
    best_idx = int(np.argmax(counts))
    boot_id = int(ids[best_idx])
    intersection = int(counts[best_idx])
    boot_size = int(np.count_nonzero(bootstrap_labels == boot_id))
    union = int(full_size + boot_size - intersection)
    return {
        "best_bootstrap_id": boot_id,
        "intersection": intersection,
        "iou": float(intersection / max(union, 1)),
        "full_support": float(intersection / max(full_size, 1)),
        "bootstrap_containment": float(intersection / max(boot_size, 1)),
        "bootstrap_points": boot_size,
    }


def prune_by_bootstrap_agreement(
    full_labels: np.ndarray,
    bootstrap_labels: np.ndarray,
    *,
    min_points: int,
    iou_threshold: float,
    full_support_threshold: float,
    bootstrap_containment_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    keep_ids: set[int] = set()
    rows: list[dict[str, Any]] = []
    for full_id, full_size in instance_sizes(full_labels).items():
        support = full_instance_bootstrap_support(full_labels, bootstrap_labels, full_id, full_size)
        keep = (
            support["iou"] >= float(iou_threshold)
            or (
                support["full_support"] >= float(full_support_threshold)
                and support["bootstrap_containment"] >= float(bootstrap_containment_threshold)
            )
        )
        if full_size < int(min_points):
            keep = False
        if keep:
            keep_ids.add(full_id)
        rows.append(
            {
                "full_id": int(full_id),
                "full_points": int(full_size),
                "kept": bool(keep),
                **support,
            }
        )

    out = np.full(full_labels.shape, -1, dtype=np.int32)
    for full_id in keep_ids:
        out[full_labels == full_id] = int(full_id)
    out = relabel_contiguous(out, min_points=min_points)
    return out, {
        "gate": "bootstrap_agreement",
        "iou_threshold": float(iou_threshold),
        "full_support_threshold": float(full_support_threshold),
        "bootstrap_containment_threshold": float(bootstrap_containment_threshold),
        "num_input_instances": int(len(rows)),
        "num_kept_instances": int(len(keep_ids)),
        "num_dropped_instances": int(len(rows) - len(keep_ids)),
        "labeled_fraction": float(np.count_nonzero(out >= 0) / max(out.shape[0], 1)),
        "instances": rows,
    }


def best_cross_granularity_support(
    labels_by_g: dict[float, np.ndarray],
    granularity: float,
    full_id: int,
    full_size: int,
) -> dict[str, Any]:
    source = labels_by_g[float(granularity)]
    full_mask = source == int(full_id)
    best = {
        "best_other_granularity": None,
        "best_other_id": None,
        "intersection": 0,
        "iou": 0.0,
        "full_support": 0.0,
        "other_containment": 0.0,
        "other_points": 0,
    }
    for other_g, other_labels in labels_by_g.items():
        if float(other_g) == float(granularity):
            continue
        overlapping_other = other_labels[full_mask]
        overlapping_other = overlapping_other[overlapping_other >= 0]
        if overlapping_other.size == 0:
            continue
        ids, counts = np.unique(overlapping_other, return_counts=True)
        best_idx = int(np.argmax(counts))
        other_id = int(ids[best_idx])
        intersection = int(counts[best_idx])
        other_size = int(np.count_nonzero(other_labels == other_id))
        union = int(full_size + other_size - intersection)
        iou = float(intersection / max(union, 1))
        if iou > float(best["iou"]) or intersection > int(best["intersection"]):
            best = {
                "best_other_granularity": float(other_g),
                "best_other_id": other_id,
                "intersection": intersection,
                "iou": iou,
                "full_support": float(intersection / max(full_size, 1)),
                "other_containment": float(intersection / max(other_size, 1)),
                "other_points": other_size,
            }
    return best


def prune_by_multigranularity_agreement(
    labels_by_g: dict[float, np.ndarray],
    granularity: float,
    *,
    min_points: int,
    iou_threshold: float,
    full_support_threshold: float,
    other_containment_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = labels_by_g[float(granularity)]
    keep_ids: set[int] = set()
    rows: list[dict[str, Any]] = []
    for full_id, full_size in instance_sizes(labels).items():
        support = best_cross_granularity_support(labels_by_g, granularity, full_id, full_size)
        keep = (
            support["iou"] >= float(iou_threshold)
            or (
                support["full_support"] >= float(full_support_threshold)
                and support["other_containment"] >= float(other_containment_threshold)
            )
        )
        if full_size < int(min_points):
            keep = False
        if keep:
            keep_ids.add(full_id)
        rows.append(
            {
                "full_id": int(full_id),
                "full_points": int(full_size),
                "kept": bool(keep),
                **support,
            }
        )

    out = np.full(labels.shape, -1, dtype=np.int32)
    for full_id in keep_ids:
        out[labels == full_id] = int(full_id)
    out = relabel_contiguous(out, min_points=min_points)
    return out, {
        "gate": "multigranularity_agreement",
        "iou_threshold": float(iou_threshold),
        "full_support_threshold": float(full_support_threshold),
        "other_containment_threshold": float(other_containment_threshold),
        "num_input_instances": int(len(rows)),
        "num_kept_instances": int(len(keep_ids)),
        "num_dropped_instances": int(len(rows) - len(keep_ids)),
        "labeled_fraction": float(np.count_nonzero(out >= 0) / max(out.shape[0], 1)),
        "instances": rows,
    }


def intersect_kept_full_labels(
    full_labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    min_points: int,
) -> np.ndarray:
    keep_mask = (first >= 0) & (second >= 0)
    out = np.where(keep_mask, full_labels, -1).astype(np.int32, copy=False)
    return relabel_contiguous(out, min_points=min_points)


def connected_component_cleanup(
    labels: np.ndarray,
    mesh_path: Path,
    *,
    min_component_points: int,
    min_component_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    ply = PlyData.read(str(mesh_path))
    if "face" not in ply:
        return labels.copy(), {"num_components_removed": 0, "removed_points": 0, "reason": "no_faces"}

    adjacency: dict[int, set[int]] = defaultdict(set)
    for face in ply["face"].data["vertex_indices"]:
        verts = [int(v) for v in face]
        for i, a in enumerate(verts):
            for b in verts[i + 1 :]:
                if labels[a] >= 0 and labels[a] == labels[b]:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    cleaned = np.full(labels.shape, -1, dtype=np.int32)
    removed_points = 0
    removed_components = 0
    kept_components = 0
    component_rows: list[dict[str, Any]] = []

    for label_id, label_size in instance_sizes(labels).items():
        vertices = np.flatnonzero(labels == label_id)
        remaining = set(int(v) for v in vertices)
        components: list[list[int]] = []
        while remaining:
            start = remaining.pop()
            comp = [start]
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nxt in adjacency.get(cur, ()):
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.append(nxt)
                        queue.append(nxt)
            components.append(comp)

        if not components:
            removed_points += int(label_size)
            removed_components += 1
            continue

        largest = max(len(c) for c in components)
        keep_threshold = max(int(min_component_points), int(math.ceil(label_size * min_component_fraction)))
        for comp in components:
            keep = len(comp) == largest or len(comp) >= keep_threshold
            if keep:
                cleaned[np.asarray(comp, dtype=np.int64)] = int(label_id)
                kept_components += 1
            else:
                removed_points += len(comp)
                removed_components += 1
            component_rows.append(
                {
                    "label_id": int(label_id),
                    "label_points": int(label_size),
                    "component_points": int(len(comp)),
                    "largest_component_points": int(largest),
                    "kept": bool(keep),
                }
            )

    cleaned = relabel_contiguous(cleaned, min_points=min_component_points)
    return cleaned, {
        "gate": "connected_component_cleanup",
        "min_component_points": int(min_component_points),
        "min_component_fraction": float(min_component_fraction),
        "num_components_kept": int(kept_components),
        "num_components_removed": int(removed_components),
        "removed_points": int(removed_points),
        "labeled_fraction": float(np.count_nonzero(cleaned >= 0) / max(cleaned.shape[0], 1)),
        "components": component_rows,
    }


def proposal_counterbalance_metrics(
    labels_list: list[np.ndarray],
    gt: np.ndarray,
    thresholds: tuple[float, ...] = (0.25, 0.50),
) -> dict[str, float | int]:
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    gt_ids, gt_counts = np.unique(gt[gt > 0], return_counts=True)
    gt_size = {int(i): int(c) for i, c in zip(gt_ids, gt_counts)}
    num_gt = len(gt_size)

    proposal_best_ious: list[float] = []
    proposal_sizes: list[int] = []
    gt_best_ious = {int(g): 0.0 for g in gt_size}
    gt_good_counts = {float(t): {int(g): 0 for g in gt_size} for t in thresholds}

    for labels in labels_list:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        pred_ids, pred_counts = np.unique(labels[labels >= 0], return_counts=True)
        pred_size = {int(i): int(c) for i, c in zip(pred_ids, pred_counts)}
        if not pred_size:
            continue

        valid = (labels >= 0) & (gt > 0)
        pair_counts = {}
        if np.any(valid):
            pairs = np.stack([labels[valid], gt[valid]], axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            pair_counts = {
                (int(pred_id), int(gt_id)): int(count)
                for (pred_id, gt_id), count in zip(unique_pairs, counts)
            }

        pred_best = {int(pid): 0.0 for pid in pred_size}
        for (pred_id, gt_id), inter in pair_counts.items():
            union = pred_size[pred_id] + gt_size[gt_id] - inter
            iou = float(inter / max(union, 1))
            pred_best[pred_id] = max(pred_best[pred_id], iou)
            gt_best_ious[gt_id] = max(gt_best_ious[gt_id], iou)
            for threshold in thresholds:
                if iou >= float(threshold):
                    gt_good_counts[float(threshold)][gt_id] += 1

        for pred_id, size in pred_size.items():
            proposal_best_ious.append(float(pred_best[pred_id]))
            proposal_sizes.append(int(size))

    out: dict[str, float | int] = {
        "num_gt": int(num_gt),
        "num_proposals": int(len(proposal_best_ious)),
        "mean_proposal_size": float(np.mean(proposal_sizes)) if proposal_sizes else 0.0,
        "median_proposal_size": float(np.median(proposal_sizes)) if proposal_sizes else 0.0,
        "mean_best_iou_per_proposal": float(np.mean(proposal_best_ious)) if proposal_best_ious else 0.0,
        "mean_best_iou_per_gt": float(np.mean(list(gt_best_ious.values()))) if gt_best_ious else 0.0,
    }

    for threshold in thresholds:
        key = int(round(float(threshold) * 100))
        precision = (
            float(np.mean(np.asarray(proposal_best_ious) >= float(threshold)))
            if proposal_best_ious
            else 0.0
        )
        recall = (
            float(np.mean(np.asarray(list(gt_best_ious.values())) >= float(threshold)))
            if gt_best_ious
            else 0.0
        )
        f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
        good_counts = list(gt_good_counts[float(threshold)].values())
        matched_counts = [c for c in good_counts if c > 0]
        out[f"proposal_precision@{key}"] = precision
        out[f"gt_recall@{key}"] = recall
        out[f"proposal_f1@{key}"] = f1
        out[f"avg_good_proposals_per_gt@{key}"] = float(np.mean(good_counts)) if good_counts else 0.0
        out[f"avg_good_proposals_per_matched_gt@{key}"] = (
            float(np.mean(matched_counts)) if matched_counts else 0.0
        )
        out[f"max_good_proposals_for_one_gt@{key}"] = int(max(good_counts)) if good_counts else 0

    return out
