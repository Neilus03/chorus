"""End-of-training evaluation for the student model.

Computes AP25/AP50, NMI, ARI against:
  1. Real ScanNet ground truth (if available)
  2. CHORUS pseudo-labels (from training pack)

Reuses evaluation logic from the CHORUS codebase.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from student.data.target_builder import InstanceTargets

log = logging.getLogger(__name__)

_CHORUS_ROOT = Path(__file__).resolve().parents[3] / "chorus"


def _ensure_chorus_importable() -> None:
    chorus_str = str(_CHORUS_ROOT)
    if chorus_str not in sys.path:
        sys.path.insert(0, chorus_str)


# ── extract proposals from model output ──────────────────────────────────


def extract_proposals(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    *,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_points: int = 30,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Convert decoder output into a list of binary proposal masks.

    Returns
    -------
    proposals : list[np.ndarray]
        Each element is a boolean (N,) mask for one predicted instance.
    scores : np.ndarray
        Confidence score (sigmoid) for each kept proposal.
    query_indices : np.ndarray
        Which query slot each proposal came from.
    """
    scores_sig = score_logits.sigmoid().cpu().numpy()
    masks_binary = (mask_logits.sigmoid() >= mask_threshold).cpu().numpy()

    proposals: list[np.ndarray] = []
    scores_out: list[float] = []
    indices_out: list[int] = []

    for q in range(mask_logits.shape[0]):
        if scores_sig[q] < score_threshold:
            continue
        mask = masks_binary[q]
        if mask.sum() < min_points:
            continue
        proposals.append(mask)
        scores_out.append(float(scores_sig[q]))
        indices_out.append(q)

    return proposals, np.array(scores_out), np.array(indices_out, dtype=np.int64)


def extract_class_aware_proposals(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    *,
    class_score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    min_points: int = 30,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Convert class/no-object logits into class-aware instance predictions."""
    probs = class_logits.softmax(dim=-1).cpu().numpy()
    fg_probs = probs[:, :-1]
    class_ids = fg_probs.argmax(axis=1).astype(np.int64)
    scores = fg_probs.max(axis=1)
    masks_binary = (mask_logits.sigmoid() >= mask_threshold).cpu().numpy()

    proposals: list[np.ndarray] = []
    scores_out: list[float] = []
    class_ids_out: list[int] = []
    indices_out: list[int] = []
    for q in range(mask_logits.shape[0]):
        if scores[q] < class_score_threshold:
            continue
        mask = masks_binary[q]
        if mask.sum() < min_points:
            continue
        proposals.append(mask)
        scores_out.append(float(scores[q]))
        class_ids_out.append(int(class_ids[q]))
        indices_out.append(q)

    return (
        proposals,
        np.array(scores_out, dtype=np.float32),
        np.array(class_ids_out, dtype=np.int64),
        np.array(indices_out, dtype=np.int64),
    )


# ── build per-point label array from proposals ──────────────────────────


def proposals_to_labels(
    proposals: list[np.ndarray],
    num_points: int,
) -> np.ndarray:
    """Assign each point to the proposal that claims it (last writer wins).

    Returns (N,) int32 with -1 for unclaimed points, 1..M for instances.
    """
    labels = np.full(num_points, -1, dtype=np.int32)
    for i, mask in enumerate(proposals):
        labels[mask] = i + 1
    return labels


# ── AP evaluation (reuses CHORUS logic) ──────────────────────────────────


def _evaluate_ap_against_gt(
    gt_ids: np.ndarray,
    proposals: list[np.ndarray],
) -> dict[str, Any]:
    """Compute AP25/AP50 by size bucket + overall, using CHORUS oracle eval."""
    _ensure_chorus_importable()
    from chorus.eval.scannet_oracle import evaluate_oracle_ap

    bucket_results = evaluate_oracle_ap(gt_ids, proposals, thresholds=(0.25, 0.50))

    all_ap25 = []
    all_ap50 = []
    total_gt = 0
    for bucket_name, vals in bucket_results.items():
        count = vals.get("Count", 0)
        total_gt += count
        if count > 0:
            all_ap25.append(vals["AP25"] * count)
            all_ap50.append(vals["AP50"] * count)

    overall_ap25 = sum(all_ap25) / max(total_gt, 1)
    overall_ap50 = sum(all_ap50) / max(total_gt, 1)

    return {
        "AP25": overall_ap25,
        "AP50": overall_ap50,
        "total_gt_instances": total_gt,
        "by_bucket": bucket_results,
    }


def _instance_masks_and_classes(
    gt_ids: np.ndarray,
    instance_class_ids: dict[int, int] | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    gt_masks: list[np.ndarray] = []
    gt_classes: list[int] = []
    for inst_id in sorted(int(x) for x in np.unique(gt_ids) if int(x) > 0):
        if instance_class_ids is None or inst_id not in instance_class_ids:
            continue
        gt_masks.append(gt_ids == inst_id)
        gt_classes.append(int(instance_class_ids[inst_id]))
    return gt_masks, np.asarray(gt_classes, dtype=np.int64)


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_class_aware_ap(
    gt_ids: np.ndarray,
    gt_instance_class_ids: dict[int, int],
    proposals: list[np.ndarray],
    scores: np.ndarray,
    pred_class_ids: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.25, 0.50),
) -> dict[str, Any]:
    """Class-aware AP with confidence sorting and one-to-one GT matching."""
    gt_masks, gt_classes = _instance_masks_and_classes(gt_ids, gt_instance_class_ids)
    total_gt = int(len(gt_masks))
    if total_gt == 0:
        return {
            "AP25": 0.0,
            "AP50": 0.0,
            "total_gt_instances": 0,
            "num_predictions": len(proposals),
            "by_threshold": {},
        }

    gt_by_class: dict[int, list[int]] = {}
    for idx, cls in enumerate(gt_classes.tolist()):
        gt_by_class.setdefault(int(cls), []).append(idx)

    pred_order = np.argsort(-scores) if scores.size else np.zeros(0, dtype=np.int64)
    by_threshold: dict[str, Any] = {}
    for thr in thresholds:
        ap_values: list[float] = []
        for cls, cls_gt_indices in gt_by_class.items():
            cls_pred_indices = [
                int(i) for i in pred_order
                if int(pred_class_ids[i]) == cls
            ]
            n_gt_cls = len(cls_gt_indices)
            if n_gt_cls == 0:
                continue
            matched_gt: set[int] = set()
            tp: list[float] = []
            fp: list[float] = []
            for pred_idx in cls_pred_indices:
                pred_mask = proposals[pred_idx].astype(bool, copy=False)
                best_iou = 0.0
                best_gt = -1
                for gt_idx in cls_gt_indices:
                    gt_mask = gt_masks[gt_idx]
                    inter = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    iou = float(inter / union) if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx
                if best_iou >= thr and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)

            if not tp:
                ap_values.append(0.0)
                continue
            tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
            fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
            recalls = tp_cum / max(n_gt_cls, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap_values.append(_voc_ap(recalls, precisions))

        key = f"AP{int(round(thr * 100))}"
        by_threshold[key] = float(np.mean(ap_values)) if ap_values else 0.0

    return {
        "AP25": by_threshold.get("AP25", 0.0),
        "AP50": by_threshold.get("AP50", 0.0),
        "total_gt_instances": total_gt,
        "num_predictions": len(proposals),
        "by_threshold": by_threshold,
    }


# ── semantic mIoU from class-aware instance predictions ─────────────────


def _semantic_labels_from_instances(
    gt_ids: np.ndarray,
    instance_class_ids: dict[int, int],
) -> np.ndarray:
    semantic = np.full(gt_ids.shape, -1, dtype=np.int64)
    for inst_id, class_id in instance_class_ids.items():
        semantic[gt_ids == int(inst_id)] = int(class_id)
    return semantic


def _semantic_labels_from_class_proposals(
    proposals: list[np.ndarray],
    scores: np.ndarray,
    pred_class_ids: np.ndarray,
    num_points: int,
) -> np.ndarray:
    semantic = np.full(num_points, -1, dtype=np.int64)
    best_scores = np.full(num_points, -np.inf, dtype=np.float32)
    for mask, score, class_id in zip(proposals, scores, pred_class_ids):
        mask_bool = mask.astype(bool, copy=False)
        update = mask_bool & (float(score) >= best_scores)
        semantic[update] = int(class_id)
        best_scores[update] = float(score)
    return semantic


def compute_semantic_miou(
    gt_semantic: np.ndarray,
    pred_semantic: np.ndarray,
    num_classes: int,
) -> float:
    """Mean class IoU over valid GT points for class-aware predictions."""
    valid = gt_semantic >= 0
    if not np.any(valid):
        return float("nan")

    ious: list[float] = []
    for class_id in range(int(num_classes)):
        gt_mask = valid & (gt_semantic == class_id)
        pred_mask = valid & (pred_semantic == class_id)
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(gt_mask, pred_mask).sum()
        ious.append(float(inter / union))
    return float(np.mean(ious)) if ious else float("nan")


def evaluate_class_aware_semantic_miou(
    gt_ids: np.ndarray,
    gt_instance_class_ids: dict[int, int],
    proposals: list[np.ndarray],
    scores: np.ndarray,
    pred_class_ids: np.ndarray,
    *,
    num_classes: int,
) -> float:
    gt_semantic = _semantic_labels_from_instances(gt_ids, gt_instance_class_ids)
    pred_semantic = _semantic_labels_from_class_proposals(
        proposals,
        scores,
        pred_class_ids,
        num_points=gt_ids.shape[0],
    )
    return compute_semantic_miou(gt_semantic, pred_semantic, num_classes)


# ── clustering metrics ───────────────────────────────────────────────────


def _compute_clustering_metrics(
    gt_ids: np.ndarray,
    pred_labels: np.ndarray,
) -> dict[str, float]:
    """NMI + ARI between GT and predicted labels on foreground points."""
    mask = gt_ids > 0
    if not np.any(mask):
        return {"NMI": float("nan"), "ARI": float("nan")}

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        log.warning("scikit-learn not installed; cannot compute NMI/ARI")
        return {"NMI": float("nan"), "ARI": float("nan")}

    y_true = gt_ids[mask].astype(np.int64)
    y_pred = pred_labels[mask].astype(np.int64)

    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    ari = adjusted_rand_score(y_true, y_pred)

    return {"NMI": float(nmi), "ARI": float(ari)}


def _normalize_eval_benchmarks(eval_benchmarks: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize eval benchmark input into a non-empty list of strings.

    Accepts:
      - "scannet200"
      - "scannet,scannet200" (comma-separated)
      - ["scannet20", "scannet200"]
    """
    if isinstance(eval_benchmarks, str):
        parts = [p.strip() for p in eval_benchmarks.split(",")]
        out = [p for p in parts if p]
        return out or ["scannet200"]
    out = [str(x).strip() for x in eval_benchmarks]
    out = [x for x in out if x]
    return out or ["scannet200"]


# ── pseudo-GT evaluation ─────────────────────────────────────────────────


def _build_pseudo_gt_ids(targets: InstanceTargets) -> np.ndarray:
    """Convert InstanceTargets into a per-point label array (like ScanNet GT).

    Returns (N,) int64: 0 = not supervised, >0 = pseudo instance ID.
    """
    N = int(targets.supervision_mask.shape[0])
    gt_ids = np.zeros(N, dtype=np.int64)
    gt_masks = targets.gt_masks.numpy()
    for m in range(targets.num_instances):
        gt_ids[gt_masks[m]] = targets.instance_ids[m] + 1
    return gt_ids


def _pseudo_instance_class_map(targets: InstanceTargets) -> dict[int, int] | None:
    if targets.class_ids is None:
        return None
    return {
        int(targets.instance_ids[i]) + 1: int(targets.class_ids[i].item())
        for i in range(targets.num_instances)
    }


# ── main evaluation entry point ──────────────────────────────────────────


def evaluate_student_predictions(
    pred: dict[str, torch.Tensor],
    targets: InstanceTargets,
    scene_dir: str | Path,
    scene_id: str,
    *,
    score_threshold: float = 0.3,
    class_score_threshold: float | None = None,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmarks: str | list[str] | tuple[str, ...] = "scannet200",
    vertex_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Full evaluation of student predictions.

    Computes metrics against:
      - Real ScanNet GT (AP25, AP50, NMI, ARI)
      - CHORUS pseudo-GT (AP25, AP50, NMI, ARI)

    Parameters
    ----------
    pred : dict with mask_logits [Q, N] and score_logits [Q]
    targets : InstanceTargets from training
    scene_dir : path to the ScanNet scene directory
    scene_id : e.g. "scene0042_00"
    """
    scene_dir = Path(scene_dir)
    mask_logits = pred["mask_logits"].detach().cpu()
    score_logits = pred["score_logits"].detach().cpu()
    class_logits = pred.get("class_logits")
    class_logits_cpu = class_logits.detach().cpu() if class_logits is not None else None

    proposals, scores, query_idx = extract_proposals(
        mask_logits, score_logits,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        min_points=min_points,
    )
    N = mask_logits.shape[1]
    pred_labels = proposals_to_labels(proposals, N)

    result: dict[str, Any] = {
        "num_proposals": len(proposals),
        "score_threshold": score_threshold,
        "class_score_threshold": (
            score_threshold if class_score_threshold is None else class_score_threshold
        ),
        "mask_threshold": mask_threshold,
    }

    class_proposals: list[np.ndarray] = []
    class_scores = np.zeros(0, dtype=np.float32)
    class_ids = np.zeros(0, dtype=np.int64)
    if class_logits_cpu is not None:
        class_proposals, class_scores, class_ids, _ = extract_class_aware_proposals(
            mask_logits,
            class_logits_cpu,
            class_score_threshold=(
                score_threshold if class_score_threshold is None else class_score_threshold
            ),
            mask_threshold=mask_threshold,
            min_points=min_points,
        )
        result["num_class_proposals"] = len(class_proposals)

    # ── 1. evaluate against pseudo-GT ──
    pseudo_gt_ids = _build_pseudo_gt_ids(targets)
    pseudo_ap = _evaluate_ap_against_gt(pseudo_gt_ids, proposals)
    pseudo_clustering = _compute_clustering_metrics(pseudo_gt_ids, pred_labels)

    result["pseudo_gt"] = {
        **pseudo_ap,
        **pseudo_clustering,
    }
    pseudo_class_map = _pseudo_instance_class_map(targets)
    if class_logits_cpu is not None and pseudo_class_map is not None:
        result["pseudo_gt_class_aware"] = evaluate_class_aware_ap(
            pseudo_gt_ids,
            pseudo_class_map,
            class_proposals,
            class_scores,
            class_ids,
        )
    log.info(
        "  [pseudo-GT] AP25=%.3f  AP50=%.3f  NMI=%.4f  ARI=%.4f  (%d GT instances)",
        pseudo_ap["AP25"], pseudo_ap["AP50"],
        pseudo_clustering["NMI"], pseudo_clustering["ARI"],
        pseudo_ap["total_gt_instances"],
    )

    # ── 2. evaluate against real ScanNet GT (optionally multiple benchmarks) ──
    benchmarks = _normalize_eval_benchmarks(eval_benchmarks)
    result["eval_benchmarks"] = benchmarks

    real_by_benchmark: dict[str, Any] = {}
    primary_benchmark = benchmarks[0]
    try:
        _ensure_chorus_importable()
        from chorus.datasets.scannet.gt import (
            load_scannet_gt_instance_ids,
            load_scannet_gt_instances,
        )

        for bench in benchmarks:
            try:
                real_instances = None
                if class_logits_cpu is not None:
                    real_instances = load_scannet_gt_instances(
                        scene_dir, scene_id, eval_benchmark=bench,
                    )
                    real_gt = real_instances.instance_ids
                else:
                    real_gt = load_scannet_gt_instance_ids(
                        scene_dir, scene_id, eval_benchmark=bench,
                    )

                real_gt_error: str | None = None
                real_gt_local = real_gt
                if vertex_indices is not None:
                    vi = vertex_indices.detach().cpu().numpy().astype(np.int64, copy=False)
                    if vi.shape[0] != N:
                        log.warning(
                            "  vertex_indices length %d != model point count %d, skipping real GT eval",
                            vi.shape[0], N,
                        )
                        real_gt_error = "vertex_indices length mismatch"
                    elif vi.size > 0 and (
                        int(vi.min()) < 0 or int(vi.max()) >= real_gt_local.shape[0]
                    ):
                        log.warning(
                            "  vertex_indices out of range [0, %d), skipping real GT eval",
                            real_gt_local.shape[0],
                        )
                        real_gt_error = "vertex_indices out of range"
                    else:
                        real_gt_local = real_gt_local[vi]

                if real_gt_error is None and real_gt_local.shape[0] != N:
                    log.warning(
                        "  GT vertex count %d != model point count %d, skipping real GT eval",
                        real_gt_local.shape[0], N,
                    )
                    real_gt_error = "vertex count mismatch"

                if real_gt_error is not None:
                    real_by_benchmark[bench] = {"error": real_gt_error, "eval_benchmark": bench}
                    continue

                real_ap = _evaluate_ap_against_gt(real_gt_local, proposals)
                real_clustering = _compute_clustering_metrics(real_gt_local, pred_labels)
                real_by_benchmark[bench] = {
                    **real_ap,
                    **real_clustering,
                    "eval_benchmark": bench,
                }
                if class_logits_cpu is not None and real_instances is not None:
                    class_ap = evaluate_class_aware_ap(
                        real_gt_local,
                        real_instances.instance_class_ids,
                        class_proposals,
                        class_scores,
                        class_ids,
                    )
                    class_ap["semantic_mIoU"] = evaluate_class_aware_semantic_miou(
                        real_gt_local,
                        real_instances.instance_class_ids,
                        class_proposals,
                        class_scores,
                        class_ids,
                        num_classes=int(class_logits_cpu.shape[-1] - 1),
                    )
                    real_by_benchmark[bench]["class_aware"] = class_ap
                log.info(
                    "  [real GT %s] AP25=%.3f  AP50=%.3f  NMI=%.4f  ARI=%.4f  (%d GT instances)",
                    bench,
                    real_ap["AP25"], real_ap["AP50"],
                    real_clustering["NMI"], real_clustering["ARI"],
                    real_ap["total_gt_instances"],
                )
                if "class_aware" in real_by_benchmark[bench]:
                    ca = real_by_benchmark[bench]["class_aware"]
                    log.info(
                        "  [real GT %s class-aware] AP25=%.3f  AP50=%.3f  sem_mIoU=%.3f  (%d predictions)",
                        bench,
                        ca["AP25"],
                        ca["AP50"],
                        ca["semantic_mIoU"],
                        ca["num_predictions"],
                    )
            except FileNotFoundError as e:
                log.warning("  Real GT not available (%s): %s", bench, e)
                real_by_benchmark[bench] = {"error": str(e), "eval_benchmark": bench}
            except Exception as e:
                log.warning("  Real GT eval failed (%s): %s", bench, e)
                real_by_benchmark[bench] = {"error": str(e), "eval_benchmark": bench}

    except Exception as e:
        # If import/setup fails, record a single shared error.
        log.warning("  Real GT eval setup failed: %s", e)
        real_by_benchmark[primary_benchmark] = {"error": str(e), "eval_benchmark": primary_benchmark}

    # Backward compatibility: keep `real_gt` as the primary benchmark result.
    result["real_gt_by_benchmark"] = real_by_benchmark
    result["real_gt"] = real_by_benchmark.get(primary_benchmark, {"error": "missing primary real GT result"})

    return result


def evaluate_student_predictions_multi(
    pred: dict,
    targets_by_granularity: dict[str, InstanceTargets],
    scene_dir: str | Path,
    scene_id: str,
    *,
    score_threshold: float = 0.3,
    class_score_threshold: float | None = None,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmarks: str | list[str] | tuple[str, ...] = "scannet200",
    vertex_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Full evaluation for each granularity head.

    Returns dict mapping granularity keys to per-head evaluation results.
    """
    result: dict[str, Any] = {}
    for g, targets_g in targets_by_granularity.items():
        head_pred = pred["heads"][g]
        result[g] = evaluate_student_predictions(
            head_pred,
            targets_g,
            scene_dir=scene_dir,
            scene_id=scene_id,
            score_threshold=score_threshold,
            class_score_threshold=class_score_threshold,
            mask_threshold=mask_threshold,
            min_points=min_points,
            eval_benchmarks=eval_benchmarks,
            vertex_indices=vertex_indices,
        )
    return result
