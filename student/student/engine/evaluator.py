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


# ── main evaluation entry point ──────────────────────────────────────────


def evaluate_student_predictions(
    pred: dict[str, torch.Tensor],
    targets: InstanceTargets,
    scene_dir: str | Path,
    scene_id: str,
    *,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmark: str = "scannet200",
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
        "mask_threshold": mask_threshold,
    }

    # ── 1. evaluate against pseudo-GT ──
    pseudo_gt_ids = _build_pseudo_gt_ids(targets)
    pseudo_ap = _evaluate_ap_against_gt(pseudo_gt_ids, proposals)
    pseudo_clustering = _compute_clustering_metrics(pseudo_gt_ids, pred_labels)

    result["pseudo_gt"] = {
        **pseudo_ap,
        **pseudo_clustering,
    }
    log.info(
        "  [pseudo-GT] AP25=%.3f  AP50=%.3f  NMI=%.4f  ARI=%.4f  (%d GT instances)",
        pseudo_ap["AP25"], pseudo_ap["AP50"],
        pseudo_clustering["NMI"], pseudo_clustering["ARI"],
        pseudo_ap["total_gt_instances"],
    )

    # ── 2. evaluate against real ScanNet GT ──
    try:
        _ensure_chorus_importable()
        from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids

        real_gt = load_scannet_gt_instance_ids(
            scene_dir, scene_id, eval_benchmark=eval_benchmark,
        )

        if real_gt.shape[0] != N:
            log.warning(
                "  GT vertex count %d != model point count %d, skipping real GT eval",
                real_gt.shape[0], N,
            )
            result["real_gt"] = {"error": "vertex count mismatch"}
        else:
            real_ap = _evaluate_ap_against_gt(real_gt, proposals)
            real_clustering = _compute_clustering_metrics(real_gt, pred_labels)

            result["real_gt"] = {
                **real_ap,
                **real_clustering,
                "eval_benchmark": eval_benchmark,
            }
            log.info(
                "  [real GT %s] AP25=%.3f  AP50=%.3f  NMI=%.4f  ARI=%.4f  (%d GT instances)",
                eval_benchmark,
                real_ap["AP25"], real_ap["AP50"],
                real_clustering["NMI"], real_clustering["ARI"],
                real_ap["total_gt_instances"],
            )

    except FileNotFoundError as e:
        log.warning("  Real GT not available: %s", e)
        result["real_gt"] = {"error": str(e)}
    except Exception as e:
        log.warning("  Real GT eval failed: %s", e)
        result["real_gt"] = {"error": str(e)}

    return result
