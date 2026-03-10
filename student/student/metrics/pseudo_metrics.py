"""Cheap train-time metrics on matched predictions vs pseudo-GT.

These are computed per training step (or every N steps) to monitor
whether the model is actually learning the pseudo-label contract.
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np

from student.data.target_builder import InstanceTargets


def compute_matched_iou(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    supervision_mask: torch.Tensor,
    pred_idx: np.ndarray,
    gt_idx: np.ndarray,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute IoU for each matched pair over supervised points.

    Returns (num_matches,) float tensor.
    """
    supervised = supervision_mask.bool()
    pred_binary = (mask_logits[pred_idx][:, supervised].sigmoid() >= mask_threshold)
    gt_binary = gt_masks[gt_idx][:, supervised].bool()

    intersection = (pred_binary & gt_binary).sum(dim=1).float()
    union = (pred_binary | gt_binary).sum(dim=1).float()

    return intersection / union.clamp(min=1.0)


def compute_pseudo_metrics(
    pred: dict[str, torch.Tensor],
    targets: InstanceTargets,
    matched_pred_idx: np.ndarray,
    matched_gt_idx: np.ndarray,
    *,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute cheap train-time diagnostic metrics.

    Parameters
    ----------
    pred:
        Decoder output with ``mask_logits`` [Q, N] and ``score_logits`` [Q].
    targets:
        InstanceTargets with ``gt_masks`` [M, N] and ``supervision_mask`` [N].
    matched_pred_idx / matched_gt_idx:
        Arrays from Hungarian matching.
    score_threshold:
        Sigmoid threshold for counting "active" queries.
    mask_threshold:
        Sigmoid threshold for binarizing predicted masks.
    """
    mask_logits = pred["mask_logits"]
    score_logits = pred["score_logits"]
    gt_masks = targets.gt_masks.to(mask_logits.device)
    supervision_mask = targets.supervision_mask.to(mask_logits.device)

    Q = score_logits.shape[0]
    num_matches = len(matched_pred_idx)

    metrics: dict[str, Any] = {}

    # ── matched IoU ──
    if num_matches > 0:
        ious = compute_matched_iou(
            mask_logits, gt_masks, supervision_mask,
            matched_pred_idx, matched_gt_idx,
            mask_threshold=mask_threshold,
        )
        metrics["matched_mean_iou"] = float(ious.mean())
        metrics["matched_median_iou"] = float(ious.median())
        metrics["matched_iou_gt_0.25"] = float((ious > 0.25).float().mean())
        metrics["matched_iou_gt_0.50"] = float((ious > 0.50).float().mean())
    else:
        metrics["matched_mean_iou"] = 0.0
        metrics["matched_median_iou"] = 0.0
        metrics["matched_iou_gt_0.25"] = 0.0
        metrics["matched_iou_gt_0.50"] = 0.0

    # ── score statistics ──
    scores = score_logits.sigmoid().detach()

    if num_matches > 0:
        matched_mask = torch.zeros(Q, dtype=torch.bool, device=scores.device)
        matched_mask[matched_pred_idx] = True
        metrics["mean_score_matched"] = float(scores[matched_mask].mean())
        metrics["mean_score_unmatched"] = float(scores[~matched_mask].mean()) if (~matched_mask).any() else 0.0
    else:
        metrics["mean_score_matched"] = 0.0
        metrics["mean_score_unmatched"] = float(scores.mean())

    metrics["num_queries_above_threshold"] = int((scores >= score_threshold).sum())
    metrics["num_matches"] = num_matches

    return metrics


def format_pseudo_metrics(m: dict[str, Any]) -> str:
    """One-line summary for logging."""
    return (
        f"mIoU={m['matched_mean_iou']:.3f}  "
        f"medIoU={m['matched_median_iou']:.3f}  "
        f"IoU>.25={m['matched_iou_gt_0.25']:.2f}  "
        f"IoU>.50={m['matched_iou_gt_0.50']:.2f}  "
        f"score_m={m['mean_score_matched']:.3f}  "
        f"score_u={m['mean_score_unmatched']:.3f}  "
        f"active={m['num_queries_above_threshold']}"
    )
