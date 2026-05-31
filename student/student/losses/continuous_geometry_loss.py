"""V2 continuous geometry decoder losses.

The base Hungarian mask/score behavior stays in :mod:`mask_set_loss`.  This
wrapper adds geometry-query losses that depend on V2 decoder outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from student.data.target_builder import InstanceTargets
from student.losses.mask_set_loss import MaskSetCriterion


def _as_unbatched_xyz(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.ndim == 3:
        if xyz.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 xyz tensors are supported, got {tuple(xyz.shape)}")
        return xyz[0]
    return xyz


def target_centroids_from_masks(
    targets: InstanceTargets,
    point_xyz: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute target centroids from supervised points."""
    point_xyz = _as_unbatched_xyz(point_xyz).to(device=device, dtype=dtype)
    gt_masks = targets.gt_masks.to(device=device).float()
    supervision = targets.supervision_mask.to(device=device).bool()

    if gt_masks.numel() == 0 or not supervision.any():
        return point_xyz.new_zeros((0, 3))

    masks_sup = gt_masks[:, supervision]
    xyz_sup = point_xyz[supervision]
    denom = masks_sup.sum(dim=1).clamp_min(1.0)
    return (masks_sup @ xyz_sup) / denom[:, None]


def matched_center_loss(
    pred: dict[str, torch.Tensor],
    targets: InstanceTargets,
    match_result: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth L1 loss between matched query anchors and target centroids."""
    mask_logits = pred["mask_logits"]
    zero = mask_logits.sum() * 0.0
    if "query_xyz" not in pred or "point_xyz" not in pred:
        return zero, zero.detach()

    pred_idx = match_result.get("matched_pred_indices")
    gt_idx = match_result.get("matched_gt_indices")
    if pred_idx is None or gt_idx is None or len(pred_idx) == 0:
        return zero, zero.detach()

    query_xyz = _as_unbatched_xyz(pred["query_xyz"]).to(
        device=mask_logits.device,
        dtype=mask_logits.dtype,
    )
    centroids = target_centroids_from_masks(
        targets,
        pred["point_xyz"],
        device=mask_logits.device,
        dtype=mask_logits.dtype,
    )
    if centroids.numel() == 0:
        return zero, zero.detach()

    pred_idx_t = torch.as_tensor(
        np.asarray(pred_idx, dtype=np.int64),
        device=mask_logits.device,
        dtype=torch.long,
    )
    gt_idx_t = torch.as_tensor(
        np.asarray(gt_idx, dtype=np.int64),
        device=mask_logits.device,
        dtype=torch.long,
    )
    matched_query_xyz = query_xyz[pred_idx_t]
    matched_centroids = centroids[gt_idx_t]
    loss = F.smooth_l1_loss(matched_query_xyz, matched_centroids, reduction="mean")
    error = torch.linalg.norm(matched_query_xyz.detach() - matched_centroids.detach(), dim=-1).mean()
    return loss, error


class ContinuousGeometryCriterion(nn.Module):
    """Single-granularity criterion for ``ContinuousGeometryQueryDecoderV2``."""

    def __init__(
        self,
        criterion: MaskSetCriterion,
        *,
        aux_weight: float = 0.0,
        granularity_weights: dict[str, float] | None = None,
        center_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.aux_weight = float(aux_weight)
        self.granularity_weights = granularity_weights or {}
        self.center_weight = float(center_weight)

    def _add_center_loss(
        self,
        pred: dict,
        targets: InstanceTargets,
        result: dict[str, Any],
    ) -> tuple[dict[str, Any], torch.Tensor]:
        loss_center, center_error = matched_center_loss(pred, targets, result)
        weighted = self.center_weight * loss_center
        result["loss_total"] = result["loss_total"] + weighted
        result["loss_center"] = loss_center.detach()
        result["center_error_mean"] = center_error.detach()
        result["center_weight"] = self.center_weight
        return result, weighted

    def forward(
        self,
        pred: dict,
        targets: InstanceTargets,
        *,
        context: str = "",
        granularity_key: str | None = None,
    ) -> dict[str, Any]:
        result = self.criterion(pred, targets, context=context)
        result, _ = self._add_center_loss(pred, targets, result)

        aux_center_terms: list[torch.Tensor] = []
        if self.aux_weight > 0 and "aux_outputs" in pred:
            aux_loss_sum = None
            for aux_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_ctx = f"{context}/aux{aux_idx}" if context else f"aux{aux_idx}"
                aux_result = self.criterion(aux_pred, targets, context=aux_ctx)
                aux_result, weighted_center = self._add_center_loss(
                    aux_pred,
                    targets,
                    aux_result,
                )
                result[f"loss_aux_layer_{aux_idx}"] = aux_result["loss_total"].detach()
                aux_center_terms.append(weighted_center.detach())
                if aux_loss_sum is None:
                    aux_loss_sum = aux_result["loss_total"]
                else:
                    aux_loss_sum = aux_loss_sum + aux_result["loss_total"]

            if aux_loss_sum is not None:
                weighted_aux = self.aux_weight * aux_loss_sum
                result["loss_aux"] = weighted_aux.detach()
                result["loss_total"] = result["loss_total"] + weighted_aux
                if aux_center_terms:
                    result["loss_center_aux"] = (
                        self.aux_weight * torch.stack(aux_center_terms).sum()
                    ).detach()

        unweighted_total = result["loss_total"]
        if granularity_key is not None:
            w = float(self.granularity_weights.get(granularity_key, 1.0))
            result["loss_total_unweighted"] = unweighted_total.detach()
        else:
            w = 1.0
        result["loss_total"] = unweighted_total * w

        return result
