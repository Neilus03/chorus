"""Set-prediction criterion for instance segmentation.

Pairs predicted queries to GT pseudo-instances via Hungarian matching,
then supervises matched masks (BCE + Dice) and all query scores.

All mask losses are computed **only over supervised points**.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from student.data.target_builder import InstanceTargets

log = logging.getLogger(__name__)


# ── primitive losses ─────────────────────────────────────────────────────


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits averaged over the given (supervised) points.

    Parameters
    ----------
    logits  : (K,) float — predicted mask logits for supervised points.
    targets : (K,) float — 0/1 GT mask values for the same points.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def sigmoid_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Dice loss computed from logits over supervised points.

    Parameters
    ----------
    logits  : (K,) float
    targets : (K,) float — 0/1
    eps     : smoothing to avoid 0/0.
    """
    probs = logits.sigmoid()
    intersection = (probs * targets).sum()
    cardinality = probs.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


# ── pairwise cost matrix ────────────────────────────────────────────────


@torch.no_grad()
def build_pairwise_cost_matrix(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Build [Q, M] cost matrix for Hungarian matching.

    Iterates over GT masks to avoid materializing a [Q, M, N] tensor.
    All costs are computed only over supervised points.
    """
    supervised = supervision_mask.bool()
    pred_sup = mask_logits[:, supervised]    # [Q, K]
    Q, K = pred_sup.shape
    M = gt_masks.shape[0]

    cost = torch.zeros(Q, M, device=mask_logits.device)

    for m in range(M):
        gt_m = gt_masks[m, supervised].float()  # [K]

        if bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(
                pred_sup, gt_m.unsqueeze(0).expand_as(pred_sup), reduction="none",
            ).mean(dim=1)  # [Q]
            cost[:, m] += bce_weight * bce

        if dice_weight > 0:
            probs = pred_sup.sigmoid()          # [Q, K]
            inter = (probs * gt_m.unsqueeze(0)).sum(dim=1)
            card = probs.sum(dim=1) + gt_m.sum()
            dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)
            cost[:, m] += dice_weight * dice

    return cost


# ── Hungarian matching ───────────────────────────────────────────────────


@torch.no_grad()
def hungarian_match(
    cost_matrix: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run scipy Hungarian on a [Q, M] cost matrix.

    Returns (pred_indices, gt_indices) — both int arrays of length M
    (assuming Q >= M, every GT gets a match).
    """
    cost_np = cost_matrix.detach().cpu().numpy()
    pred_idx, gt_idx = linear_sum_assignment(cost_np)
    return pred_idx.astype(np.int64), gt_idx.astype(np.int64)


# ── criterion ────────────────────────────────────────────────────────────


class MaskSetCriterion(nn.Module):
    """Set-prediction loss: Hungarian matching + BCE + Dice + score BCE.

    Parameters
    ----------
    bce_weight:
        Weight on per-mask BCE loss.
    dice_weight:
        Weight on per-mask Dice loss.
    score_weight:
        Weight on the objectness score BCE loss.
    cost_bce_weight / cost_dice_weight:
        Weights used inside the matching cost matrix (can differ from
        the loss weights).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        score_weight: float = 0.5,
        cost_bce_weight: float = 1.0,
        cost_dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.score_weight = score_weight
        self.cost_bce_weight = cost_bce_weight
        self.cost_dice_weight = cost_dice_weight

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        targets: InstanceTargets,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        pred:
            Dict from the decoder with at least ``mask_logits`` [Q, N]
            and ``score_logits`` [Q].
        targets:
            :class:`InstanceTargets` with ``gt_masks`` [M, N] and
            ``supervision_mask`` [N].

        Returns
        -------
        Dict with individual losses, total loss, and matching info.
        """
        mask_logits = pred["mask_logits"]          # [Q, N]
        score_logits = pred["score_logits"]        # [Q]
        gt_masks = targets.gt_masks.to(mask_logits.device)
        supervision_mask = targets.supervision_mask.to(mask_logits.device)

        Q = mask_logits.shape[0]
        M = gt_masks.shape[0]
        supervised = supervision_mask.bool()

        # ── matching ──
        cost_matrix = build_pairwise_cost_matrix(
            mask_logits, gt_masks, supervision_mask,
            bce_weight=self.cost_bce_weight,
            dice_weight=self.cost_dice_weight,
        )
        pred_idx, gt_idx = hungarian_match(cost_matrix)
        pred_idx_t = torch.from_numpy(pred_idx).to(mask_logits.device)
        gt_idx_t = torch.from_numpy(gt_idx).to(mask_logits.device)

        # ── matched mask losses (only over supervised points) ──
        if len(pred_idx) > 0:
            matched_pred = mask_logits[pred_idx_t][:, supervised]   # [M', K]
            matched_gt = gt_masks[gt_idx_t][:, supervised].float()  # [M', K]

            bce_per_pair = torch.stack([
                masked_bce_with_logits(matched_pred[i], matched_gt[i])
                for i in range(len(pred_idx))
            ])
            dice_per_pair = torch.stack([
                sigmoid_dice_loss(matched_pred[i], matched_gt[i])
                for i in range(len(pred_idx))
            ])

            loss_mask_bce = bce_per_pair.mean()
            loss_mask_dice = dice_per_pair.mean()
        else:
            loss_mask_bce = mask_logits.sum() * 0.0
            loss_mask_dice = mask_logits.sum() * 0.0

        # ── score loss ──
        score_targets = torch.zeros(Q, device=score_logits.device)
        score_targets[pred_idx_t] = 1.0
        loss_score = F.binary_cross_entropy_with_logits(
            score_logits, score_targets, reduction="mean",
        )

        # ── total ──
        loss_total = (
            self.bce_weight * loss_mask_bce
            + self.dice_weight * loss_mask_dice
            + self.score_weight * loss_score
        )

        return {
            "loss_total": loss_total,
            "loss_mask_bce": loss_mask_bce.detach(),
            "loss_mask_dice": loss_mask_dice.detach(),
            "loss_score": loss_score.detach(),
            "num_matches": len(pred_idx),
            "matched_pred_indices": pred_idx,
            "matched_gt_indices": gt_idx,
            "cost_matrix_shape": tuple(cost_matrix.shape),
        }


class MultiGranCriterion(nn.Module):
    """Applies :class:`MaskSetCriterion` independently per granularity head
    and sums the losses.

    When the decoder returns ``aux_outputs`` (intermediate layer predictions),
    the same criterion is applied to each auxiliary layer with independent
    Hungarian matching, weighted by *aux_weight*.

    Parameters
    ----------
    criterion:
        A shared :class:`MaskSetCriterion` instance (same loss weights
        for every head).
    granularity_weights:
        Optional per-head loss multiplier.  Defaults to 1.0 for each.
    aux_weight:
        Multiplier on losses from intermediate decoder layers.
    """

    def __init__(
        self,
        criterion: MaskSetCriterion,
        granularity_weights: dict[str, float] | None = None,
        aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.granularity_weights = granularity_weights or {}
        self.aux_weight = aux_weight

    def _compute_heads_loss(
        self,
        heads_pred: dict,
        targets_by_granularity: dict[str, "InstanceTargets"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute per-head losses and sum them. Returns (total, per-head dict)."""
        total_loss = None
        heads_result: dict[str, Any] = {}

        for g, targets_g in targets_by_granularity.items():
            head_pred = heads_pred[g]
            ld = self.criterion(head_pred, targets_g)

            w = self.granularity_weights.get(g, 1.0)
            weighted = w * ld["loss_total"]

            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted

            heads_result[g] = ld

        assert total_loss is not None, "No granularity heads to compute loss for"
        return total_loss, heads_result

    def forward(
        self,
        pred: dict,
        targets_by_granularity: dict[str, "InstanceTargets"],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        pred:
            Nested dict from ``MultiHeadQueryInstanceDecoder`` with
            ``pred["heads"][g]`` containing ``mask_logits`` and
            ``score_logits`` for each granularity ``g``.
            Optionally ``pred["aux_outputs"]`` — list of intermediate-layer
            prediction dicts with the same structure.
        targets_by_granularity:
            Dict mapping granularity keys to :class:`InstanceTargets`.

        Returns
        -------
        Dict with ``loss_total`` (summed), per-head loss dicts keyed by
        granularity, per-head matching info, and optional ``loss_aux``.
        """
        final_loss, heads_result = self._compute_heads_loss(
            pred["heads"], targets_by_granularity,
        )
        result: dict[str, Any] = {"heads": heads_result, "loss_total": final_loss}

        if self.aux_weight > 0 and "aux_outputs" in pred:
            aux_loss_sum = None
            for aux_pred in pred["aux_outputs"]:
                layer_loss, _ = self._compute_heads_loss(
                    aux_pred["heads"], targets_by_granularity,
                )
                if aux_loss_sum is None:
                    aux_loss_sum = layer_loss
                else:
                    aux_loss_sum = aux_loss_sum + layer_loss

            if aux_loss_sum is not None:
                weighted_aux = self.aux_weight * aux_loss_sum
                result["loss_aux"] = weighted_aux.detach()
                result["loss_total"] = result["loss_total"] + weighted_aux

        return result
