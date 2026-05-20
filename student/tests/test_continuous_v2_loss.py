from __future__ import annotations

import torch

from student.data.target_builder import build_instance_targets
from student.losses.continuous_geometry_loss import ContinuousGeometryCriterion
from student.losses.mask_set_loss import MaskSetCriterion


def _targets():
    labels = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    supervision = torch.ones_like(labels, dtype=torch.bool)
    return build_instance_targets(labels, supervision)


def test_center_loss_is_added_for_matched_v2_queries() -> None:
    targets = _targets()
    mask_logits = torch.tensor(
        [
            [20.0, 20.0, -20.0, -20.0],
            [-20.0, -20.0, 20.0, 20.0],
            [-20.0, -20.0, -20.0, -20.0],
        ]
    )
    score_logits = torch.zeros(3)
    query_xyz = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    point_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    criterion = ContinuousGeometryCriterion(
        MaskSetCriterion(bce_weight=0.0, dice_weight=0.0, score_weight=0.0),
        center_weight=1.0,
    )

    result = criterion(
        {
            "mask_logits": mask_logits,
            "score_logits": score_logits,
            "query_xyz": query_xyz,
            "point_xyz": point_xyz,
        },
        targets,
        granularity_key="g05",
    )

    assert torch.isfinite(result["loss_total"])
    assert result["loss_center"].item() > 0.0
    assert result["center_error_mean"].item() > 0.0
    result["loss_total"].backward()
    assert query_xyz.grad is not None
    assert torch.any(query_xyz.grad.abs() > 0)


def test_center_loss_is_zero_when_v2_geometry_outputs_are_absent() -> None:
    targets = _targets()
    criterion = ContinuousGeometryCriterion(
        MaskSetCriterion(bce_weight=0.0, dice_weight=0.0, score_weight=0.0),
        center_weight=1.0,
    )
    result = criterion(
        {
            "mask_logits": torch.zeros(3, 4),
            "score_logits": torch.zeros(3),
        },
        targets,
        granularity_key="g05",
    )

    assert torch.isfinite(result["loss_total"])
    assert result["loss_center"].item() == 0.0
    assert result["center_error_mean"].item() == 0.0
