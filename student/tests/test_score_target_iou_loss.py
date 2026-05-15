from __future__ import annotations

import torch
import torch.nn.functional as F

from student.data.target_builder import build_instance_targets
from student.losses.mask_set_loss import (
    MaskSetCriterion,
    MultiGranCriterion,
    SingleGranCriterion,
)


def _targets(labels: list[int]):
    labels_t = torch.tensor(labels, dtype=torch.long)
    supervision = torch.ones_like(labels_t, dtype=torch.bool)
    return build_instance_targets(labels_t, supervision)


def test_binary_score_target_mode_keeps_legacy_targets() -> None:
    targets = _targets([1, 1])
    mask_logits = torch.tensor(
        [
            [8.0, 8.0],
            [-8.0, -8.0],
        ]
    )
    score_logits = torch.tensor([0.2, -0.4])
    criterion = MaskSetCriterion(
        bce_weight=0.0,
        dice_weight=0.0,
        score_weight=1.0,
        score_target_mode="binary",
    )

    result = criterion({"mask_logits": mask_logits, "score_logits": score_logits}, targets)

    expected = F.binary_cross_entropy_with_logits(
        score_logits,
        torch.tensor([1.0, 0.0]),
        reduction="mean",
    )
    torch.testing.assert_close(result["loss_score"], expected)
    assert result["score_target_mean_matched"].item() == 1.0
    assert result["score_target_max_matched"].item() == 1.0
    assert result["score_target_min_matched"].item() == 1.0
    assert result["score_target_mean_all"].item() == 0.5
    assert result["num_score_targets_positive"].item() == 1.0


def test_iou_score_target_mode_perfect_mask_is_near_one() -> None:
    targets = _targets([1, 1])
    mask_logits = torch.tensor(
        [
            [20.0, 20.0],
            [-20.0, -20.0],
        ]
    )
    score_logits = torch.zeros(2)
    criterion = MaskSetCriterion(score_target_mode="iou")

    result = criterion({"mask_logits": mask_logits, "score_logits": score_logits}, targets)

    assert result["score_target_mean_matched"].item() > 0.99
    assert result["score_target_max_matched"].item() > 0.99
    assert result["num_score_targets_positive"].item() == 1.0


def test_iou_score_target_mode_partial_mask_is_between_zero_and_one() -> None:
    targets = _targets([1, 1])
    mask_logits = torch.tensor(
        [
            [20.0, -20.0],
            [-20.0, -20.0],
        ]
    )
    score_logits = torch.zeros(2)
    criterion = MaskSetCriterion(score_target_mode="iou")

    result = criterion({"mask_logits": mask_logits, "score_logits": score_logits}, targets)

    target = result["score_target_mean_matched"].item()
    assert 0.1 < target < 0.9
    assert result["score_target_mean_all"].item() < target


def test_iou_score_target_mode_unmatched_queries_remain_zero() -> None:
    targets = _targets([1, 1, 2, 2])
    mask_logits = torch.tensor(
        [
            [20.0, 20.0, -20.0, -20.0],
            [-20.0, -20.0, 20.0, -20.0],
            [-20.0, -20.0, -20.0, -20.0],
        ]
    )
    score_logits = torch.zeros(3)
    criterion = MaskSetCriterion(score_target_mode="iou")

    result = criterion({"mask_logits": mask_logits, "score_logits": score_logits}, targets)

    assert result["num_matches"] == 2
    assert result["num_score_targets_positive"].item() == 2.0
    assert result["score_target_max_matched"].item() > 0.99
    assert 0.1 < result["score_target_min_matched"].item() < 0.9
    assert result["score_target_mean_all"].item() < result["score_target_mean_matched"].item()


def test_iou_score_target_is_detached_from_mask_logits() -> None:
    targets = _targets([1, 1])
    mask_logits = torch.tensor(
        [
            [20.0, 20.0],
            [-20.0, -20.0],
        ],
        requires_grad=True,
    )
    score_logits = torch.zeros(2, requires_grad=True)
    criterion = MaskSetCriterion(
        bce_weight=0.0,
        dice_weight=0.0,
        score_weight=1.0,
        score_target_mode="iou",
    )

    result = criterion({"mask_logits": mask_logits, "score_logits": score_logits}, targets)
    result["loss_total"].backward()

    assert score_logits.grad is not None
    assert torch.any(score_logits.grad.abs() > 0)
    if mask_logits.grad is not None:
        torch.testing.assert_close(mask_logits.grad, torch.zeros_like(mask_logits.grad))


def test_multigran_criterion_preserves_score_target_diagnostics() -> None:
    targets = _targets([1, 1])
    pred = {
        "heads": {
            "g02": {
                "mask_logits": torch.tensor([[20.0, 20.0], [-20.0, -20.0]]),
                "score_logits": torch.zeros(2),
            },
            "g05": {
                "mask_logits": torch.tensor([[20.0, -20.0], [-20.0, -20.0]]),
                "score_logits": torch.zeros(2),
            },
        }
    }
    criterion = MultiGranCriterion(
        MaskSetCriterion(score_target_mode="iou"),
        granularity_weights={"g02": 1.0, "g05": 1.0},
    )

    result = criterion(pred, {"g02": targets, "g05": targets})

    assert torch.isfinite(result["loss_total"])
    assert result["heads"]["g02"]["score_target_mean_matched"].item() > 0.99
    assert 0.1 < result["heads"]["g05"]["score_target_mean_matched"].item() < 0.9


def test_singlegran_criterion_preserves_score_target_diagnostics() -> None:
    targets = _targets([1, 1])
    pred = {
        "mask_logits": torch.tensor([[20.0, 20.0], [-20.0, -20.0]]),
        "score_logits": torch.zeros(2),
    }
    criterion = SingleGranCriterion(
        MaskSetCriterion(score_target_mode="iou"),
        granularity_weights={"g05": 1.0},
    )

    result = criterion(pred, targets, granularity_key="g05")

    assert torch.isfinite(result["loss_total"])
    assert result["score_target_mean_matched"].item() > 0.99
    assert result["score_target_mean_all"].item() < result["score_target_mean_matched"].item()
