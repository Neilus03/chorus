from __future__ import annotations

import torch

from student.data.target_builder import build_instance_targets
from student.losses.mask_set_loss import MaskSetCriterion


def test_build_instance_targets_keeps_class_ids_aligned_after_filtering() -> None:
    labels = torch.tensor([1, 1, 2, 3, 3, 3, -1])
    supervision = torch.ones_like(labels, dtype=torch.bool)

    targets = build_instance_targets(
        labels,
        supervision,
        min_instance_points=2,
        instance_class_map={1: 4, 2: 7, 3: 9},
    )

    assert targets.instance_ids.tolist() == [1, 3]
    assert targets.instance_sizes.tolist() == [2, 3]
    assert targets.class_ids is not None
    assert targets.class_ids.tolist() == [4, 9]


def test_class_aware_criterion_penalizes_wrong_class_perfect_mask() -> None:
    labels = torch.tensor([1, 1, 2, 2])
    supervision = torch.ones_like(labels, dtype=torch.bool)
    targets = build_instance_targets(
        labels,
        supervision,
        instance_class_map={1: 0, 2: 1},
    )
    mask_logits = torch.tensor(
        [
            [8.0, 8.0, -8.0, -8.0],
            [-8.0, -8.0, 8.0, 8.0],
            [-8.0, -8.0, -8.0, -8.0],
        ]
    )
    score_logits = torch.zeros(3)
    class_logits = torch.tensor(
        [
            [-4.0, 4.0, -4.0],  # wrong class for instance 1
            [-4.0, 4.0, -4.0],  # right class for instance 2
            [-4.0, -4.0, 4.0],  # no-object
        ]
    )

    criterion = MaskSetCriterion(
        bce_weight=1.0,
        dice_weight=1.0,
        score_weight=0.0,
        class_weight=1.0,
        no_object_weight=0.1,
        cost_class_weight=1.0,
    )
    result = criterion(
        {
            "mask_logits": mask_logits,
            "score_logits": score_logits,
            "class_logits": class_logits,
        },
        targets,
    )

    assert torch.isfinite(result["loss_total"])
    assert result["loss_class"].item() > 0.5
