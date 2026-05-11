from __future__ import annotations

import numpy as np

from student.engine.evaluator import (
    evaluate_class_aware_ap,
    evaluate_class_aware_semantic_miou,
)


def test_class_aware_ap_rejects_wrong_class_perfect_mask() -> None:
    gt_ids = np.array([1, 1, 2, 2])
    gt_classes = {1: 0, 2: 1}
    proposals = [
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    ]
    scores = np.array([0.9, 0.8])
    wrong_classes = np.array([1, 0])

    result = evaluate_class_aware_ap(
        gt_ids,
        gt_classes,
        proposals,
        scores,
        wrong_classes,
    )

    assert result["AP50"] == 0.0


def test_class_aware_ap_accepts_correct_class_masks() -> None:
    gt_ids = np.array([1, 1, 2, 2])
    gt_classes = {1: 0, 2: 1}
    proposals = [
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    ]
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 1])

    result = evaluate_class_aware_ap(
        gt_ids,
        gt_classes,
        proposals,
        scores,
        classes,
    )

    assert result["AP50"] == 1.0


def test_class_aware_semantic_miou_uses_predicted_classes() -> None:
    gt_ids = np.array([1, 1, 2, 2])
    gt_classes = {1: 0, 2: 1}
    proposals = [
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    ]
    scores = np.array([0.9, 0.8])

    correct = evaluate_class_aware_semantic_miou(
        gt_ids,
        gt_classes,
        proposals,
        scores,
        np.array([0, 1]),
        num_classes=2,
    )
    wrong = evaluate_class_aware_semantic_miou(
        gt_ids,
        gt_classes,
        proposals,
        scores,
        np.array([1, 0]),
        num_classes=2,
    )

    assert correct == 1.0
    assert wrong == 0.0
