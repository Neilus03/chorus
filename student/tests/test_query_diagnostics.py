import torch

from student.engine.query_diagnostics import (
    compute_duplicate_iou_stats,
    compute_matching_calibration_stats,
    compute_query_score_mask_stats,
)


def test_query_duplicate_iou_high_for_duplicate_masks():
    mask_logits = torch.full((3, 8), -8.0)
    mask_logits[0, :4] = 8.0
    mask_logits[1, :4] = 8.0
    mask_logits[2, 4:] = 8.0
    scores = torch.tensor([3.0, 2.0, 1.0])
    stats = compute_duplicate_iou_stats(mask_logits, scores, topk=2)
    assert stats["duplicate_iou_topk_mean"] > 0.99
    assert stats["duplicate_iou_topk_max"] > 0.99


def test_query_duplicate_iou_low_for_disjoint_masks():
    mask_logits = torch.full((2, 8), -8.0)
    mask_logits[0, :4] = 8.0
    mask_logits[1, 4:] = 8.0
    stats = compute_duplicate_iou_stats(mask_logits, torch.tensor([2.0, 1.0]), topk=2)
    assert stats["duplicate_iou_topk_max"] == 0.0


def test_dead_query_fraction():
    mask_logits = torch.full((4, 10), -8.0)
    score_logits = torch.full((4,), -8.0)
    stats = compute_query_score_mask_stats(
        mask_logits,
        score_logits,
        min_points_per_proposal=3,
    )
    assert stats["dead_query_fraction"] == 1.0


def test_matching_score_gap():
    score_logits = torch.tensor([4.0, 3.0, -4.0, -5.0])
    stats = compute_matching_calibration_stats(
        score_logits,
        matched_pred_indices=torch.tensor([0, 1]),
        matched_target_indices=torch.tensor([0, 1]),
        matched_ious=torch.tensor([0.5, 0.8]),
    )
    assert stats["score_gap_matched_minus_unmatched"] > 0.9
    assert stats["matched_query_fraction"] == 0.5
