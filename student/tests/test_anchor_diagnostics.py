import torch

from student.engine.anchor_diagnostics import (
    compute_anchor_to_centroid_stats,
    compute_scale_selector_stats,
)


def test_anchor_moves_towards_centroid():
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [8.0, 8.0, 0.0],
    ])
    target_masks = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    initial = torch.tensor([[5.0, 5.0, 0.0]])
    layers = torch.tensor([[[2.0, 2.0, 0.0]], [[0.4, 0.4, 0.0]]])
    stats = compute_anchor_to_centroid_stats(
        points,
        target_masks,
        initial,
        layers,
        matched_pred_indices=torch.tensor([0]),
        matched_target_indices=torch.tensor([0]),
    )
    assert stats["dist_to_matched_centroid_final_p50"] < stats["dist_to_matched_centroid_init_p50"]
    assert stats["movement_towards_centroid_fraction"] == 1.0
    assert stats["mean_distance_reduction_to_centroid"] > 0


def test_scale_selector_entropy():
    stats = compute_scale_selector_stats(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    assert stats["selector_weight_level_0"] == 0.25
    assert stats["selector_entropy"] > 1.0
