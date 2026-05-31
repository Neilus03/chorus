from __future__ import annotations

import numpy as np

from scripts.sweep_mask_score_diagnostics import (
    _records_with_oracle_as_score,
    build_thresholded_proposals,
    scores_for_mode,
)
from student.metrics.official_instance_ap import build_instance_ap_records, evaluate_official_and_oracle_ap


def test_thresholded_proposals_filter_by_mask_threshold_and_min_points() -> None:
    mask_probs = np.asarray(
        [
            [0.90, 0.80, 0.10, 0.00],
            [0.20, 0.70, 0.80, 0.10],
            [0.60, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    proposals, keep_idx, counts, stats = build_thresholded_proposals(
        mask_probs,
        mask_threshold=0.5,
        min_points=2,
    )

    assert keep_idx.tolist() == [0, 1]
    assert counts.tolist() == [2, 2, 1]
    assert len(proposals) == 2
    assert proposals[0].tolist() == [True, True, False, False]
    assert proposals[1].tolist() == [False, True, True, False]
    assert stats["num_proposals"] == 2
    assert stats["num_min_points_removed"] == 1
    assert stats["mean_mask_area"] == 0.5


def test_score_modes_rank_expected_toy_proposals() -> None:
    mask_probs = np.asarray(
        [
            [0.90, 0.80, 0.10, 0.00],
            [0.20, 0.70, 0.80, 0.10],
            [0.60, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    proposals, keep_idx, counts, _ = build_thresholded_proposals(
        mask_probs,
        mask_threshold=0.5,
        min_points=1,
    )
    learned = np.asarray([0.1, 0.8, 0.4], dtype=np.float64)

    learned_scores = scores_for_mode(
        mode="learned",
        learned_scores=learned,
        mask_probs=mask_probs,
        keep_indices=keep_idx,
        mask_counts=counts,
        proposals=proposals,
    )
    one_scores = scores_for_mode(
        mode="ones",
        learned_scores=learned,
        mask_probs=mask_probs,
        keep_indices=keep_idx,
        mask_counts=counts,
        proposals=proposals,
    )
    area_scores = scores_for_mode(
        mode="mask_area",
        learned_scores=learned,
        mask_probs=mask_probs,
        keep_indices=keep_idx,
        mask_counts=counts,
        proposals=proposals,
    )
    mean_prob_scores = scores_for_mode(
        mode="mean_mask_prob",
        learned_scores=learned,
        mask_probs=mask_probs,
        keep_indices=keep_idx,
        mask_counts=counts,
        proposals=proposals,
    )

    assert learned_scores.tolist() == [0.1, 0.8, 0.4]
    assert one_scores.tolist() == [1.0, 1.0, 1.0]
    assert area_scores.tolist() == [0.5, 0.5, 0.25]
    np.testing.assert_allclose(mean_prob_scores, [0.85, 0.75, 0.60], rtol=1e-6)
    assert np.argsort(-learned_scores).tolist() == [1, 2, 0]
    assert np.argsort(-mean_prob_scores).tolist() == [0, 1, 2]


def test_oracle_score_replacement_emits_recoverable_ap_fields() -> None:
    gt_ids = np.asarray([1, 1, 2, 2], dtype=np.int64)
    proposals = [
        np.asarray([True, False, True, False], dtype=bool),
        np.asarray([True, True, False, False], dtype=bool),
        np.asarray([False, False, True, True], dtype=bool),
    ]
    records = build_instance_ap_records(
        scene_id="scene0000_00",
        gt_ids=gt_ids,
        proposals=proposals,
        scores=np.asarray([0.99, 0.2, 0.1], dtype=np.float64),
        query_indices=np.arange(3, dtype=np.int64),
        class_agnostic=True,
    )

    learned_metrics = evaluate_official_and_oracle_ap(records, thresholds=[0.5, 0.25])
    oracle_records = _records_with_oracle_as_score(records)
    oracle_metrics = evaluate_official_and_oracle_ap(oracle_records, thresholds=[0.5, 0.25])

    assert learned_metrics["AP50"] < 1.0
    assert learned_metrics["oracle_AP50"] == 1.0
    assert oracle_metrics["AP50"] == 1.0
    assert oracle_metrics["oracle_AP50"] == 1.0
