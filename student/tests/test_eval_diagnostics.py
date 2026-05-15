from __future__ import annotations

import json

import numpy as np

from student.metrics.eval_diagnostics import (
    build_diagnostic_report,
    build_scene_mask_diagnostics,
    duplicate_summary_from_iou,
    filter_records_topk_by_scene,
    matched_recall_from_records,
    pairwise_mask_iou,
    safe_spearman,
)
from student.metrics.official_instance_ap import build_instance_ap_records, merge_ap_record_sets


def _records(
    scene_id: str,
    gt_ids: list[int],
    proposals: list[list[bool]],
    scores: list[float],
) -> dict:
    return build_instance_ap_records(
        scene_id=scene_id,
        gt_ids=np.asarray(gt_ids, dtype=np.int64),
        proposals=[np.asarray(mask, dtype=bool) for mask in proposals],
        scores=np.asarray(scores, dtype=np.float64),
        query_indices=np.arange(len(proposals), dtype=np.int64),
    )


def test_topk_filtering_over_compact_records_is_per_scene() -> None:
    scene_a = _records(
        "scene0000_00",
        [1, 1],
        [[True, True], [True, False], [False, True]],
        [0.2, 0.9, 0.1],
    )
    scene_b = _records(
        "scene0001_00",
        [1, 1],
        [[True, True], [True, False]],
        [0.4, 0.8],
    )
    merged = merge_ap_record_sets([scene_a, scene_b])

    filtered = filter_records_topk_by_scene(merged, k=1, score_key="score")

    assert filtered["num_predictions"] == 2
    assert {
        (pred["scene_id"], pred["pred_id"])
        for pred in filtered["predictions"]
    } == {("scene0000_00", 1), ("scene0001_00", 1)}


def test_recall_from_compact_records_counts_gt_with_any_kept_match() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [[True, True, False, False]],
        [0.9],
    )

    recall = matched_recall_from_records(records)

    assert recall["recall50"] == 0.5
    assert recall["recall25"] == 0.5


def test_nms_suppresses_duplicate_masks_and_reduces_duplicate_pairs() -> None:
    proposals = [
        np.asarray([True, True, False, False]),
        np.asarray([True, True, False, False]),
        np.asarray([False, False, True, True]),
    ]
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [p.tolist() for p in proposals],
        [0.9, 0.8, 0.7],
    )
    iou = pairwise_mask_iou(proposals)
    before = duplicate_summary_from_iou(iou)["0.5"]["duplicate_pairs"]

    diagnostics = build_scene_mask_diagnostics(records, proposals, nms_thresholds=[0.5])
    kept_pred_ids = diagnostics["nms_keep_pred_ids_by_score"]["0.5"]
    kept_iou = iou[np.ix_(kept_pred_ids, kept_pred_ids)]
    after = duplicate_summary_from_iou(kept_iou)["0.5"]["duplicate_pairs"]

    assert kept_pred_ids == [0, 2]
    assert before == 1
    assert after == 0


def test_score_iou_correlation_handles_constant_scores() -> None:
    assert safe_spearman([0.5, 0.5, 0.5], [0.0, 0.5, 1.0]) is None


def test_diagnostic_json_can_be_built_from_tiny_two_scene_setup() -> None:
    scene_a_props = [
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
    ]
    scene_b_props = [
        np.asarray([True, True]),
        np.asarray([False, True]),
    ]
    scene_a = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [p.tolist() for p in scene_a_props],
        [0.2, 0.9],
    )
    scene_b = _records(
        "scene0001_00",
        [1, 1],
        [p.tolist() for p in scene_b_props],
        [0.8, 0.1],
    )
    scene_outputs = [
        {
            "scene_id": "scene0000_00",
            "records": scene_a,
            "legacy": {
                "legacy_matched_recall25": 1.0,
                "legacy_matched_recall50": 0.5,
                "matched_mean_iou": 0.5,
            },
            "proposal_stats": {"num_queries": 2, "num_min_points_removed": 0},
            "mask_diagnostics": build_scene_mask_diagnostics(scene_a, scene_a_props, nms_thresholds=[0.5]),
            "eval_scope": "full_scene",
        },
        {
            "scene_id": "scene0001_00",
            "records": scene_b,
            "legacy": {
                "legacy_matched_recall25": 1.0,
                "legacy_matched_recall50": 1.0,
                "matched_mean_iou": 1.0,
            },
            "proposal_stats": {"num_queries": 2, "num_min_points_removed": 0},
            "mask_diagnostics": build_scene_mask_diagnostics(scene_b, scene_b_props, nms_thresholds=[0.5]),
            "eval_scope": "full_scene",
        },
    ]

    report = build_diagnostic_report(
        scene_outputs=scene_outputs,
        checkpoint="best.pt",
        config="config.yaml",
        benchmark="scannet20",
        granularity="g05",
        nms_thresholds=[0.5],
        topk_values=[1, 2],
        settings={},
    )

    assert report["baseline"]["num_gt"] == 3
    assert "score_iou" in report
    assert "topk_by_score" in report
    assert "duplicates" in report
    assert "nms_by_score" in report
    assert "diagnosis" in report
    json.dumps(report)
