from __future__ import annotations

import math

import numpy as np

from student.metrics.official_instance_ap import (
    SCANNET_IGNORE_MODE,
    SCANNET_MIN_REGION_SIZE,
    STRICT_IGNORE_MODE,
    build_instance_ap_records,
    evaluate_official_and_oracle_ap,
    evaluate_official_instance_ap,
    get_iou_thresholds,
    merge_ap_record_sets,
)


def _records(
    scene_id: str,
    gt_ids: list[int],
    proposals: list[list[bool]],
    scores: list[float],
    **kwargs,
) -> dict:
    return build_instance_ap_records(
        scene_id=scene_id,
        gt_ids=np.asarray(gt_ids, dtype=np.int64),
        proposals=[np.asarray(mask, dtype=bool) for mask in proposals],
        scores=np.asarray(scores, dtype=np.float64),
        query_indices=np.arange(len(proposals), dtype=np.int64),
        **kwargs,
    )


def _eval(*records: dict, ignore_mode: str = SCANNET_IGNORE_MODE) -> dict:
    return evaluate_official_and_oracle_ap(
        merge_ap_record_sets(list(records)),
        thresholds=[0.5, 0.25],
        ignore_mode=ignore_mode,
    )


def _reference_scannet_ap_from_events(
    y_true: list[float],
    y_score: list[float],
    hard_false_negatives: int,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    score_arg_sort = np.argsort(y_score_arr)
    y_score_sorted = y_score_arr[score_arg_sort]
    y_true_sorted = y_true_arr[score_arg_sort]
    y_true_sorted_cumsum = np.cumsum(y_true_sorted)
    _, unique_indices = np.unique(y_score_sorted, return_index=True)
    num_prec_recall = len(unique_indices) + 1
    num_examples = len(y_score_sorted)
    num_true_examples = y_true_sorted_cumsum[-1]

    precision = np.zeros(num_prec_recall, dtype=np.float64)
    recall = np.zeros(num_prec_recall, dtype=np.float64)
    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0.0)

    for idx_res, idx_scores in enumerate(unique_indices):
        cumsum = y_true_sorted_cumsum[idx_scores - 1]
        tp = num_true_examples - cumsum
        fp = num_examples - idx_scores - tp
        fn = cumsum + hard_false_negatives
        precision[idx_res] = tp / max(tp + fp, 1e-12)
        recall[idx_res] = tp / max(tp + fn, 1e-12)

    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_conv = np.copy(recall)
    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
    recall_for_conv = np.append(recall_for_conv, 0.0)
    step_widths = np.convolve(recall_for_conv, [-0.5, 0.0, 0.5], "valid")
    return float(np.dot(precision, step_widths))


def test_scannet_threshold_preset_matches_mask3d_convention() -> None:
    thresholds = get_iou_thresholds("scannet_official")

    assert thresholds == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.25]


def test_perfect_prediction_gives_unit_ap() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [[True, True, False, False], [False, False, True, True]],
        [1.0, 0.9],
    )

    metrics = _eval(records)

    assert metrics["AP"] == 1.0
    assert metrics["AP50"] == 1.0
    assert metrics["AP25"] == 1.0


def test_duplicate_prediction_is_false_positive_and_lowers_ap() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
        ],
        [0.9, 0.8, 0.7],
    )

    metrics = _eval(records)
    detail = metrics["detail"]["AP50"]["object"]

    assert detail["num_tp"] == 2
    assert detail["num_fp"] == 1
    assert 0.0 < metrics["AP50"] < 1.0


def test_bad_ranking_lowers_ap_and_oracle_score_recovers() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, False, True, True],
        ],
        [0.99, 0.2, 0.1],
    )

    metrics = _eval(records)

    assert metrics["AP50"] < 1.0
    assert metrics["oracle_AP50"] == 1.0


def test_global_ap_sorts_predictions_across_scenes() -> None:
    scene_a = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [[True, False, True, False], [True, True, False, False]],
        [0.95, 0.1],
    )
    scene_b = _records(
        "scene0001_00",
        [1, 1],
        [[True, True]],
        [0.5],
    )

    metrics = _eval(scene_a, scene_b)

    assert metrics["detail"]["AP50"]["object"]["num_gt"] == 3
    assert metrics["detail"]["AP50"]["object"]["num_fp"] == 1
    assert 0.0 < metrics["AP50"] < 1.0


def test_class_agnostic_mapping_ignores_semantic_class_ids() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [[True, True, False, False], [False, False, True, True]],
        [0.8, 0.7],
        class_agnostic=True,
        gt_instance_class_ids={1: 4, 2: 17},
    )

    assert _eval(records)["AP50"] == 1.0


def test_class_agnostic_predictions_do_not_require_predicted_class_ids() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [[True, True, False, False], [False, False, True, True]],
        [0.8, 0.7],
        class_agnostic=True,
        gt_instance_class_ids={1: 4, 2: 17},
        pred_class_ids=np.asarray([999, 998], dtype=np.int64),
    )

    assert {pred["class_id"] for pred in records["predictions"]} == {"object"}
    assert _eval(records)["AP50"] == 1.0


def test_class_aware_wrong_class_does_not_match() -> None:
    records = _records(
        "scene0000_00",
        [1, 1],
        [[True, True]],
        [0.9],
        class_agnostic=False,
        gt_instance_class_ids={1: 0},
        pred_class_ids=np.asarray([1], dtype=np.int64),
    )

    metrics = _eval(records)

    assert metrics["AP50"] == 0.0


def test_invalid_scannet_gt_is_excluded_and_void_prediction_ignored() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 0, 0],
        [[False, False, True, True]],
        [0.9],
    )

    metrics = _eval(records)
    detail = metrics["detail"]["AP50"]["object"]

    assert metrics["AP50"] == 0.0
    assert detail["num_gt"] == 1
    assert detail["num_ignored_predictions"] == 1


def test_empty_predictions_with_gt_gives_zero_ap() -> None:
    records = _records("scene0000_00", [1, 1], [], [])

    metrics = _eval(records)

    assert metrics["AP50"] == 0.0


def test_empty_scene_predictions_are_fp_in_strict_mode_when_dataset_has_gt() -> None:
    empty_scene = _records("scene0000_00", [0, 0], [[True, True]], [0.99])
    gt_scene = _records("scene0001_00", [1, 1], [[True, True]], [0.5])

    metrics = _eval(empty_scene, gt_scene, ignore_mode=STRICT_IGNORE_MODE)

    assert metrics["detail"]["AP50"]["object"]["num_fp"] == 1
    assert metrics["AP50"] < 1.0


def test_whole_bucket_with_zero_gt_returns_nan() -> None:
    records = _records("scene0000_00", [0, 0], [[True, True]], [0.9])

    metrics = _eval(records, ignore_mode=STRICT_IGNORE_MODE)

    assert math.isnan(metrics["AP50"])
    assert math.isnan(metrics["AP"])


def test_threshold_sweep_ap25_and_ap50_boundaries() -> None:
    exact_half = _records(
        "scene0000_00",
        [1, 1, 1, 1],
        [[True, True, False, False]],
        [0.9],
    )
    quarter = _records(
        "scene0001_00",
        [1, 1, 1, 1],
        [[True, False, False, False]],
        [0.9],
    )

    half_metrics = evaluate_official_instance_ap(
        exact_half,
        thresholds=[0.5, 0.55, 0.25],
        ignore_mode=STRICT_IGNORE_MODE,
    )
    scannet_half_metrics = evaluate_official_instance_ap(
        exact_half,
        thresholds=[0.5, 0.25],
        ignore_mode=SCANNET_IGNORE_MODE,
    )
    quarter_metrics = _eval(quarter, ignore_mode=STRICT_IGNORE_MODE)

    assert half_metrics["AP50"] == 1.0
    assert half_metrics["by_threshold"]["AP55"] == 0.0
    assert half_metrics["AP"] == 0.5
    assert scannet_half_metrics["AP50"] == 0.0
    assert quarter_metrics["AP25"] == 1.0
    assert quarter_metrics["AP50"] == 0.0


def test_scannet_min_region_size_treats_small_gt_as_ignored() -> None:
    gt_ids = [1] * SCANNET_MIN_REGION_SIZE + [2, 2]
    valid_pred = [True] * SCANNET_MIN_REGION_SIZE + [False, False]
    small_pred = [False] * SCANNET_MIN_REGION_SIZE + [True, True]
    records = _records(
        "scene0000_00",
        gt_ids,
        [small_pred, valid_pred],
        [0.99, 0.5],
        min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
        min_valid_pred_points=1,
    )

    metrics = _eval(records)
    detail = metrics["detail"]["AP50"]["object"]

    assert detail["num_gt"] == 1
    assert detail["num_ignored_predictions"] == 1
    assert metrics["AP50"] == 1.0


def test_pseudo_gt_as_prediction_respects_supervision_mask() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 0, 0],
        [[True, True, True, True]],
        [0.9],
        eval_mask=np.asarray([True, True, False, False]),
    )

    metrics = _eval(records)

    assert metrics["AP50"] == 1.0
    assert metrics["AP25"] == 1.0


def test_reference_scannet_ap_integration_equivalence() -> None:
    records = _records(
        "scene0000_00",
        [1, 1, 2, 2],
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
        ],
        [0.9, 0.8, 0.7],
    )

    metrics = _eval(records)
    reference = _reference_scannet_ap_from_events(
        y_true=[1.0, 0.0, 1.0],
        y_score=[0.9, 0.8, 0.7],
        hard_false_negatives=0,
    )

    np.testing.assert_allclose(metrics["AP50"], reference, atol=1e-12)


def test_reference_scannet_matching_and_ignore_equivalence() -> None:
    gt_ids = [1] * SCANNET_MIN_REGION_SIZE + [2, 2, 0, 0]
    valid = [True] * SCANNET_MIN_REGION_SIZE + [False, False, False, False]
    duplicate = list(valid)
    small_ignored = [False] * SCANNET_MIN_REGION_SIZE + [True, True, False, False]
    void_ignored = [False] * (SCANNET_MIN_REGION_SIZE + 2) + [True, True]
    false_positive = [True] + [False] * (SCANNET_MIN_REGION_SIZE + 3)
    records = _records(
        "scene0000_00",
        gt_ids,
        [small_ignored, void_ignored, valid, duplicate, false_positive],
        [0.99, 0.95, 0.8, 0.7, 0.6],
        min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
        min_valid_pred_points=1,
    )

    metrics = _eval(records)
    reference = _reference_scannet_ap_from_events(
        y_true=[1.0, 0.0, 0.0],
        y_score=[0.8, 0.7, 0.6],
        hard_false_negatives=0,
    )

    detail = metrics["detail"]["AP50"]["object"]
    assert detail["num_ignored_predictions"] == 2
    assert detail["num_tp"] == 1
    assert detail["num_fp"] == 2
    np.testing.assert_allclose(metrics["AP50"], reference, atol=1e-12)
