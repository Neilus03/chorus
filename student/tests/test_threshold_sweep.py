from __future__ import annotations

import inspect

import numpy as np

from student.engine.evaluator import _score_threshold_for_granularity
from student.engine import threshold_sweep
from student.engine.multi_scene_trainer import MultiSceneTrainer


def _pack(masks: list[list[bool]]) -> np.ndarray:
    return np.packbits(np.asarray(masks, dtype=bool), axis=1, bitorder="little")


def _tiny_cache() -> dict:
    masks = [
        [True, True, False, False],
        [False, False, True, True],
        [True, False, False, False],
    ]
    return {
        "version": threshold_sweep.CACHE_VERSION,
        "cache_is_pre_score_threshold": True,
        "granularities": ["g05"],
        "eval_benchmarks": [],
        "settings": {
            "mask_threshold": 0.5,
            "min_points_per_proposal": 1,
        },
        "scenes": [
            {
                "scene_id": "scene0000_00",
                "scene_dir": "/tmp/scene0000_00",
                "num_points": 4,
                "eval_scope": "full_scene",
                "real_gt_by_benchmark": {},
                "heads": {
                    "g05": {
                        "num_points": 4,
                        "num_queries": 3,
                        "score_logits": np.asarray([2.0, -2.0, -0.2], dtype=np.float32),
                        "score_probs": np.asarray([0.9, 0.05, 0.2], dtype=np.float32),
                        "query_indices": np.arange(3, dtype=np.int64),
                        "mask_point_counts": np.asarray([2, 2, 1], dtype=np.int32),
                        "packed_masks": _pack(masks),
                        "packed_bitorder": "little",
                        "pseudo_gt_ids": np.asarray([1, 1, 2, 2], dtype=np.int32),
                        "pseudo_supervision_mask": np.asarray([True, True, True, True]),
                    },
                },
            }
        ],
    }


def test_lower_threshold_keeps_at_least_as_many_proposals() -> None:
    rows = threshold_sweep.sweep_thresholds_from_cache(_tiny_cache(), [0.5, 0.1, 0.0])
    by_threshold = {row["score_threshold"]: row for row in rows}

    assert by_threshold[0.0]["mean_kept_per_scene"] >= by_threshold[0.1]["mean_kept_per_scene"]
    assert by_threshold[0.1]["mean_kept_per_scene"] >= by_threshold[0.5]["mean_kept_per_scene"]
    assert by_threshold[0.0]["pseudo_official_num_predictions"] >= by_threshold[0.5]["pseudo_official_num_predictions"]


def test_reuse_cache_does_not_call_model_forward_path(tmp_path) -> None:
    cache_path = tmp_path / "cache.pt"
    threshold_sweep.save_prediction_cache(_tiny_cache(), cache_path)
    called = False

    def build_cache() -> dict:
        nonlocal called
        called = True
        raise AssertionError("cache builder should not run when --reuse-cache succeeds")

    cache, reused = threshold_sweep.load_or_create_prediction_cache(
        cache_path=cache_path,
        reuse_cache=True,
        write_cache=True,
        build_cache_fn=build_cache,
    )

    assert reused is True
    assert called is False
    assert cache["cache_is_pre_score_threshold"] is True


def test_threshold_sweep_is_not_wired_into_default_training_path() -> None:
    trainer_source = inspect.getsource(MultiSceneTrainer)
    cache_builder_source = inspect.getsource(threshold_sweep.build_prediction_cache)

    assert "threshold_sweep" not in trainer_source
    assert "optimizer" not in cache_builder_source
    assert ".step(" not in cache_builder_source


def test_eval_score_threshold_dict_selects_per_granularity_values() -> None:
    thresholds = {"g02": 0.0, "g05": 0.2, "default": 0.4}

    assert _score_threshold_for_granularity(thresholds, "g02") == 0.0
    assert _score_threshold_for_granularity(thresholds, "g05") == 0.2
    assert _score_threshold_for_granularity(thresholds, "g08") == 0.4
    assert _score_threshold_for_granularity(0.7, "g02") == 0.7
