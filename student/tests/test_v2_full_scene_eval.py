from __future__ import annotations

import pytest
import torch.nn as nn

from student.data.eval_sampling import resolve_eval_sampling_config
from student.engine.multi_scene_evaluator import evaluate_multi_scene


def test_eval_sampling_overrides_train_sphere_crop_to_full_scene() -> None:
    data_cfg = {"max_points": 60000, "subsampling_mode": "sphere_crop"}
    eval_cfg = {"max_points": None, "subsampling_mode": "none", "sphere_point_max": None}

    resolved = resolve_eval_sampling_config(data_cfg, eval_cfg)

    assert resolved == {
        "max_points": None,
        "subsampling_mode": "none",
        "sphere_point_max": None,
    }


def test_eval_sampling_defaults_to_full_scene_without_eval_keys() -> None:
    resolved = resolve_eval_sampling_config({"max_points": 60000, "subsampling_mode": "sphere_crop"}, {})

    assert resolved["subsampling_mode"] == "none"
    assert resolved["max_points"] is None


class _CroppedDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        import torch

        return {
            "scene_id": "scene0000_00",
            "scene_dir": "/tmp/scene0000_00",
            "points": torch.zeros(4, 3),
            "features": torch.zeros(4, 2),
            "labels_by_granularity": {"g05": torch.tensor([1, 1, -1, -1])},
            "supervision_mask": torch.ones(4, dtype=torch.bool),
            "instance_classes_by_granularity": {"g05": None},
            "eval_input_points": 4,
            "original_num_points": 8,
            "full_scene": False,
        }


def test_eval_only_requires_full_scene_when_fragment_merge_disabled() -> None:
    with pytest.raises(RuntimeError, match="full-scene validation"):
        evaluate_multi_scene(
            model=nn.Module(),
            dataset=_CroppedDataset(),  # type: ignore[arg-type]
            criterion=object(),  # type: ignore[arg-type]
            device="cpu",
            granularities=("g05",),
            require_full_scene=True,
            fragment_merge_eval=False,
        )
