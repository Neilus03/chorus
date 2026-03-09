from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chorus.common.io import verify_existing_scene_outputs, verify_scene_completion_from_summary


def _write_minimal_scene_outputs(
    scene_dir: Path,
    *,
    include_seen_points: bool = True,
    scene_meta: dict | None = None,
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    training_pack_dir = scene_dir / "training_pack"
    training_pack_dir.mkdir(parents=True, exist_ok=True)

    (scene_dir / "scene_pipeline_summary.json").write_text(
        json.dumps(
            {
                "scene_id": scene_dir.name,
                "granularities": [0.2],
                "training_pack_dir": str(training_pack_dir),
            }
        ),
        encoding="utf-8",
    )
    np.save(scene_dir / "chorus_instance_labels_g0.2.npy", np.array([0, -1], dtype=np.int32))
    (scene_dir / "chorus_instance_result_g0.2.ply").write_text("ply\n", encoding="utf-8")
    (scene_dir / "diagnostics_g0.2.json").write_text("{}", encoding="utf-8")

    np.save(training_pack_dir / "points.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(training_pack_dir / "valid_points.npy", np.array([1, 0], dtype=np.uint8))
    if include_seen_points:
        np.save(training_pack_dir / "seen_points.npy", np.array([1, 0], dtype=np.uint8))
    np.save(training_pack_dir / "supervision_mask.npy", np.array([1, 0], dtype=np.uint8))
    np.save(training_pack_dir / "labels_g0.2.npy", np.array([0, -1], dtype=np.int32))
    (training_pack_dir / "scene_meta.json").write_text(
        json.dumps(scene_meta if scene_meta is not None else {}),
        encoding="utf-8",
    )


def test_verify_existing_scene_outputs_requires_seen_points(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene0000_00"
    _write_minimal_scene_outputs(scene_dir, include_seen_points=False)

    outputs_ok, missing = verify_existing_scene_outputs(
        scene_dir=scene_dir,
        granularities=[0.2],
        require_oracle=False,
        require_training_pack=True,
    )

    assert not outputs_ok
    assert any(str(scene_dir / "training_pack" / "seen_points.npy") in item for item in missing)


def test_verify_scene_completion_checks_training_pack_contract_fields(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene0000_00"
    _write_minimal_scene_outputs(
        scene_dir,
        include_seen_points=True,
        scene_meta={
            "label_convention": {"ignore_unlabeled": -1},
            "supervision_mask_definition": "same as valid points",
            "valid_points_definition": "label >= 0",
            "seen_points_definition": "observed in at least one frame",
            "coordinate_units": "meters",
            "coordinate_frame": "scene coordinates",
            "point_source": "mesh_vertices",
            "optional_files_present": {"colors.npy": False},
        },
    )

    is_complete, _, missing_reasons = verify_scene_completion_from_summary(
        scene_dir=scene_dir,
        granularities=[0.2],
        require_oracle=False,
        require_training_pack=True,
    )

    assert not is_complete
    assert any("pack_version" in reason for reason in missing_reasons)
