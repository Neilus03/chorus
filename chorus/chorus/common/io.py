from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chorus.eval.base import DatasetEvaluationHooks


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    return data


def _granularity_tag(granularity: float) -> str:
    return f"g{granularity}"


def _validate_training_pack_scene_meta(pack_dir: Path) -> list[str]:
    scene_meta_path = pack_dir / "scene_meta.json"
    scene_meta = load_json_if_exists(scene_meta_path)
    if scene_meta is None:
        return [f"{scene_meta_path} (invalid json)"]

    required_fields = [
        "pack_version",
        "label_convention",
        "supervision_mask_definition",
        "valid_points_definition",
        "seen_points_definition",
        "coordinate_units",
        "coordinate_frame",
        "point_source",
        "optional_files_present",
    ]

    missing_fields = [field for field in required_fields if field not in scene_meta]
    if not missing_fields:
        return []

    return [
        "training pack scene_meta.json missing required fields: "
        + ", ".join(sorted(missing_fields))
    ]


def expected_scene_output_paths(
    scene_dir: Path,
    granularities: list[float],
    require_oracle: bool = True,
    require_training_pack: bool = True,
    evaluation_hooks: DatasetEvaluationHooks | None = None,
) -> list[Path]:
    scene_dir = Path(scene_dir)
    expected: list[Path] = []

    expected.append(scene_dir / "scene_pipeline_summary.json")

    for granularity in granularities:
        tag = str(granularity)
        expected.extend(
            [
                scene_dir / f"chorus_instance_labels_g{tag}.npy",
                scene_dir / f"chorus_instance_result_g{tag}.ply",
                scene_dir / f"diagnostics_g{tag}.json",
            ]
        )

    if evaluation_hooks is not None:
        expected.extend(
            evaluation_hooks.expected_output_paths(
                scene_dir=scene_dir,
                granularities=granularities,
                require_oracle=require_oracle,
                require_training_pack=require_training_pack,
            )
        )

    if require_training_pack:
        # Require the canonical `training_pack/` export dir. Legacy `litept_pack/` alone must not
        # satisfy completion checks, or streaming will skip scenes that still need a v1 training pack.
        training_pack_dir = scene_dir / "training_pack"
        expected.extend(
            [
                training_pack_dir / "points.npy",
                training_pack_dir / "valid_points.npy",
                training_pack_dir / "seen_points.npy",
                training_pack_dir / "supervision_mask.npy",
                training_pack_dir / "scene_meta.json",
            ]
        )
        for granularity in granularities:
            expected.append(training_pack_dir / f"labels_{_granularity_tag(granularity)}.npy")

    return expected


def verify_existing_scene_outputs(
    scene_dir: Path,
    granularities: list[float],
    require_oracle: bool = True,
    require_training_pack: bool = True,
    evaluation_hooks: DatasetEvaluationHooks | None = None,
) -> tuple[bool, list[str]]:
    missing_or_empty: list[str] = []

    for path in expected_scene_output_paths(
        scene_dir=scene_dir,
        granularities=granularities,
        require_oracle=require_oracle,
        require_training_pack=require_training_pack,
        evaluation_hooks=evaluation_hooks,
    ):
        if not path.exists():
            missing_or_empty.append(str(path))
            continue

        if path.is_file() and path.stat().st_size == 0:
            missing_or_empty.append(f"{path} (empty)")

    return len(missing_or_empty) == 0, missing_or_empty


def verify_scene_completion_from_summary(
    scene_dir: Path,
    granularities: list[float],
    require_oracle: bool = True,
    require_training_pack: bool = True,
    evaluation_hooks: DatasetEvaluationHooks | None = None,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    scene_dir = Path(scene_dir)
    summary_path = scene_dir / "scene_pipeline_summary.json"
    summary = load_json_if_exists(summary_path)

    if summary is None:
        return False, None, [str(summary_path)]

    missing_reasons: list[str] = []

    if summary.get("scene_id") != scene_dir.name:
        missing_reasons.append(
            f"summary scene_id mismatch: expected {scene_dir.name}, found {summary.get('scene_id')}"
        )

    summary_granularities = summary.get("granularities")
    if summary_granularities is None:
        missing_reasons.append("summary missing granularities")
    else:
        try:
            summary_g = sorted(float(g) for g in summary_granularities)
            requested_g = sorted(float(g) for g in granularities)
            if summary_g != requested_g:
                missing_reasons.append(
                    f"summary granularities mismatch: expected {requested_g}, found {summary_g}"
                )
        except Exception:
            missing_reasons.append("summary granularities could not be parsed")

    if require_training_pack:
        if not summary.get("training_pack_dir"):
            missing_reasons.append("summary missing training_pack_dir")

    if evaluation_hooks is not None:
        missing_reasons.extend(
            evaluation_hooks.verify_summary(
                scene_dir=scene_dir,
                summary=summary,
                granularities=granularities,
                require_oracle=require_oracle,
                require_training_pack=require_training_pack,
            )
        )

    outputs_ok, missing_outputs = verify_existing_scene_outputs(
        scene_dir=scene_dir,
        granularities=granularities,
        require_oracle=require_oracle,
        require_training_pack=require_training_pack,
        evaluation_hooks=evaluation_hooks,
    )
    if not outputs_ok:
        missing_reasons.extend(missing_outputs)

    if require_training_pack:
        training_pack_dir = scene_dir / "training_pack"
        if training_pack_dir.exists():
            missing_reasons.extend(_validate_training_pack_scene_meta(training_pack_dir))

    return len(missing_reasons) == 0, summary, missing_reasons