from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def expected_scene_output_paths(
    scene_dir: Path,
    granularities: list[float],
    require_oracle: bool = True,
    require_litept: bool = True,
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

    if require_oracle:
        expected.extend(
            [
                scene_dir / "oracle_metrics.json",
                scene_dir / "chorus_oracle_best_combined_labels.npy",
                scene_dir / "chorus_oracle_best_combined.ply",
            ]
        )

    if require_litept:
        litept_dir = scene_dir / "litept_pack"
        expected.extend(
            [
                litept_dir / "points.npy",
                litept_dir / "valid_points.npy",
                litept_dir / "supervision_mask.npy",
                litept_dir / "scene_meta.json",
            ]
        )
        for granularity in granularities:
            expected.append(litept_dir / f"labels_{_granularity_tag(granularity)}.npy")

    return expected


def verify_existing_scene_outputs(
    scene_dir: Path,
    granularities: list[float],
    require_oracle: bool = True,
    require_litept: bool = True,
) -> tuple[bool, list[str]]:
    missing_or_empty: list[str] = []

    for path in expected_scene_output_paths(
        scene_dir=scene_dir,
        granularities=granularities,
        require_oracle=require_oracle,
        require_litept=require_litept,
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
    require_litept: bool = True,
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

    if require_oracle and summary.get("oracle_summary") is None:
        missing_reasons.append("summary missing oracle_summary")

    if require_litept and summary.get("litept_pack_dir") is None:
        missing_reasons.append("summary missing litept_pack_dir")

    outputs_ok, missing_outputs = verify_existing_scene_outputs(
        scene_dir=scene_dir,
        granularities=granularities,
        require_oracle=require_oracle,
        require_litept=require_litept,
    )
    if not outputs_ok:
        missing_reasons.extend(missing_outputs)

    return len(missing_reasons) == 0, summary, missing_reasons