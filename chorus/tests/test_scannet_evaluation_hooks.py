from __future__ import annotations

from pathlib import Path

from chorus.datasets.scannet.evaluation import ScanNetEvaluationHooks


def test_expected_output_paths_include_both_scannet_benchmarks(tmp_path: Path) -> None:
    hooks = ScanNetEvaluationHooks(["scannet20", "scannet200"])

    paths = hooks.expected_output_paths(
        scene_dir=tmp_path,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_litept=True,
    )

    path_strings = {str(path) for path in paths}
    assert str(tmp_path / "oracle_metrics.json") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined_labels.npy") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined.ply") in path_strings
    assert str(tmp_path / "oracle_metrics_scannet200.json") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined_labels_scannet200.npy") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined_scannet200.ply") in path_strings


def test_verify_summary_checks_requested_benchmarks(tmp_path: Path) -> None:
    hooks = ScanNetEvaluationHooks(["scannet20", "scannet200"])

    summary = {
        "scene_id": "scene0000_00",
        "granularities": [0.2, 0.5, 0.8],
        "oracle_summary": {"eval_benchmark": "scannet20"},
        "oracle_summaries": {
            "scannet20": {"eval_benchmark": "scannet20"},
        },
        "eval_benchmarks": ["scannet20"],
        "litept_pack_dir": str(tmp_path / "litept_pack"),
    }

    errors = hooks.verify_summary(
        scene_dir=tmp_path,
        summary=summary,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_litept=True,
    )

    assert any("eval_benchmarks mismatch" in error for error in errors)
    assert any("missing oracle_summaries" in error for error in errors)
