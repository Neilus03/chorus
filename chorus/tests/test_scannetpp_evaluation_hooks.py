from __future__ import annotations

from pathlib import Path

from chorus.datasets.scannetpp.evaluation import ScanNetPPEvaluationHooks


def test_expected_output_paths_include_scannetpp_benchmark_suffix(tmp_path: Path) -> None:
    hooks = ScanNetPPEvaluationHooks("top100_instance")

    paths = hooks.expected_output_paths(
        scene_dir=tmp_path,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_training_pack=True,
    )

    path_strings = {str(path) for path in paths}
    assert str(tmp_path / "oracle_metrics_top100_instance.json") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined_labels_top100_instance.npy") in path_strings
    assert str(tmp_path / "chorus_oracle_best_combined_top100_instance.ply") in path_strings


def test_verify_summary_checks_requested_benchmark(tmp_path: Path) -> None:
    hooks = ScanNetPPEvaluationHooks("top100_instance")

    summary = {
        "scene_id": "abcd1234",
        "granularities": [0.2, 0.5, 0.8],
        "oracle_summary": {"eval_benchmark": "all"},
        "eval_benchmark": "all",
        "training_pack_dir": str(tmp_path / "training_pack"),
    }

    errors = hooks.verify_summary(
        scene_dir=tmp_path,
        summary=summary,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_training_pack=True,
    )

    assert any("eval_benchmark mismatch" in error for error in errors)
