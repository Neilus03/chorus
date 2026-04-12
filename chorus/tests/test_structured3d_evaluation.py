from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.common.types import ClusterOutput
from chorus.datasets.structured3d.adapter import Structured3DSceneAdapter
from chorus.datasets.structured3d.evaluation import Structured3DEvaluationHooks


def _write_minimal_vertex_ply(path: Path, n_vertices: int) -> None:
    vertex = np.zeros(
        n_vertices,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    vertex["x"] = np.arange(n_vertices, dtype=np.float32)
    vertex["y"] = 0.0
    vertex["z"] = 0.0
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def _make_cluster_output(granularity: float, labels: np.ndarray) -> ClusterOutput:
    labels = np.asarray(labels, dtype=np.int32)
    return ClusterOutput(
        granularity=granularity,
        labels=labels,
        features=np.zeros((labels.shape[0], 1), dtype=np.float32),
        seen_mask=np.ones(labels.shape[0], dtype=bool),
        ply_path=None,
        labels_path=None,
        stats={"num_clusters": int(len(np.unique(labels[labels >= 0])))},
    )


def test_expected_output_paths_skip_oracle_when_no_gt_file(tmp_path: Path) -> None:
    hooks = Structured3DEvaluationHooks("structured3d_full")
    paths = hooks.expected_output_paths(
        scene_dir=tmp_path,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_training_pack=True,
    )
    assert paths == []


def test_expected_output_paths_include_benchmark_suffix(tmp_path: Path) -> None:
    (tmp_path / "gt_instance_ids.npy").touch()
    hooks = Structured3DEvaluationHooks("structured3d_full")
    paths = hooks.expected_output_paths(
        scene_dir=tmp_path,
        granularities=[0.2, 0.5, 0.8],
        require_oracle=True,
        require_training_pack=True,
    )
    path_strings = {str(p) for p in paths}
    assert str(tmp_path / "oracle_metrics_structured3d_full.json") in path_strings


def test_evaluate_scene_skips_with_reason_when_no_gt(tmp_path: Path) -> None:
    scene_id = "scene_test"
    scene_root = tmp_path / scene_id
    scene_root.mkdir()
    _write_minimal_vertex_ply(scene_root / f"{scene_id}_vh_clean_2.ply", n_vertices=3)

    adapter = Structured3DSceneAdapter(scene_root=scene_root)
    hooks = Structured3DEvaluationHooks("structured3d_full")
    cluster_outputs = [
        _make_cluster_output(0.2, np.array([0, 0, 1], dtype=np.int32)),
    ]

    out = hooks.evaluate_scene(adapter=adapter, cluster_outputs=cluster_outputs)
    assert out is not None
    assert out["oracle_skipped_reason"] == "missing_gt_instance_ids"
    assert out["oracle_summary"] is None


def test_evaluate_scene_runs_oracle_with_gt(tmp_path: Path) -> None:
    scene_id = "scene_test"
    scene_root = tmp_path / scene_id
    scene_root.mkdir()
    n = 4
    _write_minimal_vertex_ply(scene_root / f"{scene_id}_vh_clean_2.ply", n_vertices=n)
    np.save(
        scene_root / "gt_instance_ids.npy",
        np.array([0, 1, 1, 2], dtype=np.int32),
    )

    adapter = Structured3DSceneAdapter(scene_root=scene_root)
    hooks = Structured3DEvaluationHooks("structured3d_full")
    cluster_outputs = [
        _make_cluster_output(0.2, np.array([0, 0, 0, 1], dtype=np.int32)),
        _make_cluster_output(0.5, np.array([1, 1, 1, 0], dtype=np.int32)),
    ]

    out = hooks.evaluate_scene(adapter=adapter, cluster_outputs=cluster_outputs)
    assert out is not None
    assert "oracle_skipped_reason" not in out
    assert out["oracle_summary"] is not None
    metrics_path = scene_root / "oracle_metrics_structured3d_full.json"
    assert metrics_path.exists()


def test_structured3d_oracle_raises_on_label_length_mismatch(tmp_path: Path) -> None:
    from chorus.eval.scannet_oracle import evaluate_and_save_scannet_oracle

    scene_id = "scene_test"
    scene_root = tmp_path / scene_id
    scene_root.mkdir()
    _write_minimal_vertex_ply(scene_root / f"{scene_id}_vh_clean_2.ply", n_vertices=3)
    np.save(scene_root / "gt_instance_ids.npy", np.ones(3, dtype=np.int32))

    adapter = Structured3DSceneAdapter(scene_root=scene_root)
    cluster_outputs = [
        _make_cluster_output(0.2, np.array([0, 1], dtype=np.int32)),
    ]

    with pytest.raises(RuntimeError, match="Structured3D oracle"):
        evaluate_and_save_scannet_oracle(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
            eval_benchmark="structured3d_full",
        )


def test_verify_summary_accepts_skipped_oracle(tmp_path: Path) -> None:
    hooks = Structured3DEvaluationHooks("structured3d_full")
    summary = {
        "eval_benchmark": "structured3d_full",
        "oracle_skipped_reason": "missing_gt_instance_ids",
        "oracle_summary": None,
    }
    errors = hooks.verify_summary(
        scene_dir=tmp_path,
        summary=summary,
        granularities=[0.2],
        require_oracle=True,
        require_training_pack=True,
    )
    assert errors == []


if __name__ == "__main__":
    pytest.main([__file__])
