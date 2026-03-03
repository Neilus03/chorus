
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.common.types import ClusterOutput
from chorus.eval.scannet_oracle import (
    build_oracle_best_labels,
    build_proposals_from_cluster_outputs,
    compute_additional_oracle_metrics,
)


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


def test_build_proposals_from_cluster_outputs_pools_clusters_across_granularities() -> None:
    cluster_outputs = [
        _make_cluster_output(
            0.2,
            np.array([0, 0, -1, 1, -1], dtype=np.int32),
        ),
        _make_cluster_output(
            0.8,
            np.array([-1, 0, 0, -1, 1], dtype=np.int32),
        ),
    ]

    proposals, proposal_sources = build_proposals_from_cluster_outputs(cluster_outputs)

    assert len(proposals) == 4
    assert proposal_sources == [0.2, 0.2, 0.8, 0.8]

    np.testing.assert_array_equal(
        proposals[0],
        np.array([True, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        proposals[1],
        np.array([False, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        proposals[2],
        np.array([False, True, True, False, False]),
    )
    np.testing.assert_array_equal(
        proposals[3],
        np.array([False, False, False, False, True]),
    )


def test_build_oracle_best_labels_picks_best_iou_proposal_per_gt() -> None:
    gt_ids = np.array([1, 1, 0, 2, 2, 2], dtype=np.int32)

    cluster_outputs = [
        _make_cluster_output(
            0.2,
            np.array([0, 0, -1, 1, -1, -1], dtype=np.int32),
        ),
        _make_cluster_output(
            0.8,
            np.array([-1, -1, -1, 0, 0, 0], dtype=np.int32),
        ),
    ]

    proposals, _ = build_proposals_from_cluster_outputs(cluster_outputs)

    oracle_labels = build_oracle_best_labels(
        gt_ids=gt_ids,
        proposals=proposals,
        min_iou=0.5,
    )

    expected = np.array([1, 1, -1, 2, 2, 2], dtype=np.int32)
    np.testing.assert_array_equal(oracle_labels, expected)


def test_compute_additional_oracle_metrics_tracks_winner_granularity_share() -> None:
    gt_ids = np.array([1, 1, 0, 2, 2, 2], dtype=np.int32)

    cluster_outputs = [
        _make_cluster_output(
            0.2,
            np.array([0, 0, -1, 1, -1, -1], dtype=np.int32),
        ),
        _make_cluster_output(
            0.8,
            np.array([-1, -1, -1, 0, 0, 0], dtype=np.int32),
        ),
    ]

    proposals, proposal_sources = build_proposals_from_cluster_outputs(cluster_outputs)
    extras = compute_additional_oracle_metrics(
        gt_ids=gt_ids,
        proposals=proposals,
        proposal_sources=proposal_sources,
    )

    assert "winner_granularity_share" in extras
    winner_share = extras["winner_granularity_share"]

    assert winner_share["g0.2"] == pytest.approx(0.5)
    assert winner_share["g0.8"] == pytest.approx(0.5)
    assert winner_share["no_match"] == pytest.approx(0.0)

    assert extras["topk_proposal_coverage"]["iou_0.25"]["R_at_least_1"] == pytest.approx(1.0)
    assert extras["topk_proposal_coverage"]["iou_0.50"]["R_at_least_1"] == pytest.approx(1.0)

if __name__ == "__main__":
    pytest.main([__file__])