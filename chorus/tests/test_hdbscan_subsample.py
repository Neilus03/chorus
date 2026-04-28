from __future__ import annotations

import numpy as np

from chorus.core.clustering.hdbscan_subsample import cluster_features_with_subsample_cap


def test_subsample_cap_delegates_when_n_small() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((50, 4))
    labels, stats = cluster_features_with_subsample_cap(
        features,
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.05,
        max_samples=200,
        rng=rng,
    )
    assert labels.shape == (50,)
    assert stats["hdbscan_subsample_skipped"] is True
    assert int(stats["hdbscan_num_samples"]) == 50


def test_subsample_cap_propagates_to_all_rows() -> None:
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((5, 3))
    idx = rng.integers(0, 5, size=400)
    features = centers[idx] + rng.standard_normal((400, 3)) * 0.02
    labels, stats = cluster_features_with_subsample_cap(
        features.astype(np.float64),
        min_cluster_size=15,
        min_samples=3,
        cluster_selection_epsilon=0.05,
        max_samples=120,
        rng=rng,
    )
    assert labels.shape == (400,)
    assert stats["hdbscan_subsample_skipped"] is False
    assert stats["hdbscan_subsample_used"] == 120
    assert stats["hdbscan_propagate_nn_seconds"] >= 0.0
