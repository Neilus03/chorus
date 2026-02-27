from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN


def cluster_features(
    features: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
) -> tuple[np.ndarray, dict]:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(features)

    num_noise = int(np.sum(labels < 0))
    num_clusters = int(len(np.unique(labels[labels >= 0])))

    stats = {
        "num_clusters": num_clusters,
        "num_noise_points": num_noise,
        "noise_fraction": float(num_noise / max(len(labels), 1)),
    }
    return labels, stats