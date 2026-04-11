from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN

from chorus.common.progress import heartbeat, log_progress


def cluster_features(
    features: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
) -> tuple[np.ndarray, dict]:
    log_progress(
        "HDBSCAN input: "
        f"samples={features.shape[0]}, dims={features.shape[1]}, "
        f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
        f"cluster_selection_epsilon={cluster_selection_epsilon}"
    )
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        n_jobs=-1,
        copy=False,
    )
    with heartbeat("HDBSCAN fit_predict"):
        labels = clusterer.fit_predict(features)

    num_noise = int(np.sum(labels < 0))
    num_clusters = int(len(np.unique(labels[labels >= 0])))

    stats = {
        "num_clusters": num_clusters,
        "num_noise_points": num_noise,
        "noise_fraction": float(num_noise / max(len(labels), 1)),
    }
    log_progress(
        "HDBSCAN output: "
        f"num_clusters={num_clusters}, num_noise_points={num_noise}, "
        f"noise_fraction={stats['noise_fraction']:.4f}"
    )
    return labels, stats