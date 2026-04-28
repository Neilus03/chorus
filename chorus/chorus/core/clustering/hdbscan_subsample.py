from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

from chorus.common.progress import heartbeat, log_progress

from chorus.core.clustering.hdbscan_cluster import cluster_features


def cluster_features_with_subsample_cap(
    features: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Run HDBSCAN on a random subset of rows, then assign all rows by 1-NN in feature space."""
    features = np.asarray(features)
    n = int(features.shape[0])
    if n <= 0:
        raise ValueError("features must be non-empty")
    cap = int(max(1, min(max_samples, n)))
    if cap >= n:
        labels, stats = cluster_features(
            features,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        out = dict(stats)
        out["hdbscan_subsample_cap"] = cap
        out["hdbscan_subsample_used"] = n
        out["hdbscan_subsample_skipped"] = True
        out["hdbscan_propagate_nn_seconds"] = 0.0
        return labels, out

    idx = rng.choice(n, size=cap, replace=False)
    sub = features[idx]
    labels_sub, sub_stats = cluster_features(
        sub,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    prop_start = time.perf_counter()
    with heartbeat("HDBSCAN subsample 1-NN propagate"):
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(sub)
        _, neigh = nn.kneighbors(features, return_distance=True)
    prop_elapsed_s = float(time.perf_counter() - prop_start)

    labels = np.asarray(labels_sub[neigh.reshape(-1)], dtype=np.int32)

    num_noise = int(np.sum(labels < 0))
    num_clusters = int(len(np.unique(labels[labels >= 0])))

    estimator_meta = {
        "hdbscan_backend": "sklearn_subsample_1nn",
        "hdbscan_backend_verified": True,
        "hdbscan_estimator_class": HDBSCAN.__name__,
        "hdbscan_estimator_module": HDBSCAN.__module__,
    }
    out = {
        "hdbscan_num_samples": n,
        "hdbscan_num_features": int(features.shape[1]) if features.ndim > 1 else 1,
        "hdbscan_source_input_dtype": str(features.dtype),
        "hdbscan_input_dtype": str(features.dtype),
        "hdbscan_fit_predict_seconds": float(sub_stats.get("hdbscan_fit_predict_seconds", 0.0)),
        "hdbscan_subsample_cap": cap,
        "hdbscan_subsample_used": cap,
        "hdbscan_subsample_skipped": False,
        "hdbscan_propagate_nn_seconds": prop_elapsed_s,
        "hdbscan_subsample_fit_num_samples": int(sub_stats.get("hdbscan_num_samples", cap)),
        "num_clusters": num_clusters,
        "num_noise_points": num_noise,
        "noise_fraction": float(num_noise / max(len(labels), 1)),
        **estimator_meta,
    }
    log_progress(
        "HDBSCAN subsample+propagate output: "
        f"num_clusters={num_clusters}, num_noise_points={num_noise}, "
        f"fit_on={cap}, total={n}, propagate_seconds={prop_elapsed_s:.3f}"
    )
    return labels, out
