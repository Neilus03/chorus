from __future__ import annotations

import time
import traceback

import numpy as np
from sklearn.cluster import HDBSCAN

from chorus.common.progress import heartbeat, log_progress


def cluster_features(
    features: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
) -> tuple[np.ndarray, dict]:
    features = np.asarray(features)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        n_jobs=-1,
        copy=False,
    )
    estimator_meta = {
        "hdbscan_backend": "sklearn",
        "hdbscan_backend_verified": clusterer.__class__.__module__.startswith("sklearn."),
        "hdbscan_estimator_class": clusterer.__class__.__name__,
        "hdbscan_estimator_module": clusterer.__class__.__module__,
    }
    log_progress(
        "HDBSCAN input: "
        f"samples={features.shape[0]}, dims={features.shape[1]}, "
        f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
        f"cluster_selection_epsilon={cluster_selection_epsilon}, dtype={features.dtype}, "
        f"backend={estimator_meta['hdbscan_backend']}"
    )
    fit_start = time.perf_counter()
    try:
        with heartbeat("HDBSCAN fit_predict"):
            labels = np.asarray(clusterer.fit_predict(features), dtype=np.int32).reshape(-1)
    except Exception:
        finite = np.isfinite(features)
        diagnostics = {
            "shape": tuple(int(x) for x in features.shape),
            "dtype": str(features.dtype),
            "ndim": int(features.ndim),
            "c_contiguous": bool(features.flags.c_contiguous),
            "f_contiguous": bool(features.flags.f_contiguous),
            "finite_fraction": float(np.count_nonzero(finite) / max(features.size, 1)),
            "nan_count": int(np.count_nonzero(np.isnan(features))),
            "posinf_count": int(np.count_nonzero(np.isposinf(features))),
            "neginf_count": int(np.count_nonzero(np.isneginf(features))),
        }
        if np.any(finite):
            finite_values = features[finite]
            diagnostics.update(
                {
                    "finite_min": float(np.min(finite_values)),
                    "finite_max": float(np.max(finite_values)),
                    "finite_mean": float(np.mean(finite_values)),
                    "finite_std": float(np.std(finite_values)),
                }
            )
        log_progress(f"HDBSCAN fit_predict exception diagnostics: {diagnostics}")
        traceback.print_exc()
        raise
    fit_elapsed_s = float(time.perf_counter() - fit_start)

    num_noise = int(np.sum(labels < 0))
    num_clusters = int(len(np.unique(labels[labels >= 0])))

    stats = {
        "hdbscan_num_samples": int(features.shape[0]),
        "hdbscan_num_features": int(features.shape[1]) if features.ndim > 1 else 1,
        "hdbscan_source_input_dtype": str(features.dtype),
        "hdbscan_input_dtype": str(features.dtype),
        "hdbscan_fit_predict_seconds": fit_elapsed_s,
        "num_clusters": num_clusters,
        "num_noise_points": num_noise,
        "noise_fraction": float(num_noise / max(len(labels), 1)),
        **estimator_meta,
    }
    log_progress(
        "HDBSCAN output: "
        f"num_clusters={num_clusters}, num_noise_points={num_noise}, "
        f"noise_fraction={stats['noise_fraction']:.4f}, backend={stats['hdbscan_backend']}, "
        f"fit_predict_seconds={stats['hdbscan_fit_predict_seconds']:.3f}"
    )
    return labels, stats
