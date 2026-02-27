from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def compute_svd_features(
    point_mask_matrix: csr_matrix,
    n_components: int,
) -> tuple[np.ndarray, dict]:
    n_components = min(n_components, max(2, point_mask_matrix.shape[1] - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    features = svd.fit_transform(point_mask_matrix)
    features = normalize(features, norm="l2", axis=1)

    stats = {
        "svd_components": int(n_components),
        "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return features, stats