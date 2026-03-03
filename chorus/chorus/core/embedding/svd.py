from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def compute_svd_features(
    point_mask_matrix: csr_matrix,
    n_components: int,
) -> tuple[np.ndarray, dict]:
    n_rows, n_cols = point_mask_matrix.shape

    if n_rows == 0:
        raise RuntimeError("Cannot compute SVD features on an empty matrix with zero rows.")

    if n_cols == 0:
        raise RuntimeError("Cannot compute SVD features: point-mask matrix has zero columns.")

    if n_cols == 1:
        #skip SVD and just normalize the single column
        features = point_mask_matrix.astype(np.float32).toarray()
        features = normalize(features, norm="l2", axis=1)

        stats = {
            "svd_components": 1,
            "explained_variance_sum": 1.0,
        }
        return features, stats

    effective_components = min(int(n_components), n_cols - 1)
    effective_components = max(1, effective_components)

    #perform SVD on the point-mask matrix
    svd = TruncatedSVD(n_components=effective_components, random_state=42)
    features = svd.fit_transform(point_mask_matrix)
    features = normalize(features, norm="l2", axis=1)

    stats = {
        "svd_components": int(effective_components),
        "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return features, stats