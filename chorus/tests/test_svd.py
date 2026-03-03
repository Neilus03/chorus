from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from chorus.core.embedding.svd import compute_svd_features


def _row_norms(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features, axis=1)


def test_compute_svd_features_raises_on_zero_rows() -> None:
    matrix = csr_matrix((0, 3), dtype=np.float32)

    with pytest.raises(RuntimeError, match="zero rows"):
        compute_svd_features(matrix, n_components=2)


def test_compute_svd_features_raises_on_zero_columns() -> None:
    matrix = csr_matrix((3, 0), dtype=np.float32)

    with pytest.raises(RuntimeError, match="zero columns"):
        compute_svd_features(matrix, n_components=2)


def test_compute_svd_features_one_column_matrix() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1.0],
                [2.0],
                [0.0],
            ],
            dtype=np.float32,
        )
    )

    features, stats = compute_svd_features(matrix, n_components=8)

    assert features.shape == (3, 1)
    assert stats["svd_components"] == 1
    assert stats["explained_variance_sum"] == pytest.approx(1.0)

    np.testing.assert_allclose(
        features[:, 0],
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )

    np.testing.assert_allclose(
        _row_norms(features),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_compute_svd_features_two_column_matrix_uses_one_effective_component() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    features, stats = compute_svd_features(matrix, n_components=32)

    assert features.shape == (4, 1)
    assert stats["svd_components"] == 1
    assert 0.0 <= stats["explained_variance_sum"] <= 1.0

    nonzero_rows = np.array([0, 1, 2, 3], dtype=np.int64)
    np.testing.assert_allclose(_row_norms(features)[nonzero_rows], 1.0, atol=1e-6)


def test_compute_svd_features_ordinary_matrix_returns_normalized_features() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 2.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    features, stats = compute_svd_features(matrix, n_components=2)

    assert features.shape == (5, 2)
    assert stats["svd_components"] == 2
    assert 0.0 <= stats["explained_variance_sum"] <= 1.0

    np.testing.assert_allclose(_row_norms(features), 1.0, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__])