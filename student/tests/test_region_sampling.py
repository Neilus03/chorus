"""Tests for Point-style region sampling."""

from __future__ import annotations

import numpy as np
import pytest

from student.data.region_sampling import (
    grid_sample_indices,
    sphere_crop_indices,
    sphere_crop_indices_multi_center,
)


def _is_sphere_crop_result(coords: np.ndarray, idx: np.ndarray, k: int) -> bool:
    """True iff ``idx`` equals the k nearest indices to ``coords[ci]`` for some ci."""
    want = set(idx.tolist())
    for ci in range(coords.shape[0]):
        d2 = np.sum((coords - coords[ci]) ** 2, axis=1)
        nearest = np.argsort(d2)[:k]
        if set(nearest.tolist()) == want:
            return True
    return False


def test_sphere_crop_keeps_nearest_points() -> None:
    coords = np.array(
        [[i * 0.1, 0.0, 0.0] for i in range(10)],
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    idx = sphere_crop_indices(coords, rng=rng, point_max=3)
    assert idx.shape == (3,)
    assert _is_sphere_crop_result(coords, idx, 3)


def test_sphere_crop_deterministic_under_fixed_rng() -> None:
    coords = np.random.default_rng(42).normal(size=(100, 3)).astype(np.float32)
    a = sphere_crop_indices(coords, rng=np.random.default_rng(123), point_max=20)
    b = sphere_crop_indices(coords, rng=np.random.default_rng(123), point_max=20)
    np.testing.assert_array_equal(a, b)


def test_sphere_crop_all_points_when_small() -> None:
    coords = np.random.default_rng(0).normal(size=(5, 3))
    idx = sphere_crop_indices(coords, rng=np.random.default_rng(0), point_max=100)
    np.testing.assert_array_equal(np.sort(idx), np.arange(5))


def test_grid_sample_one_per_voxel() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    idx = grid_sample_indices(coords, rng=rng, grid_size=0.5)
    assert idx.shape[0] == 2  # two voxels


def test_sphere_multi_fragments_count() -> None:
    c = np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32)
    frags = sphere_crop_indices_multi_center(
        c, rng=np.random.default_rng(2), point_max=10, num_fragments=4
    )
    assert len(frags) == 4
    assert all(f.shape[0] == 10 for f in frags)


def test_invalid_coords_raises() -> None:
    with pytest.raises(ValueError):
        sphere_crop_indices(np.zeros((3, 2)), rng=np.random.default_rng(0), point_max=5)
