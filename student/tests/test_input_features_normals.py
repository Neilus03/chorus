"""Tests for RGB+normal feature stacking and augment return shape."""

from __future__ import annotations

import numpy as np

from student.data.point_augmentations import augment_points_litept_scannet
from student.data.single_scene_dataset import build_input_features


def test_build_input_features_rgb_normals_shape() -> None:
    n = 10
    pts = np.random.randn(n, 3).astype(np.float32)
    col = np.random.rand(n, 3).astype(np.float32)
    nrm = np.random.randn(n, 3).astype(np.float32)
    feat = build_input_features(
        pts, col,
        use_normals=True,
        normals=nrm,
    )
    assert feat.shape == (n, 6)


def test_build_input_features_use_normals_missing_fills_zeros() -> None:
    n = 5
    pts = np.zeros((n, 3), dtype=np.float32)
    col = np.ones((n, 3), dtype=np.float32) * 0.5
    feat = build_input_features(pts, col, use_normals=True, normals=None)
    assert feat.shape == (n, 6)
    assert np.allclose(feat[:, 3:6], 0.0)


def test_augment_returns_three_values() -> None:
    n = 20
    pts = np.random.randn(n, 3).astype(np.float32)
    col = np.random.rand(n, 3).astype(np.float32)
    nrm = np.array([[0.0, 0.0, 1.0]] * n, dtype=np.float32)
    out = augment_points_litept_scannet(pts, col, normals=nrm)
    assert len(out) == 3
    p, c, nm = out
    assert p.shape == (n, 3)
    assert c is not None and c.shape == (n, 3)
    assert nm is not None and nm.shape == (n, 3)
