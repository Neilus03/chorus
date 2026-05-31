import torch

from student.engine.feature_diagnostics import (
    compute_feature_diagnostics,
    compute_instance_feature_separation,
    pca_rgb,
)


def test_feature_separation_ratio():
    cluster_a = torch.randn(30, 6) * 0.02 + torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
    cluster_b = torch.randn(30, 6) * 0.02 + torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
    feats = torch.cat([cluster_a, cluster_b], dim=0)
    labels = torch.tensor([1] * 30 + [2] * 30)
    stats = compute_instance_feature_separation(feats, labels, min_points=5)
    assert stats["inter_over_intra_feature_separation"] > 1.0
    assert stats["same_instance_cosine_mean"] > stats["different_instance_cosine_mean"]


def test_pca_rgb_shape_and_stats():
    feats = torch.randn(64, 8)
    rgb, info = pca_rgb(feats)
    assert rgb.shape == (64, 3)
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0
    assert len(info["explained_variance_ratio"]) == 3


def test_compute_feature_diagnostics_keys():
    feats = torch.randn(32, 8)
    labels = torch.tensor([1] * 16 + [2] * 16)
    stats = compute_feature_diagnostics(feats, labels)
    assert "point_feature_norm_mean" in stats
    assert "pca_explained_var_0" in stats
    assert "inter_over_intra_feature_separation" in stats
