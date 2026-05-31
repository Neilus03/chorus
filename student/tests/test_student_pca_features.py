from __future__ import annotations

import json

import torch

from scripts.visualize_student_point_pca import (
    feature_instance_diagnostics,
    pca_rgb,
    save_point_cloud_ply,
)


def test_student_point_pca_outputs_rgb_and_jsonable_variance(tmp_path) -> None:
    torch.manual_seed(0)
    features = torch.randn(32, 8)
    rgb, info = pca_rgb(features, max_fit_points=None)

    assert rgb.shape == (32, 3)
    assert torch.all((rgb >= 0.0) & (rgb <= 1.0))
    assert len(info["explained_variance_ratio"]) == 3
    json.dumps(info)

    ply_path = tmp_path / "features.ply"
    save_point_cloud_ply(torch.randn(32, 3), rgb, ply_path)
    assert ply_path.exists()


def test_feature_instance_diagnostics_reports_separation_ratio() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = torch.tensor([1, 1, 2, 2])

    diagnostics = feature_instance_diagnostics(features, labels, min_points=1)

    assert diagnostics["num_instances"] == 2
    assert diagnostics["mean_intra_instance_cosine_distance"] is not None
    assert diagnostics["mean_nearest_inter_instance_cosine_distance"] is not None
    assert diagnostics["separation_ratio_inter_over_intra"] is not None
