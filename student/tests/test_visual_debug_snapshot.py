import json

import torch

from student.engine.visual_debug import (
    save_point_cloud_ply,
    save_topdown_png,
    write_scene_snapshot,
)


def test_visual_snapshot_smoke(tmp_path):
    points = torch.rand(20, 3)
    mask_logits = torch.full((4, 20), -5.0)
    mask_logits[0, :5] = 5.0
    mask_logits[1, 5:10] = 5.0
    mask_logits[2, 10:15] = 5.0
    score_logits = torch.tensor([3.0, 2.0, 1.0, -1.0])
    pred = {
        "mask_logits": mask_logits,
        "score_logits": score_logits,
        "point_embed": torch.randn(20, 6),
        "query_xyz": torch.rand(4, 3),
        "debug": {
            "query_anchors_initial": torch.rand(4, 3),
            "query_anchors_by_layer": torch.rand(2, 4, 3),
            "query_source_type": torch.tensor([0, 1, 2, 0]),
        },
    }
    sample = {
        "points": points,
        "scene_dir": str(tmp_path),
    }
    labels = torch.tensor([1] * 5 + [2] * 5 + [3] * 5 + [-1] * 5)
    artifacts = write_scene_snapshot(
        tmp_path / "scene",
        sample=sample,
        pred=pred,
        granularity="g05",
        target_labels=labels,
        topk_queries=3,
        max_render_points=20,
    )
    assert (tmp_path / "scene" / "input_rgb_points.ply").is_file()
    assert (tmp_path / "scene" / "pred_vs_gt_overlay.png").is_file()
    assert (tmp_path / "scene" / "query_anchor_trajectories.png").is_file()
    assert (tmp_path / "scene" / "top_queries_table.json").is_file()
    rows = json.loads((tmp_path / "scene" / "top_queries_table.json").read_text())
    assert len(rows) == 3
    assert "debug_tensors_topk" in artifacts


def test_ply_and_png_writers(tmp_path):
    points = torch.rand(8, 3)
    rgb = torch.rand(8, 3)
    ply = tmp_path / "a.ply"
    png = tmp_path / "a.png"
    save_point_cloud_ply(points, rgb, ply)
    save_topdown_png(png, points, [("rgb", rgb)], max_points=8)
    assert ply.is_file()
    assert png.is_file()
