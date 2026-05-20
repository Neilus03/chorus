from __future__ import annotations

import torch
import torch.nn as nn

from student.models.continuous_base import is_continuous_decoder
from student.models.continuous_decoder import ContinuousQueryInstanceDecoder
from student.models.continuous_decoder_v2 import ContinuousGeometryQueryDecoderV2


def _decoder(**kwargs) -> ContinuousGeometryQueryDecoderV2:
    return ContinuousGeometryQueryDecoderV2(
        in_channels=8,
        hidden_dim=32,
        num_queries=kwargs.pop("num_queries", 16),
        num_layers=kwargs.pop("num_layers", 2),
        num_heads=4,
        learned_ratio=0.25,
        use_positional_guidance=True,
        **kwargs,
    )


def test_continuous_helper_detects_v1_and_v2() -> None:
    v1 = ContinuousQueryInstanceDecoder(
        in_channels=8,
        hidden_dim=32,
        num_queries=8,
        num_layers=1,
        num_heads=4,
    )
    v2 = _decoder(num_queries=8, num_layers=1)

    assert is_continuous_decoder(v1)
    assert is_continuous_decoder(v2)
    assert not is_continuous_decoder(nn.Linear(1, 1))


def test_v2_forward_multiscale_shapes_and_aux_outputs_are_finite() -> None:
    torch.manual_seed(0)
    decoder = _decoder(
        multi_scale_channels=[16, 8],
        continuous_v2_cfg={
            "local_aggregation": {"max_neighbors": 8, "chunk_queries": 8},
        },
    )
    N = 64
    point_feat = torch.randn(N, 8)
    point_xyz = torch.randn(N, 3)
    scene_tokens = torch.randn(32, 8)
    scene_xyz = torch.randn(32, 3)
    multi_scale_tokens = [torch.randn(12, 16), torch.randn(24, 8)]
    multi_scale_xyz = [torch.randn(12, 3), torch.randn(24, 3)]

    out = decoder(
        point_feat,
        point_xyz=point_xyz,
        scene_tokens=scene_tokens,
        scene_xyz=scene_xyz,
        multi_scale_tokens=multi_scale_tokens,
        multi_scale_xyz=multi_scale_xyz,
        target_g=0.2,
    )

    assert out["mask_logits"].shape == (16, N)
    assert out["score_logits"].shape == (16,)
    assert out["query_embed"].shape == (16, 32)
    assert out["query_xyz"].shape == (16, 3)
    assert out["query_radius"].shape == (16,)
    assert out["scale_weights"].shape == (2,)
    assert len(out["aux_outputs"]) == 1
    assert "diagnostics" in out
    for key in ("mask_logits", "score_logits", "query_embed", "query_xyz"):
        assert torch.isfinite(out[key]).all()

    loss = out["mask_logits"].mean() + out["score_logits"].mean() + out["query_xyz"].mean()
    loss.backward()
    assert any(p.grad is not None for p in decoder.parameters())


def test_v2_forward_handles_fewer_scene_tokens_than_queries() -> None:
    torch.manual_seed(1)
    decoder = _decoder(
        num_queries=12,
        num_layers=1,
        continuous_v2_cfg={
            "local_aggregation": {"max_neighbors": 16, "chunk_queries": 4},
        },
    )
    out = decoder(
        torch.randn(20, 8),
        point_xyz=torch.randn(20, 3),
        scene_tokens=torch.randn(4, 8),
        scene_xyz=torch.randn(4, 3),
        target_g=torch.tensor([0.8]),
    )

    assert out["mask_logits"].shape == (12, 20)
    assert out["query_xyz"].shape == (12, 3)
    assert torch.isfinite(out["mask_logits"]).all()
    assert torch.isfinite(out["diagnostics"]["local_neighbor_count_mean"])
