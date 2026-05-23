from __future__ import annotations

import torch
import torch.nn as nn

from student.models.continuous_base import is_continuous_decoder
from student.models.continuous_decoder import ContinuousQueryInstanceDecoder
from student.models.continuous_decoder_v2 import (
    ContinuousGeometryQueryDecoderV2,
    FiLMModulator,
    GranularityScaleSelector,
    RelationAwareQuerySelfAttention,
)


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


def test_film_modulator_is_identity_at_init() -> None:
    torch.manual_seed(0)
    film = FiLMModulator(hidden_dim=8)
    x = torch.randn(5, 8)
    cond = torch.randn(8)

    out = film(x, cond)

    assert torch.allclose(out, x, atol=1e-6)
    assert torch.count_nonzero(film.to_params.weight) == 0
    assert torch.count_nonzero(film.to_params.bias) == 0


def test_relation_bias_zero_init_and_can_be_disabled() -> None:
    torch.manual_seed(0)
    attn = RelationAwareQuerySelfAttention(
        hidden_dim=16,
        num_heads=4,
        use_relation_bias=False,
    )
    final = attn.rel_bias[-1]
    assert isinstance(final, nn.Linear)
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0

    out = attn(
        torch.randn(1, 6, 16),
        torch.randn(1, 6, 16),
        torch.randn(6, 3),
        torch.ones(6),
        torch.tensor([[0.5]]),
    )
    assert out.shape == (1, 6, 16)
    assert torch.isfinite(out).all()


def test_scale_selector_prior_strength_zero_is_uniform() -> None:
    torch.manual_seed(0)
    selector = GranularityScaleSelector(
        hidden_dim=16,
        num_scales=4,
        prior_strength=0.0,
    )

    weights = selector(torch.randn(16), torch.tensor([[0.2]]))

    assert torch.allclose(weights, torch.full((4,), 0.25), atol=1e-6)


def test_local_gate_initial_probability_is_configurable() -> None:
    decoder = _decoder(
        num_layers=1,
        continuous_v2_cfg={"local_aggregation": {"gate_init_prob": 0.10}},
    )
    linear = decoder.local_gates[0][0]
    assert isinstance(linear, nn.Linear)

    assert torch.count_nonzero(linear.weight) == 0
    assert torch.allclose(torch.sigmoid(linear.bias), torch.full_like(linear.bias, 0.10), atol=1e-6)


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


def test_anchor_refinement_runtime_scale_zero_blocks_delta_step_and_cap_bounds_it() -> None:
    torch.manual_seed(2)
    decoder = _decoder(
        num_queries=8,
        num_layers=1,
        continuous_v2_cfg={
            "anchors": {
                "delta_scale": 1.0,
                "delta_max_norm": 1.0,
                "delta_max_step": 0.03,
            },
            "local_aggregation": {"max_neighbors": 8, "chunk_queries": 8},
        },
    )
    final = decoder.delta_heads[0][-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(10.0)

    kwargs = {
        "point_xyz": torch.randn(32, 3),
        "scene_tokens": torch.randn(16, 8),
        "scene_xyz": torch.randn(16, 3),
        "target_g": 0.5,
    }
    decoder.set_anchor_refinement_scale(0.0)
    out_disabled = decoder(torch.randn(32, 8), **kwargs)
    assert out_disabled["diagnostics"]["delta_step_norm_layer_0"].item() == 0.0

    decoder.set_anchor_refinement_scale(1.0)
    out_enabled = decoder(torch.randn(32, 8), **kwargs)
    step_norm = out_enabled["diagnostics"]["delta_step_norm_layer_0"].item()
    assert 0.0 < step_norm <= 0.0301


def test_v2_feature_switches_disable_relation_local_and_head_film() -> None:
    torch.manual_seed(3)
    decoder = _decoder(
        num_queries=8,
        num_layers=1,
        continuous_v2_cfg={
            "relation_self_attention": {"use_relation_bias": False},
            "local_aggregation": {"use_local_aggregation": False},
            "head": {"use_film_heads": False},
        },
    )
    out = decoder(
        torch.randn(24, 8),
        point_xyz=torch.randn(24, 3),
        scene_tokens=torch.randn(12, 8),
        scene_xyz=torch.randn(12, 3),
        target_g=0.5,
    )

    assert out["mask_logits"].shape == (8, 24)
    assert out["diagnostics"]["local_gate_mean_layer_0"].item() == 0.0
