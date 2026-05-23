"""Continuous geometry-query decoder V2.

This module keeps the V1 continuous decoder contract while adding the minimal
geometry-first V2 components:

* stronger granularity conditioning through identity-initialized FiLM,
* a granularity-biased multi-scale selector,
* geometry/saliency/learned query initialization,
* persistent query anchors with per-layer refinement,
* sparse-token local aggregation, and
* relation-aware query self-attention.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from student.models.continuous_base import ContinuousDecoderMixin
from student.models.instance_decoder import FourierPosEnc


def _as_granularity_tensor(
    target_g: torch.Tensor | float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(target_g, torch.Tensor):
        target_g = torch.tensor([target_g], device=device, dtype=torch.float32)
    return target_g.to(device=device, dtype=torch.float32).view(1, 1)


def _as_granularity_float(target_g: torch.Tensor) -> float:
    return float(target_g.detach().flatten()[0].clamp(0.0, 1.0).item())


class GranularityEncoder(nn.Module):
    """Encode scalar granularity into the decoder hidden space."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        inner_dim = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(1, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, target_g: torch.Tensor) -> torch.Tensor:
        return self.net(target_g.view(1, 1)).squeeze(0)


class FiLMModulator(nn.Module):
    """Identity-initialized feature-wise linear modulation."""

    def __init__(self, hidden_dim: int, cond_dim: int | None = None) -> None:
        super().__init__()
        cond_dim = int(cond_dim or hidden_dim)
        self.norm = nn.LayerNorm(cond_dim)
        self.to_params = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.view(1, -1)
        scale, shift = self.to_params(self.norm(cond)).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(0)
            shift = shift.unsqueeze(0)
        return x * (1.0 + scale) + shift


class GranularityScaleSelector(nn.Module):
    """Granularity-conditioned selector over coarse-to-fine memory levels."""

    def __init__(
        self,
        hidden_dim: int,
        num_scales: int,
        temperature: float = 1.0,
        prior_only: bool = False,
        use_scale_selector: bool = True,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_scales = int(max(num_scales, 1))
        self.temperature = float(max(temperature, 1e-4))
        self.prior_only = bool(prior_only)
        self.use_scale_selector = bool(use_scale_selector)
        self.prior_strength = float(prior_strength)
        self.learned_logits = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_scales),
        )
        final = self.learned_logits[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, g_emb: torch.Tensor, target_g: torch.Tensor) -> torch.Tensor:
        if self.num_scales == 1:
            return g_emb.new_ones((1,))
        if not self.use_scale_selector:
            return g_emb.new_full((self.num_scales,), 1.0 / float(self.num_scales))

        g = _as_granularity_float(target_g)
        idx = torch.arange(self.num_scales, device=g_emb.device, dtype=g_emb.dtype)
        desired = (1.0 - g) * float(self.num_scales - 1)
        prior = self.prior_strength * (-((idx - desired) ** 2) / self.temperature)
        learned = torch.zeros_like(prior) if self.prior_only else self.learned_logits(g_emb.view(1, -1)).squeeze(0)
        logits = learned + prior
        return logits.softmax(dim=0)


def _repeat_indices(idx: torch.Tensor, count: int, total: int) -> torch.Tensor:
    if count <= 0:
        return idx.new_zeros((0,), dtype=torch.long)
    if idx.numel() == 0:
        return torch.zeros(count, device=idx.device, dtype=torch.long)
    if idx.numel() >= count:
        return idx[:count]
    repeats = (count + idx.numel() - 1) // idx.numel()
    return idx.repeat(repeats)[:count].clamp_max(max(total - 1, 0))


@torch.no_grad()
def _farthest_point_indices(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic farthest point sampling over sparse memory anchors."""
    V = int(xyz.shape[0])
    if count <= 0:
        return torch.zeros(0, device=xyz.device, dtype=torch.long)
    if V == 0:
        return torch.zeros(count, device=xyz.device, dtype=torch.long)
    if V <= count:
        return _repeat_indices(torch.arange(V, device=xyz.device), count, V)

    selected = torch.empty(count, device=xyz.device, dtype=torch.long)
    center = xyz.mean(dim=0, keepdim=True)
    dist_to_center = ((xyz - center) ** 2).sum(dim=-1)
    farthest = int(dist_to_center.argmax().item())
    closest_dist = torch.full((V,), float("inf"), device=xyz.device, dtype=xyz.dtype)

    for i in range(count):
        selected[i] = farthest
        d = ((xyz - xyz[farthest].view(1, 3)) ** 2).sum(dim=-1)
        closest_dist = torch.minimum(closest_dist, d)
        farthest = int(closest_dist.argmax().item())
    return selected


@torch.no_grad()
def _spatial_nms_topk(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    count: int,
    *,
    radius: float,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select high-score anchors with a simple spatial suppression radius."""
    V = int(scores.shape[0])
    if count <= 0:
        return torch.zeros(0, device=scores.device, dtype=torch.long)
    if V == 0:
        return torch.zeros(count, device=scores.device, dtype=torch.long)

    blocked = torch.zeros(V, device=scores.device, dtype=torch.bool)
    if exclude is not None and exclude.numel() > 0:
        blocked[exclude.clamp(0, V - 1)] = True

    order = scores.argsort(descending=True)
    chosen: list[torch.Tensor] = []
    min_r2 = float(radius * radius)
    for raw_idx in order:
        idx = int(raw_idx.item())
        if bool(blocked[idx].item()):
            continue
        if chosen:
            chosen_idx = torch.stack(chosen)
            d2 = ((xyz[chosen_idx] - xyz[idx].view(1, 3)) ** 2).sum(dim=-1)
            if bool((d2 < min_r2).any().item()):
                continue
        chosen.append(raw_idx)
        if len(chosen) >= count:
            break

    if chosen:
        idx = torch.stack(chosen).to(dtype=torch.long)
    else:
        available = order[~blocked[order]]
        idx = available[:1] if available.numel() else order[:1]
    return _repeat_indices(idx, count, V)


class GranularityGeometryQueryInitializer(nn.Module):
    """Geometry/saliency/learned query initialization for continuous scales."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        *,
        geometry_ratio: float = 0.40,
        saliency_ratio: float = 0.35,
        learned_ratio: float = 0.25,
        saliency_hidden_dim: int = 128,
        spatial_nms_radius_min: float = 0.10,
        spatial_nms_radius_max: float = 0.80,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_geometry = int(round(num_queries * geometry_ratio))
        self.num_learned = int(round(num_queries * learned_ratio))
        self.num_saliency = self.num_queries - self.num_geometry - self.num_learned
        if self.num_saliency < 0:
            self.num_saliency = 0
            self.num_geometry = self.num_queries - self.num_learned
        self.spatial_nms_radius_min = float(spatial_nms_radius_min)
        self.spatial_nms_radius_max = float(spatial_nms_radius_max)

        self.saliency_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, saliency_hidden_dim),
            nn.GELU(),
            nn.Linear(saliency_hidden_dim, 1),
        )
        self.saliency_score_proj = nn.Linear(1, hidden_dim)
        self.learned_queries = nn.Embedding(self.num_learned, hidden_dim)
        self.learned_offsets = nn.Embedding(self.num_learned, 3)
        nn.init.zeros_(self.learned_offsets.weight)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        target_g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        V = int(scene_tokens.shape[0])
        g = _as_granularity_float(target_g)
        radius = self.spatial_nms_radius_min + (
            self.spatial_nms_radius_max - self.spatial_nms_radius_min
        ) * g

        geom_idx = _farthest_point_indices(scene_xyz, self.num_geometry)
        geom_q = scene_tokens[geom_idx] if geom_idx.numel() else scene_tokens.new_zeros((0, self.hidden_dim))
        geom_xyz = scene_xyz[geom_idx] if geom_idx.numel() else scene_xyz.new_zeros((0, 3))

        saliency_scores = self.saliency_mlp(scene_tokens).squeeze(-1)
        sal_idx = _spatial_nms_topk(
            saliency_scores.detach(),
            scene_xyz,
            self.num_saliency,
            radius=radius,
            exclude=geom_idx,
        )
        if sal_idx.numel() > 0:
            sal_q = scene_tokens[sal_idx] + self.saliency_score_proj(
                saliency_scores[sal_idx].unsqueeze(-1),
            )
            sal_xyz = scene_xyz[sal_idx]
        else:
            sal_q = scene_tokens.new_zeros((0, self.hidden_dim))
            sal_xyz = scene_xyz.new_zeros((0, 3))

        if self.num_learned > 0:
            learned_q = self.learned_queries.weight
            center = scene_xyz.mean(dim=0, keepdim=True)
            learned_xyz = center + torch.tanh(self.learned_offsets.weight) * max(radius, 1e-3)
        else:
            learned_q = scene_tokens.new_zeros((0, self.hidden_dim))
            learned_xyz = scene_xyz.new_zeros((0, 3))

        q_feat = torch.cat([geom_q, sal_q, learned_q], dim=0)
        q_xyz = torch.cat([geom_xyz, sal_xyz, learned_xyz], dim=0)
        q_feat = q_feat[:self.num_queries]
        q_xyz = q_xyz[:self.num_queries]

        if q_feat.shape[0] < self.num_queries:
            pad_n = self.num_queries - q_feat.shape[0]
            pad_idx = _repeat_indices(torch.arange(q_feat.shape[0], device=q_feat.device), pad_n, q_feat.shape[0])
            q_feat = torch.cat([q_feat, q_feat[pad_idx]], dim=0)
            q_xyz = torch.cat([q_xyz, q_xyz[pad_idx]], dim=0)

        query_radius = q_feat.new_full((self.num_queries,), max(radius, 1e-3))
        q_source = torch.cat([
            torch.zeros(min(self.num_geometry, self.num_queries), device=q_feat.device, dtype=torch.long),
            torch.ones(min(self.num_saliency, max(self.num_queries - self.num_geometry, 0)), device=q_feat.device, dtype=torch.long),
            torch.full(
                (max(self.num_queries - self.num_geometry - self.num_saliency, 0),),
                2,
                device=q_feat.device,
                dtype=torch.long,
            ),
        ], dim=0)[:self.num_queries]
        return q_feat, q_xyz, query_radius, q_source


class LocalQueryAggregator(nn.Module):
    """Gather sparse-token local context around each query anchor."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_neighbors: int = 32,
        chunk_queries: int = 64,
    ) -> None:
        super().__init__()
        self.max_neighbors = int(max(max_neighbors, 1))
        self.chunk_queries = int(max(chunk_queries, 1))
        self.rel_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        q_xyz: torch.Tensor,
        memory: torch.Tensor,
        memory_xyz: torch.Tensor,
        query_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        Q = int(q_xyz.shape[0])
        V = int(memory.shape[0])
        if V == 0:
            zero = memory.new_zeros((Q, memory.shape[-1]))
            stats = {
                "neighbor_count_mean": memory.new_tensor(0.0),
                "neighbor_count_min": memory.new_tensor(0.0),
                "neighbor_count_max": memory.new_tensor(0.0),
                "neighbor_zero_frac": memory.new_tensor(1.0),
            }
            return zero, stats

        k = min(self.max_neighbors, V)
        chunks: list[torch.Tensor] = []
        count_chunks: list[torch.Tensor] = []
        for start in range(0, Q, self.chunk_queries):
            end = min(start + self.chunk_queries, Q)
            q_chunk = q_xyz[start:end]
            radius = query_radius[start:end].clamp_min(1e-3)
            dist = torch.cdist(q_chunk, memory_xyz)
            top_d, top_idx = dist.topk(k, largest=False, dim=1)
            within = top_d <= radius[:, None]
            counts = within.sum(dim=1).to(dtype=memory.dtype)
            effective = within.clone()
            no_neighbor = ~effective.any(dim=1)
            if no_neighbor.any():
                effective[no_neighbor, 0] = True

            logits = -top_d / radius[:, None]
            logits = logits.masked_fill(~effective, -1.0e4)
            weights = logits.softmax(dim=1)

            mem_vals = memory[top_idx]
            rel = memory_xyz[top_idx] - q_chunk[:, None, :]
            rel_dist = top_d.unsqueeze(-1)
            rel_feat = self.rel_mlp(torch.cat([rel, rel_dist], dim=-1))
            local = (weights.unsqueeze(-1) * (mem_vals + rel_feat)).sum(dim=1)
            chunks.append(local)
            count_chunks.append(counts)

        local_feat = self.out_proj(torch.cat(chunks, dim=0))
        counts_all = torch.cat(count_chunks, dim=0)
        stats = {
            "neighbor_count_mean": counts_all.mean().detach(),
            "neighbor_count_min": counts_all.min().detach(),
            "neighbor_count_max": counts_all.max().detach(),
            "neighbor_zero_frac": (counts_all <= 0).to(dtype=memory.dtype).mean().detach(),
        }
        return local_feat, stats


class RelationAwareQuerySelfAttention(nn.Module):
    """Query self-attention with additive relative-geometry bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        geometry_hidden_dim: int = 64,
        *,
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.use_relation_bias = bool(use_relation_bias)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.rel_bias = nn.Sequential(
            nn.Linear(7, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, num_heads),
        )
        final = self.rel_bias[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _bias(
        self,
        q_xyz: torch.Tensor,
        query_radius: torch.Tensor,
        target_g: torch.Tensor,
    ) -> torch.Tensor:
        Q = int(q_xyz.shape[0])
        delta = q_xyz[:, None, :] - q_xyz[None, :, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        r_i = query_radius[:, None, None].expand(Q, Q, 1)
        r_j = query_radius[None, :, None].expand(Q, Q, 1)
        g = target_g.to(device=q_xyz.device, dtype=q_xyz.dtype).view(1, 1, 1).expand(Q, Q, 1)
        rel = torch.cat([delta, dist, r_i, r_j, g], dim=-1)
        bias = self.rel_bias(rel).permute(2, 0, 1).contiguous()
        return bias

    def forward(
        self,
        q: torch.Tensor,
        q_pos: torch.Tensor,
        q_xyz: torch.Tensor,
        query_radius: torch.Tensor,
        target_g: torch.Tensor,
    ) -> torch.Tensor:
        q_with_pos = q + q_pos
        bias = self._bias(q_xyz, query_radius, target_g) if self.use_relation_bias else None
        out, _ = self.attn(q_with_pos, q_with_pos, q, attn_mask=bias)
        return out


class GeometryDecoderLayer(nn.Module):
    """Pre-norm decoder layer with relation-aware self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 8,
        ff_mult: int = 4,
        geometry_bias_hidden_dim: int = 64,
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = RelationAwareQuerySelfAttention(
            hidden_dim,
            num_heads,
            geometry_hidden_dim=geometry_bias_hidden_dim,
            use_relation_bias=use_relation_bias,
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(hidden_dim)
        self.norm_ca = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        q_pos: torch.Tensor,
        q_xyz: torch.Tensor,
        query_radius: torch.Tensor,
        scene_mem: torch.Tensor,
        scene_pos: torch.Tensor,
        target_g: torch.Tensor,
    ) -> torch.Tensor:
        q_n = self.norm_sa(q)
        q = q + self.self_attn(q_n, q_pos, q_xyz, query_radius, target_g)

        q_n2 = self.norm_ca(q)
        ca_out, _ = self.cross_attn(
            query=q_n2 + q_pos,
            key=scene_mem + scene_pos,
            value=scene_mem,
        )
        q = q + ca_out
        q = q + self.ffn(self.norm_ff(q))
        return q


class FiLMGranularityHead(nn.Module):
    """Granularity-modulated mask, score, and optional class heads."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int | None = None,
        *,
        logit_scale_init: float = 10.0,
        logit_scale_min: float | None = None,
        logit_scale_max: float | None = None,
        freeze_logit_scale: bool = False,
        use_film_heads: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.logit_scale_min = None if logit_scale_min is None else float(logit_scale_min)
        self.logit_scale_max = None if logit_scale_max is None else float(logit_scale_max)
        self.use_film_heads = bool(use_film_heads)
        self.mask_body = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mask_film = FiLMModulator(hidden_dim)
        self.mask_out = nn.Linear(hidden_dim, hidden_dim)

        self.score_body = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_film = FiLMModulator(hidden_dim)
        self.score_out = nn.Linear(hidden_dim, 1)

        self.class_head: nn.Module | None = None
        if self.num_classes is not None:
            self.class_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_classes + 1),
            )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(logit_scale_init))))
        if freeze_logit_scale:
            self.logit_scale.requires_grad_(False)

    def mask_embed(self, query_embed: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        x = self.mask_body(query_embed)
        if self.use_film_heads:
            x = self.mask_film(x, g_emb)
        return self.mask_out(x)

    def score_logits(self, query_embed: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        x = self.score_body(query_embed)
        if self.use_film_heads:
            x = self.score_film(x, g_emb)
        return self.score_out(x).squeeze(-1)

    def class_logits(self, query_embed: torch.Tensor) -> torch.Tensor | None:
        if self.class_head is None:
            return None
        return self.class_head(query_embed)

    def mask_logit_scale(self) -> torch.Tensor:
        scale = self.logit_scale.exp()
        if self.logit_scale_min is None and self.logit_scale_max is None:
            return scale
        return scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max)


class ContinuousGeometryQueryDecoderV2(ContinuousDecoderMixin, nn.Module):
    """Granularity-conditioned geometry query decoder with V1-compatible output."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_queries: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        learned_ratio: float = 0.25,
        use_positional_guidance: bool = True,
        multi_scale_channels: list[int] | None = None,
        num_instance_classes: int | None = None,
        continuous_v2_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = continuous_v2_cfg or {}
        gran_cfg = cfg.get("granularity", {})
        query_cfg = cfg.get("query_init", {})
        anchor_cfg = cfg.get("anchors", {})
        local_cfg = cfg.get("local_aggregation", {})
        relation_cfg = cfg.get("relation_self_attention", {})
        head_cfg = cfg.get("head", {})

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.use_positional_guidance = bool(use_positional_guidance)
        self.delta_scale = float(anchor_cfg.get("delta_scale", 0.10))
        self.use_delta_refinement = bool(anchor_cfg.get("use_delta_refinement", True))
        delta_norm_cfg = anchor_cfg.get("delta_max_norm", anchor_cfg.get("delta_norm_max", 0.0))
        self.delta_norm_max = float(delta_norm_cfg or 0.0)
        delta_step_cfg = anchor_cfg.get("delta_max_step", None)
        self.delta_max_step = None if delta_step_cfg is None else float(delta_step_cfg)
        self.anchor_refinement_runtime_scale = 1.0
        self.clamp_to_scene_bounds = bool(anchor_cfg.get("clamp_to_scene_bounds", True))
        self.use_local_aggregation = bool(
            local_cfg.get("use_local_aggregation", local_cfg.get("enabled", True))
        )
        self.local_radius_min = float(local_cfg.get("radius_min", 0.15))
        self.local_radius_max = float(local_cfg.get("radius_max", 1.20))

        self.granularity_encoder = GranularityEncoder(hidden_dim)
        num_scales = len(multi_scale_channels) if multi_scale_channels else 1
        self.scale_selector = GranularityScaleSelector(
            hidden_dim,
            num_scales,
            temperature=float(gran_cfg.get("scale_selector_temperature", 1.0)),
            prior_only=bool(gran_cfg.get("scale_selector_prior_only", False)),
            use_scale_selector=bool(
                gran_cfg.get("use_scale_selector", gran_cfg.get("scale_selector", True))
            ),
            prior_strength=float(gran_cfg.get("scale_selector_prior_strength", 1.0)),
        )
        if bool(gran_cfg.get("freeze_scale_selector", False)):
            for param in self.scale_selector.learned_logits.parameters():
                param.requires_grad_(False)
        self.query_film = FiLMModulator(hidden_dim)
        self.memory_film = FiLMModulator(hidden_dim)

        self.scene_token_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if multi_scale_channels is not None and len(multi_scale_channels) > 0:
            self.scale_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for c in multi_scale_channels
            ])
        else:
            self.scale_projs = None

        self.point_mask_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        num_fourier_bands = 16
        pos_dim = 3 * num_fourier_bands * 2
        self.pos_encoder = FourierPosEnc(in_dim=3, num_bands=num_fourier_bands)
        self.pos_proj = nn.Linear(pos_dim, hidden_dim)

        self.initializer = GranularityGeometryQueryInitializer(
            hidden_dim,
            num_queries,
            geometry_ratio=float(query_cfg.get("geometry_ratio", 0.40)),
            saliency_ratio=float(query_cfg.get("saliency_ratio", 0.35)),
            learned_ratio=float(query_cfg.get("learned_ratio", learned_ratio)),
            saliency_hidden_dim=int(query_cfg.get("saliency_hidden_dim", 128)),
            spatial_nms_radius_min=float(query_cfg.get("spatial_nms_radius_min", 0.10)),
            spatial_nms_radius_max=float(query_cfg.get("spatial_nms_radius_max", 0.80)),
        )

        self.local_aggregator = LocalQueryAggregator(
            hidden_dim,
            max_neighbors=int(local_cfg.get("max_neighbors", 32)),
            chunk_queries=int(local_cfg.get("chunk_queries", 64)),
        )
        self.local_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            for _ in range(num_layers)
        ])
        gate_init_prob = float(local_cfg.get("gate_init_prob", 0.10))
        gate_init_prob = min(max(gate_init_prob, 1e-4), 1.0 - 1e-4)
        gate_init_logit = math.log(gate_init_prob / (1.0 - gate_init_prob))
        for gate in self.local_gates:
            linear = gate[0]
            assert isinstance(linear, nn.Linear)
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, gate_init_logit)
        self.layer_query_films = nn.ModuleList([
            FiLMModulator(hidden_dim) for _ in range(num_layers)
        ])
        self.layers = nn.ModuleList([
            GeometryDecoderLayer(
                hidden_dim,
                num_heads=num_heads,
                geometry_bias_hidden_dim=int(
                    relation_cfg.get("geometry_bias_hidden_dim", 64),
                ),
                use_relation_bias=bool(
                    relation_cfg.get("use_relation_bias", relation_cfg.get("enabled", True))
                ),
            )
            for _ in range(num_layers)
        ])
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            for _ in range(num_layers)
        ])
        for head in self.delta_heads:
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        self.head = FiLMGranularityHead(
            hidden_dim,
            num_classes=num_instance_classes,
            logit_scale_init=float(head_cfg.get("logit_scale_init", 10.0)),
            logit_scale_min=head_cfg.get("logit_scale_min", None),
            logit_scale_max=head_cfg.get("logit_scale_max", None),
            freeze_logit_scale=bool(head_cfg.get("freeze_logit_scale", False)),
            use_film_heads=bool(head_cfg.get("use_film_heads", True)),
        )

    def set_anchor_refinement_scale(self, scale: float) -> None:
        self.anchor_refinement_runtime_scale = float(max(0.0, min(1.0, scale)))

    def _scale_index_for_layer(
        self,
        layer_idx: int,
        num_scales: int,
        scale_weights: torch.Tensor,
    ) -> int:
        if num_scales <= 1:
            return 0
        if len(self.layers) == 1:
            scheduled = num_scales - 1
        else:
            scheduled = round(layer_idx * (num_scales - 1) / (len(self.layers) - 1))
        preferred = int(scale_weights.detach().argmax().item())
        return int(round(0.5 * scheduled + 0.5 * preferred))

    def _position_encoding(self, xyz: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if not self.use_positional_guidance:
            return torch.zeros_like(like)
        return self.pos_proj(self.pos_encoder(xyz))

    def _prepare_memories(
        self,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        g_emb: torch.Tensor,
        multi_scale_tokens: list[torch.Tensor] | None,
        multi_scale_xyz: list[torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        use_multi_scale = (
            self.scale_projs is not None
            and multi_scale_tokens is not None
            and multi_scale_xyz is not None
            and len(multi_scale_tokens) > 0
        )
        if use_multi_scale:
            if len(multi_scale_tokens) != len(self.scale_projs):
                raise ValueError(
                    "multi_scale_tokens length must match decoder scale projections: "
                    f"{len(multi_scale_tokens)} vs {len(self.scale_projs)}"
                )
            mems = [
                self.memory_film(proj(tokens), g_emb)
                for proj, tokens in zip(self.scale_projs, multi_scale_tokens)
            ]
            xyzs = list(multi_scale_xyz)
        else:
            mems = [self.memory_film(self.scene_token_proj(scene_tokens), g_emb)]
            xyzs = [scene_xyz]

        poss = [self._position_encoding(xyz, mem) for xyz, mem in zip(xyzs, mems)]
        return mems, xyzs, poss

    def _predict(
        self,
        query_embed: torch.Tensor,
        dense_mask_feat: torch.Tensor,
        g_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_embed = F.normalize(self.head.mask_embed(query_embed, g_emb), dim=-1)
        out: dict[str, torch.Tensor] = {
            "mask_logits": self.head.mask_logit_scale() * (mask_embed @ dense_mask_feat.T),
            "score_logits": self.head.score_logits(query_embed, g_emb),
        }
        class_logits = self.head.class_logits(query_embed)
        if class_logits is not None:
            out["class_logits"] = class_logits
        return out

    def _forward_unbatched(
        self,
        point_feat: torch.Tensor,
        point_xyz: torch.Tensor | None,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        target_g: torch.Tensor | float,
        multi_scale_tokens: list[torch.Tensor] | None = None,
        multi_scale_xyz: list[torch.Tensor] | None = None,
    ) -> dict:
        assert point_feat.ndim == 2 and point_feat.shape[1] == self.in_channels
        assert scene_tokens.ndim == 2 and scene_tokens.shape[1] == self.in_channels
        assert scene_xyz.ndim == 2 and scene_xyz.shape[1] == 3
        if point_xyz is None:
            point_xyz = scene_xyz.new_zeros((point_feat.shape[0], 3))

        target_g_t = _as_granularity_tensor(target_g, device=point_feat.device)
        g_emb = self.granularity_encoder(target_g_t)
        scale_weights = self.scale_selector(g_emb, target_g_t)

        mems, xyzs, poss = self._prepare_memories(
            scene_tokens,
            scene_xyz,
            g_emb,
            multi_scale_tokens,
            multi_scale_xyz,
        )
        dense_mask_feat = F.normalize(self.point_mask_proj(point_feat), dim=-1)

        init_mem = mems[-1]
        init_xyz = xyzs[-1]
        q_feat, q_xyz, q_radius, _ = self.initializer(init_mem, init_xyz, target_g_t)
        g_float = _as_granularity_float(target_g_t)
        local_radius = self.local_radius_min + (
            self.local_radius_max - self.local_radius_min
        ) * g_float
        q_radius = q_radius.new_full(q_radius.shape, max(local_radius, 1e-3))
        q_feat = self.query_film(q_feat, g_emb)

        q_pos = self._position_encoding(q_xyz, q_feat)
        q_b = q_feat.unsqueeze(0)
        q_pos_b = q_pos.unsqueeze(0)

        aux_outputs: list[dict] = []
        local_stats_accum: list[dict[str, torch.Tensor]] = []
        raw_delta_norms: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        delta_step_norms: list[torch.Tensor] = []
        delta_norm_layer_means: list[torch.Tensor] = []
        delta_step_norm_layer_means: list[torch.Tensor] = []
        local_gate_layer_means: list[torch.Tensor] = []
        anchor_in_scene_ratios: list[torch.Tensor] = []
        xyz_min = scene_xyz.min(dim=0, keepdim=True).values
        xyz_max = scene_xyz.max(dim=0, keepdim=True).values

        for layer_idx, layer in enumerate(self.layers):
            s = self._scale_index_for_layer(layer_idx, len(mems), scale_weights)
            mem = mems[s]
            mem_xyz = xyzs[s].to(device=q_xyz.device, dtype=q_xyz.dtype)
            mem_pos = poss[s]
            scale_gain = 1.0 + scale_weights[s].to(dtype=mem.dtype)
            mem_b = (mem * scale_gain).unsqueeze(0)
            pos_b = mem_pos.unsqueeze(0)

            if self.use_local_aggregation:
                local_feat, local_stats = self.local_aggregator(
                    q_xyz,
                    mem,
                    mem_xyz,
                    q_radius,
                )
                gate = self.local_gates[layer_idx](g_emb).view(1, self.hidden_dim)
                q_b = q_b + (gate * local_feat).unsqueeze(0)
                local_gate_layer_means.append(gate.detach().mean())
            else:
                local_stats = {
                    "neighbor_count_mean": point_feat.new_tensor(0.0),
                    "neighbor_count_min": point_feat.new_tensor(0.0),
                    "neighbor_count_max": point_feat.new_tensor(0.0),
                    "neighbor_zero_frac": point_feat.new_tensor(0.0),
                }
                local_gate_layer_means.append(point_feat.new_tensor(0.0))
            local_stats_accum.append(local_stats)

            q_layer = self.layer_query_films[layer_idx](q_b.squeeze(0), g_emb)
            q_b = q_layer.unsqueeze(0)
            q_b = layer(q_b, q_pos_b, q_xyz, q_radius, mem_b, pos_b, target_g_t)

            delta_raw = torch.tanh(self.delta_heads[layer_idx](q_b.squeeze(0)))
            raw_delta_norms.append(torch.linalg.norm(delta_raw, dim=-1).detach())
            delta = delta_raw
            if self.delta_norm_max > 0.0:
                raw_norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
                delta = delta * torch.clamp(self.delta_norm_max / raw_norm, max=1.0)
            delta_norm = torch.linalg.norm(delta, dim=-1)
            delta_norms.append(delta_norm.detach())
            delta_norm_layer_means.append(delta_norm.detach().mean())
            delta_step = (
                self.anchor_refinement_runtime_scale
                * self.delta_scale
                * delta
                * q_radius[:, None].clamp_min(1e-3)
            )
            if not self.use_delta_refinement:
                delta_step = torch.zeros_like(delta_step)
            if self.delta_max_step is not None and self.delta_max_step > 0.0:
                step_norm = torch.linalg.norm(delta_step, dim=-1, keepdim=True).clamp_min(1e-6)
                delta_step = delta_step * torch.clamp(self.delta_max_step / step_norm, max=1.0)
            delta_step_norm = torch.linalg.norm(delta_step, dim=-1)
            delta_step_norms.append(delta_step_norm.detach())
            delta_step_norm_layer_means.append(delta_step_norm.detach().mean())
            q_xyz_pre_clamp = q_xyz + delta_step
            anchor_in_scene_ratios.append(
                ((q_xyz_pre_clamp >= xyz_min) & (q_xyz_pre_clamp <= xyz_max))
                .all(dim=-1)
                .to(dtype=point_feat.dtype)
                .mean()
                .detach()
            )
            q_xyz = q_xyz_pre_clamp
            if self.clamp_to_scene_bounds:
                q_xyz = torch.minimum(torch.maximum(q_xyz, xyz_min), xyz_max)
            q_pos = self._position_encoding(q_xyz, q_b.squeeze(0))
            q_pos_b = q_pos.unsqueeze(0)

            if layer_idx < len(self.layers) - 1:
                q_aux = q_b.squeeze(0)
                aux = self._predict(q_aux, dense_mask_feat, g_emb)
                aux["point_xyz"] = point_xyz
                aux["query_xyz"] = q_xyz
                aux["query_radius"] = q_radius
                aux_outputs.append(aux)

        refined_q = q_b.squeeze(0)
        out = self._predict(refined_q, dense_mask_feat, g_emb)
        out.update({
            "point_embed": dense_mask_feat,
            "point_xyz": point_xyz,
            "query_embed": refined_q,
            "query_xyz": q_xyz,
            "query_radius": q_radius,
            "scale_weights": scale_weights,
        })
        if aux_outputs:
            out["aux_outputs"] = aux_outputs

        if raw_delta_norms:
            raw_delta_all = torch.stack(raw_delta_norms, dim=0)
            raw_delta_mean = raw_delta_all.mean()
            raw_delta_max = raw_delta_all.max()
        else:
            raw_delta_mean = point_feat.new_tensor(0.0)
            raw_delta_max = point_feat.new_tensor(0.0)
        if delta_norms:
            delta_all = torch.stack(delta_norms, dim=0)
            delta_mean = delta_all.mean()
            delta_max = delta_all.max()
        else:
            delta_mean = point_feat.new_tensor(0.0)
            delta_max = point_feat.new_tensor(0.0)
        if delta_step_norms:
            delta_step_all = torch.stack(delta_step_norms, dim=0)
            delta_step_mean = delta_step_all.mean()
            delta_step_max = delta_step_all.max()
        else:
            delta_step_mean = point_feat.new_tensor(0.0)
            delta_step_max = point_feat.new_tensor(0.0)
        if anchor_in_scene_ratios:
            anchor_in_scene_ratio = torch.stack(anchor_in_scene_ratios).mean()
        else:
            anchor_in_scene_ratio = point_feat.new_tensor(1.0)
        if local_stats_accum:
            neighbor_mean = torch.stack([
                s["neighbor_count_mean"] for s in local_stats_accum
            ]).mean()
            neighbor_min = torch.stack([
                s["neighbor_count_min"] for s in local_stats_accum
            ]).min()
            neighbor_max = torch.stack([
                s["neighbor_count_max"] for s in local_stats_accum
            ]).max()
            zero_frac = torch.stack([
                s["neighbor_zero_frac"] for s in local_stats_accum
            ]).mean()
        else:
            neighbor_mean = point_feat.new_tensor(0.0)
            neighbor_min = point_feat.new_tensor(0.0)
            neighbor_max = point_feat.new_tensor(0.0)
            zero_frac = point_feat.new_tensor(0.0)

        diagnostics: dict[str, torch.Tensor] = {
            "query_anchor_xyz_mean": q_xyz.detach().mean(),
            "query_anchor_xyz_std": q_xyz.detach().std(unbiased=False),
            "query_anchor_xyz_min": q_xyz.detach().min(),
            "query_anchor_xyz_max": q_xyz.detach().max(),
            "query_delta_xyz_raw_mean": raw_delta_mean.detach(),
            "query_delta_xyz_raw_max": raw_delta_max.detach(),
            "query_delta_xyz_mean": delta_mean.detach(),
            "query_delta_xyz_max": delta_max.detach(),
            "query_delta_step_norm_mean": delta_step_mean.detach(),
            "query_delta_step_norm_max": delta_step_max.detach(),
            "query_anchor_in_scene_ratio": anchor_in_scene_ratio.detach(),
            "anchor_refinement_scale": point_feat.new_tensor(self.anchor_refinement_runtime_scale),
            "head_logit_scale_raw": self.head.logit_scale.detach(),
            "head_logit_scale": self.head.mask_logit_scale().detach(),
            "query_radius_mean": q_radius.detach().mean(),
            "query_radius_min": q_radius.detach().min(),
            "query_radius_max": q_radius.detach().max(),
            "local_neighbor_count_mean": neighbor_mean.detach(),
            "local_neighbor_count_min": neighbor_min.detach(),
            "local_neighbor_count_max": neighbor_max.detach(),
            "local_neighbor_zero_frac": zero_frac.detach(),
        }
        for i, weight in enumerate(scale_weights):
            diagnostics[f"scale_selector_weight_level_{i}"] = weight.detach()
        for i, value in enumerate(delta_norm_layer_means):
            diagnostics[f"delta_norm_layer_{i}"] = value.detach()
        for i, value in enumerate(delta_step_norm_layer_means):
            diagnostics[f"delta_step_norm_layer_{i}"] = value.detach()
        for i, value in enumerate(local_gate_layer_means):
            diagnostics[f"local_gate_mean_layer_{i}"] = value.detach()
        out["diagnostics"] = diagnostics

        return out

    def forward(
        self,
        point_feat: torch.Tensor,
        *,
        target_g: torch.Tensor | float = 0.5,
        point_xyz: torch.Tensor | None = None,
        scene_tokens: torch.Tensor | None = None,
        scene_xyz: torch.Tensor | None = None,
        multi_scale_tokens: list[torch.Tensor] | None = None,
        multi_scale_xyz: list[torch.Tensor] | None = None,
    ) -> dict:
        if scene_tokens is None or scene_xyz is None:
            raise ValueError(
                "scene_tokens and scene_xyz are required for the Transformer decoder."
            )

        ms_kw = dict(
            multi_scale_tokens=multi_scale_tokens,
            multi_scale_xyz=multi_scale_xyz,
        )
        if point_feat.ndim == 2:
            return self._forward_unbatched(
                point_feat,
                point_xyz,
                scene_tokens,
                scene_xyz,
                target_g,
                **ms_kw,
            )

        if point_feat.ndim == 3:
            assert point_feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got {tuple(point_feat.shape)}"
            )
            point_xyz_2d = None if point_xyz is None else point_xyz[0]
            out = self._forward_unbatched(
                point_feat[0],
                point_xyz_2d,
                scene_tokens,
                scene_xyz,
                target_g,
                **ms_kw,
            )
            batched: dict = {
                k: v.unsqueeze(0) for k, v in out.items()
                if isinstance(v, torch.Tensor)
            }
            if "aux_outputs" in out:
                batched["aux_outputs"] = [
                    {k: v.unsqueeze(0) for k, v in aux.items()}
                    for aux in out["aux_outputs"]
                ]
            if "diagnostics" in out:
                batched["diagnostics"] = out["diagnostics"]
            return batched

        raise ValueError(
            f"point_feat must be [N, C] or [1, N, C], got {tuple(point_feat.shape)}"
        )
