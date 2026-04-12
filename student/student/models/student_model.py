"""Full student instance segmentation model.

Pure composition of backbone + decoder.  Nothing else.

    input:  points [N, 3], features [N, C]
    output: nested dict with shared point_embed and per-head predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn

from student.data.batched_scene_batch import split_tensor_by_offsets
from student.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput
from student.models.instance_decoder import MultiHeadQueryInstanceDecoder


class StudentInstanceSegModel(nn.Module):
    """LitePT backbone -> multi-head query instance decoder.

    Parameters
    ----------
    backbone:
        Pre-constructed :class:`LitePTBackbone`.
    decoder:
        Pre-constructed :class:`MultiHeadQueryInstanceDecoder`.
    """

    def __init__(
        self,
        backbone: LitePTBackbone,
        decoder: MultiHeadQueryInstanceDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    @property
    def out_channels(self) -> int:
        return self.backbone.out_channels

    @property
    def num_queries(self) -> int:
        return self.decoder.num_queries

    def parameter_groups(self, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Return param groups with separate LR scaling for backbone vs decoder.

        The pretrained backbone should train at a lower LR than the randomly
        initialized Transformer decoder to avoid destabilizing learned features.
        """
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": backbone_lr_scale},
            {"params": list(self.decoder.parameters()), "lr_scale": 1.0},
        ]

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        *,
        point_offsets: torch.Tensor | None = None,
    ) -> dict | list[dict]:
        """
        Parameters
        ----------
        points   : [N, 3] or [1, N, 3]
        features : [N, C] or [1, N, C]

        Returns
        -------
        Nested dict:
            point_embed : [N, D]
            heads:
                g02: {mask_logits, score_logits, query_embed}
                g05: {mask_logits, score_logits, query_embed}
                g08: {mask_logits, score_logits, query_embed}
        """
        bb: LitePTBackboneOutput = self.backbone(
            points,
            features,
            point_offsets=point_offsets,
        )
        if bb.point_offsets.numel() <= 1:
            return self.decoder(
                point_feat=bb.point_feat,
                point_xyz=bb.point_xyz,
                scene_tokens=bb.scene_tokens,
                scene_xyz=bb.scene_xyz,
                multi_scale_tokens=bb.multi_scale_tokens or None,
                multi_scale_xyz=bb.multi_scale_xyz or None,
            )

        point_feats = split_tensor_by_offsets(bb.point_feat, bb.point_offsets)
        point_xyzs = split_tensor_by_offsets(bb.point_xyz, bb.point_offsets)
        scene_tokens = split_tensor_by_offsets(
            bb.scene_tokens,
            bb.scene_token_offsets,
        )
        scene_xyzs = split_tensor_by_offsets(
            bb.scene_xyz,
            bb.scene_token_offsets,
        )

        split_multi_scale_tokens = [
            split_tensor_by_offsets(tokens, offsets)
            for tokens, offsets in zip(bb.multi_scale_tokens, bb.multi_scale_offsets)
        ]
        split_multi_scale_xyz = [
            split_tensor_by_offsets(xyz, offsets)
            for xyz, offsets in zip(bb.multi_scale_xyz, bb.multi_scale_offsets)
        ]

        outputs: list[dict] = []
        for scene_idx in range(bb.point_offsets.numel()):
            multi_scale_tokens = (
                [tokens[scene_idx] for tokens in split_multi_scale_tokens]
                if split_multi_scale_tokens
                else None
            )
            multi_scale_xyz = (
                [xyz[scene_idx] for xyz in split_multi_scale_xyz]
                if split_multi_scale_xyz
                else None
            )
            outputs.append(self.decoder(
                point_feat=point_feats[scene_idx],
                point_xyz=point_xyzs[scene_idx],
                scene_tokens=scene_tokens[scene_idx],
                scene_xyz=scene_xyzs[scene_idx],
                multi_scale_tokens=multi_scale_tokens,
                multi_scale_xyz=multi_scale_xyz,
            ))
        return outputs


def build_student_model(
    litept_root: str,
    in_channels: int,
    grid_size: float = 0.02,
    litept_variant: str = "litept_s_star",
    litept_kwargs: dict | None = None,
    hidden_dim: int = 256,
    num_queries: int = 128,
    num_queries_by_granularity: dict[str, int] | None = None,
    granularities: tuple[str, ...] = ("g02", "g05", "g08"),
    num_decoder_layers: int = 4,
    num_decoder_heads: int = 8,
    query_init: str = "hybrid",
    use_positional_guidance: bool = True,
    learned_query_ratio: float = 0.25,
    multi_scale: bool = False,
) -> StudentInstanceSegModel:
    """Convenience factory from scalar config values.

    Parameters
    ----------
    num_queries:
        Default query count for all heads.
    num_queries_by_granularity:
        Optional per-head override, e.g. ``{"g02": 300, "g05": 150, "g08": 100}``.
        Granularities not listed fall back to *num_queries*.
    query_init:
        ``"hybrid"`` uses *learned_query_ratio* (default).
        ``"learned"`` forces all queries to be learned embeddings.
        ``"scene"`` forces all queries to be scene-sampled.
    litept_variant:
        ``"litept_s_star"`` (default) uses the deeper decoder from instance-seg configs;
        ``"litept_s"`` matches semantic-seg LitePT-S (``dec_depths`` all zero). Must match
        any pretrained checkpoint architecture.
    litept_kwargs:
        Optional extra keyword arguments merged into :class:`~litept.model.LitePT`.
    multi_scale:
        Enable multi-scale decoder feature capture from LitePT.  Each student
        Transformer layer cross-attends to a different backbone decoder scale
        (coarse → fine).  Requires ``litept_variant="litept_s_star"`` for
        meaningful per-scale features.
    """
    backbone = LitePTBackbone(
        litept_root=litept_root,
        in_channels=in_channels,
        grid_size=grid_size,
        litept_variant=litept_variant,
        litept_kwargs=litept_kwargs,
        multi_scale=multi_scale,
    )

    if query_init == "learned":
        effective_ratio = 1.0
    elif query_init == "scene":
        effective_ratio = 0.0
    else:
        effective_ratio = learned_query_ratio

    if num_queries_by_granularity:
        effective_queries: int | dict[str, int] = {
            g: num_queries_by_granularity.get(g, num_queries)
            for g in granularities
        }
    else:
        effective_queries = num_queries

    decoder = MultiHeadQueryInstanceDecoder(
        in_channels=backbone.out_channels,
        hidden_dim=hidden_dim,
        num_queries=effective_queries,
        granularities=granularities,
        num_layers=num_decoder_layers,
        num_heads=num_decoder_heads,
        learned_ratio=effective_ratio,
        use_positional_guidance=use_positional_guidance,
        multi_scale_channels=backbone.multi_scale_channels,
    )
    return StudentInstanceSegModel(backbone=backbone, decoder=decoder)
