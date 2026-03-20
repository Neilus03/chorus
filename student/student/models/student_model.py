"""Full student instance segmentation model.

Pure composition of backbone + decoder.  Nothing else.

    input:  points [N, 3], features [N, C]
    output: nested dict with shared point_embed and per-head predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
    ) -> dict:
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
        bb: LitePTBackboneOutput = self.backbone(points, features)
        return self.decoder(
            point_feat=bb.point_feat,
            point_xyz=bb.point_xyz,
            scene_tokens=bb.scene_tokens,
            scene_xyz=bb.scene_xyz,
        )


def build_student_model(
    litept_root: str,
    in_channels: int,
    grid_size: float = 0.02,
    hidden_dim: int = 256,
    num_queries: int = 128,
    num_queries_by_granularity: dict[str, int] | None = None,
    granularities: tuple[str, ...] = ("g02", "g05", "g08"),
    num_decoder_layers: int = 4,
    num_decoder_heads: int = 8,
    query_init: str = "hybrid",
    use_positional_guidance: bool = True,
    learned_query_ratio: float = 0.25,
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
    """
    backbone = LitePTBackbone(
        litept_root=litept_root,
        in_channels=in_channels,
        grid_size=grid_size,
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
    )
    return StudentInstanceSegModel(backbone=backbone, decoder=decoder)
