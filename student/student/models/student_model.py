"""Full student instance segmentation model.

Pure composition of backbone + decoder.  Nothing else.

    input:  points [N, 3], features [N, C]
    output: dict with mask_logits [Q, N], score_logits [Q], ...
"""

from __future__ import annotations

import torch
import torch.nn as nn

from student.models.litept_wrapper import LitePTBackbone
from student.models.instance_decoder import QueryInstanceDecoder


class StudentInstanceSegModel(nn.Module):
    """LitePT backbone → query instance decoder.

    Parameters
    ----------
    backbone:
        Pre-constructed :class:`LitePTBackbone`.
    decoder:
        Pre-constructed :class:`QueryInstanceDecoder`.
    """

    def __init__(
        self,
        backbone: LitePTBackbone,
        decoder: QueryInstanceDecoder,
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

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        points   : [N, 3] or [1, N, 3]
        features : [N, C] or [1, N, C]

        Returns
        -------
        Dict with at least:
            mask_logits  : [Q, N]
            score_logits : [Q]
            point_embed  : [N, D]
            query_embed  : [Q, D]
        """
        point_feat = self.backbone(points, features)
        return self.decoder(point_feat)


def build_student_model(
    litept_root: str,
    in_channels: int,
    grid_size: float = 0.02,
    hidden_dim: int = 256,
    num_queries: int = 128,
) -> StudentInstanceSegModel:
    """Convenience factory from scalar config values."""
    backbone = LitePTBackbone(
        litept_root=litept_root,
        in_channels=in_channels,
        grid_size=grid_size,
    )
    decoder = QueryInstanceDecoder(
        in_channels=backbone.out_channels,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
    )
    return StudentInstanceSegModel(backbone=backbone, decoder=decoder)
