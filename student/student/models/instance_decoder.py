"""Simple query-based instance decoder.

Takes per-point features from LitePT and produces per-query mask logits
and objectness scores — enough for set-prediction instance segmentation.

Output shapes (unbatched):
    mask_logits  : [Q, N]   — one row per query, one column per point
    score_logits : [Q]      — one objectness scalar per query
    point_embed  : [N, D]   — point embeddings in decoder space
    query_embed  : [Q, D]   — refined query embeddings
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryInstanceDecoder(nn.Module):
    """Project points and learned queries into a shared space, then compute
    mask logits via cosine similarity and per-query objectness scores.

    Parameters
    ----------
    in_channels:
        Dimension of per-point input features (72 for default LitePT-S).
    hidden_dim:
        Decoder's internal embedding dimension.
    num_queries:
        Number of learnable instance query slots.
    normalize_mask_embeddings:
        If True, mask logits are computed as scaled cosine similarity.
        If False, plain dot-product scaled by 1/sqrt(D).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_queries: int = 128,
        normalize_mask_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.normalize_mask_embeddings = bool(normalize_mask_embeddings)

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.point_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        self.query_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

    def _forward_unbatched(
        self, point_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert point_feat.ndim == 2, (
            f"point_feat must be [N, C], got {tuple(point_feat.shape)}"
        )
        assert point_feat.shape[1] == self.in_channels, (
            f"Expected {self.in_channels} channels, got {point_feat.shape[1]}"
        )

        point_hidden = self.input_proj(point_feat)
        point_embed = point_hidden + self.point_head(point_hidden)

        query_hidden = self.query_embed.weight
        query_embed = query_hidden + self.query_head(query_hidden)

        if self.normalize_mask_embeddings:
            point_mask = F.normalize(point_embed, dim=-1)
            query_mask = F.normalize(query_embed, dim=-1)
            mask_logits = self.logit_scale.exp() * (query_mask @ point_mask.T)
        else:
            mask_logits = (query_embed @ point_embed.T) / math.sqrt(self.hidden_dim)

        mask_weights = torch.softmax(mask_logits, dim=-1)        # [Q, N]
        pooled_query_feat = mask_weights @ point_embed             # [Q, D]
        refined_query_embed = query_embed + pooled_query_feat      # [Q, D]

        score_logits = self.score_head(refined_query_embed).squeeze(-1)

        return {
            "mask_logits": mask_logits,
            "score_logits": score_logits,
            "point_embed": point_embed,
            "query_embed": refined_query_embed,
        }

    def forward(
        self, point_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        point_feat : [N, C] or [1, N, C]

        Returns
        -------
        Dict with mask_logits [Q, N], score_logits [Q], etc.
        If input was [1, N, C], outputs get an extra leading dim of 1.
        """
        if point_feat.ndim == 2:
            return self._forward_unbatched(point_feat)

        if point_feat.ndim == 3:
            assert point_feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got {tuple(point_feat.shape)}"
            )
            out = self._forward_unbatched(point_feat[0])
            return {k: v.unsqueeze(0) for k, v in out.items()}

        raise ValueError(
            f"point_feat must be [N, C] or [1, N, C], got {tuple(point_feat.shape)}"
        )
