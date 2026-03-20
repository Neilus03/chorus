"""Multi-head query-based instance decoder.

Shared trunk computes point embeddings once from LitePT features.
Each granularity head independently decodes its own instance masks
and scores from the same point representation.

Output structure (unbatched):
    {
        "point_embed": [N, D],
        "heads": {
            "g02": {"mask_logits": [Q, N], "score_logits": [Q], "query_embed": [Q, D]},
            "g05": {"mask_logits": [Q, N], "score_logits": [Q], "query_embed": [Q, D]},
            "g08": {"mask_logits": [Q, N], "score_logits": [Q], "query_embed": [Q, D]},
        },
    }
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GranularityHead(nn.Module):
    """One query-based prediction head for a single granularity level."""

    def __init__(self, hidden_dim: int, num_queries: int) -> None:
        super().__init__()
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.query_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))


class MultiHeadQueryInstanceDecoder(nn.Module):
    """Shared-trunk decoder with per-granularity prediction heads.

    Parameters
    ----------
    in_channels:
        Dimension of per-point input features (72 for default LitePT-S).
    hidden_dim:
        Decoder's internal embedding dimension.
    num_queries:
        Number of learnable instance query slots per head.
    granularities:
        Granularity keys, one head is created per entry.
    normalize_mask_embeddings:
        If True, mask logits are computed as scaled cosine similarity.
        If False, plain dot-product scaled by 1/sqrt(D).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_queries: int = 128,
        granularities: tuple[str, ...] = ("g02", "g05", "g08"),
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

        self.heads = nn.ModuleDict({
            g: GranularityHead(self.hidden_dim, self.num_queries)
            for g in granularities
        })

    def _forward_unbatched(
        self, point_feat: torch.Tensor,
    ) -> dict:
        assert point_feat.ndim == 2, (
            f"point_feat must be [N, C], got {tuple(point_feat.shape)}"
        )
        assert point_feat.shape[1] == self.in_channels, (
            f"Expected {self.in_channels} channels, got {point_feat.shape[1]}"
        )

        # --- shared trunk: compute once ---
        point_hidden = self.input_proj(point_feat)
        point_embed = point_hidden + self.point_head(point_hidden)

        out: dict = {"point_embed": point_embed, "heads": {}}

        # --- per-granularity heads ---
        for g, head in self.heads.items():
            query_hidden = head.query_embed.weight
            query_embed = query_hidden + head.query_head(query_hidden)

            if self.normalize_mask_embeddings:
                point_mask = F.normalize(point_embed, dim=-1)
                query_mask = F.normalize(query_embed, dim=-1)
                mask_logits = head.logit_scale.exp() * (query_mask @ point_mask.T)
            else:
                mask_logits = (query_embed @ point_embed.T) / math.sqrt(
                    self.hidden_dim
                )

            mask_weights = torch.softmax(mask_logits, dim=-1)
            pooled_query_feat = mask_weights @ point_embed
            refined_query_embed = query_embed + pooled_query_feat

            score_logits = head.score_head(refined_query_embed).squeeze(-1)

            out["heads"][g] = {
                "mask_logits": mask_logits,
                "score_logits": score_logits,
                "query_embed": refined_query_embed,
            }

        return out

    def forward(
        self,
        point_feat: torch.Tensor,
        *,
        point_xyz: torch.Tensor | None = None,
        scene_tokens: torch.Tensor | None = None,
        scene_xyz: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        point_feat : [N, C] or [1, N, C]
        point_xyz : [N, 3], optional (unused until Phase 2)
        scene_tokens : [V, C], optional (unused until Phase 2)
        scene_xyz : [V, 3], optional (unused until Phase 2)

        Returns
        -------
        Nested dict with shared point_embed and per-head predictions.
        If input was [1, N, C], all tensor values get an extra leading dim of 1.
        """
        if point_feat.ndim == 2:
            return self._forward_unbatched(point_feat)

        if point_feat.ndim == 3:
            assert point_feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got {tuple(point_feat.shape)}"
            )
            out = self._forward_unbatched(point_feat[0])
            batched: dict = {"point_embed": out["point_embed"].unsqueeze(0), "heads": {}}
            for g, head_out in out["heads"].items():
                batched["heads"][g] = {
                    k: v.unsqueeze(0) for k, v in head_out.items()
                }
            return batched

        raise ValueError(
            f"point_feat must be [N, C] or [1, N, C], got {tuple(point_feat.shape)}"
        )
