"""Continuous granularity-conditioned single-head instance decoder.

Replaces :class:`MultiHeadQueryInstanceDecoder` with a single shared query
pool conditioned by a scalar granularity g ∈ [0, 1] mapped through a small
MLP into the decoder's hidden dimension.

Architecture:  Mask3D-style iterative decoder  +  MAFT-style positional guidance
             + QueryFormer-style hybrid query init  +  Continuous scale embedding

Output structure (unbatched):
    {
        "point_embed": [N, D],
        "mask_logits": [Q, N],
        "score_logits": [Q],
        "query_embed": [Q, D],
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from student.models.instance_decoder import (
    FourierPosEnc,
    GranularityHead,
    HybridQueryInitializer,
    QueryDecoderLayer,
)


class ContinuousQueryInstanceDecoder(nn.Module):
    """Iterative query-refinement decoder with continuous granularity conditioning.

    Instead of per-granularity query pools and output heads, a single query
    pool is conditioned with a scale embedding ``c = MLP(g)`` that shifts
    the decoder's attention to group points at the requested physical scale.

    Parameters
    ----------
    in_channels:
        Backbone feature dimension (72 for default LitePT-S).
    hidden_dim:
        Decoder's internal embedding dimension.
    num_queries:
        Instance query slots.
    num_layers:
        Shared Transformer decoder layers (default 4).
    num_heads:
        Attention heads per decoder layer (default 8).
    learned_ratio:
        Fraction of queries that are learned embeddings (rest are scene-sampled).
    use_positional_guidance:
        Whether to use Fourier positional encoding.
    multi_scale_channels:
        Per-level feature dimensions for multi-scale cross-attention.
    """

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
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.use_positional_guidance = use_positional_guidance

        # ── continuous scale embedding ──
        # LayerNorm prevents scale explosion; zero-init output ≈ identity at start
        self.granularity_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self._init_granularity_mlp()

        # ── single-scale scene memory projection ──
        self.scene_token_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── multi-scale per-level projections [Mask3D / M2F3D style] ──
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

        # ── dense point projection for final mask dot-product ──
        self.point_mask_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── positional encoding [MAFT] ──
        num_fourier_bands = 16
        pos_dim = 3 * num_fourier_bands * 2  # 96
        self.pos_encoder = FourierPosEnc(in_dim=3, num_bands=num_fourier_bands)
        self.pos_proj = nn.Linear(pos_dim, hidden_dim)

        # ── single query initializer (replaces per-head ModuleDict) ──
        self.initializer = HybridQueryInitializer(
            hidden_dim, num_queries, learned_ratio,
        )

        # ── shared Transformer trunk ──
        self.layers = nn.ModuleList([
            QueryDecoderLayer(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # ── single output head (replaces per-granularity ModuleDict) ──
        self.head = GranularityHead(hidden_dim)

    # ------------------------------------------------------------------ #

    def _init_granularity_mlp(self) -> None:
        """Careful init so c ≈ 0 at start → queries behave as unconditional."""
        for m in self.granularity_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero the final layer's weight for near-zero output at init
        final_linear = self.granularity_mlp[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)

    # ------------------------------------------------------------------ #

    def _scale_index_for_layer(self, layer_idx: int, num_scales: int) -> int:
        """Map a trunk layer index to a pyramid scale index (coarse → fine)."""
        num_layers = len(self.layers)
        if num_layers == 1:
            return num_scales - 1
        return round(layer_idx * (num_scales - 1) / (num_layers - 1))

    # ------------------------------------------------------------------ #

    def _forward_unbatched(
        self,
        point_feat: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        target_g: torch.Tensor | float,
        multi_scale_tokens: list[torch.Tensor] | None = None,
        multi_scale_xyz: list[torch.Tensor] | None = None,
    ) -> dict:
        """Core forward on unbatched tensors.

        Parameters
        ----------
        point_feat         : [N, C_in] — dense per-point backbone features
        scene_tokens       : [V, C_in] — sparse voxel backbone features (finest)
        scene_xyz          : [V, 3]    — voxel centroids (finest)
        target_g           : scalar or [1] tensor — granularity condition
        multi_scale_tokens : optional list of [V_i, C_i] coarse → fine
        multi_scale_xyz    : optional list of [V_i, 3]   coarse → fine
        """
        assert point_feat.ndim == 2 and point_feat.shape[1] == self.in_channels
        assert scene_tokens.ndim == 2 and scene_tokens.shape[1] == self.in_channels
        assert scene_xyz.ndim == 2 and scene_xyz.shape[1] == 3

        # ── condition vector ──
        if not isinstance(target_g, torch.Tensor):
            target_g = torch.tensor(
                [target_g], device=point_feat.device, dtype=torch.float32,
            )
        target_g = target_g.to(device=point_feat.device, dtype=torch.float32)
        c = self.granularity_mlp(target_g.view(1, 1))  # [1, D]

        use_multi_scale = (
            self.scale_projs is not None
            and multi_scale_tokens is not None
            and len(multi_scale_tokens) > 0
        )

        # ── scene memory: multi-scale or single-scale ──

        if use_multi_scale:
            num_scales = len(multi_scale_tokens)
            scale_mems = [
                proj(tokens)
                for proj, tokens in zip(self.scale_projs, multi_scale_tokens)
            ]
            if self.use_positional_guidance:
                scale_poss = [
                    self.pos_proj(self.pos_encoder(xyz))
                    for xyz in multi_scale_xyz
                ]
            else:
                scale_poss = [torch.zeros_like(m) for m in scale_mems]

            scale_mems_b = [m.unsqueeze(0) for m in scale_mems]
            scale_poss_b = [p.unsqueeze(0) for p in scale_poss]

            init_mem = scale_mems[-1]      # finest for query init
            init_xyz = multi_scale_xyz[-1]
        else:
            scene_mem = self.scene_token_proj(scene_tokens)
            if self.use_positional_guidance:
                scene_pos = self.pos_proj(self.pos_encoder(scene_xyz))
            else:
                scene_pos = torch.zeros_like(scene_mem)

            scene_mem_b = scene_mem.unsqueeze(0)
            scene_pos_b = scene_pos.unsqueeze(0)

            init_mem = scene_mem
            init_xyz = scene_xyz

        # ── dense mask features (always finest resolution) ──

        dense_mask_feat = self.point_mask_proj(point_feat)     # [N, D]
        dense_mask_feat = F.normalize(dense_mask_feat, dim=-1)

        # ── query init + conditioning ──

        q_feat, q_xyz = self.initializer(init_mem, init_xyz)

        if self.use_positional_guidance:
            q_pos = self.pos_proj(self.pos_encoder(q_xyz))
        else:
            q_pos = torch.zeros_like(q_feat)

        # Inject condition: additive shift (broadcast [1, D] + [Q, D])
        q_feat = q_feat + c.squeeze(0)

        q_b = q_feat.unsqueeze(0)        # [1, Q, D]
        q_pos_b = q_pos.unsqueeze(0)     # [1, Q, D]

        # ── Transformer trunk with auxiliary outputs ──

        num_layers = len(self.layers)
        aux_outputs: list[dict] = []

        for layer_idx, layer in enumerate(self.layers):
            if use_multi_scale:
                s = self._scale_index_for_layer(layer_idx, num_scales)
                q_b = layer(q_b, q_pos_b, scale_mems_b[s], scale_poss_b[s])
            else:
                q_b = layer(q_b, q_pos_b, scene_mem_b, scene_pos_b)

            if layer_idx < num_layers - 1:
                # Auxiliary prediction (matches existing aux_outputs contract)
                q_aux = q_b.squeeze(0)
                me_aux = F.normalize(self.head.mask_embed(q_aux), dim=-1)
                logit_s = self.head.logit_scale.exp()
                aux_outputs.append({
                    "mask_logits": logit_s * (me_aux @ dense_mask_feat.T),
                    "score_logits": self.head.score_head(q_aux).squeeze(-1),
                })

        # ── final prediction ──

        refined_q = q_b.squeeze(0)       # [Q, D]
        mask_embed = F.normalize(self.head.mask_embed(refined_q), dim=-1)
        logit_s = self.head.logit_scale.exp()

        out: dict = {
            "point_embed": dense_mask_feat,
            "mask_logits": logit_s * (mask_embed @ dense_mask_feat.T),  # [Q, N]
            "score_logits": self.head.score_head(refined_q).squeeze(-1),  # [Q]
            "query_embed": refined_q,
        }
        if aux_outputs:
            out["aux_outputs"] = aux_outputs

        return out

    # ------------------------------------------------------------------ #

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
        """
        Parameters
        ----------
        point_feat         : [N, C] or [1, N, C] — dense per-point backbone features
        target_g           : scalar or [1] tensor — granularity condition ∈ [0, 1]
        point_xyz          : [N, 3], optional — reserved for future anchor updates
        scene_tokens       : [V, C] — sparse voxel features from backbone (**required**)
        scene_xyz          : [V, 3] — voxel centroids (**required**)
        multi_scale_tokens : optional list of [V_i, C_i] per decoder scale (coarse → fine)
        multi_scale_xyz    : optional list of [V_i, 3]   per decoder scale (coarse → fine)

        Returns
        -------
        Dict with ``point_embed``, ``mask_logits``, ``score_logits``,
        ``query_embed``, and optionally ``aux_outputs``.
        """
        if scene_tokens is None or scene_xyz is None:
            raise ValueError(
                "scene_tokens and scene_xyz are required for the Transformer "
                "decoder. Pass them from LitePTBackboneOutput."
            )

        ms_kw = dict(
            multi_scale_tokens=multi_scale_tokens,
            multi_scale_xyz=multi_scale_xyz,
        )

        if point_feat.ndim == 2:
            return self._forward_unbatched(
                point_feat, scene_tokens, scene_xyz, target_g, **ms_kw,
            )

        if point_feat.ndim == 3:
            assert point_feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got {tuple(point_feat.shape)}"
            )
            out = self._forward_unbatched(
                point_feat[0], scene_tokens, scene_xyz, target_g, **ms_kw,
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
            return batched

        raise ValueError(
            f"point_feat must be [N, C] or [1, N, C], got {tuple(point_feat.shape)}"
        )
