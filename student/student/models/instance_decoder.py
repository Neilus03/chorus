"""Multi-head query-based instance decoder with iterative Transformer refinement.

Shared Transformer trunk refines instance queries via self-attention (query-query
interaction) and cross-attention (query-to-scene reasoning) over sparse voxel
tokens.  Per-granularity output heads project refined queries into masks and scores.

Architecture:  Mask3D-style iterative decoder  +  MAFT-style positional guidance
             + QueryFormer-style hybrid query init  +  SPFormer-style sparse memory

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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding [MAFT §3.2] ─────────────────────────────────────


class FourierPosEnc(nn.Module):
    """Maps 3D coordinates to high-dimensional Fourier features.

    Uses sin(x·2^k) and cos(x·2^k) for k = 0 .. num_bands-1 on each of the
    3 coordinate dimensions.  Output size: ``in_dim * num_bands * 2``.
    """

    def __init__(self, in_dim: int = 3, num_bands: int = 16) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 3]  →  [..., in_dim * num_bands * 2]
        xb = x.unsqueeze(-1) * self.freqs                         # [..., 3, B]
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)   # [..., 3, 2B]
        return enc.flatten(start_dim=-2)


# ── Transformer Decoder Layer [Mask3D §3.3 / SPFormer / DETR] ───────────


class QueryDecoderLayer(nn.Module):
    """Single pre-norm Transformer decoder layer.

    Sub-layers (all with residual connections):
      1. Self-attention among queries   — resolve duplicates
      2. Cross-attention to scene tokens — read geometry
      3. Feed-forward network            — non-linear mixing

    Positional encodings are added to Q and K only (not V) so the value
    stream carries content features uncontaminated by position.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 8, ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
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
        scene_mem: torch.Tensor,
        scene_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs have shape ``[1, seq_len, D]`` (batch_first).

        Parameters
        ----------
        q         : [1, Q, D] — instance queries (content)
        q_pos     : [1, Q, D] — query positional encodings
        scene_mem : [1, V, D] — projected sparse voxel features
        scene_pos : [1, V, D] — scene positional encodings
        """
        # 1. Self-attention — queries interact to resolve duplicates
        q_n = self.norm_sa(q)
        q_with_pos = q_n + q_pos
        sa_out, _ = self.self_attn(
            query=q_with_pos, key=q_with_pos, value=q_n,
        )
        q = q + sa_out

        # 2. Cross-attention — queries read scene geometry
        q_n2 = self.norm_ca(q)
        ca_out, _ = self.cross_attn(
            query=q_n2 + q_pos,
            key=scene_mem + scene_pos,
            value=scene_mem,
        )
        q = q + ca_out

        # 3. Feed-forward
        q = q + self.ffn(self.norm_ff(q))
        return q


# ── Hybrid Query Initialization [QueryFormer §3.2 / MAFT / SGIFormer] ──


class HybridQueryInitializer(nn.Module):
    """Initializes queries from a mix of scene tokens and learned embeddings.

    Scene-sampled queries start physically on real geometry (high coverage,
    as advocated by QueryFormer).  Learned queries act as flexible wildcards.

    Parameters
    ----------
    hidden_dim:
        Feature dimension (must match the decoder hidden space).
    num_queries:
        Total number of queries this initializer produces.
    learned_ratio:
        Fraction of queries that are learned (the rest are scene-sampled).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        learned_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_learned = int(num_queries * learned_ratio)
        self.num_scene = num_queries - self.num_learned

        self.learned_queries = nn.Embedding(self.num_learned, hidden_dim)
        self.learned_xyz = nn.Embedding(self.num_learned, 3)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        scene_tokens : [V, D] — projected sparse voxel features
        scene_xyz    : [V, 3] — voxel centroids

        Returns
        -------
        q_feat : [Q, D] — initial query features
        q_xyz  : [Q, 3] — initial query spatial anchors
        """
        V = scene_tokens.shape[0]

        if V >= self.num_scene:
            idx = torch.randperm(V, device=scene_tokens.device)[:self.num_scene]
            scene_q = scene_tokens[idx]
            scene_q_xyz = scene_xyz[idx]
        else:
            pad_n = self.num_scene - V
            pad_idx = torch.randint(V, (pad_n,), device=scene_tokens.device)
            scene_q = torch.cat([scene_tokens, scene_tokens[pad_idx]], dim=0)
            scene_q_xyz = torch.cat([scene_xyz, scene_xyz[pad_idx]], dim=0)

        q_feat = torch.cat([scene_q, self.learned_queries.weight], dim=0)
        q_xyz = torch.cat([scene_q_xyz, self.learned_xyz.weight], dim=0)
        return q_feat, q_xyz


# ── Lightweight Output Head [Mask3D / Mask2Former pattern] ───────────────


class GranularityHead(nn.Module):
    """Projects refined queries into mask embeddings and objectness scores.

    All reasoning happens in the shared Transformer trunk.  This head only
    maps final D-dim refined queries into task-specific outputs.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mask_embed = nn.Sequential(
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


# ── Main Decoder ─────────────────────────────────────────────────────────


class MultiHeadQueryInstanceDecoder(nn.Module):
    """Iterative query-refinement decoder with per-granularity output heads.

    Architecture overview:
      1. Project sparse scene tokens  →  decoder hidden space (cross-attn memory)
      2. Project dense point features →  mask embedding space (final dot-product)
      3. Fourier-encode scene & query coordinates  →  positional guidance
      4. Per granularity:
         a. HybridQueryInitializer  →  initial queries + spatial anchors
         b. Shared Transformer trunk (self-attn + cross-attn + FFN) × L layers
         c. GranularityHead  →  mask_logits + score_logits

    Parameters
    ----------
    in_channels:
        Backbone feature dimension (72 for default LitePT-S).
    hidden_dim:
        Decoder's internal embedding dimension.
    num_queries:
        Instance query slots.  Either a single int (same for all heads)
        or a dict mapping granularity keys to per-head counts,
        e.g. ``{"g02": 300, "g05": 150, "g08": 100}``.
    granularities:
        Keys identifying each granularity level.
    num_layers:
        Shared Transformer decoder layers (default 4).
    num_heads:
        Attention heads per decoder layer (default 8).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_queries: int | dict[str, int] = 128,
        granularities: tuple[str, ...] = ("g02", "g05", "g08"),
        num_layers: int = 4,
        num_heads: int = 8,
        learned_ratio: float = 0.25,
        use_positional_guidance: bool = True,
        multi_scale_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.granularities = granularities
        self.use_positional_guidance = use_positional_guidance

        if isinstance(num_queries, dict):
            self.num_queries_per_head = {g: int(num_queries[g]) for g in granularities}
        else:
            self.num_queries_per_head = {g: int(num_queries) for g in granularities}
        self.num_queries = max(self.num_queries_per_head.values())

        # --- single-scale scene memory projection (always built for fallback) ---
        self.scene_token_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- multi-scale per-level projections [Mask3D / M2F3D style] ---
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

        # --- dense point projection for final mask dot-product ---
        self.point_mask_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- positional encoding [MAFT] ---
        num_fourier_bands = 16
        pos_dim = 3 * num_fourier_bands * 2  # 96
        self.pos_encoder = FourierPosEnc(in_dim=3, num_bands=num_fourier_bands)
        self.pos_proj = nn.Linear(pos_dim, hidden_dim)

        # --- per-head query initializers [QueryFormer / MAFT] ---
        self.initializers = nn.ModuleDict({
            g: HybridQueryInitializer(
                hidden_dim, self.num_queries_per_head[g], learned_ratio,
            )
            for g in granularities
        })

        # --- shared Transformer trunk [Mask3D] ---
        self.layers = nn.ModuleList([
            QueryDecoderLayer(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # --- per-head output projections ---
        self.heads = nn.ModuleDict({
            g: GranularityHead(hidden_dim) for g in granularities
        })

    # ------------------------------------------------------------------ #

    def _scale_index_for_layer(self, layer_idx: int, num_scales: int) -> int:
        """Map a trunk layer index to a pyramid scale index (coarse → fine).

        Spreads layers evenly across available scales so that:
        - The first layer always uses the coarsest scale (index 0).
        - The last layer always uses the finest scale (index num_scales-1).
        - Intermediate layers are linearly interpolated between the two.

        When ``num_layers > num_scales``, extra layers repeat the nearest
        scale.  When ``num_layers < num_scales``, some intermediate scales
        are skipped.
        """
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
        multi_scale_tokens: list[torch.Tensor] | None = None,
        multi_scale_xyz: list[torch.Tensor] | None = None,
    ) -> dict:
        """Core forward on unbatched tensors.

        Parameters
        ----------
        point_feat         : [N, C_in] — dense per-point backbone features
        scene_tokens       : [V, C_in] — sparse voxel backbone features (finest)
        scene_xyz          : [V, 3]    — voxel centroids (finest)
        multi_scale_tokens : optional list of [V_i, C_i] coarse → fine
        multi_scale_xyz    : optional list of [V_i, 3]   coarse → fine
        """
        assert point_feat.ndim == 2 and point_feat.shape[1] == self.in_channels
        assert scene_tokens.ndim == 2 and scene_tokens.shape[1] == self.in_channels
        assert scene_xyz.ndim == 2 and scene_xyz.shape[1] == 3

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

        num_layers = len(self.layers)
        out: dict = {"point_embed": dense_mask_feat, "heads": {}}
        aux_by_layer: list[dict] = [{"heads": {}} for _ in range(num_layers - 1)]

        # ── per-granularity query decode ──

        for g in self.granularities:
            q_feat, q_xyz = self.initializers[g](init_mem, init_xyz)

            if self.use_positional_guidance:
                q_pos = self.pos_proj(self.pos_encoder(q_xyz))
            else:
                q_pos = torch.zeros_like(q_feat)

            q_b = q_feat.unsqueeze(0)        # [1, Q, D]
            q_pos_b = q_pos.unsqueeze(0)     # [1, Q, D]

            head = self.heads[g]

            for layer_idx, layer in enumerate(self.layers):
                if use_multi_scale:
                    s = self._scale_index_for_layer(layer_idx, num_scales)
                    q_b = layer(q_b, q_pos_b, scale_mems_b[s], scale_poss_b[s])
                else:
                    q_b = layer(q_b, q_pos_b, scene_mem_b, scene_pos_b)

                if layer_idx < num_layers - 1:
                    q_aux = q_b.squeeze(0)
                    me_aux = F.normalize(head.mask_embed(q_aux), dim=-1)
                    logit_s = head.logit_scale.exp()
                    aux_by_layer[layer_idx]["heads"][g] = {
                        "mask_logits": logit_s * (me_aux @ dense_mask_feat.T),
                        "score_logits": head.score_head(q_aux).squeeze(-1),
                    }

            # final layer prediction
            refined_q = q_b.squeeze(0)       # [Q, D]
            mask_embed = F.normalize(head.mask_embed(refined_q), dim=-1)
            logit_s = head.logit_scale.exp()

            out["heads"][g] = {
                "mask_logits": logit_s * (mask_embed @ dense_mask_feat.T),  # [Q, N]
                "score_logits": head.score_head(refined_q).squeeze(-1),     # [Q]
                "query_embed": refined_q,
            }

        if aux_by_layer:
            out["aux_outputs"] = aux_by_layer

        return out

    # ------------------------------------------------------------------ #

    def forward(
        self,
        point_feat: torch.Tensor,
        *,
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
        point_xyz          : [N, 3], optional — reserved for future anchor updates
        scene_tokens       : [V, C] — sparse voxel features from backbone (**required**)
        scene_xyz          : [V, 3] — voxel centroids (**required**)
        multi_scale_tokens : optional list of [V_i, C_i] per decoder scale (coarse → fine)
        multi_scale_xyz    : optional list of [V_i, 3]   per decoder scale (coarse → fine)

        Returns
        -------
        Nested dict with ``point_embed`` and per-head predictions.
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
                point_feat, scene_tokens, scene_xyz, **ms_kw,
            )

        if point_feat.ndim == 3:
            assert point_feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got {tuple(point_feat.shape)}"
            )
            out = self._forward_unbatched(
                point_feat[0], scene_tokens, scene_xyz, **ms_kw,
            )
            batched: dict = {
                "point_embed": out["point_embed"].unsqueeze(0),
                "heads": {},
            }
            for g, head_out in out["heads"].items():
                batched["heads"][g] = {
                    k: v.unsqueeze(0) for k, v in head_out.items()
                }
            if "aux_outputs" in out:
                batched["aux_outputs"] = [
                    {
                        "heads": {
                            g: {k: v.unsqueeze(0) for k, v in aux["heads"][g].items()}
                            for g in aux["heads"]
                        }
                    }
                    for aux in out["aux_outputs"]
                ]
            return batched

        raise ValueError(
            f"point_feat must be [N, C] or [1, N, C], got {tuple(point_feat.shape)}"
        )
