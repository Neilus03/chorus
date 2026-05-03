"""Thin wrapper around the standalone LitePT encoder-decoder.

Downstream code receives a structured :class:`LitePTBackboneOutput`::

    bb = backbone(coord, feat)
    bb.point_feat    # [N, 72] dense per-point features
    bb.scene_tokens  # [V, 72] sparse voxel features
    bb.point_xyz     # [N, 3]  original coordinates
    bb.scene_xyz     # [V, 3]  voxel centroids
    bb.inverse_map   # [N]     point → voxel index

By default the backbone is **LitePT-S*** (ScanNet instance-seg architecture from
``configs/scannet/insseg-litept-small-v1m2.py``): deeper decoder
``dec_depths=(2, 2, 2, 2)`` vs LitePT-S where ``dec_depths=(0, 0, 0, 0)``. Use
``litept_variant="litept_s"`` only for checkpoints trained with the shallow decoder.

Everything LitePT-specific (voxelization, Point dict, offset, grid_coord,
sparse tensors) is hidden inside this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# LitePT-S* — mirrors LitePT/configs/scannet/insseg-litept-small-v1m2.py ``backbone`` dict
# (excluding ``type`` and ``in_channels``, which come from the student config).
LITEPT_S_STAR_KWARGS: dict[str, Any] = {
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (36, 72, 144, 252, 504),
    "enc_num_head": (2, 4, 8, 14, 28),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "enc_conv": (True, True, True, False, False),
    "enc_attn": (False, False, False, True, True),
    "enc_rope_freq": (100.0, 100.0, 100.0, 100.0, 100.0),
    "dec_depths": (2, 2, 2, 2),
    "dec_channels": (72, 72, 144, 252),
    "dec_num_head": (4, 4, 8, 14),
    "dec_patch_size": (1024, 1024, 1024, 1024),
    "dec_conv": (True, True, True, False),
    "dec_attn": (False, False, False, True),
    "dec_rope_freq": (100.0, 100.0, 100.0, 100.0),
    "mlp_ratio": 4,
    "qkv_bias": True,
    "qk_scale": None,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "drop_path": 0.3,
    "pre_norm": True,
    "shuffle_orders": True,
    "enc_mode": False,
}


def _litept_constructor_kwargs(variant: str) -> dict[str, Any]:
    if variant == "litept_s_star":
        return dict(LITEPT_S_STAR_KWARGS)
    if variant == "litept_s":
        # ``litept.model.LitePT`` defaults match semantic-seg LitePT-S (no decoder blocks).
        return {}
    raise ValueError(
        f"Unknown litept_variant {variant!r}; expected 'litept_s_star' or 'litept_s'."
    )


@dataclass
class LitePTBackboneOutput:
    """Structured output from the LitePT backbone.

    Carries both the dense per-point features needed for final mask
    dot-products and the sparse voxel tokens needed for efficient
    Transformer cross-attention in the decoder.

    When ``multi_scale`` is enabled, ``multi_scale_tokens`` and
    ``multi_scale_xyz`` contain per-decoder-substage features in
    **coarse → fine** order (e.g. 4 entries for LitePT-S*).
    """

    point_feat: torch.Tensor    # [N, C] dense per-point features
    point_xyz: torch.Tensor     # [N, 3] original point coordinates
    scene_tokens: torch.Tensor  # [V, C] sparse voxel features (finest scale)
    scene_xyz: torch.Tensor     # [V, 3] voxel centroids (finest scale)
    inverse_map: torch.Tensor   # [N]    maps points → voxel token index
    point_offsets: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    scene_token_offsets: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))

    multi_scale_tokens: list[torch.Tensor] = field(default_factory=list)
    multi_scale_xyz: list[torch.Tensor] = field(default_factory=list)
    multi_scale_offsets: list[torch.Tensor] = field(default_factory=list)


class LitePTBackbone(nn.Module):
    """Wraps :class:`litept.model.LitePT` for single- or multi-scene use.

    Parameters
    ----------
    litept_root:
        Absolute path to the cloned LitePT repository.  The wrapper adds it
        to ``sys.path`` so that ``from litept.model import LitePT`` works.
    in_channels:
        Number of per-point input feature channels (must match the actual
        feature tensor passed to :meth:`forward`).
    grid_size:
        Voxel edge length in meters for grid sampling.
    """

    def __init__(
        self,
        litept_root: str,
        in_channels: int,
        grid_size: float = 0.02,
        litept_variant: str = "litept_s_star",
        litept_kwargs: dict[str, Any] | None = None,
        multi_scale: bool = False,
        multi_scale_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.litept_root = Path(litept_root).resolve()
        self.in_channels = int(in_channels)
        self.grid_size = float(grid_size)
        if litept_variant not in ("litept_s_star", "litept_s"):
            raise ValueError(
                f"litept_variant must be 'litept_s_star' or 'litept_s', got {litept_variant!r}"
            )
        self.litept_variant = litept_variant

        if not self.litept_root.exists():
            raise FileNotFoundError(
                f"LitePT root does not exist: {self.litept_root}"
            )

        litept_root_str = str(self.litept_root)
        if litept_root_str not in sys.path:
            sys.path.insert(0, litept_root_str)

        from litept.model import LitePT

        ctor: dict[str, Any] = _litept_constructor_kwargs(litept_variant)
        if litept_kwargs:
            ctor.update(litept_kwargs)
        self.model = LitePT(in_channels=self.in_channels, **ctor)

        # LitePT-S and LitePT-S* both use dec_channels[0] = 72 at the finest output.
        self.out_channels: int = 72

        self._cached_voxelization: (
            tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

        #  multi-scale decoder feature capture 
        self._multi_scale = bool(multi_scale)
        self._captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._multi_scale_channels: list[int] = []
        self._multi_scale_indices: list[int] | None = None

        if self._multi_scale and hasattr(self.model, "dec"):
            dec_channels_cfg = list(ctor.get("dec_channels", (72, 72, 144, 252)))
            # dec_channels_cfg is indexed s=0 (finest) .. s=3 (coarsest).
            # self.model.dec runs in forward order dec3→dec0 (coarsest→finest),
            # so reversed gives coarse→fine channel widths. Hook indices i are in
            # that same coarse→fine order (i ascending).
            all_channels_coarse_to_fine = list(reversed(dec_channels_cfg))
            num_dec_stages = len(self.model.dec)

            if multi_scale_indices is None:
                selected = list(range(num_dec_stages))
            else:
                selected = sorted({int(i) for i in multi_scale_indices})
                for i in selected:
                    if i < 0 or i >= num_dec_stages:
                        raise ValueError(
                            f"multi_scale_indices contains out-of-range index {i}; "
                            f"expected [0, {num_dec_stages - 1}]"
                        )

            self._multi_scale_indices = selected
            self._multi_scale_channels = [
                all_channels_coarse_to_fine[i] for i in selected
            ]

            for i in selected:
                self.model.dec[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, scale_idx: int):
        """Return a forward hook that captures (feat, coord) from a decoder substage."""
        def hook(module, input, output):
            offset = getattr(
                output,
                "offset",
                torch.tensor([output.feat.shape[0]], device=output.feat.device, dtype=torch.long),
            )
            self._captured[scale_idx] = (output.feat, output.coord, offset)
        return hook

    @property
    def multi_scale_channels(self) -> list[int] | None:
        """Channel widths of each captured decoder scale (coarse → fine), or None."""
        if self._multi_scale and self._multi_scale_channels:
            return list(self._multi_scale_channels)
        return None

    # ------------------------------------------------------------------ #

    def _normalize_inputs(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Squeeze batch dim if present and validate shapes."""
        if coord.ndim == 3:
            assert coord.shape[0] == 1, (
                f"Only batch_size=1 supported, got coord shape {tuple(coord.shape)}"
            )
            coord = coord[0]

        if feat.ndim == 3:
            assert feat.shape[0] == 1, (
                f"Only batch_size=1 supported, got feat shape {tuple(feat.shape)}"
            )
            feat = feat[0]

        assert coord.ndim == 2 and coord.shape[1] == 3, (
            f"coord must be [N, 3], got {tuple(coord.shape)}"
        )
        assert feat.ndim == 2 and feat.shape[1] == self.in_channels, (
            f"feat must be [N, {self.in_channels}], got {tuple(feat.shape)}"
        )
        assert coord.shape[0] == feat.shape[0], (
            f"coord has {coord.shape[0]} points but feat has {feat.shape[0]}"
        )

        return coord.contiguous().float(), feat.contiguous().float()

    # ------------------------------------------------------------------ #

    def _normalize_offsets(
        self,
        point_offsets: torch.Tensor | None,
        *,
        num_points: int,
        device: torch.device,
    ) -> torch.Tensor:
        if point_offsets is None:
            return torch.tensor([num_points], device=device, dtype=torch.long)

        offsets = point_offsets.to(device=device, dtype=torch.long).flatten()
        if offsets.numel() == 0:
            raise ValueError("point_offsets must contain at least one scene")
        if int(offsets[-1].item()) != num_points:
            raise ValueError(
                f"point_offsets[-1]={int(offsets[-1].item())} but coord has {num_points} points"
            )
        prev = 0
        for end in offsets.tolist():
            if end <= prev:
                raise ValueError(
                    f"point_offsets must be strictly increasing, got {offsets.tolist()}"
                )
            prev = end
        return offsets

    def _voxelize_single(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Grid-sample to one representative per voxel via mean-pooling.

        Returns
        -------
        point_dict : dict
            LitePT input dict with mean-pooled coords and features.
        inverse : Tensor [N]
            Maps each original point to its voxel index.
        scene_xyz : Tensor [V, 3]
            Voxel centroids (mean of all point coordinates per voxel).
        """
        grid_coord = torch.floor(coord / self.grid_size).to(torch.int64)
        grid_coord = grid_coord - grid_coord.min(dim=0).values

        unique_grid, inverse = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True,
        )

        V = unique_grid.shape[0]

        voxel_counts = torch.bincount(inverse, minlength=V).float().clamp(min=1.0)

        scene_xyz = torch.zeros(V, 3, device=coord.device, dtype=coord.dtype)
        scene_xyz.index_add_(0, inverse, coord)
        scene_xyz = scene_xyz / voxel_counts[:, None]

        scene_feat = torch.zeros(
            V, feat.shape[1], device=feat.device, dtype=feat.dtype,
        )
        scene_feat.index_add_(0, inverse, feat)
        scene_feat = scene_feat / voxel_counts[:, None]

        point_dict = {
            "coord": scene_xyz,
            "grid_coord": unique_grid.int(),
            "feat": scene_feat,
            "offset": torch.tensor([V], device=coord.device, dtype=torch.long),
        }

        return point_dict, inverse, scene_xyz

    # ------------------------------------------------------------------ #

    def _voxelize_batched(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_offsets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize a concatenated multi-scene batch while preserving boundaries."""
        inverse = torch.empty(coord.shape[0], device=coord.device, dtype=torch.long)
        grid_parts: list[torch.Tensor] = []
        xyz_parts: list[torch.Tensor] = []
        feat_parts: list[torch.Tensor] = []
        voxel_offsets: list[int] = []

        start = 0
        voxel_base = 0
        for end in point_offsets.tolist():
            scene_coord = coord[start:end]
            scene_feat = feat[start:end]
            point_dict, scene_inverse, scene_xyz = self._voxelize_single(scene_coord, scene_feat)

            grid_parts.append(point_dict["grid_coord"])
            xyz_parts.append(scene_xyz)
            feat_parts.append(point_dict["feat"])
            inverse[start:end] = scene_inverse + voxel_base

            voxel_base += int(scene_xyz.shape[0])
            voxel_offsets.append(voxel_base)
            start = end

        scene_xyz = torch.cat(xyz_parts, dim=0)
        point_dict = {
            "coord": scene_xyz,
            "grid_coord": torch.cat(grid_parts, dim=0),
            "feat": torch.cat(feat_parts, dim=0),
            "offset": torch.tensor(voxel_offsets, device=coord.device, dtype=torch.long),
        }
        return point_dict, inverse, scene_xyz, point_dict["offset"]

    # ------------------------------------------------------------------ #

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_offsets: torch.Tensor | None = None,
    ) -> LitePTBackboneOutput:
        """Run LitePT and return structured backbone output.

        Parameters
        ----------
        coord : (N, 3) or (1, N, 3) float
        feat  : (N, C) or (1, N, C) float

        Returns
        -------
        :class:`LitePTBackboneOutput` with dense per-point features,
        sparse voxel tokens, coordinates, and inverse mapping.
        When ``multi_scale`` is enabled, also includes per-scale
        decoder features in coarse → fine order.
        """
        coord, feat = self._normalize_inputs(coord, feat)
        point_offsets_t = self._normalize_offsets(
            point_offsets,
            num_points=coord.shape[0],
            device=coord.device,
        )
        batched = point_offsets_t.numel() > 1

        if self._cached_voxelization is not None and self.training and not batched:
            point_dict, inverse, scene_xyz, scene_token_offsets = self._cached_voxelization
        else:
            if batched:
                point_dict, inverse, scene_xyz, scene_token_offsets = self._voxelize_batched(
                    coord,
                    feat,
                    point_offsets_t,
                )
            else:
                point_dict, inverse, scene_xyz = self._voxelize_single(coord, feat)
                scene_token_offsets = point_dict["offset"]
                if self.training:
                    self._cached_voxelization = (
                        point_dict,
                        inverse,
                        scene_xyz,
                        scene_token_offsets,
                    )

        self._captured.clear()

        out = self.model(point_dict)

        scene_tokens = out.feat          # [V, C]
        point_feat = scene_tokens[inverse]  # [N, C]
        scene_token_offsets = getattr(out, "offset", scene_token_offsets)

        ms_tokens: list[torch.Tensor] = []
        ms_xyz: list[torch.Tensor] = []
        ms_offsets: list[torch.Tensor] = []
        if self._multi_scale and self._captured:
            for i in sorted(self._captured.keys()):
                cap_feat, cap_coord, cap_offset = self._captured[i]
                ms_tokens.append(cap_feat)
                ms_xyz.append(cap_coord)
                ms_offsets.append(cap_offset)

        return LitePTBackboneOutput(
            point_feat=point_feat,
            point_xyz=coord,
            scene_tokens=scene_tokens,
            scene_xyz=scene_xyz,
            inverse_map=inverse,
            point_offsets=point_offsets_t,
            scene_token_offsets=scene_token_offsets,
            multi_scale_tokens=ms_tokens,
            multi_scale_xyz=ms_xyz,
            multi_scale_offsets=ms_offsets,
        )