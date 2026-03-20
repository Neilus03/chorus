"""Thin wrapper around the standalone LitePT encoder-decoder.

Downstream code receives a structured :class:`LitePTBackboneOutput`::

    bb = backbone(coord, feat)
    bb.point_feat    # [N, 72] dense per-point features
    bb.scene_tokens  # [V, 72] sparse voxel features
    bb.point_xyz     # [N, 3]  original coordinates
    bb.scene_xyz     # [V, 3]  voxel centroids
    bb.inverse_map   # [N]     point → voxel index

Everything LitePT-specific (voxelization, Point dict, offset, grid_coord,
sparse tensors) is hidden inside this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class LitePTBackboneOutput:
    """Structured output from the LitePT backbone.

    Carries both the dense per-point features needed for final mask
    dot-products and the sparse voxel tokens needed for efficient
    Transformer cross-attention in the decoder.
    """

    point_feat: torch.Tensor    # [N, C] dense per-point features
    point_xyz: torch.Tensor     # [N, 3] original point coordinates
    scene_tokens: torch.Tensor  # [V, C] sparse voxel features
    scene_xyz: torch.Tensor     # [V, 3] voxel centroids
    inverse_map: torch.Tensor   # [N]    maps points → voxel token index


class LitePTBackbone(nn.Module):
    """Wraps :class:`litept.model.LitePT` for single-scene use.

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
    ) -> None:
        super().__init__()
        self.litept_root = Path(litept_root).resolve()
        self.in_channels = int(in_channels)
        self.grid_size = float(grid_size)

        if not self.litept_root.exists():
            raise FileNotFoundError(
                f"LitePT root does not exist: {self.litept_root}"
            )

        litept_root_str = str(self.litept_root)
        if litept_root_str not in sys.path:
            sys.path.insert(0, litept_root_str)

        from litept.model import LitePT

        self.model = LitePT(in_channels=self.in_channels)

        # Default LitePT-S decoder outputs dec_channels[0] = 72.
        self.out_channels: int = 72

        self._cached_voxelization: (
            tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None
        ) = None

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

    def _voxelize(
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

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
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
        """
        coord, feat = self._normalize_inputs(coord, feat)

        if self._cached_voxelization is not None and self.training:
            point_dict, inverse, scene_xyz = self._cached_voxelization
        else:
            point_dict, inverse, scene_xyz = self._voxelize(coord, feat)
            if self.training:
                self._cached_voxelization = (point_dict, inverse, scene_xyz)

        out = self.model(point_dict)

        scene_tokens = out.feat          # [V, C]
        point_feat = scene_tokens[inverse]  # [N, C]

        return LitePTBackboneOutput(
            point_feat=point_feat,
            point_xyz=coord,
            scene_tokens=scene_tokens,
            scene_xyz=scene_xyz,
            inverse_map=inverse,
        )