"""Thin wrapper around the standalone LitePT encoder-decoder.

Downstream code only sees:

    dense_feat = backbone(coord, feat)   # [N, 72]

Everything LitePT-specific (voxelization, Point dict, offset, grid_coord,
sparse tensors, inverse mapping) is hidden inside this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Grid-sample to one representative point per voxel.

        Returns the LitePT input dict and the ``inverse`` index
        for densifying back to the original N points.
        """
        grid_coord = torch.floor(coord / self.grid_size).to(torch.int64)
        grid_coord = grid_coord - grid_coord.min(dim=0).values

        unique_grid, inverse = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True,
        )

        N = coord.shape[0]
        V = unique_grid.shape[0]

        # Pick the first point that falls into each voxel.
        point_ids = torch.arange(N, device=coord.device, dtype=torch.long)
        first_idx = torch.full(
            (V,), fill_value=N, device=coord.device, dtype=torch.long,
        )
        first_idx.scatter_reduce_(
            0, inverse, point_ids, reduce="amin", include_self=True,
        )

        point_dict = {
            "coord": coord[first_idx],
            "grid_coord": unique_grid.int(),
            "feat": feat[first_idx],
            "offset": torch.tensor([V], device=coord.device, dtype=torch.long),
        }

        return point_dict, inverse

    # ------------------------------------------------------------------ #

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run LitePT and densify back to original points.

        Parameters
        ----------
        coord : (N, 3) or (1, N, 3) float
        feat  : (N, C) or (1, N, C) float

        Returns
        -------
        (N, 72) float — per-point features at the decoder's output resolution.
        """
        coord, feat = self._normalize_inputs(coord, feat)
        point_dict, inverse = self._voxelize(coord, feat)

        out = self.model(point_dict)

        dense_feat = out.feat[inverse]

        return dense_feat

# ── test ────────────────────────────────