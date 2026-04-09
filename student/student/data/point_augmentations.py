"""Point cloud augmentations aligned with LitePT ScanNet-style training.

Ports the geometric + chromatic stack used in LitePT configs such as
``configs/scannet/insseg-litept-small-v1m2.py`` (upstream:
https://github.com/prs-eth/LitePT).

Applied in ``__getitem__`` so each epoch sees fresh random transforms.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    from scipy import interpolate as _interp
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover
    _interp = None
    _ndimage = None

_HAS_SCIPY = _interp is not None and _ndimage is not None


def _rotation_matrix(angle: float, axis: str) -> np.ndarray:
    rot_cos, rot_sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
    if axis == "y":
        return np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
    if axis == "z":
        return np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    raise NotImplementedError(axis)


def _random_rotate(
    coord: np.ndarray,
    angle: tuple[float, float],
    axis: str,
    center: np.ndarray | list[float] | None,
    p: float,
) -> None:
    if random.random() > p:
        return
    ang = np.random.uniform(angle[0], angle[1]) * np.pi
    rot_t = _rotation_matrix(ang, axis)
    if center is None:
        x_min, y_min, z_min = coord.min(axis=0)
        x_max, y_max, z_max = coord.max(axis=0)
        c = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
            dtype=np.float32,
        )
    else:
        c = np.asarray(center, dtype=np.float32)
    coord -= c
    coord[:] = coord @ rot_t.T
    coord += c


def _random_scale(coord: np.ndarray, scale: tuple[float, float]) -> None:
    s = float(np.random.uniform(scale[0], scale[1]))
    coord *= s


def _random_flip(coord: np.ndarray, p: float) -> None:
    if np.random.rand() < p:
        coord[:, 0] *= -1.0
    if np.random.rand() < p:
        coord[:, 1] *= -1.0


def _random_jitter(coord: np.ndarray, sigma: float, clip: float) -> None:
    jitter = np.clip(
        sigma * np.random.randn(*coord.shape).astype(np.float32),
        -clip,
        clip,
    )
    coord += jitter


def _elastic_distortion(
    coord: np.ndarray,
    distortion_params: list[list[float]],
) -> None:
    if not _HAS_SCIPY or not distortion_params:
        return
    if random.random() > 0.95:
        return

    blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3.0
    blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3.0
    blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3.0
    coords_min = coord.min(0)

    for granularity, magnitude in distortion_params:
        noise_dim = ((coord - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        for _ in range(2):
            noise = _ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = _ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = _ndimage.convolve(noise, blurz, mode="constant", cval=0)

        ax = [
            np.linspace(d_min, d_max, d, dtype=np.float32)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = _interp.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coord[:] = coord + interp(coord) * magnitude


def _chromatic_jitter_float01(color: np.ndarray, p: float, std: float) -> None:
    """LitePT uses 0–255 colors with ``std * 255`` noise; here ``color`` is float 0–1."""
    if color is None or color.size == 0:
        return
    if np.random.rand() >= p:
        return
    noise = np.random.randn(color.shape[0], 3).astype(np.float32) * std
    color[:] = np.clip(color + noise, 0.0, 1.0)


def augment_points_litept_scannet(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Copy inputs, apply LitePT ScanNet-style train augmentations, return augmented arrays.

    Order matches ``configs/scannet/insseg-litept-small-v1m2.py``: z/x/y rotate,
    scale, flip, jitter, elastic, chromatic jitter.
    """
    coord = np.asarray(points, dtype=np.float32).copy()
    color: np.ndarray | None
    if colors is not None and use_colors:
        color = np.asarray(colors, dtype=np.float32).copy()
        if color.max() > 1.0:
            color = color / 255.0
    else:
        color = None

    _random_rotate(coord, (-1.0, 1.0), "z", [0.0, 0.0, 0.0], p=0.5)
    _random_rotate(coord, (-1.0 / 64.0, 1.0 / 64.0), "x", None, p=0.5)
    _random_rotate(coord, (-1.0 / 64.0, 1.0 / 64.0), "y", None, p=0.5)
    _random_scale(coord, (0.9, 1.1))
    _random_flip(coord, p=0.5)
    _random_jitter(coord, sigma=0.005, clip=0.02)

    if _HAS_SCIPY:
        _elastic_distortion(coord, [[0.2, 0.4], [0.8, 1.6]])
    if color is not None:
        _chromatic_jitter_float01(color, p=0.95, std=0.05)

    return coord, color


LITEP_SCANNET_DEFAULTS: dict[str, Any] = {
    "description": "LitePT insseg-litept-small-v1m2 train transforms (geom + chromatic)",
}
