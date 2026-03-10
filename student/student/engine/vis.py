"""PLY visualization utilities for predicted and GT instances.

Writes colored point clouds where each instance gets a distinct color.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from student.data.target_builder import InstanceTargets

log = logging.getLogger(__name__)

# 20 maximally-distinct colors (Tab20), repeated if more instances
_TAB20 = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
    [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
    [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
], dtype=np.uint8)

_UNMATCHED_COLOR = np.array([60, 60, 60], dtype=np.uint8)
_UNSUPERVISED_COLOR = np.array([30, 30, 30], dtype=np.uint8)


def _instance_palette(n: int) -> np.ndarray:
    """Return (n, 3) uint8 palette, cycling Tab20 if necessary."""
    repeats = (n // len(_TAB20)) + 1
    return np.tile(_TAB20, (repeats, 1))[:n]


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a simple colored ASCII PLY."""
    N = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(N):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )


def save_prediction_ply(
    points: np.ndarray,
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    matched_pred_idx: np.ndarray,
    *,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    path: Path | str = "student_pred.ply",
) -> None:
    """Save a PLY colored by predicted instance assignment.

    - Each high-scoring query's mask gets a distinct color.
    - Points not claimed by any query stay dark gray.
    """
    path = Path(path)
    N = points.shape[0]
    scores = score_logits.sigmoid()
    active = (scores >= score_threshold).numpy()
    masks_binary = (mask_logits.sigmoid() >= mask_threshold).numpy()  # [Q, N]

    colors = np.tile(_UNMATCHED_COLOR, (N, 1))
    palette = _instance_palette(int(active.sum()) + 1)

    color_idx = 0
    for q in range(mask_logits.shape[0]):
        if not active[q]:
            continue
        pts = masks_binary[q]
        colors[pts] = palette[color_idx]
        color_idx += 1

    _write_ply(path, points, colors)
    log.info("Prediction PLY saved: %s (%d active queries)", path, int(active.sum()))


def save_gt_ply(
    points: np.ndarray,
    targets: InstanceTargets,
    *,
    path: Path | str = "gt_instances.ply",
) -> None:
    """Save a PLY colored by ground-truth pseudo-instances.

    - Each GT instance gets a distinct color.
    - Non-supervised points are very dark.
    """
    path = Path(path)
    N = points.shape[0]
    gt_masks = targets.gt_masks.numpy()    # [M, N] bool
    sup = targets.supervision_mask.numpy()  # [N] bool

    colors = np.tile(_UNSUPERVISED_COLOR, (N, 1))
    palette = _instance_palette(targets.num_instances)

    for m in range(targets.num_instances):
        colors[gt_masks[m]] = palette[m]

    _write_ply(path, points, colors)
    log.info("GT PLY saved: %s (%d instances)", path, targets.num_instances)
