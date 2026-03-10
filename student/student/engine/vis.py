"""Mesh PLY visualization for predicted and GT instances.

Reads the original ScanNet mesh PLY (with faces), recolors vertices
by instance assignment, and writes a new binary PLY with faces preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from student.data.target_builder import InstanceTargets

log = logging.getLogger(__name__)

_UNMATCHED_COLOR = (60, 60, 60)
_UNSUPERVISED_COLOR = (30, 30, 30)


def _instance_palette(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(n, 3), dtype=np.uint8)


def _recolor_mesh(
    source_ply_path: Path,
    vertex_colors: np.ndarray,
    out_path: Path,
) -> None:
    plydata = PlyData.read(str(source_ply_path))
    if "vertex" not in plydata:
        raise RuntimeError(f"PLY has no vertex element: {source_ply_path}")

    vertex_data = plydata["vertex"].data
    n_vertices = len(vertex_data)
    if vertex_colors.shape[0] != n_vertices:
        raise ValueError(
            f"Color length {vertex_colors.shape[0]} != "
            f"vertex count {n_vertices} in {source_ply_path}"
        )

    out_verts = np.empty(
        n_vertices,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    out_verts["x"] = np.asarray(vertex_data["x"], dtype=np.float32)
    out_verts["y"] = np.asarray(vertex_data["y"], dtype=np.float32)
    out_verts["z"] = np.asarray(vertex_data["z"], dtype=np.float32)
    out_verts["red"] = vertex_colors[:, 0]
    out_verts["green"] = vertex_colors[:, 1]
    out_verts["blue"] = vertex_colors[:, 2]

    elements = [PlyElement.describe(out_verts, "vertex")]
    face_el = next((e for e in plydata.elements if e.name == "face"), None)
    if face_el is not None and len(face_el.data) > 0:
        elements.append(PlyElement.describe(face_el.data, "face"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=False).write(str(out_path))


def _resolve_source_mesh(scene_dir: str | Path, scene_meta: dict) -> Path:
    scene_dir = Path(scene_dir)
    name = scene_meta.get("geometry_path_name") or scene_meta.get("geometry_source")
    if name:
        p = scene_dir / name
        if p.exists():
            return p
    for candidate in scene_dir.glob("*_vh_clean_2.ply"):
        return candidate
    raise FileNotFoundError(
        f"No source mesh PLY in {scene_dir} "
        f"(tried geometry_path_name={name!r} and *_vh_clean_2.ply)"
    )


def save_prediction_ply(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    matched_pred_idx: np.ndarray,
    source_mesh: Path,
    *,
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    path: Path | str = "student_pred.ply",
) -> None:
    path = Path(path)
    N = mask_logits.shape[1]
    scores = score_logits.sigmoid().numpy()
    active = scores >= score_threshold
    masks_binary = (mask_logits.sigmoid() >= mask_threshold).numpy()

    colors = np.full((N, 3), _UNMATCHED_COLOR, dtype=np.uint8)
    n_active = int(active.sum())
    palette = _instance_palette(max(n_active, 1))

    color_idx = 0
    for q in range(mask_logits.shape[0]):
        if not active[q]:
            continue
        colors[masks_binary[q]] = palette[color_idx]
        color_idx += 1

    _recolor_mesh(source_mesh, colors, path)
    log.info("Prediction mesh: %s (%d active queries)", path, n_active)


def save_gt_ply(
    targets: InstanceTargets,
    source_mesh: Path,
    *,
    path: Path | str = "gt_instances.ply",
) -> None:
    path = Path(path)
    N = int(targets.supervision_mask.shape[0])
    gt_masks = targets.gt_masks.numpy()

    colors = np.full((N, 3), _UNSUPERVISED_COLOR, dtype=np.uint8)
    palette = _instance_palette(max(targets.num_instances, 1))

    for m in range(targets.num_instances):
        colors[gt_masks[m]] = palette[m]

    _recolor_mesh(source_mesh, colors, path)
    log.info("GT mesh: %s (%d instances)", path, targets.num_instances)
