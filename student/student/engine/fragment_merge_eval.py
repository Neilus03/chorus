"""Optional multi-fragment evaluation toward full-scene ScanNet metrics.

Trains on crops, so a single forward pass only sees a subset of vertices.  This
module runs several sphere crops per scene, predicts instance masks on each
crop, then **fuses** a single per-vertex label assignment for all ``N`` mesh
vertices by choosing, for each vertex, the prediction from the fragment whose
representative point (the closest-to-center point in that crop) is nearest to
the vertex.  Vertices never appearing in any crop are labeled ``-1``.

This is a pragmatic merge — not identical to Pointcept's test-time sliding-window
code — but it yields real GT metrics at full resolution without requiring
Hungarian matching across fragment instance id spaces.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from student.data.multi_scene_dataset import MultiSceneDataset
from student.data.region_sampling import sphere_crop_indices_multi_center
from student.engine.evaluator import (
    _compute_clustering_metrics,
    _evaluate_ap_against_gt,
    extract_proposals,
    proposals_to_labels,
)

log = logging.getLogger(__name__)

_CHORUS_ROOT = Path(__file__).resolve().parents[3] / "chorus"


def _ensure_chorus_importable() -> None:
    s = str(_CHORUS_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _clear_backbone_cache(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _fuse_vertex_predictions(
    coords: np.ndarray,
    fragment_indices: list[np.ndarray],
    pred_labels_per_fragment: list[np.ndarray],
) -> np.ndarray:
    """One label per vertex from overlapping fragment predictions."""
    n = coords.shape[0]
    kfrag = len(fragment_indices)
    rep_coords = np.stack([coords[fi[0]] for fi in fragment_indices], axis=0)
    pos_maps: list[np.ndarray] = []
    for fi in fragment_indices:
        pm = np.full(n, -1, dtype=np.int32)
        pm[fi] = np.arange(len(fi), dtype=np.int32)
        pos_maps.append(pm)

    merged = np.full(n, -1, dtype=np.int32)
    for v in range(n):
        best_d = float("inf")
        best_lab = -1
        for k in range(kfrag):
            local = int(pos_maps[k][v])
            if local < 0:
                continue
            lab = int(pred_labels_per_fragment[k][local])
            d = float(np.sum((coords[v] - rep_coords[k]) ** 2))
            if d < best_d:
                best_d = d
                best_lab = lab
        merged[v] = best_lab
    return merged


def _dense_labels_to_proposals(pred_labels: np.ndarray) -> list[np.ndarray]:
    props: list[np.ndarray] = []
    for uid in np.unique(pred_labels):
        if int(uid) <= 0:
            continue
        props.append(pred_labels == int(uid))
    return props


def evaluate_scene_fragment_merge(
    model: nn.Module,
    dataset: MultiSceneDataset,
    idx: int,
    *,
    device: str,
    granularities: tuple[str, ...],
    score_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_points: int = 30,
    eval_benchmark: str = "scannet200",
    fragment_num: int = 4,
    fragment_point_max: int = 50_000,
    fragment_seed: int = 0,
) -> dict[str, Any]:
    """Full-mesh real-GT metrics by fusing several sphere-crop forwards.

    Pseudo-GT metrics are omitted (would require fusing pseudo-labels the same way).
    """
    model.eval()
    full = dataset.get_full_item(idx)
    scene_dir = Path(str(full["scene_dir"]))
    scene_id = str(full["scene_id"])
    coords = full["points"].numpy().astype(np.float32)
    n = coords.shape[0]
    rng = np.random.default_rng(
        (fragment_seed + idx * 10007 + hash(scene_id) % (2**31)) & 0xFFFFFFFF
    )
    frags = sphere_crop_indices_multi_center(
        coords,
        rng=rng,
        point_max=int(fragment_point_max),
        num_fragments=int(fragment_num),
    )

    result: dict[str, Any] = {}
    for g in granularities:
        pred_labels_per_frag: list[np.ndarray] = []
        for fi in frags:
            _clear_backbone_cache(model)
            tix = torch.from_numpy(fi.astype(np.int64)).long()
            pts = full["points"][tix].to(device)
            feat = full["features"][tix].to(device)
            with torch.no_grad():
                pred = model(pts, feat)
            head = pred["heads"][g]
            props, scores, _qi = extract_proposals(
                head["mask_logits"].detach().cpu(),
                head["score_logits"].detach().cpu(),
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_points=min_points,
            )
            plab = proposals_to_labels(props, len(fi))
            pred_labels_per_frag.append(plab)

        merged_labels = _fuse_vertex_predictions(coords, frags, pred_labels_per_frag)
        proposals = _dense_labels_to_proposals(merged_labels)

        try:
            _ensure_chorus_importable()
            from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids

            real_gt = load_scannet_gt_instance_ids(
                scene_dir, scene_id, eval_benchmark=eval_benchmark,
            )
            if real_gt.shape[0] != n:
                log.warning(
                    "[fragment merge %s] GT len %d != mesh %d",
                    g, real_gt.shape[0], n,
                )
                result[g] = {"real_gt": {"error": "gt length mismatch"}, "pseudo_gt": {}}
                continue

            real_ap = _evaluate_ap_against_gt(real_gt, proposals)
            real_clustering = _compute_clustering_metrics(real_gt, merged_labels)

            result[g] = {
                "pseudo_gt": {
                    "error": "skipped_in_fragment_merge_eval",
                },
                "real_gt": {
                    **real_ap,
                    **real_clustering,
                    "eval_benchmark": eval_benchmark,
                    "fragment_merge": True,
                    "fragment_num": fragment_num,
                    "fragment_point_max": fragment_point_max,
                },
            }
            log.info(
                "  [fragment merge %s / real GT] AP25=%.3f  AP50=%.3f  NMI=%.4f  ARI=%.4f",
                g,
                real_ap["AP25"],
                real_ap["AP50"],
                real_clustering["NMI"],
                real_clustering["ARI"],
            )
        except Exception as e:
            log.warning("  [fragment merge %s] failed: %s", g, e)
            result[g] = {"real_gt": {"error": str(e)}, "pseudo_gt": {}}

    return result
