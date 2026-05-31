"""Pure anchor and geometry diagnostics for ContinuousGeometryQueryDecoderV2."""

from __future__ import annotations

from typing import Any

import torch


def _float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.detach().float().reshape(-1)[0].item()
    return float(value)


def _percentiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[float]:
    x = values.detach().float().flatten()
    if x.numel() == 0:
        return [0.0 for _ in qs]
    return [_float(torch.quantile(x, q)) for q in qs]


def _as_long(value: Any | None, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(value, dtype=torch.long, device=device).flatten()


def target_centroids_from_masks(points: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """Compute centroids for boolean/float target masks shaped ``[T, N]``."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {tuple(points.shape)}")
    if target_masks.ndim != 2:
        raise ValueError(f"target_masks must be [T, N], got {tuple(target_masks.shape)}")
    if target_masks.shape[1] != points.shape[0]:
        raise ValueError("target_masks point dimension must match points")
    masks = target_masks.to(device=points.device, dtype=points.dtype)
    denom = masks.sum(dim=1).clamp_min(1.0)
    return (masks @ points.to(dtype=masks.dtype)) / denom[:, None]


def compute_anchor_movement_stats(
    points: torch.Tensor,
    query_anchors_initial: torch.Tensor,
    query_anchors_by_layer: torch.Tensor,
    *,
    query_radii_by_layer: torch.Tensor | None = None,
    delta_clamp_threshold: float | None = None,
) -> dict[str, float]:
    """Summarize anchor location, movement, radii, and in-scene ratios."""
    if query_anchors_initial.ndim != 2 or query_anchors_initial.shape[1] != 3:
        raise ValueError("query_anchors_initial must be [Q, 3]")
    if query_anchors_by_layer.ndim != 3 or query_anchors_by_layer.shape[-1] != 3:
        raise ValueError("query_anchors_by_layer must be [L, Q, 3]")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N, 3]")

    anchors0 = query_anchors_initial.detach().float()
    layers = query_anchors_by_layer.detach().float()
    final = layers[-1] if layers.shape[0] else anchors0
    xyz_min = points.detach().float().min(dim=0).values
    xyz_max = points.detach().float().max(dim=0).values
    in_scene = ((final >= xyz_min) & (final <= xyz_max)).all(dim=1)
    stats: dict[str, float] = {
        "in_scene_ratio": _float(in_scene.float().mean()) if final.numel() else 0.0,
        "initial_xyz_mean": _float(anchors0.mean()) if anchors0.numel() else 0.0,
        "final_xyz_mean": _float(final.mean()) if final.numel() else 0.0,
        "initial_xyz_std": _float(anchors0.std(unbiased=False)) if anchors0.numel() else 0.0,
        "final_xyz_std": _float(final.std(unbiased=False)) if final.numel() else 0.0,
    }

    prev = anchors0
    for layer_idx in range(int(layers.shape[0])):
        cur = layers[layer_idx]
        delta_norm = torch.linalg.norm(cur - prev, dim=1)
        p50, p90 = _percentiles(delta_norm, (0.50, 0.90))
        stats[f"delta_norm_p50_layer{layer_idx}"] = p50
        stats[f"delta_norm_p90_layer{layer_idx}"] = p90
        stats[f"delta_norm_max_layer{layer_idx}"] = _float(delta_norm.max()) if delta_norm.numel() else 0.0
        if delta_clamp_threshold is None or delta_clamp_threshold <= 0:
            stats[f"delta_clamped_fraction_layer{layer_idx}"] = 0.0
        else:
            stats[f"delta_clamped_fraction_layer{layer_idx}"] = _float(
                (delta_norm >= float(delta_clamp_threshold) * 0.999).float().mean()
            )
        if query_radii_by_layer is not None:
            radii = query_radii_by_layer
            if radii.ndim == 1:
                rad = radii.detach().float()
            else:
                rad = radii[min(layer_idx, radii.shape[0] - 1)].detach().float()
            r50, r90 = _percentiles(rad, (0.50, 0.90))
            stats[f"radius_p50_layer{layer_idx}"] = r50
            stats[f"radius_p90_layer{layer_idx}"] = r90
        prev = cur
    return stats


def compute_anchor_to_centroid_stats(
    points: torch.Tensor,
    target_masks: torch.Tensor,
    query_anchors_initial: torch.Tensor,
    query_anchors_by_layer: torch.Tensor,
    *,
    matched_pred_indices: Any | None = None,
    matched_target_indices: Any | None = None,
) -> dict[str, float]:
    """Compute matched anchor distance reduction toward target centroids."""
    device = query_anchors_initial.device
    pred_idx = _as_long(matched_pred_indices, device=device)
    target_idx = _as_long(matched_target_indices, device=device)
    if pred_idx.numel() == 0 or target_idx.numel() == 0:
        return {
            "dist_to_matched_centroid_init_p50": 0.0,
            "dist_to_matched_centroid_final_p50": 0.0,
            "dist_to_matched_centroid_init_p90": 0.0,
            "dist_to_matched_centroid_final_p90": 0.0,
            "movement_towards_centroid_fraction": 0.0,
            "mean_distance_reduction_to_centroid": 0.0,
        }
    m = min(int(pred_idx.numel()), int(target_idx.numel()))
    pred_idx = pred_idx[:m]
    target_idx = target_idx[:m]
    valid = (
        (pred_idx >= 0)
        & (pred_idx < query_anchors_initial.shape[0])
        & (target_idx >= 0)
        & (target_idx < target_masks.shape[0])
    )
    pred_idx = pred_idx[valid]
    target_idx = target_idx[valid]
    if pred_idx.numel() == 0:
        return {
            "dist_to_matched_centroid_init_p50": 0.0,
            "dist_to_matched_centroid_final_p50": 0.0,
            "dist_to_matched_centroid_init_p90": 0.0,
            "dist_to_matched_centroid_final_p90": 0.0,
            "movement_towards_centroid_fraction": 0.0,
            "mean_distance_reduction_to_centroid": 0.0,
        }

    centroids = target_centroids_from_masks(
        points.to(device=device, dtype=query_anchors_initial.dtype),
        target_masks.to(device=device),
    )
    final = query_anchors_by_layer[-1] if query_anchors_by_layer.numel() else query_anchors_initial
    init_dist = torch.linalg.norm(query_anchors_initial[pred_idx] - centroids[target_idx], dim=1)
    final_dist = torch.linalg.norm(final[pred_idx] - centroids[target_idx], dim=1)
    init_p50, init_p90 = _percentiles(init_dist, (0.50, 0.90))
    final_p50, final_p90 = _percentiles(final_dist, (0.50, 0.90))
    reduction = init_dist - final_dist
    return {
        "dist_to_matched_centroid_init_p50": init_p50,
        "dist_to_matched_centroid_final_p50": final_p50,
        "dist_to_matched_centroid_init_p90": init_p90,
        "dist_to_matched_centroid_final_p90": final_p90,
        "movement_towards_centroid_fraction": _float((final_dist < init_dist).float().mean()),
        "mean_distance_reduction_to_centroid": _float(reduction.mean()),
    }


def compute_scale_selector_stats(scale_weights: torch.Tensor | None) -> dict[str, float]:
    """Return per-level selector weights and entropy."""
    if scale_weights is None:
        return {}
    w = scale_weights.detach().float().flatten()
    if w.numel() == 0:
        return {}
    stats = {
        f"selector_weight_level_{i}": _float(v)
        for i, v in enumerate(w)
    }
    w_safe = w.clamp_min(1e-12)
    stats["selector_entropy"] = _float(-(w_safe * w_safe.log()).sum())
    return stats
