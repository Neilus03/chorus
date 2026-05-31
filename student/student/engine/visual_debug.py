"""Local visual artifact writers for training debug snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from student.engine.feature_diagnostics import pca_rgb

log = logging.getLogger(__name__)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _to_uint8(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(rgb)
    if arr.dtype != np.uint8:
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def instance_colors(labels: torch.Tensor | np.ndarray) -> np.ndarray:
    labels_np = _to_numpy(labels).astype(np.int64)
    rgb = np.full((labels_np.shape[0], 3), 185, dtype=np.uint8)
    for inst_id in sorted(int(v) for v in np.unique(labels_np) if int(v) > 0):
        rng = np.random.default_rng(inst_id * 7919)
        rgb[labels_np == inst_id] = rng.integers(30, 245, size=3, dtype=np.uint8)
    return rgb


def save_point_cloud_ply(
    xyz: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    path: Path,
) -> None:
    """Write a compact ASCII PLY point cloud with RGB colors."""
    xyz_np = _to_numpy(xyz).astype(np.float32)
    rgb_np = _to_uint8(colors)
    if xyz_np.shape[0] != rgb_np.shape[0]:
        raise ValueError("xyz/color lengths differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz_np.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz_np, rgb_np):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def save_topdown_png(
    path: Path,
    points: torch.Tensor | np.ndarray,
    panels: list[tuple[str, torch.Tensor | np.ndarray]],
    *,
    max_points: int = 150_000,
    point_size: float = 0.2,
) -> None:
    """Save one-row top-down XY panels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional env
        log.warning("matplotlib unavailable; skipping %s (%s)", path, exc)
        return

    xyz = _to_numpy(points).astype(np.float32)
    n = int(xyz.shape[0])
    idx = np.arange(n)
    if n > int(max_points):
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, rgb) in zip(axes, panels):
        rgb_np = _to_uint8(rgb)
        ax.scatter(
            xyz[idx, 0],
            xyz[idx, 1],
            c=rgb_np[idx] / 255.0,
            s=point_size,
            marker=".",
            linewidths=0,
        )
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def render_query_trajectories_png(
    path: Path,
    points: torch.Tensor | np.ndarray,
    query_anchors_initial: torch.Tensor | np.ndarray,
    query_anchors_by_layer: torch.Tensor | np.ndarray,
    *,
    topk_indices: torch.Tensor | np.ndarray | None = None,
    matched_centroids: torch.Tensor | np.ndarray | None = None,
    query_source_type: torch.Tensor | np.ndarray | None = None,
    max_background_points: int = 80_000,
) -> None:
    """Render a top-down query anchor trajectory plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log.warning("matplotlib unavailable; skipping %s (%s)", path, exc)
        return

    xyz = _to_numpy(points).astype(np.float32)
    anchors0 = _to_numpy(query_anchors_initial).astype(np.float32)
    layers = _to_numpy(query_anchors_by_layer).astype(np.float32)
    if layers.ndim == 2:
        layers = layers[None, ...]
    idx = np.arange(xyz.shape[0])
    if idx.size > max_background_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_background_points, replace=False))
    q_idx = np.arange(anchors0.shape[0])
    if topk_indices is not None:
        q_idx = _to_numpy(topk_indices).astype(np.int64)
        q_idx = q_idx[(q_idx >= 0) & (q_idx < anchors0.shape[0])]
    source = None if query_source_type is None else _to_numpy(query_source_type).astype(np.int64)
    palette = np.array([
        [0.10, 0.35, 0.95],
        [0.05, 0.65, 0.30],
        [0.85, 0.35, 0.05],
        [0.40, 0.40, 0.40],
    ])

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0), dpi=180)
    ax.scatter(xyz[idx, 0], xyz[idx, 1], c="0.82", s=0.12, marker=".", linewidths=0)
    for q in q_idx:
        color = palette[int(source[q]) % len(palette)] if source is not None else palette[0]
        path_xy = np.concatenate([anchors0[q:q + 1], layers[:, q, :]], axis=0)
        ax.plot(path_xy[:, 0], path_xy[:, 1], color=color, linewidth=0.8, alpha=0.75)
        ax.scatter(path_xy[0, 0], path_xy[0, 1], facecolors="none", edgecolors=[color], s=22, linewidths=0.8)
        ax.scatter(path_xy[-1, 0], path_xy[-1, 1], c=[color], s=16, linewidths=0)
    if matched_centroids is not None:
        c = _to_numpy(matched_centroids).astype(np.float32)
        if c.size:
            ax.scatter(c[:, 0], c[:, 1], c="gold", edgecolors="black", marker="*", s=40, linewidths=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def topk_prediction_labels(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    *,
    topk: int = 50,
    mask_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-point top-k prediction labels and selected query indices."""
    q = int(score_logits.shape[0])
    k = min(max(int(topk), 1), q)
    idx = score_logits.detach().float().sigmoid().topk(k).indices
    labels = torch.full((mask_logits.shape[1],), -1, dtype=torch.long, device=mask_logits.device)
    masks = mask_logits.detach()[idx].sigmoid() >= float(mask_threshold)
    # Paint low-to-high score so the highest score wins ties.
    for rank, row in enumerate(reversed(range(k)), start=1):
        labels[masks[row]] = k - row
    return labels.cpu(), idx.cpu()


def make_top_queries_table(
    pred: dict[str, Any],
    *,
    topk: int = 50,
    mask_threshold: float = 0.5,
    matched_pred_indices: Any | None = None,
    matched_target_indices: Any | None = None,
    matched_ious: Any | None = None,
    points: torch.Tensor | None = None,
    target_masks: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    score_logits = pred["score_logits"].detach().float().cpu()
    mask_logits = pred["mask_logits"].detach().float().cpu()
    probs = score_logits.sigmoid()
    k = min(max(int(topk), 1), int(score_logits.shape[0]))
    top_idx = probs.topk(k).indices
    areas = (mask_logits.sigmoid() >= float(mask_threshold)).sum(dim=1)
    match_pred = torch.as_tensor([] if matched_pred_indices is None else matched_pred_indices, dtype=torch.long)
    match_tgt = torch.as_tensor([] if matched_target_indices is None else matched_target_indices, dtype=torch.long)
    match_iou = torch.as_tensor([] if matched_ious is None else matched_ious, dtype=torch.float32)
    match_lookup: dict[int, tuple[int | None, float | None]] = {}
    for i, q in enumerate(match_pred.tolist()):
        tgt = int(match_tgt[i].item()) if i < match_tgt.numel() else None
        iou = float(match_iou[i].item()) if i < match_iou.numel() else None
        match_lookup[int(q)] = (tgt, iou)

    debug = pred.get("debug", {}) if isinstance(pred.get("debug"), dict) else {}
    q0 = debug.get("query_anchors_initial")
    qlayers = debug.get("query_anchors_by_layer")
    qfinal = qlayers[-1] if isinstance(qlayers, torch.Tensor) and qlayers.numel() else pred.get("query_xyz")
    rows: list[dict[str, Any]] = []
    for q_t in top_idx:
        q = int(q_t.item())
        tgt, iou = match_lookup.get(q, (None, None))
        row = {
            "query_id": q,
            "score_logit": float(score_logits[q].item()),
            "score_probability": float(probs[q].item()),
            "mask_area": int(areas[q].item()),
            "matched_target_id": tgt,
            "matched_iou": iou,
            "is_high_score_unmatched": bool(probs[q].item() >= 0.5 and q not in match_lookup),
        }
        if isinstance(q0, torch.Tensor):
            row["anchor_initial_xyz"] = [float(v) for v in q0.detach().cpu()[q].tolist()]
        if isinstance(qfinal, torch.Tensor):
            row["anchor_final_xyz"] = [float(v) for v in qfinal.detach().cpu()[q].tolist()]
        rows.append(row)
    return rows


def write_scene_snapshot(
    out_dir: Path,
    *,
    sample: dict[str, Any],
    pred: dict[str, Any],
    granularity: str,
    target_labels: torch.Tensor,
    matched_pred_indices: Any | None = None,
    matched_target_indices: Any | None = None,
    matched_ious: Any | None = None,
    topk_queries: int = 50,
    max_render_points: int = 150_000,
    mask_threshold: float = 0.5,
    save_png: bool = True,
    save_ply: bool = True,
    save_npz: bool = True,
) -> dict[str, str]:
    """Write a compact qualitative snapshot for one scene/granularity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    points = sample["points"].detach().cpu()
    rgb = np.full((points.shape[0], 3), 175, dtype=np.uint8)
    colors_path = Path(str(sample.get("scene_dir", ""))) / "colors.npy"
    if colors_path.exists():
        try:
            colors = np.load(colors_path)
            if colors.shape[0] == points.shape[0]:
                if colors.max() <= 1.0:
                    colors = colors * 255.0
                rgb = np.clip(colors, 0, 255).astype(np.uint8)
        except Exception:
            pass

    artifacts: dict[str, str] = {}
    gt_rgb = instance_colors(target_labels)
    pred_labels, top_idx = topk_prediction_labels(
        pred["mask_logits"],
        pred["score_logits"],
        topk=topk_queries,
        mask_threshold=mask_threshold,
    )
    pred_rgb = instance_colors(pred_labels)
    if save_ply:
        save_point_cloud_ply(points, rgb, out_dir / "input_rgb_points.ply")
        save_point_cloud_ply(points, gt_rgb, out_dir / "pseudo_or_gt_instances.ply")
        save_point_cloud_ply(points, pred_rgb, out_dir / "pred_topk_instances.ply")
        artifacts.update({
            "input_rgb_points": str(out_dir / "input_rgb_points.ply"),
            "pseudo_or_gt_instances": str(out_dir / "pseudo_or_gt_instances.ply"),
            "pred_topk_instances": str(out_dir / "pred_topk_instances.ply"),
        })

    point_embed = pred.get("point_embed")
    pca_png = out_dir / "student_point_features_pca.png"
    if isinstance(point_embed, torch.Tensor):
        pca_color, pca_info = pca_rgb(point_embed.detach().cpu())
        pca_rgb_u8 = _to_uint8(pca_color)
        with (out_dir / "pca_explained_variance.json").open("w", encoding="utf-8") as f:
            json.dump(pca_info, f, indent=2)
        if save_ply:
            save_point_cloud_ply(points, pca_rgb_u8, out_dir / "student_point_features_pca.ply")
        if save_png:
            save_topdown_png(
                pca_png,
                points,
                [("RGB", rgb), ("PCA features", pca_rgb_u8)],
                max_points=max_render_points,
            )
            artifacts["student_point_features_pca"] = str(pca_png)

    overlay_png = out_dir / "pred_vs_gt_overlay.png"
    if save_png:
        save_topdown_png(
            overlay_png,
            points,
            [("RGB", rgb), (f"{granularity} target", gt_rgb), ("pred top-k", pred_rgb)],
            max_points=max_render_points,
        )
        artifacts["pred_vs_gt_overlay"] = str(overlay_png)

    debug = pred.get("debug", {}) if isinstance(pred.get("debug"), dict) else {}
    q0 = debug.get("query_anchors_initial")
    qlayers = debug.get("query_anchors_by_layer")
    qsource = debug.get("query_source_type")
    if isinstance(q0, torch.Tensor) and isinstance(qlayers, torch.Tensor) and qlayers.numel():
        traj_png = out_dir / "query_anchor_trajectories.png"
        if save_png:
            render_query_trajectories_png(
                traj_png,
                points,
                q0.detach().cpu(),
                qlayers.detach().cpu(),
                topk_indices=top_idx,
                query_source_type=(qsource.detach().cpu() if isinstance(qsource, torch.Tensor) else None),
            )
            artifacts["query_anchor_trajectories"] = str(traj_png)

    table = make_top_queries_table(
        pred,
        topk=topk_queries,
        mask_threshold=mask_threshold,
        matched_pred_indices=matched_pred_indices,
        matched_target_indices=matched_target_indices,
        matched_ious=matched_ious,
        points=points,
    )
    with (out_dir / "top_queries_table.json").open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    artifacts["top_queries_table"] = str(out_dir / "top_queries_table.json")

    if save_npz:
        npz_payload: dict[str, np.ndarray] = {
            "top_query_indices": top_idx.numpy(),
            "top_query_scores": pred["score_logits"].detach().cpu().sigmoid()[top_idx].numpy(),
        }
        if isinstance(q0, torch.Tensor):
            npz_payload["query_anchors_initial_topk"] = q0.detach().cpu()[top_idx].numpy()
        if isinstance(qlayers, torch.Tensor) and qlayers.numel():
            npz_payload["query_anchors_by_layer_topk"] = qlayers.detach().cpu()[:, top_idx].numpy()
        np.savez_compressed(out_dir / "debug_tensors_topk.npz", **npz_payload)
        artifacts["debug_tensors_topk"] = str(out_dir / "debug_tensors_topk.npz")
    return artifacts
