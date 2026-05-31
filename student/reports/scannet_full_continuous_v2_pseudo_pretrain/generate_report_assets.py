#!/usr/bin/env python3
"""Generate plots and point-cloud renders for the diagnostic README."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
ASSET_DIR = REPORT_DIR / "assets"
RUN_DIR = Path("/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_v2_pseudo_pretrain")
LOG_PATH = Path("/cluster/home/nedela/nedela/projects/chorus/student/logs/scannet_v2_pretrain_ddp4_562438.err")

GRANS = ("g02", "g05", "g08")
COLORS = {
    "g02": "#2b6cb0",
    "g05": "#2f855a",
    "g08": "#c05621",
}


def _save(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_full_eval() -> tuple[list[dict], list[dict]]:
    text = LOG_PATH.read_text(errors="replace")
    val_pat = re.compile(
        r"\[val epoch (?P<epoch>\d+)\] loss=(?P<loss>[0-9.]+)\s+"
        r"pseudo_official_AP50=(?P<pseudo>[0-9.]+)\s+"
        r"pseudo_oracle_AP50=(?P<oracle>[0-9.]+)\s+"
        r"real_official_AP50=\(scannet20=(?P<real20>[0-9.]+), scannet200=(?P<real200>[0-9.]+)\)\s+"
        r"matched_mIoU=(?P<miou>[0-9.]+)"
    )
    train_pat = re.compile(
        r"\[train-eval epoch (?P<epoch>\d+)\] loss=(?P<loss>[0-9.]+)\s+"
        r"pseudo_official\(AP25=(?P<ap25>[0-9.]+), AP50=(?P<ap50>[0-9.]+), "
        r"oracle_AP50=(?P<oracle>[0-9.]+), NMI=(?P<nmi>[0-9.]+), ARI=(?P<ari>[0-9.]+)\)\s+"
        r"real_official\(AP50=\(scannet20=(?P<real20>[0-9.]+), scannet200=(?P<real200>[0-9.]+)\)\)\s+"
        r"matched_mIoU=(?P<miou>[0-9.]+)"
    )
    val_rows = [{k: float(v) if k != "epoch" else int(v) for k, v in m.groupdict().items()} for m in val_pat.finditer(text)]
    train_rows = [{k: float(v) if k != "epoch" else int(v) for k, v in m.groupdict().items()} for m in train_pat.finditer(text)]
    return val_rows, train_rows


def parse_micro_eval() -> list[dict]:
    rows = []
    for path in sorted((RUN_DIR / "micro_eval").glob("micro_eval_epoch_*.json")):
        data = json.loads(path.read_text())
        row = {"epoch": int(data["epoch"])}
        for split in ("train", "val"):
            agg = data["splits"][split]["aggregate"]
            for key, value in agg.items():
                if isinstance(value, (int, float)):
                    row[f"{split}_{key}"] = float(value)
        rows.append(row)
    return rows


def parse_train_log() -> list[dict]:
    rows = []
    for line in (RUN_DIR / "train_log.jsonl").read_text(errors="replace").splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("debug_scalars") or data.get("epoch_summary"):
            continue
        active = data.get("active_granularities") or []
        if len(active) != 1:
            continue
        g = active[0]
        if g not in GRANS:
            continue
        row = {"epoch": int(data["epoch"]), "global_step": int(data["global_step"]), "granularity": g}
        for suffix in (
            "loss_mask_bce",
            "loss_mask_dice",
            "loss_score",
            "score_prob_mean_matched",
            "score_prob_mean_unmatched",
            "score_target_mean_matched",
            "score_target_mean_unmatched",
            "mask_prob_frac_ge_0p5",
            "score_prob_frac_gt_0p05",
            "num_score_targets_positive",
        ):
            key = f"{suffix}_{g}"
            if key in data and isinstance(data[key], (int, float)):
                row[suffix] = float(data[key])
        rows.append(row)
    return rows


def mean_by_epoch(rows: list[dict], granularity: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("granularity") == granularity and field in row:
            buckets[int(row["epoch"])].append(float(row[field]))
    epochs = np.array(sorted(buckets), dtype=float)
    values = np.array([np.mean(buckets[int(e)]) for e in epochs], dtype=float)
    return epochs, values


def plot_full_validation(val_rows: list[dict], train_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    if val_rows:
        e = np.array([r["epoch"] for r in val_rows])
        axes[0].plot(e, [r["pseudo"] for r in val_rows], marker="o", label="val pseudo AP50")
        axes[0].plot(e, [r["oracle"] for r in val_rows], marker="o", label="val pseudo oracle AP50")
        axes[0].plot(e, [r["real20"] for r in val_rows], marker="o", label="val real AP50 scannet20")
        axes[1].plot(e, [r["miou"] for r in val_rows], marker="o", label="val matched mIoU")
        axes[2].plot(e, [r["loss"] for r in val_rows], marker="o", label="val loss")
    if train_rows:
        e = np.array([r["epoch"] for r in train_rows])
        axes[0].plot(e, [r["ap50"] for r in train_rows], marker="s", linestyle="--", label="train sample AP50")
        axes[0].plot(e, [r["oracle"] for r in train_rows], marker="s", linestyle="--", label="train sample oracle AP50")
        axes[1].plot(e, [r["miou"] for r in train_rows], marker="s", linestyle="--", label="train sample matched mIoU")
        axes[2].plot(e, [r["loss"] for r in train_rows], marker="s", linestyle="--", label="train sample loss")
    axes[0].set_title("Full validation AP remains effectively zero")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AP50")
    axes[1].set_title("Full validation mIoU barely moves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[2].set_title("Validation loss is nearly flat")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    for ax in axes:
        _style_axis(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Full-validation summary", fontsize=14, fontweight="bold")
    _save(fig, "full_validation_progress.png")


def plot_micro_eval(rows: list[dict]) -> None:
    epochs = np.array([r["epoch"] for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    panels = [
        ("val_matched_mIoU_{}", "Matched mIoU", "mIoU"),
        ("val_pseudo_oracle_AP50_{}", "Oracle AP50", "AP50"),
        ("val_pseudo_AP50_{}", "Scored AP50", "AP50"),
        ("val_NMI_{}", "NMI", "NMI"),
    ]
    for ax, (fmt, title, ylabel) in zip(axes.flat, panels):
        for g in GRANS:
            ax.plot(epochs, [r.get(fmt.format(g), np.nan) for r in rows], color=COLORS[g], label=g, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Micro-validation trends by granularity", fontsize=14, fontweight="bold")
    _save(fig, "micro_eval_trends.png")


def plot_proposals_and_score_gap(rows: list[dict]) -> None:
    epochs = np.array([r["epoch"] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for g in GRANS:
        axes[0].plot(epochs, [r.get(f"val_proposal_kept_{g}", np.nan) for r in rows], color=COLORS[g], label=g, linewidth=2)
        axes[1].plot(epochs, [r.get(f"val_score_gap_matched_minus_unmatched_{g}", np.nan) for r in rows], color=COLORS[g], label=g, linewidth=2)
    axes[0].set_title("Kept proposals after mask threshold and min-points filter")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean kept proposals")
    axes[1].axhline(0.0, color="#111827", linewidth=1, linestyle="--")
    axes[1].set_title("Score gap: matched minus unmatched")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Probability gap")
    for ax in axes:
        _style_axis(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Proposal survival and score ranking", fontsize=14, fontweight="bold")
    _save(fig, "proposal_counts_and_score_gap.png")


def plot_training_score_diagnostics(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for g in GRANS:
        e, matched = mean_by_epoch(rows, g, "score_prob_mean_matched")
        _, unmatched = mean_by_epoch(rows, g, "score_prob_mean_unmatched")
        if len(e):
            axes[0, 0].plot(e, matched, color=COLORS[g], linewidth=2, label=f"{g} matched")
            axes[0, 0].plot(e, unmatched, color=COLORS[g], linewidth=1.4, linestyle="--", label=f"{g} unmatched")
        e, target_m = mean_by_epoch(rows, g, "score_target_mean_matched")
        _, target_u = mean_by_epoch(rows, g, "score_target_mean_unmatched")
        if len(e):
            axes[0, 1].plot(e, target_m, color=COLORS[g], linewidth=2, label=f"{g} matched")
            axes[0, 1].plot(e, target_u, color=COLORS[g], linewidth=1.4, linestyle="--", label=f"{g} unmatched")
        e, mask_frac = mean_by_epoch(rows, g, "mask_prob_frac_ge_0p5")
        if len(e):
            axes[1, 0].plot(e, mask_frac, color=COLORS[g], linewidth=2, label=g)
        e, dice = mean_by_epoch(rows, g, "loss_mask_dice")
        _, bce = mean_by_epoch(rows, g, "loss_mask_bce")
        if len(e):
            axes[1, 1].plot(e, dice, color=COLORS[g], linewidth=2, label=f"{g} dice")
            axes[1, 1].plot(e, bce, color=COLORS[g], linewidth=1.4, linestyle="--", label=f"{g} bce")
    axes[0, 0].set_title("Predicted score probabilities")
    axes[0, 0].set_ylabel("Sigmoid(score)")
    axes[0, 1].set_title("Score targets are tiny under IoU scoring")
    axes[0, 1].set_ylabel("Target")
    axes[1, 0].set_title("Mask confidence above hard threshold")
    axes[1, 0].set_ylabel("Fraction mask prob >= 0.5")
    axes[1, 1].set_title("Mask loss components")
    axes[1, 1].set_ylabel("Loss")
    for ax in axes.flat:
        _style_axis(ax)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Training diagnostics from train_log.jsonl", fontsize=14, fontweight="bold")
    _save(fig, "training_score_and_mask_diagnostics.png")


def copy_existing_overlays() -> None:
    copies = {
        "overlay_scene0488_g08.png": RUN_DIR / "debug_snapshots/epoch_000060/val/scene0488_00/g08/pred_vs_gt_overlay.png",
        "overlay_scene0655_g08.png": RUN_DIR / "debug_snapshots/epoch_000060/val/scene0655_00/g08/pred_vs_gt_overlay.png",
        "overlay_scene0427_g08.png": RUN_DIR / "debug_snapshots/epoch_000060/val/scene0427_00/g08/pred_vs_gt_overlay.png",
        "threshold_scene0488_g05.png": RUN_DIR / "score_threshold_visual_sweep_last/scene0488_00/g05/topdown_score_sweep.png",
        "threshold_scene0488_g08.png": RUN_DIR / "score_threshold_visual_sweep_last/scene0488_00/g08/topdown_score_sweep.png",
        "threshold_scene0568_g05.png": RUN_DIR / "score_threshold_visual_sweep_last/scene0568_00/g05/topdown_score_sweep.png",
        "threshold_scene0568_g08.png": RUN_DIR / "score_threshold_visual_sweep_last/scene0568_00/g08/topdown_score_sweep.png",
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in copies.items():
        if src.exists():
            shutil.copyfile(src, ASSET_DIR / name)


def read_ascii_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Could not find vertex count in {path}")
        data = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count)
    xyz = data[:, :3].astype(np.float32)
    rgb = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    return xyz, rgb


def render_pointcloud_triplet(scene: str, granularity: str, *, max_points: int = 70000) -> None:
    base = RUN_DIR / f"debug_snapshots/epoch_000060/val/{scene}/{granularity}"
    paths = [
        ("RGB input", base / "input_rgb_points.ply"),
        (f"{granularity} target", base / "pseudo_or_gt_instances.ply"),
        ("pred top-k", base / "pred_topk_instances.ply"),
    ]
    clouds = []
    rng = np.random.default_rng(42)
    for title, path in paths:
        xyz, rgb = read_ascii_ply_xyzrgb(path)
        if xyz.shape[0] > max_points:
            idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
        clouds.append((title, xyz, rgb))

    all_xyz = np.concatenate([xyz for _, xyz, _ in clouds], axis=0)
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    limits = np.stack([center - span / 2.0, center + span / 2.0], axis=1)

    fig = plt.figure(figsize=(15, 5.2), facecolor="#111827")
    for i, (title, xyz, rgb) in enumerate(clouds, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d", facecolor="#111827")
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rgb.astype(np.float32) / 255.0,
            s=0.25,
            alpha=0.95,
            linewidths=0,
            depthshade=False,
        )
        ax.view_init(elev=42, azim=-58)
        ax.set_proj_type("ortho")
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_box_aspect((1, 1, 0.55))
        ax.set_axis_off()
        ax.set_title(title, color="white", fontsize=12, pad=8)
    fig.subplots_adjust(top=0.88, wspace=0.04)
    fig.suptitle(
        f"Epoch 60 {scene} {granularity}: perspective point-cloud render",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _save(fig, f"render_{scene}_{granularity}_triplet.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    val_rows, train_rows = parse_full_eval()
    micro_rows = parse_micro_eval()
    train_rows_debug = parse_train_log()

    plot_full_validation(val_rows, train_rows)
    plot_micro_eval(micro_rows)
    plot_proposals_and_score_gap(micro_rows)
    plot_training_score_diagnostics(train_rows_debug)
    copy_existing_overlays()
    for scene in ("scene0488_00", "scene0655_00", "scene0427_00"):
        render_pointcloud_triplet(scene, "g08")
    print(f"Wrote assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
