#!/usr/bin/env python3
"""Summarize throughput benchmark runs from experiment output directories."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row for row in rows if row.get("train_step_profile_summary")]
    if not summaries:
        return {}
    return summaries[-1]


def _epoch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row for row in rows if row.get("epoch_summary")]
    if not summaries:
        return {}
    return summaries[-1]


def _warm_profile_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile_rows = [row for row in rows if row.get("train_step_profile")]
    if not profile_rows:
        return {}

    first_step = min(int(row["global_step"]) for row in profile_rows)
    warm_rows = [row for row in profile_rows if int(row["global_step"]) > first_step]
    if not warm_rows:
        return {}

    def _mean(key: str) -> float:
        values = [float(row[key]) for row in warm_rows]
        return sum(values) / len(values)

    return {
        "warm_profile_rows": len(warm_rows),
        "warm_step_total_ms_mean": _mean("step_total_ms"),
        "warm_step_total_ms_median": statistics.median(
            float(row["step_total_ms"]) for row in warm_rows
        ),
        "warm_data_wait_ms_mean": _mean("data_wait_ms"),
        "warm_target_build_ms_mean": _mean("target_build_ms"),
        "warm_forward_ms_mean": _mean("forward_ms"),
        "warm_loss_ms_mean": _mean("loss_ms"),
        "warm_backward_ms_mean": _mean("backward_ms"),
        "warm_optim_ms_mean": _mean("optim_ms"),
        "warm_metrics_sync_ms_mean": _mean("metrics_sync_ms"),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    final_summary = _load_json(run_dir / "final_summary.json")
    log_rows = _read_jsonl(run_dir / "train_log.jsonl")

    profile = _profile_summary(log_rows)
    epoch = _epoch_summary(log_rows)
    warm = _warm_profile_stats(log_rows)
    agg = final_summary.get("final_val_metrics", {}).get("aggregate", {})
    cfg = final_summary.get("config", {})
    train_cfg = cfg.get("train", {})

    return {
        "run_dir": str(run_dir),
        "batch_scenes_per_step": train_cfg.get("batch_scenes_per_step", 1),
        "batch_assembly_policy": train_cfg.get("batch_assembly_policy", "sequential"),
        "max_total_points_per_batch": train_cfg.get("max_total_points_per_batch", None),
        "balance_train_by_points": train_cfg.get("balance_train_by_points", False),
        "train_loss_mean": epoch.get("loss_mean"),
        "train_loss_min": epoch.get("loss_min"),
        "train_loss_max": epoch.get("loss_max"),
        "profile_step_total_ms_mean": profile.get("step_total_ms_mean"),
        "profile_step_total_ms_median": profile.get("step_total_ms_median"),
        "profile_step_total_ms_max": profile.get("step_total_ms_max"),
        "profile_data_wait_ms_mean": profile.get("data_wait_ms_mean"),
        "profile_target_build_ms_mean": profile.get("target_build_ms_mean"),
        "profile_forward_ms_mean": profile.get("forward_ms_mean"),
        "profile_loss_ms_mean": profile.get("loss_ms_mean"),
        "profile_backward_ms_mean": profile.get("backward_ms_mean"),
        "profile_optim_ms_mean": profile.get("optim_ms_mean"),
        "profile_metrics_sync_ms_mean": profile.get("metrics_sync_ms_mean"),
        "profile_num_points_total_mean": profile.get("num_points_total_mean"),
        "profile_num_points_max_scene_mean": profile.get("num_points_max_scene_mean"),
        "profile_max_rank_step_ms_over_mean": profile.get("max_rank_step_ms_over_mean"),
        "warm_profile_rows": warm.get("warm_profile_rows"),
        "warm_step_total_ms_mean": warm.get("warm_step_total_ms_mean"),
        "warm_step_total_ms_median": warm.get("warm_step_total_ms_median"),
        "warm_data_wait_ms_mean": warm.get("warm_data_wait_ms_mean"),
        "warm_target_build_ms_mean": warm.get("warm_target_build_ms_mean"),
        "warm_forward_ms_mean": warm.get("warm_forward_ms_mean"),
        "warm_loss_ms_mean": warm.get("warm_loss_ms_mean"),
        "warm_backward_ms_mean": warm.get("warm_backward_ms_mean"),
        "warm_optim_ms_mean": warm.get("warm_optim_ms_mean"),
        "warm_metrics_sync_ms_mean": warm.get("warm_metrics_sync_ms_mean"),
        "val_loss_mean": agg.get("loss_mean"),
        "val_pseudo_official_AP50_mean": agg.get("pseudo_official_AP50_mean"),
        "val_real_full_scene_official_AP50_scannet20": agg.get(
            "real_full_scene_official_AP50_scannet20"
        ),
        "val_real_full_scene_official_AP50_scannet200": agg.get(
            "real_full_scene_official_AP50_scannet200"
        ),
        "val_matched_mean_iou_mean": agg.get("matched_mean_iou_mean"),
        "total_training_time_s": final_summary.get("total_training_time_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize throughput benchmark runs")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more experiment output directories containing final_summary.json",
    )
    args = parser.parse_args()

    summaries = [summarize_run(Path(run_dir)) for run_dir in args.run_dirs]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
