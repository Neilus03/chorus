#!/usr/bin/env python3
"""Run the single-scene overfit experiment on N random scenes and summarize timing.

Example
-------
  cd student
  python scripts/run_overfit_benchmark_scenes.py \\
      --config configs/overfit_one_scene.yaml \\
      --num-scenes 5 \\
      --seed 42 \\
      --device cuda:0 \\
      --no-wandb

Uses ``run_student.py`` (or ``run_student_remap_device.py`` with ``--use-remap``).
Scene directories are discovered under ``--scans-root`` (default: parent of
``data.scene_dir`` from the YAML).

Outputs a JSON summary (default: ``<output_root>/overfit_benchmark_<timestamp>.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent

log = logging.getLogger("overfit_benchmark")


def parse_granularities_for_run_name(data_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Match ``student.config_utils.parse_granularities`` (no torch import)."""

    def _to_key(g: float | str) -> str:
        s = str(g)
        if s.startswith("g"):
            return s.replace(".", "")
        return f"g{s}".replace(".", "")

    if "granularities" in data_cfg:
        return tuple(_to_key(g) for g in data_cfg["granularities"])
    if "granularity" in data_cfg:
        return (_to_key(data_cfg["granularity"]),)
    raise KeyError("Config must specify data.granularities or data.granularity")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _has_training_pack(scene_dir: Path) -> bool:
    if (scene_dir / "scene_meta.json").is_file():
        return True
    for name in ("training_pack", "litept_pack"):
        if (scene_dir / name / "scene_meta.json").is_file():
            return True
    return False


def read_scene_id(scene_dir: Path) -> str | None:
    """Best-effort scene_id from scene_meta.json (any pack layout)."""
    for rel in (
        Path("scene_meta.json"),
        Path("training_pack") / "scene_meta.json",
        Path("litept_pack") / "scene_meta.json",
    ):
        p = scene_dir / rel
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                sid = meta.get("scene_id")
                if isinstance(sid, str):
                    return sid
            except (OSError, json.JSONDecodeError):
                pass
    return None


def discover_scene_dirs(scans_root: Path) -> list[Path]:
    if not scans_root.is_dir():
        raise FileNotFoundError(f"scans root is not a directory: {scans_root}")
    candidates: list[Path] = []
    for child in sorted(scans_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            if _has_training_pack(child):
                candidates.append(child)
    return candidates


def pick_scenes(
    scenes: list[Path],
    num: int,
    seed: int,
    exclude_ids: set[str],
) -> list[Path]:
    eligible: list[Path] = []
    for p in scenes:
        sid = read_scene_id(p) or p.name
        if sid in exclude_ids:
            continue
        eligible.append(p)
    if not eligible:
        raise RuntimeError("No eligible scenes after exclusions.")
    rng = random.Random(seed)
    k = min(num, len(eligible))
    if k < num:
        log.warning(
            "Only %d scene(s) available (requested %d). Using all eligible.",
            k,
            num,
        )
    return rng.sample(eligible, k)


def _extract_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep a small, stable subset for the benchmark JSON."""
    out: dict[str, Any] = {
        "num_points": metrics.get("num_points"),
        "total_steps": metrics.get("total_steps"),
        "best_avg_iou": metrics.get("best_avg_iou"),
        "training_time_s": metrics.get("training_time_s"),
    }
    for g in ("g02", "g05", "g08"):
        k = f"matched_mean_iou_{g}"
        if k in metrics:
            out[k] = metrics[k]
    ev = metrics.get("evaluation") or {}
    for g in ("g02", "g05", "g08"):
        if g not in ev:
            continue
        pg = ev[g].get("pseudo_gt") or {}
        rg = ev[g].get("real_gt") or {}
        out[f"pseudo_AP25_{g}"] = pg.get("AP25")
        out[f"real_AP25_{g}"] = rg.get("AP25")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit benchmark on N random training-pack scenes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/overfit_one_scene.yaml",
        help="Base config (same as run_student.py).",
    )
    parser.add_argument(
        "--scans-root",
        type=str,
        default=None,
        help="Directory containing one folder per scene. "
        "Default: parent of data.scene_dir in the config.",
    )
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=(),
        help="scene_id values to skip when sampling.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Pass through to run_student (recommended for batch).",
    )
    parser.add_argument(
        "--use-remap",
        action="store_true",
        help="Invoke run_student_remap_device.py instead of run_student.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected scenes and exit without training.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print all discovered scenes under --scans-root and exit.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Where to write the JSON summary. Default: under experiment output_root.",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra args forwarded to run_student.py (e.g. train.lr=1e-4).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    script_dir = _SCRIPT_DIR
    student_root = _STUDENT_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (student_root / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    data_cfg = cfg.get("data") or {}
    default_scene_dir = data_cfg.get("scene_dir")
    if args.scans_root is not None:
        scans_root = Path(args.scans_root).resolve()
    elif default_scene_dir:
        scans_root = Path(default_scene_dir).resolve().parent
    else:
        raise ValueError(
            "Set --scans-root or define data.scene_dir in the config.",
        )

    discovered = discover_scene_dirs(scans_root)
    log.info("Discovered %d scene(s) under %s", len(discovered), scans_root)

    if args.list_scenes:
        for p in discovered:
            sid = read_scene_id(p) or p.name
            print(f"{sid}\t{p}")
        return

    exclude = set(args.exclude)
    selected = pick_scenes(discovered, args.num_scenes, args.seed, exclude)

    exp_cfg = cfg.get("experiment") or {}
    out_root = Path(exp_cfg.get("output_root", student_root / "student_runs")).resolve()
    if not args.summary_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = out_root / f"overfit_benchmark_{ts}.json"
    else:
        summary_path = Path(args.summary_path).resolve()

    run_student = script_dir / (
        "run_student_remap_device.py" if args.use_remap else "run_student.py"
    )

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "scans_root": str(scans_root),
        "num_requested": args.num_scenes,
        "seed": args.seed,
        "exclude": sorted(exclude),
        "use_remap": args.use_remap,
        "runs": [],
    }

    log.info("Summary will be written to: %s", summary_path)

    if args.dry_run:
        for p in selected:
            sid = read_scene_id(p) or p.name
            log.info("[dry-run] %s  %s", sid, p)
        summary["runs"] = [
            {
                "scene_dir": str(p),
                "scene_id": read_scene_id(p) or p.name,
                "dry_run": True,
            }
            for p in selected
        ]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Wrote %s", summary_path)
        return

    wall_times: list[float] = []

    for i, scene_dir in enumerate(selected, start=1):
        scene_id = read_scene_id(scene_dir) or scene_dir.name
        log.info(
            "[%d/%d] scene_id=%s  dir=%s",
            i,
            len(selected),
            scene_id,
            scene_dir,
        )

        cmd: list[str] = [
            sys.executable,
            str(run_student),
            "--config",
            str(config_path),
            "--scene-dir",
            str(scene_dir),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.no_wandb:
            cmd.append("--no-wandb")
        cmd += args.extra_args

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(student_root), env=os.environ.copy())
        wall_s = time.perf_counter() - t0
        wall_times.append(wall_s)

        # Resolve output dir the same way run_student does
        granularities = parse_granularities_for_run_name(data_cfg)
        run_name = f"{scene_id}_{'_'.join(granularities)}"
        exp_name = exp_cfg.get("name", "overfit_one_scene")
        out_dir = out_root / exp_name / run_name
        metrics_path = out_dir / "eval" / "final_metrics.json"

        run_record: dict[str, Any] = {
            "scene_id": scene_id,
            "scene_dir": str(scene_dir),
            "exit_code": proc.returncode,
            "wall_time_s": wall_s,
            "output_dir": str(out_dir),
            "metrics_path": str(metrics_path),
        }

        if metrics_path.is_file():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    fm = json.load(f)
                run_record["metrics"] = _extract_metrics_summary(fm)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Could not read metrics %s: %s", metrics_path, e)
        else:
            log.warning("Missing metrics file: %s", metrics_path)

        summary["runs"].append(run_record)

    if wall_times:
        summary["aggregate"] = {
            "total_wall_time_s": sum(wall_times),
            "mean_wall_time_s": sum(wall_times) / len(wall_times),
            "min_wall_time_s": min(wall_times),
            "max_wall_time_s": max(wall_times),
            "num_completed_runs": len(wall_times),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary: %s", summary_path)

    failed = [r for r in summary["runs"] if r.get("exit_code", 0) != 0]
    if failed:
        log.error("%d run(s) failed.", len(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
