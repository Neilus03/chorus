#!/usr/bin/env python3
"""Single entry point for the first student experiment.

Usage
-----
    python scripts/run_student.py --config configs/overfit_one_scene.yaml \
        --scene-dir /path/to/scene0042_00 --granularity 0.5

Everything runs in order:
    1. load config + CLI overrides
    2. set seed
    3. build one-scene dataset
    4. build targets
    5. build model       (TODO)
    6. build loss        (TODO)
    7. build trainer     (TODO)
    8. train             (TODO)
    9. final evaluation  (TODO)
   10. save metrics JSON (TODO)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ── make student package importable from repo root ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_PKG = _SCRIPT_DIR.parent
if str(_STUDENT_PKG) not in sys.path:
    sys.path.insert(0, str(_STUDENT_PKG))

from student.data import (
    SingleSceneTrainingPackDataset,
    build_instance_targets,
    load_training_pack_scene,
)
from student.data.target_builder import log_target_stats
from student.data.training_pack import print_training_pack_summary

log = logging.getLogger("run_student")


# ── config ───────────────────────────────────────────────────────────────


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_cli_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted key=value overrides, e.g. ``train.lr=3e-4``."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── output directory ─────────────────────────────────────────────────────


def build_output_dir(cfg: dict[str, Any], scene_id: str, granularity: float) -> Path:
    root = Path(cfg["experiment"]["output_root"])
    name = cfg["experiment"].get("name", "overfit_one_scene")
    run_name = f"{scene_id}_g{granularity}"
    out = root / name / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    return out


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Student instance-seg experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene-dir", type=str, default=None)
    parser.add_argument("--granularity", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "overrides", nargs="*",
        help="dotted key=value config overrides, e.g. train.lr=3e-4",
    )
    args = parser.parse_args()

    # ── 1. config ──
    cfg = load_config(args.config)
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)

    if args.scene_dir is not None:
        cfg.setdefault("data", {})["scene_dir"] = args.scene_dir
    if args.granularity is not None:
        cfg.setdefault("data", {})["granularity"] = args.granularity
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.output_root is not None:
        cfg.setdefault("experiment", {})["output_root"] = args.output_root
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    exp_cfg = cfg["experiment"]

    scene_dir = data_cfg["scene_dir"]
    granularity = float(data_cfg["granularity"])
    device = train_cfg.get("device", "cuda:0")
    seed = exp_cfg.get("seed", 42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 2. seed ──
    set_seed(seed)
    log.info("Seed: %d  Device: %s", seed, device)

    # ── 3. dataset ──
    ds = SingleSceneTrainingPackDataset(
        scene_dir,
        granularity=granularity,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
    )
    sample = ds[0]

    # ── 4. targets ──
    targets = build_instance_targets(
        sample["labels"],
        sample["supervision_mask"],
        min_instance_points=data_cfg.get("min_instance_points", 1),
    )
    log_target_stats(targets, tag=f"{ds.scene_id}/g{granularity}")

    # ── output dir ──
    out_dir = build_output_dir(cfg, ds.scene_id, granularity)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Output dir: %s", out_dir)

    # ── 5–10: model / loss / train / eval — to be implemented ──
    log.info(
        "Data pipeline verified: %d pts, %d features, %d instances.  "
        "Model / loss / trainer / eval not yet implemented.",
        ds.num_points,
        ds.feature_dim,
        targets.num_instances,
    )

    # Placeholder: save a stub metrics file so the output structure exists.
    stub_metrics = {
        "scene_id": ds.scene_id,
        "granularity": granularity,
        "num_points": ds.num_points,
        "num_instances": targets.num_instances,
        "status": "data_pipeline_only",
    }
    with (out_dir / "eval" / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(stub_metrics, f, indent=2)

    log.info("Done — data pipeline smoke-test passed.")


if __name__ == "__main__":
    main()
