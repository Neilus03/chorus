#!/usr/bin/env python3
"""Single entry point for the first student experiment.

Usage
-----
    python scripts/run_student.py --config configs/overfit_one_scene.yaml
    python scripts/run_student.py --config configs/overfit_one_scene.yaml --max-steps 50
    python scripts/run_student.py --config configs/overfit_one_scene.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
)
from student.data.target_builder import log_target_stats
from student.losses import MaskSetCriterion
from student.models.student_model import build_student_model
from student.engine.trainer import SingleSceneTrainer

log = logging.getLogger("run_student")

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


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
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="chorus-student")
    parser.add_argument("--wandb-name", type=str, default=None)
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

    # ── 5. model ──
    bb_cfg = model_cfg["backbone"]
    model = build_student_model(
        litept_root=bb_cfg["litept_root"],
        in_channels=bb_cfg.get("in_channels", ds.feature_dim),
        grid_size=bb_cfg.get("grid_size", 0.02),
        hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        num_queries=model_cfg.get("num_queries", 128),
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s params", f"{total_params:,}")

    # ── 6. criterion ──
    criterion = MaskSetCriterion(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        score_weight=loss_cfg.get("score_weight", 0.5),
    )

    # ── wandb ──
    use_wandb = (
        not args.no_wandb
        and wandb is not None
    )
    if use_wandb:
        run_name = args.wandb_name or f"{ds.scene_id}_g{granularity}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **cfg,
                "scene_id": ds.scene_id,
                "num_points": ds.num_points,
                "feature_dim": ds.feature_dim,
                "num_instances": targets.num_instances,
                "total_params": total_params,
            },
            dir=str(out_dir),
            tags=[ds.scene_id, f"g{granularity}", "overfit_one_scene"],
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("step")
        log.info("wandb run: %s", wandb.run.url)
    else:
        log.info("wandb disabled")

    # ── 7. train ──
    trainer = SingleSceneTrainer(
        model=model,
        criterion=criterion,
        sample=sample,
        targets=targets,
        device=device,
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        max_steps=train_cfg.get("max_steps", 2000),
        log_every=train_cfg.get("log_every", 20),
        eval_every=train_cfg.get("eval_every", 100),
        save_every=train_cfg.get("save_every", 200),
        output_dir=out_dir,
        score_threshold=eval_cfg.get("score_threshold", 0.3),
        mask_threshold=eval_cfg.get("mask_threshold", 0.5),
    )

    final_metrics = trainer.train()

    # ── 8. save final metrics ──
    final_metrics.update({
        "scene_id": ds.scene_id,
        "granularity": granularity,
        "num_points": ds.num_points,
        "num_instances": targets.num_instances,
    })
    metrics_path = out_dir / "eval" / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    log.info("Final metrics saved: %s", metrics_path)

    if use_wandb:
        wandb.summary.update({
            f"final/{k}": v for k, v in final_metrics.items()
            if isinstance(v, (int, float))
        })
        wandb.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
