#!/usr/bin/env python3
"""Multi-scene training entry point.

Usage
-----
    python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml
    python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml --max-epochs 3 --no-wandb
    python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml train.lr=3e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# ── make student package importable from repo root ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import (
    load_config,
    parse_granularities,
    resolve_num_queries,
    set_seed,
)
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.losses import MaskSetCriterion, MultiGranCriterion
from student.models.student_model import build_student_model
from student.engine.multi_scene_trainer import MultiSceneTrainer

log = logging.getLogger("run_multi_scene")

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


# ── config ───────────────────────────────────────────────────────────────


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


# ── output directory ─────────────────────────────────────────────────────


def build_output_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["experiment"]["output_root"])
    name = cfg["experiment"].get("name", "multi_scene")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    return out


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scene student training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path, or use 'last'/'best' in output/checkpoints",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--no-train-metrics",
        action="store_true",
        help="Disable periodic training-scene metric evaluation/logging",
    )
    parser.add_argument(
        "--eval-train-every-epochs",
        type=int,
        default=None,
        help="Evaluate a small subset of training scenes every N epochs "
        "(default: same cadence as validation)",
    )
    parser.add_argument(
        "--eval-train-num-scenes",
        type=int,
        default=3,
        help="How many training scenes to evaluate when train-eval is enabled",
    )
    parser.add_argument(
        "--eval-train-selection",
        type=str,
        default="random",
        choices=["first", "random"],
        help="Training-scene selection strategy (first N vs random N)",
    )
    parser.add_argument(
        "--eval-train-scene-ids",
        type=str,
        default=None,
        help="Comma-separated training scene ids to evaluate (overrides num-scenes)",
    )
    parser.add_argument("--wandb-project", type=str, default="chorus-student")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--print-model", action="store_true",
        help="Log the full module tree after build.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="dotted key=value config overrides, e.g. train.lr=3e-4",
    )
    args = parser.parse_args()

    # ── 1. config ──
    cfg = load_config(args.config)
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)

    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.max_epochs is not None:
        cfg.setdefault("train", {})["max_epochs"] = args.max_epochs

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    exp_cfg = cfg["experiment"]

    granularities = parse_granularities(data_cfg)
    device = train_cfg.get("device", "cuda:0")
    seed = exp_cfg.get("seed", 42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 2. seed ──
    set_seed(seed)
    log.info("Seed: %d  Device: %s  Granularities: %s", seed, device, granularities)

    # ── 3. scene lists ──
    scans_root = Path(data_cfg["scans_root"])
    train_split = _STUDENT_ROOT / data_cfg["train_split"]
    val_split = _STUDENT_ROOT / data_cfg["val_split"]

    train_dirs = build_scene_list(train_split, scans_root)
    val_dirs = build_scene_list(val_split, scans_root)
    log.info("Train scenes: %d   Val scenes: %d", len(train_dirs), len(val_dirs))

    # ── 4. datasets ──
    max_pts = data_cfg.get("max_points", None)
    if max_pts is not None:
        log.info("max_points=%s (subsample large scenes per step)", max_pts)

    train_ds = MultiSceneDataset(
        train_dirs, granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        preload=data_cfg.get("preload", True),
        max_points=max_pts,
    )
    val_ds = MultiSceneDataset(
        val_dirs, granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        preload=data_cfg.get("preload", True),
        max_points=max_pts,
    )

    # ── 5. model ──
    bb_cfg = model_cfg["backbone"]
    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    model = build_student_model(
        litept_root=bb_cfg["litept_root"],
        in_channels=bb_cfg.get("in_channels", 3),
        grid_size=bb_cfg.get("grid_size", 0.02),
        litept_variant=bb_cfg.get("litept_variant", "litept_s_star"),
        litept_kwargs=bb_cfg.get("litept_kwargs", None),
        hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        num_queries=num_queries,
        num_queries_by_granularity=num_queries_by_granularity,
        granularities=granularities,
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        num_decoder_heads=model_cfg.get("num_decoder_heads", 8),
        query_init=model_cfg.get("query_init", "hybrid"),
        use_positional_guidance=model_cfg.get("use_positional_guidance", True),
        learned_query_ratio=model_cfg.get("learned_query_ratio", 0.25),
        multi_scale=bb_cfg.get("multi_scale", False),
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s params (%d heads)", f"{total_params:,}", len(granularities))
    if args.print_model:
        for line in str(model).splitlines():
            log.info("%s", line)

    # ── 6. criterion ──
    base_criterion = MaskSetCriterion(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        score_weight=loss_cfg.get("score_weight", 0.5),
    )
    criterion = MultiGranCriterion(
        criterion=base_criterion,
        granularity_weights=loss_cfg.get("granularity_weights", None),
        aux_weight=loss_cfg.get("aux_weight", 0.0),
    )

    # ── output dir ──
    out_dir = build_output_dir(cfg)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Output dir: %s", out_dir)

    # ── wandb ──
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        run_name = args.wandb_name or exp_cfg.get("name", "multi_scene")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **cfg,
                "train_scenes": train_ds.scene_ids,
                "val_scenes": val_ds.scene_ids,
                "total_params": total_params,
            },
            dir=str(out_dir),
            tags=["multi_scene", *granularities],
        )
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("train_scene/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val_scene/*", step_metric="epoch")
        wandb.define_metric("train_eval/*", step_metric="epoch")
        wandb.define_metric("train_eval_scene/*", step_metric="epoch")
        log.info("wandb run: %s", wandb.run.url)
    else:
        log.info("wandb disabled")

    # ── 7. train ──
    trainer = MultiSceneTrainer(
        model=model,
        criterion=criterion,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        max_epochs=train_cfg.get("max_epochs", 50),
        eval_every_epochs=train_cfg.get("eval_every_epochs", 5),
        train_eval_every_epochs=(
            None
            if args.no_train_metrics
            else (
                args.eval_train_every_epochs
                if args.eval_train_every_epochs is not None
                else eval_cfg.get(
                    "train_eval_every_epochs", train_cfg.get("eval_every_epochs", 5)
                )
            )
        ),
        train_eval_num_scenes=args.eval_train_num_scenes,
        train_eval_scene_ids=(
            [s.strip() for s in args.eval_train_scene_ids.split(",") if s.strip()]
            if args.eval_train_scene_ids
            else None
        ),
        train_eval_selection=args.eval_train_selection,
        save_every_epochs=train_cfg.get("save_every_epochs", 10),
        output_dir=out_dir,
        score_threshold=eval_cfg.get("score_threshold", 0.3),
        mask_threshold=eval_cfg.get("mask_threshold", 0.5),
        min_points_per_proposal=eval_cfg.get("min_points_per_proposal", 30),
        eval_benchmark=eval_cfg.get("scannet_benchmark", "scannet200"),
        min_instance_points=data_cfg.get("min_instance_points", 10),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        granularities=granularities,
        config=cfg,
    )

    if args.resume:
        if args.resume in {"last", "best"}:
            ckpt_path = out_dir / "checkpoints" / f"{args.resume}.pt"
        else:
            ckpt_path = Path(args.resume)
        trainer.load_checkpoint(ckpt_path)

    final_metrics = trainer.train()

    # ── 8. save final summary ──
    summary = {
        "config": cfg,
        "train_scenes": train_ds.scene_ids,
        "val_scenes": val_ds.scene_ids,
        "final_val_metrics": final_metrics.get("final_val_metrics", {}),
        "best_val_metrics": final_metrics.get("best_val_metrics", {}),
        "best_epoch": final_metrics.get("best_epoch", -1),
        "total_training_time_s": final_metrics.get("total_training_time_s", 0),
        "per_epoch_time_s": final_metrics.get("per_epoch_time_s", []),
    }
    metrics_path = out_dir / "final_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Final summary saved: %s", metrics_path)

    if use_wandb:
        wb_final: dict[str, Any] = {}
        final_val = final_metrics.get("final_val_metrics", {}).get("aggregate", {})
        for k, v in final_val.items():
            if isinstance(v, (int, float)):
                wb_final[f"final/{k}"] = v
        wb_final["final/best_epoch"] = final_metrics.get("best_epoch", -1)
        wb_final["final/best_val_metric"] = final_metrics.get("best_val_metric", 0)
        wandb.summary.update(wb_final)
        wandb.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
