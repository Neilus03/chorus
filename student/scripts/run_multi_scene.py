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
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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
from student.losses import MaskSetCriterion, MultiGranCriterion, SingleGranCriterion
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


def _parse_cuda_index(device: str) -> int | None:
    device = device.strip()
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        tail = device.split(":", 1)[1]
        try:
            return int(tail)
        except ValueError:
            return None
    return None


def _distributed_env() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_barrier() -> None:
    if _dist_ready():
        dist.barrier()


def _configure_logging(rank: int, is_main_process: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _resolve_runtime_device(
    requested_device: str,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[str, str | None]:
    device = requested_device

    if distributed:
        if device.startswith("cuda") or device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Distributed CUDA training requested, but CUDA is not available."
                )
            num_devices = torch.cuda.device_count()
            if local_rank >= num_devices:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {num_devices} CUDA device(s) are visible."
                )
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}", "nccl"
        return device, "gloo"

    idx = _parse_cuda_index(device)
    if idx is not None and idx != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = "cuda:0"
    return device, None


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
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help=(
            "Initialize model weights from a checkpoint but reset optimizer/scheduler "
            "and restart training from epoch 0."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run one validation pass and exit (use with --resume or --finetune-from).",
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
        "--wandb-run-id",
        type=str,
        default=None,
        help=(
            "Continue this W&B run (same as env WANDB_RUN_ID). "
            "Uses resume from WANDB_RESUME or defaults to 'allow'."
        ),
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Log to Weights & Biases in offline mode (no network). Sync later with: wandb sync <run_dir>",
    )
    parser.add_argument(
        "--print-model", action="store_true",
        help="Log the full module tree after build.",
    )
    parser.add_argument(
        "--augmentations",
        action="store_true",
        help=(
            "Enable LitePT ScanNet-style train augmentations (rotate, scale, flip, "
            "jitter, elastic, chromatic). Same as data.train_augmentations: true in YAML."
        ),
    )
    parser.add_argument(
        "--batch-scenes-per-step",
        type=int,
        default=None,
        help="How many scenes each rank should process per optimizer step.",
    )
    parser.add_argument(
        "--balance-train-by-points",
        action="store_true",
        default=None,
        help="Balance distributed training shards by effective scene point counts.",
    )
    parser.add_argument(
        "--drop-last-train",
        action="store_true",
        default=None,
        help="Drop remainder scenes so DDP avoids duplicate padded work.",
    )
    parser.add_argument(
        "--profile-train-steps",
        action="store_true",
        default=None,
        help="Collect sparse train-step timing buckets for throughput analysis.",
    )
    parser.add_argument(
        "--profile-every-steps",
        type=int,
        default=None,
        help="Profile one training step every N optimizer steps.",
    )
    parser.add_argument(
        "--log-scene-ids",
        action="store_true",
        default=None,
        help="Gather scene ids across ranks on logged steps (debugging only).",
    )
    parser.add_argument(
        "--no-sync-step-metrics",
        action="store_true",
        help="Skip per-step cross-rank metric reduction on the hot path.",
    )
    parser.add_argument(
        "--max-total-points-per-batch",
        type=int,
        default=None,
        help="Soft point-budget target used when batching multiple scenes per rank.",
    )
    parser.add_argument(
        "--batch-assembly-policy",
        type=str,
        choices=["sequential", "point_bucket"],
        default=None,
        help="How to group a rank's assigned scenes into optimizer steps.",
    )
    parser.add_argument(
        "--decoder-loss-mode",
        type=str,
        choices=["scene_local"],
        default=None,
        help="How batched backbone outputs are consumed by the decoder/loss path.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="dotted key=value config overrides, e.g. train.lr=3e-4",
    )
    args = parser.parse_args()
    distributed, rank, local_rank, world_size = _distributed_env()
    is_main_process = rank == 0
    _configure_logging(rank, is_main_process)

    wandb_mode_env = os.environ.get("WANDB_MODE", "").strip().lower()
    wandb_offline = args.wandb_offline or wandb_mode_env in (
        "offline",
        "dryrun",
        "dry-run",
    )
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # ── 1. config ──
    cfg = load_config(args.config)
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)

    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.max_epochs is not None:
        cfg.setdefault("train", {})["max_epochs"] = args.max_epochs
    if args.augmentations:
        cfg.setdefault("data", {})["train_augmentations"] = True
    if args.batch_scenes_per_step is not None:
        cfg.setdefault("train", {})["batch_scenes_per_step"] = args.batch_scenes_per_step
    if args.balance_train_by_points is True:
        cfg.setdefault("train", {})["balance_train_by_points"] = True
    if args.drop_last_train is True:
        cfg.setdefault("train", {})["drop_last_train"] = True
    if args.profile_train_steps is True:
        cfg.setdefault("train", {})["profile_train_steps"] = True
    if args.profile_every_steps is not None:
        cfg.setdefault("train", {})["profile_every_steps"] = args.profile_every_steps
    if args.log_scene_ids is True:
        cfg.setdefault("train", {})["log_scene_ids"] = True
    if args.no_sync_step_metrics:
        cfg.setdefault("train", {})["sync_step_metrics"] = False
    if args.max_total_points_per_batch is not None:
        cfg.setdefault("train", {})["max_total_points_per_batch"] = (
            args.max_total_points_per_batch
        )
    if args.batch_assembly_policy is not None:
        cfg.setdefault("train", {})["batch_assembly_policy"] = args.batch_assembly_policy
    if args.decoder_loss_mode is not None:
        cfg.setdefault("train", {})["decoder_loss_mode"] = args.decoder_loss_mode

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    exp_cfg = cfg["experiment"]

    granularities = parse_granularities(data_cfg)
    requested_device = train_cfg.get("device", "cuda:0")
    seed = exp_cfg.get("seed", 42)
    pg_initialized = False
    try:
        device, ddp_backend = _resolve_runtime_device(
            requested_device,
            distributed=distributed,
            local_rank=local_rank,
        )
        if distributed:
            dist.init_process_group(backend=ddp_backend or "nccl", init_method="env://")
            pg_initialized = True

        # ── 2. seed ──
        effective_seed = seed + rank
        set_seed(effective_seed)
        log.info(
            "Seed: %d (base=%d)  Device: %s  Granularities: %s",
            effective_seed,
            seed,
            device,
            granularities,
        )
        if distributed:
            log.info(
                "Distributed launch: world_size=%d rank=%d local_rank=%d backend=%s",
                world_size,
                rank,
                local_rank,
                ddp_backend,
            )

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

        train_aug = bool(
            data_cfg.get("train_augmentations", data_cfg.get("augmentations", False))
        )
        if train_aug:
            log.info("Training augmentations enabled (LitePT-style; validation unchanged)")

        use_normals = bool(data_cfg.get("use_normals", False))

        train_ds = MultiSceneDataset(
            train_dirs, granularities,
            use_colors=data_cfg.get("use_colors", True),
            append_xyz=data_cfg.get("append_xyz_to_features", False),
            use_normals=use_normals,
            preload=data_cfg.get("preload", True),
            max_points=max_pts,
            subsampling_mode=data_cfg.get("subsampling_mode", "sphere_crop"),
            sphere_point_max=data_cfg.get("sphere_point_max", None),
            train_augmentations=train_aug,
            label_source=data_cfg.get("label_source", "pack"),
            scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        )
        val_max_pts = data_cfg.get("val_max_points", None)
        val_ds = MultiSceneDataset(
            val_dirs, granularities,
            use_colors=data_cfg.get("use_colors", True),
            append_xyz=data_cfg.get("append_xyz_to_features", False),
            use_normals=use_normals,
            preload=data_cfg.get("preload", True),
            max_points=val_max_pts,
            subsampling_mode=data_cfg.get("val_subsampling_mode", data_cfg.get("subsampling_mode", "sphere_crop")),
            sphere_point_max=data_cfg.get("val_sphere_point_max", data_cfg.get("sphere_point_max", None)),
            train_augmentations=False,
            label_source=data_cfg.get("label_source", "pack"),
            scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        )
        log.info(
            "Train batching: scenes/step=%d  shard_balance=%s  drop_last=%s  "
            "batch_policy=%s  max_total_points=%s  decoder_loss_mode=%s",
            int(train_cfg.get("batch_scenes_per_step", 1)),
            bool(train_cfg.get("balance_train_by_points", False)),
            bool(train_cfg.get("drop_last_train", False)),
            train_cfg.get("batch_assembly_policy", "sequential"),
            train_cfg.get("max_total_points_per_batch", None),
            train_cfg.get("decoder_loss_mode", "scene_local"),
        )

        # ── 5. model ──
        if use_normals:
            mbb = cfg.setdefault("model", {}).setdefault("backbone", {})
            if mbb.get("in_channels", 3) == 3:
                append_xyz = bool(data_cfg.get("append_xyz_to_features", False))
                mbb["in_channels"] = 9 if append_xyz else 6
                log.info(
                    "use_normals=True: model.backbone.in_channels=%d (auto)",
                    mbb["in_channels"],
                )

        bb_cfg = model_cfg["backbone"]
        num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
        log.info("Building model ...")
        decoder_type = model_cfg.get("decoder_type", "multi_head")
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
            decoder_type=decoder_type,
        )
        total_params = sum(p.numel() for p in model.parameters())
        num_heads_display = 1 if decoder_type == "continuous" else len(granularities)
        head_label = "1 head (continuous)" if decoder_type == "continuous" else f"{num_heads_display} heads"
        log.info("Model: %s params (%s)", f"{total_params:,}", head_label)
        if args.print_model:
            for line in str(model).splitlines():
                log.info("%s", line)

        # ── 6. criterion ──
        log.info("Building criterion (decoder_type=%s) ...", decoder_type)
        base_criterion = MaskSetCriterion(
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            score_weight=loss_cfg.get("score_weight", 0.5),
        )
        if decoder_type == "continuous":
            criterion: MultiGranCriterion | SingleGranCriterion = SingleGranCriterion(
                criterion=base_criterion,
                aux_weight=loss_cfg.get("aux_weight", 0.0),
            )
        else:
            criterion = MultiGranCriterion(
                criterion=base_criterion,
                granularity_weights=loss_cfg.get("granularity_weights", None),
                aux_weight=loss_cfg.get("aux_weight", 0.0),
            )

        # ── output dir ──
        log.info("Preparing output directory ...")
        out_dir = build_output_dir(cfg)
        if is_main_process:
            with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            log.info("Output dir: %s", out_dir)
        _dist_barrier()

        # ── wandb ──
        use_wandb = is_main_process and not args.no_wandb and wandb is not None
        if use_wandb:
            run_name = args.wandb_name or exp_cfg.get("name", "multi_scene")
            init_kw: dict[str, Any] = dict(
                project=args.wandb_project,
                name=run_name,
                config={
                    **cfg,
                    "train_scenes": train_ds.scene_ids,
                    "val_scenes": val_ds.scene_ids,
                    "total_params": total_params,
                    "distributed": distributed,
                    "world_size": world_size,
                },
                dir=str(out_dir),
                tags=["multi_scene", *granularities],
            )
            if wandb_offline:
                init_kw["settings"] = wandb.Settings(mode="offline")
            wandb_run_id = (
                (args.wandb_run_id or os.environ.get("WANDB_RUN_ID") or "").strip() or None
            )
            wandb_resume = (os.environ.get("WANDB_RESUME") or "").strip() or None
            if wandb_run_id:
                init_kw["id"] = wandb_run_id
                init_kw["resume"] = wandb_resume or "allow"
                log.info(
                    "wandb run continuation: id=%s resume=%s",
                    wandb_run_id,
                    init_kw["resume"],
                )
            elif args.resume or args.finetune_from:
                log.warning(
                    "Checkpoint init (--resume/--finetune-from) but no W&B run id "
                    "(set WANDB_RUN_ID or --wandb-run-id); wandb will start a new run."
                )
            wandb.init(**init_kw)
            wandb.define_metric("epoch")
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("train_scene/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("val_scene/*", step_metric="epoch")
            wandb.define_metric("train_eval/*", step_metric="epoch")
            wandb.define_metric("train_eval_scene/*", step_metric="epoch")
            if wandb_offline:
                # wandb.run.dir is .../offline-run-.../files; metrics live in .../offline-run-.../*.wandb
                sync_dir = str(Path(wandb.run.dir).resolve().parent)
                log.info(
                    "wandb offline — sync to the hub after the run: wandb sync %s",
                    sync_dir,
                )
            else:
                log.info("wandb run: %s", wandb.run.url)
        elif is_main_process:
            log.info("wandb disabled")

        # ── 7. train ──
        log.info("Constructing trainer ...")
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
            dense_instance_ids=bool(data_cfg.get("dense_instance_ids", False)),
            fragment_merge_eval=bool(eval_cfg.get("fragment_merge_eval", False)),
            fragment_merge_num=int(eval_cfg.get("fragment_merge_num", 4)),
            fragment_merge_point_max=eval_cfg.get("fragment_merge_point_max", None),
            fragment_merge_seed=int(eval_cfg.get("fragment_merge_seed", seed)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            log_every_steps=int(train_cfg.get("log_every_steps", 1)),
            batch_scenes_per_step=int(train_cfg.get("batch_scenes_per_step", 1)),
            balance_train_by_points=bool(train_cfg.get("balance_train_by_points", False)),
            drop_last_train=bool(train_cfg.get("drop_last_train", False)),
            sync_step_metrics=bool(train_cfg.get("sync_step_metrics", True)),
            log_scene_ids=bool(train_cfg.get("log_scene_ids", False)),
            profile_train_steps=bool(train_cfg.get("profile_train_steps", False)),
            profile_every_steps=int(
                train_cfg.get("profile_every_steps", train_cfg.get("log_every_steps", 1))
            ),
            max_total_points_per_batch=train_cfg.get("max_total_points_per_batch", None),
            batch_assembly_policy=str(
                train_cfg.get("batch_assembly_policy", "sequential")
            ),
            decoder_loss_mode=str(train_cfg.get("decoder_loss_mode", "scene_local")),
            sampler_seed=seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_main_process=is_main_process,
        )

        if args.resume and args.finetune_from:
            raise ValueError("Use only one of --resume or --finetune-from.")

        if args.finetune_from:
            trainer.load_weights_only(Path(args.finetune_from))
            _dist_barrier()
        elif args.resume:
            if args.resume in {"last", "best"}:
                ckpt_path = out_dir / "checkpoints" / f"{args.resume}.pt"
            else:
                ckpt_path = Path(args.resume)
            trainer.load_checkpoint(ckpt_path)
            _dist_barrier()

        if args.eval_only:
            _dist_barrier()
            trainer._validate(epoch=trainer.current_epoch)
            _dist_barrier()
            return

        final_metrics = trainer.train()

        # ── 8. save final summary ──
        if is_main_process:
            summary = {
                "config": cfg,
                "train_scenes": train_ds.scene_ids,
                "val_scenes": val_ds.scene_ids,
                "distributed": distributed,
                "world_size": world_size,
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
    finally:
        if pg_initialized and _dist_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
