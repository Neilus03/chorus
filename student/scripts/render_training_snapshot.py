#!/usr/bin/env python3
"""Render local debug snapshots for a saved student checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.engine.debug_observability import DebugObserver
from student.losses import ContinuousGeometryCriterion, MaskSetCriterion
from student.models.student_model import build_student_model


def _build_model(cfg: dict, device: str) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    bb_cfg = dict(model_cfg["backbone"])
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        bb_cfg["in_channels"] = 9 if bool(data_cfg.get("append_xyz_to_features", False)) else 6
    granularities = parse_granularities(data_cfg)
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
        multi_scale_indices=bb_cfg.get("multi_scale_indices", None),
        decoder_type=model_cfg.get("decoder_type", "continuous_v2"),
        continuous_decoder_v2=model_cfg.get("continuous_decoder_v2", None),
    )
    model.to(device)
    return model


def _load_weights(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def _build_dataset(cfg: dict, split_key: str, granularities: tuple[str, ...]) -> MultiSceneDataset:
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    dirs = build_scene_list(_STUDENT_ROOT / data_cfg[split_key], Path(data_cfg["scans_root"]))
    return MultiSceneDataset(
        dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=False,
        max_points=None,
        subsampling_mode="none",
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(data_cfg.get("scannet_gt_supervise_all_points", False)),
    )


def _build_criterion(cfg: dict) -> ContinuousGeometryCriterion:
    loss_cfg = cfg["loss"]
    base = MaskSetCriterion(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        score_weight=loss_cfg.get("score_weight", 0.5),
        score_target_mode=loss_cfg.get("score_target_mode", "binary"),
        score_loss_balance_mode=loss_cfg.get("score_loss_balance_mode", "none"),
        score_pos_weight=loss_cfg.get("score_pos_weight", 1.0),
        score_neg_weight=loss_cfg.get("score_neg_weight", 1.0),
        score_unmatched_target_mode=loss_cfg.get("score_unmatched_target_mode", "zero"),
        score_unmatched_iou_weight=loss_cfg.get("score_unmatched_iou_weight", 0.25),
        score_unmatched_iou_cap=loss_cfg.get("score_unmatched_iou_cap", 0.25),
    )
    return ContinuousGeometryCriterion(
        base,
        aux_weight=float(loss_cfg.get("aux_weight", 0.0)),
        granularity_weights=loss_cfg.get("granularity_weights", None),
        center_weight=float((loss_cfg.get("continuous_v2", {}) or {}).get("center_weight", 0.05)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    granularities = parse_granularities(cfg["data"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["experiment"]["output_root"]) / cfg["experiment"]["name"]
    debug_cfg = dict(cfg.get("debug", {}) or {})
    debug_cfg["enabled"] = True
    debug_cfg.setdefault("rich_snapshots", {})
    debug_cfg["rich_snapshots"]["enabled"] = True
    debug_cfg["rich_snapshots"]["every_epochs"] = 1

    model = _build_model(cfg, args.device)
    _load_weights(model, Path(args.checkpoint), args.device)
    train_ds = _build_dataset(cfg, "train_split", granularities)
    val_ds = _build_dataset(cfg, "val_split", granularities)
    observer = DebugObserver(output_dir=output_dir, debug_config=debug_cfg, is_main_process=True)
    observer.write_rich_snapshots_if_due(
        epoch=max(int(args.epoch), 1),
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        criterion=_build_criterion(cfg),
        device=args.device,
        granularities=granularities,
        min_instance_points=int(cfg["data"].get("min_instance_points", 10)),
        dense_instance_ids=bool(cfg["data"].get("dense_instance_ids", False)),
    )
    observer.close()


if __name__ == "__main__":
    main()
