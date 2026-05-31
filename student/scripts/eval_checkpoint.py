#!/usr/bin/env python3
"""Eval-only runner for class-agnostic ScanNet checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.data.eval_sampling import resolve_eval_sampling_config
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.engine.multi_scene_evaluator import evaluate_multi_scene
from student.losses import ContinuousGeometryCriterion, MaskSetCriterion, MultiGranCriterion, SingleGranCriterion
from student.metrics.official_instance_ap import evaluate_official_and_oracle_ap, merge_ap_record_sets
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model

log = logging.getLogger("eval_checkpoint")


def apply_cli_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(value)


def _split_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _STUDENT_ROOT / p


def _build_output_dir(cfg: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        exp = cfg["experiment"]
        out = Path(exp["output_root"]) / exp.get("name", "eval_checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_val_dataset(cfg: dict[str, Any], granularities: tuple[str, ...]) -> MultiSceneDataset:
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    eval_sampling = resolve_eval_sampling_config(data_cfg, eval_cfg)
    val_dirs = build_scene_list(_split_path(data_cfg["val_split"]), Path(data_cfg["scans_root"]))
    log.info(
        "Eval sampling: subsampling_mode=%s max_points=%s sphere_point_max=%s",
        eval_sampling["subsampling_mode"],
        eval_sampling["max_points"],
        eval_sampling["sphere_point_max"],
    )
    return MultiSceneDataset(
        val_dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=data_cfg.get("preload", True),
        max_points=eval_sampling["max_points"],
        subsampling_mode=eval_sampling["subsampling_mode"],
        sphere_point_max=eval_sampling["sphere_point_max"],
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(
            data_cfg.get("scannet_gt_supervise_all_points", False)
        ),
    )


def _build_model(cfg: dict[str, Any], granularities: tuple[str, ...], device: str) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = dict(model_cfg["backbone"])
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        bb_cfg["in_channels"] = 9 if bool(data_cfg.get("append_xyz_to_features", False)) else 6

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
        decoder_type=model_cfg.get("decoder_type", "multi_head"),
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
        continuous_decoder_v2=model_cfg.get("continuous_decoder_v2", None),
    )

    prompt_cfg = train_cfg.get("prompt_finetune", {})
    if isinstance(prompt_cfg, bool):
        prompt_enabled = prompt_cfg
        prompt_cfg = {"enabled": prompt_enabled}
    elif isinstance(prompt_cfg, dict):
        prompt_enabled = bool(prompt_cfg.get("enabled", False))
    else:
        prompt_enabled = False
    if prompt_enabled:
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))),
            mode=str(prompt_cfg.get("mode", "learned")),
        )

    return model.to(device)


def _build_criterion(cfg: dict[str, Any]) -> torch.nn.Module:
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]
    base = MaskSetCriterion(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        score_weight=loss_cfg.get("score_weight", 0.5),
        class_weight=loss_cfg.get("class_weight", 0.0),
        no_object_weight=loss_cfg.get("no_object_weight", 0.1),
        cost_class_weight=loss_cfg.get("cost_class_weight", 0.0),
        score_target_mode=loss_cfg.get("score_target_mode", "binary"),
        score_loss_balance_mode=loss_cfg.get("score_loss_balance_mode", "none"),
        score_pos_weight=loss_cfg.get("score_pos_weight", 1.0),
        score_neg_weight=loss_cfg.get("score_neg_weight", 1.0),
        score_unmatched_target_mode=loss_cfg.get("score_unmatched_target_mode", "zero"),
        score_unmatched_iou_weight=loss_cfg.get("score_unmatched_iou_weight", 0.25),
        score_unmatched_iou_cap=loss_cfg.get("score_unmatched_iou_cap", 0.25),
    )
    aux_weight = float(loss_cfg.get("aux_weight", 0.0))
    gran_weights = loss_cfg.get("granularity_weights", None)
    decoder_type = str(model_cfg.get("decoder_type", "multi_head"))
    if decoder_type == "continuous_v2":
        return ContinuousGeometryCriterion(
            base,
            aux_weight=aux_weight,
            granularity_weights=gran_weights,
            center_weight=float((loss_cfg.get("continuous_v2", {}) or {}).get("center_weight", 0.05)),
        )
    if decoder_type == "continuous":
        return SingleGranCriterion(base, aux_weight=aux_weight, granularity_weights=gran_weights)
    return MultiGranCriterion(criterion=base, granularity_weights=gran_weights, aux_weight=aux_weight)


def _extract_state_dict(checkpoint: Any, checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, {}
    raise KeyError(
        f"Checkpoint {checkpoint_path} must be a raw state dict or contain model_state_dict"
    )


def load_checkpoint_for_eval(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    device: str,
    strict: bool,
    report_path: Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state, metadata = _extract_state_dict(checkpoint, checkpoint_path)
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    loading_base_into_prompt = (
        isinstance(model, FineTuningWrapper)
        and not any(k.startswith("model.") or k == "g_ft_logit" for k in state)
    )
    target = model.model if loading_base_into_prompt else model
    result = target.load_state_dict(state, strict=strict)
    report = {
        "checkpoint": str(checkpoint_path),
        "strict": bool(strict),
        "loading_base_into_prompt": bool(loading_base_into_prompt),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "checkpoint_epoch": metadata.get("epoch") if isinstance(metadata, dict) else None,
        "checkpoint_global_step": metadata.get("global_step") if isinstance(metadata, dict) else None,
    }
    if not strict or report["missing_keys"] or report["unexpected_keys"]:
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info("Checkpoint load report saved: %s", report_path)
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, allow_nan=False)


def _official_records_for_benchmark(
    per_scene: dict[str, dict[str, Any]],
    benchmark: str,
) -> dict[str, Any]:
    record_sets: list[dict[str, Any]] = []
    for scene_data in per_scene.values():
        eval_data = scene_data.get("eval", {})
        if not isinstance(eval_data, dict):
            continue
        for g_eval in eval_data.values():
            if not isinstance(g_eval, dict):
                continue
            real_by = g_eval.get("real_gt_by_benchmark", {})
            if not isinstance(real_by, dict):
                continue
            bench_data = real_by.get(benchmark, {})
            if isinstance(bench_data, dict) and isinstance(bench_data.get("official_records"), dict):
                record_sets.append(bench_data["official_records"])
    records = merge_ap_record_sets(record_sets)
    return {
        "benchmark": benchmark,
        "metrics": evaluate_official_and_oracle_ap(records),
        "records": records,
    }


def _print_summary(aggregate: dict[str, Any], official_by_bench: dict[str, Any]) -> None:
    print("Eval summary")
    for bench, payload in official_by_bench.items():
        metrics = payload.get("metrics", {})
        ap = metrics.get("AP", float("nan"))
        ap50 = metrics.get("AP50", float("nan"))
        ap25 = metrics.get("AP25", float("nan"))
        print(f"  {bench}: AP={ap:.4f} AP50={ap50:.4f} AP25={ap25:.4f}")
    print(
        "  proposals: "
        f"mean_score_pass={aggregate.get('proposal_score_pass_mean', 0.0):.2f} "
        f"mean_kept={aggregate.get('proposal_kept_mean', 0.0):.2f} "
        f"mean_removed_min_points={aggregate.get('proposal_removed_min_points_mean', 0.0):.2f}"
    )
    print(
        "  diagnostics: "
        f"matched_mIoU={aggregate.get('matched_mean_iou_mean', 0.0):.4f} "
        f"NMI={aggregate.get('real_NMI_mean_scannet20', aggregate.get('pseudo_NMI_mean', 0.0)):.4f} "
        f"ARI={aggregate.get('real_ARI_mean_scannet20', aggregate.get('pseudo_ARI_mean', 0.0)):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Accepted for command parity; unused.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    cfg = load_config(args.config)
    if args.overrides:
        apply_cli_overrides(cfg, args.overrides)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    device = args.device or cfg.get("train", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but CUDA is unavailable")

    out_dir = _build_output_dir(cfg, args.output_dir)
    with (out_dir / "eval_config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    granularities = parse_granularities(cfg["data"])
    dataset = _build_val_dataset(cfg, granularities)
    model = _build_model(cfg, granularities, str(device))
    criterion = _build_criterion(cfg)
    strict = not bool(cfg.get("model", {}).get("allow_partial_decoder_load", False))
    load_report = load_checkpoint_for_eval(
        model,
        Path(args.checkpoint),
        device=str(device),
        strict=strict,
        report_path=out_dir / "checkpoint_load_report.json",
    )

    eval_cfg = cfg.get("eval", {})
    score_threshold = eval_cfg.get("score_thresholds_by_granularity")
    if score_threshold is None:
        score_threshold = eval_cfg.get("score_threshold", 0.0)
    prompt_cfg = cfg.get("train", {}).get("prompt_finetune", {})
    prompt_enabled = bool(prompt_cfg.get("enabled", False)) if isinstance(prompt_cfg, dict) else bool(prompt_cfg)

    result = evaluate_multi_scene(
        model=model,
        dataset=dataset,
        criterion=criterion,
        device=str(device),
        granularities=granularities,
        score_threshold=score_threshold,
        class_score_threshold=eval_cfg.get("class_score_threshold", None),
        mask_threshold=float(eval_cfg.get("mask_threshold", 0.5)),
        min_points=int(eval_cfg.get("min_points_per_proposal", 30)),
        eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        eval_benchmarks=eval_cfg.get("scannet_benchmarks", None),
        min_instance_points=int(cfg["data"].get("min_instance_points", 10)),
        dense_instance_ids=bool(cfg["data"].get("dense_instance_ids", False)),
        fragment_merge_eval=bool(eval_cfg.get("fragment_merge_eval", False)),
        fragment_merge_num=int(eval_cfg.get("fragment_merge_num", 4)),
        fragment_merge_point_max=eval_cfg.get("fragment_merge_point_max", None),
        fragment_merge_seed=int(eval_cfg.get("fragment_merge_seed", cfg.get("experiment", {}).get("seed", 42))),
        prompt_finetune=prompt_enabled,
        prompt_target_granularity=granularities[0] if prompt_enabled else None,
        require_full_scene=not bool(eval_cfg.get("fragment_merge_eval", False)),
    )

    benchmarks = eval_cfg.get("scannet_benchmarks", [eval_cfg.get("scannet_benchmark", "scannet200")])
    official_by_bench = {
        str(bench): _official_records_for_benchmark(result["per_scene"], str(bench))
        for bench in benchmarks
    }

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "load_report": load_report,
        "aggregate": result.get("aggregate", {}),
        "official": {
            bench: payload["metrics"] for bench, payload in official_by_bench.items()
        },
    }
    _write_json(out_dir / "metrics_summary.json", summary)
    _write_json(out_dir / "per_scene_metrics.json", result.get("per_scene", {}))
    for bench, payload in official_by_bench.items():
        _write_json(out_dir / f"official_ap_{bench}.json", payload)

    _print_summary(result.get("aggregate", {}), official_by_bench)
    log.info("Saved eval outputs to %s", out_dir)


if __name__ == "__main__":
    main()
