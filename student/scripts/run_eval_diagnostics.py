#!/usr/bin/env python3
"""Run compact official-AP diagnostics for a ScanNet checkpoint.

This script is evaluation-only.  It does not alter model, loss, training,
pretraining, BatchNorm, semantic, or granularity-conditioning code paths.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.engine.evaluator import (
    _ensure_chorus_importable,
    compute_legacy_best_match_recall,
    extract_proposals,
)
from student.metrics.eval_diagnostics import (
    build_diagnostic_report,
    build_scene_mask_diagnostics,
    to_jsonable,
)
from student.metrics.official_instance_ap import SCANNET_MIN_REGION_SIZE, build_instance_ap_records
from student.models.continuous_decoder import ContinuousQueryInstanceDecoder
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model

log = logging.getLogger("run_eval_diagnostics")


_GRAN_KEY_TO_VAL = {
    "g01": 0.1,
    "g02": 0.2,
    "g03": 0.3,
    "g04": 0.4,
    "g05": 0.5,
    "g06": 0.6,
    "g07": 0.7,
    "g08": 0.8,
    "g09": 0.9,
    "g10": 1.0,
}


def _gran_key_to_float(key: str) -> float:
    if key in _GRAN_KEY_TO_VAL:
        return _GRAN_KEY_TO_VAL[key]
    return float(str(key).replace("g0", "0.").replace("g", "0."))


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


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


def _parse_cuda_index(device: str) -> int | None:
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _resolve_runtime_device(
    requested_device: str,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[str, str | None]:
    if distributed:
        if requested_device.startswith("cuda") or requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed CUDA evaluation requested, but CUDA is unavailable")
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}", "nccl"
        return requested_device, "gloo"

    device = requested_device
    idx = _parse_cuda_index(device)
    if idx is not None and idx != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = "cuda:0"
    return device, None


def _maybe_apply_known_cluster_fallbacks(cfg: dict[str, Any]) -> None:
    """Keep the standalone command usable when sbatch-exported env vars are absent."""
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    exp_cfg = cfg.setdefault("experiment", {})
    bb_cfg = model_cfg.setdefault("backbone", {})

    path_fallbacks = [
        (data_cfg, "scans_root", Path("/cluster/work/igp_psr/nedela/chorus_poc/scans")),
        (bb_cfg, "litept_root", Path("/cluster/work/igp_psr/nedela/LitePT")),
        (exp_cfg, "output_root", Path("/cluster/work/igp_psr/nedela/student_runs")),
    ]
    for section, key, candidate in path_fallbacks:
        current = section.get(key)
        if current and Path(str(current)).exists():
            continue
        if candidate.exists():
            log.warning(
                "%s=%s does not exist; using local cluster fallback %s",
                key,
                current,
                candidate,
            )
            section[key] = str(candidate)


def _build_output_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["experiment"]["output_root"])
    name = cfg["experiment"].get("name", "multi_scene")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_val_dataset(cfg: dict[str, Any], granularities: tuple[str, ...]) -> MultiSceneDataset:
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    scans_root = Path(data_cfg["scans_root"])
    val_split = _STUDENT_ROOT / data_cfg["val_split"]
    val_dirs = build_scene_list(val_split, scans_root)
    use_normals = bool(data_cfg.get("use_normals", False))
    return MultiSceneDataset(
        val_dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=use_normals,
        preload=data_cfg.get("preload", True),
        max_points=data_cfg.get("val_max_points", None),
        subsampling_mode=data_cfg.get("val_subsampling_mode", data_cfg.get("subsampling_mode", "sphere_crop")),
        sphere_point_max=data_cfg.get("val_sphere_point_max", data_cfg.get("sphere_point_max", None)),
        train_augmentations=False,
        label_source=data_cfg.get("label_source", "pack"),
        scannet_eval_benchmark=eval_cfg.get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=bool(data_cfg.get("scannet_gt_supervise_all_points", False)),
    )


def _build_model(cfg: dict[str, Any], granularities: tuple[str, ...], device: str) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = model_cfg["backbone"]
    if bool(data_cfg.get("use_normals", False)) and bb_cfg.get("in_channels", 3) == 3:
        append_xyz = bool(data_cfg.get("append_xyz_to_features", False))
        bb_cfg["in_channels"] = 9 if append_xyz else 6
        log.info("use_normals=True: model.backbone.in_channels=%d (auto)", bb_cfg["in_channels"])

    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    decoder_type = str(model_cfg.get("decoder_type", "multi_head"))
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
        decoder_type=decoder_type,
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
    )

    prompt_ft_cfg = train_cfg.get("prompt_finetune", {})
    if isinstance(prompt_ft_cfg, bool):
        prompt_enabled = prompt_ft_cfg
        prompt_ft_cfg = {"enabled": prompt_enabled}
    elif isinstance(prompt_ft_cfg, dict):
        prompt_enabled = bool(prompt_ft_cfg.get("enabled", False))
    else:
        prompt_enabled = False
    if prompt_enabled:
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_ft_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(prompt_ft_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))),
            mode=str(prompt_ft_cfg.get("mode", "learned")),
        )
        log.info("Prompt fine-tuning wrapper enabled for diagnostics")

    model.to(device)
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} missing model_state_dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    loading_base_into_prompt = (
        isinstance(model, FineTuningWrapper)
        and not any(key.startswith("model.") or key == "g_ft_logit" for key in state)
    )
    if loading_base_into_prompt:
        model.model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return checkpoint


def _clear_backbone_cache(model: torch.nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _is_continuous_model(model: torch.nn.Module) -> bool:
    unwrapped = model
    if isinstance(unwrapped, FineTuningWrapper):
        unwrapped = unwrapped.model
    return isinstance(getattr(unwrapped, "decoder", None), ContinuousQueryInstanceDecoder)


def _set_eval_mode_like_existing_eval(model: torch.nn.Module) -> None:
    model.eval()
    # Existing eval-only intentionally leaves BatchNorm in train mode.  Mirror it
    # here so diagnostics are comparable without changing canonical behavior.
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            module.train()


def _forward_granularity(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    granularity: str,
) -> dict[str, torch.Tensor]:
    if isinstance(model, FineTuningWrapper):
        pred = model(points, features)
        if isinstance(pred, list):
            raise RuntimeError("Unexpected batched list output for single-scene prompt eval")
        return pred

    if _is_continuous_model(model):
        pred = model(points, features, target_g=_gran_key_to_float(granularity))
        if isinstance(pred, list):
            raise RuntimeError("Unexpected batched list output for single-scene continuous eval")
        return pred

    pred = model(points, features)
    if isinstance(pred, list):
        raise RuntimeError("Unexpected batched list output for single-scene multi-head eval")
    return pred["heads"][granularity]


def _load_real_gt(
    *,
    scene_dir: str | Path,
    scene_id: str,
    benchmark: str,
    vertex_indices: torch.Tensor | None,
    expected_points: int,
) -> tuple[np.ndarray, str]:
    _ensure_chorus_importable()
    from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids

    real_gt = load_scannet_gt_instance_ids(Path(scene_dir), scene_id, eval_benchmark=benchmark)
    eval_scope = "crop" if vertex_indices is not None else "full_scene"
    if vertex_indices is not None:
        vi = vertex_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if vi.shape[0] != expected_points:
            raise ValueError(
                f"vertex_indices length {vi.shape[0]} != model point count {expected_points}"
            )
        if vi.size > 0 and (int(vi.min()) < 0 or int(vi.max()) >= real_gt.shape[0]):
            raise ValueError("vertex_indices out of range for ScanNet GT")
        real_gt = real_gt[vi]
    if real_gt.shape[0] != expected_points:
        raise ValueError(f"GT vertex count {real_gt.shape[0]} != model point count {expected_points}")
    return np.asarray(real_gt, dtype=np.int64), eval_scope


def _evaluate_one_scene(
    model: torch.nn.Module,
    sample: dict[str, Any],
    *,
    device: str,
    benchmark: str,
    granularity: str,
    score_threshold: float,
    legacy_score_threshold: float,
    mask_threshold: float,
    min_points: int,
    legacy_min_points: int,
    nms_thresholds: list[float],
) -> dict[str, Any]:
    scene_id = str(sample["scene_id"])
    points = sample["points"].to(device)
    features = sample["features"].to(device)
    _clear_backbone_cache(model)
    with torch.no_grad():
        pred = _forward_granularity(model, points, features, granularity=granularity)

    mask_logits = pred["mask_logits"].detach().cpu()
    score_logits = pred["score_logits"].detach().cpu()
    num_points = int(mask_logits.shape[1])

    legacy_proposals, _, _ = extract_proposals(
        mask_logits,
        score_logits,
        score_threshold=legacy_score_threshold,
        mask_threshold=mask_threshold,
        min_points=legacy_min_points,
    )
    proposals, scores, query_idx, proposal_stats = extract_proposals(
        mask_logits,
        score_logits,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        min_points=min_points,
        return_stats=True,
    )

    real_gt, eval_scope = _load_real_gt(
        scene_dir=sample["scene_dir"],
        scene_id=scene_id,
        benchmark=benchmark,
        vertex_indices=sample.get("vertex_indices"),
        expected_points=num_points,
    )
    legacy = compute_legacy_best_match_recall(real_gt, legacy_proposals)
    records = build_instance_ap_records(
        scene_id=scene_id,
        gt_ids=real_gt,
        proposals=proposals,
        scores=scores,
        query_indices=query_idx,
        class_agnostic=True,
        min_valid_gt_points=min_points,
        min_valid_pred_points=min_points,
    )
    mask_diagnostics = build_scene_mask_diagnostics(
        records,
        proposals,
        nms_thresholds=nms_thresholds,
    )
    return {
        "scene_id": scene_id,
        "records": records,
        "legacy": legacy,
        "proposal_stats": proposal_stats,
        "mask_diagnostics": mask_diagnostics,
        "eval_scope": eval_scope,
    }


def _shard_indices(
    dataset: MultiSceneDataset,
    indices: list[int],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> list[int]:
    if not distributed:
        return indices
    counts = dataset.scene_point_counts
    indexed = sorted(indices, key=lambda idx: -counts[idx])
    loads = [0] * world_size
    rank_indices: list[list[int]] = [[] for _ in range(world_size)]
    for idx in indexed:
        lightest = min(range(world_size), key=lambda r: loads[r])
        rank_indices[lightest].append(idx)
        loads[lightest] += counts[idx]
    return sorted(rank_indices[rank])


def _gather_scene_outputs(
    local_outputs: list[dict[str, Any]],
    *,
    distributed: bool,
    world_size: int,
    is_main_process: bool,
) -> list[dict[str, Any]]:
    if not distributed:
        return local_outputs
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_outputs)
    if not is_main_process:
        return []
    merged: dict[str, dict[str, Any]] = {}
    for shard in gathered:
        for item in shard or []:
            scene_id = str(item["scene_id"])
            if scene_id in merged:
                raise RuntimeError(f"Duplicate scene diagnostics gathered for {scene_id}")
            merged[scene_id] = item
    return [merged[k] for k in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CHORUS student eval diagnostics")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--no-wandb", action="store_true", help="Accepted for parity; diagnostics do not use wandb")
    parser.add_argument("--max-scenes", default=None, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--benchmark", default="scannet20", type=str)
    parser.add_argument("--granularity", default="g05", type=str)
    parser.add_argument("--nms-thresholds", default="0.3,0.5,0.7", type=str)
    parser.add_argument("--topk-values", default="1,5,10,25,50,100,150,250", type=str)
    parser.add_argument("--score-threshold", default=0.0, type=float)
    parser.add_argument("--mask-threshold", default=None, type=float)
    parser.add_argument("--min-points", default=SCANNET_MIN_REGION_SIZE, type=int)
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = _distributed_env()
    is_main_process = rank == 0
    _configure_logging(rank, is_main_process)

    cfg = load_config(args.config)
    _maybe_apply_known_cluster_fallbacks(cfg)
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    requested_device = args.device or train_cfg.get("device", "cuda:0")
    device, backend = _resolve_runtime_device(
        requested_device,
        distributed=distributed,
        local_rank=local_rank,
    )
    if distributed:
        dist.init_process_group(
            backend=backend or "nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
        )

    try:
        set_seed(int(cfg.get("experiment", {}).get("seed", 42)) + rank)
        granularities = parse_granularities(cfg["data"])
        if args.granularity not in granularities:
            raise ValueError(
                f"--granularity {args.granularity!r} is not in config granularities {granularities}"
            )
        mask_threshold = (
            float(args.mask_threshold)
            if args.mask_threshold is not None
            else float(eval_cfg.get("mask_threshold", 0.5))
        )
        legacy_score_threshold = float(eval_cfg.get("score_threshold", 0.3))
        legacy_min_points = int(eval_cfg.get("min_points_per_proposal", 30))
        nms_thresholds = _parse_csv_floats(args.nms_thresholds)
        topk_values = _parse_csv_ints(args.topk_values)

        dataset = _build_val_dataset(cfg, granularities)
        model = _build_model(cfg, granularities, device)
        checkpoint = _load_checkpoint(model, Path(args.checkpoint), device)
        _set_eval_mode_like_existing_eval(model)

        all_indices = list(range(len(dataset)))
        if args.max_scenes is not None:
            all_indices = all_indices[: max(int(args.max_scenes), 0)]
        shard_indices = _shard_indices(
            dataset,
            all_indices,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        if is_main_process:
            log.info(
                "Running diagnostics: scenes=%d benchmark=%s granularity=%s score_threshold=%.3f min_points=%d",
                len(all_indices),
                args.benchmark,
                args.granularity,
                float(args.score_threshold),
                int(args.min_points),
            )
        log.info("Rank %d evaluating %d scene(s)", rank, len(shard_indices))

        local_outputs: list[dict[str, Any]] = []
        for local_pos, idx in enumerate(shard_indices, start=1):
            sample = dataset[idx]
            scene_id = str(sample["scene_id"])
            log.info(
                "[%d/%d rank-local] %s points=%d",
                local_pos,
                len(shard_indices),
                scene_id,
                int(sample["points"].shape[0]),
            )
            scene_output = _evaluate_one_scene(
                model,
                sample,
                device=device,
                benchmark=args.benchmark,
                granularity=args.granularity,
                score_threshold=float(args.score_threshold),
                legacy_score_threshold=legacy_score_threshold,
                mask_threshold=mask_threshold,
                min_points=int(args.min_points),
                legacy_min_points=legacy_min_points,
                nms_thresholds=nms_thresholds,
            )
            local_outputs.append(scene_output)

        scene_outputs = _gather_scene_outputs(
            local_outputs,
            distributed=distributed,
            world_size=world_size,
            is_main_process=is_main_process,
        )
        if is_main_process:
            output = Path(args.output) if args.output else _build_output_dir(cfg) / "eval_diagnostics_summary.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            settings = {
                "score_threshold": float(args.score_threshold),
                "legacy_score_threshold": legacy_score_threshold,
                "mask_threshold": mask_threshold,
                "min_points": int(args.min_points),
                "legacy_min_points": legacy_min_points,
                "nms_thresholds": nms_thresholds,
                "topk_values": topk_values,
                "max_scenes": args.max_scenes,
                "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
                "checkpoint_global_step": int(checkpoint.get("global_step", 0) or 0),
                "distributed": distributed,
                "world_size": world_size,
            }
            report = build_diagnostic_report(
                scene_outputs=scene_outputs,
                checkpoint=str(Path(args.checkpoint)),
                config=str(Path(args.config)),
                benchmark=args.benchmark,
                granularity=args.granularity,
                nms_thresholds=nms_thresholds,
                topk_values=topk_values,
                settings=settings,
            )
            tmp_output = output.with_suffix(output.suffix + ".tmp")
            with tmp_output.open("w", encoding="utf-8") as f:
                json.dump(to_jsonable(report), f, indent=2, sort_keys=True, allow_nan=False)
                f.write("\n")
            tmp_output.replace(output)
            baseline = report["baseline"]
            diagnosis = report["diagnosis"]
            log.info(
                "Wrote %s  AP50=%.4f AP25=%.4f oracle_AP50=%.4f next=%s",
                output,
                float(baseline.get("official_AP50") or 0.0),
                float(baseline.get("official_AP25") or 0.0),
                float(baseline.get("oracle_AP50") or 0.0),
                diagnosis.get("recommended_next_step"),
            )
        _dist_barrier()
    finally:
        if _dist_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
