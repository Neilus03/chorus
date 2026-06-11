#!/usr/bin/env python3
"""Export CHORUS point-cloud inspection bundles.

The bundle is intentionally simple: large arrays stay as .npy files and the
JSON manifest points to them. Training-pack arrays are referenced in place;
student predictions, GT labels, and feature tensors are written under the
bundle directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_UI_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _UI_ROOT.parent
_STUDENT_ROOT = _REPO_ROOT / "student"
_STUDENT_SCRIPTS = _STUDENT_ROOT / "scripts"
_CHORUS_OUTER = _REPO_ROOT / "chorus"

for candidate in (_STUDENT_ROOT, _STUDENT_SCRIPTS, _CHORUS_OUTER):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from student.config_utils import load_config, parse_granularities, resolve_num_queries, set_seed  # noqa: E402
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list  # noqa: E402
from student.data.training_pack import _load_scene_meta, _resolve_pack_dir  # noqa: E402
from student.models.continuous_base import is_continuous_decoder  # noqa: E402
from student.models.finetune_wrapper import FineTuningWrapper  # noqa: E402
from student.models.student_model import build_student_model  # noqa: E402

log = logging.getLogger("export_inspection_bundle")

SCHEMA_VERSION = "chorus_inspection_bundle/v1"
DEFAULT_GRANULARITIES = ("g02", "g05", "g08")
DEFAULT_GT_BENCHMARKS = ("scannet20", "scannet200", "all")


def apply_cli_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)


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


def _extract_state_dict(checkpoint: Any, checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, {}
    raise KeyError(f"Checkpoint {checkpoint_path} must be a raw state dict or contain model_state_dict")


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
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(report), f, indent=2, allow_nan=False)
        log.info("Checkpoint load report saved: %s", report_path)
    return report


def normalize_granularity_key(value: str | float | int) -> str:
    if isinstance(value, str):
        key = value.strip()
        if key.startswith("g") and len(key) == 3 and key[1:].isdigit():
            return key
        if key.startswith("g"):
            key = key[1:]
        value = float(key)
    return f"g{int(round(float(value) * 10)):02d}"


def granularity_key_to_float(key: str) -> float:
    key = normalize_granularity_key(key)
    return int(key[1:]) / 10.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return _jsonable(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, allow_nan=False)


def _artifact(path: Path, *, dtype: str | None = None, shape: Iterable[int] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path.resolve())}
    if dtype is not None:
        out["dtype"] = dtype
    if shape is not None:
        out["shape"] = [int(x) for x in shape]
    return out


def _safe_array_meta(path: Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    return _artifact(path, dtype=str(arr.dtype), shape=arr.shape)


def training_pack_manifest_entries(scene_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any], int]:
    pack_dir = _resolve_pack_dir(scene_dir)
    meta = _load_scene_meta(pack_dir)
    points_path = pack_dir / "points.npy"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points.npy in {pack_dir}")

    arrays: dict[str, Any] = {"points": _safe_array_meta(points_path)}
    for optional_name in ("colors.npy", "normals.npy"):
        optional_path = pack_dir / optional_name
        if optional_path.exists():
            arrays[optional_name.removesuffix(".npy")] = _safe_array_meta(optional_path)

    pseudo: dict[str, Any] = {}
    for raw_key, file_name in sorted(meta.get("label_files", {}).items()):
        label_path = pack_dir / str(file_name)
        if not label_path.exists():
            raise FileNotFoundError(f"scene_meta declares missing label file: {label_path}")
        g_key = normalize_granularity_key(str(raw_key))
        pseudo[g_key] = {
            **_safe_array_meta(label_path),
            "granularity_key": g_key,
            "source_key": raw_key,
        }

    num_points = int(meta.get("num_points", np.load(points_path, mmap_mode="r").shape[0]))
    return pack_dir, meta, {"arrays": arrays, "pseudo": pseudo}, num_points


def prediction_arrays_from_logits(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    *,
    score_threshold: float,
    mask_threshold: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Convert query logits into point-aligned labels.

    Kept queries are painted from low score to high score, so higher-score
    masks win overlaps.
    """
    masks = (mask_logits.detach().float().sigmoid().cpu().numpy() >= float(mask_threshold))
    scores = score_logits.detach().float().sigmoid().cpu().numpy().astype(np.float32)
    if masks.ndim != 2:
        raise ValueError(f"mask_logits must be [Q, N], got masks shape {masks.shape}")

    num_queries, num_points = masks.shape
    areas = masks.sum(axis=1).astype(np.int64)
    kept = (scores >= float(score_threshold)) & (areas >= int(min_points))
    kept_query_ids = np.where(kept)[0]
    order = kept_query_ids[np.argsort(scores[kept_query_ids], kind="stable")]

    pred_labels = np.full((num_points,), -1, dtype=np.int32)
    pred_scores = np.zeros((num_points,), dtype=np.float32)
    pred_query_ids = np.full((num_points,), -1, dtype=np.int32)

    query_to_instance: dict[int, int] = {}
    for instance_label, query_id in enumerate(order.tolist()):
        mask = masks[query_id]
        pred_labels[mask] = int(instance_label)
        pred_scores[mask] = float(scores[query_id])
        pred_query_ids[mask] = int(query_id)
        query_to_instance[int(query_id)] = int(instance_label)

    query_table: list[dict[str, Any]] = []
    for query_id in range(num_queries):
        query_table.append(
            {
                "query_id": int(query_id),
                "score_probability": float(scores[query_id]),
                "mask_area": int(areas[query_id]),
                "kept": bool(kept[query_id]),
                "instance_label": query_to_instance.get(int(query_id)),
            }
        )
    return pred_labels, pred_scores, pred_query_ids, query_table


def _set_eval_mode_with_bn_training(model: torch.nn.Module) -> None:
    model.eval()
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


def _clear_backbone_cache(model: torch.nn.Module) -> None:
    target = model.model if isinstance(model, FineTuningWrapper) else model
    backbone = getattr(target, "backbone", None)
    if backbone is not None and hasattr(backbone, "_cached_voxelization"):
        backbone._cached_voxelization = None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if isinstance(model, FineTuningWrapper) else model


@torch.no_grad()
def _predict_head(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    granularity: str,
) -> dict[str, torch.Tensor]:
    _clear_backbone_cache(model)
    _set_eval_mode_with_bn_training(model)
    base = _unwrap_model(model)
    if isinstance(model, FineTuningWrapper):
        flat = model(points, features)
    elif is_continuous_decoder(getattr(base, "decoder", None)):
        flat = model(points, features, target_g=granularity_key_to_float(granularity))
    else:
        pred = model(points, features)
        if isinstance(pred, dict) and "heads" in pred and granularity in pred["heads"]:
            return pred["heads"][granularity]
        if isinstance(pred, dict) and "mask_logits" in pred:
            return pred
        raise TypeError("Expected multi-head or flat student prediction output")
    return {
        "mask_logits": flat["mask_logits"],
        "score_logits": flat["score_logits"],
        **({"class_logits": flat["class_logits"]} if "class_logits" in flat else {}),
    }


@torch.no_grad()
def _extract_feature_sources(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    feature_sources: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if not feature_sources:
        return {}
    _clear_backbone_cache(model)
    _set_eval_mode_with_bn_training(model)
    base = _unwrap_model(model)
    bb = base.backbone(points, features)
    point_feat = bb.point_feat
    out: dict[str, np.ndarray] = {}
    if "backbone_point_feat" in feature_sources:
        out["backbone_point_feat"] = F.normalize(point_feat.detach().float(), dim=-1).cpu().numpy()
    if "decoder_mask_feat" in feature_sources:
        decoder = getattr(base, "decoder", None)
        point_mask_proj = getattr(decoder, "point_mask_proj", None)
        if point_mask_proj is None:
            raise ValueError("decoder_mask_feat requires decoder.point_mask_proj")
        mask_feat = point_mask_proj(point_feat)
        out["decoder_mask_feat"] = F.normalize(mask_feat.detach().float(), dim=-1).cpu().numpy()
    unknown = sorted(set(feature_sources) - {"backbone_point_feat", "decoder_mask_feat"})
    if unknown:
        raise ValueError(f"Unsupported feature source(s): {unknown}")
    return out


def _export_gt_labels(
    scene_dir: Path,
    scene_id: str,
    out_dir: Path,
    benchmarks: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    gt_entries: dict[str, Any] = {}
    warnings: list[str] = []
    try:
        from chorus.datasets.scannet.gt import load_scannet_gt_instances
    except Exception as exc:
        return gt_entries, [f"ScanNet GT import failed: {exc}"]

    for benchmark in benchmarks:
        try:
            gt = load_scannet_gt_instances(scene_dir, scene_id, eval_benchmark=benchmark)
            labels = np.asarray(gt.instance_ids, dtype=np.int64)
            labels = np.where(labels == 0, -1, labels).astype(np.int32, copy=False)
            labels_path = out_dir / f"gt_instances_{benchmark}.npy"
            np.save(labels_path, labels)
            classes_path = out_dir / f"gt_instance_classes_{benchmark}.json"
            _write_json(classes_path, gt.instance_class_ids)
            gt_entries[benchmark] = {
                "labels": _safe_array_meta(labels_path),
                "instance_classes": {"path": str(classes_path.resolve())},
            }
        except Exception as exc:
            warnings.append(f"GT export skipped for {benchmark}: {exc}")
    return gt_entries, warnings


def write_pack_only_bundle(
    scene_dir: Path,
    out_dir: Path,
    *,
    gt_benchmarks: tuple[str, ...] = DEFAULT_GT_BENCHMARKS,
    defaults: dict[str, Any] | None = None,
) -> Path:
    pack_dir, meta, pack_entries, _ = training_pack_manifest_entries(scene_dir)
    scene_id = str(meta.get("scene_id", Path(scene_dir).name))
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_entries, warnings = _export_gt_labels(pack_dir.parent, scene_id, out_dir, gt_benchmarks)
    pseudo_keys = sorted(pack_entries["pseudo"])
    default_values = {
        "granularity": "g05" if "g05" in pseudo_keys else (pseudo_keys[0] if pseudo_keys else None),
        "score_threshold": 0.0,
        "mask_threshold": 0.5,
        "min_points": 30,
    }
    if defaults:
        default_values.update(defaults)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "dataset": meta.get("dataset"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_pack_dir": str(pack_dir.resolve()),
        "source": {"kind": "training_pack", "scene_meta": meta},
        "arrays": pack_entries["arrays"],
        "labels": {
            "pseudo": pack_entries["pseudo"],
            "predictions": {},
            "gt": gt_entries,
        },
        "features": {},
        "defaults": default_values,
        "warnings": warnings,
    }
    manifest_path = out_dir / "inspection_bundle.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _build_dataset_for_scene_dirs(
    cfg: dict[str, Any],
    scene_dirs: list[Path],
    granularities: tuple[str, ...],
) -> MultiSceneDataset:
    data_cfg = cfg["data"]
    return MultiSceneDataset(
        scene_dirs,
        granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        use_normals=bool(data_cfg.get("use_normals", False)),
        preload=False,
        max_points=None,
        subsampling_mode="none",
        sphere_point_max=None,
        train_augmentations=False,
        label_source="pack",
        scannet_eval_benchmark=cfg.get("eval", {}).get("scannet_benchmark", "all"),
        scannet_gt_supervise_all_points=False,
    )


def _scene_dirs_from_args(cfg: dict[str, Any] | None, args: argparse.Namespace) -> list[Path]:
    direct_dirs = [Path(p).resolve() for p in args.scene_dir]
    if direct_dirs:
        return direct_dirs
    if cfg is None:
        raise ValueError("--scene-dir is required when --config is not provided")
    data_cfg = cfg["data"]
    all_dirs = build_scene_list(
        Path(data_cfg["val_split"]) if Path(data_cfg["val_split"]).is_absolute() else _STUDENT_ROOT / data_cfg["val_split"],
        Path(data_cfg["scans_root"]),
    )
    if args.all_scenes:
        return all_dirs
    if not args.scenes:
        raise ValueError("Provide --scenes, --scene-dir, or --all-scenes")
    by_id = {p.name: p for p in all_dirs}
    missing = [scene for scene in args.scenes if scene not in by_id]
    if missing:
        raise ValueError(f"Scene(s) not found in validation split: {missing}")
    return [by_id[scene] for scene in args.scenes]


def export_prediction_bundle(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    scene_dirs: list[Path],
    out_root: Path,
    *,
    granularities: tuple[str, ...],
    device: str,
    score_threshold: float,
    mask_threshold: float,
    min_points: int,
    feature_sources: tuple[str, ...],
    gt_benchmarks: tuple[str, ...],
) -> list[Path]:
    log.info("Building dataset for %d scene(s): %s", len(scene_dirs), ", ".join(str(p) for p in scene_dirs))
    dataset = _build_dataset_for_scene_dirs(cfg, scene_dirs, granularities)
    log.info("Building model on %s for granularities: %s", device, ", ".join(granularities))
    model = _build_model(cfg, granularities, device)
    log.info("Loading checkpoint: %s", checkpoint_path)
    load_checkpoint_for_eval(
        model,
        checkpoint_path,
        device=device,
        strict=not bool(cfg.get("model", {}).get("allow_partial_decoder_load", False)),
        report_path=out_root / "checkpoint_load_report.json",
    )

    manifest_paths: list[Path] = []
    for idx in range(len(dataset)):
        log.info("Loading full-scene sample %d/%d", idx + 1, len(dataset))
        sample = dataset.get_full_item(idx)
        scene_id = str(sample["scene_id"])
        scene_dir = Path(sample["scene_dir"]).resolve()
        scene_out = out_root / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        manifest_path = write_pack_only_bundle(
            scene_dir,
            scene_out,
            gt_benchmarks=gt_benchmarks,
            defaults={
                "granularity": "g05" if "g05" in granularities else granularities[0],
                "score_threshold": float(score_threshold),
                "mask_threshold": float(mask_threshold),
                "min_points": int(min_points),
            },
        )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        points = sample["points"].to(device)
        input_features = sample["features"].to(device)
        predictions: dict[str, Any] = {}

        for g_key in granularities:
            log.info("[%s] exporting predictions for %s", scene_id, g_key)
            head = _predict_head(model, points, input_features, g_key)
            pred_labels, pred_scores, pred_query_ids, query_table = prediction_arrays_from_logits(
                head["mask_logits"],
                head["score_logits"],
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                min_points=min_points,
            )
            labels_path = scene_out / f"pred_labels_{g_key}.npy"
            scores_path = scene_out / f"pred_scores_{g_key}.npy"
            query_ids_path = scene_out / f"pred_query_ids_{g_key}.npy"
            table_path = scene_out / f"pred_query_table_{g_key}.json"
            np.save(labels_path, pred_labels)
            np.save(scores_path, pred_scores)
            np.save(query_ids_path, pred_query_ids)
            _write_json(table_path, query_table)
            predictions[g_key] = {
                "pred_labels": _safe_array_meta(labels_path),
                "pred_scores": _safe_array_meta(scores_path),
                "pred_query_ids": _safe_array_meta(query_ids_path),
                "query_table": {"path": str(table_path.resolve())},
            }

        feature_entries: dict[str, Any] = {}
        if feature_sources:
            log.info("[%s] exporting feature sources: %s", scene_id, ", ".join(feature_sources))
            feature_arrays = _extract_feature_sources(model, points, input_features, feature_sources)
            for name, values in feature_arrays.items():
                values = np.asarray(values, dtype=np.float32)
                values16 = values.astype(np.float16)
                feature_path = scene_out / f"features_{name}.npy"
                np.save(feature_path, values16)
                feature_entries[name] = {
                    **_safe_array_meta(feature_path),
                    "dim": int(values16.shape[1]) if values16.ndim == 2 else None,
                    "normalized": True,
                }

        manifest["labels"]["predictions"] = predictions
        manifest["features"] = feature_entries
        manifest["source"]["checkpoint"] = str(checkpoint_path.resolve())
        _write_json(manifest_path, manifest)
        manifest_paths.append(manifest_path)
    return manifest_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Student eval/training config for checkpoint export.")
    parser.add_argument("--checkpoint", default=None, help="Student checkpoint to export predictions from.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-dir", action="append", default=[], help="Direct scene or training-pack directory. May be repeated.")
    parser.add_argument("--scenes", nargs="*", default=[], help="Scene ids from the config validation split.")
    parser.add_argument("--all-scenes", action="store_true", help="Export all scenes from the config validation split.")
    parser.add_argument("--granularities", nargs="*", default=None, help="Granularity keys, e.g. g02 g05 g08.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--feature-sources", nargs="*", default=["decoder_mask_feat"], choices=["decoder_mask_feat", "backbone_point_feat"])
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--gt-benchmarks", nargs="*", default=list(DEFAULT_GT_BENCHMARKS))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
    cfg: dict[str, Any] | None = None
    if args.config:
        log.info("Loading config: %s", args.config)
        cfg = load_config(args.config)
        if args.overrides:
            log.info("Applying %d CLI override(s)", len(args.overrides))
            apply_cli_overrides(cfg, args.overrides)
        set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    scene_dirs = _scene_dirs_from_args(cfg, args)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("Output root: %s", out_root)
    granularities = tuple(
        normalize_granularity_key(g)
        for g in (args.granularities or (parse_granularities(cfg["data"]) if cfg else DEFAULT_GRANULARITIES))
    )
    score_threshold = float(args.score_threshold if args.score_threshold is not None else (cfg or {}).get("eval", {}).get("score_threshold", 0.0))
    mask_threshold = float(args.mask_threshold if args.mask_threshold is not None else (cfg or {}).get("eval", {}).get("mask_threshold", 0.5))
    min_points = int(args.min_points if args.min_points is not None else (cfg or {}).get("eval", {}).get("min_points_per_proposal", 30))
    feature_sources = tuple(() if args.no_features else args.feature_sources)
    gt_benchmarks = tuple(args.gt_benchmarks)

    if args.checkpoint:
        if cfg is None:
            raise ValueError("--checkpoint requires --config")
        device = args.device or cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}) but CUDA is unavailable")
        paths = export_prediction_bundle(
            cfg,
            Path(args.checkpoint).resolve(),
            scene_dirs,
            out_root,
            granularities=granularities,
            device=str(device),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_points=min_points,
            feature_sources=feature_sources,
            gt_benchmarks=gt_benchmarks,
        )
    else:
        paths = []
        for scene_dir in scene_dirs:
            scene_id = scene_dir.parent.name if scene_dir.name in ("training_pack", "litept_pack") else scene_dir.name
            paths.append(
                write_pack_only_bundle(
                    scene_dir,
                    out_root / scene_id,
                    gt_benchmarks=gt_benchmarks,
                    defaults={
                        "granularity": "g05" if "g05" in granularities else granularities[0],
                        "score_threshold": score_threshold,
                        "mask_threshold": mask_threshold,
                        "min_points": min_points,
                    },
                )
            )

    print("Exported inspection bundles:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
