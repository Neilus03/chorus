#!/usr/bin/env python3
"""Audit pseudo-label alignment against ScanNet real instance annotations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries

log = logging.getLogger("audit_pseudo_labels")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _to_granularity_key(value: str) -> str:
    value = str(value).strip()
    if value.startswith("g"):
        return value.replace(".", "")
    return f"g{value}".replace(".", "")


def _parse_granularities(value: str | None, available: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return available
    requested = tuple(_to_granularity_key(v) for v in value.split(",") if v.strip())
    missing = [g for g in requested if g not in available]
    if missing:
        raise ValueError(f"Requested granularities {missing} not in config granularities {available}")
    return requested


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> float | None:
    vals: list[float] = []
    for value in values:
        f = _finite_or_none(value)
        if f is not None:
            vals.append(f)
    return float(sum(vals) / len(vals)) if vals else None


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), q))


def _size_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_size_mean": float(values.mean()) if values.size else None,
        f"{prefix}_size_min": float(values.min()) if values.size else None,
        f"{prefix}_size_p10": _percentile(values, 10),
        f"{prefix}_size_p25": _percentile(values, 25),
        f"{prefix}_size_median": _percentile(values, 50),
        f"{prefix}_size_p75": _percentile(values, 75),
        f"{prefix}_size_p90": _percentile(values, 90),
        f"{prefix}_size_max": float(values.max()) if values.size else None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "split",
        "scene_id",
        "granularity",
        "benchmark",
        "num_pseudo_instances",
        "num_real_instances",
        "query_count",
        "query_to_pseudo_ratio",
        "pseudo_fraction_best_iou_ge_25",
        "pseudo_fraction_best_iou_ge_50",
        "real_fraction_best_iou_ge_25",
        "real_fraction_best_iou_ge_50",
        "pseudo_point_fraction_on_real_instances",
    ]
    fields: list[str] = []
    seen: set[str] = set()
    for key in preferred + sorted({key for row in rows for key in row}):
        if key not in seen:
            fields.append(key)
            seen.add(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})
    tmp.replace(path)


def _maybe_apply_known_cluster_fallbacks(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    exp_cfg = cfg.setdefault("experiment", {})
    bb_cfg = model_cfg.setdefault("backbone", {})
    for section, key, fallback in [
        (data_cfg, "scans_root", Path("/cluster/work/igp_psr/nedela/chorus_poc/scans")),
        (bb_cfg, "litept_root", Path("/cluster/work/igp_psr/nedela/LitePT")),
        (exp_cfg, "output_root", Path("/cluster/work/igp_psr/nedela/student_runs")),
    ]:
        current = section.get(key)
        if current and Path(str(current)).exists():
            continue
        if fallback.exists():
            log.warning("%s=%s does not exist; using %s", key, current, fallback)
            section[key] = str(fallback)
    metadata_root = Path("/cluster/work/igp_psr/nedela/LitePT/datasets/preprocessing/scannet/meta_data")
    if "CHORUS_SCANNET_METADATA_ROOT" not in os.environ and (metadata_root / "scannetv2-labels.combined.tsv").exists():
        os.environ["CHORUS_SCANNET_METADATA_ROOT"] = str(metadata_root)
        log.warning("CHORUS_SCANNET_METADATA_ROOT not set; using %s", metadata_root)


def load_scannet_instance_ids(scene_dir: Path, scene_id: str, benchmark: str) -> np.ndarray:
    from student.engine.evaluator import _ensure_chorus_importable

    _ensure_chorus_importable()
    from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids

    return np.asarray(
        load_scannet_gt_instance_ids(scene_dir, scene_id, eval_benchmark=benchmark),
        dtype=np.int64,
    )


def audit_label_alignment(
    pseudo_labels: np.ndarray,
    supervision_mask: np.ndarray,
    real_gt: np.ndarray,
    *,
    query_count: int,
    min_instance_points: int = 1,
) -> dict[str, Any]:
    """Compute pseudo-to-real and real-to-pseudo best-IoU alignment statistics."""
    pseudo = np.asarray(pseudo_labels, dtype=np.int64)
    supervision = np.asarray(supervision_mask, dtype=bool)
    real = np.asarray(real_gt, dtype=np.int64)
    if pseudo.shape != real.shape:
        raise ValueError(f"pseudo_labels shape {pseudo.shape} != real_gt shape {real.shape}")
    if supervision.shape != pseudo.shape:
        raise ValueError(f"supervision_mask shape {supervision.shape} != pseudo_labels shape {pseudo.shape}")

    pseudo_valid = supervision & (pseudo >= 0)
    pseudo_ids, pseudo_sizes = np.unique(pseudo[pseudo_valid], return_counts=True)
    keep = pseudo_sizes >= int(min_instance_points)
    pseudo_ids = pseudo_ids[keep]
    pseudo_sizes = pseudo_sizes[keep]
    pseudo_keep_set = set(int(x) for x in pseudo_ids.tolist())
    if pseudo_keep_set:
        pseudo_valid = pseudo_valid & np.isin(pseudo, list(pseudo_keep_set))
    else:
        pseudo_valid = np.zeros_like(pseudo_valid)

    real_valid = real > 0
    real_ids, real_sizes = np.unique(real[real_valid], return_counts=True)

    pseudo_size_by_id = {int(i): int(s) for i, s in zip(pseudo_ids, pseudo_sizes)}
    real_size_by_id = {int(i): int(s) for i, s in zip(real_ids, real_sizes)}
    best_pseudo = {int(i): 0.0 for i in pseudo_ids}
    best_real = {int(i): 0.0 for i in real_ids}

    overlap = pseudo_valid & real_valid
    if overlap.any():
        pairs = np.stack([pseudo[overlap], real[overlap]], axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (pid_raw, rid_raw), inter_raw in zip(unique_pairs, counts):
            pid = int(pid_raw)
            rid = int(rid_raw)
            p_size = pseudo_size_by_id.get(pid)
            r_size = real_size_by_id.get(rid)
            if p_size is None or r_size is None:
                continue
            inter = int(inter_raw)
            union = p_size + r_size - inter
            iou = float(inter / union) if union > 0 else 0.0
            if iou > best_pseudo[pid]:
                best_pseudo[pid] = iou
            if iou > best_real[rid]:
                best_real[rid] = iou

    pseudo_best_values = np.asarray(list(best_pseudo.values()), dtype=np.float64)
    real_best_values = np.asarray(list(best_real.values()), dtype=np.float64)
    pseudo_points = int(pseudo_valid.sum())
    pseudo_on_real = int((pseudo_valid & real_valid).sum())
    num_pseudo = int(len(pseudo_ids))
    num_real = int(len(real_ids))
    query_count = int(query_count)

    out: dict[str, Any] = {
        "num_points": int(pseudo.shape[0]),
        "num_supervised_points": int(supervision.sum()),
        "num_pseudo_points": pseudo_points,
        "num_real_instance_points": int(real_valid.sum()),
        "num_pseudo_instances": num_pseudo,
        "num_real_instances": num_real,
        "query_count": query_count,
        "query_to_pseudo_ratio": float(query_count / max(num_pseudo, 1)),
        "query_to_real_ratio": float(query_count / max(num_real, 1)),
        "pseudo_query_deficit": int(max(num_pseudo - query_count, 0)),
        "real_query_deficit": int(max(num_real - query_count, 0)),
        "pseudo_query_recall_cap": float(min(query_count, num_pseudo) / max(num_pseudo, 1)),
        "real_query_recall_cap": float(min(query_count, num_real) / max(num_real, 1)),
        "pseudo_point_fraction_on_real_instances": float(pseudo_on_real / max(pseudo_points, 1)),
        "pseudo_point_fraction_non_real_or_ignored": float(1.0 - pseudo_on_real / max(pseudo_points, 1)),
        "pseudo_best_iou_mean": float(pseudo_best_values.mean()) if pseudo_best_values.size else None,
        "pseudo_best_iou_median": _percentile(pseudo_best_values, 50),
        "real_best_iou_mean": float(real_best_values.mean()) if real_best_values.size else None,
        "real_best_iou_median": _percentile(real_best_values, 50),
        "pseudo_fraction_best_iou_ge_25": float(np.mean(pseudo_best_values >= 0.25)) if pseudo_best_values.size else None,
        "pseudo_fraction_best_iou_ge_50": float(np.mean(pseudo_best_values >= 0.50)) if pseudo_best_values.size else None,
        "real_fraction_best_iou_ge_25": float(np.mean(real_best_values >= 0.25)) if real_best_values.size else None,
        "real_fraction_best_iou_ge_50": float(np.mean(real_best_values >= 0.50)) if real_best_values.size else None,
    }
    out.update(_size_stats(pseudo_sizes, "pseudo"))
    out.update(_size_stats(real_sizes, "real"))
    return out


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["granularity"]), str(row["benchmark"]))].append(row)
    out: list[dict[str, Any]] = []
    for (split, granularity, benchmark), items in sorted(grouped.items()):
        agg: dict[str, Any] = {
            "split": split,
            "granularity": granularity,
            "benchmark": benchmark,
            "num_scenes": len(items),
        }
        numeric_keys = sorted(
            key
            for key in {k for item in items for k in item}
            if all((item.get(key) is None or isinstance(item.get(key), (int, float))) for item in items)
        )
        for key in numeric_keys:
            agg[f"{key}_mean"] = _mean(item.get(key) for item in items)
        out.append(agg)
    return out


def _write_markdown(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pseudo-Label Audit",
        "",
        "This report is generated by `scripts/audit_pseudo_labels.py`.",
        "",
        "| Split | Granularity | Benchmark | Scenes | Pseudo inst. | Real inst. | Pseudo IoU>=0.25 | Pseudo IoU>=0.50 | Real covered>=0.25 | Real covered>=0.50 | Pseudo pts on real |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {split} | {granularity} | {benchmark} | {num_scenes} | {pseudo:.1f} | {real:.1f} | "
            "{p25:.3f} | {p50:.3f} | {r25:.3f} | {r50:.3f} | {pts:.3f} |".format(
                split=row["split"],
                granularity=row["granularity"],
                benchmark=row["benchmark"],
                num_scenes=row["num_scenes"],
                pseudo=float(row.get("num_pseudo_instances_mean") or 0.0),
                real=float(row.get("num_real_instances_mean") or 0.0),
                p25=float(row.get("pseudo_fraction_best_iou_ge_25_mean") or 0.0),
                p50=float(row.get("pseudo_fraction_best_iou_ge_50_mean") or 0.0),
                r25=float(row.get("real_fraction_best_iou_ge_25_mean") or 0.0),
                r50=float(row.get("real_fraction_best_iou_ge_50_mean") or 0.0),
                pts=float(row.get("pseudo_point_fraction_on_real_instances_mean") or 0.0),
            )
        )
    lines.extend(["", "See `summary.json`, `scene_rows.csv`, and `aggregate.csv` for full details.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from student.data.multi_scene_dataset import build_scene_list
    from student.data.training_pack import load_training_pack_scene_multi

    cfg = load_config(args.config)
    _maybe_apply_known_cluster_fallbacks(cfg)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    bb_cfg = model_cfg.get("backbone", {})
    available_grans = parse_granularities(data_cfg)
    granularities = _parse_granularities(args.granularities, available_grans)
    benchmarks = _parse_csv(args.benchmarks)
    splits = _parse_csv(args.splits)
    num_queries, _ = resolve_num_queries(model_cfg, bb_cfg)
    min_instance_points = int(data_cfg.get("min_instance_points", 10))
    scans_root = Path(data_cfg["scans_root"])

    split_files = {
        "train": _STUDENT_ROOT / data_cfg["train_split"],
        "val": _STUDENT_ROOT / data_cfg["val_split"],
    }
    scene_rows: list[dict[str, Any]] = []
    for split in splits:
        if split not in split_files:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(split_files)}")
        limit = args.max_train_scenes if split == "train" else args.max_val_scenes
        scene_dirs = build_scene_list(split_files[split], scans_root)
        if limit is not None:
            scene_dirs = scene_dirs[: max(int(limit), 0)]
        log.info("Auditing %d %s scene(s)", len(scene_dirs), split)
        for idx, scene_dir in enumerate(scene_dirs, start=1):
            pack = load_training_pack_scene_multi(scene_dir, granularities)
            log.info("[%s %d/%d] %s", split, idx, len(scene_dirs), pack.scene_id)
            for benchmark in benchmarks:
                real_gt = load_scannet_instance_ids(pack.scene_dir, pack.scene_id, benchmark)
                if real_gt.shape[0] != pack.num_points:
                    raise ValueError(f"{pack.scene_id}: real GT length {real_gt.shape[0]} != pack points {pack.num_points}")
                for granularity in granularities:
                    stats = audit_label_alignment(
                        pack.labels_by_granularity[granularity],
                        pack.supervision_mask,
                        real_gt,
                        query_count=num_queries,
                        min_instance_points=min_instance_points,
                    )
                    scene_rows.append(
                        {
                            "split": split,
                            "scene_id": pack.scene_id,
                            "scene_dir": str(pack.scene_dir),
                            "granularity": granularity,
                            "benchmark": benchmark,
                            **stats,
                        }
                    )
    aggregate_rows = _aggregate_rows(scene_rows)
    return {
        "config": str(Path(args.config)),
        "splits": splits,
        "granularities": list(granularities),
        "benchmarks": benchmarks,
        "query_count": int(num_queries),
        "min_instance_points": min_instance_points,
        "num_scene_rows": len(scene_rows),
        "aggregate": aggregate_rows,
        "scene_rows": scene_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pseudo-label alignment to real ScanNet instances")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--splits", default="train,val", type=str)
    parser.add_argument("--granularities", default=None, type=str)
    parser.add_argument("--benchmarks", default="scannet20,scannet200", type=str)
    parser.add_argument("--max-train-scenes", default=None, type=int)
    parser.add_argument("--max-val-scenes", default=None, type=int)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    _configure_logging()
    output_dir = Path(args.output_dir)
    result = run_audit(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", {k: v for k, v in result.items() if k != "scene_rows"})
    _write_json(output_dir / "scene_rows.json", result["scene_rows"])
    _write_csv(output_dir / "scene_rows.csv", result["scene_rows"])
    _write_csv(output_dir / "aggregate.csv", result["aggregate"])
    _write_markdown(output_dir / "README.md", result["aggregate"])
    log.info("Wrote pseudo-label audit to %s", output_dir)


if __name__ == "__main__":
    main()
