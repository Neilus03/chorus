#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.core.experiments.bootstrap_scannet import BootstrapConfig, run_bootstrap_scene_experiment
from chorus.core.experiments.hybrid_fusion import make_cluster_output_from_labels
from chorus.core.experiments.quality_gating import (
    connected_component_cleanup,
    instance_sizes,
    intersect_kept_full_labels,
    proposal_counterbalance_metrics,
    prune_by_bootstrap_agreement,
    prune_by_multigranularity_agreement,
)
from chorus.core.quality.diagnostics import save_json
from chorus.datasets.scannet.adapter import ScanNetSceneAdapter
from chorus.datasets.scannet.benchmark import DEFAULT_SCANNET_EVAL_BENCHMARKS, parse_scannet_eval_benchmarks
from chorus.datasets.scannet.gt import load_scannet_gt_instance_ids
from chorus.datasets.scannet.prepare import is_rgbd_prepared
from chorus.eval.scannet_oracle import evaluate_and_save_scannet_oracle
from chorus.export.visualization import save_labeled_mesh_ply


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return obj


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_downloader_path() -> Path:
    user = os.environ.get("USER", "nedela")
    candidates = [
        os.environ.get("CHORUS_SCANNET_DOWNLOADER", ""),
        f"/scratch2/{user}/chorus/tools/download-scannet.py",
        f"/scratch2/{user}/chorus_poc/tools/download-scannet.py",
        "/scratch2/nedela/chorus_poc/tools/download-scannet.py",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()
        if path.exists():
            return path
    raise FileNotFoundError("Could not find download-scannet.py; set CHORUS_SCANNET_DOWNLOADER.")


def _load_scannet_downloader_module() -> Any:
    downloader_path = _resolve_downloader_path()
    spec = importlib.util.spec_from_file_location("scannet_dl_bootstrap_quality_gate", downloader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ScanNet downloader from {downloader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download_sens_only(scene_id: str, scene_dir: Path, *, skip_existing: bool = True) -> bool:
    sens_path = scene_dir / f"{scene_id}.sens"
    if skip_existing and sens_path.exists():
        return False
    dl = _load_scannet_downloader_module()
    print(f"Downloading ScanNet .sens only: scene={scene_id} -> {scene_dir}", flush=True)
    dl.download_scan(
        scan_id=scene_id,
        out_dir=str(scene_dir),
        file_types=[".sens"],
        use_v1_sens=True,
        skip_existing=skip_existing,
    )
    return True


def _ensure_scannet_rgbd(scene_id: str, scans_root: Path, *, auto_download: bool) -> dict[str, Any]:
    scene_dir = scans_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    had_rgbd = is_rgbd_prepared(scene_dir)
    had_sens = (scene_dir / f"{scene_id}.sens").exists()
    downloaded = False
    if not had_rgbd and not had_sens:
        if not auto_download:
            raise FileNotFoundError(f"Scene {scene_id} has no prepared RGB-D and no .sens file.")
        downloaded = _download_sens_only(scene_id, scene_dir, skip_existing=True)
    adapter = ScanNetSceneAdapter(scene_root=scene_dir)
    adapter.prepare()
    return {
        "scene_dir": scene_dir,
        "had_rgbd_before": bool(had_rgbd),
        "had_sens_before": bool(had_sens),
        "downloaded_sens": bool(downloaded),
    }


def _cleanup_streamed_scene(scene_dir: Path, scene_id: str, availability: dict[str, Any]) -> dict[str, Any]:
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    if not availability.get("had_rgbd_before"):
        for name in ("color", "depth", "pose", "intrinsic"):
            path = scene_dir / name
            if path.exists() or path.is_symlink():
                try:
                    shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
                    removed.append(str(path))
                except Exception as exc:
                    failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if availability.get("downloaded_sens") and not availability.get("had_sens_before"):
        path = scene_dir / f"{scene_id}.sens"
        if path.exists():
            try:
                path.unlink()
                removed.append(str(path))
            except Exception as exc:
                failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"removed": removed, "failed": failed}


def _cleanup_bootstrap_intermediates(scene_dir: Path, *, keep_intermediates: bool) -> dict[str, Any]:
    if keep_intermediates:
        return {"removed": [], "failed": []}
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    patterns = [
        "bootstraps/b*/unsam_masks_g*",
        "fused/training_pack",
    ]
    for pattern in patterns:
        for path in scene_dir.glob(pattern):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(str(path))
            except Exception as exc:
                failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"removed": removed, "failed": failed}


def _scene_has_full_labels(scene_dir: Path, granularities: tuple[float, ...]) -> bool:
    return all((scene_dir / f"chorus_instance_labels_g{g}.npy").exists() for g in granularities)


def _scene_has_eval_assets(scene_dir: Path, scene_id: str) -> bool:
    required = [
        scene_dir / f"{scene_id}_vh_clean_2.ply",
        scene_dir / f"{scene_id}_vh_clean_2.labels.ply",
    ]
    seg_candidates = [
        scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_id}_vh_clean.segs.json",
    ]
    agg_candidates = [
        scene_dir / f"{scene_id}.aggregation.json",
        scene_dir / f"{scene_id}_vh_clean.aggregation.json",
    ]
    return all(p.exists() for p in required) and any(p.exists() for p in seg_candidates) and any(
        p.exists() for p in agg_candidates
    )


def _summary_frame_count(scene_dir: Path) -> int | None:
    summary = _load_json(scene_dir / "scene_pipeline_summary.json")
    if not summary:
        return None
    teacher_outputs = summary.get("teacher_outputs") or []
    if teacher_outputs and isinstance(teacher_outputs[0], dict):
        value = teacher_outputs[0].get("num_mask_files")
        if value is not None:
            return int(value)
    cluster_outputs = summary.get("cluster_outputs") or []
    if cluster_outputs and isinstance(cluster_outputs[0], dict):
        stats = cluster_outputs[0].get("stats") or {}
        value = stats.get("used_frames")
        if value is not None:
            return int(value)
    return None


def _select_scene_ids(
    *,
    scans_root: Path,
    scene_list_file: Path | None,
    max_scenes: int,
    granularities: tuple[float, ...],
) -> list[str]:
    if scene_list_file is not None:
        scene_ids = [
            line.strip()
            for line in scene_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return scene_ids[:max_scenes]

    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    for scene_dir in scans_root.glob("scene*"):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        if not _scene_has_full_labels(scene_dir, granularities):
            continue
        if not _scene_has_eval_assets(scene_dir, scene_id):
            continue
        frame_count = _summary_frame_count(scene_dir)
        if frame_count is None:
            fallback.append(scene_id)
        else:
            candidates.append((int(frame_count), scene_id))
    scene_ids = [sid for _, sid in sorted(candidates)] + sorted(fallback)
    return scene_ids[:max_scenes]


def _has_bootstrap_fused_labels(scene_dir: Path, granularities: tuple[float, ...]) -> bool:
    return all((scene_dir / "fused" / f"labels_g{g}.npy").exists() for g in granularities)


def _resolve_bootstrap_scene_dir(
    *,
    scene_id: str,
    run_bootstrap_root: Path,
    reuse_bootstrap_root: Path | None,
    granularities: tuple[float, ...],
) -> tuple[Path, bool]:
    if reuse_bootstrap_root is not None:
        candidate = reuse_bootstrap_root / scene_id
        if _has_bootstrap_fused_labels(candidate, granularities):
            return candidate, True
    return run_bootstrap_root / scene_id, False


def _macro_metric(results: dict[str, Any], metric: str) -> float | None:
    vals: list[float] = []
    for bucket, row in results.items():
        if str(bucket).startswith("_") or not isinstance(row, dict):
            continue
        value = row.get(metric)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else None


def _labels_to_outputs(labels_by_g: dict[float, np.ndarray], stats_by_g: dict[float, dict[str, Any]]):
    return [
        make_cluster_output_from_labels(
            granularity=float(g),
            labels=labels_by_g[float(g)],
            stats=stats_by_g.get(float(g), {}),
        )
        for g in sorted(labels_by_g)
    ]


def _evaluate_method(
    *,
    adapter: ScanNetSceneAdapter,
    scene_id: str,
    method: str,
    labels_by_g: dict[float, np.ndarray],
    stats_by_g: dict[float, dict[str, Any]],
    eval_benchmarks: tuple[str, ...],
) -> list[dict[str, Any]]:
    outputs = _labels_to_outputs(labels_by_g, stats_by_g)
    rows: list[dict[str, Any]] = []
    for benchmark in eval_benchmarks:
        oracle = evaluate_and_save_scannet_oracle(
            adapter=adapter,
            cluster_outputs=outputs,
            eval_benchmark=benchmark,
            save_artifacts=False,
            gt_scene_root=adapter.scene_root,
        )
        gt_ids = load_scannet_gt_instance_ids(adapter.scene_root, scene_id, eval_benchmark=benchmark)
        counter = proposal_counterbalance_metrics([labels_by_g[g] for g in sorted(labels_by_g)], gt_ids)
        clustering = oracle.get("clustering_metrics") or {}
        rows.append(
            {
                "scene_id": scene_id,
                "method": method,
                "benchmark": benchmark,
                "ap25_macro": _macro_metric(oracle.get("oracle_results") or {}, "AP25"),
                "ap50_macro": _macro_metric(oracle.get("oracle_results") or {}, "AP50"),
                "nmi": clustering.get("NMI"),
                "ari": clustering.get("ARI"),
                **counter,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups = sorted({(row["method"], row["benchmark"]) for row in rows})
    metrics = [
        "ap25_macro",
        "ap50_macro",
        "proposal_precision@25",
        "gt_recall@25",
        "proposal_f1@25",
        "proposal_precision@50",
        "gt_recall@50",
        "proposal_f1@50",
        "num_proposals",
        "mean_best_iou_per_proposal",
        "mean_best_iou_per_gt",
        "avg_good_proposals_per_matched_gt@50",
        "nmi",
        "ari",
    ]
    for method, benchmark in groups:
        group = [row for row in rows if row["method"] == method and row["benchmark"] == benchmark]
        item: dict[str, Any] = {"num_rows": len(group)}
        for metric in metrics:
            vals: list[float] = []
            for row in group:
                value = row.get(metric)
                if value is None or value == "":
                    continue
                value = float(value)
                if math.isfinite(value):
                    vals.append(value)
            item[f"{metric}_mean"] = sum(vals) / len(vals) if vals else None
            item[f"{metric}_n"] = len(vals)
        summary[f"{method}__{benchmark}"] = item
    return summary


def _load_full_labels(scene_root: Path, granularities: tuple[float, ...]) -> dict[float, np.ndarray]:
    return {
        float(g): np.load(scene_root / f"chorus_instance_labels_g{g}.npy").astype(np.int32, copy=False)
        for g in granularities
    }


def _load_bootstrap_labels(bootstrap_scene_dir: Path, granularities: tuple[float, ...]) -> dict[float, np.ndarray]:
    return {
        float(g): np.load(bootstrap_scene_dir / "fused" / f"labels_g{g}.npy").astype(np.int32, copy=False)
        for g in granularities
    }


def _apply_quality_gates(
    *,
    scene_root: Path,
    scene_id: str,
    full_by_g: dict[float, np.ndarray],
    bootstrap_by_g: dict[float, np.ndarray],
    output_scene_dir: Path,
    granularities: tuple[float, ...],
    min_points: int,
    save_ply: bool,
) -> tuple[dict[str, dict[float, np.ndarray]], dict[str, dict[float, dict[str, Any]]], list[dict[str, Any]]]:
    labels_by_method: dict[str, dict[float, np.ndarray]] = {
        "full": full_by_g,
        "bootstrap_gate": {},
        "bootstrap_multig_cc": {},
    }
    stats_by_method: dict[str, dict[float, dict[str, Any]]] = {
        "full": {float(g): {"gate": "none"} for g in granularities},
        "bootstrap_gate": {},
        "bootstrap_multig_cc": {},
    }
    stat_rows: list[dict[str, Any]] = []
    geometry_path = scene_root / f"{scene_id}_vh_clean_2.ply"

    for granularity in granularities:
        g = float(granularity)
        boot_gate, boot_stats = prune_by_bootstrap_agreement(
            full_by_g[g],
            bootstrap_by_g[g],
            min_points=min_points,
            iou_threshold=0.20,
            full_support_threshold=0.15,
            bootstrap_containment_threshold=0.50,
        )
        multig_gate, multig_stats = prune_by_multigranularity_agreement(
            full_by_g,
            g,
            min_points=min_points,
            iou_threshold=0.20,
            full_support_threshold=0.30,
            other_containment_threshold=0.30,
        )
        combined = intersect_kept_full_labels(
            full_by_g[g],
            boot_gate,
            multig_gate,
            min_points=min_points,
        )
        cc, cc_stats = connected_component_cleanup(
            combined,
            geometry_path,
            min_component_points=max(30, min_points),
            min_component_fraction=0.08,
        )

        labels_by_method["bootstrap_gate"][g] = boot_gate
        labels_by_method["bootstrap_multig_cc"][g] = cc
        stats_by_method["bootstrap_gate"][g] = boot_stats
        stats_by_method["bootstrap_multig_cc"][g] = {
            "gate": "bootstrap_and_multigranularity_plus_connected_components",
            "bootstrap": boot_stats,
            "multigranularity": multig_stats,
            "connected_components": cc_stats,
            "num_kept_instances": int(len(instance_sizes(cc))),
            "labeled_fraction": float(np.count_nonzero(cc >= 0) / max(cc.shape[0], 1)),
        }

    for method, labels_by_g in labels_by_method.items():
        for granularity, labels in labels_by_g.items():
            method_dir = output_scene_dir / method
            labels_path = method_dir / f"labels_g{granularity}.npy"
            diagnostics_path = method_dir / f"diagnostics_g{granularity}.json"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels.astype(np.int32, copy=False))
            save_json(_json_safe(stats_by_method[method][granularity]), diagnostics_path)
            if save_ply:
                save_labeled_mesh_ply(
                    source_ply_path=geometry_path,
                    labels=labels,
                    out_path=method_dir / f"result_g{granularity}.ply",
                )
            stat_rows.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "granularity": float(granularity),
                    "num_instances": int(len(instance_sizes(labels))),
                    "labeled_fraction": float(np.count_nonzero(labels >= 0) / max(labels.shape[0], 1)),
                    "labels_path": str(labels_path),
                    "diagnostics_path": str(diagnostics_path),
                }
            )

    return labels_by_method, stats_by_method, stat_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bootstrap validation and quality-gated pruning for ScanNet scenes using existing full "
            "CHORUS labels."
        )
    )
    parser.add_argument("--scans-root", type=Path, default=Path("/scratch2/nedela/chorus_poc/scans"))
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch2/nedela/scannet_quality_gate_bootstrap"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--reuse-bootstrap-root",
        type=Path,
        default=None,
        help="Optional root with <scene_id>/fused/labels_g*.npy from a previous bootstrap run.",
    )
    parser.add_argument("--granularities", type=_parse_float_tuple, default=(0.2, 0.5, 0.8))
    parser.add_argument("--bootstrap-num-bootstraps", type=int, default=4)
    parser.add_argument("--bootstrap-frame-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-frame-skip", type=int, default=10)
    parser.add_argument("--bootstrap-frame-sampling", choices=("disjoint", "all"), default="disjoint")
    parser.add_argument("--bootstrap-max-frames-per-bootstrap", type=int, default=30)
    parser.add_argument("--hdbscan-max-samples", type=int, default=50_000)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-iou-threshold", type=float, default=0.35)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument(
        "--scannet-eval-benchmark",
        type=str,
        default=os.environ.get("CHORUS_SCANNET_EVAL_BENCHMARK", ",".join(DEFAULT_SCANNET_EVAL_BENCHMARKS)),
    )
    parser.add_argument("--device", type=str, default=os.environ.get("CHORUS_DEVICE", "cuda:0"))
    parser.add_argument("--debug-first-n-frames", type=int, default=3)
    parser.add_argument("--no-auto-download", action="store_true", default=False)
    parser.add_argument("--keep-intermediates", action="store_true", default=False)
    parser.add_argument("--no-save-ply", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from chorus.core.teacher.unsamv2 import UnSAMv2Teacher

    scans_root = args.scans_root.resolve()
    granularities = tuple(float(g) for g in args.granularities)
    eval_benchmarks = tuple(parse_scannet_eval_benchmarks(args.scannet_eval_benchmark))
    run_id = args.run_id or f"scannet_quality_gate_bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root.resolve() / run_id
    bootstrap_root = run_dir / "bootstrap_scenes"
    gated_root = run_dir / "gated_scenes"
    for path in (bootstrap_root, gated_root):
        path.mkdir(parents=True, exist_ok=True)

    scene_ids = _select_scene_ids(
        scans_root=scans_root,
        scene_list_file=args.scene_list_file,
        max_scenes=int(args.max_scenes),
        granularities=granularities,
    )
    bootstrap_config = BootstrapConfig(
        granularities=granularities,
        num_bootstraps=int(args.bootstrap_num_bootstraps),
        frame_fraction=float(args.bootstrap_frame_fraction),
        frame_skip=int(args.bootstrap_frame_skip),
        max_frames_per_bootstrap=int(args.bootstrap_max_frames_per_bootstrap)
        if args.bootstrap_max_frames_per_bootstrap is not None
        else None,
        frame_sampling=str(args.bootstrap_frame_sampling),
        hdbscan_max_samples=int(args.hdbscan_max_samples),
        support_threshold=float(args.support_threshold),
        cluster_iou_threshold=float(args.cluster_iou_threshold),
        min_fused_points=int(args.min_points),
        svd_components=int(args.svd_components),
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        run_oracle_eval=False,
        export_bootstrap_ply=False,
        overwrite_teacher=False,
        eval_benchmarks=eval_benchmarks,
    )

    print(f"run_dir={run_dir}", flush=True)
    print(f"scans_root={scans_root}", flush=True)
    print(f"num_scenes={len(scene_ids)}", flush=True)
    print(f"scene_ids={scene_ids}", flush=True)
    print(f"granularities={granularities}", flush=True)
    print(f"eval_benchmarks={eval_benchmarks}", flush=True)
    print(f"reuse_bootstrap_root={args.reuse_bootstrap_root}", flush=True)

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=False,
    )

    metric_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()

    for idx, scene_id in enumerate(scene_ids, start=1):
        print("\n" + "=" * 90, flush=True)
        print(f"[{idx}/{len(scene_ids)}] scene={scene_id}", flush=True)
        print("=" * 90, flush=True)
        scene_start = time.perf_counter()
        scene_root = scans_root / scene_id
        availability: dict[str, Any] | None = None
        cleanup: dict[str, Any] | None = None
        try:
            if not _scene_has_full_labels(scene_root, granularities):
                raise FileNotFoundError(f"Missing full CHORUS labels in {scene_root}")

            bootstrap_scene_dir, reused_bootstrap = _resolve_bootstrap_scene_dir(
                scene_id=scene_id,
                run_bootstrap_root=bootstrap_root,
                reuse_bootstrap_root=args.reuse_bootstrap_root.resolve()
                if args.reuse_bootstrap_root is not None
                else None,
                granularities=granularities,
            )
            if reused_bootstrap:
                print(f"Reusing bootstrap labels: {bootstrap_scene_dir}", flush=True)
            else:
                availability = _ensure_scannet_rgbd(
                    scene_id,
                    scans_root,
                    auto_download=not args.no_auto_download,
                )
                bootstrap_summary = run_bootstrap_scene_experiment(
                    scene_root=scene_root,
                    output_scene_dir=bootstrap_scene_dir,
                    teacher=teacher,
                    config=bootstrap_config,
                    frame_seed=0,
                )
                print(f"Bootstrap summary: {bootstrap_summary.get('summary_path')}", flush=True)
                _cleanup_bootstrap_intermediates(
                    bootstrap_scene_dir,
                    keep_intermediates=bool(args.keep_intermediates),
                )

            adapter = ScanNetSceneAdapter(scene_root=scene_root, eval_benchmarks=eval_benchmarks)
            full_by_g = _load_full_labels(scene_root, granularities)
            bootstrap_by_g = _load_bootstrap_labels(bootstrap_scene_dir, granularities)
            labels_by_method, stats_by_method, scene_stat_rows = _apply_quality_gates(
                scene_root=scene_root,
                scene_id=scene_id,
                full_by_g=full_by_g,
                bootstrap_by_g=bootstrap_by_g,
                output_scene_dir=gated_root / scene_id,
                granularities=granularities,
                min_points=int(args.min_points),
                save_ply=not args.no_save_ply,
            )
            stat_rows.extend(scene_stat_rows)
            for method in ("full", "bootstrap_gate", "bootstrap_multig_cc"):
                rows = _evaluate_method(
                    adapter=adapter,
                    scene_id=scene_id,
                    method=method,
                    labels_by_g=labels_by_method[method],
                    stats_by_g=stats_by_method[method],
                    eval_benchmarks=eval_benchmarks,
                )
                metric_rows.extend(rows)
                primary = rows[0]
                print(
                    f"{method}: AP25={primary['ap25_macro']} AP50={primary['ap50_macro']} "
                    f"P50={primary['proposal_precision@50']} F50={primary['proposal_f1@50']} "
                    f"num_props={primary['num_proposals']}",
                    flush=True,
                )

            if availability is not None:
                cleanup = _cleanup_streamed_scene(scene_root, scene_id, availability)
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "status": "done",
                    "duration_seconds": float(time.perf_counter() - scene_start),
                    "reused_bootstrap": bool(reused_bootstrap),
                    "bootstrap_scene_dir": str(bootstrap_scene_dir),
                    "gated_scene_dir": str(gated_root / scene_id),
                    "availability": availability,
                    "cleanup": cleanup,
                }
            )
        except Exception as exc:
            if availability is not None:
                cleanup = _cleanup_streamed_scene(scene_root, scene_id, availability)
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "status": "failed",
                    "duration_seconds": float(time.perf_counter() - scene_start),
                    "error": f"{type(exc).__name__}: {exc}",
                    "availability": availability,
                    "cleanup": cleanup,
                }
            )
            print(f"FAILED {scene_id}: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                break

        _write_csv(run_dir / "quality_gate_metrics.csv", metric_rows)
        _write_csv(run_dir / "quality_gate_label_stats.csv", stat_rows)
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "scans_root": str(scans_root),
            "scene_ids": scene_ids,
            "scene_results": scene_rows,
            "done": sum(1 for row in scene_rows if row["status"] == "done"),
            "failed": sum(1 for row in scene_rows if row["status"] == "failed"),
            "elapsed_seconds": float(time.perf_counter() - started_run),
            "config": {
                "granularities": list(granularities),
                "eval_benchmarks": list(eval_benchmarks),
                "bootstrap": bootstrap_config.__dict__,
                "min_points": int(args.min_points),
                "save_ply": not args.no_save_ply,
                "reuse_bootstrap_root": str(args.reuse_bootstrap_root)
                if args.reuse_bootstrap_root is not None
                else None,
            },
            "aggregates": _summarize(metric_rows),
        }
        save_json(_json_safe(summary), run_dir / "quality_gate_summary.json")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scans_root": str(scans_root),
        "scene_ids": scene_ids,
        "scene_results": scene_rows,
        "done": sum(1 for row in scene_rows if row["status"] == "done"),
        "failed": sum(1 for row in scene_rows if row["status"] == "failed"),
        "elapsed_seconds": float(time.perf_counter() - started_run),
        "config": {
            "granularities": list(granularities),
            "eval_benchmarks": list(eval_benchmarks),
            "bootstrap": bootstrap_config.__dict__,
            "min_points": int(args.min_points),
            "save_ply": not args.no_save_ply,
            "reuse_bootstrap_root": str(args.reuse_bootstrap_root)
            if args.reuse_bootstrap_root is not None
            else None,
        },
        "aggregates": _summarize(metric_rows),
    }
    save_json(_json_safe(summary), run_dir / "quality_gate_summary.json")
    print("\nSaved:", flush=True)
    print(run_dir / "quality_gate_metrics.csv", flush=True)
    print(run_dir / "quality_gate_label_stats.csv", flush=True)
    print(run_dir / "quality_gate_summary.json", flush=True)
    print(json.dumps(_json_safe(summary["aggregates"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
