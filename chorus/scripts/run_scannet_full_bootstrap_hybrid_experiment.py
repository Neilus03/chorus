#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chorus.core.experiments.bootstrap_scannet import (  # noqa: E402
    BootstrapConfig,
    run_bootstrap_scene_experiment,
)
from chorus.core.experiments.hybrid_fusion import (  # noqa: E402
    HybridFusionConfig,
    run_hybrid_fusion_adapter_experiment,
)
from chorus.core.pipeline.project_cluster_stage import run_project_cluster_stage  # noqa: E402
from chorus.core.quality.diagnostics import save_json  # noqa: E402
from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics  # noqa: E402
from chorus.datasets.scannet.adapter import ScanNetSceneAdapter  # noqa: E402
from chorus.datasets.scannet.benchmark import (  # noqa: E402
    DEFAULT_SCANNET_EVAL_BENCHMARKS,
    parse_scannet_eval_benchmarks,
)
from chorus.datasets.scannet.prepare import is_rgbd_prepared  # noqa: E402
from chorus.export.training_pack import export_training_scene_pack  # noqa: E402
from chorus.export.visualization import save_labeled_mesh_ply  # noqa: E402
from chorus.orchestrators.streaming import read_scene_ids  # noqa: E402


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and obj != obj:
        return None
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


def _safe_get_bucket(summary: dict[str, Any], bucket: str, key: str) -> float | None:
    oracle = summary.get("oracle_summary") or {}
    results = oracle.get("oracle_results") or {}
    for bucket_name, metrics in results.items():
        if str(bucket_name).lower().startswith(bucket) and isinstance(metrics, dict):
            value = metrics.get(key)
            return None if value is None else float(value)
    return None


def _metric_block(summary: dict[str, Any]) -> dict[str, float | None]:
    oracle = summary.get("oracle_summary") or {}
    clustering = oracle.get("clustering_metrics") or {}
    out: dict[str, float | None] = {
        "nmi": clustering.get("NMI"),
        "ari": clustering.get("ARI"),
        "ap25_small": _safe_get_bucket(summary, "small", "AP25"),
        "ap50_small": _safe_get_bucket(summary, "small", "AP50"),
        "ap25_medium": _safe_get_bucket(summary, "medium", "AP25"),
        "ap50_medium": _safe_get_bucket(summary, "medium", "AP50"),
        "ap25_large": _safe_get_bucket(summary, "large", "AP25"),
        "ap50_large": _safe_get_bucket(summary, "large", "AP50"),
    }
    ap25 = [out[k] for k in ("ap25_small", "ap25_medium", "ap25_large") if out[k] is not None]
    ap50 = [out[k] for k in ("ap50_small", "ap50_medium", "ap50_large") if out[k] is not None]
    out["ap25_macro"] = sum(ap25) / len(ap25) if ap25 else None
    out["ap50_macro"] = sum(ap50) / len(ap50) if ap50 else None
    return out


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
    spec = importlib.util.spec_from_file_location("scannet_dl_paired_experiment", downloader_path)
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
    print(f"Downloading ScanNet .sens only: scene={scene_id} -> {scene_dir}")
    dl.download_scan(
        scan_id=scene_id,
        out_dir=str(scene_dir),
        file_types=[".sens"],
        use_v1_sens=True,
        skip_existing=skip_existing,
    )
    return True


def _scene_has_static_eval_assets(scene_dir: Path, scene_id: str) -> bool:
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
    return all(path.exists() for path in required) and any(p.exists() for p in seg_candidates) and any(
        p.exists() for p in agg_candidates
    )


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


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _clear_regular_outputs(scene_dir: Path) -> None:
    patterns = [
        "unsam_masks_g*",
        "chorus_instance_labels_g*.npy",
        "chorus_instance_result_g*.ply",
        "svd_features_g*.npy",
        "diagnostics_g*.json",
        "oracle_metrics*.json",
        "chorus_oracle_best_combined*.npy",
        "chorus_oracle_best_combined*.ply",
        "training_pack",
        "scene_pipeline_summary.json",
    ]
    for pattern in patterns:
        for path in scene_dir.glob(pattern):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def _prepare_isolated_scannet_scene(source_scene_dir: Path, dest_scene_dir: Path, *, overwrite: bool) -> None:
    dest_scene_dir.mkdir(parents=True, exist_ok=True)
    sid = source_scene_dir.name
    for name in ("color", "depth", "pose", "intrinsic"):
        _link_or_copy(source_scene_dir / name, dest_scene_dir / name)
    for suffix in (
        ".aggregation.json",
        ".txt",
        "_vh_clean.ply",
        "_vh_clean_2.0.010000.segs.json",
        "_vh_clean_2.ply",
        "_vh_clean.segs.json",
        "_vh_clean.aggregation.json",
        "_vh_clean_2.labels.ply",
    ):
        src = source_scene_dir / f"{sid}{suffix}"
        if src.exists():
            _link_or_copy(src, dest_scene_dir / src.name)
    if overwrite:
        _clear_regular_outputs(dest_scene_dir)


def _cleanup_paths(paths: list[Path]) -> dict[str, Any]:
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    for path in paths:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
                removed.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
        except Exception as exc:
            failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"removed": removed, "failed": failed}


def _cleanup_scene_rgbd_if_streamed(scene_dir: Path, scene_id: str, availability: dict[str, Any]) -> dict[str, Any]:
    if availability.get("had_rgbd_before"):
        return {"skipped": True, "reason": "scene had prepared RGB-D before this run"}
    paths = [
        scene_dir / "color",
        scene_dir / "depth",
        scene_dir / "pose",
        scene_dir / "intrinsic",
    ]
    if not availability.get("had_sens_before"):
        paths.append(scene_dir / f"{scene_id}.sens")
    return _cleanup_paths(paths)


def _cleanup_regular_intermediates(scene_dir: Path) -> dict[str, Any]:
    paths = [*scene_dir.glob("unsam_masks_g*"), *scene_dir.glob("svd_features_g*.npy")]
    return _cleanup_paths(paths)


def _cleanup_bootstrap_intermediates(scene_dir: Path) -> dict[str, Any]:
    paths = [scene_dir / "shared"]
    paths.extend(scene_dir.glob("bootstraps/b*/unsam_masks_g*"))
    return _cleanup_paths([p for p in paths if p.exists() or p.is_symlink()])


def _save_regular_cluster_artifacts(
    *,
    adapter: ScanNetSceneAdapter,
    cluster_output: Any,
) -> Any:
    labels_path = adapter.scene_root / f"chorus_instance_labels_g{cluster_output.granularity}.npy"
    ply_path = adapter.scene_root / f"chorus_instance_result_g{cluster_output.granularity}.ply"
    diagnostics_path = adapter.scene_root / f"diagnostics_g{cluster_output.granularity}.json"
    np.save(labels_path, cluster_output.labels)
    save_labeled_mesh_ply(
        source_ply_path=adapter.get_geometry_record().geometry_path,
        labels=cluster_output.labels,
        out_path=ply_path,
    )
    save_json(
        {
            "stats": cluster_output.stats,
            "intrinsic_metrics": cluster_output.stats.get("intrinsic_metrics"),
            "labels_path": str(labels_path),
            "ply_path": str(ply_path),
        },
        diagnostics_path,
    )
    return cluster_output.__class__(
        granularity=cluster_output.granularity,
        labels=cluster_output.labels,
        features=cluster_output.features,
        seen_mask=cluster_output.seen_mask,
        ply_path=ply_path,
        labels_path=labels_path,
        stats={**cluster_output.stats, "diagnostics_path": str(diagnostics_path)},
    )


def _run_timed_regular_scene(
    *,
    adapter: ScanNetSceneAdapter,
    teacher: Any,
    granularities: tuple[float, ...],
    frame_skip: int,
    svd_components: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    hdbscan_max_samples: int | None,
    run_oracle_eval: bool,
    export_training_pack_outputs: bool,
    cleanup_masks_after_granularity: bool,
) -> dict[str, Any]:
    scene_start = time.perf_counter()

    t0 = time.perf_counter()
    adapter.prepare()
    prepare_s = time.perf_counter() - t0

    cluster_outputs = []
    teacher_outputs = []
    per_granularity: dict[str, Any] = {}
    total_teacher_s = 0.0
    total_project_cluster_s = 0.0
    total_hdbscan_s = 0.0

    for granularity in granularities:
        t0 = time.perf_counter()
        teacher_output = teacher.run(
            adapter=adapter,
            granularity=float(granularity),
            frame_skip=frame_skip,
        )
        teacher_s = time.perf_counter() - t0
        total_teacher_s += teacher_s
        teacher_outputs.append(teacher_output)

        t0 = time.perf_counter()
        cluster_output = run_project_cluster_stage(
            adapter=adapter,
            teacher_output=teacher_output,
            frame_skip=frame_skip,
            svd_components=svd_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            save_outputs=False,
            hdbscan_max_samples=hdbscan_max_samples,
            hdbscan_subsample_seed=0,
        )
        project_cluster_s = time.perf_counter() - t0
        cluster_output = _save_regular_cluster_artifacts(adapter=adapter, cluster_output=cluster_output)
        hdbscan_s = float(cluster_output.stats.get("hdbscan_cluster_wall_seconds", 0.0))
        total_project_cluster_s += project_cluster_s
        total_hdbscan_s += hdbscan_s
        cluster_outputs.append(cluster_output)

        mask_dir = teacher_output.frame_mask_paths[0].parent if teacher_output.frame_mask_paths else None
        if cleanup_masks_after_granularity and mask_dir is not None:
            _cleanup_paths([mask_dir])

        g_key = f"g{teacher_output.granularity}"
        per_granularity[g_key] = {
            "granularity": float(teacher_output.granularity),
            "teacher_masks": int(teacher_output.total_masks),
            "teacher_seconds": float(teacher_s),
            "project_cluster_seconds": float(project_cluster_s),
            "hdbscan_seconds": float(hdbscan_s),
            "projection_svd_seconds_estimate": float(max(0.0, project_cluster_s - hdbscan_s)),
            "num_clusters": cluster_output.stats.get("num_clusters"),
            "num_seen_points": cluster_output.stats.get("num_seen_points"),
            "seen_points_fraction": cluster_output.stats.get("seen_points_fraction"),
            "labels_path": str(cluster_output.labels_path) if cluster_output.labels_path else None,
            "ply_path": str(cluster_output.ply_path) if cluster_output.ply_path else None,
        }

    scene_intrinsic_metrics = compute_scene_intrinsic_metrics(cluster_outputs)

    t0 = time.perf_counter()
    evaluation_summary = None
    if run_oracle_eval:
        evaluation_summary = adapter.get_evaluation_hooks().evaluate_scene(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
        )
    oracle_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    training_pack_dir = None
    if export_training_pack_outputs:
        clustering_backend = cluster_outputs[0].stats.get("hdbscan_backend") if cluster_outputs else None
        training_pack_dir = export_training_scene_pack(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
            teacher_name=teacher.__class__.__name__,
            projection_type="zbuffer_rgbd",
            embedding_type="truncated_svd",
            clustering_type="hdbscan_full",
            clustering_backend=clustering_backend,
            frame_skip=frame_skip,
            scene_intrinsic_metrics=scene_intrinsic_metrics,
        )
    training_pack_s = time.perf_counter() - t0

    timing = {
        "prepare": float(prepare_s),
        "teacher": float(total_teacher_s),
        "project_cluster": float(total_project_cluster_s),
        "hdbscan": float(total_hdbscan_s),
        "projection_svd_estimate": float(max(0.0, total_project_cluster_s - total_hdbscan_s)),
        "oracle_eval": float(oracle_s),
        "training_pack": float(training_pack_s),
        "total_wall": float(time.perf_counter() - scene_start),
    }
    summary: dict[str, Any] = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "method": "regular_chorus_isolated_full_hdbscan"
        if hdbscan_max_samples is None
        else "regular_chorus_isolated_capped_hdbscan",
        "granularities": [float(g) for g in granularities],
        "frame_skip": int(frame_skip),
        "clustering_type": "hdbscan_full" if hdbscan_max_samples is None else "hdbscan_subsample_cap",
        "hdbscan_max_samples": int(hdbscan_max_samples) if hdbscan_max_samples is not None else None,
        "teacher_outputs": [
            {
                "granularity": float(t.granularity),
                "num_mask_files": len(t.frame_mask_paths),
                "total_masks": int(t.total_masks),
                "output_dir": str(t.frame_mask_paths[0].parent) if t.frame_mask_paths else None,
            }
            for t in teacher_outputs
        ],
        "cluster_outputs": [
            {
                "granularity": float(c.granularity),
                "labels_path": str(c.labels_path) if c.labels_path else None,
                "ply_path": str(c.ply_path) if c.ply_path else None,
                "stats": c.stats,
            }
            for c in cluster_outputs
        ],
        "by_granularity": per_granularity,
        "scene_intrinsic_metrics": scene_intrinsic_metrics,
        "training_pack_dir": str(training_pack_dir) if training_pack_dir is not None else None,
        "timing_seconds": timing,
    }
    if evaluation_summary is not None:
        summary.update(evaluation_summary)
    summary_path = adapter.scene_root / "scene_pipeline_summary.json"
    save_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def _add_metric_prefix(row: dict[str, Any], prefix: str, summary: dict[str, Any]) -> None:
    metrics = _metric_block(summary)
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value
    timing = summary.get("timing_seconds") or {}
    for key in ("total_wall", "teacher", "hdbscan", "project_cluster", "projection_svd", "fusion_total"):
        if timing.get(key) is not None:
            row[f"{prefix}_{key}_seconds"] = float(timing[key])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "done"]
    aggregate: dict[str, Any] = {"num_done": len(done)}
    for method in ("full", "bootstrap", "hybrid"):
        for metric in ("ap25_macro", "ap50_macro", "nmi", "ari"):
            vals = [row.get(f"{method}_{metric}") for row in done if row.get(f"{method}_{metric}") is not None]
            aggregate[f"{method}_{metric}_mean"] = float(sum(map(float, vals)) / len(vals)) if vals else None
        vals = [row.get(f"{method}_total_wall_seconds") for row in done if row.get(f"{method}_total_wall_seconds") is not None]
        aggregate[f"{method}_total_wall_seconds_mean"] = float(sum(map(float, vals)) / len(vals)) if vals else None
    return aggregate


def _plot_method_bars(rows: list[dict[str, Any]], out_dir: Path) -> None:
    done = [row for row in rows if row.get("status") == "done"]
    if not done:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ("full", "bootstrap", "hybrid")
    colors = {"full": "#4c78a8", "bootstrap": "#f58518", "hybrid": "#54a24b"}
    scenes = [row["scene_id"] for row in done]
    for metric in ("ap25_macro", "ap50_macro", "nmi", "ari"):
        x = np.arange(len(done))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(done) * 0.9), 4.5), constrained_layout=True)
        for offset, method in zip((-width, 0.0, width), methods):
            vals = [row.get(f"{method}_{metric}") for row in done]
            ax.bar(x + offset, [np.nan if v is None else float(v) for v in vals], width, label=method, color=colors[method])
        ax.set_title(metric)
        ax.set_xticks(x, scenes, rotation=35, ha="right")
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.savefig(out_dir / f"metrics_{metric}.png", dpi=180)
        plt.close(fig)

    x = np.arange(len(done))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(done) * 0.9), 4.5), constrained_layout=True)
    for offset, method in zip((-width, 0.0, width), methods):
        vals = [row.get(f"{method}_total_wall_seconds") for row in done]
        ax.bar(x + offset, [(np.nan if v is None else float(v) / 60.0) for v in vals], width, label=method, color=colors[method])
    ax.set_title("runtime")
    ax.set_ylabel("minutes")
    ax.set_xticks(x, scenes, rotation=35, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.savefig(out_dir / "runtime_minutes.png", dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScanNet regular CHORUS, B4 bootstrap, and full+bootstrap hybrid in one isolated experiment."
    )
    parser.add_argument("--scans-root", type=Path, default=Path("/scratch2/nedela/chorus_poc/scans"))
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--use-release-list", action="store_true", default=False)
    parser.add_argument("--output-root", type=Path, default=Path("/scratch2/nedela/scannet_full_bootstrap_hybrid"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--granularities", type=_parse_float_tuple, default=(0.2, 0.5, 0.8))
    parser.add_argument("--regular-frame-skip", type=int, default=10)
    parser.add_argument("--bootstrap-frame-skip", type=int, default=10)
    parser.add_argument("--bootstrap-num-bootstraps", type=int, default=4)
    parser.add_argument("--bootstrap-frame-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-frame-sampling", choices=("disjoint", "all"), default="disjoint")
    parser.add_argument("--bootstrap-max-frames-per-bootstrap", type=int, default=30)
    parser.add_argument("--hdbscan-max-samples", type=int, default=50_000)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-iou-threshold", type=float, default=0.35)
    parser.add_argument("--min-fused-points", type=int, default=30)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--scannet-eval-benchmark",
        type=str,
        default=os.environ.get("CHORUS_SCANNET_EVAL_BENCHMARK", ",".join(DEFAULT_SCANNET_EVAL_BENCHMARKS)),
    )
    parser.add_argument("--device", type=str, default=os.environ.get("CHORUS_DEVICE", "cuda:0"))
    parser.add_argument("--debug-first-n-frames", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no-auto-download", action="store_true", default=False)
    parser.add_argument("--no-oracle-eval", action="store_true", default=False)
    parser.add_argument("--export-training-pack", action="store_true", default=False)
    parser.add_argument("--keep-intermediates", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from chorus.core.teacher.unsamv2 import UnSAMv2Teacher

    scans_root = args.scans_root.resolve()
    run_id = args.run_id or f"scannet_full_bootstrap_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root.resolve() / run_id
    regular_root = run_dir / "regular_scenes"
    bootstrap_root = run_dir / "bootstrap_scenes"
    hybrid_root = run_dir / "hybrid_scenes"
    plot_dir = run_dir / "plots"
    for d in (regular_root, bootstrap_root, hybrid_root, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    scene_ids = read_scene_ids(
        scans_root=scans_root,
        scene_list_file=args.scene_list_file,
        max_scenes=args.max_scenes,
        use_release_list=bool(args.use_release_list),
    )
    eval_benchmarks = tuple(parse_scannet_eval_benchmarks(args.scannet_eval_benchmark))
    granularities = tuple(float(g) for g in args.granularities)

    print(f"run_dir={run_dir}")
    print(f"scans_root={scans_root}")
    print(f"scene_ids={scene_ids}")
    print(f"eval_benchmarks={eval_benchmarks}")
    print(f"regular_frame_skip={args.regular_frame_skip}")
    print(f"bootstrap_max_frames_per_bootstrap={args.bootstrap_max_frames_per_bootstrap}")

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite,
    )

    bootstrap_config = BootstrapConfig(
        granularities=granularities,
        num_bootstraps=int(args.bootstrap_num_bootstraps),
        frame_fraction=float(args.bootstrap_frame_fraction),
        frame_skip=int(args.bootstrap_frame_skip),
        max_frames_per_bootstrap=args.bootstrap_max_frames_per_bootstrap,
        frame_sampling=str(args.bootstrap_frame_sampling),
        hdbscan_max_samples=int(args.hdbscan_max_samples),
        hdbscan_subsample_seed=0,
        support_threshold=float(args.support_threshold),
        cluster_iou_threshold=float(args.cluster_iou_threshold),
        min_fused_points=int(args.min_fused_points),
        svd_components=int(args.svd_components),
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        run_oracle_eval=not args.no_oracle_eval,
        export_bootstrap_ply=False,
        overwrite_teacher=bool(args.overwrite),
        eval_benchmarks=eval_benchmarks,
    )
    hybrid_config = HybridFusionConfig(granularities=granularities)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    run_summary_path = run_dir / "scannet_full_bootstrap_hybrid_summary.json"
    csv_path = run_dir / "scannet_full_bootstrap_hybrid_metrics.csv"

    for idx, scene_id in enumerate(scene_ids, start=1):
        print("\n" + "=" * 90)
        print(f"[{idx}/{len(scene_ids)}] scene={scene_id}")
        print("=" * 90)
        scene_start = time.perf_counter()
        row: dict[str, Any] = {
            "scene_id": scene_id,
            "status": "running",
            "source_scene_dir": str(scans_root / scene_id),
        }
        try:
            source_scene_dir = scans_root / scene_id
            if not _scene_has_static_eval_assets(source_scene_dir, scene_id):
                raise FileNotFoundError(
                    f"Missing static ScanNet mesh/GT assets in {source_scene_dir}; "
                    "this script downloads .sens only."
                )
            availability = _ensure_scannet_rgbd(
                scene_id,
                scans_root,
                auto_download=not args.no_auto_download,
            )
            row.update({f"availability_{k}": v for k, v in availability.items() if k != "scene_dir"})

            regular_scene_dir = regular_root / scene_id
            _prepare_isolated_scannet_scene(source_scene_dir, regular_scene_dir, overwrite=args.overwrite)
            regular_adapter = ScanNetSceneAdapter(scene_root=regular_scene_dir, eval_benchmarks=eval_benchmarks)
            regular_summary = _run_timed_regular_scene(
                adapter=regular_adapter,
                teacher=teacher,
                granularities=granularities,
                frame_skip=int(args.regular_frame_skip),
                svd_components=int(args.svd_components),
                min_cluster_size=int(args.min_cluster_size),
                min_samples=int(args.min_samples),
                cluster_selection_epsilon=float(args.cluster_selection_epsilon),
                hdbscan_max_samples=int(args.hdbscan_max_samples)
                if args.hdbscan_max_samples is not None
                else None,
                run_oracle_eval=not args.no_oracle_eval,
                export_training_pack_outputs=bool(args.export_training_pack),
                cleanup_masks_after_granularity=not args.keep_intermediates,
            )
            if not args.keep_intermediates:
                row["regular_cleanup"] = _cleanup_regular_intermediates(regular_scene_dir)

            bootstrap_summary = run_bootstrap_scene_experiment(
                scene_root=source_scene_dir,
                output_scene_dir=bootstrap_root / scene_id,
                teacher=teacher,
                config=bootstrap_config,
                frame_seed=0,
            )
            if not args.keep_intermediates:
                row["bootstrap_cleanup"] = _cleanup_bootstrap_intermediates(bootstrap_root / scene_id)

            hybrid_adapter = ScanNetSceneAdapter(scene_root=source_scene_dir, eval_benchmarks=eval_benchmarks)
            hybrid_summary = run_hybrid_fusion_adapter_experiment(
                adapter=hybrid_adapter,
                full_scene_dir=regular_scene_dir,
                bootstrap_scene_dir=bootstrap_root / scene_id,
                output_scene_dir=hybrid_root / scene_id,
                config=hybrid_config,
                run_oracle_eval=not args.no_oracle_eval,
                eval_benchmarks=eval_benchmarks,
            )
            if not args.export_training_pack and not args.keep_intermediates:
                _cleanup_paths(
                    [
                        bootstrap_root / scene_id / "fused" / "training_pack",
                        hybrid_root / scene_id / "hybrid" / "training_pack",
                    ]
                )

            row.update(
                {
                    "status": "done",
                    "regular_summary_path": regular_summary.get("summary_path"),
                    "bootstrap_summary_path": bootstrap_summary.get("summary_path"),
                    "hybrid_summary_path": hybrid_summary.get("summary_path"),
                    "duration_seconds": float(time.perf_counter() - scene_start),
                }
            )
            _add_metric_prefix(row, "full", regular_summary)
            _add_metric_prefix(row, "bootstrap", bootstrap_summary)
            _add_metric_prefix(row, "hybrid", hybrid_summary)
            if not args.keep_intermediates:
                row["source_rgbd_cleanup"] = _cleanup_scene_rgbd_if_streamed(source_scene_dir, scene_id, availability)

            print(
                "done "
                f"full_ap50={row.get('full_ap50_macro')} "
                f"bootstrap_ap50={row.get('bootstrap_ap50_macro')} "
                f"hybrid_ap50={row.get('hybrid_ap50_macro')} "
                f"duration_min={row['duration_seconds'] / 60:.2f}"
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "duration_seconds": float(time.perf_counter() - scene_start),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(row)
            print(f"failed scene={scene_id}: {row['error']}")
            if not args.continue_on_error:
                rows.append(row)
                break
        rows.append(row)
        aggregate = _aggregate_rows(rows)
        save_json(
            _json_safe(
                {
                    "experiment": "scannet_full_bootstrap_hybrid",
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "scans_root": str(scans_root),
                    "scene_ids": scene_ids,
                    "config": {
                        "granularities": [float(g) for g in granularities],
                        "regular_frame_skip": int(args.regular_frame_skip),
                        "bootstrap_config": bootstrap_config.__dict__,
                        "hybrid_config": hybrid_config.__dict__,
                        "eval_benchmarks": list(eval_benchmarks),
                    },
                    "done": len([r for r in rows if r.get("status") == "done"]),
                    "failed": len([r for r in rows if r.get("status") == "failed"]),
                    "failures": failures,
                    "aggregate": aggregate,
                    "rows": rows,
                    "elapsed_seconds": float(time.perf_counter() - run_start),
                }
            ),
            run_summary_path,
        )
        _write_csv(csv_path, rows)

    _plot_method_bars(rows, plot_dir)
    final = _load_json(run_summary_path) or {}
    final["finished_at"] = datetime.now().isoformat()
    final["elapsed_seconds"] = float(time.perf_counter() - run_start)
    final["aggregate"] = _aggregate_rows(rows)
    final["plots"] = {
        "ap25": str(plot_dir / "metrics_ap25_macro.png"),
        "ap50": str(plot_dir / "metrics_ap50_macro.png"),
        "nmi": str(plot_dir / "metrics_nmi.png"),
        "ari": str(plot_dir / "metrics_ari.png"),
        "runtime": str(plot_dir / "runtime_minutes.png"),
    }
    save_json(_json_safe(final), run_summary_path)
    _write_csv(csv_path, rows)
    print(f"\nSaved summary: {run_summary_path}")
    print(f"Saved CSV: {csv_path}")
    print(json.dumps(_json_safe(final.get("aggregate", {})), indent=2))


if __name__ == "__main__":
    main()
