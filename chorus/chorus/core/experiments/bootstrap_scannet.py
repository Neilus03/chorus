from __future__ import annotations

import math
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from chorus.common.types import ClusterOutput, FrameRecord, TeacherOutput
from chorus.core.quality.diagnostics import save_json
from chorus.export.training_pack import export_training_scene_pack

if TYPE_CHECKING:
    from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
    from chorus.datasets.scannet.adapter import ScanNetSceneAdapter
    from chorus.datasets.base import SceneAdapter


@dataclass(frozen=True)
class BootstrapConfig:
    granularities: tuple[float, ...] = (0.2, 0.5, 0.8)
    num_bootstraps: int = 4
    frame_fraction: float = 0.25
    frame_skip: int = 10
    max_frames_per_bootstrap: int | None = None
    frame_sampling: str = "disjoint"
    hdbscan_max_samples: int = 75_000
    hdbscan_subsample_seed: int = 0
    support_threshold: float = 0.5
    cluster_iou_threshold: float = 0.35
    min_fused_points: int = 30
    svd_components: int = 32
    min_cluster_size: int = 100
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.1
    run_oracle_eval: bool = True
    export_bootstrap_ply: bool = False
    overwrite_teacher: bool = False
    eval_benchmarks: tuple[str, ...] = ("scannet20", "scannet200")


def split_frames_for_bootstraps(
    frames: Sequence[FrameRecord],
    *,
    num_bootstraps: int,
    frame_fraction: float,
    frame_sampling: str = "disjoint",
    seed: int = 0,
    max_frames_per_bootstrap: int | None = None,
) -> list[list[FrameRecord]]:
    if num_bootstraps <= 0:
        raise ValueError("num_bootstraps must be positive")
    if not (0.0 < float(frame_fraction) <= 1.0):
        raise ValueError(f"frame_fraction must be in (0, 1], got {frame_fraction}")
    if frame_sampling not in {"disjoint", "all"}:
        raise ValueError(
            f"Unsupported frame_sampling={frame_sampling!r}; expected one of: 'disjoint', 'all'"
        )

    frames = list(frames)
    if not frames:
        return [[] for _ in range(num_bootstraps)]
    if frame_sampling == "all":
        return [list(frames) for _ in range(num_bootstraps)]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames))
    target = max(1, int(math.ceil(len(frames) * float(frame_fraction))))
    if max_frames_per_bootstrap is not None:
        target = min(target, max(1, int(max_frames_per_bootstrap)))
    subsets: list[list[FrameRecord]] = []

    for boot_idx in range(num_bootstraps):
        start = boot_idx * target
        stop = min(start + target, len(order))
        if start >= len(order):
            chosen = np.array([], dtype=np.int64)
        else:
            chosen = order[start:stop]
        chosen = np.sort(chosen)
        subsets.append([frames[int(i)] for i in chosen])

    return subsets


def _unique_instance_masks(labels: np.ndarray) -> list[np.ndarray]:
    ids = np.unique(labels)
    ids = ids[ids >= 0]
    return [labels == int(inst_id) for inst_id in ids]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter / max(union, 1))


def fuse_stable_core_labels(
    bootstrap_labels: Sequence[np.ndarray],
    *,
    support_threshold: float,
    cluster_iou_threshold: float,
    min_fused_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bootstrap_labels:
        raise ValueError("bootstrap_labels must be non-empty")
    if not (0.0 < float(support_threshold) <= 1.0):
        raise ValueError(f"support_threshold must be in (0, 1], got {support_threshold}")
    if not (0.0 <= float(cluster_iou_threshold) <= 1.0):
        raise ValueError(f"cluster_iou_threshold must be in [0, 1], got {cluster_iou_threshold}")

    labels0 = np.asarray(bootstrap_labels[0])
    num_points = int(labels0.shape[0])
    for labels in bootstrap_labels:
        if np.asarray(labels).shape != labels0.shape:
            raise ValueError("all bootstrap label arrays must have the same shape")

    required_support = int(math.ceil(len(bootstrap_labels) * float(support_threshold)))
    required_support = max(1, required_support)

    groups: list[list[tuple[int, np.ndarray]]] = []
    proposals_seen = 0
    for boot_idx, labels in enumerate(bootstrap_labels):
        for mask in _unique_instance_masks(np.asarray(labels)):
            proposals_seen += 1
            best_group_idx = None
            best_iou = 0.0
            for group_idx, group in enumerate(groups):
                if any(existing_boot == boot_idx for existing_boot, _ in group):
                    continue
                group_iou = max(_mask_iou(mask, existing_mask) for _, existing_mask in group)
                if group_iou > best_iou:
                    best_iou = group_iou
                    best_group_idx = group_idx

            if best_group_idx is not None and best_iou >= float(cluster_iou_threshold):
                groups[best_group_idx].append((boot_idx, mask))
            else:
                groups.append([(boot_idx, mask)])

    candidate_masks: list[tuple[int, int, np.ndarray]] = []
    dropped_low_support = 0
    dropped_too_small = 0
    for group in groups:
        supporting_bootstraps = {boot_idx for boot_idx, _ in group}
        if len(supporting_bootstraps) < required_support:
            dropped_low_support += 1
            continue

        support_counts = np.zeros(num_points, dtype=np.int16)
        for _, mask in group:
            support_counts += mask.astype(np.int16)
        stable_mask = support_counts >= required_support
        size = int(np.count_nonzero(stable_mask))
        if size < int(min_fused_points):
            dropped_too_small += 1
            continue
        candidate_masks.append((len(supporting_bootstraps), size, stable_mask))

    candidate_masks.sort(key=lambda item: (item[0], item[1]), reverse=True)

    fused = np.full(num_points, -1, dtype=np.int32)
    kept = 0
    assigned_points = np.zeros(num_points, dtype=bool)
    for _, _, mask in candidate_masks:
        write_mask = mask & ~assigned_points
        if int(np.count_nonzero(write_mask)) < int(min_fused_points):
            dropped_too_small += 1
            continue
        fused[write_mask] = kept
        assigned_points |= write_mask
        kept += 1

    stats = {
        "num_bootstraps": int(len(bootstrap_labels)),
        "support_threshold": float(support_threshold),
        "required_support": int(required_support),
        "cluster_iou_threshold": float(cluster_iou_threshold),
        "min_fused_points": int(min_fused_points),
        "num_input_proposals": int(proposals_seen),
        "num_candidate_groups": int(len(groups)),
        "num_groups_dropped_low_support": int(dropped_low_support),
        "num_groups_dropped_too_small": int(dropped_too_small),
        "num_fused_instances": int(kept),
        "num_labeled_points": int(np.count_nonzero(fused >= 0)),
        "labeled_points_fraction": float(np.count_nonzero(fused >= 0) / max(num_points, 1)),
    }
    return fused, stats


def _save_cluster_artifacts(
    *,
    source_adapter: "SceneAdapter",
    cluster_output: ClusterOutput,
    output_dir: Path,
    labels_name: str,
    ply_name: str,
    diagnostics_name: str,
    save_ply: bool = True,
) -> ClusterOutput:
    from chorus.export.visualization import save_labeled_mesh_ply

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / labels_name
    diagnostics_path = output_dir / diagnostics_name
    np.save(labels_path, cluster_output.labels)

    ply_path = None
    if save_ply:
        ply_path = output_dir / ply_name
        save_labeled_mesh_ply(
            source_ply_path=source_adapter.get_geometry_record().geometry_path,
            labels=cluster_output.labels,
            out_path=ply_path,
        )

    diagnostics = {
        "stats": cluster_output.stats,
        "labels_path": str(labels_path),
        "ply_path": str(ply_path) if ply_path is not None else None,
    }
    save_json(diagnostics, output_dir / diagnostics_name)

    return ClusterOutput(
        granularity=cluster_output.granularity,
        labels=cluster_output.labels,
        features=cluster_output.features,
        seen_mask=cluster_output.seen_mask,
        ply_path=ply_path,
        labels_path=labels_path,
        stats={**cluster_output.stats, "diagnostics_path": str(output_dir / diagnostics_name)},
    )


def _load_saved_cluster_artifacts(
    *,
    granularity: float,
    output_dir: Path,
    labels_name: str,
    diagnostics_name: str,
) -> ClusterOutput | None:
    labels_path = output_dir / labels_name
    diagnostics_path = output_dir / diagnostics_name
    if not labels_path.exists():
        return None

    labels = np.load(labels_path).astype(np.int32, copy=False)
    stats: dict[str, Any] = {}
    if diagnostics_path.exists():
        try:
            with diagnostics_path.open("r", encoding="utf-8") as f:
                diagnostics = json.load(f)
            raw_stats = diagnostics.get("stats") if isinstance(diagnostics, dict) else None
            if isinstance(raw_stats, dict):
                stats.update(raw_stats)
        except Exception as exc:
            stats["resume_diagnostics_load_error"] = f"{type(exc).__name__}: {exc}"

    seen_mask = labels >= 0
    stats.update(
        {
            "granularity": float(granularity),
            "num_points": int(labels.shape[0]),
            "num_labeled_points": int(np.count_nonzero(seen_mask)),
            "labeled_points_fraction": float(np.count_nonzero(seen_mask) / max(labels.shape[0], 1)),
            "resumed_from_existing_artifacts": True,
            "diagnostics_path": str(diagnostics_path) if diagnostics_path.exists() else None,
        }
    )
    return ClusterOutput(
        granularity=float(granularity),
        labels=labels,
        features=np.zeros((labels.shape[0], 1), dtype=np.float32),
        seen_mask=seen_mask,
        ply_path=None,
        labels_path=labels_path,
        stats=stats,
    )


def _make_fused_cluster_output(
    *,
    granularity: float,
    labels: np.ndarray,
    bootstrap_outputs: Sequence[ClusterOutput],
    fusion_stats: dict[str, Any],
) -> ClusterOutput:
    if bootstrap_outputs:
        seen_mask = np.any(np.stack([c.seen_mask for c in bootstrap_outputs], axis=0), axis=0)
        features = np.zeros_like(bootstrap_outputs[0].features)
    else:
        seen_mask = labels >= 0
        features = np.zeros((labels.shape[0], 1), dtype=np.float32)

    stats = {
        "granularity": float(granularity),
        "num_points": int(labels.shape[0]),
        "num_clusters": int(len(np.unique(labels[labels >= 0]))),
        "num_labeled_points": int(np.count_nonzero(labels >= 0)),
        "labeled_points_fraction": float(np.count_nonzero(labels >= 0) / max(labels.shape[0], 1)),
        "num_seen_points": int(np.count_nonzero(seen_mask)),
        "seen_points_fraction": float(np.count_nonzero(seen_mask) / max(labels.shape[0], 1)),
        "bootstrap_fusion": fusion_stats,
    }
    return ClusterOutput(
        granularity=float(granularity),
        labels=labels.astype(np.int32, copy=False),
        features=features,
        seen_mask=seen_mask,
        ply_path=None,
        labels_path=None,
        stats=stats,
    )


def _load_baseline_summary(scene_root: Path) -> dict[str, Any] | None:
    path = scene_root / "scene_pipeline_summary.json"
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def run_bootstrap_adapter_experiment(
    *,
    adapter: "SceneAdapter",
    output_scene_dir: Path,
    teacher: "UnSAMv2Teacher",
    config: BootstrapConfig,
    frame_seed: int = 0,
) -> dict[str, Any]:
    from chorus.core.pipeline.project_cluster_stage import (
        _dispatch_hdbscan,
        compute_project_cluster_svd_stage,
        finalize_project_cluster_output,
        run_project_cluster_stage,
    )
    from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics
    from chorus.eval.scannet_oracle import (
        evaluate_and_save_scannet_oracle,
        flatten_oracle_ap_bucket_metrics,
        flatten_oracle_map_bucket_metrics,
    )

    scene_start = time.perf_counter()
    print(f"[bootstrap {adapter.scene_id}] adapter.prepare start", flush=True)
    adapter.prepare()
    print(f"[bootstrap {adapter.scene_id}] adapter.prepare done", flush=True)

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap {adapter.scene_id}] list_frames start", flush=True)
    frames_all = adapter.list_frames()
    print(f"[bootstrap {adapter.scene_id}] list_frames done: {len(frames_all)} frames", flush=True)
    if config.frame_skip <= 0:
        raise ValueError(f"frame_skip must be positive, got {config.frame_skip}")
    frames = frames_all[:: int(config.frame_skip)]
    print(f"[bootstrap {adapter.scene_id}] frame skip done: {len(frames)} candidate frames", flush=True)
    frame_subsets = split_frames_for_bootstraps(
        frames,
        num_bootstraps=config.num_bootstraps,
        frame_fraction=config.frame_fraction,
        frame_sampling=config.frame_sampling,
        seed=frame_seed,
        max_frames_per_bootstrap=config.max_frames_per_bootstrap,
    )
    print(
        f"[bootstrap {adapter.scene_id}] split done: {[len(s) for s in frame_subsets]} frames/subset",
        flush=True,
    )

    fused_outputs: list[ClusterOutput] = []
    by_granularity: dict[str, Any] = {}
    total_teacher_s = 0.0
    total_cluster_s = 0.0
    total_projection_svd_s = 0.0
    total_hdbscan_s = 0.0
    total_fusion_s = 0.0

    for granularity in config.granularities:
        g_key = f"g{granularity}"
        print(f"[bootstrap {adapter.scene_id}] granularity {g_key} start", flush=True)
        bootstrap_outputs: list[ClusterOutput] = []
        bootstrap_rows: list[dict[str, Any]] = []

        if config.frame_sampling == "all":
            mask_dir = output_scene_dir / "shared" / f"unsam_masks_{g_key}"
            t0 = time.perf_counter()
            teacher_output = teacher.run_on_frames(
                adapter=adapter,
                granularity=float(granularity),
                frames=frames,
                output_dir=mask_dir,
                overwrite=config.overwrite_teacher,
                mask_dtype=np.uint16,
            )
            teacher_s = time.perf_counter() - t0
            total_teacher_s += teacher_s

            t0 = time.perf_counter()
            svd_stage, feature_artifacts = compute_project_cluster_svd_stage(
                adapter=adapter,
                teacher_output=teacher_output,
                frame_skip=1,
                svd_components=config.svd_components,
                min_cluster_size=config.min_cluster_size,
                min_samples=config.min_samples,
                cluster_selection_epsilon=config.cluster_selection_epsilon,
            )
            projection_svd_s = time.perf_counter() - t0
            total_projection_svd_s += projection_svd_s

            for boot_idx, subset in enumerate(frame_subsets):
                boot_root = output_scene_dir / "bootstraps" / f"b{boot_idx}"
                seed = config.hdbscan_subsample_seed + boot_idx
                t0 = time.perf_counter()
                labels_seen, clustering_stats = _dispatch_hdbscan(
                    svd_stage.features_seen,
                    min_cluster_size=config.min_cluster_size,
                    min_samples=config.min_samples,
                    cluster_selection_epsilon=config.cluster_selection_epsilon,
                    hdbscan_max_samples=config.hdbscan_max_samples,
                    hdbscan_subsample_seed=seed,
                )
                hdbscan_s = time.perf_counter() - t0
                total_hdbscan_s += hdbscan_s
                cluster_s = (projection_svd_s if boot_idx == 0 else 0.0) + hdbscan_s
                total_cluster_s += cluster_s

                cluster_output = finalize_project_cluster_output(
                    adapter=adapter,
                    teacher_output=teacher_output,
                    svd_stage=svd_stage,
                    feature_artifacts=feature_artifacts,
                    labels_seen=labels_seen,
                    clustering_stats=clustering_stats,
                    save_outputs=False,
                    stats_overlay={
                        "bootstrap_index": int(boot_idx),
                        "frame_sampling": config.frame_sampling,
                        "hdbscan_subsample_seed": int(seed),
                        "shared_teacher_masks": True,
                        "shared_projection_svd": True,
                    },
                )

                saved_output = _save_cluster_artifacts(
                    source_adapter=adapter,
                    cluster_output=cluster_output,
                    output_dir=boot_root / g_key,
                    labels_name="labels.npy",
                    ply_name="result.ply",
                    diagnostics_name="diagnostics.json",
                    save_ply=config.export_bootstrap_ply,
                )
                bootstrap_outputs.append(saved_output)
                bootstrap_rows.append(
                    {
                        "bootstrap_index": int(boot_idx),
                        "num_frames": int(len(subset)),
                        "frame_ids": [f.frame_id for f in subset],
                        "teacher_seconds": float(teacher_s if boot_idx == 0 else 0.0),
                        "project_svd_seconds": float(projection_svd_s if boot_idx == 0 else 0.0),
                        "hdbscan_seconds": float(hdbscan_s),
                        "project_cluster_seconds": float(cluster_s),
                        "total_masks": int(teacher_output.total_masks),
                        "mask_dir": str(mask_dir),
                        "shared_teacher_masks": True,
                        "shared_projection_svd": True,
                        "hdbscan_subsample_seed": int(seed),
                        "num_clusters": saved_output.stats.get("num_clusters"),
                        "labels_path": str(saved_output.labels_path),
                        "diagnostics_path": saved_output.stats.get("diagnostics_path"),
                    }
                )
        else:
            for boot_idx, subset in enumerate(frame_subsets):
                boot_root = output_scene_dir / "bootstraps" / f"b{boot_idx}"
                mask_dir = boot_root / f"unsam_masks_g{granularity}"
                existing_output = _load_saved_cluster_artifacts(
                    granularity=float(granularity),
                    output_dir=boot_root / g_key,
                    labels_name="labels.npy",
                    diagnostics_name="diagnostics.json",
                )
                if existing_output is not None:
                    print(
                        f"[bootstrap {adapter.scene_id}] {g_key} b{boot_idx} resume existing labels: "
                        f"{existing_output.labels_path}",
                        flush=True,
                    )
                    bootstrap_outputs.append(existing_output)
                    bootstrap_rows.append(
                        {
                            "bootstrap_index": int(boot_idx),
                            "num_frames": int(len(subset)),
                            "frame_ids": [f.frame_id for f in subset],
                            "teacher_seconds": 0.0,
                            "project_svd_seconds": 0.0,
                            "hdbscan_seconds": 0.0,
                            "project_cluster_seconds": 0.0,
                            "total_masks": existing_output.stats.get("total_masks"),
                            "mask_dir": str(mask_dir),
                            "shared_teacher_masks": False,
                            "shared_projection_svd": False,
                            "hdbscan_subsample_seed": int(config.hdbscan_subsample_seed + boot_idx),
                            "num_clusters": existing_output.stats.get("num_clusters"),
                            "labels_path": str(existing_output.labels_path),
                            "diagnostics_path": existing_output.stats.get("diagnostics_path"),
                            "resumed_from_existing_artifacts": True,
                        }
                    )
                    continue

                t0 = time.perf_counter()
                print(
                    f"[bootstrap {adapter.scene_id}] {g_key} b{boot_idx} teacher start: "
                    f"{len(subset)} frames",
                    flush=True,
                )
                teacher_output = teacher.run_on_frames(
                    adapter=adapter,
                    granularity=float(granularity),
                    frames=subset,
                    output_dir=mask_dir,
                    overwrite=config.overwrite_teacher,
                    mask_dtype=np.uint16,
                )
                teacher_s = time.perf_counter() - t0
                total_teacher_s += teacher_s
                print(
                    f"[bootstrap {adapter.scene_id}] {g_key} b{boot_idx} teacher done in "
                    f"{teacher_s:.1f}s",
                    flush=True,
                )

                t0 = time.perf_counter()
                print(f"[bootstrap {adapter.scene_id}] {g_key} b{boot_idx} cluster start", flush=True)
                cluster_output = run_project_cluster_stage(
                    adapter=adapter,
                    teacher_output=teacher_output,
                    frame_skip=1,
                    svd_components=config.svd_components,
                    min_cluster_size=config.min_cluster_size,
                    min_samples=config.min_samples,
                    cluster_selection_epsilon=config.cluster_selection_epsilon,
                    save_outputs=False,
                    hdbscan_max_samples=config.hdbscan_max_samples,
                    hdbscan_subsample_seed=config.hdbscan_subsample_seed + boot_idx,
                )
                cluster_s = time.perf_counter() - t0
                hdbscan_s = float(cluster_output.stats.get("hdbscan_cluster_wall_seconds", 0.0))
                projection_svd_s = max(0.0, cluster_s - hdbscan_s)
                total_cluster_s += cluster_s
                total_projection_svd_s += projection_svd_s
                total_hdbscan_s += hdbscan_s
                print(
                    f"[bootstrap {adapter.scene_id}] {g_key} b{boot_idx} cluster done in "
                    f"{cluster_s:.1f}s",
                    flush=True,
                )

                saved_output = _save_cluster_artifacts(
                    source_adapter=adapter,
                    cluster_output=cluster_output,
                    output_dir=boot_root / g_key,
                    labels_name="labels.npy",
                    ply_name="result.ply",
                    diagnostics_name="diagnostics.json",
                    save_ply=config.export_bootstrap_ply,
                )
                bootstrap_outputs.append(saved_output)
                bootstrap_rows.append(
                    {
                        "bootstrap_index": int(boot_idx),
                        "num_frames": int(len(subset)),
                        "frame_ids": [f.frame_id for f in subset],
                        "teacher_seconds": float(teacher_s),
                        "project_svd_seconds": float(projection_svd_s),
                        "hdbscan_seconds": float(hdbscan_s),
                        "project_cluster_seconds": float(cluster_s),
                        "total_masks": int(teacher_output.total_masks),
                        "mask_dir": str(mask_dir),
                        "shared_teacher_masks": False,
                        "shared_projection_svd": False,
                        "hdbscan_subsample_seed": int(config.hdbscan_subsample_seed + boot_idx),
                        "num_clusters": saved_output.stats.get("num_clusters"),
                        "labels_path": str(saved_output.labels_path),
                        "diagnostics_path": saved_output.stats.get("diagnostics_path"),
                    }
                )

        t0 = time.perf_counter()
        fused_labels, fusion_stats = fuse_stable_core_labels(
            [c.labels for c in bootstrap_outputs],
            support_threshold=config.support_threshold,
            cluster_iou_threshold=config.cluster_iou_threshold,
            min_fused_points=config.min_fused_points,
        )
        fusion_s = time.perf_counter() - t0
        total_fusion_s += fusion_s
        fusion_stats["fusion_seconds"] = float(fusion_s)

        fused_output = _make_fused_cluster_output(
            granularity=float(granularity),
            labels=fused_labels,
            bootstrap_outputs=bootstrap_outputs,
            fusion_stats=fusion_stats,
        )
        fused_output = _save_cluster_artifacts(
            source_adapter=adapter,
            cluster_output=fused_output,
            output_dir=output_scene_dir / "fused",
            labels_name=f"labels_{g_key}.npy",
            ply_name=f"result_{g_key}.ply",
            diagnostics_name=f"diagnostics_{g_key}.json",
            save_ply=True,
        )
        fused_outputs.append(fused_output)

        by_granularity[g_key] = {
            "granularity": float(granularity),
            "bootstraps": bootstrap_rows,
            "fusion": fusion_stats,
            "fused_labels_path": str(fused_output.labels_path),
            "fused_ply_path": str(fused_output.ply_path),
            "fused_diagnostics_path": fused_output.stats.get("diagnostics_path"),
        }

    scene_intrinsic_metrics = compute_scene_intrinsic_metrics(fused_outputs)
    training_pack_dir = export_training_scene_pack(
        adapter=adapter,
        cluster_outputs=fused_outputs,
        output_dir=output_scene_dir / "fused" / "training_pack",
        teacher_name="UnSAMv2BootstrapStableCore",
        projection_type="zbuffer_rgbd_bootstrap",
        embedding_type="truncated_svd",
        clustering_type="hdbscan_subsample_bootstrap_stable_core",
        clustering_backend="sklearn_subsample_1nn",
        frame_skip=None,
        scene_intrinsic_metrics=scene_intrinsic_metrics,
    )

    evaluation_summary = None
    if config.run_oracle_eval:
        oracle_summaries: dict[str, Any] = {}
        for benchmark in config.eval_benchmarks:
            oracle_summaries[benchmark] = evaluate_and_save_scannet_oracle(
                adapter=adapter,
                cluster_outputs=fused_outputs,
                eval_benchmark=benchmark,
                save_artifacts=True,
                output_dir=output_scene_dir / "fused" / "oracle",
                gt_scene_root=adapter.scene_root if adapter.dataset_name == "scannet" else None,
            )
        primary = oracle_summaries.get(config.eval_benchmarks[0]) if config.eval_benchmarks else None
        evaluation_summary = {
            "eval_benchmarks": list(config.eval_benchmarks),
            "oracle_summaries": {
                k: {
                    **v,
                    "metrics_path": str(v["metrics_path"]) if v.get("metrics_path") is not None else None,
                    "labels_path": str(v["labels_path"]) if v.get("labels_path") is not None else None,
                    "ply_path": str(v["ply_path"]) if v.get("ply_path") is not None else None,
                }
                for k, v in oracle_summaries.items()
            },
        }
        if primary is not None:
            evaluation_summary["oracle_summary"] = evaluation_summary["oracle_summaries"][config.eval_benchmarks[0]]

    flat_oracle: dict[str, Any] = {}
    if evaluation_summary is not None:
        primary = evaluation_summary.get("oracle_summary") or {}
        flat_oracle.update(flatten_oracle_ap_bucket_metrics(primary.get("oracle_results")))
        flat_oracle.update(flatten_oracle_map_bucket_metrics(primary.get("additional_metrics")))
        clustering = primary.get("clustering_metrics") or {}
        flat_oracle["oracle_nmi"] = clustering.get("NMI")
        flat_oracle["oracle_ari"] = clustering.get("ARI")

    summary = {
        "experiment": "scannet_bootstrap_stable_core",
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "source_scene_root": str(adapter.scene_root),
        "output_scene_dir": str(output_scene_dir),
        "config": {
            **config.__dict__,
            "granularities": [float(g) for g in config.granularities],
            "eval_benchmarks": list(config.eval_benchmarks),
        },
        "num_frames_total": int(len(frames_all)),
        "num_candidate_frames": int(len(frames)),
        "frame_skip": int(config.frame_skip),
        "max_frames_per_bootstrap": config.max_frames_per_bootstrap,
        "frame_subsets": [[f.frame_id for f in subset] for subset in frame_subsets],
        "by_granularity": by_granularity,
        "fused_outputs": [
            {
                "granularity": float(c.granularity),
                "labels_path": str(c.labels_path) if c.labels_path is not None else None,
                "ply_path": str(c.ply_path) if c.ply_path is not None else None,
                "stats": c.stats,
            }
            for c in fused_outputs
        ],
        "scene_intrinsic_metrics": scene_intrinsic_metrics,
        "training_pack_dir": str(training_pack_dir),
        "timing_seconds": {
            "teacher": float(total_teacher_s),
            "projection_svd": float(total_projection_svd_s),
            "hdbscan": float(total_hdbscan_s),
            "project_cluster": float(total_cluster_s),
            "fusion": float(total_fusion_s),
            "total_wall": float(time.perf_counter() - scene_start),
        },
        "baseline_summary_path": str(adapter.scene_root / "scene_pipeline_summary.json")
        if (adapter.scene_root / "scene_pipeline_summary.json").exists()
        else None,
        "baseline_summary": _load_baseline_summary(adapter.scene_root),
        **flat_oracle,
    }
    if evaluation_summary is not None:
        summary.update(evaluation_summary)

    summary_path = output_scene_dir / "bootstrap_experiment_summary.json"
    save_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def run_bootstrap_scene_experiment(
    *,
    scene_root: Path,
    output_scene_dir: Path,
    teacher: "UnSAMv2Teacher",
    config: BootstrapConfig,
    frame_seed: int = 0,
) -> dict[str, Any]:
    from chorus.datasets.scannet.adapter import ScanNetSceneAdapter

    adapter = ScanNetSceneAdapter(scene_root=Path(scene_root), eval_benchmarks=list(config.eval_benchmarks))
    return run_bootstrap_adapter_experiment(
        adapter=adapter,
        output_scene_dir=output_scene_dir,
        teacher=teacher,
        config=config,
        frame_seed=frame_seed,
    )
