from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chorus.common.types import ClusterOutput, TeacherOutput
from chorus.common.progress import log_progress, phase_timer
from chorus.core.clustering.hdbscan_cluster import cluster_features
from chorus.core.clustering.hdbscan_subsample import cluster_features_with_subsample_cap
from chorus.core.embedding.svd import compute_svd_features
from chorus.core.lifting.project import project_points_to_image
from chorus.core.lifting.visibility import compute_visible_points
from chorus.core.lifting.voting import build_point_mask_matrix
from chorus.core.quality.diagnostics import save_json
from chorus.core.quality.intrinsic_metrics import compute_cluster_intrinsic_metrics
from chorus.datasets.base import SceneAdapter
from chorus.export.visualization import save_labeled_mesh_ply


def _resize_depth_to_mask_shape(depth_map_m: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = mask_shape
    if depth_map_m.shape == (target_h, target_w):
        return depth_map_m
    return cv2.resize(depth_map_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_save_hdbscan_features(
    *,
    adapter: SceneAdapter,
    granularity: float,
    features_seen: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> dict[str, str]:
    if not _env_flag_enabled("CHORUS_SAVE_HDBSCAN_FEATURES", default=False):
        return {}

    output_dir_raw = os.environ.get("CHORUS_HDBSCAN_FEATURES_DIR")
    output_dir = Path(output_dir_raw) if output_dir_raw else (adapter.scene_root / "benchmark_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"hdbscan_features_scene_{adapter.scene_id}_g{granularity}"
    features_path = output_dir / f"{stem}.npy"
    meta_path = output_dir / f"{stem}.json"

    np.save(features_path, features_seen)
    save_json(
        {
            "dataset": adapter.dataset_name,
            "scene_id": adapter.scene_id,
            "granularity": float(granularity),
            "features_path": str(features_path),
            "num_samples": int(features_seen.shape[0]),
            "num_features": int(features_seen.shape[1]) if features_seen.ndim > 1 else 1,
            "dtype": str(features_seen.dtype),
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
        },
        meta_path,
    )
    log_progress(
        "Saved HDBSCAN benchmark features: "
        f"scene={adapter.scene_id}, granularity={granularity}, path={features_path}"
    )
    return {
        "hdbscan_features_path": str(features_path),
        "hdbscan_features_meta_path": str(meta_path),
    }


@dataclass
class ProjectClusterSvdStage:
    features_seen: np.ndarray
    seen_mask: np.ndarray
    num_points: int
    used_frames: int
    skipped_frames: int
    num_seen_points: int
    unseen_points: int
    voting_stats: dict[str, Any]
    svd_stats: dict[str, Any]


def compute_project_cluster_svd_stage(
    adapter: SceneAdapter,
    teacher_output: TeacherOutput,
    frame_skip: int,
    svd_components: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> tuple[ProjectClusterSvdStage, dict[str, str]]:
    points_3d = adapter.load_geometry_points()
    num_points = points_3d.shape[0]
    visibility_cfg = adapter.get_visibility_config()

    frames = adapter.list_frames()[::frame_skip]
    frame_id_to_mask_path = {p.stem: p for p in teacher_output.frame_mask_paths}

    point_assignments: list[np.ndarray] = []
    mask_assignments: list[np.ndarray] = []
    used_frames = 0
    skipped_frames = 0

    log_progress(
        f"Project+Cluster: scene={adapter.scene_id}, granularity={teacher_output.granularity}, "
        f"frames_considered={len(frames)}, num_points={num_points}"
    )

    with phase_timer(
        f"Project+Cluster projection loop: scene={adapter.scene_id}, granularity={teacher_output.granularity}"
    ):
        for frame in frames:
            mask_path = frame_id_to_mask_path.get(frame.frame_id)
            if mask_path is None or not mask_path.exists():
                skipped_frames += 1
                continue

            try:
                pose_c2w = adapter.load_pose_c2w(frame)
            except Exception as exc:
                log_progress(f"Skipping frame {frame.frame_id} due to invalid pose: {exc}")
                skipped_frames += 1
                continue

            intrinsics = adapter.load_intrinsics(frame)
            depth_map_m = adapter.load_depth_m(frame)
            pred_2d_mask = np.load(mask_path)

            if pred_2d_mask.ndim != 2:
                raise RuntimeError(f"Expected 2D mask array in {mask_path}, got shape={pred_2d_mask.shape}")

            depth_map_m = _resize_depth_to_mask_shape(depth_map_m, pred_2d_mask.shape)

            u, v, z, valid_indices = project_points_to_image(
                points_3d=points_3d,
                pose_c2w=pose_c2w,
                intrinsics=intrinsics,
            )

            visible_indices, visible_u, visible_v = compute_visible_points(
                u=u,
                v=v,
                z=z,
                valid_indices=valid_indices,
                depth_map_m=depth_map_m,
                visibility_cfg=visibility_cfg,
            )

            predicted_local_ids = pred_2d_mask[visible_v, visible_u]

            point_assignments.append(visible_indices)
            mask_assignments.append(predicted_local_ids)
            used_frames += 1

    with phase_timer(
        f"Project+Cluster sparse vote matrix build: scene={adapter.scene_id}, granularity={teacher_output.granularity}"
    ):
        point_mask_matrix, voting_stats = build_point_mask_matrix(
            point_assignments=point_assignments,
            mask_assignments=mask_assignments,
            num_points=num_points,
        )

    votes_per_point = np.asarray(point_mask_matrix.sum(axis=1)).reshape(-1)
    seen_mask = votes_per_point > 0
    num_seen_points = int(np.sum(seen_mask))
    unseen_points = int(num_points - num_seen_points)

    log_progress(
        "Project+Cluster visibility summary: "
        f"scene={adapter.scene_id}, granularity={teacher_output.granularity}, "
        f"used_frames={used_frames}, seen_points={num_seen_points}/{num_points} "
        f"({num_seen_points / max(num_points, 1):.3f}), "
        f"point_mask_matrix_shape={point_mask_matrix.shape}, nnz={int(point_mask_matrix.nnz)}"
    )

    if num_seen_points == 0:
        raise RuntimeError("No 3D points received any valid 2D mask votes. Check teacher outputs and paths.")

    point_mask_matrix_seen = point_mask_matrix[seen_mask]

    features_seen, svd_stats = compute_svd_features(
        point_mask_matrix=point_mask_matrix_seen,
        n_components=svd_components,
    )
    feature_artifacts = _maybe_save_hdbscan_features(
        adapter=adapter,
        granularity=teacher_output.granularity,
        features_seen=features_seen,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    stage = ProjectClusterSvdStage(
        features_seen=features_seen,
        seen_mask=seen_mask,
        num_points=num_points,
        used_frames=used_frames,
        skipped_frames=skipped_frames,
        num_seen_points=num_seen_points,
        unseen_points=unseen_points,
        voting_stats=voting_stats,
        svd_stats=svd_stats,
    )
    return stage, feature_artifacts


def _dispatch_hdbscan(
    features_seen: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    hdbscan_max_samples: int | None,
    hdbscan_subsample_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(features_seen.shape[0])
    t_cluster = time.perf_counter()
    if hdbscan_max_samples is not None and n > int(hdbscan_max_samples):
        labels_seen, clustering_stats = cluster_features_with_subsample_cap(
            features_seen,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_samples=int(hdbscan_max_samples),
            rng=np.random.default_rng(int(hdbscan_subsample_seed)),
        )
    else:
        labels_seen, clustering_stats = cluster_features(
            features_seen,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
    clustering_stats = dict(clustering_stats)
    clustering_stats["hdbscan_cluster_wall_seconds"] = float(time.perf_counter() - t_cluster)
    return labels_seen, clustering_stats


def finalize_project_cluster_output(
    adapter: SceneAdapter,
    teacher_output: TeacherOutput,
    svd_stage: ProjectClusterSvdStage,
    feature_artifacts: dict[str, str],
    labels_seen: np.ndarray,
    clustering_stats: dict[str, Any],
    save_outputs: bool,
    stats_overlay: dict[str, Any] | None = None,
) -> ClusterOutput:
    num_points = svd_stage.num_points
    seen_mask = svd_stage.seen_mask
    features_seen = svd_stage.features_seen
    used_frames = svd_stage.used_frames
    skipped_frames = svd_stage.skipped_frames
    num_seen_points = svd_stage.num_seen_points
    unseen_points = svd_stage.unseen_points
    voting_stats = svd_stage.voting_stats
    svd_stats = svd_stage.svd_stats

    full_labels = np.full(num_points, -1, dtype=np.int32)
    full_labels[seen_mask] = labels_seen

    full_features = np.zeros((num_points, features_seen.shape[1]), dtype=features_seen.dtype)
    full_features[seen_mask] = features_seen

    num_labeled_points = int(np.sum(full_labels >= 0))
    num_noise_points_seen = int(np.sum(seen_mask & (full_labels < 0)))

    labels_path = None
    ply_path = None
    diagnostics_path = None

    stats = {
        "scene_id": adapter.scene_id,
        "dataset": adapter.dataset_name,
        "granularity": teacher_output.granularity,
        "num_points": int(num_points),
        "used_frames": int(used_frames),
        "skipped_frames": int(skipped_frames),
        "num_seen_points": int(num_seen_points),
        "seen_points_fraction": float(num_seen_points / max(num_points, 1)),
        "unseen_points": int(unseen_points),
        "unseen_points_fraction": float(unseen_points / max(num_points, 1)),
        "num_labeled_points": int(num_labeled_points),
        "labeled_points_fraction": float(num_labeled_points / max(num_points, 1)),
        "labeled_points_fraction_seen": float(num_labeled_points / max(num_seen_points, 1)),
        "num_noise_points_seen": int(num_noise_points_seen),
        "noise_fraction_seen": float(num_noise_points_seen / max(num_seen_points, 1)),
        "noise_fraction_all_points": float(num_noise_points_seen / max(num_points, 1)),
        **voting_stats,
        **svd_stats,
        **feature_artifacts,
        **clustering_stats,
    }
    if stats_overlay:
        stats.update(stats_overlay)

    cluster_output = ClusterOutput(
        granularity=teacher_output.granularity,
        labels=full_labels,
        features=full_features,
        seen_mask=seen_mask,
        ply_path=None,
        labels_path=None,
        stats=stats,
    )

    intrinsic_metrics = compute_cluster_intrinsic_metrics(cluster_output)

    if save_outputs:
        with phase_timer(
            f"Project+Cluster export: scene={adapter.scene_id}, granularity={teacher_output.granularity}"
        ):
            labels_path = adapter.scene_root / f"chorus_instance_labels_g{teacher_output.granularity}.npy"
            np.save(labels_path, full_labels)

            features_path = adapter.scene_root / f"svd_features_g{teacher_output.granularity}.npy"
            np.save(features_path, full_features)

            ply_path = adapter.scene_root / f"chorus_instance_result_g{teacher_output.granularity}.ply"
            geometry_record = adapter.get_geometry_record()
            save_labeled_mesh_ply(
                source_ply_path=geometry_record.geometry_path,
                labels=full_labels,
                out_path=ply_path,
            )

            diagnostics_path = adapter.scene_root / f"diagnostics_g{teacher_output.granularity}.json"
            save_json(
                {
                    "stats": stats,
                    "intrinsic_metrics": intrinsic_metrics,
                    "labels_path": str(labels_path),
                    "features_path": str(features_path),
                    "ply_path": str(ply_path),
                },
                diagnostics_path,
            )

    stats["intrinsic_metrics"] = intrinsic_metrics
    if diagnostics_path is not None:
        stats["diagnostics_path"] = str(diagnostics_path)

    log_progress(
        f"Project+Cluster complete: scene={adapter.scene_id}, granularity={teacher_output.granularity}, "
        f"clusters={stats['num_clusters']}, noise_fraction_seen={stats['noise_fraction_seen']:.3f}, "
        f"unseen_fraction={stats['unseen_points_fraction']:.3f}"
    )

    return ClusterOutput(
        granularity=teacher_output.granularity,
        labels=full_labels,
        features=full_features,
        seen_mask=seen_mask,
        ply_path=ply_path,
        labels_path=labels_path,
        stats=stats,
    )


def run_project_cluster_stage(
    adapter: SceneAdapter,
    teacher_output: TeacherOutput,
    frame_skip: int,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    save_outputs: bool = True,
    hdbscan_max_samples: int | None = None,
    hdbscan_subsample_seed: int = 0,
) -> ClusterOutput:
    svd_stage, feature_artifacts = compute_project_cluster_svd_stage(
        adapter=adapter,
        teacher_output=teacher_output,
        frame_skip=frame_skip,
        svd_components=svd_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels_seen, clustering_stats = _dispatch_hdbscan(
        svd_stage.features_seen,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        hdbscan_max_samples=hdbscan_max_samples,
        hdbscan_subsample_seed=hdbscan_subsample_seed,
    )
    return finalize_project_cluster_output(
        adapter=adapter,
        teacher_output=teacher_output,
        svd_stage=svd_stage,
        feature_artifacts=feature_artifacts,
        labels_seen=labels_seen,
        clustering_stats=clustering_stats,
        save_outputs=save_outputs,
        stats_overlay=None,
    )


def _hdbscan_cap_from_seen_fraction(
    num_seen: int,
    fraction: float,
    min_cluster_size: int,
) -> int:
    """Max HDBSCAN rows = round(fraction * seen), floored at min_cluster_size, capped at num_seen."""
    if num_seen <= 0:
        return 0
    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"subsample fraction must be in (0, 1], got {fraction}")
    m = int(round(frac * num_seen))
    m = max(int(min_cluster_size), m)
    return min(m, int(num_seen))


def run_project_cluster_hdbscan_subsample_ablation(
    adapter: SceneAdapter,
    teacher_output: TeacherOutput,
    frame_skip: int,
    svd_components: int,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    subsample_fractions: Sequence[float] | None,
    subsample_seed: int,
    eval_benchmark: str | None,
) -> dict[str, Any]:
    """One SVD lift, full HDBSCAN once, then per-fraction capped subsample + 1-NN vs full (no artifact writes)."""
    from sklearn.metrics import adjusted_rand_score

    from chorus.eval.scannet_oracle import (
        evaluate_and_save_scannet_oracle,
        flatten_oracle_ap_bucket_metrics,
    )

    fractions = tuple(subsample_fractions) if subsample_fractions is not None else (0.9, 0.75, 0.5, 0.25)

    svd_stage, feature_artifacts = compute_project_cluster_svd_stage(
        adapter=adapter,
        teacher_output=teacher_output,
        frame_skip=frame_skip,
        svd_components=svd_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    n_seen = int(svd_stage.num_seen_points)
    labels_full, stats_full = _dispatch_hdbscan(
        svd_stage.features_seen,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        hdbscan_max_samples=None,
        hdbscan_subsample_seed=0,
    )
    cluster_full = finalize_project_cluster_output(
        adapter=adapter,
        teacher_output=teacher_output,
        svd_stage=svd_stage,
        feature_artifacts=feature_artifacts,
        labels_seen=labels_full,
        clustering_stats=stats_full,
        save_outputs=False,
        stats_overlay={"hdbscan_ablation_branch": "full"},
    )

    oracle_full = evaluate_and_save_scannet_oracle(
        adapter=adapter,
        cluster_outputs=[cluster_full],
        eval_benchmark=eval_benchmark,
        save_artifacts=False,
    )
    cm_full = oracle_full.get("clustering_metrics") or {}
    ap_full = flatten_oracle_ap_bucket_metrics(oracle_full.get("oracle_results"))
    wall_full = float(stats_full.get("hdbscan_cluster_wall_seconds", 0.0))

    full_block: dict[str, Any] = {
        "timing_seconds": {
            "full_hdbscan_cluster_wall": wall_full,
            "full_hdbscan_fit_predict": float(stats_full.get("hdbscan_fit_predict_seconds", 0.0)),
        },
        "oracle_clustering_metrics": dict(cm_full),
        "oracle_ap": dict(ap_full),
        "intrinsic_metrics": cluster_full.stats.get("intrinsic_metrics"),
    }

    by_fraction: dict[str, Any] = {}
    for frac in fractions:
        cap = _hdbscan_cap_from_seen_fraction(n_seen, frac, min_cluster_size)
        frac_key = f"{frac:g}"
        equiv_full = cap >= n_seen

        if equiv_full:
            by_fraction[frac_key] = {
                "subsample_fraction": float(frac),
                "hdbscan_max_samples_effective": int(cap),
                "subsample_equivalent_to_full": True,
                "pseudolabel_ari_vs_full": 1.0,
                "timing_seconds": {
                    "subsample_hdbscan_cluster_wall": 0.0,
                    "speedup_full_over_sub": None,
                    "subsample_hdbscan_fit_predict": 0.0,
                    "subsample_propagate_nn": 0.0,
                },
                "oracle_clustering_metrics_sub": dict(cm_full),
                "oracle_nmi_delta_sub_minus_full": 0.0
                if cm_full.get("NMI") is not None
                else None,
                "oracle_ari_delta_sub_minus_full": 0.0
                if cm_full.get("ARI") is not None
                else None,
                "oracle_ap": {
                    "sub": dict(ap_full),
                    "delta_sub_minus_full": {},
                },
                "intrinsic_sub": cluster_full.stats.get("intrinsic_metrics"),
            }
            continue

        labels_sub, stats_sub = _dispatch_hdbscan(
            svd_stage.features_seen,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            hdbscan_max_samples=int(cap),
            hdbscan_subsample_seed=subsample_seed,
        )
        cluster_sub = finalize_project_cluster_output(
            adapter=adapter,
            teacher_output=teacher_output,
            svd_stage=svd_stage,
            feature_artifacts=feature_artifacts,
            labels_seen=labels_sub,
            clustering_stats=stats_sub,
            save_outputs=False,
            stats_overlay={"hdbscan_ablation_branch": "subsample"},
        )

        pseudolabel_ari = float(adjusted_rand_score(labels_full, labels_sub))

        oracle_sub = evaluate_and_save_scannet_oracle(
            adapter=adapter,
            cluster_outputs=[cluster_sub],
            eval_benchmark=eval_benchmark,
            save_artifacts=False,
        )
        cm_s = oracle_sub.get("clustering_metrics") or {}
        ap_s = flatten_oracle_ap_bucket_metrics(oracle_sub.get("oracle_results"))
        ap_delta: dict[str, float] = {}
        for k in sorted(set(ap_full) | set(ap_s)):
            vf, vs = ap_full.get(k), ap_s.get(k)
            if vf is not None and vs is not None:
                ap_delta[k] = float(vs) - float(vf)

        wall_sub = float(stats_sub.get("hdbscan_cluster_wall_seconds", 0.0))
        speedup = (wall_full / wall_sub) if wall_sub > 1e-9 else None

        by_fraction[frac_key] = {
            "subsample_fraction": float(frac),
            "hdbscan_max_samples_effective": int(cap),
            "subsample_equivalent_to_full": False,
            "pseudolabel_ari_vs_full": pseudolabel_ari,
            "timing_seconds": {
                "subsample_hdbscan_cluster_wall": wall_sub,
                "speedup_full_over_sub": speedup,
                "subsample_hdbscan_fit_predict": float(stats_sub.get("hdbscan_fit_predict_seconds", 0.0)),
                "subsample_propagate_nn": float(stats_sub.get("hdbscan_propagate_nn_seconds", 0.0)),
            },
            "oracle_clustering_metrics_sub": dict(cm_s),
            "oracle_nmi_delta_sub_minus_full": float(cm_s.get("NMI", 0.0)) - float(cm_full.get("NMI", 0.0))
            if cm_full.get("NMI") is not None and cm_s.get("NMI") is not None
            else None,
            "oracle_ari_delta_sub_minus_full": float(cm_s.get("ARI", 0.0)) - float(cm_full.get("ARI", 0.0))
            if cm_full.get("ARI") is not None and cm_s.get("ARI") is not None
            else None,
            "oracle_ap": {
                "sub": dict(ap_s),
                "delta_sub_minus_full": dict(ap_delta),
            },
            "intrinsic_sub": cluster_sub.stats.get("intrinsic_metrics"),
        }

    return {
        "scene_id": adapter.scene_id,
        "granularity": float(teacher_output.granularity),
        "num_seen_points": n_seen,
        "subsample_fractions": [float(f) for f in fractions],
        "subsample_seed": int(subsample_seed),
        "full": full_block,
        "by_fraction": by_fraction,
    }