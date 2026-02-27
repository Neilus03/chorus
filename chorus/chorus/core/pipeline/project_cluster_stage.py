from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from chorus.common.types import ClusterOutput, TeacherOutput
from chorus.core.clustering.hdbscan_cluster import cluster_features
from chorus.core.embedding.svd import compute_svd_features
from chorus.core.lifting.project import project_points_to_image
from chorus.core.lifting.visibility import compute_visible_points
from chorus.core.lifting.voting import build_point_mask_matrix
from chorus.datasets.base import SceneAdapter


def _resize_depth_to_mask_shape(depth_map_m: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = mask_shape
    if depth_map_m.shape == (target_h, target_w):
        return depth_map_m
    return cv2.resize(depth_map_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _save_colored_point_cloud(points_3d: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    point_colors = np.zeros((points_3d.shape[0], 3), dtype=np.float64)
    valid_mask = labels >= 0

    if np.any(valid_mask):
        max_label = int(labels[valid_mask].max())
        rng = np.random.default_rng(42)
        colors = rng.random((max_label + 1, 3))
        point_colors[valid_mask] = colors[labels[valid_mask]]

    point_colors[~valid_mask] = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    o3d.io.write_point_cloud(str(out_path), pcd)


def run_project_cluster_stage(
    adapter: SceneAdapter,
    teacher_output: TeacherOutput,
    frame_skip: int,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    save_outputs: bool = True,
) -> ClusterOutput:
    points_3d = adapter.load_geometry_points()
    num_points = points_3d.shape[0]
    visibility_cfg = adapter.get_visibility_config()

    frames = adapter.list_frames()[::frame_skip]
    frame_id_to_mask_path = {p.stem: p for p in teacher_output.frame_mask_paths}

    point_assignments: list[np.ndarray] = []
    mask_assignments: list[np.ndarray] = []
    used_frames = 0
    skipped_frames = 0

    print(
        f"Project+Cluster: scene={adapter.scene_id}, granularity={teacher_output.granularity}, "
        f"frames_considered={len(frames)}, num_points={num_points}"
    )

    for frame in frames:
        mask_path = frame_id_to_mask_path.get(frame.frame_id)
        if mask_path is None or not mask_path.exists():
            skipped_frames += 1
            continue

        try:
            pose_c2w = adapter.load_pose_c2w(frame)
        except Exception as exc:
            print(f"Skipping frame {frame.frame_id} due to invalid pose: {exc}")
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

    point_mask_matrix, voting_stats = build_point_mask_matrix(
        point_assignments=point_assignments,
        mask_assignments=mask_assignments,
        num_points=num_points,
    )

    votes_per_point = np.asarray(point_mask_matrix.sum(axis=1)).reshape(-1)
    unseen_points = int(np.sum(votes_per_point == 0))

    features, svd_stats = compute_svd_features(
        point_mask_matrix=point_mask_matrix,
        n_components=svd_components,
    )

    labels, clustering_stats = cluster_features(
        features=features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    labels_path = None
    ply_path = None

    if save_outputs:
        labels_path = adapter.scene_root / f"chorus_instance_labels_g{teacher_output.granularity}.npy"
        np.save(labels_path, labels)

        features_path = adapter.scene_root / f"svd_features_g{teacher_output.granularity}.npy"
        np.save(features_path, features)

        ply_path = adapter.scene_root / f"chorus_instance_result_g{teacher_output.granularity}.ply"
        _save_colored_point_cloud(points_3d=points_3d, labels=labels, out_path=ply_path)

    stats = {
        "scene_id": adapter.scene_id,
        "dataset": adapter.dataset_name,
        "granularity": teacher_output.granularity,
        "num_points": int(num_points),
        "used_frames": int(used_frames),
        "skipped_frames": int(skipped_frames),
        "unseen_points": int(unseen_points),
        "unseen_points_fraction": float(unseen_points / max(num_points, 1)),
        **voting_stats,
        **svd_stats,
        **clustering_stats,
    }

    print(
        f"Project+Cluster complete: scene={adapter.scene_id}, granularity={teacher_output.granularity}, "
        f"clusters={stats['num_clusters']}, noise_fraction={stats['noise_fraction']:.3f}, "
        f"unseen_fraction={stats['unseen_points_fraction']:.3f}"
    )

    return ClusterOutput(
        granularity=teacher_output.granularity,
        labels=labels,
        features=features,
        ply_path=ply_path,
        labels_path=labels_path,
        stats=stats,
    )