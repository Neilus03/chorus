import os

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from sklearn.cluster import Birch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from sklearn.cluster import HDBSCAN

from pathlib import Path



# --- CONFIG ---
SCENE_DIR = os.environ.get("SCENE_DIR", "scene0000_00")
SCENE_NAME = Path(SCENE_DIR).name  # This safely extracts 'scene0140_00'
FRAME_SKIP = 10
GRANULARITY = float(os.environ.get("GRANULARITY", "0.5"))
MASK_DIR = os.path.join(SCENE_DIR, f"unsam_masks_g{GRANULARITY}")
SVD_COMPONENTS = 32

CLUSTER_BACKEND = os.environ.get("CLUSTER_BACKEND", "cpu_hdbscan").strip().lower()

HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE", "100"))
HDBSCAN_MIN_SAMPLES = int(os.environ.get("HDBSCAN_MIN_SAMPLES", "5"))
HDBSCAN_EPS = float(os.environ.get("HDBSCAN_EPS", "0.1"))
HDBSCAN_N_JOBS = int(os.environ.get("HDBSCAN_N_JOBS", "-1"))

BIRCH_THRESHOLD = float(os.environ.get("BIRCH_THRESHOLD", "0.7"))
BIRCH_BRANCHING_FACTOR = int(os.environ.get("BIRCH_BRANCHING_FACTOR", "50"))
GPU_FALLBACK_TO_CPU = os.environ.get("GPU_FALLBACK_TO_CPU", "1").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def main() -> None:
    print("Loading 3D geometry...")
    mesh_path = os.path.join(SCENE_DIR, f"{SCENE_NAME}_vh_clean_2.ply")
    pcd = o3d.io.read_point_cloud(mesh_path)
    points_3d = np.asarray(pcd.points)
    num_points = points_3d.shape[0]

    intrinsic_path = os.path.join(SCENE_DIR, "intrinsic", "intrinsic_color.txt")
    k = np.loadtxt(intrinsic_path)[:3, :3]

    color_dir = os.path.join(SCENE_DIR, "color")
    target_frames = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )[::FRAME_SKIP]

    print("Bridging 2D masks to 3D space...")
    row_idx = []
    col_idx = []
    global_mask_counter = 0

    for frame_name in target_frames:
        frame_idx = frame_name.split(".")[0]
        pose_path = os.path.join(SCENE_DIR, "pose", f"{frame_idx}.txt")
        depth_path = os.path.join(SCENE_DIR, "depth", f"{frame_idx}.png")
        mask_path = os.path.join(MASK_DIR, f"{frame_idx}.npy")

        if not (
            os.path.exists(pose_path)
            and os.path.exists(depth_path)
            and os.path.exists(mask_path)
        ):
            continue

        pose = np.loadtxt(pose_path)
        if np.isinf(pose).any() or np.isnan(pose).any():
            continue

        pred_2d_mask = np.load(mask_path)
        color_h, color_w = pred_2d_mask.shape

        raw_depth_map = cv2.imread(depth_path, -1) / 1000.0
        depth_map = cv2.resize(raw_depth_map, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
        h, w = depth_map.shape

        # 1. Geometric projection
        world_to_cam = np.linalg.inv(pose)
        points_homo = np.hstack((points_3d, np.ones((num_points, 1))))
        points_cam = (world_to_cam @ points_homo.T).T

        valid_z_mask = points_cam[:, 2] > 0.1
        points_2d_homo = (k @ points_cam[valid_z_mask, :3].T).T
        u = (points_2d_homo[:, 0] / points_2d_homo[:, 2]).astype(np.int32)
        v = (points_2d_homo[:, 1] / points_2d_homo[:, 2]).astype(np.int32)
        z_point = points_cam[valid_z_mask, 2]

        valid_uv_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid_uv_mask]
        v = v[valid_uv_mask]
        z_point = z_point[valid_uv_mask]
        original_indices = np.where(valid_z_mask)[0][valid_uv_mask]

        # 2. Z-buffering
        z_depth_map = depth_map[v, u]
        is_visible = (np.abs(z_point - z_depth_map) < 0.10) & (z_depth_map > 0.1)
        visible_indices = original_indices[is_visible]
        visible_u = u[is_visible]
        visible_v = v[is_visible]

        # 3. Sparse consensus recording
        predicted_local_ids = pred_2d_mask[visible_v, visible_u]
        unique_masks_in_frame = np.unique(predicted_local_ids[predicted_local_ids > 0])

        for local_mask_id in unique_masks_in_frame:
            points_in_mask = visible_indices[predicted_local_ids == local_mask_id]
            if points_in_mask.size == 0:
                continue
            row_idx.extend(points_in_mask.tolist())
            col_idx.extend([global_mask_counter] * points_in_mask.size)
            global_mask_counter += 1

    print(f"Total unique 2D masks across frames: {global_mask_counter}")
    if global_mask_counter == 0:
        raise RuntimeError("No masks were bridged to 3D. Check teacher outputs and paths.")

    data = np.ones(len(row_idx), dtype=np.int8)
    point_mask_matrix = coo_matrix(
        (data, (np.asarray(row_idx), np.asarray(col_idx))),
        shape=(num_points, global_mask_counter),
        dtype=np.int8,
    ).tocsr()

    # Calculate how many points were literally never seen
    votes_per_point = np.array(point_mask_matrix.sum(axis=1)).flatten()
    unseen_points = np.sum(votes_per_point == 0)
    print(f"DEBUG: {unseen_points} points ({unseen_points/num_points*100:.1f}%) were NEVER seen by the camera.")

    n_components = min(SVD_COMPONENTS, max(2, global_mask_counter - 1))
    print(f"Compressing voting history with TruncatedSVD(n_components={n_components})...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    point_features = svd.fit_transform(point_mask_matrix)

    # This turns Euclidean distance into Cosine distance
    print("Normalizing feature vectors...")
    point_features = normalize(point_features, norm='l2', axis=1)
    point_features = point_features.astype(np.float32, copy=False)

    # Save the heavy SVD features for later use
    np.save(os.path.join(SCENE_DIR, f"svd_features_g{GRANULARITY}.npy"), point_features)

    # Points unseen by any mask are guaranteed low-confidence; skip them for clustering speed.
    seen_mask = votes_per_point > 0
    num_seen = int(np.sum(seen_mask))
    print(f"Clustering backend: {CLUSTER_BACKEND}")
    print(f"Seen points for clustering: {num_seen}/{num_points} ({num_seen/num_points*100:.1f}%)")
    if num_seen == 0:
        raise RuntimeError("No seen points found after bridging. Cannot cluster.")

    seen_features = point_features[seen_mask]
    seen_features = np.ascontiguousarray(seen_features, dtype=np.float32)

    def _fit_cpu_hdbscan(features: np.ndarray) -> np.ndarray:
        try:
            clusterer = HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                cluster_selection_epsilon=HDBSCAN_EPS,
                n_jobs=HDBSCAN_N_JOBS,
            )
        except TypeError:
            # Older sklearn builds may not expose n_jobs for HDBSCAN.
            clusterer = HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                cluster_selection_epsilon=HDBSCAN_EPS,
            )
        return clusterer.fit_predict(features)

    if CLUSTER_BACKEND == "cpu_hdbscan":
        print("Clustering points into 3D objects (CPU HDBSCAN)...")
        seen_labels = _fit_cpu_hdbscan(seen_features)
    elif CLUSTER_BACKEND == "gpu_hdbscan":
        print("Clustering points into 3D objects (GPU HDBSCAN via cuML)...")
        try:
            import cupy as cp
            from cuml.cluster import HDBSCAN as CuHDBSCAN
        except ImportError as e:
            raise RuntimeError(
                "CLUSTER_BACKEND=gpu_hdbscan requested, but cuML/CuPy is not installed."
            ) from e

        seen_features_gpu = cp.asarray(seen_features, dtype=cp.float32)
        gpu_clusterer = CuHDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_epsilon=HDBSCAN_EPS,
        )
        try:
            seen_labels = cp.asnumpy(gpu_clusterer.fit_predict(seen_features_gpu))
        except Exception as e:
            if not GPU_FALLBACK_TO_CPU:
                raise
            print(
                f"WARNING: GPU HDBSCAN failed ({type(e).__name__}: {e}). "
                "Falling back to CPU HDBSCAN for this scene."
            )
            seen_labels = _fit_cpu_hdbscan(seen_features)
    elif CLUSTER_BACKEND == "birch_fast":
        print("Clustering points into 3D objects (BIRCH fast approximation)...")
        birch = Birch(
            threshold=BIRCH_THRESHOLD,
            branching_factor=BIRCH_BRANCHING_FACTOR,
            n_clusters=None,
        )
        seen_labels = birch.fit_predict(seen_features)
    else:
        raise ValueError(
            f"Unknown CLUSTER_BACKEND='{CLUSTER_BACKEND}'. "
            "Expected one of: cpu_hdbscan, gpu_hdbscan, birch_fast."
        )

    explicit_pseudo_labels = np.full(num_points, -1, dtype=np.int32)
    explicit_pseudo_labels[seen_mask] = seen_labels.astype(np.int32, copy=False)

    print("Saving outputs...")
    labels_path = os.path.join(SCENE_DIR, f"chorus_instance_labels_g{GRANULARITY}.npy")
    np.save(labels_path, explicit_pseudo_labels)

    max_label = int(explicit_pseudo_labels.max())
    rng = np.random.default_rng(42)
    colors = rng.random((max(1, max_label + 1), 3))
    point_colors = np.zeros_like(points_3d)

    valid_mask = explicit_pseudo_labels >= 0
    if np.any(valid_mask):
        point_colors[valid_mask] = colors[explicit_pseudo_labels[valid_mask]]
    point_colors[~valid_mask] = [0.5, 0.5, 0.5]

    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    ply_path = os.path.join(SCENE_DIR, f"chorus_instance_result_g{GRANULARITY}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)

    print(f"Saved labels: {labels_path}")
    print(f"Saved colored cloud: {ply_path}")


if __name__ == "__main__":
    main()
