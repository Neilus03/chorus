import os

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

try:
    # Newer sklearn variants may expose HDBSCAN here.
    from sklearn.cluster import HDBSCAN  # type: ignore
except Exception:  # pragma: no cover
    from hdbscan import HDBSCAN  # type: ignore


# --- CONFIG ---
SCENE_DIR = "scene0000_00"
FRAME_SKIP = 10
GRANULARITY = 0.8
MASK_DIR = os.path.join(SCENE_DIR, f"unsam_masks_g{GRANULARITY}")
SVD_COMPONENTS = 32


def main() -> None:
    print("Loading 3D geometry...")
    mesh_path = os.path.join(SCENE_DIR, f"{SCENE_DIR}_vh_clean_2.ply")
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

    # This turns Euclidean distance into Cosine distance!
    print("Normalizing feature vectors...")
    point_features = normalize(point_features, norm='l2', axis=1)

    print("Clustering points into 3D objects (HDBSCAN)...")
    clusterer = HDBSCAN(
        min_cluster_size=100,            # Force it to look for furniture-sized objects, not tennis balls
        min_samples=5,                   # KEEP THIS EXPLICITLY LOW to drop noise back to ~5-10%
        cluster_selection_epsilon=0.1,
        )
    explicit_pseudo_labels = clusterer.fit_predict(point_features)

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
