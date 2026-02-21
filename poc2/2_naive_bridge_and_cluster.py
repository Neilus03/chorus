import os
import cv2
import numpy as np
import open3d as o3d

# --- CONFIG ---
SCENE_DIR = "scene0000_00"
FRAME_SKIP = 10
GRANULARITY = 0.8
MASK_DIR = os.path.join(SCENE_DIR, f"unsam_masks_g{GRANULARITY}")
IOU_THRESHOLD = 0.5  # If a new mask overlaps an old one by 50%, merge them.

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

    # Initialize all points as -1 (unassigned)
    global_point_labels = np.full(num_points, -1, dtype=np.int32)
    next_global_id = 0

    print("Running Naive Greedy Projection...")
    for frame_name in target_frames:
        frame_idx = frame_name.split(".")[0]
        pose_path = os.path.join(SCENE_DIR, "pose", f"{frame_idx}.txt")
        depth_path = os.path.join(SCENE_DIR, "depth", f"{frame_idx}.png")
        mask_path = os.path.join(MASK_DIR, f"{frame_idx}.npy")

        if not (os.path.exists(pose_path) and os.path.exists(depth_path) and os.path.exists(mask_path)): continue

        pose = np.loadtxt(pose_path)
        if np.isinf(pose).any() or np.isnan(pose).any(): continue

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
        u, v, z_point = u[valid_uv_mask], v[valid_uv_mask], z_point[valid_uv_mask]
        original_indices = np.where(valid_z_mask)[0][valid_uv_mask]

        # 2. Z-buffering
        z_depth_map = depth_map[v, u]
        is_visible = (np.abs(z_point - z_depth_map) < 0.10) & (z_depth_map > 0.1)
        visible_indices = original_indices[is_visible]
        visible_u, visible_v = u[is_visible], v[is_visible]

        # 3. Greedy Integration
        predicted_local_ids = pred_2d_mask[visible_v, visible_u]
        unique_masks_in_frame = np.unique(predicted_local_ids[predicted_local_ids > 0])

        for local_mask_id in unique_masks_in_frame:
            points_in_mask = visible_indices[predicted_local_ids == local_mask_id]
            if len(points_in_mask) < 50: continue # Ignore tiny 2D noise

            # Check what labels these points already have in 3D
            existing_labels = global_point_labels[points_in_mask]
            valid_existing_labels = existing_labels[existing_labels != -1]

            if len(valid_existing_labels) > 0:
                # Find the most common existing 3D label this mask overlaps with
                most_common_label = np.bincount(valid_existing_labels).argmax()
                overlap_ratio = np.sum(valid_existing_labels == most_common_label) / len(points_in_mask)

                if overlap_ratio > IOU_THRESHOLD:
                    # Merge: Assign this label to the newly discovered points in this mask
                    global_point_labels[points_in_mask] = most_common_label
                else:
                    # Overlap too small, treat as new object
                    global_point_labels[points_in_mask] = next_global_id
                    next_global_id += 1
            else:
                # Completely new, unseen points
                global_point_labels[points_in_mask] = next_global_id
                next_global_id += 1

    print("Saving outputs...")
    # Save with a specific "naive" suffix so we don't overwrite your good ones!
    labels_path = os.path.join(SCENE_DIR, f"naive_instance_labels_g{GRANULARITY}.npy")
    np.save(labels_path, global_point_labels)

    # Visualization
    max_label = int(global_point_labels.max())
    rng = np.random.default_rng(42)
    colors = rng.random((max(1, max_label + 1), 3))
    point_colors = np.zeros_like(points_3d)

    valid_mask = global_point_labels >= 0
    if np.any(valid_mask):
        point_colors[valid_mask] = colors[global_point_labels[valid_mask]]
    point_colors[~valid_mask] = [0.5, 0.5, 0.5]

    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    ply_path = os.path.join(SCENE_DIR, f"naive_instance_result_g{GRANULARITY}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)

    print(f"Saved naive labels: {labels_path}")

if __name__ == "__main__":
    main()