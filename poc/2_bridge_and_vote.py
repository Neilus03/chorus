import os
import numpy as np
import open3d as o3d
import cv2

# --- CHORUS PoC CONFIG ---
SCENE_DIR = "scene0000_00"
FRAME_SKIP = 10
# The official ScanNet20 classes and their corresponding NYUv40 IDs
SCANNET20_MAPPING = {
    "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5,
    "sofa": 6, "table": 7, "door": 8, "window": 9, "bookshelf": 10,
    "picture": 11, "counter": 12, "desk": 14, "curtain": 16,
    "refrigerator": 24, "shower curtain": 28, "toilet": 33,
    "sink": 34, "bathtub": 36, "otherfurniture": 39
}
# Extract just the names for Grounding DINO to use as the text prompt
CLASSES = list(SCANNET20_MAPPING.keys())
NUM_CLASSES = len(CLASSES) + 1  # +1 to make room for the '0 = unlabelled' class!

# Color map for visualization
COLORS = np.random.rand(NUM_CLASSES, 3)
COLORS[0] = [0.5, 0.5, 0.5] # Unlabelled points are gray

print("Loading 3D Geometry...")
# Read the decimated mesh. Open3D will safely extract just the vertices as points.
mesh_path = os.path.join(SCENE_DIR, f"{SCENE_DIR}_vh_clean_2.ply")
pcd = o3d.io.read_point_cloud(mesh_path)
points_3d = np.asarray(pcd.points)
num_points = points_3d.shape[0]

# Initialize Chorus Consensus Module (Voting Matrix)
votes = np.zeros((num_points, NUM_CLASSES), dtype=np.int32)

# Load ScanNet Intrinsics (Color camera)
intrinsic_path = os.path.join(SCENE_DIR, "intrinsic", "intrinsic_color.txt")
K_4x4 = np.loadtxt(intrinsic_path)
K = K_4x4[:3, :3] # We only need the top-left 3x3 for projection

color_dir = os.path.join(SCENE_DIR, "color")
all_frames = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
target_frames = all_frames[::FRAME_SKIP]

print("Bridging 2D Masks to 3D Point Cloud...")
for frame_name in target_frames:
    frame_idx = frame_name.split('.')[0]

    pose_path = os.path.join(SCENE_DIR, "pose", f"{frame_idx}.txt")
    depth_path = os.path.join(SCENE_DIR, "depth", f"{frame_idx}.png")
    mask_path = os.path.join(SCENE_DIR, "sam_masks", f"{frame_idx}.npy")

    if not (os.path.exists(pose_path) and os.path.exists(depth_path) and os.path.exists(mask_path)):
        continue

    pose = np.loadtxt(pose_path) # Extrinsic matrix (cam to world)
    if np.isinf(pose).any() or np.isnan(pose).any():
        continue

    # Load mask and get its high-res dimensions (e.g., 968x1296)
    pred_2d_mask = np.load(mask_path)
    color_h, color_w = pred_2d_mask.shape

    # ScanNet depth is stored in millimeters. Convert to meters.
    raw_depth_map = cv2.imread(depth_path, -1) / 1000.0

    # ScanNet depth is stored in millimeters. Convert to meters.
    # We use INTER_NEAREST to avoid interpolating depths across object boundaries
    depth_map = cv2.resize(raw_depth_map, (color_w, color_h), interpolation=cv2.INTER_NEAREST)

    # Now h and w safely match the color camera and our mask!
    h, w = depth_map.shape

    # --- 1. GEOMETRIC PROJECTION  ---
    world_to_cam = np.linalg.inv(pose)
    points_homo = np.hstack((points_3d, np.ones((num_points, 1))))
    points_cam = (world_to_cam @ points_homo.T).T

    valid_z_mask = points_cam[:, 2] > 0.1 # Keep points in front of camera

    points_2d_homo = (K @ points_cam[valid_z_mask, :3].T).T
    u = (points_2d_homo[:, 0] / points_2d_homo[:, 2]).astype(int)
    v = (points_2d_homo[:, 1] / points_2d_homo[:, 2]).astype(int)
    z_point = points_cam[valid_z_mask, 2]

    valid_uv_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z_point = u[valid_uv_mask], v[valid_uv_mask], z_point[valid_uv_mask]
    original_indices = np.where(valid_z_mask)[0][valid_uv_mask]

    # --- 2. Z-BUFFERING (Occlusion Handling)  ---
    z_depth_map = depth_map[v, u]
    is_visible = (np.abs(z_point - z_depth_map) < 0.10) & (z_depth_map > 0.1)

    visible_indices = original_indices[is_visible]
    visible_u, visible_v = u[is_visible], v[is_visible]

    # --- 3. CONSENSUS ACCUMULATION ---
    predicted_classes = pred_2d_mask[visible_v, visible_u]
    valid_preds = predicted_classes > 0 # Ignore unlabelled pixels
    votes[visible_indices[valid_preds], predicted_classes[valid_preds]] += 1

print("Resolving Conflicts (Majority Voting)...")
pseudo_labels = np.argmax(votes, axis=1)
no_vote_mask = np.sum(votes, axis=1) == 0
pseudo_labels[no_vote_mask] = 0 # Re-assign 0 to points that were never seen/voted on

print("Visualizing Result...")
# Paint the point cloud
pcd.colors = o3d.utility.Vector3dVector(COLORS[pseudo_labels])
# Save the colored point cloud instead of trying to render a window
output_file = "chorus_poc_result.ply"
o3d.io.write_point_cloud(output_file, pcd)
print(f"Successfully saved explicit pseudo-labels to: {output_file}")
# Save the raw predictions for evaluation
np.save(os.path.join(SCENE_DIR, "chorus_pseudo_labels.npy"), pseudo_labels)