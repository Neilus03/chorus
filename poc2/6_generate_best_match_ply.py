import os
import numpy as np
from pathlib import Path
import importlib
from plyfile import PlyData, PlyElement

# --- CONFIG ---
SCENE_DIR = "scene0000_00"
GRANULARITIES = [0.2, 0.5, 0.8]
OUT_PATH = os.path.join(SCENE_DIR, "chorus_best_match_potential.ply")

def main():
    # 1. Load Ground Truth and Proposals
    # We borrow the loader from your oracle script
    oracle_script = importlib.import_module("5_evaluate_combined_oracle")
    eval_level = importlib.import_module("4_evaluate_instances_by_level")

    scene_dir = Path(SCENE_DIR)
    scene_name = scene_dir.name

    print(f"Loading GT for {scene_name}...")
    gt_ids = eval_level.load_gt_instance_ids(scene_dir, scene_name)

    # Load original PLY to get coordinates (x, y, z)
    original_ply_path = scene_dir / f"{scene_name}_vh_clean_2.ply"
    plydata = PlyData.read(str(original_ply_path))
    vertices = plydata['vertex']

    print("Loading all granularity levels...")
    # Load proposals as a list of (mask, level_id)
    proposal_pool = []
    for g in GRANULARITIES:
        path = os.path.join(SCENE_DIR, f"chorus_instance_labels_g{g}.npy")
        if os.path.exists(path):
            labels = np.load(path)
            for inst_id in np.unique(labels):
                if inst_id == -1: continue
                proposal_pool.append(labels == inst_id)

    # 2. Build the "Best-Match" Label Array
    # Initialize with -1 (noise)
    best_match_labels = np.full(gt_ids.shape, -1, dtype=np.int32)

    print(f"Finding best matches for {len(np.unique(gt_ids)[1:])} GT objects...")

    gt_instances = np.unique(gt_ids)
    gt_instances = gt_instances[gt_instances > 0]

    # For each GT object, find the single best proposal in the whole pool
    for g_id in gt_instances:
        g_mask = (gt_ids == g_id)
        g_area = np.sum(g_mask)

        best_iou = 0
        best_proposal_mask = None

        for p_mask in proposal_pool:
            intersection = np.sum(p_mask & g_mask)
            if intersection == 0: continue

            union = np.sum(p_mask) + g_area - intersection
            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_proposal_mask = p_mask

        # If we found a decent match, burn it into our final label map
        # We use the g_id to keep the colors consistent with Ground Truth
        if best_iou > 0.1: # Minimal threshold to avoid noise
            best_match_labels[best_proposal_mask] = g_id

    # 3. Save as PLY with Colors
    print(f"Saving best-match PLY to {OUT_PATH}...")

    # Generate random colors for the IDs
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(max(gt_instances) + 1, 3), dtype=np.uint8)
    # Background color
    colors[0] = [50, 50, 50]

    vertex_colors = colors[np.where(best_match_labels == -1, 0, best_match_labels)]

    # Create the new PLY elements
    new_vertices = np.empty(len(vertices), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    new_vertices['x'] = vertices['x']
    new_vertices['y'] = vertices['y']
    new_vertices['z'] = vertices['z']
    new_vertices['red'] = vertex_colors[:, 0]
    new_vertices['green'] = vertex_colors[:, 1]
    new_vertices['blue'] = vertex_colors[:, 2]

    PlyData([PlyElement.describe(new_vertices, 'vertex')]).write(OUT_PATH)
    print("Done!")

if __name__ == "__main__":
    main()