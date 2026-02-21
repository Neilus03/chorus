import os
import numpy as np
from plyfile import PlyData

# --- CHORUS PoC CONFIG ---
SCENE_DIR = "scene0000_00"

# ScanNet20 Mapping (Name -> NYUv40 ID)
SCANNET20_MAPPING = {
    "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5,
    "sofa": 6, "table": 7, "door": 8, "window": 9, "bookshelf": 10,
    "picture": 11, "counter": 12, "desk": 14, "curtain": 16,
    "refrigerator": 24, "shower curtain": 28, "toilet": 33,
    "sink": 34, "bathtub": 36, "otherfurniture": 39
}

print("Loading Ground Truth Labels...")
gt_path = os.path.join(SCENE_DIR, f"{SCENE_DIR}_vh_clean_2.labels.ply")
plydata = PlyData.read(gt_path)
# Extract the 'label' scalar field from the vertices
gt_labels = np.asarray(plydata.elements[0].data['label'])

print("Loading CHORUS Pseudo-Labels...")
pred_path = os.path.join(SCENE_DIR, "chorus_pseudo_labels.npy")
raw_preds = np.load(pred_path) # These are indices (1 to 20) based on our CLASSES list

# --- MAP PREDICTIONS TO NYUv40 IDs ---
# We must align our 1-20 indices to the official ScanNet 1-40 IDs
mapped_preds = np.zeros_like(raw_preds)
class_names = list(SCANNET20_MAPPING.keys())

for idx, class_name in enumerate(class_names):
    # Remember: in bridge_and_vote, 0 was unlabelled, so our classes are 1-indexed (idx + 1)
    nyu_id = SCANNET20_MAPPING[class_name]
    mapped_preds[raw_preds == (idx + 1)] = nyu_id

# --- EVALUATION (Intersection over Union) ---
print("\n--- RESULTS ---")
ious = []
valid_gt_mask = np.isin(gt_labels, list(SCANNET20_MAPPING.values()))

for class_name, nyu_id in SCANNET20_MAPPING.items():
    # Only evaluate on points where the GT actually contains this class,
    # or our model predicted this class where GT is valid
    gt_class_mask = (gt_labels == nyu_id)
    pred_class_mask = (mapped_preds == nyu_id) & valid_gt_mask

    intersection = np.sum(gt_class_mask & pred_class_mask)
    union = np.sum(gt_class_mask | pred_class_mask)

    if union > 0:
        iou = intersection / union
        ious.append(iou)
        print(f"{class_name.ljust(15)}: {iou*100:.2f}%")
    else:
        # Class doesn't exist in this specific room
        print(f"{class_name.ljust(15)}: N/A (Not in scene)")

print("-" * 20)
print(f"Mean IoU (mIoU) : {np.mean(ious)*100:.2f}%")
print(f"Overall Acc     : {np.sum(gt_labels[valid_gt_mask] == mapped_preds[valid_gt_mask]) / np.sum(valid_gt_mask)*100:.2f}%")