import os
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import importlib

# --- CONFIG ---
SCENE_DIR = "scene0000_00"
GRANULARITY = 0.5

# 1. Load the pre-calculated heavy math
features_path = os.path.join(SCENE_DIR, f"svd_features_g{GRANULARITY}.npy")
print(f"Loading features from {features_path}...")
point_features = np.load(features_path)

# 2. Import your evaluation script logic
try:
    eval_script = importlib.import_module("3_evaluate_instances")
    gt_ids = eval_script.load_gt_instance_ids(Path(SCENE_DIR), SCENE_DIR)
except Exception as e:
    print(f"Error loading GT or eval script: {e}")
    exit()

valid_gt_mask = gt_ids > 0
gt_valid = gt_ids[valid_gt_mask]

# 3. The Expanded Grid Search
# We are testing smaller sizes (for pure geometry) and larger sizes/epsilons (for human scale)
min_cluster_sizes = [50, 100, 200, 300]
epsilons = [0.08, 0.10, 0.15, 0.20, 0.25]

print(f"\n{'Size':<4} | {'Eps':<4} | {'Clsts':<5} | {'Noise':<5} | {'ARI':<6} | {'NMI':<6} | {'mIoU':<6} | {'AP@25':<6} | {'AP@50':<6}")
print("-" * 75)

for min_size in min_cluster_sizes:
    for eps in epsilons:
        # Run HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_size,
            min_samples=5,
            cluster_selection_epsilon=eps
        )
        labels = clusterer.fit_predict(point_features)

        # Calculate cluster stats
        num_clusters = int(labels.max()) + 1
        noise_pct = np.sum(labels == -1) / len(labels) * 100

        # Calculate Clustering Metrics (ARI & NMI)
        pred_valid = labels[valid_gt_mask]
        if len(np.unique(pred_valid)) > 1:
            ari = adjusted_rand_score(gt_valid, pred_valid)
            nmi = normalized_mutual_info_score(gt_valid, pred_valid)
        else:
            ari, nmi = 0.0, 0.0

        # Calculate Detection Metrics (mIoU, AP25, AP50) using your exact function!
        ap_metrics = eval_script.evaluate_class_agnostic_ap_and_miou(gt_ids, labels)
        miou = ap_metrics['mIoU']
        ap25 = ap_metrics['AP25']
        ap50 = ap_metrics['AP50']

        print(f"{min_size:<4} | {eps:<4.2f} | {num_clusters:<5} | {noise_pct:<4.1f}% | {ari:<6.4f} | {nmi:<6.4f} | {miou:<6.4f} | {ap25:<6.4f} | {ap50:<6.4f}")

print("\n" + "="*75)
print("Grid Search Complete. Look for the row that balances ARI/NMI and AP@50!")