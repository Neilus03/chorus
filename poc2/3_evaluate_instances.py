import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# --- CONFIG ---
SCENE_DIR = "scene0000_00"
GRANULARITY = 0.8
PRED_PATH = os.path.join(SCENE_DIR, f"naive_instance_labels_g{GRANULARITY}.npy")


def _load_instance_ids_from_ply(labels_ply: Path) -> np.ndarray | None:
    plydata = PlyData.read(str(labels_ply))
    vertex_data = plydata.elements[0].data
    for field in ("objectId", "instance", "instance_id"):
        if field in vertex_data.dtype.names:
            return np.asarray(vertex_data[field]).astype(np.int64)
    return None


def _load_instance_ids_from_aggregation(scene_dir: Path, scene_name: str, n_vertices: int) -> np.ndarray | None:
    seg_paths = [
        scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_name}_vh_clean.segs.json",
    ]
    agg_paths = [
        scene_dir / f"{scene_name}.aggregation.json",
        scene_dir / f"{scene_name}_vh_clean.aggregation.json",
    ]

    seg_path = next((p for p in seg_paths if p.exists()), None)
    agg_path = next((p for p in agg_paths if p.exists()), None)
    if seg_path is None or agg_path is None:
        return None

    with open(seg_path, "r", encoding="utf-8") as f:
        seg_json = json.load(f)
    with open(agg_path, "r", encoding="utf-8") as f:
        agg_json = json.load(f)

    seg_indices = np.asarray(seg_json.get("segIndices", []), dtype=np.int64)
    if seg_indices.shape[0] != n_vertices:
        raise RuntimeError(
            f"segIndices length ({seg_indices.shape[0]}) != num vertices ({n_vertices})"
        )

    seg_to_instance = {}
    for group in agg_json.get("segGroups", []):
        inst_id = int(group.get("objectId", group.get("id", -1)))
        if inst_id < 0:
            continue
        for seg_id in group.get("segments", []):
            seg_to_instance[int(seg_id)] = inst_id

    gt_instance_ids = np.zeros(n_vertices, dtype=np.int64)
    for i, seg_id in enumerate(seg_indices):
        gt_instance_ids[i] = seg_to_instance.get(int(seg_id), 0)

    return gt_instance_ids


def load_gt_instance_ids(scene_dir: Path, scene_name: str) -> np.ndarray:
    labels_ply = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    plydata = PlyData.read(str(labels_ply))
    n_vertices = len(plydata.elements[0].data)

    gt_instance_ids = _load_instance_ids_from_ply(labels_ply)
    if gt_instance_ids is not None:
        return gt_instance_ids

    gt_instance_ids = _load_instance_ids_from_aggregation(scene_dir, scene_name, n_vertices)
    if gt_instance_ids is not None:
        return gt_instance_ids

    raise RuntimeError(
        "Could not find GT instance ids in labels ply or aggregation+segments files."
    )


def main() -> None:
    scene_dir = Path(SCENE_DIR)
    scene_name = scene_dir.name

    print("Loading ground-truth instance IDs...")
    gt_instance_ids = load_gt_instance_ids(scene_dir, scene_name)

    print("Loading CHORUS instance pseudo-labels...")
    pred_instance_ids = np.load(PRED_PATH)

    if pred_instance_ids.shape[0] != gt_instance_ids.shape[0]:
        raise RuntimeError(
            f"Prediction length ({pred_instance_ids.shape[0]}) != GT length ({gt_instance_ids.shape[0]})"
        )

    valid_gt_mask = gt_instance_ids > 0
    gt_valid = gt_instance_ids[valid_gt_mask]
    pred_valid = pred_instance_ids[valid_gt_mask]

    print("\n--- CLASS-AGNOSTIC INSTANCE RESULTS ---")
    ari_score = adjusted_rand_score(gt_valid, pred_valid)
    nmi_score = normalized_mutual_info_score(gt_valid, pred_valid)

    print(f"Adjusted Rand Index (ARI)       : {ari_score:.4f}  (1.0 is perfect)")
    print(f"Normalized Mutual Info (NMI)    : {nmi_score:.4f}  (1.0 is perfect)")
    print("-" * 40)
    print(f"Evaluated granularity           : {GRANULARITY}")

    noise_pts = np.sum(pred_instance_ids == -1)
    print(f"Percentage of noise points: {noise_pts / len(pred_instance_ids) * 100:.2f}%")



if __name__ == "__main__":
    main()
