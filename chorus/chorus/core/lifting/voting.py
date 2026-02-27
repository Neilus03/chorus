from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def build_point_mask_matrix(
    point_assignments: list[np.ndarray],
    mask_assignments: list[np.ndarray],
    num_points: int,
):
    row_idx = []
    col_idx = []
    global_mask_counter = 0
    masks_per_frame = []

    for points_in_frame, local_mask_ids in zip(point_assignments, mask_assignments):
        frame_mask_count = 0
        unique_masks = np.unique(local_mask_ids[local_mask_ids > 0])

        for local_mask_id in unique_masks:
            pts = points_in_frame[local_mask_ids == local_mask_id]
            if pts.size == 0:
                continue
            row_idx.extend(pts.tolist())
            col_idx.extend([global_mask_counter] * pts.size)
            global_mask_counter += 1
            frame_mask_count += 1

        masks_per_frame.append(frame_mask_count)

    if global_mask_counter == 0:
        raise RuntimeError("No masks were bridged to 3D.")

    data = np.ones(len(row_idx), dtype=np.int8)
    matrix = coo_matrix(
        (data, (np.asarray(row_idx), np.asarray(col_idx))),
        shape=(num_points, global_mask_counter),
        dtype=np.int8,
    ).tocsr()

    stats = {
        "num_points": int(num_points),
        "num_2d_masks_total": int(global_mask_counter),
        "avg_masks_per_frame": float(np.mean(masks_per_frame)) if masks_per_frame else 0.0,
    }
    return matrix, stats
