from __future__ import annotations

from pathlib import Path

from chorus.core.pipeline.project_cluster_stage import run_project_cluster_stage
from chorus.core.pipeline.teacher_stage import run_teacher_stage
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter
from chorus.eval.scannet_oracle import evaluate_and_save_scannet_oracle
from chorus.export.litept_pack import export_litept_scene_pack


def run_scene_pipeline(
    adapter: SceneAdapter,
    teacher: TeacherModel,
    granularities: list[float],
    frame_skip: int = 10,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    run_oracle_eval: bool = True,
    export_litept: bool = True,
) -> dict:
    print(f"Preparing scene: dataset={adapter.dataset_name}, scene={adapter.scene_id}")
    adapter.prepare()

    teacher_outputs = run_teacher_stage(
        adapter=adapter,
        teacher=teacher,
        granularities=granularities,
        frame_skip=frame_skip,
    )

    cluster_outputs = []
    for teacher_output in teacher_outputs:
        cluster_output = run_project_cluster_stage(
            adapter=adapter,
            teacher_output=teacher_output,
            frame_skip=frame_skip,
            svd_components=svd_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            save_outputs=True,
        )
        cluster_outputs.append(cluster_output)

    oracle_summary = None
    if run_oracle_eval and adapter.dataset_name == "scannet":
        oracle_summary = evaluate_and_save_scannet_oracle(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
        )

    litept_pack_dir = None
    if export_litept:
        litept_pack_dir = export_litept_scene_pack(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
        )

    summary = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "granularities": granularities,
        "teacher_outputs": [
            {
                "granularity": float(t.granularity),
                "num_mask_files": len(t.frame_mask_paths),
                "total_masks": int(t.total_masks),
                "output_dir": str(t.frame_mask_paths[0].parent) if len(t.frame_mask_paths) > 0 else None,
            }
            for t in teacher_outputs
        ],
        "cluster_outputs": [
            {
                "granularity": float(c.granularity),
                "labels_path": str(c.labels_path) if c.labels_path is not None else None,
                "ply_path": str(c.ply_path) if c.ply_path is not None else None,
                "stats": c.stats,
            }
            for c in cluster_outputs
        ],
        "oracle_summary": {
            "metrics_path": str(oracle_summary["metrics_path"]),
            "labels_path": str(oracle_summary["labels_path"]),
            "ply_path": str(oracle_summary["ply_path"]),
        }
        if oracle_summary is not None
        else None,
        "litept_pack_dir": str(litept_pack_dir) if litept_pack_dir is not None else None,
    }

    return summary