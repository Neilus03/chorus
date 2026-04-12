from __future__ import annotations

from chorus.common.progress import phase_timer
from chorus.core.pipeline.project_cluster_stage import run_project_cluster_stage
from chorus.core.pipeline.teacher_stage import run_teacher_stage
from chorus.core.quality.diagnostics import save_json
from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter
from chorus.export.training_pack import export_training_scene_pack


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
    export_training_pack: bool = True,
) -> dict:
    print(f"Preparing scene: dataset={adapter.dataset_name}, scene={adapter.scene_id}")
    adapter.prepare()

    with phase_timer(f"Teacher stage: scene={adapter.scene_id}"):
        teacher_outputs = run_teacher_stage(
            adapter=adapter,
            teacher=teacher,
            granularities=granularities,
            frame_skip=frame_skip,
        )

    cluster_outputs = []
    for teacher_output in teacher_outputs:
        with phase_timer(
            f"Project+Cluster stage: scene={adapter.scene_id}, granularity={teacher_output.granularity}"
        ):
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

    scene_intrinsic_metrics = compute_scene_intrinsic_metrics(cluster_outputs)
    clustering_backend = None
    if cluster_outputs:
        clustering_backend = cluster_outputs[0].stats.get("hdbscan_backend")

    evaluation_hooks = adapter.get_evaluation_hooks()
    evaluation_summary = None
    if run_oracle_eval:
        with phase_timer(f"Oracle evaluation: scene={adapter.scene_id}"):
            evaluation_summary = evaluation_hooks.evaluate_scene(
                adapter=adapter,
                cluster_outputs=cluster_outputs,
            )

    training_pack_dir = None
    if export_training_pack:
        with phase_timer(f"Training pack export: scene={adapter.scene_id}"):
            training_pack_dir = export_training_scene_pack(
                adapter=adapter,
                cluster_outputs=cluster_outputs,
                teacher_name=teacher.__class__.__name__,
                projection_type="zbuffer_rgbd",
                embedding_type="truncated_svd",
                clustering_type="hdbscan",
                clustering_backend=clustering_backend,
                frame_skip=frame_skip,
                scene_intrinsic_metrics=scene_intrinsic_metrics,
            )

    summary = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "granularities": granularities,
        "clustering_type": "hdbscan",
        "clustering_backend": clustering_backend,
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
        "scene_intrinsic_metrics": scene_intrinsic_metrics,
        "training_pack_dir": str(training_pack_dir) if training_pack_dir is not None else None,
    }
    if evaluation_summary is not None:
        summary.update(evaluation_summary)

    summary_path = adapter.scene_root / "scene_pipeline_summary.json"
    save_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)

    return summary