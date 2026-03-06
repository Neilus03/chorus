from __future__ import annotations

from chorus.core.pipeline.project_cluster_stage import run_project_cluster_stage
from chorus.core.pipeline.teacher_stage import run_teacher_stage
from chorus.core.quality.diagnostics import save_json
from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter
from chorus.datasets.scannet.benchmark import (
    SCANNET_EVAL_BENCHMARK_20,
    parse_scannet_eval_benchmarks,
    primary_scannet_eval_benchmark,
)
from chorus.eval.scannet_oracle import evaluate_and_save_scannet_oracle
from chorus.export.litept_pack import export_litept_scene_pack


def run_scene_pipeline(
    adapter: SceneAdapter,
    teacher: TeacherModel,
    granularities: list[float],
    scannet_eval_benchmarks: list[str] | tuple[str, ...] | str | None = None,
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

    scene_intrinsic_metrics = compute_scene_intrinsic_metrics(cluster_outputs)

    oracle_summaries = None
    if run_oracle_eval and adapter.dataset_name == "scannet":
        benchmarks = parse_scannet_eval_benchmarks(scannet_eval_benchmarks)
        oracle_summaries = {}
        for eval_benchmark in benchmarks:
            oracle_summaries[eval_benchmark] = evaluate_and_save_scannet_oracle(
                adapter=adapter,
                cluster_outputs=cluster_outputs,
                eval_benchmark=eval_benchmark,
            )

    primary_eval_benchmark = primary_scannet_eval_benchmark(scannet_eval_benchmarks)
    primary_oracle_summary = None
    if oracle_summaries is not None:
        primary_oracle_summary = oracle_summaries.get(primary_eval_benchmark)
        if primary_oracle_summary is None and SCANNET_EVAL_BENCHMARK_20 in oracle_summaries:
            primary_oracle_summary = oracle_summaries[SCANNET_EVAL_BENCHMARK_20]
        if primary_oracle_summary is None and oracle_summaries:
            primary_oracle_summary = next(iter(oracle_summaries.values()))

    litept_pack_dir = None
    if export_litept:
        litept_pack_dir = export_litept_scene_pack(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
            teacher_name=teacher.__class__.__name__,
            projection_type="zbuffer_rgbd",
            embedding_type="truncated_svd",
            clustering_type="hdbscan",
            frame_skip=frame_skip,
            scene_intrinsic_metrics=scene_intrinsic_metrics,
        )

    summary = {
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "eval_benchmark": primary_eval_benchmark,
        "eval_benchmarks": parse_scannet_eval_benchmarks(scannet_eval_benchmarks),
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
        "scene_intrinsic_metrics": scene_intrinsic_metrics,
        "oracle_summary": {
            "eval_benchmark": primary_oracle_summary["eval_benchmark"],
            "metrics_path": str(primary_oracle_summary["metrics_path"]),
            "labels_path": str(primary_oracle_summary["labels_path"]),
            "ply_path": str(primary_oracle_summary["ply_path"]),
            "oracle_results": primary_oracle_summary["oracle_results"],
            "additional_metrics": primary_oracle_summary["additional_metrics"],
            "clustering_metrics": primary_oracle_summary["clustering_metrics"],
        }
        if primary_oracle_summary is not None
        else None,
        "oracle_summaries": {
            benchmark: {
                "eval_benchmark": summary["eval_benchmark"],
                "metrics_path": str(summary["metrics_path"]),
                "labels_path": str(summary["labels_path"]),
                "ply_path": str(summary["ply_path"]),
                "oracle_results": summary["oracle_results"],
                "additional_metrics": summary["additional_metrics"],
                "clustering_metrics": summary["clustering_metrics"],
            }
            for benchmark, summary in (oracle_summaries or {}).items()
        }
        if oracle_summaries is not None
        else None,
        "litept_pack_dir": str(litept_pack_dir) if litept_pack_dir is not None else None,
    }

    summary_path = adapter.scene_root / "scene_pipeline_summary.json"
    save_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)

    return summary