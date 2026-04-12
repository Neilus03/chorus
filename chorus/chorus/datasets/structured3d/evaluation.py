from __future__ import annotations

from pathlib import Path
from typing import Any

from chorus.common.types import ClusterOutput
from chorus.datasets.structured3d.benchmark import (
    DEFAULT_STRUCTURED3D_EVAL_BENCHMARK,
    normalize_structured3d_eval_benchmark,
)
from chorus.eval.base import DatasetEvaluationHooks


def _safe_mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _bucket_key(name: str) -> str | None:
    lower = str(name).strip().lower()
    if lower.startswith("small"):
        return "small"
    if lower.startswith("medium"):
        return "medium"
    if lower.startswith("large"):
        return "large"
    return None


class Structured3DEvaluationHooks(DatasetEvaluationHooks):
    def __init__(self, eval_benchmark: str | None = None):
        self.eval_benchmark = normalize_structured3d_eval_benchmark(
            eval_benchmark or DEFAULT_STRUCTURED3D_EVAL_BENCHMARK
        )

    def scene_metric_fieldnames(self) -> list[str]:
        return [
            "oracle_nmi",
            "oracle_ari",
            "oracle_ap25_small",
            "oracle_ap50_small",
            "oracle_ap25_medium",
            "oracle_ap50_medium",
            "oracle_ap25_large",
            "oracle_ap50_large",
            "oracle_map_25_95_small",
            "oracle_map_25_95_medium",
            "oracle_map_25_95_large",
            "oracle_topk_iou025_r1",
            "oracle_topk_iou025_r3",
            "oracle_topk_iou025_r5",
            "oracle_topk_iou050_r1",
            "oracle_topk_iou050_r3",
            "oracle_topk_iou050_r5",
            "oracle_winner_share_g0_2",
            "oracle_winner_share_g0_5",
            "oracle_winner_share_g0_8",
            "oracle_winner_share_no_match",
        ]

    def evaluate_scene(
        self,
        adapter: Any,
        cluster_outputs: list[ClusterOutput],
    ) -> dict[str, Any] | None:
        if adapter.load_gt_instance_ids() is None:
            print(
                f"Structured3D oracle skipped: missing gt_instance_ids.npy under {adapter.scene_root} "
                "(no instance.png-derived labels during prepare)."
            )
            return {
                "eval_benchmark": self.eval_benchmark,
                "oracle_summary": None,
                "oracle_skipped_reason": "missing_gt_instance_ids",
            }

        from chorus.eval.scannet_oracle import evaluate_and_save_scannet_oracle

        oracle_summary = evaluate_and_save_scannet_oracle(
            adapter=adapter,
            cluster_outputs=cluster_outputs,
            eval_benchmark=self.eval_benchmark,
        )
        return {
            "eval_benchmark": self.eval_benchmark,
            "oracle_summary": {
                "eval_benchmark": oracle_summary["eval_benchmark"],
                "metrics_path": str(oracle_summary["metrics_path"]),
                "labels_path": str(oracle_summary["labels_path"]),
                "ply_path": str(oracle_summary["ply_path"]),
                "oracle_results": oracle_summary["oracle_results"],
                "additional_metrics": oracle_summary["additional_metrics"],
                "clustering_metrics": oracle_summary["clustering_metrics"],
            },
        }

    def flatten_scene_summary(self, scene_summary: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        oracle_summary = scene_summary.get("oracle_summary", {}) or {}
        if not oracle_summary:
            return flat

        clustering_metrics = oracle_summary.get("clustering_metrics", {}) or {}
        flat["oracle_nmi"] = clustering_metrics.get("NMI")
        flat["oracle_ari"] = clustering_metrics.get("ARI")

        oracle_results = oracle_summary.get("oracle_results", {}) or {}
        for bucket_name, bucket_metrics in oracle_results.items():
            bucket = _bucket_key(bucket_name)
            if bucket is None or not isinstance(bucket_metrics, dict):
                continue
            flat[f"oracle_ap25_{bucket}"] = bucket_metrics.get("AP25")
            flat[f"oracle_ap50_{bucket}"] = bucket_metrics.get("AP50")

        additional_metrics = oracle_summary.get("additional_metrics", {}) or {}

        map_by_bucket = additional_metrics.get("oracle_mAP_25_95_by_bucket", {}) or {}
        for bucket_name, value in map_by_bucket.items():
            bucket = _bucket_key(bucket_name)
            if bucket is None:
                continue
            flat[f"oracle_map_25_95_{bucket}"] = value

        topk = additional_metrics.get("topk_proposal_coverage", {}) or {}
        topk_025 = topk.get("iou_0.25", {}) or {}
        flat["oracle_topk_iou025_r1"] = topk_025.get("R_at_least_1")
        flat["oracle_topk_iou025_r3"] = topk_025.get("R_at_least_3")
        flat["oracle_topk_iou025_r5"] = topk_025.get("R_at_least_5")

        topk_050 = topk.get("iou_0.50", {}) or {}
        flat["oracle_topk_iou050_r1"] = topk_050.get("R_at_least_1")
        flat["oracle_topk_iou050_r3"] = topk_050.get("R_at_least_3")
        flat["oracle_topk_iou050_r5"] = topk_050.get("R_at_least_5")

        winner_share = additional_metrics.get("winner_granularity_share", {}) or {}
        for key, value in winner_share.items():
            safe_key = str(key).replace(".", "_")
            flat[f"oracle_winner_share_{safe_key}"] = value

        return flat

    def expected_output_paths(
        self,
        scene_dir: Path,
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[Path]:
        if not require_oracle:
            return []

        gt_path = scene_dir / "gt_instance_ids.npy"
        if not gt_path.exists():
            return []

        suffix = f"_{self.eval_benchmark}"
        return [
            scene_dir / f"oracle_metrics{suffix}.json",
            scene_dir / f"chorus_oracle_best_combined_labels{suffix}.npy",
            scene_dir / f"chorus_oracle_best_combined{suffix}.ply",
        ]

    def verify_summary(
        self,
        scene_dir: Path,
        summary: dict[str, Any],
        granularities: list[float],
        require_oracle: bool = True,
        require_training_pack: bool = True,
    ) -> list[str]:
        missing_reasons: list[str] = []
        if not require_oracle:
            return missing_reasons

        actual_benchmark = summary.get("eval_benchmark")
        if actual_benchmark != self.eval_benchmark:
            missing_reasons.append(
                f"summary eval_benchmark mismatch: expected {self.eval_benchmark}, found {actual_benchmark}"
            )

        if summary.get("oracle_skipped_reason") == "missing_gt_instance_ids":
            return missing_reasons

        if summary.get("oracle_summary") is None:
            missing_reasons.append("summary missing oracle_summary")

        return missing_reasons

    def running_summary_payload(
        self,
        done_rows: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        return {
            "summary/running_avg_oracle_nmi": _safe_mean([row.get("oracle_nmi") for row in done_rows]),
            "summary/running_avg_oracle_ari": _safe_mean([row.get("oracle_ari") for row in done_rows]),
        }

    def render_run_summary(
        self,
        run_summary: dict[str, Any],
        granularities: list[float],
    ) -> list[str]:
        scene_results = run_summary.get("scene_results", []) or []
        done_rows = [row for row in scene_results if row.get("status") in {"done", "skipped_done"}]
        if not done_rows:
            return []

        avg_nmi = _safe_mean([row.get("oracle_nmi") for row in done_rows])
        avg_ari = _safe_mean([row.get("oracle_ari") for row in done_rows])
        avg_ap50_large = _safe_mean([row.get("oracle_ap50_large") for row in done_rows])
        return [
            (
                f"Structured3D oracle ({self.eval_benchmark}): "
                f"NMI={_format_metric(avg_nmi)} | "
                f"ARI={_format_metric(avg_ari)} | "
                f"AP50-large={_format_metric(avg_ap50_large)}"
            )
        ]
