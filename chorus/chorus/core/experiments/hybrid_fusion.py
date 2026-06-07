from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chorus.common.types import ClusterOutput
from chorus.core.quality.diagnostics import save_json
from chorus.core.quality.intrinsic_metrics import compute_scene_intrinsic_metrics
from chorus.export.training_pack import export_training_scene_pack
from chorus.export.visualization import save_labeled_mesh_ply


@dataclass(frozen=True)
class HybridFusionConfig:
    granularities: tuple[float, ...] = (0.2, 0.5, 0.8)
    match_iou_threshold: float = 0.25
    bootstrap_containment_threshold: float = 0.50
    full_support_threshold: float = 0.10
    min_points: int = 30
    keep_unmatched_bootstrap: bool = True
    keep_unmatched_full: bool = True


@dataclass(frozen=True)
class SplitAwareFusionConfig:
    min_points: int = 30
    expansion_iou_threshold: float = 0.35
    expansion_bootstrap_containment_threshold: float = 0.70
    expansion_full_support_threshold: float = 0.25
    split_bootstrap_containment_threshold: float = 0.35
    split_full_support_threshold: float = 0.05
    split_min_bootstraps: int = 2
    keep_full_residual: bool = True
    keep_unmatched_full: bool = True


def _instance_sizes(labels: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(labels, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) >= 0}


def _label_indices(labels: np.ndarray, label_id: int) -> np.ndarray:
    return np.flatnonzero(labels == int(label_id))


def _contiguous_labels_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    num_points: int,
    min_points: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    candidates = sorted(
        candidates,
        key=lambda row: (
            float(row["priority"]),
            float(row.get("score", 0.0)),
            int(row["num_points"]),
        ),
        reverse=True,
    )

    fused = np.full(num_points, -1, dtype=np.int32)
    assigned = np.zeros(num_points, dtype=bool)
    kept: list[dict[str, Any]] = []

    for candidate in candidates:
        indices = np.asarray(candidate["indices"], dtype=np.int64)
        if indices.size == 0:
            continue
        write_indices = indices[~assigned[indices]]
        if write_indices.size < int(min_points):
            continue
        instance_id = len(kept)
        fused[write_indices] = instance_id
        assigned[write_indices] = True
        kept.append(
            {
                k: v
                for k, v in candidate.items()
                if k not in {"indices"}
            }
            | {
                "hybrid_instance_id": int(instance_id),
                "assigned_points": int(write_indices.size),
            }
        )

    return fused, kept


def fuse_full_and_bootstrap_labels(
    full_labels: np.ndarray,
    bootstrap_labels: np.ndarray,
    *,
    config: HybridFusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    full_labels = np.asarray(full_labels).reshape(-1)
    bootstrap_labels = np.asarray(bootstrap_labels).reshape(-1)
    if full_labels.shape != bootstrap_labels.shape:
        raise ValueError(
            f"full/bootstrap label shape mismatch: {full_labels.shape} vs {bootstrap_labels.shape}"
        )

    num_points = int(full_labels.shape[0])
    full_sizes = _instance_sizes(full_labels)
    bootstrap_sizes = _instance_sizes(bootstrap_labels)

    matched_full_ids: set[int] = set()
    candidates: list[dict[str, Any]] = []
    bootstrap_unmatched = 0
    bootstrap_expanded = 0

    for boot_id, boot_size in bootstrap_sizes.items():
        boot_indices = _label_indices(bootstrap_labels, boot_id)
        overlapping_full = full_labels[boot_indices]
        valid_full = overlapping_full[overlapping_full >= 0]

        best_full_id: int | None = None
        best_intersection = 0
        if valid_full.size > 0:
            ids, counts = np.unique(valid_full, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_full_id = int(ids[best_idx])
            best_intersection = int(counts[best_idx])

        use_full_expansion = False
        match_stats: dict[str, Any] = {}
        if best_full_id is not None:
            full_size = int(full_sizes[best_full_id])
            union = int(boot_size + full_size - best_intersection)
            iou = float(best_intersection / max(union, 1))
            boot_containment = float(best_intersection / max(boot_size, 1))
            full_support = float(best_intersection / max(full_size, 1))
            use_full_expansion = (
                iou >= float(config.match_iou_threshold)
                or (
                    boot_containment >= float(config.bootstrap_containment_threshold)
                    and full_support >= float(config.full_support_threshold)
                )
            )
            match_stats = {
                "matched_full_id": int(best_full_id),
                "full_bootstrap_iou": iou,
                "bootstrap_containment_in_full": boot_containment,
                "full_support_from_bootstrap": full_support,
                "intersection_points": int(best_intersection),
                "full_points": int(full_size),
                "bootstrap_points": int(boot_size),
            }

        if use_full_expansion and best_full_id is not None:
            matched_full_ids.add(best_full_id)
            bootstrap_expanded += 1
            indices = _label_indices(full_labels, best_full_id)
            candidates.append(
                {
                    "source": "bootstrap_anchor_full_expansion",
                    "priority": 3.0,
                    "score": float(match_stats["full_bootstrap_iou"]),
                    "num_points": int(indices.size),
                    "bootstrap_id": int(boot_id),
                    **match_stats,
                    "indices": indices,
                }
            )
        elif config.keep_unmatched_bootstrap:
            bootstrap_unmatched += 1
            candidates.append(
                {
                    "source": "bootstrap_only",
                    "priority": 2.0,
                    "score": 1.0,
                    "num_points": int(boot_indices.size),
                    "bootstrap_id": int(boot_id),
                    **match_stats,
                    "indices": boot_indices,
                }
            )

    full_unmatched = 0
    if config.keep_unmatched_full:
        for full_id, full_size in full_sizes.items():
            if full_id in matched_full_ids:
                continue
            if full_size < int(config.min_points):
                continue
            full_unmatched += 1
            indices = _label_indices(full_labels, full_id)
            candidates.append(
                {
                    "source": "full_only",
                    "priority": 1.0,
                    "score": 0.0,
                    "num_points": int(indices.size),
                    "full_id": int(full_id),
                    "indices": indices,
                }
            )

    fused, kept_candidates = _contiguous_labels_from_candidates(
        candidates,
        num_points=num_points,
        min_points=int(config.min_points),
    )
    stats = {
        "num_points": int(num_points),
        "num_full_instances": int(len(full_sizes)),
        "num_bootstrap_instances": int(len(bootstrap_sizes)),
        "num_candidates": int(len(candidates)),
        "num_kept_instances": int(len(kept_candidates)),
        "num_bootstrap_expanded_to_full": int(bootstrap_expanded),
        "num_bootstrap_kept_unmatched": int(bootstrap_unmatched),
        "num_full_kept_unmatched": int(full_unmatched),
        "num_labeled_points": int(np.count_nonzero(fused >= 0)),
        "labeled_points_fraction": float(np.count_nonzero(fused >= 0) / max(num_points, 1)),
        "config": {
            "match_iou_threshold": float(config.match_iou_threshold),
            "bootstrap_containment_threshold": float(config.bootstrap_containment_threshold),
            "full_support_threshold": float(config.full_support_threshold),
            "min_points": int(config.min_points),
            "keep_unmatched_bootstrap": bool(config.keep_unmatched_bootstrap),
            "keep_unmatched_full": bool(config.keep_unmatched_full),
        },
        "kept_candidates": kept_candidates,
    }
    return fused, stats


def fuse_split_aware_candidate_graph_labels(
    full_labels: np.ndarray,
    bootstrap_labels: np.ndarray,
    *,
    config: SplitAwareFusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    full_labels = np.asarray(full_labels).reshape(-1)
    bootstrap_labels = np.asarray(bootstrap_labels).reshape(-1)
    if full_labels.shape != bootstrap_labels.shape:
        raise ValueError(
            f"full/bootstrap label shape mismatch: {full_labels.shape} vs {bootstrap_labels.shape}"
        )

    num_points = int(full_labels.shape[0])
    full_sizes = _instance_sizes(full_labels)
    bootstrap_sizes = _instance_sizes(bootstrap_labels)

    full_to_boots: dict[int, list[dict[str, Any]]] = {full_id: [] for full_id in full_sizes}
    boot_to_fulls: dict[int, list[dict[str, Any]]] = {boot_id: [] for boot_id in bootstrap_sizes}
    valid = (full_labels >= 0) & (bootstrap_labels >= 0)
    if np.any(valid):
        pairs = np.stack([full_labels[valid], bootstrap_labels[valid]], axis=1).astype(np.int64)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (full_id_raw, boot_id_raw), intersection_raw in zip(unique_pairs, counts):
            full_id = int(full_id_raw)
            boot_id = int(boot_id_raw)
            intersection = int(intersection_raw)
            if intersection < int(config.min_points):
                continue
            full_size = int(full_sizes.get(full_id, 0))
            boot_size = int(bootstrap_sizes.get(boot_id, 0))
            union = int(full_size + boot_size - intersection)
            row = {
                "full_id": int(full_id),
                "bootstrap_id": int(boot_id),
                "intersection_points": int(intersection),
                "full_points": int(full_size),
                "bootstrap_points": int(boot_size),
                "iou": float(intersection / max(union, 1)),
                "bootstrap_containment_in_full": float(intersection / max(boot_size, 1)),
                "full_support_from_bootstrap": float(intersection / max(full_size, 1)),
            }
            full_to_boots.setdefault(full_id, []).append(row)
            boot_to_fulls.setdefault(boot_id, []).append(row)

    significant_full_to_boots: dict[int, list[dict[str, Any]]] = {}
    for full_id, rows in full_to_boots.items():
        significant = [
            row
            for row in rows
            if row["bootstrap_containment_in_full"] >= float(config.split_bootstrap_containment_threshold)
            and row["full_support_from_bootstrap"] >= float(config.split_full_support_threshold)
        ]
        significant_full_to_boots[full_id] = significant

    split_full_ids = {
        full_id
        for full_id, rows in significant_full_to_boots.items()
        if len(rows) >= int(config.split_min_bootstraps)
    }

    candidates: list[dict[str, Any]] = []
    matched_full_ids: set[int] = set()
    expanded_bootstraps = 0
    kept_bootstraps = 0

    for boot_id, boot_size in bootstrap_sizes.items():
        rows = boot_to_fulls.get(boot_id) or []
        best = max(rows, key=lambda row: row["intersection_points"], default=None)

        use_full_expansion = False
        if best is not None and int(best["full_id"]) not in split_full_ids:
            use_full_expansion = (
                best["iou"] >= float(config.expansion_iou_threshold)
                or (
                    best["bootstrap_containment_in_full"]
                    >= float(config.expansion_bootstrap_containment_threshold)
                    and best["full_support_from_bootstrap"]
                    >= float(config.expansion_full_support_threshold)
                )
            )

        if use_full_expansion and best is not None:
            full_id = int(best["full_id"])
            matched_full_ids.add(full_id)
            expanded_bootstraps += 1
            quality = (
                float(best["iou"])
                + 0.5 * float(best["bootstrap_containment_in_full"])
                + 0.5 * float(best["full_support_from_bootstrap"])
            )
            indices = _label_indices(full_labels, full_id)
            candidates.append(
                {
                    "source": "split_aware_full_expansion",
                    "priority": 4.0 + quality,
                    "score": quality,
                    "num_points": int(indices.size),
                    **best,
                    "indices": indices,
                }
            )
        else:
            kept_bootstraps += 1
            indices = _label_indices(bootstrap_labels, boot_id)
            quality = float(best["iou"]) if best is not None else 0.0
            candidates.append(
                {
                    "source": "split_aware_bootstrap_core",
                    "priority": 3.0 + quality,
                    "score": quality,
                    "num_points": int(indices.size),
                    "bootstrap_id": int(boot_id),
                    **({} if best is None else best),
                    "indices": indices,
                }
            )

    residual_fulls = 0
    full_only = 0
    for full_id, full_size in full_sizes.items():
        if full_id in matched_full_ids:
            continue
        if full_size < int(config.min_points):
            continue
        significant = significant_full_to_boots.get(full_id) or []
        if full_id in split_full_ids and config.keep_full_residual:
            residual_fulls += 1
            quality = max((float(row["full_support_from_bootstrap"]) for row in significant), default=0.0)
            indices = _label_indices(full_labels, full_id)
            candidates.append(
                {
                    "source": "split_aware_full_residual",
                    "priority": 2.0 + quality,
                    "score": quality,
                    "num_points": int(indices.size),
                    "full_id": int(full_id),
                    "num_significant_bootstrap_overlaps": int(len(significant)),
                    "indices": indices,
                }
            )
        elif config.keep_unmatched_full:
            full_only += 1
            quality = max(
                (float(row["full_support_from_bootstrap"]) for row in full_to_boots.get(full_id, [])),
                default=0.0,
            )
            indices = _label_indices(full_labels, full_id)
            candidates.append(
                {
                    "source": "split_aware_full_fill",
                    "priority": 1.5 + quality,
                    "score": quality,
                    "num_points": int(indices.size),
                    "full_id": int(full_id),
                    "indices": indices,
                }
            )

    fused, kept_candidates = _contiguous_labels_from_candidates(
        candidates,
        num_points=num_points,
        min_points=int(config.min_points),
    )

    stats = {
        "num_points": int(num_points),
        "num_full_instances": int(len(full_sizes)),
        "num_bootstrap_instances": int(len(bootstrap_sizes)),
        "num_candidate_nodes": int(len(candidates)),
        "num_kept_instances": int(len(kept_candidates)),
        "num_split_full_instances": int(len(split_full_ids)),
        "num_bootstrap_expanded_to_full": int(expanded_bootstraps),
        "num_bootstrap_kept_as_core": int(kept_bootstraps),
        "num_full_residual_candidates": int(residual_fulls),
        "num_full_fill_candidates": int(full_only),
        "num_labeled_points": int(np.count_nonzero(fused >= 0)),
        "labeled_points_fraction": float(np.count_nonzero(fused >= 0) / max(num_points, 1)),
        "config": {
            "min_points": int(config.min_points),
            "expansion_iou_threshold": float(config.expansion_iou_threshold),
            "expansion_bootstrap_containment_threshold": float(
                config.expansion_bootstrap_containment_threshold
            ),
            "expansion_full_support_threshold": float(config.expansion_full_support_threshold),
            "split_bootstrap_containment_threshold": float(config.split_bootstrap_containment_threshold),
            "split_full_support_threshold": float(config.split_full_support_threshold),
            "split_min_bootstraps": int(config.split_min_bootstraps),
            "keep_full_residual": bool(config.keep_full_residual),
            "keep_unmatched_full": bool(config.keep_unmatched_full),
        },
        "kept_candidates": kept_candidates,
    }
    return fused, stats


def make_cluster_output_from_labels(
    *,
    granularity: float,
    labels: np.ndarray,
    stats: dict[str, Any],
    labels_path: Path | None = None,
    ply_path: Path | None = None,
) -> ClusterOutput:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    seen_mask = labels >= 0
    cluster_ids = np.unique(labels[labels >= 0])
    output_stats = {
        "granularity": float(granularity),
        "num_points": int(labels.shape[0]),
        "num_clusters": int(cluster_ids.shape[0]),
        "num_labeled_points": int(np.count_nonzero(seen_mask)),
        "labeled_points_fraction": float(np.count_nonzero(seen_mask) / max(labels.shape[0], 1)),
        "num_seen_points": int(np.count_nonzero(seen_mask)),
        "seen_points_fraction": float(np.count_nonzero(seen_mask) / max(labels.shape[0], 1)),
        **stats,
    }
    return ClusterOutput(
        granularity=float(granularity),
        labels=labels,
        features=np.zeros((labels.shape[0], 1), dtype=np.float32),
        seen_mask=seen_mask,
        ply_path=ply_path,
        labels_path=labels_path,
        stats=output_stats,
    )


def run_hybrid_fusion_adapter_experiment(
    *,
    adapter: Any,
    full_scene_dir: Path,
    bootstrap_scene_dir: Path,
    output_scene_dir: Path,
    config: HybridFusionConfig,
    run_oracle_eval: bool = True,
    eval_benchmarks: tuple[str, ...] = ("structured3d_full",),
) -> dict[str, Any]:
    from chorus.eval.scannet_oracle import (
        evaluate_and_save_scannet_oracle,
        flatten_oracle_ap_bucket_metrics,
        flatten_oracle_map_bucket_metrics,
    )

    start = time.perf_counter()
    adapter.prepare()
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    cluster_outputs: list[ClusterOutput] = []
    by_granularity: dict[str, Any] = {}
    geometry_path = adapter.get_geometry_record().geometry_path

    for granularity in config.granularities:
        g_key = f"g{granularity}"
        full_labels_path = full_scene_dir / f"chorus_instance_labels_{g_key}.npy"
        bootstrap_labels_path = bootstrap_scene_dir / "fused" / f"labels_{g_key}.npy"
        if not full_labels_path.exists():
            raise FileNotFoundError(f"Missing full CHORUS labels: {full_labels_path}")
        if not bootstrap_labels_path.exists():
            raise FileNotFoundError(f"Missing bootstrap labels: {bootstrap_labels_path}")

        t0 = time.perf_counter()
        full_labels = np.load(full_labels_path)
        bootstrap_labels = np.load(bootstrap_labels_path)
        fused_labels, fusion_stats = fuse_full_and_bootstrap_labels(
            full_labels,
            bootstrap_labels,
            config=config,
        )
        fusion_seconds = time.perf_counter() - t0
        fusion_stats["fusion_seconds"] = float(fusion_seconds)

        labels_path = output_scene_dir / "hybrid" / f"labels_{g_key}.npy"
        ply_path = output_scene_dir / "hybrid" / f"result_{g_key}.ply"
        diagnostics_path = output_scene_dir / "hybrid" / f"diagnostics_{g_key}.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, fused_labels)
        save_labeled_mesh_ply(
            source_ply_path=geometry_path,
            labels=fused_labels,
            out_path=ply_path,
        )

        cluster_output = make_cluster_output_from_labels(
            granularity=float(granularity),
            labels=fused_labels,
            stats={"hybrid_fusion": fusion_stats},
            labels_path=labels_path,
            ply_path=ply_path,
        )
        cluster_outputs.append(cluster_output)
        save_json(
            {
                "full_labels_path": str(full_labels_path),
                "bootstrap_labels_path": str(bootstrap_labels_path),
                "hybrid_labels_path": str(labels_path),
                "hybrid_ply_path": str(ply_path),
                "stats": cluster_output.stats,
            },
            diagnostics_path,
        )
        by_granularity[g_key] = {
            "granularity": float(granularity),
            "full_labels_path": str(full_labels_path),
            "bootstrap_labels_path": str(bootstrap_labels_path),
            "hybrid_labels_path": str(labels_path),
            "hybrid_ply_path": str(ply_path),
            "diagnostics_path": str(diagnostics_path),
            "stats": cluster_output.stats,
        }

    scene_intrinsic_metrics = compute_scene_intrinsic_metrics(cluster_outputs)
    training_pack_dir = export_training_scene_pack(
        adapter=adapter,
        cluster_outputs=cluster_outputs,
        output_dir=output_scene_dir / "hybrid" / "training_pack",
        teacher_name="UnSAMv2FullBootstrapHybrid",
        projection_type="offline_full_plus_bootstrap",
        embedding_type="full_chorus_and_bootstrap_labels",
        clustering_type="hybrid_bootstrap_anchor_full_expansion",
        clustering_backend="offline_label_fusion",
        frame_skip=None,
        scene_intrinsic_metrics=scene_intrinsic_metrics,
    )

    evaluation_summary: dict[str, Any] | None = None
    if run_oracle_eval:
        oracle_summaries: dict[str, Any] = {}
        for benchmark in eval_benchmarks:
            oracle_summaries[benchmark] = evaluate_and_save_scannet_oracle(
                adapter=adapter,
                cluster_outputs=cluster_outputs,
                eval_benchmark=benchmark,
                save_artifacts=True,
                output_dir=output_scene_dir / "hybrid" / "oracle",
                gt_scene_root=adapter.scene_root if adapter.dataset_name == "scannet" else None,
            )
        primary = oracle_summaries.get(eval_benchmarks[0]) if eval_benchmarks else None
        evaluation_summary = {
            "eval_benchmarks": list(eval_benchmarks),
            "oracle_summaries": {
                k: {
                    **v,
                    "metrics_path": str(v["metrics_path"]) if v.get("metrics_path") is not None else None,
                    "labels_path": str(v["labels_path"]) if v.get("labels_path") is not None else None,
                    "ply_path": str(v["ply_path"]) if v.get("ply_path") is not None else None,
                }
                for k, v in oracle_summaries.items()
            },
        }
        if primary is not None and eval_benchmarks:
            evaluation_summary["oracle_summary"] = evaluation_summary["oracle_summaries"][eval_benchmarks[0]]

    flat_oracle: dict[str, Any] = {}
    if evaluation_summary is not None:
        primary = evaluation_summary.get("oracle_summary") or {}
        flat_oracle.update(flatten_oracle_ap_bucket_metrics(primary.get("oracle_results")))
        flat_oracle.update(flatten_oracle_map_bucket_metrics(primary.get("additional_metrics")))
        clustering = primary.get("clustering_metrics") or {}
        flat_oracle["oracle_nmi"] = clustering.get("NMI")
        flat_oracle["oracle_ari"] = clustering.get("ARI")

    summary = {
        "experiment": "full_chorus_plus_bootstrap_hybrid",
        "dataset": adapter.dataset_name,
        "scene_id": adapter.scene_id,
        "full_scene_dir": str(full_scene_dir),
        "bootstrap_scene_dir": str(bootstrap_scene_dir),
        "output_scene_dir": str(output_scene_dir),
        "config": {
            **config.__dict__,
            "granularities": [float(g) for g in config.granularities],
        },
        "by_granularity": by_granularity,
        "scene_intrinsic_metrics": scene_intrinsic_metrics,
        "training_pack_dir": str(training_pack_dir),
        "timing_seconds": {
            "fusion_total": float(sum(
                row["stats"]["hybrid_fusion"]["fusion_seconds"]
                for row in by_granularity.values()
            )),
            "total_wall": float(time.perf_counter() - start),
        },
        **flat_oracle,
    }
    if evaluation_summary is not None:
        summary.update(evaluation_summary)

    summary_path = output_scene_dir / "hybrid_experiment_summary.json"
    save_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary
