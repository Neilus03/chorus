#!/usr/bin/env python3
"""Ablation: full HDBSCAN vs subsample+1-NN on ScanNet++ (same SVD lift, teacher run once per scene)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.core.pipeline.project_cluster_stage import run_project_cluster_hdbscan_subsample_ablation
from chorus.core.quality.diagnostics import save_json
from chorus.datasets.scannetpp.download import (
    read_split_scene_ids,
    resolve_scannetpp_dataset_root,
)


def _map_scratch2_to_euler(path: Path) -> Path:
    path = Path(path)
    raw = str(path)
    if raw.startswith("/scratch2/nedela/"):
        mapped = Path(raw.replace("/scratch2/nedela/", "/cluster/work/igp_psr/nedela/", 1))
        if mapped.exists():
            return mapped
    return path


def _parse_granularities(raw: str) -> list[float]:
    return [float(g.strip()) for g in raw.split(",") if g.strip()]


def _parse_subsample_fractions(raw: str) -> list[float]:
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    for f in out:
        if not (0.0 < f <= 1.0):
            raise argparse.ArgumentTypeError(f"subsample fraction must be in (0, 1], got {f}")
    return out


def _read_scannetpp_scene_ids(
    *,
    dataset_root: Path,
    scene_list_file: Path | None,
    split: str | None,
    max_scenes: int | None,
) -> list[str]:
    """Same selection rules as orchestrators.streaming.read_scannetpp_scene_ids (lightweight import)."""
    if scene_list_file is not None:
        scene_ids = [
            line.strip()
            for line in scene_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif split is not None:
        scene_ids = read_split_scene_ids(split=split, dataset_root=dataset_root)
    else:
        data_root = dataset_root / "data"
        if not data_root.exists():
            scene_ids = []
        else:
            scene_ids = sorted([path.name for path in data_root.iterdir() if path.is_dir()])
    if max_scenes is not None:
        scene_ids = scene_ids[: int(max_scenes)]
    return scene_ids


def _safe_mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
    if not xs:
        return None
    return float(statistics.mean(xs))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation: compare full HDBSCAN vs subsampled HDBSCAN+1-NN on ScanNet++ scenes."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("CHORUS_SCANNETPP_DATA_ROOT", "/scratch2/nedela/scannetpp_data")),
    )
    p.add_argument("--split", type=str, default=os.environ.get("CHORUS_SCANNETPP_SPLIT", "nvs_sem_val"))
    p.add_argument("--scene-list-file", type=Path, default=None)
    p.add_argument("--max-scenes", type=int, default=3)
    p.add_argument("--granularities", type=str, default="0.2,0.5,0.8")
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument(
        "--scannetpp-eval-benchmark",
        type=str,
        default=os.environ.get("CHORUS_SCANNETPP_EVAL_BENCHMARK", "top100_instance"),
    )
    p.add_argument("--device", type=str, default=os.environ.get("CHORUS_DEVICE", "cuda:0"))
    p.add_argument("--svd-components", type=int, default=32)
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    p.add_argument(
        "--subsample-fractions",
        type=str,
        default="0.9,0.75,0.5,0.25",
        help="Comma-separated fractions in (0,1] of seen points: HDBSCAN cap = round(fraction * num_seen), "
        "floored at min_cluster_size. If cap >= num_seen, branch matches full (no re-cluster).",
    )
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the ablation report (default: report_dir / ablation_hdbscan_subsample_<ts>.json).",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for the JSON report when --output-json is not set.",
    )
    p.add_argument("--debug-first-n-frames", type=int, default=0)
    p.add_argument("--overwrite-teacher", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = resolve_scannetpp_dataset_root(dataset_root=_map_scratch2_to_euler(args.dataset_root))
    scenes_root = dataset_root / "data"
    granularities = _parse_granularities(args.granularities)

    scene_ids = _read_scannetpp_scene_ids(
        dataset_root=dataset_root,
        scene_list_file=args.scene_list_file,
        split=args.split if args.scene_list_file is None else None,
        max_scenes=args.max_scenes,
    )

    report_dir = args.report_dir or (dataset_root / "_chorus_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_json or (
        report_dir / f"ablation_hdbscan_subsample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print(f"dataset_root={dataset_root}")
    print(f"num_scenes={len(scene_ids)}")
    subsample_fractions = _parse_subsample_fractions(args.subsample_fractions)
    print(f"granularities={granularities}")
    print(f"subsample_fractions={subsample_fractions}")
    print(f"output_json={out_path}")

    if len(scene_ids) == 0:
        report = {
            "created_at": datetime.now().isoformat(),
            "config": {"dataset_root": str(dataset_root), "split": args.split},
            "summary": {"num_scenes_requested": 0, "note": "no scenes selected"},
            "scenes": [],
        }
        save_json(_json_safe(report), out_path)
        print(f"Wrote empty report to {out_path}")
        return

    from chorus.core.pipeline.teacher_stage import run_teacher_stage
    from chorus.core.teacher.unsamv2 import UnSAMv2Teacher
    from chorus.datasets.scannetpp.adapter import ScanNetPPSceneAdapter

    teacher = UnSAMv2Teacher(
        device=args.device,
        debug_first_n_frames=args.debug_first_n_frames,
        overwrite=args.overwrite_teacher,
    )

    rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for scene_id in scene_ids:
        scene_dir = scenes_root / scene_id
        adapter = ScanNetPPSceneAdapter(
            scene_root=scene_dir,
            eval_benchmark=args.scannetpp_eval_benchmark,
        )
        print(f"\n=== Scene {scene_id} ===")
        t_scene = time.perf_counter()
        try:
            adapter.prepare()
            teacher_outputs = run_teacher_stage(
                adapter=adapter,
                teacher=teacher,
                granularities=granularities,
                frame_skip=args.frame_skip,
            )
            by_g: dict[str, Any] = {}
            for teacher_output in teacher_outputs:
                if float(teacher_output.granularity) not in set(granularities):
                    continue
                t0 = time.perf_counter()
                ab = run_project_cluster_hdbscan_subsample_ablation(
                    adapter=adapter,
                    teacher_output=teacher_output,
                    frame_skip=args.frame_skip,
                    svd_components=args.svd_components,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    cluster_selection_epsilon=args.cluster_selection_epsilon,
                    subsample_fractions=subsample_fractions,
                    subsample_seed=args.subsample_seed,
                    eval_benchmark=args.scannetpp_eval_benchmark,
                )
                ab["wall_seconds_scene_granularity"] = float(time.perf_counter() - t0)
                by_g[f"g{teacher_output.granularity}"] = ab
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "ok",
                    "wall_seconds_scene": float(time.perf_counter() - t_scene),
                    "by_granularity": by_g,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "failed",
                    "error": repr(exc),
                    "wall_seconds_scene": float(time.perf_counter() - t_scene),
                }
            )
            print(f"FAILED {scene_id}: {exc}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    mean_by_granularity: dict[str, Any] = {}
    for r in ok_rows:
        for gkey, ab in (r.get("by_granularity") or {}).items():
            if gkey not in mean_by_granularity:
                mean_by_granularity[gkey] = {
                    "oracle_nmi_full": [],
                    "oracle_ari_full": [],
                    "full_hdbscan_cluster_wall_s": [],
                    "by_fraction": defaultdict(
                        lambda: {
                            "pseudolabel_ari_vs_full": [],
                            "oracle_nmi_delta_sub_minus_full": [],
                            "oracle_ari_delta_sub_minus_full": [],
                            "speedup_full_over_sub": [],
                            "subsample_hdbscan_cluster_wall_s": [],
                            "ap_delta_keys": defaultdict(list),
                        }
                    ),
                }
            bucket = mean_by_granularity[gkey]
            full = ab.get("full") or {}
            om = full.get("oracle_clustering_metrics") or {}
            if om.get("NMI") is not None:
                bucket["oracle_nmi_full"].append(float(om["NMI"]))
            if om.get("ARI") is not None:
                bucket["oracle_ari_full"].append(float(om["ARI"]))
            tsf = (full.get("timing_seconds") or {}).get("full_hdbscan_cluster_wall")
            if tsf is not None:
                bucket["full_hdbscan_cluster_wall_s"].append(float(tsf))

            for fk, fr in (ab.get("by_fraction") or {}).items():
                bfr = bucket["by_fraction"][fk]
                if fr.get("pseudolabel_ari_vs_full") is not None:
                    bfr["pseudolabel_ari_vs_full"].append(float(fr["pseudolabel_ari_vs_full"]))
                if fr.get("oracle_nmi_delta_sub_minus_full") is not None:
                    bfr["oracle_nmi_delta_sub_minus_full"].append(float(fr["oracle_nmi_delta_sub_minus_full"]))
                if fr.get("oracle_ari_delta_sub_minus_full") is not None:
                    bfr["oracle_ari_delta_sub_minus_full"].append(float(fr["oracle_ari_delta_sub_minus_full"]))
                sp = (fr.get("timing_seconds") or {}).get("speedup_full_over_sub")
                if sp is not None:
                    bfr["speedup_full_over_sub"].append(float(sp))
                sw = (fr.get("timing_seconds") or {}).get("subsample_hdbscan_cluster_wall")
                if sw is not None:
                    bfr["subsample_hdbscan_cluster_wall_s"].append(float(sw))
                dap = (fr.get("oracle_ap") or {}).get("delta_sub_minus_full") or {}
                for ap_k, ap_v in dap.items():
                    if ap_v is not None and isinstance(ap_v, (int, float)):
                        bfr["ap_delta_keys"][str(ap_k)].append(float(ap_v))

    def _finalize_fraction_block(raw: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for fk, fr in raw.items():
            out[fk] = {
                "mean_pseudolabel_ari_vs_full": _safe_mean(fr["pseudolabel_ari_vs_full"]),
                "mean_oracle_nmi_delta_sub_minus_full": _safe_mean(fr["oracle_nmi_delta_sub_minus_full"]),
                "mean_oracle_ari_delta_sub_minus_full": _safe_mean(fr["oracle_ari_delta_sub_minus_full"]),
                "mean_speedup_full_over_sub": _safe_mean(fr["speedup_full_over_sub"]),
                "mean_subsample_hdbscan_cluster_wall_s": _safe_mean(fr["subsample_hdbscan_cluster_wall_s"]),
                "mean_oracle_ap_delta_sub_minus_full": {
                    k: _safe_mean(v) for k, v in fr["ap_delta_keys"].items()
                },
            }
        return out

    finalized_gran: dict[str, Any] = {}
    for gkey, bucket in mean_by_granularity.items():
        finalized_gran[gkey] = {
            "mean_oracle_nmi_full": _safe_mean(bucket["oracle_nmi_full"]),
            "mean_oracle_ari_full": _safe_mean(bucket["oracle_ari_full"]),
            "mean_full_hdbscan_cluster_wall_s": _safe_mean(bucket["full_hdbscan_cluster_wall_s"]),
            "by_fraction": _finalize_fraction_block(bucket["by_fraction"]),
        }

    summary: dict[str, Any] = {
        "num_scenes_requested": len(scene_ids),
        "num_scenes_ok": len(ok_rows),
        "num_scenes_failed": len(rows) - len(ok_rows),
        "subsample_fractions": subsample_fractions,
        "subsample_seed": int(args.subsample_seed),
        "granularities": granularities,
        "eval_benchmark": args.scannetpp_eval_benchmark,
        "wall_seconds_total": float(time.perf_counter() - run_started),
        "mean_by_granularity": finalized_gran,
    }

    report = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "dataset_root": str(dataset_root),
            "split": args.split,
            "frame_skip": args.frame_skip,
            "svd_components": args.svd_components,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
            "subsample_fractions": subsample_fractions,
        },
        "summary": summary,
        "scenes": rows,
    }
    save_json(_json_safe(report), out_path)
    print("\n" + "=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote ablation report to {out_path}")


if __name__ == "__main__":
    main()
