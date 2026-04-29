#!/usr/bin/env python3
"""Aggregate per-scene ScanNet oracle JSONs over a scene list (val macro means).

Reads ``oracle_metrics.json`` and ``oracle_metrics_scannet200.json`` under each
``<scans_root>/<scene_id>/`` and reports mean (and std) of per-bucket / clustering
values across scenes. Bucket names in files are percentile-based and differ by
scene; we map them to small / medium / large by prefix, matching
``ScanNetEvaluationHooks._bucket_key``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.core.quality.diagnostics import save_json


def _bucket_key(name: str) -> str | None:
    lower = str(name).strip().lower()
    if lower.startswith("small"):
        return "small"
    if lower.startswith("medium"):
        return "medium"
    if lower.startswith("large"):
        return "large"
    return None


def _read_scene_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _safe_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(v)


def _per_scene_instance_weighted_ap(row: dict[str, Any]) -> dict[str, float] | None:
    """For one scene: total GT-weighted mean of bucket AP@0.25 and AP@0.5 (IoU on proposals).

    ``sum_b (AP_b * Count_b) / sum_b Count_b`` (same as micro AP within the scene
    if each instance falls in one size bucket)."""
    buckets = row.get("buckets")
    if not buckets:
        return None
    out: dict[str, float] = {}
    for m in ("AP25", "AP50"):
        num = 0.0
        den = 0
        for bname in ("small", "medium", "large"):
            b = buckets.get(bname) or {}
            aps = b.get(m) or []
            counts = b.get("Count") or []
            if not aps or not counts:
                continue
            ap, c = float(aps[0]), int(counts[0])
            if c <= 0:
                continue
            num += ap * c
            den += c
        if den == 0:
            return None
        out[m] = num / den
    return out


def _micro_pooled_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | None] | None:
    """Val-wide micro: sum over all scenes and all buckets of (AP * Count) / total GT count."""
    num25, num50, den = 0.0, 0.0, 0
    for row in rows:
        buckets = row.get("buckets") or {}
        for bname in ("small", "medium", "large"):
            b = buckets.get(bname) or {}
            aps25 = b.get("AP25") or []
            aps50 = b.get("AP50") or []
            cts = b.get("Count") or []
            if not cts:
                continue
            c = int(cts[0])
            if c <= 0:
                continue
            a25 = float(aps25[0]) if aps25 else 0.0
            a50 = float(aps50[0]) if aps50 else 0.0
            num25 += a25 * c
            num50 += a50 * c
            den += c
    if den == 0:
        return None
    return {
        "AP25": num25 / den,
        "AP50": num50 / den,
        "total_gt_instances": float(den),
    }


def _ingest_per_scene(
    data: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "buckets": defaultdict(
            lambda: {
                "AP25": [],
                "AP50": [],
                "Count": [],
            }
        ),
        "clustering": {"NMI": None, "ARI": None},
        "map_25_95": defaultdict(list),
        "topk_025": defaultdict(list),
        "topk_050": defaultdict(list),
        "winner_share": defaultdict(list),
    }

    for k, v in data.items():
        if not k or str(k).startswith("_"):
            continue
        if not isinstance(v, dict):
            continue
        bucket = _bucket_key(k)
        if bucket is None:
            continue
        b = out["buckets"][bucket]
        if "AP25" in v and isinstance(v["AP25"], (int, float)):
            b["AP25"].append(float(v["AP25"]))
        if "AP50" in v and isinstance(v["AP50"], (int, float)):
            b["AP50"].append(float(v["AP50"]))
        if "Count" in v and isinstance(v["Count"], (int, float)):
            b["Count"].append(int(v["Count"]))

    clustering = data.get("_clustering") or {}
    if isinstance(clustering, dict):
        for key in ("NMI", "ARI"):
            if key in clustering and isinstance(clustering[key], (int, float)):
                out["clustering"][key] = float(clustering[key])

    extras = data.get("_extras") or {}
    if isinstance(extras, dict):
        mbb = extras.get("oracle_mAP_25_95_by_bucket") or {}
        if isinstance(mbb, dict):
            for name, val in mbb.items():
                bk = _bucket_key(name)
                if bk is not None and isinstance(val, (int, float)):
                    out["map_25_95"][bk].append(float(val))

        topk = extras.get("topk_proposal_coverage") or {}
        if isinstance(topk, dict):
            t025 = topk.get("iou_0.25") or {}
            t050 = topk.get("iou_0.50") or {}
            if isinstance(t025, dict):
                for rk, rv in t025.items():
                    if isinstance(rv, (int, float)):
                        out["topk_025"][str(rk)].append(float(rv))
            if isinstance(t050, dict):
                for rk, rv in t050.items():
                    if isinstance(rv, (int, float)):
                        out["topk_050"][str(rk)].append(float(rv))

        ws = extras.get("winner_granularity_share") or {}
        if isinstance(ws, dict):
            for wk, wv in ws.items():
                if isinstance(wv, (int, float)):
                    out["winner_share"][str(wk).replace(".", "_")].append(float(wv))

    return out


def _finalize_macro(
    series: dict[str, list[float]],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for k, xs in series.items():
        m, s = _safe_mean_std(xs)
        out[k] = {"mean": m, "std": s, "n": float(len(xs))}
    return out


def _finalize_buckets(
    buckets: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bname in ("small", "medium", "large"):
        b = buckets.get(bname) or {}
        out_b: dict[str, Any] = {}
        for metric in ("AP25", "AP50"):
            m, s = _safe_mean_std(b.get(metric) or [])
            out_b[metric] = {"mean": m, "std": s, "n_scenes": len(b.get(metric) or [])}
        if b.get("Count"):
            m, s = _safe_mean_std([float(x) for x in b["Count"]])
            out_b["Count"] = {"mean": m, "std": s, "n_scenes": len(b["Count"])}
        result[bname] = out_b
    return result


def _merge_scene_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_buckets: Any = defaultdict(
        lambda: {"AP25": [], "AP50": [], "Count": []}
    )
    nmi: list[float] = []
    ari: list[float] = []
    map_25_95: Any = defaultdict(list)
    topk_025: Any = defaultdict(list)
    topk_050: Any = defaultdict(list)
    winner: Any = defaultdict(list)

    for row in rows:
        for bname in ("small", "medium", "large"):
            b = (row.get("buckets") or {}).get(bname) or {}
            d = merged_buckets[bname]
            for m in ("AP25", "AP50"):
                src = b.get(m) or []
                if src:
                    d[m].extend(src)
            c = b.get("Count") or []
            if c:
                d["Count"].extend(c)

        cl = row.get("clustering") or {}
        if cl.get("NMI") is not None:
            nmi.append(float(cl["NMI"]))
        if cl.get("ARI") is not None:
            ari.append(float(cl["ARI"]))

        for bk, vs in (row.get("map_25_95") or {}).items():
            map_25_95[bk].extend(vs)
        for rk, vs in (row.get("topk_025") or {}).items():
            topk_025[rk].extend(vs)
        for rk, vs in (row.get("topk_050") or {}).items():
            topk_050[rk].extend(vs)
        for wk, vs in (row.get("winner_share") or {}).items():
            winner[wk].extend(vs)

    cl_out = {}
    for key, series in (("NMI", nmi), ("ARI", ari)):
        m, s = _safe_mean_std(series)
        cl_out[key] = {"mean": m, "std": s, "n_scenes": len(series)}

    return {
        "buckets": _finalize_buckets(merged_buckets),
        "clustering": cl_out,
        "oracle_mAP_25_95_by_bucket": _finalize_macro(map_25_95),
        "topk_proposal_iou_0.25": _finalize_macro(topk_025),
        "topk_proposal_iou_0.50": _finalize_macro(topk_050),
        "winner_granularity_share": _finalize_macro(winner),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Macro-aggregate ScanNet oracle_metrics*.json over a val scene list.",
    )
    p.add_argument(
        "--scans-root",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "CHORUS_SCANS_ROOT", "/scratch2/nedela/chorus_poc/scans"
            )
        ),
        help="Root with sceneXXXX_XX subfolders",
    )
    p.add_argument(
        "--scene-list-file",
        type=Path,
        required=True,
        help="Text file, one scene id per line (lines starting with # ignored)",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the aggregate report as JSON",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scene_ids = _read_scene_ids(args.scene_list_file)
    root = args.scans_root.resolve()

    benchmarks: list[tuple[str, str]] = [
        ("scannet20", ""),
        ("scannet200", "_scannet200"),
    ]

    report: dict[str, Any] = {
        "scans_root": str(root),
        "scene_list_file": str(args.scene_list_file.resolve()),
        "num_listed": len(scene_ids),
        "benchmarks": {},
    }

    for bname, suffix in benchmarks:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for sid in scene_ids:
            p = root / sid / f"oracle_metrics{suffix}.json"
            if not p.is_file():
                missing.append(sid)
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                missing.append(f"{sid} (error: {e})")
                continue
            if not isinstance(data, dict):
                continue
            rows.append(_ingest_per_scene(data))

        merged = _merge_scene_rows(rows)
        merged["num_scenes_with_metrics"] = len(rows)
        merged["num_missing"] = len(scene_ids) - len(rows)
        if missing:
            merged["missing_or_failed"] = missing[:200]
            if len(missing) > 200:
                merged["missing_note"] = f"truncated; total {len(missing)}"

        per_scene_ap25: list[float] = []
        per_scene_ap50: list[float] = []
        for row in rows:
            w = _per_scene_instance_weighted_ap(row)
            if w is None:
                continue
            per_scene_ap25.append(w["AP25"])
            per_scene_ap50.append(w["AP50"])

        m25, s25 = _safe_mean_std(per_scene_ap25)
        m50, s50 = _safe_mean_std(per_scene_ap50)
        overall_block: dict[str, Any] = {
            "overall_AP_instance_weighted": {
                "note": (
                    "Per scene: sum_b (AP_b * Count_b) / sum_b Count_b (buckets are size-stratified GT). "
                    "macro_mean_of_scenes: mean/std of that per-scene value. "
                    "micro_pooled: one ratio over all val GT instances."
                ),
                "macro_mean_of_scenes": {
                    "AP25": {
                        "mean": m25,
                        "std": s25,
                        "n_scenes": len(per_scene_ap25),
                    },
                    "AP50": {
                        "mean": m50,
                        "std": s50,
                        "n_scenes": len(per_scene_ap50),
                    },
                },
            }
        }
        micro = _micro_pooled_from_rows(rows)
        if micro is not None:
            overall_block["overall_AP_instance_weighted"]["micro_pooled_over_all_val_gt"] = micro

        report["benchmarks"][bname] = {**overall_block, **merged}

    text = json.dumps(report, indent=2)
    print(text)

    if args.output_json is not None:
        out = args.output_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_json(report, out)
        print(f"\nWrote: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
