#!/usr/bin/env python3
"""Cluster-only benchmark for sklearn HDBSCAN on a saved dense feature matrix (.npy).

This is intended for repeatable timing against the same inputs produced by a scene run
(see CHORUS_SAVE_HDBSCAN_FEATURES in docs/hdbscan_benchmarking.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _safe_version(dist_name: str) -> str | None:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": _safe_version("numpy"),
        "scikit_learn_version": _safe_version("scikit-learn"),
    }


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _run_cluster_worker(args: argparse.Namespace) -> dict[str, Any]:
    from chorus.core.clustering.hdbscan_cluster import cluster_features

    features = np.load(args.features_path)
    wall_start = time.perf_counter()
    labels, stats = cluster_features(
        features=features,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    wall_time_s = float(time.perf_counter() - wall_start)

    return {
        "mode": "cluster-worker",
        "features_path": str(args.features_path),
        "wall_time_s": wall_time_s,
        "labels_sha256": hashlib.sha256(np.asarray(labels, dtype=np.int32).tobytes()).hexdigest(),
        "cluster_stats": stats,
        "runtime": _runtime_metadata(),
    }


def _run_worker_subprocess(
    *,
    features_path: Path,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="chorus_hdbscan_bench_") as tmp_dir:
        output_json = Path(tmp_dir) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_cluster_worker",
            "--features-path",
            str(features_path),
            "--min-cluster-size",
            str(min_cluster_size),
            "--min-samples",
            str(min_samples),
            "--cluster-selection-epsilon",
            str(cluster_selection_epsilon),
            "--output-json",
            str(output_json),
        ]
        subprocess.run(command, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
        with output_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def _cluster_only_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    runs = [
        _run_worker_subprocess(
            features_path=args.features_path,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
        )
        for _ in range(args.repeat)
    ]
    wall = [float(r["wall_time_s"]) for r in runs]
    fit = [float(r["cluster_stats"]["hdbscan_fit_predict_seconds"]) for r in runs]
    summary: dict[str, Any] = {
        "wall_time_median_s": _median_or_none(wall),
        "fit_predict_median_s": _median_or_none(fit),
    }
    if summary["wall_time_median_s"] and summary["fit_predict_median_s"]:
        summary["wall_over_fit_predict_ratio"] = (
            summary["wall_time_median_s"] / summary["fit_predict_median_s"]
        )
    return {
        "mode": "cluster-only",
        "features_path": str(args.features_path),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
        "repeat": args.repeat,
        "runs": runs,
        "summary": summary,
        "benchmark_runtime": _runtime_metadata(),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark sklearn HDBSCAN on a feature matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("_cluster_worker", help=argparse.SUPPRESS)
    worker.add_argument("--features-path", type=Path, required=True)
    worker.add_argument("--min-cluster-size", type=int, default=100)
    worker.add_argument("--min-samples", type=int, default=5)
    worker.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    worker.add_argument("--output-json", type=Path, required=True)

    cluster_only = subparsers.add_parser(
        "cluster-only",
        help="Time HDBSCAN fit_predict on a saved .npy feature matrix (median over repeats).",
    )
    cluster_only.add_argument("--features-path", type=Path, required=True)
    cluster_only.add_argument("--min-cluster-size", type=int, default=100)
    cluster_only.add_argument("--min-samples", type=int, default=5)
    cluster_only.add_argument("--cluster-selection-epsilon", type=float, default=0.1)
    cluster_only.add_argument("--repeat", type=int, default=3)
    cluster_only.add_argument("--output-json", type=Path, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "_cluster_worker":
        payload = _run_cluster_worker(args)
        _write_json(args.output_json, payload)
        return

    if args.command == "cluster-only":
        result = _cluster_only_benchmark(args)
        if args.output_json is not None:
            _write_json(args.output_json, result)
        print(json.dumps(result["summary"], indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
