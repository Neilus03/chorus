from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class LocalTableReporter:
    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scene_csv_path = self.report_dir / f"scene_table_{timestamp}.csv"
        self.summary_json_path = self.report_dir / "latest_run_summary.json"

        self._fieldnames = [
            "scene_id",
            "status",
            "duration_seconds",
            "downloaded",
            "download_attempts",
            "cleanup_deleted_count",
            "avg_noise_fraction_seen",
            "avg_unseen_fraction",
            "avg_labeled_fraction_seen",
            "total_clusters_across_granularities",
            "total_teacher_masks_across_granularities",
            "oracle_nmi",
            "oracle_ari",
            "teacher_total_masks_g0.2",
            "teacher_total_masks_g0.5",
            "teacher_total_masks_g0.8",
            "clusters_g0.2",
            "clusters_g0.5",
            "clusters_g0.8",
            "noise_fraction_seen_g0.2",
            "noise_fraction_seen_g0.5",
            "noise_fraction_seen_g0.8",
            "unseen_fraction_g0.2",
            "unseen_fraction_g0.5",
            "unseen_fraction_g0.8",
            "labeled_fraction_seen_g0.2",
            "labeled_fraction_seen_g0.5",
            "labeled_fraction_seen_g0.8",
            "used_frames_g0.2",
            "used_frames_g0.5",
            "used_frames_g0.8",
            "num_2d_masks_total_g0.2",
            "num_2d_masks_total_g0.5",
            "num_2d_masks_total_g0.8",
            "reason",
            "error",
            "summary_path",
            "manifest_path",
        ]
        self._rows: list[dict[str, Any]] = []

        with self.scene_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()

    def log_scene(self, result: dict[str, Any]) -> None:
        cleanup = result.get("cleanup")
        cleanup_deleted_count = 0
        if isinstance(cleanup, dict):
            cleanup_deleted_count = len(cleanup.get("deleted", []))

        row = {
            "scene_id": result.get("scene_id"),
            "status": result.get("status"),
            "duration_seconds": result.get("duration_seconds", 0.0),
            "downloaded": bool(result.get("downloaded", False)),
            "download_attempts": int(result.get("download_attempts", 0)),
            "cleanup_deleted_count": cleanup_deleted_count,
            "avg_noise_fraction_seen": result.get("avg_noise_fraction_seen"),
            "avg_unseen_fraction": result.get("avg_unseen_fraction"),
            "avg_labeled_fraction_seen": result.get("avg_labeled_fraction_seen"),
            "total_clusters_across_granularities": result.get("total_clusters_across_granularities"),
            "total_teacher_masks_across_granularities": result.get("total_teacher_masks_across_granularities"),
            "oracle_nmi": result.get("oracle_nmi"),
            "oracle_ari": result.get("oracle_ari"),
            "teacher_total_masks_g0.2": result.get("teacher_total_masks_g0.2"),
            "teacher_total_masks_g0.5": result.get("teacher_total_masks_g0.5"),
            "teacher_total_masks_g0.8": result.get("teacher_total_masks_g0.8"),
            "clusters_g0.2": result.get("clusters_g0.2"),
            "clusters_g0.5": result.get("clusters_g0.5"),
            "clusters_g0.8": result.get("clusters_g0.8"),
            "noise_fraction_seen_g0.2": result.get("noise_fraction_seen_g0.2"),
            "noise_fraction_seen_g0.5": result.get("noise_fraction_seen_g0.5"),
            "noise_fraction_seen_g0.8": result.get("noise_fraction_seen_g0.8"),
            "unseen_fraction_g0.2": result.get("unseen_fraction_g0.2"),
            "unseen_fraction_g0.5": result.get("unseen_fraction_g0.5"),
            "unseen_fraction_g0.8": result.get("unseen_fraction_g0.8"),
            "labeled_fraction_seen_g0.2": result.get("labeled_fraction_seen_g0.2"),
            "labeled_fraction_seen_g0.5": result.get("labeled_fraction_seen_g0.5"),
            "labeled_fraction_seen_g0.8": result.get("labeled_fraction_seen_g0.8"),
            "used_frames_g0.2": result.get("used_frames_g0.2"),
            "used_frames_g0.5": result.get("used_frames_g0.5"),
            "used_frames_g0.8": result.get("used_frames_g0.8"),
            "num_2d_masks_total_g0.2": result.get("num_2d_masks_total_g0.2"),
            "num_2d_masks_total_g0.5": result.get("num_2d_masks_total_g0.5"),
            "num_2d_masks_total_g0.8": result.get("num_2d_masks_total_g0.8"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "summary_path": result.get("summary_path"),
            "manifest_path": result.get("manifest_path"),
        }
        self._rows.append(row)

        with self.scene_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def log_summary(self, summary: dict[str, Any]) -> None:
        payload = {
            "summary": summary,
            "scene_rows": self._rows,
            "scene_csv_path": str(self.scene_csv_path),
            "updated_at": datetime.now().isoformat(),
        }
        with self.summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def finish(self) -> None:
        pass