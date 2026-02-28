
from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbReporter:
    def __init__(
        self,
        enabled: bool = False,
        project: str = "chorus",
        entity: str | None = None,
        mode: str = "online",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.enabled = bool(enabled)
        self.run = None
        self._wandb = None
        self._scene_rows: list[dict[str, Any]] = []

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "W&B reporting was requested, but wandb is not installed."
            ) from exc

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=run_name,
            config=config or {},
        )

    def log_scene(self, result: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return

        cleanup = result.get("cleanup")
        cleanup_deleted_count = 0
        if isinstance(cleanup, dict):
            cleanup_deleted_count = len(cleanup.get("deleted", []))

        row = {
            "scene_id": result.get("scene_id"),
            "status": result.get("status"),
            "duration_seconds": result.get("duration_seconds", 0.0),
            "downloaded": int(bool(result.get("downloaded", False))),
            "download_attempts": int(result.get("download_attempts", 0)),
            "cleanup_deleted_count": cleanup_deleted_count,
            "reason": result.get("reason"),
        }
        self._scene_rows.append(row)

        self.run.log(
            {
                "scene/status_done": 1 if result.get("status") == "done" else 0,
                "scene/status_failed": 1 if result.get("status") == "failed" else 0,
                "scene/duration_seconds": result.get("duration_seconds", 0.0),
                "scene/downloaded": int(bool(result.get("downloaded", False))),
                "scene/download_attempts": int(result.get("download_attempts", 0)),
                "scene/cleanup_deleted_count": cleanup_deleted_count,
            }
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled or self.run is None or self._wandb is None:
            return

        table = self._wandb.Table(
            columns=[
                "scene_id",
                "status",
                "duration_seconds",
                "downloaded",
                "download_attempts",
                "cleanup_deleted_count",
                "reason",
            ]
        )

        for row in self._scene_rows:
            table.add_data(
                row["scene_id"],
                row["status"],
                row["duration_seconds"],
                row["downloaded"],
                row["download_attempts"],
                row["cleanup_deleted_count"],
                row["reason"],
            )

        self.run.log({"scene_table": table})

        self.run.summary["summary/done"] = summary.get("done", 0)
        self.run.summary["summary/skipped_done"] = summary.get("skipped_done", 0)
        self.run.summary["summary/failed"] = summary.get("failed", 0)
        self.run.summary["summary/failed_scenes"] = summary.get("failed_scenes", [])

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()