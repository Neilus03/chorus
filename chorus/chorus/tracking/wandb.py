from __future__ import annotations

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
            "oracle_nmi": result.get("oracle_nmi"),
            "oracle_ari": result.get("oracle_ari"),
            "oracle_ap25_small": result.get("oracle_ap25_small"),
            "oracle_ap50_small": result.get("oracle_ap50_small"),
            "oracle_ap25_medium": result.get("oracle_ap25_medium"),
            "oracle_ap50_medium": result.get("oracle_ap50_medium"),
            "oracle_ap25_large": result.get("oracle_ap25_large"),
            "oracle_ap50_large": result.get("oracle_ap50_large"),
            "oracle_map_25_95_small": result.get("oracle_map_25_95_small"),
            "oracle_map_25_95_medium": result.get("oracle_map_25_95_medium"),
            "oracle_map_25_95_large": result.get("oracle_map_25_95_large"),
            "oracle_topk_iou025_r1": result.get("oracle_topk_iou025_r1"),
            "oracle_topk_iou025_r3": result.get("oracle_topk_iou025_r3"),
            "oracle_topk_iou025_r5": result.get("oracle_topk_iou025_r5"),
            "oracle_topk_iou050_r1": result.get("oracle_topk_iou050_r1"),
            "oracle_topk_iou050_r3": result.get("oracle_topk_iou050_r3"),
            "oracle_topk_iou050_r5": result.get("oracle_topk_iou050_r5"),
            "oracle_winner_share_g0_2": result.get("oracle_winner_share_g0_2"),
            "oracle_winner_share_g0_5": result.get("oracle_winner_share_g0_5"),
            "oracle_winner_share_g0_8": result.get("oracle_winner_share_g0_8"),
            "oracle_winner_share_no_match": result.get("oracle_winner_share_no_match"),
            "oracle_nmi_scannet200": result.get("oracle_nmi_scannet200"),
            "oracle_ari_scannet200": result.get("oracle_ari_scannet200"),
            "oracle_ap25_small_scannet200": result.get("oracle_ap25_small_scannet200"),
            "oracle_ap50_small_scannet200": result.get("oracle_ap50_small_scannet200"),
            "oracle_ap25_medium_scannet200": result.get("oracle_ap25_medium_scannet200"),
            "oracle_ap50_medium_scannet200": result.get("oracle_ap50_medium_scannet200"),
            "oracle_ap25_large_scannet200": result.get("oracle_ap25_large_scannet200"),
            "oracle_ap50_large_scannet200": result.get("oracle_ap50_large_scannet200"),
        }
        self._scene_rows.append(row)

        payload = {
            "scene/status_done": 1 if result.get("status") == "done" else 0,
            "scene/status_failed": 1 if result.get("status") == "failed" else 0,
            "scene/duration_seconds": result.get("duration_seconds", 0.0),
            "scene/downloaded": int(bool(result.get("downloaded", False))),
            "scene/download_attempts": int(result.get("download_attempts", 0)),
            "scene/cleanup_deleted_count": cleanup_deleted_count,
            "scene/oracle_ap25_small": result.get("oracle_ap25_small"),
            "scene/oracle_ap50_small": result.get("oracle_ap50_small"),
            "scene/oracle_ap25_medium": result.get("oracle_ap25_medium"),
            "scene/oracle_ap50_medium": result.get("oracle_ap50_medium"),
            "scene/oracle_ap25_large": result.get("oracle_ap25_large"),
            "scene/oracle_ap50_large": result.get("oracle_ap50_large"),
            "scene/oracle_map_25_95_small": result.get("oracle_map_25_95_small"),
            "scene/oracle_map_25_95_medium": result.get("oracle_map_25_95_medium"),
            "scene/oracle_map_25_95_large": result.get("oracle_map_25_95_large"),
            "scene/oracle_topk_iou025_r1": result.get("oracle_topk_iou025_r1"),
            "scene/oracle_topk_iou025_r3": result.get("oracle_topk_iou025_r3"),
            "scene/oracle_topk_iou025_r5": result.get("oracle_topk_iou025_r5"),
            "scene/oracle_topk_iou050_r1": result.get("oracle_topk_iou050_r1"),
            "scene/oracle_topk_iou050_r3": result.get("oracle_topk_iou050_r3"),
            "scene/oracle_topk_iou050_r5": result.get("oracle_topk_iou050_r5"),
            "scene/oracle_winner_share_g0_2": result.get("oracle_winner_share_g0_2"),
            "scene/oracle_winner_share_g0_5": result.get("oracle_winner_share_g0_5"),
            "scene/oracle_winner_share_g0_8": result.get("oracle_winner_share_g0_8"),
            "scene/oracle_winner_share_no_match": result.get("oracle_winner_share_no_match"),
            "scene/oracle_nmi_scannet200": result.get("oracle_nmi_scannet200"),
            "scene/oracle_ari_scannet200": result.get("oracle_ari_scannet200"),
            "scene/oracle_ap25_small_scannet200": result.get("oracle_ap25_small_scannet200"),
            "scene/oracle_ap50_small_scannet200": result.get("oracle_ap50_small_scannet200"),
            "scene/oracle_ap25_medium_scannet200": result.get("oracle_ap25_medium_scannet200"),
            "scene/oracle_ap50_medium_scannet200": result.get("oracle_ap50_medium_scannet200"),
            "scene/oracle_ap25_large_scannet200": result.get("oracle_ap25_large_scannet200"),
            "scene/oracle_ap50_large_scannet200": result.get("oracle_ap50_large_scannet200"),
        }

        for key, value in result.items():
            if key in {
                "scene_id",
                "scene_dir",
                "status",
                "reason",
                "error",
                "summary_path",
                "manifest_path",
                "existing_summary",
                "cleanup",
                "missing_outputs",
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
                "oracle_nmi_scannet200",
                "oracle_ari_scannet200",
                "oracle_ap25_small_scannet200",
                "oracle_ap50_small_scannet200",
                "oracle_ap25_medium_scannet200",
                "oracle_ap50_medium_scannet200",
                "oracle_ap25_large_scannet200",
                "oracle_ap50_large_scannet200",
            }:
                continue

            if isinstance(value, (int, float)) and value is not None:
                payload[f"scene/{key}"] = value

        self.run.log(payload)


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
                "oracle_nmi_scannet200",
                "oracle_ari_scannet200",
                "oracle_ap25_small_scannet200",
                "oracle_ap50_small_scannet200",
                "oracle_ap25_medium_scannet200",
                "oracle_ap50_medium_scannet200",
                "oracle_ap25_large_scannet200",
                "oracle_ap50_large_scannet200",
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
                row["oracle_nmi"],
                row["oracle_ari"],
                row["oracle_ap25_small"],
                row["oracle_ap50_small"],
                row["oracle_ap25_medium"],
                row["oracle_ap50_medium"],
                row["oracle_ap25_large"],
                row["oracle_ap50_large"],
                row["oracle_map_25_95_small"],
                row["oracle_map_25_95_medium"],
                row["oracle_map_25_95_large"],
                row["oracle_topk_iou025_r1"],
                row["oracle_topk_iou025_r3"],
                row["oracle_topk_iou025_r5"],
                row["oracle_topk_iou050_r1"],
                row["oracle_topk_iou050_r3"],
                row["oracle_topk_iou050_r5"],
                row["oracle_winner_share_g0_2"],
                row["oracle_winner_share_g0_5"],
                row["oracle_winner_share_g0_8"],
                row["oracle_winner_share_no_match"],
                row["oracle_nmi_scannet200"],
                row["oracle_ari_scannet200"],
                row["oracle_ap25_small_scannet200"],
                row["oracle_ap50_small_scannet200"],
                row["oracle_ap25_medium_scannet200"],
                row["oracle_ap50_medium_scannet200"],
                row["oracle_ap25_large_scannet200"],
                row["oracle_ap50_large_scannet200"],
            )

        self.run.log({"scene_table": table})

        self.run.summary["summary/done"] = summary.get("done", 0)
        self.run.summary["summary/skipped_done"] = summary.get("skipped_done", 0)
        self.run.summary["summary/failed"] = summary.get("failed", 0)
        self.run.summary["summary/failed_scenes"] = summary.get("failed_scenes", [])

    def log_event(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log(payload)

    def set_summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        for k, v in payload.items():
            self.run.summary[k] = v

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()