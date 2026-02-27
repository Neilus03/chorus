import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class WandbReporter:
    def __init__(self, cfg: Dict[str, Any], run_config: Dict[str, Any], report_path: Path):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.report_path = report_path
        self.run = None
        self._wandb = None
        self._scene_table = None
        self.run_config = run_config

        if not self.enabled:
            return

        mode = str(self.cfg.get("mode", "online"))
        if mode == "disabled":
            self.enabled = False
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "W&B is enabled in config but 'wandb' is not installed. "
                "Install it or set wandb.enabled=false."
            ) from exc

        self._wandb = wandb
        run_name = self.run_config.get("run_name")
        run_id = self.run_config.get("run_id")
        self.run = wandb.init(
            project=self.cfg.get("project", "chorus-poc3"),
            entity=self.cfg.get("entity"),
            job_type=self.cfg.get("job_type", "streaming-scannet"),
            tags=self.cfg.get("tags", []),
            notes=self.cfg.get("notes"),
            mode=mode,
            resume=self.cfg.get("resume", "allow"),
            id=run_id,
            name=run_name,
            save_code=bool(self.cfg.get("save_code", True)),
            config=self.run_config,
        )

        self._scene_table = wandb.Table(
            columns=[
                "scene_idx",
                "scene_id",
                "status",
                "keep_full_scene",
                "duration_seconds",
                "steps_count",
                "deleted_items_count",
                "missing_outputs_count",
                "ap25_mean",
                "ap50_mean",
            ]
        )

        self.run.log({"run_report_path": str(report_path)}, step=0)

    def _load_oracle_metrics(self, scene_dir: Path) -> Dict[str, float]:
        metrics_file = scene_dir / "oracle_metrics.json"
        if not metrics_file.exists():
            return {
                "ap25_mean": 0.0,
                "ap50_mean": 0.0,
                "winner_g0_2": 0.0,
                "winner_g0_5": 0.0,
                "winner_g0_8": 0.0,
                "winner_no_match": 0.0,
            }
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        ap25 = []
        ap50 = []
        for _, row in data.items():
            if isinstance(row, dict):
                if "AP25" in row:
                    ap25.append(float(row["AP25"]))
                if "AP50" in row:
                    ap50.append(float(row["AP50"]))
        winner = (
            data.get("_extras", {})
            .get("winner_granularity_share", {})
        )
        return {
            "ap25_mean": float(sum(ap25) / len(ap25)) if ap25 else 0.0,
            "ap50_mean": float(sum(ap50) / len(ap50)) if ap50 else 0.0,
            "winner_g0_2": float(winner.get("g0.2", 0.0)),
            "winner_g0_5": float(winner.get("g0.5", 0.0)),
            "winner_g0_8": float(winner.get("g0.8", 0.0)),
            "winner_no_match": float(winner.get("no_match", 0.0)),
        }

    def log_scene(self, scene_idx: int, scene_id: str, manifest: Dict[str, Any], scene_dir: Path) -> None:
        if not self.enabled or self.run is None or self._scene_table is None:
            return

        status = manifest.get("status", "unknown")
        duration = float(manifest.get("duration_seconds", 0.0))
        keep_full = bool(manifest.get("keep_full_scene", False))
        steps_count = len(manifest.get("steps", []))
        deleted_count = len(manifest.get("cleanup", {}).get("deleted", []))
        missing_outputs_count = len(manifest.get("verification", {}).get("missing_or_empty", []))
        oracle = self._load_oracle_metrics(scene_dir)

        self._scene_table.add_data(
            scene_idx,
            scene_id,
            status,
            keep_full,
            duration,
            steps_count,
            deleted_count,
            missing_outputs_count,
            oracle["ap25_mean"],
            oracle["ap50_mean"],
        )

        self.run.log(
            {
                "scene/status": 1 if status == "done" else 0,
                "scene/duration_seconds": duration,
                "scene/steps_count": steps_count,
                "scene/deleted_items_count": deleted_count,
                "scene/missing_outputs_count": missing_outputs_count,
                "scene/ap25_mean": oracle["ap25_mean"],
                "scene/ap50_mean": oracle["ap50_mean"],
                "scene/winner_g0_2": oracle["winner_g0_2"],
                "scene/winner_g0_5": oracle["winner_g0_5"],
                "scene/winner_g0_8": oracle["winner_g0_8"],
                "scene/winner_no_match": oracle["winner_no_match"],
            },
            step=scene_idx,
        )

        if bool(self.cfg.get("log_manifest_artifact_per_scene", True)):
            manifest_path = scene_dir / "poc3_manifest.json"
            if manifest_path.exists():
                artifact = self._wandb.Artifact(
                    name=f"poc3-manifest-{scene_id}",
                    type="scene-manifest",
                    metadata={"scene_id": scene_id, "status": status},
                )
                artifact.add_file(str(manifest_path))
                self.run.log_artifact(artifact, aliases=["latest", status])

    def log_summary(self, summary: Dict[str, Any]) -> None:
        if not self.enabled or self.run is None or self._scene_table is None:
            return

        self.run.log({"scene_summary_table": self._scene_table})
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                self.run.summary[f"summary/{k}"] = v
        self.run.summary["summary/failed_scenes"] = summary.get("failed_scenes", [])

        if bool(self.cfg.get("log_report_artifact", True)) and self.report_path.exists():
            artifact = self._wandb.Artifact(
                name=f"poc3-run-report-{os.path.splitext(self.report_path.name)[0]}",
                type="run-report",
            )
            artifact.add_file(str(self.report_path))
            self.run.log_artifact(artifact, aliases=["latest"])

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()

