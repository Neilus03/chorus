
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_scene_manifest(
    scene_id: str,
    scene_dir: Path,
    dataset: str,
    granularities: list[float],
    frame_skip: int,
    run_oracle_eval: bool,
    export_litept: bool,
    overwrite_existing: bool,
    auto_download_missing: bool,
    cleanup_after_success: bool,
    download_only: bool,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "dataset": dataset,
        "status": "running",
        "started_at": utcnow_iso(),
        "updated_at": utcnow_iso(),
        "finished_at": None,
        "config": {
            "granularities": [float(g) for g in granularities],
            "frame_skip": int(frame_skip),
            "run_oracle_eval": bool(run_oracle_eval),
            "export_litept": bool(export_litept),
            "overwrite_existing": bool(overwrite_existing),
            "auto_download_missing": bool(auto_download_missing),
            "cleanup_after_success": bool(cleanup_after_success),
            "download_only": bool(download_only),
        },
        "download": {
            "attempts": 0,
            "downloaded": False,
            "errors": [],
            "status": "not_started",
        },
        "verification": {
            "ok": False,
            "missing_outputs": [],
        },
        "cleanup": None,
        "summary_path": None,
        "reason": None,
        "error": None,
        "events": [],
    }


def add_manifest_event(
    manifest: dict[str, Any],
    phase: str,
    status: str,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    event = {
        "time": utcnow_iso(),
        "phase": phase,
        "status": status,
        "message": message,
    }
    if extra:
        event["extra"] = extra
    manifest.setdefault("events", []).append(event)
    manifest["updated_at"] = utcnow_iso()


def finalize_scene_manifest(
    manifest: dict[str, Any],
    status: str,
    reason: str | None = None,
    error: str | None = None,
    summary_path: str | None = None,
    cleanup: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> None:
    manifest["status"] = status
    manifest["reason"] = reason
    manifest["error"] = error
    manifest["summary_path"] = summary_path
    manifest["cleanup"] = cleanup
    if verification is not None:
        manifest["verification"] = verification
    manifest["updated_at"] = utcnow_iso()
    manifest["finished_at"] = utcnow_iso()


def write_scene_manifest(scene_dir: Path, manifest: dict[str, Any]) -> Path:
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / "scene_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path