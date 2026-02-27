import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from config import (
    CHORUS_ORACLE_EVAL_SCRIPT,
    CHORUS_PROJECT_CLUSTER_SCRIPT,
    CHORUS_TEACHER_SCRIPT,
    DEFAULT_FRAME_SKIP,
    POC3_DIR,
)
from io_utils import (
    cleanup_intermediate_data,
    has_cluster_outputs,
    has_complete_teacher_outputs,
    has_oracle_outputs,
    stable_keep_full,
    verify_final_outputs,
    write_manifest,
)
from sens_extract import extract_rgbd


def _run_chorus_script(
    script_path: Path,
    scene_dir: Path,
    granularity: str | None = None,
    granularities: List[str] | None = None,
) -> None:
    env = os.environ.copy()
    env["SCENE_DIR"] = str(scene_dir)
    if granularity is not None:
        env["GRANULARITY"] = granularity
    if granularities is not None:
        env["GRANULARITIES"] = ",".join(granularities)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(POC3_DIR),
        env=env,
        check=True,
    )


def process_scene_chorus(
    scene_id: str,
    scene_dir: Path,
    granularities: List[str],
    keep_full_modulo: int,
    delete_intermediate: bool,
) -> Dict:
    started_dt = datetime.now(timezone.utc)
    manifest: Dict = {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "started_at": started_dt.isoformat(),
        "granularities": granularities,
        "status": "running",
        "steps": [],
        "keep_full_scene": stable_keep_full(scene_id, keep_full_modulo),
    }

    try:
        needs_scene_compute = any(
            not has_cluster_outputs(scene_dir, g) for g in granularities
        ) or (not has_oracle_outputs(scene_dir))

        if needs_scene_compute:
            extract_rgbd(scene_dir)
            manifest["steps"].append("extract_rgbd_done")
        else:
            manifest["steps"].append("skip_extract_all_outputs_already_done")

        for g in granularities:
            if has_cluster_outputs(scene_dir, g):
                manifest["steps"].append(f"skip_chorus_project_cluster_g{g}_already_done")
                continue

            if not has_complete_teacher_outputs(scene_dir, g, DEFAULT_FRAME_SKIP):
                _run_chorus_script(
                    CHORUS_TEACHER_SCRIPT, scene_dir, granularity=g, granularities=granularities
                )
                manifest["steps"].append(f"chorus_teacher_g{g}_done")
            else:
                manifest["steps"].append(f"skip_chorus_teacher_g{g}_already_done")

            _run_chorus_script(
                CHORUS_PROJECT_CLUSTER_SCRIPT, scene_dir, granularity=g, granularities=granularities
            )
            manifest["steps"].append(f"chorus_project_cluster_g{g}_done")

        if not has_oracle_outputs(scene_dir):
            _run_chorus_script(
                CHORUS_ORACLE_EVAL_SCRIPT,
                scene_dir,
                granularity=None,
                granularities=granularities,
            )
            manifest["steps"].append("chorus_oracle_eval_done")
        else:
            manifest["steps"].append("skip_chorus_oracle_eval_already_done")

        ok, missing = verify_final_outputs(scene_dir, granularities)
        manifest["verification"] = {"ok": ok, "missing_or_empty": missing}
        if not ok:
            raise RuntimeError(f"Final output verification failed: {missing}")

        if delete_intermediate and not manifest["keep_full_scene"]:
            manifest["cleanup"] = cleanup_intermediate_data(scene_dir, granularities)
        else:
            manifest["cleanup"] = {
                "deleted": [],
                "skipped": ["keep_full_scene=True or delete_intermediate=False"],
            }

        manifest["status"] = "done"
        finished_dt = datetime.now(timezone.utc)
        manifest["finished_at"] = finished_dt.isoformat()
        manifest["duration_seconds"] = round((finished_dt - started_dt).total_seconds(), 3)
        write_manifest(scene_dir, manifest)
        return manifest

    except Exception as exc:
        finished_dt = datetime.now(timezone.utc)
        manifest["status"] = "failed"
        manifest["finished_at"] = finished_dt.isoformat()
        manifest["duration_seconds"] = round((finished_dt - started_dt).total_seconds(), 3)
        manifest["error"] = {"message": str(exc), "traceback": traceback.format_exc()}
        write_manifest(scene_dir, manifest)
        return manifest

