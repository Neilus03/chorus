from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from chorus.common.io import verify_scene_completion_from_summary
from chorus.common.manifest import (
    add_manifest_event,
    finalize_scene_manifest,
    init_scene_manifest,
    write_scene_manifest,
)
from chorus.core.pipeline.scene_pipeline import run_scene_pipeline
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.scannet.adapter import ScanNetSceneAdapter
from chorus.datasets.scannet.download import download_scene, load_release_scene_ids
from chorus.datasets.scannet.prepare import is_rgbd_prepared
from chorus.orchestrators.cleanup import cleanup_scene_intermediates


def read_scene_ids(
    scans_root: Path,
    scene_list_file: Path | None = None,
    max_scenes: int | None = None,
    use_release_list: bool = False,
) -> list[str]:
    scans_root = Path(scans_root)

    if scene_list_file is not None:
        scene_ids = [
            line.strip()
            for line in scene_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif use_release_list:
        scene_ids = load_release_scene_ids()
    else:
        if not scans_root.exists():
            scene_ids = []
        else:
            scene_ids = sorted(
                [
                    p.name
                    for p in scans_root.iterdir()
                    if p.is_dir() and p.name.startswith("scene")
                ]
            )

    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]

    return scene_ids


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    mins = int(seconds // 60)
    secs = seconds - 60 * mins
    return f"{mins}m {secs:.1f}s"


def _print_scene_header(idx: int, total: int, scene_id: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{idx}/{total}] scene={scene_id}")
    print("=" * 90)


def _print_scene_result(result: dict[str, Any]) -> None:
    status = result["status"]
    scene_id = result["scene_id"]
    duration = _format_duration(result.get("duration_seconds", 0.0))

    print(f"status={status} | scene={scene_id} | duration={duration}")

    if result.get("reason"):
        print(f"reason={result['reason']}")

    if "downloaded" in result:
        print(f"downloaded={result['downloaded']}")

    if "download_attempts" in result:
        print(f"download_attempts={result['download_attempts']}")

    if result.get("cleanup"):
        print(f"cleanup={result['cleanup']}")

    if result.get("missing_outputs"):
        print("missing_outputs:")
        for item in result["missing_outputs"]:
            print(f"  - {item}")

    if result.get("error"):
        print(f"error={result['error']}")

    if result.get("summary_path"):
        print(f"summary_path={result['summary_path']}")

    if result.get("manifest_path"):
        print(f"manifest_path={result['manifest_path']}")


def _scene_has_raw_sens(scene_dir: Path) -> bool:
    sens_path = scene_dir / f"{scene_dir.name}.sens"
    return sens_path.exists()


def _ensure_scene_available(
    scene_id: str,
    scene_dir: Path,
    scans_root: Path,
    auto_download_missing: bool,
    max_download_retries: int,
) -> tuple[bool, bool, int, list[str], str | None]:
    if is_rgbd_prepared(scene_dir) or _scene_has_raw_sens(scene_dir):
        return True, False, 0, [], None

    if not auto_download_missing:
        return (
            False,
            False,
            0,
            [],
            "scene has neither prepared RGB-D data nor raw .sens source and auto-download is disabled",
        )

    errors: list[str] = []

    for attempt in range(1, max_download_retries + 1):
        try:
            download_scene(scene_id=scene_id, scans_root=scans_root, skip_existing=True)

            if is_rgbd_prepared(scene_dir) or _scene_has_raw_sens(scene_dir):
                return True, True, attempt, errors, None

            errors.append(
                f"attempt {attempt}: downloader returned but scene still has no prepared RGB-D and no .sens file"
            )
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

    return (
        False,
        False,
        max_download_retries,
        errors,
        f"download failed after {max_download_retries} attempts",
    )

def _flatten_scene_quality(scene_summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    scene_metrics = scene_summary.get("scene_intrinsic_metrics", {}) or {}
    flat["avg_noise_fraction_seen"] = scene_metrics.get("avg_noise_fraction_seen")
    flat["avg_unseen_fraction"] = scene_metrics.get("avg_unseen_fraction")
    flat["avg_labeled_fraction_seen"] = scene_metrics.get("avg_labeled_fraction_seen")
    flat["total_clusters_across_granularities"] = scene_metrics.get("total_clusters_across_granularities")

    teacher_by_g = {}
    for t in scene_summary.get("teacher_outputs", []) or []:
        g = f"g{t.get('granularity')}"
        teacher_by_g[g] = t

    cluster_by_g = {}
    for c in scene_summary.get("cluster_outputs", []) or []:
        g = f"g{c.get('granularity')}"
        cluster_by_g[g] = c

    by_g = scene_metrics.get("by_granularity", {}) or {}
    granularities = sorted(set(list(teacher_by_g.keys()) + list(cluster_by_g.keys()) + list(by_g.keys())))

    total_teacher_masks = 0
    for g in granularities:
        teacher_total_masks = (teacher_by_g.get(g, {}) or {}).get("total_masks")
        if teacher_total_masks is not None:
            total_teacher_masks += int(teacher_total_masks)
        flat[f"teacher_total_masks_{g}"] = teacher_total_masks

        cluster_stats = (cluster_by_g.get(g, {}) or {}).get("stats", {}) or {}
        flat[f"clusters_{g}"] = cluster_stats.get("num_clusters")
        flat[f"used_frames_{g}"] = cluster_stats.get("used_frames")
        flat[f"num_2d_masks_total_{g}"] = cluster_stats.get("num_2d_masks_total")

        g_metrics = by_g.get(g, {}) or {}
        flat[f"noise_fraction_seen_{g}"] = g_metrics.get("noise_fraction_seen")
        flat[f"unseen_fraction_{g}"] = g_metrics.get("unseen_points_fraction")
        flat[f"labeled_fraction_seen_{g}"] = g_metrics.get("labeled_points_fraction_seen")

    flat["total_teacher_masks_across_granularities"] = total_teacher_masks
    return flat

def run_streaming_scannet(
    scans_root: Path,
    scene_ids: list[str],
    teacher: TeacherModel,
    granularities: list[float],
    frame_skip: int = 10,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    run_oracle_eval: bool = True,
    export_litept: bool = True,
    overwrite_existing: bool = False,
    continue_on_error: bool = True,
    auto_download_missing: bool = True,
    cleanup_after_success: bool = True,
    download_only: bool = False,
    max_download_retries: int = 3,
    reporter: Any | None = None,
) -> dict[str, Any]:
    scans_root = Path(scans_root)



    run_summary: dict[str, Any] = {
        "scans_root": str(scans_root),
        "num_scenes_requested": len(scene_ids),
        "granularities": [float(g) for g in granularities],
        "frame_skip": int(frame_skip),
        "run_oracle_eval": bool(run_oracle_eval),
        "export_litept": bool(export_litept),
        "overwrite_existing": bool(overwrite_existing),
        "auto_download_missing": bool(auto_download_missing),
        "cleanup_after_success": bool(cleanup_after_success),
        "download_only": bool(download_only),
        "max_download_retries": int(max_download_retries),
        "done": 0,
        "skipped_done": 0,
        "failed": 0,
        "failed_scenes": [],
        "scene_results": [],
    }

    total = len(scene_ids)


    for idx, scene_id in enumerate(scene_ids, start=1):
        _print_scene_header(idx, total, scene_id)

        started = time.perf_counter()
        scene_dir = scans_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        manifest = init_scene_manifest(
            scene_id=scene_id,
            scene_dir=scene_dir,
            dataset="scannet",
            granularities=granularities,
            frame_skip=frame_skip,
            run_oracle_eval=run_oracle_eval,
            export_litept=export_litept,
            overwrite_existing=overwrite_existing,
            auto_download_missing=auto_download_missing,
            cleanup_after_success=cleanup_after_success,
            download_only=download_only,
        )
        add_manifest_event(manifest, phase="scene", status="started", message="scene processing started")
        manifest_path = write_scene_manifest(scene_dir, manifest)

        if not overwrite_existing:
            is_complete, existing_summary, missing = verify_scene_completion_from_summary(
                scene_dir=scene_dir,
                granularities=granularities,
                require_oracle=run_oracle_eval,
                require_litept=export_litept,
            )

            if is_complete:
                add_manifest_event(manifest, phase="skip", status="done", message="summary + outputs already verified")
                finalize_scene_manifest(
                    manifest,
                    status="skipped_done",
                    reason="summary + outputs already verified",
                    summary_path=str(scene_dir / "scene_pipeline_summary.json"),
                    verification={"ok": True, "missing_outputs": []},
                )
                write_scene_manifest(scene_dir, manifest)

                result = {
                    "scene_id": scene_id,
                    "scene_dir": str(scene_dir),
                    "status": "skipped_done",
                    "duration_seconds": time.perf_counter() - started,
                    "reason": "summary + outputs already verified",
                    "missing_outputs": [],
                    "summary_path": str(scene_dir / "scene_pipeline_summary.json"),
                    "existing_summary": existing_summary,
                    "downloaded": False,
                    "download_attempts": 0,
                    "manifest_path": str(manifest_path),
                }
                run_summary["done"] += 1
                run_summary["skipped_done"] += 1
                run_summary["scene_results"].append(result)
                _print_scene_result(result)
                if reporter is not None:
                    reporter.log_scene(result)
                continue

        add_manifest_event(manifest, phase="download", status="running", message="checking scene availability")
        available, downloaded, download_attempts, download_errors, availability_reason = _ensure_scene_available(
            scene_id=scene_id,
            scene_dir=scene_dir,
            scans_root=scans_root,
            auto_download_missing=auto_download_missing,
            max_download_retries=max_download_retries,
        )
        manifest["download"]["attempts"] = int(download_attempts)
        manifest["download"]["downloaded"] = bool(downloaded)
        manifest["download"]["errors"] = download_errors

        if available:
            manifest["download"]["status"] = "available"
            add_manifest_event(
                manifest,
                phase="download",
                status="done",
                message="scene is available",
                extra={"downloaded": downloaded, "attempts": download_attempts},
            )
            write_scene_manifest(scene_dir, manifest)
        else:
            manifest["download"]["status"] = "failed"
            add_manifest_event(
                manifest,
                phase="download",
                status="failed",
                message=availability_reason,
                extra={"errors": download_errors, "attempts": download_attempts},
            )
            finalize_scene_manifest(
                manifest,
                status="failed",
                reason=availability_reason,
                error=availability_reason,
            )
            write_scene_manifest(scene_dir, manifest)

            result = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "status": "failed",
                "duration_seconds": time.perf_counter() - started,
                "reason": availability_reason,
                "error": availability_reason,
                "downloaded": downloaded,
                "download_attempts": download_attempts,
                "manifest_path": str(manifest_path),
            }
            run_summary["failed"] += 1
            run_summary["failed_scenes"].append(scene_id)
            run_summary["scene_results"].append(result)
            _print_scene_result(result)
            if reporter is not None:
                reporter.log_scene(result)

            if not continue_on_error:
                break
            continue

        if download_only:
            add_manifest_event(
                manifest,
                phase="download_only",
                status="done",
                message="prefetch complete, pipeline skipped by request",
            )
            finalize_scene_manifest(
                manifest,
                status="downloaded_only",
                reason="download_only mode requested",
                verification={"ok": True, "missing_outputs": []},
            )
            write_scene_manifest(scene_dir, manifest)

            result = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "status": "downloaded_only",
                "duration_seconds": time.perf_counter() - started,
                "reason": "download_only mode requested",
                "downloaded": downloaded,
                "download_attempts": download_attempts,
                "manifest_path": str(manifest_path),
            }
            run_summary["done"] += 1
            run_summary["scene_results"].append(result)
            _print_scene_result(result)
            if reporter is not None:
                reporter.log_scene(result)
            continue

        quality_summary: dict[str, Any] = {}
        try:
            add_manifest_event(manifest, phase="pipeline", status="running", message="running CHORUS scene pipeline")
            write_scene_manifest(scene_dir, manifest)

            adapter = ScanNetSceneAdapter(scene_root=scene_dir)
            scene_summary = run_scene_pipeline(
                adapter=adapter,
                teacher=teacher,
                granularities=granularities,
                frame_skip=frame_skip,
                svd_components=svd_components,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                run_oracle_eval=run_oracle_eval,
                export_litept=export_litept,
            )

            quality_summary = _flatten_scene_quality(scene_summary)

            verified_ok, _, missing = verify_scene_completion_from_summary(
                scene_dir=scene_dir,
                granularities=granularities,
                require_oracle=run_oracle_eval,
                require_litept=export_litept,
            )

            if not verified_ok:
                add_manifest_event(
                    manifest,
                    phase="verification",
                    status="failed",
                    message="pipeline ran but verification failed",
                    extra={"missing_outputs": missing},
                )
                finalize_scene_manifest(
                    manifest,
                    status="failed",
                    reason="pipeline ran but verification failed",
                    summary_path=str(scene_dir / "scene_pipeline_summary.json"),
                    verification={"ok": False, "missing_outputs": missing},
                )
                write_scene_manifest(scene_dir, manifest)

                result = {
                    "scene_id": scene_id,
                    "scene_dir": str(scene_dir),
                    "status": "failed",
                    "duration_seconds": time.perf_counter() - started,
                    "reason": "pipeline ran but verification failed",
                    "missing_outputs": missing,
                    "summary_path": str(scene_dir / "scene_pipeline_summary.json"),
                    "downloaded": downloaded,
                    "download_attempts": download_attempts,
                    "manifest_path": str(manifest_path),
                    **quality_summary,
                }
                run_summary["failed"] += 1
                run_summary["failed_scenes"].append(scene_id)
                run_summary["scene_results"].append(result)
                _print_scene_result(result)
                if reporter is not None:
                    reporter.log_scene(result)

                if not continue_on_error:
                    break
                continue

            cleanup_info = None
            if cleanup_after_success:
                add_manifest_event(manifest, phase="cleanup", status="running", message="cleaning intermediate files")
                write_scene_manifest(scene_dir, manifest)

                cleanup_info = cleanup_scene_intermediates(
                    scene_dir=scene_dir,
                    granularities=granularities,
                    delete_rgbd=True,
                    delete_teacher_masks=True,
                    delete_svd_features=True,
                    delete_raw_scannet_files=True,
                )

                verified_ok_after_cleanup, _, missing_after_cleanup = verify_scene_completion_from_summary(
                    scene_dir=scene_dir,
                    granularities=granularities,
                    require_oracle=run_oracle_eval,
                    require_litept=export_litept,
                )

                if not verified_ok_after_cleanup:
                    add_manifest_event(
                        manifest,
                        phase="cleanup",
                        status="failed",
                        message="cleanup removed files required by final verification",
                        extra={"missing_outputs": missing_after_cleanup, "cleanup": cleanup_info},
                    )
                    finalize_scene_manifest(
                        manifest,
                        status="failed",
                        reason="cleanup removed files required by final verification",
                        summary_path=str(scene_dir / "scene_pipeline_summary.json"),
                        cleanup=cleanup_info,
                        verification={"ok": False, "missing_outputs": missing_after_cleanup},
                    )
                    write_scene_manifest(scene_dir, manifest)

                    result = {
                        "scene_id": scene_id,
                        "scene_dir": str(scene_dir),
                        "status": "failed",
                        "duration_seconds": time.perf_counter() - started,
                        "reason": "cleanup removed files required by final verification",
                        "missing_outputs": missing_after_cleanup,
                        "summary_path": str(scene_dir / "scene_pipeline_summary.json"),
                        "downloaded": downloaded,
                        "download_attempts": download_attempts,
                        "cleanup": cleanup_info,
                        "manifest_path": str(manifest_path),
                        **quality_summary,
                    }
                    run_summary["failed"] += 1
                    run_summary["failed_scenes"].append(scene_id)
                    run_summary["scene_results"].append(result)
                    _print_scene_result(result)
                    if reporter is not None:
                        reporter.log_scene(result)

                    if not continue_on_error:
                        break
                    continue

            add_manifest_event(manifest, phase="scene", status="done", message="scene processing completed")
            finalize_scene_manifest(
                manifest,
                status="done",
                reason=None,
                summary_path=scene_summary.get("summary_path"),
                cleanup=cleanup_info,
                verification={"ok": True, "missing_outputs": []},
            )
            write_scene_manifest(scene_dir, manifest)

            result = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "status": "done",
                "duration_seconds": time.perf_counter() - started,
                "reason": None,
                "missing_outputs": [],
                "summary_path": scene_summary.get("summary_path"),
                "downloaded": downloaded,
                "download_attempts": download_attempts,
                "cleanup": cleanup_info,
                "manifest_path": str(manifest_path),
                **quality_summary,
            }
            run_summary["done"] += 1
            run_summary["scene_results"].append(result)
            _print_scene_result(result)
            if reporter is not None:
                reporter.log_scene(result)

        except Exception as exc:
            add_manifest_event(
                manifest,
                phase="pipeline",
                status="failed",
                message="exception during pipeline execution",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            finalize_scene_manifest(
                manifest,
                status="failed",
                reason="exception during pipeline execution",
                error=f"{type(exc).__name__}: {exc}",
            )
            write_scene_manifest(scene_dir, manifest)

            result = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "status": "failed",
                "duration_seconds": time.perf_counter() - started,
                "reason": "exception during pipeline execution",
                "error": f"{type(exc).__name__}: {exc}",
                "downloaded": downloaded,
                "download_attempts": download_attempts,
                "manifest_path": str(manifest_path),
                **quality_summary,
            }
            run_summary["failed"] += 1
            run_summary["failed_scenes"].append(scene_id)
            run_summary["scene_results"].append(result)
            _print_scene_result(result)
            if reporter is not None:
                reporter.log_scene(result)

            if not continue_on_error:
                break

    if reporter is not None:
        reporter.log_summary(run_summary)

    return run_summary