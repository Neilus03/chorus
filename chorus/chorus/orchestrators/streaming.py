from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from chorus.common.io import verify_scene_completion_from_summary
from chorus.common.manifest import (
    add_manifest_event,
    finalize_scene_manifest,
    init_scene_manifest,
    utcnow_iso,
    write_scene_manifest,
)
from chorus.core.pipeline.scene_pipeline import run_scene_pipeline
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter
from chorus.datasets.scannet.adapter import ScanNetSceneAdapter
from chorus.datasets.scannet.benchmark import parse_scannet_eval_benchmarks
from chorus.datasets.scannet.download import (
    download_scene as download_scannet_scene,
    load_release_scene_ids,
)
from chorus.datasets.scannet.prepare import is_rgbd_prepared
from chorus.datasets.scannetpp.adapter import ScanNetPPSceneAdapter
from chorus.datasets.scannetpp.benchmark import normalize_scannetpp_eval_benchmark
from chorus.datasets.scannetpp.download import (
    download_scene as download_scannetpp_scene,
    read_split_scene_ids as read_scannetpp_split_scene_ids,
    resolve_scannetpp_dataset_root,
)
from chorus.datasets.scannetpp.prepare import (
    has_raw_scene_assets as scannetpp_has_raw_scene_assets,
    is_prepared as is_scannetpp_prepared,
)
from chorus.eval.base import DatasetEvaluationHooks
from chorus.orchestrators.cleanup import cleanup_scene_intermediates

SceneAdapterFactory = Callable[[Path], SceneAdapter]
SceneAvailabilityFn = Callable[
    [str, Path, Path, bool, int],
    tuple[bool, bool, int, list[str], str | None],
]


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
                    path.name
                    for path in scans_root.iterdir()
                    if path.is_dir() and path.name.startswith("scene")
                ]
            )

    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    return scene_ids


def read_scannetpp_scene_ids(
    dataset_root: Path,
    scene_list_file: Path | None = None,
    split: str | None = "nvs_sem_val",
    max_scenes: int | None = None,
) -> list[str]:
    dataset_root = resolve_scannetpp_dataset_root(dataset_root=dataset_root)

    if scene_list_file is not None:
        scene_ids = [
            line.strip()
            for line in scene_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif split is not None:
        scene_ids = read_scannetpp_split_scene_ids(split=split, dataset_root=dataset_root)
    else:
        data_root = dataset_root / "data"
        if not data_root.exists():
            scene_ids = []
        else:
            scene_ids = sorted([path.name for path in data_root.iterdir() if path.is_dir()])

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
    if result.get("download_errors"):
        for idx, err in enumerate(result["download_errors"], 1):
            print(f"  download_error_{idx}: {err}")
    if result.get("summary_path"):
        print(f"summary_path={result['summary_path']}")
    if result.get("manifest_path"):
        print(f"manifest_path={result['manifest_path']}")


def _flatten_scene_quality(scene_summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    scene_metrics = scene_summary.get("scene_intrinsic_metrics", {}) or {}
    flat["avg_noise_fraction_seen"] = scene_metrics.get("avg_noise_fraction_seen")
    flat["avg_unseen_fraction"] = scene_metrics.get("avg_unseen_fraction")
    flat["avg_labeled_fraction_seen"] = scene_metrics.get("avg_labeled_fraction_seen")
    flat["total_clusters_across_granularities"] = scene_metrics.get(
        "total_clusters_across_granularities"
    )

    teacher_by_g = {
        f"g{teacher.get('granularity')}": teacher
        for teacher in scene_summary.get("teacher_outputs", []) or []
    }
    cluster_by_g = {
        f"g{cluster.get('granularity')}": cluster
        for cluster in scene_summary.get("cluster_outputs", []) or []
    }
    by_g = scene_metrics.get("by_granularity", {}) or {}
    granularities = sorted(set(teacher_by_g) | set(cluster_by_g) | set(by_g))

    total_teacher_masks = 0
    for granularity_key in granularities:
        teacher_total_masks = (teacher_by_g.get(granularity_key, {}) or {}).get("total_masks")
        if teacher_total_masks is not None:
            total_teacher_masks += int(teacher_total_masks)
        flat[f"teacher_total_masks_{granularity_key}"] = teacher_total_masks

        cluster_stats = (cluster_by_g.get(granularity_key, {}) or {}).get("stats", {}) or {}
        flat[f"clusters_{granularity_key}"] = cluster_stats.get("num_clusters")
        flat[f"used_frames_{granularity_key}"] = cluster_stats.get("used_frames")
        flat[f"num_2d_masks_total_{granularity_key}"] = cluster_stats.get("num_2d_masks_total")

        granularity_metrics = by_g.get(granularity_key, {}) or {}
        flat[f"noise_fraction_seen_{granularity_key}"] = granularity_metrics.get(
            "noise_fraction_seen"
        )
        flat[f"unseen_fraction_{granularity_key}"] = granularity_metrics.get(
            "unseen_points_fraction"
        )
        flat[f"labeled_fraction_seen_{granularity_key}"] = granularity_metrics.get(
            "labeled_points_fraction_seen"
        )

    flat["total_teacher_masks_across_granularities"] = total_teacher_masks
    return flat


def _safe_mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _emit_running_summary(
    reporter: Any | None,
    run_summary: dict[str, Any],
    evaluation_hooks: DatasetEvaluationHooks,
) -> None:
    if reporter is None or not hasattr(reporter, "log_event"):
        return

    completed = run_summary.get("scene_results", []) or []
    done_rows = [row for row in completed if row.get("status") in {"done", "skipped_done"}]

    payload = {
        "summary/running_done": run_summary.get("done", 0),
        "summary/running_failed": run_summary.get("failed", 0),
        "summary/running_skipped_done": run_summary.get("skipped_done", 0),
        "summary/running_avg_noise_fraction_seen": _safe_mean(
            [row.get("avg_noise_fraction_seen") for row in done_rows]
        ),
        "summary/running_avg_unseen_fraction": _safe_mean(
            [row.get("avg_unseen_fraction") for row in done_rows]
        ),
        "progress/heartbeat": time.time(),
    }
    payload.update(evaluation_hooks.running_summary_payload(done_rows))
    reporter.log_event(payload)


def _record_scene_result(
    run_summary: dict[str, Any],
    result: dict[str, Any],
    reporter: Any | None,
    evaluation_hooks: DatasetEvaluationHooks,
) -> None:
    status = result.get("status")
    scene_id = result.get("scene_id")

    if status in {"done", "skipped_done", "downloaded_only"}:
        run_summary["done"] += 1
    if status == "skipped_done":
        run_summary["skipped_done"] += 1
    if status == "failed":
        run_summary["failed"] += 1
        if scene_id is not None:
            run_summary["failed_scenes"].append(scene_id)

    run_summary["scene_results"].append(result)
    _print_scene_result(result)
    if reporter is not None:
        reporter.log_scene(result)
    _emit_running_summary(reporter, run_summary, evaluation_hooks)


def _scene_has_raw_sens(scene_dir: Path) -> bool:
    return (scene_dir / f"{scene_dir.name}.sens").exists()


def _ensure_scannet_scene_available(
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
            download_scannet_scene(scene_id=scene_id, scans_root=scans_root, skip_existing=True)
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


def _build_scannetpp_availability_checker(
    dataset_root: Path,
    require_annotations: bool,
) -> SceneAvailabilityFn:
    dataset_root = resolve_scannetpp_dataset_root(dataset_root=dataset_root)

    def _ensure_scannetpp_scene_available(
        scene_id: str,
        scene_dir: Path,
        scenes_root: Path,
        auto_download_missing: bool,
        max_download_retries: int,
    ) -> tuple[bool, bool, int, list[str], str | None]:
        del scenes_root

        if is_scannetpp_prepared(scene_dir) or scannetpp_has_raw_scene_assets(
            scene_dir,
            require_annotations=require_annotations,
        ):
            return True, False, 0, [], None

        if not auto_download_missing:
            reason = (
                "scene is missing the raw ScanNet++ assets required for this run and "
                "auto-download is disabled"
            )
            return False, False, 0, [], reason

        errors: list[str] = []
        for attempt in range(1, max_download_retries + 1):
            try:
                download_scannetpp_scene(
                    scene_id=scene_id,
                    dataset_root=dataset_root,
                    require_annotations=require_annotations,
                    skip_existing=True,
                )
                if is_scannetpp_prepared(scene_dir) or scannetpp_has_raw_scene_assets(
                    scene_dir,
                    require_annotations=require_annotations,
                ):
                    return True, True, attempt, errors, None
                errors.append(
                    f"attempt {attempt}: downloader returned but required ScanNet++ assets are still missing"
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

    return _ensure_scannetpp_scene_available


def run_streaming_dataset(
    dataset_name: str,
    scenes_root: Path,
    scene_ids: list[str],
    teacher: TeacherModel,
    granularities: list[float],
    adapter_factory: SceneAdapterFactory,
    ensure_scene_available: SceneAvailabilityFn,
    frame_skip: int = 10,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    run_oracle_eval: bool = True,
    export_training_pack: bool = True,
    overwrite_existing: bool = False,
    continue_on_error: bool = True,
    auto_download_missing: bool = True,
    cleanup_after_success: bool = True,
    download_only: bool = False,
    max_download_retries: int = 3,
    reporter: Any | None = None,
    run_summary_extra: dict[str, Any] | None = None,
    cleanup_raw_source_suffixes: tuple[str, ...] = (".sens", ".zip"),
) -> dict[str, Any]:
    scenes_root = Path(scenes_root)

    run_wall_start_perf = time.perf_counter()
    run_summary: dict[str, Any] = {
        "dataset": dataset_name,
        "scenes_root": str(scenes_root),
        "started_at": utcnow_iso(),
        "num_scenes_requested": len(scene_ids),
        "granularities": [float(g) for g in granularities],
        "frame_skip": int(frame_skip),
        "run_oracle_eval": bool(run_oracle_eval),
        "export_training_pack": bool(export_training_pack),
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
    if run_summary_extra:
        run_summary.update(run_summary_extra)

    total = len(scene_ids)
    for idx, scene_id in enumerate(scene_ids, start=1):
        _print_scene_header(idx, total, scene_id)

        started = time.perf_counter()
        scene_dir = scenes_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        adapter = adapter_factory(scene_dir)
        evaluation_hooks = adapter.get_evaluation_hooks()

        manifest = init_scene_manifest(
            scene_id=scene_id,
            scene_dir=scene_dir,
            dataset=dataset_name,
            granularities=granularities,
            frame_skip=frame_skip,
            run_oracle_eval=run_oracle_eval,
            export_training_pack=export_training_pack,
            overwrite_existing=overwrite_existing,
            auto_download_missing=auto_download_missing,
            cleanup_after_success=cleanup_after_success,
            download_only=download_only,
        )
        add_manifest_event(manifest, phase="scene", status="started", message="scene processing started")
        manifest_path = write_scene_manifest(scene_dir, manifest)

        if not overwrite_existing:
            is_complete, existing_summary, _ = verify_scene_completion_from_summary(
                scene_dir=scene_dir,
                granularities=granularities,
                require_oracle=run_oracle_eval,
                require_training_pack=export_training_pack,
                evaluation_hooks=evaluation_hooks,
            )
            if is_complete:
                add_manifest_event(
                    manifest,
                    phase="skip",
                    status="done",
                    message="summary + outputs already verified",
                )
                finalize_scene_manifest(
                    manifest,
                    status="skipped_done",
                    reason="summary + outputs already verified",
                    summary_path=str(scene_dir / "scene_pipeline_summary.json"),
                    verification={"ok": True, "missing_outputs": []},
                )
                write_scene_manifest(scene_dir, manifest)

                existing_quality_summary = {}
                if existing_summary is not None:
                    existing_quality_summary = {
                        **_flatten_scene_quality(existing_summary),
                        **evaluation_hooks.flatten_scene_summary(existing_summary),
                    }

                _record_scene_result(
                    run_summary,
                    {
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
                        **existing_quality_summary,
                    },
                    reporter,
                    evaluation_hooks,
                )
                continue

        add_manifest_event(manifest, phase="download", status="running", message="checking scene availability")
        available, downloaded, download_attempts, download_errors, availability_reason = ensure_scene_available(
            scene_id,
            scene_dir,
            scenes_root,
            auto_download_missing,
            max_download_retries,
        )
        manifest["download"]["attempts"] = int(download_attempts)
        manifest["download"]["downloaded"] = bool(downloaded)
        manifest["download"]["errors"] = download_errors

        if not available:
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

            _record_scene_result(
                run_summary,
                {
                    "scene_id": scene_id,
                    "scene_dir": str(scene_dir),
                    "status": "failed",
                    "duration_seconds": time.perf_counter() - started,
                    "reason": availability_reason,
                    "error": availability_reason,
                    "download_errors": download_errors,
                    "downloaded": downloaded,
                    "download_attempts": download_attempts,
                    "manifest_path": str(manifest_path),
                },
                reporter,
                evaluation_hooks,
            )
            if not continue_on_error:
                break
            continue

        manifest["download"]["status"] = "available"
        add_manifest_event(
            manifest,
            phase="download",
            status="done",
            message="scene is available",
            extra={"downloaded": downloaded, "attempts": download_attempts},
        )
        write_scene_manifest(scene_dir, manifest)

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

            _record_scene_result(
                run_summary,
                {
                    "scene_id": scene_id,
                    "scene_dir": str(scene_dir),
                    "status": "downloaded_only",
                    "duration_seconds": time.perf_counter() - started,
                    "reason": "download_only mode requested",
                    "downloaded": downloaded,
                    "download_attempts": download_attempts,
                    "manifest_path": str(manifest_path),
                },
                reporter,
                evaluation_hooks,
            )
            continue

        quality_summary: dict[str, Any] = {}
        try:
            add_manifest_event(
                manifest,
                phase="pipeline",
                status="running",
                message="running CHORUS scene pipeline",
            )
            write_scene_manifest(scene_dir, manifest)

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
                export_training_pack=export_training_pack,
            )
            quality_summary = {
                **_flatten_scene_quality(scene_summary),
                **evaluation_hooks.flatten_scene_summary(scene_summary),
            }

            verified_ok, _, missing = verify_scene_completion_from_summary(
                scene_dir=scene_dir,
                granularities=granularities,
                require_oracle=run_oracle_eval,
                require_training_pack=export_training_pack,
                evaluation_hooks=evaluation_hooks,
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

                _record_scene_result(
                    run_summary,
                    {
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
                    },
                    reporter,
                    evaluation_hooks,
                )
                if not continue_on_error:
                    break
                continue

            cleanup_info = None
            if cleanup_after_success:
                add_manifest_event(
                    manifest,
                    phase="cleanup",
                    status="running",
                    message="cleaning intermediate files",
                )
                write_scene_manifest(scene_dir, manifest)

                cleanup_info = cleanup_scene_intermediates(
                    scene_dir=scene_dir,
                    granularities=granularities,
                    delete_rgbd=True,
                    delete_teacher_masks=True,
                    delete_svd_features=True,
                    delete_raw_source_files=bool(cleanup_raw_source_suffixes),
                    raw_source_suffixes=cleanup_raw_source_suffixes,
                )
                verified_ok_after_cleanup, _, missing_after_cleanup = verify_scene_completion_from_summary(
                    scene_dir=scene_dir,
                    granularities=granularities,
                    require_oracle=run_oracle_eval,
                    require_training_pack=export_training_pack,
                    evaluation_hooks=evaluation_hooks,
                )
                if not verified_ok_after_cleanup:
                    add_manifest_event(
                        manifest,
                        phase="cleanup",
                        status="failed",
                        message="cleanup removed files required by final verification",
                        extra={
                            "missing_outputs": missing_after_cleanup,
                            "cleanup": cleanup_info,
                        },
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

                    _record_scene_result(
                        run_summary,
                        {
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
                        },
                        reporter,
                        evaluation_hooks,
                    )
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

            _record_scene_result(
                run_summary,
                {
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
                },
                reporter,
                evaluation_hooks,
            )
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

            _record_scene_result(
                run_summary,
                {
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
                },
                reporter,
                evaluation_hooks,
            )
            if not continue_on_error:
                break

    wall_seconds = time.perf_counter() - run_wall_start_perf
    scene_results = run_summary.get("scene_results") or []
    full_done = [row for row in scene_results if row.get("status") == "done"]
    num_full = len(full_done)
    sum_pipeline_seconds = sum(float(row.get("duration_seconds") or 0.0) for row in full_done)
    wall_hours = wall_seconds / 3600.0 if wall_seconds > 0 else 0.0
    pipeline_hours = sum_pipeline_seconds / 3600.0 if sum_pipeline_seconds > 0 else 0.0

    run_summary["finished_at"] = utcnow_iso()
    run_summary["run_timing"] = {
        "wall_clock_seconds": round(wall_seconds, 3),
        "wall_clock_hours": round(wall_hours, 4),
        "num_scenes_fully_processed": num_full,
        "num_scenes_skipped_already_done": int(run_summary.get("skipped_done") or 0),
        "num_scenes_failed": int(run_summary.get("failed") or 0),
        "sum_pipeline_seconds_full_scenes": round(sum_pipeline_seconds, 3),
        "sum_pipeline_hours_full_scenes": round(pipeline_hours, 4),
        "pace_full_pipeline_scenes_per_hour": (
            round(num_full / pipeline_hours, 4) if pipeline_hours > 1e-9 else None
        ),
        "pace_requested_scenes_per_hour_wall": (
            round(len(scene_ids) / wall_hours, 4) if wall_hours > 1e-9 else None
        ),
        "pace_full_scenes_per_hour_wall": (
            round(num_full / wall_hours, 4) if wall_hours > 1e-9 and num_full > 0 else None
        ),
    }

    if reporter is not None:
        reporter.log_summary(run_summary)
        if hasattr(reporter, "log_event"):
            reporter.log_event({"summary": run_summary})
    return run_summary


def run_streaming_scannet(
    scans_root: Path,
    scene_ids: list[str],
    teacher: TeacherModel,
    granularities: list[float],
    scannet_eval_benchmarks: list[str] | tuple[str, ...] | str | None = None,
    frame_skip: int = 10,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    run_oracle_eval: bool = True,
    export_training_pack: bool = True,
    overwrite_existing: bool = False,
    continue_on_error: bool = True,
    auto_download_missing: bool = True,
    cleanup_after_success: bool = True,
    download_only: bool = False,
    max_download_retries: int = 3,
    reporter: Any | None = None,
) -> dict[str, Any]:
    parsed_benchmarks = parse_scannet_eval_benchmarks(scannet_eval_benchmarks)
    scans_root = Path(scans_root)
    return run_streaming_dataset(
        dataset_name="scannet",
        scenes_root=scans_root,
        scene_ids=scene_ids,
        teacher=teacher,
        granularities=granularities,
        adapter_factory=lambda scene_dir: ScanNetSceneAdapter(
            scene_root=scene_dir,
            eval_benchmarks=parsed_benchmarks,
        ),
        ensure_scene_available=_ensure_scannet_scene_available,
        frame_skip=frame_skip,
        svd_components=svd_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        run_oracle_eval=run_oracle_eval,
        export_training_pack=export_training_pack,
        overwrite_existing=overwrite_existing,
        continue_on_error=continue_on_error,
        auto_download_missing=auto_download_missing,
        cleanup_after_success=cleanup_after_success,
        download_only=download_only,
        max_download_retries=max_download_retries,
        reporter=reporter,
        run_summary_extra={
            "scans_root": str(scans_root),
            "scannet_eval_benchmarks": list(parsed_benchmarks),
        },
        cleanup_raw_source_suffixes=(".sens", ".zip"),
    )


def run_streaming_scannetpp(
    dataset_root: Path,
    scene_ids: list[str],
    teacher: TeacherModel,
    granularities: list[float],
    eval_benchmark: str | None = None,
    frame_skip: int = 10,
    svd_components: int = 32,
    min_cluster_size: int = 100,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    run_oracle_eval: bool = True,
    export_training_pack: bool = True,
    overwrite_existing: bool = False,
    continue_on_error: bool = True,
    auto_download_missing: bool = True,
    cleanup_after_success: bool = True,
    download_only: bool = False,
    max_download_retries: int = 3,
    reporter: Any | None = None,
) -> dict[str, Any]:
    dataset_root = resolve_scannetpp_dataset_root(dataset_root=dataset_root)
    scenes_root = dataset_root / "data"
    normalized_benchmark = normalize_scannetpp_eval_benchmark(eval_benchmark)

    return run_streaming_dataset(
        dataset_name="scannetpp",
        scenes_root=scenes_root,
        scene_ids=scene_ids,
        teacher=teacher,
        granularities=granularities,
        adapter_factory=lambda scene_dir: ScanNetPPSceneAdapter(
            scene_root=scene_dir,
            eval_benchmark=normalized_benchmark,
        ),
        ensure_scene_available=_build_scannetpp_availability_checker(
            dataset_root=dataset_root,
            require_annotations=run_oracle_eval,
        ),
        frame_skip=frame_skip,
        svd_components=svd_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        run_oracle_eval=run_oracle_eval,
        export_training_pack=export_training_pack,
        overwrite_existing=overwrite_existing,
        continue_on_error=continue_on_error,
        auto_download_missing=auto_download_missing,
        cleanup_after_success=cleanup_after_success,
        download_only=download_only,
        max_download_retries=max_download_retries,
        reporter=reporter,
        run_summary_extra={
            "dataset_root": str(dataset_root),
            "eval_benchmark": normalized_benchmark,
        },
        cleanup_raw_source_suffixes=(),
    )
