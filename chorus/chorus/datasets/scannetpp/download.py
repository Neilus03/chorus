from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from chorus.datasets.scannetpp.data_download.scene_release import ScannetppScene_Release

ROOT_URL_TEMPLATE = (
    "https://scannetpp.mlsg.cit.tum.de/scannetpp/download/v2"
    "?version=v1&token=TOKEN&file=FILEPATH"
)

SCANNETPP_SPLITS = (
    "nvs_sem_train",
    "nvs_sem_val",
    "nvs_test",
    "sem_test",
    "nvs_test_iphone",
)

SCANNETPP_META_FILES = (
    "splits/nvs_sem_train.txt",
    "splits/nvs_sem_val.txt",
    "splits/nvs_test.txt",
    "splits/nvs_test_small.txt",
    "splits/sem_test.txt",
    "splits/nvs_test_iphone.txt",
    "metadata/scene_types.json",
    "metadata/semantic_classes.txt",
    "metadata/instance_classes.txt",
    "metadata/semantic_benchmark/top100.txt",
    "metadata/semantic_benchmark/top100_instance.txt",
    "metadata/semantic_benchmark/map_benchmark.csv",
)

SCANNETPP_STREAMING_ASSETS = (
    "scan_mesh_path",
    "iphone_video_path",
    "iphone_depth_path",
    "iphone_pose_intrinsic_imu_path",
)

SCANNETPP_ANNOTATION_ASSETS = (
    "scan_mesh_segs_path",
    "scan_anno_json_path",
)

SCANNETPP_OPTIONAL_ASSETS = (
    "scan_sem_mesh_path",
)

SCANNETPP_ZIPPED_ASSETS = frozenset(
    {
        "scan_mesh_path",
        "scan_sem_mesh_path",
        "scan_mesh_segs_path",
        "scan_anno_json_path",
        "iphone_depth_path",
        "iphone_pose_intrinsic_imu_path",
    }
)

SCANNETPP_EXCLUDED_ASSETS_BY_SPLIT = {
    "nvs_test": frozenset(
        {
            "iphone_depth_path",
            "scan_mesh_path",
            "scan_mesh_segs_path",
            "scan_anno_json_path",
            "scan_sem_mesh_path",
        }
    ),
    "sem_test": frozenset(
        {
            "scan_mesh_segs_path",
            "scan_anno_json_path",
            "scan_sem_mesh_path",
        }
    ),
    "nvs_test_iphone": frozenset(
        {
            "scan_mesh_path",
            "scan_mesh_segs_path",
            "scan_anno_json_path",
            "scan_sem_mesh_path",
        }
    ),
}


def _expanded_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def _normalize_dataset_root(path: str | Path) -> Path:
    candidate = _expanded_path(path)
    if candidate.name == "data":
        return candidate.parent
    if candidate.parent.name == "data":
        return candidate.parent.parent
    return candidate


def _candidate_dataset_roots(
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> list[Path]:
    candidates: list[Path] = []

    if dataset_root is not None:
        candidates.append(_normalize_dataset_root(dataset_root))

    for env_key in ("CHORUS_SCANNETPP_DATA_ROOT", "SCANNETPP_DATA_ROOT"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(_normalize_dataset_root(env_value))

    if scene_root is not None:
        scene_path = _expanded_path(scene_root)
        if scene_path.parent.name == "data":
            candidates.append(scene_path.parent.parent)
        elif scene_path.name == "data":
            candidates.append(scene_path.parent)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    return unique_candidates


def resolve_scannetpp_dataset_root(
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> Path:
    candidates = _candidate_dataset_roots(dataset_root=dataset_root, scene_root=scene_root)
    if not candidates:
        raise FileNotFoundError(
            "Could not infer the ScanNet++ dataset root. "
            "Set CHORUS_SCANNETPP_DATA_ROOT or pass dataset_root explicitly."
        )

    for candidate in candidates:
        if (
            (candidate / "data").exists()
            or (candidate / "metadata").exists()
            or (candidate / "splits").exists()
        ):
            return candidate

    return candidates[0]


def resolve_scannetpp_data_root(
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> Path:
    root = resolve_scannetpp_dataset_root(dataset_root=dataset_root, scene_root=scene_root)
    return root / "data"


def resolve_scannetpp_token(token: str | None = None) -> str:
    value = (
        token
        or os.environ.get("CHORUS_SCANNETPP_TOKEN")
        or os.environ.get("SCANNETPP_TOKEN")
    )
    if value:
        return value.strip()

    raise RuntimeError(
        "Missing ScanNet++ download token. "
        "Set CHORUS_SCANNETPP_TOKEN (or SCANNETPP_TOKEN)."
    )


def get_scene_release(scene_root: str | Path) -> ScannetppScene_Release:
    scene_root = _expanded_path(scene_root)
    return ScannetppScene_Release(scene_root.name, data_root=scene_root.parent)


def build_streaming_asset_list(
    require_annotations: bool = True,
    include_semantic_mesh: bool = False,
) -> tuple[str, ...]:
    assets = list(SCANNETPP_STREAMING_ASSETS)
    if require_annotations:
        assets.extend(SCANNETPP_ANNOTATION_ASSETS)
    if include_semantic_mesh:
        assets.extend(SCANNETPP_OPTIONAL_ASSETS)

    ordered_assets: list[str] = []
    seen: set[str] = set()
    for asset in assets:
        if asset in seen:
            continue
        seen.add(asset)
        ordered_assets.append(asset)
    return tuple(ordered_assets)


def expected_scene_asset_paths(
    scene_root: str | Path,
    require_annotations: bool = True,
    include_semantic_mesh: bool = False,
) -> dict[str, Path]:
    scene = get_scene_release(scene_root)
    asset_paths = {
        asset: Path(getattr(scene, asset))
        for asset in build_streaming_asset_list(
            require_annotations=require_annotations,
            include_semantic_mesh=include_semantic_mesh,
        )
    }
    return asset_paths


def _install_proxy_handler() -> None:
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")

    if not http_proxy and not https_proxy:
        return

    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    proxy_handler = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(proxy_handler)
    opener.addheaders = [("User-Agent", "Wget/1.21")]
    urllib.request.install_opener(opener)


def _download_url(remote_path: str | Path, token: str) -> str:
    remote_path = str(remote_path).replace("\\", "/")
    return ROOT_URL_TEMPLATE.replace("TOKEN", token).replace("FILEPATH", remote_path)


def _urlretrieve_multi_trials(url: str, filename: Path, max_trials: int = 5) -> None:
    for attempt in range(1, max_trials + 1):
        try:
            urllib.request.urlretrieve(url, filename)
            time.sleep(0.2)
            return
        except urllib.error.ContentTooShortError:
            if filename.exists():
                filename.unlink()
            if attempt == max_trials:
                raise
            time.sleep(0.5)
        except urllib.error.HTTPError as exc:
            if filename.exists():
                filename.unlink()
            raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc


def _download_release_file(
    remote_path: str | Path,
    local_path: Path,
    token: str,
    skip_existing: bool = True,
) -> Path:
    local_path = Path(local_path)
    if local_path.exists() and skip_existing:
        return local_path

    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    _install_proxy_handler()
    _urlretrieve_multi_trials(_download_url(remote_path, token), local_path)
    return local_path


def ensure_metadata(
    dataset_root: str | Path | None = None,
    token: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = resolve_scannetpp_dataset_root(dataset_root=dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    required_paths = [root / rel_path for rel_path in SCANNETPP_META_FILES]
    need_download = overwrite or any(not path.exists() for path in required_paths)
    if need_download:
        resolved_token = resolve_scannetpp_token(token)
        for rel_path in SCANNETPP_META_FILES:
            _download_release_file(
                remote_path=rel_path,
                local_path=root / rel_path,
                token=resolved_token,
                skip_existing=not overwrite,
            )

    return root


def read_split_scene_ids(
    split: str,
    dataset_root: str | Path | None = None,
) -> list[str]:
    normalized_split = str(split).strip()
    if normalized_split not in SCANNETPP_SPLITS:
        supported = ", ".join(SCANNETPP_SPLITS)
        raise ValueError(
            f"Unsupported ScanNet++ split '{split}'. Expected one of: {supported}."
        )

    root = resolve_scannetpp_dataset_root(dataset_root=dataset_root)
    split_path = root / "splits" / f"{normalized_split}.txt"
    if not split_path.exists():
        root = ensure_metadata(dataset_root=dataset_root)
        split_path = root / "splits" / f"{normalized_split}.txt"
    return [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def infer_scene_split(
    scene_id: str,
    dataset_root: str | Path | None = None,
) -> str:
    root = ensure_metadata(dataset_root=dataset_root)
    for split in SCANNETPP_SPLITS:
        if scene_id in read_split_scene_ids(split=split, dataset_root=root):
            return split
    raise ValueError(f"Scene {scene_id} does not appear in any downloaded ScanNet++ split.")


def download_scene(
    scene_id: str,
    dataset_root: str | Path,
    require_annotations: bool = True,
    skip_existing: bool = True,
    include_semantic_mesh: bool = False,
    token: str | None = None,
) -> Path:
    root = ensure_metadata(dataset_root=dataset_root, token=token)
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    resolved_token = resolve_scannetpp_token(token)
    split = infer_scene_split(scene_id=scene_id, dataset_root=root)
    excluded_assets = SCANNETPP_EXCLUDED_ASSETS_BY_SPLIT.get(split, frozenset())

    asset_names = build_streaming_asset_list(
        require_annotations=require_annotations,
        include_semantic_mesh=include_semantic_mesh,
    )
    missing_required_assets = [asset for asset in asset_names if asset in excluded_assets]
    if missing_required_assets:
        raise ValueError(
            f"Scene {scene_id} belongs to split '{split}', which does not provide "
            f"the assets required for this CHORUS run: {missing_required_assets}"
        )

    source_scene = ScannetppScene_Release(scene_id, data_root="data")
    target_scene = ScannetppScene_Release(scene_id, data_root=data_root)

    for asset in asset_names:
        if asset in excluded_assets:
            continue

        target_path = Path(getattr(target_scene, asset))
        remote_path = Path(getattr(source_scene, asset))

        if asset in SCANNETPP_ZIPPED_ASSETS:
            if (target_path.is_file() or target_path.is_dir()) and skip_existing:
                continue

            zipped_remote_path = remote_path.with_suffix(".zip")
            zipped_local_path = target_path.with_suffix(".zip")
            _download_release_file(
                remote_path=zipped_remote_path,
                local_path=zipped_local_path,
                token=resolved_token,
                skip_existing=skip_existing,
            )
            with zipfile.ZipFile(zipped_local_path, "r") as zip_ref:
                zip_ref.extractall(zipped_local_path.parent)
            zipped_local_path.unlink(missing_ok=True)
        else:
            _download_release_file(
                remote_path=remote_path,
                local_path=target_path,
                token=resolved_token,
                skip_existing=skip_existing,
            )

    return target_scene.scene_root_dir
