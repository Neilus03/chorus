"""Multi-scene dataset for epoch-based training.

Wraps multiple CHORUS training-pack scenes into a single
:class:`torch.utils.data.Dataset` where ``__len__`` is the
number of scenes and ``__getitem__(i)`` yields the same dict
structure as ``MultiGranSceneDataset[0]``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from student.data.region_sampling import sphere_crop_indices
from student.data.training_pack import (
    MultiGranTrainingPackScene,
    _load_scene_meta,
    _resolve_pack_dir,
    load_training_pack_scene_multi,
)
from student.data.point_augmentations import augment_points_litept_scannet
from student.data.single_scene_dataset import build_input_features

log = logging.getLogger(__name__)


def _normals_array_for_scene(
    scene: MultiGranTrainingPackScene,
    *,
    use_normals: bool,
    warned_ids: set[str],
) -> np.ndarray | None:
    """Return per-point normals for augmentation/features, or None if disabled."""
    if not use_normals:
        return None
    if scene.normals is not None:
        return np.asarray(scene.normals, dtype=np.float32)
    if scene.scene_id not in warned_ids:
        warned_ids.add(scene.scene_id)
        log.warning(
            "[%s] use_normals=True but normals.npy missing; using zeros",
            scene.scene_id,
        )
    return np.zeros((scene.num_points, 3), dtype=np.float32)


def build_scene_list(
    scene_list_file: Path,
    scans_root: Path,
) -> list[Path]:
    """Read a text file of scene IDs and resolve to full paths.

    Parameters
    ----------
    scene_list_file:
        Text file with one scene ID per line.  Blank lines and lines
        starting with ``#`` are skipped.
    scans_root:
        Root directory containing scene subdirectories.

    Returns
    -------
    List of resolved, validated scene directory paths.

    Raises
    ------
    FileNotFoundError
        If the list file, a scene directory, or its training pack is missing.
    ValueError
        If the file contains no scenes.
    """
    scene_list_file = Path(scene_list_file)
    scans_root = Path(scans_root)

    if not scene_list_file.is_file():
        raise FileNotFoundError(f"Scene list file not found: {scene_list_file}")

    scene_dirs: list[Path] = []
    for line in scene_list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scene_dir = scans_root / line
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        _resolve_pack_dir(scene_dir)
        scene_dirs.append(scene_dir)

    if not scene_dirs:
        raise ValueError(f"No scenes found in {scene_list_file}")

    log.info("Loaded %d scene paths from %s", len(scene_dirs), scene_list_file)
    return scene_dirs


class MultiSceneDataset(Dataset):
    """Dataset that iterates over multiple scenes.

    Each item is a complete sample dict with the same keys and types
    as ``MultiGranSceneDataset[0]``.

    Parameters
    ----------
    scene_dirs:
        Resolved scene directory paths.
    granularities:
        Dot-free granularity keys, e.g. ``("g02", "g05", "g08")``.
    use_colors:
        Use RGB from ``colors.npy`` as input features.
    append_xyz:
        Append XYZ coordinates to the feature vector.
    use_normals:
        If True, concatenate mesh normals (from ``normals.npy`` or zeros if missing).
    preload:
        Load all scenes into memory at init time.  Recommended for
        small-scale experiments (10–20 scenes).
    max_points:
        If set, caps the number of points per ``__getitem__`` (same indices for
        points, features, labels, masks).  Ignored when *subsampling_mode* is
        ``"none"`` and neither *max_points* nor *sphere_point_max* applies.
    subsampling_mode:
        ``"sphere_crop"`` (default): Point-style sphere around a random point,
        keep nearest *cap* points.  ``"randperm"``: legacy global random subset.
        ``"none"``: never subsample (full scene each step; needs enough GPU RAM).
    sphere_point_max:
        Optional override for the sphere crop size; defaults to *max_points*
        when unset.
    train_augmentations:
        If True, apply LitePT ScanNet-style geometric and (when colors are used)
        chromatic jitter on **training** samples only.  Ignores precomputed
        feature cache so augmentations vary each epoch.
    """

    def __init__(
        self,
        scene_dirs: list[Path],
        granularities: tuple[str, ...],
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
        use_normals: bool = False,
        preload: bool = True,
        max_points: int | None = None,
        subsampling_mode: str = "sphere_crop",
        sphere_point_max: int | None = None,
        train_augmentations: bool = False,
    ) -> None:
        super().__init__()
        self._scene_dirs = list(scene_dirs)
        self._granularities = granularities
        self._use_colors = use_colors
        self._append_xyz = append_xyz
        self._use_normals = bool(use_normals)
        self._missing_normals_warned: set[str] = set()
        self._max_points = max_points
        self._subsampling_mode = subsampling_mode
        self._sphere_point_max = sphere_point_max
        self._train_augmentations = bool(train_augmentations)
        self._scenes: list[MultiGranTrainingPackScene] = []
        self._scene_point_counts: list[int] = []
        self._features_cache: list[np.ndarray] | None = [] if not train_augmentations else None

        if self._subsampling_mode not in ("sphere_crop", "randperm", "none"):
            raise ValueError(
                f"subsampling_mode must be 'sphere_crop', 'randperm', or 'none', "
                f"got {self._subsampling_mode!r}",
            )
        if self._subsampling_mode == "none" and (
            self._max_points is not None or self._sphere_point_max is not None
        ):
            log.warning(
                "subsampling_mode=none: ignoring max_points / sphere_point_max (full scenes)",
            )

        if self._train_augmentations:
            try:
                import scipy.ndimage  # noqa: F401
            except ImportError:
                log.warning(
                    "train_augmentations=True but scipy is not installed; "
                    "elastic distortion will be skipped (install scipy for full LitePT-style aug).",
                )
            else:
                log.info(
                    "train_augmentations=True (LitePT ScanNet-style: rotate, scale, "
                    "flip, jitter, elastic, chromatic)",
                )

        self._output_cap: int | None
        if self._sphere_point_max is not None:
            self._output_cap = int(self._sphere_point_max)
        elif self._max_points is not None:
            self._output_cap = int(self._max_points)
        else:
            self._output_cap = None

        if preload:
            for sd in self._scene_dirs:
                scene = load_training_pack_scene_multi(sd, granularities)
                n_arr = _normals_array_for_scene(
                    scene,
                    use_normals=self._use_normals,
                    warned_ids=self._missing_normals_warned,
                )
                feats = build_input_features(
                    scene.points, scene.colors,
                    use_colors=use_colors,
                    append_xyz=append_xyz,
                    use_normals=self._use_normals,
                    normals=n_arr if self._use_normals else None,
                )
                self._scenes.append(scene)
                effective_points = scene.num_points
                if self._output_cap is not None:
                    effective_points = min(effective_points, self._output_cap)
                self._scene_point_counts.append(int(effective_points))
                if self._features_cache is not None:
                    self._features_cache.append(feats)
                log.info(
                    "[%s] Preloaded: %d points, %d features%s",
                    scene.scene_id,
                    scene.num_points,
                    feats.shape[1],
                    " (features rebuilt each step — augmentations on)" if self._train_augmentations else "",
                )
        else:
            for sd in self._scene_dirs:
                meta = _load_scene_meta(_resolve_pack_dir(sd))
                effective_points = int(meta["num_points"])
                if self._output_cap is not None:
                    effective_points = min(effective_points, self._output_cap)
                self._scene_point_counts.append(effective_points)

    def __len__(self) -> int:
        return len(self._scene_dirs)

    def _should_subsample(self) -> bool:
        if self._subsampling_mode == "none":
            return False
        return self._output_cap is not None

    def _subsample_indices(self, n: int, points_np: np.ndarray) -> torch.Tensor:
        cap = self._output_cap
        assert cap is not None
        if n <= cap:
            return torch.arange(n, dtype=torch.long)

        if self._subsampling_mode == "randperm":
            return torch.randperm(n)[:cap]

        if self._subsampling_mode == "sphere_crop":
            assert self._output_cap is not None
            rng = np.random.default_rng()
            idx_np = sphere_crop_indices(
                np.asarray(points_np, dtype=np.float32),
                rng=rng,
                point_max=int(self._output_cap),
            )
            return torch.from_numpy(idx_np).long()

        raise RuntimeError(f"Unhandled subsampling_mode={self._subsampling_mode!r}")

    def _compose_item(self, idx: int, *, subsample: bool) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self._scenes:
            scene = self._scenes[idx]
            base_points = scene.points
            base_colors = scene.colors
        else:
            scene = load_training_pack_scene_multi(
                self._scene_dirs[idx], self._granularities,
            )
            base_points = scene.points
            base_colors = scene.colors

        n_arr = _normals_array_for_scene(
            scene,
            use_normals=self._use_normals,
            warned_ids=self._missing_normals_warned,
        )

        if self._train_augmentations:
            aug_pts, aug_cols, aug_nrm = augment_points_litept_scannet(
                base_points,
                base_colors,
                use_colors=self._use_colors,
                normals=n_arr,
            )
            features = build_input_features(
                aug_pts,
                aug_cols,
                use_colors=self._use_colors,
                append_xyz=self._append_xyz,
                use_normals=self._use_normals,
                normals=aug_nrm if self._use_normals else None,
            )
            points_np = aug_pts
        elif self._scenes and self._features_cache is not None:
            features = self._features_cache[idx]
            points_np = base_points
        else:
            features = build_input_features(
                base_points,
                base_colors,
                use_colors=self._use_colors,
                append_xyz=self._append_xyz,
                use_normals=self._use_normals,
                normals=n_arr if self._use_normals else None,
            )
            points_np = base_points

        labels_by_gran = {
            g: torch.from_numpy(scene.labels_by_granularity[g]).long()
            for g in self._granularities
        }

        points_t = torch.from_numpy(np.asarray(points_np)).float()
        features_t = torch.from_numpy(features).float()
        valid_t = torch.from_numpy(scene.valid_points).bool()
        seen_t = torch.from_numpy(scene.seen_points).bool()
        sup_t = torch.from_numpy(scene.supervision_mask).bool()

        n = points_t.shape[0]
        vertex_indices: torch.Tensor | None = None
        if subsample and self._should_subsample():
            idx_t = self._subsample_indices(n, np.asarray(points_np, dtype=np.float32))
            vertex_indices = idx_t.clone()
            points_t = points_t[idx_t]
            features_t = features_t[idx_t]
            valid_t = valid_t[idx_t]
            seen_t = seen_t[idx_t]
            sup_t = sup_t[idx_t]
            labels_by_gran = {g: labels_by_gran[g][idx_t] for g in self._granularities}

        out: dict[str, Any] = {
            "scene_id": scene.scene_id,
            "scene_dir": str(scene.scene_dir),
            "points": points_t,
            "features": features_t,
            "labels_by_granularity": labels_by_gran,
            "valid_points": valid_t,
            "seen_points": seen_t,
            "supervision_mask": sup_t,
            "scene_meta": scene.scene_meta,
            "granularities": self._granularities,
        }
        if vertex_indices is not None:
            out["vertex_indices"] = vertex_indices
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._compose_item(idx, subsample=True)

    def get_full_item(self, idx: int) -> dict[str, Any]:
        """Same as ``__getitem__`` but without point subsampling (full scene).

        For optional fragment-based evaluation that needs the full vertex list.
        """
        return self._compose_item(idx, subsample=False)

    @property
    def scene_ids(self) -> list[str]:
        """List of all scene IDs in dataset order."""
        if self._scenes:
            return [s.scene_id for s in self._scenes]
        return [d.name for d in self._scene_dirs]

    @property
    def scene_point_counts(self) -> list[int]:
        """Effective point counts after optional max-point clipping."""
        return list(self._scene_point_counts)


# ── smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Smoke test for MultiSceneDataset",
    )
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--scans-root", type=str, required=True)
    args = parser.parse_args()

    dirs = build_scene_list(Path(args.scene_list), Path(args.scans_root))
    ds = MultiSceneDataset(dirs, granularities=("g02", "g05", "g08"))

    print(f"\nlen(dataset): {len(ds)}")
    for i in range(len(ds)):
        s = ds[i]
        print(
            f"  [{i:2d}] {s['scene_id']:16s}  "
            f"points={tuple(s['points'].shape)}  "
            f"features={tuple(s['features'].shape)}  "
            f"grans={list(s['labels_by_granularity'].keys())}  "
            f"supervised={s['supervision_mask'].sum().item()}"
        )
