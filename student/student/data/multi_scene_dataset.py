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

from student.data.training_pack import (
    MultiGranTrainingPackScene,
    _resolve_pack_dir,
    load_training_pack_scene_multi,
)
from student.data.single_scene_dataset import build_input_features

log = logging.getLogger(__name__)


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
    preload:
        Load all scenes into memory at init time.  Recommended for
        small-scale experiments (10–20 scenes).
    max_points:
        If set, each ``__getitem__`` randomly subsamples at most this many
        points (same indices for points, features, labels, masks).  Use when
        full scenes exceed GPU memory — decoder cross-attention cost grows with
        scene size.
    """

    def __init__(
        self,
        scene_dirs: list[Path],
        granularities: tuple[str, ...],
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
        preload: bool = True,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        self._scene_dirs = list(scene_dirs)
        self._granularities = granularities
        self._use_colors = use_colors
        self._append_xyz = append_xyz
        self._max_points = max_points
        self._scenes: list[MultiGranTrainingPackScene] = []
        self._features_cache: list[np.ndarray] = []

        if preload:
            for sd in self._scene_dirs:
                scene = load_training_pack_scene_multi(sd, granularities)
                feats = build_input_features(
                    scene.points, scene.colors,
                    use_colors=use_colors, append_xyz=append_xyz,
                )
                self._scenes.append(scene)
                self._features_cache.append(feats)
                log.info(
                    "[%s] Preloaded: %d points, %d features",
                    scene.scene_id, scene.num_points, feats.shape[1],
                )

    def __len__(self) -> int:
        return len(self._scene_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self._scenes:
            scene = self._scenes[idx]
            features = self._features_cache[idx]
        else:
            scene = load_training_pack_scene_multi(
                self._scene_dirs[idx], self._granularities,
            )
            features = build_input_features(
                scene.points, scene.colors,
                use_colors=self._use_colors, append_xyz=self._append_xyz,
            )

        labels_by_gran = {
            g: torch.from_numpy(scene.labels_by_granularity[g]).long()
            for g in self._granularities
        }

        points_t = torch.from_numpy(scene.points).float()
        features_t = torch.from_numpy(features).float()
        valid_t = torch.from_numpy(scene.valid_points).bool()
        seen_t = torch.from_numpy(scene.seen_points).bool()
        sup_t = torch.from_numpy(scene.supervision_mask).bool()

        n = points_t.shape[0]
        if self._max_points is not None and n > self._max_points:
            idx = torch.randperm(n)[: self._max_points]
            points_t = points_t[idx]
            features_t = features_t[idx]
            valid_t = valid_t[idx]
            seen_t = seen_t[idx]
            sup_t = sup_t[idx]
            labels_by_gran = {g: labels_by_gran[g][idx] for g in self._granularities}

        return {
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

    @property
    def scene_ids(self) -> list[str]:
        """List of all scene IDs in dataset order."""
        if self._scenes:
            return [s.scene_id for s in self._scenes]
        return [d.name for d in self._scene_dirs]


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
