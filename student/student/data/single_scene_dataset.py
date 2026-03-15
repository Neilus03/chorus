"""Wrap one loaded CHORUS training-pack scene into a torch Dataset.

Length is 1 — designed for single-scene overfitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from student.data.training_pack import (
    TrainingPackScene,
    MultiGranTrainingPackScene,
    load_training_pack_scene,
    load_training_pack_scene_multi,
    print_training_pack_summary,
)

log = logging.getLogger(__name__)


# ── feature construction ─────────────────────────────────────────────────


def build_input_features(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
    append_xyz: bool = False,
) -> np.ndarray:
    """Build the (N, C) input feature matrix for LitePT.

    Rules (first version — no learned transforms):
        * ``use_colors=True`` and colors exist → RGB  (C=3)
        * ``use_colors=True`` but no colors    → zeros (C=3)
        * ``use_colors=False``                 → zeros (C=3), unless append_xyz
        * ``append_xyz=True``                  → append XYZ coords

    Returns float32 array of shape (N, C) where C ∈ {3, 6}.
    """
    N = points.shape[0]
    parts: list[np.ndarray] = []

    if use_colors and colors is not None:
        c = colors.astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        parts.append(c)
    elif not append_xyz or use_colors:
        parts.append(np.zeros((N, 3), dtype=np.float32))

    if append_xyz:
        parts.append(points.astype(np.float32))

    if not parts:
        parts.append(points.astype(np.float32))

    return np.concatenate(parts, axis=1)


# ── dataset ──────────────────────────────────────────────────────────────


class SingleSceneTrainingPackDataset(Dataset):
    """One scene, one sample, one granularity.

    ``__getitem__(0)`` returns a dict of tensors with all fields the
    downstream model and loss need.

    Parameters
    ----------
    scene_dir:
        Path to the scene directory (or directly to the pack subdirectory).
    granularity:
        Which pseudo-label granularity to load.
    use_colors:
        Use RGB from ``colors.npy`` as input features (else zeros).
    append_xyz:
        Append XYZ coordinates to the feature vector.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        granularity: float,
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
    ) -> None:
        super().__init__()
        self.scene: TrainingPackScene = load_training_pack_scene(
            scene_dir, granularity
        )
        self.granularity = granularity
        self._use_colors = use_colors
        self._append_xyz = append_xyz

        self._features = build_input_features(
            self.scene.points,
            self.scene.colors,
            use_colors=use_colors,
            append_xyz=append_xyz,
        )

        print_training_pack_summary(self.scene)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx != 0:
            raise IndexError(idx)

        return {
            "scene_id": self.scene.scene_id,
            "scene_dir": str(self.scene.scene_dir),
            "points": torch.from_numpy(self.scene.points).float(),
            "features": torch.from_numpy(self._features).float(),
            "labels": torch.from_numpy(self.scene.labels).long(),
            "valid_points": torch.from_numpy(self.scene.valid_points).bool(),
            "seen_points": torch.from_numpy(self.scene.seen_points).bool(),
            "supervision_mask": torch.from_numpy(self.scene.supervision_mask).bool(),
            "scene_meta": self.scene.scene_meta,
            "granularity": self.granularity,
        }

    # ------------------------------------------------------------------ #

    @property
    def scene_id(self) -> str:
        return self.scene.scene_id

    @property
    def num_points(self) -> int:
        return self.scene.num_points

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]


class MultiGranSceneDataset(Dataset):
    """One scene, one sample, all granularities.

    ``__getitem__(0)`` returns a dict with ``labels_by_granularity``
    mapping granularity keys to label tensors.

    Parameters
    ----------
    scene_dir:
        Path to the scene directory (or directly to the pack subdirectory).
    granularities:
        Dot-free granularity keys, e.g. ``("g02", "g05", "g08")``.
    use_colors:
        Use RGB from ``colors.npy`` as input features (else zeros).
    append_xyz:
        Append XYZ coordinates to the feature vector.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        granularities: tuple[str, ...] = ("g02", "g05", "g08"),
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
    ) -> None:
        super().__init__()
        self.scene: MultiGranTrainingPackScene = load_training_pack_scene_multi(
            scene_dir, granularities,
        )
        self.granularities = granularities
        self._use_colors = use_colors
        self._append_xyz = append_xyz

        self._features = build_input_features(
            self.scene.points,
            self.scene.colors,
            use_colors=use_colors,
            append_xyz=append_xyz,
        )

        log.info(
            "[%s] Loaded %d granularities: %s  (%d points, %d features)",
            self.scene.scene_id,
            len(granularities),
            ", ".join(granularities),
            self.scene.num_points,
            self._features.shape[1],
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx != 0:
            raise IndexError(idx)

        labels_by_gran = {
            g: torch.from_numpy(self.scene.labels_by_granularity[g]).long()
            for g in self.granularities
        }

        return {
            "scene_id": self.scene.scene_id,
            "scene_dir": str(self.scene.scene_dir),
            "points": torch.from_numpy(self.scene.points).float(),
            "features": torch.from_numpy(self._features).float(),
            "labels_by_granularity": labels_by_gran,
            "valid_points": torch.from_numpy(self.scene.valid_points).bool(),
            "seen_points": torch.from_numpy(self.scene.seen_points).bool(),
            "supervision_mask": torch.from_numpy(self.scene.supervision_mask).bool(),
            "scene_meta": self.scene.scene_meta,
            "granularities": self.granularities,
        }

    @property
    def scene_id(self) -> str:
        return self.scene.scene_id

    @property
    def num_points(self) -> int:
        return self.scene.num_points

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]


# ── smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m student.data.single_scene_dataset <scene_dir> [granularity]")
        raise SystemExit(1)

    scene_dir = sys.argv[1]
    granularity = float(sys.argv[2]) if len(sys.argv) == 3 else 0.5

    ds = SingleSceneTrainingPackDataset(scene_dir, granularity=granularity)
    print(f"\nlen(dataset)        : {len(ds)}")
    print(f"feature_dim         : {ds.feature_dim}")

    sample = ds[0]
    print(f"\nsample keys         : {sorted(k for k in sample if isinstance(sample[k], torch.Tensor))}")
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: {v.shape}  {v.dtype}")
        else:
            print(f"  {k:20s}: {v}")
