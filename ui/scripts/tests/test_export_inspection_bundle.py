from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_inspection_bundle import (  # noqa: E402
    normalize_granularity_key,
    prediction_arrays_from_logits,
    write_pack_only_bundle,
)


def _write_synthetic_pack(scene_dir: Path) -> Path:
    pack_dir = scene_dir / "training_pack"
    pack_dir.mkdir(parents=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, -1], dtype=np.int32)
    np.save(pack_dir / "points.npy", points)
    np.save(pack_dir / "colors.npy", np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [64, 64, 64]], dtype=np.uint8))
    np.save(pack_dir / "labels_g0.5.npy", labels)
    np.save(pack_dir / "valid_points.npy", (labels >= 0).astype(np.uint8))
    np.save(pack_dir / "seen_points.npy", np.ones(4, dtype=np.uint8))
    np.save(pack_dir / "supervision_mask.npy", (labels >= 0).astype(np.uint8))
    meta = {
        "pack_version": "1.0",
        "dataset": "synthetic",
        "scene_id": scene_dir.name,
        "num_points": 4,
        "granularities": [0.5],
        "label_files": {"g0.5": "labels_g0.5.npy"},
        "optional_files_present": {"colors.npy": True},
    }
    (pack_dir / "scene_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return pack_dir


class ExportInspectionBundleTests(unittest.TestCase):
    def test_normalize_granularity_key(self) -> None:
        self.assertEqual(normalize_granularity_key("g05"), "g05")
        self.assertEqual(normalize_granularity_key("g0.5"), "g05")
        self.assertEqual(normalize_granularity_key(0.8), "g08")

    def test_prediction_arrays_from_logits_score_wins_overlap(self) -> None:
        mask_logits = torch.full((3, 5), -8.0)
        mask_logits[0, [0, 1, 2]] = 8.0
        mask_logits[1, [2, 3]] = 8.0
        mask_logits[2, [4]] = 8.0
        score_logits = torch.tensor([0.0, 2.0, 5.0])

        labels, scores, query_ids, table = prediction_arrays_from_logits(
            mask_logits,
            score_logits,
            score_threshold=0.0,
            mask_threshold=0.5,
            min_points=2,
        )

        self.assertEqual(query_ids.tolist(), [0, 0, 1, 1, -1])
        self.assertEqual(int(labels[2]), 1)
        self.assertGreater(float(scores[2]), float(scores[0]))
        self.assertFalse(table[2]["kept"])

    def test_write_pack_only_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_dir = tmp_path / "scene0000_00"
            _write_synthetic_pack(scene_dir)
            manifest_path = write_pack_only_bundle(
                scene_dir,
                tmp_path / "bundle" / "scene0000_00",
                gt_benchmarks=(),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "chorus_inspection_bundle/v1")
            self.assertEqual(manifest["scene_id"], "scene0000_00")
            self.assertEqual(manifest["arrays"]["points"]["shape"], [4, 3])
            self.assertIn("g05", manifest["labels"]["pseudo"])
            self.assertEqual(manifest["labels"]["predictions"], {})


if __name__ == "__main__":
    unittest.main()
