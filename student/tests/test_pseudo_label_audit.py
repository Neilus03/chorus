from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_pseudo_labels import audit_label_alignment


def test_audit_label_alignment_reports_iou_and_query_budget_stats() -> None:
    pseudo = np.asarray([1, 1, 2, 2, 3, -1], dtype=np.int64)
    supervision = np.ones_like(pseudo, dtype=bool)
    real = np.asarray([10, 10, 0, 20, 20, 0], dtype=np.int64)

    stats = audit_label_alignment(
        pseudo,
        supervision,
        real,
        query_count=2,
        min_instance_points=1,
    )

    assert stats["num_points"] == 6
    assert stats["num_pseudo_instances"] == 3
    assert stats["num_real_instances"] == 2
    assert stats["query_to_pseudo_ratio"] == 2 / 3
    assert stats["pseudo_query_deficit"] == 1
    assert stats["pseudo_query_recall_cap"] == 2 / 3
    assert stats["pseudo_size_median"] == 2
    assert stats["real_size_median"] == 2
    assert stats["pseudo_point_fraction_on_real_instances"] == 4 / 5
    assert stats["pseudo_point_fraction_non_real_or_ignored"] == pytest.approx(1 / 5)
    assert stats["pseudo_fraction_best_iou_ge_25"] == 1.0
    assert stats["pseudo_fraction_best_iou_ge_50"] == 2 / 3
    assert stats["real_fraction_best_iou_ge_25"] == 1.0
    assert stats["real_fraction_best_iou_ge_50"] == 1.0


def test_audit_label_alignment_respects_min_instance_points() -> None:
    pseudo = np.asarray([1, 1, 2, 3, 3, -1], dtype=np.int64)
    supervision = np.ones_like(pseudo, dtype=bool)
    real = np.asarray([10, 10, 20, 20, 20, 0], dtype=np.int64)

    stats = audit_label_alignment(
        pseudo,
        supervision,
        real,
        query_count=4,
        min_instance_points=2,
    )

    assert stats["num_pseudo_instances"] == 2
    assert stats["num_pseudo_points"] == 4
    assert stats["pseudo_query_deficit"] == 0
    assert stats["pseudo_query_recall_cap"] == 1.0
