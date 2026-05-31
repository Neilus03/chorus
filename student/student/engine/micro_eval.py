"""Cheap fixed-scene micro-evaluation for debug observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from student.data.target_builder import build_instance_targets_multi
from student.engine.evaluator import (
    _build_pseudo_gt_ids,
    _compute_clustering_metrics,
    extract_proposals,
    proposals_to_labels,
)
from student.engine.query_diagnostics import (
    compute_duplicate_iou_stats,
    compute_matching_calibration_stats,
)
from student.metrics.official_instance_ap import (
    build_instance_ap_records,
    evaluate_official_and_oracle_ap,
)
from student.models.continuous_base import is_continuous_decoder
from student.models.finetune_wrapper import FineTuningWrapper


_GRAN_KEY_TO_VAL = {
    "g01": 0.1,
    "g02": 0.2,
    "g03": 0.3,
    "g04": 0.4,
    "g05": 0.5,
    "g06": 0.6,
    "g07": 0.7,
    "g08": 0.8,
    "g09": 0.9,
    "g10": 1.0,
}


def granularity_key_to_float(key: str) -> float:
    if key in _GRAN_KEY_TO_VAL:
        return _GRAN_KEY_TO_VAL[key]
    return float(key.replace("g0", "0.").replace("g", "0."))


def _is_continuous_model(model: torch.nn.Module) -> bool:
    m = model.model if isinstance(model, FineTuningWrapper) else model
    return is_continuous_decoder(getattr(m, "decoder", None))


def _cap_sample(sample: dict[str, Any], max_points: int | None) -> dict[str, Any]:
    if max_points is None:
        return sample
    n = int(sample["points"].shape[0])
    cap = int(max_points)
    if n <= cap:
        return sample
    idx = torch.arange(cap, dtype=torch.long)
    out = dict(sample)
    for key in ("points", "features", "valid_points", "seen_points", "supervision_mask"):
        if key in out and isinstance(out[key], torch.Tensor) and out[key].shape[0] == n:
            out[key] = out[key][idx]
    out["labels_by_granularity"] = {
        g: labels[idx] for g, labels in sample["labels_by_granularity"].items()
    }
    out["vertex_indices"] = idx
    out["eval_input_points"] = cap
    out["original_num_points"] = int(sample.get("original_num_points", n))
    out["full_scene"] = False
    return out


def _forward_for_granularity(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    granularity: str,
    *,
    return_debug: bool = False,
) -> dict[str, torch.Tensor]:
    if isinstance(model, FineTuningWrapper):
        pred = model(points, features)
    elif _is_continuous_model(model):
        pred = model(
            points,
            features,
            target_g=granularity_key_to_float(granularity),
            return_debug=return_debug,
        )
    else:
        out = model(points, features, return_debug=return_debug)
        pred = out["heads"][granularity]
    if isinstance(pred, list):
        raise RuntimeError("micro_eval expects one scene at a time")
    return pred


def evaluate_micro_scenes(
    *,
    model: torch.nn.Module,
    dataset: Any,
    scene_indices: list[int],
    criterion: torch.nn.Module,
    device: str,
    granularities: tuple[str, ...],
    min_instance_points: int = 10,
    dense_instance_ids: bool = False,
    max_points: int | None = 60_000,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
    min_points_per_proposal: int = 30,
    topk_values: list[int] | None = None,
) -> dict[str, Any]:
    """Run pseudo-label AP/matching diagnostics on a small fixed scene set."""
    was_training = model.training
    model.eval()
    topk = max(topk_values or [50])
    per_scene: dict[str, Any] = {}
    accum: dict[str, list[float]] = {}

    def add(name: str, value: float | int) -> None:
        accum.setdefault(name, []).append(float(value))

    with torch.no_grad():
        for idx in scene_indices:
            sample = _cap_sample(dataset.get_full_item(idx), max_points)
            scene_id = str(sample["scene_id"])
            points = sample["points"].to(device)
            features = sample["features"].to(device)
            targets_by_gran = build_instance_targets_multi(
                sample["labels_by_granularity"],
                sample["supervision_mask"],
                min_instance_points=min_instance_points,
                dense_instance_ids=dense_instance_ids,
                instance_class_maps=sample.get("instance_classes_by_granularity"),
            )
            scene_payload: dict[str, Any] = {}
            for g in granularities:
                pred = _forward_for_granularity(
                    model,
                    points,
                    features,
                    g,
                    return_debug=False,
                )
                targets = targets_by_gran[g]
                loss_result = criterion(pred, targets, context=f"micro_eval/{g}", granularity_key=g)
                proposals, scores, qidx, prop_stats = extract_proposals(
                    pred["mask_logits"].detach().cpu(),
                    pred["score_logits"].detach().cpu(),
                    score_threshold=float(score_threshold),
                    mask_threshold=float(mask_threshold),
                    min_points=int(min_points_per_proposal),
                    return_stats=True,
                )
                gt_ids = _build_pseudo_gt_ids(targets)
                records = build_instance_ap_records(
                    scene_id=scene_id,
                    gt_ids=gt_ids,
                    proposals=proposals,
                    scores=scores,
                    query_indices=qidx,
                    class_agnostic=True,
                    eval_mask=targets.supervision_mask.detach().cpu().numpy(),
                )
                ap = evaluate_official_and_oracle_ap(records)
                pred_labels = proposals_to_labels(proposals, pred["mask_logits"].shape[1])
                clustering = _compute_clustering_metrics(gt_ids, pred_labels)
                matched_ious = loss_result.get("matched_ious")
                matched_miou = (
                    float(matched_ious.detach().float().mean().item())
                    if isinstance(matched_ious, torch.Tensor) and matched_ious.numel()
                    else 0.0
                )
                duplicate = compute_duplicate_iou_stats(
                    pred["mask_logits"].detach(),
                    pred["score_logits"].detach(),
                    mask_threshold=mask_threshold,
                    topk=topk,
                )
                calibration = compute_matching_calibration_stats(
                    pred["score_logits"].detach(),
                    matched_pred_indices=loss_result.get("matched_pred_indices"),
                    matched_target_indices=loss_result.get("matched_target_indices", loss_result.get("matched_gt_indices")),
                    matched_ious=matched_ious,
                )
                metrics = {
                    "loss": float(loss_result["loss_total"].detach().item()),
                    "pseudo_AP25": float(ap.get("AP25", 0.0) or 0.0),
                    "pseudo_AP50": float(ap.get("AP50", 0.0) or 0.0),
                    "pseudo_oracle_AP25": float(ap.get("oracle_AP25", 0.0) or 0.0),
                    "pseudo_oracle_AP50": float(ap.get("oracle_AP50", 0.0) or 0.0),
                    "matched_mIoU": matched_miou,
                    "NMI": float(clustering.get("NMI", 0.0)),
                    "ARI": float(clustering.get("ARI", 0.0)),
                    "proposal_score_pass": float(prop_stats["num_score_pass"]),
                    "proposal_kept": float(prop_stats["num_proposals"]),
                    "proposal_removed_min_points": float(prop_stats["num_min_points_removed"]),
                    "duplicate_iou_topk_mean": float(duplicate["duplicate_iou_topk_mean"]),
                    "score_gap_matched_minus_unmatched": float(
                        calibration["score_gap_matched_minus_unmatched"]
                    ),
                }
                scene_payload[g] = metrics
                for key, value in metrics.items():
                    add(f"{key}_{g}", value)
            per_scene[scene_id] = scene_payload

    if was_training:
        model.train()

    aggregate = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in sorted(accum.items())
    }
    return {"per_scene": per_scene, "aggregate": aggregate}


def write_micro_eval_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)
