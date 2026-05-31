"""Config-gated local TensorBoard, JSONL, and artifact observability."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from student.data.target_builder import InstanceTargets, build_instance_targets_multi
from student.engine.anchor_diagnostics import (
    compute_anchor_movement_stats,
    compute_anchor_to_centroid_stats,
    compute_scale_selector_stats,
    target_centroids_from_masks,
)
from student.engine.feature_diagnostics import compute_feature_diagnostics
from student.engine.micro_eval import evaluate_micro_scenes, granularity_key_to_float, write_micro_eval_json
from student.engine.query_diagnostics import (
    compute_matching_calibration_stats,
    compute_query_score_mask_stats,
)
from student.engine.visual_debug import write_scene_snapshot

log = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - depends on optional tensorboard install
    SummaryWriter = None  # type: ignore[assignment]


def _enabled(cfg: dict[str, Any] | None) -> bool:
    return bool(cfg and cfg.get("enabled", False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _mean_accum(accum: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(sum(values) / len(values))
        for key, values in sorted(accum.items())
        if values
    }


def _add(accum: dict[str, list[float]], key: str, value: Any) -> None:
    val = _as_float(value)
    if val is not None:
        accum.setdefault(key, []).append(val)


def _target_labels(targets: InstanceTargets) -> torch.Tensor:
    n = int(targets.supervision_mask.shape[0])
    labels = torch.full((n,), -1, dtype=torch.long, device=targets.gt_masks.device)
    for row, inst_id in enumerate(targets.instance_ids):
        labels[targets.gt_masks[row].bool()] = int(inst_id) + 1
    return labels


class DebugObserver:
    """Owns local debug logging for rank-0 training."""

    def __init__(
        self,
        *,
        output_dir: Path | str,
        debug_config: dict[str, Any] | None,
        is_main_process: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cfg = debug_config or {}
        self.is_main_process = bool(is_main_process)
        self.enabled = self.is_main_process and _enabled(self.cfg)
        self.writer: Any | None = None
        self._param_snapshots: dict[str, torch.Tensor] = {}
        if not self.enabled:
            return

        tb_cfg = self.cfg.get("tensorboard", {}) or {}
        if bool(tb_cfg.get("enabled", True)) and SummaryWriter is not None:
            log_dir = self.output_dir / str(tb_cfg.get("log_dir_name", "tensorboard"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(log_dir),
                flush_secs=int(tb_cfg.get("flush_secs", 30)),
            )
            log.info("TensorBoard debug logging enabled: %s", log_dir)
        elif bool(tb_cfg.get("enabled", True)):
            log.warning("TensorBoard requested but torch.utils.tensorboard is unavailable")

    @property
    def scalars_cfg(self) -> dict[str, Any]:
        return self.cfg.get("scalars", {}) or {}

    @property
    def snapshot_cfg(self) -> dict[str, Any]:
        return self.cfg.get("rich_snapshots", {}) or {}

    @property
    def micro_cfg(self) -> dict[str, Any]:
        return self.cfg.get("micro_eval", {}) or {}

    def should_log_step(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        every = int(self.scalars_cfg.get("every_steps", 50) or 0)
        return every > 0 and int(global_step) % every == 0

    def should_return_debug_for_step(self, global_step: int) -> bool:
        if self.should_log_step(global_step):
            return True
        snap = self.snapshot_cfg
        return bool(snap.get("enabled", False)) and snap.get("every_steps") is not None and int(global_step) % int(snap["every_steps"]) == 0

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _iter_scene_debug_inputs(
        self,
        *,
        pred: dict[str, Any] | list[dict[str, Any]],
        loss_result: dict[str, Any],
        targets_by_scene: list[dict[str, InstanceTargets]],
        active_granularities: list[str],
    ) -> list[tuple[int, str, dict[str, Any], InstanceTargets, dict[str, Any]]]:
        preds = pred if isinstance(pred, list) else [pred]
        scene_losses = loss_result.get("debug_scene_losses")
        out: list[tuple[int, str, dict[str, Any], InstanceTargets, dict[str, Any]]] = []
        for scene_idx, scene_pred in enumerate(preds):
            for g in active_granularities:
                if "heads" in scene_pred:
                    head_pred = scene_pred["heads"][g]
                else:
                    head_pred = scene_pred
                targets = targets_by_scene[scene_idx][g]
                if isinstance(scene_losses, list) and scene_idx < len(scene_losses):
                    head_loss = scene_losses[scene_idx]
                else:
                    head_loss = loss_result.get("heads", {}).get(g, loss_result)
                out.append((scene_idx, g, head_pred, targets, head_loss))
        return out

    def collect_step_scalars(
        self,
        *,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        pred: dict[str, Any] | list[dict[str, Any]],
        loss_result: dict[str, Any],
        targets_by_scene: list[dict[str, InstanceTargets]],
        batch: Any,
        active_granularities: list[str],
        sampled_g_key: str | None,
        sampled_g_val: float | None,
        optimizer: torch.optim.Optimizer,
        grad_norm_pre_clip: float,
        grad_norm_post_clip: float,
        device: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        metrics: dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "active_granularities": list(active_granularities),
            "debug/data/scene_ids": list(batch.scene_ids),
        }
        if sampled_g_key is not None:
            metrics["debug/data/active_granularity"] = sampled_g_key
        if sampled_g_val is not None:
            metrics["debug/data/active_granularity_value"] = float(sampled_g_val)

        _add_single = lambda k, v: metrics.__setitem__(k, v) if _as_float(v) is not None else None
        _add_single("debug/loss/total", loss_result.get("loss_total"))
        if "loss_aux" in loss_result:
            _add_single("debug/loss/aux", loss_result["loss_aux"])

        point_counts = list(getattr(batch, "point_counts", []) or [])
        if point_counts:
            metrics["debug/data/num_points_total"] = float(sum(point_counts))
            metrics["debug/data/num_scenes"] = float(len(point_counts))
            metrics["debug/data/max_scene_points"] = float(max(point_counts))
            metrics["debug/data/min_scene_points"] = float(min(point_counts))
            metrics["debug/data/mean_scene_points"] = float(sum(point_counts) / len(point_counts))

        loss_accum: dict[str, list[float]] = {}
        target_accum: dict[str, list[float]] = {}
        query_accum: dict[str, list[float]] = {}
        score_accum: dict[str, list[float]] = {}
        mask_accum: dict[str, list[float]] = {}
        matching_accum: dict[str, list[float]] = {}
        calibration_accum: dict[str, list[float]] = {}
        anchor_accum: dict[str, list[float]] = {}
        scale_accum: dict[str, list[float]] = {}
        local_accum: dict[str, list[float]] = {}
        feature_accum: dict[str, list[float]] = {}

        mask_threshold = float(self.cfg.get("mask_threshold", 0.5))
        min_points = int(self.cfg.get("min_points_per_proposal", 30))
        topk = int(self.snapshot_cfg.get("topk_queries", 50))
        for scene_idx, g, scene_pred, targets, head_loss in self._iter_scene_debug_inputs(
            pred=pred,
            loss_result=loss_result,
            targets_by_scene=targets_by_scene,
            active_granularities=active_granularities,
        ):
            # Loss decomposition.
            for src_key, out_key in (
                ("loss_total", f"debug/loss/loss_{g}"),
                ("loss_mask_bce", f"debug/loss/mask_bce_{g}"),
                ("loss_mask_dice", f"debug/loss/mask_dice_{g}"),
                ("loss_score", f"debug/loss/score_{g}"),
                ("loss_class", f"debug/loss/class_{g}"),
                ("loss_center", f"debug/loss/center_{g}"),
                ("loss_center_aux", f"debug/loss/center_aux_{g}"),
                ("loss_total_unweighted", f"debug/loss/unweighted_total_{g}"),
            ):
                if src_key in head_loss:
                    _add(loss_accum, out_key, head_loss[src_key])
            if f"loss_{g}" in head_loss:
                _add(loss_accum, f"debug/loss/weighted_total_{g}", head_loss[f"loss_{g}"])
            for aux_idx in range(8):
                key = f"loss_aux_layer_{aux_idx}"
                if key in head_loss:
                    _add(loss_accum, f"debug/loss/aux_layer_{aux_idx}_{g}", head_loss[key])

            # Target/data stats.
            n = int(targets.supervision_mask.numel())
            supervised = int(targets.supervision_mask.sum().item())
            valid = int(targets.gt_masks.any(dim=0).sum().item()) if targets.gt_masks.numel() else 0
            sizes = torch.as_tensor(targets.instance_sizes, dtype=torch.float32)
            _add(target_accum, f"debug/target/num_instances_{g}", targets.num_instances)
            _add(target_accum, f"debug/target/supervised_points_{g}", supervised)
            _add(target_accum, f"debug/target/supervised_fraction_{g}", supervised / max(n, 1))
            _add(target_accum, f"debug/target/valid_points_{g}", valid)
            _add(target_accum, f"debug/target/ignore_fraction_{g}", 1.0 - supervised / max(n, 1))
            if sizes.numel():
                for name, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
                    _add(target_accum, f"debug/target/instance_size_{name}_{g}", torch.quantile(sizes, q))
                _add(target_accum, f"debug/target/instance_size_min_{g}", sizes.min())
                _add(target_accum, f"debug/target/instance_size_max_{g}", sizes.max())

            qstats = compute_query_score_mask_stats(
                scene_pred["mask_logits"].detach(),
                scene_pred["score_logits"].detach(),
                scene_pred.get("query_embed"),
                mask_threshold=mask_threshold,
                min_points_per_proposal=min_points,
                topk=topk,
            )
            for key, value in qstats.items():
                if key.startswith(("prob_", "logit_")):
                    _add(score_accum, f"debug/score/{key}_{g}", value)
                elif key.startswith(("area_", "topk_area_", "empty_", "huge_")):
                    _add(mask_accum, f"debug/mask/{key}_{g}", value)
                else:
                    _add(query_accum, f"debug/query/{key}_{g}", value)

            mstats = compute_matching_calibration_stats(
                scene_pred["score_logits"].detach(),
                matched_pred_indices=head_loss.get("matched_pred_indices"),
                matched_target_indices=head_loss.get("matched_target_indices", head_loss.get("matched_gt_indices")),
                matched_ious=head_loss.get("matched_ious"),
            )
            for key, value in mstats.items():
                if key.startswith("score_") or key.startswith("high_score"):
                    _add(calibration_accum, f"debug/calibration/{key}_{g}", value)
                else:
                    _add(matching_accum, f"debug/matching/{key}_{g}", value)

            debug = scene_pred.get("debug", {}) if isinstance(scene_pred.get("debug"), dict) else {}
            q0 = debug.get("query_anchors_initial")
            qlayers = debug.get("query_anchors_by_layer")
            if isinstance(q0, torch.Tensor) and isinstance(qlayers, torch.Tensor):
                astats = compute_anchor_movement_stats(
                    batch.split_tensor(batch.points.to(device))[scene_idx],
                    q0.detach(),
                    qlayers.detach(),
                    query_radii_by_layer=debug.get("query_radii_by_layer"),
                )
                for key, value in astats.items():
                    _add(anchor_accum, f"debug/anchor/{key}_{g}", value)
                cstats = compute_anchor_to_centroid_stats(
                    batch.split_tensor(batch.points.to(device))[scene_idx],
                    targets.gt_masks.to(device),
                    q0.detach(),
                    qlayers.detach(),
                    matched_pred_indices=head_loss.get("matched_pred_indices"),
                    matched_target_indices=head_loss.get("matched_target_indices", head_loss.get("matched_gt_indices")),
                )
                for key, value in cstats.items():
                    _add(anchor_accum, f"debug/anchor/{key}_{g}", value)
            for key, value in compute_scale_selector_stats(scene_pred.get("scale_weights")).items():
                _add(scale_accum, f"debug/scale/{key}_{g}", value)
            diag = scene_pred.get("diagnostics", {}) if isinstance(scene_pred.get("diagnostics"), dict) else {}
            neighbor_by_layer = debug.get("local_neighbor_count_by_layer")
            zero_by_layer = debug.get("local_neighbor_zero_fraction_by_layer")
            if isinstance(neighbor_by_layer, torch.Tensor):
                for layer_idx, value in enumerate(neighbor_by_layer.detach().flatten()):
                    _add(local_accum, f"debug/local/neighbor_count_p50_layer{layer_idx}_{g}", value)
                    _add(local_accum, f"debug/local/neighbor_count_p90_layer{layer_idx}_{g}", value)
            if isinstance(zero_by_layer, torch.Tensor):
                for layer_idx, value in enumerate(zero_by_layer.detach().flatten()):
                    _add(local_accum, f"debug/local/neighbor_zero_fraction_layer{layer_idx}_{g}", value)
            for key, value in diag.items():
                if key.startswith("local_neighbor"):
                    _add(local_accum, f"debug/local/{key}_{g}", value)
                elif key.startswith("local_gate_mean_layer_"):
                    layer = key.removeprefix("local_gate_mean_layer_")
                    _add(local_accum, f"debug/local/gate_mean_layer{layer}_{g}", value)

            point_embed = scene_pred.get("point_embed")
            if isinstance(point_embed, torch.Tensor):
                labels = _target_labels(targets).to(point_embed.device)
                fstats = compute_feature_diagnostics(
                    point_embed.detach(),
                    labels,
                    max_pca_points=int(self.cfg.get("feature_sample_points", 4096)),
                    seed=int(global_step) + scene_idx,
                )
                for key, value in fstats.items():
                    _add(feature_accum, f"debug/feature/{key}", value)

        for group in (
            loss_accum,
            target_accum,
            query_accum,
            score_accum,
            mask_accum,
            matching_accum,
            calibration_accum,
            anchor_accum,
            scale_accum,
            local_accum,
            feature_accum,
        ):
            metrics.update(_mean_accum(group))

        metrics.update(self._collect_optimization_stats(
            model=model,
            optimizer=optimizer,
            grad_norm_pre_clip=grad_norm_pre_clip,
            grad_norm_post_clip=grad_norm_post_clip,
        ))
        return {k: _json_safe(v) for k, v in metrics.items()}

    def _module_bucket(self, name: str) -> str | None:
        lower = name.lower()
        if "backbone" in lower:
            return "backbone"
        if "score" in lower:
            return "score_head"
        if "mask" in lower:
            return "mask_head"
        if "initializer" in lower or "query_init" in lower or "learned_queries" in lower:
            return "query_init"
        if "delta_heads" in lower:
            return "delta_heads"
        if "granularity_encoder" in lower:
            return "granularity_encoder"
        if "local_aggregator" in lower or "local_gates" in lower:
            return "local_aggregation"
        if "self_attn" in lower or "rel_bias" in lower:
            return "relation_attention"
        if "decoder" in lower:
            return "decoder"
        return None

    def _collect_optimization_stats(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_norm_pre_clip: float,
        grad_norm_post_clip: float,
    ) -> dict[str, float]:
        out: dict[str, float] = {
            "debug/grad/pre_clip": float(grad_norm_pre_clip),
            "debug/grad/post_clip": float(grad_norm_post_clip),
            "debug/grad/clipped_bool": float(grad_norm_post_clip < grad_norm_pre_clip - 1e-6),
        }
        grad_sq: dict[str, float] = {}
        params_by_bucket: dict[str, list[torch.nn.Parameter]] = {}
        for name, param in model.named_parameters():
            bucket = self._module_bucket(name)
            if bucket is None:
                continue
            params_by_bucket.setdefault(bucket, []).append(param)
            if param.grad is not None:
                grad_sq[bucket] = grad_sq.get(bucket, 0.0) + float((param.grad.detach().float() ** 2).sum().item())
        for bucket, sq in grad_sq.items():
            out[f"debug/grad/{bucket}"] = math.sqrt(max(sq, 0.0))
        for idx, group in enumerate(optimizer.param_groups):
            out[f"debug/optimization/lr_group_{idx}"] = float(group.get("lr", 0.0))

        sample_budget = int(self.cfg.get("update_ratio_sample_params", 100_000))
        for bucket, params in params_by_bucket.items():
            cur = self._sample_params(params, sample_budget)
            prev = self._param_snapshots.get(bucket)
            if prev is not None and prev.shape == cur.shape:
                update_norm = torch.linalg.norm(cur - prev)
                param_norm = torch.linalg.norm(cur).clamp_min(1e-12)
                out[f"debug/update/{bucket}_update_over_param"] = float((update_norm / param_norm).item())
            self._param_snapshots[bucket] = cur
        return out

    @staticmethod
    def _sample_params(params: list[torch.nn.Parameter], budget: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = max(int(budget), 1)
        for param in params:
            flat = param.detach().float().flatten()
            if flat.numel() == 0:
                continue
            take = min(int(flat.numel()), remaining)
            chunks.append(flat[:take].cpu())
            remaining -= take
            if remaining <= 0:
                break
        return torch.cat(chunks) if chunks else torch.zeros(1)

    def log_scalars(self, metrics: dict[str, Any], *, step: int) -> None:
        if not self.enabled or self.writer is None or not bool(self.scalars_cfg.get("log_to_tensorboard", True)):
            return
        for key, value in metrics.items():
            val = _as_float(value)
            if val is not None:
                self.writer.add_scalar(key, val, int(step))
        self.writer.flush()

    def _selected_indices(self, dataset: Any, scene_ids: list[str], max_scenes: int | None = None) -> list[int]:
        by_id = {scene_id: idx for idx, scene_id in enumerate(dataset.scene_ids)}
        out = [by_id[s] for s in scene_ids if s in by_id]
        if max_scenes is not None:
            out = out[: int(max_scenes)]
        return out

    def run_micro_eval_if_due(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        criterion: torch.nn.Module,
        device: str,
        granularities: tuple[str, ...],
        min_instance_points: int,
        dense_instance_ids: bool,
    ) -> dict[str, Any] | None:
        cfg = self.micro_cfg
        if not self.enabled or not bool(cfg.get("enabled", False)):
            return None
        every = int(cfg.get("every_epochs", 1) or 0)
        if every <= 0 or int(epoch) % every != 0:
            return None
        selected = {
            "train": self._selected_indices(train_dataset, list(cfg.get("train_scenes", []) or [])),
            "val": self._selected_indices(val_dataset, list(cfg.get("val_scenes", []) or [])),
        }
        result: dict[str, Any] = {"epoch": int(epoch), "splits": {}}
        for split, indices in selected.items():
            if not indices:
                continue
            split_result = evaluate_micro_scenes(
                model=model,
                dataset=train_dataset if split == "train" else val_dataset,
                scene_indices=indices,
                criterion=criterion,
                device=device,
                granularities=tuple(cfg.get("granularities", granularities)),
                min_instance_points=min_instance_points,
                dense_instance_ids=dense_instance_ids,
                max_points=(None if bool(cfg.get("full_scene", False)) else cfg.get("max_points", 60_000)),
                score_threshold=float(cfg.get("score_threshold", 0.0)),
                mask_threshold=float(cfg.get("mask_threshold", 0.5)),
                topk_values=list(cfg.get("topk_values", [1, 5, 10, 25, 50, 100])),
            )
            result["splits"][split] = split_result
            for key, value in split_result.get("aggregate", {}).items():
                val = _as_float(value)
                if val is not None and self.writer is not None:
                    self.writer.add_scalar(f"micro_eval/{split}/{key}", val, int(epoch))
        out_path = self.output_dir / "micro_eval" / f"micro_eval_epoch_{int(epoch):06d}.json"
        write_micro_eval_json(result, out_path)
        if self.writer is not None:
            self.writer.flush()
        return result

    def write_rich_snapshots_if_due(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        criterion: torch.nn.Module,
        device: str,
        granularities: tuple[str, ...],
        min_instance_points: int,
        dense_instance_ids: bool,
    ) -> dict[str, Any] | None:
        cfg = self.snapshot_cfg
        if not self.enabled or not bool(cfg.get("enabled", False)):
            return None
        every = int(cfg.get("every_epochs", 0) or 0)
        if every <= 0 or int(epoch) % every != 0:
            return None
        max_scenes = int(cfg.get("max_scenes", 6))
        selections = {
            "train": self._selected_indices(train_dataset, list(cfg.get("train_scenes", []) or []), max_scenes),
            "val": self._selected_indices(val_dataset, list(cfg.get("val_scenes", []) or []), max_scenes),
        }
        snapshot_root = self.output_dir / "debug_snapshots" / f"epoch_{int(epoch):06d}"
        manifest: dict[str, Any] = {"epoch": int(epoch), "splits": {}}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for split, indices in selections.items():
                dataset = train_dataset if split == "train" else val_dataset
                split_payload: dict[str, Any] = {}
                for idx in indices:
                    sample = dataset.get_full_item(idx)
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
                    for g in tuple(cfg.get("granularities", granularities)):
                        pred = model(
                            points,
                            features,
                            target_g=granularity_key_to_float(g),
                            return_debug=True,
                        )
                        if isinstance(pred, list):
                            raise RuntimeError("Snapshot expects one scene")
                        targets = targets_by_gran[g]
                        loss = criterion(pred, targets, context=f"snapshot/{g}", granularity_key=g)
                        scene_out = snapshot_root / split / scene_id / g
                        artifacts = write_scene_snapshot(
                            scene_out,
                            sample=sample,
                            pred=pred,
                            granularity=g,
                            target_labels=sample["labels_by_granularity"][g],
                            matched_pred_indices=loss.get("matched_pred_indices"),
                            matched_target_indices=loss.get("matched_target_indices", loss.get("matched_gt_indices")),
                            matched_ious=loss.get("matched_ious"),
                            topk_queries=int(cfg.get("topk_queries", 50)),
                            max_render_points=int(cfg.get("max_render_points", 150_000)),
                            mask_threshold=float(cfg.get("mask_threshold", 0.5)),
                            save_png=bool(cfg.get("save_png", True)),
                            save_ply=bool(cfg.get("save_ply", True)),
                            save_npz=bool(cfg.get("save_npz", True)),
                        )
                        scene_payload[g] = artifacts
                        if self.writer is not None and bool(cfg.get("log_png_to_tensorboard", True)):
                            self._log_snapshot_images(split, scene_id, g, artifacts, epoch)
                    split_payload[scene_id] = scene_payload
                manifest["splits"][split] = split_payload
        if was_training:
            model.train()
        snapshot_root.mkdir(parents=True, exist_ok=True)
        with (snapshot_root / "manifest.json").open("w", encoding="utf-8") as f:
            import json

            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        if self.writer is not None:
            self.writer.flush()
        return manifest

    def _log_snapshot_images(
        self,
        split: str,
        scene_id: str,
        granularity: str,
        artifacts: dict[str, str],
        epoch: int,
    ) -> None:
        if self.writer is None:
            return
        try:
            import matplotlib.image as mpimg
        except Exception:  # pragma: no cover
            return
        tag_map = {
            "pred_vs_gt_overlay": "pred_vs_gt",
            "query_anchor_trajectories": "query_trajectories",
            "student_point_features_pca": "pca_features",
        }
        for key, tag_suffix in tag_map.items():
            path = artifacts.get(key)
            if not path:
                continue
            img_path = Path(path)
            if not img_path.is_file():
                continue
            img = mpimg.imread(str(img_path))
            if img.ndim == 2:
                img = img[:, :, None]
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            chw = np.transpose(img, (2, 0, 1))
            self.writer.add_image(
                f"snapshots/{split}/{scene_id}/{granularity}/{tag_suffix}",
                chw,
                int(epoch),
            )
