#!/usr/bin/env python3
"""End-to-end pipeline check: data → targets → model → grads → alignment."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_STUDENT_PKG = Path(__file__).resolve().parent.parent
if str(_STUDENT_PKG) not in sys.path:
    sys.path.insert(0, str(_STUDENT_PKG))

import numpy as np
import torch

from student.data import (
    SingleSceneTrainingPackDataset,
    build_instance_targets,
    load_training_pack_scene,
)
from student.data.target_builder import log_target_stats
from student.data.training_pack import print_training_pack_summary
from student.models.student_model import build_student_model

SCENE_DIR = "/scratch2/nedela/chorus_poc/scans/scene0042_00"
LITEPT_ROOT = "/home/nedela/projects/LitePT"
GRANULARITY = 0.5
DEVICE = "cuda"


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── 1. Training pack ──────────────────────────────────────────
    sep("1. TRAINING PACK")
    t0 = time.time()
    scene = load_training_pack_scene(SCENE_DIR, GRANULARITY)
    print(f"  loaded in {time.time() - t0:.2f}s")
    print(f"  scene_id         : {scene.scene_id}")
    print(f"  pack_dir         : {scene.training_pack_dir}")
    print(f"  points           : {scene.points.shape}  {scene.points.dtype}")
    print(f"  colors           : {scene.colors.shape if scene.colors is not None else 'None'}")
    print(f"  labels           : {scene.labels.shape}  min={scene.labels.min()} max={scene.labels.max()}")
    print(f"  valid_points     : {scene.valid_points.sum()} / {scene.num_points}")
    print(f"  seen_points      : {scene.seen_points.sum()} / {scene.num_points}")
    print(f"  supervision_mask : {scene.supervision_mask.sum()} / {scene.num_points}")
    stats = print_training_pack_summary(scene)

    # ── 2. Dataset ────────────────────────────────────────────────
    sep("2. DATASET")
    ds = SingleSceneTrainingPackDataset(SCENE_DIR, granularity=GRANULARITY)
    sample = ds[0]
    print(f"  len(dataset)     : {len(ds)}")
    print(f"  feature_dim      : {ds.feature_dim}")
    print(f"  sample tensors:")
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {str(tuple(v.shape)):24s}  {v.dtype}")
        else:
            print(f"    {k:20s}: {type(v).__name__}")

    # ── 3. Targets ────────────────────────────────────────────────
    sep("3. INSTANCE TARGETS")
    targets = build_instance_targets(
        sample["labels"], sample["supervision_mask"], min_instance_points=10,
    )
    log_target_stats(targets, tag=f"{ds.scene_id}/g{GRANULARITY}")
    print(f"  gt_masks         : {tuple(targets.gt_masks.shape)}  {targets.gt_masks.dtype}")
    print(f"  instance_ids     : {targets.instance_ids.shape}  {targets.instance_ids.dtype}")
    print(f"  num_instances    : {targets.num_instances}")
    print(f"  instance_sizes   : min={targets.instance_sizes.min()} "
          f"mean={targets.instance_sizes.mean():.0f} "
          f"max={targets.instance_sizes.max()}")
    supervised_in_gt = targets.gt_masks.any(dim=0).sum().item()
    print(f"  points covered by at least 1 GT mask: {supervised_in_gt}")

    # ── 4. Model build ───────────────────────────────────────────
    sep("4. MODEL")
    t0 = time.time()
    model = build_student_model(
        litept_root=LITEPT_ROOT,
        in_channels=ds.feature_dim,
        hidden_dim=256,
        num_queries=128,
    ).to(DEVICE)
    print(f"  built in {time.time() - t0:.2f}s")

    bb_params = sum(p.numel() for p in model.backbone.parameters())
    dc_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  backbone params  : {bb_params:,}")
    print(f"  decoder params   : {dc_params:,}")
    print(f"  total params     : {bb_params + dc_params:,}")
    print(f"  backbone out_ch  : {model.backbone.out_channels}")
    print(f"  decoder in_ch    : {model.decoder.in_channels}")
    print(f"  decoder hidden   : {model.decoder.hidden_dim}")
    print(f"  num_queries      : {model.num_queries}")

    # ── 5. Forward pass ──────────────────────────────────────────
    sep("5. FORWARD PASS")
    points = sample["points"].to(DEVICE)
    features = sample["features"].to(DEVICE)
    print(f"  input points     : {tuple(points.shape)}")
    print(f"  input features   : {tuple(features.shape)}")

    t0 = time.time()
    out = model(points, features)
    fwd_ms = (time.time() - t0) * 1000
    print(f"  forward time     : {fwd_ms:.0f} ms")
    print(f"  outputs:")
    for k, v in out.items():
        extra = ""
        if k == "mask_logits":
            extra = f"  min={v.min():.2f} max={v.max():.2f} mean={v.mean():.2f}"
        elif k == "score_logits":
            extra = f"  min={v.min():.2f} max={v.max():.2f} mean={v.mean():.2f}"
        print(f"    {k:14s}: {str(tuple(v.shape)):24s}  {v.dtype}{extra}")

    ml = out["mask_logits"]
    print(f"  mask_logits MB   : {ml.numel() * 4 / 1e6:.1f}")

    # ── 6. Backward pass ─────────────────────────────────────────
    sep("6. BACKWARD PASS")
    t0 = time.time()
    loss = out["mask_logits"].sum() + out["score_logits"].sum()
    loss.backward()
    bwd_ms = (time.time() - t0) * 1000
    print(f"  backward time    : {bwd_ms:.0f} ms")

    bb_grad = sum(1 for p in model.backbone.parameters() if p.grad is not None)
    bb_total = sum(1 for _ in model.backbone.parameters())
    dc_grad = sum(1 for p in model.decoder.parameters() if p.grad is not None)
    dc_total = sum(1 for _ in model.decoder.parameters())
    print(f"  backbone grads   : {bb_grad} / {bb_total}")
    print(f"  decoder grads    : {dc_grad} / {dc_total}")

    # ── 7. Score scene-dependence ─────────────────────────────────
    sep("7. SCORE SCENE-DEPENDENCE (gradient proof)")
    model.zero_grad()
    probe = torch.randn(100, model.backbone.out_channels, device=DEVICE, requires_grad=True)
    model.decoder(probe)["score_logits"].sum().backward()
    g = probe.grad.norm().item()
    print(f"  d(scores)/d(point_feat) norm : {g:.6f}")
    print(f"  scene-dependent              : {'YES' if g > 0 else 'NO — BUG!'}")

    # ── 8. Shape alignment ────────────────────────────────────────
    sep("8. SHAPE ALIGNMENT")
    Q, N_pred = out["mask_logits"].shape
    M, N_gt = targets.gt_masks.shape
    print(f"  pred masks       : [{Q}, {N_pred}]")
    print(f"  gt masks         : [{M}, {N_gt}]")
    match = N_pred == N_gt
    print(f"  N dimension      : {'MATCH' if match else 'MISMATCH'}")
    print(f"  Q={Q} query slots for M={M} GT instances (headroom: {Q - M})")

    # ── 9. Matching + Loss ────────────────────────────────────────
    sep("9. MATCHING + LOSS")
    from student.losses.mask_set_loss import MaskSetCriterion

    criterion = MaskSetCriterion(
        bce_weight=1.0, dice_weight=1.0, score_weight=0.5,
    )

    model.zero_grad()
    out2 = model(points, features)

    t0 = time.time()
    loss_dict = criterion(out2, targets)
    loss_ms = (time.time() - t0) * 1000
    print(f"  criterion time   : {loss_ms:.0f} ms")
    print(f"  cost_matrix      : {loss_dict['cost_matrix_shape']}")
    print(f"  num_matches      : {loss_dict['num_matches']} / {M} GT instances")
    print(f"  matched pred idx : {loss_dict['matched_pred_indices'][:8]}{'...' if loss_dict['num_matches'] > 8 else ''}")
    print(f"  matched gt idx   : {loss_dict['matched_gt_indices'][:8]}{'...' if loss_dict['num_matches'] > 8 else ''}")
    print(f"  loss_mask_bce    : {loss_dict['loss_mask_bce']:.4f}")
    print(f"  loss_mask_dice   : {loss_dict['loss_mask_dice']:.4f}")
    print(f"  loss_score       : {loss_dict['loss_score']:.4f}")
    print(f"  loss_total       : {loss_dict['loss_total'].item():.4f}")
    finite = loss_dict["loss_total"].isfinite().item()
    print(f"  loss is finite   : {finite}")

    # ── 10. Backward through real loss ────────────────────────────
    sep("10. BACKWARD THROUGH REAL LOSS")
    t0 = time.time()
    loss_dict["loss_total"].backward()
    bwd2_ms = (time.time() - t0) * 1000
    print(f"  backward time    : {bwd2_ms:.0f} ms")

    bb_grad2 = sum(1 for p in model.backbone.parameters() if p.grad is not None)
    dc_grad2 = sum(1 for p in model.decoder.parameters() if p.grad is not None)
    print(f"  backbone grads   : {bb_grad2} / {bb_total}")
    print(f"  decoder grads    : {dc_grad2} / {dc_total}")

    bb_gnorm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), float("inf")).item()
    dc_gnorm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), float("inf")).item()
    print(f"  backbone grad norm : {bb_gnorm:.4f}")
    print(f"  decoder grad norm  : {dc_gnorm:.4f}")

    # ── summary ───────────────────────────────────────────────────
    sep("SUMMARY")
    all_ok = (
        bb_grad > 0 and dc_grad > 0 and g > 0 and match
        and finite and bb_grad2 > 0 and dc_grad2 > 0
        and loss_dict["num_matches"] == M
    )
    checks = [
        ("training pack loads", True),
        ("dataset returns correct tensors", True),
        ("targets built with supervision mask", targets.num_instances > 0),
        ("model forward produces right shapes", Q == 128 and N_pred == scene.num_points),
        ("backward flows through backbone", bb_grad > 0),
        ("backward flows through decoder", dc_grad > 0),
        ("scores are scene-dependent", g > 0),
        ("pred/gt N dimension aligned", match),
        ("enough query slots", Q >= M),
        ("matching finds all GT instances", loss_dict["num_matches"] == M),
        ("loss is finite", finite),
        ("real loss backward reaches backbone", bb_grad2 > 0),
        ("real loss backward reaches decoder", dc_grad2 > 0),
        ("grad norms are finite", np.isfinite(bb_gnorm) and np.isfinite(dc_gnorm)),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print(f"\n{'ALL OK — ready for training loop.' if all_ok else 'SOMETHING FAILED.'}")


if __name__ == "__main__":
    main()
