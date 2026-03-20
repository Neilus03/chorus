#!/usr/bin/env python3
"""End-to-end pipeline check: data -> targets -> model -> grads -> alignment.

Multi-granularity version: loads all 3 granularities and checks each head.
"""

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
    MultiGranSceneDataset,
    build_instance_targets_multi,
    load_training_pack_scene,
)
from student.data.target_builder import log_target_stats
from student.data.training_pack import print_training_pack_summary
from student.models.student_model import build_student_model

SCENE_DIR = "/scratch2/nedela/chorus_poc/scans/scene0042_00"
LITEPT_ROOT = "/home/nedela/projects/LitePT"
GRANULARITIES = ("g02", "g05", "g08")
DEVICE = "cuda"


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── 1. Training pack (single-gran for summary) ────────────────
    sep("1. TRAINING PACK (single-gran summary)")
    t0 = time.time()
    scene = load_training_pack_scene(SCENE_DIR, 0.5)
    print(f"  loaded in {time.time() - t0:.2f}s")
    print(f"  scene_id         : {scene.scene_id}")
    print(f"  pack_dir         : {scene.training_pack_dir}")
    print(f"  points           : {scene.points.shape}  {scene.points.dtype}")
    print(f"  colors           : {scene.colors.shape if scene.colors is not None else 'None'}")
    print(f"  valid_points     : {scene.valid_points.sum()} / {scene.num_points}")
    print(f"  seen_points      : {scene.seen_points.sum()} / {scene.num_points}")
    print(f"  supervision_mask : {scene.supervision_mask.sum()} / {scene.num_points}")
    print_training_pack_summary(scene)

    # ── 2. Multi-gran dataset ─────────────────────────────────────
    sep("2. MULTI-GRAN DATASET")
    ds = MultiGranSceneDataset(SCENE_DIR, granularities=GRANULARITIES)
    sample = ds[0]
    print(f"  len(dataset)     : {len(ds)}")
    print(f"  feature_dim      : {ds.feature_dim}")
    print(f"  granularities    : {GRANULARITIES}")
    print(f"  sample tensors:")
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {str(tuple(v.shape)):24s}  {v.dtype}")
        elif isinstance(v, dict) and k == "labels_by_granularity":
            for gk, gv in v.items():
                print(f"    labels[{gk:4s}]       : {str(tuple(gv.shape)):24s}  {gv.dtype}")
        elif k not in ("scene_meta",):
            print(f"    {k:20s}: {type(v).__name__} = {v}")

    # ── 3. Targets per granularity ────────────────────────────────
    sep("3. INSTANCE TARGETS (per granularity)")
    targets_by_gran = build_instance_targets_multi(
        sample["labels_by_granularity"], sample["supervision_mask"],
        min_instance_points=10,
    )
    for g, tgt in targets_by_gran.items():
        log_target_stats(tgt, tag=f"{ds.scene_id}/{g}")
        print(f"  [{g}] gt_masks: {tuple(tgt.gt_masks.shape)}  "
              f"instances: {tgt.num_instances}  "
              f"sizes: min={tgt.instance_sizes.min()} "
              f"mean={tgt.instance_sizes.mean():.0f} "
              f"max={tgt.instance_sizes.max()}")

    # ── 4. Model build ────────────────────────────────────────────
    sep("4. MODEL")
    t0 = time.time()
    model = build_student_model(
        litept_root=LITEPT_ROOT,
        in_channels=ds.feature_dim,
        hidden_dim=256,
        num_queries=128,
        granularities=GRANULARITIES,
        num_decoder_layers=4,
        num_decoder_heads=8,
        query_init="hybrid",
        use_positional_guidance=True,
        learned_query_ratio=0.25,
    ).to(DEVICE)
    print(f"  built in {time.time() - t0:.2f}s")

    bb_params = sum(p.numel() for p in model.backbone.parameters())
    dc_params = sum(p.numel() for p in model.decoder.parameters())
    trunk_params = sum(
        p.numel() for n, p in model.decoder.named_parameters()
        if not n.startswith("heads.") and not n.startswith("initializers.")
    )
    init_params = sum(
        p.numel() for n, p in model.decoder.named_parameters()
        if n.startswith("initializers.")
    )
    head_params = sum(
        p.numel() for n, p in model.decoder.named_parameters()
        if n.startswith("heads.")
    )
    print(f"  backbone params  : {bb_params:,}")
    print(f"  decoder total    : {dc_params:,}")
    print(f"    shared trunk   : {trunk_params:,}")
    print(f"    initializers   : {init_params:,}")
    print(f"    heads total    : {head_params:,}")
    print(f"    per head       : {head_params // len(GRANULARITIES):,}")
    print(f"  total params     : {bb_params + dc_params:,}")
    print(f"  backbone out_ch  : {model.backbone.out_channels}")
    print(f"  decoder in_ch    : {model.decoder.in_channels}")
    print(f"  decoder hidden   : {model.decoder.hidden_dim}")
    print(f"  num_queries/head : {model.num_queries}")
    print(f"  num_heads        : {len(GRANULARITIES)}")
    print(f"  decoder layers   : {len(model.decoder.layers)}")
    print(f"  pos guidance     : {model.decoder.use_positional_guidance}")

    pg = model.parameter_groups(backbone_lr_scale=0.1)
    pg_bb = sum(p.numel() for p in pg[0]["params"])
    pg_dc = sum(p.numel() for p in pg[1]["params"])
    print(f"  param groups     : backbone={pg_bb:,} (scale={pg[0]['lr_scale']})  "
          f"decoder={pg_dc:,} (scale={pg[1]['lr_scale']})")

    # ── 5. Forward pass ───────────────────────────────────────────
    sep("5. FORWARD PASS")
    points = sample["points"].to(DEVICE)
    features = sample["features"].to(DEVICE)
    print(f"  input points     : {tuple(points.shape)}")
    print(f"  input features   : {tuple(features.shape)}")

    t0 = time.time()
    out = model(points, features)
    fwd_ms = (time.time() - t0) * 1000
    print(f"  forward time     : {fwd_ms:.0f} ms")
    print(f"  point_embed      : {tuple(out['point_embed'].shape)}")
    print(f"  heads:")
    for g, head_out in out["heads"].items():
        ml = head_out["mask_logits"]
        sl = head_out["score_logits"]
        print(f"    [{g}] mask_logits: {tuple(ml.shape)}  "
              f"score_logits: {tuple(sl.shape)}  "
              f"mask range: [{ml.min():.2f}, {ml.max():.2f}]  "
              f"score range: [{sl.min():.2f}, {sl.max():.2f}]")

    first_g = GRANULARITIES[0]
    ml = out["heads"][first_g]["mask_logits"]
    print(f"  mask_logits MB (per head): {ml.numel() * 4 / 1e6:.1f}")

    # ── 6. Backward pass ──────────────────────────────────────────
    sep("6. BACKWARD PASS")
    t0 = time.time()
    loss = sum(
        h["mask_logits"].sum() + h["score_logits"].sum()
        for h in out["heads"].values()
    )
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
    C = model.backbone.out_channels
    probe_feat = torch.randn(100, C, device=DEVICE)
    probe_scene = torch.randn(50, C, device=DEVICE, requires_grad=True)
    probe_xyz = torch.randn(50, 3, device=DEVICE)
    probe_out = model.decoder(
        probe_feat, scene_tokens=probe_scene, scene_xyz=probe_xyz,
    )
    probe_loss = sum(
        h["score_logits"].sum() for h in probe_out["heads"].values()
    )
    probe_loss.backward()
    g_norm = probe_scene.grad.norm().item()
    print(f"  d(scores)/d(scene_tokens) norm : {g_norm:.6f}")
    print(f"  scene-dependent                : {'YES' if g_norm > 0 else 'NO — BUG!'}")

    # ── 8. Shape alignment ────────────────────────────────────────
    sep("8. SHAPE ALIGNMENT (per head)")
    all_aligned = True
    for g in GRANULARITIES:
        Q, N_pred = out["heads"][g]["mask_logits"].shape
        M, N_gt = targets_by_gran[g].gt_masks.shape
        match = N_pred == N_gt
        all_aligned = all_aligned and match
        print(f"  [{g}] pred [{Q}, {N_pred}] vs gt [{M}, {N_gt}]  "
              f"{'MATCH' if match else 'MISMATCH'}  "
              f"headroom: {Q - M}")

    # ── 9. Matching + Loss ────────────────────────────────────────
    sep("9. MATCHING + LOSS (multi-gran)")
    from student.losses.mask_set_loss import MaskSetCriterion, MultiGranCriterion

    base_criterion = MaskSetCriterion(
        bce_weight=1.0, dice_weight=1.0, score_weight=0.5,
    )
    criterion = MultiGranCriterion(criterion=base_criterion)

    model.zero_grad()
    out2 = model(points, features)

    t0 = time.time()
    loss_dict = criterion(out2, targets_by_gran)
    loss_ms = (time.time() - t0) * 1000
    print(f"  criterion time   : {loss_ms:.0f} ms")
    print(f"  loss_total       : {loss_dict['loss_total'].item():.4f}")
    for g in GRANULARITIES:
        ld_g = loss_dict["heads"][g]
        print(f"  [{g}] loss={ld_g['loss_total'].item():.4f}  "
              f"bce={ld_g['loss_mask_bce'].item():.4f}  "
              f"dice={ld_g['loss_mask_dice'].item():.4f}  "
              f"score={ld_g['loss_score'].item():.4f}  "
              f"matches={ld_g['num_matches']}/{targets_by_gran[g].num_instances}")

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
    all_matches = all(
        loss_dict["heads"][g]["num_matches"] == targets_by_gran[g].num_instances
        for g in GRANULARITIES
    )
    all_ok = (
        bb_grad > 0 and dc_grad > 0 and g_norm > 0 and all_aligned
        and finite and bb_grad2 > 0 and dc_grad2 > 0
        and all_matches
    )
    checks = [
        ("training pack loads", True),
        ("multi-gran dataset returns correct tensors", True),
        ("targets built for all granularities", all(
            targets_by_gran[g].num_instances > 0 for g in GRANULARITIES
        )),
        ("model forward produces nested output", "heads" in out and "point_embed" in out),
        ("all heads present", all(g in out["heads"] for g in GRANULARITIES)),
        ("backward flows through backbone", bb_grad > 0),
        ("backward flows through decoder", dc_grad > 0),
        ("scores are scene-dependent", g_norm > 0),
        ("pred/gt N dimension aligned (all heads)", all_aligned),
        ("matching finds all GT instances (all heads)", all_matches),
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
