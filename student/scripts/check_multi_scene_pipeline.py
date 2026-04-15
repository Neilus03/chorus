#!/usr/bin/env python3
"""End-to-end pipeline check for multi-scene training.

Validates data loading, target building, model forward/backward,
cache invalidation, loss, optimizer, LR schedule, and evaluation
on real scenes — without running a full training loop.

Examples::

    python scripts/check_multi_scene_pipeline.py
    python scripts/check_multi_scene_pipeline.py --config configs/multi_scene_10_3.yaml
    python scripts/check_multi_scene_pipeline.py --device cpu
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDENT_ROOT = _SCRIPT_DIR.parent
if str(_STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDENT_ROOT))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from student.config_utils import (
    load_config,
    parse_granularities,
    resolve_num_queries,
    set_seed,
)
from student.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from student.data.target_builder import build_instance_targets_multi, log_target_stats
from student.losses import MaskSetCriterion, MultiGranCriterion
from student.models.student_model import build_student_model


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _maybe_cuda_empty_cache(device: torch.device | str) -> None:
    """Drop allocator fragmentation / free large tensors between heavy sections."""
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="End-to-end pipeline check for multi-scene training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_STUDENT_ROOT / "configs" / "multi_scene_10_3.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-scenes", type=int, default=2,
        help="Max scenes per split to check (default: 2, for speed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    model_cfg = cfg["model"]
    loss_cfg = cfg.get("loss", {})
    exp_cfg = cfg.get("experiment", {})

    granularities = parse_granularities(data_cfg)
    device = args.device or train_cfg.get("device", "cuda:0")
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)

    print(f"\n  config           : {Path(args.config).resolve()}")
    print(f"  granularities    : {granularities}")
    print(f"  device           : {device}")
    print(f"  max_scenes check : {args.max_scenes}")

    # ── 1. Scene lists ───────────────────────────────────────────
    sep("1. SCENE LISTS")
    scans_root = Path(data_cfg["scans_root"])
    train_split = _STUDENT_ROOT / data_cfg["train_split"]
    val_split = _STUDENT_ROOT / data_cfg["val_split"]

    train_dirs = build_scene_list(train_split, scans_root)
    val_dirs = build_scene_list(val_split, scans_root)
    print(f"  train split      : {train_split} ({len(train_dirs)} scenes)")
    print(f"  val split        : {val_split} ({len(val_dirs)} scenes)")
    for i, d in enumerate(train_dirs):
        print(f"    train[{i}] {d.name}")
    for i, d in enumerate(val_dirs):
        print(f"    val[{i}]   {d.name}")

    overlap = set(d.name for d in train_dirs) & set(d.name for d in val_dirs)
    print(f"  train/val overlap: {len(overlap)} {'— OK' if not overlap else '— OVERLAP: ' + str(overlap)}")

    # ── 2. Datasets ──────────────────────────────────────────────
    sep("2. DATASETS (preload)")
    n_train = min(len(train_dirs), args.max_scenes)
    n_val = min(len(val_dirs), args.max_scenes)

    t0 = time.time()
    max_pts = data_cfg.get("max_points", None)
    train_ds = MultiSceneDataset(
        train_dirs[:n_train], granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        max_points=max_pts,
    )
    val_ds = MultiSceneDataset(
        val_dirs[:n_val], granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        max_points=max_pts,
    )
    print(f"  loaded in {time.time() - t0:.2f}s")
    if max_pts is not None:
        print(f"  max_points       : {max_pts} (random subsample when N > cap)")
    print(f"  train dataset    : {len(train_ds)} scenes")
    print(f"  val dataset      : {len(val_ds)} scenes")
    print(f"  scene_ids (train): {train_ds.scene_ids}")
    print(f"  scene_ids (val)  : {val_ds.scene_ids}")

    # ── 3. Sample dict structure ─────────────────────────────────
    sep("3. SAMPLE DICT STRUCTURE")
    required_keys = {
        "scene_id", "scene_dir", "points", "features",
        "labels_by_granularity", "valid_points", "seen_points",
        "supervision_mask", "scene_meta", "granularities",
    }
    optional_keys = {"vertex_indices"}
    sample = train_ds[0]
    actual_keys = set(sample.keys())
    keys_ok = required_keys <= actual_keys and (actual_keys - required_keys) <= optional_keys
    print(f"  required keys    : {sorted(required_keys)}")
    print(f"  actual keys      : {sorted(actual_keys)}")
    print(f"  keys match       : {'YES' if keys_ok else 'NO — missing: ' + str(required_keys - actual_keys) + ' extra: ' + str(actual_keys - required_keys - optional_keys)}")

    print(f"  sample tensors:")
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {str(tuple(v.shape)):24s}  {v.dtype}")
        elif isinstance(v, dict) and k == "labels_by_granularity":
            for gk, gv in v.items():
                print(f"    labels[{gk:4s}]       : {str(tuple(gv.shape)):24s}  {gv.dtype}")

    # Verify all scenes have consistent structure
    for i in range(len(train_ds)):
        s = train_ds[i]
        assert set(s.keys()) == expected_keys, f"train[{i}] keys mismatch"
        assert s["points"].ndim == 2 and s["points"].shape[1] == 3
        assert s["features"].shape[0] == s["points"].shape[0]
        assert set(s["labels_by_granularity"].keys()) == set(granularities)
    for i in range(len(val_ds)):
        s = val_ds[i]
        sk = set(s.keys())
        assert required_keys <= sk and (sk - required_keys) <= optional_keys, (
            f"val[{i}] keys mismatch"
        )
    print(f"  all {len(train_ds) + len(val_ds)} samples structurally valid")

    # ── 4. Targets ───────────────────────────────────────────────
    sep("4. INSTANCE TARGETS")
    min_inst = data_cfg.get("min_instance_points", 10)
    targets_s0 = build_instance_targets_multi(
        sample["labels_by_granularity"],
        sample["supervision_mask"],
        min_instance_points=min_inst,
    )
    for g, tgt in targets_s0.items():
        log_target_stats(tgt, tag=f"{sample['scene_id']}/{g}")
    targets_ok = all(targets_s0[g].num_instances > 0 for g in granularities)
    print(f"  all heads have instances: {'YES' if targets_ok else 'NO — check pseudo-labels'}")

    # ── 5. DataLoader ────────────────────────────────────────────
    sep("5. DATALOADER (shuffle + collate)")
    loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda batch: batch[0], num_workers=0,
    )
    batch = next(iter(loader))
    loader_ok = isinstance(batch, dict) and "scene_id" in batch
    print(f"  collate returns dict: {'YES' if loader_ok else 'NO'}")
    print(f"  batch scene_id   : {batch['scene_id']}")

    # ── 6. Model ─────────────────────────────────────────────────
    sep("6. MODEL")
    t0 = time.time()
    bb_cfg = model_cfg["backbone"]
    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    model = build_student_model(
        litept_root=bb_cfg["litept_root"],
        in_channels=bb_cfg.get("in_channels", 3),
        grid_size=bb_cfg.get("grid_size", 0.02),
        litept_variant=bb_cfg.get("litept_variant", "litept_s_star"),
        litept_kwargs=bb_cfg.get("litept_kwargs", None),
        hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        num_queries=num_queries,
        num_queries_by_granularity=num_queries_by_granularity,
        granularities=granularities,
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        num_decoder_heads=model_cfg.get("num_decoder_heads", 8),
        query_init=model_cfg.get("query_init", "hybrid"),
        use_positional_guidance=model_cfg.get("use_positional_guidance", True),
        learned_query_ratio=model_cfg.get("learned_query_ratio", 0.25),
        multi_scale=bb_cfg.get("multi_scale", False),
    ).to(device)
    print(f"  built in {time.time() - t0:.2f}s")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total params     : {total_params:,}")
    print(f"  backbone out_ch  : {model.backbone.out_channels}")
    print(f"  num_queries      : {model.decoder.num_queries_per_head}")
    print(f"  decoder layers   : {len(model.decoder.layers)}")

    # ── 7. Forward + cache invalidation ──────────────────────────
    sep("7. FORWARD + CACHE INVALIDATION")
    model.train()

    s0 = train_ds[0]
    s1 = train_ds[min(1, len(train_ds) - 1)]
    pts0 = s0["points"].to(device)
    feat0 = s0["features"].to(device)
    pts1 = s1["points"].to(device)
    feat1 = s1["features"].to(device)

    # Forward scene 0
    t0 = time.time()
    out0 = model(pts0, feat0)
    fwd0_ms = (time.time() - t0) * 1000
    print(f"  scene 0 ({s0['scene_id']}, {pts0.shape[0]} pts): {fwd0_ms:.0f} ms")

    cache_before = model.backbone._cached_voxelization is not None
    print(f"  cache populated after fwd: {cache_before}")

    # Clear cache and forward scene 1
    model.backbone._cached_voxelization = None
    cache_after_clear = model.backbone._cached_voxelization is None
    print(f"  cache cleared            : {cache_after_clear}")

    t0 = time.time()
    out1 = model(pts1, feat1)
    fwd1_ms = (time.time() - t0) * 1000
    print(f"  scene 1 ({s1['scene_id']}, {pts1.shape[0]} pts): {fwd1_ms:.0f} ms")

    # Verify outputs have correct N dimension per scene
    n0 = out0["heads"][granularities[0]]["mask_logits"].shape[1]
    n1 = out1["heads"][granularities[0]]["mask_logits"].shape[1]
    cache_ok = (n0 == pts0.shape[0]) and (n1 == pts1.shape[0])
    print(f"  out0 N={n0} (expect {pts0.shape[0]}), out1 N={n1} (expect {pts1.shape[0]})")
    print(f"  N dims correct   : {'YES' if cache_ok else 'NO — cache invalidation bug!'}")

    # Release large forward outputs before loss section (avoids OOM on a second full forward).
    out0_has_heads = "heads" in out0
    del out0, out1, pts1, feat1
    model.backbone._cached_voxelization = None
    _maybe_cuda_empty_cache(device)

    # ── 8. Loss computation ──────────────────────────────────────
    sep("8. LOSS (multi-gran criterion)")
    base_criterion = MaskSetCriterion(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        score_weight=loss_cfg.get("score_weight", 0.5),
    )
    criterion = MultiGranCriterion(
        criterion=base_criterion,
        granularity_weights=loss_cfg.get("granularity_weights", None),
        aux_weight=loss_cfg.get("aux_weight", 0.0),
    )

    model.backbone._cached_voxelization = None
    model.zero_grad()
    pred = model(pts0, feat0)
    loss_result = criterion(pred, targets_s0)

    loss_val = loss_result["loss_total"].item()
    finite = loss_result["loss_total"].isfinite().item()
    print(f"  loss_total       : {loss_val:.4f}")
    print(f"  loss is finite   : {finite}")
    for g in granularities:
        ld = loss_result["heads"][g]
        print(f"  [{g}] loss={ld['loss_total'].item():.4f}  "
              f"bce={ld['loss_mask_bce'].item():.4f}  "
              f"dice={ld['loss_mask_dice'].item():.4f}  "
              f"score={ld['loss_score'].item():.4f}  "
              f"matches={ld['num_matches']}/{targets_s0[g].num_instances}")
    has_aux = "loss_aux" in loss_result
    print(f"  aux loss present : {has_aux}" +
          (f" ({loss_result['loss_aux'].item():.4f})" if has_aux else ""))

    # ── 9. Backward + grad clipping ─────────────────────────────
    sep("9. BACKWARD + GRADIENT CLIPPING")
    t0 = time.time()
    loss_result["loss_total"].backward()
    bwd_ms = (time.time() - t0) * 1000
    print(f"  backward time    : {bwd_ms:.0f} ms")

    bb_grad = sum(1 for p in model.backbone.parameters() if p.grad is not None)
    bb_total = sum(1 for _ in model.backbone.parameters())
    dc_grad = sum(1 for p in model.decoder.parameters() if p.grad is not None)
    dc_total = sum(1 for _ in model.decoder.parameters())
    print(f"  backbone grads   : {bb_grad} / {bb_total}")
    print(f"  decoder grads    : {dc_grad} / {dc_total}")

    clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    total_norm = clip_grad_norm_(model.parameters(), clip_norm).item()
    print(f"  grad norm before clip : {total_norm:.4f}")
    print(f"  clip threshold   : {clip_norm}")
    grad_finite = np.isfinite(total_norm)
    print(f"  grad norm finite : {grad_finite}")

    del pred, loss_result
    model.backbone._cached_voxelization = None
    model.zero_grad(set_to_none=True)
    _maybe_cuda_empty_cache(device)

    # ── 10. Optimizer + LR scheduler ─────────────────────────────
    sep("10. OPTIMIZER + LR SCHEDULER")
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    lr = train_cfg.get("lr", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    max_epochs = train_cfg.get("max_epochs", 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    print(f"  optimizer        : AdamW (lr={lr}, wd={wd})")
    print(f"  param groups     : {len(optimizer.param_groups)} (uniform LR)")

    if warmup_epochs > 0 and max_epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(max_epochs, 1))

    lrs = []
    for e in range(max_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    print(f"  warmup epochs    : {warmup_epochs}")
    print(f"  max epochs       : {max_epochs}")
    print(f"  LR at epoch 1    : {lrs[0]:.2e}")
    print(f"  LR at epoch {warmup_epochs}    : {lrs[min(warmup_epochs, len(lrs)-1)]:.2e}")
    print(f"  LR at epoch {max_epochs//2}   : {lrs[min(max_epochs//2, len(lrs)-1)]:.2e}")
    print(f"  LR at epoch {max_epochs}   : {lrs[-1]:.2e}")
    warmup_ok = lrs[0] < lrs[min(warmup_epochs - 1, len(lrs) - 1)]
    print(f"  warmup increases LR: {'YES' if warmup_ok else 'NO'}")

    # ── 11. Eval dry run ─────────────────────────────────────────
    sep("11. EVALUATION DRY RUN (1 val scene)")
    from student.engine.multi_scene_evaluator import evaluate_multi_scene

    val_1 = MultiSceneDataset(
        val_dirs[:1], granularities,
        use_colors=data_cfg.get("use_colors", True),
        append_xyz=data_cfg.get("append_xyz_to_features", False),
        max_points=max_pts,
    )
    t0 = time.time()
    eval_result = evaluate_multi_scene(
        model=model, dataset=val_1, criterion=criterion,
        device=device, granularities=granularities,
        score_threshold=0.3, mask_threshold=0.5,
        min_points=30, eval_benchmark="scannet200",
        min_instance_points=min_inst,
    )
    eval_ms = (time.time() - t0) * 1000
    model.train()

    agg = eval_result["aggregate"]
    print(f"  eval time        : {eval_ms:.0f} ms")
    print(f"  loss_mean        : {agg['loss_mean']:.4f}")
    print(f"  pseudo_AP50_mean : {agg['pseudo_AP50_mean']:.4f}")
    print(f"  matched_iou_mean : {agg['matched_mean_iou_mean']:.4f}")
    eval_ok = "per_scene" in eval_result and "aggregate" in eval_result
    print(f"  result structure : {'OK' if eval_ok else 'BROKEN'}")

    # ── Summary ──────────────────────────────────────────────────
    sep("SUMMARY")
    checks = [
        ("train scene list loads", len(train_dirs) > 0),
        ("val scene list loads", len(val_dirs) > 0),
        ("no train/val overlap", len(overlap) == 0),
        ("sample dict has expected keys", keys_ok),
        ("all samples structurally valid", True),
        ("targets built for all granularities", targets_ok),
        ("DataLoader collate works", loader_ok),
        ("model forward produces correct output", out0_has_heads),
        ("cache invalidation: N dims correct per scene", cache_ok),
        ("loss is finite", finite),
        ("aux loss present", has_aux),
        ("backward flows through backbone", bb_grad > 0),
        ("backward flows through decoder", dc_grad > 0),
        ("grad norm is finite", grad_finite),
        ("LR warmup increases over epochs", warmup_ok),
        ("evaluate_multi_scene returns valid result", eval_ok),
    ]
    all_ok = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print(f"\n{'ALL OK — ready for multi-scene training.' if all_ok else 'SOMETHING FAILED.'}")


if __name__ == "__main__":
    main()
