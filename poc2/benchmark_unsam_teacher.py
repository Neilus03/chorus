import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# --- BENCH CONFIG ---
SCENE_DIR = "scene0000_00"
DEVICE = "cuda:1"
GRANULARITY = float(os.environ.get("BENCH_GRANULARITY", "0.8"))
FRAME_SKIP = int(os.environ.get("BENCH_FRAME_SKIP", "10"))
NUM_BENCH_FRAMES = int(os.environ.get("BENCH_NUM_FRAMES", "50"))
WARMUP_FRAMES = int(os.environ.get("BENCH_WARMUP_FRAMES", "3"))
SCANNET_NUM_SCENES = int(os.environ.get("SCANNET_NUM_SCENES", "1513"))

UNSAM_ROOT = Path(__file__).resolve().parent / "UnSAMv2"
SAM2_PYTHON_ROOT = UNSAM_ROOT / "sam2"
DEFAULT_SCRATCH_CKPT = Path(
    f"/scratch2/{os.environ.get('USER', 'nedela')}/chorus_poc2/checkpoints/unsamv2/unsamv2_plus_ckpt.pt"
)
DEFAULT_LOCAL_CKPT = SAM2_PYTHON_ROOT / "checkpoints" / "unsamv2_plus_ckpt.pt"
CHECKPOINT_PATH = Path(
    os.environ.get(
        "UNSAMV2_CKPT",
        str(DEFAULT_SCRATCH_CKPT if DEFAULT_SCRATCH_CKPT.exists() else DEFAULT_LOCAL_CKPT),
    )
)
MODEL_CFG = os.environ.get("UNSAMV2_CFG", "configs/unsamv2_small.yaml")

if str(SAM2_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_PYTHON_ROOT))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402


def build_generator(model):
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=64,
        points_per_batch=128,
        mask_threshold=-1,
        pred_iou_thresh=0.77,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
        output_mode="binary_mask",
    )


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    if m or h or d:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    scene_path = Path(SCENE_DIR)
    color_dir = scene_path / "color"
    all_frames = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    sampled_frames = all_frames[::FRAME_SKIP]
    bench_frames = sampled_frames[:NUM_BENCH_FRAMES]

    if not bench_frames:
        raise RuntimeError("No frames selected for benchmark.")

    print(f"Loading UnSAMv2 on {DEVICE}...")
    model = build_sam2(MODEL_CFG, str(CHECKPOINT_PATH), device=DEVICE, apply_postprocessing=True)
    generator = build_generator(model)

    print(
        f"Benchmarking {len(bench_frames)} frames "
        f"(warmup={WARMUP_FRAMES}, skip={FRAME_SKIP}, granularity={GRANULARITY})"
    )

    timings = []
    saved_masks = []

    for i, frame_name in enumerate(bench_frames):
        img_path = color_dir / frame_name
        image = np.array(Image.open(img_path).convert("RGB"))

        t0 = time.perf_counter()
        masks_data = generator.generate(image, gra=GRANULARITY)
        dt = time.perf_counter() - t0

        # Use same painting rule as teacher to estimate effective mask count.
        frame_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        local_mask_id = 1
        masks_sorted = sorted(masks_data, key=lambda m: m.get("area", 0), reverse=True)
        for mask_dict in masks_sorted:
            bool_mask = mask_dict.get("segmentation")
            if bool_mask is None:
                continue
            fill_region = bool_mask & (frame_mask == 0)
            if np.any(fill_region):
                frame_mask[fill_region] = local_mask_id
                local_mask_id += 1

        effective_masks = local_mask_id - 1
        saved_masks.append(effective_masks)

        phase = "warmup" if i < WARMUP_FRAMES else "bench"
        print(f"[{phase}] {frame_name}: {dt:.3f}s, effective_masks={effective_masks}")

        if i >= WARMUP_FRAMES:
            timings.append(dt)

    if not timings:
        raise RuntimeError("No benchmark timings collected. Reduce BENCH_WARMUP_FRAMES.")

    per_frame_mean = float(np.mean(timings))
    per_frame_median = float(np.median(timings))
    per_frame_p95 = float(np.percentile(timings, 95))

    scene_total_frames = len(all_frames)
    scene_sampled_frames = len(sampled_frames)

    total_scannet_frames = scene_sampled_frames * SCANNET_NUM_SCENES
    proj_seconds = total_scannet_frames * per_frame_mean

    print("\n--- Benchmark Summary ---")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Model cfg: {MODEL_CFG}")
    print(f"Scene frames total: {scene_total_frames}")
    print(f"Scene frames sampled (@skip={FRAME_SKIP}): {scene_sampled_frames}")
    print(f"Measured frames (post-warmup): {len(timings)}")
    print(f"sec/frame mean   : {per_frame_mean:.3f}")
    print(f"sec/frame median : {per_frame_median:.3f}")
    print(f"sec/frame p95    : {per_frame_p95:.3f}")
    print(f"avg effective masks/frame: {float(np.mean(saved_masks)):.2f}")

    print("\n--- ScanNet Projection ---")
    print(f"Assumed #scenes: {SCANNET_NUM_SCENES}")
    print(f"Total sampled frames: {total_scannet_frames}")
    print(f"Projected wall-clock (1 GPU): {fmt_duration(proj_seconds)}")
    print(f"Projected GPU-days: {proj_seconds / 86400.0:.2f}")


if __name__ == "__main__":
    main()
