import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# --- CONFIG ---
SCENE_DIR = "scene0000_00"
FRAME_SKIP = 10
DEVICE = "cuda:1"
GRANULARITY = 0.8 # 0.8=whole objects, 0.2=parts
DEBUG_FIRST_N_FRAMES = 10

# UnSAMv2 local checkout root
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

OUTPUT_DIR = os.path.join(SCENE_DIR, f"unsam_masks_g{GRANULARITY}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not SAM2_PYTHON_ROOT.exists():
    raise FileNotFoundError(
        f"Expected UnSAMv2 checkout at: {SAM2_PYTHON_ROOT}. "
        "Clone UnSAMv2 under poc2 first."
    )

if str(SAM2_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_PYTHON_ROOT))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402


def _build_mask_generator(model):
    # Matches the project's whole-image notebook settings.
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


def _build_relaxed_mask_generator(model):
    # Fallback for difficult frames where strict filtering returns empty output.
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        points_per_batch=128,
        mask_threshold=-1,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
        output_mode="binary_mask",
    )


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"UnSAMv2 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Download checkpoint from UnSAMv2 model zoo and set UNSAMV2_CKPT if needed."
        )

    print(f"Loading UnSAMv2 onto {DEVICE}...")
    sam2_model = build_sam2(
        MODEL_CFG,
        str(CHECKPOINT_PATH),
        device=DEVICE,
        apply_postprocessing=True,
    )
    mask_generator = _build_mask_generator(sam2_model)
    relaxed_mask_generator = _build_relaxed_mask_generator(sam2_model)

    color_dir = os.path.join(SCENE_DIR, "color")
    target_frames = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )[::FRAME_SKIP]

    print(f"Processing {len(target_frames)} frames at granularity={GRANULARITY}...")
    total_saved_masks = 0

    for frame_i, frame_name in enumerate(target_frames):
        img_path = os.path.join(color_dir, frame_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        masks_data = mask_generator.generate(image, gra=GRANULARITY)
        used_fallback = False
        if len(masks_data) == 0:
            masks_data = relaxed_mask_generator.generate(image, gra=GRANULARITY)
            used_fallback = True

        frame_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        local_mask_id = 1

        # Paint larger masks first, then fill remaining holes by smaller masks.
        masks_sorted = sorted(masks_data, key=lambda m: m.get("area", 0), reverse=True)
        for mask_dict in masks_sorted:
            bool_mask = mask_dict.get("segmentation")
            if bool_mask is None:
                continue
            fill_region = bool_mask & (frame_mask == 0)
            if np.any(fill_region):
                frame_mask[fill_region] = local_mask_id
                local_mask_id += 1

        saved_count = local_mask_id - 1
        total_saved_masks += saved_count

        save_name = frame_name.replace(".jpg", ".npy")
        np.save(os.path.join(OUTPUT_DIR, save_name), frame_mask)

        debug_suffix = " [fallback]" if used_fallback else ""
        print(f"Saved {saved_count} masks for {frame_name}{debug_suffix}")

        if frame_i < DEBUG_FIRST_N_FRAMES:
            ious = [m.get("predicted_iou", 0.0) for m in masks_data]
            if ious:
                print(
                    f"  debug: raw_masks={len(masks_data)}, "
                    f"iou[min/mean/max]={min(ious):.3f}/{np.mean(ious):.3f}/{max(ious):.3f}"
                )
            else:
                print("  debug: raw_masks=0 (even after fallback)")

    print(f"Teacher inference complete. Total saved masks: {total_saved_masks}")


if __name__ == "__main__":
    main()
