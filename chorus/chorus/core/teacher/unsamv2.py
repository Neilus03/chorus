
from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from chorus.common.types import TeacherOutput
from chorus.core.teacher.base import TeacherModel
from chorus.datasets.base import SceneAdapter


def _resolve_unsam_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    default_root = repo_root / "UnSAMv2"
    return Path(
        os.path.expandvars(
            os.path.expanduser(
                os.environ.get("UNSAM_ROOT", str(default_root))
            )
        )
    ).resolve()


def _resolve_checkpoint_path(unsam_root: Path) -> Path:
    user = os.environ.get("USER", "nedela")

    default_scratch_ckpt = Path(
        f"/scratch2/{user}/chorus/checkpoints/unsamv2/unsamv2_plus_ckpt.pt"
    )
    legacy_poc_ckpt = Path(
        f"/scratch2/{user}/chorus_poc/checkpoints/unsamv2/unsamv2_plus_ckpt.pt"
    )
    legacy_poc2_ckpt = Path(
        f"/scratch2/{user}/chorus_poc2/checkpoints/unsamv2/unsamv2_plus_ckpt.pt"
    )
    local_ckpt = unsam_root / "sam2" / "checkpoints" / "unsamv2_plus_ckpt.pt"

    default_choice = (
        default_scratch_ckpt
        if default_scratch_ckpt.exists()
        else (
            legacy_poc_ckpt
            if legacy_poc_ckpt.exists()
            else (legacy_poc2_ckpt if legacy_poc2_ckpt.exists() else local_ckpt)
        )
    )

    return Path(
        os.path.expandvars(
            os.path.expanduser(
                os.environ.get("UNSAMV2_CKPT", str(default_choice))
            )
        )
    ).resolve()


def _build_mask_generator(model, sam2_cls):
    return sam2_cls(
        model=model,
        points_per_side=32,
        points_per_batch=256,
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


def _build_relaxed_mask_generator(model, sam2_cls):
    return sam2_cls(
        model=model,
        points_per_side=32,
        points_per_batch=256,
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


class UnSAMv2Teacher(TeacherModel):
    def __init__(
        self,
        device: str | None = None,
        model_cfg: str | None = None,
        debug_first_n_frames: int = 10,
        overwrite: bool = False,
    ):
        self.device = device or os.environ.get("CHORUS_DEVICE", "cuda:0")
        self.model_cfg = model_cfg or os.environ.get("UNSAMV2_CFG", "configs/unsamv2_small.yaml")
        self.debug_first_n_frames = int(debug_first_n_frames)
        self.overwrite = bool(overwrite)

        self.unsam_root = _resolve_unsam_root()
        self.checkpoint_path = _resolve_checkpoint_path(self.unsam_root)
        self.sam2_python_root = self.unsam_root / "sam2"

        self._loaded = False
        self._mask_generator = None
        self._relaxed_mask_generator = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.sam2_python_root.exists():
            raise FileNotFoundError(
                f"Expected UnSAMv2 checkout at {self.sam2_python_root}. "
                "Set UNSAM_ROOT or place UnSAMv2 in the repository root."
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"UnSAMv2 checkpoint not found: {self.checkpoint_path}\n"
                "Set UNSAMV2_CKPT or place the checkpoint in the expected location."
            )

        if str(self.sam2_python_root) not in sys.path:
            sys.path.insert(0, str(self.sam2_python_root))

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore

        print(f"Loading UnSAMv2 on device={self.device}")
        model = build_sam2(
            self.model_cfg,
            str(self.checkpoint_path),
            device=self.device,
            apply_postprocessing=True,
        )

        self._mask_generator = _build_mask_generator(model, SAM2AutomaticMaskGenerator)
        self._relaxed_mask_generator = _build_relaxed_mask_generator(model, SAM2AutomaticMaskGenerator)
        self._loaded = True

    def _autocast_context(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def run(
        self,
        adapter: SceneAdapter,
        granularity: float,
        frame_skip: int,
    ) -> TeacherOutput:
        self._ensure_loaded()

        output_dir = adapter.scene_root / f"unsam_masks_g{granularity}"
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = adapter.list_frames()[::frame_skip]
        print(
            f"Teacher UnSAMv2: dataset={adapter.dataset_name}, scene={adapter.scene_id}, "
            f"granularity={granularity}, frames={len(frames)}"
        )

        frame_mask_paths: list[Path] = []
        total_masks = 0

        for frame_idx, frame in enumerate(frames):
            mask_path = output_dir / f"{frame.frame_id}.npy"

            if mask_path.exists() and not self.overwrite:
                frame_mask = np.load(mask_path)
                saved_count = int(np.max(frame_mask)) if frame_mask.size > 0 else 0
                total_masks += saved_count
                frame_mask_paths.append(mask_path)
                print(f"Reusing existing masks for frame {frame.frame_id}: {saved_count}")
                continue

            image = adapter.load_rgb(frame)

            with torch.inference_mode(), self._autocast_context():
                masks_data = self._mask_generator.generate(image, gra=granularity)
                used_fallback = False
                if len(masks_data) == 0:
                    masks_data = self._relaxed_mask_generator.generate(image, gra=granularity)
                    used_fallback = True

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

            saved_count = local_mask_id - 1
            total_masks += saved_count
            np.save(mask_path, frame_mask)
            frame_mask_paths.append(mask_path)

            suffix = " [fallback]" if used_fallback else ""
            print(f"Saved {saved_count} masks for frame {frame.frame_id}{suffix}")

            if frame_idx < self.debug_first_n_frames:
                ious = [m.get("predicted_iou", 0.0) for m in masks_data]
                if ious:
                    print(
                        f"  debug: raw_masks={len(masks_data)}, "
                        f"iou[min/mean/max]={min(ious):.3f}/{np.mean(ious):.3f}/{max(ious):.3f}"
                    )
                else:
                    print("  debug: raw_masks=0 (even after fallback)")

        print(
            f"Teacher complete: scene={adapter.scene_id}, granularity={granularity}, total_saved_masks={total_masks}"
        )

        return TeacherOutput(
            granularity=granularity,
            frame_mask_paths=frame_mask_paths,
            total_masks=total_masks,
        )