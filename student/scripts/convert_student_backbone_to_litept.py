#!/usr/bin/env python3
"""Convert a student checkpoint's LitePT backbone to upstream LitePT format.

The student model stores the upstream LitePT module under
``backbone.model.*`` (or ``model.backbone.model.*`` when wrapped for prompt
fine-tuning). Upstream LitePT recipes expect checkpoint keys like
``backbone.*`` inside a top-level ``state_dict``.

This script writes a weight-only checkpoint suitable for:

    sh scripts/train.sh -d scannet -c insseg-litept-small-v1m2 \
      -n insseg-litept-small-v1m2-student-init \
      -w /path/to/converted.pth

When a template checkpoint is provided, tensor shapes are checked against it.
The common student-vs-upstream input stem mismatch is handled explicitly:
student ScanNet runs may use RGB+normal+XYZ (9 channels), while upstream LitePT
uses RGB+normal (6 channels). In that exact case, the first 6 input channels
are copied, matching the documented student feature order.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(payload, dict) and all(
        hasattr(v, "shape") for v in payload.values()
    ):
        return payload
    raise TypeError("Could not find a state_dict/model_state_dict in checkpoint")


def _strip_module(key: str) -> str:
    return key.removeprefix("module.")


def _map_student_backbone_key(key: str) -> str | None:
    key = _strip_module(key)
    if key.startswith("model."):
        key = key.removeprefix("model.")
    if key.startswith("backbone.model."):
        return "backbone." + key.removeprefix("backbone.model.")
    return None


def _template_shapes(template_sd: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in template_sd.items():
        if hasattr(value, "shape"):
            shapes[_strip_module(key)] = tuple(value.shape)
    return shapes


def _adapt_tensor(
    key: str,
    tensor: torch.Tensor,
    target_shape: tuple[int, ...] | None,
) -> tuple[torch.Tensor | None, str]:
    if target_shape is None:
        return tensor, "copied_without_template_shape"

    source_shape = tuple(tensor.shape)
    if source_shape == target_shape:
        return tensor, "copied"

    # Student feature order is RGB, normal, optional XYZ. Upstream LitePT
    # ScanNet recipes use RGB+normal, so a [*, 9, ...] -> [*, 6, ...] mismatch
    # can be adapted by taking the first 6 channels.
    if (
        len(source_shape) >= 2
        and len(target_shape) == len(source_shape)
        and source_shape[0] == target_shape[0]
        and source_shape[1] == 9
        and target_shape[1] == 6
        and source_shape[2:] == target_shape[2:]
    ):
        return tensor[:, :6, ...].contiguous(), "cropped_rgb_normal_from_9ch"

    if (
        len(source_shape) >= 2
        and len(target_shape) == len(source_shape)
        and source_shape[:-1] == target_shape[:-1]
        and source_shape[-1] == 9
        and target_shape[-1] == 6
    ):
        return tensor[..., :6].contiguous(), "cropped_rgb_normal_from_9ch"

    return None, f"shape_mismatch_source={source_shape}_target={target_shape}"


def convert(
    student_ckpt: Path,
    output: Path,
    *,
    template_ckpt: Path | None = None,
    output_module_prefix: bool = False,
) -> dict[str, Any]:
    student_payload = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    student_sd = _state_dict(student_payload)

    template_shapes: dict[str, tuple[int, ...]] = {}
    if template_ckpt is not None:
        template_payload = torch.load(template_ckpt, map_location="cpu", weights_only=False)
        template_shapes = _template_shapes(_state_dict(template_payload))

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    stats: dict[str, Any] = {
        "student_ckpt": str(student_ckpt),
        "template_ckpt": str(template_ckpt) if template_ckpt is not None else None,
        "output": str(output),
        "input_keys": len(student_sd),
        "mapped_backbone_keys": 0,
        "written_keys": 0,
        "cropped_9ch_to_6ch": 0,
        "skipped": [],
    }

    for source_key in sorted(student_sd.keys()):
        mapped_key = _map_student_backbone_key(source_key)
        if mapped_key is None:
            continue
        stats["mapped_backbone_keys"] += 1

        tensor = student_sd[source_key]
        if not hasattr(tensor, "shape"):
            stats["skipped"].append({
                "source_key": source_key,
                "target_key": mapped_key,
                "reason": "non_tensor",
            })
            continue

        adapted, reason = _adapt_tensor(
            mapped_key,
            tensor.detach().cpu(),
            template_shapes.get(mapped_key) if template_shapes else None,
        )
        if adapted is None:
            stats["skipped"].append({
                "source_key": source_key,
                "target_key": mapped_key,
                "reason": reason,
            })
            continue

        if reason == "cropped_rgb_normal_from_9ch":
            stats["cropped_9ch_to_6ch"] += 1

        out_key = f"module.{mapped_key}" if output_module_prefix else mapped_key
        converted[out_key] = adapted

    stats["written_keys"] = len(converted)
    payload = {
        "state_dict": converted,
        "epoch": 0,
        "best_metric_value": -1.0,
        "converter": stats,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    stats_path = output.with_suffix(output.suffix + ".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("student_ckpt", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--template",
        type=Path,
        default=Path("/cluster/work/igp_psr/nedela/LitePT/pretrained/model_best.pth"),
        help="Optional LitePT checkpoint used only for tensor shape checks.",
    )
    ap.add_argument(
        "--module-prefix",
        action="store_true",
        help="Write keys as module.backbone.*. LitePT's loader also accepts unprefixed keys.",
    )
    args = ap.parse_args()

    template = args.template if args.template and args.template.is_file() else None
    stats = convert(
        args.student_ckpt,
        args.output,
        template_ckpt=template,
        output_module_prefix=args.module_prefix,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
