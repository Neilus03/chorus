#!/usr/bin/env python3
"""Dump checkpoint parameter names and shapes without constructing the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _state_dict_from_checkpoint(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]
    raise TypeError("Could not find a tensor state_dict in the checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = _state_dict_from_checkpoint(ckpt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"checkpoint: {args.checkpoint}")
    if isinstance(ckpt, dict):
        meta = {k: v for k, v in ckpt.items() if k not in {"model", "model_state_dict", "state_dict", "optimizer"}}
        if meta:
            lines.append(f"metadata_keys: {sorted(meta.keys())}")
    lines.append("")
    lines.append(f"{'parameter':105s} {'shape':32s} {'numel':>14s} dtype")
    lines.append("-" * 165)

    total = 0
    for name, tensor in state.items():
        shape = "x".join(str(v) for v in tensor.shape) or "scalar"
        numel = tensor.numel()
        total += numel
        lines.append(f"{name:105s} {shape:32s} {numel:14,d} {str(tensor.dtype)}")
    lines.append("-" * 165)
    lines.append(f"{'TOTAL':105s} {'':32s} {total:14,d}")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"parameters: {len(state):,}")
    print(f"total numel: {total:,}")


if __name__ == "__main__":
    main()
