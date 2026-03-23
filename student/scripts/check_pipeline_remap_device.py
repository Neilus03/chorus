#!/usr/bin/env python3
"""
Wrapper to make `--device cuda:K` behave consistently by remapping GPUs.

Why: in some environments, running directly on `cuda:1` (or other non-zero
indices) can fail due to stale GPU state / driver quirks / kernel edge-cases.
Remapping via `CUDA_VISIBLE_DEVICES=K` makes the target GPU appear as
`cuda:0` inside the child process, which matches the most-common code paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_cuda_index(device: str) -> int | None:
    device = device.strip()
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        tail = device.split(":", 1)[1]
        try:
            return int(tail)
        except ValueError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run check_pipeline.py with optional CUDA remapping.",
        add_help=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda:0 or cuda:1 (forwarded to check_pipeline.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="check_pipeline.py --config value",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        default=None,
        help="check_pipeline.py --scene-dir value",
    )

    args, unknown = parser.parse_known_args()

    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve().parent / "check_pipeline.py"),
    ]

    if args.config is not None:
        cmd += ["--config", args.config]
    if args.scene_dir is not None:
        cmd += ["--scene-dir", args.scene_dir]
    if args.device is not None:
        cmd += ["--device", args.device]
    cmd += unknown

    env = os.environ.copy()

    if args.device is not None:
        idx = _parse_cuda_index(args.device)
        if idx is not None and idx != 0 and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            cmd = ["cuda:0" if c == args.device else c for c in cmd]

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
