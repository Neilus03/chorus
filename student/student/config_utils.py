"""Shared YAML loading and model/dataset config helpers for student scripts."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    import os
    import yaml

    def _expand_env_in_str(s: str) -> str:
        """
        Expand env vars in strings.

        Supports:
          - $VAR / ${VAR} (via os.path.expandvars)
          - ${VAR:-default} (bash-style default if unset/empty)
        """
        # First, handle ${VAR:-default} (os.path.expandvars doesn't support :-).
        pattern = re.compile(r"\$\{([^}]+)\}")

        def repl(m: re.Match[str]) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                val = os.environ.get(var, "")
                return val if val else default
            return os.environ.get(expr, m.group(0))

        out = pattern.sub(repl, s)
        out = os.path.expandvars(out)
        out = os.path.expanduser(out)
        return out

    def _expand(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_expand(v) for v in obj)
        if isinstance(obj, str):
            return _expand_env_in_str(obj)
        return obj

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand(cfg)


def parse_granularities(data_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Extract dot-free granularity keys from ``data`` config.

    Supports:
        granularities: [0.2, 0.5, 0.8]    ->  ("g02", "g05", "g08")
        granularity: 0.8                   ->  ("g08",)
    """

    def _to_key(g: float | str) -> str:
        s = str(g)
        if s.startswith("g"):
            return s.replace(".", "")
        return f"g{s}".replace(".", "")

    if "granularities" in data_cfg:
        return tuple(_to_key(g) for g in data_cfg["granularities"])
    if "granularity" in data_cfg:
        return (_to_key(data_cfg["granularity"]),)
    raise KeyError("Config must specify data.granularities or data.granularity")


def resolve_num_queries(
    model_cfg: dict[str, Any],
    bb_cfg: dict[str, Any],
) -> tuple[int, dict[str, int] | None]:
    """Match ``run_student`` query resolution.

    Prefer ``model.num_queries`` / ``model.num_queries_by_granularity``; if absent,
    fall back to the same keys under ``model.backbone`` (common in YAML layouts).
    """
    nq = model_cfg.get("num_queries")
    if nq is None:
        nq = bb_cfg.get("num_queries", 128)
    nqb = model_cfg.get("num_queries_by_granularity")
    if nqb is None:
        nqb = bb_cfg.get("num_queries_by_granularity")
    return int(nq), nqb
