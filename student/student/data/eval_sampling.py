"""Helpers for validation/eval point sampling configuration."""

from __future__ import annotations

from typing import Any


def resolve_eval_sampling_config(
    data_cfg: dict[str, Any],
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve eval-specific sampling settings.

    Eval defaults to full-scene input.  Explicit ``eval.*`` keys take precedence
    over legacy ``data.val_*`` keys, which take precedence over the full-scene
    default.
    """
    eval_cfg = eval_cfg or {}
    return {
        "max_points": eval_cfg.get("max_points", data_cfg.get("val_max_points", None)),
        "subsampling_mode": eval_cfg.get(
            "subsampling_mode",
            data_cfg.get("val_subsampling_mode", "none"),
        ),
        "sphere_point_max": eval_cfg.get(
            "sphere_point_max",
            data_cfg.get("val_sphere_point_max", None),
        ),
    }


def is_full_scene_sampling(sampling: dict[str, Any]) -> bool:
    """Return True when a dataset configured with *sampling* cannot crop."""
    return (
        str(sampling.get("subsampling_mode", "none")) == "none"
        and sampling.get("max_points") is None
        and sampling.get("sphere_point_max") is None
    )
