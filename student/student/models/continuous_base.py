"""Shared helpers for continuous granularity decoders."""

from __future__ import annotations

from typing import Any


class ContinuousDecoderMixin:
    """Marker mixin for decoders that consume a scalar ``target_g`` prompt."""


def is_continuous_decoder(decoder: Any) -> bool:
    """Return whether *decoder* follows the continuous granularity contract."""
    return isinstance(decoder, ContinuousDecoderMixin)
