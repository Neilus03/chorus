from __future__ import annotations

DEFAULT_STRUCTURED3D_EVAL_BENCHMARK = "structured3d_full"


def normalize_structured3d_eval_benchmark(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return DEFAULT_STRUCTURED3D_EVAL_BENCHMARK
    return str(value).strip().lower()
