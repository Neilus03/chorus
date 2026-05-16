from student.metrics.official_instance_ap import (
    SCANNET_IGNORE_MODE,
    SCANNET_MIN_REGION_SIZE,
    STRICT_IGNORE_MODE,
    build_instance_ap_records,
    evaluate_official_and_oracle_ap,
    evaluate_official_instance_ap,
    get_iou_thresholds,
    merge_ap_record_sets,
)

_PSEUDO_EXPORTS = {"compute_pseudo_metrics", "format_pseudo_metrics"}

__all__ = [
    "SCANNET_IGNORE_MODE",
    "SCANNET_MIN_REGION_SIZE",
    "STRICT_IGNORE_MODE",
    "build_instance_ap_records",
    "compute_pseudo_metrics",
    "evaluate_official_and_oracle_ap",
    "evaluate_official_instance_ap",
    "format_pseudo_metrics",
    "get_iou_thresholds",
    "merge_ap_record_sets",
]


def __getattr__(name: str):
    if name in _PSEUDO_EXPORTS:
        import importlib

        pseudo_metrics = importlib.import_module("student.metrics.pseudo_metrics")
        value = getattr(pseudo_metrics, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
