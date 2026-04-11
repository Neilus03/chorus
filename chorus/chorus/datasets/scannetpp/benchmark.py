from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from chorus.datasets.scannetpp.download import resolve_scannetpp_dataset_root

SCANNETPP_EVAL_BENCHMARK_ALL = "all"
SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE = "top100_instance"

DEFAULT_SCANNETPP_EVAL_BENCHMARK = SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE

SUPPORTED_SCANNETPP_EVAL_BENCHMARKS = frozenset(
    {
        SCANNETPP_EVAL_BENCHMARK_ALL,
        SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE,
    }
)

IGNORE_INSTANCE_CLASSES = {"wall", "floor", "ceiling"}


def normalize_scannetpp_eval_benchmark(benchmark: str | None) -> str:
    value = benchmark or DEFAULT_SCANNETPP_EVAL_BENCHMARK
    normalized = str(value).strip().lower()
    aliases = {
        "default": SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE,
        "top100": SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE,
        "instance": SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE,
        "top100instance": SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE,
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_SCANNETPP_EVAL_BENCHMARKS:
        supported = ", ".join(sorted(SUPPORTED_SCANNETPP_EVAL_BENCHMARKS))
        raise ValueError(
            f"Unsupported ScanNet++ eval benchmark '{benchmark}'. "
            f"Expected one of: {supported}."
        )
    return normalized


def resolve_scannetpp_metadata_root(
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> Path:
    return resolve_scannetpp_dataset_root(
        dataset_root=dataset_root,
        scene_root=scene_root,
    ) / "metadata"


def _read_label_file(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@lru_cache(maxsize=None)
def _load_benchmark_classes_cached(
    metadata_root_str: str,
    eval_benchmark: str,
) -> tuple[str, ...]:
    metadata_root = Path(metadata_root_str)
    normalized = normalize_scannetpp_eval_benchmark(eval_benchmark)

    candidates: list[Path] = []
    if normalized == SCANNETPP_EVAL_BENCHMARK_TOP100_INSTANCE:
        candidates.append(metadata_root / "semantic_benchmark" / "top100_instance.txt")
    candidates.append(metadata_root / "instance_classes.txt")

    for candidate in candidates:
        if candidate.exists():
            labels = _read_label_file(candidate)
            if labels:
                return labels
    return ()


def load_scannetpp_benchmark_classes(
    eval_benchmark: str = DEFAULT_SCANNETPP_EVAL_BENCHMARK,
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> tuple[str, ...]:
    metadata_root = resolve_scannetpp_metadata_root(
        dataset_root=dataset_root,
        scene_root=scene_root,
    )
    return _load_benchmark_classes_cached(
        str(metadata_root),
        normalize_scannetpp_eval_benchmark(eval_benchmark),
    )


def _pick_mapping_key(fieldnames: list[str] | None, candidates: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


@lru_cache(maxsize=None)
def _load_instance_mapping_cached(metadata_root_str: str) -> dict[str, str | None]:
    metadata_root = Path(metadata_root_str)
    mapping_path = metadata_root / "semantic_benchmark" / "map_benchmark.csv"
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        class_key = _pick_mapping_key(fieldnames, ("class", "raw_class", "label"))
        target_key = _pick_mapping_key(
            fieldnames,
            ("instance_map_to", "instance_target", "instance_label"),
        )
        if class_key is None or target_key is None:
            return {}

        mapping: dict[str, str | None] = {}
        for row in reader:
            raw_label = str(row.get(class_key, "")).strip().lower()
            if not raw_label:
                continue

            mapped_label_raw = row.get(target_key)
            if mapped_label_raw is None:
                mapping[raw_label] = raw_label
                continue

            mapped_label = str(mapped_label_raw).strip()
            if not mapped_label or mapped_label.lower() == "nan":
                mapping[raw_label] = raw_label
            elif mapped_label.lower() == "none":
                mapping[raw_label] = None
            else:
                mapping[raw_label] = mapped_label.lower()

    return mapping


def load_scannetpp_instance_mapping(
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> dict[str, str | None]:
    metadata_root = resolve_scannetpp_metadata_root(
        dataset_root=dataset_root,
        scene_root=scene_root,
    )
    return dict(_load_instance_mapping_cached(str(metadata_root)))


def map_scannetpp_instance_label(
    label: str,
    eval_benchmark: str = DEFAULT_SCANNETPP_EVAL_BENCHMARK,
    dataset_root: str | Path | None = None,
    scene_root: str | Path | None = None,
) -> str | None:
    normalized_label = str(label).strip().lower()
    if not normalized_label or normalized_label in IGNORE_INSTANCE_CLASSES:
        return None

    normalized_benchmark = normalize_scannetpp_eval_benchmark(eval_benchmark)
    if normalized_benchmark == SCANNETPP_EVAL_BENCHMARK_ALL:
        return normalized_label

    mapping = load_scannetpp_instance_mapping(
        dataset_root=dataset_root,
        scene_root=scene_root,
    )
    mapped_label = mapping.get(normalized_label, normalized_label)
    if mapped_label is None:
        return None

    benchmark_classes = set(
        load_scannetpp_benchmark_classes(
            eval_benchmark=normalized_benchmark,
            dataset_root=dataset_root,
            scene_root=scene_root,
        )
    )
    if benchmark_classes and mapped_label not in benchmark_classes:
        return None

    return mapped_label
