# Evaluation Layer

The evaluation layer provides dataset-owned extension points without forcing the core pipeline to know benchmark details.

## Hook Interface

`eval/base.py` defines `DatasetEvaluationHooks`.

Datasets can implement hooks that provide:

- `evaluate_scene(...)`
- `flatten_scene_summary(...)`
- `expected_output_paths(...)`
- `verify_summary(...)`
- `running_summary_payload(...)`
- `render_run_summary(...)`
- `scene_metric_fieldnames(...)`

## Architectural Intent

Shared code should ask:

- how should this dataset evaluate a scene?
- how should evaluation results be flattened for reporting?
- which dataset-specific files prove completion?
- how should end-of-run metrics be summarized?

Shared code should not ask:

- which benchmark IDs belong to ScanNet20 or ScanNet200?
- what are the dataset's oracle output file names?
- how are dataset-specific metrics named in reports?

## Current Implementation

ScanNet is the reference implementation of this pattern in:

- `datasets/scannet/evaluation.py`

If another dataset later needs its own evaluation protocol, it should follow that same shape instead of adding conditionals to `core/`, `common/`, or generic reporter code.
