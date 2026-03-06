# Dataset Integration

Datasets in CHORUS are integrated through `SceneAdapter`.

## Adapter Responsibilities

Each dataset adapter should provide:

- scene preparation
- frame enumeration
- RGB, depth, pose, and intrinsics loading
- 3D geometry loading
- dataset visibility configuration
- optional GT instance loading

## Evaluation Responsibilities

Dataset-specific evaluation should not leak into shared pipeline code.

Instead, adapters can override:

- `get_evaluation_hooks()`

The returned hooks object can own:

- scene evaluation
- summary flattening for reporting
- expected dataset-specific output files
- summary verification rules
- dataset-specific aggregate run summaries

## Why This Split Exists

The goal is to keep pseudo-label generation generic across datasets while still allowing each dataset to define its own evaluation protocol.

That means adding a new dataset should mostly involve:

1. implementing a new adapter
2. optionally implementing dataset-specific evaluation hooks
3. adding dataset-specific scripts/configs if desired
