# Training Pack Specification

## Purpose

The CHORUS `training_pack/` directory is a versioned data contract between the producer project (`chorus/`) and any downstream consumer project, including `student/`.

CHORUS is responsible for generating the pack. Consumers are responsible for reading only the documented contract, not CHORUS-internal implementation details.

## Pack Location

For each processed scene, CHORUS writes:

```text
<scene_dir>/training_pack/
```

Older scenes may still contain `litept_pack/` as a legacy location, but new exports should use `training_pack/`.

## Required Files

The following files are required for a valid training pack:

- `points.npy`
- `labels_g*.npy`
- `valid_points.npy`
- `seen_points.npy`
- `supervision_mask.npy`
- `scene_meta.json`

## File Semantics

### `points.npy`

Array of 3D points used as the supervision reference geometry for the scene.

- Shape: typically `[N, 3]`
- Units: meters
- Coordinate frame: scene-level geometry coordinates from the dataset adapter
- Point source: described in `scene_meta.json`

### `labels_g*.npy`

Per-point instance labels for one granularity.

- Filename pattern: `labels_g{granularity}.npy`
- Shape: typically `[N]`
- `-1` means ignore or unlabeled
- Non-negative integers are instance ids within that single label file
- Instance ids are not guaranteed to match across different granularities

### `valid_points.npy`

Binary per-point mask indicating whether a point received a usable non-negative label in at least one exported granularity.

- Shape: `[N]`
- Values: `0` or `1`
- Definition: `1` iff the point has `label >= 0` in at least one exported `labels_g*.npy`

### `seen_points.npy`

Binary per-point mask indicating whether a point was observed in at least one processed frame.

- Shape: `[N]`
- Values: `0` or `1`
- Definition: `1` iff the point was seen by the pipeline in at least one processed frame across the exported granularities

### `supervision_mask.npy`

Binary per-point mask identifying which points should be used for downstream supervision.

- Shape: `[N]`
- Values: `0` or `1`
- In pack version `1.0`, this is identical to `valid_points.npy`

### `scene_meta.json`

Machine-readable contract metadata for the pack.

Required fields:

- `pack_version`
- `label_convention`
- `supervision_mask_definition`
- `valid_points_definition`
- `seen_points_definition`
- `coordinate_units`
- `coordinate_frame`
- `point_source`
- `optional_files_present`

Important additional fields currently written by CHORUS include:

- `pack_name`
- `dataset`
- `scene_id`
- `geometry_type`
- `geometry_source`
- `num_points`
- `num_frames_total`
- `num_frames_used`
- `frame_skip`
- `granularities`
- `label_files`
- `teacher_name`
- `projection_type`
- `embedding_type`
- `clustering_type`
- `cluster_stats`
- `scene_intrinsic_metrics`

## Optional Files

Optional files must be declared in `scene_meta.json` under `optional_files_present`.

Current optional file:

- `colors.npy`: present only when the dataset adapter provides per-point colors

Consumers should check `scene_meta.json` instead of assuming optional files exist.

## Versioning Rules

- The pack format is a public contract between subprojects.
- Any change to file semantics, label meaning, mask meaning, coordinate interpretation, or required/optional file behavior must update `pack_version`.
- Backward-incompatible changes must not silently reuse the same `pack_version`.

## Consumer Expectations

Consumers such as `student/` should rely on:

- documented filenames
- documented semantics
- `scene_meta.json`

Consumers should not rely on:

- CHORUS internal function names
- CHORUS pipeline stage names
- undocumented extra fields
- assumptions about optional files unless declared in `scene_meta.json`
