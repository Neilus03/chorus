# Core Pipeline

This directory owns the dataset-agnostic CHORUS pipeline.

## What Lives Here

- Teacher inference over RGB-D frames
- 2D-to-3D projection
- Feature reduction and clustering
- Scene-level intrinsic metrics
- Export of pseudo-label artifacts that do not depend on a dataset-specific benchmark

## What Does Not Live Here

- Dataset-specific ground-truth parsing
- Dataset-specific oracle evaluation rules
- Dataset-specific completion checks
- Dataset-specific reporting/summary formatting

Those responsibilities are delegated through `SceneAdapter.get_evaluation_hooks()`.

## Design Rule

`core/` should only need:

- a `SceneAdapter`
- a `TeacherModel`
- generic clustering/export configuration

If a new dataset requires special evaluation behavior, that logic should be implemented in the dataset package and exposed through evaluation hooks rather than imported directly into `core/`.
