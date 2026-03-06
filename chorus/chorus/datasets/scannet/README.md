# ScanNet Integration

This package contains ScanNet-specific dataset logic for CHORUS.

## Responsibilities

- RGB-D extraction from `.sens`
- ScanNet geometry loading
- ScanNet GT instance loading from labels + aggregation files
- ScanNet benchmark filtering for oracle evaluation
- ScanNet-specific evaluation/reporting/verification hooks

## Benchmarks

CHORUS can evaluate ScanNet scenes under multiple class-agnostic instance protocols:

- `scannet20`
- `scannet200`

Both are still class-agnostic at scoring time. The difference is which GT instances are kept before evaluation:

- `scannet20`: keep only GT instances whose raw category maps into the official ScanNet20 benchmark classes
- `scannet200`: keep only GT instances whose raw category maps into the official ScanNet200 benchmark classes

The GT filtering lives in:

- `gt.py`
- `benchmark.py`

## Evaluation Hooks

`evaluation.py` owns ScanNet-specific behavior that used to leak into shared code:

- running oracle evaluation for one or more ScanNet benchmarks
- flattening oracle summaries into scene-level report fields
- declaring expected oracle output files
- validating that saved summaries match the requested ScanNet benchmarks
- rendering the final human-readable terminal summary for a run

## Output Convention

For backward compatibility:

- ScanNet20 keeps the legacy unsuffixed oracle files
- ScanNet200 writes suffixed files such as `_scannet200`

This allows one run to produce both protocols side by side without breaking existing ScanNet20 consumers.
