# Student

Training and evaluation code for the **student** model: multi-granularity instance masks on 3D scans, using a LitePT backbone and a learned query decoder. This folder holds the implementation used by Chorus.

## Layout

| Path | Role |
|------|------|
| `student/` | Python package (`models`, `data`, `engine`, `losses`, `metrics`) |
| `configs/` | YAML experiments (paths, model, training) |
| `configs/splits/` | Train/val scene lists (`.txt`) |
| `scripts/` | CLI entry points |

Long-form design and training notes live in `architecture_plan.md` and `training_plan.md` in this directory.

## Running

Run commands **from this directory** (`student/`) so imports resolve correctly.

**Single-scene (overfit / debugging)**

```bash
cd student
python scripts/run_student.py --config configs/overfit_one_scene.yaml
python scripts/run_student.py --config configs/overfit_one_scene.yaml --max-steps 50 --no-wandb
```

**Multi-scene training**

```bash
cd student
python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml
python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml --max-epochs 3 --no-wandb
```

Optional overrides use `key=value` syntax (dotted keys), for example:

```bash
python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml train.lr=3e-4
```

## Configuration

Edit the YAML files for your machine: `data.scans_root`, backbone `litept_root`, and `experiment.output_root` are absolute paths in the checked-in examples. Scene splits under `configs/splits/` list ScanNet-style scene IDs (one per line).

## Optional checks

Sanity scripts under `scripts/` (e.g. `check_multi_scene_pipeline.py`, `check_pipeline.py`) validate data and model wiring; see each file’s docstring for usage.

## Dependencies

Use the repository’s Python environment and requirements (see the repo root `requirements.txt` or your project venv). Weights & Biases is optional; pass `--no-wandb` if it is not configured.
