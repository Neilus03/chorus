# Multi-Scene Training & Validation Plan

> **My goal** is to move from single-scene overfitting to training on **10 scenes** and
> evaluating on **3 unseen scenes**, measuring both pseudo-label reproduction and real
> ScanNet GT quality. This is a **small-scale validation experiment** — not a full
> ScanNet-scale training run. The purpose is to verify that my student model
> generalizes across scenes before investing in larger-scale training.
>
> **My guiding principle**: each phase should leave a fully working pipeline. I keep
> the existing `run_student.py` and `SingleSceneTrainer` untouched and working. New
> code lives in new files. I reuse every existing component I can (`MultiGranCriterion`,
> `evaluate_student_predictions_multi`, `build_student_model`, `build_instance_targets_multi`,
> `training_pack.py` loaders, etc.) and only add what is genuinely missing.

---

## Starting point (what I have now)

My student pipeline trains on **one scene at a time** with no data iteration:

```
MultiGranSceneDataset(scene_dir)       → one sample dict (all points in memory)
build_instance_targets_multi(sample)   → InstanceTargets per granularity
SingleSceneTrainer.train()             → forward same sample max_steps times
  ├─ loss: MultiGranCriterion (Hungarian matching + BCE/Dice + aux)
  ├─ pseudo metrics every eval_every steps (matched IoU, score gap)
  └─ full eval every full_eval_every steps (AP25/AP50 vs pseudo-GT and real GT)
```

`run_overfit_benchmark_scenes.py` can run this across N scenes, but it launches
**independent subprocesses** — no shared gradient, no generalization signal.

**What is missing for multi-scene training:**

| Gap | Why it matters |
|-----|---------------|
| No multi-scene `Dataset` | Each scene is a separate `MultiGranSceneDataset` with `__len__==1`; no iteration over scenes |
| No train/val split | No concept of held-out scenes; evaluation is only on the training scene |
| No epoch-based loop | `SingleSceneTrainer` runs a fixed `max_steps` on one sample; no dataloader |
| No cross-scene metric aggregation | `evaluate_student_predictions_multi` returns per-scene dicts; nothing averages across scenes |
| LitePT voxelization cache assumes one scene | The backbone caches voxelization for single-scene reuse; must be invalidated per scene |

---

## Key design decisions

### Why scene-level iteration (not point-cloud batching)?

3D instance segmentation methods (Mask3D [[1]](#references), SPFormer [[2]](#references),
OneFormer3D [[3]](#references)) train with **batch size 1** or very small batches at the
scene level. Each scene is a full point cloud (50k–250k points) processed as one sample.
The reasons:

1. **Variable point counts**: scenes range from ~40k to ~240k points. Padding or truncating
   to a fixed size discards geometry or wastes compute. Scene-level processing preserves the
   full spatial structure.
2. **Voxelization is scene-specific**: LitePT's `GridPooling`/`GridUnpooling` builds a scene-local
   voxel grid. Mixing multiple scenes into one voxel grid would corrupt spatial relationships.
3. **Hungarian matching is per-scene**: the loss matches predicted queries to GT instances
   within one scene. Cross-scene matching is meaningless.
4. **GPU memory**: one scene already occupies significant VRAM (backbone + decoder + matching).
   Multi-scene batching would require gradient accumulation anyway.

**My approach**: iterate over scenes one at a time within each epoch, accumulating gradients
if needed. This matches Mask3D, SPFormer, and MAFT training protocols.

### Why 10 train / 3 val (not more)?

This is a **validation experiment** to answer: "does my student generalize at all?" Before
scaling to hundreds of scenes, I need to confirm that:

1. Loss decreases on training scenes across epochs
2. Validation metrics do not collapse (the model does not just memorize)
3. The training loop is mechanically correct (data loading, gradient flow, checkpointing)
4. Per-step time and memory are acceptable

The 10-scene PoC list from `poc2/7_run_10_scenes.py` is a natural starting point — those
scenes already have training packs generated. I hold out 3 of them for validation. This
gives a 10/3 split where both sets are small enough to evaluate exhaustively every epoch.

### Why invalidate the voxelization cache?

`LitePTBackbone` caches voxelization results for single-scene overfitting (Phase 1, Step 1.3
in `architecture_plan.md`). When training on multiple scenes, the cache must be invalidated
every time the input scene changes. The simplest approach: **disable caching** when
`multi_scene=True` (or call `backbone.clear_cache()` before each new scene).

---

## Scene split definition

From the existing PoC 10-scene list (`poc2/7_run_10_scenes.py`):

```
scene0000_00, scene0140_00, scene0263_00, scene0340_01, scene0381_02,
scene0396_01, scene0399_01, scene0420_02, scene0654_00, scene0662_01
```

**Finalized split** (verified on disk — all 13 scenes have complete training packs under
`/scratch2/nedela/chorus_poc/scans` with `scene_meta.json`, `points.npy`, `colors.npy`,
`valid_points.npy`, `seen_points.npy`, `supervision_mask.npy`, and all three granularity
label files):

| Set | Scene | Points | Notes |
|-----|-------|--------|-------|
| **Train** | scene0062_00 | 51,610 | smallest train scene |
| **Train** | scene0068_00 | 69,549 | |
| **Train** | scene0063_00 | 76,966 | |
| **Train** | scene0000_00 | 81,369 | original PoC scene |
| **Train** | scene0042_00 | 96,573 | |
| **Train** | scene0049_00 | 118,972 | |
| **Train** | scene0060_00 | 135,544 | |
| **Train** | scene0159_00 | 168,576 | |
| **Train** | scene0064_00 | 230,672 | |
| **Train** | scene0140_00 | 372,941 | largest train scene |
| **Val** | scene0078_00 | 69,783 | small val scene |
| **Val** | scene0072_02 | 132,461 | medium val scene |
| **Val** | scene0024_01 | 225,484 | large val scene |

Training set: 10 scenes (52k–373k pts, mean ~140k). Selected from scenes with complete
training packs (all required `.npy` files present). Includes `scene0000_00` (original PoC)
and `scene0140_00` from the initial 10-scene list; the other 8 were replaced because the
original `litept_pack` scenes lacked `seen_points.npy`.

Validation set: 3 unseen scenes (70k–225k pts). Chosen to span the size distribution of
the training set — one small, one medium, one large — so validation metrics reflect
performance across scene scales. None overlap with the training set.

**Implementation**: scene lists are plain text files (one scene ID per line), stored under
`student/configs/splits/`. This matches the pattern in `chorus/scripts/make_scene_list.py`.

```
configs/splits/train_10.txt    → 10 scene IDs
configs/splits/val_3.txt       → 3 scene IDs
```

---

## Phase 1: Multi-Scene Dataset ✅

> **Files**: `student/data/multi_scene_dataset.py` (new), `student/data/__init__.py`
> **Risk**: low — additive; does not modify existing data classes
> **Outcome**: a PyTorch `Dataset` that iterates over multiple scenes, each yielding a
> complete sample dict compatible with the existing loss and evaluator.

### Step 1.1 — Verify available scenes and finalize split ✅

**Done.** Scanned `/scratch2/nedela/chorus_poc/scans` — 416 scenes have complete training
packs with all three granularity labels. Split files created:

- `configs/splits/train_10.txt` — 10 training scenes (51k–373k pts)
- `configs/splits/val_3.txt` — 3 validation scenes (72k–225k pts)

All 13 scenes verified to have `scene_meta.json`, `points.npy`, `supervision_mask.npy`,
and `labels_g{0.2,0.5,0.8}.npy`. Some use `training_pack/` layout, some use `litept_pack/`
— the existing `_resolve_pack_dir()` in `training_pack.py` handles both transparently.

### Step 1.2 — Implement `MultiSceneDataset` ✅

**File**: `student/data/multi_scene_dataset.py`

A PyTorch `Dataset` where `__len__` equals the number of scenes and `__getitem__(i)` loads
scene `i`:

```python
class MultiSceneDataset(Dataset):
    """Dataset that iterates over multiple scenes.

    Each item is a complete sample dict (same contract as
    MultiGranSceneDataset[0]) — points, features, labels_by_granularity,
    supervision_mask, scene_id, scene_dir, scene_meta.
    """

    def __init__(
        self,
        scene_dirs: list[Path],
        granularities: tuple[str, ...],
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
        preload: bool = True,
    ):
        ...
```

**Key design decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Preloading** | Load all scenes into memory at init time (`preload=True`) | With 10–13 scenes, total RAM is ~2–5 GB for points+labels+features. Avoids repeated disk I/O every epoch. Lazy loading is the fallback for larger datasets. |
| **Return format** | Same dict structure as `MultiGranSceneDataset[0]` | All downstream code (target builder, loss, evaluator) already consumes this format. Zero adapter code needed. |
| **No collation / no DataLoader batching** | `batch_size=1` with manual iteration | Variable point counts make collation non-trivial. Scene-level iteration is standard (see design decisions above). |
| **Shuffling** | Shuffle scene order each epoch via `DataLoader(shuffle=True)` | Prevents the model from memorizing a fixed scene order. Standard practice. |

**Why a new file instead of extending `single_scene_dataset.py`**: the existing file's
classes have `__len__ == 1` as a fundamental contract. Adding multi-scene logic there
would muddy the abstraction. A separate file with a clean class is clearer and lets me
keep the overfitting path completely untouched.

**Target building**: `MultiSceneDataset.__getitem__` returns the raw sample dict. Target
building (`build_instance_targets_multi`) happens **outside** the dataset, in the trainer,
because `InstanceTargets` contain tensors that should be on the correct device. This
matches the current pattern in `run_student.py`.

**Preload implementation**: at `__init__`, iterate over `scene_dirs`, call
`load_training_pack_scene_multi(scene_dir, granularities)` and `build_input_features()`
for each, and store the results. `__getitem__` constructs the tensor dict from the cached
numpy arrays. This reuses all existing loading and validation logic from `training_pack.py`.

### Step 1.3 — Add factory function `build_scene_list` ✅

**File**: `student/data/multi_scene_dataset.py`

```python
def build_scene_list(
    scene_list_file: Path,
    scans_root: Path,
) -> list[Path]:
    """Read a text file of scene IDs and resolve to full paths."""
```

Reads `configs/splits/train_10.txt`, prepends `scans_root`, validates each directory
exists and has a training pack. This centralizes path resolution and validation.

### Step 1.4 — Update `student/data/__init__.py` ✅

Export `MultiSceneDataset` and `build_scene_list` so they're importable from
`student.data`.

### Step 1.5 — Verify Phase 1 ✅

Write a quick smoke test (can be a `__main__` block in `multi_scene_dataset.py`):

```bash
cd student
python -m student.data.multi_scene_dataset \
    --scene-list configs/splits/train_10.txt \
    --scans-root /scratch2/nedela/chorus_poc/scans
```

**Checks**:
- `len(dataset) == 10`
- Each `dataset[i]` returns a dict with the expected keys
- `points` shapes vary across scenes (confirming different point counts)
- `labels_by_granularity` has keys `g02`, `g05`, `g08` for each scene
- `supervision_mask` has non-zero entries for each scene

---

## Phase 2: Multi-Scene Trainer ✅

> **Files**: `student/engine/multi_scene_trainer.py` (new)
> **Risk**: medium — core training logic; but I reuse `MultiGranCriterion` and
> `evaluate_student_predictions_multi` directly
> **Outcome**: epoch-based training loop that iterates over scenes, with periodic
> validation on held-out scenes.

### Step 2.1 — Design the training loop ✅

The `MultiSceneTrainer` replaces `SingleSceneTrainer` for multi-scene experiments. The
core loop structure:

```
for epoch in 1..max_epochs:
    shuffle train scenes
    for scene in train_scenes:
        targets = build_instance_targets_multi(scene)  # on device
        model.train()
        pred = model(scene.points, scene.features)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        # log per-scene train loss

    if epoch % eval_every_epochs == 0:
        run_validation(val_scenes)

    if epoch % save_every_epochs == 0:
        save_checkpoint()
```

**Why epoch-based, not step-based**: with 10 scenes, one epoch = 10 forward+backward
passes. Epoch-based counting is standard for dataset-scale training and makes it easy
to reason about "how many times has the model seen each scene." I still log step-level
metrics (where step = one scene forward/backward).

**Why no gradient accumulation in v1**: with batch_size=1 at the scene level (standard
for 3D instance seg), each backward pass produces a full gradient from one scene. This
is how Mask3D, SPFormer, and MAFT train. Gradient accumulation across scenes is a
possible future optimization but not needed for 10 scenes.

### Step 2.2 — Implement `MultiSceneTrainer` ✅

**File**: `student/engine/multi_scene_trainer.py`

```python
class MultiSceneTrainer:
    """Epoch-based trainer over multiple scenes with validation.

    Parameters
    ----------
    model:
        StudentInstanceSegModel.
    criterion:
        MultiGranCriterion instance.
    train_dataset:
        MultiSceneDataset for training scenes.
    val_dataset:
        MultiSceneDataset for validation scenes.
    ...
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: MultiGranCriterion,
        train_dataset: MultiSceneDataset,
        val_dataset: MultiSceneDataset,
        *,
        device: str = "cuda:0",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        max_epochs: int = 50,
        eval_every_epochs: int = 5,
        save_every_epochs: int = 10,
        output_dir: Path,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        min_points_per_proposal: int = 30,
        eval_benchmark: str = "scannet200",
        min_instance_points: int = 10,
    ):
        ...
```

**Key implementation details:**

| Aspect | Implementation | Rationale |
|--------|---------------|-----------|
| **Scene shuffling** | Use `torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x[0])` | `collate_fn=lambda x: x[0]` unwraps the DataLoader's list-of-one-item batching, yielding the raw sample dict. Shuffling randomizes scene order each epoch. |
| **Target building** | Build `InstanceTargets` per scene inside the training loop, on device | Targets contain `gt_masks [M, N]` which vary in both M and N per scene. Cannot be pre-stacked. Building on-device avoids a CPU→GPU transfer of the mask tensor. |
| **Backbone cache** | Call `model.backbone.clear_cache()` before each scene (or disable caching) | The voxelization cache in `LitePTBackbone` stores results for one specific point cloud. Processing a different scene with a stale cache would produce wrong results. |
| **Optimizer** | One `AdamW` optimizer, **uniform LR** for backbone and decoder | Both backbone and decoder are trained from scratch (no pretrained LitePT weights). There is no reason to use separate LR groups when all parameters are randomly initialized — the standard practice of lower backbone LR only applies to fine-tuning pretrained weights [[5]](#references). A single parameter group simplifies the optimizer and avoids an unnecessary hyperparameter. |
| **LR scheduler** | `CosineAnnealingLR` with warmup | Mask3D [[1]](#references) uses polynomial LR decay; SPFormer [[2]](#references) uses step decay. For this small experiment, cosine annealing with linear warmup (first 5 epochs) is a reasonable middle ground. Warmup stabilizes early training when all parameters are randomly initialized and the model sees diverse scenes for the first time. |
| **Gradient clipping** | Same `clip_grad_norm_` as `SingleSceneTrainer` | Transformer decoders are prone to gradient spikes, especially early in training from scratch with diverse scenes. |
| **wandb logging** | Per-step: scene_id, loss, grad_norm. Per-epoch: mean train loss across scenes. Per-val: per-scene + aggregated metrics. | Matches existing logging granularity but adds cross-scene aggregation. |

### Step 2.3 — Implement `_train_one_epoch` ✅

**Method on `MultiSceneTrainer`**

```python
def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
    """Train on all scenes once. Returns epoch-level metrics."""
```

For each scene in the (shuffled) train DataLoader:

1. Move `points`, `features` to device
2. Build `InstanceTargets` from `labels_by_granularity` + `supervision_mask`
3. Clear backbone voxelization cache
4. Forward pass → loss → backward → clip → step
5. Log per-scene loss, grad norm, timing

After all scenes: compute and log epoch-average loss.

**Memory management**: after processing each scene, the point cloud tensors and targets
from the previous scene are no longer referenced and will be garbage collected. For safety,
explicitly `del` large tensors and call `torch.cuda.empty_cache()` between scenes if
memory pressure is observed. This is only relevant if scene sizes vary dramatically
(e.g., 40k vs 240k points).

### Step 2.4 — Implement `_validate` ✅

**Method on `MultiSceneTrainer`**

```python
def _validate(self, epoch: int) -> dict[str, Any]:
    """Run full evaluation on all validation scenes. Returns aggregated metrics."""
```

For each scene in val_dataset:

1. Load scene, build targets, move to device
2. `model.eval()` + `torch.no_grad()`
3. Forward pass
4. Compute pseudo metrics via `compute_pseudo_metrics_multi` (reuse existing)
5. Compute full eval via `evaluate_student_predictions_multi` (reuse existing)
6. Collect per-scene results

After all val scenes: aggregate metrics across scenes.

**Aggregation strategy** (matches standard 3D instance seg evaluation):

| Metric | Aggregation | Justification |
|--------|-------------|---------------|
| AP25 / AP50 (pseudo-GT) | Mean across scenes | Standard in ScanNet eval — each scene contributes equally regardless of instance count [[4]](#references) |
| AP25 / AP50 (real GT) | Mean across scenes | Same |
| NMI / ARI | Mean across scenes | Clustering quality should be consistent |
| Matched mean IoU | Mean across scenes | Overall mask quality |
| Loss | Mean across scenes | Training diagnostic |

Per-scene results are also logged individually for debugging (identify which scenes
are hard/easy).

### Step 2.5 — Implement checkpointing ✅

Save full state for resumability:

```python
checkpoint = {
    "epoch": epoch,
    "global_step": self.global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_metric": self.best_val_metric,
    "config": self.config,
}
```

Track best checkpoint by validation pseudo-GT AP50 (mean across val scenes). Save both
`last.pt` and `best.pt`.

**Why pseudo-GT AP50 as the selection metric**: the student is trained to reproduce pseudo
labels. AP50 against pseudo-GT directly measures that objective. Real GT metrics are
informative but should not drive checkpoint selection, since the pseudo labels are the
training signal and we want to validate the student's ability to reproduce them before
assessing downstream quality.

### Step 2.6 — Implement `train()` main loop ✅

```python
def train(self) -> dict[str, Any]:
    for epoch in range(1, self.max_epochs + 1):
        epoch_metrics = self._train_one_epoch(epoch)
        self.scheduler.step()

        if epoch % self.eval_every_epochs == 0 or epoch == self.max_epochs:
            val_metrics = self._validate(epoch)
            # track best, save best checkpoint

        if epoch % self.save_every_epochs == 0:
            self._save_checkpoint("last")

    self._save_checkpoint("last")
    final_val = self._validate(self.current_epoch)
    return final_val
```

### Step 2.7 — Verify Phase 2

Run a minimal training session (not yet run — requires GPU):

```bash
python scripts/run_multi_scene.py \
    --config configs/multi_scene_10_3.yaml \
    --max-epochs 3 \
    --no-wandb
```

**Checks**:
- Training completes 3 epochs × 10 scenes = 30 forward/backward passes
- Loss decreases (or at least doesn't diverge) across steps
- Validation runs on 3 held-out scenes and produces AP/IoU numbers
- Checkpoint files are saved
- No CUDA OOM (if OOM, reduce `num_queries` or `decoder_hidden_dim`)
- Backbone voxelization cache does not corrupt results (compare a scene's loss when it
  appears first vs later in the epoch)

---

## Phase 3: Config and Entry Point ✅

> **Files**: `configs/multi_scene_10_3.yaml` (new), `configs/splits/train_10.txt` (new),
> `configs/splits/val_3.txt` (new), `scripts/run_multi_scene.py` (new)
> **Risk**: low — config and CLI plumbing
> **Outcome**: a single command trains on 10 scenes and evaluates on 3.

### Step 3.1 — Create scene split files ✅

**Files**: `configs/splits/train_10.txt`, `configs/splits/val_3.txt` (already created)

Plain text, one scene ID per line:

```
# configs/splits/train_10.txt
scene0000_00
scene0042_00
scene0049_00
scene0060_00
scene0062_00
scene0063_00
scene0064_00
scene0068_00
scene0140_00
scene0159_00
```

```
# configs/splits/val_3.txt
scene0024_01
scene0072_02
scene0078_00
```

### Step 3.2 — Create `configs/multi_scene_10_3.yaml` ✅

```yaml
experiment:
  name: multi_scene_10_3
  seed: 42
  output_root: /scratch2/nedela/student_runs

data:
  scans_root: /scratch2/nedela/chorus_poc/scans
  train_split: configs/splits/train_10.txt
  val_split: configs/splits/val_3.txt
  granularities: [0.2, 0.5, 0.8]
  use_colors: true
  append_xyz_to_features: false
  min_instance_points: 10
  preload: true

model:
  backbone:
    name: litept
    litept_root: /home/nedela/projects/LitePT
    litept_variant: litept_s_star
    in_channels: 3
    grid_size: 0.02
    multi_scale: true
    num_queries: 200
    num_queries_by_granularity:
      g02: 350
      g05: 200
      g08: 150
  decoder_hidden_dim: 256
  num_decoder_layers: 3
  num_decoder_heads: 8
  query_init: learned
  use_positional_guidance: true
  learned_query_ratio: 0.5

loss:
  bce_weight: 1.0
  dice_weight: 1.0
  score_weight: 0.5
  aux_weight: 0.2

train:
  device: cuda:0
  lr: 1.0e-4
  # No backbone_lr_scale — both backbone and decoder are trained from scratch
  # (no pretrained LitePT weights). Uniform LR for all parameters.
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0
  max_epochs: 50
  eval_every_epochs: 5
  save_every_epochs: 10
  log_every_steps: 1      # log every scene (10 scenes/epoch = 10 logs/epoch)
  warmup_epochs: 5

eval:
  scannet_benchmark: scannet200
  score_threshold: 0.3
  mask_threshold: 0.5
  min_points_per_proposal: 30
```

**Key config differences from `overfit_one_scene.yaml`:**

| Parameter | Overfit value | Multi-scene value | Why |
|-----------|--------------|-------------------|-----|
| `lr` | `5e-4` | `1e-4` | Lower LR for generalization; overfitting used aggressive LR to converge fast on one scene |
| `backbone_lr_scale` | `0.75` | **removed** | Both backbone and decoder are trained from scratch (no pretrained LitePT weights). Separate LR scaling is a fine-tuning technique — when everything is randomly initialized, all parameters should train at the same rate. |
| `max_steps` → `max_epochs` | 5000 steps | 50 epochs (=500 steps) | Epoch-based; 50 epochs × 10 scenes = 500 scene-level gradient updates, comparable to ~500 steps of single-scene training. This is intentionally modest for a validation experiment. |
| `warmup_epochs` | N/A | 5 | Linear LR warmup over first 5 epochs prevents early instability when all parameters are randomly initialized and the model sees diverse scenes for the first time |
| `data.scans_root` + splits | `data.scene_dir` (one path) | Root + split files | Multi-scene addressing |

**Why these specific hyperparameters**: Mask3D trains with LR=1e-4, weight decay=1e-4,
batch size 1, for 600 epochs on ScanNet (1201 train scenes) [[1]](#references). For my
10 scenes / 50 epochs, I scale down epochs proportionally while keeping the same LR and
decay. The warmup prevents the common failure mode where randomly initialized parameters
take huge early steps when first exposed to diverse scenes.

**Training from scratch**: my LitePT backbone has **no pretrained weights** — both the
encoder and decoder are randomly initialized. This means:
- No backbone LR scaling (fine-tuning technique, irrelevant here)
- Potentially slower convergence than if backbone features were pretrained
- All parameters contribute equally to the gradient — uniform LR is appropriate
- The warmup is even more important: with no pretrained features to anchor the
  backbone, early gradients can be noisy and large

### Step 3.3 — Create `scripts/run_multi_scene.py` ✅

**File**: `scripts/run_multi_scene.py`

Entry point analogous to `run_student.py`. Structure:

```python
def main():
    # 1. Parse args (--config, --device, --max-epochs, --no-wandb, overrides)
    # 2. Load config
    # 3. Build train/val scene lists from split files + scans_root
    # 4. Build MultiSceneDataset for train and val
    # 5. Build model (same build_student_model factory)
    # 6. Build criterion (same MultiGranCriterion)
    # 7. Setup wandb
    # 8. Create MultiSceneTrainer
    # 9. trainer.train()
    # 10. Save final metrics summary
```

**CLI arguments** (superset of `run_student.py`):

| Argument | Purpose |
|----------|---------|
| `--config` | YAML config path |
| `--device` | Override CUDA device |
| `--max-epochs` | Override max epochs |
| `--no-wandb` | Disable wandb |
| `--wandb-project` | wandb project name |
| `--print-model` | Log model architecture |
| `overrides` | Dotted key=value overrides (reuse `apply_cli_overrides`) |

### Step 3.4 — Verify Phase 3

Not yet run — requires GPU.

```bash
# Dry run: verify config loads and datasets build
python scripts/run_multi_scene.py \
    --config configs/multi_scene_10_3.yaml \
    --max-epochs 1 \
    --no-wandb

# Full validation experiment
python scripts/run_multi_scene.py \
    --config configs/multi_scene_10_3.yaml \
    --device cuda:0
```

**Checks**:
- Config resolves correctly (all paths valid, splits loaded)
- Both train and val datasets build without errors
- Model builds with same architecture as overfit experiments
- One epoch completes (10 train scenes + 3 val scenes evaluated)
- Output directory structure is created with checkpoints and logs

---

## Phase 4: Evaluation and Metrics Aggregation ✅

> **Files**: `student/engine/multi_scene_evaluator.py` (new)
> **Risk**: low — wraps existing evaluation code
> **Outcome**: aggregated train/val metrics across scenes, logged to wandb and JSON.

### Step 4.1 — Implement `evaluate_multi_scene` ✅

**File**: `student/engine/multi_scene_evaluator.py`

```python
def evaluate_multi_scene(
    model: nn.Module,
    dataset: MultiSceneDataset,
    criterion: MultiGranCriterion,
    *,
    device: str,
    granularities: tuple[str, ...],
    score_threshold: float,
    mask_threshold: float,
    min_points: int,
    eval_benchmark: str,
    min_instance_points: int,
) -> dict[str, Any]:
    """Evaluate model on all scenes in a dataset. Returns per-scene and aggregated metrics."""
```

For each scene in the dataset:

1. Load sample, build targets on device
2. Forward pass (eval mode, no_grad)
3. Compute pseudo metrics (`compute_pseudo_metrics_multi`)
4. Compute full eval (`evaluate_student_predictions_multi`)
5. Store per-scene results

Aggregate across scenes:

```python
result = {
    "per_scene": {scene_id: {...} for each scene},
    "aggregate": {
        "pseudo_AP25_mean": ...,    # mean across scenes
        "pseudo_AP50_mean": ...,
        "real_AP25_mean": ...,
        "real_AP50_mean": ...,
        "pseudo_NMI_mean": ...,
        "pseudo_ARI_mean": ...,
        "real_NMI_mean": ...,
        "real_ARI_mean": ...,
        "matched_mean_iou_mean": ...,  # per granularity
        "loss_mean": ...,
    },
}
```

**Why a separate file**: the existing `evaluator.py` handles single-scene evaluation
perfectly. The multi-scene evaluator wraps it with iteration and aggregation. Keeping it
separate avoids bloating the single-scene code path.

### Step 4.2 — wandb metric definitions ✅

In `run_multi_scene.py`, after `wandb.init`:

```python
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")
wandb.define_metric("train_scene/*", step_metric="global_step")
wandb.define_metric("epoch")
wandb.define_metric("global_step")
```

**Logged metrics:**

| Scope | Metrics | Frequency |
|-------|---------|-----------|
| Per scene-step | `train_scene/loss_{scene_id}`, `train_scene/grad_norm` | Every step (every scene) |
| Per epoch | `train/loss_mean`, `train/loss_min`, `train/loss_max` | Every epoch |
| Per val epoch | `val/pseudo_AP25_mean`, `val/pseudo_AP50_mean`, `val/real_AP25_mean`, `val/real_AP50_mean`, `val/matched_mean_iou_{g}_mean` | Every `eval_every_epochs` |
| Per val scene | `val/{scene_id}/pseudo_AP25`, `val/{scene_id}/real_AP25` | Every `eval_every_epochs` |

### Step 4.3 — JSON summary output ✅

At the end of training, save a comprehensive JSON summary:

```python
{
    "config": {...},
    "train_scenes": [...],
    "val_scenes": [...],
    "final_train_metrics": {...},     # last epoch's train metrics
    "final_val_metrics": {...},       # last epoch's val metrics (per-scene + aggregate)
    "best_val_metrics": {...},        # metrics at the best checkpoint
    "best_epoch": ...,
    "total_training_time_s": ...,
    "per_epoch_time_s": [...],
}
```

### Step 4.4 — Verify Phase 4

Not yet run — requires GPU. Run the full experiment (50 epochs) and verify:

1. **Training loss** decreases across epochs
2. **Validation pseudo-GT AP** is non-zero and ideally increases
3. **Real GT AP** is non-zero (even if lower than pseudo-GT AP)
4. **Per-scene variance** — some scenes will be harder than others; verify the model
   doesn't collapse to only predicting well on one scene
5. **wandb dashboard** shows clean curves for both train and val metrics
6. **Best checkpoint** corresponds to the epoch with highest val pseudo-GT AP50

---

## Phase 5: Validation and Sanity Checks

> **Files**: no new files — this phase is about running experiments and verifying results
> **Risk**: medium — debugging and hyperparameter adjustment
> **Outcome**: confidence that the multi-scene pipeline is correct and the student
> generalizes across scenes.

### Step 5.1 — Quick sanity check (overfit-all-scenes baseline)

**Purpose**: verify the training loop works by checking the model can overfit 10 scenes
when trained for many epochs (expected: yes, since I proved single-scene overfitting works).

```bash
python scripts/run_multi_scene.py \
    --config configs/multi_scene_10_3.yaml \
    train.max_epochs=200 \
    train.eval_every_epochs=10 \
    train.lr=5e-4 \
    --no-wandb
```

**Expected**: train loss → 0, train scene AP → high, val AP may be lower (generalization
gap is expected). If train loss does not decrease, there is a bug in the training loop.

### Step 5.2 — Generalization check

**Purpose**: does the model produce non-trivial predictions on unseen val scenes?

Run the full 50-epoch experiment:

```bash
python scripts/run_multi_scene.py \
    --config configs/multi_scene_10_3.yaml
```

**Thresholds for "success"** (this is a validation experiment, not SOTA):

| Metric | Where | Minimum for "working" |
|--------|-------|----------------------|
| Train pseudo AP50 (mean) | Train scenes | > 0.2 |
| Val pseudo AP50 (mean) | Val scenes | > 0.05 |
| Val real AP50 (mean) | Val scenes | > 0.01 |
| Val matched IoU (mean) | Val scenes | > 0.15 |

These are deliberately low thresholds. The point is not SOTA but proving the loop works
and the model transfers at all. If val metrics are literally zero, something is wrong
(data loading bug, eval bug, or the model has zero generalization).

### Step 5.3 — Compare with independent overfitting baseline

**Purpose**: is joint training better or worse than independent per-scene overfitting?

Use the existing `run_overfit_benchmark_scenes.py` to overfit each of the 3 val scenes
independently, then compare their metrics against the jointly-trained model's val metrics.

```bash
python scripts/run_overfit_benchmark_scenes.py \
    --config configs/overfit_one_scene.yaml \
    --scans-root /scratch2/nedela/chorus_poc/scans \
    --num-scenes 3 \
    --max-steps 5000
```

**Expected outcome**: the independently-overfitted models should achieve higher metrics
on their specific scene (they train much longer on it), but the jointly-trained model
should show non-trivial generalization. This comparison validates that the multi-scene
setup is measuring something meaningful.

### Step 5.4 — Learning rate sensitivity check (optional)

Run 2–3 LR variants to verify the chosen LR is in a reasonable range:

```bash
# Lower LR
python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml train.lr=5e-5
# Higher LR
python scripts/run_multi_scene.py --config configs/multi_scene_10_3.yaml train.lr=3e-4
```

**Expected**: too-high LR → loss oscillates or diverges; too-low LR → loss decreases very
slowly. The chosen 1e-4 should be in the stable zone.

---

## File Change Summary

| File | Phase | Type | Description |
|------|-------|------|-------------|
| `student/data/multi_scene_dataset.py` | 1 | **New** | `MultiSceneDataset`, `build_scene_list` |
| `student/data/__init__.py` | 1 | Modify | Export new classes |
| `configs/splits/train_10.txt` | 3 | **New** | Training scene IDs |
| `configs/splits/val_3.txt` | 3 | **New** | Validation scene IDs |
| `configs/multi_scene_10_3.yaml` | 3 | **New** | Multi-scene experiment config |
| `scripts/run_multi_scene.py` | 3 | **New** | Entry point script |
| `student/engine/multi_scene_trainer.py` | 2 | **New** | `MultiSceneTrainer` |
| `student/engine/multi_scene_evaluator.py` | 4 | **New** | `evaluate_multi_scene` |

**Files that require NO changes** (contract preserved):

| File | Reason |
|------|--------|
| `student/models/student_model.py` | Same model, same `build_student_model` factory |
| `student/models/instance_decoder.py` | Same decoder, same output contract |
| `student/models/litept_wrapper.py` | Same backbone; cache invalidation via existing `clear_cache()` or disabling |
| `student/losses/mask_set_loss.py` | Same `MultiGranCriterion`; called per-scene as before |
| `student/engine/evaluator.py` | Same `evaluate_student_predictions_multi`; called per-scene from new evaluator |
| `student/engine/trainer.py` | `SingleSceneTrainer` untouched; overfitting still works |
| `student/metrics/pseudo_metrics.py` | Same functions, called per-scene |
| `student/data/training_pack.py` | Same loaders, called from new dataset |
| `student/data/single_scene_dataset.py` | Untouched; overfitting path preserved |
| `student/data/target_builder.py` | Same `build_instance_targets_multi`, called per-scene in trainer |
| `student/config_utils.py` | Reused for config loading, granularity parsing, seed setting |
| `scripts/run_student.py` | Untouched; single-scene overfitting still works |
| `scripts/run_overfit_benchmark_scenes.py` | Untouched; used for baseline comparison in Phase 5 |

---

## Estimated Complexity per Phase

| Phase | Steps | Files touched | Estimated effort | Produces working pipeline? |
|-------|-------|---------------|------------------|---------------------------|
| 1: Multi-Scene Dataset | 1.1–1.5 | 2 (1 new, 1 modify) | Small | Dataset only |
| 2: Multi-Scene Trainer | 2.1–2.7 | 1 (new) | Medium-Large | Yes (train + val) |
| 3: Config & Entry Point | 3.1–3.4 | 4 (all new) | Small | Yes (end-to-end) |
| 4: Evaluation & Metrics | 4.1–4.4 | 1 (new) | Small-Medium | Yes (with aggregation) |
| 5: Validation & Sanity | 5.1–5.4 | 0 (run only) | Medium (GPU time) | — |

---

## Relationship to Future Work

This 10/3 experiment is explicitly **not** the final training setup. It validates:

1. **Mechanical correctness**: the loop works, data flows, metrics are computed
2. **Generalization signal**: the model produces non-trivial predictions on unseen scenes
3. **Resource estimation**: per-epoch time and memory usage inform scaling decisions
4. **Hyperparameter ranges**: LR, warmup, grad clipping are in a reasonable range

**What comes after** (out of scope for this plan):

| Future step | Depends on |
|-------------|-----------|
| Scale to full ScanNet train split (~1201 scenes) | Phase 5 showing generalization works |
| Data augmentation (rotation, scaling, elastic distortion, point dropout) | Needed for full-scale training; not needed for 10-scene validation |
| Learning rate schedulers with longer schedules | Depends on knowing the right epoch count |
| Multi-GPU / DDP training | Needed for scale; overkill for 10 scenes |
| Pre-training the LitePT backbone (e.g. on ScanNet semantic seg) | Currently training from scratch; pretrained weights would enable backbone LR scaling and likely improve convergence |
| Backbone LR scaling (fine-tuning regime) | Only meaningful once pretrained backbone weights are available |

---

## References

1. <a id="ref-mask3d"></a>**[Mask3D]** Schult et al., *Mask3D for 3D Semantic Instance Segmentation*, ICRA 2023. [arXiv:2210.03105](https://arxiv.org/abs/2210.03105). Training protocol: batch size 1, LR 1e-4, AdamW, 600 epochs on ScanNet, cosine schedule.

2. <a id="ref-spformer"></a>**[SPFormer]** Sun et al., *Superpoint Transformer for 3D Scene Instance Segmentation*, AAAI 2023. [arXiv:2211.15766](https://arxiv.org/abs/2211.15766). Scene-level processing with sparse superpoints.

3. <a id="ref-oneformer3d"></a>**[OneFormer3D]** Kolodiazhnyi et al., *OneFormer3D: One Transformer for Unified Point Cloud Segmentation*, CVPR 2024. [arXiv:2311.14405](https://arxiv.org/abs/2311.14405).

4. <a id="ref-scannet"></a>**[ScanNet]** Dai et al., *ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes*, CVPR 2017. [arXiv:1702.04405](https://arxiv.org/abs/1702.04405). Standard evaluation: per-scene AP, then mean across scenes.

5. <a id="ref-finetune"></a>**[Fine-tuning LR scaling]** Common practice in transfer learning — training a pretrained backbone at 10× lower LR than the task-specific head. Used in Mask R-CNN (Detectron2), DETR, and downstream 3D methods. **Not applicable here**: my LitePT backbone is trained from scratch with no pretrained weights, so uniform LR is used. This technique becomes relevant if/when pretrained backbone weights are introduced.
