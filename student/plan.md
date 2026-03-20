# Multigranular Query-Based Decoder Upgrade Plan

> **Goal**: Replace the current one-shot prototype-matching decoder with a modern iterative
> query-refinement Transformer decoder, aligned with Mask3D, SPFormer, MAFT, and QueryFormer.
>
> **Guiding principle**: Each phase produces a fully working pipeline. No phase breaks
> `run_student.py` or `check_pipeline.py`. The output contract
> `{"point_embed", "heads": {"gXX": {"mask_logits", "score_logits", "query_embed"}}}` is
> preserved so the loss, evaluator, and metrics code keep working with minimal changes.

---

## Current Architecture (what we have)

```
LitePTBackbone
  forward(coord, feat) → dense_feat [N, 72]
    ├─ _voxelize: picks FIRST point per voxel (unstable)
    └─ returns: single tensor, no geometry, no sparse tokens

MultiHeadQueryInstanceDecoder
  forward(point_feat [N, 72]) → dict
    ├─ input_proj + point_head  →  point_embed [N, D]       (shared)
    ├─ per head:
    │     query_embed  →  nn.Embedding [Q, D]               (learned only)
    │     mask_logits  =  cosine_sim(query, point_embed)     (one-shot)
    │     soft_pool    =  softmax(mask) @ point_embed        (single pooling step)
    │     score_logits =  score_head(query + pooled)         (no iterative refinement)
    └─ no self-attention, no cross-attention, no positional guidance

StudentInstanceSegModel
  forward(points, features)
    point_feat = backbone(points, features)   # single tensor
    return decoder(point_feat)                # decoder sees no geometry
```

**What is missing** (vs. Mask3D / SPFormer / MAFT / QueryFormer):
- No iterative query refinement (multiple decoder layers)
- No query self-attention (queries can't resolve duplicates)
- No cross-attention to sparse scene tokens (queries attend to ALL dense points)
- No explicit 3D positional encoding (decoder is geometry-blind)
- No scene-aware query initialization (only learned embeddings)
- No deep supervision (loss only on final output)
- Backbone discards sparse voxel tokens and coordinates

---

## Target Architecture (what we are building)

```
LitePTBackbone
  forward(coord, feat) → LitePTBackboneOutput
    ├─ point_feat   [N, C]    dense per-point features  (for final mask dot-product)
    ├─ point_xyz    [N, 3]    original coordinates
    ├─ scene_tokens [V, C]    sparse voxel features     (for cross-attention memory)
    ├─ scene_xyz    [V, 3]    voxel centroids           (for positional encoding)
    └─ inverse_map  [N]       point → voxel mapping

MultiHeadQueryInstanceDecoder
  forward(point_feat, point_xyz, scene_tokens, scene_xyz) → dict
    ├─ scene_token_proj(scene_tokens)  →  scene_mem [V, D]     (sparse memory)
    ├─ point_mask_proj(point_feat)     →  mask_feat [N, D]     (dense mask basis)
    ├─ FourierPosEnc(scene_xyz)        →  scene_pos [V, D]     (3D position encoding)
    ├─ per head:
    │     HybridQueryInit(scene_tokens, scene_xyz) → q [Q, D], q_xyz [Q, 3]
    │     FourierPosEnc(q_xyz) → q_pos [Q, D]
    │     for layer in shared_trunk (4-6 layers):        ← ITERATIVE REFINEMENT
    │         q = self_attn(q, q)                        ← query-query interaction
    │         q = cross_attn(q, scene_mem)               ← query-to-scene reasoning
    │         q = FFN(q)
    │         [optional: predict intermediate masks]      ← deep supervision
    │     mask_logits = mask_embed(q) @ mask_feat.T
    │     score_logits = score_head(q)
    └─ output: same dict contract as before + optional aux_outputs

StudentInstanceSegModel
  forward(points, features)
    bb = backbone(points, features)          # structured dataclass
    return decoder(                          # decoder gets full geometry
      point_feat=bb.point_feat,
      point_xyz=bb.point_xyz,
      scene_tokens=bb.scene_tokens,
      scene_xyz=bb.scene_xyz,
    )
```

---

## Paper References

Each upgrade in this plan is grounded in one or more published methods. Short tags used
throughout (e.g. **[Mask3D]**) map to these entries:

| Tag | Paper | Key idea adopted | Ref |
|-----|-------|-----------------|-----|
| **[Mask3D]** | Schult et al., "Mask3D for 3D Semantic Instance Segmentation", ICRA 2023 | Iterative Transformer decoder that refines instance queries via multi-scale cross-attention to point-cloud features; deep supervision at every decoder layer | arXiv 2210.03105 |
| **[SPFormer]** | Sun et al., "Superpoint Transformer for 3D Scene Instance Segmentation", AAAI 2023 | Cross-attention over sparse superpoints instead of raw points — makes the decoder both faster and cleaner | arXiv 2211.15766 |
| **[MAFT]** | Lai et al., "Mask-Attention-Free Transformer for 3D Instance Segmentation", ICCV 2023 | Replaces mask-attention with position queries + Fourier relative position encoding + auxiliary center regression; much faster convergence | arXiv 2309.01692 |
| **[QueryFormer]** | Lu et al., "Query Refinement Transformer for 3D Instance Segmentation", ICCV 2023 | Query Initialization Module for high-coverage / low-repetition seeds; denoising + contrastive loss to suppress background queries | ICCV 2023 |
| **[SGIFormer]** | — | Semantic-guided query initialization + geometric-enhanced interleaving Transformer decoder | — |
| **[OneFormer3D]** | Kolodiazhnyi et al., "OneFormer3D: One Transformer for Unified Point Cloud Segmentation", CVPR 2024 | Unified instance/semantic/panoptic decoder with efficient query selection and matching | arXiv 2311.14405 |
| **[CompetitorFormer]** | — | Explicit competition-aware design so one dominant query emerges per instance instead of many duplicates fighting | — |
| **[Relation3D]** | — | Adaptive superpoint aggregation + relation-aware self-attention with positional and geometric relationships between queries | — |
| **[LaSSM]** | — | Local aggregation + coordinate-guided state-space decoder; replaces heavy global attention with efficient SSM refinement | — |
| **[DETR]** | Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020 | Set-prediction paradigm: learned queries + Hungarian matching + auxiliary losses at each decoder layer | arXiv 2005.12872 |
| **[Mask2Former]** | Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR 2022 | Masked cross-attention, per-pixel embedding for mask rendering separate from memory features | arXiv 2112.01527 |
| **[M2F3D]** | Nguyen & Vu, "Mask2Former for 3D Instance Segmentation", CVPR 2023 Workshop | Direct 3D validation of the Mask2Former recipe — shows the iterative decoder, separate mask-embed / memory-features streams, and set-prediction loss transfer to point clouds with minimal 3D-specific changes. Key bridge between [Mask2Former] (2D) and the fully 3D-native designs like [Mask3D] | — |

---

## Phase 1: Backbone Refactor ✅

> **Files**: `litept_wrapper.py`, `student_model.py`
> **Papers**: SPFormer (sparse scene representation), Mask3D (multi-scale backbone output)
> **Risk**: Low — backbone changes are isolated behind a clean API boundary
> **Outcome**: Backbone now exposes sparse tokens + geometry. Pipeline still works.

### Step 1.1 — Define `LitePTBackboneOutput` dataclass ✅

**File**: `litept_wrapper.py`

Add a `@dataclass` that defines the backbone's output contract:

| Field          | Shape    | Purpose                                   |
|----------------|----------|-------------------------------------------|
| `point_feat`   | `[N, C]` | Dense per-point features for mask logits  |
| `point_xyz`    | `[N, 3]` | Original point coordinates                |
| `scene_tokens` | `[V, C]` | Sparse voxel features for cross-attention |
| `scene_xyz`    | `[V, 3]` | Voxel centroids for positional encoding   |
| `inverse_map`  | `[N]`    | Maps each point to its voxel index        |

**Why**: Every modern decoder needs both sparse scene tokens (for efficient cross-attention)
and dense point features (for mask rendering). SPFormer uses superpoints; Mask3D uses
multi-scale sparse features. Our sparse tokens are the voxel-level features from LitePT.

### Step 1.2 — Upgrade `_voxelize` to mean-pool ✅

**File**: `litept_wrapper.py`

Replace the current "pick first point per voxel" strategy with mean-pooling:
- **Coordinates**: compute voxel centroid (mean of all points in the voxel)
- **Features**: compute mean of all point features in the voxel

**Implementation**: Use `torch.bincount` for voxel counts, `index_add_` for scatter-sum,
then divide by counts.

**Why**: The current approach picks an arbitrary point, introducing random variation in
voxel representatives across runs. Mean-pooling produces stable, representative anchors —
this matters when the decoder uses voxel centroids as geometric references.

### Step 1.3 — Add voxelization cache for single-scene overfitting ✅

**File**: `litept_wrapper.py`

Add `self._cached_voxelization = None` and cache the voxelization result during training
when `self.training is True`. Invalidate when switching to eval mode.

**Why**: In the `overfit_one_scene` pipeline, the exact same point cloud is processed every
iteration. Recomputing `torch.unique(grid_coord, ...)` thousands of times wastes GPU cycles.
This is purely an engineering optimization for the current training setup.

### Step 1.4 — Update `forward()` to return `LitePTBackboneOutput` ✅

**File**: `litept_wrapper.py`

Change the return type from `torch.Tensor` to `LitePTBackboneOutput`. Store both the raw
`out.feat` (sparse voxel features) and the densified `out.feat[inverse]` (dense point features).
Return the voxel centroids from `_voxelize`.

**This is a breaking change to the backbone API**, which is why Step 1.5 exists.

### Step 1.5 — Update `StudentInstanceSegModel` to unpack structured output ✅

**File**: `student_model.py`

Change the forward method from:
```python
point_feat = self.backbone(points, features)
return self.decoder(point_feat)
```
to:
```python
bb = self.backbone(points, features)
return self.decoder(
    point_feat=bb.point_feat,
    point_xyz=bb.point_xyz,
    scene_tokens=bb.scene_tokens,
    scene_xyz=bb.scene_xyz,
)
```

Also update the `build_student_model` factory to accept the new decoder parameters
(`num_decoder_layers`, `num_decoder_heads`, `query_init`, `use_positional_guidance`).

### Step 1.6 — Temporary backward-compatible decoder forward ✅

**File**: `instance_decoder.py`

Temporarily change `MultiHeadQueryInstanceDecoder.forward()` to accept the new keyword
arguments (`point_feat`, `point_xyz`, `scene_tokens`, `scene_xyz`) but internally only use
`point_feat` — same old logic, just a new signature.

This lets us land Phase 1 as a working commit before rewriting decoder internals.

### Step 1.7 — Verify Phase 1 ✅

Run `check_pipeline.py`. All checks must pass. Output shapes, gradients, and loss computation
must be identical to before. The only observable difference should be slightly different
numerical values from mean-pooling vs first-point voxelization.

---

## Phase 2: Decoder Rewrite — Core Components ✅

> **Files**: `instance_decoder.py`
> **Papers**: Mask3D (iterative decoder), MAFT (positional guidance), QueryFormer (query init),
> SPFormer (sparse cross-attention)
> **Risk**: Medium — this is the largest code change, but the output contract is preserved
> **Outcome**: Decoder now iteratively refines queries via Transformer layers with 3D awareness.

### Step 2.1 — Implement `FourierPosEnc` ✅

**File**: `instance_decoder.py`
**Primary source**: **[MAFT]** (arXiv 2309.01692, §3.2 Position-aware Cross-Attention)

A stateless module that maps `[..., 3]` coordinates to `[..., 3 * num_bands * 2]` embeddings
using `sin(x * 2^k)` and `cos(x * 2^k)` for `k = 0, ..., num_bands-1`.

**Why**: Transformers are permutation-invariant — without explicit positional encoding,
the decoder has zero awareness of 3D spatial relationships. **[MAFT]** demonstrates that
position-aware query designs (as opposed to mask-attention) lead to faster convergence and
better recall. **[Mask3D]** also uses positional encoding when attending to multi-scale
features. Fourier features capture high-frequency spatial variation that simple linear
projections of raw xyz cannot.

**Tensor flow**:
```
scene_xyz [V, 3]  →  FourierPosEnc  →  [V, 96]  →  Linear  →  [V, D]
query_xyz [Q, 3]  →  FourierPosEnc  →  [Q, 96]  →  Linear  →  [Q, D]
```
(96 = 3 coords × 16 bands × 2 trig functions, with `num_bands=16`)

### Step 2.2 — Implement `QueryDecoderLayer` ✅

**File**: `instance_decoder.py`
**Primary sources**: **[Mask3D]** (arXiv 2210.03105, §3.3 Transformer Decoder),
**[SPFormer]** (sparse cross-attention), **[DETR]** (set-prediction decoder pattern)

A single Transformer decoder layer with:

| Sub-layer         | Inputs                          | Purpose (paper reference)                |
|-------------------|---------------------------------|------------------------------------------|
| Self-Attention    | `(q + q_pos, q + q_pos, q)`    | Queries resolve duplicates (Mask3D, CompetitorFormer) |
| Cross-Attention   | `(q + q_pos, scene + s_pos, scene)` | Queries read scene geometry (SPFormer, Mask3D) |
| FFN               | `q`                             | Non-linear feature mixing                |

**Architecture choices**:
- **Pre-LayerNorm** (norm before attention, not after): empirically more stable for deep
  Transformers, standard in modern implementations.
- **Residual connections** on all three sub-layers.
- **Positional encoding added to Q and K only** (not V): the position tells attention *where*
  to look, but the value stream carries content features uncontaminated by position.
- `num_heads=8`, `ff_mult=4` as defaults (standard Transformer hyperparameters).

**Tensor shapes through one layer** (batch dim omitted for clarity):
```
q         [Q, D]   ← instance queries
q_pos     [Q, D]   ← Fourier pos of query anchors
scene     [V, D]   ← sparse voxel tokens (projected)
scene_pos [V, D]   ← Fourier pos of voxel centroids

Self-Attn:   Q×Q attention matrix over D-dim queries  → updated q [Q, D]
Cross-Attn:  Q×V attention matrix (queries attend to scene) → updated q [Q, D]
FFN:         Q independent position-wise transforms → updated q [Q, D]
```

**Why this is the single biggest upgrade**: The current decoder does ONE pass: query → similarity
→ pool → score. A 4-layer trunk gives queries 4 rounds of self-correction and scene-reading.
This is the core mechanism in Mask3D, SPFormer, MAFT, QueryFormer, SGIFormer, Relation3D, and
OneFormer3D.

### Step 2.3 — Implement `HybridQueryInitializer` ✅

**File**: `instance_decoder.py`
**Primary sources**: **[QueryFormer]** (ICCV 2023, §3.2 Query Initialization Module),
**[MAFT]** (position queries), **[SGIFormer]** (semantic-guided init),
**[Mask3D]** (sampled-point query init)

Initializes `Q` queries per granularity head as:
- **75% scene-sampled**: randomly select voxel tokens from `scene_tokens` (features) and their
  coordinates from `scene_xyz`. These queries start physically *on top of* real geometry.
- **25% learned**: `nn.Embedding(num_learned, hidden_dim)` free queries plus
  `nn.Embedding(num_learned, 3)` learned spatial anchors. These act as "wildcard" slots.

Returns `(q_feat [Q, D], q_xyz [Q, 3])`.

**Why (QueryFormer, MAFT, SGIFormer)**:
- QueryFormer explicitly shows that poor query initialization leads to low coverage (some objects
  get no query near them) and high repetition (multiple queries land on the same object).
- MAFT shows that position-anchored queries converge faster than content-only queries.
- SGIFormer uses semantic-guided initialization.
- LaSSM uses semantic-spatial initialization from superpoints.

Our hybrid approach balances scene grounding (sampled) with learning flexibility (free).

**Edge case**: If the scene has fewer voxels than `num_scene`, sample with replacement.

### Step 2.4 — Redesign `GranularityHead` as a lightweight output head ✅

**File**: `instance_decoder.py`
**Primary sources**: **[Mask3D]** / **[Mask2Former]** / **[M2F3D]** (mask-embed + class/score head
pattern validated in 3D), **[OneFormer3D]** (unified lightweight heads)

The current `GranularityHead` contains the query bank, the query MLP, and the score head.
In the new design, query initialization and refinement live in the shared trunk. Each head
becomes a thin projection:

| Component    | Architecture                         | Output       |
|--------------|--------------------------------------|--------------|
| `mask_embed` | LayerNorm → Linear → GELU → Linear  | `[Q, D]`     |
| `score_head` | LayerNorm → Linear → GELU → Linear  | `[Q, 1]`     |
| `logit_scale`| Learnable scalar (for cosine sim)    | scalar       |

**Why**: The shared trunk does the reasoning. The heads just project refined queries into
task-specific spaces. This is much better parameter allocation — three granularities share
the expensive self-attention and cross-attention weights but diverge only at the output.

### Step 2.5 — Rewrite `MultiHeadQueryInstanceDecoder` ✅

**File**: `instance_decoder.py`
**Primary sources**: **[Mask3D]** + **[SPFormer]** (shared iterative trunk over sparse tokens),
**[MAFT]** (positional guidance in cross-attention), **[Mask2Former]** / **[M2F3D]** (separate
mask-embed stream from memory stream — M2F3D validated this works in 3D)

Replace the entire class internals. The new structure:

```
__init__:
  scene_token_proj    (Linear stack: in_channels → hidden_dim)
  point_mask_proj     (Linear stack: in_channels → hidden_dim)
  pos_encoder         (FourierPosEnc)
  pos_proj            (Linear: fourier_dim → hidden_dim)
  initializers        (ModuleDict: one HybridQueryInitializer per granularity)
  layers              (ModuleList: N × QueryDecoderLayer)
  heads               (ModuleDict: one GranularityHead per granularity)

forward(point_feat, point_xyz, scene_tokens, scene_xyz):
  1. Project scene tokens → scene_mem [V, D]
  2. Compute scene positional encoding → scene_pos [V, D]
  3. Project dense points → mask_feat [N, D] (L2-normalized)
  4. For each granularity g:
     a. Initialize queries: q [Q, D], q_xyz [Q, 3]
     b. Compute query positional encoding: q_pos [Q, D]
     c. Add batch dim [1, Q, D] for nn.MultiheadAttention
     d. For each decoder layer: q = layer(q, q_pos, scene_mem, scene_pos)
     e. Predict: mask_logits = scale * (mask_embed(q) @ mask_feat.T)
     f. Predict: score_logits = score_head(q).squeeze(-1)
  5. Return same output dict contract
```

**Key design decision — separate projection streams**:
- `scene_token_proj`: maps raw LitePT voxel features into the decoder's hidden space.
  Used as keys/values in cross-attention. Operates on **sparse** tokens [V, D].
- `point_mask_proj`: maps raw LitePT per-point features into the mask embedding space.
  Used only at the final step for mask logit computation. Operates on **dense** features [N, D].

This separation is critical. SPFormer's core insight is that queries should cross-attend to a
sparse representation, not to all raw points. By keeping point_mask_proj separate, we avoid
the O(Q×N) cost during iterative refinement and only pay it once at the end.

**Output contract**: Identical to the current decoder. The loss, evaluator, metrics, trainer,
and vis code need zero changes (except for the optional `aux_outputs` addition in Phase 3).

### Step 2.6 — Verify Phase 2 ✅

Run `check_pipeline.py`:
- All shape checks must pass
- All gradient flow checks must pass
- Loss must be finite
- Forward and backward must complete without error

Run `run_student.py --max-steps 50` to verify training loop works end-to-end.

**Expected**: Metrics will initially be worse than the old decoder (random init of new params).
This is normal. The new decoder has much more capacity and needs training to converge.

---

## Phase 3: Config, Builder, and Training Integration ✅

> **Files**: `student_model.py`, `run_student.py`, `overfit_one_scene.yaml`, `trainer.py`,
> `check_pipeline.py`
> **Risk**: Low — mechanical config plumbing
> **Outcome**: New decoder parameters are fully configurable. Training loop is updated.

### Step 3.1 — Update `build_student_model` factory ✅

**File**: `student_model.py`

Add new parameters and pass them to the decoder constructor:

| Parameter                  | Default   | Controls                              |
|----------------------------|-----------|---------------------------------------|
| `num_decoder_layers`       | `4`       | Transformer trunk depth               |
| `num_decoder_heads`        | `8`       | Multi-head attention heads            |
| `query_init`               | `"hybrid"`| `"learned"` / `"scene"` / `"hybrid"` |
| `use_positional_guidance`  | `True`    | Fourier pos enc on/off                |
| `learned_query_ratio`      | `0.25`    | Fraction of learned vs scene queries  |

### Step 3.2 — Update `overfit_one_scene.yaml` ✅

**File**: `configs/overfit_one_scene.yaml`

Add decoder config block:
```yaml
model:
  backbone:
    name: litept
    litept_root: /home/nedela/projects/LitePT
    in_channels: 3
    grid_size: 0.02
  num_queries: 200
  decoder_hidden_dim: 256
  num_decoder_layers: 4
  num_decoder_heads: 8
  query_init: hybrid
  use_positional_guidance: true
  learned_query_ratio: 0.25
```

Increase `max_steps` from 5000 to at least 8000 — the Transformer decoder has more parameters
and needs more iterations to converge, especially with hybrid query init introducing randomness.

Consider lowering `lr` to `5e-5` as Transformers are sensitive to high learning rates.

### Step 3.3 — Update `run_student.py` to pass new config values ✅

**File**: `scripts/run_student.py`

Read new keys from `model_cfg` and pass them to `build_student_model`:
```python
model = build_student_model(
    ...
    num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
    num_decoder_heads=model_cfg.get("num_decoder_heads", 8),
    query_init=model_cfg.get("query_init", "hybrid"),
    use_positional_guidance=model_cfg.get("use_positional_guidance", True),
    learned_query_ratio=model_cfg.get("learned_query_ratio", 0.25),
)
```

### Step 3.4 — Add parameter groups for separate backbone/decoder LR ✅

**File**: `student_model.py` (method) + `trainer.py` (usage)

Add a `parameter_groups` method to `StudentInstanceSegModel`:
```python
def parameter_groups(self, backbone_lr_scale=0.1):
    return [
        {"params": list(self.backbone.parameters()), "lr_scale": backbone_lr_scale},
        {"params": list(self.decoder.parameters()), "lr_scale": 1.0},
    ]
```

Update `SingleSceneTrainer.__init__` to optionally use these groups with AdamW, applying
`lr_scale` as a multiplier on the base `lr`.

**Why**: The LitePT backbone is a pretrained encoder. Training it at the same rate as a
randomly initialized Transformer decoder typically destabilizes features. A 10x lower backbone
LR is standard practice.

### Step 3.5 — Update `check_pipeline.py` ✅

**File**: `scripts/check_pipeline.py`

Update `build_student_model` call to include `num_decoder_layers`. Update any decoder
introspection (e.g., parameter counting) to reflect the new trunk vs head split. Keep all
existing checks (shapes, gradients, losses, matching).

### Step 3.6 — Verify Phase 3 ✅

Run `check_pipeline.py` — all checks pass.
Run `run_student.py --config configs/overfit_one_scene.yaml --max-steps 100` — training
loop completes, losses decrease, wandb logging works with all new metrics.

---

## Phase 4: Auxiliary Deep Supervision ✅

> **Files**: `instance_decoder.py`, `mask_set_loss.py`
> **Papers**: Mask3D, DETR (auxiliary losses at each decoder layer)
> **Risk**: Low — additive change, existing loss contract unchanged
> **Outcome**: Loss is computed at every decoder layer, stabilizing Transformer training.

### Step 4.1 — Return intermediate predictions from decoder ✅

**File**: `instance_decoder.py`

After each decoder layer (not just the last), predict masks and scores using the same
GranularityHead. Collect these into an `aux_outputs` list:

```python
out["aux_outputs"] = [
    {"heads": {g: {"mask_logits": ..., "score_logits": ...} for g in self.granularities}}
    for layer_idx in range(num_layers - 1)  # all layers except the last
]
```

The final layer's output goes into the main `out["heads"]` as before.

**Why**: This is standard in Transformer-based set prediction (DETR, Mask3D, Mask2Former).
Early layers produce rough masks; applying loss at every layer creates a curriculum that helps
the network learn faster and avoids the "dead query" problem where some queries never activate.

### Step 4.2 — Update `MultiGranCriterion` to handle `aux_outputs` ✅

**File**: `mask_set_loss.py`

If `pred` contains an `aux_outputs` key, iterate over each auxiliary prediction dict and
compute the same MaskSetCriterion loss. Weight auxiliary losses by a decay factor
(e.g., `aux_weight = 0.4`).

```python
if "aux_outputs" in pred:
    for aux_pred in pred["aux_outputs"]:
        aux_loss = self._compute_all_heads(aux_pred, targets)
        total_loss += aux_weight * aux_loss
```

**Important**: Each auxiliary layer should run its own Hungarian matching, not reuse the
final layer's matching. This is standard practice from DETR and Mask3D.

### Step 4.3 — Add `aux_loss_weight` to config ✅

**File**: `configs/overfit_one_scene.yaml`

```yaml
loss:
  aux_weight: 0.4   # weight on intermediate decoder layer losses
```

### Step 4.4 — Verify Phase 4 ✅

Run training for 200 steps. Verify:
- `aux_outputs` are present in the forward pass output
- Loss includes auxiliary terms
- Loss decreases smoothly (should be more stable than without aux supervision)
- No NaN/Inf in gradients

---

## Phase 5: Validation and Comparison (ready to run)

> **Outcome**: Quantitative evidence that the new decoder improves over the baseline.
>
> All config knobs and infrastructure are in place. These steps require GPU runs.

### Step 5.1 — Run full baseline (old decoder) if not already saved

If no baseline metrics exist, temporarily revert to the old decoder and run the full
`overfit_one_scene` training (5000 steps). Record:
- Final matched mean IoU per granularity
- Final AP25/AP50 vs pseudo-GT
- Final AP25/AP50 vs real ScanNet GT
- Training time per step
- Total parameter count

### Step 5.2 — Run full new decoder training

Run the new decoder with the current config:
```bash
python scripts/run_student.py --config configs/overfit_one_scene.yaml
```

Expected improvements:
- Higher matched IoU (iterative refinement produces tighter masks)
- Higher AP50 (better boundary precision from 3D position guidance)
- Better score calibration (self-attention helps suppress background queries)
- Possibly slower per-step time (Transformer overhead), but better final metrics

### Step 5.3 — Ablation runs (optional but recommended)

Quick ablations to quantify the contribution of each component.
All configurable via CLI overrides:
1. **No positional guidance**: `model.use_positional_guidance=false`
2. **Learned-only queries**: `model.query_init=learned`
3. **2 decoder layers** vs **4 layers** vs **6 layers**: `model.num_decoder_layers=2`
4. **No auxiliary losses**: `loss.aux_weight=0.0`

Example:
```bash
python scripts/run_student.py --config configs/overfit_one_scene.yaml \
    model.num_decoder_layers=2 loss.aux_weight=0.0
```

---

## Phase 6: Multi-Scale Backbone Features (Mask3D / Mask2Former3D style)

> **Files**: `litept_wrapper.py`, `instance_decoder.py`, `student_model.py`,
> `configs/overfit_one_scene.yaml`
> **Papers**: Mask3D (multi-scale cross-attention), Mask2Former / M2F3D (per-layer scale
> assignment), SPFormer (sparse multi-scale features)
> **Risk**: Medium — requires extracting LitePT encoder internals and reworking decoder
> cross-attention, but the output contract is preserved
> **Outcome**: Decoder cross-attends to different spatial resolutions at different layers,
> giving coarse-to-fine refinement like the Mask2Former3D architecture diagram.

### Why multi-scale matters

Currently, all decoder layers cross-attend to **one** set of sparse voxel tokens (the final
LitePT encoder output). This means every refinement layer sees the same resolution.

In Mask3D and Mask2Former3D, each decoder layer attends to a **different feature pyramid
level**: early layers see coarse/large-receptive-field features (good for finding large
objects and rough shapes), later layers see fine/high-resolution features (good for refining
boundaries). This coarse-to-fine progression is a key reason those architectures work well.

### Background: LitePT's internal structure

LitePT has a U-Net-like encoder-decoder with multiple stages:

```
Encoder stages (LitePT.enc):
  Stage 0: enc_channels[0] = 36,  no pooling (original resolution)
  Stage 1: enc_channels[1] = 72,  GridPooling stride=2
  Stage 2: enc_channels[2] = 144, GridPooling stride=2
  Stage 3: enc_channels[3] = 252, GridPooling stride=2
  Stage 4: enc_channels[4] = 504, GridPooling stride=2

Decoder stages (LitePT.dec):
  Stage 0: dec_channels[0] = 72   (finest, after upsampling from stage 1)
  Stage 1: dec_channels[1] = 72
  Stage 2: dec_channels[2] = 144
  Stage 3: dec_channels[3] = 252
```

Currently we only use `out.feat` (the final decoded output at stage 0, channel dim 72).
To get multi-scale features, we need to **intercept intermediate stage outputs**.

### Step 6.1 — Understand LitePT's forward flow and identify tap points

**File**: `LitePT/litept/model.py` (read-only analysis)

LitePT's `forward()` runs:
1. `self.embedding(point)` — initial projection
2. `self.enc(point)` — sequential encoder stages (with GridPooling between stages)
3. `self.dec(point)` — sequential decoder stages (with UnPooling between stages)
4. Returns final `point` with `.feat` at decoder stage 0

**Tap points** for multi-scale features (prefer decoder stages since they incorporate
skip connections from the encoder):

| Level | Source           | Approx token count (V) | Channel dim | Spatial resolution |
|-------|------------------|------------------------|-------------|-------------------|
| Fine  | dec stage 0 out  | ~V (same as current)   | 72          | grid_size          |
| Mid   | dec stage 1 out  | ~V/4                   | 72          | grid_size × 2      |
| Coarse| dec stage 2 out  | ~V/16                  | 144         | grid_size × 4      |

We want 2–3 levels. Using more is diminishing returns and increases memory.

### Step 6.2 — Add hooks or modify LitePT forward to capture multi-scale outputs

**File**: `litept_wrapper.py`

**Option A — Register forward hooks (non-invasive, preferred)**:
```python
self._multi_scale_outputs: list[dict] = []

def _hook_fn(stage_idx):
    def hook(module, input, output):
        # output is a Point dict; capture feat and coord
        self._multi_scale_outputs.append({
            "feat": output.feat.clone(),
            "coord": output.coord.clone(),
        })
    return hook

# In __init__, register hooks on specific decoder stages:
for idx in [0, 1, 2]:
    self.model.dec[idx].register_forward_hook(_hook_fn(idx))
```

**Option B — Subclass LitePT.forward() (more control)**:
Override `forward()` to run encoder and decoder stage-by-stage, capturing intermediate
`Point` objects. More invasive but avoids hook fragility.

**Recommendation**: Start with Option A (hooks). Switch to B only if hooks cause issues
with gradient flow or caching.

### Step 6.3 — Extend `LitePTBackboneOutput` to carry multi-scale features

**File**: `litept_wrapper.py`

Add a new field to the dataclass:

```python
@dataclass
class LitePTBackboneOutput:
    point_feat: torch.Tensor       # [N, C] dense per-point features (unchanged)
    point_xyz: torch.Tensor        # [N, 3]
    scene_tokens: torch.Tensor     # [V, C] finest scale (unchanged, for backward compat)
    scene_xyz: torch.Tensor        # [V, 3]
    inverse_map: torch.Tensor      # [N]

    # NEW: multi-scale sparse features from decoder stages
    multi_scale_tokens: list[torch.Tensor]  # [tokens_fine, tokens_mid, tokens_coarse]
    multi_scale_xyz: list[torch.Tensor]     # [xyz_fine, xyz_mid, xyz_coarse]
```

Each entry in `multi_scale_tokens[i]` has shape `[V_i, C_i]` where `V_i` decreases and
`C_i` may differ across scales.

`scene_tokens` / `scene_xyz` remain as the finest scale for backward compatibility.

### Step 6.4 — Add per-scale projection layers to the decoder

**File**: `instance_decoder.py`

Currently there is one `scene_token_proj` that maps `[V, C_in] → [V, D]`. For multi-scale,
add one projection per scale level:

```python
self.scale_projs = nn.ModuleList([
    nn.Sequential(
        nn.Linear(channels_i, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    for channels_i in scale_channels  # e.g. [72, 72, 144]
])
```

Also add per-scale positional encoding (reuse `FourierPosEnc` + `pos_proj`, or add
per-scale `pos_proj` if channel dims differ).

### Step 6.5 — Assign scales to decoder layers (round-robin or coarse-to-fine)

**File**: `instance_decoder.py`

**Strategy 1 — Coarse-to-fine** (Mask3D default):
- Layer 0 cross-attends to coarsest scale (large receptive field, rough localization)
- Layer 1 cross-attends to mid scale
- Layer 2 cross-attends to finest scale (boundary refinement)
- If more layers than scales, cycle back (layer 3 → coarse, etc.)

```python
def _get_scale_for_layer(self, layer_idx: int) -> int:
    num_scales = len(self.scale_projs)
    # Coarse-to-fine: layer 0 → coarsest, layer (num_scales-1) → finest
    # Reverse index, then cycle
    return (num_scales - 1) - (layer_idx % num_scales)
```

**Strategy 2 — All-to-all** (Mask2Former style):
Each layer attends to all scales concatenated. Simpler but more memory.

**Recommendation**: Start with Strategy 1 (coarse-to-fine round-robin). It is what Mask3D
uses and is more memory-efficient.

### Step 6.6 — Update `_forward_unbatched` to use multi-scale cross-attention

**File**: `instance_decoder.py`

Change the decoder loop from:

```python
# Current: single scale
scene_mem = self.scene_token_proj(scene_tokens)
scene_pos = self.pos_proj(self.pos_encoder(scene_xyz))
for layer in self.layers:
    q = layer(q, q_pos, scene_mem, scene_pos)
```

to:

```python
# New: per-layer scale selection
scale_mems = [proj(tokens) for proj, tokens in
              zip(self.scale_projs, multi_scale_tokens)]
scale_poss = [self.pos_proj(self.pos_encoder(xyz))
              for xyz in multi_scale_xyz]

for layer_idx, layer in enumerate(self.layers):
    s = self._get_scale_for_layer(layer_idx)
    q = layer(q, q_pos, scale_mems[s], scale_poss[s])
```

`QueryDecoderLayer` itself needs **no changes** — it already takes generic
`(q, q_pos, scene_mem, scene_pos)` tensors. The only difference is which scale's
memory and positions are passed in.

### Step 6.7 — Update `StudentInstanceSegModel` to pass multi-scale data

**File**: `student_model.py`

Update `forward()` to pass the new fields:

```python
bb = self.backbone(points, features)
return self.decoder(
    point_feat=bb.point_feat,
    point_xyz=bb.point_xyz,
    scene_tokens=bb.scene_tokens,
    scene_xyz=bb.scene_xyz,
    multi_scale_tokens=bb.multi_scale_tokens,
    multi_scale_xyz=bb.multi_scale_xyz,
)
```

### Step 6.8 — Add config knobs

**File**: `configs/overfit_one_scene.yaml`

```yaml
model:
  backbone:
    multi_scale: true          # enable multi-scale feature extraction
    multi_scale_stages: [0, 1, 2]  # which decoder stages to tap
  # num_decoder_layers should ideally be >= len(multi_scale_stages)
  num_decoder_layers: 3
```

### Step 6.9 — Backward compatibility

When `multi_scale: false` (default), the backbone returns empty lists for
`multi_scale_tokens` / `multi_scale_xyz`, and the decoder falls back to the current
single-scale behavior. This ensures no existing configs or runs break.

### Step 6.10 — Verify Phase 6

Run `check_pipeline.py` with `multi_scale: true`:
- Verify multi-scale tensors have expected shapes and decreasing token counts
- Verify gradient flow through all scale projections
- Verify decoder output shapes are unchanged

Run `run_student.py --max-steps 200` and compare loss/metrics to single-scale baseline.

**Expected**: Slightly slower per-step (more projections), but potentially better
convergence on `g02` (fine-grained), where multi-scale context should help most.

---

## Phase 7: Future Improvements (do NOT implement yet)

> These are second-wave upgrades. Build them only after Phase 6 shows the base decoder works.

### 7.1 — Duplicate query penalty (CompetitorFormer)

Add a lightweight regularizer that penalizes high cosine similarity between top-scoring
queries. This discourages multiple queries from competing for the same instance.

```python
def duplicate_query_penalty(query_embed, score_logits, topk=32):
    scores = score_logits.sigmoid()
    idx = scores.topk(min(topk, scores.numel())).indices
    q = F.normalize(query_embed[idx], dim=-1)
    sim = q @ q.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim[~eye].pow(2).mean()
```

Add with small weight (~0.01) after the main decoder is working well.

### 7.2 — FPS-based query initialization

Replace random voxel sampling with Farthest Point Sampling on voxel centroids. This
guarantees maximum spatial coverage and directly addresses QueryFormer's concern about
low-coverage initialization.

### 7.3 — Per-layer anchor update (MAFT)

Add a small MLP that predicts a `delta_xyz` at each decoder layer, updating the query's
spatial anchor. This lets queries "walk" toward object centers during refinement.

### 7.4 — Per-head query counts

Allow different numbers of queries per granularity:
```yaml
num_queries_by_granularity:
  g02: 96    # fine: fewer, smaller objects
  g05: 128   # medium
  g08: 192   # coarse: more, but larger objects
```

### 7.5 — Relation-aware self-attention (Relation3D)

Add geometric-aware self-attention where the attention bias includes the Euclidean distance
between query anchors. This helps queries understand spatial relationships between instances.

### 7.6 — Mask-guided cross-attention refinement

After the first decoder layer produces initial masks, use those masks to restrict which
scene tokens each query attends to in subsequent layers (similar to Mask2Former's masked
attention, adapted for 3D).

---

## File Change Summary

| File | Phase | Type of change |
|------|-------|---------------|
| `student/models/litept_wrapper.py` | 1, 6 | Dataclass output, mean-pool voxel, cache (P1); multi-scale hooks + extended dataclass (P6) |
| `student/models/instance_decoder.py` | 1, 2, 4, 6 | Temporary compat (P1), full rewrite (P2), aux outputs (P4), per-scale projections + layer-scale assignment (P6) |
| `student/models/student_model.py` | 1, 3, 6 | Structured backbone → decoder bridge, param groups (P1/P3); pass multi-scale data (P6) |
| `student/losses/mask_set_loss.py` | 4 | Auxiliary deep supervision |
| `configs/overfit_one_scene.yaml` | 3, 4, 6 | New decoder + loss config params (P3/P4); multi_scale backbone config (P6) |
| `scripts/run_student.py` | 3 | Pass new config keys to builder |
| `scripts/check_pipeline.py` | 3 | Update for new API |
| `student/engine/trainer.py` | 3 | Parameter groups, separate LRs |

**Files that require NO changes** (output contract preserved):
| File | Reason |
|------|--------|
| `student/engine/evaluator.py` | Consumes `heads[g]["mask_logits"]` / `heads[g]["score_logits"]` — unchanged |
| `student/engine/vis.py` | Same |
| `student/metrics/pseudo_metrics.py` | Same |
| `student/data/target_builder.py` | Data pipeline is independent of decoder |
| `student/data/single_scene_dataset.py` | Same |
| `student/data/training_pack.py` | Same |

---

## Estimated Complexity per Phase

| Phase | Steps | Files touched | Estimated effort | Produces working pipeline? |
|-------|-------|---------------|------------------|---------------------------|
| 1: Backbone Refactor | 1.1–1.7 | 3 | Small | Yes |
| 2: Decoder Rewrite | 2.1–2.6 | 1 | Large | Yes |
| 3: Config & Training | 3.1–3.6 | 5 | Medium | Yes |
| 4: Aux Supervision | 4.1–4.4 | 3 | Small | Yes |
| 5: Validation | 5.1–5.3 | 0 (config only) | Medium (GPU time) | — |
| 6: Multi-Scale Features | 6.1–6.10 | 4 | Medium | Yes |

---

## Key Design Decisions and Rationale

### Why sparse scene tokens instead of all dense points for cross-attention?

A scene with N=100k points and Q=200 queries would create a Q×N = 20M attention matrix **per
layer, per head**. With 4 layers and 8 heads, that is 640M attention weights — infeasible.

Using V~5k sparse voxel tokens (at grid_size=0.02), the attention matrix is Q×V = 1M per
layer per head, a 20× reduction. SPFormer's superpoint cross-attention and Mask3D's multi-scale
sparse features both demonstrate this approach works without mask quality loss.

### Why hybrid query initialization instead of pure learned or pure scene-sampled?

- **Pure learned** (current): Queries have no spatial grounding. They must learn, from scratch,
  where objects tend to appear. QueryFormer shows this leads to low coverage and duplicates.
- **Pure scene-sampled**: All queries are tied to existing geometry. No capacity to hallucinate
  objects that span multiple voxels or to act as "background absorbers."
- **Hybrid** (75/25): Scene-sampled queries guarantee coverage; learned queries provide
  flexibility. This is the approach used by BFL and similar to SGIFormer's design.

### Why shared trunk instead of per-head decoders?

Three separate 4-layer Transformers would triple the decoder parameters with no benefit.
The three granularities differ at the output level (what size objects to segment), not at
the reasoning level (how to understand scene geometry). A shared trunk learns one strong
representation; heads just project it into granularity-specific predictions.

### Why Fourier positional encoding instead of learned positional encoding?

Fourier features generalize to arbitrary coordinate ranges without needing to discretize or
bin space. Learned positional encodings (e.g., lookup tables) require knowing the spatial
extent at init time and don't extrapolate. MAFT uses relative position encoding;
our Fourier approach is simpler and achieves the same goal of injecting geometric awareness.
