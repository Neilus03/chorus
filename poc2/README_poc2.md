# CHORUS PoC2 (Class-Agnostic Instances with UnSAMv2)

This folder contains a 3-step pipeline:

1. `1_run_unsam_teacher.py`
   - Runs UnSAMv2 whole-image segmentation at fixed granularity (`GRANULARITY=0.8` by default).
   - Saves per-frame local instance masks to `scene0000_00/unsam_masks_g0.8/*.npy`.

2. `2_bridge_and_cluster.py`
   - Bridges 2D local masks to 3D vertices.
   - Builds sparse point-mask signatures.
   - Compresses with TruncatedSVD and clusters with HDBSCAN.
   - Saves `scene0000_00/chorus_instance_labels_g0.8.npy` and colored `.ply`.

3. `3_evaluate_instances.py`
   - Computes class-agnostic instance quality with ARI/NMI against ScanNet GT instances.

## Scratch-first storage layout

To avoid home-directory quota, this setup stores heavy artifacts in `/scratch2/nedela` by default:

- venv/conda env: `/scratch2/nedela/chorus_poc2/envs/...`
- package/model caches: `/scratch2/nedela/chorus_poc2/caches/...`
- UnSAMv2 checkpoint: `/scratch2/nedela/chorus_poc2/checkpoints/unsamv2/unsamv2_plus_ckpt.pt`

You can override with:

```bash
export SCRATCH_ROOT=/scratch2/nedela
export RUNTIME_ROOT=/scratch2/nedela/chorus_poc2
```

## Setup

```bash
cd /home/nedela/projects/chorus/poc2
bash setup_unsamv2.sh
```

## UnSAMv2 checkpoint

Put the checkpoint at:

- `/scratch2/nedela/chorus_poc2/checkpoints/unsamv2/unsamv2_plus_ckpt.pt`

or override path explicitly:

```bash
export UNSAMV2_CKPT=/path/to/unsamv2_plus_ckpt.pt
```

Optional config override:

```bash
export UNSAMV2_CFG=configs/unsamv2_small.yaml
```

## Run

```bash
cd /home/nedela/projects/chorus/poc2
python 1_run_unsam_teacher.py
python 2_bridge_and_cluster.py
python 3_evaluate_instances.py
```

To test finer over-segmentation behavior, change `GRANULARITY` from `0.8` to `0.2` in all three scripts and rerun.


## Benchmark throughput

Use the benchmark utility to estimate real sec/frame and projected ScanNet wall-clock:

```bash
cd /home/nedela/projects/chorus/poc2
python benchmark_unsam_teacher.py
```

Optional overrides:

```bash
BENCH_NUM_FRAMES=50 BENCH_WARMUP_FRAMES=3 BENCH_FRAME_SKIP=10 SCANNET_NUM_SCENES=1513 python benchmark_unsam_teacher.py
```
