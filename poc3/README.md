# PoC3 Streaming Pipeline

`poc3` is a modular streaming pipeline to process ScanNet scenes one-by-one:

1. download raw ScanNet scene files,
2. extract RGB-D frames from `.sens`,
3. run CHORUS stages (teacher labels -> 3D projection/voting -> SVD -> HDBSCAN),
4. evaluate oracle metrics,
5. write per-scene manifest,
6. optionally clean heavy intermediates.

## Files

- `config.yaml` - primary runtime configuration
- `config.py` - YAML loader + resolved path/constants
- `scannet_download.py` - non-interactive SceneNet scene downloader wrapper
- `sens_extract.py` - `.sens` -> `color/depth/pose/intrinsic`
- `io_utils.py` - output checks, cleanup, manifest helpers
- `chorus_teacher.py` - CHORUS teacher mask generation
- `chorus_project_cluster.py` - 2D-to-3D projection, voting, SVD, HDBSCAN
- `chorus_oracle_eval.py` - combined oracle evaluation
- `chorus_eval_utils.py` - GT instance loading helpers
- `chorus_pipeline.py` - per-scene CHORUS orchestration
- `run_streaming_scannet.py` - top-level streaming orchestrator
- `wandb_utils.py` - Weights & Biases reporting integration

## Requirements

The runtime environment must have:

- `imageio`
- `numpy`
- `PyYAML`
- `wandb` (if `wandb.enabled: true` in `config.yaml`)
- `torch`
- `Pillow`
- `open3d`
- `opencv-python`
- `scipy`
- `scikit-learn`
- `plyfile`

`sens_extract.py` expects `SensorData.py` from:

`/home/nedela/projects/LitePT/datasets/preprocessing/scannet/scannet_pair`

`chorus_teacher.py` expects an UnSAMv2 checkout at:

- `${UNSAM_ROOT}` if set, otherwise
- `poc3/UnSAMv2`

## Usage

Run a small test:

```bash
python run_streaming_scannet.py --scene-list-file /path/to/scenes.txt --max-scenes 2 --continue-on-error
```

Run full release list in streaming mode:

```bash
python run_streaming_scannet.py --continue-on-error
```

Disable cleanup:

```bash
python run_streaming_scannet.py --scene-list-file /path/to/scenes.txt --no-delete-intermediate
```

Disable W&B for one run:

```bash
python run_streaming_scannet.py --wandb-off
```

## Outputs

- Run-level report:
  - `/scratch2/nedela/chorus_poc/reports_poc3/poc3_streaming_run_<timestamp>.txt`
- Scene-level manifest:
  - `<scene_dir>/poc3_manifest.json`

## Cleanup policy

When cleanup is enabled, for non-retained scenes the pipeline deletes:

- `color/`, `depth/`, `pose/`, `intrinsic/`
- `unsam_masks_g*/`
- `*.sens`, `*_2d-*.zip`
- `svd_features_g*.npy`

Retained scenes are selected deterministically by hash with:

`keep_full_scene = (sha1(scene_id) % keep_full_modulo == 0)`

## W&B Reporting

If enabled in `config.yaml`, `poc3` reports:

- run-level config and summary counters
- per-scene status, duration, cleanup and verification stats
- per-scene AP mean metrics from `oracle_metrics.json`
- scene summary table (`wandb.Table`)
- scene manifest artifacts (`poc3_manifest.json`)
- run report artifact (`poc3_streaming_run_<timestamp>.txt`)

