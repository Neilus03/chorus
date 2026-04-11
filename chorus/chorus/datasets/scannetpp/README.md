# ScanNet++ Dataset Support

CHORUS supports an iPhone-first ScanNet++ workflow that keeps the official dataset layout:

- `metadata/`
- `splits/`
- `data/<scene_id>/`

## Required Scene Assets

The streaming path expects these raw files under `data/<scene_id>/`:

- `scans/mesh_aligned_0.05.ply`
- `iphone/rgb.mkv`
- `iphone/depth.bin`
- `iphone/pose_intrinsic_imu.json`

Oracle evaluation additionally requires:

- `scans/segments.json`
- `scans/segments_anno.json`

`mesh_aligned_0.05_semantic.ply` is optional for v1 and is not required by the CHORUS adapter.

## Downloading

Set a token before using the downloader or streaming entrypoint:

```bash
export CHORUS_SCANNETPP_TOKEN="..."
```

Programmatic helpers live in `chorus/datasets/scannetpp/download.py`.

## Supported Splits

- `nvs_sem_train`: training/inference + oracle evaluation
- `nvs_sem_val`: validation/inference + oracle evaluation
- `sem_test`: inference only, no oracle evaluation
- `nvs_test`: unsupported for CHORUS 3D lifting because scan assets are absent
- `nvs_test_iphone`: unsupported for CHORUS 3D lifting because scan assets are absent

## Entry Points

Single scene:

```bash
python chorus/scripts/run_scene.py --dataset scannetpp --scene-dir /path/to/scannetpp/data/<scene_id>
```

Streaming:

```bash
python chorus/scripts/run_streaming_scannetpp.py --dataset-root /path/to/scannetpp --split nvs_sem_val
```
