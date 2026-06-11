# CHORUS Point Cloud Inspector

Interactive point-cloud inspection UI for CHORUS training packs and student prediction bundles.

## Run

```bash
cd ui
npm install
npm run dev -- --host 0.0.0.0
```

Open the printed Vite URL.

For the `Run/load` prediction button, start the dev server from an environment
that can import the student stack and can run the checkpoint on the requested
device:

```bash
cd ui
export CHORUS_UI_PYTHON=/path/to/python-with-numpy-torch-yaml
export LITEPT_ROOT=/cluster/work/igp_psr/nedela/LitePT
npm run dev -- --host 0.0.0.0
```

On a login node without CUDA, the button will fail fast with an environment
error instead of silently showing an empty prediction layer. For prediction
export, submit the exporter through Slurm or run it inside the `srun --pty`
shell. Do not `ssh` into the allocated node and run the exporter there; that
can bypass parts of the Slurm step environment and makes debugging harder.

## Open A Training Pack

```text
http://<host>:5173/?pack=/path/to/scene_or_training_pack
```

The Vite dev server resolves either a scene directory containing `training_pack/` or a pack directory containing `scene_meta.json`.

Verified ScanNet example:

```text
http://<host>:5173/?pack=/cluster/work/igp_psr/nedela/chorus_poc/scans/scene0000_00
```

## Export And Open A Prediction Bundle

```bash
python ui/scripts/export_inspection_bundle.py \
  --config student/configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --scenes scene0488_00 \
  --out-dir /path/to/inspection_bundles
```

Then open:

```text
http://<host>:5173/?bundle=/path/to/inspection_bundles/scene0488_00/inspection_bundle.json
```

On Euler, a practical one-scene GPU export is:

```bash
sbatch --partition=gpuhe.4h --gpus=rtx_3090:1 --cpus-per-task=8 --mem-per-cpu=4G --time=02:00:00 \
  --job-name=chorus_ui_export \
  --output=/cluster/work/igp_psr/nedela/chorus_ui_bundles/export_%j.out \
  --error=/cluster/work/igp_psr/nedela/chorus_ui_bundles/export_%j.err \
  --wrap='cd /cluster/home/nedela/nedela/projects/chorus && export LITEPT_ROOT=/cluster/work/igp_psr/nedela/LitePT && /cluster/work/igp_psr/nedela/litept-env/bin/python -u ui/scripts/export_inspection_bundle.py --config student/configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml --checkpoint /cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_v2_pseudo_pretrain/checkpoints/best.pt --scene-dir /cluster/work/igp_psr/nedela/chorus_poc/scans/scene0000_00 --out-dir /cluster/work/igp_psr/nedela/chorus_ui_bundles --granularities g05 --device cuda:0'
```

Watch progress with:

```bash
squeue -u "$USER" --name=chorus_ui_export
tail -f /cluster/work/igp_psr/nedela/chorus_ui_bundles/export_<jobid>.out
tail -f /cluster/work/igp_psr/nedela/chorus_ui_bundles/export_<jobid>.err
```

## Local File Access

The dev server only serves files under allowlisted roots. By default this includes the repo root plus common CHORUS/ScanNet output roots. Override with:

```bash
export CHORUS_UI_ROOTS=/path/to/scans:/path/to/runs:/path/to/chorus
```

## Verification

```bash
cd ui
npm run test
npm run build
npm run test:e2e
python3 -m unittest discover scripts/tests
```
