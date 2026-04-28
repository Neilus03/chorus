#!/bin/bash
#SBATCH --job-name=student_scannet_ft_gt_ddp4
#SBATCH --output=student/logs/scannet_ft_gt_ddp4_%j.out
#SBATCH --error=student/logs/scannet_ft_gt_ddp4_%j.err
#SBATCH --partition=gpuhe.120h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:4
#SBATCH --time=48:00:00

set -euo pipefail

# 1. Load cluster modules (adjust to your cluster if needed)
module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy

# 1b. Proxy (only needed if your cluster requires it)
export http_proxy=http://proxy.ethz.ch:3128
export https_proxy=http://proxy.ethz.ch:3128
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

# 2. Activate your Python environment
source /cluster/work/igp_psr/nedela/litept-env/bin/activate

# 3. Go to repo student root (so relative config paths work)
cd /cluster/home/nedela/nedela/projects/chorus/student
mkdir -p logs

# Ensure `chorus.datasets.*` is importable in DataLoader workers too.
# Repo layout is `chorus/chorus/chorus/...`, so we need the outer `.../chorus/chorus`.
export PYTHONPATH="/cluster/home/nedela/nedela/projects/chorus/chorus:${PYTHONPATH:-}"

# 4. --- runtime knobs ---
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB__SERVICE_WAIT=120
export WANDB_DIR=${TMPDIR:-/tmp}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline

# 5. Optional: set portable path env vars used by YAML configs.
export CHORUS_OUTPUT_ROOT="/cluster/work/igp_psr/nedela/student_runs"
export SCANNET_SCANS_ROOT="/cluster/work/igp_psr/nedela/chorus_poc/scans"
export LITEPT_ROOT="/cluster/work/igp_psr/nedela/LitePT"

# PointROPE CUDA extension may be compiled for a different SM; force torch fallback.
export POINTROPE_FORCE_TORCH=1

CONFIG_ARG="${1:-configs/scannet_full_aug_ddp4_batch_pointcept_ft100_gt.yaml}"
shift || true

# Normalize config paths: this script `cd`s into the student root, so allow
# callers to pass either `configs/foo.yaml` or `student/configs/foo.yaml`.
if [[ "${CONFIG_ARG}" == student/* ]]; then
  CONFIG_ARG="${CONFIG_ARG#student/}"
fi

echo "Running multi-scene finetune (ScanNet GT)"
echo "  CONFIG=${CONFIG_ARG}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  EXTRA_ARGS=$*"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/run_multi_scene.py --config "${CONFIG_ARG}" "$@"

