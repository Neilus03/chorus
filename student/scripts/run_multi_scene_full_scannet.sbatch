#!/bin/bash
#SBATCH --job-name=student_full_scannet
#SBATCH --output=logs/full_scannet_%j.out
#SBATCH --error=logs/full_scannet_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
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

# 4. --- THE GOLDEN FIXES ---
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB__SERVICE_WAIT=120
export WANDB_DIR=${TMPDIR:-/tmp}
# Euler proxy blocks wandb API (403); offline run → sync from a machine with working net:
#   wandb sync <run_dir printed in the log>
export WANDB_MODE=offline

# 5. Optional: set portable path env vars used by YAML configs.
# If you don't set these, the defaults embedded in the YAML are used.
export CHORUS_OUTPUT_ROOT="/cluster/work/igp_psr/nedela/student_runs"
export SCANNET_SCANS_ROOT="/cluster/work/igp_psr/nedela/chorus_poc/scans"
export LITEPT_ROOT="/cluster/work/igp_psr/nedela/LitePT"


CONFIG_ARG="${1:-full_scannet}"
shift || true

# Allow either a short name or a direct path
case "${CONFIG_ARG}" in
  full_scannet|scannet_full)
    CONFIG_PATH="configs/scannet_full.yaml"
    ;;
  *)
    CONFIG_PATH="${CONFIG_ARG}"
    ;;
esac

echo "Running multi-scene training"
echo "  CONFIG=${CONFIG_PATH}"
echo "  EXTRA_ARGS=$*"

python -u scripts/run_multi_scene.py --resume /cluster/work/igp_psr/nedela/student_runs/full_scannet/checkpoints/last.pt --config "${CONFIG_PATH}" "$@"

