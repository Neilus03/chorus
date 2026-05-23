#!/bin/bash
#SBATCH --job-name=score_thresh_sweep
#SBATCH --output=student/logs/score_thresh_sweep_%j.out
#SBATCH --error=student/logs/score_thresh_sweep_%j.err
#SBATCH --partition=gpuhe.120h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00

set -euo pipefail

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy

export http_proxy=http://proxy.ethz.ch:3128
export https_proxy=http://proxy.ethz.ch:3128
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

source /cluster/work/igp_psr/nedela/litept-env/bin/activate

cd /cluster/home/nedela/nedela/projects/chorus/student
mkdir -p logs

export PYTHONPATH="/cluster/home/nedela/nedela/projects/chorus/chorus:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline

export CHORUS_OUTPUT_ROOT="/cluster/work/igp_psr/nedela/student_runs"
export SCANNET_SCANS_ROOT="/cluster/work/igp_psr/nedela/chorus_poc/scans"
export LITEPT_ROOT="/cluster/work/igp_psr/nedela/LitePT"
export CHORUS_SCANNET_METADATA_ROOT="${LITEPT_ROOT}/datasets/preprocessing/scannet/meta_data"
export POINTROPE_FORCE_TORCH=1

DEFAULT_CONFIG="configs/scannet_full_continuous_ft_gt_scannet20_classagnostic.yaml"
DEFAULT_RUN_DIR="/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_prompt_ft_gt_scannet20_classagnostic_iou_score_balanced"
DEFAULT_CHECKPOINT="${DEFAULT_RUN_DIR}/checkpoints/best.pt"
DEFAULT_OUTPUT_DIR="${DEFAULT_RUN_DIR}/threshold_sweep"
DEFAULT_CACHE_PATH="${DEFAULT_OUTPUT_DIR}/val_predictions_pre_score_threshold.pt"
DEFAULT_THRESHOLDS="0.0,0.001,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5"

CONFIG_ARG="${DEFAULT_CONFIG}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_ARG="${1}"
  shift
fi

CHECKPOINT_ARG="${DEFAULT_CHECKPOINT}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CHECKPOINT_ARG="${1}"
  shift
fi

if [[ "${CONFIG_ARG}" == student/* ]]; then
  CONFIG_ARG="${CONFIG_ARG#student/}"
fi

echo "Running eval-only score-threshold sweep"
echo "  CONFIG=${CONFIG_ARG}"
echo "  CHECKPOINT=${CHECKPOINT_ARG}"
echo "  OUTPUT_DIR=${DEFAULT_OUTPUT_DIR}"
echo "  CACHE_PATH=${DEFAULT_CACHE_PATH}"
echo "  EXTRA_ARGS=$*"

python scripts/sweep_score_thresholds.py \
  --config "${CONFIG_ARG}" \
  --checkpoint "${CHECKPOINT_ARG}" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  --cache-path "${DEFAULT_CACHE_PATH}" \
  --thresholds "${DEFAULT_THRESHOLDS}" \
  --device cuda:0 \
  --write-cache \
  "$@"

