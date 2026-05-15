#!/bin/bash
#SBATCH --job-name=student_vis_compare
#SBATCH --output=student/logs/vis_compare_%j.out
#SBATCH --error=student/logs/vis_compare_%j.err
#SBATCH --partition=gpuhe.4h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=02:00:00

set -euo pipefail

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy
source /cluster/work/igp_psr/nedela/litept-env/bin/activate

cd /cluster/home/nedela/nedela/projects/chorus/student
mkdir -p logs

export PYTHONPATH="/cluster/home/nedela/nedela/projects/chorus/chorus:${PYTHONPATH:-}"
export CHORUS_OUTPUT_ROOT="/cluster/work/igp_psr/nedela/student_runs"
export SCANNET_SCANS_ROOT="/cluster/work/igp_psr/nedela/chorus_poc/scans"
export LITEPT_ROOT="/cluster/work/igp_psr/nedela/LitePT"
export CHORUS_SCANNET_METADATA_ROOT="${LITEPT_ROOT}/datasets/preprocessing/scannet/meta_data"
export POINTROPE_FORCE_TORCH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_ARG="${1:-configs/scannet_full_continuous_ft_gt_scannet20_classagnostic.yaml}"
shift || true

python scripts/visualize_checkpoint_comparison.py --config "${CONFIG_ARG}" "$@"
