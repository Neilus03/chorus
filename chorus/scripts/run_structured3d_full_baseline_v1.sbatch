#!/bin/bash
#SBATCH --job-name=s3d_pl_full
#SBATCH --output=/cluster/home/nedela/nedela/projects/chorus/chorus/logs/structured3d_full_%A_%a.out
#SBATCH --error=/cluster/home/nedela/nedela/projects/chorus/chorus/logs/structured3d_full_%A_%a.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_3090:1
#SBATCH --array=1-32%8

set -euo pipefail

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy
source /cluster/work/igp_psr/nedela/litept-env/bin/activate

REPO_ROOT="${CHORUS_REPO_ROOT:-/cluster/home/nedela/nedela/projects/chorus/chorus}"
SCANS_ROOT="${CHORUS_STRUCTURED3D_SCANS_ROOT:-/cluster/work/igp_psr/nedela/chorus_poc/scans_structured3d}"
RAW_ZIPS_DIR="${CHORUS_STRUCTURED3D_RAW_ZIPS_DIR:-/cluster/work/igp_psr/nedela/structured3d_raw}"
SCENE_LIST_DIR="${CHORUS_STRUCTURED3D_SCENE_LIST_DIR:-${REPO_ROOT}/scene_lists/structured3d_full_baseline_v1}"
REPORT_DIR="${CHORUS_STRUCTURED3D_REPORT_DIR:-${SCANS_ROOT}/_chorus_reports}"
RUN_ID="${CHORUS_STRUCTURED3D_RUN_ID:-structured3d_full_baseline_v1}"

cd "${REPO_ROOT}"
mkdir -p logs "${SCANS_ROOT}" "${REPORT_DIR}"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${TMPDIR:-/tmp}"
export CHORUS_WANDB_DIR="${WANDB_DIR}"

SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
SHARD_FILE="${SCENE_LIST_DIR}/shard_$(printf "%03d" "${SHARD_INDEX}").txt"
if [[ ! -s "${SHARD_FILE}" ]]; then
  echo "Missing or empty shard file: ${SHARD_FILE}" >&2
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
case "${GPU_NAME}" in
  *"RTX 4090"*|*"RTX 3090"*) ;;
  *)
    echo "Refusing to run on unsupported GPU: ${GPU_NAME}" >&2
    exit 43
    ;;
esac

EXTRA_ARGS=()
if [[ "${KEEP_INTERMEDIATES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-cleanup-after-success)
fi
if [[ "${OVERWRITE_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite-existing)
fi

echo "run_id=${RUN_ID}"
echo "host=$(hostname)"
echo "gpu=${GPU_NAME}"
echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "shard_file=${SHARD_FILE}"
echo "shard_scene_count=$(wc -l < "${SHARD_FILE}")"
echo "scans_root=${SCANS_ROOT}"
echo "raw_zips_dir=${RAW_ZIPS_DIR}"
echo "report_dir=${REPORT_DIR}"
echo "keep_intermediates=${KEEP_INTERMEDIATES:-0}"
echo "overwrite_existing=${OVERWRITE_EXISTING:-0}"
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader || true

python scripts/run_streaming_structured3d.py \
  --scans-root "${SCANS_ROOT}" \
  --structured3d-raw-zips-dir "${RAW_ZIPS_DIR}" \
  --scene-list-file "${SHARD_FILE}" \
  --granularities "${CHORUS_GRANULARITIES:-0.2,0.5,0.8}" \
  --frame-skip "${CHORUS_FRAME_SKIP:-1}" \
  --structured3d-eval-benchmark "${CHORUS_STRUCTURED3D_EVAL_BENCHMARK:-structured3d_full}" \
  --svd-components "${CHORUS_SVD_COMPONENTS:-32}" \
  --min-cluster-size "${CHORUS_MIN_CLUSTER_SIZE:-100}" \
  --min-samples "${CHORUS_MIN_SAMPLES:-5}" \
  --cluster-selection-epsilon "${CHORUS_CLUSTER_SELECTION_EPSILON:-0.1}" \
  --debug-first-n-frames "${CHORUS_DEBUG_FIRST_N_FRAMES:-0}" \
  --report-dir "${REPORT_DIR}" \
  --continue-on-error \
  --no-auto-download-missing \
  --no-wandb \
  "${EXTRA_ARGS[@]}"
