#!/bin/bash
#SBATCH --job-name=chorus_data_gen
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:24g
#SBATCH --time=48:00:00

set -euo pipefail

# 1) Load Euler modules.
module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1 eth_proxy

# 2) Activate Python environment from WORK.
source /cluster/work/igp_psr/nedela/litept-env/bin/activate

# 3) Move to repository.
cd ~/nedela/projects/chorus/chorus

# 4) Runtime safeguards for shared filesystems / cluster execution.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB__SERVICE_WAIT=120
export WANDB_DIR=${TMPDIR:-/tmp}
export CHORUS_WANDB_DIR="$WANDB_DIR"

# 5) Dataset/offline config.
# Supported: scannet | structured3d
export CHORUS_DATASET=${CHORUS_DATASET:-structured3d}

# Root containing processed scene directories (scene*/).
# Example Structured3D prepared scenes root:
#   /cluster/work/igp_psr/nedela/chorus_poc/structured3d_scans
export CHORUS_SCANS_ROOT=${CHORUS_SCANS_ROOT:-/cluster/work/igp_psr/nedela/chorus_poc/structured3d_scans}

# Optional scene list file (one scene ID per line). Leave empty to auto-discover.
export CHORUS_SCENE_LIST_FILE=${CHORUS_SCENE_LIST_FILE:-}

# Optional Structured3D raw zips location (used only if a scene needs prepare()).
export CHORUS_STRUCTURED3D_RAW_ZIPS_DIR=${CHORUS_STRUCTURED3D_RAW_ZIPS_DIR:-/cluster/work/igp_psr/nedela/structured3d_raw}
export CHORUS_REPORT_DIR_BASE=${CHORUS_REPORT_DIR_BASE:-${CHORUS_SCANS_ROOT}/_chorus_reports}

# Dataset-tuned defaults (override via env when needed).
if [[ "${CHORUS_DATASET}" == "structured3d" ]]; then
  # Dense frame sampling improves 3D coverage on Structured3D and lowers unseen fraction.
  export CHORUS_FRAME_SKIP=${CHORUS_FRAME_SKIP:-1}
  # Slightly less strict default helps avoid tiny-scene HDBSCAN crashes.
  export CHORUS_MIN_SAMPLES=${CHORUS_MIN_SAMPLES:-3}
  # Avoid sklearn HDBSCAN epsilon_search instability seen on some scenes.
  export CHORUS_CLUSTER_SELECTION_EPSILON=${CHORUS_CLUSTER_SELECTION_EPSILON:-0.0}
else
  export CHORUS_FRAME_SKIP=${CHORUS_FRAME_SKIP:-10}
  export CHORUS_MIN_SAMPLES=${CHORUS_MIN_SAMPLES:-5}
  export CHORUS_CLUSTER_SELECTION_EPSILON=${CHORUS_CLUSTER_SELECTION_EPSILON:-0.1}
fi
export CHORUS_MIN_CLUSTER_SIZE=${CHORUS_MIN_CLUSTER_SIZE:-100}

echo "Running dataset generation"
echo "  CHORUS_DATASET=${CHORUS_DATASET}"
echo "  CHORUS_SCANS_ROOT=${CHORUS_SCANS_ROOT}"
echo "  CHORUS_SCENE_LIST_FILE=${CHORUS_SCENE_LIST_FILE}"
echo "  CHORUS_REPORT_DIR_BASE=${CHORUS_REPORT_DIR_BASE}"
echo "  CHORUS_FRAME_SKIP=${CHORUS_FRAME_SKIP}"
echo "  CHORUS_MIN_SAMPLES=${CHORUS_MIN_SAMPLES}"
echo "  CHORUS_MIN_CLUSTER_SIZE=${CHORUS_MIN_CLUSTER_SIZE}"
echo "  CHORUS_CLUSTER_SELECTION_EPSILON=${CHORUS_CLUSTER_SELECTION_EPSILON}"

if [[ ! -d "${CHORUS_SCANS_ROOT}" ]]; then
  echo "ERROR: CHORUS_SCANS_ROOT does not exist: ${CHORUS_SCANS_ROOT}" >&2
  exit 1
fi

TMP_SPLIT_DIR="${TMPDIR:-/tmp}/chorus_split_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMP_SPLIT_DIR}" "${CHORUS_REPORT_DIR_BASE}"
SCENE_LIST_ALL="${TMP_SPLIT_DIR}/all_scenes.txt"
SCENE_LIST_0="${TMP_SPLIT_DIR}/scenes_gpu0.txt"
SCENE_LIST_1="${TMP_SPLIT_DIR}/scenes_gpu1.txt"

if [[ -n "${CHORUS_SCENE_LIST_FILE}" ]]; then
  if [[ ! -f "${CHORUS_SCENE_LIST_FILE}" ]]; then
    echo "ERROR: CHORUS_SCENE_LIST_FILE not found: ${CHORUS_SCENE_LIST_FILE}" >&2
    exit 1
  fi
  awk 'NF {print $1}' "${CHORUS_SCENE_LIST_FILE}" > "${SCENE_LIST_ALL}"
else
  shopt -s nullglob
  for d in "${CHORUS_SCANS_ROOT}"/scene*; do
    [[ -d "${d}" ]] && basename "${d}"
  done | sort > "${SCENE_LIST_ALL}"
  shopt -u nullglob
fi

NUM_SCENES=$(wc -l < "${SCENE_LIST_ALL}" | tr -d ' ')
if [[ "${NUM_SCENES}" -eq 0 ]]; then
  echo "ERROR: No scenes found to process." >&2
  exit 1
fi

HALF=$(( (NUM_SCENES + 1) / 2 ))
head -n "${HALF}" "${SCENE_LIST_ALL}" > "${SCENE_LIST_0}"
tail -n +"$((HALF + 1))" "${SCENE_LIST_ALL}" > "${SCENE_LIST_1}"

echo "Total scenes: ${NUM_SCENES}"
echo "GPU0 scenes: $(wc -l < "${SCENE_LIST_0}" | tr -d ' ')"
echo "GPU1 scenes: $(wc -l < "${SCENE_LIST_1}" | tr -d ' ')"

BASE_CMD=(
  python -u scripts/run_dataset_gen.py
  --dataset "${CHORUS_DATASET}"
  --scans-root "${CHORUS_SCANS_ROOT}"
  --structured3d-raw-zips-dir "${CHORUS_STRUCTURED3D_RAW_ZIPS_DIR}"
  --frame-skip "${CHORUS_FRAME_SKIP}"
  --min-samples "${CHORUS_MIN_SAMPLES}"
  --min-cluster-size "${CHORUS_MIN_CLUSTER_SIZE}"
  --cluster-selection-epsilon "${CHORUS_CLUSTER_SELECTION_EPSILON}"
  --wandb-mode offline
)

CUDA_VISIBLE_DEVICES=0 CHORUS_WANDB_DIR="${WANDB_DIR}/gpu0" \
  "${BASE_CMD[@]}" \
  --device cuda:0 \
  --scene-list-file "${SCENE_LIST_0}" \
  --report-dir "${CHORUS_REPORT_DIR_BASE}/gpu0" &
PID0=$!

PID1=""
if [[ -s "${SCENE_LIST_1}" ]]; then
  CUDA_VISIBLE_DEVICES=1 CHORUS_WANDB_DIR="${WANDB_DIR}/gpu1" \
    "${BASE_CMD[@]}" \
    --device cuda:0 \
    --scene-list-file "${SCENE_LIST_1}" \
    --report-dir "${CHORUS_REPORT_DIR_BASE}/gpu1" &
  PID1=$!
fi

set +e
wait "${PID0}"
RC0=$?
RC1=0
if [[ -n "${PID1}" ]]; then
  wait "${PID1}"
  RC1=$?
fi
set -e

if [[ "${RC0}" -ne 0 || "${RC1}" -ne 0 ]]; then
  echo "ERROR: One or more workers failed (gpu0=${RC0}, gpu1=${RC1})." >&2
  exit 1
fi

echo "Both workers completed successfully."
