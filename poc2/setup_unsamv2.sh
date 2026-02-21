#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch2/${USER}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$SCRATCH_ROOT/chorus_poc2}"

# Keep large artifacts in scratch.
VENV_DIR="$RUNTIME_ROOT/envs/unsamv2"
UV_CACHE_DIR="$RUNTIME_ROOT/caches/uv"
PIP_CACHE_DIR="$RUNTIME_ROOT/caches/pip"
HF_HOME="$RUNTIME_ROOT/caches/huggingface"
TORCH_HOME="$RUNTIME_ROOT/caches/torch"
XDG_CACHE_HOME="$RUNTIME_ROOT/caches/xdg"
UNSAM_CKPT_DIR="$RUNTIME_ROOT/checkpoints/unsamv2"

mkdir -p "$VENV_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$UNSAM_CKPT_DIR"

export UV_CACHE_DIR PIP_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME

# Keep repo checkout in project folder unless user overrides.
UNSAM_DIR="${UNSAM_DIR:-$ROOT_DIR/UnSAMv2}"
if [[ ! -d "$UNSAM_DIR" ]]; then
  git clone https://github.com/yujunwei04/UnSAMv2.git "$UNSAM_DIR"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_PREFIX="$RUNTIME_ROOT/envs/conda_unsamv2"
  echo "[setup] Using conda env at: $CONDA_ENV_PREFIX"
  eval "$(conda shell.bash hook)"
  conda create --prefix "$CONDA_ENV_PREFIX" python=3.10 -y || true
  conda activate "$CONDA_ENV_PREFIX"

  python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

  python -m pip install -r "$UNSAM_DIR/requirements.txt"
  python -m pip install -r "$ROOT_DIR/requirements_poc2.txt"

  pushd "$UNSAM_DIR/sam2" >/dev/null
  python -m pip install -e ".[notebooks]"
  popd >/dev/null

  echo "[setup] Done. Activate later with: conda activate $CONDA_ENV_PREFIX"
else
  echo "[setup] conda not found. Using uv venv at: $VENV_DIR"
  command -v uv >/dev/null 2>&1 || { echo "uv is required when conda is unavailable."; exit 1; }

  uv venv --python 3.10 "$VENV_DIR" || uv venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  VENV_PY="$VENV_DIR/bin/python"

  uv pip install --python "$VENV_PY" torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

  uv pip install --python "$VENV_PY" -r "$UNSAM_DIR/requirements.txt"
  uv pip install --python "$VENV_PY" -r "$ROOT_DIR/requirements_poc2.txt"

  pushd "$UNSAM_DIR/sam2" >/dev/null
  uv pip install --python "$VENV_PY" -e ".[notebooks]"
  popd >/dev/null

  echo "[setup] Done. Activate later with: source $VENV_DIR/bin/activate"
fi

echo "[setup] Place UnSAMv2 checkpoint at: $UNSAM_CKPT_DIR/unsamv2_plus_ckpt.pt"
echo "[setup] Then export: UNSAMV2_CKPT=$UNSAM_CKPT_DIR/unsamv2_plus_ckpt.pt"
