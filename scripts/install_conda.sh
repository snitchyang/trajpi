#!/usr/bin/env bash
# Conda-based install for openpi (alternative to `uv sync`).
# Prerequisites: conda (Miniconda/Mambaforge), NVIDIA driver for JAX/torch CUDA wheels.
#
# Usage:
#   bash scripts/install_conda.sh              # create env + pip install
#   bash scripts/install_conda.sh --skip-create # conda env already active / exists; only pip
#   CONDA_ENV_NAME=myopenpi bash scripts/install_conda.sh
#   bash scripts/install_conda.sh --with-dev --with-rlds
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_NAME="${CONDA_ENV_NAME:-openpi}"
SKIP_CREATE=0
WITH_DEV=0
WITH_RLDS=0
for arg in "$@"; do
  case "${arg}" in
    --skip-create) SKIP_CREATE=1 ;;
    --with-dev) WITH_DEV=1 ;;
    --with-rlds) WITH_RLDS=1 ;;
    -h|--help)
      echo "Usage: bash scripts/install_conda.sh [--skip-create] [--with-dev] [--with-rlds]"
      exit 0
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found."
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${SKIP_CREATE}" -eq 0 ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' exists; updating from environment.yml (conda only)..."
    conda env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml" --prune
  else
    echo "Creating conda env '${ENV_NAME}' from environment.yml..."
    conda env create -n "${ENV_NAME}" -f "${ROOT}/environment.yml"
  fi
else
  echo "Skipping conda create/update (--skip-create)."
fi

conda activate "${ENV_NAME}"

python -c "import sys; assert sys.version_info >= (3, 11), 'Need Python >= 3.11'"

export GIT_LFS_SKIP_SMUDGE=1

pip install --upgrade pip setuptools wheel hatchling

echo "Installing workspace package openpi-client (editable)..."
pip install -e "${ROOT}/packages/openpi-client"

echo "Installing locked pip dependencies (see requirements-conda.txt)..."
pip install -r "${ROOT}/requirements-conda.txt"

if [[ "${WITH_DEV}" -eq 1 ]]; then
  echo "Installing dev dependency-group..."
  pip install \
    "pytest>=8.3.4" "ruff>=0.8.6" "pre-commit>=4.0.1" "ipykernel>=6.29.5" \
    "ipywidgets>=8.1.5" "matplotlib>=3.10.0" "pynvml>=12.0.0"
fi

if [[ "${WITH_RLDS}" -eq 1 ]]; then
  echo "Installing RLDS dependency-group..."
  pip install -r "${ROOT}/requirements-conda-rlds.txt"
fi

echo "Installing openpi (editable root)..."
pip install -e "${ROOT}"

echo ""
echo "Done. Activate with: conda activate ${ENV_NAME}"
