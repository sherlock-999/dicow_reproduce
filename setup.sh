#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup.sh                         # environment dicow-reproduce, CUDA 12.4
#   ./setup.sh my-environment cu124    # custom environment name
#   ./setup.sh my-environment cpu      # CPU-only PyTorch

ENV_NAME="${1:-dicow-reproduce}"
TORCH_VARIANT="${2:-cu124}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_COMMAND="${CONDA_EXE:-conda}"

if ! command -v "${CONDA_COMMAND}" >/dev/null 2>&1; then
    echo "Conda was not found. Install Miniconda, then run this script again." >&2
    exit 1
fi

case "${TORCH_VARIANT}" in
    cu124)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_VERSION="2.6.0+cu124"
        ;;
    cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_VERSION="2.6.0+cpu"
        ;;
    *)
        echo "Unsupported PyTorch variant: ${TORCH_VARIANT}. Use cu124 or cpu." >&2
        exit 2
        ;;
esac

"${CONDA_COMMAND}" create --yes --name "${ENV_NAME}" python=3.10 ffmpeg
"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install --upgrade pip
"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install \
    --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCH_VERSION}"
"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install --requirement "${SCRIPT_DIR}/requirements.txt"

echo
echo "Environment created successfully."
echo "Activate it with: conda activate ${ENV_NAME}"
echo "Then run: cd ${SCRIPT_DIR} && python -m pytest -q tests"
