#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup.sh
#   ./setup.sh my-environment cu124
#   ./setup.sh my-environment cpu /path/to/whisper-large-v3-turbo
#
# Positional arguments:
#   1. Conda environment name (default: dicow-reproduce)
#   2. PyTorch build: cu124 or cpu (default: cu124)
#   3. Local checkpoint directory (default: <repository>/whisper-large-v3-turbo)

ENV_NAME="${1:-dicow-reproduce}"
TORCH_VARIANT="${2:-cu124}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="openai/whisper-large-v3-turbo"
CHECKPOINT_DIR="${3:-${SCRIPT_DIR}/whisper-large-v3-turbo}"
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

ENV_PREFIX="$("${CONDA_COMMAND}" env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}')"
if [[ -n "${ENV_PREFIX}" ]]; then
    if ! "${CONDA_COMMAND}" run --name "${ENV_NAME}" \
        python -c "import sys; assert sys.version_info[:2] == (3, 10)" \
        >/dev/null 2>&1; then
        echo "Existing environment ${ENV_NAME} does not use Python 3.10." >&2
        echo "Choose another environment name or remove that environment first." >&2
        exit 3
    fi
    echo "Using existing Conda environment: ${ENV_NAME}"
    "${CONDA_COMMAND}" install --yes --name "${ENV_NAME}" python=3.10 ffmpeg
else
    "${CONDA_COMMAND}" create --yes --name "${ENV_NAME}" python=3.10 ffmpeg
fi

"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install --upgrade pip
"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install \
    --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCH_VERSION}"
"${CONDA_COMMAND}" run --name "${ENV_NAME}" \
    python -m pip install --requirement "${SCRIPT_DIR}/requirements.txt"

echo "Downloading ${MODEL_ID} to ${CHECKPOINT_DIR} ..."
"${CONDA_COMMAND}" run --no-capture-output --name "${ENV_NAME}" \
    python - "${MODEL_ID}" "${CHECKPOINT_DIR}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
checkpoint_dir = Path(sys.argv[2]).expanduser().resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# snapshot_download verifies cached files and resumes an interrupted download.
snapshot_download(repo_id=model_id, local_dir=checkpoint_dir)

required_files = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
)
missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
has_weights = any(checkpoint_dir.glob("*.safetensors")) or any(
    checkpoint_dir.glob("pytorch_model*.bin")
)
if missing or not has_weights:
    details = ", ".join(missing) if missing else "model weights"
    raise RuntimeError(f"Checkpoint download is incomplete; missing: {details}")

print(f"Checkpoint ready: {checkpoint_dir}")
PY

"${CONDA_COMMAND}" run --name "${ENV_NAME}" python -c \
    "import accelerate, lhotse, meeteval, torch, torchaudio, transformers"

echo
echo "Setup completed successfully."
echo "Environment: ${ENV_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Activate it with: conda activate ${ENV_NAME}"
echo "Verify it with: cd ${SCRIPT_DIR} && python -m pytest -q tests"
