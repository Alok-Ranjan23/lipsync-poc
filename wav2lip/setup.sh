#!/usr/bin/env bash
# CPU setup for the Wav2Lip ONNX pipeline.  (GPU variant: ./setup_gpu.sh)
#
# Uses uv + Python 3.12 (requirements.txt pins scipy==1.18.0 which needs >=3.12;
# also avoids missing wheels on boxes that only ship 3.13/3.14). Creates the
# repo-root .venv, installs CPU deps, downloads the ONNX weights.
#
# Usage (from lipsync-poc/wav2lip):  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (lipsync-poc)
VENV="${WAV2LIP_VENV:-.venv}"   # selectable so CPU + GPU can coexist

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

uv pip install -r requirements.txt          # includes CPU onnxruntime
python wav2lip/download_models.py

echo "==> CPU setup complete. Run:  bash wav2lip/run.sh"
