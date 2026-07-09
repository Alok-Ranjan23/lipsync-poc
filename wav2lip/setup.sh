#!/usr/bin/env bash
# CPU setup for the Wav2Lip ONNX pipeline.  (GPU variant: ./setup_gpu.sh)
#
# Uses uv + Python 3.11 (robust on boxes that only ship Python 3.13/3.14, where
# some wheels like numba/librosa can be missing). Creates the repo-root .venv,
# installs CPU deps, downloads the ONNX weights.
#
# Usage (from lipsync-poc/wav2lip):  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (lipsync-poc)

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

uv pip install -r requirements.txt          # includes CPU onnxruntime
python wav2lip/download_models.py

echo "==> CPU setup complete. Run:  bash wav2lip/run.sh"
