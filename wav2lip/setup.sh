#!/usr/bin/env bash
# CPU setup for the Wav2Lip ONNX pipeline.  (GPU variant: ./setup_gpu.sh)
# Creates the repo-root .venv, installs CPU deps, downloads ONNX weights.
#
# Usage (from lipsync-poc/wav2lip):  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (lipsync-poc)

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # includes CPU onnxruntime
python wav2lip/download_models.py

echo "==> CPU setup complete. Run:  bash wav2lip/run.sh"
