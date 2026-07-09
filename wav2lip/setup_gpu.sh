#!/usr/bin/env bash
# GPU setup for the Wav2Lip ONNX pipeline.  (CPU variant: ./setup.sh)
#
# Uses uv + Python 3.11. Installs onnxruntime-gpu + cuDNN 9 (no sudo); providers.py
# then auto-uses CUDA and run.sh sets LD_LIBRARY_PATH so it finds libcudnn.so.9.
#
# Note: Wav2Lip is a tiny 96x96 model, so CPU is already only seconds - the GPU gain
# is marginal, and onnxruntime-gpu may lack Blackwell (sm_120) kernels (then it falls
# back to CPU). The real GPU wins are LatentSync / MuseTalk.
#
# Usage (from lipsync-poc/wav2lip):  CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (lipsync-poc)
VENV="${WAV2LIP_VENV:-.venv-gpu}"   # default a separate venv so the CPU .venv is kept

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

# deps WITHOUT the CPU onnxruntime (it would clash with onnxruntime-gpu)
uv pip install opencv-python-headless numpy scipy librosa soundfile imageio-ffmpeg huggingface_hub tqdm
uv pip install onnxruntime-gpu nvidia-cudnn-cu12
python wav2lip/download_models.py

echo "==> GPU setup complete (venv: $VENV). Run:  WAV2LIP_VENV=$VENV CUDA_VISIBLE_DEVICES=1 bash wav2lip/run.sh"
