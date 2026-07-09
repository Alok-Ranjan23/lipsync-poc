#!/usr/bin/env bash
# GPU setup for the Wav2Lip ONNX pipeline.  (CPU variant: ./setup.sh)
#
# Installs onnxruntime-gpu + cuDNN 9 (no sudo). providers.py then auto-uses CUDA.
# run.sh sets LD_LIBRARY_PATH so onnxruntime can find libcudnn.so.9.
#
# Note: Wav2Lip is a tiny 96x96 model, so CPU is already only seconds - the GPU
# gain is marginal. Also, onnxruntime-gpu may lack Blackwell (sm_120) kernels; if
# so it falls back to CPU. The heavy GPU wins are LatentSync / MuseTalk.
#
# Usage (from lipsync-poc/wav2lip):  CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (lipsync-poc)

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip
# deps WITHOUT the CPU onnxruntime (it would clash with onnxruntime-gpu)
pip install opencv-python-headless numpy scipy librosa soundfile imageio-ffmpeg huggingface_hub tqdm
pip install onnxruntime-gpu nvidia-cudnn-cu12
python wav2lip/download_models.py

echo "==> GPU setup complete. Run:  CUDA_VISIBLE_DEVICES=1 bash wav2lip/run.sh"
