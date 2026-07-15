#!/usr/bin/env bash
# Set up EchoMimicV3 Flash on a CUDA GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail

case "$0" in
  */*) cd "${0%/*}" ;;
esac

: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES
: "${UV_HTTP_TIMEOUT:=300}"
: "${UV_HTTP_RETRIES:=5}"
export UV_HTTP_TIMEOUT
export UV_HTTP_RETRIES

TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
TORCHAUDIO_VERSION=2.7.1
NUMPY_VERSION=1.26.4
OPENCV_VERSION=4.9.0.80
DIFFUSERS_VERSION=0.32.2
TRANSFORMERS_VERSION=4.48.0
ACCELERATE_VERSION=0.26.1
WAN_MODEL=Wan2.1-Fun-V1.1-1.3B-InP
FLASH_MODEL=echomimicv3-flash-pro

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv is required; install it before running this script" >&2
  exit 1
fi
uv python install 3.11

if [ ! -d EchoMimicV3 ]; then
  git clone https://github.com/antgroup/echomimic_v3.git EchoMimicV3
fi

uv venv --clear --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

uv pip install --reinstall \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128

cd EchoMimicV3
uv pip install -r requirements.txt pyloudnorm
uv pip install --reinstall \
  "diffusers==$DIFFUSERS_VERSION" \
  "transformers==$TRANSFORMERS_VERSION" \
  "accelerate==$ACCELERATE_VERSION"
uv pip install --reinstall \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install modelscope
uv pip install --reinstall "numpy==$NUMPY_VERSION" "opencv-python==$OPENCV_VERSION"

mkdir -p ../weights
python ../download_models.py
modelscope download --model TencentGameMate/chinese-wav2vec2-base --local_dir ../weights/chinese-wav2vec2-base

python ../verify_env.py
echo "==> done. Run: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash run_image_audio.sh <image> <audio> <output.mp4>"
