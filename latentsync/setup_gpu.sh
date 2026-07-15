#!/usr/bin/env bash
# Set up LatentSync 1.6 to run on an NVIDIA GPU.
#
# Uses `uv` to get Python 3.11 (GPU boxes often ship only Python 3.13/3.14, which
# lack stable wheels for onnx/numba/etc. and cause dependency whack-a-mole).
#
# Target box: 2x RTX 5090 (Blackwell, sm_120), driver CUDA 13.2.
# Blackwell needs PyTorch >= 2.7 built for CUDA 12.8 (cu128); older torch has no
# kernels for sm_120 and fails with "no kernel image is available".
#
# Usage (from lipsync-poc/latentsync):
#   CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh     # GPU 1 is the free one on this box
set -euo pipefail
cd "$(dirname "$0")"

: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES
echo "==> using GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
TORCHAUDIO_VERSION=2.7.1
NUMPY_VERSION=1.26.4
OPENCV_VERSION=4.9.0.80

echo "==> [1/6] ensure uv + Python 3.11"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

echo "==> [2/6] clone LatentSync"
[ -d LatentSync ] || git clone https://github.com/bytedance/LatentSync.git
cd LatentSync

echo "==> [3/6] Python 3.11 venv (via uv)"
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "==> [4/6] matched Blackwell-capable PyTorch stack (cu128)"
uv pip install --reinstall \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128

command -v ffmpeg >/dev/null || echo "!! ffmpeg not found - install it: sudo apt-get install -y ffmpeg libgl1"

echo "==> [5/6] repo deps (strip torch pins so cu128 torch stays; relax mediapipe) + known fixes"
sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d; s/mediapipe==[0-9.]*/mediapipe/' requirements.txt
uv pip install -r requirements.txt
# accelerate -> diffusers clash; huggingface_hub<1.0 for transformers/tokenizers;
# ml_dtypes>=0.5 for the onnx pulled in by insightface. Pin NumPy 1.x because
# OpenCV 4.9's binary wheel cannot import against NumPy 2.x.
uv pip install -U accelerate "huggingface_hub>=0.24,<1.0" "ml_dtypes>=0.5.0"
uv pip install --reinstall "numpy==$NUMPY_VERSION" "opencv-python==$OPENCV_VERSION"

echo "==> [6/6] download LatentSync 1.6 checkpoints"
python -c "from huggingface_hub import snapshot_download; snapshot_download('ByteDance/LatentSync-1.6', local_dir='checkpoints', allow_patterns=['latentsync_unet.pt', 'whisper/tiny.pt', 'stable_syncnet.pt']); print('checkpoints ready')"

python -c "import cv2, numpy, torch, torchvision; assert numpy.__version__ == '$NUMPY_VERSION', numpy.__version__; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| opencv', cv2.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"
echo "==> done. Now run:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ../run.sh"
