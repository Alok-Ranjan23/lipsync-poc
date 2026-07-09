#!/usr/bin/env bash
# Set up LatentSync 1.6 to run on an NVIDIA GPU.
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

echo "==> [1/5] clone LatentSync"
[ -d LatentSync ] || git clone https://github.com/bytedance/LatentSync.git
cd LatentSync

echo "==> [2/5] Python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip

echo "==> [3/5] Blackwell-capable PyTorch (cu128). For an older GPU, change the cu index."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

command -v ffmpeg >/dev/null || echo "!! ffmpeg not found - install it: sudo apt-get install -y ffmpeg libgl1"

echo "==> [4/5] repo deps (strip their torch pins so cu128 torch stays; relax mediapipe)"
sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d; s/mediapipe==[0-9.]*/mediapipe/' requirements.txt
pip install -r requirements.txt
# accelerate fixes the diffusers<->accelerate clash; keep huggingface_hub <1.0 so
# transformers/tokenizers stay happy (hf_hub 1.x drops APIs they need).
pip install -U accelerate "huggingface_hub>=0.24,<1.0"

echo "==> [5/5] download LatentSync 1.6 checkpoints"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('ByteDance/LatentSync-1.6', local_dir='checkpoints',
    allow_patterns=['latentsync_unet.pt', 'whisper/tiny.pt', 'stable_syncnet.pt'])
print("checkpoints ready")
PY

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "==> done. Now run:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ../run.sh"
