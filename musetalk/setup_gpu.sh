#!/usr/bin/env bash
# One-command GPU setup for MuseTalk 1.5.  After it finishes:
#     CUDA_VISIBLE_DEVICES=1 bash run.sh
#
# Tested target: 2x RTX 5090 (Blackwell / sm_120), CUDA driver 13.2, CUDA toolkit 12.4,
# g++ 15, box ships only Python 3.13/3.14.
#
# It automates every snag hit on such a box:
#   - only Python 3.13/3.14 available   -> uv installs Python 3.11 (stable wheels)
#   - Blackwell (sm_120)                -> cu128 PyTorch (>=2.7)
#   - no prebuilt mmcv wheel for cu128  -> build mmcv from source, CPU-ops only
#       (dodges the CUDA host-compiler gate: nvcc for CUDA 12.4 needs g++<14, box has g++15)
#   - dwpose forced to CPU (mmcv CPU ops); MuseTalk UNet + VAE still run on the GPU
#   - version pins: numpy<2, huggingface_hub<1.0, setuptools 69.5.1 (pkg_resources for the build)
#
# Creates the venv at musetalk/.venv (same place run.sh looks) so run.sh works CPU or GPU.
#
# Usage (from lipsync-poc/musetalk):
#   CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES
echo "==> GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "==> [1/7] uv + Python 3.11 venv (musetalk/.venv)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "==> [2/7] clone MuseTalk"
[ -d MuseTalk ] || git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

echo "==> [3/7] PyTorch cu128 (Blackwell)"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> [4/7] core deps (numpy<2, huggingface_hub<1.0, build tools)"
uv pip install "numpy<2" diffusers==0.30.2 accelerate transformers==4.39.2 \
  "huggingface_hub<1.0" librosa einops omegaconf ffmpeg-python moviepy soundfile \
  opencv-python==4.9.0.80 requests imageio imageio-ffmpeg gdown \
  pip "setuptools==69.5.1" wheel ninja

echo "==> [5/7] mmlab: build mmcv from source, CPU-ops only (avoids the CUDA host-compiler gate)"
uv pip install mmengine
uv pip install chumpy==0.70 --no-build-isolation || true
uv pip install "numpy<2"     # re-assert: earlier steps can pull numpy 2.x, which breaks the cv2 import in mmcv's setup.py
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 uv pip install "mmcv==2.1.0" --no-build-isolation
uv pip install "mmdet==3.2.0" "mmpose==1.3.1" --no-build-isolation

echo "==> [6/7] force dwpose landmark step onto CPU (mmcv is CPU-ops; UNet/VAE stay on GPU)"
sed -i 's/torch.device("cuda" if torch.cuda.is_available() else "cpu")/torch.device("cpu")/' \
  musetalk/utils/preprocessing.py || true

echo "==> [7/7] weights (~4 GB) + POC inputs + config"
python - <<'PY'
from huggingface_hub import hf_hub_download
import urllib.request, os
M = "models"
for r, f, o in [("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", M),
                ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", M),
                ("stabilityai/sd-vae-ft-mse", "config.json", M+"/sd-vae"),
                ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", M+"/sd-vae"),
                ("openai/whisper-tiny", "config.json", M+"/whisper"),
                ("openai/whisper-tiny", "pytorch_model.bin", M+"/whisper"),
                ("openai/whisper-tiny", "preprocessor_config.json", M+"/whisper"),
                ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", M+"/dwpose")]:
    hf_hub_download(repo_id=r, filename=f, local_dir=o)
os.makedirs(M+"/face-parse-bisent", exist_ok=True)
urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-5c106cde.pth",
                           M+"/face-parse-bisent/resnet18-5c106cde.pth")
print("weights ready")
PY
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
cp ../../assets/sample_face.png  data/poc_face.png
cp ../../assets/sample_audio.wav data/poc_audio.wav
printf 'task_0:\n video_path: "data/poc_face.png"\n audio_path: "data/poc_audio.wav"\n' \
  > configs/inference/poc.yaml

python -c "import torch, mmpose; print('OK | torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| mmpose', mmpose.__version__)"
echo "==> setup complete. Run:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ../run.sh"
