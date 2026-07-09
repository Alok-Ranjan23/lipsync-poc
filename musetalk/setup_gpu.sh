#!/usr/bin/env bash
# Set up MuseTalk 1.5 to run on an NVIDIA GPU.
#
# Target box: 2x RTX 5090 (Blackwell, sm_120), driver CUDA 13.2.
#
# ⚠️ HONEST WARNING: MuseTalk depends on mmcv/mmpose, which must be compiled against
# the exact PyTorch build. On Blackwell we need cu128 torch (>=2.7), and there is NO
# prebuilt mmcv wheel for that combo -> mmcv builds from source and needs the CUDA
# TOOLKIT (nvcc) present, not just the driver. This may fail on a driver-only box.
# If it does, either: (a) run MuseTalk on CPU via ./setup.sh (works, ~14 min), or
# (b) use LatentSync on this GPU (../latentsync), which needs no mmcv.
#
# Usage (from lipsync-poc/musetalk):
#   CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"

: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES
echo "==> using GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "==> [1/6] clone MuseTalk"
[ -d MuseTalk ] || git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

echo "==> [2/6] Python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip setuptools==69.5.1 wheel

echo "==> [3/6] Blackwell-capable PyTorch (cu128)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> [4/6] inference deps"
pip install "numpy<2" diffusers==0.30.2 accelerate transformers==4.39.2 \
  "huggingface_hub<1.0" librosa einops omegaconf ffmpeg-python moviepy soundfile \
  opencv-python==4.9.0.80 requests imageio imageio-ffmpeg gdown

echo "==> [5/6] mmlab (mmcv likely builds from source on Blackwell - needs nvcc/CUDA toolkit)"
pip install -U openmim
pip install chumpy==0.70 --no-build-isolation || true
# Let mim resolve an mmcv compatible with the installed torch; falls back to source build.
mim install mmengine "mmcv>=2.1.0,<2.2.0" "mmdet==3.2.0" "mmpose==1.3.1" || {
  echo "!! mmcv/mmpose install FAILED (expected on a driver-only Blackwell box)."
  echo "   Options: run MuseTalk on CPU (bash ../setup.sh) or use ../latentsync on the GPU."
  exit 1
}

echo "==> [6/6] download weights (~4 GB)"
python - <<'PY'
from huggingface_hub import hf_hub_download
import urllib.request, os
M="models"
for r,f,o in [("TMElyralab/MuseTalk","musetalkV15/musetalk.json",M),
              ("TMElyralab/MuseTalk","musetalkV15/unet.pth",M),
              ("stabilityai/sd-vae-ft-mse","config.json",M+"/sd-vae"),
              ("stabilityai/sd-vae-ft-mse","diffusion_pytorch_model.bin",M+"/sd-vae"),
              ("openai/whisper-tiny","config.json",M+"/whisper"),
              ("openai/whisper-tiny","pytorch_model.bin",M+"/whisper"),
              ("openai/whisper-tiny","preprocessor_config.json",M+"/whisper"),
              ("yzd-v/DWPose","dw-ll_ucoco_384.pth",M+"/dwpose")]:
    hf_hub_download(repo_id=r, filename=f, local_dir=o)
os.makedirs(M+"/face-parse-bisent", exist_ok=True)
urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-5c106cde.pth",
    M+"/face-parse-bisent/resnet18-5c106cde.pth")
print("weights (except face-parse gdrive) ready")
PY
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth

cp ../../assets/sample_face.png  data/poc_face.png
cp ../../assets/sample_audio.wav data/poc_audio.wav
printf 'task_0:\n video_path: "data/poc_face.png"\n audio_path: "data/poc_audio.wav"\n' \
  > configs/inference/poc.yaml
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "==> done. Run:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ../run.sh"
