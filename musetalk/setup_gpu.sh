#!/usr/bin/env bash
# GPU setup for MuseTalk 1.5.  (CPU variant: ./setup.sh)
#
# Creates the venv at musetalk/.venv (the SAME place run.sh looks), so the one
# shared run.sh works whether you set up CPU (setup.sh) or GPU (this script).
# Uses uv + Python 3.11 (GPU boxes often ship only 3.13/3.14, which break wheels).
#
# ⚠️ mmcv is the risk on Blackwell (RTX 5090, sm_120): it needs cu128 torch (>=2.7),
# and there's no prebuilt mmcv wheel for that -> mmcv builds from SOURCE, which needs
# the CUDA TOOLKIT (nvcc), not just the driver. Check first:  nvcc --version
# If mmcv fails, use the CPU path instead:  bash setup.sh   (then bash run.sh)
#
# Usage (from lipsync-poc/musetalk):
#   CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES
echo "==> using GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "==> [1/6] ensure uv + Python 3.11, create venv at musetalk/.venv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "==> [2/6] clone MuseTalk"
[ -d MuseTalk ] || git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

echo "==> [3/6] Blackwell-capable PyTorch (cu128)"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> [4/6] core deps"
uv pip install "numpy<2" diffusers==0.30.2 accelerate transformers==4.39.2 \
  "huggingface_hub<1.0" librosa einops omegaconf ffmpeg-python moviepy soundfile \
  opencv-python==4.9.0.80 requests imageio imageio-ffmpeg gdown pip "setuptools==69.5.1" wheel

echo "==> [5/6] mmlab (mmcv builds from source on Blackwell/cu128)"
uv pip install -U openmim ninja
uv pip install mmengine
uv pip install chumpy==0.70 --no-build-isolation || true
# --no-build-isolation so mmcv's setup.py uses our setuptools 69.5.1 (has pkg_resources).
# The compile step needs the CUDA toolkit (nvcc); if absent it fails here.
uv pip install "mmcv==2.1.0" --no-build-isolation || {
  echo "!! mmcv build FAILED. If it's a compiler/nvcc error, the box lacks the CUDA toolkit"
  echo "   needed to compile mmcv's CUDA ops. Fall back to CPU:  bash setup.sh"
  exit 1
}
uv pip install "mmdet==3.2.0" "mmpose==1.3.1" --no-build-isolation

echo "==> [6/6] download weights (~4 GB) + prepare POC inputs"
python - <<'PY'
from huggingface_hub import hf_hub_download
import urllib.request, os
M = "models"
for repo, fn, out in [
 ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", M),
 ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", M),
 ("stabilityai/sd-vae-ft-mse", "config.json", M+"/sd-vae"),
 ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", M+"/sd-vae"),
 ("openai/whisper-tiny", "config.json", M+"/whisper"),
 ("openai/whisper-tiny", "pytorch_model.bin", M+"/whisper"),
 ("openai/whisper-tiny", "preprocessor_config.json", M+"/whisper"),
 ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", M+"/dwpose"),
]:
    hf_hub_download(repo_id=repo, filename=fn, local_dir=out)
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "==> done. Run:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ../run.sh"
