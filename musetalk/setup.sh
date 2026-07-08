#!/usr/bin/env bash
# One-time setup for local MuseTalk (CPU). Creates a Python 3.10 venv, installs
# CPU PyTorch + the mmlab stack, clones the MuseTalk repo, and downloads weights.
#
# Prereq: `uv` (https://astral.sh). Install with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:  bash setup.sh   (run from poc/musetalk/)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/6] Python 3.10 venv"
uv venv --python 3.10 .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "==> [2/6] clone MuseTalk"
[ -d MuseTalk ] || git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

echo "==> [3/6] CPU PyTorch"
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

echo "==> [4/6] core deps (numpy pinned; tensorflow/gradio skipped - unused for inference)"
uv pip install "numpy==1.23.5" diffusers==0.30.2 accelerate==0.28.0 transformers==4.39.2 \
  huggingface_hub==0.30.2 librosa==0.11.0 einops==0.8.1 omegaconf ffmpeg-python moviepy \
  soundfile==0.12.1 opencv-python==4.9.0.80 requests imageio imageio-ffmpeg gdown

echo "==> [5/6] mmlab stack (CPU wheels)"
uv pip install pip "setuptools==69.5.1" wheel                 # pkg_resources + build backend
uv pip install mmengine "mmcv==2.1.0" \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
uv pip install chumpy==0.70 --no-build-isolation              # mmpose dep, needs pip present
uv pip install "mmdet==3.2.0" "mmpose==1.3.1" --no-build-isolation

echo "==> [6/6] download weights (~4 GB) into MuseTalk/models"
python - <<'PY'
from huggingface_hub import hf_hub_download
import urllib.request, os
M = "models"
hf = [
 ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", M),
 ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", M),
 ("stabilityai/sd-vae-ft-mse", "config.json", M+"/sd-vae"),
 ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", M+"/sd-vae"),
 ("openai/whisper-tiny", "config.json", M+"/whisper"),
 ("openai/whisper-tiny", "pytorch_model.bin", M+"/whisper"),
 ("openai/whisper-tiny", "preprocessor_config.json", M+"/whisper"),
 ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", M+"/dwpose"),
]
for repo, fn, out in hf:
    hf_hub_download(repo_id=repo, filename=fn, local_dir=out); print("ok", fn)
os.makedirs(M+"/face-parse-bisent", exist_ok=True)
urllib.request.urlretrieve(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    M+"/face-parse-bisent/resnet18-5c106cde.pth"); print("ok resnet18")
PY
# face-parsing weight lives on Google Drive (new gdown syntax):
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth

echo "==> prepare POC inputs (image + audio + config)"
cp ../../assets/sample_face.png  data/poc_face.png
cp ../../assets/sample_audio.wav data/poc_audio.wav
printf 'task_0:\n video_path: "data/poc_face.png"\n audio_path: "data/poc_audio.wav"\n' \
  > configs/inference/poc.yaml

echo "==> done. Now run:  bash ../run.sh"
