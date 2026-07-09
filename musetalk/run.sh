#!/usr/bin/env bash
# Run local MuseTalk lip-sync on the POC sample (or your own image/video + audio).
# Usage:  bash run.sh          (run from anywhere)
set -euo pipefail
cd "$(dirname "$0")"

# New PyTorch (>=2.6) defaults torch.load to weights_only=True, which refuses to
# unpickle MuseTalk/dwpose checkpoints (they contain numpy objects). These are the
# official checkpoints, so flip the default back for the whole run.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Call the venv's Python by absolute path (robust even if another venv is active).
VENV_PY="$PWD/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "!! venv missing - run setup.sh first"; exit 1; }
cd MuseTalk

# Make ffmpeg available via the bundled imageio-ffmpeg binary (no system install needed).
FF="$("$VENV_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p .ffbin && ln -sf "$FF" .ffbin/ffmpeg
export PATH="$PWD/.ffbin:$PATH"
export FFMPEG_PATH="$PWD/.ffbin"

OUT="results/poc/v15/poc_face_poc_audio.mp4"
rm -f "$OUT"   # avoid reporting a stale previous result as success

# Report device (works for both the CPU and GPU venvs; GPU is used automatically
# whenever the installed torch has CUDA).
"$VENV_PY" -c "import torch; print('[device]', 'CUDA' if torch.cuda.is_available() else 'CPU', '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "==> running MuseTalk (CPU ~14 min for a 6s clip; near real-time on GPU)"
# The repo throws a harmless 'save_dir_full' error AFTER the mp4 is written for
# still-image input, so we don't abort on it.
_start=$(date +%s)
"$VENV_PY" -m scripts.inference \
  --inference_config configs/inference/poc.yaml \
  --result_dir results/poc \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config  models/musetalkV15/musetalk.json \
  --version v15 || true

if [ -f "$OUT" ]; then
  cp "$OUT" ../../outputs/musetalk_demo.mp4
  echo "==> done -> outputs/musetalk_demo.mp4   [TIME] end-to-end: $(( $(date +%s) - _start ))s"
else
  echo "!! output not produced - check the log above"; exit 1
fi
