#!/usr/bin/env bash
# Run local MuseTalk lip-sync on the POC sample (or your own image/video + audio).
# Usage:  bash run.sh          (run from anywhere)
set -euo pipefail
cd "$(dirname "$0")"

# New PyTorch (>=2.6) defaults torch.load to weights_only=True, which refuses to
# unpickle MuseTalk/dwpose checkpoints (they contain numpy objects). These are the
# official checkpoints, so flip the default back for the whole run.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# venv is selectable so CPU (.venv-cpu) and GPU (.venv) can coexist:
#   bash run.sh                         # uses .venv     (GPU, from setup_gpu.sh)
#   MUSETALK_VENV=.venv-cpu bash run.sh # uses .venv-cpu (CPU, from setup.sh)
VENV="${MUSETALK_VENV:-.venv}"
VENV_PY="$PWD/$VENV/bin/python"
[ -x "$VENV_PY" ] || { echo "!! venv '$VENV' missing - run setup.sh (CPU) or setup_gpu.sh (GPU) first"; exit 1; }
CONFIG="${MUSETALK_CONFIG:-configs/inference/poc.yaml}"
case "$CONFIG" in
  /*) ;;
  *) CONFIG="$PWD/$CONFIG" ;;
esac
if [ ! -f "$CONFIG" ]; then
  echo "!! inference config not found: $CONFIG" >&2
  exit 1
fi
cd MuseTalk

# Make ffmpeg available via the bundled imageio-ffmpeg binary (no system install needed).
FF="$("$VENV_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p .ffbin && ln -sf "$FF" .ffbin/ffmpeg
export PATH="$PWD/.ffbin:$PATH"
export FFMPEG_PATH="$PWD/.ffbin"

OUT="${MUSETALK_OUTPUT:-results/poc/v15/poc_face_poc_audio.mp4}"
case "$OUT" in
  /*) ;;
  *) OUT="$PWD/$OUT" ;;
esac
rm -f "$OUT"   # avoid reporting a stale previous result as success

# Report device (works for both the CPU and GPU venvs; GPU is used automatically
# whenever the installed torch has CUDA).
"$VENV_PY" -c "import torch; print('[device]', 'CUDA' if torch.cuda.is_available() else 'CPU', '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "==> running MuseTalk (CPU: ~1-15 min depending on the machine; near real-time on GPU)"
# The repo throws a harmless 'save_dir_full' error AFTER the mp4 is written for
# still-image input, so we don't abort on it.
_start=$(date +%s)
"$VENV_PY" -m scripts.inference \
  --inference_config "$CONFIG" \
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
