#!/usr/bin/env bash
# Run local MuseTalk lip-sync on the POC sample (or your own image/video + audio).
# Usage:  bash run.sh          (run from anywhere)
set -euo pipefail
cd "$(dirname "$0")"

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

echo "==> running MuseTalk (CPU ~14 min for a 6s clip; auto-uses GPU if CUDA torch installed)"
# The repo throws a harmless 'save_dir_full' error AFTER the mp4 is written for
# still-image input, so we don't abort on it.
"$VENV_PY" -m scripts.inference \
  --inference_config configs/inference/poc.yaml \
  --result_dir results/poc \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config  models/musetalkV15/musetalk.json \
  --version v15 || true

if [ -f "$OUT" ]; then
  cp "$OUT" ../../outputs/musetalk_demo.mp4
  echo "==> done -> poc/outputs/musetalk_demo.mp4"
else
  echo "!! output not produced - check the log above"; exit 1
fi
