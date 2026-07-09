#!/usr/bin/env bash
# Run LatentSync 1.6 inference on the GPU.
#
# Usage (from lipsync-poc/latentsync):
#   CUDA_VISIBLE_DEVICES=1 bash run.sh
#   CUDA_VISIBLE_DEVICES=1 bash run.sh <video> <audio> <out.mp4>
#
# LatentSync is video->video. Default uses its bundled demo video + our sample audio.
# For a photo, first make a static video (see the commented block below).
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES
cd LatentSync
# shellcheck disable=SC1091
. .venv/bin/activate

VIDEO="${1:-assets/demo1_video.mp4}"          # bundled real footage (best quality)
AUDIO="${2:-../../assets/sample_audio.wav}"   # our TTS demo clip
OUT="${3:-../../outputs/latentsync_out.mp4}"
mkdir -p ../../outputs

# If a still image is passed as the "video", wrap it into a static video matched
# to the audio length (LatentSync only accepts video). Head stays frozen; mouth moves.
case "${VIDEO,,}" in
  *.png|*.jpg|*.jpeg|*.bmp|*.webp)
    echo "==> image input -> building static video (frozen head, mouth-only motion)"
    ffmpeg -y -loglevel error -loop 1 -i "$VIDEO" -i "$AUDIO" -shortest \
      -r 25 -vf scale=512:512 -pix_fmt yuv420p -c:v libx264 _static_from_image.mp4
    VIDEO=_static_from_image.mp4 ;;
esac

# --- photo -> static video (uncomment to drive from lipsync-poc/assets/sample_face.png) ---
# DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 ../../assets/sample_audio.wav)
# ffmpeg -y -loop 1 -i ../../assets/sample_face.png -t "$DUR" -r 25 -vf scale=512:512 -pix_fmt yuv420p static_face.mp4
# VIDEO=static_face.mp4

echo "==> LatentSync: video=$VIDEO audio=$AUDIO -> $OUT  (GPU $CUDA_VISIBLE_DEVICES)"
python -c "import torch; print('[device]', 'CUDA' if torch.cuda.is_available() else 'CPU', '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
_start=$(date +%s)
python -m scripts.inference \
  --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --inference_steps 20 --guidance_scale 1.5 \
  --video_path "$VIDEO" --audio_path "$AUDIO" --video_out_path "$OUT"
echo "==> done -> $OUT   [TIME] end-to-end: $(( $(date +%s) - _start ))s"
