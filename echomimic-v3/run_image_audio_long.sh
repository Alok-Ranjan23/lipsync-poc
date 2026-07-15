#!/usr/bin/env bash
# Generate a long talking video from one image and audio using EchoMimicV3 Preview.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_image_audio_long.sh <image> <audio> <output.mp4> [prompt]
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <image> <audio> <output.mp4> [prompt]" >&2
  exit 1
fi

case "$0" in
  */*) cd "${0%/*}" ;;
esac

: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

IMAGE="$1"
AUDIO="$2"
OUTPUT="$3"
PROMPT="${4:-A person is speaking naturally.}"

case "$IMAGE" in
  /*) ;;
  *) IMAGE="$PWD/$IMAGE" ;;
esac

case "$AUDIO" in
  /*) ;;
  *) AUDIO="$PWD/$AUDIO" ;;
esac

case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$PWD/$OUTPUT" ;;
esac

# shellcheck disable=SC1091
. .venv/bin/activate

SECONDS=0
ECHO_IMAGE="$IMAGE" \
ECHO_AUDIO="$AUDIO" \
ECHO_OUTPUT="$OUTPUT" \
ECHO_PROMPT="$PROMPT" \
ECHO_PARTIAL_VIDEO_LENGTH="${ECHO_PARTIAL_VIDEO_LENGTH:-81}" \
ECHO_OVERLAP_VIDEO_LENGTH="${ECHO_OVERLAP_VIDEO_LENGTH:-8}" \
ECHO_MAX_VRAM="${ECHO_MAX_VRAM:-0.75}" \
python run_preview_long_offload.py

echo "==> done -> $OUTPUT   [TIME] end-to-end: ${SECONDS}s"
