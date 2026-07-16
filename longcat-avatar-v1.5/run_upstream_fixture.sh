#!/usr/bin/env bash
# Generate an Avatar 1.5 talking video from the upstream image/audio fixture.
set -euo pipefail

case "$0" in
  */*) cd "${0%/*}" ;;
esac

: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
: "${LONGCAT_RESOLUTION:=480p}"
: "${LONGCAT_NUM_SEGMENTS:=1}"

ROOT="$PWD"
SOURCE="$ROOT/LongCat-Video"
VENV="$ROOT/.venv"
CHECKPOINT="$SOURCE/weights/LongCat-Video-Avatar-1.5"
BASE_MODEL="$SOURCE/weights/LongCat-Video"
OUTPUT="$ROOT/outputs/upstream-single-fixture"

if [ ! -x "$VENV/bin/torchrun" ]; then
  echo "missing isolated environment: $VENV" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/ffmpeg" ]; then
  echo "missing POC-local FFmpeg binary: $VENV/bin/ffmpeg" >&2
  exit 1
fi

if [ ! -d "$CHECKPOINT" ] || [ ! -d "$BASE_MODEL" ]; then
  echo "missing LongCat Avatar 1.5 or base-model weights" >&2
  exit 1
fi

export PATH="$VENV/bin:$PATH"
mkdir -p "$OUTPUT"
cd "$SOURCE"

exec "$VENV/bin/torchrun" \
  --nproc_per_node=2 \
  run_demo_avatar_single_audio_to_video.py \
  --context_parallel_size=2 \
  --checkpoint_dir="$CHECKPOINT" \
  --stage_1=ai2v \
  --input_json=assets/avatar/single_example_1.json \
  --output_dir="$OUTPUT" \
  --resolution="$LONGCAT_RESOLUTION" \
  --num_segments="$LONGCAT_NUM_SEGMENTS" \
  --ref_img_index=10 \
  --mask_frame_range=3 \
  --use_distill \
  --model_type=avatar-v1.5 \
  --use_int8
