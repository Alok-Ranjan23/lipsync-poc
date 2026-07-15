#!/usr/bin/env bash
# Generate a direct talking video from one image and audio.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_image_audio.sh <image> <audio> <output.mp4> [prompt]
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

if [ ! -f "$IMAGE" ]; then
  echo "image not found: $IMAGE" >&2
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "audio not found: $AUDIO" >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT%/*}
if [ "$OUTPUT_DIR" = "$OUTPUT" ]; then
  OUTPUT_DIR=.
fi
mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC1091
. .venv/bin/activate
cd EchoMimicV3

IMAGE_NAME=${IMAGE##*/}
IMAGE_STEM=${IMAGE_NAME%.*}
RESULT_DIR=../outputs
GENERATED="$RESULT_DIR/${IMAGE_STEM}_output.mp4"

SECONDS=0
python infer_flash.py \
  --image_path "$IMAGE" \
  --audio_path "$AUDIO" \
  --prompt "$PROMPT" \
  --num_inference_steps 8 \
  --config_path config/config.yaml \
  --model_name ../weights/Wan2.1-Fun-V1.1-1.3B-InP \
  --ckpt_idx 50000 \
  --transformer_path ../weights/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors \
  --save_path "$RESULT_DIR" \
  --wav2vec_model_dir ../weights/chinese-wav2vec2-base \
  --sampler_name Flow_Unipc \
  --video_length 81 \
  --guidance_scale 6.0 \
  --audio_guidance_scale 3.0 \
  --audio_scale 1.0 \
  --neg_scale 1.0 \
  --neg_steps 0 \
  --seed 43 \
  --enable_teacache \
  --teacache_threshold 0.1 \
  --num_skip_start_steps 5 \
  --riflex_k 6 \
  --ulysses_degree 1 \
  --ring_degree 1 \
  --weight_dtype bfloat16 \
  --sample_size 768 768 \
  --fps 25 \
  --add_prompt "" \
  --negative_prompt "" \
  --shift 5.0

if [ ! -f "$GENERATED" ]; then
  echo "EchoMimicV3 did not produce expected output: $GENERATED" >&2
  exit 1
fi

mv "$GENERATED" "$OUTPUT"
echo "==> done -> $OUTPUT   [TIME] end-to-end: ${SECONDS}s"
