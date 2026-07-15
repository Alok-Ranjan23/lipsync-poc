#!/usr/bin/env bash
# Select EchoMimicV3 Flash or Preview based on audio duration.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_image_audio_auto.sh [--mode auto|short|long] <image> <audio> <output.mp4> [--prompt TEXT]
set -euo pipefail

case "$0" in
  */*) cd "${0%/*}" ;;
esac

exec .venv/bin/python run_image_audio_auto.py "$@"
