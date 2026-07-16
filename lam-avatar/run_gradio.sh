#!/usr/bin/env bash
# Start the official LAM Gradio UI for photo-to-avatar generation and ZIP export.
set -euo pipefail

case "$0" in
  */*) cd "${0%/*}" ;;
esac

if [ ! -x .venv/bin/python ] || [ ! -d LAM ]; then
  echo "LAM is not set up. Run: LAM_PYTHON=python3.10 bash setup_gpu.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate
cd LAM

if [ -n "${BLENDER_PATH:-}" ]; then
  exec python app_lam.py --blender_path "$BLENDER_PATH"
fi

echo "BLENDER_PATH is not set; avatar generation can run, but ZIP export requires Blender." >&2
exec python app_lam.py
