#!/usr/bin/env bash
# Set up the official LAM avatar generator in an isolated environment.
set -euo pipefail

case "$0" in
  */*) cd "${0%/*}" ;;
esac

: "${LAM_PYTHON:=python3.10}"

if ! command -v "$LAM_PYTHON" >/dev/null; then
  echo "Python interpreter not found: $LAM_PYTHON" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$LAM_PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip

if [ ! -d LAM ]; then
  git clone https://github.com/aigc3d/LAM.git LAM
fi

cd LAM
sh scripts/install/install_cu121.sh
python -m pip install "huggingface_hub[cli]"

huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar
rm -rf ./tmp

huggingface-cli download \
  3DAIGC/LAM-20K \
  --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/

echo "==> LAM setup complete. Set BLENDER_PATH and run ../run_gradio.sh"
