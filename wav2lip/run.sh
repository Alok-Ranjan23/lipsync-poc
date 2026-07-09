#!/usr/bin/env bash
# Run the Wav2Lip pipeline (works for both the CPU and GPU setups).
#
# Usage (from lipsync-poc/wav2lip):
#   bash run.sh                                   # sample face + audio -> outputs/wav2lip_demo.mp4
#   bash run.sh <face> <audio> <out.mp4> [gpen]   # your own inputs; 4th arg 'gpen' = enhancer
#   CUDA_VISIBLE_DEVICES=1 bash run.sh            # pick a GPU (if GPU setup installed)
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
# venv selectable so CPU (.venv) and GPU (.venv-gpu) can coexist:
#   bash run.sh                          # .venv     (CPU, from setup.sh)
#   WAV2LIP_VENV=.venv-gpu bash run.sh   # .venv-gpu (GPU, from setup_gpu.sh)
PY="${WAV2LIP_VENV:-.venv}/bin/python"
[ -x "$PY" ] || { echo "!! venv missing - run setup.sh (CPU) or setup_gpu.sh (GPU) first"; exit 1; }

# If the cuDNN wheel is present (GPU setup), add it to the loader path so
# onnxruntime-gpu can find libcudnn.so.9. Harmless on a CPU-only install.
CUDNN_LIB="$("$PY" - <<'PY'
try:
    import nvidia.cudnn as c, os
    print(os.path.join(list(c.__path__)[0], "lib"))
except Exception:
    print("")
PY
)"
[ -n "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"

FACE="${1:-assets/sample_face.png}"
AUDIO="${2:-assets/sample_audio.wav}"
OUT="${3:-outputs/wav2lip_demo.mp4}"
ENH="${4:-none}"                 # pass 'gpen' for the sharper-mouth enhancer
mkdir -p outputs

"$PY" wav2lip/infer.py --face "$FACE" --audio "$AUDIO" --out "$OUT" --enhance "$ENH"
