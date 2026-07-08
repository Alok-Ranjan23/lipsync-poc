"""Fetch the model files needed by the CPU Wav2Lip POC into ../models.

  - wav2lip_gan.onnx  : Wav2Lip generator (ONNX)  -> HuggingFace
  - yunet.onnx        : YuNet face detector (ONNX) -> OpenCV Zoo (git-lfs)
  - sface.onnx        : SFace recognizer for CSIM  -> OpenCV Zoo (git-lfs)
  - gpen_bfr_512.onnx : GPEN face restorer (--enhancer, optional) -> GitHub release
"""
import argparse
import os
import shutil
import urllib.request

HERE = os.path.dirname(__file__)
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
os.makedirs(MODELS, exist_ok=True)

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
GPEN_URL = (
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/"
    "Models/GPEN-BFR-512.onnx"
)


def fetch_wav2lip():
    dst = os.path.join(MODELS, "wav2lip_gan.onnx")
    if os.path.exists(dst):
        print("wav2lip_gan.onnx already present")
        return
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id="wanesoft/faceswap_pack", filename="wav2lip_gan.onnx")
    shutil.copy(p, dst)
    print("downloaded wav2lip_gan.onnx ->", dst)


def _fetch_url(url, name, min_size=100_000):
    dst = os.path.join(MODELS, name)
    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"{name} already present")
        return
    urllib.request.urlretrieve(url, dst)
    print(f"downloaded {name} ->", dst, os.path.getsize(dst), "bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhancer", action="store_true",
                    help="also download GPEN-BFR-512 face restorer (~284 MB)")
    args = ap.parse_args()

    fetch_wav2lip()
    _fetch_url(YUNET_URL, "yunet.onnx")
    _fetch_url(SFACE_URL, "sface.onnx")
    if args.enhancer:
        _fetch_url(GPEN_URL, "gpen_bfr_512.onnx", min_size=1_000_000)
    print("done.")
