"""Local Wav2Lip lip-sync POC using ONNX Runtime.

Runs on CPU by default; auto-detects and uses CUDA if `onnxruntime-gpu` is
installed (see providers.py). No PyTorch required.

Pipeline:
  audio        -> mel spectrogram (Wav2Lip params)                 [audio.py]
  face img/vid -> YuNet face detection -> 96x96 crop (lower half masked)
  ONNX Wav2Lip generator (mel + masked face) -> new mouth region
  feather-blend back into the frame  (optional GPEN mouth sharpening)
  pipe raw frames -> ffmpeg (H.264 encode + original-audio mux)

Quality is the classic Wav2Lip 96x96 baseline (soft mouth); for more realistic
output use MuseTalk (../musetalk) or the GPU Colab notebooks (../colab_gpu).
"""
import argparse
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import audio as audio_lib  # noqa: E402

IMG_SIZE = 96
MEL_STEP_SIZE = 16
HERE = os.path.dirname(__file__)
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))


_feather_cache = {}


def _feather_mask(bh, bw):
    """Soft-edged HxWx3 alpha mask (1 in center, fading to 0 at the border)."""
    key = (bh, bw)
    if key not in _feather_cache:
        m = np.zeros((bh, bw), np.float32)
        b = max(2, int(min(bh, bw) * 0.12))
        m[b:bh - b, b:bw - b] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=b / 2.0, sigmaY=b / 2.0)
        _feather_cache[key] = m[:, :, None]
    return _feather_cache[key]


def make_detector():
    return cv2.FaceDetectorYN.create(
        os.path.join(MODELS, "yunet.onnx"), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
    )


def detect_face(det, frame, pad=(0.0, 0.10, 0.0, 0.0)):
    """Return (x1, y1, x2, y2) of the highest-confidence face, with padding.

    pad = (top, bottom, left, right) as fractions of box height/width. Keep this
    small: Wav2Lip expects a tight forehead-to-chin crop (its lower half is what
    gets regenerated). Too much bottom padding pushes the generated mouth onto
    the neck. Real Wav2Lip uses ~10px bottom padding only.
    """
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        raise RuntimeError("No face detected")
    face = max(faces, key=lambda f: f[-1])  # highest score
    x1, y1, bw, bh = face[:4].astype(int)
    x2, y2 = x1 + bw, y1 + bh
    pt, pb, pl, pr = pad
    x1 = max(0, int(x1 - pl * bw))
    x2 = min(w, int(x2 + pr * bw))
    y1 = max(0, int(y1 - pt * bh))
    y2 = min(h, int(y2 + pb * bh))
    return x1, y1, x2, y2


def read_frames(path, out_fps):
    """Yield BGR frames. A still image becomes a single frame (reused for all)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return [img], True
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return frames, False


def build_mel_chunks(mel, fps):
    chunks = []
    idx_mult = 80.0 / fps
    i = 0
    n = mel.shape[1]
    while True:
        start = int(i * idx_mult)
        if start + MEL_STEP_SIZE > n:
            chunks.append(mel[:, n - MEL_STEP_SIZE:])
            break
        chunks.append(mel[:, start:start + MEL_STEP_SIZE])
        i += 1
    return chunks


def main():
    ap = argparse.ArgumentParser(description="CPU Wav2Lip ONNX lip-sync POC")
    ap.add_argument("--face", required=True, help="face image or video")
    ap.add_argument("--audio", required=True, help="driving audio (wav)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--model", default=os.path.join(MODELS, "wav2lip_gan.onnx"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--enhance", choices=["none", "gpen"], default="none",
                    help="face restorer to sharpen the generated mouth (CPU, slower)")
    args = ap.parse_args()

    enhancer = None
    if args.enhance == "gpen":
        from enhancer import GPENEnhancer
        print("    (loading GPEN face enhancer)")
        enhancer = GPENEnhancer()

    print("[1/5] loading models...")
    from providers import get_providers
    sess = ort.InferenceSession(args.model, providers=get_providers())
    in_mel, in_face = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    face_net = make_detector()

    print("[2/5] audio -> mel...")
    wav = audio_lib.load_wav(args.audio)
    mel = audio_lib.melspectrogram(wav)
    if np.isnan(mel).any():
        raise ValueError("Mel contains NaN; check the audio file.")
    mel_chunks = build_mel_chunks(mel, args.fps)

    print("[3/5] reading face + detecting...")
    frames, is_still = read_frames(args.face, args.fps)
    n = len(mel_chunks)
    if is_still:
        base = frames[0]
        frames = [base] * n
    else:
        # loop video (ping-pong) to cover the audio length
        seq = frames + frames[::-1]
        frames = [seq[i % len(seq)] for i in range(n)]

    boxes = [detect_face(face_net, frames[0])]  # detect once, reuse for a still
    if not is_still:
        boxes = [detect_face(face_net, f) for f in frames]
    else:
        boxes = boxes * n

    h, w = frames[0].shape[:2]

    # Encode by piping raw BGR frames straight into ffmpeg (audio muxed in the
    # same pass). This avoids cv2.VideoWriter, whose codec backend is missing in
    # some opencv-python-headless builds and fails silently.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", str(args.fps), "-i", "-",           # raw video from stdin
        "-i", args.audio,                          # audio track
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"[4/5] running Wav2Lip on {n} frames (CPU)...")
    for s in tqdm(range(0, n, args.batch)):
        e = min(n, s + args.batch)
        img_batch, mel_batch, meta = [], [], []
        for i in range(s, e):
            frame = frames[i].copy()
            x1, y1, x2, y2 = boxes[i]
            face = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
            masked = face.copy()
            masked[IMG_SIZE // 2:] = 0
            img = np.concatenate([masked, face], axis=2) / 255.0  # HxWx6
            img_batch.append(img)
            mel_batch.append(mel_chunks[i])
            meta.append((frame, (x1, y1, x2, y2)))

        img_np = np.asarray(img_batch, dtype=np.float32).transpose(0, 3, 1, 2)
        mel_np = np.asarray(mel_batch, dtype=np.float32)[:, np.newaxis, :, :]
        pred = sess.run([out_name], {in_mel: mel_np, in_face: img_np})[0]
        pred = pred.transpose(0, 2, 3, 1) * 255.0  # -> NHWC

        for j, (frame, (x1, y1, x2, y2)) in enumerate(meta):
            bw, bh = x2 - x1, y2 - y1
            p = np.clip(pred[j], 0, 255).astype(np.uint8)
            p = cv2.resize(p, (bw, bh))
            # Feather the paste so the regenerated crop blends into the original
            # instead of showing a hard rectangular seam.
            mask = _feather_mask(bh, bw)
            roi = frame[y1:y2, x1:x2].astype(np.float32)
            blended = (p.astype(np.float32) * mask + roi * (1.0 - mask)).astype(np.uint8)
            if enhancer is not None:
                # Restore the whole generated face crop, then feather back in so
                # the sharpened mouth matches the rest of the face.
                restored = enhancer.enhance(blended)
                blended = (restored.astype(np.float32) * mask
                           + roi * (1.0 - mask)).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    print("[5/5] finalizing video (ffmpeg encode + audio mux)...")
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
