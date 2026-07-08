"""Offline, reference-free quality metrics for a lip-sync output video (CPU).

Metrics computed here (no ground-truth video required):
  * CSIM        : identity preservation. Cosine similarity between the ArcFace-style
                  SFace embedding of the source face and each generated frame's face.
                  Range ~[-1, 1]; higher = the person still looks like themselves.
                  >0.6 is generally "same person".
  * mouth_sharp : sharpness of the mouth region (variance of Laplacian). Higher =
                  crisper mouth; low values reveal the Wav2Lip 96x96 blur.
  * face_sharp  : same metric over the whole face, for reference/ratio.

Not computed here (need extra infra):
  * LSE-C / LSE-D : the STANDARD lip-sync metrics. Require SyncNet (PyTorch). Run on
                    the GPU box / Colab:  https://github.com/joonson/syncnet_python
  * FID / FVD / SSIM / PSNR / LPIPS : need a ground-truth reference video.

Usage:
  python wav2lip_cpu/evaluate.py --video outputs/wav2lip_demo.mp4 --source assets/sample_face.png
"""
import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))


def make_detector():
    return cv2.FaceDetectorYN.create(
        os.path.join(MODELS, "yunet.onnx"), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
    )


def biggest_face(det, frame):
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[-1])


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mouth_sharpness(frame, face):
    # YuNet landmarks: reye, leye, nose, rmouth, lmouth (x,y each) at index 4..14
    lms = face[4:14].reshape(5, 2)
    (rmx, rmy), (lmx, lmy) = lms[3], lms[4]
    cx, cy = int((rmx + lmx) / 2), int((rmy + lmy) / 2)
    mw = int(abs(lmx - rmx) * 1.6) or 40
    mh = int(mw * 0.7)
    x1, y1 = max(0, cx - mw // 2), max(0, cy - mh // 2)
    x2, y2 = cx + mw // 2, cy + mh // 2
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return laplacian_var(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))


def compute_metrics(video, source, sample_every=5):
    """Return reference-free quality metrics for one video as a dict."""
    det = make_detector()
    recog = cv2.FaceRecognizerSF.create(os.path.join(MODELS, "sface.onnx"), "")

    src = cv2.imread(source)
    if src is None:
        raise RuntimeError(f"cannot read source: {source}")
    sf = biggest_face(det, src)
    if sf is None:
        raise RuntimeError("no face in source image")
    src_feat = recog.feature(recog.alignCrop(src, sf))

    cap = cv2.VideoCapture(video)
    csims, mouth_s, face_s = [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            f = biggest_face(det, fr)
            if f is not None:
                feat = recog.feature(recog.alignCrop(fr, f))
                csims.append(float(recog.match(
                    src_feat, feat, cv2.FaceRecognizerSF_FR_COSINE)))
                mouth_s.append(mouth_sharpness(fr, f))
                x, y, bw, bh = f[:4].astype(int)
                roi = fr[max(0, y):y + bh, max(0, x):x + bw]
                if roi.size:
                    face_s.append(laplacian_var(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        i += 1
    cap.release()

    mm = float(np.mean(mouth_s)) if mouth_s else 0.0
    fm = float(np.mean(face_s)) if face_s else 0.0
    return {
        "frames": len(csims),
        "csim": float(np.mean(csims)) if csims else 0.0,
        "mouth_sharpness": mm,
        "face_sharpness": fm,
        "mouth_face_ratio": (mm / fm) if fm else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Reference-free lip-sync quality metrics")
    ap.add_argument("--video", required=True)
    ap.add_argument("--source", required=True, help="source face image/frame")
    ap.add_argument("--sample-every", type=int, default=5, help="eval every Nth frame")
    args = ap.parse_args()

    m = compute_metrics(args.video, args.source, args.sample_every)
    print(f"frames evaluated : {m['frames']}")
    print(f"CSIM (identity)  : mean={m['csim']:.3f}   [>0.6 = same person]")
    print(f"mouth sharpness  : {m['mouth_sharpness']:.1f}   (variance of Laplacian; higher=crisper)")
    print(f"face sharpness   : {m['face_sharpness']:.1f}")
    print(f"mouth/face ratio : {m['mouth_face_ratio']:.2f}")
    print("\nNote: run SyncNet LSE-C/LSE-D for the standard lip-sync-accuracy score.")


if __name__ == "__main__":
    main()
