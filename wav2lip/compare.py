"""Compare lip-sync outputs side by side with reference-free metrics (CPU).

Runs the same metrics (CSIM identity + mouth/face sharpness) over several videos
and prints one comparison table. Defaults to the three POC outputs.

Usage:
  python wav2lip_cpu/compare.py --source assets/sample_face.png \
      --videos outputs/wav2lip_demo.mp4 outputs/wav2lip_enhanced.mp4 outputs/musetalk_demo.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Compare lip-sync outputs")
    ap.add_argument("--source", default="assets/sample_face.png")
    ap.add_argument("--videos", nargs="+", default=[
        "outputs/wav2lip_demo.mp4",
        "outputs/wav2lip_enhanced.mp4",
        "outputs/musetalk_demo.mp4",
    ])
    ap.add_argument("--sample-every", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for v in args.videos:
        if not os.path.exists(v):
            print(f"[skip] not found: {v}")
            continue
        print(f"[eval] {v} ...")
        m = compute_metrics(v, args.source, args.sample_every)
        rows.append((os.path.basename(v), m))

    if not rows:
        print("No videos evaluated.")
        return

    name_w = max(len(n) for n, _ in rows) + 2
    print("\n" + "=" * (name_w + 56))
    print(f"{'method'.ljust(name_w)}{'CSIM':>8}{'mouth_sharp':>14}"
          f"{'face_sharp':>12}{'m/f ratio':>12}")
    print("-" * (name_w + 56))
    for name, m in rows:
        print(f"{name.ljust(name_w)}{m['csim']:>8.3f}{m['mouth_sharpness']:>14.1f}"
              f"{m['face_sharpness']:>12.1f}{m['mouth_face_ratio']:>12.2f}")
    print("=" * (name_w + 56))
    print("CSIM: identity (higher=better).  mouth_sharp: crispness of mouth (higher=better).")
    print("Note: mouth/face ratio differs by method (Wav2Lip re-pastes a resized crop, lowering")
    print("      face_sharp; MuseTalk keeps the face native). For sync accuracy use SyncNet LSE.")


if __name__ == "__main__":
    main()
