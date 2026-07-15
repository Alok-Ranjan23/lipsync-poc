import argparse
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent
FPS = 25
FLASH_MAX_FRAMES = 81

parser = argparse.ArgumentParser(
    description="Choose EchoMimicV3 Flash or Preview long-video inference."
)
parser.add_argument(
    "--mode",
    choices=("auto", "short", "long"),
    default="auto",
    help="auto selects Flash for <=81 frames and Preview for longer audio",
)
parser.add_argument("image", type=Path)
parser.add_argument("audio", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument(
    "--prompt",
    default="A person is speaking naturally.",
)
parser.add_argument(
    "--partial-video-length",
    type=int,
    default=81,
    help="Preview chunk length in frames",
)
parser.add_argument(
    "--max-vram",
    default="0.60",
    help="Preview mmgp GPU-memory budget ratio",
)
args = parser.parse_args()

if not args.image.is_file():
    raise FileNotFoundError(f"Image not found: {args.image}")

if not args.audio.is_file():
    raise FileNotFoundError(f"Audio not found: {args.audio}")

if shutil.which("ffprobe") is not None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(args.audio),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
elif args.audio.suffix.lower() == ".wav":
    with wave.open(str(args.audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
else:
    raise RuntimeError(
        "ffprobe is required for non-WAV audio. Install FFmpeg or provide a WAV file."
    )

frames = int(math.ceil(duration * FPS))

mode = args.mode
if mode == "auto":
    mode = "short" if frames <= FLASH_MAX_FRAMES else "long"

script_name = (
    "run_image_audio.sh" if mode == "short" else "run_image_audio_long.sh"
)
script = RUNNER_ROOT / script_name

environment = os.environ.copy()
if mode == "long":
    environment["ECHO_PARTIAL_VIDEO_LENGTH"] = str(args.partial_video_length)
    environment["ECHO_MAX_VRAM"] = args.max_vram

print(
    f"==> EchoMimic mode={mode} audio={duration:.2f}s "
    f"frames={frames} script={script_name}"
)
subprocess.run(
    [
        "bash",
        str(script),
        str(args.image.resolve()),
        str(args.audio.resolve()),
        str(args.output.resolve()),
        args.prompt,
    ],
    check=True,
    env=environment,
)
