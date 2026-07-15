import os
import shutil
import sys
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent
ECHO_ROOT = RUNNER_ROOT / "EchoMimicV3"
WEIGHTS = RUNNER_ROOT / "weights"

image_path = Path(os.environ["ECHO_IMAGE"]).resolve()
audio_path = Path(os.environ["ECHO_AUDIO"]).resolve()
output_path = Path(os.environ["ECHO_OUTPUT"]).resolve()
prompt = os.environ.get("ECHO_PROMPT", "A person is speaking naturally.")
partial_video_length = int(os.environ.get("ECHO_PARTIAL_VIDEO_LENGTH", "81"))
max_vram = os.environ.get("ECHO_MAX_VRAM", "0.75")
negative_prompt = (
    "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. "
    "Bad fingers. Unclear and blurry hands. Unclear gestures, broken hands, "
    "more than five fingers on one hand, extra fingers, fused fingers."
)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not audio_path.is_file():
    raise FileNotFoundError(f"Audio not found: {audio_path}")


def ensure_link(link: Path, target: Path):
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target)


models_dir = ECHO_ROOT / "models"
models_dir.mkdir(exist_ok=True)
ensure_link(models_dir / "Wan2.1-Fun-V1.1-1.3B-InP", WEIGHTS / "Wan2.1-Fun-V1.1-1.3B-InP")
ensure_link(models_dir / "transformer", WEIGHTS / "EchoMimicV3" / "transformer")
ensure_link(models_dir / "wav2vec2-base-960h", WEIGHTS / "wav2vec2-base-960h")

os.chdir(ECHO_ROOT)
sys.path.insert(0, str(ECHO_ROOT))
sys.argv = [sys.argv[0], "--max_vram", max_vram]

import app_mm

generated_path, _ = app_mm.generate(
    str(image_path),
    str(audio_path),
    prompt,
    negative_prompt,
    partial_video_length,
    app_mm.config.guidance_scale,
    app_mm.config.audio_guidance_scale,
    43,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(generated_path, output_path)
print(f"Saved long-video output to: {output_path}")
