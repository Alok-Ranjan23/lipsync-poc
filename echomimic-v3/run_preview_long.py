import os
import shutil
import sys
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent
ECHO_ROOT = RUNNER_ROOT / "EchoMimicV3"

image_path = Path(os.environ["ECHO_IMAGE"]).resolve()
audio_path = Path(os.environ["ECHO_AUDIO"]).resolve()
output_path = Path(os.environ["ECHO_OUTPUT"]).resolve()
prompt = os.environ.get("ECHO_PROMPT", "A person is speaking naturally.")
partial_video_length = int(os.environ.get("ECHO_PARTIAL_VIDEO_LENGTH", "81"))
overlap_video_length = int(os.environ.get("ECHO_OVERLAP_VIDEO_LENGTH", "8"))

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not audio_path.is_file():
    raise FileNotFoundError(f"Audio not found: {audio_path}")

os.chdir(ECHO_ROOT)
sys.path.insert(0, str(ECHO_ROOT))

import infer_preview

case_name = "poc_input"
dataset_root = ECHO_ROOT / "datasets" / "echomimicv3_poc"
images_dir = dataset_root / "imgs"
audios_dir = dataset_root / "audios"
prompts_dir = dataset_root / "prompts"
output_root = ECHO_ROOT / "outputs" / "poc_long"

for directory in (images_dir, audios_dir, prompts_dir, output_root):
    directory.mkdir(parents=True, exist_ok=True)

image_destination = images_dir / f"{case_name}{image_path.suffix.lower()}"
audio_destination = audios_dir / f"{case_name}{audio_path.suffix.lower()}"
shutil.copy2(image_path, image_destination)
shutil.copy2(audio_path, audio_destination)
(prompts_dir / f"{case_name}.txt").write_text(prompt, encoding="utf-8")

original_init = infer_preview.Config.__init__


def configure_poc(self):
    original_init(self)
    self.base_dir = f"{dataset_root}/"
    self.test_name_list = [case_name]
    self.model_name = "../weights/Wan2.1-Fun-V1.1-1.3B-InP"
    self.transformer_path = (
        "../weights/EchoMimicV3/transformer/"
        "diffusion_pytorch_model.safetensors"
    )
    self.wav2vec_model_dir = "../weights/wav2vec2-base-960h"
    self.save_path = str(output_root)
    self.use_longvideo_cfg = True
    self.partial_video_length = partial_video_length
    self.overlap_video_length = overlap_video_length


infer_preview.Config.__init__ = configure_poc
infer_preview.main()

outputs = sorted(output_root.rglob(f"{case_name}_audio.mp4"))
if not outputs:
    raise FileNotFoundError("EchoMimicV3 Preview did not produce an audio video")

output_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(outputs[-1], output_path)
print(f"Saved long-video output to: {output_path}")
