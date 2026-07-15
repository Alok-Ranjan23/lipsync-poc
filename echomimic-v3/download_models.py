from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"

snapshot_download(
    "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    local_dir=WEIGHTS / "Wan2.1-Fun-V1.1-1.3B-InP",
)
snapshot_download(
    "BadToBest/EchoMimicV3",
    local_dir=WEIGHTS / "EchoMimicV3",
    allow_patterns=["echomimicv3-flash-pro/*"],
)

print("Hugging Face weights ready")
