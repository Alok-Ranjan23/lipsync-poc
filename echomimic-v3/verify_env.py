import cv2
import numpy
import torch
import torchvision

expected_numpy = "1.26.4"

if numpy.__version__ != expected_numpy:
    raise RuntimeError(
        f"Expected NumPy {expected_numpy}, found {numpy.__version__}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print(
    "torch",
    torch.__version__,
    "| torchvision",
    torchvision.__version__,
    "| numpy",
    numpy.__version__,
    "| opencv",
    cv2.__version__,
    "| cuda",
    torch.version.cuda,
    "| gpu",
    torch.cuda.get_device_name(0),
)
