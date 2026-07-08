"""ONNX Runtime execution-provider selection.

Prefers CUDA (GPU) when available, otherwise falls back to CPU. This lets the
same code run on a GPU box (with `onnxruntime-gpu` installed) or on a CPU-only
machine like WSL2 without any changes.
"""
import onnxruntime as ort


def get_providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
