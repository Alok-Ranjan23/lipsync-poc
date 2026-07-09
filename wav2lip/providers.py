"""ONNX Runtime execution-provider selection.

Auto-picks the best available accelerator, falling back to CPU:
  * CUDA  (NVIDIA)         -> install `onnxruntime-gpu`
  * DirectML (any DX12 GPU on Windows: AMD/Intel/NVIDIA) -> install `onnxruntime-directml`
  * CPU                   -> plain `onnxruntime`
The same code runs unchanged on a CUDA box, a Windows AMD/Intel GPU, or CPU-only.
"""
import onnxruntime as ort


def get_providers():
    available = ort.get_available_providers()
    order = []
    if "CUDAExecutionProvider" in available:      # NVIDIA
        order.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:        # AMD/Intel/NVIDIA on Windows (DirectML)
        order.append("DmlExecutionProvider")
    order.append("CPUExecutionProvider")
    return order
