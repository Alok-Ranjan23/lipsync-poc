"""Offline ONNX face restorer (GPEN-BFR-512) to sharpen the Wav2Lip mouth region.

Wav2Lip generates a 96x96 mouth that looks soft when pasted onto a high-res face.
Running a face-restoration model over the generated face crop brings the mouth up
to full detail. Runs on CPU via ONNX Runtime (no torch).

Model: GPEN-BFR-512  (input/output: [1,3,512,512], normalized to [-1, 1]).
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from providers import get_providers  # noqa: E402

HERE = os.path.dirname(__file__)
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))


class GPENEnhancer:
    def __init__(self, model_path=None):
        model_path = model_path or os.path.join(MODELS, "gpen_bfr_512.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=get_providers()
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.size = 512

    def _pre(self, face_bgr):
        f = cv2.resize(face_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        f = (f - 0.5) / 0.5
        return f.transpose(2, 0, 1)[None].astype(np.float32)

    def _post(self, out, dst_w, dst_h):
        o = out[0].transpose(1, 2, 0)
        o = (np.clip(o, -1, 1) + 1) / 2.0 * 255.0
        o = cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(o, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    def enhance(self, face_bgr):
        """Restore a face crop; returns an enhanced crop of the same size."""
        h, w = face_bgr.shape[:2]
        out = self.sess.run([self.out_name], {self.in_name: self._pre(face_bgr)})[0]
        return self._post(out, w, h)
