"""Streaming Silero VAD (the ONNX model that ships inside faster-whisper)."""
from __future__ import annotations

import os

import numpy as np

FRAME = 512          # samples per VAD step at 16 kHz (32 ms)
_CONTEXT = 64


class SileroVAD:
    """Stateful: call with consecutive 512-sample frames, get a speech probability each time."""

    def __init__(self, path: str | None = None):
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        path = path or os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(_CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._ctx, frame]).astype(np.float32, copy=False)[None, :]
        out, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = frame[-_CONTEXT:]
        return float(out.reshape(-1)[0])
