"""Streaming resampler to the 16 kHz mono float32 that Whisper and the VAD expect."""
from __future__ import annotations

import numpy as np

TARGET_RATE = 16000


class Resampler:
    def __init__(self, target: int = TARGET_RATE):
        self.target = target
        self._rate: int | None = None
        self._stream = None
        self._tail = np.zeros(0, dtype=np.float32)   # numpy fallback state

    def _reset(self, rate: int) -> None:
        self._rate = rate
        self._tail = np.zeros(0, dtype=np.float32)
        self._stream = None
        if rate != self.target:
            try:
                import soxr
                self._stream = soxr.ResampleStream(rate, self.target, num_channels=1,
                                                   dtype="float32", quality="HQ")
            except ImportError:
                self._stream = None

    def process(self, x: np.ndarray, rate: int) -> np.ndarray:
        if rate != self._rate:
            self._reset(rate)
        if rate == self.target:
            return x
        if self._stream is not None:
            return self._stream.resample_chunk(x).astype(np.float32, copy=False)
        return self._fallback(x, rate)

    def _fallback(self, x: np.ndarray, rate: int) -> np.ndarray:
        """Box-filter + linear interpolation. Good enough for speech when soxr is not installed."""
        ratio = rate / self.target
        if ratio > 1:
            k = max(1, int(round(ratio)))
            if k > 1 and len(x) >= k:
                x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
        data = np.concatenate([self._tail, x])
        n_out = int((len(data) - 1) / ratio)
        if n_out <= 0:
            self._tail = data
            return np.zeros(0, dtype=np.float32)
        pos = np.arange(n_out) * ratio
        out = np.interp(pos, np.arange(len(data)), data).astype(np.float32)
        self._tail = data[int(n_out * ratio):]
        return out
