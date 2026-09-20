from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

# on_audio(samples: float32 mono in [-1, 1], sample_rate)
AudioCallback = Callable[[np.ndarray, int], None]


class AudioSourceError(RuntimeError):
    """Raised by ``AudioSource.start`` with a message that is safe to show to the user."""


class AudioSource(ABC):
    name: str = "audio"

    @abstractmethod
    def start(self, on_audio: AudioCallback) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


@dataclass
class AudioApp:
    key: str            # stable id stored in config: exe name (Windows) / bundle id (macOS)
    name: str           # display name
    pid: int = 0        # best pid *right now* (0 = not resolved)
    active: bool = False  # currently producing sound (Windows only)
    title: str = ""     # window title, helps telling browser tabs / meetings apart

    @property
    def label(self) -> str:
        dot = "● " if self.active else "   "
        extra = f" — {self.title[:48]}" if self.title else ""
        return f"{dot}{self.name}{extra}"


def to_mono(frames: np.ndarray) -> np.ndarray:
    """(n,) or (n, ch) float array → contiguous float32 mono."""
    if frames.ndim == 2:
        frames = frames.mean(axis=1)
    return np.ascontiguousarray(frames, dtype=np.float32)
