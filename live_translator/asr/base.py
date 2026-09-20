from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..languages import base_lang


class AsrError(RuntimeError):
    """Recoverable recognition failure with a user-presentable message."""


@dataclass
class AsrResult:
    text: str
    language: str = ""            # internal code ("en", "zh", ...); "" when the backend did not say
    language_prob: float = 0.0
    elapsed_ms: float = 0.0


class Recognizer(ABC):
    name = "asr"
    supports_partials = True

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        """Load / connect. May be slow (model download); report human-readable progress."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None, *, final: bool,
                   prompt: str = "") -> AsrResult:
        """``audio``: 16 kHz mono float32. ``language``: internal code or None for auto-detect."""

    def close(self) -> None:
        pass


class LanguageTracker:
    """Stabilises Whisper's per-utterance language guess.

    Short utterances ("Okay.", "Yeah") are often mis-detected, so a lone low-confidence outlier must not
    flip the subtitle language. Keeps a small window of confident observations and reports the winner.
    """

    CONFIDENT = 0.80

    def __init__(self, window: int = 6):
        self._obs: deque[str] = deque(maxlen=window)

    def observe(self, lang: str, prob: float) -> None:
        if lang and prob >= 0.5:
            # a very confident reading counts double so genuine language switches are picked up fast
            self._obs.append(base_lang(lang))
            if prob >= self.CONFIDENT:
                self._obs.append(base_lang(lang))

    @property
    def current(self) -> str | None:
        if not self._obs:
            return None
        return max(set(self._obs), key=list(self._obs).count)

    def reset(self) -> None:
        self._obs.clear()

    def resolve(self, detected: str, prob: float, duration_s: float = 0.0) -> str:
        """Language to use for this utterance given a fresh detection.

        Detection on a short clip is unreliable, on a longer one it is trustworthy, so the bar for
        overriding the established language drops as the clip gets longer."""
        cur = self.current
        trust = 0.92 if duration_s < 2.5 else 0.70
        if cur is None or base_lang(detected) == cur or prob >= trust:
            return base_lang(detected) or cur or ""
        return cur           # low-confidence disagreement: stay with the established language
