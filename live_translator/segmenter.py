"""Cuts a continuous audio stream into utterances and emits *partial* snapshots while one is in progress.

Pure logic (the VAD is injected), so it can be tested deterministically.

Latency levers:
  * ``min_silence_ms``  — how long a pause must last before the utterance is finalised (the dominant
    contribution to end-of-speech latency). It shrinks automatically for long utterances so that a
    non-stop speaker does not make subtitles lag.
  * ``partial_interval_ms`` — cadence of interim ASR snapshots, so text appears while the speaker talks.
  * ``max_utterance_s`` — hard cap; the cut is placed at the quietest point of the last few seconds
    rather than mid-word.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SegmenterCfg
from .vad import FRAME

RATE = 16000
FRAME_MS = FRAME * 1000 / RATE            # 32 ms
START_FRAMES = 3                          # ≈96 ms of consecutive speech to open an utterance
TAIL_PAD_FRAMES = 5                       # ≈160 ms of trailing silence kept on the final audio
MIN_PARTIAL_FRAMES = 22                   # don't decode < ~0.7 s of speech: Whisper hallucinates on it
CUT_SEARCH_S = 3.0


@dataclass
class SegEvent:
    kind: str                             # "partial" | "final" | "discard"
    utt_id: int
    audio: np.ndarray | None = None
    t_speech_end: float = 0.0             # monotonic clock; for finals = when the speaker actually stopped
    reason: str = ""                      # finals: "silence" | "maxlen" | "flush"


class Segmenter:
    def __init__(self, vad: Callable[[np.ndarray], float], cfg: SegmenterCfg,
                 clock: Callable[[], float] = time.monotonic):
        self.vad = vad
        self.cfg = cfg
        self.clock = clock
        self._leftover = np.zeros(0, dtype=np.float32)
        self._next_id = 1
        self._preroll: deque[tuple[np.ndarray, float]] = deque(
            maxlen=max(1, int(cfg.preroll_ms / FRAME_MS)) + START_FRAMES)
        self._reset_utterance()

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def in_speech(self) -> bool:
        return self._speech

    def feed(self, samples: np.ndarray) -> list[SegEvent]:
        """Consume 16 kHz mono float32 audio of any length; returns events in order."""
        out: list[SegEvent] = []
        data = np.concatenate([self._leftover, samples]) if len(self._leftover) else samples
        n = len(data) // FRAME
        for i in range(n):
            self._on_frame(data[i * FRAME:(i + 1) * FRAME], out)
        self._leftover = data[n * FRAME:].copy()
        return out

    def flush(self) -> list[SegEvent]:
        out: list[SegEvent] = []
        if self._speech:
            self._finalize(out, "flush")
        return out

    # ── internals ────────────────────────────────────────────────────────────

    def _reset_utterance(self) -> None:
        self._speech = False
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._voiced = 0
        self._silence = 0
        self._trigger = 0
        self._since_partial = 0
        self._partial_sent = False
        self._cur_id = 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _effective_min_silence(self, n_frames: int) -> float:
        base = float(self.cfg.min_silence_ms)
        dur, cap = n_frames * FRAME_MS, self.cfg.max_utterance_s * 1000
        if dur > 0.7 * cap:
            base *= 0.6
        elif dur > 0.45 * cap:
            base *= 0.8
        return max(200.0, base)

    def _on_frame(self, frame: np.ndarray, out: list[SegEvent]) -> None:
        p = self.vad(frame)
        thr = self.cfg.vad_threshold
        neg = max(0.05, thr - 0.15)               # hysteresis: once talking, dips above `neg` still count as speech

        if not self._speech:
            self._preroll.append((frame, p))
            self._trigger = self._trigger + 1 if p >= thr else 0
            if self._trigger >= START_FRAMES:
                self._speech = True
                self._cur_id = self._new_id()
                self._frames = [f for f, _ in self._preroll]
                self._probs = [q for _, q in self._preroll]
                self._voiced = self._trigger
                self._silence = 0
                self._since_partial = 0
                self._partial_sent = False
            return

        self._frames.append(frame)
        self._probs.append(p)
        self._since_partial += 1
        if p >= thr:
            self._voiced += 1
        if p >= neg:
            self._silence = 0
        else:
            self._silence += 1

        n = len(self._frames)
        if self._silence * FRAME_MS >= self._effective_min_silence(n):
            self._finalize(out, "silence")
        elif n * FRAME_MS >= self.cfg.max_utterance_s * 1000:
            self._force_cut(out)
        elif (self._since_partial * FRAME_MS >= self.cfg.partial_interval_ms
              and self._voiced >= MIN_PARTIAL_FRAMES):
            self._since_partial = 0
            self._partial_sent = True
            out.append(SegEvent("partial", self._cur_id, np.concatenate(self._frames), self.clock()))

    def _min_speech_frames(self) -> int:
        return max(2, int(self.cfg.min_speech_ms / FRAME_MS))

    def _finalize(self, out: list[SegEvent], reason: str) -> None:
        trim = max(0, self._silence - TAIL_PAD_FRAMES)
        frames = self._frames[:len(self._frames) - trim] if trim else self._frames
        if self._voiced >= self._min_speech_frames():
            out.append(SegEvent("final", self._cur_id, np.concatenate(frames),
                                self.clock() - self._silence * FRAME_MS / 1000, reason))
        elif self._partial_sent:
            out.append(SegEvent("discard", self._cur_id))     # let the UI retract the interim line
        self._reset_utterance()
        self._preroll.clear()

    def _force_cut(self, out: list[SegEvent]) -> None:
        """Split at the quietest point of the last few seconds; the remainder continues as a new utterance."""
        n = len(self._frames)
        span = min(n, int(CUT_SEARCH_S * 1000 / FRAME_MS))
        lo = max(int(n * 0.4), n - span)
        probs = np.asarray(self._probs[lo:n], dtype=np.float32)
        smooth = np.convolve(probs, np.ones(3, dtype=np.float32) / 3, mode="same") if len(probs) >= 3 else probs
        k = lo + int(np.argmin(smooth))
        cut = k + 1 if smooth.min() < 0.5 else n
        head, tail = self._frames[:cut], self._frames[cut:]
        head_probs, tail_probs = self._probs[:cut], self._probs[cut:]
        thr = self.cfg.vad_threshold

        if sum(q >= thr for q in head_probs) >= self._min_speech_frames():
            out.append(SegEvent("final", self._cur_id, np.concatenate(head), self.clock(), "maxlen"))
        elif self._partial_sent:
            out.append(SegEvent("discard", self._cur_id))

        self._frames, self._probs = tail, tail_probs
        self._cur_id = self._new_id()
        self._voiced = sum(q >= thr for q in tail_probs)
        self._silence = 0
        for q in reversed(tail_probs):
            if q >= max(0.05, thr - 0.15):
                break
            self._silence += 1
        self._since_partial = 0
        self._partial_sent = False
