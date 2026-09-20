"""Play an audio file into the pipeline as if it were live (for demos, tests and benchmarking)."""
from __future__ import annotations

import threading
import time

import numpy as np

from .base import AudioCallback, AudioSource, AudioSourceError

_RATE = 16000


class FileSource(AudioSource):
    def __init__(self, path: str, speed: float = 1.0, tail_silence_s: float = 2.0):
        self.path = path
        self.speed = max(0.1, speed)
        self.tail_silence_s = tail_silence_s   # trailing silence so the last utterance gets closed by the VAD
        self.name = f"文件 {path}"
        self.done = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_audio: AudioCallback) -> None:
        try:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(self.path, sampling_rate=_RATE)
        except Exception as e:
            raise AudioSourceError(f"无法读取音频文件 {self.path}：{e}") from e
        audio = np.concatenate([audio, np.zeros(int(self.tail_silence_s * _RATE), dtype=np.float32)])
        self.done.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(audio, on_audio), daemon=True)
        self._thread.start()

    def _run(self, audio: np.ndarray, on_audio: AudioCallback) -> None:
        step = _RATE // 10                                    # 100 ms
        t0 = time.monotonic()
        for i in range(0, len(audio), step):
            if self._stop.is_set():
                return
            on_audio(audio[i:i + step], _RATE)
            due = t0 + (i + step) / _RATE / self.speed        # pace to (speed × real time)
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.done.set()

    def stop(self) -> None:
        self._stop.set()
