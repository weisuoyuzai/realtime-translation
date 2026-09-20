"""Local recognition with faster-whisper (CTranslate2)."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import numpy as np

from ..config import AsrCfg
from ..languages import normalize_lang
from ..runtime import models_dir, recommended_asr_model, resolve_asr_device
from .base import AsrError, AsrResult, Recognizer
from .filters import SegStats, clean_text, keep_segment

log = logging.getLogger(__name__)

# Tiny/base models tend to read an initial_prompt back as the transcript, so they don't get one.
_PROMPT_SAFE = ("small", "medium", "large", "distil", "turbo")


class LocalWhisper(Recognizer):
    def __init__(self, cfg: AsrCfg):
        self.cfg = cfg
        self.model_name = cfg.model
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = None
        self.name = "本地 faster-whisper"

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        say = progress or (lambda _m: None)
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise AsrError("缺少依赖 faster-whisper，请执行: pip install faster-whisper") from e

        def attempt(device: str, compute: str) -> None:
            name = self.cfg.model if self.cfg.model != "auto" else recommended_asr_model(device)
            say(f"正在加载 Whisper 模型 {name}（{device}/{compute}）…")
            threads = min(8, os.cpu_count() or 4) if device == "cpu" else 0

            def build(local_only: bool):
                return WhisperModel(name, device=device, compute_type=compute, cpu_threads=threads,
                                    local_files_only=local_only, download_root=models_dir())

            try:
                # Cached model → no network round-trips at all: instant start, and it works offline or when
                # huggingface.co is slow/blocked (the default path would wait for connection time-outs).
                self._model = build(True)
            except Exception:
                say(f"本地还没有模型 {name}，正在下载（首次使用，可能需要几分钟）…")
                self._model = build(False)
            self.model_name, self.device, self.compute_type = name, device, compute
            self.name = f"本地 Whisper {name} · {device}"
            say("预热识别引擎…")
            # The first call pays for CUDA/cuDNN initialisation (seconds) and is also where missing
            # GPU libraries show up, so a failure here is treated like a failed load.
            self.transcribe((np.random.randn(16000 * 2) * 0.01).astype(np.float32), "en", final=False)

        device, compute = resolve_asr_device(self.cfg.device, self.cfg.compute_type)
        try:
            attempt(device, compute)
        except Exception as e:
            self._model = None
            if device != "cuda":
                raise AsrError(f"加载 Whisper 模型失败：{e}") from e
            log.warning("CUDA failed (%s); falling back to CPU", e)
            say(f"GPU 不可用（{e}），改用 CPU。")
            try:
                attempt("cpu", "int8")
            except Exception as e2:
                self._model = None
                raise AsrError(f"加载 Whisper 模型失败：{e2}") from e2

    def transcribe(self, audio: np.ndarray, language: str | None, *, final: bool,
                   prompt: str = "") -> AsrResult:
        if self._model is None:
            raise AsrError("识别模型尚未加载")
        t0 = time.perf_counter()
        lang = None if not language or language == "auto" else language
        use_prompt = bool(prompt) and any(k in self.model_name for k in _PROMPT_SAFE)
        try:
            segments, info = self._model.transcribe(
                audio,
                language=lang,
                task="transcribe",
                beam_size=max(1, self.cfg.beam_final) if final else 1,
                best_of=1,
                temperature=0.0,                    # single pass: fallback re-decodes would blow the latency budget
                condition_on_previous_text=False,
                initial_prompt=prompt if use_prompt else None,
                without_timestamps=True,
                vad_filter=False,                   # the segmenter already did VAD
                compression_ratio_threshold=2.4,
            )
            kept = []
            for seg in segments:
                stats = SegStats(seg.text, seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio)
                if keep_segment(stats):
                    kept.append(seg.text.strip())
        except Exception as e:
            raise AsrError(f"识别失败：{e}") from e

        text = clean_text(" ".join(kept), prompt if use_prompt else "")
        return AsrResult(text=text, language=normalize_lang(info.language),
                         language_prob=float(info.language_probability or 0.0),
                         elapsed_ms=(time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self._model = None
