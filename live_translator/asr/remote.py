"""Remote recognition through any OpenAI-compatible ``/audio/transcriptions`` endpoint.

Works with OpenAI (whisper-1, gpt-4o-transcribe), Groq (whisper-large-v3-turbo — very fast), SiliconFlow,
self-hosted faster-whisper-server / Speaches, and the remote_whisper_server.py from jt-live-whisper.
"""
from __future__ import annotations

import io
import logging
import time
import wave
from typing import Callable

import httpx
import numpy as np

from ..config import AsrCfg
from ..languages import normalize_lang
from .base import AsrError, AsrResult, Recognizer
from .filters import SegStats, clean_text, keep_segment

log = logging.getLogger(__name__)


def pcm16_wav(audio: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class RemoteWhisper(Recognizer):
    def __init__(self, cfg: AsrCfg):
        self.cfg = cfg
        self.supports_partials = cfg.remote_partials
        self.name = f"远程 {cfg.remote_model}"
        self._client: httpx.Client | None = None
        self._format = "verbose_json"           # downgraded to "json" if the model rejects it

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        base = self.cfg.remote_base_url.strip().rstrip("/")
        if not base:
            raise AsrError("请填写远程识别 API 地址")
        headers = {"Authorization": f"Bearer {self.cfg.remote_api_key}"} if self.cfg.remote_api_key else {}
        self._client = httpx.Client(base_url=base, headers=headers,
                                    timeout=httpx.Timeout(connect=6, read=30, write=20, pool=5))
        if progress:
            progress(f"远程识别：{self.cfg.remote_model} @ {httpx.URL(base).host}")

    def transcribe(self, audio: np.ndarray, language: str | None, *, final: bool,
                   prompt: str = "") -> AsrResult:
        if self._client is None:
            raise AsrError("识别服务尚未连接")
        t0 = time.perf_counter()
        wav = pcm16_wav(audio)
        for attempt in range(3):
            data = {"model": self.cfg.remote_model, "response_format": self._format, "temperature": "0"}
            if language and language != "auto":
                data["language"] = language.split("-")[0]
            if prompt:
                data["prompt"] = prompt
            try:
                r = self._client.post("/audio/transcriptions", data=data,
                                      files={"file": ("audio.wav", wav, "audio/wav")})
            except httpx.TimeoutException as e:
                if attempt == 2:
                    raise AsrError("远程识别超时") from e
                continue
            except httpx.HTTPError as e:
                if attempt == 2:
                    raise AsrError(f"远程识别网络错误：{e}") from e
                time.sleep(0.3)
                continue

            if r.status_code == 400 and self._format == "verbose_json" and "response_format" in r.text:
                self._format = "json"           # e.g. gpt-4o-transcribe only supports json/text
                continue
            if r.status_code in (401, 403):
                raise AsrError("远程识别 API Key 无效或无权限")
            if r.status_code == 402:
                raise AsrError("远程识别账户余额不足（HTTP 402），请到服务商控制台充值")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 2:
                    raise AsrError(f"远程识别服务繁忙（HTTP {r.status_code}）")
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise AsrError(f"远程识别失败（HTTP {r.status_code}）：{r.text[:200]}")
            return self._parse(r, prompt, (time.perf_counter() - t0) * 1000)
        raise AsrError("远程识别失败")

    def _parse(self, r: httpx.Response, prompt: str, elapsed_ms: float) -> AsrResult:
        try:
            body = r.json()
        except ValueError:
            body = {"text": r.text}
        segments = body.get("segments") or []
        if segments:
            kept = [s.get("text", "").strip() for s in segments
                    if keep_segment(SegStats(s.get("text", ""), s.get("avg_logprob", 0.0),
                                             s.get("no_speech_prob", 0.0), s.get("compression_ratio", 0.0)))]
            text = " ".join(kept)
        else:
            text = (body.get("text") or "").strip()
            if not keep_segment(SegStats(text)):
                text = ""
        return AsrResult(text=clean_text(text, prompt), language=normalize_lang(body.get("language")),
                         language_prob=0.9 if body.get("language") else 0.0, elapsed_ms=elapsed_ms)

    def close(self) -> None:
        c, self._client = self._client, None
        if c is not None:
            c.close()
