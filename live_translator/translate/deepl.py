"""DeepL: fast, high-quality machine translation (a good "remote API" choice when LLM latency is too high)."""
from __future__ import annotations

import threading
from typing import Sequence

import httpx

from ..languages import base_lang
from .base import DeltaCallback, TranslateError, Translator

_TARGET = {
    "en": "EN-US", "zh-Hans": "ZH-HANS", "zh-Hant": "ZH-HANT", "zh": "ZH-HANS", "ja": "JA", "ko": "KO",
    "fr": "FR", "de": "DE", "es": "ES", "pt": "PT-BR", "it": "IT", "ru": "RU", "ar": "AR", "tr": "TR",
    "nl": "NL", "pl": "PL", "uk": "UK", "cs": "CS", "sv": "SV", "da": "DA", "fi": "FI", "no": "NB",
    "el": "EL", "hu": "HU", "ro": "RO", "id": "ID",
}
_SOURCE = {"pt", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "ar", "tr", "nl", "pl", "uk", "cs",
           "sv", "da", "fi", "no", "el", "hu", "ro", "id"}


class DeepLTranslator(Translator):
    def __init__(self, api_key: str, free: bool = True, base_url: str = ""):
        if not api_key.strip():
            raise TranslateError("请填写 DeepL API Key")
        self.name = "DeepL"
        self._client = httpx.Client(
            base_url=base_url or ("https://api-free.deepl.com" if free else "https://api.deepl.com"),
            headers={"Authorization": f"DeepL-Auth-Key {api_key.strip()}"},
            timeout=httpx.Timeout(connect=6, read=15, write=10, pool=5))

    def translate(self, text: str, src: str, tgt: str, *, context: Sequence[tuple[str, str]] = (),
                  on_delta: DeltaCallback | None = None, cancel: threading.Event | None = None) -> str:
        text = text.strip()
        if not text:
            return ""
        target = _TARGET.get(tgt) or _TARGET.get(base_lang(tgt))
        if not target:
            raise TranslateError(f"DeepL 不支持目标语言 {tgt}")
        payload: dict = {"text": [text], "target_lang": target, "preserve_formatting": True}
        if src and base_lang(src) in _SOURCE:
            payload["source_lang"] = base_lang(src).upper()
        if context:                                    # `context` is not billed and steers disambiguation
            payload["context"] = " ".join(s for s, _ in context[-2:])
        try:
            r = self._client.post("/v2/translate", json=payload)
        except httpx.HTTPError as e:
            raise TranslateError(f"DeepL 请求失败：{e}") from e
        if r.status_code in (401, 403):
            raise TranslateError("DeepL API Key 无效（免费版 Key 以 :fx 结尾，需勾选“免费版”）")
        if r.status_code == 456:
            raise TranslateError("DeepL 额度已用完")
        if r.status_code >= 400:
            raise TranslateError(f"DeepL 错误（HTTP {r.status_code}）：{r.text[:160]}")
        out = r.json()["translations"][0]["text"].strip()
        if on_delta and out:
            on_delta(out)
        return out

    def close(self) -> None:
        self._client.close()
