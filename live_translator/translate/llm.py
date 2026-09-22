"""Translation through any OpenAI-compatible chat-completions endpoint, streamed.

One class covers both "remote API" (OpenAI, DeepSeek, Qwen/DashScope, Groq, SiliconFlow, Gemini's OpenAI
endpoint, OpenRouter ...) and "local model" (Ollama, LM Studio, llama.cpp server, vLLM ...).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Sequence

import httpx

from ..glossary import parse_glossary, prompt_block
from ..text_utils import clean_translation, looks_like_language
from .base import DeltaCallback, TranslateError, Translator
from .prompts import build_correction_messages, build_messages

log = logging.getLogger(__name__)


def default_extra_body(base_url: str, model: str) -> dict:
    """Provider-specific switches that turn *thinking* off (thinking is pure latency for translation).
    Every one of these is dropped automatically if the server rejects it (see ``_adapt``)."""
    u, m = base_url.lower(), model.lower()
    if "dashscope" in u:
        return {"enable_thinking": False}
    if "11434" in u or "ollama" in u:
        return {"reasoning_effort": "none"} if "qwen3" in m or "gpt-oss" in m or "deepseek-r1" in m else {}
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": "minimal"} if m.startswith("gpt-5") else {}
    if "siliconflow" in u and ("qwen3" in m or "deepseek-v3.1" in m):
        return {"enable_thinking": False}
    return {}


class LLMTranslator(Translator):
    streaming = True
    can_correct = True

    def __init__(self, base_url: str, api_key: str, model: str, *, extra_body: str = "",
                 glossary: str = "", topic: str = "", timeout: float = 30.0, label: str = ""):
        if not base_url.strip():
            raise TranslateError("请填写翻译 API 地址")
        if not model.strip():
            raise TranslateError("请填写翻译模型名称")
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.name = f"{label or 'LLM'} · {self.model}"
        self._gloss = prompt_block(parse_glossary(glossary))
        self._topic = topic
        self._extra: dict = default_extra_body(self.base_url, self.model)
        if extra_body.strip():
            try:
                user_extra = json.loads(extra_body)
                if not isinstance(user_extra, dict):
                    raise ValueError
                self._extra = user_extra
            except ValueError:
                raise TranslateError("“额外请求参数”必须是 JSON 对象，例如 {\"enable_thinking\": false}") from None
        self._use_temperature = True
        self._token_param = "max_tokens"
        self._stream = True
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, headers=headers,
                                    timeout=httpx.Timeout(connect=6, read=timeout, write=15, pool=5))

    # ── public ───────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Establish the connection (TLS) and, for local servers, make them load the model into memory."""
        # A separate long-timeout client: a cold local model can take a minute to load.
        with httpx.Client(base_url=self.base_url, headers=self._client.headers,
                          timeout=httpx.Timeout(connect=6, read=180, write=15, pool=5)) as slow:
            self._once("Hello.", "en", "zh-Hans", [], None, None, max_tokens=8, client=slow)

    def translate(self, text: str, src: str, tgt: str, *, context: Sequence[tuple[str, str]] = (),
                  on_delta: DeltaCallback | None = None, cancel: threading.Event | None = None) -> str:
        text = text.strip()
        if not text:
            return ""
        out = self._once(text, src, tgt, context, on_delta, cancel)
        if cancel is not None and cancel.is_set():
            return out
        if not out or not looks_like_language(out, tgt, text):
            log.info("suspicious translation %r → retrying without context", out[:60])
            retry = self._once(text, src, tgt, [], on_delta, cancel)
            out = retry or out
        return out

    def correct(self, text: str, lang: str, *, context: Sequence[tuple[str, str]] = ()) -> str:
        text = text.strip()
        if not text:
            return text
        messages = build_correction_messages(text, lang, context, self._gloss, self._topic)
        limit = max(32, min(512, int(len(text) * 2) + 32))
        try:
            out = self._complete(messages, limit)
        except TranslateError:
            log.info("source correction failed; keeping the original text", exc_info=True)
            return text
        out = out.strip()
        # same sanity check translate() uses: a corrupted or empty reply must never replace good text
        return out if out and looks_like_language(out, lang, text) else text

    def close(self) -> None:
        self._client.close()

    # ── request machinery ────────────────────────────────────────────────────

    def _body(self, messages: list[dict], max_tokens: int) -> dict:
        body: dict = {"model": self.model, "messages": messages, "stream": self._stream,
                      self._token_param: max_tokens}
        if self._use_temperature:
            body["temperature"] = 0
        body.update(self._extra)
        return body

    def _adapt(self, error_text: str, body: dict) -> bool:
        """React to a 400 by removing whatever the server said it doesn't support. True if something changed."""
        low = error_text.lower()
        changed = False
        if "temperature" in low and self._use_temperature:
            self._use_temperature, changed = False, True
        if "max_completion_tokens" in low and self._token_param == "max_tokens":
            self._token_param, changed = "max_completion_tokens", True
        elif "max_tokens" in low and self._token_param == "max_completion_tokens":
            self._token_param, changed = "max_tokens", True
        if "stream" in low and self._stream:
            self._stream, changed = False, True
        if self._extra and any(k in low for k in ("unrecognized", "unknown", "unexpected", "not support",
                                                   "extra", "invalid", "reasoning", "think")):
            self._extra, changed = {}, True
        return changed

    def _once(self, text: str, src: str, tgt: str, context, on_delta, cancel,
              max_tokens: int | None = None, client: httpx.Client | None = None) -> str:
        messages = build_messages(text, src, tgt, context, self._gloss, self._topic)
        limit = max_tokens or max(64, min(1024, int(len(text) * 3) + 48))
        return self._complete(messages, limit, on_delta, cancel, client)

    def _complete(self, messages: list[dict], max_tokens: int, on_delta=None, cancel=None,
                  client: httpx.Client | None = None) -> str:
        for _ in range(4):
            body = self._body(messages, max_tokens)
            try:
                return self._send(body, on_delta, cancel, client or self._client)
            except _BadRequest as e:
                if not self._adapt(e.text, body):
                    raise TranslateError(f"翻译服务拒绝了请求：{e.text[:240]}") from None
                log.info("adapted request after 400: %s", e.text[:120])
        raise TranslateError("翻译服务参数协商失败")

    def _send(self, body: dict, on_delta, cancel, client: httpx.Client) -> str:
        try:
            with client.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    self._raise_for(r)
                ctype = r.headers.get("content-type", "")
                if "text/event-stream" not in ctype:
                    r.read()
                    msg = r.json()["choices"][0]["message"]["content"] or ""
                    out = clean_translation(msg)
                    if on_delta and out:
                        on_delta(out)
                    return out
                acc, last = "", ""
                for line in r.iter_lines():
                    if cancel is not None and cancel.is_set():
                        break                                   # leaving the block closes the connection
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        acc += delta
                        shown = clean_translation(acc)
                        if on_delta and shown and shown != last:
                            last = shown
                            on_delta(shown)
                return clean_translation(acc)
        except httpx.ConnectError as e:
            raise TranslateError(f"无法连接翻译服务 {self.base_url}：{e}") from e
        except httpx.TimeoutException as e:
            raise TranslateError("翻译服务响应超时") from e
        except httpx.HTTPError as e:
            raise TranslateError(f"翻译请求失败：{e}") from e

    def _raise_for(self, r: httpx.Response) -> None:
        text = r.text
        if r.status_code in (400, 422):
            raise _BadRequest(text)
        if r.status_code in (401, 403):
            raise TranslateError("翻译 API Key 无效或无权限")
        if r.status_code == 402:
            raise TranslateError("翻译 API 账户余额不足（HTTP 402），请到服务商控制台充值，或换用免费/本地模型")
        if r.status_code == 404:
            raise TranslateError(f"翻译接口或模型不存在（HTTP 404）：{text[:160]}")
        if r.status_code == 429:
            raise TranslateError("翻译 API 触发限流（HTTP 429），请稍后再试或降低频率")
        raise TranslateError(f"翻译服务错误（HTTP {r.status_code}）：{text[:160]}")


class _BadRequest(Exception):
    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


# Providers list every model they host (embeddings, rerankers, image/speech models ...). Only chat models are useful
# for translation, so drop the rest. Short names must match a whole token ("tts", "wan"), long ones may match anywhere.
_NON_CHAT_TOKENS = {"tts", "wan", "flux", "sdxl", "kolors", "bge", "gte", "dall", "moderation"}
_NON_CHAT_PARTS = ("embed", "rerank", "whisper", "sensevoice", "cosyvoice", "diffusion", "fish-speech", "text2video")


def is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    tokens = set(re.split(r"[/_\-.: ]+", low))
    return not (tokens & _NON_CHAT_TOKENS or any(p in low for p in _NON_CHAT_PARTS))


def list_models(base_url: str, api_key: str = "", timeout: float = 8.0) -> list[str]:
    """``GET /models``: what the UI offers in the model drop-down. Returns chat-capable model ids, sorted."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # SiliconFlow can filter server-side (95 models -> 64 chat models); other providers ignore unknown params.
    params = {"sub_type": "chat"} if "siliconflow" in base_url.lower() else None
    try:
        r = httpx.get(base_url.rstrip("/") + "/models", headers=headers, params=params, timeout=timeout)
        if r.status_code in (401, 403):
            raise TranslateError("API Key 无效或无权限，无法获取模型列表")
        r.raise_for_status()
        ids = sorted({m["id"] for m in r.json().get("data", [])})
    except TranslateError:
        raise
    except (httpx.HTTPError, ValueError, KeyError) as e:
        raise TranslateError(f"无法获取模型列表：{e}") from e
    return [i for i in ids if is_chat_model(i)]
