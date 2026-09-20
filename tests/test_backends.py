"""LLM translator and remote ASR against a local fake OpenAI server."""
import threading
import time

import numpy as np
import pytest

from live_translator.asr.base import AsrError
from live_translator.asr.remote import RemoteWhisper
from live_translator.config import AsrCfg
from live_translator.translate import TranslateError
from live_translator.translate.deepl import DeepLTranslator
from live_translator.translate.llm import LLMTranslator, default_extra_body, is_chat_model, list_models


def make(server, **kw):
    return LLMTranslator(server.base_url, "key", "test-model", **kw)


# ── LLM translator ───────────────────────────────────────────────────────────

def test_streams_deltas_and_returns_full_text(server):
    seen = []
    out = make(server).translate("Hello world", "en", "zh-Hans", on_delta=seen.append)
    assert out == "译:Hello world"
    assert len(seen) >= 2 and seen[-1] == out                       # progressive, accumulated, ends complete
    assert all(b.startswith(a) for a, b in zip(seen, seen[1:]))


def test_sends_context_as_chat_history_and_auth_header(server):
    make(server, glossary="GPU = 显卡", topic="hardware").translate(
        "It is fast.", "en", "zh-Hans", context=[("Hi.", "你好。")])
    req = server.chat_requests[-1]
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "GPU → 显卡" in req["messages"][0]["content"]
    assert req["stream"] is True and req["temperature"] == 0


def test_adapts_when_server_rejects_temperature(server):
    server.reject_temperature = True                                  # e.g. OpenAI reasoning models
    t = make(server)
    assert t.translate("Hi", "en", "zh-Hans") == "译:Hi"
    assert "temperature" not in server.chat_requests[-1]
    n = len(server.chat_requests)
    t.translate("Again", "en", "zh-Hans")
    assert len(server.chat_requests) == n + 1                         # remembered: no second failed attempt


def test_drops_unsupported_extra_params(server):
    server.reject_extra = True
    t = make(server, extra_body='{"enable_thinking": false}')
    assert t.translate("Hi", "en", "zh-Hans") == "译:Hi"
    assert "enable_thinking" not in server.chat_requests[-1]


def test_user_extra_body_is_sent(server):
    make(server, extra_body='{"top_p": 0.5}').translate("Hi", "en", "zh-Hans")
    assert server.chat_requests[-1]["top_p"] == 0.5


def test_invalid_extra_body_is_reported():
    with pytest.raises(TranslateError, match="JSON"):
        LLMTranslator("http://x/v1", "", "m", extra_body="[1,2]")


def test_retries_once_without_context_when_model_echoes_source(server):
    calls = []

    def reply(user_text, n):
        calls.append(n)
        return "Hello there, how are you" if n == 1 else "你好，你好吗"      # 1st: echoed English

    server.reply = reply
    out = make(server).translate("Hello there, how are you", "en", "zh-Hans", context=[("a", "b")])
    assert out == "你好，你好吗" and calls == [1, 2]
    assert len(server.chat_requests[1]["messages"]) == 2              # retry carried no context


def test_cancel_stops_streaming_early(server):
    server.chunk_delay = 0.15
    server.reply = lambda u, n: "一二三四五六七八九十" * 3
    cancel, seen = threading.Event(), []

    def on_delta(acc):
        seen.append(acc)
        cancel.set()                                                   # cancel after the first delta

    t0 = time.monotonic()
    make(server).translate("x", "en", "zh-Hans", on_delta=on_delta, cancel=cancel)
    assert time.monotonic() - t0 < 1.0 and len(seen) == 1


def test_error_messages_are_actionable(server):
    server.chat_status = 401
    with pytest.raises(TranslateError, match="API Key"):
        make(server).translate("Hi", "en", "zh-Hans")
    server.chat_status = 402                                            # e.g. SiliconFlow with an empty balance
    with pytest.raises(TranslateError, match="余额不足"):
        make(server).translate("Hi", "en", "zh-Hans")
    server.chat_status = 429
    with pytest.raises(TranslateError, match="限流"):
        make(server).translate("Hi", "en", "zh-Hans")
    with pytest.raises(TranslateError, match="无法连接"):
        LLMTranslator("http://127.0.0.1:9/v1", "", "m").translate("Hi", "en", "zh-Hans")


def test_warmup_and_list_models(server):
    make(server).warmup()
    assert server.chat_requests[-1]["max_tokens"] == 8
    assert list_models(server.base_url) == ["m-a", "m-b"]


def test_is_chat_model_filters_non_chat_models():
    for ok in ("Qwen/Qwen3.6-35B-A3B", "deepseek-ai/DeepSeek-V3", "gpt-4o-mini", "qwen2.5:7b", "THUDM/glm-4-9b-chat",
               "meta-llama/Llama-3.3-70B-Instruct", "google/gemma-3-27b-it"):
        assert is_chat_model(ok), ok
    for bad in ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-8B", "BAAI/bge-reranker-v2-m3", "black-forest-labs/FLUX.1-dev",
                "FunAudioLLM/SenseVoiceSmall", "FunAudioLLM/CosyVoice2-0.5B", "openai/whisper-large-v3", "tts-1",
                "Wan-AI/Wan2.1-T2V-14B", "stabilityai/stable-diffusion-3-5-large", "text-embedding-3-small"):
        assert not is_chat_model(bad), bad


def test_list_models_drops_non_chat_and_dedupes(server):
    server.models = ["Qwen/Qwen3.6-35B-A3B", "BAAI/bge-m3", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3.6-35B-A3B", "FLUX.1-dev"]
    assert list_models(server.base_url) == ["Qwen/Qwen3.6-35B-A3B", "deepseek-ai/DeepSeek-V3"]


def test_list_models_asks_siliconflow_for_chat_models_only(monkeypatch):
    import httpx
    seen = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.update(url=url, params=params, auth=headers.get("Authorization"))
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.6-35B-A3B"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert list_models("https://api.siliconflow.cn/v1", "sk-x") == ["Qwen/Qwen3.6-35B-A3B"]
    assert seen == {"url": "https://api.siliconflow.cn/v1/models", "params": {"sub_type": "chat"}, "auth": "Bearer sk-x"}
    list_models("https://api.openai.com/v1", "sk-x")
    assert seen["params"] is None


def test_list_models_errors(server):
    server.models_status = 401
    with pytest.raises(TranslateError, match="API Key"):
        list_models(server.base_url, "bad")
    server.models_status = 500
    with pytest.raises(TranslateError, match="无法获取"):
        list_models(server.base_url)
    with pytest.raises(TranslateError, match="无法获取"):
        list_models("http://127.0.0.1:9/v1")


def test_default_extra_body_switches_off_thinking_per_provider():
    assert default_extra_body("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-32b") == {"enable_thinking": False}
    assert default_extra_body("http://127.0.0.1:11434/v1", "qwen3:8b") == {"reasoning_effort": "none"}
    assert default_extra_body("http://127.0.0.1:11434/v1", "qwen2.5:7b") == {}
    assert default_extra_body("https://api.openai.com/v1", "gpt-4o-mini") == {}
    assert default_extra_body("https://api.siliconflow.cn/v1", "Qwen/Qwen3.6-35B-A3B") == {"enable_thinking": False}


# ── DeepL (against the fake server's 404 → error path; payload checked separately) ──

def test_deepl_requires_key_and_maps_targets():
    with pytest.raises(TranslateError):
        DeepLTranslator("")
    d = DeepLTranslator("k:fx", base_url="http://127.0.0.1:9")
    with pytest.raises(TranslateError, match="不支持"):
        d.translate("hi", "en", "th")


# ── remote ASR ───────────────────────────────────────────────────────────────

def remote(server, **kw):
    cfg = AsrCfg(mode="remote", remote_base_url=server.base_url, remote_api_key="k", remote_model="whisper-1", **kw)
    r = RemoteWhisper(cfg)
    r.load()
    return r


AUDIO = (np.random.randn(16000) * 0.1).astype(np.float32)


def test_remote_asr_parses_text_and_language(server):
    res = remote(server).transcribe(AUDIO, None, final=True)
    assert res.text == "hello world" and res.language == "en"
    body = server.asr_requests[-1]
    assert b"verbose_json" in body and b"whisper-1" in body and b"audio.wav" in body


def test_remote_asr_sends_language_hint_and_prompt(server):
    remote(server).transcribe(AUDIO, "zh-Hans", final=True, prompt="Kubernetes")
    body = server.asr_requests[-1]
    assert b'name="language"\r\n\r\nzh' in body and b"Kubernetes" in body


def test_remote_asr_falls_back_to_json_format(server):
    server.reject_verbose_json = True                                   # e.g. gpt-4o-transcribe
    server.asr_reply = {"text": "plain result"}
    r = remote(server)
    res = r.transcribe(AUDIO, None, final=True)
    assert res.text == "plain result" and res.language == ""
    assert b"verbose_json" not in server.asr_requests[-1]


def test_remote_asr_filters_hallucinated_segments(server):
    server.asr_reply = {"text": "x", "language": "en", "segments": [
        {"text": "Real speech here", "avg_logprob": -0.3, "no_speech_prob": 0.02, "compression_ratio": 1.2},
        {"text": "Thanks for watching!", "avg_logprob": -0.4, "no_speech_prob": 0.4, "compression_ratio": 1.0}]}
    assert remote(server).transcribe(AUDIO, None, final=True).text == "Real speech here"


def test_remote_asr_insufficient_balance(server):
    server.asr_status = 402
    with pytest.raises(AsrError, match="余额不足"):
        remote(server).transcribe(AUDIO, None, final=True)


def test_remote_asr_auth_and_missing_url(server):
    server.asr_status = 401
    with pytest.raises(AsrError, match="Key"):
        remote(server).transcribe(AUDIO, None, final=True)
    with pytest.raises(AsrError, match="地址"):
        RemoteWhisper(AsrCfg(mode="remote", remote_base_url="")).load()


def test_remote_partials_flag():
    assert RemoteWhisper(AsrCfg(mode="remote")).supports_partials is False
    assert RemoteWhisper(AsrCfg(mode="remote", remote_partials=True)).supports_partials is True
