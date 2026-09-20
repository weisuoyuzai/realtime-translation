"""Models directory management, process-wide settings (proxy / mirror / models dir) and their wiring."""
import os
import sys
import types

import pytest

from live_translator import runtime, storage
from live_translator.config import AppConfig


def make_model(root, repo, size):
    """Create a fake hub-cache entry: models--org--name/{blobs/<file>, snapshots/<rev>/model.bin, refs/main}."""
    d = root / ("models--" + repo.replace("/", "--"))
    (d / "blobs").mkdir(parents=True)
    (d / "blobs" / "abc").write_bytes(b"x" * size)
    (d / "snapshots" / "rev1").mkdir(parents=True)
    (d / "snapshots" / "rev1" / "model.bin").write_bytes(b"x" * size)      # a real copy (no symlink privilege needed)
    (d / "refs").mkdir()
    (d / "refs" / "main").write_text("rev1")
    return d


# ── storage ──────────────────────────────────────────────────────────────────

def test_repo_id_from_dirname():
    assert storage.repo_id_from_dirname("models--Systran--faster-whisper-small") == "Systran/faster-whisper-small"
    assert storage.repo_id_from_dirname("models--gpt2") == "gpt2"
    assert storage.repo_id_from_dirname(".locks") is None
    assert storage.repo_id_from_dirname("datasets--a--b") is None


def test_list_models_sizes_and_ordering(tmp_path):
    make_model(tmp_path, "Systran/faster-whisper-small", 1000)
    make_model(tmp_path, "JustFrederik/nllb-200-distilled-600M-ct2-int8", 5000)
    (tmp_path / ".locks").mkdir()
    (tmp_path / "notes.txt").write_text("hi")
    models = storage.list_models(str(tmp_path))
    assert [m.repo_id for m in models] == ["JustFrederik/nllb-200-distilled-600M-ct2-int8", "Systran/faster-whisper-small"]
    assert models[0].size >= 10000 and models[1].size >= 2000           # blob + snapshot copy
    assert storage.list_models(str(tmp_path / "missing")) == []


def test_delete_model(tmp_path):
    make_model(tmp_path, "a/b", 10)
    [m] = storage.list_models(str(tmp_path))
    storage.delete_model(m)
    assert storage.list_models(str(tmp_path)) == []


def test_move_models_moves_and_never_overwrites(tmp_path):
    src, dst = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    make_model(src, "a/one", 100)
    make_model(src, "a/two", 200)
    make_model(dst, "a/two", 1)                                          # already there, must survive untouched
    msgs = []
    moved = storage.move_models(str(src), str(dst), msgs.append)
    assert moved == ["a/one"]
    assert {m.repo_id for m in storage.list_models(str(dst))} == {"a/one", "a/two"}
    assert [m.repo_id for m in storage.list_models(str(src))] == ["a/two"]      # skipped one is left in place
    assert (dst / "models--a--two" / "blobs" / "abc").read_bytes() == b"x"      # not overwritten
    assert any("跳过" in m for m in msgs)


def test_check_writable(tmp_path):
    assert storage.check_writable(str(tmp_path / "new" / "deep")) is None        # created on demand
    assert (tmp_path / "new" / "deep").is_dir()
    f = tmp_path / "afile"
    f.write_text("x")
    assert "无法使用该目录" in storage.check_writable(str(f))                   # a file, not a folder


def test_human_size_and_same_dir(tmp_path):
    assert storage.human_size(512) == "512 B"
    assert storage.human_size(2048) == "2 KB"
    assert storage.human_size(5 * 1024 ** 3) == "5.0 GB"
    assert storage.same_dir(str(tmp_path), str(tmp_path / "."))
    assert not storage.same_dir(str(tmp_path), str(tmp_path / "other"))


# ── runtime settings ─────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch):
    for v in runtime._PROXY_VARS + ("NO_PROXY", "HF_HUB_CACHE", "HF_HOME", "HF_ENDPOINT"):
        monkeypatch.delenv(v, raising=False)
    runtime._proxy_vars_we_set.clear()
    yield
    runtime._apply_proxy("")                                             # restore anything we set
    runtime.set_models_dir("")


def test_default_models_dir_honours_hf_variables(clean_env, monkeypatch, tmp_path):
    assert runtime.default_models_dir().endswith(os.path.join(".cache", "huggingface", "hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert runtime.default_models_dir() == str(tmp_path / "hub")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "custom"))
    assert runtime.default_models_dir() == str(tmp_path / "custom")


def test_models_dir_setting(clean_env, tmp_path):
    assert runtime.models_dir() is None
    runtime.set_models_dir(f"  {tmp_path}  ")
    assert runtime.models_dir() == str(tmp_path) and runtime.effective_models_dir() == str(tmp_path)
    runtime.set_models_dir("")
    assert runtime.models_dir() is None


def test_proxy_is_applied_bypasses_loopback_and_restored(clean_env, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:8080")                 # the user's own setting
    cfg = AppConfig()
    cfg.proxy = "http://127.0.0.1:7890"
    runtime.apply_settings(cfg)
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890" and os.environ["all_proxy"] == "http://127.0.0.1:7890"
    assert {"127.0.0.1", "localhost"} <= set(os.environ["NO_PROXY"].split(","))
    cfg.proxy = ""
    runtime.apply_settings(cfg)
    assert os.environ["HTTPS_PROXY"] == "http://corp:8080"                # the user's value comes back
    assert "all_proxy" not in os.environ and "HTTP_PROXY" not in os.environ


def test_empty_proxy_never_touches_the_users_environment(clean_env, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://theirs:1")
    runtime.apply_settings(AppConfig())
    assert os.environ["HTTP_PROXY"] == "http://theirs:1"


def test_mirror_and_models_dir_applied_together(clean_env, tmp_path):
    cfg = AppConfig()
    cfg.hf_endpoint, cfg.models_dir = "https://hf-mirror.com/", str(tmp_path)
    runtime.apply_settings(cfg)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com" and runtime.models_dir() == str(tmp_path)


def test_config_roundtrip_of_new_fields(tmp_path):
    from live_translator.config import load_config, save_config
    cfg = AppConfig()
    cfg.models_dir, cfg.transcripts_dir, cfg.proxy = "D:/models", "D:/subs", "socks5://127.0.0.1:1080"
    save_config(cfg, tmp_path / "c.json")
    back = load_config(tmp_path / "c.json")
    assert (back.models_dir, back.transcripts_dir, back.proxy) == ("D:/models", "D:/subs", "socks5://127.0.0.1:1080")


# ── wiring into the loaders ──────────────────────────────────────────────────

def test_whisper_loader_uses_the_configured_models_dir(clean_env, monkeypatch, tmp_path):
    import numpy as np
    from live_translator.asr.local import LocalWhisper
    from live_translator.config import AsrCfg

    seen = []

    class FakeModel:
        def __init__(self, name, **kw):
            seen.append(kw)

        def transcribe(self, audio, **kw):
            return iter([]), types.SimpleNamespace(language="en", language_probability=1.0)

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    runtime.set_models_dir(str(tmp_path))
    LocalWhisper(AsrCfg(model="small", device="cpu")).load()
    assert seen[-1]["download_root"] == str(tmp_path)
    runtime.set_models_dir("")
    LocalWhisper(AsrCfg(model="small", device="cpu")).load()
    assert seen[-1]["download_root"] is None                              # library default cache


def test_nllb_loader_uses_the_configured_models_dir(clean_env, monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from live_translator.translate.nllb import NllbTranslator

    calls = []

    def fake_snapshot(repo, **kw):
        calls.append(kw)
        raise OSError("stop here")                                        # we only care about the arguments

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    runtime.set_models_dir(str(tmp_path))
    with pytest.raises(Exception):
        NllbTranslator("org/model").warmup()
    assert calls and all(c["cache_dir"] == str(tmp_path) for c in calls)
    assert calls[0]["local_files_only"] is True                           # offline-first, then a download attempt


def test_transcript_writer_honours_the_directory(tmp_path):
    from live_translator.models import Line
    from live_translator.pipeline import _TranscriptWriter

    w = _TranscriptWriter(True, str(tmp_path / "subs" / "deep"))
    w.write(Line(1, src="Hello", dst="你好", src_lang="en"))
    w.close()
    [f] = list((tmp_path / "subs" / "deep").glob("*.txt"))
    assert "Hello" in f.read_text(encoding="utf-8") and "你好" in f.read_text(encoding="utf-8")
    assert _TranscriptWriter(False, str(tmp_path / "nothing")).path is None


def test_configured_proxy_reaches_httpx_clients_and_loopback_bypasses_it(clean_env):
    """What the user sets in Preferences must really route API calls through the proxy - but never local servers."""
    import httpx
    cfg = AppConfig()
    cfg.proxy = "http://10.9.8.7:7890"
    runtime.apply_settings(cfg)
    with httpx.Client() as c:
        remote = c._transport_for_url(httpx.URL("https://api.siliconflow.cn/v1/models"))
        local_llm = c._transport_for_url(httpx.URL("http://127.0.0.1:11434/v1/models"))
        localhost = c._transport_for_url(httpx.URL("http://localhost:1234/v1/models"))
        assert remote is not c._transport                                   # goes through the proxy transport
        assert local_llm is c._transport and localhost is c._transport      # Ollama / LM Studio: direct
    cfg.proxy = ""
    runtime.apply_settings(cfg)
    with httpx.Client() as c:
        assert c._transport_for_url(httpx.URL("https://api.siliconflow.cn/v1/models")) is c._transport
