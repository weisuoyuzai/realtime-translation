"""Model loading strategy of the local recognizer, with a fake faster_whisper (no model, no GPU)."""
import sys
import types

import numpy as np
import pytest

from live_translator.asr.base import AsrError
from live_translator.asr.local import LocalWhisper
from live_translator.config import AsrCfg


class FakeSegment:
    def __init__(self, text, avg=-0.2, nsp=0.01, cr=1.2):
        self.text, self.avg_logprob, self.no_speech_prob, self.compression_ratio = text, avg, nsp, cr


class FakeInfo:
    language, language_probability = "en", 0.97


def install_fake(monkeypatch, cached: bool, fail_cuda: bool = False):
    calls = []

    class FakeModel:
        def __init__(self, name, device="cpu", compute_type="int8", cpu_threads=0, local_files_only=False, **kw):
            calls.append({"name": name, "device": device, "local_only": local_files_only})
            if local_files_only and not cached:
                raise OSError("not in local cache")
            if device == "cuda" and fail_cuda:
                raise RuntimeError("cublas64_12.dll not found")

        def transcribe(self, audio, **kw):
            self.kw = kw
            return iter([FakeSegment(" Hello there. "), FakeSegment("Thanks for watching!")]), FakeInfo()

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return calls


def load(cfg=None, **kw):
    rec = LocalWhisper(cfg or AsrCfg(model="small", device="cpu", **kw))
    msgs = []
    rec.load(msgs.append)
    return rec, msgs


def test_cached_model_loads_without_touching_the_network(monkeypatch):
    calls = install_fake(monkeypatch, cached=True)
    _rec, msgs = load()
    assert [c["local_only"] for c in calls] == [True]
    assert not any("下载" in m for m in msgs)


def test_uncached_model_falls_back_to_download(monkeypatch):
    calls = install_fake(monkeypatch, cached=False)
    _rec, msgs = load()
    assert [c["local_only"] for c in calls] == [True, False]
    assert any("下载" in m for m in msgs)


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    calls = install_fake(monkeypatch, cached=True, fail_cuda=True)
    rec, msgs = load(AsrCfg(model="small", device="cuda"))
    assert rec.device == "cpu" and calls[-1]["device"] == "cpu"
    assert any("GPU 不可用" in m for m in msgs) and "cpu" in rec.name


def test_total_failure_is_reported_as_asr_error(monkeypatch):
    install_fake(monkeypatch, cached=False)

    class Boom(Exception):
        pass

    sys.modules["faster_whisper"].WhisperModel.__init__ = lambda self, *a, **k: (_ for _ in ()).throw(Boom("no disk"))
    with pytest.raises(AsrError, match="加载 Whisper 模型失败"):
        load()


def test_transcribe_filters_hallucinated_segments_and_reports_language(monkeypatch):
    install_fake(monkeypatch, cached=True)
    rec, _ = load()
    res = rec.transcribe(np.zeros(16000, dtype=np.float32), None, final=True)
    assert res.text == "Hello there." and res.language == "en" and res.language_prob > 0.9


def test_confidence_reflects_the_worst_kept_segments_avg_logprob(monkeypatch):
    install_fake(monkeypatch, cached=True)
    rec, _ = load()
    rec._model.transcribe = lambda audio, **kw: (
        iter([FakeSegment("Hello", avg=-0.2), FakeSegment("there", avg=-1.0)]), FakeInfo())
    res = rec.transcribe(np.zeros(16000, dtype=np.float32), None, final=True)
    assert res.confidence == pytest.approx((-1.0 + 1.6) / 1.6)                  # dragged down by the worse segment

    rec._model.transcribe = lambda audio, **kw: (iter([]), FakeInfo())          # nothing kept: no signal either way
    assert rec.transcribe(np.zeros(16000, dtype=np.float32), None, final=True).confidence == 1.0


def test_final_uses_beam_and_partial_is_greedy_and_prompt_only_for_capable_models(monkeypatch):
    install_fake(monkeypatch, cached=True)
    rec, _ = load(AsrCfg(model="small", device="cpu", beam_final=4))
    rec.transcribe(np.zeros(16000, dtype=np.float32), "en", final=True, prompt="Kubernetes")
    assert rec._model.kw["beam_size"] == 4 and rec._model.kw["initial_prompt"] == "Kubernetes"
    assert rec._model.kw["language"] == "en" and rec._model.kw["condition_on_previous_text"] is False
    rec.transcribe(np.zeros(16000, dtype=np.float32), None, final=False, prompt="Kubernetes")
    assert rec._model.kw["beam_size"] == 1 and rec._model.kw["language"] is None

    tiny, _ = load(AsrCfg(model="tiny", device="cpu"))
    tiny.transcribe(np.zeros(16000, dtype=np.float32), "en", final=True, prompt="Kubernetes")
    assert tiny._model.kw["initial_prompt"] is None                     # tiny models parrot the prompt back


def test_download_progress_reports_bytes_and_percentage(monkeypatch, tmp_path):
    import time
    from live_translator import download, runtime

    monkeypatch.setattr(runtime, "_models_dir", str(tmp_path))
    monkeypatch.setattr(download, "_expected_bytes", lambda repo, patterns: 4 * 1024 ** 2)
    blobs = tmp_path / "models--Systran--faster-whisper-large-v3" / "blobs"
    blobs.mkdir(parents=True)
    msgs: list[str] = []
    with download.report_download("Systran/faster-whisper-large-v3", "large-v3", msgs.append, interval=0.05):
        (blobs / "abc.incomplete").write_bytes(b"x" * 2 * 1024 ** 2)
        time.sleep(0.4)
    assert any("large-v3" in m and "2 MB / 4 MB" in m and "50%" in m for m in msgs), msgs
