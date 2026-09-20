"""Pipeline behaviour with fake recognizer / translator / audio source (no models, no GPU)."""
import threading
import time

import numpy as np
import pytest

import live_translator.pipeline as pl
from live_translator.asr.base import AsrError, AsrResult, Recognizer
from live_translator.audio.base import AudioSource
from live_translator.config import AppConfig
from live_translator.translate.base import TranslateError, Translator


def fake_vad(frame):
    return float(frame[0])


class FakeVAD:
    def __call__(self, frame):
        return fake_vad(frame)


class ScriptedSource(AudioSource):
    """Plays [(seconds, level)] segments, level 0.9 = speech / 0 = silence, at high speed."""
    name = "scripted"

    def __init__(self, script, stall_after=False):
        self.script, self.stall_after = script, stall_after
        self.done = threading.Event()

    def start(self, on_audio):
        def run():
            for secs, level in self.script:
                for _ in range(int(secs * 10)):
                    on_audio(np.full(1600, level, dtype=np.float32), 16000)
                    time.sleep(0.004)
            self.done.set()
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        pass


class FakeRecognizer(Recognizer):
    name = "fake-asr"

    def __init__(self, lang="en", texts=None, delay=0.0):
        self.lang, self.texts, self.delay = lang, texts or {}, delay
        self.calls = []                       # (final, language_arg)
        self.n = 0

    def transcribe(self, audio, language, *, final, prompt=""):
        self.calls.append((final, language))
        time.sleep(self.delay)
        lang = self.lang.pop(0) if isinstance(self.lang, list) else self.lang
        if not final:
            return AsrResult("partial text here", language or lang, 0.95, 5)
        self.n += 1                           # counts finals only
        return AsrResult(self.texts.get(self.n, f"sentence {self.n}"), language or lang, 0.95, 5)


class FakeTranslator(Translator):
    name = "fake-tr"
    streaming = True

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []                       # (text, src, tgt, context)

    def translate(self, text, src, tgt, *, context=(), on_delta=None, cancel=None):
        self.calls.append((text, src, tgt, list(context)))
        out = "译:" + text
        for i in range(2, len(out) + 2, max(1, len(out) // 3)):
            if cancel is not None and cancel.is_set():
                return out[:i]
            if on_delta:
                on_delta(out[:i])
            time.sleep(self.delay)
        return out


class Harness:
    def __init__(self, monkeypatch, rec, tr, source, cfg=None):
        self.lines, self.statuses, self.updates = {}, [], []
        self.lock = threading.Lock()
        self.cfg = cfg or AppConfig()
        self.cfg.save_transcript = False
        self.cfg.lang.target = "zh-Hans"
        self.rec, self.tr = rec, tr
        monkeypatch.setattr(pl, "create_recognizer", lambda c: rec)
        monkeypatch.setattr(pl, "create_translator", lambda c, p=None: tr)
        monkeypatch.setattr(pl, "SileroVAD", FakeVAD)
        self.pipe = pl.Pipeline(self.cfg, self.on_line, lambda lv, m: self.statuses.append((lv, m)), source=source)

    def on_line(self, line):
        with self.lock:
            self.updates.append(line)
            if line.removed:
                self.lines.pop(line.id, None)
            else:
                self.lines[line.id] = line

    def wait(self, cond, timeout=8.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if cond():
                    return True
            time.sleep(0.02)
        return False

    def __enter__(self):
        self.pipe.start()
        return self

    def __exit__(self, *a):
        self.pipe.stop()


def one_sentence():
    return ScriptedSource([(0.3, 0.0), (2.5, 0.9), (1.5, 0.0)])


def test_full_flow_streams_source_then_translation(monkeypatch):
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(delay=0.01), one_sentence()) as h:
        assert h.wait(lambda: any(l.dst_final for l in h.lines.values()))
        line = next(iter(h.lines.values()))
        assert line.src == "sentence " + str(h.rec.n) and line.src_final
        assert line.dst == "译:" + line.src and line.dst_final and not line.dst_draft
        assert line.src_lang == "en" and line.asr_ms > 0 and line.tr_ms > 0
        assert line.latency_ms > 0
        # interim recognition was shown before the final, and the translation streamed in pieces
        assert any(not u.src_final and u.src == "partial text here" for u in h.updates)
        assert len({u.dst for u in h.updates if u.dst and not u.dst_final}) >= 1
        assert h.statuses[-1][0] == "ready" and "fake-asr" in h.statuses[-1][1]


def test_same_language_is_passed_through_untranslated(monkeypatch):
    tr = FakeTranslator()
    with Harness(monkeypatch, FakeRecognizer(lang="zh"), tr, one_sentence()) as h:
        assert h.wait(lambda: any(l.src_final for l in h.lines.values()))
        line = next(iter(h.lines.values()))
        assert line.dst_final and line.skipped and line.dst == ""
        assert tr.calls == []


def test_no_translator_means_transcription_only(monkeypatch):
    with Harness(monkeypatch, FakeRecognizer(), None, one_sentence()) as h:
        assert h.wait(lambda: any(l.src_final and l.dst_final for l in h.lines.values()))
        assert not next(iter(h.lines.values())).skipped


def test_previous_sentences_are_passed_as_context(monkeypatch):
    tr = FakeTranslator()
    src = ScriptedSource([(0.3, 0), (2.0, .9), (1.2, 0), (2.0, .9), (1.5, 0)])
    with Harness(monkeypatch, FakeRecognizer(), tr, src) as h:
        assert h.wait(lambda: len(tr.calls) >= 2 and sum(l.dst_final for l in h.lines.values()) >= 2)
        assert tr.calls[0][3] == []
        assert tr.calls[1][3] == [(tr.calls[0][0], "译:" + tr.calls[0][0])]


def test_empty_final_transcript_retracts_the_line(monkeypatch):
    rec = FakeRecognizer(texts={1: "", 2: ""})
    tr = FakeTranslator()
    with Harness(monkeypatch, rec, tr, one_sentence()) as h:
        assert h.wait(lambda: any(u.removed for u in h.updates))
        assert h.lines == {} and tr.calls == []


def test_draft_translation_shows_early_and_never_overwrites_final(monkeypatch):
    cfg = AppConfig()
    cfg.translate.draft = True
    rec, tr = FakeRecognizer(delay=0.02), FakeTranslator(delay=0.03)
    with Harness(monkeypatch, rec, tr, ScriptedSource([(0.3, 0), (3.5, .9), (1.5, 0)]), cfg) as h:
        assert h.wait(lambda: any(u.dst_draft for u in h.updates)), "a draft translation should appear while talking"
        assert h.wait(lambda: any(l.dst_final for l in h.lines.values()))
        time.sleep(0.4)                                            # let any straggling draft thread finish
        seen_final = False
        for u in h.updates:                                        # once final, a line must never regress to draft
            if u.dst_final:
                seen_final = True
            elif seen_final:
                pytest.fail(f"draft/stream update after final: {u}")
        assert next(iter(h.lines.values())).dst.startswith("译:")


def test_auto_language_is_sticky_and_low_confidence_outlier_is_redecoded(monkeypatch):
    class Rec(FakeRecognizer):
        def transcribe(self, audio, language, *, final, prompt=""):
            self.calls.append((final, language))
            if not final:
                return AsrResult("partial text here", language or "zh", 0.95, 1)
            if language is None:                                    # auto: 1,2 → zh (confident); 3 → 'cy' (unsure)
                self.n += 1
                lang, prob = ("zh", 0.97) if self.n <= 2 else ("cy", 0.5)
                return AsrResult(f"sentence {self.n}", lang, prob, 1)
            return AsrResult("redecoded", language, 1.0, 1)

    src = ScriptedSource([(0.3, 0)] + [(1.5, .9), (1.0, 0)] * 3 + [(1, 0)])
    with Harness(monkeypatch, Rec(), FakeTranslator(), src) as h:
        assert h.wait(lambda: sum(l.src_final for l in h.lines.values()) >= 3, timeout=10)
        third = sorted(h.lines.values(), key=lambda l: l.id)[2]
        assert third.src == "redecoded" and third.src_lang == "zh"
        assert (True, "zh") in h.rec.calls                          # decoded again with the established language


def test_source_that_stops_delivering_packets_still_finalises(monkeypatch):
    """Loopback devices go silent (no packets) when the app stops playing; the utterance must still end."""
    src = ScriptedSource([(0.3, 0), (2.0, 0.9)])                    # speech, then nothing at all
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), src) as h:
        assert h.wait(lambda: any(l.src_final for l in h.lines.values()), timeout=8)


def test_boot_failure_reports_error_and_stops(monkeypatch):
    class Bad(FakeRecognizer):
        def load(self, progress=None):
            raise AsrError("模型下载失败")

    with Harness(monkeypatch, Bad(), FakeTranslator(), one_sentence()) as h:
        assert h.wait(lambda: any(s[0] == "stopped" for s in h.statuses))
        assert ("error", "模型下载失败") in h.statuses


def test_translator_failure_is_reported_and_pipeline_keeps_running(monkeypatch):
    class Flaky(FakeTranslator):
        def translate(self, text, *a, **k):
            if not self.calls:
                self.calls.append(1)
                raise TranslateError("翻译 API Key 无效")
            return super().translate(text, *a, **k)

    src = ScriptedSource([(0.3, 0), (2.0, .9), (1.2, 0), (2.0, .9), (1.5, 0)])
    with Harness(monkeypatch, FakeRecognizer(), Flaky(), src) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)
        lines = sorted(h.lines.values(), key=lambda l: l.id)
        assert lines[0].error and lines[1].dst.startswith("译:")
        assert ("error", "翻译 API Key 无效") in h.statuses


def test_slow_recognition_drops_stale_partials_but_keeps_finals(monkeypatch):
    rec = FakeRecognizer(delay=0.4)                                 # much slower than real time
    with Harness(monkeypatch, rec, FakeTranslator(), ScriptedSource([(0.3, 0), (4.0, .9), (1.5, 0)])) as h:
        assert h.wait(lambda: any(l.src_final for l in h.lines.values()), timeout=15)
        partial_calls = sum(1 for f, _ in rec.calls if not f)
        assert partial_calls < 6                                    # ~6 were offered; the stale ones were skipped


def test_stop_is_prompt(monkeypatch):
    h = Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), one_sentence())
    h.pipe.start()
    assert h.wait(lambda: h.statuses and h.statuses[-1][0] == "ready")
    t0 = time.monotonic()
    h.pipe.stop()
    assert time.monotonic() - t0 < 3.0
    assert not [t for t in h.pipe._threads if t.is_alive()]
