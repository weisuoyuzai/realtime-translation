"""Speaker tracking: feature extraction, online clustering, and how it is wired into the pipeline (fake embedder)."""
import numpy as np
import pytest

import live_translator.pipeline as pl
from live_translator import speaker as sp
from live_translator.config import AppConfig, SpeakerCfg
from live_translator.speaker import SpeakerTracker

from test_pipeline import FakeRecognizer, FakeTranslator, Harness, ScriptedSource


def unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


A, B, C = unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)


def jitter(v, k=0.15, seed=0):
    r = np.random.default_rng(seed).normal(size=v.shape) * k
    x = v + r
    return x / np.linalg.norm(x)


# ── features ─────────────────────────────────────────────────────────────────

def test_fbank_has_80_bins_per_10ms_frame_and_is_mean_normalised():
    t = np.arange(16000 * 2) / 16000
    f = sp.fbank((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    assert f.shape == (198, 80) and np.isfinite(f).all()
    assert np.abs(f.mean(axis=0)).max() < 1e-4
    assert sp.fbank(np.zeros(100, dtype=np.float32)).shape == (1, 80)          # shorter than one frame: no crash


def test_mel_filters_are_triangles_covering_the_band():
    bank = sp._mel_bank()
    assert bank.shape == (80, 256) and (bank >= 0).all() and (bank.max(axis=1) > 0).all()
    assert bank.max() <= 1.0 + 1e-6


# ── clustering ───────────────────────────────────────────────────────────────

def test_same_voice_keeps_its_number_and_new_voices_get_new_numbers_in_order():
    t = SpeakerTracker(threshold=0.5)
    got = [t.assign(v, 3.0) for v in (A, jitter(A, seed=1), B, jitter(B, seed=2), jitter(A, seed=3), C, jitter(B, seed=4))]
    assert got == [1, 1, 2, 2, 1, 3, 2] and t.count == 3


def test_short_or_missing_voiceprints_reuse_the_previous_speaker_and_learn_nothing():
    t = SpeakerTracker()
    assert t.assign(None, 0.3) == 0                                  # nobody heard yet: unknown
    assert t.assign(A, 3.0) == 1 and t.assign(B, 3.0) == 2
    assert t.assign(A, 0.4) == 2                                     # too short to trust: still the last speaker
    assert t.assign(None, 5.0) == 2
    assert t.count == 2


def test_one_to_two_second_clips_join_a_known_speaker_but_never_create_or_teach():
    t = SpeakerTracker(threshold=0.55)
    D = unit(0, 0, 0, 1)
    assert t.assign(jitter(A, seed=1), 1.5) == 0 and t.count == 0     # nobody known yet: a short clip cannot start a speaker
    assert t.assign(A, 3.0) == 1 and t.assign(B, 3.0) == 2
    assert t.assign(jitter(A, 0.2, seed=2), 1.4) == 1                  # matches speaker 1 …
    assert t.assign(jitter(B, 0.2, seed=3), 1.4) == 2                  # … or 2
    assert t.assign(D, 1.4) == 2 and t.count == 2                      # unknown voice: keeps the previous speaker, no new one
    before = t._centroids().copy()
    t.assign(jitter(A, 0.2, seed=4), 1.9)
    assert np.allclose(before, t._centroids())                         # a short clip never changes the profiles


def test_longer_utterances_weigh_more_in_a_speakers_profile():
    t = SpeakerTracker(threshold=0.3)
    t.assign(A, 3.0)
    drift = unit(1, 0.9, 0, 0)
    t.assign(drift, 2.0)                                                # short-ish: small say
    short_pull = float(t._centroids()[0] @ drift)
    t2 = SpeakerTracker(threshold=0.3)
    t2.assign(A, 3.0)
    t2.assign(drift, 10.0)                                              # long: pulls the profile further
    assert float(t2._centroids()[0] @ drift) > short_pull


def test_max_speakers_folds_new_voices_into_the_closest_existing_one():
    t = SpeakerTracker(threshold=0.9, max_speakers=2)
    assert [t.assign(v, 3.0) for v in (A, B, C, unit(1, 0.2, 0, 0))] == [1, 2, 1 if C @ A >= C @ B else 2, 1]
    assert t.count == 2


def test_threshold_decides_how_easily_a_voice_is_split():
    near = unit(1, 0.6, 0, 0)                                        # cosine to A = 0.86
    assert SpeakerTracker(threshold=0.8).assign(A, 3) == 1
    strict, loose = SpeakerTracker(threshold=0.9), SpeakerTracker(threshold=0.7)
    for t in (strict, loose):
        t.assign(A, 3)
    assert strict.assign(near, 3) == 2 and loose.assign(near, 3) == 1


def test_ambiguous_flags_a_voiceprint_sitting_between_two_known_speakers():
    t = SpeakerTracker(threshold=0.55)
    t.assign(A, 3.0), t.assign(B, 3.0)
    blend = unit(1, 1, 0, 0)                                          # equidistant from A and B
    assert t.ambiguous(blend, 3.0) is True
    assert t.ambiguous(A, 3.0) is False                                # clearly one speaker: not ambiguous
    assert t.ambiguous(C, 3.0) is False                                # close to neither: a new voice, not a blend


def test_ambiguous_needs_two_known_speakers_and_a_long_enough_clip():
    t = SpeakerTracker(threshold=0.55)
    assert t.ambiguous(unit(1, 1, 0, 0), 3.0) is False                 # nobody known yet
    t.assign(A, 3.0)
    assert t.ambiguous(unit(1, 1, 0, 0), 3.0) is False                 # only one known speaker
    t.assign(B, 3.0)
    assert t.ambiguous(unit(1, 1, 0, 0), 1.5) is False                 # too short to trust
    assert t.ambiguous(None, 3.0) is False


def test_reset_starts_numbering_again():
    t = SpeakerTracker()
    t.assign(A, 3), t.assign(B, 3)
    t.reset()
    assert t.count == 0 and t.assign(B, 3) == 1


# ── pipeline wiring ──────────────────────────────────────────────────────────

class FakeEmbedder:
    """Voiceprint alternates A, B, A … per utterance (in the order they are recognised)."""

    def __init__(self):
        self.n = 0

    def __call__(self, audio):
        self.n += 1
        return A if self.n % 2 else B


def two_sentences():
    return ScriptedSource([(0.3, 0), (2.0, .9), (1.2, 0), (2.0, .9), (1.5, 0)])


def speaker_cfg(enabled=True) -> AppConfig:
    cfg = AppConfig()
    cfg.speaker = SpeakerCfg(enabled=enabled)
    return cfg


def test_lines_get_a_speaker_number_when_tracking_is_on(monkeypatch):
    monkeypatch.setattr(pl, "load_embedder", lambda progress=None, models_dir=None: FakeEmbedder())
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), two_sentences(), speaker_cfg()) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)
        ordered = [h.lines[i] for i in sorted(h.lines)]
        assert [l.speaker for l in ordered[:2]] == [1, 2]
        assert all(l.speaker == 0 for l in h.updates if not l.src_final)        # interim text carries no speaker yet


def test_no_tracking_no_speaker_and_no_model_load(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the model must not be loaded when the feature is off")
    monkeypatch.setattr(pl, "load_embedder", boom)
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), two_sentences(), speaker_cfg(False)) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)
        assert all(l.speaker == 0 for l in h.updates)


def test_untranslated_same_language_lines_get_a_speaker_too(monkeypatch):
    monkeypatch.setattr(pl, "load_embedder", lambda progress=None, models_dir=None: FakeEmbedder())
    with Harness(monkeypatch, FakeRecognizer(lang="zh"), FakeTranslator(), two_sentences(), speaker_cfg()) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)
        assert [h.lines[i].speaker for i in sorted(h.lines)[:2]] == [1, 2]


def test_a_model_that_fails_to_load_only_disables_the_feature(monkeypatch):
    def fail(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(pl, "load_embedder", fail)
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), two_sentences(), speaker_cfg()) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)      # translation carries on
        assert any(lv == "warn" and "说话人" in m and "no network" in m for lv, m in h.statuses)
        assert all(l.speaker == 0 for l in h.updates) and h.statuses[-1][0] == "ready"


def test_a_crashing_embedder_does_not_break_recognition(monkeypatch):
    class Bad:
        def __call__(self, audio):
            raise RuntimeError("onnx exploded")
    monkeypatch.setattr(pl, "load_embedder", lambda progress=None, models_dir=None: Bad())
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), two_sentences(), speaker_cfg()) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 2)
        assert all(l.speaker == 0 for l in h.updates)


def test_reset_speakers_restarts_the_numbering(monkeypatch):
    monkeypatch.setattr(pl, "load_embedder", lambda progress=None, models_dir=None: FakeEmbedder())
    h = Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), two_sentences(), speaker_cfg())
    h.pipe._speakers.assign(A, 3.0)
    h.pipe._speakers.assign(B, 3.0)
    assert h.pipe._speakers.count == 2
    h.pipe.reset_speakers()
    assert h.pipe._speakers.count == 0


def three_sentences():
    return ScriptedSource([(0.3, 0), (2.0, .9), (1.2, 0), (2.0, .9), (1.2, 0), (2.0, .9), (1.5, 0)])


class BlendEmbedder:
    """Voiceprints A, B, then a blend of A and B — as if the third utterance had two people talking over each other."""

    def __init__(self):
        self.n = 0

    def __call__(self, audio):
        self.n += 1
        return (A, B, unit(1, 1, 0, 0))[self.n - 1]


def test_a_voiceprint_blended_from_two_known_speakers_flags_the_line_uncertain(monkeypatch):
    monkeypatch.setattr(pl, "load_embedder", lambda progress=None, models_dir=None: BlendEmbedder())
    with Harness(monkeypatch, FakeRecognizer(), FakeTranslator(), three_sentences(), speaker_cfg()) as h:
        assert h.wait(lambda: sum(l.dst_final for l in h.lines.values()) >= 3)
        ordered = [h.lines[i] for i in sorted(h.lines)]
        assert ordered[0].uncertain_reason == "" and ordered[1].uncertain_reason == ""
        assert "多人同时说话" in ordered[2].uncertain_reason


def test_transcript_file_names_the_speaker(tmp_path):
    from live_translator.models import Line
    w = pl._TranscriptWriter(True, str(tmp_path))
    w.write(Line(1, src="hello", src_lang="en", speaker=2))
    w.write(Line(2, src="again", src_lang="en"))
    w.close()
    text = w.path.read_text(encoding="utf-8")
    assert "[说话人 2] hello" in text and "说话人" not in text.split("hello")[1]


def test_transcript_file_marks_uncertain_lines(tmp_path):
    from live_translator.models import Line
    w = pl._TranscriptWriter(True, str(tmp_path))
    w.write(Line(1, src="garbled", src_lang="en", uncertain_reason="识别把握较低"))
    w.write(Line(2, src="clean", src_lang="en"))
    w.close()
    lines = w.path.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("⚠ garbled")
    assert lines[1].endswith(") clean") and "⚠" not in lines[1]


# ── settings / display ───────────────────────────────────────────────────────

def test_speaker_settings_roundtrip_through_panel_and_preferences(qapp):
    from live_translator.ui.bridge import Bridge
    from live_translator.ui.preferences import PreferencesDialog
    from live_translator.ui.settings_panel import SettingsPanel
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.speaker = SpeakerCfg(enabled=True, similarity=0.65, max_speakers=4)
    panel.load(cfg)
    assert panel.spk_cb.isChecked() and panel.collect().speaker == cfg.speaker
    panel.spk_cb.setChecked(False)
    assert not panel.collect().speaker.enabled
    dlg = PreferencesDialog(cfg)
    dlg.spk_threshold.setValue(0.4)
    dlg.spk_max.setValue(2)
    out = dlg.apply_to(cfg)
    assert (out.speaker.similarity, out.speaker.max_speakers, out.speaker.enabled) == (0.4, 2, True)
    panel.apply_general(out)                                         # preferences reach the next run via the panel
    got = panel.collect().speaker
    assert (got.similarity, got.max_speakers) == (0.4, 2) and not got.enabled


def test_overlay_tags_speakers_only_once_a_second_voice_appears(qapp):
    from live_translator.config import OverlayCfg
    from live_translator.models import Line
    from live_translator.ui.overlay import SubtitleOverlay
    o = SubtitleOverlay(OverlayCfg(w=520, max_sentences=3, show_source=False))
    o.move(-6000, 300)
    o.show_line(Line(1, src="a", dst="第一句", dst_final=True, src_final=True, speaker=1))
    assert o._dst.text() == "第一句" and o._dst._prefix_tok is None          # one speaker so far: no tag
    o.show_line(Line(2, src="b", dst="第二句", dst_final=True, src_final=True, speaker=2))
    assert o._dst._prefix_tok.base == "[2]" and o._hist[0]._prefix_tok.base == "[1]"
    assert o._dst.text() == "第二句"                                          # the tag is not part of the text
    o.show_line(Line(3, src="c", dst="第三句", src_final=True))                # speaker not known yet (0): untagged
    assert o._dst._prefix_tok is None
    o.reset_speakers()
    assert o._dst._prefix_tok is None and o._hist[0]._prefix_tok is None


def test_overlay_shows_a_warning_glyph_on_uncertain_lines(qapp):
    from live_translator.config import OverlayCfg
    from live_translator.models import Line
    from live_translator.ui.overlay import SubtitleOverlay
    o = SubtitleOverlay(OverlayCfg(w=520, max_sentences=3, show_source=False))
    o.move(-6000, 300)
    o.show_line(Line(1, src="a", dst="第一句", dst_final=True, src_final=True, uncertain_reason="不确定"))
    assert o._dst._prefix_tok.base == "⚠"
    o.show_line(Line(2, src="b", dst="第二句", dst_final=True, src_final=True, speaker=1))
    o.show_line(Line(3, src="c", dst="第三句", dst_final=True, src_final=True, speaker=2, uncertain_reason="不确定"))
    assert o._dst._prefix_tok.base == "[2]⚠"


def test_transcript_rows_show_a_coloured_speaker_label(qapp):
    from live_translator.models import Line
    from live_translator.ui.transcript import TranscriptView
    v = TranscriptView()
    v.upsert(Line(1, src="hello", src_final=True, dst_final=True, speaker=2))
    v.upsert(Line(2, src="again", src_final=True, dst_final=True))
    v._render()
    assert v._rows[1].spk.text() == "说话人 2" and not v._rows[1].spk.isHidden()
    assert v._rows[2].spk.isHidden()
    assert "[说话人 2] hello" in v.plain_text() and "[说话人" not in v.plain_text().split("hello")[1]


def test_transcript_rows_show_an_uncertainty_warning(qapp):
    from live_translator.models import Line
    from live_translator.ui.transcript import TranscriptView
    v = TranscriptView()
    v.upsert(Line(1, src="garbled", src_final=True, dst_final=True, uncertain_reason="识别把握较低"))
    v.upsert(Line(2, src="clean", src_final=True, dst_final=True))
    v._render()
    assert "识别把握较低" in v._rows[1].warn.text() and not v._rows[1].warn.isHidden()
    assert v._rows[2].warn.isHidden()


def test_speaker_colours_are_distinct_and_wrap_around():
    from live_translator.ui.ruby_text import speaker_color
    assert len({speaker_color(i) for i in range(1, 9)}) == 8
    assert speaker_color(9) == speaker_color(1) and speaker_color(1, dark=False) != speaker_color(1, dark=True)


# ── model loading: pinned revision + checksum ────────────────────────────────

def test_the_model_is_fetched_at_the_pinned_revision_and_checked_against_its_sha256(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub
    f = tmp_path / "model.onnx"
    f.write_bytes(b"not the real model")
    calls = []

    def fake_download(**kw):
        calls.append(kw)
        return str(f)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="校验失败"):
        sp.load_embedder(None, str(tmp_path))                          # wrong file: refused, never loaded
    assert calls[0]["revision"] == sp.SPEAKER_MODEL_REVISION and calls[0]["repo_id"] == sp.SPEAKER_MODEL_REPO
    assert calls[0]["local_files_only"] is True                        # cache first, no network round-trip

    monkeypatch.setattr(sp, "SPEAKER_MODEL_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    monkeypatch.setattr(sp, "SpeakerEmbedder", lambda path: ("loaded", path))
    assert sp.load_embedder(None, str(tmp_path)) == ("loaded", str(f))


def test_a_missing_local_model_is_downloaded_once_with_a_progress_message(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x")
    log = []

    def fake_download(**kw):
        log.append("local" if kw.get("local_files_only") else "network")
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return str(f)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(sp, "SPEAKER_MODEL_SHA256", hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr(sp, "SpeakerEmbedder", lambda path: "ok")
    said = []
    assert sp.load_embedder(said.append, None) == "ok"
    assert log == ["local", "network"] and any("下载" in m for m in said)


def test_defaults_match_what_was_measured():
    c = SpeakerCfg()
    assert c.similarity == 0.55 and c.max_speakers == 0 and not c.enabled
    assert sp.MIN_EMBED_S == 1.0 and sp.FULL_EMBED_S == 2.0
