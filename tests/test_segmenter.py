import numpy as np

from live_translator.config import SegmenterCfg
from live_translator.segmenter import FRAME_MS, Segmenter
from live_translator.vad import FRAME


def fake_vad(frame: np.ndarray) -> float:
    """Frames carry their own 'probability' as a constant value, so tests can script the VAD."""
    return float(frame[0])


def audio(*parts):
    """parts: (seconds, prob) tuples → float32 signal whose frames read back as `prob`."""
    return np.concatenate([np.full(int(s * 16000 // FRAME) * FRAME, p, dtype=np.float32) for s, p in parts])


def run(sig, cfg=None, chunk=1600):
    seg = Segmenter(fake_vad, cfg or SegmenterCfg(), clock=lambda: 0.0)
    events = []
    for i in range(0, len(sig), chunk):
        events += seg.feed(sig[i:i + chunk])
    return seg, events


def kinds(events):
    return [e.kind for e in events]


def test_speech_then_silence_yields_partials_then_one_final():
    _, ev = run(audio((0.5, 0.0), (3.0, 0.9), (1.0, 0.0)))
    assert kinds(ev)[-1] == "final"
    assert kinds(ev).count("final") == 1
    assert "partial" in kinds(ev)
    assert {e.utt_id for e in ev} == {1}
    final = ev[-1]
    assert final.reason == "silence"
    # ~3 s speech + preroll + 160 ms tail pad; silence beyond the pad is trimmed
    assert 3.0 <= len(final.audio) / 16000 <= 3.7


def test_final_is_not_delayed_beyond_min_silence():
    cfg = SegmenterCfg(min_silence_ms=400)
    seg = Segmenter(fake_vad, cfg, clock=lambda: 0.0)
    events = seg.feed(audio((1.0, 0.9)))
    assert "final" not in kinds(events)
    silence_needed = 0.4 + 2 * FRAME_MS / 1000
    events = seg.feed(audio((silence_needed, 0.0)))
    assert kinds(events).count("final") == 1


def test_short_pause_does_not_split_an_utterance():
    _, ev = run(audio((1.5, 0.9), (0.25, 0.0), (1.5, 0.9), (1.0, 0.0)))
    assert kinds(ev).count("final") == 1


def test_click_noise_is_ignored():
    _, ev = run(audio((0.5, 0.0), (0.1, 0.9), (2.0, 0.0)))
    assert "final" not in kinds(ev)


def test_short_burst_after_partials_is_discarded_not_finalised():
    # 0.8 s of speech triggers a partial, but that's still < min_speech? no → make min_speech larger
    cfg = SegmenterCfg(min_speech_ms=2000)
    _, ev = run(audio((0.2, 0.0), (1.0, 0.9), (1.0, 0.0)), cfg)
    assert kinds(ev)[-1] == "discard"
    assert "final" not in kinds(ev)


def test_two_utterances_get_increasing_ids():
    _, ev = run(audio((1.5, 0.9), (1.0, 0.0), (1.5, 0.9), (1.0, 0.0)))
    finals = [e for e in ev if e.kind == "final"]
    assert [f.utt_id for f in finals] == [1, 2]


def test_partial_cadence():
    cfg = SegmenterCfg(partial_interval_ms=700)
    _, ev = run(audio((0.1, 0.0), (4.0, 0.9)), cfg)
    partials = [e for e in ev if e.kind == "partial"]
    assert 4 <= len(partials) <= 6
    lens = [len(p.audio) for p in partials]
    assert lens == sorted(lens)                   # each snapshot contains the previous one


def test_max_length_cut_happens_at_the_quietest_point():
    cfg = SegmenterCfg(max_utterance_s=8.0, min_silence_ms=450)
    # speech with a brief dip (0.3 s at p=0.3, i.e. below the hysteresis floor but shorter than min silence)
    sig = audio((0.1, 0.0), (5.0, 0.9), (0.3, 0.3), (6.0, 0.9), (1.0, 0.0))
    _, ev = run(sig, cfg)
    finals = [e for e in ev if e.kind == "final"]
    assert len(finals) == 2
    assert finals[0].reason == "maxlen"
    assert finals[0].utt_id != finals[1].utt_id
    # the cut fell on the dip: first chunk ≈ 5.1–5.5 s, not ≈ 8 s
    assert 5.0 <= len(finals[0].audio) / 16000 <= 5.7


def test_nonstop_speech_is_still_cut_at_max_length():
    cfg = SegmenterCfg(max_utterance_s=6.0)
    _, ev = run(audio((0.1, 0.0), (20.0, 0.9), (1.0, 0.0)), cfg)
    finals = [e for e in ev if e.kind == "final"]
    assert len(finals) >= 3
    assert all(len(f.audio) / 16000 <= 6.2 for f in finals)


def test_flush_finalises_open_utterance():
    seg, ev = run(audio((2.0, 0.9)))
    assert seg.in_speech
    out = seg.flush()
    assert kinds(out) == ["final"] and out[0].reason == "flush"
    assert not seg.in_speech


def test_chunk_size_does_not_matter():
    sig = audio((0.3, 0.0), (2.0, 0.9), (1.0, 0.0), (1.2, 0.9), (1.0, 0.0))
    _, a = run(sig, chunk=1600)
    _, b = run(sig, chunk=777)
    assert [(e.kind, e.utt_id, len(e.audio) if e.audio is not None else 0) for e in a] == \
           [(e.kind, e.utt_id, len(e.audio) if e.audio is not None else 0) for e in b]
