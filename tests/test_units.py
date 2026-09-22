import json

from live_translator.asr.base import LanguageTracker
from live_translator.asr.filters import SegStats, clean_text, collapse_repeats, keep_segment
from live_translator.config import AppConfig, apply_preset, load_config, save_config
from live_translator.glossary import asr_hint, parse_glossary, prompt_block
from live_translator.languages import normalize_lang, same_language
from live_translator.resample import Resampler
from live_translator.text_utils import (clean_translation, guess_language, looks_like_language,
                                        split_sentences, visible_text)
from live_translator.translate.prompts import build_correction_messages, build_messages

import numpy as np


# ── text_utils ───────────────────────────────────────────────────────────────

def test_visible_text_hides_think_blocks_even_while_streaming():
    assert visible_text("<think>hmm</think>你好") == "你好"
    assert visible_text("<think>still thinking...") == ""
    assert visible_text("你好") == "你好"


def test_clean_translation_strips_wrappers():
    assert clean_translation("<think>x</think>\n译文：你好世界") == "你好世界"
    assert clean_translation('"Hello there"') == "Hello there"
    assert clean_translation("```\n你好\n```") == "你好"
    assert clean_translation("第一行\n第二行") == "第一行 第二行"
    assert clean_translation('He said "hi" to me') == 'He said "hi" to me'      # inner quotes untouched


def test_looks_like_language():
    assert looks_like_language("你好，世界", "zh-Hans")
    assert not looks_like_language("Hello world, this is English", "zh-Hans")     # echoed the source
    assert looks_like_language("こんにちは", "ja")
    assert looks_like_language("Bonjour tout le monde", "fr")
    assert not looks_like_language("你好世界这是中文", "en")
    assert looks_like_language("42", "zh-Hans")                                    # nothing to judge


def test_guess_language():
    assert guess_language("こんにちは、元気ですか") == "ja"
    assert guess_language("안녕하세요") == "ko"
    assert guess_language("你好世界") == "zh"
    assert guess_language("Привет мир") == "ru"
    assert guess_language("hello") == "en"


def test_split_sentences():
    assert split_sentences("Hello there. How are you? Fine!") == ["Hello there.", "How are you?", "Fine!"]
    assert split_sentences("你好。今天天气不错！") == ["你好。", "今天天气不错！"]
    long = ("word " * 100).strip()
    assert all(len(p) <= 225 for p in split_sentences(long))


# ── languages ────────────────────────────────────────────────────────────────

def test_language_normalisation():
    assert normalize_lang("english") == "en"
    assert normalize_lang("Chinese") == "zh"
    assert normalize_lang("yue") == "zh"
    assert normalize_lang("zh-CN") == "zh"
    assert normalize_lang("") == ""
    assert same_language("zh", "zh-Hans") and same_language("zh", "zh-Hant")
    assert not same_language("en", "zh-Hans")
    assert not same_language("", "")


# ── asr filters & tracker ────────────────────────────────────────────────────

def test_hallucinated_outro_is_dropped_but_real_thanks_is_kept():
    assert not keep_segment(SegStats("Thanks for watching!", avg_logprob=-0.3))
    assert not keep_segment(SegStats("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", avg_logprob=-0.3))
    assert keep_segment(SegStats("Thank you.", avg_logprob=-0.2, no_speech_prob=0.02))     # someone really said it
    assert not keep_segment(SegStats("Thank you.", avg_logprob=-1.1, no_speech_prob=0.5))  # model unsure → noise


def test_silence_and_repetition_rules():
    assert not keep_segment(SegStats("blah", avg_logprob=-1.2, no_speech_prob=0.8))
    assert not keep_segment(SegStats("la la la", compression_ratio=3.0))
    assert not keep_segment(SegStats("  ♪ ♪ "))
    assert keep_segment(SegStats("We ship on Friday.", avg_logprob=-0.4, no_speech_prob=0.05))


def test_repetition_collapse_and_prompt_echo():
    assert collapse_repeats("go go go go go go go") == "go"
    assert collapse_repeats("ok fine") == "ok fine"
    assert clean_text("Kubernetes, Envoy", prompt="Kubernetes, Envoy, Istio") == ""
    assert clean_text("We use Kubernetes", prompt="Kubernetes, Envoy") == "We use Kubernetes"


def test_language_tracker_ignores_low_confidence_outliers():
    t = LanguageTracker()
    for _ in range(3):
        t.observe("zh", 0.97)
    assert t.current == "zh"
    assert t.resolve("cy", 0.55, duration_s=1.0) == "zh"           # "Yeah." mis-heard as Welsh
    assert t.resolve("en", 0.97, duration_s=1.0) == "en"           # a confident switch is honoured
    assert t.resolve("en", 0.75, duration_s=4.0) == "en"           # long clip: detection is trustworthy
    assert t.resolve("en", 0.75, duration_s=1.0) == "zh"           # short clip: not enough


def test_language_tracker_switches_after_sustained_change():
    t = LanguageTracker()
    for _ in range(3):
        t.observe("zh", 0.9)
    for _ in range(4):
        t.observe("en", 0.95)
    assert t.current == "en"


# ── glossary / prompts ───────────────────────────────────────────────────────

def test_glossary_parsing():
    terms = parse_glossary("# comment\nKubernetes\nGPU = 显卡\nOpenAI -> 开放人工智能\n\n")
    assert [(t.source, t.target) for t in terms] == [("Kubernetes", ""), ("GPU", "显卡"), ("OpenAI", "开放人工智能")]
    assert asr_hint(terms, "cloud talk") == "cloud talk. Kubernetes, GPU, OpenAI"
    block = prompt_block(terms)
    assert "GPU → 显卡" in block and "Kubernetes (keep unchanged)" in block


def test_prompt_contains_context_as_chat_history_and_glossary():
    msgs = build_messages("It works.", "en", "zh-Hans", [("Hi.", "你好。"), ("", "x")],
                          prompt_block(parse_glossary("GPU = 显卡")), "hardware review")
    assert msgs[0]["role"] == "system"
    assert "Simplified Chinese" in msgs[0]["content"] and "GPU → 显卡" in msgs[0]["content"]
    assert "hardware review" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]     # the empty pair was skipped
    assert msgs[-1]["content"] == "It works."


def test_correction_prompt_asks_for_same_language_cleanup_not_translation():
    msgs = build_correction_messages("我看到電源大聲指責", "zh", [("Hi.", "你好。"), ("", "x")],
                                     prompt_block(parse_glossary("店员")), "客服录音")
    assert msgs[0]["role"] == "system"
    sysmsg = msgs[0]["content"]
    assert "SAME language" in sysmsg and "店员 (keep unchanged)" in sysmsg and "客服录音" in sysmsg
    assert "never rephrase, summarise, translate" in sysmsg.lower()
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]     # the empty pair was skipped
    assert msgs[1]["content"] == "Hi." and msgs[2]["content"] == "Hi."        # echoed, not translated
    assert msgs[-1]["content"] == "我看到電源大聲指責"


# ── config ───────────────────────────────────────────────────────────────────

def test_config_roundtrip_and_tolerance(tmp_path):
    p = tmp_path / "c.json"
    cfg = AppConfig()
    cfg.translate.remote.api_key = "sk-test"
    cfg.lang.target = "ja"
    cfg.seg.vad_threshold = 1                                # int where float expected
    save_config(cfg, p)
    back = load_config(p)
    assert back.translate.remote.api_key == "sk-test" and back.lang.target == "ja"
    assert back.seg.vad_threshold == 1.0 and isinstance(back.seg.vad_threshold, float)

    data = json.loads(p.read_text(encoding="utf-8"))
    data["future_field"] = 1                                 # unknown key
    data["asr"]["beam_final"] = "not a number"               # wrong type → default kept
    data["lang"] = "garbage"
    p.write_text(json.dumps(data), encoding="utf-8")
    back = load_config(p)
    assert back.asr.beam_final == AppConfig().asr.beam_final
    assert back.lang.target == "zh-Hans"

    p.write_text("{not json", encoding="utf-8")
    assert load_config(p).lang.source == "auto"


def test_presets():
    cfg = AppConfig()
    apply_preset(cfg, "fast")
    assert cfg.seg.min_silence_ms < AppConfig().seg.min_silence_ms and cfg.translate.draft
    apply_preset(cfg, "accurate")
    assert cfg.asr.beam_final == 5 and not cfg.translate.draft


# ── resampler ────────────────────────────────────────────────────────────────

def test_resampler_48k_to_16k_keeps_duration_and_tone():
    t = np.arange(48000) / 48000
    x = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    r = Resampler()
    y = np.concatenate([r.process(x[i:i + 2400], 48000) for i in range(0, len(x), 2400)])
    assert abs(len(y) - 16000) < 400                             # soxr's streaming filter holds back ~20 ms
    assert 0.3 < np.sqrt(np.mean(y[1000:] ** 2)) < 0.4          # 0.5 amplitude sine → RMS ≈ 0.354


def test_resampler_passthrough_at_16k():
    x = np.random.randn(1000).astype(np.float32)
    assert np.array_equal(Resampler().process(x, 16000), x)


def test_cjk_punctuation_normalisation():
    from live_translator.text_utils import normalize_cjk_punct as n
    assert n("所以,我的美国同胞.", "zh-Hans") == "所以，我的美国同胞。"
    assert n("你好!你好吗?", "zh-Hant") == "你好！你好吗？"
    assert n("こんにちは,元気ですか.", "ja") == "こんにちは、元気ですか。"
    assert n("版本 3.5 发布了.", "zh-Hans") == "版本 3.5 发布了。"           # decimals untouched
    assert n("使用 GPU, 很快", "zh-Hans") == "使用 GPU, 很快"               # after Latin: untouched
    assert n("Hello, world.", "en") == "Hello, world."


def test_mt_runaway_guards():
    from live_translator.text_utils import strip_trailing_ellipsis as strip
    from live_translator.text_utils import trim_runaway as trim
    assert strip("I have here...") == "I have here"
    assert strip("Well… ") == "Well"
    assert strip("Version 3.5 is out.") == "Version 3.5 is out."
    assert strip("...") == "..."                                             # never reduce to nothing
    # the real NLLB failure seen on "I have here...": a hallucinated, repeated continuation
    assert trim("I have here", "我在这里。现在，我们要做什么？现在，我们要做什么？") == "我在这里。"
    # long sources are never cut: a legitimate two-sentence translation survives
    long_src = "Well I think we should go now and then we can all have some lunch together"
    assert trim(long_src, "我认为我们该走了。然后我们可以一起吃午饭。") == "我认为我们该走了。然后我们可以一起吃午饭。"
    # exact repeats collapse even for long sources
    assert trim(long_src, "我们该走了。我们该走了。") == "我们该走了。"
    assert trim("A pendulum", "一个摆锤。") == "一个摆锤。"
    assert trim("你好吗", "How are you? I am fine.") == "How are you?"       # short CJK source


def test_mic_without_portaudio_library(monkeypatch):
    """Linux without libportaudio2: `import sounddevice` raises OSError, which must not crash the UI or the pipeline."""
    import builtins

    import pytest

    from live_translator.audio import mic
    from live_translator.audio.base import AudioSourceError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert mic.list_input_devices() == []
    with pytest.raises(AudioSourceError, match="PortAudio"):
        mic.MicSource().start(lambda *_: None)
