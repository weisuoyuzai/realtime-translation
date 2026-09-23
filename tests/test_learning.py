"""Multi-line overlay, learning mode (pinyin ruby + speaker button), TTS voice choice and transcript scrolling."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QLocale  # noqa: E402

from live_translator import ruby  # noqa: E402
from live_translator.config import AppConfig, OverlayCfg  # noqa: E402
from live_translator.models import Line  # noqa: E402
import live_translator.ui.main_window as mw  # noqa: E402
from live_translator.ui.main_window import MainWindow  # noqa: E402
from live_translator.ui.overlay import SubtitleOverlay  # noqa: E402
from live_translator.ui.preferences import PreferencesDialog  # noqa: E402
from live_translator.ui.ruby_text import RubyText  # noqa: E402
from live_translator.ui.transcript import TranscriptView  # noqa: E402
from live_translator.ui.tts import Tts  # noqa: E402

needs_pinyin = pytest.mark.skipif("pypinyin" in ruby.missing_packages(), reason="pypinyin not installed")
needs_kakasi = pytest.mark.skipif("pykakasi" in ruby.missing_packages(), reason="pykakasi not installed")
needs_espeak = pytest.mark.skipif("espeakng-loader" in ruby.missing_packages(), reason="espeakng-loader not installed")

LONG_ZH = "敏捷的棕色狐狸在大家的注视下跳过了那只懒狗，然后消失在了远处的森林里，再也没有回来，谁也不知道它去了哪里。"


def fin(i, **kw) -> Line:
    return Line(i, src_final=True, dst_final=True, **kw)


# ── ruby tokens ──────────────────────────────────────────────────────────────

@needs_pinyin
def test_chinese_gets_pinyin_per_character_and_punctuation_glues_to_the_previous_token():
    toks = ruby.tokenize("你好，世界", "zh-Hans", ruby=True)
    assert [(t.base, t.ruby) for t in toks] == [("你", "nǐ"), ("好", "hǎo"), ("，", ""), ("世", "shì"), ("界", "jiè")]
    assert [t.glue for t in toks] == [False, False, True, False, False]


@needs_pinyin
def test_polyphones_are_resolved_from_context():
    assert [t.ruby for t in ruby.tokenize("银行", "zh", ruby=True)] == ["yín", "háng"]
    assert [t.ruby for t in ruby.tokenize("行走", "zh", ruby=True)] == ["xíng", "zǒu"]


@needs_pinyin
def test_no_pinyin_for_japanese_or_when_learning_is_off_and_latin_words_stay_whole():
    assert all(not t.ruby or "぀" <= t.ruby[0] <= "ゟ" for t in ruby.tokenize("漢字を読む", "ja", ruby=True))   # hiragana, never pinyin
    assert not any(t.ruby for t in ruby.tokenize("你好", "zh", ruby=False))
    toks = ruby.tokenize("Hello world, 你好", "zh", ruby=True)
    assert [(t.base, t.space) for t in toks][:2] == [("Hello", False), ("world,", True)]
    assert toks[2].base == "你" and toks[2].space and toks[2].ruby == "nǐ"


def test_language_is_guessed_when_unknown_and_opening_brackets_stay_with_the_next_token():
    assert ruby.ruby_kind("你好") == "zh" and ruby.ruby_kind("こんにちは") == "ja" and ruby.ruby_kind("hello") == "en"
    assert ruby.ruby_kind("漢字", "ja") == "ja" and ruby.ruby_kind("你好", "zh-Hans") == "zh" and ruby.ruby_kind("x", "en") == "en"
    assert ruby.ruby_kind("안녕", "ko") == "ko" and ruby.ruby_kind("Привет") == "ru" and ruby.ruby_kind("สวัสดี", "th") == ""
    toks = ruby.tokenize("「你好」", "zh")
    assert [t.glue for t in toks] == [False, True, False, True]


@needs_kakasi
def test_japanese_kanji_get_furigana_and_okurigana_stay_plain():
    got = {t.base: t.ruby for t in ruby.tokenize("漢字を読む。取り組みを続ける", "ja", ruby=True)}
    assert got["漢字"] == "かんじ" and got["読"] == "よ" and got["取"] == "と" and got["組"] == "く" and got["続"] == "つづ"
    assert got["を"] == "" and got["む"] == "" and got["り"] == ""                # kana are never annotated
    assert not any(t.ruby for t in ruby.tokenize("漢字を読む", "ja", ruby=False))


@needs_kakasi
def test_furigana_covers_compounds_and_iteration_marks_and_reads_kyou_not_konnichiwa():
    got = {t.base: t.ruby for t in ruby.tokenize("人々は東京大学で学ぶ", "ja", ruby=True)}
    assert got["人々"] == "ひとびと" and got["東京大学"] == "とうきょうだいがく" and got["学"] == "まな"
    assert {t.base: t.ruby for t in ruby.tokenize("今日は天気がいい", "ja", ruby=True)}["今日"] == "きょう"
    assert {t.base: t.ruby for t in ruby.tokenize("今日は！", "ja", ruby=True)}["今日"] == "こんにち"   # the greeting


def test_korean_words_get_romanization_as_pronounced():
    got = {t.base: t.ruby for t in ruby.tokenize("안녕하세요, 감사합니다! 국물 같이 신라 많이 OK", "ko", ruby=True)}
    assert got == {"안녕하세요,": "annyeonghaseyo", "감사합니다!": "gamsahamnida", "국물": "gungmul", "같이": "gachi",
                   "신라": "silla", "많이": "mani", "OK": ""}
    assert not any(t.ruby for t in ruby.tokenize("감사합니다", "ko", ruby=False))


@needs_espeak
def test_other_languages_get_ipa_per_word_but_not_over_numbers_or_foreign_script():
    got = {t.base: t.ruby for t in ruby.tokenize("Hello world, a 3 Привет", "en", ruby=True)}
    assert got["Hello"] == "həlˈoʊ" and got["world,"] == "wˈɜːld" and got["a"] == "ə"
    assert got["3"] == "" and got["Привет"] == ""
    got = {t.base: t.ruby for t in ruby.tokenize("Привет, iPhone", "ru", ruby=True)}
    assert got["Привет,"].startswith("prʲi") and got["iPhone"] == ""
    assert all(t.ruby for t in ruby.tokenize("Bonjour tout le monde", "fr", ruby=True))
    assert not any(t.ruby for t in ruby.tokenize("Xin chào", "vi", ruby=True))           # Vietnamese: not annotated


@needs_kakasi
def test_japanese_tokens_rebuild_the_original_text():
    text = "私はPythonが好きです。「東京」へ行った"
    assert "".join(t.base for t in ruby.tokenize(text, "ja", ruby=True)).replace(" ", "") == text.replace(" ", "")


# ── RubyText ─────────────────────────────────────────────────────────────────

def test_ruby_text_wraps_and_keeps_only_the_last_lines(qapp):
    w = RubyText()
    w.setText(LONG_ZH, "zh-Hans")
    one = w.heightForWidth(10_000)
    width = w.sizeHint().width() // 4                     # relative to the text, so it holds whatever CJK font (if any)
    assert w.heightForWidth(width) > one > 0
    n = len(w._lines(width))
    assert n >= 3
    w.setMaxLines(2)
    assert len(w._lines(width)) == 2 and w.heightForWidth(width) == pytest.approx(2 * one, abs=2)
    w.setText("")
    assert w.heightForWidth(width) == 0


@needs_pinyin
def test_ruby_row_makes_lines_taller_and_never_splits_punctuation_from_its_word(qapp):
    w = RubyText()
    w.setText(LONG_ZH, "zh-Hans", ruby=False)
    plain = w.heightForWidth(10_000)
    w.setText(LONG_ZH, "zh-Hans", ruby=True)
    assert w.has_ruby() and w.heightForWidth(10_000) > plain
    for width in range(150, 400, 7):
        for line in w._lines(width):
            assert not line[0][0].glue or line[0][0].base not in "，。"          # no line starts with punctuation


# ── overlay ──────────────────────────────────────────────────────────────────

def make_overlay(**kw) -> SubtitleOverlay:
    o = SubtitleOverlay(OverlayCfg(w=520, **kw))
    o.move(-6000, 300)
    return o


def test_overlay_height_follows_the_sentences_shown_and_grows_upwards(qapp):
    o = make_overlay(max_sentences=2, show_source=False)
    o.show_line(fin(1, src="x", dst="短句一", dst_lang="zh-Hans"))
    h1, bottom = o.height(), o.y() + o.height()
    o.show_line(fin(2, src="y", dst="短句二", dst_lang="zh-Hans"))
    assert o.height() > h1 and o.y() + o.height() == bottom       # a second sentence: taller, growing upwards
    o.set_max_sentences(1)
    assert o.height() == h1 and o.y() + o.height() == bottom      # back to just the current one


def test_long_sentences_are_shown_in_full_across_several_lines(qapp):
    o = make_overlay(max_sentences=2, show_source=False)
    o.show_line(fin(1, src="a", dst=LONG_ZH, dst_lang="zh-Hans"))
    o.show_line(fin(2, src="b", dst=LONG_ZH, dst_lang="zh-Hans"))
    assert texts(o) == [LONG_ZH, LONG_ZH]                          # the earlier sentence is not pushed out by a long one
    assert len(o._dst._lines(o._dst.width())) >= 3 and len(o._hist[0]._lines(o._hist[0].width())) >= 3
    o.show_line(fin(3, src="c", dst="字" * 2000, dst_lang="zh-Hans"))
    assert len(o._dst._lines(o._dst.width())) == 8                 # only an absurdly long one is cut, keeping its end


def test_default_overlay_shows_two_sentences_with_automatic_height(qapp):
    c = OverlayCfg()
    assert c.max_sentences == 2 and c.auto_height and not c.learning and c.tts_read == "dst"


def test_custom_height_keeps_the_bar_and_clips_the_oldest_text(qapp):
    o = make_overlay(max_sentences=4, show_source=False)
    for i in range(1, 5):
        o.show_line(fin(i, src=str(i), dst=f"第 {i} 句", dst_lang="zh-Hans"))
    auto_h = o.height()
    o.cfg.auto_height = False
    o.resize(o.width(), auto_h + 120)                                # the user drags the bar taller …
    o.show_line(fin(5, src="5", dst="第 5 句", dst_lang="zh-Hans"))
    assert o.height() == auto_h + 120                                # … and it stays that tall
    o.resize(o.width(), 90)                                          # a bar shorter than its content
    o.show_line(fin(6, src="6", dst="第 6 句", dst_lang="zh-Hans"))
    body = o._body
    assert o.height() == 90 and body.y() < 0 and body.y() + body.height() == 90   # newest at the bottom, top clipped


def test_dragging_the_bar_taller_switches_off_automatic_height_but_width_only_does_not(qapp):
    o = make_overlay(max_sentences=2, show_source=False)
    o.show_line(fin(1, src="a", dst="第一句", dst_lang="zh-Hans"))
    auto_h = o.height()
    o.resize(700, auto_h)                                            # width only: still automatic
    assert o.cfg.auto_height and o.height() == auto_h
    o.resize(700, auto_h + 80)                                       # vertical drag: the user's own height
    assert not o.cfg.auto_height and o.height() == auto_h + 80
    o.set_auto_height(True)                                          # right-click menu / preferences: back to automatic
    assert o.cfg.auto_height and o.height() == auto_h


def texts(o: SubtitleOverlay) -> list[str]:
    """What the bar shows, oldest sentence first (the current sentence is the last entry)."""
    hist = [h.text() for h in reversed(o._hist) if not h.isHidden()]
    return hist + [o._dst.text()]


def test_previous_sentence_stays_on_screen_while_the_next_one_streams_in(qapp):
    """The bug: a new (still interim) sentence replaced the previous one immediately, so it flashed by."""
    o = make_overlay(max_sentences=2, show_source=False)
    o.show_line(fin(1, src="a", dst="第一句", dst_lang="zh-Hans"))
    assert texts(o) == ["第一句"]
    o.show_line(Line(2, src="b", src_lang="en"))                    # interim recognition of the next sentence
    assert texts(o) == ["第一句", "b"]
    o.show_line(Line(2, src="b c", dst="第二", src_lang="en", dst_lang="zh-Hans", dst_draft=True))
    assert texts(o) == ["第一句", "第二"]
    o.show_line(fin(2, src="b c", dst="第二句", dst_lang="zh-Hans"))
    assert texts(o) == ["第一句", "第二句"]


def test_the_sentence_count_limits_how_many_stay_and_the_oldest_leave_first(qapp):
    o = make_overlay(max_sentences=2, show_source=False)
    for i, t in enumerate(["一", "二", "三"], 1):
        o.show_line(fin(i, src=t, dst=t, dst_lang="zh-Hans"))
    assert texts(o) == ["二", "三"]                                   # two sentences: the current one + one before it
    o.set_max_sentences(1)
    assert texts(o) == ["三"]                                         # 1 = only the current sentence
    o.set_max_sentences(8)
    assert texts(o) == ["一", "二", "三"]
    for i in range(4, 12):
        o.show_line(fin(i, src=str(i), dst=f"第{i}", dst_lang="zh-Hans"))
    assert texts(o) == [f"第{i}" for i in range(4, 12)]               # never more than 8, whatever the setting


def test_late_update_removal_and_fade_of_sentences_on_screen(qapp):
    o = make_overlay(max_sentences=3, show_source=False)
    o.show_line(Line(1, src="one", src_lang="en", src_final=True))     # translation still pending
    o.show_line(fin(2, src="two", dst="二", dst_lang="zh-Hans"))
    assert texts(o) == ["one", "二"]
    o.show_line(fin(1, src="one", dst="一", dst_lang="zh-Hans"))       # its translation arrives late: updated in place
    assert texts(o) == ["一", "二"] and o._line.id == 2                # …without becoming the current sentence
    o.show_line(Line(2, removed=True))                                # current one retracted → the earlier one is current
    assert texts(o) == ["一"] and o._line.id == 1
    o._fade()
    assert texts(o) == [""] and o._line is None and all(h.isHidden() for h in o._hist)


def test_history_rows_get_ruby_too_and_the_source_is_shown_for_the_current_sentence_only(qapp):
    o = make_overlay(max_sentences=3, learning=True)
    o.show_line(fin(1, src="Hello", dst="你好", src_lang="en", dst_lang="zh-Hans"))
    o.show_line(fin(2, src="Bye", dst="再见", src_lang="en", dst_lang="zh-Hans"))
    assert o._src.text() == "Bye" and not o._hist[0].isHidden() and o._hist[0].text() == "你好"
    assert o._hist[0].has_ruby() == ("pypinyin" not in ruby.missing_packages())


# ── which text gets ruby ─────────────────────────────────────────────────────

@needs_pinyin
def test_ruby_scope_chooses_translation_original_or_both_in_the_overlay(qapp):
    o = make_overlay(learning=True, ruby_scope="dst")
    o.show_line(fin(1, src="你好世界", dst="你好", src_lang="zh", dst_lang="zh-Hans"))
    assert o._dst.has_ruby() and not o._src.has_ruby()
    o.cfg.ruby_scope = "src"
    o._render()
    assert not o._dst.has_ruby() and o._src.has_ruby()
    o.cfg.ruby_scope = "both"
    o._render()
    assert o._dst.has_ruby() and o._src.has_ruby()
    o.cfg.learning = False
    o._render()
    assert not o._dst.has_ruby() and not o._src.has_ruby()


@needs_pinyin
def test_ruby_scope_in_the_transcript(qapp):
    v = filled_view(qapp, n=1)
    v.upsert(fin(1, src="你好世界", dst="你好", src_lang="zh", dst_lang="zh-Hans"))
    row = v._rows[1]
    for scope, want_src, want_dst in (("dst", False, True), ("src", True, False), ("both", True, True)):
        v.set_learning(True, scope)
        v._render()
        assert (row.src.has_ruby(), row.dst.has_ruby()) == (want_src, want_dst), scope
    v.set_learning(False)
    v._render()
    assert not row.src.has_ruby() and not row.dst.has_ruby()


# ── no voice installed: three levels of notice ───────────────────────────────

MISSING_JA = "系统里没有安装「日本語」语音\n装好后请重启"


def test_missing_voice_crosses_out_the_speaker_and_pops_up_once_per_language(qapp, window, monkeypatch):
    popups = []
    monkeypatch.setattr(window, "_show_voice_notice", popups.append)
    window.overlay.cfg.learning = True
    window.overlay.show_line(fin(1, src="今日は", dst="今日は", dst_lang="ja"))
    window._on_voice_missing("ja", MISSING_JA)
    assert len(popups) == 1 and "日本語" in popups[0]                         # loud: a non-blocking dialog
    assert "日本語" in window.statusBar().currentMessage()                      # medium: the status bar
    assert window.overlay._speaker._warning and "日本語" in window.overlay._speaker.toolTip()   # quiet: the icon itself
    window._on_voice_missing("ja", MISSING_JA)
    assert len(popups) == 1                                                    # not nagged again for the same language
    window._on_voice_missing("ko", "系统里没有安装「한국어」语音\n装好后请重启")
    assert len(popups) == 2
    window.tts.speaking_changed.emit(True)                                     # a later successful read clears the cross
    assert not window.overlay._speaker._warning


def test_enabling_learning_checks_the_voice_for_the_language_that_will_be_read(qapp, window, monkeypatch):
    popups = []
    monkeypatch.setattr(window, "_show_voice_notice", popups.append)
    window.tts.missing = {"ja"}
    window.panel.tgt_lang.setCurrentIndex(window.panel.tgt_lang.findData("ja"))
    window.learn_cb.setChecked(True)                                           # reads the translation (ja) → warned now
    assert len(popups) == 1 and window.overlay._speaker._warning
    window.learn_cb.setChecked(False)
    window.overlay.cfg.tts_read = "src"                                        # reads the original instead
    window.panel.src_lang.setCurrentIndex(window.panel.src_lang.findData("zh"))
    window.learn_cb.setChecked(True)
    assert len(popups) == 1                                                    # Chinese voice exists → no new warning
    window.learn_cb.setChecked(False)
    window.panel.src_lang.setCurrentIndex(window.panel.src_lang.findData("auto"))
    window.learn_cb.setChecked(True)
    assert len(popups) == 1                                                    # auto-detect: nothing to check up front


# ── main window: learning mode + speaking ────────────────────────────────────

class FakeTts(Tts):
    """The real Tts object (so the main window's signal wiring is the real one) with the audio parts replaced."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calls = []
        self.missing = set()

    @property
    def speaking(self):
        return False

    def voice_problem(self, lang):
        return ("x", "missing") if lang.split("-")[0] in self.missing else None

    def toggle(self, text, lang):
        self.calls.append((text, lang))

    def stop(self):
        self.calls.append("stop")


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mw, "Tts", FakeTts)
    cfg = AppConfig()
    cfg.translate.mode = "none"
    w = MainWindow(cfg)
    w.move(-6000, 0)
    w.show()
    yield w
    w.close()


def test_learning_checkbox_drives_overlay_transcript_and_menu(qapp, window):
    window.learn_cb.setChecked(True)
    assert window.overlay.cfg.learning and window.transcript._learning and window.current_config().overlay.learning
    assert window.act_learn.isChecked()
    window.act_learn.setChecked(False)
    assert not window.learn_cb.isChecked() and not window.overlay.cfg.learning and "stop" in window.tts.calls
    window.overlay._toggle_learning(True)                        # right-click menu on the bar
    assert window.learn_cb.isChecked() and window.transcript._learning


def test_speaker_reads_translation_or_original_per_preference(qapp, window):
    line = fin(1, src="Hello", dst="你好", src_lang="en", dst_lang="zh-Hans")
    window._speak_line(line)
    assert window.tts.calls[-1] == ("你好", "zh-Hans")
    window.overlay.cfg.tts_read = "src"
    window._speak_line(line)
    assert window.tts.calls[-1] == ("Hello", "en")
    window._speak_line(Line(2, src="こんにちは", src_final=True))     # untranslated: reads what exists, language guessed
    assert window.tts.calls[-1] == ("こんにちは", "ja")
    n = len(window.tts.calls)
    window._speak_line(Line(3))
    window._speak_line(None)
    assert len(window.tts.calls) == n


def test_preferences_roundtrip_multiline_learning_and_tts_fields(qapp):
    cfg = AppConfig()
    cfg.overlay.max_sentences, cfg.overlay.learning, cfg.overlay.tts_read, cfg.overlay.ruby_scope = 4, True, "src", "dst"
    cfg.overlay.auto_height = False
    dlg = PreferencesDialog(cfg)
    assert dlg.ov_sentences.value() == 4 and not dlg.ov_auto_h.isChecked() and dlg.ov_learn.isChecked() and dlg.ov_tts_read.currentData() == "src"
    assert dlg.ov_ruby.currentData() == "dst"
    dlg.ov_sentences.setValue(1)
    dlg.ov_auto_h.setChecked(True)
    dlg.ov_learn.setChecked(False)
    dlg.ov_tts_read.setCurrentIndex(dlg.ov_tts_read.findData("dst"))
    dlg.ov_ruby.setCurrentIndex(dlg.ov_ruby.findData("src"))
    out = dlg.apply_to(cfg).overlay
    assert (out.max_sentences, out.auto_height, out.learning, out.tts_read, out.ruby_scope) == (1, True, False, "dst", "src")


def test_apply_preferences_pushes_learning_and_lines_to_the_live_widgets(qapp, window):
    new = window.current_config()
    new.overlay.learning, new.overlay.max_sentences = True, 5
    new.overlay.ruby_scope = "dst"
    window.apply_preferences(new)
    assert window.learn_cb.isChecked() and window.transcript._learning and window.transcript._scope == "dst"
    for i in range(1, 8):
        window.overlay.show_line(fin(i, src="a", dst=str(i)))
    assert [h.text() for h in reversed(window.overlay._hist) if not h.isHidden()] == ["3", "4", "5", "6"]


# ── TTS voice choice (no audio: a fake engine records what it is told) ───────

class FakeEngine:
    def __init__(self, locales):
        self._locales, self.locale, self.said, self.stopped = [QLocale(n) for n in locales], None, [], 0

    def availableLocales(self):
        return self._locales

    def setLocale(self, loc):
        self.locale = loc

    def availableVoices(self):
        return []

    def setVoice(self, _v):
        pass

    def say(self, text):
        self.said.append(text)

    def stop(self):
        self.stopped += 1

    def state(self):
        from PySide6.QtTextToSpeech import QTextToSpeech
        return QTextToSpeech.State.Ready


def test_tts_picks_a_voice_for_the_language_and_reports_missing_ones(qapp):
    pytest.importorskip("PySide6.QtTextToSpeech")
    t = Tts()
    t._engine = FakeEngine(["zh_CN", "en_GB", "en_US"])
    problems = []
    t.unavailable.connect(lambda key, msg: problems.append((key, msg)))
    t.speak("你好", "zh-Hans")
    t.speak("Hello", "en-US")
    assert t._engine.said == ["你好", "Hello"] and t._engine.locale.name() == "en_US"     # exact territory preferred
    assert t.voice_problem("zh-Hans") is None and t.voice_problem("en") is None
    t.speak("こんにちは", "ja")
    assert t._engine.said == ["你好", "Hello"]                                            # nothing spoken …
    (key, msg), = problems                                                              # … and the user is told why and how
    assert key == "ja" and "日本語" in msg and "重启" in msg
    assert t.voice_problem("ja")[0] == "ja"
    t.speak("   ", "en")
    assert len(t._engine.said) == 2


# ── transcript scrolling ─────────────────────────────────────────────────────

def settle(qapp, rounds=6):
    """Let the scroll area finish its (multi-pass, height-for-width) layout so the scroll range is final."""
    for _ in range(rounds):
        qapp.processEvents()


def filled_view(qapp, n=40) -> TranscriptView:
    v = TranscriptView()
    v.resize(500, 300)
    v.move(-6000, 0)
    v.show()
    for i in range(1, n + 1):
        v.upsert(fin(i, src=f"Sentence {i}", dst=f"第 {i} 句", src_lang="en", dst_lang="zh-Hans"))
    v._render()
    settle(qapp)
    return v


def test_transcript_follows_the_newest_line_while_at_the_bottom(qapp):
    v = filled_view(qapp)
    bar = v.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    v.upsert(fin(41, src="Sentence 41", dst="第 41 句"))
    v._render()
    settle(qapp)
    assert bar.value() == bar.maximum()


def test_transcript_stays_put_when_scrolled_up_and_new_lines_arrive(qapp):
    """The bug: every update rebuilt the document and reset the scroll position, so the view could not be dragged."""
    v = filled_view(qapp)
    bar = v.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    settle(qapp)
    pos = bar.value()
    assert pos < bar.maximum()
    for i in range(41, 60):                                       # new lines arrive, and the last one keeps streaming
        v.upsert(Line(i, src=f"Sentence {i}", dst=f"第 {i} 句" * 8, src_final=True))
        v._render()
        settle(qapp)
    assert bar.value() == pos
    v.upsert(fin(59, src="Sentence 59", dst="长" * 200, src_lang="en", dst_lang="zh-Hans"))
    v._render()
    settle(qapp)
    assert bar.value() == pos


def test_transcript_does_not_trim_rows_above_the_viewport_while_scrolled_up(qapp):
    from live_translator.ui import transcript as tr
    v = filled_view(qapp, n=tr._SHOWN)
    bar = v.verticalScrollBar()
    bar.setValue(0)
    settle(qapp)
    for i in range(tr._SHOWN + 1, tr._SHOWN + 11):
        v.upsert(fin(i, src=f"Sentence {i}", dst=f"第 {i} 句"))
    v._render()
    settle(qapp)
    assert len(v._rows) == tr._SHOWN + 10 and bar.value() == 0
    bar.setValue(bar.maximum())                                   # back at the bottom → old rows can go again
    settle(qapp)
    v.upsert(fin(tr._SHOWN + 11, src="one more", dst="再来一句"))
    v._render()
    assert len(v._rows) == tr._SHOWN


def test_transcript_clear_and_a_late_update_of_a_scrolled_out_line(qapp):
    from live_translator.ui import transcript as tr
    v = filled_view(qapp, n=tr._SHOWN + 5)
    assert 1 not in v._rows and v.line(1) is not None            # scrolled out of the window, still in the record
    v.upsert(fin(1, src="Sentence 1", dst="迟到的更新"))
    v._render()
    assert 1 not in v._rows                                       # does not reappear at the bottom
    assert "迟到的更新" in v.plain_text()
    v.clear_all()
    assert not v._rows and v.plain_text() == "" and not v._hint.isHidden()


@needs_pinyin
def test_transcript_rows_get_pinyin_and_speaker_in_learning_mode(qapp):
    v = filled_view(qapp, n=2)
    row = v._rows[1]
    assert row.speaker.isHidden() and not row.dst.has_ruby()
    v.set_learning(True)
    v._render()
    assert not row.speaker.isHidden() and row.dst.has_ruby()
    assert row.src.has_ruby() == ("espeakng-loader" not in ruby.missing_packages())    # English source: IPA
    got = []
    v.speak_requested.connect(got.append)
    row.speaker.click()
    assert got == [v.line(1)]
