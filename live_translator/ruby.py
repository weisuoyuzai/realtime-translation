"""Learning mode: split subtitle text into drawable tokens and attach ruby (注音) — pinyin over Chinese characters,
furigana (hiragana) over Japanese kanji, Revised Romanization over Korean words and IPA over the words of the other
languages espeak-ng handles well.

Pure text logic (no Qt) so it can be tested headless. The annotators are optional packages (pypinyin, pykakasi,
espeakng-loader; Korean needs nothing): without one, text in that language is simply shown without ruby.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .languages import base_lang
from .text_utils import guess_language

try:
    from pypinyin import Style, pinyin as _pinyin
except ImportError:                                   # optional: learning mode then just has no annotations
    _pinyin = None

try:
    import pykakasi
except ImportError:
    pykakasi = None

try:
    import espeakng_loader
except ImportError:
    espeakng_loader = None

_kks = None                                          # created on first use (loads the dictionary, ~0.15 s)


@dataclass(frozen=True)
class Token:
    base: str
    ruby: str = ""
    space: bool = False           # whitespace before this token (word gap in Latin-style text)
    glue: bool = False            # may not start a line: closing punctuation, or the token after an opening bracket


_HAN = r"㐀-䶿一-鿿"
_KANJI = _HAN + r"々〆ヶ"                               # what Japanese counts as a kanji (iteration marks included)
_KANJI_RUN = re.compile(f"[{_KANJI}]+")
_PIECE = re.compile(
    f"(?P<han>[{_KANJI}]+)"
    r"|(?P<sp>\s+)"
    f"|(?P<word>[^\\s{_KANJI}\\u3040-\\u30ff\\u3000-\\u303f\\uff00-\\uffef]+)"
    r"|(?P<one>.)")                                   # kana, CJK / full-width punctuation: one token each

_CLOSERS = set("，。、！？；：）」』》】〕〉…—～·”’,.!?;:)]}%")
_OPENERS = set("（「『《【〔〈“‘([{")


# espeak-ng voice per language that gets IPA. Left out: Thai (espeak doesn't segment it — the output is garbage),
# Vietnamese (the spelling already marks the tones; espeak writes them as digits) and Filipino (no espeak voice).
_IPA_VOICES = {"en": "en-us", "fr": "fr", "de": "de", "es": "es", "pt": "pt-br", "it": "it", "nl": "nl", "sv": "sv",
               "da": "da", "no": "nb", "fi": "fi", "pl": "pl", "cs": "cs", "ro": "ro", "hu": "hu", "tr": "tr",
               "id": "id", "ms": "ms", "ru": "ru", "uk": "uk", "el": "el", "ar": "ar", "fa": "fa", "he": "he",
               "hi": "hi"}
_NON_LATIN = {"ru", "uk", "el", "ar", "fa", "he", "hi"}


def available() -> bool:
    return not missing_packages()


def missing_packages() -> list[str]:
    return [name for name, mod in (("pypinyin", _pinyin), ("pykakasi", pykakasi), ("espeakng-loader", espeakng_loader))
            if mod is None]


def ruby_kind(text: str, lang: str = "") -> str:
    """Which annotation this text gets: "zh" (pinyin), "ja" (furigana), "ko" (romanization), a language code from
    ``_IPA_VOICES`` (IPA) or "" (none). Uses the reported language, else a script guess (any kana ⇒ Japanese)."""
    b = base_lang(lang or guess_language(text))
    return b if b in ("zh", "ja", "ko") or b in _IPA_VOICES else ""


# ── Chinese ──────────────────────────────────────────────────────────────────

def _han_ruby(run: str) -> list[str]:
    """One pinyin syllable per character of a Han run (pypinyin resolves polyphones from context)."""
    if _pinyin is None:
        return [""] * len(run)
    syl = [s[0] if s else "" for s in _pinyin(run, style=Style.TONE)]
    if len(syl) != len(run):
        return [""] * len(run)
    return [r if re.match(f"[{_HAN}]", c) else "" for c, r in zip(run, syl)]


# ── Japanese ─────────────────────────────────────────────────────────────────

def _to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _align(orig: str, hira: str) -> list[tuple[str, str]]:
    """Split one word into (kanji run, reading) pairs, leaving okurigana out of the ruby: 取り組み/とりくみ →
    [(取, と), (組, く)]. If the reading can't be lined up with the kana in the word, the whole word gets one ruby."""
    parts = re.findall(f"[{_KANJI}]+|[^{_KANJI}]+", orig)
    pattern = "".join("(.+?)" if _KANJI_RUN.fullmatch(p) else re.escape(_to_hira(p)) for p in parts)
    m = re.fullmatch(pattern, hira)
    if not m:
        return [(orig, hira)]
    readings = iter(m.groups())
    return [(p, next(readings) if _KANJI_RUN.fullmatch(p) else "") for p in parts]


@lru_cache(maxsize=512)
def _kanji_readings(text: str) -> dict[int, tuple[int, str]]:
    """start index → (length, hiragana reading) for every run of kanji in Japanese ``text``."""
    global _kks
    if pykakasi is None:
        return {}
    if _kks is None:
        _kks = pykakasi.kakasi()
    out: dict[int, tuple[int, str]] = {}
    pos = 0
    for item in _kks.convert(text):
        orig, hira = item["orig"], _to_hira(item["hira"])
        if orig == "今日は" and text[pos + 3:pos + 4] not in ("", "、", "。", "！", "!", "…", " "):
            hira = "きょうは"                          # pykakasi always reads it as the greeting こんにちは
        if _KANJI_RUN.search(orig) and orig != hira:
            at = pos
            for piece, reading in _align(orig, hira):
                if reading:
                    out[at] = (len(piece), reading)
                at += len(piece)
        pos += len(orig)
    return out


# ── Korean ───────────────────────────────────────────────────────────────────

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
_CHO_R = dict(zip(_CHO, ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p",
                         "h"]))
_JUNG_R = dict(zip(_JUNG, ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we",
                           "wi", "yu", "eu", "ui", "i"]))
_JONG_R = {"": "", "ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ng"}
_SPLIT = {"ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ",
          "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}
_NEUTRAL = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ",
            "ㅎ": "ㄷ", "ㅍ": "ㅂ", "ㅄ": "ㅂ", "ㄿ": "ㅂ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄻ": "ㅁ", "ㄼ": "ㄹ", "ㄽ": "ㄹ",
            "ㄾ": "ㄹ", "ㅀ": "ㄹ"}
_ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}


def _ko_join(jong: str, cho: str, jung: str) -> tuple[str, str]:
    """Sound changes across a syllable boundary (final ``jong``, next initial ``cho`` and vowel ``jung``): liaison,
    aspiration, palatalization, nasalization, ㄹ assimilation. Returns the new (final, initial)."""
    if not jong:
        return jong, cho
    if cho == "ㅇ":                                    # liaison: the final moves over (ㅇ stays, ㅎ goes silent)
        if jong == "ㅇ":
            return jong, cho
        if jong == "ㅎ":                               # 좋아 → 조아
            return "", cho
        if jong in ("ㄶ", "ㅀ"):                       # 많이 → 마니
            return "", _SPLIT[jong][0]
        if jong in _SPLIT:                             # 닭이 → 달기
            return _SPLIT[jong][0], _SPLIT[jong][1]
        if jong in ("ㄷ", "ㅌ") and jung == "ㅣ":        # 같이 → 가치
            return "", "ㅈ" if jong == "ㄷ" else "ㅊ"
        return "", jong
    if cho == "ㅎ":                                    # 축하 → 추카, 닫히다 → 다치다
        first, last = _SPLIT.get(jong, ("", jong))
        n = _NEUTRAL.get(last, last)
        if n in _ASPIRATE and last != "ㅎ":
            return first, "ㅊ" if last in ("ㅈ", "ㅊ") or (n == "ㄷ" and jung == "ㅣ") else _ASPIRATE[n]
    if jong in ("ㅎ", "ㄶ", "ㅀ"):                     # 좋다 → 조타, 않는 → 안는, 싫네 → 실레
        rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[jong]
        if cho in _ASPIRATE:
            return rest, _ASPIRATE[cho]
        if cho == "ㅅ":
            return rest, "ㅆ"
        if cho == "ㄴ":
            jong = rest or "ㄴ"
    jong = _NEUTRAL.get(jong, jong)
    if cho in ("ㄴ", "ㅁ"):                            # 합니다 → 함니다
        jong = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}.get(jong, jong)
    elif cho == "ㄹ":
        if jong in ("ㄱ", "ㅂ"):                       # 국립 → 궁닙
            jong, cho = {"ㄱ": "ㅇ", "ㅂ": "ㅁ"}[jong], "ㄴ"
        elif jong in ("ㅁ", "ㅇ"):                     # 종로 → 종노
            cho = "ㄴ"
        elif jong == "ㄴ":                             # 신라 → 실라
            jong = "ㄹ"
    if jong == "ㄹ" and cho == "ㄴ":                   # 설날 → 설랄
        cho = "ㄹ"
    return jong, cho


@lru_cache(maxsize=2048)
def _ko_romanize(word: str) -> str:
    """Revised Romanization of the Hangul syllables in ``word``, as pronounced (국물 → gungmul, 신라 → silla)."""
    syl = []
    for ch in word:
        k = ord(ch) - 0xAC00
        if 0 <= k < 11172:
            syl.append([_CHO[k // 588], _JUNG[k % 588 // 28], _JONG[k % 28]])
    for a, b in zip(syl, syl[1:]):
        a[2], b[0] = _ko_join(a[2], b[0], b[1])
    out = []
    for i, (cho, jung, jong) in enumerate(syl):
        r = "l" if cho == "ㄹ" and i and syl[i - 1][2] == "ㄹ" else _CHO_R[cho]
        out.append(r + _JUNG_R[jung] + _JONG_R[_NEUTRAL.get(jong, jong)])
    return "".join(out)


# ── IPA (espeak-ng) ──────────────────────────────────────────────────────────

_espeak = None                                        # the loaded library; False once loading failed
_espeak_lock = threading.Lock()                       # espeak-ng keeps global state: one caller at a time


def _load_espeak():
    global _espeak
    if _espeak is None:
        _espeak = False
        try:
            lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
            data = espeakng_loader.get_data_path()
            data_b = data.encode("mbcs") if sys.platform == "win32" else os.fsencode(data)   # espeak opens it with fopen
            lib.espeak_Initialize.restype = ctypes.c_int
            lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            lib.espeak_SetVoiceByName.restype = ctypes.c_int
            lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
            lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
            lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
            # AUDIO_OUTPUT_RETRIEVAL (no sound device), espeakINITIALIZE_DONT_EXIT (report errors, never exit())
            if lib.espeak_Initialize(1, 0, data_b, 0x8000) > 0:
                _espeak = lib
        except (OSError, AttributeError, RuntimeError, UnicodeError):
            pass
    return _espeak


@lru_cache(maxsize=4096)
def _word_ipa(lang: str, word: str) -> str:
    """IPA for one word (stress marks included), or "" when espeak isn't available / has no voice."""
    if espeakng_loader is None:
        return ""
    if lang == "en" and word in ("a", "A"):            # on its own espeak reads the letter name /eɪ/
        return "ə"
    with _espeak_lock:
        lib = _load_espeak()
        if not lib or lib.espeak_SetVoiceByName(_IPA_VOICES[lang].encode()) != 0:
            return ""
        buf = ctypes.create_string_buffer(word.encode())
        ptr = ctypes.c_void_p(ctypes.addressof(buf))
        parts = []
        while ptr.value:                              # one call per clause; a single word is normally one
            out = lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 0x02)    # espeakCHARS_UTF8, espeakPHONEMES_IPA
            if out:
                parts.append(out.decode("utf-8", "replace"))
    ipa = re.sub(r'\([a-z-]+\)|[_"]', "", " ".join(parts))   # language-switch markers such as (en), pauses, …
    return " ".join(ipa.split()).strip("-")


def _word_core(word: str) -> str:
    """The word without the punctuation / symbols around it ("world," → "world")."""
    i, j = 0, len(word)
    while i < j and unicodedata.category(word[i])[0] in "PS":
        i += 1
    while j > i and unicodedata.category(word[j - 1])[0] in "PS":
        j -= 1
    return word[i:j]


def _word_ruby(kind: str, word: str) -> str:
    """Ruby for one space-delimited word of Korean / IPA text. Words in another script (an English name inside
    Russian text, Cyrillic inside English) and numbers get none."""
    if kind == "ko":
        return _ko_romanize(word) if any("가" <= c <= "힣" for c in word) else ""
    core = _word_core(word)
    letters = [c for c in core if unicodedata.category(c).startswith("L")]
    latin = sum(c < "ɐ" for c in letters)             # Basic Latin … Latin Extended-B
    if not letters or (latin if kind in _NON_LATIN else latin < len(letters)):
        return ""
    return _word_ipa(kind, core)


# ── tokens ───────────────────────────────────────────────────────────────────

def tokenize(text: str, lang: str = "", ruby: bool = False) -> list[Token]:
    """Tokens in reading order. ``ruby=True`` annotates the text in its own language: pinyin per Han character in
    Chinese, furigana per kanji run in Japanese, romanization per word in Korean, IPA per word elsewhere."""
    kind = ruby_kind(text, lang) if ruby else ""
    readings = _kanji_readings(text) if kind == "ja" else {}
    out: list[Token] = []
    space = glue_next = False
    for m in _PIECE.finditer(text):
        s = m.group()
        if m.lastgroup == "sp":
            space = bool(out)
            continue
        if m.lastgroup == "han" and kind == "ja":
            pieces, i = [], 0
            while i < len(s):
                n, r = readings.get(m.start() + i, (1, ""))
                n = min(n, len(s) - i)
                pieces.append((s[i:i + n], r))
                i += n
        elif m.lastgroup == "han":
            pieces = list(zip(s, _han_ruby(s) if kind == "zh" else [""] * len(s)))
        elif m.lastgroup == "word" and kind not in ("", "zh", "ja"):
            pieces = [(s, _word_ruby(kind, s))]
        else:
            pieces = [(s, "")]
        for i, (base, r) in enumerate(pieces):
            out.append(Token(base, r, space=space and i == 0, glue=glue_next or (not r and all(c in _CLOSERS for c in base))))
            glue_next = m.lastgroup != "han" and all(c in _OPENERS for c in base)
        space = False
    return out
