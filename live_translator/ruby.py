"""Learning mode: split subtitle text into drawable tokens and attach ruby (注音) — pinyin over Chinese characters,
furigana (hiragana) over Japanese kanji.

Pure text logic (no Qt) so it can be tested headless. Both annotators are optional packages (pypinyin, pykakasi):
without one, text in that language is simply shown without ruby.
"""
from __future__ import annotations

import re
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

_kks = None                                           # created on first use (loads the dictionary, ~0.15 s)


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


def available() -> bool:
    return _pinyin is not None and pykakasi is not None


def missing_packages() -> list[str]:
    return [name for name, mod in (("pypinyin", _pinyin), ("pykakasi", pykakasi)) if mod is None]


def ruby_kind(text: str, lang: str = "") -> str:
    """Which annotation this text gets: "zh" (pinyin), "ja" (furigana) or "" (none). Uses the reported language,
    else a script guess (any kana ⇒ Japanese)."""
    b = base_lang(lang or guess_language(text))
    return b if b in ("zh", "ja") else ""


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


# ── tokens ───────────────────────────────────────────────────────────────────

def tokenize(text: str, lang: str = "", ruby: bool = False) -> list[Token]:
    """Tokens in reading order. ``ruby=True`` annotates Han characters: pinyin per character in Chinese text,
    furigana per kanji run in Japanese text."""
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
        else:
            pieces = [(s, "")]
        for i, (base, r) in enumerate(pieces):
            out.append(Token(base, r, space=space and i == 0, glue=glue_next or (not r and all(c in _CLOSERS for c in base))))
            glue_next = m.lastgroup != "han" and all(c in _OPENERS for c in base)
        space = False
    return out
