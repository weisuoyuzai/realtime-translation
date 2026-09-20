"""Text helpers: cleaning LLM output, script detection, sentence splitting."""
from __future__ import annotations

import re

from .languages import base_lang

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*", re.S | re.I)
_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_LABEL = re.compile(r"^\s*(?:translation|translated text|译文|翻译|訳|번역)\s*[:：]\s*", re.I)


def visible_text(raw: str) -> str:
    """Text a viewer should see from a (possibly still streaming) model output: hides <think> blocks."""
    raw = _THINK_CLOSED.sub("", raw)
    return _THINK_OPEN.sub("", raw)


def clean_translation(raw: str) -> str:
    text = visible_text(raw).strip()
    text = _FENCE.sub("", text).strip()
    text = _LABEL.sub("", text)
    if len(text) >= 2 and text[0] in "\"“「『'" and text[-1] in "\"”」』'":
        inner = text[1:-1]
        if text[0] not in inner and text[-1] not in inner:      # strip wrapping quotes only
            text = inner.strip()
    return re.sub(r"\s*\n+\s*", " ", text).strip()


# ── scripts ──────────────────────────────────────────────────────────────────

_RANGES = {
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "kana": [(0x3040, 0x30FF)],
    "hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "hebrew": [(0x0590, 0x05FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "devanagari": [(0x0900, 0x097F)],
    "greek": [(0x0370, 0x03FF)],
}

# language → the script(s) its text must contain (Latin-script languages are not checked)
_LANG_SCRIPTS = {
    "zh": ("cjk",), "ja": ("kana", "cjk"), "ko": ("hangul",), "ru": ("cyrillic",), "uk": ("cyrillic",),
    "ar": ("arabic",), "fa": ("arabic",), "he": ("hebrew",), "th": ("thai",), "hi": ("devanagari",),
    "el": ("greek",),
}


def _count(text: str, script: str) -> int:
    rs = _RANGES[script]
    return sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in rs))


def guess_language(text: str) -> str:
    """Very rough script-based guess, used only when the recognizer did not report a language."""
    if _count(text, "kana"):
        return "ja"
    for lang in ("ko", "ru", "ar", "he", "th", "hi", "el"):
        if _count(text, _LANG_SCRIPTS[lang][0]):
            return lang
    if _count(text, "cjk"):
        return "zh"
    return "en"


def looks_like_language(text: str, lang: str, source_text: str = "") -> bool:
    """Sanity-check a translation: right script for the target, and not just an echo of the source."""
    b = base_lang(lang)
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3:
        return True                                            # numbers / short interjections: nothing to check
    scripts = _LANG_SCRIPTS.get(b)
    if scripts:
        return any(_count(text, s) for s in scripts)
    # Latin-script target: reject output that is mostly CJK/Cyrillic/... (the model copied the source)
    foreign = sum(_count(text, s) for s in _RANGES)
    return foreign / letters < 0.3


_SENT_END = re.compile(r"(?<=[.!?。！？…])\s*")


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split on sentence punctuation; over-long pieces are cut at commas/spaces so a model never sees a wall of text."""
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = max(p.rfind(c, 0, max_chars) for c in (",", "，", "、", ";", "；", " "))
            cut = cut if cut > max_chars // 3 else max_chars
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out or [text]


def is_cjk_lang(lang: str) -> bool:
    return base_lang(lang) in ("zh", "ja")


_FULLWIDTH = {"zh": {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"},
              "ja": {",": "、", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}}


def normalize_cjk_punct(text: str, lang: str) -> str:
    """Machine-translation models often emit ASCII punctuation in Chinese/Japanese output ("你好,世界.").
    Convert it to full-width when it directly follows a CJK character; digits and Latin text are untouched."""
    table = _FULLWIDTH.get(base_lang(lang))
    if not table:
        return text
    out = []
    for i, ch in enumerate(text):
        if ch in table and i > 0 and (_count(text[i - 1], "cjk") or _count(text[i - 1], "kana")
                                      or text[i - 1] in "，。！？：；、」』）"):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not (ch == "." and nxt.isdigit()):
                ch = table[ch]
        out.append(ch)
    return "".join(out)


_TRAILING_ELLIPSIS = re.compile(r"\s*(?:\.{2,}|…+|。{2,})\s*$")


def strip_trailing_ellipsis(text: str) -> str:
    """ASR adds "..." when a speaker trails off. MT models read it as "keep going" and invent a continuation."""
    return _TRAILING_ELLIPSIS.sub("", text).strip() or text


def _is_short_fragment(text: str, max_words: int) -> bool:
    words = text.split()
    if len(words) > 1 or text.isascii():
        return len(words) <= max_words
    return len(text.strip()) <= 2 * max_words        # unspaced scripts (Chinese/Japanese): count characters


def trim_runaway(src: str, out: str, max_words: int = 6) -> str:
    """Machine-translation models sometimes keep "talking" after a very short fragment, or repeat themselves.
    Collapse exactly repeated sentences; and if the source is only a few words, keep just the first output sentence
    (a second sentence from ≤ 6 words is essentially never legitimate). Longer sources are left untouched so that
    real content is never dropped."""
    sentences = split_sentences(out)
    deduped: list[str] = []
    for sent in sentences:
        if not deduped or sent.strip().lower() != deduped[-1].strip().lower():
            deduped.append(sent)
    if _is_short_fragment(src, max_words) and len(deduped) > 1:
        deduped = deduped[:1]
    sep = "" if is_cjk_lang(guess_language(out)) else " "
    return sep.join(deduped) if len(deduped) != len(sentences) else out
