"""Language tables and code normalisation.

Internal language codes are ISO-639-1 style. Chinese is special: speech is just
``zh`` (Whisper does not distinguish scripts), while *translation targets* are
``zh-Hans`` / ``zh-Hant``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    code: str
    english: str   # name used inside LLM prompts
    native: str    # name shown in the UI
    nllb: str      # FLORES-200 code for the offline NLLB model


_TABLE = [
    Lang("en", "English", "English", "eng_Latn"),
    Lang("zh", "Chinese", "中文", "zho_Hans"),
    Lang("zh-Hans", "Simplified Chinese", "简体中文", "zho_Hans"),
    Lang("zh-Hant", "Traditional Chinese", "繁體中文", "zho_Hant"),
    Lang("ja", "Japanese", "日本語", "jpn_Jpan"),
    Lang("ko", "Korean", "한국어", "kor_Hang"),
    Lang("fr", "French", "Français", "fra_Latn"),
    Lang("de", "German", "Deutsch", "deu_Latn"),
    Lang("es", "Spanish", "Español", "spa_Latn"),
    Lang("pt", "Portuguese", "Português", "por_Latn"),
    Lang("it", "Italian", "Italiano", "ita_Latn"),
    Lang("ru", "Russian", "Русский", "rus_Cyrl"),
    Lang("ar", "Arabic", "العربية", "arb_Arab"),
    Lang("hi", "Hindi", "हिन्दी", "hin_Deva"),
    Lang("th", "Thai", "ไทย", "tha_Thai"),
    Lang("vi", "Vietnamese", "Tiếng Việt", "vie_Latn"),
    Lang("id", "Indonesian", "Bahasa Indonesia", "ind_Latn"),
    Lang("ms", "Malay", "Bahasa Melayu", "zsm_Latn"),
    Lang("tr", "Turkish", "Türkçe", "tur_Latn"),
    Lang("nl", "Dutch", "Nederlands", "nld_Latn"),
    Lang("pl", "Polish", "Polski", "pol_Latn"),
    Lang("uk", "Ukrainian", "Українська", "ukr_Cyrl"),
    Lang("cs", "Czech", "Čeština", "ces_Latn"),
    Lang("sv", "Swedish", "Svenska", "swe_Latn"),
    Lang("da", "Danish", "Dansk", "dan_Latn"),
    Lang("fi", "Finnish", "Suomi", "fin_Latn"),
    Lang("no", "Norwegian", "Norsk", "nob_Latn"),
    Lang("el", "Greek", "Ελληνικά", "ell_Grek"),
    Lang("he", "Hebrew", "עברית", "heb_Hebr"),
    Lang("fa", "Persian", "فارسی", "pes_Arab"),
    Lang("ro", "Romanian", "Română", "ron_Latn"),
    Lang("hu", "Hungarian", "Magyar", "hun_Latn"),
    Lang("tl", "Filipino", "Filipino", "tgl_Latn"),
]

LANGS: dict[str, Lang] = {l.code: l for l in _TABLE}

AUTO = "auto"

# (code, label) lists for the UI. Source: auto + spoken languages. Target: no bare "zh".
SOURCE_CHOICES: list[tuple[str, str]] = [(AUTO, "自动检测")] + [
    (l.code, l.native) for l in _TABLE if l.code not in ("zh-Hans", "zh-Hant")
]
TARGET_CHOICES: list[tuple[str, str]] = [(l.code, l.native) for l in _TABLE if l.code != "zh"]

# Whisper-side aliases → internal code (Cantonese is spoken Chinese).
_WHISPER_ALIASES = {"yue": "zh", "jw": "id", "nn": "no", "haw": "en"}
_NAME_TO_CODE = {l.english.lower(): l.code for l in _TABLE}
_NAME_TO_CODE.update({"chinese": "zh", "mandarin": "zh", "cantonese": "zh", "tagalog": "tl",
                      "norwegian": "no", "farsi": "fa"})


def base_lang(code: str | None) -> str:
    """``zh-Hant`` -> ``zh``; ``EN`` -> ``en``; falsy -> ``""``."""
    if not code:
        return ""
    return code.split("-")[0].lower()


def normalize_lang(value: str | None) -> str:
    """Map whatever an ASR backend returns (``"en"``, ``"english"``, ``"yue"``) to an internal code."""
    if not value:
        return ""
    v = value.strip().lower()
    if v in _NAME_TO_CODE:
        return _NAME_TO_CODE[v]
    v = _WHISPER_ALIASES.get(v, v)
    if v.startswith("zh"):
        return "zh"
    return v


def same_language(a: str | None, b: str | None) -> bool:
    ba, bb = base_lang(normalize_lang(a)), base_lang(normalize_lang(b))
    return bool(ba) and ba == bb


def lang_english(code: str) -> str:
    l = LANGS.get(code) or LANGS.get(base_lang(code))
    return l.english if l else code


def lang_native(code: str) -> str:
    l = LANGS.get(code) or LANGS.get(base_lang(code))
    return l.native if l else code


def nllb_code(code: str) -> str | None:
    l = LANGS.get(code) or LANGS.get(base_lang(code))
    return l.nllb if l else None
