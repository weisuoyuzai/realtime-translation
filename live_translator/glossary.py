"""User glossary: one entry per line, ``term = translation`` or just ``term`` (meaning "keep as is")."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SEP = re.compile(r"\s*(?:=>|->|→|＝|=|:|：)\s*")


@dataclass(frozen=True)
class Term:
    source: str
    target: str = ""       # empty → keep the term unchanged in the translation


def parse_glossary(text: str) -> list[Term]:
    terms: list[Term] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _SEP.split(line, maxsplit=1)
        src = parts[0].strip()
        if src:
            terms.append(Term(src, parts[1].strip() if len(parts) > 1 else ""))
    return terms


def asr_hint(terms: list[Term], topic: str = "", limit: int = 400) -> str:
    """Text for Whisper's ``initial_prompt`` / ``hotwords``: nudges spelling of names and jargon."""
    words = ", ".join(t.source for t in terms)
    hint = f"{topic.strip()}. {words}".strip(" .") if topic.strip() else words
    return hint[:limit]


def prompt_block(terms: list[Term]) -> str:
    """Glossary section for the translation system prompt."""
    if not terms:
        return ""
    lines = [f"- {t.source} → {t.target}" if t.target else f"- {t.source} (keep unchanged)" for t in terms]
    return "Glossary (always follow it):\n" + "\n".join(lines)
