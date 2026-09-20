from __future__ import annotations

from typing import Callable

from ..config import TranslateCfg
from .base import TranslateError, Translator


def create_translator(cfg: TranslateCfg, progress: Callable[[str], None] | None = None) -> Translator | None:
    """Build the translator selected in settings; ``None`` means "transcribe only"."""
    mode = cfg.mode
    if mode == "none":
        return None
    if mode in ("llm_remote", "llm_local"):
        from .llm import LLMTranslator
        ep = cfg.remote if mode == "llm_remote" else cfg.local
        return LLMTranslator(ep.base_url, ep.api_key, ep.model, extra_body=ep.extra_body,
                             glossary=cfg.glossary, topic=cfg.topic,
                             label="远程" if mode == "llm_remote" else "本地")
    if mode == "nllb":
        from .nllb import NllbTranslator
        return NllbTranslator(cfg.nllb_model, progress=progress)
    if mode == "deepl":
        from .deepl import DeepLTranslator
        return DeepLTranslator(cfg.deepl_key, cfg.deepl_free)
    raise TranslateError(f"未知的翻译方式：{mode}")


__all__ = ["TranslateError", "Translator", "create_translator"]
