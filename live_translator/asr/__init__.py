from __future__ import annotations

from ..config import AsrCfg
from .base import AsrError, AsrResult, LanguageTracker, Recognizer


def create_recognizer(cfg: AsrCfg) -> Recognizer:
    if cfg.mode == "remote":
        from .remote import RemoteWhisper
        return RemoteWhisper(cfg)
    from .local import LocalWhisper
    return LocalWhisper(cfg)


__all__ = ["AsrError", "AsrResult", "LanguageTracker", "Recognizer", "create_recognizer"]
