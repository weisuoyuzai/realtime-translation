from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Sequence

DeltaCallback = Callable[[str], None]     # receives the *accumulated* translation so far


class TranslateError(RuntimeError):
    """Failure with a user-presentable message."""


class Translator(ABC):
    name = "translator"
    streaming = False
    can_correct = False     # True → ``correct`` does real work (currently only the LLM backend)

    def warmup(self) -> None:
        """Load / connect ahead of the first sentence so it doesn't pay the cold-start cost."""

    @abstractmethod
    def translate(self, text: str, src: str, tgt: str, *,
                  context: Sequence[tuple[str, str]] = (),
                  on_delta: DeltaCallback | None = None,
                  cancel: threading.Event | None = None) -> str:
        """``src`` may be "" when unknown. ``context`` = recent (source, translation) pairs, oldest first."""

    def correct(self, text: str, lang: str, *, context: Sequence[tuple[str, str]] = ()) -> str:
        """Best-effort fix of likely ASR mis-hearings in ``text`` itself (homophones, mangled names), using
        ``context`` = recent (source, translation) pairs for continuity. Same language in and out; never a
        translation. The default no-op is what backends without ``can_correct`` fall back to."""
        return text

    def close(self) -> None:
        pass
