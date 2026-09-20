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

    def warmup(self) -> None:
        """Load / connect ahead of the first sentence so it doesn't pay the cold-start cost."""

    @abstractmethod
    def translate(self, text: str, src: str, tgt: str, *,
                  context: Sequence[tuple[str, str]] = (),
                  on_delta: DeltaCallback | None = None,
                  cancel: threading.Event | None = None) -> str:
        """``src`` may be "" when unknown. ``context`` = recent (source, translation) pairs, oldest first."""

    def close(self) -> None:
        pass
