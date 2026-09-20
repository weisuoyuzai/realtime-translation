"""Marshals pipeline callbacks (worker threads) onto the Qt UI thread."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..models import Line


class Bridge(QObject):
    line = Signal(object)          # Line
    status = Signal(str, str)      # level, message
    level = Signal(float)          # audio RMS
    stopped = Signal()             # pipeline fully shut down
    apps = Signal(list)            # list[AudioApp] from a background refresh
    test_result = Signal(str, bool, str)   # which test, ok, message

    # Emitting a Signal from a non-Qt thread is safe: delivery is queued to the receiver's thread.
    def on_line(self, line: Line) -> None:
        self.line.emit(line)

    def on_status(self, level: str, message: str) -> None:
        self.status.emit(level, message)

    def on_level(self, rms: float) -> None:
        self.level.emit(rms)
