from __future__ import annotations

import sys

from ..config import AppConfig


def run_gui(cfg: AppConfig) -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Live Translator")
    win = MainWindow(cfg)
    win.show()
    return app.exec()
