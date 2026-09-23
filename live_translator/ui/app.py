from __future__ import annotations

import sys

from ..config import AppConfig


def run_gui(cfg: AppConfig) -> int:
    from PySide6.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    if sys.platform == "win32":                 # own taskbar identity, so Windows shows our icon rather than python.exe's
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LiveTranslator.LiveTranslator")
        except (AttributeError, OSError):
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Live Translator")
    app.setWindowIcon(theme.app_icon())
    win = MainWindow(cfg)
    win.show()
    return app.exec()
