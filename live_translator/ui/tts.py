"""Read subtitles aloud with the operating system's speech voices (Qt TextToSpeech: SAPI/WinRT on Windows,
AVSpeech on macOS, speech-dispatcher on Linux). Nothing is downloaded and nothing leaves the machine."""
from __future__ import annotations

import sys

from PySide6.QtCore import QLocale, QObject, Signal

from ..languages import base_lang, lang_native


def install_hint() -> str:
    if sys.platform == "win32":
        how = "Windows：设置 → 时间和语言 → 语音 → 管理语音 → 添加语音。"
    elif sys.platform == "darwin":
        how = "macOS：系统设置 → 辅助功能 → 朗读内容 → 系统语音 → 管理语音。"
    else:
        how = "请安装 speech-dispatcher 以及对应语言的语音。"
    return how + "装好后请重启本程序。"


class Tts(QObject):
    speaking_changed = Signal(bool)
    unavailable = Signal(str, str)      # (key, user-facing message); key = language code, or "engine" if there is none

    def __init__(self, parent=None):
        super().__init__(parent)
        self._engine = None
        self._failed = False
        self._key: tuple[str, str] | None = None      # (text, lang) being spoken

    @property
    def speaking(self) -> bool:
        from PySide6.QtTextToSpeech import QTextToSpeech
        return self._engine is not None and self._engine.state() == QTextToSpeech.State.Speaking

    def _ensure(self) -> bool:
        if self._engine is not None:
            return True
        if self._failed:
            return False
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            engine = QTextToSpeech(self)
            if not engine.availableLocales():
                raise RuntimeError("no voices")
        except Exception:
            self._failed = True
            return False
        engine.stateChanged.connect(lambda st: self.speaking_changed.emit(st == QTextToSpeech.State.Speaking))
        self._engine = engine
        return True

    def voice_problem(self, lang: str) -> tuple[str, str] | None:
        """``(key, message)`` if ``lang`` can't be read aloud on this machine, else None. Never speaks."""
        if not self._ensure():
            return "engine", f"这台电脑没有可用的语音合成引擎，朗读按钮暂时无法使用。\n{install_hint()}"
        want = QLocale(lang.replace("-", "_"))
        if not any(l.language() == want.language() for l in self._engine.availableLocales()):
            name = lang_native(lang)
            return base_lang(lang), f"系统里没有安装「{name}」语音，朗读按钮暂时无法使用。\n{install_hint()}"
        return None

    def _select_voice(self, lang: str) -> None:
        want = QLocale(lang.replace("-", "_"))
        same_lang = [l for l in self._engine.availableLocales() if l.language() == want.language()]
        exact = [l for l in same_lang if l.territory() == want.territory()]      # en_GB vs en_US, zh_TW vs zh_CN
        self._engine.setLocale((exact or same_lang)[0])
        voices = self._engine.availableVoices()
        if voices:
            self._engine.setVoice(voices[0])

    def speak(self, text: str, lang: str) -> None:
        text = text.strip()
        if not text:
            return
        problem = self.voice_problem(lang)
        if problem:
            self.unavailable.emit(*problem)
            return
        self._engine.stop()
        self._select_voice(lang)
        self._key = (text, lang)
        self._engine.say(text)

    def toggle(self, text: str, lang: str) -> None:
        """Speak ``text``; if exactly that text is already being read, stop instead."""
        if self.speaking and self._key == (text.strip(), lang):
            self.stop()
        else:
            self.speak(text, lang)

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._key = None
