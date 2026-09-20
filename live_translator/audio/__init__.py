"""Audio capture front-door: ``create_source`` builds the right source for the current OS and settings."""
from __future__ import annotations

import sys

from ..config import AudioCfg
from .base import AudioApp, AudioSource, AudioSourceError

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def app_capture_supported() -> bool:
    return IS_WINDOWS or IS_MAC


def list_audio_apps() -> list[AudioApp]:
    if IS_WINDOWS:
        from . import windows
        return windows.list_audio_apps()
    if IS_MAC:
        from . import macos
        return macos.list_audio_apps()
    return []


def create_source(cfg: AudioCfg) -> AudioSource:
    kind = cfg.source
    if kind == "mic":
        from .mic import MicSource
        return MicSource(cfg.mic_device)
    if kind == "file":
        from .file import FileSource
        return FileSource(cfg.file_path)

    if IS_WINDOWS:
        from . import windows
        if kind == "app":
            if not cfg.app_key:
                raise AudioSourceError("请先选择要捕获的应用")
            pid = windows.resolve_app_pid(cfg.app_key)
            if pid is None:
                raise AudioSourceError(f"找不到正在运行的 {cfg.app_name or cfg.app_key}，请先启动它")
            return windows.ProcessLoopbackSource(pid, cfg.app_name or cfg.app_key)
        return windows.SystemLoopbackSource()

    if IS_MAC:
        from . import macos
        if kind == "app":
            if not cfg.app_key:
                raise AudioSourceError("请先选择要捕获的应用")
            return macos.MacCaptureSource(cfg.app_key, cfg.app_name)
        return macos.MacCaptureSource(None)

    raise AudioSourceError("当前系统仅支持 Windows 与 macOS 的系统/应用音频捕获；可改用麦克风。")


__all__ = ["AudioApp", "AudioSource", "AudioSourceError", "app_capture_supported",
           "create_source", "list_audio_apps"]
