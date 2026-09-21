"""macOS capture through a small ScreenCaptureKit helper (see native/macos/lt_sck_capture.swift).

ScreenCaptureKit can hand us either all system audio or the audio of chosen applications, without
BlackHole / multi-output devices. It needs macOS 13+, Xcode command line tools (to compile the helper
once) and the "Screen Recording" permission for the app that launches Python (Terminal, VS Code, ...).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..config import config_dir
from .base import AudioApp, AudioCallback, AudioSource, AudioSourceError, to_mono

log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent.parent / "native" / "macos" / "lt_sck_capture.swift"
_RATE, _CHANNELS = 48000, 2


def helper_path() -> Path:
    d = config_dir() / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d / "lt-sck-audio"


def bundled_helper() -> Path | None:
    """The helper precompiled at packaging time (scripts/build.py), present only in packaged builds."""
    root = getattr(sys, "_MEIPASS", None)
    if not root:
        return None
    path = Path(root) / "native" / "macos" / "lt-sck-audio"
    return path if path.is_file() else None


def ensure_helper() -> Path:
    """Path of the helper binary: the precompiled one shipped in a packaged build, otherwise compile the
    Swift source if the binary is missing or the source changed."""
    bundled = bundled_helper()
    if bundled:
        return bundled
    out = helper_path()
    stamp = out.with_suffix(".sha")
    digest = hashlib.sha256(_SRC.read_bytes()).hexdigest()
    if out.is_file() and stamp.is_file() and stamp.read_text() == digest:
        return out
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise AudioSourceError("需要 Xcode 命令行工具来编译音频捕获组件，请先运行: xcode-select --install")
    log.info("compiling ScreenCaptureKit helper ...")
    cmd = [swiftc, "-O", "-o", str(out), str(_SRC),
           "-framework", "ScreenCaptureKit", "-framework", "AVFoundation",
           "-framework", "CoreMedia", "-framework", "CoreGraphics"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioSourceError(f"编译音频捕获组件失败：\n{r.stderr[-800:]}")
    stamp.write_text(digest)
    return out


def _run_json(*args: str, timeout: float = 20) -> dict:
    r = subprocess.run([str(ensure_helper()), *args], capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise AudioSourceError((r.stderr or "音频捕获组件无输出").strip()[-400:])


def permission_granted() -> bool:
    try:
        return bool(_run_json("--check").get("permission"))
    except Exception:
        return False


def request_permission() -> bool:
    try:
        return bool(_run_json("--request").get("permission"))
    except Exception:
        return False


def list_audio_apps() -> list[AudioApp]:
    data = _run_json("--list-apps")
    apps = [AudioApp(key=a["bundle"], name=a["name"], pid=int(a.get("pid", 0))) for a in data.get("apps", [])]
    return sorted(apps, key=lambda a: a.name.lower())


class MacCaptureSource(AudioSource):
    """``bundle_id=None`` captures all system audio, otherwise only that application."""

    def __init__(self, bundle_id: str | None = None, name: str = ""):
        self.bundle_id = bundle_id
        self.name = name or (bundle_id or "系统全部声音")
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._err_tail: deque[str] = deque(maxlen=8)

    def start(self, on_audio: AudioCallback) -> None:
        binary = ensure_helper()
        if not permission_granted():
            request_permission()
            raise AudioSourceError(
                "需要「屏幕录制」权限（ScreenCaptureKit 即使只取音频也归在此权限下）。\n"
                "请在 系统设置 → 隐私与安全性 → 屏幕录制 中勾选启动本程序的终端/应用，然后完全退出并重新打开它。")
        cmd = [str(binary), "--rate", str(_RATE), "--channels", str(_CHANNELS)]
        if self.bundle_id:
            cmd += ["--app", self.bundle_id]
        self._stop.clear()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        time.sleep(0.7)                                   # let the helper fail fast on permission / missing app
        if self._proc.poll() is not None:
            msg = "; ".join(self._err_tail) or f"exit {self._proc.returncode}"
            self._proc = None
            raise AudioSourceError(f"无法开始捕获 {self.name}：{msg}")
        threading.Thread(target=self._reader, args=(on_audio,), daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        for line in iter(proc.stderr.readline, b""):
            self._err_tail.append(line.decode("utf-8", "replace").strip())

    def _reader(self, on_audio: AudioCallback) -> None:
        proc = self._proc
        block = (_RATE // 20) * _CHANNELS * 4                # 50 ms
        pending = b""
        while not self._stop.is_set():
            try:
                piece = proc.stdout.read(block - len(pending))
            except (OSError, ValueError):
                break
            if not piece:
                break
            pending += piece
            if len(pending) < block:                          # pipes may return short reads; never zero-pad
                continue
            audio = np.frombuffer(pending, dtype=np.float32).reshape(-1, _CHANNELS)
            pending = b""
            on_audio(to_mono(audio), _RATE)

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
