"""Windows capture: whole-system loopback and per-application (process-tree) loopback.

* System: WASAPI loopback on the default render device (via PyAudioWPatch).
* Application: WASAPI *process loopback* (Windows 10 2004+ / 11) via ``proc-tap``. It captures the
  target process **and its children**, which is what makes browsers / Electron apps work — their
  sound is produced by a child "audio service" process, not the process you see in the taskbar.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
from ctypes import wintypes

import numpy as np

from .base import AudioApp, AudioCallback, AudioSource, AudioSourceError, to_mono

log = logging.getLogger(__name__)


# ── capture ──────────────────────────────────────────────────────────────────

def decode_pcm(data: bytes, bits: int, channels: int) -> np.ndarray:
    """Raw WASAPI buffer -> mono float32. proc-tap's native layer delivers float32 or 16-bit PCM."""
    if bits == 32:
        x = np.frombuffer(data, dtype=np.float32)
    elif bits == 16:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"unsupported sample size: {bits} bits")
    usable = len(x) - len(x) % max(1, channels)                  # never let a torn buffer break the reshape
    return to_mono(x[:usable].reshape(-1, channels)) if channels > 1 else np.ascontiguousarray(x[:usable])


class ProcessLoopbackSource(AudioSource):
    """Captures one process tree through WASAPI process loopback (proc-tap's native module, used directly).

    Why not ``proctap.ProcessAudioCapture``: its Python layer imports scipy the first time a capture is created,
    which took 10+ seconds on a loaded machine and is not needed here (we resample with soxr ourselves).

    The native capture is created, read and stopped on ONE dedicated long-lived thread. WASAPI/COM objects are tied
    to the apartment of the thread that created them: if that thread exits (for example the short-lived pipeline
    start-up thread) the stream silently stops delivering audio, with no error.
    """

    START_TIMEOUT_S = 30

    def __init__(self, pid: int, name: str = ""):
        self.pid = pid
        self.name = name or f"pid {pid}"
        self._cb: AudioCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._ready = threading.Event()
        self._error: AudioSourceError | None = None

    def start(self, on_audio: AudioCallback) -> None:
        try:
            from proctap import _native  # noqa: F401
        except ImportError as e:
            raise AudioSourceError("缺少依赖 proc-tap，请执行: pip install proc-tap") from e
        self._cb = on_audio
        self._error = None
        self._stop_evt.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="proc-capture", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.START_TIMEOUT_S):
            self.stop()
            raise AudioSourceError(f"启动 {self.name} 的音频捕获超时")
        if self._error is not None:
            raise self._error

    def _fail(self, msg: str) -> None:
        self._error = AudioSourceError(
            f"无法捕获 {self.name} (pid {self.pid}) 的音频：{msg}\n"
            "按应用捕获需要 Windows 10 2004 或更高版本，且进程必须仍在运行。")
        self._ready.set()

    def _run(self) -> None:
        from proctap._native import ProcessLoopback
        try:
            native = ProcessLoopback(self.pid)
            native.start()
            fmt = native.get_format()
            if not native.is_process_specific():
                native.stop()
                # Silently capturing the whole system would break the promise of "only this application".
                return self._fail("系统没有启用按进程捕获（已退化为全系统捕获）")
            bits, channels, rate = int(fmt["bits_per_sample"]), int(fmt["channels"]), int(fmt["sample_rate"])
        except Exception as e:
            return self._fail(str(e))
        self._ready.set()
        try:
            while not self._stop_evt.is_set():
                data = native.read()
                if not data:
                    self._stop_evt.wait(0.002)
                    continue
                cb = self._cb
                if cb is not None:
                    cb(decode_pcm(data, bits, channels), rate)
        except Exception:
            log.exception("process capture loop failed")
        finally:
            try:
                native.stop()
            except Exception:
                log.exception("stopping process capture")

    def stop(self) -> None:
        self._cb = None
        self._stop_evt.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3)


class SystemLoopbackSource(AudioSource):
    name = "系统全部声音"

    def __init__(self):
        self._pa = None
        self._stream = None

    def start(self, on_audio: AudioCallback) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise AudioSourceError("缺少依赖 PyAudioWPatch，请执行: pip install PyAudioWPatch") from e
        try:
            pa = pyaudio.PyAudio()
            info = pa.get_default_wasapi_loopback()
        except Exception as e:
            raise AudioSourceError(f"找不到可用的 WASAPI loopback 输出设备：{e}") from e
        channels = int(info["maxInputChannels"])
        rate = int(info["defaultSampleRate"])

        def _cb(in_data, frame_count, time_info, status):
            if in_data:
                x = np.frombuffer(in_data, dtype=np.float32)
                on_audio(to_mono(x.reshape(-1, channels)), rate)
            return (None, pyaudio.paContinue)

        try:
            self._stream = pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                                   input_device_index=info["index"], frames_per_buffer=rate // 20,
                                   stream_callback=_cb)
        except Exception as e:
            pa.terminate()
            raise AudioSourceError(f"无法打开 loopback 设备 {info.get('name')}：{e}") from e
        self._pa = pa
        self.name = f"系统声音（{info.get('name', '')}）"

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        pa, self._pa = self._pa, None
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        finally:
            if pa is not None:
                pa.terminate()


# ── application discovery ────────────────────────────────────────────────────

def _visible_window_titles() -> dict[int, str]:
    """pid → title of its first visible, un-cloaked top-level window."""
    user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
    titles: dict[int, str] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        cloaked = wintypes.DWORD(0)
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))  # DWMWA_CLOAKED
        if cloaked.value:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        titles.setdefault(pid.value, buf.value)
        return True

    user32.EnumWindows(_enum, 0)
    return titles


def _audio_session_pids() -> dict[int, bool]:
    """pid → has an *active* audio session, for every process that owns an audio session."""
    try:
        import comtypes
        try:
            comtypes.CoInitialize()
        except OSError:
            pass
        from pycaw.pycaw import AudioUtilities
    except ImportError:
        return {}
    out: dict[int, bool] = {}
    try:
        for s in AudioUtilities.GetAllSessions():
            pid = s.ProcessId
            if pid:
                out[pid] = out.get(pid, False) or s.State == 1     # AudioSessionStateActive
    except Exception:
        log.exception("enumerating audio sessions")
    return out


def _root_process(proc):
    """Climb to the top-most ancestor that runs the *same executable* (chrome.exe helpers → main chrome.exe)."""
    import psutil
    try:
        name = proc.name().lower()
    except psutil.Error:
        return proc
    cur = proc
    while True:
        try:
            parent = cur.parent()
            if parent is None or parent.name().lower() != name:
                return cur
            cur = parent
        except psutil.Error:
            return cur


_SKIP_EXES = {"audiodg.exe", "textinputhost.exe", "applicationframehost.exe", "searchhost.exe",
              "startmenuexperiencehost.exe", "shellexperiencehost.exe", "systemsettings.exe"}


def list_audio_apps() -> list[AudioApp]:
    """Apps that own an audio session or a visible window, sound-producing ones first."""
    import os
    import psutil

    me = os.getpid()
    titles = _visible_window_titles()
    sessions = _audio_session_pids()
    apps: dict[str, AudioApp] = {}

    def add(pid: int, active: bool) -> None:
        try:
            proc = psutil.Process(pid)
            exe = proc.name()
        except psutil.Error:
            return
        if exe.lower() in _SKIP_EXES or pid == me:
            return
        root = _root_process(proc)
        key = exe.lower()
        app = apps.get(key)
        if app is None:
            app = apps[key] = AudioApp(key=exe, name=exe[:-4] if exe.lower().endswith(".exe") else exe,
                                       pid=root.pid)
        app.active = app.active or active
        if active:
            app.pid = root.pid                 # prefer the tree that is actually making noise
        title = titles.get(pid) or titles.get(root.pid)
        if title and not app.title:
            app.title = title

    for pid, active in sessions.items():
        add(pid, active)
    for pid in titles:
        add(pid, False)
    return sorted(apps.values(), key=lambda a: (not a.active, a.name.lower()))


def resolve_app_pid(key: str) -> int | None:
    """Pid of the process tree root for executable ``key`` (e.g. ``chrome.exe``), or None if not running."""
    for app in list_audio_apps():
        if app.key.lower() == key.lower():
            return app.pid
    import psutil
    for proc in psutil.process_iter(["name"]):
        if (proc.info["name"] or "").lower() == key.lower():
            return _root_process(proc).pid
    return None


def supported() -> bool:
    return sys.platform == "win32"
