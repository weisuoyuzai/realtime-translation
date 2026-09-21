"""PyInstaller entry point (the package's own ``__main__`` uses a relative import, so it cannot be frozen directly).

Extras over ``python -m live_translator``:
  * ``--self-test``   import every native dependency and run the VAD once; used by CI to smoke-test the frozen app.
  * windowed builds have no stdout/stderr (they are ``None``): send them to a log file so libraries that print or
    show progress bars (huggingface_hub, tqdm) don't crash, and so users have something to attach to a bug report.
"""
import multiprocessing
import os
import sys


def _redirect_missing_streams() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        from live_translator.config import config_dir
        stream = open(config_dir() / "live-translator.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def _self_test() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np

    from live_translator import runtime
    runtime.setup_environment()
    checks = []

    def check(name, fn):
        try:
            checks.append((name, True, str(fn() or "")))
        except Exception as e:                                  # noqa: BLE001 - report every failure, not just the first
            checks.append((name, False, f"{type(e).__name__}: {e}"))

    def vad():
        from live_translator.vad import SileroVAD
        p = SileroVAD()(np.zeros(512, dtype=np.float32))
        assert 0.0 <= p <= 1.0
        return f"p(silence)={p:.3f}"

    def qt():
        from PySide6.QtWidgets import QApplication

        from live_translator.ui.main_window import MainWindow  # noqa: F401
        QApplication.instance() or QApplication([])

    def ctranslate2():
        import ctranslate2 as ct
        return f"ctranslate2 {ct.__version__}, cuda devices: {ct.get_cuda_device_count()}"

    def whisper():
        import faster_whisper
        from faster_whisper.utils import get_assets_path
        return f"faster-whisper {faster_whisper.__version__}, assets: {os.listdir(get_assets_path())}"

    def audio():
        # Import the platform backend and touch its native part; don't enumerate apps (CI runners have no audio session).
        if sys.platform == "win32":
            import proctap._native, pyaudiowpatch, pycaw.pycaw  # noqa: F401,E401
            from live_translator.audio import windows  # noqa: F401
            return "WASAPI backends import"
        if sys.platform == "darwin":
            import json
            import subprocess

            from live_translator.audio import macos
            if getattr(sys, "frozen", False):
                assert macos.bundled_helper(), "Swift helper missing from the bundle"
            out = subprocess.run([str(macos.ensure_helper()), "--check"], capture_output=True, text=True, timeout=30)
            return f"helper: {json.loads(out.stdout.strip().splitlines()[-1])}"
        return "no per-app capture on this OS (microphone / file only)"

    def misc():
        import av, httpx, huggingface_hub, psutil, sentencepiece, soxr  # noqa: F401,E401

    def portaudio():
        try:
            import sounddevice
        except OSError as e:
            if sys.platform.startswith("linux"):                # a system package there, not part of the bundle
                return f"WARNING: {e} (install libportaudio2 for microphone support)"
            raise
        return f"PortAudio {sounddevice.get_portaudio_version()[1]}"

    check("ctranslate2", ctranslate2)
    check("faster-whisper + silero asset", whisper)
    check("silero VAD (onnxruntime)", vad)
    check("Qt + main window import", qt)
    check("audio backends", audio)
    check("PortAudio (sounddevice)", portaudio)
    check("other libraries", misc)
    for name, ok, info in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name} {info}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _redirect_missing_streams()
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    from live_translator.cli import main
    sys.exit(main())
