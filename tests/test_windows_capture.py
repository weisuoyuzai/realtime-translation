"""Real WASAPI process-loopback tests (Windows 10 2004+ only). Silent: they capture this very process, which
plays nothing but still yields a continuous stream of (silent) buffers."""
import os
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def _count_callbacks(seconds: float) -> int:
    from live_translator.audio.windows import ProcessLoopbackSource

    n = [0]
    src = ProcessLoopbackSource(os.getpid(), "self")
    err = []

    def start_in_short_lived_thread():        # mirrors Pipeline._boot: the thread ends right after start()
        try:
            src.start(lambda x, sr: n.__setitem__(0, n[0] + 1))
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=start_in_short_lived_thread)
    t.start()
    t.join()
    if err:
        pytest.skip(f"process loopback unavailable here: {err[0]}")
    time.sleep(seconds)
    src.stop()
    return n[0]


def test_capture_keeps_delivering_after_the_starting_thread_exits():
    """Regression: capture created on a thread that then exits went silently dead (0 callbacks)."""
    assert _count_callbacks(1.5) > 50


def test_stop_is_clean_and_source_can_restart():
    assert _count_callbacks(0.5) > 10
    assert _count_callbacks(0.5) > 10


def test_app_discovery_lists_this_python_process_tree_or_windows():
    from live_translator.audio.windows import list_audio_apps, resolve_app_pid

    apps = list_audio_apps()
    assert apps and all(a.key and a.pid for a in apps)
    assert [a.active for a in apps] == sorted((a.active for a in apps), reverse=True)     # active ones first
    assert resolve_app_pid(apps[0].key) == apps[0].pid
    assert resolve_app_pid("definitely-not-running.exe") is None


def test_decode_pcm_handles_float32_int16_and_torn_buffers():
    import numpy as np
    from live_translator.audio.windows import decode_pcm

    stereo = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
    out = decode_pcm(stereo.tobytes(), 32, 2)
    assert out.dtype == np.float32 and np.allclose(out, [0.0, 0.5])              # channels averaged
    i16 = np.array([16384, 16384, -32768, 0], dtype="<i2")                        # frames: (0.5, 0.5), (-1.0, 0.0)
    assert np.allclose(decode_pcm(i16.tobytes(), 16, 2), [0.5, -0.5])
    assert np.allclose(decode_pcm(np.array([0.25, 0.5], dtype=np.float32).tobytes(), 32, 1), [0.25, 0.5])
    torn = np.array([0.5, 0.5, 0.5], dtype=np.float32).tobytes()                 # 3 samples for 2 channels
    assert len(decode_pcm(torn, 32, 2)) == 1
    with pytest.raises(ValueError):
        decode_pcm(b"\x00" * 12, 24, 2)


def test_capture_start_does_not_import_scipy():
    """Regression: proctap's Python layer pulled in scipy (10+ s cold) on the first capture."""
    import subprocess
    code = ("import sys, os; from live_translator.audio.windows import ProcessLoopbackSource as P\n"
            "p = P(os.getpid()); p.start(lambda x, sr: None); p.stop()\n"
            "print('scipy' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=90)
    assert r.stdout.strip().endswith("False"), r.stdout + r.stderr
