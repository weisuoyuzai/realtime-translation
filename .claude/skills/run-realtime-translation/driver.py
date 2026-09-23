"""Drive the Live Translator GUI programmatically: launch the real MainWindow in-process, run commands, take screenshots.

Usage (from the repo root, with the project venv):
    .venv/Scripts/python.exe .claude/skills/run-realtime-translation/driver.py [options] CMD [CMD ...]
    .venv/Scripts/python.exe .claude/skills/run-realtime-translation/driver.py [options] -      # REPL on stdin

Each CMD is one quoted string, e.g. "ss out/main.png". Commands:
    ss PATH                  screenshot of the main window
    ss-overlay PATH          screenshot of the floating subtitle bar (shown if hidden)
    ss-prefs TAB PATH        open 偏好设置 on TAB (0 存储, 1 网络, 2 悬浮字幕, 3 高级), screenshot, close
    demo                     inject 4 fake subtitle lines (3 speakers, one draft) for visual checks
    card N                   open step card N of the sidebar (0 音频来源 … 4 速度/准确度)
    set ATTR VALUE           set a widget on the window, e.g. "set panel.tgt_lang ja", "set learn_cb 1",
                             "set panel.topic Kubernetes"; combos/segmented match by data, then by text
    click TEXT               click the first visible button whose text / tooltip / objectName contains TEXT
    play WAV [TIMEOUT]       real run: feed WAV into the pipeline as live audio, press 开始, wait until the
                             file is done (or TIMEOUT s, default 180), press 停止
    wait SECONDS             keep the event loop running
    dump                     print state, status line, stats and the transcript text
    eval EXPR                eval Python with `w` (window) and `app` in scope; prints the result
    quit                     close the window and exit (also happens after the last CMD)

Options:
    --offscreen              QT_QPA_PLATFORM=offscreen: no window on screen (screenshots still work)
    --home DIR               config/transcript folder (default: a fresh temp dir, so your settings are untouched)
    --asr-model NAME         local Whisper model for `play` (default: tiny; downloads on first use)
    --translator fake|none|keep
                             fake (default): start tests/fake_server.py and translate through it ("译:<text>");
                             none: transcription only; keep: whatever the (home) config says
    --size WxH               main window size (default 1280x800)
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repo root: .claude/skills/<name>/driver.py
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))             # fake_server.py
for s in (sys.stdin, sys.stdout, sys.stderr):     # Windows pipes default to the ANSI code page (GBK here)
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Translator GUI driver", add_help=True)
    p.add_argument("--offscreen", action="store_true")
    p.add_argument("--home")
    p.add_argument("--asr-model", default="tiny")
    p.add_argument("--translator", choices=["fake", "none", "keep"], default="fake")
    p.add_argument("--size", default="1280x800")
    p.add_argument("cmds", nargs="*")
    return p.parse_args()


A = parse()
if A.offscreen:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # the offscreen platform has no font database: without this every glyph renders as a box ("tofu")
    fontdir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" if sys.platform == "win32" else None
    if fontdir and fontdir.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fontdir))
os.environ["LIVE_TRANSLATOR_HOME"] = A.home or tempfile.mkdtemp(prefix="lt_driver_")

from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox  # noqa: E402

app = QApplication(sys.argv[:1])
from live_translator.config import load_config  # noqa: E402
from live_translator.models import Line  # noqa: E402
from live_translator.ui import main_window as mw  # noqa: E402
from live_translator.ui import preferences as prefs  # noqa: E402
from live_translator.ui.widgets import Segmented  # noqa: E402

print(f"[driver] home = {os.environ['LIVE_TRANSLATOR_HOME']}", flush=True)
cfg = load_config()
cfg.asr.mode, cfg.asr.model, cfg.asr.device = "local", A.asr_model, cfg.asr.device
fake = None
if A.translator == "fake":
    from fake_server import FakeOpenAI
    fake = FakeOpenAI().start()
    cfg.translate.mode = "llm_remote"
    cfg.translate.remote.base_url, cfg.translate.remote.api_key, cfg.translate.remote.model = fake.base_url, "sk-fake", "m-a"
elif A.translator == "none":
    cfg.translate.mode = "none"

w = mw.MainWindow(cfg)
width, height = (int(x) for x in A.size.lower().split("x"))
w.resize(width, height)
w.show()


def pump(seconds: float = 0.0) -> None:
    end = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def settle() -> None:
    w.transcript._render()               # the transcript repaints on a 70 ms timer; force it before a screenshot
    pump(0.25)


def save(widget, path: str) -> None:
    settle()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = widget.grab().save(path)
    print(f"[driver] {'saved' if ok else 'FAILED to save'} {path}", flush=True)


def resolve(attr: str):
    obj = w
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def set_value(obj, value: str) -> None:
    if isinstance(obj, (QComboBox, Segmented)):
        i = obj.findData(value)
        if i < 0:
            i = next((k for k in range(obj.count()) if value in obj.itemText(k)), -1)
        if i < 0 and isinstance(obj, QComboBox) and obj.isEditable():
            obj.setEditText(value)
        elif i < 0:
            raise ValueError(f"no item {value!r}; items: {[obj.itemText(k) for k in range(obj.count())]}")
        else:
            obj.setCurrentIndex(i)
    elif isinstance(obj, QAbstractButton) and obj.isCheckable():
        obj.setChecked(value.lower() in ("1", "true", "on", "yes"))
    elif isinstance(obj, (QSpinBox, QDoubleSpinBox)):
        obj.setValue(type(obj.value())(value))
    elif isinstance(obj, QLineEdit):
        obj.setText(value)
    elif hasattr(obj, "setPlainText"):
        obj.setPlainText(value.replace("\\n", "\n"))
    else:
        raise TypeError(f"don't know how to set {type(obj).__name__}")


def find_button(text: str) -> QAbstractButton:
    for b in w.findChildren(QAbstractButton):
        if b.isVisible() and (text in b.text() or text in b.toolTip() or text == b.objectName()):
            return b
    raise LookupError(f"no visible button matching {text!r}")


def demo() -> None:
    now = time.time()
    rows = [
        ("Welcome back to the platform engineering series.", "欢迎回到平台工程系列。", 1, True),
        ("Kubernetes operators encode operational knowledge into software.", "Kubernetes Operator 把运维知识编码进软件里。", 1, True),
        ("Right, and that's why reconciliation loops matter.", "没错，这也是调谐循环如此重要的原因。", 2, True),
        ("First, we define a custom resource that", "首先，我们定义一个自定义资源，它", 3, False),
    ]
    for i, (src, dst, spk, final) in enumerate(rows, 1):
        w._on_line(Line(i, src=src, dst=dst, src_final=final, dst_final=final, dst_draft=not final, src_lang="en",
                        dst_lang="zh-Hans", speaker=spk, latency_ms=850 if final else 0, asr_ms=300, tr_ms=500,
                        created=now - 40 + i * 10))


def play(wav: str, timeout: float = 180) -> None:
    """Real end-to-end run through the UI: the start button builds the pipeline; we only swap the audio source."""
    from live_translator.audio.file import FileSource
    wav = str(Path(wav).resolve())
    if not os.path.isfile(wav):
        raise FileNotFoundError(wav)
    src = FileSource(wav, speed=1.0)
    real = mw.Pipeline
    mw.Pipeline = lambda c, *a, **k: real(c, *a, source=src, **k)
    try:
        find_button("开始").click()
        t0 = time.monotonic()
        quiet_since, last = t0, ""
        # same rule as the CLI: once the file has been played, stop after 4 s without new subtitle text
        while w.state in ("starting", "running") and time.monotonic() - t0 < timeout:
            pump(0.2)
            cur = w.transcript.plain_text()
            if cur != last:
                quiet_since, last = time.monotonic(), cur
            if src.done.is_set() and time.monotonic() - quiet_since > 4:
                break
        if w.state in ("starting", "running"):
            w.btn.click()
        t1 = time.monotonic()
        while w.state != "idle" and time.monotonic() - t1 < 20:
            pump(0.1)
    finally:
        mw.Pipeline = real
    print(f"[driver] play finished in {time.monotonic() - t0:.1f}s, state={w.state}, lines={w.transcript.count()}", flush=True)


def dump() -> None:
    settle()
    print(f"state: {w.state}\nstatus: [{w.pill_text.text()}] {w.status.text()}\nstats: {w.stats.text()}")
    print("transcript:\n" + (w.transcript.plain_text() or "(empty)"), flush=True)


def run(line: str) -> bool:
    lex = shlex.shlex(line, posix=True)
    lex.whitespace_split, lex.escape = True, ""         # quotes group words; backslashes stay (Windows paths)
    args = list(lex)
    if not args:
        return True
    cmd, rest = args[0], args[1:]
    if cmd == "quit":
        return False
    if cmd == "ss":
        save(w, rest[0])
    elif cmd == "ss-overlay":
        if not w.overlay.isVisible():
            w.overlay_cb.setChecked(True)
        for l in list(w.transcript._lines.values())[-3:]:
            w.overlay.show_line(l)
        save(w.overlay, rest[0])
    elif cmd == "ss-prefs":
        d = prefs.PreferencesDialog(w.current_config(), running=w.state != "idle", parent=w, tab=int(rest[0]))
        d.show()
        save(d, rest[1])
        d.close()
    elif cmd == "demo":
        demo()
    elif cmd == "card":
        w.panel.reveal(int(rest[0]))
    elif cmd == "set":
        set_value(resolve(rest[0]), " ".join(rest[1:]))
    elif cmd == "click":
        find_button(" ".join(rest)).click()
    elif cmd == "play":
        play(rest[0], float(rest[1]) if len(rest) > 1 else 180)
    elif cmd == "wait":
        pump(float(rest[0]))
    elif cmd == "dump":
        dump()
    elif cmd == "eval":
        print(repr(eval(" ".join(rest), {"w": w, "app": app, "pump": pump})), flush=True)
    else:
        print(f"[driver] unknown command {cmd!r}", flush=True)
    pump(0.05)
    return True


def main() -> int:
    pump(0.3)
    cmds = A.cmds
    code = 0
    if cmds == ["-"]:                        # REPL: a reader thread feeds lines; the main thread keeps Qt alive
        q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=lambda: ([q.put(l) for l in sys.stdin], q.put(None)), daemon=True).start()
        print("[driver] ready (REPL)", flush=True)
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                pump(0.05)
                continue
            if line is None:
                break
            try:
                if not run(line.strip()):
                    break
            except Exception as e:
                print(f"[driver] error: {type(e).__name__}: {e}", flush=True)
            print("[driver] ok", flush=True)
    else:
        for c in cmds:
            try:
                if not run(c):
                    break
            except Exception as e:
                print(f"[driver] error in {c!r}: {type(e).__name__}: {e}", flush=True)
                code = 1
                break
    w.close()
    pump(0.2)
    if fake is not None:
        fake._httpd.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
