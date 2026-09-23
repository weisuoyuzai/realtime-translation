---
name: run-realtime-translation
description: Build, run, and drive Live Translator (the PySide6 real-time audio translation desktop app). Use when asked to start / launch the app, take a screenshot of its UI (main window, floating subtitle bar, preferences), click through it, do a real transcription + translation run from an audio file, or run its tests.
---

Live Translator is a PySide6 desktop GUI (Windows-first). Agents drive it with
`.claude/skills/run-realtime-translation/driver.py`: it starts the real `MainWindow` in-process and takes commands
(screenshots, clicks, setting widgets, a full audio-file run through the 开始 button). All paths are relative to the repo root.
Verified on Windows 11 with Git Bash and the project `.venv` (Python 3.12, PySide6 6.11).

## Setup (once)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`run.bat` does the same on first launch. `requirements-cuda.txt` (NVIDIA libs) is optional — without it CUDA fails and the
app falls back to CPU (see Gotchas).

A speech sample for real runs (Windows built-in TTS; pick the English voice explicitly — the default here is Chinese):

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Zira Desktop")
$s.SetOutputToWaveFile("$env:TEMP\sample_en.wav")
$s.Speak("Hello everyone. Today we are going to talk about real time translation. The meeting starts at three o'clock.")
$s.Dispose()
```

## Run (agent path): the driver

One-shot: each quoted argument is one command, executed in order, then the app exits.

```bash
D=.claude/skills/run-realtime-translation/driver.py
# visual check, no window on screen (~2.5 s)
.venv/Scripts/python.exe $D --offscreen --translator none "demo" "click 学习" "ss $TEMP/lt/main.png"
# real end-to-end run: sample → Whisper tiny → fake LLM translator → UI (~15 s once the model is cached)
.venv/Scripts/python.exe $D "set panel.src_lang en" "play $TEMP/sample_en.wav 240" "dump" \
    "ss $TEMP/lt/after_play.png" "ss-overlay $TEMP/lt/overlay.png" "ss-prefs 2 $TEMP/lt/prefs.png"
```

`dump` prints what the user sees, e.g.

```
state: idle
status: [就绪] 已停止。
stats: 共 3 句 · 最近 10 句平均延迟 0.7s （从说完到译文出现）
transcript:
[20:26:10] Hello everyone.
    → 译:Hello everyone.
```

REPL mode (`-`): commands on stdin, one per line; every command answers `[driver] ok` (or `[driver] error: …`).

```bash
printf 'demo\ncard 3\nclick 学习\neval w.learn_cb.isChecked()\nset panel.tgt_lang ja\nss %s/lt/repl.png\nquit\n' "$TEMP" \
  | .venv/Scripts/python.exe $D --offscreen --translator none -
```

Commands (full list in the driver's docstring):

| command | does |
|---|---|
| `ss PATH` / `ss-overlay PATH` / `ss-prefs TAB PATH` | screenshot main window / floating subtitle bar / 偏好设置 page (0 存储 1 网络 2 悬浮字幕 3 高级) |
| `demo` | inject 4 fake subtitle lines (3 speakers, one still-recognising draft) |
| `play WAV [TIMEOUT]` | real run: the WAV becomes the audio source, presses 开始, waits for the end of the file, presses 停止 |
| `click TEXT` | first visible button whose text / tooltip / objectName contains TEXT (`开始`, `学习`, `悬浮字幕`, `偏好设置`…) |
| `set ATTR VALUE` | set a widget by attribute path on the window: `panel.tgt_lang ja`, `panel.src_kind mic`, `learn_cb 1`, `panel.topic …` |
| `card N` | open sidebar step N (0 音频来源 … 4 速度/准确度) |
| `eval EXPR` | Python with `w` = MainWindow, `app`, `pump(seconds)` |
| `wait S`, `dump`, `quit` | |

Options: `--offscreen` (headless), `--translator fake|none|keep` (default `fake`: starts `tests/fake_server.py`, translations
come back as `译:<text>`), `--asr-model tiny` (default), `--home DIR` (default: fresh temp dir — your real settings in
`%APPDATA%` are never touched), `--size 1280x800`.

**Look at the screenshots** (Read the PNG) — the driver only proves the widgets exist, not that they render.

## Run (human path)

```bash
run.bat            # or: .venv/Scripts/python.exe -m live_translator
```

Opens the window and blocks until closed. Headless CLI without the GUI (useful to isolate pipeline problems from UI ones):

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m live_translator --cli --ignore-saved --file "$TEMP/sample_en.wav" \
    --speed 2 --model tiny --device cpu --src en --translator none
```

## Test

```bash
.venv/Scripts/python.exe -m pytest -q        # 206 tests, ~60 s; includes Qt widget tests (they open real windows)
```

## Gotchas

- **Offscreen = tofu without fonts.** `QT_QPA_PLATFORM=offscreen` has no font database; every glyph renders as a box.
  The driver sets `QT_QPA_FONTDIR=%WINDIR%\Fonts` for `--offscreen`. Offscreen text is also rendered without ClearType
  and with slightly different fallback fonts — for pixel-accurate review drop `--offscreen` (window flashes on screen).
- **Piped stdin is GBK on Chinese Windows.** `click 学习` sent through a pipe arrived as `瀛︿範` until the driver
  reconfigured stdin to UTF-8. Same for the app's own CLI output in Git Bash: set `PYTHONIOENCODING=utf-8` or the
  `# loading:` lines are mojibake.
- **The GUI has no "audio file" source.** The 音频来源 card only offers 系统 / 应用 / 麦克风, and a config with
  `source=file` is loaded as 系统. `play` therefore patches `main_window.Pipeline` to inject a `FileSource`, then clicks
  the real 开始 button — everything else (validation, config save, UI updates, stop) is the real code path.
- **`FileSource.done` is a `threading.Event`, not a method**; end-of-run = `done.is_set()` and 4 s without new text (the
  CLI uses the same rule). Stopping right at `done` loses the last sentence.
- **CUDA falls back silently.** With device 自动 and no `requirements-cuda.txt`, the log says
  `CUDA failed (… cublas64_12.dll is not found …); falling back to CPU`. The status bar's `CUDA · <GPU name>` shows the
  detected GPU, not the device actually used — don't trust it for "is it on GPU".
- **First model download is slow and looks stuck.** Whisper `tiny` took ~50 s; the status sat at `0 MB … 没有数据` for
  ~45 s before completing. Models go to `%USERPROFILE%\.cache\huggingface\hub` (shared across `--home` dirs).
- **The main window is frameless** (custom title bar, `ui/titlebar.py`); the menus live in the title bar and
  `w.menuBar()` is overridden to return them. PowerShell's `MainWindowTitle` for the process comes back empty.
- The transcript repaints on a 70 ms timer — the driver calls `transcript._render()` before every screenshot; do the same
  in `eval` code before reading row widgets.

## Troubleshooting

| symptom | fix |
|---|---|
| Screenshot full of □ boxes | you ran offscreen without `QT_QPA_FONTDIR`; use the driver's `--offscreen` |
| `LookupError: no visible button matching '瀛︿範'` | stdin not UTF-8 — use the driver (it fixes this) or `PYTHONIOENCODING=utf-8` |
| `FileNotFoundError: C:\Users10226AppData…` | backslashes eaten by POSIX shlex; fixed in the driver (literal `\`). `$TEMP` in Git Bash is a Windows path with `\` |
| `play` returns with `lines=0` | model still downloading past TIMEOUT — rerun with a larger timeout, or pre-download via the CLI command above |
| `CUDA failed (识别失败：Library cublas64_12.dll …)` | harmless (CPU fallback); `pip install -r requirements-cuda.txt` for GPU |
