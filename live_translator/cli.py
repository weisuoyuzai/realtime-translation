"""Command-line entry: headless mode (``--cli``) and helpers such as ``--list-apps``."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from .config import AppConfig, apply_preset, load_config
from .models import Line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="live_translator", description="实时音频翻译（默认打开图形界面）")
    p.add_argument("--cli", action="store_true", help="无界面运行，字幕打印到终端")
    p.add_argument("--list-apps", action="store_true", help="列出可捕获的应用后退出")
    p.add_argument("--ignore-saved", action="store_true", help="不读取已保存的设置")
    p.add_argument("-v", "--verbose", action="store_true")

    a = p.add_argument_group("音频来源")
    a.add_argument("--source", choices=["system", "app", "mic", "file"])
    a.add_argument("--app", help="要捕获的应用（Windows: 进程名如 chrome.exe；macOS: bundle id）")
    a.add_argument("--file", help="用音频文件模拟实时输入")
    a.add_argument("--speed", type=float, default=1.0, help="文件回放倍速（默认 1.0 = 实时）")

    r = p.add_argument_group("语音识别")
    r.add_argument("--asr", choices=["local", "remote"])
    r.add_argument("--model", help="本地 Whisper 模型，如 large-v3-turbo / small / 本地路径")
    r.add_argument("--device", choices=["auto", "cuda", "cpu"])
    r.add_argument("--asr-url")
    r.add_argument("--asr-key")
    r.add_argument("--asr-model")

    l = p.add_argument_group("语言与翻译")
    l.add_argument("--src", help="源语言代码，auto = 自动检测")
    l.add_argument("--tgt", help="目标语言，如 zh-Hans zh-Hant en ja ko")
    l.add_argument("--translator", choices=["llm_remote", "llm_local", "nllb", "deepl", "none"])
    l.add_argument("--tr-url")
    l.add_argument("--tr-key")
    l.add_argument("--tr-model")
    l.add_argument("--glossary", help="术语表文件路径（每行 term = 译法）")
    l.add_argument("--topic")
    l.add_argument("--preset", choices=["fast", "balanced", "accurate"])
    l.add_argument("--draft", action="store_true", help="说话过程中就翻译草稿")
    l.add_argument("--partials", action="store_true", help="打印中间结果")
    return p


def apply_args(cfg: AppConfig, a: argparse.Namespace) -> AppConfig:
    if a.preset:
        apply_preset(cfg, a.preset)
    if a.source:
        cfg.audio.source = a.source
    if a.app:
        cfg.audio.source, cfg.audio.app_key, cfg.audio.app_name = "app", a.app, a.app
    if a.file:
        cfg.audio.source, cfg.audio.file_path = "file", a.file
    if a.asr:
        cfg.asr.mode = a.asr
    if a.model:
        cfg.asr.model = a.model
    if a.device:
        cfg.asr.device = a.device
    if a.asr_url:
        cfg.asr.remote_base_url = a.asr_url
    if a.asr_key:
        cfg.asr.remote_api_key = a.asr_key
    if a.asr_model:
        cfg.asr.remote_model = a.asr_model
    if a.src:
        cfg.lang.source = a.src
    if a.tgt:
        cfg.lang.target = a.tgt
    if a.translator:
        cfg.translate.mode = a.translator
    ep = {"llm_remote": cfg.translate.remote, "llm_local": cfg.translate.local}.get(cfg.translate.mode)
    if ep is not None:
        ep.base_url = a.tr_url or ep.base_url
        ep.api_key = a.tr_key or ep.api_key
        ep.model = a.tr_model or ep.model
    if a.glossary:
        cfg.translate.glossary = open(a.glossary, encoding="utf-8").read()
    if a.topic:
        cfg.translate.topic = a.topic
    if a.draft:
        cfg.translate.draft = True
    cfg.save_transcript = False
    return cfg


def run_cli(cfg: AppConfig, a: argparse.Namespace) -> int:
    from .audio.file import FileSource
    from .pipeline import Pipeline

    for stream in (sys.stdout, sys.stderr):         # legacy Windows code pages can't encode every language
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    seen: dict[int, tuple[str, str]] = {}
    finished: set[int] = set()
    last_activity = [time.monotonic()]
    failed = threading.Event()
    lock = threading.Lock()

    def on_line(line: Line) -> None:
        last_activity[0] = time.monotonic()
        with lock:
            if line.removed or line.id in finished:
                return
            done = line.dst_final                     # also True for pass-through / transcription-only lines
            if not done and not (a.partials and seen.get(line.id) != (line.src, line.dst)):
                return
            seen[line.id] = (line.src, line.dst)
            if done:
                finished.add(line.id)
            tag = "" if done else "…"
            timing = (f"  (asr {line.asr_ms:.0f}ms, translate {line.tr_ms:.0f}ms, "
                      f"end-to-end {line.latency_ms:.0f}ms)") if done and line.latency_ms else ""
            print(f"{tag}[{line.src_lang or '?'}] {line.src}")
            if line.dst:
                print(f"      → {line.dst}{timing}")
            elif timing:
                print(f"     {timing}")
            sys.stdout.flush()

    def on_status(level: str, msg: str) -> None:
        if msg:
            print(f"# {level}: {msg}", file=sys.stderr, flush=True)
        if level in ("error", "stopped"):
            failed.set()

    source = None
    if cfg.audio.source == "file":
        source = FileSource(cfg.audio.file_path, speed=a.speed)
    pipe = Pipeline(cfg, on_line, on_status, source=source)
    pipe.start()
    try:
        while not failed.is_set():
            time.sleep(0.2)
            if isinstance(source, FileSource) and source.done.is_set():
                # let in-flight ASR/translation drain, then leave
                if time.monotonic() - last_activity[0] > 4.0:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
    return 1 if failed.is_set() and not (isinstance(source, FileSource) and source.done.is_set()) else 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = AppConfig() if a.ignore_saved else load_config()

    from .runtime import apply_settings
    apply_settings(cfg)

    if a.list_apps:
        from .audio import list_audio_apps
        for app in list_audio_apps():
            print(f"{'*' if app.active else ' '} {app.key:32s} pid={app.pid:<7d} {app.title[:60]}")
        return 0
    if a.cli:
        return run_cli(apply_args(cfg, a), a)

    from .ui.app import run_gui
    return run_gui(cfg)
