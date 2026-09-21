# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Live Translator. Use `python scripts/build.py` (or the GitHub workflow) rather than calling this directly.
#
# Environment variables read here:
#   LT_VERSION      version string written into the macOS Info.plist (default: 0.0.0)
#   LT_MAC_HELPER   path of the precompiled ScreenCaptureKit helper to embed (macOS only; built by scripts/build.py)
import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent                                    # noqa: F821  (SPECPATH is provided by PyInstaller)
NAME = "LiveTranslator"
IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"
VERSION = os.environ.get("LT_VERSION", "0.0.0")

datas, binaries, hiddenimports = [], [], []


def grab(pkg, *, data=True, libs=True, submodules=False):
    """Collect a package's data files / native libraries (skipped silently when it isn't installed)."""
    try:
        if importlib.util.find_spec(pkg) is None:
            return
    except ImportError:                         # dotted name whose parent isn't installed
        return
    if data:
        datas.extend(collect_data_files(pkg))
    if libs:
        binaries.extend(collect_dynamic_libs(pkg))
    if submodules:
        hiddenimports.extend(collect_submodules(pkg))


grab("faster_whisper", submodules=True)     # assets/silero_vad_v6.onnx is loaded by live_translator.vad; utils is imported lazily
hiddenimports += [f"faster_whisper.{m}" for m in                # also covers editable installs, which collect_submodules can't see
                  ("audio", "feature_extractor", "tokenizer", "transcribe", "utils", "vad", "version")]
grab("ctranslate2")
grab("onnxruntime")
grab("sentencepiece")
grab("av")
grab("soxr")
grab("sounddevice")                             # _sounddevice_data carries PortAudio on Windows / macOS
grab("_sounddevice_data")
grab("nvidia.cublas", data=False)               # only present with requirements-cuda.txt (Windows CUDA build)
grab("nvidia.cudnn", data=False)
hiddenimports += ["socksio"]                    # httpx[socks]: imported lazily, invisible to static analysis
if IS_WIN:
    for pkg in ("proctap", "pyaudiowpatch", "pycaw"):
        grab(pkg)
    hiddenimports += collect_submodules("comtypes") + ["proctap._native", "pyaudiowpatch"]

if IS_MAC:
    helper = os.environ.get("LT_MAC_HELPER")
    if not helper or not Path(helper).is_file():
        raise SystemExit("LT_MAC_HELPER must point to the compiled lt-sck-audio helper (run scripts/build.py)")
    binaries.append((helper, "native/macos"))

a = Analysis(                                   # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtPdf", "PySide6.QtCharts"],
    noarchive=False,
)
pyz = PYZ(a.pure)                               # noqa: F821


def make_exe(name, console):
    return EXE(pyz, a.scripts, [], exclude_binaries=True, name=name, console=console,   # noqa: F821
               upx=False, disable_windowed_traceback=False)


# Windows / macOS: no console window for the GUI. Windows additionally gets a console twin for --cli / --list-apps
# (on macOS run Contents/MacOS/LiveTranslator --cli from a terminal instead).
exes = [make_exe(NAME, console=not (IS_WIN or IS_MAC))]
if IS_WIN:
    exes.append(make_exe("live-translator-cli", console=True))

coll = COLLECT(*exes, a.binaries, a.datas, strip=False, upx=False, name=NAME)   # noqa: F821

if IS_MAC:
    app = BUNDLE(                               # noqa: F821
        coll,
        name=f"{NAME}.app",
        bundle_identifier="io.github.weisuoyuzai.live-translator",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "Live Translator",
            "CFBundleShortVersionString": VERSION,
            "LSMinimumSystemVersion": "13.0",   # ScreenCaptureKit audio capture
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Live Translator listens to the microphone to transcribe and translate speech.",
        },
    )
