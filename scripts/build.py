#!/usr/bin/env python3
"""Build a standalone Live Translator app for the current OS (Windows / macOS / Linux) with PyInstaller.

    python scripts/build.py                  # fresh .venv-build, install deps, build, smoke-test, archive
    python scripts/build.py --current-env    # reuse the running interpreter's environment (faster for iteration)
    python scripts/build.py --cuda           # Windows: also bundle cuBLAS/cuDNN so NVIDIA GPUs work out of the box (~1 GB more)

Output (in ./dist):
    Windows  LiveTranslator/  + LiveTranslator-<ver>-windows-x64.zip       (LiveTranslator.exe, live-translator-cli.exe)
    macOS    LiveTranslator.app + LiveTranslator-<ver>-macos-arm64.dmg     (Swift capture helper precompiled inside)
    Linux    LiveTranslator/  + LiveTranslator-<ver>-linux-x64.tar.gz

PyInstaller cannot cross-compile: run this on each OS you want a build for (the GitHub workflow does exactly that).
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST, WORK = ROOT / "dist", ROOT / "build"
NAME = "LiveTranslator"
IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"
OS_NAME = "windows" if IS_WIN else "macos" if IS_MAC else "linux"
ARCH = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(),
                                                                                    platform.machine().lower())


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def project_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "0.0.0"


def prepare_python(current_env: bool, cuda: bool) -> Path:
    """Return the interpreter to build with, creating .venv-build and installing dependencies unless told not to."""
    if current_env:
        py = Path(sys.executable)
    else:
        venv = ROOT / ".venv-build"
        py = venv / ("Scripts/python.exe" if IS_WIN else "bin/python")
        if not py.is_file():
            run([sys.executable, "-m", "venv", venv])
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    reqs = [ROOT / "requirements-build.txt"] + ([ROOT / "requirements-cuda.txt"] if cuda else [])
    run([py, "-m", "pip", "install", *[a for r in reqs for a in ("-r", r)]])
    return py


def compile_mac_helper() -> Path:
    """Precompile the ScreenCaptureKit helper so end users don't need Xcode command line tools."""
    swiftc = shutil.which("swiftc")
    if not swiftc:
        sys.exit("swiftc not found - install the Xcode command line tools: xcode-select --install")
    out = WORK / "lt-sck-audio"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pin the deployment target: by default swiftc targets the build machine's macOS, which would break older Macs.
    run([swiftc, "-O", "-target", f"{platform.machine()}-apple-macos13.0", "-o", out,
         ROOT / "live_translator/native/macos/lt_sck_capture.swift",
         "-framework", "ScreenCaptureKit", "-framework", "AVFoundation",
         "-framework", "CoreMedia", "-framework", "CoreGraphics"])
    return out


def app_executable(cli: bool = False) -> Path:
    if IS_MAC:
        return DIST / f"{NAME}.app/Contents/MacOS/{NAME}"
    if IS_WIN:
        return DIST / NAME / ("live-translator-cli.exe" if cli else f"{NAME}.exe")
    return DIST / NAME / NAME


def smoke_test() -> None:
    """Run the frozen app's --self-test (imports every native dependency, runs the VAD once)."""
    with tempfile.TemporaryDirectory(prefix="lt_selftest_") as home:
        env = dict(os.environ, LIVE_TRANSLATOR_HOME=home, QT_QPA_PLATFORM="offscreen", HF_HUB_CACHE=home)
        run([app_executable(cli=True), "--self-test"], env=env, timeout=300)


def make_archive(version: str, suffix: str) -> Path:
    base = f"{NAME}-{version}-{OS_NAME}-{ARCH}{suffix}"
    if IS_WIN:
        out = DIST / f"{base}.zip"
        out.unlink(missing_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for f in sorted((DIST / NAME).rglob("*")):
                if f.is_file():
                    z.write(f, Path(NAME) / f.relative_to(DIST / NAME))
    elif IS_MAC:
        out = DIST / f"{base}.dmg"
        out.unlink(missing_ok=True)
        run(["hdiutil", "create", "-volname", "Live Translator", "-srcfolder", DIST / f"{NAME}.app",
             "-ov", "-format", "UDZO", out])
    else:
        out = DIST / f"{base}.tar.gz"
        with tarfile.open(out, "w:gz", compresslevel=6) as t:                # tar keeps the executable bits, zip doesn't
            t.add(DIST / NAME, arcname=NAME)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current-env", action="store_true",
                    help="build with the running interpreter instead of a fresh .venv-build (dependencies must be installed)")
    ap.add_argument("--cuda", action="store_true", help="Windows only: bundle the NVIDIA cuBLAS/cuDNN wheels")
    ap.add_argument("--version", default=os.environ.get("LT_VERSION") or project_version(),
                    help="version used in the archive name (default: pyproject.toml; a leading 'v' is stripped)")
    ap.add_argument("--no-test", action="store_true", help="skip the smoke test of the built app")
    ap.add_argument("--no-archive", action="store_true", help="only build dist/, don't create the zip / dmg / tar.gz")
    ap.add_argument("--keep-build", action="store_true", help="don't wipe build/ and dist/ first")
    a = ap.parse_args()
    version = a.version.lstrip("v")
    if a.cuda and not IS_WIN:
        sys.exit("--cuda is only supported for Windows builds (Linux users with a system CUDA 12 + cuDNN 9 install "
                 "get GPU support automatically).")

    if not a.keep_build:
        for d in (DIST, WORK):
            shutil.rmtree(d, ignore_errors=True)
    py = prepare_python(a.current_env, a.cuda)

    numeric = re.match(r"\d+(\.\d+){0,2}", version)                # Info.plist versions must be numeric ("0.1.0-dev.abc" isn't)
    env = dict(os.environ, LT_VERSION=numeric.group(0) if numeric else "0.0.0")
    if IS_MAC:
        env["LT_MAC_HELPER"] = str(compile_mac_helper())
    run([py, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", DIST, "--workpath", WORK,
         ROOT / "packaging/live_translator.spec"], env=env, cwd=ROOT)

    if not a.no_test:
        smoke_test()
    if a.no_archive:
        print(f"\nBuilt: {app_executable()}")
    else:
        out = make_archive(version, "-cuda" if a.cuda else "")
        print(f"\nBuilt: {app_executable()}\nArchive: {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
