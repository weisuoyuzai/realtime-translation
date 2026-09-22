"""Process-wide environment setup and hardware probing."""
from __future__ import annotations

import glob
import logging
import os
import sys

log = logging.getLogger(__name__)

_cuda_paths_done = False
_models_dir: str | None = None
_proxy_vars_we_set: dict[str, str | None] = {}      # env var -> value it had before we touched it
# Windows environment variable names are case-insensitive: HTTPS_PROXY and https_proxy are ONE variable there, so
# handling both spellings would record our own value as the "original" and break restoring the user's setting.
_PROXY_VARS = (("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if sys.platform == "win32"
               else ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"))


def setup_environment(hf_endpoint: str = "") -> None:
    """Call once at startup, before any model is loaded."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # hf-xet (installed by default with huggingface_hub) chunks large files across many small parallel requests and
    # is prone to stalling outright on multi-GB files like large-v3's model.bin - with or without a mirror. A plain
    # ranged HTTP download is slower per-connection but does not hang, so it wins here unconditionally.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
    _add_cuda_dll_dirs()


def apply_settings(cfg) -> None:
    """Apply the process-wide parts of the settings (models dir, proxy, mirror). Safe to call repeatedly.

    Note: ``huggingface_hub`` reads ``HF_ENDPOINT`` and its HTTP session when it is first imported/used, so a changed
    mirror or proxy is only guaranteed to affect model downloads after a restart. API calls made through our own
    httpx clients pick a new proxy up immediately."""
    setup_environment(cfg.hf_endpoint)
    set_models_dir(cfg.models_dir)
    _apply_proxy(cfg.proxy.strip())


def _apply_proxy(proxy: str) -> None:
    for var in _PROXY_VARS:
        if proxy:
            _proxy_vars_we_set.setdefault(var, os.environ.get(var))
            os.environ[var] = proxy
        elif var in _proxy_vars_we_set:                 # we set it earlier and the user cleared it → restore
            old = _proxy_vars_we_set.pop(var)
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
    if proxy:                                           # never send local LLM / loopback traffic through a proxy
        keep = [h for h in os.environ.get("NO_PROXY", "").split(",") if h]
        os.environ["NO_PROXY"] = ",".join(dict.fromkeys(keep + ["127.0.0.1", "localhost", "::1"]))


def default_models_dir() -> str:
    """Where Hugging Face keeps models when nothing is configured (honours the user's own HF_* variables)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def set_models_dir(path: str) -> None:
    global _models_dir
    _models_dir = path.strip() or None


def models_dir() -> str | None:
    """Explicitly configured models directory, or None to let the libraries use their default."""
    return _models_dir


def effective_models_dir() -> str:
    return _models_dir or default_models_dir()


def _add_cuda_dll_dirs() -> None:
    """CTranslate2 needs cuBLAS/cuDNN. When they come from the ``nvidia-*-cu12`` pip wheels
    (instead of a system CUDA install) their DLL folders must be made discoverable on Windows."""
    global _cuda_paths_done
    if _cuda_paths_done or sys.platform != "win32":
        return
    _cuda_paths_done = True
    roots = {p for p in sys.path if p.endswith("site-packages")}
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):      # packaged build: wheels are unpacked next to the exe
        roots.add(sys._MEIPASS)
    for sp in roots:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def gpu_name() -> str:
    """Name of GPU 0 as reported by nvidia-smi ("" when there is none)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return ""


def gpu_memory_gb() -> float:
    """Total VRAM of GPU 0 in GiB (0 when unknown)."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def resolve_asr_device(device: str = "auto", compute_type: str = "auto") -> tuple[str, str]:
    """Pick (device, compute_type) for faster-whisper."""
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def recommended_asr_model(device: str) -> str:
    """Sensible default per hardware: turbo on a GPU with room to spare, ``small`` otherwise."""
    if device == "cuda":
        return "large-v3-turbo" if gpu_memory_gb() >= 5.5 else "small"
    return "small"
