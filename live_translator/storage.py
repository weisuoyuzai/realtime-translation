"""Inspecting and managing the folder that holds downloaded models.

The folder uses the Hugging Face hub cache layout (``models--<org>--<name>/{blobs,snapshots,refs}``), which is what
faster-whisper and huggingface_hub read and write when they are given ``download_root`` / ``cache_dir``.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ModelEntry:
    repo_id: str          # e.g. "Systran/faster-whisper-small"
    size: int             # bytes on disk
    path: str             # the models--org--name folder


def repo_id_from_dirname(name: str) -> str | None:
    """``models--Systran--faster-whisper-small`` -> ``Systran/faster-whisper-small``."""
    if not name.startswith("models--"):
        return None
    parts = name[len("models--"):].split("--")
    return "/".join(parts) if len(parts) > 1 else parts[0] or None


def dir_size(path: str | Path) -> int:
    """Total size of the files under ``path`` (each blob counted once; symlinks are not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def list_models(models_dir: str) -> list[ModelEntry]:
    """Models present in ``models_dir`` (empty list if it does not exist), biggest first."""
    out: list[ModelEntry] = []
    try:
        entries = list(os.scandir(models_dir))
    except OSError:
        return out
    for e in entries:
        repo = repo_id_from_dirname(e.name)
        if repo and e.is_dir():
            out.append(ModelEntry(repo, dir_size(e.path), e.path))
    return sorted(out, key=lambda m: -m.size)


def delete_model(entry: ModelEntry) -> None:
    shutil.rmtree(entry.path)


def human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit in ("B", "KB") else f"{v:.1f} {unit}"
        v /= 1024
    return f"{n} B"


def check_writable(path: str) -> str | None:
    """Create ``path`` if needed and make sure we can write to it. Returns an error message, or None if fine."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".lt_write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        return f"无法使用该目录：{e.strerror or e}"
    return None


def same_dir(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def move_models(src: str, dst: str, progress: Callable[[str], None] | None = None) -> list[str]:
    """Move every model folder from ``src`` into ``dst``. Returns the repo ids that were moved.

    A model that already exists in ``dst`` is left untouched in ``src`` (never overwritten)."""
    say = progress or (lambda _m: None)
    os.makedirs(dst, exist_ok=True)
    moved: list[str] = []
    for entry in list_models(src):
        target = os.path.join(dst, os.path.basename(entry.path))
        if os.path.exists(target):
            say(f"跳过 {entry.repo_id}（目标位置已存在）")
            continue
        say(f"正在移动 {entry.repo_id}（{human_size(entry.size)}）…")
        shutil.move(entry.path, target)      # rename on the same volume, copy + delete across volumes
        moved.append(entry.repo_id)
    return moved
