"""Progress reporting for model downloads.

``WhisperModel(...)`` and ``snapshot_download(...)`` block for minutes on a multi-GB model and say nothing while they
do, so the window looks frozen. The download itself is left alone; a helper thread watches the repo's ``blobs``
folder (finished + ``*.incomplete`` files) and reports size, percentage and speed. That works the same for the plain
HTTP and the hf-xet download paths and does not depend on huggingface_hub's progress-bar internals."""
from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

from .runtime import effective_models_dir

log = logging.getLogger(__name__)

_STALL_HINT_S = 20


def _fmt_bytes(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


def blobs_dir(repo_id: str) -> str:
    return os.path.join(effective_models_dir(), "models--" + repo_id.replace("/", "--"), "blobs")


def _bytes_on_disk(path: str) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    total += e.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _expected_bytes(repo_id: str, patterns: Sequence[str]) -> int:
    """Size of the files that will be fetched, from the hub's file listing (0 when it cannot be had)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True, timeout=8)
        return sum(s.size or 0 for s in info.siblings or ()
                   if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))
    except Exception:
        log.debug("could not look up the size of %s", repo_id, exc_info=True)
        return 0


@contextmanager
def report_download(repo_id: str, label: str, say: Callable[[str], None], patterns: Sequence[str] = ("*",),
                    interval: float = 1.0) -> Iterator[None]:
    """While the body runs, call ``say`` about once a second with how much of ``repo_id`` has arrived."""
    stop = threading.Event()
    # Looking up the expected total size is its own network round-trip (up to 8s, and subject to the same flaky
    # connection as the download itself). Doing it before the disk-polling loop starts would delay the first real
    # progress update by that much - long enough for small/fast files to finish downloading before a single byte
    # count is ever shown. So it runs on the side and the loop below just uses whatever it has so far (0 = unknown).
    total_box = [0]

    def fetch_total() -> None:
        total_box[0] = _expected_bytes(repo_id, patterns)

    def watch() -> None:
        say(f"正在连接下载源，准备下载 {label}…")
        folder = blobs_dir(repo_id)
        start_bytes = last_bytes = _bytes_on_disk(folder)
        last_t = last_move = time.monotonic()
        speed = 0.0
        while not stop.wait(interval):
            now = time.monotonic()
            done = _bytes_on_disk(folder)
            if done != last_bytes:
                last_move = now
            speed = 0.6 * speed + 0.4 * max(0, done - last_bytes) / max(now - last_t, 1e-3)
            last_bytes, last_t = done, now
            total = total_box[0]
            head = f"正在下载 {label}：{_fmt_bytes(done)}"
            if total:
                head += f" / {_fmt_bytes(total)}（{min(100, done * 100 // total)}%）"
            tail = f" · {_fmt_bytes(speed)}/s" if speed >= 1024 else ""
            if now - last_move > _STALL_HINT_S:
                tail = f" · 已 {int(now - last_move)} 秒没有新数据，网络可能卡住了（可在设置里换镜像 / 设置代理）"
            elif done == start_bytes and speed < 1024:
                tail = " · 等待服务器响应…"
            say(head + tail)

    threading.Thread(target=fetch_total, name="download-size-lookup", daemon=True).start()
    t = threading.Thread(target=watch, name="download-progress", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
