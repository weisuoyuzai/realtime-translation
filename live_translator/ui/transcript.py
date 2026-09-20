"""Scrolling subtitle history: original text (small) and translation (large) per utterance."""
from __future__ import annotations

import html
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QTextBrowser

from ..languages import lang_native
from ..models import Line

_SHOWN = 80          # lines rendered; older ones stay in memory (and in the saved transcript file)


class TranscriptView(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setPlaceholderText("点击「开始」后，字幕会显示在这里。")
        self._lines: dict[int, Line] = {}
        self._order: list[int] = []
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(70)                 # coalesce bursts of streaming updates into ≤ 14 repaints/s
        self._timer.timeout.connect(self._render)
        self._timer.start()

    # ── model ────────────────────────────────────────────────────────────────

    def upsert(self, line: Line) -> None:
        if line.removed:
            if line.id in self._lines:
                del self._lines[line.id]
                self._order.remove(line.id)
                self._dirty = True
            return
        if line.id not in self._lines:
            self._order.append(line.id)
        self._lines[line.id] = line
        self._dirty = True

    def clear_all(self) -> None:
        self._lines.clear()
        self._order.clear()
        self.clear()

    def plain_text(self) -> str:
        out = []
        for i in self._order:
            l = self._lines[i]
            out.append(f"[{time.strftime('%H:%M:%S', time.localtime(l.created))}] {l.src}")
            if l.dst:
                out.append(f"    → {l.dst}")
        return "\n".join(out)

    # ── rendering ────────────────────────────────────────────────────────────

    def _colors(self) -> dict[str, str]:
        pal = self.palette()
        text = pal.color(QPalette.ColorRole.Text)
        dark = pal.color(QPalette.ColorRole.Base).lightness() < 128
        return {
            "src": "#9aa4b2" if dark else "#5b6673",
            "dst": text.name(),
            "draft": "#7d8794" if dark else "#8a94a0",
            "meta": "#6e7783" if dark else "#98a1ad",
            "err": "#ff6b6b" if dark else "#c62828",
            "accent": "#5ab0ff" if dark else "#1565c0",
        }

    def _render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 8
        c = self._colors()
        parts = []
        for i in self._order[-_SHOWN:]:
            parts.append(self._line_html(self._lines[i], c))
        self.setHtml("".join(parts))
        if at_bottom:
            bar.setValue(bar.maximum())

    @staticmethod
    def _line_html(l: Line, c: dict[str, str]) -> str:
        e = html.escape
        ts = time.strftime("%H:%M:%S", time.localtime(l.created))
        meta = [ts]
        if l.src_lang:
            meta.append(lang_native(l.src_lang))
        if l.dst_final and l.latency_ms:
            detail = f"识别 {l.asr_ms / 1000:.1f}s" + (f" · 翻译 {l.tr_ms / 1000:.1f}s" if l.tr_ms else "")
            meta.append(f"延迟 {l.latency_ms / 1000:.1f}s（{detail}）")
        elif l.skipped:
            meta.append("与目标语言相同，未翻译")

        src_style = f"color:{c['src']};font-size:13px;" + ("" if l.src_final else "font-style:italic;")
        body = f'<div style="{src_style}">{e(l.src)}{"" if l.src_final else " …"}</div>'
        if l.dst:
            done = l.dst_final
            style = f"color:{c['dst'] if not l.dst_draft else c['draft']};font-size:17px;font-weight:{600 if done else 400};"
            if l.dst_draft:
                style += "font-style:italic;"
            cursor = "" if done else f' <span style="color:{c["accent"]}">▍</span>'
            body += f'<div style="{style}margin-top:2px;">{e(l.dst)}{cursor}</div>'
        elif l.src_final and not l.dst_final and not l.skipped:
            body += f'<div style="color:{c["meta"]};font-size:13px;margin-top:2px;">翻译中…</div>'
        if l.error:
            body += f'<div style="color:{c["err"]};font-size:12px;">⚠ {e(l.error)}</div>'
        return (f'<div style="margin:0 0 12px 0;">'
                f'<div style="color:{c["meta"]};font-size:11px;">{e(" · ".join(meta))}</div>{body}</div>')
