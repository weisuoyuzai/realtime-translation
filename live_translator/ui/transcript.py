"""Scrolling subtitle history: original text (small) and translation (large) per utterance.

Each utterance is a row widget that is updated in place, so streaming text never rebuilds the document and never
touches the scroll position. The view follows the newest line only while you are at the bottom; scroll up (wheel or
scrollbar) and it stays exactly where you left it."""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QVBoxLayout, QWidget)

from ..languages import lang_native
from ..models import Line
from .ruby_text import RubyText, SpeakerButton, speaker_color

_SHOWN = 80          # rows kept; older ones stay in memory (and in the saved transcript file)
_BOTTOM_SLACK = 8    # px from the bottom that still counts as "at the bottom"


def _px(size: int, weight: int = 400, italic: bool = False) -> QFont:
    f = QFont()
    f.setPixelSize(size)
    f.setWeight(QFont.Weight(weight))
    f.setItalic(italic)
    return f


class _Row(QWidget):
    """One utterance. Talks to the view only through a signal (no back-reference: rows and view would form a cycle
    that Python's garbage collector then tears down at an arbitrary moment)."""
    speak_requested = Signal(object)          # the Line to read

    def __init__(self):
        super().__init__()
        self.line: Line | None = None
        self.meta = QLabel()
        self.meta.setFont(_px(11))
        self.spk = QLabel()                                   # "说话人 2", in that speaker's colour
        self.spk.setFont(_px(11, 600))
        self.src = RubyText(centered=False)
        self.dst = RubyText(centered=False)
        self.wait = QLabel("翻译中…")
        self.wait.setFont(_px(13))
        self.warn = QLabel()
        self.warn.setFont(_px(11))
        self.warn.setWordWrap(True)
        self.err = QLabel()
        self.err.setFont(_px(12))
        self.err.setWordWrap(True)
        self.speaker = SpeakerButton(size=24)
        self.speaker.clicked.connect(self._speak)
        for w in (self.meta, self.wait, self.warn, self.err):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(self.spk)
        head.addWidget(self.meta, 1)
        col.addLayout(head)
        for w in (self.src, self.dst, self.wait, self.warn, self.err):
            col.addWidget(w)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(8)
        lay.addWidget(self.speaker, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(col, 1)

    def _speak(self) -> None:
        if self.line is not None:
            self.speak_requested.emit(self.line)

    def update_line(self, l: Line, c: dict[str, str], learning: bool, ruby_src: bool, ruby_dst: bool) -> None:
        self.line = l
        ts = time.strftime("%H:%M:%S", time.localtime(l.created))
        meta = [ts]
        if l.src_lang:
            meta.append(lang_native(l.src_lang))
        if l.dst_final and l.latency_ms:
            detail = f"识别 {l.asr_ms / 1000:.1f}s" + (f" · 翻译 {l.tr_ms / 1000:.1f}s" if l.tr_ms else "")
            meta.append(f"延迟 {l.latency_ms / 1000:.1f}s（{detail}）")
        elif l.skipped:
            meta.append("与目标语言相同，未翻译")
        self.meta.setText(" · ".join(meta))
        self.meta.setStyleSheet(f"color:{c['meta']};")
        self.spk.setText(f"说话人 {l.speaker}" if l.speaker else "")
        self.spk.setStyleSheet(f"color:{speaker_color(l.speaker, c['dark'] == '1')};")
        self.spk.setVisible(bool(l.speaker))

        self.src.setFont(_px(13, italic=not l.src_final))
        self.src.setColor(c["src"])
        self.src.setText(l.src + ("" if l.src_final else " …"), l.src_lang, ruby_src)

        if l.dst:
            self.dst.setFont(_px(17, 600 if l.dst_final else 400, italic=l.dst_draft))
            self.dst.setColor(c["draft"] if l.dst_draft else c["dst"])
            self.dst.setText(l.dst + ("" if l.dst_final else " ▍"), l.dst_lang, ruby_dst)
        else:
            self.dst.setText("")
        self.dst.setVisible(bool(l.dst))
        self.wait.setStyleSheet(f"color:{c['meta']};")
        self.wait.setVisible(not l.dst and l.src_final and not l.dst_final and not l.skipped)
        self.warn.setText(f"⚠ {l.uncertain_reason}" if l.uncertain_reason else "")
        self.warn.setStyleSheet(f"color:{c['warn']};")
        self.warn.setVisible(bool(l.uncertain_reason))
        self.err.setText(f"⚠ {l.error}" if l.error else "")
        self.err.setStyleSheet(f"color:{c['err']};")
        self.err.setVisible(bool(l.error))
        self.speaker.setColor(c["accent"])
        self.speaker.setVisible(learning and bool(l.src or l.dst))

    def contextMenuEvent(self, e) -> None:
        l = self.line
        if l is None:
            return
        m = QMenu(self)
        clip = QGuiApplication.clipboard()
        if l.src:
            m.addAction("复制原文", lambda: clip.setText(l.src))
        if l.dst:
            m.addAction("复制译文", lambda: clip.setText(l.dst))
        m.addAction("朗读", self._speak)
        m.exec(e.globalPos())


class TranscriptView(QScrollArea):
    speak_requested = Signal(object)          # a Line

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setBackgroundRole(QPalette.ColorRole.Base)
        self._body = QWidget()
        self._body.setBackgroundRole(QPalette.ColorRole.Base)
        self._body.setAutoFillBackground(True)
        self._col = QVBoxLayout(self._body)
        self._col.setContentsMargins(14, 10, 14, 10)
        self._col.setSpacing(0)
        self._hint = QLabel("点击「开始」后，字幕会显示在这里。")
        self._hint.setEnabled(False)                       # greyed like placeholder text
        self._col.addWidget(self._hint)
        self._col.addStretch(1)
        self.setWidget(self._body)

        self._lines: dict[int, Line] = {}
        self._order: list[int] = []
        self._rows: dict[int, _Row] = {}
        self._dirty: set[int] = set()
        self._trimmed: set[int] = set()
        self._learning = False
        self._scope = "both"
        self._follow = True                                # stick to the newest line while at the bottom
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(self._on_range)
        self._timer = QTimer(self)
        self._timer.setInterval(70)                        # coalesce bursts of streaming updates into ≤ 14 updates/s
        self._timer.timeout.connect(self._render)
        self._timer.start()

    # ── model ────────────────────────────────────────────────────────────────

    def line(self, line_id: int) -> Line | None:
        return self._lines.get(line_id)

    def upsert(self, line: Line) -> None:
        if line.removed:
            if line.id in self._lines:
                del self._lines[line.id]
                self._order.remove(line.id)
                self._dirty.discard(line.id)
                self._drop_row(line.id)
                self._hint.setVisible(not self._order)
            return
        if line.id not in self._lines:
            self._order.append(line.id)
        self._lines[line.id] = line
        self._dirty.add(line.id)

    def clear_all(self) -> None:
        self._lines.clear()
        self._order.clear()
        self._dirty.clear()
        self._trimmed.clear()
        for i in list(self._rows):
            self._drop_row(i)
        self._hint.setVisible(True)
        self._follow = True

    def set_learning(self, on: bool, scope: str | None = None) -> None:
        """Learning mode on/off; ``scope`` says which text gets ruby: "src", "dst" or "both"."""
        scope = scope or self._scope
        if (on, scope) != (self._learning, self._scope):
            self._learning, self._scope = on, scope
            self._dirty.update(self._rows)

    def plain_text(self) -> str:
        out = []
        for i in self._order:
            l = self._lines[i]
            who = f"[说话人 {l.speaker}] " if l.speaker else ""
            out.append(f"[{time.strftime('%H:%M:%S', time.localtime(l.created))}] {who}{l.src}")
            if l.dst:
                out.append(f"    → {l.dst}")
        return "\n".join(out)

    # ── rendering ────────────────────────────────────────────────────────────

    def _colors(self) -> dict[str, str]:
        pal = self.palette()
        dark = pal.color(QPalette.ColorRole.Base).lightness() < 128
        return {
            "src": "#9aa4b2" if dark else "#5b6673",
            "dst": pal.color(QPalette.ColorRole.Text).name(),
            "draft": "#7d8794" if dark else "#8a94a0",
            "meta": "#6e7783" if dark else "#98a1ad",
            "warn": "#ffb454" if dark else "#a06400",
            "err": "#ff6b6b" if dark else "#c62828",
            "accent": "#5ab0ff" if dark else "#1565c0",
            "dark": "1" if dark else "",
        }

    def _render(self) -> None:
        if not self._dirty:
            return
        dirty, self._dirty = self._dirty, set()
        c = self._colors()
        self._hint.setVisible(False)
        for i in self._order:                               # creation order == display order (new ids are appended)
            if i not in dirty:
                continue
            row = self._rows.get(i)
            if row is None:
                if i in self._trimmed:                      # a late update of a line that already scrolled out
                    continue
                row = self._rows[i] = _Row()
                row.speak_requested.connect(self.speak_requested)
                self._col.insertWidget(self._col.count() - 1, row)      # above the trailing stretch
            row.update_line(self._lines[i], c, self._learning, self._learning and self._scope in ("both", "src"),
                            self._learning and self._scope in ("both", "dst"))
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest rows — but only while following the newest line: removing rows above the viewport
        would shift what the reader is looking at."""
        if not self._follow:
            return
        while len(self._rows) > _SHOWN:
            oldest = next(iter(self._rows))
            self._trimmed.add(oldest)
            self._drop_row(oldest)

    def _drop_row(self, line_id: int) -> None:
        row = self._rows.pop(line_id, None)
        if row is not None:
            self._col.removeWidget(row)
            row.deleteLater()

    # ── scrolling ────────────────────────────────────────────────────────────

    def _on_scrolled(self, value: int) -> None:
        self._follow = value >= self.verticalScrollBar().maximum() - _BOTTOM_SLACK

    def _on_range(self, _lo: int, hi: int) -> None:
        if self._follow:                                    # content grew while at the bottom → follow it
            self.verticalScrollBar().setValue(hi)
