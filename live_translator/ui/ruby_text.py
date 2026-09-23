"""Widgets shared by the subtitle overlay and the transcript: word-wrapping text with ruby (注音) above characters,
and a small painted speaker button. Qt rich text has no <ruby>, so the text is laid out and painted by hand."""
from __future__ import annotations

import dataclasses

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy, QWidget

from ..ruby import Token, tokenize

_RUBY_SCALE = 0.5
_MIN_RUBY_PT = 7.0
_RUBY_PAD = 2.0               # extra room next to a character whose annotation is wider than the character

# One colour per speaker (1-based, wraps around): brighter set for dark backgrounds, deeper set for light ones.
SPEAKER_COLORS_DARK = ["#5ab0ff", "#ffb454", "#5fdc8c", "#ff7b8f", "#b6a8ff", "#f5d547", "#4fd8c4", "#ff9a76"]
SPEAKER_COLORS_LIGHT = ["#1565c0", "#c26a00", "#1b8a4b", "#c62848", "#5b4bc4", "#a08100", "#0b8577", "#c2492a"]


_IPA_FONTS = ["Segoe UI", "Lucida Grande", "DejaVu Sans", "Arial Unicode MS"]   # full IPA; kana etc. fall through
_ipa_fams: list[str] | None = None


def _ipa_families() -> list[str]:
    global _ipa_fams
    if _ipa_fams is None:
        have = set(QFontDatabase.families())
        _ipa_fams = [f for f in _IPA_FONTS if f in have][:1]
    return _ipa_fams


def speaker_color(n: int, dark: bool = True) -> str:
    colors = SPEAKER_COLORS_DARK if dark else SPEAKER_COLORS_LIGHT
    return colors[(max(1, n) - 1) % len(colors)]


class RubyText(QWidget):
    """Read-only text that wraps at token boundaries (per character for CJK, per word otherwise) and draws each
    token's ruby centred above it. ``setMaxLines(n)`` keeps only the last *n* wrapped lines (live-caption style)."""

    def __init__(self, parent=None, centered: bool = True):
        super().__init__(parent)
        self._text = ""
        self._tokens: list[Token] = []
        self._ruby_on = False
        self._max_lines = 0
        self._centered = centered
        self._color = QColor("#ffffff")
        self._prefix_tok: Token | None = None
        self._prefix_color = QColor("#ffffff")
        self._cache: dict[int, list[list[tuple[Token, float, float]]]] = {}
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)     # clicks / drags reach the parent

    # ── content ──────────────────────────────────────────────────────────────

    def text(self) -> str:
        return self._text

    def setText(self, text: str, lang: str = "", ruby: bool = False, prefix: str = "",
                prefix_color: str | QColor = "#ffffff") -> None:
        """``prefix`` is a short tag (e.g. a speaker number) drawn in its own colour before the text; it is not part
        of ``text()``."""
        self._text = text
        self._tokens = tokenize(text, lang, ruby)
        self._prefix_tok = None
        if prefix and self._tokens:
            self._prefix_tok = Token(prefix)
            self._prefix_color = QColor(prefix_color)
            self._tokens = [self._prefix_tok, dataclasses.replace(self._tokens[0], space=True), *self._tokens[1:]]
        self._ruby_on = any(t.ruby for t in self._tokens)
        self._changed()

    def setColor(self, color: str | QColor) -> None:
        self._color = QColor(color)
        self.update()

    def setMaxLines(self, n: int) -> None:
        self._max_lines = max(0, n)
        self._changed()

    def has_ruby(self) -> bool:
        return self._ruby_on

    def changeEvent(self, e) -> None:
        if e.type() == e.Type.FontChange:
            self._changed()
        super().changeEvent(e)

    def _changed(self) -> None:
        self._cache.clear()
        self.updateGeometry()
        self.update()

    # ── layout ───────────────────────────────────────────────────────────────

    def _ruby_font(self) -> QFont:
        f = QFont(self.font())
        f.setFamilies(_ipa_families() + f.families())   # CJK UI fonts draw IPA's ˈ ː ʲ as marks over the next letter
        f.setBold(False)
        f.setItalic(False)
        if f.pointSizeF() > 0:
            f.setPointSizeF(max(_MIN_RUBY_PT, f.pointSizeF() * _RUBY_SCALE))
        else:
            f.setPixelSize(max(9, round(f.pixelSize() * _RUBY_SCALE)))
        return f

    def _line_height(self) -> float:
        h = QFontMetricsF(self.font()).height()
        return h + (QFontMetricsF(self._ruby_font()).height() if self._ruby_on else 0.0)

    def _lines(self, width: int) -> list[list[tuple[Token, float, float]]]:
        """Wrapped lines as (token, x, token width); x is relative to the line start."""
        if width in self._cache:
            return self._cache[width]
        fm, rfm = QFontMetricsF(self.font()), QFontMetricsF(self._ruby_font())
        gap = fm.horizontalAdvance(" ")
        units: list[list[tuple[Token, float]]] = []
        for t in self._tokens:                                # a unit never breaks inside: token + trailing punctuation
            w = fm.horizontalAdvance(t.base)
            if t.ruby:
                w = max(w, rfm.horizontalAdvance(t.ruby) + _RUBY_PAD)
            if t.glue and units:
                units[-1].append((t, w))
            else:
                units.append([(t, w)])
        lines: list[list[tuple[Token, float, float]]] = [[]]
        x = 0.0
        for unit in units:
            lead = gap if unit[0][0].space else 0.0
            uw = sum(w for _t, w in unit)
            if lines[-1] and x + lead + uw > width:
                lines.append([])
                x = 0.0
            if lines[-1]:
                x += lead
            for t, w in unit:
                lines[-1].append((t, x, w))
                x += w
        if not lines[-1]:
            lines.pop()
        if self._max_lines and len(lines) > self._max_lines:
            lines = lines[-self._max_lines:]
        self._cache[width] = lines
        return lines

    def line_count(self, width: int) -> int:
        """Wrapped lines at ``width`` (after the ``setMaxLines`` cap)."""
        return len(self._lines(max(1, width))) if self._tokens else 0

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        if not self._tokens:
            return 0
        return int(len(self._lines(max(1, w))) * self._line_height() + 0.999)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self.font())
        w = int(sum(fm.horizontalAdvance(t.base) for t in self._tokens)) + 2
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QSize:
        return QSize(40, int(self._line_height()) if self._tokens else 0)

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, _e) -> None:
        if not self._tokens:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font, rfont = self.font(), self._ruby_font()
        fm, rfm = QFontMetricsF(font), QFontMetricsF(rfont)
        rcolor = QColor(self._color)
        rcolor.setAlpha(190)
        lh = self._line_height()
        y = 0.0
        for line in self._lines(self.width()):
            end = line[-1][1] + line[-1][2]
            off = (self.width() - end) / 2 if self._centered else 0.0
            baseline = y + (rfm.height() if self._ruby_on else 0.0) + fm.ascent()
            for tok, x, w in line:
                p.setFont(font)
                p.setPen(self._prefix_color if tok is self._prefix_tok else self._color)
                p.drawText(QPointF(off + x + (w - fm.horizontalAdvance(tok.base)) / 2, baseline), tok.base)
                if tok.ruby:
                    p.setFont(rfont)
                    p.setPen(rcolor)
                    p.drawText(QPointF(off + x + (w - rfm.horizontalAdvance(tok.ruby)) / 2, y + rfm.ascent()), tok.ruby)
            y += lh


class SpeakerButton(QAbstractButton):
    """A small loudspeaker icon drawn with QPainter (no icon files or emoji fonts needed). ``set_warning(text)``
    dims it, crosses it out and shows ``text`` as the tooltip — a quiet "this can't speak right now" signal."""

    _TIP = "朗读这句字幕（再点一次停止）"

    def __init__(self, parent=None, size: int = 26):
        super().__init__(parent)
        self._color = QColor("#dfe6ee")
        self._warning = ""
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip(self._TIP)

    def setColor(self, color: str | QColor) -> None:
        self._color = QColor(color)
        self.update()

    def set_warning(self, text: str) -> None:
        self._warning = text
        self.setToolTip(text or self._TIP)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        if self.underMouse():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(128, 128, 128, 60))
            p.drawRoundedRect(r, 6, 6)
        s = min(r.width(), r.height())
        cx, cy = r.center().x(), r.center().y()
        u = s / 26.0                                            # design grid: 26 × 26
        color = QColor(self._color)
        if self.isDown() or self._warning:
            color.setAlpha(150 if self.isDown() else 110)
        body = QPainterPath()                                   # speaker body: small box + cone
        body.moveTo(cx - 8 * u, cy - 3 * u)
        body.lineTo(cx - 4 * u, cy - 3 * u)
        body.lineTo(cx + 1 * u, cy - 7 * u)
        body.lineTo(cx + 1 * u, cy + 7 * u)
        body.lineTo(cx - 4 * u, cy + 3 * u)
        body.lineTo(cx - 8 * u, cy + 3 * u)
        body.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPath(body)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 1.6 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for rad in (4.5, 8.5):                    # sound waves
            rr = rad * u
            p.drawArc(QRectF(cx + 1 * u - rr, cy - rr, 2 * rr, 2 * rr), -45 * 16, 90 * 16)
        if self._warning:                                       # a red slash across the icon
            p.setPen(QPen(QColor("#e5484d"), 2 * u, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(cx - 9 * u, cy + 9 * u), QPointF(cx + 9 * u, cy - 9 * u))

    def enterEvent(self, e) -> None:
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.update()
        super().leaveEvent(e)
