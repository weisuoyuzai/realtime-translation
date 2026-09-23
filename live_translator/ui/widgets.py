"""Small custom widgets the redesigned UI is built from. Where a widget replaces a stock one (a segmented control
instead of a combo box, a switch instead of a check box) it keeps the stock API, so code and tests that call
``findData`` / ``currentData`` / ``isChecked`` keep working."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from . import theme


def label(text: str = "", name: str = "", px: int = 0, weight: int = 0, color: str = "") -> QLabel:
    l = QLabel(text)
    if name:
        l.setObjectName(name)
    if px or weight or color:
        css = []
        if px:
            css.append(f"font-size:{px}px")
        if weight:
            css.append(f"font-weight:{weight}")
        if color:
            css.append(f"color:{color}")
        l.setStyleSheet(";".join(css))
    return l


def icon_label(name: str, color: str = theme.TEXT_3, size: int = 14) -> QLabel:
    l = QLabel()
    l.setPixmap(theme.pixmap(name, color, size))
    l.setFixedSize(size, size)
    return l


def vline(h: int = 22) -> QFrame:
    f = QFrame()
    f.setObjectName("vline")
    f.setFixedSize(1, h)
    return f


def icon_button(name: str, tip: str, size: int = 16, obj: str = "iconBtn") -> QToolButton:
    b = QToolButton()
    b.setObjectName(obj)
    b.setIcon(theme.icon(name, theme.TEXT_2, size=size))
    b.setIconSize(QSize(size, size))
    b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def section_label(text: str, icon: str = "") -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    if icon:
        h.addWidget(icon_label(icon, theme.TEXT_3, 13))
    h.addWidget(label(text, "section"))
    h.addStretch(1)
    return w


def banner(kind: str, icon: str, text: str) -> QFrame:
    """kind: "warnBanner" or "infoBanner"."""
    f = QFrame()
    f.setObjectName(kind)
    h = QHBoxLayout(f)
    h.setContentsMargins(12, 9, 12, 9)
    h.setSpacing(10)
    h.addWidget(icon_label(icon, theme.WARN if kind == "warnBanner" else theme.ACCENT, 14), 0, Qt.AlignmentFlag.AlignTop)
    t = QLabel(text)
    t.setWordWrap(True)
    h.addWidget(t, 1)
    f.text_label = t
    return f


class PillToggle(QToolButton):
    """Checkable icon + text button used in the top bar's toggle group (behaves like a check box)."""

    def __init__(self, icon: str, text: str, tip: str = ""):
        super().__init__()
        self.setObjectName("pill")
        self.setCheckable(True)
        self.setText(text)
        self.setToolTip(tip)
        self.setIcon(theme.icon(icon, theme.TEXT_3, theme.ACCENT, size=14))
        self.setIconSize(QSize(14, 14))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class Segmented(QFrame):
    """A row of mutually exclusive buttons with the QComboBox API the rest of the code relies on."""
    currentIndexChanged = Signal(int)

    def __init__(self, items=(), expand: bool = False):
        super().__init__()
        self.setObjectName("seg")
        self._data: list = []
        self._expand = expand
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idToggled.connect(self._toggled)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(3, 3, 3, 3)
        self._lay.setSpacing(3)
        for it in items:
            self.addItem(*it)
        if not expand:
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def addItem(self, text: str, data=None, icon: str = "") -> None:
        b = QToolButton()
        b.setObjectName("segBtn")
        b.setCheckable(True)
        b.setText(text)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        if icon:
            b.setIcon(theme.icon(icon, theme.TEXT_3, theme.ACCENT, size=14))
            b.setIconSize(QSize(14, 14))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        if self._expand:
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        i = len(self._data)
        self._data.append(data)
        self._group.addButton(b, i)
        self._lay.addWidget(b)
        if i == 0:
            b.setChecked(True)

    def _toggled(self, i: int, on: bool) -> None:
        if on:
            self.currentIndexChanged.emit(i)

    def count(self) -> int:
        return len(self._data)

    def itemData(self, i: int):
        return self._data[i] if 0 <= i < len(self._data) else None

    def itemText(self, i: int) -> str:
        b = self._group.button(i)
        return b.text() if b else ""

    def findData(self, data) -> int:
        return self._data.index(data) if data in self._data else -1

    def currentIndex(self) -> int:
        return self._group.checkedId()

    def currentData(self):
        return self.itemData(self.currentIndex())

    def currentText(self) -> str:
        return self.itemText(self.currentIndex())

    def setCurrentIndex(self, i: int) -> None:
        b = self._group.button(i)
        if b is not None:
            b.setChecked(True)

    def isEditable(self) -> bool:
        return False

    def setItemEnabled(self, i: int, on: bool) -> None:
        b = self._group.button(i)
        if b is not None:
            b.setEnabled(on)


class Switch(QCheckBox):
    """A check box drawn as a label (with an optional dimmer description line) and a toggle switch on the right.
    A description in full-width parentheses at the end of the text is split off automatically."""

    _W, _H = 34, 20

    def __init__(self, text: str = "", desc: str | None = None):
        if desc is None and text.endswith("）") and "（" in text:
            text, desc = text[:text.index("（")], text[text.index("（") + 1:-1]
        super().__init__(text)
        self._desc = desc or ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def description(self) -> str:
        return self._desc

    def _desc_font(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(11)
        return f

    def _text_rects(self, w: int) -> tuple[QRectF, QRectF]:
        tw = max(40, w - self._W - 12)
        fm, dm = QFontMetrics(self.font()), QFontMetrics(self._desc_font())
        flags = int(Qt.TextFlag.TextWordWrap)
        th = fm.boundingRect(0, 0, tw, 10000, flags, self.text()).height()
        dh = dm.boundingRect(0, 0, tw, 10000, flags, self._desc).height() if self._desc else 0
        return QRectF(0, 0, tw, th), QRectF(0, th + (2 if dh else 0), tw, dh)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        t, d = self._text_rects(w)
        return int(max(self._H, d.bottom() if self._desc else t.bottom())) + 4

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        w = min(360, fm.horizontalAdvance(self.text()) + self._W + 16)
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QSize:
        return QSize(self._W + 60, self._H)

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        t, d = self._text_rects(self.width())
        total = d.bottom() if self._desc else t.bottom()
        oy = max(0.0, (self.height() - total) / 2)
        p.setFont(self.font())
        p.setPen(QColor(theme.TEXT if enabled else theme.TEXT_3))
        p.drawText(t.translated(0, oy), int(Qt.TextFlag.TextWordWrap), self.text())
        if self._desc:
            p.setFont(self._desc_font())
            p.setPen(QColor(theme.TEXT_3))
            p.drawText(d.translated(0, oy), int(Qt.TextFlag.TextWordWrap), self._desc)
        on = self.isChecked()
        x, y = self.width() - self._W, (self.height() - self._H) / 2
        track = QColor(theme.ACCENT if on else theme.BORDER)
        if not enabled:
            track.setAlpha(110)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(x, y, self._W, self._H), self._H / 2, self._H / 2)
        p.setBrush(QColor("#FFFFFF" if enabled else "#9AA4B2"))
        kx = x + (self._W - self._H + 2 if on else 2)
        p.drawEllipse(QRectF(kx, y + 2, self._H - 4, self._H - 4))


class LevelMeter(QWidget):
    """Input level as a row of bars (0–100). Same value API as the QProgressBar it replaces."""

    _N = 12
    _SHAPE = (0.3, 0.5, 0.75, 1.0, 0.65, 0.9, 0.45, 0.6, 0.35, 0.25, 0.2, 0.2)

    def __init__(self):
        super().__init__()
        self._v = 0
        self.setFixedSize(self._N * 5, 20)
        self.setToolTip("输入音量")

    def value(self) -> int:
        return self._v

    def setValue(self, v: int) -> None:
        v = max(0, min(100, int(v)))
        if v != self._v:
            self._v = v
            self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        lit = round(self._v / 100 * self._N)
        for i, s in enumerate(self._SHAPE):
            h = 3 + (self.height() - 3) * s * (0.35 + 0.65 * self._v / 100)
            p.setBrush(QColor(theme.LIVE if i < lit else theme.BORDER))
            p.drawRoundedRect(QRectF(i * 5, self.height() - h, 3, h), 1.5, 1.5)


class ElidedLabel(QLabel):
    """Single-line label that elides with "…" instead of growing; the full text is in the tooltip."""

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text if len(text) > 30 else "")

    def minimumSizeHint(self) -> QSize:
        return QSize(40, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(120, super().sizeHint().height())

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        fm = QFontMetrics(self.font())
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width()))


class _CardHeader(QAbstractButton):
    def __init__(self, icon: str, title: str):
        super().__init__()
        self._icon, self._title, self._summary, self._open = icon, title, "", False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(54)

    def set_summary(self, s: str) -> None:
        self._summary = s
        self.setToolTip(s)
        self.update()

    def set_open(self, on: bool) -> None:
        self._open = on
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(260, 54)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hover = self.underMouse()
        if hover and not self._open:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.SURFACE_2))
            p.drawRoundedRect(QRectF(self.rect()), 10, 10)
        badge = QRectF(12, 13, 28, 28)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(79, 163, 255, 36) if self._open else QColor(theme.SURFACE_2 if not hover else theme.BORDER))
        p.drawRoundedRect(badge, 7, 7)
        p.drawPixmap(int(badge.x() + 6), int(badge.y() + 6), theme.pixmap(self._icon, theme.ACCENT if self._open else theme.TEXT_2, 16))
        tx, tw = 52, self.width() - 52 - 34
        f = QFont(self.font())
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT if self.isEnabled() else theme.TEXT_2))
        p.drawText(QRectF(tx, 9, tw, 20), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self._title)
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_3))
        s = QFontMetrics(f).elidedText(self._summary, Qt.TextElideMode.ElideRight, int(tw))
        p.drawText(QRectF(tx, 28, tw, 18), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), s)
        p.drawPixmap(self.width() - 26, 20, theme.pixmap("chevron-up" if self._open else "chevron-down", theme.TEXT_3, 14))

    def enterEvent(self, e) -> None:
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.update()
        super().leaveEvent(e)


class Card(QFrame):
    """Collapsible step card: a header (icon, title, one-line summary) and a body shown when open."""
    toggled = Signal(bool)

    def __init__(self, icon: str, title: str, body: QWidget):
        super().__init__()
        self.setObjectName("card")
        self.header = _CardHeader(icon, title)
        self.body = body
        wrap = QWidget()
        bl = QVBoxLayout(wrap)
        bl.setContentsMargins(12, 0, 12, 14)
        bl.addWidget(body)
        self._wrap = wrap
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(wrap)
        self.header.clicked.connect(lambda: self.set_open(not self.is_open(), user=True))
        self.set_open(False)

    def is_open(self) -> bool:
        return not self._wrap.isHidden()

    def set_open(self, on: bool, user: bool = False) -> None:
        self._wrap.setVisible(on)
        self.header.set_open(on)
        self.setProperty("open", "true" if on else "false")
        theme.repolish(self)
        if user:
            self.toggled.emit(on)

    def set_summary(self, s: str) -> None:
        self.header.set_summary(s)


class NavItem(QAbstractButton):
    """Preferences navigation entry: icon, title and a one-line subtitle."""

    def __init__(self, icon: str, title: str, subtitle: str):
        super().__init__()
        self._icon, self._title, self._sub = icon, title, subtitle
        self._dim = False
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_dimmed(self, on: bool) -> None:
        self._dim = on
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(180, 52)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        if on or self.underMouse():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.SURFACE_2 if on else "#161A21"))
            p.drawRoundedRect(QRectF(self.rect()), 8, 8)
        if self._dim:
            p.setOpacity(0.35)
        p.drawPixmap(12, 18, theme.pixmap(self._icon, theme.ACCENT if on else theme.TEXT_3, 16))
        f = QFont(self.font())
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.DemiBold if on else QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT if on else theme.TEXT_2))
        p.drawText(QRectF(38, 8, self.width() - 44, 19), int(Qt.AlignmentFlag.AlignVCenter), self._title)
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_3))
        p.drawText(QRectF(38, 27, self.width() - 44, 17), int(Qt.AlignmentFlag.AlignVCenter), self._sub)

    def enterEvent(self, e) -> None:
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.update()
        super().leaveEvent(e)
