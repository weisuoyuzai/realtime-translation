"""Custom window title bar for the frameless main window: logo, title, the menu bar, and minimize / maximize /
close buttons. Dragging the empty part moves the window through the OS (so Windows snap layouts still work),
double-clicking it maximizes / restores."""
from __future__ import annotations

import sys

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton, QFrame, QHBoxLayout, QLabel, QMenuBar, QSizePolicy, QWidget

from . import theme

# macOS keeps its native frame (traffic lights, global menu bar); elsewhere the main window draws its own.
CUSTOM_FRAME = sys.platform != "darwin"
CLOSE_HOVER = "#E81123"


class WindowButton(QAbstractButton):
    """46×36 caption button, painted so the icon can turn white on the red close hover."""

    def __init__(self, icon: str, tip: str, close: bool = False):
        super().__init__()
        self._icon = icon
        self._close = close
        self.setToolTip(tip)
        self.setFixedSize(46, 36)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_icon(self, icon: str, tip: str) -> None:
        self._icon = icon
        self.setToolTip(tip)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(46, 36)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        hover, down = self.underMouse(), self.isDown()
        if hover or down:
            if self._close:
                bg = QColor(CLOSE_HOVER)
                if down:
                    bg = bg.darker(115)
            else:
                bg = QColor(theme.BORDER if down else theme.SURFACE_2)
            p.fillRect(self.rect(), bg)
        active = self.window().isActiveWindow()
        color = "#FFFFFF" if self._close and (hover or down) else theme.TEXT_2 if active else theme.TEXT_3
        size = 12 if self._icon in ("maximize", "restore") else 14
        pm = theme.pixmap(self._icon, color, size, 1.6)
        p.drawPixmap(int((self.width() - size) / 2), int((self.height() - size) / 2), pm)

    def enterEvent(self, e) -> None:
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.update()
        super().leaveEvent(e)


class TitleBar(QFrame):
    def __init__(self, window: QWidget, menubar: QMenuBar, title: str):
        super().__init__()
        self._win = window
        self.setObjectName("titleBar")
        self.setFixedHeight(36)
        logo = QLabel()
        logo.setFixedSize(18, 18)
        logo.setPixmap(theme.app_icon().pixmap(QSize(18, 18), self.devicePixelRatioF()))
        self.title = QLabel(title)
        self.title.setObjectName("winTitle")
        menubar.setObjectName("titleMenu")
        menubar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.min_btn = WindowButton("minus", "最小化")
        self.max_btn = WindowButton("maximize", "最大化")
        self.close_btn = WindowButton("x", "关闭", close=True)
        self.min_btn.clicked.connect(window.showMinimized)
        self.max_btn.clicked.connect(self.toggle_maximized)
        self.close_btn.clicked.connect(window.close)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(logo)
        lay.addWidget(self.title)
        lay.addSpacing(2)
        lay.addWidget(menubar)
        lay.addStretch(1)                              # the empty stretch is the drag area
        btns = QHBoxLayout()
        btns.setSpacing(0)
        for b in (self.min_btn, self.max_btn, self.close_btn):
            btns.addWidget(b)
        lay.addLayout(btns)

    def toggle_maximized(self) -> None:
        w = self._win
        w.showNormal() if w.isMaximized() else w.showMaximized()

    def sync(self) -> None:
        """Follow the window state (maximized ↔ restore icon) and focus (dimmed when inactive)."""
        if self._win.isMaximized():
            self.max_btn.set_icon("restore", "向下还原")
        else:
            self.max_btn.set_icon("maximize", "最大化")
        active = self._win.isActiveWindow()
        self.title.setStyleSheet(f"color:{theme.TEXT if active else theme.TEXT_3};")
        for b in (self.min_btn, self.max_btn, self.close_btn):
            b.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            handle = self._win.windowHandle()
            if handle is not None and handle.startSystemMove():
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximized()
            e.accept()


class FramelessMixin:
    """Resize handles, outline and Windows 11 rounded corners for a frameless QMainWindow.

    The window keeps a thin margin around its content; the margin belongs to the window itself, so it receives the
    mouse there and hands the drag to the OS (``startSystemResize``)."""
    _EDGE = 4
    _frameless = False                # set by the window; events can arrive before its __init__ runs

    def _init_frame(self) -> None:
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setMouseTracking(True)
        self._corner_done = False
        self._update_margins()

    def _update_margins(self) -> None:
        m = 0 if self.isMaximized() or self.isFullScreen() else self._EDGE
        self.setContentsMargins(m, m, m, m)

    def _edges(self, pos) -> Qt.Edge:
        if self.isMaximized() or self.isFullScreen():
            return Qt.Edge(0)
        e, r = Qt.Edge(0), self.rect()
        g = self._EDGE + 2
        if pos.x() <= g:
            e |= Qt.Edge.LeftEdge
        elif pos.x() >= r.width() - g:
            e |= Qt.Edge.RightEdge
        if pos.y() <= g:
            e |= Qt.Edge.TopEdge
        elif pos.y() >= r.height() - g:
            e |= Qt.Edge.BottomEdge
        return e

    @staticmethod
    def _cursor(edges) -> Qt.CursorShape:
        E = Qt.Edge
        if edges in (E.LeftEdge | E.TopEdge, E.RightEdge | E.BottomEdge):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (E.RightEdge | E.TopEdge, E.LeftEdge | E.BottomEdge):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (E.LeftEdge | E.RightEdge):
            return Qt.CursorShape.SizeHorCursor
        if edges & (E.TopEdge | E.BottomEdge):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def mouseMoveEvent(self, e) -> None:
        if not self._frameless:
            return super().mouseMoveEvent(e)
        self.setCursor(self._cursor(self._edges(e.position().toPoint())))
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e) -> None:
        edges = self._edges(e.position().toPoint()) if self._frameless else Qt.Edge(0)
        if e.button() == Qt.MouseButton.LeftButton and edges and self.windowHandle() is not None:
            if self.windowHandle().startSystemResize(edges):
                e.accept()
                return
        super().mousePressEvent(e)

    def leaveEvent(self, e) -> None:
        if self._frameless:
            self.unsetCursor()
        super().leaveEvent(e)

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        if self._frameless and not (self.isMaximized() or self.isFullScreen()):
            p = QPainter(self)
            p.setPen(QColor(theme.BORDER_HOVER if self.isActiveWindow() else theme.BORDER))
            p.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))

    def _round_corners(self) -> None:
        """Windows 11: ask DWM for rounded corners (no-op elsewhere / on older Windows)."""
        if self._corner_done or sys.platform != "win32":
            return
        self._corner_done = True
        try:
            import ctypes
            pref = ctypes.c_int(2)                       # DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), 33, ctypes.byref(pref), ctypes.sizeof(pref))
        except Exception:
            pass
