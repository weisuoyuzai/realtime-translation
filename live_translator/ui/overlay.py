"""Always-on-top subtitle bar that floats above any application (video players, meeting windows, games in
borderless mode)."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QMenu, QSizeGrip, QVBoxLayout, QWidget

from ..config import OverlayCfg
from ..models import Line

_IDLE_HIDE_MS = 9000      # fade the text away when nothing new arrives, so the screen isn't cluttered


class SubtitleOverlay(QWidget):
    geometry_changed = Signal()
    closed = Signal()

    def __init__(self, cfg: OverlayCfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # macOS hides Tool windows whenever the app loses focus — exactly when subtitles matter most.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setMinimumSize(320, 70)

        self._src = QLabel()
        self._dst = QLabel()
        for l in (self._src, self._dst):
            l.setWordWrap(True)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._grip = QSizeGrip(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(4)
        lay.addWidget(self._src)
        lay.addWidget(self._dst, 1)
        lay.addWidget(self._grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self._drag: QPoint | None = None
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self._fade)
        self._current_id = -1
        self.apply_style()
        self.resize(cfg.w, cfg.h)
        self._place()

    # ── content ──────────────────────────────────────────────────────────────

    def show_line(self, line: Line) -> None:
        if line.removed:
            if line.id == self._current_id:
                self._fade()
            return
        self._current_id = line.id
        dst = line.dst or ""
        # Until a translation exists show the recognised text large so there is never a blank bar.
        self._dst.setText(dst if dst else line.src)
        self._src.setText(line.src if dst and self.cfg.show_source else "")
        self._src.setVisible(bool(self._src.text()))
        self._dst.setStyleSheet(f"color:{'#dfe6ee' if (line.dst_draft or not dst) else '#ffffff'};")
        f = self._dst.font()
        f.setItalic(line.dst_draft or not dst)
        self._dst.setFont(f)
        self.show()
        self._idle.start(_IDLE_HIDE_MS)

    def _fade(self) -> None:
        self._src.clear()
        self._dst.clear()
        self._src.hide()

    # ── look & behaviour ─────────────────────────────────────────────────────

    def apply_style(self) -> None:
        big = QFont(self.font())
        big.setPointSize(max(10, self.cfg.font_size))
        big.setBold(True)
        small = QFont(self.font())
        small.setPointSize(max(8, int(self.cfg.font_size * 0.55)))
        self._dst.setFont(big)
        self._src.setFont(small)
        self._src.setStyleSheet("color:#b8c2cf;")
        self._dst.setStyleSheet("color:#ffffff;")
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self.cfg.click_through)
        self._grip.setVisible(not self.cfg.click_through)
        if self.isVisible():
            self.show()                          # changing window flags hides the window

    def _place(self) -> None:
        if self.cfg.x >= 0 and self.cfg.y >= 0:
            self.move(self.cfg.x, self.cfg.y)
            return
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width() // 2, screen.bottom() - self.height() - 60)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(12, 14, 18, int(255 * self.cfg.opacity)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 14, 14)

    def _persist(self) -> None:
        p = self.pos()
        self.cfg.x, self.cfg.y, self.cfg.w, self.cfg.h = p.x(), p.y(), self.width(), self.height()
        self.geometry_changed.emit()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e) -> None:
        if self._drag is not None:
            self._drag = None
            self._persist()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.isVisible():
            self._persist()

    def contextMenuEvent(self, e) -> None:
        m = QMenu(self)
        m.addAction("字体放大", lambda: self._font(+2))
        m.addAction("字体缩小", lambda: self._font(-2))
        a = m.addAction("显示原文")
        a.setCheckable(True)
        a.setChecked(self.cfg.show_source)
        a.toggled.connect(self._toggle_source)
        m.addSeparator()
        m.addAction("关闭字幕窗", self._close)
        m.exec(e.globalPos())

    def _font(self, delta: int) -> None:
        self.cfg.font_size = max(12, min(72, self.cfg.font_size + delta))
        self.apply_style()
        self._persist()

    def _toggle_source(self, on: bool) -> None:
        self.cfg.show_source = on
        self._persist()

    def _close(self) -> None:
        self.hide()
        self.closed.emit()
