"""Always-on-top subtitle bar that floats above any application (video players, meeting windows, games in
borderless mode).

The bar shows the last few *sentences* (setting: 句数). Each sentence is shown in full, the current one at the bottom
with its original text above it; earlier ones stack above it, dimmer. By default the bar is exactly as tall as that
content (growing upwards); drag the bottom-right corner vertically to give it a height of your own instead — the
newest sentence stays at the bottom and whatever does not fit at the top is cut off.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMenu, QSizeGrip, QVBoxLayout, QWidget

from ..config import OverlayCfg, ruby_wanted
from ..models import Line
from .ruby_text import RubyText, SpeakerButton, speaker_color

_IDLE_HIDE_MS = 9000      # fade the text away when nothing new arrives, so the screen isn't cluttered
_IDLE_HIDE_LEARN_MS = 30000   # …but give a learner time to read the ruby and press the speaker
_SENTENCE_CHOICES = (1, 2, 3, 4, 6, 8)
_MAX_SENTENCES = 8
_HISTORY_ROWS = _MAX_SENTENCES - 1
_SENTENCE_LINES = 8       # a single absurdly long sentence is cut to its last 8 lines
_OLD_COLOR = "#aeb8c5"    # earlier sentences are dimmer than the current one
_MIN_H = 70
_CUSTOM_H_SLACK = 6       # px the user must be off the automatic height before it counts as "my own height"


class SubtitleOverlay(QWidget):
    geometry_changed = Signal()
    closed = Signal()
    speak_requested = Signal(object)          # the Line whose speaker button was pressed
    learning_changed = Signal(bool)           # toggled from the right-click menu

    def __init__(self, cfg: OverlayCfg):
        super().__init__()
        self.cfg = cfg
        self._drag: QPoint | None = None
        self._fitting = False
        self._ready = False                       # False during construction: resizes then are ours, not the user's
        self._entries: list[Line] = []            # recent sentences, oldest first; the last one is "current"
        self._speakers: set[int] = set()          # speaker numbers seen: tags are only shown once there are two
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # macOS hides Tool windows whenever the app loses focus — exactly when subtitles matter most.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setMinimumSize(320, _MIN_H)

        self._hist = [RubyText() for _ in range(_HISTORY_ROWS)]    # _hist[0] sits right above the current sentence
        self._src = RubyText()
        self._dst = RubyText()
        self._speaker = SpeakerButton()
        self._speaker.clicked.connect(self._speak)
        self._balance = QWidget()                 # mirrors the speaker so the text stays centred in the bar
        self._balance.setFixedSize(self._speaker.size())
        self._grip = QSizeGrip(self)              # floats in the corner
        self._col = col = QVBoxLayout()
        col.setSpacing(4)
        for h in reversed(self._hist):
            h.hide()
            col.addWidget(h)
        col.addWidget(self._src)
        col.addWidget(self._dst)
        # The content lives in a child that is placed by hand (bottom-aligned), so that when the bar is shorter than
        # its content the oldest text is clipped at the top instead of the layout squeezing everything together.
        self._body = QWidget(self)
        lay = QHBoxLayout(self._body)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(8)
        lay.addWidget(self._speaker, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(col, 1)
        lay.addWidget(self._balance, 0, Qt.AlignmentFlag.AlignVCenter)

        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self._fade)
        self.apply_style()
        self.resize(cfg.w, cfg.h)
        self._fit()
        self._place()
        self._ready = True

    # ── content ──────────────────────────────────────────────────────────────

    @property
    def _line(self) -> Line | None:
        return self._entries[-1] if self._entries else None

    def show_line(self, line: Line) -> None:
        if line.removed:
            n = len(self._entries)
            self._entries = [l for l in self._entries if l.id != line.id]
            if len(self._entries) != n:
                self._render() if self._entries else self._fade()
            return
        for i, old in enumerate(self._entries):
            if old.id == line.id:
                self._entries[i] = line              # a streaming update of a sentence already on screen
                break
        else:
            self._entries.append(line)
            del self._entries[:-_MAX_SENTENCES]
        if line.speaker:
            self._speakers.add(line.speaker)
        self._render()
        self.show()
        self._idle.start(_IDLE_HIDE_LEARN_MS if self.cfg.learning else _IDLE_HIDE_MS)

    def _tag(self, line: Line) -> tuple[str, str]:
        """(prefix, colour): who spoke (only once more than one speaker has been heard), plus a "⚠" when this
        line's recognition may be unreliable (e.g. overlapping voices)."""
        if line.speaker and len(self._speakers) > 1:
            prefix, color = f"[{line.speaker}]", speaker_color(line.speaker)
            return (f"{prefix}⚠" if line.uncertain_reason else prefix), color
        if line.uncertain_reason:
            return "⚠", "#ffb454"
        return "", "#ffffff"

    def _render(self) -> None:
        """Current sentence at the bottom (original above, translation below); earlier ones stack above it. A new
        sentence appearing therefore never wipes out the one before it."""
        self._sync_speaker()
        line, cfg = self._line, self.cfg
        if line is None:
            for w in (self._src, self._dst, *self._hist):
                w.setText("")
                w.hide()
            self._fit()
            return
        n = min(max(1, cfg.max_sentences), _MAX_SENTENCES)
        dst = line.dst or ""
        prefix, color = self._tag(line)
        # Until a translation exists show the recognised text large so there is never a blank bar.
        self._dst.show()
        self._dst.setMaxLines(_SENTENCE_LINES)
        self._dst.setText(dst or line.src, line.dst_lang if dst else line.src_lang, ruby_wanted(cfg, bool(dst)),
                          prefix, color)
        src = line.src if dst and cfg.show_source else ""
        self._src.setMaxLines(_SENTENCE_LINES)
        self._src.setText(src, line.src_lang, ruby_wanted(cfg, False))
        self._src.setVisible(bool(src))
        draft = line.dst_draft or not dst
        self._dst.setColor("#dfe6ee" if draft else "#ffffff")
        f = self._dst.font()
        f.setItalic(draft)
        self._dst.setFont(f)

        older = self._entries[-n:-1] if n > 1 else []
        for i, row in enumerate(self._hist):
            if i < len(older):
                old = older[-1 - i]                  # _hist[0] = the sentence just before the current one
                prefix, color = self._tag(old)
                row.setMaxLines(_SENTENCE_LINES)
                row.setColor(_OLD_COLOR)
                row.setText(old.dst or old.src, old.dst_lang if old.dst else old.src_lang,
                            ruby_wanted(cfg, bool(old.dst)), prefix, color)
                row.show()
            else:
                row.setText("")
                row.hide()
        self._fit()

    def _fade(self) -> None:
        self._entries.clear()
        self._render()

    def _sync_speaker(self) -> None:
        # A click-through bar never receives clicks, so the button would be dead: hide it instead.
        line = self._line
        show = self.cfg.learning and not self.cfg.click_through and line is not None and bool(line.dst or line.src)
        self._speaker.setVisible(show)
        self._balance.setVisible(show)

    def _speak(self) -> None:
        if self._line is not None:
            self.speak_requested.emit(self._line)

    def set_speaker_warning(self, text: str) -> None:
        """Quietly flag the speaker button as unusable (e.g. no voice installed); "" clears it."""
        self._speaker.set_warning(text)

    def reset_speakers(self) -> None:
        self._speakers.clear()
        self._render()

    # ── look & behaviour ─────────────────────────────────────────────────────

    def apply_style(self) -> None:
        big = QFont(self.font())
        big.setPointSize(max(10, self.cfg.font_size))
        big.setBold(True)
        small = QFont(self.font())
        small.setPointSize(max(8, int(self.cfg.font_size * 0.55)))
        self._dst.setFont(big)
        self._src.setFont(small)
        for h in self._hist:
            h.setFont(big)
        self._src.setColor("#b8c2cf")
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self.cfg.click_through)
        self._grip.setVisible(not self.cfg.click_through)
        self._render()                           # learning mode / click-through change what is shown
        if self.isVisible():
            self.show()                          # changing window flags hides the window

    def _content_height(self) -> int:
        lay = self._body.layout()
        self._col.invalidate()                   # layouts cache their height-for-width until the next event loop turn,
        lay.invalidate()                         # and invalidating the outer one does not reach the nested one
        return max(1, lay.totalHeightForWidth(self.width()))

    def _layout_body(self) -> None:
        h = self._content_height()
        self._body.setGeometry(0, self.height() - h, self.width(), h)     # bottom-aligned; overflow is clipped on top

    def _fit(self) -> None:
        """Automatic height: exactly as tall as the content, growing upwards (the bottom edge stays put) so the
        sentence you are reading does not jump. With a custom height the bar keeps it."""
        if self._fitting:
            return
        if self.cfg.auto_height:
            h = max(self.minimumHeight(), self._content_height())
            if h != self.height():
                self._fitting = True
                try:
                    self.setGeometry(self.x(), self.y() + self.height() - h, self.width(), h)
                finally:
                    self._fitting = False
        self._layout_body()

    def set_auto_height(self, on: bool) -> None:
        self.cfg.auto_height = on
        self._fit()
        self._persist()

    def _place(self) -> None:
        if self.cfg.x >= 0 and self.cfg.y >= 0:
            self.move(self.cfg.x, self.cfg.y + self.cfg.h - self.height())     # remembered bottom edge
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
        self._grip.move(self.width() - self._grip.width(), self.height() - self._grip.height())
        if not self._fitting and self._ready:
            auto = max(self.minimumHeight(), self._content_height())
            if self.cfg.auto_height and abs(self.height() - auto) > _CUSTOM_H_SLACK:
                self.cfg.auto_height = False     # the user dragged the corner vertically: keep the height they chose
            else:
                self._fit()                      # width only: the height follows the content
        self._layout_body()
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
        sentences = m.addMenu("同时显示句数")
        for n in _SENTENCE_CHOICES:
            act = sentences.addAction("只显示当前一句" if n == 1 else f"{n} 句")
            act.setCheckable(True)
            act.setChecked(self.cfg.max_sentences == n)
            act.triggered.connect(lambda _c=False, n=n: self.set_max_sentences(n))
        a = m.addAction("自动调整高度（关闭后可拖右下角自定义）")
        a.setCheckable(True)
        a.setChecked(self.cfg.auto_height)
        a.toggled.connect(self.set_auto_height)
        a = m.addAction("学习模式（注音 + 朗读）")
        a.setCheckable(True)
        a.setChecked(self.cfg.learning)
        a.toggled.connect(self._toggle_learning)
        m.addSeparator()
        m.addAction("关闭字幕窗", self._close)
        m.exec(e.globalPos())

    def _font(self, delta: int) -> None:
        self.cfg.font_size = max(12, min(72, self.cfg.font_size + delta))
        self.apply_style()
        self._persist()

    def _toggle_source(self, on: bool) -> None:
        self.cfg.show_source = on
        self._render()
        self._persist()

    def set_max_sentences(self, n: int) -> None:
        self.cfg.max_sentences = min(max(1, n), _MAX_SENTENCES)
        self._render()
        self._persist()

    def _toggle_learning(self, on: bool) -> None:
        self.set_learning(on)
        self.learning_changed.emit(on)

    def set_learning(self, on: bool) -> None:
        self.cfg.learning = on
        self._render()

    def _close(self) -> None:
        self.hide()
        self.closed.emit()
