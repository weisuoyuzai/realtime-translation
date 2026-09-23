from __future__ import annotations

import dataclasses
import math
import os
import platform
import sys
import threading

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QAction, QColor, QGuiApplication, QKeySequence
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenuBar, QMessageBox,
                               QPushButton, QScrollArea, QSplitter, QStatusBar, QVBoxLayout, QWidget)

from .. import __version__, ruby, runtime
from ..config import AppConfig, config_dir, save_config
from ..models import Line
from ..pipeline import Pipeline
from ..text_utils import guess_language
from . import theme
from .bridge import Bridge
from .overlay import SubtitleOverlay
from .preferences import TAB_STORAGE, PreferencesDialog, open_folder
from .ruby_text import speaker_color
from .settings_panel import SettingsPanel
from .titlebar import CUSTOM_FRAME, FramelessMixin, TitleBar
from .transcript import TranscriptView
from .tts import Tts
from .widgets import ElidedLabel, LevelMeter, PillToggle, icon_button, icon_label, label, vline

# status level -> (pill text, colour)
_DOT = {"idle": ("就绪", theme.TEXT_3), "loading": ("启动中", theme.WARN), "ready": ("识别中", theme.LIVE),
        "warn": ("注意", theme.WARN), "error": ("出错", theme.DANGER)}


def validate(cfg: AppConfig) -> str | None:
    """First problem that would make the pipeline fail at start, phrased for the user."""
    a, s, t = cfg.audio, cfg.asr, cfg.translate
    if a.source == "app" and not a.app_key:
        return "请选择要捕获的应用（点「刷新」重新扫描）。"
    if a.source == "file" and not os.path.isfile(a.file_path):
        return "找不到音频文件。"
    if s.mode == "remote" and not (s.remote_base_url and s.remote_model):
        return "远程识别需要填写 API 地址和模型名。"
    if t.mode in ("llm_remote", "llm_local"):
        ep = t.remote if t.mode == "llm_remote" else t.local
        if not (ep.base_url and ep.model):
            return "翻译需要填写 API 地址和模型名（本地服务可点「获取已安装的模型列表」）。"
        if t.mode == "llm_remote" and not ep.api_key and "127.0.0.1" not in ep.base_url and "localhost" not in ep.base_url:
            return "远程翻译 API 需要填写 API Key。"
    if t.mode == "deepl" and not t.deepl_key:
        return "请填写 DeepL API Key。"
    return None


class MainWindow(FramelessMixin, QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.setWindowTitle(f"Live Translator {__version__} — 实时音频翻译")
        self.resize(1180, 780)
        self.cfg = cfg
        self.bridge = Bridge()
        self.pipeline: Pipeline | None = None
        self.state = "idle"                     # idle | starting | running | stopping
        self._latencies: list[float] = []
        self._counted: set[int] = set()
        self._level = 0.0

        self.overlay = SubtitleOverlay(cfg.overlay)
        self.overlay.closed.connect(lambda: self.overlay_cb.setChecked(False))
        self.tts = Tts(self)
        self._voice_warned: set[str] = set()                  # languages already announced as having no voice
        self.tts.unavailable.connect(self._on_voice_missing)
        self.tts.speaking_changed.connect(lambda on: on and self.overlay.set_speaker_warning(""))

        theme.apply(QApplication.instance())

        # ── window title bar (frameless on Windows / Linux; macOS keeps its native frame) ──
        self._frameless = CUSTOM_FRAME
        self.titlebar: TitleBar | None = None
        if self._frameless:
            self._menubar = QMenuBar()
            self.titlebar = TitleBar(self, self._menubar, "Live Translator")
            self.setMenuWidget(self.titlebar)
            self._init_frame()

        # ── top bar ──
        self.btn = QPushButton("开始")
        self.btn.setObjectName("startBtn")
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setMinimumWidth(92)
        self.btn.clicked.connect(self.toggle)
        self._set_btn(False)
        self.pill = QFrame()
        self.pill.setObjectName("statusPill")
        pl = QHBoxLayout(self.pill)
        pl.setContentsMargins(8, 3, 10, 3)
        pl.setSpacing(6)
        self.dot = QLabel("●")
        self.pill_text = QLabel()
        pl.addWidget(self.dot)
        pl.addWidget(self.pill_text)
        self.status = ElidedLabel("就绪。请在左侧选择音频来源、语言和引擎，然后点「开始」。")
        self.status.setObjectName("muted")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meter = LevelMeter()
        self.latency = QLabel("—")
        self.latency.setFont(theme.mono_font(13, 600))
        self.latency.setAlignment(Qt.AlignmentFlag.AlignRight)
        lat_cap = label("平均延迟", px=10, color=theme.TEXT_3)
        lat = QVBoxLayout()
        lat.setSpacing(0)
        lat.addWidget(self.latency, 0, Qt.AlignmentFlag.AlignRight)
        lat.addWidget(lat_cap, 0, Qt.AlignmentFlag.AlignRight)
        self.overlay_cb = PillToggle("overlay", "悬浮字幕", "在所有窗口上方显示字幕条（可拖动 / 右下角缩放 / 右键菜单）")
        self.overlay_cb.setChecked(cfg.overlay.enabled)
        self.overlay_cb.toggled.connect(self._toggle_overlay)
        self.click_cb = PillToggle("pointer-off", "穿透", "鼠标穿透：开启后字幕条不拦截鼠标，可以直接点击下面的窗口")
        self.click_cb.setChecked(cfg.overlay.click_through)
        self.click_cb.toggled.connect(self._toggle_click_through)
        self.learn_cb = PillToggle("learn", "学习", "学习模式：汉字上方标注拼音，并在字幕前显示朗读按钮（点一下朗读这句，再点一下停止）")
        self.learn_cb.setChecked(cfg.overlay.learning)
        self.learn_cb.toggled.connect(self._toggle_learning)
        pills = QFrame()
        pills.setObjectName("pillGroup")
        pg = QHBoxLayout(pills)
        pg.setContentsMargins(3, 3, 3, 3)
        pg.setSpacing(4)
        for b in (self.overlay_cb, self.click_cb, self.learn_cb):
            pg.addWidget(b)
        copy = icon_button("copy", "复制全部字幕")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.transcript.plain_text()))
        export = icon_button("download", "导出字幕…（Ctrl+S）")
        export.clicked.connect(self._export)
        clear = icon_button("trash", "清空字幕（Ctrl+L）")
        clear.clicked.connect(self._clear)
        prefs = icon_button("sliders", "偏好设置…（Ctrl+,）")
        prefs.clicked.connect(lambda: self._open_prefs())

        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(56)
        top = QHBoxLayout(bar)
        top.setContentsMargins(16, 0, 14, 0)
        top.setSpacing(12)
        top.addWidget(self.btn)
        top.addWidget(self.pill, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.status, 1)
        top.addWidget(self.meter)
        lat_box = QWidget()
        lat_box.setLayout(lat)
        lat.setContentsMargins(0, 0, 0, 0)
        top.addWidget(lat_box, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(vline())
        top.addWidget(pills, 0, Qt.AlignmentFlag.AlignVCenter)
        tools = QHBoxLayout()
        tools.setSpacing(2)
        for b in (copy, export, clear, prefs):
            tools.addWidget(b)
        top.addLayout(tools)

        # ── sidebar: the five pipeline steps ──
        self.panel = SettingsPanel(self.bridge)
        self.panel.load(cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.panel)
        side = QFrame()
        side.setObjectName("sidebar")
        side.setMinimumWidth(330)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 14, 8, 12)
        sl.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 6, 0)
        head.addWidget(label("翻译管线", "section"))
        head.addStretch(1)
        self.lock_note = QWidget()
        ll = QHBoxLayout(self.lock_note)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        ll.addWidget(icon_label("lock", theme.TEXT_3, 12))
        ll.addWidget(label("运行中不可修改，停止后再改", px=11, color=theme.TEXT_3))
        self.lock_note.hide()
        head.addWidget(self.lock_note)
        sl.addLayout(head)
        sl.addWidget(scroll, 1)
        foot = QFrame()
        foot.setObjectName("footNote")
        foot.setCursor(Qt.CursorShape.PointingHandCursor)
        foot.setToolTip("点击打开字幕记录文件夹")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(12, 9, 12, 9)
        fl.setSpacing(8)
        fl.addWidget(icon_label("drive", theme.TEXT_3, 14))
        self.tx_path = ElidedLabel()
        self.tx_path.setStyleSheet(f"color:{theme.TEXT_3};font-size:11px;")
        fl.addWidget(self.tx_path, 1)
        foot.mousePressEvent = lambda _e: open_folder(self._transcripts_path())
        sl.addWidget(foot)

        # ── subtitles ──
        self.transcript = TranscriptView()
        self.transcript.set_learning(cfg.overlay.learning, cfg.overlay.ruby_scope)
        self.transcript.speak_requested.connect(self._speak_line)
        self.overlay.speak_requested.connect(self._speak_line)
        self.overlay.learning_changed.connect(self.learn_cb.setChecked)
        tr_head = QFrame()
        tr_head.setObjectName("trHeader")
        th = QHBoxLayout(tr_head)
        th.setContentsMargins(28, 12, 28, 12)
        th.setSpacing(10)
        th.addWidget(label("字幕", px=15, weight=600))
        self.count_lbl = QLabel("")
        self.count_lbl.setFont(theme.mono_font(12))
        self.count_lbl.setStyleSheet(f"color:{theme.TEXT_3};")
        th.addWidget(self.count_lbl)
        th.addStretch(1)
        self.legend = QHBoxLayout()
        self.legend.setSpacing(14)
        th.addLayout(self.legend)
        self._legend_seen: set[int] = set()
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(tr_head)
        rl.addWidget(self.transcript, 1)

        split = QSplitter()
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([350, 830])

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(bar)
        rl.addWidget(split, 1)
        self.setCentralWidget(root)
        self._build_menus()

        # ── status bar ──
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        gpu = runtime.gpu_name() if runtime.cuda_available() else ""
        self.device_lbl = self._sb_item(sb, "cpu", f"CUDA · {gpu}" if gpu else "CPU")
        self.timing_lbl = self._sb_item(sb, "timer", "")
        self.spk_lbl = self._sb_item(sb, "users", "")
        self.stats = QLabel("")
        sb.addPermanentWidget(self.stats)
        ver = QLabel(f"v{__version__}")
        ver.setFont(theme.mono_font(11))
        sb.addPermanentWidget(ver)
        self._update_tx_path()

        b = self.bridge
        b.line.connect(self._on_line)
        b.status.connect(self._on_status)
        b.level.connect(self._on_level)
        b.stopped.connect(self._on_stopped)
        self._set_dot("idle")
        if cfg.overlay.enabled:
            self.overlay.show()

    # ── window frame ─────────────────────────────────────────────────────────

    def menuBar(self) -> QMenuBar:
        """With the custom title bar the menus live inside it rather than in QMainWindow's menu-bar slot."""
        return self._menubar if self._frameless else super().menuBar()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if self._frameless and e.type() in (QEvent.Type.WindowStateChange, QEvent.Type.ActivationChange):
            self._update_margins()
            if self.titlebar is not None:
                self.titlebar.sync()
            self.update()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._frameless:
            self._round_corners()

    # ── menus / preferences ──────────────────────────────────────────────────

    def _build_menus(self) -> None:
        mb = self.menuBar()

        def act(menu, text, slot=None, shortcut=None, checkable=False, checked=False) -> QAction:
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            if checkable:
                a.setChecked(checked)
            if slot:
                (a.toggled if checkable else a.triggered).connect(slot)
            menu.addAction(a)
            return a

        m = mb.addMenu("文件(&F)")
        act(m, "导出字幕…", self._export, "Ctrl+S")
        act(m, "复制全部字幕", lambda: QGuiApplication.clipboard().setText(self.transcript.plain_text()))
        act(m, "清空字幕", self._clear, "Ctrl+L")
        self.act_reset_spk = act(m, "重置说话人编号（换了一批人时用）", self._reset_speakers)
        m.addSeparator()
        act(m, "打开字幕记录文件夹", lambda: open_folder(self._transcripts_path()))
        act(m, "打开配置文件夹", lambda: open_folder(str(config_dir())))
        m.addSeparator()
        act(m, "退出", self.close, "Ctrl+Q")

        m = mb.addMenu("设置(&S)")
        act(m, "偏好设置…", lambda: self._open_prefs(), "Ctrl+,")
        act(m, "模型管理（存放位置 / 删除）…", lambda: self._open_prefs(TAB_STORAGE))
        m.addSeparator()
        act(m, "打开模型文件夹", lambda: open_folder(runtime.effective_models_dir()))

        m = mb.addMenu("视图(&V)")
        self.act_overlay = act(m, "悬浮字幕", self.overlay_cb.setChecked, "Ctrl+Shift+O", True, self.overlay_cb.isChecked())
        self.act_click = act(m, "字幕条鼠标穿透", self.click_cb.setChecked, None, True, self.click_cb.isChecked())
        self.act_source = act(m, "悬浮字幕显示原文", self.overlay._toggle_source, None, True, self.overlay.cfg.show_source)
        self.act_learn = act(m, "学习模式（拼音 + 朗读）", self.learn_cb.setChecked, None, True, self.learn_cb.isChecked())
        self.overlay_cb.toggled.connect(self.act_overlay.setChecked)      # keep the toolbar and the menu in step
        self.click_cb.toggled.connect(self.act_click.setChecked)
        self.learn_cb.toggled.connect(self.act_learn.setChecked)

        m = mb.addMenu("帮助(&H)")
        act(m, "关于 Live Translator", self._about)

    def _transcripts_path(self) -> str:
        return self.current_config().transcripts_dir or os.path.join(str(config_dir()), "transcripts")

    def _update_tx_path(self) -> None:
        on = self.panel.save_tx.isChecked()
        self.tx_path.setText(f"字幕记录{'保存到' if on else '（未开启保存）'} {self._transcripts_path()}")

    @staticmethod
    def _sb_item(sb: QStatusBar, icon: str, text: str) -> QLabel:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 0, 8, 0)
        h.setSpacing(6)
        h.addWidget(icon_label(icon, theme.TEXT_3, 12))
        lbl = QLabel(text)
        lbl.setStyleSheet("padding:0")
        h.addWidget(lbl)
        w.setVisible(bool(text))
        sb.addWidget(w)
        lbl.box = w
        return lbl

    @staticmethod
    def _sb_set(lbl: QLabel, text: str) -> None:
        lbl.setText(text)
        lbl.box.setVisible(bool(text))

    def _set_btn(self, running: bool) -> None:
        self.btn.setText("停止" if running else "开始")
        self.btn.setIcon(theme.icon("stop" if running else "play", theme.DANGER if running else theme.ON_ACCENT))
        self.btn.setProperty("running", "true" if running else "false")
        theme.repolish(self.btn)

    def _open_prefs(self, tab: int = 0) -> None:
        dlg = PreferencesDialog(self.current_config(), running=self.state != "idle", parent=self, tab=tab)
        dlg.applied.connect(self.apply_preferences)
        if dlg.exec() == PreferencesDialog.DialogCode.Accepted:
            self.apply_preferences(dlg.result_config())
            if dlg.restart_requested:
                self._restart()

    def _restart(self) -> None:
        """Start a fresh copy of the program (same interpreter and arguments), then close this one."""
        args = sys.argv[1:] if getattr(sys, "frozen", False) else list(getattr(sys, "orig_argv", sys.argv)[1:])
        if QProcess.startDetached(sys.executable, args)[0]:
            self.close()
        else:
            QMessageBox.warning(self, "无法自动重启", "请手动关闭并重新打开本程序，让新的网络设置生效。")

    def apply_preferences(self, new: AppConfig) -> None:
        oc = self.overlay.cfg
        for f in dataclasses.fields(oc):
            setattr(oc, f.name, getattr(new.overlay, f.name))
        self.overlay.apply_style()
        self.overlay.update()
        self.click_cb.setChecked(oc.click_through)
        self.learn_cb.setChecked(oc.learning)
        self.transcript.set_learning(oc.learning, oc.ruby_scope)
        self.act_source.blockSignals(True)
        self.act_source.setChecked(oc.show_source)
        self.act_source.blockSignals(False)
        self.panel.apply_general(new)
        runtime.apply_settings(new)
        self._update_tx_path()
        try:
            save_config(self.current_config())
        except OSError:
            pass
        later = "；识别与断句参数将在下次点「开始」时生效" if self.state != "idle" else ""
        self.statusBar().showMessage(f"设置已保存{later}", 6000)

    def _about(self) -> None:
        gpu = runtime.gpu_name()
        cfg = self.current_config()
        QMessageBox.about(self, "关于 Live Translator", (
            f"<h3>Live Translator {__version__}</h3>"
            "<p>实时音频翻译：按应用捕获声音，本地或远程识别，本地或远程翻译。</p>"
            f"<p>识别设备：{('NVIDIA ' + gpu + '（CUDA 可用）') if runtime.cuda_available() and gpu else 'CPU'}<br>"
            f"系统：{platform.platform()}<br>Python {sys.version.split()[0]}</p>"
            f"<p>模型目录：{runtime.effective_models_dir()}<br>"
            f"字幕记录：{self._transcripts_path()}<br>配置目录：{config_dir()}</p>"))

    # ── run control ──────────────────────────────────────────────────────────

    def toggle(self) -> None:
        if self.state == "idle":
            self._start()
        elif self.state in ("starting", "running"):
            self._stop()

    def current_config(self) -> AppConfig:
        cfg = self.panel.collect()
        cfg.overlay = self.overlay.cfg
        cfg.overlay.enabled = self.overlay_cb.isChecked()
        cfg.overlay.click_through = self.click_cb.isChecked()
        cfg.overlay.learning = self.learn_cb.isChecked()
        return cfg

    def _start(self) -> None:
        cfg = self.current_config()
        problem = validate(cfg)
        if problem:
            self._reveal_problem(cfg)
            QMessageBox.warning(self, "还不能开始", problem)
            return
        self.cfg = cfg
        try:
            save_config(cfg)
        except OSError:
            pass
        if cfg.overlay.learning:
            self._check_voice()
        self._latencies.clear()
        self._counted.clear()
        self.state = "starting"
        self._set_btn(True)
        self.panel.set_running(True)
        self.lock_note.show()
        self._reset_legend()
        self._update_tx_path()
        self.pipeline = Pipeline(cfg, self.bridge.on_line, self.bridge.on_status, self.bridge.on_level)
        self.overlay.reset_speakers()
        self.pipeline.start()
        self._on_status("loading", "正在启动…")

    def _stop(self) -> None:
        pipe, self.pipeline = self.pipeline, None
        if pipe is None:
            return
        self.state = "stopping"
        self.btn.setEnabled(False)
        self._on_status("loading", "正在停止…")

        def work():
            pipe.stop()
            self.bridge.stopped.emit()

        threading.Thread(target=work, daemon=True).start()

    def _on_stopped(self) -> None:
        self.pipeline = None
        self.state = "idle"
        self.btn.setEnabled(True)
        self._set_btn(False)
        self.panel.set_running(False)
        self.lock_note.hide()
        self.meter.setValue(0)
        self._level = 0.0
        if self.status.text() in ("正在停止…", "正在启动…"):
            self._on_status("idle", "已停止。")

    # ── events ───────────────────────────────────────────────────────────────

    def _on_status(self, level: str, msg: str) -> None:
        if level == "stopped":                       # pipeline gave up during start-up
            self._on_stopped()
            return
        was_starting = self.state == "starting"
        self._set_dot(level if level in _DOT else "idle")
        if msg:
            self.status.setText(msg)
        if level == "ready":
            self.state = "running"
        elif level == "error" and was_starting:
            QMessageBox.critical(self, "启动失败", msg)

    def _set_dot(self, level: str) -> None:
        text, color = _DOT[level]
        if level == "loading" and self.state == "stopping":
            text = "停止中"
        c = QColor(color)
        self.pill.setStyleSheet(f"QFrame#statusPill{{background:rgba({c.red()},{c.green()},{c.blue()},0.13);"
                                f"border:none;border-radius:11px;}}")
        self.dot.setStyleSheet(f"color:{color};font-size:9px;")
        self.pill_text.setStyleSheet(f"color:{color};font-size:12px;font-weight:600;")
        self.pill_text.setText(text)

    def _reveal_problem(self, cfg: AppConfig) -> None:
        """Open the step whose settings stop the start, so the user lands right on the field to fix."""
        a, s = cfg.audio, cfg.asr
        if (a.source == "app" and not a.app_key) or (a.source == "file" and not os.path.isfile(a.file_path)):
            self.panel.reveal(0)
        elif s.mode == "remote" and not (s.remote_base_url and s.remote_model):
            self.panel.reveal(2)
        else:
            self.panel.reveal(3)

    def _reset_legend(self) -> None:
        self._legend_seen.clear()
        while self.legend.count():
            w = self.legend.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._sb_set(self.spk_lbl, "")

    def _add_legend(self, n: int) -> None:
        self._legend_seen.add(n)
        lbl = QLabel(f'<span style="color:{speaker_color(n)}">●</span>&nbsp; 说话人 {n}')
        lbl.setStyleSheet(f"color:{theme.TEXT_2};font-size:12px;")
        self.legend.addWidget(lbl)
        self._sb_set(self.spk_lbl, f"{len(self._legend_seen)} 位说话人")

    def _on_line(self, line: Line) -> None:
        self.transcript.upsert(line)
        if self.overlay_cb.isChecked():
            self.overlay.show_line(line)
        if line.speaker and line.speaker not in self._legend_seen:
            self._add_legend(line.speaker)
        n = self.transcript.count()
        self.count_lbl.setText(f"{n} 句" if n else "")
        if line.dst_final and line.latency_ms and line.id not in self._counted:
            self._counted.add(line.id)
            self._latencies.append(line.latency_ms)
            recent = self._latencies[-10:]
            avg = sum(recent) / len(recent) / 1000
            self.stats.setText(f"共 {len(self._latencies)} 句 · 最近 10 句平均延迟 {avg:.1f}s "
                               f"（从说完到译文出现） ")
            self.latency.setText(f"{avg:.2f} s")
            self._sb_set(self.timing_lbl, f"识别 {line.asr_ms / 1000:.2f} s" + (f" · 翻译 {line.tr_ms / 1000:.2f} s" if line.tr_ms else ""))

    def _on_level(self, rms: float) -> None:
        db = 20 * math.log10(max(rms, 1e-5))
        v = max(0.0, min(1.0, (db + 60) / 60)) * 100
        self._level = max(v, self._level * 0.8)          # fast attack, slow release
        self.meter.setValue(int(self._level))

    # ── misc ─────────────────────────────────────────────────────────────────

    def _toggle_overlay(self, on: bool) -> None:
        self.overlay.cfg.enabled = on
        if on:
            self.overlay.show()
        else:
            self.overlay.hide()

    def _toggle_click_through(self, on: bool) -> None:
        self.overlay.cfg.click_through = on
        self.overlay.apply_style()

    def _reset_speakers(self) -> None:
        self.overlay.reset_speakers()
        if self.pipeline is not None:
            self.pipeline.reset_speakers()
            self.statusBar().showMessage("已重置说话人编号：之后的发言从 1 号重新排。", 6000)
        else:
            self.statusBar().showMessage("说话人编号在每次点「开始」时自动重置。", 6000)

    def _toggle_learning(self, on: bool) -> None:
        self.overlay.set_learning(on)
        self.transcript.set_learning(on, self.overlay.cfg.ruby_scope)
        if not on:
            self.tts.stop()
            return
        if ruby.missing_packages():
            names = " ".join(ruby.missing_packages())
            self.statusBar().showMessage(f"未安装 {names}，对应语言无法注音（pip install {names}）；朗读功能不受影响。", 8000)
        self._check_voice()

    def _check_voice(self) -> None:
        """Learning mode is on: tell the user now (not at the first click) if the language it will read has no voice."""
        lang = (self.panel.tgt_lang if self.overlay.cfg.tts_read == "dst" else self.panel.src_lang).currentData()
        if lang and lang != "auto":
            problem = self.tts.voice_problem(lang)
            if problem:
                self._on_voice_missing(*problem)

    def _on_voice_missing(self, key: str, message: str) -> None:
        """Three quiet-to-loud signals: the speaker icon gets crossed out (tooltip explains), the status bar shows
        it, and — once per language per session — a non-blocking dialog says how to fix it."""
        self.statusBar().showMessage(message.replace("\n", " "), 12000)
        self.overlay.set_speaker_warning(message.split("\n")[0])
        if key not in self._voice_warned:
            self._voice_warned.add(key)
            self._show_voice_notice(message)

    def _show_voice_notice(self, message: str) -> None:
        box = QMessageBox(QMessageBox.Icon.Information, "没有可用的朗读语音", message, QMessageBox.StandardButton.Ok, self)
        box.setInformativeText("其他功能（包括注音）不受影响。")
        box.setWindowModality(Qt.WindowModality.NonModal)         # never blocks the live subtitles
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()

    def _speak_line(self, line: Line | None) -> None:
        """Read a subtitle aloud: the translation or the original, per the preference (falls back to whichever exists)."""
        if line is None:
            return
        dst = (line.dst, line.dst_lang)
        src = (line.src, line.src_lang)
        for text, lang in (src, dst) if self.overlay.cfg.tts_read == "src" else (dst, src):
            if text.strip():
                self.tts.toggle(text, lang or guess_language(text))
                return

    def _clear(self) -> None:
        self.transcript.clear_all()
        self._latencies.clear()
        self.stats.setText("")
        self.count_lbl.setText("")
        self.latency.setText("—")
        self._sb_set(self.timing_lbl, "")
        self._reset_legend()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出字幕", "transcript.txt", "文本文件 (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.transcript.plain_text())

    def closeEvent(self, e) -> None:
        pipe, self.pipeline = self.pipeline, None
        if pipe is not None:
            pipe.stop()
        try:
            save_config(self.current_config())
        except OSError:
            pass
        self.overlay.close()
        super().closeEvent(e)
