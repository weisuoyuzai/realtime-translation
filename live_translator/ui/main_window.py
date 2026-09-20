from __future__ import annotations

import dataclasses
import math
import os
import platform
import sys
import threading

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

from .. import __version__, runtime
from ..config import AppConfig, config_dir, save_config
from ..models import Line
from ..pipeline import Pipeline
from .bridge import Bridge
from .overlay import SubtitleOverlay
from .preferences import TAB_STORAGE, PreferencesDialog, open_folder
from .settings_panel import SettingsPanel
from .transcript import TranscriptView

_DOT = {"idle": "#8b95a1", "loading": "#e0a800", "ready": "#2a9d4a", "warn": "#e08a00", "error": "#d33"}


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


class MainWindow(QMainWindow):
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

        # ── top bar ──
        self.btn = QPushButton("▶  开始")
        self.btn.setMinimumHeight(38)
        self.btn.setMinimumWidth(120)
        self.btn.clicked.connect(self.toggle)
        self.dot = QLabel("●")
        self.status = QLabel("就绪。请在左侧选择音频来源、语言和引擎，然后点「开始」。")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedSize(90, 8)
        self.overlay_cb = QCheckBox("悬浮字幕")
        self.overlay_cb.setToolTip("在所有窗口上方显示字幕条（可拖动 / 右下角缩放 / 右键菜单）")
        self.overlay_cb.setChecked(cfg.overlay.enabled)
        self.overlay_cb.toggled.connect(self._toggle_overlay)
        self.click_cb = QCheckBox("鼠标穿透")
        self.click_cb.setToolTip("开启后字幕条不拦截鼠标，可以直接点击下面的窗口")
        self.click_cb.setChecked(cfg.overlay.click_through)
        self.click_cb.toggled.connect(self._toggle_click_through)
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear)
        copy = QPushButton("复制")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.transcript.plain_text()))
        export = QPushButton("导出…")
        export.clicked.connect(self._export)

        top = QHBoxLayout()
        top.setContentsMargins(12, 10, 12, 6)
        top.addWidget(self.btn)
        top.addSpacing(8)
        top.addWidget(self.dot)
        top.addWidget(self.status, 1)
        top.addWidget(self.meter)
        top.addSpacing(8)
        top.addWidget(self.overlay_cb)
        top.addWidget(self.click_cb)
        top.addSpacing(8)
        for b in (clear, copy, export):
            top.addWidget(b)

        # ── body ──
        self.panel = SettingsPanel(self.bridge)
        self.panel.load(cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.panel)
        scroll.setMinimumWidth(390)
        self.transcript = TranscriptView()
        self.transcript.setStyleSheet("QTextBrowser{padding:10px 14px;border:none;}")
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.transcript)
        split.setStretchFactor(1, 1)
        split.setSizes([410, 770])

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addLayout(top)
        rl.addWidget(split, 1)
        self.setCentralWidget(root)
        self._build_menus()
        self.stats = QLabel("")
        self.statusBar().addPermanentWidget(self.stats)
        self.statusBar().showMessage(f"配置与字幕记录：{config_dir()}")

        b = self.bridge
        b.line.connect(self._on_line)
        b.status.connect(self._on_status)
        b.level.connect(self._on_level)
        b.stopped.connect(self._on_stopped)
        self._set_dot("idle")
        if cfg.overlay.enabled:
            self.overlay.show()

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
        self.overlay_cb.toggled.connect(self.act_overlay.setChecked)      # keep the toolbar and the menu in step
        self.click_cb.toggled.connect(self.act_click.setChecked)

        m = mb.addMenu("帮助(&H)")
        act(m, "关于 Live Translator", self._about)

    def _transcripts_path(self) -> str:
        return self.current_config().transcripts_dir or os.path.join(str(config_dir()), "transcripts")

    def _open_prefs(self, tab: int = 0) -> None:
        dlg = PreferencesDialog(self.current_config(), running=self.state != "idle", parent=self, tab=tab)
        if dlg.exec() == PreferencesDialog.DialogCode.Accepted:
            self.apply_preferences(dlg.result_config())

    def apply_preferences(self, new: AppConfig) -> None:
        oc = self.overlay.cfg
        for f in dataclasses.fields(oc):
            setattr(oc, f.name, getattr(new.overlay, f.name))
        self.overlay.apply_style()
        self.overlay.update()
        self.click_cb.setChecked(oc.click_through)
        self.act_source.blockSignals(True)
        self.act_source.setChecked(oc.show_source)
        self.act_source.blockSignals(False)
        self.panel.apply_general(new)
        runtime.apply_settings(new)
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
        return cfg

    def _start(self) -> None:
        cfg = self.current_config()
        problem = validate(cfg)
        if problem:
            QMessageBox.warning(self, "还不能开始", problem)
            return
        self.cfg = cfg
        try:
            save_config(cfg)
        except OSError:
            pass
        self._latencies.clear()
        self._counted.clear()
        self.state = "starting"
        self.btn.setText("■  停止")
        self.panel.set_running(True)
        self.pipeline = Pipeline(cfg, self.bridge.on_line, self.bridge.on_status, self.bridge.on_level)
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
        self.btn.setText("▶  开始")
        self.panel.set_running(False)
        self.meter.setValue(0)
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
        self.dot.setStyleSheet(f"color:{_DOT[level]};font-size:16px;")

    def _on_line(self, line: Line) -> None:
        self.transcript.upsert(line)
        if self.overlay_cb.isChecked():
            self.overlay.show_line(line)
        if line.dst_final and line.latency_ms and line.id not in self._counted:
            self._counted.add(line.id)
            self._latencies.append(line.latency_ms)
            recent = self._latencies[-10:]
            self.stats.setText(f"共 {len(self._latencies)} 句 · 最近 10 句平均延迟 {sum(recent) / len(recent) / 1000:.1f}s "
                               f"（从说完到译文出现） ")

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

    def _clear(self) -> None:
        self.transcript.clear_all()
        self._latencies.clear()
        self.stats.setText("")

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
