"""Preferences dialog: storage locations + model manager, network, subtitle overlay, advanced tuning."""
from __future__ import annotations

import copy
import os
import threading
import time

from PySide6.QtCore import QEventLoop, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressDialog,
                               QPushButton, QSlider, QSpinBox, QTabWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import runtime, storage
from ..config import AppConfig, AsrCfg, SegmenterCfg
from .settings_panel import _form, _hint

TAB_STORAGE, TAB_NETWORK, TAB_OVERLAY, TAB_ADVANCED = range(4)

MIRRORS = [("官方 huggingface.co", ""), ("hf-mirror.com（国内镜像）", "https://hf-mirror.com")]
COMPUTE_TYPES = [("自动", "auto"), ("float16（GPU）", "float16"), ("int8_float16（GPU，省显存）", "int8_float16"),
                 ("int8（CPU，更快）", "int8"), ("float32", "float32")]


def open_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class _Signals(QObject):
    moved = Signal(object, str)          # moved repo ids, error message
    step = Signal(str)                   # progress text while moving
    net = Signal(bool, str)              # ok, message


class PreferencesDialog(QDialog):
    def __init__(self, cfg: AppConfig, running: bool = False, parent=None, tab: int = TAB_STORAGE):
        super().__init__(parent)
        self.setWindowTitle("偏好设置")
        self.setMinimumWidth(640)
        self._orig = copy.deepcopy(cfg)
        self._running = running
        self._sig = _Signals()
        self._sig.net.connect(self._show_net)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_storage(), "存储")
        self.tabs.addTab(self._build_network(), "网络")
        self.tabs.addTab(self._build_overlay(), "悬浮字幕")
        self.tabs.addTab(self._build_advanced(), "高级")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs)
        lay.addWidget(self.buttons)

        self._load(cfg)
        self.tabs.setCurrentIndex(tab)

    # ── storage ──────────────────────────────────────────────────────────────

    def _dir_row(self, edit: QLineEdit, placeholder: str, browse_title: str) -> QWidget:
        edit.setPlaceholderText(placeholder)
        browse, reset, opn = QPushButton("浏览…"), QPushButton("恢复默认"), QPushButton("打开")
        browse.clicked.connect(lambda: self._browse(edit, browse_title))
        reset.clicked.connect(edit.clear)
        opn.clicked.connect(lambda: open_folder(edit.text().strip() or edit.placeholderText()))
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        for b in (browse, reset, opn):
            h.addWidget(b)
        return w

    def _browse(self, edit: QLineEdit, title: str) -> None:
        start = edit.text().strip() or edit.placeholderText()
        path = QFileDialog.getExistingDirectory(self, title, start if os.path.isdir(start) else os.path.expanduser("~"))
        if path:
            edit.setText(os.path.normpath(path))

    def _build_storage(self) -> QWidget:
        self.models_dir = QLineEdit()
        self.transcripts_dir = QLineEdit()
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["已下载的模型", "大小"])
        self.models_tree.setRootIsDecorated(False)
        self.models_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.models_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.models_tree.setMinimumHeight(150)
        self.models_total = QLabel()
        delete = QPushButton("删除所选")
        delete.clicked.connect(self._delete_selected)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_models)
        row = QHBoxLayout()
        row.addWidget(self.models_total, 1)
        row.addWidget(refresh)
        row.addWidget(delete)
        self.storage_note = QLabel("识别正在运行：请先停止，再修改存储位置或删除模型。")
        self.storage_note.setStyleSheet("color:#c77700")
        self.storage_note.setVisible(self._running)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self.storage_note)
        lay.addWidget(QLabel("<b>本地模型存放位置</b>"))
        lay.addWidget(self._dir_row(self.models_dir, runtime.default_models_dir(), "选择模型存放目录"))
        lay.addWidget(_hint("faster-whisper 识别模型和内置离线翻译（NLLB）都下载到这里。更换位置后，如果旧位置里已有模型，"
                            "保存时可以选择一并移动。留空 = 使用 Hugging Face 默认缓存目录。"))
        lay.addWidget(self.models_tree, 1)
        lay.addLayout(row)
        lay.addSpacing(8)
        lay.addWidget(QLabel("<b>字幕记录保存位置</b>"))
        lay.addWidget(self._dir_row(self.transcripts_dir, os.path.join(str(self._config_dir()), "transcripts"),
                                    "选择字幕记录目录"))
        self.models_dir.textChanged.connect(self._refresh_models)
        self._storage_widgets = (self.models_dir, self.transcripts_dir, delete)
        return w

    @staticmethod
    def _config_dir():
        from ..config import config_dir
        return config_dir()

    def _shown_models_dir(self) -> str:
        return self.models_dir.text().strip() or runtime.default_models_dir()

    def _refresh_models(self) -> None:
        self.models_tree.clear()
        models = storage.list_models(self._shown_models_dir())
        for m in models:
            item = QTreeWidgetItem([m.repo_id, storage.human_size(m.size)])
            item.setData(0, Qt.ItemDataRole.UserRole, m)
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.models_tree.addTopLevelItem(item)
        total = sum(m.size for m in models)
        self.models_total.setText(f"共 {len(models)} 个模型，占用 {storage.human_size(total)}" if models else "该目录下还没有模型")

    def _delete_selected(self) -> None:
        items = self.models_tree.selectedItems()
        if not items:
            return
        entries = [i.data(0, Qt.ItemDataRole.UserRole) for i in items]
        names = "\n".join(f"· {e.repo_id}（{storage.human_size(e.size)}）" for e in entries)
        if QMessageBox.question(self, "删除模型", f"确定删除以下模型吗？之后使用时需要重新下载。\n\n{names}") \
                != QMessageBox.StandardButton.Yes:
            return
        errors = []
        for e in entries:
            try:
                storage.delete_model(e)
            except OSError as ex:
                errors.append(f"{e.repo_id}：{ex.strerror or ex}")
        if errors:
            QMessageBox.warning(self, "部分模型未能删除", "\n".join(errors))
        self._refresh_models()

    # ── network ──────────────────────────────────────────────────────────────

    def _build_network(self) -> QWidget:
        self.proxy = QLineEdit()
        self.proxy.setPlaceholderText("留空 = 不使用代理；例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080")
        self.mirror = QComboBox()
        self.mirror.setEditable(True)
        for label, url in MIRRORS:
            self.mirror.addItem(label, url)
        self.net_test = QPushButton("测试模型下载站连接")
        self.net_test.clicked.connect(self._test_network)
        self.net_result = QLabel()
        self.net_result.setWordWrap(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_form(("网络代理", self.proxy), ("模型下载站", self.mirror)))
        lay.addWidget(_hint("代理用于远程 API 调用和模型下载；本机地址（Ollama 等）不会走代理。"
                            "国内访问 huggingface.co 较慢时，可把下载站改为镜像。"))
        lay.addWidget(self.net_test)
        lay.addWidget(self.net_result)
        note = QLabel("⚠ 代理与下载站在重启程序后完全生效（模型下载组件只在启动时读取它们）。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#c77700")
        lay.addWidget(note)
        lay.addStretch(1)
        return w

    def _mirror_value(self) -> str:
        i = self.mirror.currentIndex()
        if i >= 0 and self.mirror.currentText() == self.mirror.itemText(i):
            return str(self.mirror.itemData(i) or "")
        return self.mirror.currentText().strip().rstrip("/")

    def _test_network(self) -> None:
        import httpx
        base = self._mirror_value() or "https://huggingface.co"
        proxy = self.proxy.text().strip() or None
        self.net_test.setEnabled(False)
        self.net_result.setText("测试中…")
        self.net_result.setStyleSheet("")

        def work():
            t0 = time.perf_counter()
            try:
                with httpx.Client(proxy=proxy, timeout=12, follow_redirects=True) as c:
                    r = c.get(f"{base}/api/models/Systran/faster-whisper-small")
                ms = (time.perf_counter() - t0) * 1000
                self._sig.net.emit(r.status_code < 400, f"{base} → HTTP {r.status_code}，耗时 {ms:.0f} ms")
            except Exception as e:
                self._sig.net.emit(False, f"无法连接 {base}：{e}")

        threading.Thread(target=work, daemon=True).start()

    def _show_net(self, ok: bool, msg: str) -> None:
        self.net_test.setEnabled(True)
        self.net_result.setText(("✓ " if ok else "✗ ") + msg)
        self.net_result.setStyleSheet("color:#2a9d4a" if ok else "color:#d33")

    # ── overlay ──────────────────────────────────────────────────────────────

    def _build_overlay(self) -> QWidget:
        self.ov_font = QSpinBox()
        self.ov_font.setRange(12, 72)
        self.ov_font.setSuffix(" pt")
        self.ov_opacity = QSlider(Qt.Orientation.Horizontal)
        self.ov_opacity.setRange(20, 100)
        self.ov_opacity_label = QLabel()
        self.ov_opacity.valueChanged.connect(lambda v: self.ov_opacity_label.setText(f"{v}%"))
        op = QWidget()
        h = QHBoxLayout(op)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.ov_opacity, 1)
        h.addWidget(self.ov_opacity_label)
        self.ov_source = QCheckBox("在译文上方显示原文")
        self.ov_click = QCheckBox("鼠标穿透（字幕条不拦截点击）")
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_form(("译文字号", self.ov_font), ("背景不透明度", op)))
        lay.addWidget(self.ov_source)
        lay.addWidget(self.ov_click)
        lay.addWidget(_hint("字幕条位置和大小直接拖动、缩放即可，会自动记住。"))
        lay.addStretch(1)
        return w

    # ── advanced ─────────────────────────────────────────────────────────────

    def _build_advanced(self) -> QWidget:
        d = SegmenterCfg()
        self.vad = QDoubleSpinBox()
        self.vad.setRange(0.2, 0.9)
        self.vad.setSingleStep(0.05)
        self.vad.setToolTip("越低越敏感（更容易把噪音当成语音），越高越严格")
        self.min_sil = QSpinBox()
        self.min_sil.setRange(150, 2000)
        self.min_sil.setSingleStep(50)
        self.min_sil.setSuffix(" ms")
        self.min_sil.setToolTip("停顿多久算一句话结束：越短字幕越快，但可能把一句话切碎")
        self.max_utt = QDoubleSpinBox()
        self.max_utt.setRange(4, 28)
        self.max_utt.setSuffix(" 秒")
        self.max_utt.setToolTip("一句话的最长时长，超过会在最安静处切开")
        self.partial = QSpinBox()
        self.partial.setRange(300, 2000)
        self.partial.setSingleStep(100)
        self.partial.setSuffix(" ms")
        self.partial.setToolTip("说话过程中刷新中间识别结果的间隔")
        self.beam = QSpinBox()
        self.beam.setRange(1, 8)
        self.beam.setToolTip("1 = 最快；越大越准也越慢")
        self.compute = QComboBox()
        for label, v in COMPUTE_TYPES:
            self.compute.addItem(label, v)
        reset = QPushButton("恢复默认")
        reset.clicked.connect(lambda: self._set_advanced(SegmenterCfg(), AsrCfg()))
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>断句</b>"))
        lay.addWidget(_form(("语音检测阈值", self.vad), ("句末停顿", self.min_sil), ("单句最长", self.max_utt),
                            ("中间结果间隔", self.partial)))
        lay.addWidget(QLabel("<b>本地识别</b>"))
        lay.addWidget(_form(("最终识别束宽 (beam)", self.beam), ("计算精度", self.compute)))
        lay.addWidget(_hint(f"默认值：阈值 {d.vad_threshold}、停顿 {d.min_silence_ms} ms、最长 {d.max_utterance_s:.0f} 秒、"
                            f"间隔 {d.partial_interval_ms} ms。主界面选择“速度 / 准确度”预设会覆盖这里的断句与束宽数值。"))
        lay.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)
        return w

    def _set_advanced(self, seg: SegmenterCfg, asr: AsrCfg) -> None:
        self.vad.setValue(seg.vad_threshold)
        self.min_sil.setValue(seg.min_silence_ms)
        self.max_utt.setValue(seg.max_utterance_s)
        self.partial.setValue(seg.partial_interval_ms)
        self.beam.setValue(asr.beam_final)
        i = self.compute.findData(asr.compute_type)
        self.compute.setCurrentIndex(max(0, i))

    # ── cfg <-> widgets ──────────────────────────────────────────────────────

    def _load(self, cfg: AppConfig) -> None:
        self.models_dir.setText(cfg.models_dir)
        self.transcripts_dir.setText(cfg.transcripts_dir)
        self.proxy.setText(cfg.proxy)
        i = self.mirror.findData(cfg.hf_endpoint)
        if i >= 0:
            self.mirror.setCurrentIndex(i)
        else:
            self.mirror.setEditText(cfg.hf_endpoint)
        o = cfg.overlay
        self.ov_font.setValue(o.font_size)
        self.ov_opacity.setValue(int(round(o.opacity * 100)))
        self.ov_opacity_label.setText(f"{self.ov_opacity.value()}%")
        self.ov_source.setChecked(o.show_source)
        self.ov_click.setChecked(o.click_through)
        self._set_advanced(cfg.seg, cfg.asr)
        if self._running:
            for w in self._storage_widgets:
                w.setEnabled(False)
        self._refresh_models()

    def apply_to(self, cfg: AppConfig) -> AppConfig:
        """A copy of ``cfg`` with everything from this dialog applied."""
        out = copy.deepcopy(cfg)
        out.models_dir = self.models_dir.text().strip()
        out.transcripts_dir = self.transcripts_dir.text().strip()
        out.proxy = self.proxy.text().strip()
        out.hf_endpoint = self._mirror_value()
        o = out.overlay
        o.font_size, o.opacity = self.ov_font.value(), self.ov_opacity.value() / 100
        o.show_source, o.click_through = self.ov_source.isChecked(), self.ov_click.isChecked()
        s = out.seg
        s.vad_threshold, s.min_silence_ms = round(self.vad.value(), 2), self.min_sil.value()
        s.max_utterance_s, s.partial_interval_ms = self.max_utt.value(), self.partial.value()
        out.asr.beam_final = self.beam.value()
        out.asr.compute_type = self.compute.currentData()
        return out

    # ── saving ───────────────────────────────────────────────────────────────

    def accept(self) -> None:
        new = self.apply_to(self._orig)
        for label, path in (("模型存放位置", new.models_dir), ("字幕记录位置", new.transcripts_dir)):
            if path and (err := storage.check_writable(path)):
                QMessageBox.warning(self, "目录不可用", f"{label}：{err}")
                return
        old_dir = self._orig.models_dir or runtime.default_models_dir()
        new_dir = new.models_dir or runtime.default_models_dir()
        if not self._running and not storage.same_dir(old_dir, new_dir):
            if not self._offer_move(old_dir, new_dir):
                return                                   # cancelled or failed → stay in the dialog
        self._result = new
        super().accept()

    def result_config(self) -> AppConfig:
        return self._result

    def _offer_move(self, old_dir: str, new_dir: str) -> bool:
        """Ask whether to move already-downloaded models. Returns False if the user cancelled / the move failed."""
        existing = storage.list_models(old_dir)
        if not existing:
            return True
        size = storage.human_size(sum(m.size for m in existing))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("更换模型存放位置")
        box.setText(f"旧位置里有 {len(existing)} 个模型（共 {size}）。要一起移动到新位置吗？")
        box.setInformativeText(f"旧位置：{old_dir}\n新位置：{new_dir}\n\n不移动的话，需要用到时会在新位置重新下载。")
        move = box.addButton("移动到新位置", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("不移动", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(move)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel or clicked is None:
            return False
        if clicked is move:
            return self._move_models(old_dir, new_dir)
        return True

    def _move_models(self, old_dir: str, new_dir: str) -> bool:
        progress = QProgressDialog("正在移动模型…（大文件跨磁盘移动需要几分钟）", None, 0, 0, self)
        progress.setWindowTitle("移动模型")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        result: dict = {}
        loop = QEventLoop(self)
        sig = self._sig

        def finish(moved, err):
            result.update(moved=moved, err=err)
            loop.quit()

        sig.moved.connect(finish)
        sig.step.connect(progress.setLabelText)      # queued to the UI thread; never touch widgets from the worker

        def work():
            try:
                moved = storage.move_models(old_dir, new_dir, sig.step.emit)
                sig.moved.emit(moved, "")
            except Exception as e:
                sig.moved.emit([], str(e))

        threading.Thread(target=work, daemon=True).start()
        progress.show()
        loop.exec()
        progress.close()
        sig.moved.disconnect(finish)
        sig.step.disconnect(progress.setLabelText)
        if result.get("err"):
            QMessageBox.warning(self, "移动失败", f"{result['err']}\n\n已移动的部分保留在新位置，其余仍在旧位置。")
            return False
        return True
