"""Preferences dialog: storage locations + model manager, network, subtitle overlay, advanced tuning."""
from __future__ import annotations

import copy
import os
import threading
import time

from PySide6.QtCore import QEventLoop, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QLinearGradient, QPainter, QShortcut
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressDialog,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import ruby, runtime, storage
from ..config import AppConfig, AsrCfg, OverlayCfg, SegmenterCfg, SpeakerCfg
from . import theme
from .ruby_text import RubyText, speaker_color
from .settings_panel import _hint
from .widgets import NavItem, Segmented, Switch, banner, icon_button, icon_label, label, section_label

TAB_STORAGE, TAB_NETWORK, TAB_OVERLAY, TAB_ADVANCED = range(4)

MIRRORS = [("官方 huggingface.co", ""), ("hf-mirror.com（国内镜像）", "https://hf-mirror.com")]
COMPUTE_TYPES = [("自动", "auto"), ("float16（GPU）", "float16"), ("int8_float16（GPU，省显存）", "int8_float16"),
                 ("int8（CPU，更快）", "int8"), ("float32", "float32")]
_NAV = [("drive", "存储", "模型与记录"), ("globe", "网络", "代理与下载站"),
        ("overlay", "悬浮字幕", "外观与学习模式"), ("sliders", "高级", "断句、识别、说话人")]


def open_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def model_kind(repo_id: str) -> str:
    r = repo_id.lower()
    if "whisper" in r:
        return "识别"
    if "nllb" in r:
        return "翻译"
    if "campplus" in r or "speaker" in r or "wespeaker" in r:
        return "说话人"
    return "其他"


class _Signals(QObject):
    moved = Signal(object, str)          # moved repo ids, error message
    step = Signal(str)                   # progress text while moving
    net = Signal(bool, str)              # ok, message


def _row(title: str, control: QWidget | None, desc: str = "", badge: bool = False) -> QFrame:
    """A settings row: title (+ optional "预设" badge) and a dim description on the left, the control on the right."""
    f = QFrame()
    f.setObjectName("settingRow")
    h = QHBoxLayout(f)
    h.setContentsMargins(0, 9, 0, 9)
    h.setSpacing(12)
    left = QVBoxLayout()
    left.setSpacing(2)
    head = QHBoxLayout()
    head.setSpacing(6)
    head.addWidget(QLabel(title))
    if badge:
        b = label("预设", px=10, weight=600, color=theme.ACCENT)
        b.setStyleSheet(b.styleSheet() + f";background:{theme.ACCENT_SOFT};border-radius:4px;padding:1px 5px;")
        b.setToolTip("主界面的「速度 / 准确度」预设会覆盖这一项")
        head.addWidget(b)
    head.addStretch(1)
    left.addLayout(head)
    if desc:
        d = label(desc, "muted")
        d.setStyleSheet("font-size:11px;")
        d.setWordWrap(True)
        left.addWidget(d)
    h.addLayout(left, 1)
    if control is not None:
        h.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    return f


def _switch_row(sw: Switch) -> QFrame:
    f = QFrame()
    f.setObjectName("settingRow")
    h = QHBoxLayout(f)
    h.setContentsMargins(0, 8, 0, 8)
    h.addWidget(sw)
    return f


def _group(title: str, icon: str, *rows: QWidget) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    v.addWidget(section_label(title, icon))
    v.addSpacing(2)
    for r in rows:
        v.addWidget(r)
    return w


class _PreviewBar(QWidget):
    """The subtitle bar inside the live preview: painted background with the configured opacity."""

    def __init__(self):
        super().__init__()
        self.opacity = 0.72

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(12, 14, 18, int(255 * self.opacity)))
        p.setPen(QColor(255, 255, 255, 26))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)


class _Preview(QWidget):
    """What the floating subtitle bar will look like with the settings on the page (scaled down a little)."""
    _OLD = ("没错，这也是调谐循环如此重要的原因。", "zh-Hans")
    _SRC = ("Exactly. Let's look at a real example.", "en")
    _DST = ("没错，我们来看一个真实的例子。", "zh-Hans")

    def __init__(self):
        super().__init__()
        self.setFixedHeight(172)
        self.bar = _PreviewBar()
        self.old, self.src, self.dst = RubyText(), RubyText(), RubyText()
        bl = QVBoxLayout(self.bar)
        bl.setContentsMargins(16, 10, 16, 10)
        bl.setSpacing(3)
        for w in (self.old, self.src, self.dst):
            bl.addWidget(w)
        tag = label("  实时预览", px=10, color="#FFFFFFAA")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.addWidget(tag, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.bar, 6)
        row.addStretch(1)
        lay.addLayout(row)

    def refresh(self, o: OverlayCfg) -> None:
        pt = max(9, round(min(o.font_size, 40) * 0.62))
        big = self.font()
        big.setPointSize(pt)
        big.setBold(True)
        small = self.font()
        small.setPointSize(max(8, int(pt * 0.6)))
        self.old.setFont(big)
        self.dst.setFont(big)
        self.src.setFont(small)
        self.old.setColor("#aeb8c5")
        self.src.setColor("#b8c2cf")
        self.dst.setColor("#ffffff")
        ruby_dst = o.learning and o.ruby_scope in ("both", "dst")
        ruby_src = o.learning and o.ruby_scope in ("both", "src")
        self.old.setText(*self._OLD, ruby_dst, "[2]", speaker_color(2))
        self.old.setVisible(o.max_sentences > 1)
        self.src.setText(*self._SRC, ruby_src)
        self.src.setVisible(o.show_source)
        self.dst.setText(*self._DST, ruby_dst, "[1]", speaker_color(1))
        self.bar.opacity = o.opacity
        self.bar.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0, QColor("#2A3446"))
        g.setColorAt(1, QColor("#141821"))
        p.setBrush(g)
        p.setPen(QColor(theme.BORDER))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)


class PreferencesDialog(QDialog):
    applied = Signal(object)             # AppConfig, when「应用」saved the settings without closing

    def __init__(self, cfg: AppConfig, running: bool = False, parent=None, tab: int = TAB_STORAGE):
        super().__init__(parent)
        self.setWindowTitle("偏好设置")
        self.resize(940, 680)
        self.setMinimumSize(760, 560)
        self._orig = copy.deepcopy(cfg)
        self._running = running
        self._result: AppConfig | None = None
        self.restart_requested = False
        self._sig = _Signals()
        self._sig.net.connect(self._show_net)

        # header: title + search
        head = QFrame()
        head.setObjectName("dlgHeader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(20, 10, 20, 10)
        hl.setSpacing(10)
        hl.addWidget(icon_label("sliders", theme.TEXT, 16))
        hl.addWidget(label("偏好设置", px=15, weight=600))
        hl.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索设置…  Ctrl+F")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(theme.icon("search", theme.TEXT_3, size=14), QLineEdit.ActionPosition.LeadingPosition)
        self.search.setFixedWidth(250)
        self.search.textChanged.connect(self._search)
        QShortcut(QKeySequence.StandardKey.Find, self, self.search.setFocus)
        hl.addWidget(self.search)

        # navigation + pages
        nav = QFrame()
        nav.setObjectName("navPane")
        nav.setFixedWidth(200)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(10, 14, 10, 14)
        nl.setSpacing(2)
        self.nav_group = QButtonGroup(self)
        self.nav_items: list[NavItem] = []
        for i, (ic, title, sub) in enumerate(_NAV):
            it = NavItem(ic, title, sub)
            self.nav_group.addButton(it, i)
            self.nav_items.append(it)
            nl.addWidget(it)
        nl.addStretch(1)
        self.tabs = QStackedWidget()
        for page in (self._build_storage(), self._build_network(), self._build_overlay(), self._build_advanced()):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.Shape.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            sc.setWidget(page)
            self.tabs.addWidget(sc)
        self.nav_group.idClicked.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(lambda i: self.nav_items[i].setChecked(True))
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(nav)
        body.addWidget(self.tabs, 1)

        # footer
        foot = QFrame()
        foot.setObjectName("dlgFooter")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(20, 12, 20, 12)
        fl.setSpacing(10)
        reset = QPushButton("恢复本页默认")
        reset.setObjectName("link")
        reset.setIcon(theme.icon("reset", theme.TEXT_3, size=13))
        reset.clicked.connect(self._reset_page)
        fl.addWidget(reset)
        fl.addStretch(1)
        cancel, apply, ok = QPushButton("取消"), QPushButton("应用"), QPushButton("确定")
        ok.setObjectName("primary")
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        apply.clicked.connect(self._apply)
        ok.clicked.connect(self.accept)
        for b in (cancel, apply, ok):
            b.setMinimumWidth(72)
            fl.addWidget(b)
        self.apply_btn = apply

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(head)
        lay.addLayout(body, 1)
        lay.addWidget(foot)

        self._load(cfg)
        self.tabs.setCurrentIndex(tab)
        self.nav_items[tab].setChecked(True)
        self.nav_items[tab].setFocus()                   # not the search box: typing should not start a search

    # ── page scaffolding ─────────────────────────────────────────────────────

    @staticmethod
    def _page(title: str, desc: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(28, 22, 28, 22)
        v.setSpacing(16)
        h = QVBoxLayout()
        h.setSpacing(4)
        h.addWidget(label(title, "pageTitle"))
        d = label(desc, "muted")
        d.setWordWrap(True)
        h.addWidget(d)
        v.addLayout(h)
        return w, v

    def _search(self, text: str) -> None:
        """Dim the pages that do not mention the query and jump to the first one that does."""
        q = text.strip().lower()
        hits = []
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            words = [w.text() for w in page.findChildren(QLabel)] + [w.text() for w in page.findChildren(QAbstractButton)]
            words += [w.description() for w in page.findChildren(Switch)] + [_NAV[i][1], _NAV[i][2]]
            hit = not q or any(q in t.lower() for t in words)
            self.nav_items[i].set_dimmed(not hit)
            if hit:
                hits.append(i)
        if q and hits and self.tabs.currentIndex() not in hits:
            self.tabs.setCurrentIndex(hits[0])

    # ── storage ──────────────────────────────────────────────────────────────

    def _dir_row(self, edit: QLineEdit, placeholder: str, browse_title: str) -> QWidget:
        edit.setPlaceholderText(placeholder)
        edit.setFont(theme.mono_font(12))
        browse = QPushButton("浏览…")
        opn = icon_button("folder", "打开文件夹", 14, "fieldBtn")
        reset = icon_button("reset", "恢复默认位置", 14, "fieldBtn")
        browse.clicked.connect(lambda: self._browse(edit, browse_title))
        reset.clicked.connect(edit.clear)
        opn.clicked.connect(lambda: open_folder(edit.text().strip() or edit.placeholderText()))
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(edit, 1)
        for b in (browse, opn, reset):
            h.addWidget(b)
        w.buttons = (browse, reset)
        return w

    def _browse(self, edit: QLineEdit, title: str) -> None:
        start = edit.text().strip() or edit.placeholderText()
        path = QFileDialog.getExistingDirectory(self, title, start if os.path.isdir(start) else os.path.expanduser("~"))
        if path:
            edit.setText(os.path.normpath(path))

    def _build_storage(self) -> QWidget:
        w, lay = self._page("存储", "模型和字幕记录保存在哪里；管理已下载的模型。")
        self.models_dir = QLineEdit()
        self.transcripts_dir = QLineEdit()
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["已下载的模型", "类型", "大小"])
        self.models_tree.setRootIsDecorated(False)
        self.models_tree.setAlternatingRowColors(True)
        self.models_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        hdr = self.models_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.models_tree.setMinimumHeight(170)
        self.models_tree.itemSelectionChanged.connect(self._on_model_selection)
        self.models_total = label("", "muted")
        self.delete_btn = QPushButton("删除所选")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.setIcon(theme.icon("trash", theme.DANGER, size=13))
        self.delete_btn.clicked.connect(self._delete_selected)
        refresh = QPushButton("刷新")
        refresh.setIcon(theme.icon("refresh", theme.TEXT_2, size=13))
        refresh.clicked.connect(self._refresh_models)
        self.storage_note = banner("warnBanner", "lock", "识别正在运行：请先停止，再修改存储位置或删除模型。")
        self.storage_note.setVisible(self._running)

        lay.addWidget(self.storage_note)
        models = QVBoxLayout()
        models.setSpacing(8)
        models.addWidget(section_label("本地模型存放位置", "layers"))
        self._models_row = self._dir_row(self.models_dir, runtime.default_models_dir(), "选择模型存放目录")
        models.addWidget(self._models_row)
        models.addWidget(_hint("faster-whisper 识别模型和内置离线翻译（NLLB）都下载到这里。更换位置后，如果旧位置里已有模型，"
                               "保存时可以选择一并移动。留空 = 使用 Hugging Face 默认缓存目录。"))
        lay.addLayout(models)
        listing = QVBoxLayout()
        listing.setSpacing(8)
        lh = QHBoxLayout()
        lh.addWidget(section_label("已下载的模型", "drive"), 1)
        lh.addWidget(self.models_total)
        listing.addLayout(lh)
        listing.addWidget(self.models_tree, 1)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.delete_btn)
        row.addWidget(refresh)
        row.addStretch(1)
        listing.addLayout(row)
        lay.addLayout(listing, 1)
        tx = QVBoxLayout()
        tx.setSpacing(8)
        tx.addWidget(section_label("字幕记录保存位置", "file"))
        self._tx_row = self._dir_row(self.transcripts_dir, os.path.join(str(self._config_dir()), "transcripts"),
                                     "选择字幕记录目录")
        tx.addWidget(self._tx_row)
        lay.addLayout(tx)
        self.models_dir.textChanged.connect(self._refresh_models)
        self._storage_widgets = (self.models_dir, self.transcripts_dir, self.delete_btn,
                                 *self._models_row.buttons, *self._tx_row.buttons)
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
            item = QTreeWidgetItem([m.repo_id, model_kind(m.repo_id), storage.human_size(m.size)])
            item.setData(0, Qt.ItemDataRole.UserRole, m)
            item.setFont(0, theme.mono_font(12))
            item.setForeground(1, QColor(theme.TEXT_2))
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.models_tree.addTopLevelItem(item)
        total = sum(m.size for m in models)
        self.models_total.setText(f"共 {len(models)} 个模型，占用 {storage.human_size(total)}" if models else "该目录下还没有模型")
        self._on_model_selection()

    def _on_model_selection(self) -> None:
        n = len(self.models_tree.selectedItems())
        self.delete_btn.setText(f"删除所选 ({n})" if n else "删除所选")

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
        w, lay = self._page("网络", "远程 API 调用和模型下载使用的代理与下载站。本机服务（Ollama 等）不走代理。")
        self.proxy = QLineEdit()
        self.proxy.setFont(theme.mono_font(12))
        self.proxy.setPlaceholderText("留空 = 不使用代理；例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080")
        self.mirror = QComboBox()
        self.mirror.setEditable(True)
        for lbl, url in MIRRORS:
            self.mirror.addItem(lbl, url)
        self.net_test = QPushButton("测试连接")
        self.net_test.setIcon(theme.icon("activity", theme.TEXT_2, size=13))
        self.net_test.clicked.connect(self._test_network)
        self.net_result = QLabel()
        self.net_result.setWordWrap(True)
        self.net_result.setFont(theme.mono_font(12))

        g = QVBoxLayout()
        g.setSpacing(8)
        g.addWidget(section_label("网络代理", "shield"))
        g.addWidget(self.proxy)
        g.addWidget(_hint("留空 = 不使用代理。支持 http:// 与 socks5://"))
        lay.addLayout(g)
        g = QVBoxLayout()
        g.setSpacing(8)
        g.addWidget(section_label("模型下载站", "cloud"))
        g.addWidget(self.mirror)
        g.addWidget(_hint("国内访问 huggingface.co 较慢时，可改为镜像；也可以直接输入自建镜像的地址。"))
        lay.addLayout(g)
        t = QHBoxLayout()
        t.setSpacing(12)
        t.addWidget(self.net_test)
        t.addWidget(self.net_result, 1)
        lay.addLayout(t)
        lay.addStretch(1)
        note = QFrame()
        note.setObjectName("box")
        nl = QHBoxLayout(note)
        nl.setContentsMargins(12, 10, 12, 10)
        nl.setSpacing(10)
        nl.addWidget(icon_label("restart", theme.WARN, 14))
        msg = label("代理与下载站在重启程序后完全生效（模型下载组件只在启动时读取它们）。", "muted")
        msg.setWordWrap(True)
        nl.addWidget(msg, 1)
        self.restart_btn = QPushButton("保存并重启")
        self.restart_btn.setObjectName("warnBtn")
        self.restart_btn.clicked.connect(self._save_and_restart)
        nl.addWidget(self.restart_btn)
        lay.addWidget(note)
        return w

    def _save_and_restart(self) -> None:
        self.restart_requested = True
        self.accept()
        if self.result() != QDialog.DialogCode.Accepted:
            self.restart_requested = False

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
        self.net_result.setStyleSheet(f"color:{theme.TEXT_3}")

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
        self.net_result.setStyleSheet(f"color:{theme.LIVE if ok else theme.DANGER}")

    # ── overlay ──────────────────────────────────────────────────────────────

    def _build_overlay(self) -> QWidget:
        w, lay = self._page("悬浮字幕", "显示在所有窗口上方的字幕条。拖动即可移动，拖右下角改大小，右键有快捷菜单。")
        self.preview = _Preview()
        self.ov_font = QSlider(Qt.Orientation.Horizontal)
        self.ov_font.setRange(12, 72)
        self.ov_font.setFixedWidth(120)
        self.ov_font_label = label("", px=12, color=theme.TEXT_2)
        self.ov_font_label.setFont(theme.mono_font(12))
        self.ov_font_label.setFixedWidth(40)
        self.ov_font.valueChanged.connect(lambda v: self.ov_font_label.setText(f"{v}pt"))
        self.ov_opacity = QSlider(Qt.Orientation.Horizontal)
        self.ov_opacity.setRange(20, 100)
        self.ov_opacity.setFixedWidth(120)
        self.ov_opacity_label = label("", px=12, color=theme.TEXT_2)
        self.ov_opacity_label.setFont(theme.mono_font(12))
        self.ov_opacity_label.setFixedWidth(40)
        self.ov_opacity.valueChanged.connect(lambda v: self.ov_opacity_label.setText(f"{v}%"))
        self.ov_sentences = QSpinBox()
        self.ov_sentences.setRange(1, 8)
        self.ov_sentences.setSuffix(" 句")
        self.ov_sentences.setFixedWidth(84)
        self.ov_sentences.setToolTip("悬浮字幕同时显示的句数。1 = 只显示当前这一句；2 以上时，前面的句子留在当前句上方"
                                     "（颜色稍暗），最旧的先消失。每句完整显示，可以折成多行")
        self.ov_auto_h = Switch("自动调整高度", "关闭后保持你拖出来的高度，放不下的旧字幕从顶部裁掉")
        self.ov_source = Switch("显示原文", "在译文上方显示原文")
        self.ov_click = Switch("鼠标穿透", "字幕条不拦截点击")
        self.ov_learn = Switch("启用学习模式", "中文汉字标拼音、日文汉字标假名，字幕前显示朗读按钮")
        self.ov_ruby = Segmented([("译文", "dst"), ("原文", "src"), ("两者", "both")])
        self.ov_tts_read = Segmented([("译文", "dst"), ("原文", "src")])

        def slider(s, lbl):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(10)
            h.addWidget(s)
            h.addWidget(lbl)
            return box

        look = _group("外观", "palette",
                      _row("译文字号", slider(self.ov_font, self.ov_font_label)),
                      _row("同时显示句数", self.ov_sentences, "1 = 只显示当前句"),
                      _row("背景不透明度", slider(self.ov_opacity, self.ov_opacity_label)),
                      _switch_row(self.ov_auto_h), _switch_row(self.ov_source), _switch_row(self.ov_click))
        deps = QFrame()
        deps.setObjectName("box")
        dl = QVBoxLayout(deps)
        dl.setContentsMargins(12, 10, 12, 10)
        dl.setSpacing(7)
        missing = ruby.missing_packages()
        for pkg, what in (("pypinyin", "中文注音"), ("pykakasi", "日文注音"), ("espeakng-loader", "其他语言音标")):
            ok = pkg not in missing
            r = QHBoxLayout()
            r.setSpacing(8)
            r.addWidget(icon_label("check" if ok else "alert", theme.LIVE if ok else theme.WARN, 13))
            r.addWidget(label(f"{pkg} {'已安装' if ok else '未安装'} · {what}{'可用' if ok else '不可用：pip install ' + pkg}",
                              px=12, color=theme.TEXT_2), 1)
            dl.addLayout(r)
        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(icon_label("info", theme.TEXT_3, 13), 0, Qt.AlignmentFlag.AlignTop)
        v = label("朗读使用系统自带的语音；缺少对应语言的语音包时，开启学习模式或点「开始」会提示如何安装。"
                  "开启「鼠标穿透」时悬浮字幕上的朗读按钮点不到，会被隐藏。", px=11, color=theme.TEXT_3)
        v.setWordWrap(True)
        r.addWidget(v, 1)
        dl.addLayout(r)
        learn = _group("学习模式", "learn", _switch_row(self.ov_learn), _row("注音标在", self.ov_ruby),
                       _row("朗读内容", self.ov_tts_read))
        learn.layout().addSpacing(12)
        learn.layout().addWidget(deps)
        learn.layout().addStretch(1)
        cols = QHBoxLayout()
        cols.setSpacing(28)
        cols.addWidget(look, 1, Qt.AlignmentFlag.AlignTop)
        cols.addWidget(learn, 1)
        lay.addWidget(self.preview)
        lay.addLayout(cols)
        lay.addStretch(1)
        for sig in (self.ov_font.valueChanged, self.ov_opacity.valueChanged, self.ov_sentences.valueChanged,
                    self.ov_source.toggled, self.ov_learn.toggled, self.ov_ruby.currentIndexChanged):
            sig.connect(self._refresh_preview)
        return w

    def _overlay_values(self, o: OverlayCfg) -> OverlayCfg:
        o.font_size, o.opacity = self.ov_font.value(), self.ov_opacity.value() / 100
        o.show_source, o.click_through = self.ov_source.isChecked(), self.ov_click.isChecked()
        o.max_sentences, o.auto_height = self.ov_sentences.value(), self.ov_auto_h.isChecked()
        o.learning, o.tts_read = self.ov_learn.isChecked(), self.ov_tts_read.currentData()
        o.ruby_scope = self.ov_ruby.currentData()
        return o

    def _refresh_preview(self, *_a) -> None:
        self.preview.refresh(self._overlay_values(copy.deepcopy(self._orig.overlay)))

    def _set_overlay(self, o: OverlayCfg) -> None:
        self.ov_font.setValue(o.font_size)
        self.ov_font_label.setText(f"{o.font_size}pt")
        self.ov_opacity.setValue(int(round(o.opacity * 100)))
        self.ov_opacity_label.setText(f"{self.ov_opacity.value()}%")
        self.ov_sentences.setValue(o.max_sentences)
        self.ov_auto_h.setChecked(o.auto_height)
        self.ov_source.setChecked(o.show_source)
        self.ov_click.setChecked(o.click_through)
        self.ov_learn.setChecked(o.learning)
        self.ov_tts_read.setCurrentIndex(max(0, self.ov_tts_read.findData(o.tts_read)))
        self.ov_ruby.setCurrentIndex(max(0, self.ov_ruby.findData(o.ruby_scope)))
        self._refresh_preview()

    # ── advanced ─────────────────────────────────────────────────────────────

    def _build_advanced(self) -> QWidget:
        w, lay = self._page("高级", "断句、识别精度与说话人区分的细节参数。不确定时保持默认即可。")
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
        self.max_utt.setDecimals(1)
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
        for lbl, v in COMPUTE_TYPES:
            self.compute.addItem(lbl, v)
        self.spk_threshold = QDoubleSpinBox()
        self.spk_threshold.setRange(0.2, 0.9)
        self.spk_threshold.setSingleStep(0.05)
        self.spk_threshold.setToolTip("声音多相似才算同一个人。调高：更容易把同一个人分成两个；调低：更容易把不同的人并成一个")
        self.spk_max = QSpinBox()
        self.spk_max.setRange(0, 12)
        self.spk_max.setSpecialValueText("不限")
        self.spk_max.setToolTip("最多区分几个人；达到上限后新的声音会归到最像的那个人")
        for s in (self.vad, self.min_sil, self.max_utt, self.partial, self.beam, self.spk_threshold, self.spk_max):
            s.setFixedWidth(112)
        self.compute.setFixedWidth(190)

        lay.addWidget(banner("infoBanner", "info", "主界面选择「速度 / 准确度」预设时，会覆盖标有「预设」的数值。"))
        seg = _group("断句", "scissors",
                     _row("语音检测阈值", self.vad, "越高越不容易把噪音当成说话"),
                     _row("句末停顿", self.min_sil, "停顿多久算一句结束", badge=True),
                     _row("单句最长", self.max_utt, "超过会在最安静处切开", badge=True),
                     _row("中间结果间隔", self.partial, "流式字幕刷新频率", badge=True))
        asr = _group("本地识别", "cpu",
                     _row("束宽 (beam)", self.beam, "1 = 最快；越大越准也越慢", badge=True),
                     _row("计算精度", self.compute))
        spk = _group("说话人区分", "users",
                     _row("相似度阈值", self.spk_threshold, "调高更容易拆开，调低更容易合并"),
                     _row("最多说话人数", self.spk_max, "在主界面勾选「区分说话人」后生效"))
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(asr)
        right.addWidget(spk)
        right.addStretch(1)
        cols = QHBoxLayout()
        cols.setSpacing(28)
        cols.addWidget(seg, 1, Qt.AlignmentFlag.AlignTop)
        cols.addLayout(right, 1)
        lay.addLayout(cols)
        lay.addWidget(_hint(f"默认值：阈值 {d.vad_threshold}、停顿 {d.min_silence_ms} ms、最长 {d.max_utterance_s:.0f} 秒、"
                            f"间隔 {d.partial_interval_ms} ms。"))
        lay.addStretch(1)
        return w

    def _set_advanced(self, seg: SegmenterCfg, asr: AsrCfg, spk: SpeakerCfg) -> None:
        self.spk_threshold.setValue(spk.similarity)
        self.spk_max.setValue(spk.max_speakers)
        self.vad.setValue(seg.vad_threshold)
        self.min_sil.setValue(seg.min_silence_ms)
        self.max_utt.setValue(seg.max_utterance_s)
        self.partial.setValue(seg.partial_interval_ms)
        self.beam.setValue(asr.beam_final)
        i = self.compute.findData(asr.compute_type)
        self.compute.setCurrentIndex(max(0, i))

    def _reset_page(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_STORAGE:
            if not self._running:
                self.models_dir.clear()
                self.transcripts_dir.clear()
        elif i == TAB_NETWORK:
            self.proxy.clear()
            self.mirror.setCurrentIndex(0)
        elif i == TAB_OVERLAY:
            self._set_overlay(OverlayCfg())
        else:
            self._set_advanced(SegmenterCfg(), AsrCfg(), SpeakerCfg())

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
        self._set_overlay(cfg.overlay)
        self._set_advanced(cfg.seg, cfg.asr, cfg.speaker)
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
        self._overlay_values(out.overlay)
        s = out.seg
        s.vad_threshold, s.min_silence_ms = round(self.vad.value(), 2), self.min_sil.value()
        s.max_utterance_s, s.partial_interval_ms = self.max_utt.value(), self.partial.value()
        out.speaker.similarity, out.speaker.max_speakers = round(self.spk_threshold.value(), 2), self.spk_max.value()
        out.asr.beam_final = self.beam.value()
        out.asr.compute_type = self.compute.currentData()
        return out

    # ── saving ───────────────────────────────────────────────────────────────

    def _commit(self) -> AppConfig | None:
        """Validate and (if the model folder changed) offer to move the models. None = stay in the dialog."""
        new = self.apply_to(self._orig)
        for lbl, path in (("模型存放位置", new.models_dir), ("字幕记录位置", new.transcripts_dir)):
            if path and (err := storage.check_writable(path)):
                QMessageBox.warning(self, "目录不可用", f"{lbl}：{err}")
                return None
        old_dir = self._orig.models_dir or runtime.default_models_dir()
        new_dir = new.models_dir or runtime.default_models_dir()
        if not self._running and not storage.same_dir(old_dir, new_dir):
            if not self._offer_move(old_dir, new_dir):
                return None                              # cancelled or failed → stay in the dialog
        return new

    def accept(self) -> None:
        new = self._commit()
        if new is None:
            return
        self._result = new
        super().accept()

    def _apply(self) -> None:
        new = self._commit()
        if new is None:
            return
        self._orig = copy.deepcopy(new)                  # a later move offer compares against what is now saved
        self._result = new
        self.applied.emit(new)

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
