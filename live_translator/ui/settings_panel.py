"""Settings side panel. ``load(cfg)`` fills the widgets, ``collect()`` returns a new AppConfig from them."""
from __future__ import annotations

import copy
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QCompleter, QPlainTextEdit, QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import audio as audio_mod
from ..config import AppConfig, apply_preset
from ..languages import SOURCE_CHOICES, TARGET_CHOICES
from ..runtime import cuda_available
from .bridge import Bridge

WHISPER_MODELS = [
    ("自动（按硬件推荐）", "auto"),
    ("tiny — 最快，准确度低", "tiny"),
    ("base", "base"),
    ("small — CPU 推荐", "small"),
    ("medium", "medium"),
    ("large-v3-turbo — GPU 推荐，快且准", "large-v3-turbo"),
    ("large-v3 — 最准，最慢", "large-v3"),
    ("distil-large-v3 — 仅英语，很快", "distil-large-v3"),
]

ASR_REMOTE_PRESETS = [
    ("OpenAI", "https://api.openai.com/v1", "whisper-1"),
    ("Groq（极快）", "https://api.groq.com/openai/v1", "whisper-large-v3-turbo"),
    ("SiliconFlow 硅基流动", "https://api.siliconflow.cn/v1", "FunAudioLLM/SenseVoiceSmall"),
    ("自建 faster-whisper-server", "http://127.0.0.1:8000/v1", "Systran/faster-whisper-large-v3"),
    ("自定义", "", ""),
]

LLM_REMOTE_PRESETS = [
    ("SiliconFlow 硅基流动", "https://api.siliconflow.cn/v1", "Qwen/Qwen3.6-35B-A3B"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("通义千问 DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("Groq（极快）", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    ("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    ("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ("自定义", "", ""),
]

LLM_LOCAL_PRESETS = [
    ("Ollama", "http://127.0.0.1:11434/v1", "qwen2.5:7b"),
    ("LM Studio", "http://127.0.0.1:1234/v1", ""),
    ("llama.cpp server", "http://127.0.0.1:8080/v1", ""),
    ("vLLM", "http://127.0.0.1:8000/v1", ""),
    ("自定义", "", ""),
]

NLLB_MODELS = [
    ("NLLB-200 600M（int8，约 0.6 GB）", "JustFrederik/nllb-200-distilled-600M-ct2-int8"),
    ("NLLB-200 1.3B（int8，约 1.4 GB，更准）", "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"),
]

TR_MODES = [
    ("远程 API（OpenAI 兼容）", "llm_remote"),
    ("本地模型服务（Ollama / LM Studio …）", "llm_local"),
    ("内置离线模型（NLLB，无需服务）", "nllb"),
    ("DeepL", "deepl"),
    ("仅转写，不翻译", "none"),
]

PRESET_CHOICES = [("速度优先", "fast"), ("均衡", "balanced"), ("准确优先", "accurate")]


class _Stack(QStackedWidget):
    """QStackedWidget normally reserves the height of its tallest page. Pages that are not showing are given an
    ``Ignored`` size policy so the stack is exactly as tall as the visible page."""

    def __init__(self):
        super().__init__()
        self.currentChanged.connect(self._sync)

    def addWidget(self, w: QWidget) -> int:
        i = super().addWidget(w)
        self._sync()
        return i

    def _sync(self, *_args) -> None:
        cur = self.currentIndex()
        for i in range(self.count()):
            p = QSizePolicy.Policy.Preferred if i == cur else QSizePolicy.Policy.Ignored
            self.widget(i).setSizePolicy(p, p)
        self.updateGeometry()


def _combo(items, editable=False) -> QComboBox:
    cb = QComboBox()
    cb.setEditable(editable)
    # Without this the widest item label sets the minimum width and pushes the panel past the scroll area.
    cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(8)
    for label, data in items:
        cb.addItem(label, data)
    return cb


def _select(cb: QComboBox, data) -> None:
    i = cb.findData(data)
    if i >= 0:
        cb.setCurrentIndex(i)
    elif cb.isEditable():
        cb.setEditText(str(data))


def _value(cb: QComboBox) -> str:
    """Value of an editable combo whose items have a label ≠ data: the item's data unless the user typed something else."""
    i = cb.currentIndex()
    if i >= 0 and cb.currentText() == cb.itemText(i):
        return str(cb.itemData(i) or cb.itemText(i))
    return cb.currentText().strip()


def _line(placeholder="", password=False) -> QLineEdit:
    e = QLineEdit()
    e.setPlaceholderText(placeholder)
    if password:
        e.setEchoMode(QLineEdit.EchoMode.Password)
    return e


def _form(*rows) -> QWidget:
    w = QWidget()
    f = QFormLayout(w)
    f.setContentsMargins(0, 0, 0, 0)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for label, widget in rows:
        f.addRow(label, widget) if label else f.addRow(widget)
    return w


def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setWordWrap(True)
    l.setEnabled(False)          # dimmed
    return l


class SettingsPanel(QWidget):
    changed = Signal()

    def __init__(self, bridge: Bridge):
        super().__init__()
        self.bridge = bridge
        self._base = AppConfig()
        self._loading = False
        self._apps: list = []
        self._fetching: set[str] = set()
        self._fetched: dict[str, tuple[str, str]] = {}      # side -> (url, key) of the last list we obtained

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        for g in (self._build_audio(), self._build_lang(), self._build_asr(), self._build_translate(),
                  self._build_perf()):
            lay.addWidget(g)
        lay.addStretch(1)
        bridge.apps.connect(self._set_apps)
        bridge.test_result.connect(self._show_test)

    # ── audio ────────────────────────────────────────────────────────────────

    def _build_audio(self) -> QGroupBox:
        g = QGroupBox("① 音频来源")
        self.src_kind = _combo([("系统全部声音", "system"), ("指定应用的声音", "app"), ("麦克风", "mic")])
        if not audio_mod.app_capture_supported():
            self.src_kind.model().item(1).setEnabled(False)
        self.app_combo = QComboBox()
        self.app_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.app_combo.setMinimumContentsLength(18)
        self.app_refresh = QPushButton("刷新")
        self.app_refresh.setToolTip("重新扫描正在运行 / 正在发声的应用（● 表示正在发声）")
        app_row = QWidget()
        h = QHBoxLayout(app_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.app_combo, 1)
        h.addWidget(self.app_refresh)
        self.app_row = app_row
        self.mic_combo = QComboBox()
        self.app_hint = _hint("只捕获所选应用（含其子进程）的声音，其他应用和系统提示音不会混入。"
                              + ("首次使用需授予「屏幕录制」权限。" if audio_mod.IS_MAC else ""))
        self.src_form = _form(("来源", self.src_kind), ("应用", app_row), ("设备", self.mic_combo))
        lay = QVBoxLayout(g)
        lay.addWidget(self.src_form)
        lay.addWidget(self.app_hint)
        self._form_rows = self.src_form.layout()

        self.src_kind.currentIndexChanged.connect(self._on_src_kind)
        self.app_refresh.clicked.connect(self.refresh_apps)
        return g

    def _on_src_kind(self) -> None:
        kind = self.src_kind.currentData()
        self._form_rows.setRowVisible(self.app_row, kind == "app")
        self._form_rows.setRowVisible(self.mic_combo, kind == "mic")
        self.app_hint.setVisible(kind == "app")
        if kind == "app" and not self._apps:
            self.refresh_apps()
        if kind == "mic" and self.mic_combo.count() == 0:
            from ..audio.mic import list_input_devices
            self.mic_combo.addItem("系统默认麦克风", "")
            for n in list_input_devices():
                self.mic_combo.addItem(n, n)
        self._emit()

    def refresh_apps(self) -> None:
        self.app_refresh.setEnabled(False)
        self.app_refresh.setText("扫描中…")

        def work():
            try:
                apps = audio_mod.list_audio_apps()
            except Exception as e:                       # e.g. macOS helper not compiled / no permission
                self.bridge.status.emit("warn", f"无法列出应用：{e}")
                apps = []
            self.bridge.apps.emit(apps)

        threading.Thread(target=work, daemon=True).start()

    def _set_apps(self, apps: list) -> None:
        self._apps = apps
        self.app_refresh.setEnabled(True)
        self.app_refresh.setText("刷新")
        selected = self.app_combo.currentData()                   # (key, name) tuple or None
        cur = selected[0] if selected else self._base.audio.app_key
        self.app_combo.blockSignals(True)
        self.app_combo.clear()
        for a in apps:
            self.app_combo.addItem(a.label, (a.key, a.name))
        keys = [a.key for a in apps]
        if cur and cur not in keys:
            self.app_combo.insertItem(0, f"   {self._base.audio.app_name or cur}（当前未运行）", (cur, self._base.audio.app_name or cur))
        for i in range(self.app_combo.count()):
            if self.app_combo.itemData(i)[0] == cur:
                self.app_combo.setCurrentIndex(i)
                break
        self.app_combo.blockSignals(False)
        self._emit()

    # ── language ─────────────────────────────────────────────────────────────

    def _build_lang(self) -> QGroupBox:
        g = QGroupBox("② 语言")
        self.src_lang = _combo([(label, code) for code, label in SOURCE_CHOICES])
        self.tgt_lang = _combo([(label, code) for code, label in TARGET_CHOICES])
        self.lang_hint = _hint("自动检测会在多语言混杂时自动切换；如果只有一种语言，手动指定更快更准。")
        lay = QVBoxLayout(g)
        lay.addWidget(_form(("原语言", self.src_lang), ("翻译为", self.tgt_lang)))
        lay.addWidget(self.lang_hint)
        return g

    # ── ASR ──────────────────────────────────────────────────────────────────

    def _build_asr(self) -> QGroupBox:
        g = QGroupBox("③ 语音识别")
        self.asr_mode = _combo([("本地（faster-whisper）", "local"), ("远程 API", "remote")])

        self.asr_model = _combo(WHISPER_MODELS, editable=True)
        self.asr_model.setToolTip("也可以直接填 Hugging Face 仓库名或本地模型目录")
        self.asr_device = _combo([("自动", "auto"), ("GPU (CUDA)", "cuda"), ("CPU", "cpu")])
        gpu = "已检测到 NVIDIA GPU，将使用 CUDA 加速。" if cuda_available() else \
            "未检测到 NVIDIA GPU，将使用 CPU（建议 small 及以下模型；Mac 用 Apple 芯片同样走 CPU int8）。"
        local = QWidget()
        ll = QVBoxLayout(local)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(_form(("模型", self.asr_model), ("设备", self.asr_device)))
        ll.addWidget(_hint(gpu + " 首次使用会自动下载模型。"))

        self.asr_preset = _combo([(n, i) for i, (n, *_r) in enumerate(ASR_REMOTE_PRESETS)])
        self.asr_url = _line("https://api.openai.com/v1")
        self.asr_key = _line("API Key", password=True)
        self.asr_rmodel = _line("whisper-1")
        self.asr_rpartials = QCheckBox("识别中间结果（每次请求都计费，默认关闭）")
        self.asr_test = QPushButton("测试连接")
        self.asr_test_result = QLabel()
        self.asr_test_result.setWordWrap(True)
        remote = QWidget()
        rl = QVBoxLayout(remote)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(_form(("常用服务", self.asr_preset), ("API 地址", self.asr_url), ("API Key", self.asr_key),
                           ("模型", self.asr_rmodel)))
        rl.addWidget(self.asr_rpartials)
        rl.addWidget(self.asr_test)
        rl.addWidget(self.asr_test_result)

        self.asr_stack = _Stack()
        self.asr_stack.addWidget(local)
        self.asr_stack.addWidget(remote)
        lay = QVBoxLayout(g)
        lay.addWidget(_form(("方式", self.asr_mode)))
        lay.addWidget(self.asr_stack)

        self.asr_mode.currentIndexChanged.connect(
            lambda: (self.asr_stack.setCurrentIndex(0 if self.asr_mode.currentData() == "local" else 1), self._emit()))
        self.asr_preset.currentIndexChanged.connect(self._fill_asr_preset)
        self.asr_test.clicked.connect(lambda: self._run_test("asr"))
        return g

    def _fill_asr_preset(self) -> None:
        if self._loading:
            return
        _n, url, model = ASR_REMOTE_PRESETS[self.asr_preset.currentData()]
        if url:
            self.asr_url.setText(url)
            self.asr_rmodel.setText(model)

    # ── translation ──────────────────────────────────────────────────────────

    def _llm_page(self, presets, with_models: bool):
        preset = _combo([(n, i) for i, (n, *_r) in enumerate(presets)])
        url, key, model = _line("https://…/v1"), _line("API Key", password=True), _combo([], editable=True)
        extra = _line('可选，JSON，例如 {"enable_thinking": false}')
        rows = [("常用服务", preset), ("API 地址", url), ("API Key", key), ("模型", model)]
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_form(*rows, ("额外参数", extra)))
        fetch = QPushButton("获取已安装的模型列表" if with_models else "获取模型列表")
        lay.addWidget(fetch)
        comp = model.completer()                       # type to filter a long list (SiliconFlow has 60+ chat models)
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

        side = "local" if with_models else "remote"

        def fill():
            if self._loading:
                return
            _n, u, m = presets[preset.currentData()]
            if u:
                url.setText(u)
                model.clear()                              # the old list belongs to the previous provider
                model.setEditText(m)
                self._auto_fetch(side)
        preset.currentIndexChanged.connect(fill)
        fetch.clicked.connect(lambda: self._fetch_models(side, manual=True))
        key.editingFinished.connect(lambda: self._auto_fetch(side))
        url.editingFinished.connect(lambda: self._auto_fetch(side))
        return w, url, key, model, extra, fetch, preset

    def _build_translate(self) -> QGroupBox:
        g = QGroupBox("④ 翻译")
        self.tr_mode = _combo(TR_MODES)

        (remote, self.trr_url, self.trr_key, self.trr_model, self.trr_extra, self.trr_fetch, self.trr_preset) = self._llm_page(LLM_REMOTE_PRESETS, False)
        (local, self.trl_url, self.trl_key, self.trl_model, self.trl_extra, self.trl_fetch, self.trl_preset) = self._llm_page(LLM_LOCAL_PRESETS, True)

        self.nllb_model = _combo(NLLB_MODELS, editable=True)
        nllb = QWidget()
        nl = QVBoxLayout(nllb)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(_form(("模型", self.nllb_model)))
        nl.addWidget(_hint("完全离线、延迟极低，但译文质量不及大模型。首次使用自动下载。"))

        self.deepl_key = _line("DeepL API Key", password=True)
        self.deepl_free = QCheckBox("免费版 Key（以 :fx 结尾）")
        deepl = QWidget()
        dl = QVBoxLayout(deepl)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(_form(("API Key", self.deepl_key)))
        dl.addWidget(self.deepl_free)

        self.tr_stack = _Stack()
        for w in (remote, local, nllb, deepl, QWidget()):
            self.tr_stack.addWidget(w)

        self.tr_test = QPushButton("测试翻译")
        self.tr_test_result = QLabel()
        self.tr_test_result.setWordWrap(True)

        self.topic = _line("例如：Kubernetes 技术分享 / 医学讲座")
        self.glossary = QPlainTextEdit()
        self.glossary.setPlaceholderText("术语表，每行一条：\nGPU = 显卡\nKubernetes\n（只写词表示保持原样）")
        self.glossary.setFixedHeight(76)
        self.tr_context = QSpinBox()
        self.tr_context.setRange(0, 12)
        self.tr_context.setToolTip("把前几句原文/译文作为上下文喂给模型，代词和术语更连贯")
        self.tr_draft = QCheckBox("说话时就出草稿译文（更快看到结果，会多消耗调用）")

        lay = QVBoxLayout(g)
        lay.addWidget(_form(("方式", self.tr_mode)))
        lay.addWidget(self.tr_stack)
        lay.addWidget(self.tr_test)
        lay.addWidget(self.tr_test_result)
        lay.addWidget(_form(("主题", self.topic), ("术语表", self.glossary), ("上下文句数", self.tr_context)))
        lay.addWidget(self.tr_draft)

        self.tr_mode.currentIndexChanged.connect(self._on_tr_mode)
        self.tr_test.clicked.connect(lambda: self._run_test("tr"))
        return g

    def _on_tr_mode(self, *_args, fetch: bool = True) -> None:
        modes = [m for _l, m in TR_MODES]
        mode = self.tr_mode.currentData()
        self.tr_stack.setCurrentIndex(modes.index(mode))
        self.tr_test.setVisible(mode in ("llm_remote", "llm_local", "deepl"))
        self.tr_test_result.clear()
        self._emit()
        if fetch and mode in ("llm_remote", "llm_local"):
            self._auto_fetch("remote" if mode == "llm_remote" else "local")

    def _side_widgets(self, side: str):
        return ((self.trr_url, self.trr_key, self.trr_model, self.trr_fetch) if side == "remote"
                else (self.trl_url, self.trl_key, self.trl_model, self.trl_fetch))

    def _auto_fetch(self, side: str) -> None:
        """Fetch the provider's model list when it can succeed and we don't already have it for these credentials."""
        if self._loading:
            return
        url, key, model, _btn = self._side_widgets(side)
        u, k = url.text().strip(), key.text().strip()
        if not u or (side == "remote" and not k):          # hosted APIs need a key; a local server does not
            return
        if self._fetched.get(side) == (u, k) and model.count():
            return
        self._fetch_models(side, manual=False)

    def _fetch_models(self, side: str, manual: bool) -> None:
        from ..translate.llm import list_models
        if side in self._fetching:
            return
        url, key, _model, btn = self._side_widgets(side)
        u, k = url.text().strip(), key.text().strip()
        self._fetching.add(side)
        btn.setEnabled(False)
        btn.setText("获取中…")
        if manual:
            self.tr_test_result.setText("正在获取模型列表…")
            self.tr_test_result.setStyleSheet("")

        def work():
            try:
                names = list_models(u, k)
                self._fetched[side] = (u, k)
                self.bridge.test_result.emit(f"models_{side}", True, "\n".join(names))
            except Exception as e:
                self.bridge.test_result.emit(f"models_{side}", False, ("!" if manual else "") + str(e))

        threading.Thread(target=work, daemon=True).start()

    # ── connection tests ─────────────────────────────────────────────────────

    def _run_test(self, which: str) -> None:
        cfg = self.collect()
        btn, out = (self.asr_test, self.asr_test_result) if which == "asr" else (self.tr_test, self.tr_test_result)
        btn.setEnabled(False)
        out.setText("测试中…")
        out.setStyleSheet("")

        def work():
            import time
            try:
                if which == "asr":
                    import numpy as np
                    from ..asr.remote import RemoteWhisper
                    r = RemoteWhisper(cfg.asr)
                    r.load()
                    t0 = time.perf_counter()
                    r.transcribe(np.zeros(16000, dtype=np.float32), None, final=True)
                    r.close()
                    self.bridge.test_result.emit("asr", True, f"连接正常，识别 1 秒音频耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
                else:
                    from ..translate import create_translator
                    tr = create_translator(cfg.translate)
                    t0 = time.perf_counter()
                    text = tr.translate("Hello, how are you today? The meeting starts at three.", "en", cfg.lang.target)
                    ms = (time.perf_counter() - t0) * 1000
                    self.bridge.test_result.emit("tr", True, f"{text}\n（{ms:.0f} ms，含首次连接）")
            except Exception as e:
                self.bridge.test_result.emit(which, False, str(e))

        threading.Thread(target=work, daemon=True).start()

    def _show_test(self, which: str, ok: bool, msg: str) -> None:
        if which.startswith("models_"):
            side = which[len("models_"):]
            _u, _k, model, btn = self._side_widgets(side)
            self._fetching.discard(side)
            btn.setEnabled(True)
            btn.setText("获取已安装的模型列表" if side == "local" else "获取模型列表")
            if ok:
                names, cur = msg.splitlines(), model.currentText()
                model.clear()
                model.addItems(names)
                model.setEditText(cur if cur or not names else names[0])   # never overwrite what the user chose/typed
                note = "" if cur in names or not cur else f"（当前模型 {cur} 不在列表中，请确认名称）"
                self.tr_test_result.setText(f"已获取 {len(names)} 个可用聊天模型，可在“模型”框里输入关键字过滤。{note}")
                self.tr_test_result.setStyleSheet("")
            else:
                self._fetched.pop(side, None)
                if msg.startswith("!"):                     # manual click -> show the error; automatic attempts stay quiet
                    self.tr_test_result.setText(msg[1:])
                    self.tr_test_result.setStyleSheet("color:#d33")
            return
        btn, out = (self.asr_test, self.asr_test_result) if which == "asr" else (self.tr_test, self.tr_test_result)
        btn.setEnabled(True)
        out.setText(("✓ " if ok else "✗ ") + msg)
        out.setStyleSheet("color:#2a9d4a" if ok else "color:#d33")

    # ── performance ──────────────────────────────────────────────────────────

    def _build_perf(self) -> QGroupBox:
        g = QGroupBox("⑤ 速度 / 准确度")
        self.preset = _combo(PRESET_CHOICES)
        self.perf_hint = _hint("速度优先：更短的停顿判定 + 草稿译文；准确优先：更大的束搜索与更长上下文。")
        self.save_tx = QCheckBox("保存字幕记录到文本文件")
        lay = QVBoxLayout(g)
        lay.addWidget(_form(("预设", self.preset)))
        lay.addWidget(self.perf_hint)
        lay.addWidget(self.save_tx)
        self.preset.currentIndexChanged.connect(self._on_preset)
        return g

    def _on_preset(self) -> None:
        if self._loading:
            return
        apply_preset(self._base, self.preset.currentData())
        self._loading = True
        self.tr_draft.setChecked(self._base.translate.draft)
        self.tr_context.setValue(self._base.translate.context_size)
        self._loading = False
        self._emit()

    # ── cfg <-> widgets ──────────────────────────────────────────────────────

    def _emit(self) -> None:
        if not self._loading:
            self.changed.emit()

    def load(self, cfg: AppConfig) -> None:
        self._loading = True
        self._base = copy.deepcopy(cfg)
        a, s, l, t = cfg.audio, cfg.asr, cfg.lang, cfg.translate
        _select(self.src_kind, a.source if a.source in ("system", "app", "mic") else "system")
        if a.app_key:
            self._set_apps([])                            # placeholder entry for the saved app until a scan finishes
            self._apps = []
        _select(self.src_lang, l.source)
        _select(self.tgt_lang, l.target)
        _select(self.asr_mode, s.mode)
        self.asr_stack.setCurrentIndex(0 if s.mode == "local" else 1)
        _select(self.asr_model, s.model)
        _select(self.asr_device, s.device)
        self.asr_url.setText(s.remote_base_url)
        self.asr_key.setText(s.remote_api_key)
        self.asr_rmodel.setText(s.remote_model)
        self.asr_rpartials.setChecked(s.remote_partials)
        _select(self.tr_mode, t.mode)
        for url, key, model, extra, ep in ((self.trr_url, self.trr_key, self.trr_model, self.trr_extra, t.remote),
                                           (self.trl_url, self.trl_key, self.trl_model, self.trl_extra, t.local)):
            url.setText(ep.base_url)
            key.setText(ep.api_key)
            model.setEditText(ep.model)
            extra.setText(ep.extra_body)
        for combo, presets, ep in ((self.trr_preset, LLM_REMOTE_PRESETS, t.remote), (self.trl_preset, LLM_LOCAL_PRESETS, t.local)):
            combo.setCurrentIndex(next((i for i, (_n, u, _m) in enumerate(presets) if u and u == ep.base_url), len(presets) - 1))
        _select(self.nllb_model, t.nllb_model)
        self.deepl_key.setText(t.deepl_key)
        self.deepl_free.setChecked(t.deepl_free)
        self.topic.setText(t.topic)
        self.glossary.setPlainText(t.glossary)
        self.tr_context.setValue(t.context_size)
        self.tr_draft.setChecked(t.draft)
        _select(self.preset, cfg.preset)
        self.save_tx.setChecked(cfg.save_transcript)
        self._loading = False
        self._on_src_kind()
        self._on_tr_mode(fetch=False)
        if t.mode == "llm_remote" and t.remote.api_key:
            self._auto_fetch("remote")             # app start with a saved key: populate the model list
        if a.source == "mic" and a.mic_device:
            _select(self.mic_combo, a.mic_device)

    def collect(self) -> AppConfig:
        cfg = copy.deepcopy(self._base)
        a, s, l, t = cfg.audio, cfg.asr, cfg.lang, cfg.translate
        a.source = self.src_kind.currentData()
        data = self.app_combo.currentData()
        if data:
            a.app_key, a.app_name = data
        a.mic_device = self.mic_combo.currentData() or ""
        l.source, l.target = self.src_lang.currentData(), self.tgt_lang.currentData()
        s.mode = self.asr_mode.currentData()
        s.model = _value(self.asr_model)
        s.device = self.asr_device.currentData()
        s.remote_base_url = self.asr_url.text().strip()
        s.remote_api_key = self.asr_key.text().strip()
        s.remote_model = self.asr_rmodel.text().strip()
        s.remote_partials = self.asr_rpartials.isChecked()
        t.mode = self.tr_mode.currentData()
        for url, key, model, extra, ep in ((self.trr_url, self.trr_key, self.trr_model, self.trr_extra, t.remote),
                                           (self.trl_url, self.trl_key, self.trl_model, self.trl_extra, t.local)):
            ep.base_url, ep.api_key = url.text().strip(), key.text().strip()
            ep.model, ep.extra_body = model.currentText().strip(), extra.text().strip()
        t.nllb_model = _value(self.nllb_model)
        t.deepl_key, t.deepl_free = self.deepl_key.text().strip(), self.deepl_free.isChecked()
        t.topic, t.glossary = self.topic.text().strip(), self.glossary.toPlainText().strip()
        t.context_size, t.draft = self.tr_context.value(), self.tr_draft.isChecked()
        cfg.preset = self.preset.currentData()
        cfg.save_transcript = self.save_tx.isChecked()
        return cfg

    def apply_general(self, cfg: AppConfig) -> None:
        """Take over the settings that are edited in the Preferences dialog (they live in the base config that
        ``collect()`` starts from, so they survive the next start / save)."""
        b = self._base
        b.models_dir, b.transcripts_dir, b.proxy, b.hf_endpoint = cfg.models_dir, cfg.transcripts_dir, cfg.proxy, cfg.hf_endpoint
        b.seg = copy.deepcopy(cfg.seg)
        b.asr.beam_final, b.asr.compute_type = cfg.asr.beam_final, cfg.asr.compute_type

    def set_running(self, running: bool) -> None:
        self.setEnabled(not running)          # settings are applied at start; stop first to change them
