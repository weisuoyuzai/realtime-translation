import time

import pytest

pytest.importorskip("PySide6")

from live_translator.config import AppConfig  # noqa: E402
from live_translator.models import Line  # noqa: E402
import live_translator.ui.main_window as mw  # noqa: E402
from live_translator.ui.bridge import Bridge  # noqa: E402
from live_translator.ui.main_window import MainWindow, validate  # noqa: E402
from live_translator.ui.overlay import SubtitleOverlay  # noqa: E402
from live_translator.ui.settings_panel import SettingsPanel  # noqa: E402
from live_translator.ui.transcript import TranscriptView  # noqa: E402


def test_default_config_survives_panel_roundtrip(qapp):
    """Regression: language combos had (label, code) swapped, silently turning the default target into English."""
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    panel.load(cfg)
    out = panel.collect()
    assert out.lang.target == "zh-Hans" and out.lang.source == "auto"
    assert out.asr == cfg.asr and out.translate == cfg.translate and out.audio.source == "system"


def test_panel_roundtrip_with_custom_values(qapp):
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.lang.source, cfg.lang.target = "ja", "zh-Hant"
    cfg.asr.mode, cfg.asr.model, cfg.asr.device = "remote", "small", "cpu"
    cfg.asr.remote_base_url, cfg.asr.remote_api_key, cfg.asr.remote_model = "http://h/v1", "k", "m"
    cfg.translate.mode = "llm_local"
    cfg.translate.local.base_url, cfg.translate.local.model = "http://127.0.0.1:1234/v1", "gemma"
    cfg.translate.glossary, cfg.translate.topic, cfg.translate.draft = "GPU = 显卡", "hardware", True
    cfg.translate.nllb_model = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"
    cfg.audio.source, cfg.audio.mic_device = "mic", ""
    panel.load(cfg)
    out = panel.collect()
    assert (out.lang.source, out.lang.target) == ("ja", "zh-Hant")
    assert (out.asr.mode, out.asr.model, out.asr.device) == ("remote", "small", "cpu")
    assert (out.asr.remote_base_url, out.asr.remote_api_key, out.asr.remote_model) == ("http://h/v1", "k", "m")
    assert out.translate.mode == "llm_local" and out.translate.local.model == "gemma"
    assert out.translate.glossary == "GPU = 显卡" and out.translate.topic == "hardware" and out.translate.draft
    assert out.translate.nllb_model.endswith("1.3B-ct2-int8")
    assert out.audio.source == "mic"


def test_custom_typed_model_name_is_kept(qapp):
    panel = SettingsPanel(Bridge())
    panel.load(AppConfig())
    panel.asr_model.setEditText("Systran/faster-distil-whisper-large-v3")
    assert panel.collect().asr.model == "Systran/faster-distil-whisper-large-v3"
    panel.asr_model.setCurrentIndex(panel.asr_model.findData("large-v3-turbo"))
    assert panel.collect().asr.model == "large-v3-turbo"


def test_saved_app_shown_even_when_not_running(qapp):
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.audio.source, cfg.audio.app_key, cfg.audio.app_name = "app", "zoom.exe", "Zoom"
    panel.load(cfg)
    panel._set_apps([])                                   # scan finished, Zoom not running
    out = panel.collect()
    assert out.audio.app_key == "zoom.exe" and out.audio.app_name == "Zoom"
    assert "未运行" in panel.app_combo.currentText()


def test_preset_updates_visible_controls(qapp):
    panel = SettingsPanel(Bridge())
    panel.load(AppConfig())
    panel.preset.setCurrentIndex(panel.preset.findData("fast"))
    assert panel.tr_draft.isChecked() and panel.tr_context.value() == 2
    out = panel.collect()
    assert out.seg.min_silence_ms == 300 and out.asr.beam_final == 1


def test_validate_messages():
    c = AppConfig()
    c.translate.mode = "none"
    assert validate(c) is None
    c.audio.source = "app"
    assert "应用" in validate(c)
    c.audio.source, c.asr.mode, c.asr.remote_model = "system", "remote", ""
    assert "远程识别" in validate(c)
    c.asr.mode, c.translate.mode = "local", "llm_remote"
    assert "API Key" in validate(c)
    c.translate.remote.api_key = "sk"
    assert validate(c) is None
    c.translate.mode, c.translate.local.base_url, c.translate.local.model = "llm_local", "http://127.0.0.1:11434/v1", ""
    assert "模型名" in validate(c)
    c.translate.mode = "deepl"
    assert "DeepL" in validate(c)


def test_transcript_upsert_replace_remove_and_text(qapp):
    v = TranscriptView()
    v.upsert(Line(1, src="Hello", src_final=True))
    v.upsert(Line(1, src="Hello there", dst="你好", src_final=True, dst_final=True, latency_ms=900, asr_ms=200, tr_ms=300))
    v.upsert(Line(2, src="noise", src_final=False))
    v.upsert(Line(2, removed=True))
    v._render()
    text = v.toPlainText()
    assert "Hello there" in text and "你好" in text and "noise" not in text and "延迟 0.9s" in text
    assert v.plain_text().count("→") == 1


def test_transcript_escapes_html(qapp):
    v = TranscriptView()
    v.upsert(Line(1, src="<b>bold</b> & more", dst="<script>x</script>", src_final=True, dst_final=True))
    v._render()
    assert "<b>bold</b> & more" in v.toPlainText() and "<script>x</script>" in v.toPlainText()


def test_overlay_shows_translation_else_recognised_text(qapp):
    o = SubtitleOverlay(AppConfig().overlay)
    o.move(-6000, 0)
    o.show_line(Line(1, src="Hello there", src_final=True))
    assert o._dst.text() == "Hello there" and o._src.text() == ""       # nothing to show above yet
    o.show_line(Line(1, src="Hello there", dst="你好", src_final=True, dst_final=True))
    assert o._dst.text() == "你好" and o._src.text() == "Hello there"
    o.cfg.show_source = False
    o.show_line(Line(1, src="Hello there", dst="你好", src_final=True, dst_final=True))
    assert o._src.text() == ""
    o.show_line(Line(1, removed=True))
    assert o._dst.text() == ""


class FakePipeline:
    instances = []

    def __init__(self, cfg, on_line, on_status, on_level=None, source=None):
        self.on_line, self.on_status, self.on_level = on_line, on_status, on_level
        self.stopped = False
        FakePipeline.instances.append(self)

    def start(self):
        self.on_status("loading", "loading…")
        self.on_status("ready", "监听中 · fake")

    def stop(self):
        self.stopped = True


def pump(app, seconds):
    """Run the Qt event loop for a while WITHOUT starving worker threads (QTest.qWait keeps the GIL while it waits)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_until(app, cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mw, "Pipeline", FakePipeline)
    FakePipeline.instances.clear()
    cfg = AppConfig()
    cfg.translate.mode = "none"
    w = MainWindow(cfg)
    w.move(-6000, 0)
    w.show()
    yield w
    w.close()


def test_start_stop_lifecycle_and_live_lines(qapp, window):
    w = window
    w.btn.click()
    assert wait_until(qapp, lambda: w.state == "running")
    assert w.btn.text().endswith("停止") and not w.panel.isEnabled()
    pipe = FakePipeline.instances[-1]
    pipe.on_line(Line(1, src="Hello", dst="你好", src_final=True, dst_final=True, latency_ms=1200, asr_ms=1, tr_ms=1))
    pipe.on_level(0.1)
    assert wait_until(qapp, lambda: "1.2s" in w.stats.text())
    w.transcript._render()
    assert "你好" in w.transcript.toPlainText() and w.meter.value() > 0

    w.btn.click()
    assert wait_until(qapp, lambda: w.state == "idle")
    assert pipe.stopped and w.btn.text().endswith("开始") and w.panel.isEnabled()


def test_invalid_config_blocks_start_with_a_message(qapp, window, monkeypatch):
    shown = []
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a: shown.append(a[2]))
    window.panel.src_kind.setCurrentIndex(window.panel.src_kind.findData("app"))
    window.panel.app_combo.clear()
    window.btn.click()
    assert shown and "应用" in shown[0]
    assert window.state == "idle" and not FakePipeline.instances


def test_boot_failure_resets_ui_and_alerts(qapp, window, monkeypatch):
    alerts = []
    monkeypatch.setattr(mw.QMessageBox, "critical", lambda *a: alerts.append(a[2]))

    def failing_start(self):
        self.on_status("error", "模型下载失败")
        self.on_status("stopped", "")

    monkeypatch.setattr(FakePipeline, "start", failing_start)
    window.btn.click()
    assert wait_until(qapp, lambda: window.state == "idle")
    assert alerts == ["模型下载失败"] and window.btn.text().endswith("开始") and window.panel.isEnabled()


def test_config_is_saved_on_start(qapp, window):
    from live_translator.config import load_config
    window.panel.tgt_lang.setCurrentIndex(window.panel.tgt_lang.findData("ja"))
    window.btn.click()
    assert wait_until(qapp, lambda: window.state == "running")
    assert load_config().lang.target == "ja"


def test_siliconflow_is_the_default_remote_llm_and_presets_follow_the_loaded_config(qapp):
    cfg = AppConfig()
    assert cfg.translate.remote.base_url == "https://api.siliconflow.cn/v1"
    assert cfg.translate.remote.model == "Qwen/Qwen3.6-35B-A3B"
    panel = SettingsPanel(Bridge())
    panel.load(cfg)
    assert "SiliconFlow" in panel.trr_preset.currentText()
    assert panel.collect().translate.remote.model == "Qwen/Qwen3.6-35B-A3B"
    cfg.translate.remote.base_url = "https://my.proxy/v1"
    cfg.translate.local.base_url = "http://127.0.0.1:1234/v1"
    panel.load(cfg)
    assert panel.trr_preset.currentText() == "自定义" and panel.trl_preset.currentText() == "LM Studio"
    assert panel.collect().translate.remote.base_url == "https://my.proxy/v1"       # showing a preset never rewrites the URL


def _wait(app, cond, timeout=4.0):
    return wait_until(app, cond, timeout)


def test_model_list_is_fetched_automatically_once_a_key_is_present(qapp, server):
    server.models = ["Qwen/Qwen3.6-35B-A3B", "BAAI/bge-m3", "deepseek-ai/DeepSeek-V3"]
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.translate.remote.base_url, cfg.translate.remote.api_key = server.base_url, "sk-test"
    cfg.translate.remote.model = "my-custom-model"
    panel.load(cfg)                                                   # saved key -> list is fetched on start-up
    assert _wait(qapp, lambda: panel.trr_model.count() == 2)
    assert [panel.trr_model.itemText(i) for i in range(2)] == ["Qwen/Qwen3.6-35B-A3B", "deepseek-ai/DeepSeek-V3"]
    assert panel.trr_model.currentText() == "my-custom-model"          # the user's choice is never overwritten
    assert "已获取 2 个" in panel.tr_test_result.text() and "不在列表中" in panel.tr_test_result.text()
    n = len(server.models_queries)
    panel._auto_fetch("remote")                                        # same credentials -> no repeat request
    pump(qapp, 0.15)
    assert len(server.models_queries) == n


def test_no_key_means_no_automatic_request_for_hosted_apis(qapp, server):
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.translate.remote.base_url = server.base_url
    panel.load(cfg)
    panel._auto_fetch("remote")
    pump(qapp, 0.15)
    assert server.models_queries == [] and panel.trr_model.count() == 0


def test_changing_key_refetches_and_manual_failures_are_shown(qapp, server):
    panel = SettingsPanel(Bridge())
    cfg = AppConfig()
    cfg.translate.remote.base_url = server.base_url
    panel.load(cfg)
    panel.trr_key.setText("sk-one")
    panel.trr_key.editingFinished.emit()                                # user finished typing the key
    assert _wait(qapp, lambda: panel.trr_model.count() == 2)
    server.models_status = 401
    panel.trr_key.setText("sk-bad")
    panel.trr_key.editingFinished.emit()                                # automatic attempt fails -> stays quiet
    assert _wait(qapp, lambda: not panel._fetching)                     # the automatic attempt has finished (and failed)
    assert "Key" not in panel.tr_test_result.text()
    panel.trr_fetch.click()                                             # manual click -> the error is shown
    assert _wait(qapp, lambda: "API Key 无效" in panel.tr_test_result.text())
    assert panel.trr_fetch.isEnabled() and panel.trr_fetch.text() == "获取模型列表"


def test_choosing_a_preset_replaces_the_stale_model_list(qapp, server):
    panel = SettingsPanel(Bridge())
    panel.load(AppConfig())
    panel.trr_model.addItems(["old-provider-model"])
    panel.trr_preset.setCurrentIndex(panel.trr_preset.findText("DeepSeek"))
    assert panel.trr_model.count() == 0 and panel.trr_model.currentText() == "deepseek-chat"
    assert panel.trr_url.text() == "https://api.deepseek.com/v1"
