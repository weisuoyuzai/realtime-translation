import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

import live_translator.ui.main_window as mw  # noqa: E402
import live_translator.ui.preferences as prefs  # noqa: E402
from live_translator import runtime, storage  # noqa: E402
from live_translator.config import AppConfig, load_config  # noqa: E402
from live_translator.ui.main_window import MainWindow  # noqa: E402
from live_translator.ui.preferences import PreferencesDialog  # noqa: E402
from test_settings_backend import make_model  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    # A test that opens a real modal box would block forever; fail fast instead.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: pytest.fail("unexpected modal dialog: " + self.text()))
    yield
    runtime.set_models_dir("")
    runtime._apply_proxy("")


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_defaults_roundtrip_unchanged(qapp):
    cfg = AppConfig()
    dlg = PreferencesDialog(cfg)
    out = dlg.apply_to(cfg)
    assert out == cfg                                             # opening and saving without edits changes nothing


def test_custom_values_are_applied(qapp):
    cfg = AppConfig()
    dlg = PreferencesDialog(cfg)
    dlg.models_dir.setText(r"D:\ai\models")
    dlg.transcripts_dir.setText(r"D:\ai\subs")
    dlg.proxy.setText("socks5://127.0.0.1:1080")
    dlg.mirror.setCurrentIndex(dlg.mirror.findData("https://hf-mirror.com"))
    dlg.ov_font.setValue(40)
    dlg.ov_opacity.setValue(55)
    dlg.ov_source.setChecked(False)
    dlg.ov_click.setChecked(True)
    dlg.vad.setValue(0.65)
    dlg.min_sil.setValue(600)
    dlg.max_utt.setValue(9)
    dlg.partial.setValue(900)
    dlg.beam.setValue(5)
    dlg.compute.setCurrentIndex(dlg.compute.findData("int8_float16"))
    out = dlg.apply_to(cfg)
    assert (out.models_dir, out.transcripts_dir, out.proxy) == (r"D:\ai\models", r"D:\ai\subs", "socks5://127.0.0.1:1080")
    assert out.hf_endpoint == "https://hf-mirror.com"
    assert (out.overlay.font_size, out.overlay.opacity, out.overlay.show_source, out.overlay.click_through) == (40, 0.55, False, True)
    assert (out.seg.vad_threshold, out.seg.min_silence_ms, out.seg.max_utterance_s, out.seg.partial_interval_ms) == (0.65, 600, 9.0, 900)
    assert (out.asr.beam_final, out.asr.compute_type) == (5, "int8_float16")
    assert cfg == AppConfig()                                     # the input config is never mutated


def test_custom_mirror_url_roundtrips(qapp):
    cfg = AppConfig()
    cfg.hf_endpoint = "https://my.mirror.example/"
    dlg = PreferencesDialog(cfg)
    assert dlg.apply_to(cfg).hf_endpoint == "https://my.mirror.example"      # trailing slash trimmed
    cfg.hf_endpoint = "https://hf-mirror.com"
    assert "国内镜像" in PreferencesDialog(cfg).mirror.currentText()


def test_model_list_shows_what_is_in_the_chosen_folder(qapp, tmp_path):
    make_model(tmp_path, "Systran/faster-whisper-small", 1000)
    make_model(tmp_path, "JustFrederik/nllb-200-distilled-600M-ct2-int8", 4000)
    cfg = AppConfig()
    cfg.models_dir = str(tmp_path)
    dlg = PreferencesDialog(cfg)
    assert dlg.models_tree.topLevelItemCount() == 2 and "共 2 个模型" in dlg.models_total.text()
    dlg.models_dir.setText(str(tmp_path / "empty"))
    assert dlg.models_tree.topLevelItemCount() == 0 and "还没有模型" in dlg.models_total.text()
    dlg.models_dir.clear()                                          # "" = the default cache directory
    assert dlg.models_dir.placeholderText() == runtime.default_models_dir()


def test_delete_selected_models(qapp, tmp_path, monkeypatch):
    make_model(tmp_path, "a/big", 5000)
    make_model(tmp_path, "a/small", 10)
    cfg = AppConfig()
    cfg.models_dir = str(tmp_path)
    dlg = PreferencesDialog(cfg)
    dlg.models_tree.topLevelItem(0).setSelected(True)               # the biggest one
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.No)
    dlg._delete_selected()
    assert len(storage.list_models(str(tmp_path))) == 2             # declined → nothing happens
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    dlg._delete_selected()
    assert [m.repo_id for m in storage.list_models(str(tmp_path))] == ["a/small"]
    assert dlg.models_tree.topLevelItemCount() == 1


def test_unusable_directory_blocks_saving(qapp, tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x")
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(a[2]))
    dlg = PreferencesDialog(AppConfig())
    dlg.models_dir.setText(str(f))
    dlg.accept()
    assert shown and "模型存放位置" in shown[0] and dlg.result() != dlg.DialogCode.Accepted


def test_saving_creates_missing_folders_and_returns_the_config(qapp, tmp_path):
    dlg = PreferencesDialog(AppConfig())
    dlg.models_dir.setText(str(tmp_path / "new" / "models"))
    dlg.transcripts_dir.setText(str(tmp_path / "new" / "subs"))
    dlg.accept()
    assert dlg.result() == dlg.DialogCode.Accepted
    assert (tmp_path / "new" / "models").is_dir() and (tmp_path / "new" / "subs").is_dir()
    assert dlg.result_config().models_dir == str(tmp_path / "new" / "models")


def test_changing_folder_offers_to_move_models_and_cancel_keeps_the_dialog_open(qapp, tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    make_model(old, "a/one", 100)
    cfg = AppConfig()
    cfg.models_dir = str(old)
    dlg = PreferencesDialog(cfg)
    dlg.models_dir.setText(str(new))
    asked = []
    monkeypatch.setattr(PreferencesDialog, "_offer_move", lambda self, o, n: asked.append((o, n)) or False)
    dlg.accept()
    assert asked == [(str(old), str(new))] and dlg.result() != dlg.DialogCode.Accepted
    monkeypatch.setattr(PreferencesDialog, "_offer_move", lambda self, o, n: True)
    dlg.accept()
    assert dlg.result() == dlg.DialogCode.Accepted


def test_no_move_offer_when_nothing_to_move_or_folder_unchanged(qapp, tmp_path):
    dlg = PreferencesDialog(AppConfig())
    assert dlg._offer_move(str(tmp_path / "empty_old"), str(tmp_path / "new")) is True     # nothing to offer
    cfg = AppConfig()
    cfg.models_dir = str(tmp_path)
    dlg = PreferencesDialog(cfg)
    dlg.accept()                                                    # unchanged folder → accepted without any question
    assert dlg.result() == dlg.DialogCode.Accepted


def test_move_models_with_progress_dialog(qapp, tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    make_model(old, "a/one", 100)
    make_model(old, "a/two", 100)
    dlg = PreferencesDialog(AppConfig())
    assert dlg._move_models(str(old), str(new)) is True
    assert {m.repo_id for m in storage.list_models(str(new))} == {"a/one", "a/two"}
    assert storage.list_models(str(old)) == []


def test_move_failure_is_reported(qapp, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(a[2:4]))
    monkeypatch.setattr(storage, "move_models", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    dlg = PreferencesDialog(AppConfig())
    assert dlg._move_models(str(tmp_path / "a"), str(tmp_path / "b")) is False
    assert shown and "disk full" in shown[0][0]


def test_storage_is_locked_while_running(qapp, tmp_path, monkeypatch):
    make_model(tmp_path, "a/one", 100)
    cfg = AppConfig()
    cfg.models_dir = str(tmp_path)
    dlg = PreferencesDialog(cfg, running=True)
    assert not dlg.models_dir.isEnabled() and not dlg.transcripts_dir.isEnabled()
    assert not dlg.storage_note.isHidden() or dlg.storage_note.isVisibleTo(dlg)
    dlg.models_dir.setEnabled(True)
    dlg.models_dir.setText(str(tmp_path / "elsewhere"))
    monkeypatch.setattr(PreferencesDialog, "_offer_move", lambda *a: pytest.fail("must not move models while running"))
    dlg.accept()                                                    # path may change, but nothing is moved
    assert dlg.result() == dlg.DialogCode.Accepted


def test_reset_button_restores_advanced_defaults(qapp):
    cfg = AppConfig()
    cfg.seg.min_silence_ms, cfg.asr.beam_final = 900, 7
    dlg = PreferencesDialog(cfg)
    assert dlg.min_sil.value() == 900
    dlg._set_advanced(prefs.SegmenterCfg(), prefs.AsrCfg())
    assert dlg.min_sil.value() == 450 and dlg.beam.value() == 3


# ── main window: menus and applying preferences ──────────────────────────────

@pytest.fixture
def window(qapp):
    cfg = AppConfig()
    cfg.translate.mode = "none"
    w = MainWindow(cfg)
    w.move(-6000, 0)
    w.show()
    yield w
    w.close()


def menu_titles(w):
    return [a.text() for a in w.menuBar().actions()]


def actions_of(w, title_part):
    menu = next(a.menu() for a in w.menuBar().actions() if title_part in a.text())
    return {a.text(): a for a in menu.actions() if a.text()}


def test_menu_bar_structure(qapp, window):
    assert menu_titles(window) == ["文件(&F)", "设置(&S)", "视图(&V)", "帮助(&H)"]
    assert {"导出字幕…", "清空字幕", "退出", "打开字幕记录文件夹"} <= set(actions_of(window, "文件"))
    assert {"偏好设置…", "模型管理（存放位置 / 删除）…", "打开模型文件夹"} <= set(actions_of(window, "设置"))
    assert actions_of(window, "设置")["偏好设置…"].shortcut().toString() == "Ctrl+,"


def test_view_menu_stays_in_sync_with_the_toolbar(qapp, window):
    view = actions_of(window, "视图")
    window.overlay_cb.setChecked(True)
    assert view["悬浮字幕"].isChecked() and window.overlay.isVisible()
    view["悬浮字幕"].setChecked(False)
    assert not window.overlay_cb.isChecked() and not window.overlay.isVisible()
    view["字幕条鼠标穿透"].setChecked(True)
    assert window.click_cb.isChecked() and window.overlay.cfg.click_through
    window.click_cb.setChecked(False)
    assert not view["字幕条鼠标穿透"].isChecked()
    view["悬浮字幕显示原文"].setChecked(False)
    assert window.overlay.cfg.show_source is False


def test_apply_preferences_updates_everything_and_persists(qapp, window, tmp_path):
    new = window.current_config()
    new.models_dir, new.transcripts_dir = str(tmp_path / "m"), str(tmp_path / "t")
    new.proxy, new.hf_endpoint = "http://127.0.0.1:7890", "https://hf-mirror.com"
    new.overlay.font_size, new.overlay.opacity, new.overlay.show_source, new.overlay.click_through = 44, 0.5, False, True
    new.seg.min_silence_ms, new.asr.beam_final, new.asr.compute_type = 700, 5, "int8"
    window.apply_preferences(new)

    assert runtime.models_dir() == str(tmp_path / "m")
    assert window.overlay.cfg.font_size == 44 and window.overlay.cfg.opacity == 0.5
    assert window.click_cb.isChecked() and not actions_of(window, "视图")["悬浮字幕显示原文"].isChecked()
    got = window.current_config()                                    # what the next run / save will use
    assert (got.models_dir, got.transcripts_dir, got.proxy) == (str(tmp_path / "m"), str(tmp_path / "t"), "http://127.0.0.1:7890")
    assert (got.seg.min_silence_ms, got.asr.beam_final, got.asr.compute_type) == (700, 5, "int8")
    saved = load_config()
    assert saved.models_dir == str(tmp_path / "m") and saved.overlay.font_size == 44 and saved.asr.compute_type == "int8"


def test_preferences_survive_starting_a_run(qapp, window, tmp_path, monkeypatch):
    """The pipeline must be built from a config that still contains the preferences (panel.collect keeps them)."""
    built = []

    class FakePipeline:
        def __init__(self, cfg, *a, **k):
            built.append(cfg)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(mw, "Pipeline", FakePipeline)
    new = window.current_config()
    new.models_dir, new.seg.max_utterance_s = str(tmp_path), 20.0
    window.apply_preferences(new)
    window.btn.click()
    assert built and built[0].models_dir == str(tmp_path) and built[0].seg.max_utterance_s == 20.0


def test_open_prefs_applies_only_when_saved(qapp, window, monkeypatch):
    applied = []
    monkeypatch.setattr(window, "apply_preferences", applied.append)
    monkeypatch.setattr(PreferencesDialog, "exec", lambda self: PreferencesDialog.DialogCode.Rejected)
    window._open_prefs()
    assert applied == []

    def fake_exec(self):
        self._result = self.apply_to(self._orig)
        return PreferencesDialog.DialogCode.Accepted

    monkeypatch.setattr(PreferencesDialog, "exec", fake_exec)
    window._open_prefs(prefs.TAB_ADVANCED)
    assert len(applied) == 1


def test_open_folder_actions_use_the_configured_paths(qapp, window, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(mw, "open_folder", opened.append)
    new = window.current_config()
    new.models_dir, new.transcripts_dir = str(tmp_path / "m"), str(tmp_path / "t")
    window.apply_preferences(new)
    actions_of(window, "设置")["打开模型文件夹"].trigger()
    actions_of(window, "文件")["打开字幕记录文件夹"].trigger()
    assert opened == [str(tmp_path / "m"), str(tmp_path / "t")]


def test_about_box_mentions_paths(qapp, window, monkeypatch):
    texts = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: texts.append(text))
    actions_of(window, "帮助")["关于 Live Translator"].trigger()
    assert texts and "模型目录" in texts[0] and runtime.effective_models_dir() in texts[0]
