"""Application settings: plain dataclasses persisted as JSON."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("LIVE_TRANSLATOR_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "LiveTranslator"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "LiveTranslator"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "LiveTranslator"
    base.mkdir(parents=True, exist_ok=True)
    return base


def config_path() -> Path:
    return config_dir() / "config.json"


# ── sections ─────────────────────────────────────────────────────────────────

@dataclass
class AudioCfg:
    source: str = "system"        # system | app | mic | file
    app_key: str = ""             # Windows: exe name (chrome.exe); macOS: bundle id
    app_name: str = ""            # display only
    mic_device: str = ""          # sounddevice device name; "" = default
    file_path: str = ""           # for source == "file"


@dataclass
class AsrCfg:
    mode: str = "local"           # local | remote
    model: str = "auto"           # auto → picked from hardware
    device: str = "auto"          # auto | cuda | cpu
    compute_type: str = "auto"
    beam_final: int = 3
    partials: bool = True         # show live partial text while speaking
    # remote (OpenAI-compatible /audio/transcriptions)
    remote_base_url: str = "https://api.openai.com/v1"
    remote_api_key: str = ""
    remote_model: str = "whisper-1"
    remote_partials: bool = False  # partials cost an API call each; off by default


@dataclass
class LangCfg:
    source: str = "auto"
    target: str = "zh-Hans"


@dataclass
class LlmEndpoint:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    extra_body: str = ""          # optional JSON merged into the request body


@dataclass
class TranslateCfg:
    mode: str = "llm_remote"      # llm_remote | llm_local | nllb | deepl | none
    remote: LlmEndpoint = field(default_factory=lambda: LlmEndpoint(
        base_url="https://api.siliconflow.cn/v1", model="Qwen/Qwen3.6-35B-A3B"))
    local: LlmEndpoint = field(default_factory=lambda: LlmEndpoint(
        base_url="http://127.0.0.1:11434/v1", api_key="ollama", model="qwen2.5:7b"))
    nllb_model: str = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    deepl_key: str = ""
    deepl_free: bool = True
    glossary: str = ""            # one "term = translation" (or just "term") per line
    topic: str = ""               # free-text hint: "Kubernetes conference talk"
    context_size: int = 4         # previous sentences fed back as few-shot context
    draft: bool = False           # translate partials while the speaker is still talking
    correct_source: bool = False  # ask the LLM to silently fix likely ASR mis-hearings in the displayed original
                                   # text too (not just the translation); one extra request per sentence, so off
                                   # by default. Only takes effect with an LLM translator (llm_remote / llm_local).


@dataclass
class SegmenterCfg:
    vad_threshold: float = 0.5
    min_silence_ms: int = 450
    max_utterance_s: float = 12.0
    partial_interval_ms: int = 700
    preroll_ms: int = 240
    min_speech_ms: int = 250


@dataclass
class SpeakerCfg:
    enabled: bool = False         # tell speakers apart by voice (downloads a 28 MB model on first use)
    similarity: float = 0.55      # cosine similarity that counts as "same person": higher = splits more, lower = merges
    max_speakers: int = 0         # 0 = no limit


@dataclass
class OverlayCfg:
    enabled: bool = False
    show_source: bool = True
    max_sentences: int = 2        # sentences on screen at once (1 = only the current one); earlier sentences stay
                                  # above it, dimmer, and the oldest drop out first
    auto_height: bool = True      # bar height follows the content; False = keep the height you dragged it to (h)
    learning: bool = False        # learning mode (also affects the main window): ruby (pinyin / furigana / romanization / IPA) + speaker
    ruby_scope: str = "both"      # what gets ruby in learning mode: dst (translation) | src (original) | both
    tts_read: str = "dst"         # what the speaker button reads aloud: dst (translation) | src (original)
    font_size: int = 26
    opacity: float = 0.72
    click_through: bool = False
    x: int = -1
    y: int = -1
    w: int = 900
    h: int = 150


def ruby_wanted(o: OverlayCfg, translation: bool) -> bool:
    """Whether learning mode annotates this text: the translation (``translation=True``) or the original."""
    return o.learning and o.ruby_scope in ("both", "dst" if translation else "src")


@dataclass
class AppConfig:
    audio: AudioCfg = field(default_factory=AudioCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    lang: LangCfg = field(default_factory=LangCfg)
    translate: TranslateCfg = field(default_factory=TranslateCfg)
    seg: SegmenterCfg = field(default_factory=SegmenterCfg)
    speaker: SpeakerCfg = field(default_factory=SpeakerCfg)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    preset: str = "balanced"      # fast | balanced | accurate
    hf_endpoint: str = ""         # e.g. https://hf-mirror.com for regions where huggingface.co is slow
    save_transcript: bool = True
    models_dir: str = ""          # where downloaded Whisper / NLLB models live; "" = the standard Hugging Face cache
    transcripts_dir: str = ""     # where subtitle logs are written; "" = <config dir>/transcripts
    proxy: str = ""               # http://host:port or socks5://host:port for API calls and model downloads


# Presets trade latency against accuracy; they only touch the knobs that matter.
PRESETS: dict[str, dict] = {
    "fast":     {"seg": {"min_silence_ms": 300, "partial_interval_ms": 500, "max_utterance_s": 8.0},
                 "asr": {"beam_final": 1}, "translate": {"draft": True, "context_size": 2}},
    "balanced": {"seg": {"min_silence_ms": 450, "partial_interval_ms": 700, "max_utterance_s": 12.0},
                 "asr": {"beam_final": 3}, "translate": {"context_size": 4}},
    "accurate": {"seg": {"min_silence_ms": 700, "partial_interval_ms": 900, "max_utterance_s": 18.0},
                 "asr": {"beam_final": 5}, "translate": {"draft": False, "context_size": 6}},
}


def apply_preset(cfg: AppConfig, name: str) -> None:
    preset = PRESETS.get(name)
    if not preset:
        return
    cfg.preset = name
    for section, values in preset.items():
        target = getattr(cfg, section)
        for k, v in values.items():
            setattr(target, k, v)


# ── (de)serialisation ────────────────────────────────────────────────────────

def _from_dict(cls, data):
    """Build dataclass ``cls`` from ``data``, ignoring unknown keys and keeping defaults for missing ones."""
    if not isinstance(data, dict):
        return cls()
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v, t = data[f.name], hints[f.name]
        if dataclasses.is_dataclass(t):
            kwargs[f.name] = _from_dict(t, v)
            continue
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(t, type) and not isinstance(v, t):
            continue                      # wrong type in a hand-edited file → keep the default
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: Path | None = None) -> AppConfig:
    path = path or config_path()
    try:
        return _from_dict(AppConfig, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return AppConfig()


def save_config(cfg: AppConfig, path: Path | None = None) -> None:
    path = path or config_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)   # holds API keys
    except OSError:
        pass
    tmp.replace(path)
