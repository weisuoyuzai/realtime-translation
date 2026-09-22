"""Real-time pipeline: audio → VAD/segmenter → ASR → translation → subtitle lines.

Threads (all daemon):
  capture callback (owned by the audio source)  → bounded queue, never blocks the OS audio thread
  segmenter   resample + Silero VAD + utterance state machine
  asr         one worker; interim (partial) requests are "latest wins" so it never falls behind; when speaker
              tracking is on, each final utterance also gets a voiceprint here (~50 ms) → Line.speaker
  translate   final translations, in order, streamed; keeps the running conversation context
  draft       optional: translates interim text while the speaker is still talking; cancelled by finals
"""
from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .asr import AsrError, LanguageTracker, Recognizer, create_recognizer
from .audio import AudioSource, AudioSourceError, create_source
from .config import AppConfig, config_dir
from .glossary import asr_hint, parse_glossary
from .languages import base_lang, normalize_lang, same_language
from .models import Line
from .resample import Resampler
from .runtime import models_dir
from .speaker import MIN_EMBED_S, SpeakerEmbedder, SpeakerTracker, load_embedder
from .segmenter import SegEvent, Segmenter
from .text_utils import guess_language
from .translate import TranslateError, Translator, create_translator
from .vad import SileroVAD

log = logging.getLogger(__name__)

StatusCallback = Callable[[str, str], None]        # (level: info|loading|ready|warn|error, message)
LineCallback = Callable[[Line], None]

_MIN_DRAFT_CHARS = 6
_MAX_UTTERANCE_HARD_CAP_S = 28.0                   # Whisper's window is 30 s
_KEEP_LINES = 400
_LOW_CONFIDENCE = 0.28                             # below this, flag the line instead of silently trusting it
_LOW_CONFIDENCE_MIN_CHARS = 4                       # don't flag short utterances: noisy scoring, low stakes


class _AsrQueue:
    """FIFO with stale-work elimination: only the newest interim request is ever kept."""

    def __init__(self):
        self._items: list[tuple[str, SegEvent]] = []
        self._cv = threading.Condition()

    def put(self, kind: str, ev: SegEvent) -> None:
        with self._cv:
            if kind == "partial":
                self._items = [i for i in self._items if i[0] != "partial"]
            else:
                self._items = [i for i in self._items if not (i[0] == "partial" and i[1].utt_id == ev.utt_id)]
            self._items.append((kind, ev))
            self._cv.notify()

    def get(self, timeout: float) -> tuple[str, SegEvent] | None:
        with self._cv:
            if not self._items:
                self._cv.wait(timeout)
            return self._items.pop(0) if self._items else None


class _TranscriptWriter:
    def __init__(self, enabled: bool, directory: str = ""):
        self._f = None
        self.path: Path | None = None
        if enabled:
            d = Path(directory) if directory else config_dir() / "transcripts"
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{datetime.now():%Y%m%d-%H%M%S}.txt"
            self._f = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, line: Line) -> None:
        if self._f is None:
            return
        with self._lock:
            ts = datetime.fromtimestamp(line.created).strftime("%H:%M:%S")
            who = f"[说话人 {line.speaker}] " if line.speaker else ""
            warn = "⚠ " if line.uncertain_reason else ""
            self._f.write(f"[{ts}] ({line.src_lang or '?'}) {who}{warn}{line.src}\n")
            if line.dst:
                self._f.write(f"           → {line.dst}\n")
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()


class Pipeline:
    def __init__(self, cfg: AppConfig, on_line: LineCallback, on_status: StatusCallback,
                 on_level: Callable[[float], None] | None = None, source: AudioSource | None = None):
        self.cfg = cfg
        self._on_line, self._on_status, self._on_level = on_line, on_status, on_level
        self._source = source

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._audio_q: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=400)
        self._asr_q = _AsrQueue()
        self._tr_q: queue.Queue[tuple[int, str, str, float] | None] = queue.Queue()
        self._cor_q: queue.Queue[tuple[int, str, str] | None] = queue.Queue()

        self._lines: dict[int, Line] = {}
        self._lock = threading.RLock()
        self._context: deque[tuple[str, str]] = deque(maxlen=max(0, cfg.translate.context_size))
        self._tracker = LanguageTracker()

        self._embedder: SpeakerEmbedder | None = None
        self._speakers = SpeakerTracker(cfg.speaker.similarity, cfg.speaker.max_speakers)
        self._recognizer: Recognizer | None = None
        self._translator: Translator | None = None
        self._segmenter: Segmenter | None = None
        self._writer: _TranscriptWriter | None = None

        self._draft_cv = threading.Condition()
        self._draft_job: tuple[int, str, str] | None = None
        self._draft_cancel: tuple[int, threading.Event] | None = None
        self._draft_last: dict[int, str] = {}

        self._last_err: dict[str, float] = {}
        self._last_level = 0.0
        self._audio_dropped = 0
        self._got_audio = False
        self.running = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Non-blocking: model loading happens on a background thread and is reported via ``on_status``."""
        t = threading.Thread(target=self._boot, name="boot", daemon=True)
        self._threads.append(t)
        t.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        src, self._source = self._source, None
        if src is not None:
            try:
                src.stop()
            except Exception:
                log.exception("stopping audio source")
        with self._draft_cv:
            self._draft_cv.notify_all()
        if self._draft_cancel:
            self._draft_cancel[1].set()
        self._tr_q.put(None)
        self._cor_q.put(None)
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2.0)
        alive = [t.name for t in self._threads if t.is_alive()]
        if not alive:                                   # only release models nobody is still using
            for obj in (self._recognizer, self._translator):
                try:
                    obj and obj.close()
                except Exception:
                    log.exception("closing %r", obj)
        if self._writer:
            self._writer.close()

    def _boot(self) -> None:
        cfg = self.cfg
        try:
            self._status("loading", "正在准备识别引擎…")
            self._recognizer = create_recognizer(cfg.asr)
            self._recognizer.load(lambda m: self._status("loading", m))
            if self._stop.is_set():
                return

            tr_cfg = cfg.translate
            self._status("loading", "正在连接翻译服务…")
            self._translator = create_translator(tr_cfg, lambda m: self._status("loading", m))
            if self._translator is not None:
                self._translator.warmup()
            if self._stop.is_set():
                return

            if cfg.speaker.enabled:
                try:
                    self._embedder = load_embedder(lambda m: self._status("loading", m), models_dir())
                except Exception as e:
                    log.exception("speaker model failed to load")
                    self._status("warn", f"说话人识别模型加载失败，本次不区分说话人：{e}")
            if self._stop.is_set():
                return

            self._segmenter = Segmenter(SileroVAD(), dataclasses.replace(
                cfg.seg, max_utterance_s=min(cfg.seg.max_utterance_s, _MAX_UTTERANCE_HARD_CAP_S)))
            self._writer = _TranscriptWriter(cfg.save_transcript, cfg.transcripts_dir)
            for name, fn in (("segmenter", self._segment_loop), ("asr", self._asr_loop),
                             ("translate", self._translate_loop), ("draft", self._draft_loop),
                             ("correct", self._correct_loop)):
                th = threading.Thread(target=fn, name=name, daemon=True)
                self._threads.append(th)
                th.start()

            self._status("loading", "正在启动音频捕获…")
            if self._source is None:
                self._source = create_source(cfg.audio)
            self._source.start(self._on_audio)
            if self._stop.is_set():
                return
            self.running = True
            parts = [self._source.name, self._recognizer.name]
            if self._translator:
                parts.append(self._translator.name)
            self._status("ready", "监听中 · " + " → ".join(parts))
        except (AsrError, TranslateError, AudioSourceError) as e:
            self._status("error", str(e))
            self._abort()
        except Exception as e:
            log.exception("pipeline boot failed")
            self._status("error", f"启动失败：{e}")
            self._abort()

    def _abort(self) -> None:
        threading.Thread(target=self.stop, daemon=True).start()
        self._on_status("stopped", "")

    # ── audio in ─────────────────────────────────────────────────────────────

    def _on_audio(self, samples: np.ndarray, rate: int) -> None:
        self._got_audio = True
        try:
            self._audio_q.put_nowait((samples, rate))
        except queue.Full:
            self._audio_dropped += 1                    # never block the OS audio thread
        if self._on_level:
            now = time.monotonic()
            if now - self._last_level >= 0.05:
                self._last_level = now
                self._on_level(float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0)

    def _segment_loop(self) -> None:
        seg, resampler = self._segmenter, Resampler()
        last_audio = started = time.monotonic()
        warned = False
        while not self._stop.is_set():
            try:
                samples, rate = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                now = time.monotonic()
                if seg.in_speech and now - last_audio > 0.25:
                    # Some loopback sources stop delivering packets when the app goes quiet; without
                    # data the VAD would never see the silence that ends the utterance.
                    self._handle(seg.feed(np.zeros(1600, dtype=np.float32)))
                elif not self._got_audio and not warned and now - started > 5:
                    warned = True
                    self._status("warn", "5 秒内没有收到任何音频数据：请确认目标应用正在发声，且未被静音")
                continue
            last_audio = time.monotonic()
            y = resampler.process(samples, rate)
            if len(y):
                self._handle(seg.feed(y))

    def _handle(self, events: list[SegEvent]) -> None:
        for ev in events:
            if ev.kind == "discard":
                self._update(ev.utt_id, removed=True)
            elif ev.kind == "partial":
                if self.cfg.asr.partials and self._recognizer.supports_partials:
                    self._asr_q.put("partial", ev)
            else:
                self._asr_q.put("final", ev)

    # ── recognition ──────────────────────────────────────────────────────────

    def _asr_prompt(self) -> str:
        t = self.cfg.translate
        return asr_hint(parse_glossary(t.glossary), t.topic)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Bring quiet sources (low app volume) up to a level Whisper handles well."""
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if 1e-4 < peak < 0.3:
            return (audio * min(8.0, 0.5 / peak)).astype(np.float32)
        return audio

    def _recognize(self, kind: str, ev: SegEvent):
        rec, final = self._recognizer, kind == "final"
        audio, prompt = self._normalize(ev.audio), self._asr_prompt()
        configured = self.cfg.lang.source
        if configured != "auto":
            return rec.transcribe(audio, base_lang(configured), final=final, prompt=prompt)

        if not final:                                   # interim: reuse the established language, skip detection
            return rec.transcribe(audio, self._tracker.current, final=False, prompt=prompt)

        res = rec.transcribe(audio, None, final=True, prompt=prompt)
        det_lang, det_prob = res.language, res.language_prob
        chosen = self._tracker.resolve(det_lang, det_prob, len(audio) / 16000) if det_lang else ""
        if chosen and chosen != base_lang(det_lang):
            # A low-confidence outlier ("Yeah." heard as Welsh): decode again in the established language.
            log.info("language %s (%.2f) overridden by %s", det_lang, det_prob, chosen)
            res = rec.transcribe(audio, chosen, final=True, prompt=prompt)
            res.language = chosen
        elif det_lang:
            self._tracker.observe(det_lang, det_prob)
        return res

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            item = self._asr_q.get(0.2)
            if item is None:
                continue
            kind, ev = item
            try:
                res = self._recognize(kind, ev)
            except AsrError as e:
                self._report("asr", str(e))
                continue
            except Exception as e:
                log.exception("recognition crashed")
                self._report("asr", f"识别出错：{e}")
                continue

            text = res.text.strip()
            lang = normalize_lang(res.language) or (guess_language(text) if text else "")
            if kind == "partial":
                if text:
                    self._update(ev.utt_id, src=text, src_lang=lang, src_final=False)
                    self._maybe_draft(ev.utt_id, text, lang)
                continue
            self._on_final_text(ev, text, lang, res.elapsed_ms, res.confidence)

    def _on_final_text(self, ev: SegEvent, text: str, lang: str, asr_ms: float, confidence: float = 1.0) -> None:
        self._cancel_draft(ev.utt_id)
        if not text:
            self._update(ev.utt_id, removed=True)
            return
        tgt = self.cfg.lang.target
        speaker, reason = self._speaker_of(ev)
        if not reason and confidence < _LOW_CONFIDENCE and len(text) >= _LOW_CONFIDENCE_MIN_CHARS:
            reason = "识别把握较低，可能有误（环境噪音、多人说话重叠等都会造成这种情况）"
        self._maybe_correct(ev.utt_id, text, lang)
        passthrough = self._translator is None or same_language(lang, tgt)
        if passthrough:
            line = self._update(ev.utt_id, src=text, src_lang=lang, src_final=True, asr_ms=asr_ms, dst="",
                                dst_final=True, dst_draft=False, skipped=self._translator is not None,
                                speaker=speaker, uncertain_reason=reason,
                                latency_ms=(time.monotonic() - ev.t_speech_end) * 1000)
            if line:
                self._save(line)
            return
        self._update(ev.utt_id, src=text, src_lang=lang, src_final=True, asr_ms=asr_ms, dst_lang=tgt,
                     speaker=speaker, uncertain_reason=reason)
        self._tr_q.put((ev.utt_id, text, lang, ev.t_speech_end))

    def _speaker_of(self, ev: SegEvent) -> tuple[int, str]:
        """Speaker number for a finished utterance (0 when tracking is off or failed), and a reason to flag the
        line as uncertain when the voiceprint looks like a blend of two known speakers ("" otherwise)."""
        if self._embedder is None:
            return 0, ""
        try:
            dur = len(ev.audio) / 16000
            emb = self._embedder(ev.audio) if dur >= MIN_EMBED_S else None
            # ambiguity must be judged against the centroids as they stood *before* this embedding teaches them
            # anything, or a blended voiceprint learned into its closer match would erase its own signature
            ambiguous = self._speakers.ambiguous(emb, dur)
            speaker = self._speakers.assign(emb, dur)
            reason = "检测到疑似多人同时说话，识别可能不准确" if ambiguous else ""
            return speaker, reason
        except Exception:
            log.exception("speaker identification failed")
            return 0, ""

    def reset_speakers(self) -> None:
        """Forget everyone heard so far (new meeting / different people): numbering starts again at 1."""
        self._speakers.reset()

    # ── translation ──────────────────────────────────────────────────────────

    def _translate_loop(self) -> None:
        tr, tgt = self._translator, self.cfg.lang.target
        while not self._stop.is_set():
            try:
                job = self._tr_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None or tr is None:
                continue
            utt_id, text, lang, t_end = job
            self._cancel_draft(utt_id)
            last = [0.0]

            def on_delta(acc: str, _id=utt_id) -> None:
                now = time.monotonic()
                if now - last[0] >= 0.04:               # ≤ 25 UI updates/s
                    last[0] = now
                    self._update(_id, dst=acc, dst_draft=False, dst_lang=tgt)

            t0 = time.perf_counter()
            try:
                out = tr.translate(text, lang, tgt, context=list(self._context), on_delta=on_delta)
            except TranslateError as e:
                self._report("translate", str(e))
                self._update(utt_id, dst_final=True, dst_draft=False, error=str(e))
                continue
            except Exception as e:
                log.exception("translation crashed")
                self._report("translate", f"翻译出错：{e}")
                self._update(utt_id, dst_final=True, dst_draft=False, error=str(e))
                continue
            if out:
                self._context.append((text, out))
            line = self._update(utt_id, dst=out, dst_final=True, dst_draft=False, dst_lang=tgt,
                                tr_ms=(time.perf_counter() - t0) * 1000,
                                latency_ms=(time.monotonic() - t_end) * 1000)
            if line:
                self._save(line)

    # ── source correction ───────────────────────────────────────────────────────

    def _maybe_correct(self, utt_id: int, text: str, lang: str) -> None:
        """Queue an LLM pass that fixes likely ASR mis-hearings in the *displayed original text* itself (opt-in:
        one extra request per sentence). Runs alongside translation, not instead of it."""
        tr = self._translator
        if self.cfg.translate.correct_source and tr is not None and tr.can_correct:
            self._cor_q.put((utt_id, text, lang))

    def _correct_loop(self) -> None:
        tr = self._translator
        while not self._stop.is_set():
            try:
                job = self._cor_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None or tr is None:
                continue
            utt_id, text, lang = job
            try:
                fixed = tr.correct(text, lang, context=list(self._context))
            except Exception:
                log.exception("source correction crashed")
                continue
            if fixed and fixed != text:
                self._update(utt_id, lambda l: l is not None and l.src == text, src=fixed)

    def _maybe_draft(self, utt_id: int, text: str, lang: str) -> None:
        if not (self.cfg.translate.draft and self._translator) or same_language(lang, self.cfg.lang.target):
            return
        if len(text) < _MIN_DRAFT_CHARS and not any(ord(c) > 0x2E80 for c in text):
            return
        if self._draft_last.get(utt_id) == text:
            return
        self._draft_last[utt_id] = text
        if len(self._draft_last) > 20:
            self._draft_last.pop(next(iter(self._draft_last)))
        with self._draft_cv:
            self._draft_job = (utt_id, text, lang)
            if self._draft_cancel:
                self._draft_cancel[1].set()             # superseded: stop generating the older draft
            self._draft_cv.notify()

    def _cancel_draft(self, utt_id: int) -> None:
        with self._draft_cv:
            if self._draft_job and self._draft_job[0] == utt_id:
                self._draft_job = None
            if self._draft_cancel and self._draft_cancel[0] == utt_id:
                self._draft_cancel[1].set()

    def _draft_loop(self) -> None:
        tr, tgt = self._translator, self.cfg.lang.target
        while not self._stop.is_set():
            with self._draft_cv:
                if self._draft_job is None:
                    self._draft_cv.wait(0.3)
                job, self._draft_job = self._draft_job, None
                if job is None or tr is None:
                    continue
                cancel = threading.Event()
                self._draft_cancel = (job[0], cancel)
            utt_id, text, lang = job

            def on_delta(acc: str, _id=utt_id, _c=cancel) -> None:
                # vetoed once a final translation has taken over (cancel is set before it starts)
                self._update(_id, lambda l: l is not None and not _c.is_set() and not l.dst_final,
                             dst=acc, dst_draft=True, dst_lang=tgt)

            try:
                tr.translate(text, lang, tgt, context=list(self._context), on_delta=on_delta, cancel=cancel)
            except TranslateError as e:
                self._report("translate", str(e))
            except Exception:
                log.exception("draft translation crashed")

    # ── shared state ─────────────────────────────────────────────────────────

    def _update(self, utt_id: int, _guard: Callable[[Line | None], bool] | None = None, **fields) -> Line | None:
        """Atomically apply ``fields`` to a line and notify the UI. ``_guard`` runs under the lock and can veto
        the write (used so a stale draft translation can never overwrite a final one)."""
        with self._lock:
            line = self._lines.get(utt_id)
            if _guard is not None and not _guard(line):
                return None
            if line is None:
                if fields.get("removed"):
                    return None
                line = self._lines[utt_id] = Line(id=utt_id)
                if len(self._lines) > _KEEP_LINES:
                    for k in sorted(self._lines)[:len(self._lines) - _KEEP_LINES]:
                        del self._lines[k]
            for k, v in fields.items():
                setattr(line, k, v)
            snapshot = dataclasses.replace(line)
        self._on_line(snapshot)
        return snapshot

    def _save(self, line: Line) -> None:
        if self._writer and line.src:
            self._writer.write(line)

    def _status(self, level: str, msg: str) -> None:
        self._on_status(level, msg)

    def _report(self, key: str, msg: str) -> None:
        """Errors repeat on every utterance; tell the user once per 8 s."""
        now = time.monotonic()
        if now - self._last_err.get(key + msg, -99) > 8:
            self._last_err[key + msg] = now
            self._status("error" if key == "translate" else "warn", msg)
