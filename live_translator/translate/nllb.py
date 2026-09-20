"""Built-in offline translation: Meta NLLB-200 running on CTranslate2 (already a dependency of faster-whisper).

No server, no API key. Quality is below a good LLM but latency is tiny and it works fully offline.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Sequence

from ..languages import nllb_code
from ..runtime import cuda_available, models_dir
from ..text_utils import (guess_language, is_cjk_lang, normalize_cjk_punct, split_sentences,
                          strip_trailing_ellipsis, trim_runaway)
from .base import DeltaCallback, TranslateError, Translator

log = logging.getLogger(__name__)


class NllbTranslator(Translator):
    def __init__(self, model_id: str, device: str = "auto", progress: Callable[[str], None] | None = None):
        self.model_id = model_id
        self.device = device
        self._progress = progress or (lambda _m: None)
        self.name = f"NLLB 离线 · {model_id.split('/')[-1]}"
        self._translator = None
        self._sp = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def warmup(self) -> None:
        with self._load_lock:                              # draft and final lanes may both trigger the first load
            if self._translator is None:
                self._load()

    def _load(self) -> None:
        try:
            import ctranslate2
            import sentencepiece as spm
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise TranslateError(f"缺少依赖：{e.name}（pip install ctranslate2 sentencepiece huggingface_hub）") from e
        patterns = ["model.bin", "*.json", "*.model", "*.txt"]
        self._progress(f"正在加载离线翻译模型 {self.model_id}…")
        try:
            path = snapshot_download(self.model_id, allow_patterns=patterns, local_files_only=True,
                                     cache_dir=models_dir())
            if not os.path.isfile(os.path.join(path, "model.bin")):
                raise FileNotFoundError("model.bin")           # earlier download was interrupted
        except Exception:
            self._progress(f"本地还没有 {self.model_id}，正在下载（首次使用；600M int8 版约 0.6 GB）…")
            try:
                path = snapshot_download(self.model_id, allow_patterns=patterns, cache_dir=models_dir())
            except Exception as e:
                raise TranslateError(f"下载离线翻译模型失败：{e}") from e
        use_cuda = self.device == "cuda" or (self.device == "auto" and cuda_available())
        try:
            self._translator = ctranslate2.Translator(
                path, device="cuda" if use_cuda else "cpu",
                compute_type="float16" if use_cuda else "int8", inter_threads=1,
                intra_threads=0 if use_cuda else 4)
        except Exception as e:
            if not use_cuda:
                raise TranslateError(f"加载离线翻译模型失败：{e}") from e
            self._translator = ctranslate2.Translator(path, device="cpu", compute_type="int8",
                                                      inter_threads=1, intra_threads=4)
        self._sp = spm.SentencePieceProcessor(model_file=f"{path}/sentencepiece.bpe.model")
        self.name += f" · {'gpu' if use_cuda else 'cpu'}"

    def translate(self, text: str, src: str, tgt: str, *, context: Sequence[tuple[str, str]] = (),
                  on_delta: DeltaCallback | None = None, cancel: threading.Event | None = None) -> str:
        text = text.strip()
        if not text:
            return ""
        self.warmup()
        src_code = nllb_code(src or guess_language(text))
        tgt_code = nllb_code(tgt)
        if not src_code or not tgt_code:
            raise TranslateError(f"离线模型不支持语言：{src or '?'} → {tgt}")

        sentences = split_sentences(strip_trailing_ellipsis(text))
        batch = [[src_code, *self._sp.encode(x, out_type=str), "</s>"] for x in sentences]
        with self._lock:                                   # CT2 is thread-safe, but drafts must not starve finals
            results = self._translator.translate_batch(
                batch, target_prefix=[[tgt_code]] * len(batch), beam_size=3,
                # An output much longer than its source is a runaway, so bound it (short inputs get short caps).
                max_decoding_length=min(256, max(int(len(b) * 1.8) + 8 for b in batch)),
                no_repeat_ngram_size=4, repetition_penalty=1.1, disable_unk=True)
        pieces = []
        for sent, r in zip(sentences, results):
            piece = self._sp.decode(r.hypotheses[0][1:]).replace("⁇", "").strip()
            pieces.append(trim_runaway(sent, piece))
        out = ("" if is_cjk_lang(tgt) else " ").join(p for p in pieces if p).strip()
        out = normalize_cjk_punct(out, tgt)
        if on_delta and out:
            on_delta(out)
        return out
