"""Who is talking: speaker embeddings + online clustering, no PyTorch needed.

Each finished utterance is turned into a 192-number "voiceprint" by the 3D-Speaker CAM++ model (ONNX, 28 MB, trained
on large Chinese + English speaker sets with noise/music augmentation; runs on onnxruntime, already a dependency).
Voiceprints are compared by cosine similarity with the speakers heard so far: close enough → same person (and that
person's profile is refined), otherwise a new speaker is added. Labels are assigned in order of first appearance.

Why this model: measured on real speech (AISHELL-1 / LibriSpeech speakers) mixed with real music (GTZAN), the previous
model (WeSpeaker ResNet34) split 9 speakers into 18 at 5 dB and 35 at 0 dB of background music; CAM++ kept 9 with
a 1% equal-error-rate at 0 dB.

Limits worth knowing: utterances shorter than ~2 s carry little voice (under 1 s: none — the previous speaker is
reused; 1-2 s: only matched to an already known speaker), overlapping speech and music that is louder than the voice blur the voiceprint, and two similar voices can
be merged. The tracker never looks back — a line keeps the label it got when it was spoken.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# A mirror of Alibaba's iic/speech_campplus_sv_zh_en_16k-common_advanced (Apache-2.0) converted to ONNX, with its
# export recipe published. Both the revision and the file's SHA-256 are pinned: a different file is refused.
SPEAKER_MODEL_REPO = "echo-hello/speech_campplus_sv_zh_en_16k-common_advanced-onnx"
SPEAKER_MODEL_REVISION = "1a64735d2ab8c9b005101d5b0deff78c148248c0"
SPEAKER_MODEL_FILE = "onnx/speech_campplus_sv_zh_en_16k-common_advanced.onnx"
SPEAKER_MODEL_SHA256 = "945bcefe95672434d6ecf4c45bd0464d7e82be77d55985d0ec2f3253638110e1"
SAMPLE_RATE = 16000
MIN_EMBED_S = 1.0            # shorter audio carries no reliable voiceprint: reuse the previous speaker
FULL_EMBED_S = 2.0           # from here on an utterance may start a new speaker and refine profiles (EER ~1% vs 5%)
WEAK_MARGIN = 0.30           # 1-2 s utterances only join the nearest known speaker (this much below the threshold is fine)
_FRAME_LEN, _FRAME_SHIFT, _N_FFT, _N_MELS = 400, 160, 512, 80


# ── features ─────────────────────────────────────────────────────────────────

def _mel(f):
    return 1127.0 * np.log1p(f / 700.0)


def _mel_bank(n_mels: int = _N_MELS, n_fft: int = _N_FFT, rate: int = SAMPLE_RATE,
              low: float = 20.0, high: float | None = None) -> np.ndarray:
    """Kaldi-style triangular mel filters, shape (n_mels, n_fft // 2)."""
    high = rate / 2 if high is None else high
    mel_low, mel_high = _mel(low), _mel(high)
    delta = (mel_high - mel_low) / (n_mels + 1)
    mel_of_bin = _mel(np.arange(n_fft // 2) * rate / n_fft)
    left = mel_low + np.arange(n_mels)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (mel_of_bin - left) / (center - left)
    down = (right - mel_of_bin) / (right - center)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


_BANK = _mel_bank()
_WINDOW = np.hamming(_FRAME_LEN).astype(np.float32)          # symmetric, like torchaudio's kaldi.fbank


def fbank(audio: np.ndarray) -> np.ndarray:
    """80-dim log-mel filterbank (Kaldi conventions, as the WeSpeaker and 3D-Speaker models expect), mean-normalised. (frames, 80)."""
    x = np.asarray(audio, dtype=np.float32) * 32768.0
    if len(x) < _FRAME_LEN:
        x = np.pad(x, (0, _FRAME_LEN - len(x)))
    n = 1 + (len(x) - _FRAME_LEN) // _FRAME_SHIFT
    idx = np.arange(_FRAME_LEN)[None, :] + _FRAME_SHIFT * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)                        # remove DC offset
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)   # pre-emphasis
    spec = np.fft.rfft(frames * _WINDOW, n=_N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)[:, :_N_FFT // 2].astype(np.float32)
    feats = np.log(np.maximum(power @ _BANK.T, np.finfo(np.float32).eps))
    return feats - feats.mean(axis=0, keepdims=True)


# ── embedding model ──────────────────────────────────────────────────────────

class SpeakerEmbedder:
    def __init__(self, model_path: str):
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        self._input = self._sess.get_inputs()[0].name

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Unit-length voiceprint of 16 kHz mono float audio."""
        emb = self._sess.run(None, {self._input: fbank(audio)[None]})[0][0].astype(np.float64)
        return emb / (np.linalg.norm(emb) + 1e-9)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embedder(progress: Callable[[str], None] | None = None, models_dir: str | None = None) -> SpeakerEmbedder:
    """Find the model in the local cache (download it on first use), check it is the expected file, and load it."""
    from huggingface_hub import hf_hub_download
    say = progress or (lambda _m: None)
    say("正在加载说话人识别模型…")
    kw = dict(repo_id=SPEAKER_MODEL_REPO, filename=SPEAKER_MODEL_FILE, revision=SPEAKER_MODEL_REVISION,
              cache_dir=models_dir)
    try:
        path = hf_hub_download(local_files_only=True, **kw)
    except Exception:
        say("本地还没有说话人识别模型，正在下载（首次使用，约 28 MB）…")
        path = hf_hub_download(**kw)
    if _sha256(path) != SPEAKER_MODEL_SHA256:
        raise RuntimeError(f"说话人识别模型文件校验失败（{SPEAKER_MODEL_REPO}）。请在「设置 → 模型管理」里删除它后重新开始。")
    return SpeakerEmbedder(path)


# ── online clustering ────────────────────────────────────────────────────────

class SpeakerTracker:
    """Assigns speaker numbers (1, 2, ...) to voiceprints as they arrive.

    ``threshold``: cosine similarity needed to call two utterances the same person. Higher = more speakers
    (splits one person more easily), lower = fewer (merges similar voices). ``max_speakers`` 0 = unlimited; when the
    limit is reached a new voice is assigned to the closest existing speaker instead.

    How much an utterance may do depends on its length: under ``MIN_EMBED_S`` nothing (previous speaker is
    reused); up to ``FULL_EMBED_S`` it may only join an existing speaker (never create one or change profiles);
    longer ones may also start a new speaker, and refine profiles in proportion to their length.
    """

    def __init__(self, threshold: float = 0.55, max_speakers: int = 0):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._sum: list[np.ndarray] = []          # length-weighted sum of voiceprints per speaker
        self._n: list[int] = []
        self._last = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._n)

    def reset(self) -> None:
        with self._lock:
            self._sum.clear()
            self._n.clear()
            self._last = 0

    def _centroids(self) -> np.ndarray:
        c = np.stack(self._sum)
        return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)

    def assign(self, emb: np.ndarray | None, duration_s: float = 10.0) -> int:
        """Speaker number for this voiceprint (0 = unknown, only before anyone has been heard)."""
        with self._lock:
            if emb is None or duration_s < MIN_EMBED_S:
                return self._last
            weight = min(duration_s, 10.0) / 10.0
            if not self._n:
                return self._add(emb, weight) if duration_s >= FULL_EMBED_S else self._last
            sims = self._centroids() @ emb
            best = int(np.argmax(sims))
            if duration_s < FULL_EMBED_S:                          # short: recognise, never create or teach
                if sims[best] >= self.threshold - WEAK_MARGIN:
                    self._last = best + 1
                return self._last
            if sims[best] >= self.threshold or (self.max_speakers and len(self._n) >= self.max_speakers):
                return self._learn(best, emb, weight)
            return self._add(emb, weight)

    def ambiguous(self, emb: np.ndarray | None, duration_s: float = 10.0) -> bool:
        """True when this voiceprint sits about equally close to two *different* known speakers instead of clearly
        matching one — the signature of two people's voices blended together in the same utterance (talking over
        each other), rather than one recognisable voice. Only meaningful once at least two speakers are known and
        the utterance is long enough for ``assign`` itself to trust the embedding."""
        with self._lock:
            if emb is None or duration_s < FULL_EMBED_S or len(self._n) < 2:
                return False
            sims = np.sort(self._centroids() @ emb)[::-1]
            best, second = float(sims[0]), float(sims[1])
            return best >= self.threshold and second >= self.threshold - WEAK_MARGIN and best - second < 0.08

    def _add(self, emb: np.ndarray, weight: float) -> int:
        self._sum.append(emb * weight)
        self._n.append(1)
        self._last = len(self._n)
        return self._last

    def _learn(self, i: int, emb: np.ndarray, weight: float) -> int:
        self._sum[i] += emb * weight
        self._n[i] += 1
        self._last = i + 1
        return self._last
