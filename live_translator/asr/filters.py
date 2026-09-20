"""Whisper is known to invent text on silence / noise / music. These filters remove the usual offenders."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Whole-utterance matches that are (almost) never real speech in a live setting.
_ALWAYS = [
    "thanks for watching", "thank you for watching", "please subscribe", "subscribe to my channel",
    "like and subscribe", "see you in the next video", "amara.org", "subtitles by",
    "字幕由amara.org社区提供", "字幕志愿者", "中文字幕由", "请不吝点赞", "订阅我的频道", "感谢观看",
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録", "最後までご視聴",
    "구독과 좋아요", "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다",
]
# Legit in real conversation, so only dropped when the model itself is unsure.
_WEAK = {"thank you", "thanks", "you", "bye", "bye bye", "okay", "ok", "yeah", "um", "uh", "hmm",
         "嗯", "啊", "好", "谢谢", "谢谢大家", "ありがとう", "はい", "네", "감사합니다"}

_PUNCT = re.compile(r"[\s\.\,\!\?。，！？、…~～\-—\"'“”‘’()（）\[\]♪♫♬]+")
_LOOP = re.compile(r"(.{2,40}?)(?:[\s,，、]*\1){3,}", re.S)   # a phrase repeated ≥4× in a row
_CHAR_LOOP = re.compile(r"(.)\1{5,}")              # one char repeated ≥6×


@dataclass
class SegStats:
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0


def _norm(s: str) -> str:
    return _PUNCT.sub(" ", s.lower()).strip()


def collapse_repeats(text: str) -> str:
    text = _LOOP.sub(r"\1", text)
    return _CHAR_LOOP.sub(lambda m: m.group(1) * 3, text)


def keep_segment(s: SegStats) -> bool:
    if not _norm(s.text):
        return False
    if s.compression_ratio > 2.4:                                   # runaway repetition
        return False
    if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:             # Whisper's own "this is silence" rule
        return False
    if s.avg_logprob < -1.6:                                        # decoding was mostly guessing
        return False
    norm = _norm(s.text)
    if any(p in norm for p in _ALWAYS) and len(norm) < 60:
        return False
    if norm in _WEAK and (s.no_speech_prob > 0.35 or s.avg_logprob < -0.9):
        return False
    return True


def clean_text(text: str, prompt: str = "") -> str:
    text = collapse_repeats(text).strip()
    if prompt and len(text) > 3 and _norm(text) and _norm(text) in _norm(prompt):
        return ""                                                   # the model just parroted the prompt
    return text
