from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Line:
    """One utterance as it moves through recognition and translation. The UI upserts these by ``id``."""
    id: int
    src: str = ""
    dst: str = ""
    src_lang: str = ""            # detected / configured spoken language ("" if unknown)
    dst_lang: str = ""
    src_final: bool = False       # False → interim recognition, may still change
    dst_final: bool = False       # False → streaming or draft translation
    dst_draft: bool = False       # translation of an interim (not yet final) source text
    skipped: bool = False         # source already in the target language → not translated
    removed: bool = False         # retract this line (it turned out to be noise)
    speaker: int = 0              # who spoke: 1, 2, 3 … in order of first appearance; 0 = unknown / not tracked
    uncertain_reason: str = ""    # non-empty → this line's recognition may be unreliable (why, for a tooltip)
    asr_ms: float = 0.0           # recognition time of the final pass
    tr_ms: float = 0.0            # translation time of the final pass
    latency_ms: float = 0.0       # speaker stopped talking → translation complete
    error: str = ""
    created: float = field(default_factory=time.time)
