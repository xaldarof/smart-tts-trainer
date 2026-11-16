"""Per-segment quality metrics and accept/review/reject decision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config.settings import ProcessingSettings
from app.services.transcription.normalize import is_hallucination_like, text_warnings
from app.utils.audio import clipping_ratio, estimate_snr_db, rms_db


@dataclass
class Quality:
    duration: float
    rms_db: float
    snr_db: float
    clipping: float
    avg_prob: float
    min_prob: float
    cps: float
    speaker_sim: float | None
    overlap: bool
    status: str = "accepted"
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0


def measure(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    return rms_db(x), estimate_snr_db(x, sr), clipping_ratio(x)


def assess(text: str, x: np.ndarray, sr: int, avg_prob: float, min_prob: float, speaker_sim: float | None,
           overlap: bool, cfg: ProcessingSettings, extra_review: list[str] | None = None) -> Quality:
    dur = len(x) / sr
    r, snr, clip = measure(x, sr)
    letters = sum(ch.isalpha() for ch in text)
    cps = letters / dur if dur > 0 else 0.0
    q = Quality(dur, r, snr, clip, avg_prob, min_prob, cps, speaker_sim, overlap)
    reject, review = [], []

    if dur < cfg.min_segment_sec:
        reject.append(f"too short ({dur:.1f}s)")
    if dur > cfg.max_segment_sec + 0.75:
        reject.append(f"too long ({dur:.1f}s)")
    if not text.strip():
        reject.append("empty transcript")
    if overlap:
        reject.append("overlapping/uncertain speaker region")
    if clip > cfg.max_clipping_ratio:
        reject.append(f"clipping {clip * 100:.2f}%")
    if r < cfg.min_rms_db:
        reject.append(f"too quiet ({r:.0f} dBFS)")
    if avg_prob < cfg.min_avg_word_prob:
        reject.append(f"low ASR confidence ({avg_prob:.2f})")
    elif min_prob < cfg.min_word_prob:
        review.append(f"uncertain word (p={min_prob:.2f})")
    if text and is_hallucination_like(text):
        reject.append("likely ASR hallucination")
    if text and not (cfg.min_chars_per_sec <= cps <= cfg.max_chars_per_sec):
        review.append(f"unusual speaking rate ({cps:.1f} chars/s)")
    if snr < cfg.min_snr_db:
        review.append(f"noisy (SNR≈{snr:.0f} dB)")
    if speaker_sim is not None:
        if speaker_sim < cfg.speaker_sim_reject:
            reject.append(f"different speaker (sim {speaker_sim:.2f})")
        elif speaker_sim < cfg.speaker_sim_accept:
            review.append(f"speaker identity uncertain (sim {speaker_sim:.2f})")
    else:
        review.append("speaker similarity unavailable")
    review += text_warnings(text) if text.strip() else []
    review += extra_review or []

    q.reasons = reject + review
    q.status = "rejected" if reject else ("review" if review else "accepted")
    # Score 0..1 for ranking
    s = 0.35 * min(1.0, max(0.0, (avg_prob - 0.5) / 0.5))
    s += 0.25 * min(1.0, max(0.0, (snr - 5) / 30))
    s += 0.25 * (min(1.0, max(0.0, (speaker_sim - 0.3) / 0.6)) if speaker_sim is not None else 0.5)
    s += 0.15 * (1.0 if 3.0 <= dur <= 10.0 else 0.5)
    q.score = round(float(s), 3)
    return q
