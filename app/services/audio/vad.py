"""Silero VAD speech detection (CPU; the model is tiny and fast)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.utils.audio import load_mono
from app.utils.jobs import JobContext

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def detect_speech(wav: Path, ctx: JobContext, threshold: float = 0.5, min_speech_ms: int = 250,
                  min_silence_ms: int = 300) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    x, sr = load_mono(wav, 16000)
    if x.size == 0:
        return []

    def cb(p: float):
        ctx.check()
        ctx.progress(p / 100.0, "Detecting speech")

    ts = get_speech_timestamps(
        torch.from_numpy(x), _model(), threshold=threshold, sampling_rate=16000,
        min_speech_duration_ms=min_speech_ms, min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=60, return_seconds=True, progress_tracking_callback=cb,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def speech_stats(regions: list[tuple[float, float]], total: float) -> dict:
    sp = float(sum(e - s for s, e in regions))
    return {"speech_sec": sp, "total_sec": total, "speech_ratio": sp / total if total else 0.0,
            "n_regions": len(regions)}


def silence_regions(regions: list[tuple[float, float]], total: float, min_len: float = 0.5) -> list[tuple[float, float]]:
    out, cur = [], 0.0
    for s, e in regions:
        if s - cur >= min_len:
            out.append((cur, s))
        cur = max(cur, e)
    if total - cur >= min_len:
        out.append((cur, total))
    return out


def intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    i = j = 0
    out = []
    a, b = sorted(a), sorted(b)
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total_duration(regs) -> float:
    return float(np.sum([e - s for s, e in regs])) if regs else 0.0
