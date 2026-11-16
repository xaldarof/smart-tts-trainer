"""Align words with speaker turns and cut them into TTS-sized segments.

Pure functions (no models) so they are unit-testable. Input words come from
faster-whisper word timestamps; speaker turns from diarization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_SENT_END = (".", "!", "?", "…")
_CLAUSE_END = (",", ";", ":", "-")


@dataclass
class AWord:
    start: float
    end: float
    text: str
    prob: float
    speaker: str | None = None
    uncertain: bool = False


@dataclass
class Candidate:
    start: float
    end: float
    speaker: str
    words: list[AWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def avg_prob(self) -> float:
        return sum(w.prob for w in self.words) / len(self.words) if self.words else 0.0

    @property
    def min_prob(self) -> float:
        return min((w.prob for w in self.words), default=0.0)

    @property
    def uncertain(self) -> bool:
        return any(w.uncertain for w in self.words)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words: list[AWord], turns: list[tuple[float, float, str]],
                    uncertain: list[tuple[float, float]]) -> list[AWord]:
    """Label each word with the speaker whose turns cover most of it.
    Words covered by no turn get ``None``; words touching an uncertain/overlap region are flagged."""
    turns = sorted(turns)
    for w in words:
        best, best_ov = None, 0.0
        for s, e, lab in turns:
            if s > w.end:
                break
            ov = _overlap(w.start, w.end, s, e)
            if ov > best_ov:
                best, best_ov = lab, ov
        w.speaker = best if best_ov >= 0.3 * max(1e-3, w.end - w.start) else None
        w.uncertain = any(_overlap(w.start, w.end, s, e) > 0 for s, e in uncertain)
    return words


def build_candidates(words: list[AWord], target_speakers: set[str], min_sec: float, max_sec: float,
                     pause_sec: float, pad_sec: float = 0.15) -> list[Candidate]:
    """Group consecutive words of target speakers into segments.

    Breaks on speaker change, pauses >= ``pause_sec``, and keeps segments <= ``max_sec`` by
    cutting at the last sentence (preferred) or clause boundary. Boundaries are padded by
    ``pad_sec`` but never cross into neighbouring words.
    """
    groups: list[list[AWord]] = []
    cur: list[AWord] = []
    for w in words:
        if cur and (w.speaker != cur[-1].speaker or w.start - cur[-1].end >= pause_sec):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    pieces: list[list[AWord]] = []
    for g in groups:
        if not g or g[0].speaker not in target_speakers:
            continue
        pieces.extend(_split_long(g, min_sec, max_sec))

    idx = {id(w): i for i, w in enumerate(words)}
    out = []
    for p in pieces:
        i0, i1 = idx[id(p[0])], idx[id(p[-1])]
        prev_end = words[i0 - 1].end if i0 > 0 else 0.0
        next_start = words[i1 + 1].start if i1 + 1 < len(words) else p[-1].end + 10
        start = max(p[0].start - pad_sec, (prev_end + p[0].start) / 2 if i0 > 0 else 0.0, 0.0)
        end = min(p[-1].end + pad_sec, (p[-1].end + next_start) / 2)
        out.append(Candidate(start, end, p[0].speaker, p))
    return out


def _split_long(g: list[AWord], min_sec: float, max_sec: float) -> list[list[AWord]]:
    out = []
    while g:
        if g[-1].end - g[0].start <= max_sec:
            out.append(g)
            break
        # largest prefix within max_sec
        k = 0
        while k + 1 < len(g) and g[k + 1].end - g[0].start <= max_sec:
            k += 1
        cut = None
        for ends in (_SENT_END, _CLAUSE_END):
            for j in range(k, -1, -1):
                if g[j].end - g[0].start < min_sec:
                    break
                if g[j].text.strip().endswith(ends):
                    cut = j
                    break
            if cut is not None:
                break
        if cut is None:  # no punctuation: cut at the largest inter-word gap in range
            best_gap, cut = -1.0, k
            for j in range(k):
                if g[j].end - g[0].start < min_sec:
                    continue
                gap = g[j + 1].start - g[j].end
                if gap > best_gap:
                    best_gap, cut = gap, j
        out.append(g[: cut + 1])
        g = g[cut + 1:]
    return out
