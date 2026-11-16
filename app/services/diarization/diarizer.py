"""Speaker diarization + speaker embeddings.

Backends
--------
* ``pyannote``: pyannote/speaker-diarization-community-1 (pyannote.audio 4.x, CC-BY-4.0).
  Best quality incl. overlapped-speech detection. The model is *gated* on Hugging
  Face: accept its terms on the model page and provide a token (HF_TOKEN env var or
  `huggingface-cli login`). Audio is passed in memory, so torchcodec/FFmpeg DLLs are
  not needed for decoding.
* ``ecapa``: no-login fallback. SpeechBrain ECAPA-TDNN (Apache-2.0, ungated) embeddings
  over sliding windows inside VAD speech regions, agglomerative clustering. It cannot
  detect overlapped speech directly; windows whose embedding is ambiguous between two
  clusters are reported as "uncertain" regions, and segments touching them are
  rejected or sent to review.

Regardless of backend, per-speaker embeddings are computed with ECAPA so speakers
can be compared across videos with one consistent embedding space.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config.settings import hf_token
from app.utils.audio import load_mono
from app.utils.gpu import free_gpu_memory
from app.utils.jobs import JobContext

log = logging.getLogger(__name__)

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


class DiarizationError(RuntimeError):
    pass


@dataclass
class DiarizationResult:
    backend: str
    turns: list[tuple[float, float, str]]                       # (start, end, label)
    uncertain: list[tuple[float, float]] = field(default_factory=list)  # overlap / ambiguous regions

    def labels(self) -> list[str]:
        return sorted({t[2] for t in self.turns})

    def durations(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for s, e, lab in self.turns:
            d[lab] = d.get(lab, 0.0) + (e - s)
        return d


# --------------------------------------------------------------- embeddings
class SpeakerEmbedder:
    """ECAPA speaker embeddings (192-d, L2-normalized)."""

    def __init__(self, device: str, cache_dir: Path):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL, savedir=str(cache_dir / "ecapa"), run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )

    def embed(self, chunks: list[np.ndarray], batch: int | None = None) -> np.ndarray:
        """Embed 16 kHz float32 chunks. Returns (N, D) L2-normalized.

        Chunks are length-sorted to minimise padding. On CPU, large batches hit a pathological
        slow path in PyTorch's conv kernels (measured: 32x1.5 s took 20 s at batch 32 vs 0.6 s at
        batch 8), so CPU uses batch 4.
        """
        import torch

        if not chunks:
            return np.zeros((0, 192), dtype=np.float32)
        batch = batch or (32 if self.device == "cuda" else 4)
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        res: list[np.ndarray | None] = [None] * len(chunks)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            part = [chunks[k] if len(chunks[k]) >= 1600 else np.pad(chunks[k], (0, 1600 - len(chunks[k])))
                    for k in idx]
            n = max(len(c) for c in part)
            wavs = torch.zeros(len(part), n)
            lens = torch.zeros(len(part))
            for j, c in enumerate(part):
                wavs[j, : len(c)] = torch.from_numpy(np.ascontiguousarray(c))
                lens[j] = len(c) / n
            with torch.inference_mode():
                emb = self.model.encode_batch(wavs.to(self.device), lens.to(self.device)).squeeze(1).cpu().numpy()
            for j, k in enumerate(idx):
                res[k] = emb[j]
        e = np.stack(res).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def close(self) -> None:
        self.model = None
        free_gpu_memory()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def speaker_embedding(embedder: SpeakerEmbedder, x16: np.ndarray, turns: list[tuple[float, float]],
                      max_chunks: int = 40) -> np.ndarray | None:
    """Mean ECAPA embedding over a speaker's longest turns (each cropped to <= 6 s)."""
    sr = 16000
    turns = sorted(turns, key=lambda t: t[1] - t[0], reverse=True)[:max_chunks]
    chunks = []
    for s, e in turns:
        if e - s < 1.0:
            continue
        e = min(e, s + 6.0)
        chunks.append(x16[int(s * sr): int(e * sr)])
    if not chunks:
        return None
    emb = embedder.embed(chunks).mean(0)
    return (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- pyannote
def diarize_pyannote(wav: Path, device: str, ctx: JobContext, num_speakers: int = 0) -> DiarizationResult:
    import torch
    from pyannote.audio import Pipeline

    tok = hf_token()
    if not tok:
        raise DiarizationError(
            f"pyannote needs a Hugging Face token: accept the terms at https://huggingface.co/{PYANNOTE_MODEL} "
            "and set HF_TOKEN (or run `huggingface-cli login`). Alternatively choose the 'ecapa' backend in Settings."
        )
    try:
        pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, token=tok)
    except Exception as e:
        raise DiarizationError(f"Could not load {PYANNOTE_MODEL}: {e}. Make sure you accepted the model's "
                               "conditions on Hugging Face with the account that owns the token.") from e
    if pipe is None:
        raise DiarizationError(f"Access to {PYANNOTE_MODEL} was denied. Accept its terms on Hugging Face.")
    pipe.to(torch.device(device))
    x, sr = load_mono(wav, 16000)

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        ctx.check()
        if total:
            ctx.progress(completed / total, f"Diarization: {step_name}")

    kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
    out = pipe({"waveform": torch.from_numpy(x)[None], "sample_rate": sr}, hook=hook, **kwargs)
    ann = getattr(out, "speaker_diarization", out)
    excl = getattr(out, "exclusive_speaker_diarization", None)
    overlaps = [(float(seg.start), float(seg.end)) for seg in ann.get_overlap()]
    src = excl if excl is not None else ann
    turns = [(float(seg.start), float(seg.end), str(lab)) for seg, _, lab in src.itertracks(yield_label=True)]
    del pipe
    free_gpu_memory()
    return DiarizationResult("pyannote", turns, overlaps)


# ------------------------------------------------------------------- ECAPA
def diarize_ecapa(wav: Path, speech: list[tuple[float, float]], embedder: SpeakerEmbedder, ctx: JobContext,
                  num_speakers: int = 0, threshold: float = 0.55, win: float = 1.5, hop: float = 0.75,
                  ambiguity_margin: float = 0.08) -> DiarizationResult:
    from scipy.cluster.hierarchy import fcluster, linkage

    x, sr = load_mono(wav, 16000)
    windows: list[tuple[float, float]] = []
    for s, e in speech:
        if e - s < 0.5:
            continue
        if e - s <= win:
            windows.append((s, e))
            continue
        t = s
        while t + win <= e:
            windows.append((t, t + win))
            t += hop
        if e - (t - hop + win) > 0.3:
            windows.append((e - win, e))
    if not windows:
        return DiarizationResult("ecapa", [], [])
    ctx.progress(0.1, f"Embedding {len(windows)} windows")
    chunks = [x[int(s * sr): int(e * sr)] for s, e in windows]
    embs = []
    step = 256
    for i in range(0, len(chunks), step):
        ctx.check()
        embs.append(embedder.embed(chunks[i:i + step]))
        ctx.progress(0.1 + 0.7 * min(1.0, (i + step) / len(chunks)), "Embedding speech windows")
    E = np.concatenate(embs)

    if len(E) == 1:
        labels = np.array([1])
    else:
        Z = linkage(E, method="average", metric="cosine")
        if num_speakers > 0:
            labels = fcluster(Z, t=num_speakers, criterion="maxclust")
        else:
            labels = fcluster(Z, t=threshold, criterion="distance")
    # Re-assign tiny clusters (< 3 windows) to nearest big cluster: they are usually noise.
    uniq, counts = np.unique(labels, return_counts=True)
    big = [u for u, c in zip(uniq, counts) if c >= 3] or list(uniq)
    cents = {u: E[labels == u].mean(0) for u in big}
    cents = {u: c / np.linalg.norm(c) for u, c in cents.items()}
    sims = np.stack([E @ cents[u] for u in big], axis=1)  # (N, K)
    labels = np.array([big[i] for i in sims.argmax(1)])
    order = np.argsort(-sims, axis=1)
    ambiguous = np.zeros(len(E), dtype=bool)
    if len(big) > 1:
        top = sims[np.arange(len(E)), order[:, 0]]
        second = sims[np.arange(len(E)), order[:, 1]]
        ambiguous = (top - second) < ambiguity_margin

    # Map cluster ids to SPEAKER_00.. ordered by talk time.
    talk = {u: float(np.sum(labels == u)) for u in big}
    names = {u: f"SPEAKER_{i:02d}" for i, u in enumerate(sorted(big, key=lambda u: -talk[u]))}

    # Turns: each window owns [center-hop/2, center+hop/2] clipped to its speech region.
    turns: list[list] = []
    uncertain: list[tuple[float, float]] = []
    for (s, e), lab, amb in zip(windows, labels, ambiguous):
        c = (s + e) / 2
        reg = next(((rs, re_) for rs, re_ in speech if rs <= c <= re_), (s, e))
        a, b = max(reg[0], c - hop / 2), min(reg[1], c + hop / 2)
        # first/last window of a speech region owns the region edge
        if s <= reg[0] + 1e-6:
            a = reg[0]
        if e >= reg[1] - 1e-6:
            b = reg[1]
        if amb:
            uncertain.append((a, b))
        name = names[lab]
        if turns and turns[-1][2] == name and a - turns[-1][1] < 0.05:
            turns[-1][1] = max(turns[-1][1], b)
        else:
            turns.append([a, b, name])
    out = [(float(a), float(b), n) for a, b, n in turns]
    ctx.progress(1.0, f"{len(names)} speakers")
    return DiarizationResult("ecapa", out, _merge(uncertain))


def _merge(regs: list[tuple[float, float]], gap: float = 0.05) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(regs):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


def make_preview(x: np.ndarray, sr: int, turns: list[tuple[float, float]], max_sec: float = 12.0) -> np.ndarray:
    """Concatenate the longest turns (with short gaps) for an audio preview."""
    out, total = [], 0.0
    gap = np.zeros(int(0.35 * sr), dtype=np.float32)
    for s, e in sorted(turns, key=lambda t: t[1] - t[0], reverse=True):
        if total >= max_sec:
            break
        e = min(e, s + (max_sec - total))
        out += [x[int(s * sr): int(e * sr)], gap]
        total += e - s
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
