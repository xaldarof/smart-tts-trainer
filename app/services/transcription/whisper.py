"""faster-whisper transcription with word timestamps and confidence."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from app.utils.audio import load_mono
from app.utils.gpu import free_gpu_memory, is_oom
from app.utils.jobs import JobContext

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[Word]


class Transcriber:
    def __init__(self, model_size: str, device: str, compute_type: str = "auto", cache_dir: Path | None = None):
        self.model_size, self.device = model_size, device
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.compute_type = compute_type
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Loading faster-whisper %s on %s (%s)", self.model_size, self.device, self.compute_type)
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type,
                                       download_root=str(self.cache_dir) if self.cache_dir else None)
        return self._model

    def close(self):
        self._model = None
        free_gpu_memory()

    def transcribe(self, wav: Path, ctx: JobContext, language: str = "ru", beam_size: int = 5) -> list[Segment]:
        try:
            return self._run(wav, ctx, language, beam_size)
        except Exception as e:
            if self.device == "cuda" and is_oom(e):
                log.warning("Whisper OOM on GPU; retrying with int8_float16")
                self._model = None
                free_gpu_memory()
                self.compute_type = "int8_float16"
                return self._run(wav, ctx, language, beam_size)
            raise

    def _run(self, wav: Path, ctx: JobContext, language: str, beam_size: int) -> list[Segment]:
        model = self._load()
        audio, _ = load_mono(wav, 16000)
        total = len(audio) / 16000
        segs, info = model.transcribe(
            audio, language=language, beam_size=beam_size, word_timestamps=True,
            condition_on_previous_text=False,       # limits hallucination loops
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
            temperature=[0.0, 0.2, 0.4],
        )
        out: list[Segment] = []
        for s in segs:  # generator: work happens while iterating
            ctx.check()
            words = [Word(float(w.start), float(w.end), w.word, float(w.probability)) for w in (s.words or [])]
            out.append(Segment(float(s.start), float(s.end), s.text.strip(), float(s.avg_logprob),
                               float(s.no_speech_prob), float(s.compression_ratio), words))
            ctx.progress(min(1.0, s.end / total) if total else 1.0, f"Transcribing {s.end:.0f}/{total:.0f}s")
        return out


def to_json(segments: list[Segment]) -> list[dict]:
    return [asdict(s) for s in segments]


def from_json(data: list[dict]) -> list[Segment]:
    return [Segment(**{**d, "words": [Word(**w) for w in d["words"]]}) for d in data]
