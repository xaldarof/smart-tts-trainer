"""XTTS v2 inference: fine-tuned model, pretrained baseline, and reference-audio cloning.

Modes (stored in generation history so they are never confused):

* ``trained``    - fine-tuned GPT weights from a training run + conditioning latents computed
                   from that run's own dataset clips (XTTS always conditions on reference audio;
                   after fine-tuning the *weights* themselves are adapted to the voice).
* ``pretrained`` - original XTTS v2 weights with the same reference clips: the baseline for
                   A/B comparison with the fine-tuned model.
* ``clone``      - original XTTS v2 weights with user-chosen reference audio. Zero-shot cloning,
                   NOT fine-tuning.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from app.config.settings import Settings
from app.database.db import Database
from app.utils.audio import save_wav
from app.utils.gpu import free_gpu_memory, gpu_stage, is_oom, pick_device
from app.utils.jobs import JobContext

log = logging.getLogger(__name__)

XTTS_RU_CHAR_LIMIT = 182      # TTS.tts.layers.xtts.tokenizer char_limits["ru"]
CHUNK_CHARS = 170
MAX_TEXT_CHARS = 20000
OUT_SR = 24000


class InferenceError(RuntimeError):
    pass


def validate_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        raise InferenceError("Enter some text to synthesize.")
    if len(t) > MAX_TEXT_CHARS:
        raise InferenceError(f"Text is {len(t)} characters; the limit is {MAX_TEXT_CHARS}. Split it into parts.")
    if not re.search(r"[А-Яа-яЁё]", t):
        raise InferenceError("The text contains no Cyrillic letters; this voice is trained for Russian.")
    return t


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split into sentence-based chunks <= limit chars (XTTS truncates/garbles beyond ~182 for ru)."""
    sents = re.split(r"(?<=[.!?…])\s+", text.strip())
    pieces: list[str] = []
    for s in sents:
        while len(s) > limit:
            cut = max(s.rfind(p, 0, limit) for p in (", ", "; ", ": ", " - "))
            if cut < limit // 3:
                cut = s.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            pieces.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            pieces.append(s)
    chunks: list[str] = []
    for p in pieces:
        if chunks and len(chunks[-1]) + 1 + len(p) <= limit:
            chunks[-1] = f"{chunks[-1]} {p}"
        else:
            chunks.append(p)
    return chunks


class XttsEngine:
    """Loads one XTTS model at a time; caches conditioning latents per reference set."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = pick_device(settings.device)
        self.model = None
        self.model_dir: Path | None = None
        self._latents: dict[tuple, tuple] = {}

    def load(self, model_dir: Path) -> None:
        if self.model is not None and self.model_dir == model_dir:
            return
        self.unload()
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        for f in ("config.json", "model.pth", "vocab.json"):
            if not (model_dir / f).exists():
                raise InferenceError(f"Model file missing: {model_dir / f}. Re-export the model or re-download XTTS.")
        cfg = XttsConfig()
        cfg.load_json(str(model_dir / "config.json"))
        model = Xtts.init_from_config(cfg)
        try:
            model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), checkpoint_path=str(model_dir / "model.pth"),
                                  vocab_path=str(model_dir / "vocab.json"), use_deepspeed=False, eval=True)
        except RuntimeError as e:
            raise InferenceError(f"Checkpoint is incompatible with XTTS v2: {e}") from e
        self.model = model.to(self.device)
        self.model_dir = model_dir
        log.info("Loaded XTTS from %s on %s", model_dir, self.device)

    def unload(self) -> None:
        self.model = None
        self.model_dir = None
        self._latents.clear()
        free_gpu_memory()

    def latents(self, refs: list[Path]):
        key = (str(self.model_dir), tuple(sorted(str(r) for r in refs)))
        if key not in self._latents:
            missing = [r for r in refs if not Path(r).exists()]
            if missing:
                raise InferenceError(f"Reference audio not found: {missing[0]}")
            gpt_cond, spk = self.model.get_conditioning_latents(audio_path=[str(r) for r in refs], gpt_cond_len=30,
                                                                gpt_cond_chunk_len=6, max_ref_length=60)
            self._latents[key] = (gpt_cond, spk)
        return self._latents[key]

    def synthesize(self, text: str, refs: list[Path], language: str, ctx: JobContext, temperature: float = 0.7,
                   speed: float = 1.0, top_p: float = 0.85, repetition_penalty: float = 5.0) -> np.ndarray:
        import torch

        chunks = chunk_text(text)
        gpt_cond, spk = self.latents(refs)
        pause = np.zeros(int(0.18 * OUT_SR), dtype=np.float32)
        out = []
        for i, ch in enumerate(chunks):
            ctx.check()
            ctx.progress(i / len(chunks), f"Chunk {i + 1}/{len(chunks)}")
            with torch.inference_mode():
                r = self.model.inference(ch, language, gpt_cond, spk, temperature=temperature, top_p=top_p,
                                         repetition_penalty=repetition_penalty, speed=speed,
                                         enable_text_splitting=False)
            wav = np.asarray(r["wav"], dtype=np.float32).reshape(-1)
            if wav.size == 0 or not np.isfinite(wav).all():
                raise InferenceError(f"Model produced invalid audio for chunk {i + 1}.")
            out += [wav, pause]
        ctx.progress(1.0, "Done")
        audio = np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return audio * (0.95 / peak) if peak > 0.95 else audio


def model_refs(db: Database, model_id: int) -> tuple[Path, list[Path], dict]:
    m = db.q1("SELECT * FROM models WHERE id=?", (model_id,))
    if m is None:
        raise InferenceError("Model not found in the library.")
    meta = json.loads(m["meta_json"] or "{}")
    d = Path(m["path"])
    refs = [d / r for r in meta.get("refs", [])]
    return d, refs, dict(m) | {"meta": meta}


def generate(db: Database, settings: Settings, engine: XttsEngine, ctx: JobContext, *, mode: str, text: str,
             model_id: int | None = None, ref_wavs: list[Path] | None = None, compare_model_id: int | None = None,
             **params) -> int:
    """Synthesize and record a generation. Returns generations.id."""
    from app.services.training import base_models

    text = validate_text(text)
    language = "ru"
    if mode == "trained":
        d, refs, m = model_refs(db, model_id)
        if m["kind"] != "finetuned":
            raise InferenceError("Select a fine-tuned model for 'Trained voice' mode.")
        language = m["language"]
    elif mode == "pretrained":
        if compare_model_id is None:
            raise InferenceError("Pick the fine-tuned model whose reference clips should be used for comparison.")
        _, refs, _ = model_refs(db, compare_model_id)
        d = base_models.xtts_dir(settings)
    elif mode == "clone":
        refs = [Path(r) for r in (ref_wavs or [])]
        if not refs:
            raise InferenceError("Choose at least one reference audio file (6–30 s of clean speech).")
        d = base_models.xtts_dir(settings)
    else:
        raise ValueError(mode)
    if mode != "trained" and not base_models.xtts_present(settings):
        raise InferenceError("XTTS v2 base weights are not downloaded. Go to Settings → Models.")
    if not refs:
        raise InferenceError("This model has no reference clips; re-export it from the training run.")

    with gpu_stage("tts"):
        try:
            engine.load(d)
            audio = engine.synthesize(text, refs, language, ctx, **params)
        except Exception as e:
            if is_oom(e) and engine.device == "cuda":
                engine.unload()
                raise InferenceError("GPU out of memory during synthesis. Stop training/processing jobs and retry.") from e
            raise
    out = settings.sub("generated_audio") / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}.wav"
    save_wav(out, audio, OUT_SR)
    return db.insert("generations", model_id=model_id if mode == "trained" else (compare_model_id if mode == "pretrained" else None),
                     mode=mode, text=text, output_path=str(out), duration=len(audio) / OUT_SR, created_at=time.time(),
                     params_json=json.dumps({**params, "refs": [str(r) for r in refs], "model_dir": str(d)},
                                            ensure_ascii=False))
