"""Pretrained model weights management (download, verify, license gating)."""
from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import Settings
from app.utils.jobs import JobContext
from app.utils.system_check import ensure_free_space

log = logging.getLogger(__name__)

XTTS_REPO = "coqui/XTTS-v2"
XTTS_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"   # pinned for reproducibility
XTTS_FILES = ["config.json", "vocab.json", "mel_stats.pth", "dvae.pth", "speakers_xtts.pth", "model.pth"]
XTTS_LICENSE_URL = "https://coqui.ai/cpml"
XTTS_LICENSE_TEXT = (
    "XTTS v2 model weights are released under the Coqui Public Model License (CPML) 1.0.0. "
    "The CPML permits NON-COMMERCIAL use only: you may not use the weights, or any model fine-tuned "
    "from them, or audio they generate, for commercial purposes. Read the full license at "
    f"{XTTS_LICENSE_URL} before continuing."
)


class LicenseNotAccepted(PermissionError):
    pass


def xtts_dir(settings: Settings) -> Path:
    return settings.models_cache / "xtts_v2"


def xtts_present(settings: Settings) -> bool:
    d = xtts_dir(settings)
    return all((d / f).exists() and (d / f).stat().st_size > 0 for f in XTTS_FILES)


def ensure_xtts(settings: Settings, ctx: JobContext) -> Path:
    """Download XTTS v2 weights (≈1.9 GB) once. Requires prior CPML acceptance in Settings."""
    d = xtts_dir(settings)
    if xtts_present(settings):
        return d
    if not settings.cpml_accepted:
        raise LicenseNotAccepted("Accept the Coqui Public Model License (non-commercial) in Settings → Models "
                                 "before downloading XTTS v2.")
    ensure_free_space(settings.models_cache, 3.0, "XTTS v2 model download")
    from huggingface_hub import hf_hub_download

    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(XTTS_FILES):
        ctx.check()
        ctx.progress(i / len(XTTS_FILES), f"Downloading XTTS v2: {f}")
        hf_hub_download(XTTS_REPO, f, revision=XTTS_REVISION, local_dir=str(d))
    ctx.progress(1.0, "XTTS v2 ready")
    return d


def prefetch_processing_models(settings: Settings, ctx: JobContext) -> None:
    """Download processing models so later stages work offline."""
    from faster_whisper.utils import download_model

    ctx.progress(0.05, "Silero VAD")
    from silero_vad import load_silero_vad

    load_silero_vad()
    ctx.progress(0.15, "ECAPA speaker embeddings")
    from app.services.diarization.diarizer import SpeakerEmbedder

    SpeakerEmbedder("cpu", settings.models_cache).close()
    ctx.check()
    ctx.progress(0.3, f"faster-whisper {settings.processing.whisper_model}")
    download_model(settings.processing.whisper_model, cache_dir=str(settings.models_cache / "whisper"))
    ctx.check()
    if settings.processing.music_separation:
        ctx.progress(0.8, "Demucs htdemucs")
        from demucs.pretrained import get_model

        get_model("htdemucs")
    ctx.progress(1.0, "Processing models ready")
