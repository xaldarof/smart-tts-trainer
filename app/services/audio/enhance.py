"""Speech enhancement: Demucs vocal isolation (removes music/background) + optional mild denoise.

DeepFilterNet was evaluated and NOT used: its last release (0.5.6, 2023) pins
numpy<2 and imports torchaudio APIs removed in torchaudio >= 2.1, so it cannot be
installed alongside the rest of this stack. Demucs (htdemucs, MIT) is actively
maintained and isolates the vocal stem, which handles the dominant YouTube problem
(background music). A gentle spectral-gating denoiser (noisereduce) is available
but disabled by default, because aggressive denoising audibly distorts timbre and
that distortion gets baked into a fine-tuned voice.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.utils.audio import load_mono, save_wav
from app.utils.gpu import free_gpu_memory, is_oom
from app.utils.jobs import Cancelled, JobContext

log = logging.getLogger(__name__)


class Enhancer:
    def __init__(self, device: str, music_separation: bool = True, denoise: bool = False, denoise_strength: float = 0.5):
        self.device = device
        self.music_separation = music_separation
        self.denoise = denoise
        self.denoise_strength = float(np.clip(denoise_strength, 0.0, 1.0))
        self._sep = None

    def _separator(self, device: str):
        if self._sep is None or str(self._sep._device) != device:
            from demucs.api import Separator

            # segment ~7.8 s is htdemucs' training length; keeps VRAM under ~3 GB.
            self._sep = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True, progress=False)
        return self._sep

    def close(self) -> None:
        self._sep = None
        free_gpu_memory()

    def process(self, src_wav: Path, dst_wav: Path, ctx: JobContext) -> dict:
        x, sr = load_mono(src_wav)
        meta = {"music_separation": False, "denoise": False}
        if self.music_separation:
            x = self._separate(x, sr, ctx)
            meta["music_separation"] = True
        ctx.check()
        if self.denoise and self.denoise_strength > 0:
            import noisereduce as nr

            ctx.progress(0.95, "Mild denoise")
            x = nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=self.denoise_strength).astype(np.float32)
            meta["denoise"] = True
            meta["denoise_strength"] = self.denoise_strength
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.99:  # separation can push peaks above full scale
            x = x * (0.99 / peak)
        save_wav(dst_wav, x, sr)
        return meta

    def _separate(self, x: np.ndarray, sr: int, ctx: JobContext) -> np.ndarray:
        import torch

        def cb(d: dict):
            if ctx.cancelled:
                raise Cancelled("Cancelled during source separation")
            if d.get("audio_length"):
                ctx.progress(0.9 * d["segment_offset"] / d["audio_length"], "Isolating vocals (Demucs)")

        def run(device: str) -> np.ndarray:
            sep = self._separator(device)
            sep.update_parameter(callback=cb)
            wav = torch.from_numpy(np.stack([x, x])).float()  # htdemucs expects stereo
            _, stems = sep.separate_tensor(wav, sr)
            vocals = stems["vocals"].mean(0).cpu().numpy().astype(np.float32)
            from app.utils.audio import resample

            return resample(vocals, sep.samplerate, sr)

        try:
            return run(self.device)
        except Exception as e:
            if self.device == "cuda" and is_oom(e):
                log.warning("Demucs OOM on GPU, falling back to CPU")
                self._sep = None
                free_gpu_memory()
                return run("cpu")
            raise
