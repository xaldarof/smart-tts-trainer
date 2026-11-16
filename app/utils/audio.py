"""Audio I/O and FFmpeg helpers."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class FFmpegError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError(
            "FFmpeg was not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`) and restart the app, "
            "or run scripts\\install.ps1 which checks for it."
        )
    return exe


def run_ffmpeg(args: list[str], timeout: float | None = None) -> None:
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise FFmpegError(f"FFmpeg failed ({p.returncode}): {p.stderr.strip()[-800:]}")


def extract_audio(src: Path, dst: Path, sample_rate: int = 44100) -> Path:
    """Decode any media file into mono PCM16 WAV. Raises FFmpegError for corrupt input."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)])
    info = sf.info(str(tmp))
    if info.frames == 0:
        tmp.unlink(missing_ok=True)
        raise FFmpegError(f"No audio stream could be decoded from {src.name}; the file may be corrupted.")
    os.replace(tmp, dst)
    return dst


def load_mono(path: Path | str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono, optionally resampled."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr and sr != file_sr:
        data = resample(data, file_sr, sr)
        file_sr = sr
    return data, file_sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def save_wav(path: Path | str, data: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(str(tmp), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
    os.replace(tmp, path)
    return path


def export_audio(src_wav: Path, dst: Path, fmt: str, mp3_bitrate: str = "192k") -> Path:
    """Export WAV to wav/mp3. MP3 uses FFmpeg's libmp3lame."""
    fmt = fmt.lower()
    dst = dst.with_suffix("." + fmt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "wav":
        if Path(src_wav).resolve() != dst.resolve():
            shutil.copyfile(src_wav, dst)
    elif fmt == "mp3":
        run_ffmpeg(["-i", str(src_wav), "-c:a", "libmp3lame", "-b:a", mp3_bitrate, str(dst)])
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return dst


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def rms_db(x: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0)


def clipping_ratio(x: np.ndarray, thresh: float = 0.999) -> float:
    return float(np.mean(np.abs(x) >= thresh)) if x.size else 0.0


def estimate_snr_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Rough SNR: loud-frame energy (speech) vs. quiet-frame energy (noise floor).

    Uses frame RMS percentiles (90th vs 10th). Adequate for ranking/filtering clips,
    not a calibrated measurement.
    """
    n = max(1, int(sr * frame_ms / 1000))
    if x.size < n * 4:
        return 0.0
    frames = x[: (x.size // n) * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    return float(20 * np.log10(np.percentile(rms, 90) / np.percentile(rms, 10)))
