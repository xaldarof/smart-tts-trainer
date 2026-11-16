"""Application settings: paths, processing thresholds, and user preferences.

Settings are persisted as JSON in ``<data_dir>/settings.json``. Secrets (the
Hugging Face token used for gated pyannote models) are NOT stored here; they are
read from the ``HF_TOKEN`` environment variable or from the OS user's
``huggingface-cli login`` cache.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "Local Voice Trainer"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    env = os.environ.get("LVT_DATA_DIR")
    return Path(env) if env else PROJECT_ROOT / "data"


@dataclass
class ProcessingSettings:
    # Enhancement
    music_separation: bool = True          # Demucs vocals stem (GPU-heavy, removes music)
    denoise: bool = False                  # mild spectral gating (noisereduce); off by default
    denoise_strength: float = 0.5          # 0..1, prop_decrease
    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    # Diarization
    diarization_backend: str = "auto"      # auto | pyannote | ecapa
    num_speakers: int = 0                  # 0 = automatic
    ecapa_cluster_threshold: float = 0.55  # cosine distance threshold for agglomerative clustering
    # Transcription
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "auto"     # auto | float16 | int8_float16 | int8 | float32
    whisper_beam_size: int = 5
    language: str = "ru"
    # Segmentation / filtering
    min_segment_sec: float = 1.5
    max_segment_sec: float = 11.0
    split_pause_sec: float = 0.35
    min_word_prob: float = 0.40            # any word below -> review
    min_avg_word_prob: float = 0.70        # mean below -> reject
    max_clipping_ratio: float = 0.001
    min_snr_db: float = 15.0
    min_rms_db: float = -40.0
    min_chars_per_sec: float = 5.0
    max_chars_per_sec: float = 25.0
    speaker_sim_accept: float = 0.70       # segment-to-target cosine similarity: accept above
    speaker_sim_reject: float = 0.45       # reject below; in-between -> review
    eval_fraction: float = 0.05


@dataclass
class TrainingDefaults:
    preset: str = "balanced"


@dataclass
class Settings:
    data_dir: str = field(default_factory=lambda: str(default_data_dir()))
    device: str = "auto"                  # auto | cuda | cpu
    cpml_accepted: bool = False           # Coqui Public Model License (XTTS v2 weights)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    max_parallel_downloads: int = 2
    download_retries: int = 4

    # ------------------------------------------------------------------ paths
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def sub(self, name: str) -> Path:
        p = self.data_path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_path / "lvt.sqlite3"

    @property
    def models_cache(self) -> Path:
        return self.sub("model_cache")

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or default_data_dir() / "settings.json"
        s = cls()
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            s = _from_dict(cls, raw)
        s.data_path.mkdir(parents=True, exist_ok=True)
        return s

    def save(self) -> None:
        path = self.data_path / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


def _from_dict(cls, raw: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        if f.name == "processing":
            val = _from_dict(ProcessingSettings, val)
        elif f.name == "training":
            val = _from_dict(TrainingDefaults, val)
        kwargs[f.name] = val
    return cls(**kwargs)


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None
