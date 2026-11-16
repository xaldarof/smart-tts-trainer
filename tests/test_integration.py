"""Integration tests with REAL models (no mocks). Opt-in: LVT_INTEGRATION=1.

They download model weights on first run (ECAPA ~80 MB, Whisper tiny ~75 MB, XTTS v2 ~1.9 GB
for the training test). The XTTS test additionally needs LVT_ACCEPT_CPML=1.
"""
from __future__ import annotations

import os

import pytest

from tests.conftest import FIXTURES

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    # reuse the developer's model cache if present to avoid re-downloads
    from app.config.settings import PROJECT_ROOT

    c = PROJECT_ROOT / "data" / "model_cache"
    return c if c.exists() else tmp_path_factory.mktemp("cache")


def test_vad_and_ecapa_diarization(shared_cache):
    from app.services.audio.vad import detect_speech
    from app.services.diarization.diarizer import SpeakerEmbedder, cosine, diarize_ecapa, speaker_embedding
    from app.utils.audio import load_mono
    from app.utils.jobs import JobContext

    wav = FIXTURES / "short_two_speakers.wav"
    speech = detect_speech(wav, JobContext())
    assert 8 <= len(speech) <= 30
    emb = SpeakerEmbedder("cpu", shared_cache)
    res = diarize_ecapa(wav, speech, emb, JobContext())
    d = res.durations()
    assert len(d) == 2, d
    x, _ = load_mono(wav, 16000)
    e = {lab: speaker_embedding(emb, x, [(s, en) for s, en, l2 in res.turns if l2 == lab]) for lab in d}
    a, b = list(e.values())
    assert cosine(a, b) < 0.6


def test_whisper_russian(shared_cache):
    from app.services.transcription.whisper import Transcriber
    from app.utils.jobs import JobContext

    tr = Transcriber("tiny", "cpu", cache_dir=shared_cache / "whisper")
    segs = tr.transcribe(FIXTURES / "short_two_speakers.wav", JobContext(), "ru", beam_size=1)
    text = " ".join(s.text for s in segs).lower()
    assert any(w in text for w in ("синтез", "речи", "голос", "записи")), text
    assert all(w.start <= w.end for s in segs for w in s.words)


@pytest.mark.skipif(os.environ.get("LVT_ACCEPT_CPML") != "1", reason="set LVT_ACCEPT_CPML=1 (XTTS v2 is CPML)")
def test_xtts_finetune_resume_export_infer(tmp_path):
    """Real XTTS GPT fine-tuning steps + checkpoint resume + export + synthesis (CPU ok, ~10 min)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gpu_smoke_test

    from app.config.settings import PROJECT_ROOT

    # point at a temp data dir but reuse cached XTTS weights via symlink/copy-free path
    data = tmp_path / "data"
    (data / "model_cache").mkdir(parents=True)
    src = PROJECT_ROOT / "data" / "model_cache" / "xtts_v2"
    if src.exists():
        (data / "model_cache" / "xtts_v2").symlink_to(src, target_is_directory=True)
    res = gpu_smoke_test.run(steps=2, allow_cpu=True, accept_cpml=True, data_dir=data)
    failed = {k: v for k, v in res["steps"].items() if v["status"] != "PASS"}
    assert not failed, failed
    assert len(res["steps"]) == 5
