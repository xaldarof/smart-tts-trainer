import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.database.db import confirm_consent
from app.services.dataset.builder import export_dataset, project_stats, validate_dataset
from app.services.training.presets import PRESETS, TrainConfig, oom_downgrade, preset
from app.utils.audio import export_audio, extract_audio, load_mono, save_wav
from app.utils.jobs import JobContext
from tests.conftest import FIXTURES, tone


def _seed_segments(db, settings, project, n=40, dur=4.0, status="accepted"):
    vid = db.insert("videos", project_id=project, youtube_id="v" * 11, url="https://youtu.be/vvvvvvvvvvv",
                    title="Test", download_status="downloaded")
    spk = db.insert("speakers", video_id=vid, label="SPEAKER_00", role="target")
    d = settings.sub("audio") / "segs"
    for i in range(n):
        p = save_wav(d / f"{i}.wav", tone(dur, 44100, 200 + i), 44100)
        db.insert("segments", project_id=project, video_id=vid, speaker_id=spk, start=i * 5.0, end=i * 5.0 + dur,
                  audio_path=str(p), text=f'Фраза номер {i} | с "кавычками"', text_raw="raw", status=status,
                  quality_score=0.9, speaker_sim=0.9, created_at=time.time())
    return vid


def test_stats_and_readiness(db, settings, project):
    _seed_segments(db, settings, project, n=10)
    s = project_stats(db, project)
    assert s.n_accepted == 10 and abs(s.accepted_sec - 40) < 1e-6
    assert not s.ready and "consent" in s.readiness.lower()
    confirm_consent(db, project, "Speaker", "self")
    s = project_stats(db, project)
    assert not s.ready  # 40 s < 2 min
    _seed_segments_more = _seed_segments  # noqa: F841


def test_export_requires_consent(db, settings, project):
    _seed_segments(db, settings, project)
    with pytest.raises(PermissionError):
        export_dataset(db, settings, project, "x", JobContext())


def test_export_and_validate_roundtrip_with_coqui_formatter(db, settings, project):
    _seed_segments(db, settings, project, n=40)
    confirm_consent(db, project, "Test Speaker", "self")
    ds_id = export_dataset(db, settings, project, "my set", JobContext(), eval_fraction=0.1)
    row = db.q1("SELECT * FROM datasets WHERE id=?", (ds_id,))
    path = Path(row["path"])
    assert row["n_train"] + row["n_eval"] == 40 and row["n_eval"] == 4
    rep = validate_dataset(path)
    assert rep.ok, rep.errors
    info = sf.info(str(path / "wavs" / "1.wav"))
    assert info.samplerate == 22050 and info.channels == 1
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert man["clips"][0]["video"] == "v" * 11 and "start" in man["clips"][0]
    # The real Coqui formatter must parse what we wrote (text with quotes/pipes sanitized)
    from TTS.tts.datasets.formatters import coqui

    items = coqui(str(path), "metadata_train.csv")
    assert len(items) == 36
    assert all("|" not in it["text"] and '"' not in it["text"] for it in items)
    assert items[0]["speaker_name"] == "Test_Speaker"
    assert Path(items[0]["audio_file"]).exists()


def test_validate_detects_problems(tmp_path):
    (tmp_path / "wavs").mkdir()
    sf.write(tmp_path / "wavs" / "a.wav", tone(1.0, 44100), 44100)
    with open(tmp_path / "metadata_train.csv", "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\nwavs/a.wav|Текст|s\nwavs/missing.wav||s\n")
    rep = validate_dataset(tmp_path)
    assert not rep.ok
    joined = "\n".join(rep.errors)
    assert "22050" in joined and "missing" in joined and "empty text" in joined and "metadata_eval.csv" in joined


def test_presets_fit_16gb_and_oom_downgrade():
    assert set(PRESETS) == {"low_vram", "balanced", "custom"}
    lv, bal = preset("low_vram"), preset("balanced")
    assert lv.grad_checkpointing and lv.batch_size < bal.batch_size
    assert lv.effective_batch == bal.effective_batch == 32
    c = TrainConfig(batch_size=4, grad_accum=8, grad_checkpointing=False)
    c1 = oom_downgrade(c)
    assert c1.grad_checkpointing and c1.batch_size == 4
    c2 = oom_downgrade(c1)
    assert (c2.batch_size, c2.grad_accum) == (2, 16)
    assert oom_downgrade(TrainConfig(batch_size=1, grad_accum=32, grad_checkpointing=True)) is None
    assert TrainConfig.from_dict({**c.as_dict(), "unknown": 1}).batch_size == 4


def test_model_registry_export(db, settings, project, monkeypatch, tmp_path):
    """export_model strips optimizer state, copies base config/vocab and reference clips."""
    import torch

    from app.services.training import base_models, manager

    base = base_models.xtts_dir(settings)
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "vocab.json").write_text("{}")
    _seed_segments(db, settings, project, n=40, dur=6.0)
    confirm_consent(db, project, "S", "self")
    ds_id = export_dataset(db, settings, project, "d", JobContext())
    run = db.insert("training_runs", project_id=project, dataset_id=ds_id, backend="xtts_v2", preset="balanced",
                    config_json=json.dumps(preset("balanced").as_dict()), status="stopped", output_dir=str(tmp_path),
                    created_at=0, current_step=10)
    exp = tmp_path / "exp"
    exp.mkdir()
    torch.save({"model": {"xtts.gpt.w": torch.ones(2), "dvae.x": torch.ones(1), "xtts.dvae.y": torch.ones(1),
                          "torch_mel_spectrogram_dvae.z": torch.ones(1)}, "optimizer": {"big": torch.ones(100)},
                "step": 10}, exp / "checkpoint_10.pth")
    db.update("training_runs", run, exp_dir=str(exp))
    mid = manager.export_model(db, settings, run, "Voice")
    m = db.q1("SELECT * FROM models WHERE id=?", (mid,))
    out = Path(m["path"])
    sd = torch.load(out / "model.pth", weights_only=False)
    assert set(sd) == {"model"} and set(sd["model"]) == {"xtts.gpt.w"}
    meta = json.loads((out / "model_meta.json").read_text(encoding="utf-8"))
    assert len(meta["refs"]) == 4 and all((out / r).exists() for r in meta["refs"])
    assert m["kind"] == "finetuned" and "CPML" in meta["license"]
    pre = manager.register_pretrained(db, settings)
    assert manager.register_pretrained(db, settings) == pre


def test_audio_extract_and_export(tmp_path):
    src = FIXTURES / "short_two_speakers.wav"
    out = extract_audio(src, tmp_path / "x.wav", 44100)
    x, sr = load_mono(out)
    assert sr == 44100 and abs(len(x) / sr - 40.0) < 0.05
    wav = save_wav(tmp_path / "g.wav", x[: sr * 2], sr)
    mp3 = export_audio(wav, tmp_path / "exp" / "speech", "mp3")
    assert mp3.suffix == ".mp3" and mp3.stat().st_size > 1000
    w2 = export_audio(wav, tmp_path / "exp" / "speech", "wav")
    assert sf.info(str(w2)).frames == sr * 2
    with pytest.raises(ValueError):
        export_audio(wav, tmp_path / "a", "ogg")


def test_extract_rejects_corrupt(tmp_path):
    from app.utils.audio import FFmpegError

    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(FFmpegError):
        extract_audio(bad, tmp_path / "o.wav")
