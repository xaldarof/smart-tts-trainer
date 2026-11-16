"""Dataset statistics, export to the Coqui/XTTS trainer format, and validation.

Export layout (consumed by ``TTS.tts.datasets.formatters.coqui``)::

    <dataset>/
        wavs/<segment_id>.wav        22.05 kHz mono PCM16 (XTTS GPT training rate)
        metadata_train.csv           audio_file|text|speaker_name   (header row)
        metadata_eval.csv
        manifest.json                source video + timestamps for every clip
"""
from __future__ import annotations

import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

from app.config.settings import Settings
from app.database.db import Database, has_consent
from app.utils.audio import load_mono, save_wav
from app.utils.jobs import JobContext
from app.utils.system_check import ensure_free_space

XTTS_SR = 22050
XTTS_MAX_WAV_SEC = 255995 / XTTS_SR      # GPTArgs.max_wav_length
XTTS_MAX_TEXT = 200                      # GPTArgs.max_text_length
MIN_TOTAL_MIN = 2.0                      # hard minimum to start training
RECOMMENDED_MIN = 20.0


@dataclass
class DatasetStats:
    n_total: int = 0
    n_accepted: int = 0
    n_review: int = 0
    n_rejected: int = 0
    accepted_sec: float = 0.0
    avg_clip_sec: float = 0.0
    missing_transcripts: int = 0
    speakers: dict[str, float] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ready: bool = False
    readiness: str = ""


def project_stats(db: Database, project_id: int) -> DatasetStats:
    s = DatasetStats()
    rows = db.q("SELECT s.status, s.end - s.start AS dur, s.text, s.reasons, v.title, sp.label "
                "FROM segments s JOIN videos v ON v.id=s.video_id LEFT JOIN speakers sp ON sp.id=s.speaker_id "
                "WHERE s.project_id=?", (project_id,))
    s.n_total = len(rows)
    for r in rows:
        if r["status"] == "accepted":
            s.n_accepted += 1
            s.accepted_sec += r["dur"]
            key = f"{(r['title'] or '')[:40]} / {r['label']}"
            s.speakers[key] = s.speakers.get(key, 0.0) + r["dur"]
        elif r["status"] == "review":
            s.n_review += 1
        else:
            s.n_rejected += 1
            for reason in (r["reasons"] or "").split("; "):
                k = re.sub(r"\s*\(.*\)$", "", reason).strip()
                if k:
                    s.reject_reasons[k] = s.reject_reasons.get(k, 0) + 1
        if r["status"] != "rejected" and not (r["text"] or "").strip():
            s.missing_transcripts += 1
    s.avg_clip_sec = s.accepted_sec / s.n_accepted if s.n_accepted else 0.0
    mins = s.accepted_sec / 60
    if s.n_review:
        s.warnings.append(f"{s.n_review} clips need review (they are excluded until you accept them).")
    if s.missing_transcripts:
        s.warnings.append(f"{s.missing_transcripts} clips have empty transcripts.")
    if mins < MIN_TOTAL_MIN:
        s.readiness = f"Not ready: {mins:.1f} min accepted, at least {MIN_TOTAL_MIN:.0f} min required."
    elif mins < RECOMMENDED_MIN:
        s.ready = True
        s.readiness = (f"Trainable but small: {mins:.1f} min. {RECOMMENDED_MIN:.0f}+ min of clean speech gives "
                       "noticeably better fine-tuning results.")
    else:
        s.ready = True
        s.readiness = f"Ready: {mins:.1f} min of accepted speech."
    if not has_consent(db, project_id):
        s.ready = False
        s.readiness = "Speaker consent has not been confirmed for this project."
    return s


def export_dataset(db: Database, settings: Settings, project_id: int, name: str, ctx: JobContext,
                   eval_fraction: float | None = None, seed: int = 1234) -> int:
    if not has_consent(db, project_id):
        raise PermissionError("Speaker consent must be confirmed before exporting a training dataset.")
    rows = db.q("SELECT s.*, v.youtube_id, v.url, v.title FROM segments s JOIN videos v ON v.id=s.video_id "
                "WHERE s.project_id=? AND s.status='accepted' ORDER BY s.id", (project_id,))
    rows = [r for r in rows if (r["text"] or "").strip()]
    if not rows:
        raise ValueError("No accepted segments with transcripts. Review segments in the Dataset Manager first.")
    total = sum(r["end"] - r["start"] for r in rows)
    if total / 60 < MIN_TOTAL_MIN:
        raise ValueError(f"Only {total / 60:.1f} min of accepted speech; at least {MIN_TOTAL_MIN:.0f} min is required.")
    ensure_free_space(settings.data_path, total * XTTS_SR * 2 / 1e9 * 1.2 + 0.5, "dataset export")
    safe = re.sub(r"[^\w\-]+", "_", name).strip("_") or "dataset"
    out = settings.sub("datasets") / str(project_id) / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}"
    wavs = out / "wavs"
    wavs.mkdir(parents=True)
    speaker = (db.scalar("SELECT target_speaker_name FROM projects WHERE id=?", (project_id,)) or "speaker")
    speaker = re.sub(r"[|\s]+", "_", speaker)
    items, manifest = [], []
    for i, r in enumerate(rows):
        ctx.check()
        x, _ = load_mono(r["audio_path"], XTTS_SR)
        if len(x) / XTTS_SR > XTTS_MAX_WAV_SEC:
            continue
        fn = f"{r['id']}.wav"
        save_wav(wavs / fn, x, XTTS_SR)
        # Coqui's formatter reads with csv default quoting; drop '"' (unpronounced) and '|' (delimiter).
        text = re.sub(r"\s+", " ", r["text"].replace("|", " ").replace('"', "")).strip()[:XTTS_MAX_TEXT]
        items.append((f"wavs/{fn}", text, speaker))
        manifest.append({"file": f"wavs/{fn}", "segment_id": r["id"], "video": r["youtube_id"], "url": r["url"],
                         "title": r["title"], "start": r["start"], "end": r["end"], "text": text,
                         "text_raw": r["text_raw"], "quality": r["quality_score"], "speaker_sim": r["speaker_sim"]})
        ctx.progress((i + 1) / len(rows), f"Exporting clip {i + 1}/{len(rows)}")
    frac = settings.processing.eval_fraction if eval_fraction is None else eval_fraction
    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    n_eval = max(1, min(int(round(len(items) * frac)), 256)) if len(items) > 4 else 1
    eval_idx = set(idx[:n_eval])
    train = [items[i] for i in range(len(items)) if i not in eval_idx]
    evals = [items[i] for i in range(len(items)) if i in eval_idx]
    for fname, data in (("metadata_train.csv", train), ("metadata_eval.csv", evals)):
        with open(out / fname, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
            w.writerow(["audio_file", "text", "speaker_name"])
            w.writerows(data)
    dur = sum(m["end"] - m["start"] for m in manifest)
    lang = settings.processing.language
    (out / "manifest.json").write_text(json.dumps({"language": lang, "speaker": speaker, "sample_rate": XTTS_SR,
                                                   "clips": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    return db.insert("datasets", project_id=project_id, name=name, path=str(out), language=lang, n_train=len(train),
                     n_eval=len(evals), total_duration=dur, created_at=time.time(),
                     meta_json=json.dumps({"speaker": speaker}))


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    n_train: int
    n_eval: int
    total_sec: float

    def as_dict(self):
        return asdict(self)


def validate_dataset(path: Path) -> ValidationReport:
    """Check an exported dataset against XTTS trainer constraints."""
    errors, warnings = [], []
    path = Path(path)
    counts, total = {}, 0.0
    for split in ("train", "eval"):
        meta = path / f"metadata_{split}.csv"
        if not meta.exists():
            errors.append(f"Missing {meta.name}")
            counts[split] = 0
            continue
        with open(meta, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")  # same dialect as TTS formatters.coqui
            if reader.fieldnames is None or not {"audio_file", "text"} <= set(reader.fieldnames):
                errors.append(f"{meta.name}: header must contain audio_file|text|speaker_name")
                counts[split] = 0
                continue
            n = 0
            for i, row in enumerate(reader, start=2):
                n += 1
                wav = path / row["audio_file"]
                text = (row.get("text") or "").strip()
                if not text:
                    errors.append(f"{meta.name}:{i}: empty text")
                if len(text) > XTTS_MAX_TEXT:
                    errors.append(f"{meta.name}:{i}: text longer than {XTTS_MAX_TEXT} chars")
                if not wav.exists():
                    errors.append(f"{meta.name}:{i}: missing audio {row['audio_file']}")
                    continue
                try:
                    info = sf.info(str(wav))
                except Exception as e:
                    errors.append(f"{row['audio_file']}: unreadable ({e})")
                    continue
                if info.samplerate != XTTS_SR or info.channels != 1:
                    errors.append(f"{row['audio_file']}: must be {XTTS_SR} Hz mono, got {info.samplerate} Hz x{info.channels}")
                if info.duration > XTTS_MAX_WAV_SEC:
                    errors.append(f"{row['audio_file']}: {info.duration:.1f}s exceeds {XTTS_MAX_WAV_SEC:.1f}s")
                if info.duration < 0.5:
                    warnings.append(f"{row['audio_file']}: very short ({info.duration:.2f}s)")
                total += info.duration
            counts[split] = n
        if len(errors) > 50:
            errors.append("… too many errors, stopping")
            break
    if counts.get("train", 0) == 0:
        errors.append("Training split is empty.")
    if counts.get("eval", 0) == 0:
        errors.append("Evaluation split is empty.")
    if total / 60 < MIN_TOTAL_MIN:
        errors.append(f"Only {total / 60:.1f} min of audio; at least {MIN_TOTAL_MIN:.0f} min required.")
    elif total / 60 < RECOMMENDED_MIN:
        warnings.append(f"{total / 60:.1f} min of audio; {RECOMMENDED_MIN:.0f}+ min recommended.")
    return ValidationReport(not errors, errors, warnings, counts.get("train", 0), counts.get("eval", 0), total)
