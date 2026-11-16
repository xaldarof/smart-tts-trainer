"""Processing pipeline orchestration.

Stages per video (each recorded in ``video_stages``; completed stages are skipped on
re-run, so a crash or reboot never repeats finished work):

    extract   -> source.wav        (FFmpeg, mono 44.1 kHz; original download preserved)
    enhance   -> enhanced.wav      (Demucs vocals [+ optional mild denoise])
    vad       -> vad.json          (Silero VAD on enhanced audio)
    diarize   -> diarization.json  (+ speakers/turns rows, ECAPA speaker embeddings, previews)
    transcribe-> transcript.json   (faster-whisper, word timestamps)

Enhancement runs before VAD/diarization because background music is the main cause
of false VAD triggers and spurious "speakers" on YouTube audio.

Work is executed *stage-major* (all videos through stage 1, then stage 2, ...) so each
GPU model is loaded once and fully released before the next one, keeping peak VRAM
well under 16 GB.

Segment building (``build_segments``) runs only after the user has selected the target
speaker(s) and confirmed consent.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

from app.config.settings import Settings
from app.database.db import Database, has_consent
from app.services.audio import vad as vadmod
from app.utils.audio import extract_audio, load_mono, save_wav
from app.utils.gpu import free_gpu_memory, gpu_stage, pick_device
from app.utils.jobs import Cancelled, JobContext
from app.utils.system_check import ensure_free_space

log = logging.getLogger(__name__)

STAGES = ["extract", "enhance", "vad", "diarize", "transcribe"]
WORK_SR = 44100


def video_dir(settings: Settings, project_id: int, video_id: int) -> Path:
    d = settings.sub("audio") / str(project_id) / str(video_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Pipeline:
    def __init__(self, db: Database, settings: Settings):
        self.db, self.settings, self.cfg = db, settings, settings.processing
        self.device = pick_device(settings.device)
        self._embedder = None

    # ------------------------------------------------------------- helpers
    def embedder(self):
        if self._embedder is None:
            from app.services.diarization.diarizer import SpeakerEmbedder

            self._embedder = SpeakerEmbedder(self.device, self.settings.models_cache)
        return self._embedder

    def release(self):
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None
        free_gpu_memory()

    def _run_stage(self, stage: str, video_ids: list[int], ctx: JobContext, fn) -> dict[int, str]:
        errors: dict[int, str] = {}
        todo = [v for v in video_ids if not self.db.stage_done(v, stage)]
        for i, vid in enumerate(todo):
            ctx.check()
            # skip if an earlier stage failed for this video
            prev = STAGES[: STAGES.index(stage)]
            if any(not self.db.stage_done(vid, p) for p in prev):
                continue
            title = self.db.scalar("SELECT title FROM videos WHERE id=?", (vid,)) or f"video {vid}"
            sub = ctx.sub(i / len(todo), (i + 1) / len(todo))
            sub.progress(0.0, f"{stage}: {title}")
            self.db.mark_stage(vid, stage, "running")
            try:
                out, meta = fn(vid, video_dir(self.settings, self._pid(vid), vid), sub)
                self.db.mark_stage(vid, stage, "done", str(out), meta)
            except Cancelled:
                self.db.mark_stage(vid, stage, "cancelled")
                raise
            except Exception as e:
                msg = f"{stage} failed for '{title}': {e}"
                log.error(msg)
                self.db.mark_stage(vid, stage, "failed", error=str(e))
                self.db.log_error(f"pipeline:{stage}:{vid}", msg, traceback.format_exc())
                errors[vid] = msg
        return errors

    def _pid(self, vid: int) -> int:
        return int(self.db.scalar("SELECT project_id FROM videos WHERE id=?", (vid,)))

    # -------------------------------------------------------------- stages
    def run(self, video_ids: list[int], ctx: JobContext) -> dict[int, str]:
        ensure_free_space(self.settings.data_path, 2.0, "audio processing")
        errors: dict[int, str] = {}
        weights = {"extract": (0.0, 0.05), "enhance": (0.05, 0.4), "vad": (0.4, 0.45), "diarize": (0.45, 0.7),
                   "transcribe": (0.7, 1.0)}
        try:
            errors |= self._run_stage("extract", video_ids, ctx.sub(*weights["extract"]), self._extract)
            with gpu_stage("enhance"):
                from app.services.audio.enhance import Enhancer

                enh = Enhancer(self.device, self.cfg.music_separation, self.cfg.denoise, self.cfg.denoise_strength)
                try:
                    errors |= self._run_stage("enhance", video_ids, ctx.sub(*weights["enhance"]),
                                              lambda v, d, c: self._enhance(enh, v, d, c))
                finally:
                    enh.close()
            errors |= self._run_stage("vad", video_ids, ctx.sub(*weights["vad"]), self._vad)
            with gpu_stage("diarize"):
                try:
                    errors |= self._run_stage("diarize", video_ids, ctx.sub(*weights["diarize"]), self._diarize)
                finally:
                    self.release()
            with gpu_stage("transcribe"):
                from app.services.transcription.whisper import Transcriber

                tr = Transcriber(self.cfg.whisper_model, self.device, self.cfg.whisper_compute_type,
                                 self.settings.models_cache / "whisper")
                try:
                    errors |= self._run_stage("transcribe", video_ids, ctx.sub(*weights["transcribe"]),
                                              lambda v, d, c: self._transcribe(tr, v, d, c))
                finally:
                    tr.close()
        finally:
            self.release()
        ctx.progress(1.0, "Processing finished" + (f" with {len(errors)} error(s)" if errors else ""))
        return errors

    def _extract(self, vid: int, d: Path, ctx: JobContext):
        row = self.db.q1("SELECT download_path FROM videos WHERE id=?", (vid,))
        src = Path(row["download_path"] or "")
        if not src.exists():
            raise FileNotFoundError(f"Downloaded file missing: {src}. Re-download the video.")
        out = extract_audio(src, d / "source.wav", WORK_SR)
        x, sr = load_mono(out)
        from app.utils.audio import clipping_ratio, rms_db

        dur = len(x) / sr
        self.db.execute("UPDATE videos SET duration=? WHERE id=? AND (duration IS NULL OR duration=0)", (dur, vid))
        return out, {"duration": dur, "rms_db": rms_db(x), "clipping": clipping_ratio(x)}

    def _enhance(self, enh, vid: int, d: Path, ctx: JobContext):
        src = d / "source.wav"
        out = d / "enhanced.wav"
        meta = enh.process(src, out, ctx)
        return out, meta

    def _vad(self, vid: int, d: Path, ctx: JobContext):
        wav = d / "enhanced.wav"
        regions = vadmod.detect_speech(wav, ctx, self.cfg.vad_threshold, self.cfg.vad_min_speech_ms,
                                       self.cfg.vad_min_silence_ms)
        ext = self.db.stage_done(vid, "extract")
        total = float(json.loads(ext["meta_json"] or "{}").get("duration", 0.0)) if ext else 0.0
        if not regions:
            raise ValueError("No speech detected in this audio (music-only or silent).")
        out = d / "vad.json"
        _write_json(out, {"speech": regions, "silence": vadmod.silence_regions(regions, total)})
        return out, vadmod.speech_stats(regions, total)

    def _diarize(self, vid: int, d: Path, ctx: JobContext):
        from app.services.diarization import diarizer as dz

        wav = d / "enhanced.wav"
        speech = [tuple(r) for r in read_json(d / "vad.json")["speech"]]
        backend = self.cfg.diarization_backend
        res = None
        if backend in ("auto", "pyannote"):
            try:
                res = dz.diarize_pyannote(wav, self.device, ctx.sub(0, 0.7), self.cfg.num_speakers)
            except dz.DiarizationError as e:
                if backend == "pyannote":
                    raise
                log.warning("pyannote unavailable (%s); using ECAPA clustering fallback", e)
            except Cancelled:
                raise
            except Exception as e:
                if backend == "pyannote":
                    raise dz.DiarizationError(f"pyannote diarization failed: {e}") from e
                log.warning("pyannote failed (%s); using ECAPA clustering fallback", e)
                free_gpu_memory()
        if res is None:
            res = dz.diarize_ecapa(wav, speech, self.embedder(), ctx.sub(0, 0.7), self.cfg.num_speakers,
                                   self.cfg.ecapa_cluster_threshold)
        # keep only speech inside VAD regions (removes diarization on residual noise)
        turns = []
        for s, e, lab in res.turns:
            for a, b in vadmod.intersect([(s, e)], speech):
                if b - a >= 0.2:
                    turns.append((a, b, lab))
        if not turns:
            raise ValueError("Diarization produced no speaker turns.")
        res.turns = turns
        ctx.progress(0.75, "Computing speaker embeddings")
        x16, _ = load_mono(wav, 16000)
        xw, srw = load_mono(wav)
        pid = self._pid(vid)
        target = self._target_embedding(pid)
        with self.db.tx() as c:
            c.execute("DELETE FROM speakers WHERE video_id=?", (vid,))  # cascades turns
        spk_dir = d / "speakers"
        for lab in res.labels():
            ctx.check()
            lt = [(s, e) for s, e, l2 in turns if l2 == lab]
            emb = dz.speaker_embedding(self.embedder(), x16, lt)
            prev = spk_dir / f"{lab}.wav"
            save_wav(prev, dz.make_preview(xw, srw, lt), srw)
            sim = dz.cosine(emb, target) if (emb is not None and target is not None) else None
            sid = self.db.insert("speakers", video_id=vid, label=lab, total_duration=sum(e - s for s, e in lt),
                                 n_turns=len(lt), preview_path=str(prev),
                                 embedding=emb.tobytes() if emb is not None else None, similarity=sim)
            with self.db.tx() as c:
                c.executemany("INSERT INTO turns(video_id, speaker_id, start, end) VALUES(?,?,?,?)",
                              [(vid, sid, s, e) for s, e in lt])
        out = d / "diarization.json"
        _write_json(out, {"backend": res.backend, "turns": turns, "uncertain": res.uncertain})
        return out, {"backend": res.backend, "speakers": res.durations(), "uncertain_sec":
                     vadmod.total_duration(res.uncertain)}

    def _transcribe(self, tr, vid: int, d: Path, ctx: JobContext):
        from app.services.transcription.whisper import to_json

        segs = tr.transcribe(d / "enhanced.wav", ctx, self.cfg.language, self.cfg.whisper_beam_size)
        if not segs:
            raise ValueError("Whisper produced no transcript (no recognizable speech).")
        out = d / "transcript.json"
        _write_json(out, {"model": tr.model_size, "language": self.cfg.language, "segments": to_json(segs)})
        return out, {"n_segments": len(segs), "n_words": sum(len(s.words) for s in segs)}

    # ------------------------------------------------------- speaker roles
    def _target_embedding(self, project_id: int) -> np.ndarray | None:
        b = self.db.scalar("SELECT target_embedding FROM projects WHERE id=?", (project_id,))
        return np.frombuffer(b, dtype=np.float32) if b else None


def set_speaker_role(db: Database, speaker_id: int, role: str) -> None:
    if role not in ("unassigned", "target", "excluded"):
        raise ValueError(role)
    db.update("speakers", speaker_id, role=role)
    pid = db.scalar("SELECT v.project_id FROM speakers s JOIN videos v ON v.id=s.video_id WHERE s.id=?", (speaker_id,))
    refresh_target_embedding(db, int(pid))


def refresh_target_embedding(db: Database, project_id: int) -> None:
    """Target embedding = normalized mean of embeddings of user-confirmed target speakers.
    Updates cross-video similarity for every speaker in the project."""
    from app.services.diarization.diarizer import cosine

    rows = db.q("SELECT s.embedding FROM speakers s JOIN videos v ON v.id=s.video_id "
                "WHERE v.project_id=? AND s.role='target' AND s.embedding IS NOT NULL", (project_id,))
    if not rows:
        db.execute("UPDATE projects SET target_embedding=NULL WHERE id=?", (project_id,))
        db.execute("UPDATE speakers SET similarity=NULL WHERE video_id IN (SELECT id FROM videos WHERE project_id=?)",
                   (project_id,))
        return
    E = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    t = E.mean(0)
    t = (t / (np.linalg.norm(t) + 1e-9)).astype(np.float32)
    db.update("projects", project_id, target_embedding=t.tobytes())
    with db.tx() as c:
        for r in c.execute("SELECT s.id, s.embedding FROM speakers s JOIN videos v ON v.id=s.video_id "
                           "WHERE v.project_id=?", (project_id,)).fetchall():
            if r["embedding"]:
                c.execute("UPDATE speakers SET similarity=? WHERE id=?",
                          (cosine(np.frombuffer(r["embedding"], dtype=np.float32), t), r["id"]))


# ---------------------------------------------------------- segment building
def build_segments(db: Database, settings: Settings, project_id: int, video_ids: list[int], ctx: JobContext) -> dict:
    """Create dataset candidate segments for the user-selected target speaker(s)."""
    from app.services.dataset import quality as q
    from app.services.dataset.segmenter import AWord, assign_speakers, build_candidates
    from app.services.diarization.diarizer import SpeakerEmbedder, cosine
    from app.services.transcription.normalize import normalize_ru

    if not has_consent(db, project_id):
        raise PermissionError("Confirm the speaker's consent for this project before building a dataset.")
    cfg = settings.processing
    target_b = db.scalar("SELECT target_embedding FROM projects WHERE id=?", (project_id,))
    target = np.frombuffer(target_b, dtype=np.float32) if target_b else None
    device = pick_device(settings.device)
    stats = {"created": 0, "accepted": 0, "review": 0, "rejected": 0, "videos": 0}
    with gpu_stage("segments"):
        emb = SpeakerEmbedder(device, settings.models_cache)
        try:
            for i, vid in enumerate(video_ids):
                ctx.check()
                sub = ctx.sub(i / max(1, len(video_ids)), (i + 1) / max(1, len(video_ids)))
                if not all(db.stage_done(vid, s) for s in ("diarize", "transcribe")):
                    continue
                spk = db.q("SELECT id, label FROM speakers WHERE video_id=? AND role='target'", (vid,))
                if not spk:
                    continue
                stats["videos"] += 1
                label_to_id = {r["label"]: r["id"] for r in spk}
                d = video_dir(settings, project_id, vid)
                dia = read_json(d / "diarization.json")
                tr = read_json(d / "transcript.json")
                words = [AWord(w["start"], w["end"], w["word"], w["prob"]) for s in tr["segments"] for w in s["words"]]
                words.sort(key=lambda w: w.start)
                turns = [tuple(t) for t in dia["turns"]]
                uncertain = [tuple(u) for u in dia["uncertain"]]
                assign_speakers(words, turns, uncertain)
                cands = build_candidates(words, set(label_to_id), cfg.min_segment_sec, cfg.max_segment_sec,
                                         cfg.split_pause_sec)
                x, sr = load_mono(d / "enhanced.wav")
                x16 = None
                # keep user-edited segments, rebuild the rest
                kept = db.q("SELECT start, end FROM segments WHERE video_id=? AND edited=1", (vid,))
                old = db.q("SELECT audio_path FROM segments WHERE video_id=? AND edited=0", (vid,))
                db.execute("DELETE FROM segments WHERE video_id=? AND edited=0", (vid,))
                for r in old:
                    Path(r["audio_path"]).unlink(missing_ok=True)
                cands = [c for c in cands if not any(min(c.end, k["end"]) - max(c.start, k["start"]) > 0 for k in kept)]
                if not cands:
                    continue
                # speaker similarity per segment (batched)
                sims: list[float | None] = [None] * len(cands)
                if target is not None:
                    if x16 is None:
                        x16, _ = load_mono(d / "enhanced.wav", 16000)
                    chunks = [x16[int(c.start * 16000): int(c.end * 16000)] for c in cands]
                    E = emb.embed(chunks)
                    sims = [cosine(e, target) for e in E]
                seg_dir = d / "segments"
                now = time.time()
                for j, c in enumerate(cands):
                    sub.check()
                    audio = x[int(c.start * sr): int(c.end * sr)]
                    text = normalize_ru(c.text)
                    extra = []
                    if any(ch.isdigit() for ch in c.text):
                        # num2words yields nominative forms ("от три до десять"); a human must fix the case.
                        extra.append("numbers auto-expanded - check grammatical case")
                    qa = q.assess(text, audio, sr, c.avg_prob, c.min_prob, sims[j], c.uncertain, cfg, extra)
                    sid = db.insert("segments", project_id=project_id, video_id=vid, speaker_id=label_to_id[c.speaker],
                                    start=c.start, end=c.end, audio_path="", text=text, text_raw=c.text,
                                    avg_word_prob=c.avg_prob, min_word_prob=c.min_prob, snr_db=qa.snr_db,
                                    rms_db=qa.rms_db, clipping_ratio=qa.clipping, speaker_sim=sims[j],
                                    overlap=int(c.uncertain), quality_score=qa.score, status=qa.status,
                                    reasons="; ".join(qa.reasons), created_at=now)
                    path = save_wav(seg_dir / f"{sid}.wav", audio, sr)
                    db.update("segments", sid, audio_path=str(path))
                    stats["created"] += 1
                    stats[qa.status] += 1
                    sub.progress((j + 1) / len(cands), f"Segments for video {vid}: {j + 1}/{len(cands)}")
        finally:
            emb.close()
    stats["duplicates"] = mark_duplicates(db, project_id)
    ctx.progress(1.0, f"{stats['created']} segments created")
    return stats


def _dedup_key(text: str) -> str:
    import re

    return re.sub(r"[^\wа-яё]+", " ", text.lower()).strip()


def mark_duplicates(db: Database, project_id: int) -> int:
    """Reject repeated transcripts (channel intros/outros), keeping the best-scoring copy."""
    rows = db.q("SELECT id, text, quality_score FROM segments WHERE project_id=? AND status<>'rejected' "
                "ORDER BY quality_score DESC", (project_id,))
    seen, n = set(), 0
    with db.tx() as c:
        for r in rows:
            k = _dedup_key(r["text"])
            if len(k) < 8:
                continue
            if k in seen:
                c.execute("UPDATE segments SET status='rejected', reasons=CASE WHEN reasons='' THEN 'duplicate' "
                          "ELSE reasons || '; duplicate' END WHERE id=?", (r["id"],))
                n += 1
            else:
                seen.add(k)
    return n


def recut_segment(db: Database, settings: Settings, segment_id: int, start: float, end: float) -> None:
    """Trim/extend a segment's boundaries and re-extract its audio from enhanced.wav."""
    r = db.q1("SELECT * FROM segments WHERE id=?", (segment_id,))
    if r is None:
        raise ValueError("segment not found")
    if end - start < 0.3:
        raise ValueError("Segment must be at least 0.3 s long.")
    d = video_dir(settings, r["project_id"], r["video_id"])
    x, sr = load_mono(d / "enhanced.wav")
    start = max(0.0, start)
    end = min(len(x) / sr, end)
    save_wav(Path(r["audio_path"]), x[int(start * sr): int(end * sr)], sr)
    db.update("segments", segment_id, start=start, end=end, edited=1)
