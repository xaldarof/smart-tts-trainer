"""Headless end-to-end run of the real pipeline (no GUI, no mocks).

media file -> extract -> enhance -> VAD -> diarize -> transcribe -> segments ->
dataset export/validation -> XTTS fine-tuning (N steps) -> checkpoint export ->
trained-model inference -> WAV + MP3 export.

Example (uses a synthetic recording you own; see make_test_audio.py):

    python scripts/e2e_pipeline.py data/test_media/two_speakers.wav --whisper small --train-steps 2 \
        --target-label SPEAKER_00 --consent "Test recording generated with macOS say"

On the RTX target, run with ``--train-steps 50`` (and default whisper large-v3) as a smoke test.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import Settings  # noqa: E402
from app.database.db import Database, confirm_consent, get_or_create_project  # noqa: E402
from app.services.audio.pipeline import Pipeline, build_segments, set_speaker_role  # noqa: E402
from app.services.dataset.builder import export_dataset, project_stats, validate_dataset  # noqa: E402
from app.services.youtube.downloader import import_local_file  # noqa: E402
from app.utils.jobs import JobContext  # noqa: E402
from app.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--project", default="e2e-test")
    ap.add_argument("--whisper", default=None)
    ap.add_argument("--diarization", default="ecapa")
    ap.add_argument("--target-label", default=None, help="speaker label to mark as target (else print and stop)")
    ap.add_argument("--consent", default=None)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--preset", default="low_vram")
    ap.add_argument("--accept-review", action="store_true", help="accept 'review' segments (test only)")
    ap.add_argument("--text", default="Привет! Это проверка синтеза речи дообученной моделью. Всё работает локально.")
    a = ap.parse_args()

    s = Settings.load()
    if a.whisper:
        s.processing.whisper_model = a.whisper
    s.processing.diarization_backend = a.diarization
    setup_logging(s.sub("logs"))
    db = Database(s.db_path)
    pid = get_or_create_project(db, a.project)
    vid = import_local_file(db, pid, Path(a.media))

    def prog(f, m):
        print(f"\r[{f * 100:5.1f}%] {m[:90]:<90}", end="", flush=True)

    t0 = time.time()
    errs = Pipeline(db, s).run([vid], JobContext(prog))
    print(f"\nprocessing done in {time.time() - t0:.0f}s, errors={errs}")
    if errs:
        return 1
    for r in db.q("SELECT id, label, total_duration, n_turns, role, similarity FROM speakers WHERE video_id=?", (vid,)):
        print(dict(r))
    if not a.target_label:
        print("Pass --target-label to continue.")
        return 0
    sp = db.q1("SELECT id FROM speakers WHERE video_id=? AND label=?", (vid, a.target_label))
    set_speaker_role(db, sp["id"], "target")
    confirm_consent(db, pid, "Test speaker", a.consent or "")
    st = build_segments(db, s, pid, [vid], JobContext(prog))
    print("\nsegments:", st)
    if a.accept_review:
        db.execute("UPDATE segments SET status='accepted' WHERE project_id=? AND status='review'", (pid,))
    for r in db.q("SELECT id, round(start,2) s, round(end,2) e, status, reasons, text FROM segments "
                  "WHERE project_id=? ORDER BY start LIMIT 8", (pid,)):
        print(dict(r))
    stats = project_stats(db, pid)
    print(stats.readiness, f"accepted={stats.n_accepted} review={stats.n_review} rejected={stats.n_rejected}")
    if not stats.ready:
        return 1
    ds_id = export_dataset(db, s, pid, "e2e", JobContext(prog))
    ds = db.q1("SELECT * FROM datasets WHERE id=?", (ds_id,))
    rep = validate_dataset(Path(ds["path"]))
    print("\nvalidation:", json.dumps(rep.as_dict(), ensure_ascii=False)[:500])
    if not rep.ok or not a.train_steps:
        return 0 if rep.ok else 1

    from app.services.training import base_models, manager
    from app.services.training.presets import preset

    s.cpml_accepted = True  # the user running this script accepts CPML explicitly by passing --train-steps
    base_models.ensure_xtts(s, JobContext(prog))
    cfg = preset(a.preset)
    cfg.max_steps, cfg.epochs, cfg.num_workers, cfg.save_step = a.train_steps, 1000, 0, 10**6
    if a.train_steps <= 4:
        cfg.batch_size, cfg.grad_accum = 1, 1
    run_id = manager.create_run(db, s, pid, ds_id, cfg)

    def on_ev(ev):
        if ev["event"] in ("train_step", "eval", "started", "checkpoint", "stopping", "data", "init", "info", "failed"):
            print("EVENT", json.dumps(ev, ensure_ascii=False)[:300])

    tp = manager.TrainingProcess(db, s, run_id, on_ev)
    status = tp.execute()
    run = db.q1("SELECT * FROM training_runs WHERE id=?", (run_id,))
    print("training status:", status, "step", run["current_step"], "loss", run["last_loss"], "err", run["error"])
    if status == "failed":
        print("\n".join(list(tp.tail)[-30:]))
        return 1
    ck = db.q("SELECT path, step FROM checkpoints WHERE run_id=?", (run_id,))
    print("checkpoints:", [dict(c) for c in ck])
    mid = manager.export_model(db, s, run_id, "e2e-voice")
    print("exported model", mid)

    from app.services.inference.xtts_infer import XttsEngine, generate
    from app.utils.audio import export_audio

    eng = XttsEngine(s)
    gid = generate(db, s, eng, JobContext(prog), mode="trained", text=a.text, model_id=mid)
    g = db.q1("SELECT * FROM generations WHERE id=?", (gid,))
    mp3 = export_audio(Path(g["output_path"]), Path(g["output_path"]).with_suffix(".mp3"), "mp3")
    print(f"\ngenerated {g['output_path']} ({g['duration']:.1f}s) and {mp3}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
