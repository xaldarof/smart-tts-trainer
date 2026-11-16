import threading
import time

import pytest

from app.database.db import Database, StateConflict, confirm_consent, has_consent
from app.utils.jobs import Cancelled, JobContext


def test_schema_and_reopen(settings):
    db = Database(settings.db_path)
    pid = db.insert("projects", name="p", created_at=time.time())
    db.close()
    db2 = Database(settings.db_path)        # survives "restart"
    assert db2.scalar("SELECT name FROM projects WHERE id=?", (pid,)) == "p"
    assert db2.scalar("SELECT value FROM meta WHERE key='schema_version'") == "1"


def test_atomic_transition(db, project):
    vid = db.insert("videos", project_id=project, youtube_id="a" * 11, url="u")
    db.transition("videos", vid, ["discovered"], "downloading", column="download_status")
    with pytest.raises(StateConflict):
        db.transition("videos", vid, ["discovered"], "downloading", column="download_status")


def test_transition_is_race_free(settings, project):
    db = Database(settings.db_path)
    vid = db.insert("videos", project_id=project, youtube_id="b" * 11, url="u")
    wins = []

    def claim():
        d = Database(settings.db_path)
        try:
            d.transition("videos", vid, ["discovered"], "downloading", column="download_status")
            wins.append(1)
        except StateConflict:
            pass
        d.close()

    ts = [threading.Thread(target=claim) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(wins) == 1


def test_tx_rollback(db, project):
    with pytest.raises(RuntimeError):
        with db.tx() as c:
            c.execute("INSERT INTO sources(project_id, url, kind, added_at) VALUES(?,?,?,?)", (project, "u", "video", 0))
            raise RuntimeError("boom")
    assert db.scalar("SELECT COUNT(*) FROM sources") == 0


def test_recover_interrupted(db, project):
    vid = db.insert("videos", project_id=project, youtube_id="c" * 11, url="u", download_status="downloading")
    db.insert("jobs", kind="process", status="running", created_at=0)
    db.mark_stage(vid, "extract", "done", "x")
    db.mark_stage(vid, "enhance", "running")
    ds = db.insert("datasets", project_id=project, name="d", path="p", language="ru", n_train=1, n_eval=1,
                   total_duration=1, created_at=0)
    db.insert("training_runs", project_id=project, dataset_id=ds, backend="xtts_v2", preset="b", config_json="{}",
              status="running", output_dir="o", created_at=0)
    rec = db.recover_interrupted()
    assert rec == {"jobs": 1, "downloads": 1, "stages": 1, "training": 1}
    assert db.stage_done(vid, "extract") is not None     # completed stage kept
    assert db.q1("SELECT * FROM video_stages WHERE stage='enhance'") is None
    assert db.scalar("SELECT status FROM training_runs") == "interrupted"


def test_consent(db, project):
    assert not has_consent(db, project)
    with pytest.raises(ValueError):
        confirm_consent(db, project, "Name", "")
    confirm_consent(db, project, "Name", "I am the speaker")
    assert has_consent(db, project)


def test_job_context_cancel_and_progress():
    seen = []
    ctx = JobContext(lambda f, m: seen.append(round(f, 3)))
    sub = ctx.sub(0.5, 1.0)
    sub.progress(0.5, "half")
    assert seen == [0.75]
    ctx.cancel()
    with pytest.raises(Cancelled):
        sub.check()


def test_pipeline_skips_completed_stages(db, project, settings, monkeypatch):
    """Resume behaviour: a stage marked done is never executed again."""
    from app.services.audio import pipeline as pl

    vid = db.insert("videos", project_id=project, youtube_id="d" * 11, url="u", download_status="downloaded")
    db.mark_stage(vid, "extract", "done", "x", {"duration": 1})
    calls = []
    p = pl.Pipeline.__new__(pl.Pipeline)
    p.db, p.settings = db, settings

    def fn(v, d, c):
        calls.append(v)
        return "out", {}

    errs = p._run_stage("extract", [vid], JobContext(), fn)
    assert errs == {} and calls == []
    errs = p._run_stage("enhance", [vid], JobContext(), fn)
    assert calls == [vid] and db.stage_done(vid, "enhance")


def test_pipeline_stage_failure_is_recorded(db, project, settings):
    from app.services.audio import pipeline as pl

    vid = db.insert("videos", project_id=project, youtube_id="e" * 11, url="u", download_status="downloaded")
    p = pl.Pipeline.__new__(pl.Pipeline)
    p.db, p.settings = db, settings

    def bad(v, d, c):
        raise ValueError("corrupt audio")

    errs = p._run_stage("extract", [vid], JobContext(), bad)
    assert "corrupt audio" in errs[vid]
    assert db.q1("SELECT status FROM video_stages WHERE video_id=?", (vid,))["status"] == "failed"
    assert db.scalar("SELECT COUNT(*) FROM errors") == 1


def test_pipeline_cancellation(db, project, settings):
    from app.services.audio import pipeline as pl

    vids = [db.insert("videos", project_id=project, youtube_id=f"{i}" * 11, url="u", download_status="downloaded")
            for i in range(3)]
    p = pl.Pipeline.__new__(pl.Pipeline)
    p.db, p.settings = db, settings
    ctx = JobContext()

    def fn(v, d, c):
        ctx.cancel()
        c.check()

    with pytest.raises(Cancelled):
        p._run_stage("extract", vids, ctx, fn)
    assert db.q1("SELECT status FROM video_stages WHERE video_id=?", (vids[0],))["status"] == "cancelled"
    assert db.stage_done(vids[1], "extract") is None
