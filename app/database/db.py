"""SQLite persistence layer.

Design notes
------------
* One connection per thread (sqlite3 objects are not shareable across threads);
  ``Database.conn`` returns the calling thread's connection.
* WAL mode so GUI reads don't block worker writes.
* State transitions use ``UPDATE ... WHERE status IN (...)`` and check
  ``rowcount`` so concurrent workers cannot both claim the same item.
* On startup ``recover_interrupted()`` moves rows left in a "running" state by a
  crash back to a resumable state. Completed pipeline stages are recorded in
  ``video_stages`` and are never re-run.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    target_speaker_name TEXT,
    consent_confirmed_at REAL,
    consent_statement TEXT,
    target_embedding BLOB
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    added_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    youtube_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    uploader TEXT,
    duration REAL,
    upload_date TEXT,
    filesize_estimate INTEGER,
    availability TEXT,
    selected INTEGER NOT NULL DEFAULT 0,
    authorized INTEGER NOT NULL DEFAULT 0,
    download_status TEXT NOT NULL DEFAULT 'discovered',
    download_path TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at REAL,
    UNIQUE(project_id, youtube_id)
);

CREATE TABLE IF NOT EXISTS video_stages (
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    output_path TEXT,
    meta_json TEXT,
    finished_at REAL,
    error TEXT,
    PRIMARY KEY (video_id, stage)
);

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    total_duration REAL NOT NULL DEFAULT 0,
    n_turns INTEGER NOT NULL DEFAULT 0,
    preview_path TEXT,
    embedding BLOB,
    role TEXT NOT NULL DEFAULT 'unassigned',
    similarity REAL,
    UNIQUE(video_id, label)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    audio_path TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    text_raw TEXT NOT NULL DEFAULT '',
    avg_word_prob REAL,
    min_word_prob REAL,
    snr_db REAL,
    rms_db REAL,
    clipping_ratio REAL,
    speaker_sim REAL,
    overlap INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    status TEXT NOT NULL DEFAULT 'review',
    reasons TEXT NOT NULL DEFAULT '',
    edited INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id, status);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    kind TEXT NOT NULL,
    target_id INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    n_train INTEGER NOT NULL,
    n_eval INTEGER NOT NULL,
    total_duration REAL NOT NULL,
    created_at REAL NOT NULL,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    backend TEXT NOT NULL,
    preset TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    output_dir TEXT NOT NULL,
    exp_dir TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    current_step INTEGER NOT NULL DEFAULT 0,
    current_epoch INTEGER NOT NULL DEFAULT 0,
    last_loss REAL,
    best_eval_loss REAL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    epoch INTEGER,
    split TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, split, name);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    step INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    run_id INTEGER REFERENCES training_runs(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    backend TEXT NOT NULL,
    kind TEXT NOT NULL,              -- finetuned | pretrained
    path TEXT NOT NULL,              -- directory with model files
    language TEXT NOT NULL DEFAULT 'ru',
    created_at REAL NOT NULL,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    mode TEXT NOT NULL,               -- trained | pretrained | clone
    text TEXT NOT NULL,
    output_path TEXT NOT NULL,
    duration REAL,
    created_at REAL NOT NULL,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY,
    context TEXT NOT NULL,
    message TEXT NOT NULL,
    detail TEXT,
    created_at REAL NOT NULL
);
"""


class StateConflict(RuntimeError):
    """Raised when an atomic state transition does not find the expected state."""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True) if self.path != ":memory:" else None
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self.conn.executescript(SCHEMA)  # executescript manages its own transaction
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))

    # ----------------------------------------------------------- connection
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            if self.path != ":memory:":
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self):
        c = self.conn
        if c.in_transaction:  # nested: join the outer transaction
            yield c
            return
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    def q1(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        r = self.q1(sql, params)
        return None if r is None else r[0]

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.tx() as c:
            return c.execute(sql, tuple(params))

    def insert(self, table: str, **values) -> int:
        cols = ", ".join(values)
        ph = ", ".join("?" for _ in values)
        with self.tx() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({ph})", tuple(values.values()))
            return int(cur.lastrowid)

    def update(self, table: str, row_id: int, **values) -> None:
        sets = ", ".join(f"{k}=?" for k in values)
        with self.tx() as c:
            c.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*values.values(), row_id))

    def transition(self, table: str, row_id: int, from_states: Iterable[str], to_state: str,
                   column: str = "status", **extra) -> None:
        """Atomically move a row from one of ``from_states`` to ``to_state``."""
        from_states = list(from_states)
        sets = ", ".join([f"{column}=?"] + [f"{k}=?" for k in extra])
        ph = ", ".join("?" for _ in from_states)
        with self.tx() as c:
            cur = c.execute(
                f"UPDATE {table} SET {sets} WHERE id=? AND {column} IN ({ph})",
                (to_state, *extra.values(), row_id, *from_states),
            )
            if cur.rowcount != 1:
                row = c.execute(f"SELECT {column} FROM {table} WHERE id=?", (row_id,)).fetchone()
                raise StateConflict(
                    f"{table}#{row_id}: expected {column} in {from_states}, found {row[0] if row else 'missing row'}"
                )

    # ------------------------------------------------------------- helpers
    def log_error(self, context: str, message: str, detail: str | None = None) -> None:
        try:
            self.insert("errors", context=context, message=message, detail=detail, created_at=time.time())
        except Exception:
            pass

    def stage_done(self, video_id: int, stage: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM video_stages WHERE video_id=? AND stage=? AND status='done'", (video_id, stage))

    def mark_stage(self, video_id: int, stage: str, status: str, output_path: str | None = None,
                   meta: dict | None = None, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO video_stages(video_id, stage, status, output_path, meta_json, finished_at, error) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(video_id, stage) DO UPDATE SET status=excluded.status, "
                "output_path=excluded.output_path, meta_json=excluded.meta_json, finished_at=excluded.finished_at, "
                "error=excluded.error",
                (video_id, stage, status, output_path, json.dumps(meta or {}, ensure_ascii=False),
                 time.time() if status == "done" else None, error),
            )

    def clear_stages(self, video_id: int, stages: Iterable[str]) -> None:
        stages = list(stages)
        ph = ", ".join("?" for _ in stages)
        self.execute(f"DELETE FROM video_stages WHERE video_id=? AND stage IN ({ph})", (video_id, *stages))

    def recover_interrupted(self) -> dict[str, int]:
        """Reset rows that were mid-flight when the process died."""
        out = {}
        with self.tx() as c:
            out["jobs"] = c.execute(
                "UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('running','pending')",
                (time.time(),),
            ).rowcount
            out["downloads"] = c.execute(
                "UPDATE videos SET download_status='queued' WHERE download_status='downloading'"
            ).rowcount
            out["stages"] = c.execute("DELETE FROM video_stages WHERE status='running'").rowcount
            out["training"] = c.execute(
                "UPDATE training_runs SET status='interrupted' WHERE status IN ('running','pending','stopping')"
            ).rowcount
        return out

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


# ---------------------------------------------------------------- projects
def get_or_create_project(db: Database, name: str) -> int:
    row = db.q1("SELECT id FROM projects WHERE name=?", (name,))
    if row:
        return int(row["id"])
    return db.insert("projects", name=name, created_at=time.time())


def confirm_consent(db: Database, project_id: int, speaker_name: str, statement: str) -> None:
    if not speaker_name.strip() or not statement.strip():
        raise ValueError("Speaker name and consent statement are required.")
    db.update("projects", project_id, target_speaker_name=speaker_name.strip(),
              consent_statement=statement.strip(), consent_confirmed_at=time.time())


def has_consent(db: Database, project_id: int) -> bool:
    return bool(db.scalar("SELECT consent_confirmed_at FROM projects WHERE id=?", (project_id,)))
