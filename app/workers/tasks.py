"""Qt background task infrastructure.

* ``TaskManager.submit`` runs a service function ``fn(ctx)`` on a thread pool, records
  it in the ``jobs`` table, and relays progress / completion through Qt signals
  (signals are delivered on the GUI thread, so the UI never blocks).
* Two pools: ``io`` (downloads, parallel) and ``gpu`` (one thread: GPU-heavy stages run
  strictly sequentially).
* ``TrainingController`` owns training child processes and relays their events.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.config.settings import Settings
from app.database.db import Database
from app.utils.jobs import Cancelled, JobContext

log = logging.getLogger(__name__)


class _Signals(QObject):
    progress = Signal(int, float, str)            # job_id, fraction, message
    finished = Signal(int, str, object)           # job_id, status, result-or-error-message


class _Runnable(QRunnable):
    def __init__(self, job_id: int, fn: Callable[[JobContext], Any], ctx: JobContext, sig: _Signals, db: Database):
        super().__init__()
        self.job_id, self.fn, self.ctx, self.sig, self.db = job_id, fn, ctx, sig, db
        self.setAutoDelete(True)

    def run(self):
        db = self.db
        try:
            db.update("jobs", self.job_id, status="running", started_at=time.time())
            if self.ctx.cancelled:
                raise Cancelled()
            result = self.fn(self.ctx)
            db.update("jobs", self.job_id, status="done", progress=1.0, finished_at=time.time())
            self.sig.finished.emit(self.job_id, "done", result)
        except Cancelled:
            db.update("jobs", self.job_id, status="cancelled", finished_at=time.time())
            self.sig.finished.emit(self.job_id, "cancelled", "Cancelled")
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Job %s failed: %s\n%s", self.job_id, e, tb)
            db.update("jobs", self.job_id, status="failed", error=str(e), finished_at=time.time())
            db.log_error(f"job:{self.job_id}", str(e), tb)
            self.sig.finished.emit(self.job_id, "failed", str(e))
        finally:
            db.close()  # thread-local connection of this pool thread


class TaskManager(QObject):
    progress = Signal(int, float, str)
    finished = Signal(int, str, object)
    changed = Signal()

    def __init__(self, db: Database, settings: Settings):
        super().__init__()
        self.db, self.settings = db, settings
        self.io_pool = QThreadPool()
        self.io_pool.setMaxThreadCount(max(1, settings.max_parallel_downloads))
        self.gpu_pool = QThreadPool()
        self.gpu_pool.setMaxThreadCount(1)
        self._ctx: dict[int, JobContext] = {}
        self._callbacks: dict[int, Callable[[str, Any], None]] = {}
        self._last_db_write: dict[int, float] = {}
        self._sig = _Signals()
        self._sig.progress.connect(self._on_progress)
        self._sig.finished.connect(self._on_finished)

    def submit(self, kind: str, fn: Callable[[JobContext], Any], *, project_id: int | None = None,
               target_id: int | None = None, pool: str = "gpu",
               on_done: Callable[[str, Any], None] | None = None) -> int:
        job_id = self.db.insert("jobs", project_id=project_id, kind=kind, target_id=target_id, status="pending",
                                created_at=time.time(), message="Queued")
        ctx = JobContext(lambda f, m, j=job_id: self._sig.progress.emit(j, f, m), threading.Event())
        self._ctx[job_id] = ctx
        if on_done:
            self._callbacks[job_id] = on_done
        (self.io_pool if pool == "io" else self.gpu_pool).start(_Runnable(job_id, fn, ctx, self._sig, self.db))
        self.changed.emit()
        return job_id

    def cancel(self, job_id: int) -> None:
        ctx = self._ctx.get(job_id)
        if ctx:
            ctx.cancel()

    def cancel_kind(self, kind_prefix: str) -> int:
        n = 0
        for jid, ctx in list(self._ctx.items()):
            row = self.db.q1("SELECT kind FROM jobs WHERE id=?", (jid,))
            if row and row["kind"].startswith(kind_prefix):
                ctx.cancel()
                n += 1
        return n

    def active(self) -> list[int]:
        return list(self._ctx)

    def _on_progress(self, job_id: int, frac: float, msg: str) -> None:
        now = time.time()
        if now - self._last_db_write.get(job_id, 0) > 0.5:
            self._last_db_write[job_id] = now
            try:
                self.db.update("jobs", job_id, progress=frac, message=msg[:300])
            except Exception:
                pass
        self.progress.emit(job_id, frac, msg)

    def _on_finished(self, job_id: int, status: str, result: Any) -> None:
        self._ctx.pop(job_id, None)
        self._last_db_write.pop(job_id, None)
        cb = self._callbacks.pop(job_id, None)
        if cb:
            try:
                cb(status, result)
            except Exception:
                log.exception("job callback failed")
        self.finished.emit(job_id, status, result)
        self.changed.emit()

    def shutdown(self) -> None:
        for ctx in self._ctx.values():
            ctx.cancel()
        self.io_pool.waitForDone(5000)
        self.gpu_pool.waitForDone(5000)


class _TrainSignals(QObject):
    event = Signal(int, dict)
    done = Signal(int, str)


class TrainingController(QObject):
    event = Signal(int, dict)       # run_id, event
    done = Signal(int, str)         # run_id, final status

    def __init__(self, db: Database, settings: Settings):
        super().__init__()
        self.db, self.settings = db, settings
        self._procs: dict[int, Any] = {}
        self._sig = _TrainSignals()
        self._sig.event.connect(self.event)
        self._sig.done.connect(self._finished)

    def is_running(self, run_id: int | None = None) -> bool:
        return bool(self._procs) if run_id is None else run_id in self._procs

    def start(self, run_id: int) -> None:
        from app.services.training.manager import TrainingProcess

        if self._procs:
            raise RuntimeError("A training run is already active. Only one run can use the GPU at a time.")
        tp = TrainingProcess(self.db, self.settings, run_id, lambda ev: self._sig.event.emit(run_id, ev))
        self._procs[run_id] = tp

        def body():
            try:
                status = tp.execute()
            except Exception as e:
                log.exception("training run %s crashed", run_id)
                self.db.update("training_runs", run_id, status="failed", error=str(e))
                self.db.log_error(f"training:{run_id}", str(e), traceback.format_exc())
                self._sig.event.emit(run_id, {"event": "failed", "message": str(e)})
                status = "failed"
            finally:
                self.db.close()
            self._sig.done.emit(run_id, status)

        threading.Thread(target=body, name=f"train-{run_id}", daemon=True).start()

    def stop(self, run_id: int) -> None:
        tp = self._procs.get(run_id)
        if tp:
            tp.request_stop()

    def kill_all(self) -> None:
        for tp in self._procs.values():
            tp.request_stop()
        deadline = time.time() + 60
        for tp in self._procs.values():
            while tp.proc and tp.proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
            tp.kill()

    def _finished(self, run_id: int, status: str) -> None:
        self._procs.pop(run_id, None)
        self.done.emit(run_id, status)
