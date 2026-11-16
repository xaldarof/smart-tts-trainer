"""GUI construction tests (offscreen). Every page builds and refreshes against a real SQLite DB."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def window(qtbot, settings, db):
    from app.gui.context import AppContext
    from app.gui.main_window import MainWindow
    from app.workers.tasks import TaskManager, TrainingController

    ctx = AppContext(db, settings, TaskManager(db, settings), TrainingController(db, settings))
    w = MainWindow(ctx)
    qtbot.addWidget(w)
    yield w
    ctx.tasks.shutdown()


def test_all_pages_render(window, qtbot):
    for i in range(window.stack.count()):
        window.stack.setCurrentIndex(i)
        qtbot.wait(20)
        window.stack.currentWidget().refresh()
    assert window.project.count() == 1  # default project auto-created


def test_task_manager_runs_and_cancels(window, qtbot):
    import time

    from app.utils.jobs import Cancelled

    tasks = window.ctx.tasks
    results = []
    with qtbot.waitSignal(tasks.finished, timeout=5000):
        tasks.submit("t", lambda c: 42, pool="io", on_done=lambda s, r: results.append((s, r)))
    assert results == [("done", 42)]

    def slow(c):
        for _ in range(200):
            c.check()
            time.sleep(0.01)
        return "never"

    jid = tasks.submit("slow", slow, pool="io", on_done=lambda s, r: results.append((s, r)))
    tasks.cancel(jid)
    qtbot.waitUntil(lambda: len(results) == 2, timeout=5000)
    assert results[1][0] == "cancelled"
    assert window.ctx.db.scalar("SELECT status FROM jobs WHERE id=?", (jid,)) == "cancelled"


def test_failed_job_records_error(window, qtbot):
    tasks = window.ctx.tasks

    def boom(c):
        raise RuntimeError("disk full")

    with qtbot.waitSignal(tasks.finished, timeout=5000) as sig:
        jid = tasks.submit("boom", boom, pool="io")
    assert sig.args[1] == "failed" and "disk full" in sig.args[2]
    assert window.ctx.db.scalar("SELECT error FROM jobs WHERE id=?", (jid,)) == "disk full"
    assert window.ctx.db.scalar("SELECT COUNT(*) FROM errors") >= 1
