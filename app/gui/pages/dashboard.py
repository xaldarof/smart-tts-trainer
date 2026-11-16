"""Dashboard: real statistics from SQLite and the filesystem."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from PySide6.QtWidgets import QGridLayout, QHBoxLayout

from app.gui.context import AppContext, Page
from app.gui.widgets import (ERR, MUTED, OK, WARN, StatCard, card, fill_table, fmt_bytes, fmt_dur, make_table,
                             page_header)


def dir_size(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class DashboardPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header("Dashboard", "Live statistics for the current project and workspace."))
        grid = QGridLayout()
        grid.setSpacing(10)
        self.cards = {}
        names = [("videos", "Videos discovered"), ("downloaded", "Downloaded"), ("processed", "Fully processed"),
                 ("speech", "Accepted speech"), ("segments", "Segments (acc / review / rej)"),
                 ("datasets", "Datasets"), ("runs", "Training runs"), ("models", "Trained models"),
                 ("gens", "Generations"), ("disk", "Data directory size"), ("free", "Free disk"), ("gpu", "GPU")]
        for i, (k, label) in enumerate(names):
            c = StatCard(label)
            self.cards[k] = c
            grid.addWidget(c, i // 4, i % 4)
        self.body.addLayout(grid)

        row = QHBoxLayout()
        f1, l1 = card("Recent jobs")
        self.jobs = make_table(["ID", "Kind", "Status", "Progress", "Message", "Started"], 4)
        self.jobs.setMinimumHeight(240)
        l1.addWidget(self.jobs)
        f2, l2 = card("Recent errors")
        self.errors = make_table(["When", "Context", "Message"], 2)
        self.errors.setMinimumHeight(240)
        l2.addWidget(self.errors)
        row.addWidget(f1, 3)
        row.addWidget(f2, 2)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self._disk_cache: tuple[float, int] = (0.0, 0)

    def on_data_changed(self, topic):
        if self.isVisible():
            self.refresh()

    def refresh(self):
        db, pid = self.db, self.pid or -1
        q = lambda sql, *a: db.scalar(sql, a) or 0  # noqa: E731
        nv = q("SELECT COUNT(*) FROM videos WHERE project_id=?", pid)
        dur = q("SELECT SUM(duration) FROM videos WHERE project_id=?", pid)
        self.cards["videos"].set(str(nv), f"Videos discovered · {fmt_dur(dur)} total")
        nd = q("SELECT COUNT(*) FROM videos WHERE project_id=? AND download_status='downloaded'", pid)
        nf = q("SELECT COUNT(*) FROM videos WHERE project_id=? AND download_status='failed'", pid)
        self.cards["downloaded"].set(str(nd), f"Downloaded · {nf} failed")
        npr = q("SELECT COUNT(*) FROM video_stages s JOIN videos v ON v.id=s.video_id WHERE v.project_id=? "
                "AND s.stage='transcribe' AND s.status='done'", pid)
        self.cards["processed"].set(str(npr))
        acc = q("SELECT SUM(end-start) FROM segments WHERE project_id=? AND status='accepted'", pid)
        self.cards["speech"].set(fmt_dur(acc), "Accepted clean speech (h:m:s)")
        a = q("SELECT COUNT(*) FROM segments WHERE project_id=? AND status='accepted'", pid)
        r = q("SELECT COUNT(*) FROM segments WHERE project_id=? AND status='review'", pid)
        j = q("SELECT COUNT(*) FROM segments WHERE project_id=? AND status='rejected'", pid)
        self.cards["segments"].set(f"{a} / {r} / {j}")
        self.cards["datasets"].set(str(q("SELECT COUNT(*) FROM datasets WHERE project_id=?", pid)))
        running = q("SELECT COUNT(*) FROM training_runs WHERE project_id=? AND status='running'", pid)
        self.cards["runs"].set(str(q("SELECT COUNT(*) FROM training_runs WHERE project_id=?", pid)),
                               f"Training runs · {running} running")
        self.cards["models"].set(str(q("SELECT COUNT(*) FROM models WHERE project_id=? AND kind='finetuned'", pid)))
        self.cards["gens"].set(str(q("SELECT COUNT(*) FROM generations")))
        data = self.ctx.settings.data_path
        now = time.time()
        if now - self._disk_cache[0] > 30:
            self._disk_cache = (now, dir_size(data))
        self.cards["disk"].set(fmt_bytes(self._disk_cache[1]))
        self.cards["free"].set(fmt_bytes(shutil.disk_usage(data).free), f"Free disk at {data}")
        from app.utils.gpu import gpu_info

        g = gpu_info()
        if g.get("available"):
            self.cards["gpu"].set(f"{g['used_gb']:.1f}/{g['total_gb']:.0f} GB", g["name"])
        else:
            self.cards["gpu"].set("CPU only", "No CUDA GPU detected")

        rows = db.q("SELECT id, kind, status, progress, message, started_at FROM jobs ORDER BY id DESC LIMIT 30")
        colors = {"done": OK, "failed": ERR, "cancelled": MUTED, "interrupted": WARN, "running": None}
        fill_table(self.jobs, [(r["id"], r["kind"], r["status"], f"{(r['progress'] or 0) * 100:.0f}%", r["message"] or "",
                                time.strftime("%m-%d %H:%M", time.localtime(r["started_at"])) if r["started_at"] else "")
                               for r in rows], [r["id"] for r in rows], [colors.get(r["status"]) for r in rows])
        er = db.q("SELECT created_at, context, message FROM errors ORDER BY id DESC LIMIT 30")
        fill_table(self.errors, [(time.strftime("%m-%d %H:%M", time.localtime(e["created_at"])), e["context"],
                                  e["message"]) for e in er])
