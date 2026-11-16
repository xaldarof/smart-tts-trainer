"""YouTube import: discover → review → authorize → download (with progress, cancel, retry)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QSpinBox)

from app.gui.context import AppContext, Page
from app.gui.widgets import (ERR, MUTED, OK, WARN, button, card, error_box, fill_table, fmt_bytes, fmt_dur,
                             make_table, page_header, selected_ids)
from app.services.youtube import downloader as yd
from app.services.youtube.urls import InvalidURL, parse_youtube_url

STATUS_COLORS = {"downloaded": OK, "failed": ERR, "downloading": "#5b8cff", "cancelled": MUTED, "queued": WARN}


class YouTubePage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header(
            "YouTube Import",
            "Paste a video, playlist, or channel URL. Videos are listed first; nothing is downloaded until you "
            "select videos and confirm you are authorized to use their audio."))
        f, lay = card("Source")
        row = QHBoxLayout()
        self.url = QLineEdit()
        self.url.setPlaceholderText("https://www.youtube.com/@channel  ·  playlist?list=…  ·  watch?v=…")
        self.url.returnPressed.connect(self.discover)
        self.limit = QSpinBox()
        self.limit.setRange(0, 5000)
        self.limit.setValue(200)
        self.limit.setPrefix("max ")
        self.limit.setSpecialValueText("no limit")
        self.limit.setToolTip("Maximum number of videos to list from a channel/playlist (0 = all)")
        self.discover_btn = button("Discover", "Primary", self.discover)
        row.addWidget(self.url, 1)
        row.addWidget(self.limit)
        row.addWidget(self.discover_btn)
        row.addWidget(button("Import local file…", slot=self.import_local,
                             tooltip="Add an audio/video file you own (bypasses YouTube)"))
        lay.addLayout(row)
        self.discover_status = QLabel("")
        self.discover_status.setObjectName("Subtitle")
        lay.addWidget(self.discover_status)
        self.body.addWidget(f)

        f2, lay2 = card("Videos in project")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay2.addWidget(self.summary)
        self.table = make_table(["ID", "Title", "Uploader", "Duration", "Est. size", "Status", "Attempts", "Error"], 1)
        self.table.setMinimumHeight(360)
        self.table.itemSelectionChanged.connect(self._update_summary)
        lay2.addWidget(self.table)
        self.auth = QCheckBox("I confirm I own these videos or have the speaker's and rights holder's permission to "
                              "process their audio for voice training.")
        lay2.addWidget(self.auth)
        brow = QHBoxLayout()
        brow.addWidget(button("Select all", slot=self.table.selectAll))
        self.dl_btn = button("Download selected", "Primary", self.download_selected)
        brow.addWidget(self.dl_btn)
        brow.addWidget(button("Retry failed", slot=self.retry_failed))
        brow.addWidget(button("Cancel downloads", "Danger", self.cancel))
        brow.addWidget(button("Remove selected", slot=self.remove_selected))
        brow.addStretch(1)
        lay2.addLayout(brow)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Subtitle")
        lay2.addWidget(self.progress)
        lay2.addWidget(self.progress_label)
        self.body.addWidget(f2)
        self._dl_jobs: dict[int, int] = {}       # job_id -> video_id
        self._job_progress: dict[int, float] = {}
        ctx.tasks.progress.connect(self._on_progress)

    # ----------------------------------------------------------- discovery
    def discover(self):
        if self.pid is None:
            return
        try:
            parse_youtube_url(self.url.text())
        except InvalidURL as e:
            error_box(self, "Invalid URL", str(e))
            return
        url, limit, pid = self.url.text(), self.limit.value() or None, self.pid
        self.discover_btn.setEnabled(False)
        self.discover_status.setText("Querying YouTube…")

        def work(ctx):
            parsed, title, vids = yd.discover(url, ctx, limit)
            src, new = yd.save_discovery(self.db, pid, parsed, title, vids)
            return parsed.kind, title, len(vids), new

        def done(status, res):
            self.discover_btn.setEnabled(True)
            if status == "done":
                kind, title, n, new = res
                self.discover_status.setText(f"{kind.title()} '{title}': {n} videos found, {new} new. Review the "
                                             "list, select what you are authorized to use, then download.")
                self.ctx.notify("videos")
                self.refresh()
            else:
                self.discover_status.setText(f"Discovery failed: {res}")
                if status == "failed":
                    error_box(self, "Discovery failed", str(res))

        self.ctx.tasks.submit("discover", work, project_id=pid, pool="io", on_done=done)

    def import_local(self):
        if self.pid is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Import media you own", "",
                                                "Media (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.mp4 *.mkv *.webm *.mov)")
        for f in files:
            yd.import_local_file(self.db, self.pid, Path(f))
        if files:
            self.ctx.notify("videos")
            self.refresh()

    # ------------------------------------------------------------ download
    def download_selected(self):
        ids = selected_ids(self.table)
        if not ids:
            error_box(self, "Nothing selected", "Select one or more videos in the table.")
            return
        if not self.auth.isChecked():
            error_box(self, "Authorization required", "Tick the authorization checkbox to confirm you may use the "
                                                       "selected videos' audio.")
            return
        rows = self.db.q(f"SELECT * FROM videos WHERE id IN ({','.join('?' * len(ids))})", ids)
        todo = [r for r in rows if r["download_status"] not in ("downloaded", "downloading")]
        if not todo:
            return
        total_dur = sum(r["duration"] or 0 for r in todo)
        total_size = sum(r["filesize_estimate"] or 0 for r in todo)
        if len(todo) > 5:
            ok = QMessageBox.question(
                self, "Confirm download",
                f"Download audio for {len(todo)} videos?\n\nTotal duration: {fmt_dur(total_dur)}\n"
                f"Estimated download: {fmt_bytes(total_size)}\nEstimated processing on RTX 4060 Ti: "
                f"~{total_dur / 3600 * 12:.0f}–{total_dur / 3600 * 25:.0f} min")
            if ok != QMessageBox.Yes:
                return
        out_dir = self.ctx.settings.sub("downloads") / str(self.pid)
        retries = self.ctx.settings.download_retries
        for r in todo:
            self.db.update("videos", r["id"], authorized=1, selected=1, download_status="queued")
            vid = r["id"]
            jid = self.ctx.tasks.submit("download", lambda c, v=vid: str(yd.download_audio(self.db, v, out_dir, c, retries)),
                                        project_id=self.pid, target_id=vid, pool="io",
                                        on_done=lambda st, res, v=vid: self._dl_done(v, st, res))
            self._dl_jobs[jid] = vid
        self.refresh()

    def _dl_done(self, vid, status, res):
        self._dl_jobs = {j: v for j, v in self._dl_jobs.items() if v != vid}
        self.ctx.notify("videos")
        self.refresh()
        if not self._dl_jobs:
            self.progress.setValue(1000)
            self.progress_label.setText("All downloads finished")
            self._job_progress.clear()

    def _on_progress(self, job_id, frac, msg):
        if job_id not in self._dl_jobs:
            return
        self._job_progress[job_id] = frac
        avg = sum(self._job_progress.values()) / max(1, len(self._dl_jobs))
        self.progress.setValue(int(avg * 1000))
        title = self.db.scalar("SELECT title FROM videos WHERE id=?", (self._dl_jobs[job_id],)) or ""
        self.progress_label.setText(f"{len(self._dl_jobs)} active · {title[:50]}: {msg}")

    def retry_failed(self):
        if self.pid is None:
            return
        ids = [r["id"] for r in self.db.q("SELECT id FROM videos WHERE project_id=? AND download_status IN "
                                          "('failed','cancelled') AND authorized=1", (self.pid,))]
        if not ids:
            return
        self.table.clearSelection()
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) in ids:
                self.table.selectRow(r)
        self.auth.setChecked(True)
        self.download_selected()

    def cancel(self):
        for jid in list(self._dl_jobs):
            self.ctx.tasks.cancel(jid)
        self.db.execute("UPDATE videos SET download_status='cancelled' WHERE download_status='queued' AND project_id=?",
                        (self.pid,))
        self.refresh()

    def remove_selected(self):
        ids = selected_ids(self.table)
        if not ids or QMessageBox.question(self, "Remove", f"Remove {len(ids)} videos and all derived data from this "
                                           "project? Downloaded files are kept on disk.") != QMessageBox.Yes:
            return
        self.db.execute(f"DELETE FROM videos WHERE id IN ({','.join('?' * len(ids))})", ids)
        self.ctx.notify("videos")
        self.refresh()

    # ------------------------------------------------------------- display
    def refresh(self):
        if self.pid is None:
            return
        rows = self.db.q("SELECT * FROM videos WHERE project_id=? ORDER BY id", (self.pid,))
        fill_table(self.table, [(r["id"], r["title"], r["uploader"] or "", fmt_dur(r["duration"]),
                                 fmt_bytes(r["filesize_estimate"]), r["download_status"], r["attempts"],
                                 r["error"] or "") for r in rows],
                   [r["id"] for r in rows], [STATUS_COLORS.get(r["download_status"]) for r in rows])
        self._update_summary()

    def _update_summary(self):
        if self.pid is None:
            return
        allv = self.db.q("SELECT duration, filesize_estimate, download_status FROM videos WHERE project_id=?", (self.pid,))
        ids = selected_ids(self.table)
        sel = self.db.q(f"SELECT duration, filesize_estimate FROM videos WHERE id IN ({','.join('?' * len(ids))})",
                        ids) if ids else []
        dur = sum(r["duration"] or 0 for r in sel)
        self.summary.setText(
            f"{len(allv)} videos · {sum(1 for r in allv if r['download_status'] == 'downloaded')} downloaded · "
            f"total {fmt_dur(sum(r['duration'] or 0 for r in allv))}.   Selected: {len(sel)} videos, {fmt_dur(dur)}, "
            f"~{fmt_bytes(sum(r['filesize_estimate'] or 0 for r in sel))} audio"
            + (f", est. processing {dur / 3600 * 12:.0f}–{dur / 3600 * 25:.0f} min on GPU" if dur else ""))
