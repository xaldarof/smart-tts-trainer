"""Audio processing + speaker identification + consent + segment building."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QProgressBar, QSplitter, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from app.database.db import confirm_consent, has_consent
from app.gui.context import AppContext, Page
from app.gui.widgets import (ERR, MUTED, OK, WARN, AudioPlayer, button, card, error_box, fill_table, fmt_dur,
                             info_box, make_table, page_header, selected_ids)
from app.services.audio.pipeline import STAGES, Pipeline, build_segments, refresh_target_embedding, set_speaker_role

ROLE_COLORS = {"target": OK, "excluded": ERR}


class ConsentDialog(QDialog):
    def __init__(self, parent, current_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Speaker consent")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        msg = QLabel("Training a voice model requires the speaker's explicit permission. Record who the speaker is "
                     "and how consent was obtained. This is stored with the project and in every exported dataset.")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        form = QFormLayout()
        self.name = QLineEdit(current_name)
        self.name.setPlaceholderText("Full name or channel of the speaker")
        self.statement = QPlainTextEdit()
        self.statement.setPlaceholderText("e.g. 'I am the speaker' or 'Written permission from <name> on <date>, "
                                          "stored at …'")
        self.statement.setFixedHeight(90)
        form.addRow("Speaker", self.name)
        form.addRow("Consent evidence", self.statement)
        lay.addLayout(form)
        self.check = QCheckBox("I confirm the speaker has given permission to train a synthetic voice from these "
                               "recordings, and I am authorized to process the source material.")
        lay.addWidget(self.check)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        if not self.check.isChecked() or not self.name.text().strip() or not self.statement.toPlainText().strip():
            QMessageBox.warning(self, "Incomplete", "Fill in the speaker, the consent evidence, and tick the box.")
            return
        self.accept()


class ProcessingPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header(
            "Audio Processing & Speakers",
            "Extract → enhance (Demucs) → VAD (Silero) → diarize → transcribe (faster-whisper). Completed stages are "
            "saved and never repeated. Then mark which detected speaker is the consenting target voice."))
        f, lay = card("Downloaded media")
        self.videos = make_table(["ID", "Title", "Duration"] + [s.title() for s in STAGES] + ["Speech", "Speakers"], 1)
        self.videos.setMinimumHeight(240)
        self.videos.itemSelectionChanged.connect(self.load_speakers)
        lay.addWidget(self.videos)
        row = QHBoxLayout()
        self.run_btn = button("Process selected", "Primary", self.process_selected)
        row.addWidget(button("Select all", slot=self.videos.selectAll))
        row.addWidget(self.run_btn)
        row.addWidget(button("Cancel", "Danger", self.cancel))
        row.addWidget(button("Re-run diarization", slot=lambda: self.reset_stages(["diarize"]),
                             tooltip="Clear diarization results (e.g. after changing the backend) and process again"))
        row.addWidget(button("Re-run all stages", slot=lambda: self.reset_stages(STAGES)))
        row.addStretch(1)
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.plabel = QLabel("")
        self.plabel.setObjectName("Subtitle")
        lay.addWidget(self.progress)
        lay.addWidget(self.plabel)
        self.body.addWidget(f)

        f2, lay2 = card("Speakers in selected video(s)")
        self.consent_label = QLabel("")
        self.consent_label.setWordWrap(True)
        lay2.addWidget(self.consent_label)
        self.speakers = make_table(["ID", "Video", "Label", "Speech", "Turns", "Similarity to target", "Role"], 1)
        self.speakers.setMinimumHeight(220)
        lay2.addWidget(self.speakers)
        srow = QHBoxLayout()
        self.player = AudioPlayer()
        srow.addWidget(button("Preview speaker", slot=self.preview))
        srow.addWidget(button("Set as TARGET", "Primary", lambda: self.set_role("target")))
        srow.addWidget(button("Exclude", slot=lambda: self.set_role("excluded")))
        srow.addWidget(button("Unassign", slot=lambda: self.set_role("unassigned")))
        srow.addWidget(button("Suggest matches across videos…", slot=self.suggest,
                              tooltip="Mark speakers whose voice embedding matches your confirmed target"))
        srow.addStretch(1)
        lay2.addLayout(srow)
        lay2.addWidget(self.player)
        brow = QHBoxLayout()
        brow.addWidget(button("Confirm speaker consent…", slot=self.consent))
        self.seg_btn = button("Build dataset segments", "Primary", self.build)
        brow.addWidget(self.seg_btn)
        brow.addStretch(1)
        lay2.addLayout(brow)
        self.body.addWidget(f2)
        self._job: int | None = None
        ctx.tasks.progress.connect(self._on_progress)

    # ------------------------------------------------------------- display
    def refresh(self):
        if self.pid is None:
            return
        rows = self.db.q("SELECT * FROM videos WHERE project_id=? AND download_status='downloaded' ORDER BY id",
                         (self.pid,))
        st = {}
        for r in self.db.q("SELECT s.video_id, s.stage, s.status, s.meta_json FROM video_stages s JOIN videos v ON "
                           "v.id=s.video_id WHERE v.project_id=?", (self.pid,)):
            st[(r["video_id"], r["stage"])] = (r["status"], json.loads(r["meta_json"] or "{}"))
        sym = {"done": "✓", "failed": "✗ failed", "running": "…", "cancelled": "cancelled"}
        data, colors = [], []
        for r in rows:
            stages = [sym.get(st.get((r["id"], s), ("", {}))[0], "") for s in STAGES]
            vad = st.get((r["id"], "vad"), ("", {}))[1]
            spk = self.db.scalar("SELECT COUNT(*) FROM speakers WHERE video_id=?", (r["id"],))
            data.append((r["id"], r["title"], fmt_dur(r["duration"]), *stages,
                         fmt_dur(vad.get("speech_sec")) if vad else "", spk or ""))
            colors.append(ERR if any("failed" in s for s in stages) else (OK if all(s == "✓" for s in stages) else None))
        sel = selected_ids(self.videos)
        fill_table(self.videos, data, [r["id"] for r in rows], colors)
        self._reselect(sel)
        consent = self.db.q1("SELECT target_speaker_name, consent_confirmed_at FROM projects WHERE id=?", (self.pid,))
        if consent and consent["consent_confirmed_at"]:
            self.consent_label.setText(f"<span style='color:{OK}'>Consent confirmed</span> for "
                                       f"<b>{consent['target_speaker_name']}</b>.")
        else:
            self.consent_label.setText(f"<span style='color:{WARN}'>Speaker consent not yet confirmed.</span> "
                                       "Dataset building is disabled until you confirm it.")
        self.load_speakers()

    def _reselect(self, ids):
        if not ids:
            return
        for r in range(self.videos.rowCount()):
            if self.videos.item(r, 0).data(Qt.UserRole) in ids:
                self.videos.selectRow(r)

    def load_speakers(self):
        ids = selected_ids(self.videos)
        if not ids:
            ids = [r["id"] for r in self.db.q("SELECT id FROM videos WHERE project_id=?", (self.pid or -1,))]
        if not ids:
            fill_table(self.speakers, [])
            return
        rows = self.db.q(f"SELECT s.*, v.title FROM speakers s JOIN videos v ON v.id=s.video_id WHERE s.video_id IN "
                         f"({','.join('?' * len(ids))}) ORDER BY s.video_id, s.total_duration DESC", ids)
        fill_table(self.speakers, [(r["id"], r["title"], r["label"], fmt_dur(r["total_duration"]), r["n_turns"],
                                    f"{r['similarity']:.2f}" if r["similarity"] is not None else "—", r["role"])
                                   for r in rows], [r["id"] for r in rows], [ROLE_COLORS.get(r["role"]) for r in rows])

    # ------------------------------------------------------------- actions
    def process_selected(self):
        ids = selected_ids(self.videos)
        if not ids:
            error_box(self, "Nothing selected", "Select downloaded videos to process.")
            return
        if self._job is not None:
            error_box(self, "Busy", "Processing is already running.")
            return
        db, settings = self.db, self.ctx.settings

        def work(c):
            return Pipeline(db, settings).run(ids, c)

        def done(status, res):
            self._job = None
            self.run_btn.setEnabled(True)
            self.refresh()
            self.ctx.notify("speakers")
            if status == "done" and res:
                error_box(self, "Some videos failed", "\n\n".join(res.values()))
            elif status == "failed":
                error_box(self, "Processing failed", str(res))
            elif status == "done":
                self.plabel.setText("Processing complete. Review speakers below.")

        self.run_btn.setEnabled(False)
        self._job = self.ctx.tasks.submit("process", work, project_id=self.pid, on_done=done)

    def cancel(self):
        if self._job is not None:
            self.ctx.tasks.cancel(self._job)
            self.plabel.setText("Cancelling after the current step…")

    def reset_stages(self, stages):
        ids = selected_ids(self.videos)
        if not ids:
            return
        if QMessageBox.question(self, "Reset", f"Clear {', '.join(stages)} results for {len(ids)} video(s)? Their "
                                "speakers and segments will be rebuilt.") != QMessageBox.Yes:
            return
        later = STAGES[STAGES.index(stages[0]):]
        for v in ids:
            self.db.clear_stages(v, later)
            if "diarize" in later:
                self.db.execute("DELETE FROM speakers WHERE video_id=?", (v,))
                self.db.execute("DELETE FROM segments WHERE video_id=?", (v,))
        self.refresh()

    def _on_progress(self, job_id, frac, msg):
        if job_id == self._job:
            self.progress.setValue(int(frac * 1000))
            self.plabel.setText(msg)

    def preview(self):
        ids = selected_ids(self.speakers)
        if ids:
            p = self.db.scalar("SELECT preview_path FROM speakers WHERE id=?", (ids[0],))
            self.player.play(p)

    def set_role(self, role):
        ids = selected_ids(self.speakers)
        for sid in ids:
            set_speaker_role(self.db, sid, role)
        self.load_speakers()
        self.ctx.notify("speakers")

    def suggest(self):
        if self.pid is None:
            return
        refresh_target_embedding(self.db, self.pid)
        thr = self.ctx.settings.processing.speaker_sim_accept
        cand = self.db.q("SELECT s.id, s.label, s.similarity, v.title FROM speakers s JOIN videos v ON v.id=s.video_id "
                         "WHERE v.project_id=? AND s.role='unassigned' AND s.similarity>=? ORDER BY s.similarity DESC",
                         (self.pid, thr))
        if not self.db.scalar("SELECT target_embedding FROM projects WHERE id=?", (self.pid,)):
            info_box(self, "No target yet", "First mark at least one speaker as TARGET manually.")
            return
        if not cand:
            info_box(self, "No matches", f"No unassigned speakers with similarity ≥ {thr:.2f}.")
            return
        listing = "\n".join(f"{c['title'][:50]} / {c['label']}: {c['similarity']:.2f}" for c in cand[:30])
        if QMessageBox.question(self, "Confirm matches",
                                f"Mark these {len(cand)} speakers as TARGET? Listen to previews if unsure.\n\n{listing}") \
                == QMessageBox.Yes:
            for c in cand:
                set_speaker_role(self.db, c["id"], "target")
        self.load_speakers()

    def consent(self):
        if self.pid is None:
            return
        name = self.db.scalar("SELECT target_speaker_name FROM projects WHERE id=?", (self.pid,)) or ""
        d = ConsentDialog(self, name)
        if d.exec() == QDialog.Accepted:
            confirm_consent(self.db, self.pid, d.name.text(), d.statement.toPlainText())
            self.refresh()

    def build(self):
        if self.pid is None:
            return
        if not has_consent(self.db, self.pid):
            self.consent()
            if not has_consent(self.db, self.pid):
                return
        vids = [r["video_id"] for r in self.db.q(
            "SELECT DISTINCT s.video_id FROM speakers s JOIN videos v ON v.id=s.video_id WHERE v.project_id=? "
            "AND s.role='target'", (self.pid,))]
        if not vids:
            error_box(self, "No target speaker", "Mark the target speaker in at least one video first.")
            return
        db, settings, pid = self.db, self.ctx.settings, self.pid

        def done(status, res):
            self.seg_btn.setEnabled(True)
            self._job = None
            self.ctx.notify("segments")
            if status == "done":
                info_box(self, "Segments built",
                         f"{res['created']} segments from {res['videos']} video(s): {res['accepted']} accepted, "
                         f"{res['review']} need review, {res['rejected']} rejected, {res['duplicates']} duplicates "
                         "removed. Continue in the Dataset Manager.")
            elif status == "failed":
                error_box(self, "Segment building failed", str(res))

        self.seg_btn.setEnabled(False)
        self._job = self.ctx.tasks.submit("segments", lambda c: build_segments(db, settings, pid, vids, c),
                                          project_id=pid, on_done=done)
