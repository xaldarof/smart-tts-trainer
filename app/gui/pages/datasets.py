"""Dataset manager: review segments, edit transcripts, trim, export & validate."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                               QPlainTextEdit, QProgressBar, QSplitter, QWidget, QVBoxLayout)
from PySide6.QtCore import Qt

from app.gui.context import AppContext, Page
from app.gui.widgets import (ERR, OK, WARN, AudioPlayer, StatCard, Waveform, button, card, error_box, fill_table,
                             fmt_dur, info_box, make_table, page_header, selected_ids)
from app.services.audio.pipeline import recut_segment, video_dir
from app.services.dataset.builder import export_dataset, project_stats, validate_dataset
from app.services.transcription.normalize import normalize_ru, text_warnings
from app.utils.audio import load_mono

STATUS_COLORS = {"accepted": OK, "review": WARN, "rejected": ERR}


class DatasetPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header(
            "Dataset Manager",
            "Only ACCEPTED clips with transcripts are exported. Clips marked REVIEW were flagged (uncertain speaker, "
            "low ASR confidence, auto-expanded numbers…) and need your decision."))
        grid = QGridLayout()
        self.cards = {k: StatCard(lbl) for k, lbl in [
            ("clips", "Accepted clips"), ("dur", "Clean speech"), ("avg", "Average clip"), ("review", "Need review"),
            ("rej", "Rejected"), ("missing", "Missing transcripts")]}
        for i, c in enumerate(self.cards.values()):
            grid.addWidget(c, 0, i)
        self.body.addLayout(grid)
        self.readiness = QLabel("")
        self.readiness.setWordWrap(True)
        self.body.addWidget(self.readiness)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setObjectName("Subtitle")
        self.body.addWidget(self.details)

        f, lay = card("Segments")
        frow = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItems(["review", "accepted", "rejected", "all"])
        self.filter.currentTextChanged.connect(lambda _: self.load_segments())
        frow.addWidget(QLabel("Show"))
        frow.addWidget(self.filter)
        frow.addStretch(1)
        frow.addWidget(button("Accept selected", slot=lambda: self.set_status("accepted")))
        frow.addWidget(button("Reject selected", slot=lambda: self.set_status("rejected")))
        frow.addWidget(button("Mark for review", slot=lambda: self.set_status("review")))
        lay.addLayout(frow)
        split = QSplitter(Qt.Vertical)
        self.table = make_table(["ID", "Video", "Start", "Dur", "Status", "Score", "Spk sim", "ASR p", "Reasons",
                                 "Transcript"], 9, fixed={1: 150, 8: 260})
        self.table.itemSelectionChanged.connect(self.load_editor)
        split.addWidget(self.table)
        ed = QWidget()
        el = QVBoxLayout(ed)
        el.setContentsMargins(0, 6, 0, 0)
        self.wave = Waveform()
        self.wave.bounds_changed.connect(self._bounds)
        el.addWidget(self.wave)
        trow = QHBoxLayout()
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for s in (self.start, self.end):
            s.setDecimals(2)
            s.setRange(0, 1e6)
            s.setSingleStep(0.05)
            s.setSuffix(" s")
        self.player = AudioPlayer()
        trow.addWidget(QLabel("Start"))
        trow.addWidget(self.start)
        trow.addWidget(QLabel("End"))
        trow.addWidget(self.end)
        trow.addWidget(button("Apply trim", slot=self.apply_trim))
        trow.addWidget(self.player, 1)
        el.addLayout(trow)
        self.text = QPlainTextEdit()
        self.text.setFixedHeight(70)
        self.raw = QLabel("")
        self.raw.setObjectName("Subtitle")
        self.raw.setWordWrap(True)
        el.addWidget(self.raw)
        el.addWidget(self.text)
        brow = QHBoxLayout()
        brow.addWidget(button("Save transcript", slot=lambda: self.save_text(None)))
        brow.addWidget(button("Save && accept", "Primary", lambda: self.save_text("accepted")))
        brow.addWidget(button("Normalize text", slot=lambda: self.text.setPlainText(normalize_ru(self.text.toPlainText()))))
        brow.addStretch(1)
        el.addLayout(brow)
        split.addWidget(ed)
        split.setSizes([380, 300])
        lay.addWidget(split)
        f.setMinimumHeight(720)
        self.body.addWidget(f)

        f2, lay2 = card("Exported datasets")
        self.ds = make_table(["ID", "Name", "Train", "Eval", "Duration", "Path"], 5)
        self.ds.setMinimumHeight(150)
        lay2.addWidget(self.ds)
        r2 = QHBoxLayout()
        self.export_btn = button("Export training dataset…", "Primary", self.export)
        r2.addWidget(self.export_btn)
        r2.addWidget(button("Validate selected", slot=self.validate))
        r2.addStretch(1)
        lay2.addLayout(r2)
        self.ex_progress = QProgressBar()
        self.ex_progress.setRange(0, 1000)
        self.ex_progress.hide()
        lay2.addWidget(self.ex_progress)
        self.body.addWidget(f2)
        self._seg: dict | None = None
        self._job = None
        ctx.tasks.progress.connect(lambda j, f_, m: self.ex_progress.setValue(int(f_ * 1000)) if j == self._job else None)

    def on_data_changed(self, topic):
        if topic in ("segments", "datasets") and self.isVisible():
            self.refresh()

    # ------------------------------------------------------------ display
    def refresh(self):
        if self.pid is None:
            return
        s = project_stats(self.db, self.pid)
        self.cards["clips"].set(str(s.n_accepted))
        self.cards["dur"].set(fmt_dur(s.accepted_sec))
        self.cards["avg"].set(f"{s.avg_clip_sec:.1f} s")
        self.cards["review"].set(str(s.n_review))
        self.cards["rej"].set(str(s.n_rejected))
        self.cards["missing"].set(str(s.missing_transcripts))
        color = OK if s.ready and "small" not in s.readiness else (WARN if s.ready else ERR)
        self.readiness.setText(f"<b style='color:{color}'>{s.readiness}</b>  " + " ".join(s.warnings))
        spk = ", ".join(f"{k}: {fmt_dur(v)}" for k, v in sorted(s.speakers.items(), key=lambda kv: -kv[1])[:8])
        rej = ", ".join(f"{k} ×{v}" for k, v in sorted(s.reject_reasons.items(), key=lambda kv: -kv[1])[:8])
        self.details.setText(f"Speaker distribution (accepted): {spk or '—'}\nRejection reasons: {rej or '—'}")
        self.export_btn.setEnabled(s.ready)
        self.load_segments()
        rows = self.db.q("SELECT * FROM datasets WHERE project_id=? ORDER BY id DESC", (self.pid,))
        fill_table(self.ds, [(r["id"], r["name"], r["n_train"], r["n_eval"], fmt_dur(r["total_duration"]), r["path"])
                             for r in rows], [r["id"] for r in rows])

    def load_segments(self):
        if self.pid is None:
            return
        st = self.filter.currentText()
        where = "" if st == "all" else "AND s.status=?"
        args = (self.pid,) if st == "all" else (self.pid, st)
        rows = self.db.q(f"SELECT s.*, v.title FROM segments s JOIN videos v ON v.id=s.video_id WHERE s.project_id=? "
                         f"{where} ORDER BY s.video_id, s.start LIMIT 5000", args)
        fill_table(self.table, [(r["id"], (r["title"] or "")[:28], round(r["start"], 2), round(r["end"] - r["start"], 2),
                                 r["status"], r["quality_score"],
                                 round(r["speaker_sim"], 2) if r["speaker_sim"] is not None else "",
                                 round(r["avg_word_prob"] or 0, 2), r["reasons"], r["text"]) for r in rows],
                   [r["id"] for r in rows], [STATUS_COLORS.get(r["status"]) for r in rows])

    def load_editor(self):
        ids = selected_ids(self.table)
        if len(ids) != 1:
            return
        r = self.db.q1("SELECT * FROM segments WHERE id=?", (ids[0],))
        if r is None:
            return
        self._seg = dict(r)
        self.text.setPlainText(r["text"])
        warn = text_warnings(r["text"])
        self.raw.setText(f"ASR raw: {r['text_raw']}" + (f"   ⚠ {', '.join(warn)}" if warn else ""))
        self.start.setValue(r["start"])
        self.end.setValue(r["end"])
        # waveform: show the segment with 1.5 s context from the enhanced track
        try:
            d = video_dir(self.ctx.settings, r["project_id"], r["video_id"])
            x, sr = load_mono(d / "enhanced.wav")
            a = max(0.0, r["start"] - 1.5)
            b = min(len(x) / sr, r["end"] + 1.5)
            self.wave.set_audio(x[int(a * sr): int(b * sr)], sr, a, (r["start"], r["end"]))
        except Exception as e:
            self.wave.clear()
            self.raw.setText(f"Could not load audio: {e}")
        self.player.load(r["audio_path"])

    def _bounds(self, s, e):
        self.start.setValue(s)
        self.end.setValue(e)

    # ------------------------------------------------------------ editing
    def apply_trim(self):
        if not self._seg:
            return
        try:
            recut_segment(self.db, self.ctx.settings, self._seg["id"], self.start.value(), self.end.value())
        except Exception as e:
            error_box(self, "Trim failed", str(e))
            return
        self.player.load(self._seg["audio_path"])
        self.player.play()
        self.load_segments()

    def save_text(self, status):
        if not self._seg:
            return
        t = self.text.toPlainText().strip()
        if status == "accepted" and not t:
            error_box(self, "Empty transcript", "A clip without a transcript cannot be accepted.")
            return
        upd = {"text": t, "edited": 1}
        if status:
            upd["status"] = status
            upd["reasons"] = "manually reviewed"
        self.db.update("segments", self._seg["id"], **upd)
        self.ctx.notify("segments")
        self.refresh()

    def set_status(self, status):
        ids = selected_ids(self.table)
        if not ids:
            return
        if status == "accepted":
            empty = self.db.scalar(f"SELECT COUNT(*) FROM segments WHERE id IN ({','.join('?' * len(ids))}) AND "
                                   "trim(text)=''", ids)
            if empty:
                error_box(self, "Empty transcripts", f"{empty} selected clip(s) have no transcript; fix them first.")
                return
        self.db.execute(f"UPDATE segments SET status=?, edited=1, reasons=CASE WHEN ?='accepted' THEN 'manually reviewed'"
                        f" ELSE reasons END WHERE id IN ({','.join('?' * len(ids))})", (status, status, *ids))
        self.ctx.notify("segments")
        self.refresh()

    # ------------------------------------------------------------- export
    def export(self):
        name, ok = QInputDialog.getText(self, "Export dataset", "Dataset name:", text="voice")
        if not ok or not name.strip():
            return
        db, settings, pid = self.db, self.ctx.settings, self.pid
        self.ex_progress.show()

        def done(status, res):
            self.ex_progress.hide()
            self._job = None
            if status == "done":
                rep = validate_dataset(Path(db.scalar("SELECT path FROM datasets WHERE id=?", (res,))))
                info_box(self, "Dataset exported", f"{rep.n_train} train / {rep.n_eval} eval clips, "
                         f"{fmt_dur(rep.total_sec)}.\nValidation: {'OK' if rep.ok else 'FAILED'}\n" +
                         "\n".join(rep.errors[:10] + rep.warnings[:5]))
                self.ctx.notify("datasets")
                self.refresh()
            elif status == "failed":
                error_box(self, "Export failed", str(res))

        self._job = self.ctx.tasks.submit("export_dataset", lambda c: export_dataset(db, settings, pid, name, c),
                                          project_id=pid, pool="io", on_done=done)

    def validate(self):
        ids = selected_ids(self.ds)
        if not ids:
            return
        rep = validate_dataset(Path(self.db.scalar("SELECT path FROM datasets WHERE id=?", (ids[0],))))
        (info_box if rep.ok else error_box)(self, "Validation", json.dumps(rep.as_dict(), indent=1, ensure_ascii=False)[:3000])
