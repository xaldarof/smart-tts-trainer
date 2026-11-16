"""TTS generator: trained voice / pretrained baseline / reference cloning; playback, export, history."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QPlainTextEdit, QProgressBar)

from app.gui.context import AppContext, Page
from app.gui.widgets import (WARN, AudioPlayer, button, card, error_box, fill_table, fmt_dur, info_box, make_table,
                             page_header, selected_ids)
from app.services.inference.xtts_infer import MAX_TEXT_CHARS, XttsEngine, chunk_text, generate
from app.utils.audio import export_audio

MODES = [("trained", "Trained voice (fine-tuned model)"),
         ("pretrained", "Pretrained XTTS v2 baseline (for comparison)"),
         ("clone", "Reference-audio cloning (zero-shot, NOT fine-tuned)")]


class InferencePage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.engine = XttsEngine(ctx.settings)
        self.body.addWidget(page_header(
            "TTS Generator",
            "Type Russian text and synthesize it. 'Trained voice' uses your fine-tuned checkpoint (conditioned on "
            "clips from its own training data). Cloning mode only imitates reference audio with the base model."))
        top = QHBoxLayout()
        f, lay = card("Voice")
        form = QFormLayout()
        self.mode = QComboBox()
        for k, v in MODES:
            self.mode.addItem(v, k)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.model = QComboBox()
        self.backend = QComboBox()
        self.backend.addItem("XTTS v2 (Coqui, CPML non-commercial)", "xtts_v2")
        self.refs = QListWidget()
        self.refs.setFixedHeight(80)
        self.ref_btn = button("Choose reference audio…", slot=self.pick_refs)
        self.temp = QDoubleSpinBox()
        self.temp.setRange(0.05, 1.5)
        self.temp.setSingleStep(0.05)
        self.temp.setValue(0.7)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setSingleStep(0.05)
        self.speed.setValue(1.0)
        self.mode_note = QLabel("")
        self.mode_note.setWordWrap(True)
        self.mode_note.setStyleSheet(f"color:{WARN}")
        form.addRow("Mode", self.mode)
        form.addRow("Backend", self.backend)
        form.addRow("Model", self.model)
        form.addRow("References", self.refs)
        form.addRow("", self.ref_btn)
        form.addRow("Temperature", self.temp)
        form.addRow("Speed", self.speed)
        lay.addLayout(form)
        lay.addWidget(self.mode_note)
        self.meta = QLabel("")
        self.meta.setObjectName("Subtitle")
        self.meta.setWordWrap(True)
        lay.addWidget(self.meta)
        self.model.currentIndexChanged.connect(self._show_meta)
        top.addWidget(f, 2)

        f2, lay2 = card("Text (Russian)")
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Введите текст для озвучивания…")
        self.text.setMinimumHeight(220)
        self.text.textChanged.connect(self._count)
        lay2.addWidget(self.text)
        self.count = QLabel("")
        self.count.setObjectName("Subtitle")
        lay2.addWidget(self.count)
        brow = QHBoxLayout()
        self.gen_btn = button("Generate", "Primary", self.generate)
        brow.addWidget(self.gen_btn)
        brow.addWidget(button("Cancel", "Danger", self.cancel))
        brow.addWidget(button("Unload model (free VRAM)", slot=self.unload))
        brow.addStretch(1)
        lay2.addLayout(brow)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.plabel = QLabel("")
        self.plabel.setObjectName("Subtitle")
        lay2.addWidget(self.progress)
        lay2.addWidget(self.plabel)
        self.player = AudioPlayer()
        lay2.addWidget(self.player)
        erow = QHBoxLayout()
        erow.addWidget(button("Export WAV…", slot=lambda: self.export("wav")))
        erow.addWidget(button("Export MP3…", slot=lambda: self.export("mp3")))
        erow.addStretch(1)
        lay2.addLayout(erow)
        top.addWidget(f2, 3)
        self.body.addLayout(top)

        f3, lay3 = card("Generation history")
        self.hist = make_table(["ID", "When", "Mode", "Model", "Duration", "Text", "File"], 5, fixed={6: 300})
        self.hist.setMinimumHeight(240)
        self.hist.itemSelectionChanged.connect(self._hist_sel)
        lay3.addWidget(self.hist)
        self.body.addWidget(f3)
        self._job = None
        self._last: str | None = None
        self._ref_paths: list[str] = []
        ctx.tasks.progress.connect(self._on_progress)
        self._mode_changed()

    def on_data_changed(self, topic):
        if topic in ("models", "generations") and self.isVisible():
            self.refresh()

    def refresh(self):
        cur = self.model.currentData()
        self.model.clear()
        for r in self.db.q("SELECT * FROM models WHERE kind='finetuned' ORDER BY id DESC"):
            self.model.addItem(f"#{r['id']} {r['name']}", r["id"])
        if cur is not None and self.model.findData(cur) >= 0:
            self.model.setCurrentIndex(self.model.findData(cur))
        rows = self.db.q("SELECT g.*, m.name FROM generations g LEFT JOIN models m ON m.id=g.model_id "
                         "ORDER BY g.id DESC LIMIT 300")
        fill_table(self.hist, [(r["id"], time.strftime("%m-%d %H:%M", time.localtime(r["created_at"])), r["mode"],
                                r["name"] or "", fmt_dur(r["duration"]), r["text"][:120], r["output_path"])
                               for r in rows], [r["id"] for r in rows])
        self._show_meta()

    def _mode_changed(self):
        m = self.mode.currentData()
        self.refs.setEnabled(m == "clone")
        self.ref_btn.setEnabled(m == "clone")
        self.model.setEnabled(m in ("trained", "pretrained"))
        self.mode_note.setText({
            "trained": "",
            "pretrained": "Uses the ORIGINAL XTTS weights with the selected model's reference clips, so you can hear "
                          "what fine-tuning changed.",
            "clone": "Zero-shot cloning from reference audio. This is NOT a trained model. Only use recordings of "
                     "speakers who consented.",
        }[m])

    def _show_meta(self):
        mid = self.model.currentData()
        if mid is None:
            self.meta.setText("No fine-tuned models yet. Train and export one on the Voice Training page.")
            return
        m = self.db.q1("SELECT * FROM models WHERE id=?", (mid,))
        meta = json.loads(m["meta_json"] or "{}")
        self.meta.setText(f"Run #{meta.get('run_id')} · step {meta.get('step')} · epoch {meta.get('epoch')} · "
                          f"eval loss {meta.get('best_eval_loss')} · dataset {meta.get('dataset_minutes', 0):.1f} min · "
                          f"{len(meta.get('refs', []))} reference clips · {meta.get('license', '')}")

    def _count(self):
        t = self.text.toPlainText()
        n = len(t)
        chunks = len(chunk_text(t)) if t.strip() else 0
        self.count.setText(f"{n}/{MAX_TEXT_CHARS} characters · {chunks} chunk(s)")
        self.gen_btn.setEnabled(0 < n <= MAX_TEXT_CHARS and self._job is None)

    def pick_refs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Reference audio (consenting speaker)", "",
                                                "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)")
        if files:
            self._ref_paths = files
            self.refs.clear()
            self.refs.addItems([Path(f).name for f in files])

    def generate(self):
        mode = self.mode.currentData()
        text = self.text.toPlainText()
        mid = self.model.currentData()
        if mode in ("trained", "pretrained") and mid is None:
            error_box(self, "No model", "Select a fine-tuned model.")
            return
        db, settings, eng = self.db, self.ctx.settings, self.engine
        params = {"temperature": self.temp.value(), "speed": self.speed.value()}
        refs = [Path(p) for p in self._ref_paths]

        def work(c):
            return generate(db, settings, eng, c, mode=mode, text=text,
                            model_id=mid if mode == "trained" else None,
                            compare_model_id=mid if mode == "pretrained" else None, ref_wavs=refs, **params)

        def done(status, res):
            self._job = None
            self._count()
            if status == "done":
                g = db.q1("SELECT * FROM generations WHERE id=?", (res,))
                self._last = g["output_path"]
                self.plabel.setText(f"Generated {fmt_dur(g['duration'])} of audio")
                self.player.play(self._last)
                self.ctx.notify("generations")
                self.refresh()
            elif status == "failed":
                self.plabel.setText("Failed")
                error_box(self, "Generation failed", str(res))
            else:
                self.plabel.setText("Cancelled")

        self.gen_btn.setEnabled(False)
        self.plabel.setText("Loading model…")
        self._job = self.ctx.tasks.submit("tts", work, on_done=done)

    def cancel(self):
        if self._job is not None:
            self.ctx.tasks.cancel(self._job)

    def unload(self):
        if self._job is None:
            self.engine.unload()
            self.plabel.setText("Model unloaded")

    def _on_progress(self, job_id, frac, msg):
        if job_id == self._job:
            self.progress.setValue(int(frac * 1000))
            self.plabel.setText(msg)

    def _hist_sel(self):
        ids = selected_ids(self.hist)
        if ids:
            p = self.db.scalar("SELECT output_path FROM generations WHERE id=?", (ids[0],))
            self._last = p
            self.player.load(p)

    def export(self, fmt):
        if not self._last or not Path(self._last).exists():
            error_box(self, "Nothing to export", "Generate audio or select an item in the history first.")
            return
        dst, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", str(Path.home() / f"speech.{fmt}"),
                                             f"{fmt.upper()} (*.{fmt})")
        if not dst:
            return
        try:
            out = export_audio(Path(self._last), Path(dst), fmt)
            info_box(self, "Exported", str(out))
        except Exception as e:
            error_box(self, "Export failed", str(e))
