"""Model library: trained voices + base models, metadata, open folder, delete."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPlainTextEdit

from app.gui.context import AppContext, Page
from app.gui.widgets import button, card, error_box, fill_table, fmt_bytes, make_table, page_header, selected_ids
from app.services.training import base_models


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class LibraryPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header("Model Library", "Fine-tuned voices and pretrained base models stored locally."))
        f, lay = card("Voice models")
        self.table = make_table(["ID", "Name", "Kind", "Backend", "Language", "Created", "Size", "Path"], 7)
        self.table.setMinimumHeight(300)
        self.table.itemSelectionChanged.connect(self._sel)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        row.addWidget(button("Open folder", slot=self.open_folder))
        row.addWidget(button("Delete selected", "Danger", self.delete))
        row.addStretch(1)
        lay.addLayout(row)
        self.meta = QPlainTextEdit()
        self.meta.setReadOnly(True)
        self.meta.setMinimumHeight(220)
        lay.addWidget(self.meta)
        self.body.addWidget(f)
        f2, lay2 = card("Base models")
        self.base = make_table(["Model", "Status", "License", "Location"], 3)
        self.base.setMinimumHeight(120)
        lay2.addWidget(self.base)
        self.body.addWidget(f2)
        self.body.addStretch(1)

    def on_data_changed(self, topic):
        if topic == "models" and self.isVisible():
            self.refresh()

    def refresh(self):
        rows = self.db.q("SELECT * FROM models ORDER BY id DESC")
        fill_table(self.table, [(r["id"], r["name"], r["kind"], r["backend"], r["language"],
                                 time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"])),
                                 fmt_bytes(_size(Path(r["path"]))) if r["kind"] == "finetuned" else "",
                                 r["path"]) for r in rows], [r["id"] for r in rows])
        s = self.ctx.settings
        d = base_models.xtts_dir(s)
        fill_table(self.base, [("XTTS v2 (coqui/XTTS-v2 @ " + base_models.XTTS_REVISION[:8] + ")",
                                "downloaded" if base_models.xtts_present(s) else "not downloaded",
                                "CPML 1.0 (non-commercial)", str(d)),
                               (f"faster-whisper {s.processing.whisper_model}", "cached on first use", "MIT",
                                str(s.models_cache / "whisper")),
                               ("Demucs htdemucs", "cached on first use", "MIT", "torch hub cache"),
                               ("SpeechBrain ECAPA (spkrec-ecapa-voxceleb)", "cached on first use", "Apache-2.0",
                                str(s.models_cache / "ecapa")),
                               ("pyannote speaker-diarization-community-1", "optional (gated, HF token)", "CC-BY-4.0",
                                "HF cache"),
                               ("Silero VAD v6", "bundled in pip package", "MIT", "site-packages")])

    def _sel(self):
        ids = selected_ids(self.table)
        if ids:
            r = self.db.q1("SELECT * FROM models WHERE id=?", (ids[0],))
            self.meta.setPlainText(json.dumps(json.loads(r["meta_json"] or "{}"), indent=2, ensure_ascii=False))

    def open_folder(self):
        ids = selected_ids(self.table)
        if ids:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.db.scalar("SELECT path FROM models WHERE id=?", (ids[0],))))

    def delete(self):
        ids = selected_ids(self.table)
        if not ids:
            return
        r = self.db.q1("SELECT * FROM models WHERE id=?", (ids[0],))
        if r["kind"] != "finetuned":
            error_box(self, "Not allowed", "Base models are managed in Settings.")
            return
        if QMessageBox.question(self, "Delete model", f"Delete '{r['name']}' and its files ({r['path']})?") != \
                QMessageBox.Yes:
            return
        shutil.rmtree(r["path"], ignore_errors=True)
        self.db.execute("DELETE FROM models WHERE id=?", (r["id"],))
        self.ctx.notify("models")
        self.refresh()
