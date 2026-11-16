"""Settings: device, processing thresholds, model downloads & licenses, diagnostics."""
from __future__ import annotations

from dataclasses import fields

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QSpinBox, QWidget)

from app.config.settings import ProcessingSettings, hf_token
from app.gui.context import AppContext, Page
from app.gui.widgets import ERR, OK, WARN, button, card, error_box, fill_table, info_box, make_table, page_header
from app.services.training import base_models
from app.utils import system_check

CHOICES = {
    "diarization_backend": ["auto", "pyannote", "ecapa"],
    "whisper_model": ["large-v3", "large-v3-turbo", "medium", "small", "base"],
    "whisper_compute_type": ["auto", "float16", "int8_float16", "int8", "float32"],
}
LABELS = {
    "music_separation": "Remove music (Demucs vocal isolation)",
    "denoise": "Additional mild denoise (may color the voice)",
    "ecapa_cluster_threshold": "ECAPA clustering distance threshold",
    "num_speakers": "Number of speakers (0 = auto)",
    "speaker_sim_accept": "Speaker similarity: accept ≥",
    "speaker_sim_reject": "Speaker similarity: reject <",
    "min_avg_word_prob": "Min. average word confidence",
    "min_word_prob": "Min. single-word confidence (else review)",
}


class SettingsPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        s = ctx.settings
        self.body.addWidget(page_header("Settings", f"Data directory: {s.data_path}"))

        f0, l0 = card("General")
        form0 = QFormLayout()
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setCurrentText(s.device)
        self.parallel = QSpinBox()
        self.parallel.setRange(1, 8)
        self.parallel.setValue(s.max_parallel_downloads)
        self.retries = QSpinBox()
        self.retries.setRange(0, 10)
        self.retries.setValue(s.download_retries)
        form0.addRow("Compute device", self.device)
        form0.addRow("Parallel downloads", self.parallel)
        form0.addRow("Download retries", self.retries)
        tok = hf_token()
        form0.addRow("Hugging Face token", QLabel(
            f"<span style='color:{OK}'>found</span> (from HF_TOKEN / huggingface-cli login)" if tok else
            f"<span style='color:{WARN}'>not set</span> - only needed for pyannote diarization. Set the HF_TOKEN "
            "environment variable or run <code>huggingface-cli login</code>; tokens are never stored by this app."))
        l0.addLayout(form0)
        self.body.addWidget(f0)

        f1, l1 = card("Processing & filtering thresholds")
        grid = QGridLayout()
        self.widgets: dict[str, QWidget] = {}
        for i, fd in enumerate(fields(ProcessingSettings)):
            val = getattr(s.processing, fd.name)
            if fd.name in CHOICES:
                w = QComboBox()
                w.addItems(CHOICES[fd.name])
                w.setEditable(fd.name == "whisper_model")
                w.setCurrentText(str(val))
            elif isinstance(val, bool):
                w = QCheckBox()
                w.setChecked(val)
            elif isinstance(val, int):
                w = QSpinBox()
                w.setRange(0, 100000)
                w.setValue(val)
            elif isinstance(val, float):
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(-100, 1000)
                w.setSingleStep(0.05)
                w.setValue(val)
            else:
                w = QLineEdit(str(val))
            self.widgets[fd.name] = w
            grid.addWidget(QLabel(LABELS.get(fd.name, fd.name.replace("_", " ").capitalize())), i // 2, (i % 2) * 2)
            grid.addWidget(w, i // 2, (i % 2) * 2 + 1)
        l1.addLayout(grid)
        r = QHBoxLayout()
        r.addWidget(button("Save settings", "Primary", self.save))
        r.addWidget(button("Restore defaults", slot=self.defaults))
        r.addStretch(1)
        l1.addLayout(r)
        self.body.addWidget(f1)

        f2, l2 = card("Models & licenses")
        lic = QLabel(base_models.XTTS_LICENSE_TEXT)
        lic.setWordWrap(True)
        l2.addWidget(lic)
        self.cpml = QCheckBox("I have read and accept the Coqui Public Model License (non-commercial use only).")
        self.cpml.setChecked(s.cpml_accepted)
        self.cpml.toggled.connect(self._cpml)
        l2.addWidget(self.cpml)
        r2 = QHBoxLayout()
        self.dl_xtts = button("Download XTTS v2 (~1.9 GB)", "Primary", self.download_xtts)
        r2.addWidget(self.dl_xtts)
        r2.addWidget(button("Download processing models (Whisper, Demucs, ECAPA)", slot=self.prefetch))
        r2.addStretch(1)
        l2.addLayout(r2)
        self.mprog = QProgressBar()
        self.mprog.setRange(0, 1000)
        self.mlabel = QLabel("")
        self.mlabel.setObjectName("Subtitle")
        l2.addWidget(self.mprog)
        l2.addWidget(self.mlabel)
        self.body.addWidget(f2)

        f3, l3 = card("System diagnostics")
        self.diag = make_table(["Check", "Status", "Details"], 2)
        self.diag.setMinimumHeight(300)
        l3.addWidget(self.diag)
        l3.addWidget(button("Run diagnostics", slot=self.run_diag))
        self.body.addWidget(f3)
        self.body.addStretch(1)
        self._job = None
        ctx.tasks.progress.connect(self._on_progress)

    def refresh(self):
        self.dl_xtts.setText("XTTS v2 downloaded ✓" if base_models.xtts_present(self.ctx.settings)
                             else "Download XTTS v2 (~1.9 GB)")

    def _cpml(self, on):
        self.ctx.settings.cpml_accepted = on
        self.ctx.settings.save()

    def save(self):
        s = self.ctx.settings
        s.device = self.device.currentText()
        s.max_parallel_downloads = self.parallel.value()
        s.download_retries = self.retries.value()
        for name, w in self.widgets.items():
            cur = getattr(s.processing, name)
            if isinstance(w, QComboBox):
                v = w.currentText()
            elif isinstance(w, QCheckBox):
                v = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                v = type(cur)(w.value())
            else:
                v = w.text()
            setattr(s.processing, name, v)
        if s.processing.speaker_sim_reject > s.processing.speaker_sim_accept:
            error_box(self, "Invalid thresholds", "Speaker 'reject' threshold must be ≤ 'accept' threshold.")
            return
        if s.processing.min_segment_sec >= s.processing.max_segment_sec:
            error_box(self, "Invalid thresholds", "Minimum segment length must be shorter than maximum.")
            return
        s.save()
        self.ctx.tasks.io_pool.setMaxThreadCount(s.max_parallel_downloads)
        info_box(self, "Saved", "Settings saved. Changed thresholds apply the next time segments are built; "
                 "changed processing options apply to stages that have not run yet (use 'Re-run' to redo them).")

    def defaults(self):
        if QMessageBox.question(self, "Defaults", "Restore default processing thresholds?") == QMessageBox.Yes:
            self.ctx.settings.processing = ProcessingSettings()
            self.ctx.settings.save()
            d = ProcessingSettings()
            for name, w in self.widgets.items():
                v = getattr(d, name)
                if isinstance(w, QComboBox):
                    w.setCurrentText(str(v))
                elif isinstance(w, QCheckBox):
                    w.setChecked(v)
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.setValue(v)
                else:
                    w.setText(str(v))

    def download_xtts(self):
        if not self.ctx.settings.cpml_accepted:
            error_box(self, "License", "Accept the CPML first (checkbox above).")
            return
        s = self.ctx.settings

        def done(status, res):
            self._job = None
            self.refresh()
            if status == "failed":
                error_box(self, "Download failed", f"{res}\n\nCheck your internet connection and disk space, then retry.")
            elif status == "done":
                from app.services.training.manager import register_pretrained

                register_pretrained(self.db, s)
                self.ctx.notify("models")
                self.mlabel.setText("XTTS v2 ready")

        self._job = self.ctx.tasks.submit("download_xtts", lambda c: base_models.ensure_xtts(s, c), pool="io",
                                          on_done=done)

    def prefetch(self):
        s = self.ctx.settings

        def done(status, res):
            self._job = None
            if status == "failed":
                error_box(self, "Download failed", str(res))
            elif status == "done":
                self.mlabel.setText("Processing models ready")

        self._job = self.ctx.tasks.submit("download_models", lambda c: base_models.prefetch_processing_models(s, c),
                                          pool="io", on_done=done)

    def _on_progress(self, job_id, frac, msg):
        if job_id == self._job:
            self.mprog.setValue(int(frac * 1000))
            self.mlabel.setText(msg)

    def run_diag(self):
        data = self.ctx.settings.data_path

        def done(status, res):
            if status == "done":
                colors = [OK if c.ok else (WARN if c.severity == "warning" else ERR) for c in res]
                fill_table(self.diag, [c.as_row() for c in res], None, colors)
            else:
                error_box(self, "Diagnostics failed", str(res))

        self.ctx.tasks.submit("diagnostics", lambda c: system_check.run_all(data), pool="io", on_done=done)
