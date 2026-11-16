"""Voice training: configure, start/stop/resume XTTS fine-tuning, live metrics, export model."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar, QSpinBox, QWidget)

from app.gui.context import AppContext, Page
from app.gui.widgets import (ERR, MUTED, OK, WARN, LineChart, StatCard, button, card, error_box, fill_table, fmt_dur,
                             info_box, make_table, page_header, selected_ids)
from app.services.dataset.builder import validate_dataset
from app.services.training import base_models, manager
from app.services.training.presets import PRESET_LABELS, TrainConfig, preset

STATUS_COLORS = {"completed": OK, "failed": ERR, "running": "#5b8cff", "stopped": MUTED, "interrupted": WARN,
                 "stopping": WARN}


class TrainingPage(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx)
        self.body.addWidget(page_header(
            "Voice Training",
            "Fine-tunes the GPT decoder of XTTS v2 on your dataset (real gradient training via the Coqui trainer, "
            "not reference cloning). Runs in a separate process; checkpoints allow stopping and resuming."))
        top = QHBoxLayout()
        f, lay = card("Configuration")
        form = QFormLayout()
        self.dataset = QComboBox()
        self.preset = QComboBox()
        for k, v in PRESET_LABELS.items():
            self.preset.addItem(v, k)
        self.preset.setCurrentIndex(list(PRESET_LABELS).index(ctx.settings.training.preset))
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.accum = QSpinBox()
        self.accum.setRange(1, 512)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(8)
        self.lr.setRange(1e-8, 1e-3)
        self.lr.setSingleStep(1e-6)
        self.precision = QComboBox()
        self.precision.addItems(["bf16", "fp16", "fp32"])
        self.ckpt = QCheckBox("Gradient checkpointing (less VRAM, ~25% slower)")
        self.workers = QSpinBox()
        self.workers.setRange(0, 16)
        self.save_step = QSpinBox()
        self.save_step.setRange(50, 1_000_000)
        self.save_step.setSingleStep(250)
        self.eff = QLabel("")
        self.eff.setObjectName("Subtitle")
        for w in (self.batch, self.accum):
            w.valueChanged.connect(self._eff)
        form.addRow("Dataset", self.dataset)
        form.addRow("Preset", self.preset)
        form.addRow("Epochs", self.epochs)
        form.addRow("Batch size", self.batch)
        form.addRow("Gradient accumulation", self.accum)
        form.addRow("", self.eff)
        form.addRow("Learning rate", self.lr)
        form.addRow("Precision", self.precision)
        form.addRow("", self.ckpt)
        form.addRow("DataLoader workers", self.workers)
        form.addRow("Checkpoint every N steps", self.save_step)
        lay.addLayout(form)
        brow = QHBoxLayout()
        self.start_btn = button("Start training", "Primary", self.start)
        self.stop_btn = button("Stop (save checkpoint)", "Danger", self.stop)
        self.resume_btn = button("Resume selected run", slot=self.resume)
        brow.addWidget(self.start_btn)
        brow.addWidget(self.stop_btn)
        brow.addWidget(self.resume_btn)
        lay.addLayout(brow)
        top.addWidget(f, 2)

        f2, lay2 = card("Live progress")
        stats = QHBoxLayout()
        self.cards = {k: StatCard(lbl) for k, lbl in [("step", "Step"), ("epoch", "Epoch"), ("loss", "Train loss"),
                                                      ("eval", "Best eval loss"), ("gpu", "GPU memory (peak)")]}
        for c in self.cards.values():
            stats.addWidget(c)
        lay2.addLayout(stats)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.plabel = QLabel("Idle")
        self.plabel.setObjectName("Subtitle")
        lay2.addWidget(self.progress)
        lay2.addWidget(self.plabel)
        self.chart = LineChart()
        lay2.addWidget(self.chart, 1)
        top.addWidget(f2, 3)
        self.body.addLayout(top)

        f3, lay3 = card("Runs")
        self.runs = make_table(["ID", "Status", "Preset", "Dataset", "Step", "Epoch", "Loss", "Best eval", "Started",
                                "Error"], 9)
        self.runs.setMinimumHeight(170)
        self.runs.itemSelectionChanged.connect(self._load_run)
        lay3.addWidget(self.runs)
        r3 = QHBoxLayout()
        r3.addWidget(button("Export selected run as voice model…", "Primary", self.export_model))
        r3.addStretch(1)
        lay3.addLayout(r3)
        self.body.addWidget(f3)
        f4, lay4 = card("Training log")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setMinimumHeight(200)
        lay4.addWidget(self.log)
        self.body.addWidget(f4)

        self._active_run: int | None = None
        self._view_run: int | None = None
        ctx.training.event.connect(self._on_event)
        ctx.training.done.connect(self._on_done)
        self._apply_preset()
        self._update_buttons()

    # ------------------------------------------------------------ config
    def _apply_preset(self):
        c = preset(self.preset.currentData())
        self._set_cfg(c)
        custom = self.preset.currentData() == "custom"
        for w in (self.batch, self.accum, self.lr, self.precision, self.ckpt, self.workers):
            w.setEnabled(custom)

    def _set_cfg(self, c: TrainConfig):
        self.epochs.setValue(c.epochs)
        self.batch.setValue(c.batch_size)
        self.accum.setValue(c.grad_accum)
        self.lr.setValue(c.lr)
        self.precision.setCurrentText(c.precision)
        self.ckpt.setChecked(c.grad_checkpointing)
        self.workers.setValue(c.num_workers)
        self.save_step.setValue(c.save_step)
        self._eff()

    def _eff(self):
        self.eff.setText(f"Effective batch = {self.batch.value() * self.accum.value()}")

    def _cfg(self) -> TrainConfig:
        return TrainConfig(preset=self.preset.currentData(), epochs=self.epochs.value(), batch_size=self.batch.value(),
                           grad_accum=self.accum.value(), lr=self.lr.value(), precision=self.precision.currentText(),
                           grad_checkpointing=self.ckpt.isChecked(), num_workers=self.workers.value(),
                           save_step=self.save_step.value())

    # ----------------------------------------------------------- display
    def refresh(self):
        if self.pid is None:
            return
        cur = self.dataset.currentData()
        self.dataset.clear()
        for r in self.db.q("SELECT * FROM datasets WHERE project_id=? ORDER BY id DESC", (self.pid,)):
            self.dataset.addItem(f"#{r['id']} {r['name']} — {r['n_train']}+{r['n_eval']} clips, "
                                 f"{fmt_dur(r['total_duration'])}", r["id"])
        if cur is not None:
            i = self.dataset.findData(cur)
            if i >= 0:
                self.dataset.setCurrentIndex(i)
        self._load_runs()
        self._update_buttons()

    def _load_runs(self):
        rows = self.db.q("SELECT * FROM training_runs WHERE project_id=? ORDER BY id DESC", (self.pid,))
        fill_table(self.runs, [(r["id"], r["status"], r["preset"], r["dataset_id"], r["current_step"], r["current_epoch"],
                                round(r["last_loss"], 4) if r["last_loss"] is not None else "",
                                round(r["best_eval_loss"], 4) if r["best_eval_loss"] is not None else "",
                                time.strftime("%m-%d %H:%M", time.localtime(r["started_at"])) if r["started_at"] else "",
                                r["error"] or "") for r in rows],
                   [r["id"] for r in rows], [STATUS_COLORS.get(r["status"]) for r in rows])

    def _load_run(self):
        ids = selected_ids(self.runs)
        if ids:
            self._view_run = ids[0]
            self._draw_chart(ids[0])
            r = self.db.q1("SELECT * FROM training_runs WHERE id=?", (ids[0],))
            self._show_stats(r)
        self._update_buttons()

    def _show_stats(self, r):
        self.cards["step"].set(str(r["current_step"]))
        self.cards["epoch"].set(str(r["current_epoch"]))
        self.cards["loss"].set(f"{r['last_loss']:.4f}" if r["last_loss"] is not None else "—")
        self.cards["eval"].set(f"{r['best_eval_loss']:.4f}" if r["best_eval_loss"] is not None else "—")

    def _draw_chart(self, run_id):
        self.chart.clear()
        for split, name, color in (("train", "loss", "#5b8cff"), ("eval", "avg_loss", "#3fb950")):
            rows = self.db.q("SELECT step, value FROM metrics WHERE run_id=? AND split=? AND name=? ORDER BY step",
                             (run_id, split, name))
            if split == "train" and len(rows) > 600:   # downsample by averaging
                k = len(rows) // 300
                rows = [(rows[i]["step"], sum(x["value"] for x in rows[i:i + k]) / len(rows[i:i + k]))
                        for i in range(0, len(rows), k)]
            else:
                rows = [(r["step"], r["value"]) for r in rows]
            if rows:
                self.chart.set_series(f"{split} loss", [a for a, _ in rows], [b for _, b in rows], color)

    def _update_buttons(self):
        busy = self.ctx.training.is_running()
        self.start_btn.setEnabled(not busy and self.dataset.count() > 0)
        self.stop_btn.setEnabled(busy)
        sel = selected_ids(self.runs)
        st = self.db.scalar("SELECT status FROM training_runs WHERE id=?", (sel[0],)) if sel else None
        self.resume_btn.setEnabled(not busy and st in ("stopped", "interrupted", "failed"))

    # ----------------------------------------------------------- actions
    def _preflight(self) -> bool:
        if not base_models.xtts_present(self.ctx.settings):
            if not self.ctx.settings.cpml_accepted:
                error_box(self, "License", "Accept the XTTS v2 model license (CPML, non-commercial) in Settings → "
                          "Models and download the base model first.")
            else:
                error_box(self, "Base model missing", "Download XTTS v2 weights in Settings → Models.")
            return False
        from app.utils.gpu import gpu_info

        g = gpu_info()
        if not g.get("available"):
            return QMessageBox.question(
                self, "No CUDA GPU",
                "No CUDA GPU is available. XTTS fine-tuning on CPU is impractically slow (hours per epoch for small "
                "datasets). Continue anyway?") == QMessageBox.Yes
        if g["total_gb"] < 8:
            error_box(self, "Insufficient VRAM", f"{g['name']} has {g['total_gb']:.1f} GB; XTTS fine-tuning needs ≥ 8 GB.")
            return False
        if g["free_gb"] < 6:
            return QMessageBox.question(
                self, "GPU busy", f"Only {g['free_gb']:.1f} GB VRAM free. Close other GPU apps or stop processing jobs. "
                "Start anyway (may OOM; automatic batch reduction will be attempted)?") == QMessageBox.Yes
        return True

    def start(self):
        ds = self.dataset.currentData()
        if ds is None or not self._preflight():
            return
        rep = validate_dataset(Path(self.db.scalar("SELECT path FROM datasets WHERE id=?", (ds,))))
        if not rep.ok:
            error_box(self, "Invalid dataset", "\n".join(rep.errors[:15]))
            return
        try:
            run_id = manager.create_run(self.db, self.ctx.settings, self.pid, ds, self._cfg())
            self._launch(run_id)
        except Exception as e:
            error_box(self, "Cannot start training", str(e))

    def resume(self):
        ids = selected_ids(self.runs)
        if not ids or not self._preflight():
            return
        try:
            self._launch(ids[0])
        except Exception as e:
            error_box(self, "Cannot resume", str(e))

    def _launch(self, run_id):
        self.log.clear()
        self.ctx.training.start(run_id)
        self._active_run = self._view_run = run_id
        self.plabel.setText(f"Run #{run_id}: starting (loading model and data)…")
        self.chart.clear()
        self._load_runs()
        self._update_buttons()

    def stop(self):
        if self._active_run is not None:
            self.ctx.training.stop(self._active_run)
            self.plabel.setText("Stopping: the trainer will save a checkpoint at the next step…")

    def _on_event(self, run_id, ev):
        kind = ev.get("event")
        if kind == "log":
            self.log.appendPlainText(ev["line"])
            return
        if kind not in ("progress", "train_step"):
            self.log.appendPlainText(f"[{kind}] " + json.dumps({k: v for k, v in ev.items() if k not in ("event", "t")},
                                                               ensure_ascii=False))
        if kind == "progress":
            spe = ev.get("steps_per_epoch") or 0
            ep, eps = ev.get("epoch", 0), ev.get("epochs", 1)
            frac = ((ep + ((ev["step"] % spe) / spe if spe else 0)) / eps) if eps else 0
            self.progress.setValue(int(min(1.0, frac) * 1000))
            el = ev.get("elapsed", 0)
            self.plabel.setText(f"Run #{run_id} · epoch {ep + 1}/{eps} · step {ev['step']}"
                                + (f" · {spe} steps/epoch" if spe else "") + f" · elapsed {fmt_dur(el)}")
            self.cards["step"].set(str(ev["step"]))
            self.cards["epoch"].set(f"{ep + 1}/{eps}")
            if "gpu_peak_gb" in ev:
                self.cards["gpu"].set(f"{ev['gpu_alloc_gb']:.1f} ({ev['gpu_peak_gb']:.1f}) GB")
            if ev["step"] % 20 == 0:
                self._draw_chart(run_id)
        elif kind == "train_step" and "loss" in ev:
            self.cards["loss"].set(f"{ev['loss']:.4f}")
        elif kind == "eval":
            self._draw_chart(run_id)
            self._load_runs()
        elif kind == "failed":
            self.plabel.setText(f"Failed: {ev.get('message')}")
        elif kind in ("checkpoint", "started", "epoch_end"):
            self._load_runs()

    def _on_done(self, run_id, status):
        self._active_run = None
        self._load_runs()
        self._draw_chart(run_id)
        self._update_buttons()
        self.ctx.notify("runs")
        msg = {"completed": "Training completed.", "stopped": "Training stopped; checkpoint saved. Resume anytime.",
               "failed": "Training failed: " + (self.db.scalar("SELECT error FROM training_runs WHERE id=?",
                                                               (run_id,)) or "")}.get(status, status)
        self.plabel.setText(f"Run #{run_id}: {msg}")
        (error_box if status == "failed" else info_box)(self, "Training", msg)

    def export_model(self):
        ids = selected_ids(self.runs)
        if not ids:
            return
        if self.ctx.training.is_running(ids[0]):
            error_box(self, "Busy", "Stop the run before exporting (export reads the saved checkpoint).")
            return
        ck = manager.best_checkpoint(self.db, ids[0])
        if ck is None:
            error_box(self, "No checkpoint", "This run has not saved a checkpoint yet.")
            return
        name, ok = QInputDialog.getText(self, "Export voice model", f"Model name (from {ck.name}):", text="My voice")
        if not ok or not name.strip():
            return
        db, settings, rid = self.db, self.ctx.settings, ids[0]

        def done(status, res):
            if status == "done":
                info_box(self, "Model exported", "The voice model is now available in the Model Library and TTS "
                                                 "Generator.")
                self.ctx.notify("models")
            elif status == "failed":
                error_box(self, "Export failed", str(res))

        self.ctx.tasks.submit("export_model", lambda c: manager.export_model(db, settings, rid, name, ck),
                              project_id=self.pid, pool="io", on_done=done)
