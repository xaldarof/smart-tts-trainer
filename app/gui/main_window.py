"""Main window: sidebar navigation, project selector, page stack, log dock, GPU status."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDockWidget, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from app.config.settings import APP_NAME
from app.database.db import get_or_create_project
from app.gui.context import AppContext
from app.gui.pages.dashboard import DashboardPage
from app.gui.pages.datasets import DatasetPage
from app.gui.pages.inference import InferencePage
from app.gui.pages.library import LibraryPage
from app.gui.pages.processing import ProcessingPage
from app.gui.pages.settings import SettingsPage
from app.gui.pages.training import TrainingPage
from app.gui.pages.youtube import YouTubePage
from app.gui.widgets import ERR, MUTED, WARN, button
from app.utils.logging_setup import add_log_listener, remove_log_listener

log = logging.getLogger(__name__)

PAGES = [("Dashboard", DashboardPage), ("YouTube Import", YouTubePage), ("Audio Processing", ProcessingPage),
         ("Dataset Manager", DatasetPage), ("Voice Training", TrainingPage), ("TTS Generator", InferencePage),
         ("Model Library", LibraryPage), ("Settings", SettingsPage)]


class MainWindow(QMainWindow):
    _log_line = Signal(str, int)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(230)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 16, 12, 12)
        title = QLabel(f"<b style='font-size:16px'>{APP_NAME}</b><br><span style='color:{MUTED}'>local · private</span>")
        sv.addWidget(title)
        sv.addSpacing(10)
        sv.addWidget(QLabel(f"<span style='color:{MUTED}'>Project</span>"))
        self.project = QComboBox()
        self.project.currentIndexChanged.connect(self._project_selected)
        sv.addWidget(self.project)
        sv.addWidget(button("+ New project", slot=self.new_project))
        sv.addSpacing(12)
        self.stack = QStackedWidget()
        self.group = QButtonGroup(self)
        self.pages = []
        for i, (name, cls) in enumerate(PAGES):
            b = QPushButton(name)
            b.setObjectName("Nav")
            b.setCheckable(True)
            self.group.addButton(b, i)
            sv.addWidget(b)
            page = cls(ctx)
            self.pages.append(page)
            self.stack.addWidget(page)
        self.group.idClicked.connect(self.stack.setCurrentIndex)
        self.group.button(0).setChecked(True)
        sv.addStretch(1)
        self.gpu_label = QLabel("")
        self.gpu_label.setWordWrap(True)
        self.gpu_label.setStyleSheet(f"color:{MUTED}; font-size:12px")
        sv.addWidget(self.gpu_label)
        h.addWidget(side)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        # log dock
        dock = QDockWidget("Activity log", self)
        dock.setObjectName("LogDock")
        self.logview = QPlainTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setMaximumBlockCount(5000)
        self.logview.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size: 12px;")
        dock.setWidget(self.logview)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [110], Qt.Vertical)
        self._log_line.connect(self._append_log)
        self._listener = lambda m, lvl: self._log_line.emit(m, lvl)
        add_log_listener(self._listener)

        self.status = self.statusBar()
        self.jobs_label = QLabel("")
        self.status.addPermanentWidget(self.jobs_label)
        ctx.tasks.progress.connect(lambda j, f, m: self.status.showMessage(f"Job {j}: {m} ({f * 100:.0f}%)", 4000))
        ctx.tasks.changed.connect(self._jobs_changed)
        ctx.tasks.finished.connect(self._job_finished)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(2500)
        self._load_projects()
        self._tick()

    # ------------------------------------------------------------ projects
    def _load_projects(self, select: int | None = None):
        self.project.blockSignals(True)
        self.project.clear()
        rows = self.ctx.db.q("SELECT id, name FROM projects ORDER BY id")
        if not rows:
            get_or_create_project(self.ctx.db, "Default")
            rows = self.ctx.db.q("SELECT id, name FROM projects ORDER BY id")
        for r in rows:
            self.project.addItem(r["name"], r["id"])
        self.project.blockSignals(False)
        idx = self.project.findData(select) if select else 0
        self.project.setCurrentIndex(max(0, idx))
        self._project_selected()

    def _project_selected(self):
        pid = self.project.currentData()
        if pid is not None:
            self.ctx.set_project(int(pid))

    def new_project(self):
        name, ok = QInputDialog.getText(self, "New project", "Project name (one target voice per project):")
        if ok and name.strip():
            pid = get_or_create_project(self.ctx.db, name.strip())
            self._load_projects(pid)

    # -------------------------------------------------------------- status
    def _append_log(self, msg: str, level: int):
        color = ERR if level >= logging.ERROR else (WARN if level >= logging.WARNING else None)
        if color:
            self.logview.appendHtml(f"<span style='color:{color}'>{msg.replace('<', '&lt;')}</span>")
        else:
            self.logview.appendPlainText(msg)

    def _jobs_changed(self):
        n = len(self.ctx.tasks.active())
        t = " · training running" if self.ctx.training.is_running() else ""
        self.jobs_label.setText(f"{n} active job(s){t}")

    def _job_finished(self, job_id, status, result):
        if status == "failed":
            self.status.showMessage(f"Job {job_id} failed: {result}", 10000)
        self._jobs_changed()

    def _tick(self):
        from app.utils.gpu import gpu_info

        g = gpu_info()
        if g.get("available"):
            self.gpu_label.setText(f"GPU: {g['name']}\nVRAM {g['used_gb']:.1f} / {g['total_gb']:.1f} GB used")
        else:
            self.gpu_label.setText("GPU: not available (CPU mode)")
        self._jobs_changed()

    def closeEvent(self, e):
        busy = self.ctx.tasks.active() or self.ctx.training.is_running()
        if busy and QMessageBox.question(
                self, "Quit", "Jobs are still running. Quit anyway? Training will save a checkpoint first (may take a "
                "minute); other jobs resume from their last completed stage next time.") != QMessageBox.Yes:
            e.ignore()
            return
        remove_log_listener(self._listener)
        self.ctx.tasks.shutdown()
        self.ctx.training.kill_all()
        for p in self.pages:
            if isinstance(p, InferencePage):
                p.engine.unload()
        e.accept()
