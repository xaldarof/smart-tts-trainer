"""Shared state handed to every page."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from app.config.settings import Settings
from app.database.db import Database
from app.workers.tasks import TaskManager, TrainingController


class AppContext(QObject):
    project_changed = Signal(int)
    data_changed = Signal(str)      # topic: videos | speakers | segments | datasets | runs | models | generations

    def __init__(self, db: Database, settings: Settings, tasks: TaskManager, training: TrainingController):
        super().__init__()
        self.db, self.settings, self.tasks, self.training = db, settings, tasks, training
        self.project_id: int | None = None

    def set_project(self, pid: int) -> None:
        self.project_id = pid
        self.project_changed.emit(pid)

    def notify(self, topic: str) -> None:
        self.data_changed.emit(topic)


class Page(QWidget):
    """Base class: scrollable page with a vertical layout ``self.body``."""

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        inner.setObjectName("Page")
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(24, 20, 24, 20)
        self.body.setSpacing(12)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        ctx.project_changed.connect(lambda _: self.refresh())
        ctx.data_changed.connect(self.on_data_changed)

    @property
    def db(self):
        return self.ctx.db

    @property
    def pid(self) -> int | None:
        return self.ctx.project_id

    def refresh(self) -> None:  # pragma: no cover - overridden
        pass

    def on_data_changed(self, topic: str) -> None:
        pass

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
