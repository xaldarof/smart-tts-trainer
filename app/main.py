"""Local Voice Trainer entry point:  python -m app.main"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from app.config.settings import APP_NAME, Settings
    from app.utils.logging_setup import setup_logging

    settings = Settings.load()
    setup_logging(settings.sub("logs"))
    log = logging.getLogger("app")
    log.info("Starting %s, data dir %s", APP_NAME, settings.data_path)

    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.database.db import Database
    from app.gui.context import AppContext
    from app.gui.main_window import MainWindow
    from app.gui.widgets import STYLESHEET
    from app.workers.tasks import TaskManager, TrainingController

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    db = Database(settings.db_path)
    rec = db.recover_interrupted()
    if any(rec.values()):
        log.warning("Recovered after unclean shutdown: %s", rec)

    def excepthook(t, v, tb):
        import traceback

        detail = "".join(traceback.format_exception(t, v, tb))
        log.error("Unhandled exception:\n%s", detail)
        db.log_error("gui", str(v), detail)
        QMessageBox.critical(None, "Unexpected error", f"{t.__name__}: {v}\n\nDetails were written to the log.")

    sys.excepthook = excepthook

    ctx = AppContext(db, settings, TaskManager(db, settings), TrainingController(db, settings))
    win = MainWindow(ctx)
    win.show()
    if any(rec.values()):
        win.statusBar().showMessage(
            f"Recovered from an unclean shutdown: {rec['jobs']} job(s), {rec['downloads']} download(s), "
            f"{rec['training']} training run(s) can be resumed. Completed stages are kept.", 15000)
    if "--smoke" in sys.argv:  # used by tests/CI: show window briefly, then exit
        from PySide6.QtCore import QTimer

        def cycle():
            for i in range(len(win.pages)):
                win.stack.setCurrentIndex(i)
                app.processEvents()
            app.quit()

        QTimer.singleShot(1500, cycle)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
