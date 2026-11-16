"""Rotating-file + console logging, plus an in-process fan-out for the GUI log view."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Callable

_listeners: list[Callable[[str, int], None]] = []


class _FanoutHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        for cb in list(_listeners):
            try:
                cb(msg, record.levelno)
            except Exception:
                pass


def add_log_listener(cb: Callable[[str, int], None]) -> None:
    _listeners.append(cb)


def remove_log_listener(cb: Callable[[str, int], None]) -> None:
    if cb in _listeners:
        _listeners.remove(cb)


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fan = _FanoutHandler()
    fan.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(fan)
    for noisy in ("urllib3", "httpx", "numba", "matplotlib", "fsspec", "filelock", "httpcore", "httpx2", "speechbrain", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
