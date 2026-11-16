"""Framework-agnostic job context: progress reporting + cooperative cancellation.

Services accept a ``JobContext`` and never import Qt, so they can be used from
GUI workers, CLI scripts and tests alike.
"""
from __future__ import annotations

import threading
from typing import Callable


class Cancelled(Exception):
    """Raised inside a job when the user requested cancellation."""


class JobContext:
    def __init__(self, on_progress: Callable[[float, str], None] | None = None,
                 cancel_event: threading.Event | None = None):
        self._on_progress = on_progress
        self.cancel_event = cancel_event or threading.Event()
        self._scale = (0.0, 1.0)

    # progress ------------------------------------------------------------
    def progress(self, fraction: float, message: str = "") -> None:
        lo, hi = self._scale
        frac = lo + (hi - lo) * max(0.0, min(1.0, fraction))
        if self._on_progress:
            self._on_progress(frac, message)

    def sub(self, lo: float, hi: float) -> "JobContext":
        """Child context mapping its 0..1 progress into [lo, hi] of this one."""
        child = JobContext(self._on_progress, self.cancel_event)
        plo, phi = self._scale
        child._scale = (plo + (phi - plo) * lo, plo + (phi - plo) * hi)
        return child

    # cancellation --------------------------------------------------------
    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled by user")


NULL_CONTEXT = JobContext()
