"""Shared GUI widgets: theme, audio player, waveform editor, line chart, table helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

ACCENT = "#5b8cff"
BG = "#0f1115"
PANEL = "#171a21"
PANEL2 = "#1e222b"
BORDER = "#2a2f3a"
TEXT = "#e6e8ee"
MUTED = "#8b93a7"
OK = "#3fb950"
WARN = "#d29922"
ERR = "#f85149"

STYLESHEET = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#Page, QStackedWidget {{ background: {BG}; }}
QWidget#Sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
QPushButton#Nav {{ text-align: left; padding: 10px 16px; border: none; border-radius: 8px; background: transparent;
    color: {MUTED}; font-size: 14px; }}
QPushButton#Nav:hover {{ background: {PANEL2}; color: {TEXT}; }}
QPushButton#Nav:checked {{ background: #22304f; color: white; }}
QLabel#Title {{ font-size: 22px; font-weight: 600; }}
QLabel#Subtitle {{ color: {MUTED}; }}
QLabel#StatValue {{ font-size: 24px; font-weight: 600; }}
QLabel#StatLabel {{ color: {MUTED}; font-size: 12px; }}
QFrame#Card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #555b6b; border-color: #22262e; }}
QPushButton#Primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600; }}
QPushButton#Primary:disabled {{ background: #2b3550; color: #7a8299; }}
QPushButton#Danger {{ background: #3a1d1f; border: 1px solid #5c2a2d; color: #ffb4b0; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {PANEL2};
    border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QListWidget {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; }}
QListWidget:disabled {{ background: {PANEL}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; border: 1px solid {BORDER}; }}
QTableWidget, QTableView {{ background: {PANEL}; alternate-background-color: #1a1e26; gridline-color: {BORDER};
    border: 1px solid {BORDER}; border-radius: 8px; selection-background-color: #26375e; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; padding: 6px; border: none;
    border-bottom: 1px solid {BORDER}; }}
QProgressBar {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; }}
QTabBar::tab {{ background: {PANEL2}; padding: 7px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: white; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QToolTip {{ background: {PANEL2}; border: 1px solid {BORDER}; color: {TEXT}; }}
"""


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 12)
    if title:
        t = QLabel(title)
        t.setStyleSheet(f"color:{MUTED}; font-weight:600;")
        lay.addWidget(t)
    return f, lay


def page_header(title: str, subtitle: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 8)
    t = QLabel(title)
    t.setObjectName("Title")
    lay.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("Subtitle")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def button(text: str, kind: str = "", slot=None, tooltip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if slot:
        b.clicked.connect(slot)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def make_table(headers: Sequence[str], stretch_col: int | None = None,
               fixed: dict[int, int] | None = None) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(list(headers))
    t.setAlternatingRowColors(True)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.verticalHeader().setVisible(False)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch_col is not None:
        h.setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    for col, width in (fixed or {}).items():   # long free-text columns: fixed width + tooltip
        h.setSectionResizeMode(col, QHeaderView.Interactive)
        t.setColumnWidth(col, width)
    t.setWordWrap(False)
    return t


def fill_table(t: QTableWidget, rows: Iterable[Sequence[Any]], ids: Iterable[Any] | None = None,
               colors: Iterable[str | None] | None = None) -> None:
    rows = list(rows)
    ids = list(ids) if ids is not None else [None] * len(rows)
    colors = list(colors) if colors is not None else [None] * len(rows)
    t.setSortingEnabled(False)
    t.setRowCount(len(rows))
    for r, (vals, rid, col) in enumerate(zip(rows, ids, colors)):
        for c, v in enumerate(vals):
            it = QTableWidgetItem("" if v is None else str(v))
            if isinstance(v, str) and len(v) > 40:
                it.setToolTip(v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                it.setData(Qt.EditRole, v)
            if c == 0:
                it.setData(Qt.UserRole, rid)
            if col:
                it.setForeground(QBrush(QColor(col)))
            t.setItem(r, c, it)
    t.setSortingEnabled(True)


def selected_ids(t: QTableWidget) -> list[Any]:
    rows = sorted({i.row() for i in t.selectedIndexes()})
    return [t.item(r, 0).data(Qt.UserRole) for r in rows if t.item(r, 0) is not None]


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bytes(n: float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def error_box(parent, title: str, msg: str) -> None:
    QMessageBox.critical(parent, title, msg)


def info_box(parent, title: str, msg: str) -> None:
    QMessageBox.information(parent, title, msg)


class StatCard(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.value = QLabel("—")
        self.value.setObjectName("StatValue")
        self.label = QLabel(label)
        self.label.setObjectName("StatLabel")
        lay.addWidget(self.value)
        lay.addWidget(self.label)

    def set(self, value: str, sub: str | None = None):
        self.value.setText(value)
        if sub is not None:
            self.label.setText(sub)


class AudioPlayer(QWidget):
    """Minimal play/stop control around QMediaPlayer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.out = QAudioOutput(self)
        self.player.setAudioOutput(self.out)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.play_btn = button("▶ Play", slot=self.toggle)
        self.label = QLabel("")
        self.label.setObjectName("Subtitle")
        lay.addWidget(self.play_btn)
        lay.addWidget(self.label, 1)
        self.player.playbackStateChanged.connect(self._state)
        self.player.errorOccurred.connect(lambda e, s: self.label.setText(f"Playback error: {s}"))
        self._path: str | None = None

    def load(self, path: str | Path | None):
        self.player.stop()
        self._path = str(path) if path else None
        self.label.setText(Path(path).name if path else "")
        if path:
            self.player.setSource(QUrl.fromLocalFile(str(path)))

    def play(self, path: str | Path | None = None):
        if path is not None and str(path) != self._path:
            self.load(path)
        if self._path:
            self.player.setPosition(0)
            self.player.play()

    def toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.stop()
        else:
            self.play()

    def stop(self):
        self.player.stop()

    def _state(self, st):
        self.play_btn.setText("■ Stop" if st == QMediaPlayer.PlayingState else "▶ Play")


class Waveform(QWidget):
    """Waveform with draggable start/end handles (for trimming). Times are in seconds
    relative to ``offset`` (absolute position of the first displayed sample)."""

    bounds_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.env: np.ndarray | None = None
        self.duration = 0.0
        self.offset = 0.0
        self.sel = (0.0, 0.0)
        self._drag: str | None = None
        self.setMouseTracking(True)

    def set_audio(self, x: np.ndarray, sr: int, offset: float, sel: tuple[float, float]):
        self.duration = len(x) / sr if sr else 0.0
        self.offset = offset
        n = 800
        if len(x) >= n:
            frames = x[: (len(x) // n) * n].reshape(n, -1)
            self.env = np.abs(frames).max(axis=1)
        else:
            self.env = np.abs(x)
        self.sel = sel
        self.update()

    def clear(self):
        self.env = None
        self.update()

    def _t2x(self, t: float) -> float:
        return (t - self.offset) / self.duration * self.width() if self.duration else 0

    def _x2t(self, x: float) -> float:
        return self.offset + max(0.0, min(1.0, x / max(1, self.width()))) * self.duration

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        if self.env is None or not len(self.env):
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No segment selected")
            return
        w, h = self.width(), self.height()
        mid = h / 2
        a, b = self._t2x(self.sel[0]), self._t2x(self.sel[1])
        p.fillRect(QRectF(a, 0, b - a, h), QColor(91, 140, 255, 40))
        peak = float(self.env.max()) or 1.0
        path = QPainterPath()
        n = len(self.env)
        for i, v in enumerate(self.env):
            x = i / n * w
            y = v / peak * (h / 2 - 6)
            path.moveTo(QPointF(x, mid - y))
            path.lineTo(QPointF(x, mid + y))
        p.setPen(QPen(QColor("#7fa3ff"), 1))
        p.drawPath(path)
        p.setPen(QPen(QColor(OK), 2))
        p.drawLine(QPointF(a, 0), QPointF(a, h))
        p.setPen(QPen(QColor(ERR), 2))
        p.drawLine(QPointF(b, 0), QPointF(b, h))

    def mousePressEvent(self, e):
        if self.env is None:
            return
        x = e.position().x()
        a, b = self._t2x(self.sel[0]), self._t2x(self.sel[1])
        self._drag = "start" if abs(x - a) <= abs(x - b) else "end"
        self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if not self._drag or not (e.buttons() & Qt.LeftButton):
            return
        t = self._x2t(e.position().x())
        s, en = self.sel
        if self._drag == "start":
            s = min(t, en - 0.3)
        else:
            en = max(t, s + 0.3)
        self.sel = (s, en)
        self.update()
        self.bounds_changed.emit(s, en)

    def mouseReleaseEvent(self, _):
        self._drag = None


class LineChart(QWidget):
    """Tiny multi-series line chart (no extra dependencies)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.series: dict[str, tuple[list[float], list[float], str]] = {}

    def set_series(self, name: str, xs: list[float], ys: list[float], color: str):
        self.series[name] = (xs, ys, color)
        self.update()

    def clear(self):
        self.series.clear()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        pts = [(x, y) for xs, ys, _ in self.series.values() for x, y in zip(xs, ys) if np.isfinite(y)]
        if not pts:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No metrics yet")
            return
        L, R, T, B = 50, 12, 12, 24
        w, h = self.width() - L - R, self.height() - T - B
        x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
        y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
        if x1 == x0:
            x1 = x0 + 1
        if y1 == y0:
            y1 = y0 + 1e-3
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(BORDER))
        for i in range(5):
            y = T + h * i / 4
            p.drawLine(L, int(y), L + w, int(y))
            p.setPen(QColor(MUTED))
            p.drawText(2, int(y) + 4, f"{y1 - (y1 - y0) * i / 4:.3g}")
            p.setPen(QColor(BORDER))
        p.setPen(QColor(MUTED))
        p.drawText(L, self.height() - 6, f"step {int(x0)}")
        p.drawText(L + w - 70, self.height() - 6, f"step {int(x1)}")
        lx = L + 8
        for name, (xs, ys, color) in self.series.items():
            path = QPainterPath()
            first = True
            for x, y in zip(xs, ys):
                if not np.isfinite(y):
                    continue
                px = L + (x - x0) / (x1 - x0) * w
                py = T + (1 - (y - y0) / (y1 - y0)) * h
                if first:
                    path.moveTo(px, py)
                    first = False
                else:
                    path.lineTo(px, py)
            p.setPen(QPen(QColor(color), 1.6))
            p.drawPath(path)
            p.drawText(lx, T + 12, name)
            lx += 90
