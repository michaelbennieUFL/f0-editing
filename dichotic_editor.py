"""Dichotic F0 editor with Praat's native overlap-add resynthesis.

Orange tier -> left channel; blue tier -> right channel. These are independent
Praat PitchTier objects, both resynthesized from the SAME mono input.

This program does not resynthesize during a drag. Re-synthesis is scheduled
when a drag ends, and Play waits for the latest rendered sound if necessary.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QBrush, QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStatusBar, QVBoxLayout,
    QWidget, QTabBar,
)

LEFT = "left"
RIGHT = "right"
COLORS = {LEFT: QColor("#ec8738"), RIGHT: QColor("#2885e3")}
BG = QColor("#fcfdff")
TEXT = QColor("#253246")
MUTED = QColor("#65758a")
GRID = QColor("#e5ebf2")


from praat_backend import (
    PraatEngine, checked_points, infer_segment_breaks, normalize_breaks,
    resample_editable_pitch, insert_editable_point, delete_editable_point,
    simplify_segmented_pitch, mix_white_noise, normalize_stereo_levels,
)
from dichotic.project import PROJECT_FORMAT, migrate_legacy, new_project, resolve_audio_path
from dichotic.interfaces.pitchflower import PitchFlowerEngine, PitchFlowerResult


@dataclass
class PlaybackRequest:
    start: float
    end: float
    mode: str  # stereo, left, right, original, swap


class WorkSignals(QObject):
    ready = pyqtSignal(int, int, object, object)  # session, revision, audio, error


class PitchCanvas(QWidget):
    """Two overlapped editable curves, with the active channel in the foreground."""

    change_committed = pyqtSignal()
    selection_changed = pyqtSignal()
    active_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(310)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.left = np.empty((0, 2), dtype=float)
        self.right = np.empty((0, 2), dtype=float)
        self.original = np.empty((0, 2), dtype=float)
        self.original_breaks: tuple[int, ...] = ()
        self.result_left = np.empty((0, 2), dtype=float)
        self.result_right = np.empty((0, 2), dtype=float)
        # Indices of the first point in each voiced region after a gap.
        # Independent for left and right; these survive point re-spacing.
        self.segment_breaks: dict[str, tuple[int, ...]] = {LEFT: (), RIGHT: ()}
        self.active = LEFT
        self.selected_index: int | None = None
        self._dragging = False
        self._dragged = False
        self._clicked_before_drag: Callable[[], None] | None = None
        self.duration = 1.0
        self.ymin = 60.0
        self.ymax = 650.0
        self.margin_l = 64
        self.margin_r = 20
        self.margin_t = 24
        self.margin_b = 40
        self.on_begin_edit: Callable[[], None] | None = None

    def set_data(self, points: np.ndarray, duration: float, low: float, high: float):
        self.left = points.copy()
        self.right = points.copy()
        self.original = points.copy()  # Immutable initial analysis reference.
        original_breaks = infer_segment_breaks(points)
        self.original_breaks = original_breaks
        self.segment_breaks = {LEFT: original_breaks, RIGHT: original_breaks}
        self.duration = duration
        self.ymin = max(30.0, low * 0.72)
        self.ymax = high * 1.12
        self.selected_index = None
        self.update()

    def set_result(self, left: np.ndarray | None, right: np.ndarray | None):
        """Set measured synthesis output; never changes editable control points."""
        self.result_left = (np.empty((0, 2), dtype=float) if left is None
                            else np.asarray(left, dtype=float).reshape((-1, 2)).copy())
        self.result_right = (np.empty((0, 2), dtype=float) if right is None
                             else np.asarray(right, dtype=float).reshape((-1, 2)).copy())
        self.update()

    def set_active(self, active: str):
        if active not in (LEFT, RIGHT):
            return
        if self.active != active:
            self.active = active
            self.selected_index = None
            self.active_changed.emit(active)
            self.selection_changed.emit()
            self.update()

    def points(self, side: str | None = None) -> np.ndarray:
        return self.left if (side or self.active) == LEFT else self.right

    def set_points(self, side: str, points: np.ndarray, breaks=None):
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        setattr(self, side, points.copy())
        self.segment_breaks[side] = (infer_segment_breaks(points) if breaks is None
                                     else normalize_breaks(breaks, len(points)))
        self.selected_index = None
        self.selection_changed.emit()
        self.update()

    def plot_rect(self) -> QRectF:
        return QRectF(float(self.margin_l), float(self.margin_t),
                      float(max(20, self.width() - self.margin_l - self.margin_r)),
                      float(max(20, self.height() - self.margin_t - self.margin_b)))

    def coord(self, x: float, y: float) -> QPointF:
        rect = self.plot_rect()
        return QPointF(rect.left() + (x / max(self.duration, 1e-9)) * rect.width(),
                       rect.bottom() - ((y - self.ymin) / max(self.ymax - self.ymin, 1e-9)) * rect.height())

    def point_data(self, x: float, y: float) -> tuple[float, float]:
        rect = self.plot_rect()
        xx = (x - rect.left()) / rect.width() * self.duration
        yy = self.ymin + (rect.bottom() - y) / rect.height() * (self.ymax - self.ymin)
        return float(np.clip(xx, 0, self.duration)), float(np.clip(yy, self.ymin, self.ymax))

    def nearest(self, pos: QPointF, radius: float = 13) -> int | None:
        tier = self.points()
        if len(tier) == 0:
            return None
        rect = self.plot_rect()
        px = rect.left() + tier[:, 0] * rect.width() / self.duration
        py = rect.bottom() - (tier[:, 1] - self.ymin) * rect.height() / (self.ymax - self.ymin)
        d2 = (px - pos.x())**2 + (py - pos.y())**2
        idx = int(np.argmin(d2))
        return idx if d2[idx] <= radius**2 else None

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)
        rect = self.plot_rect()
        p.setFont(QFont("Sans", 9))
        p.setPen(QPen(GRID, 1))
        p.drawRect(rect)
        # Ruler and grid.
        f_step = 50 if self.ymax - self.ymin < 700 else 100
        for f in np.arange(math.ceil(self.ymin / f_step) * f_step, self.ymax + 1e-6, f_step):
            pt = self.coord(0, float(f))
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(rect.left(), pt.y()), QPointF(rect.right(), pt.y()))
            p.setPen(MUTED)
            p.drawText(QRectF(0, pt.y()-9, self.margin_l-8, 18), Qt.AlignmentFlag.AlignRight, f"{f:.0f}")
        step = nice_time_tick(self.duration)
        for t in np.arange(0, self.duration + 1e-8, step):
            pt = self.coord(float(t), self.ymin)
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(pt.x(), rect.top()), QPointF(pt.x(), rect.bottom()))
            p.setPen(MUTED)
            p.drawText(QRectF(pt.x()-30, rect.bottom()+4, 60, 20), Qt.AlignmentFlag.AlignHCenter, f"{t:.2f}")
        p.setPen(TEXT)
        p.setFont(QFont("Sans", 10, QFont.Weight.DemiBold))
        p.drawText(int(rect.left()), 17, "F0 (Hz)   |   Orange ▼ LEFT     Blue ▲ RIGHT     (double-click: add point)")
        p.setFont(QFont("Sans", 9))
        p.setPen(MUTED)
        p.drawText(QRectF(rect.left(), rect.bottom()+23, rect.width(), 17), Qt.AlignmentFlag.AlignCenter, "Time (seconds)")
        # Original is a faint dashed reference, and active channel is on top.
        self._draw_tier(p, self.original, QColor("#a4a9b2"), dashed=True,
                        breaks=self.original_breaks)
        # Measured generated F0 is deliberately a result-only overlay. It is
        # not fed back into either editable tier.
        self._draw_tier(p, self.result_left, QColor("#d34b94"), dashed=True)
        self._draw_tier(p, self.result_right, QColor("#7654c9"), dashed=True)
        for side in (RIGHT, LEFT) if self.active == LEFT else (LEFT, RIGHT):
            self._draw_tier(p, self.points(side), COLORS[side], side=side,
                            breaks=self.segment_breaks[side], foreground=(side == self.active))
        p.end()

    def _draw_tier(self, painter: QPainter, tier: np.ndarray, color: QColor,
                   dashed: bool = False, foreground: bool = False, side: str | None = None,
                   breaks: tuple[int, ...] = ()):
        if not len(tier):
            return
        painter.save()
        if len(tier) >= 2:
            pen = QPen(color, 2.3 if foreground else 1.8)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidthF(1.0)
            painter.setPen(pen)
            # Do not draw a connector across unvoiced pauses / separate islands.
            for segment in np.split(tier, breaks):
                if len(segment) < 2:
                    continue
                path = QPainterPath(self.coord(float(segment[0, 0]), float(segment[0, 1])))
                for x, y in segment[1:]:
                    path.lineTo(self.coord(float(x), float(y)))
                painter.drawPath(path)
        if not dashed:
            for i, (x, y) in enumerate(tier):
                pt = self.coord(float(x), float(y))
                selected = foreground and self.selected_index == i
                radius = 7.2 if selected else (6.1 if foreground else 5.0)
                # Left channel ▼ points downward, right channel ▲ points upward.
                if side == LEFT:
                    corners = [QPointF(pt.x()-radius, pt.y()-radius*0.75),
                               QPointF(pt.x()+radius, pt.y()-radius*0.75),
                               QPointF(pt.x(), pt.y()+radius)]
                else:
                    corners = [QPointF(pt.x()-radius, pt.y()+radius*0.75),
                               QPointF(pt.x()+radius, pt.y()+radius*0.75),
                               QPointF(pt.x(), pt.y()-radius)]
                painter.setPen(QPen(QColor("#152439") if selected else color, 1.4))
                painter.setBrush(QBrush(QColor("#ffffff") if selected else color))
                painter.drawPolygon(QPolygonF(corners))
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus()
        idx = self.nearest(event.position())
        self.selected_index = idx
        self._dragging = idx is not None
        self._dragged = False
        self.selection_changed.emit()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if not self._dragging or self.selected_index is None:
            return super().mouseMoveEvent(event)
        arr = self.points()
        idx = self.selected_index
        t, f = self.point_data(event.position().x(), event.position().y())
        epsilon = min(0.0001, self.duration / 10000)
        before = arr[idx - 1, 0] + epsilon if idx > 0 else 0
        after = arr[idx + 1, 0] - epsilon if idx < len(arr) - 1 else self.duration
        t = np.clip(t, before, after)
        if not self._dragged:
            self._dragged = True
            if self.on_begin_edit:
                self.on_begin_edit()
        arr[idx] = (t, f)
        self.selection_changed.emit()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        if self._dragging and self._dragged:
            self.change_committed.emit()
        self._dragging = False
        self._dragged = False
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.plot_rect().contains(event.position()):
            return
        t, f = self.point_data(event.position().x(), event.position().y())
        modifiers = event.modifiers()
        side = (LEFT if modifiers & Qt.KeyboardModifier.ShiftModifier else
                RIGHT if modifiers & Qt.KeyboardModifier.ControlModifier else self.active)
        try:
            new_points, new_breaks, idx = insert_editable_point(
                self.points(side), self.segment_breaks[side], t, f)
        except ValueError:
            return
        if self.on_begin_edit:
            self.on_begin_edit()
        self.set_active(side)
        setattr(self, side, new_points)
        self.segment_breaks[side] = new_breaks
        self.selected_index = idx
        self.selection_changed.emit()
        self.change_committed.emit()
        self.update()
        event.accept()


def nice_time_tick(duration: float) -> float:
    ideal = max(0.02, duration / 7.5)
    exponent = 10**math.floor(math.log10(ideal))
    for multiplier in (1, 2, 5, 10):
        if multiplier * exponent >= ideal:
            return multiplier * exponent
    return 10 * exponent


class WaveTimeline(QWidget):
    """Waveform region drag + bottom timeline with modifier-click playback."""
    play_requested = pyqtSignal(float, str)
    region_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.duration = 1.0
        self.waveform = np.zeros(2)
        self.generated_stereo: np.ndarray | None = None
        self.rms_times = np.empty(0, dtype=float)
        self.rms_values = np.empty((0, 2), dtype=float)
        self._select_start: float | None = None
        self.selection: tuple[float, float] | None = None
        self._dragging = False
        self.cursor = 0.0
        self.margin_l = 64
        self.margin_r = 20
        self.playbar_h = 40

    def set_audio(self, mono: np.ndarray, rate: int, generated: np.ndarray | None = None):
        self.duration = len(mono) / rate
        self.waveform = np.asarray(mono, dtype=float)
        self.set_generated(generated, rate)
        self.selection = None
        self.cursor = 0
        self.update()

    def set_generated(self, stereo: np.ndarray | None, rate: int):
        self.generated_stereo = None if stereo is None else np.asarray(stereo, dtype=float).reshape((-1, 2))
        if self.generated_stereo is None or len(self.generated_stereo) == 0:
            self.rms_times = np.empty(0, dtype=float)
            self.rms_values = np.empty((0, 2), dtype=float)
            self.update()
            return
        width = max(32, min(800, self.width() or 400))
        block = max(1, int(np.ceil(len(self.generated_stereo) / width)))
        count = int(np.ceil(len(self.generated_stereo) / block))
        values = np.zeros((count, 2), dtype=float)
        for index in range(count):
            chunk = self.generated_stereo[index * block:(index + 1) * block]
            values[index] = np.sqrt(np.mean(np.square(chunk), axis=0))
        self.rms_values = values
        self.rms_times = (np.arange(count) * block + block / 2) / rate
        self.update()

    def _x(self, time_s: float) -> float:
        return self.margin_l + time_s / max(self.duration, 1e-9) * max(1, self.width()-self.margin_l-self.margin_r)

    def _t(self, x: float) -> float:
        return float(np.clip((x-self.margin_l) / max(1,self.width()-self.margin_l-self.margin_r)*self.duration,0,self.duration))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#f7f9fc"))
        left, right = self.margin_l, self.width()-self.margin_r
        midy = (self.height()-self.playbar_h)/2
        wav_h = max(10, self.height()-self.playbar_h-14)
        # Fast peak envelope so multi-minute files are still responsive.
        waveform = self.waveform
        if len(waveform) > 1:
            p.setPen(QPen(QColor("#8896a6"), 1))
            width = max(1,int(right-left))
            scale = max(float(np.max(np.abs(waveform))), 0.01)
            for i in range(width):
                a = int(i*len(waveform)/width)
                b = min(len(waveform), int((i+1)*len(waveform)/width) + 1)
                if a >= len(waveform):
                    break
                window = waveform[a:b]
                ymin = float(np.min(window))
                ymax = float(np.max(window))
                peakscale = wav_h * 0.45 / scale
                p.drawLine(QPointF(left+i, midy-ymin*peakscale), QPointF(left+i, midy-ymax*peakscale))
        if len(self.rms_values):
            source_scale = max(float(np.max(np.abs(waveform))), 0.01)
            rms_scale = wav_h * 0.45 / source_scale
            width = max(1, int(right - left))
            # Show generated left/right RMS as colored envelopes over the
            # waveform. The labels report absolute RMS and dBFS.
            for channel, color, baseline_sign, label in (
                (0, QColor("#d47724"), -1.0, "L"),
                (1, QColor("#287ac4"), 1.0, "R"),
            ):
                p.setPen(QPen(color, 2))
                points = []
                for x in range(width):
                    t = (x / max(1, width - 1)) * self.duration
                    rms = float(np.interp(t, self.rms_times, self.rms_values[:, channel], left=self.rms_values[0, channel], right=self.rms_values[-1, channel]))
                    points.append(QPointF(left + x, midy + baseline_sign * rms * rms_scale))
                for first, second in zip(points, points[1:]):
                    p.drawLine(first, second)
                mean_rms = float(np.sqrt(np.mean(np.square(self.generated_stereo[:, channel]))))
                db = 20 * np.log10(max(mean_rms, 1e-9))
                p.drawText(QRectF(left + 8, 5 + channel * 16, 250, 16), f"RMS {label}: {mean_rms:.4f} ({db:.1f} dBFS)")
        if self.selection is not None:
            a,b = self.selection
            srect = QRectF(self._x(a), 1, max(1,self._x(b)-self._x(a)), self.height()-self.playbar_h-2)
            p.fillRect(srect, QColor(43,124,214,40))
            p.setPen(QPen(QColor("#5891bd"), 1))
            p.drawLine(QPointF(self._x(a), 0), QPointF(self._x(a), self.height()-self.playbar_h))
            p.drawLine(QPointF(self._x(b), 0), QPointF(self._x(b), self.height()-self.playbar_h))
        playrect = QRectF(left, self.height()-self.playbar_h, max(1,right-left),self.playbar_h-2)
        p.fillRect(playrect, QColor("#e5edf6"))
        p.setPen(QPen(QColor("#c7d6e6"), 1))
        p.drawRect(playrect)
        tick = nice_time_tick(self.duration)
        p.setFont(QFont("Sans",9))
        for t in np.arange(0,self.duration+1e-8,tick):
            xx = self._x(float(t))
            p.setPen(QPen(QColor("#9eafc3"), 1))
            p.drawLine(QPointF(xx,self.height()-self.playbar_h),QPointF(xx,self.height()-self.playbar_h+8))
            p.setPen(QColor("#485b72"))
            p.drawText(QRectF(xx-26,self.height()-self.playbar_h+11,52,22),Qt.AlignmentFlag.AlignHCenter,f"{t:.2f}s")
        p.setPen(QPen(QColor("#3b6b9e"), 2))
        cx = self._x(self.cursor)
        p.drawLine(QPointF(cx,0), QPointF(cx,self.height()-3))
        p.setPen(TEXT)
        p.setFont(QFont("Sans",9,QFont.Weight.DemiBold))
        p.drawText(8,18,"Audio")
        p.drawText(6,self.height()-15,"PLAY")
        p.end()

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:
            return
        if event.position().y() >= self.height()-self.playbar_h:
            self.cursor = self._t(event.position().x())
            mods=event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and mods & Qt.KeyboardModifier.ShiftModifier:
                mode="swap"
            elif mods & Qt.KeyboardModifier.ControlModifier:
                mode="right"
            elif mods & Qt.KeyboardModifier.ShiftModifier:
                mode="left"
            elif mods & Qt.KeyboardModifier.AltModifier:
                mode="original"
            else:
                mode="stereo"
            self.play_requested.emit(self.cursor,mode)
            self.update()
        else:
            self._dragging=True
            self._select_start=self._t(event.position().x())
            self.selection=None
            self.update()

    def mouseMoveEvent(self,event):
        if self._dragging and self._select_start is not None:
            other=self._t(event.position().x())
            self.selection=(min(self._select_start,other),max(self._select_start,other))
            self.update()

    def mouseReleaseEvent(self,event):
        if self._dragging:
            self._dragging=False
            if self.selection is not None and self.selection[1]-self.selection[0]<0.01:
                self.selection=None
            self.region_changed.emit(self.selection)
            self.update()


class Editor(QMainWindow):
    def __init__(self, audio: str | None = None):
        super().__init__()
        self.setWindowTitle("Dichotic Pitch Editor — Praat overlap-add")
        self.resize(1250, 950)
        self.engine: PraatEngine | None = None
        self.engines: dict[str, object] = {}
        self.interface_states: dict[str, dict] = {}
        self.interface_errors: dict[str, str] = {}
        self.interface_name = "praat"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="praat-resynth")
        self.signals=WorkSignals()
        self.signals.ready.connect(self.on_render_ready)
        self.timer=QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(140)
        self.timer.timeout.connect(self.start_render)
        self.session=0
        self.revision=0
        self.cache_revision=-1
        self.cached_stereo: np.ndarray | None=None
        self.raw_stereo: np.ndarray | None=None
        self.render_inflight=False
        self.pending_play: PlaybackRequest | None=None
        self.pending_export: str | None=None
        self.cached_result_f0: tuple[np.ndarray, np.ndarray] | None = None
        self.undo_stack: list[tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]] = []
        self.redo_stack: list[tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]] = []
        self._build_ui()
        self._shortcuts()
        if audio:
            QTimer.singleShot(0,lambda: self.open_project(audio) if audio.lower().endswith(".json") else self.open_audio(audio))
        else:
            self.statusBar().showMessage("Open a WAV, FLAC, AIFF, or other soundfile-compatible recording.")

    def _build_ui(self):
        root=QWidget()
        scroll=QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout=QVBoxLayout(root)
        layout.setContentsMargins(14,10,14,10)
        layout.setSpacing(10)
        bar=QHBoxLayout()
        btn=lambda label, fn: self._add_button(bar,label,fn)
        btn("Open audio…", self.choose_audio)
        btn("Save project…", self.save_project)
        btn("Load project…",self.load_project)
        btn("Export stereo WAV…",self.export_stereo)
        btn("Export F0 CSV…",self.export_csv)
        bar.addStretch()
        self.filename=QLabel("No recording loaded")
        self.filename.setStyleSheet("font-weight: 600; color: #344960")
        bar.addWidget(self.filename)
        layout.addLayout(bar)

        tabs=QHBoxLayout()
        tabs.addWidget(QLabel("Synthesis interface:"))
        self.interface_tabs=QTabBar()
        self.interface_tabs.addTab("Praat")
        self.interface_tabs.addTab("PitchFlower")
        self.interface_tabs.currentChanged.connect(self.switch_interface)
        tabs.addWidget(self.interface_tabs)
        tabs.addStretch()
        layout.addLayout(tabs)

        analysis=QHBoxLayout()
        analysis.addWidget(QLabel("Pitch analysis: floor"))
        self.floor_spin=self._spin(75,50,500,1)
        analysis.addWidget(self.floor_spin)
        analysis.addWidget(QLabel("Hz    ceiling"))
        self.ceiling_spin=self._spin(600,100,1200,10)
        analysis.addWidget(self.ceiling_spin)
        analysis.addWidget(QLabel("Hz"))
        self._add_button(analysis,"Reanalyze (resets curves)",self.reanalyze)
        analysis.addSpacing(24)
        analysis.addWidget(QLabel("Edit contour:"))
        self.side_select=QComboBox()
        self.side_select.addItems(["Left / orange", "Right / blue"])
        self.side_select.currentIndexChanged.connect(lambda i:self.pitch.set_active(LEFT if i==0 else RIGHT))
        analysis.addWidget(self.side_select)
        analysis.addStretch()
        layout.addLayout(analysis)

        self.pitchflower_controls=QHBoxLayout()
        self.pitchflower_controls.addWidget(QLabel("PitchFlower flow steps"))
        self.flow_steps_spin=self._spin(100, 1, 500, 1)
        self.pitchflower_controls.addWidget(self.flow_steps_spin)
        self.pitchflower_controls.addWidget(QLabel("Guidance"))
        self.flow_guidance_spin=self._spin(2.0, 0.0, 20.0, 0.1, decimals=2)
        self.pitchflower_controls.addWidget(self.flow_guidance_spin)
        self.pitchflower_controls.addWidget(QLabel("Seed"))
        self.flow_seed_spin=self._spin(0, 0, 2_147_483_647, 1)
        self.pitchflower_controls.addWidget(self.flow_seed_spin)
        self._add_button(self.pitchflower_controls, "Run PitchFlower", self.run_pitchflower)
        self.pitchflower_controls.addWidget(QLabel("Results are measured with WORLD and overlaid as dashed pink/purple curves."))
        self.pitchflower_controls.addStretch()
        layout.addLayout(self.pitchflower_controls)
        self._set_pitchflower_controls_visible(False)
        for control in (self.flow_steps_spin, self.flow_guidance_spin, self.flow_seed_spin):
            control.valueChanged.connect(self.pitchflower_settings_changed)

        self.pitch=PitchCanvas()
        self.pitch.on_begin_edit=self._snapshot
        self.pitch.change_committed.connect(self.commit)
        self.pitch.selection_changed.connect(self._sync_selected)
        self.pitch.active_changed.connect(self._on_active)
        layout.addWidget(self.pitch,stretch=1)

        edit=QHBoxLayout()
        self._add_button(edit,"Undo  Ctrl+Z",self.undo)
        self._add_button(edit,"Redo  Ctrl+Y",self.redo)
        self._add_button(edit,"Delete point  Del",self.remove_selected)
        self._add_button(edit,"Reset active",self.reset_active)
        self._add_button(edit,"Copy active → other",self.copy_active)
        edit.addSpacing(14)
        edit.addWidget(QLabel("Simplify tolerance (st)"))
        self.simplify_tolerance=self._spin(0.35,0.05,4.0,0.05,decimals=2)
        edit.addWidget(self.simplify_tolerance)
        self._add_button(edit,"Simplify active",self.simplify)
        edit.addStretch()
        layout.addLayout(edit)

        regen=QHBoxLayout()
        regen.addWidget(QLabel("Editable point spacing"))
        self.spacing_spin=self._spin(20, 1, 500, 5, decimals=1)
        self.spacing_spin.setSuffix(" ms")
        self.spacing_spin.setToolTip("Rebuild points from the CURRENT edited contour (linear interpolation), not original F0 analysis. Keeps voiced-region gaps.")
        regen.addWidget(self.spacing_spin)
        self._add_button(regen,"Regenerate LEFT ▼",lambda: self.regenerate_points(LEFT))
        self._add_button(regen,"Regenerate RIGHT ▲",lambda: self.regenerate_points(RIGHT))
        self._add_button(regen,"Regenerate BOTH",lambda: self.regenerate_points("both"))
        regen.addStretch()
        layout.addLayout(regen)

        precision=QHBoxLayout()
        self.selected_label=QLabel("Selected point: none")
        precision.addWidget(self.selected_label)
        precision.addSpacing(14)
        precision.addWidget(QLabel("Time (s)"))
        self.time_spin=self._spin(0,0,36000,0.001,decimals=4)
        precision.addWidget(self.time_spin)
        precision.addWidget(QLabel("Pitch (Hz)"))
        self.f0_spin=self._spin(0,25,1500,1,decimals=2)
        precision.addWidget(self.f0_spin)
        self._add_button(precision,"Apply values",self.apply_values)
        precision.addStretch()
        layout.addLayout(precision)
        self._sync_selected()

        insertion=QHBoxLayout()
        insertion.addWidget(QLabel("New point at time"))
        self.insert_time_spin=self._spin(0.5, 0, 36000, 0.01, decimals=4)
        self.insert_time_spin.setSuffix(" s")
        insertion.addWidget(self.insert_time_spin)
        insertion.addWidget(QLabel("F0"))
        self.insert_f0_spin=self._spin(200, 25, 1500, 1, decimals=2)
        self.insert_f0_spin.setSuffix(" Hz")
        insertion.addWidget(self.insert_f0_spin)
        self._add_button(insertion,"+ Add LEFT ▼",lambda: self.insert_point(LEFT))
        self._add_button(insertion,"+ Add RIGHT ▲",lambda: self.insert_point(RIGHT))
        insertion.addStretch()
        layout.addLayout(insertion)

        layout.addWidget(QLabel("Drag on waveform to select a time region; click the PLAY ruler below to listen from that point."))
        self.timeline=WaveTimeline()
        self.timeline.play_requested.connect(self.timeline_play)
        layout.addWidget(self.timeline)
        selectbar=QHBoxLayout()
        self.use_selection=QCheckBox("Use selected region for playback")
        self.timeline.region_changed.connect(lambda selection: self.use_selection.setChecked(selection is not None))
        selectbar.addWidget(self.use_selection)
        self._add_button(selectbar,"Clear region",self.clear_region)
        selectbar.addStretch()
        selectbar.addWidget(QLabel("Ruler: click = stereo | Shift = left | Ctrl = right | Alt = original | Ctrl+Shift = swap"))
        layout.addLayout(selectbar)

        playback=QHBoxLayout()
        self._add_button(playback,"▶ Stereo   Space",lambda:self.play_mode("stereo"))
        self._add_button(playback,"Left only   L",lambda:self.play_mode("left"))
        self._add_button(playback,"Right only   R",lambda:self.play_mode("right"))
        self._add_button(playback,"Original   O",lambda:self.play_mode("original"))
        self._add_button(playback,"Swap   S",lambda:self.play_mode("swap"))
        self._add_button(playback,"■ Stop   Esc",self.stop_audio)
        playback.addStretch()
        layout.addLayout(playback)
        noise_controls=QHBoxLayout()
        noise_controls.addWidget(QLabel("White noise level"))
        self.noise_slider=QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setSingleStep(1)
        self.noise_slider.setPageStep(10)
        self.noise_slider.setTickInterval(10)
        self.noise_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.noise_slider.setValue(0)
        self.noise_slider.setToolTip("Playback white-noise RMS as a percentage of the original recording RMS. Each active ear gets independent noise.")
        self.noise_slider.valueChanged.connect(self._update_noise_label)
        noise_controls.addWidget(self.noise_slider, stretch=1)
        self.noise_label=QLabel("Off (0%)")
        self.noise_label.setMinimumWidth(130)
        noise_controls.addWidget(self.noise_label)
        self.noise_export_check=QCheckBox("Include noise in exported WAV")
        self.noise_export_check.setToolTip("Off by default: white noise is used for auditioning, not permanently mixed into exports.")
        noise_controls.addWidget(self.noise_export_check)
        layout.addLayout(noise_controls)

        level_controls=QHBoxLayout()
        self.level_match_check=QCheckBox("Match generated channel RMS to source")
        self.level_match_check.setChecked(True)
        level_controls.addWidget(self.level_match_check)
        level_controls.addWidget(QLabel("Output gain"))
        self.output_gain_spin=self._spin(0.0,-24.0,12.0,0.5,decimals=1)
        self.output_gain_spin.setSuffix(" dB")
        level_controls.addWidget(self.output_gain_spin)
        level_controls.addWidget(QLabel("Peak ceiling"))
        self.peak_ceiling_spin=self._spin(-1.0,-24.0,0.0,0.5,decimals=1)
        self.peak_ceiling_spin.setSuffix(" dBFS")
        level_controls.addWidget(self.peak_ceiling_spin)
        level_controls.addWidget(QLabel("RMS envelopes appear over the waveform after rendering."))
        level_controls.addStretch()
        layout.addLayout(level_controls)
        self.level_match_check.toggled.connect(self._output_levels_changed)
        self.output_gain_spin.valueChanged.connect(self._output_levels_changed)
        self.peak_ceiling_spin.valueChanged.connect(self._output_levels_changed)

        hint=QLabel("Orange ▼ = LEFT F0    •    Blue ▲ = RIGHT F0    •    Gray dashed = untouched original reference. "
                    "Double-click to add to active track; Shift+double-click adds LEFT and Ctrl+double-click adds RIGHT. "
                    "Regenerate samples the CURRENT edited curves, not original analysis. "
                    "White noise affects playback (optional for WAV export).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #617086; font-size: 12px")
        layout.addWidget(hint)
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _add_button(row:QHBoxLayout,label:str,handler:Callable):
        button=QPushButton(label)
        button.clicked.connect(handler)
        row.addWidget(button)
        return button

    @staticmethod
    def _spin(value,minimum,maximum,step,decimals=0):
        spin=QDoubleSpinBox()
        spin.setRange(minimum,maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumWidth(83)
        return spin

    def _shortcuts(self):
        bindings={"Ctrl+Z":self.undo,"Ctrl+Y":self.redo,"Delete":self.remove_selected,
                  "Space":lambda:self.play_mode("stereo"),"L":lambda:self.play_mode("left"),
                  "R":lambda:self.play_mode("right"),"O":lambda:self.play_mode("original"),
                  "S":lambda:self.play_mode("swap"),"Esc":self.stop_audio}
        for sequence,callback in bindings.items():
            action=QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def _set_pitchflower_controls_visible(self, visible: bool):
        for index in range(self.pitchflower_controls.count()):
            item = self.pitchflower_controls.itemAt(index)
            if item.widget():
                item.widget().setVisible(visible)

    def _capture_interface_state(self):
        if not self.engine:
            return
        self.interface_states[self.interface_name] = {
            "left": self.pitch.left.copy(), "right": self.pitch.right.copy(),
            "left_breaks": tuple(self.pitch.segment_breaks[LEFT]),
            "right_breaks": tuple(self.pitch.segment_breaks[RIGHT]),
            "result": self.cached_result_f0,
            "audio": self.cached_stereo,
            "raw_audio": self.raw_stereo,
            "point_spacing_ms": float(self.spacing_spin.value()),
            "steps": int(self.flow_steps_spin.value()),
            "guidance": float(self.flow_guidance_spin.value()),
            "seed": int(self.flow_seed_spin.value()),
        }

    def pitchflower_settings_changed(self, _value):
        if self.interface_name != "pitchflower" or not self.engine:
            return
        self.revision += 1
        self.cache_revision = -1
        self.cached_result_f0 = None
        self.raw_stereo = None
        self.pitch.set_result(None, None)
        self.timeline.set_generated(None, self.engine.sr)
        self.interface_states["pitchflower"]["result"] = None
        self.statusBar().showMessage("PitchFlower settings changed; click Run PitchFlower to generate new audio.")

    def run_pitchflower(self):
        if self.interface_name != "pitchflower" or not self.engine:
            return
        self.revision += 1
        self.cache_revision = -1
        self.cached_stereo = None
        self.raw_stereo = None
        self.cached_result_f0 = None
        self.pitch.set_result(None, None)
        self.timeline.set_generated(None, self.engine.sr)
        self.interface_states["pitchflower"]["result"] = None
        self.timer.stop()
        self.start_render()
        self.statusBar().showMessage("Running independent left and right PitchFlower synthesis…")

    def switch_interface(self, index: int):
        name = "praat" if index == 0 else "pitchflower"
        if name == self.interface_name or not self.engines:
            return
        if name not in self.engines:
            QMessageBox.warning(self, "PitchFlower unavailable", self.interface_errors.get(name, "PitchFlower is unavailable."))
            self.interface_tabs.blockSignals(True)
            self.interface_tabs.setCurrentIndex(0)
            self.interface_tabs.blockSignals(False)
            return
        self._capture_interface_state()
        self.stop_audio()
        self.interface_name = name
        self.engine = self.engines[name]
        state = self.interface_states[name]
        self.pitch.set_data(self.engine.initial_pitch, self.engine.duration,
                            self.engine.floor, self.engine.ceiling)
        self.pitch.set_points(LEFT, state["left"], state["left_breaks"])
        self.pitch.set_points(RIGHT, state["right"], state["right_breaks"])
        result = state.get("result")
        self.cached_result_f0 = result
        self.pitch.set_result(*(result if result else (None, None)))
        self.spacing_spin.setValue(state.get("point_spacing_ms", 20))
        self.flow_steps_spin.setValue(state.get("steps", 100))
        self.flow_guidance_spin.setValue(state.get("guidance", 2.0))
        self.flow_seed_spin.setValue(state.get("seed", 0))
        self.raw_stereo = state.get("raw_audio")
        self.cached_stereo = state.get("audio")
        self.cache_revision = -1
        # A pending render belongs to the previously selected interface; its
        # completion must not replace the newly selected interface's audio.
        self.revision += 1
        self.timeline.set_audio(self.engine.samples, self.engine.sr, state.get("audio"))
        self.filename.setText(f"{Path(self.engine.path).name}  |  {self.engine.duration:.2f}s  |  {name.title()}")
        self._set_pitchflower_controls_visible(name == "pitchflower")
        self.statusBar().showMessage(f"Switched to {name.title()}; its F0 curves are independent from the other interface.")

    def choose_audio(self):
        file,_=QFileDialog.getOpenFileName(self,"Open recording",str(Path.home()),
                                           "Audio (*.wav *.flac *.aif *.aiff *.ogg);;All files (*)")
        if file: self.open_audio(file)

    def open_audio(self,path:str, floor=None, ceiling=None):
        try:
            self.stop_audio()
            self.statusBar().showMessage("Analyzing pitch with Praat…")
            QApplication.processEvents()
            floor=float(floor if floor is not None else self.floor_spin.value())
            ceiling=float(ceiling if ceiling is not None else self.ceiling_spin.value())
            if floor>=ceiling:
                raise ValueError("Pitch ceiling must exceed pitch floor")
            new_engine=PraatEngine(path,floor,ceiling)
            new_engines: dict[str, object] = {"praat": new_engine}
            errors: dict[str, str] = {}
            try:
                new_engines["pitchflower"] = PitchFlowerEngine(path, floor, ceiling)
            except Exception as exc:
                # The editor stays usable for Praat if the optional neural
                # stack is missing or the source exceeds its verified limit.
                errors["pitchflower"] = str(exc)
            self.session+=1
            self.engines=new_engines
            self.interface_errors=errors
            self.interface_name="praat"
            self.engine=new_engine
            self.interface_states={}
            praat_breaks=infer_segment_breaks(new_engine.initial_pitch)
            self.interface_states["praat"]={
                "left":new_engine.initial_pitch.copy(), "right":new_engine.initial_pitch.copy(),
                "left_breaks":praat_breaks, "right_breaks":praat_breaks, "result":None, "audio":None,
                "point_spacing_ms":20.0, "steps":100, "guidance":2.0, "seed":0,
            }
            if "pitchflower" in new_engines:
                pf=new_engines["pitchflower"]
                pf_breaks=infer_segment_breaks(pf.initial_pitch)
                self.interface_states["pitchflower"]={
                    "left":pf.initial_pitch.copy(), "right":pf.initial_pitch.copy(),
                    "left_breaks":pf_breaks, "right_breaks":pf_breaks, "result":None, "audio":None,
                    "point_spacing_ms":20.0, "steps":100, "guidance":2.0, "seed":0,
                }
            self.revision=0
            self.cached_stereo=None
            self.raw_stereo=None
            self.cached_result_f0=None
            self.cache_revision=-1
            self.render_inflight=False  # stale jobs are ignored by session ID
            self.pending_play=None
            self.pending_export=None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.pitch.set_data(new_engine.initial_pitch,new_engine.duration,floor,ceiling)
            self.pitch.set_result(None, None)
            self.timeline.set_audio(new_engine.samples,new_engine.sr)
            self.insert_time_spin.setRange(0.0, new_engine.duration)
            self.insert_time_spin.setValue(new_engine.duration/2)
            if len(new_engine.initial_pitch):
                self.insert_f0_spin.setValue(float(np.median(new_engine.initial_pitch[:, 1])))
            self.use_selection.setChecked(False)
            self.floor_spin.setValue(floor)
            self.ceiling_spin.setValue(ceiling)
            self.filename.setText(f"{Path(path).name}  |  {new_engine.duration:.2f}s")
            self.setWindowTitle(f"Dichotic Pitch Editor — {Path(path).name}")
            self.statusBar().showMessage(f"Detected {len(new_engine.initial_pitch)} Praat F0 points. "
                "Select a contour, drag points, then click the bottom playback ruler.")
            self.start_render()
            self.interface_tabs.blockSignals(True)
            self.interface_tabs.setCurrentIndex(0)
            self.interface_tabs.blockSignals(False)
            self._set_pitchflower_controls_visible(False)
            if len(new_engine.initial_pitch)==0:
                QMessageBox.warning(self,"No detected F0","Praat found no voiced pitch points. "
                    "Try changing the pitch analysis range. Adding points alone cannot make unvoiced audio voiced.")
        except Exception as exc:
            self._error("Could not open recording",exc)

    def reanalyze(self):
        if not self.engine:return
        if self.undo_stack:
            result=QMessageBox.question(self,"Reset pitch edits?",
                "Reanalysis replaces both edited contours. Continue?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
            if result != QMessageBox.StandardButton.Yes:return
        self.open_audio(self.engine.path, float(self.floor_spin.value()), float(self.ceiling_spin.value()))

    def _pitch_state(self):
        return (self.pitch.left.copy(), self.pitch.right.copy(),
                tuple(self.pitch.segment_breaks[LEFT]), tuple(self.pitch.segment_breaks[RIGHT]))

    def _restore_pitch_state(self, state):
        left, right, left_breaks, right_breaks = state
        self.pitch.set_points(LEFT, left, left_breaks)
        self.pitch.set_points(RIGHT, right, right_breaks)

    def _snapshot(self):
        if not self.engine:return
        self.undo_stack.append(self._pitch_state())
        self.undo_stack=self.undo_stack[-60:]
        self.redo_stack.clear()

    def commit(self):
        if not self.engine:return
        self.revision+=1
        self.cache_revision=-1
        self.cached_result_f0=None
        self.raw_stereo=None
        self.timeline.set_generated(None, self.engine.sr)
        if self.interface_name in self.interface_states:
            self.interface_states[self.interface_name]["audio"] = None
            self.interface_states[self.interface_name]["raw_audio"] = None
        self.pitch.set_result(None, None)
        if self.interface_name in self.interface_states:
            self.interface_states[self.interface_name]["result"] = None
        self.timer.start()
        self.statusBar().showMessage(f"Pitch changed; preparing updated {self.interface_name.title()} stereo…")

    def undo(self):
        if not self.undo_stack or not self.engine:return
        self.redo_stack.append(self._pitch_state())
        self._restore_pitch_state(self.undo_stack.pop())
        self.commit()

    def redo(self):
        if not self.redo_stack or not self.engine:return
        self.undo_stack.append(self._pitch_state())
        self._restore_pitch_state(self.redo_stack.pop())
        self.commit()

    def _on_active(self,side:str):
        self.side_select.blockSignals(True)
        self.side_select.setCurrentIndex(0 if side==LEFT else 1)
        self.side_select.blockSignals(False)

    def _sync_selected(self):
        idx=self.pitch.selected_index
        arr=self.pitch.points()
        valid=idx is not None and idx<len(arr)
        self.selected_label.setText(
            f"Selected {self.pitch.active}: point #{idx+1}" if valid else "Selected point: none")
        for spin in (self.time_spin,self.f0_spin):
            spin.setEnabled(valid)
        if valid:
            self.time_spin.blockSignals(True)
            self.f0_spin.blockSignals(True)
            self.time_spin.setRange(0.0,self.pitch.duration)
            self.time_spin.setValue(float(arr[idx,0]))
            self.f0_spin.setValue(float(arr[idx,1]))
            self.time_spin.blockSignals(False)
            self.f0_spin.blockSignals(False)

    def apply_values(self):
        idx=self.pitch.selected_index
        arr=self.pitch.points()
        if not self.engine or idx is None or idx>=len(arr):return
        time_s=float(self.time_spin.value())
        value=float(self.f0_spin.value())
        epsilon=1e-4
        lo=float(arr[idx-1,0]+epsilon) if idx>0 else 0.0
        hi=float(arr[idx+1,0]-epsilon) if idx<len(arr)-1 else self.pitch.duration
        if time_s<lo or time_s>hi:
            QMessageBox.warning(self,"Invalid point time",
                f"Time must be between {lo:.4f} and {hi:.4f} s to preserve point order.")
            return
        self._snapshot()
        arr[idx]=[time_s,value]
        self.pitch.update()
        self.commit()

    def remove_selected(self):
        idx=self.pitch.selected_index
        arr=self.pitch.points()
        if not self.engine or idx is None or idx>=len(arr):return
        updated, breaks = delete_editable_point(arr, self.pitch.segment_breaks[self.pitch.active], idx)
        self._snapshot()
        self.pitch.set_points(self.pitch.active, updated, breaks)
        self.commit()

    def reset_active(self):
        if not self.engine:return
        self._snapshot()
        self.pitch.set_points(self.pitch.active, self.engine.initial_pitch,
                              infer_segment_breaks(self.engine.initial_pitch))
        self.commit()

    def copy_active(self):
        if not self.engine:return
        self._snapshot()
        other=RIGHT if self.pitch.active==LEFT else LEFT
        self.pitch.set_points(other, self.pitch.points().copy(),
                              self.pitch.segment_breaks[self.pitch.active])
        self.commit()

    def simplify(self):
        if not self.engine:return
        points=self.pitch.points()
        if len(points)<3:return
        reduced, breaks = simplify_segmented_pitch(
            points, self.pitch.segment_breaks[self.pitch.active],
            float(self.simplify_tolerance.value()))
        if len(reduced)==len(points):
            self.statusBar().showMessage("Contour is already within the chosen tolerance.")
            return
        self._snapshot()
        self.pitch.set_points(self.pitch.active, reduced, breaks)
        self.commit()
        self.statusBar().showMessage(f"Simplified {len(points)} → {len(reduced)} points on {self.pitch.active}.")

    def insert_point(self, side: str):
        if not self.engine:
            return
        t = float(self.insert_time_spin.value())
        f0 = float(self.insert_f0_spin.value())
        try:
            points, breaks, idx = insert_editable_point(
                self.pitch.points(side), self.pitch.segment_breaks[side], t, f0)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot insert pitch point", str(exc))
            return
        self._snapshot()
        self.pitch.set_active(side)
        self.pitch.set_points(side, points, breaks)
        self.pitch.selected_index = idx
        self.pitch.selection_changed.emit()
        self.commit()
        self.statusBar().showMessage(f"Added {side.upper()} point at {t:.4f}s, {f0:.1f} Hz.")

    def regenerate_points(self, side: str):
        if not self.engine:
            return
        sides = (LEFT, RIGHT) if side == "both" else (side,)
        spacing_s = float(self.spacing_spin.value()) / 1000
        try:
            regenerated = {
                target: resample_editable_pitch(
                    self.pitch.points(target), spacing_s, self.pitch.segment_breaks[target])
                for target in sides
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot regenerate points", str(exc))
            return
        summaries = [f"{target}: {len(self.pitch.points(target))} → {len(regenerated[target][0])}"
                     for target in sides]
        self._snapshot()
        for target in sides:
            points, breaks = regenerated[target]
            self.pitch.set_points(target, points, breaks)
        self.commit()
        self.statusBar().showMessage(
            f"Regenerated from edited F0 at {spacing_s * 1000:g} ms; "
            + ", ".join(summaries) + ". Undo restores the previous points.")

    def _update_noise_label(self, level: int):
        if level == 0:
            self.noise_label.setText("Off (0%)")
        else:
            self.noise_label.setText(f"{level}% of original RMS")

    def start_render(self):
        if self.engine is None or self.render_inflight:
            return
        if self.cached_stereo is not None and self.cache_revision==self.revision:
            return
        engine=self.engine
        session=self.session
        revision=self.revision
        left=self.pitch.left.copy()
        right=self.pitch.right.copy()
        self.render_inflight=True
        if self.interface_name == "pitchflower":
            future=self.executor.submit(
                engine.stereo, left, right,
                steps=int(self.flow_steps_spin.value()), guidance=float(self.flow_guidance_spin.value()),
                seed=int(self.flow_seed_spin.value()),
            )
        else:
            future=self.executor.submit(engine.stereo,left,right)
        def done(future):
            try:
                result=future.result()
                error=None
            except Exception as exc:
                result=None
                error=f"{type(exc).__name__}: {exc}"
            self.signals.ready.emit(session,revision,result,error)
        future.add_done_callback(done)

    def _output_levels_changed(self, _value=None):
        if self.raw_stereo is None or self.engine is None:
            return
        self._refresh_output_audio()

    def _refresh_output_audio(self):
        if self.raw_stereo is None or self.engine is None:
            return
        self.cached_stereo = normalize_stereo_levels(
            self.raw_stereo,
            self.engine.samples,
            match_rms=self.level_match_check.isChecked(),
            gain_db=float(self.output_gain_spin.value()),
            peak_dbfs=float(self.peak_ceiling_spin.value()),
        )
        if self.interface_name in self.interface_states:
            self.interface_states[self.interface_name]["audio"] = self.cached_stereo
            self.interface_states[self.interface_name]["raw_audio"] = self.raw_stereo
        self.timeline.set_generated(self.cached_stereo, self.engine.sr)

    def on_render_ready(self,session:int,revision:int,audio, error):
        if session!=self.session:return
        self.render_inflight=False
        if error:
            self.pending_play=None
            self.pending_export=None
            QMessageBox.critical(self, f"{self.interface_name.title()} resynthesis failed", str(error))
            return
        if revision!=self.revision:
            # Last edit wins. Don't play stale stereo after a new drag.
            self.start_render()
            return
        if isinstance(audio, PitchFlowerResult):
            self.raw_stereo=audio.stereo
            self.cached_result_f0=(audio.left_result_f0, audio.right_result_f0)
            self.pitch.set_result(*self.cached_result_f0)
            self.interface_states[self.interface_name]["result"] = self.cached_result_f0
        else:
            self.raw_stereo=audio
            self.cached_result_f0=None
            self.pitch.set_result(None, None)
        self._refresh_output_audio()
        self.cache_revision=revision
        self.statusBar().showMessage(
            f"{self.interface_name.title()} stereo ready: left (orange) + right (blue), {self.engine.sr} Hz")
        if self.pending_export:
            filename=self.pending_export
            self.pending_export=None
            self._write_wav(filename)
        if self.pending_play:
            request=self.pending_play
            self.pending_play=None
            self.play(request)

    def timeline_play(self,time_s:float,mode:str):
        self.play_mode(mode,start=time_s)

    def play_mode(self,mode:str,start:float=0):
        if not self.engine:return
        if self.use_selection.isChecked() and self.timeline.selection:
            start,end=self.timeline.selection
        else:
            end=self.engine.duration
        self.play(PlaybackRequest(float(start),float(end),mode))

    def play(self,request:PlaybackRequest):
        if self.engine is None:return
        if request.end<=request.start:
            return
        if request.mode!="original" and (self.cache_revision!=self.revision or self.cached_stereo is None):
            self.pending_play=request
            self.start_render()
            self.statusBar().showMessage("Rendering edited F0 tracks; stereo playback will start automatically…")
            return
        sr=self.engine.sr
        a=max(0,int(round(request.start*sr)))
        b=min(len(self.engine.samples),int(round(request.end*sr)))
        if b<=a:return
        if request.mode=="original":
            original=self.engine.samples[a:b].astype(np.float32)
            stereo=np.column_stack((original,original))
        else:
            source=self.cached_stereo[a:b]
            if request.mode=="stereo":stereo=source
            elif request.mode=="left":stereo=np.column_stack((source[:,0],np.zeros(b-a,dtype=np.float32)))
            elif request.mode=="right":stereo=np.column_stack((np.zeros(b-a,dtype=np.float32),source[:,1]))
            elif request.mode=="swap":stereo=source[:,::-1].copy()
            else:raise ValueError(f"Unknown mode: {request.mode}")
        # White noise is mixed at playback, not cached with the Praat synthesis;
        # changing the slider therefore does not trigger expensive overlap-add.
        if self.noise_slider.value():
            active = ((True, False) if request.mode == "left" else
                      (False, True) if request.mode == "right" else (True, True))
            reference_rms = float(np.sqrt(np.mean(self.engine.samples ** 2)))
            stereo = mix_white_noise(stereo, self.noise_slider.value(), reference_rms,
                                     active_channels=active, start_sample=a)
        # sd.play is nonblocking. Praat synthesis never runs in the audio callback.
        try:
            sd.stop()
            sd.play(stereo, sr, blocking=False)
            self.timeline.cursor=request.start
            self.timeline.update()
            self.statusBar().showMessage(f"Playing {request.mode} | {request.start:.2f}–{request.end:.2f}s")
        except Exception as exc:
            self._error("Audio playback failed",exc)

    def stop_audio(self):
        sd.stop()
        self.pending_play=None

    def clear_region(self):
        self.timeline.selection=None
        self.use_selection.setChecked(False)
        self.timeline.update()

    def export_stereo(self):
        if not self.engine:return
        suggested=Path(self.engine.path).with_name(Path(self.engine.path).stem+"_dichotic.wav")
        file,_=QFileDialog.getSaveFileName(self,"Export stereo WAV",str(suggested),"WAV (*.wav)")
        if not file:return
        self.pending_export=file
        if self.cached_stereo is None or self.cache_revision!=self.revision:
            self.start_render()
        else:
            self.pending_export=None
            self._write_wav(file)

    def _write_wav(self,file:str):
        try:
            assert self.engine is not None and self.cached_stereo is not None
            audio = self.cached_stereo
            if self.noise_export_check.isChecked() and self.noise_slider.value():
                reference_rms = float(np.sqrt(np.mean(self.engine.samples ** 2)))
                audio = mix_white_noise(audio, self.noise_slider.value(), reference_rms)
            sf.write(file, audio, self.engine.sr, subtype="PCM_24")
            suffix = " (with white noise)" if self.noise_export_check.isChecked() and self.noise_slider.value() else " (clean)"
            self.statusBar().showMessage(f"Exported stereo WAV{suffix}: {file}")
        except Exception as exc:self._error("Could not write WAV",exc)

    def export_csv(self):
        if not self.engine:return
        suggested=Path(self.engine.path).with_name(Path(self.engine.path).stem+"_F0.csv")
        file,_=QFileDialog.getSaveFileName(self,"Export pitch points CSV",str(suggested),"CSV (*.csv)")
        if not file:return
        try:
            with open(file,"w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f)
                writer.writerow(["channel","time_s","f0_hz"])
                for side in (LEFT,RIGHT):
                    for t,hz in self.pitch.points(side):
                        writer.writerow([side,f"{t:.9f}",f"{hz:.9f}"])
            self.statusBar().showMessage(f"Exported F0 CSV: {file}")
        except Exception as exc:self._error("Could not write CSV",exc)

    def save_project(self):
        if not self.engine:return
        suggested=Path(self.engine.path).with_name(Path(self.engine.path).stem+"_dichotic.json")
        file,_=QFileDialog.getSaveFileName(self,"Save editing project",str(suggested),"JSON (*.json)")
        if not file:return
        try:
            self._capture_interface_state()
            interfaces={}
            for name, state in self.interface_states.items():
                engine=self.engines[name]
                result=state.get("result")
                interfaces[name]={
                    "version":1,
                    "left_pitch_points":state["left"].tolist(),
                    "right_pitch_points":state["right"].tolist(),
                    "left_segment_starts":list(state["left_breaks"]),
                    "right_segment_starts":list(state["right_breaks"]),
                    "point_spacing_ms":state.get("point_spacing_ms",20.0),
                    "settings":({"pitch_floor_hz":engine.floor,"pitch_ceiling_hz":engine.ceiling}
                                 if name=="praat" else
                                 {"flow_steps":state.get("steps",100),"guidance":state.get("guidance",2.0),"seed":state.get("seed",0)}),
                    "result":None if result is None else {
                        "left_measured_f0":result[0].tolist(), "right_measured_f0":result[1].tolist(),
                    },
                }
            data=new_project(
                self.engine.path, file,
                shared={"active_interface":self.interface_name, "active_contour":self.pitch.active,
                        "white_noise_percent":int(self.noise_slider.value()),
                        "white_noise_in_export":bool(self.noise_export_check.isChecked()),
                        "match_output_rms":bool(self.level_match_check.isChecked()),
                        "output_gain_db":float(self.output_gain_spin.value()),
                        "peak_ceiling_dbfs":float(self.peak_ceiling_spin.value()),
                        "selection":self.timeline.selection},
                engines=interfaces,
            )
            Path(file).write_text(json.dumps(data,indent=2),encoding="utf-8")
            self.statusBar().showMessage(f"Saved project: {file}")
        except Exception as exc:self._error("Could not save project",exc)

    def load_project(self):
        file,_=QFileDialog.getOpenFileName(self,"Load editing project",str(Path.home()),"JSON (*.json)")
        if file:self.open_project(file)

    def open_project(self,file:str):
        try:
            data=migrate_legacy(json.loads(Path(file).read_text(encoding="utf-8")))
            if data.get("format")!=PROJECT_FORMAT:
                raise ValueError("Not a supported Dichotic Editor project")
            path=resolve_audio_path(file,data)
            praat_data=data.get("interfaces",{}).get("praat") or {}
            settings=praat_data.get("settings",praat_data)
            self.open_audio(path,settings.get("pitch_floor_hz",75),settings.get("pitch_ceiling_hz",600))
            if self.engine is None:return
            for name, payload in data.get("interfaces",{}).items():
                if not payload or name not in self.engines:
                    continue
                target=self.engines[name]
                left=checked_points(np.array(payload["left_pitch_points"],dtype=float),0,target.duration)
                right=checked_points(np.array(payload["right_pitch_points"],dtype=float),0,target.duration)
                measured=payload.get("result")
                result=None if not measured else (
                    np.asarray(measured.get("left_measured_f0",[]),dtype=float).reshape((-1,2)),
                    np.asarray(measured.get("right_measured_f0",[]),dtype=float).reshape((-1,2)),
                )
                ps=payload.get("settings",{})
                self.interface_states[name].update({
                    "left":left,"right":right,
                    "left_breaks":tuple(payload.get("left_segment_starts",infer_segment_breaks(left))),
                    "right_breaks":tuple(payload.get("right_segment_starts",infer_segment_breaks(right))),
                    "result":result,"point_spacing_ms":float(payload.get("point_spacing_ms",20)),
                    "steps":int(ps.get("flow_steps",100)),"guidance":float(ps.get("guidance",2.0)),
                    "seed":int(ps.get("seed",0)),
                })
            shared=data.get("shared",{})
            self.noise_slider.setValue(int(shared.get("white_noise_percent", 0)))
            self.noise_export_check.setChecked(bool(shared.get("white_noise_in_export", False)))
            self.level_match_check.setChecked(bool(shared.get("match_output_rms", True)))
            self.output_gain_spin.setValue(float(shared.get("output_gain_db", 0.0)))
            self.peak_ceiling_spin.setValue(float(shared.get("peak_ceiling_dbfs", -1.0)))
            selection=shared.get("selection")
            if selection:
                a,b=map(float,selection)
                if 0<=a<b<=self.engine.duration:
                    self.timeline.selection=(a,b)
                    self.use_selection.setChecked(True)
            active=shared.get("active_interface","praat")
            self.interface_tabs.setCurrentIndex(0 if active=="praat" else 1)
            if active=="praat":
                state=self.interface_states["praat"]
                self.pitch.set_points(LEFT,state["left"],state["left_breaks"])
                self.pitch.set_points(RIGHT,state["right"],state["right_breaks"])
                self.cached_result_f0=state.get("result")
                self.pitch.set_result(*(self.cached_result_f0 or (None,None)))
            self.pitch.set_active(shared.get("active_contour",LEFT))
            # Saved result F0 remains visible as provenance. Audio itself is
            # intentionally not embedded and will be re-rendered on playback.
            self.revision += 1
            self.cache_revision = -1
            self.statusBar().showMessage(f"Loaded project: {file}")
        except Exception as exc:self._error("Could not load project",exc)

    def _error(self,title:str,exc:Exception):
        QMessageBox.critical(self,title,str(exc))
        self.statusBar().showMessage(f"{title}: {exc}")

    def closeEvent(self,event):
        self.timer.stop()
        sd.stop()
        self.executor.shutdown(wait=False,cancel_futures=True)
        event.accept()


def main() -> None:
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    if sys.platform.startswith("linux"):
        # Allow stereo devices selected by the operating system to be used.
        pass
    window=Editor(sys.argv[1] if len(sys.argv)>1 else None)
    window.show()
    sys.exit(app.exec())


if __name__=="__main__":
    main()
