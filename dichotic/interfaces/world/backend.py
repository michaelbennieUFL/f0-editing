"""PyWORLD adapter for shared-spectrum dichotic pitch synthesis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from praat_backend import fit_audio

SAMPLE_RATE = 24_000
FRAME_PERIOD_MS = 5.0
F0_FLOOR = 50.0
F0_CEILING = 600.0


def _resample_to_24khz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if int(sample_rate) == SAMPLE_RATE:
        return np.ascontiguousarray(signal)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("WORLD requires scipy for 24 kHz resampling. Install the WORLD extra.") from exc
    return np.ascontiguousarray(
        resample_poly(signal, SAMPLE_RATE, int(sample_rate)).astype(np.float64)
    )


def _load_pyworld():
    try:
        import pyworld as pw
    except ImportError as exc:
        raise RuntimeError("WORLD dependencies are unavailable. Run `uv sync --extra world`.") from exc
    return pw


def _points_from_f0(f0: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(f0, dtype=float).reshape(-1)
    frame_times = np.asarray(times, dtype=float).reshape(-1)
    if len(values) != len(frame_times):
        raise ValueError("WORLD F0 and time arrays must have the same length")
    voiced = np.isfinite(values) & (values > 0)
    return np.column_stack((frame_times[voiced], values[voiced]))


def target_f0_from_points(
    source_f0: np.ndarray,
    frame_times: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply drawn points while retaining the source voicing mask.

    Interpolation is performed separately for each source-voiced island. A
    point placed in a source-unvoiced gap therefore cannot create voiced audio.
    """
    source = np.asarray(source_f0, dtype=float).reshape(-1)
    times = np.asarray(frame_times, dtype=float).reshape(-1)
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if len(source) != len(times):
        raise ValueError("Source F0 and frame times must have the same length")
    if len(pts) and (not np.isfinite(pts).all() or np.any(pts[:, 1] <= 0)):
        raise ValueError("Pitch points must be finite and positive")
    if len(pts) and np.any(np.diff(pts[:, 0]) <= 0):
        raise ValueError("Pitch point times must be strictly increasing")

    target = np.zeros_like(source, dtype=float)
    voiced = np.isfinite(source) & (source > 0)
    if not np.any(voiced):
        return target
    if not len(pts):
        target[voiced] = source[voiced]
        return target

    voiced_indices = np.flatnonzero(voiced)
    breaks = np.flatnonzero(np.diff(voiced_indices) > 1) + 1
    for island in np.split(voiced_indices, breaks):
        start_time, end_time = times[island[0]], times[island[-1]]
        island_points = pts[(pts[:, 0] >= start_time) & (pts[:, 0] <= end_time)]
        if len(island_points):
            target[island] = np.interp(
                times[island], island_points[:, 0], island_points[:, 1],
                left=island_points[0, 1], right=island_points[-1, 1],
            )
        else:
            target[island] = source[island]
    return target


class WorldEngine:
    """Analyze one source once and synthesize two channels from shared spectra."""

    name = "world"

    def __init__(self, path: str, pitch_floor: float = F0_FLOOR, pitch_ceiling: float = F0_CEILING):
        if not 1 <= pitch_floor < pitch_ceiling:
            raise ValueError("WORLD pitch floor must be below its pitch ceiling")
        self._pw = _load_pyworld()
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
        if samples.shape[0] < 128:
            raise ValueError("Recording is too short")
        mono = samples.mean(axis=1)
        self.path = str(Path(path).resolve())
        self.samples = _resample_to_24khz(mono, int(rate))
        if len(self.samples) < 128:
            raise ValueError("Recording is too short after resampling")
        self.sr = SAMPLE_RATE
        self.duration = len(self.samples) / self.sr
        self.floor = float(pitch_floor)
        self.ceiling = float(pitch_ceiling)

        raw_f0, self.frame_times = self._pw.dio(
            self.samples, self.sr, frame_period=FRAME_PERIOD_MS,
            f0_floor=self.floor, f0_ceil=self.ceiling,
        )
        self.source_f0 = np.asarray(
            self._pw.stonemask(self.samples, raw_f0, self.frame_times, self.sr), dtype=float
        )
        self.initial_pitch = _points_from_f0(self.source_f0, self.frame_times)
        self.sp = np.ascontiguousarray(
            self._pw.cheaptrick(self.samples, self.source_f0, self.frame_times, self.sr),
            dtype=np.float64,
        )
        self.ap = np.ascontiguousarray(
            self._pw.d4c(self.samples, self.source_f0, self.frame_times, self.sr),
            dtype=np.float64,
        )

    def _target_f0(self, points: np.ndarray) -> np.ndarray:
        return target_f0_from_points(self.source_f0, self.frame_times, points)

    def resynthesize(self, points: np.ndarray) -> np.ndarray:
        target = self._target_f0(points)
        output = self._pw.synthesize(target, self.sp, self.ap, self.sr)
        return fit_audio(output, len(self.samples)).astype(np.float32)

    def stereo(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
        left = self.resynthesize(left_points)
        right = self.resynthesize(right_points)
        return np.column_stack((left, right)).astype(np.float32)
