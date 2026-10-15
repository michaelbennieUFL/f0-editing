"""Native Praat overlap-add engine and pitch-contour helpers."""
from __future__ import annotations

import os
import math

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

def checked_points(points: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Validate sorted PitchTier points before passing them to Praat."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Pitch points must have shape (N, 2)")
    if not np.isfinite(pts).all() or np.any(pts[:, 1] <= 0):
        raise ValueError("Pitch must be finite and positive")
    if np.any(np.diff(pts[:, 0]) <= 0):
        raise ValueError("Pitch point times must be strictly increasing")
    if pts[0, 0] < xmin - 1e-9 or pts[-1, 0] > xmax + 1e-9:
        raise ValueError("Pitch points are outside the sound")
    return pts.copy()


def simplify_pitch(points: np.ndarray, tolerance_st: float = 0.4, gap_s: float = 0.05) -> np.ndarray:
    """Semitone-domain RDP simplification within voiced islands only."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy()
    breaks = np.where(np.diff(pts[:, 0]) > gap_s)[0] + 1
    chunks = np.split(pts, breaks)
    kept: list[np.ndarray] = []
    for chunk in chunks:
        if len(chunk) <= 2:
            kept.append(chunk)
            continue
        times = chunk[:, 0]
        y = 12 * np.log2(chunk[:, 1])
        chosen = {0, len(chunk) - 1}
        stack = [(0, len(chunk) - 1)]
        while stack:
            i, j = stack.pop()
            if j - i <= 1:
                continue
            k_candidates = np.arange(i + 1, j)
            interpolated = y[i] + (y[j] - y[i]) * (times[k_candidates] - times[i]) / (times[j] - times[i])
            errors = np.abs(y[k_candidates] - interpolated)
            max_idx = int(np.argmax(errors))
            if errors[max_idx] > tolerance_st:
                k = int(k_candidates[max_idx])
                chosen.add(k)
                stack.extend([(i, k), (k, j)])
        kept.append(chunk[sorted(chosen)])
    return np.concatenate(kept, axis=0)


def fit_audio(data: np.ndarray, length: int) -> np.ndarray:
    """Praat can differ by a sample at edges; keep both channels sample-aligned."""
    audio = np.asarray(data, dtype=np.float64).reshape(-1)
    if audio.size < length:
        return np.pad(audio, (0, length - audio.size))
    return audio[:length]


class PraatEngine:
    """Owns exactly one Praat Manipulation, used serially from one worker."""

    def __init__(self, path: str, pitch_floor: float, pitch_ceiling: float):
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
        if samples.shape[0] < 128:
            raise ValueError("Recording is too short")
        if samples.shape[1] > 1:
            # Both tiers use the same source, so pitch analysis is coherent.
            samples = samples.mean(axis=1, keepdims=True)
        self.samples = np.ascontiguousarray(samples[:, 0])
        self.sr = int(rate)
        self.path = os.path.realpath(path)
        self.duration = len(self.samples) / self.sr
        self.floor = float(pitch_floor)
        self.ceiling = float(pitch_ceiling)
        self.sound = parselmouth.Sound(self.samples, sampling_frequency=float(self.sr))
        self.manipulation = call(self.sound, "To Manipulation", 0.01, self.floor, self.ceiling)
        self.initial_pitch = self.read_pitch()

    def read_pitch(self) -> np.ndarray:
        tier = call(self.manipulation, "Extract pitch tier")
        count = int(call(tier, "Get number of points"))
        if not count:
            return np.empty((0, 2), dtype=float)
        return np.array([
            [float(call(tier, "Get time from index", i)),
             float(call(tier, "Get value at index", i))]
            for i in range(1, count + 1)
        ], dtype=float)

    def resynthesize(self, points: np.ndarray) -> np.ndarray:
        pts = checked_points(points, 0.0, self.duration)
        tier = call("Create PitchTier", "modified", 0.0, self.duration)
        for time_s, frequency_hz in pts:
            call(tier, "Add point", float(time_s), float(frequency_hz))
        call([tier, self.manipulation], "Replace pitch tier")
        result = call(self.manipulation, "Get resynthesis (overlap-add)")
        return fit_audio(result.values[0], len(self.samples))

    def stereo(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
        left = self.resynthesize(left_points)
        right = self.resynthesize(right_points)
        # Exact channel order: 0 = left, 1 = right.
        return np.column_stack((left, right)).astype(np.float32)



# ----- Editable pitch-tier sampling (never modifies the analysis reference) -----
MAX_EDIT_POINTS = 20000


def infer_segment_breaks(points: np.ndarray, gap_s: float = 0.05) -> tuple[int, ...]:
    """Find voiced-region boundaries in an initially analyzed Praat PitchTier.

    Returns indices of the first point AFTER each gap. Keep these indices as
    explicit metadata when resampling: after changing the point spacing, a gap
    cannot reliably be inferred from time differences alone.
    """
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if not np.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("Gap threshold must be positive")
    return tuple(int(i) for i in np.where(np.diff(pts[:, 0]) > gap_s)[0] + 1)


def normalize_breaks(breaks, length: int) -> tuple[int, ...]:
    normalized = tuple(int(i) for i in breaks)
    if any(i <= 0 or i >= length for i in normalized) or list(normalized) != sorted(set(normalized)):
        raise ValueError("Invalid voiced-region break indices")
    return normalized


def split_at_breaks(points: np.ndarray, breaks=()) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if len(pts) == 0:
        return []
    return np.split(pts, normalize_breaks(breaks, len(pts)))


def resample_editable_pitch(points: np.ndarray, spacing_s: float, breaks=(),
                            max_points: int = MAX_EDIT_POINTS) -> tuple[np.ndarray, tuple[int, ...]]:
    """Rebuild editable control points by interpolating CURRENT edits.

    It does not consult the initial analysis or redo pitch detection. Breaks
    preserve unvoiced gaps even when spacing exceeds the original 10-ms grid.
    Segment endpoints are always preserved exactly; interior positions form a
    spacing_s grid. Coarse spacing may necessarily discard short peaks.
    """
    if not np.isfinite(spacing_s) or spacing_s <= 0:
        raise ValueError("Spacing must be a finite positive number of seconds")
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if len(pts) and (not np.isfinite(pts).all() or np.any(pts[:, 1] <= 0)
                     or np.any(np.diff(pts[:, 0]) <= 0)):
        raise ValueError("Pitch points must be finite, positive and time-ordered")
    if len(pts) == 0:
        return pts.copy(), ()
    parts: list[np.ndarray] = []
    new_breaks: list[int] = []
    count = 0
    for segment in split_at_breaks(pts, breaks):
        first, last = float(segment[0, 0]), float(segment[-1, 0])
        if len(segment) == 1:
            times = np.array([first])
        else:
            # Avoid generating huge arrays until we've checked the allocation.
            regular_count = int(np.floor((last - first) / spacing_s)) + 1
            if count + regular_count + 1 > max_points:
                raise ValueError(f"Spacing too fine: more than {max_points:,} edit points. Increase spacing.")
            times = first + np.arange(regular_count, dtype=float) * spacing_s
            if last - times[-1] > 1e-8:
                times = np.append(times, last)
            else:
                times[-1] = last
        if count + len(times) > max_points:
            raise ValueError(f"Spacing too fine: more than {max_points:,} edit points. Increase spacing.")
        frequencies = np.interp(times, segment[:, 0], segment[:, 1])
        if count:
            new_breaks.append(count)
        parts.append(np.column_stack((times, frequencies)))
        count += len(times)
    return np.concatenate(parts), tuple(new_breaks)


def insert_editable_point(points: np.ndarray, breaks, time_s: float, pitch_hz: float,
                          min_separation_s: float = 0.0002) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Insert a point into a selected channel, returning updated gap boundaries.

    Inserting inside a known unvoiced gap creates an isolated point island rather
    than connecting it to either adjacent voiced phrase.
    """
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    existing_breaks = normalize_breaks(breaks, len(pts)) if len(pts) else ()
    if not np.isfinite(time_s) or not np.isfinite(pitch_hz) or pitch_hz <= 0:
        raise ValueError("New point needs a finite time and positive F0")
    if len(pts) and np.any(np.abs(pts[:, 0] - time_s) < min_separation_s):
        raise ValueError("A pitch point already exists at approximately this time")
    index = int(np.searchsorted(pts[:, 0], time_s))
    updated = np.insert(pts, index, [time_s, pitch_hz], axis=0)
    new_breaks = []
    for boundary in existing_breaks:
        if index == boundary:
            new_breaks.extend((index, index + 1))
        else:
            new_breaks.append(boundary + (index < boundary))
    new_breaks = tuple(i for i in sorted(set(new_breaks)) if 0 < i < len(updated))
    return updated, new_breaks, index


def delete_editable_point(points: np.ndarray, breaks, index: int) -> tuple[np.ndarray, tuple[int, ...]]:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if not 0 <= index < len(pts):
        raise IndexError("Pitch point index out of range")
    existing_breaks = normalize_breaks(breaks, len(pts))
    updated = np.delete(pts, index, axis=0)
    new_breaks = tuple(sorted(set(b - (b > index) for b in existing_breaks
                                  if 0 < b - (b > index) < len(updated))))
    return updated, new_breaks


def simplify_segmented_pitch(points: np.ndarray, breaks=(), tolerance_st: float = 0.4
                             ) -> tuple[np.ndarray, tuple[int, ...]]:
    segments = split_at_breaks(points, breaks)
    if not segments:
        return np.empty((0, 2), dtype=float), ()
    simplified = [simplify_pitch(seg, tolerance_st=tolerance_st, gap_s=float('inf'))
                  for seg in segments]
    new_breaks = []
    count = 0
    for chunk in simplified:
        if count:
            new_breaks.append(count)
        count += len(chunk)
    return np.concatenate(simplified), tuple(new_breaks)


def _shape_noise(noise: np.ndarray, sample_rate: int, noise_type: str) -> np.ndarray:
    """Apply a deterministic spectral profile while preserving time length."""
    kind = str(noise_type).strip().lower().replace("-", " ")
    aliases = {"white": "white", "white noise": "white",
               "pink": "pink", "pink noise": "pink",
               "stationary speech shaped": "speech",
               "stationary speech shaped noise": "speech",
               "speech shaped": "speech", "speech shaped noise": "speech",
               "speech": "speech"}
    kind = aliases.get(kind, kind)
    if kind == "white":
        return noise
    if kind not in {"pink", "speech"}:
        raise ValueError("Noise type must be White noise, Pink noise, or Stationary speech-shaped noise")
    count = len(noise)
    spectrum = np.fft.rfft(noise, axis=0)
    frequencies = np.fft.rfftfreq(count, 1.0 / sample_rate)
    if kind == "pink":
        weight = 1.0 / np.sqrt(np.maximum(frequencies, 1.0))
        weight[0] = weight[1] if len(weight) > 1 else 1.0
    else:
        # Smooth fixed long-term-average speech spectrum, in dB relative to
        # the mid-band peak. This remains stationary and needs no extra DSP
        # dependency.
        base_hz = np.array([0, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000], dtype=float)
        base_db = np.array([-12, -8, -4, 0, -3, -8, -12, -18, -24, -30], dtype=float)
        nyquist = sample_rate / 2.0
        keep = base_hz < nyquist
        profile_hz = np.concatenate((base_hz[keep], [nyquist]))
        profile_db = np.concatenate((base_db[keep], [base_db[keep][-1] - 6]))
        db = np.interp(frequencies, profile_hz, profile_db)
        weight = np.power(10.0, db / 20.0)
    shaped = np.fft.irfft(spectrum * weight[:, None], n=count, axis=0)
    return shaped.astype(np.float32)


def mix_white_noise(audio: np.ndarray, level_percent: float, reference_rms: float,
                    *, active_channels: tuple[bool, bool] = (True, True),
                    start_sample: int = 0, seed: int = 49273,
                    linked_channels: bool = False, noise_type: str = "White noise",
                    sample_rate: int = 24_000) -> np.ndarray:
    """Add uncorrelated white noise to active ears, for playback or optional export.

    100% noise means an RMS equal to the ORIGINAL source's RMS; 0% means off.
    The fixed seed makes repeated playback from the same time reproducible.
    Only the active ear gets noise in one-ear-only playback modes. Set
    ``linked_channels=True`` when stereo playback should use the same noise
    waveform in both ears.
    """
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Stereo audio must have shape (samples, 2)")
    if not np.isfinite(level_percent) or not 0 <= level_percent <= 100:
        raise ValueError("Noise percentage must lie between 0 and 100")
    if not np.isfinite(reference_rms) or reference_rms < 0:
        raise ValueError("Reference RMS must be finite and nonnegative")
    if level_percent == 0 or source.size == 0 or reference_rms == 0:
        return source.copy()
    gain = float(level_percent) / 100 * reference_rms
    # Seed by region start so a repeated click on the same region generates
    # exactly the same noise. By default left/right channels are statistically
    # independent; linked mode intentionally mirrors one noise signal to both.
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(start_sample)]))
    if linked_channels:
        shared = generator.standard_normal((source.shape[0], 1))
        white = np.repeat(_shape_noise(shared, sample_rate, noise_type), 2, axis=1)
    else:
        white = _shape_noise(generator.standard_normal(source.shape), sample_rate, noise_type)
    mask = np.array(active_channels, dtype=bool)
    for channel in range(2):
        if mask[channel]:
            rms = float(np.sqrt(np.mean(np.square(white[:, channel]))))
            white[:, channel] *= gain / max(rms, 1e-12)
        else:
            white[:, channel] = 0
    combined = source.astype(np.float64) + white
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        # Uniform attenuation avoids clipping without changing stereo balance.
        combined *= 0.99 / peak
    return combined.astype(np.float32)


def normalize_stereo_levels(audio: np.ndarray, reference: np.ndarray, *,
                            match_rms: bool = True, gain_db: float = 0.0,
                            peak_dbfs: float = -1.0, max_channel_gain_db: float = 12.0
                            ) -> np.ndarray:
    """Make generated channels comparable while preserving their F0 content.

    Each channel can be matched to the source RMS, but gain is bounded so a
    nearly silent/unstable render cannot be amplified without limit. A shared
    peak guard then controls high-level ringing and prevents clipping.
    """
    result = np.asarray(audio, dtype=np.float64).reshape((-1, 2)).copy()
    source = np.asarray(reference, dtype=np.float64).reshape(-1)
    if result.size == 0:
        return result.astype(np.float32)
    if not np.isfinite(gain_db) or not np.isfinite(peak_dbfs):
        raise ValueError("Output gain and peak ceiling must be finite")
    if peak_dbfs > 0:
        raise ValueError("Peak ceiling must be at or below 0 dBFS")
    if max_channel_gain_db < 0:
        raise ValueError("Maximum channel gain must be nonnegative")
    if match_rms:
        target = float(np.sqrt(np.mean(np.square(source)))) if source.size else 0.0
        if target > 1e-9:
            limit = 10 ** (float(max_channel_gain_db) / 20.0)
            for channel in range(2):
                current = float(np.sqrt(np.mean(np.square(result[:, channel]))))
                if current > 1e-9:
                    result[:, channel] *= np.clip(target / current, 1.0 / limit, limit)
    result *= 10 ** (float(gain_db) / 20.0)
    ceiling = 10 ** (float(peak_dbfs) / 20.0)
    peak = float(np.max(np.abs(result)))
    if peak > ceiling > 0:
        result *= ceiling / peak
    return np.clip(result, -ceiling, ceiling).astype(np.float32)
