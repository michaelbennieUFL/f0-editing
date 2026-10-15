"""Lazy ProMoNet adapter for independent dichotic pitch synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from praat_backend import fit_audio

SAMPLE_RATE = 22_050
HOPSIZE = 256
F0_FLOOR = 50.0
F0_CEILING = 550.0


def _points_from_pitch(pitch: np.ndarray) -> np.ndarray:
    """Convert a ProMoNet/Penn F0 tensor into time/F0 editor points."""
    values = np.asarray(pitch, dtype=float).reshape(-1)
    times = np.arange(len(values), dtype=float) * HOPSIZE / SAMPLE_RATE
    voiced = np.isfinite(values) & (values > 0)
    return np.column_stack((times[voiced], values[voiced]))


def _measure_f0(audio: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    """Measure output F0 with WORLD at the actual ProMoNet sample rate."""
    try:
        import pyworld as pw
    except ImportError as exc:
        raise RuntimeError("ProMoNet result measurement requires pyworld. Install the promonet extra.") from exc
    signal = np.asarray(audio, dtype=np.float64)
    raw, times = pw.harvest(signal, fs=SAMPLE_RATE, frame_period=HOPSIZE * 1000 / SAMPLE_RATE,
                            f0_floor=float(floor), f0_ceil=float(ceiling))
    f0 = pw.stonemask(signal, raw, times, fs=SAMPLE_RATE)
    return np.column_stack((times[f0 > 0], f0[f0 > 0]))


@dataclass(frozen=True)
class ProMoNetResult:
    stereo: np.ndarray
    left_result_f0: np.ndarray
    right_result_f0: np.ndarray


class ProMoNetEngine:
    """Preprocess one source once and synthesize each ear from its own F0 curve."""

    name = "promonet"

    def __init__(self, path: str, pitch_floor: float = F0_FLOOR, pitch_ceiling: float = F0_CEILING):
        if not 1 <= pitch_floor < pitch_ceiling:
            raise ValueError("ProMoNet pitch floor must be below its pitch ceiling")
        try:
            import promonet
            import torch
        except ImportError as exc:
            raise RuntimeError("ProMoNet dependencies are unavailable. Run `uv sync --extra promonet`.") from exc
        self._promonet = promonet
        self._torch = torch
        self.path = str(Path(path).resolve())
        self.floor = max(float(pitch_floor), F0_FLOOR)
        self.ceiling = min(float(pitch_ceiling), F0_CEILING)
        if self.floor >= self.ceiling:
            raise ValueError("ProMoNet supports an F0 range within 50–550 Hz")
        # Keep ProMoNet's required [channels, samples] shape for feature
        # extraction.  Its preprocessing uses ``audio.squeeze(1)`` and will
        # reject a flattened [samples] tensor.  Keep a separate mono 1-D
        # waveform for the editor timeline and playback.
        source_audio = promonet.load.audio(self.path)
        self.samples = np.asarray(source_audio.squeeze(0).detach().cpu(), dtype=np.float32).reshape(-1)
        self.sr = SAMPLE_RATE
        self.duration = len(self.samples) / self.sr
        self._gpu = 0 if torch.cuda.is_available() else None
        features = promonet.preprocess.from_audio(
            source_audio, self.sr, self._gpu,
            features=["loudness", "pitch", "periodicity", "ppg"],
        )
        self._features = tuple(features)
        self._source_pitch = np.asarray(features[1].detach().cpu(), dtype=float).reshape(-1)
        self.initial_pitch = _points_from_pitch(self._source_pitch)

    def _target_pitch(self, points: np.ndarray):
        source = self._features[1]
        frames = int(source.shape[-1])
        times = np.arange(frames, dtype=float) * HOPSIZE / SAMPLE_RATE
        pts = np.asarray(points, dtype=float).reshape((-1, 2))
        target = self._source_pitch.copy()
        if len(pts):
            requested = np.interp(times, pts[:, 0], pts[:, 1], left=pts[0, 1], right=pts[-1, 1])
            voiced = np.isfinite(target) & (target > 0)
            target[voiced] = np.clip(requested[voiced], F0_FLOOR, F0_CEILING)
        return self._torch.as_tensor(target, dtype=source.dtype, device=source.device).reshape_as(source)

    def _render_channel(self, points: np.ndarray, *, checkpoint: str | None, speaker: int,
                        spectral_balance: float) -> np.ndarray:
        loudness, _pitch, periodicity, ppg = self._features
        output = self._promonet.synthesize.from_features(
            loudness, self._target_pitch(points), periodicity, ppg,
            speaker=int(speaker), spectral_balance_ratio=float(spectral_balance),
            checkpoint=checkpoint or None, gpu=self._gpu,
        )
        return fit_audio(np.asarray(output.detach().cpu(), dtype=np.float32).reshape(-1), len(self.samples))

    def stereo(self, left_points: np.ndarray, right_points: np.ndarray, *, checkpoint: str | None = None,
               speaker: int = 0, spectral_balance: float = 1.0) -> ProMoNetResult:
        if not 0.25 <= spectral_balance <= 4.0:
            raise ValueError("ProMoNet spectral balance must be between 0.25 and 4.0")
        if checkpoint and not Path(checkpoint).expanduser().exists():
            raise ValueError(f"ProMoNet checkpoint does not exist: {checkpoint}")
        left = self._render_channel(left_points, checkpoint=checkpoint, speaker=speaker,
                                    spectral_balance=spectral_balance)
        right = self._render_channel(right_points, checkpoint=checkpoint, speaker=speaker,
                                     spectral_balance=spectral_balance)
        return ProMoNetResult(
            stereo=np.column_stack((left, right)).astype(np.float32),
            left_result_f0=_measure_f0(left, self.floor, self.ceiling),
            right_result_f0=_measure_f0(right, self.floor, self.ceiling),
        )
