"""Lazy PitchFlower adapter with independent dichotic channel synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from praat_backend import fit_audio

SAMPLE_RATE = 24_000
FRAME_PERIOD_MS = 32 / 3
F0_FLOOR = 20.0
MAX_SAMPLES = 36_096 * 3


def world_f0(audio: np.ndarray) -> np.ndarray:
    """Extract a WORLD contour, retaining zero in unvoiced frames."""
    try:
        import pyworld as pw
    except ImportError as exc:  # Delay optional inference dependencies until used.
        raise RuntimeError("PitchFlower requires pyworld. Install the PitchFlower extra.") from exc
    signal = np.asarray(audio, dtype=np.float64)
    raw, times = pw.harvest(signal, fs=SAMPLE_RATE, frame_period=FRAME_PERIOD_MS, f0_floor=F0_FLOOR)
    return pw.stonemask(signal, raw, times, fs=SAMPLE_RATE)


def f0_points(audio: np.ndarray) -> np.ndarray:
    values = world_f0(audio)
    times = np.arange(len(values), dtype=float) * FRAME_PERIOD_MS / 1000.0
    return np.column_stack((times[values > 0], values[values > 0]))


@dataclass(frozen=True)
class PitchFlowerResult:
    stereo: np.ndarray
    left_result_f0: np.ndarray
    right_result_f0: np.ndarray


class PitchFlowerEngine:
    """A 24-kHz source and lazily-loaded neural model for one editor session."""

    name = "pitchflower"

    def __init__(self, path: str):
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
        mono = samples.mean(axis=1)
        if rate != SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("PitchFlower requires scipy for 24-kHz resampling.") from exc
            mono = resample_poly(mono, SAMPLE_RATE, int(rate)).astype(np.float64)
        if len(mono) > MAX_SAMPLES:
            seconds = MAX_SAMPLES / SAMPLE_RATE
            raise ValueError(
                f"PitchFlower supports audio up to {seconds:.2f} seconds at 24 kHz; "
                f"this source becomes {len(mono) / SAMPLE_RATE:.2f} seconds after resampling."
            )
        if len(mono) < 128:
            raise ValueError("Recording is too short for PitchFlower")
        self.path = str(Path(path).resolve())
        self.samples = np.ascontiguousarray(mono)
        self.sr = SAMPLE_RATE
        self.duration = len(mono) / SAMPLE_RATE
        self.initial_pitch = f0_points(self.samples)
        self._model = None
        self._vocos = None
        self._torch = None
        self._device = None

    def _load_models(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchaudio
            from pitchflower.synthesizer import PitchFlowerSynthesizer
            from vocos import Vocos
        except ImportError as exc:
            raise RuntimeError(
                "PitchFlower dependencies are unavailable. Run `uv sync --extra pitchflower`."
            ) from exc
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = PitchFlowerSynthesizer.from_pretrained("diegotg343/PitchFlower").to(self._device)
        self._vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self._device)
        self._torchaudio = torchaudio

    def _target(self, points: np.ndarray) -> np.ndarray:
        frames = np.arange(len(world_f0(self.samples)), dtype=float)
        times = frames * FRAME_PERIOD_MS / 1000.0
        pts = np.asarray(points, dtype=float).reshape((-1, 2))
        if len(pts) == 0:
            return np.zeros(len(times), dtype=np.float32)
        f0 = np.interp(times, pts[:, 0], pts[:, 1], left=pts[0, 1], right=pts[-1, 1])
        source_f0 = world_f0(self.samples)
        target = np.zeros_like(source_f0)
        target[source_f0 > 0] = f0[source_f0 > 0]
        target[source_f0 > 0] = np.log2(np.maximum(target[source_f0 > 0], 1e-6))
        return target.astype(np.float32)

    def _render_channel(self, points: np.ndarray, *, steps: int, guidance: float, seed: int) -> np.ndarray:
        self._load_models()
        assert self._torch is not None and self._device is not None
        torch = self._torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        source = torch.from_numpy(self.samples.astype(np.float32))[None].to(self._device)
        target = torch.from_numpy(self._target(points))[None].to(self._device)
        with torch.inference_mode():
            mels = self._model.reconstruct_mels(source, target, steps=int(steps), w=float(guidance))
            output = self._vocos.decode(mels)[0].detach().cpu().numpy()
        return fit_audio(output, len(self.samples)).astype(np.float32)

    def stereo(self, left_points: np.ndarray, right_points: np.ndarray, *, steps: int = 100,
               guidance: float = 2.0, seed: int = 0) -> PitchFlowerResult:
        if steps < 1:
            raise ValueError("PitchFlower flow steps must be at least 1")
        left = self._render_channel(left_points, steps=steps, guidance=guidance, seed=seed)
        right = self._render_channel(right_points, steps=steps, guidance=guidance, seed=seed + 1)
        return PitchFlowerResult(
            stereo=np.column_stack((left, right)).astype(np.float32),
            left_result_f0=f0_points(left),
            right_result_f0=f0_points(right),
        )
