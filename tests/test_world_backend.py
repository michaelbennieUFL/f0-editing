import numpy as np
import pytest

from dichotic.interfaces.world.backend import target_f0_from_points


def test_world_target_preserves_unvoiced_frames_and_interpolates_islands():
    times = np.arange(10, dtype=float) * 0.01
    source = np.array([0, 100, 100, 100, 0, 0, 200, 200, 200, 0], dtype=float)
    points = np.array([[0.01, 120], [0.03, 180], [0.06, 240], [0.08, 300]], dtype=float)

    result = target_f0_from_points(source, times, points)

    np.testing.assert_allclose(result, [0, 120, 150, 180, 0, 0, 240, 270, 300, 0])


def test_world_point_in_unvoiced_gap_cannot_create_voicing():
    times = np.arange(7, dtype=float) * 0.01
    source = np.array([0, 100, 100, 0, 0, 200, 200], dtype=float)
    points = np.array([[0.01, 120], [0.04, 400], [0.06, 240]], dtype=float)

    result = target_f0_from_points(source, times, points)

    assert np.all(result[[0, 3, 4]] == 0)
    np.testing.assert_allclose(result[[1, 2, 5, 6]], [120, 120, 240, 240])


def test_world_target_rejects_invalid_points():
    source = np.array([100.0, 100.0])
    times = np.array([0.0, 0.01])
    with pytest.raises(ValueError, match="positive"):
        target_f0_from_points(source, times, np.array([[0.0, -1.0]]))
    with pytest.raises(ValueError, match="strictly increasing"):
        target_f0_from_points(source, times, np.array([[0.01, 100.0], [0.0, 120.0]]))


def test_world_engine_uses_shared_spectral_features_for_both_channels(tmp_path):
    pyworld = pytest.importorskip("pyworld")
    soundfile = pytest.importorskip("soundfile")
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=float) / sample_rate
    signal = .18 * np.sin(2 * np.pi * 190 * t) + .06 * np.sin(2 * np.pi * 380 * t)
    path = tmp_path / "vowel.wav"
    soundfile.write(path, signal, sample_rate)

    from dichotic.interfaces.world.backend import WorldEngine

    engine = WorldEngine(str(path), 75, 500)
    assert engine.sr == 24_000
    assert engine.sp.ndim == 2 and engine.ap.ndim == 2
    assert engine.sp.shape[0] == engine.ap.shape[0]
    left = engine.initial_pitch.copy()
    right = engine.initial_pitch.copy()
    left[:, 1] *= 1.2
    right[:, 1] *= 0.8
    stereo = engine.stereo(left, right)
    assert stereo.shape == (len(engine.samples), 2)
    assert stereo.dtype == np.float32
    assert np.isfinite(stereo).all()
    assert np.max(np.abs(stereo[:, 0] - stereo[:, 1])) > 0.01
