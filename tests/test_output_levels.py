import numpy as np
import pytest

from praat_backend import apply_shared_amplitude_modulation, normalize_stereo_levels


def test_normalize_stereo_levels_matches_source_and_limits_peak():
    source = np.full(4000, 0.2, dtype=np.float32)
    generated = np.column_stack((
        np.full(4000, 0.06, dtype=np.float32),
        np.full(4000, 0.4, dtype=np.float32),
    ))
    result = normalize_stereo_levels(generated, source, match_rms=True, peak_dbfs=-6)
    rms = np.sqrt(np.mean(result**2, axis=0))
    assert np.allclose(rms[0], rms[1], atol=1e-5)
    assert np.max(np.abs(result)) <= 10 ** (-6 / 20) + 1e-6


def test_normalize_stereo_levels_can_leave_channel_balance_untouched():
    source = np.full(1000, 0.2, dtype=np.float32)
    generated = np.column_stack((source, source * 0.5))
    result = normalize_stereo_levels(generated, source, match_rms=False, peak_dbfs=0)
    np.testing.assert_allclose(result, generated)


def test_shared_amplitude_modulation_uses_identical_gain_for_both_channels():
    audio = np.column_stack((np.ones(24000), np.full(24000, 2.0))).astype(np.float32)
    result = apply_shared_amplitude_modulation(audio, 24000, 4.0, 20.0)
    np.testing.assert_allclose(result[:, 1], result[:, 0] * 2.0, atol=1e-6)
    assert np.isclose(result[0, 0], 1.0)
    assert np.max(result[:, 0]) <= 1.2 + 1e-5
    assert np.min(result[:, 0]) >= 0.8 - 1e-5


def test_shared_amplitude_modulation_validates_depth():
    audio = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="depth"):
        apply_shared_amplitude_modulation(audio, 24000, 4.0, 101.0)
    with pytest.raises(ValueError, match="between 2 and 10"):
        apply_shared_amplitude_modulation(audio, 24000, 1.0, 20.0)
