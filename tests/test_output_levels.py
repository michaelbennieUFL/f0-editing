import numpy as np

from praat_backend import normalize_stereo_levels


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
