import numpy as np

from dichotic_editor import hz_to_mel, mel_to_hz, perceptual_midpoint


def test_mel_conversion_round_trips_hertz():
    frequencies = np.array([55.0, 110.0, 220.0, 440.0, 880.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(frequencies)), frequencies, rtol=1e-12)


def test_perceptual_midpoints_use_log_frequency_weights():
    left = np.array([200.0])
    right = np.array([800.0])
    np.testing.assert_allclose(perceptual_midpoint(left, right, 0.5), [400.0])
    np.testing.assert_allclose(perceptual_midpoint(left, right, 1 / 3), [np.cbrt(200.0 * 800.0 * 800.0)])
    np.testing.assert_allclose(perceptual_midpoint(left, right, 2 / 3), [np.cbrt(200.0 * 200.0 * 800.0)])
