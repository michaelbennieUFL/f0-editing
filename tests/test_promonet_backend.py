import numpy as np

from dichotic.interfaces.promonet.backend import F0_CEILING, F0_FLOOR, _points_from_pitch


def test_promonet_pitch_points_keep_only_voiced_frames():
    points = _points_from_pitch(np.array([0.0, 105.0, np.nan, 220.0]))
    assert len(points) == 2
    assert np.allclose(points[:, 1], [105.0, 220.0])
    assert points[1, 0] > points[0, 0]


def test_promonet_supported_f0_limits_are_documented_by_adapter():
    assert (F0_FLOOR, F0_CEILING) == (50.0, 550.0)
