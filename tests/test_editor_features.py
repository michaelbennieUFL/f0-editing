"""Regression tests for triangle-track editing, regridding, gaps and noise.

Only tests the numerical/editing model; requires neither GUI nor sound device.
"""

import numpy as np
import pytest

from praat_backend import (
    infer_segment_breaks, normalize_breaks, resample_editable_pitch,
    insert_editable_point, delete_editable_point, simplify_segmented_pitch,
    mix_white_noise,
)


def example_tier():
    # Two voiced regions with a silent 300 ms gap and one manual F0 peak.
    return np.array([
        [0.10, 190.0], [0.12, 230.0], [0.15, 195.0],
        [0.45, 220.0], [0.47, 260.0], [0.50, 235.0],
    ])


def test_regenerate_uses_current_edits_not_original_detection():
    original = example_tier()
    edited = original.copy()
    edited[1, 1] = 350.0  # deliberately different from original
    result, breaks = resample_editable_pitch(edited, 0.01, (3,))
    assert breaks == (6,)
    assert result[2, 0] == pytest.approx(0.12)
    assert result[2, 1] == pytest.approx(350.0)
    assert original[1, 1] == 230.0  # immutable reference
    # The two regions are not connected by new points inside the gap.
    assert not np.any((result[:, 0] > 0.15) & (result[:, 0] < 0.45))
    assert len(result) > len(edited)


def test_coarse_then_fine_resampling_retains_voiceless_gaps():
    source = example_tier()
    breaks = infer_segment_breaks(source)
    assert breaks == (3,)
    coarse, coarse_breaks = resample_editable_pitch(source, 0.10, breaks)
    assert coarse_breaks == (2,)
    np.testing.assert_allclose(coarse[:, 0], [0.10, 0.15, 0.45, 0.50])
    refined, refined_breaks = resample_editable_pitch(coarse, 0.01, coarse_breaks)
    assert refined_breaks == (6,)
    assert not np.any((refined[:, 0] > 0.15) & (refined[:, 0] < 0.45))


def test_regenerate_left_and_right_independent_and_undo_state_possible():
    original = example_tier()
    left = original.copy()
    right = original.copy()
    left[:, 1] *= 1.25
    right[:, 1] *= 0.75
    snapshot = (left.copy(), right.copy(), (3,), (3,))
    new_left, new_left_breaks = resample_editable_pitch(left, 0.01, (3,))
    # Data of the other channel stays untouched; snapshot supports Ctrl+Z.
    np.testing.assert_array_equal(right, snapshot[1])
    assert len(new_left) > len(left)
    assert new_left_breaks == (6,)
    np.testing.assert_array_equal(snapshot[0], left)


def test_insert_on_specific_track_and_inside_gap():
    original = example_tier()
    left, breaks, idx = insert_editable_point(original, (3,), 0.30, 285.0)
    assert idx == 3
    assert len(left) == len(original) + 1
    assert breaks == (3, 4)  # new isolated voiced region
    np.testing.assert_array_equal(original, example_tier())
    assert left[idx, 1] == pytest.approx(285)
    regen, new_breaks = resample_editable_pitch(left, 0.01, breaks)
    assert len(new_breaks) == 2
    assert (np.abs(regen[:, 0] - .30) < 1e-7).sum() == 1
    with pytest.raises(ValueError, match='exists'):
        insert_editable_point(left, breaks, .30, 300.)


def test_delete_and_simplify_keep_region_breaks():
    source = example_tier()
    inserted, breaks, idx = insert_editable_point(source, (3,), 0.30, 265)
    result, updated_breaks = delete_editable_point(inserted, breaks, idx)
    np.testing.assert_array_equal(result, source)
    assert updated_breaks == (3,)
    simple, simple_breaks = simplify_segmented_pitch(source, (3,), 15.)
    assert simple_breaks == (2,)
    assert len(simple) == 4
    np.testing.assert_allclose(simple[:, 0], [.10, .15, .45, .50])


def test_invalid_spacing_and_breaks():
    source = example_tier()
    for spacing in (-.02, 0, float('nan')):
        with pytest.raises(ValueError):
            resample_editable_pitch(source, spacing, (3,))
    with pytest.raises(ValueError, match='too fine'):
        resample_editable_pitch(source, .001, (3,), max_points=5)
    with pytest.raises(ValueError, match='Invalid'):
        normalize_breaks((3, 3), len(source))


def test_white_noise_level_off_and_reproducible():
    reference = np.zeros((2000, 2), dtype=np.float32)
    clean = mix_white_noise(reference, 0, .1)
    assert clean.shape == reference.shape
    assert np.array_equal(clean, reference)
    noisy = mix_white_noise(reference, 50, .1, start_sample=220)
    again = mix_white_noise(reference, 50, .1, start_sample=220)
    assert np.array_equal(noisy, again)
    assert not np.array_equal(noisy, mix_white_noise(reference, 50, .1, start_sample=221))
    np.testing.assert_allclose(np.sqrt(np.mean(noisy**2, axis=0)), .05, atol=1e-5)
    assert abs(np.corrcoef(noisy.T)[0, 1]) < .1  # independent ears


def test_noise_only_in_audible_channel_and_peak_guard():
    silent = np.zeros((1000, 2), dtype=np.float32)
    only_left = mix_white_noise(silent, 25, .1, active_channels=(True, False))
    assert np.max(np.abs(only_left[:, 0])) > .01
    assert np.count_nonzero(only_left[:, 1]) == 0
    high = np.ones((1000, 2), dtype=np.float32) * .98
    noisy = mix_white_noise(high, 100, .4)
    assert np.max(np.abs(noisy)) <= .990001
    assert np.all(np.isfinite(noisy))
    with pytest.raises(ValueError, match='percentage'):
        mix_white_noise(silent, 101, .1)


def test_resampled_edited_contours_work_with_native_praat(tmp_path):
    import soundfile as sf
    from praat_backend import PraatEngine
    sr = 16000
    t = np.arange(sr) / sr
    signal = .16 * np.sin(2 * np.pi * 190 * t) + .06 * np.sin(2 * np.pi * 380 * t)
    path = tmp_path / "original.wav"
    sf.write(path, signal, sr)
    engine = PraatEngine(str(path), 75, 500)
    original = engine.initial_pitch.copy()
    left = original.copy()
    right = original.copy()
    left[:, 1] *= 1.17
    right[:, 1] *= 0.88
    breaks = infer_segment_breaks(original)
    left, lb = resample_editable_pitch(left, .025, breaks)
    right, rb = resample_editable_pitch(right, .050, breaks)
    assert lb == rb  # this synthetic vowel has no unvoiced breaks
    stereo = engine.stereo(left, right)
    assert stereo.shape == (len(engine.samples), 2)
    assert np.isfinite(stereo).all()
    assert np.max(np.abs(stereo[:, 0] - stereo[:, 1])) > .01
    np.testing.assert_array_equal(engine.initial_pitch, original)
