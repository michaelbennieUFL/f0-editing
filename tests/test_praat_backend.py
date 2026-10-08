"""Tests use a synthetic voiced sound, so no audio hardware or GUI is needed."""

import numpy as np
import pytest
import parselmouth
import soundfile as sf

from praat_backend import PraatEngine, checked_points, simplify_pitch, fit_audio


@pytest.fixture
def voiced_engine(tmp_path):
    sr=16000
    t=np.arange(round(sr*1.25))/sr
    # Harmonic complex better approximates a short voiced vowel than a pure sine.
    signal=.18*np.sin(2*np.pi*185*t)+.08*np.sin(2*np.pi*370*t)+.035*np.sin(2*np.pi*555*t)
    path=tmp_path/"vowel.wav"
    sf.write(path,signal,sr)
    return PraatEngine(str(path),75,500)


def median_pitch(sound,rate):
    result=parselmouth.Sound(sound,rate).to_pitch(pitch_floor=75,pitch_ceiling=500)
    f0=result.selected_array["frequency"]
    voiced=f0[f0>0]
    assert len(voiced) > 20
    return float(np.median(voiced))


def test_native_stereo_psola(voiced_engine,tmp_path):
    engine=voiced_engine
    assert engine.initial_pitch.shape[1]==2
    assert len(engine.initial_pitch)>60
    left=engine.initial_pitch.copy()
    right=engine.initial_pitch.copy()
    left[:,1]*=1.30
    right[:,1]*=0.80
    stereo=engine.stereo(left,right)
    assert stereo.dtype==np.float32
    assert stereo.shape==(len(engine.samples),2)
    assert np.max(np.abs(stereo[:,0]-stereo[:,1]))>0.01
    left_hz=median_pitch(stereo[:,0],engine.sr)
    right_hz=median_pitch(stereo[:,1],engine.sr)
    assert left_hz>right_hz*1.35
    assert 225<left_hz<258
    assert 138<right_hz<157
    output=tmp_path/"stereo.wav"
    sf.write(output,stereo,engine.sr,subtype="PCM_24")
    reread,sr=sf.read(output,always_2d=True)
    assert reread.shape==stereo.shape and sr==engine.sr
    assert np.max(np.abs(reread[:,0]-reread[:,1]))>0.01


def test_stylize_splits_voicing_gaps():
    first=np.column_stack((np.arange(.1,.51,.01), 200+10*np.sin(np.arange(.1,.51,.01)*20)))
    second=np.column_stack((np.arange(.8,1.11,.01), 260+5*np.cos(np.arange(.8,1.11,.01)*20)))
    original=np.concatenate([first,second])
    out=simplify_pitch(original,0.5,gap_s=.05)
    assert len(out)<len(original)
    for t in (first[0,0],first[-1,0],second[0,0],second[-1,0]):
        assert np.min(np.abs(out[:,0]-t))<1e-6
    assert np.all(np.diff(out[:,0])>0)


def test_invalid_tier_rejected(voiced_engine):
    engine=voiced_engine
    with pytest.raises(ValueError,match="increasing"):
        checked_points(np.array([[0.5,200],[0.4,250]]),0,engine.duration)
    with pytest.raises(ValueError,match="positive"):
        checked_points(np.array([[0.5,-200]]),0,engine.duration)
    with pytest.raises(ValueError,match="outside"):
        checked_points(np.array([[engine.duration+1,200]]),0,engine.duration)
    assert checked_points(np.empty((0,2)),0,engine.duration).shape==(0,2)


def test_empty_pitch_tier_and_add_point(voiced_engine):
    engine=voiced_engine
    # Point insertion/removal is the editing primitive used by the GUI.
    pts=np.array([[.25,170.0],[.55,245.0],[1.0,185.0]])
    output=engine.stereo(pts,engine.initial_pitch.copy())
    assert output.shape==(len(engine.samples),2)
    assert np.isfinite(output).all()


def test_fit_audio():
    np.testing.assert_array_equal(fit_audio(np.array([1,2]),4),[1,2,0,0])
    np.testing.assert_array_equal(fit_audio(np.array([1,2,3,4]),2),[1,2])
