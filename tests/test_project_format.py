from pathlib import Path

from dichotic import PROJECT_FORMAT
from dichotic.project import migrate_legacy, new_project, resolve_audio_path


def test_legacy_praat_project_migrates_to_isolated_interface_state(tmp_path):
    legacy = {
        "format": "dichotic-praat-editor-v1",
        "audio_path": "voice.wav",
        "pitch_floor_hz": 75,
        "pitch_ceiling_hz": 600,
        "left_pitch_points": [[0.1, 120]],
        "right_pitch_points": [[0.1, 125]],
    }
    result = migrate_legacy(legacy)
    assert result["format"] == PROJECT_FORMAT
    assert result["interfaces"]["praat"]["left_pitch_points"] == [[0.1, 120]]
    assert result["interfaces"]["pitchflower"] is None
    assert resolve_audio_path(str(tmp_path / "edit.json"), result) == str(tmp_path / "voice.wav")


def test_v2_project_keeps_engine_payloads_separate(tmp_path):
    project = new_project(
        str(tmp_path / "input.wav"), str(tmp_path / "edit.json"),
        shared={"active_interface": "pitchflower"},
        engines={"praat": {"left_pitch_points": [[0, 100]]},
                 "pitchflower": {"left_pitch_points": [[0, 200]]}},
    )
    assert project["format"] == PROJECT_FORMAT
    assert project["interfaces"]["praat"]["left_pitch_points"] == [[0, 100]]
    assert project["interfaces"]["pitchflower"]["left_pitch_points"] == [[0, 200]]


def test_project_can_store_independent_promonet_settings(tmp_path):
    project = new_project(
        str(tmp_path / "input.wav"), str(tmp_path / "edit.json"),
        shared={"active_interface": "promonet"},
        engines={"promonet": {"left_pitch_points": [[0, 200]],
                                "settings": {"checkpoint_mode": 1, "checkpoint_path": "adapted",
                                             "speaker": 3, "spectral_balance": 1.1}}},
    )
    settings = project["interfaces"]["promonet"]["settings"]
    assert settings["checkpoint_path"] == "adapted"
    assert settings["speaker"] == 3


def test_project_can_store_world_interface_state(tmp_path):
    project = new_project(
        str(tmp_path / "input.wav"), str(tmp_path / "edit.json"),
        shared={"active_interface": "world"},
        engines={"world": {
            "left_pitch_points": [[0, 200]], "right_pitch_points": [[0, 220]],
            "settings": {"pitch_floor_hz": 60, "pitch_ceiling_hz": 700},
        }},
    )
    assert project["interfaces"]["world"]["settings"]["pitch_ceiling_hz"] == 700
