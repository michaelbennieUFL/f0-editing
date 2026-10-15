"""Versioned, engine-isolated project-file helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import PROJECT_FORMAT


def new_project(audio_path: str, project_path: str, *, shared: dict[str, Any], engines: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 document without embedding source or generated audio."""
    return {
        "format": PROJECT_FORMAT,
        "audio_path": os.path.relpath(audio_path, Path(project_path).parent),
        "shared": shared,
        "interfaces": engines,
    }


def migrate_legacy(data: dict[str, Any]) -> dict[str, Any]:
    """Convert the former Praat-only document into an isolated v2 payload."""
    if data.get("format") != "dichotic-praat-editor-v1":
        return data
    praat = {
        "version": 1,
        "pitch_floor_hz": data["pitch_floor_hz"],
        "pitch_ceiling_hz": data["pitch_ceiling_hz"],
        "left_pitch_points": data["left_pitch_points"],
        "right_pitch_points": data["right_pitch_points"],
        "left_segment_starts": data.get("left_segment_starts", []),
        "right_segment_starts": data.get("right_segment_starts", []),
        "point_spacing_ms": data.get("point_spacing_ms", 20),
        "result": None,
    }
    return {
        "format": PROJECT_FORMAT,
        "audio_path": data["audio_path"],
        "shared": {
            "active_interface": "praat",
            "white_noise_percent": data.get("white_noise_percent", 0),
            "white_noise_in_export": data.get("white_noise_in_export", False),
            "selection": data.get("selection"),
        },
        "interfaces": {"praat": praat, "pitchflower": None, "promonet": None},
    }


def resolve_audio_path(project_path: str, data: dict[str, Any]) -> str:
    return str((Path(project_path).parent / data["audio_path"]).resolve())
