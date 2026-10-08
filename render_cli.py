"""Headless renderer using the exact same Praat overlap-add engine as the GUI.

Example:
 uv run python render_cli.py input.wav -o output.wav --left-semitones 3 --right-semitones -2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from praat_backend import PraatEngine, checked_points, mix_white_noise


def main():
    parser = argparse.ArgumentParser(description="Native Praat two-channel overlap-add stereo renderer")
    parser.add_argument("input",help="Original mono or stereo recording")
    parser.add_argument("-o","--output",required=True,help="Output two-channel WAV")
    parser.add_argument("--left-semitones",type=float,default=0,help="Pitch transposition of left channel")
    parser.add_argument("--right-semitones",type=float,default=0,help="Pitch transposition of right channel")
    parser.add_argument("--pitch-floor",type=float,default=None)
    parser.add_argument("--pitch-ceiling",type=float,default=None)
    parser.add_argument("--project",help="Use pitch contours from a saved GUI JSON project")
    parser.add_argument("--noise-percent", type=float, default=0, metavar="0..100",
                        help="Optional white-noise RMS as percent of original recording RMS; default clean")
    args=parser.parse_args()
    obj=None
    if args.project:
        import json
        obj=json.loads(Path(args.project).read_text(encoding="utf-8"))
        if obj.get("format")!="dichotic-praat-editor-v1":
            parser.error("Invalid project format")
    floor=args.pitch_floor if args.pitch_floor is not None else (obj["pitch_floor_hz"] if obj else 75)
    ceiling=args.pitch_ceiling if args.pitch_ceiling is not None else (obj["pitch_ceiling_hz"] if obj else 600)
    engine=PraatEngine(args.input,floor,ceiling)
    left=engine.initial_pitch.copy()
    right=engine.initial_pitch.copy()
    if obj:
        left=checked_points(np.asarray(obj["left_pitch_points"],dtype=float),0,engine.duration)
        right=checked_points(np.asarray(obj["right_pitch_points"],dtype=float),0,engine.duration)
    left[:,1]*=2**(args.left_semitones/12)
    right[:,1]*=2**(args.right_semitones/12)
    stereo=engine.stereo(left,right)
    if args.noise_percent:
        reference_rms = float(np.sqrt(np.mean(engine.samples ** 2)))
        stereo = mix_white_noise(stereo, args.noise_percent, reference_rms)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output,stereo,engine.sr,subtype="PCM_24")
    print(f"Wrote {args.output}: {len(stereo)/engine.sr:.3f}s, "
          f"{engine.sr} Hz, 2 channels; {len(left)} left F0 points, {len(right)} right F0 points")


if __name__=="__main__":
    main()
