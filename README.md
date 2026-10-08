# Dichotic Pitch Editor (Praat overlap-add)

A standalone Python application for creating **two independently editable F0 contours** for the same recording, with the original time alignment preserved.

**Version 0.2:** Left points are **orange downward triangles (▼)** and right points are **blue upward triangles (▲)**. You can regenerate the *editable* points at a new spacing, insert precise extra points into either track, and mix adjustable white noise during listening.

- **Orange ▼** PitchTier → **left-ear audio**.
- **Blue ▲** PitchTier → **right-ear audio**.
- The initial analysis appears as a **gray dashed, immutable reference**; re-spacing never replaces the edited tiers from this reference.
- Stereo, one-ear-only, channel-swapped, or original playback from a clickable timeline.
- **Native Praat overlap-add resynthesis via Parselmouth**, not a generic FFT pitch shifter.
- Adjustable **White noise level** slider (0–100% of the original audio RMS) during playback.
- White noise is off at 0%; generated independently for active left/right ears, and reproducible when replaying the same time segment.
- Save/load pitch-edit projects, export stereo WAV, and export F0 point CSV.

## Ubuntu 26.04 installation

This is an **independent desktop editor**, not an extension installed inside Praat. Praat itself does **not** need to be running. It uses the Praat engine bundled with Python's `praat-parselmouth` package.

Install the system libraries (some may already be installed):

```bash
sudo apt update
sudo apt install libportaudio2 libsndfile1 libxcb-cursor0 libxkbcommon-x11-0 libgl1
```

If `uv` is not already installed, follow https://docs.astral.sh/uv/getting-started/installation/ .

Then extract the package and run:

```bash
unzip -o dichotic_praat_editor_v0.2.zip
cd dichotic_praat_editor
uv python install 3.12
uv sync --python 3.12
uv run --python 3.12 python dichotic_editor.py
```

Open a file using **Open audio…**, or specify it on the command line:

```bash
uv run --python 3.12 python dichotic_editor.py "/path/to/recording.wav"
```

You can also use `./run_ubuntu.sh` after installing `uv`. The `.python-version` file requests Python 3.12, which avoids relying on Ubuntu 26.04's default Python version.

## Edit both pitch contours

1. Load a short audio sample (ideally 1–15 seconds for responsive playback).
2. Select **Left / orange** or **Right / blue** using the **Edit contour** selector.
3. **Drag** a triangle to change its time and F0. Dragging only moves the active color. **Orange triangles point down** and **blue triangles point up** so the channels are distinguishable without relying on color.
4. **Double-click** the plot to add a point to the active channel. To add to either track regardless of active selection: **Shift + double-click** inserts a **LEFT ▼** point; **Ctrl + double-click** inserts a **RIGHT ▲** point.
5. For exact insertion, enter **New point at time (s)** and **F0 (Hz)** and press **+ Add LEFT ▼** or **+ Add RIGHT ▲**. You can also select an existing point and edit its exact time and Hz with **Apply values**.
6. Select an existing point and press **Delete** to remove it. Use **Ctrl+Z** / **Ctrl+Y** to undo / redo all point edits, including regeneration.
7. Use **Simplify active** (e.g. 0.35 semitones) to reduce dense 10-ms points without connecting voiced regions across silence.
8. Use **Copy active → other** or **Reset active** if you want matching pitch contours again.

The initial orange and blue curves overlap (same detected F0). The active curve draws in front, so you may only see one until you edit or switch channels. A faint gray dashed line marks the **original, untouched detected F0**.

### Regenerate pitch points at a different spacing

To change the editable point density, set **Editable point spacing**, e.g. `10 ms`, `25 ms`, `50 ms`, or `100 ms`, and choose one of:

- **Regenerate LEFT ▼** — resamples only the orange editable points.
- **Regenerate RIGHT ▲** — resamples only the blue editable points.
- **Regenerate BOTH** — resamples both independent curves using the selected spacing.

The application linearly interpolates **your currently edited contour**, *not* the initial Praat analysis. The original gray dashed points are never regenerated or changed. Endpoints of voiced regions are kept, and the project stores explicit voiced-region boundaries so moving from coarse to fine spacing doesn't accidentally fill pauses between syllables.

If you insert a point into an unvoiced gap, it is kept as its own region on subsequent regeneration. Resampling with **coarser** spacing can smooth away short peaks between the new points; use **Ctrl+Z** to reverse it. Reducing spacing again cannot reconstruct detail you deliberately removed by coarse resampling.

**Example:** Edit one orange point to 290 Hz. Set spacing to 20 ms, then press **Regenerate LEFT ▼**. The regenerated orange points follow the edited curve; blue points and the original gray reference stay unchanged.

## Hear changes from the timeline

The bottom pane has a waveform and a labeled **PLAY** ruler. Click the playback ruler to start audio at that time:

| Click / modifier | Left ear | Right ear |
| --- | --- | --- |
| Click | Synthesized orange | Synthesized blue |
| **Shift** + click | Synthesized orange | Silence |
| **Ctrl** + click | Silence | Synthesized blue |
| **Alt** + click | Unmodified source | Unmodified source |
| **Ctrl+Shift** + click | Synthesized blue | Synthesized orange |

On some Ubuntu desktops, **Alt+click** may be intercepted by the window manager. In that case use the **Original** button or press **O**.

To listen to a time range, **drag across the waveform** (above the ruler). This sets a selection and turns on **Use selected region for playback**. Any click on the PLAY ruler now plays that range; uncheck that option or choose **Clear region** to return to click-from-position playback.

Keyboard alternatives: **Space** stereo, **L** left, **R** right, **O** unmodified original, **S** swapped, **Esc** stop, **Ctrl+Z** undo, **Ctrl+Y** redo, **Delete** remove selected point. The buttons perform the same actions without a shortcut.

### White-noise masking during playback

The **White noise level** bar is immediately below the playback buttons:

| Setting | Result |
| --- | --- |
| **0%** | No added noise (default) |
| **10%** | Noise RMS = 10% of the original recording RMS (−20 dB relative to original RMS) |
| **50%** | Noise RMS = 50% of the original recording RMS (about −6 dB) |
| **100%** | Noise RMS = original recording RMS (0 dB relative) |

Both channels receive **independent, uncorrelated** white noise. For Left-only or Right-only playback, the inactive ear stays silent. Original/Swapped/Stereo playback get noise in both ears. The same segment uses the same random seed each replay for repeatable comparisons. At high levels, the program may attenuate both channels together to prevent clipping.

Moving the noise slider changes the **next playback immediately** (no new Praat resynthesis). Noise is **not** automatically saved into your stereo WAV: check **Include noise in exported WAV** to bake the currently selected level into the export.

### What "real time" means

**The pitch graph moves immediately as you drag.** The program regenerates the modified left and right channels **after the edit ends**, asynchronously, with the actual Praat overlap-add algorithm. Playback waits until the latest edit is ready, so clicking Play will not use a stale version. It does **not** continuously pitch-change an already-playing audio stream while you drag a point. That would need an audio-thread buffering design.

## Save and export

- **Save project…** saves a JSON file with both independent F0 tiers, voiced-region boundary indices, point-spacing field, current noise level, export-noise preference, and a path to the original audio. Keep the source recording available when reopening; the audio is **not embedded** into the JSON. Existing v1 projects continue to load.
- **Load project…** restores both edits (also supported as a `.json` command-line argument).
- **Export stereo WAV…** exports left orange and right blue as 24-bit stereo PCM. Export is **clean by default**, and includes noise only when the corresponding checkbox is enabled.
- **Export F0 CSV…** saves `channel,time_s,f0_hz` for both curves, for data analysis or plotting.

Stereo input files are downmixed to mono before pitch analysis: both edited channels are derived from **the same signal**. Headphones and a stereo output device are needed. Check that your OS audio accessibility settings are not mixing both ears into mono.

## Non-GUI synthesis / tests

You can render a quick dichotic sample without opening the GUI:

```bash
uv run --python 3.12 python render_cli.py input.wav -o compare.wav \
  --left-semitones 3 --right-semitones -2
```

The CLI also supports `--project saved.json` to use contours saved by the GUI, followed by optional additional semitone transposition. For batch white-noise mixing, pass `--noise-percent 25` (default is 0, meaning a clean WAV). The project saved noise-slider setting is not applied implicitly by the CLI; its `--noise-percent` parameter takes precedence.

Run audio-engine tests:

```bash
uv run --python 3.12 --with pytest pytest -q
```

Tests use synthetic harmonic recordings and do **not** need headphones or an active audio device. `tests/demo_vowel.wav` and `tests/demo_stereo.wav` demonstrate input and two-channel output.

## Engine and fidelity

We use Praat's normal processing sequence for **each** ear independently:

```python
manipulation = praat.call(sound, "To Manipulation", 0.01, 75, 600)
pitch_tier = praat.call("Create PitchTier", "modified", 0, duration)
for time_s, f0_hz in points:
    praat.call(pitch_tier, "Add point", time_s, f0_hz)
praat.call([pitch_tier, manipulation], "Replace pitch tier")
processed = praat.call(manipulation, "Get resynthesis (overlap-add)")
```

Then we combine the left/right generated samples as a true **two-channel PCM stereo** stream. Both use the original duration and synchronized sample index; they are not played as independently timed streams.

This calls Praat's **native** overlap-add implementation through Parselmouth. The exact Praat version inside Parselmouth may differ from the separately installed Praat 7 desktop version, so bit-identical results across the two installations are not guaranteed. Pitch detection and overlap-add can sound unnatural for extreme F0 edits or noisy/creaky/unvoiced audio; those are limitations of the underlying method.

Sources:
- https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html
- https://praat.org/manual/Manipulation.html
- https://praat.org/manual/Manipulation__Get_resynthesis__overlap-add_.html

## Platform and test status

The **Praat backend** was run successfully on Linux with Parselmouth and NumPy. Automated tests cover actual left/right Praat overlap-add pitch separation, editable-point re-spacing without overwriting the original curve, voiced-gap preservation, insertion/removal, segmentation, deterministic white-noise mixing, and stereo export. The GUI source was syntax-checked, but **not run interactively** because PyQt6/sounddevice cannot be installed in this packaging environment (network/DNS unavailable). First live GUI validation must occur after `uv sync` on your Ubuntu desktop.
