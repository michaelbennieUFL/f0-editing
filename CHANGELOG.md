# Changelog

## 0.2.0 - independent editable tiers and noise controls

- Left pitch points rendered as orange downward-pointing triangles; right as blue upward-pointing triangles. The original Praat pitch contour remains an unchanged gray dashed reference.
- Added editable-grid spacing control (1-500 ms), with independent **Regenerate LEFT**, **Regenerate RIGHT**, and **Regenerate BOTH** actions. The current edited contour is linearly interpolated; no new pitch detection occurs.
- Persist voiced-region break indices across regeneration, simplification, manual point insertion/removal, undo/redo, and JSON saves. Gaps are visibly disconnected in the pitch chart.
- Extra points can be added to either track through exact time/Hz fields and dedicated buttons, or by modifier+double-click in the pitch chart.
- Added white-noise playback slider (0%-100% of original recording RMS), uncorrelated per active ear, deterministic on repeat plays, with a checkbox to include noise in exported WAV files (off by default).
- The command-line renderer accepts `--noise-percent`.
- Existing v1 JSON projects remain supported; new optional fields preserve gap boundaries and noise controls.
- 14 automated tests passed using native Parselmouth overlap-add; GUI source compiles but requires an on-desktop integration test.
