# PawCapture — CHANGELOG

Working log so context can be restored across sessions. Newest first.

---

## 2026-05-07 — phase tag (baseline / post-drug / antagonist) folder + filename

Adds a session-global **Phase** selector to the legend bar with options *None / Baseline / Post-Drug / Antagonist / Custom…* plus a *Prefix / Suffix* position toggle. When a phase is set, recordings land in `recordings/YYYY-MM-DD/<phase>/` and the tag is added to the filename (`baseline_CAM_1_…mp4` or `CAM_1_…_baseline.mp4`). Date folder stays auto. None reproduces the prior path layout exactly.

### Implementation
- `Recorder._resolve_path` / `Recorder.start` gain `phase_tag` + `phase_position` kwargs. Empty `phase_tag` short-circuits back to the old `day_dir / f"{label}{ts}.mp4"` shape, so existing behavior is preserved when phase is unset.
- `MainWindow.current_phase_tag()` / `current_phase_position()` are read by `CameraPanel.start_rec` via `self.window()`. Custom names are sanitized to `[a-z0-9_-]` (lowercase, spaces → `_`, leading/trailing punctuation stripped).
- Per-camera RECORD and global RECORD ALL both honor the phase. Session manifest sidecar lands inside the phase folder alongside the recordings (no manifest code changes needed — it derives `target_dir` from the first cam's file path).

### Why
Drug-experiment workflow: each session is one phase, three cameras roll, files need to be sorted into baseline/post-drug/antagonist buckets at write time so PixelPaws sees the right grouping without manual triage.

---

## 2026-05-06 (round 14) — feature bundle: manifest, reader, marks, scale bar, etc.

Bumped to `PAWCAPTURE_VERSION = "1.0.0"` — first stable release. Pairs with the `pawcapture.session/v1` manifest schema and the `pixelpaws_calibrated=1` mp4 sentinel: PawCapture is now declaring a stable interface for PixelPaws to consume. Adds 10 features in one pass — they're loosely coupled and the testing surface is mostly UI-level.

### PixelPaws integration trio
- **Software version stamp** — `pawcapture_version` is now in every recording's mp4 metadata (no longer gated on calibration), every profile JSON (`_pawcapture_version`), and every session manifest. Lets PixelPaws branch on schema if metadata evolves.
- **Session manifest sidecar** — every RECORD ALL session writes `session_<id>.json` in the day-folder of the first cam's out_dir. Schema header `pawcapture.session/v1`. Holds: per-cam file/calibration/resolution/fps/crop/encoder, sync marks, profile name, machine label, start/end timestamps. PixelPaws ingests one JSON per session instead of `ffprobe`-ing each file.
- **`pawcapture_meta.py`** — standalone module (no PyQt/cv2 deps) with `read_calibration(mp4)`, `read_session_manifest(json)`, `find_session_for_video(mp4)`. Drop into PixelPaws unmodified. Resolves `ffprobe` from PATH or next to a sibling `ffmpeg`.

### Quality-of-life
- **Date-based recording subfolders** — `Recorder._resolve_path` now nests output under `out_dir/YYYY-MM-DD/`. Created on demand by `Recorder.start`.
- **Pre-record disk-space check** — `_disk_space_ok` warns (with proceed-anyway prompt) when free space on any active out_dir's drive is below 1 GB per camera using that drive.
- **Keyboard shortcuts** — `Space` toggles RECORD ALL; `M` drops a sync marker. Guarded against firing when a text input has focus.
- **Calibration scale-bar overlay** — when a camera is calibrated, `_paint_canvas` draws a 100 mm scale bar in the bottom-left corner of the live preview. Catches "I clicked the wrong end" or stale-after-zoom-twist mistakes before recording.

### Workflow tools
- **📍 MARK button** — only enabled while recording. Pops a label dialog, appends `{t_seconds, label, wall_time}` to the active session's `_session_marks` list, written into the manifest at stop.
- **TEST button** — 2-second dry run on every connected camera, verifies output files are produced and >1 KB. Reports OK/FAIL per cam in a popup. Doesn't write a session manifest (intentional — these aren't real sessions).
- **Profile IMPORT/EXPORT** — `QFileDialog`-backed export of current state to any path; import copies a chosen JSON into `~/PawCapture/profiles/` (with overwrite prompt on collision).

### Internal refactor
`MainWindow._toggle_rec` was getting unwieldy — split into `_start_recording_session` and `_stop_recording_session` with helpers `_panel_session_record`, `_write_session_manifest`, `_disk_space_ok`, `_record_mark`. Each is short and single-purpose.

### Restart required
Same Python-edit pattern. The PyInstaller bundle from round 13 needs a rebuild (`build.bat`) to ship these.

---

## 2026-05-06 (round 13) — portable PyInstaller bundle

### Goal
Ship PawCapture to other lab machines without requiring a Python install. Target: drop a folder, double-click an exe, done.

### Files
- `PawCapture.spec` — PyInstaller spec, `--onedir` layout with `contents_directory='.'` so deps land flat alongside `PawCapture.exe` (no `_internal\`). Bundles `ffmpeg\ffmpeg.exe` to the dist root via `binaries=[(..., '.')]`, which lines up with `_find_ffmpeg()`'s `Path(sys.executable).parent / "ffmpeg.exe"` lookup. UPX disabled (Defender false-positives). Excludes WebEngine, Bluetooth, Multimedia, NfC, Sensors, SerialPort, RemoteObjects, Qml/Quick, Sql, Test, plus tkinter/matplotlib/pandas/scipy/etc — saves ~150 MB.
- `build.bat` — primary build script; uses `py -3 -m PyInstaller` (the `py` launcher works around the Microsoft-Store-stub `python` resolution on stock Win10/11). Auto-installs `requirements.txt`. `--clean` flag for a fresh build.
- `build.ps1` — equivalent PowerShell script for users who prefer it (note: stock execution policy may block; users either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once or fall back to `build.bat`).
- `requirements.txt` — pinned to PyQt5 5.15.x, opencv-python 4.8+, numpy 1.24+, pyinstaller 6.x. Removed unused `pygrabber` (was replaced by raw COM in session 4).

### Verified
First build on this machine: 316 MB total bundle, ffmpeg.exe present at the bundle root, exit 0. End-user instructions: zip `dist\PawCapture\` and drop on any Win10/11 machine — no Python install, no per-machine setup beyond first-launch SmartScreen click-through.

### Caveats
- Bundle is unsigned. Windows SmartScreen will flag it on first run; users click "More info → Run anyway" once. Code signing (~$200/yr) only worth it for distribution outside the lab.
- Camera slot pinning lives in `~/PawCapture/camera_slots.json`, which is per-user — different rigs build their own pin file on first run. Profiles already fall back to device-name slot matching when ID's don't match (round-9 collision-recovery code), so a profile shared across machines still works.
- ffmpeg.exe bundled is whatever the dev had in `camsync\ffmpeg\` — currently the official Win64 build with QSV/NVENC/AMF/libx264 support. Swap-in is supported (just replace the file in dist root).

### Restart required
For source devs only — the spec doesn't change runtime behavior.

---

## 2026-05-06 (round 12) — overwrite-loaded-profile prompt + in-app HELP

### Save Profile UX
Previously `save_profile` opened a blank text dialog every time, so iterating on a loaded profile spawned a new file each save (e.g. `Crop_final`, `Crop_final_final`, `Crop_final_final_final` — the user's actual profiles directory had several of these). New behavior:

- If a profile is loaded and its file still exists, clicking **SAVE** opens a 3-button QMessageBox: **Overwrite** (writes to the loaded path), **Save As New** (falls through to the text dialog, pre-filled with the loaded name), or **Cancel**.
- If the typed name in "Save As New" collides with another existing profile, ask before clobbering.
- After any save the combo re-selects the saved profile so the loaded-state stays consistent.
- `MainWindow._loaded_profile_name` tracks the loaded profile; set on `load_profile`, cleared on `delete_profile` if it matched.

### HELP button
New `? HELP` button in the topbar (between camera +/− and RECORD ALL). Opens a modal `_HelpDialog` (QTextBrowser, HTML-styled) covering:

- Quick start
- Camera-panel header buttons
- Calibration walkthrough + when to recalibrate
- mp4 metadata key list + ffprobe command for PixelPaws integration
- Profile save/load/delete (incl. the new overwrite prompt)
- Recording details (FFmpeg encoder chain, output dirs, filename suffix)
- Hardware notes (See3CAM_CU27 + CB-2812-3MP varifocal)
- Troubleshooting common symptoms

HTML is held in `_HELP_HTML` string; styled with the app's color tokens (BG_DEEP / TEXT_HI / BORDER) for visual consistency.

### Restart required
Same Python-edit pattern.

---

## 2026-05-06 (round 11) — per-camera mm/pixel calibration → mp4 metadata

### Why
User swapped the See3CAM_CU27's stock M12 lens for a varifocal CB-2812-3MP (2.8–12 mm). With a fixed-FOV lens we could compute mm_per_pixel from spec-sheet HFOV + working distance, but a varifocal's effective focal length depends on wherever the user sets the zoom ring — and the wide end has visible barrel distortion. Spec-sheet geometry is no longer reliable. Each camera also gets set to a different zoom for its angle/distance, so a single global value won't do.

### Approach
Per-camera 2-point calibration. User places a known-length reference (ruler, tape, printed marker) inside the behavior box, clicks "📏 CAL" on the panel, clicks both ends of the reference, types its real length in mm. App computes `mm_per_pixel = ref_length_mm / pixel_distance` and saves it on the panel + into the profile. Optional working-distance field is metadata-only (PixelPaws doesn't compute from it; just nice to record).

At record start, `start_rec` passes the calibration through the recorder/CameraThread chain into FFmpeg as `-metadata key=value` flags, with `+use_metadata_tags` added to `-movflags` so the mp4 muxer actually writes them to the udta atom (default mp4 muxer drops unknown keys). Both the Phase 1 inline-encode path and the Phase 2 subprocess fallback emit the same flags.

### Metadata keys (PixelPaws contract)
- `mm_per_pixel` — float, primary. mm in world / px in recording.
- `working_distance_mm` — optional, float. Camera→subject distance.
- `pixelpaws_calibrated` — "1" sentinel for filtering.
- `pixelpaws_ref_length_mm`, `pixelpaws_ref_pixels` — for audit / re-calibration.

PixelPaws reads with `ffprobe -v 0 -show_entries format_tags -of json file.mp4` — the keys appear under `format.tags`.

### Files touched
- `CameraThread.start_recording` / `_open_ffmpeg_capture` (record branch): metadata flows through `_record_args["metadata"]`; `+use_metadata_tags` gated on metadata presence.
- `Recorder.start` (Phase 1 + Phase 2 + libx264 fallback path): same.
- `CameraPanel`: new `_mm_per_pixel` / `_working_distance_mm` / `_calib_ref_length_mm` / `_calib_ref_pixels` fields, `📏 CAL` header button, `_open_calibration_dialog`, `_update_calib_btn_label`, `_calibration_metadata` helper. Profile get/apply extended.
- New `_CalibClickLabel` + `_CalibrationDialog` classes (model after `_CropPreviewLabel`/`_CropDialog`).

### Calibration scope
`mm_per_pixel` is a property of the recording's pixels — i.e., post-transform. Crop doesn't change pixel density so the calibration survives a crop change. 90°/270° rotation swaps W/H but pixel scale stays the same; flip is invariant. If the user moves the camera, changes the zoom/focus ring, or resizes the recording to a different resolution, recalibrate.

### Restart required
Same Python-edit pattern.

---

## 2026-05-06 (round 10) — recording rate stamped from `_cap_fps`, not measured

### Symptom
After round 9's timescale fix, the mp4 muxer correctly tagged files at the configured rate, but the panel's STOP button still labeled itself "61fps" / "62fps" because `start_rec()` was using `round(self._actual_fps)` (measured live rate) as `rec_fps` — the See3CAMs' MJPEG@100 native input still has slight clock skew, so the *measured* rate Python observes through the bgr24 preview pipe drifts.

### Cause
Session-1's "FPS stamping fix" added `rec_fps = round(self._actual_fps)` because the OLD recording path (legacy two-process pipeline) actually stamped the container with `rec_fps` and could mis-play if the cam delivered at a different rate than configured. Round 4's single-FFmpeg refactor + round 5's `fps={N}` filter + round 9's `-video_track_timescale {N*256}` together now strictly enforce the output rate at the configured value, so passing measured-fps through to `rec_fps` is both unnecessary and wrong — it caused the recording stamp to chase per-cam clock skew.

### Fix
`start_rec` now uses `rec_fps = self._cap_fps` unconditionally. The measured rate is still computed (every 1 second by `CameraThread`) and surfaces as `fps_note` when it differs from `cap_fps` by more than 2 — so a genuinely under-delivering camera (e.g., UYVY auto-exposure capping at 50fps) is still visible in the button label as `~50fps measured`, while normal ~1fps clock skew no longer hijacks the recording rate.

### Restart required
Same Python-edit pattern.

---

## 2026-05-06 (round 9) — per-cam fps drift over long recordings (60 / 61 / 62)

### Symptom
30-second triple-cam recording: CAM_3 came out clean at `60 fps, 60 tbr, 15360 tbn`. CAM_1 was `61 fps, 61 tbr, 15616 tbn`. CAM_2 was `62 fps, 62 tbr, 15872 tbn`. Round-5's `fps=60` filter + `-fps_mode cfr -r 60` were not enough — round-5's smoke test was a 5-second single-cam recording and didn't reveal long-run drift.

### Cause
The mp4 muxer auto-picks the video timescale (`tbn`) as `256 × round(measured_avg_fps)` per file. Each See3CAM's MJPEG@100 input stream has slight clock skew (one cam ticks at ~99.95fps, another at ~100.05fps), so over 30 seconds each cam delivers a different total frame count. The `fps=60` filter resamples to 60 fps based on input timestamps, so its output frame count over 30 s also varies slightly per cam (1788 / 1818 / 1847 frames in the test), and the muxer rounds the avg fps to a different integer for each. With different timescales, frame intervals are stamped 256/15360s vs 256/15616s vs 256/15872s — and players/probes report the divergent rates.

### Fix
Added `-video_track_timescale {r_fps*256}` to the recording branch's output args. With timescale 15360 forced (for 60 fps target), every frame's PTS is an exact integer multiple of 256 ticks regardless of the true input clock skew or measured frame count, and the muxer reports `60 fps, 60 tbr, 15360 tbn` consistently.

The preview branch isn't affected by this issue (no mp4 muxer; bgr24 raw pipe).

### Restart required
Same Python-edit pattern.

---

## 2026-05-06 (round 8) — flip/rotation active during crop dialog gave wrong coords

### Symptom
With `flip` (or rotation) active and the user opening the crop dialog, the rect they dragged matched the visible content in the dialog but the recording captured a *different* region. Recorded mp4 dimensions were correct (W/H matched spinners); only the X/Y placement was wrong.

### Cause
Round-6 cleared the *crop* on the thread for the dialog's lifetime so the user could see the full source frame. It did not clear flip or rotation. The dialog therefore showed a flipped (or rotated) full source frame, and the user's drag was mapped to source coordinates assuming a non-transformed display. The FFmpeg filter chain runs `crop` *before* `hflip` / `transpose`, so the cropped region is taken from raw source — the mirror of what the user saw.

Concretely with `flip` active and source 1280×720, dragging a rect at displayed X=256, W=736 gives source coords `crop=736:?:256:?`. But the user was looking at a horizontally-flipped preview, so the *visually* selected region corresponded to source X = `1280 - 256 - 736 = 288`. Off by 32 columns.

### Fix
`_open_crop_dialog` now clears all three transforms on the thread (flip, rotation, crop) for the dialog's lifetime — not just the crop. The user always edits against the raw source frame regardless of what flip/rotation state the panel is in. Panel state (`_flip_h`, `_rotation`) is not modified — restored to the thread by the post-dialog `_push_transforms_to_thread` along with whatever crop ended up. Same two-respawn cost as before (~1 second total preview gap on dialog open + close).

### Restart required
Same Python-edit story as before — close and re-launch.

---

## 2026-05-06 (round 7) — preview pipe was offset by 1 byte (red-dash artifact)

### Symptom
Vertical red dashes on the right edge of the live preview window. Recordings looked clean (mp4 muxer is independent of the preview pipe), but the panel canvas and the larger preview window both showed mis-colored content with a column of red dashes near the right side.

### Cause
`_open_ffmpeg_capture` probes whether FFmpeg is alive by spawning a thread that reads 1 byte from `proc.stdout` and checks for non-empty output. That byte was discarded — never returned to the caller. The run loop then started reading at byte 1 of the stream instead of byte 0. With BGR24 packing (3 bytes/pixel), every pixel's channels were shifted by 1: Python's `B` channel was actually the original `G`, `G` was `R`, and the new `R` byte came from the *next* pixel's `B`. At the right edge of each row, that "next pixel" was actually pixel `(row+1, col=0)` — so any bright pixel in the **first column** of the cropped frame projected onto the **R channel of the last column** of every row. With dots in column 0 (the See3CAM's left dot row), this produced a vertical column of red dashes at the right edge for every dot's row.

The fix was tractable but not trivial: pipes don't support unread/seek, so we have to stash the probe byte and prepend it to the first real frame read.

### Fix
- `_open_ffmpeg_capture` now attaches the probe byte to the proc as `proc._stash_first_byte = b"\\x?"` instead of discarding it.
- `CameraThread.run`'s outer respawn loop pulls the stash at the start of each FFmpeg session (`_stash = getattr(ff_proc, "_stash_first_byte", b"")`), then on the **first** frame read does `raw = _stash + stdout.read(frame_bytes - 1)` and clears the stash. Subsequent reads in the session do straight `stdout.read(frame_bytes)`.
- The mp4 recording was unaffected (it goes through FFmpeg's own muxer, not Python's pipe), so existing recordings have no red-dash issue — only the preview was broken.

### Restart required
This is a Python change — users need to close and re-launch the script for it to take effect. Ditto for round 6's stale-frame fix (the dialog now waits for a full-source frame before allowing draws).

---

## 2026-05-06 (round 6) — crop dialog used wrong scale during the respawn gap

### Symptom
User dragged a rect on the crop preview, accepted, recorded — and the captured area didn't match what the rect visually showed. Spinner values and recorded mp4 dimensions matched each other (716×698 in their case), so the *crop math* was right; the *source coordinates the user thought they were selecting* were wrong.

### Cause
`_open_crop_dialog` clears any active crop and pushes transforms to the thread (~0.5–1 s respawn) before opening the dialog. During that gap, `panel._last_frame_bgr` was still the stale **cropped** frame from before the respawn. The dialog's 100 ms refresh timer grabbed it and called `_CropPreviewLabel.set_frame(bgr)` — which computed `_scale = scaled.width() / self._src_w` using the *source* width while the pixmap was actually the smaller cropped frame's width. Every `_label_to_src` call then mapped display pixels to source pixels with the wrong ratio. The user dragged a rect that visually framed the desired area on the cropped preview, but `_rect_src` ended up pointing at a different region of the actual source.

### Fix
- `set_frame` now rejects frames whose dimensions don't match `(_src_w, _src_h)`. Cropped fragments from before the respawn are silently ignored — the dialog stays in its "no frame yet" state until a real full-source frame arrives.
- `paintEvent` shows a centered "Waiting for full-frame preview…" message in that state so the user isn't staring at a black box wondering what's wrong. Mouse press was already a no-op while `_frame_size.isEmpty()`, so dragging is naturally disabled until a valid frame paints.
- `set_rect_src` now snaps width/height to even — the painted rect is now exactly the recorded region, with no 1-pixel inclusive/exclusive ambiguity from odd values.

---

## 2026-05-06 (round 5) — recordings were 61 fps instead of 60 (cfr fix)

### Symptom
ffprobe reported `61 fps, 61 tbr, 15616 tbn` on every fresh recording — one frame above the configured 60. The pre-refactor March files showed clean `60 fps, 60 tbr, 15360 tbn`.

### Cause
With MJPEG-native input at 100 fps and `-r 60` only on the output side, FFmpeg was generating output frames at 100/N intervals (variable spacing) and the mp4 muxer picked the avg-rate that best fit the actual timestamps — 61, not 60. The output `-r N` flag re-stamps but doesn't guarantee constant-rate.

### Fix
Added an explicit `fps=N` filter inside the chain (right after the transform chain, before `split`) so frames are emitted at exact `1/N` second intervals before they reach the encoder, plus `-fps_mode cfr` on both `[rec]` and `[prev]` outputs to force constant-rate muxing. The filter handles both upsampling and downsampling. Verified: 5 s test recording at crop 726×704 now probes as `726x704, 60 fps, 60 tbr, 15360 tbn` — identical timebase to the working March recordings.

### Crop coords were correct, just per-panel
User reported "the file used different coords than the dialog". Probing the three latest recordings showed CAM_3 was 726×704 (exactly matching the dialog they showed); CAM_1 and CAM_2 were 750×720 and 790×720 — different crops set independently on each panel. Each panel keeps its own `_crop_rect`, so a dialog opened from one panel doesn't reflect another's setting. No bug here.

---

## 2026-05-06 (round 4) — duplicate device-id bug + resize-from-handles

### Root-cause fix — every same-model camera was getting the same device_id
`enumerate_cameras`'s `_wmi_id_map` built a `{name: device_id}` dict, so all 3 See3CAMs collided on the same key — only the last WMI entry survived, and all three combo entries got the same `device_id`. Saving a multi-cam profile then wrote the same id for every panel, and loading the profile matched every panel to the first combo entry with that id (always position 0). Even my round-2 `apply_settings` name-slot fallback didn't help because device-id matching was succeeding (just incorrectly).

Fix:
- `_wmi_id_map` now returns `{name: [device_id, ...]}` preserving WMI's natural enumeration order.
- `enumerate_cameras` parses FFmpeg's "Alternative name" PnP path lines too (kept in a parallel `ffmpeg_alts` list) and uses them as a fallback device_id when WMI runs out.
- Each ffmpeg-enumerated camera now picks the n-th WMI entry for its name (n = same-name occurrences seen so far). Verified: the three See3CAMs now produce three unique device_ids.
- `_usb_port_label` updated to use the per-cam segment of the USB instance path (e.g. `F6248A2`, `1AAED507`, `1A18659`) instead of the always-identical `&0&0000` tail. Combo dropdown now shows visibly distinct entries: `See3CAM_CU27 [F6248A2]` etc.

### Backstop — collision recovery in `load_profile`
Existing profiles already saved before this fix have duplicate device_ids; on those, `apply_settings` would still set every panel to the same combo index. Added a post-`apply_settings` collision check in `MainWindow.load_profile`: if two or more panels resolved to the same combo entry, force each through `_select_dev_by_name_slot(saved_dev_name)` and update their pin file. Old profiles auto-correct on first load — no manual intervention needed.

### Feature — resize crop rect from edges and corners
`_CropPreviewLabel` now has 8 hit zones (4 corners + 4 edges + interior + exterior). Mouse press on a handle enters `resize_<zone>` mode; drag updates only the touched edge(s) of the rect, with the opposite edge anchored. Inversion through the anchor is allowed (the rect "flips" without breaking). Cursors reflect the zone: `SizeFDiagCursor` / `SizeBDiagCursor` for corners, `SizeVerCursor` / `SizeHorCursor` for edges, `OpenHandCursor` for interior, `CrossCursor` outside. Visible 7×7 px handle squares are painted at the 8 hit-zone centers so users can see where to grab.

---

## 2026-05-06 (round 3) — preview at native cropped size + drag-to-move + clear-crop fix

### Bug — "Clear crop" required disconnect/reconnect to take effect
`set_transforms` used `None` as a sentinel meaning "leave this field unchanged". The dialog's _Clear_ button calls `panel._crop_rect = None` then pushes transforms, so the explicit `crop_rect=None` was silently discarded — the active crop persisted in `_t_crop_rect` until something else triggered a respawn (disconnect/reconnect rebuilds the thread state from scratch). Fix: defaults changed to `flip_h=False, rotation=0, crop_rect=None`; every call now overwrites the full state. The panel always pushes all three values together so no caller is affected by the default change.

### Bug — preview was stretched after a crop was applied
The preview output had `-s {source_w}x{source_h}` so FFmpeg scaled the cropped frame back up to source dimensions before piping bgr24 to Python. Result: a 16:9 canvas displayed a 4:3 cropped scene horizontally stretched. Fix:

- New `CameraThread._effective_size()` returns post-transform dimensions (`(crop_w, crop_h)` after crop, swapped for ±90° rotation, unchanged by flip).
- `_open_ffmpeg_capture` sets `self._eff_w/_eff_h`, computes `frame_bytes = eff_w * eff_h * 3`, and passes `-s {eff_w}x{eff_h}` to the preview output. The filter chain already produces those dimensions, so this just declares the rawvideo container size.
- The run loop's `np.frombuffer(...).reshape((self._eff_h, self._eff_w, 3))` uses the same effective dims; without crop/rotate they equal source dims so the no-transform path is identical to before.
- `CameraPanel._fit_frame` already does aspect-preserving fit, so the canvas now letterboxes correctly instead of stretching.

### Feature — drag-to-move on the crop preview
`_CropPreviewLabel` previously started a fresh draw on every mouse press. Added a mode flag (`idle` / `draw` / `move`):

- Click _inside_ the existing rect → `move` mode: rect translates by the mouse delta, dimensions preserved, clamped to source bounds. Cursor: closed-hand while dragging, open-hand on hover.
- Click _outside_ → `draw` mode (existing behavior): drag-to-resize a new rect. Cursor: crosshair.
- Hover (no button down): cursor switches between open-hand and crosshair as a visual hint.

---

## 2026-05-06 (continued) — device-pinning fallback + visual crop dialog

### Problem 3 — profile load mapped all panels to the cam at position 0
After loading a multi-cam profile, all three panels showed the same camera (the one at dshow index 0). Two bugs compounding:

1. **`_on_devices_found`'s slot-default `_base_name` didn't strip the `[N]` index prefix** that `_on_devices_found` itself adds to combo entries. So three See3CAMs would normalize to `"[0] See3CAM_CU27"`, `"[2] See3CAM_CU27"`, `"[3] See3CAM_CU27"` — three distinct labels — and the dominant-bucket grouping always had only one row per bucket. Panels 1 and 2 then fell through to a `cv_idx == cam_index` rule that grabbed the HP webcam at slot 1.
2. **`apply_settings` had no fallback** when the saved `_device_id` wasn't in the current enumeration. The combo silently stayed at whatever it was showing (index 0).

### Fix 3
- New static helper `CameraPanel._normalize_device_label(display)` properly strips both the leading `[N] ` index prefix and any trailing ` [...]` disambiguator.
- New helper `_select_dev_by_name_slot(name)` picks the Nth combo entry whose normalized name matches `name`, where N is `self.cam_index`. Reused by both `_on_devices_found` and `apply_settings`.
- `apply_settings` now: try exact device-id match → fall back to name-slot match → only persist `_pinned_device_id` when something actually resolved (avoids poisoning `camera_slots.json` with stale IDs from a different machine's profile).

### Feature — visual crop dialog
Replaced the spinner-only crop dialog with a larger one (960×540) that has both a drag-select preview and the numeric inputs:

- New `_CropPreviewLabel` (QLabel subclass) — paints the latest source frame, draws a translucent dim outside the crop rect and an accent-colored outline around it, handles mouse press/move/release for drag selection. `rectChanged(QRect)` emits in source-pixel coords so the dialog spinners can mirror without knowing about display scaling.
- `_CropDialog` runs a 100 ms `QTimer` that pulls `panel._last_frame_bgr` and feeds the preview, so the dialog stays live (helpful for framing). Spinners and visual rect are bidirectionally synced.
- `_open_crop_dialog` temporarily clears any active crop on the thread for the dialog's lifetime so the user sees the full source frame; restores (or replaces) the crop on close.
- Width/height clamped to even (h264 yuv420p), single-step 2 in spinners.
- Required new imports: `QDialog`, `QSpinBox`, `QGridLayout`, `QDialogButtonBox` (Widgets); `QRect`, `QPoint` (Core); `QPainter`, `QPen`, `QBrush` (Gui).

---

## 2026-05-06 — moov-atom fix (every recording was unplayable) + crop/flip/rotate in FFmpeg

### Problem 1 — every mp4 since the session-4 refactor was unplayable
Every recording from 2026-05-05 onward returned `moov atom not found` to ffprobe. Pre-refactor March files played fine. Cause: `CameraThread.run`'s teardown block called `proc.terminate()` (TerminateProcess on Windows = unconditional kill) on the inline-encode FFmpeg, so the muxer never got to write the `moov` trailer that makes the mp4 seekable.

### Fix 1
Added `CameraThread._graceful_stop_ffmpeg(p)` that sends `q\n` on FFmpeg's stdin (its quit-and-finalize command), drains stdout/stderr in background daemons so FFmpeg never blocks on a full pipe, waits up to 5 s for clean exit, then falls back to terminate→kill. Required `stdin=subprocess.PIPE` on the Phase-1 Popen (camsync_precursor.py:482). Replaced `terminate()` at both teardown sites: the inner respawn loop (~line 779) and the thread `stop()` (~line 945). Verified end-to-end with `test_graceful_moov.py` — finalize takes ~0.44 s and the resulting file probes cleanly with duration/codec/fps. Old broken files cannot be auto-recovered (no moov to recover from); `untrunc` with a reference good file is the only path.

### Problem 2 — flip/rotate were preview-only
`_apply_rotation` ran `cv2.flip` / `cv2.rotate` on Python-side frames after FFmpeg's split, so the recording branch (the `[rec]` leg of `filter_complex`) saw raw frames. Recordings ignored the user's flip/rotate buttons.

### Fix 2 — transforms moved into FFmpeg filter graph
- New `CameraThread._t_flip_h`, `_t_rotation`, `_t_crop_rect` fields; new `set_transforms(flip_h, rotation, crop_rect)` setter that triggers a respawn when anything actually changed (~0.5 s preview gap, same mechanism as record toggle).
- New `CameraThread._build_transform_chain()` produces `crop=W:H:X:Y,hflip,transpose=N` (each conditional). Inserted before the `split` in recording mode and as `-vf` in preview-only mode. Output `-s WxH` on the bgr24 preview pipe still scales back to source dimensions so Python's `frame_bytes = w * h * 3` calc stays valid regardless of crop/rotate.
- `_apply_rotation` now skips its cv2 ops when `self.thread.is_phase1()` is True (FFmpeg already did them) — keeps the cv2 path live for Phase-2 OpenCV fallback.
- Order in filter chain is `crop → hflip → transpose` so the crop rect is in source-pixel coordinates regardless of rotation state.

### Feature — crop the recording area
- New ✂ CROP button next to ⇆ FLIP / ↻ rotate.
- Click opens `_CropDialog`: spinners for X / Y / Width / Height, all even-only (h264 yuv420p constraint), clamped to current capture size. "Clear crop" resets to full frame.
- `_crop_rect` (`(x, y, w, h)` or `None`) saved/loaded with the profile JSON alongside `_flip_h` / `_rotation`.
- Recordings get the cropped dimensions natively (smaller files, less encode work). Preview pipe is scaled back to source size by FFmpeg's `-s` so the Python preview-canvas layout doesn't have to handle variable sizes.

### Compute cost (the user asked)
- `crop`: metadata-only in libavfilter — zero per-pixel cost. Smaller frame downstream actually *reduces* encoder work proportional to cropped area.
- `hflip` / `transpose`: per-pixel YUV memcpy. ~270 MB/s per cam at 1280×720@100, ~810 MB/s for 3 cams — ~2 % of typical DDR4 bandwidth. Few % of one CPU core total.
- Filters are placed *before* the `split` so they run once per frame and feed both `[rec]` and `[prev]`, not twice.

---

## 2026-05-05 — `_force_capture_fps` is silent no-op; switch to native-enumeration ingestion

### Diagnosis
Empirically verified that `IAMStreamConfig::SetFormat` from `_force_capture_fps` returns `S_OK` but produces zero observable change on See3CAM_CU27. `GetFormat` returns `UYVY 1920x1080 @ 60fps` before and after the call, in the same-process apartment and in a child process. The driver silently ignores SetFormat on a pin that isn't part of an active filter graph; OBS works because it keeps the filter alive in the streaming graph, but our function releases pStreamCfg immediately and FFmpeg later opens a fresh moniker.

So the entire 60fps story in session 3 was load-bearing on a no-op. The reason recordings *do* land at 60fps is incidental: the device's *default* enumerated mode happens to be UYVY 1080p@60. Whatever resolution/codec the user picks in their profile, the camera was running at 1080p UYVY anyway, and the output stage's `-s WxH` was scaling on the way out.

Verification harness: `diag_setformat_persistence.py` (probe before/after SetFormat in same and child processes), `test_dual_60_locked.py` (dual-cam capture confirming format actually used).

### Fix
Refactored `_open_ffmpeg_capture` (camsync_precursor.py:388) to drop `_force_capture_fps` entirely and instead ingest at a natively enumerated mode, clamping the rate at the output stage with `-r`:
- **Attempt ① — MJPEG @ size** (`-vcodec mjpeg -video_size WxH`). MJPEG is camera-compressed (low USB load). See3CAM enumerates MJPEG at fixed high rates per resolution (1280×720=100, 1920×1080=100, 640×480=120). Output `-r fps` drops to the requested fps.
- **Attempt ② — UYVY @ size + `-framerate fps`** (`-pixel_format uyvy422 -video_size WxH -framerate N`). Works when fps is in the enumerated [Min, Max] range — e.g., UYVY 1280×720 = [50, 80] accepts 60. Lossless transit but ISP auto-exposure may sap the actual delivery rate in low light.
- **Attempt ③ — size only**. Last resort; lets the driver pick whatever default it has.

Both branches of `out_args` now carry `-r {fps}` so the preview pipe is also rate-limited (a 100fps MJPEG input would otherwise dump 1.7× more BGR24 frames into Python than necessary).

### What this changes for users
Recordings still land at 60fps (the requested value), but now reflect the user's profile:
- A 1280×720@60 profile actually captures at 1280×720, not 1920×1080.
- USB bandwidth drops from ~250 MB/s per cam (UYVY 1080p) to ~3.5 MB/s (MJPEG 720p), or ~106 MB/s if attempt ② lands (UYVY 720p). Big drop for the MJPEG path.
- MJPEG at native 100fps + `-r 60` output drops 40% of input frames; the file is exactly 60fps with no buffer overruns. Verified loss-free dual-cam.

### Dead code
`_force_capture_fps()` (camsync_precursor.py:1031, ~270 lines) is now uncalled by app code but kept in place for now — could be revived if the project ever moves to in-process FFmpeg via PyAV (where SetFormat would actually persist).

### Watch out
Rapid-fire SetFormat / open-close cycles via the diagnostic scripts can leave the See3CAM ISP in a stuck state where the device enumerates fine but `Could not run graph` on stream start. USB unplug/replug clears it. Same general failure class as memory's "tile mode" warning.

---

## 2026-05-05 — 30fps cap fix: single-FFmpeg-per-camera (OBS Source Record style)

### Problem
Recordings were capped at ~30fps even though `_force_capture_fps` (IAMStreamConfig::SetFormat in `camsync_precursor.py` line ~752) was already in place. The configured fps was 60, but the encoder couldn't keep up.

### Diagnosis
The recording pipeline was:
```
dshow → FFmpeg₁ → bgr24 → pipe → Python (GIL) → pipe → FFmpeg₂ (NVENC) → mp4
```
At 1920×1080 bgr24, that pipe carries ~370 MB/s **per camera**. With 3 cameras, ~1.1 GB/s funneled through Python — hard cap well below 60fps regardless of what the camera was actually delivering.

OBS Source Record never does this round-trip. It hands the GPU surface straight to NVENC.

### Fix (in progress)
Replace two-process pipeline with **one FFmpeg per camera**:
```
dshow → tee:
  ├─ NVENC → mp4 file        (recording, GPU-resident)
  └─ scale+bgr24 → pipe      (preview only, low-rate)
```
When not recording: simple `dshow → bgr24 → pipe` as before.

### Follow-up fixes (2026-05-05, same session)

**3. 30fps cap persists despite SetFormat → FFmpeg dshow attempt order.**
After fixes 1 + 2, recordings still showed 31fps. Cause: FFmpeg 8's dshow demuxer (a) dropped `-video_codec` from accepted input options entirely, and (b) when `-framerate 60` is passed and the device enumerates `[Min, Max] = [30, 30]` (the See3CAM firmware lie), FFmpeg silently falls back to 30fps AND re-issues `SetFormat` with the unmodified 30fps type — overwriting the 60fps we already pinned via `_force_capture_fps`. OBS's libdshowcapture avoids this by issuing a single `SetFormat` with no follow-up override. Fix: reorder attempts so the **fully-negotiated form (no `-video_size`, no `-framerate`)** is tried first; FFmpeg in that path uses the pin's *current* format (= our 60fps SetFormat) and doesn't re-issue SetFormat. Old size+framerate attempt becomes the last fallback. Verified empirically that FFmpeg 8 rejects `-video_codec` outright (exit code 8 "Unrecognized option").


**1. NVENC unavailable → encoder fallback chain.**
The bundled FFmpeg requires NVENC API 13.0 (NVIDIA driver 570+); this machine has driver 12.2. Replaced `_probe_nvenc` with `_probe_gpu_encoder` which walks `h264_nvenc → h264_qsv → h264_amf → libx264` and picks the first that actually opens an encoder. Verified: this machine selects `h264_qsv` (Intel QuickSync). New `_encoder_args(codec, bitrate)` helper returns the right flag shape per encoder. UI button now shows `GPU NVENC` / `GPU QSV` / `GPU AMF` instead of just `GPU`.

**2. Sliders not working in Phase 1 → COM-based property control.**
The previous design relied on a secondary `cv2.VideoCapture(idx, CAP_DSHOW)` handle for slider changes; this often failed silently when FFmpeg already had the device open. Added `_DshowCameraControl` class which holds `IAMVideoProcAmp` + `IAMCameraControl` interfaces via raw COM vtable calls (same pattern as `_force_capture_fps`). Sliders now route through COM first, fall back to the legacy `prop_cap` only for un-routed props. `_read_props_via_com` mirrors the cv-side getter for SYNC. Verified with Python: opens all 3 cameras, reads accurate min/max/current, `Set(prop, value, manual)` succeeds while FFmpeg is unaware. UI should now reflect slider changes live.

**Verified probes (this machine):**
- `_GPU_CODEC = "h264_qsv"`, reason `"NVIDIA NVENC: driver too old (need 570+)"`
- `_DshowCameraControl` opens all 3 See3CAMs; ranges look correct (Brightness 0-238, Exposure -11 to -1, etc.)

### Plan / status
- [x] **Task 1** — Write this CHANGELOG.
- [x] **Task 2** — `CameraThread._open_ffmpeg_capture` now reads `self._record_args`. When set, emits `filter_complex [0:v]split=2[rec][prev]` with `[rec]→NVENC→mp4` and `[prev]→bgr24→pipe:1`. When None, plain bgr24 pipe (preview-only) — same as before.
- [x] **Task 3** — `CameraThread` now has `start_recording`/`stop_recording`/`is_phase1`. `run()` Phase 1 wraps the capture loop in an outer respawn loop. Inner loop checks `_respawn_pending` under lock and breaks; outer terminates FFmpeg (3s wait+kill), reopens with new args. If encode pipeline fails to launch, falls back to preview-only and emits `camera_error`.
- [x] **Task 4** — `Recorder` is now dual-mode. `attach(cam_thread)` binds to the thread; `start()` checks `cam_thread.is_phase1()`: True → inline (calls `start_recording` on thread), False → legacy subprocess. `write()` is a no-op when inline. `stop()` calls `stop_recording` for inline or shuts down subprocess for legacy.
- [x] **Task 5** — `CameraPanel.start_rec` calls `self.recorder.attach(self.thread)` before `recorder.start()`. Button label includes `inline` tag when in Phase-1 mode.
- [x] **Task 6** — Phase 2 (OpenCV fallback) keeps full legacy behavior via the same `Recorder` class. `CameraThread._phase` is set to `1` or `2` in `run()`; recorder branches off that.

### Key files / line refs (as of 2026-05-05)
- `camsync_precursor.py` — main file (~2165 lines)
  - `CameraThread` ~ line 283
  - `_open_ffmpeg_capture` ~ line 369
  - `CameraThread.run()` ~ line 538 (Phase 1 FFmpeg) and ~ line 637 (Phase 2 OpenCV fallback)
  - `_force_capture_fps` ~ line 752 (already in place)
  - `Recorder` ~ line 1024
  - `CameraPanel.start_rec` ~ line 2163
  - `CameraPanel._toggle_cam_rec` ~ line 2122

### Notes for next session
- See3CAM_CU27 ISP can lock into tile-diagnostic mode if SetFormat is followed by partial graph teardown — current `_force_capture_fps` uses raw vtable calls (no graph), so safe.
- NVENC fallback to libx264 is already wired in old `Recorder.start()`; keep that fallback path in the new code.
- `_actual_fps` measurement (Gap 3 from session 2) drives the deferred record-start logic — keep that path intact.
