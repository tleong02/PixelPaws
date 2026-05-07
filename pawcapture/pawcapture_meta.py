"""
pawcapture_meta — read PawCapture calibration + session metadata.

Drop this single file into PixelPaws (no install needed) and you get:

    from pawcapture_meta import read_calibration, read_session_manifest

    cal = read_calibration("CAM_1_20260506_151651.mp4")
    if cal:
        print(cal["mm_per_pixel"])

    sess = read_session_manifest("session_20260506_151651.json")
    if sess:
        for cam in sess["cameras"]:
            print(cam["label"], cam["mm_per_pixel"], cam["file"])

The mp4 reader uses ffprobe under the hood — point it at your ffprobe
binary if it isn't on PATH:

    cal = read_calibration("file.mp4", ffprobe="C:/path/to/ffprobe.exe")

The manifest reader is pure stdlib (json), no ffprobe needed.

Both readers return None when no calibration / manifest is present, so
callers can write `if cal := read_calibration(p): ...` and skip the
file gracefully.

Compatible with Python 3.8+. No third-party deps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

# Tag keys camsync writes into the mp4 udta atom. PixelPaws should treat
# `mm_per_pixel` as the source of truth; the rest is audit / context.
CALIBRATION_KEYS = (
    "mm_per_pixel",
    "working_distance_mm",
    "pixelpaws_calibrated",
    "pixelpaws_ref_length_mm",
    "pixelpaws_ref_pixels",
    "pawcapture_version",
)

_FLOAT_KEYS = {
    "mm_per_pixel",
    "working_distance_mm",
    "pixelpaws_ref_length_mm",
    "pixelpaws_ref_pixels",
}


def _resolve_ffprobe(explicit: Optional[PathLike]) -> Optional[str]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        return None
    # Try PATH, then look next to a sibling ffmpeg if one is available.
    found = shutil.which("ffprobe")
    if found:
        return found
    sibling = shutil.which("ffmpeg")
    if sibling:
        cand = Path(sibling).with_name("ffprobe.exe" if sibling.lower().endswith(".exe")
                                        else "ffprobe")
        if cand.exists():
            return str(cand)
    return None


def read_calibration(
    mp4_path: PathLike,
    ffprobe: Optional[PathLike] = None,
) -> Optional[Dict[str, Any]]:
    """Return the PawCapture calibration tags from an mp4, or None if the
    file isn't calibrated / can't be read.

    Returns a dict with float-typed numeric fields (mm_per_pixel etc) and
    string fields for the rest. Only present keys are included; absent
    tags are simply omitted.
    """
    probe = _resolve_ffprobe(ffprobe)
    if probe is None:
        raise RuntimeError(
            "ffprobe not found. Install FFmpeg or pass ffprobe=... explicitly."
        )
    cmd = [
        probe, "-v", "error",
        "-show_entries", "format_tags",
        "-of", "json",
        str(mp4_path),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            creationflags=0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0,
        )
    except subprocess.CalledProcessError:
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    tags = (data.get("format") or {}).get("tags") or {}
    if not tags:
        return None
    result: Dict[str, Any] = {}
    for k in CALIBRATION_KEYS:
        if k in tags:
            v = tags[k]
            if k in _FLOAT_KEYS:
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    result[k] = v
            else:
                result[k] = v
    if "mm_per_pixel" not in result:
        # No calibration → not interesting to PixelPaws.
        return None
    return result


def read_session_manifest(json_path: PathLike) -> Optional[Dict[str, Any]]:
    """Return the parsed session manifest dict, or None on parse failure /
    schema mismatch. Validates the schema header so an unrelated JSON
    file doesn't slip through."""
    try:
        text = Path(json_path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    schema = data.get("schema", "")
    if not isinstance(schema, str) or not schema.startswith("pawcapture.session/"):
        return None
    if not isinstance(data.get("cameras"), list):
        return None
    return data


def find_session_for_video(mp4_path: PathLike) -> Optional[Path]:
    """Locate the session manifest that lists `mp4_path`, if any. Looks in
    the file's directory for `session_*.json`. Returns the manifest path
    or None.

    Useful when PixelPaws is given an mp4 and wants to grab the broader
    context (sync marks, the other cameras' files, profile name)."""
    p = Path(mp4_path)
    parent = p.parent
    target = str(p.resolve())
    for cand in sorted(parent.glob("session_*.json")):
        sess = read_session_manifest(cand)
        if not sess:
            continue
        for cam in sess.get("cameras", []):
            f = cam.get("file")
            if f and Path(f).resolve() == Path(target):
                return cand
    return None


__all__ = [
    "read_calibration",
    "read_session_manifest",
    "find_session_for_video",
    "CALIBRATION_KEYS",
]
