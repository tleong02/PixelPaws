#!/usr/bin/env python3
"""
PawCapture — Multi-Camera Controller for PixelPaws
Clean rebuild: OpenCV-only capture (single handle = props always work).
FFmpeg used only for recording output. No secondary handle needed.
"""
import sys, json, subprocess, threading, time, shutil, re
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# ── Locate FFmpeg ──────────────────────────────────────────────────────────────
def _find_ffmpeg() -> str:
    for p in [
        Path(sys.executable).parent / "ffmpeg.exe",
        Path(__file__).parent / "ffmpeg.exe",
        Path(__file__).parent / "ffmpeg" / "ffmpeg.exe",
    ]:
        if p.exists():
            return str(p)
    return "ffmpeg"

FFMPEG = _find_ffmpeg()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QScrollArea, QFrame,
    QStatusBar, QInputDialog, QMessageBox, QFileDialog, QLineEdit,
    QSizePolicy, QScrollBar, QDoubleSpinBox, QStackedWidget,
    QDialog, QSpinBox, QGridLayout, QDialogButtonBox, QTextBrowser,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette, QCursor, QPainter, QPen, QBrush

# ── Constants ──────────────────────────────────────────────────────────────────
PAWCAPTURE_VERSION = "1.0.0"
PREVIEW_W, PREVIEW_H = 360, 203
DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 1280, 720, 60

PROFILES_DIR     = Path.home() / "PawCapture" / "profiles"
RECORDINGS_DIR   = Path.home() / "PawCapture" / "recordings"
LOGS_DIR         = Path.home() / "PawCapture" / "logs"
OFRS_CONFIG_FILE = Path.home() / "PawCapture" / "ofrs_config.json"

PROBE_RESOLUTIONS = [
    (640,360),(640,480),(800,600),(1024,576),
    (1280,720),(1280,960),(1920,1080),(2560,1440),(3840,2160),
]
# See3CAM_CU27: MJPEG Full HD @ up to 100fps. Driver always lies and reports 30fps.
PROBE_FPS = [15, 24, 25, 30, 50, 60, 75, 90, 100]

PROPS = {
    "Brightness":    (cv2.CAP_PROP_BRIGHTNESS,              0,   255,  128,   1),
    "Contrast":      (cv2.CAP_PROP_CONTRAST,                0,   255,  128,   1),
    "Saturation":    (cv2.CAP_PROP_SATURATION,              0,   255,  128,   1),
    "Sharpness":     (cv2.CAP_PROP_SHARPNESS,               0,   255,  128,   1),
    "Gain":          (cv2.CAP_PROP_GAIN,                    0,   255,    0,   1),
    "Gamma":         (cv2.CAP_PROP_GAMMA,                 100,   500,  220,   1),
    "Exposure":      (cv2.CAP_PROP_EXPOSURE,              -15,     0,   -6,   1),
    "White Balance": (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 2800,  6500, 4600, 100),
}
AUTO_CTRL = {
    "Exposure":      (cv2.CAP_PROP_AUTO_EXPOSURE, 3, 1),
    "White Balance": (cv2.CAP_PROP_AUTO_WB,        1, 0),
}

ACCENT  = "#FF7A00"; ACCENT2 = "#4A9EFF"; DANGER  = "#FF3B30"
SUCCESS = "#30D158"; WARN    = "#FFD60A"
BG_DEEP = "#0D0D18"; BG_CARD = "#16162A"; BG_MID  = "#1A1A2E"
BORDER  = "#2A2A45"; TEXT_DIM= "#666688"; TEXT_MED= "#9999BB"; TEXT_HI = "#DDDDEE"
FONT    = "Consolas, Courier New, monospace"

# ── Device enumeration ─────────────────────────────────────────────────────────
CONFIG_FILE = Path.home() / "PawCapture" / "camera_slots.json"

def _load_slots() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_slots(slots: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(slots, indent=2))

# ── RWD OFRS pairing ──────────────────────────────────────────────────────────
# OFRS (RWD's fiber photometry app) writes per-recording session folders named
# `YYYY_MM_DD-HH_MM_SS` containing Events.csv (`TimeStamp,Name,State`) where
# TimeStamps are *session-relative ms* (same clock as Fluorescence.csv, which
# starts at 0.000). The folder name is the session wall-clock start. Both
# facts are used by `_align_ofrs_events` to merge events into the PawCapture
# session manifest as MARKs.
OFRS_SESSION_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")

def _load_ofrs_config() -> dict:
    try:
        if OFRS_CONFIG_FILE.exists():
            return json.loads(OFRS_CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_ofrs_config(cfg: dict):
    OFRS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFRS_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def _scan_ofrs_sessions(root) -> set:
    """Walk `root` recursively and return a set of resolved Paths to OFRS
    session folders (folder name matches OFRS_SESSION_RE and contains
    Events.csv). Empty set on missing/unreadable root. We anchor on the
    Events.csv file rather than on directory names alone so a half-written
    folder without Events.csv yet doesn't get picked up as a "new" session."""
    if not root:
        return set()
    p = Path(root)
    if not p.exists():
        return set()
    out = set()
    try:
        for events_csv in p.rglob("Events.csv"):
            d = events_csv.parent
            if OFRS_SESSION_RE.match(d.name):
                try:
                    out.add(d.resolve())
                except OSError:
                    out.add(d)
    except OSError:
        pass
    return out

def _parse_ofrs_session_start(session_dir):
    """OFRS folder name → naive local datetime. Returns None on bad shape."""
    name = Path(session_dir).name
    if not OFRS_SESSION_RE.match(name):
        return None
    try:
        return datetime.strptime(name, "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        return None

def _read_ofrs_events(session_dir):
    """Parse Events.csv into [{'ts_ms': float, 'name': str, 'state': int|str}].
    Returns [] on missing/unreadable file. Header row is skipped."""
    f = Path(session_dir) / "Events.csv"
    if not f.exists():
        return []
    rows = []
    try:
        text = f.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            ts_ms = float(parts[0])
        except ValueError:
            continue
        try:
            state = int(parts[2])
        except ValueError:
            state = parts[2]
        rows.append({"ts_ms": ts_ms, "name": parts[1], "state": state})
    return rows

def _align_ofrs_events(session_dir, pawcapture_start):
    """Read OFRS Events.csv and translate each row into a (t_seconds, label,
    wall_time) mark relative to `pawcapture_start` (a datetime). Returns
    list of mark dicts with the same shape as MainWindow._session_marks."""
    ofrs_start = _parse_ofrs_session_start(session_dir)
    events     = _read_ofrs_events(session_dir)
    if ofrs_start is None or not events:
        return []
    base_offset_s = (ofrs_start - pawcapture_start).total_seconds()
    marks = []
    for ev in events:
        t = base_offset_s + ev["ts_ms"] / 1000.0
        # Wall time of the event = OFRS folder start + ts. Round to ms.
        wall = (ofrs_start.timestamp() + ev["ts_ms"] / 1000.0)
        wall_iso = datetime.fromtimestamp(wall).isoformat(timespec="milliseconds")
        marks.append({
            "t_seconds": round(t, 3),
            "label":     f"OFRS:{ev['name']}={ev['state']}",
            "wall_time": wall_iso,
            "source":    "ofrs",
        })
    return marks

def _usb_port_label(device_id: str) -> str:
    """Short, human-readable disambiguator from a USB DeviceID or PnP path.
    Picks the per-cam segment of the instance path (e.g., 'F6248A2' from
    'USB\\VID_2560&PID_C12C&MI_00\\7&F6248A2&0&0000') so multiple cameras of
    the same model get visibly distinct combo labels."""
    sep = "\\" if "\\" in device_id else "/"
    parts = device_id.split(sep)
    if len(parts) >= 2:
        port_parts = parts[-1].split("&")
        # Most USB instance paths end with "&0&0000" — those tail segments
        # are the same on every device. The second segment carries the
        # unique-per-cam hash, so prefer it; fall back to the original
        # tail-2 join if the path doesn't fit the expected shape.
        if len(port_parts) >= 2 and port_parts[1]:
            return port_parts[1][:12]
        if len(port_parts) >= 2:
            return ("&".join(port_parts[-2:]))[:12]
        return parts[-1][:12]
    return device_id[-8:]

def enumerate_cameras() -> list:
    """
    Returns [(cv_index, display_name, device_id), ...]
    device_id is stable per USB port location.
    """
    import re

    # WMI gives a stable per-USB-port DeviceID, but only one entry per Name
    # (dict keys collide for identical model names). To handle 3× See3CAMs
    # we need *all* IDs preserved, so wmi_map values are LISTS in WMI's
    # natural enumeration order — paired against FFmpeg's enumeration order
    # by the (name, occurrence) index below.
    def _wmi_id_map():
        try:
            r = subprocess.run(
                ["wmic", "path", "Win32_PnPEntity", "where",
                 "PNPClass='Camera' or PNPClass='Image'",
                 "get", "Name,DeviceID", "/format:csv"],
                capture_output=True, timeout=6,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            out = r.stdout.decode(errors="replace")
            # wmic /format:csv on modern Windows HTML-escapes ampersands in
            # DeviceIDs (USB DeviceIDs always contain "&", e.g.
            # "USB\VID_xxxx&PID_xxxx&...").  Without this decode, slot
            # pinning to camera_slots.json gets corrupted IDs that never
            # match on next launch.
            import html as _html
            out = _html.unescape(out)
            result = {}
            for line in out.splitlines():
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    device_id = parts[1].strip()
                    name      = ",".join(parts[2:]).strip()
                    if device_id and name and device_id != "DeviceID":
                        result.setdefault(name, []).append(device_id)
            return result
        except Exception:
            return {}

    wmi_map = _wmi_id_map()

    # Primary enumeration: FFmpeg `-list_devices`. This is a passive query of
    # the DirectShow registry — it does NOT open any device, so it works
    # safely while other cameras are streaming. We capture the "Alternative
    # name" PnP path for each device too — it's unique per physical USB
    # port and so doubles as a stable device_id when WMI doesn't have a
    # matching entry (or has fewer entries than dshow enumerates).
    ffmpeg_names = []
    ffmpeg_alts  = []   # parallel list: PnP path per ffmpeg_names entry
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, timeout=8,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        out = (r.stdout + r.stderr).decode(errors="replace")
        cur_idx = -1
        for line in out.splitlines():
            m = re.search(r'"([^"]+)"\s+\(video\)', line)
            if m:
                ffmpeg_names.append(m.group(1).strip())
                ffmpeg_alts.append("")
                cur_idx = len(ffmpeg_names) - 1
                continue
            if cur_idx >= 0:
                m2 = re.search(r'Alternative name\s+"([^"]+)"', line)
                if m2 and not ffmpeg_alts[cur_idx]:
                    ffmpeg_alts[cur_idx] = m2.group(1).strip()
    except Exception:
        pass

    cameras = []
    name_seen = {}
    for idx, base in enumerate(ffmpeg_names):
        # FFmpeg list_devices order matches DirectShow's ICreateDevEnum order,
        # which matches what cv2.VideoCapture(idx, CAP_DSHOW) would target.
        # So idx in this list IS the cv index.
        n = name_seen.get(base, 0)
        name_seen[base] = n + 1

        # Pick a unique-per-physical-cam device_id, in priority order:
        #   1. WMI DeviceID (stable across USB ports for the same physical
        #      camera if it carries a USB serial). Use the n-th entry for
        #      this name to handle multiple cameras of the same model.
        #   2. FFmpeg's Alternative name (PnP path). Unique per port —
        #      changes if you reconnect to a different USB port, but at
        #      least uniquely identifies physical cams within a session.
        #   3. Fallback: synthetic "FALLBACK_INDEX_<n>".
        device_id = ""
        suffix = ""
        wmi_ids = wmi_map.get(base, [])
        if n < len(wmi_ids):
            device_id = wmi_ids[n]
            if len(wmi_ids) > 1 or n > 0:
                suffix = f" [{_usb_port_label(device_id)}]"
        elif ffmpeg_alts[idx]:
            device_id = ffmpeg_alts[idx]
            if n > 0:
                suffix = f" [{_usb_port_label(device_id)}]"
        else:
            device_id = f"FALLBACK_INDEX_{idx}"
            if n > 0:
                suffix = f" [{n}]"
        cameras.append((idx, base + suffix, device_id))

    # If FFmpeg enumeration returned nothing (no FFmpeg, etc.), fall back to
    # a SINGLE cv2 probe pass — but only if no cameras are currently open
    # by another process. Skip the fallback entirely on a refresh while
    # something is streaming, since opening busy cameras can disrupt them.
    if not cameras:
        try:
            for idx in range(10):
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    opened = cap.isOpened()
                    cap.release()
                except Exception:
                    opened = False
                    break
                if not opened:
                    break
                cameras.append((idx, f"Camera {idx}", f"FALLBACK_INDEX_{idx}"))
        except Exception:
            pass

    return cameras


class DeviceEnumThread(QThread):
    result = pyqtSignal(list)
    def run(self): self.result.emit(enumerate_cameras())


class ProbeThread(QThread):
    """Probe supported resolutions and FPS for a camera index.

    Strategy (in priority order):
      1. FFmpeg -list_options via DirectShow — reads from IAMStreamConfig directly,
         so it sees the true hardware caps even when the driver lies via OpenCV.
         CameraThread._list_dshow_options() is reused (already used by DIAGNOSE).
      2. OpenCV CAP_DSHOW fallback — used only when FFmpeg is unavailable or
         returns no results.  The fps tolerance check is dropped because
         cap.get(CAP_PROP_FPS) always returns 30 on the See3CAM_CU27 regardless
         of the mode requested; resolution negotiation is the only reliable signal.
    """
    result = pyqtSignal(dict)
    error  = pyqtSignal(str)

    def __init__(self, index, device_name=""):
        super().__init__()
        self.index       = index
        self.device_name = device_name

    def run(self):
        modes = {}

        # ── Strategy 1: FFmpeg -list_options ──────────────────────────────────
        if _FFMPEG_OK and self.device_name:
            try:
                # CameraThread is defined later in the file but is already
                # constructed by the time ProbeThread runs.
                caps, _ = CameraThread._list_dshow_options(self.device_name)
                if caps:
                    # Build {(w,h): best_fps_max} from all reported media types.
                    best = {}
                    for cap in caps:
                        w, h    = cap["width"], cap["height"]
                        fps_max = cap["fps_max"]
                        if fps_max > 0:
                            best[(w, h)] = max(best.get((w, h), 0), fps_max)
                    # Populate modes: include every PROBE_FPS value ≤ fps_max so
                    # the UI shows all selectable rates, not just the ceiling.
                    for (w, h), fps_max in best.items():
                        fps_list = [f for f in PROBE_FPS if f <= fps_max]
                        if not fps_list:
                            fps_list = [min(PROBE_FPS, key=lambda f: abs(f - fps_max))]
                        modes[(w, h)] = fps_list
            except Exception:
                pass  # fall through to OpenCV strategy

        # ── Strategy 2: OpenCV fallback ───────────────────────────────────────
        if not modes:
            try:
                cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    self.error.emit(f"Cannot open camera {self.index}"); return
                MJPEG = cv2.VideoWriter_fourcc(*"MJPG")
                for w, h in PROBE_RESOLUTIONS:
                    cap.set(cv2.CAP_PROP_FOURCC,       MJPEG)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS,          DEFAULT_FPS)
                    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Accept the resolution if the driver negotiated it correctly.
                    # The fps tolerance check is intentionally omitted: the
                    # See3CAM_CU27 driver always reports 30fps via CAP_PROP_FPS
                    # even for modes it physically runs at 60/100fps, so checking
                    # af >= fps * 0.8 would produce false negatives for every fps
                    # above 37.  We show the full PROBE_FPS list for each accepted
                    # resolution; the user and the recording stamper use _actual_fps
                    # as the real-time ground truth.
                    if aw == w and ah == h:
                        modes[(w, h)] = list(PROBE_FPS)
                cap.release()
            except Exception as e:
                if not modes:
                    self.error.emit(str(e)); return

        if not modes:
            modes = {(DEFAULT_W, DEFAULT_H): list(PROBE_FPS)}
        self.result.emit(modes)


# ── Camera capture thread ──────────────────────────────────────────────────────
# Phase 1 — FFmpeg capture: genuine fps (ignores driver's reported cap), frames
#   from stdout pipe. Tries a secondary OpenCV handle for property sliders.
#   Some cameras allow it alongside FFmpeg; if not, sliders queue silently.
# Phase 2 — OpenCV fallback: single handle, sliders always work, fps limited
#   by driver metadata (typically 30fps on See3CAM even if you ask for 60).

class CameraThread(QThread):
    frame_ready  = pyqtSignal(object)
    camera_error = pyqtSignal(str)
    connected    = pyqtSignal()
    props_read   = pyqtSignal(dict)   # {name: actual_value}
    ranges_read  = pyqtSignal(dict)   # {name: (mn, mx, current)}
    fps_measured = pyqtSignal(float)

    def __init__(self, index, initial_props, initial_auto,
                 width=DEFAULT_W, height=DEFAULT_H, fps=DEFAULT_FPS,
                 device_name="", ff_instance=0):
        super().__init__()
        self.index          = index
        self._initial_props = initial_props.copy()
        self._initial_auto  = initial_auto.copy()
        self._width         = width
        self._height        = height
        self._fps           = fps
        self._device_name   = device_name
        self._ff_instance   = ff_instance
        self._queue         = []
        self._lock          = threading.Lock()
        self.running        = False
        self.cap            = None
        self._sync_requested = False
        # Reported after connect
        self._using_mjpeg    = False
        self._negotiated_fps = 0.0
        self._negotiated_w   = width
        self._negotiated_h   = height
        self._ffmpeg_proc    = None   # always None — kept for UI compat
        self.recorder_ref    = None   # set by CameraPanel for Phase-2 fallback recording
        # Inline-encode recording (OBS Source Record style): when set, _open_ffmpeg_capture
        # builds a tee output (NVENC→file + bgr24 preview pipe) instead of preview-only.
        # Mutated only via start_recording/stop_recording from the Qt thread.
        self._record_args      = None   # dict or None
        self._respawn_pending  = False  # set when record state changed; capture loop re-opens FFmpeg
        self._phase            = 0      # 1 = FFmpeg capture, 2 = OpenCV fallback (set in run())
        # Image transforms applied in FFmpeg's filter graph (so they affect
        # both the recorded mp4 and the preview pipe). Toggling any of these
        # triggers a respawn (~0.5 s preview gap) — same mechanism record
        # toggle uses. Crop rect is (x, y, w, h) in source-pixel coordinates,
        # or None for no crop.
        self._t_flip_h         = False
        self._t_rotation       = 0      # 0 / 90 / 180 / 270
        self._t_crop_rect      = None   # (x, y, w, h) or None
        # Effective frame dimensions after the transform chain. Set by
        # _open_ffmpeg_capture and used by the run loop's reshape so the
        # preview pipe carries native cropped/rotated frames (no stretching
        # back to source size, no aspect distortion in the canvas).
        self._eff_w            = width
        self._eff_h            = height

    # ── FFmpeg dshow helpers (used only for DIAGNOSE button) ───────────────────
    @staticmethod
    def _list_dshow_devices():
        import re, collections
        result = collections.defaultdict(list)
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True, timeout=8,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            output = (r.stdout + r.stderr).decode(errors="replace")
            current_name = None
            for line in output.splitlines():
                m = re.search(r'"([^"]+)"\s+\(video\)', line)
                if m:
                    current_name = m.group(1).strip(); continue
                m = re.search(r'Alternative name\s+"(@device[^"]+)"', line)
                if m and current_name:
                    result[current_name].append(m.group(1))
        except Exception:
            pass
        return dict(result)

    @staticmethod
    def _list_dshow_options(dshow_name):
        import re
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-f", "dshow",
                 "-list_options", "true", "-i", f"video={dshow_name}"],
                capture_output=True, timeout=6,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            output = (r.stdout + r.stderr).decode(errors="replace")
            caps = []
            for line in output.splitlines():
                m = re.search(
                    r'(vcodec|pixel_format)=(\S+)\s+min s=(\d+)x(\d+) fps=[\d.]+\s+max s=(\d+)x(\d+) fps=([\d.]+)',
                    line)
                if m:
                    fmt_type, fmt_val, _, _, w2, h2, fps2 = m.groups()
                    caps.append({
                        "vcodec"      : fmt_val if fmt_type == "vcodec" else None,
                        "pixel_format": fmt_val if fmt_type == "pixel_format" else None,
                        "width": int(w2), "height": int(h2),
                        "fps_max": float(fps2),
                    })
            return caps, output
        except Exception as e:
            return [], str(e)

    # ── FFmpeg capture ─────────────────────────────────────────────────────────
    def _open_ffmpeg_capture(self):
        """
        Launch FFmpeg as frame source via DirectShow.
        Captures at a natively enumerated mode; the output stage uses -r to
        clamp recording (and preview) to the requested fps.
        Returns (proc, frame_bytes, err_str) or (None, 0, err_str) on failure.
        """
        if not self._device_name:
            return None, 0, "no device name"
        import queue as _queue

        w, h, fps   = self._width, self._height, self._fps
        # Effective (post-transform) dimensions. Used to size the preview
        # pipe and Python's reshape so cropped/rotated frames arrive at
        # native size — no FFmpeg-side rescale, no canvas stretching.
        eff_w, eff_h = self._effective_size()
        self._eff_w, self._eff_h = eff_w, eff_h
        frame_bytes = eff_w * eff_h * 3

        # Resolve PnP path for duplicate-camera disambiguation
        dev_map  = self._list_dshow_devices()
        pnp_list = dev_map.get(self._device_name, [])
        if self._ff_instance < len(pnp_list):
            dshow_name = pnp_list[self._ff_instance]
        else:
            dshow_name = self._device_name

        # Preview pipe size = effective size after transforms. The filter
        # chain already produces these dimensions, so -s here just declares
        # the output container size to FFmpeg's rawvideo encoder.
        out_size = ["-s", f"{eff_w}x{eff_h}"]

        # Image-transform chain (crop/hflip/transpose). Applied once before
        # the split so it lands on both [rec] and [prev].
        tx_chain = self._build_transform_chain()  # "" or e.g. "crop=800:600:0:0,hflip"

        # ── Build output args ────────────────────────────────────────────────
        # Two shapes, picked at FFmpeg-launch time based on self._record_args:
        #   • record_args is None  → preview-only:  -f rawvideo bgr24 → pipe:1
        #   • record_args is dict  → inline encode: filter_complex split,
        #       [rec] → NVENC/libx264 → mp4 file, [prev] → bgr24 → pipe:1
        # The encode branch keeps frames on the GPU end-to-end (dshow MJPEG
        # decode is CPU but the bgr24 round-trip through Python is gone),
        # which is the OBS Source Record approach.
        record_args = self._record_args
        if record_args:
            r_path = str(record_args["path"])
            r_fps  = int(record_args["fps"])
            r_br   = float(record_args["bitrate"])
            r_codec = record_args["codec"]
            r_meta = record_args.get("metadata") or {}
            # `+use_metadata_tags` makes the mp4 muxer write arbitrary
            # `-metadata key=value` pairs into the file's udta atom (where
            # the default mp4 muxer would silently drop unknown keys).
            # Only enable it when there's metadata to emit so existing
            # recordings keep the same container shape.
            movflags = "+faststart+use_metadata_tags" if r_meta else "+faststart"
            common_enc = ["-pix_fmt", "yuv420p", "-profile:v", "main",
                          "-movflags", movflags]
            meta_args = []
            for k, v in r_meta.items():
                meta_args += ["-metadata", f"{k}={v}"]
            enc_extra = _encoder_args(r_codec, r_br)
            # `fps=N` resamples input frames to N output frames per input
            # second. `-video_track_timescale {N*256}` then forces the mp4
            # muxer to use a fixed timescale (15360 for 60fps), so every
            # frame's PTS is an exact integer multiple of `N*256/N = 256`
            # ticks. Without the explicit timescale, the muxer auto-picks
            # `256 * round(measured_avg_fps)` — and per-cam clock skew in
            # the MJPEG@100 input stream causes each cam to round to a
            # different integer (60, 61, 62), producing differently-timebased
            # files even though the filter graph is identical.
            tscale = r_fps * 256
            chain_pieces = [tx_chain] if tx_chain else []
            chain_pieces.append(f"fps={r_fps}")
            chain_str = ",".join(chain_pieces)
            filter_complex = f"[0:v]{chain_str},split=2[rec][prev]"
            out_args = (
                ["-filter_complex", filter_complex,
                 "-map", "[rec]", "-c:v", r_codec,
                 "-fps_mode", "cfr", "-r", str(r_fps),
                 "-video_track_timescale", str(tscale)]
                + enc_extra + common_enc + meta_args + [r_path,
                 "-map", "[prev]", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-fps_mode", "cfr", "-r", str(r_fps)]
                + out_size + ["pipe:1"]
            )
        else:
            # Preview-only mode: rate-limit input to fps too.
            chain_pieces = [tx_chain] if tx_chain else []
            chain_pieces.append(f"fps={fps}")
            vf_args = ["-vf", ",".join(chain_pieces)]
            out_args = (
                vf_args
                + ["-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-fps_mode", "cfr", "-r", str(fps)]
                + out_size + ["pipe:1"]
            )

        # IAMStreamConfig::SetFormat on the See3CAM is a silent no-op when
        # called from a transient COM apartment — the driver only honors it
        # while an active filter graph is holding the pin.  Verified
        # empirically (diag_setformat_persistence.py, 2026-05-05): GetFormat
        # returned UYVY 1920x1080@60 before AND after `_force_capture_fps`,
        # in same-process and child-process apartments.  So we don't try to
        # pre-configure; we let FFmpeg pick a natively enumerated mode and
        # clamp the rate at the output stage with `-r` (added to out_args).
        #
        # Ingestion order — every attempt asks for the requested resolution.
        # Camera-side fps is whatever's enumerated for that mode:
        #   ① MJPEG @ size — primary path. The See3CAM enumerates MJPEG at
        #     fixed high rates per resolution (1280x720=100, 1920x1080=100,
        #     640x480=120). Camera-compressed → low USB load. Output -r
        #     drops to the requested fps.
        #   ② UYVY @ size with -framerate fps — works when fps is in the
        #     enumerated [Min, Max] (e.g. UYVY 1280x720 = [50, 80] accepts
        #     60). Lossless transit but ISP auto-exposure may drop actual
        #     delivery in low light; user can lock exposure via the slider.
        #   ③ Size only — last resort, picks whatever the driver hands us.
        attempts = [
            [FFMPEG, "-hide_banner", "-loglevel", "info",
             "-f", "dshow", "-vcodec", "mjpeg",
             "-video_size", f"{w}x{h}",
             "-rtbufsize", "702000k", "-i", f"video={dshow_name}"] + out_args,
            [FFMPEG, "-hide_banner", "-loglevel", "info",
             "-f", "dshow", "-pixel_format", "uyvy422",
             "-video_size", f"{w}x{h}", "-framerate", str(fps),
             "-rtbufsize", "702000k", "-i", f"video={dshow_name}"] + out_args,
            [FFMPEG, "-hide_banner", "-loglevel", "info",
             "-f", "dshow",
             "-video_size", f"{w}x{h}",
             "-rtbufsize", "702000k", "-i", f"video={dshow_name}"] + out_args,
        ]

        last_err = ""
        for cmd in attempts:
            try:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=0x08000000 if sys.platform == "win32" else 0,
                )
                # Probe via a 1-byte stdout read to confirm FFmpeg is producing
                # frames (not silently stuck in init). The byte MUST be stashed
                # and prepended to the first frame read by the run loop —
                # otherwise every preview frame is offset by 1 byte for this
                # FFmpeg's lifetime, which channel-shifts BGR24 (the last
                # pixel of each row "borrows" its R from row+1 col 0, painting
                # a column of red dashes on the right edge of the preview).
                q = _queue.Queue()
                def _r(p=proc, qu=q):
                    try: qu.put(p.stdout.read(1))
                    except: qu.put(b"")
                t = threading.Thread(target=_r, daemon=True)
                t.start(); t.join(timeout=8.0)
                first_byte = q.get() if not q.empty() else b""
                if first_byte != b"" and proc.poll() is None:
                    proc._stash_first_byte = first_byte
                    # drain stderr in background
                    threading.Thread(target=lambda p=proc: p.stderr.read(),
                                     daemon=True).start()
                    return proc, frame_bytes, ""
                err = proc.stderr.read(2000).decode(errors="replace").strip()
                try: proc.terminate()
                except: pass
                last_err = err or last_err
            except Exception as e:
                last_err = str(e)

        return None, 0, last_err

    def _graceful_stop_ffmpeg(self, p):
        """Ask FFmpeg to exit cleanly so the mp4 muxer can write its trailer
        (the moov atom that makes the file playable). Sends 'q' on stdin,
        drains stdout/stderr in background so FFmpeg doesn't block on a full
        pipe, waits up to 5s, then falls back to terminate→kill."""
        if p is None or p.poll() is not None:
            return
        # Background drainers — keep FFmpeg unblocked while it finalizes.
        def _drain(pipe):
            try:
                while pipe.read(65536):
                    pass
            except Exception:
                pass
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                threading.Thread(target=_drain, args=(pipe,), daemon=True).start()
        # 'q' is FFmpeg's quit-and-finalize command.
        try:
            if p.stdin is not None and not p.stdin.closed:
                p.stdin.write(b"q\n")
                p.stdin.flush()
                p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=5.0)
            return
        except Exception:
            pass
        try: p.terminate()
        except Exception: pass
        try:
            p.wait(timeout=2.0)
        except Exception:
            try: p.kill()
            except Exception: pass

    def _open_prop_cap(self):
        """
        Try to open a secondary DirectShow handle for property control.
        On some systems this works alongside FFmpeg; on others it fails silently.
        Returns cap or None — callers must handle None gracefully.

        With Phase-1 COM-based property control (`_DshowCameraControl`) now
        primary, this is only a fallback for properties COM doesn't expose.
        Since the half-disconnected camera path can throw "VIDEOIO(DSHOW):
        raised unknown C++ exception!", every cv2 call is wrapped — a single
        flaky camera must not crash the thread.
        """
        for _ in range(3):
            cap = None
            try:
                cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception: pass
                    return cap
            except Exception:
                pass
            if cap is not None:
                try: cap.release()
                except Exception: pass
            time.sleep(0.3)
        return None

    # ── Property helpers ───────────────────────────────────────────────────────
    def set_raw(self, cv_id, value):
        with self._lock: self._queue.append((cv_id, value))

    def set_prop(self, name, value):
        if name in PROPS:
            with self._lock: self._queue.append((PROPS[name][0], value))

    def request_sync(self):
        with self._lock: self._sync_requested = True

    def _read_all_props(self) -> dict:
        return self._read_props_via(self.cap)

    def _read_props_via(self, cap) -> dict:
        result = {}
        if not cap: return result
        for name, (cv_id, mn, mx, default, step) in PROPS.items():
            try:
                val = cap.get(cv_id)
                if val is not None and val != -1.0:
                    result[name] = val
            except Exception:
                pass
        return result

    def _read_props_via_com(self) -> dict:
        """Read current property values via the COM control. Mirrors
        _read_props_via(cap) but uses IAMVideoProcAmp/IAMCameraControl."""
        result = {}
        ctrl = getattr(self, "_dshow_ctrl", None)
        if not (ctrl and ctrl.is_open()):
            return result
        for name, (cv_id, mn, mx, default, step) in PROPS.items():
            if cv_id not in _CV2_TO_DSHOW:
                continue
            kind, prop_id = _CV2_TO_DSHOW[cv_id]
            val, _flag = ctrl._get(kind, prop_id)
            if val is not None:
                result[name] = float(val)
        return result

    def _probe_ranges(self) -> dict:
        return self._probe_ranges_via(self.cap)

    def _probe_ranges_via(self, cap) -> dict:
        """Push extreme values and read back clamped hardware limits."""
        result = {}
        if not cap: return result
        for name, (cv_id, soft_mn, soft_mx, default, step) in PROPS.items():
            if name in ("Exposure", "White Balance"):
                try:
                    cur = cap.get(cv_id)
                    if cur is not None and cur != -1.0:
                        result[name] = (soft_mn, soft_mx, cur)
                except Exception:
                    pass
                continue
            try:
                cur = cap.get(cv_id)
                if cur is None or cur == -1.0: continue
                cap.set(cv_id, 1e6);  time.sleep(0.02)
                actual_max = cap.get(cv_id)
                cap.set(cv_id, -1e6); time.sleep(0.02)
                actual_min = cap.get(cv_id)
                cap.set(cv_id, cur);  time.sleep(0.02)
                if (actual_min is None or actual_max is None or
                        actual_min == -1.0 or actual_max == -1.0 or
                        actual_min >= actual_max):
                    result[name] = (soft_mn, soft_mx, cur)
                else:
                    result[name] = (actual_min, actual_max, cur)
            except Exception:
                pass
        return result

    # ── Recording control (inline-encode, OBS Source Record style) ─────────────
    # When in Phase 1, recording is performed by the same FFmpeg that captures —
    # frames never round-trip through Python. start_recording() / stop_recording()
    # set _record_args and flag a respawn; the run() loop tears down the current
    # FFmpeg and re-opens with the encode branch enabled (or disabled). Brief
    # preview gap on toggle (~0.5–1 s) is the only visible side-effect.
    def start_recording(self, path, width, height, fps, bitrate_mbps, codec,
                        metadata=None):
        with self._lock:
            self._record_args = {
                "path": path, "width": width, "height": height,
                "fps": int(fps), "bitrate": float(bitrate_mbps), "codec": codec,
                "metadata": dict(metadata or {}),
            }
            self._respawn_pending = True

    def stop_recording(self):
        with self._lock:
            self._record_args = None
            self._respawn_pending = True

    def set_transforms(self, flip_h=False, rotation=0, crop_rect=None):
        """Update image transforms (applied in FFmpeg's filter graph). Each
        call overwrites the full state — crop_rect=None explicitly clears
        the crop. Triggers a respawn if any value actually changed."""
        changed = False
        with self._lock:
            new_flip = bool(flip_h)
            if new_flip != self._t_flip_h:
                self._t_flip_h = new_flip; changed = True
            new_rot = int(rotation) % 360
            if new_rot != self._t_rotation:
                self._t_rotation = new_rot; changed = True
            if crop_rect and len(crop_rect) == 4 and int(crop_rect[2]) > 0 and int(crop_rect[3]) > 0:
                cr = (int(crop_rect[0]), int(crop_rect[1]),
                      int(crop_rect[2]), int(crop_rect[3]))
            else:
                cr = None
            if cr != self._t_crop_rect:
                self._t_crop_rect = cr; changed = True
            if changed and self._phase == 1:
                self._respawn_pending = True
        return changed

    def _effective_size(self):
        """Source dimensions after the transform chain (crop → rotate).
        Flip doesn't change dimensions. Returns (eff_w, eff_h)."""
        eff_w, eff_h = self._width, self._height
        if self._t_crop_rect:
            _, _, eff_w, eff_h = self._t_crop_rect
        if self._t_rotation in (90, 270):
            eff_w, eff_h = eff_h, eff_w
        return int(eff_w), int(eff_h)

    def _build_transform_chain(self):
        """Return an FFmpeg filter chain string (without trailing comma) for
        the current crop/flip/rotation state. Empty string if no transforms."""
        parts = []
        cr = self._t_crop_rect
        if cr:
            x, y, cw, ch = (int(v) for v in cr)
            if cw > 0 and ch > 0:
                parts.append(f"crop={cw}:{ch}:{x}:{y}")
        if self._t_flip_h:
            parts.append("hflip")
        # transpose=1 = 90° clockwise, =2 = 90° CCW. 180° = transpose,transpose
        # (cheap — two passes over the small image).
        r = self._t_rotation
        if r == 90:
            parts.append("transpose=1")
        elif r == 270:
            parts.append("transpose=2")
        elif r == 180:
            parts.append("transpose=1,transpose=1")
        return ",".join(parts)

    def is_phase1(self):
        return self._phase == 1

    # ── Main capture loop ──────────────────────────────────────────────────────
    def run(self):
        # Top-level safety net: any uncaught exception in run() should surface
        # via camera_error rather than terminating the QThread silently (or
        # crashing the whole process for some C-level errors).
        try:
            self._run_inner()
        except Exception as e:
            import traceback as _tb
            self.camera_error.emit(
                f"CameraThread crashed: {e}\n{_tb.format_exc()[-800:]}")

    def _run_inner(self):
        # ── Phase 1: FFmpeg capture ────────────────────────────────────────────
        ff_proc, frame_bytes, ff_err = (
            self._open_ffmpeg_capture() if _FFMPEG_OK and self._device_name
            else (None, 0, "FFmpeg not available")
        )

        if ff_proc:
            self._phase           = 1
            self._ffmpeg_proc     = ff_proc
            self._using_mjpeg     = True
            self._negotiated_fps  = float(self._fps)
            self._negotiated_w    = self._width
            self._negotiated_h    = self._height
            self.cap              = None

            # ── Property control: prefer COM (IAMVideoProcAmp /
            # IAMCameraControl) over a secondary cv2.VideoCapture handle.
            # COM works alongside FFmpeg; the cv2 secondary handle is
            # disruptive — opening it can interrupt other cameras already
            # streaming on the bus.  Only fall back to prop_cap if COM fails.
            self._dshow_ctrl = _DshowCameraControl(self.index)
            ctrl_err = self._dshow_ctrl.open()
            prop_cap = None
            if ctrl_err:
                import sys as _sys
                print(f"[PawCapture] DshowCameraControl open: {ctrl_err}; "
                      "falling back to cv2.VideoCapture for sliders",
                      file=_sys.stderr)
                prop_cap = self._open_prop_cap()

            # Apply initial props. COM first; only fall through to prop_cap
            # for items COM didn't handle.
            for name, is_auto in self._initial_auto.items():
                if name in AUTO_CTRL:
                    aid, av, mv = AUTO_CTRL[name]
                    routed = self._dshow_ctrl.set_via_cv2(aid, av if is_auto else mv)
                    if not routed and prop_cap:
                        prop_cap.set(aid, av if is_auto else mv)
                        time.sleep(0.04)
            for name, value in self._initial_props.items():
                if name in PROPS and not self._initial_auto.get(name, False):
                    cv_id = PROPS[name][0]
                    routed = self._dshow_ctrl.set_via_cv2(cv_id, value)
                    if not routed and prop_cap:
                        prop_cap.set(cv_id, value)
                        time.sleep(0.02)

            # Probe ranges. COM's GetRange returns the device's true min/max;
            # prop_cap is a fallback when COM isn't available.
            if self._dshow_ctrl.is_open():
                ranges = self._dshow_ctrl.probe_ranges()
            elif prop_cap:
                ranges = self._probe_ranges_via(prop_cap)
            else:
                ranges = {}

            self.running = True
            self._sync_requested = True
            self.connected.emit()
            if ranges:
                self.ranges_read.emit(ranges)

            # Outer respawn loop. Each iteration runs one FFmpeg capture session.
            # Iteration ends on: (a) self.running → False (normal shutdown),
            # (b) FFmpeg dies on its own (error), (c) _respawn_pending → True
            # (record toggle).  Only (c) re-enters the loop.
            while self.running:
                stdout           = ff_proc.stdout
                _fps_count       = 0
                _fps_t0          = time.perf_counter()
                _quick_fps_done  = False
                _ffmpeg_died     = False
                # Recover the byte _open_ffmpeg_capture probed off the front
                # of stdout. Without prepending it to the first frame read,
                # every preview frame is offset by 1 byte (channel-shifts
                # BGR24 and produces a red-dash artifact at the right edge).
                _stash = getattr(ff_proc, "_stash_first_byte", b"")
                if _stash:
                    ff_proc._stash_first_byte = b""

                while self.running:
                    with self._lock:
                        pending  = self._queue[:]
                        self._queue.clear()
                        do_sync  = self._sync_requested
                        self._sync_requested = False
                        respawn_now = self._respawn_pending

                    if respawn_now:
                        break  # exit inner; outer loop re-opens FFmpeg

                    # Apply slider/auto-toggle changes. COM first; prop_cap is
                    # a fallback for any cv2 IDs COM doesn't route.
                    for cv_id, val in pending:
                        routed = self._dshow_ctrl.set_via_cv2(cv_id, val)
                        if not routed and prop_cap:
                            prop_cap.set(cv_id, val)
                    if do_sync:
                        if self._dshow_ctrl.is_open():
                            self.props_read.emit(self._read_props_via_com())
                        elif prop_cap:
                            self.props_read.emit(self._read_props_via(prop_cap))

                    try:
                        if _stash:
                            need = frame_bytes - len(_stash)
                            raw = _stash + (stdout.read(need) if need > 0 else b"")
                            _stash = b""
                        else:
                            raw = stdout.read(frame_bytes)
                    except Exception:
                        raw = b""

                    if len(raw) == frame_bytes:
                        # Effective dims reflect the active transform chain
                        # (crop/rotate); without crop/rotate they equal the
                        # source dims, so this is identical for the no-
                        # transform case.
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                            (self._eff_h, self._eff_w, 3)).copy()
                        # Inline-encode mode: recorder_ref.write() is a no-op
                        # (Recorder detects Phase 1 and skips its own pipe).
                        # Phase 2 (OpenCV) keeps the legacy write() path below.
                        if self.recorder_ref and self.recorder_ref.active:
                            self.recorder_ref.write(frame)
                        self.frame_ready.emit(frame)
                        _fps_count += 1
                        _now = time.perf_counter()
                        _elapsed = _now - _fps_t0
                        if not _quick_fps_done and _fps_count >= 15 and _elapsed >= 0.05:
                            quick = _fps_count / _elapsed
                            self._negotiated_fps = quick
                            self.fps_measured.emit(quick)
                            _quick_fps_done = True
                        if _elapsed >= 1.0:
                            self.fps_measured.emit(_fps_count / _elapsed)
                            _fps_count = 0; _fps_t0 = _now
                    elif len(raw) == 0:
                        if ff_proc.poll() is not None:
                            _ffmpeg_died = True
                            # Release the COM property interfaces proactively;
                            # the device may be physically gone, in which case
                            # any further raw-vtable Set() call on the stale
                            # IAMVideoProcAmp/IAMCameraControl pointer would
                            # be an access violation and crash the process.
                            try: self._dshow_ctrl.close()
                            except Exception: pass
                        break

                # Tear down the current FFmpeg before deciding next step.
                # Graceful 'q' on stdin so the mp4 muxer writes its trailer
                # (moov atom). Without this, every recording was unplayable.
                self._graceful_stop_ffmpeg(ff_proc)
                self._ffmpeg_proc = None

                if not self.running:
                    break  # normal shutdown

                if _ffmpeg_died and not self._respawn_pending:
                    # Process died unexpectedly — surface and exit Phase 1
                    try:
                        tail = ff_proc.stderr.read(500).decode(errors="replace").strip()
                    except Exception:
                        tail = ""
                    self.camera_error.emit(
                        ("FFmpeg capture stopped.\n" + tail) if tail else
                        "FFmpeg capture stopped.")
                    break

                # Respawn requested: re-open FFmpeg with current _record_args.
                # If encoding is requested but FFmpeg fails to launch, fall back
                # to preview-only so the camera doesn't go dark.
                with self._lock:
                    self._respawn_pending = False
                ff_proc, frame_bytes, ff_err = self._open_ffmpeg_capture()
                if not ff_proc and self._record_args:
                    self.camera_error.emit(
                        f"Encode pipeline failed to start: {ff_err}\n"
                        "Reverting to preview-only.")
                    with self._lock:
                        self._record_args = None
                    if self.recorder_ref:
                        self.recorder_ref.active = False
                    ff_proc, frame_bytes, ff_err = self._open_ffmpeg_capture()
                if not ff_proc:
                    self.camera_error.emit(
                        f"FFmpeg respawn failed: {ff_err}")
                    break
                self._ffmpeg_proc = ff_proc

            if prop_cap:
                try: prop_cap.release()
                except: pass
            try: self._dshow_ctrl.close()
            except Exception: pass
            return

        # ── Phase 2: OpenCV fallback ───────────────────────────────────────────
        # Inline encode is not possible here (no FFmpeg capture process to tee
        # from), so the legacy Recorder.write() path is used for recording —
        # Recorder detects _phase != 1 and falls back to its own subprocess.
        self._phase       = 2
        self._ffmpeg_proc = None
        if self._device_name:
            self.cap = cv2.VideoCapture(f"video={self._device_name}", cv2.CAP_DSHOW)
        if not self._device_name or not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.camera_error.emit(
                f"Cannot open camera {self.index}"
                + (f"\nFFmpeg also failed: {ff_err}" if ff_err else ""))
            return

        MJPEG = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC,       MJPEG)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self.cap.set(cv2.CAP_PROP_FPS,          self._fps)  # second set sometimes sticks
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        actual_fourcc        = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self._using_mjpeg    = (actual_fourcc == MJPEG)
        self._negotiated_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._negotiated_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._negotiated_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for name, is_auto in self._initial_auto.items():
            if name in AUTO_CTRL:
                aid, av, mv = AUTO_CTRL[name]
                self.cap.set(aid, av if is_auto else mv)
                time.sleep(0.04)
        for name, value in self._initial_props.items():
            if name in PROPS and not self._initial_auto.get(name, False):
                self.cap.set(PROPS[name][0], value)
                time.sleep(0.02)

        ranges = self._probe_ranges()
        self.running = True
        self._sync_requested = True
        self.connected.emit()
        if ranges:
            self.ranges_read.emit(ranges)

        _fps_count      = 0
        _fps_t0         = time.perf_counter()
        _quick_fps_done = False   # fires once after first 15 frames (~250-500 ms)

        while self.running:
            with self._lock:
                pending  = self._queue[:]
                self._queue.clear()
                do_sync  = self._sync_requested
                self._sync_requested = False

            for cv_id, val in pending:
                self.cap.set(cv_id, val)

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if do_sync:
                self.props_read.emit(self._read_all_props())

            if self.recorder_ref and self.recorder_ref.active:
                self.recorder_ref.write(frame)
            self.frame_ready.emit(frame)
            _fps_count += 1
            _now = time.perf_counter()
            _elapsed = _now - _fps_t0
            # Quick early estimate after 15 frames — same logic as Phase 1.
            if not _quick_fps_done and _fps_count >= 15 and _elapsed >= 0.05:
                quick = _fps_count / _elapsed
                self._negotiated_fps = quick
                self.fps_measured.emit(quick)
                _quick_fps_done = True
            if _elapsed >= 1.0:
                self.fps_measured.emit(_fps_count / _elapsed)
                _fps_count = 0; _fps_t0 = _now

        self.cap.release()
        self.cap = None

    def stop(self):
        # Setting running=False alone isn't enough: the capture loop blocks
        # in stdout.read(frame_bytes) waiting on FFmpeg, so it won't notice
        # the flag until the next frame arrives. Graceful 'q' to FFmpeg so
        # any active mp4 recording finalizes its trailer, then the closed
        # stdout naturally unblocks the read. Also signal a respawn so any
        # respawn-blocked iteration exits cleanly.
        self.running = False
        with self._lock:
            self._respawn_pending = True
        try:
            proc = self._ffmpeg_proc
            if proc and proc.poll() is None:
                self._graceful_stop_ffmpeg(proc)
        except Exception:
            pass
        self.wait(8000)
        # If still alive, hard kill
        try:
            proc = self._ffmpeg_proc
            if proc and proc.poll() is None:
                proc.kill()
        except Exception:
            pass


# ── GPU encoder probe ──────────────────────────────────────────────────────────
# Tries NVIDIA NVENC → Intel QuickSync → AMD AMF → libx264 in order. Returns
# the name of the first one that actually opens an encoder (not just "build
# has it"). Reason explains why earlier candidates were rejected, so the UI
# can surface (e.g.) "driver too old → using QSV".
def _probe_gpu_encoder() -> tuple:
    """Returns (codec_name: str, reason: str). codec_name is always non-empty;
    "libx264" is the final fallback. reason is "" when the first choice
    (NVENC) succeeded, otherwise describes the chain."""
    candidates = [
        # Use 320x240 yuv420p — every hardware encoder supports this size.
        # 64x64 is below the NVENC minimum and produced false negatives.
        ("h264_nvenc", "NVIDIA NVENC"),
        ("h264_qsv",   "Intel QuickSync"),
        ("h264_amf",   "AMD AMF"),
    ]
    rejections = []
    for codec, label in candidates:
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=320x240",
                 "-vframes", "1", "-pix_fmt", "yuv420p",
                 "-c:v", codec, "-f", "null", "-"],
                capture_output=True, timeout=8,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            if r.returncode == 0:
                return codec, " → ".join(rejections) if rejections else ""
            err = (r.stdout + r.stderr).decode(errors="replace")
            if "Unknown encoder" in err or f"Encoder {codec} not found" in err:
                rejections.append(f"{label}: build lacks it")
            elif ("No NVENC capable devices" in err or
                  "Cannot load nvcuda" in err):
                rejections.append(f"{label}: no NVIDIA GPU")
            elif "Driver does not support the required nvenc API" in err:
                rejections.append(f"{label}: driver too old (need 570+)")
            elif "No QSV" in err or "qsv" in err.lower() and "fail" in err.lower():
                rejections.append(f"{label}: no Intel iGPU")
            elif "No AMF" in err.lower() or "amf" in err.lower() and "fail" in err.lower():
                rejections.append(f"{label}: no AMD GPU")
            else:
                rejections.append(f"{label}: unavailable")
        except Exception as e:
            rejections.append(f"{label}: {e}")
    return "libx264", " → ".join(rejections) if rejections else "no GPU encoder"

_GPU_CODEC, _GPU_REASON = _probe_gpu_encoder()
# Backwards-compatible aliases used elsewhere in the file:
_NVENC_AVAILABLE = (_GPU_CODEC == "h264_nvenc")
_NVENC_REASON    = _GPU_REASON if _GPU_CODEC != "h264_nvenc" else ""
_FFMPEG_OK = bool(FFMPEG)


def _encoder_args(codec: str, bitrate_mbps: float) -> list:
    """Return FFmpeg flags appropriate for the chosen encoder. Same overall
    quality target across encoders — different flag shapes."""
    bv   = f"{bitrate_mbps}M"
    maxr = f"{bitrate_mbps * 1.25}M"
    bufz = f"{bitrate_mbps * 2}M"
    if codec == "h264_nvenc":
        return ["-preset", "p4", "-b:v", bv, "-maxrate", maxr, "-bufsize", bufz]
    if codec == "h264_qsv":
        # QSV: VBR with global_quality is more reliable than -b:v alone
        return ["-preset", "medium", "-b:v", bv, "-maxrate", maxr]
    if codec == "h264_amf":
        return ["-quality", "balanced", "-b:v", bv, "-maxrate", maxr]
    # libx264 software fallback
    if bitrate_mbps <= 4:
        crf = str(max(18, 28 - int(bitrate_mbps * 2)))
        return ["-preset", "fast", "-crf", crf]
    return ["-preset", "fast", "-b:v", bv, "-maxrate", maxr, "-bufsize", bufz]


# ── IAMStreamConfig fps forcing (OBS approach) ─────────────────────────────────
def _force_capture_fps(device_index: int, width: int, height: int, fps: int) -> str:
    """
    Call IAMStreamConfig::SetFormat on the DirectShow capture pin for
    device_index to force MJPEG @ width×height @ fps BEFORE FFmpeg opens the
    device.  This is exactly how OBS forces high framerates past driver caps.

    Uses raw COM vtable calls via ctypes — no DirectShow filter graph is
    created, so there is no risk of putting the See3CAM ISP into tile-
    diagnostic mode (that only happens when a SampleGrabber graph is torn
    down partially).

    Returns "" on success, a human-readable error string on failure.
    Failure is non-fatal: the FFmpeg attempts that follow still run, they
    just may not get the requested fps.
    """
    if sys.platform != "win32":
        return "DirectShow: Windows only"
    try:
        import ctypes
        import ctypes.wintypes as wt
        import comtypes
        from comtypes import GUID
    except ImportError:
        return "comtypes not installed — pip install comtypes"

    HRESULT = ctypes.HRESULT
    c_vp    = ctypes.c_void_p

    # ── ctypes structures ──────────────────────────────────────────────────
    class _RECT(ctypes.Structure):
        _fields_ = [("left", wt.LONG), ("top", wt.LONG),
                    ("right", wt.LONG), ("bottom", wt.LONG)]

    class _BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize",          wt.DWORD), ("biWidth",         wt.LONG),
            ("biHeight",        wt.LONG),  ("biPlanes",        wt.WORD),
            ("biBitCount",      wt.WORD),  ("biCompression",   wt.DWORD),
            ("biSizeImage",     wt.DWORD), ("biXPelsPerMeter", wt.LONG),
            ("biYPelsPerMeter", wt.LONG),  ("biClrUsed",       wt.DWORD),
            ("biClrImportant",  wt.DWORD),
        ]

    class _VIDEOINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("rcSource", _RECT), ("rcTarget", _RECT),
            ("dwBitRate", wt.DWORD), ("dwBitErrorRate", wt.DWORD),
            ("AvgTimePerFrame", ctypes.c_longlong),   # 100-ns units
            ("bmiHeader", _BITMAPINFOHEADER),
        ]

    class _AM_MEDIA_TYPE(ctypes.Structure):
        _fields_ = [
            ("majortype",            GUID),
            ("subtype",              GUID),
            ("bFixedSizeSamples",    wt.BOOL),
            ("bTemporalCompression", wt.BOOL),
            ("lSampleSize",          wt.ULONG),
            ("formattype",           GUID),
            ("pUnk",                 c_vp),   # IUnknown*
            ("cbFormat",             wt.ULONG),
            ("pbFormat",             c_vp),   # BYTE*
        ]

    # ── GUID constants ─────────────────────────────────────────────────────
    def _g(s): return GUID(s)
    CLSID_SystemDeviceEnum         = _g("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
    CLSID_VideoInputDeviceCategory = _g("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
    IID_ICreateDevEnum             = _g("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    IID_IBaseFilter                = _g("{56A86895-0AD4-11CE-B03A-0020AF0BA770}")
    IID_IAMStreamConfig            = _g("{C6E13340-30AC-11d0-A18C-00A0C9118956}")
    MEDIASUBTYPE_MJPG              = _g("{47504A4D-0000-0010-8000-00AA00389B71}")
    FORMAT_VideoInfo               = _g("{05589F80-C356-11CE-BF01-00AA0055595A}")
    FORMAT_VideoInfo2              = _g("{F72A76A0-EB0A-11D0-ACE4-0000C0CC16BA}")
    PINDIR_OUTPUT = 1

    # ── raw vtable helpers ─────────────────────────────────────────────────
    def _vtcall(ptr, idx, restype, argtypes, *args):
        vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(c_vp)))
        fn = ctypes.WINFUNCTYPE(restype, *([c_vp] + list(argtypes)))(vt[0][idx])
        return fn(ptr, *args)

    def _qi(ptr, iid):
        """QueryInterface → raw c_void_p, or None on failure."""
        out = c_vp()
        hr  = _vtcall(ptr, 0, HRESULT,
                      [ctypes.POINTER(GUID), ctypes.POINTER(c_vp)],
                      ctypes.byref(iid), ctypes.byref(out))
        return out if (hr == 0 and out) else None

    def _release(ptr):
        if ptr: _vtcall(ptr, 2, wt.ULONG, [])

    def _free_pmt(ole32, pmt_ptr):
        """Free AM_MEDIA_TYPE allocated by GetStreamCaps."""
        try:
            pmt = ctypes.cast(pmt_ptr, ctypes.POINTER(_AM_MEDIA_TYPE)).contents
            if pmt.pbFormat:
                ole32.CoTaskMemFree(pmt.pbFormat)
        except Exception:
            pass
        try:
            ole32.CoTaskMemFree(pmt_ptr)
        except Exception:
            pass

    # ── main logic ─────────────────────────────────────────────────────────
    try:
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)

        # CoCreateInstance → ICreateDevEnum
        pDevEnum = c_vp()
        hr = ole32.CoCreateInstance(
            ctypes.byref(CLSID_SystemDeviceEnum), None, 1,
            ctypes.byref(IID_ICreateDevEnum), ctypes.byref(pDevEnum))
        if hr != 0 or not pDevEnum:
            return f"CoCreateInstance: 0x{hr & 0xFFFFFFFF:08X}"

        # ICreateDevEnum::CreateClassEnumerator(vtable[3])
        pEnum = c_vp()
        hr = _vtcall(pDevEnum, 3, HRESULT,
                     [ctypes.POINTER(GUID), ctypes.POINTER(c_vp), wt.DWORD],
                     ctypes.byref(CLSID_VideoInputDeviceCategory),
                     ctypes.byref(pEnum), 0)
        _release(pDevEnum)
        if hr != 0 or not pEnum:
            return "CreateClassEnumerator: no video devices"

        # IEnumMoniker::Next(vtable[3]) — iterate to device_index
        moniker = None
        for i in range(device_index + 1):
            pMon   = c_vp()
            nFetch = ctypes.c_ulong(0)
            hr = _vtcall(pEnum, 3, HRESULT,
                         [ctypes.c_ulong, ctypes.POINTER(c_vp), ctypes.POINTER(ctypes.c_ulong)],
                         1, ctypes.byref(pMon), ctypes.byref(nFetch))
            if hr != 0 or nFetch.value == 0:
                _release(pMon); _release(pEnum)
                return f"Device index {device_index} not found ({i} devices found)"
            if i < device_index:
                _release(pMon)
            else:
                moniker = pMon
        _release(pEnum)

        # IMoniker::BindToObject(vtable[8]) → IBaseFilter
        pFilter = c_vp()
        pBindCtx = c_vp()
        ole32.CreateBindCtx(0, ctypes.byref(pBindCtx))
        hr = _vtcall(moniker, 8, HRESULT,
                     [c_vp, c_vp, ctypes.POINTER(GUID), ctypes.POINTER(c_vp)],
                     pBindCtx, None,
                     ctypes.byref(IID_IBaseFilter), ctypes.byref(pFilter))
        _release(pBindCtx); _release(moniker)
        if hr != 0 or not pFilter:
            return f"BindToObject: 0x{hr & 0xFFFFFFFF:08X}"

        # IBaseFilter::EnumPins(vtable[10])
        pEnumPins = c_vp()
        hr = _vtcall(pFilter, 10, HRESULT,
                     [ctypes.POINTER(c_vp)], ctypes.byref(pEnumPins))
        _release(pFilter)
        if hr != 0 or not pEnumPins:
            return "EnumPins failed"

        # Find output pin → IAMStreamConfig
        pStreamCfg = None
        while not pStreamCfg:
            pPin   = c_vp()
            nFetch = ctypes.c_ulong(0)
            hr = _vtcall(pEnumPins, 3, HRESULT,
                         [ctypes.c_ulong, ctypes.POINTER(c_vp), ctypes.POINTER(ctypes.c_ulong)],
                         1, ctypes.byref(pPin), ctypes.byref(nFetch))
            if hr != 0 or nFetch.value == 0:
                _release(pPin); break
            # IPin::QueryDirection(vtable[9])
            direction = ctypes.c_int(-1)
            _vtcall(pPin, 9, HRESULT,
                    [ctypes.POINTER(ctypes.c_int)], ctypes.byref(direction))
            if direction.value == PINDIR_OUTPUT:
                pStreamCfg = _qi(pPin, IID_IAMStreamConfig)
            _release(pPin)
        _release(pEnumPins)
        if not pStreamCfg:
            return "No IAMStreamConfig on capture pin"

        # IAMStreamConfig::GetNumberOfCapabilities(vtable[5])
        nCaps = ctypes.c_int(0)
        nSize = ctypes.c_int(0)
        hr = _vtcall(pStreamCfg, 5, HRESULT,
                     [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
                     ctypes.byref(nCaps), ctypes.byref(nSize))
        if hr != 0 or nCaps.value == 0:
            _release(pStreamCfg)
            return "GetNumberOfCapabilities failed"

        scc_size = max(nSize.value, 88)
        scc_buf  = (ctypes.c_byte * scc_size)()

        # IAMStreamConfig::GetStreamCaps(vtable[6]) loop — find MJPEG type
        # Preference: exact size match > any MJPEG.  On the See3CAM, there
        # will be multiple MJPEG entries (one per resolution); we take the
        # one that matches the requested width×height.
        best_pmt   = None
        best_score = -1

        for i in range(nCaps.value):
            pmt_ptr = c_vp()
            hr = _vtcall(pStreamCfg, 6, HRESULT,
                         [ctypes.c_int, ctypes.POINTER(c_vp), ctypes.POINTER(ctypes.c_byte)],
                         i, ctypes.byref(pmt_ptr),
                         ctypes.cast(ctypes.byref(scc_buf), ctypes.POINTER(ctypes.c_byte)))
            if hr != 0 or not pmt_ptr:
                continue
            try:
                pmt = ctypes.cast(pmt_ptr, ctypes.POINTER(_AM_MEDIA_TYPE)).contents
                is_mjpeg = (pmt.subtype    == MEDIASUBTYPE_MJPG)
                is_vinfo = (pmt.formattype == FORMAT_VideoInfo or
                            pmt.formattype == FORMAT_VideoInfo2)
                if is_mjpeg and is_vinfo and pmt.pbFormat and pmt.cbFormat >= 72:
                    vih = ctypes.cast(pmt.pbFormat,
                                      ctypes.POINTER(_VIDEOINFOHEADER)).contents
                    w_ok  = (vih.bmiHeader.biWidth  == width)
                    h_ok  = (abs(vih.bmiHeader.biHeight) == height)
                    score = 2 if (w_ok and h_ok) else 1 if w_ok else 0
                    if score > best_score:
                        if best_pmt:
                            _free_pmt(ole32, best_pmt)
                        best_pmt   = pmt_ptr
                        best_score = score
                        continue
            except Exception:
                pass
            _free_pmt(ole32, pmt_ptr)

        if not best_pmt:
            _release(pStreamCfg)
            return "No MJPEG media type found in GetStreamCaps"

        # Modify AvgTimePerFrame (and clamp dimensions for exact-match types)
        try:
            pmt = ctypes.cast(best_pmt, ctypes.POINTER(_AM_MEDIA_TYPE)).contents
            vih = ctypes.cast(pmt.pbFormat,
                              ctypes.POINTER(_VIDEOINFOHEADER)).contents
            vih.AvgTimePerFrame = 10_000_000 // fps   # 100-ns units
            if best_score >= 2:                        # exact size already set
                pass
            else:                                      # resize to requested
                vih.bmiHeader.biWidth  = width
                vih.bmiHeader.biHeight = (
                    -height if vih.bmiHeader.biHeight < 0 else height)
        except Exception as e:
            _free_pmt(ole32, best_pmt)
            _release(pStreamCfg)
            return f"Could not modify media type: {e}"

        # IAMStreamConfig::SetFormat(vtable[3])
        hr = _vtcall(pStreamCfg, 3, HRESULT,
                     [ctypes.POINTER(_AM_MEDIA_TYPE)],
                     ctypes.cast(best_pmt, ctypes.POINTER(_AM_MEDIA_TYPE)))
        _free_pmt(ole32, best_pmt)
        _release(pStreamCfg)

        if hr != 0:
            return f"SetFormat: 0x{hr & 0xFFFFFFFF:08X}"
        return ""   # success

    except Exception as e:
        return f"_force_capture_fps: {e}"


# ── DirectShow camera property control (OBS-style) ─────────────────────────────
# IAMVideoProcAmp + IAMCameraControl let us change brightness/contrast/exposure
# etc. on the device while FFmpeg has the capture pin open. This is exactly
# what OBS does for its camera-property sliders. The previous approach
# (a secondary cv2.VideoCapture handle for property control) silently failed
# whenever the device wouldn't allow a second open — so sliders did nothing.

# IAMVideoProcAmp property IDs (KSPROPERTY_VIDEOPROCAMP_*)
_PROCAMP_BRIGHTNESS    = 0
_PROCAMP_CONTRAST      = 1
_PROCAMP_HUE           = 2
_PROCAMP_SATURATION    = 3
_PROCAMP_SHARPNESS     = 4
_PROCAMP_GAMMA         = 5
_PROCAMP_WHITEBALANCE  = 7
_PROCAMP_GAIN          = 9

# IAMCameraControl property IDs
_CAMCTRL_EXPOSURE      = 4

_FLAG_AUTO   = 1
_FLAG_MANUAL = 2

# cv2 prop IDs → (interface_kind, property_id). interface_kind: 'p' = ProcAmp,
# 'c' = CameraControl. Auto props (CAP_PROP_AUTO_*) toggle the flag of the
# associated value property rather than carrying a value of their own.
_CV2_TO_DSHOW = {
    cv2.CAP_PROP_BRIGHTNESS:           ("p", _PROCAMP_BRIGHTNESS),
    cv2.CAP_PROP_CONTRAST:             ("p", _PROCAMP_CONTRAST),
    cv2.CAP_PROP_SATURATION:           ("p", _PROCAMP_SATURATION),
    cv2.CAP_PROP_SHARPNESS:            ("p", _PROCAMP_SHARPNESS),
    cv2.CAP_PROP_GAMMA:                ("p", _PROCAMP_GAMMA),
    cv2.CAP_PROP_GAIN:                 ("p", _PROCAMP_GAIN),
    cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: ("p", _PROCAMP_WHITEBALANCE),
    cv2.CAP_PROP_EXPOSURE:             ("c", _CAMCTRL_EXPOSURE),
}
# Auto-mode toggles: cv_id → (interface_kind, prop_id, auto_marker_value).
# When cap.set(cv_id, auto_marker) is requested, we call Set(prop, prev_value, AUTO).
# Otherwise (any other value) we call Set(prop, prev_value, MANUAL).
_CV2_AUTO_TOGGLES = {
    cv2.CAP_PROP_AUTO_EXPOSURE: ("c", _CAMCTRL_EXPOSURE, 3),     # 3 = auto, 1 = manual
    cv2.CAP_PROP_AUTO_WB:       ("p", _PROCAMP_WHITEBALANCE, 1), # 1 = auto, 0 = manual
}


class _DshowCameraControl:
    """Opens IAMVideoProcAmp and IAMCameraControl on a DirectShow video
    capture device, lets us Set() property values while FFmpeg is streaming.

    Lifetime: open() once when CameraThread starts Phase 1, close() at end.
    Thread-safe: set_via_cv2() is called from the capture loop only.
    """

    # GUIDs needed in addition to those declared in _force_capture_fps.
    _IID_IAMVideoProcAmp  = "{C6E13360-30AC-11d0-A18C-00A0C9118956}"
    _IID_IAMCameraControl = "{C6E13370-30AC-11d0-A18C-00A0C9118956}"

    def __init__(self, device_index):
        self._index    = device_index
        self._procamp  = None    # c_void_p
        self._camctrl  = None    # c_void_p
        self._co_init  = False
        self._open_err = ""

    @property
    def open_error(self):
        return self._open_err

    def is_open(self):
        return self._procamp is not None or self._camctrl is not None

    def open(self):
        """Initialize COM, find the device, QI the property interfaces.
        Returns "" on success or a human-readable error string."""
        if sys.platform != "win32":
            self._open_err = "DirectShow: Windows only"
            return self._open_err
        try:
            import ctypes
            import ctypes.wintypes as wt
            from comtypes import GUID
        except ImportError:
            self._open_err = "comtypes not installed"
            return self._open_err

        HRESULT = ctypes.HRESULT
        c_vp    = ctypes.c_void_p

        def _g(s): return GUID(s)
        CLSID_SystemDeviceEnum         = _g("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
        CLSID_VideoInputDeviceCategory = _g("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
        IID_ICreateDevEnum             = _g("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
        IID_IBaseFilter                = _g("{56A86895-0AD4-11CE-B03A-0020AF0BA770}")
        IID_IAMVideoProcAmp            = _g(self._IID_IAMVideoProcAmp)
        IID_IAMCameraControl           = _g(self._IID_IAMCameraControl)

        def _vtcall(ptr, idx, restype, argtypes, *args):
            vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(c_vp)))
            fn = ctypes.WINFUNCTYPE(restype, *([c_vp] + list(argtypes)))(vt[0][idx])
            return fn(ptr, *args)

        def _qi(ptr, iid):
            out = c_vp()
            hr = _vtcall(ptr, 0, HRESULT,
                         [ctypes.POINTER(GUID), ctypes.POINTER(c_vp)],
                         ctypes.byref(iid), ctypes.byref(out))
            return out if (hr == 0 and out) else None

        def _release(ptr):
            if ptr:
                _vtcall(ptr, 2, wt.ULONG, [])

        try:
            ole32 = ctypes.windll.ole32
            ole32.CoInitialize(None)
            self._co_init = True

            pDevEnum = c_vp()
            hr = ole32.CoCreateInstance(
                ctypes.byref(CLSID_SystemDeviceEnum), None, 1,
                ctypes.byref(IID_ICreateDevEnum), ctypes.byref(pDevEnum))
            if hr != 0 or not pDevEnum:
                self._open_err = f"CoCreateInstance: 0x{hr & 0xFFFFFFFF:08X}"
                return self._open_err

            pEnum = c_vp()
            hr = _vtcall(pDevEnum, 3, HRESULT,
                         [ctypes.POINTER(GUID), ctypes.POINTER(c_vp), wt.DWORD],
                         ctypes.byref(CLSID_VideoInputDeviceCategory),
                         ctypes.byref(pEnum), 0)
            _release(pDevEnum)
            if hr != 0 or not pEnum:
                self._open_err = "no video input devices"
                return self._open_err

            moniker = None
            for i in range(self._index + 1):
                pMon   = c_vp()
                nFetch = ctypes.c_ulong(0)
                hr = _vtcall(pEnum, 3, HRESULT,
                             [ctypes.c_ulong, ctypes.POINTER(c_vp),
                              ctypes.POINTER(ctypes.c_ulong)],
                             1, ctypes.byref(pMon), ctypes.byref(nFetch))
                if hr != 0 or nFetch.value == 0:
                    _release(pMon); _release(pEnum)
                    self._open_err = f"device index {self._index} not found"
                    return self._open_err
                if i < self._index:
                    _release(pMon)
                else:
                    moniker = pMon
            _release(pEnum)

            pFilter  = c_vp()
            pBindCtx = c_vp()
            ole32.CreateBindCtx(0, ctypes.byref(pBindCtx))
            hr = _vtcall(moniker, 8, HRESULT,
                         [c_vp, c_vp, ctypes.POINTER(GUID), ctypes.POINTER(c_vp)],
                         pBindCtx, None,
                         ctypes.byref(IID_IBaseFilter), ctypes.byref(pFilter))
            _release(pBindCtx); _release(moniker)
            if hr != 0 or not pFilter:
                self._open_err = f"BindToObject: 0x{hr & 0xFFFFFFFF:08X}"
                return self._open_err

            self._procamp = _qi(pFilter, IID_IAMVideoProcAmp)
            self._camctrl = _qi(pFilter, IID_IAMCameraControl)
            _release(pFilter)

            if not (self._procamp or self._camctrl):
                self._open_err = "device exposes neither IAMVideoProcAmp nor IAMCameraControl"
                return self._open_err
            return ""
        except Exception as e:
            self._open_err = f"_DshowCameraControl.open: {e}"
            return self._open_err

    def _ifc(self, kind):
        return self._procamp if kind == "p" else self._camctrl

    def _set(self, kind, prop_id, value, flag):
        """Call Set on the chosen interface. vtable[4] for both."""
        ifc = self._ifc(kind)
        if not ifc:
            return False
        try:
            import ctypes
            import ctypes.wintypes as wt
            HRESULT = ctypes.HRESULT
            c_vp    = ctypes.c_void_p
            vt = ctypes.cast(ifc, ctypes.POINTER(ctypes.POINTER(c_vp)))
            fn = ctypes.WINFUNCTYPE(HRESULT, c_vp, ctypes.c_long,
                                    ctypes.c_long, ctypes.c_long)(vt[0][4])
            hr = fn(ifc, prop_id, int(value), flag)
            return hr == 0
        except Exception:
            return False

    def _get(self, kind, prop_id):
        """Call Get; returns (value, flag) or (None, None)."""
        ifc = self._ifc(kind)
        if not ifc:
            return None, None
        try:
            import ctypes
            import ctypes.wintypes as wt
            HRESULT = ctypes.HRESULT
            c_vp    = ctypes.c_void_p
            val  = ctypes.c_long(0)
            flag = ctypes.c_long(0)
            vt = ctypes.cast(ifc, ctypes.POINTER(ctypes.POINTER(c_vp)))
            fn = ctypes.WINFUNCTYPE(
                HRESULT, c_vp, ctypes.c_long,
                ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long))(vt[0][5])
            hr = fn(ifc, prop_id, ctypes.byref(val), ctypes.byref(flag))
            if hr == 0:
                return val.value, flag.value
            return None, None
        except Exception:
            return None, None

    def get_range(self, kind, prop_id):
        """Call GetRange. Returns (mn, mx, step, default, current_value, flag)
        or None on failure."""
        ifc = self._ifc(kind)
        if not ifc:
            return None
        try:
            import ctypes
            HRESULT = ctypes.HRESULT
            c_vp    = ctypes.c_void_p
            mn   = ctypes.c_long(0); mx   = ctypes.c_long(0)
            step = ctypes.c_long(0); dflt = ctypes.c_long(0)
            caps = ctypes.c_long(0)
            vt = ctypes.cast(ifc, ctypes.POINTER(ctypes.POINTER(c_vp)))
            fn = ctypes.WINFUNCTYPE(
                HRESULT, c_vp, ctypes.c_long,
                ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long))(vt[0][3])
            hr = fn(ifc, prop_id, ctypes.byref(mn), ctypes.byref(mx),
                    ctypes.byref(step), ctypes.byref(dflt), ctypes.byref(caps))
            if hr != 0:
                return None
            cur, flag = self._get(kind, prop_id)
            return (mn.value, mx.value, step.value, dflt.value,
                    cur if cur is not None else dflt.value,
                    flag if flag is not None else _FLAG_MANUAL)
        except Exception:
            return None

    def set_via_cv2(self, cv_id, value):
        """Map a cv2 prop ID to the appropriate ProcAmp/CamCtrl call.
        Returns True iff the relevant COM interface is open AND the
        property is one we route. Returns False so the caller can fall
        back to prop_cap for un-routed props or when COM is unavailable."""
        if cv_id in _CV2_AUTO_TOGGLES:
            kind, prop_id, auto_marker = _CV2_AUTO_TOGGLES[cv_id]
            if self._ifc(kind) is None:
                return False
            cur, _ = self._get(kind, prop_id)
            if cur is None:
                cur = 0
            flag = _FLAG_AUTO if int(value) == auto_marker else _FLAG_MANUAL
            self._set(kind, prop_id, cur, flag)
            return True
        if cv_id in _CV2_TO_DSHOW:
            kind, prop_id = _CV2_TO_DSHOW[cv_id]
            if self._ifc(kind) is None:
                return False
            self._set(kind, prop_id, value, _FLAG_MANUAL)
            return True
        return False

    def probe_ranges(self):
        """Return {prop_name: (mn, mx, current)} for the names in PROPS that
        we can route through COM. Used to populate slider ranges."""
        result = {}
        for name, (cv_id, soft_mn, soft_mx, dflt, step) in PROPS.items():
            if cv_id not in _CV2_TO_DSHOW:
                continue
            kind, prop_id = _CV2_TO_DSHOW[cv_id]
            r = self.get_range(kind, prop_id)
            if r is None:
                # Fall back to soft defaults so the slider is still usable
                result[name] = (soft_mn, soft_mx, dflt)
            else:
                mn, mx, _step, _dflt, cur, _flag = r
                if mn >= mx:
                    result[name] = (soft_mn, soft_mx, cur)
                else:
                    result[name] = (mn, mx, cur)
        return result

    def close(self):
        try:
            import ctypes
            import ctypes.wintypes as wt
            c_vp = ctypes.c_void_p
            for ptr in (self._procamp, self._camctrl):
                if ptr:
                    vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(c_vp)))
                    fn = ctypes.WINFUNCTYPE(wt.ULONG, c_vp)(vt[0][2])
                    fn(ptr)
            self._procamp = None
            self._camctrl = None
            if self._co_init:
                ctypes.windll.ole32.CoUninitialize()
                self._co_init = False
        except Exception:
            pass


# ── Recorder ───────────────────────────────────────────────────────────────────
class Recorder:
    """
    Dual-mode recorder.

    • Phase 1 (FFmpeg capture): "inline" mode — encoding happens inside the
      CameraThread's FFmpeg via filter_complex split (NVENC mp4 + bgr24
      preview pipe). Frames never round-trip through Python. Matches the
      OBS Source Record approach. start() asks CameraThread to respawn its
      FFmpeg with the encode branch attached; stop() asks for a respawn
      without it. write() is a no-op in this mode.

    • Phase 2 (OpenCV fallback): "subprocess" mode — legacy behavior.
      This Recorder owns its own FFmpeg subprocess fed by bgr24 frames
      from CameraThread.run() via recorder_ref.write(). Used only when
      Phase 1 is unavailable (FFmpeg binary missing, etc).
    """

    def __init__(self, label):
        self.label       = label
        self.path        = None
        self.active      = False
        self.codec_used  = ""
        # Phase-1 (inline) state
        self._cam_thread = None
        self._inline     = False
        # Phase-2 (subprocess) state
        self.proc        = None
        self._log        = None
        self._frame_w    = DEFAULT_W
        self._frame_h    = DEFAULT_H

    def attach(self, cam_thread):
        """Bind to a CameraThread before start() so Phase 1 can use inline mode."""
        self._cam_thread = cam_thread

    def _resolve_path(self, out_dir, suffix_fmt, phase_tag="", phase_position="prefix"):
        now = datetime.now()
        try:
            ts = now.strftime(suffix_fmt) if suffix_fmt else ""
        except Exception:
            ts = now.strftime("_%Y%m%d_%H%M%S")
        # Group recordings by day so a `recordings/` folder doesn't grow into
        # one giant flat list. The day folder is created on demand by start().
        day_dir = out_dir / now.strftime("%Y-%m-%d")
        target_dir = day_dir / phase_tag if phase_tag else day_dir
        if phase_tag and phase_position == "suffix":
            stem = f"{self.label}{ts}_{phase_tag}"
        elif phase_tag:
            stem = f"{phase_tag}_{self.label}{ts}"
        else:
            stem = f"{self.label}{ts}"
        return target_dir / f"{stem}.mp4", ts

    def start(self, out_dir, width=DEFAULT_W, height=DEFAULT_H,
              fps=DEFAULT_FPS, bitrate_mbps=8, suffix_fmt="_%Y%m%d_%H%M%S",
              metadata=None, phase_tag="", phase_position="prefix"):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path, ts = self._resolve_path(out_dir, suffix_fmt, phase_tag, phase_position)
        # _resolve_path returns a path inside a YYYY-MM-DD subfolder; make
        # sure that subfolder exists before FFmpeg tries to write into it.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_w  = width
        self._frame_h  = height

        codec = _GPU_CODEC
        self.codec_used = codec
        meta = dict(metadata or {})

        # ── Phase 1: inline encode via CameraThread's FFmpeg ─────────────────
        if self._cam_thread is not None and self._cam_thread.is_phase1():
            self._inline = True
            self._cam_thread.start_recording(
                path=self.path, width=width, height=height,
                fps=fps, bitrate_mbps=bitrate_mbps, codec=codec,
                metadata=meta,
            )
            self.active = True
            # Note: FFmpeg respawn is asynchronous. If the encode pipeline
            # fails to start, CameraThread emits camera_error and clears
            # _record_args; UI handles via _on_error.
            return True, ""

        # ── Phase 2: legacy subprocess (OpenCV fallback only) ────────────────
        self._inline = False
        log_path = LOGS_DIR / f"{self.label}_{ts}_ffmpeg.log"
        # See Phase-1 record branch in CameraThread._open_ffmpeg_capture for
        # why `+use_metadata_tags` is gated on metadata presence.
        movflags = "+faststart+use_metadata_tags" if meta else "+faststart"
        common   = ["-pix_fmt", "yuv420p", "-profile:v", "main",
                    "-movflags", movflags]
        meta_args = []
        for k, v in meta.items():
            meta_args += ["-metadata", f"{k}={v}"]
        extra    = _encoder_args(codec, bitrate_mbps)

        cmd = [
            FFMPEG, "-y",
            "-f","rawvideo","-vcodec","rawvideo",
            "-pix_fmt","bgr24","-s",f"{width}x{height}",
            "-r",str(fps),"-i","pipe:0",
            "-c:v", codec,
        ] + extra + common + meta_args + [str(self.path)]

        try:
            self._log = open(log_path, "wb")
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=self._log, stderr=self._log,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            time.sleep(0.05)
            if self.proc.poll() is not None:
                self._log.close()
                err = log_path.read_text(errors="replace")[-400:] if log_path.exists() else ""
                if codec != "libx264":
                    # Probe said this codec was OK but the actual launch failed
                    # (e.g., NVENC session limit hit, GPU busy). Fall back to
                    # software libx264 so the user still gets a recording.
                    self.codec_used = "libx264"
                    fallback_extra = _encoder_args("libx264", bitrate_mbps)
                    cmd2 = [
                        FFMPEG, "-y",
                        "-f","rawvideo","-vcodec","rawvideo",
                        "-pix_fmt","bgr24","-s",f"{width}x{height}",
                        "-r",str(fps),"-i","pipe:0",
                        "-c:v","libx264",
                    ] + fallback_extra + common + meta_args + [str(self.path)]
                    self._log = open(log_path, "wb")
                    self.proc = subprocess.Popen(
                        cmd2, stdin=subprocess.PIPE,
                        stdout=self._log, stderr=self._log,
                        creationflags=0x08000000 if sys.platform == "win32" else 0,
                    )
                    time.sleep(0.05)
                    if self.proc.poll() is not None:
                        self._log.close()
                        err2 = log_path.read_text(errors="replace")[-400:] if log_path.exists() else ""
                        raise RuntimeError(f"FFmpeg (libx264 fallback) failed.\n{err2}")
                    self.active = True
                    return True, f"{codec} failed at runtime — using libx264"
                raise RuntimeError(f"FFmpeg exited immediately.\n{err}")
            self.active = True
            return True, ""
        except FileNotFoundError:
            return False, "ffmpeg.exe not found"
        except RuntimeError as e:
            self.proc = None
            return False, str(e)
        except OSError as e:
            return False, str(e)

    def write(self, frame):
        # Phase 1 inline mode: FFmpeg already has the frame via filter_complex,
        # so this is a no-op. CameraThread.run() calls write() unconditionally
        # in both phases; the no-op keeps that loop simple.
        if self._inline:
            return
        if not (self.active and self.proc and self.proc.stdin): return
        if self.proc.poll() is not None:
            self.active = False; return
        try:
            fh, fw = frame.shape[:2]
            if fw != self._frame_w or fh != self._frame_h:
                frame = cv2.resize(frame, (self._frame_w, self._frame_h))
            self.proc.stdin.write(frame.tobytes())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.active = False

    def stop(self):
        self.active = False
        if self._inline:
            self._inline = False
            if self._cam_thread is not None:
                self._cam_thread.stop_recording()
            return self.path
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self._log:
            try: self._log.close()
            except: pass
        return self.path


# ── Preview popup ──────────────────────────────────────────────────────────────
class PreviewWindow(QWidget):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(f"Preview — {title}")
        self.setStyleSheet("background:#000;")
        self.setMinimumSize(640, 360)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background:#000;")
        lay.addWidget(self.canvas)
        self.resize(1280, 720)
        self._rotation = 0

    def set_rotation(self, deg: int):
        self._rotation = deg % 360
        cam = self.windowTitle().split(" — ")[-1].split(" [")[0]
        suffix = f" [{deg}°]" if deg else ""
        self.setWindowTitle(f"Preview — {cam}{suffix}")

    def update_frame(self, frame):
        if not self.isVisible(): return
        cw = max(self.canvas.width(), 640)
        ch = max(self.canvas.height(), 360)
        fh, fw = frame.shape[:2]
        if fw < 1 or fh < 1: return
        scale  = min(cw / fw, ch / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        rgb = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB))
        qi = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(qi))
        del rgb

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_W): self.close()


# ── Resizable canvas ──────────────────────────────────────────────────────────
class ResizableCanvas(QLabel):
    double_clicked = pyqtSignal()
    _MIN_W, _MAX_W = 240, 1200

    def __init__(self):
        super().__init__()
        self._dragging = False
        self._drag_x   = 0
        self._drag_w   = PREVIEW_W
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.ArrowCursor))

    def _near_edge(self, x): return x >= self.width() - 10

    def mouseMoveEvent(self, e):
        if self._dragging:
            delta = e.globalX() - self._drag_x
            new_w = max(self._MIN_W, min(self._MAX_W, self._drag_w + delta))
            new_h = int(new_w * 9 / 16)
            self.setFixedSize(new_w, new_h)
        else:
            self.setCursor(QCursor(Qt.SizeHorCursor if self._near_edge(e.x()) else Qt.ArrowCursor))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._near_edge(e.x()):
            self._dragging = True; self._drag_x = e.globalX(); self._drag_w = self.width()

    def mouseReleaseEvent(self, e):
        self._dragging = False; self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseDoubleClickEvent(self, e): self.double_clicked.emit()

    def leaveEvent(self, e):
        if not self._dragging: self.setCursor(QCursor(Qt.ArrowCursor))


# ── Clickable value widget ─────────────────────────────────────────────────────
class _ClickableValue(QStackedWidget):
    committed = pyqtSignal(int)

    def __init__(self, value, color, slider_ref):
        super().__init__()
        self._slider = slider_ref
        self.setFixedWidth(48)
        self._lbl = QLabel(str(value))
        self._lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._lbl.setStyleSheet(f"color:{color}; font:bold 10px {FONT};")
        self._lbl.setCursor(QCursor(Qt.IBeamCursor))
        self._lbl.setToolTip("Click to enter value")
        self.addWidget(self._lbl)
        self._edit = QLineEdit(str(value))
        self._edit.setAlignment(Qt.AlignRight)
        self._edit.setStyleSheet(f"""QLineEdit{{
            background:#0A0A1C; color:{color}; border:1px solid {color};
            border-radius:2px; padding:0px 2px; font:bold 10px {FONT};}}""")
        self.addWidget(self._edit)
        self._edit.returnPressed.connect(self._commit)
        self._edit.editingFinished.connect(self._commit)
        self.setCurrentIndex(0)

    def mousePressEvent(self, e):
        if self.currentIndex() == 0:
            self._edit.setText(self._lbl.text())
            self.setCurrentIndex(1)
            self._edit.selectAll(); self._edit.setFocus()
        super().mousePressEvent(e)

    def _commit(self):
        if self.currentIndex() != 1: return
        try:
            v = int(self._edit.text())
            mn, mx = self._slider.minimum(), self._slider.maximum()
            v = max(mn, min(mx, v))
        except ValueError:
            v = self._slider.value()
        self._lbl.setText(str(v)); self.setCurrentIndex(0); self.committed.emit(v)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape: self.setCurrentIndex(0)
        super().keyPressEvent(e)

    def setText(self, s):
        self._lbl.setText(s)
        if self.currentIndex() == 0: self._edit.setText(s)

    def setColor(self, color):
        self._lbl.setStyleSheet(f"color:{color}; font:bold 10px {FONT};")
        self._edit.setStyleSheet(f"""QLineEdit{{
            background:#0A0A1C; color:{color}; border:1px solid {color};
            border-radius:2px; padding:0px 2px; font:bold 10px {FONT};}}""")


# ── PropSlider ─────────────────────────────────────────────────────────────────
class PropSlider(QWidget):
    changed = pyqtSignal(str, int)
    def __init__(self, name, mn, mx, default, step):
        super().__init__(); self.name = name
        row = QHBoxLayout(self); row.setContentsMargins(0,1,0,1); row.setSpacing(8)
        self._lbl_widget = QLabel(name); self._lbl_widget.setFixedWidth(96)
        self._lbl_widget.setStyleSheet(f"color:{TEXT_DIM}; font:10px {FONT};")
        row.addWidget(self._lbl_widget)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(mn, mx); self.slider.setValue(default)
        self.slider.setSingleStep(step); self.slider.setPageStep(step*5)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal{{height:3px;background:#252540;border-radius:2px;}}
            QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}
            QSlider::handle:horizontal{{background:{ACCENT};border:2px solid {BG_CARD};
                width:13px;height:13px;margin:-5px 0;border-radius:7px;}}
            QSlider::handle:horizontal:hover{{background:#FFAA44;}}""")
        row.addWidget(self.slider, 1)
        self.val_lbl = _ClickableValue(default, ACCENT, self.slider)
        row.addWidget(self.val_lbl)
        self.slider.valueChanged.connect(lambda v: (
            self.val_lbl.setText(str(v)), self.changed.emit(name, v)))
        self.val_lbl.committed.connect(self._on_manual)

    def _on_manual(self, v):
        self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)
        self.changed.emit(self.name, v)

    def get(self): return self.slider.value()
    def put(self, v):
        v = int(round(float(v))); v = max(self.slider.minimum(), min(self.slider.maximum(), v))
        self.slider.blockSignals(True); self.slider.setValue(v)
        self.val_lbl.setText(str(v)); self.slider.blockSignals(False)
    def set_range(self, mn, mx, current=None):
        if mn >= mx: return
        self.slider.blockSignals(True)
        self.slider.setRange(int(mn), int(mx))
        step = max(1, (mx - mn) // 100)
        self.slider.setSingleStep(step); self.slider.setPageStep(step * 5)
        val = int(round(float(current))) if current is not None else self.slider.value()
        val = max(int(mn), min(int(mx), val))
        self.slider.setValue(val); self.val_lbl.setText(str(val))
        self.slider.blockSignals(False)
        self._lbl_widget.setToolTip(f"Range: {int(mn)} – {int(mx)}")


# ── AutoToggleSlider ───────────────────────────────────────────────────────────
class AutoToggleSlider(QWidget):
    changed      = pyqtSignal(str, int)
    mode_changed = pyqtSignal(str, bool)
    def __init__(self, name, mn, mx, default, step):
        super().__init__(); self.name = name; self._is_auto = True
        root = QVBoxLayout(self); root.setContentsMargins(0,2,0,2); root.setSpacing(3)
        top  = QHBoxLayout(); top.setSpacing(6)
        self._name_lbl = QLabel(name); self._name_lbl.setFixedWidth(96)
        self._name_lbl.setStyleSheet(f"color:{TEXT_DIM}; font:10px {FONT};")
        top.addWidget(self._name_lbl)
        self.auto_btn   = QPushButton("Auto");   self.auto_btn.setFixedSize(46,18)
        self.manual_btn = QPushButton("Manual"); self.manual_btn.setFixedSize(52,18)
        self.auto_btn.clicked.connect(self._set_auto)
        self.manual_btn.clicked.connect(self._set_manual)
        top.addWidget(self.auto_btn); top.addWidget(self.manual_btn); top.addStretch()
        root.addLayout(top)
        bot = QHBoxLayout(); bot.setSpacing(8)
        spacer = QWidget(); spacer.setFixedWidth(96); bot.addWidget(spacer)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(mn, mx); self.slider.setValue(default)
        self.slider.setSingleStep(step); self.slider.setPageStep(step*5)
        self.slider.valueChanged.connect(self._on_slider)
        bot.addWidget(self.slider, 1)
        self.val_lbl = _ClickableValue(default, TEXT_DIM, self.slider)
        bot.addWidget(self.val_lbl)
        self.val_lbl.committed.connect(self._on_manual)
        root.addLayout(bot)
        self._refresh_style()

    def _on_manual(self, v):
        if self._is_auto: return
        self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)
        self.changed.emit(self.name, v)

    def _refresh_style(self):
        if self._is_auto:
            self.auto_btn.setStyleSheet(f"""QPushButton{{background:{ACCENT};color:#000;
                border:1px solid {ACCENT};border-radius:3px;font:bold 8px {FONT};}}""")
            self.manual_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{TEXT_DIM};
                border:1px solid {BORDER};border-radius:3px;font:bold 8px {FONT};}}
                QPushButton:hover{{color:{TEXT_MED};border-color:{TEXT_MED};}}""")
            self.slider.setEnabled(False)
            self.slider.setStyleSheet(f"""
                QSlider::groove:horizontal{{height:3px;background:#1A1A30;border-radius:2px;}}
                QSlider::sub-page:horizontal{{background:#2A2A44;border-radius:2px;}}
                QSlider::handle:horizontal{{background:#2A2A44;border:2px solid {BG_CARD};
                    width:13px;height:13px;margin:-5px 0;border-radius:7px;}}""")
            self.val_lbl.setColor(TEXT_DIM)
        else:
            self.auto_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{TEXT_DIM};
                border:1px solid {BORDER};border-radius:3px;font:bold 8px {FONT};}}
                QPushButton:hover{{color:{TEXT_MED};border-color:{TEXT_MED};}}""")
            self.manual_btn.setStyleSheet(f"""QPushButton{{background:{ACCENT2};color:#000010;
                border:1px solid {ACCENT2};border-radius:3px;font:bold 8px {FONT};}}""")
            self.slider.setEnabled(True)
            self.slider.setStyleSheet(f"""
                QSlider::groove:horizontal{{height:3px;background:#252540;border-radius:2px;}}
                QSlider::sub-page:horizontal{{background:{ACCENT2};border-radius:2px;}}
                QSlider::handle:horizontal{{background:{ACCENT2};border:2px solid {BG_CARD};
                    width:13px;height:13px;margin:-5px 0;border-radius:7px;}}
                QSlider::handle:horizontal:hover{{background:#88CCFF;}}""")
            self.val_lbl.setColor(ACCENT2)

    def _set_auto(self):
        if not self._is_auto:
            self._is_auto = True; self._refresh_style()
            self.mode_changed.emit(self.name, True)
    def _set_manual(self):
        if self._is_auto:
            self._is_auto = False; self._refresh_style()
            self.mode_changed.emit(self.name, False)
            self.changed.emit(self.name, self.slider.value())
    def _on_slider(self, v):
        self.val_lbl.setText(str(v))
        if not self._is_auto: self.changed.emit(self.name, v)
    def is_auto(self): return self._is_auto
    def get(self): return self.slider.value()
    def put(self, v):
        v = int(round(float(v))); v = max(self.slider.minimum(), min(self.slider.maximum(), v))
        self.slider.blockSignals(True); self.slider.setValue(v)
        self.val_lbl.setText(str(v)); self.slider.blockSignals(False)
    def set_range(self, mn, mx, current=None):
        if mn >= mx: return
        self.slider.blockSignals(True)
        self.slider.setRange(int(mn), int(mx))
        step = max(1, (mx - mn) // 100)
        self.slider.setSingleStep(step); self.slider.setPageStep(step * 5)
        val = int(round(float(current))) if current is not None else self.slider.value()
        val = max(int(mn), min(int(mx), val))
        self.slider.setValue(val); self.val_lbl.setText(str(val))
        self.slider.blockSignals(False)
        self._name_lbl.setToolTip(f"Range: {int(mn)} – {int(mx)}")
    def set_auto_mode(self, is_auto):
        self._is_auto = is_auto; self._refresh_style()


# ── Crop dialog ────────────────────────────────────────────────────────────────
# Lets the user set a crop rectangle in source-pixel coordinates. Two paths:
# drag a rectangle on a live preview of the source frame, or type the
# coordinates into the X/Y/W/H spinners. Constraints: x+w ≤ source_w,
# y+h ≤ source_h, w/h must be even (h264 encoders reject odd dimensions for
# yuv420p).
class _CropPreviewLabel(QLabel):
    """Shows a frame and lets the user drag a crop rectangle. Emits
    rectChanged(QRect) in source-pixel coordinates so the dialog's spinners
    can mirror without needing to know about display→source scaling."""
    rectChanged = pyqtSignal(QRect)

    def __init__(self, src_w, src_h, parent=None):
        super().__init__(parent)
        self._src_w  = int(src_w)
        self._src_h  = int(src_h)
        self._frame_origin = QPoint(0, 0)   # top-left of frame within label
        self._frame_size   = QSize(0, 0)    # rendered frame size in label
        self._scale        = 1.0            # display_px = source_px * scale
        self._rect_src     = QRect(0, 0, self._src_w, self._src_h)
        self._mode         = "idle"     # "idle" | "draw" | "move" | "resize_<zone>"
        self._drag_start_src = QPoint()
        self._move_offset_src = QPoint()
        self._press_rect_src = QRect()
        self._press_pos_src  = QPoint()
        self.setMinimumSize(640, 360)
        self.setStyleSheet(f"background:#07070F;border:1px solid {BORDER};")
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setAlignment(Qt.AlignCenter)

    def set_frame(self, bgr):
        if bgr is None: return
        h, w = bgr.shape[:2]
        # Reject frames that don't match the declared source size. When the
        # dialog opens with a crop already active, the panel briefly still
        # has the cropped frame buffered (the respawn that clears the crop
        # is async, ~0.5 s). Painting that stale cropped frame would compute
        # `_scale` against the source size while the pixmap is actually
        # smaller — every drag coordinate then maps to the wrong source
        # pixel. Better to leave the preview blank until a full-source
        # frame arrives.
        if w != self._src_w or h != self._src_h:
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qi)
        cw, ch = self.width(), self.height()
        if cw < 1 or ch < 1: return
        scaled = pix.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._frame_size = scaled.size()
        self._frame_origin = QPoint((cw - scaled.width()) // 2,
                                    (ch - scaled.height()) // 2)
        self._scale = scaled.width() / float(self._src_w) if self._src_w else 1.0
        self._cached_pix = scaled
        self.update()

    def set_rect_src(self, rect):
        # Snap W/H to even — the recording filter and the spinners both
        # require even values (h264 yuv420p), so painting an odd-width rect
        # would mislead the user about exactly which pixels get captured.
        r = QRect(rect).intersected(QRect(0, 0, self._src_w, self._src_h))
        w = max(2, r.width())  & ~1
        h = max(2, r.height()) & ~1
        self._rect_src = QRect(r.x(), r.y(), w, h)
        self.update()

    def get_rect_src(self):
        return QRect(self._rect_src)

    # ── coordinate conversion ──
    def _label_to_src(self, p):
        """label-pixel QPoint → source-pixel QPoint (clamped to frame)."""
        if self._scale <= 0: return QPoint(0, 0)
        x = (p.x() - self._frame_origin.x()) / self._scale
        y = (p.y() - self._frame_origin.y()) / self._scale
        x = max(0, min(self._src_w - 1, int(round(x))))
        y = max(0, min(self._src_h - 1, int(round(y))))
        return QPoint(x, y)

    def _src_to_label(self, rect):
        s = self._scale
        x = self._frame_origin.x() + int(round(rect.x() * s))
        y = self._frame_origin.y() + int(round(rect.y() * s))
        w = max(1, int(round(rect.width()  * s)))
        h = max(1, int(round(rect.height() * s)))
        return QRect(x, y, w, h)

    # ── mouse ──
    def _rect_has_area(self):
        return self._rect_src.width() > 4 and self._rect_src.height() > 4

    def _classify_zone(self, p_src):
        """Return the hit zone for a point in source coords:
        'TL'/'TR'/'BL'/'BR' (corner handles), 'T'/'B'/'L'/'R' (edge handles),
        'MOVE' (interior, away from edges), 'OUT' (outside the rect)."""
        if not self._rect_has_area():
            return "OUT"
        r = self._rect_src
        # Hit-tolerance is ~10 display pixels worth of source pixels
        tol = max(4, int(round(10.0 / max(self._scale, 1e-3))))
        x, y = p_src.x(), p_src.y()
        l, t = r.left(), r.top()
        rt, b = r.left() + r.width(), r.top() + r.height()  # exclusive edges
        if x < l - tol or x > rt + tol or y < t - tol or y > b + tol:
            return "OUT"
        near_l = abs(x - l)  <= tol
        near_r = abs(x - rt) <= tol
        near_t = abs(y - t)  <= tol
        near_b = abs(y - b)  <= tol
        if near_l and near_t: return "TL"
        if near_r and near_t: return "TR"
        if near_l and near_b: return "BL"
        if near_r and near_b: return "BR"
        if near_l: return "L"
        if near_r: return "R"
        if near_t: return "T"
        if near_b: return "B"
        if l <= x < rt and t <= y < b: return "MOVE"
        return "OUT"

    @staticmethod
    def _cursor_for_zone(zone, pressed=False):
        if zone == "MOVE":
            return Qt.ClosedHandCursor if pressed else Qt.OpenHandCursor
        return {
            "TL": Qt.SizeFDiagCursor, "BR": Qt.SizeFDiagCursor,
            "TR": Qt.SizeBDiagCursor, "BL": Qt.SizeBDiagCursor,
            "T":  Qt.SizeVerCursor,   "B":  Qt.SizeVerCursor,
            "L":  Qt.SizeHorCursor,   "R":  Qt.SizeHorCursor,
        }.get(zone, Qt.CrossCursor)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        if self._frame_size.isEmpty(): return
        p_src = self._label_to_src(e.pos())
        zone = self._classify_zone(p_src)
        self._press_rect_src = QRect(self._rect_src)
        self._press_pos_src  = QPoint(p_src)
        if zone == "OUT":
            # Click outside the rect → start a fresh draw
            self._mode = "draw"
            self._drag_start_src = p_src
            self._rect_src = QRect(p_src, p_src)
        elif zone == "MOVE":
            # Click in the interior → translate the existing rect
            self._mode = "move"
            self._move_offset_src = p_src - self._rect_src.topLeft()
        else:
            # Click on an edge or corner handle → resize from that handle
            self._mode = "resize_" + zone
        self.setCursor(self._cursor_for_zone(zone, pressed=True))
        self.update()
        self.rectChanged.emit(self._rect_src)

    def mouseMoveEvent(self, e):
        p_src = self._label_to_src(e.pos())
        if self._mode == "idle":
            # Hover hint: cursor reflects the zone under the pointer.
            self.setCursor(self._cursor_for_zone(self._classify_zone(p_src)))
            return
        if self._mode == "draw":
            self._rect_src = QRect(self._drag_start_src, p_src).normalized()
        elif self._mode == "move":
            new_tl = p_src - self._move_offset_src
            rw, rh = self._rect_src.width(), self._rect_src.height()
            nx = max(0, min(self._src_w - rw, new_tl.x()))
            ny = max(0, min(self._src_h - rh, new_tl.y()))
            self._rect_src = QRect(nx, ny, rw, rh)
        elif self._mode.startswith("resize_"):
            zone = self._mode[len("resize_"):]
            r0 = self._press_rect_src
            dx = p_src.x() - self._press_pos_src.x()
            dy = p_src.y() - self._press_pos_src.y()
            x1, y1 = r0.left(), r0.top()
            x2, y2 = r0.left() + r0.width(), r0.top() + r0.height()
            if 'L' in zone: x1 = r0.left() + dx
            if 'R' in zone: x2 = r0.left() + r0.width() + dx
            if 'T' in zone: y1 = r0.top() + dy
            if 'B' in zone: y2 = r0.top() + r0.height() + dy
            # Allow inversion through the opposite edge (handle "flips")
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            x1 = max(0, min(self._src_w, x1))
            y1 = max(0, min(self._src_h, y1))
            x2 = max(0, min(self._src_w, x2))
            y2 = max(0, min(self._src_h, y2))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self._rect_src = QRect(x1, y1, x2 - x1, y2 - y1)
        self.update()
        self.rectChanged.emit(self._rect_src)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton: return
        self._mode = "idle"
        self.setCursor(Qt.CrossCursor)
        self.rectChanged.emit(self._rect_src)

    # ── paint ──
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07070F"))
        if hasattr(self, "_cached_pix") and not self._cached_pix.isNull():
            painter.drawPixmap(self._frame_origin, self._cached_pix)
        if self._frame_size.isEmpty():
            # No valid full-source frame yet — let the user know rather than
            # showing a silent black box (and don't allow drawing yet, since
            # source-coord conversions need _scale set first).
            painter.setPen(QColor(TEXT_DIM))
            f = painter.font(); f.setPointSize(11); painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Waiting for full-frame preview…")
            return
        rect_disp = self._src_to_label(self._rect_src)
        # Dim outside the crop rect (overlay against the frame area only)
        frame_rect = QRect(self._frame_origin, self._frame_size)
        painter.setBrush(QColor(0, 0, 0, 110))
        painter.setPen(Qt.NoPen)
        for r in (
            QRect(frame_rect.left(),  frame_rect.top(),
                  frame_rect.width(), rect_disp.top()  - frame_rect.top()),
            QRect(frame_rect.left(),  rect_disp.bottom() + 1,
                  frame_rect.width(), frame_rect.bottom() - rect_disp.bottom()),
            QRect(frame_rect.left(),  rect_disp.top(),
                  rect_disp.left() - frame_rect.left(), rect_disp.height()),
            QRect(rect_disp.right() + 1, rect_disp.top(),
                  frame_rect.right() - rect_disp.right(), rect_disp.height()),
        ):
            if r.isValid():
                painter.fillRect(r.intersected(frame_rect), QColor(0, 0, 0, 110))
        # Crop rectangle outline
        painter.setBrush(Qt.NoBrush)
        pen = QPen(QColor(WARN)); pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect_disp)
        # Resize handles: small filled squares at the 4 corners + 4 edge mids
        if rect_disp.width() >= 16 and rect_disp.height() >= 16:
            hs = 7  # handle size in display pixels
            cx = rect_disp.left() + rect_disp.width() // 2
            cy = rect_disp.top()  + rect_disp.height() // 2
            handle_pts = [
                (rect_disp.left(),  rect_disp.top()),     # TL
                (cx,                rect_disp.top()),     # T
                (rect_disp.right(), rect_disp.top()),     # TR
                (rect_disp.right(), cy),                  # R
                (rect_disp.right(), rect_disp.bottom()),  # BR
                (cx,                rect_disp.bottom()),  # B
                (rect_disp.left(),  rect_disp.bottom()),  # BL
                (rect_disp.left(),  cy),                  # L
            ]
            painter.setBrush(QBrush(QColor(WARN)))
            painter.setPen(QPen(QColor("#000000"), 1))
            for hx, hy in handle_pts:
                painter.drawRect(hx - hs // 2, hy - hs // 2, hs, hs)


class _CropDialog(QDialog):
    def __init__(self, panel, src_w, src_h, current=None):
        super().__init__(panel)
        self.setWindowTitle("Crop recording area")
        self._panel = panel
        self._src_w, self._src_h = int(src_w), int(src_h)
        self._cleared = False

        cur_x, cur_y, cur_w, cur_h = (current if current
                                      else (0, 0, self._src_w, self._src_h))
        init_rect = QRect(cur_x, cur_y, cur_w, cur_h)

        # ── Visual preview with drag selection ──
        self.preview = _CropPreviewLabel(self._src_w, self._src_h)
        self.preview.set_rect_src(init_rect)
        self.preview.rectChanged.connect(self._on_preview_rect_changed)

        # ── Numeric spinners (synchronized with preview) ──
        def _spin(mn, mx, val):
            sb = QSpinBox(); sb.setRange(int(mn), int(mx)); sb.setValue(int(val))
            sb.setSingleStep(2)
            return sb
        self.x_sb = _spin(0, self._src_w - 2, cur_x)
        self.y_sb = _spin(0, self._src_h - 2, cur_y)
        self.w_sb = _spin(2, self._src_w,     cur_w)
        self.h_sb = _spin(2, self._src_h,     cur_h)
        for sb in (self.x_sb, self.y_sb, self.w_sb, self.h_sb):
            sb.valueChanged.connect(self._on_spin_changed)

        info = QLabel(f"Source: {self._src_w} × {self._src_h} px\n"
                      "Drag on preview to set crop, or type values below.\n"
                      "Width/Height are even-rounded (h264).")
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")

        right = QGridLayout()
        right.addWidget(info, 0, 0, 1, 2)
        right.addWidget(QLabel("X"), 1, 0); right.addWidget(self.x_sb, 1, 1)
        right.addWidget(QLabel("Y"), 2, 0); right.addWidget(self.y_sb, 2, 1)
        right.addWidget(QLabel("W"), 3, 0); right.addWidget(self.w_sb, 3, 1)
        right.addWidget(QLabel("H"), 4, 0); right.addWidget(self.h_sb, 4, 1)
        right.setRowStretch(5, 1)

        self.bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        reset_btn = self.bb.button(QDialogButtonBox.Reset)
        reset_btn.setText("Clear crop")
        reset_btn.clicked.connect(self._clear)

        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setMaximumWidth(220)

        body = QHBoxLayout()
        body.addWidget(self.preview, stretch=1)
        body.addWidget(right_panel)

        root = QVBoxLayout()
        root.addLayout(body, stretch=1)
        root.addWidget(self.bb)
        self.setLayout(root)
        self.resize(960, 540)

        # Live-update the preview from the panel's latest frame so the user
        # sees motion (helpful for framing). 10 Hz is plenty for crop work.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_frame)
        self._refresh_timer.start(100)
        self._refresh_frame()  # initial paint

    def _refresh_frame(self):
        f = getattr(self._panel, "_last_frame_bgr", None)
        if f is not None:
            try:
                self.preview.set_frame(f)
            except Exception:
                pass

    def _on_preview_rect_changed(self, qrect):
        # Round w/h down to even, then sync to spinners without re-emitting
        x = max(0, qrect.x())
        y = max(0, qrect.y())
        w = max(2, qrect.width())  & ~1
        h = max(2, qrect.height()) & ~1
        if x + w > self._src_w: w = (self._src_w - x) & ~1
        if y + h > self._src_h: h = (self._src_h - y) & ~1
        for sb, v in ((self.x_sb, x), (self.y_sb, y),
                      (self.w_sb, w), (self.h_sb, h)):
            sb.blockSignals(True); sb.setValue(v); sb.blockSignals(False)
        self.preview.set_rect_src(QRect(x, y, w, h))

    def _on_spin_changed(self):
        x, y = self.x_sb.value(), self.y_sb.value()
        w, h = self.w_sb.value() & ~1, self.h_sb.value() & ~1
        if x + w > self._src_w: w = (self._src_w - x) & ~1
        if y + h > self._src_h: h = (self._src_h - y) & ~1
        # Avoid feedback loops via blockSignals on width/height adjustments
        if self.w_sb.value() != w:
            self.w_sb.blockSignals(True); self.w_sb.setValue(w); self.w_sb.blockSignals(False)
        if self.h_sb.value() != h:
            self.h_sb.blockSignals(True); self.h_sb.setValue(h); self.h_sb.blockSignals(False)
        self.preview.set_rect_src(QRect(x, y, w, h))

    def _clear(self):
        self._cleared = True
        self.accept()

    def result_rect(self):
        if self._cleared:
            return None
        x, y = self.x_sb.value(), self.y_sb.value()
        w, h = self.w_sb.value() & ~1, self.h_sb.value() & ~1
        if w == self._src_w and h == self._src_h and x == 0 and y == 0:
            return None
        return (x, y, w, h)

    def closeEvent(self, e):
        try: self._refresh_timer.stop()
        except Exception: pass
        super().closeEvent(e)


# ── Calibration dialog ─────────────────────────────────────────────────────────
class _CalibClickLabel(QLabel):
    """Shows a frame and lets the user click two points on a reference object.
    Emits pointsChanged([(x, y), ...]) in source-pixel coordinates after each
    click. Up to two points; a third click resets to a single new point."""
    pointsChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src_w        = 0
        self._src_h        = 0
        self._frame_origin = QPoint(0, 0)
        self._frame_size   = QSize(0, 0)
        self._scale        = 1.0
        self._points_src   = []          # list of (x, y) source-pixel tuples
        self._cached_pix   = QPixmap()
        self.setMinimumSize(640, 360)
        self.setStyleSheet(f"background:#07070F;border:1px solid {BORDER};")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)

    def set_frame(self, bgr):
        if bgr is None: return
        h, w = bgr.shape[:2]
        # First frame defines the source size; later frames must match
        # (otherwise click→source mapping would silently drift).
        if self._src_w == 0:
            self._src_w, self._src_h = int(w), int(h)
        elif w != self._src_w or h != self._src_h:
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qi)
        cw, ch = self.width(), self.height()
        if cw < 1 or ch < 1: return
        scaled = pix.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._frame_size = scaled.size()
        self._frame_origin = QPoint((cw - scaled.width()) // 2,
                                    (ch - scaled.height()) // 2)
        self._scale = scaled.width() / float(self._src_w) if self._src_w else 1.0
        self._cached_pix = scaled
        self.update()

    def src_size(self):
        return (self._src_w, self._src_h)

    def points(self):
        return list(self._points_src)

    def clear_points(self):
        self._points_src = []
        self.update()
        self.pointsChanged.emit(self.points())

    def _label_to_src(self, p):
        if self._scale <= 0 or self._src_w == 0:
            return None
        x = (p.x() - self._frame_origin.x()) / self._scale
        y = (p.y() - self._frame_origin.y()) / self._scale
        if x < 0 or y < 0 or x > self._src_w or y > self._src_h:
            return None
        x = max(0, min(self._src_w - 1, int(round(x))))
        y = max(0, min(self._src_h - 1, int(round(y))))
        return (x, y)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        if self._frame_size.isEmpty(): return
        p = self._label_to_src(e.pos())
        if p is None:
            return
        if len(self._points_src) >= 2:
            # third click resets to a single new point
            self._points_src = [p]
        else:
            self._points_src.append(p)
        self.update()
        self.pointsChanged.emit(self.points())

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07070F"))
        if not self._cached_pix.isNull():
            painter.drawPixmap(self._frame_origin, self._cached_pix)
        if self._frame_size.isEmpty():
            painter.setPen(QColor(TEXT_DIM))
            f = painter.font(); f.setPointSize(11); painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Waiting for preview frame…")
            return
        if not self._points_src:
            return
        s = self._scale
        ox, oy = self._frame_origin.x(), self._frame_origin.y()
        disp_pts = [QPoint(ox + int(round(px * s)),
                           oy + int(round(py * s)))
                    for (px, py) in self._points_src]
        if len(disp_pts) == 2:
            pen = QPen(QColor(WARN)); pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(disp_pts[0], disp_pts[1])
        pen = QPen(QColor("#000000")); pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(WARN)))
        for pt in disp_pts:
            painter.drawEllipse(pt, 6, 6)


class _CalibrationDialog(QDialog):
    """Modal: snapshot frame, click two points on a known-length reference,
    type the real length in mm, app computes mm_per_pixel. Working distance
    is optional metadata (just stored, not used in the computation)."""

    def __init__(self, panel, frame_bgr, current):
        super().__init__(panel)
        self.setWindowTitle("Calibrate pixel scale")
        self._panel = panel
        self._result = None       # (mm_per_pixel, working_mm, ref_len_mm, ref_pixels) or None
        self._cleared = False

        self.preview = _CalibClickLabel()
        self.preview.set_frame(frame_bgr)
        self.preview.pointsChanged.connect(self._on_points_changed)

        # Inputs
        self.ref_len_sb = QDoubleSpinBox()
        self.ref_len_sb.setRange(0.1, 100000.0)
        self.ref_len_sb.setDecimals(2); self.ref_len_sb.setSingleStep(1.0)
        self.ref_len_sb.setSuffix(" mm")
        self.ref_len_sb.setValue(float(current.get("ref_length_mm") or 100.0))
        self.ref_len_sb.valueChanged.connect(lambda *_: self._recompute())

        self.dist_sb = QDoubleSpinBox()
        self.dist_sb.setRange(0.0, 100000.0)
        self.dist_sb.setDecimals(1); self.dist_sb.setSingleStep(10.0)
        self.dist_sb.setSpecialValueText("(not set)")
        self.dist_sb.setSuffix(" mm")
        self.dist_sb.setValue(float(current.get("working_distance_mm") or 0.0))

        # Read-outs
        self.pix_lbl = QLabel("Click the two ends of a known-length object.")
        self.pix_lbl.setStyleSheet(f"color:{TEXT_MED};font:10px {FONT};")
        self.scale_lbl = QLabel("mm/pixel: —")
        self.scale_lbl.setStyleSheet(f"color:{TEXT_HI};font:bold 12px {FONT};")

        reset_btn = QPushButton("Reset points")
        reset_btn.clicked.connect(self.preview.clear_points)
        clear_btn = QPushButton("Clear calibration")
        clear_btn.setToolTip("Remove calibration from this camera's profile")
        clear_btn.clicked.connect(self._on_clear)

        info = QLabel(
            "Place a ruler (or any object you know the real length of) inside "
            "the camera's view, click one end, then the other. Enter the real "
            "length below. The mm/pixel ratio is saved with the profile and "
            "stamped into the recorded mp4 metadata for PixelPaws."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{TEXT_MED};font:10px {FONT};")

        # Layout
        grid = QGridLayout()
        grid.setHorizontalSpacing(8); grid.setVerticalSpacing(6)
        row = 0
        grid.addWidget(QLabel("Reference length:"), row, 0)
        grid.addWidget(self.ref_len_sb, row, 1)
        grid.addWidget(QLabel("Working distance:"), row, 2)
        grid.addWidget(self.dist_sb, row, 3)
        row += 1
        grid.addWidget(self.pix_lbl, row, 0, 1, 4)
        row += 1
        grid.addWidget(self.scale_lbl, row, 0, 1, 4)
        for w in (self.ref_len_sb, self.dist_sb):
            w.setStyleSheet(f"""QDoubleSpinBox{{
                background:#0C0C1E;color:{TEXT_HI};border:1px solid {BORDER};
                border-radius:3px;padding:2px 6px;font:10px {FONT};}}""")
        for i in range(4):
            grid.setColumnStretch(i, 1 if i in (1, 3) else 0)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(reset_btn); btn_row.addWidget(clear_btn)
        btn_row.addStretch(); btn_row.addWidget(bb)

        root = QVBoxLayout(self)
        root.addWidget(info)
        root.addWidget(self.preview, 1)
        root.addLayout(grid)
        root.addLayout(btn_row)
        self.resize(820, 720)
        self._recompute()

    def _pixel_distance(self):
        pts = self.preview.points()
        if len(pts) != 2:
            return None
        (x1, y1), (x2, y2) = pts
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _recompute(self):
        d = self._pixel_distance()
        if d is None or d < 1.0:
            self.pix_lbl.setText(f"Points: {len(self.preview.points())}/2")
            self.scale_lbl.setText("mm/pixel: —")
            return
        ref_mm = self.ref_len_sb.value()
        if ref_mm <= 0:
            self.scale_lbl.setText("mm/pixel: —")
            return
        mm_per_px = ref_mm / d
        self.pix_lbl.setText(f"Pixel distance: {d:.2f} px")
        self.scale_lbl.setText(
            f"mm/pixel: {mm_per_px:.4f}    "
            f"(1 mm ≈ {1.0 / mm_per_px:.2f} px)"
        )

    def _on_points_changed(self, _pts):
        self._recompute()

    def _on_clear(self):
        self._cleared = True
        self._result = None
        self.accept()

    def _on_accept(self):
        d = self._pixel_distance()
        if d is None or d < 1.0:
            QMessageBox.warning(self, "Calibrate",
                "Click two points on a reference object before accepting.")
            return
        ref_mm = self.ref_len_sb.value()
        if ref_mm <= 0:
            QMessageBox.warning(self, "Calibrate",
                "Reference length must be greater than zero.")
            return
        wd = self.dist_sb.value()
        self._result = (ref_mm / d, wd if wd > 0 else None, ref_mm, d)
        self.accept()

    def cleared(self):
        return self._cleared

    def result(self):
        return self._result


# ── Camera Panel ───────────────────────────────────────────────────────────────
class CameraPanel(QWidget):
    def __init__(self, index, label):
        super().__init__()
        self.cam_index    = index
        self.label        = label
        self.thread       = None
        self.probe_thread = None
        self._enum_thread = None
        self.recorder     = Recorder(label.replace(" ","_"))
        self.sliders      = {}
        self.auto_sliders = {}
        self._live        = False
        self._modes       = {}
        self._cap_w       = DEFAULT_W
        self._cap_h       = DEFAULT_H
        self._cap_fps     = DEFAULT_FPS
        self._out_dir     = RECORDINGS_DIR
        self._filename    = label.replace(" ","_")
        self._preview_win = None
        self._settings_visible = True
        self._rec_start   = None
        self._cap_bitrate = 8
        self._rotation    = 0
        self._flip_h      = False
        self._crop_rect   = None    # (x, y, w, h) source-pixel, or None
        self._last_frame_bgr = None # latest frame for the crop dialog snapshot
        self._suffix_fmt  = "_%Y%m%d_%H%M%S"
        self._actual_fps  = None
        self._pinned_device_id = None
        # Pixel-space calibration (PixelPaws). mm_per_pixel is the only
        # value PixelPaws actually needs; the rest are kept so the user can
        # see what they typed last time and re-open the dialog with sane
        # defaults. None until the user calibrates.
        self._mm_per_pixel        = None
        self._working_distance_mm = None
        self._calib_ref_length_mm = None
        self._calib_ref_pixels    = None
        self._load_pin()
        self._build()

    # ── Build UI ───────────────────────────────────────────────────────────────
    def _build(self):
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(pal.Window, QColor(BG_CARD))
        self.setPalette(pal)
        self.setStyleSheet(f"CameraPanel{{border:1px solid {BORDER};border-radius:8px;}}")
        root = QVBoxLayout(self)
        root.setContentsMargins(12,10,12,12); root.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        self.dot = QLabel("●"); self.dot.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self.title_lbl = QLabel(self.label)
        self.title_lbl.setStyleSheet(f"color:{TEXT_HI};font:bold 13px {FONT};letter-spacing:2px;")
        self.rec_lbl = QLabel("● REC")
        self.rec_lbl.setStyleSheet(f"color:{DANGER};font:bold 10px {FONT};")
        self.rec_lbl.setVisible(False)
        self.timer_lbl = QLabel("00:00:00")
        self.timer_lbl.setStyleSheet(f"color:{WARN};font:bold 10px {FONT};")
        self.timer_lbl.setVisible(False)
        self.settings_btn = QPushButton("▾ SETTINGS")
        self.settings_btn.setFixedHeight(20)
        self.settings_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{TEXT_MED};}}""")
        self.settings_btn.clicked.connect(self._toggle_settings)
        self.sync_btn = QPushButton("↺ SYNC")
        self.sync_btn.setFixedHeight(20)
        self.sync_btn.setToolTip("Read actual current values from camera hardware")
        self.sync_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{ACCENT2};}}
            QPushButton:disabled{{color:#333355;}}""")
        self.sync_btn.setEnabled(False)
        self.sync_btn.clicked.connect(self._request_sync)
        self.rotate_btn = QPushButton("↻ 0°")
        self.rotate_btn.setFixedHeight(20)
        self.rotate_btn.setToolTip("Rotate preview 90° clockwise")
        self.rotate_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{WARN};}}""")
        self.rotate_btn.clicked.connect(self._cycle_rotation)
        self.flip_btn = QPushButton("⇆ FLIP")
        self.flip_btn.setFixedHeight(20)
        self.flip_btn.setCheckable(True)
        self.flip_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{WARN};}}
            QPushButton:checked{{color:{WARN};}}""")
        self.flip_btn.clicked.connect(self._toggle_flip)
        self.crop_btn = QPushButton("✂ CROP")
        self.crop_btn.setFixedHeight(20)
        self.crop_btn.setToolTip("Crop the recording area (X, Y, W, H)")
        self.crop_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{WARN};}}""")
        self.crop_btn.clicked.connect(self._open_crop_dialog)
        self.cal_btn = QPushButton("📏 CAL")
        self.cal_btn.setFixedHeight(20)
        self.cal_btn.setToolTip("Calibrate mm-per-pixel from a reference object in the scene")
        self.cal_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:none;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{ACCENT2};}}""")
        self.cal_btn.clicked.connect(self._open_calibration_dialog)
        hdr.addWidget(self.dot); hdr.addSpacing(4)
        hdr.addWidget(self.title_lbl); hdr.addStretch()
        hdr.addWidget(self.timer_lbl); hdr.addSpacing(6)
        hdr.addWidget(self.rec_lbl); hdr.addSpacing(8)
        hdr.addWidget(self.flip_btn); hdr.addSpacing(4)
        hdr.addWidget(self.crop_btn); hdr.addSpacing(4)
        hdr.addWidget(self.cal_btn); hdr.addSpacing(4)
        hdr.addWidget(self.rotate_btn); hdr.addSpacing(4)
        hdr.addWidget(self.sync_btn); hdr.addSpacing(4)
        hdr.addWidget(self.settings_btn)
        root.addLayout(hdr)

        # Canvas
        self.canvas = ResizableCanvas()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet(f"background:#07070F;border:1px solid {BORDER};border-radius:5px;color:{TEXT_DIM};font:11px {FONT};")
        self.canvas.setText("NO SIGNAL\n(double-click to expand, drag right edge to resize)")
        self.canvas.double_clicked.connect(self._open_preview)
        root.addWidget(self.canvas, 0, Qt.AlignHCenter)

        # Device row
        sel = QHBoxLayout(); sel.setSpacing(4)
        dev_lbl = QLabel("Device:"); dev_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};")
        self.dev_combo = QComboBox()
        self.dev_combo.setMinimumWidth(170)
        self.dev_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.dev_combo.setStyleSheet(self._combo_style())
        self._style_combo_view(self.dev_combo)
        self.dev_combo.addItem(f"Camera {self.cam_index}", userData=self.cam_index)
        self.refresh_btn = QPushButton("⟳"); self.refresh_btn.setFixedSize(26,26)
        self.refresh_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_MED};background:transparent;
            border:1px solid {BORDER};border-radius:4px;font:bold 13px {FONT};}}
            QPushButton:hover{{color:{TEXT_HI};border-color:{TEXT_MED};}}
            QPushButton:disabled{{color:{TEXT_DIM};}}""")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        self.conn_btn = QPushButton("CONNECT")
        self.conn_btn.setStyleSheet(self._btn(ACCENT2,"#0D2040"))
        self.conn_btn.clicked.connect(self._toggle_connect)
        sel.addWidget(dev_lbl); sel.addWidget(self.dev_combo)
        sel.addWidget(self.refresh_btn); sel.addStretch()
        sel.addWidget(self.conn_btn)
        root.addLayout(sel)
        self._refresh_devices()

        # Resolution / FPS row
        mode_row = QHBoxLayout(); mode_row.setSpacing(6)
        res_lbl = QLabel("Res:"); res_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};")
        self.res_combo = QComboBox(); self.res_combo.setFixedWidth(105)
        self.res_combo.setStyleSheet(self._combo_style())
        self._style_combo_view(self.res_combo)
        for _rw, _rh in PROBE_RESOLUTIONS:
            self.res_combo.addItem(f"{_rw}x{_rh}", userData=(_rw, _rh))
        _def_idx = next((i for i in range(self.res_combo.count())
                         if self.res_combo.itemData(i) == (DEFAULT_W, DEFAULT_H)), 0)
        self.res_combo.setCurrentIndex(_def_idx)
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)
        fps_lbl = QLabel("FPS:"); fps_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};")
        self.fps_combo = QComboBox(); self.fps_combo.setFixedWidth(60)
        self.fps_combo.setStyleSheet(self._combo_style())
        self._style_combo_view(self.fps_combo)
        for _fps in PROBE_FPS:
            self.fps_combo.addItem(str(_fps), userData=_fps)
        _def_fps_idx = next((i for i in range(self.fps_combo.count())
                              if self.fps_combo.itemData(i) == DEFAULT_FPS), 0)
        self.fps_combo.setCurrentIndex(_def_fps_idx)
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        self.detect_btn = QPushButton("DETECT")
        self.detect_btn.setToolTip("Probe camera for supported resolutions & framerates")
        self.detect_btn.setStyleSheet(f"""QPushButton{{color:{WARN};background:transparent;
            border:1px solid {WARN};border-radius:4px;padding:3px 8px;
            font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{background:{WARN};color:#000;}}
            QPushButton:disabled{{color:{TEXT_DIM};border-color:{TEXT_DIM};}}""")
        self.detect_btn.clicked.connect(self._run_probe)
        self.diag_btn = QPushButton("DIAGNOSE")
        self.diag_btn.setToolTip("Show raw FFmpeg format list for this camera")
        self.diag_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_DIM};background:transparent;
            border:1px solid {BORDER};border-radius:4px;padding:3px 8px;
            font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{WARN};border-color:{WARN};}}""")
        self.diag_btn.clicked.connect(self._run_diagnose)
        mode_row.addWidget(res_lbl); mode_row.addWidget(self.res_combo)
        mode_row.addWidget(fps_lbl); mode_row.addWidget(self.fps_combo)
        mode_row.addStretch(); mode_row.addWidget(self.diag_btn); mode_row.addWidget(self.detect_btn)
        root.addLayout(mode_row)

        # Quality / bitrate row
        qual_row = QHBoxLayout(); qual_row.setSpacing(6)
        qual_lbl = QLabel("Quality:"); qual_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};"); qual_lbl.setFixedWidth(42)
        QUALITY_PRESETS = [
            ("Low  (2 Mbps)",       2),
            ("Med  (5 Mbps)",       5),
            ("High (8 Mbps)",       8),
            ("Very High (15 Mbps)", 15),
            ("Ultra (25 Mbps)",     25),
            ("Custom",             -1),
        ]
        self.qual_combo = QComboBox(); self.qual_combo.setFixedWidth(148)
        self.qual_combo.setStyleSheet(self._combo_style())
        self._style_combo_view(self.qual_combo)
        for label, mbps in QUALITY_PRESETS:
            self.qual_combo.addItem(label, userData=mbps)
        self.qual_combo.setCurrentIndex(2)
        self.bitrate_spin = QDoubleSpinBox()
        self.bitrate_spin.setRange(0.5, 100.0); self.bitrate_spin.setSingleStep(0.5)
        self.bitrate_spin.setDecimals(1); self.bitrate_spin.setSuffix(" Mbps")
        self.bitrate_spin.setValue(8.0); self.bitrate_spin.setFixedWidth(88)
        self.bitrate_spin.setEnabled(False)
        self.bitrate_spin.setStyleSheet(f"""QDoubleSpinBox{{
            background:#0C0C1E; color:{TEXT_MED}; border:1px solid {BORDER};
            border-radius:3px; padding:2px 6px; font:10px {FONT};}}
            QDoubleSpinBox:enabled{{color:{TEXT_HI};border-color:{ACCENT2};}}
            QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{{
            width:14px;background:#111126;border:none;}}""")
        self.size_est_lbl = QLabel("")
        self.size_est_lbl.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")

        def _on_qual_changed(idx):
            mbps = self.qual_combo.itemData(idx)
            if mbps == -1:
                self.bitrate_spin.setEnabled(True)
                self._cap_bitrate = self.bitrate_spin.value()
            else:
                self.bitrate_spin.setEnabled(False)
                self.bitrate_spin.setValue(float(mbps))
                self._cap_bitrate = mbps
            _update_size_est()

        def _on_spin_changed(val):
            if self.qual_combo.currentData() == -1:
                self._cap_bitrate = val; _update_size_est()

        def _update_size_est():
            mb  = self._cap_bitrate * 30 * 60 / 8
            gb  = mb / 1024
            self.size_est_lbl.setText(f"~{gb:.1f} GB / 30 min" if gb >= 1 else f"~{int(mb)} MB / 30 min")

        self.qual_combo.currentIndexChanged.connect(_on_qual_changed)
        self.bitrate_spin.valueChanged.connect(_on_spin_changed)
        _update_size_est()
        _orig_res = self._on_res_changed
        def _res_and_est(idx): _orig_res(idx); _update_size_est()
        self._on_res_changed = _res_and_est

        qual_row.addWidget(qual_lbl); qual_row.addWidget(self.qual_combo)
        qual_row.addWidget(self.bitrate_spin); qual_row.addStretch()
        qual_row.addWidget(self.size_est_lbl)
        root.addLayout(qual_row)

        # Save location
        save_row = QHBoxLayout(); save_row.setSpacing(4)
        save_lbl = QLabel("Folder:"); save_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};"); save_lbl.setFixedWidth(42)
        self.dir_btn = QPushButton("📁  " + _short_path(self._out_dir))
        self.dir_btn.setToolTip(str(self._out_dir))
        self.dir_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_MED};background:#0C0C1E;
            border:1px solid {BORDER};border-radius:3px;padding:3px 8px;
            font:10px {FONT};text-align:left;}}
            QPushButton:hover{{border-color:{ACCENT2};color:{TEXT_HI};}}""")
        self.dir_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dir_btn.clicked.connect(self._choose_dir)
        save_row.addWidget(save_lbl); save_row.addWidget(self.dir_btn)
        root.addLayout(save_row)

        # Filename row
        name_row = QHBoxLayout(); name_row.setSpacing(4)
        name_lbl = QLabel("File:"); name_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};"); name_lbl.setFixedWidth(42)
        self.name_edit = QLineEdit(self._filename)
        self.name_edit.setStyleSheet(f"""QLineEdit{{background:#0C0C1E;color:{TEXT_MED};
            border:1px solid {BORDER};border-radius:3px;padding:3px 8px;font:10px {FONT};}}
            QLineEdit:focus{{border-color:{ACCENT2};color:{TEXT_HI};}}""")
        SUFFIX_PRESETS = [
            ("Date + Time",    "_%Y%m%d_%H%M%S"),
            ("Date + Time ms", "_%Y%m%d_%H%M%S_%f"),
            ("Date only",      "_%Y%m%d"),
            ("Time only",      "_%H%M%S"),
            ("None",           ""),
            ("Custom…",        None),
        ]
        self.suffix_combo = QComboBox(); self.suffix_combo.setFixedWidth(110)
        self.suffix_combo.setStyleSheet(self._combo_style())
        self._style_combo_view(self.suffix_combo)
        for lbl, fmt in SUFFIX_PRESETS:
            self.suffix_combo.addItem(lbl, userData=fmt)
        self.suffix_combo.setCurrentIndex(0)
        self.suffix_edit = QLineEdit(); self.suffix_edit.setPlaceholderText("strftime format…")
        self.suffix_edit.setFixedWidth(120); self.suffix_edit.setVisible(False)
        self.suffix_edit.setText(self._suffix_fmt)
        self.suffix_edit.setStyleSheet(f"""QLineEdit{{background:#0C0C1E;color:{WARN};
            border:1px solid {BORDER};border-radius:3px;padding:2px 6px;font:9px {FONT};}}
            QLineEdit:focus{{border-color:{WARN};}}""")
        self.suffix_preview = QLabel()
        self.suffix_preview.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")

        def _update_suffix_preview():
            try: s = datetime.now().strftime(self._suffix_fmt)
            except: s = "(invalid)"
            name = self.name_edit.text().strip() or self.label.replace(" ","_")
            self.suffix_preview.setText(f"{name}{s}.mp4")

        def _on_suffix_combo(idx):
            fmt = self.suffix_combo.itemData(idx)
            if fmt is None:
                self.suffix_edit.setVisible(True)
                self._suffix_fmt = self.suffix_edit.text()
            else:
                self.suffix_edit.setVisible(False)
                self._suffix_fmt = fmt
            _update_suffix_preview()

        self.suffix_combo.currentIndexChanged.connect(_on_suffix_combo)
        self.suffix_edit.textChanged.connect(lambda t: (setattr(self,"_suffix_fmt",t), _update_suffix_preview()))
        self.name_edit.textChanged.connect(lambda t: (
            setattr(self,"_filename", t.strip() or self.label.replace(" ","_")),
            _update_suffix_preview()
        ))
        _update_suffix_preview()

        name_row.addWidget(name_lbl); name_row.addWidget(self.name_edit)
        name_row.addWidget(self.suffix_combo); name_row.addWidget(self.suffix_edit)
        root.addLayout(name_row)

        prev_row = QHBoxLayout(); prev_row.addSpacing(46)
        prev_row.addWidget(self.suffix_preview); prev_row.addStretch()
        root.addLayout(prev_row)

        # Per-camera record button
        rec_row = QHBoxLayout()
        self.cam_rec_btn = QPushButton("⏺  RECORD THIS CAM")
        self.cam_rec_btn.setFixedHeight(28)
        self.cam_rec_btn.setStyleSheet(self._rec_btn_style(False))
        self.cam_rec_btn.setEnabled(False)
        self.cam_rec_btn.clicked.connect(self._toggle_cam_rec)
        rec_row.addWidget(self.cam_rec_btn)
        root.addLayout(rec_row)

        # Settings panel
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{BORDER};"); root.addWidget(line)
        self.settings_panel = QWidget()
        pl = QVBoxLayout(self.settings_panel)
        pl.setContentsMargins(0,0,0,0); pl.setSpacing(4)
        for name, (cv_id, mn, mx, default, step) in PROPS.items():
            if name in AUTO_CTRL:
                w = AutoToggleSlider(name, mn, mx, default, step)
                w.changed.connect(self._on_slider); w.mode_changed.connect(self._on_mode_change)
                self.auto_sliders[name] = w
            else:
                w = PropSlider(name, mn, mx, default, step)
                w.changed.connect(self._on_slider)
                self.sliders[name] = w
            pl.addWidget(w)
        root.addWidget(self.settings_panel)
        root.addStretch()

        self._tick = QTimer(); self._tick.timeout.connect(self._update_timer)

    # ── Style helpers ──────────────────────────────────────────────────────────
    def _btn(self, fg, bg):
        return (f"QPushButton{{color:{fg};background:{bg};border:1px solid {fg};"
                f"border-radius:4px;padding:5px 12px;font:bold 10px {FONT};letter-spacing:1px;}}"
                f"QPushButton:hover{{background:{fg};color:#000010;}}"
                f"QPushButton:disabled{{color:{TEXT_DIM};border-color:{TEXT_DIM};background:transparent;}}")

    def _combo_style(self):
        return (
            f"QComboBox{{"
            f"  background:#111126; color:{TEXT_MED}; border:1px solid {BORDER};"
            f"  border-radius:3px; padding:2px 6px; font:10px {FONT};"
            f"}}"
            f"QComboBox:hover{{ border-color:{TEXT_MED}; }}"
            f"QComboBox::drop-down{{"
            f"  subcontrol-origin:padding; subcontrol-position:top right;"
            f"  width:18px; border-left:1px solid {BORDER};"
            f"}}"
        )

    def _style_combo_view(self, combo):
        combo.view().setStyleSheet(
            f"QAbstractItemView {{"
            f"  background-color: #111126; color: {TEXT_HI};"
            f"  border: 1px solid {BORDER};"
            f"  selection-background-color: #2A2A4A;"
            f"  selection-color: {TEXT_HI};"
            f"  padding: 2px; outline: 0;"
            f"}}"
        )

    def _rec_btn_style(self, active):
        if active:
            return (f"QPushButton{{color:{DANGER};background:#1E0606;border:1px solid {DANGER};"
                    f"border-radius:4px;padding:4px 10px;font:bold 9px {FONT};letter-spacing:1px;}}"
                    f"QPushButton:hover{{background:{DANGER};color:white;}}")
        return (f"QPushButton{{color:{SUCCESS};background:#061A0C;border:1px solid {SUCCESS};"
                f"border-radius:4px;padding:4px 10px;font:bold 9px {FONT};letter-spacing:1px;}}"
                f"QPushButton:hover{{background:{SUCCESS};color:#000;}}")

    # ── Settings toggle ────────────────────────────────────────────────────────
    def _toggle_settings(self):
        self._settings_visible = not self._settings_visible
        self.settings_panel.setVisible(self._settings_visible)
        self.settings_btn.setText("▾ SETTINGS" if self._settings_visible else "▸ SETTINGS")

    def _choose_dir(self):
        folder = QFileDialog.getExistingDirectory(self, f"Save folder — {self.label}", str(self._out_dir), QFileDialog.ShowDirsOnly)
        if folder:
            self._out_dir = Path(folder)
            self.dir_btn.setText("📁  " + _short_path(self._out_dir))
            self.dir_btn.setToolTip(folder)

    def _open_preview(self):
        if not self._live: return
        if self._preview_win is None or not self._preview_win.isVisible():
            self._preview_win = PreviewWindow(self.label)
            self._preview_win.set_rotation(self._rotation)
            self._preview_win.show()
        else:
            self._preview_win.raise_(); self._preview_win.activateWindow()

    # ── Camera slot pinning ────────────────────────────────────────────────────
    def _load_pin(self):
        slots = _load_slots()
        self._pinned_device_id = slots.get(str(self.cam_index))

    def _save_pin(self):
        slots = _load_slots()
        slots[str(self.cam_index)] = self._pinned_device_id
        _save_slots(slots)

    # ── Device enumeration ─────────────────────────────────────────────────────
    def _refresh_devices(self):
        if self._live: return
        self.refresh_btn.setEnabled(False); self.refresh_btn.setText("…")
        self.dev_combo.setEnabled(False)
        self._enum_thread = DeviceEnumThread()
        self._enum_thread.result.connect(self._on_devices_found)
        self._enum_thread.start()

    @staticmethod
    def _normalize_device_label(display):
        """Reduce a combo display like '[2] See3CAM_CU27 [USB hub 3]' to its
        bare device name 'See3CAM_CU27' for grouping. The leading '[N] '
        prefix is the cv2 index added by _on_devices_found, and any trailing
        ' [...]' is a disambiguator suffix (e.g., port info)."""
        import re as _re
        s = _re.sub(r"^\[\d+\]\s*", "", display)
        return s.split(" [")[0].strip()

    def _select_dev_by_name_slot(self, want_name):
        """Pick the Nth combo entry whose normalized name matches want_name,
        where N is this panel's cam_index (so panel 0 picks the 1st See3CAM,
        panel 1 picks the 2nd, etc.). Returns True if a selection was made."""
        if not want_name or self.dev_combo.count() == 0:
            return False
        target = self._normalize_device_label(want_name)
        rows = [i for i in range(self.dev_combo.count())
                if self._normalize_device_label(self.dev_combo.itemText(i)) == target]
        if 0 <= self.cam_index < len(rows):
            self.dev_combo.setCurrentIndex(rows[self.cam_index])
            return True
        return False

    def _on_devices_found(self, devices):
        self.dev_combo.blockSignals(True); self.dev_combo.clear()
        if devices:
            for idx, name, dev_id in devices:
                self.dev_combo.addItem(f"[{idx}] {name}", userData=(idx, dev_id))
            selected = False
            if self._pinned_device_id:
                for i in range(self.dev_combo.count()):
                    _, did = self.dev_combo.itemData(i)
                    if did == self._pinned_device_id:
                        self.dev_combo.setCurrentIndex(i); selected = True; break
            if not selected:
                # Slot-default: group entries by their normalized device name
                # (e.g., all 3× "See3CAM_CU27" map to one bucket regardless
                # of [N] prefix), then pick the Nth entry of the dominant
                # bucket where N is this panel's cam_index.
                name_count = {}
                for i in range(self.dev_combo.count()):
                    label = self._normalize_device_label(self.dev_combo.itemText(i))
                    name_count[label] = name_count.get(label, 0) + 1
                dominant = max(name_count, key=name_count.get) if name_count else ""
                if self._select_dev_by_name_slot(dominant):
                    selected = True
            if not selected:
                for i in range(self.dev_combo.count()):
                    cv_idx, _ = self.dev_combo.itemData(i)
                    if cv_idx == self.cam_index:
                        self.dev_combo.setCurrentIndex(i); break
        else:
            self.dev_combo.addItem("No devices found", userData=(0, "NONE"))
        self.dev_combo.blockSignals(False)
        self.dev_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True); self.refresh_btn.setText("⟳")

    def _selected_cam_index(self):
        d = self.dev_combo.currentData()
        return d[0] if d else self.cam_index

    def _selected_device_id(self):
        d = self.dev_combo.currentData()
        return d[1] if d else None

    # ── Probe / Diagnose ───────────────────────────────────────────────────────
    def _run_diagnose(self):
        import re
        dev_display = self.dev_combo.currentText()
        base = re.sub(r"^\[\d+\]\s*", "", dev_display).split(" [")[0].strip()
        self.diag_btn.setEnabled(False); self.diag_btn.setText("QUERYING…")
        QApplication.processEvents()
        caps, raw = CameraThread._list_dshow_options(base)
        self.diag_btn.setEnabled(True); self.diag_btn.setText("DIAGNOSE")
        if not raw: raw = "(no output — is FFmpeg installed?)"
        relevant = [l for l in raw.splitlines()
                    if any(k in l for k in ("vcodec","pixel_format","dshow","error","Error","Could not"))]
        display = "\n".join(relevant) if relevant else raw[:1200]
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Camera Formats: {base}")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            f"<b>FFmpeg -list_options for:</b> <code>{base}</code><br><br>"
            f"The See3CAM_CU27 driver <b>always reports fps=30</b> for all modes — "
            f"firmware metadata lie. The camera physically supports MJPEG Full HD @ up to 100fps."
        )
        msg.setDetailedText(display); msg.exec_()

    def _run_probe(self):
        if self._live: return
        self.detect_btn.setEnabled(False); self.detect_btn.setText("SCANNING…")
        self.res_combo.setEnabled(False); self.fps_combo.setEnabled(False)
        self.dev_combo.setEnabled(False)
        import re as _re
        sel_base = _re.sub(r'^\[\d+\]\s*', '', self.dev_combo.currentText()).split(' [')[0].strip()
        self.probe_thread = ProbeThread(self._selected_cam_index(), device_name=sel_base)
        self.probe_thread.result.connect(self._on_probe_result)
        self.probe_thread.error.connect(self._on_probe_done)
        self.probe_thread.start()

    def _on_probe_result(self, modes):
        self._modes = modes
        self.res_combo.blockSignals(True); self.res_combo.clear()
        for (w, h) in modes.keys():
            self.res_combo.addItem(f"{w}x{h}", userData=(w, h))
        self._style_combo_view(self.res_combo)
        self.res_combo.blockSignals(False)
        self.res_combo.setCurrentIndex(0); self._on_res_changed(0)
        self._on_probe_done()

    def _on_probe_done(self, *_):
        self.detect_btn.setEnabled(True); self.detect_btn.setText("DETECT")
        self.res_combo.setEnabled(True); self.fps_combo.setEnabled(True)
        self.dev_combo.setEnabled(True)

    def _on_res_changed(self, idx):
        data = self.res_combo.itemData(idx)
        if data is None: return
        w, h = data; self._cap_w, self._cap_h = w, h
        if not self._modes:
            self._on_fps_changed(self.fps_combo.currentIndex()); return
        fps_list = self._modes.get((w, h), [DEFAULT_FPS])
        self.fps_combo.blockSignals(True); self.fps_combo.clear()
        for fps in fps_list:
            self.fps_combo.addItem(str(fps), userData=fps)
        self._style_combo_view(self.fps_combo)
        self.fps_combo.blockSignals(False)
        self.fps_combo.setCurrentIndex(0); self._on_fps_changed(0)

    def _on_fps_changed(self, idx):
        d = self.fps_combo.itemData(idx)
        if d is not None: self._cap_fps = d

    # ── Connect / disconnect ───────────────────────────────────────────────────
    def _toggle_connect(self):
        if self._live: self._disconnect()
        else: self._connect()

    def _connect(self):
        idx = self._selected_cam_index()
        dev_id = self._selected_device_id()
        if dev_id:
            self._pinned_device_id = dev_id; self._save_pin()
        props = {n: s.get() for n, s in self.sliders.items()}
        props.update({n: s.get() for n, s in self.auto_sliders.items()})
        auto_modes = {n: s.is_auto() for n, s in self.auto_sliders.items()}
        import re as _re
        sel_base = _re.sub(r'^\[\d+\]\s*', '', self.dev_combo.currentText()).split(' [')[0].strip()
        sel_pos  = self.dev_combo.currentIndex()
        ff_instance = sum(
            1 for i in range(sel_pos)
            if _re.sub(r'^\[\d+\]\s*', '', self.dev_combo.itemText(i)).split(' [')[0].strip() == sel_base
        )
        self.thread = CameraThread(idx, props, auto_modes,
                                   width=self._cap_w, height=self._cap_h, fps=self._cap_fps,
                                   device_name=sel_base, ff_instance=ff_instance)
        self.thread.recorder_ref = None  # set when recording starts
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.camera_error.connect(self._on_error)
        self.thread.connected.connect(self._on_connected)
        self.thread.props_read.connect(self._on_props_read)
        self.thread.ranges_read.connect(self._on_ranges_read)
        self.thread.fps_measured.connect(self._on_fps_measured)
        # Push current transforms before the capture loop starts so the
        # first FFmpeg launch already includes them.
        try:
            self.thread.set_transforms(
                flip_h=self._flip_h,
                rotation=self._rotation,
                crop_rect=self._crop_rect,
            )
        except Exception:
            pass
        self.thread.start()
        self.conn_btn.setEnabled(False); self.conn_btn.setText("CONNECTING…")
        self.dev_combo.setEnabled(False); self.refresh_btn.setEnabled(False)

    def _on_connected(self):
        self._live = True
        self.conn_btn.setEnabled(True); self.conn_btn.setText("DISCONNECT")
        self.conn_btn.setStyleSheet(self._btn(DANGER,"#200808"))
        self.dot.setStyleSheet(f"color:{SUCCESS};font-size:11px;")
        self.cam_rec_btn.setEnabled(True)
        self.sync_btn.setEnabled(True)
        self.sync_btn.setToolTip("Read actual current values from camera hardware")
        QTimer.singleShot(300, self._update_stream_format_tip)

    def _update_stream_format_tip(self):
        if not (self.thread and self._live): return
        mjpeg   = getattr(self.thread, "_using_mjpeg",    None)
        neg_fps = getattr(self.thread, "_negotiated_fps", None)
        neg_w   = getattr(self.thread, "_negotiated_w",   None)
        neg_h   = getattr(self.thread, "_negotiated_h",   None)
        backend = "OpenCV+MJPEG" if mjpeg else "OpenCV+YUY2"
        res     = f"{neg_w}x{neg_h}" if neg_w else "?"
        fps_str = f"{neg_fps:.1f}" if neg_fps else "?"
        if mjpeg:
            self.dot.setStyleSheet(f"color:{SUCCESS};font-size:11px;")
            color_note = "MJPEG active — good"
        else:
            self.dot.setStyleSheet(f"color:#FF9F0A;font-size:11px;")
            color_note = "YUY2 — high USB bandwidth"
        tip = f"Format: {backend}  |  Negotiated: {res} @ {fps_str}fps\n{color_note}"
        if neg_fps and abs(neg_fps - self._cap_fps) > 2:
            tip += f"\n⚠ Driver delivered {fps_str}fps, not {self._cap_fps}fps"
            tip += "\nNote: driver metadata lies — actual fps may differ"
        self.dot.setToolTip(tip)

    def _disconnect(self):
        if self.recorder.active: self.stop_rec()
        if self.thread: self.thread.stop(); self.thread = None
        self._live = False
        if self._preview_win: self._preview_win.close()
        self.canvas.setPixmap(QPixmap())
        self.canvas.setText("NO SIGNAL\n(double-click to expand, drag right edge to resize)")
        self.conn_btn.setText("CONNECT")
        self.conn_btn.setStyleSheet(self._btn(ACCENT2,"#0D2040"))
        self.dot.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self.dev_combo.setEnabled(True); self.refresh_btn.setEnabled(True)
        self.cam_rec_btn.setEnabled(False)
        self.cam_rec_btn.setText("⏺  RECORD THIS CAM")
        self.cam_rec_btn.setStyleSheet(self._rec_btn_style(False))
        self.sync_btn.setEnabled(False)

    def _on_error(self, msg):
        self._live = False; self.canvas.setText(f"ERROR\n{msg}")
        self.conn_btn.setEnabled(True); self.conn_btn.setText("CONNECT")
        self.conn_btn.setStyleSheet(self._btn(ACCENT2,"#0D2040"))
        self.dev_combo.setEnabled(True); self.refresh_btn.setEnabled(True)
        self.cam_rec_btn.setEnabled(False)

    # ── Frame handling ─────────────────────────────────────────────────────────
    def _on_frame(self, frame):
        # frame is already a copy from the capture loop, safe to retain.
        # Stored uncropped so the crop dialog can show the full source frame.
        self._last_frame_bgr = frame
        display = self._apply_rotation(frame)
        if self._preview_win and self._preview_win.isVisible():
            self._preview_win.update_frame(display)
        self._paint_canvas(self.canvas, display)

    def _fit_frame(self, frame, cw, ch):
        fh, fw = frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if nw == cw and nh == ch: return resized
        canvas = np.zeros((ch, cw, frame.shape[2]), dtype=frame.dtype)
        canvas[(ch-nh)//2:(ch-nh)//2+nh, (cw-nw)//2:(cw-nw)//2+nw] = resized
        return canvas

    def _paint_canvas(self, label, frame):
        cw, ch = label.width(), label.height()
        if cw < 1 or ch < 1: return
        if self._mm_per_pixel:
            frame = self._draw_scale_bar(frame)
        fitted = self._fit_frame(frame, cw, ch)
        rgb    = np.ascontiguousarray(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB))
        h, w, c = rgb.shape
        qi = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi))
        del rgb

    def _draw_scale_bar(self, frame):
        """Overlay a 100 mm scale bar in the bottom-left corner so the user
        can sanity-check calibration at a glance. Returns a new frame —
        the caller's array is left intact (the calibration dialog
        snapshots `_last_frame_bgr` and would otherwise show the bar)."""
        mm_per_px = self._mm_per_pixel
        if not mm_per_px or mm_per_px <= 0:
            return frame
        h, w = frame.shape[:2]
        bar_px = int(round(100.0 / mm_per_px))
        # Skip when it'd be invisible or absurdly large (probably means
        # calibration is stale relative to a recent resolution change).
        if bar_px < 8 or bar_px > w - 40:
            return frame
        out = frame.copy()
        margin = 16
        y  = h - margin
        x1 = margin
        x2 = x1 + bar_px
        color = (10, 214, 255)  # WARN (#FFD60A) in BGR
        cv2.line(out, (x1, y), (x2, y), color, 2)
        cv2.line(out, (x1, y - 6), (x1, y + 6), color, 2)
        cv2.line(out, (x2, y - 6), (x2, y + 6), color, 2)
        cv2.putText(out, f"100 mm  ({mm_per_px:.3f} mm/px)",
                    (x1, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        return out

    def _toggle_flip(self):
        self._flip_h = self.flip_btn.isChecked()
        self.flip_btn.setText("⇆ FLIP ●" if self._flip_h else "⇆ FLIP")
        self._push_transforms_to_thread()

    def _cycle_rotation(self):
        self._rotation = (self._rotation + 90) % 360
        self.rotate_btn.setText(f"↻ {self._rotation}°")
        if self._preview_win and self._preview_win.isVisible():
            self._preview_win.set_rotation(self._rotation)
        self._push_transforms_to_thread()

    def _open_crop_dialog(self):
        """Modal dialog to set/clear the crop rectangle in source-pixel coords.

        Crop coordinates are always in raw source pixel space — the FFmpeg
        filter chain runs `crop` *before* hflip/transpose, so a flip or
        rotation that's currently applied would otherwise show the user a
        transformed preview while the spinners they edit map to the
        un-transformed source. Temporarily clear all three transforms on
        the thread for the dialog's lifetime so the user always edits
        against the raw source frame; restore the panel's state after.

        Panel state itself (`_flip_h`, `_rotation`, `_crop_rect`) is not
        modified mid-dialog except `_crop_rect` at accept; restoration is
        a single `_push_transforms_to_thread` at the end."""
        saved_crop = self._crop_rect
        # Clear all transforms on the thread (not panel state) so the
        # dialog shows raw, untransformed source frames.
        if self.thread:
            try:
                self.thread.set_transforms(
                    flip_h=False, rotation=0, crop_rect=None)
            except Exception:
                pass
        dlg = _CropDialog(self, self._cap_w, self._cap_h, saved_crop)
        accepted = (dlg.exec_() == _CropDialog.Accepted)
        if accepted:
            self._crop_rect = dlg.result_rect()
        else:
            self._crop_rect = saved_crop
        self._update_crop_btn_label()
        # Restore the full transform state (panel's flip/rotation +
        # whatever crop ended up). One respawn applies all of it.
        self._push_transforms_to_thread()

    def _update_crop_btn_label(self):
        if self._crop_rect:
            x, y, w, h = self._crop_rect
            self.crop_btn.setText(f"✂ {w}×{h} ●")
        else:
            self.crop_btn.setText("✂ CROP")

    def _open_calibration_dialog(self):
        """Snapshot the current preview frame and let the user click two
        points on a reference object of known length. Result (mm_per_pixel,
        working_distance_mm, ref_length_mm, ref_pixels) is saved on the
        panel and written into mp4 metadata at record start."""
        frame = getattr(self, "_last_frame_bgr", None)
        if frame is None:
            QMessageBox.information(self, "Calibrate",
                "No preview frame yet — connect the camera first.")
            return
        current = {
            "ref_length_mm":       self._calib_ref_length_mm,
            "working_distance_mm": self._working_distance_mm,
        }
        # Snapshot a copy so the live preview update can't race the dialog.
        snap = frame.copy() if frame is not None else None
        dlg = _CalibrationDialog(self, snap, current)
        if dlg.exec_() != QDialog.Accepted:
            return
        if dlg.cleared():
            self._mm_per_pixel        = None
            self._working_distance_mm = None
            self._calib_ref_length_mm = None
            self._calib_ref_pixels    = None
        else:
            res = dlg.result()
            if res is None:
                return
            mm_per_px, wd, ref_mm, ref_px = res
            self._mm_per_pixel        = float(mm_per_px)
            self._working_distance_mm = float(wd) if wd is not None else None
            self._calib_ref_length_mm = float(ref_mm)
            self._calib_ref_pixels    = float(ref_px)
        self._update_calib_btn_label()

    def _update_calib_btn_label(self):
        if self._mm_per_pixel:
            self.cal_btn.setText(f"📏 {self._mm_per_pixel:.3f} mm/px ●")
        else:
            self.cal_btn.setText("📏 CAL")

    def _calibration_metadata(self):
        """Return a dict of FFmpeg `-metadata key=value` pairs for this
        recording. Always includes `pawcapture_version`; calibration keys
        only appear when the camera has been calibrated. Values are
        stringified because FFmpeg's CLI takes everything as strings."""
        md = {"pawcapture_version": PAWCAPTURE_VERSION}
        if self._mm_per_pixel:
            md["mm_per_pixel"]         = f"{self._mm_per_pixel:.6f}"
            md["pixelpaws_calibrated"] = "1"
            if self._working_distance_mm:
                md["working_distance_mm"] = f"{self._working_distance_mm:.2f}"
            if self._calib_ref_length_mm:
                md["pixelpaws_ref_length_mm"] = f"{self._calib_ref_length_mm:.2f}"
            if self._calib_ref_pixels:
                md["pixelpaws_ref_pixels"] = f"{self._calib_ref_pixels:.2f}"
        return md

    def _push_transforms_to_thread(self):
        """Forward current panel transform state to the capture thread.
        Triggers an FFmpeg respawn (~0.5 s preview gap) if anything changed."""
        if not self.thread:
            return
        try:
            self.thread.set_transforms(
                flip_h=self._flip_h,
                rotation=self._rotation,
                crop_rect=self._crop_rect,
            )
        except Exception:
            pass

    def _apply_rotation(self, frame):
        # Phase 1 (FFmpeg) applies crop/flip/rotate inside the filter graph,
        # so the frames arriving here are already transformed — don't double-
        # apply. Phase 2 (OpenCV fallback) hands raw frames to Python, so the
        # cv2 ops are still needed there.
        if self.thread and self.thread.is_phase1():
            return frame
        if self._flip_h: frame = cv2.flip(frame, 1)
        if self._rotation == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self._rotation == 180: return cv2.rotate(frame, cv2.ROTATE_180)
        if self._rotation == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _request_sync(self):
        if self.thread:
            self.sync_btn.setText("…"); self.sync_btn.setEnabled(False)
            self.thread.request_sync()

    def _on_fps_measured(self, fps: float):
        self._actual_fps = fps

    def _on_ranges_read(self, ranges: dict):
        for name, (mn, mx, cur) in ranges.items():
            if name in self.sliders:        self.sliders[name].set_range(mn, mx, cur)
            elif name in self.auto_sliders: self.auto_sliders[name].set_range(mn, mx, cur)

    def _on_props_read(self, vals: dict):
        for name, raw_val in vals.items():
            val = int(round(raw_val))
            if name in self.sliders:        self.sliders[name].put(val)
            elif name in self.auto_sliders: self.auto_sliders[name].put(val)
        self.sync_btn.setText("↺ SYNC"); self.sync_btn.setEnabled(True)

    def _on_slider(self, name, value):
        if self.thread: self.thread.set_prop(name, value)

    def _on_mode_change(self, name, is_auto):
        if not self.thread: return
        aid, av, mv = AUTO_CTRL[name]
        if is_auto:
            self.thread.set_raw(aid, av)
        else:
            self.thread.set_raw(aid, mv)
            def _push():
                time.sleep(0.12)
                if self.thread: self.thread.set_prop(name, self.auto_sliders[name].get())
            threading.Thread(target=_push, daemon=True).start()

    # ── Recording ──────────────────────────────────────────────────────────────
    def _toggle_cam_rec(self):
        if self.recorder.active:
            self.stop_rec()
        else:
            # If _actual_fps hasn't arrived yet, defer up to 1.4 s rather than
            # stamping the container with the unverified configured fps.
            # Gap 3's early-15-frame measurement means this wait is usually
            # ≤ 300 ms; the 1.4 s cap covers slow camera ISP warm-up.
            if self._live and self._actual_fps is None:
                self._rec_defer_retries = 7          # 7 × 200 ms = 1.4 s max
                self.cam_rec_btn.setEnabled(False)
                self.cam_rec_btn.setText("⏳  MEASURING FPS…")
                QTimer.singleShot(200, self._deferred_start_rec)
            else:
                self._do_start_rec()

    def _deferred_start_rec(self):
        """Retry recording start once _actual_fps is measured (non-blocking)."""
        if not self._live:
            self.cam_rec_btn.setEnabled(True)
            self.cam_rec_btn.setText("⏺  RECORD THIS CAM")
            self.cam_rec_btn.setStyleSheet(self._rec_btn_style(False))
            return
        if self._actual_fps and self._actual_fps > 1:
            self.cam_rec_btn.setEnabled(True)
            self._do_start_rec()
        elif self._rec_defer_retries > 0:
            self._rec_defer_retries -= 1
            QTimer.singleShot(200, self._deferred_start_rec)
        else:
            # Timed out — start anyway with configured fps (shows warning in label)
            self.cam_rec_btn.setEnabled(True)
            self._do_start_rec()

    def _do_start_rec(self):
        ok, err = self.start_rec()
        if not ok:
            QMessageBox.critical(self, "Record Failed",
                f"FFmpeg could not start ({_GPU_CODEC}).\n\n{err}\n\nCheck logs in:\n{LOGS_DIR}")

    def start_rec(self):
        if not self._live: return False, ""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.recorder.label = self._filename or self.label.replace(" ","_")

        # Bind recorder to thread so Phase 1 picks the inline (OBS Source
        # Record) encode path. In Phase 2 (OpenCV fallback) the recorder
        # detects is_phase1() == False and uses the legacy subprocess path.
        if self.thread:
            self.recorder.attach(self.thread)

        # Stamp the container with the *configured* rate. With round-9's
        # `-video_track_timescale {fps*256}` forcing the muxer's timebase
        # and the in-chain `fps={fps}` resampler enforcing constant rate
        # before the encoder, the output is always at cap_fps regardless
        # of cam clock skew. The measured rate is informational only —
        # surfaced as a warning when it differs significantly so a flat-out
        # under-delivering camera (e.g., exposure-capped) is still visible.
        rec_fps = self._cap_fps
        if self._actual_fps and self._actual_fps > 1:
            measured = round(self._actual_fps)
            fps_note = f"~{measured}fps measured" if abs(measured - rec_fps) > 2 else ""
        else:
            fps_note = ""

        win = self.window()
        phase_tag = win.current_phase_tag() if hasattr(win, "current_phase_tag") else ""
        phase_pos = win.current_phase_position() if hasattr(win, "current_phase_position") else "prefix"
        ok, err = self.recorder.start(self._out_dir, width=self._cap_w, height=self._cap_h,
                                      fps=rec_fps, bitrate_mbps=self._cap_bitrate,
                                      suffix_fmt=self._suffix_fmt,
                                      metadata=self._calibration_metadata(),
                                      phase_tag=phase_tag, phase_position=phase_pos)
        if ok:
            # recorder_ref is still set so the Phase-2 capture loop can write()
            # frames to the recorder's own subprocess.  In Phase 1 the recorder
            # is in inline mode and write() is a no-op, so the assignment is
            # harmless.
            if self.thread:
                self.thread.recorder_ref = self.recorder
            self.rec_lbl.setVisible(True); self.timer_lbl.setVisible(True)
            self._rec_start = datetime.now(); self._tick.start(1000)
            codec = getattr(self.recorder, "codec_used", "?")
            # Map codec name → short UI tag.  GPU encoders all show as "GPU"
            # with the vendor in parens; libx264 shows the chain reason so
            # the user knows which GPU encoders were rejected and why.
            gpu_label = {
                "h264_nvenc": "GPU NVENC",
                "h264_qsv":   "GPU QSV",
                "h264_amf":   "GPU AMF",
            }.get(codec)
            if gpu_label:
                enc_tag = gpu_label
            else:
                enc_tag = f"CPU ({_GPU_REASON})" if _GPU_REASON else "CPU"
            inline_tag = " inline" if self.thread and self.thread.is_phase1() else ""
            btn_label = f"⏹  STOP  [{enc_tag}{inline_tag} {int(rec_fps)}fps]"
            if fps_note: btn_label += f"  {fps_note}"
            self.cam_rec_btn.setText(btn_label)
            self.cam_rec_btn.setStyleSheet(self._rec_btn_style(True))
        return ok, err

    def stop_rec(self):
        if self.thread:
            self.thread.recorder_ref = None
        self._tick.stop()
        self.timer_lbl.setText("00:00:00"); self._rec_start = None
        path = self.recorder.stop()
        self.rec_lbl.setVisible(False); self.timer_lbl.setVisible(False)
        self.cam_rec_btn.setText("⏺  RECORD THIS CAM")
        self.cam_rec_btn.setStyleSheet(self._rec_btn_style(False))
        return path

    def _update_timer(self):
        if self._rec_start:
            secs = int((datetime.now() - self._rec_start).total_seconds())
            h, r = divmod(secs, 3600); m, s = divmod(r, 60)
            self.timer_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # ── Profile serialisation ──────────────────────────────────────────────────
    def get_settings(self):
        data = {n: s.get() for n, s in self.sliders.items()}
        data.update({n: s.get() for n, s in self.auto_sliders.items()})
        data["_auto_modes"]  = {n: s.is_auto() for n, s in self.auto_sliders.items()}
        data["_resolution"]  = f"{self._cap_w}x{self._cap_h}"
        data["_fps"]         = self._cap_fps
        data["_out_dir"]     = str(self._out_dir)
        data["_filename"]    = self._filename
        data["_bitrate"]     = self._cap_bitrate
        data["_device_id"]   = self._pinned_device_id or ""
        data["_device_name"] = self.dev_combo.currentText()
        data["_rotation"]    = self._rotation
        data["_flip_h"]      = self._flip_h
        data["_crop_rect"]   = list(self._crop_rect) if self._crop_rect else None
        data["_suffix_fmt"]  = self._suffix_fmt
        data["_mm_per_pixel"]        = self._mm_per_pixel
        data["_working_distance_mm"] = self._working_distance_mm
        data["_calib_ref_length_mm"] = self._calib_ref_length_mm
        data["_calib_ref_pixels"]    = self._calib_ref_pixels
        data["_pawcapture_version"]  = PAWCAPTURE_VERSION
        return data

    def apply_settings(self, data):
        saved_dev_id   = data.get("_device_id", "")
        saved_dev_name = data.get("_device_name", "")
        # Selection priority:
        #   1. Exact device-id match — only when unambiguous (one matching
        #      entry). Old/buggy profiles where every panel saved the same
        #      _device_id (a now-fixed bug in enumerate_cameras) have many
        #      combo entries with that same ID; in that case fall through.
        #   2. Pick the Nth entry whose normalized name matches the saved
        #      device_name, where N = self.cam_index (so panel 1 picks the
        #      2nd See3CAM, etc.).
        #   3. Leave combo untouched, don't poison camera_slots.json.
        matched = False
        if saved_dev_id:
            candidates = [i for i in range(self.dev_combo.count())
                          if (self.dev_combo.itemData(i) and
                              self.dev_combo.itemData(i)[1] == saved_dev_id)]
            if len(candidates) == 1:
                self.dev_combo.setCurrentIndex(candidates[0]); matched = True
            # len > 1 → ambiguous, intentionally skip and use name-slot below
        if not matched and saved_dev_name:
            if self._select_dev_by_name_slot(saved_dev_name):
                matched = True
        if matched:
            d = self.dev_combo.currentData()
            new_id = d[1] if d else saved_dev_id
            self._pinned_device_id = new_id
            self._save_pin()
        res_str = data.get("_resolution", f"{DEFAULT_W}x{DEFAULT_H}")
        fps_val = data.get("_fps", DEFAULT_FPS)
        try:
            rw, rh = (int(x) for x in res_str.split("x"))
            self._cap_w, self._cap_h, self._cap_fps = rw, rh, fps_val
            lbl = f"{rw}x{rh}"
            if self.res_combo.findText(lbl) == -1: self.res_combo.addItem(lbl, userData=(rw, rh))
            self.res_combo.setCurrentText(lbl)
            fl = str(fps_val)
            if self.fps_combo.findText(fl) == -1: self.fps_combo.addItem(fl, userData=fps_val)
            self.fps_combo.setCurrentText(fl)
        except Exception: pass
        if "_out_dir" in data:
            self._out_dir = Path(data["_out_dir"])
            self.dir_btn.setText("📁  " + _short_path(self._out_dir))
            self.dir_btn.setToolTip(str(self._out_dir))
        if "_filename" in data:
            self._filename = data["_filename"]; self.name_edit.setText(self._filename)
        if "_bitrate" in data:
            br = float(data["_bitrate"]); self._cap_bitrate = br
            self.bitrate_spin.setValue(br)
            matched = False
            for i in range(self.qual_combo.count()):
                if self.qual_combo.itemData(i) == int(br):
                    self.qual_combo.setCurrentIndex(i); matched = True; break
            if not matched:
                for i in range(self.qual_combo.count()):
                    if self.qual_combo.itemData(i) == -1:
                        self.qual_combo.setCurrentIndex(i); break
        if "_rotation" in data:
            self._rotation = int(data["_rotation"]) % 360
            self.rotate_btn.setText(f"↻ {self._rotation}°" if self._rotation else "↻ 0°")
        if "_flip_h" in data:
            self._flip_h = bool(data["_flip_h"])
            self.flip_btn.setChecked(self._flip_h)
            self.flip_btn.setText("⇆ FLIP ●" if self._flip_h else "⇆ FLIP")
        if "_crop_rect" in data:
            cr = data["_crop_rect"]
            if cr and len(cr) == 4 and all(int(v) >= 0 for v in cr) and int(cr[2]) > 0 and int(cr[3]) > 0:
                self._crop_rect = (int(cr[0]), int(cr[1]), int(cr[2]), int(cr[3]))
            else:
                self._crop_rect = None
            self._update_crop_btn_label()
        if "_suffix_fmt" in data:
            self._suffix_fmt = data["_suffix_fmt"]
            matched = False
            for i in range(self.suffix_combo.count()):
                if self.suffix_combo.itemData(i) == self._suffix_fmt:
                    self.suffix_combo.setCurrentIndex(i); matched = True; break
            if not matched:
                self.suffix_combo.setCurrentIndex(self.suffix_combo.count() - 1)
                self.suffix_edit.setText(self._suffix_fmt)
        def _opt_float(key):
            if key not in data: return "_skip"
            v = data[key]
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        for _k in ("_mm_per_pixel", "_working_distance_mm",
                   "_calib_ref_length_mm", "_calib_ref_pixels"):
            v = _opt_float(_k)
            if v != "_skip":
                setattr(self, _k, v)
        self._update_calib_btn_label()
        auto_modes = data.get("_auto_modes", {})
        for name, val in data.items():
            if name.startswith("_"): continue
            if name in self.sliders:
                self.sliders[name].put(int(val))
                if self.thread: self.thread.set_prop(name, int(val))
            elif name in self.auto_sliders:
                self.auto_sliders[name].put(int(val))
                is_auto = auto_modes.get(name, True)
                self.auto_sliders[name].set_auto_mode(is_auto)
                if self.thread:
                    aid, av, mv = AUTO_CTRL[name]
                    self.thread.set_raw(aid, av if is_auto else mv)
                    if not is_auto: self.thread.set_prop(name, int(val))

    def closeEvent(self, e):
        self._disconnect(); super().closeEvent(e)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _short_path(p, n=38):
    s = str(p)
    return s if len(s) <= n else "…" + s[-(n-1):]


def _machine_label():
    """Best-effort machine identifier for session manifests. No network
    calls, no privileged APIs — just env vars that are always set on
    Windows. Returns 'computer/user' or just 'computer' if user is unset."""
    import os
    host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown"
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    return f"{host}/{user}" if user else host


# ── Help dialog ────────────────────────────────────────────────────────────────
_HELP_HTML = """
<style>
  body { font-family: Consolas, "Courier New", monospace; font-size: 11px; line-height: 1.45; }
  h2   { color: #FFD60A; font-size: 14px; margin: 14px 0 4px 0; }
  h3   { color: #4A9EFF; font-size: 12px; margin: 10px 0 4px 0; }
  code { background: #0C0C1E; color: #DDDDEE; padding: 1px 4px; border-radius: 3px; }
  kbd  { background: #1A1A2E; color: #DDDDEE; border: 1px solid #2A2A45;
         border-radius: 3px; padding: 1px 4px; font-size: 10px; }
  .dim { color: #9999BB; }
  ul   { margin: 4px 0 4px 18px; padding: 0; }
  li   { margin: 2px 0; }
  table { margin: 4px 0; }
  td   { padding: 2px 8px 2px 0; vertical-align: top; }
  .key { color: #FFD60A; }
</style>

<h2>Quick start</h2>
<ol>
  <li>Click <b>CONNECT</b> on each camera panel — preview should appear.</li>
  <li>(Optional) Set crop / flip / rotate to frame the behavior box.</li>
  <li>Click <b>📏 CAL</b> on each camera and calibrate against a ruler in the scene.</li>
  <li>(Optional) Click <b>TEST</b> to verify all cameras can write a 2-second file.</li>
  <li>Click <b>SAVE</b> in the top bar to save a profile so you don't redo this next session.</li>
  <li>Hit <b>⏺ RECORD ALL</b> (or per-camera record). Press <kbd>Space</kbd> for the same.</li>
</ol>

<h3>Keyboard shortcuts</h3>
<table>
  <tr><td><kbd>Space</kbd></td>  <td>Toggle RECORD ALL</td></tr>
  <tr><td><kbd>M</kbd></td>      <td>Drop a sync marker (recording only)</td></tr>
</table>
<p class="dim">Shortcuts auto-skip when a text input has focus.</p>

<h2>Camera panel buttons (header row)</h2>
<table>
  <tr><td><b>⇆ FLIP</b></td><td>Horizontal mirror. Toggle.</td></tr>
  <tr><td><b>✂ CROP</b></td><td>Drag a rectangle on the source frame. Recording uses cropped pixels.</td></tr>
  <tr><td><b>📏 CAL</b></td><td>Calibrate mm-per-pixel against a known-length object. See below.</td></tr>
  <tr><td><b>↻ 0°</b></td><td>Rotate preview 90° clockwise per click.</td></tr>
  <tr><td><b>↺ SYNC</b></td><td>Re-read current camera property values from hardware.</td></tr>
  <tr><td><b>▾ SETTINGS</b></td><td>Show / hide the slider panel.</td></tr>
</table>

<h2>📏 Calibration (mm-per-pixel)</h2>
<p>Each camera gets its own calibration because the lens is varifocal — every camera's
zoom-ring position changes its FOV. Without this, PixelPaws can't convert paw measurements
in pixels to real-world millimeters.</p>

<h3>How to calibrate</h3>
<ol>
  <li>Place a ruler (or anything you know the real length of) inside the camera's view.</li>
  <li>Click <b>📏 CAL</b>. A modal opens with a snapshot of the current preview.</li>
  <li>Click one end of the ruler, then the other. Yellow line + dots show your selection.</li>
  <li>Type the real length in <b>Reference length</b> (mm).</li>
  <li>Optionally type the camera-to-subject distance in <b>Working distance</b> (metadata only).</li>
  <li>Click <b>OK</b>. Button label switches to <code>📏 0.148 mm/px ●</code>.</li>
</ol>

<h3>When to recalibrate</h3>
<ul>
  <li>You moved the camera or changed its height.</li>
  <li>You twisted the zoom or focus ring on the lens.</li>
  <li>You changed the recording resolution.</li>
</ul>
<p class="dim">Crop, flip, and 90°/180°/270° rotation do <i>not</i> invalidate calibration —
pixel density stays the same.</p>

<h3>Where it goes</h3>
<ul>
  <li>Saved with the profile (<code>_mm_per_pixel</code>, <code>_working_distance_mm</code>, etc.).</li>
  <li>Stamped into every <code>.mp4</code> recorded after calibration as container metadata.</li>
  <li>Echoed in each session manifest sidecar JSON.</li>
  <li>Drawn live as a 100 mm scale bar on the bottom-left of the preview, so a wrong calibration is obvious before you record.</li>
</ul>

<h2>mp4 metadata (for PixelPaws)</h2>
<p>When a camera is calibrated, FFmpeg writes these tags into the recorded mp4's
<code>udta</code> atom (via <code>+use_metadata_tags</code>):</p>
<table>
  <tr><td><span class="key">mm_per_pixel</span></td>           <td>primary — mm in world / px in recording</td></tr>
  <tr><td><span class="key">working_distance_mm</span></td>     <td>optional, camera→subject distance</td></tr>
  <tr><td><span class="key">pixelpaws_calibrated</span></td>    <td><code>"1"</code> sentinel; absent on uncalibrated files</td></tr>
  <tr><td><span class="key">pixelpaws_ref_length_mm</span></td> <td>the real length you typed at calibration</td></tr>
  <tr><td><span class="key">pixelpaws_ref_pixels</span></td>    <td>pixel distance between the two clicks</td></tr>
</table>
<p>To read them:</p>
<pre><code>ffprobe -v 0 -show_entries format_tags -of json file.mp4</code></pre>
<p>Other tools that read these tags out of the box: <b>PyAV</b> (<code>container.metadata</code>),
<b>MediaInfo</b>, <b>MP4Box</b>, <b>mutagen</b>, <b>VLC</b> (Media Information → Metadata).
<b>OpenCV's VideoCapture</b> does <i>not</i> expose container metadata — read it separately.</p>

<h2>Profiles</h2>
<p>Profiles store every panel's settings: device pinning, resolution/fps, crop/flip/rotation,
slider values, output dir, filename, suffix format, and calibration.</p>
<ul>
  <li><b>SAVE</b> — if a profile is loaded, you'll be asked whether to overwrite it or save under a new name.</li>
  <li><b>LOAD</b> — applies the selected profile and starts connecting cameras one at a time.</li>
  <li><b>DELETE</b> — removes the profile file.</li>
  <li><b>EXPORT</b> — write the current profile to any path (great for sharing with another rig).</li>
  <li><b>IMPORT</b> — copy a profile JSON from any path into your profiles folder. Doesn't auto-load — pick it from the combo and click LOAD.</li>
</ul>
<p class="dim">Profiles live in <code>~/PawCapture/profiles/</code>.</p>

<h2>Recording</h2>
<ul>
  <li>Output rate is whatever you set in the <b>FPS</b> dropdown — strictly enforced via
      FFmpeg's <code>fps=</code> filter and forced timescale, regardless of clock skew.</li>
  <li>Encoder: NVENC → QSV → AMF → libx264, picked once at startup. Button label shows which.</li>
  <li>Default output dir: <code>~/PawCapture/recordings/</code>. Override per-camera in the panel.</li>
  <li>Recordings are grouped by day: <code>recordings/YYYY-MM-DD/</code>. The day folder is created on demand at record start.</li>
  <li><b>Phase</b> (top bar) tags a session as <i>baseline</i>, <i>post-drug</i>, <i>antagonist</i>, or a custom name.
      When set, files go into <code>recordings/YYYY-MM-DD/&lt;phase&gt;/</code> and the tag is added as a prefix or suffix on the filename.</li>
  <li>Default filename: <code>CAM_N_YYYYMMDD_HHMMSS.mp4</code>. Suffix format is configurable per camera.</li>
  <li>Before RECORD ALL starts, free space on each output drive is checked — you'll be warned if there's less than ~1 GB per active camera.</li>
</ul>

<h2>TEST button</h2>
<p>2-second dry run on every connected camera. Verifies each output file is produced and parses — useful before a long session to catch a misconfigured cam.</p>

<h2>📍 MARK button (sync markers)</h2>
<p>Only enabled while RECORD ALL is active. Drops a timestamped marker (with optional label) into the session manifest at the current recording offset. Bound to <kbd>M</kbd>.</p>

<h2>OFRS pairing (RWD photometry)</h2>
<p>Click <b>OFRS…</b> in the legend bar to set RWD-FPsystem's data root (typically <code>D:\\RWD-OFRS\\RWD-Data</code>) and toggle auto-pair. When auto-pair is on, PawCapture snapshots existing OFRS session folders at RECORD ALL start and, on stop, locates any new session folder created during the recording window. Events from that session's <code>Events.csv</code> are aligned to the PawCapture timeline (folder name → wall-clock start; <code>TimeStamp</code> column → session-relative ms) and merged as <b>MARKs</b>. The raw OFRS session metadata also lands in the manifest under <code>ofrs_sessions</code> so PixelPaws can read it without re-walking disk.</p>

<h2>Session manifest sidecar</h2>
<p>Every RECORD ALL session writes a <code>session_YYYYMMDD_HHMMSS.json</code> next to the recordings (in the day-folder) with:</p>
<ul>
  <li>Schema version (<code>pawcapture.session/v1</code>) and PawCapture version</li>
  <li>Per-camera files, calibration, resolution/fps/crop, and encoder used</li>
  <li>Sync marks (from the MARK button), profile name, machine label, start/end times</li>
</ul>
<p>PixelPaws can ingest the manifest in one shot instead of <code>ffprobe</code>'ing each file. The bundled <code>pawcapture_meta.py</code> module has helpers (<code>read_session_manifest</code>, <code>find_session_for_video</code>).</p>

<h2>PixelPaws reader module</h2>
<p>Drop <code>pawcapture_meta.py</code> (next to <code>camsync_precursor.py</code> in the source tree) into PixelPaws — no install needed. Provides:</p>
<pre><code>from pawcapture_meta import read_calibration, read_session_manifest

cal = read_calibration("CAM_1.mp4")          # dict | None
sess = read_session_manifest("session_*.json")</code></pre>
<p class="dim">Pure stdlib + ffprobe. Returns <code>None</code> for uncalibrated files so callers can skip them gracefully.</p>

<h2>Hardware notes — See3CAM_CU27 + CB-2812-3MP</h2>
<ul>
  <li>Native modes: MJPEG up to 100 fps, UYVY up to 60 fps. The driver always reports
      <i>30 fps</i> via OpenCV — that's a firmware lie, FFmpeg gets the real rates.</li>
  <li>Lens is a 2.8–12 mm varifocal CS-mount. Zoom ring position is per-camera, so
      each gets its own calibration.</li>
  <li>Wide end (2.8 mm) has visible barrel distortion; pixel scale is least accurate
      near frame edges.</li>
  <li><b>If a camera shows a tiled / mosaic pattern:</b> unplug and re-plug its USB to
      reset the ISP. Don't run two camsync instances against the same camera.</li>
</ul>

<h2>Troubleshooting</h2>
<table>
  <tr><td><b>No preview after CONNECT</b></td>
      <td>Click <b>⟳</b> next to the device combo to re-enumerate. Check Windows
          isn't holding the camera in another app.</td></tr>
  <tr><td><b>FPS shows "~50fps measured"</b></td>
      <td>Camera is genuinely under-delivering — usually UYVY auto-exposure
          dropping in low light. Lower exposure manually or add light.</td></tr>
  <tr><td><b>Sliders don't move the image</b></td>
      <td>Camera was reconnected mid-session — click <b>↺ SYNC</b>, or disconnect/connect.</td></tr>
  <tr><td><b>Recording smaller than configured</b></td>
      <td>You have a crop set — see the size shown on the <b>✂ CROP</b> button.</td></tr>
  <tr><td><b>mp4 has no calibration tags</b></td>
      <td>Either the camera wasn't calibrated before recording, or you loaded an
          old profile that overwrote the calibration. Recalibrate and re-record.</td></tr>
</table>

<p class="dim">Logs: <code>~/PawCapture/logs/</code> — useful when reporting issues.</p>
"""


class _HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PawCapture — Help")
        self.resize(860, 720)
        body = QTextBrowser()
        body.setOpenExternalLinks(True)
        body.setHtml(_HELP_HTML)
        body.setStyleSheet(f"""
            QTextBrowser{{
                background:{BG_DEEP};color:{TEXT_HI};
                border:1px solid {BORDER};border-radius:4px;
                padding:8px;
            }}
        """)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        # The Close button on QDialogButtonBox uses RejectRole — wire both
        # so Esc and the explicit click both dismiss the dialog cleanly.
        for b in bb.buttons():
            b.clicked.connect(self.accept)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(body)
        root.addWidget(bb)


class _OfrsConfigDialog(QDialog):
    """Sets the OFRS data root + auto-pair toggle. Persisted to
    OFRS_CONFIG_FILE. The dialog returns the new config dict via .result()
    when accepted; cancel leaves disk unchanged."""

    def __init__(self, parent, cfg):
        super().__init__(parent)
        from PyQt5.QtWidgets import QCheckBox
        self.setWindowTitle("OFRS pairing")
        self.resize(580, 240)
        self._cfg = dict(cfg or {})
        self.setStyleSheet(f"QDialog{{background:{BG_DEEP};color:{TEXT_HI};}}"
                           f"QLabel{{color:{TEXT_HI};font:11px {FONT};}}")

        v = QVBoxLayout(self); v.setContentsMargins(14, 12, 14, 12); v.setSpacing(8)
        title = QLabel("Auto-merge RWD OFRS events into the session manifest")
        title.setStyleSheet(f"color:{ACCENT};font:bold 12px {FONT};letter-spacing:1px;")
        v.addWidget(title)

        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Data root:"))
        self.root_edit = QLineEdit(self._cfg.get("data_root", ""))
        self.root_edit.setStyleSheet(f"""QLineEdit{{background:#0C0C1E;color:{TEXT_MED};
            border:1px solid {BORDER};border-radius:3px;padding:4px 8px;font:10px {FONT};}}
            QLineEdit:focus{{border-color:{ACCENT};color:{TEXT_HI};}}""")
        row.addWidget(self.root_edit)
        pick = QPushButton("Browse…"); pick.setFixedHeight(26)
        pick.setStyleSheet(f"""QPushButton{{color:{TEXT_HI};background:{BG_MID};
            border:1px solid {BORDER};border-radius:3px;padding:2px 10px;font:10px {FONT};}}
            QPushButton:hover{{border-color:{ACCENT};}}""")
        pick.clicked.connect(self._pick_root)
        row.addWidget(pick)
        v.addLayout(row)

        self.auto_cb = QCheckBox("Auto-pair OFRS session on RECORD ALL stop")
        self.auto_cb.setChecked(bool(self._cfg.get("auto_pair", True)))
        self.auto_cb.setStyleSheet(f"QCheckBox{{color:{TEXT_HI};font:11px {FONT};}}")
        v.addWidget(self.auto_cb)

        desc = QLabel(
            "When on, PawCapture snapshots OFRS session folders under the data "
            "root at RECORD ALL start, then on stop locates any new OFRS "
            "session(s) created during the recording window and merges "
            "Events.csv into the session manifest as MARKs.\n\n"
            "Default root for RWD-FPsystem on this machine is typically "
            "D:\\RWD-OFRS\\RWD-Data."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{TEXT_MED};font:10px {FONT};")
        v.addWidget(desc)
        v.addStretch()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "OFRS data root", self.root_edit.text() or str(Path.home()))
        if d:
            self.root_edit.setText(d)

    def chosen_config(self):
        return {
            "data_root": self.root_edit.text().strip(),
            "auto_pair": self.auto_cb.isChecked(),
        }


# ── Main Window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PawCapture")
        self.setMinimumSize(1100, 780)
        self.is_recording = False
        self._rec_start   = None
        # Tracks the profile most recently loaded so save_profile can
        # offer to overwrite it instead of silently making a new file.
        self._loaded_profile_name = None
        # Per-recording session state. Reset at every RECORD ALL start.
        # _session_panels holds the panels that were live at start time so
        # stop has a stable list (panels added/removed mid-record won't
        # cause weirdness in the manifest).
        self._session_id     = None
        self._session_marks  = []   # list of {"t_seconds": float, "label": str}
        self._session_panels = []
        # OFRS pairing state. Snapshot of OFRS session folders that exist at
        # RECORD ALL start; the diff at stop is what we merge.
        self._ofrs_cfg          = _load_ofrs_config()
        self._ofrs_pre_snapshot = set()
        self._ofrs_session_info = []  # populated at stop, used by manifest writer
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._build()

    def _build(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG_DEEP};color:{TEXT_HI};}}
            QScrollArea{{border:none;background:transparent;}}
            QScrollBar:vertical{{background:{BG_MID};width:6px;border-radius:3px;}}
            QScrollBar::handle:vertical{{background:{BORDER};border-radius:3px;}}
            QScrollBar:horizontal{{background:{BG_MID};height:6px;border-radius:3px;}}
            QScrollBar::handle:horizontal{{background:{BORDER};border-radius:3px;}}
        """)
        central = QWidget(); self.setCentralWidget(central)
        vbox = QVBoxLayout(central); vbox.setContentsMargins(0,0,0,0); vbox.setSpacing(0)

        # Top bar
        topbar = QWidget(); topbar.setFixedHeight(58)
        topbar.setStyleSheet(f"background:{BG_DEEP};border-bottom:1px solid {BORDER};")
        tbl = QHBoxLayout(topbar); tbl.setContentsMargins(18,0,18,0); tbl.setSpacing(8)
        brand = QLabel("PAWCAPTURE")
        brand.setStyleSheet(f"color:{ACCENT};font:bold 18px {FONT};letter-spacing:4px;")
        tbl.addWidget(brand)
        sub = QLabel("Multi-Camera Controller")
        sub.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};")
        tbl.addWidget(sub); tbl.addStretch()

        self.global_timer_lbl = QLabel("00:00:00")
        self.global_timer_lbl.setStyleSheet(f"color:{WARN};font:bold 13px {FONT};")
        self.global_timer_lbl.setVisible(False)
        tbl.addWidget(self.global_timer_lbl); tbl.addSpacing(12)

        prf_lbl = QLabel("PROFILE:")
        prf_lbl.setStyleSheet(f"color:{TEXT_DIM};font:10px {FONT};letter-spacing:1px;")
        self.profile_cb = QComboBox(); self.profile_cb.setFixedWidth(180)
        self.profile_cb.setStyleSheet(f"""QComboBox{{background:{BG_MID};color:{TEXT_MED};
            border:1px solid {BORDER};border-radius:4px;padding:4px 8px;font:10px {FONT};}}
            QComboBox::drop-down{{border:none;width:20px;}}
            QComboBox QAbstractItemView{{background:{BG_MID};color:{TEXT_MED};
                selection-background-color:{BORDER};}}""")
        self._refresh_profiles()
        load_btn = QPushButton("LOAD"); save_btn = QPushButton("SAVE"); del_btn = QPushButton("DELETE")
        imp_btn  = QPushButton("IMPORT"); exp_btn = QPushButton("EXPORT")
        for b, fg, bg in [(load_btn,ACCENT2,"#0D2040"),(save_btn,"#AAAACC","#1A1A2E"),
                          (del_btn,DANGER,"#200808"),
                          (imp_btn,TEXT_MED,BG_DEEP),(exp_btn,TEXT_MED,BG_DEEP)]:
            b.setStyleSheet(self._topbtn(fg,bg)); b.setFixedHeight(28)
        load_btn.clicked.connect(self.load_profile)
        save_btn.clicked.connect(self.save_profile)
        del_btn.clicked.connect(self.delete_profile)
        imp_btn.clicked.connect(self.import_profile)
        exp_btn.clicked.connect(self.export_profile)
        imp_btn.setToolTip("Import a profile JSON from any path (copies into ~/PawCapture/profiles)")
        exp_btn.setToolTip("Export the current profile JSON to any path")

        self.add_cam_btn = QPushButton("＋ Camera"); self.add_cam_btn.setFixedHeight(28)
        self.add_cam_btn.setStyleSheet(self._topbtn(SUCCESS,"#061A0C"))
        self.add_cam_btn.clicked.connect(self._add_camera)
        self.rem_cam_btn = QPushButton("－ Camera"); self.rem_cam_btn.setFixedHeight(28)
        self.rem_cam_btn.setStyleSheet(self._topbtn(TEXT_DIM,BG_DEEP))
        self.rem_cam_btn.clicked.connect(self._remove_camera)
        self.help_btn = QPushButton("? HELP"); self.help_btn.setFixedHeight(28)
        self.help_btn.setStyleSheet(self._topbtn(WARN,"#1A1605"))
        self.help_btn.clicked.connect(self._open_help)
        # Test recording: 2-second dry run on each connected camera, verifies
        # output files are produced and parse, reports OK/FAIL per camera.
        self.test_btn = QPushButton("TEST"); self.test_btn.setFixedHeight(28)
        self.test_btn.setStyleSheet(self._topbtn(ACCENT2,"#0D2040"))
        self.test_btn.setToolTip("2-second dry-run recording on every connected camera")
        self.test_btn.clicked.connect(self._run_test_recording)
        # Sync marker — only enabled while recording. Stamps a (t, label)
        # entry into the active session manifest.
        self.mark_btn = QPushButton("📍 MARK"); self.mark_btn.setFixedHeight(32)
        self.mark_btn.setStyleSheet(self._topbtn(WARN,"#1A1605"))
        self.mark_btn.setToolTip("Drop a sync marker into the active session (recording only)")
        self.mark_btn.setEnabled(False)
        self.mark_btn.clicked.connect(self._record_mark)

        tbl.addWidget(prf_lbl); tbl.addWidget(self.profile_cb)
        tbl.addWidget(load_btn); tbl.addWidget(save_btn); tbl.addWidget(del_btn)
        tbl.addWidget(imp_btn); tbl.addWidget(exp_btn)
        tbl.addSpacing(12)
        tbl.addWidget(self.add_cam_btn); tbl.addWidget(self.rem_cam_btn)
        tbl.addSpacing(8)
        tbl.addWidget(self.test_btn)
        tbl.addSpacing(8)
        tbl.addWidget(self.help_btn)
        tbl.addSpacing(12)
        self.rec_btn = QPushButton("⏺  RECORD ALL"); self.rec_btn.setFixedHeight(32)
        self.rec_btn.setStyleSheet(self._rec_style(False))
        self.rec_btn.clicked.connect(self._toggle_rec)
        tbl.addWidget(self.rec_btn)
        tbl.addSpacing(6)
        tbl.addWidget(self.mark_btn)

        # Keyboard shortcuts.  Space toggles RECORD ALL; M drops a sync
        # marker. Both auto-skip when a focused widget already accepts the
        # key (text inputs etc.) — Qt's default ShortcutContext is window-
        # wide which hits text edits too, so we use ApplicationShortcut +
        # an `acceptsKey` guard.
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        def _guarded(callback):
            def _fire():
                fw = QApplication.focusWidget()
                if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox)):
                    return
                callback()
            return _fire
        QShortcut(QKeySequence(Qt.Key_Space), self,
                  activated=_guarded(self._toggle_rec))
        QShortcut(QKeySequence("M"), self,
                  activated=_guarded(self._record_mark))
        vbox.addWidget(topbar)

        # Legend / phase bar. Phase = experimental stage tag (baseline,
        # post-drug, antagonist, custom). When set, recordings are placed in
        # a phase subfolder under the auto YYYY-MM-DD folder and the tag is
        # added as a prefix/suffix on the filename. Date folder stays auto.
        legend = QWidget(); legend.setFixedHeight(28)
        legend.setStyleSheet(f"background:{BG_MID};border-bottom:1px solid {BORDER};")
        lrow = QHBoxLayout(legend); lrow.setContentsMargins(20,0,20,0); lrow.setSpacing(6)

        phase_lbl = QLabel("PHASE:")
        phase_lbl.setStyleSheet(f"color:{TEXT_DIM};font:bold 10px {FONT};letter-spacing:1px;")
        self.phase_combo = QComboBox(); self.phase_combo.setFixedWidth(130)
        self.phase_combo.setStyleSheet(f"""QComboBox{{background:{BG_DEEP};color:{TEXT_HI};
            border:1px solid {BORDER};border-radius:3px;padding:2px 6px;font:10px {FONT};}}
            QComboBox:hover{{border-color:{ACCENT};}}
            QComboBox::drop-down{{border:none;width:18px;}}
            QComboBox QAbstractItemView{{background:{BG_MID};color:{TEXT_HI};
                selection-background-color:{BORDER};}}""")
        # userData is the sanitized tag used in folder/filename (None / "" =
        # off; "__custom__" sentinel reveals the free-text edit).
        for label, tag in [
            ("None",       ""),
            ("Baseline",   "baseline"),
            ("Post-Drug",  "post-drug"),
            ("Antagonist", "antagonist"),
            ("Custom…",    "__custom__"),
        ]:
            self.phase_combo.addItem(label, userData=tag)
        self.phase_custom_edit = QLineEdit()
        self.phase_custom_edit.setPlaceholderText("custom phase…")
        self.phase_custom_edit.setFixedWidth(140); self.phase_custom_edit.setVisible(False)
        self.phase_custom_edit.setStyleSheet(f"""QLineEdit{{background:{BG_DEEP};color:{ACCENT};
            border:1px solid {BORDER};border-radius:3px;padding:2px 6px;font:10px {FONT};}}
            QLineEdit:focus{{border-color:{ACCENT};}}""")
        self.phase_pos_combo = QComboBox(); self.phase_pos_combo.setFixedWidth(80)
        self.phase_pos_combo.setStyleSheet(f"""QComboBox{{background:{BG_DEEP};color:{TEXT_MED};
            border:1px solid {BORDER};border-radius:3px;padding:2px 6px;font:10px {FONT};}}
            QComboBox::drop-down{{border:none;width:18px;}}
            QComboBox QAbstractItemView{{background:{BG_MID};color:{TEXT_HI};
                selection-background-color:{BORDER};}}""")
        self.phase_pos_combo.addItem("Prefix", userData="prefix")
        self.phase_pos_combo.addItem("Suffix", userData="suffix")
        self.phase_preview_lbl = QLabel("")
        self.phase_preview_lbl.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")

        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        self.phase_pos_combo.currentIndexChanged.connect(self._update_phase_preview)
        self.phase_custom_edit.textChanged.connect(self._update_phase_preview)

        lrow.addWidget(phase_lbl); lrow.addWidget(self.phase_combo)
        lrow.addWidget(self.phase_custom_edit)
        lrow.addSpacing(6); lrow.addWidget(self.phase_pos_combo)
        lrow.addSpacing(8); lrow.addWidget(self.phase_preview_lbl)
        lrow.addStretch()

        # OFRS pairing — opens a small dialog; the status label reflects the
        # current config so users can tell at a glance whether events from
        # RWD's photometry app will be auto-merged into the manifest.
        self.ofrs_btn = QPushButton("OFRS…"); self.ofrs_btn.setFixedHeight(22)
        self.ofrs_btn.setToolTip("Configure RWD OFRS session pairing")
        self.ofrs_btn.setStyleSheet(f"""QPushButton{{color:{TEXT_MED};background:{BG_DEEP};
            border:1px solid {BORDER};border-radius:3px;padding:1px 8px;font:bold 9px {FONT};letter-spacing:1px;}}
            QPushButton:hover{{color:{ACCENT};border-color:{ACCENT};}}""")
        self.ofrs_btn.clicked.connect(self._open_ofrs_dialog)
        self.ofrs_status_lbl = QLabel("")
        self.ofrs_status_lbl.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")
        lrow.addWidget(self.ofrs_btn); lrow.addSpacing(4); lrow.addWidget(self.ofrs_status_lbl)
        lrow.addSpacing(20)

        for color, text in [(ACCENT,"● Auto — camera controls value"),(ACCENT2,"● Manual — slider sets value directly")]:
            l = QLabel(text); l.setStyleSheet(f"color:{color};font:9px {FONT};")
            lrow.addWidget(l); lrow.addSpacing(20)
        vbox.addWidget(legend)
        self._update_phase_preview()
        self._update_ofrs_status()

        # Camera panels
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.inner  = QWidget(); self.inner.setStyleSheet(f"background:{BG_DEEP};")
        self.panels_row = QHBoxLayout(self.inner)
        self.panels_row.setContentsMargins(16,16,16,16); self.panels_row.setSpacing(12)
        self.panels_row.addStretch()
        self.panels = []
        for i in range(3):
            self._insert_panel(CameraPanel(i, f"CAM {i+1}"))
        self.scroll.setWidget(self.inner)
        vbox.addWidget(self.scroll)

        # Status bar
        self.sb = QStatusBar()
        self.sb.setStyleSheet(f"QStatusBar{{background:{BG_DEEP};color:{TEXT_DIM};font:10px {FONT};border-top:1px solid {BORDER};}}")
        self.setStatusBar(self.sb)
        self.sb.showMessage(f"Recordings → {RECORDINGS_DIR}  |  FFmpeg logs → {LOGS_DIR}")

        self._global_tick = QTimer()
        self._global_tick.timeout.connect(self._update_global_timer)

    def _insert_panel(self, panel):
        self.panels_row.insertWidget(self.panels_row.count() - 1, panel)
        self.panels.append(panel); self._update_remove_btn()

    def _add_camera(self):
        n = len(self.panels)
        self._insert_panel(CameraPanel(n, f"CAM {n+1}"))

    def _remove_camera(self):
        if len(self.panels) <= 1: return
        panel = self.panels.pop()
        panel._disconnect(); self.panels_row.removeWidget(panel)
        panel.deleteLater(); self._update_remove_btn()

    def _update_remove_btn(self):
        self.rem_cam_btn.setEnabled(len(self.panels) > 1)

    # ── Phase tag (baseline / post-drug / antagonist / custom) ────────────────
    @staticmethod
    def _sanitize_phase_tag(s):
        """Strip phase tag down to filename-safe chars (alnum, dash, under).
        Spaces collapse to underscore, everything else dropped, lowercase."""
        out = []
        for ch in (s or "").strip().lower():
            if ch.isalnum() or ch in "-_":
                out.append(ch)
            elif ch.isspace():
                out.append("_")
        # Avoid leading/trailing punctuation that would look weird in a path.
        return "".join(out).strip("-_")

    def current_phase_tag(self):
        """Sanitized phase tag for folder + filename, or '' when off."""
        if not hasattr(self, "phase_combo"):
            return ""
        raw = self.phase_combo.currentData()
        if raw == "__custom__":
            raw = self.phase_custom_edit.text()
        return self._sanitize_phase_tag(raw)

    def current_phase_position(self):
        if not hasattr(self, "phase_pos_combo"):
            return "prefix"
        return self.phase_pos_combo.currentData() or "prefix"

    def _on_phase_changed(self, idx):
        is_custom = (self.phase_combo.itemData(idx) == "__custom__")
        self.phase_custom_edit.setVisible(is_custom)
        self._update_phase_preview()

    def _update_phase_preview(self):
        tag = self.current_phase_tag()
        if not tag:
            self.phase_preview_lbl.setText("→ recordings/YYYY-MM-DD/CAM_N.mp4")
            return
        pos = self.current_phase_position()
        sample = f"{tag}_CAM_N" if pos == "prefix" else f"CAM_N_{tag}"
        self.phase_preview_lbl.setText(f"→ recordings/YYYY-MM-DD/{tag}/{sample}.mp4")

    # ── RWD OFRS pairing ──────────────────────────────────────────────────────
    def _update_ofrs_status(self):
        cfg = self._ofrs_cfg or {}
        root = cfg.get("data_root", "")
        auto = bool(cfg.get("auto_pair", False))
        if not root:
            self.ofrs_status_lbl.setText("not configured")
            self.ofrs_status_lbl.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")
            return
        short = _short_path(Path(root)) if root else ""
        if auto:
            self.ofrs_status_lbl.setText(f"auto-pair on  ({short})")
            self.ofrs_status_lbl.setStyleSheet(f"color:{SUCCESS};font:9px {FONT};")
        else:
            self.ofrs_status_lbl.setText(f"off  ({short})")
            self.ofrs_status_lbl.setStyleSheet(f"color:{TEXT_DIM};font:9px {FONT};")

    def _open_ofrs_dialog(self):
        dlg = _OfrsConfigDialog(self, self._ofrs_cfg)
        if dlg.exec_() == QDialog.Accepted:
            self._ofrs_cfg = dlg.chosen_config()
            try:
                _save_ofrs_config(self._ofrs_cfg)
            except OSError as e:
                QMessageBox.warning(self, "OFRS config",
                                    f"Couldn't save {OFRS_CONFIG_FILE}:\n{e}")
            self._update_ofrs_status()

    def _ofrs_should_pair(self):
        cfg = self._ofrs_cfg or {}
        return bool(cfg.get("auto_pair")) and bool(cfg.get("data_root"))

    def _ofrs_take_snapshot(self):
        """Called at RECORD ALL start. Captures the set of existing OFRS
        session folders so the diff at stop is what was just recorded."""
        if not self._ofrs_should_pair():
            self._ofrs_pre_snapshot = set()
            return
        try:
            self._ofrs_pre_snapshot = _scan_ofrs_sessions(self._ofrs_cfg.get("data_root"))
        except Exception:
            self._ofrs_pre_snapshot = set()

    def _ofrs_collect_new(self, session_start, session_end):
        """Return a list of OFRS session folders that appeared during the
        recording window. Filters by folder-name datetime falling within
        [start - 60 s, end + 60 s] to absorb minor clock skew between OFRS
        and PawCapture (both run on this host so the slack is generous)."""
        if not self._ofrs_should_pair():
            return []
        post = _scan_ofrs_sessions(self._ofrs_cfg.get("data_root"))
        new  = post - (self._ofrs_pre_snapshot or set())
        if not new:
            return []
        from datetime import timedelta
        lo = session_start - timedelta(seconds=60)
        hi = session_end   + timedelta(seconds=60)
        out = []
        for d in new:
            ofrs_t = _parse_ofrs_session_start(d)
            if ofrs_t is None:
                continue
            if lo <= ofrs_t <= hi:
                out.append(d)
        # Deterministic order: by folder-name datetime (== creation time).
        out.sort(key=lambda d: _parse_ofrs_session_start(d) or datetime.min)
        return out

    def _topbtn(self, fg, bg):
        return (f"QPushButton{{color:{fg};background:{bg};border:1px solid {fg};"
                f"border-radius:4px;padding:4px 10px;font:bold 10px {FONT};letter-spacing:1px;}}"
                f"QPushButton:hover{{background:{fg};color:#000010;}}"
                f"QPushButton:disabled{{color:{TEXT_DIM};border-color:{TEXT_DIM};background:transparent;}}")

    def _rec_style(self, active):
        if active:
            return (f"QPushButton{{color:{DANGER};background:#1E0606;border:1px solid {DANGER};"
                    f"border-radius:4px;padding:4px 16px;font:bold 11px {FONT};letter-spacing:1px;}}"
                    f"QPushButton:hover{{background:{DANGER};color:white;}}")
        return (f"QPushButton{{color:{SUCCESS};background:#061A0C;border:1px solid {SUCCESS};"
                f"border-radius:4px;padding:4px 16px;font:bold 11px {FONT};letter-spacing:1px;}}"
                f"QPushButton:hover{{background:{SUCCESS};color:#000;}}")

    def _toggle_rec(self):
        if self.is_recording:
            self._stop_recording_session()
        else:
            self._start_recording_session()

    def _stop_recording_session(self):
        end_time = datetime.now()
        # Snapshot pre-state we'll need for the manifest *before* clearing.
        cam_records = []
        paths = []
        for p in self._session_panels:
            cam_data = self._panel_session_record(p)
            path = p.stop_rec()
            if path and str(path) != "None":
                paths.append(str(path))
                cam_data["file"] = str(path)
            cam_records.append(cam_data)
        # Resolve any new OFRS sessions that appeared during the recording
        # window and merge their events into the marks list. Done before
        # writing the manifest so it captures both the raw OFRS records and
        # the aligned marks.
        ofrs_msg = self._merge_ofrs_for_session(end_time)
        manifest_path = self._write_session_manifest(cam_records, end_time)
        self.is_recording = False
        self._global_tick.stop()
        self.global_timer_lbl.setText("00:00:00"); self.global_timer_lbl.setVisible(False)
        self._rec_start = None
        self._session_panels = []
        self._session_marks  = []
        self._ofrs_session_info = []
        self.rec_btn.setText("⏺  RECORD ALL"); self.rec_btn.setStyleSheet(self._rec_style(False))
        if hasattr(self, "mark_btn"): self.mark_btn.setEnabled(False)
        msg = f"Saved: {' | '.join(paths) if paths else 'no files'}"
        if manifest_path:
            msg += f"  |  manifest: {manifest_path.name}"
        if ofrs_msg:
            msg += f"  |  {ofrs_msg}"
        self.sb.showMessage(msg)

    def _merge_ofrs_for_session(self, end_time):
        """Find OFRS sessions created during the recording window and merge
        their events into self._session_marks. Records the session metadata
        on self._ofrs_session_info so the manifest writer can emit it. Returns
        a short status string for the status bar, or '' when nothing merged."""
        self._ofrs_session_info = []
        if not self._ofrs_should_pair() or self._rec_start is None:
            return ""
        new_dirs = self._ofrs_collect_new(self._rec_start, end_time)
        if not new_dirs:
            return "OFRS: no new session"
        total_events = 0
        for d in new_dirs:
            ofrs_start = _parse_ofrs_session_start(d)
            events     = _read_ofrs_events(d)
            marks      = _align_ofrs_events(d, self._rec_start)
            self._session_marks.extend(marks)
            total_events += len(events)
            self._ofrs_session_info.append({
                "session_dir":  str(d),
                "session_name": Path(d).name,
                "session_start_local": (ofrs_start.isoformat(timespec="seconds")
                                        if ofrs_start else None),
                "event_count":  len(events),
                "events":       events,
            })
        n_sess = len(new_dirs)
        return f"OFRS: paired {n_sess} session{'s' if n_sess != 1 else ''} ({total_events} events)"

    def _start_recording_session(self):
        live = [p for p in self.panels if p._live]
        if not live:
            QMessageBox.warning(self, "No Camera", "Connect at least one camera first."); return
        ok, msg = self._disk_space_ok(live)
        if not ok:
            if QMessageBox.question(self, "Low Disk Space",
                    f"{msg}\n\nProceed anyway?") != QMessageBox.Yes:
                return
        errors = []; started = []
        for p in live:
            ok, err = p.start_rec()
            if ok: started.append(p)
            else: errors.append(f"{p.label}: {err}")
        if errors:
            for p in started: p.stop_rec()
            QMessageBox.critical(self, "Record Error",
                f"FFmpeg failed:\n\n" + "\n".join(errors) + f"\n\nCheck logs:\n{LOGS_DIR}")
            return
        self.is_recording = True; self._rec_start = datetime.now()
        self._session_id     = self._rec_start.strftime("%Y%m%d_%H%M%S")
        self._session_marks  = []
        self._session_panels = list(started)
        self._ofrs_session_info = []
        self._ofrs_take_snapshot()
        if hasattr(self, "mark_btn"): self.mark_btn.setEnabled(True)
        codecs    = [getattr(p.recorder, "codec_used", "?") for p in started]
        gpu_codecs = ("h264_nvenc", "h264_qsv", "h264_amf")
        gpu_count = sum(1 for c in codecs if c in gpu_codecs)
        cpu_count = len(codecs) - gpu_count
        enc_info  = []
        if gpu_count:
            vendor = next((c for c in codecs if c in gpu_codecs), "")
            tag = {"h264_nvenc": "NVENC", "h264_qsv": "QSV",
                   "h264_amf": "AMF"}.get(vendor, "GPU")
            enc_info.append(f"{gpu_count}× GPU ({tag})")
        if cpu_count: enc_info.append(f"{cpu_count}× CPU (x264)")
        self._global_tick.start(1000); self.global_timer_lbl.setVisible(True)
        self.rec_btn.setText("⏹  STOP ALL"); self.rec_btn.setStyleSheet(self._rec_style(True))
        self.sb.showMessage(f"Recording {len(started)} camera(s) — {' + '.join(enc_info)}")

    def _update_global_timer(self):
        if self._rec_start:
            secs = int((datetime.now() - self._rec_start).total_seconds())
            h, r = divmod(secs, 3600); m, s = divmod(r, 60)
            self.global_timer_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _panel_session_record(self, panel):
        """Snapshot the per-panel info that goes into the session manifest."""
        rec = {
            "label":         panel.label,
            "device_name":   panel.dev_combo.currentText(),
            "device_id":     panel._pinned_device_id or "",
            "resolution":    f"{panel._cap_w}x{panel._cap_h}",
            "fps":           panel._cap_fps,
            "bitrate_mbps":  panel._cap_bitrate,
            "rotation":      panel._rotation,
            "flip_h":        panel._flip_h,
            "crop_rect":     list(panel._crop_rect) if panel._crop_rect else None,
            "mm_per_pixel":  panel._mm_per_pixel,
            "working_distance_mm":   panel._working_distance_mm,
            "ref_length_mm":         panel._calib_ref_length_mm,
            "ref_pixels":            panel._calib_ref_pixels,
            "encoder":       getattr(panel.recorder, "codec_used", ""),
        }
        return rec

    def _write_session_manifest(self, cam_records, end_time):
        """Write a JSON sidecar describing the just-completed session.
        Saved next to the videos (in the day-folder of the first cam's
        out_dir). Returns the manifest path on success, None on failure."""
        if not cam_records or self._rec_start is None:
            return None
        # Pick the dir of the first cam that produced a file; fall back to
        # the first session panel's _out_dir + day folder if none did.
        target_dir = None
        for rec in cam_records:
            f = rec.get("file")
            if f:
                target_dir = Path(f).parent; break
        if target_dir is None and self._session_panels:
            base = self._session_panels[0]._out_dir
            target_dir = base / self._rec_start.strftime("%Y-%m-%d")
        if target_dir is None:
            return None
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        manifest = {
            "schema":        "pawcapture.session/v1",
            "pawcapture_version": PAWCAPTURE_VERSION,
            "session_id":    self._session_id,
            "started_at":    self._rec_start.isoformat(timespec="seconds"),
            "ended_at":      end_time.isoformat(timespec="seconds"),
            "duration_s":    round((end_time - self._rec_start).total_seconds(), 3),
            "profile":       self._loaded_profile_name,
            "machine":       _machine_label(),
            "marks":         list(self._session_marks),
            "cameras":       cam_records,
        }
        if self._ofrs_session_info:
            manifest["ofrs_sessions"] = list(self._ofrs_session_info)
        out_path = target_dir / f"session_{self._session_id}.json"
        try:
            out_path.write_text(json.dumps(manifest, indent=2))
        except OSError:
            return None
        return out_path

    def _disk_space_ok(self, panels):
        """Check unique out_dirs used by `panels` have enough free space.
        Heuristic: 1 GB per active camera per dir is plenty for several
        minutes at 8 Mbps per cam. Returns (ok, message)."""
        per_cam_bytes = 1 * 1024 * 1024 * 1024
        # Group panels by (resolved) out_dir so two panels writing to the
        # same volume only get checked once but with a doubled requirement.
        by_dir = {}
        for p in panels:
            try:
                d = Path(p._out_dir).resolve()
            except OSError:
                d = Path(p._out_dir)
            by_dir.setdefault(d, 0)
            by_dir[d] += 1
        warnings = []
        for d, count in by_dir.items():
            try:
                free = shutil.disk_usage(d if d.exists() else d.anchor).free
            except OSError:
                continue
            need = per_cam_bytes * count
            if free < need:
                warnings.append(
                    f"{d}: {free/1e9:.1f} GB free, {need/1e9:.1f} GB recommended "
                    f"({count} cam{'s' if count != 1 else ''})"
                )
        if warnings:
            return False, "Low disk space:\n  " + "\n  ".join(warnings)
        return True, ""

    def _record_mark(self):
        """Append a sync marker to the active session. No-op when not
        recording — the button should already be disabled then."""
        if not self.is_recording or self._rec_start is None:
            return
        t = (datetime.now() - self._rec_start).total_seconds()
        label, ok = QInputDialog.getText(
            self, "Sync mark",
            f"Marker label (optional). t = {t:.2f}s into recording.",
        )
        # Cancel → don't record the mark.
        if not ok:
            return
        self._session_marks.append({
            "t_seconds": round(t, 3),
            "label":     label.strip(),
            "wall_time": datetime.now().isoformat(timespec="seconds"),
        })
        self.sb.showMessage(
            f"Mark @ {t:.2f}s  ({len(self._session_marks)} total)"
        )

    def save_profile(self):
        # When a profile is currently loaded, default to overwriting it so
        # the user can iterate (tweak settings → save) without spawning a
        # new file every time. They can still pick "Save As New" to fork.
        loaded = self._loaded_profile_name
        target_name = None
        if loaded and (PROFILES_DIR / f"{loaded}.json").exists():
            box = QMessageBox(self)
            box.setWindowTitle("Save Profile")
            box.setText(f"Overwrite profile '{loaded}'?")
            box.setInformativeText("Yes overwrites the loaded profile. "
                                   "No saves under a new name.")
            ow_btn  = box.addButton("Overwrite", QMessageBox.AcceptRole)
            new_btn = box.addButton("Save As New", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is ow_btn:
                target_name = loaded
            elif clicked is new_btn:
                target_name = None
            else:
                return
        if target_name is None:
            name, ok = QInputDialog.getText(self, "Save Profile", "Profile name:",
                                            text=loaded or "")
            if not ok or not name.strip(): return
            target_name = name.strip()
            collide_path = PROFILES_DIR / f"{target_name}.json"
            if target_name != loaded and collide_path.exists():
                if QMessageBox.question(
                        self, "Save Profile",
                        f"'{target_name}' already exists. Overwrite?"
                    ) != QMessageBox.Yes:
                    return
        path = PROFILES_DIR / f"{target_name}.json"
        path.write_text(json.dumps({"cameras": [p.get_settings() for p in self.panels]}, indent=2))
        self._loaded_profile_name = target_name
        self._refresh_profiles()
        # Re-select the saved profile in the combo for clarity.
        idx = self.profile_cb.findText(target_name)
        if idx >= 0:
            self.profile_cb.setCurrentIndex(idx)
        self.sb.showMessage(f"Profile saved: {path.name}")

    def load_profile(self):
        name = self.profile_cb.currentText()
        if not name or name.startswith("—"): return
        path = PROFILES_DIR / f"{name}.json"
        if not path.exists(): return
        data = json.loads(path.read_text())
        cam_list = data.get("cameras", [])
        for p in self.panels:
            if p._live: p._disconnect()
        for i, cam_data in enumerate(cam_list):
            if i < len(self.panels): self.panels[i].apply_settings(cam_data)
        # Collision recovery: legacy profiles saved the same _device_id for
        # every panel (a bug in enumerate_cameras' WMI map, fixed 2026-05-06).
        # Loading those would set every panel's combo to the first matching
        # entry. If two or more panels ended up on the same combo index, fall
        # back to the name-slot rule for each so panel N maps to the Nth
        # device of that name.
        active_panels = self.panels[:len(cam_list)]
        sel_indices = [p.dev_combo.currentIndex() for p in active_panels]
        if len(set(sel_indices)) < len(sel_indices):
            for p, cd in zip(active_panels, cam_list):
                name = cd.get("_device_name", "")
                if name and p._select_dev_by_name_slot(name):
                    d = p.dev_combo.currentData()
                    if d:
                        p._pinned_device_id = d[1]
                        p._save_pin()
        saved_ids = {cd.get("_device_id","") for cd in cam_list if cd.get("_device_id")}

        def _connect_next(idx):
            if idx >= len(self.panels): return
            p = self.panels[idx]
            if not p._live:
                panel_saved_id = p._pinned_device_id or ""
                if panel_saved_id in saved_ids or idx < len(cam_list):
                    p._connect()
            QTimer.singleShot(900, lambda: _connect_next(idx + 1))

        QTimer.singleShot(200, lambda: _connect_next(0))
        self._loaded_profile_name = name
        self.sb.showMessage(f"Profile '{name}' loaded — connecting cameras…")

    def delete_profile(self):
        name = self.profile_cb.currentText()
        if not name or name.startswith("—"): return
        if QMessageBox.question(self,"Delete",f"Delete '{name}'?") == QMessageBox.Yes:
            (PROFILES_DIR / f"{name}.json").unlink(missing_ok=True)
            if self._loaded_profile_name == name:
                self._loaded_profile_name = None
            self._refresh_profiles()

    def export_profile(self):
        """Write the current panel state to a user-chosen path so it can be
        emailed/shared without the recipient having to dig in
        ~/PawCapture/profiles."""
        default_name = (self._loaded_profile_name or "profile") + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", default_name, "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(
                {"cameras": [p.get_settings() for p in self.panels]}, indent=2
            ))
        except OSError as e:
            QMessageBox.warning(self, "Export failed", str(e)); return
        self.sb.showMessage(f"Profile exported: {Path(path).name}")

    def import_profile(self):
        """Copy a profile JSON from anywhere into PROFILES_DIR. Validates
        it parses + has a `cameras` array; doesn't auto-load (user picks
        from the combo)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Import failed",
                f"Couldn't parse {path}:\n{e}"); return
        if not isinstance(data, dict) or "cameras" not in data:
            QMessageBox.warning(self, "Import failed",
                "JSON is missing the top-level 'cameras' array.")
            return
        target = PROFILES_DIR / Path(path).name
        if target.exists():
            if QMessageBox.question(self, "Import Profile",
                f"A profile named '{target.stem}' already exists in your "
                "profiles folder. Overwrite?") != QMessageBox.Yes:
                return
        try:
            target.write_text(Path(path).read_text())
        except OSError as e:
            QMessageBox.warning(self, "Import failed", str(e)); return
        self._refresh_profiles()
        idx = self.profile_cb.findText(target.stem)
        if idx >= 0:
            self.profile_cb.setCurrentIndex(idx)
        self.sb.showMessage(f"Profile imported: {target.name} (select and click LOAD)")

    def _run_test_recording(self):
        """Two-second dry-run: kick off recording on every live cam, wait,
        stop, then verify each output file exists and is non-empty.
        Doesn't write a session manifest (intentionally — these aren't
        real sessions). Returns immediately if a real recording is
        active."""
        if self.is_recording:
            QMessageBox.information(self, "Test", "Stop the active recording first.")
            return
        live = [p for p in self.panels if p._live]
        if not live:
            QMessageBox.warning(self, "Test", "Connect at least one camera first.")
            return
        self.test_btn.setEnabled(False); self.test_btn.setText("TESTING…")
        QApplication.processEvents()
        started = []
        errors = []
        for p in live:
            ok, err = p.start_rec()
            if ok: started.append(p)
            else: errors.append(f"{p.label}: {err}")
        # Hold the recording for ~2 seconds without blocking the event
        # loop — keep the preview alive and the UI responsive.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)
        results = []
        for p in started:
            path = p.stop_rec()
            try:
                ok = bool(path) and Path(str(path)).exists() and Path(str(path)).stat().st_size > 1024
            except OSError:
                ok = False
            results.append((p.label, "OK" if ok else "FAIL", str(path) if path else "(no file)"))
        self.test_btn.setEnabled(True); self.test_btn.setText("TEST")
        body_lines = []
        if errors:
            body_lines.append("Failed to start:")
            body_lines += [f"  • {e}" for e in errors]
        if results:
            body_lines.append("Results:")
            body_lines += [f"  • {lbl}: {status}  —  {pth}" for (lbl, status, pth) in results]
        any_fail = bool(errors) or any(r[1] != "OK" for r in results)
        msgbox = QMessageBox.warning if any_fail else QMessageBox.information
        msgbox(self, "Test recording",
               "\n".join(body_lines) if body_lines else "No cameras tested.")

    def _refresh_profiles(self):
        self.profile_cb.clear(); self.profile_cb.addItem("— Select Profile —")
        for f in sorted(PROFILES_DIR.glob("*.json")): self.profile_cb.addItem(f.stem)

    def _open_help(self):
        _HelpDialog(self).exec_()

    def closeEvent(self, e):
        if self.is_recording:
            for p in self.panels: p.stop_rec()
        for p in self.panels: p._disconnect()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor(BG_DEEP))
    pal.setColor(QPalette.WindowText,    QColor(TEXT_HI))
    pal.setColor(QPalette.Base,          QColor(BG_MID))
    pal.setColor(QPalette.AlternateBase, QColor(BG_CARD))
    pal.setColor(QPalette.Text,          QColor(TEXT_HI))
    pal.setColor(QPalette.Button,        QColor(BG_CARD))
    pal.setColor(QPalette.ButtonText,    QColor(TEXT_HI))
    app.setPalette(pal)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
