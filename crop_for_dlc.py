"""
crop_for_dlc.py — Video cropping tool for DeepLabCut prep.

Crops videos spatially so DLC can be run on a focused ROI.
Saves a sidecar JSON with crop parameters that map directly to
crop_offset_x / crop_offset_y in PixelPaws brightness_features.py.

Usage
-----
  # Standalone tkinter GUI (double-click or launched from PixelPaws Tools tab)
  python crop_for_dlc.py

  # Interactive crop on a single video (OpenCV preview)
  python crop_for_dlc.py session.mp4

  # Direct params — no preview needed
  python crop_for_dlc.py session.mp4 --x 120 --y 40 --w 640 --h 480

  # Batch — same crop applied to every .mp4 in a folder
  python crop_for_dlc.py --batch videos/ --x 120 --y 40 --w 640 --h 480

  # Write offsets into PixelPaws project config as well
  python crop_for_dlc.py session.mp4 --project /path/to/project
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH and responds to --version."""
    return shutil.which("ffmpeg") is not None


def parse_time(s: str):
    """Parse HH:MM:SS, MM:SS, or plain seconds string. Returns float or None if blank."""
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse time '{s}' — use HH:MM:SS or seconds")


def _get_video_info(video_path: str):
    """Returns (total_frames, fps, duration_sec) or (0, 30.0, 0) on failure."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total, fps, total / fps
    except Exception:
        return 0, 30.0, 0


def _get_frame_count(video_path: str, start_time=None, end_time=None) -> int:
    total, fps, duration = _get_video_info(video_path)
    if total == 0:
        return 0
    start = start_time or 0.0
    end = min(end_time, duration) if end_time is not None else duration
    return max(0, int((end - start) * fps))


def crop_video_ffmpeg(input_path: str, output_path: str,
                      x: int, y: int, w: int, h: int,
                      crf: int = 23, total_frames: int = 0,
                      start_time=None, end_time=None,
                      progress_fn=None) -> None:
    cmd = ["ffmpeg", "-y"]
    if start_time is not None:
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", input_path]
    if end_time is not None:
        cmd += ["-to", str(end_time)]
    cmd += [
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "ultrafast",
        "-an",
        "-progress", "pipe:1",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        if line.startswith("frame=") and progress_fn and total_frames > 0:
            try:
                progress_fn(min(int(line.split("=")[1]) / total_frames, 1.0))
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed — check that the crop region fits within the video frame.")


def crop_video_opencv(input_path: str, output_path: str,
                      x: int, y: int, w: int, h: int,
                      log_fn=None, progress_fn=None,
                      start_time=None, end_time=None) -> None:
    """Fallback: crop frame-by-frame with OpenCV VideoWriter (mp4v)."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for the fallback encoder.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    end_frame = int(end_time * fps) if end_time is not None else total
    trimmed_total = end_frame - int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_num = 0
    while True:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[y:y + h, x:x + w])
        frame_num += 1
        if progress_fn and trimmed_total > 0 and frame_num % 30 == 0:
            progress_fn(frame_num / trimmed_total)
        elif log_fn and frame_num % 100 == 0:
            log_fn(f"  Processed {frame_num}/{trimmed_total} frames…")

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# Interactive crop selection (OpenCV mouse UI)
# ---------------------------------------------------------------------------

def select_crop_interactive(video_path: str):
    """
    Show the first frame of *video_path* in an OpenCV window.
    The user clicks and drags to select a crop rectangle.

    Returns (x, y, w, h) or None if the user cancelled.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for interactive crop selection.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Could not read frame from: {video_path}")

    state = {
        "drawing": False,
        "start": None,
        "end": None,
        "confirmed": False,
        "cancelled": False,
    }
    display = frame.copy()

    def mouse_cb(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (mx, my)
            state["end"] = (mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (mx, my)

    win = "Crop Selection — drag rectangle | Enter=confirm  R=reset  Q=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        img = display.copy()

        if state["start"] and state["end"]:
            x0, y0 = state["start"]
            x1, y1 = state["end"]
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            label = f"({rx}, {ry})  {rw}x{rh}"
            cv2.putText(img, label, (rx, max(ry - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            if state["start"] and state["end"]:
                state["confirmed"] = True
                break
        elif key in (ord("r"), ord("R")):
            state["start"] = None
            state["end"] = None
        elif key in (ord("q"), ord("Q"), 27):  # Q or Esc
            state["cancelled"] = True
            break

    cv2.destroyAllWindows()

    if state["cancelled"] or not (state["start"] and state["end"]):
        return None

    x0, y0 = state["start"]
    x1, y1 = state["end"]
    rx, ry = min(x0, x1), min(y0, y1)
    rw, rh = abs(x1 - x0), abs(y1 - y0)
    if rw < 4 or rh < 4:
        return None
    return rx, ry, rw, rh


# ---------------------------------------------------------------------------
# Sidecar JSON + project config
# ---------------------------------------------------------------------------

def save_crop_sidecar(video_path: str, x: int, y: int, w: int, h: int) -> str:
    """Write <stem>_crop.json next to the video. Returns the sidecar path."""
    p = Path(video_path)
    sidecar = p.with_name(p.stem + "_crop.json")
    data = {"x": x, "y": y, "w": w, "h": h, "source": p.name}
    sidecar.write_text(json.dumps(data, indent=2))
    return str(sidecar)


def update_project_config(project_folder: str, x: int, y: int) -> None:
    """Add crop_offset_x/y to PixelPaws_project.json (create if absent)."""
    cfg_path = Path(project_folder) / "PixelPaws_project.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            pass
    cfg["crop_offset_x"] = x
    cfg["crop_offset_y"] = y
    cfg_path.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# High-level processing
# ---------------------------------------------------------------------------

def process_single(video_path: str, x: int, y: int, w: int, h: int,
                   output_dir: str = None, project_folder: str = None,
                   crf: int = 23, start_time=None, end_time=None,
                   log_fn=None, progress_fn=None) -> str:
    _log = log_fn or print

    p = Path(video_path)
    out_dir = Path(output_dir) if output_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / (p.stem + "_cropped" + p.suffix))

    _log(f"Cropping: {p.name}  →  {Path(output_path).name}")
    _log(f"  Region: x={x} y={y} w={w} h={h}")
    if start_time is not None or end_time is not None:
        _log(f"  Time:   {start_time or 0:.1f}s → {end_time or 'end'}")

    total = _get_frame_count(video_path, start_time, end_time)

    if check_ffmpeg():
        _log(f"  Encoder: FFmpeg (H.264, CRF {crf})")
        crop_video_ffmpeg(video_path, output_path, x, y, w, h,
                          crf=crf, total_frames=total,
                          start_time=start_time, end_time=end_time,
                          progress_fn=progress_fn)
    else:
        _log("  Encoder: OpenCV fallback (FFmpeg not found)")
        crop_video_opencv(video_path, output_path, x, y, w, h,
                          log_fn=_log, progress_fn=progress_fn,
                          start_time=start_time, end_time=end_time)

    sidecar = save_crop_sidecar(video_path, x, y, w, h)
    _log(f"  Sidecar: {Path(sidecar).name}")

    if project_folder:
        update_project_config(project_folder, x, y)
        _log(f"  Project config updated: {project_folder}")

    _log("  Done.")
    return output_path


def process_batch(folder: str, x: int, y: int, w: int, h: int,
                  output_dir: str = None, project_folder: str = None,
                  crf: int = 23, start_time=None, end_time=None,
                  log_fn=None, progress_fn=None) -> list:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    videos = sorted(f for f in Path(folder).iterdir()
                    if f.is_file() and f.suffix.lower() in video_exts)
    if not videos:
        (log_fn or print)(f"No video files found in: {folder}")
        return []

    outputs = []
    for i, vp in enumerate(videos):
        def _prog(p, i=i, n=len(videos)):
            if progress_fn:
                progress_fn((i + p) / n)
        out = process_single(str(vp), x, y, w, h, output_dir, project_folder,
                             crf=crf, start_time=start_time, end_time=end_time,
                             log_fn=log_fn, progress_fn=_prog)
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# Standalone tkinter GUI
# ---------------------------------------------------------------------------

class CropForDLCApp:
    """Self-contained tkinter window for the crop tool."""

    def __init__(self, root: tk.Tk, initial_project: str = ""):
        self.root = root
        root.title("PixelPaws — Crop Video for DLC")
        root.geometry("700x600")
        root.resizable(True, True)

        self._build_ui()

        if initial_project:
            self.project_var.set(initial_project)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ---- File selection ----
        file_frame = ttk.LabelFrame(self.root, text="Video Source")
        file_frame.pack(fill="x", padx=12, pady=8)

        # Mode
        self.mode_var = tk.StringVar(value="single")
        mode_row = ttk.Frame(file_frame)
        mode_row.pack(fill="x", **pad)
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Single video", variable=self.mode_var,
                        value="single", command=self._on_mode_change).pack(side="left", padx=4)
        ttk.Radiobutton(mode_row, text="Batch folder", variable=self.mode_var,
                        value="batch", command=self._on_mode_change).pack(side="left", padx=4)

        # Video / folder path
        path_row = ttk.Frame(file_frame)
        path_row.pack(fill="x", **pad)
        ttk.Label(path_row, text="Video / Folder:").pack(side="left")
        self.video_var = tk.StringVar()
        self.video_var.trace_add("write", self._on_video_changed)
        ttk.Entry(path_row, textvariable=self.video_var, width=52).pack(
            side="left", padx=4, fill="x", expand=True)
        self.browse_btn = ttk.Button(path_row, text="Browse…",
                                     command=self._browse_video)
        self.browse_btn.pack(side="left")
        self.duration_label = ttk.Label(file_frame, text="", foreground="gray")
        self.duration_label.pack(**pad)

        # Output dir (optional)
        out_row = ttk.Frame(file_frame)
        out_row.pack(fill="x", **pad)
        ttk.Label(out_row, text="Output folder (optional):").pack(side="left")
        self.outdir_var = tk.StringVar()
        ttk.Entry(out_row, textvariable=self.outdir_var, width=40).pack(
            side="left", padx=4, fill="x", expand=True)
        ttk.Button(out_row, text="Browse…", command=self._browse_outdir).pack(side="left")

        # ---- Crop parameters ----
        crop_frame = ttk.LabelFrame(self.root, text="Crop Parameters")
        crop_frame.pack(fill="x", padx=12, pady=4)

        param_row = ttk.Frame(crop_frame)
        param_row.pack(**pad)

        self.x_var = tk.IntVar(value=286)
        self.y_var = tk.IntVar(value=0)
        self.w_var = tk.IntVar(value=761)
        self.h_var = tk.IntVar(value=720)

        for label, var in [("X:", self.x_var), ("Y:", self.y_var),
                           ("W:", self.w_var), ("H:", self.h_var)]:
            ttk.Label(param_row, text=label).pack(side="left")
            ttk.Spinbox(param_row, textvariable=var, from_=0, to=9999,
                        width=6).pack(side="left", padx=4)

        quality_row = ttk.Frame(crop_frame)
        quality_row.pack(**pad)
        self.crf_var = tk.IntVar(value=23)
        ttk.Label(quality_row, text="Quality (CRF):").pack(side="left")
        ttk.Spinbox(quality_row, textvariable=self.crf_var, from_=18, to=40,
                    width=4).pack(side="left", padx=4)
        ttk.Label(quality_row, text="18 = best quality / largest   40 = smallest / lower quality",
                  foreground="gray").pack(side="left", padx=4)

        ttk.Button(crop_frame, text="Preview & Select (OpenCV)",
                   command=self._preview_and_select).pack(pady=4)

        # ---- Time trim ----
        time_frame = ttk.LabelFrame(self.root, text="Time Trim (optional)")
        time_frame.pack(fill="x", padx=12, pady=4)

        self.trim_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(time_frame, text="Enable time trim",
                        variable=self.trim_enabled,
                        command=self._on_trim_toggle).pack(anchor="w", **pad)

        trim_row = ttk.Frame(time_frame)
        trim_row.pack(fill="x", **pad)
        ttk.Label(trim_row, text="Start (HH:MM:SS):").pack(side="left")
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(trim_row, textvariable=self.start_var, width=10, state="disabled")
        self.start_entry.pack(side="left", padx=4)
        ttk.Label(trim_row, text="End (HH:MM:SS):").pack(side="left", padx=(12, 0))
        self.end_var = tk.StringVar()
        self.end_entry = ttk.Entry(trim_row, textvariable=self.end_var, width=10, state="disabled")
        self.end_entry.pack(side="left", padx=4)
        ttk.Label(trim_row, text="leave blank = beginning / end of file",
                  foreground="gray").pack(side="left", padx=8)

        # ---- Project config ----
        proj_frame = ttk.LabelFrame(self.root, text="PixelPaws Project (optional)")
        proj_frame.pack(fill="x", padx=12, pady=4)

        proj_row = ttk.Frame(proj_frame)
        proj_row.pack(fill="x", **pad)
        self.save_to_project_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(proj_row, text="Save crop offsets to project config",
                        variable=self.save_to_project_var).pack(side="left")

        proj_path_row = ttk.Frame(proj_frame)
        proj_path_row.pack(fill="x", **pad)
        ttk.Label(proj_path_row, text="Project folder:").pack(side="left")
        self.project_var = tk.StringVar()
        ttk.Entry(proj_path_row, textvariable=self.project_var, width=46).pack(
            side="left", padx=4, fill="x", expand=True)
        ttk.Button(proj_path_row, text="Browse…",
                   command=self._browse_project).pack(side="left")

        # ---- Action buttons ----
        btn_row = ttk.Frame(self.root)
        btn_row.pack(pady=8)
        self.run_btn = ttk.Button(btn_row, text="Crop Video(s)",
                                  command=self._run, style="Accent.TButton")
        self.run_btn.pack(side="left", padx=8, ipadx=10, ipady=4)
        ttk.Button(btn_row, text="Clear Log",
                   command=self._clear_log).pack(side="left", padx=4)

        # ---- Progress bar ----
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill="x", padx=12, pady=(0, 4))
        self.progress_bar = ttk.Progressbar(prog_frame, mode="determinate",
                                            maximum=100, value=0)
        self.progress_bar.pack(fill="x")

        # ---- Log ----
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=4)
        self.log = scrolledtext.ScrolledText(log_frame, height=10,
                                             state="disabled", wrap="word",
                                             font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    def _on_mode_change(self):
        pass  # could adjust label text; kept simple

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _on_video_changed(self, *_):
        path = self.video_var.get().strip()
        if os.path.isfile(path):
            threading.Thread(target=self._load_duration, args=(path,), daemon=True).start()
        else:
            self.duration_label.config(text="")

    def _load_duration(self, path):
        try:
            _, fps, dur = _get_video_info(path)
            h = int(dur // 3600)
            m = int((dur % 3600) // 60)
            s = dur % 60
            text = f"Duration: {h:02d}:{m:02d}:{s:05.2f}  ({fps:.2f} fps)"
        except Exception:
            text = ""
        self.root.after(0, self.duration_label.config, {"text": text})

    def _on_trim_toggle(self):
        state = "normal" if self.trim_enabled.get() else "disabled"
        self.start_entry.config(state=state)
        self.end_entry.config(state=state)

    def _browse_video(self):
        if self.mode_var.get() == "batch":
            path = filedialog.askdirectory(title="Select folder containing videos")
        else:
            path = filedialog.askopenfilename(
                title="Select video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                           ("All files", "*.*")])
        if path:
            self.video_var.set(path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.outdir_var.set(path)

    def _browse_project(self):
        path = filedialog.askdirectory(title="Select PixelPaws project folder")
        if path:
            self.project_var.set(path)
            self.save_to_project_var.set(True)

    # ------------------------------------------------------------------
    # Interactive crop selection
    # ------------------------------------------------------------------

    def _preview_and_select(self):
        video = self.video_var.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showwarning("No video", "Please select a video file first.")
            return

        self._log("Opening OpenCV preview — drag a rectangle, then press Enter…")
        try:
            result = select_crop_interactive(video)
        except Exception as exc:
            self._log(f"Error during preview: {exc}")
            messagebox.showerror("Preview Error", str(exc))
            return

        if result is None:
            self._log("Preview cancelled — crop params unchanged.")
            return

        x, y, w, h = result
        self.x_var.set(x)
        self.y_var.set(y)
        self.w_var.set(w)
        self.h_var.set(h)
        self._log(f"Crop selected: x={x}  y={y}  w={w}  h={h}")

    # ------------------------------------------------------------------
    # Batch confirmation dialog
    # ------------------------------------------------------------------

    def _ask_batch_confirm(self, video_path, x, y, w, h, result_holder, event):
        """
        Show a modal dialog for one video in a batch run.
        Called on the main thread via root.after().
        Sets result_holder[0] to 'crop', 'skip', or 'cancel', then signals event.
        """
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Confirm crop — {Path(video_path).name}")
        dlg.grab_set()
        dlg.resizable(True, True)

        # --- Preview frame ---
        preview_img = None
        try:
            import cv2
            from PIL import Image, ImageDraw, ImageTk

            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fh, fw = frame_rgb.shape[:2]

                max_disp_w, max_disp_h = 720, 540
                scale = min(max_disp_w / fw, max_disp_h / fh, 1.0)
                disp_w = int(fw * scale)
                disp_h = int(fh * scale)

                pil_img = Image.fromarray(frame_rgb).resize(
                    (disp_w, disp_h), Image.LANCZOS)
                draw = ImageDraw.Draw(pil_img)

                rx0 = int(x * scale)
                ry0 = int(y * scale)
                rx1 = int((x + w) * scale)
                ry1 = int((y + h) * scale)
                # Green crop rectangle with a dark shadow for visibility
                draw.rectangle([rx0 + 1, ry0 + 1, rx1 + 1, ry1 + 1],
                               outline=(0, 0, 0), width=2)
                draw.rectangle([rx0, ry0, rx1, ry1],
                               outline=(0, 220, 0), width=2)

                preview_img = ImageTk.PhotoImage(pil_img)
        except Exception:
            pass

        if preview_img:
            canvas = tk.Canvas(dlg, width=preview_img.width(),
                               height=preview_img.height(),
                               highlightthickness=0)
            canvas.pack(padx=8, pady=(8, 4))
            canvas.create_image(0, 0, anchor="nw", image=preview_img)
            canvas._img = preview_img  # prevent GC
        else:
            ttk.Label(dlg,
                      text="(Frame preview unavailable — install Pillow and OpenCV)",
                      foreground="gray").pack(padx=16, pady=12)

        ttk.Label(
            dlg,
            text=f"{Path(video_path).name}\n"
                 f"Crop: x={x}  y={y}  w={w}  h={h}",
            justify="center",
        ).pack(pady=4)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(pady=(4, 10))

        def _done(choice):
            result_holder[0] = choice
            dlg.destroy()
            event.set()

        ttk.Button(btn_row, text="Crop this video",
                   command=lambda: _done("crop")).pack(
            side="left", padx=8, ipadx=8, ipady=3)
        ttk.Button(btn_row, text="Skip",
                   command=lambda: _done("skip")).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Cancel all",
                   command=lambda: _done("cancel")).pack(side="left", padx=8)

        dlg.protocol("WM_DELETE_WINDOW", lambda: _done("cancel"))
        dlg.wait_visibility()
        dlg.lift()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        video = self.video_var.get().strip()
        if not video:
            messagebox.showwarning("Missing input", "Please select a video or folder.")
            return

        try:
            x = int(self.x_var.get())
            y = int(self.y_var.get())
            w = int(self.w_var.get())
            h = int(self.h_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid params", "X, Y, W, H must be integers.")
            return

        if w <= 0 or h <= 0:
            messagebox.showwarning("Invalid params", "W and H must be > 0.")
            return

        outdir = self.outdir_var.get().strip() or None
        project = self.project_var.get().strip() if self.save_to_project_var.get() else None
        crf = int(self.crf_var.get())

        start_time = end_time = None
        if self.trim_enabled.get():
            try:
                start_time = parse_time(self.start_var.get())
                end_time = parse_time(self.end_var.get())
            except ValueError as e:
                messagebox.showwarning("Invalid time", str(e))
                return

        self.progress_bar["value"] = 0
        self.run_btn.config(state="disabled")
        t = threading.Thread(target=self._run_thread,
                             args=(video, x, y, w, h, outdir, project, crf, start_time, end_time),
                             daemon=True)
        t.start()

    def _run_thread(self, video, x, y, w, h, outdir, project, crf, start_time, end_time):
        def log(msg):
            self.root.after(0, self._log, msg)
        def progress(pct):
            self.root.after(0, self._set_progress, pct)

        try:
            if self.mode_var.get() == "batch":
                if not os.path.isdir(video):
                    log(f"Error: not a directory: {video}")
                    return

                video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
                videos = sorted(f for f in Path(video).iterdir()
                                if f.is_file() and f.suffix.lower() in video_exts)
                if not videos:
                    log(f"No video files found in: {video}")
                    return

                n = len(videos)
                log(f"Found {n} video(s). Confirm each crop before processing.")

                for i, vp in enumerate(videos):
                    result_holder = [None]
                    ev = threading.Event()
                    self.root.after(
                        0, self._ask_batch_confirm,
                        str(vp), x, y, w, h, result_holder, ev,
                    )
                    ev.wait()

                    decision = result_holder[0]
                    if decision == "cancel":
                        log("Batch cancelled.")
                        break
                    elif decision == "skip":
                        log(f"Skipped: {vp.name}")
                        continue

                    def _prog(p, i=i, n=n):
                        progress((i + p) / n)

                    process_single(str(vp), x, y, w, h, outdir, project,
                                   crf=crf, start_time=start_time, end_time=end_time,
                                   log_fn=log, progress_fn=_prog)
            else:
                if not os.path.isfile(video):
                    log(f"Error: file not found: {video}")
                    return
                process_single(video, x, y, w, h, outdir, project,
                               crf=crf, start_time=start_time, end_time=end_time,
                               log_fn=log, progress_fn=progress)
            log("All done.")
        except Exception as exc:
            log(f"ERROR: {exc}")
        finally:
            self.root.after(0, self._set_progress, 1.0)
            self.root.after(0, self.run_btn.config, {"state": "normal"})

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _set_progress(self, pct: float):
        self.progress_bar["value"] = pct * 100

    def _log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crop videos spatially before running DeepLabCut.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?", help="Input video file (optional in GUI mode)")
    parser.add_argument("--batch", metavar="FOLDER",
                        help="Process all videos in FOLDER with the same crop")
    parser.add_argument("--x", type=int, default=None, help="Crop left edge (pixels)")
    parser.add_argument("--y", type=int, default=None, help="Crop top edge (pixels)")
    parser.add_argument("--w", type=int, default=None, help="Crop width (pixels)")
    parser.add_argument("--h", type=int, default=None, help="Crop height (pixels)")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Directory for cropped output files (default: same as input)")
    parser.add_argument("--project", metavar="FOLDER",
                        help="PixelPaws project folder — writes crop_offset_x/y to config")
    parser.add_argument("--start", metavar="TIME",
                        help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", metavar="TIME",
                        help="End time (HH:MM:SS or seconds)")
    args = parser.parse_args()

    try:
        start_time = parse_time(args.start) if args.start else None
        end_time = parse_time(args.end) if args.end else None
    except ValueError as e:
        parser.error(str(e))

    if args.batch:
        if None in (args.x, args.y, args.w, args.h):
            parser.error("--batch requires --x --y --w --h")
        process_batch(args.batch, args.x, args.y, args.w, args.h,
                      args.output_dir, args.project,
                      start_time=start_time, end_time=end_time)
        return

    if args.video:
        if None in (args.x, args.y, args.w, args.h):
            print("No crop params given — opening interactive preview…")
            result = select_crop_interactive(args.video)
            if result is None:
                print("Cancelled.")
                return
            x, y, w, h = result
        else:
            x, y, w, h = args.x, args.y, args.w, args.h
        process_single(args.video, x, y, w, h, args.output_dir, args.project,
                       start_time=start_time, end_time=end_time)
        return

    # ------------------------------------------------------------------
    # GUI mode (no CLI args given, or only --project)
    # ------------------------------------------------------------------
    root = tk.Tk()
    app = CropForDLCApp(root, initial_project=args.project or "")
    root.mainloop()


if __name__ == "__main__":
    main()
