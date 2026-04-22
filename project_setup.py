"""
project_setup.py — PixelPaws Project Setup Wizard
==================================================
3-step modal wizard that runs on startup to select or create a project folder.

Steps
-----
1. Choose Project  — New or Open Existing
2. Configure       — video extension, behaviors, body parts, ROI size
3. Extract Features — run feature extraction on videos/ subfolder; skip option

On completion the wizard calls::
    app.current_project_folder.set(project_folder)
    app.root.deiconify()

If the user closes the wizard without finishing::
    app.root.destroy()
"""

import os
import json
import pathlib
import glob as _glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

_BP_DEFAULTS = ['hrpaw', 'hlpaw', 'snout']

try:
    from project_config import ProjectConfig
    _PROJECT_CONFIG_AVAILABLE = True
except ImportError:
    ProjectConfig = None
    _PROJECT_CONFIG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Recent projects persistence
# ---------------------------------------------------------------------------
_RECENT_FILE = pathlib.Path.home() / '.pixelpaws_recent.json'
_MAX_RECENT  = 8

def _load_recent() -> list:
    """Return list of valid recent project folder paths."""
    try:
        paths = json.loads(_RECENT_FILE.read_text())
        return [p for p in paths if os.path.isdir(p)]
    except Exception:
        return []

def _save_recent(folder: str):
    """Prepend folder to recent list and persist."""
    recent = [folder] + [p for p in _load_recent() if p != folder]
    try:
        _RECENT_FILE.write_text(json.dumps(recent[:_MAX_RECENT], indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ProjectSetupWizard
# ---------------------------------------------------------------------------
class ProjectSetupWizard(tk.Toplevel):
    """Modal 3-step wizard shown on application startup."""

    def __init__(self, root, app):
        super().__init__(root)
        self.root = root
        self.app  = app

        self.project_folder = ''

        self.title("PixelPaws — Project Setup")
        self.resizable(True, True)
        self.minsize(750, 660)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Centre on screen — taller when recent projects exist
        self.update_idletasks()
        w = 820
        h = 850 if _load_recent() else 750
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # Make modal — deiconify needed because root is withdrawn on startup
        self.transient(root)
        self.deiconify()
        self.grab_set()

        self._build_shell()
        self._show_step1()

    # ------------------------------------------------------------------ Shell

    def _build_shell(self):
        """Build the persistent header + content area + nav bar."""
        # Header — use theme primary color if ttkbootstrap available
        try:
            _hdr_bg = self.app.style.colors.primary
        except Exception:
            _hdr_bg = '#2c5f8a'
        hdr = tk.Frame(self, bg=_hdr_bg, pady=12)
        hdr.pack(fill='x')
        tk.Label(hdr, text="🐾 PixelPaws — Project Setup",
                 bg=_hdr_bg, fg='white',
                 font=('Arial', 14, 'bold')).pack()
        self.step_label = tk.Label(hdr, text="",
                                   bg=_hdr_bg, fg='#c8e0f4',
                                   font=('Arial', 9))
        self.step_label.pack()

        # Content frame (swapped per step)
        self.content = ttk.Frame(self, padding=15)
        self.content.pack(fill='both', expand=True)

        # Nav buttons
        nav = ttk.Frame(self, padding=(15, 8))
        nav.pack(fill='x', side='bottom')
        self.btn_back = ttk.Button(nav, text="◀ Back",  command=self._back,   width=10)
        self.btn_next = ttk.Button(nav, text="Next ▶",  command=self._next,   width=10)
        self.btn_back.pack(side='left',  padx=4)
        self.btn_next.pack(side='right', padx=4)

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ------------------------------------------------------------------ Step 1

    def _show_step1(self):
        self._current_step = 1
        self.project_folder = ''  # reset stale folder when navigating back
        self._clear_content()
        self.step_label.config(text="Step 1 of 3 — Choose Project")
        self.btn_back.config(state='disabled')
        self.btn_next.config(state='disabled')

        recent = _load_recent()

        if recent:
            ttk.Label(self.content,
                      text="Welcome to PixelPaws!",
                      font=('Arial', 11, 'bold'), justify='center').pack(pady=(10, 6))

            ttk.Label(self.content, text="Recent Projects:",
                      font=('Arial', 10)).pack(anchor='w', pady=(4, 2))

            list_frame = ttk.Frame(self.content)
            list_frame.pack(fill='both', expand=True)

            self._recent_listbox = tk.Listbox(
                list_frame, height=min(len(recent), 6),
                selectmode='single', exportselection=False,
                activestyle='dotbox', font=('Arial', 9))
            sb = ttk.Scrollbar(list_frame, command=self._recent_listbox.yview)
            self._recent_listbox.config(yscrollcommand=sb.set)
            self._recent_listbox.pack(side='left', fill='both', expand=True)
            sb.pack(side='right', fill='y')

            self._recent_paths = recent
            for p in recent:
                name = os.path.basename(p)
                self._recent_listbox.insert(tk.END, f"{name}   ({p})")
            self._recent_listbox.selection_set(0)
            self._recent_listbox.bind('<Double-1>', lambda _e: self._open_recent())

            ttk.Button(self.content, text="Open Selected",
                       command=self._open_recent, width=20).pack(pady=(6, 4))

            ttk.Separator(self.content, orient='horizontal').pack(fill='x', pady=8)

            btn_frame = ttk.Frame(self.content)
            btn_frame.pack()
            ttk.Button(btn_frame, text="🆕  New Project",
                       width=22, command=self._new_project).grid(
                           row=0, column=0, padx=8, pady=4)
            ttk.Button(btn_frame, text="📂  Open Other…",
                       width=22, command=self._open_project).grid(
                           row=0, column=1, padx=8, pady=4)
        else:
            # No recent projects — original two-button layout
            ttk.Label(self.content,
                      text="Welcome to PixelPaws!\n\nCreate a new project or open an existing one.",
                      font=('Arial', 10), justify='center').pack(pady=(10, 20))

            btn_frame = ttk.Frame(self.content)
            btn_frame.pack(expand=True)
            ttk.Button(btn_frame, text="🆕  New Project",
                       width=22, command=self._new_project).grid(
                           row=0, column=0, padx=10, pady=8)
            ttk.Button(btn_frame, text="📂  Open Existing",
                       width=22, command=self._open_project).grid(
                           row=1, column=0, padx=10, pady=8)

        self.step1_status = ttk.Label(self.content, text="", foreground='gray')
        self.step1_status.pack(pady=8)

    def _open_recent(self):
        """Open a project selected from the recent list."""
        sel = self._recent_listbox.curselection()
        if not sel:
            return
        folder = self._recent_paths[sel[0]]
        if not os.path.isdir(folder):
            messagebox.showwarning("Not Found",
                                   f"Folder no longer exists:\n{folder}")
            return
        self._open_folder(folder)

    def _new_project(self):
        folder = filedialog.askdirectory(title="Select or Create Project Folder")
        if not folder:
            return
        self.project_folder = os.path.abspath(folder)
        for sub in ('videos', 'behavior_labels'):
            os.makedirs(os.path.join(self.project_folder, sub), exist_ok=True)
        self.step1_status.config(
            text=f"Project: {os.path.basename(self.project_folder)}",
            foreground='green')
        self._show_step2()

    def _open_project(self):
        """Browse for a project folder (or PixelPaws_project.json file)."""
        folder = filedialog.askdirectory(title="Select Existing Project Folder")
        if not folder:
            return
        self._open_folder(folder)

    def _open_folder(self, folder: str):
        """Open an existing project folder; save to recent list."""
        folder = os.path.abspath(folder)
        _save_recent(folder)
        config_path = os.path.join(folder, 'PixelPaws_project.json')
        if os.path.isfile(config_path):
            # Complete project — load and finish immediately
            self.project_folder = folder
            self.app.current_project_folder.set(folder)
            self.app._load_project_config(config_path, silent=True)
            self.grab_release()
            self.destroy()
            self.root.deiconify()
        else:
            # No config yet — treat as new
            self.project_folder = folder
            for sub in ('videos', 'behavior_labels'):
                os.makedirs(os.path.join(self.project_folder, sub), exist_ok=True)
            self._show_step2()

    # ------------------------------------------------------------------ Step 2

    def _show_step2(self):
        self._current_step = 2
        self._clear_content()
        self.step_label.config(text="Step 2 of 3 — Configure")
        self.btn_back.config(state='normal')
        self.btn_next.config(state='normal', text="Next ▶", command=self._next)

        # Load any partial existing config
        existing = {}
        config_path = os.path.join(self.project_folder, 'PixelPaws_project.json')
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        cf = ttk.Frame(self.content)
        cf.pack(fill='both', expand=True)
        cf.columnconfigure(1, weight=1)

        row = 0

        # ---- Video extension ----
        ttk.Label(cf, text="Video Extension:").grid(
            row=row, column=0, sticky='w', pady=4)
        self._video_ext = tk.StringVar(value=existing.get('video_ext', '.mp4'))
        ext_cb = ttk.Combobox(cf, textvariable=self._video_ext,
                               values=['.mp4', '.avi', '.mov', '.wmv'], width=8,
                               state='readonly')
        ext_cb.grid(row=row, column=1, sticky='w', padx=5, pady=4)
        row += 1

        # ---- Behaviors (multi-item list) ----
        ttk.Label(cf, text="Behaviors:").grid(
            row=row, column=0, sticky='nw', pady=(8, 2))

        beh_outer = ttk.Frame(cf)
        beh_outer.grid(row=row, column=1, columnspan=2, sticky='ew',
                       padx=5, pady=(8, 2))
        beh_outer.columnconfigure(0, weight=1)

        # Entry + Add button
        add_row = ttk.Frame(beh_outer)
        add_row.grid(row=0, column=0, sticky='ew')
        add_row.columnconfigure(0, weight=1)
        self._behavior_entry = ttk.Entry(add_row)
        self._behavior_entry.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self._behavior_entry.bind('<Return>', lambda _: self._add_behavior())
        ttk.Button(add_row, text="Add", width=6,
                   command=self._add_behavior).grid(row=0, column=1)

        # Listbox + Remove button
        lb_row = ttk.Frame(beh_outer)
        lb_row.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        lb_row.columnconfigure(0, weight=1)
        self._behaviors_listbox = tk.Listbox(lb_row, height=4,
                                              selectmode='single',
                                              exportselection=False)
        self._behaviors_listbox.grid(row=0, column=0, sticky='ew')
        ttk.Button(lb_row, text="Remove", width=6,
                   command=self._remove_behavior).grid(row=0, column=1,
                                                       padx=(4, 0), sticky='n')

        # Pre-populate (handle old single-string format for backward compat)
        behaviors = existing.get('behaviors') or []
        if not behaviors and existing.get('behavior_name'):
            behaviors = [existing['behavior_name']]
        for b in behaviors:
            self._behaviors_listbox.insert(tk.END, b)

        row += 1

        # ---- Brightness Body Parts ----
        ttk.Label(cf, text="Brightness Body Parts:").grid(
            row=row, column=0, sticky='nw', pady=(10, 2))

        bp_outer = ttk.Frame(cf)
        bp_outer.grid(row=row, column=1, sticky='ew', padx=5, pady=(10, 2))

        bp_list_frame = ttk.Frame(bp_outer)
        bp_list_frame.pack(fill='x')
        self._bp_listbox = tk.Listbox(bp_list_frame, selectmode='multiple',
                                       height=5, exportselection=False)
        bp_scroll = ttk.Scrollbar(bp_list_frame, command=self._bp_listbox.yview)
        self._bp_listbox.config(yscrollcommand=bp_scroll.set)
        self._bp_listbox.pack(side='left', fill='both', expand=True)
        bp_scroll.pack(side='right', fill='y')

        _refresh_btn = ttk.Button(cf, text="🔄", width=3,
                   command=self._parse_bp_from_h5)
        _refresh_btn.grid(row=row, column=2, padx=2, sticky='n', pady=(10, 0))
        try:
            from ui_utils import ToolTip
            ToolTip(_refresh_btn, "Auto-detect body parts from DLC .h5 files")
        except ImportError:
            pass

        # Performance disclaimer
        ttk.Label(bp_outer,
                  text="ℹ  Brightness extraction is the slowest step. "
                       "3–5 body parts is recommended — each additional part "
                       "adds significant processing time.",
                  foreground='gray', wraplength=340, justify='left',
                  font=('Arial', 8)).pack(anchor='w', pady=(4, 0))

        row += 1

        # ---- ROI size ----
        ttk.Label(cf, text="Square ROI Size (px):").grid(
            row=row, column=0, sticky='w', pady=4)
        self._roi_size = tk.StringVar(value=str(existing.get('roi_size', 20)))
        ttk.Entry(cf, textvariable=self._roi_size, width=8).grid(
            row=row, column=1, sticky='w', padx=5, pady=4)
        row += 1

        # ---- Optical Flow ----
        self._include_optflow = tk.BooleanVar(
            value=existing.get('include_optical_flow', True))
        ttk.Checkbutton(cf, text="Include Optical Flow Features (slower)",
                        variable=self._include_optflow).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(10, 2))
        row += 1

        ttk.Label(cf, text="Optical Flow Body Parts:").grid(
            row=row, column=0, sticky='w', pady=2)
        saved_optflow = existing.get('bp_optflow_list', ['hrpaw', 'hlpaw', 'snout'])
        if isinstance(saved_optflow, list):
            saved_optflow = ','.join(saved_optflow)
        self._bp_optflow = tk.StringVar(value=saved_optflow)
        ttk.Entry(cf, textvariable=self._bp_optflow, width=30).grid(
            row=row, column=1, sticky='w', padx=5, pady=2)
        row += 1

        # ---- Populate body parts: saved config → defaults ----
        saved_bp = existing.get('bp_pixbrt_list', [])
        if isinstance(saved_bp, str):
            saved_bp = [b.strip() for b in saved_bp.split(',') if b.strip()]
        parts_to_show = saved_bp if saved_bp else _BP_DEFAULTS
        for bp in parts_to_show:
            self._bp_listbox.insert(tk.END, bp)
            self._bp_listbox.selection_set(tk.END)

    # ---- behavior helpers ------------------------------------------------

    def _add_behavior(self):
        name = self._behavior_entry.get().strip()
        if not name:
            return
        existing = [self._behaviors_listbox.get(i)
                    for i in range(self._behaviors_listbox.size())]
        if name not in existing:
            self._behaviors_listbox.insert(tk.END, name)
        self._behavior_entry.delete(0, tk.END)

    def _remove_behavior(self):
        sel = self._behaviors_listbox.curselection()
        if sel:
            self._behaviors_listbox.delete(sel[0])

    def _get_behaviors(self):
        return [self._behaviors_listbox.get(i)
                for i in range(self._behaviors_listbox.size())]

    # ---- body part helpers -----------------------------------------------

    def _parse_bp_from_h5(self, silent=False):
        """Populate body-part listbox from any H5 file found in videos/."""
        videos_dir = os.path.join(self.project_folder, 'videos')
        h5_files = _glob.glob(os.path.join(videos_dir, '*.h5'))
        if not h5_files:
            if not silent:
                messagebox.showinfo("No H5 Files",
                                    "No DLC .h5 files found in videos/.\n"
                                    "Add your DLC output files and try again.")
            return
        try:
            import pandas as pd
            df = pd.read_hdf(h5_files[0])
            # MultiIndex columns: (scorer, bodypart, coord) — level 1 is body part
            bodyparts = sorted({col[1] for col in df.columns})
            if not bodyparts:
                raise ValueError("No body parts found in H5 file")
            self._bp_listbox.delete(0, tk.END)
            for bp in bodyparts:
                self._bp_listbox.insert(tk.END, str(bp))
            # Pre-select defaults if present; select all if none match
            defaults_selected = False
            for i, bp in enumerate(bodyparts):
                if bp in _BP_DEFAULTS:
                    self._bp_listbox.selection_set(i)
                    defaults_selected = True
            if not defaults_selected:
                self._bp_listbox.selection_set(0, tk.END)
        except Exception as e:
            if not silent:
                messagebox.showwarning("Parse Error",
                                       f"Could not read body parts from H5:\n{e}")

    def _get_selected_bodyparts(self):
        idxs = self._bp_listbox.curselection()
        return [self._bp_listbox.get(i) for i in idxs]

    def _save_step2_config(self):
        """Persist Step 2 fields to PixelPaws_project.json."""
        try:
            roi = int(self._roi_size.get())
        except ValueError:
            roi = 20

        if _PROJECT_CONFIG_AVAILABLE:
            cfg = ProjectConfig.load(self.project_folder)
            cfg.video_ext = self._video_ext.get()
            cfg.behaviors = self._get_behaviors()
            cfg.bp_pixbrt_list = self._get_selected_bodyparts()
            cfg.roi_size = roi
            cfg.include_optical_flow = self._include_optflow.get()
            cfg.bp_optflow_list = [x.strip() for x in self._bp_optflow.get().split(',') if x.strip()]
            cfg.save(self.project_folder)
            return

        # Inline fallback
        config_path = os.path.join(self.project_folder, 'PixelPaws_project.json')
        existing = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update({
            'project_folder': self.project_folder,
            'video_ext':      self._video_ext.get(),
            'behaviors':      self._get_behaviors(),
            'bp_pixbrt_list': self._get_selected_bodyparts(),
            'roi_size':       roi,
            'include_optical_flow': self._include_optflow.get(),
            'bp_optflow_list': [x.strip() for x in self._bp_optflow.get().split(',') if x.strip()],
        })
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error",
                                 f"Could not save project config:\n{e}")

    # ------------------------------------------------------------------ Step 3

    def _show_step3(self):
        self._current_step = 3
        self._clear_content()
        self.step_label.config(text="Step 3 of 3 — Extract Features")
        self.btn_back.config(state='normal')
        self.btn_next.config(state='disabled', text="✅ Finish", command=self._finish)

        videos_dir = os.path.join(self.project_folder, 'videos')
        ext = getattr(self, '_video_ext', tk.StringVar(value='.mp4')).get()

        # Find paired video + h5 files (deduplicate across case variants)
        self._video_pairs = []
        if os.path.isdir(videos_dir):
            _seen = {}
            for vf in (_glob.glob(os.path.join(videos_dir, f'*{ext}')) +
                       _glob.glob(os.path.join(videos_dir, f'*{ext.upper()}'))):
                _seen[os.path.normcase(vf)] = vf
            for vf in sorted(_seen.values()):
                base = os.path.splitext(os.path.basename(vf))[0]
                h5_matches = _glob.glob(os.path.join(videos_dir, f'{base}DLC*.h5'))
                if h5_matches:
                    self._video_pairs.append((vf, h5_matches[0]))

        # Info label
        if self._video_pairs:
            info = (f"Found {len(self._video_pairs)} video(s) with DLC files in videos/.\n"
                    "Click 'Extract Features' to process them, or skip for now.")
        else:
            info = ("No paired video+DLC files found in videos/ yet.\n"
                    "You can skip this step and extract features later.")
        ttk.Label(self.content, text=info, wraplength=540, justify='left').pack(
            anchor='w', pady=(0, 8))

        # Video list
        if self._video_pairs:
            list_frame = ttk.LabelFrame(self.content, text="Videos to Process", padding=5)
            list_frame.pack(fill='x', pady=4)
            lb = tk.Listbox(list_frame, height=min(len(self._video_pairs), 4))
            lb.pack(fill='x')
            for vf, _ in self._video_pairs:
                lb.insert(tk.END, os.path.basename(vf))

        # Progress log
        log_frame = ttk.LabelFrame(self.content, text="Extraction Log", padding=5)
        log_frame.pack(fill='both', expand=True, pady=4)
        self._extract_log = scrolledtext.ScrolledText(log_frame, height=7, wrap=tk.WORD,
                                                       state='disabled')
        self._extract_log.pack(fill='both', expand=True)

        # Buttons
        btn_row = ttk.Frame(self.content)
        btn_row.pack(fill='x', pady=(6, 0))

        self._extract_btn = ttk.Button(
            btn_row, text="▶ Extract Features",
            command=self._run_extraction,
            state='normal' if self._video_pairs else 'disabled')
        self._extract_btn.pack(side='left', padx=4)

        ttk.Button(btn_row, text="📋 Key File",
                   command=self._open_key_file_dialog).pack(side='left', padx=4)

        ttk.Button(btn_row, text="⏭ Skip — I'll do this later",
                   command=self._skip_extraction).pack(side='left', padx=4)

    def _log_extract(self, msg):
        self._extract_log.config(state='normal')
        self._extract_log.insert(tk.END, msg + '\n')
        self._extract_log.see(tk.END)
        self._extract_log.config(state='disabled')

    def _run_extraction(self):
        # Use cached values from Step 2 (widgets are destroyed by _show_step3)
        bp_list = self._saved_bp_list
        roi = self._saved_roi

        self._extract_btn.config(state='disabled')
        self.btn_back.config(state='disabled')

        def worker():
            try:
                from PixelPaws_GUI import (PixelPaws_ExtractFeatures,
                                           _atomic_pickle_save,
                                           PixelPawsGUI)
            except ImportError as imp_err:
                self.after(0, lambda: self._log_extract(
                    f"ERROR: Could not import extraction functions — {imp_err}"))
                self.after(0, lambda: self.btn_next.config(state='normal'))
                return

            # Read optical flow settings from Step 2 widgets
            include_optflow = getattr(self, '_include_optflow',
                                      tk.BooleanVar(value=False)).get()
            bp_optflow_list = ([x.strip() for x in getattr(self, '_bp_optflow',
                tk.StringVar(value='')).get().split(',') if x.strip()]
                if include_optflow else [])

            # Build config dict matching the main GUI's cache scheme
            cfg = {
                'bp_include_list': None,  # all body parts for pose
                'bp_pixbrt_list': bp_list,
                'square_size': [roi],
                'pix_threshold': 0.3,
                'include_optical_flow': include_optflow,
                'bp_optflow_list': bp_optflow_list,
            }
            cfg_hash = PixelPawsGUI._feature_hash_key(cfg)
            cache_dir = os.path.join(self.project_folder, 'features')
            os.makedirs(cache_dir, exist_ok=True)

            for vf, h5f in self._video_pairs:
                name = os.path.basename(vf)
                session_name = os.path.splitext(name)[0]
                cache_file = os.path.join(
                    cache_dir, f"{session_name}_features_{cfg_hash}.pkl")

                if os.path.isfile(cache_file):
                    self.after(0, lambda n=name: self._log_extract(
                        f"Skipping (cached): {n}"))
                    continue

                self.after(0, lambda n=name: self._log_extract(f"Processing: {n}"))
                try:
                    X = PixelPaws_ExtractFeatures(
                        pose_data_file=h5f,
                        video_file_path=vf,
                        bp_pixbrt_list=bp_list,
                        square_size=roi,
                        pix_threshold=0.3,
                        bp_include_list=None,
                        include_optical_flow=include_optflow,
                        bp_optflow_list=bp_optflow_list,
                    )
                    X = X.reset_index(drop=True)
                    nan_mask = X.isna().any(axis=1)
                    if nan_mask.any():
                        self.after(0, lambda n=name, cnt=int(nan_mask.sum()):
                                   self._log_extract(
                                       f"  Dropped {cnt} NaN rows"))
                        X = X[~nan_mask].reset_index(drop=True)
                    _atomic_pickle_save(X, cache_file)
                    self.after(0, lambda n=name: self._log_extract(
                        f"  ✓ Done: {n}"))
                except Exception as e:
                    self.after(0, lambda n=name, err=str(e):
                               self._log_extract(f"  ✗ Error ({n}): {err}"))

            self.after(0, lambda: self._log_extract("\n✅ Feature extraction complete."))
            self.after(0, lambda: self.btn_next.config(state='normal'))
            self.after(0, lambda: self.btn_back.config(state='normal'))

        threading.Thread(target=worker, daemon=True).start()

    def _open_key_file_dialog(self):
        """Open KeyFileGeneratorDialog for videos in this project."""
        videos_dir = os.path.join(self.project_folder, 'videos')
        # Collect all video basenames (already discovered in _video_pairs, plus any without h5)
        ext = getattr(self, '_video_ext', tk.StringVar(value='.mp4')).get()
        _seen = {}
        if os.path.isdir(videos_dir):
            for vf in (_glob.glob(os.path.join(videos_dir, f'*{ext}')) +
                       _glob.glob(os.path.join(videos_dir, f'*{ext.upper()}'))):
                _seen[os.path.normcase(vf)] = vf
        basenames = [os.path.splitext(os.path.basename(v))[0]
                     for v in sorted(_seen.values())]
        if not basenames:
            messagebox.showinfo("No Videos",
                                "No video files found in videos/.\n"
                                "Add your videos first.", parent=self)
            return
        # Load existing key file if present
        existing = {}
        key_path = os.path.join(self.project_folder, 'key_file.csv')
        if os.path.isfile(key_path):
            try:
                import csv
                with open(key_path, newline='') as f:
                    for row in csv.DictReader(f):
                        existing[row.get('Subject', '')] = row.get('Treatment', '')
            except Exception as _key_err:
                print(f"Warning: could not load existing key file: {_key_err}")
        KeyFileGeneratorDialog(self, self.project_folder, basenames, existing)

    def _skip_extraction(self):
        if not messagebox.askyesno(
                "Skip Feature Extraction?",
                "You can extract features later from the Tools tab.\n\n"
                "Skip for now?", parent=self):
            return
        self._log_extract("(Skipped — you can run feature extraction later.)")
        self.btn_next.config(state='normal')

    # ------------------------------------------------------------------ Nav

    def _back(self):
        if self._current_step == 2:
            self._show_step1()
        elif self._current_step == 3:
            self._show_step2()

    def _next(self):
        if self._current_step == 1:
            pass  # handled by the New / Open buttons
        elif self._current_step == 2:
            self._saved_bp_list = self._get_selected_bodyparts()
            try:
                self._saved_roi = int(self._roi_size.get())
            except (ValueError, tk.TclError):
                self._saved_roi = 20
            self._save_step2_config()
            self._show_step3()
        elif self._current_step == 3:
            self._finish()

    def _finish(self):
        # Config was already saved when transitioning from step 2 → 3
        _save_recent(self.project_folder)
        self.app.current_project_folder.set(self.project_folder)
        config_path = os.path.join(self.project_folder, 'PixelPaws_project.json')
        if os.path.isfile(config_path):
            self.app._load_project_config(config_path, silent=True)
        self.grab_release()
        self.destroy()
        self.root.deiconify()

    def _on_close(self):
        """User closed the wizard — exit at startup, return to main window mid-session."""
        self.grab_release()
        self.destroy()
        if self.root.winfo_viewable():
            self.root.deiconify()   # mid-session: just go back to main window
        else:
            self.root.destroy()     # startup: exit the app


# ---------------------------------------------------------------------------
# KeyFileGeneratorDialog
# ---------------------------------------------------------------------------
class KeyFileGeneratorDialog(tk.Toplevel):
    """
    Modal dialog for assigning group labels (e.g. Control / Treatment) to
    each video in a project.  Saves to <project>/key_file.csv and records
    the path in PixelPaws_project.json.
    """

    def __init__(self, parent, project_folder: str, video_basenames: list,
                 existing_groups: dict = None, on_save=None):
        super().__init__(parent)
        self.title("PixelPaws — Key File")
        self.project_folder = project_folder
        self.video_basenames = list(video_basenames)
        self.on_save = on_save
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        existing_groups = existing_groups or {}

        w = 540
        h = min(80 + 36 * len(video_basenames) + 120, 620)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # Header — use theme primary color if ttkbootstrap available
        try:
            _hdr_bg = self.master.nametowidget('.').style.colors.primary
        except Exception:
            _hdr_bg = '#2c5f8a'
        hdr = tk.Frame(self, bg=_hdr_bg, pady=10)
        hdr.pack(fill='x')
        tk.Label(hdr, text="📋 Key File — Group Assignment",
                 bg=_hdr_bg, fg='white', font=('Arial', 12, 'bold')).pack()
        tk.Label(hdr,
                 text="Assign each video to a group (e.g. Control, Treatment, Sham).",
                 bg=_hdr_bg, fg='#c8e0f4', font=('Arial', 9)).pack()

        # Quick-fill bar
        qf = ttk.Frame(self, padding=(10, 6, 10, 2))
        qf.pack(fill='x')
        ttk.Label(qf, text="Set all to:").pack(side='left')
        self._qf_var = tk.StringVar()
        ttk.Entry(qf, textvariable=self._qf_var, width=14).pack(side='left', padx=4)
        ttk.Button(qf, text="Apply", command=self._apply_quick_fill).pack(side='left')

        # Scrollable table
        cf = ttk.Frame(self, padding=6)
        cf.pack(fill='both', expand=True)
        canvas = tk.Canvas(cf, highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        self._table = ttk.Frame(canvas)
        _cwin = canvas.create_window((0, 0), window=self._table, anchor='nw')

        ttk.Label(self._table, text="Subject (video base name)", font=('Arial', 9, 'bold'),
                  width=38).grid(row=0, column=0, sticky='w', padx=4, pady=2)
        ttk.Label(self._table, text="Treatment / Group", font=('Arial', 9, 'bold'),
                  width=16).grid(row=0, column=1, sticky='w', padx=4, pady=2)
        ttk.Separator(self._table, orient='horizontal').grid(
            row=1, column=0, columnspan=2, sticky='ew', pady=2)

        self._group_vars = {}
        for i, base in enumerate(self.video_basenames):
            ttk.Label(self._table, text=base, anchor='w').grid(
                row=i + 2, column=0, sticky='w', padx=4, pady=2)
            var = tk.StringVar(value=existing_groups.get(base, ''))
            ttk.Entry(self._table, textvariable=var, width=20).grid(
                row=i + 2, column=1, padx=4, pady=2, sticky='w')
            self._group_vars[base] = var

        self._table.bind('<Configure>',
                         lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda _e: canvas.itemconfig(_cwin, width=_e.width))

        # Buttons
        bf = ttk.Frame(self, padding=(10, 6))
        bf.pack(fill='x', side='bottom')
        ttk.Button(bf, text="💾 Save", command=self._save).pack(side='right', padx=4)
        ttk.Button(bf, text="Cancel", command=self.destroy).pack(side='right', padx=4)

    def _apply_quick_fill(self):
        val = self._qf_var.get().strip()
        if val:
            for var in self._group_vars.values():
                var.set(val)

    def _save(self):
        import csv
        key_path = os.path.join(self.project_folder, 'key_file.csv')
        rows = [(base, self._group_vars[base].get().strip())
                for base in self.video_basenames]
        try:
            with open(key_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Treatment'])
                writer.writerows(rows)
            # Record path in project JSON
            config_path = os.path.join(self.project_folder, 'PixelPaws_project.json')
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    cfg['key_file'] = 'key_file.csv'
                    with open(config_path, 'w') as f:
                        json.dump(cfg, f, indent=2)
                except Exception:
                    pass
            if self.on_save:
                self.on_save({base: grp for base, grp in rows})
            messagebox.showinfo("Saved",
                                f"Key file saved to:\n{key_path}", parent=self)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Save Error",
                                 f"Could not save key file:\n{e}", parent=self)
