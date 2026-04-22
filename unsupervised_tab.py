"""
unsupervised_tab.py — PixelPaws Unsupervised Behavior Discovery Tab
====================================================================
Pools per-frame pose features across sessions, reduces dimensionality
with UMAP, and clusters with HDBSCAN to find discrete behavioral states.

Outputs are compatible with the existing Render / Analysis / Evaluation
pipeline (one binary column per cluster, same format as prediction CSVs).

Dependencies (pip install umap-learn hdbscan):
    umap-learn  — UMAP dimensionality reduction
    hdbscan     — HDBSCAN clustering
    scikit-learn — StandardScaler (usually present; ships with umap-learn)
"""

import os
import glob
import json
import pickle
import hashlib
import threading
import traceback
from typing import Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import tkinter.simpledialog as sd
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional: matplotlib for embedded plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


from ui_utils import (ToolTip as _ToolTip, _bind_tight_layout_on_resize)


# ---------------------------------------------------------------------------
# Optional: UMAP + HDBSCAN + StandardScaler
# ---------------------------------------------------------------------------
try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    UMAP_HDBSCAN_AVAILABLE = True
except ImportError:
    UMAP_HDBSCAN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pose feature version (for cache hash replication)
# ---------------------------------------------------------------------------
try:
    from pose_features import POSE_FEATURE_VERSION
except ImportError:
    POSE_FEATURE_VERSION = 5

# ---------------------------------------------------------------------------
# Centralised feature cache (shared hash + search logic)
# ---------------------------------------------------------------------------
try:
    from feature_cache import FeatureCacheManager
    _FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FeatureCacheManager = None
    _FEATURE_CACHE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Session discovery (shared with training / evaluation tabs)
# ---------------------------------------------------------------------------
try:
    from evaluation_tab import find_session_triplets
    _FIND_SESSIONS_AVAILABLE = True
except ImportError:
    find_session_triplets = None
    _FIND_SESSIONS_AVAILABLE = False


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _feature_hash_key(cfg: dict) -> str:
    """Compute feature cache hash — delegates to FeatureCacheManager when available."""
    if _FEATURE_CACHE_AVAILABLE:
        return FeatureCacheManager.compute_hash(cfg)
    # Inline fallback (identical logic)
    key_dict = {
        'bp_include_list':      cfg.get('bp_include_list'),
        'bp_pixbrt_list':       list(cfg.get('bp_pixbrt_list', [])),
        'square_size':          [int(x) for x in cfg.get('square_size', [])],
        'pix_threshold':        round(float(cfg.get('pix_threshold', 0.3)), 6),
        'pose_feature_version': int(POSE_FEATURE_VERSION),
        'include_optical_flow': bool(cfg.get('include_optical_flow', False)),
        'bp_optflow_list':      list(cfg.get('bp_optflow_list', [])),
    }
    return hashlib.md5(repr(key_dict).encode('utf-8')).hexdigest()[:8]


def _categorize_col(col: str) -> str:
    """
    Assign a feature column to one of seven categories based on naming
    conventions used by PoseFeatureExtractor and BrightnessExtractor.

    Returns one of:
        'brightness', 'angles', 'accel_jerk', 'velocities',
        'body_shape', 'distances', 'confidence'
    """
    # Brightness: Pix_{bp}, Log10(ratio), |d/dt(...)| derivatives, optical flow
    if (col.startswith('Pix_') or col.startswith('Log10(')
            or '|d/dt(' in col or '_Flow' in col):
        return 'brightness'
    # Angles: Ang_{bp1}-{bp2}-{bp3}
    if col.startswith('Ang_'):
        return 'angles'
    # Detection confidence: {bp}_inFrame_p{threshold}
    if '_inFrame_' in col:
        return 'confidence'
    # Acceleration / Jerk: {bp}_Accel{t}, {bp}_Jerk{t}
    if '_Accel' in col or '_Jerk' in col:
        return 'accel_jerk'
    # Velocities: {bp}_Vel{t}, Dis_{bp1-bp2}_Vel{t}, sum_Vel{t}
    # Excludes rolling-window stats (_VelMaxW / _VelStdW)
    if '_Vel' in col and '_VelMaxW' not in col and '_VelStdW' not in col:
        return 'velocities'
    # Body shape: rolling-window summaries
    if '_VelMaxW' in col or '_VelStdW' in col:
        return 'body_shape'
    # Distances: Dis_{bp1}-{bp2} (pairwise Euclidean)
    if col.startswith('Dis_') or '_' in col:
        return 'distances'
    return 'body_shape'


def _n_clusters_label(labels: np.ndarray) -> str:
    n = len(set(labels)) - (1 if -1 in labels else 0)
    return f"{n} cluster{'s' if n != 1 else ''}"


# ===========================================================================
# UnsupervisedTab
# ===========================================================================

class UnsupervisedTab(ttk.Frame):
    """
    Unsupervised behavior discovery tab.

    Parameters
    ----------
    parent : tk widget
        The container frame added to the notebook.
    parent_app : PixelPawsGUI
        Reference to the main application instance.  Used to access:
            app.root                  — Tk root (for after() calls)
            app.current_project_folder — shared project-folder StringVar
    """

    # Human-readable labels for the six feature category checkboxes
    CAT_LABELS = {
        'velocities': 'Velocities',
        'distances':  'Distances',
        'angles':     'Angles',
        'accel_jerk': 'Acceleration / Jerk',
        'body_shape': 'Body shape',
        'brightness': 'Brightness',
        'confidence': 'Confidence (inFrame)',
    }

    # Categories that start unchecked (detection quality scores are noisy
    # as behavioral predictors; keep off by default)
    _CAT_DEFAULTS_OFF = {'confidence'}

    def __init__(self, parent, parent_app):
        super().__init__(parent)
        self.app = parent_app
        self.pack(fill='both', expand=True)

        # ── In-memory state ────────────────────────────────────────────────
        self._sessions: list          = []    # list of session dicts from scan
        self._model_bundle: dict      = None  # last fitted or loaded bundle
        self._embedding: np.ndarray   = None  # (N, 2) UMAP embedding
        self._labels: np.ndarray      = None  # (N,)  cluster labels
        self._session_row_map: dict   = {}    # {name: (start, end)} into arrays
        self._selected_cluster: int   = None
        self._frame_indices: dict     = {}
        self._video_paths: dict       = {}
        self._viz_win                 = None  # detached Toplevel for UMAP scatter
        self._inspect_win             = None  # Inspector Toplevel
        self._bar_fig                 = None  # cluster-sizes figure (in inspector)
        self._bar_ax                  = None
        self._bar_canvas              = None
        self._legend_canvas           = None  # Tk scrollable legend
        self._legend_inner            = None  # inner Frame inside legend canvas
        self._inspect_title_var       = None  # StringVar for inspector title label
        self._thumb_canvases          = []
        self._thumb_label             = None  # set when inspector panel is built

        # ── Threading ──────────────────────────────────────────────────────
        self._fit_thread: threading.Thread = None
        self._cancel_flag = threading.Event()

        self._build_ui()

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self):
        if not UMAP_HDBSCAN_AVAILABLE:
            self._build_missing_deps_ui()
            return

        # Title strip
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(hdr, text="🔍 Unsupervised Behavior Discovery",
                  font=('Arial', 14, 'bold')).pack(side='left')

        # ── Three-column paned window (fills the tab) ─────────────────────
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left_frame  = ttk.Frame(paned, width=260)
        mid_frame   = ttk.Frame(paned, width=260)
        right_frame = ttk.Frame(paned, width=360)

        paned.add(left_frame,  weight=1)
        paned.add(mid_frame,   weight=1)
        paned.add(right_frame, weight=2)

        self._build_sessions_panel(left_frame)
        self._build_settings_panel(mid_frame)
        self._build_actions_panel(right_frame)

    def _build_missing_deps_ui(self):
        """Friendly install-instructions widget shown when deps are absent."""
        wrapper = ttk.Frame(self)
        wrapper.place(relx=0.5, rely=0.4, anchor='center')

        ttk.Label(wrapper, text="⚠️  Missing dependencies",
                  font=('Arial', 14, 'bold')).pack(pady=(0, 12))
        ttk.Label(wrapper,
                  text="The Discover tab requires umap-learn and hdbscan.\n"
                       "Install them in your PixelPaws environment:",
                  justify='center').pack()

        code = tk.Text(wrapper, height=3, width=52, font=('Consolas', 11),
                       relief='solid', borderwidth=1)
        code.insert('1.0',
                    "# activate your venv first, then:\n"
                    "pip install umap-learn hdbscan\n"
                    "# Restart PixelPaws afterwards.")
        try:
            _code_bg = self.app.style.colors.inputbg
        except Exception:
            _code_bg = '#f5f5f5'
        code.config(state='disabled', bg=_code_bg)
        code.pack(pady=8)

    # ------------------------------------------------------------------
    # Left column: Sessions
    # ------------------------------------------------------------------

    def _build_sessions_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="Sessions", padding=5)
        lf.pack(fill='both', expand=True, padx=4, pady=4)

        # Toolbar buttons
        btn_row = ttk.Frame(lf)
        btn_row.pack(fill='x', pady=(0, 4))
        ttk.Button(btn_row, text="🔄 Scan",  width=8,
                   command=self._scan_sessions).pack(side='left', padx=(0, 4))
        ttk.Button(btn_row, text="✓ All",  width=6,
                   command=lambda: self._select_all(True)).pack(side='left', padx=(0, 2))
        ttk.Button(btn_row, text="✗ None", width=6,
                   command=lambda: self._select_all(False)).pack(side='left')

        # Treeview
        cols = ('frames', 'status')
        self._tree = ttk.Treeview(lf, columns=cols, show='tree headings',
                                  selectmode='extended', height=12)
        self._tree.heading('#0',      text='Session')
        self._tree.heading('frames',  text='Frames')
        self._tree.heading('status',  text='Status')
        self._tree.column('#0',      width=120, stretch=True)
        self._tree.column('frames',  width=55,  anchor='e')
        self._tree.column('status',  width=90,  anchor='center')

        vsb = ttk.Scrollbar(lf, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        # Colour tags
        self._tree.tag_configure('in_model', foreground='#2a7a2a')
        self._tree.tag_configure('new',      foreground='#b07000')
        self._tree.tag_configure('excluded', foreground='#888888')

        # Video extension selector
        ext_row = ttk.Frame(lf)
        ext_row.pack(fill='x', pady=(6, 0))
        ttk.Label(ext_row, text="Video ext:").pack(side='left')
        self._video_ext_var = tk.StringVar(value='.mp4')
        ttk.Combobox(ext_row, textvariable=self._video_ext_var,
                     values=['.mp4', '.avi', '.MP4', '.AVI'],
                     width=7).pack(side='left', padx=4)

    # ------------------------------------------------------------------
    # Middle column: Settings
    # ------------------------------------------------------------------

    def _build_settings_panel(self, parent):
        outer_lf = ttk.LabelFrame(parent, text="Settings", padding=5)
        outer_lf.pack(fill='both', expand=True, padx=4, pady=4)

        # ── Scrollable canvas wrapper ──────────────────────────────────────
        _canvas = tk.Canvas(outer_lf, borderwidth=0, highlightthickness=0)
        _vsb    = ttk.Scrollbar(outer_lf, orient='vertical',
                                command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side='right', fill='y')
        _canvas.pack(side='left', fill='both', expand=True)

        lf = ttk.Frame(_canvas)          # inner frame — all content goes here
        _cw = _canvas.create_window((0, 0), window=lf, anchor='nw')

        def _on_inner_cfg(e):
            _canvas.configure(scrollregion=_canvas.bbox('all'))
        def _on_canvas_cfg(e):
            _canvas.itemconfig(_cw, width=e.width)
        def _on_mw(e):
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')

        lf.bind('<Configure>', _on_inner_cfg)
        _canvas.bind('<Configure>', _on_canvas_cfg)
        # Activate scroll wheel only while the mouse is inside the settings pane
        _canvas.bind('<Enter>', lambda e: _canvas.bind_all('<MouseWheel>', _on_mw))
        _canvas.bind('<Leave>', lambda e: _canvas.unbind_all('<MouseWheel>'))

        # Feature categories
        cat_lf = ttk.LabelFrame(lf, text="Feature categories", padding=4)
        cat_lf.pack(fill='x', pady=(0, 8))
        self._cat_vars = {}
        _cat_tips = {
            'velocities':  "Speed of each body part between frames. Core motion signal for most behaviors.",
            'distances':   "Pairwise distances between body parts. Captures body posture and proximity.",
            'angles':      "Joint angles between body-part triplets. Sensitive to limb configuration and posture.",
            'accel_jerk':  "Rate of change of velocity. Captures abrupt movements and impact events.",
            'body_shape':  "Rolling-window velocity summaries (max, std). Captures movement variability over short windows.",
            'brightness':  "Pixel brightness under each body part. Useful for environment-linked behaviors (e.g., light/dark zones).",
            'confidence':  "DLC detection quality scores. Noisy as behavioral features — leave off unless you specifically need them.",
        }
        for key, label in self.CAT_LABELS.items():
            default = key not in self._CAT_DEFAULTS_OFF
            var = tk.BooleanVar(value=default)
            self._cat_vars[key] = var
            cb = ttk.Checkbutton(cat_lf, text=label, variable=var)
            cb.pack(anchor='w')
            self._tip(cb, _cat_tips.get(key, ''))

        # Temporal parameters
        temp_lf = ttk.LabelFrame(lf, text="Temporal", padding=4)
        temp_lf.pack(fill='x', pady=(0, 8))
        self._target_fps_var    = tk.IntVar(value=20)
        self._fallback_fps_var  = tk.IntVar(value=60)
        self._smooth_ms_var     = tk.IntVar(value=60)
        self._min_bout_ms_var   = tk.IntVar(value=200)
        lbl, sb = self._spinrow(temp_lf, "Target fps:", self._target_fps_var, 0, 500, 0)
        ttk.Label(temp_lf, text="0 = skip windowing",
                  foreground='grey').grid(row=0, column=2, sticky='w', padx=2)
        _t = ("Downsample the feature time series to this rate before UMAP. "
              "Lower = faster fit, smoother clusters, but loses fast events. "
              "0 = use the original video frame rate.")
        self._tip(lbl, _t); self._tip(sb, _t)

        lbl, sb = self._spinrow(temp_lf, "Fallback fps:", self._fallback_fps_var, 1, 500, 1)
        ttk.Label(temp_lf, text="if video unreadable",
                  foreground='grey').grid(row=1, column=2, sticky='w', padx=2)
        _t = ("Used when the video file cannot be opened to read its actual fps. "
              "Has no effect if all videos are readable.")
        self._tip(lbl, _t); self._tip(sb, _t)

        lbl, sb = self._spinrow(temp_lf, "Smooth (ms):", self._smooth_ms_var, 0, 5000, 2)
        _t = ("Median-filter cluster labels after HDBSCAN — removes isolated label "
              "flickers shorter than this duration. 0 = no smoothing.")
        self._tip(lbl, _t); self._tip(sb, _t)
        self._make_ms_indicator(temp_lf, self._smooth_ms_var, self._target_fps_var, row=2)

        lbl, sb = self._spinrow(temp_lf, "Min bout (ms):", self._min_bout_ms_var, 0, 10000, 3)
        _t = ("Drop any contiguous cluster run shorter than this after smoothing. "
              "Prevents spurious short clusters in exported CSVs.")
        self._tip(lbl, _t); self._tip(sb, _t)
        self._make_ms_indicator(temp_lf, self._min_bout_ms_var, self._target_fps_var, row=3)

        # UMAP parameters
        umap_lf = ttk.LabelFrame(lf, text="UMAP", padding=4)
        umap_lf.pack(fill='x', pady=(0, 8))
        self._umap_neighbors_var  = tk.IntVar(value=15)
        self._umap_auto_neighbors = tk.BooleanVar(value=False)
        self._umap_min_dist_var   = tk.StringVar(value='0.0')
        self._max_fit_frames_var  = tk.IntVar(value=100000)
        self._pca_min_var_var     = tk.StringVar(value='0.70')

        lbl, sb = self._spinrow(umap_lf, "n_neighbors:", self._umap_neighbors_var, 2, 500, 0)
        self._neighbors_spin = sb
        auto_cb = ttk.Checkbutton(
            umap_lf, text="Auto (sqrt N)",
            variable=self._umap_auto_neighbors,
            command=self._toggle_auto_neighbors)
        auto_cb.grid(row=0, column=2, sticky='w', padx=2)
        _t = ("Balances local vs global structure. Low (5–10) = fine local detail. "
              "High (30–100) = broad global topology. Typical: 15.\n"
              "Auto = sqrt(N) as in B-SOiD (scales with dataset size).")
        self._tip(lbl, _t); self._tip(sb, _t); self._tip(auto_cb, _t)

        lbl, ent = self._entryrow(umap_lf, "min_dist:", self._umap_min_dist_var, 1)
        _t = ("How tightly UMAP packs points in 2D. "
              "0.0 = maximum cluster density. Higher values spread clusters apart.")
        self._tip(lbl, _t); self._tip(ent, _t)

        lbl, sb = self._spinrow(umap_lf, "Max frames for fit:", self._max_fit_frames_var, 0, 2000000, 2)
        ttk.Label(umap_lf, text="0 = use all frames",
                  foreground='grey').grid(row=3, column=1, sticky='w', padx=4)
        _t = ("UMAP training cap. Excess frames are randomly sampled; all frames are "
              "transformed afterwards. 0 = no limit (slow on large datasets).")
        self._tip(lbl, _t); self._tip(sb, _t)

        lbl, ent = self._entryrow(umap_lf, "PCA min var:", self._pca_min_var_var, 4)
        ttk.Label(umap_lf, text="0.0 = skip PCA",
                  foreground='grey').grid(row=4, column=2, sticky='w', padx=2)
        _t = ("Reduce features to PCA components explaining this fraction of variance "
              "before UMAP. Speeds up fitting on high-dimensional data. 0.0 = skip PCA.")
        self._tip(lbl, _t); self._tip(ent, _t)

        # HDBSCAN parameters
        hdb_lf = ttk.LabelFrame(lf, text="HDBSCAN", padding=4)
        hdb_lf.pack(fill='x', pady=(0, 8))
        self._hdb_min_cluster_var = tk.IntVar(value=300)
        self._hdb_min_samples_var = tk.IntVar(value=1)

        lbl, sb = self._spinrow(hdb_lf, "min_cluster_size:", self._hdb_min_cluster_var, 2, 10000, 0)
        _t = ("Smallest group of frames that can form a cluster. Increase to merge "
              "small noisy clusters; decrease to split fine-grained behaviors.")
        self._tip(lbl, _t); self._tip(sb, _t)

        lbl, sb = self._spinrow(hdb_lf, "min_samples:", self._hdb_min_samples_var, 1, 1000, 1)
        _t = ("Controls noise sensitivity. 1 = most frames assigned to a cluster. "
              "Higher values mark more frames as noise (label −1).")
        self._tip(lbl, _t); self._tip(sb, _t)

        # Run name
        run_lf = ttk.LabelFrame(lf, text="Run", padding=4)
        run_lf.pack(fill='x')
        self._run_name_var = tk.StringVar(value='run1')
        run_lbl = ttk.Label(run_lf, text="Name:")
        run_lbl.grid(row=0, column=0, sticky='w')
        run_ent = ttk.Entry(run_lf, textvariable=self._run_name_var, width=16)
        run_ent.grid(row=0, column=1, padx=4, sticky='w')
        _t = ("Subfolder under unsupervised/ where output CSVs and the model bundle "
              "are saved. Change this to keep multiple runs side-by-side.")
        self._tip(run_lbl, _t); self._tip(run_ent, _t)

    def _spinrow(self, parent, label, var, from_, to, row):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky='w', pady=2)
        sb = ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=8)
        sb.grid(row=row, column=1, sticky='w', padx=4, pady=2)
        return lbl, sb

    def _entryrow(self, parent, label, var, row):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky='w', pady=2)
        ent = ttk.Entry(parent, textvariable=var, width=8)
        ent.grid(row=row, column=1, sticky='w', padx=4, pady=2)
        return lbl, ent

    def _tip(self, widget, text):
        """Attach a hover tooltip to *widget*."""
        _ToolTip(widget, text)

    def _make_ms_indicator(self, parent, ms_var, fps_var, row, col=2):
        """Create a live '= N fr' label in *parent* at (row, col)."""
        ind = ttk.Label(parent, foreground='grey')
        ind.grid(row=row, column=col, sticky='w', padx=2)

        def _refresh(*_):
            try:
                ms  = int(ms_var.get())
                fps = int(fps_var.get())
            except (ValueError, tk.TclError):
                ind.config(text='')
                return
            if fps > 0 and ms > 0:
                n = max(1, round(ms * fps / 1000))
                ind.config(text=f'= {n} fr')
            elif ms > 0:
                ind.config(text='(orig fps)')
            else:
                ind.config(text='off')

        ms_var.trace_add('write', _refresh)
        fps_var.trace_add('write', _refresh)
        _refresh()
        return ind

    def _toggle_auto_neighbors(self):
        """Enable/disable the n_neighbors spinbox based on auto checkbox."""
        if self._umap_auto_neighbors.get():
            self._neighbors_spin.config(state='disabled')
        else:
            self._neighbors_spin.config(state='normal')

    # ------------------------------------------------------------------
    # Right column: Actions + Log
    # ------------------------------------------------------------------

    def _build_actions_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="Actions", padding=5)
        lf.pack(fill='both', expand=True, padx=4, pady=4)

        actions = [
            ("⚙️  Fit model",          self._start_fit,
             "Fit UMAP + HDBSCAN on selected sessions (replaces existing model)"),
            ("➕  Assign new",         self._start_assign,
             "Transform new sessions using the saved model (no re-fit)"),
            ("📂  Load model",         self._load_model_from_file,
             "Load a previously saved model.pkl and restore the scatter"),
            ("📊  Open visualization", self._open_viz_window,
             "Open the UMAP scatter + cluster inspector in a resizable window"),
            ("🔀  Merge clusters",     self._merge_clusters,
             "Reassign all frames of one or more clusters into another cluster"),
            ("💾  Export labels",      self._export_labels,
             "Write per-session cluster CSVs to unsupervised/<run>/"),
            ("🌲  Train RF classifier", self._start_train_rf,
             "Train a Random Forest on cluster labels and save to classifiers/"),
            ("📷  Export figures",     self._export_figures,
             "Save UMAP scatter + cluster-size chart as high-resolution PNG or PDF"),
            ("🗑  Cancel",             self._cancel,
             "Cancel the running operation"),
        ]
        for text, cmd, _ in actions:
            ttk.Button(lf, text=text, command=cmd).pack(fill='x', pady=2)

        self._progress = ttk.Progressbar(lf, mode='indeterminate')
        self._progress.pack(fill='x', pady=(8, 2))

        log_lf = ttk.LabelFrame(lf, text="Log", padding=3)
        log_lf.pack(fill='both', expand=True, pady=(6, 0))
        self._log = scrolledtext.ScrolledText(
            log_lf, height=8, width=40,
            font=('Consolas', 9), state='disabled')
        self._log.pack(fill='both', expand=True)

    # ------------------------------------------------------------------
    # Detached visualization window
    # ------------------------------------------------------------------

    def _open_viz_window(self):
        """Open (or raise) the detached UMAP scatter + Tkinter legend Toplevel."""
        # Re-use existing window if still alive
        if self._viz_win is not None:
            try:
                self._viz_win.lift()
                self._viz_win.focus_set()
                # Redraw in case labels changed (e.g. after merge)
                if self._embedding is not None and self._labels is not None:
                    self._draw_scatter(self._embedding, self._labels,
                                       self._selected_cluster)
                return
            except tk.TclError:
                pass  # window was destroyed externally

        # Clear stale figure references so _build_viz_panel creates fresh ones
        self._fig          = None
        self._canvas       = None
        self._ax_scatter   = None
        self._legend_canvas = None
        self._legend_inner  = None

        win = tk.Toplevel(self.app.root)
        win.title("UMAP Visualization — PixelPaws Discover")
        _sw = win.winfo_screenwidth()
        _sh = win.winfo_screenheight()
        _w = int(_sw * 0.75)
        _h = int(_sh * 0.75)
        win.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", self._on_viz_win_close)
        self._viz_win = win

        win.bind('<Left>',  lambda e: self._select_adjacent_cluster(-1))
        win.bind('<Right>', lambda e: self._select_adjacent_cluster(+1))
        ttk.Label(win, text="← → arrow keys to cycle clusters",
                  foreground='grey').pack(side='bottom', pady=2)

        self._build_viz_panel(win)

        # Draw existing data immediately if available
        if self._embedding is not None and self._labels is not None:
            self._draw_scatter(self._embedding, self._labels,
                               self._selected_cluster)

    def _on_viz_win_close(self):
        """Clean up figure memory and references when the viz window is closed."""
        try:
            if self._fig is not None:
                plt.close(self._fig)
        except Exception:
            pass
        self._fig           = None
        self._canvas        = None
        self._ax_scatter    = None
        self._legend_canvas = None
        self._legend_inner  = None
        try:
            self._viz_win.destroy()
        except Exception:
            pass
        self._viz_win = None

    # ------------------------------------------------------------------
    # Bottom: Visualization
    # ------------------------------------------------------------------

    def _build_viz_panel(self, parent):
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent,
                      text="matplotlib not available — install it for scatter plots."
                      ).pack(padx=10, pady=10)
            self._fig    = None
            self._canvas = None
            return

        # ── Horizontal layout: scatter (left, expands) + legend (right, fixed) ──
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True)

        scatter_frame = ttk.Frame(main_frame)
        scatter_frame.pack(side='left', fill='both', expand=True)

        legend_lf = ttk.LabelFrame(main_frame, text="Clusters", width=130)
        legend_lf.pack(side='right', fill='y')
        legend_lf.pack_propagate(False)
        self._build_legend_panel(legend_lf)

        self._fig, self._ax_scatter = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        self._canvas = FigureCanvasTkAgg(self._fig, master=scatter_frame)
        toolbar = NavigationToolbar2Tk(self._canvas, scatter_frame)
        toolbar.update()
        toolbar.pack(side='top', fill='x')
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(self._canvas, self._fig)
        self._canvas.mpl_connect('button_press_event', self._on_scatter_click)

        self._ax_scatter.set_title("UMAP embedding — run Fit model to populate")
        self._ax_scatter.set_xlabel("UMAP 1")
        self._ax_scatter.set_ylabel("UMAP 2")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Legend panel (scrollable Tkinter, replaces matplotlib legend)
    # ------------------------------------------------------------------

    def _build_legend_panel(self, parent):
        """Build a scrollable Tkinter legend inside *parent*."""
        self._legend_canvas = tk.Canvas(parent, width=120, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient='vertical',
                           command=self._legend_canvas.yview)
        self._legend_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._legend_canvas.pack(side='left', fill='both', expand=True)
        self._legend_inner = ttk.Frame(self._legend_canvas)
        self._legend_canvas.create_window((0, 0), window=self._legend_inner,
                                          anchor='nw')
        self._legend_inner.bind(
            '<Configure>',
            lambda e: self._legend_canvas.configure(
                scrollregion=self._legend_canvas.bbox('all')))
        self._legend_canvas.bind_all(
            '<MouseWheel>',
            lambda e: self._legend_canvas.yview_scroll(
                -1 * (e.delta // 120), 'units'))

    def _update_legend(self, cluster_ids, cmap, n_clusters, selected_cluster):
        """Rebuild the Tkinter legend entries to match current clusters."""
        if self._legend_inner is None:
            return
        for w in self._legend_inner.winfo_children():
            w.destroy()
        try:
            bg = self._legend_inner.winfo_toplevel().cget('bg')
        except Exception:
            bg = '#f0f0f0'
        for i, lbl in enumerate(cluster_ids):
            col = cmap(i / max(n_clusters - 1, 1))
            hex_col = '#{:02x}{:02x}{:02x}'.format(
                int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            row_bg = '#ffe8e8' if lbl == selected_cluster else bg
            row = tk.Frame(self._legend_inner, bg=row_bg, cursor='hand2')
            row.pack(fill='x', pady=0)
            dot = tk.Canvas(row, width=14, height=14, bg=row_bg,
                            highlightthickness=0)
            dot.create_oval(2, 2, 12, 12, fill=hex_col, outline='')
            dot.pack(side='left', padx=(2, 1))
            lbl_w = tk.Label(row, text=f'C{lbl}', font=('TkDefaultFont', 8),
                             bg=row_bg, cursor='hand2')
            lbl_w.pack(side='left')
            for w in (row, dot, lbl_w):
                w.bind('<Button-1>',
                       lambda e, cid=lbl: self._on_legend_click(cid))

    def _on_legend_click(self, cluster_id: int):
        """Handle a click on a legend entry."""
        self._selected_cluster = cluster_id
        self._draw_scatter(self._embedding, self._labels,
                           selected_cluster=cluster_id)
        self.app.root.after(0, lambda: self._show_cluster_inspector(cluster_id))

    # ------------------------------------------------------------------
    # Inspector window
    # ------------------------------------------------------------------

    def _open_inspect_win(self):
        """Create (or raise) the cluster inspector Toplevel."""
        if self._inspect_win is not None:
            try:
                self._inspect_win.lift()
                return
            except tk.TclError:
                pass
        # Clear stale references
        self._bar_fig = self._bar_ax = self._bar_canvas = None
        self._etho_fig = self._etho_ax = self._etho_canvas = None
        self._thumb_canvases = []
        win = tk.Toplevel(self.app.root)
        win.title("Cluster Inspector — PixelPaws Discover")
        _sw = win.winfo_screenwidth()
        _sh = win.winfo_screenheight()
        _w = int(_sw * 0.75)
        _h = int(_sh * 0.70)
        win.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", self._on_inspect_win_close)
        self._inspect_win = win
        self._inspect_title_var = tk.StringVar(
            value="Select a cluster in the scatter")
        ttk.Label(win, textvariable=self._inspect_title_var,
                  font=('TkDefaultFont', 10, 'bold')).pack(pady=4)
        self._build_inspect_panel(win)

    def _on_inspect_win_close(self):
        """Clean up inspector-specific figures when the inspector is closed."""
        for fig in (self._etho_fig, self._bar_fig):
            try:
                if fig:
                    plt.close(fig)
            except Exception:
                pass
        self._etho_fig = self._etho_ax = self._etho_canvas = None
        self._bar_fig = self._bar_ax = self._bar_canvas = None
        self._thumb_canvases = []
        try:
            self._inspect_win.destroy()
        except Exception:
            pass
        self._inspect_win = None

    def _build_inspect_panel(self, parent):
        """Build thumbnail grid, ethogram, and bar chart inside the inspector."""
        content = ttk.Frame(parent)
        content.pack(fill='both', expand=True, padx=4, pady=4)

        # ── Left column: thumbnail grid ──────────────────────────────────
        thumb_lf = ttk.LabelFrame(content, text="Sample Frames", padding=4)
        thumb_lf.pack(side='left', fill='both')

        ttk.Button(thumb_lf, text="▶  Play bouts",
                   command=lambda: self._open_bout_player(
                       self._selected_cluster)
                   ).grid(row=0, column=0, columnspan=3, sticky='ew',
                          padx=2, pady=(0, 4))

        self._thumb_canvases = []
        for r in range(3):
            for c in range(3):
                try:
                    _thumb_bg = self.app.style.colors.bg
                except Exception:
                    _thumb_bg = '#1e1e1e'
                cv = tk.Canvas(thumb_lf, width=140, height=105, bg=_thumb_bg)
                cv.grid(row=r + 1, column=c, padx=2, pady=2)
                self._thumb_canvases.append(cv)

        self._thumb_label = ttk.Label(thumb_lf, text="")
        self._thumb_label.grid(row=4, column=0, columnspan=3)

        # ── Right column: ethogram + bar chart stacked ───────────────────
        right_col = ttk.Frame(content)
        right_col.pack(side='left', fill='both', expand=True, padx=(8, 0))

        etho_lf = ttk.LabelFrame(right_col, text="Session Ethogram", padding=4)
        etho_lf.pack(fill='both', expand=True)

        bar_lf = ttk.LabelFrame(right_col, text="Cluster Sizes", padding=4)
        bar_lf.pack(fill='both', expand=True, pady=(6, 0))

        if MATPLOTLIB_AVAILABLE:
            self._etho_fig, self._etho_ax = plt.subplots(figsize=(6, 2.5), constrained_layout=True)
            self._etho_canvas = FigureCanvasTkAgg(
                self._etho_fig, master=etho_lf)
            self._etho_canvas.get_tk_widget().pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(self._etho_canvas, self._etho_fig)

            self._bar_fig, self._bar_ax = plt.subplots(figsize=(6, 2.5), constrained_layout=True)
            self._bar_canvas = FigureCanvasTkAgg(
                self._bar_fig, master=bar_lf)
            self._bar_canvas.get_tk_widget().pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(self._bar_canvas, self._bar_fig)
        else:
            self._etho_fig = self._etho_ax = self._etho_canvas = None
            self._bar_fig = self._bar_ax = self._bar_canvas = None

    def _draw_bar_chart(self):
        """Draw the cluster-size bar chart in the inspector window."""
        if not MATPLOTLIB_AVAILABLE or self._bar_ax is None or self._labels is None:
            return
        ax = self._bar_ax
        ax.clear()
        cluster_ids = sorted(lbl for lbl in set(self._labels) if lbl >= 0)
        n_clusters = len(cluster_ids)
        cmap = plt.get_cmap('turbo' if n_clusters > 20 else 'tab20',
                            max(n_clusters, 1))
        if not cluster_ids:
            try:
                self._bar_canvas.draw()
            except Exception:
                pass
            return
        counts = [(lbl, int((self._labels == lbl).sum())) for lbl in cluster_ids]
        bar_lbls, bar_counts = zip(*counts)
        bar_cols = [cmap(cluster_ids.index(l) / max(n_clusters - 1, 1))
                    for l in bar_lbls]
        ax.bar([f'C{l}' for l in bar_lbls], bar_counts, color=bar_cols)
        ax.set_ylabel("Frames")
        ax.set_title("Cluster sizes", fontsize=9)
        noise_pct = (self._labels == -1).mean() * 100
        ax.set_xlabel(f"noise: {noise_pct:.1f}%")
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        pass  # constrained_layout handles this
        try:
            self._bar_canvas.draw()
        except Exception:
            pass

    # ======================================================================
    # Logging
    # ======================================================================

    def _log_msg(self, msg: str):
        """Thread-safe log append."""
        def _do():
            self._log.config(state='normal')
            self._log.insert('end', msg + '\n')
            self._log.see('end')
            self._log.config(state='disabled')
        try:
            self.app.root.after(0, _do)
        except Exception:
            pass

    def _clear_log(self):
        def _do():
            self._log.config(state='normal')
            self._log.delete('1.0', 'end')
            self._log.config(state='disabled')
        self.app.root.after(0, _do)

    # ======================================================================
    # Session management
    # ======================================================================

    def on_project_changed(self):
        """Called by PixelPawsGUI._on_project_folder_changed."""
        folder = self.app.current_project_folder.get()
        if folder and os.path.isdir(folder):
            self._scan_sessions()

    def _scan_sessions(self):
        """Populate treeview from the current project folder."""
        if not _FIND_SESSIONS_AVAILABLE:
            messagebox.showerror(
                "Missing module",
                "evaluation_tab.py is required for session discovery.\n"
                "Place it in the same folder as PixelPaws_GUI.py.")
            return

        folder = self.app.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No project",
                                   "Please select a project folder first.")
            return

        ext = self._video_ext_var.get() or '.mp4'
        try:
            sessions = find_session_triplets(
                folder,
                video_ext=ext,
                require_labels=False,   # unsupervised — labels not needed
            )
        except Exception as exc:
            messagebox.showerror("Scan error", str(exc))
            return

        self._sessions = sessions

        # Determine which sessions are already in the saved model
        bundle       = self._try_load_model_bundle(folder)
        in_model_set = set(bundle.get('sessions', {}).keys()) if bundle else set()

        # Feature cache hash for cache-status column
        cfg      = self._load_feature_cfg(folder)
        cfg_hash = _feature_hash_key(cfg) if cfg else None
        cache_dir = os.path.join(folder, 'features')

        # Clear treeview
        for iid in self._tree.get_children():
            self._tree.delete(iid)

        for s in sessions:
            name = s['session_name']

            # Is there a feature cache for this session?
            cached = self._find_feature_cache(s, cfg_hash, cache_dir) is not None \
                if cfg_hash else False

            if name in in_model_set:
                status = 'in model'
                tag    = 'in_model'
            else:
                status = 'new'
                tag    = 'new'

            status_str = f"{status}  ({'cached' if cached else 'needs extract'})"
            self._tree.insert('', 'end', iid=name, text=name,
                              values=('—', status_str), tags=(tag,))

        self._select_all(True)
        self._log_msg(f"[Scan] {len(sessions)} session(s) in {folder}")

    def _select_all(self, select: bool):
        items = self._tree.get_children()
        if select:
            self._tree.selection_set(items)
        else:
            self._tree.selection_remove(items)

    def _selected_names(self) -> list:
        return list(self._tree.selection())

    # ======================================================================
    # Feature config + cache
    # ======================================================================

    def _load_feature_cfg(self, folder: str) -> dict:
        """
        Build a feature-extraction config dict for this project.

        Sources (highest priority first):
          1. The last trained classifier .pkl — it always stores the exact
             params that were used, so the hash will match existing cache files.
          2. PixelPaws_project.json — fallback when no classifier exists yet.

        Legacy mapping: project.json may store ``roi_size`` (int or list)
        instead of ``square_size``; both forms are normalised to a list.
        """
        # --- 1. Load project.json as base ---
        proj_json_path = os.path.join(folder, 'PixelPaws_project.json')
        cfg = {}
        if os.path.isfile(proj_json_path):
            try:
                with open(proj_json_path, 'r') as fh:
                    cfg = json.load(fh)
            except Exception as _e:
                print(f"Warning: could not load project config {proj_json_path}: {_e}")

        # Legacy: roi_size → square_size
        if 'square_size' not in cfg and 'roi_size' in cfg:
            rs = cfg['roi_size']
            cfg['square_size'] = [int(rs)] if not isinstance(rs, list) \
                                  else [int(x) for x in rs]

        # --- 2. Override with values from the last saved classifier ---
        clf_path = cfg.get('last_classifier', '')
        if clf_path and os.path.isfile(clf_path):
            try:
                with open(clf_path, 'rb') as fh:
                    clf_data = pickle.load(fh)
                if isinstance(clf_data, dict):
                    for key in ('bp_include_list', 'bp_pixbrt_list',
                                'square_size', 'pix_threshold',
                                'include_optical_flow', 'bp_optflow_list'):
                        if key in clf_data:
                            cfg[key] = clf_data[key]
            except Exception as _e:
                print(f"Warning: could not load classifier {clf_path}: {_e}")

        # --- 3. Fill remaining gaps with safe defaults ---
        cfg.setdefault('square_size',          [40])
        cfg.setdefault('pix_threshold',        0.3)
        cfg.setdefault('bp_pixbrt_list',       [])
        cfg.setdefault('bp_include_list',      None)
        cfg.setdefault('include_optical_flow', False)
        cfg.setdefault('bp_optflow_list',      [])
        return cfg

    def _find_feature_cache(self, session: dict, cfg_hash: Optional[str],
                            cache_dir: str) -> Optional[str]:
        """
        Return path to a feature .pkl file for *session*, or None.

        Delegates to FeatureCacheManager when available, with an inline
        fallback for graceful degradation.
        """
        name = session['session_name']
        vdir = (session.get('video_dir') or
                os.path.dirname(session.get('video_path', '') or
                                session.get('video', '')))
        pdir = (session.get('project_dir') or
                os.path.dirname(vdir))

        if _FEATURE_CACHE_AVAILABLE:
            # Pass 1: exact hash
            if cfg_hash:
                found = FeatureCacheManager.find_cache(
                    name, cfg_hash, cache_dir, vdir, project_root=pdir)
                if found:
                    return found
            # Pass 2: any hash (fallback)
            return FeatureCacheManager.find_any_cache(
                name, cache_dir, vdir, project_root=pdir)

        # Inline fallback
        search_dirs = [
            cache_dir,
            os.path.join(pdir, 'features'),
            os.path.join(pdir, 'FeatureCache'),
            os.path.join(vdir, 'features'),
            os.path.join(vdir, 'FeatureCache'),
            vdir, pdir,
        ]
        seen = set()
        dirs = []
        for d in search_dirs:
            if d and d not in seen:
                seen.add(d)
                dirs.append(d)
        if cfg_hash:
            fname = f"{name}_features_{cfg_hash}.pkl"
            for d in dirs:
                candidate = os.path.join(d, fname)
                if os.path.isfile(candidate):
                    return candidate
        pattern = f"{name}_features_*.pkl"
        for d in dirs:
            if not os.path.isdir(d):
                continue
            matches = glob.glob(os.path.join(d, pattern))
            if matches:
                plain = [m for m in matches if '_corrected' not in m]
                return sorted(plain or matches, key=os.path.getmtime)[-1]
        return None

    def _load_features_for_session(self, session: dict, cfg: dict,
                                   cache_dir: str) -> Optional[pd.DataFrame]:
        """
        Return a feature DataFrame for *session*.

        Tries the feature cache first; falls back to on-the-fly extraction
        (using PixelPaws_ExtractFeatures via a late import) and saves the
        result to the cache.
        """
        name     = session['session_name']
        cfg_hash = _feature_hash_key(cfg)
        hit      = self._find_feature_cache(session, cfg_hash, cache_dir)

        if hit:
            hit_hash = os.path.basename(hit).split('_features_')[1].replace('.pkl', '')
            if hit_hash != cfg_hash:
                self._log_msg(
                    f"  [Cache] {name}  (hash mismatch: file={hit_hash}, "
                    f"expected={cfg_hash} — using existing file)")
            else:
                self._log_msg(f"  [Cache] {name}")
            with open(hit, 'rb') as fh:
                X = pickle.load(fh)
            return X

        # --- On-the-fly extraction ---
        self._log_msg(f"  [Extract] {name} — no cache, extracting...")
        try:
            # Late import: PixelPaws_GUI is fully initialised by the time
            # this method is called, so circular-import risk is zero.
            from PixelPaws_GUI import PixelPaws_ExtractFeatures  # noqa: PLC0415

            X = PixelPaws_ExtractFeatures(
                pose_data_file   = session.get('dlc') or session.get('pose_path'),
                video_file_path  = session.get('video') or session.get('video_path'),
                bp_pixbrt_list   = cfg.get('bp_pixbrt_list') or [],
                square_size      = cfg.get('square_size') or [40],
                pix_threshold    = cfg.get('pix_threshold', 0.3),
                bp_include_list  = cfg.get('bp_include_list'),
                config_yaml_path = cfg.get('dlc_config') or None,
                include_optical_flow = cfg.get('include_optical_flow', False),
                bp_optflow_list  = cfg.get('bp_optflow_list') or None,
            )
            X = X.reset_index(drop=True)

            # Save to cache so subsequent runs are instant
            os.makedirs(cache_dir, exist_ok=True)
            out = os.path.join(cache_dir, f"{name}_features_{cfg_hash}.pkl")
            with open(out, 'wb') as fh:
                pickle.dump(X, fh)
            self._log_msg(f"  [Cache] Saved → {out}")
            return X

        except Exception as exc:
            self._log_msg(f"  [Extract] FAILED for {name}: {exc}")
            return None

    def _select_feature_cols(self, X: pd.DataFrame, enabled_cats: set) -> list:
        """Return column names whose category is in *enabled_cats*."""
        return [c for c in X.columns if _categorize_col(c) in enabled_cats]

    # ======================================================================
    # B-SOiD feature helpers
    # ======================================================================

    def _detect_session_fps(self, session: dict, fallback: int) -> float:
        """Auto-detect the video fps for *session*; return *fallback* on failure."""
        vpath = session.get('video') or session.get('video_path', '')
        if vpath and os.path.isfile(vpath):
            try:
                import cv2
                cap = cv2.VideoCapture(vpath)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    return fps
            except Exception:
                pass
        return float(fallback)

    def _apply_temporal_window(self, X: pd.DataFrame,
                                window_frames: int,
                                smooth_frames: int):
        """
        Aggregate X into non-overlapping windows.

        Returns (X_windowed, window_start_indices).
        window_start_indices[i] = index into the original (NaN-dropped) X of
        the first frame of window i — used to map windowed rows back to video
        frame numbers.
        """
        if window_frames <= 1:
            return X.reset_index(drop=True), np.arange(len(X))

        n_win = len(X) // window_frames
        if n_win == 0:
            return X.reset_index(drop=True), np.arange(len(X))

        X_trunc = X.iloc[:n_win * window_frames].copy()
        win_idx  = np.repeat(np.arange(n_win), window_frames)
        agg = {c: ('sum' if _categorize_col(c) in ('velocities', 'accel_jerk')
                   else 'mean')
               for c in X.columns}
        X_trunc['_w'] = win_idx
        X_agg = X_trunc.groupby('_w').agg(agg).reset_index(drop=True)

        if smooth_frames > 1:
            X_agg = (X_agg
                     .rolling(window=smooth_frames, min_periods=1, center=True)
                     .mean()
                     .reset_index(drop=True))

        win_starts = np.arange(n_win) * window_frames   # indices into original X
        return X_agg, win_starts

    def _smooth_labels(self, labels: np.ndarray, min_frames: int) -> np.ndarray:
        """Replace short isolated cluster runs with the surrounding label."""
        if min_frames <= 0:
            return labels
        smoothed = labels.copy()
        n = len(smoothed)
        i = 0
        while i < n:
            j = i + 1
            while j < n and smoothed[j] == smoothed[i]:
                j += 1
            if (j - i) < min_frames and smoothed[i] != -1:
                prev_lbl = smoothed[i - 1] if i > 0 else -1
                next_lbl = smoothed[j]     if j < n else -1
                fill = prev_lbl if prev_lbl == next_lbl else \
                       (prev_lbl if prev_lbl != -1 else next_lbl)
                smoothed[i:j] = fill
            i = j
        return smoothed

    # ======================================================================
    # Model persistence
    # ======================================================================

    def _model_dir(self, folder: str, run_name: Optional[str] = None) -> str:
        rn = run_name or self._run_name_var.get()
        return os.path.join(folder, 'unsupervised', rn)

    def _model_path(self, folder: str, run_name: Optional[str] = None) -> str:
        return os.path.join(self._model_dir(folder, run_name), 'model.pkl')

    def _try_load_model_bundle(self, folder: str,
                               run_name: Optional[str] = None) -> Optional[dict]:
        path = self._model_path(folder, run_name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'rb') as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def _save_model_bundle(self, bundle: dict, folder: str,
                           run_name: Optional[str] = None):
        out = self._model_path(folder, run_name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as fh:
            pickle.dump(bundle, fh)

    # ======================================================================
    # CSV export helper
    # ======================================================================

    def _export_session_csv(self, session_name: str, labels: np.ndarray,
                            out_dir: str) -> str:
        """
        Write two files:
            <session_name>_clusters.csv    — binary columns (one per cluster + noise)
            <session_name>_cluster_ids.csv — single integer cluster_id column

        Returns the path to the binary-column CSV.
        """
        os.makedirs(out_dir, exist_ok=True)
        n_frames    = len(labels)
        cluster_ids = sorted(lbl for lbl in set(labels) if lbl >= 0)

        # Binary-column CSV
        rows = []
        for i, lbl in enumerate(labels):
            row = {'frame': i}
            for cid in cluster_ids:
                row[f'cluster_{cid}'] = int(lbl == cid)
            row['noise'] = int(lbl == -1)
            rows.append(row)
        bin_path = os.path.join(out_dir, f"{session_name}_clusters.csv")
        pd.DataFrame(rows).to_csv(bin_path, index=False)

        # Raw-id CSV
        raw_path = os.path.join(out_dir, f"{session_name}_cluster_ids.csv")
        pd.DataFrame({'frame': range(n_frames),
                      'cluster_id': labels.astype(int)}).to_csv(raw_path, index=False)

        return bin_path

    # ======================================================================
    # Capture UI settings (call on main thread before spawning worker)
    # ======================================================================

    def _capture_settings(self) -> dict:
        try:
            pca_min_var = float(self._pca_min_var_var.get())
        except ValueError:
            pca_min_var = 0.0
        return {
            'enabled_cats':        {k for k, v in self._cat_vars.items() if v.get()},
            'umap_n_neighbors':    self._umap_neighbors_var.get(),
            'umap_auto_neighbors': self._umap_auto_neighbors.get(),
            'umap_min_dist':       float(self._umap_min_dist_var.get()),
            'max_fit_frames':      self._max_fit_frames_var.get(),
            'hdb_min_cluster':     self._hdb_min_cluster_var.get(),
            'hdb_min_samples':     self._hdb_min_samples_var.get(),
            'target_fps':          self._target_fps_var.get(),
            'fallback_fps':        self._fallback_fps_var.get(),
            'smooth_ms':           self._smooth_ms_var.get(),
            'min_bout_ms':         self._min_bout_ms_var.get(),
            'pca_min_var':         pca_min_var,
            'run_name':            self._run_name_var.get(),
            'selected_sessions':   self._selected_names(),
            'project_folder':      self.app.current_project_folder.get(),
        }

    # ======================================================================
    # Fit model
    # ======================================================================

    def _start_fit(self):
        if self._fit_thread and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "An operation is already running.\n"
                                           "Click Cancel to stop it first.")
            return

        # Prompt for a run name before launching the background thread
        default_name = self._run_name_var.get() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_name = sd.askstring(
            "Name this run",
            "Enter a name for this unsupervised run:",
            initialvalue=default_name,
            parent=self,
        )
        if not run_name:
            return  # user cancelled
        run_name = run_name.strip().replace(" ", "_")
        self._run_name_var.set(run_name)

        # Warn if the output directory already exists
        folder = self.app.current_project_folder.get()
        out_dir = self._model_dir(folder, run_name)
        if os.path.isdir(out_dir):
            if not messagebox.askyesno(
                "Overwrite?",
                f"Run '{run_name}' already exists at:\n{out_dir}\n\nOverwrite it?",
                parent=self,
            ):
                return

        settings = self._capture_settings()
        self._cancel_flag.clear()
        self._clear_log()
        self._progress.start(10)
        self._fit_thread = threading.Thread(
            target=self._fit_model_thread, args=(settings,), daemon=True)
        self._fit_thread.start()

    def _fit_model_thread(self, settings: dict):
        try:
            self._fit_model(settings)
        except Exception:
            self._log_msg(f"[ERROR]\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, self._progress.stop)

    def _fit_model(self, settings: dict):
        folder   = settings['project_folder']
        selected = settings['selected_sessions']
        run_name = settings['run_name']

        if not folder or not os.path.isdir(folder):
            self._log_msg("[Fit] No project folder — aborting.")
            return
        if not selected:
            self._log_msg("[Fit] No sessions selected — aborting.")
            return

        cfg       = self._load_feature_cfg(folder)
        cache_dir = os.path.join(folder, 'features')
        out_dir   = self._model_dir(folder, run_name)

        target_fps   = settings['target_fps']
        fallback_fps = settings['fallback_fps']
        smooth_ms    = settings['smooth_ms']
        min_bout_ms  = settings['min_bout_ms']

        # ── 1. Load features ─────────────────────────────────────────────
        self._log_msg(f"[Fit] Loading features for {len(selected)} session(s)…")
        session_blocks        = []   # [(name, DataFrame)]
        session_row_map       = {}   # {name: (start, end)}
        session_frame_indices = {}   # {name: np.array of original video frame numbers}
        session_video_paths   = {}   # {name: video file path}
        total_rows = 0

        for name in selected:
            if self._cancel_flag.is_set():
                self._log_msg("[Fit] Cancelled.")
                return
            session = next((s for s in self._sessions
                            if s['session_name'] == name), None)
            if session is None:
                self._log_msg(f"  [Skip] {name} — not in scan results")
                continue

            X_full = self._load_features_for_session(session, cfg, cache_dir)
            if X_full is None:
                self._log_msg(f"  [Skip] {name} — could not load features")
                continue

            # Capture original frame indices BEFORE NaN-drop
            nan_mask = X_full.isna().any(axis=1)
            kept_frames = np.where(~nan_mask.values)[0]
            if nan_mask.any():
                X_full = X_full[~nan_mask].reset_index(drop=True)

            # ── Feature A: temporal windowing (per session) ───────────────
            source_fps = self._detect_session_fps(session, fallback_fps)
            if target_fps > 0:
                window_frames = max(1, round(source_fps / target_fps))
                smooth_frames = max(1, round(smooth_ms * target_fps / 1000)) \
                                if smooth_ms > 0 else 1
            else:
                window_frames = 1
                smooth_frames = 1

            if window_frames > 1:
                X_win, win_starts = self._apply_temporal_window(
                    X_full, window_frames, smooth_frames)
                kept_frames_win = kept_frames[win_starts]
                n = len(X_win)
                self._log_msg(
                    f"  ✓ {name}: {len(X_full)} frames → "
                    f"{n} windows ({window_frames} frames/window, "
                    f"src {source_fps:.1f} fps)")
            else:
                X_win = X_full
                kept_frames_win = kept_frames
                n = len(X_win)
                self._log_msg(f"  ✓ {name}: {n} frames")

            session_row_map[name]       = (total_rows, total_rows + n)
            session_frame_indices[name] = kept_frames_win
            session_video_paths[name]   = (session.get('video') or
                                           session.get('video_path') or '')
            total_rows += n
            session_blocks.append((name, X_win))

        if not session_blocks:
            self._log_msg("[Fit] No frames loaded — aborting.")
            return

        # ── 2. Pool and select feature columns ───────────────────────────
        X_pool    = pd.concat([df for _, df in session_blocks],
                              axis=0, ignore_index=True)
        feat_cols = self._select_feature_cols(X_pool, settings['enabled_cats'])
        if not feat_cols:
            self._log_msg("[Fit] No feature columns match the enabled categories — "
                          "aborting.")
            return

        X = X_pool[feat_cols].values.astype(np.float32)
        self._log_msg(f"[Fit] Matrix: {X.shape[0]} frames × {X.shape[1]} features")

        # ── 3. Normalise ─────────────────────────────────────────────────
        self._log_msg("[Fit] Normalising with StandardScaler…")
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self._cancel_flag.is_set():
            self._log_msg("[Fit] Cancelled.")
            return

        # ── 3b. Subsample for UMAP fit (B-SOiD strategy) ─────────────────
        max_fit = settings.get('max_fit_frames', 0)
        n_total = len(X_scaled)
        if max_fit > 0 and n_total > max_fit:
            rng        = np.random.RandomState(42)
            fit_idx    = rng.choice(n_total, max_fit, replace=False)
            X_fit      = X_scaled[fit_idx]
            subsampled = True
            self._log_msg(
                f"[Fit] Subsampling {n_total:,} → {max_fit:,} frames for UMAP fit")
        else:
            X_fit      = X_scaled
            fit_idx    = None
            subsampled = False

        # ── Feature B: PCA before UMAP ───────────────────────────────────
        pca_min_var = settings.get('pca_min_var', 0.0)
        if pca_min_var > 0.0:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_min_var, svd_solver='full')
            X_pca_fit = pca.fit_transform(X_fit)
            X_pca_all = pca.transform(X_scaled) if subsampled else X_pca_fit
            self._log_msg(
                f"[Fit] PCA: {X.shape[1]} → {pca.n_components_} components "
                f"({pca_min_var:.0%} variance threshold)")
        else:
            pca = None
            X_pca_fit = X_fit
            X_pca_all = X_scaled

        # ── 4. UMAP ──────────────────────────────────────────────────────
        if settings.get('umap_auto_neighbors', False):
            n_neighbors = max(2, int(round(np.sqrt(len(X_pca_fit)))))
            self._log_msg(f"[Fit] Auto n_neighbors = sqrt({len(X_pca_fit):,}) = {n_neighbors}")
        else:
            n_neighbors = settings['umap_n_neighbors']
        min_dist    = settings['umap_min_dist']
        self._log_msg(f"[Fit] UMAP(n_neighbors={n_neighbors}, min_dist={min_dist})…")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            verbose=False,
        )
        reducer.fit(X_pca_fit)
        embedding_fit = reducer.embedding_           # subsample embedding (n_fit × 2)

        if subsampled:
            self._log_msg(
                f"[Fit] UMAP done  →  transforming all {n_total:,} frames…")
            embedding = reducer.transform(X_pca_all)  # full embedding (n_total × 2)
        else:
            embedding = embedding_fit
        self._log_msg(f"[Fit] UMAP done  →  {embedding.shape}")

        if self._cancel_flag.is_set():
            self._log_msg("[Fit] Cancelled.")
            return

        # ── 5. HDBSCAN ───────────────────────────────────────────────────
        min_cluster = settings['hdb_min_cluster']
        min_samples = settings['hdb_min_samples']
        self._log_msg(
            f"[Fit] HDBSCAN(min_cluster_size={min_cluster}, "
            f"min_samples={min_samples})…")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min_samples,
            prediction_data=True,   # required for approximate_predict later
        )
        clusterer.fit(embedding_fit)   # cluster on subsample (or full if no subsampling)

        if subsampled:
            self._log_msg(
                f"[Fit] HDBSCAN done on subsample — assigning all {n_total:,} frames…")
            labels, _ = hdbscan.approximate_predict(clusterer, embedding)
        else:
            labels = clusterer.labels_

        # ── Feature C: minimum bout smoothing ────────────────────────────
        if min_bout_ms > 0:
            if target_fps > 0:
                min_frames = max(1, round(min_bout_ms * target_fps / 1000))
            else:
                # Without windowing use a representative fps; apply per-session
                # For the pooled array we use the mean of session fps values,
                # but since we need per-session we smooth each segment separately.
                min_frames = 0  # handled per-segment below
            if min_frames > 0:
                self._log_msg(
                    f"[Fit] Smoothing labels (min {min_frames} frames = "
                    f"{min_bout_ms} ms)")
                # Smooth each session segment independently
                smoothed = labels.copy()
                for name, (start, end) in session_row_map.items():
                    smoothed[start:end] = self._smooth_labels(
                        labels[start:end], min_frames)
                labels = smoothed
            else:
                # Per-session fps-based smoothing (target_fps == 0)
                smoothed = labels.copy()
                for name, (start, end) in session_row_map.items():
                    sess_obj = next((s for s in self._sessions
                                     if s['session_name'] == name), None)
                    src_fps = self._detect_session_fps(
                        sess_obj, fallback_fps) if sess_obj else float(fallback_fps)
                    mf = max(1, round(min_bout_ms * src_fps / 1000))
                    smoothed[start:end] = self._smooth_labels(
                        labels[start:end], mf)
                labels = smoothed
                self._log_msg(
                    f"[Fit] Smoothing labels (min bout = {min_bout_ms} ms, "
                    f"per-session fps)")

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = (labels == -1).mean()
        self._log_msg(
            f"[Fit] HDBSCAN done  →  {n_clusters} cluster(s), "
            f"{noise_frac:.1%} noise")

        # ── 6. Save model bundle ─────────────────────────────────────────
        temporal_params = {
            'target_fps':   target_fps,
            'fallback_fps': fallback_fps,
            'smooth_ms':    smooth_ms,
            'min_bout_ms':  min_bout_ms,
        }
        bundle = {
            'umap':             reducer,
            'hdbscan':          clusterer,
            'scaler':           scaler,
            'pca':              pca,
            'feature_cols':     feat_cols,
            'sessions':         session_row_map,
            'embedding':        embedding,
            'cluster_labels':   labels,
            'frame_indices':    session_frame_indices,
            'video_paths':      session_video_paths,
            'temporal_params':  temporal_params,
            'cfg':              cfg,
        }
        self._save_model_bundle(bundle, folder, run_name)
        self._log_msg(f"[Fit] Model saved → {self._model_path(folder, run_name)}")

        # ── 7. Export per-session CSVs ───────────────────────────────────
        for name, (start, end) in session_row_map.items():
            lbls     = labels[start:end]
            csv_path = self._export_session_csv(name, lbls, out_dir)
            n_c      = len(set(lbls)) - (1 if -1 in lbls else 0)
            self._log_msg(f"  CSV  {n_c} cluster(s) → {os.path.basename(csv_path)}")

        # ── 8. Update in-memory state and refresh UI ─────────────────────
        self._embedding       = embedding
        self._labels          = labels
        self._session_row_map = session_row_map
        self._model_bundle    = bundle
        self._frame_indices   = session_frame_indices
        self._video_paths     = session_video_paths

        def _refresh():
            self._update_treeview_status(session_row_map)
            self._open_viz_window()
            self._draw_scatter(embedding, labels)
            self._log_msg("[Fit] Done ✓")

        self.app.root.after(0, _refresh)

    # ======================================================================
    # Assign new sessions
    # ======================================================================

    def _start_assign(self):
        if self._fit_thread and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "An operation is already running.")
            return

        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project", "Select a project folder first.")
            return

        run_name = self._run_name_var.get()
        bundle   = self._try_load_model_bundle(folder, run_name)
        if bundle is None:
            messagebox.showwarning("No model",
                                   f"No model found for run '{run_name}'.\n"
                                   "Run 'Fit model' first.")
            return

        settings = self._capture_settings()
        self._cancel_flag.clear()
        self._progress.start(10)
        self._fit_thread = threading.Thread(
            target=self._assign_thread, args=(settings, bundle), daemon=True)
        self._fit_thread.start()

    def _assign_thread(self, settings: dict, bundle: dict):
        try:
            self._do_assign(settings, bundle)
        except Exception:
            self._log_msg(f"[ERROR]\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, self._progress.stop)

    def _do_assign(self, settings: dict, bundle: dict):
        folder   = settings['project_folder']
        run_name = settings['run_name']

        reducer   = bundle['umap']
        clusterer = bundle['hdbscan']
        scaler    = bundle['scaler']
        pca       = bundle.get('pca')
        feat_cols = bundle['feature_cols']

        # Replay temporal params from the bundle (ignore current UI settings)
        tp           = bundle.get('temporal_params', {})
        target_fps   = tp.get('target_fps',   settings.get('target_fps',   0))
        fallback_fps = tp.get('fallback_fps',  settings.get('fallback_fps', 60))
        smooth_ms    = tp.get('smooth_ms',     settings.get('smooth_ms',    0))
        min_bout_ms  = settings.get('min_bout_ms', 0)

        # Sessions already in the model
        in_model_map: dict = dict(bundle.get('sessions', {}))

        # Which selected sessions are NOT yet in the model?
        new_names = [n for n in settings['selected_sessions']
                     if n not in in_model_map]

        if not new_names:
            self._log_msg("[Assign] No new sessions — all selected sessions are "
                          "already in the model.")
            return

        cfg       = self._load_feature_cfg(folder)
        cache_dir = os.path.join(folder, 'features')
        out_dir   = self._model_dir(folder, run_name)

        self._log_msg(f"[Assign] Processing {len(new_names)} new session(s)…")

        # Starting row index for new sessions (after the existing embedding)
        existing_emb = bundle.get('embedding')
        base_offset  = len(existing_emb) if existing_emb is not None else 0
        running      = base_offset

        new_embeddings    = []
        new_label_lists   = []
        new_session_map   = {}
        new_frame_indices = {}
        new_video_paths   = {}

        for name in new_names:
            if self._cancel_flag.is_set():
                self._log_msg("[Assign] Cancelled.")
                return

            session = next((s for s in self._sessions
                            if s['session_name'] == name), None)
            if session is None:
                self._log_msg(f"  [Skip] {name} — not in scan results")
                continue

            X_full = self._load_features_for_session(session, cfg, cache_dir)
            if X_full is None:
                continue

            # Drop NaN rows
            nan_mask = X_full.isna().any(axis=1)
            kept_frames = np.where(~nan_mask.values)[0]
            if nan_mask.any():
                X_full = X_full[~nan_mask].reset_index(drop=True)

            # Check all required feature columns are present
            missing = [c for c in feat_cols if c not in X_full.columns]
            if missing:
                self._log_msg(f"  [Skip] {name} — {len(missing)} model feature "
                              f"columns missing (different feature config?)")
                continue

            # Replay temporal windowing
            source_fps = self._detect_session_fps(session, fallback_fps)
            if target_fps > 0:
                window_frames = max(1, round(source_fps / target_fps))
                smooth_frames = max(1, round(smooth_ms * target_fps / 1000)) \
                                if smooth_ms > 0 else 1
            else:
                window_frames = 1
                smooth_frames = 1

            if window_frames > 1:
                X_win, win_starts = self._apply_temporal_window(
                    X_full, window_frames, smooth_frames)
                kept_frames_win = kept_frames[win_starts]
            else:
                X_win = X_full
                kept_frames_win = kept_frames

            X        = X_win[feat_cols].values.astype(np.float32)
            X_scaled = scaler.transform(X)

            # Replay PCA
            X_pca = pca.transform(X_scaled) if pca is not None else X_scaled

            self._log_msg(f"  [Transform] {name}: {len(X)} frames → UMAP…")
            emb_new = reducer.transform(X_pca)

            self._log_msg(f"  [Predict] {name}: HDBSCAN approximate_predict…")
            lbls, _ = hdbscan.approximate_predict(clusterer, emb_new)

            # Replay bout smoothing
            if min_bout_ms > 0:
                if target_fps > 0:
                    mf = max(1, round(min_bout_ms * target_fps / 1000))
                else:
                    mf = max(1, round(min_bout_ms * source_fps / 1000))
                lbls = self._smooth_labels(lbls, mf)

            n = len(lbls)
            new_session_map[name]   = (running, running + n)
            new_frame_indices[name] = kept_frames_win
            new_video_paths[name]   = (session.get('video') or
                                       session.get('video_path') or '')
            running += n

            new_embeddings.append(emb_new)
            new_label_lists.append(lbls)

            csv_path = self._export_session_csv(name, lbls, out_dir)
            n_c = len(set(lbls)) - (1 if -1 in lbls else 0)
            self._log_msg(f"  ✓ {name}: {n_c} cluster(s) → "
                          f"{os.path.basename(csv_path)}")

        if not new_embeddings:
            self._log_msg("[Assign] No sessions were successfully processed.")
            return

        # Merge new sessions into bundle
        in_model_map.update(new_session_map)

        new_emb  = np.vstack(new_embeddings)
        new_lbls = np.concatenate(new_label_lists)

        if existing_emb is not None:
            all_emb  = np.vstack([existing_emb,   new_emb])
            all_lbls = np.concatenate([bundle.get('cluster_labels',
                                                   np.array([])), new_lbls])
        else:
            all_emb  = new_emb
            all_lbls = new_lbls

        # Merge frame indices and video paths
        merged_fi = dict(bundle.get('frame_indices', {}))
        merged_fi.update(new_frame_indices)
        merged_vp = dict(bundle.get('video_paths', {}))
        merged_vp.update(new_video_paths)

        bundle['sessions']       = in_model_map
        bundle['embedding']      = all_emb
        bundle['cluster_labels'] = all_lbls
        bundle['frame_indices']  = merged_fi
        bundle['video_paths']    = merged_vp
        self._save_model_bundle(bundle, folder, run_name)
        self._log_msg(f"[Assign] Model bundle updated → "
                      f"{self._model_path(folder, run_name)}")

        # Update in-memory state
        self._embedding       = all_emb
        self._labels          = all_lbls
        self._session_row_map = in_model_map
        self._model_bundle    = bundle
        self._frame_indices   = merged_fi
        self._video_paths     = merged_vp

        def _refresh():
            self._update_treeview_status(in_model_map)
            self._open_viz_window()
            self._draw_scatter(all_emb, all_lbls)
            self._log_msg("[Assign] Done ✓")

        self.app.root.after(0, _refresh)

    # ======================================================================
    # Export labels (standalone)
    # ======================================================================

    def _export_labels(self):
        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project", "Select a project folder first.")
            return

        run_name = self._run_name_var.get()
        out_dir  = self._model_dir(folder, run_name)

        # If we have in-memory labels, use them
        if self._labels is not None and self._session_row_map:
            for name, (start, end) in self._session_row_map.items():
                lbls = self._labels[start:end]
                csv_path = self._export_session_csv(name, lbls, out_dir)
                self._log_msg(f"  CSV → {csv_path}")
            self._log_msg(f"[Export] Done ✓  ({out_dir})")
            messagebox.showinfo("Export complete",
                                f"Label CSVs written to:\n{out_dir}")
            return

        # Check if CSVs already exist on disk
        if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, '*_clusters.csv')):
            messagebox.showinfo(
                "Already exported",
                f"Cluster CSVs are already on disk from the last Fit/Assign run.\n\n"
                f"Location:\n{out_dir}")
        else:
            messagebox.showwarning(
                "No data",
                "No in-memory labels found.  Run 'Fit model' or 'Assign new' first.")

    # ======================================================================
    # Cancel
    # ======================================================================

    def _cancel(self):
        self._cancel_flag.set()
        self._progress.stop()
        self._log_msg("[Cancel] Cancellation requested.")

    # ======================================================================
    # Export figures
    # ======================================================================

    def _export_figures(self):
        if self._embedding is None or self._labels is None:
            messagebox.showwarning("No data", "Run Fit model first.")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            title="Save UMAP figures (high quality, fixed size)",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"),
                       ("PDF document", "*.pdf"),
                       ("SVG vector", "*.svg")],
            initialfile="umap_clusters",
        )
        if not path:
            return
        try:
            self._save_hq_figure(path)
            self._log_msg(f"[Export] HQ figure saved → {path}")
            messagebox.showinfo("Saved", f"Figure saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _save_hq_figure(self, path: str, dpi: int = 300):
        """Render a fresh fixed-size figure independent of the window and save it.

        Scatter is always 12" × 8".  The bar-chart width scales with cluster
        count so bars stay readable even with 80+ clusters.
        """
        embedding = self._embedding
        labels    = self._labels
        sel       = self._selected_cluster

        unique_labels = sorted(set(labels))
        cluster_ids   = [lbl for lbl in unique_labels if lbl >= 0]
        n_clusters    = len(cluster_ids)
        cmap          = plt.get_cmap('turbo' if n_clusters > 20 else 'tab20',
                                     max(n_clusters, 1))

        scatter_w = 12.0
        bar_w     = max(4.0, n_clusters * 0.18)   # ~0.18" per bar, min 4"
        fig_h     = 8.0

        fig, (ax_s, ax_b) = plt.subplots(
            1, 2,
            figsize=(scatter_w + bar_w, fig_h),
            gridspec_kw={'width_ratios': [scatter_w, bar_w]},
            constrained_layout=True)

        # Scatter — no display sub-sampling; use rasterized for file size
        for lbl in unique_labels:
            mask = labels == lbl
            if lbl == -1:
                kw = dict(c='lightgrey', s=2, alpha=0.3,
                          label='noise', rasterized=True)
            elif lbl == sel:
                kw = dict(c='red', s=8, alpha=0.9, zorder=5,
                          label=f'C{lbl} (selected)', rasterized=True)
            else:
                col_idx = cluster_ids.index(lbl) / max(n_clusters - 1, 1)
                col = cmap(col_idx)
                alpha = 0.2 if sel is not None else 0.5
                kw = dict(c=[col], s=3, alpha=alpha,
                          label=f'C{lbl}', rasterized=True)
            ax_s.scatter(embedding[mask, 0], embedding[mask, 1], **kw)

        noise_pct = (labels == -1).mean() * 100
        ax_s.set_title(f"UMAP  ({n_clusters} cluster(s), {noise_pct:.1f}% noise)",
                       fontsize=14)
        ax_s.set_xlabel("UMAP 1", fontsize=12)
        ax_s.set_ylabel("UMAP 2", fontsize=12)
        if cluster_ids:
            leg_handles = [
                mlines.Line2D([], [], marker='o', linestyle='none',
                              color=cmap(i / max(n_clusters - 1, 1)),
                              markersize=5, label=f'C{lbl}')
                for i, lbl in enumerate(cluster_ids)
            ]
            ncol = max(1, min(10, n_clusters // 8 + 1))
            ax_s.legend(handles=leg_handles, ncol=ncol,
                        fontsize=max(4, min(7, int(80 / max(n_clusters, 1)) + 4)),
                        loc='best', framealpha=0.7,
                        handletextpad=0.3, columnspacing=0.8)

        # Bar chart
        if cluster_ids:
            counts   = [(lbl, int((labels == lbl).sum())) for lbl in cluster_ids]
            bar_lbls, bar_counts = zip(*counts)
            bar_cols = [cmap(cluster_ids.index(lbl) / max(n_clusters - 1, 1))
                        for lbl in bar_lbls]
            ax_b.bar([f'C{lbl}' for lbl in bar_lbls], bar_counts, color=bar_cols)
            ax_b.set_ylabel("Frames", fontsize=12)
            ax_b.set_title("Cluster sizes", fontsize=14)
            # Scale tick font down for many clusters so labels don't overlap
            tick_fs = max(4, min(9, int(120 / max(n_clusters, 1))))
            ax_b.tick_params(axis='x', rotation=90, labelsize=tick_fs)
            ax_b.set_xlabel(f"noise: {noise_pct:.1f}%", fontsize=10)

        ax_s.spines['top'].set_visible(False)
        ax_s.spines['right'].set_visible(False)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)

        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    # ======================================================================
    # Load model from file
    # ======================================================================

    def _load_model_from_file(self):
        from tkinter.filedialog import askopenfilename
        path = askopenfilename(
            title="Select model .pkl file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, 'rb') as fh:
                bundle = pickle.load(fh)
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load bundle:\n{exc}")
            return

        embedding = bundle.get('embedding')
        labels    = bundle.get('cluster_labels')
        row_map   = bundle.get('sessions')
        if embedding is None or labels is None or row_map is None:
            messagebox.showerror("Load error",
                                 "Bundle is missing embedding / labels / sessions.")
            return

        self._model_bundle    = bundle
        self._embedding       = embedding
        self._labels          = labels
        self._session_row_map = row_map
        self._frame_indices   = bundle.get('frame_indices', {})
        self._video_paths     = bundle.get('video_paths',   {})

        self._open_viz_window()
        self._draw_scatter(embedding, labels)
        self._update_treeview_status(row_map)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self._log_msg(
            f"[Load] {os.path.basename(path)}  "
            f"→  {len(row_map)} session(s), {n_clusters} cluster(s)")

    # ======================================================================
    # Treeview update
    # ======================================================================

    def _update_treeview_status(self, session_row_map: dict):
        """Mark sessions in *session_row_map* as 'in model' in the treeview."""
        for iid in self._tree.get_children():
            if iid in session_row_map:
                self._tree.item(iid,
                                tags=('in_model',),
                                values=('—', 'in model'))

    # ======================================================================
    # Scatter plot
    # ======================================================================

    def _draw_scatter(self, embedding: np.ndarray, labels: np.ndarray,
                      selected_cluster: Optional[int] = None):
        """Render the UMAP scatter and update the Tkinter legend."""
        if not MATPLOTLIB_AVAILABLE or self._fig is None:
            return

        ax_s = self._ax_scatter
        ax_s.clear()

        unique_labels = sorted(set(labels))
        cluster_ids   = [lbl for lbl in unique_labels if lbl >= 0]
        n_clusters    = len(cluster_ids)
        cmap          = plt.get_cmap('turbo' if n_clusters > 20 else 'tab20',
                                     max(n_clusters, 1))

        # Display-subsampling for scatter speed (clustering unchanged)
        MAX_DISPLAY = 150_000
        if len(embedding) > MAX_DISPLAY:
            rng = np.random.default_rng(0)
            disp_idx = rng.choice(len(embedding), MAX_DISPLAY, replace=False)
            disp_idx.sort()
            embedding_disp = embedding[disp_idx]
            labels_disp    = labels[disp_idx]
        else:
            embedding_disp = embedding
            labels_disp    = labels

        for lbl in unique_labels:
            mask = labels_disp == lbl
            if lbl == -1:
                kw = dict(c='lightgrey', s=1, alpha=0.3, label='noise',
                          rasterized=True)
            elif lbl == selected_cluster:
                kw = dict(c='red', s=6, alpha=0.9, zorder=5,
                          label=f'C{lbl} (selected)', rasterized=True)
            else:
                col_idx = cluster_ids.index(lbl) / max(n_clusters - 1, 1)
                col = cmap(col_idx)
                alpha = 0.2 if selected_cluster is not None else 0.5
                kw = dict(c=[col], s=2, alpha=alpha, label=f'C{lbl}',
                          rasterized=True)
            ax_s.scatter(embedding_disp[mask, 0], embedding_disp[mask, 1], **kw)

        noise_pct = (labels == -1).mean() * 100
        ax_s.set_title(f"UMAP  ({n_clusters} cluster(s), {noise_pct:.1f}% noise)")
        ax_s.set_xlabel("UMAP 1")
        ax_s.set_ylabel("UMAP 2")

        pass  # constrained_layout handles this
        try:
            self._canvas.draw()
        except Exception:
            pass

        self._update_legend(cluster_ids, cmap, n_clusters, selected_cluster)

    def _on_scatter_click(self, event):
        """Highlight the cluster whose centroid is closest to the click point."""
        if event.inaxes != self._ax_scatter:
            return
        if self._embedding is None or self._labels is None:
            return

        cx, cy = event.xdata, event.ydata
        cluster_ids = [lbl for lbl in set(self._labels) if lbl >= 0]
        if not cluster_ids:
            return

        best_lbl, best_dist = -1, float('inf')
        for lbl in cluster_ids:
            mask = self._labels == lbl
            cx_c = self._embedding[mask, 0].mean()
            cy_c = self._embedding[mask, 1].mean()
            d = (cx - cx_c) ** 2 + (cy - cy_c) ** 2
            if d < best_dist:
                best_dist = d
                best_lbl  = lbl

        self._selected_cluster = best_lbl
        n_frames = int((self._labels == best_lbl).sum())
        self._log_msg(f"[Click] Cluster {best_lbl} — {n_frames} frames")
        self._draw_scatter(self._embedding, self._labels,
                           selected_cluster=best_lbl)
        self.app.root.after(0, lambda: self._show_cluster_inspector(best_lbl))

    def _select_adjacent_cluster(self, offset: int):
        """Select the next/previous cluster by sorted ID order (arrow-key nav)."""
        if self._labels is None:
            return
        cluster_ids = sorted(lbl for lbl in set(self._labels) if lbl >= 0)
        if not cluster_ids:
            return
        if self._selected_cluster is None or self._selected_cluster not in cluster_ids:
            new_lbl = cluster_ids[0]
        else:
            idx = cluster_ids.index(self._selected_cluster)
            new_lbl = cluster_ids[(idx + offset) % len(cluster_ids)]
        self._selected_cluster = new_lbl
        n_frames = int((self._labels == new_lbl).sum())
        self._log_msg(f"[Key ◀▶] Cluster {new_lbl} — {n_frames} frames")
        self._draw_scatter(self._embedding, self._labels, selected_cluster=new_lbl)
        self.app.root.after(0, lambda: self._show_cluster_inspector(new_lbl))

    def _show_cluster_inspector(self, cluster_id: int):
        """Populate the inspector for *cluster_id* (main thread)."""
        if self._labels is None or self._embedding is None:
            return
        # Open inspector window if not already open
        if self._inspect_win is None or not self._inspect_win.winfo_exists():
            self._open_inspect_win()
        # Update title
        n = int((self._labels == cluster_id).sum())
        if self._inspect_title_var is not None:
            self._inspect_title_var.set(f"Cluster {cluster_id}  —  {n:,} frames")
        self._draw_ethogram(cluster_id)
        self._draw_bar_chart()
        t = threading.Thread(
            target=self._load_frames_thread,
            args=(cluster_id,),
            daemon=True)
        t.start()

    def _draw_ethogram(self, cluster_id: int):
        """Draw a per-session timeline strip showing cluster presence."""
        if not MATPLOTLIB_AVAILABLE or self._etho_ax is None:
            return
        ax = self._etho_ax
        ax.clear()

        if not self._session_row_map:
            try:
                self._etho_canvas.draw()
            except Exception:
                pass
            return

        n_clusters  = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        cluster_ids = sorted(lbl for lbl in set(self._labels) if lbl >= 0)

        sorted_sessions = sorted(self._session_row_map.items())
        for row_i, (name, (start, end)) in enumerate(sorted_sessions):
            seg_labels = self._labels[start:end]
            n = len(seg_labels)
            if n == 0:
                continue
            # Draw a binary strip: 1 where this cluster appears, 0 elsewhere
            strip = np.where(seg_labels == cluster_id, 1, 0).reshape(1, -1)
            ax.imshow(strip, aspect='auto',
                      extent=[0, 1, row_i, row_i + 1],
                      cmap='RdGy_r', vmin=0, vmax=1,
                      interpolation='nearest')

        n_sess = len(self._session_row_map)
        ax.set_yticks(np.arange(n_sess) + 0.5)
        ax.set_yticklabels(
            [n for n, _ in sorted(self._session_row_map.items())], fontsize=7)
        ax.set_xlabel("Time (normalised, labeled frames only)")
        ax.set_title(f"Cluster {cluster_id} presence per session", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_sess)
        pass  # constrained_layout handles this
        try:
            self._etho_canvas.draw()
        except Exception:
            pass

    def _load_frames_thread(self, cluster_id: int):
        """Background thread: load up to 9 sample frames from the cluster."""
        try:
            import cv2
        except ImportError:
            def _no_cv2():
                if self._thumb_label is not None:
                    self._thumb_label.config(
                        text="cv2 not available — install opencv-python")
            self.app.root.after(0, _no_cv2)
            return

        if self._labels is None:
            return

        indices = np.where(self._labels == cluster_id)[0]
        if len(indices) == 0:
            return

        rng    = np.random.default_rng(cluster_id)
        sample = rng.choice(indices, min(9, len(indices)), replace=False)
        sample.sort()

        frame_indices = getattr(self, '_frame_indices', {})
        video_paths   = getattr(self, '_video_paths',   {})

        frames_out = []
        for global_idx in sample:
            matched = None
            for name, (start, end) in self._session_row_map.items():
                if start <= global_idx < end:
                    matched = (name, int(global_idx - start))
                    break
            if matched is None:
                frames_out.append(None)
                continue

            name, local_row = matched
            fi     = frame_indices.get(name)
            vpath  = video_paths.get(name, '')
            vidframe = int(fi[local_row]) if (fi is not None and
                                              local_row < len(fi)) else local_row

            if not vpath or not os.path.isfile(vpath):
                frames_out.append(None)
                continue

            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vidframe)
            ok, frame = cap.read()
            cap.release()
            frames_out.append(frame if ok else None)

        self.app.root.after(
            0, lambda f=frames_out, cid=cluster_id:
            self._display_thumbnails(f, cid))

    def _display_thumbnails(self, frames: list, cluster_id: int):
        """Render loaded frames into the 3×3 thumbnail canvas grid (main thread)."""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            if self._thumb_label is not None:
                self._thumb_label.config(
                    text="Pillow not available — pip install Pillow")
            return

        import cv2
        TW, TH = 140, 105
        for i, cv_canvas in enumerate(self._thumb_canvases):
            cv_canvas.delete('all')
            if i >= len(frames) or frames[i] is None:
                cv_canvas.create_text(TW // 2, TH // 2, text="N/A",
                                      fill='#888', font=('Consolas', 8))
                continue
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame_rgb).resize((TW, TH), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            cv_canvas.create_image(0, 0, anchor='nw', image=photo)
            cv_canvas.image = photo   # prevent GC

        n = int((self._labels == cluster_id).sum())
        if self._thumb_label is not None:
            self._thumb_label.config(
                text=f"Cluster {cluster_id}  —  {n:,} frames total")

    # ======================================================================
    # Feature D — Cluster merging
    # ======================================================================

    def _merge_clusters(self):
        if self._labels is None:
            messagebox.showwarning("No data",
                                   "Run Fit model or Load model first.")
            return
        dlg = tk.Toplevel(self.app.root)
        dlg.title("Merge Clusters")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Merge FROM (IDs, space/comma separated):"
                  ).grid(row=0, column=0, sticky='w', padx=8, pady=4)
        from_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=from_var, width=20
                  ).grid(row=0, column=1, padx=8, pady=4)
        ttk.Label(dlg, text="Merge INTO (single ID):"
                  ).grid(row=1, column=0, sticky='w', padx=8, pady=4)
        into_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=into_var, width=10
                  ).grid(row=1, column=1, padx=8, pady=4)

        def _do():
            try:
                from_ids = [int(x.strip())
                            for x in from_var.get().replace(',', ' ').split()
                            if x.strip()]
                into_id  = int(into_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "Enter integer cluster IDs.", parent=dlg)
                return
            mask = np.isin(self._labels, from_ids)
            n_merged = int(mask.sum())
            self._labels[mask] = into_id
            self._draw_scatter(self._embedding, self._labels)
            self._log_msg(
                f"[Merge] {from_ids} → {into_id}  ({n_merged} frames reassigned)")
            dlg.destroy()

        ttk.Button(dlg, text="Merge", command=_do
                   ).grid(row=2, column=0, columnspan=2, pady=8)

    # ======================================================================
    # Feature E — Video bout preview
    # ======================================================================

    def _find_bouts(self, cluster_id: int, max_bouts: int = 20,
                    min_frames: int = 1) -> list:
        """Return list of (session_name, global_start, global_end) for each
        contiguous run of frames labelled *cluster_id*."""
        bouts = []
        for name, (seg_start, seg_end) in sorted(self._session_row_map.items()):
            seg = self._labels[seg_start:seg_end]
            i = 0
            while i < len(seg) and len(bouts) < max_bouts:
                if seg[i] == cluster_id:
                    j = i + 1
                    while j < len(seg) and seg[j] == cluster_id:
                        j += 1
                    if (j - i) >= min_frames:
                        bouts.append((name, seg_start + i, seg_start + j))
                    i = j
                else:
                    i += 1
        return bouts

    def _open_bout_player(self, cluster_id):
        if cluster_id is None:
            messagebox.showwarning("No cluster",
                                   "Click a cluster in the scatter first.")
            return
        if self._labels is None or not self._session_row_map:
            messagebox.showwarning("No data",
                                   "Run Fit model or Load model first.")
            return

        tp           = self._model_bundle.get('temporal_params', {}) if self._model_bundle else {}
        target_fps   = tp.get('target_fps',  0)
        fallback_fps = tp.get('fallback_fps', self._fallback_fps_var.get())
        effective_fps = target_fps if target_fps > 0 else fallback_fps

        def _ms_to_frames(ms):
            return max(1, round(ms * effective_fps / 1000)) if effective_fps > 0 else max(1, ms)

        saved_min_ms = tp.get('min_bout_ms', self._min_bout_ms_var.get())
        min_ms_var   = tk.IntVar(value=max(1, saved_min_ms))
        bouts = self._find_bouts(cluster_id, max_bouts=500,
                                 min_frames=_ms_to_frames(min_ms_var.get()))
        if not bouts:
            messagebox.showinfo("No bouts",
                                f"Cluster {cluster_id} has no contiguous bouts "
                                f"(≥ {min_ms_var.get()} ms).")
            return

        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError as exc:
            messagebox.showerror("Missing dependency", str(exc))
            return

        win = tk.Toplevel(self.app.root)
        win.title(f"Bout Player — Cluster {cluster_id}")
        win.resizable(True, True)

        CW, CH = 640, 480
        canvas = tk.Canvas(win, width=CW, height=CH, bg='black')
        canvas.grid(row=0, column=0, columnspan=4, padx=4, pady=4)

        info_lbl = ttk.Label(win, text="", anchor='center')
        info_lbl.grid(row=1, column=0, columnspan=4, pady=(0, 2))

        pos_scale = ttk.Scale(win, from_=0, to=1, orient='horizontal',
                              length=CW)
        pos_scale.grid(row=2, column=0, columnspan=4, padx=4)

        btn_prev  = ttk.Button(win, text="◀ Prev")
        btn_play  = ttk.Button(win, text="▶ Play")
        btn_next  = ttk.Button(win, text="Next ▶")
        btn_prev.grid(row=3, column=0, padx=4, pady=4)
        btn_play.grid(row=3, column=1, padx=4, pady=4)
        btn_next.grid(row=3, column=2, padx=4, pady=4)

        speed_frame = ttk.Frame(win)
        speed_frame.grid(row=3, column=3, padx=4, pady=4)
        ttk.Label(speed_frame, text="Speed:").pack(side='left')
        speed_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(speed_frame, from_=0.25, to=4.0, increment=0.25,
                    textvariable=speed_var, width=5).pack(side='left')

        min_frame_ctrl = ttk.Frame(win)
        min_frame_ctrl.grid(row=3, column=4, padx=4, pady=4)
        ttk.Label(min_frame_ctrl, text="Min bout (ms):").pack(side='left')
        ttk.Spinbox(min_frame_ctrl, from_=1, to=99999, textvariable=min_ms_var,
                    width=6).pack(side='left')

        fr_ind = ttk.Label(min_frame_ctrl, foreground='grey', width=10)
        fr_ind.pack(side='left', padx=(4, 0))

        def _refresh_fr_ind(*_):
            try:
                ms = int(min_ms_var.get())
            except (ValueError, tk.TclError):
                fr_ind.config(text='')
                return
            n = _ms_to_frames(ms)
            if effective_fps > 0:
                fr_ind.config(text=f'= {n} fr @ {effective_fps:.0f} fps')
            else:
                fr_ind.config(text=f'= {n} fr')

        min_ms_var.trace_add('write', _refresh_fr_ind)
        _refresh_fr_ind()

        # Playback state stored on the window object
        state = {
            'bouts':          bouts,
            'bout_idx':       0,
            'frame_pos':      0,
            'playing':        False,
            'cap':            None,
            'after_id':       None,
            '_photo':         None,
            '_scale_update':  False,   # guard: pos_scale.set() must not re-enter _on_scale
        }

        def _resolve_video(bout_name):
            return self._video_paths.get(bout_name, '')

        def _bout_length(bout_idx):
            _, g_start, g_end = state['bouts'][bout_idx]
            return g_end - g_start

        def _show_frame(bout_idx, frame_pos):
            name, g_start, g_end = state['bouts'][bout_idx]
            seg_start, _ = self._session_row_map[name]
            local_idx = g_start - seg_start + frame_pos   # index into session's kept_frames

            fi    = self._frame_indices.get(name)
            vpath = _resolve_video(name)
            vidframe = int(fi[local_idx]) \
                if fi is not None and local_idx < len(fi) else local_idx

            n_frames = g_end - g_start
            n_bouts  = len(state['bouts'])
            dur_str = f"{n_frames / effective_fps:.2f} s" if effective_fps > 0 else f"{n_frames} fr"
            info_lbl.config(
                text=(f"Bout {bout_idx + 1} of {n_bouts}  —  {dur_str}  —  "
                      f"frame {frame_pos + 1}/{n_frames}  [{name}]"))

            # Update scale without triggering _on_scale (would recurse)
            state['_scale_update'] = True
            pos_scale.config(to=max(1, n_frames - 1))
            pos_scale.set(frame_pos)
            state['_scale_update'] = False

            if not vpath or not os.path.isfile(vpath):
                canvas.delete('all')
                canvas.create_text(CW // 2, CH // 2, text="Video not found",
                                   fill='white', font=('Arial', 14))
                return

            if state['cap'] is None or state.get('_cap_path') != vpath:
                if state['cap']:
                    state['cap'].release()
                state['cap'] = cv2.VideoCapture(vpath)
                state['_cap_path'] = vpath

            state['cap'].set(cv2.CAP_PROP_POS_FRAMES, vidframe)
            ok, frame = state['cap'].read()
            if not ok:
                canvas.delete('all')
                canvas.create_text(CW // 2, CH // 2, text="Read error",
                                   fill='red', font=('Arial', 14))
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame_rgb).resize((CW, CH), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=photo)
            canvas.image = photo
            state['_photo'] = photo

        def _step():
            if not state['playing']:
                return
            n_frames = _bout_length(state['bout_idx'])
            state['frame_pos'] += 1
            if state['frame_pos'] >= n_frames:
                state['frame_pos'] = 0
                state['playing'] = False
                btn_play.config(text='▶ Play')
                _show_frame(state['bout_idx'], 0)
                return
            _show_frame(state['bout_idx'], state['frame_pos'])
            try:
                spd   = float(speed_var.get())
                fps   = 30.0
                delay = max(1, int(1000 / (fps * spd)))
            except Exception:
                delay = 33
            state['after_id'] = win.after(delay, _step)

        def _play_pause():
            state['playing'] = not state['playing']
            btn_play.config(text='⏸ Pause' if state['playing'] else '▶ Play')
            if state['playing']:
                _step()

        def _prev():
            state['playing'] = False
            btn_play.config(text='▶ Play')
            state['bout_idx'] = (state['bout_idx'] - 1) % len(state['bouts'])
            state['frame_pos'] = 0
            _show_frame(state['bout_idx'], 0)

        def _next():
            state['playing'] = False
            btn_play.config(text='▶ Play')
            state['bout_idx'] = (state['bout_idx'] + 1) % len(state['bouts'])
            state['frame_pos'] = 0
            _show_frame(state['bout_idx'], 0)

        def _on_scale(val):
            # Guard: ignore programmatic updates from _show_frame
            if state['_scale_update']:
                return
            if not state['playing']:
                state['frame_pos'] = int(float(val))
                _show_frame(state['bout_idx'], state['frame_pos'])

        def _on_close():
            state['playing'] = False
            if state['cap']:
                state['cap'].release()
            win.destroy()

        def _reload_bouts():
            new_bouts = self._find_bouts(cluster_id, max_bouts=500,
                                         min_frames=_ms_to_frames(min_ms_var.get()))
            if not new_bouts:
                return
            state['playing'] = False
            btn_play.config(text='▶ Play')
            state['bouts']     = new_bouts
            state['bout_idx']  = 0
            state['frame_pos'] = 0
            _show_frame(0, 0)

        min_ms_var.trace_add('write', lambda *_: _reload_bouts())

        btn_prev.config(command=_prev)
        btn_play.config(command=_play_pause)
        btn_next.config(command=_next)
        pos_scale.config(command=_on_scale)
        win.protocol("WM_DELETE_WINDOW", _on_close)

        _show_frame(0, 0)

    # ======================================================================
    # Feature F — Train RF classifier from clusters
    # ======================================================================

    def _start_train_rf(self):
        if self._labels is None or self._model_bundle is None:
            messagebox.showwarning("No data",
                                   "Run Fit model or Load model first.")
            return
        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project",
                                   "Select a project folder first.")
            return
        settings = self._capture_settings()
        self._cancel_flag.clear()
        self._progress.start(10)
        t = threading.Thread(
            target=self._train_rf_thread,
            args=(settings, self._model_bundle),
            daemon=True)
        t.start()

    def _train_rf_thread(self, settings: dict, bundle: dict):
        try:
            self._do_train_rf(settings, bundle)
        except Exception:
            self._log_msg(f"[RF] ERROR\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, self._progress.stop)

    def _do_train_rf(self, settings: dict, bundle: dict):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        folder   = settings['project_folder']
        run_name = settings['run_name']
        cfg      = bundle.get('cfg') or self._load_feature_cfg(folder)
        cache_dir = os.path.join(folder, 'features')

        scaler    = bundle['scaler']
        pca       = bundle.get('pca')
        feat_cols = bundle['feature_cols']
        tp        = bundle.get('temporal_params', {})
        target_fps   = tp.get('target_fps',   0)
        fallback_fps = tp.get('fallback_fps',  60)
        smooth_ms    = tp.get('smooth_ms',     0)

        labels_all = self._labels
        row_map    = self._session_row_map

        self._log_msg(f"[RF] Loading features for {len(row_map)} session(s)…")

        X_parts, y_parts = [], []
        for name, (seg_start, seg_end) in sorted(row_map.items()):
            session = next((s for s in self._sessions
                            if s['session_name'] == name), None)
            if session is None:
                self._log_msg(f"  [Skip] {name} — not in scan results")
                continue

            X_full = self._load_features_for_session(session, cfg, cache_dir)
            if X_full is None:
                continue

            nan_mask = X_full.isna().any(axis=1)
            if nan_mask.any():
                X_full = X_full[~nan_mask].reset_index(drop=True)

            source_fps = self._detect_session_fps(session, fallback_fps)
            if target_fps > 0:
                window_frames = max(1, round(source_fps / target_fps))
                smooth_frames = max(1, round(smooth_ms * target_fps / 1000)) \
                                if smooth_ms > 0 else 1
            else:
                window_frames = 1
                smooth_frames = 1

            if window_frames > 1:
                X_win, _ = self._apply_temporal_window(
                    X_full, window_frames, smooth_frames)
            else:
                X_win = X_full

            missing = [c for c in feat_cols if c not in X_win.columns]
            if missing:
                self._log_msg(f"  [Skip] {name} — {len(missing)} feature columns "
                              f"missing")
                continue

            X_arr    = X_win[feat_cols].values.astype(np.float32)
            X_scaled = scaler.transform(X_arr)
            X_pca    = pca.transform(X_scaled) if pca is not None else X_scaled

            seg_labels = labels_all[seg_start:seg_end]
            n = min(len(X_pca), len(seg_labels))
            X_parts.append(X_pca[:n])
            y_parts.append(seg_labels[:n])

        if not X_parts:
            self._log_msg("[RF] No data loaded — aborting.")
            return

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)

        # Exclude noise frames
        mask  = y_all != -1
        X_all = X_all[mask]
        y_all = y_all[mask]

        if len(X_all) == 0:
            self._log_msg("[RF] All frames are noise — aborting.")
            return

        self._log_msg(
            f"[RF] Training RF on {len(X_all):,} frames, "
            f"{len(set(y_all))} classes…")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            if len(set(y_all)) > 1 else None)

        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)

        train_acc = accuracy_score(y_tr, rf.predict(X_tr))
        test_acc  = accuracy_score(y_te, rf.predict(X_te))
        self._log_msg(
            f"[RF] Train acc: {train_acc:.3f}  |  Test acc: {test_acc:.3f}")

        # Save classifier bundle
        clf_dir = os.path.join(folder, 'classifiers')
        os.makedirs(clf_dir, exist_ok=True)
        out_path = os.path.join(clf_dir, f"{run_name}_clusters.pkl")

        bundle_out = {
            'clf_model':        rf,
            'feature_names':    feat_cols,
            'Behavior_type':    f'clusters_{run_name}',
            'classifier_type':  'multiclass_rf',
            'cluster_ids':      sorted(set(y_all.tolist())),
            'best_thresh':      0.5,
            'bp_include_list':  cfg.get('bp_include_list'),
            'bp_pixbrt_list':   cfg.get('bp_pixbrt_list', []),
            'square_size':      cfg.get('square_size', [40]),
            'pix_threshold':    cfg.get('pix_threshold', 0.3),
            'min_bout':         1,
            'min_after_bout':   0,
            'max_gap':          0,
            'scaler':           scaler,
            'pca':              pca,
            'temporal_params':  tp,
        }
        with open(out_path, 'wb') as fh:
            pickle.dump(bundle_out, fh)
        self._log_msg(f"[RF] Classifier saved → {out_path}")

        msg = (f"Random Forest trained successfully.\n\n"
               f"Train accuracy: {train_acc:.1%}\n"
               f"Test accuracy:  {test_acc:.1%}\n\n"
               f"Saved to:\n{out_path}")
        self.app.root.after(
            0, lambda: messagebox.showinfo("RF Classifier saved", msg))
