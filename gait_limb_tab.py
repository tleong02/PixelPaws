"""
Gait & Limb Use Analysis Tab  (gait_limb_tab.py)
==================================================
Metrics are derived from DLC pose coordinates and optional video brightness —
no force plate or pressure mat required.

Contact detection methods:
  height   — paw height below threshold (original method)
  speed    — paw speed below threshold (Kumar Lab, Cell Reports 2022)
  combined — both height AND speed must agree

Metrics computed per session (and per time bin when requested):
  contact_pct_{HL/HR/FL/FR}  — % frames paw in stance (ground contact)
  WBI_hind                    — HL / (HL+HR) * 100  (50 = symmetric)
  SI_hind                     — (HL-HR) / (HL+HR) * 100  (0 = symmetric)
  WBI_fore / SI_fore          — same for fore paws (if configured)
  brightness_{HL/HR/FL/FR}    — mean ROI brightness during contact frames
  brightness_ratio_HL_HR      — brightness_HL / brightness_HR
  hind_fore_ratio             — mean hind contact% / mean fore contact%

Gait timing metrics (per paw):
  stance_dur, swing_dur, stride_dur, duty_cycle, cadence, n_strides
  swing_speed (stride_len / swing_dur), stride_cv (coefficient of variation)

Locomotion metrics:
  body_speed_mean, body_speed_loco

Gait spatial metrics:
  stride_len (per paw), step_len_hind/fore, step_width_hind/fore

Interlimb coordination (if all 4 paws mapped):
  phase_HL_HR, phase_diagonal, phase_FL_FR, phase_HL_FL, phase_HR_FR, phase_HL_FR
  regularity_index, print_position_L/R
  support_0paw_pct .. support_4paw_pct

Gait symmetry:
  stance_SI_hind, stride_len_SI_hind

Paw contour area (optional, requires video):
  paw_area, paw_spread, contact_intensity, paw_area_ratio
  paw_width, paw_solidity, paw_aspect_ratio, paw_circularity
"""

import os
import re
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import median_filter as _median_filter
    _SCIPY_NDIMAGE_OK = True
except ImportError:
    _SCIPY_NDIMAGE_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from scipy import stats as _sp_stats
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False

from pose_features import PoseFeatureExtractor
from brightness_features import PixelBrightnessExtractorOptimized

from ui_utils import (ToolTip as _ToolTip,
                      _bind_tight_layout_on_resize, _draw_canvas_fit)

try:
    from evaluation_tab import find_session_triplets
except ImportError:
    def find_session_triplets(folder, **kw):
        return []

try:
    from PixelPaws_GUI import extract_subject_id_from_filename as _extract_sid
except ImportError:
    _extract_sid = None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level graph helpers
# ─────────────────────────────────────────────────────────────────────────────

_SIG_STYLES = {
    'asterisk': ('*', '**', '***'),
    'hash':     ('#', '##', '###'),
    'dagger':   ('\u2020', '\u2020\u2020', '\u2020\u2020\u2020'),
    'letters':  ('a', 'b', 'c'),
}


def _p_label(p: float, style: str = 'asterisk') -> str:
    syms = _SIG_STYLES.get(style, _SIG_STYLES['asterisk'])
    if p < 0.001:
        return syms[2]
    if p < 0.01:
        return syms[1]
    if p < 0.05:
        return syms[0]
    return ''


def _draw_bracket(ax, x1: int, x2: int, y: float, label: str):
    ax.plot([x1, x1, x2, x2], [y * 0.97, y, y, y * 0.97],
            color='black', linewidth=1)
    ax.text((x1 + x2) / 2, y * 1.02, label,
            ha='center', va='bottom', fontsize=12, fontweight='bold')


def _autofit_treeview_window(win, tree):
    """Resize *win* so all columns of *tree* are visible, including bottom widgets.

    Must be called AFTER all widgets (buttons, notes) are packed.
    """
    win.update_idletasks()
    style_font = ttk.Style().lookup('Treeview', 'font') or 'TkDefaultFont'
    font = tkfont.nametofont(style_font)
    total_w = 0
    for col in tree['columns']:
        hdr_w = font.measure(tree.heading(col, 'text')) + 24
        max_cell = hdr_w
        for iid in tree.get_children():
            val = str(tree.set(iid, col))
            max_cell = max(max_cell, font.measure(val) + 24)
        tree.column(col, width=max_cell)
        total_w += max_cell
    total_w += 40  # scrollbar + frame padding
    # Let tkinter compute required height from ALL packed widgets
    win.update_idletasks()
    req_h = win.winfo_reqheight()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(total_w, int(sw * 0.9))
    h = min(max(req_h, 300), int(sh * 0.9))
    x, y = (sw - w) // 2, (sh - h) // 2
    win.geometry(f'{w}x{h}+{x}+{y}')


# ─────────────────────────────────────────────────────────────────────────────
# GaitLimbTab
# ─────────────────────────────────────────────────────────────────────────────

class GaitLimbTab(ttk.Frame):
    """Gait & Limb Use analysis tab."""

    ROLES = ('HL', 'HR', 'FL', 'FR')
    ROLE_LABELS = {
        'HL': 'Hind-Left paw:',
        'HR': 'Hind-Right paw:',
        'FL': 'Fore-Left paw:',
        'FR': 'Fore-Right paw:',
    }
    ROLE_DEFAULTS = {'HL': 'hlpaw', 'HR': 'hrpaw', 'FL': 'flpaw', 'FR': 'frpaw'}

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self.pack(fill='both', expand=True)

        self._sessions: list = []
        self._key_df: pd.DataFrame = None
        self._key_scan_paths: list = []
        self._summary_df: pd.DataFrame = None
        self._bins_df: pd.DataFrame = None
        self._enable_stats_var       = tk.BooleanVar(value=True)
        self._stats_test_var         = tk.StringVar(value='auto')
        self._stats_alpha_var        = tk.DoubleVar(value=0.05)
        self._timecourse_posthoc_var = tk.BooleanVar(value=False)
        self._stats_paradigm_var    = tk.StringVar(value='parametric')
        self._contact_method_var  = tk.StringVar(value='height')
        self._speed_thresh_var   = tk.StringVar(value='auto')
        self._median_filter_var  = tk.IntVar(value=50)
        self._min_bout_var       = tk.IntVar(value=30)
        self._likelihood_thresh_var = tk.DoubleVar(value=0.6)
        self._use_likelihood_var = tk.BooleanVar(value=False)
        self._loco_filter_var    = tk.BooleanVar(value=False)
        self._loco_thresh_var    = tk.DoubleVar(value=20.0)
        self._paw_contour_var    = tk.BooleanVar(value=True)
        self._contour_forelimbs_var = tk.BooleanVar(value=False)
        self._min_stance_ms_var  = tk.IntVar(value=100)
        self._fit_thread: threading.Thread = None
        self._cancel_flag = threading.Event()
        self._bodyparts: list = []
        self._last_graph_cfg = None
        self._session_intermediates = {}
        self._pawlike_thresholds = {'solidity': 1.00, 'aspect_ratio': 1.6, 'circularity': 0.10}

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════
    # UI construction
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(hdr, text="🐾  Gait & Limb Use Analysis",
                  font=('Arial', 13, 'bold')).pack(side='left')

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left  = ttk.Frame(paned, width=220)
        mid   = ttk.Frame(paned, width=260)
        right = ttk.Frame(paned, width=420)

        paned.add(left,  weight=1)
        paned.add(mid,   weight=1)
        paned.add(right, weight=2)

        self._build_sessions_panel(left)
        self._build_settings_panel(mid)
        self._build_results_panel(right)

    # ── Left: Sessions ──────────────────────────────────────────────────────

    def _build_sessions_panel(self, parent):
        self._override_folder_var = tk.StringVar(value='')

        lf = ttk.LabelFrame(parent, text="Sessions", padding=5)
        lf.pack(fill='both', expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill='x', pady=(0, 4))
        ttk.Button(btn_row, text="Scan",
                   command=self._scan_sessions).pack(side='left', padx=2)
        ttk.Button(btn_row, text="All",
                   command=self._select_all).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Clear",
                   command=self._clear_selection).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Browse…",
                   command=self._browse_sessions_folder).pack(side='left', padx=2)

        cols = ('name', 'subject', 'vid')
        self._sess_tree = ttk.Treeview(lf, columns=cols, show='headings',
                                       selectmode='extended', height=20)
        self._sess_tree.heading('name',    text='Session')
        self._sess_tree.heading('subject', text='Subject')
        self._sess_tree.heading('vid',     text='Video?')
        self._sess_tree.column('name',    width=120, stretch=True)
        self._sess_tree.column('subject', width=60,  stretch=False)
        self._sess_tree.column('vid',     width=45,  stretch=False)

        sb = ttk.Scrollbar(lf, orient='vertical', command=self._sess_tree.yview)
        self._sess_tree.config(yscrollcommand=sb.set)
        self._sess_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        self._sess_lbl = ttk.Label(parent, text='', foreground='grey')
        self._sess_lbl.pack(anchor='w', padx=6, pady=(2, 0))
        self._folder_lbl = ttk.Label(parent, text='', foreground='grey',
                                      wraplength=200, font=('Arial', 8))
        self._folder_lbl.pack(anchor='w', padx=6)

    # ── Middle: Settings ─────────────────────────────────────────────────────

    def _make_scrollable_tab(self, notebook, tab_label):
        """Create a scrollable frame inside a notebook tab, return the inner frame."""
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=tab_label)
        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(inner_id, width=e.width))
        return inner

    def _build_settings_panel(self, parent):
        # ── Run / Cancel / Progress (always visible at top) ───────────────
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill='x', padx=2, pady=(4, 2))

        self._run_btn = ttk.Button(run_frame, text="▶  Run Analysis",
                                   command=self._start_analysis)
        self._run_btn.pack(side='left', padx=2)
        self._cancel_btn = ttk.Button(run_frame, text="■  Cancel",
                                      command=self._cancel_analysis,
                                      state='disabled')
        self._cancel_btn.pack(side='left', padx=2)

        # ── Settings notebook ─────────────────────────────────────
        settings_nb = ttk.Notebook(parent)
        settings_nb.pack(fill='both', expand=True, padx=2, pady=(4, 2))

        # ── Tab 1: Setup (Key File + Paw Mapping) ────────────────────
        setup_inner = self._make_scrollable_tab(settings_nb, "Setup")

        kf_lf = ttk.LabelFrame(setup_inner, text="Key File", padding=5)
        kf_lf.pack(fill='x', pady=(0, 6), padx=2)

        kf_row = ttk.Frame(kf_lf)
        kf_row.pack(fill='x', pady=2)
        ttk.Label(kf_row, text="File:", width=7).pack(side='left')
        self._key_file_var = tk.StringVar()
        self._key_combo = ttk.Combobox(kf_row, textvariable=self._key_file_var,
                                       state='normal', width=22)
        self._key_combo.pack(side='left', padx=3)
        self._key_combo.bind('<<ComboboxSelected>>', self._on_key_combo_selected)
        ttk.Button(kf_row, text="Browse", width=7,
                   command=self._browse_key_file).pack(side='left')
        ttk.Button(kf_row, text="Generate…", width=9,
                   command=self._generate_key_file).pack(side='left', padx=(4, 0))

        pfx_row = ttk.Frame(kf_lf)
        pfx_row.pack(fill='x', pady=2)
        ttk.Label(pfx_row, text="Prefix:", width=7).pack(side='left')
        self._prefix_var = tk.StringVar()
        pfx_ent = ttk.Entry(pfx_row, textvariable=self._prefix_var, width=22)
        pfx_ent.pack(side='left', padx=3)
        self._tip(pfx_ent,
                  "Filename prefix to strip before extracting subject ID.\n"
                  "e.g. '260129_Formalin_' → next underscore-token = subject.")

        self._key_status_lbl = ttk.Label(kf_lf, text='No key file loaded',
                                         foreground='grey', wraplength=230,
                                         justify='left')
        self._key_status_lbl.pack(anchor='w', pady=(2, 0))

        pm_lf = ttk.LabelFrame(setup_inner, text="Paw Mapping", padding=5)
        pm_lf.pack(fill='x', pady=(0, 6), padx=2)

        self._role_vars   = {}
        self._role_combos = {}
        for i, role in enumerate(self.ROLES):
            ttk.Label(pm_lf, text=self.ROLE_LABELS[role]).grid(
                row=i, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=self.ROLE_DEFAULTS[role])
            self._role_vars[role] = var
            cb = ttk.Combobox(pm_lf, textvariable=var, width=14, state='normal')
            cb.grid(row=i, column=1, sticky='w', padx=4, pady=2)
            self._role_combos[role] = cb
            if role in ('FL', 'FR'):
                ttk.Label(pm_lf, text='(optional)',
                          foreground='grey').grid(row=i, column=2, sticky='w')

        self._use_fore_var = tk.BooleanVar(value=False)
        self._use_fore_chk = ttk.Checkbutton(
            pm_lf, text='Include fore paws',
            variable=self._use_fore_var,
            command=self._on_use_fore_changed)
        self._use_fore_chk.grid(
            row=len(self.ROLES), column=0, columnspan=3, sticky='w', pady=(4, 0))

        ttk.Button(pm_lf, text="Auto-detect from DLC",
                   command=self._autodetect_bodyparts).grid(
            row=len(self.ROLES) + 1, column=0, columnspan=3, sticky='ew', pady=(6, 2))

        # ── Tab 2: Parameters ─────────────────────────────────────
        params_inner = self._make_scrollable_tab(settings_nb, "Parameters")

        par_lf = ttk.LabelFrame(params_inner, text="Parameters", padding=5)
        par_lf.pack(fill='x', pady=(0, 6), padx=2)

        self._contact_thresh_var = tk.IntVar(value=15)
        self._height_window_var  = tk.IntVar(value=500)
        self._bin_seconds_var    = tk.IntVar(value=1)
        self._bin_unit_var       = tk.StringVar(value='minutes')
        self._fallback_fps_var   = tk.DoubleVar(value=60.0)
        self._use_brightness_var = tk.BooleanVar(value=True)
        self._brt_thresh_var     = tk.IntVar(value=0)
        self._brt_weight_var        = tk.DoubleVar(value=1.0)
        self._extraction_stride_var = tk.IntVar(value=1)
        self._roi_size_vars      = {
            'HL': tk.IntVar(value=20),
            'HR': tk.IntVar(value=20),
            'FL': tk.IntVar(value=15),
            'FR': tk.IntVar(value=15),
        }
        self._contour_roi_size_vars = {
            'HL': tk.IntVar(value=60),
            'HR': tk.IntVar(value=60),
            'FL': tk.IntVar(value=40),
            'FR': tk.IntVar(value=40),
        }

        spinrows = [
            ("Contact thresh (px):", self._contact_thresh_var, 0, 500,
             "Paw height (px) below which a frame is counted as ground contact.\n\n"
             "Height = rolling_max(paw_y, window=height_window) − current_paw_y.\n"
             "It measures how far the paw has risen above the estimated floor level.\n\n"
             "Typical: 5–30 px.  Too low → very few contact frames.  "
             "Too high → stance inflated by swing phase."),
            ("Height window (fr):", self._height_window_var, 1, 10000,
             "Rolling-max window for floor estimation. Larger = more stable."),
            ("Fallback fps:", self._fallback_fps_var, 1, 500,
             "Used when the video cannot be opened to read its actual fps."),
            ("Brightness threshold:", self._brt_thresh_var, 0, 255,
             "Pixel intensity cutoff for brightness extraction.\n"
             "0 = auto-detect (≈ mean frame brightness × 0.5).\n"
             "Use the Preview button to set this visually."),
            ("Extraction stride:", self._extraction_stride_var, 1, 16,
             "Read every Nth video frame during brightness extraction.\n"
             "1 = every frame (full accuracy).\n"
             "2 = every 2nd frame (~2× faster, minimal accuracy loss for 30-s bins).\n"
             "4 = every 4th frame (~4× faster, suitable for quick exploration runs)."),
        ]
        for r, (lbl_text, var, from_, to, tip) in enumerate(spinrows):
            lbl = ttk.Label(par_lf, text=lbl_text)
            lbl.grid(row=r, column=0, sticky='w', pady=2)
            sb = ttk.Spinbox(par_lf, from_=from_, to=to, textvariable=var, width=8)
            sb.grid(row=r, column=1, sticky='w', padx=4, pady=2)
            self._tip(lbl, tip)
            self._tip(sb, tip)

        brt_cb = ttk.Checkbutton(par_lf, text="Brightness-weighted (needs video)",
                                  variable=self._use_brightness_var)
        brt_cb.grid(row=len(spinrows), column=0, columnspan=2, sticky='w', pady=(4, 0))
        self._tip(brt_cb,
                  "Extract mean pixel brightness in each paw ROI during contact frames.\n"
                  "Requires a video file for each session. Skipped if video is missing.")

        ttk.Button(par_lf, text="Preview brightness…",
                   command=self._open_brightness_preview).grid(
            row=len(spinrows) + 1, column=0, columnspan=2, sticky='w', pady=(4, 0))

        ttk.Button(par_lf, text="Detect cached brightness…",
                   command=self._detect_brightness_caches).grid(
            row=len(spinrows) + 2, column=0, columnspan=2, sticky='w', pady=(2, 0))

        bw_row = ttk.Frame(par_lf)
        bw_row.grid(row=len(spinrows) + 3, column=0, columnspan=2, sticky='w', pady=2)
        bw_lbl = ttk.Label(bw_row, text="Brt contact weight:")
        bw_lbl.pack(side='left')
        bw_sb  = ttk.Spinbox(bw_row, from_=0.0, to=1.0, increment=0.05,
                              textvariable=self._brt_weight_var, width=6, format='%.2f')
        bw_sb.pack(side='left', padx=4)
        _bw_tip = ("Weight given to ROI brightness when determining contact frames.\n\n"
                   "0 = height only (default).  1 = brightness only.\n"
                   "0.3–0.5 recommended for glass-floor / transilluminated setups.\n"
                   "Requires 'Brightness-weighted (needs video)' to be enabled.")
        self._tip(bw_lbl, _bw_tip)
        self._tip(bw_sb,  _bw_tip)

        contour_cb = ttk.Checkbutton(par_lf, text="Compute paw contour area",
                                      variable=self._paw_contour_var)
        contour_cb.grid(row=len(spinrows) + 4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        self._tip(contour_cb,
                  "Detect paw outline within the contour ROI using Otsu thresholding.\n"
                  "Requires brightness enabled + video.\n\n"
                  "Output metrics per paw:\n"
                  "  paw_area — contour area in px² (larger = more paw contact)\n"
                  "  paw_spread — max(width, height) of contour bounding box (px)\n"
                  "  contact_intensity — mean pixel brightness within the contour shape")

        ttk.Button(par_lf, text="Preview contour…",
                   command=self._open_contour_preview).grid(
            row=len(spinrows) + 5, column=0, columnspan=2, sticky='w', pady=(2, 0))

        ttk.Button(par_lf, text="Detect cached contour…",
                   command=self._detect_contour_caches).grid(
            row=len(spinrows) + 6, column=0, columnspan=2, sticky='w', pady=(2, 0))

        ttk.Button(par_lf, text="Analyze both caches…",
                   command=self._detect_both_caches).grid(
            row=len(spinrows) + 7, column=0, columnspan=2, sticky='w', pady=(2, 0))

        contour_roi_lbl = ttk.Label(par_lf, text="Contour ROI half-size (px):")
        contour_roi_lbl.grid(row=len(spinrows) + 8, column=0, columnspan=2,
                             sticky='w', pady=(6, 0))
        _croi_tip = ("Half-width of the ROI crop used for paw contour detection.\n"
                     "Larger = captures more of the paw shape but may include background.\n"
                     "Set independently from brightness ROI.")
        self._tip(contour_roi_lbl, _croi_tip)

        croi_frame = ttk.Frame(par_lf)
        croi_frame.grid(row=len(spinrows) + 9, column=0, columnspan=2,
                        sticky='w', pady=(0, 2))
        for i, role in enumerate(self.ROLES):
            ttk.Label(croi_frame, text=f"{role}:").grid(
                row=i // 2, column=(i % 2) * 2, sticky='w', padx=(0, 2))
            sb = ttk.Spinbox(croi_frame, from_=5, to=200,
                             textvariable=self._contour_roi_size_vars[role], width=5)
            sb.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky='w', padx=(0, 8))
            self._tip(sb, _croi_tip)

        fore_cb = ttk.Checkbutton(par_lf, text="Include forelimbs in contour",
                                    variable=self._contour_forelimbs_var)
        fore_cb.grid(row=len(spinrows) + 10, column=0, columnspan=2, sticky='w', pady=(2, 0))
        self._tip(fore_cb,
                  "By default, paw contour analysis is limited to hind limbs (HL, HR).\n"
                  "Enable this to also extract contour metrics for forelimbs (FL, FR).")

        # Time Bins
        tb_lf = ttk.LabelFrame(params_inner, text="Time Bins", padding=5)
        tb_lf.pack(fill='x', pady=(0, 6), padx=2)

        tb_row = ttk.Frame(tb_lf)
        tb_row.pack(fill='x', pady=2)
        ttk.Label(tb_row, text="Bin size:", width=10).pack(side='left')
        _bin_spx = ttk.Spinbox(tb_row, from_=0, to=3600,
                                textvariable=self._bin_seconds_var, width=7)
        _bin_spx.pack(side='left', padx=4)

        unit_frame = ttk.Frame(tb_row)
        unit_frame.pack(side='left', padx=4)
        ttk.Radiobutton(unit_frame, text="minutes", variable=self._bin_unit_var,
                        value='minutes').pack(side='left')
        ttk.Radiobutton(unit_frame, text="seconds", variable=self._bin_unit_var,
                        value='seconds').pack(side='left', padx=(6, 0))

        ttk.Label(tb_lf,
                  text="Video divided into equal bins. 0 = full session only (no bins).",
                  font=('Arial', 8), foreground='gray').pack(anchor='w', pady=(2, 0))
        self._tip(_bin_spx,
                  "Number of minutes (or seconds) per time bin.\n"
                  "0 = output the full session as a single row only.")

        # DLC Crop Offset
        self._dlc_config_var = tk.StringVar()
        self._crop_x_var     = tk.IntVar(value=0)
        self._crop_y_var     = tk.IntVar(value=0)

        crop_lf = ttk.LabelFrame(params_inner, text="DLC Crop Offset", padding=5)
        crop_lf.pack(fill='x', pady=(0, 6), padx=2)
        self._tip(crop_lf,
                  "Apply when DLC tracking was done on a cropped video.\n"
                  "X/Y offset = top-left pixel of the crop region in the original video.")

        ttk.Label(crop_lf, text="config.yaml:").grid(
            row=0, column=0, sticky='w', pady=2)
        cfg_ent = ttk.Entry(crop_lf, textvariable=self._dlc_config_var, width=20)
        cfg_ent.grid(row=0, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(crop_lf, text="Browse",
                   command=self._browse_dlc_config).grid(
            row=0, column=2, sticky='w', pady=2)

        ttk.Label(crop_lf, text="X offset (px):").grid(
            row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(crop_lf, from_=0, to=4000, textvariable=self._crop_x_var,
                    width=8).grid(row=1, column=1, sticky='w', padx=4, pady=2)

        ttk.Label(crop_lf, text="Y offset (px):").grid(
            row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(crop_lf, from_=0, to=4000, textvariable=self._crop_y_var,
                    width=8).grid(row=2, column=1, sticky='w', padx=4, pady=2)

        ttk.Button(crop_lf, text="Detect from config.yaml",
                   command=self._detect_crop_from_config).grid(
            row=3, column=0, columnspan=3, sticky='ew', pady=(6, 2))

        crop_lf.columnconfigure(1, weight=1)

        # ── Tab 3: Detection ────────────────────────────────────
        detect_inner = self._make_scrollable_tab(settings_nb, "Detection")

        cd_lf = ttk.LabelFrame(detect_inner, text="Contact Detection", padding=5)
        cd_lf.pack(fill='x', pady=(0, 6), padx=2)

        ttk.Label(cd_lf, text="Method:").grid(row=0, column=0, sticky='w', pady=2)
        cd_rb_frame = ttk.Frame(cd_lf)
        cd_rb_frame.grid(row=0, column=1, sticky='w', padx=4, pady=2)
        for txt, val in [("Height", "height"), ("Speed", "speed"), ("Combined", "combined")]:
            ttk.Radiobutton(cd_rb_frame, text=txt, variable=self._contact_method_var,
                            value=val).pack(side='left', padx=(0, 6))
        self._tip(cd_lf,
                  "Height: paw height below threshold (original).\n"
                  "Speed: paw speed below threshold (Kumar Lab 2022).\n"
                  "Combined: both must agree (logical AND).")

        ttk.Label(cd_lf, text="Speed thresh (px/s):").grid(row=1, column=0, sticky='w', pady=2)
        speed_ent = ttk.Entry(cd_lf, textvariable=self._speed_thresh_var, width=10)
        speed_ent.grid(row=1, column=1, sticky='w', padx=4, pady=2)
        self._tip(speed_ent,
                  "'auto' = 20th percentile of paw speed.\n"
                  "Or enter a numeric value in px/s.")

        ttk.Label(cd_lf, text="Median filter (ms):").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(cd_lf, from_=0, to=500, textvariable=self._median_filter_var,
                    width=8).grid(row=2, column=1, sticky='w', padx=4, pady=2)

        ttk.Label(cd_lf, text="Min bout (ms):").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(cd_lf, from_=0, to=500, textvariable=self._min_bout_var,
                    width=8).grid(row=3, column=1, sticky='w', padx=4, pady=2)

        _msd_lbl = ttk.Label(cd_lf, text="Min stance dur (ms):")
        _msd_lbl.grid(row=4, column=0, sticky='w', pady=2)
        _msd_sb = ttk.Spinbox(cd_lf, from_=0, to=500,
                               textvariable=self._min_stance_ms_var, width=8)
        _msd_sb.grid(row=4, column=1, sticky='w', padx=4, pady=2)
        _msd_tip = ("Minimum stance bout duration for gait metrics.\n"
                     "Stance bouts shorter than this are discarded as\n"
                     "physiologically implausible (noise).\n"
                     "Default 100 ms. Set 0 to disable.")
        self._tip(_msd_lbl, _msd_tip)
        self._tip(_msd_sb, _msd_tip)

        # DLC Confidence Filter
        dlc_lf = ttk.LabelFrame(detect_inner, text="DLC Confidence Filter", padding=5)
        dlc_lf.pack(fill='x', pady=(0, 6), padx=2)
        ttk.Checkbutton(dlc_lf, text="Filter low-confidence frames",
                        variable=self._use_likelihood_var).pack(anchor='w')
        lk_row = ttk.Frame(dlc_lf)
        lk_row.pack(fill='x', pady=2)
        ttk.Label(lk_row, text="Likelihood threshold:").pack(side='left')
        ttk.Spinbox(lk_row, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._likelihood_thresh_var, width=6,
                    format='%.2f').pack(side='left', padx=4)
        self._tip(dlc_lf,
                  "Exclude frames where DLC confidence < threshold from gait metrics.\n"
                  "Default 0.6. Higher = stricter filtering.")

        # Locomotion Filter (with improved tooltip)
        loco_lf = ttk.LabelFrame(detect_inner, text="Locomotion Filter", padding=5)
        loco_lf.pack(fill='x', pady=(0, 6), padx=2)
        ttk.Checkbutton(loco_lf, text="Restrict gait metrics to locomotion epochs",
                        variable=self._loco_filter_var).pack(anchor='w')
        loco_row = ttk.Frame(loco_lf)
        loco_row.pack(fill='x', pady=2)
        ttk.Label(loco_row, text="Tailbase speed thresh (px/s):").pack(side='left')
        ttk.Spinbox(loco_row, from_=0.0, to=500.0, increment=0.5,
                    textvariable=self._loco_thresh_var, width=8,
                    format='%.1f').pack(side='left', padx=4)
        self._tip(loco_lf,
                  "Only compute gait metrics during epochs where tailbase speed\n"
                  "exceeds the threshold. Prevents nonsensical stride metrics\n"
                  "during stationary, grooming, or rearing episodes.\n\n"
                  "The right value depends on your camera resolution and arena size.\n"
                  "Typical ranges (assuming ~640 px across a 30 cm arena):\n"
                  "  5 px/s  ≈ 2 mm/s — very permissive, barely filters anything\n"
                  "  20 px/s ≈ 9 mm/s — excludes grooming/resting, keeps walking\n"
                  "  50 px/s ≈ 23 mm/s — only fast locomotion\n\n"
                  "Tip: run once, check the 'Time Moving %' metric — if it's >90%\n"
                  "the threshold is probably too low.")
        ttk.Button(loco_lf, text="Preview locomotion\u2026",
                   command=self._open_locomotion_preview).pack(anchor='w', pady=(4, 0))

        # (Statistical tests settings moved to Graph Settings dialog)

    # ── Locomotion preview ─────────────────────────────────────────────────

    def _open_locomotion_preview(self):
        if not _CV2_OK:
            messagebox.showerror("Missing dependency",
                                 "OpenCV (cv2) is required for the locomotion preview.",
                                 parent=self)
            return

        # --- find a usable session (needs video + DLC) ---
        selected = [self._sess_tree.item(i, 'values')[0]
                    for i in self._sess_tree.selection()]
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and s.get('video') and os.path.isfile(s['video'])
                      and s.get('dlc')   and os.path.isfile(s['dlc'])]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session that has both a video and DLC file.",
                parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # --- load DLC data with active paws + tailbase ---
        active_paws = {r: self._role_vars[r].get().strip()
                       for r in self.ROLES if self._role_vars[r].get().strip()}
        bp_list = list(set(active_paws.values()) | {'tailbase'})
        try:
            ext    = PoseFeatureExtractor(bp_list)
            dlc_df = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return

        # --- find body center (tailbase → hind-paw midpoint fallback) ---
        tb_x_col = None
        for candidate_name in ['tailbase', 'tail_base', 'tb']:
            tb_x_col = next((c for c in bp_xcord.columns
                             if candidate_name in c.lower()), None)
            if tb_x_col:
                tb_y_col = next((c for c in bp_ycord.columns
                                 if candidate_name in c.lower()), None)
                break
        if tb_x_col and tb_y_col:
            cx = bp_xcord[tb_x_col].values.astype(float)
            cy = bp_ycord[tb_y_col].values.astype(float)
        else:
            hl_bp = active_paws.get('HL', '')
            hr_bp = active_paws.get('HR', '')
            hl_x = next((c for c in bp_xcord.columns if hl_bp.lower() in c.lower()), None) if hl_bp else None
            hr_x = next((c for c in bp_xcord.columns if hr_bp.lower() in c.lower()), None) if hr_bp else None
            hl_y = next((c for c in bp_ycord.columns if hl_bp.lower() in c.lower()), None) if hl_bp else None
            hr_y = next((c for c in bp_ycord.columns if hr_bp.lower() in c.lower()), None) if hr_bp else None
            if hl_x and hr_x and hl_y and hr_y:
                cx = (bp_xcord[hl_x].values + bp_xcord[hr_x].values) / 2.0
                cy = (bp_ycord[hl_y].values + bp_ycord[hr_y].values) / 2.0
            else:
                messagebox.showerror(
                    "No body center",
                    "Could not find tailbase or hind-paw coordinates for body center.",
                    parent=self)
                return

        # --- open video, get fps ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                 f"Cannot open:\n{video_path}", parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Clamp coordinate arrays to video frame count
        cx = cx[:n_frames]
        cy = cy[:n_frames]

        # --- precompute speed arrays (done once) ---
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        frame_displacements = np.sqrt(dx**2 + dy**2)
        body_speed = frame_displacements * fps

        # --- build window ---
        win = tk.Toplevel(self)
        win.title(f"Locomotion Preview — {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.60)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        # Scrollable ctrl panel — pack FIRST so it gets priority for space
        ctrl_outer = ttk.Frame(win)
        ctrl_outer.pack(side='right', fill='both', padx=6, pady=6)
        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0, width=300)
        ctrl_sb_y = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_sb_x = ttk.Scrollbar(ctrl_outer, orient='horizontal', command=ctrl_canvas.xview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb_y.set, xscrollcommand=ctrl_sb_x.set)
        ctrl_sb_x.pack(side='bottom', fill='x')
        ctrl_sb_y.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_win_id = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(ctrl_win_id, width=e.width))

        # Video canvas — packed after ctrl so it fills remaining space
        canvas = tk.Canvas(win, bg='black', width=480, height=380)
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        # Frame slider
        ttk.Label(ctrl, text="Frame:").pack(anchor='w')
        frame_var = tk.IntVar(value=0)
        ttk.Spinbox(ctrl, from_=0, to=max(n_frames - 1, 0),
                     textvariable=frame_var, width=8).pack(anchor='w', pady=(0, 2))
        ttk.Scale(ctrl, from_=0, to=max(n_frames - 1, 0),
                  variable=frame_var, orient='vertical', length=180).pack(anchor='w', pady=(0, 8))

        # Locomotion threshold
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Loco threshold (px/s):").pack(anchor='w')
        local_thresh_var = tk.DoubleVar(value=self._loco_thresh_var.get())
        ttk.Spinbox(ctrl, from_=0.0, to=500.0, increment=0.5,
                     textvariable=local_thresh_var, width=8,
                     format='%.1f').pack(anchor='w', pady=(0, 2))
        ttk.Scale(ctrl, from_=0, to=200,
                  variable=local_thresh_var, orient='vertical', length=140).pack(anchor='w', pady=(0, 8))

        # Smoothing window
        ttk.Label(ctrl, text="Smoothing (frames):").pack(anchor='w')
        smooth_var = tk.IntVar(value=5)
        ttk.Spinbox(ctrl, from_=1, to=31, increment=2,
                     textvariable=smooth_var, width=8).pack(anchor='w', pady=(0, 2))
        ttk.Scale(ctrl, from_=1, to=31,
                  variable=smooth_var, orient='vertical', length=80).pack(anchor='w', pady=(0, 8))

        # Minimum bout duration filter
        ttk.Label(ctrl, text="Min bout (frames):").pack(anchor='w')
        min_bout_var = tk.IntVar(value=3)
        ttk.Spinbox(ctrl, from_=1, to=15, increment=1,
                     textvariable=min_bout_var, width=8).pack(anchor='w', pady=(0, 2))
        ttk.Scale(ctrl, from_=1, to=15,
                  variable=min_bout_var, orient='vertical', length=80).pack(anchor='w', pady=(0, 8))

        # Summary stats label
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Summary:", font=('Arial', 9, 'bold')).pack(anchor='w')
        summary_lbl = ttk.Label(ctrl, text="", wraplength=270, justify='left')
        summary_lbl.pack(anchor='w', pady=(2, 6))

        # Current frame readout
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Current frame:", font=('Arial', 9, 'bold')).pack(anchor='w')
        frame_speed_lbl = ttk.Label(ctrl, text="Speed: — px/s", wraplength=270)
        frame_speed_lbl.pack(anchor='w')
        frame_state_lbl = ttk.Label(ctrl, text="State: —", wraplength=270)
        frame_state_lbl.pack(anchor='w', pady=(0, 6))

        # Apply button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        def _apply():
            self._loco_thresh_var.set(local_thresh_var.get())
            self._log_ui(f"Applied loco threshold: {local_thresh_var.get():.1f} px/s")

        ttk.Button(ctrl, text="Apply to settings",
                   command=_apply).pack(fill='x', pady=(0, 2))

        # Playback controls
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Playback:", font=('Arial', 9, 'bold')).pack(anchor='w')

        playing = [False]
        play_after_id = [None]
        speed_var = tk.DoubleVar(value=1.0)

        play_btn = ttk.Button(ctrl, text="\u25b6 Play")
        play_btn.pack(fill='x', pady=(2, 4))

        spd_row = ttk.Frame(ctrl)
        spd_row.pack(fill='x', pady=(0, 4))
        ttk.Label(spd_row, text="Speed:").pack(side='left')
        for mult in (0.25, 0.5, 1.0, 2.0):
            ttk.Radiobutton(spd_row, text=f"{mult}x", variable=speed_var,
                            value=mult).pack(side='left', padx=2)

        # --- precompute loco_mask (recomputed on threshold/smoothing change) ---
        loco_state = {'mask': body_speed > local_thresh_var.get(),
                      'smoothed_speed': body_speed.copy()}

        def _update_summary(*_):
            try:
                thr = local_thresh_var.get()
                win_size = max(1, smooth_var.get())
                min_bout = max(1, min_bout_var.get())
            except (tk.TclError, ValueError):
                return
            # Smooth the speed signal with a uniform rolling average
            if win_size > 1:
                kernel = np.ones(win_size) / win_size
                smoothed_speed = np.convolve(body_speed, kernel, mode='same')
            else:
                smoothed_speed = body_speed
            mask = smoothed_speed > thr
            if min_bout > 1:
                mask = GaitLimbTab._debounce(mask, min_bout)
            loco_state['mask'] = mask
            loco_state['smoothed_speed'] = smoothed_speed
            n_moving = int(mask.sum())
            pct = 100.0 * n_moving / max(len(mask), 1)
            mean_spd = float(smoothed_speed.mean())
            max_spd = float(smoothed_speed.max())
            summary_lbl.config(
                text=f"Moving: {n_moving}/{len(mask)} ({pct:.1f}%)\n"
                     f"Mean speed: {mean_spd:.1f} px/s\n"
                     f"Max speed: {max_spd:.1f} px/s")

        _update_summary()

        # --- render function (debounced 50 ms) ---
        _after_id = [None]
        TRAIL_LEN = 30

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _do_render():
            try:
                fi = int(frame_var.get())
            except (tk.TclError, ValueError):
                fi = 0
            fi = max(0, min(fi, n_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return
            vis = frame.copy()
            fh, fw = vis.shape[:2]

            mask = loco_state['mask']
            is_moving = bool(mask[fi]) if fi < len(mask) else False
            smoothed = loco_state.get('smoothed_speed', body_speed)
            spd = float(smoothed[fi]) if fi < len(smoothed) else 0.0

            # --- tint: green if moving, red+dim if stationary ---
            overlay = vis.copy()
            if is_moving:
                overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(int) + 40, 0, 255).astype(np.uint8)
                cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)
            else:
                overlay[:, :, 2] = np.clip(overlay[:, :, 2].astype(int) + 40, 0, 255).astype(np.uint8)
                vis_dimmed = (overlay * 0.7).astype(np.uint8)
                cv2.addWeighted(vis_dimmed, 0.40, vis, 0.60, 0, vis)

            # --- body center dot ---
            bx = int(cx[fi]) if fi < len(cx) else 0
            by = int(cy[fi]) if fi < len(cy) else 0
            dot_color = (0, 255, 0) if is_moving else (0, 0, 255)
            cv2.circle(vis, (bx, by), 8, (255, 255, 255), 2)
            cv2.circle(vis, (bx, by), 8, dot_color, -1)

            # --- trajectory trail (last TRAIL_LEN frames) ---
            start_i = max(0, fi - TRAIL_LEN)
            for ti in range(start_i, fi):
                if ti + 1 >= len(cx):
                    break
                alpha = (ti - start_i + 1) / (fi - start_i + 1)
                seg_moving = bool(mask[ti]) if ti < len(mask) else False
                seg_color = (0, int(200 * alpha), 0) if seg_moving else (0, 0, int(200 * alpha))
                pt1 = (int(cx[ti]), int(cy[ti]))
                pt2 = (int(cx[ti + 1]), int(cy[ti + 1]))
                cv2.line(vis, pt1, pt2, seg_color, 2, cv2.LINE_AA)

            # --- HUD text ---
            try:
                thr = local_thresh_var.get()
            except (tk.TclError, ValueError):
                thr = 20.0
            state_str = "MOVING" if is_moving else "STATIONARY"
            hud_lines = [
                f'Speed: {spd:.1f} px/s  |  {state_str}',
                f'Threshold: {thr:.1f} px/s  |  Frame: {fi}/{n_frames}',
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, line in enumerate(hud_lines):
                y_pos = 22 + i * 22
                (tw, th), _ = cv2.getTextSize(line, font, 0.55, 1)
                cv2.rectangle(vis, (4, y_pos - th - 4), (8 + tw, y_pos + 4), (0, 0, 0), -1)
                color = (0, 255, 0) if is_moving else (0, 100, 255)
                cv2.putText(vis, line, (6, y_pos), font, 0.55, color, 1, cv2.LINE_AA)

            # --- speed gauge bar at bottom ---
            bar_h = 20
            bar_y1 = fh - bar_h - 6
            bar_y2 = fh - 6
            bar_x1 = 10
            bar_x2 = fw - 10
            bar_w = bar_x2 - bar_x1
            cv2.rectangle(vis, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)

            # Speed fill
            max_display = max(float(body_speed.max()), thr * 2, 1.0)
            fill_frac = min(spd / max_display, 1.0)
            fill_x = bar_x1 + int(bar_w * fill_frac)
            fill_color = (0, 200, 0) if is_moving else (0, 0, 200)
            cv2.rectangle(vis, (bar_x1, bar_y1), (fill_x, bar_y2), fill_color, -1)

            # Threshold marker (yellow line)
            thr_frac = min(thr / max_display, 1.0)
            thr_x = bar_x1 + int(bar_w * thr_frac)
            cv2.line(vis, (thr_x, bar_y1 - 2), (thr_x, bar_y2 + 2), (0, 255, 255), 2)
            cv2.putText(vis, 'thr', (thr_x - 10, bar_y1 - 4), font, 0.35, (0, 255, 255), 1)

            # Outline
            cv2.rectangle(vis, (bar_x1, bar_y1), (bar_x2, bar_y2), (120, 120, 120), 1)

            # --- update frame readout labels ---
            frame_speed_lbl.config(text=f"Speed: {spd:.1f} px/s")
            frame_state_lbl.config(
                text=f"State: {state_str}",
                foreground='green' if is_moving else 'red')

            # --- display on canvas ---
            cw = canvas.winfo_width()  or 680
            ch = canvas.winfo_height() or 520
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
            canvas.image = photo

        # --- playback functions ---
        def _toggle_play():
            playing[0] = not playing[0]
            if playing[0]:
                play_btn.config(text="\u23f8 Pause")
                _play_tick()
            else:
                play_btn.config(text="\u25b6 Play")
                if play_after_id[0]:
                    win.after_cancel(play_after_id[0])
                    play_after_id[0] = None

        play_btn.config(command=_toggle_play)

        def _play_tick():
            if not playing[0]:
                return
            fi = frame_var.get()
            if fi >= n_frames - 1:
                playing[0] = False
                play_btn.config(text="\u25b6 Play")
                return
            frame_var.set(fi + 1)  # triggers _render via trace
            interval = max(1, int((1000.0 / fps) / speed_var.get()))
            play_after_id[0] = win.after(interval, _play_tick)

        def _on_close():
            playing[0] = False
            if play_after_id[0]:
                win.after_cancel(play_after_id[0])
            cap.release()
            win.destroy()

        win.protocol('WM_DELETE_WINDOW', _on_close)
        win.bind('<space>', lambda e: _toggle_play())

        # Wire traces
        frame_var.trace_add('write', _render)
        local_thresh_var.trace_add('write', lambda *_: (_update_summary(), _render()))
        smooth_var.trace_add('write', lambda *_: (_update_summary(), _render()))
        min_bout_var.trace_add('write', lambda *_: (_update_summary(), _render()))
        win.bind('<Configure>', _render)
        win.after(100, _do_render)

    # ── Brightness preview ───────────────────────────────────────────────

    def _open_brightness_preview(self):
        if not _CV2_OK:
            messagebox.showerror("Missing dependency",
                                 "OpenCV (cv2) is required for the brightness preview.",
                                 parent=self)
            return

        # --- find a usable session (needs video + DLC) ---
        selected = [self._sess_tree.item(i, 'values')[0]
                    for i in self._sess_tree.selection()]
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and s.get('video') and os.path.isfile(s['video'])
                      and s.get('dlc')   and os.path.isfile(s['dlc'])]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session that has both a video and DLC file.",
                parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # --- read DLC coordinates + paw heights ---
        active_paws = {r: self._role_vars[r].get().strip()
                       for r in self.ROLES if self._role_vars[r].get().strip()}
        active_bps  = list(set(active_paws.values()))
        try:
            ext      = PoseFeatureExtractor(active_bps)
            dlc_df   = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return

        try:
            height_df = ext.calculate_paw_height(
                bp_xcord, bp_ycord, window=self._height_window_var.get())
        except Exception:
            height_df = None

        # --- open video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                 f"Cannot open:\n{video_path}", parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- local preview vars ---
        contact_var    = tk.IntVar(value=self._contact_thresh_var.get())
        roi_vars       = {role: tk.IntVar(value=self._roi_size_vars[role].get())
                          for role in active_paws}
        crop_x_var     = tk.IntVar(value=self._crop_x_var.get())
        crop_y_var     = tk.IntVar(value=self._crop_y_var.get())
        brt_weight_var = tk.DoubleVar(value=self._brt_weight_var.get())

        # --- build window ---
        win = tk.Toplevel(self)
        win.title(f"Brightness Preview — {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.60)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.protocol('WM_DELETE_WINDOW', lambda: (cap.release(), win.destroy()))

        # Scrollable ctrl panel — pack FIRST so it gets priority for space
        ctrl_outer = ttk.Frame(win)
        ctrl_outer.pack(side='right', fill='both', padx=6, pady=6)
        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0, width=300)
        ctrl_sb_y = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_sb_x = ttk.Scrollbar(ctrl_outer, orient='horizontal', command=ctrl_canvas.xview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb_y.set, xscrollcommand=ctrl_sb_x.set)
        ctrl_sb_x.pack(side='bottom', fill='x')
        ctrl_sb_y.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_win_id = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(ctrl_win_id, width=e.width))

        # Video canvas — packed after ctrl so it fills remaining space
        canvas = tk.Canvas(win, bg='black', width=480, height=380)
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        # Frame slider
        ttk.Label(ctrl, text="Frame:").pack(anchor='w')
        frame_var = tk.IntVar(value=0)
        frame_sb = ttk.Spinbox(ctrl, from_=0, to=max(n_frames - 1, 0),
                                textvariable=frame_var, width=8)
        frame_sb.pack(anchor='w', pady=(0, 2))
        frame_slider = ttk.Scale(ctrl, from_=0, to=max(n_frames - 1, 0),
                                  variable=frame_var, orient='vertical', length=180)
        frame_slider.pack(anchor='w', pady=(0, 8))

        # Brightness threshold slider
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Brt threshold (0=auto):").pack(anchor='w')
        thresh_var = tk.IntVar(value=self._brt_thresh_var.get())
        thresh_sb = ttk.Spinbox(ctrl, from_=0, to=255, textvariable=thresh_var, width=8)
        thresh_sb.pack(anchor='w', pady=(0, 2))
        thresh_slider = ttk.Scale(ctrl, from_=0, to=255,
                                   variable=thresh_var, orient='vertical', length=160)
        thresh_slider.pack(anchor='w', pady=(0, 4))

        # Contact threshold spinbox
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Contact thresh (px):").pack(anchor='w')
        contact_sb = ttk.Spinbox(ctrl, from_=0, to=500, textvariable=contact_var, width=8)
        contact_sb.pack(anchor='w', pady=(0, 6))

        # Crop offset spinboxes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Crop offset (px):").pack(anchor='w')
        crop_grid = ttk.Frame(ctrl)
        crop_grid.pack(anchor='w', pady=(0, 6))
        ttk.Label(crop_grid, text="X:").grid(row=0, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_x_var,
                    width=6).grid(row=0, column=1, sticky='w')
        ttk.Label(crop_grid, text="Y:").grid(row=1, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_y_var,
                    width=6).grid(row=1, column=1, sticky='w')

        # Brt contact weight
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Brt contact weight:").pack(anchor='w')
        ttk.Spinbox(ctrl, from_=0.0, to=1.0, increment=0.05,
                    textvariable=brt_weight_var, width=6, format='%.2f').pack(
                        anchor='w', pady=(0, 6))

        # Per-paw ROI half-sizes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="ROI half-size (px):").pack(anchor='w')
        roi_grid = ttk.Frame(ctrl)
        roi_grid.pack(anchor='w', pady=(0, 6))
        for i, (role, var) in enumerate(roi_vars.items()):
            ttk.Label(roi_grid, text=f"{role}:").grid(
                row=i // 2, column=(i % 2) * 2, sticky='w', padx=(0, 2))
            ttk.Spinbox(roi_grid, from_=5, to=200, textvariable=var, width=5).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky='w', padx=(0, 6))

        # Per-paw readout labels
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Paw readouts:").pack(anchor='w')
        paw_labels = {}
        for role, bp in active_paws.items():
            lbl = ttk.Label(ctrl, text=f"{role} ({bp}): —", wraplength=270)
            lbl.pack(anchor='w')
            paw_labels[role] = lbl

        # Auto-detect brightness label
        auto_lbl = ttk.Label(ctrl, text='', foreground='grey', wraplength=270)
        auto_lbl.pack(anchor='w', pady=(4, 0))

        # "Apply all to main settings" button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        def _apply_all():
            self._brt_thresh_var.set(thresh_var.get())
            self._contact_thresh_var.set(contact_var.get())
            self._crop_x_var.set(crop_x_var.get())
            self._crop_y_var.set(crop_y_var.get())
            self._brt_weight_var.set(brt_weight_var.get())
            for role in active_paws:
                self._roi_size_vars[role].set(roi_vars[role].get())
            roi_str = ', '.join(f'{r}={roi_vars[r].get()}' for r in active_paws)
            self._log_ui(
                f"Applied: brt_thresh={thresh_var.get() or 'auto'}, "
                f"contact_thresh={contact_var.get()}, "
                f"brt_weight={brt_weight_var.get():.2f}, "
                f"crop=({crop_x_var.get()},{crop_y_var.get()}), "
                f"ROI={{{roi_str}}}"
            )

        ttk.Button(ctrl, text="Apply all to main settings",
                   command=_apply_all).pack(fill='x', pady=(0, 2))

        # --- render function (debounced 50 ms) ---
        _after_id = [None]

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except (tk.TclError, ValueError):
                return default

        def _do_render():
            fi   = _safe_int(frame_var, 0)
            thr  = _safe_int(thresh_var, 0)
            cthr = _safe_int(contact_var, self._contact_thresh_var.get())
            cx   = _safe_int(crop_x_var, 0)
            cy   = _safe_int(crop_y_var, 0)

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Effective brightness threshold
            if thr == 0:
                effective = float(gray.mean() * 0.5)
                auto_lbl.config(text=f'Auto → {effective:.1f}')
            else:
                effective = float(thr)
                auto_lbl.config(text='')

            # Pixels below brightness threshold → dimmed blue
            vis = frame.copy()
            below_mask = gray < effective
            vis[below_mask] = (vis[below_mask] * 0.25).astype(np.uint8)
            vis[below_mask, 0] = np.clip(
                vis[below_mask, 0].astype(int) + 60, 0, 255).astype(np.uint8)

            _paw_colors = {'HL': (0, 255, 0), 'HR': (0, 0, 255),
                           'FL': (255, 255, 0), 'FR': (0, 165, 255)}

            # Draw ROI boxes per paw
            for role, bp in active_paws.items():
                x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
                y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
                if x_col is None or y_col is None:
                    continue
                if fi >= len(bp_xcord):
                    continue

                bx = int(bp_xcord[x_col].iloc[fi]) + cx
                by = int(bp_ycord[y_col].iloc[fi]) + cy

                rh = _safe_int(roi_vars[role], 25)   # half-size in px
                fh, fw = frame.shape[:2]
                x1 = max(0, bx - rh);  x2 = min(fw, bx + rh)
                y1 = max(0, by - rh);  y2 = min(fh, by + rh)

                # Brightness in ROI
                roi_gray = gray[y1:y2, x1:x2].copy()
                roi_gray[roi_gray < effective] = 1.0
                brt = float(roi_gray.mean()) if roi_gray.size > 0 else 0.0

                # Paw height + contact state
                paw_h = None
                if height_df is not None:
                    h_col = next((c for c in height_df.columns
                                  if bp.lower() in c.lower()), None)
                    if h_col and fi < len(height_df):
                        paw_h = float(height_df[h_col].iloc[fi])
                bw = brt_weight_var.get()
                if bw > 0 and paw_h is not None:
                    h_score = max(0.0, min(1.0, 1.0 - paw_h / max(float(cthr), 1.0)))
                    # Use effective*2 as rough session-peak proxy (preview has no full series)
                    b_score = min(1.0, brt / max(effective * 2.0, 1.0))
                    in_contact = ((1.0 - bw) * h_score + bw * b_score) > 0.5
                else:
                    in_contact = (paw_h is not None and paw_h < cthr)

                # Update paw label
                h_str = f'h={paw_h:.1f}' if paw_h is not None else 'h=?'
                contact_str = ' ✓' if in_contact else ''
                paw_labels[role].config(
                    text=f"{role} ({bp}): brt={brt:.1f} {h_str}{contact_str}")

                color = _paw_colors.get(role, (255, 255, 255))

                # Contact fill overlay (semi-transparent)
                if in_contact:
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

                lw = 3 if in_contact else 1
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)
                cv2.circle(vis, (bx, by), 4, color, -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(role, font, 0.55, 1)
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(vis, role, (x1 + 2, y1 - 2), font, 0.55, color, 1)

            # HUD text
            hud = f'brt_thr={effective:.1f}  contact_thr={cthr}'
            if cx or cy:
                hud += f'  crop=({cx},{cy})'
            cv2.putText(vis, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # Display on canvas
            cw = canvas.winfo_width()  or 680
            ch = canvas.winfo_height() or 520
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
            canvas.image = photo   # keep reference

        # Wire all traces → debounced render
        frame_var.trace_add('write', _render)
        thresh_var.trace_add('write', _render)
        contact_var.trace_add('write', _render)
        crop_x_var.trace_add('write', _render)
        crop_y_var.trace_add('write', _render)
        brt_weight_var.trace_add('write', _render)
        for var in roi_vars.values():
            var.trace_add('write', _render)
        win.bind('<Configure>', _render)
        win.after(100, _do_render)   # initial render after window maps

    # ── Contour preview ─────────────────────────────────────────────────────

    def _open_contour_preview(self):
        if not _CV2_OK:
            messagebox.showerror("Missing dependency",
                                 "OpenCV (cv2) is required for the contour preview.",
                                 parent=self)
            return

        # --- find a usable session (needs video + DLC) ---
        selected = [self._sess_tree.item(i, 'values')[0]
                    for i in self._sess_tree.selection()]
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and s.get('video') and os.path.isfile(s['video'])
                      and s.get('dlc')   and os.path.isfile(s['dlc'])]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session that has both a video and DLC file.",
                parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # --- read DLC coordinates ---
        active_paws = {r: self._role_vars[r].get().strip()
                       for r in self.ROLES if self._role_vars[r].get().strip()}
        active_bps  = list(set(active_paws.values()))
        try:
            ext      = PoseFeatureExtractor(active_bps)
            dlc_df   = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return

        # --- open video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                 f"Cannot open:\n{video_path}", parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- local preview vars ---
        roi_vars       = {role: tk.IntVar(value=self._contour_roi_size_vars[role].get())
                          for role in active_paws}
        crop_x_var     = tk.IntVar(value=self._crop_x_var.get())
        crop_y_var     = tk.IntVar(value=self._crop_y_var.get())
        thresh_mode_var = tk.StringVar(value='otsu')
        manual_thresh_var = tk.IntVar(value=128)
        blur_kernel_var = tk.IntVar(value=3)
        min_area_var    = tk.IntVar(value=10)

        # --- build window ---
        win = tk.Toplevel(self)
        win.title(f"Contour Preview \u2014 {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.60)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.protocol('WM_DELETE_WINDOW', lambda: (cap.release(), win.destroy()))

        # Scrollable ctrl panel — pack FIRST so it gets priority for space
        ctrl_outer = ttk.Frame(win)
        ctrl_outer.pack(side='right', fill='both', padx=6, pady=6)
        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0, width=300)
        ctrl_sb_y = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_sb_x = ttk.Scrollbar(ctrl_outer, orient='horizontal', command=ctrl_canvas.xview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb_y.set, xscrollcommand=ctrl_sb_x.set)
        ctrl_sb_x.pack(side='bottom', fill='x')
        ctrl_sb_y.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_win_id = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(ctrl_win_id, width=e.width))

        # Video canvas — packed after ctrl so it fills remaining space
        canvas = tk.Canvas(win, bg='black', width=480, height=380)
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        # Frame slider
        ttk.Label(ctrl, text="Frame:").pack(anchor='w')
        frame_var = tk.IntVar(value=0)
        frame_sb = ttk.Spinbox(ctrl, from_=0, to=max(n_frames - 1, 0),
                                textvariable=frame_var, width=8)
        frame_sb.pack(anchor='w', pady=(0, 2))
        frame_slider = ttk.Scale(ctrl, from_=0, to=max(n_frames - 1, 0),
                                  variable=frame_var, orient='vertical', length=180)
        frame_slider.pack(anchor='w', pady=(0, 8))

        # Contour threshold mode
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Threshold mode:").pack(anchor='w')
        ttk.Radiobutton(ctrl, text="Otsu (auto)", variable=thresh_mode_var,
                        value='otsu').pack(anchor='w')
        ttk.Radiobutton(ctrl, text="Manual", variable=thresh_mode_var,
                        value='manual').pack(anchor='w')

        # Manual threshold
        ttk.Label(ctrl, text="Manual threshold:").pack(anchor='w', pady=(4, 0))
        manual_sb = ttk.Spinbox(ctrl, from_=0, to=255, textvariable=manual_thresh_var,
                                width=8, state='disabled')
        manual_sb.pack(anchor='w', pady=(0, 4))

        def _on_mode_change(*_):
            manual_sb.config(state='normal' if thresh_mode_var.get() == 'manual' else 'disabled')
        thresh_mode_var.trace_add('write', _on_mode_change)

        # Gaussian blur kernel
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Gaussian blur kernel:").pack(anchor='w')
        ttk.Spinbox(ctrl, values=(1, 3, 5, 7), textvariable=blur_kernel_var,
                    width=8).pack(anchor='w', pady=(0, 4))

        # Min contour area
        ttk.Label(ctrl, text="Min contour area (px\u00b2):").pack(anchor='w')
        ttk.Spinbox(ctrl, from_=0, to=5000, textvariable=min_area_var,
                    width=8).pack(anchor='w', pady=(0, 4))

        # Per-paw ROI half-sizes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Contour ROI half-size (px):").pack(anchor='w')
        roi_grid = ttk.Frame(ctrl)
        roi_grid.pack(anchor='w', pady=(0, 6))
        for i, (role, var) in enumerate(roi_vars.items()):
            ttk.Label(roi_grid, text=f"{role}:").grid(
                row=i // 2, column=(i % 2) * 2, sticky='w', padx=(0, 2))
            ttk.Spinbox(roi_grid, from_=5, to=200, textvariable=var, width=5).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky='w', padx=(0, 6))

        # Crop offset spinboxes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Crop offset (px):").pack(anchor='w')
        crop_grid = ttk.Frame(ctrl)
        crop_grid.pack(anchor='w', pady=(0, 6))
        ttk.Label(crop_grid, text="X:").grid(row=0, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_x_var,
                    width=6).grid(row=0, column=1, sticky='w')
        ttk.Label(crop_grid, text="Y:").grid(row=1, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_y_var,
                    width=6).grid(row=1, column=1, sticky='w')

        # Per-paw readout labels
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Paw readouts:").pack(anchor='w')
        paw_labels = {}
        for role, bp in active_paws.items():
            lbl = ttk.Label(ctrl, text=f"{role} ({bp}): \u2014", wraplength=270)
            lbl.pack(anchor='w')
            paw_labels[role] = lbl

        # "Apply contour ROI sizes to main settings" button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        def _apply_roi():
            self._crop_x_var.set(crop_x_var.get())
            self._crop_y_var.set(crop_y_var.get())
            for role in active_paws:
                self._contour_roi_size_vars[role].set(roi_vars[role].get())
            roi_str = ', '.join(f'{r}={roi_vars[r].get()}' for r in active_paws)
            self._log_ui(
                f"Applied: crop=({crop_x_var.get()},{crop_y_var.get()}), "
                f"contour ROI={{{roi_str}}}"
            )

        ttk.Button(ctrl, text="Apply contour ROI to main settings",
                   command=_apply_roi).pack(fill='x', pady=(0, 2))

        # --- render function (debounced 50 ms) ---
        _after_id = [None]

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except (tk.TclError, ValueError):
                return default

        def _do_render():
            fi  = _safe_int(frame_var, 0)
            cx  = _safe_int(crop_x_var, 0)
            cy  = _safe_int(crop_y_var, 0)
            k   = _safe_int(blur_kernel_var, 3)
            min_ca = _safe_int(min_area_var, 10)
            m_thr  = _safe_int(manual_thresh_var, 128)

            # Ensure kernel is odd and >= 1
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            _paw_colors = {'HL': (0, 255, 0), 'HR': (0, 0, 255),
                           'FL': (255, 255, 0), 'FR': (0, 165, 255)}

            for role, bp in active_paws.items():
                x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
                y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
                if x_col is None or y_col is None:
                    continue
                if fi >= len(bp_xcord):
                    continue

                bx = int(bp_xcord[x_col].iloc[fi]) + cx
                by = int(bp_ycord[y_col].iloc[fi]) + cy

                rh = _safe_int(roi_vars[role], 25)
                fh, fw = frame.shape[:2]
                x1 = max(0, bx - rh);  x2 = min(fw, bx + rh)
                y1 = max(0, by - rh);  y2 = min(fh, by + rh)

                color = _paw_colors.get(role, (255, 255, 255))

                if x2 <= x1 or y2 <= y1:
                    paw_labels[role].config(
                        text=f"{role} ({bp}): ROI out of bounds")
                    continue

                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    paw_labels[role].config(
                        text=f"{role} ({bp}): empty ROI")
                    continue

                # Gaussian blur + threshold
                blurred = cv2.GaussianBlur(roi, (k, k), 0)
                if thresh_mode_var.get() == 'otsu':
                    _, thresh_img = cv2.threshold(
                        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, thresh_img = cv2.threshold(
                        blurred, m_thr, 255, cv2.THRESH_BINARY)

                # Find contours
                contours, _ = cv2.findContours(
                    thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter by min area
                contours = [c for c in contours if cv2.contourArea(c) >= min_ca]

                area = 0.0
                spread = 0.0
                mean_int = 0.0

                if contours:
                    best = max(contours, key=lambda c: cv2.contourArea(c))
                    area = cv2.contourArea(best)

                    xb, yb, wb, hb = cv2.boundingRect(best)
                    spread = float(max(wb, hb))

                    # Mean intensity within contour
                    mask_c = np.zeros(roi.shape, dtype=np.uint8)
                    cv2.drawContours(mask_c, [best], -1, 255, -1)
                    mean_int = cv2.mean(roi, mask=mask_c)[0]

                    # Draw contour on visualization frame, offset to frame coords
                    contour_offset = best.copy()
                    contour_offset[:, :, 0] += x1
                    contour_offset[:, :, 1] += y1
                    cv2.drawContours(vis, [contour_offset], -1, (0, 255, 0), 2)

                # Draw ROI rectangle
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.circle(vis, (bx, by), 4, color, -1)

                # Label
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th_t), _ = cv2.getTextSize(role, font, 0.55, 1)
                cv2.rectangle(vis, (x1, y1 - th_t - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(vis, role, (x1 + 2, y1 - 2), font, 0.55, color, 1)

                # Area / spread / intensity text overlay
                info_txt = f"A={area:.0f} S={spread:.0f} I={mean_int:.1f}"
                cv2.putText(vis, info_txt, (x1 + 2, y2 + 14),
                            font, 0.4, color, 1)

                # Update readout label
                paw_labels[role].config(
                    text=f"{role} ({bp}): area={area:.0f}  spread={spread:.0f}  int={mean_int:.1f}")

            # HUD text
            mode_str = thresh_mode_var.get()
            if mode_str == 'manual':
                mode_str += f'={m_thr}'
            hud = f'thresh={mode_str}  blur={k}  min_area={min_ca}'
            if cx or cy:
                hud += f'  crop=({cx},{cy})'
            cv2.putText(vis, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # Display on canvas
            cw = canvas.winfo_width()  or 680
            ch = canvas.winfo_height() or 520
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
            canvas.image = photo   # keep reference

        # Wire all traces -> debounced render
        frame_var.trace_add('write', _render)
        thresh_mode_var.trace_add('write', _render)
        manual_thresh_var.trace_add('write', _render)
        blur_kernel_var.trace_add('write', _render)
        min_area_var.trace_add('write', _render)
        crop_x_var.trace_add('write', _render)
        crop_y_var.trace_add('write', _render)
        for var in roi_vars.values():
            var.trace_add('write', _render)
        win.bind('<Configure>', _render)
        win.after(100, _do_render)   # initial render after window maps

    # ── Detect cached brightness ─────────────────────────────────────────────

    def _detect_brightness_caches(self):
        import json, glob as _glob

        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project",
                                   "Open a project first.", parent=self)
            return

        # Collect session names to scan
        selected_iids = self._sess_tree.selection()
        if selected_iids:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in selected_iids]
        else:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in self._sess_tree.get_children()]

        if not session_names:
            messagebox.showinfo("No sessions",
                                "No sessions found in the session list.",
                                parent=self)
            return

        wb_dirs = [os.path.join(folder, 'gait_limb_analysis'),
                   os.path.join(folder, 'weight_bearing_analysis')]

        # Build current settings snapshot for comparison (use bp names as keys, same as cache)
        _cur_paw_map = {r: self._role_vars[r].get().strip() for r in self.ROLES}
        current = {
            'brt_thresh':        self._brt_thresh_var.get(),
            'extraction_stride': self._extraction_stride_var.get(),
            'roi_sizes':         sorted(
                {bp: self._roi_size_vars[r].get()
                 for r, bp in _cur_paw_map.items() if bp}.items()),
            'crop_x':            self._crop_x_var.get(),
            'crop_y':            self._crop_y_var.get(),
        }

        # Scan for cache CSVs; load sidecar JSON if present
        found = []  # (session_name, csv_path, settings_dict_or_None)
        for sname in session_names:
            for wb_dir in wb_dirs:
                pattern = os.path.join(wb_dir, f'{sname}_brt_*.csv')
                for csv_path in _glob.glob(pattern):
                    sidecar = csv_path.replace('.csv', '.json')
                    settings = None
                    if os.path.isfile(sidecar):
                        try:
                            with open(sidecar) as fh:
                                settings = json.load(fh)
                        except Exception:
                            pass
                    found.append((sname, csv_path, settings))

        # ── Build dialog ─────────────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Cached Brightness Files")
        win.resizable(True, True)

        if not found:
            msg = (f"No brightness cache files were found for the selected "
                   f"sessions under:\n{wb_dir}")
            ttk.Label(win, text=msg, wraplength=740,
                      justify='left', padding=20).pack(fill='both', expand=True)
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)
            return

        # Scrollable table
        outer = ttk.Frame(win, padding=8)
        outer.pack(fill='both', expand=True)

        cols = ('session', 'thresh', 'stride', 'roi_hl', 'roi_hr', 'roi_fl', 'roi_fr',
                'crop_x', 'crop_y', 'match')
        tree = ttk.Treeview(outer, columns=cols, show='headings', height=12)
        headers = [
            ('session', 'Session',       140),
            ('thresh',  'Threshold',      70),
            ('stride',  'Stride',         50),
            ('roi_hl',  'HL (px)',         60),
            ('roi_hr',  'HR (px)',         60),
            ('roi_fl',  'FL (px)',         60),
            ('roi_fr',  'FR (px)',         60),
            ('crop_x',  'Crop X',          55),
            ('crop_y',  'Crop Y',          55),
            ('match',   'Match?',         160),
        ]
        for col, hdr, w in headers:
            tree.heading(col, text=hdr)
            tree.column(col, width=w, stretch=(col in ('session', 'match')))

        vsb = ttk.Scrollbar(outer, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(outer, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        tree.tag_configure('match',    foreground='green')
        tree.tag_configure('mismatch', foreground='red')
        tree.tag_configure('unknown',  foreground='gray')

        paw_map = {r: self._role_vars[r].get().strip() for r in self.ROLES}

        for sname, csv_path, settings in found:
            if settings is None:
                # No sidecar — cache predates sidecar support; settings unknown
                tree.insert('', 'end', tags=('unknown',), values=(
                    sname, '?', '?', '?', '?', '?', '?', '?', '?',
                    'no settings file (pre-dates sidecar support)',
                ))
                continue

            roi_list = settings.get('roi_sizes', [])

            # Match bp names from current paw mapping to get per-role ROI sizes
            roi_by_role = {}
            for role in self.ROLES:
                bp = paw_map.get(role, '')
                for bp_name, sz in (roi_list if isinstance(roi_list, list) else []):
                    if bp_name == bp:
                        roi_by_role[role] = sz
                        break

            cached_thresh  = settings.get('brt_thresh', '?')
            cached_stride  = settings.get('extraction_stride', '?')
            cached_crop_x  = settings.get('crop_x', '?')
            cached_crop_y  = settings.get('crop_y', '?')
            cached_roi_key = sorted(settings.get('roi_sizes', []))

            diffs = []
            if cached_thresh != current['brt_thresh']:
                diffs.append(f"thresh: {cached_thresh}→{current['brt_thresh']}")
            if cached_stride != current['extraction_stride']:
                diffs.append(f"stride: {cached_stride}→{current['extraction_stride']}")
            if cached_crop_x != current['crop_x']:
                diffs.append(f"crop_x: {cached_crop_x}→{current['crop_x']}")
            if cached_crop_y != current['crop_y']:
                diffs.append(f"crop_y: {cached_crop_y}→{current['crop_y']}")
            if cached_roi_key != current['roi_sizes']:
                diffs.append("roi_sizes differ")

            match_str = '\u2713' if not diffs else ', '.join(diffs)
            tag       = 'match' if not diffs else 'mismatch'

            tree.insert('', 'end', tags=(tag,), values=(
                sname,
                cached_thresh,
                cached_stride,
                roi_by_role.get('HL', '?'),
                roi_by_role.get('HR', '?'),
                roi_by_role.get('FL', '?'),
                roi_by_role.get('FR', '?'),
                cached_crop_x,
                cached_crop_y,
                match_str,
            ))

        note = ttk.Label(win,
                         text=("Cached files are reused automatically when settings match. "
                               "Mismatched sessions will trigger fresh extraction during analysis."),
                         font=('Arial', 8), foreground='gray', wraplength=780)
        note.pack(side='bottom', anchor='w', padx=10, pady=(4, 6))

        btn_row = ttk.Frame(win)
        btn_row.pack(side='bottom', pady=6)

        cached_session_names = list({sname for sname, _, _ in found})

        # Collect unique cached settings from sidecars (ignore file mtimes)
        def _settings_key(s):
            if s is None:
                return None
            return (s.get('brt_thresh'), s.get('extraction_stride'),
                    s.get('crop_x'), s.get('crop_y'),
                    tuple(tuple(x) for x in sorted(s.get('roi_sizes', []))))

        settings_with_sidecar = [s for _, _, s in found if s is not None]
        unique_keys = list(dict.fromkeys(_settings_key(s) for s in settings_with_sidecar))
        # Pick the most recent sidecar's settings (sort found by CSV mtime descending)
        found_with_sidecar = [(sname, csv_path, s)
                              for sname, csv_path, s in found if s is not None]
        if found_with_sidecar:
            found_with_sidecar.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            offer_settings = found_with_sidecar[0][2]
        else:
            offer_settings = None

        def _select_sessions():
            iids = [iid for iid in self._sess_tree.get_children()
                    if self._sess_tree.item(iid, 'values')[0] in cached_session_names]
            if iids:
                self._sess_tree.selection_set(iids)

        def _analyze_cached():
            """Apply cached settings to UI, then run analysis."""
            s = offer_settings
            if s:
                if 'brt_thresh' in s:
                    self._brt_thresh_var.set(s['brt_thresh'])
                if 'extraction_stride' in s:
                    self._extraction_stride_var.set(s['extraction_stride'])
                if 'crop_x' in s:
                    self._crop_x_var.set(s['crop_x'])
                if 'crop_y' in s:
                    self._crop_y_var.set(s['crop_y'])
                # Map bp_name → role via current role vars, then set ROI sizes
                roi_list = s.get('roi_sizes', [])
                if roi_list:
                    paw_map = {self._role_vars[r].get().strip(): r
                               for r in self.ROLES if self._role_vars[r].get().strip()}
                    for bp_name, sz in roi_list:
                        role = paw_map.get(bp_name)
                        if role:
                            self._roi_size_vars[role].set(sz)
                self._use_brightness_var.set(True)
            win.destroy()
            _select_sessions()
            self._start_analysis()

        def _analyze_current():
            win.destroy()
            _select_sessions()
            self._start_analysis()

        if offer_settings:
            if len(unique_keys) > 1:
                # Build list of unique settings with human-readable summaries
                key_to_settings = {}
                for s in settings_with_sidecar:
                    k = _settings_key(s)
                    if k not in key_to_settings:
                        key_to_settings[k] = s

                settings_list = []
                summaries = []
                for idx, k in enumerate(unique_keys):
                    s = key_to_settings[k]
                    t = s.get('brt_thresh', '?')
                    stride = s.get('extraction_stride', '?')
                    roi_list = s.get('roi_sizes', [])
                    roi_str = (', '.join(f'{bp}={sz}' for bp, sz in roi_list)
                               if roi_list else '?')
                    cx = s.get('crop_x', 0)
                    cy = s.get('crop_y', 0)
                    label = (f"Set {idx+1}: thresh={t}, stride={stride}, "
                             f"ROI=[{roi_str}], crop=({cx},{cy})")
                    settings_list.append(s)
                    summaries.append(label)

                sel_frame = ttk.LabelFrame(
                    win, text="Select settings to apply", padding=5)
                sel_frame.pack(fill='x', padx=10, pady=(4, 2))

                sel_var = tk.StringVar(value=summaries[0])
                sel_combo = ttk.Combobox(
                    sel_frame, textvariable=sel_var,
                    values=summaries, state='readonly', width=70)
                sel_combo.pack(fill='x', pady=2)

                def _analyze_selected():
                    idx = (summaries.index(sel_var.get())
                           if sel_var.get() in summaries else 0)
                    chosen = settings_list[idx]
                    if chosen:
                        if 'brt_thresh' in chosen:
                            self._brt_thresh_var.set(chosen['brt_thresh'])
                        if 'extraction_stride' in chosen:
                            self._extraction_stride_var.set(
                                chosen['extraction_stride'])
                        if 'crop_x' in chosen:
                            self._crop_x_var.set(chosen['crop_x'])
                        if 'crop_y' in chosen:
                            self._crop_y_var.set(chosen['crop_y'])
                        roi_list = chosen.get('roi_sizes', [])
                        if roi_list:
                            paw_map = {
                                self._role_vars[r].get().strip(): r
                                for r in self.ROLES
                                if self._role_vars[r].get().strip()}
                            for bp_name, sz in roi_list:
                                role = paw_map.get(bp_name)
                                if role:
                                    self._roi_size_vars[role].set(sz)
                        self._use_brightness_var.set(True)
                    win.destroy()
                    _select_sessions()
                    self._start_analysis()

                ttk.Button(btn_row, text="Analyze with Selected Settings",
                           command=_analyze_selected).pack(side='left', padx=6)
            else:
                ttk.Button(btn_row, text="Analyze with Cached Settings",
                           command=_analyze_cached).pack(side='left', padx=6)

        ttk.Button(btn_row, text="Analyze with Current Settings",
                   command=_analyze_current).pack(side='left', padx=6)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side='left', padx=6)
        _autofit_treeview_window(win, tree)

    def _detect_contour_caches(self):
        import json, glob as _glob

        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project",
                                   "Open a project first.", parent=self)
            return

        # Collect session names to scan
        selected_iids = self._sess_tree.selection()
        if selected_iids:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in selected_iids]
        else:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in self._sess_tree.get_children()]

        if not session_names:
            messagebox.showinfo("No sessions",
                                "No sessions found in the session list.",
                                parent=self)
            return

        wb_dirs = [os.path.join(folder, 'gait_limb_analysis'),
                   os.path.join(folder, 'weight_bearing_analysis')]

        # Build current settings snapshot for comparison
        _cur_paw_map = {r: self._role_vars[r].get().strip() for r in self.ROLES}
        current = {
            'contour_roi_sizes': sorted(
                {bp: self._contour_roi_size_vars[r].get()
                 for r, bp in _cur_paw_map.items() if bp}.items()),
            'crop_x':            self._crop_x_var.get(),
            'crop_y':            self._crop_y_var.get(),
            'extraction_stride': self._extraction_stride_var.get(),
        }

        # Scan for cache CSVs; load sidecar JSON if present
        found = []  # (session_name, csv_path, settings_dict_or_None)
        for sname in session_names:
            for wb_dir in wb_dirs:
                pattern = os.path.join(wb_dir, f'{sname}_contour_*.csv')
                for csv_path in _glob.glob(pattern):
                    sidecar = csv_path.replace('.csv', '.json')
                    settings = None
                    if os.path.isfile(sidecar):
                        try:
                            with open(sidecar) as fh:
                                settings = json.load(fh)
                        except Exception:
                            pass
                    found.append((sname, csv_path, settings))

        # ── Build dialog ─────────────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Cached Contour Files")
        win.resizable(True, True)

        if not found:
            msg = (f"No contour cache files were found for the selected "
                   f"sessions under:\n{wb_dir}")
            ttk.Label(win, text=msg, wraplength=720,
                      justify='left', padding=20).pack(fill='both', expand=True)
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)
            return

        # Scrollable table
        outer = ttk.Frame(win, padding=8)
        outer.pack(fill='both', expand=True)

        cols = ('session', 'roi_hl', 'roi_hr', 'roi_fl', 'roi_fr',
                'stride', 'crop_x', 'crop_y', 'match')
        tree = ttk.Treeview(outer, columns=cols, show='headings', height=12)
        headers = [
            ('session', 'Session',    140),
            ('roi_hl',  'HL (px)',      60),
            ('roi_hr',  'HR (px)',      60),
            ('roi_fl',  'FL (px)',      60),
            ('roi_fr',  'FR (px)',      60),
            ('stride',  'Stride',       50),
            ('crop_x',  'Crop X',       55),
            ('crop_y',  'Crop Y',       55),
            ('match',   'Match?',      180),
        ]
        for col, hdr, w in headers:
            tree.heading(col, text=hdr)
            tree.column(col, width=w, stretch=(col in ('session', 'match')))

        vsb = ttk.Scrollbar(outer, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(outer, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        tree.tag_configure('match',    foreground='green')
        tree.tag_configure('mismatch', foreground='red')
        tree.tag_configure('unknown',  foreground='gray')

        paw_map = {r: self._role_vars[r].get().strip() for r in self.ROLES}

        for sname, csv_path, settings in found:
            if settings is None:
                tree.insert('', 'end', tags=('unknown',), values=(
                    sname, '?', '?', '?', '?', '?', '?', '?',
                    'no settings file',
                ))
                continue

            roi_list = settings.get('contour_roi_sizes', [])

            # Match bp names from current paw mapping to get per-role ROI sizes
            roi_by_role = {}
            for role in self.ROLES:
                bp = paw_map.get(role, '')
                for bp_name, sz in (roi_list if isinstance(roi_list, list) else []):
                    if bp_name == bp:
                        roi_by_role[role] = sz
                        break

            cached_stride = settings.get('extraction_stride', '?')
            cached_crop_x = settings.get('crop_x', '?')
            cached_crop_y = settings.get('crop_y', '?')
            cached_roi_key = sorted(settings.get('contour_roi_sizes', []))

            diffs = []
            if cached_stride != current['extraction_stride']:
                diffs.append(f"stride: {cached_stride}\u2192{current['extraction_stride']}")
            if cached_crop_x != current['crop_x']:
                diffs.append(f"crop_x: {cached_crop_x}\u2192{current['crop_x']}")
            if cached_crop_y != current['crop_y']:
                diffs.append(f"crop_y: {cached_crop_y}\u2192{current['crop_y']}")
            if cached_roi_key != current['contour_roi_sizes']:
                diffs.append("roi_sizes differ")

            match_str = '\u2713' if not diffs else ', '.join(diffs)
            tag       = 'match' if not diffs else 'mismatch'

            tree.insert('', 'end', tags=(tag,), values=(
                sname,
                roi_by_role.get('HL', '?'),
                roi_by_role.get('HR', '?'),
                roi_by_role.get('FL', '?'),
                roi_by_role.get('FR', '?'),
                cached_stride,
                cached_crop_x,
                cached_crop_y,
                match_str,
            ))

        note = ttk.Label(win,
                         text=("Cached files are reused automatically when settings match. "
                               "Mismatched sessions will trigger fresh extraction during analysis."),
                         font=('Arial', 8), foreground='gray', wraplength=740)
        note.pack(side='bottom', anchor='w', padx=10, pady=(4, 6))

        btn_row = ttk.Frame(win)
        btn_row.pack(side='bottom', pady=6)

        cached_session_names = list({sname for sname, _, _ in found})

        # Collect unique cached settings from sidecars
        def _settings_key(s):
            if s is None:
                return None
            return (s.get('extraction_stride'),
                    s.get('crop_x'), s.get('crop_y'),
                    tuple(tuple(x) for x in sorted(s.get('contour_roi_sizes', []))))

        settings_with_sidecar = [s for _, _, s in found if s is not None]
        unique_keys = list(dict.fromkeys(_settings_key(s) for s in settings_with_sidecar))
        found_with_sidecar = [(sname, csv_path, s)
                              for sname, csv_path, s in found if s is not None]
        if found_with_sidecar:
            found_with_sidecar.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            offer_settings = found_with_sidecar[0][2]
        else:
            offer_settings = None

        def _select_sessions():
            iids = [iid for iid in self._sess_tree.get_children()
                    if self._sess_tree.item(iid, 'values')[0] in cached_session_names]
            if iids:
                self._sess_tree.selection_set(iids)

        def _apply_contour_settings(s):
            """Apply cached contour settings to UI variables."""
            if s is None:
                return
            if 'extraction_stride' in s:
                self._extraction_stride_var.set(s['extraction_stride'])
            if 'crop_x' in s:
                self._crop_x_var.set(s['crop_x'])
            if 'crop_y' in s:
                self._crop_y_var.set(s['crop_y'])
            roi_list = s.get('contour_roi_sizes', [])
            if roi_list:
                bp_to_role = {
                    self._role_vars[r].get().strip(): r
                    for r in self.ROLES
                    if self._role_vars[r].get().strip()}
                for bp_name, sz in roi_list:
                    role = bp_to_role.get(bp_name)
                    if role:
                        self._contour_roi_size_vars[role].set(sz)
            self._paw_contour_var.set(True)

        def _analyze_cached():
            _apply_contour_settings(offer_settings)
            win.destroy()
            _select_sessions()
            self._start_analysis()

        def _analyze_current():
            win.destroy()
            _select_sessions()
            self._start_analysis()

        if offer_settings:
            if len(unique_keys) > 1:
                key_to_settings = {}
                for s in settings_with_sidecar:
                    k = _settings_key(s)
                    if k not in key_to_settings:
                        key_to_settings[k] = s

                settings_list = []
                summaries = []
                for idx, k in enumerate(unique_keys):
                    s = key_to_settings[k]
                    stride = s.get('extraction_stride', '?')
                    roi_list = s.get('contour_roi_sizes', [])
                    roi_str = (', '.join(f'{bp}={sz}' for bp, sz in roi_list)
                               if roi_list else '?')
                    cx = s.get('crop_x', 0)
                    cy = s.get('crop_y', 0)
                    label = (f"Set {idx+1}: stride={stride}, "
                             f"ROI=[{roi_str}], crop=({cx},{cy})")
                    settings_list.append(s)
                    summaries.append(label)

                sel_frame = ttk.LabelFrame(
                    win, text="Select settings to apply", padding=5)
                sel_frame.pack(fill='x', padx=10, pady=(4, 2))

                sel_var = tk.StringVar(value=summaries[0])
                sel_combo = ttk.Combobox(
                    sel_frame, textvariable=sel_var,
                    values=summaries, state='readonly', width=70)
                sel_combo.pack(fill='x', pady=2)

                def _analyze_selected():
                    idx = (summaries.index(sel_var.get())
                           if sel_var.get() in summaries else 0)
                    chosen = settings_list[idx]
                    _apply_contour_settings(chosen)
                    win.destroy()
                    _select_sessions()
                    self._start_analysis()

                ttk.Button(btn_row, text="Analyze with Selected Settings",
                           command=_analyze_selected).pack(side='left', padx=6)
            else:
                ttk.Button(btn_row, text="Analyze with Cached Settings",
                           command=_analyze_cached).pack(side='left', padx=6)

        ttk.Button(btn_row, text="Analyze with Current Settings",
                   command=_analyze_current).pack(side='left', padx=6)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side='left', padx=6)
        _autofit_treeview_window(win, tree)

    # ── Detect both caches (brightness + contour) ──────────────────────────

    def _detect_both_caches(self):
        import json, glob as _glob

        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project",
                                   "Open a project first.", parent=self)
            return

        # Collect session names
        selected_iids = self._sess_tree.selection()
        if selected_iids:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in selected_iids]
        else:
            session_names = [self._sess_tree.item(i, 'values')[0]
                             for i in self._sess_tree.get_children()]

        if not session_names:
            messagebox.showinfo("No sessions",
                                "No sessions found in the session list.",
                                parent=self)
            return

        wb_dirs = [os.path.join(folder, 'gait_limb_analysis'),
                   os.path.join(folder, 'weight_bearing_analysis')]

        # ── Scan for brightness and contour caches per session ────────────
        per_session = {}  # {sname: {'brt': [(csv, settings), ...], 'contour': [...]}}
        for sname in session_names:
            brt_hits = []
            for wb_dir in wb_dirs:
                for csv_path in _glob.glob(os.path.join(wb_dir, f'{sname}_brt_*.csv')):
                    sidecar = csv_path.replace('.csv', '.json')
                    settings = None
                    if os.path.isfile(sidecar):
                        try:
                            with open(sidecar) as fh:
                                settings = json.load(fh)
                        except Exception:
                            pass
                    brt_hits.append((csv_path, settings))

            cnt_hits = []
            for wb_dir in wb_dirs:
                for csv_path in _glob.glob(os.path.join(wb_dir, f'{sname}_contour_*.csv')):
                    sidecar = csv_path.replace('.csv', '.json')
                    settings = None
                    if os.path.isfile(sidecar):
                        try:
                            with open(sidecar) as fh:
                                settings = json.load(fh)
                        except Exception:
                            pass
                    cnt_hits.append((csv_path, settings))

            if brt_hits and cnt_hits:
                per_session[sname] = {'brt': brt_hits, 'contour': cnt_hits}

        # ── Build dialog ──────────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Cached Brightness + Contour Files")
        win.resizable(True, True)

        if not per_session:
            dirs_str = '\n'.join(wb_dirs)
            msg = ("No sessions were found that have BOTH brightness and contour "
                   f"cache files under:\n{dirs_str}\n\n"
                   "Use the individual 'Detect cached brightness…' or "
                   "'Detect cached contour…' buttons for sessions with only one type.")
            ttk.Label(win, text=msg, wraplength=860,
                      justify='left', padding=20).pack(fill='both', expand=True)
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)
            return

        # For each session pick the most recent (by mtime) brightness & contour cache
        rows = []  # (sname, brt_csv, brt_settings, cnt_csv, cnt_settings)
        for sname in sorted(per_session):
            brt_list = per_session[sname]['brt']
            cnt_list = per_session[sname]['contour']
            brt_list.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
            cnt_list.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
            rows.append((sname, brt_list[0][0], brt_list[0][1],
                         cnt_list[0][0], cnt_list[0][1]))

        # Scrollable table
        outer = ttk.Frame(win, padding=8)
        outer.pack(fill='both', expand=True)

        cols = ('session', 'brt_thresh', 'brt_roi', 'contour_roi',
                'stride', 'crop', 'shared_match')
        tree = ttk.Treeview(outer, columns=cols, show='headings', height=12)
        headers = [
            ('session',      'Session',      140),
            ('brt_thresh',   'Brt Thresh',    70),
            ('brt_roi',      'Brt ROI',      120),
            ('contour_roi',  'Contour ROI',  120),
            ('stride',       'Stride',        50),
            ('crop',         'Crop',          80),
            ('shared_match', 'Shared Match', 200),
        ]
        for col, hdr, w in headers:
            tree.heading(col, text=hdr)
            tree.column(col, width=w, stretch=(col in ('session', 'shared_match')))

        vsb = ttk.Scrollbar(outer, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(outer, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        tree.tag_configure('match',    foreground='green')
        tree.tag_configure('mismatch', foreground='red')
        tree.tag_configure('unknown',  foreground='gray')

        def _roi_summary(roi_list):
            if not roi_list:
                return '?'
            return ', '.join(f'{bp}={sz}' for bp, sz in roi_list)

        for sname, brt_csv, brt_s, cnt_csv, cnt_s in rows:
            if brt_s is None or cnt_s is None:
                tree.insert('', 'end', tags=('unknown',), values=(
                    sname, '?', '?', '?', '?', '?',
                    'missing sidecar'))
                continue

            brt_thresh = brt_s.get('brt_thresh', '?')
            brt_roi = _roi_summary(brt_s.get('roi_sizes', []))
            cnt_roi = _roi_summary(cnt_s.get('contour_roi_sizes', []))

            # Check shared settings agreement
            shared_fields = ['extraction_stride', 'crop_x', 'crop_y']
            diffs = []
            for field in shared_fields:
                bval = brt_s.get(field, '?')
                cval = cnt_s.get(field, '?')
                if bval != cval:
                    short = field.replace('extraction_', '')
                    diffs.append(f"{short}: {bval}\u2192{cval}")

            if diffs:
                match_str = 'shared mismatch: ' + ', '.join(diffs)
                tag = 'mismatch'
            else:
                match_str = '\u2713'
                tag = 'match'

            stride_val = brt_s.get('extraction_stride', '?')
            crop_val = f"({brt_s.get('crop_x', 0)}, {brt_s.get('crop_y', 0)})"

            tree.insert('', 'end', tags=(tag,), values=(
                sname, brt_thresh, brt_roi, cnt_roi,
                stride_val, crop_val, match_str))

        note = ttk.Label(win,
                         text=("Shows sessions that have BOTH brightness and contour caches. "
                               "Green = shared settings (stride, crop) agree between caches."),
                         font=('Arial', 8), foreground='gray', wraplength=880)
        note.pack(side='bottom', anchor='w', padx=10, pady=(4, 6))

        btn_row = ttk.Frame(win)
        btn_row.pack(side='bottom', pady=6)

        cached_session_names = list(per_session.keys())

        # Collect unique composite settings keys
        def _composite_key(brt_s, cnt_s):
            if brt_s is None or cnt_s is None:
                return None
            return (
                brt_s.get('brt_thresh'),
                brt_s.get('extraction_stride'),
                brt_s.get('crop_x'), brt_s.get('crop_y'),
                tuple(tuple(x) for x in sorted(brt_s.get('roi_sizes', []))),
                tuple(tuple(x) for x in sorted(cnt_s.get('contour_roi_sizes', []))),
            )

        rows_with_sidecars = [(sname, brt_csv, brt_s, cnt_csv, cnt_s)
                              for sname, brt_csv, brt_s, cnt_csv, cnt_s in rows
                              if brt_s is not None and cnt_s is not None]
        unique_keys = list(dict.fromkeys(
            _composite_key(brt_s, cnt_s)
            for _, _, brt_s, _, cnt_s in rows_with_sidecars))
        # Pick the most recent pair for default offer
        if rows_with_sidecars:
            rows_with_sidecars.sort(
                key=lambda x: max(os.path.getmtime(x[1]), os.path.getmtime(x[3])),
                reverse=True)
            offer_brt = rows_with_sidecars[0][2]
            offer_cnt = rows_with_sidecars[0][4]
        else:
            offer_brt = offer_cnt = None

        def _select_sessions():
            iids = [iid for iid in self._sess_tree.get_children()
                    if self._sess_tree.item(iid, 'values')[0] in cached_session_names]
            if iids:
                self._sess_tree.selection_set(iids)

        def _apply_both(brt_s, cnt_s):
            """Apply both brightness and contour settings to UI."""
            bp_to_role = {
                self._role_vars[r].get().strip(): r
                for r in self.ROLES
                if self._role_vars[r].get().strip()}
            # Brightness settings
            if brt_s:
                if 'brt_thresh' in brt_s:
                    self._brt_thresh_var.set(brt_s['brt_thresh'])
                if 'extraction_stride' in brt_s:
                    self._extraction_stride_var.set(brt_s['extraction_stride'])
                if 'crop_x' in brt_s:
                    self._crop_x_var.set(brt_s['crop_x'])
                if 'crop_y' in brt_s:
                    self._crop_y_var.set(brt_s['crop_y'])
                for bp_name, sz in brt_s.get('roi_sizes', []):
                    role = bp_to_role.get(bp_name)
                    if role:
                        self._roi_size_vars[role].set(sz)
                self._use_brightness_var.set(True)
            # Contour settings
            if cnt_s:
                for bp_name, sz in cnt_s.get('contour_roi_sizes', []):
                    role = bp_to_role.get(bp_name)
                    if role:
                        self._contour_roi_size_vars[role].set(sz)
                self._paw_contour_var.set(True)

        def _analyze_cached():
            _apply_both(offer_brt, offer_cnt)
            win.destroy()
            _select_sessions()
            self._start_analysis()

        def _analyze_current():
            win.destroy()
            _select_sessions()
            self._start_analysis()

        if offer_brt and offer_cnt:
            if len(unique_keys) > 1:
                # Build combo selector for multiple setting combinations
                key_to_pair = {}
                for _, _, brt_s, _, cnt_s in rows_with_sidecars:
                    k = _composite_key(brt_s, cnt_s)
                    if k not in key_to_pair:
                        key_to_pair[k] = (brt_s, cnt_s)

                settings_list = []
                summaries = []
                for idx, k in enumerate(unique_keys):
                    bs, cs = key_to_pair[k]
                    t = bs.get('brt_thresh', '?')
                    stride = bs.get('extraction_stride', '?')
                    brt_roi = _roi_summary(bs.get('roi_sizes', []))
                    cnt_roi = _roi_summary(cs.get('contour_roi_sizes', []))
                    cx = bs.get('crop_x', 0)
                    cy = bs.get('crop_y', 0)
                    label = (f"Set {idx+1}: thresh={t}, stride={stride}, "
                             f"brtROI=[{brt_roi}], cntROI=[{cnt_roi}], "
                             f"crop=({cx},{cy})")
                    settings_list.append((bs, cs))
                    summaries.append(label)

                sel_frame = ttk.LabelFrame(
                    win, text="Select settings to apply", padding=5)
                sel_frame.pack(fill='x', padx=10, pady=(4, 2))

                sel_var = tk.StringVar(value=summaries[0])
                sel_combo = ttk.Combobox(
                    sel_frame, textvariable=sel_var,
                    values=summaries, state='readonly', width=80)
                sel_combo.pack(fill='x', pady=2)

                def _analyze_selected():
                    idx = (summaries.index(sel_var.get())
                           if sel_var.get() in summaries else 0)
                    bs, cs = settings_list[idx]
                    _apply_both(bs, cs)
                    win.destroy()
                    _select_sessions()
                    self._start_analysis()

                ttk.Button(btn_row, text="Analyze with Selected Settings",
                           command=_analyze_selected).pack(side='left', padx=6)
            else:
                ttk.Button(btn_row, text="Analyze with Cached Settings",
                           command=_analyze_cached).pack(side='left', padx=6)

        ttk.Button(btn_row, text="Analyze with Current Settings",
                   command=_analyze_current).pack(side='left', padx=6)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side='left', padx=6)
        _autofit_treeview_window(win, tree)

    # ── Right: Results ───────────────────────────────────────────────────────

    def _build_results_panel(self, parent):
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill='x', padx=4, pady=(4, 2))

        self._export_sum_btn = ttk.Button(btn_row, text="Export Summary CSV",
                                          command=self._export_summary,
                                          state='disabled')
        self._export_sum_btn.pack(side='left', padx=2)
        self._export_bin_btn = ttk.Button(btn_row, text="Export Bins CSV",
                                          command=self._export_bins,
                                          state='disabled')
        self._export_bin_btn.pack(side='left', padx=2)
        self._graphs_btn = ttk.Button(btn_row, text="Graphs",
                                      command=self._open_graphs,
                                      state='disabled')
        self._graphs_btn.pack(side='left', padx=2)
        self._adjust_contact_btn = ttk.Button(
            btn_row, text="Adjust Contact",
            command=self._open_contact_adjustment, state='disabled')
        self._adjust_contact_btn.pack(side='left', padx=2)

        # ── Key Metrics Summary ───────────────────────────────────
        self._summary_frame = ttk.LabelFrame(parent, text="Summary", padding=5)
        self._summary_frame.pack(fill='x', padx=4, pady=(0, 4))
        self._summary_placeholder = ttk.Label(
            self._summary_frame, text="No results yet",
            foreground='grey', font=('TkDefaultFont', 9))
        self._summary_placeholder.pack(anchor='w')

        # Results treeview
        res_lf = ttk.LabelFrame(parent, text="Results", padding=3)
        res_lf.pack(fill='both', expand=True, padx=4, pady=(0, 2))

        tree_frame = ttk.Frame(res_lf)
        tree_frame.pack(fill='both', expand=True)

        res_cols = ('session', 'subject', 'treatment',
                    'wbi_h', 'si_h', 'wbi_f', 'si_f', 'brt',
                    'hl_c', 'hr_c', 'fl_c', 'fr_c',
                    'stance_h', 'stance_r', 'duty_h', 'duty_r', 'stride_l')
        self._res_tree = ttk.Treeview(tree_frame, columns=res_cols,
                                      show='headings', height=14)
        hdrs = [
            ('session',   'Session',   130),
            ('subject',   'Subject',    70),
            ('treatment', 'Treatment',  80),
            ('wbi_h',     'WBI hind',   68),
            ('si_h',      'SI hind',    60),
            ('wbi_f',     'WBI fore',   62),
            ('si_f',      'SI fore',    58),
            ('brt',       'Brt ratio',  62),
            ('hl_c',      'HL %',       52),
            ('hr_c',      'HR %',       52),
            ('fl_c',      'FL %',       52),
            ('fr_c',      'FR %',       52),
            ('stance_h',  'Stance HL',  58),
            ('stance_r',  'Stance HR',  58),
            ('duty_h',    'Duty HL',    52),
            ('duty_r',    'Duty HR',    52),
            ('stride_l',  'Stride L',   55),
        ]
        for col, hdr, w in hdrs:
            self._res_tree.heading(col, text=hdr)
            self._res_tree.column(col, width=w, stretch=(col == 'session'))

        # Column-header mouseover tooltips
        _COL_TIPS = {
            'session':   None,
            'subject':   None,
            'treatment': None,
            'wbi_h':  'WBI hind\nHL / (HL+HR) × 100\n50 = symmetric; >50 = more weight left hind.',
            'si_h':   'SI hind\n(HL−HR) / (HL+HR) × 100\n0 = symmetric; positive = left bias.',
            'wbi_f':  'WBI fore\nFL / (FL+FR) × 100\n50 = symmetric; >50 = more weight left fore.',
            'si_f':   'SI fore\n(FL−FR) / (FL+FR) × 100.',
            'brt':    'Brightness ratio\nMean ROI brightness HL ÷ HR during contact frames.',
            'hl_c':   'Contact %\nFrames paw in stance (hind left).',
            'hr_c':   'Contact %\nFrames paw in stance (hind right).',
            'fl_c':   'Contact %\nFrames paw in stance (fore left).',
            'fr_c':   'Contact %\nFrames paw in stance (fore right).',
            'stance_h': 'Mean stance duration (s) — hind left.',
            'stance_r': 'Mean stance duration (s) — hind right.',
            'duty_h':   'Duty cycle (%) — stance / stride × 100 — hind left.',
            'duty_r':   'Duty cycle (%) — stance / stride × 100 — hind right.',
            'stride_l': 'Mean stride length (px) — hind left.',
        }
        _RES_COLS = ('session', 'subject', 'treatment',
                     'wbi_h', 'si_h', 'wbi_f', 'si_f', 'brt',
                     'hl_c', 'hr_c', 'fl_c', 'fr_c',
                     'stance_h', 'stance_r', 'duty_h', 'duty_r', 'stride_l')
        _tree_tip = [None]

        def _on_tree_motion(event):
            region = self._res_tree.identify_region(event.x, event.y)
            if region != 'heading':
                if _tree_tip[0]:
                    _tree_tip[0].destroy()
                    _tree_tip[0] = None
                return
            col_idx = int(self._res_tree.identify_column(event.x).lstrip('#')) - 1
            try:
                col_name = _RES_COLS[col_idx]
            except IndexError:
                return
            tip_text = _COL_TIPS.get(col_name)
            if not tip_text:
                if _tree_tip[0]:
                    _tree_tip[0].destroy()
                    _tree_tip[0] = None
                return
            if _tree_tip[0]:
                _tree_tip[0].destroy()
            tip = tk.Toplevel(self._res_tree)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(
                f'+{self._res_tree.winfo_rootx() + event.x + 12}'
                f'+{self._res_tree.winfo_rooty() + event.y + 16}')
            tk.Label(tip, text=tip_text, background='#ffffcc', relief='solid',
                     borderwidth=1, font=('Arial', 9), wraplength=280,
                     justify='left').pack(ipadx=4, ipady=2)
            _tree_tip[0] = tip

        def _on_tree_leave(event):
            if _tree_tip[0]:
                _tree_tip[0].destroy()
                _tree_tip[0] = None

        self._res_tree.bind('<Motion>', _on_tree_motion)
        self._res_tree.bind('<Leave>',  _on_tree_leave)

        res_vsb = ttk.Scrollbar(tree_frame, orient='vertical',
                                command=self._res_tree.yview)
        res_hsb = ttk.Scrollbar(tree_frame, orient='horizontal',
                                command=self._res_tree.xview)
        self._res_tree.config(yscrollcommand=res_vsb.set,
                              xscrollcommand=res_hsb.set)
        self._res_tree.grid(row=0, column=0, sticky='nsew')
        res_vsb.grid(row=0, column=1, sticky='ns')
        res_hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Log
        log_lf = ttk.LabelFrame(parent, text="Log", padding=3)
        log_lf.pack(fill='x', padx=4, pady=(0, 2))
        self._log_text = tk.Text(log_lf, height=3, wrap='word',
                                 state='disabled', font=('Consolas', 8))
        log_sb = ttk.Scrollbar(log_lf, orient='vertical',
                               command=self._log_text.yview)
        self._log_text.config(yscrollcommand=log_sb.set)
        self._log_text.pack(side='left', fill='both', expand=True)
        log_sb.pack(side='right', fill='y')

        # ── Progress bar (hidden until analysis runs) ─────────────
        self._progress_frame = ttk.Frame(parent)
        # not packed yet — shown/hidden by _start_analysis / _on_analysis_complete
        self._progress = ttk.Progressbar(self._progress_frame, mode='determinate')
        self._progress.pack(fill='x', padx=4, pady=(0, 1))
        self._sub_progress_label = ttk.Label(
            self._progress_frame, text="", font=('TkDefaultFont', 8))
        self._sub_progress_label.pack(fill='x', padx=4, pady=(0, 2))

    # ═══════════════════════════════════════════════════════════════════════
    # Summary panel refresh
    # ═══════════════════════════════════════════════════════════════════════

    def _refresh_summary_panel(self):
        """Populate the summary frame with key metrics (mean +/- SEM)."""
        for w in self._summary_frame.winfo_children():
            w.destroy()

        if self._summary_df is None or self._summary_df.empty:
            ttk.Label(self._summary_frame, text="No results yet",
                      foreground='grey', font=('TkDefaultFont', 9)).pack(anchor='w')
            return

        df = self._summary_df
        metrics = [
            ('WBI hind',   'WBI_hind',              '50 = sym'),
            ('SI hind',    'SI_hind',                '0 = sym'),
            ('WBI fore',   'WBI_fore',              '50 = sym'),
            ('SI fore',    'SI_fore',                '0 = sym'),
            ('Contact HL', 'contact_pct_HL',         '%'),
            ('Contact HR', 'contact_pct_HR',         '%'),
            ('Brt ratio',  'brightness_ratio_HL_HR', '1.0 = sym'),
        ]

        has_treatment = ('treatment' in df.columns
                         and df['treatment'].notna().any()
                         and df['treatment'].ne('').any())
        if has_treatment:
            groups = [(t, df[df['treatment'] == t])
                      for t in df['treatment'].dropna().unique()
                      if str(t).strip()]
        else:
            groups = [('All', df)]

        row_idx = 0
        # Header row
        ttk.Label(self._summary_frame, text="Metric",
                  font=('TkDefaultFont', 8, 'bold')).grid(
            row=row_idx, column=0, sticky='w', padx=(0, 8))
        for ci, (gname, _) in enumerate(groups):
            ttk.Label(self._summary_frame, text=str(gname),
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=row_idx, column=ci + 1, sticky='e', padx=(4, 8))
        row_idx += 1

        ttk.Separator(self._summary_frame, orient='horizontal').grid(
            row=row_idx, column=0, columnspan=len(groups) + 1,
            sticky='ew', pady=2)
        row_idx += 1

        for label, col, ref in metrics:
            if col not in df.columns:
                continue
            ttk.Label(self._summary_frame, text=label,
                      font=('TkDefaultFont', 8)).grid(
                row=row_idx, column=0, sticky='w', padx=(0, 8))
            for ci, (_, gdf) in enumerate(groups):
                vals = gdf[col].dropna()
                if len(vals) == 0:
                    txt = '--'
                elif len(vals) == 1:
                    txt = f'{vals.iloc[0]:.1f}'
                else:
                    m = vals.mean()
                    sem = (vals.std(ddof=1) / np.sqrt(len(vals)))
                    txt = f'{m:.1f} \u00b1 {sem:.1f}'
                ttk.Label(self._summary_frame, text=txt,
                          font=('TkDefaultFont', 8)).grid(
                    row=row_idx, column=ci + 1, sticky='e', padx=(4, 8))
            row_idx += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Tooltip helper
    # ═══════════════════════════════════════════════════════════════════════

    def _tip(self, widget, text):
        _ToolTip(widget, text)

    # ═══════════════════════════════════════════════════════════════════════
    # Fore-paw toggle
    # ═══════════════════════════════════════════════════════════════════════

    def _on_use_fore_changed(self):
        enabled = self._use_fore_var.get()
        state = 'normal' if enabled else 'disabled'
        for role in ('FL', 'FR'):
            self._role_combos[role].configure(state=state)
        if not enabled:
            for role in ('FL', 'FR'):
                self._role_vars[role].set('')

    # ═══════════════════════════════════════════════════════════════════════
    # Thread-safe logging
    # ═══════════════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.app.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg: str):
        self._log_text.config(state='normal')
        self._log_text.insert('end', msg + '\n')
        self._log_text.see('end')
        self._log_text.config(state='disabled')

    # ═══════════════════════════════════════════════════════════════════════
    # Project folder integration
    # ═══════════════════════════════════════════════════════════════════════

    def on_project_changed(self):
        """Called by PixelPawsGUI._on_project_folder_changed."""
        self._scan_sessions()

    # ═══════════════════════════════════════════════════════════════════════
    # Session scanning
    # ═══════════════════════════════════════════════════════════════════════

    def _browse_sessions_folder(self):
        folder = filedialog.askdirectory(title="Select sessions folder")
        if folder:
            self._override_folder_var.set(folder)
            self._scan_sessions()

    def _scan_sessions(self):
        folder = self._override_folder_var.get() or self.app.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            return

        try:
            self._sessions = find_session_triplets(folder, require_labels=False)
        except Exception as e:
            self._log_ui(f"Session scan error: {e}")
            self._sessions = []

        for item in self._sess_tree.get_children():
            self._sess_tree.delete(item)

        # Auto-detect body parts from first available DLC h5
        for sess in self._sessions:
            if sess.get('dlc') and os.path.isfile(sess['dlc']):
                self._auto_populate_bodyparts(sess['dlc'])
                break

        for sess in self._sessions:
            subj = self._resolve_subject(sess['session_name'])
            has_vid = '✓' if (sess.get('video') and os.path.isfile(sess['video'])) else '✗'
            self._sess_tree.insert('', 'end',
                                   values=(sess['session_name'], subj, has_vid))

        n = len(self._sessions)
        self._sess_lbl.config(text=f'{n} session{"s" if n != 1 else ""} found')
        self._folder_lbl.config(
            text=os.path.basename(folder) if folder else '')

        self._scan_key_files(folder)

    def _select_all(self):
        self._sess_tree.selection_set(self._sess_tree.get_children())

    def _clear_selection(self):
        self._sess_tree.selection_remove(self._sess_tree.get_children())

    # ═══════════════════════════════════════════════════════════════════════
    # Body part detection
    # ═══════════════════════════════════════════════════════════════════════

    def _auto_populate_bodyparts(self, h5_path: str):
        """Detect body parts from a DLC h5 file and fill paw comboboxes."""
        try:
            df = pd.read_hdf(h5_path)
            if isinstance(df.columns, pd.MultiIndex):
                bps = list(df.columns.get_level_values(1).unique())
            else:
                bps = list({c.rsplit('_', 1)[0] for c in df.columns
                            if c.endswith(('_x', '_y'))})
            self._bodyparts = sorted(bps)
        except Exception:
            self._bodyparts = []

        for cb in self._role_combos.values():
            cb.config(values=self._bodyparts)

    def _autodetect_bodyparts(self):
        """Button: re-scan first DLC h5 and fill combos."""
        for sess in self._sessions:
            if sess.get('dlc') and os.path.isfile(sess['dlc']):
                self._auto_populate_bodyparts(sess['dlc'])
                self._log_ui(f"Detected {len(self._bodyparts)} body parts")
                return
        messagebox.showinfo("No sessions", "Scan sessions first.", parent=self)

    # ═══════════════════════════════════════════════════════════════════════
    # Key file handling
    # ═══════════════════════════════════════════════════════════════════════

    def _scan_key_files(self, folder: str):
        """Walk project folder for CSV/XLSX files with Subject+Treatment cols."""
        _SKIP    = {'__pycache__', '.git', '.claude', 'node_modules', '.idea'}
        _PRED_KW = ('prediction', 'predictions', 'pred', 'bout', 'bouts')
        candidates = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in sorted(dirs)
                       if d not in _SKIP and not d.startswith('.')]
            for fname in files:
                fl = fname.lower()
                if not fl.endswith(('.csv', '.xlsx')):
                    continue
                if any(kw in fl for kw in _PRED_KW):
                    continue
                full = os.path.join(root, fname)
                try:
                    if full.endswith('.xlsx'):
                        cols = pd.read_excel(full, nrows=0).columns.tolist()
                    else:
                        cols = pd.read_csv(full, nrows=0).columns.tolist()
                    if 'Subject' in cols and 'Treatment' in cols:
                        candidates.append(full)
                except Exception:
                    pass

        self._key_scan_paths = candidates
        labels = [os.path.relpath(p, folder).replace(os.sep, '/') for p in candidates]
        self._key_combo.config(values=labels)
        if len(candidates) == 1 and not self._key_file_var.get():
            self._key_combo.current(0)
            self._on_key_combo_selected()

    def _on_key_combo_selected(self, event=None):
        idx = self._key_combo.current()
        if 0 <= idx < len(self._key_scan_paths):
            self._load_key_file(self._key_scan_paths[idx])

    def _browse_dlc_config(self):
        path = filedialog.askopenfilename(
            title="Select DLC config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self._dlc_config_var.set(path)
            self._detect_crop_from_config()

    def _detect_crop_from_config(self):
        path = self._dlc_config_var.get()
        if not path or not os.path.isfile(path):
            return
        try:
            import yaml
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
            x1 = int(cfg.get('x1', 0) or 0)
            y1 = int(cfg.get('y1', 0) or 0)
            self._crop_x_var.set(x1)
            self._crop_y_var.set(y1)
            self._log_ui(f"Crop offset detected: x+{x1}, y+{y1}")
        except Exception as e:
            self._log_ui(f"Could not read crop from config: {e}")

    def _browse_key_file(self):
        path = filedialog.askopenfilename(
            title="Select Key File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            parent=self)
        if path:
            self._key_file_var.set(path)
            self._load_key_file(path)

    def _generate_key_file(self):
        """Open KeyFileGeneratorDialog to create a key_file.csv for this project."""
        try:
            from project_setup import KeyFileGeneratorDialog
        except ImportError:
            messagebox.showerror("Unavailable",
                                 "project_setup.py not found — cannot open key file generator.",
                                 parent=self)
            return

        folder = getattr(self.app, 'current_project_folder', None)
        folder = folder.get() if folder else ''
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Project",
                                   "Please open a project first so PixelPaws knows "
                                   "where to find your videos and save the key file.",
                                   parent=self)
            return

        import glob as _g
        videos_dir = os.path.join(folder, 'videos')
        _seen = {}
        for ext in ('.mp4', '.avi', '.mov', '.wmv', '.MP4', '.AVI', '.MOV', '.WMV'):
            for vf in _g.glob(os.path.join(videos_dir, f'*{ext}')):
                _seen[os.path.normcase(vf)] = vf
        basenames = [os.path.splitext(os.path.basename(v))[0]
                     for v in sorted(_seen.values())]
        if not basenames:
            messagebox.showinfo("No Videos",
                                "No video files found in videos/.\n"
                                "Add your videos first, then generate the key file.",
                                parent=self)
            return

        existing = {}
        key_path = os.path.join(folder, 'key_file.csv')
        if os.path.isfile(key_path):
            try:
                import csv
                with open(key_path, newline='') as f:
                    for row in csv.DictReader(f):
                        s = row.get('Subject', '').strip()
                        t = row.get('Treatment', '').strip()
                        if s:
                            existing[s] = t
            except Exception:
                pass

        def _on_save(data):
            if hasattr(self.app, 'key_file_data'):
                self.app.key_file_data = data
            saved_path = os.path.join(folder, 'key_file.csv')
            if os.path.isfile(saved_path):
                self._key_file_var.set(saved_path)
                self._load_key_file(saved_path)

        KeyFileGeneratorDialog(
            self.winfo_toplevel(), folder, basenames,
            existing_groups=existing, on_save=_on_save)

    def _load_key_file(self, path: str):
        try:
            df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Key file error", str(e), parent=self)
            return
        missing = [c for c in ('Subject', 'Treatment') if c not in df.columns]
        if missing:
            messagebox.showerror("Invalid key file",
                                 f"Missing columns: {', '.join(missing)}", parent=self)
            return
        df['Subject'] = df['Subject'].astype(str)
        self._key_df = df
        n_subj = len(df)
        treatments = df['Treatment'].unique()
        self._key_status_lbl.config(
            text=(f"✓ {n_subj} subjects, {len(treatments)} treatment(s): "
                  f"{', '.join(map(str, treatments))}"),
            foreground='green')
        self._log_ui(f"Key file loaded: {os.path.basename(path)}")

        # Refresh subject column in session tree now that key is known
        for item in self._sess_tree.get_children():
            vals = list(self._sess_tree.item(item, 'values'))
            vals[1] = self._resolve_subject(vals[0])
            self._sess_tree.item(item, values=vals)

    # ═══════════════════════════════════════════════════════════════════════
    # Subject / treatment resolution
    # ═══════════════════════════════════════════════════════════════════════

    def _resolve_subject(self, session_name: str) -> str:
        """Extract subject ID via 4-strategy fallback (mirrors analysis_tab)."""
        stem = session_name

        # 1. Key-file token match
        if self._key_df is not None:
            tokens = stem.split('_')
            for subj in self._key_df['Subject']:
                if str(subj) in tokens:
                    return str(subj)
            for subj in self._key_df['Subject']:
                if f'_{subj}_' in f'_{stem}_':
                    return str(subj)

        # 2. Prefix strip
        pfx = self._prefix_var.get().strip()
        if pfx and stem.startswith(pfx):
            remainder = stem[len(pfx):]
            token = remainder.split('_')[0] if remainder else ''
            if token:
                return token

        # 3. PixelPaws_GUI legacy helper
        if _extract_sid is not None:
            sid = _extract_sid(session_name)
            if sid:
                return str(sid)

        # 4. First 4-digit token heuristic
        for token in stem.split('_'):
            if re.match(r'^\d{4}$', token):
                return token

        return stem

    def _get_treatment(self, subject: str) -> str:
        if self._key_df is None:
            return ''
        row = self._key_df[self._key_df['Subject'] == str(subject)]
        return str(row.iloc[0]['Treatment']) if not row.empty else ''

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis: launch / cancel
    # ═══════════════════════════════════════════════════════════════════════

    def _start_analysis(self):
        if self._fit_thread and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis is already running.", parent=self)
            return

        selected_items = self._sess_tree.selection()
        if not selected_items:
            messagebox.showwarning("No sessions",
                                   "Select at least one session.", parent=self)
            return

        selected_names = {self._sess_tree.item(i, 'values')[0]
                         for i in selected_items}
        sessions = [s for s in self._sessions
                    if s['session_name'] in selected_names]

        paw_map = {role: self._role_vars[role].get().strip()
                   for role in self.ROLES}
        if not paw_map.get('HL') or not paw_map.get('HR'):
            messagebox.showerror("Paw mapping",
                                 "Configure at least HL and HR body parts.",
                                 parent=self)
            return

        # Validate brightness requires video
        if self._use_brightness_var.get():
            no_video = [s['session_name'] for s in sessions if not s.get('video_path')]
            if no_video:
                msg = ("Brightness extraction is enabled but these sessions have no video:\n\n"
                       + "\n".join(no_video[:10])
                       + ("\n..." if len(no_video) > 10 else "")
                       + "\n\nBrightness will be skipped for those sessions.")
                messagebox.showwarning("Missing videos", msg, parent=self)

        # Parse speed threshold
        speed_thresh_raw = self._speed_thresh_var.get().strip()
        speed_thresh = 'auto'
        if speed_thresh_raw.lower() != 'auto':
            try:
                speed_thresh = float(speed_thresh_raw)
            except ValueError:
                speed_thresh = 'auto'

        params = {
            'contact_threshold': self._contact_thresh_var.get(),
            'height_window':     self._height_window_var.get(),
            'bin_seconds':       self._bin_seconds_var.get(),
            'bin_unit':          self._bin_unit_var.get(),
            'fallback_fps':      float(self._fallback_fps_var.get()),
            'use_brightness':    self._use_brightness_var.get(),
            'brt_threshold':     self._brt_thresh_var.get(),
            'brt_weight':        self._brt_weight_var.get(),
            'roi_sizes':         {role: self._roi_size_vars[role].get()
                                  for role in self.ROLES},
            'crop_offset_x':      self._crop_x_var.get(),
            'crop_offset_y':      self._crop_y_var.get(),
            'extraction_stride':  self._extraction_stride_var.get(),
            'contact_method':     self._contact_method_var.get(),
            'speed_threshold':    speed_thresh,
            'median_filter_ms':   self._median_filter_var.get(),
            'min_bout_ms':        self._min_bout_var.get(),
            'min_stance_ms':      self._min_stance_ms_var.get(),
            'use_likelihood':     self._use_likelihood_var.get(),
            'likelihood_threshold': self._likelihood_thresh_var.get(),
            'loco_filter':        self._loco_filter_var.get(),
            'loco_threshold':     self._loco_thresh_var.get(),
            'paw_contour':        self._paw_contour_var.get(),
            'contour_roi_sizes':  {role: self._contour_roi_size_vars[role].get()
                                   for role in self.ROLES},
            'contour_forelimbs':  self._contour_forelimbs_var.get(),
        }

        self._cancel_flag.clear()
        self._run_btn.config(state='disabled')
        self._cancel_btn.config(state='normal')
        self._progress.config(maximum=max(len(sessions), 1), value=0)
        self._sub_progress_label.config(text='')
        self._progress_frame.pack(fill='x', padx=4, pady=(0, 4))
        self._export_sum_btn.config(state='disabled')
        self._export_bin_btn.config(state='disabled')
        self._graphs_btn.config(state='disabled')

        self._fit_thread = threading.Thread(
            target=self._analysis_thread,
            args=(sessions, paw_map, params),
            daemon=True)
        self._fit_thread.start()

    def _cancel_analysis(self):
        self._cancel_flag.set()
        self._log("Cancelling…")

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis: background thread
    # ═══════════════════════════════════════════════════════════════════════

    def _analysis_thread(self, sessions, paw_map, params):
        summary_rows = []
        bin_rows = []
        for sess in sessions:
            if self._cancel_flag.is_set():
                self._log("Cancelled.")
                break
            name = sess['session_name']
            self._log(f"Processing: {name}")
            try:
                result = self._analyze_session(sess, paw_map, params)
                if result:
                    subj      = self._resolve_subject(name)
                    treatment = self._get_treatment(subj)
                    base = dict(session=name, subject=subj, treatment=treatment)
                    srow = {**base, **result['summary']}
                    summary_rows.append(srow)
                    for brow in result['bins']:
                        bin_rows.append({**base, **brow})
            except Exception as e:
                self._log(f"  ERROR: {e}")
            try:
                self.app.root.after(0, self._progress.step, 1)
            except tk.TclError:
                pass

        self._log(f"Analysis loop finished: {len(summary_rows)}/{len(sessions)} sessions produced results.")

        try:
            self.app.root.after(0, self._on_analysis_complete, summary_rows, bin_rows)
        except tk.TclError:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # Speed-based contact detection  (Kumar Lab, Cell Reports 2022)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_speed_contact(paw_x, paw_y, fps,
                               threshold='auto',
                               median_ms=50,
                               min_bout_ms=30):
        """Return boolean stance mask using paw speed thresholding.

        Parameters
        ----------
        paw_x, paw_y : array-like  — DLC x,y coordinates for one paw
        fps           : float       — video frame rate
        threshold     : float|'auto' — speed cutoff (px/s); 'auto' = 20th pctile
        median_ms     : int          — median filter window in ms
        min_bout_ms   : int          — debounce: remove bouts shorter than this

        Returns
        -------
        np.ndarray[bool] — True = stance (contact), False = swing
        """
        x = np.asarray(paw_x, dtype=float)
        y = np.asarray(paw_y, dtype=float)
        n = len(x)
        if n < 2:
            return np.ones(n, dtype=bool)

        # Frame-to-frame speed in px/s
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx**2 + dy**2) * fps

        # Median filter (smooth jitter)
        if _SCIPY_NDIMAGE_OK and median_ms > 0:
            win = max(1, round(median_ms / 1000.0 * fps))
            if win % 2 == 0:
                win += 1  # median_filter needs odd window
            speed = _median_filter(speed, size=win)

        # Threshold
        if threshold == 'auto' or threshold is None:
            threshold = float(np.percentile(speed, 20))
        else:
            threshold = float(threshold)

        stance = speed < threshold

        # Debounce: remove stance/swing bouts shorter than min_bout_ms
        if min_bout_ms > 0:
            min_frames = max(1, round(min_bout_ms / 1000.0 * fps))
            stance = GaitLimbTab._debounce(stance, min_frames)

        return stance

    @staticmethod
    def _debounce(mask, min_frames):
        """Remove boolean runs shorter than min_frames."""
        out = mask.copy()
        changes = np.diff(out.astype(int), prepend=int(out[0]) ^ 1)
        starts = np.where(changes != 0)[0]
        ends = np.append(starts[1:], len(out))
        for s, e in zip(starts, ends):
            if (e - s) < min_frames:
                out[s:e] = not out[s]  # flip short bout to surrounding state
        return out

    # ═══════════════════════════════════════════════════════════════════════
    # Gait bout extraction helper
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gait_bouts(mask, fps, min_stride_ms=0):
        """Find onset/offset of stance (True) and swing (False) runs.

        Parameters
        ----------
        mask : array-like of bool — per-frame contact mask
        fps : float — frames per second
        min_stride_ms : float — minimum stance bout duration in ms;
            bouts shorter than this are discarded as noise (default 0 = no filter)

        Returns
        -------
        stance_durs : list[float] — durations of stance bouts in seconds
        swing_durs  : list[float] — durations of swing bouts in seconds
        stance_onsets : list[int] — frame indices where stance begins
        """
        n = len(mask)
        if n == 0:
            return [], [], []
        m = np.asarray(mask, dtype=bool)
        d = np.diff(m.astype(int), prepend=int(~m[0]))
        raw_stance_onsets = np.where(d == 1)[0]   # swing→stance transitions
        stance_offsets = np.where(d == -1)[0]      # stance→swing transitions

        # Ensure balanced pairs
        if len(raw_stance_onsets) == 0:
            # All stance or all swing
            if m[0]:
                return [n / fps], [], [0]
            else:
                return [], [n / fps], []

        min_frames = max(1, round(min_stride_ms / 1000.0 * fps)) if min_stride_ms > 0 else 0

        # Build stance bouts and filter by minimum duration
        stance_onsets = []
        stance_durs = []
        for i, on in enumerate(raw_stance_onsets):
            idx = np.searchsorted(stance_offsets, on, side='right')
            off = int(stance_offsets[idx]) if idx < len(stance_offsets) else n
            bout_len = off - on
            if min_frames > 0 and bout_len < min_frames:
                continue
            stance_onsets.append(int(on))
            stance_durs.append(bout_len / fps)

        if not stance_onsets:
            return [], [], []

        # Swing durations: gap between end of one stance and start of the next
        swing_durs = []
        stance_onsets_arr = np.array(stance_onsets)
        for i in range(len(stance_onsets) - 1):
            # Find the offset of the current stance bout
            idx = np.searchsorted(stance_offsets, stance_onsets[i], side='right')
            off = int(stance_offsets[idx]) if idx < len(stance_offsets) else n
            swing_dur = (stance_onsets[i + 1] - off) / fps
            if swing_dur > 0:
                swing_durs.append(swing_dur)

        return stance_durs, swing_durs, stance_onsets

    @staticmethod
    def _regularity_index(masks, fps, frame_slice, loco_filter_mask,
                          confidence_mask, min_stance_ms=0):
        """Compute regularity index (RI) — percentage of normal step sequences.

        RI = (NSSP × 4 / total_paw_placements) × 100
        where NSSP = number of normal step-sequence patterns.
        Normal patterns: any cycle of 4 consecutive paw placements where
        all 4 paws appear exactly once.

        Returns float or None if insufficient data.
        """
        if not all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
            return None

        # Get stance onsets for all 4 paws
        all_onsets = []
        for role in ('HL', 'HR', 'FL', 'FR'):
            mask_arr = (masks[role].values.astype(bool) if hasattr(masks[role], 'values')
                        else np.asarray(masks[role], dtype=bool))
            if loco_filter_mask is not None:
                lm = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                mask_arr = mask_arr & lm[:len(mask_arr)]
            if confidence_mask is not None:
                cm = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                mask_arr = mask_arr & cm[:len(mask_arr)]
            _, _, onsets = GaitLimbTab._gait_bouts(mask_arr, fps, min_stance_ms)
            for o in onsets:
                all_onsets.append((o, role))

        if len(all_onsets) < 8:
            return None

        # Sort by frame
        all_onsets.sort(key=lambda x: x[0])
        total_placements = len(all_onsets)

        # Slide a window of 4 and check if all 4 paws are represented
        nssp = 0
        for i in range(0, len(all_onsets) - 3, 4):
            group_roles = {all_onsets[i + k][1] for k in range(4)}
            if len(group_roles) == 4:
                nssp += 1

        if total_placements < 4:
            return None

        ri = round(nssp * 4 / total_placements * 100, 2)
        return ri

    @staticmethod
    def _print_position(masks, paw_xy, hind_role, fore_role, fps, frame_slice,
                        loco_filter_mask, confidence_mask, min_stance_ms=0):
        """Compute print position — distance between hind paw strike and
        most recent ipsilateral fore paw strike position (px).

        Returns float (mean distance in px) or NaN.
        """
        # Get stance onsets for hind and fore paw
        for role in (hind_role, fore_role):
            if role not in masks or role not in paw_xy:
                return float('nan')

        def _get_onsets(role):
            mask_arr = (masks[role].values.astype(bool) if hasattr(masks[role], 'values')
                        else np.asarray(masks[role], dtype=bool))
            if loco_filter_mask is not None:
                lm = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                mask_arr = mask_arr & lm[:len(mask_arr)]
            if confidence_mask is not None:
                cm = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                mask_arr = mask_arr & cm[:len(mask_arr)]
            _, _, onsets = GaitLimbTab._gait_bouts(mask_arr, fps, min_stance_ms)
            return onsets

        hind_onsets = _get_onsets(hind_role)
        fore_onsets = _get_onsets(fore_role)

        if len(fore_onsets) < 2 or len(hind_onsets) < 1:
            return float('nan')

        hpx, hpy = paw_xy[hind_role]
        fpx, fpy = paw_xy[fore_role]
        sl_start = (frame_slice.start or 0) if frame_slice is not None else 0

        fore_arr = np.array(fore_onsets)
        dists = []
        for ho in hind_onsets:
            abs_ho = sl_start + ho
            # Find the most recent fore onset before this hind onset
            idx = np.searchsorted(fore_arr, ho, side='left') - 1
            if idx < 0:
                continue
            fo = fore_onsets[idx]
            abs_fo = sl_start + fo
            if abs_ho < len(hpx) and abs_fo < len(fpx):
                d = np.sqrt((hpx[abs_ho] - fpx[abs_fo])**2 +
                            (hpy[abs_ho] - fpy[abs_fo])**2)
                dists.append(d)

        return round(float(np.mean(dists)), 2) if dists else float('nan')

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis: per-session core
    # ═══════════════════════════════════════════════════════════════════════

    def _brt_cache_path(self, session_name: str, cache_key: dict):
        """Return path for brightness series cache CSV, or None if unavailable."""
        import hashlib, json
        folder = self.app.current_project_folder.get()
        if not folder:
            return None
        h = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]
        fname = f'{session_name}_brt_{h}.csv'
        # Check legacy directory first
        legacy = os.path.join(folder, 'weight_bearing_analysis', fname)
        if os.path.isfile(legacy):
            return legacy
        feat_dir = os.path.join(folder, 'gait_limb_analysis')
        os.makedirs(feat_dir, exist_ok=True)
        return os.path.join(feat_dir, fname)

    def _contour_cache_path(self, session_name: str, cache_key: dict):
        """Return path for paw contour cache CSV, or None if unavailable."""
        import hashlib, json
        folder = self.app.current_project_folder.get()
        if not folder:
            return None
        h = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]
        fname = f'{session_name}_contour_{h}.csv'
        # Check legacy directory first
        legacy = os.path.join(folder, 'weight_bearing_analysis', fname)
        if os.path.isfile(legacy):
            return legacy
        feat_dir = os.path.join(folder, 'gait_limb_analysis')
        os.makedirs(feat_dir, exist_ok=True)
        return os.path.join(feat_dir, fname)

    def _analyze_session(self, sess, paw_map, params):
        """
        Returns {'summary': dict, 'bins': list_of_dicts}, or None on failure.
        """
        dlc_file   = sess.get('dlc')
        video_file = sess.get('video')

        if not dlc_file or not os.path.isfile(dlc_file):
            self._log("  Skipped (no DLC file)")
            return None

        active_paws = {role: bp for role, bp in paw_map.items() if bp}
        active_bps  = list(set(active_paws.values()))

        # ── Paw heights + coordinates ───────────────────────────────────────
        extractor = PoseFeatureExtractor(active_bps)
        try:
            dlc_df = extractor.load_dlc_data(dlc_file)
            bp_xcord, bp_ycord, bp_prob = extractor.get_bodypart_coords(dlc_df)
            height_df = extractor.calculate_paw_height(
                bp_xcord, bp_ycord, window=params['height_window'])
        except Exception as e:
            self._log(f"  DLC error: {e}")
            return None

        n_frames = len(height_df)

        # ── FPS ──────────────────────────────────────────────────────────────
        fps = params['fallback_fps']
        _used_fallback_fps = True
        if video_file and os.path.isfile(video_file) and _CV2_OK:
            try:
                cap = cv2.VideoCapture(video_file)
                fps_v = cap.get(cv2.CAP_PROP_FPS)
                if fps_v > 0:
                    fps = fps_v
                    _used_fallback_fps = False
                cap.release()
            except Exception as _vid_err:
                print(f"Warning: could not read FPS from {video_file}: {_vid_err}")

        # ── DLC confidence mask (Step 6) ─────────────────────────────────────
        confidence_mask = None
        if params.get('use_likelihood') and bp_prob is not None:
            lk_thresh = params.get('likelihood_threshold', 0.6)
            # Per-paw confidence: mark frame as low-confidence if ANY
            # active paw is below threshold
            paw_ok = np.ones(n_frames, dtype=bool)
            for role, bp in active_paws.items():
                prob_col = next((c for c in bp_prob.columns
                                 if bp.lower() in c.lower()), None)
                if prob_col is not None:
                    paw_ok &= (bp_prob[prob_col].values[:n_frames] >= lk_thresh)
            confidence_mask = paw_ok
            n_low = int((~paw_ok).sum())
            if n_low > 0:
                self._log(f"  DLC filter: {n_low} frames ({100*n_low/n_frames:.1f}%) below likelihood {lk_thresh}")

        # ── Locomotion filter mask & body speed (always computed) ─────────
        loco_mask = None
        body_speed = None
        frame_displacements = None
        loco_thresh = params.get('loco_threshold', 20.0)
        # Try to find tailbase coordinates
        tb_bp = None
        tb_x_col = None
        for candidate in ['tailbase', 'tail_base', 'tb']:
            tb_x_col = next((c for c in bp_xcord.columns
                             if candidate in c.lower()), None)
            if tb_x_col:
                tb_bp = candidate
                break
        if tb_x_col is None:
            # Fallback: use the midpoint of hind paws as body center
            hl_bp = active_paws.get('HL', '')
            hr_bp = active_paws.get('HR', '')
            hl_x_col = next((c for c in bp_xcord.columns if hl_bp.lower() in c.lower()), None)
            hr_x_col = next((c for c in bp_xcord.columns if hr_bp.lower() in c.lower()), None)
            hl_y_col = next((c for c in bp_ycord.columns if hl_bp.lower() in c.lower()), None)
            hr_y_col = next((c for c in bp_ycord.columns if hr_bp.lower() in c.lower()), None)
            if hl_x_col and hr_x_col:
                cx = (bp_xcord[hl_x_col].values + bp_xcord[hr_x_col].values) / 2.0
                cy = (bp_ycord[hl_y_col].values + bp_ycord[hr_y_col].values) / 2.0
            else:
                cx = cy = None
        else:
            tb_y_col = next((c for c in bp_ycord.columns
                             if candidate in c.lower()), None)
            cx = bp_xcord[tb_x_col].values[:n_frames].astype(float)
            cy = bp_ycord[tb_y_col].values[:n_frames].astype(float) if tb_y_col else None

        if cx is not None and cy is not None:
            dx = np.diff(cx, prepend=cx[0])
            dy = np.diff(cy, prepend=cy[0])
            frame_displacements = np.sqrt(dx**2 + dy**2)
            body_speed = frame_displacements * fps
            loco_mask = body_speed > loco_thresh
            n_loco = int(loco_mask.sum())
            self._log(f"  Body speed computed: {n_loco}/{n_frames} frames above {loco_thresh:.1f} px/s")
        else:
            self._log("  Body speed: no tailbase/body center found, skipping")

        # ── Contact masks ────────────────────────────────────────────────────
        contact_masks = {}
        contact_method = params.get('contact_method', 'height')
        thresh = params['contact_threshold']

        # Helper to get x,y arrays for a body part
        def _get_xy(bp):
            x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
            y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
            if x_col and y_col:
                return (bp_xcord[x_col].values[:n_frames].astype(float),
                        bp_ycord[y_col].values[:n_frames].astype(float))
            return None, None

        # Store per-paw x,y for gait spatial metrics later
        paw_xy = {}

        for role, bp in active_paws.items():
            px, py = _get_xy(bp)
            if px is not None:
                paw_xy[role] = (px, py)

            # Height-based contact
            h_col = f'{bp}_Height'
            if h_col not in height_df.columns:
                matches = [c for c in height_df.columns if bp.lower() in c.lower()]
                h_col = matches[0] if matches else None

            height_mask = None
            if h_col:
                height_mask = (height_df[h_col].values[:n_frames] < thresh)

            # Speed-based contact
            speed_mask = None
            if contact_method in ('speed', 'combined') and px is not None:
                speed_mask = self._compute_speed_contact(
                    px, py, fps,
                    threshold=params.get('speed_threshold', 'auto'),
                    median_ms=params.get('median_filter_ms', 50),
                    min_bout_ms=params.get('min_bout_ms', 30))

            # Combine according to method
            if contact_method == 'height':
                mask = height_mask
            elif contact_method == 'speed':
                mask = speed_mask if speed_mask is not None else height_mask
            elif contact_method == 'combined':
                if height_mask is not None and speed_mask is not None:
                    mask = height_mask & speed_mask
                else:
                    mask = height_mask if height_mask is not None else speed_mask
            else:
                mask = height_mask

            if mask is not None:
                contact_masks[role] = pd.Series(mask, dtype=bool).reset_index(drop=True)

        # ── Brightness (optional) ────────────────────────────────────────────
        brightness_series = {}
        paw_contour_data = {}  # may be populated during brightness pass or standalone
        contour_cache_path = None
        contour_cache_key = None
        # Build contour paw subset (hind-only by default)
        contour_paws = {}
        if params.get('paw_contour'):
            _hind_roles = ('HL', 'HR')
            contour_paws = {r: bp for r, bp in active_paws.items()
                           if r in _hind_roles or params.get('contour_forelimbs')}
        if contour_paws and video_file and os.path.isfile(video_file) and _CV2_OK:
            contour_roi_sizes = params.get('contour_roi_sizes', params.get('roi_sizes', {}))
            contour_cache_key = {
                'video_mtime': round(os.path.getmtime(video_file), 2),
                'dlc_mtime':   round(os.path.getmtime(dlc_file), 2),
                'contour_roi_sizes': sorted(
                    {contour_paws[r]: contour_roi_sizes.get(r, 20) for r in contour_paws}.items()),
                'crop_x': params.get('crop_offset_x', 0),
                'crop_y': params.get('crop_offset_y', 0),
                'extraction_stride': params.get('extraction_stride', 1),
            }
            contour_cache_path = self._contour_cache_path(sess['session_name'], contour_cache_key)

            # --- try contour cache load ---
            if contour_cache_path and os.path.isfile(contour_cache_path):
                try:
                    cached_cdf = pd.read_csv(contour_cache_path)
                    metric_names = ['areas', 'spreads', 'intensities', 'widths',
                                    'solidities', 'aspect_ratios', 'circularities']
                    for role in contour_paws:
                        role_data = {}
                        for mn in metric_names:
                            col = f'{mn}_{role}'
                            if col in cached_cdf.columns:
                                arr = cached_cdf[col].values
                                if len(arr) > n_frames:
                                    arr = arr[:n_frames]
                                elif len(arr) < n_frames:
                                    arr = np.pad(arr, (0, n_frames - len(arr)))
                                role_data[mn] = arr.astype(float)
                        if role_data:
                            paw_contour_data[role] = role_data
                    # Load cached contour shapes (.npz) if available
                    shapes_path = contour_cache_path.replace('.csv', '_shapes.npz')
                    if os.path.isfile(shapes_path):
                        try:
                            _npz = np.load(shapes_path, allow_pickle=False)
                            for role in contour_paws:
                                if role in _npz and role in paw_contour_data:
                                    arr = _npz[role]
                                    if arr.ndim == 3 and arr.shape[1:] == (64, 2):
                                        paw_contour_data[role]['contour_shapes'] = list(arr)
                            self._log("  Contour shapes loaded from cache.")
                        except Exception:
                            pass
                    if paw_contour_data:
                        self._log("  Contour data loaded from cache.")
                except Exception as e:
                    self._log(f"  Contour cache load failed ({e}), will re-extract.")
                    paw_contour_data = {}

        if params['use_brightness'] and video_file and os.path.isfile(video_file):
            # --- build cache key (depends only on inputs, not analysis params) ---
            roi_sizes = params.get('roi_sizes', {})
            cache_key = {
                'video_mtime': round(os.path.getmtime(video_file), 2),
                'dlc_mtime':   round(os.path.getmtime(dlc_file),   2),
                'roi_sizes':   sorted(
                    {active_paws[r]: roi_sizes.get(r, 50) for r in active_paws}.items()),
                'crop_x':           params.get('crop_offset_x', 0),
                'crop_y':           params.get('crop_offset_y', 0),
                'brt_thresh':       params.get('brt_threshold', 0),
                'extraction_stride': params.get('extraction_stride', 1),
            }
            cache_path = self._brt_cache_path(sess['session_name'], cache_key)

            # --- try cache load ---
            if cache_path and os.path.isfile(cache_path):
                try:
                    cached_df = pd.read_csv(cache_path)
                    for role, bp in active_paws.items():
                        col = f'Pix_{bp}'
                        if col in cached_df.columns:
                            s = cached_df[col].reset_index(drop=True)
                            if len(s) > n_frames:
                                s = s.iloc[:n_frames].reset_index(drop=True)
                            elif len(s) < n_frames:
                                s = s.reindex(range(n_frames))
                            brightness_series[role] = s
                    self._log(f"  Brightness loaded from cache.")
                except Exception as e:
                    self._log(f"  Cache load failed ({e}), re-extracting.")
                    brightness_series = {}

            # --- extract fresh if cache miss ---
            if not brightness_series:
                self._log(f"  Brightness cache miss — extracting from video (this may take a while).")
                try:
                    thresh_val  = params.get('brt_threshold', 0)
                    square_size = {active_paws[role]: roi_sizes.get(role, 50)
                                   for role in active_paws}

                    # When brightness doesn't affect contact detection (brt_weight==0),
                    # only decode frames where at least one paw is in contact — much faster.
                    hint_mask = None
                    if abs(params.get('brt_weight', 1.0)) < 1e-6 and contact_masks:
                        hint_mask = np.zeros(n_frames, dtype=bool)
                        for mask_arr in contact_masks.values():
                            hint_mask |= mask_arr.values

                    # ── Contour callback (piggyback on brightness video pass) ──
                    contour_callback = None
                    if (contour_paws and _CV2_OK
                            and not paw_contour_data
                            and video_file and os.path.isfile(video_file)):
                        # Pre-allocate contour arrays (including toe-spreading)
                        for role in contour_paws:
                            paw_contour_data[role] = {
                                'areas': np.zeros(n_frames, dtype=float),
                                'spreads': np.zeros(n_frames, dtype=float),
                                'intensities': np.zeros(n_frames, dtype=float),
                                'widths': np.zeros(n_frames, dtype=float),
                                'solidities': np.zeros(n_frames, dtype=float),
                                'aspect_ratios': np.zeros(n_frames, dtype=float),
                                'circularities': np.zeros(n_frames, dtype=float),
                                'contour_shapes': [],   # normalized (64,2) arrays
                                'contour_solidities': [],  # extraction-time solidity per shape
                            }
                        contour_roi_sizes = params.get('contour_roi_sizes',
                                                       params.get('roi_sizes', {}))
                        _max_shapes = 500  # limit stored shapes per paw
                        _stride_val_shapes = max(1, params.get('extraction_stride', 1))
                        _shape_every = max(1, n_frames // (_max_shapes * _stride_val_shapes))
                        contour_crop_x = params.get('crop_offset_x', 0)
                        contour_crop_y = params.get('crop_offset_y', 0)

                        # Sub-progress setup
                        _stride_val = max(1, params.get('extraction_stride', 1))
                        total_contour_frames = len(range(0, n_frames, _stride_val))
                        self.app.root.after(0, self._sub_progress_label.config,
                                            {'text': 'Contour extraction: 0%'})
                        _contour_update_interval = max(1, total_contour_frames // 20)

                        def contour_callback(i_frame, gray_u8, frame):
                            """Called by brightness extractor for each decoded frame."""
                            fh, fw = gray_u8.shape[:2]
                            for role, bp in contour_paws.items():
                                if role not in paw_xy:
                                    continue
                                px_arr, py_arr = paw_xy[role]
                                if i_frame >= len(px_arr):
                                    continue
                                bx = int(px_arr[i_frame]) + contour_crop_x
                                by = int(py_arr[i_frame]) + contour_crop_y
                                rh = contour_roi_sizes.get(role, 20)
                                x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                                y1 = max(0, by - rh); y2 = min(fh, by + rh)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                roi = gray_u8[y1:y2, x1:x2]
                                if roi.size == 0:
                                    continue
                                blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                                _, thresh_img = cv2.threshold(blurred, 0, 255,
                                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)
                                if not contours:
                                    continue
                                best = max(contours, key=lambda c: cv2.contourArea(c))
                                area = cv2.contourArea(best)
                                if area < 4:
                                    continue
                                if area > 0:
                                    paw_contour_data[role]['areas'][i_frame] = area
                                    x_b, y_b, w_b, h_b = cv2.boundingRect(best)
                                    paw_contour_data[role]['spreads'][i_frame] = max(w_b, h_b)
                                    mask_c = np.zeros(roi.shape, dtype=np.uint8)
                                    cv2.drawContours(mask_c, [best], -1, 255, -1)
                                    paw_contour_data[role]['intensities'][i_frame] = cv2.mean(roi, mask=mask_c)[0]
                                    # Toe-spreading metrics
                                    paw_contour_data[role]['widths'][i_frame] = min(w_b, h_b)
                                    hull = cv2.convexHull(best)
                                    hull_area = cv2.contourArea(hull)
                                    paw_contour_data[role]['solidities'][i_frame] = area / hull_area if hull_area > 0 else 0.0
                                    dim_max = max(w_b, h_b)
                                    dim_min = min(w_b, h_b)
                                    paw_contour_data[role]['aspect_ratios'][i_frame] = dim_max / dim_min if dim_min > 0 else 0.0
                                    perimeter = cv2.arcLength(best, True)
                                    paw_contour_data[role]['circularities'][i_frame] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                                    # Store normalized contour shape (subsampled)
                                    if (i_frame % _shape_every == 0
                                            and len(paw_contour_data[role]['contour_shapes']) < _max_shapes):
                                        pts = best.squeeze()
                                        if pts.ndim == 2 and len(pts) >= 3:
                                            resampled = GaitLimbTab._resample_contour(pts, 64)
                                            normed = GaitLimbTab._normalize_contour(resampled, area)
                                            if normed is not None:
                                                paw_contour_data[role]['contour_shapes'].append(normed)
                                                paw_contour_data[role]['contour_solidities'].append(
                                                    paw_contour_data[role]['solidities'][i_frame])

                            # Progress update
                            frame_idx = i_frame // _stride_val
                            if frame_idx % _contour_update_interval == 0:
                                pct = int(100 * frame_idx / total_contour_frames)
                                self.app.root.after(0, self._sub_progress_label.config,
                                                    {'text': f'Contour extraction: {pct}%'})

                    brt_ex = PixelBrightnessExtractorOptimized(
                        active_bps,
                        square_size=square_size,
                        pixel_threshold=float(thresh_val) if thresh_val > 0 else None,
                        crop_offset_x=params.get('crop_offset_x', 0),
                        crop_offset_y=params.get('crop_offset_y', 0),
                    )
                    brt_df = brt_ex.extract_brightness_features(
                        dlc_file, video_file,
                        stride=params.get('extraction_stride', 1),
                        frame_mask=hint_mask,
                        cancel_flag=self._cancel_flag,
                        frame_callback=contour_callback,
                    )
                    for role, bp in active_paws.items():
                        col = f'Pix_{bp}'
                        if col not in brt_df.columns:
                            matches = [c for c in brt_df.columns
                                       if bp.lower() in c.lower()
                                       and c.startswith('Pix_')]
                            col = matches[0] if matches else None
                        if col:
                            s = brt_df[col].reset_index(drop=True)
                            if len(s) > n_frames:
                                s = s.iloc[:n_frames].reset_index(drop=True)
                            elif len(s) < n_frames:
                                s = s.reindex(range(n_frames))
                            brightness_series[role] = s
                    # --- save to cache ---
                    if cache_path and brightness_series:
                        try:
                            pd.DataFrame({f'Pix_{active_paws[r]}': brightness_series[r]
                                          for r in brightness_series}).to_csv(cache_path, index=False)
                            self._log(f"  Brightness cached.")
                            import json as _json
                            sidecar = cache_path.replace('.csv', '.json')
                            with open(sidecar, 'w') as _f:
                                _json.dump(cache_key, _f, indent=2)
                        except Exception:
                            pass
                    # --- save contour to cache ---
                    if contour_cache_path and paw_contour_data:
                        try:
                            contour_cols = {}
                            for role, arrays in paw_contour_data.items():
                                for metric_name, arr in arrays.items():
                                    if metric_name in ('contour_shapes', 'contour_solidities'):
                                        continue  # not cacheable as CSV column
                                    contour_cols[f'{metric_name}_{role}'] = arr
                            pd.DataFrame(contour_cols).to_csv(contour_cache_path, index=False)
                            self._log("  Contour data cached.")
                            import json as _json2
                            sidecar = contour_cache_path.replace('.csv', '.json')
                            with open(sidecar, 'w') as _f:
                                _json2.dump(contour_cache_key, _f, indent=2)
                            # Save contour shapes as .npz
                            _shape_arrays = {}
                            for _sr, _sd in paw_contour_data.items():
                                _sl = _sd.get('contour_shapes', [])
                                if _sl:
                                    _shape_arrays[_sr] = np.array(_sl)
                            if _shape_arrays:
                                np.savez_compressed(
                                    contour_cache_path.replace('.csv', '_shapes.npz'),
                                    **_shape_arrays)
                                self._log("  Contour shapes cached.")
                        except Exception:
                            pass
                    # Clean up contour progress if callback was used
                    if contour_callback is not None:
                        self.app.root.after(0, self._sub_progress_label.config, {'text': ''})
                        self._log("  Paw contour extraction complete (during brightness pass).")
                except Exception as e:
                    self._log(f"  Brightness skipped: {e}")

        # ── Brightness-weighted contact refinement ────────────────────────────
        brt_weight = params.get('brt_weight', 0.0)
        if brt_weight > 0 and brightness_series:
            thresh = params['contact_threshold']
            for role, bp in active_paws.items():
                if role not in brightness_series or role not in contact_masks:
                    continue

                # Retrieve raw height values for this paw
                h_col = f'{bp}_Height'
                if h_col not in height_df.columns:
                    matches = [c for c in height_df.columns if bp.lower() in c.lower()]
                    h_col = matches[0] if matches else None
                if h_col is None:
                    continue

                h_vals = height_df[h_col].values[:n_frames].astype(float)
                b_vals = brightness_series[role].values.astype(float)

                # Height score: 1 at floor level, 0 at threshold or above
                h_score = np.clip(1.0 - h_vals / max(float(thresh), 1.0), 0.0, 1.0)

                # Brightness score: normalise to 90th percentile of session brightness
                valid_b = b_vals[np.isfinite(b_vals) & (b_vals > 0)]
                brt_90 = float(np.percentile(valid_b, 90)) if len(valid_b) > 0 else 1.0
                b_score = np.clip(b_vals / max(brt_90, 1.0), 0.0, 1.0)

                combined = (1.0 - brt_weight) * h_score + brt_weight * b_score
                contact_masks[role] = pd.Series(combined > 0.5, dtype=bool)
                self._log(f"  {role}: brt_weight={brt_weight:.2f}, brt_90th={brt_90:.1f}")

        # ── Paw contour area extraction (standalone fallback) ─────────────
        # Only runs when contour wasn't already done during brightness pass
        # (i.e. brightness cached, brightness disabled, or contour-only mode)
        if (contour_paws and not paw_contour_data
                and video_file and os.path.isfile(video_file) and _CV2_OK):
            self._log("  Extracting paw contour areas (standalone)…")
            try:
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    roi_sizes = params.get('contour_roi_sizes', params.get('roi_sizes', {}))
                    crop_x = params.get('crop_offset_x', 0)
                    crop_y = params.get('crop_offset_y', 0)
                    stride = max(1, params.get('extraction_stride', 1))

                    # Pre-allocate arrays (including toe-spreading)
                    _max_shapes_b = 500
                    _shape_every_b = max(1, n_frames // (_max_shapes_b * stride))
                    for role in contour_paws:
                        paw_contour_data[role] = {
                            'areas': np.zeros(n_frames, dtype=float),
                            'spreads': np.zeros(n_frames, dtype=float),
                            'intensities': np.zeros(n_frames, dtype=float),
                            'widths': np.zeros(n_frames, dtype=float),
                            'solidities': np.zeros(n_frames, dtype=float),
                            'aspect_ratios': np.zeros(n_frames, dtype=float),
                            'circularities': np.zeros(n_frames, dtype=float),
                            'contour_shapes': [],
                            'contour_solidities': [],  # extraction-time solidity per shape
                        }

                    total_contour_frames = len(range(0, n_frames, stride))
                    self.app.root.after(0, self._sub_progress_label.config,
                                        {'text': 'Contour extraction: 0%'})
                    _update_interval = max(1, total_contour_frames // 20)

                    frame_idx = 0
                    for fi in range(n_frames):
                        if self._cancel_flag.is_set():
                            break

                        if fi % stride != 0:
                            cap.grab()   # advance without decoding — fast
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            frame_idx += 1
                            continue

                        # Progress update
                        if frame_idx % _update_interval == 0:
                            pct = int(100 * frame_idx / total_contour_frames)
                            self.app.root.after(0, self._sub_progress_label.config,
                                                {'text': f'Contour extraction: {pct}%'})

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        fh, fw = frame.shape[:2]

                        for role, bp in contour_paws.items():
                            if role not in paw_xy:
                                continue
                            px_arr, py_arr = paw_xy[role]
                            if fi >= len(px_arr):
                                continue
                            bx = int(px_arr[fi]) + crop_x
                            by = int(py_arr[fi]) + crop_y
                            rh = roi_sizes.get(role, 20)
                            x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                            y1 = max(0, by - rh); y2 = min(fh, by + rh)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = gray[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue

                            blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                            _, thresh_img = cv2.threshold(
                                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                            contours, _ = cv2.findContours(
                                thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            if not contours:
                                continue

                            best = max(contours, key=lambda c: cv2.contourArea(c))
                            area = cv2.contourArea(best)
                            if area > 0:
                                paw_contour_data[role]['areas'][fi] = area
                                x_b, y_b, w_b, h_b = cv2.boundingRect(best)
                                paw_contour_data[role]['spreads'][fi] = max(w_b, h_b)
                                mask_c = np.zeros(roi.shape, dtype=np.uint8)
                                cv2.drawContours(mask_c, [best], -1, 255, -1)
                                paw_contour_data[role]['intensities'][fi] = cv2.mean(roi, mask=mask_c)[0]
                                # Toe-spreading metrics
                                paw_contour_data[role]['widths'][fi] = min(w_b, h_b)
                                hull = cv2.convexHull(best)
                                hull_area = cv2.contourArea(hull)
                                paw_contour_data[role]['solidities'][fi] = area / hull_area if hull_area > 0 else 0.0
                                dim_max = max(w_b, h_b)
                                dim_min = min(w_b, h_b)
                                paw_contour_data[role]['aspect_ratios'][fi] = dim_max / dim_min if dim_min > 0 else 0.0
                                perimeter = cv2.arcLength(best, True)
                                paw_contour_data[role]['circularities'][fi] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                                # Store normalized contour shape (subsampled)
                                if (fi % _shape_every_b == 0
                                        and len(paw_contour_data[role]['contour_shapes']) < _max_shapes_b):
                                    pts = best.squeeze()
                                    if pts.ndim == 2 and len(pts) >= 3:
                                        resampled = GaitLimbTab._resample_contour(pts, 64)
                                        normed = GaitLimbTab._normalize_contour(resampled, area)
                                        if normed is not None:
                                            paw_contour_data[role]['contour_shapes'].append(normed)
                                            paw_contour_data[role]['contour_solidities'].append(
                                                paw_contour_data[role]['solidities'][fi])

                        frame_idx += 1

                    # Reset sub-progress
                    self.app.root.after(0, self._sub_progress_label.config, {'text': ''})

                    cap.release()
                    self._log("  Paw contour extraction complete (standalone).")
                    # --- save contour to cache ---
                    if contour_cache_path and paw_contour_data:
                        try:
                            contour_cols = {}
                            for role, arrays in paw_contour_data.items():
                                for metric_name, arr in arrays.items():
                                    if metric_name in ('contour_shapes', 'contour_solidities'):
                                        continue  # not cacheable as CSV column
                                    contour_cols[f'{metric_name}_{role}'] = arr
                            pd.DataFrame(contour_cols).to_csv(contour_cache_path, index=False)
                            self._log("  Contour data cached.")
                            import json as _json3
                            sidecar = contour_cache_path.replace('.csv', '.json')
                            with open(sidecar, 'w') as _f:
                                _json3.dump(contour_cache_key, _f, indent=2)
                            # Save contour shapes as .npz
                            _shape_arrays_b = {}
                            for _sr, _sd in paw_contour_data.items():
                                _sl = _sd.get('contour_shapes', [])
                                if _sl:
                                    _shape_arrays_b[_sr] = np.array(_sl)
                            if _shape_arrays_b:
                                np.savez_compressed(
                                    contour_cache_path.replace('.csv', '_shapes.npz'),
                                    **_shape_arrays_b)
                                self._log("  Contour shapes cached.")
                        except Exception:
                            pass
            except Exception as e:
                self._log(f"  Paw contour extraction failed: {e}")
                paw_contour_data = {}

        # ── Metric computation helper ────────────────────────────────────────
        def _metrics(frame_slice=None):
            if frame_slice is not None:
                masks = {r: m.iloc[frame_slice].reset_index(drop=True)
                         for r, m in contact_masks.items()}
                n = len(next(iter(masks.values()))) if masks else 0
            else:
                masks = contact_masks
                n = n_frames

            m = {'n_frames': n, 'fps': round(fps, 2),
                 'fallback_fps_used': _used_fallback_fps}

            for role, mask in masks.items():
                m[f'contact_pct_{role}'] = round(float(mask.mean()) * 100, 2)
                m[f'n_contact_{role}']   = int(mask.sum())

            # Hind WBI / SI
            if 'HL' in masks and 'HR' in masks:
                cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
                tot = cHL + cHR
                m['WBI_hind'] = round(cHL / tot * 100, 2) if tot > 0 else float('nan')
                m['SI_hind']  = round((cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')

            # Fore WBI / SI
            if 'FL' in masks and 'FR' in masks:
                cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
                tot = cFL + cFR
                m['WBI_fore'] = round(cFL / tot * 100, 2) if tot > 0 else float('nan')
                m['SI_fore']  = round((cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

            # Per-paw WBI — each paw's fraction of total contact time
            total_contact = sum(m[f'contact_pct_{r}'] for r in masks)
            if total_contact > 0:
                for role in masks:
                    m[f'WBI_{role}'] = round(
                        m[f'contact_pct_{role}'] / total_contact * 100, 2)

            # SBI (absolute symmetry index)
            if 'HL' in masks and 'HR' in masks:
                cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
                tot = cHL + cHR
                m['SBI_hind'] = round(
                    2 * abs(cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')
            if 'FL' in masks and 'FR' in masks:
                cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
                tot = cFL + cFR
                m['SBI_fore'] = round(
                    2 * abs(cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

            # Hind / fore ratio
            if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                hind = (m['contact_pct_HL'] + m['contact_pct_HR']) / 2
                fore = (m['contact_pct_FL'] + m['contact_pct_FR']) / 2
                m['hind_fore_ratio'] = round(hind / fore, 4) if fore > 0 else float('nan')

            # Brightness during contact
            for role, brt_full in brightness_series.items():
                mask = masks.get(role)
                if mask is None:
                    continue
                if frame_slice is not None:
                    brt_slice = brt_full.iloc[frame_slice].reset_index(drop=True)
                else:
                    brt_slice = brt_full
                contact_brt = brt_slice.values[mask.values.astype(bool)]
                if len(contact_brt) > 0:
                    m[f'brightness_{role}'] = round(float(np.nanmean(contact_brt)), 4)
                else:
                    m[f'brightness_{role}'] = float('nan')

            if 'brightness_HL' in m and 'brightness_HR' in m:
                bHL = m['brightness_HL']
                bHR = m['brightness_HR']
                if (not (np.isnan(bHL) or np.isnan(bHR))) and bHR > 0:
                    m['brightness_ratio_HL_HR'] = round(bHL / bHR, 4)

            # ── Distance & movement time ──────────────────────────────────
            if frame_displacements is not None:
                disp_sl = frame_displacements[frame_slice] if frame_slice is not None else frame_displacements
                m['total_distance'] = round(float(np.nansum(disp_sl)), 2)
                if loco_mask is not None:
                    lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
                    lm_sl = lm_sl[:len(disp_sl)]
                    m['loco_total_distance'] = round(float(np.nansum(disp_sl[lm_sl])), 2)
                    m['time_moving_s'] = round(float(lm_sl.sum()) / fps, 2) if fps > 0 else 0.0
                    m['time_moving_pct'] = round(float(lm_sl.mean()) * 100, 2)

            # ── Body speed mean ───────────────────────────────────────────
            if body_speed is not None:
                bs_sl = body_speed[frame_slice] if frame_slice is not None else body_speed
                m['body_speed_mean'] = round(float(np.nanmean(bs_sl)), 2)
                if loco_mask is not None:
                    lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
                    lm_sl = lm_sl[:len(bs_sl)]
                    loco_bs = bs_sl[lm_sl]
                    m['body_speed_loco'] = round(float(np.nanmean(loco_bs)), 2) if len(loco_bs) > 0 else float('nan')

            # ── Support patterns (paw count per frame during locomotion) ──
            if len(masks) >= 4 and loco_mask is not None:
                lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
                all_contact = np.zeros(n, dtype=int)
                for role_sp, mask_sp in masks.items():
                    ma = mask_sp.values.astype(bool) if hasattr(mask_sp, 'values') else np.asarray(mask_sp, dtype=bool)
                    all_contact[:len(ma)] += ma[:n].astype(int)
                lm_frames = lm_sl[:n]
                n_loco_frames = int(lm_frames.sum())
                if n_loco_frames > 0:
                    for npaws in range(5):
                        cnt = int(((all_contact == npaws) & lm_frames).sum())
                        m[f'support_{npaws}paw_pct'] = round(cnt / n_loco_frames * 100, 2)

            # ── Gait metrics (dual: all frames + locomotion only) ─────────
            def _gait_block(m, masks, loco_filter_mask, prefix=''):
                """Compute gait timing, spatial, phase, symmetry metrics with optional prefix."""
                for role, mask in masks.items():
                    mask_arr = mask.values.astype(bool) if hasattr(mask, 'values') else np.asarray(mask, dtype=bool)

                    # Apply confidence and locomotion filters to gait computation
                    gait_valid = np.ones(len(mask_arr), dtype=bool)
                    if confidence_mask is not None:
                        cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                        if len(cm_slice) == len(mask_arr):
                            gait_valid &= cm_slice
                    if loco_filter_mask is not None:
                        lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                        if len(lm_slice) == len(mask_arr):
                            gait_valid &= lm_slice

                    # Mask out invalid frames for gait analysis
                    gait_contact = mask_arr & gait_valid

                    _min_stance = params.get('min_stance_ms', 0)
                    stance_durs, swing_durs, stance_onsets = GaitLimbTab._gait_bouts(
                        gait_contact, fps, min_stride_ms=_min_stance)

                    if stance_durs:
                        m[f'{prefix}stance_dur_{role}'] = round(float(np.mean(stance_durs)), 4)
                        m[f'{prefix}n_strides_{role}'] = len(stance_durs)
                    else:
                        m[f'{prefix}stance_dur_{role}'] = float('nan')
                        m[f'{prefix}n_strides_{role}'] = 0

                    if swing_durs:
                        m[f'{prefix}swing_dur_{role}'] = round(float(np.mean(swing_durs)), 4)
                    else:
                        m[f'{prefix}swing_dur_{role}'] = float('nan')

                    # Stride = stance + swing
                    if stance_durs and swing_durs:
                        stride_dur = m[f'{prefix}stance_dur_{role}'] + m[f'{prefix}swing_dur_{role}']
                        m[f'{prefix}stride_dur_{role}'] = round(stride_dur, 4)
                        m[f'{prefix}duty_cycle_{role}'] = round(
                            m[f'{prefix}stance_dur_{role}'] / stride_dur * 100, 2) if stride_dur > 0 else float('nan')
                        m[f'{prefix}cadence_{role}'] = round(
                            60.0 / stride_dur, 2) if stride_dur > 0 else float('nan')
                    else:
                        m[f'{prefix}stride_dur_{role}'] = float('nan')
                        m[f'{prefix}duty_cycle_{role}'] = float('nan')
                        m[f'{prefix}cadence_{role}'] = float('nan')

                    # Stride length (distance between consecutive foot-strikes)
                    if role in paw_xy and len(stance_onsets) >= 2:
                        px_arr, py_arr = paw_xy[role]
                        if frame_slice is not None:
                            sl_start = frame_slice.start or 0
                            abs_onsets = [sl_start + o for o in stance_onsets
                                          if sl_start + o < len(px_arr)]
                        else:
                            abs_onsets = [o for o in stance_onsets if o < len(px_arr)]
                        if len(abs_onsets) >= 2:
                            dists = []
                            for j in range(1, len(abs_onsets)):
                                i0, i1 = abs_onsets[j - 1], abs_onsets[j]
                                d = np.sqrt((px_arr[i1] - px_arr[i0])**2 +
                                            (py_arr[i1] - py_arr[i0])**2)
                                dists.append(d)
                            m[f'{prefix}stride_len_{role}'] = round(float(np.mean(dists)), 2)
                        else:
                            m[f'{prefix}stride_len_{role}'] = float('nan')
                    else:
                        m[f'{prefix}stride_len_{role}'] = float('nan')

                    # Swing speed = stride_length / swing_duration
                    sl_key = f'{prefix}stride_len_{role}'
                    sw_key = f'{prefix}swing_dur_{role}'
                    if (sl_key in m and sw_key in m
                            and not np.isnan(m[sl_key]) and not np.isnan(m[sw_key])
                            and m[sw_key] > 0):
                        m[f'{prefix}swing_speed_{role}'] = round(m[sl_key] / m[sw_key], 2)
                    else:
                        m[f'{prefix}swing_speed_{role}'] = float('nan')

                    # Stride-to-stride CV (coefficient of variation of stance durations)
                    if len(stance_durs) >= 3:
                        sd_arr = np.array(stance_durs)
                        sd_mean = np.mean(sd_arr)
                        if sd_mean > 0:
                            m[f'{prefix}stride_cv_{role}'] = round(
                                float(np.std(sd_arr, ddof=1) / sd_mean * 100), 2)
                        else:
                            m[f'{prefix}stride_cv_{role}'] = float('nan')
                    else:
                        m[f'{prefix}stride_cv_{role}'] = float('nan')

                # ── Step length / width (contralateral) ──────────────────
                for pair_name, left_role, right_role in [('hind', 'HL', 'HR'), ('fore', 'FL', 'FR')]:
                    if left_role not in masks or right_role not in masks:
                        continue
                    # Get stance onsets for both sides
                    l_mask = masks[left_role].values.astype(bool) if hasattr(masks[left_role], 'values') else np.asarray(masks[left_role], dtype=bool)
                    r_mask = masks[right_role].values.astype(bool) if hasattr(masks[right_role], 'values') else np.asarray(masks[right_role], dtype=bool)

                    # Apply filters for loco variant
                    if loco_filter_mask is not None:
                        lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                        l_mask = l_mask & lm_slice[:len(l_mask)]
                        r_mask = r_mask & lm_slice[:len(r_mask)]
                    if confidence_mask is not None:
                        cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                        l_mask = l_mask & cm_slice[:len(l_mask)]
                        r_mask = r_mask & cm_slice[:len(r_mask)]

                    _, _, l_onsets = GaitLimbTab._gait_bouts(l_mask, fps, _min_stance)
                    _, _, r_onsets = GaitLimbTab._gait_bouts(r_mask, fps, _min_stance)

                    if (left_role in paw_xy and right_role in paw_xy
                            and l_onsets and r_onsets):
                        lpx, lpy = paw_xy[left_role]
                        rpx, rpy = paw_xy[right_role]
                        sl_start = (frame_slice.start or 0) if frame_slice is not None else 0

                        # Step length: distance between contralateral strikes
                        all_strikes = sorted(
                            [('L', sl_start + o) for o in l_onsets] +
                            [('R', sl_start + o) for o in r_onsets],
                            key=lambda x: x[1])
                        step_lens = []
                        for j in range(1, len(all_strikes)):
                            if all_strikes[j][0] != all_strikes[j-1][0]:
                                i0 = all_strikes[j-1][1]
                                i1 = all_strikes[j][1]
                                if i0 < len(lpx) and i1 < len(lpx):
                                    # Use position of the striking paw at its onset
                                    if all_strikes[j-1][0] == 'L':
                                        x0, y0 = lpx[i0], lpy[i0]
                                    else:
                                        x0, y0 = rpx[i0], rpy[i0]
                                    if all_strikes[j][0] == 'L':
                                        x1, y1 = lpx[i1], lpy[i1]
                                    else:
                                        x1, y1 = rpx[i1], rpy[i1]
                                    step_lens.append(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
                        m[f'{prefix}step_len_{pair_name}'] = round(float(np.mean(step_lens)), 2) if step_lens else float('nan')

                        # Step width: lateral distance at mid-stance
                        widths = []
                        r_onsets_arr = np.array(r_onsets)
                        for lo in l_onsets:
                            abs_lo = sl_start + lo
                            # Find nearest right onset using searchsorted
                            idx = np.searchsorted(r_onsets_arr, lo)
                            candidates = []
                            if idx < len(r_onsets_arr):
                                candidates.append(r_onsets_arr[idx])
                            if idx > 0:
                                candidates.append(r_onsets_arr[idx - 1])
                            if candidates:
                                nearest_r = min(candidates, key=lambda ro: abs(ro - lo))
                                abs_r = sl_start + nearest_r
                                if abs_lo < len(lpx) and abs_r < len(rpx):
                                    widths.append(abs(lpy[abs_lo] - rpy[abs_r]))
                        m[f'{prefix}step_width_{pair_name}'] = round(float(np.mean(widths)), 2) if widths else float('nan')
                    else:
                        m[f'{prefix}step_len_{pair_name}'] = float('nan')
                        m[f'{prefix}step_width_{pair_name}'] = float('nan')

                # ── Interlimb coordination (phase) ───────────────────────
                if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                    def _phase(ref_role, test_role):
                        ref_mask_arr = masks[ref_role].values.astype(bool) if hasattr(masks[ref_role], 'values') else np.asarray(masks[ref_role], dtype=bool)
                        tst_mask_arr = masks[test_role].values.astype(bool) if hasattr(masks[test_role], 'values') else np.asarray(masks[test_role], dtype=bool)

                        if loco_filter_mask is not None:
                            lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                            ref_mask_arr = ref_mask_arr & lm_slice[:len(ref_mask_arr)]
                            tst_mask_arr = tst_mask_arr & lm_slice[:len(tst_mask_arr)]
                        if confidence_mask is not None:
                            cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                            ref_mask_arr = ref_mask_arr & cm_slice[:len(ref_mask_arr)]
                            tst_mask_arr = tst_mask_arr & cm_slice[:len(tst_mask_arr)]

                        _, _, ref_on = GaitLimbTab._gait_bouts(ref_mask_arr, fps, _min_stance)
                        _, _, tst_on = GaitLimbTab._gait_bouts(tst_mask_arr, fps, _min_stance)
                        if len(ref_on) < 3 or not tst_on:
                            return float('nan')
                        phases = []
                        tst_arr = np.array(tst_on)
                        for i in range(len(ref_on) - 1):
                            cycle_len = ref_on[i+1] - ref_on[i]
                            if cycle_len <= 0:
                                continue
                            # Find test onsets within this cycle using searchsorted
                            lo = np.searchsorted(tst_arr, ref_on[i], side='left')
                            hi = np.searchsorted(tst_arr, ref_on[i+1], side='left')
                            for j in range(lo, hi):
                                phases.append((tst_arr[j] - ref_on[i]) / cycle_len)
                        return round(float(np.mean(phases)), 3) if phases else float('nan')

                    m[f'{prefix}phase_HL_HR'] = _phase('HR', 'HL')
                    m[f'{prefix}phase_diagonal'] = _phase('HR', 'FL')  # HR-FL diagonal pair
                    m[f'{prefix}phase_FL_FR'] = _phase('FR', 'FL')
                    m[f'{prefix}phase_HL_FL'] = _phase('FL', 'HL')
                    m[f'{prefix}phase_HR_FR'] = _phase('FR', 'HR')
                    m[f'{prefix}phase_HL_FR'] = _phase('FR', 'HL')

                # ── Regularity index ──────────────────────────────────────
                if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                    ri = GaitLimbTab._regularity_index(masks, fps, frame_slice,
                                                        loco_filter_mask, confidence_mask,
                                                        params.get('min_stance_ms', 0))
                    if ri is not None:
                        m[f'{prefix}regularity_index'] = ri

                # ── Print position (hind-fore overlap) ────────────────────
                if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                    for side_lbl, hind_r, fore_r in [('L', 'HL', 'FL'), ('R', 'HR', 'FR')]:
                        if hind_r in paw_xy and fore_r in paw_xy:
                            pp = GaitLimbTab._print_position(
                                masks, paw_xy, hind_r, fore_r, fps, frame_slice,
                                loco_filter_mask, confidence_mask,
                                params.get('min_stance_ms', 0))
                            m[f'{prefix}print_position_{side_lbl}'] = pp

                # ── Gait symmetry indices ────────────────────────────────
                if f'{prefix}stance_dur_HL' in m and f'{prefix}stance_dur_HR' in m:
                    sHL = m[f'{prefix}stance_dur_HL']
                    sHR = m[f'{prefix}stance_dur_HR']
                    tot = sHL + sHR
                    m[f'{prefix}stance_SI_hind'] = round((sHL - sHR) / tot * 100, 2) if tot > 0 and not (np.isnan(sHL) or np.isnan(sHR)) else float('nan')
                if f'{prefix}stride_len_HL' in m and f'{prefix}stride_len_HR' in m:
                    lHL = m[f'{prefix}stride_len_HL']
                    lHR = m[f'{prefix}stride_len_HR']
                    tot = lHL + lHR
                    m[f'{prefix}stride_len_SI_hind'] = round((lHL - lHR) / tot * 100, 2) if tot > 0 and not (np.isnan(lHL) or np.isnan(lHR)) else float('nan')

            _gait_block(m, masks, None, prefix='')           # All frames
            if loco_mask is not None:
                _gait_block(m, masks, loco_mask, prefix='loco_')  # Locomotion only

            # ── Paw contour area (Step 5) ────────────────────────────────────
            if paw_contour_data:
                # Build full-stance mask (all active paws in contact)
                stance_mask_all = None
                if masks:
                    _contour_roles = [r for r in paw_contour_data if r in masks]
                    stance_arrays = [masks[r].values.astype(bool) if hasattr(masks[r], 'values')
                                     else np.asarray(masks[r], dtype=bool) for r in _contour_roles]
                    if stance_arrays:
                        stance_mask_all = stance_arrays[0].copy()
                        for _sm in stance_arrays[1:]:
                            _sml = min(len(stance_mask_all), len(_sm))
                            stance_mask_all = stance_mask_all[:_sml] & _sm[:_sml]

                for role in list(paw_contour_data.keys()):
                    areas_full = paw_contour_data[role]['areas']
                    spreads_full = paw_contour_data[role]['spreads']
                    intensities_full = paw_contour_data[role]['intensities']
                    mask_arr = masks[role].values.astype(bool) if role in masks else np.ones(n_frames, dtype=bool)
                    sl = frame_slice
                    areas_sl = areas_full[sl] if sl is not None else areas_full
                    spreads_sl = spreads_full[sl] if sl is not None else spreads_full
                    ints_sl = intensities_full[sl] if sl is not None else intensities_full
                    mask_sl = mask_arr

                    widths_full = paw_contour_data[role].get('widths')
                    solidities_full = paw_contour_data[role].get('solidities')
                    aspect_ratios_full = paw_contour_data[role].get('aspect_ratios')
                    circularities_full = paw_contour_data[role].get('circularities')
                    widths_sl = (widths_full[sl] if sl is not None else widths_full) if widths_full is not None else None
                    solidities_sl = (solidities_full[sl] if sl is not None else solidities_full) if solidities_full is not None else None
                    ar_sl = (aspect_ratios_full[sl] if sl is not None else aspect_ratios_full) if aspect_ratios_full is not None else None
                    circ_sl = (circularities_full[sl] if sl is not None else circularities_full) if circularities_full is not None else None

                    # ── Regular (per-paw contact mask) ──
                    valid = mask_sl & (areas_sl[:len(mask_sl)] > 0)
                    _ca = areas_sl[:len(mask_sl)][valid]
                    m[f'paw_area_{role}'] = round(float(np.nanmean(_ca)), 2) if len(_ca) > 0 else float('nan')
                    _cs = spreads_sl[:len(mask_sl)][valid]
                    m[f'paw_spread_{role}'] = round(float(np.nanmean(_cs)), 2) if len(_cs) > 0 else float('nan')
                    _ci = ints_sl[:len(mask_sl)][valid]
                    m[f'contact_intensity_{role}'] = round(float(np.nanmean(_ci)), 2) if len(_ci) > 0 else float('nan')
                    if widths_sl is not None:
                        m[f'paw_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid])), 2) if valid.any() else float('nan')
                        m[f'paw_solidity_{role}'] = round(float(np.nanmean(solidities_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                        m[f'paw_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                        m[f'paw_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')

                    # ── Paw-like filtered (solidity + AR + circularity) ──
                    PAWLIKE_SOL = self._pawlike_thresholds.get('solidity', 1.00)
                    PAWLIKE_AR = self._pawlike_thresholds.get('aspect_ratio', 1.6)
                    PAWLIKE_CIRC = self._pawlike_thresholds.get('circularity', 0.10)
                    if solidities_sl is not None:
                        sol_arr = solidities_sl[:len(mask_sl)]
                        valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                        if ar_sl is not None:
                            valid_paw = valid_paw & (ar_sl[:len(mask_sl)] <= PAWLIKE_AR)
                        if circ_sl is not None:
                            valid_paw = valid_paw & (circ_sl[:len(mask_sl)] >= PAWLIKE_CIRC)
                        _pca = areas_sl[:len(mask_sl)][valid_paw]
                        m[f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                        _pcs = spreads_sl[:len(mask_sl)][valid_paw]
                        m[f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                        _pci = ints_sl[:len(mask_sl)][valid_paw]
                        m[f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                        if widths_sl is not None:
                            m[f'pawlike_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid_paw])), 2) if valid_paw.any() else float('nan')
                            m[f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                            m[f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')
                            m[f'pawlike_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')

                    # ── Full-stance variant (all contour paws in contact) ──
                    if stance_mask_all is not None:
                        stance_sl = stance_mask_all[sl] if sl is not None else stance_mask_all
                        _ml = min(len(mask_sl), len(stance_sl))
                        stance_valid = mask_sl[:_ml] & stance_sl[:_ml] & (areas_sl[:_ml] > 0)
                        _sca = areas_sl[:_ml][stance_valid]
                        m[f'paw_area_stance_{role}'] = round(float(np.nanmean(_sca)), 2) if len(_sca) > 0 else float('nan')
                        _scs = spreads_sl[:_ml][stance_valid]
                        m[f'paw_spread_stance_{role}'] = round(float(np.nanmean(_scs)), 2) if len(_scs) > 0 else float('nan')
                        _sci = ints_sl[:_ml][stance_valid]
                        m[f'contact_intensity_stance_{role}'] = round(float(np.nanmean(_sci)), 2) if len(_sci) > 0 else float('nan')
                        if widths_sl is not None:
                            m[f'paw_width_stance_{role}'] = round(float(np.nanmean(widths_sl[:_ml][stance_valid])), 2) if stance_valid.any() else float('nan')
                            m[f'paw_solidity_stance_{role}'] = round(float(np.nanmean(solidities_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                            m[f'paw_aspect_ratio_stance_{role}'] = round(float(np.nanmean(ar_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                            m[f'paw_circularity_stance_{role}'] = round(float(np.nanmean(circ_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')

                # Area ratios (regular)
                if 'paw_area_HL' in m and 'paw_area_HR' in m:
                    aHL, aHR = m['paw_area_HL'], m['paw_area_HR']
                    if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                        m['paw_area_ratio_hind'] = round(aHL / aHR, 4)
                # Area ratios (stance)
                if 'paw_area_stance_HL' in m and 'paw_area_stance_HR' in m:
                    aHL_s, aHR_s = m['paw_area_stance_HL'], m['paw_area_stance_HR']
                    if not (np.isnan(aHL_s) or np.isnan(aHR_s)) and aHR_s > 0:
                        m['paw_area_ratio_stance_hind'] = round(aHL_s / aHR_s, 4)
                # Intensity ratios (regular)
                if 'contact_intensity_HL' in m and 'contact_intensity_HR' in m:
                    iHL, iHR = m['contact_intensity_HL'], m['contact_intensity_HR']
                    if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                        m['contact_intensity_ratio_hind'] = round(iHL / iHR, 4)
                # Intensity ratios (stance)
                if 'contact_intensity_stance_HL' in m and 'contact_intensity_stance_HR' in m:
                    iHL_s, iHR_s = m['contact_intensity_stance_HL'], m['contact_intensity_stance_HR']
                    if not (np.isnan(iHL_s) or np.isnan(iHR_s)) and iHR_s > 0:
                        m['contact_intensity_ratio_stance_hind'] = round(iHL_s / iHR_s, 4)
                # Area ratios (paw-like)
                if 'pawlike_area_HL' in m and 'pawlike_area_HR' in m:
                    aHL_p, aHR_p = m['pawlike_area_HL'], m['pawlike_area_HR']
                    if not (np.isnan(aHL_p) or np.isnan(aHR_p)) and aHR_p > 0:
                        m['pawlike_area_ratio_hind'] = round(aHL_p / aHR_p, 4)
                # Intensity ratios (paw-like)
                if 'pawlike_intensity_HL' in m and 'pawlike_intensity_HR' in m:
                    iHL_p, iHR_p = m['pawlike_intensity_HL'], m['pawlike_intensity_HR']
                    if not (np.isnan(iHL_p) or np.isnan(iHR_p)) and iHR_p > 0:
                        m['pawlike_intensity_ratio_hind'] = round(iHL_p / iHR_p, 4)

            return m

        # ── Overall summary ───────────────────────────────────────────────────
        self._log("  Computing metrics…")
        summary = _metrics()
        self._log("  Summary metrics done.")

        # ── Per-bin ───────────────────────────────────────────────────────────
        bin_rows = []
        bin_val  = params['bin_seconds']
        bin_unit = params.get('bin_unit', 'seconds')
        bin_sec  = bin_val * 60 if bin_unit == 'minutes' else bin_val
        if bin_sec > 0:
            bin_frames = max(1, round(bin_sec * fps))
            n_bins = n_frames // bin_frames
            for i in range(n_bins):
                start = i * bin_frames
                end   = min(start + bin_frames, n_frames)
                row = _metrics(slice(start, end))
                row['bin_index']   = i
                row['bin_start_s'] = round(start / fps, 2) if fps > 0 else start
                row['bin_end_s']   = round(end   / fps, 2) if fps > 0 else end
                bin_rows.append(row)
            self._log(f"  {n_bins} time bins computed.")

        # Store intermediates for post-analysis contact re-adjustment
        self._session_intermediates[sess['session_name']] = {
            'height_df': height_df,
            'bp_xcord': bp_xcord,
            'bp_ycord': bp_ycord,
            'bp_prob': bp_prob,
            'fps': fps,
            '_used_fallback_fps': _used_fallback_fps,
            'n_frames': n_frames,
            'active_paws': active_paws,
            'contact_masks': contact_masks,
            'paw_xy': paw_xy,
            'brightness_series': brightness_series,
            'paw_contour_data': paw_contour_data,
            'confidence_mask': confidence_mask,
            'loco_mask': loco_mask,
            'body_speed': body_speed,
            'frame_displacements': frame_displacements,
            'params': params,
        }

        return {'summary': summary, 'bins': bin_rows}

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis: completion callback (main thread)
    # ═══════════════════════════════════════════════════════════════════════

    def _on_analysis_complete(self, summary_rows, bin_rows):
        self._run_btn.config(state='normal')
        self._cancel_btn.config(state='disabled')
        self._progress_frame.pack_forget()
        self._sub_progress_label.config(text='')

        if not summary_rows:
            self._log_ui("No results — all sessions were skipped or failed. Check log for ERROR messages.")
            return

        self._summary_df = pd.DataFrame(summary_rows)
        self._bins_df    = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame()

        self._refresh_results_table()
        self._refresh_summary_panel()
        self._export_sum_btn.config(state='normal')
        if not self._bins_df.empty:
            self._export_bin_btn.config(state='normal')
        if _PLOT_OK:
            self._graphs_btn.config(state='normal')
        if self._session_intermediates:
            self._adjust_contact_btn.config(state='normal')

        # Warn about fallback FPS usage
        fallback_sessions = [r.get('session', '?') for r in summary_rows
                             if r.get('fallback_fps_used', False)]
        if fallback_sessions:
            self._log_ui(f"  ⚠ {len(fallback_sessions)} session(s) used fallback FPS "
                         f"(video FPS could not be detected).")

        self._log_ui(f"Done. {len(summary_rows)} session(s) processed.")

    # ═══════════════════════════════════════════════════════════════════════
    # Post-analysis contact re-adjustment
    # ═══════════════════════════════════════════════════════════════════════

    def _open_contact_adjustment(self):
        """Open dialog to adjust contact detection params and recompute metrics."""
        if not self._session_intermediates:
            messagebox.showinfo("No data", "Run analysis first.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Adjust Contact Detection")
        win.resizable(False, False)
        win.grab_set()

        ttk.Label(win, text="Re-compute contact masks with new parameters",
                  font=('Arial', 10, 'bold')).pack(padx=12, pady=(10, 6))

        frm = ttk.Frame(win, padding=10)
        frm.pack(fill='x')

        # Contact method
        ttk.Label(frm, text="Method:").grid(row=0, column=0, sticky='w', pady=3)
        method_var = tk.StringVar(value=self._contact_method_var.get())
        rb_frame = ttk.Frame(frm)
        rb_frame.grid(row=0, column=1, sticky='w', padx=4)
        for txt, val in [("Height", "height"), ("Speed", "speed"),
                         ("Combined", "combined")]:
            ttk.Radiobutton(rb_frame, text=txt, variable=method_var,
                            value=val).pack(side='left', padx=(0, 6))

        # Contact threshold
        ttk.Label(frm, text="Contact thresh (px):").grid(
            row=1, column=0, sticky='w', pady=3)
        ct_var = tk.IntVar(value=self._contact_thresh_var.get())
        ttk.Spinbox(frm, from_=0, to=500, textvariable=ct_var, width=8).grid(
            row=1, column=1, sticky='w', padx=4)

        # Speed threshold
        ttk.Label(frm, text="Speed thresh (px/s):").grid(
            row=2, column=0, sticky='w', pady=3)
        spd_var = tk.StringVar(value=self._speed_thresh_var.get())
        ttk.Entry(frm, textvariable=spd_var, width=10).grid(
            row=2, column=1, sticky='w', padx=4)

        # Median filter
        ttk.Label(frm, text="Median filter (ms):").grid(
            row=3, column=0, sticky='w', pady=3)
        med_var = tk.IntVar(value=self._median_filter_var.get())
        ttk.Spinbox(frm, from_=0, to=500, textvariable=med_var, width=8).grid(
            row=3, column=1, sticky='w', padx=4)

        # Min bout
        ttk.Label(frm, text="Min bout (ms):").grid(
            row=4, column=0, sticky='w', pady=3)
        bout_var = tk.IntVar(value=self._min_bout_var.get())
        ttk.Spinbox(frm, from_=0, to=500, textvariable=bout_var, width=8).grid(
            row=4, column=1, sticky='w', padx=4)

        # Brightness weight
        ttk.Label(frm, text="Brt contact weight:").grid(
            row=5, column=0, sticky='w', pady=3)
        bw_var = tk.DoubleVar(value=self._brt_weight_var.get())
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05,
                    textvariable=bw_var, width=8, format='%.2f').grid(
            row=5, column=1, sticky='w', padx=4)

        # Buttons
        dlg_btn_row = ttk.Frame(win)
        dlg_btn_row.pack(pady=(6, 12))

        def _apply():
            win.destroy()
            self._recompute_contact({
                'contact_method': method_var.get(),
                'contact_threshold': ct_var.get(),
                'speed_threshold': spd_var.get().strip(),
                'median_filter_ms': med_var.get(),
                'min_bout_ms': bout_var.get(),
                'brt_weight': bw_var.get(),
            })

        ttk.Button(dlg_btn_row, text="Apply", command=_apply).pack(
            side='left', padx=6)
        ttk.Button(dlg_btn_row, text="Cancel", command=win.destroy).pack(
            side='left', padx=6)

    def _recompute_contact(self, new_params):
        """Recompute contact masks and metrics using stored intermediates."""
        self._log_ui("Re-computing contact with adjusted parameters...")

        # Parse speed threshold
        speed_thresh_raw = new_params.get('speed_threshold', 'auto')
        if isinstance(speed_thresh_raw, str) and speed_thresh_raw.lower() != 'auto':
            try:
                speed_thresh = float(speed_thresh_raw)
            except ValueError:
                speed_thresh = 'auto'
        else:
            speed_thresh = 'auto' if speed_thresh_raw == 'auto' else speed_thresh_raw

        contact_method = new_params['contact_method']
        thresh = new_params['contact_threshold']

        summary_rows = []
        bin_rows = []

        for session_name, inter in self._session_intermediates.items():
            height_df = inter['height_df']
            bp_xcord = inter['bp_xcord']
            bp_ycord = inter['bp_ycord']
            fps = inter['fps']
            n_frames = inter['n_frames']
            active_paws = inter['active_paws']
            paw_xy = inter['paw_xy']
            brightness_series = inter['brightness_series']
            paw_contour_data = inter['paw_contour_data']
            confidence_mask = inter['confidence_mask']
            loco_mask = inter['loco_mask']
            body_speed = inter.get('body_speed')
            frame_displacements = inter.get('frame_displacements')
            _used_fallback_fps = inter['_used_fallback_fps']
            params = inter['params']

            # Helper to get x,y arrays for a body part
            def _get_xy(bp, _bpx=bp_xcord, _bpy=bp_ycord, _nf=n_frames):
                x_col = next((c for c in _bpx.columns
                              if bp.lower() in c.lower()), None)
                y_col = next((c for c in _bpy.columns
                              if bp.lower() in c.lower()), None)
                if x_col and y_col:
                    return (_bpx[x_col].values[:_nf].astype(float),
                            _bpy[y_col].values[:_nf].astype(float))
                return None, None

            # Rebuild contact masks
            contact_masks = {}
            for role, bp in active_paws.items():
                px, py = _get_xy(bp)

                h_col = f'{bp}_Height'
                if h_col not in height_df.columns:
                    matches = [c for c in height_df.columns
                               if bp.lower() in c.lower()]
                    h_col = matches[0] if matches else None

                height_mask = None
                if h_col:
                    height_mask = (height_df[h_col].values[:n_frames] < thresh)

                speed_mask = None
                if contact_method in ('speed', 'combined') and px is not None:
                    speed_mask = self._compute_speed_contact(
                        px, py, fps,
                        threshold=speed_thresh,
                        median_ms=new_params.get('median_filter_ms', 50),
                        min_bout_ms=new_params.get('min_bout_ms', 30))

                if contact_method == 'height':
                    mask = height_mask
                elif contact_method == 'speed':
                    mask = speed_mask if speed_mask is not None else height_mask
                elif contact_method == 'combined':
                    if height_mask is not None and speed_mask is not None:
                        mask = height_mask & speed_mask
                    else:
                        mask = height_mask if height_mask is not None else speed_mask
                else:
                    mask = height_mask

                if mask is not None:
                    contact_masks[role] = pd.Series(
                        mask, dtype=bool).reset_index(drop=True)

            # Brightness-weighted contact refinement
            brt_weight = new_params.get('brt_weight', 0.0)
            if brt_weight > 0 and brightness_series:
                for role, bp in active_paws.items():
                    if role not in brightness_series or role not in contact_masks:
                        continue
                    h_col = f'{bp}_Height'
                    if h_col not in height_df.columns:
                        matches = [c for c in height_df.columns
                                   if bp.lower() in c.lower()]
                        h_col = matches[0] if matches else None
                    if h_col is None:
                        continue
                    h_vals = height_df[h_col].values[:n_frames].astype(float)
                    b_vals = brightness_series[role].values.astype(float)
                    h_score = np.clip(
                        1.0 - h_vals / max(float(thresh), 1.0), 0.0, 1.0)
                    valid_b = b_vals[np.isfinite(b_vals) & (b_vals > 0)]
                    brt_90 = (float(np.percentile(valid_b, 90))
                              if len(valid_b) > 0 else 1.0)
                    b_score = np.clip(b_vals / max(brt_90, 1.0), 0.0, 1.0)
                    combined = ((1.0 - brt_weight) * h_score
                                + brt_weight * b_score)
                    contact_masks[role] = pd.Series(combined > 0.5, dtype=bool)

            # Update stored intermediates with new contact masks
            inter['contact_masks'] = contact_masks

            # Metric computation (mirrors _metrics in _analyze_session)
            def _metrics(frame_slice=None,
                         _cm=contact_masks, _nf=n_frames, _fps=fps,
                         _ufps=_used_fallback_fps, _bs=brightness_series,
                         _pxy=paw_xy, _pcd=paw_contour_data,
                         _conf=confidence_mask, _loco=loco_mask,
                         _ap=active_paws,
                         _fd=frame_displacements, _bs_spd=body_speed):
                if frame_slice is not None:
                    masks = {r: m.iloc[frame_slice].reset_index(drop=True)
                             for r, m in _cm.items()}
                    n = len(next(iter(masks.values()))) if masks else 0
                else:
                    masks = _cm
                    n = _nf

                m = {'n_frames': n, 'fps': round(_fps, 2),
                     'fallback_fps_used': _ufps}

                for role, mask in masks.items():
                    m[f'contact_pct_{role}'] = round(
                        float(mask.mean()) * 100, 2)
                    m[f'n_contact_{role}'] = int(mask.sum())

                if 'HL' in masks and 'HR' in masks:
                    cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
                    tot = cHL + cHR
                    m['WBI_hind'] = round(cHL / tot * 100, 2) if tot > 0 else float('nan')
                    m['SI_hind'] = round((cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')

                if 'FL' in masks and 'FR' in masks:
                    cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
                    tot = cFL + cFR
                    m['WBI_fore'] = round(cFL / tot * 100, 2) if tot > 0 else float('nan')
                    m['SI_fore'] = round((cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

                total_contact = sum(m[f'contact_pct_{r}'] for r in masks)
                if total_contact > 0:
                    for role in masks:
                        m[f'WBI_{role}'] = round(
                            m[f'contact_pct_{role}'] / total_contact * 100, 2)

                if 'HL' in masks and 'HR' in masks:
                    cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
                    tot = cHL + cHR
                    m['SBI_hind'] = round(2 * abs(cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')
                if 'FL' in masks and 'FR' in masks:
                    cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
                    tot = cFL + cFR
                    m['SBI_fore'] = round(2 * abs(cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

                if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                    hind = (m['contact_pct_HL'] + m['contact_pct_HR']) / 2
                    fore = (m['contact_pct_FL'] + m['contact_pct_FR']) / 2
                    m['hind_fore_ratio'] = round(hind / fore, 4) if fore > 0 else float('nan')

                # Brightness during contact
                for role, brt_full in _bs.items():
                    mask = masks.get(role)
                    if mask is None:
                        continue
                    if frame_slice is not None:
                        brt_slice = brt_full.iloc[frame_slice].reset_index(
                            drop=True)
                    else:
                        brt_slice = brt_full
                    contact_brt = brt_slice.values[mask.values.astype(bool)]
                    if len(contact_brt) > 0:
                        m[f'brightness_{role}'] = round(
                            float(np.nanmean(contact_brt)), 4)
                    else:
                        m[f'brightness_{role}'] = float('nan')

                if 'brightness_HL' in m and 'brightness_HR' in m:
                    bHL = m['brightness_HL']
                    bHR = m['brightness_HR']
                    if (not (np.isnan(bHL) or np.isnan(bHR))) and bHR > 0:
                        m['brightness_ratio_HL_HR'] = round(bHL / bHR, 4)

                # ── Distance & movement time ──────────────────────────────
                if _fd is not None:
                    disp_sl = _fd[frame_slice] if frame_slice is not None else _fd
                    m['total_distance'] = round(float(np.nansum(disp_sl)), 2)
                    if _loco is not None:
                        lm_sl = _loco[frame_slice] if frame_slice is not None else _loco
                        lm_sl = lm_sl[:len(disp_sl)]
                        m['loco_total_distance'] = round(float(np.nansum(disp_sl[lm_sl])), 2)
                        m['time_moving_s'] = round(float(lm_sl.sum()) / _fps, 2) if _fps > 0 else 0.0
                        m['time_moving_pct'] = round(float(lm_sl.mean()) * 100, 2)

                # ── Body speed mean ───────────────────────────────────────
                if _bs_spd is not None:
                    bs_sl = _bs_spd[frame_slice] if frame_slice is not None else _bs_spd
                    m['body_speed_mean'] = round(float(np.nanmean(bs_sl)), 2)
                    if _loco is not None:
                        lm_sl = _loco[frame_slice] if frame_slice is not None else _loco
                        lm_sl = lm_sl[:len(bs_sl)]
                        loco_bs = bs_sl[lm_sl]
                        m['body_speed_loco'] = round(float(np.nanmean(loco_bs)), 2) if len(loco_bs) > 0 else float('nan')

                # ── Support patterns ──────────────────────────────────────
                if len(masks) >= 4 and _loco is not None:
                    lm_sl = _loco[frame_slice] if frame_slice is not None else _loco
                    all_contact = np.zeros(n, dtype=int)
                    for role_sp, mask_sp in masks.items():
                        ma = mask_sp.values.astype(bool) if hasattr(mask_sp, 'values') else np.asarray(mask_sp, dtype=bool)
                        all_contact[:len(ma)] += ma[:n].astype(int)
                    lm_frames = lm_sl[:n]
                    n_loco_frames = int(lm_frames.sum())
                    if n_loco_frames > 0:
                        for npaws in range(5):
                            cnt = int(((all_contact == npaws) & lm_frames).sum())
                            m[f'support_{npaws}paw_pct'] = round(cnt / n_loco_frames * 100, 2)

                # ── Gait metrics (dual: all frames + locomotion only) ─────
                def _gait_block(m, masks, loco_filter_mask, prefix=''):
                    """Compute gait timing, spatial, phase, symmetry metrics with optional prefix."""
                    for role, mask in masks.items():
                        mask_arr = (mask.values.astype(bool) if hasattr(mask, 'values')
                                    else np.asarray(mask, dtype=bool))

                        # Apply confidence and locomotion filters to gait computation
                        gait_valid = np.ones(len(mask_arr), dtype=bool)
                        if _conf is not None:
                            cm_sl = (_conf[frame_slice] if frame_slice is not None
                                     else _conf)
                            if len(cm_sl) == len(mask_arr):
                                gait_valid &= cm_sl
                        if loco_filter_mask is not None:
                            lm_sl = (loco_filter_mask[frame_slice] if frame_slice is not None
                                     else loco_filter_mask)
                            if len(lm_sl) == len(mask_arr):
                                gait_valid &= lm_sl

                        # Mask out invalid frames for gait analysis
                        gait_contact = mask_arr & gait_valid

                        _min_stance_r = params.get('min_stance_ms', 0)
                        stance_durs, swing_durs, stance_onsets = (
                            GaitLimbTab._gait_bouts(gait_contact, _fps,
                                                     min_stride_ms=_min_stance_r))

                        if stance_durs:
                            m[f'{prefix}stance_dur_{role}'] = round(
                                float(np.mean(stance_durs)), 4)
                            m[f'{prefix}n_strides_{role}'] = len(stance_durs)
                        else:
                            m[f'{prefix}stance_dur_{role}'] = float('nan')
                            m[f'{prefix}n_strides_{role}'] = 0

                        if swing_durs:
                            m[f'{prefix}swing_dur_{role}'] = round(
                                float(np.mean(swing_durs)), 4)
                        else:
                            m[f'{prefix}swing_dur_{role}'] = float('nan')

                        if stance_durs and swing_durs:
                            stride_dur = (m[f'{prefix}stance_dur_{role}']
                                          + m[f'{prefix}swing_dur_{role}'])
                            m[f'{prefix}stride_dur_{role}'] = round(stride_dur, 4)
                            m[f'{prefix}duty_cycle_{role}'] = (
                                round(m[f'{prefix}stance_dur_{role}'] / stride_dur * 100, 2)
                                if stride_dur > 0 else float('nan'))
                            m[f'{prefix}cadence_{role}'] = (
                                round(60.0 / stride_dur, 2)
                                if stride_dur > 0 else float('nan'))
                        else:
                            m[f'{prefix}stride_dur_{role}'] = float('nan')
                            m[f'{prefix}duty_cycle_{role}'] = float('nan')
                            m[f'{prefix}cadence_{role}'] = float('nan')

                        # Stride length
                        if role in _pxy and len(stance_onsets) >= 2:
                            px_arr, py_arr = _pxy[role]
                            if frame_slice is not None:
                                sl_start = frame_slice.start or 0
                                abs_onsets = [sl_start + o for o in stance_onsets
                                              if sl_start + o < len(px_arr)]
                            else:
                                abs_onsets = [o for o in stance_onsets
                                              if o < len(px_arr)]
                            if len(abs_onsets) >= 2:
                                dists = []
                                for j in range(1, len(abs_onsets)):
                                    i0, i1 = abs_onsets[j - 1], abs_onsets[j]
                                    d = np.sqrt((px_arr[i1] - px_arr[i0])**2
                                                + (py_arr[i1] - py_arr[i0])**2)
                                    dists.append(d)
                                m[f'{prefix}stride_len_{role}'] = round(
                                    float(np.mean(dists)), 2)
                            else:
                                m[f'{prefix}stride_len_{role}'] = float('nan')
                        else:
                            m[f'{prefix}stride_len_{role}'] = float('nan')

                        # Swing speed = stride_length / swing_duration
                        sl_key = f'{prefix}stride_len_{role}'
                        sw_key = f'{prefix}swing_dur_{role}'
                        if (sl_key in m and sw_key in m
                                and not np.isnan(m[sl_key]) and not np.isnan(m[sw_key])
                                and m[sw_key] > 0):
                            m[f'{prefix}swing_speed_{role}'] = round(m[sl_key] / m[sw_key], 2)
                        else:
                            m[f'{prefix}swing_speed_{role}'] = float('nan')

                        # Stride-to-stride CV
                        if len(stance_durs) >= 3:
                            sd_arr = np.array(stance_durs)
                            sd_mean = np.mean(sd_arr)
                            if sd_mean > 0:
                                m[f'{prefix}stride_cv_{role}'] = round(
                                    float(np.std(sd_arr, ddof=1) / sd_mean * 100), 2)
                            else:
                                m[f'{prefix}stride_cv_{role}'] = float('nan')
                        else:
                            m[f'{prefix}stride_cv_{role}'] = float('nan')

                    # ── Step length / width (contralateral) ──────────────
                    for pair_name, left_role, right_role in [
                            ('hind', 'HL', 'HR'), ('fore', 'FL', 'FR')]:
                        if left_role not in masks or right_role not in masks:
                            continue
                        l_mask = (masks[left_role].values.astype(bool)
                                  if hasattr(masks[left_role], 'values')
                                  else np.asarray(masks[left_role], dtype=bool))
                        r_mask = (masks[right_role].values.astype(bool)
                                  if hasattr(masks[right_role], 'values')
                                  else np.asarray(masks[right_role], dtype=bool))

                        # Apply filters for loco variant
                        if loco_filter_mask is not None:
                            lm_slice = (loco_filter_mask[frame_slice] if frame_slice is not None
                                        else loco_filter_mask)
                            l_mask = l_mask & lm_slice[:len(l_mask)]
                            r_mask = r_mask & lm_slice[:len(r_mask)]
                        if _conf is not None:
                            cm_slice = (_conf[frame_slice] if frame_slice is not None
                                        else _conf)
                            l_mask = l_mask & cm_slice[:len(l_mask)]
                            r_mask = r_mask & cm_slice[:len(r_mask)]

                        _, _, l_onsets = GaitLimbTab._gait_bouts(l_mask, _fps, _min_stance_r)
                        _, _, r_onsets = GaitLimbTab._gait_bouts(r_mask, _fps, _min_stance_r)

                        if (left_role in _pxy and right_role in _pxy
                                and l_onsets and r_onsets):
                            lpx, lpy = _pxy[left_role]
                            rpx, rpy = _pxy[right_role]
                            sl_start = ((frame_slice.start or 0)
                                        if frame_slice is not None else 0)
                            all_strikes = sorted(
                                [('L', sl_start + o) for o in l_onsets]
                                + [('R', sl_start + o) for o in r_onsets],
                                key=lambda x: x[1])
                            step_lens = []
                            for j in range(1, len(all_strikes)):
                                if all_strikes[j][0] != all_strikes[j - 1][0]:
                                    i0 = all_strikes[j - 1][1]
                                    i1 = all_strikes[j][1]
                                    if i0 < len(lpx) and i1 < len(lpx):
                                        if all_strikes[j - 1][0] == 'L':
                                            x0, y0 = lpx[i0], lpy[i0]
                                        else:
                                            x0, y0 = rpx[i0], rpy[i0]
                                        if all_strikes[j][0] == 'L':
                                            x1, y1 = lpx[i1], lpy[i1]
                                        else:
                                            x1, y1 = rpx[i1], rpy[i1]
                                        step_lens.append(
                                            np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
                            m[f'{prefix}step_len_{pair_name}'] = (
                                round(float(np.mean(step_lens)), 2)
                                if step_lens else float('nan'))
                            widths = []
                            for lo in l_onsets:
                                abs_lo = sl_start + lo
                                nearest_r = min(
                                    r_onsets, key=lambda ro: abs(ro - lo),
                                    default=None)
                                if nearest_r is not None:
                                    abs_r = sl_start + nearest_r
                                    if abs_lo < len(lpx) and abs_r < len(rpx):
                                        widths.append(
                                            abs(lpy[abs_lo] - rpy[abs_r]))
                            m[f'{prefix}step_width_{pair_name}'] = (
                                round(float(np.mean(widths)), 2)
                                if widths else float('nan'))
                        else:
                            m[f'{prefix}step_len_{pair_name}'] = float('nan')
                            m[f'{prefix}step_width_{pair_name}'] = float('nan')

                    # ── Interlimb coordination (phase) ───────────────────
                    if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                        def _phase(ref_role, test_role, _m=masks):
                            ref_arr = (_m[ref_role].values.astype(bool)
                                       if hasattr(_m[ref_role], 'values')
                                       else np.asarray(_m[ref_role], dtype=bool))
                            tst_arr = (_m[test_role].values.astype(bool)
                                       if hasattr(_m[test_role], 'values')
                                       else np.asarray(_m[test_role], dtype=bool))

                            if loco_filter_mask is not None:
                                lm_slice = (loco_filter_mask[frame_slice] if frame_slice is not None
                                            else loco_filter_mask)
                                ref_arr = ref_arr & lm_slice[:len(ref_arr)]
                                tst_arr = tst_arr & lm_slice[:len(tst_arr)]
                            if _conf is not None:
                                cm_slice = (_conf[frame_slice] if frame_slice is not None
                                            else _conf)
                                ref_arr = ref_arr & cm_slice[:len(ref_arr)]
                                tst_arr = tst_arr & cm_slice[:len(tst_arr)]

                            _, _, ref_on = GaitLimbTab._gait_bouts(
                                ref_arr, _fps, _min_stance_r)
                            _, _, tst_on = GaitLimbTab._gait_bouts(
                                tst_arr, _fps, _min_stance_r)
                            if len(ref_on) < 3 or not tst_on:
                                return float('nan')
                            phases = []
                            for i in range(len(ref_on) - 1):
                                cycle_len = ref_on[i + 1] - ref_on[i]
                                if cycle_len <= 0:
                                    continue
                                for to in tst_on:
                                    if ref_on[i] <= to < ref_on[i + 1]:
                                        phases.append(
                                            (to - ref_on[i]) / cycle_len)
                            return (round(float(np.mean(phases)), 3)
                                    if phases else float('nan'))
                        m[f'{prefix}phase_HL_HR'] = _phase('HR', 'HL')
                        m[f'{prefix}phase_diagonal'] = _phase('HR', 'FL')
                        m[f'{prefix}phase_FL_FR'] = _phase('FR', 'FL')
                        m[f'{prefix}phase_HL_FL'] = _phase('FL', 'HL')
                        m[f'{prefix}phase_HR_FR'] = _phase('FR', 'HR')
                        m[f'{prefix}phase_HL_FR'] = _phase('FR', 'HL')

                    # ── Regularity index ──────────────────────────────────
                    if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                        ri = GaitLimbTab._regularity_index(
                            masks, _fps, frame_slice,
                            loco_filter_mask, _conf, _min_stance_r)
                        if ri is not None:
                            m[f'{prefix}regularity_index'] = ri

                    # ── Print position ────────────────────────────────────
                    if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                        for side_lbl, hind_r, fore_r in [('L', 'HL', 'FL'), ('R', 'HR', 'FR')]:
                            if hind_r in _pxy and fore_r in _pxy:
                                pp = GaitLimbTab._print_position(
                                    masks, _pxy, hind_r, fore_r, _fps, frame_slice,
                                    loco_filter_mask, _conf, _min_stance_r)
                                m[f'{prefix}print_position_{side_lbl}'] = pp

                    # ── Gait symmetry indices ────────────────────────────
                    if f'{prefix}stance_dur_HL' in m and f'{prefix}stance_dur_HR' in m:
                        sHL = m[f'{prefix}stance_dur_HL']
                        sHR = m[f'{prefix}stance_dur_HR']
                        tot = sHL + sHR
                        m[f'{prefix}stance_SI_hind'] = (
                            round((sHL - sHR) / tot * 100, 2)
                            if tot > 0 and not (np.isnan(sHL) or np.isnan(sHR))
                            else float('nan'))
                    if f'{prefix}stride_len_HL' in m and f'{prefix}stride_len_HR' in m:
                        lHL = m[f'{prefix}stride_len_HL']
                        lHR = m[f'{prefix}stride_len_HR']
                        tot = lHL + lHR
                        m[f'{prefix}stride_len_SI_hind'] = (
                            round((lHL - lHR) / tot * 100, 2)
                            if tot > 0 and not (np.isnan(lHL) or np.isnan(lHR))
                            else float('nan'))

                _gait_block(m, masks, None, prefix='')           # All frames
                if _loco is not None:
                    _gait_block(m, masks, _loco, prefix='loco_')  # Locomotion only

                # Paw contour
                if _pcd:
                    # Build full-stance mask (all contour paws in contact)
                    _stance_mask_all = None
                    if masks:
                        _c_roles = [r for r in _pcd if r in masks]
                        _stance_arrays = [masks[r].values.astype(bool) if hasattr(masks[r], 'values')
                                          else np.asarray(masks[r], dtype=bool) for r in _c_roles]
                        if _stance_arrays:
                            _stance_mask_all = _stance_arrays[0].copy()
                            for _sm in _stance_arrays[1:]:
                                _sml = min(len(_stance_mask_all), len(_sm))
                                _stance_mask_all = _stance_mask_all[:_sml] & _sm[:_sml]

                    for role in list(_pcd.keys()):
                        areas_full = _pcd[role]['areas']
                        spreads_full = _pcd[role]['spreads']
                        intensities_full = _pcd[role]['intensities']
                        mask_arr = (masks[role].values.astype(bool)
                                    if role in masks
                                    else np.ones(n, dtype=bool))
                        sl = frame_slice
                        areas_sl = areas_full[sl] if sl is not None else areas_full
                        spreads_sl = spreads_full[sl] if sl is not None else spreads_full
                        ints_sl = intensities_full[sl] if sl is not None else intensities_full
                        mask_sl = mask_arr

                        widths_full = _pcd[role].get('widths')
                        solidities_full = _pcd[role].get('solidities')
                        aspect_ratios_full = _pcd[role].get('aspect_ratios')
                        circularities_full = _pcd[role].get('circularities')
                        widths_sl = (widths_full[sl] if sl is not None else widths_full) if widths_full is not None else None
                        solidities_sl = (solidities_full[sl] if sl is not None else solidities_full) if solidities_full is not None else None
                        ar_sl = (aspect_ratios_full[sl] if sl is not None else aspect_ratios_full) if aspect_ratios_full is not None else None
                        circ_sl = (circularities_full[sl] if sl is not None else circularities_full) if circularities_full is not None else None

                        # ── Regular (per-paw contact mask) ──
                        valid = mask_sl & (areas_sl[:len(mask_sl)] > 0)
                        _ca = areas_sl[:len(mask_sl)][valid]
                        m[f'paw_area_{role}'] = round(float(np.nanmean(_ca)), 2) if len(_ca) > 0 else float('nan')
                        _cs = spreads_sl[:len(mask_sl)][valid]
                        m[f'paw_spread_{role}'] = round(float(np.nanmean(_cs)), 2) if len(_cs) > 0 else float('nan')
                        _ci = ints_sl[:len(mask_sl)][valid]
                        m[f'contact_intensity_{role}'] = round(float(np.nanmean(_ci)), 2) if len(_ci) > 0 else float('nan')
                        if widths_sl is not None:
                            m[f'paw_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid])), 2) if valid.any() else float('nan')
                            m[f'paw_solidity_{role}'] = round(float(np.nanmean(solidities_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                            m[f'paw_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                            m[f'paw_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')

                        # ── Full-stance variant ──
                        if _stance_mask_all is not None:
                            _stance_sl = _stance_mask_all[sl] if sl is not None else _stance_mask_all
                            _ml = min(len(mask_sl), len(_stance_sl))
                            stance_valid = mask_sl[:_ml] & _stance_sl[:_ml] & (areas_sl[:_ml] > 0)
                            _sca = areas_sl[:_ml][stance_valid]
                            m[f'paw_area_stance_{role}'] = round(float(np.nanmean(_sca)), 2) if len(_sca) > 0 else float('nan')
                            _scs = spreads_sl[:_ml][stance_valid]
                            m[f'paw_spread_stance_{role}'] = round(float(np.nanmean(_scs)), 2) if len(_scs) > 0 else float('nan')
                            _sci = ints_sl[:_ml][stance_valid]
                            m[f'contact_intensity_stance_{role}'] = round(float(np.nanmean(_sci)), 2) if len(_sci) > 0 else float('nan')
                            if widths_sl is not None:
                                m[f'paw_width_stance_{role}'] = round(float(np.nanmean(widths_sl[:_ml][stance_valid])), 2) if stance_valid.any() else float('nan')
                                m[f'paw_solidity_stance_{role}'] = round(float(np.nanmean(solidities_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                                m[f'paw_aspect_ratio_stance_{role}'] = round(float(np.nanmean(ar_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                                m[f'paw_circularity_stance_{role}'] = round(float(np.nanmean(circ_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')

                        # ── Paw-like filtered (solidity + AR + circularity) ──
                        PAWLIKE_SOL = self._pawlike_thresholds.get('solidity', 1.00)
                        PAWLIKE_AR = self._pawlike_thresholds.get('aspect_ratio', 1.6)
                        PAWLIKE_CIRC = self._pawlike_thresholds.get('circularity', 0.10)
                        if solidities_sl is not None:
                            sol_arr = solidities_sl[:len(mask_sl)]
                            valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                            if ar_sl is not None:
                                valid_paw = valid_paw & (ar_sl[:len(mask_sl)] <= PAWLIKE_AR)
                            if circ_sl is not None:
                                valid_paw = valid_paw & (circ_sl[:len(mask_sl)] >= PAWLIKE_CIRC)
                            _pca = areas_sl[:len(mask_sl)][valid_paw]
                            m[f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                            _pcs = spreads_sl[:len(mask_sl)][valid_paw]
                            m[f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                            _pci = ints_sl[:len(mask_sl)][valid_paw]
                            m[f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                            if widths_sl is not None:
                                m[f'pawlike_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid_paw])), 2) if valid_paw.any() else float('nan')
                                m[f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                                m[f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')
                                m[f'pawlike_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')

                    # Area ratios (regular)
                    if 'paw_area_HL' in m and 'paw_area_HR' in m:
                        aHL, aHR = m['paw_area_HL'], m['paw_area_HR']
                        if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                            m['paw_area_ratio_hind'] = round(aHL / aHR, 4)
                    # Area ratios (stance)
                    if 'paw_area_stance_HL' in m and 'paw_area_stance_HR' in m:
                        aHL_s, aHR_s = m['paw_area_stance_HL'], m['paw_area_stance_HR']
                        if not (np.isnan(aHL_s) or np.isnan(aHR_s)) and aHR_s > 0:
                            m['paw_area_ratio_stance_hind'] = round(aHL_s / aHR_s, 4)
                    # Intensity ratios (regular)
                    if 'contact_intensity_HL' in m and 'contact_intensity_HR' in m:
                        iHL, iHR = m['contact_intensity_HL'], m['contact_intensity_HR']
                        if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                            m['contact_intensity_ratio_hind'] = round(iHL / iHR, 4)
                    # Intensity ratios (stance)
                    if 'contact_intensity_stance_HL' in m and 'contact_intensity_stance_HR' in m:
                        iHL_s, iHR_s = m['contact_intensity_stance_HL'], m['contact_intensity_stance_HR']
                        if not (np.isnan(iHL_s) or np.isnan(iHR_s)) and iHR_s > 0:
                            m['contact_intensity_ratio_stance_hind'] = round(iHL_s / iHR_s, 4)
                    # Area ratios (paw-like)
                    if 'pawlike_area_HL' in m and 'pawlike_area_HR' in m:
                        aHL_p, aHR_p = m['pawlike_area_HL'], m['pawlike_area_HR']
                        if not (np.isnan(aHL_p) or np.isnan(aHR_p)) and aHR_p > 0:
                            m['pawlike_area_ratio_hind'] = round(aHL_p / aHR_p, 4)
                    # Intensity ratios (paw-like)
                    if 'pawlike_intensity_HL' in m and 'pawlike_intensity_HR' in m:
                        iHL_p, iHR_p = m['pawlike_intensity_HL'], m['pawlike_intensity_HR']
                        if not (np.isnan(iHL_p) or np.isnan(iHR_p)) and iHR_p > 0:
                            m['pawlike_intensity_ratio_hind'] = round(iHL_p / iHR_p, 4)

                return m

            # Overall + bins
            summary = _metrics()
            bin_rows_sess = []
            bin_val = params['bin_seconds']
            bin_unit = params.get('bin_unit', 'seconds')
            bin_sec = bin_val * 60 if bin_unit == 'minutes' else bin_val
            if bin_sec > 0:
                bin_frames = max(1, round(bin_sec * fps))
                for i in range(n_frames // bin_frames):
                    start = i * bin_frames
                    end = min(start + bin_frames, n_frames)
                    row = _metrics(slice(start, end))
                    row['bin_index'] = i
                    row['bin_start_s'] = (round(start / fps, 2)
                                          if fps > 0 else start)
                    row['bin_end_s'] = (round(end / fps, 2)
                                        if fps > 0 else end)
                    bin_rows_sess.append(row)

            subj = self._resolve_subject(session_name)
            treatment = self._get_treatment(subj)
            base = dict(session=session_name, subject=subj,
                        treatment=treatment)
            summary_rows.append({**base, **summary})
            for brow in bin_rows_sess:
                bin_rows.append({**base, **brow})

        # Update results
        self._summary_df = pd.DataFrame(summary_rows)
        self._bins_df = (pd.DataFrame(bin_rows) if bin_rows
                         else pd.DataFrame())
        self._refresh_results_table()
        self._export_sum_btn.config(state='normal')
        if not self._bins_df.empty:
            self._export_bin_btn.config(state='normal')
        self._log_ui(
            f"Contact re-computation complete. "
            f"{len(summary_rows)} session(s) updated.")

    # ═══════════════════════════════════════════════════════════════════════
    # Results table
    # ═══════════════════════════════════════════════════════════════════════

    def _refresh_results_table(self):
        for item in self._res_tree.get_children():
            self._res_tree.delete(item)
        if self._summary_df is None:
            return

        def _fmt(v):
            if isinstance(v, float):
                return '' if np.isnan(v) else f'{v:.1f}'
            return str(v) if v != '' else ''

        for _, row in self._summary_df.iterrows():
            self._res_tree.insert('', 'end', values=(
                row.get('session', ''),
                row.get('subject', ''),
                row.get('treatment', ''),
                _fmt(row.get('WBI_hind',              float('nan'))),
                _fmt(row.get('SI_hind',               float('nan'))),
                _fmt(row.get('WBI_fore',              float('nan'))),
                _fmt(row.get('SI_fore',               float('nan'))),
                _fmt(row.get('brightness_ratio_HL_HR', float('nan'))),
                _fmt(row.get('contact_pct_HL',        float('nan'))),
                _fmt(row.get('contact_pct_HR',        float('nan'))),
                _fmt(row.get('contact_pct_FL',        float('nan'))),
                _fmt(row.get('contact_pct_FR',        float('nan'))),
                _fmt(row.get('stance_dur_HL',         float('nan'))),
                _fmt(row.get('stance_dur_HR',         float('nan'))),
                _fmt(row.get('duty_cycle_HL',         float('nan'))),
                _fmt(row.get('duty_cycle_HR',         float('nan'))),
                _fmt(row.get('stride_len_HL',         float('nan'))),
            ))

    # ═══════════════════════════════════════════════════════════════════════
    # Export
    # ═══════════════════════════════════════════════════════════════════════

    def _export_summary(self):
        self._export_df(self._summary_df, 'wb_summary')

    def _export_bins(self):
        self._export_df(self._bins_df, 'wb_bins')

    def _export_df(self, df: pd.DataFrame, prefix: str):
        if df is None or df.empty:
            messagebox.showinfo("Nothing to export", "Run analysis first.", parent=self)
            return
        folder = self.app.current_project_folder.get()
        analysis_dir = os.path.join(folder, 'analysis') if folder else ''
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialdir=analysis_dir or None,
            initialfile=f'{prefix}_{ts}.csv',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            parent=self)
        if path:
            df.to_csv(path, index=False)
            self._log_ui(f"Saved: {os.path.basename(path)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Graphs
    # ═══════════════════════════════════════════════════════════════════════

    def _open_bin_graphs(self):
        if not _PLOT_OK:
            messagebox.showerror("Missing deps",
                                 "matplotlib and scipy are required for graphs.",
                                 parent=self)
            return
        if self._bins_df is None or self._bins_df.empty:
            messagebox.showinfo("No bin data",
                                "Run analysis with a bin size > 0 first.",
                                parent=self)
            return

        bdf = self._bins_df

        # Build list of (column, display_label, reference, y_axis_label) for scalar metrics
        _SCALAR = [
            ('WBI_hind',            'WBI Hind',           50.0,  'Weight Bearing Index — hind (%)'),
            ('SI_hind',             'SI Hind',             0.0,  'Symmetry Index — hind (%)'),
            ('SBI_hind',            'SBI Hind',            0.0,  'SBI — hind (%)'),
            ('WBI_fore',            'WBI Fore',           50.0,  'Weight Bearing Index — fore (%)'),
            ('SI_fore',             'SI Fore',             0.0,  'Symmetry Index — fore (%)'),
            ('SBI_fore',            'SBI Fore',            0.0,  'SBI — fore (%)'),
            ('contact_pct_HL',      'Contact % HL',       None,  'Contact % — HL'),
            ('contact_pct_HR',      'Contact % HR',       None,  'Contact % — HR'),
            ('contact_pct_FL',      'Contact % FL',       None,  'Contact % — FL'),
            ('contact_pct_FR',      'Contact % FR',       None,  'Contact % — FR'),
            ('brightness_HL',       'Brightness HL',      None,  'Mean brightness — HL'),
            ('brightness_HR',       'Brightness HR',      None,  'Mean brightness — HR'),
            ('brightness_FL',       'Brightness FL',      None,  'Mean brightness — FL'),
            ('brightness_FR',       'Brightness FR',      None,  'Mean brightness — FR'),
            ('brightness_ratio_HL_HR', 'Brightness Ratio HL/HR', 1.0, 'HL / HR brightness'),
            ('hind_fore_ratio',     'Hind/Fore Ratio',    None,  'Mean hind / mean fore contact %'),
            # Gait metrics
            ('stance_dur_HL',       'Stance Dur HL',      None,  'Stance duration (s) — HL'),
            ('stance_dur_HR',       'Stance Dur HR',      None,  'Stance duration (s) — HR'),
            ('swing_dur_HL',        'Swing Dur HL',       None,  'Swing duration (s) — HL'),
            ('swing_dur_HR',        'Swing Dur HR',       None,  'Swing duration (s) — HR'),
            ('duty_cycle_HL',       'Duty Cycle HL',      None,  'Duty cycle (%) — HL'),
            ('duty_cycle_HR',       'Duty Cycle HR',      None,  'Duty cycle (%) — HR'),
            ('cadence_HL',          'Cadence HL',         None,  'Cadence (strides/min) — HL'),
            ('cadence_HR',          'Cadence HR',         None,  'Cadence (strides/min) — HR'),
            ('stride_len_HL',       'Stride Len HL',      None,  'Stride length (px) — HL'),
            ('stride_len_HR',       'Stride Len HR',      None,  'Stride length (px) — HR'),
            ('step_len_hind',       'Step Len Hind',      None,  'Step length (px) — hind'),
            ('step_width_hind',     'Step Width Hind',    None,  'Step width (px) — hind'),
            ('stance_SI_hind',      'Stance SI Hind',      0.0,  'Stance symmetry index (%)'),
            ('stride_len_SI_hind',  'Stride Len SI Hind',  0.0,  'Stride length symmetry index (%)'),
            ('phase_HL_HR',         'Phase HL-HR',         0.5,  'HL-HR phase (0.5 = alternating)'),
        ]

        # Keep only metrics with actual data
        available_scalar = [(col, lbl, ref, ylbl)
                            for col, lbl, ref, ylbl in _SCALAR
                            if col in bdf.columns and bdf[col].notna().any()]

        # Build per-paw contact % checkbox options separately
        paw_tc_available = [(f'contact_pct_{r}', r)
                            for r in self.ROLES
                            if f'contact_pct_{r}' in bdf.columns
                            and bdf[f'contact_pct_{r}'].notna().any()]

        if not available_scalar and not paw_tc_available:
            messagebox.showinfo("No data", "No bin metrics available to plot.", parent=self)
            return

        # ── Selection dialog ──────────────────────────────────────────────────
        sel_win = tk.Toplevel(self)
        sel_win.title("Select bin metrics to graph")
        sel_win.resizable(False, False)
        sel_win.grab_set()

        ttk.Label(sel_win, text="Choose metrics to display over time bins:",
                  font=('Arial', 10, 'bold'), padding=(10, 8, 10, 4)).pack(anchor='w')

        chk_frame = ttk.Frame(sel_win, padding=(14, 0, 14, 4))
        chk_frame.pack(fill='x')

        vars_scalar = {}
        for col, lbl, ref, ylbl in available_scalar:
            v = tk.BooleanVar(value=True)
            vars_scalar[col] = v
            ttk.Checkbutton(chk_frame, text=lbl, variable=v).pack(anchor='w')

        paw_vars = {}
        if paw_tc_available:
            ttk.Separator(chk_frame, orient='horizontal').pack(fill='x', pady=4)
            for col, role in paw_tc_available:
                v = tk.BooleanVar(value=True)
                paw_vars[col] = v
                ttk.Checkbutton(chk_frame, text=f'Contact % {role}', variable=v).pack(anchor='w')

        btn_bar = ttk.Frame(sel_win, padding=(14, 6))
        btn_bar.pack(fill='x')

        def _select_all():
            for v in vars_scalar.values():
                v.set(True)
            for v in paw_vars.values():
                v.set(True)

        def _deselect_all():
            for v in vars_scalar.values():
                v.set(False)
            for v in paw_vars.values():
                v.set(False)

        ttk.Button(btn_bar, text="Select All",   command=_select_all).pack(side='left', padx=2)
        ttk.Button(btn_bar, text="Deselect All", command=_deselect_all).pack(side='left', padx=2)

        def _on_ok():
            chosen_scalar = [(col, lbl, ref, ylbl)
                             for col, lbl, ref, ylbl in available_scalar
                             if vars_scalar[col].get()]
            chosen_paw_cols = [col for col, v in paw_vars.items() if v.get()]
            sel_win.destroy()
            if not chosen_scalar and not chosen_paw_cols:
                return
            # ── Settings dialog ───────────────────────────────────────────
            treats = []
            if 'treatment' in bdf.columns:
                treats = [str(t) for t in bdf['treatment'].dropna().unique()
                          if str(t).strip()]
            if not treats:
                treats = ['All sessions']
            mtm = bdf['bin_end_s'].max() / 60.0 if 'bin_end_s' in bdf.columns else None
            cfg = self._build_graph_settings_dlg(self, treats, max_time_min=mtm)
            if cfg is None:
                return
            self._last_graph_cfg = cfg
            self._show_bin_graphs(chosen_scalar, chosen_paw_cols, cfg)

        ttk.Button(btn_bar, text="OK",     command=_on_ok).pack(side='right', padx=4)
        ttk.Button(btn_bar, text="Cancel", command=sel_win.destroy).pack(side='right', padx=2)
        sel_win.wait_window()

    def _show_bin_graphs(self, chosen_scalar, chosen_paw_cols, graph_cfg):
        bdf = self._bins_df

        win = tk.Toplevel(self)
        win.title("Gait & Limb Use — Bin Graphs")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        gw = int(sw * 0.75)
        gh = int(sh * 0.82)
        gx = (sw - gw) // 2
        gy = (sh - gh) // 2
        win.geometry(f"{gw}x{gh}+{gx}+{gy}")

        desc_lbl = ttk.Label(win, text='', wraplength=940, foreground='#444',
                             font=('Arial', 9, 'italic'), padding=(6, 2))
        desc_lbl.pack(fill='x', padx=6)

        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True, padx=6, pady=6)

        _tab_descs = {}

        _DESCS = {
            'WBI_hind':            'HL / (HL+HR) × 100 across time bins.  Reference 50 = symmetric.',
            'SI_hind':             'SI hind across time bins.  Reference 0 = symmetric.',
            'SBI_hind':            '2×|HL−HR|/(HL+HR)×100 across bins.  0 = perfect symmetry.',
            'WBI_fore':            'FL / (FL+FR) × 100 across time bins.  Reference 50 = symmetric.',
            'SI_fore':             'SI fore across time bins.  Reference 0.',
            'SBI_fore':            '2×|FL−FR|/(FL+FR)×100 across bins.',
            'contact_pct_HL':      'HL contact % across time bins.',
            'contact_pct_HR':      'HR contact % across time bins.',
            'contact_pct_FL':      'FL contact % across time bins.',
            'contact_pct_FR':      'FR contact % across time bins.',
            'brightness_HL':       'Mean HL ROI brightness during contact frames, across bins.',
            'brightness_HR':       'Mean HR ROI brightness during contact frames, across bins.',
            'brightness_FL':       'Mean FL ROI brightness during contact frames, across bins.',
            'brightness_FR':       'Mean FR ROI brightness during contact frames, across bins.',
            'brightness_ratio_HL_HR': 'HL/HR brightness ratio across bins.  Reference 1 = equal.',
            'hind_fore_ratio':     'Mean hind contact% ÷ mean fore contact% across bins.',
        }

        for col, lbl, ref, ylbl in chosen_scalar:
            self._add_timecourse_tab(nb, bdf, col, lbl,
                                     reference=ref, y_label=ylbl,
                                     graph_cfg=graph_cfg)
            _tab_descs[lbl] = _DESCS.get(col, lbl)

        for col in chosen_paw_cols:
            role = col.replace('contact_pct_', '')
            tab_name = f'Contact % {role}'
            self._add_timecourse_tab(nb, bdf, col, tab_name,
                                     reference=None,
                                     y_label=f'Contact % — {role}',
                                     graph_cfg=graph_cfg)
            _tab_descs[tab_name] = f'{role} contact % across time bins (mean ± error).'

        def _on_tab_change(event):
            try:
                tab_text = nb.tab(nb.select(), 'text')
                desc_lbl.config(text=_tab_descs.get(tab_text, ''))
            except Exception:
                pass
        nb.bind('<<NotebookTabChanged>>', _on_tab_change)
        if nb.tabs():
            nb.event_generate('<<NotebookTabChanged>>')

    # ═══════════════════════════════════════════════════════════════════════
    # Live recomputation of Filtered Contour tab
    # ═══════════════════════════════════════════════════════════════════════

    def _recompute_pawlike_metrics(self):
        """Recompute pawlike columns in _summary_df and _bins_df using current thresholds."""
        if self._summary_df is None:
            return
        PAWLIKE_SOL = self._pawlike_thresholds.get('solidity', 1.00)
        PAWLIKE_AR = self._pawlike_thresholds.get('aspect_ratio', 1.6)
        PAWLIKE_CIRC = self._pawlike_thresholds.get('circularity', 0.10)

        for idx, row in self._summary_df.iterrows():
            sess_name = row.get('session')
            if sess_name not in self._session_intermediates:
                continue
            inter = self._session_intermediates[sess_name]
            pcd = inter.get('paw_contour_data', {})
            if not pcd:
                continue
            fps = inter.get('fps', 30)
            n_frames = inter.get('n_frames', 0)
            contact_masks = inter.get('contact_masks', {})

            for role in list(pcd.keys()):
                areas = pcd[role]['areas']
                spreads = pcd[role]['spreads']
                intensities = pcd[role]['intensities']
                cm = contact_masks.get(role)
                if cm is not None:
                    mask_arr = cm.values.astype(bool) if hasattr(cm, 'values') else np.asarray(cm, dtype=bool)
                else:
                    mask_arr = np.ones(n_frames, dtype=bool)

                widths = pcd[role].get('widths')
                solidities = pcd[role].get('solidities')
                aspect_ratios = pcd[role].get('aspect_ratios')
                circularities = pcd[role].get('circularities')

                valid = mask_arr[:len(areas)] & (areas[:len(mask_arr)] > 0)
                if solidities is not None:
                    sol_arr = solidities[:len(mask_arr)]
                    valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                    if aspect_ratios is not None:
                        valid_paw = valid_paw & (aspect_ratios[:len(mask_arr)] <= PAWLIKE_AR)
                    if circularities is not None:
                        valid_paw = valid_paw & (circularities[:len(mask_arr)] >= PAWLIKE_CIRC)

                    _pca = areas[:len(mask_arr)][valid_paw]
                    self._summary_df.at[idx, f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                    _pcs = spreads[:len(mask_arr)][valid_paw]
                    self._summary_df.at[idx, f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                    _pci = intensities[:len(mask_arr)][valid_paw]
                    self._summary_df.at[idx, f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                    if widths is not None:
                        self._summary_df.at[idx, f'pawlike_width_{role}'] = round(float(np.nanmean(widths[:len(mask_arr)][valid_paw])), 2) if valid_paw.any() else float('nan')
                        self._summary_df.at[idx, f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                        self._summary_df.at[idx, f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(aspect_ratios[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')
                        self._summary_df.at[idx, f'pawlike_circularity_{role}'] = round(float(np.nanmean(circularities[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')

            # Recompute ratios
            df = self._summary_df
            if f'pawlike_area_HL' in df.columns and f'pawlike_area_HR' in df.columns:
                aHL, aHR = df.at[idx, 'pawlike_area_HL'], df.at[idx, 'pawlike_area_HR']
                if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                    df.at[idx, 'pawlike_area_ratio_hind'] = round(aHL / aHR, 4)
                else:
                    df.at[idx, 'pawlike_area_ratio_hind'] = float('nan')
            if f'pawlike_intensity_HL' in df.columns and f'pawlike_intensity_HR' in df.columns:
                iHL, iHR = df.at[idx, 'pawlike_intensity_HL'], df.at[idx, 'pawlike_intensity_HR']
                if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                    df.at[idx, 'pawlike_intensity_ratio_hind'] = round(iHL / iHR, 4)
                else:
                    df.at[idx, 'pawlike_intensity_ratio_hind'] = float('nan')

        # Recompute bins
        if self._bins_df is not None and not self._bins_df.empty:
            for bidx, brow in self._bins_df.iterrows():
                sess_name = brow.get('session')
                if sess_name not in self._session_intermediates:
                    continue
                inter = self._session_intermediates[sess_name]
                pcd = inter.get('paw_contour_data', {})
                if not pcd:
                    continue
                fps = inter.get('fps', 30)
                n_frames = inter.get('n_frames', 0)
                contact_masks = inter.get('contact_masks', {})
                bin_start_s = brow.get('bin_start_s', 0)
                bin_end_s = brow.get('bin_end_s', 0)
                start_f = int(round(bin_start_s * fps))
                end_f = int(round(bin_end_s * fps))
                frame_slice = slice(start_f, end_f)

                for role in list(pcd.keys()):
                    areas_full = pcd[role]['areas']
                    spreads_full = pcd[role]['spreads']
                    ints_full = pcd[role]['intensities']
                    cm = contact_masks.get(role)
                    if cm is not None:
                        mask_full = cm.values.astype(bool) if hasattr(cm, 'values') else np.asarray(cm, dtype=bool)
                    else:
                        mask_full = np.ones(n_frames, dtype=bool)
                    mask_arr = mask_full[frame_slice]
                    areas_sl = areas_full[frame_slice]
                    spreads_sl = spreads_full[frame_slice]
                    ints_sl = ints_full[frame_slice]

                    widths_full = pcd[role].get('widths')
                    solidities_full = pcd[role].get('solidities')
                    ar_full = pcd[role].get('aspect_ratios')
                    circ_full = pcd[role].get('circularities')
                    widths_sl = widths_full[frame_slice] if widths_full is not None else None
                    solidities_sl = solidities_full[frame_slice] if solidities_full is not None else None
                    ar_sl = ar_full[frame_slice] if ar_full is not None else None
                    circ_sl = circ_full[frame_slice] if circ_full is not None else None

                    valid = mask_arr[:len(areas_sl)] & (areas_sl[:len(mask_arr)] > 0)
                    if solidities_sl is not None:
                        sol_arr = solidities_sl[:len(mask_arr)]
                        valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                        if ar_sl is not None:
                            valid_paw = valid_paw & (ar_sl[:len(mask_arr)] <= PAWLIKE_AR)
                        if circ_sl is not None:
                            valid_paw = valid_paw & (circ_sl[:len(mask_arr)] >= PAWLIKE_CIRC)

                        _pca = areas_sl[:len(mask_arr)][valid_paw]
                        self._bins_df.at[bidx, f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                        _pcs = spreads_sl[:len(mask_arr)][valid_paw]
                        self._bins_df.at[bidx, f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                        _pci = ints_sl[:len(mask_arr)][valid_paw]
                        self._bins_df.at[bidx, f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                        if widths_sl is not None:
                            self._bins_df.at[bidx, f'pawlike_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_arr)][valid_paw])), 2) if valid_paw.any() else float('nan')
                            self._bins_df.at[bidx, f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                            self._bins_df.at[bidx, f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')
                            self._bins_df.at[bidx, f'pawlike_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_arr)][valid_paw])), 4) if valid_paw.any() else float('nan')

                # Recompute bin ratios
                bdf = self._bins_df
                if 'pawlike_area_HL' in bdf.columns and 'pawlike_area_HR' in bdf.columns:
                    aHL, aHR = bdf.at[bidx, 'pawlike_area_HL'], bdf.at[bidx, 'pawlike_area_HR']
                    if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                        bdf.at[bidx, 'pawlike_area_ratio_hind'] = round(aHL / aHR, 4)
                    else:
                        bdf.at[bidx, 'pawlike_area_ratio_hind'] = float('nan')
                if 'pawlike_intensity_HL' in bdf.columns and 'pawlike_intensity_HR' in bdf.columns:
                    iHL, iHR = bdf.at[bidx, 'pawlike_intensity_HL'], bdf.at[bidx, 'pawlike_intensity_HR']
                    if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                        bdf.at[bidx, 'pawlike_intensity_ratio_hind'] = round(iHL / iHR, 4)
                    else:
                        bdf.at[bidx, 'pawlike_intensity_ratio_hind'] = float('nan')

    def _rebuild_filtered_contour_tab(self):
        """Recompute pawlike metrics and rebuild the Filtered Contour tab."""
        self._recompute_pawlike_metrics()

        nb = getattr(self, '_contour_pawlike_nb', None)
        build_fn = getattr(self, '_contour_build_fn', None)
        metrics = getattr(self, '_contour_pawlike_metrics', None)
        if nb is None or build_fn is None or metrics is None:
            return

        # Close any matplotlib figures in existing tab contents
        try:
            import matplotlib.pyplot as plt
            for tab_id in nb.tabs():
                w = nb.nametowidget(tab_id)
                for child in w.winfo_children():
                    child.destroy()
                w.destroy()
            # Remove all tabs
            for tab_id in list(nb.tabs()):
                nb.forget(tab_id)
        except Exception:
            pass

        # Rebuild with updated data
        df = self._summary_df
        bdf = self._bins_df if self._bins_df is not None and not self._bins_df.empty else None
        build_fn(nb, metrics, False, df, bdf,
                 'pawlike_area_ratio_hind', 'pawlike_intensity_ratio_hind',
                 filter_paw=True)

    def _open_graphs(self):
        if not _PLOT_OK:
            messagebox.showerror("Missing deps",
                                 "matplotlib and scipy are required for graphs.",
                                 parent=self)
            return
        if self._summary_df is None or self._summary_df.empty:
            messagebox.showinfo("No data", "Run analysis first.", parent=self)
            return

        df = self._summary_df

        # ── Settings dialog ───────────────────────────────────────────────
        treatments = []
        if 'treatment' in df.columns:
            treatments = [str(t) for t in df['treatment'].dropna().unique()
                          if str(t).strip()]
        if not treatments:
            treatments = ['All sessions']
        max_time_min = None
        if self._bins_df is not None and not self._bins_df.empty:
            if 'bin_end_s' in self._bins_df.columns:
                max_time_min = self._bins_df['bin_end_s'].max() / 60.0
        graph_cfg = self._build_graph_settings_dlg(self, treatments, max_time_min)
        if graph_cfg is None:
            return
        self._last_graph_cfg = graph_cfg
        self._enable_stats_var.set(graph_cfg['show_stats'])
        self._log("Building graphs…")

        win = tk.Toplevel(self)
        win.title("Gait & Limb Use Graphs")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        gw = int(sw * 0.75)
        gh = int(sh * 0.82)
        gx = (sw - gw) // 2
        gy = (sh - gh) // 2
        win.geometry(f"{gw}x{gh}+{gx}+{gy}")

        desc_lbl = ttk.Label(win, text='', wraplength=940, foreground='#444',
                             font=('Arial', 9, 'italic'), padding=(6, 2))
        desc_lbl.pack(fill='x', padx=6)

        outer_nb = ttk.Notebook(win)
        outer_nb.pack(fill='both', expand=True, padx=6, pady=6)

        def _close_figures_recursive(widget):
            fig_obj = getattr(widget, 'figure', None)
            if fig_obj is not None:
                try:
                    plt.close(fig_obj)
                except Exception:
                    pass
            for child in getattr(widget, 'winfo_children', lambda: [])():
                _close_figures_recursive(child)

        def _on_graph_win_close():
            for widget in win.winfo_children():
                _close_figures_recursive(widget)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_graph_win_close)

        _tab_descs = {}

        # ── Graph set selection from settings ────────────────────────────
        _sets = graph_cfg.get('graph_sets', {})
        show_wb       = _sets.get('weight_bearing', True)
        show_gait     = _sets.get('gait', True)
        show_movement = _sets.get('movement', True)
        show_contour  = _sets.get('paw_contour', True)
        show_stats    = _sets.get('statistics', True)

        def _make_group(name):
            """Create a top-level group tab containing a notebook."""
            f = ttk.Frame(outer_nb)
            outer_nb.add(f, text=name)
            group_nb = ttk.Notebook(f)
            group_nb.pack(fill='both', expand=True)
            return group_nb

        def _make_category(name, parent=None):
            """Create a category tab (inner notebook) under the given parent."""
            target = parent if parent is not None else outer_nb
            f = ttk.Frame(target)
            target.add(f, text=name)
            inner = ttk.Notebook(f)
            inner.pack(fill='both', expand=True)
            return inner

        bdf = (self._bins_df
               if self._bins_df is not None and not self._bins_df.empty
               else None)

        # Determine whether any fore-paw data exists
        has_fore = (
            ('WBI_fore' in df.columns and df['WBI_fore'].notna().any()) or
            ('SI_fore'  in df.columns and df['SI_fore'].notna().any())  or
            ('SBI_fore' in df.columns and df['SBI_fore'].notna().any())
        )

        # ── Lazy group helper ──────────────────────────────────────────────
        _deferred_builds = {}  # outer_nb tab_id -> (build_fn, built_flag)

        def _register_deferred(group_frame, build_fn):
            """Register a build function to run when this group tab is first selected."""
            _deferred_builds[str(group_frame)] = [build_fn, False]

        def _check_deferred(evt=None):
            """Build deferred group content on first view."""
            try:
                sel = outer_nb.select()
                if not sel:
                    return
                key = sel
            except Exception:
                return
            entry = _deferred_builds.get(key)
            if entry and not entry[1]:
                entry[1] = True
                tab_name = outer_nb.tab(sel, 'text')
                self._log(f"  Building {tab_name} graphs…")
                try:
                    win.config(cursor='watch')
                    win.update_idletasks()
                except Exception:
                    pass
                entry[0]()
                try:
                    win.config(cursor='')
                    win.update_idletasks()
                except Exception:
                    pass

        outer_nb.bind('<<NotebookTabChanged>>', _check_deferred, add='+')

        # ── Limb Use group ────────────────────────────────────────────────
        wb_group   = _make_group("Limb Use") if show_wb else None
        hind_sel    = (self._make_metric_selector(wb_group, "Hind Paw",
                       desc_lbl, _tab_descs) if show_wb else None)
        fore_sel    = (self._make_metric_selector(wb_group, "Fore Paw",
                       desc_lbl, _tab_descs)
                      if show_wb and has_fore else None)
        contact_sel = (self._make_metric_selector(wb_group, "Contact %",
                       desc_lbl, _tab_descs) if show_wb else None)
        bright_sel  = (self._make_metric_selector(wb_group, "Brightness",
                       desc_lbl, _tab_descs) if show_wb else None)

        _tab_descs['Limb Use'] = (
            'Limb use indices, paw contact percentages, and brightness ratios.')

        # ── Gait group ───────────────────────────────────────────────────
        has_gait_timing = any(f'stance_dur_{r}' in df.columns and df[f'stance_dur_{r}'].notna().any()
                              for r in self.ROLES)
        has_gait_spatial = any(f'stride_len_{r}' in df.columns and df[f'stride_len_{r}'].notna().any()
                               for r in self.ROLES)
        has_gait_sym = ('stance_SI_hind' in df.columns and df['stance_SI_hind'].notna().any()) or \
                       ('stride_len_SI_hind' in df.columns and df['stride_len_SI_hind'].notna().any())
        has_any_gait = show_gait and (has_gait_timing or has_gait_spatial or has_gait_sym)

        gait_group      = _make_group("Gait") if has_any_gait else None
        gait_timing_sel = (self._make_metric_selector(
                               gait_group, "Timing", desc_lbl, _tab_descs)
                           if has_any_gait and has_gait_timing else None)
        gait_spatial_sel = (self._make_metric_selector(
                                gait_group, "Spatial", desc_lbl, _tab_descs)
                            if has_any_gait and has_gait_spatial else None)
        gait_sym_sel    = (self._make_metric_selector(
                               gait_group, "Symmetry", desc_lbl, _tab_descs)
                           if has_any_gait and has_gait_sym else None)

        _tab_descs['Gait'] = (
            'Gait timing, spatial parameters, and symmetry indices.')

        # ── Movement group ───────────────────────────────────────────────
        has_loco = ('total_distance' in df.columns and df['total_distance'].notna().any())
        has_coord = any(c in df.columns and df[c].notna().any()
                        for c in ['regularity_index', 'print_position_L',
                                  'support_0paw_pct', 'phase_FL_FR'])
        has_any_movement = show_movement and (has_loco or has_coord)

        movement_group = _make_group("Movement") if has_any_movement else None
        loco_sel  = (self._make_metric_selector(
                         movement_group, "Locomotion", desc_lbl, _tab_descs)
                     if has_any_movement and has_loco else None)
        coord_sel = (self._make_metric_selector(
                         movement_group, "Coordination", desc_lbl, _tab_descs)
                     if has_any_movement and has_coord else None)

        _tab_descs['Movement'] = (
            'Locomotion distance and speed, coordination and support patterns.')

        # ── Hind Paw ──────────────────────────────────────────────────────
        if show_wb:
            self._log("  Registering Limb Use metrics…")
        if hind_sel is not None:
            _reg = self._register_metric
            if 'WBI_hind' in df.columns:
                _reg(hind_sel, 'WBI Hind', 'bar', df, 'WBI_hind',
                     reference=50.0, y_label='Weight Bearing Index — hind (%)',
                     graph_cfg=graph_cfg,
                     description='HL / (HL+HR) \u00d7 100.  Reference 50 = symmetric.  >50 = more stance on left hind.')
            if 'SI_hind' in df.columns:
                _reg(hind_sel, 'SI Hind (Box)', 'box', df, 'SI_hind',
                     reference=0.0, y_label='Symmetry Index — hind (%)',
                     graph_cfg=graph_cfg,
                     description='(HL\u2212HR) / (HL+HR) \u00d7 100.  Reference 0 = symmetric.  Positive = left bias.')
                _reg(hind_sel, 'SI Hind (Violin)', 'violin', df, 'SI_hind',
                     reference=0.0, y_label='Symmetry Index — hind (%)',
                     graph_cfg=graph_cfg,
                     description='(HL\u2212HR) / (HL+HR) \u00d7 100.  Reference 0 = symmetric.  Positive = left bias.')
            if 'SBI_hind' in df.columns and df['SBI_hind'].notna().any():
                _reg(hind_sel, 'SBI Hind', 'bar', df, 'SBI_hind',
                     reference=0.0, y_label='Symmetry Balance Index — hind (%)',
                     graph_cfg=graph_cfg,
                     description='2 \u00d7 |HL\u2212HR| / (HL+HR) \u00d7 100.  Always \u2265 0.  0 = perfect symmetry.')
            if bdf is not None:
                for col, lbl, ref, ylbl, desc in [
                    ('SI_hind', 'SI Hind \u2014 TC', None,
                     'Symmetry Index \u2014 hind (%)',
                     'SI hind across time bins (mean \u00b1 SEM).  Dashed line = 0 (symmetric).'),
                    ('WBI_hind', 'WBI Hind \u2014 TC', 50.0,
                     'Weight Bearing Index \u2014 hind (%)',
                     'WBI hind across time bins. Reference 50 = symmetric.'),
                    ('SBI_hind', 'SBI Hind \u2014 TC', 0.0,
                     'Symmetry Balance Index \u2014 hind (%)',
                     'SBI hind across time bins. 0 = perfect symmetry.'),
                ]:
                    if col in bdf.columns and bdf[col].notna().any():
                        _reg(hind_sel, lbl, 'timecourse', bdf, col,
                             reference=ref, y_label=ylbl, graph_cfg=graph_cfg,
                             description=desc)
            if hind_sel['registry']:
                win.after_idle(lambda _s=hind_sel: _s['show'](0))

        # ── Fore Paw ──────────────────────────────────────────────────────
        if fore_sel is not None:
            _reg = self._register_metric
            if 'WBI_fore' in df.columns and df['WBI_fore'].notna().any():
                _reg(fore_sel, 'WBI Fore', 'bar', df, 'WBI_fore',
                     reference=50.0, y_label='Weight Bearing Index \u2014 fore (%)',
                     graph_cfg=graph_cfg,
                     description='FL / (FL+FR) \u00d7 100.  Reference 50 = symmetric.')
            if 'SI_fore' in df.columns and df['SI_fore'].notna().any():
                _reg(fore_sel, 'SI Fore (Box)', 'box', df, 'SI_fore',
                     reference=0.0, y_label='Symmetry Index \u2014 fore (%)',
                     graph_cfg=graph_cfg,
                     description='(FL\u2212FR) / (FL+FR) \u00d7 100.  Reference 0.')
                _reg(fore_sel, 'SI Fore (Violin)', 'violin', df, 'SI_fore',
                     reference=0.0, y_label='Symmetry Index \u2014 fore (%)',
                     graph_cfg=graph_cfg)
            if 'SBI_fore' in df.columns and df['SBI_fore'].notna().any():
                _reg(fore_sel, 'SBI Fore', 'bar', df, 'SBI_fore',
                     reference=0.0, y_label='Symmetry Balance Index \u2014 fore (%)',
                     graph_cfg=graph_cfg,
                     description='2 \u00d7 |FL\u2212FR| / (FL+FR) \u00d7 100.  Always \u2265 0.')
            if bdf is not None:
                for col, lbl, ref, ylbl, desc in [
                    ('WBI_fore', 'WBI Fore \u2014 TC', 50.0,
                     'Weight Bearing Index \u2014 fore (%)',
                     'WBI fore across time bins. Reference 50 = symmetric.'),
                    ('SI_fore', 'SI Fore \u2014 TC', 0.0,
                     'Symmetry Index \u2014 fore (%)',
                     'SI fore across time bins. Reference 0 = symmetric.'),
                    ('SBI_fore', 'SBI Fore \u2014 TC', 0.0,
                     'Symmetry Balance Index \u2014 fore (%)',
                     'SBI fore across time bins. 0 = perfect symmetry.'),
                ]:
                    if col in bdf.columns and bdf[col].notna().any():
                        _reg(fore_sel, lbl, 'timecourse', bdf, col,
                             reference=ref, y_label=ylbl, graph_cfg=graph_cfg,
                             description=desc)
            if fore_sel['registry']:
                _fore_shown = [False]
                def _show_fore_first(evt=None):
                    if not _fore_shown[0]:
                        _fore_shown[0] = True
                        fore_sel['show'](0)
                fore_sel['frame'].bind('<Map>', _show_fore_first)

        # ── Contact % ─────────────────────────────────────────────────────
        if contact_sel is not None:
            _reg = self._register_metric
            # Grouped bar chart as custom entry
            _reg(contact_sel, 'Contact % (All Paws)', 'custom', None, None,
                 graph_cfg=graph_cfg,
                 description='Percentage of frames each paw is in contact with the surface.',
                 create_fn=lambda f, _df=df, _gc=graph_cfg:
                     self._add_paw_contact_bar_tab(f, _df, graph_cfg=_gc,
                                                   as_frame=True))
            for role in self.ROLES:
                col = f'contact_pct_{role}'
                if (bdf is not None and col in bdf.columns
                        and 'bin_start_s' in bdf.columns):
                    _reg(contact_sel, f'Contact % {role}', 'timecourse',
                         bdf, col, reference=None,
                         y_label=f'Contact % \u2014 {role}',
                         graph_cfg=graph_cfg,
                         description=f'{role} paw contact % across time bins.')
            if (bdf is not None and 'hind_fore_ratio' in bdf.columns
                    and bdf['hind_fore_ratio'].notna().any()):
                _reg(contact_sel, 'Hind/Fore Ratio \u2014 TC', 'timecourse',
                     bdf, 'hind_fore_ratio', reference=None,
                     y_label='Mean hind / mean fore contact %',
                     graph_cfg=graph_cfg,
                     description='Ratio of mean hind to mean fore contact % across time bins.')
            if contact_sel['registry']:
                _contact_shown = [False]
                def _show_contact_first(evt=None):
                    if not _contact_shown[0]:
                        _contact_shown[0] = True
                        contact_sel['show'](0)
                contact_sel['frame'].bind('<Map>', _show_contact_first)

        # ── Brightness ────────────────────────────────────────────────────
        if bright_sel is not None:
            _reg = self._register_metric
            if ('brightness_ratio_HL_HR' in df.columns
                    and df['brightness_ratio_HL_HR'].notna().any()):
                _reg(bright_sel, 'Brightness Ratio', 'bar', df,
                     'brightness_ratio_HL_HR', reference=1.0,
                     y_label='HL / HR brightness (contact frames)',
                     graph_cfg=graph_cfg,
                     description='HL/HR brightness ratio. Reference 1.0 = equal.')
            if bdf is not None:
                for col, lbl, ref, ylbl, desc in [
                    ('brightness_HL', 'Brightness HL \u2014 TC', None,
                     'Mean brightness \u2014 HL',
                     'Mean brightness in HL ROI during contact frames.'),
                    ('brightness_HR', 'Brightness HR \u2014 TC', None,
                     'Mean brightness \u2014 HR',
                     'Mean brightness in HR ROI during contact frames.'),
                    ('brightness_FL', 'Brightness FL \u2014 TC', None,
                     'Mean brightness \u2014 FL',
                     'Mean brightness in FL ROI during contact frames.'),
                    ('brightness_FR', 'Brightness FR \u2014 TC', None,
                     'Mean brightness \u2014 FR',
                     'Mean brightness in FR ROI during contact frames.'),
                    ('brightness_ratio_HL_HR', 'Brightness Ratio \u2014 TC', 1.0,
                     'HL / HR brightness',
                     'HL/HR brightness ratio across time bins. Reference 1.0 = equal.'),
                ]:
                    if col in bdf.columns and bdf[col].notna().any():
                        _reg(bright_sel, lbl, 'timecourse', bdf, col,
                             reference=ref, y_label=ylbl, graph_cfg=graph_cfg,
                             description=desc)
            if bright_sel['registry']:
                _bright_shown = [False]
                def _show_bright_first(evt=None):
                    if not _bright_shown[0]:
                        _bright_shown[0] = True
                        bright_sel['show'](0)
                bright_sel['frame'].bind('<Map>', _show_bright_first)

        # ── Gait graphs (deferred) ────────────────────────────────────
        def _build_gait():
            self._log("  Registering Gait metrics…")
            _reg = self._register_metric
            if gait_timing_sel is not None:
                for role in self.ROLES:
                    for col, lbl, ref, ylbl, desc in [
                        (f'stance_dur_{role}', f'Stance Dur {role}', None,
                         f'Stance duration (s) \u2014 {role}',
                         f'Mean stance duration (s) for {role}.'),
                        (f'swing_dur_{role}', f'Swing Dur {role}', None,
                         f'Swing duration (s) \u2014 {role}',
                         f'Mean swing duration (s) for {role}.'),
                        (f'duty_cycle_{role}', f'Duty Cycle {role}', None,
                         f'Duty cycle (%) \u2014 {role}',
                         f'Duty cycle (%) for {role}. >50% = more stance than swing.'),
                    ]:
                        if col in df.columns and df[col].notna().any():
                            _reg(gait_timing_sel, lbl, 'bar', df, col,
                                 reference=ref, y_label=ylbl,
                                 graph_cfg=graph_cfg, description=desc)
                    for col, lbl, ylbl, desc in [
                        (f'swing_speed_{role}', f'Swing Speed {role}',
                         f'Swing speed (px/s) \u2014 {role}',
                         f'Mean swing speed for {role}. Injured animals swing slower on affected side.'),
                    ]:
                        if col in df.columns and df[col].notna().any():
                            _reg(gait_timing_sel, lbl, 'bar', df, col,
                                 reference=None, y_label=ylbl,
                                 graph_cfg=graph_cfg, description=desc)
                if bdf is not None:
                    for role in self.ROLES:
                        for col, lbl, ylbl, desc in [
                            (f'stance_dur_{role}', f'Stance {role} \u2014 TC',
                             f'Stance dur (s) \u2014 {role}',
                             f'Stance duration for {role} across time bins.'),
                            (f'duty_cycle_{role}', f'Duty {role} \u2014 TC',
                             f'Duty cycle (%) \u2014 {role}',
                             f'Duty cycle for {role} across time bins.'),
                            (f'cadence_{role}', f'Cadence {role} \u2014 TC',
                             f'Cadence (strides/min) \u2014 {role}',
                             f'Stride cadence for {role} across time bins.'),
                            (f'swing_speed_{role}', f'Swing Spd {role} \u2014 TC',
                             f'Swing speed (px/s) \u2014 {role}',
                             f'Swing speed for {role} across time bins.'),
                        ]:
                            if col in bdf.columns and bdf[col].notna().any():
                                _reg(gait_timing_sel, lbl, 'timecourse',
                                     bdf, col, reference=None, y_label=ylbl,
                                     graph_cfg=graph_cfg, description=desc)
                if gait_timing_sel['registry']:
                    gait_timing_sel['show'](0)

            if gait_spatial_sel is not None:
                for role in self.ROLES:
                    col = f'stride_len_{role}'
                    if col in df.columns and df[col].notna().any():
                        _reg(gait_spatial_sel, f'Stride Len {role}', 'bar',
                             df, col, reference=None,
                             y_label=f'Stride length (px) \u2014 {role}',
                             graph_cfg=graph_cfg,
                             description=f'Mean stride length (px) for {role}.')
                for pair in ['hind', 'fore']:
                    for col, lbl, ylbl in [
                        (f'step_len_{pair}', f'Step Len {pair}',
                         f'Step length (px) \u2014 {pair}'),
                        (f'step_width_{pair}', f'Step Width {pair}',
                         f'Step width (px) \u2014 {pair}'),
                    ]:
                        if col in df.columns and df[col].notna().any():
                            _reg(gait_spatial_sel, lbl, 'bar', df, col,
                                 reference=None, y_label=ylbl,
                                 graph_cfg=graph_cfg, description=ylbl)
                if bdf is not None:
                    for role in self.ROLES:
                        col = f'stride_len_{role}'
                        if col in bdf.columns and bdf[col].notna().any():
                            _reg(gait_spatial_sel,
                                 f'Stride Len {role} \u2014 TC', 'timecourse',
                                 bdf, col, reference=None,
                                 y_label=f'Stride length (px) \u2014 {role}',
                                 graph_cfg=graph_cfg,
                                 description=f'Stride length for {role} across time bins.')
                if gait_spatial_sel['registry']:
                    gait_spatial_sel['show'](0)

            if gait_sym_sel is not None:
                for col, lbl, ref, ylbl in [
                    ('stance_SI_hind', 'Stance SI Hind', 0.0,
                     'Stance Symmetry Index (%)'),
                    ('stride_len_SI_hind', 'Stride Len SI Hind', 0.0,
                     'Stride Length Symmetry Index (%)'),
                ]:
                    if col in df.columns and df[col].notna().any():
                        _reg(gait_sym_sel, lbl, 'bar', df, col,
                             reference=ref, y_label=ylbl, graph_cfg=graph_cfg,
                             description=f'{ylbl}. Reference 0 = symmetric.')
                        _reg(gait_sym_sel, f'{lbl} (Box)', 'box', df, col,
                             reference=ref, y_label=ylbl, graph_cfg=graph_cfg,
                             description=f'{ylbl} \u2014 box plot distribution.')
                for role in self.ROLES:
                    col = f'stride_cv_{role}'
                    if col in df.columns and df[col].notna().any():
                        _reg(gait_sym_sel, f'Stride CV {role}', 'bar',
                             df, col, reference=None,
                             y_label=f'Stride CV (%) \u2014 {role}',
                             graph_cfg=graph_cfg,
                             description=f'Stride-to-stride CV for {role}. Higher = less rhythmic.')
                for col, lbl, ylbl in [
                    ('phase_HL_HR', 'Phase HL-HR',
                     'HL-HR phase (0.5 = alternating)'),
                    ('phase_diagonal', 'Phase Diagonal',
                     'HR-FL diagonal phase'),
                ]:
                    if col in df.columns and df[col].notna().any():
                        _reg(gait_sym_sel, lbl, 'bar', df, col,
                             reference=0.5, y_label=ylbl,
                             graph_cfg=graph_cfg,
                             description=f'{ylbl}. Reference 0.5 = alternating.')
                if bdf is not None:
                    for col, lbl, ylbl, desc in [
                        ('stance_SI_hind', 'Stance SI \u2014 TC',
                         'Stance SI (%)',
                         'Stance Symmetry Index across time bins.'),
                        ('stride_len_SI_hind', 'Stride Len SI \u2014 TC',
                         'Stride Len SI (%)',
                         'Stride Length SI across time bins.'),
                    ]:
                        if col in bdf.columns and bdf[col].notna().any():
                            _reg(gait_sym_sel, lbl, 'timecourse', bdf, col,
                                 reference=0.0, y_label=ylbl,
                                 graph_cfg=graph_cfg, description=desc)
                if gait_sym_sel['registry']:
                    gait_sym_sel['show'](0)

        if has_any_gait:
            _register_deferred(gait_group.master, _build_gait)

        # ── Movement graphs (deferred) ─────────────────────────────────
        def _build_movement():
            self._log("  Registering Movement metrics…")
            _reg = self._register_metric
            if loco_sel is not None:
                for col, lbl, ylbl, desc in [
                    ('total_distance', 'Total Distance', 'Distance (px)',
                     'Total body displacement across the session.'),
                    ('loco_total_distance', 'Distance (Moving)',
                     'Distance (px) \u2014 locomotion only',
                     'Displacement during locomotion bouts only.'),
                    ('time_moving_s', 'Time Moving', 'Time moving (s)',
                     'Total time in locomotion (seconds).'),
                    ('time_moving_pct', 'Time Moving %',
                     'Time in locomotion (%)',
                     'Percentage of session spent in locomotion.'),
                    ('body_speed_mean', 'Body Speed Mean',
                     'Body speed (px/s)',
                     'Mean body centroid speed across all frames.'),
                    ('body_speed_loco', 'Body Speed (Loco)',
                     'Body speed (px/s) \u2014 locomotion',
                     'Mean body speed during locomotion bouts only.'),
                ]:
                    if col in df.columns and df[col].notna().any():
                        _reg(loco_sel, lbl, 'bar', df, col,
                             y_label=ylbl, graph_cfg=graph_cfg,
                             description=desc)
                if bdf is not None:
                    for col, lbl, ylbl, desc in [
                        ('total_distance', 'Distance \u2014 TC',
                         'Distance (px)',
                         'Total body displacement per time bin.'),
                        ('loco_total_distance', 'Distance (Moving) \u2014 TC',
                         'Distance (px) \u2014 locomotion',
                         'Displacement during locomotion per time bin.'),
                        ('time_moving_s', 'Time Moving \u2014 TC',
                         'Time moving (s)',
                         'Time in locomotion per time bin.'),
                        ('time_moving_pct', 'Time Moving % \u2014 TC',
                         'Time in locomotion (%)',
                         'Percentage of each time bin in locomotion.'),
                        ('body_speed_mean', 'Body Speed \u2014 TC',
                         'Body speed (px/s)',
                         'Mean body speed per time bin.'),
                        ('body_speed_loco', 'Body Speed Loco \u2014 TC',
                         'Body speed (px/s) \u2014 loco',
                         'Body speed during locomotion per time bin.'),
                    ]:
                        if col in bdf.columns and bdf[col].notna().any():
                            _reg(loco_sel, lbl, 'timecourse', bdf, col,
                                 y_label=ylbl, graph_cfg=graph_cfg,
                                 description=desc)
                if loco_sel['registry']:
                    loco_sel['show'](0)

            if coord_sel is not None:
                if ('regularity_index' in df.columns
                        and df['regularity_index'].notna().any()):
                    _reg(coord_sel, 'Regularity Index', 'bar', df,
                         'regularity_index', reference=100.0,
                         y_label='Regularity Index (%)',
                         graph_cfg=graph_cfg,
                         description='% of steps following normal 4-paw pattern. 100% = perfect.')
                for side in ['L', 'R']:
                    col = f'print_position_{side}'
                    if col in df.columns and df[col].notna().any():
                        _reg(coord_sel, f'Print Position {side}', 'bar',
                             df, col, reference=None,
                             y_label=f'Print position (px) \u2014 {side}',
                             graph_cfg=graph_cfg,
                             description=f'Hind-fore paw overlap distance ({side}). Smaller = better.')
                support_cols = [f'support_{n}paw_pct' for n in range(5)]
                if all(c in df.columns for c in support_cols[:3]):
                    _reg(coord_sel, 'Support Patterns', 'custom', None, None,
                         graph_cfg=graph_cfg,
                         description='Distribution of paw support during locomotion.',
                         create_fn=lambda f, _df=df, _gc=graph_cfg:
                             self._add_support_pattern_tab(f, _df,
                                                           graph_cfg=_gc,
                                                           as_frame=True))
                for col, lbl, ylbl in [
                    ('phase_HL_HR', 'Phase HL-HR', 'HL-HR phase'),
                    ('phase_diagonal', 'Phase Diagonal',
                     'HR-FL diagonal phase'),
                    ('phase_FL_FR', 'Phase FL-FR', 'FL-FR phase'),
                    ('phase_HL_FL', 'Phase HL-FL', 'HL-FL phase'),
                    ('phase_HR_FR', 'Phase HR-FR', 'HR-FR phase'),
                    ('phase_HL_FR', 'Phase HL-FR', 'HL-FR phase'),
                ]:
                    if col in df.columns and df[col].notna().any():
                        _reg(coord_sel, lbl, 'bar', df, col,
                             reference=0.5, y_label=ylbl,
                             graph_cfg=graph_cfg,
                             description=f'{ylbl}. 0.5 = alternating (normal).')
                if coord_sel['registry']:
                    coord_sel['show'](0)

        if has_any_movement:
            _register_deferred(movement_group.master, _build_movement)

        # ── Paw Contour graphs (deferred) ───────────────────────────────
        _contour_metrics = [
            ('paw_area',          'Area',         'Paw area (px\u00b2) \u2014 {}'),
            ('paw_spread',        'Spread',       'Paw spread (px) \u2014 {}'),
            ('contact_intensity', 'Intensity',    'Contact intensity \u2014 {}'),
            ('paw_width',         'Width',        'Paw width (px) \u2014 {}'),
            ('paw_solidity',      'Solidity',     'Paw solidity \u2014 {}'),
            ('paw_aspect_ratio',  'Aspect Ratio', 'Aspect ratio \u2014 {}'),
            ('paw_circularity',   'Circularity',  'Circularity \u2014 {}'),
        ]
        has_contour = any(
            f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
            for mk, _, _ in _contour_metrics for role in self.ROLES
        ) or ('paw_area_ratio_hind' in df.columns and df['paw_area_ratio_hind'].notna().any()) or ('contact_intensity_ratio_hind' in df.columns and df['contact_intensity_ratio_hind'].notna().any())
        _contour_stance_metrics = [
            ('paw_area_stance',          'Area',         'Paw area (px\u00b2) \u2014 {}'),
            ('paw_spread_stance',        'Spread',       'Paw spread (px) \u2014 {}'),
            ('contact_intensity_stance', 'Intensity',    'Contact intensity \u2014 {}'),
            ('paw_width_stance',         'Width',        'Paw width (px) \u2014 {}'),
            ('paw_solidity_stance',      'Solidity',     'Paw solidity \u2014 {}'),
            ('paw_aspect_ratio_stance',  'Aspect Ratio', 'Aspect ratio \u2014 {}'),
            ('paw_circularity_stance',   'Circularity',  'Circularity \u2014 {}'),
        ]
        has_stance_contour = any(
            f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
            for mk, _, _ in _contour_stance_metrics for role in self.ROLES
        )
        _contour_pawlike_metrics = [
            ('pawlike_area',          'Area',         'Paw area (px\u00b2) \u2014 {}'),
            ('pawlike_spread',        'Spread',       'Paw spread (px) \u2014 {}'),
            ('pawlike_intensity',     'Intensity',    'Contact intensity \u2014 {}'),
            ('pawlike_width',         'Width',        'Paw width (px) \u2014 {}'),
            ('pawlike_solidity',      'Solidity',     'Paw solidity \u2014 {}'),
            ('pawlike_aspect_ratio',  'Aspect Ratio', 'Aspect ratio \u2014 {}'),
            ('pawlike_circularity',   'Circularity',  'Circularity \u2014 {}'),
        ]
        has_pawlike_contour = any(
            f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
            for mk, _, _ in _contour_pawlike_metrics for role in self.ROLES
        )

        contour_nb = (_make_group("Paw Contour")
                      if show_contour and (has_contour or has_stance_contour or has_pawlike_contour)
                      else None)
        contour_all_nb = None
        contour_stance_nb = None
        _contour_paw_nbs = []

        def _build_contour():
            self._log("  Building Paw Contour graphs…")
            if contour_nb is not None:
                _paw_labels = {'HL': 'Hind Left', 'HR': 'Hind Right',
                               'FL': 'Fore Left', 'FR': 'Fore Right'}

                def _build_contour_paw_tabs(parent_nb, metrics_list, stance_suffix, df_src, bdf_src,
                                            ratio_key, intensity_ratio_key, filter_paw=False):
                    """Build per-paw metric selectors inside parent_nb."""
                    _reg = self._register_metric
                    for role in self.ROLES:
                        has_paw = any(
                            f'{mk}_{role}' in df_src.columns and df_src[f'{mk}_{role}'].notna().any()
                            for mk, _, _ in metrics_list
                        )
                        if not has_paw:
                            continue
                        paw_label = _paw_labels.get(role, role)
                        paw_sel = self._make_metric_selector(
                            parent_nb, paw_label, desc_lbl, _tab_descs)

                        # Shape (custom)
                        _reg(paw_sel, 'Shape', 'custom', None, None,
                             description='Mean paw contour shape.',
                             create_fn=lambda f, _r=role, _fp=filter_paw:
                                 self._add_contour_shape_tab(
                                     f, df_src, self._session_intermediates,
                                     _r, tab_name='Shape', graph_cfg=graph_cfg,
                                     filter_paw=_fp, as_frame=True))
                        # Prints
                        _reg(paw_sel, 'Print (Single)', 'custom', None, None,
                             description='Single representative paw print.',
                             create_fn=lambda f, _r=role, _fp=filter_paw:
                                 self._add_contour_print_tab(
                                     f, df_src, self._session_intermediates,
                                     _r, tab_name='Single', graph_cfg=graph_cfg,
                                     n_prints=1, filter_paw=_fp, as_frame=True))
                        _reg(paw_sel, 'Prints (Multi 5)', 'custom', None, None,
                             description='Five representative paw prints.',
                             create_fn=lambda f, _r=role, _fp=filter_paw:
                                 self._add_contour_print_tab(
                                     f, df_src, self._session_intermediates,
                                     _r, tab_name='Multi (5)', graph_cfg=graph_cfg,
                                     n_prints=5, filter_paw=_fp, as_frame=True))
                        # All contour metrics
                        for metric_key, tab_prefix, ylbl_template in metrics_list:
                            col = f'{metric_key}_{role}'
                            ylbl = ylbl_template.format(paw_label)
                            if col in df_src.columns and df_src[col].notna().any():
                                _reg(paw_sel, tab_prefix, 'bar', df_src, col,
                                     y_label=ylbl, graph_cfg=graph_cfg,
                                     description=ylbl)
                            if (bdf_src is not None and col in bdf_src.columns
                                    and bdf_src[col].notna().any()):
                                _reg(paw_sel, f'{tab_prefix} \u2014 TC',
                                     'timecourse', bdf_src, col,
                                     y_label=ylbl, graph_cfg=graph_cfg,
                                     description=f'{ylbl} across time bins.')
                        if paw_sel['registry']:
                            paw_sel['show'](0)

                    # Ratios
                    has_ratios = False
                    for rk in [ratio_key, intensity_ratio_key]:
                        if rk in df_src.columns and df_src[rk].notna().any():
                            has_ratios = True
                        if (bdf_src is not None and rk in bdf_src.columns
                                and bdf_src[rk].notna().any()):
                            has_ratios = True
                    if has_ratios:
                        ratio_sel = self._make_metric_selector(
                            parent_nb, 'Ratios', desc_lbl, _tab_descs)
                        ylbl_sfx = ' (stance)' if stance_suffix else ''
                        if ratio_key in df_src.columns and df_src[ratio_key].notna().any():
                            _reg(ratio_sel, 'Area Ratio Hind', 'bar',
                                 df_src, ratio_key, reference=1.0,
                                 y_label=f'Paw area ratio HL/HR{ylbl_sfx}',
                                 graph_cfg=graph_cfg,
                                 description='HL/HR paw area ratio. 1.0 = equal.')
                        if (intensity_ratio_key in df_src.columns
                                and df_src[intensity_ratio_key].notna().any()):
                            _reg(ratio_sel, 'Intensity Ratio Hind', 'bar',
                                 df_src, intensity_ratio_key, reference=1.0,
                                 y_label=f'Intensity ratio HL/HR{ylbl_sfx}',
                                 graph_cfg=graph_cfg,
                                 description='HL/HR intensity ratio. 1.0 = equal.')
                        if bdf_src is not None:
                            if (ratio_key in bdf_src.columns
                                    and bdf_src[ratio_key].notna().any()):
                                _reg(ratio_sel, 'Area Ratio \u2014 TC',
                                     'timecourse', bdf_src, ratio_key,
                                     reference=1.0,
                                     y_label=f'Paw area ratio HL/HR{ylbl_sfx}',
                                     graph_cfg=graph_cfg,
                                     description='HL/HR area ratio across time bins.')
                            if (intensity_ratio_key in bdf_src.columns
                                    and bdf_src[intensity_ratio_key].notna().any()):
                                _reg(ratio_sel, 'Intensity Ratio \u2014 TC',
                                     'timecourse', bdf_src,
                                     intensity_ratio_key, reference=1.0,
                                     y_label=f'Intensity ratio HL/HR{ylbl_sfx}',
                                     graph_cfg=graph_cfg,
                                     description='HL/HR intensity ratio across time bins.')
                        if ratio_sel['registry']:
                            ratio_sel['show'](0)

                # ── All Frames (default / first tab) ──
                if has_contour:
                    contour_all_nb = ttk.Notebook(contour_nb)
                    contour_nb.add(contour_all_nb, text='All Frames')
                    _contour_paw_nbs.append(contour_all_nb)
                    _build_contour_paw_tabs(
                        contour_all_nb, _contour_metrics, False, df, bdf,
                        'paw_area_ratio_hind', 'contact_intensity_ratio_hind')

                # ── Filter Preview (deferred until tab is clicked) ──
                if has_pawlike_contour:
                    contour_nb_frame = ttk.Frame(contour_nb)
                    contour_nb.add(contour_nb_frame, text='Filter Preview')
                    _filter_preview_built = [False]

                    def _build_filter_preview_content():
                        if _filter_preview_built[0]:
                            return
                        _filter_preview_built[0] = True
                        win = contour_nb.winfo_toplevel()
                        try:
                            win.config(cursor='watch')
                            win.update_idletasks()
                        except Exception:
                            pass
                        self._filter_sync_lr_var = tk.BooleanVar(value=True)
                        self._filter_slider_vars = {}
                        sync_frame = ttk.Frame(contour_nb_frame, padding=(6, 4, 6, 0))
                        sync_frame.pack(side='top', fill='x')
                        ttk.Checkbutton(sync_frame, text="Sync Left/Right thresholds",
                                        variable=self._filter_sync_lr_var).pack(anchor='w')
                        filter_nb = ttk.Notebook(contour_nb_frame)
                        filter_nb.pack(fill='both', expand=True)
                        _contour_paw_nbs.append(filter_nb)
                        for role in self.ROLES:
                            has_paw = any(
                                f'{mk}_{role}' in df.columns and df[f'{mk}_{role}'].notna().any()
                                for mk, _, _ in _contour_pawlike_metrics
                            )
                            if has_paw:
                                self._add_contour_filter_preview_tab(
                                    filter_nb, df, self._session_intermediates,
                                    role, tab_name=_paw_labels.get(role, role),
                                    graph_cfg=graph_cfg)
                        filter_nb.bind('<<NotebookTabChanged>>', _on_tab_change)
                        try:
                            win.config(cursor='')
                            win.update_idletasks()
                        except Exception:
                            pass

                    def _check_filter_preview_tab(evt=None):
                        try:
                            sel = contour_nb.select()
                            if sel == str(contour_nb_frame):
                                _build_filter_preview_content()
                        except Exception:
                            pass

                    contour_nb.bind('<<NotebookTabChanged>>', _check_filter_preview_tab, add='+')
                    _tab_descs['Filter Preview'] = (
                        'Interactive preview of the paw-like contour filter. '
                        'Adjust solidity, aspect ratio, and circularity thresholds '
                        'to see which paw shapes are included or excluded.')

                # ── Full Stance (optional) ──
                show_full_stance = _sets.get('full_stance', False)
                if has_stance_contour and show_full_stance:
                    contour_stance_nb = ttk.Notebook(contour_nb)
                    contour_nb.add(contour_stance_nb, text='Full Stance')
                    _contour_paw_nbs.append(contour_stance_nb)
                    _build_contour_paw_tabs(
                        contour_stance_nb, _contour_stance_metrics, True, df, bdf,
                        'paw_area_ratio_stance_hind', 'contact_intensity_ratio_stance_hind')

                # ── Filtered Contour (renamed from Paw-like) ──
                if has_pawlike_contour:
                    contour_pawlike_nb = ttk.Notebook(contour_nb)
                    contour_nb.add(contour_pawlike_nb, text='Filtered Contour')
                    _contour_paw_nbs.append(contour_pawlike_nb)
                    self._contour_pawlike_nb = contour_pawlike_nb
                    self._contour_nb = contour_nb
                    self._contour_pawlike_metrics = _contour_pawlike_metrics
                    self._contour_graph_cfg = graph_cfg
                    self._contour_build_fn = _build_contour_paw_tabs
                    _build_contour_paw_tabs(
                        contour_pawlike_nb, _contour_pawlike_metrics, False, df, bdf,
                        'pawlike_area_ratio_hind', 'pawlike_intensity_ratio_hind',
                        filter_paw=True)

                # Paw contour descriptions
                _contour_descs = {
                    'Area':         ('Mean paw contour area (px\u00b2) during contact frames. '
                                     'Larger area indicates greater paw-surface contact, which may '
                                     'reflect normal weight-bearing.'),
                    'Spread':       ('Maximum dimension (px) of paw contour bounding box. '
                                     'Larger spread indicates more toe-spreading or a flatter paw placement.'),
                    'Intensity':    ('Mean pixel brightness within paw contour shape during contact. '
                                     'Higher intensity indicates stronger paw-surface contact signal.'),
                    'Width':        ('Minimum dimension (px) of paw contour bounding box. '
                                     'Represents the narrower axis of the paw print.'),
                    'Solidity':     ('Solidity of paw contour \u2014 ratio of contour area to convex hull area. '
                                     'Values near 1.0 indicate a solid, compact paw print; lower values suggest '
                                     'irregular or fragmented contact.'),
                    'Aspect Ratio': ('Aspect ratio of paw contour bounding box (max/min dimension). '
                                     'Higher values indicate an elongated paw print; values near 1.0 indicate '
                                     'a round print.'),
                    'Circularity':  ('Circularity of paw contour \u2014 4\u03c0\u00d7area/perimeter\u00b2. '
                                     'Values near 1.0 indicate a circular shape; lower values indicate '
                                     'irregular or elongated contours.'),
                }
                for tab_prefix, desc in _contour_descs.items():
                    _tab_descs[tab_prefix] = desc
                    _tab_descs[f'{tab_prefix} — TC'] = desc + ' Shown across time bins.'

                for pl in _paw_labels.values():
                    _tab_descs[pl] = f'Contour metrics for {pl} paw.'
                _tab_descs['All Frames'] = 'Contour metrics computed on all contact frames (per-paw stance detection).'
                _tab_descs['Full Stance'] = 'Contour metrics restricted to frames where ALL contour paws are simultaneously in ground contact.'
                _tab_descs['Filtered Contour'] = ('Contour metrics filtered by solidity, aspect ratio, and circularity thresholds. '
                                                  'Adjust thresholds in Filter Preview and click Apply to update.')
                _tab_descs['Ratios'] = 'Left/Right paw contour ratios (HL/HR). Reference 1.0 = equal between sides.'
                _tab_descs['Shape'] = ('Mean paw contour outline averaged across contact frames. '
                                       'Normalized by contour area for size-independent shape comparison. '
                                       '\u00b11 SD envelope shown as shaded region.')
                _tab_descs['Area Ratio Hind'] = (
                    'Ratio of HL to HR paw contour area. Reference 1.0 = equal area. '
                    'Values >1 indicate larger left hind contact area.')
                _tab_descs['Area Ratio Hind — TC'] = (
                    'Ratio of HL to HR paw contour area across time bins. Reference 1.0 = equal area.')
                _tab_descs['Intensity Ratio Hind'] = (
                    'Ratio of HL to HR paw contour intensity. Reference 1.0 = equal intensity. '
                    'Values >1 indicate brighter left hind contact.')
                _tab_descs['Intensity Ratio Hind — TC'] = (
                    'Ratio of HL to HR paw contour intensity across time bins. Reference 1.0 = equal intensity.')

                # Bind tab-change handler to dynamically created inner notebooks
                for _nb in [contour_nb] + _contour_paw_nbs:
                    if _nb is not None:
                        _nb.bind('<<NotebookTabChanged>>', _on_tab_change)

        if contour_nb is not None:
            _register_deferred(contour_nb.master, _build_contour)

        # ── Statistics tab (deferred until clicked) ─────────────────────
        if show_stats and bdf is not None:
            _stats_group = ttk.Frame(outer_nb)
            outer_nb.add(_stats_group, text="\U0001f4ca Statistics")

            def _build_stats():
                self._log("  Building Statistics tables…")
                max_t = bdf['bin_start_s'].max() / 60.0
                self._create_wb_statistics_tab(
                    _stats_group, self._summary_df, bdf, treatments, max_t)

            _register_deferred(_stats_group, _build_stats)

        # ── Tab change handler (works for outer and all inner notebooks) ───
        def _on_tab_change(event):
            try:
                tab_text = event.widget.tab(event.widget.select(), 'text')
                desc_lbl.config(text=_tab_descs.get(tab_text, ''))
            except Exception:
                pass

        outer_nb.bind('<<NotebookTabChanged>>', _on_tab_change, add='+')
        for _inner in [wb_group, gait_group, movement_group,
                       contour_nb, contour_all_nb,
                       contour_stance_nb] + _contour_paw_nbs:
            if _inner is not None:
                _inner.bind('<<NotebookTabChanged>>', _on_tab_change)
        self._log("Graph window opened.")

    # ── Graph helpers ────────────────────────────────────────────────────────

    def _build_graph_settings_dlg(self, parent, treatments, max_time_min=None):
        """Show a graph settings dialog and return a config dict or None if cancelled."""
        dlg = tk.Toplevel(parent)
        dlg.title("Graph Settings")
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dw, dh = int(sw * 0.45), int(sh * 0.7)
        dlg.geometry(f"{dw}x{dh}+{(sw-dw)//2}+{(sh-dh)//2}")
        dlg.resizable(False, True)
        dlg.grab_set()
        _prev = self._last_graph_cfg or {}

        ttk.Label(dlg, text="Graph Settings",
                  font=('Arial', 13, 'bold'), padding=(12, 8, 12, 4)).pack(anchor='w')

        # ── Scrollable body ──────────────────────────────────────────────
        _scroll_outer = ttk.Frame(dlg)
        _scroll_outer.pack(fill='both', expand=True)
        _scroll_canvas = tk.Canvas(_scroll_outer, highlightthickness=0)
        _scroll_sb = ttk.Scrollbar(_scroll_outer, orient='vertical', command=_scroll_canvas.yview)
        _scroll_body = ttk.Frame(_scroll_canvas)

        _scroll_body.bind(
            '<Configure>',
            lambda e: _scroll_canvas.configure(scrollregion=_scroll_canvas.bbox('all')))
        _scroll_win = _scroll_canvas.create_window((0, 0), window=_scroll_body, anchor='nw')
        _scroll_canvas.configure(yscrollcommand=_scroll_sb.set)
        _scroll_canvas.bind('<Configure>',
            lambda e: _scroll_canvas.itemconfigure(_scroll_win, width=e.width))

        _scroll_canvas.pack(side='left', fill='both', expand=True)
        _scroll_sb.pack(side='right', fill='y')

        def _on_mousewheel(event):
            _scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        _scroll_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        def _cleanup_mousewheel():
            try:
                _scroll_canvas.unbind_all('<MouseWheel>')
            except Exception:
                pass
        dlg.bind('<Destroy>', lambda e: _cleanup_mousewheel() if e.widget is dlg else None)

        # Redirect all section packing to _scroll_body instead of dlg
        dlg_body = _scroll_body

        _sec = [0]  # mutable counter for dynamic section numbering
        def _next_sec():
            _sec[0] += 1
            return f"{_sec[0]}."

        # ── Treatment order (only if >1 treatment) ───────────────────────
        if len(treatments) > 1:
            ttk.Label(dlg_body, text=f"{_next_sec()} Treatment Order  (drag to reorder)",
                      font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
            lb_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 4))
            lb_frame.pack(fill='x')
            listbox = tk.Listbox(lb_frame, font=('Arial', 10),
                                 height=min(len(treatments), 8), selectmode='single')
            listbox.pack(side='left', fill='x', expand=True)
            sb = ttk.Scrollbar(lb_frame, orient='vertical', command=listbox.yview)
            sb.pack(side='right', fill='y')
            listbox.config(yscrollcommand=sb.set)
            for t in treatments:
                listbox.insert('end', t)

            def _drag_start(event):
                idx = listbox.nearest(event.y)
                listbox.selection_clear(0, 'end')
                listbox.selection_set(idx)
                listbox.activate(idx)
                listbox._drag_idx = idx
                listbox._drag_item = listbox.get(idx)

            def _drag_motion(event):
                idx = listbox.nearest(event.y)
                if hasattr(listbox, '_drag_idx') and idx != listbox._drag_idx:
                    listbox.delete(listbox._drag_idx)
                    listbox.insert(idx, listbox._drag_item)
                    listbox._drag_idx = idx
                    listbox.selection_clear(0, 'end')
                    listbox.selection_set(idx)

            listbox.bind('<Button-1>', _drag_start)
            listbox.bind('<B1-Motion>', _drag_motion)
        else:
            listbox = None

        # ── Treatment colors ──────────────────────────────────────────────
        ttk.Label(dlg_body, text=f"{_next_sec()} Treatment Colors",
                  font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
        color_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 4))
        color_frame.pack(fill='x')

        color_options = {
            'Teal':                  '#66c2a5',
            'Orange':                '#fc8d62',
            'Purple':                '#8da0cb',
            'Pink':                  '#e78ac3',
            'Green':                 '#a6d854',
            'Yellow':                '#ffd92f',
            'Brown':                 '#e5c494',
            'Gray':                  '#b3b3b3',
            'Red':                   '#d62728',
            'Blue':                  '#1f77b4',
            'Navy':                  '#2c4a7c',
            'Coral':                 '#ff6b6b',
            'Lavender':              '#b39ddb',
            'Mint':                  '#69d2b8',
            'Gold':                  '#e6a817',
            'Slate':                 '#78909c',
            'Salmon':                '#fa8072',
            'Olive':                 '#8fad52',
            'White (black outline)': 'white_black',
        }
        PRESETS = {
            'Colorblind-safe': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'],
            'Vivid':           ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
            'Pastel':          ['#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc'],
        }
        _VEH_KW = {'vehicle', 'veh', 'saline', 'control', 'ctrl', 'acsf', 'water', 'pbs', 'naive'}
        def _is_veh(t):
            return any(kw in str(t).lower() for kw in _VEH_KW)

        non_veh_keys = [k for k in color_options if k != 'White (black outline)']
        picked_hex = {}
        swatch_labels = {}
        color_vars = {}

        def _swatch_color(t):
            if t in picked_hex:
                return picked_hex[t]
            val = color_options.get(color_vars[t].get(), '#cccccc')
            return val if val != 'white_black' else 'white'

        _preview_hooks = []  # filled after _update_marker_preview is defined

        def _update_swatch(t):
            swatch_labels[t].configure(bg=_swatch_color(t))
            for _hook in _preview_hooks:
                _hook()

        def _apply_preset(hex_list):
            non_veh = [t for t in treatments if not _is_veh(t)]
            veh_list = [t for t in treatments if _is_veh(t)]
            for i, t in enumerate(non_veh):
                picked_hex[t] = hex_list[i % len(hex_list)]
                _update_swatch(t)
            for t in veh_list:
                picked_hex.pop(t, None)
                color_vars[t].set('White (black outline)')
                _update_swatch(t)

        # ── Color mode selection ─────────────────────────────────────────
        mode_row = ttk.Frame(color_frame)
        mode_row.pack(fill='x', pady=(4, 2))
        _prev_color_mode = _prev.get('color_mode', 'individual')
        color_mode_var = tk.StringVar(value=_prev_color_mode)
        ttk.Label(mode_row, text="Color mode:", font=('Arial', 9, 'bold')).pack(side='left', padx=(0, 8))
        ttk.Radiobutton(mode_row, text='Individual',
                        variable=color_mode_var, value='individual').pack(side='left')
        ttk.Radiobutton(mode_row, text='Gradient (dose response)',
                        variable=color_mode_var, value='gradient').pack(side='left', padx=10)

        # ── Gradient controls frame ─────────────────────────────────────
        gradient_frame = ttk.Frame(color_frame, padding=(8, 4))

        gmap_row = ttk.Frame(gradient_frame)
        gmap_row.pack(fill='x', pady=2)
        ttk.Label(gmap_row, text="Colormap:", width=15).pack(side='left')
        _prev_grad_cmap = _prev.get('gradient_cmap', 'Blues')
        gradient_cmap_var = tk.StringVar(value=_prev_grad_cmap)
        gradient_cmaps = [
            'Blues', 'Purples', 'Oranges', 'Greens', 'Reds',
            'YlOrRd', 'YlGnBu', 'PuRd', 'BuGn', 'RdPu', 'BuPu', 'GnBu', 'OrRd',
            'plasma', 'viridis', 'magma', 'inferno', 'cividis',
            'cool', 'hot', 'copper', 'bone',
        ]
        ttk.Combobox(gmap_row, textvariable=gradient_cmap_var,
                     values=gradient_cmaps, state='readonly', width=15).pack(side='left', padx=5)

        gdir_row = ttk.Frame(gradient_frame)
        gdir_row.pack(fill='x', pady=2)
        ttk.Label(gdir_row, text="Direction:", width=15).pack(side='left')
        _prev_grad_dir = _prev.get('gradient_dir', 'light_to_dark')
        gradient_dir_var = tk.StringVar(value=_prev_grad_dir)
        ttk.Radiobutton(gdir_row, text='Light \u2192 Dark (increasing dose)',
                        variable=gradient_dir_var, value='light_to_dark').pack(side='left')
        ttk.Radiobutton(gdir_row, text='Dark \u2192 Light',
                        variable=gradient_dir_var, value='dark_to_light').pack(side='left', padx=10)

        gveh_row = ttk.Frame(gradient_frame)
        gveh_row.pack(fill='x', pady=2)
        ttk.Label(gveh_row, text="Vehicle/Control:", width=15).pack(side='left')
        _auto_veh = next((t for t in treatments if _is_veh(t)), treatments[0])
        _prev_grad_veh = _prev.get('gradient_vehicle', _auto_veh)
        if _prev_grad_veh not in treatments:
            _prev_grad_veh = _auto_veh
        gradient_veh_var = tk.StringVar(value=_prev_grad_veh)
        ttk.Combobox(gveh_row, textvariable=gradient_veh_var,
                     values=list(treatments), state='readonly', width=15).pack(side='left', padx=5)
        ttk.Label(gveh_row, text='\u2192 white with black outline',
                  font=('Arial', 9), foreground='gray').pack(side='left')

        # Gradient preview swatches
        grad_preview_row = ttk.Frame(gradient_frame)
        grad_preview_row.pack(fill='x', pady=(4, 2))
        grad_swatch_labels = {}
        for t in treatments:
            tf = ttk.Frame(grad_preview_row)
            tf.pack(side='left', padx=3)
            gsw = tk.Label(tf, width=2, bg='white', relief='solid', bd=1)
            gsw.pack(side='left')
            ttk.Label(tf, text=t, font=('Arial', 8)).pack(side='left', padx=(2, 0))
            grad_swatch_labels[t] = gsw

        def _refresh_grad_swatches(*_):
            import matplotlib.colors as mcolors
            veh = gradient_veh_var.get()
            try:
                cmap = plt.get_cmap(gradient_cmap_var.get())
            except Exception:
                return
            non_veh = [t for t in treatments if t != veh]
            n = len(non_veh)
            if gradient_dir_var.get() == 'light_to_dark':
                positions = np.linspace(0.35, 0.90, max(n, 1))
            else:
                positions = np.linspace(0.90, 0.35, max(n, 1))
            for t, gsw in grad_swatch_labels.items():
                if t == veh:
                    gsw.configure(bg='white')
                else:
                    idx = non_veh.index(t) if t in non_veh else 0
                    gsw.configure(bg=mcolors.to_hex(cmap(positions[idx])) if idx < len(positions) else '#ccc')

        gradient_cmap_var.trace_add('write', _refresh_grad_swatches)
        gradient_dir_var.trace_add('write', _refresh_grad_swatches)
        gradient_veh_var.trace_add('write', _refresh_grad_swatches)
        _refresh_grad_swatches()

        # ── Individual colors frame (preset + per-treatment rows) ────────
        individual_frame = ttk.Frame(color_frame)

        preset_row = ttk.Frame(individual_frame)
        preset_row.pack(fill='x', pady=(0, 6))
        ttk.Label(preset_row, text="Quick palettes:", font=('Arial', 9)).pack(side='left', padx=(0, 8))
        for lbl, hexes in PRESETS.items():
            ttk.Button(preset_row, text=lbl, width=16,
                       command=lambda h=hexes: _apply_preset(h)).pack(side='left', padx=3)

        def _toggle_color_mode(*_):
            if color_mode_var.get() == 'gradient':
                individual_frame.pack_forget()
                gradient_frame.pack(fill='x', pady=(4, 2))
            else:
                gradient_frame.pack_forget()
                individual_frame.pack(fill='x')

        color_mode_var.trace_add('write', _toggle_color_mode)
        _toggle_color_mode()

        # ── Per-treatment style option lists ──────────────────────────────
        _LINESTYLE_OPTIONS = [
            ("Solid \u2500",      "-"),
            ("Dashed \u2504",     "--"),
            ("Dotted \u00b7\u00b7\u00b7",   ":"),
            ("Dash-dot \u2500\u00b7",  "-."),
        ]
        _ls_labels = [lbl for lbl, _ in _LINESTYLE_OPTIONS]
        _ls_values = [val for _, val in _LINESTYLE_OPTIONS]

        _MARKER_OPTIONS = [
            ("None (line only)", "none"),
            ("Circle \u25cf",   "o"),
            ("Square \u25a0",   "s"),
            ("Triangle \u25b2", "^"),
            ("Diamond \u25c6",  "d"),
            ("Plus +",          "+"),
            ("X \u2715",        "x"),
        ]
        _mk_labels = [lbl for lbl, _ in _MARKER_OPTIONS]
        _mk_values = [val for _, val in _MARKER_OPTIONS]

        # Backward compat: migrate old global marker/line_style keys
        if _prev.get('line_style') == 'markers' and 'marker_shape' not in _prev:
            _prev['marker_shape'] = 'o'
            _prev['marker_size'] = 4
        if 'marker_shape' in _prev and 'marker_shapes' not in _prev:
            _prev['marker_shapes'] = {t: _prev['marker_shape'] for t in treatments}
            _prev['marker_sizes'] = {t: _prev.get('marker_size', 5) for t in treatments}

        line_style_vars = {}
        line_width_vars = {}
        marker_shape_combos = {}
        marker_fill_combos = {}
        marker_size_vars = {}
        opacity_vars = {}
        marker_edge_combos = {}

        _EDGE_OPTIONS = [
            ("Auto",       "auto"),
            ("Black",      "black"),
            ("White",      "white"),
            ("Match line", "match"),
        ]
        _edge_labels = [lbl for lbl, _ in _EDGE_OPTIONS]
        _edge_values = [val for _, val in _EDGE_OPTIONS]

        _FILL_OPTIONS = [
            ("Filled",       "full"),
            ("Open",         "none"),
            ("Left half",    "left"),
            ("Right half",   "right"),
            ("Bottom half",  "bottom"),
            ("Top half",     "top"),
        ]
        _fill_labels = [lbl for lbl, _ in _FILL_OPTIONS]
        _fill_values = [val for _, val in _FILL_OPTIONS]

        _nv_idx = 0
        for _ti, treat in enumerate(treatments):
            treat_frame = ttk.Frame(individual_frame)
            treat_frame.pack(fill='x', pady=(4, 2))

            # Row 1: color swatch + name + color combo + custom button
            row1 = ttk.Frame(treat_frame)
            row1.pack(fill='x')
            sw = tk.Label(row1, width=2, bg='white', relief='solid', bd=1)
            sw.pack(side='left', padx=(0, 5))
            swatch_labels[treat] = sw
            ttk.Label(row1, text=f"{treat}:", width=15, anchor='w').pack(side='left')
            if _is_veh(treat):
                default = 'White (black outline)'
            else:
                default = non_veh_keys[_nv_idx % len(non_veh_keys)]
                _nv_idx += 1
            cv = tk.StringVar(value=default)
            color_vars[treat] = cv
            cb = ttk.Combobox(row1, textvariable=cv, values=list(color_options.keys()),
                              state='readonly', width=20)
            cb.pack(side='left', padx=5)
            cb.bind('<<ComboboxSelected>>',
                    lambda e, t=treat: (picked_hex.pop(t, None), _update_swatch(t)))

            def _pick_custom(t=treat):
                from tkinter import colorchooser
                res = colorchooser.askcolor(color=_swatch_color(t),
                                            title=f"Pick color for {t}", parent=dlg)
                if res and res[1]:
                    picked_hex[t] = res[1]
                    _update_swatch(t)
            ttk.Button(row1, text="Custom\u2026", command=_pick_custom, width=8).pack(side='left', padx=2)
            _update_swatch(treat)

            # Row 2: line style, width, marker shape, marker size, opacity
            row2 = ttk.Frame(treat_frame)
            row2.pack(fill='x', padx=(22, 0), pady=(1, 0))

            ttk.Label(row2, text="Line:", font=('Arial', 9)).pack(side='left')
            _prev_ls = _prev.get('line_styles', {}).get(treat, '-')
            _ls_init = _ls_values.index(_prev_ls) if _prev_ls in _ls_values else 0
            ls_combo = ttk.Combobox(row2, values=_ls_labels, state='readonly', width=14)
            ls_combo.current(_ls_init)
            ls_combo.pack(side='left', padx=(2, 6))
            line_style_vars[treat] = ls_combo

            ttk.Label(row2, text="W:", font=('Arial', 9)).pack(side='left')
            _prev_lw = _prev.get('line_widths', {}).get(treat, 1.8)
            lw_var = tk.DoubleVar(value=_prev_lw)
            lw_spin = ttk.Spinbox(row2, from_=0.5, to=5.0, increment=0.5,
                                  textvariable=lw_var, width=4)
            lw_spin.pack(side='left', padx=(2, 6))
            line_width_vars[treat] = lw_var

            ttk.Label(row2, text="Marker:", font=('Arial', 9)).pack(side='left')
            _prev_mk = _prev.get('marker_shapes', {}).get(treat, 'o')
            _mk_init = _mk_values.index(_prev_mk) if _prev_mk in _mk_values else 0
            mk_combo = ttk.Combobox(row2, values=_mk_labels, state='readonly', width=14)
            mk_combo.current(_mk_init)
            mk_combo.pack(side='left', padx=(2, 6))
            marker_shape_combos[treat] = mk_combo

            ttk.Label(row2, text="Fill:", font=('Arial', 9)).pack(side='left')
            _prev_fill = _prev.get('marker_fills', {}).get(treat, 'full')
            _fill_init = _fill_values.index(_prev_fill) if _prev_fill in _fill_values else 0
            fill_combo = ttk.Combobox(row2, values=_fill_labels, state='readonly', width=10)
            fill_combo.current(_fill_init)
            fill_combo.pack(side='left', padx=(2, 6))
            marker_fill_combos[treat] = fill_combo

            ttk.Label(row2, text="Sz:", font=('Arial', 9)).pack(side='left')
            _prev_ms = _prev.get('marker_sizes', {}).get(treat, 5)
            ms_var = tk.IntVar(value=_prev_ms)
            ms_spin = ttk.Spinbox(row2, from_=3, to=12, increment=1,
                                  textvariable=ms_var, width=3)
            ms_spin.pack(side='left', padx=(2, 6))
            marker_size_vars[treat] = ms_var

            ttk.Label(row2, text="\u03b1:", font=('Arial', 9)).pack(side='left')
            _prev_op = _prev.get('opacities', {}).get(treat, 1.0)
            op_var = tk.DoubleVar(value=_prev_op)
            op_spin = ttk.Spinbox(row2, from_=0.1, to=1.0, increment=0.1,
                                  textvariable=op_var, width=4)
            op_spin.pack(side='left', padx=(2, 6))
            opacity_vars[treat] = op_var

            ttk.Label(row2, text="Edge:", font=('Arial', 9)).pack(side='left')
            _prev_edge = _prev.get('marker_edge_colors', {}).get(treat, 'auto')
            _edge_init = _edge_values.index(_prev_edge) if _prev_edge in _edge_values else 0
            edge_combo = ttk.Combobox(row2, values=_edge_labels, state='readonly', width=10)
            edge_combo.current(_edge_init)
            edge_combo.pack(side='left', padx=(2, 0))
            marker_edge_combos[treat] = edge_combo

        # ── Error bar type ────────────────────────────────────────────────
        ttk.Label(dlg_body, text=f"{_next_sec()} Error Bar Type",
                  font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
        err_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 4))
        err_frame.pack(fill='x')
        error_var = tk.StringVar(value=_prev.get('error_type', 'SEM'))
        ttk.Radiobutton(err_frame, text="SEM (Standard Error of the Mean)",
                        variable=error_var, value='SEM').pack(anchor='w')
        ttk.Radiobutton(err_frame, text="SD (Standard Deviation)",
                        variable=error_var, value='SD').pack(anchor='w')
        ttk.Radiobutton(err_frame, text="95% CI (Confidence Interval)",
                        variable=error_var, value='95CI').pack(anchor='w')
        ttk.Label(err_frame, text="SEM = SD/\u221an    95% CI = t* \u00d7 SEM  (uses Student's t distribution)",
                  font=('Arial', 9), foreground='gray').pack(anchor='w', pady=(2, 0))

        ttk.Label(err_frame, text="Display style:", font=('Arial', 10),
                  padding=(0, 6, 0, 0)).pack(anchor='w')
        err_display_var = tk.StringVar(value=_prev.get('error_display', 'circles_caps'))
        ttk.Radiobutton(err_frame, text="Shaded region",
                        variable=err_display_var, value='shaded').pack(anchor='w')
        ttk.Radiobutton(err_frame, text="Line with caps",
                        variable=err_display_var, value='caps').pack(anchor='w')
        ttk.Radiobutton(err_frame, text="Circles + capped error bars",
                        variable=err_display_var, value='circles_caps').pack(anchor='w')

        # ── Display Options (preview + individual traces) ───────────────
        sig_style_var = tk.StringVar(value=_prev.get('sig_style', 'asterisk'))
        ttk.Label(dlg_body, text=f"{_next_sec()} Display Options",
                  font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
        disp_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 8))
        disp_frame.pack(fill='x')
        indiv_var = tk.BooleanVar(value=_prev.get('show_individual', False))
        ttk.Checkbutton(disp_frame,
                        text="Show individual animal traces on timecourse (spaghetti plot)",
                        variable=indiv_var).pack(anchor='w')

        # Live preview canvas
        _preview_frame = ttk.Frame(disp_frame, padding=(12, 4, 12, 0))
        _preview_frame.pack(anchor='w', fill='x')
        _prev_fig, _prev_ax = plt.subplots(figsize=(3.5, 1.0), constrained_layout=True)
        _prev_fig.patch.set_facecolor('#f0f0f0')
        _prev_canvas = FigureCanvasTkAgg(_prev_fig, master=_preview_frame)
        _prev_canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(_prev_canvas, _prev_fig)

        def _resolve_preview_colors():
            """Get first two treatment colors + raw values from the picker."""
            clrs = []
            raws = []
            for t in list(treatments)[:2]:
                if t in picked_hex:
                    c = picked_hex[t]
                else:
                    c = color_options.get(color_vars[t].get(), '#1f77b4')
                raws.append(c)
                clrs.append('black' if c == 'white_black' else c)
            while len(clrs) < 2:
                clrs.append('#1f77b4')
                raws.append(None)
            return clrs, raws

        def _update_marker_preview(*_args):
            _prev_ax.clear()
            _prev_ax.set_xlim(-0.3, 4.8)
            _prev_ax.set_ylim(-1.8, 2.0)
            _prev_ax.axis('off')
            _xp = np.linspace(0, 4, 7)
            _yps = [np.sin(_xp * 1.1) * 0.7 + 0.3,
                    np.cos(_xp * 1.1) * 0.5 - 0.2]
            _errs = [np.abs(np.cos(_xp * 0.8)) * 0.3 + 0.15,
                     np.abs(np.sin(_xp * 0.9)) * 0.25 + 0.12]
            _pclrs, _raws = _resolve_preview_colors()
            _disp_mode = err_display_var.get()
            for _idx, treat in enumerate(list(treatments)[:2]):
                _yp = _yps[_idx] if _idx < len(_yps) else _yps[0]
                _er = _errs[_idx] if _idx < len(_errs) else _errs[0]
                _clr = _pclrs[_idx] if _idx < len(_pclrs) else '#1f77b4'
                _raw_c = _raws[_idx] if _idx < len(_raws) else None
                _ls_cb = line_style_vars.get(treat)
                _ls_idx = _ls_cb.current() if _ls_cb else 0
                _ls_val = _ls_values[_ls_idx] if 0 <= _ls_idx < len(_ls_values) else '-'
                _lw = line_width_vars.get(treat, tk.DoubleVar(value=1.8)).get()
                _mk_cb = marker_shape_combos.get(treat)
                _mk_idx = _mk_cb.current() if _mk_cb else 0
                _mk_val = _mk_values[_mk_idx] if 0 <= _mk_idx < len(_mk_values) else 'o'
                _fl_cb = marker_fill_combos.get(treat)
                _fl_idx = _fl_cb.current() if _fl_cb else 0
                _fl_val = _fill_values[_fl_idx] if 0 <= _fl_idx < len(_fill_values) else 'full'
                _ms = marker_size_vars.get(treat, tk.IntVar(value=5)).get()
                _op = opacity_vars.get(treat, tk.DoubleVar(value=1.0)).get()
                pkw = dict(color=_clr, linewidth=_lw, linestyle=_ls_val, alpha=_op)
                if _mk_val != 'none':
                    pkw['marker'] = _mk_val
                    pkw['markersize'] = _ms
                    pkw['fillstyle'] = _fl_val
                    if _fl_val == 'none':
                        pkw['markeredgewidth'] = 1.2
                    elif _fl_val in ('left', 'right', 'top', 'bottom'):
                        pkw['markerfacecoloralt'] = 'white'
                if _raw_c == 'white_black':
                    pkw['markerfacecolor'] = 'white'
                    pkw['markeredgecolor'] = 'black'
                _ec_cb = marker_edge_combos.get(treat)
                _ec_idx = _ec_cb.current() if _ec_cb else 0
                _ec_val = _edge_values[_ec_idx] if 0 <= _ec_idx < len(_edge_values) else 'auto'
                if _ec_val == 'black':
                    pkw['markeredgecolor'] = 'black'
                elif _ec_val == 'white':
                    pkw['markeredgecolor'] = 'white'
                elif _ec_val == 'match':
                    pkw['markeredgecolor'] = _clr
                if _disp_mode == 'circles_caps':
                    _line_kw = {k: v for k, v in pkw.items()
                                if k not in ('marker', 'markersize', 'fillstyle',
                                             'markeredgewidth', 'markerfacecoloralt',
                                             'markerfacecolor', 'markeredgecolor')}
                    _prev_ax.plot(_xp, _yp, **_line_kw)
                else:
                    _prev_ax.plot(_xp, _yp, **pkw)
                if _disp_mode == 'caps':
                    _prev_ax.errorbar(_xp, _yp, yerr=_er, capsize=4,
                                      capthick=1.5, fmt='none', alpha=0.6 * _op,
                                      ecolor=_clr)
                elif _disp_mode == 'circles_caps':
                    _prev_ax.errorbar(_xp, _yp, yerr=_er, capsize=4,
                                      capthick=1.5, fmt='none', alpha=0.6 * _op,
                                      ecolor=_clr)
                    _mkw = dict(linestyle='none', color=_clr,
                                marker='o', markersize=_ms, alpha=_op,
                                fillstyle=_fl_val)
                    if _fl_val == 'none':
                        _mkw['markeredgewidth'] = 1.2
                    elif _fl_val in ('left', 'right', 'top', 'bottom'):
                        _mkw['markerfacecoloralt'] = 'white'
                    if _raw_c == 'white_black':
                        _mkw['markerfacecolor'] = 'white'
                        _mkw['markeredgecolor'] = 'black'
                    if _ec_val == 'black':
                        _mkw['markeredgecolor'] = 'black'
                    elif _ec_val == 'white':
                        _mkw['markeredgecolor'] = 'white'
                    elif _ec_val == 'match':
                        _mkw['markeredgecolor'] = _clr
                    _prev_ax.plot(_xp, _yp, **_mkw)
                else:
                    _prev_ax.fill_between(_xp, _yp - _er, _yp + _er,
                                          alpha=0.18 * _op, color=_clr)
            # Sample significance marker
            from matplotlib.transforms import blended_transform_factory
            _ptrans = blended_transform_factory(_prev_ax.transData,
                                                _prev_ax.transAxes)
            _sig_sample = _SIG_STYLES.get(sig_style_var.get(), ('*',))[0]
            _prev_ax.text(_xp[3], 0.95, _sig_sample, transform=_ptrans,
                          ha='center', va='top', fontsize=10, color='black')
            # Error type label
            _edisp_lbl = f'{error_var.get()} ({err_display_var.get()})'
            _prev_ax.text(0.99, 0.02, _edisp_lbl,
                          transform=_prev_ax.transAxes, ha='right', va='bottom',
                          fontsize=8, color='gray', fontstyle='italic')
            _prev_canvas.draw_idle()

        # Bind per-treatment combos/spinboxes to preview update
        for _t in treatments:
            if _t in line_style_vars:
                line_style_vars[_t].bind('<<ComboboxSelected>>', _update_marker_preview)
            if _t in marker_shape_combos:
                marker_shape_combos[_t].bind('<<ComboboxSelected>>', _update_marker_preview)
            if _t in marker_fill_combos:
                marker_fill_combos[_t].bind('<<ComboboxSelected>>', _update_marker_preview)
            if _t in line_width_vars:
                line_width_vars[_t].trace_add('write', _update_marker_preview)
            if _t in marker_size_vars:
                marker_size_vars[_t].trace_add('write', _update_marker_preview)
            if _t in opacity_vars:
                opacity_vars[_t].trace_add('write', _update_marker_preview)
            if _t in marker_edge_combos:
                marker_edge_combos[_t].bind('<<ComboboxSelected>>', _update_marker_preview)
        error_var.trace_add('write', _update_marker_preview)
        err_display_var.trace_add('write', _update_marker_preview)
        sig_style_var.trace_add('write', _update_marker_preview)
        _preview_hooks.append(_update_marker_preview)
        _update_marker_preview()  # initial draw

        # ── Statistical Tests ────────────────────────────────────────────
        ttk.Label(dlg_body, text=f"{_next_sec()} Statistical Tests",
                  font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
        st_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 8))
        st_frame.pack(fill='x')

        stats_var = tk.BooleanVar(value=_prev.get('show_stats', self._enable_stats_var.get()))
        ttk.Checkbutton(st_frame,
                        text="Show significance markers on graphs  (* p<0.05  ** p<0.01  *** p<0.001)",
                        variable=stats_var).pack(anchor='w')

        _test_row = ttk.Frame(st_frame)
        _test_row.pack(fill='x', pady=(4, 2))
        ttk.Label(_test_row, text="Test:", width=6).pack(side='left')
        dlg_test_var = tk.StringVar(value=self._stats_test_var.get())
        for txt, val in [("Auto", "auto"), ("t-test", "t-test"),
                         ("ANOVA", "ANOVA"), ("Non-param", "nonparametric")]:
            ttk.Radiobutton(_test_row, text=txt, variable=dlg_test_var,
                            value=val).pack(side='left', padx=(0, 4))

        _alpha_row = ttk.Frame(st_frame)
        _alpha_row.pack(fill='x', pady=(0, 2))
        ttk.Label(_alpha_row, text="Alpha:", width=6).pack(side='left')
        dlg_alpha_var = tk.DoubleVar(value=self._stats_alpha_var.get())
        for txt, val in [("0.05", 0.05), ("0.01", 0.01), ("0.001", 0.001)]:
            ttk.Radiobutton(_alpha_row, text=txt, variable=dlg_alpha_var,
                            value=val).pack(side='left', padx=(0, 4))

        _par_row = ttk.Frame(st_frame)
        _par_row.pack(fill='x', pady=(0, 2))
        ttk.Label(_par_row, text="Mode:", width=6).pack(side='left')
        dlg_paradigm_var = tk.StringVar(value=self._stats_paradigm_var.get())
        for txt, val in [("Parametric", "parametric"),
                         ("Non-param", "nonparametric"),
                         ("Auto", "auto")]:
            ttk.Radiobutton(_par_row, text=txt, variable=dlg_paradigm_var,
                            value=val).pack(side='left', padx=(0, 4))

        dlg_posthoc_var = tk.BooleanVar(value=self._timecourse_posthoc_var.get())
        ttk.Checkbutton(st_frame, text="Per-bin post-hoc on timecourse",
                        variable=dlg_posthoc_var).pack(anchor='w', pady=(2, 0))

        _sig_row = ttk.Frame(st_frame)
        _sig_row.pack(fill='x', pady=(4, 0))
        ttk.Label(_sig_row, text="Marker style:", width=12).pack(side='left')
        for txt, val in [("Asterisk (* ** ***)", "asterisk"),
                         ("Hash (# ## ###)", "hash"),
                         ("Dagger (\u2020 \u2020\u2020 \u2020\u2020\u2020)", "dagger"),
                         ("Letters (a b c)", "letters")]:
            ttk.Radiobutton(_sig_row, text=txt, variable=sig_style_var,
                            value=val).pack(side='left', padx=(0, 6))

        # ── Time window (only for bin graphs) ────────────────────────────
        time_var = None
        if max_time_min is not None:
            ttk.Label(dlg_body, text=f"{_next_sec()} Time Window",
                      font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
            tw_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 8))
            tw_frame.pack(fill='x')
            ttk.Label(tw_frame, text="Show data up to:").pack(side='left')
            time_var = tk.IntVar(value=int(max_time_min))
            ttk.Spinbox(tw_frame, from_=1, to=int(max_time_min),
                        textvariable=time_var, width=8).pack(side='left', padx=5)
            ttk.Label(tw_frame, text=f"minutes  (max: {int(max_time_min)} min)").pack(side='left')

        # ── Re-bin (aggregate bins for display) ──────────────────────────
        rebin_var = None
        if max_time_min is not None:
            ttk.Label(dlg_body, text=f"{_next_sec()} Display Bin Size",
                      font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
            rb_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 8))
            rb_frame.pack(fill='x')
            ttk.Label(rb_frame, text="Aggregate bins to:").pack(side='left')
            rebin_var = tk.DoubleVar(value=_prev.get('rebin_minutes', 0))
            ttk.Spinbox(rb_frame, from_=0, to=max_time_min,
                        increment=0.5,
                        textvariable=rebin_var, width=8).pack(side='left', padx=5)
            ttk.Label(rb_frame, text="minutes  (0 = use original bins)").pack(side='left')

        # ── Graph Sets ─────────────────────────────────────────────────────
        ttk.Label(dlg_body, text=f"{_next_sec()} Graph Sets to Show",
                  font=('Arial', 11, 'bold'), padding=(12, 6, 12, 2)).pack(anchor='w')
        gs_frame = ttk.Frame(dlg_body, padding=(16, 0, 16, 8))
        gs_frame.pack(fill='x')

        _graph_sets = [
            ('weight_bearing', 'Limb Use',
             'Hind/fore paw limb use indices, contact %, brightness ratios'),
            ('gait', 'Gait',
             'Stance/swing timing, stride/step lengths, symmetry indices'),
            ('movement', 'Movement',
             'Locomotion distance & speed, coordination & support patterns'),
            ('paw_contour', 'Paw Contour',
             'Paw print shape, area, spread, intensity, contour filter'),
            ('statistics', 'Statistics',
             'Summary statistics table and significance tests across metrics'),
        ]

        gs_vars = {}
        prev_sets = _prev.get('graph_sets', {})
        for key, label, desc in _graph_sets:
            var = tk.BooleanVar(value=prev_sets.get(key, True))
            gs_vars[key] = var
            row_f = ttk.Frame(gs_frame)
            row_f.pack(fill='x', pady=1)
            ttk.Checkbutton(row_f, text=label, variable=var).pack(side='left')
            ttk.Label(row_f, text=f"  \u2014 {desc}", foreground='gray',
                      font=('Arial', 9)).pack(side='left')

        # Sub-option for Full Stance under Paw Contour
        fs_var = tk.BooleanVar(value=prev_sets.get('full_stance', False))
        gs_vars['full_stance'] = fs_var
        fs_row = ttk.Frame(gs_frame)
        fs_row.pack(fill='x', pady=1, padx=(24, 0))
        ttk.Checkbutton(fs_row, text="Include Full Stance metrics",
                        variable=fs_var).pack(side='left')
        ttk.Label(fs_row, text="  \u2014 Paw metrics only during frames where all 4 paws contact simultaneously",
                  foreground='gray', font=('Arial', 9)).pack(side='left')

        # ── Buttons ───────────────────────────────────────────────────────
        result = [None]

        def _on_ok():
            order = ([listbox.get(i) for i in range(listbox.size())]
                     if listbox is not None else list(treatments))
            if color_mode_var.get() == 'gradient':
                import matplotlib.colors as mcolors
                veh = gradient_veh_var.get()
                cmap = plt.get_cmap(gradient_cmap_var.get())
                non_veh = [t for t in order if t != veh]
                n = len(non_veh)
                if gradient_dir_var.get() == 'light_to_dark':
                    positions = np.linspace(0.35, 0.90, max(n, 1))
                else:
                    positions = np.linspace(0.90, 0.35, max(n, 1))
                colors = {t: mcolors.to_hex(cmap(pos))
                          for t, pos in zip(non_veh, positions)}
                colors[veh] = 'white_black'
                for t in treatments:
                    if t not in colors:
                        colors[t] = '#b3b3b3'
            else:
                colors = {}
                for t in treatments:
                    if t in picked_hex:
                        colors[t] = picked_hex[t]
                    else:
                        colors[t] = color_options[color_vars[t].get()]
            # Propagate stats settings back to instance vars
            self._stats_test_var.set(dlg_test_var.get())
            self._stats_alpha_var.set(dlg_alpha_var.get())
            self._stats_paradigm_var.set(dlg_paradigm_var.get())
            self._timecourse_posthoc_var.set(dlg_posthoc_var.get())
            # Build per-treatment style dicts
            _ls_dict, _mk_dict, _mf_dict = {}, {}, {}
            _ms_dict, _lw_dict, _op_dict = {}, {}, {}
            _me_dict = {}
            for t in treatments:
                _lc = line_style_vars.get(t)
                _li = _lc.current() if _lc else 0
                _ls_dict[t] = _ls_values[_li] if 0 <= _li < len(_ls_values) else '-'
                _mc = marker_shape_combos.get(t)
                _mi = _mc.current() if _mc else 0
                _mk_dict[t] = _mk_values[_mi] if 0 <= _mi < len(_mk_values) else 'o'
                _fc = marker_fill_combos.get(t)
                _fi = _fc.current() if _fc else 0
                _mf_dict[t] = _fill_values[_fi] if 0 <= _fi < len(_fill_values) else 'full'
                _ms_dict[t] = marker_size_vars.get(t, tk.IntVar(value=5)).get()
                _lw_dict[t] = line_width_vars.get(t, tk.DoubleVar(value=1.8)).get()
                _op_dict[t] = opacity_vars.get(t, tk.DoubleVar(value=1.0)).get()
                _ec = marker_edge_combos.get(t)
                _ei = _ec.current() if _ec else 0
                _me_dict[t] = _edge_values[_ei] if 0 <= _ei < len(_edge_values) else 'auto'
            result[0] = {
                'order':           order,
                'colors':          colors,
                'error_type':      error_var.get(),
                'error_display':   err_display_var.get(),
                'time_window':     int(time_var.get()) if time_var is not None else None,
                'show_stats':      stats_var.get(),
                'show_individual': indiv_var.get(),
                'line_styles':     _ls_dict,
                'marker_shapes':   _mk_dict,
                'marker_fills':    _mf_dict,
                'marker_sizes':    _ms_dict,
                'line_widths':     _lw_dict,
                'opacities':       _op_dict,
                'marker_edge_colors': _me_dict,
                'rebin_minutes':   float(rebin_var.get()) if rebin_var is not None else 0,
                'sig_style':       sig_style_var.get(),
                'graph_sets':      {k: v.get() for k, v in gs_vars.items()},
                'color_mode':       color_mode_var.get(),
                'gradient_cmap':    gradient_cmap_var.get(),
                'gradient_dir':     gradient_dir_var.get(),
                'gradient_vehicle': gradient_veh_var.get(),
            }
            dlg.destroy()

        btn_bar = ttk.Frame(dlg, padding=(12, 8))
        btn_bar.pack(fill='x')
        ttk.Button(btn_bar, text="OK",     command=_on_ok,        width=14).pack(side='right', padx=4)
        ttk.Button(btn_bar, text="Cancel", command=dlg.destroy,   width=10).pack(side='right', padx=2)

        dlg.wait_window()
        plt.close(_prev_fig)
        return result[0]

    def _treatment_groups(self, df: pd.DataFrame, metric: str) -> dict:
        """Return {treatment_label: np.array_of_values}."""
        if ('treatment' in df.columns
                and df['treatment'].ne('').any()
                and df['treatment'].notna().any()):
            groups = {}
            for tr, sub in df.groupby('treatment'):
                vals = sub[metric].dropna().values
                if len(vals) > 0:
                    groups[str(tr)] = vals
            if groups:
                return groups
        vals = df[metric].dropna().values
        return {'All sessions': vals} if len(vals) > 0 else {}

    @property
    def _sig_style(self):
        cfg = self._last_graph_cfg or {}
        return cfg.get('sig_style', 'asterisk')

    def _add_stat_annotation(self, ax, groups: dict, y_top: float):
        """Add significance bracket / ANOVA note."""
        keys = list(groups.keys())
        vals = [groups[k] for k in keys]
        if len(vals) < 2:
            return
        paradigm = self._stats_paradigm_var.get()
        use_nonparam = paradigm == 'nonparametric'
        if paradigm == 'auto':
            for v in vals:
                if len(v) >= 3:
                    try:
                        _, sw_p = _sp_stats.shapiro(v)
                        if sw_p < 0.05:
                            use_nonparam = True
                            break
                    except Exception as _sw_err:
                        print(f"Warning: Shapiro-Wilk test failed: {_sw_err}")
        try:
            if len(vals) == 2:
                if use_nonparam:
                    _, p = _sp_stats.mannwhitneyu(vals[0], vals[1], alternative='two-sided')
                else:
                    _, p = _sp_stats.ttest_ind(vals[0], vals[1], equal_var=False)
                label = _p_label(p, self._sig_style)
                if label:
                    _draw_bracket(ax, 0, 1, y_top * 1.05, label)
            else:
                if use_nonparam:
                    _, p = _sp_stats.kruskal(*vals)
                    test_name = 'Kruskal-Wallis'
                else:
                    _, p = _sp_stats.f_oneway(*vals)
                    test_name = 'ANOVA'
                label = _p_label(p, self._sig_style)
                if label:
                    ax.text(0.5, 0.97,
                            f"{test_name}: {label}  (p = {p:.3f})",
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=11, color='darkred')
        except Exception as _stats_err:
            print(f"Warning: statistical test failed in limb use graph: {_stats_err}")

    def _embed_figure(self, frame, fig, ax=None):
        canvas = FigureCanvasTkAgg(fig, master=frame)

        # Pack bottom widgets first (side='bottom') so canvas can expand
        if ax is not None:
            axis_row = ttk.Frame(frame)
            axis_row.pack(side='bottom', fill='x', padx=4, pady=(0, 2))
            ttk.Label(axis_row, text="Y:", font=('Arial', 9)).pack(side='left')
            y_min_e = ttk.Entry(axis_row, width=6)
            y_min_e.pack(side='left', padx=2)
            ttk.Label(axis_row, text="to", font=('Arial', 9)).pack(side='left')
            y_max_e = ttk.Entry(axis_row, width=6)
            y_max_e.pack(side='left', padx=2)
            ttk.Label(axis_row, text="   X:", font=('Arial', 9)).pack(side='left')
            x_min_e = ttk.Entry(axis_row, width=6)
            x_min_e.pack(side='left', padx=2)
            ttk.Label(axis_row, text="to", font=('Arial', 9)).pack(side='left')
            x_max_e = ttk.Entry(axis_row, width=6)
            x_max_e.pack(side='left', padx=2)

            # Store original limits for reset
            _orig_ylim = ax.get_ylim()
            _orig_xlim = ax.get_xlim()

            def _apply_range():
                try:
                    yl = list(ax.get_ylim())
                    if y_min_e.get().strip():
                        yl[0] = float(y_min_e.get())
                    if y_max_e.get().strip():
                        yl[1] = float(y_max_e.get())
                    ax.set_ylim(*yl)
                except ValueError:
                    pass
                try:
                    xl = list(ax.get_xlim())
                    if x_min_e.get().strip():
                        xl[0] = float(x_min_e.get())
                    if x_max_e.get().strip():
                        xl[1] = float(x_max_e.get())
                    ax.set_xlim(*xl)
                except ValueError:
                    pass
                canvas.draw_idle()

            def _reset_range():
                y_min_e.delete(0, 'end')
                y_max_e.delete(0, 'end')
                x_min_e.delete(0, 'end')
                x_max_e.delete(0, 'end')
                ax.set_ylim(*_orig_ylim)
                ax.set_xlim(*_orig_xlim)
                canvas.draw_idle()

            ttk.Button(axis_row, text="Apply", command=_apply_range,
                       width=6).pack(side='left', padx=(6, 2))
            ttk.Button(axis_row, text="Reset", command=_reset_range,
                       width=6).pack(side='left', padx=2)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(canvas, fig)
        _draw_canvas_fit(canvas, fig)

    def _style_ax(self, ax, title='', xlabel='', ylabel=''):
        """Apply publication-quality styling to an axes."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold')

    @staticmethod
    def _calc_error(values, error_type):
        """Compute error bar half-width for a 1-D array of values.

        error_type: 'SEM', 'SD', or '95CI'.
        Returns a single float.
        """
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        n = len(v)
        if n < 2:
            return 0.0
        sd = np.nanstd(v, ddof=1)
        if error_type == 'SD':
            return sd
        sem = sd / np.sqrt(n)
        if error_type == '95CI':
            from scipy.stats import t as _t_dist
            return _t_dist.ppf(0.975, df=n - 1) * sem
        return sem  # default SEM

    @staticmethod
    def _error_label(error_type):
        """Return a display string for the error type (e.g. 'mean \u00b1 SEM')."""
        labels = {'SEM': 'mean \u00b1 SEM', 'SD': 'mean \u00b1 SD',
                  '95CI': 'mean \u00b1 95% CI'}
        return labels.get(error_type, 'mean \u00b1 SEM')

    # ── Metric selector infrastructure ──────────────────────────────────

    def _make_metric_selector(self, parent_nb, category_name,
                              desc_lbl=None, tab_descs=None):
        """Create a category tab with combobox metric selector + graph area.

        Returns a dict with keys:
            'frame'       – the ttk.Frame added to parent_nb
            'combo'       – the ttk.Combobox widget
            'graph_frame' – frame where the selected figure is rendered
            'registry'    – list of metric entry dicts (append via _register_metric)
            'show'        – callable(index) to display a metric by index
            '_current_fig'– mutable list [fig_or_None] for cleanup
        """
        cat_frame = ttk.Frame(parent_nb)
        parent_nb.add(cat_frame, text=category_name)

        # --- selector row ---
        sel_row = ttk.Frame(cat_frame)
        sel_row.pack(side='top', fill='x', padx=6, pady=(4, 2))
        ttk.Label(sel_row, text="Metric:", font=('Arial', 10)).pack(
            side='left', padx=(0, 4))
        combo = ttk.Combobox(sel_row, state='readonly', width=50,
                             font=('Arial', 10))
        combo.pack(side='left', fill='x', expand=True, padx=(0, 6))

        registry = []
        _current_fig = [None]  # track for cleanup

        def _show(idx):
            if idx < 0 or idx >= len(registry):
                return
            combo.current(idx)
            entry = registry[idx]
            # Clear previous figure contents
            if _current_fig[0] is not None:
                try:
                    plt.close(_current_fig[0])
                except Exception:
                    pass
                _current_fig[0] = None
            for w in graph_frame.winfo_children():
                w.destroy()
            # Build the graph
            if entry.get('create_fn'):
                entry['create_fn'](graph_frame)
            else:
                gtype = entry['graph_type']
                if gtype == 'bar':
                    self._build_bar_graph(
                        graph_frame, entry['data'], entry['column'],
                        entry['display_name'], entry.get('reference'),
                        entry.get('y_label', ''), entry.get('graph_cfg'))
                elif gtype == 'box':
                    self._build_box_graph(
                        graph_frame, entry['data'], entry['column'],
                        entry['display_name'], entry.get('reference'),
                        entry.get('y_label', ''), entry.get('graph_cfg'))
                elif gtype == 'violin':
                    self._build_violin_graph(
                        graph_frame, entry['data'], entry['column'],
                        entry['display_name'], entry.get('reference'),
                        entry.get('y_label', ''), entry.get('graph_cfg'))
                elif gtype == 'timecourse':
                    self._build_timecourse_graph(
                        graph_frame, entry['data'], entry['column'],
                        entry['display_name'], entry.get('reference'),
                        entry.get('y_label', ''), entry.get('graph_cfg'))
            # Find embedded figure for cleanup on next switch
            for child in graph_frame.winfo_children():
                fig_obj = getattr(child, 'figure', None)
                if fig_obj is not None:
                    _current_fig[0] = fig_obj
                    break
                for sub in getattr(child, 'winfo_children', lambda: [])():
                    fig_obj = getattr(sub, 'figure', None)
                    if fig_obj is not None:
                        _current_fig[0] = fig_obj
                        break
            # Update description
            if desc_lbl is not None and tab_descs is not None:
                desc_lbl.config(text=entry.get('description', ''))

        def _on_combo(event=None):
            _show(combo.current())

        def _prev():
            idx = combo.current()
            if idx > 0:
                _show(idx - 1)

        def _next():
            idx = combo.current()
            if idx < len(registry) - 1:
                _show(idx + 1)

        combo.bind('<<ComboboxSelected>>', _on_combo)
        ttk.Button(sel_row, text="\u25c0", width=3,
                   command=_prev).pack(side='left', padx=1)
        ttk.Button(sel_row, text="\u25b6", width=3,
                   command=_next).pack(side='left', padx=1)

        # --- graph area ---
        graph_frame = ttk.Frame(cat_frame)
        graph_frame.pack(side='top', fill='both', expand=True)

        return {
            'frame': cat_frame,
            'combo': combo,
            'graph_frame': graph_frame,
            'registry': registry,
            'show': _show,
            '_current_fig': _current_fig,
        }

    def _register_metric(self, selector, display_name, graph_type,
                         data_source, column, reference=None,
                         y_label='', graph_cfg=None, description='',
                         create_fn=None):
        """Register a metric in a selector's registry (no figure created)."""
        entry = {
            'display_name': display_name,
            'graph_type': graph_type,
            'data': data_source,
            'column': column,
            'reference': reference,
            'y_label': y_label,
            'graph_cfg': graph_cfg,
            'description': description,
            'create_fn': create_fn,
        }
        selector['registry'].append(entry)
        # Update combobox values
        selector['combo']['values'] = [
            e['display_name'] for e in selector['registry']]

    # ── Build-graph methods (figure creation into existing frame) ─────

    def _build_bar_graph(self, frame, df, metric, tab_name,
                         reference=None, y_label='', graph_cfg=None):
        """Create a bar chart figure and embed it in the given frame."""
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        groups = self._treatment_groups(df, metric)
        if not groups:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            if graph_cfg and graph_cfg.get('order'):
                ordered_keys = [t for t in graph_cfg['order'] if t in groups]
                ordered_keys += [t for t in groups if t not in ordered_keys]
            else:
                ordered_keys = list(groups.keys())
            treatments = ordered_keys
            vals_list = [groups[t] for t in treatments]
            _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'
            means = [np.nanmean(v) for v in vals_list]
            errs  = [self._calc_error(v, _err_type) for v in vals_list]
            x_pos = np.arange(len(treatments))
            rng = np.random.default_rng(42)
            for xi, (t, vals) in enumerate(zip(treatments, vals_list)):
                if graph_cfg and graph_cfg.get('colors') and t in graph_cfg['colors']:
                    raw = graph_cfg['colors'][t]
                else:
                    raw = 'steelblue'
                if raw == 'white_black':
                    bar_color, edge_color, txt_color = 'white', 'black', 'black'
                else:
                    bar_color, edge_color, txt_color = raw, 'black', 'black'
                ax.bar(xi, means[xi], yerr=errs[xi], capsize=4,
                       color=bar_color, alpha=0.85, edgecolor=edge_color,
                       linewidth=0.9)
                jitter = rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(xi + jitter, vals, color=txt_color, s=30,
                           zorder=5, alpha=0.8)
            if reference is not None:
                ax.axhline(reference, color='crimson', linestyle='--',
                           linewidth=1.2, alpha=0.7,
                           label=f'Reference = {reference}')
                ax.legend(fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(treatments, fontsize=11)
            self._style_ax(ax, title=tab_name, ylabel=y_label or metric)
            y_top = max(means) if means else 0
            if self._enable_stats_var.get():
                self._add_stat_annotation(
                    ax, {t: groups[t] for t in treatments}, y_top)
        self._add_export_buttons(frame, fig, groups, metric, tab_name,
                                 graph_cfg=graph_cfg, data_type='summary')
        self._embed_figure(frame, fig, ax=ax)

    def _build_box_graph(self, frame, df, metric, tab_name,
                         reference=None, y_label='', graph_cfg=None):
        """Create a box plot figure and embed it in the given frame."""
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        groups = self._treatment_groups(df, metric)
        if not groups:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            if graph_cfg and graph_cfg.get('order'):
                ordered_keys = [t for t in graph_cfg['order'] if t in groups]
                ordered_keys += [t for t in groups if t not in ordered_keys]
            else:
                ordered_keys = list(groups.keys())
            treatments = ordered_keys
            data = [groups[t] for t in treatments]
            bp_dict = ax.boxplot(data, labels=treatments, patch_artist=True,
                                 medianprops=dict(color='black', linewidth=2))
            fallback_colors = plt.cm.Set2.colors
            for i, (patch, t) in enumerate(zip(bp_dict['boxes'], treatments)):
                if (graph_cfg and graph_cfg.get('colors')
                        and t in graph_cfg['colors']):
                    raw = graph_cfg['colors'][t]
                    fc = 'white' if raw == 'white_black' else raw
                    patch.set_facecolor(fc)
                else:
                    patch.set_facecolor(
                        fallback_colors[i % len(fallback_colors)])
                patch.set_alpha(0.7)
            rng = np.random.default_rng(42)
            for xi, vals in enumerate(data, 1):
                jitter = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(xi + jitter, vals, color='black', s=28,
                           zorder=5, alpha=0.8)
            if reference is not None:
                ax.axhline(reference, color='crimson', linestyle='--',
                           linewidth=1.2, alpha=0.7)
            self._style_ax(ax, title=tab_name, ylabel=y_label or metric)
            y_top = max(np.nanmax(v) for v in data if len(v) > 0)
            if self._enable_stats_var.get():
                self._add_stat_annotation(ax, groups, y_top)
        self._add_export_buttons(frame, fig, groups, metric, tab_name,
                                 graph_cfg=graph_cfg, data_type='summary')
        self._embed_figure(frame, fig, ax=ax)

    def _build_violin_graph(self, frame, df, metric, tab_name,
                            reference=None, y_label='', graph_cfg=None):
        """Create a violin plot figure and embed it in the given frame."""
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        groups = self._treatment_groups(df, metric)
        if not groups:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            if graph_cfg and graph_cfg.get('order'):
                ordered_keys = [t for t in graph_cfg['order'] if t in groups]
                ordered_keys += [t for t in groups if t not in ordered_keys]
            else:
                ordered_keys = list(groups.keys())
            treatments = ordered_keys
            data = [groups[t] for t in treatments]
            parts = ax.violinplot(data, positions=range(len(treatments)),
                                  showmeans=True, showmedians=True,
                                  showextrema=False)
            for i, (pc, t) in enumerate(zip(parts['bodies'], treatments)):
                if (graph_cfg and graph_cfg.get('colors')
                        and t in graph_cfg['colors']):
                    raw = graph_cfg['colors'][t]
                    fc = 'white' if raw == 'white_black' else raw
                else:
                    fc = plt.cm.Set2.colors[i % len(plt.cm.Set2.colors)]
                pc.set_facecolor(fc)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
            rng = np.random.default_rng(42)
            for xi, vals in enumerate(data):
                jitter = rng.uniform(-0.1, 0.1, len(vals))
                ax.scatter(xi + jitter, vals, color='black', s=28,
                           zorder=5, alpha=0.8)
            if reference is not None:
                ax.axhline(reference, color='crimson', linestyle='--',
                           linewidth=1.2, alpha=0.7)
            ax.set_xticks(range(len(treatments)))
            ax.set_xticklabels(treatments, fontsize=11)
            self._style_ax(ax, title=tab_name, ylabel=y_label or metric)
            y_top = max(np.nanmax(v) for v in data if len(v) > 0)
            if self._enable_stats_var.get():
                self._add_stat_annotation(
                    ax, {t: groups[t] for t in treatments}, y_top)
        self._add_export_buttons(frame, fig, groups, metric, tab_name,
                                 graph_cfg=graph_cfg, data_type='summary')
        self._embed_figure(frame, fig, ax=ax)

    def _build_timecourse_graph(self, frame, bins_df, metric, tab_name,
                                reference=None, y_label=None,
                                graph_cfg=None):
        """Create a timecourse figure and embed it in the given frame."""
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        if reference is None:
            if 'SI' in metric or 'symmetry' in metric.lower():
                reference = 0.0
            elif 'WBI' in metric or 'SBI' in metric:
                reference = 50.0
        if reference is not None:
            ax.axhline(reference, color='crimson', linestyle='--',
                       linewidth=1.1, alpha=0.6)

        _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

        # Filter bins to the user-specified time window so lines don't
        # extend beyond the requested range.
        if graph_cfg and graph_cfg.get('time_window') is not None:
            _tw_sec = graph_cfg['time_window'] * 60.0
            bins_df = bins_df[bins_df['bin_start_s'] <= _tw_sec]

        if ('treatment' in bins_df.columns
                and bins_df['treatment'].ne('').any()):
            all_treats = bins_df['treatment'].dropna().unique().tolist()
            if graph_cfg and graph_cfg.get('order'):
                ordered = [t for t in graph_cfg['order'] if t in all_treats]
                ordered += [t for t in all_treats if t not in ordered]
            else:
                ordered = all_treats

            for treatment in ordered:
                grp = bins_df[bins_df['treatment'] == treatment]
                if grp.empty:
                    continue
                tg = grp.groupby('bin_start_s')[metric]
                tmean = tg.mean()
                if _err_type == 'SD':
                    terr = tg.std(ddof=1).fillna(0)
                elif _err_type == '95CI':
                    terr = tg.apply(
                        lambda x: self._calc_error(x.values, '95CI')).fillna(0)
                else:
                    terr = tg.sem().fillna(0)
                t_min = tmean.index.values / 60.0

                raw_color = None
                if graph_cfg and graph_cfg.get('colors'):
                    raw_color = graph_cfg['colors'].get(str(treatment))
                line_color = ('black' if raw_color == 'white_black'
                              else (raw_color if raw_color else None))
                _tkey = str(treatment)
                _lstyle = (graph_cfg.get('line_styles', {}).get(_tkey, '-')
                           if graph_cfg else '-')
                _mshape = (graph_cfg.get('marker_shapes', {}).get(_tkey, 'o')
                           if graph_cfg else 'o')
                _msize = (graph_cfg.get('marker_sizes', {}).get(_tkey, 5)
                          if graph_cfg else 5)
                _lw = (graph_cfg.get('line_widths', {}).get(_tkey, 1.8)
                       if graph_cfg else 1.8)
                _alpha = (graph_cfg.get('opacities', {}).get(_tkey, 1.0)
                          if graph_cfg else 1.0)
                if not _mshape or _mshape == 'none':
                    _mshape_g = (graph_cfg.get('marker_shape', 'none')
                                 if graph_cfg else 'none')
                    if _mshape_g and _mshape_g != 'none':
                        _mshape = _mshape_g
                        _msize = (graph_cfg.get('marker_size', 5)
                                  if graph_cfg else 5)
                _mfill = (graph_cfg.get('marker_fills', {}).get(_tkey, 'full')
                          if graph_cfg else 'full')
                plot_kw = dict(label=_tkey, linewidth=_lw,
                               linestyle=_lstyle, alpha=_alpha)
                if _mshape and _mshape != 'none':
                    plot_kw['marker'] = _mshape
                    plot_kw['markersize'] = _msize
                    plot_kw['fillstyle'] = _mfill
                    if _mfill == 'none':
                        plot_kw['markeredgewidth'] = 1.2
                    elif _mfill in ('left', 'right', 'top', 'bottom'):
                        plot_kw['markerfacecoloralt'] = 'white'
                if line_color:
                    plot_kw['color'] = line_color
                if raw_color == 'white_black':
                    plot_kw['markerfacecolor'] = 'white'
                    plot_kw['markeredgecolor'] = 'black'
                _edge_sel = graph_cfg.get('marker_edge_colors', {}).get(_tkey, 'auto') if graph_cfg else 'auto'
                if _edge_sel == 'black':
                    plot_kw['markeredgecolor'] = 'black'
                elif _edge_sel == 'white':
                    plot_kw['markeredgecolor'] = 'white'
                elif _edge_sel == 'match':
                    plot_kw['markeredgecolor'] = line_color or plot_kw.get('color', 'black')
                rebin = graph_cfg.get('rebin_minutes') if graph_cfg else None
                if rebin and rebin > 0:
                    t_min, _means, _errs = GaitLimbTab._rebin_timecourse(
                        list(t_min), list(tmean.values),
                        list(terr.values), rebin)
                    t_min = np.array(t_min)
                    tmean_v = np.array(_means)
                    terr_v = np.array(_errs)
                else:
                    tmean_v = tmean.values
                    terr_v = terr.values
                _err_disp = (graph_cfg.get('error_display', 'circles_caps')
                             if graph_cfg else 'circles_caps')
                # When circles_caps, draw line without markers (markers drawn with errorbar)
                if _err_disp == 'circles_caps':
                    _line_kw = {k: v for k, v in plot_kw.items()
                                if k not in ('marker', 'markersize', 'fillstyle',
                                             'markeredgewidth', 'markerfacecoloralt',
                                             'markerfacecolor', 'markeredgecolor')}
                    ax.plot(t_min, tmean_v, **_line_kw)
                else:
                    ax.plot(t_min, tmean_v, **plot_kw)
                _ec = line_color or plot_kw.get('color')
                if _err_disp == 'caps':
                    ax.errorbar(t_min, tmean_v, yerr=terr_v, capsize=4,
                                capthick=1.5, fmt='none', alpha=0.6,
                                ecolor=_ec)
                elif _err_disp == 'circles_caps':
                    ax.errorbar(t_min, tmean_v, yerr=terr_v, capsize=4,
                                capthick=1.5, fmt='none', alpha=0.6 * _alpha,
                                ecolor=_ec)
                    _msz = _msize if (_mshape and _mshape != 'none') else 5
                    _mkw = dict(linestyle='none', color=_ec,
                                marker='o', markersize=_msz, alpha=_alpha)
                    _mkw['fillstyle'] = _mfill
                    if _mfill == 'none':
                        _mkw['markeredgewidth'] = 1.2
                    elif _mfill in ('left', 'right', 'top', 'bottom'):
                        _mkw['markerfacecoloralt'] = 'white'
                    if raw_color == 'white_black':
                        _mkw['markerfacecolor'] = 'white'
                        _mkw['markeredgecolor'] = 'black'
                    if _edge_sel == 'black':
                        _mkw['markeredgecolor'] = 'black'
                    elif _edge_sel == 'white':
                        _mkw['markeredgecolor'] = 'white'
                    elif _edge_sel == 'match':
                        _mkw['markeredgecolor'] = _ec or 'black'
                    ax.plot(t_min, tmean_v, **_mkw)
                else:
                    ax.fill_between(
                        t_min, tmean_v - terr_v, tmean_v + terr_v,
                        alpha=0.18,
                        **(dict(color=line_color) if line_color else {}))
                if graph_cfg and graph_cfg.get('show_individual'):
                    for subj in grp['subject'].unique():
                        subj_data = grp[grp['subject'] == subj]
                        sg = subj_data.groupby('bin_start_s')[metric].mean()
                        s_min = sg.index.values / 60.0
                        ax.plot(s_min, sg.values, alpha=0.3, linewidth=0.8,
                                **(dict(color=line_color)
                                   if line_color else {}))
            ax.legend(fontsize=10)

            if self._enable_stats_var.get() and len(ordered) >= 2:
                from matplotlib.transforms import blended_transform_factory
                import warnings as _warnings
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                tw = graph_cfg.get('time_window') if graph_cfg else None
                alpha = self._stats_alpha_var.get()

                # Step 1: Two-way ANOVA gatekeeper
                _do_posthoc = False
                _anova_text = ''
                try:
                    import statsmodels.api as _sm
                    from statsmodels.formula.api import ols as _ols
                    _adf = bins_df[['subject', 'treatment', 'bin_start_s', metric]].dropna().copy()
                    _adf['treatment'] = _adf['treatment'].astype('category')
                    _adf['bin_start_s'] = _adf['bin_start_s'].astype('category')
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings('ignore', module='statsmodels')
                        _model = _ols(f'{metric} ~ C(treatment) + C(bin_start_s) + C(treatment):C(bin_start_s)',
                                      data=_adf).fit()
                        _at = _sm.stats.anova_lm(_model, typ=2)
                    _treat_p = _at.loc['C(treatment)', 'PR(>F)']
                    _inter_p = _at.loc['C(treatment):C(bin_start_s)', 'PR(>F)']
                    if _inter_p < alpha or _treat_p < alpha:
                        _do_posthoc = True
                    def _pfmt(p):
                        if p < 0.001: return 'p<0.001***'
                        if p < 0.01:  return f'p={p:.3f}**'
                        if p < alpha: return f'p={p:.3f}*'
                        return f'p={p:.3f} ns'
                    _anova_text = f"Two-way ANOVA: Treatment {_pfmt(_treat_p)}, Interaction {_pfmt(_inter_p)}"
                except Exception:
                    _do_posthoc = True

                # Step 2: Per-bin post-hoc (only if ANOVA is significant)
                if _do_posthoc:
                    _rebin_min = graph_cfg.get('rebin_minutes') if graph_cfg else None
                    if _rebin_min and _rebin_min > 0:
                        _rebin_s = _rebin_min * 60.0
                        _all_bins = sorted(bins_df['bin_start_s'].unique())
                        _rebin_windows = {}
                        for bs in _all_bins:
                            win_start = int(bs // _rebin_s) * _rebin_s
                            _rebin_windows.setdefault(win_start, []).append(bs)
                        _bin_iter = [
                            (win_start / 60.0,
                             bins_df[bins_df['bin_start_s'].isin(win_bins)])
                            for win_start, win_bins in sorted(_rebin_windows.items())
                        ]
                    else:
                        _bin_iter = [
                            (bin_t / 60.0, bgrp)
                            for bin_t, bgrp in bins_df.groupby('bin_start_s')
                        ]

                    paradigm = self._stats_paradigm_var.get()
                    for t_min_val, bin_data in _bin_iter:
                        if tw is not None and t_min_val > tw:
                            continue
                        gvals = []
                        for t in ordered:
                            if _rebin_min and _rebin_min > 0:
                                t_data = bin_data[bin_data['treatment'] == t]
                                vals = t_data.groupby('subject')[metric].mean().dropna().values
                            else:
                                vals = bin_data[bin_data['treatment'] == t][metric].dropna().values
                            if len(vals) >= 2:
                                gvals.append(vals)
                        if len(gvals) < 2:
                            continue

                        use_nonparam = paradigm == 'nonparametric'
                        if paradigm == 'auto':
                            for g in gvals:
                                if len(g) >= 3:
                                    try:
                                        _, sw_p = _sp_stats.shapiro(g)
                                        if sw_p < 0.05:
                                            use_nonparam = True
                                            break
                                    except Exception:
                                        pass

                        try:
                            if len(gvals) == 2:
                                if use_nonparam:
                                    _, p = _sp_stats.mannwhitneyu(gvals[0], gvals[1], alternative='two-sided')
                                else:
                                    _, p = _sp_stats.ttest_ind(gvals[0], gvals[1], equal_var=False)
                            else:
                                if use_nonparam:
                                    _, p = _sp_stats.kruskal(*gvals)
                                else:
                                    _, p = _sp_stats.f_oneway(*gvals)
                            lbl = _p_label(p, self._sig_style)
                            if lbl:
                                ax.text(t_min_val, 0.98, lbl, transform=trans,
                                        ha='center', va='top', fontsize=9, color='black')
                        except Exception:
                            pass

                # Show ANOVA annotation
                if _anova_text:
                    ax.text(0.02, 0.98, _anova_text, transform=ax.transAxes,
                            va='top', ha='left', fontsize=7, family='monospace',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
        else:
            tg = bins_df.groupby('bin_start_s')[metric]
            tmean = tg.mean()
            t_min = tmean.index.values / 60.0
            rebin = graph_cfg.get('rebin_minutes') if graph_cfg else None
            if rebin and rebin > 0:
                t_min, _means, _ = GaitLimbTab._rebin_timecourse(
                    list(t_min), list(tmean.values),
                    [0] * len(t_min), rebin)
                t_min = np.array(t_min)
                tmean_v = np.array(_means)
            else:
                tmean_v = tmean.values
            _st_plot_kw = dict(linewidth=1.8, color='steelblue')
            _ls_d = graph_cfg.get('line_styles', {}) if graph_cfg else {}
            if _ls_d:
                _ft = next(iter(_ls_d))
                _st_plot_kw['linestyle'] = _ls_d.get(_ft, '-')
                _st_plot_kw['linewidth'] = (
                    graph_cfg.get('line_widths', {}).get(_ft, 1.8))
                _st_plot_kw['alpha'] = (
                    graph_cfg.get('opacities', {}).get(_ft, 1.0))
                _mk = graph_cfg.get('marker_shapes', {}).get(_ft, 'o')
                if _mk and _mk != 'none':
                    _st_plot_kw['marker'] = _mk
                    _st_plot_kw['markersize'] = (
                        graph_cfg.get('marker_sizes', {}).get(_ft, 5))
                    _mf = graph_cfg.get('marker_fills', {}).get(_ft, 'full')
                    _st_plot_kw['fillstyle'] = _mf
                    if _mf == 'none':
                        _st_plot_kw['markeredgewidth'] = 1.2
                    elif _mf in ('left', 'right', 'top', 'bottom'):
                        _st_plot_kw['markerfacecoloralt'] = 'white'
                _me = graph_cfg.get('marker_edge_colors', {}).get(_ft, 'auto') if graph_cfg else 'auto'
                if _me == 'black':
                    _st_plot_kw['markeredgecolor'] = 'black'
                elif _me == 'white':
                    _st_plot_kw['markeredgecolor'] = 'white'
                elif _me == 'match':
                    _st_plot_kw['markeredgecolor'] = _st_plot_kw.get('color', 'steelblue')
            else:
                _mshape = (graph_cfg.get('marker_shape', 'o')
                           if graph_cfg else 'o')
                if _mshape and _mshape != 'none':
                    _st_plot_kw['marker'] = _mshape
                    _st_plot_kw['markersize'] = (
                        graph_cfg.get('marker_size', 5))
            ax.plot(t_min, tmean_v, **_st_plot_kw)

        if graph_cfg and graph_cfg.get('time_window') is not None:
            ax.set_xlim(-0.5, graph_cfg['time_window'] + 0.5)
        else:
            ax.margins(x=0.02)

        self._style_ax(ax, title=tab_name,
                        xlabel='Time (min)',
                        ylabel=y_label or metric.replace('_', ' '))
        # Annotate error type on the plot
        ax.annotate(self._error_label(_err_type),
                    xy=(1, 1), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8,
                    color='gray', style='italic')

        self._add_export_buttons(frame, fig, None, metric, tab_name,
                                 graph_cfg=graph_cfg, data_type='timecourse',
                                 bins_df=bins_df)
        self._embed_figure(frame, fig, ax=ax)

    def _add_export_buttons(self, frame, fig, groups, metric, tab_name,
                            graph_cfg=None, data_type='summary',
                            bins_df=None):
        """Add Export Graph / Export Data buttons to a frame."""
        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        def _exp_graph(f=fig, n=tab_name):
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'),
                           ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile=n.replace(' ', '_') + '.png',
                parent=frame.winfo_toplevel())
            if path:
                f.savefig(path, dpi=300, bbox_inches='tight')

        def _exp_data():
            from tkinter import filedialog
            if data_type == 'timecourse' and bins_df is not None:
                cols = [c for c in ['treatment', 'bin_start_s', metric]
                        if c in bins_df.columns]
                out = bins_df[cols].copy()
                if 'bin_start_s' in out.columns:
                    out.insert(
                        out.columns.get_loc('bin_start_s') + 1,
                        'bin_start_min', out['bin_start_s'] / 60.0)
            elif groups:
                gc = graph_cfg or {}
                ordered_keys = (
                    [t for t in gc.get('order', []) if t in groups]
                    + [t for t in groups if t not in gc.get('order', [])])
                rows = [{'treatment': t, metric: float(v)}
                        for t in (ordered_keys or list(groups.keys()))
                        for v in groups[t]]
                out = pd.DataFrame(rows)
            else:
                return
            path = filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                initialfile=tab_name.replace(' ', '_') + '_data.csv',
                parent=frame.winfo_toplevel())
            if path:
                out.to_csv(path, index=False)

        ttk.Button(btn_bar, text="Export Graph",
                   command=_exp_graph).pack(side='left', padx=4)
        ttk.Button(btn_bar, text="Export Data",
                   command=_exp_data).pack(side='left', padx=2)

    # ── Original tab-based graph methods (now delegate to build methods) ─

    def _add_bar_tab(self, nb, df, metric, tab_name,
                     reference=None, y_label='', graph_cfg=None):
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)
        self._build_bar_graph(frame, df, metric, tab_name,
                              reference, y_label, graph_cfg)

    def _add_box_tab(self, nb, df, metric, tab_name,
                     reference=None, y_label='', graph_cfg=None):
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)
        self._build_box_graph(frame, df, metric, tab_name,
                              reference, y_label, graph_cfg)

    def _add_violin_tab(self, nb, df, metric, tab_name,
                        reference=None, y_label='', graph_cfg=None):
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)
        self._build_violin_graph(frame, df, metric, tab_name,
                                 reference, y_label, graph_cfg)

    def _add_timecourse_tab(self, nb, bins_df, metric, tab_name,
                            reference=None, y_label=None, graph_cfg=None):
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)
        self._build_timecourse_graph(frame, bins_df, metric, tab_name,
                                     reference, y_label, graph_cfg)
    def _perform_wb_statistical_test(self, data_by_treatment, treatments):
        """Statistical test with parametric/non-parametric support and effect sizes."""
        from scipy import stats as _scipy_stats

        if not self._enable_stats_var.get():
            return None

        groups = [data_by_treatment.get(t, np.array([])) for t in treatments]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return None

        alpha     = self._stats_alpha_var.get()
        test_type = self._stats_test_var.get()
        paradigm  = self._stats_paradigm_var.get()

        # Determine parametric vs non-parametric
        use_nonparam = False
        if test_type == 'nonparametric' or paradigm == 'nonparametric':
            use_nonparam = True
        elif paradigm == 'auto':
            # Shapiro-Wilk on each group
            for g in groups:
                if len(g) >= 3:
                    try:
                        _, sw_p = _scipy_stats.shapiro(g)
                        if sw_p < 0.05:
                            use_nonparam = True
                            break
                    except Exception as _sw_err:
                        print(f"Warning: Shapiro-Wilk normality test failed: {_sw_err}")

        # Auto-select test based on group count
        if test_type == 'auto' or test_type == 'nonparametric':
            test_type = 't-test' if len(groups) == 2 else 'ANOVA'

        results = {'alpha': alpha}

        if test_type == 't-test' and len(groups) == 2:
            if use_nonparam:
                stat, p_val = _scipy_stats.mannwhitneyu(groups[0], groups[1],
                                                         alternative='two-sided')
                results['test_type'] = 'Mann-Whitney U'
            else:
                stat, p_val = _scipy_stats.ttest_ind(groups[0], groups[1],
                                                      equal_var=False)
                results['test_type'] = "Welch's t-test"
            results['p_value']     = float(p_val)
            results['significant'] = bool(p_val < alpha)
            results['comparison']  = f"{treatments[0]} vs {treatments[1]}"
            # Cohen's d
            n1, n2 = len(groups[0]), len(groups[1])
            if n1 > 1 and n2 > 1:
                m1, m2 = np.mean(groups[0]), np.mean(groups[1])
                s1, s2 = np.std(groups[0], ddof=1), np.std(groups[1], ddof=1)
                pooled_s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                if pooled_s > 0:
                    results['effect_size'] = float(abs(m1 - m2) / pooled_s)
                    results['effect_size_type'] = "Cohen's d"

        else:  # ANOVA / Kruskal-Wallis
            if use_nonparam:
                stat, p_val = _scipy_stats.kruskal(*groups)
                results['test_type'] = 'Kruskal-Wallis'
            else:
                stat, p_val = _scipy_stats.f_oneway(*groups)
                results['test_type'] = 'ANOVA'
            results['p_value']     = float(p_val)
            results['significant'] = bool(p_val < alpha)

            # Eta-squared
            if not use_nonparam:
                grand_mean = np.mean(np.concatenate(groups))
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
                if ss_total > 0:
                    results['effect_size'] = float(ss_between / ss_total)
                    results['effect_size_type'] = 'eta-squared'

            if p_val < alpha:
                pairwise = {}
                valid_treats = [t for t in treatments
                                if len(data_by_treatment.get(t, [])) > 0]
                valid_groups = [data_by_treatment[t] for t in valid_treats]
                n_comparisons = len(valid_treats) * (len(valid_treats) - 1) // 2
                for i in range(len(valid_treats)):
                    for j in range(i + 1, len(valid_treats)):
                        if use_nonparam:
                            _, pp = _scipy_stats.mannwhitneyu(
                                valid_groups[i], valid_groups[j],
                                alternative='two-sided')
                        else:
                            _, pp = _scipy_stats.ttest_ind(
                                valid_groups[i], valid_groups[j],
                                equal_var=False)
                        pp_adj = min(float(pp) * n_comparisons, 1.0)
                        key = f"{valid_treats[i]}_vs_{valid_treats[j]}"
                        pairwise[key] = {
                            'p_value':     pp_adj,
                            'significant': bool(pp_adj < alpha),
                        }
                results['pairwise'] = pairwise

        return results

    def _add_paw_contact_bar_tab(self, nb, df, graph_cfg=None,
                                 as_frame=False):
        """Grouped bar chart: per-paw contact%, one bar group per treatment."""
        _PAW_COLORS = {
            'contact_pct_HL': '#2ca02c',
            'contact_pct_HR': '#d62728',
            'contact_pct_FL': '#bcbd22',
            'contact_pct_FR': '#ff7f0e',
        }
        paw_roles = ['HL', 'HR', 'FL', 'FR']
        paw_cols  = [f'contact_pct_{r}' for r in paw_roles]
        active    = [(r, c) for r, c in zip(paw_roles, paw_cols)
                     if c in df.columns and df[c].notna().any()]
        if not active:
            return

        has_treats = ('treatment' in df.columns
                      and df['treatment'].ne('').any()
                      and df['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in df['treatment'].dropna().unique() if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']
        n_treats = len(treatment_labels)

        _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

        if n_treats > 1:
            colors = list(plt.cm.Set2(np.linspace(0, 0.8, n_treats)))
        else:
            colors = [_PAW_COLORS.get(active[0][1], '#1f77b4')]

        if as_frame:
            frame = nb  # nb is already the target frame
        else:
            frame = ttk.Frame(nb)
            nb.add(frame, text='Contact %')

        fig, ax = plt.subplots(
            figsize=(max(5, len(active) * 1.4 + 1), 4), constrained_layout=True)
        x     = np.arange(len(active))
        bar_w = 0.7 / n_treats
        rng   = np.random.default_rng(42)

        for ti, treat in enumerate(treatment_labels):
            xpos = x - 0.35 + (ti + 0.5) * bar_w
            for pi, (role, col) in enumerate(active):
                if has_treats and n_treats > 1:
                    subset = df[df['treatment'] == treat][col].dropna()
                else:
                    subset = df[col].dropna()
                mean_ = subset.mean() if len(subset) else 0
                err_  = self._calc_error(subset.values, _err_type)
                # Color: from graph_cfg if available, else fallback
                if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
                    raw = graph_cfg['colors'][treat]
                    color = 'white' if raw == 'white_black' else raw
                    edge  = 'black' if raw == 'white_black' else 'black'
                elif n_treats > 1:
                    color = colors[ti]
                    edge = 'black'
                else:
                    color = _PAW_COLORS.get(col, colors[0])
                    edge = 'black'
                ax.bar(xpos[pi], mean_, bar_w * 0.9,
                       color=color, edgecolor=edge, yerr=err_, capsize=4,
                       label=treat if pi == 0 else None)
                jx = xpos[pi] + rng.uniform(-bar_w * 0.3, bar_w * 0.3, len(subset))
                ax.scatter(jx, subset.values, color=color, s=22, alpha=0.6,
                           zorder=3, edgecolors='black', linewidths=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([r for r, _ in active], fontsize=11)
        self._style_ax(ax, title='Per-paw contact', ylabel='Contact %')
        if n_treats > 1:
            ax.legend(fontsize=10)

        # Per-paw significance annotation
        if self._enable_stats_var.get():
            if n_treats == 2:
                for pi, (role, col) in enumerate(active):
                    v0 = (df[df['treatment'] == treatment_labels[0]][col].dropna().values
                          if has_treats else df[col].dropna().values)
                    v1 = (df[df['treatment'] == treatment_labels[1]][col].dropna().values
                          if has_treats else np.array([]))
                    if len(v0) > 0 and len(v1) > 0:
                        try:
                            _, p = _sp_stats.ttest_ind(v0, v1, equal_var=False)
                            lbl = _p_label(p, self._sig_style)
                            if lbl:
                                x0 = x[pi] - 0.35 + 0.5 * bar_w
                                x1 = x[pi] - 0.35 + 1.5 * bar_w
                                y_ann = ax.get_ylim()[1] * 0.96
                                ax.plot([x0, x0, x1, x1],
                                        [y_ann * 0.97, y_ann, y_ann, y_ann * 0.97],
                                        color='black', linewidth=1)
                                ax.text((x0 + x1) / 2, y_ann * 1.01, lbl,
                                        ha='center', va='bottom',
                                        fontsize=11, fontweight='bold')
                        except Exception as _stats_err:
                            print(f"Warning: pairwise stats failed: {_stats_err}")
            elif n_treats > 2:
                for pi, (role, col) in enumerate(active):
                    grp_vals = [df[df['treatment'] == t][col].dropna().values
                                for t in treatment_labels]
                    grp_vals = [v for v in grp_vals if len(v) > 0]
                    if len(grp_vals) > 1:
                        try:
                            _, p = _sp_stats.f_oneway(*grp_vals)
                            lbl = _p_label(p, self._sig_style)
                            if lbl:
                                ax.text(x[pi], ax.get_ylim()[1] * 0.96, lbl,
                                        ha='center', va='bottom',
                                        fontsize=11, fontweight='bold')
                        except Exception as _stats_err:
                            print(f"Warning: ANOVA stats failed: {_stats_err}")

        # -- export button bar --
        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        def _exp_graph(f=fig):
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile='Contact_pct.png',
                parent=frame.winfo_toplevel())
            if path:
                f.savefig(path, dpi=300, bbox_inches='tight')

        def _exp_data(d=df, act=active, ht=has_treats):
            from tkinter import filedialog
            paw_cols_active = [col for _, col in act]
            export_cols = (['treatment'] + paw_cols_active if ht else paw_cols_active)
            out = d[[c for c in export_cols if c in d.columns]].copy()
            path = filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                initialfile='Contact_pct_data.csv',
                parent=frame.winfo_toplevel())
            if path:
                out.to_csv(path, index=False)

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
        ttk.Button(btn_bar, text="Export Data",  command=_exp_data).pack(side='left', padx=2)
        self._embed_figure(frame, fig, ax=ax)

    def _add_support_pattern_tab(self, nb, df, graph_cfg=None,
                                 as_frame=False):
        """Stacked bar chart: support pattern distribution per treatment."""
        support_cols = [f'support_{n}paw_pct' for n in range(5)]
        active = [c for c in support_cols if c in df.columns and df[c].notna().any()]
        if not active:
            return

        has_treats = ('treatment' in df.columns
                      and df['treatment'].ne('').any()
                      and df['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in df['treatment'].dropna().unique() if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']

        if as_frame:
            frame = nb
        else:
            frame = ttk.Frame(nb)
            nb.add(frame, text='Support Patterns')

        fig, ax = plt.subplots(figsize=(max(5, len(treatment_labels) * 1.5 + 1), 4),
                               constrained_layout=True)
        x = np.arange(len(treatment_labels))
        _support_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        _support_labels = ['0 paw', '1 paw', '2 paw', '3 paw', '4 paw']

        bottom = np.zeros(len(treatment_labels))
        for i, col in enumerate(support_cols):
            if col not in df.columns:
                continue
            means = []
            for treat in treatment_labels:
                if has_treats and len(treatment_labels) > 1:
                    vals = df[df['treatment'] == treat][col].dropna().values
                else:
                    vals = df[col].dropna().values
                means.append(np.nanmean(vals) if len(vals) > 0 else 0)
            means = np.array(means)
            ax.bar(x, means, bottom=bottom, color=_support_colors[i],
                   label=_support_labels[i], edgecolor='black', linewidth=0.5)
            bottom += means

        ax.set_xticks(x)
        ax.set_xticklabels(treatment_labels, fontsize=11)
        self._style_ax(ax, title='Support Patterns (locomotion)',
                       ylabel='% of locomotion frames')
        ax.legend(fontsize=9, loc='upper right')

        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        def _exp_graph(f=fig):
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile='Support_patterns.png',
                parent=frame.winfo_toplevel())
            if path:
                f.savefig(path, dpi=300, bbox_inches='tight')

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
        self._embed_figure(frame, fig, ax=ax)

    # ── Contour grouped graphs ──────────────────────────────────────────

    _PAW_COLORS_ROLE = {
        'HL': '#2ca02c', 'HR': '#d62728',
        'FL': '#bcbd22', 'FR': '#ff7f0e',
    }

    @staticmethod
    def _resample_contour(pts, n_points=64):
        """Resample a contour to a fixed number of evenly spaced points.

        pts: (M, 2) array of contour coordinates.
        Returns: (n_points, 2) array.
        """
        if len(pts) < 3:
            return None
        # Close the contour
        closed = np.vstack([pts, pts[0:1]])
        # Cumulative arc length
        diffs = np.diff(closed, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        if total_len <= 0:
            return None
        # Evenly spaced parameter values (exclude endpoint to avoid duplicate)
        t_new = np.linspace(0, total_len, n_points, endpoint=False)
        # Interpolate x and y
        x_new = np.interp(t_new, cum_len, closed[:, 0])
        y_new = np.interp(t_new, cum_len, closed[:, 1])
        return np.column_stack([x_new, y_new])

    @staticmethod
    def _normalize_contour(pts, area):
        """Center contour at origin and normalize by sqrt(area)."""
        if pts is None or area <= 0:
            return None
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        scale = np.sqrt(area)
        if scale > 0:
            centered = centered / scale
        return centered

    @staticmethod
    def _shape_metrics(pts):
        """Compute aspect_ratio and circularity from (N,2) contour points."""
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        dim_max, dim_min = max(w, h), min(w, h)
        ar = dim_max / dim_min if dim_min > 0 else 999.0
        # Perimeter (sum of segment lengths)
        diffs = np.diff(np.vstack([pts, pts[0:1]]), axis=0)
        perimeter = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))
        # Area (shoelace)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        return ar, circ

    @staticmethod
    def _shape_metrics_batch(stacked):
        """Vectorized aspect_ratio + circularity for (N, 64, 2) contour array."""
        mins = stacked.min(axis=1)
        maxs = stacked.max(axis=1)
        wh = maxs - mins
        dim_max = wh.max(axis=1)
        dim_min = wh.min(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ar = np.where(dim_min > 0, dim_max / dim_min, 999.0)
        closed = np.concatenate([stacked, stacked[:, 0:1, :]], axis=1)
        diffs = np.diff(closed, axis=1)
        perimeter = np.sqrt((diffs ** 2).sum(axis=2)).sum(axis=1)
        x, y = stacked[:, :, 0], stacked[:, :, 1]
        area = 0.5 * np.abs(
            (x * np.roll(y, -1, axis=1)).sum(axis=1) -
            (y * np.roll(x, -1, axis=1)).sum(axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            circ = np.where(perimeter > 0,
                            (4 * np.pi * area) / (perimeter ** 2), 0.0)
        return ar, circ

    def _add_contour_shape_tab(self, nb, sessions_df, intermediates,
                               role, tab_name='Shape', graph_cfg=None,
                               filter_paw=False, as_frame=False):
        """Add a tab showing mean contour outline for a paw, per treatment group."""
        # Gather normalized contour shapes from intermediates
        # Group by treatment
        has_treats = ('treatment' in sessions_df.columns
                      and sessions_df['treatment'].ne('').any()
                      and sessions_df['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in sessions_df['treatment'].dropna().unique()
                          if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']

        # Collect shapes (and solidities for filtering) per treatment
        treat_shapes = {t: [] for t in treatment_labels}
        treat_sols = {t: [] for t in treatment_labels}

        for _, row in sessions_df.iterrows():
            sess_name = row.get('session', '')
            treat = str(row.get('treatment', '')) if has_treats else 'All sessions'
            if treat not in treat_shapes:
                continue
            interm = intermediates.get(sess_name, {})
            pcd = interm.get('paw_contour_data', {})
            role_data = pcd.get(role, {})
            shapes = role_data.get('contour_shapes')
            if shapes:
                treat_shapes[treat].extend(shapes)
                sols = role_data.get('contour_solidities', [])
                if len(sols) < len(shapes):
                    sols = list(sols) + [1.0] * (len(shapes) - len(sols))
                treat_sols[treat].extend(sols[:len(shapes)])

        # Apply paw-like filter if requested
        treat_total = {}  # original counts (before filtering)
        if filter_paw:
            sol_thresh = self._pawlike_thresholds.get('solidity', 1.00)
            ar_thresh = self._pawlike_thresholds.get('aspect_ratio', 1.6)
            circ_thresh = self._pawlike_thresholds.get('circularity', 0.10)
            for treat in treatment_labels:
                shapes = treat_shapes[treat]
                if not shapes:
                    continue
                treat_total[treat] = len(shapes)
                stacked_t = np.array(shapes)
                ar_t, circ_t = self._shape_metrics_batch(stacked_t)
                sols_t = np.array(treat_sols[treat][:len(shapes)], dtype=float)
                if len(sols_t) < len(shapes):
                    sols_t = np.concatenate([sols_t, np.ones(len(shapes) - len(sols_t))])
                mask = (sols_t <= sol_thresh) & (ar_t <= ar_thresh) & (circ_t >= circ_thresh)
                treat_shapes[treat] = [shapes[i] for i in np.where(mask)[0]]

        # Check we have any shapes
        if not any(treat_shapes.values()):
            return

        if as_frame:
            frame = nb
        else:
            frame = ttk.Frame(nb)
            nb.add(frame, text=tab_name)

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.set_aspect('equal')

        paw_label = {'HL': 'Hind Left', 'HR': 'Hind Right',
                     'FL': 'Fore Left', 'FR': 'Fore Right'}.get(role, role)

        for treat in treatment_labels:
            shapes = treat_shapes[treat]
            if not shapes:
                continue
            # Stack all shapes: (N_shapes, 64, 2)
            stacked = np.array(shapes)
            mean_shape = stacked.mean(axis=0)
            sd_shape = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean_shape)

            # Close the polygon for plotting
            mean_closed = np.vstack([mean_shape, mean_shape[0:1]])
            sd_closed = np.vstack([sd_shape, sd_shape[0:1]])

            if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
                raw = graph_cfg['colors'][treat]
                color = 'black' if raw == 'white_black' else raw
            else:
                color = self._PAW_COLORS_ROLE.get(role, '#1f77b4')

            if filter_paw and treat in treat_total:
                _shape_lbl = f'{treat} (n={len(stacked)}/{treat_total[treat]})'
            else:
                _shape_lbl = f'{treat} (n={len(stacked)})'
            ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                    color=color, linewidth=2.0, label=_shape_lbl)

            # SD envelope: offset contour points radially by ±1 SD
            radial_sd = np.sqrt(sd_closed[:, 0] ** 2 + sd_closed[:, 1] ** 2)
            centroid = mean_closed[:-1].mean(axis=0)
            directions = mean_closed - centroid
            norms = np.sqrt((directions ** 2).sum(axis=1, keepdims=True))
            norms[norms == 0] = 1
            unit_dirs = directions / norms

            outer = mean_closed + unit_dirs * radial_sd[:, np.newaxis]
            inner = mean_closed - unit_dirs * radial_sd[:, np.newaxis]

            # Draw SD band as a filled ring (outer → reversed inner)
            ring_x = np.concatenate([outer[:, 0], inner[::-1, 0]])
            ring_y = np.concatenate([outer[:, 1], inner[::-1, 1]])
            ax.fill(ring_x, ring_y, alpha=0.15, color=color)

        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        self._style_ax(ax, title=f'Mean Contour — {paw_label}',
                       xlabel='Normalized X', ylabel='Normalized Y')
        if len(treatment_labels) > 1:
            ax.legend(fontsize=9)
        # Invert y-axis (image coordinates: y increases downward)
        ax.invert_yaxis()

        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        def _exp_graph(f=fig, n=f'contour_shape_{role}'):
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile=f'{n}.png',
                parent=frame.winfo_toplevel())
            if path:
                f.savefig(path, dpi=300, bbox_inches='tight')

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
        self._embed_figure(frame, fig, ax=ax)

    def _add_contour_print_tab(self, nb, sessions_df, intermediates,
                               role, tab_name='Paw Print', graph_cfg=None,
                               n_prints=5, group_by='treatment', filter_paw=False,
                               as_frame=False):
        """Add a tab showing representative individual paw-print contours.

        Parameters
        ----------
        group_by : str
            'treatment' — one subplot per treatment group (default).
            'subject'  — one subplot per subject, colored by treatment.
        n_prints : int
            Number of representative contours to draw. 1 gives a single
            clean print with no background cloud.
        filter_paw : bool
            If True, only show shapes with solidity <= 0.78 (paw-like).
        """
        has_treats = ('treatment' in sessions_df.columns
                      and sessions_df['treatment'].ne('').any()
                      and sessions_df['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in sessions_df['treatment'].dropna().unique()
                          if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']

        # ---- Gather shapes + solidities keyed by group (treatment or subject) ----
        group_shapes = {}       # group_key -> list of (64,2) arrays
        group_solidities = {}   # group_key -> list of float (extraction-time solidity)
        group_treat = {}        # group_key -> treatment label (for subject coloring)

        for _, row in sessions_df.iterrows():
            sess_name = row.get('session', '')
            treat = str(row.get('treatment', '')) if has_treats else 'All sessions'
            interm = intermediates.get(sess_name, {})
            pcd = interm.get('paw_contour_data', {})
            role_data = pcd.get(role, {})
            shapes = role_data.get('contour_shapes')
            if not shapes:
                continue
            sols = role_data.get('contour_solidities', [])
            # Pad solidities with 1.0 if missing (legacy data without stored solidities)
            if len(sols) < len(shapes):
                sols = list(sols) + [1.0] * (len(shapes) - len(sols))

            if group_by == 'subject':
                subj = str(row.get('subject', sess_name))
                if not subj.strip():
                    subj = sess_name
                group_shapes.setdefault(subj, []).extend(shapes)
                group_solidities.setdefault(subj, []).extend(sols[:len(shapes)])
                group_treat[subj] = treat
            else:
                if treat not in treatment_labels:
                    continue
                group_shapes.setdefault(treat, []).extend(shapes)
                group_solidities.setdefault(treat, []).extend(sols[:len(shapes)])
                group_treat[treat] = treat

        if not any(group_shapes.values()):
            return

        # Determine ordered group keys
        if group_by == 'subject':
            # Sort subjects by treatment order, then alphabetically within
            treat_rank = {t: i for i, t in enumerate(treatment_labels)}
            active_groups = sorted(
                [g for g in group_shapes if group_shapes[g]],
                key=lambda g: (treat_rank.get(group_treat.get(g, ''), 999), g))
        else:
            active_groups = [t for t in treatment_labels if group_shapes.get(t)]

        n_groups = len(active_groups)
        if n_groups == 0:
            return

        if as_frame:
            frame = nb
        else:
            frame = ttk.Frame(nb)
            nb.add(frame, text=tab_name)

        # Layout: wrap to multiple rows when many subjects
        max_cols = min(n_groups, 6)
        n_rows = (n_groups + max_cols - 1) // max_cols
        n_cols = min(n_groups, max_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 4.5 * n_rows),
                                 constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()

        paw_label = {'HL': 'Hind Left', 'HR': 'Hind Right',
                     'FL': 'Fore Left', 'FR': 'Fore Right'}.get(role, role)

        # Pre-compute per-group data for drawing (and interactive navigation)
        group_data = []  # list of dicts with stacked, sorted_idx, color, label
        for grp in active_groups:
            shapes = group_shapes[grp]
            stacked = np.array(shapes)  # (N, 64, 2)
            treat_for_color = group_treat.get(grp, '')
            if graph_cfg and graph_cfg.get('colors') and treat_for_color in graph_cfg['colors']:
                raw = graph_cfg['colors'][treat_for_color]
                color = 'black' if raw == 'white_black' else raw
            else:
                color = self._PAW_COLORS_ROLE.get(role, '#1f77b4')
            total_all = len(stacked)
            mean_shape = stacked.mean(axis=0)
            dists = ((stacked - mean_shape) ** 2).sum(axis=(1, 2))
            sorted_idx = np.argsort(dists)  # all indices, ranked by closeness

            # Apply paw-like filter using stored solidities + shape metrics
            if filter_paw:
                sol_thresh = self._pawlike_thresholds.get('solidity', 1.00)
                ar_thresh = self._pawlike_thresholds.get('aspect_ratio', 1.6)
                circ_thresh = self._pawlike_thresholds.get('circularity', 0.10)
                grp_sols = np.array(group_solidities.get(grp, [1.0] * total_all))
                ar_all, circ_all = self._shape_metrics_batch(stacked)
                paw_mask = ((grp_sols <= sol_thresh)
                            & (ar_all <= ar_thresh)
                            & (circ_all >= circ_thresh))
                sorted_idx = np.array([i for i in sorted_idx if paw_mask[i]])
                if len(sorted_idx) == 0:
                    sorted_idx = np.argsort(dists)[:1]  # fallback: keep closest

            group_data.append(dict(
                stacked=stacked, sorted_idx=sorted_idx,
                color=color, grp=grp, treat=treat_for_color,
                total=len(sorted_idx), total_all=total_all))

        # Maximum offset: limited by the smallest group
        max_offset = max(0, min(gd['total'] - n_prints for gd in group_data))

        # Mutable state for navigation
        state = dict(offset=0, canvas_widget=None, fig=None)

        def _draw(offset):
            """Draw (or redraw) all subplots at the given offset."""
            if state['fig'] is not None:
                plt.close(state['fig'])
            if state['canvas_widget'] is not None:
                state['canvas_widget'].destroy()

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4.5 * n_cols, 4.5 * n_rows),
                                     constrained_layout=True, squeeze=False)
            axes_flat = axes.ravel()

            for idx, gd in enumerate(group_data):
                ax = axes_flat[idx]
                stacked = gd['stacked']
                si = gd['sorted_idx']
                color = gd['color']
                k = min(n_prints, gd['total'] - offset)
                if k <= 0:
                    k = 1
                sel = si[offset:offset + k]

                best_alpha = 0.85 if k == 1 else 0.7

                # Background representatives (indices 1..k-1)
                for ci in sel[1:]:
                    pts = stacked[ci]
                    closed = np.vstack([pts, pts[0:1]])
                    ax.fill(closed[:, 0], closed[:, 1],
                            facecolor=color, alpha=0.15,
                            edgecolor=color, linewidth=0.8)

                # Primary contour on top
                best = stacked[sel[0]]
                best_closed = np.vstack([best, best[0:1]])
                ax.fill(best_closed[:, 0], best_closed[:, 1],
                        facecolor=color, alpha=best_alpha,
                        edgecolor=color, linewidth=0.8)

                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

                rank_start = offset + 1  # 1-based for display
                if filter_paw and gd.get('total_all', gd['total']) != gd['total']:
                    count_lbl = f'#{rank_start} of {gd["total"]} paw-like / {gd["total_all"]} total'
                else:
                    count_lbl = f'#{rank_start} of {gd["total"]}'
                if group_by == 'subject':
                    subtitle = f'{gd["grp"]} ({count_lbl})'
                elif n_groups > 1:
                    subtitle = f'{gd["treat"]} ({count_lbl})'
                else:
                    subtitle = f'{paw_label} ({count_lbl})'
                self._style_ax(ax, title=subtitle,
                               xlabel='Normalized X', ylabel='Normalized Y')

            for j in range(n_groups, len(axes_flat)):
                axes_flat[j].set_visible(False)

            if group_by == 'subject':
                fig.suptitle(f'Paw Print — {paw_label} (by Subject)', fontsize=12)
            elif n_groups > 1:
                fig.suptitle(f'Paw Print — {paw_label}', fontsize=12)

            state['fig'] = fig
            canvas = FigureCanvasTkAgg(fig, master=frame)
            cw = canvas.get_tk_widget()
            cw.pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(canvas, fig)
            _draw_canvas_fit(canvas, fig)
            state['canvas_widget'] = cw
            # Update button states
            if max_offset > 0:
                btn_prev.config(state='disabled' if offset == 0 else 'normal')
                btn_next.config(state='disabled' if offset >= max_offset else 'normal')

        # --- Button bar (packed at bottom before the figure) ---
        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        suffix = f'_{group_by}' if group_by == 'subject' else ''
        def _exp_graph():
            if state['fig'] is None:
                return
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile=f'contour_print_{role}{suffix}.png',
                parent=frame.winfo_toplevel())
            if path:
                state['fig'].savefig(path, dpi=300, bbox_inches='tight')

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

        # Navigation buttons (prev / next contour selection)
        def _prev():
            if state['offset'] > 0:
                state['offset'] -= 1
                _draw(state['offset'])

        def _next():
            if state['offset'] < max_offset:
                state['offset'] += 1
                _draw(state['offset'])

        btn_prev = ttk.Button(btn_bar, text="\u25C0 Prev", command=_prev)
        btn_next = ttk.Button(btn_bar, text="Next \u25B6", command=_next)
        if max_offset > 0:
            btn_prev.pack(side='left', padx=4)
            btn_next.pack(side='left', padx=4)

        # Initial draw
        _draw(0)

    def _add_contour_filter_preview_tab(self, nb, sessions_df, intermediates,
                                        role, tab_name='Filter Preview',
                                        graph_cfg=None):
        """Interactive filter preview with sliders for solidity, aspect ratio, circularity."""
        GRID_COLS = 4
        GRID_ROWS = 5
        PAGE_SIZE = GRID_COLS * GRID_ROWS

        # ---- Gather all shapes + extraction-time solidities ----
        all_shapes = []
        all_sols = []
        for _, row in sessions_df.iterrows():
            sess_name = row.get('session', '')
            interm = intermediates.get(sess_name, {})
            pcd = interm.get('paw_contour_data', {})
            role_data = pcd.get(role, {})
            shapes = role_data.get('contour_shapes')
            if not shapes:
                continue
            sols = role_data.get('contour_solidities', [])
            # Pad with 1.0 for legacy data without stored solidities
            if len(sols) < len(shapes):
                sols = list(sols) + [1.0] * (len(shapes) - len(sols))
            all_shapes.extend(shapes)
            all_sols.extend(sols[:len(shapes)])

        if not all_shapes:
            return

        stacked = np.array(all_shapes)  # (N, 64, 2)
        sol_vals = np.array(all_sols, dtype=float)
        total_all = len(stacked)

        # Pre-compute aspect ratio and circularity for all shapes (vectorized)
        ar_vals, circ_vals = self._shape_metrics_batch(stacked)

        paw_label = {'HL': 'Hind Left', 'HR': 'Hind Right',
                     'FL': 'Fore Left', 'FR': 'Fore Right'}.get(role, role)
        role_color = self._PAW_COLORS_ROLE.get(role, '#1f77b4')

        # ---- Build UI ----
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)

        # -- Slider controls --
        ctrl_frame = ttk.LabelFrame(frame, text='Filter Thresholds', padding=6)
        ctrl_frame.pack(side='top', fill='x', padx=6, pady=(4, 2))

        sol_var = tk.DoubleVar(value=self._pawlike_thresholds.get('solidity', 1.00))
        ar_var = tk.DoubleVar(value=self._pawlike_thresholds.get('aspect_ratio', 1.6))
        circ_var = tk.DoubleVar(value=self._pawlike_thresholds.get('circularity', 0.10))

        # Store slider vars for sync support — callback added after _on_slider_change defined
        _PAIRED = {'HL': 'HR', 'HR': 'HL', 'FL': 'FR', 'FR': 'FL'}
        _syncing = [False]  # guard against infinite recursion

        def _make_slider(parent, label, var, from_, to, resolution, row, tooltip):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 4))
            scale = ttk.Scale(parent, from_=from_, to=to, variable=var,
                              orient='horizontal', length=260,
                              command=lambda _: _on_slider_change())
            scale.grid(row=row, column=1, sticky='ew', padx=4)
            val_lbl = ttk.Label(parent, text=f'{var.get():.2f}', width=6)
            val_lbl.grid(row=row, column=2, padx=(0, 4))
            _ToolTip(scale, tooltip)
            return val_lbl

        ctrl_frame.columnconfigure(1, weight=1)
        sol_lbl = _make_slider(ctrl_frame, 'Solidity  \u2264', sol_var, 0.50, 1.00, 0.01, 0,
                               'Shapes with solidity \u2264 threshold pass (lower = stricter)')
        ar_lbl = _make_slider(ctrl_frame, 'Aspect Ratio  \u2264', ar_var, 1.0, 10.0, 0.1, 1,
                              'Shapes with aspect ratio \u2264 threshold pass (filters elongated shapes)')
        circ_lbl = _make_slider(ctrl_frame, 'Circularity  \u2265', circ_var, 0.0, 1.0, 0.01, 2,
                                'Shapes with circularity \u2265 threshold pass (filters near-circular blobs)')

        # Status label
        status_var = tk.StringVar(value='')
        ttk.Label(ctrl_frame, textvariable=status_var, font=('TkDefaultFont', 9, 'bold')).grid(
            row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

        # -- View toggle --
        view_frame = ttk.Frame(frame)
        view_frame.pack(side='top', fill='x', padx=6, pady=(2, 2))
        view_var = tk.StringVar(value='Included')
        ttk.Label(view_frame, text='View:').pack(side='left', padx=(0, 4))
        view_combo = ttk.Combobox(view_frame, textvariable=view_var,
                                  values=['Included', 'Excluded'],
                                  state='readonly', width=12)
        view_combo.pack(side='left')

        # -- Button bar (pack before canvas so it reserves space at bottom) --
        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 4))

        # -- Canvas area --
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(side='top', fill='both', expand=True, padx=4, pady=2)

        # Mutable state
        state = dict(page=0, canvas_widget=None, fig=None,
                     included_idx=np.array([], dtype=int),
                     excluded_idx=np.array([], dtype=int))

        def _compute_mask():
            """Recompute filter mask from current slider values."""
            sol_t = sol_var.get()
            ar_t = ar_var.get()
            circ_t = circ_var.get()
            mask = (sol_vals <= sol_t) & (ar_vals <= ar_t) & (circ_vals >= circ_t)
            state['included_idx'] = np.where(mask)[0]
            state['excluded_idx'] = np.where(~mask)[0]
            n_inc = len(state['included_idx'])
            pct = int(100 * n_inc / total_all) if total_all > 0 else 0
            status_var.set(f'Keeping {n_inc} of {total_all} shapes ({pct}%)')
            # Update value labels
            sol_lbl.config(text=f'{sol_t:.2f}')
            ar_lbl.config(text=f'{ar_t:.2f}')
            circ_lbl.config(text=f'{circ_t:.2f}')

        def _draw_page(page):
            if state['fig'] is not None:
                plt.close(state['fig'])
            if state['canvas_widget'] is not None:
                state['canvas_widget'].destroy()
                state['canvas_widget'] = None

            viewing = view_var.get()
            idx_pool = state['included_idx'] if viewing == 'Included' else state['excluded_idx']
            total_pool = len(idx_pool)

            if total_pool == 0:
                # Show empty message
                lbl = ttk.Label(canvas_frame,
                                text=f'No {viewing.lower()} shapes with current thresholds.',
                                font=('TkDefaultFont', 10))
                lbl.pack(fill='both', expand=True)
                state['canvas_widget'] = lbl
                state['fig'] = None
                btn_prev.config(state='disabled')
                btn_next.config(state='disabled')
                return

            n_pages = max(1, (total_pool + PAGE_SIZE - 1) // PAGE_SIZE)
            page = max(0, min(page, n_pages - 1))
            state['page'] = page

            start = page * PAGE_SIZE
            end = min(start + PAGE_SIZE, total_pool)
            page_indices = idx_pool[start:end]
            n_show = len(page_indices)
            n_r = (n_show + GRID_COLS - 1) // GRID_COLS
            n_c = min(n_show, GRID_COLS)

            fig, axes = plt.subplots(n_r, n_c,
                                     figsize=(2.5 * n_c, 2.5 * n_r),
                                     constrained_layout=True, squeeze=False)
            axes_flat = axes.ravel()

            for i, si in enumerate(page_indices):
                ax = axes_flat[i]
                pts = stacked[si]
                closed = np.vstack([pts, pts[0:1]])
                if viewing == 'Included':
                    fc, ec = role_color, role_color
                else:
                    fc, ec = '#999999', '#666666'
                ax.fill(closed[:, 0], closed[:, 1],
                        facecolor=fc, alpha=0.5,
                        edgecolor=ec, linewidth=0.8)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.set_title(f'sol={sol_vals[si]:.2f}  ar={ar_vals[si]:.1f}  ci={circ_vals[si]:.2f}',
                             fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

            for j in range(n_show, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(
                f'{viewing}: {total_pool} shapes — {paw_label}  '
                f'(page {page + 1}/{n_pages})',
                fontsize=10, y=1.02)

            state['fig'] = fig
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            cw = canvas.get_tk_widget()
            cw.pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(canvas, fig)
            _draw_canvas_fit(canvas, fig)
            state['canvas_widget'] = cw

            btn_prev.config(state='disabled' if page == 0 else 'normal')
            btn_next.config(state='disabled' if page >= n_pages - 1 else 'normal')

        def _on_slider_change():
            _compute_mask()
            state['page'] = 0
            _draw_page(0)
            # Sync paired paw if enabled
            if _syncing[0]:
                return
            if (hasattr(self, '_filter_sync_lr_var')
                    and self._filter_sync_lr_var.get()
                    and hasattr(self, '_filter_slider_vars')):
                paired = _PAIRED.get(role)
                if paired and paired in self._filter_slider_vars:
                    p_sol, p_ar, p_circ, p_cb = self._filter_slider_vars[paired]
                    p_sol.set(sol_var.get())
                    p_ar.set(ar_var.get())
                    p_circ.set(circ_var.get())
                    _syncing[0] = True
                    try:
                        p_cb()
                    finally:
                        _syncing[0] = False

        def _on_view_change(_event=None):
            state['page'] = 0
            _draw_page(0)

        view_combo.bind('<<ComboboxSelected>>', _on_view_change)

        # Register slider vars + callback for sync support
        if hasattr(self, '_filter_slider_vars'):
            self._filter_slider_vars[role] = (sol_var, ar_var, circ_var, _on_slider_change)

        def _prev():
            if state['page'] > 0:
                _draw_page(state['page'] - 1)

        def _next():
            _draw_page(state['page'] + 1)

        btn_prev = ttk.Button(btn_bar, text="\u25C0 Prev", command=_prev)
        btn_prev.pack(side='left', padx=4)
        btn_next = ttk.Button(btn_bar, text="Next \u25B6", command=_next)
        btn_next.pack(side='left', padx=4)

        def _exp_graph():
            if state['fig'] is None:
                return
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile=f'contour_filter_preview_{role}.png',
                parent=frame.winfo_toplevel())
            if path:
                state['fig'].savefig(path, dpi=300, bbox_inches='tight')

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)

        def _apply():
            """Store current thresholds and live-regenerate the Filtered Contour tab."""
            self._pawlike_thresholds['solidity'] = sol_var.get()
            self._pawlike_thresholds['aspect_ratio'] = ar_var.get()
            self._pawlike_thresholds['circularity'] = circ_var.get()
            try:
                self._rebuild_filtered_contour_tab()
                self._refresh_results_table()
            except Exception as exc:
                messagebox.showerror(
                    'Rebuild Error',
                    f'Failed to regenerate Filtered Contour tab:\n{exc}',
                    parent=frame.winfo_toplevel())

        ttk.Button(btn_bar, text="Apply", command=_apply).pack(side='right', padx=4)

        # Defer initial draw until this paw tab becomes visible
        _initial_drawn = [False]
        def _on_map(evt=None):
            if not _initial_drawn[0]:
                _initial_drawn[0] = True
                _compute_mask()
                _draw_page(0)
        frame.bind('<Map>', _on_map)

    def _add_contour_grouped_bar_tab(self, nb, df, metric_key, tab_name,
                                      y_label='', graph_cfg=None):
        """Grouped bar chart: paws on x-axis, treatments as grouped bars."""
        paw_cols = [(r, f'{metric_key}_{r}') for r in self.ROLES
                    if f'{metric_key}_{r}' in df.columns
                    and df[f'{metric_key}_{r}'].notna().any()]
        if not paw_cols:
            return

        has_treats = ('treatment' in df.columns
                      and df['treatment'].ne('').any()
                      and df['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in df['treatment'].dropna().unique() if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']
        n_treats = len(treatment_labels)

        _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)

        fig, ax = plt.subplots(
            figsize=(max(5, len(paw_cols) * 1.4 + 1), 4), constrained_layout=True)
        x     = np.arange(len(paw_cols))
        bar_w = 0.7 / n_treats
        rng   = np.random.default_rng(42)

        for ti, treat in enumerate(treatment_labels):
            xpos = x - 0.35 + (ti + 0.5) * bar_w
            for pi, (role, col) in enumerate(paw_cols):
                if has_treats and n_treats > 1:
                    subset = df[df['treatment'] == treat][col].dropna()
                else:
                    subset = df[col].dropna()
                mean_ = subset.mean() if len(subset) else 0
                err_  = self._calc_error(subset.values, _err_type)
                if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
                    raw = graph_cfg['colors'][treat]
                    color = 'white' if raw == 'white_black' else raw
                    edge  = 'black' if raw == 'white_black' else 'black'
                elif n_treats > 1:
                    color = list(plt.cm.Set2(np.linspace(0, 0.8, n_treats)))[ti]
                    edge = 'black'
                else:
                    color = self._PAW_COLORS_ROLE.get(role, '#1f77b4')
                    edge = 'black'
                ax.bar(xpos[pi], mean_, bar_w * 0.9,
                       color=color, edgecolor=edge, yerr=err_, capsize=4,
                       label=treat if pi == 0 else None)
                jx = xpos[pi] + rng.uniform(-bar_w * 0.3, bar_w * 0.3, len(subset))
                ax.scatter(jx, subset.values, color=color, s=22, alpha=0.6,
                           zorder=3, edgecolors='black', linewidths=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([r for r, _ in paw_cols], fontsize=11)
        self._style_ax(ax, title=tab_name, ylabel=y_label or metric_key)
        if n_treats > 1:
            ax.legend(fontsize=10)

        if self._enable_stats_var.get() and n_treats == 2:
            for pi, (role, col) in enumerate(paw_cols):
                v0 = (df[df['treatment'] == treatment_labels[0]][col].dropna().values
                      if has_treats else df[col].dropna().values)
                v1 = (df[df['treatment'] == treatment_labels[1]][col].dropna().values
                      if has_treats else np.array([]))
                if len(v0) > 0 and len(v1) > 0:
                    try:
                        _, p = _sp_stats.ttest_ind(v0, v1, equal_var=False)
                        lbl = _p_label(p, self._sig_style)
                        if lbl:
                            x0 = x[pi] - 0.35 + 0.5 * bar_w
                            x1 = x[pi] - 0.35 + 1.5 * bar_w
                            y_ann = ax.get_ylim()[1] * 0.96
                            ax.plot([x0, x0, x1, x1],
                                    [y_ann * 0.97, y_ann, y_ann, y_ann * 0.97],
                                    color='black', linewidth=1)
                            ax.text((x0 + x1) / 2, y_ann * 1.01, lbl,
                                    ha='center', va='bottom',
                                    fontsize=11, fontweight='bold')
                    except Exception:
                        pass
        elif self._enable_stats_var.get() and n_treats > 2:
            for pi, (role, col) in enumerate(paw_cols):
                grp_vals = [df[df['treatment'] == t][col].dropna().values
                            for t in treatment_labels]
                grp_vals = [v for v in grp_vals if len(v) > 0]
                if len(grp_vals) > 1:
                    try:
                        _, p = _sp_stats.f_oneway(*grp_vals)
                        lbl = _p_label(p, self._sig_style)
                        if lbl:
                            ax.text(x[pi], ax.get_ylim()[1] * 0.96, lbl,
                                    ha='center', va='bottom',
                                    fontsize=11, fontweight='bold')
                    except Exception:
                        pass

        btn_bar = ttk.Frame(frame)
        btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

        def _exp_graph(f=fig, n=tab_name):
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                initialfile=f'{n.replace(" ", "_")}.png',
                parent=frame.winfo_toplevel())
            if path:
                f.savefig(path, dpi=300, bbox_inches='tight')

        def _exp_data(d=df, act=paw_cols, ht=has_treats):
            cols_active = [col for _, col in act]
            export_cols = (['treatment'] + cols_active if ht else cols_active)
            out = d[[c for c in export_cols if c in d.columns]].copy()
            path = filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                initialfile=f'{tab_name.replace(" ", "_")}_data.csv',
                parent=frame.winfo_toplevel())
            if path:
                out.to_csv(path, index=False)

        ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
        ttk.Button(btn_bar, text="Export Data",  command=_exp_data).pack(side='left', padx=2)
        self._embed_figure(frame, fig, ax=ax)

    @staticmethod
    def _rebin_timecourse(xs, means, errs, rebin_min):
        """Aggregate timecourse data into larger bins."""
        if not xs or rebin_min <= 0:
            return xs, means, errs
        new_xs, new_means, new_errs = [], [], []
        i = 0
        while i < len(xs):
            bin_start = xs[i]
            bin_end = bin_start + rebin_min
            group_m, group_e = [], []
            while i < len(xs) and xs[i] < bin_end:
                group_m.append(means[i])
                group_e.append(errs[i])
                i += 1
            if group_m:
                new_xs.append(bin_start)
                new_means.append(float(np.mean(group_m)))
                # Propagate error: average of errors (simple approach)
                new_errs.append(float(np.mean(group_e)))
        return new_xs, new_means, new_errs

    def _add_contour_grouped_tc_tab(self, nb, bdf, metric_key, tab_name,
                                     y_label='', graph_cfg=None):
        """Timecourse with one sub-tab per paw for cleaner display."""
        paw_cols = [(r, f'{metric_key}_{r}') for r in self.ROLES
                    if f'{metric_key}_{r}' in bdf.columns
                    and bdf[f'{metric_key}_{r}'].notna().any()]
        if not paw_cols or 'bin_start_s' not in bdf.columns:
            return

        has_treats = ('treatment' in bdf.columns
                      and bdf['treatment'].ne('').any()
                      and bdf['treatment'].notna().any())
        if has_treats:
            all_treats = [str(t) for t in bdf['treatment'].dropna().unique() if str(t).strip()]
            if graph_cfg and graph_cfg.get('order'):
                treatment_labels = [t for t in graph_cfg['order'] if t in all_treats]
                treatment_labels += [t for t in all_treats if t not in treatment_labels]
            else:
                treatment_labels = sorted(all_treats)
        else:
            treatment_labels = ['All sessions']

        _err_type = graph_cfg.get('error_type', 'SEM') if graph_cfg else 'SEM'

        # Re-bin support
        rebin = graph_cfg.get('rebin_minutes') if graph_cfg else None

        # Filter bins to the user-specified time window
        if graph_cfg and graph_cfg.get('time_window') is not None:
            _tw_sec = graph_cfg['time_window'] * 60.0
            bdf = bdf[bdf['bin_start_s'] <= _tw_sec].copy()

        outer_frame = ttk.Frame(nb)
        nb.add(outer_frame, text=tab_name)
        paw_nb = ttk.Notebook(outer_frame)
        paw_nb.pack(fill='both', expand=True)

        for role, col in paw_cols:
            paw_frame = ttk.Frame(paw_nb)
            paw_nb.add(paw_frame, text=role)

            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

            for ti, treat in enumerate(treatment_labels):
                if has_treats and len(treatment_labels) > 1:
                    sub = bdf[bdf['treatment'] == treat]
                else:
                    sub = bdf
                bins = sorted(sub['bin_start_s'].unique())

                means, errs, xs = [], [], []
                for b in bins:
                    vals = sub[sub['bin_start_s'] == b][col].dropna().values
                    if len(vals) > 0:
                        means.append(np.nanmean(vals))
                        errs.append(self._calc_error(vals, _err_type))
                        xs.append(b / 60.0)

                if rebin and rebin > 0 and xs:
                    xs, means, errs = self._rebin_timecourse(xs, means, errs, rebin)

                if xs:
                    if graph_cfg and graph_cfg.get('colors') and treat in graph_cfg['colors']:
                        raw = graph_cfg['colors'][treat]
                        color = 'black' if raw == 'white_black' else raw
                    else:
                        color = self._PAW_COLORS_ROLE.get(role, '#1f77b4')
                    _tkey = str(treat)
                    _lstyle = graph_cfg.get('line_styles', {}).get(_tkey, '-') if graph_cfg else '-'
                    _mshape = graph_cfg.get('marker_shapes', {}).get(_tkey, 'o') if graph_cfg else 'o'
                    _msize  = graph_cfg.get('marker_sizes', {}).get(_tkey, 4) if graph_cfg else 4
                    _lw     = graph_cfg.get('line_widths', {}).get(_tkey, 1.8) if graph_cfg else 1.8
                    _alpha  = graph_cfg.get('opacities', {}).get(_tkey, 1.0) if graph_cfg else 1.0
                    _mfill = graph_cfg.get('marker_fills', {}).get(_tkey, 'full') if graph_cfg else 'full'
                    _pkw = dict(color=color, linewidth=_lw, linestyle=_lstyle,
                                alpha=_alpha, label=treat)
                    if _mshape and _mshape != 'none':
                        _pkw['marker'] = _mshape
                        _pkw['markersize'] = _msize
                        _pkw['fillstyle'] = _mfill
                        if _mfill == 'none':
                            _pkw['markeredgewidth'] = 1.2
                        elif _mfill in ('left', 'right', 'top', 'bottom'):
                            _pkw['markerfacecoloralt'] = 'white'
                    if raw == 'white_black':
                        _pkw['markerfacecolor'] = 'white'
                        _pkw['markeredgecolor'] = 'black'
                    _edge_sel = graph_cfg.get('marker_edge_colors', {}).get(_tkey, 'auto') if graph_cfg else 'auto'
                    if _edge_sel == 'black':
                        _pkw['markeredgecolor'] = 'black'
                    elif _edge_sel == 'white':
                        _pkw['markeredgecolor'] = 'white'
                    elif _edge_sel == 'match':
                        _pkw['markeredgecolor'] = color
                    _err_disp = graph_cfg.get('error_display', 'circles_caps') if graph_cfg else 'circles_caps'
                    if _err_disp == 'circles_caps':
                        _line_kw = {k: v for k, v in _pkw.items()
                                    if k not in ('marker', 'markersize', 'fillstyle',
                                                 'markeredgewidth', 'markerfacecoloralt',
                                                 'markerfacecolor', 'markeredgecolor')}
                        ax.plot(xs, means, **_line_kw)
                    else:
                        ax.plot(xs, means, **_pkw)
                    if _err_disp == 'caps':
                        ax.errorbar(xs, means, yerr=errs, capsize=4,
                                    capthick=1.5, fmt='none',
                                    alpha=0.6 * _alpha, ecolor=color)
                    elif _err_disp == 'circles_caps':
                        ax.errorbar(xs, means, yerr=errs, capsize=4,
                                    capthick=1.5, fmt='none',
                                    alpha=0.6 * _alpha, ecolor=color)
                        _msz = _msize if (_mshape and _mshape != 'none') else 5
                        _mkw = dict(linestyle='none', color=color,
                                    marker='o', markersize=_msz, alpha=_alpha)
                        _mkw['fillstyle'] = _mfill
                        if _mfill == 'none':
                            _mkw['markeredgewidth'] = 1.2
                        elif _mfill in ('left', 'right', 'top', 'bottom'):
                            _mkw['markerfacecoloralt'] = 'white'
                        if raw == 'white_black':
                            _mkw['markerfacecolor'] = 'white'
                            _mkw['markeredgecolor'] = 'black'
                        if _edge_sel == 'black':
                            _mkw['markeredgecolor'] = 'black'
                        elif _edge_sel == 'white':
                            _mkw['markeredgecolor'] = 'white'
                        elif _edge_sel == 'match':
                            _mkw['markeredgecolor'] = color
                        ax.plot(xs, means, **_mkw)
                    else:
                        ax.fill_between(xs,
                                        np.array(means) - np.array(errs),
                                        np.array(means) + np.array(errs),
                                        alpha=0.18 * _alpha, color=color)

            paw_label = {'HL': 'Hind Left', 'HR': 'Hind Right',
                         'FL': 'Fore Left', 'FR': 'Fore Right'}.get(role, role)
            self._style_ax(ax, title=f'{tab_name} — {paw_label}',
                           xlabel='Time (min)', ylabel=y_label or metric_key)
            ax.annotate(self._error_label(_err_type),
                        xy=(1, 1), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=8,
                        color='gray', style='italic')
            if len(treatment_labels) > 1:
                ax.legend(fontsize=10)

            # Per-bin significance markers (two-way ANOVA gatekeeper)
            if self._enable_stats_var.get() and len(treatment_labels) >= 2 and has_treats:
                from matplotlib.transforms import blended_transform_factory
                import warnings as _warnings
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                tw = graph_cfg.get('time_window') if graph_cfg else None
                alpha = self._stats_alpha_var.get()

                # Step 1: Two-way ANOVA gatekeeper
                _do_posthoc = False
                _anova_text = ''
                try:
                    import statsmodels.api as _sm
                    from statsmodels.formula.api import ols as _ols
                    _adf = bdf[['subject', 'treatment', 'bin_start_s', col]].dropna().copy()
                    _adf['treatment'] = _adf['treatment'].astype('category')
                    _adf['bin_start_s'] = _adf['bin_start_s'].astype('category')
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings('ignore', module='statsmodels')
                        _model = _ols(f'{col} ~ C(treatment) + C(bin_start_s) + C(treatment):C(bin_start_s)',
                                      data=_adf).fit()
                        _at = _sm.stats.anova_lm(_model, typ=2)
                    _treat_p = _at.loc['C(treatment)', 'PR(>F)']
                    _inter_p = _at.loc['C(treatment):C(bin_start_s)', 'PR(>F)']
                    if _inter_p < alpha or _treat_p < alpha:
                        _do_posthoc = True
                    def _pfmt(p):
                        if p < 0.001: return 'p<0.001***'
                        if p < 0.01:  return f'p={p:.3f}**'
                        if p < alpha: return f'p={p:.3f}*'
                        return f'p={p:.3f} ns'
                    _anova_text = f"Two-way ANOVA: Treatment {_pfmt(_treat_p)}, Interaction {_pfmt(_inter_p)}"
                except Exception:
                    _do_posthoc = True

                # Step 2: Per-bin post-hoc (only if ANOVA is significant)
                if _do_posthoc:
                    bins_unique = sorted(bdf['bin_start_s'].unique())
                    paradigm = self._stats_paradigm_var.get()
                    for bin_t in bins_unique:
                        t_min_val = bin_t / 60.0
                        if tw is not None and t_min_val > tw:
                            continue
                        bin_data = bdf[bdf['bin_start_s'] == bin_t]
                        gvals = [bin_data[bin_data['treatment'] == t][col].dropna().values
                                 for t in treatment_labels if t in bin_data['treatment'].values]
                        gvals = [v for v in gvals if len(v) >= 2]
                        if len(gvals) < 2:
                            continue

                        use_nonparam = paradigm == 'nonparametric'
                        if paradigm == 'auto':
                            for g in gvals:
                                if len(g) >= 3:
                                    try:
                                        _, sw_p = _sp_stats.shapiro(g)
                                        if sw_p < 0.05:
                                            use_nonparam = True
                                            break
                                    except Exception:
                                        pass

                        try:
                            if len(gvals) == 2:
                                if use_nonparam:
                                    _, p = _sp_stats.mannwhitneyu(gvals[0], gvals[1], alternative='two-sided')
                                else:
                                    _, p = _sp_stats.ttest_ind(gvals[0], gvals[1], equal_var=False)
                            else:
                                if use_nonparam:
                                    _, p = _sp_stats.kruskal(*gvals)
                                else:
                                    _, p = _sp_stats.f_oneway(*gvals)
                            lbl = _p_label(p, self._sig_style)
                            if lbl:
                                ax.text(t_min_val, 0.98, lbl, transform=trans,
                                        ha='center', va='top', fontsize=9, color='black')
                        except Exception:
                            pass

                # Show ANOVA annotation
                if _anova_text:
                    ax.text(0.02, 0.98, _anova_text, transform=ax.transAxes,
                            va='top', ha='left', fontsize=7, family='monospace',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

            if graph_cfg and graph_cfg.get('time_window') is not None:
                ax.set_xlim(-0.5, graph_cfg['time_window'] + 0.5)
            else:
                ax.margins(x=0.02)

            btn_bar = ttk.Frame(paw_frame)
            btn_bar.pack(side='bottom', fill='x', padx=4, pady=(0, 2))

            def _exp_graph(f=fig, n=f'{tab_name}_{role}'):
                path = filedialog.asksaveasfilename(
                    defaultextension='.png',
                    filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'), ('PDF', '*.pdf')],
                    initialfile=f'{n.replace(" ", "_")}.png',
                    parent=paw_frame.winfo_toplevel())
                if path:
                    f.savefig(path, dpi=300, bbox_inches='tight')

            ttk.Button(btn_bar, text="Export Graph", command=_exp_graph).pack(side='left', padx=4)
            self._embed_figure(paw_frame, fig, ax=ax)

    # ═══════════════════════════════════════════════════════════════════════
    # Statistics tab methods
    # ═══════════════════════════════════════════════════════════════════════

    def _add_wb_stats_section(self, parent_frame, title, df, metric,
                              treatments, agg_method):
        """Create a descriptive-stats + test-result block for one WB metric.

        agg_method: 'value' — one row per subject (no aggregation needed)
                    'mean'  — aggregate per subject across time bins
        Column names: treatment (lower), subject (lower).
        """
        section_frame = ttk.LabelFrame(parent_frame, text=title, padding=10)
        section_frame.pack(fill='x', padx=10, pady=10)

        # ── Aggregate per subject ────────────────────────────────────────
        if agg_method == 'mean':
            per_subject = (df.groupby(['subject', 'treatment'])[metric]
                           .mean().reset_index())
        else:
            # 'value': already one row per session/subject
            cols = [c for c in ('subject', 'treatment', metric) if c in df.columns]
            per_subject = df[cols].copy()

        # ── Descriptive statistics table ─────────────────────────────────
        ttk.Label(section_frame, text="Descriptive Statistics:",
                  font=('Arial', 10, 'bold')).pack(anchor='w')

        desc_table = ttk.Frame(section_frame)
        desc_table.pack(fill='x', pady=5)

        headers = ['Treatment', 'N', 'Mean', 'SD', 'SEM', 'Min', 'Max']
        for i, hdr in enumerate(headers):
            ttk.Label(desc_table, text=hdr, font=('Arial', 9, 'bold'),
                      relief='solid', borderwidth=1, width=12).grid(
                row=0, column=i, sticky='ew', padx=1, pady=1)

        for row_idx, treat in enumerate(treatments, start=1):
            vals = per_subject[per_subject['treatment'] == treat][metric].dropna().values
            if len(vals) == 0:
                continue
            n    = len(vals)
            mean = np.mean(vals)
            sd   = np.std(vals, ddof=1) if n > 1 else 0.0
            sem  = sd / np.sqrt(n)
            mn   = np.min(vals)
            mx   = np.max(vals)
            for col_idx, cell in enumerate(
                    [treat, str(n), f'{mean:.2f}', f'{sd:.2f}',
                     f'{sem:.2f}', f'{mn:.2f}', f'{mx:.2f}']):
                ttk.Label(desc_table, text=cell, relief='solid',
                          borderwidth=1, width=12).grid(
                    row=row_idx, column=col_idx, sticky='ew', padx=1, pady=1)

        # ── Statistical test ─────────────────────────────────────────────
        ttk.Label(section_frame, text="Statistical Test:",
                  font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))

        data_by_treatment = {t: per_subject[per_subject['treatment'] == t][metric]
                               .dropna().values
                             for t in treatments}
        stats_res = self._perform_wb_statistical_test(data_by_treatment, treatments)

        if stats_res is None:
            ttk.Label(section_frame,
                      text="No statistical test performed (enable in Statistical Tests panel).",
                      foreground='gray', font=('Arial', 9, 'italic')).pack(anchor='w')
            return

        p_val = stats_res['p_value']
        alpha = stats_res['alpha']
        if p_val < 0.001:
            p_text, sig = 'p < 0.001', '***'
        elif p_val < 0.01:
            p_text, sig = f'p = {p_val:.4f}', '**'
        elif p_val < alpha:
            p_text, sig = f'p = {p_val:.4f}', '*'
        else:
            p_text, sig = f'p = {p_val:.4f}', 'ns'

        result_text = f"{stats_res['test_type']}: {p_text} {sig}"
        ttk.Label(section_frame, text=result_text, font=('Arial', 10),
                  foreground='darkblue').pack(anchor='w')

        if 'effect_size' in stats_res:
            es_text = f"Effect size ({stats_res['effect_size_type']}): {stats_res['effect_size']:.3f}"
            ttk.Label(section_frame, text=es_text, font=('Arial', 10),
                      foreground='darkblue').pack(anchor='w')

        # ── Pairwise comparisons (ANOVA + significant) ───────────────────
        if 'pairwise' in stats_res and stats_res['pairwise']:
            ttk.Label(section_frame, text="Pairwise Comparisons:",
                      font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
            pw_table = ttk.Frame(section_frame)
            pw_table.pack(fill='x', pady=5)

            for col_idx, (hdr, w) in enumerate(
                    [('Comparison', 30), ('p-value', 15), ('Significance', 15)]):
                ttk.Label(pw_table, text=hdr, font=('Arial', 9, 'bold'),
                          relief='solid', borderwidth=1, width=w).grid(
                    row=0, column=col_idx, sticky='ew', padx=1, pady=1)

            for row_idx, (comparison, result) in enumerate(
                    stats_res['pairwise'].items(), start=1):
                comp_text = comparison.replace('_vs_', ' vs ')
                p = result['p_value']
                a = stats_res['alpha']
                if p < 0.001:
                    p_display, sig, fg = 'p < 0.001', '***', 'darkgreen'
                elif p < 0.01:
                    p_display, sig, fg = f'p = {p:.4f}', '**', 'green'
                elif p < a:
                    p_display, sig, fg = f'p = {p:.4f}', '*', 'orange'
                else:
                    p_display, sig, fg = f'p = {p:.4f}', 'ns', 'gray'

                ttk.Label(pw_table, text=comp_text, relief='solid',
                          borderwidth=1, width=30).grid(
                    row=row_idx, column=0, sticky='ew', padx=1, pady=1)
                ttk.Label(pw_table, text=p_display, relief='solid',
                          borderwidth=1, width=15).grid(
                    row=row_idx, column=1, sticky='ew', padx=1, pady=1)
                ttk.Label(pw_table, text=sig, relief='solid', borderwidth=1,
                          width=15, foreground=fg,
                          font=('Arial', 9, 'bold')).grid(
                    row=row_idx, column=2, sticky='ew', padx=1, pady=1)

    def _add_wb_timecourse_stats_section(self, parent_frame, bins_df,
                                         metric, treatments):
        """Two-way ANOVA (treatment × time) + optional per-timepoint post-hoc.

        Time column in bins_df is bin_start_s (seconds).
        """
        from scipy import stats as _scipy_stats

        section_frame = ttk.LabelFrame(
            parent_frame,
            text=f"Time Course Statistics — {metric.replace('_', ' ')}",
            padding=10)
        section_frame.pack(fill='x', padx=10, pady=10)

        alpha = self._stats_alpha_var.get()

        # ── Part A: Two-Way ANOVA ────────────────────────────────────────
        ttk.Label(section_frame,
                  text="═══ Two-Way ANOVA (Treatment × Time) ═══",
                  font=('Arial', 10, 'bold'), foreground='darkblue').pack(
            anchor='w', pady=5)

        try:
            import statsmodels.api as _sm
            from statsmodels.formula.api import ols as _ols

            anova_df = bins_df[['subject', 'treatment', 'bin_start_s', metric]].dropna().copy()
            anova_df['treatment'] = anova_df['treatment'].astype('category')
            anova_df['bin_start_s'] = anova_df['bin_start_s'].astype('category')

            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings('ignore', module='statsmodels')
                model = _ols(
                    f'{metric} ~ C(treatment) + C(bin_start_s) + C(treatment):C(bin_start_s)',
                    data=anova_df).fit()
                anova_table = _sm.stats.anova_lm(model, typ=2)

            main_effects_frame = ttk.Frame(section_frame)
            main_effects_frame.pack(fill='x', pady=5, padx=20)

            headers = ['Source', 'df', 'Sum Sq', 'F-value', 'p-value', 'Significance']
            for i, hdr in enumerate(headers):
                ttk.Label(main_effects_frame, text=hdr,
                          font=('Arial', 9, 'bold'), relief='solid',
                          borderwidth=1, width=13).grid(
                    row=0, column=i, sticky='ew', padx=1, pady=1)

            sources = [
                ('Treatment',       'C(treatment)'),
                ('Time',            'C(bin_start_s)'),
                ('Time×Treatment',  'C(treatment):C(bin_start_s)'),
            ]
            treatment_p   = None
            time_p        = None
            interaction_p = None

            for row_idx, (source_name, source_key) in enumerate(sources, start=1):
                if source_key not in anova_table.index:
                    continue
                row_data = anova_table.loc[source_key]
                df_val  = int(row_data['df'])
                sum_sq  = row_data['sum_sq']
                f_val   = row_data['F']
                p_val   = row_data['PR(>F)']

                if source_name == 'Treatment':
                    treatment_p = p_val
                elif source_name == 'Time':
                    time_p = p_val
                else:
                    interaction_p = p_val

                if p_val < 0.001:
                    p_text, sig, fg = 'p < 0.001', '***', 'darkgreen'
                elif p_val < 0.01:
                    p_text, sig, fg = f'p = {p_val:.4f}', '**', 'green'
                elif p_val < alpha:
                    p_text, sig, fg = f'p = {p_val:.4f}', '*', 'orange'
                else:
                    p_text, sig, fg = f'p = {p_val:.4f}', 'ns', 'gray'

                for col_i, cell in enumerate(
                        [source_name, str(df_val), f'{sum_sq:.2f}',
                         f'{f_val:.3f}', p_text, sig]):
                    kw = {}
                    if col_i == 5:
                        kw = {'foreground': fg, 'font': ('Arial', 9, 'bold')}
                    ttk.Label(main_effects_frame, text=cell,
                              relief='solid', borderwidth=1, width=13,
                              **kw).grid(row=row_idx, column=col_i,
                                         sticky='ew', padx=1, pady=1)

            # Interpretation text
            interp = "Interpretation: "
            if treatment_p is not None and treatment_p < alpha:
                interp += "Treatment groups differ overall. "
            if time_p is not None and time_p < alpha:
                interp += "Metric changes over time. "
            if interaction_p is not None:
                if interaction_p < alpha:
                    interp += "Groups show different time patterns (interaction significant)."
                else:
                    interp += "Groups show similar time patterns (no interaction)."
            ttk.Label(section_frame, text=interp,
                      font=('Arial', 9, 'italic'), foreground='darkblue',
                      wraplength=700).pack(anchor='w', padx=20, pady=5)

        except Exception as e:
            ttk.Label(section_frame,
                      text=f"Two-way ANOVA failed: {e}",
                      foreground='red', font=('Arial', 9, 'italic')).pack(anchor='w')

        ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)

        # ── Part B: Per-Timepoint Post-hoc ──────────────────────────────
        if not self._timecourse_posthoc_var.get():
            ttk.Label(section_frame,
                      text='Enable "Show pairwise post-hoc at each timepoint" to see per-bin results.',
                      foreground='gray', font=('Arial', 9, 'italic')).pack(anchor='w')
            return

        ttk.Label(section_frame,
                  text="═══ Post-hoc Tests (Per Timepoint) ═══",
                  font=('Arial', 10, 'bold'), foreground='darkblue').pack(
            anchor='w', pady=5)
        ttk.Label(section_frame,
                  text="Tests if treatments differ at each individual timepoint. "
                       "Only significant results shown.",
                  font=('Arial', 9, 'italic'), foreground='gray').pack(
            anchor='w', pady=(0, 5))

        time_bins   = sorted(bins_df['bin_start_s'].dropna().unique())
        table_frame = ttk.Frame(section_frame)
        table_frame.pack(fill='x', pady=5)

        col_specs = [('Time (s)', 15), ('Test', 18), ('Statistic', 15),
                     ('p-value', 15), ('Significance', 12)]
        for col_idx, (hdr, w) in enumerate(col_specs):
            ttk.Label(table_frame, text=hdr, font=('Arial', 9, 'bold'),
                      relief='solid', borderwidth=1, width=w).grid(
                row=0, column=col_idx, sticky='ew', padx=1, pady=1)

        row_idx = 1
        n_sig   = 0

        # Count testable bins for Bonferroni correction (2-group case)
        n_tests = 0
        for bin_s in time_bins:
            bin_df = bins_df[bins_df['bin_start_s'] == bin_s]
            _gc = []
            for treat in treatments:
                vals = bin_df[bin_df['treatment'] == treat][metric].dropna().values
                if len(vals) >= 2:
                    _gc.append(vals)
            if len(_gc) >= 2:
                n_tests += 1
        n_tests = max(n_tests, 1)

        for bin_s in time_bins:
            bin_df = bins_df[bins_df['bin_start_s'] == bin_s]
            groups = []
            for treat in treatments:
                vals = bin_df[bin_df['treatment'] == treat][metric].dropna().values
                if len(vals) >= 2:
                    groups.append(vals)
            if len(groups) < 2:
                continue

            if len(groups) == 2:
                t_stat, p_val = _scipy_stats.ttest_ind(groups[0], groups[1], equal_var=False)
                if np.isnan(t_stat) or np.isnan(p_val):
                    continue
                p_val = min(p_val * n_tests, 1.0)  # Bonferroni
                test_name    = f"Welch's t (Bonf. \u00d7{n_tests})"
                stat_display = f't={t_stat:.3f}'
            else:
                f_stat, anova_p = _scipy_stats.f_oneway(*groups)
                if np.isnan(anova_p) or anova_p >= alpha:
                    continue
                try:
                    from scipy.stats import tukey_hsd
                    res_hsd      = tukey_hsd(*groups)
                    p_val        = float(res_hsd.pvalue.min())
                    test_name    = f'Tukey HSD ({len(groups)} groups)'
                    stat_display = f'q(min)={res_hsd.statistic.min():.3f}'
                except (ImportError, AttributeError):
                    min_p = 1.0
                    for gi in range(len(groups)):
                        for gj in range(gi + 1, len(groups)):
                            _, pp = _scipy_stats.ttest_ind(groups[gi], groups[gj], equal_var=False)
                            if not np.isnan(pp):
                                min_p = min(min_p, pp)
                    p_val        = min_p
                    test_name    = f'Bonferroni ({len(groups)} groups)'
                    stat_display = f'p(min)={min_p:.4f}'

            if np.isnan(p_val) or p_val >= alpha:
                continue

            n_sig += 1
            if p_val < 0.001:
                p_display, sig, fg = 'p < 0.001', '***', 'darkgreen'
            elif p_val < 0.01:
                p_display, sig, fg = f'p = {p_val:.4f}', '**', 'green'
            else:
                p_display, sig, fg = f'p = {p_val:.4f}', '*', 'orange'

            cells = [(f'{bin_s:.1f}', 15), (test_name, 18),
                     (stat_display, 15), (p_display, 15), (sig, 12)]
            for col_idx, (cell, w) in enumerate(cells):
                kw = {}
                if col_idx == 4:
                    kw = {'foreground': fg, 'font': ('Arial', 9, 'bold')}
                ttk.Label(table_frame, text=cell, relief='solid',
                          borderwidth=1, width=w, **kw).grid(
                    row=row_idx, column=col_idx, sticky='ew', padx=1, pady=1)
            row_idx += 1

        if n_sig == 0:
            ttk.Label(section_frame,
                      text='No significant differences found at any timepoint.',
                      foreground='gray', font=('Arial', 9, 'italic')).pack(
                anchor='w', pady=5)
        else:
            ttk.Label(section_frame,
                      text=f'Found {n_sig} significant timepoint(s) out of {len(time_bins)} bins.',
                      foreground='darkblue', font=('Arial', 9, 'bold')).pack(
                anchor='w', pady=5)

    def _create_wb_statistics_tab(self, nb, summary_df, bins_df,
                                   treatments, max_time_min):
        """Create a Statistics notebook tab for limb use results.

        nb            — ttk.Notebook to add the tab to
        summary_df    — per-session summary DataFrame (lowercase columns)
        bins_df       — per-bin DataFrame (bin_start_s in seconds)
        treatments    — ordered list of treatment labels
        max_time_min  — maximum time in minutes (from bins_df)
        """
        frame = nb  # Build directly into passed frame (deferred tab)

        max_time_int = max(1, int(max_time_min))

        # ── Control bar (shared across sub-tabs) ─────────────────────────
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill='x', padx=10, pady=(8, 4))

        ttk.Label(ctrl_frame, text="Statistics time window:").pack(side='left')
        stats_time_var = tk.IntVar(value=max_time_int)
        ttk.Spinbox(ctrl_frame, from_=1, to=max_time_int,
                    textvariable=stats_time_var, width=8).pack(side='left', padx=5)
        ttk.Label(ctrl_frame, text="min").pack(side='left')

        status_lbl = ttk.Label(ctrl_frame,
                               text=f"(showing 0\u2013{max_time_int} min)",
                               font=('Arial', 9), foreground='gray')
        status_lbl.pack(side='left', padx=10)

        # ── Category notebook for sub-tabs ────────────────────────────────
        stats_nb = ttk.Notebook(frame)
        stats_nb.pack(fill='both', expand=True)

        # Metrics grouped by category
        hind_metrics = ['WBI_hind', 'SI_hind', 'SBI_hind']
        fore_metrics = (['WBI_fore', 'SI_fore', 'SBI_fore']
                        if self._use_fore_var.get() else [])
        contact_metrics = [f'contact_pct_{r}' for r in self.ROLES]
        brightness_metrics = ['brightness_HL', 'brightness_HR',
                              'brightness_ratio_HL_HR']
        gait_metrics = [
            'stance_dur_HL', 'stance_dur_HR', 'swing_dur_HL', 'swing_dur_HR',
            'stride_dur_HL', 'stride_dur_HR', 'cadence_HL', 'cadence_HR',
            'duty_cycle_HL', 'duty_cycle_HR',
            'stride_len_HL', 'stride_len_HR',
            'step_len_hind', 'step_width_hind',
            'swing_speed_HL', 'swing_speed_HR',
            'stride_cv_HL', 'stride_cv_HR',
            'stance_SI_hind', 'stride_len_SI_hind',
        ]
        loco_metrics = ['total_distance', 'loco_total_distance',
                        'time_moving_s', 'time_moving_pct',
                        'body_speed_mean', 'body_speed_loco']
        phase_metrics = ['phase_HL_HR', 'phase_diagonal',
                         'phase_FL_FR', 'phase_HL_FL', 'phase_HR_FR', 'phase_HL_FR']
        coordination_metrics = ['regularity_index',
                                'print_position_L', 'print_position_R',
                                'support_0paw_pct', 'support_1paw_pct',
                                'support_2paw_pct', 'support_3paw_pct',
                                'support_4paw_pct']
        contour_metrics = [f'paw_area_{r}' for r in ('HL', 'HR')]
        contour_metrics += [f'paw_spread_{r}' for r in ('HL', 'HR')]
        contour_metrics += ['paw_area_ratio_hind']

        # All metrics for CSV export
        all_metrics = (hind_metrics + fore_metrics + contact_metrics
                       + brightness_metrics + gait_metrics
                       + loco_metrics + phase_metrics
                       + coordination_metrics + contour_metrics)

        # Define sub-tab categories
        _stats_categories = [
            ('Limb Use', hind_metrics + fore_metrics + contact_metrics + brightness_metrics),
            ('Gait',           gait_metrics),
            ('Movement',       loco_metrics + phase_metrics + coordination_metrics),
            ('Paw Contour',    contour_metrics),
        ]

        # Content holders per sub-tab (for rebuild on recalculate)
        _cat_holders = {}

        def _build_category_content(holder, cat_metrics, s_df, b_df):
            """Build scrollable statistics content for one category."""
            for w in holder.winfo_children():
                w.destroy()

            canvas = tk.Canvas(holder, bg='white')
            sb_inner = ttk.Scrollbar(holder, orient='vertical',
                                     command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind(
                '<Configure>',
                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
            canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
            canvas.configure(yscrollcommand=sb_inner.set)

            win_val = stats_time_var.get()
            sec = 1
            for metric in cat_metrics:
                if s_df is None or metric not in s_df.columns:
                    continue
                if s_df[metric].dropna().empty:
                    continue
                self._add_wb_stats_section(
                    scrollable_frame,
                    f"{sec}. {metric.replace('_', ' ')} \u2014 Summary",
                    s_df, metric, treatments, 'value')
                sec += 1

            if b_df is not None and not b_df.empty:
                for metric in cat_metrics:
                    if metric not in b_df.columns:
                        continue
                    if b_df[metric].dropna().empty:
                        continue
                    self._add_wb_timecourse_stats_section(
                        scrollable_frame, b_df, metric, treatments)
                    sec += 1

            canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)
            sb_inner.pack(side='right', fill='y')

            export_frame = ttk.Frame(holder)
            export_frame.pack(side='bottom', fill='x', padx=10, pady=5)
            stats_data = {'summary_df': s_df, 'bins_df': b_df,
                          'treatments': treatments, 'metrics': cat_metrics}
            ttk.Button(export_frame, text="\U0001f4ca Export Statistics CSV",
                       command=lambda sd=stats_data: self._export_wb_statistics(sd)
                       ).pack()

        # Create sub-tabs
        for cat_name, cat_metrics in _stats_categories:
            # Only add sub-tab if at least one metric has data
            has_data = any(
                m in summary_df.columns and summary_df[m].notna().any()
                for m in cat_metrics
            )
            if not has_data:
                continue
            cat_frame = ttk.Frame(stats_nb)
            stats_nb.add(cat_frame, text=cat_name)
            holder = ttk.Frame(cat_frame)
            holder.pack(fill='both', expand=True)
            _cat_holders[cat_name] = (holder, cat_metrics)
            _build_category_content(holder, cat_metrics, summary_df, bins_df)

        def on_recalculate():
            win_min = stats_time_var.get()
            win_sec = win_min * 60.0
            filtered_bins = (bins_df[bins_df['bin_start_s'] <= win_sec].copy()
                             if bins_df is not None and not bins_df.empty
                             else bins_df)
            status_lbl.config(text=f"(showing 0\u2013{win_min} min)")
            for cat_name, (holder, cat_metrics) in _cat_holders.items():
                _build_category_content(holder, cat_metrics, summary_df, filtered_bins)

        ttk.Button(ctrl_frame, text="\u21ba Recalculate",
                   command=on_recalculate).pack(side='left')

    def _export_wb_statistics(self, stats_data):
        """Export limb use statistics summary to CSV."""
        s_df       = stats_data.get('summary_df')
        treatments = stats_data.get('treatments', [])
        metrics    = stats_data.get('metrics', [])

        folder = self.app.current_project_folder.get()
        analysis_dir = os.path.join(folder, 'analysis') if folder else ''
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        path = filedialog.asksaveasfilename(
            title="Save Statistics CSV",
            initialdir=analysis_dir or None,
            initialfile=f'wb_statistics_{ts}.csv',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            parent=self)
        if not path:
            return

        rows = []
        blank = {'Section': '', 'Treatment': '', 'N': '', 'Mean': '', 'SD': '',
                 'SEM': '', 'Min': '', 'Max': '', 'Test': '', 'p_value': '',
                 'Significance': ''}

        for metric in metrics:
            if s_df is None or metric not in s_df.columns:
                continue
            rows.append({**blank, 'Section': f'Summary \u2014 {metric}'})

            for treat in treatments:
                vals = s_df[s_df['treatment'] == treat][metric].dropna().values
                if len(vals) == 0:
                    continue
                n   = len(vals)
                sd  = np.std(vals, ddof=1) if n > 1 else 0.0
                sem = sd / np.sqrt(n)
                rows.append({
                    'Section':      f'Summary \u2014 {metric}',
                    'Treatment':    treat,
                    'N':            n,
                    'Mean':         f'{np.mean(vals):.4f}',
                    'SD':           f'{sd:.4f}',
                    'SEM':          f'{sem:.4f}',
                    'Min':          f'{np.min(vals):.4f}',
                    'Max':          f'{np.max(vals):.4f}',
                    'Test':         '',
                    'p_value':      '',
                    'Significance': '',
                })

            # Statistical test row
            data_by_t = {t: s_df[s_df['treatment'] == t][metric].dropna().values
                         for t in treatments}
            res = self._perform_wb_statistical_test(data_by_t, treatments)
            if res:
                p = res['p_value']
                a = res['alpha']
                sig = ('***' if p < 0.001 else
                       '**'  if p < 0.01  else
                       '*'   if p < a     else 'ns')
                rows.append({**blank,
                             'Section':      f'Test \u2014 {metric}',
                             'Test':         res['test_type'],
                             'p_value':      f'{p:.4f}',
                             'Significance': sig})

            rows.append(blank.copy())

        pd.DataFrame(rows).to_csv(path, index=False)
        self._log_ui(f"Statistics saved: {os.path.basename(path)}")
