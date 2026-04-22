"""
evaluation_tab.py — PixelPaws Evaluation Tab
=============================================
Standalone module for evaluating trained classifiers against labelled test data.

Provides the EvaluationTab class, a ttk.Frame subclass that is dropped into the
main PixelPawsGUI notebook.  All evaluation logic (session discovery, feature
extraction/caching, prediction, scoring, plotting, SHAP analysis) lives here so
that PixelPaws_GUI.py only needs to instantiate it.

Usage (inside PixelPaws_GUI.py):
    from evaluation_tab import EvaluationTab
    ...
    eval_frame = ttk.Frame(self.notebook)
    self.notebook.add(eval_frame, text="📊 Evaluate")
    self.evaluation_tab = EvaluationTab(eval_frame, self)
    self.evaluation_tab.pack(fill='both', expand=True)
"""

import os
import glob
import pickle
import hashlib
import threading
import traceback

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from feature_cache import FeatureCacheManager
    _FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FeatureCacheManager = None
    _FEATURE_CACHE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper — applies the same bout-filtering rules as the prediction pipeline
# ---------------------------------------------------------------------------
def _apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap):
    """Apply bout filtering: min-bout removal and gap bridging."""
    y_filtered = y_pred.copy()

    # --- min_bout: remove bouts shorter than threshold ---
    in_bout = False
    bout_start = 0
    for i in range(len(y_filtered)):
        if y_filtered[i] == 1 and not in_bout:
            bout_start = i
            in_bout = True
        elif y_filtered[i] == 0 and in_bout:
            if (i - bout_start) < min_bout:
                y_filtered[bout_start:i] = 0
            in_bout = False
    if in_bout and (len(y_filtered) - bout_start) < min_bout:
        y_filtered[bout_start:] = 0

    # --- max_gap: bridge short gaps between bouts ---
    if max_gap > 0:
        i = 0
        while i < len(y_filtered):
            if y_filtered[i] == 1:
                gap_start = i + 1
                while gap_start < len(y_filtered) and y_filtered[gap_start] == 0:
                    gap_start += 1
                gap_len = gap_start - i - 1
                if 0 < gap_len <= max_gap and gap_start < len(y_filtered):
                    if y_filtered[gap_start] == 1:
                        y_filtered[i + 1:gap_start] = 1
                i = gap_start
            else:
                i += 1

    return y_filtered


def fit_hmm_transitions(y):
    """Fit a two-state HMM transition matrix from a binary label sequence.

    Uses Laplace smoothing (+1 to each count) to avoid zero-probability
    transitions on short recordings.

    Returns
    -------
    log_trans : np.ndarray shape (2, 2) — log P(next_state | current_state)
    log_prior : np.ndarray shape (2,)   — log P(state at t=0)
    """
    y = np.asarray(y, dtype=int)
    trans = np.ones((2, 2), dtype=float)   # Laplace smoothing
    for t in range(len(y) - 1):
        s, s_ = int(y[t]), int(y[t + 1])
        if s in (0, 1) and s_ in (0, 1):
            trans[s, s_] += 1
    trans /= trans.sum(axis=1, keepdims=True)
    log_trans = np.log(trans)
    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    log_prior = np.log(np.array([1 - prevalence, prevalence]))
    return log_trans, log_prior


def viterbi_smooth(probas, log_trans, log_prior):
    """Two-state Viterbi MAP decoder over per-frame behavior probabilities.

    Emission model: P(obs=p | state=1) = p,  P(obs=p | state=0) = 1-p.
    This is consistent with calibrated classifier output and does not
    require a hard threshold — the label sequence is determined entirely
    by the HMM transition prior and the raw probability stream.

    Parameters
    ----------
    probas    : array-like  — raw per-frame P(behavior), shape (n,)
    log_trans : np.ndarray shape (2, 2) — from fit_hmm_transitions
    log_prior : np.ndarray shape (2,)   — from fit_hmm_transitions

    Returns
    -------
    np.ndarray of int (0/1), shape (n,)
    """
    probas = np.clip(np.asarray(probas, dtype=float), 1e-10, 1 - 1e-10)
    log_emit = np.column_stack([np.log(1 - probas), np.log(probas)])  # (n, 2)
    n = len(probas)
    vt = np.full((n, 2), -np.inf)
    bp = np.zeros((n, 2), dtype=np.int8)
    vt[0] = log_prior + log_emit[0]
    for t in range(1, n):
        for s in range(2):
            scores = vt[t - 1] + log_trans[:, s]
            best = int(np.argmax(scores))
            bp[t, s] = best
            vt[t, s] = scores[best] + log_emit[t, s]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(vt[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def count_bouts(y_pred: np.ndarray, fps: float) -> dict:
    """
    Count bouts in a binary prediction array and return summary statistics.

    Parameters
    ----------
    y_pred : np.ndarray
        Binary array (0/1) of per-frame predictions.
    fps : float
        Frames per second — used to convert frame counts to seconds.

    Returns
    -------
    dict with keys:
        n_bouts         int    number of bouts
        bouts           list   [{'start': int, 'end': int,
                                  'duration_frames': int,
                                  'duration_sec': float}, ...]
        mean_dur_sec    float  mean bout duration in seconds (0 if no bouts)
        median_dur_sec  float  median bout duration in seconds
        min_dur_sec     float  shortest bout in seconds
        max_dur_sec     float  longest bout in seconds
    """
    bouts = []
    in_bout = False
    bout_start = 0

    for i, val in enumerate(y_pred):
        if val == 1 and not in_bout:
            bout_start = i
            in_bout = True
        elif val == 0 and in_bout:
            dur = i - bout_start
            bouts.append({'start': bout_start, 'end': i - 1,
                          'duration_frames': dur,
                          'duration_sec': dur / fps if fps else 0.0})
            in_bout = False

    if in_bout:
        dur = len(y_pred) - bout_start
        bouts.append({'start': bout_start, 'end': len(y_pred) - 1,
                      'duration_frames': dur,
                      'duration_sec': dur / fps if fps else 0.0})

    dur_secs = [b['duration_sec'] for b in bouts]
    return {
        'n_bouts':        len(bouts),
        'bouts':          bouts,
        'mean_dur_sec':   float(np.mean(dur_secs))   if dur_secs else 0.0,
        'median_dur_sec': float(np.median(dur_secs)) if dur_secs else 0.0,
        'min_dur_sec':    float(np.min(dur_secs))    if dur_secs else 0.0,
        'max_dur_sec':    float(np.max(dur_secs))    if dur_secs else 0.0,
    }


# ---------------------------------------------------------------------------
# Shared session discovery — used by training, evaluation, active learning
# ---------------------------------------------------------------------------
def find_session_triplets(
    folder: str,
    video_ext: str = '.mp4',
    prefer_filtered: bool = True,
    require_labels: bool = True,
    recursive: bool = False,
) -> list:
    """
    Walk *folder* and return a list of matched (video, DLC, labels) session dicts.

    The function starts from DLC .h5 files (same strategy as training) and
    resolves the video and labels file for each one by searching a consistent
    set of candidate locations.  Missing labels are allowed when
    ``require_labels=False`` (prediction / active-learning pre-screening).

    Parameters
    ----------
    folder : str
        Project or video folder to scan.  Both flat and sub-folder layouts are
        supported::

            flat:        folder/*.mp4, folder/*.h5, folder/*_labels.csv
            structured:  folder/Videos/*.mp4, folder/Videos/*.h5,
                         folder/Labels/*_labels.csv

    video_ext : str
        Primary video extension to look for (e.g. ``'.mp4'``, ``'.avi'``).
        The upper-case variant is always checked as well.

    prefer_filtered : bool
        When multiple DLC files match a video, prefer the one whose name
        contains ``'filtered'``.

    require_labels : bool
        If *True* (default), sessions without a labels file are skipped.
        If *False*, the ``'labels'`` key will be *None* for those sessions.

    Returns
    -------
    list of dict
        Each dict contains:

        ``session_name``  base name of the video (no extension, no DLC suffix)
        ``video``         absolute path to the video file
        ``dlc``           absolute path to the DLC .h5 pose file
        ``labels``        absolute path to the labels CSV, or *None*
        ``video_dir``     directory containing the video
        ``project_dir``   parent of that directory (useful for cache paths)
    """
    folder = os.path.abspath(folder)

    # ── Determine video / DLC search root ────────────────────────────────────
    videos_sub = os.path.join(folder, 'Videos')
    _videos_sub_has_h5 = os.path.isdir(videos_sub) and bool(
        glob.glob(os.path.join(videos_sub, '*.h5')) or
        glob.glob(os.path.join(videos_sub, '**', '*.h5'), recursive=True)
    )
    if _videos_sub_has_h5:
        search_root = videos_sub
        project_root = folder
    else:
        search_root = folder
        project_root = os.path.dirname(folder)

    # ── Find all DLC .h5 files ────────────────────────────────────────────────
    if recursive:
        dlc_files = []
        for dirpath, _, fnames in os.walk(search_root):
            for fn in sorted(fnames):
                if fn.lower().endswith('.h5'):
                    dlc_files.append(os.path.join(dirpath, fn))
    else:
        dlc_files = glob.glob(os.path.join(search_root, '*.h5'))
    if not dlc_files:
        return []

    sessions = []
    seen_bases = set()

    for dlc_path in sorted(dlc_files):
        dlc_name = os.path.basename(dlc_path)

        # ── Derive video base name ────────────────────────────────────────────
        base = dlc_name.split('DLC')[0] if 'DLC' in dlc_name else os.path.splitext(dlc_name)[0]
        # Strip labelling suffixes that sometimes end up in the DLC filename
        for _sfx in ('_Labels', '_labels', '_LABELS', '_perframe'):
            if base.endswith(_sfx):
                base = base[: -len(_sfx)]
                break

        seen_key = (os.path.dirname(dlc_path), base) if recursive else base
        if seen_key in seen_bases:
            continue  # deduplicate when multiple DLC files share a base

        # ── Resolve video ─────────────────────────────────────────────────────
        video_search_dir = os.path.dirname(dlc_path) if recursive else search_root
        video_path = None
        for ext in [video_ext, video_ext.upper(), '.mp4', '.avi', '.MP4', '.AVI']:
            candidate = os.path.join(video_search_dir, base + ext)
            if os.path.isfile(candidate):
                video_path = candidate
                break
        if not video_path:
            continue  # no video → skip

        video_dir = os.path.dirname(video_path)

        # ── Resolve DLC file (prefer filtered if requested) ───────────────────
        # There may be multiple DLC files for this base; pick the best one.
        dlc_dir = os.path.dirname(dlc_path) if recursive else search_root
        dlc_candidates = glob.glob(os.path.join(dlc_dir, f'{base}DLC*.h5'))
        if not dlc_candidates:
            dlc_candidates = [dlc_path]
        filtered_dlc = [f for f in dlc_candidates if 'filtered' in f.lower()]
        if prefer_filtered and filtered_dlc:
            best_dlc = filtered_dlc[0]
        elif dlc_candidates:
            best_dlc = dlc_candidates[0]
        else:
            best_dlc = dlc_path

        # ── Resolve labels file ───────────────────────────────────────────────
        label_candidates = [
            # Project-level behavior_labels/ folder (canonical new location)
            os.path.join(project_root, 'behavior_labels', f'{base}_labels.csv'),
            os.path.join(project_root, 'behavior_labels', f'{base}.csv'),
            # Same folder as video (canonical location)
            os.path.join(video_dir, f'{base}_labels.csv'),
            os.path.join(video_dir, f'{base}_Labels.csv'),
            os.path.join(video_dir, f'{base}_perframe.csv'),   # legacy BORIS output
            # Sibling labels folder (legacy)
            os.path.join(project_root, 'labels',  f'{base}_labels.csv'),
            os.path.join(project_root, 'labels',  f'{base}_Labels.csv'),
            os.path.join(project_root, 'Labels',  f'{base}_labels.csv'),
            os.path.join(project_root, 'Labels',  f'{base}.csv'),
            os.path.join(project_root, 'Targets', f'{base}.csv'),
            os.path.join(project_root, 'targets', f'{base}.csv'),
            os.path.join(project_root, 'targets', f'{base}_labels.csv'),
        ]
        labels_path = next((p for p in label_candidates if os.path.isfile(p)), None)

        if require_labels and labels_path is None:
            continue  # no labels and they're required → skip

        seen_bases.add(seen_key)
        if recursive:
            rel = os.path.relpath(os.path.dirname(dlc_path), search_root)
            session_display = base if rel == '.' else f"{rel}/{base}".replace('\\', '/')
        else:
            session_display = base
        sessions.append({
            'session_name': session_display,
            'video':        video_path,
            'dlc':          best_dlc,
            'labels':       labels_path,
            'video_dir':    video_dir,
            'project_dir':  project_root,
            # Aliases kept for backward compatibility with training code
            'video_path':   video_path,
            'pose_path':    best_dlc,
            'target_path':  labels_path,
        })

    return sessions


# ---------------------------------------------------------------------------
# EvaluationTab
# ---------------------------------------------------------------------------
class EvaluationTab(ttk.Frame):
    """
    Full evaluation tab: load a classifier, point at a test folder, run.

    Parameters
    ----------
    parent : tk widget
        The ttk.Frame (or any container) that this tab lives inside.
    app : PixelPawsGUI
        Reference to the main application, used for:
            • app.root  — the Tk root (for after() and Toplevel dialogs)
            • app.theme — colour theme object (optional, gracefully absent)
    """

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app  = app
        self.root = app.root

        # StringVars / BooleanVars set up before building the UI
        self.eval_classifier_options = {}
        self.eval_classifier_path = tk.StringVar()
        self.eval_test_folder     = tk.StringVar()
        self.eval_dlc_config_path = tk.StringVar()
        self.eval_generate_plots  = tk.BooleanVar(value=True)
        self.eval_save_predictions= tk.BooleanVar(value=True)
        self.eval_detailed_report = tk.BooleanVar(value=True)
        self.eval_apply_bout_filter = tk.BooleanVar(value=True)
        self.eval_smoothing_mode    = tk.StringVar(value='bout_filters')
        self._eval_cancel_flag = threading.Event()
        self._optimized_params = None   # set after CV optimization completes

        # Custom parameter overrides
        self.eval_use_custom_params = tk.BooleanVar(value=False)
        self.eval_custom_threshold  = tk.DoubleVar(value=0.5)
        self.eval_custom_min_bout   = tk.IntVar(value=1)
        self.eval_custom_min_after  = tk.IntVar(value=1)
        self.eval_custom_max_gap    = tk.IntVar(value=0)

        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        """Construct the full evaluation tab layout."""
        # Scrollable canvas
        canvas   = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        sf = ttk.Frame(canvas)  # scrollable_frame
        sf.bind('<Configure>',
                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=sf, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # ── Classifier selection ──────────────────────────────────────────
        sel = ttk.LabelFrame(sf, text='Classifier Selection', padding=10)
        sel.pack(fill='x', padx=5, pady=5)

        ttk.Label(sel, text='Classifier File:').grid(row=0, column=0, sticky='w', pady=2)
        self.eval_classifier_combo = ttk.Combobox(
            sel, textvariable=self.eval_classifier_path, width=46, state='readonly')
        self.eval_classifier_combo.grid(row=0, column=1, padx=5, pady=2)
        self.eval_classifier_combo.bind('<<ComboboxSelected>>', self._on_classifier_selected)
        ttk.Button(sel, text='🔄', width=3,
                   command=self.refresh_classifiers).grid(row=0, column=2, pady=2)
        ttk.Button(sel, text='📁', width=3,
                   command=self._browse_classifier).grid(row=0, column=3, pady=2)
        ttk.Button(sel, text='📋 Load Classifier Info',
                   command=self._load_classifier_info).grid(row=1, column=1, sticky='w', pady=5)

        # ── Classifier info display ───────────────────────────────────────
        info_frame = ttk.LabelFrame(sf, text='Classifier Information', padding=10)
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.eval_info_text = scrolledtext.ScrolledText(info_frame, height=8, wrap=tk.WORD)
        self.eval_info_text.pack(fill='both', expand=True)

        # ── Test data ─────────────────────────────────────────────────────
        tst = ttk.LabelFrame(sf, text='Test Data', padding=10)
        tst.pack(fill='x', padx=5, pady=5)

        ttk.Label(tst, text='Test Video Folder:').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(tst, textvariable=self.eval_test_folder, width=50).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(tst, text='📁 Browse',
                   command=self._browse_test_folder).grid(row=0, column=2, pady=2)

        ttk.Label(tst, text='Video Extension:').grid(row=1, column=0, sticky='w', pady=2)
        self.eval_video_ext = ttk.Combobox(
            tst, values=['.mp4', '.avi', '.MP4', '.AVI'], width=10)
        self.eval_video_ext.set('.mp4')
        self.eval_video_ext.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(tst, text='DLC Config (optional):').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(tst, textvariable=self.eval_dlc_config_path, width=50).grid(
            row=2, column=1, padx=5, pady=2)
        ttk.Button(tst, text='📁 Browse',
                   command=self._browse_dlc_config).grid(row=2, column=2, pady=2)
        ttk.Label(tst,
                  text='(For DLC crop offset correction — same config used during training)',
                  font=('Arial', 8), foreground='gray').grid(row=3, column=1, sticky='w', padx=5)

        ttk.Button(tst, text='🔍 Scan Test Sessions',
                   command=self.scan_sessions).grid(row=4, column=1, sticky='w', pady=5)

        # ── Options ───────────────────────────────────────────────────────
        opt = ttk.LabelFrame(sf, text='Evaluation Options', padding=10)
        opt.pack(fill='x', padx=5, pady=5)

        ttk.Checkbutton(opt, text='Generate confusion matrix and performance plots',
                        variable=self.eval_generate_plots).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Checkbutton(opt, text='Save predictions for all test videos',
                        variable=self.eval_save_predictions).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Checkbutton(opt, text='Generate detailed per-video performance report',
                        variable=self.eval_detailed_report).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Checkbutton(opt, text='Apply classifier bout filtering before scoring',
                        variable=self.eval_apply_bout_filter).grid(row=3, column=0, sticky='w', pady=2)

        # Smoothing mode selector (replaces / extends bout-filter checkbox)
        smooth_frame = ttk.Frame(opt)
        smooth_frame.grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 2))
        ttk.Label(smooth_frame, text="Smoothing:").pack(side='left')
        for _lbl, _val in [("Bout filters", "bout_filters"),
                            ("HMM Viterbi", "hmm_viterbi"),
                            ("None", "none")]:
            ttk.Radiobutton(smooth_frame, text=_lbl,
                            variable=self.eval_smoothing_mode,
                            value=_val).pack(side='left', padx=(8, 0))

        # Custom parameter overrides
        ttk.Checkbutton(opt, text='Use custom parameters (override classifier defaults)',
                        variable=self.eval_use_custom_params,
                        command=self._toggle_custom_params).grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(8, 2))

        ttk.Label(opt, text='Threshold:').grid(row=5, column=0, sticky='w', padx=(20, 5), pady=2)
        self._spin_threshold = ttk.Spinbox(
            opt, from_=0.0, to=1.0, increment=0.01,
            textvariable=self.eval_custom_threshold, width=8, state='disabled')
        self._spin_threshold.grid(row=5, column=1, sticky='w', pady=2)

        ttk.Label(opt, text='Min bout (frames):').grid(row=6, column=0, sticky='w', padx=(20, 5), pady=2)
        self._spin_min_bout = ttk.Spinbox(
            opt, from_=1, to=999, increment=1,
            textvariable=self.eval_custom_min_bout, width=8, state='disabled')
        self._spin_min_bout.grid(row=6, column=1, sticky='w', pady=2)

        ttk.Label(opt, text='Min after bout (frames):').grid(row=7, column=0, sticky='w', padx=(20, 5), pady=2)
        self._spin_min_after = ttk.Spinbox(
            opt, from_=1, to=999, increment=1,
            textvariable=self.eval_custom_min_after, width=8, state='disabled')
        self._spin_min_after.grid(row=7, column=1, sticky='w', pady=2)

        ttk.Label(opt, text='Max gap (frames):').grid(row=8, column=0, sticky='w', padx=(20, 5), pady=2)
        self._spin_max_gap = ttk.Spinbox(
            opt, from_=0, to=999, increment=1,
            textvariable=self.eval_custom_max_gap, width=8, state='disabled')
        self._spin_max_gap.grid(row=8, column=1, sticky='w', pady=2)

        # ── Action buttons ────────────────────────────────────────────────
        act = ttk.Frame(sf)
        act.pack(fill='x', padx=5, pady=10)

        self._eval_run_btn = ttk.Button(act, text='▶ RUN EVALUATION',
                   command=self.run_evaluation,
                   style='Accent.TButton')
        self._eval_run_btn.pack(side='left', padx=5)
        self._eval_stop_btn = ttk.Button(act, text='■  Stop',
                   command=self._cancel_evaluation,
                   state='disabled')
        self._eval_stop_btn.pack(side='left', padx=5)
        ttk.Button(act, text='🎯 Optimize Parameters',
                   command=self._run_cv_optimization).pack(side='left', padx=5)
        self._save_params_btn = ttk.Button(act, text='💾 Save Params to Classifier',
                   command=self._save_optimized_params_to_classifier,
                   state='disabled')
        self._save_params_btn.pack(side='left', padx=5)
        ttk.Button(act, text='🔬 SHAP Analysis',
                   command=self.run_shap_analysis).pack(side='left', padx=5)

        # ── Results ───────────────────────────────────────────────────────
        res = ttk.LabelFrame(sf, text='Evaluation Results', padding=5)
        res.pack(fill='both', expand=True, padx=5, pady=5)
        self.eval_results_text = scrolledtext.ScrolledText(res, height=15, wrap=tk.WORD)
        self.eval_results_text.pack(fill='both', expand=True)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mousewheel scrolling
        from ui_utils import bind_mousewheel
        bind_mousewheel(canvas)

    # --------------------------------------------------------------- Browse --

    def refresh_classifiers(self):
        """Populate the classifier dropdown from project + global classifiers."""
        from user_config import get_global_classifiers_folder
        pf = getattr(self.app, 'current_project_folder', None)
        self.eval_classifier_options = {}

        # Local project classifiers
        clf_dir = os.path.join(pf.get() if pf else '', 'classifiers')
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    self.eval_classifier_options[f"[Project] {f}"] = os.path.join(clf_dir, f)

        # Global classifiers library
        gcf = get_global_classifiers_folder()
        if os.path.isdir(gcf):
            for f in sorted(os.listdir(gcf)):
                if f.endswith('.pkl'):
                    self.eval_classifier_options[f"[Global] {f}"] = os.path.join(gcf, f)

        if hasattr(self, 'eval_classifier_combo'):
            self.eval_classifier_combo['values'] = list(self.eval_classifier_options.keys())

    def _on_classifier_selected(self, event=None):
        """Update the full path StringVar when a dropdown item is chosen."""
        name = self.eval_classifier_combo.get()
        if name in self.eval_classifier_options:
            clf_path = self.eval_classifier_options[name]
            self.eval_classifier_path.set(clf_path)
            try:
                from prediction_pipeline import check_classifier_portability
                for _w in check_classifier_portability(clf_path):
                    print(f"⚠️  [{name}] {_w}")
            except Exception:
                pass
        self._optimized_params = None
        if hasattr(self, '_save_params_btn'):
            self._save_params_btn.config(state='disabled')

    def _browse_classifier(self):
        p = filedialog.askopenfilename(
            title='Select Classifier File',
            filetypes=[('Pickle files', '*.pkl'), ('All files', '*.*')])
        if p:
            self.eval_classifier_path.set(p)

    def _browse_test_folder(self):
        d = filedialog.askdirectory(title='Select Test Data Folder')
        if d:
            self.eval_test_folder.set(d)
            self.refresh_classifiers()

    def _browse_dlc_config(self):
        p = filedialog.askopenfilename(
            title='Select DLC Config File',
            filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')])
        if p:
            self.eval_dlc_config_path.set(p)

    # ------------------------------------------------ Custom params toggle ----

    def _toggle_custom_params(self):
        """Enable/disable the custom parameter spinboxes."""
        state = 'normal' if self.eval_use_custom_params.get() else 'disabled'
        for spin in (self._spin_threshold, self._spin_min_bout,
                     self._spin_min_after, self._spin_max_gap):
            spin.config(state=state)

    # ----------------------------------------- Save optimized params --------

    def _save_optimized_params_to_classifier(self):
        """Save the optimized post-processing params back into the loaded classifier .pkl."""
        if self._optimized_params is None:
            messagebox.showwarning('No Optimized Params', 'Run optimization first.')
            return

        clf_path = self.eval_classifier_path.get()
        if not clf_path or not os.path.isfile(clf_path):
            messagebox.showerror('Classifier Not Found', f'Cannot find: {clf_path}')
            return

        if not messagebox.askyesno('Update Classifier',
                f'Overwrite post-processing params in:\n{os.path.basename(clf_path)}\n\n'
                f'Threshold: {self._optimized_params["best_thresh"]:.3f}\n'
                f'Min Bout: {self._optimized_params["min_bout"]}\n'
                f'Min After: {self._optimized_params["min_after_bout"]}\n'
                f'Max Gap: {self._optimized_params["max_gap"]}\n\n'
                'Original params will be overwritten.'):
            return

        try:
            import pickle
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)

            clf_data['best_thresh']    = self._optimized_params['best_thresh']
            clf_data['min_bout']       = self._optimized_params['min_bout']
            clf_data['min_after_bout'] = self._optimized_params['min_after_bout']
            clf_data['max_gap']        = self._optimized_params['max_gap']

            from PixelPaws_GUI import _atomic_pickle_save
            _atomic_pickle_save(clf_data, clf_path)

            self.eval_results_text.insert(tk.END,
                f'\n✓ Saved optimized params to {os.path.basename(clf_path)}\n')
            self.eval_results_text.see(tk.END)

        except Exception as e:
            messagebox.showerror('Save Failed', str(e))

    # -------------------------------------------------- Classifier info ------

    def _load_classifier_info(self):
        """Load and display metadata from the selected classifier pickle."""
        clf_path = self.eval_classifier_path.get()
        if not clf_path or not os.path.isfile(clf_path):
            messagebox.showwarning('No File', 'Please select a valid classifier file.')
            return
        try:
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)

            lines = ['=== Classifier Information ===\n',
                     f'File: {os.path.basename(clf_path)}\n']

            for key, label in [('Behavior_type',   'Behavior'),
                                ('best_thresh',     'Best Threshold'),
                                ('min_bout',        'Min Bout (frames)'),
                                ('min_after_bout',  'Min After Bout (frames)'),
                                ('max_gap',         'Max Gap (frames)'),
                                ('bp_pixbrt_list',  'Pixel Brightness Body Parts'),
                                ('pix_threshold',   'Pixel Threshold')]:
                if key in clf_data:
                    val = clf_data[key]
                    if isinstance(val, float):
                        val = f'{val:.3f}'
                    lines.append(f'{label}: {val}\n')

            if 'clf_model' in clf_data:
                model = clf_data['clf_model']
                lines.append(f'\nModel Type: {type(model).__name__}\n')
                if hasattr(model, 'feature_names_in_'):
                    lines.append(f'Number of Features: {len(model.feature_names_in_)}\n')
                if hasattr(model, 'n_estimators'):
                    lines.append(f'n_estimators: {model.n_estimators}\n')

            self.eval_info_text.delete('1.0', tk.END)
            self.eval_info_text.insert('1.0', ''.join(lines))

            # Auto-populate custom parameter spinboxes with classifier defaults
            self.eval_custom_threshold.set(clf_data.get('best_thresh', 0.5))
            self.eval_custom_min_bout.set(clf_data.get('min_bout', 1))
            self.eval_custom_min_after.set(clf_data.get('min_after_bout', 1))
            self.eval_custom_max_gap.set(clf_data.get('max_gap', 0))

        except Exception as e:
            messagebox.showerror('Error', f'Failed to load classifier:\n{e}')

    # -------------------------------------------------- Session discovery ----

    def _find_sessions(self):
        """
        Walk the test folder and return a list of complete session dicts.
        Delegates to find_session_triplets() for consistent discovery across tabs.
        """
        folder = self.eval_test_folder.get()
        ext    = self.eval_video_ext.get() if self.eval_video_ext.get() else '.mp4'
        return find_session_triplets(folder, video_ext=ext,
                                     prefer_filtered=True, require_labels=True)

    def scan_sessions(self):
        """Scan the test folder and report found sessions to the results box."""
        if not self.eval_test_folder.get():
            messagebox.showwarning('No Folder', 'Please select a test data folder first.')
            return

        self.eval_results_text.delete('1.0', tk.END)
        self.eval_results_text.insert(tk.END, 'Scanning test folder…\n\n')

        sessions = self._find_sessions()

        if not sessions:
            self.eval_results_text.insert(tk.END,
                '✗ No complete sessions found.\n\n'
                'Each session needs:\n'
                '  • A video file (.mp4 or .avi)\n'
                '  • A matching DLC .h5 file (same base name)\n'
                '  • A matching _labels.csv file\n\n'
                'Label files are searched as:\n'
                '  <video_name>_labels.csv   (same folder)\n'
                '  ../labels/<video_name>_labels.csv\n')
            return

        self.eval_results_text.insert(
            tk.END, f'Found {len(sessions)} session(s) ready for evaluation:\n\n')
        for i, s in enumerate(sessions, 1):
            self.eval_results_text.insert(tk.END,
                f'  [{i}] {os.path.basename(s["video"])}\n'
                f'       DLC:    {os.path.basename(s["dlc"])}\n'
                f'       Labels: {os.path.basename(s["labels"])}\n\n')

    # --------------------------------------------------------- Evaluation ----

    def _cancel_evaluation(self):
        self._eval_cancel_flag.set()
        self._log('\nCancelling evaluation…\n')

    def run_evaluation(self):
        """Entry point — validates inputs, then starts the evaluation thread."""
        if not self.eval_classifier_path.get():
            messagebox.showwarning('No Classifier', 'Please select a classifier file.')
            return
        if not self.eval_test_folder.get():
            messagebox.showwarning('No Test Data', 'Please select a test data folder.')
            return
        self._eval_cancel_flag.clear()
        self._eval_run_btn.config(state='disabled')
        self._eval_stop_btn.config(state='normal')

        def _run():
            try:
                self._evaluation_thread()
            finally:
                self._safe_after(lambda: self._eval_run_btn.config(state='normal'))
                self._safe_after(lambda: self._eval_stop_btn.config(state='disabled'))

        threading.Thread(target=_run, daemon=True).start()

    def _log(self, msg):
        """Thread-safe append to the results text box."""
        self._safe_after(lambda m=msg: (
            self.eval_results_text.insert(tk.END, m + '\n'),
            self.eval_results_text.see(tk.END)
        ))

    def _safe_after(self, callback):
        """Schedule *callback* on the main thread, swallowing errors if the
        window has been destroyed (e.g. user closed it during evaluation)."""
        try:
            self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _evaluation_thread(self):
        """
        Full evaluation pipeline (runs in background thread):
          1. Load classifier
          2. Read DLC crop config (if provided)
          3. Discover sessions
          4. For each session: extract/cache features → predict → score
          5. Aggregate metrics and optional plots
        """
        # Lazy import of module-level helpers from the main GUI module to avoid
        # circular imports at load time.  By the time a user clicks Run, the
        # parent module is fully initialised.
        try:
            import PixelPaws_GUI as _gui
            _extract   = _gui.PixelPaws_ExtractFeatures
            _predict   = _gui.predict_with_xgboost
            _clean_bp  = _gui.clean_bodyparts_list
            _auto_bp   = _gui.auto_detect_bodyparts_from_model
        except (ImportError, AttributeError) as e:
            self._log(f'✗ Could not import PixelPaws_GUI helpers: {e}')
            self._log('  Make sure evaluation_tab.py is in the same folder as PixelPaws_GUI.py')
            return

        try:
            self._safe_after(lambda: self.eval_results_text.delete('1.0', tk.END))
            self._log('=' * 60)
            self._log('PixelPaws Classifier Evaluation')
            self._log('=' * 60)

            clf_path          = self.eval_classifier_path.get()
            dlc_config_path   = self.eval_dlc_config_path.get()
            apply_bout_filter = self.eval_apply_bout_filter.get()

            # ── Load classifier ──────────────────────────────────────────
            self._log(f'\nLoading classifier: {os.path.basename(clf_path)}')
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)

            clf_data['bp_include_list'] = _clean_bp(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list']  = _clean_bp(clf_data.get('bp_pixbrt_list',  []))
            clf_data = _auto_bp(clf_data, verbose=False)

            model         = clf_data['clf_model']
            best_thresh   = clf_data['best_thresh']
            behavior_name = clf_data.get('Behavior_type', 'Behavior')
            min_bout      = clf_data.get('min_bout',       1)
            min_after     = clf_data.get('min_after_bout', 1)
            max_gap       = clf_data.get('max_gap',        0)

            # Apply user overrides if enabled
            if self.eval_use_custom_params.get():
                best_thresh = self.eval_custom_threshold.get()
                min_bout    = self.eval_custom_min_bout.get()
                min_after   = self.eval_custom_min_after.get()
                max_gap     = self.eval_custom_max_gap.get()
                self._log(f'  [Custom overrides] threshold={best_thresh:.3f}, '
                          f'min_bout={min_bout}, min_after={min_after}, max_gap={max_gap}')

            self._log(f'  Behavior:  {behavior_name}')
            self._log(f'  Threshold: {best_thresh:.3f}')
            if apply_bout_filter:
                self._log(
                    f'  Bout filter: min_bout={min_bout}, '
                    f'min_after={min_after}, max_gap={max_gap}')

            # ── DLC crop offset ──────────────────────────────────────────
            crop_x, crop_y = 0, 0
            if dlc_config_path and os.path.isfile(dlc_config_path):
                try:
                    import yaml
                    with open(dlc_config_path) as f:
                        config = yaml.safe_load(f)
                    if config.get('cropping', False):
                        crop_x = config.get('x1', 0)
                        crop_y = config.get('y1', 0)
                        self._log(f'  DLC crop offset: x+{crop_x}, y+{crop_y}')
                except ImportError:
                    self._log('  ⚠️  PyYAML not installed — cannot read DLC config')
                except Exception as e:
                    self._log(f'  ⚠️  Could not read DLC config: {e}')

            # ── Find sessions ────────────────────────────────────────────
            self._log('\nScanning test folder…')
            sessions = self._find_sessions()
            if not sessions:
                self._log('✗ No complete sessions found (need video + DLC .h5 + _labels.csv).')
                self._safe_after(lambda: messagebox.showerror(
                    'No Sessions',
                    'No complete test sessions found.\n\n'
                    'Each session needs:\n'
                    '  • A video (.mp4 / .avi)\n'
                    '  • A matching DLC .h5 file\n'
                    '  • A matching _labels.csv file'))
                return

            self._log(f'Found {len(sessions)} session(s).\n')

            # ── Per-session loop ─────────────────────────────────────────
            all_y_true        = []
            all_y_pred        = []
            per_video_results = []
            project_folder = getattr(self.app, 'current_project_folder', None)
            pf = project_folder.get() if project_folder else ''
            if pf and os.path.isdir(pf):
                output_folder = os.path.join(pf, 'evaluations')
            else:
                output_folder = self.eval_test_folder.get()
            os.makedirs(output_folder, exist_ok=True)

            for idx, session in enumerate(sessions, 1):
                if self._eval_cancel_flag.is_set():
                    self._log('\nEvaluation cancelled by user.')
                    return

                video_path  = session['video']
                dlc_path    = session['dlc']
                labels_path = session['labels']
                video_name  = os.path.basename(video_path)
                base_name   = os.path.splitext(video_name)[0]

                self._log(f'[{idx}/{len(sessions)}] {video_name}')

                # ── Load ground-truth labels ─────────────────────────────
                try:
                    labels_df = pd.read_csv(labels_path)
                except Exception as e:
                    self._log(f'  ✗ Could not read labels: {e}  — skipping.')
                    continue

                # Resolve behavior column
                if behavior_name in labels_df.columns:
                    label_col = behavior_name
                else:
                    ci_matches = [c for c in labels_df.columns
                                  if c.lower() == behavior_name.lower()
                                  and c.lower() != 'frame']
                    if ci_matches:
                        label_col = ci_matches[0]
                    else:
                        non_frame = [c for c in labels_df.columns
                                     if c.lower() != 'frame']
                        if len(non_frame) == 1:
                            label_col = non_frame[0]
                            self._log(
                                f'  ⚠️  Behavior column "{behavior_name}" not found; '
                                f'using "{label_col}".')
                        else:
                            self._log(
                                f'  ✗ Behavior column "{behavior_name}" not found. '
                                f'Columns: {list(labels_df.columns)}  — skipping.')
                            continue

                y_true = labels_df[label_col].values.astype(int)

                # ── Features: load cache or extract ──────────────────────
                X = None
                video_dir = os.path.dirname(video_path)
                if pf and os.path.isdir(pf):
                    cache_dir = os.path.join(pf, 'features')
                else:
                    cache_dir = os.path.join(video_dir, 'features')
                os.makedirs(cache_dir, exist_ok=True)

                # Use the same hash function as training (bp_include_list always None in training).
                # crop_offset is NOT in the hash so zero-offset eval caches are shared with training.
                cfg_hash = _gui.PixelPawsGUI._feature_hash_key(
                    {**clf_data, 'bp_include_list': None})
                _cache_fname = f'{base_name}_features_{cfg_hash}.pkl'
                cache_file = os.path.join(cache_dir, _cache_fname)

                # If not in canonical location, search fallback dirs
                if not os.path.isfile(cache_file):
                    if _FEATURE_CACHE_AVAILABLE:
                        _found = FeatureCacheManager.find_cache(
                            base_name, cfg_hash, cache_dir, video_dir,
                            project_root=pf)
                        if _found:
                            cache_file = _found
                            self._log(f'  [Cache] Found in fallback: {_found}')
                    else:
                        _fallback_locs = [
                            os.path.join(video_dir, 'PredictionCache', _cache_fname),
                            os.path.join(video_dir, 'FeatureCache', _cache_fname),
                            os.path.join(video_dir, _cache_fname),
                        ]
                        _ancestor = video_dir
                        while True:
                            _parent = os.path.dirname(_ancestor)
                            if _parent == _ancestor:
                                break
                            _ancestor = _parent
                            _fallback_locs.append(os.path.join(_ancestor, 'features', _cache_fname))
                            _fallback_locs.append(os.path.join(_ancestor, 'FeatureCache', _cache_fname))
                            if pf and os.path.normpath(_ancestor) == os.path.normpath(pf):
                                break
                        for _loc in _fallback_locs:
                            if os.path.isfile(_loc):
                                cache_file = _loc
                                self._log(f'  [Cache] Found in fallback: {_loc}')
                                break

                try:
                    X = _gui._load_features_for_prediction(
                        cache_file=cache_file,
                        model=model,
                        extract_fn=lambda: _extract(
                            pose_data_file=dlc_path,
                            video_file_path=video_path,
                            bp_include_list=None,
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            crop_offset_x=crop_x,
                            crop_offset_y=crop_y,
                            config_yaml_path=dlc_config_path or None,
                            cancel_flag=self._eval_cancel_flag,
                        ),
                        save_path=cache_file,
                        log_fn=self._log,
                        dlc_path=dlc_path,
                        clf_data=clf_data,
                    )
                except Exception as e:
                    self._log(f'  ✗ Feature extraction failed: {e}  — skipping.')
                    continue

                # ── Post-cache feature augmentation (match training) ────
                X = _gui.augment_features_post_cache(X, clf_data, model, dlc_path, log_fn=self._log)

                # ── Predict ──────────────────────────────────────────────
                try:
                    y_proba = _predict(
                        model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))
                except Exception as e:
                    self._log(f'  ✗ Prediction failed: {e}  — skipping.')
                    continue

                # Apply smoothing (bout filters / HMM Viterbi / none)
                try:
                    from prediction_pipeline import apply_smoothing as _smooth_fn
                    _smode = self.eval_smoothing_mode.get()
                    y_pred = _smooth_fn(y_proba, clf_data, _smode)
                except Exception as _se:
                    # Fallback to legacy bout-filter path on import/runtime error
                    y_pred = (y_proba >= best_thresh).astype(int)
                    if apply_bout_filter and min_bout > 1:
                        y_pred = _apply_bout_filtering(y_pred, min_bout, min_after, max_gap)

                # ── Align lengths ─────────────────────────────────────────
                n = min(len(y_true), len(y_pred))
                if len(y_true) != len(y_pred):
                    self._log(
                        f'  ⚠️  Length mismatch: labels={len(y_true)}, '
                        f'predictions={len(y_pred)}. Using first {n} frames.')
                y_true          = y_true[:n]
                y_pred          = y_pred[:n]
                y_proba_clipped = y_proba[:n]

                # ── Per-video metrics ─────────────────────────────────────
                f1, prec, rec, acc = self._compute_metrics(y_true, y_pred)
                bout_m = self._compute_bout_metrics(y_true, y_pred)
                self._log(
                    f'  F1={f1:.3f}  Prec={prec:.3f}  '
                    f'Rec={rec:.3f}  Acc={acc:.3f}  '
                    f'BoutF1={bout_m["bout_f1"]:.3f} '
                    f'({bout_m["n_true_bouts"]}t/{bout_m["n_pred_bouts"]}p)')

                all_y_true.extend(y_true.tolist())
                all_y_pred.extend(y_pred.tolist())

                per_video_results.append({
                    'video':      video_name,
                    'n_frames':   n,
                    'n_true':     int(np.sum(y_true)),
                    'n_pred':     int(np.sum(y_pred)),
                    'f1':         f1,
                    'precision':  prec,
                    'recall':     rec,
                    'accuracy':   acc,
                    'n_true_bouts':  bout_m['n_true_bouts'],
                    'n_pred_bouts':  bout_m['n_pred_bouts'],
                    'bout_f1':       bout_m['bout_f1'],
                    'bout_precision': bout_m['bout_precision'],
                    'bout_recall':   bout_m['bout_recall'],
                    'y_true':     y_true,
                    'y_pred':     y_pred,
                    'y_proba':    y_proba_clipped,
                    'base_name':  base_name,
                    'video_path': video_path,
                })

                # ── Save predictions CSV ─────────────────────────────────
                if self.eval_save_predictions.get():
                    pred_csv = os.path.join(
                        output_folder, f'{base_name}_eval_predictions.csv')
                    pd.DataFrame({
                        'Frame':       np.arange(n),
                        'y_true':      y_true,
                        'y_pred':      y_pred,
                        'Probability': y_proba_clipped,
                    }).to_csv(pred_csv, index=False)
                    self._log(f'  ✓ Saved: {os.path.basename(pred_csv)}')

            # ── Aggregate ────────────────────────────────────────────────
            if not all_y_true:
                self._log('\n✗ No sessions were successfully evaluated.')
                return

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

            ov_f1, ov_prec, ov_rec, ov_acc = self._compute_metrics(
                all_y_true, all_y_pred)

            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(all_y_true, all_y_pred)
            except Exception:
                tp = int(np.sum((all_y_pred == 1) & (all_y_true == 1)))
                fp = int(np.sum((all_y_pred == 1) & (all_y_true == 0)))
                fn = int(np.sum((all_y_pred == 0) & (all_y_true == 1)))
                tn = int(np.sum((all_y_pred == 0) & (all_y_true == 0)))
                cm = np.array([[tn, fp], [fn, tp]])

            self._log('\n' + '=' * 60)
            self._log('OVERALL RESULTS')
            self._log('=' * 60)
            self._log(f'Sessions evaluated:  {len(per_video_results)}')
            self._log(f'Total frames:        {len(all_y_true)}')
            self._log(
                f'True positives:      {int(np.sum(all_y_true))} frames '
                f'({100 * np.mean(all_y_true):.1f}%)')
            self._log(
                f'Predicted positives: {int(np.sum(all_y_pred))} frames '
                f'({100 * np.mean(all_y_pred):.1f}%)')
            self._log(f'\nF1 Score:   {ov_f1:.4f}')
            self._log(f'Precision:  {ov_prec:.4f}')
            self._log(f'Recall:     {ov_rec:.4f}')
            self._log(f'Accuracy:   {ov_acc:.4f}')

            # Aggregate bout-level metrics
            ov_bout = self._compute_bout_metrics(all_y_true, all_y_pred)
            self._log(f'\n--- Bout-Level Metrics ---')
            self._log(f'Labeled bouts:   {ov_bout["n_true_bouts"]}    '
                      f'Predicted bouts: {ov_bout["n_pred_bouts"]}')
            self._log(f'Bout Precision:  {ov_bout["bout_precision"]:.4f}   '
                      f'Bout Recall: {ov_bout["bout_recall"]:.4f}   '
                      f'Bout F1: {ov_bout["bout_f1"]:.4f}')

            tn_, fp_, fn_, tp_ = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            self._log('\nConfusion Matrix (aggregate):')
            self._log('              Predicted 0  Predicted 1')
            self._log(f'  Actual 0       {tn_:>8}     {fp_:>8}')
            self._log(f'  Actual 1       {fn_:>8}     {tp_:>8}')

            # ── Per-video table ───────────────────────────────────────────
            if self.eval_detailed_report.get() and per_video_results:
                self._log('\n' + '=' * 60)
                self._log('PER-VIDEO BREAKDOWN')
                self._log('=' * 60)
                self._log(
                    f'{"Video":<35} {"Frames":>7} {"TruePos%":>9} '
                    f'{"F1":>7} {"Prec":>7} {"Rec":>7} {"Acc":>7} '
                    f'{"TBout":>6} {"PBout":>6} {"BF1":>7}')
                self._log('-' * 100)
                for r in per_video_results:
                    true_pct = 100 * r['n_true'] / r['n_frames'] if r['n_frames'] else 0
                    self._log(
                        f'{r["video"][:35]:<35} {r["n_frames"]:>7} '
                        f'{true_pct:>8.1f}% {r["f1"]:>7.3f} '
                        f'{r["precision"]:>7.3f} {r["recall"]:>7.3f} '
                        f'{r["accuracy"]:>7.3f} '
                        f'{r.get("n_true_bouts", 0):>6} '
                        f'{r.get("n_pred_bouts", 0):>6} '
                        f'{r.get("bout_f1", 0):>7.3f}')

            # ── Save text report ──────────────────────────────────────────
            report_path = os.path.join(
                output_folder, f'evaluation_report_{behavior_name}.txt')
            try:
                with open(report_path, 'w') as f:
                    f.write(self.eval_results_text.get('1.0', tk.END))
                self._log(f'\n✓ Report saved: {os.path.basename(report_path)}')
            except Exception as e:
                self._log(f'\n⚠️  Could not save report: {e}')

            # ── Plots ─────────────────────────────────────────────────────
            if self.eval_generate_plots.get() and plt is not None:
                self._log('\nGenerating plots…')
                try:
                    self._generate_plots(
                        per_video_results, all_y_true, all_y_pred,
                        cm, behavior_name, output_folder)
                    self._log('✓ Plots saved.')
                except Exception as e:
                    self._log(f'⚠️  Could not generate plots: {e}')
                try:
                    self._generate_session_raster_plots(
                        per_video_results, behavior_name,
                        output_folder, best_thresh)
                except Exception as e:
                    self._log(f'⚠️  Could not generate raster plots: {e}')

            self._log('\n✓ Evaluation complete!')
            self._safe_after(lambda: messagebox.showinfo(
                'Evaluation Complete',
                f'Evaluated {len(per_video_results)} session(s).\n\n'
                f'Overall F1:        {ov_f1:.4f}\n'
                f'Overall Precision: {ov_prec:.4f}\n'
                f'Overall Recall:    {ov_rec:.4f}\n'
                f'Overall Accuracy:  {ov_acc:.4f}\n\n'
                f'Bout F1:           {ov_bout["bout_f1"]:.4f}\n'
                f'Bout Precision:    {ov_bout["bout_precision"]:.4f}\n'
                f'Bout Recall:       {ov_bout["bout_recall"]:.4f}\n\n'
                f'Results saved to:\n{output_folder}'))

        except Exception as e:
            err = traceback.format_exc()
            self._log(f'\n✗ ERROR:\n{err}')
            self._safe_after(lambda: messagebox.showerror(
                'Evaluation Error', f'Evaluation failed:\n\n{str(e)}'))

    @staticmethod
    def _compute_metrics(y_true, y_pred):
        """Return (f1, precision, recall, accuracy) without requiring sklearn."""
        try:
            from sklearn.metrics import (f1_score, precision_score,
                                         recall_score, accuracy_score)
            return (f1_score(y_true, y_pred, zero_division=0),
                    precision_score(y_true, y_pred, zero_division=0),
                    recall_score(y_true, y_pred, zero_division=0),
                    accuracy_score(y_true, y_pred))
        except Exception:
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            n  = len(y_true)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            acc  = (tp + tn) / n if n else 0.0
            return f1, prec, rec, acc

    @staticmethod
    def _compute_bout_metrics(y_true, y_pred, fps=30.0):
        """Bout-level precision, recall, F1.

        A predicted bout is a true positive if it overlaps any labeled bout.
        A labeled bout is detected if it overlaps any predicted bout.
        """
        true_bouts = count_bouts(y_true, fps)['bouts']
        pred_bouts = count_bouts(y_pred, fps)['bouts']

        matched_pred = 0
        for pb in pred_bouts:
            for tb in true_bouts:
                if pb['start'] <= tb['end'] and pb['end'] >= tb['start']:
                    matched_pred += 1
                    break

        matched_true = 0
        for tb in true_bouts:
            for pb in pred_bouts:
                if tb['start'] <= pb['end'] and tb['end'] >= pb['start']:
                    matched_true += 1
                    break

        n_true = len(true_bouts)
        n_pred = len(pred_bouts)

        bout_precision = matched_pred / n_pred if n_pred else 0.0
        bout_recall = matched_true / n_true if n_true else 0.0
        bout_f1 = (2 * bout_precision * bout_recall /
                   (bout_precision + bout_recall)) if (bout_precision + bout_recall) else 0.0

        return {
            'n_true_bouts': n_true,
            'n_pred_bouts': n_pred,
            'matched_pred': matched_pred,
            'matched_true': matched_true,
            'bout_precision': bout_precision,
            'bout_recall': bout_recall,
            'bout_f1': bout_f1,
        }

    # ----------------------------------------- CV Parameter Optimization ----

    @staticmethod
    def _grid_search_params(y_proba, y_true, metric='f1'):
        """Grid search over threshold/bout-filter params on pre-computed probabilities.

        Returns dict with best {threshold, min_bout, min_after_bout, max_gap,
        f1, accuracy, precision, recall}.
        """
        from sklearn.metrics import (f1_score, accuracy_score,
                                     precision_score, recall_score)

        thresholds = np.arange(0.3, 0.8, 0.05)
        min_bouts = [1, 3, 5, 10, 15]
        min_after_bouts = [0, 1, 3, 5]
        max_gaps = [0, 5, 10, 20]

        best_score = -1
        best_params = {}

        for thresh in thresholds:
            for mb in min_bouts:
                for ma in min_after_bouts:
                    for mg in max_gaps:
                        y_pred = (y_proba >= thresh).astype(int)
                        y_pred = _apply_bout_filtering(y_pred, mb, ma, mg)

                        f1  = f1_score(y_true, y_pred, zero_division=0)
                        acc = accuracy_score(y_true, y_pred)

                        score = f1 if metric == 'f1' else (
                            acc if metric == 'accuracy' else (f1 + acc) / 2)

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'threshold': float(thresh),
                                'min_bout': mb,
                                'min_after_bout': ma,
                                'max_gap': mg,
                                'f1': f1,
                                'accuracy': acc,
                                'precision': precision_score(y_true, y_pred, zero_division=0),
                                'recall': recall_score(y_true, y_pred, zero_division=0),
                            }

        return best_params

    def _run_cv_optimization(self):
        """Entry point for cross-validated parameter optimization."""
        if not self.eval_classifier_path.get():
            messagebox.showwarning('No Classifier', 'Please select a classifier file.')
            return
        if not self.eval_test_folder.get():
            messagebox.showwarning('No Test Data', 'Please select a test data folder.')
            return

        sessions = self._find_sessions()
        if len(sessions) < 2:
            messagebox.showwarning(
                'Not Enough Data',
                'Need at least 2 scored videos for cross-validated optimization.\n\n'
                f'Found {len(sessions)} session(s) with labels.')
            return

        # Progress dialog
        dialog = tk.Toplevel(self.root)
        dialog.title('Cross-Validated Parameter Optimization')
        _sw, _sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        dialog.geometry(f'700x500+{(_sw-700)//2}+{(_sh-500)//2}')
        dialog.transient(self.root)

        ttk.Label(dialog, text='Cross-Validated Parameter Optimization',
                  font=('Arial', 12, 'bold')).pack(pady=10)

        log_text = scrolledtext.ScrolledText(dialog, height=22, wrap=tk.WORD)
        log_text.pack(fill='both', expand=True, padx=10, pady=5)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)
        close_btn = ttk.Button(btn_frame, text='Close', command=dialog.destroy,
                               state='disabled')
        close_btn.pack()

        def _log(msg):
            self._safe_after(lambda m=msg: (
                log_text.insert(tk.END, m + '\n'),
                log_text.see(tk.END)
            ))

        def _done():
            self._safe_after(lambda: close_btn.config(state='normal'))

        threading.Thread(
            target=self._cv_optimization_thread,
            args=(sessions, _log, _done),
            daemon=True).start()

    def _cv_optimization_thread(self, sessions, log_fn, done_fn):
        """Background thread: leave-one-video-out parameter optimization."""
        try:
            import PixelPaws_GUI as _gui
            _extract  = _gui.PixelPaws_ExtractFeatures
            _predict  = _gui.predict_with_xgboost
            _clean_bp = _gui.clean_bodyparts_list
            _auto_bp  = _gui.auto_detect_bodyparts_from_model
        except (ImportError, AttributeError) as e:
            log_fn(f'ERROR: Could not import PixelPaws_GUI helpers: {e}')
            done_fn()
            return

        try:
            clf_path = self.eval_classifier_path.get()
            dlc_config_path = self.eval_dlc_config_path.get()

            log_fn('=' * 60)
            log_fn('CROSS-VALIDATED PARAMETER OPTIMIZATION')
            log_fn('=' * 60)

            # Load classifier
            log_fn(f'\nLoading classifier: {os.path.basename(clf_path)}')
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)

            clf_data['bp_include_list'] = _clean_bp(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list']  = _clean_bp(clf_data.get('bp_pixbrt_list', []))
            clf_data = _auto_bp(clf_data, verbose=False)

            model         = clf_data['clf_model']
            behavior_name = clf_data.get('Behavior_type', 'Behavior')
            default_thresh = clf_data.get('best_thresh', 0.5)
            default_mb     = clf_data.get('min_bout', 1)
            default_ma     = clf_data.get('min_after_bout', 1)
            default_mg     = clf_data.get('max_gap', 0)

            log_fn(f'  Behavior: {behavior_name}')
            log_fn(f'  Classifier defaults: thresh={default_thresh:.3f}, '
                   f'min_bout={default_mb}, min_after={default_ma}, max_gap={default_mg}')

            # DLC crop offset
            crop_x, crop_y = 0, 0
            if dlc_config_path and os.path.isfile(dlc_config_path):
                try:
                    import yaml
                    with open(dlc_config_path) as f:
                        config = yaml.safe_load(f)
                    if config.get('cropping', False):
                        crop_x = config.get('x1', 0)
                        crop_y = config.get('y1', 0)
                except Exception:
                    pass

            N = len(sessions)
            log_fn(f'\nFound {N} scored sessions — running {N}-fold leave-one-out\n')

            # Step 1: extract features + probabilities for all videos
            pf = getattr(self.app, 'current_project_folder', None)
            pf = pf.get() if pf else ''

            video_data = []  # list of (y_proba, y_true, video_name)

            for idx, session in enumerate(sessions, 1):
                video_path  = session['video']
                dlc_path    = session['dlc']
                labels_path = session['labels']
                video_name  = os.path.basename(video_path)
                base_name   = os.path.splitext(video_name)[0]

                log_fn(f'[{idx}/{N}] Loading {video_name}…')

                # Load labels
                labels_df = pd.read_csv(labels_path)
                if behavior_name in labels_df.columns:
                    label_col = behavior_name
                else:
                    ci = [c for c in labels_df.columns
                          if c.lower() == behavior_name.lower() and c.lower() != 'frame']
                    if ci:
                        label_col = ci[0]
                    else:
                        non_frame = [c for c in labels_df.columns if c.lower() != 'frame']
                        if len(non_frame) == 1:
                            label_col = non_frame[0]
                        else:
                            log_fn(f'  Skipping — behavior column not found')
                            continue

                y_true = labels_df[label_col].values.astype(int)

                # Load/extract features (same logic as _evaluation_thread)
                video_dir = os.path.dirname(video_path)
                cache_dir = os.path.join(pf, 'features') if pf and os.path.isdir(pf) else os.path.join(video_dir, 'features')
                os.makedirs(cache_dir, exist_ok=True)

                cfg_hash = _gui.PixelPawsGUI._feature_hash_key(
                    {**clf_data, 'bp_include_list': None})
                _cache_fname = f'{base_name}_features_{cfg_hash}.pkl'
                cache_file = os.path.join(cache_dir, _cache_fname)

                X = None
                if not os.path.isfile(cache_file):
                    if _FEATURE_CACHE_AVAILABLE:
                        _found = FeatureCacheManager.find_cache(
                            base_name, cfg_hash, cache_dir, video_dir, project_root=pf)
                        if _found:
                            cache_file = _found
                    else:
                        for _loc in [
                            os.path.join(video_dir, 'PredictionCache', _cache_fname),
                            os.path.join(video_dir, 'FeatureCache', _cache_fname),
                            os.path.join(video_dir, _cache_fname),
                        ]:
                            if os.path.isfile(_loc):
                                cache_file = _loc
                                break

                try:
                    X = _gui._load_features_for_prediction(
                        cache_file=cache_file,
                        model=model,
                        extract_fn=lambda: _extract(
                            pose_data_file=dlc_path,
                            video_file_path=video_path,
                            bp_include_list=None,
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            crop_offset_x=crop_x,
                            crop_offset_y=crop_y,
                            config_yaml_path=dlc_config_path or None,
                        ),
                        save_path=cache_file,
                        log_fn=log_fn,
                        dlc_path=dlc_path,
                        clf_data=clf_data,
                    )
                except Exception as e:
                    log_fn(f'  Skipping — feature extraction failed: {e}')
                    continue

                X = _gui.augment_features_post_cache(X, clf_data, model, dlc_path)

                # Predict
                try:
                    y_proba = _predict(
                        model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))
                except Exception as e:
                    log_fn(f'  Skipping — prediction failed: {e}')
                    continue

                n = min(len(y_true), len(y_proba))
                video_data.append({
                    'y_proba': y_proba[:n],
                    'y_true': y_true[:n],
                    'name': video_name,
                })

            if len(video_data) < 2:
                log_fn(f'\nERROR: Only {len(video_data)} video(s) loaded successfully. '
                       f'Need at least 2.')
                done_fn()
                return

            N = len(video_data)
            log_fn(f'\n{"=" * 60}')
            log_fn(f'LEAVE-ONE-OUT OPTIMIZATION ({N} folds)')
            log_fn('=' * 60)

            # Step 2: LOO grid search
            fold_results = []
            for k in range(N):
                train_proba = np.concatenate(
                    [video_data[i]['y_proba'] for i in range(N) if i != k])
                train_labels = np.concatenate(
                    [video_data[i]['y_true'] for i in range(N) if i != k])

                best = self._grid_search_params(train_proba, train_labels)

                # Evaluate on held-out fold with best params for bout metrics
                _ho_pred = (video_data[k]['y_proba'] >= best['threshold']).astype(int)
                _ho_pred = _apply_bout_filtering(
                    _ho_pred, best['min_bout'], best['min_after_bout'], best['max_gap'])
                _ho_bout = self._compute_bout_metrics(video_data[k]['y_true'], _ho_pred)
                best['bout_f1'] = _ho_bout['bout_f1']

                fold_results.append(best)

                log_fn(f'\n  Fold {k+1}/{N} (held out: {video_data[k]["name"][:30]})')
                log_fn(f'    Best: thresh={best["threshold"]:.3f}, '
                       f'min_bout={best["min_bout"]}, min_after={best["min_after_bout"]}, '
                       f'max_gap={best["max_gap"]}  →  F1={best["f1"]:.3f}  '
                       f'BoutF1={best["bout_f1"]:.3f}')

            # Step 3: Average params
            avg_thresh = round(float(np.mean([r['threshold'] for r in fold_results])), 3)
            avg_mb = int(round(float(np.median([r['min_bout'] for r in fold_results]))))
            avg_ma = int(round(float(np.median([r['min_after_bout'] for r in fold_results]))))
            avg_mg = int(round(float(np.median([r['max_gap'] for r in fold_results]))))

            log_fn(f'\n{"=" * 60}')
            log_fn('AVERAGED PARAMETERS')
            log_fn('=' * 60)
            log_fn(f'  Threshold:       {avg_thresh:.3f}  (mean)')
            log_fn(f'  Min bout:        {avg_mb}  (median)')
            log_fn(f'  Min after bout:  {avg_ma}  (median)')
            log_fn(f'  Max gap:         {avg_mg}  (median)')

            # Step 4: Final evaluation with averaged params on ALL videos
            all_y_true = np.concatenate([v['y_true'] for v in video_data])
            all_y_proba = np.concatenate([v['y_proba'] for v in video_data])
            all_y_pred = (all_y_proba >= avg_thresh).astype(int)
            all_y_pred = _apply_bout_filtering(all_y_pred, avg_mb, avg_ma, avg_mg)

            f1, prec, rec, acc = self._compute_metrics(all_y_true, all_y_pred)

            lovo_bout = self._compute_bout_metrics(all_y_true, all_y_pred)

            log_fn(f'\n{"=" * 60}')
            log_fn('FINAL METRICS (averaged params on all videos)')
            log_fn('=' * 60)
            log_fn(f'  F1:        {f1:.4f}')
            log_fn(f'  Precision: {prec:.4f}')
            log_fn(f'  Recall:    {rec:.4f}')
            log_fn(f'  Accuracy:  {acc:.4f}')
            log_fn(f'  Bout F1:        {lovo_bout["bout_f1"]:.4f}')
            log_fn(f'  Bout Precision: {lovo_bout["bout_precision"]:.4f}')
            log_fn(f'  Bout Recall:    {lovo_bout["bout_recall"]:.4f}')

            # Compare with classifier defaults
            def_y_pred = (all_y_proba >= default_thresh).astype(int)
            def_y_pred = _apply_bout_filtering(def_y_pred, default_mb, default_ma, default_mg)
            def_f1, def_prec, def_rec, def_acc = self._compute_metrics(all_y_true, def_y_pred)

            log_fn(f'\n{"=" * 60}')
            log_fn('COMPARISON vs CLASSIFIER DEFAULTS')
            log_fn('=' * 60)
            log_fn(f'  {"":20s} {"Optimized":>10s}  {"Default":>10s}  {"Diff":>10s}')
            log_fn(f'  {"F1":20s} {f1:>10.4f}  {def_f1:>10.4f}  {f1-def_f1:>+10.4f}')
            log_fn(f'  {"Precision":20s} {prec:>10.4f}  {def_prec:>10.4f}  {prec-def_prec:>+10.4f}')
            log_fn(f'  {"Recall":20s} {rec:>10.4f}  {def_rec:>10.4f}  {rec-def_rec:>+10.4f}')
            log_fn(f'  {"Accuracy":20s} {acc:>10.4f}  {def_acc:>10.4f}  {acc-def_acc:>+10.4f}')

            # Fold stability
            log_fn(f'\n{"=" * 60}')
            log_fn('PER-FOLD PARAMETER STABILITY')
            log_fn('=' * 60)
            log_fn(f'  {"Fold":>5s}  {"Threshold":>10s}  {"MinBout":>8s}  '
                   f'{"MinAfter":>9s}  {"MaxGap":>7s}  {"F1":>7s}  {"BoutF1":>7s}')
            log_fn(f'  {"-"*5}  {"-"*10}  {"-"*8}  {"-"*9}  {"-"*7}  {"-"*7}  {"-"*7}')
            for k, r in enumerate(fold_results):
                log_fn(f'  {k+1:>5d}  {r["threshold"]:>10.3f}  {r["min_bout"]:>8d}  '
                       f'{r["min_after_bout"]:>9d}  {r["max_gap"]:>7d}  {r["f1"]:>7.3f}  '
                       f'{r.get("bout_f1", 0):>7.3f}')

            log_fn(f'\n✓ Optimization complete!')

            # Populate custom parameter spinboxes
            self._safe_after(lambda: self.eval_use_custom_params.set(True))
            self._safe_after(lambda: self._toggle_custom_params())
            self._safe_after(lambda: self.eval_custom_threshold.set(avg_thresh))
            self._safe_after(lambda: self.eval_custom_min_bout.set(avg_mb))
            self._safe_after(lambda: self.eval_custom_min_after.set(avg_ma))
            self._safe_after(lambda: self.eval_custom_max_gap.set(avg_mg))

            self._optimized_params = {
                'best_thresh': avg_thresh,
                'min_bout': avg_mb,
                'min_after_bout': avg_ma,
                'max_gap': avg_mg,
            }
            self._safe_after(lambda: self._save_params_btn.config(state='normal'))

            log_fn('\nOptimized parameters have been applied to the custom parameter fields.')
            log_fn('Click "💾 Save Params to Classifier" to persist them, or "RUN EVALUATION" to test.')

        except Exception as e:
            err = traceback.format_exc()
            log_fn(f'\nERROR:\n{err}')
        finally:
            done_fn()

    def _generate_plots(self, per_video_results, all_y_true, all_y_pred,
                        cm, behavior_name, output_folder):
        """
        Save a two-panel PNG:
          Left  — aggregate confusion matrix heatmap
          Right — per-video F1 / Precision / Recall bar chart
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Evaluation — {behavior_name}', fontsize=13, fontweight='bold')
        for a in axes:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        axes[1].yaxis.grid(True, alpha=0.3, linestyle='--')

        # Confusion matrix
        ax = axes[0]
        if cm is not None and cm.size == 4:
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('Predicted label', fontsize=12)
            ax.set_ylabel('True label', fontsize=12)
            ax.set_title('Confusion Matrix (aggregate)', fontsize=13, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['0 (no)', '1 (yes)'])
            ax.set_yticklabels(['0 (no)', '1 (yes)'])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=14,
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
        else:
            ax.text(0.5, 0.5, 'Confusion matrix\nnot available',
                    ha='center', va='center', transform=ax.transAxes)

        # Per-video bar chart
        ax2 = axes[1]
        if per_video_results:
            names = [r['video'][:20]  for r in per_video_results]
            f1s   = [r['f1']          for r in per_video_results]
            precs = [r['precision']   for r in per_video_results]
            recs  = [r['recall']      for r in per_video_results]
            x = np.arange(len(names))
            w = 0.25
            ax2.bar(x - w, f1s,   width=w, label='F1',       color='steelblue')
            ax2.bar(x,     precs, width=w, label='Precision', color='darkorange')
            ax2.bar(x + w, recs,  width=w, label='Recall',    color='seagreen')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Score', fontsize=12)
            ax2.set_title('Per-video Performance', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            if f1s:
                ax2.axhline(y=np.mean(f1s), color='steelblue',
                            linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, 'No per-video data',
                     ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        plot_path = os.path.join(
            output_folder, f'evaluation_plots_{behavior_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self._log(f'  ✓ {os.path.basename(plot_path)}')

    def _generate_session_raster_plots(self, per_video_results, behavior_name,
                                        output_folder, best_thresh):
        """Generate per-session 3-panel diagnostic PNGs (raster | CM | time-bins)."""
        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None

        def _bouts_from_array(arr):
            padded = np.concatenate([[0], arr.astype(int), [0]])
            diff   = np.diff(padded)
            starts = np.where(diff ==  1)[0]
            ends   = np.where(diff == -1)[0]
            return list(zip(starts.tolist(), (ends - starts).tolist()))

        from sklearn.metrics import f1_score as _f1, confusion_matrix as _cm
        from matplotlib.lines import Line2D

        for r in per_video_results:
            y_true = r['y_true']
            y_pred = r['y_pred']
            base   = r['base_name']
            vpath  = r.get('video_path', '')

            # FPS
            fps = 30.0
            if vpath and os.path.isfile(vpath):
                try:
                    import cv2
                    cap  = cv2.VideoCapture(vpath)
                    _fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if _fps > 0:
                        fps = float(_fps)
                except Exception:
                    pass

            # Metrics
            f1     = _f1(y_true, y_pred, zero_division=0)
            cm_raw = _cm(y_true, y_pred, labels=[0, 1])
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm_raw / row_sums

            # Time bins (10 s)
            bin_frames = max(1, int(10 * fps))
            n_frames   = len(y_true)
            n_bins     = max(1, int(np.ceil(n_frames / bin_frames)))
            human_s = np.zeros(n_bins)
            model_s = np.zeros(n_bins)
            for k in range(n_bins):
                sl = slice(k * bin_frames, (k + 1) * bin_frames)
                human_s[k] = y_true[sl].sum() / fps
                model_s[k] = y_pred[sl].sum() / fps
            r_val = float('nan')
            if pearsonr is not None and n_bins > 1:
                r_val, _ = pearsonr(human_s, model_s)

            # Figure — 3-panel layout matching training-tab rasters
            fig = plt.figure(figsize=(16, 4), constrained_layout=True)
            gs  = fig.add_gridspec(1, 3, width_ratios=[5, 2, 4], wspace=0.35)
            ax_raster, ax_cm, ax_bins = gs.subplots()

            # Panel 1 — Raster
            bouts_true = _bouts_from_array(y_true)
            bouts_pred = _bouts_from_array(y_pred)
            if bouts_true:
                ax_raster.broken_barh(bouts_true, (2.6, 0.8), facecolors='black')
            if bouts_pred:
                ax_raster.broken_barh(bouts_pred, (1.2, 0.8), facecolors='#E87722')
            ax_raster.set_yticks([1.6, 3.0])
            ax_raster.set_yticklabels(['Model', 'Human'])
            ax_raster.set_ylim(0.8, 3.8)
            ax_raster.set_xlabel('Frame')
            ax_raster.set_ylabel('Labels')
            ax_raster.set_title(
                f"{behavior_name} raster: {base}\n(thr={best_thresh:.2f})")
            ax_raster.legend(handles=[
                Line2D([0], [0], color='black',   linewidth=8, label='Human'),
                Line2D([0], [0], color='#E87722', linewidth=8, label='Model'),
            ], loc='upper left', fontsize=10)

            # Panel 2 — Confusion Matrix
            ax_cm.imshow(cm_norm, cmap='RdPu', vmin=0, vmax=1)
            for row in range(2):
                for col in range(2):
                    v = cm_norm[row, col]
                    ax_cm.text(col, row, f"{v:.2f}", ha='center', va='center',
                               color='white' if v > 0.5 else 'black', fontsize=10)
            ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['0', '1']); ax_cm.set_yticklabels(['0', '1'])
            ax_cm.set_xlabel('Pred'); ax_cm.set_ylabel('True')
            ax_cm.set_title(f"F1={f1:.2f}")

            # Panel 3 — Time Bins
            x_centers = np.arange(n_bins) * 10 + 5
            width = 0.4
            ax_bins.bar(x_centers - width / 2, human_s,
                        width=width, color='steelblue', label='Human')
            ax_bins.bar(x_centers + width / 2, model_s,
                        width=width, color='#E87722', label='Model')
            ax_bins.set_xlabel('Time (s)')
            ax_bins.set_ylabel('Seconds/bin')
            r_str = f"{r_val:.2f}" if not np.isnan(r_val) else "n/a"
            ax_bins.set_title(f"Time bins (10s);  R = {r_str}")
            ax_bins.legend(fontsize=10)
            ax_bins.spines['top'].set_visible(False)
            ax_bins.spines['right'].set_visible(False)

            out_path = os.path.join(
                output_folder,
                f'PixelPaws_{behavior_name}_Raster_{base}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self._log(f'  ✓ Raster: {os.path.basename(out_path)}')

    # --------------------------------------------------------- SHAP --------

    def run_shap_analysis(self):
        """Open the SHAP data-source dialog and start the analysis thread."""
        try:
            import shap  # noqa: F401
        except ImportError:
            messagebox.showerror(
                'SHAP Not Installed',
                'SHAP library is not installed.\n\n'
                'Install with:  pip install shap\n'
                'or:            conda install -c conda-forge shap')
            return

        if not self.eval_classifier_path.get():
            messagebox.showwarning('No Classifier',
                                   'Please select a classifier file first.')
            return

        try:
            with open(self.eval_classifier_path.get(), 'rb') as f:
                clf_data = pickle.load(f)
            model = clf_data.get('clf_model') or clf_data.get('model')
            if model is None:
                messagebox.showerror('Error', 'Could not find model in classifier file.')
                return
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load classifier:\n{e}')
            return

        # --- Dialog to choose data source ---
        dialog = tk.Toplevel(self.root)
        dialog.title('SHAP Analysis — Data Source')
        _sw, _sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        dialog.geometry(f'600x360+{(_sw-600)//2}+{(_sh-360)//2}')
        dialog.grab_set()

        ttk.Label(dialog, text='Select Data Source for SHAP Analysis',
                  font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Label(dialog,
                  text='SHAP will explain which features drive the model\'s predictions.',
                  font=('Arial', 9), foreground='gray').pack(pady=4)

        source_var = tk.StringVar(value='test')
        ttk.Radiobutton(dialog, text='Use test data folder (if specified above)',
                        variable=source_var, value='test').pack(anchor='w', padx=30, pady=4)
        ttk.Radiobutton(dialog, text='Select a features file (.pkl)',
                        variable=source_var, value='file').pack(anchor='w', padx=30, pady=4)
        ttk.Radiobutton(dialog, text='Use training data (if embedded in classifier)',
                        variable=source_var, value='training').pack(anchor='w', padx=30, pady=4)

        ttk.Label(dialog,
                  text='Samples to analyse (more = slower but more accurate):').pack(
            pady=(12, 4))
        n_samples_var = tk.IntVar(value=1000)
        ttk.Spinbox(dialog, from_=100, to=50000, textvariable=n_samples_var,
                    width=10, increment=1000).pack()
        ttk.Label(dialog, text='Recommended: 1000–5000 (100 samples ≈ 1 second)',
                  font=('Arial', 8), foreground='gray').pack(pady=4)

        result = {'cancelled': True}

        def on_ok():
            result['cancelled'] = False
            result['source']    = source_var.get()
            result['n_samples'] = n_samples_var.get()
            dialog.destroy()

        ttk.Button(dialog, text='Generate SHAP Plots', command=on_ok).pack(pady=8)
        ttk.Button(dialog, text='Cancel', command=dialog.destroy).pack()
        dialog.wait_window()

        if result['cancelled']:
            return

        threading.Thread(
            target=self._shap_thread,
            args=(clf_data, model, result['source'], result['n_samples']),
            daemon=True).start()

    def _shap_thread(self, clf_data, model, source, n_samples):
        """Background thread that computes and saves SHAP plots and a report."""
        try:
            import shap
            import matplotlib.pyplot as mpl_plt

            def _log_shap(msg):
                self._safe_after(lambda m=msg: (
                    self.eval_results_text.insert(tk.END, m + '\n'),
                    self.eval_results_text.see(tk.END)
                ))

            behavior_name = clf_data.get('Behavior_type', 'Behavior')

            _log_shap(f"\n{'=' * 60}\nSHAP ANALYSIS\n{'=' * 60}")
            _log_shap('Loading data…')

            X = None
            feature_names = None

            if source == 'training':
                if 'X_train' in clf_data:
                    X = clf_data['X_train']
                    feature_names = (clf_data.get('feature_names') or
                                     (X.columns.tolist() if hasattr(X, 'columns') else None))
                    _log_shap(f'✓ Using training data from classifier ({len(X)} samples)')
                else:
                    self._safe_after(lambda: messagebox.showerror(
                        'Error', 'No training data found in classifier file.'))
                    return

            elif source == 'test':
                test_folder = self.eval_test_folder.get()
                if not test_folder:
                    self._safe_after(lambda: messagebox.showerror(
                        'Error', 'No test folder specified.'))
                    return
                for dirpath, _, files in os.walk(test_folder):
                    for fname in files:
                        if fname.endswith('.pkl') and 'features' in fname.lower():
                            fp = os.path.join(dirpath, fname)
                            try:
                                with open(fp, 'rb') as f:
                                    X = pickle.load(f)
                                _log_shap(f'✓ Loaded features from: {fname}')
                                break
                            except Exception:
                                continue
                    if X is not None:
                        break
                if X is None:
                    self._safe_after(lambda: messagebox.showerror(
                        'Error', 'No feature .pkl files found in test folder.'))
                    return

            elif source == 'file':
                file_path = filedialog.askopenfilename(
                    title='Select Features File',
                    filetypes=[('Pickle files', '*.pkl'), ('All files', '*.*')])
                if not file_path:
                    return
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        if 'X' in data:
                            X = data['X']
                        elif 'X_train' in data:
                            X = data['X_train']
                            feature_names = data.get('feature_names')
                        else:
                            self._safe_after(lambda ks=list(data.keys()):
                                messagebox.showerror(
                                    'Error',
                                    f'Unknown dict format. Keys: {ks}\n\n'
                                    f"Expected 'X' or 'X_train'"))
                            return
                    else:
                        X = data
                    _log_shap(f'✓ Loaded features from: {os.path.basename(file_path)}')
                except Exception as e:
                    self._safe_after(lambda: messagebox.showerror(
                        'Error', f'Failed to load features:\n{e}'))
                    return

            # Ensure DataFrame
            if not isinstance(X, pd.DataFrame):
                _log_shap(f'Converting {type(X).__name__} to DataFrame…')
                if feature_names is not None:
                    try:
                        X = pd.DataFrame(X, columns=feature_names)
                    except Exception:
                        X = pd.DataFrame(X)
                else:
                    X = pd.DataFrame(X)

            # Keep only numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(X.columns):
                dropped = len(X.columns) - len(numeric_cols)
                _log_shap(f'⚠️  Dropping {dropped} non-numeric column(s)')
                X = X[numeric_cols]

            if X.shape[1] == 0:
                self._safe_after(lambda: messagebox.showerror(
                    'Error', 'No numeric features found in the selected data.'))
                return

            _log_shap(f'Data shape: {X.shape[0]} rows × {X.shape[1]} features')

            # Sub-sample if large
            if len(X) > n_samples:
                X = X.sample(n=n_samples, random_state=42)
                _log_shap(f'Using {n_samples} random samples')

            _log_shap('\nComputing SHAP values (this may take a minute)…')
            self._safe_after(lambda: self.eval_results_text.see(tk.END))

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            _log_shap('✓ SHAP values computed\n\nGenerating plots…')

            pf = getattr(self.app, 'current_project_folder', None)
            pf = pf.get() if pf else ''
            if pf and os.path.isdir(pf):
                output_dir = os.path.join(pf, 'evaluations', f'SHAP_{behavior_name}')
            else:
                output_dir = os.path.join(
                    os.path.dirname(self.eval_classifier_path.get()), f'SHAP_{behavior_name}')
            os.makedirs(output_dir, exist_ok=True)

            # 1. Feature importance bar
            mpl_plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type='bar', show=False, max_display=20)
            mpl_plt.tight_layout()
            mpl_plt.savefig(os.path.join(output_dir, '1_feature_importance.png'),
                            dpi=300, bbox_inches='tight')
            mpl_plt.close()
            _log_shap('✓ Feature importance plot saved')

            # 2. Beeswarm / feature effects
            mpl_plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False, max_display=20)
            mpl_plt.tight_layout()
            mpl_plt.savefig(os.path.join(output_dir, '2_feature_effects_beeswarm.png'),
                            dpi=300, bbox_inches='tight')
            mpl_plt.close()
            _log_shap('✓ Feature effects plot saved')

            # 3. Top-10 dependence plots
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(feature_importance)[-10:][::-1]
            for i, idx in enumerate(top_idx, 1):
                fname_feat = (X.columns[idx] if hasattr(X, 'columns')
                              else f'Feature_{idx}')
                mpl_plt.figure(figsize=(8, 6))
                shap.dependence_plot(idx, shap_values, X, show=False)
                mpl_plt.title(f'Dependence: {fname_feat}')
                mpl_plt.tight_layout()
                safe = fname_feat[:30].replace(os.sep, '_')
                mpl_plt.savefig(os.path.join(
                    output_dir, f'3_dependence_{i:02d}_{safe}.png'),
                    dpi=300, bbox_inches='tight')
                mpl_plt.close()
                if i % 3 == 0:
                    _log_shap(f'✓ Dependence plots: {i}/10 done')

            # Text report
            report_path = os.path.join(output_dir, 'SHAP_report.txt')
            with open(report_path, 'w') as f:
                f.write('=' * 80 + '\n')
                f.write('SHAP FEATURE IMPORTANCE ANALYSIS\n')
                f.write('=' * 80 + '\n\n')
                f.write(f'Classifier: {os.path.basename(self.eval_classifier_path.get())}\n')
                f.write(f'Samples:    {len(X)}\n')
                f.write(f'Features:   {X.shape[1]}\n\n')
                f.write('-' * 80 + '\n')
                f.write('TOP 20 MOST IMPORTANT FEATURES\n')
                f.write('-' * 80 + '\n\n')
                for rank, idx in enumerate(np.argsort(feature_importance)[-20:][::-1], 1):
                    feat = (X.columns[idx] if hasattr(X, 'columns')
                            else f'Feature_{idx}')
                    f.write(f'{rank:2d}. {feat:<50s} {feature_importance[idx]:.4f}\n')

            _log_shap(f"\n{'=' * 60}")
            _log_shap('✓ SHAP Analysis Complete!')
            _log_shap(f'Results saved to:\n{output_dir}')
            _log_shap(f"{'=' * 60}")

            self._safe_after(lambda: messagebox.showinfo(
                'SHAP Analysis Complete',
                f'Results saved to:\n{output_dir}\n\n'
                'Files generated:\n'
                '  • 1_feature_importance.png\n'
                '  • 2_feature_effects_beeswarm.png\n'
                '  • 3_dependence_*.png (top 10 features)\n'
                '  • SHAP_report.txt'))

        except Exception as e:
            self._safe_after(lambda: messagebox.showerror(
                'SHAP Error', f'Error during SHAP analysis:\n\n{str(e)}'))
            traceback.print_exc()
