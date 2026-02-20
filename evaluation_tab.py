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
        while i < len(y_filtered) - max_gap:
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
    if os.path.isdir(videos_sub):
        search_root = videos_sub
        project_root = folder
    else:
        search_root = folder
        project_root = os.path.dirname(folder)

    # ── Find all DLC .h5 files ────────────────────────────────────────────────
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

        if base in seen_bases:
            continue  # deduplicate when multiple DLC files share a base

        # ── Resolve video ─────────────────────────────────────────────────────
        video_path = None
        for ext in [video_ext, video_ext.upper(), '.mp4', '.avi', '.MP4', '.AVI']:
            candidate = os.path.join(search_root, base + ext)
            if os.path.isfile(candidate):
                video_path = candidate
                break
        if not video_path:
            continue  # no video → skip

        video_dir = os.path.dirname(video_path)

        # ── Resolve DLC file (prefer filtered if requested) ───────────────────
        # There may be multiple DLC files for this base; pick the best one.
        dlc_candidates = glob.glob(os.path.join(search_root, f'{base}DLC*.h5'))
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

        seen_bases.add(base)
        sessions.append({
            'session_name': base,
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

        # ── Action buttons ────────────────────────────────────────────────
        act = ttk.Frame(sf)
        act.pack(fill='x', padx=5, pady=10)

        ttk.Button(act, text='▶ RUN EVALUATION',
                   command=self.run_evaluation,
                   style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(act, text='🔬 SHAP Analysis',
                   command=self.run_shap_analysis).pack(side='left', padx=5)

        # ── Results ───────────────────────────────────────────────────────
        res = ttk.LabelFrame(sf, text='Evaluation Results', padding=5)
        res.pack(fill='both', expand=True, padx=5, pady=5)
        self.eval_results_text = scrolledtext.ScrolledText(res, height=15, wrap=tk.WORD)
        self.eval_results_text.pack(fill='both', expand=True)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    # --------------------------------------------------------------- Browse --

    def refresh_classifiers(self):
        """Populate the classifier dropdown from project classifiers/ folder."""
        pf = getattr(self.app, 'current_project_folder', None)
        clf_dir = os.path.join(pf.get() if pf else '', 'classifiers')
        self.eval_classifier_options = {}
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    self.eval_classifier_options[f] = os.path.join(clf_dir, f)
        if hasattr(self, 'eval_classifier_combo'):
            self.eval_classifier_combo['values'] = list(self.eval_classifier_options.keys())

    def _on_classifier_selected(self, event=None):
        """Update the full path StringVar when a dropdown item is chosen."""
        name = self.eval_classifier_combo.get()
        if name in self.eval_classifier_options:
            self.eval_classifier_path.set(self.eval_classifier_options[name])

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
            # Propagate to shared project folder if available
            if hasattr(self.app, 'current_project_folder'):
                self.app.current_project_folder.set(d)
            self.refresh_classifiers()

    def _browse_dlc_config(self):
        p = filedialog.askopenfilename(
            title='Select DLC Config File',
            filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')])
        if p:
            self.eval_dlc_config_path.set(p)

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

    def run_evaluation(self):
        """Entry point — validates inputs, then starts the evaluation thread."""
        if not self.eval_classifier_path.get():
            messagebox.showwarning('No Classifier', 'Please select a classifier file.')
            return
        if not self.eval_test_folder.get():
            messagebox.showwarning('No Test Data', 'Please select a test data folder.')
            return
        threading.Thread(target=self._evaluation_thread, daemon=True).start()

    def _log(self, msg):
        """Thread-safe append to the results text box."""
        self.root.after(0, lambda m=msg: (
            self.eval_results_text.insert(tk.END, m + '\n'),
            self.eval_results_text.see(tk.END)
        ))

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
            self.root.after(0, lambda: self.eval_results_text.delete('1.0', tk.END))
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
                self.root.after(0, lambda: messagebox.showerror(
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
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = self.eval_test_folder.get()

            for idx, session in enumerate(sessions, 1):
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
                if pf and os.path.isdir(pf):
                    cache_dir = os.path.join(pf, 'features')
                else:
                    cache_dir = os.path.join(os.path.dirname(video_path), 'features')
                os.makedirs(cache_dir, exist_ok=True)

                cfg_key = {
                    'bp_include_list': clf_data.get('bp_include_list'),
                    'bp_pixbrt_list':  clf_data.get('bp_pixbrt_list', []),
                    'square_size':     clf_data.get('square_size', [40]),
                    'pix_threshold':   clf_data.get('pix_threshold', 0.3),
                    'crop_offset':     (crop_x, crop_y),
                }
                cfg_hash   = hashlib.md5(repr(cfg_key).encode()).hexdigest()[:8]
                cache_file = os.path.join(cache_dir,
                                          f'{base_name}_features_{cfg_hash}.pkl')

                if os.path.isfile(cache_file):
                    self._log('  Loading cached features…')
                    try:
                        with open(cache_file, 'rb') as f:
                            X = pickle.load(f)
                        self._log(
                            f'  ✓ Cache loaded: {X.shape[0]} frames, '
                            f'{X.shape[1]} features')
                    except Exception as e:
                        self._log(f'  ⚠️  Cache load failed ({e}), re-extracting…')
                        X = None

                if X is None:
                    self._log('  Extracting features (this may take a while)…')
                    try:
                        X = _extract(
                            pose_data_file=dlc_path,
                            video_file_path=video_path,
                            bp_include_list=clf_data.get('bp_include_list'),
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            use_gpu=True,
                            crop_offset_x=crop_x,
                            crop_offset_y=crop_y,
                            config_yaml_path=dlc_config_path or None,
                        )
                        with open(cache_file, 'wb') as f:
                            pickle.dump(X, f)
                        self._log(
                            f'  ✓ Features extracted & cached: '
                            f'{X.shape[0]} frames, {X.shape[1]} features')
                    except Exception as e:
                        self._log(f'  ✗ Feature extraction failed: {e}  — skipping.')
                        continue

                # ── Predict ──────────────────────────────────────────────
                try:
                    y_proba = _predict(model, X)
                except Exception as e:
                    self._log(f'  ✗ Prediction failed: {e}  — skipping.')
                    continue

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
                self._log(
                    f'  F1={f1:.3f}  Prec={prec:.3f}  '
                    f'Rec={rec:.3f}  Acc={acc:.3f}')

                all_y_true.extend(y_true.tolist())
                all_y_pred.extend(y_pred.tolist())

                per_video_results.append({
                    'video':     video_name,
                    'n_frames':  n,
                    'n_true':    int(np.sum(y_true)),
                    'n_pred':    int(np.sum(y_pred)),
                    'f1':        f1,
                    'precision': prec,
                    'recall':    rec,
                    'accuracy':  acc,
                    'y_true':    y_true,
                    'y_pred':    y_pred,
                    'y_proba':   y_proba_clipped,
                    'base_name': base_name,
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
                    f'{"F1":>7} {"Prec":>7} {"Rec":>7} {"Acc":>7}')
                self._log('-' * 80)
                for r in per_video_results:
                    true_pct = 100 * r['n_true'] / r['n_frames'] if r['n_frames'] else 0
                    self._log(
                        f'{r["video"][:35]:<35} {r["n_frames"]:>7} '
                        f'{true_pct:>8.1f}% {r["f1"]:>7.3f} '
                        f'{r["precision"]:>7.3f} {r["recall"]:>7.3f} '
                        f'{r["accuracy"]:>7.3f}')

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

            self._log('\n✓ Evaluation complete!')
            self.root.after(0, lambda: messagebox.showinfo(
                'Evaluation Complete',
                f'Evaluated {len(per_video_results)} session(s).\n\n'
                f'Overall F1:        {ov_f1:.4f}\n'
                f'Overall Precision: {ov_prec:.4f}\n'
                f'Overall Recall:    {ov_rec:.4f}\n'
                f'Overall Accuracy:  {ov_acc:.4f}\n\n'
                f'Results saved to:\n{output_folder}'))

        except Exception as e:
            err = traceback.format_exc()
            self._log(f'\n✗ ERROR:\n{err}')
            self.root.after(0, lambda: messagebox.showerror(
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

    def _generate_plots(self, per_video_results, all_y_true, all_y_pred,
                        cm, behavior_name, output_folder):
        """
        Save a two-panel PNG:
          Left  — aggregate confusion matrix heatmap
          Right — per-video F1 / Precision / Recall bar chart
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Evaluation — {behavior_name}', fontsize=13, fontweight='bold')

        # Confusion matrix
        ax = axes[0]
        if cm is not None and cm.size == 4:
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title('Confusion Matrix (aggregate)')
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
            ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Score')
            ax2.set_title('Per-video Performance')
            ax2.legend()
            if f1s:
                ax2.axhline(y=np.mean(f1s), color='steelblue',
                            linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, 'No per-video data',
                     ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        plot_path = os.path.join(
            output_folder, f'evaluation_plots_{behavior_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self._log(f'  ✓ {os.path.basename(plot_path)}')

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
        dialog.geometry('500x310')
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
                self.root.after(0, lambda m=msg: (
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
                    self.root.after(0, lambda: messagebox.showerror(
                        'Error', 'No training data found in classifier file.'))
                    return

            elif source == 'test':
                test_folder = self.eval_test_folder.get()
                if not test_folder:
                    self.root.after(0, lambda: messagebox.showerror(
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
                    self.root.after(0, lambda: messagebox.showerror(
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
                            self.root.after(0, lambda ks=list(data.keys()):
                                messagebox.showerror(
                                    'Error',
                                    f'Unknown dict format. Keys: {ks}\n\n'
                                    f"Expected 'X' or 'X_train'"))
                            return
                    else:
                        X = data
                    _log_shap(f'✓ Loaded features from: {os.path.basename(file_path)}')
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
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
                self.root.after(0, lambda: messagebox.showerror(
                    'Error', 'No numeric features found in the selected data.'))
                return

            _log_shap(f'Data shape: {X.shape[0]} rows × {X.shape[1]} features')

            # Sub-sample if large
            if len(X) > n_samples:
                X = X.sample(n=n_samples, random_state=42)
                _log_shap(f'Using {n_samples} random samples')

            _log_shap('\nComputing SHAP values (this may take a minute)…')
            self.root.after(0, lambda: self.eval_results_text.see(tk.END))

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
            shap.summary_plot(shap_values, X, plot_type='bar', show=False)
            mpl_plt.tight_layout()
            mpl_plt.savefig(os.path.join(output_dir, '1_feature_importance.png'),
                            dpi=150, bbox_inches='tight')
            mpl_plt.close()
            _log_shap('✓ Feature importance plot saved')

            # 2. Beeswarm / feature effects
            mpl_plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            mpl_plt.tight_layout()
            mpl_plt.savefig(os.path.join(output_dir, '2_feature_effects_beeswarm.png'),
                            dpi=150, bbox_inches='tight')
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
                    dpi=150, bbox_inches='tight')
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

            self.root.after(0, lambda: messagebox.showinfo(
                'SHAP Analysis Complete',
                f'Results saved to:\n{output_dir}\n\n'
                'Files generated:\n'
                '  • 1_feature_importance.png\n'
                '  • 2_feature_effects_beeswarm.png\n'
                '  • 3_dependence_*.png (top 10 features)\n'
                '  • SHAP_report.txt'))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                'SHAP Error', f'Error during SHAP analysis:\n\n{str(e)}'))
            traceback.print_exc()
