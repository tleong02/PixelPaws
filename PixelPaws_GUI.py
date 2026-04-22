"""
PixelPaws - Complete Integrated GUI Application
Automated animal behavior analysis using machine learning

Enhanced features:
1. Video Preview with Predictions - Visual validation with overlay
2. Real-Time Training Visualization - Live progress plots
3. Auto-Label Suggestion Mode - Intelligent labeling assistance
4. Behavior Ethogram Generator - Automated analysis outputs
5. Data Quality Checker - Pre-training validation
6. Dark Mode - Eye-friendly interface

Modular feature extraction system:
- pose_features.py - Pose/kinematic features (distances, angles, velocities)
- brightness_features.py - Pixel brightness features (light-based depth analysis)
- classifier_training.py - XGBoost training pipeline

A unified interface for training classifiers, evaluating models, and analyzing
animal behavior videos using DeepLabCut pose estimation.
"""

import os
import sys
import glob
import hashlib
import pickle
import time
import re
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.font import Font

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True

    # ttkbootstrap's LabelFrame wrapper does not accept 'padding'.
    # Patch it so existing code using padding=N keeps working.
    _OrigLabelFrame = ttk.LabelFrame

    class _PatchedLabelFrame(_OrigLabelFrame):
        def __init__(self, *args, **kw):
            kw.pop('padding', None)
            super().__init__(*args, **kw)

    ttk.LabelFrame = _PatchedLabelFrame

    # ttkbootstrap uses Panedwindow (lowercase w); alias for compatibility
    if not hasattr(ttk, 'PanedWindow') and hasattr(ttk, 'Panedwindow'):
        ttk.PanedWindow = ttk.Panedwindow
except ImportError:
    from tkinter import ttk
    TTKBOOTSTRAP_AVAILABLE = False

import numpy as np
import pandas as pd
import cv2

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from analysis_tab import AnalysisTab
    ANALYSIS_TAB_AVAILABLE = True
except ImportError:
    ANALYSIS_TAB_AVAILABLE = False
    print("Warning: analysis_tab.py not found. Analysis tab will be disabled.")

try:
    from evaluation_tab import EvaluationTab, _apply_bout_filtering, count_bouts, find_session_triplets, fit_hmm_transitions
    EVALUATION_TAB_AVAILABLE = True
    _FIND_SESSIONS_AVAILABLE = True
except ImportError:
    EVALUATION_TAB_AVAILABLE = False
    _FIND_SESSIONS_AVAILABLE = False
    find_session_triplets = None
    print("Warning: evaluation_tab.py not found. Evaluation tab will be disabled.")

try:
    from unsupervised_tab import UnsupervisedTab
    UNSUPERVISED_TAB_AVAILABLE = True
except ImportError:
    UNSUPERVISED_TAB_AVAILABLE = False

try:
    from gait_limb_tab import GaitLimbTab
    GAIT_LIMB_TAB_AVAILABLE = True
except ImportError:
    GAIT_LIMB_TAB_AVAILABLE = False

try:
    from transitions_tab import TransitionsTab
    TRANSITIONS_TAB_AVAILABLE = True
except ImportError:
    TRANSITIONS_TAB_AVAILABLE = False

try:
    from feature_cache import FeatureCacheManager
    FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FeatureCacheManager = None
    FEATURE_CACHE_AVAILABLE = False

try:
    from project_config import ProjectConfig
    PROJECT_CONFIG_AVAILABLE = True
except ImportError:
    ProjectConfig = None
    PROJECT_CONFIG_AVAILABLE = False

# ============================================================================
# SMART DEFAULT FEATURE EXTRACTION SETTINGS
# ============================================================================
# These defaults balance speed, storage, and reusability
# Strategy: Extract MAXIMUM useful features once, let classifiers select what they need

# Brightness bodyparts (common across most classifiers)
DEFAULT_BRIGHTNESS_BODYPARTS = ['hrpaw', 'hlpaw', 'snout']
DEFAULT_SQUARE_SIZE = [40]
DEFAULT_PIX_THRESHOLD = 0.3

# Feature extraction strategy:
# 1. Extract ALL pose features from DLC file (all available bodyparts)
# 2. Extract brightness features for DEFAULT_BRIGHTNESS_BODYPARTS
# 3. Cache with video-based hash (not classifier-specific)
# 4. When predicting:
#    a. Check if cached features are a SUPERSET of what classifier needs
#    b. If yes: SELECT required columns from cached features
#    c. If no: Re-extract with classifier-specific bodyparts
#
# This allows:
# - BAREfoot (339 features) can use PixelPaws cache (375 features) by selecting 339 columns
# - PixelPaws (375 features) needs all 375, uses full cache
# - Both share same cache file!
# ============================================================================

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


try:
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    from sklearn.model_selection import KFold
    from sklearn.utils import resample
except ImportError:
    pass

# Import new PixelPaws modules
try:
    from pose_features import PoseFeatureExtractor, POSE_FEATURE_VERSION
    from brightness_features import PixelBrightnessExtractor
    from classifier_training import BehaviorClassifier
    PIXELPAWS_MODULES_AVAILABLE = True
    pass  # modules loaded
except ImportError as e:
    PIXELPAWS_MODULES_AVAILABLE = False
    POSE_FEATURE_VERSION = 5  # fallback if module unavailable
    print(f"Error: Could not import PixelPaws modules: {e}")
    print("Please ensure pose_features.py, brightness_features.py, and classifier_training.py are in the same directory")

# Import Active Learning v2 module
try:
    from active_learning_v2 import (ALSessionV2, LabelingInterface, BoutLabelingInterface,
                                     run_directed_discovery)
    ACTIVE_LEARNING_AVAILABLE = True
    pass  # module loaded
except ImportError as e:
    ACTIVE_LEARNING_AVAILABLE = False
    print(f"Note: Active Learning v2 module not available: {e}")

try:
    from project_setup import KeyFileGeneratorDialog
    _KEY_FILE_DIALOG_AVAILABLE = True
except ImportError:
    _KEY_FILE_DIALOG_AVAILABLE = False


# ============================================================================
# ToolTip + shared UI helpers (from ui_utils)
# ============================================================================

from ui_utils import (ToolTip, bind_mousewheel,
                      _bind_tight_layout_on_resize, _draw_canvas_fit)
from sidebar_nav import SidebarNav


# ============================================================================
# Helper Functions
# ============================================================================

def _atomic_pickle_save(data, target_path):
    """Write pickle atomically: temp file in same dir -> os.replace.

    Prevents data loss if the process crashes mid-write (the old file is
    only replaced once the new one is fully flushed to disk).
    """
    import tempfile
    dir_path = os.path.dirname(target_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, target_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------
# Prediction pipeline — moved to prediction_pipeline.py.
# Re-exported here so existing `from PixelPaws_GUI import X` still works.
# ----------------------------------------------------------------------------
from prediction_pipeline import (
    clean_bodyparts_list,
    extract_subject_id_from_filename,
    auto_detect_bodyparts_from_model,
    PixelPaws_ExtractFeatures,
    predict_with_xgboost,
    augment_features_post_cache,
    _load_features_for_prediction,
    check_classifier_portability,
    apply_smoothing,
)


# ----------------------------------------------------------------------------
# Dialog / window classes — moved to dialogs.py.
# ----------------------------------------------------------------------------
from dialogs import (
    Theme,
    VideoPreviewWindow,
    TrainingVisualizationWindow,
    AutoLabelWindow,
    SideBySidePreview,
    DataQualityChecker,
    EthogramGenerator,
    ConfidenceHistogramDialog,
    OVERLAY_COLOR_SCHEMES,
)


class PixelPawsGUI:
    """Complete integrated application class with all enhanced features"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PixelPaws - Behavioral Analysis & Recognition")
        # Size to 80% of screen, centered
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = int(sw * 0.8)
        h = int(sh * 0.8)
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        
        # Set application icon
        self.set_app_icon()
        
        # Theme
        self.theme = Theme('light')

        # Application state
        self.project_folder = tk.StringVar()
        self.classifier_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self._project_display_name = tk.StringVar(value="")

        # ── Shared project folder ─────────────────────────────────────────────
        # A single StringVar written by any "browse project" action; observed by
        # training, batch, and evaluation tabs so users don't re-enter it.
        self.current_project_folder = tk.StringVar()
        self.current_project_folder.trace_add('write', self._on_project_folder_changed)

        # Key file: {Subject: Treatment} — loaded from <project>/key_file.csv
        self.key_file_data = {}
        
        # Training visualization window
        self.train_viz_window = None
        
        # Initialize training variables (will be created in create_training_tab)
        self.train_project_folder = None
        self.train_single_folder = None
        self.train_video_ext = None
        self.train_behavior_name = None
        self.train_min_bout = None
        self.train_min_after_bout = None
        self.train_max_gap = None
        self.train_bp_pixbrt = None
        self.train_square_sizes = None
        self.train_pix_threshold = None
        self.train_n_estimators = None
        self.train_max_depth = None
        self.train_learning_rate = None
        self.train_subsample = None
        self.train_colsample = None
        self.train_n_folds = None
        self.train_use_balancing = None
        self.train_imbalance_thresh = None
        self.train_use_scale_pos_weight = None
        self.train_use_early_stopping = None
        self.train_early_stopping_rounds = None
        self.train_generate_plots = None
        self.train_opt_strategy = None
        self.train_trim_to_last_positive = None
        self.train_use_optuna = None
        self.train_optuna_trials = None
        self.train_use_lag_features = None
        self.train_use_egocentric = None
        self.train_use_contact_features = None
        self.train_contact_threshold = None
        self.train_correlation_filter = None
        self.train_use_calibration = None
        self.train_use_fold_ensemble = None
        self._train_warning_lbl = None
        self.pred_smoothing_mode = None
        self._session_tree = None
        self._session_checked = {}
        self._scanned_sessions = []
        self._session_count_label = None
        self._sess_expanded = False
        self._sess_toggle_btn = None
        self._sess_content_frame = None
        self.train_learning_curve = None
        self.train_log = None
        
        # Cancel flags for long-running threads
        self._batch_cancel_flag = threading.Event()
        self._predict_cancel_flag = threading.Event()
        self._training_cancel_flag = threading.Event()
        self._feature_cancel_flag = threading.Event()

        # Initialize batch variables (will be created in create_batch_tab)
        self.batch_folder = None
        self.batch_video_ext = None
        self.batch_prefer_filtered = None
        self.batch_clf_listbox = None
        self.batch_classifiers = {}
        self.batch_save_labels = None
        self.batch_graph_settings = {}
        self.batch_progress_label = None
        self.batch_progress = None
        self.batch_log = None
        
        # Initialize evaluation variables (will be created in create_evaluation_tab)
        self.eval_classifier_path = None
        self.eval_test_folder = None
        self.eval_video_ext = None
        self.eval_dlc_config_path = None
        self.eval_info_text = None
        self.eval_generate_plots = None
        self.eval_save_predictions = None
        self.eval_detailed_report = None
        self.eval_apply_bout_filter = None
        self.eval_results_text = None
        
        # Initialize prediction variables (will be created in create_prediction_tab)
        self.pred_classifier_options = {}
        self.pred_classifier_path = None
        self.pred_video_path = None
        self.pred_dlc_path = None
        self.pred_output_folder = None
        self.pred_save_csv = None
        self.pred_save_video = None
        self.pred_save_summary = None
        self.pred_generate_ethogram = None
        self.pred_results_text = None

        # Last prediction results (populated by _predict_thread on success)
        self._last_pred_y_pred        = None
        self._last_pred_y_proba       = None
        self._last_pred_fps           = None
        self._last_pred_n_frames      = None
        self._last_pred_video_path    = None
        self._last_pred_behavior_name = None
        self._last_pred_output_folder = None
        self._last_pred_base_name     = None
        self._last_pred_dlc_path      = None          # stored so export can reload DLC coords
        self._last_pred_crop_offset   = (0, 0)        # (x1, y1) from DLC config crop
        self.lv_skeleton_dots         = tk.BooleanVar(value=True)
        self.lv_frame_tint            = tk.BooleanVar(value=False)
        self.lv_timeline_strip        = tk.BooleanVar(value=True)
        self.lv_halo_border           = tk.BooleanVar(value=True)
        self.lv_bout_counter          = tk.BooleanVar(value=False)
        self.lv_behavior_color        = tk.StringVar(value='#E00000')   # red   → (0,0,224) BGR
        self.lv_nobehavior_color      = tk.StringVar(value='#707070')   # gray  → (112,112,112) BGR
        self.lv_color_scheme          = tk.StringVar(value='Red / Gray')
        self.lv_tint_opacity          = tk.DoubleVar(value=0.22)
        self.lv_hud_position          = tk.StringVar(value='top')
        self.lv_preview_photo         = None   # GC anchor for preview PhotoImage

        # Setup UI
        self.setup_ui()
        self.apply_theme()

        # Ensure global classifiers folder exists
        try:
            from user_config import get_global_classifiers_folder
            os.makedirs(get_global_classifiers_folder(), exist_ok=True)
        except OSError:
            pass

        # Show startup wizard (hides root until a project is chosen)
        self.root.after(100, self._show_startup_wizard)
    
    def set_app_icon(self):
        """Set paw print app icon."""
        # 64x64 PNG rendered from Segoe UI Emoji 🐾
        _PAW_PNG_B64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAAXNSR0IArs4c6QAAAARnQU1B"
            "AACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAPSSURBVHhe7ZiBjdVADESvABqgARqg"
            "ASqgAjqgA1qgBWqgCbqgmSNPwtJoZG82P8mPI/2RRvoXb/bW3rHXm7cXXnjhhY34uPDbwu//"
            "yW+eBdbst8aPhe8Fsa3ZPyy8LX4vzBzbwj8LbxkEpJw59AhRwq3AjrkTXxYGPi90O88Cmf1W"
            "NYEipov/utChTq7ZISr4tLA1KtlXmLUr/y5UNbXBqOBVRWzWnvHnwjYYHWUQZTiQfdhJGYen"
            "UcZs3qeD4pQtjByGv+QZOw0z5yo778dcjNF3YKWep8EXxUIdEYStzGTuwcvU81Ro7lOgMhAU"
            "XfQsq6NPx1zeI2gA+J0hCwBjlW6Hlbx1DOq6FMhUF5QtWtOE8ZVjnk5Zf0AvoGN451JoNYf0"
            "7ipdzuywZQ45fD6tKczL/Gpv0Rz5oiCypibE3zPOB7KgVv+jBVyWzi3OBzwITgJy+RGoIAjZ"
            "Lj3ifKAKAjvfynnAgnyhe5wPZEFoeTP0BuXI6uxzt/w+oGd51RDtgRbUM+bfjVgcPGOHvEdo"
            "B13cGc1J+wCoRGfu6hQynIIzRc07znbwG9/IqaxvGHV0fsK0aYAUfuGpGpVR05RdpZnD+4sj"
            "jtdToCdBkIKIY9wJXMYZGcNY3sm+NBGMDASWd2AV+JH9ELhU14gzWfc4oqcKatD6EyQlGTuy"
            "n9JQMemMUywgdmJGGTjhzs+8t8bT0omJs0DwLPusjTSzFMJxTgqXbZYej9IDeyhYeOTeLNZy"
            "FZW5EwSJdwhuFsiRveWpMoLfDTIZ67GcKc6DUAW7FdhB7zdIkQyMHdkJis4DSc/LvzJXqApe"
            "Jd8IwJo9I4E4tS5sxVq1z+RLWozs1ASdI2OLtMh2ioDgQASGHdPF+jsjOzbm4lTxfoH5L4fn"
            "vBc8pBo25I5DOl7pdneQIPn7l6tAF0MwMqylSMXMOS+O2QnyVOhikGqGmXx2VsXRe4zqfz4N"
            "uphKAdoVMoYcV7DTWhSDt1DAlhqwVrQ8x29RA2ZOgXg2A3eS361PAaCOZty60GynM16++4oq"
            "CI/u0igIKKFVJwiyG+BeiWZBwPlWOx9wBex1PoCznvvtdh/oAlnwkfBCS0FsBZd/dv/fC02F"
            "qkm6DL5D3ugcAZxuGwBtduBsd0agZvO5dQCABmCtAOK0Snotpz3F2tUA4KdAVQdcLcFR0HT3"
            "4RkpthuZY7Sw7B7gOMsuPEocVedIJe8DWso/sPXa6+f7DGdrxmXwVKjIzs6oQtne+QDSHe2u"
            "f8QgTTzPlQQ1UulWYMf4jo/DkMCM+nicRBExnt+3dPyFFy7D29s/u81UFjhJd+cAAAAASUVO"
            "RK5CYII="
        )
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ico_path = os.path.join(script_dir, "pixelpaws_icon.ico")

            # Windows: give the process its own App User Model ID so Windows
            # shows it as a distinct taskbar entry instead of grouping it under
            # python.exe, then load the ICO and send WM_SETICON directly.
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    'PixelPaws.BehaviorAnalysis.1'
                )
                if os.path.exists(ico_path):
                    self.root.update_idletasks()  # ensure HWND exists
                    hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                    if not hwnd:
                        hwnd = self.root.winfo_id()
                    hicon = ctypes.windll.user32.LoadImageW(
                        None, ico_path, 1, 0, 0, 0x10  # IMAGE_ICON, LR_LOADFROMFILE
                    )
                    if hicon:
                        ctypes.windll.user32.SendMessageW(hwnd, 0x80, 1, hicon)  # ICON_BIG
                        ctypes.windll.user32.SendMessageW(hwnd, 0x80, 0, hicon)  # ICON_SMALL
            except Exception:
                pass

            # Title bar icon via tkinter (works on all platforms)
            if os.path.exists(ico_path):
                try:
                    self.root.iconbitmap(ico_path)
                except Exception:
                    pass
            photo = tk.PhotoImage(data=_PAW_PNG_B64)
            self.root.iconphoto(True, photo)
            self.root._icon_photo = photo  # prevent GC

        except Exception as e:
            print(f"Could not set icon: {e}")
        
    def setup_ui(self):
        """Create the main UI layout"""
        
        # Create menu bar
        self.create_menu()
        
        # Main container with sidebar navigation
        self.notebook = SidebarNav(self.root, width=280, groups={
            "Train & Label": ["🎓 Train Classifier", "🧠 Active Learning"],
            "Predict & Evaluate": ["🎬 Predict", "📊 Evaluate", "📦 Batch"],
            "Analyze": ["📈 Analysis", "🔀 Transitions"],
            "Discover": ["🔍 Discover"],
            "Locomotion": ["🐾 Gait & Limb Use"],
            "Tools": ["🛠 Tools"],
        })
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs (workflow order: Train → Predict → Evaluate → Batch)
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_evaluation_tab()
        self.create_batch_tab()
        
        # Analysis tab (for batch analysis and graphing)
        if ANALYSIS_TAB_AVAILABLE:
            self.analysis_tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.analysis_tab_frame, text="📈 Analysis")
            self.analysis_tab = AnalysisTab(self.analysis_tab_frame, self)
            self.analysis_tab.pack(fill='both', expand=True)

        # Transitions tab (state transition analysis)
        if TRANSITIONS_TAB_AVAILABLE:
            self.transitions_tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.transitions_tab_frame, text="🔀 Transitions")
            self.transitions_tab = TransitionsTab(self.transitions_tab_frame, self)
            self.transitions_tab.pack(fill='both', expand=True)

        # Active Learning tab (after Analysis, before Discover)
        self._create_active_learning_tab_v2()

        # Unsupervised behavior discovery tab
        if UNSUPERVISED_TAB_AVAILABLE:
            self.unsupervised_tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.unsupervised_tab_frame, text="🔍 Discover")
            self.unsupervised_tab = UnsupervisedTab(self.unsupervised_tab_frame, self)
            self.unsupervised_tab.pack(fill='both', expand=True)

        # Gait & Limb Use analysis tab
        if GAIT_LIMB_TAB_AVAILABLE:
            self.wb_tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.wb_tab_frame, text="🐾 Gait & Limb Use")
            self.wb_tab = GaitLimbTab(self.wb_tab_frame, self)
            self.wb_tab.pack(fill='both', expand=True)

        self.create_tools_tab()  # New tab for enhanced tools

        # Hide sidebar items for unavailable modules
        if not ANALYSIS_TAB_AVAILABLE:
            self.notebook.hide_item("📈 Analysis")
        if not UNSUPERVISED_TAB_AVAILABLE:
            self.notebook.hide_item("🔍 Discover")
        if not GAIT_LIMB_TAB_AVAILABLE:
            self.notebook.hide_item("🐾 Gait & Limb Use")
        if not TRANSITIONS_TAB_AVAILABLE:
            self.notebook.hide_item("🔀 Transitions")

        # Status bar at bottom
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=5, pady=5)

        status_inner = ttk.Frame(status_frame)
        status_inner.pack(fill='x')
        ttk.Label(status_inner, textvariable=self._project_display_name,
                 font=('Arial', 9, 'bold'), anchor='w').pack(side='left', padx=(4, 10))
        ttk.Label(status_inner, textvariable=self.status_text,
                 relief='sunken', anchor='w').pack(fill='x', expand=True)
        
    def create_menu(self):
        """Create application menu bar"""
        import tkinter.font as tkFont

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ── File menu ──────────────────────────────────────────────
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Project\u2026", command=self._show_startup_wizard)

        # Recent Projects submenu — rebuilt each time the menu is opened
        self._recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self._recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Open Project Folder",
                              command=self._open_project_folder)
        file_menu.add_command(label="Export Project as ZIP\u2026",
                              command=self._export_project_zip)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Populate recent projects dynamically
        def _refresh_recent():
            self._recent_menu.delete(0, 'end')
            try:
                from project_setup import _load_recent
                recents = _load_recent()
            except Exception:
                recents = []
            if not recents:
                self._recent_menu.add_command(label="(none)", state='disabled')
            else:
                for path in recents:
                    self._recent_menu.add_command(
                        label=path,
                        command=lambda p=path: self._open_recent_project(p))
        file_menu.configure(postcommand=_refresh_recent)

        # ── View menu ─────────────────────────────────────────────
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_theme,
                              accelerator="Ctrl+D")
        view_menu.add_separator()

        # Font size controls
        self._base_font_size = tkFont.nametofont('TkDefaultFont').actual()['size']
        self._current_font_size = self._base_font_size
        view_menu.add_command(label="Increase Font Size",
                              command=lambda: self._change_font_size(1),
                              accelerator="Ctrl++")
        view_menu.add_command(label="Decrease Font Size",
                              command=lambda: self._change_font_size(-1),
                              accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Font Size",
                              command=lambda: self._change_font_size(0))
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Sidebar",
                              command=lambda: self.notebook.toggle_collapse()
                              if hasattr(self.notebook, 'toggle_collapse') else None)

        # Keyboard accelerators for font size
        self.root.bind_all('<Control-plus>', lambda e: self._change_font_size(1))
        self.root.bind_all('<Control-equal>', lambda e: self._change_font_size(1))
        self.root.bind_all('<Control-minus>', lambda e: self._change_font_size(-1))
        self.root.bind_all('<Control-d>', lambda e: self.toggle_theme())

        # ── Tools menu ────────────────────────────────────────────
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Quality Checker",
                               command=self.open_quality_checker)
        tools_menu.add_command(label="Auto-Label Assistant",
                               command=self.open_auto_labeler)
        tools_menu.add_command(label="Video Preview",
                               command=self.open_video_preview)
        tools_menu.add_separator()
        tools_menu.add_command(label="Skeleton Video Renderer",
                               command=self.open_skeleton_renderer)
        tools_menu.add_command(label="Brightness Preview",
                               command=self.show_brightness_preview)
        tools_menu.add_command(label="Crop Video for DLC\u2026",
                               command=self.crop_video_for_dlc)
        tools_menu.add_separator()
        tools_menu.add_command(label="Key File (Group Assignment)\u2026",
                               command=self._open_key_file_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Generate Ethogram (Coming Soon)",
                               command=self.generate_ethogram)

        # ── Help menu ─────────────────────────────────────────────
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="Keyboard Shortcuts",
                              command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates\u2026",
                              command=self._check_for_updates)
    
    # ── Menu handler helpers ──────────────────────────────────────────

    def _open_recent_project(self, folder_path):
        """Load a project from the recent-projects list."""
        if not os.path.isdir(folder_path):
            messagebox.showerror("Not Found",
                                 f"Project folder no longer exists:\n{folder_path}")
            return
        self.current_project_folder.set(folder_path)
        config_path = os.path.join(folder_path, 'PixelPaws_project.json')
        if os.path.isfile(config_path):
            self._load_project_config(config_path, silent=True)

    def _open_project_folder(self):
        """Open the current project folder in the system file manager."""
        folder = self.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("No Project",
                                "No project folder is currently loaded.")
            return
        os.startfile(folder)

    def _export_project_zip(self):
        """Export the current project folder as a ZIP archive."""
        import shutil
        folder = self.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("No Project",
                                "No project folder is currently loaded.")
            return
        default_name = os.path.basename(folder)
        dest = filedialog.asksaveasfilename(
            title="Export Project as ZIP",
            defaultextension='.zip',
            initialfile=default_name,
            filetypes=[("ZIP archive", "*.zip"), ("All files", "*.*")])
        if not dest:
            return
        # Strip .zip — shutil.make_archive adds it
        if dest.lower().endswith('.zip'):
            dest = dest[:-4]

        def _do_zip():
            try:
                shutil.make_archive(dest, 'zip', root_dir=os.path.dirname(folder),
                                    base_dir=os.path.basename(folder))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Export Complete", f"Project exported to:\n{dest}.zip"))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror(
                    "Export Failed", str(exc)))

        threading.Thread(target=_do_zip, daemon=True).start()

    def _change_font_size(self, delta):
        """Increase (+1), decrease (-1), or reset (0) the UI font size."""
        import tkinter.font as tkFont
        if delta == 0:
            self._current_font_size = self._base_font_size
        else:
            self._current_font_size = max(6, self._current_font_size + delta)
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont',
                     'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont',
                     'TkIconFont', 'TkTooltipFont', 'TkFixedFont'):
            try:
                tkFont.nametofont(name).configure(size=self._current_font_size)
            except Exception:
                pass

    def _check_for_updates(self):
        """Check GitHub releases for a newer version of PixelPaws."""
        import urllib.request, json as _json
        _CURRENT = "1.0.0"
        _REPO = "https://api.github.com/repos/rslivicki/PixelPaws/releases/latest"
        try:
            req = urllib.request.Request(_REPO,
                                        headers={'User-Agent': 'PixelPaws'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())
            tag = data.get('tag_name', '').lstrip('vV')
            if tag and tag != _CURRENT:
                messagebox.showinfo("Update Available",
                                    f"A new version is available: v{tag}\n"
                                    f"You are running v{_CURRENT}\n\n"
                                    f"Visit the GitHub releases page to download.")
            else:
                messagebox.showinfo("Up to Date",
                                    f"You are running the latest version (v{_CURRENT}).")
        except Exception:
            messagebox.showinfo("Check for Updates",
                                "Could not reach GitHub.\n"
                                "Check your internet connection and try again.")

    def create_training_tab(self):
        """Create the classifier training tab with all options"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="🎓 Train Classifier")
        
        # Create scrollable canvas
        canvas = tk.Canvas(train_frame)
        scrollbar = ttk.Scrollbar(train_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === PROJECT SETUP ===
        setup_frame = ttk.LabelFrame(scrollable_frame, text="Project Setup", padding=10)
        setup_frame.pack(fill='x', padx=15, pady=8)
        
        ttk.Label(setup_frame, text="Project Folder:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_project_folder = tk.StringVar()
        ttk.Entry(setup_frame, textvariable=self.train_project_folder, width=50).grid(
            row=0, column=1, columnspan=2, padx=5, pady=2)
        ttk.Button(setup_frame, text="📁 Browse", 
                  command=self.browse_train_project).grid(row=0, column=3, pady=2)
        
        # Folder structure option
        self.train_single_folder = tk.BooleanVar(value=False)
        ttk.Checkbutton(setup_frame, text="All files in single folder (no Videos/Targets subfolders)",
                       variable=self.train_single_folder).grid(row=1, column=1, columnspan=2, sticky='w', pady=2)
        
        # Video extension
        ttk.Label(setup_frame, text="Video Extension:").grid(row=2, column=0, sticky='w', pady=2)
        self.train_video_ext = ttk.Combobox(setup_frame, values=['.mp4', '.avi'], width=10)
        self.train_video_ext.set('.mp4')
        self.train_video_ext.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # === BEHAVIOR CONFIGURATION ===
        behavior_frame = ttk.LabelFrame(scrollable_frame, text="Behavior Configuration", padding=10)
        behavior_frame.pack(fill='x', padx=15, pady=8)
        
        ttk.Label(behavior_frame, text="Behavior Name:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_behavior_name = tk.StringVar(value="")
        ttk.Entry(behavior_frame, textvariable=self.train_behavior_name, width=30).grid(
            row=0, column=1, padx=5, pady=2, sticky='w')
        
        # Auto-detect button
        ttk.Button(behavior_frame, text="🔍 Auto-Detect", 
                  command=self.auto_detect_behavior_names, width=12).grid(
            row=0, column=2, padx=5, pady=2, sticky='w')
        
        ttk.Label(behavior_frame, text="(Must match CSV column name)", 
                 foreground='gray').grid(row=0, column=3, sticky='w')
        
        # Filtering parameters
        ttk.Label(behavior_frame, text="Min Bout (frames):").grid(row=2, column=0, sticky='w', pady=2)
        self.train_min_bout = tk.IntVar(value=3)
        ttk.Spinbox(behavior_frame, from_=1, to=100, textvariable=self.train_min_bout, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(behavior_frame, text="Minimum consecutive frames for valid bout", 
                 foreground='gray').grid(row=2, column=3, sticky='w')
        
        ttk.Label(behavior_frame, text="Min After Bout (frames):").grid(row=3, column=0, sticky='w', pady=2)
        self.train_min_after_bout = tk.IntVar(value=1)
        ttk.Spinbox(behavior_frame, from_=1, to=100, textvariable=self.train_min_after_bout, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(behavior_frame, text="Minimum frames after bout ends", 
                 foreground='gray').grid(row=3, column=3, sticky='w')
        
        ttk.Label(behavior_frame, text="Max Gap (frames):").grid(row=4, column=0, sticky='w', pady=2)
        self.train_max_gap = tk.IntVar(value=5)
        ttk.Spinbox(behavior_frame, from_=0, to=100, textvariable=self.train_max_gap, width=10).grid(
            row=4, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(behavior_frame, text="Maximum frames to bridge between bouts", 
                 foreground='gray').grid(row=4, column=3, sticky='w')
        
        # Auto-suggest button
        ttk.Button(behavior_frame, text="🤖 Auto-Suggest Bout Parameters", 
                  command=self.auto_suggest_bout_params).grid(row=5, column=0, columnspan=4, pady=10)
        
        # === FEATURE CONFIGURATION ===
        feature_frame = ttk.LabelFrame(scrollable_frame, text="Feature Configuration", padding=10)
        feature_frame.pack(fill='x', padx=15, pady=8)
        self._feature_frame = feature_frame
        
        ttk.Label(feature_frame, text="Pixel Brightness Body Parts:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_bp_pixbrt = tk.StringVar(value="hrpaw,hlpaw,snout")
        ttk.Entry(feature_frame, textvariable=self.train_bp_pixbrt, width=30).grid(
            row=0, column=1, padx=5, pady=2, sticky='w')
        ttk.Label(feature_frame, text="Body parts for brightness analysis (comma-separated)", foreground='gray').grid(row=0, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="Square Sizes:").grid(row=1, column=0, sticky='w', pady=2)
        self.train_square_sizes = tk.StringVar(value="40,40,40")
        _sz_row = ttk.Frame(feature_frame)
        _sz_row.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        ttk.Entry(_sz_row, textvariable=self.train_square_sizes, width=22).pack(side='left')
        ttk.Button(_sz_row, text="👁 Preview ROIs",
                   command=self._open_roi_preview_dialog).pack(side='left', padx=(6, 0))
        ttk.Label(feature_frame, text="Window size for each body part (pixels) — preview overlays boxes on a video frame", foreground='gray').grid(row=1, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="Pixel Threshold:").grid(row=2, column=0, sticky='w', pady=2)
        self.train_pix_threshold = tk.DoubleVar(value=0.3)
        ttk.Entry(feature_frame, textvariable=self.train_pix_threshold, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(feature_frame, text="Brightness cutoff: <1 = fraction, ≥1 = raw value", foreground='gray').grid(row=2, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="DLC Config (optional):").grid(row=3, column=0, sticky='w', pady=2)
        self.train_dlc_config = tk.StringVar()
        ttk.Entry(feature_frame, textvariable=self.train_dlc_config, width=30).grid(
            row=3, column=1, padx=5, pady=2, sticky='w')
        ttk.Button(feature_frame, text="📁 Browse",
                  command=self.browse_train_dlc_config).grid(row=3, column=2, sticky='w', padx=2)
        ttk.Label(feature_frame, text="For DLC crop offset in brightness extraction",
                 foreground='gray').grid(row=4, column=1, columnspan=2, sticky='w')

        # Optical Flow Features
        self.train_include_optical_flow = tk.BooleanVar(value=True)
        ttk.Checkbutton(feature_frame,
                        text="Include Optical Flow Features  (slower — reads video frames)",
                        variable=self.train_include_optical_flow).grid(
            row=5, column=1, columnspan=2, sticky='w', pady=2)
        ttk.Label(feature_frame, text="Optical Flow Body Parts:").grid(row=6, column=0, sticky='w', pady=2)
        self.train_bp_optflow = tk.StringVar(value="hrpaw,hlpaw,snout")
        ttk.Entry(feature_frame, textvariable=self.train_bp_optflow, width=30).grid(
            row=6, column=1, padx=5, pady=2, sticky='w')
        ttk.Label(feature_frame, text="Body parts for optical flow (comma-separated)",
                 foreground='gray').grid(row=6, column=2, sticky='w')

        # Runtime warning banner — empty by default, populated by trace callbacks
        self._train_warning_lbl = tk.Label(
            scrollable_frame, text="",
            foreground='#b8860b', font=('Arial', 9, 'bold'),
            anchor='w')
        self._train_warning_lbl.pack(fill='x', padx=18, pady=(0, 2))

        # === ADVANCED SETTINGS (collapsible) ===
        self._advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Show Advanced Settings",
                        variable=self._advanced_visible,
                        command=self._toggle_advanced).pack(anchor='w', padx=10, pady=(8, 2))

        # === XGBOOST PARAMETERS ===
        xgb_frame = ttk.LabelFrame(scrollable_frame, text="XGBoost Model Parameters", padding=10)
        self._xgb_frame = xgb_frame
        
        ttk.Label(xgb_frame, text="Number of Trees:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_n_estimators = tk.IntVar(value=1700)
        ttk.Spinbox(xgb_frame, from_=100, to=5000, increment=100,
                   textvariable=self.train_n_estimators, width=10).grid(
            row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(xgb_frame, text="More trees = better fit but slower (1000-2000 typical)", foreground='gray').grid(row=0, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Max Tree Depth:").grid(row=1, column=0, sticky='w', pady=2)
        self.train_max_depth = tk.IntVar(value=6)
        ttk.Spinbox(xgb_frame, from_=3, to=15, textvariable=self.train_max_depth, width=10).grid(
            row=1, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(xgb_frame, text="Tree complexity: 4-8 typical, higher = risk overfitting", foreground='gray').grid(row=1, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Learning Rate:").grid(row=2, column=0, sticky='w', pady=2)
        self.train_learning_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(xgb_frame, textvariable=self.train_learning_rate, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(xgb_frame, text="Step size for updates: 0.01-0.1 typical, lower = more stable", foreground='gray').grid(row=2, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Subsample Ratio:").grid(row=3, column=0, sticky='w', pady=2)
        self.train_subsample = tk.DoubleVar(value=0.8)
        ttk.Entry(xgb_frame, textvariable=self.train_subsample, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(xgb_frame, text="Fraction of data per tree: 0.5-0.9, prevents overfitting", foreground='gray').grid(row=3, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Feature Sampling:").grid(row=4, column=0, sticky='w', pady=2)
        self.train_colsample = tk.DoubleVar(value=0.2)
        ttk.Entry(xgb_frame, textvariable=self.train_colsample, width=10).grid(
            row=4, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(xgb_frame, text="Fraction of features per tree: 0.2-0.5, adds diversity", foreground='gray').grid(row=4, column=2, sticky='w')
        
        # === TRAINING PARAMETERS ===
        params_frame = ttk.LabelFrame(scrollable_frame, text="Training Parameters", padding=10)
        self._params_frame = params_frame
        
        ttk.Label(params_frame, text="K-Fold Cross-Validation:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_n_folds = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=2, to=10, textvariable=self.train_n_folds, width=10).grid(
            row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(params_frame, text="Number of validation folds: 5 typical, 10 for smaller datasets", foreground='gray').grid(row=0, column=2, sticky='w')
        
        # scale_pos_weight (replaces downsampling as the default)
        self.train_use_scale_pos_weight = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use scale_pos_weight for class imbalance",
                       variable=self.train_use_scale_pos_weight).grid(row=1, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Recommended: weights positives without discarding any frames", foreground='gray').grid(row=1, column=2, sticky='w')

        # Fallback downsampling (kept for compatibility)
        self.train_use_balancing = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Also apply downsampling (legacy fallback)",
                       variable=self.train_use_balancing).grid(row=2, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Downsamples negatives — use only if scale_pos_weight is off", foreground='gray').grid(row=2, column=2, sticky='w')

        ttk.Label(params_frame, text="Imbalance Threshold:").grid(row=3, column=0, sticky='w', pady=2)
        self.train_imbalance_thresh = tk.DoubleVar(value=0.05)
        ttk.Entry(params_frame, textvariable=self.train_imbalance_thresh, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(params_frame, text="Apply downsampling only if positive ratio < this value", foreground='gray').grid(row=3, column=2, sticky='w')

        # Early stopping
        self.train_use_early_stopping = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use early stopping (auto-selects n_estimators)",
                       variable=self.train_use_early_stopping).grid(row=4, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Stops adding trees when val F1 plateaus — prevents overfitting", foreground='gray').grid(row=4, column=2, sticky='w')

        ttk.Label(params_frame, text="Early Stopping Rounds:").grid(row=5, column=0, sticky='w', pady=2)
        self.train_early_stopping_rounds = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=10, to=200, increment=10,
                   textvariable=self.train_early_stopping_rounds, width=10).grid(
            row=5, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(params_frame, text="Stop if no improvement for this many trees (50 typical)", foreground='gray').grid(row=5, column=2, sticky='w')

        # Generate plots
        self.train_generate_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Generate performance and SHAP plots",
                       variable=self.train_generate_plots).grid(row=7, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Creates threshold curve and feature importance figures", foreground='gray').grid(row=7, column=2, sticky='w')

        # Gain-importance pruning (2nd-pass retrain on top-N features)
        self.train_prune_by_gain = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Prune by gain + retrain (2nd pass with top features only)",
                       variable=self.train_prune_by_gain).grid(row=8, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Train on all features first, then retrain on top features by XGBoost gain importance — reduces noise", foreground='gray').grid(row=8, column=2, sticky='w')

        ttk.Label(params_frame, text="Top Features (by gain):").grid(row=9, column=0, sticky='w', pady=2)
        self.train_prune_top_n = tk.IntVar(value=120)
        ttk.Spinbox(params_frame, from_=10, to=200, increment=5,
                   textvariable=self.train_prune_top_n, width=10).grid(
            row=9, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(params_frame, text="Number of features to keep after pruning (10–200)", foreground='gray').grid(row=9, column=2, sticky='w')

        # Trim to last positive
        self.train_trim_to_last_positive = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Trim sessions to last labeled event",
                        variable=self.train_trim_to_last_positive).grid(row=10, column=1, sticky='w', pady=2)
        ttk.Label(params_frame,
                 text="Remove frames after the last '1' in each label file "
                      "(prevents BORIS trailing zeros from flooding training)",
                 foreground='gray').grid(row=10, column=2, sticky='w')

        # Correlation pre-filter
        self.train_correlation_filter = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Drop highly correlated features (|r| > 0.95)",
                       variable=self.train_correlation_filter).grid(row=11, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Removes near-duplicate features before training — reduces noise, never alters cached files",
                 foreground='gray').grid(row=11, column=2, sticky='w')

        # ── Advanced ML options (Optuna, lag features, egocentric) ──
        ttk.Separator(params_frame, orient='horizontal').grid(
            row=12, column=0, columnspan=3, sticky='ew', pady=(8, 4))

        # Optuna auto-tuning
        self.train_use_optuna = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Optuna auto-tune hyperparameters",
                       variable=self.train_use_optuna).grid(row=13, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Searches max_depth, learning_rate, colsample_bytree, subsample (slower training)",
                 foreground='gray').grid(row=13, column=2, sticky='w')

        ttk.Label(params_frame, text="Optuna Trials:").grid(row=14, column=0, sticky='w', pady=2)
        self.train_optuna_trials = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=10, to=100, increment=5,
                   textvariable=self.train_optuna_trials, width=10).grid(
            row=14, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(params_frame, text="More trials = better search but slower (25 typical)",
                 foreground='gray').grid(row=14, column=2, sticky='w')

        # Lag/lead features
        self.train_use_lag_features = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Include lag/lead features (temporal context)",
                       variable=self.train_use_lag_features).grid(row=15, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Adds +/-1,2 frame shifts of top features — helps detect onset/offset",
                 foreground='gray').grid(row=15, column=2, sticky='w')

        # Egocentric normalization
        self.train_use_egocentric = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Include egocentric (centroid-relative) features",
                       variable=self.train_use_egocentric).grid(row=16, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Position-invariant distances and velocities — helps if animal moves around",
                 foreground='gray').grid(row=16, column=2, sticky='w')

        # Contact state features
        self.train_use_contact_features = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Include contact state features",
                       variable=self.train_use_contact_features).grid(row=17, column=1, sticky='w', pady=2)
        contact_sub = ttk.Frame(params_frame)
        contact_sub.grid(row=17, column=2, sticky='w')
        ttk.Label(contact_sub, text="Threshold (px):", foreground='gray').pack(side='left')
        self.train_contact_threshold = tk.DoubleVar(value=15.0)
        ttk.Spinbox(contact_sub, from_=1, to=100, increment=1,
                   textvariable=self.train_contact_threshold, width=6).pack(side='left', padx=4)
        ttk.Label(contact_sub, text="Binary paw contact, transitions, duty cycle",
                 foreground='gray').pack(side='left', padx=4)

        # Probability calibration (isotonic fit on OOF)
        self.train_use_calibration = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Calibrate probabilities (isotonic on OOF)",
                       variable=self.train_use_calibration).grid(row=18, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Fits IsotonicRegression on OOF probabilities — improves P=0.5 thresholding and AL uncertainty",
                 foreground='gray').grid(row=18, column=2, sticky='w')

        # CV fold-ensemble at inference
        self.train_use_fold_ensemble = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Save fold ensemble (average K CV models at predict time)",
                       variable=self.train_use_fold_ensemble).grid(row=19, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Stores all K fold models; inference averages them with the final model (larger pkl, lower variance)",
                 foreground='gray').grid(row=19, column=2, sticky='w')

        # Learning curve diagnostic
        self.train_learning_curve = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Show learning curve (~3x slower)",
                       variable=self.train_learning_curve).grid(row=20, column=1, sticky='w', pady=2)
        ttk.Label(params_frame, text="Trains on 25/50/75/100% of data to show if more labeling would help",
                 foreground='gray').grid(row=20, column=2, sticky='w')

        # Post-processing optimization strategy
        ttk.Separator(params_frame, orient='horizontal').grid(
            row=21, column=0, columnspan=3, sticky='ew', pady=(8, 4))
        ttk.Label(params_frame, text="Post-processing\noptimization:").grid(row=22, column=0, sticky='w', pady=2)
        self.train_opt_strategy = tk.StringVar(value='auto')
        opt_radio_frame = ttk.Frame(params_frame)
        opt_radio_frame.grid(row=22, column=1, columnspan=2, sticky='w')
        ttk.Radiobutton(opt_radio_frame, text="Auto (LOVO if ≥2 sessions, else OOF)",
                        variable=self.train_opt_strategy, value='auto').pack(anchor='w')
        ttk.Radiobutton(opt_radio_frame, text="OOF only  (faster, no leave-one-video-out)",
                        variable=self.train_opt_strategy, value='oof').pack(anchor='w')
        ttk.Radiobutton(opt_radio_frame, text="LOVO / in-session  (more conservative)",
                        variable=self.train_opt_strategy, value='lovo').pack(anchor='w')

        # Runtime-warning banner trace — fires when Optuna + learning curve are
        # both enabled (the only combination that breaches ~30 min runtime).
        for _v in (self.train_use_optuna, self.train_learning_curve):
            _v.trace_add('write', self._update_training_warning)
        self._update_training_warning()

        # === SESSION SELECTION ===
        # Collapsible header
        self._sess_toggle_btn = ttk.Button(
            scrollable_frame, text="▶ Session Selection",
            command=self._toggle_session_panel)
        self._sess_toggle_btn.pack(fill='x', padx=15, pady=(8, 0))

        # Content (collapsed by default) — outer frame always packed (holds position in stack)
        session_frame = ttk.Frame(scrollable_frame, relief='groove', borderwidth=1)
        self._sess_content_frame = session_frame
        session_frame.pack(fill='x', padx=15, pady=(0, 8))

        # Inner frame holds all visible content; toggled show/hide
        inner = ttk.Frame(session_frame)
        self._sess_inner_frame = inner
        # NOT packed yet — starts collapsed

        sess_btn_frame = ttk.Frame(inner)
        sess_btn_frame.pack(fill='x', pady=(0, 4))

        ttk.Button(sess_btn_frame, text="📋 Scan Sessions",
                   command=self._scan_and_populate_sessions, width=18).pack(side='left', padx=5)
        ttk.Button(sess_btn_frame, text="Select All",
                   command=self._session_select_all, width=12).pack(side='left', padx=5)
        ttk.Button(sess_btn_frame, text="Select Labeled",
                   command=self._session_select_labeled, width=14).pack(side='left', padx=5)
        ttk.Button(sess_btn_frame, text="Deselect All",
                   command=self._session_deselect_all, width=12).pack(side='left', padx=5)

        tree_frame = ttk.Frame(inner)
        tree_frame.pack(fill='x', padx=5, pady=2)

        self._session_tree = ttk.Treeview(
            tree_frame,
            columns=("check", "session", "labels", "video"),
            show="headings",
            selectmode="none",
            height=6,
        )
        self._session_tree.heading("check", text="✓")
        self._session_tree.heading("session", text="Session Name")
        self._session_tree.heading("labels", text="Labels")
        self._session_tree.heading("video", text="Video")
        self._session_tree.column("check", width=30, anchor="center", stretch=False)
        self._session_tree.column("session", width=250, anchor="w")
        self._session_tree.column("labels", width=80, anchor="center")
        self._session_tree.column("video", width=250, anchor="w")
        self._session_tree.tag_configure('no_labels', foreground='gray')
        tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self._session_tree.yview)
        self._session_tree.configure(yscrollcommand=tree_scroll.set)
        self._session_tree.pack(side='left', fill='x', expand=True)
        tree_scroll.pack(side='right', fill='y')
        self._session_tree.bind("<ButtonRelease-1>", self._on_session_tree_click)

        self._session_count_label = ttk.Label(inner, text="", foreground='gray')
        self._session_count_label.pack(anchor='w', padx=5, pady=(0, 2))

        # === QUICK ACTIONS ===
        action_frame = ttk.LabelFrame(scrollable_frame, text="Actions", padding=10)
        action_frame.pack(fill='x', padx=15, pady=(12, 8))

        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill='x')

        ttk.Button(btn_frame, text="🔍 Check Data Quality",
                  command=self.open_quality_checker, width=20).pack(side='left', padx=5, pady=5)
        ttk.Button(btn_frame, text="💾 Save Configuration",
                  command=self.save_training_config, width=20).pack(side='left', padx=5, pady=5)
        ttk.Button(btn_frame, text="📂 Load Configuration",
                  command=self.load_training_config, width=20).pack(side='left', padx=5, pady=5)

        # Start button (larger, prominent)
        start_frame = ttk.Frame(action_frame)
        start_frame.pack(fill='x', pady=10)
        self._train_start_btn = ttk.Button(start_frame, text="▶ START TRAINING",
                  command=self.start_training,
                  style='Accent.TButton')
        self._train_start_btn.pack(side='left', padx=5)
        self._train_all_btn = ttk.Button(start_frame, text="▶▶ Train All Behaviors",
                  command=self.start_training_all_behaviors)
        self._train_all_btn.pack(side='left', padx=5)
        self._train_cancel_btn = ttk.Button(start_frame, text="■ Cancel Training",
                  command=self._cancel_training, state='disabled')
        self._train_cancel_btn.pack(side='left', padx=5)

        # Training progress bar
        self._train_progress = ttk.Progressbar(action_frame, length=400, mode='determinate')
        self._train_progress.pack(fill='x', padx=5, pady=(5, 0))
        self._train_progress_label = ttk.Label(action_frame, text="")
        self._train_progress_label.pack(anchor='w', padx=5)

        # === TRAINING LOG ===
        log_frame = ttk.LabelFrame(scrollable_frame, text="Training Log", padding=5)
        log_frame.pack(fill='both', expand=True, padx=15, pady=8)
        
        self.train_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.train_log.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel(canvas)

    def create_evaluation_tab(self):
        """Create the evaluation tab — delegates to evaluation_tab.EvaluationTab."""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="📊 Evaluate")

        if EVALUATION_TAB_AVAILABLE:
            self.evaluation_tab = EvaluationTab(eval_frame, self)
            self.evaluation_tab.pack(fill='both', expand=True)
            # Keep backward-compatible references that other code might use
            self.eval_classifier_path = self.evaluation_tab.eval_classifier_path
            self.eval_test_folder     = self.evaluation_tab.eval_test_folder
            self.eval_info_text       = self.evaluation_tab.eval_info_text
            self.eval_results_text    = self.evaluation_tab.eval_results_text
        else:
            ttk.Label(eval_frame,
                      text="⚠️  evaluation_tab.py not found.\n\n"
                           "Place evaluation_tab.py in the same folder as PixelPaws_GUI.py.",
                      font=('Arial', 11), foreground='red',
                      justify='center').pack(expand=True)
    
    def create_prediction_tab(self):
        """Create the single video prediction tab"""
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="🎬 Predict")
        
        # Create scrollable canvas
        canvas = tk.Canvas(pred_frame)
        scrollbar = ttk.Scrollbar(pred_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === CLASSIFIER SELECTION ===
        clf_frame = ttk.LabelFrame(scrollable_frame, text="Classifier", padding=10)
        clf_frame.pack(fill='x', padx=15, pady=8)
        
        ttk.Label(clf_frame, text="Classifier File:").grid(row=0, column=0, sticky='w', pady=2)
        self.pred_classifier_path = tk.StringVar()
        self.pred_classifier_combo = ttk.Combobox(
            clf_frame, textvariable=self.pred_classifier_path, width=46, state='readonly')
        self.pred_classifier_combo.grid(row=0, column=1, padx=5, pady=2)
        self.pred_classifier_combo.bind('<<ComboboxSelected>>', self._on_pred_classifier_selected)
        ttk.Button(clf_frame, text="🔄", width=3,
                   command=self.refresh_pred_classifiers).grid(row=0, column=2, pady=2)
        ttk.Button(clf_frame, text="📁", width=3,
                   command=self.browse_pred_classifier).grid(row=0, column=3, pady=2)
        
        ttk.Button(clf_frame, text="📋 View Classifier Info",
                  command=self.view_pred_classifier_info).grid(row=1, column=1, sticky='w', pady=5)

        # Smoothing mode selector
        self.pred_smoothing_mode = tk.StringVar(value='bout_filters')
        smooth_row = ttk.Frame(clf_frame)
        smooth_row.grid(row=2, column=0, columnspan=4, sticky='w', pady=(4, 2))
        ttk.Label(smooth_row, text="Smoothing:").pack(side='left')
        for _lbl, _val in [("Bout filters", "bout_filters"),
                            ("HMM Viterbi", "hmm_viterbi"),
                            ("None", "none")]:
            ttk.Radiobutton(smooth_row, text=_lbl,
                            variable=self.pred_smoothing_mode,
                            value=_val).pack(side='left', padx=(8, 0))
        ttk.Label(smooth_row,
                  text="  — Bout filters: min_bout/gap rules  ·  "
                       "HMM Viterbi: probabilistic sequence decoding  ·  "
                       "None: raw threshold only",
                  foreground='gray', font=('Arial', 8)).pack(side='left', padx=(6, 0))

        # === VIDEO SELECTION ===
        video_frame = ttk.LabelFrame(scrollable_frame, text="Video Files", padding=10)
        video_frame.pack(fill='x', padx=15, pady=8)
        
        ttk.Label(video_frame, text="Video File:").grid(row=0, column=0, sticky='w', pady=2)
        self.pred_video_path = tk.StringVar()
        self.pred_video_options = {}
        self.pred_video_combo = ttk.Combobox(
            video_frame, textvariable=self.pred_video_path, width=46)
        self.pred_video_combo.grid(row=0, column=1, padx=5, pady=2)
        self.pred_video_combo.bind('<<ComboboxSelected>>', self._on_pred_video_selected)
        ttk.Button(video_frame, text="🔄", width=3,
                   command=self.refresh_pred_videos).grid(row=0, column=2, pady=2)
        ttk.Button(video_frame, text="📁", width=3,
                   command=self.browse_pred_video).grid(row=0, column=3, pady=2)
        
        ttk.Label(video_frame, text="DLC Pose File:").grid(row=1, column=0, sticky='w', pady=2)
        self.pred_dlc_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.pred_dlc_path, width=50).grid(
            row=1, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_pred_dlc).grid(row=1, column=2, pady=2)
        
        ttk.Button(video_frame, text="🔍 Auto-Find DLC File", 
                  command=self.auto_find_dlc).grid(row=2, column=1, sticky='w', pady=5)
        
        # Features file (optional)
        ttk.Label(video_frame, text="Features File (optional):").grid(row=3, column=0, sticky='w', pady=2)
        self.pred_features_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.pred_features_path, width=50).grid(
            row=3, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_pred_features).grid(row=3, column=2, pady=2)
        
        ttk.Label(video_frame, text="Skip feature extraction if file provided", 
                 font=('Arial', 8), foreground='gray').grid(row=3, column=1, sticky='w', padx=5)
        
        # DLC Config for crop parameters
        ttk.Label(video_frame, text="DLC Config (for crop):").grid(row=4, column=0, sticky='w', pady=2)
        self.pred_dlc_config_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.pred_dlc_config_path, width=50).grid(
            row=4, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_pred_dlc_config).grid(row=4, column=2, pady=2)
        
        ttk.Button(video_frame, text="🔍 Auto-Find Config", 
                  command=self.auto_find_dlc_config).grid(row=5, column=1, sticky='w', pady=5)
        
        # Human labels (optional)
        ttk.Label(video_frame, text="Human Labels (optional):").grid(row=6, column=0, sticky='w', pady=2)
        self.pred_human_labels_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.pred_human_labels_path, width=50).grid(
            row=6, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_pred_human_labels).grid(row=6, column=2, pady=2)
        
        # === OUTPUT OPTIONS ===
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Options", padding=10)
        output_frame.pack(fill='x', padx=15, pady=8)

        self.pred_save_csv = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Save frame-by-frame predictions (CSV)", 
                       variable=self.pred_save_csv).grid(row=0, column=0, sticky='w', pady=2)
        
        self.pred_save_video = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Create labeled video (slower)",
                       variable=self.pred_save_video).grid(row=1, column=0, sticky='w', pady=2)

        self.pred_clip_start = tk.StringVar(value="")
        self.pred_clip_end   = tk.StringVar(value="")
        clip_row = ttk.Frame(output_frame)
        clip_row.grid(row=2, column=0, columnspan=3, sticky='w', padx=(25, 0), pady=1)
        ttk.Label(clip_row, text="Clip:  From").pack(side='left')
        ttk.Entry(clip_row, textvariable=self.pred_clip_start, width=8).pack(side='left', padx=3)
        ttk.Label(clip_row, text="To").pack(side='left', padx=(4, 0))
        ttk.Entry(clip_row, textvariable=self.pred_clip_end, width=8).pack(side='left', padx=3)
        ttk.Label(clip_row, text="(frame number, seconds as 1.5, or H:MM:SS — blank = full video)",
                  foreground='gray').pack(side='left', padx=5)

        self.pred_save_summary = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Save behavior summary statistics",
                       variable=self.pred_save_summary).grid(row=3, column=0, sticky='w', pady=2)

        self.pred_generate_ethogram = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Generate ethogram plots",
                       variable=self.pred_generate_ethogram).grid(row=4, column=0, sticky='w', pady=2)

        ttk.Label(output_frame, text="Output Folder:").grid(row=5, column=0, sticky='w', pady=5)
        self.pred_output_folder = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.pred_output_folder, width=50).grid(
            row=5, column=1, padx=5, pady=2)
        ttk.Button(output_frame, text="📁 Browse",
                  command=self.browse_pred_output).grid(row=5, column=2, pady=2)
        ttk.Label(output_frame, text="(Leave empty to use project results/ folder, or video folder if no project is set)", foreground='gray').grid(row=6, column=1, sticky='w')
        
        # === ACTIONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(action_frame, text="🎥 Preview Video", 
                  command=self.preview_pred_video).pack(side='left', padx=5)
        ttk.Button(action_frame, text="🔍 Preview with Predictions", 
                  command=self.preview_with_predictions).pack(side='left', padx=5)
        ttk.Button(action_frame, text="🎯 Optimize Parameters", 
                  command=self.optimize_parameters).pack(side='left', padx=5)
        self._pred_run_btn = ttk.Button(action_frame, text="▶ RUN PREDICTION",
                  command=self.run_single_prediction,
                  style='Accent.TButton')
        self._pred_run_btn.pack(side='left', padx=5)
        self._pred_stop_btn = ttk.Button(action_frame, text="■  Stop",
                  command=self._cancel_prediction,
                  state='disabled')
        self._pred_stop_btn.pack(side='left', padx=5)
        self.pred_export_video_btn = ttk.Button(
            action_frame, text="🎬 Export Labeled Video",
            command=self.export_labeled_video, state='disabled')
        self.pred_export_video_btn.pack(side='left', padx=5)

        # === EXPORT VIDEO OVERLAYS ===
        overlay_opts = ttk.LabelFrame(scrollable_frame, text="Export Video Overlays", padding=4)
        overlay_opts.pack(fill='x', padx=5, pady=(0, 4))

        # Row 0 — checkboxes
        cb_row = ttk.Frame(overlay_opts)
        cb_row.grid(row=0, column=0, columnspan=4, sticky='w', pady=(2, 4))
        ttk.Checkbutton(cb_row, text="Skeleton dots",
                        variable=self.lv_skeleton_dots).pack(side='left', padx=6)
        ttk.Checkbutton(cb_row, text="Frame tint",
                        variable=self.lv_frame_tint,
                        command=self._on_tint_toggle).pack(side='left', padx=6)
        ttk.Checkbutton(cb_row, text="Timeline strip",
                        variable=self.lv_timeline_strip,
                        command=self._refresh_overlay_preview).pack(side='left', padx=6)
        ttk.Checkbutton(cb_row, text="Halo border",
                        variable=self.lv_halo_border,
                        command=self._refresh_overlay_preview).pack(side='left', padx=6)
        ttk.Checkbutton(cb_row, text="Bout counter in HUD",
                        variable=self.lv_bout_counter,
                        command=self._refresh_overlay_preview).pack(side='left', padx=6)

        # Row 1 — scheme preset + color pickers + tint opacity
        color_row = ttk.Frame(overlay_opts)
        color_row.grid(row=1, column=0, columnspan=4, sticky='w', pady=2)
        ttk.Label(color_row, text="Scheme:").pack(side='left', padx=(6, 2))
        self._lv_scheme_combo = ttk.Combobox(
            color_row,
            textvariable=self.lv_color_scheme,
            values=list(OVERLAY_COLOR_SCHEMES.keys()) + ['Custom'],
            state='readonly', width=18)
        self._lv_scheme_combo.pack(side='left', padx=(0, 12))
        self._lv_scheme_combo.bind('<<ComboboxSelected>>', lambda _: self._on_color_scheme_changed())
        ttk.Label(color_row, text="Behavior color:").pack(side='left', padx=(6, 2))
        self._lv_beh_color_btn = tk.Button(
            color_row, text="  ", width=3,
            bg=self.lv_behavior_color.get(), relief='raised',
            command=lambda: self._pick_overlay_color(self.lv_behavior_color,
                                                     self._lv_beh_color_btn))
        self._lv_beh_color_btn.pack(side='left', padx=(0, 8))

        ttk.Label(color_row, text="No-behavior color:").pack(side='left', padx=(0, 2))
        self._lv_nobeh_color_btn = tk.Button(
            color_row, text="  ", width=3,
            bg=self.lv_nobehavior_color.get(), relief='raised',
            command=lambda: self._pick_overlay_color(self.lv_nobehavior_color,
                                                     self._lv_nobeh_color_btn))
        self._lv_nobeh_color_btn.pack(side='left', padx=(0, 12))

        ttk.Label(color_row, text="Tint opacity:").pack(side='left', padx=(0, 2))
        self._lv_tint_scale = ttk.Scale(
            color_row, from_=0.05, to=0.50, length=90,
            variable=self.lv_tint_opacity,
            command=lambda _v: self._refresh_overlay_preview())
        self._lv_tint_scale.pack(side='left')
        self._on_tint_toggle()   # set initial enabled/disabled state

        # Row 2 — HUD position
        hud_row = ttk.Frame(overlay_opts)
        hud_row.grid(row=2, column=0, columnspan=4, sticky='w', pady=2)
        ttk.Label(hud_row, text="HUD position:").pack(side='left', padx=(6, 4))
        for _val, _lbl in (('top', 'Top'), ('bottom', 'Bottom')):
            ttk.Radiobutton(hud_row, text=_lbl, variable=self.lv_hud_position,
                            value=_val,
                            command=self._refresh_overlay_preview).pack(side='left', padx=3)

        # Row 3 — in-tab preview
        prev_row = ttk.Frame(overlay_opts)
        prev_row.grid(row=3, column=0, columnspan=4, sticky='w', pady=(4, 2))
        self.lv_preview_canvas = tk.Canvas(prev_row, width=320, height=200, bg='black',
                                           highlightthickness=1, highlightbackground='#555')
        self.lv_preview_canvas.pack(side='left', padx=(6, 4))
        _prev_ctrl = ttk.Frame(prev_row)
        _prev_ctrl.pack(side='left', anchor='n', padx=4)
        ttk.Button(_prev_ctrl, text="↺ Refresh Preview",
                   command=self._refresh_overlay_preview).pack(anchor='w')
        ttk.Label(_prev_ctrl, text="(behavior-active overlay preview)",
                  foreground='gray').pack(anchor='w', pady=(4, 0))

        # Auto-refresh preview when video path changes
        self.pred_video_path.trace_add('write', lambda *_: self._refresh_overlay_preview())

        # === RESULTS DISPLAY ===
        results_frame = ttk.LabelFrame(scrollable_frame, text="Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.pred_results_text = scrolledtext.ScrolledText(results_frame, height=12, wrap=tk.WORD)
        self.pred_results_text.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel(canvas)

    def create_batch_tab(self):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="📦 Batch")
        
        # Create scrollable canvas
        canvas = tk.Canvas(batch_frame)
        scrollbar = ttk.Scrollbar(batch_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === INPUT FOLDER ===
        input_frame = ttk.LabelFrame(scrollable_frame, text="Input Files", padding=10)
        input_frame.pack(fill='x', padx=15, pady=8)
        
        ttk.Label(input_frame, text="Data Folder:").grid(row=0, column=0, sticky='w', pady=2)
        self.batch_folder = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.batch_folder, width=50).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(input_frame, text="📁 Browse", 
                  command=self.browse_batch_folder).grid(row=0, column=2, pady=2)
        
        ttk.Label(input_frame, text="Video Extension:").grid(row=1, column=0, sticky='w', pady=2)
        self.batch_video_ext = ttk.Combobox(input_frame, values=['.mp4', '.avi'], width=10)
        self.batch_video_ext.set('.mp4')
        self.batch_video_ext.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        self.batch_prefer_filtered = tk.BooleanVar(value=True)
        ttk.Checkbutton(input_frame, text="Prefer 'filtered' DLC files when available", 
                       variable=self.batch_prefer_filtered).grid(row=2, column=1, sticky='w', pady=2)
        
        ttk.Label(input_frame, text="DLC Config (optional):").grid(row=3, column=0, sticky='w', pady=2)
        self.batch_dlc_config = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.batch_dlc_config, width=50).grid(
            row=3, column=1, padx=5, pady=2)
        ttk.Button(input_frame, text="📁 Browse", 
                  command=self.browse_batch_dlc_config).grid(row=3, column=2, pady=2)
        
        ttk.Label(input_frame, text="For DLC crop offset in brightness extraction", 
                 font=('Arial', 8), foreground='gray').grid(row=4, column=1, sticky='w', padx=5)
        
        # === CLASSIFIERS ===
        clf_frame = ttk.LabelFrame(scrollable_frame, text="Classifiers", padding=10)
        clf_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(clf_frame)
        list_frame.pack(fill='both', expand=True)
        
        self.batch_clf_listbox = tk.Listbox(list_frame, height=8)
        self.batch_clf_listbox.pack(side='left', fill='both', expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', 
                                      command=self.batch_clf_listbox.yview)
        list_scrollbar.pack(side='right', fill='y')
        self.batch_clf_listbox.config(yscrollcommand=list_scrollbar.set)
        
        # Classifier buttons
        btn_frame = ttk.Frame(clf_frame)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame, text="➕ Add Classifier", 
                  command=self.batch_add_classifier).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="➖ Remove Selected", 
                  command=self.batch_remove_classifier).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="⚙ Edit Settings", 
                  command=self.batch_edit_classifier).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="👁 Preview Mapping",
                  command=self.batch_preview_mapping).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="🔄 Auto-Detect All",
                   command=self.batch_autodetect_classifiers).pack(side='left', padx=2)
        
        # Batch classifier storage
        self.batch_classifiers = {}  # {path: {min_bout_sec, bin_size_sec}}
        
        # === OUTPUT OPTIONS ===
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Options", padding=10)
        output_frame.pack(fill='x', padx=15, pady=8)

        self.batch_save_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Save frame-by-frame labels for each video",
                       variable=self.batch_save_labels).grid(row=0, column=0, sticky='w', pady=2)
        
        # === ACTIONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(action_frame, text="🔍 Check Feature Status", 
                  command=self.check_batch_features, 
                  ).pack(side='left', padx=5)
        
        self._batch_run_btn = ttk.Button(action_frame, text="▶ RUN BATCH ANALYSIS",
                  command=self.run_batch_analysis,
                  style='Accent.TButton')
        self._batch_run_btn.pack(side='left', padx=5)
        self._batch_stop_btn = ttk.Button(action_frame, text="■  Stop",
                  command=self._cancel_batch_analysis,
                  state='disabled')
        self._batch_stop_btn.pack(side='left', padx=5)
        
        # === PROGRESS ===
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Progress", padding=5)
        progress_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.batch_progress_label = ttk.Label(progress_frame, text="Ready to process")
        self.batch_progress_label.pack(pady=5)
        
        self.batch_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.batch_progress.pack(pady=5)
        
        self.batch_log = scrolledtext.ScrolledText(progress_frame, height=8, wrap=tk.WORD)
        self.batch_log.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel(canvas)

    def create_tools_tab(self):
        """Create tools tab with grouped tool sections"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="🛠 Tools")

        # Scrollable area for tools
        tools_canvas = tk.Canvas(tools_frame)
        tools_sb = ttk.Scrollbar(tools_frame, orient='vertical', command=tools_canvas.yview)
        tools_sf = ttk.Frame(tools_canvas)
        tools_sf.bind('<Configure>',
                      lambda e: tools_canvas.configure(scrollregion=tools_canvas.bbox('all')))
        tools_canvas.create_window((0, 0), window=tools_sf, anchor='nw')
        tools_canvas.configure(yscrollcommand=tools_sb.set)

        # Title
        ttk.Label(tools_sf, text="Tools",
                 font=('Arial', 14, 'bold')).pack(pady=10)

        def _add_section(parent, title, tools_list):
            lf = ttk.LabelFrame(parent, text=title, padding=10)
            lf.pack(fill='x', padx=15, pady=8)
            gf = ttk.Frame(lf)
            gf.pack(fill='x')
            for i, (text, cmd) in enumerate(tools_list):
                r, c = divmod(i, 2)
                btn = ttk.Button(gf, text=text, command=cmd, width=24)
                btn.grid(row=r, column=c, padx=8, pady=6, sticky='ew')
            gf.columnconfigure(0, weight=1)
            gf.columnconfigure(1, weight=1)

        _add_section(tools_sf, "Video Tools", [
            ("🎥 Video Preview with Predictions", self.open_video_preview),
            ("🦴 Skeleton Video Renderer", self.open_skeleton_renderer),
            ("✂️ Crop Video for DLC", self.crop_video_for_dlc),
            ("🌟 Brightness Preview", self.show_brightness_preview),
        ])

        _add_section(tools_sf, "Analysis Tools", [
            ("🤖 Auto-Label Assistant", self.open_auto_labeler),
            ("🔍 Data Quality Checker", self.open_quality_checker),
            ("💡 Brightness Diagnostics", self.run_brightness_diagnostics),
            ("📋 Feature File Inspector", self.inspect_features_file),
            ("🎯 Optimize Parameters", self.optimize_parameters),
            ("📈 Training Visualization", self.show_training_viz),
            ("🔄 BORIS to PixelPaws", self.convert_boris_to_pixelpaws),
        ])

        _add_section(tools_sf, "Configuration", [
            ("🔧 Correct Crop Offset (Single)", self.correct_crop_offset_single),
            ("🔧 Correct Crop Offset (Batch)", self.correct_crop_offset_batch),
            ("⚙️ Feature Extraction", self.open_feature_extraction),
        ])

        tools_canvas.pack(side='left', fill='both', expand=True)
        tools_sb.pack(side='right', fill='y')
        bind_mousewheel(tools_canvas)

        # ── Classifier Library ──
        clf_lib_frame = ttk.LabelFrame(tools_sf, text="Classifier Library",
                                        padding=10)
        clf_lib_frame.pack(fill='x', padx=15, pady=8)

        ttk.Label(clf_lib_frame,
                  text="Share trained classifiers across projects via a global folder.",
                  font=('Arial', 9)).pack(anchor='w', pady=(0, 8))

        clf_btn_frame = ttk.Frame(clf_lib_frame)
        clf_btn_frame.pack(fill='x')
        clf_btn_frame.columnconfigure(0, weight=1)
        clf_btn_frame.columnconfigure(1, weight=1)

        ttk.Button(clf_btn_frame, text="📤 Export to Global",
                   command=self._export_classifier_to_global).grid(
                       row=0, column=0, padx=8, pady=6, sticky='ew')
        ttk.Button(clf_btn_frame, text="📥 Import to Project",
                   command=self._import_classifier_from_global).grid(
                       row=0, column=1, padx=8, pady=6, sticky='ew')
        ttk.Button(clf_btn_frame, text="📂 Open Global Folder",
                   command=self._open_global_clf_folder).grid(
                       row=1, column=0, padx=8, pady=6, sticky='ew')
        ttk.Button(clf_btn_frame, text="⚙️ Change Global Folder",
                   command=self._change_global_clf_folder).grid(
                       row=1, column=1, padx=8, pady=6, sticky='ew')
    
    
    # ── Classifier Library methods ──────────────────────────────────────

    def _export_classifier_to_global(self):
        """Copy a .pkl from current project (or any path) to the global library."""
        import shutil
        from user_config import get_global_classifiers_folder
        initial = os.path.join(self.current_project_folder.get(), 'classifiers')
        if not os.path.isdir(initial):
            initial = self.current_project_folder.get() or os.getcwd()
        src = filedialog.askopenfilename(
            title="Select classifier to export",
            initialdir=initial,
            filetypes=[("Classifier", "*.pkl")])
        if not src:
            return
        dst_dir = get_global_classifiers_folder()
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"{os.path.basename(src)} already exists in global library. Overwrite?"):
                return
        shutil.copy2(src, dst)
        self.refresh_pred_classifiers()
        messagebox.showinfo("Exported",
                            f"Classifier copied to global library:\n{dst}")

    def _import_classifier_from_global(self):
        """Copy a .pkl from the global library into the current project."""
        import shutil
        from user_config import get_global_classifiers_folder
        gcf = get_global_classifiers_folder()
        if not os.path.isdir(gcf) or not any(
                f.endswith('.pkl') for f in os.listdir(gcf)):
            messagebox.showinfo("Empty",
                                "Global classifiers folder is empty or doesn't exist.")
            return
        src = filedialog.askopenfilename(
            title="Select classifier to import",
            initialdir=gcf,
            filetypes=[("Classifier", "*.pkl")])
        if not src:
            return
        pf = self.current_project_folder.get()
        if not pf:
            messagebox.showwarning("No Project",
                                   "Please open a project first.")
            return
        dst_dir = os.path.join(pf, 'classifiers')
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"{os.path.basename(src)} already exists in project. Overwrite?"):
                return
        shutil.copy2(src, dst)
        self.refresh_pred_classifiers()
        messagebox.showinfo("Imported",
                            f"Classifier copied to project:\n{dst}")

    def _open_global_clf_folder(self):
        """Open the global classifiers folder in file explorer."""
        from user_config import get_global_classifiers_folder
        folder = get_global_classifiers_folder()
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)

    def _change_global_clf_folder(self):
        """Let user choose a different global classifiers folder."""
        from user_config import get_global_classifiers_folder, set_global_classifiers_folder
        current = get_global_classifiers_folder()
        new = filedialog.askdirectory(
            title="Select Global Classifiers Folder",
            initialdir=current if os.path.isdir(current) else os.getcwd())
        if new:
            set_global_classifiers_folder(new)
            self.refresh_pred_classifiers()
            # Also refresh eval tab if available
            if hasattr(self, 'eval_tab') and hasattr(self.eval_tab, 'refresh_classifiers'):
                self.eval_tab.refresh_classifiers()
            messagebox.showinfo("Updated",
                                f"Global classifiers folder set to:\n{new}")

    def open_skeleton_renderer(self):
        """Open the Skeleton Video Renderer tool as a Toplevel form."""
        import os, sys, threading, subprocess

        _render_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'render_skeleton_video.py'
        )

        win = tk.Toplevel(self.root)
        win.title("🦴 Skeleton Video Renderer")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.82)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.resizable(True, True)

        # ── helpers ──────────────────────────────────────────────────────────
        proj = self.current_project_folder.get()
        default_dir = os.path.join(proj, 'videos') if (
            proj and os.path.isdir(os.path.join(proj, 'videos'))
        ) else (proj or os.getcwd())

        sessions = find_session_triplets(proj, require_labels=False, recursive=True) if proj else []

        def _outputs_dir():
            p = self.current_project_folder.get()
            if p:
                d = os.path.join(p, 'outputs')
                os.makedirs(d, exist_ok=True)
                return d
            return None

        def _default_out(h5_path):
            stem = os.path.splitext(os.path.basename(h5_path))[0]
            d = _outputs_dir()
            return os.path.join(d, stem + '_skeleton.mp4') if d \
                   else os.path.splitext(h5_path)[0] + '_skeleton.mp4'

        def _browse(var, filetypes, title, save=False):
            if save:
                path = tk.filedialog.asksaveasfilename(
                    title=title, initialdir=default_dir,
                    defaultextension='.mp4', filetypes=filetypes, parent=win)
            else:
                path = tk.filedialog.askopenfilename(
                    title=title, initialdir=default_dir,
                    filetypes=filetypes, parent=win)
            if path:
                var.set(path)
                if var is h5_var and not out_var.get():
                    out_var.set(_default_out(path))

        # ── Files frame ──────────────────────────────────────────────────────
        files_frame = ttk.LabelFrame(win, text=" Files ", padding=8)
        files_frame.pack(fill='x', padx=10, pady=(10, 4))

        h5_var            = tk.StringVar()
        vid_var           = tk.StringVar()
        out_var           = tk.StringVar()
        pred_file_var     = tk.StringVar()
        bout_col_var      = tk.StringVar(value="auto")
        extra_vid_dir_var = tk.StringVar()

        # ── Video folder row (row 0) — optional extra scan root ──────────────
        ttk.Label(files_frame, text="Video folder:", width=13, anchor='e').grid(
            row=0, column=0, sticky='e', padx=(0, 4), pady=3)
        ttk.Entry(files_frame, textvariable=extra_vid_dir_var, width=65).grid(
            row=0, column=1, sticky='ew', pady=3)

        def _browse_extra_dir():
            d = tk.filedialog.askdirectory(title="Select extra video folder", parent=win)
            if d:
                extra_vid_dir_var.set(d)
                _refresh_sessions()

        ttk.Button(files_frame, text="Browse", command=_browse_extra_dir).grid(
            row=0, column=2, padx=(4, 0), pady=3)

        # ── Session row (row 1) ───────────────────────────────────────────────
        ttk.Label(files_frame, text="Session:", width=13, anchor='e').grid(
            row=1, column=0, sticky='e', padx=(0, 4), pady=3)

        session_names = [s['session_name'] for s in sessions]
        session_combo = ttk.Combobox(files_frame, width=45, state='readonly')
        if session_names:
            session_combo.config(values=session_names)
            session_combo.current(0)
        else:
            session_combo.config(values=['— no sessions found —'])
            session_combo.current(0)
            session_combo.config(state='disabled')
        session_combo.grid(row=1, column=1, sticky='ew', pady=3)

        def _find_predictions_csv(session_base):
            p = self.current_project_folder.get()
            if not p:
                return ''
            import glob as _g

            # 1. results/ subfolders — canonical location
            hits = _g.glob(os.path.join(p, 'results', '**',
                           f'{session_base}*_predictions.csv'), recursive=True)
            if not hits:
                hits = _g.glob(os.path.join(p, 'results', '**',
                               f'*{session_base}*_predictions.csv'), recursive=True)

            # 2. Same directory as the currently selected video file
            if not hits:
                vid = vid_var.get().strip()
                if vid:
                    vdir = os.path.dirname(vid)
                    hits = _g.glob(os.path.join(vdir, f'{session_base}*_predictions.csv'))
                    if not hits:
                        hits = _g.glob(os.path.join(vdir, f'*{session_base}*_predictions.csv'))

            # 3. Anywhere in project tree (broad fallback)
            if not hits:
                hits = _g.glob(os.path.join(p, '**',
                               f'{session_base}*_predictions.csv'), recursive=True)
            if not hits:
                hits = _g.glob(os.path.join(p, '**',
                               f'*{session_base}*_predictions.csv'), recursive=True)

            return sorted(hits)[0] if hits else ''

        def _on_session_select(event=None):
            name = session_combo.get()
            match = next((s for s in sessions if s['session_name'] == name), None)
            if not match:
                return
            h5_var.set(match['dlc'])
            vid_var.set(match['video'])
            out_var.set(_default_out(match['dlc']))
            pred_file_var.set(_find_predictions_csv(match['session_name']))

        session_combo.bind('<<ComboboxSelected>>', _on_session_select)
        if sessions:
            _on_session_select()

        def _refresh_sessions():
            nonlocal sessions
            proj2 = self.current_project_folder.get()
            sessions = find_session_triplets(proj2, require_labels=False, recursive=True) if proj2 else []
            extra = extra_vid_dir_var.get().strip()
            if extra and os.path.isdir(extra):
                extra_sessions = find_session_triplets(extra, require_labels=False, recursive=True)
                seen = {s['session_name'] for s in sessions}
                sessions += [s for s in extra_sessions if s['session_name'] not in seen]
            names = [s['session_name'] for s in sessions]
            if names:
                session_combo.config(state='readonly', values=names)
                session_combo.current(0)
                _on_session_select()
            else:
                session_combo.config(state='disabled', values=['— no sessions found —'])
                session_combo.current(0)

        ttk.Button(files_frame, text="↺ Refresh", command=_refresh_sessions).grid(
            row=1, column=2, padx=(4, 0), pady=3)

        # ── File-picker rows (rows 2–4) ───────────────────────────────────────
        for row_idx, (label, var, ftypes, save_flag) in enumerate([
            ("Pose (.h5):",  h5_var,  [("HDF5 files", "*.h5"), ("All files", "*.*")], False),
            ("Video file:",  vid_var, [("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")], False),
            ("Output .mp4:", out_var, [("MP4 files", "*.mp4"), ("All files", "*.*")], True),
        ], start=2):
            ttk.Label(files_frame, text=label, width=13, anchor='e').grid(
                row=row_idx, column=0, sticky='e', padx=(0, 4), pady=3)
            ttk.Entry(files_frame, textvariable=var, width=65).grid(
                row=row_idx, column=1, sticky='ew', pady=3)
            ttk.Button(
                files_frame, text="Browse",
                command=lambda v=var, f=ftypes, t=label, s=save_flag: _browse(v, f, t, s)
            ).grid(row=row_idx, column=2, padx=(4, 0), pady=3)

        # ── Row 5: Predictions file + column picker ───────────────────────────
        ttk.Label(files_frame, text="Predictions:", width=13, anchor='e').grid(
            row=5, column=0, sticky='e', padx=(0, 4), pady=3)
        _pred_entry = ttk.Entry(files_frame, textvariable=pred_file_var, width=46)
        _pred_entry.grid(row=5, column=1, sticky='ew', pady=3)
        ToolTip(_pred_entry,
                "PixelPaws predictions CSV. When set, one clip is rendered per "
                "detected behavior bout — no need to enter Start/End manually.")
        ttk.Button(files_frame, text="Browse",
                   command=lambda: _browse(pred_file_var,
                       [("CSV files", "*.csv"), ("All files", "*.*")],
                       "Select predictions CSV")).grid(row=5, column=2, padx=(4, 0), pady=3)
        bout_col_combo = ttk.Combobox(files_frame, textvariable=bout_col_var,
                                      values=['auto'], state='readonly', width=16)
        bout_col_combo.current(0)
        bout_col_combo.grid(row=5, column=3, padx=(6, 0), pady=3, sticky='w')
        ToolTip(bout_col_combo,
                "Which column to use as the behavior label. 'auto' picks "
                "'prediction' or the first non-frame/probability column.")

        def _on_pred_file_change(*_):
            path = pred_file_var.get().strip()
            if path and os.path.isfile(path):
                try:
                    import pandas as _pd
                    cols = ['auto'] + [c for c in _pd.read_csv(path, nrows=0).columns
                                       if c not in ('frame', 'probability')]
                    bout_col_combo.config(values=cols)
                    if bout_col_var.get() not in cols:
                        bout_col_var.set('auto')
                except Exception:
                    pass
            else:
                bout_col_combo.config(values=['auto'])
                bout_col_var.set('auto')

        pred_file_var.trace_add('write', _on_pred_file_change)

        files_frame.columnconfigure(1, weight=1)

        # ── Bout clip suggestions ─────────────────────────────────────────────
        n_bouts_var = tk.IntVar(value=4)
        sug_pad_var = tk.StringVar(value="120")

        def _suggest_clips():
            """Populate the suggestion listbox with N-bout sliding-window clips."""
            pred_path = pred_file_var.get().strip()
            _sug_lb.delete(0, tk.END)
            if not pred_path or not os.path.isfile(pred_path):
                _sug_lb.insert(tk.END, "— load a predictions CSV first —")
                return
            try:
                import pandas as _pd
                import numpy as _np
                df = _pd.read_csv(pred_path)
                # Resolve column — identical logic to load_bouts
                col = bout_col_var.get().strip()
                if col == 'auto' or col not in df.columns:
                    cands = [c for c in df.columns if c not in ('frame', 'probability')]
                    col = ('prediction' if 'prediction' in df.columns
                           else (cands[0] if cands else None))
                if col is None or col not in df.columns:
                    _sug_lb.insert(tk.END, "— could not find prediction column —")
                    return
                # Build dense array (matches load_bouts exactly)
                if 'frame' in df.columns:
                    pred = dict(zip(df['frame'].astype(int), df[col].astype(int)))
                else:
                    pred = {i: int(v) for i, v in enumerate(df[col])}
                if not pred:
                    _sug_lb.insert(tk.END, "— empty predictions —")
                    return
                max_f = max(pred)
                arr = _np.array([pred.get(i, 0) for i in range(max_f + 1)], dtype=_np.int8)
                # Detect bouts (contiguous runs of non-zero)
                bouts, in_bout, bout_start = [], False, 0
                for i, v in enumerate(arr):
                    if v and not in_bout:
                        bout_start = i; in_bout = True
                    elif not v and in_bout:
                        bouts.append((bout_start, i)); in_bout = False
                if in_bout:
                    bouts.append((bout_start, max_f + 1))
                if not bouts:
                    _sug_lb.insert(tk.END, "— no positive bouts found —")
                    return
                try:
                    pad = int(sug_pad_var.get().strip() or '0')
                except ValueError:
                    pad = 120
                n = n_bouts_var.get()
                if n >= len(bouts):
                    ws = max(0, bouts[0][0] - pad)
                    we = bouts[-1][1] + pad
                    _sug_lb.insert(tk.END,
                        f"All {len(bouts)} bout(s):  frames {ws} – {we}  ({we - ws} fr)")
                else:
                    for i in range(len(bouts) - n + 1):
                        ws = max(0, bouts[i][0] - pad)
                        we = bouts[i + n - 1][1] + pad
                        _sug_lb.insert(tk.END,
                            f"Bouts {i+1}–{i+n}:  frames {ws} – {we}  ({we - ws} fr)")
                _sug_lb.selection_set(0)
                _use_suggestion()
            except Exception as exc:
                _sug_lb.insert(tk.END, f"Error: {exc}")

        def _use_suggestion(event=None):
            sel = _sug_lb.curselection()
            if not sel:
                return
            item = _sug_lb.get(sel[0])
            import re as _re
            m = _re.search(r'frames\s+(\d+)\s+[–\-]\s+(\d+)', item)
            if m:
                start_var.set(m.group(1))
                end_var.set(m.group(2))

        suggest_frame = ttk.LabelFrame(win, text=" Bout clip suggestions ", padding=8)
        suggest_frame.pack(fill='x', padx=10, pady=4)

        _sug_ctrl = ttk.Frame(suggest_frame)
        _sug_ctrl.pack(fill='x')
        ttk.Label(_sug_ctrl, text="Bouts per clip:").pack(side='left')
        ttk.Spinbox(_sug_ctrl, from_=1, to=50, textvariable=n_bouts_var,
                    width=4).pack(side='left', padx=(4, 12))
        ttk.Label(_sug_ctrl, text="Padding (fr):").pack(side='left')
        ttk.Entry(_sug_ctrl, textvariable=sug_pad_var,
                  width=6).pack(side='left', padx=(4, 12))
        ttk.Button(_sug_ctrl, text="Suggest",
                   command=_suggest_clips).pack(side='left')

        _sug_list_frame = ttk.Frame(suggest_frame)
        _sug_list_frame.pack(fill='x', pady=(4, 0))
        _sug_lb = tk.Listbox(_sug_list_frame, height=4, selectmode='single',
                             activestyle='dotbox', font=('Courier', 9))
        _sug_sb = ttk.Scrollbar(_sug_list_frame, command=_sug_lb.yview)
        _sug_lb.configure(yscrollcommand=_sug_sb.set)
        _sug_sb.pack(side='right', fill='y')
        _sug_lb.pack(side='left', fill='x', expand=True)
        _sug_lb.insert(tk.END, "— click Suggest after loading a predictions CSV —")
        _sug_lb.bind('<<ListboxSelect>>', _use_suggestion)

        # ── Parameters frame ─────────────────────────────────────────────────
        params_frame = ttk.LabelFrame(win, text=" Parameters ", padding=8)
        params_frame.pack(fill='x', padx=10, pady=4)

        decay_var           = tk.StringVar(value="0.82")
        glow_var            = tk.StringVar(value="0.2")
        glow_sigma_var      = tk.StringVar(value="2.0")
        lk_var              = tk.StringVar(value="0.3")
        sz_var              = tk.StringVar(value="30")
        hindpaw_sz_var      = tk.StringVar(value="50")
        thr_var             = tk.StringVar(value="0.58")
        trail_interval_var  = tk.StringVar(value="10")
        trail_fade_var      = tk.StringVar(value="0.993")
        grey_paws_var       = tk.BooleanVar(value=False)
        export_orig_var     = tk.BooleanVar(value=False)
        glow_enabled_var    = tk.BooleanVar(value=True)
        trail_enabled_var   = tk.BooleanVar(value=True)
        skel_enabled_var    = tk.BooleanVar(value=True)
        label_bouts_var     = tk.BooleanVar(value=True)
        colorway_var        = tk.StringVar(value="default")
        crop_var            = tk.StringVar(value="")

        # Per-paw custom colours — stored as 'B,G,R' strings (populated from colorway on change)
        _PAW_ORDER = ('hrpaw', 'hlpaw', 'frpaw', 'flpaw')
        _PAW_LABEL = {'hrpaw': 'HR', 'hlpaw': 'HL', 'frpaw': 'FR', 'flpaw': 'FL'}
        _PAW_TIPS  = {'hrpaw': 'Right hind paw', 'hlpaw': 'Left hind paw',
                      'frpaw': 'Right front paw', 'flpaw': 'Left front paw'}
        # BGR colours mirrored from render_skeleton_video.py COLORWAYS
        _GUI_PAW_COLORS = {
            'default': {'hrpaw':(220,210,0),'hlpaw':(200,0,200),'frpaw':(0,155,255),'flpaw':(160,210,0)},
            'redblue': {'hrpaw':(0,80,240),'hlpaw':(200,60,0),'frpaw':(20,140,255),'flpaw':(220,100,20)},
            'neon':    {'hrpaw':(0,255,255),'hlpaw':(255,0,255),'frpaw':(255,255,0),'flpaw':(0,255,128)},
            'pastel':  {'hrpaw':(140,180,210),'hlpaw':(180,130,200),'frpaw':(210,190,130),'flpaw':(130,195,155)},
            'mono':    {'hrpaw':(220,220,220),'hlpaw':(165,165,165),'frpaw':(200,200,200),'flpaw':(145,145,145)},
        }
        paw_custom_bgr = {bp: list(_GUI_PAW_COLORS['default'][bp]) for bp in _PAW_ORDER}

        def _bgr_to_hex(b, g, r):
            return f'#{r:02x}{g:02x}{b:02x}'

        _TIPS = {
            "Trail Decay:":         "How quickly the live skeleton trail fades each frame (0 = instant, 1 = never fade).",
            "Glow Strength:":       "Intensity of the bloom/glow blended around each paw stamp.",
            "Glow Sigma:":          "Spread of the glow — smaller = crisper tight halo, larger = wide soft bloom.",
            "Bright Threshold:":    "Minimum pixel brightness to include in a paw stamp. Higher = only the brightest pad pixels survive → crisper, less noise.",
            "Forepaw size:":        "Half-width (pixels) of the ROI box stamped for front paws (flpaw, frpaw).",
            "Hindpaw size:":        "Half-width (pixels) of the ROI box stamped for hind paws (hlpaw, hrpaw). Hindpaws are larger so a bigger box is appropriate.",
            "Trail Interval (fr):": "Frames between each hindpaw footprint stamp. 10 fr ≈ every 0.4 s at 25 fps.",
            "Trail Fade:":          "Per-frame decay of hindpaw trail stamps. 0.993 ≈ fades out over ~25–30 s.",
            "Min Likelihood:":      "DLC pose confidence threshold. Body-part positions below this value are hidden.",
        }

        param_rows = [
            [("Trail Decay:",         decay_var,          8), ("Glow Strength:",      glow_var,       8)],
            [("Glow Sigma:",          glow_sigma_var,     8), ("Bright Threshold:",   thr_var,        8)],
            [("Forepaw size:",        sz_var,             8), ("Hindpaw size:",        hindpaw_sz_var, 8)],
            [("Trail Interval (fr):", trail_interval_var, 8), ("Trail Fade:",          trail_fade_var, 8)],
            [("Min Likelihood:",      lk_var,             8)],
        ]
        for r, cols in enumerate(param_rows):
            for c, (lbl, var, w) in enumerate(cols):
                lbl_widget = ttk.Label(params_frame, text=lbl, anchor='e')
                lbl_widget.grid(row=r, column=c*2, sticky='e', padx=(8, 4), pady=3)
                ent = ttk.Entry(params_frame, textvariable=var, width=w)
                ent.grid(row=r, column=c*2+1, sticky='w', pady=3)
                if lbl in _TIPS:
                    ToolTip(lbl_widget, _TIPS[lbl])
                    ToolTip(ent, _TIPS[lbl])

        # ── Row 5: Feature toggles ────────────────────────────────────────────
        _tog_frame = ttk.Frame(params_frame)
        _tog_frame.grid(row=5, column=0, columnspan=5, sticky='w', padx=(8,0), pady=(6,2))
        _glow_cb = ttk.Checkbutton(_tog_frame, text="Glow", variable=glow_enabled_var)
        _glow_cb.pack(side='left', padx=(0, 12))
        ToolTip(_glow_cb, "Enable/disable the bloom glow effect. "
                          "Disable for the crispest paw stamps.")
        _trail_cb = ttk.Checkbutton(_tog_frame, text="Hindpaw trail", variable=trail_enabled_var)
        _trail_cb.pack(side='left', padx=(0, 12))
        ToolTip(_trail_cb, "Enable/disable the hindpaw footprint trail. "
                           "Disable to show only the live skeleton without persistent stamps.")
        _skel_cb = ttk.Checkbutton(_tog_frame, text="Skeleton lines", variable=skel_enabled_var)
        _skel_cb.pack(side='left', padx=(0, 12))
        ToolTip(_skel_cb, "Enable/disable the skeleton stick-figure lines drawn on each frame.")
        _gp_cb = ttk.Checkbutton(_tog_frame, text="Grey/natural paws", variable=grey_paws_var)
        _gp_cb.pack(side='left', padx=(12, 0))
        ToolTip(_gp_cb, "When checked, paw stamps show the raw video pixel colours "
                        "without any colour tint. Overrides the Colorway for paw stamps.")
        _lb_cb = ttk.Checkbutton(_tog_frame, text="Label bouts", variable=label_bouts_var)
        _lb_cb.pack(side='left', padx=(12, 0))
        ToolTip(_lb_cb, 'Overlay "<behavior> detected" text on frames during '
                        'each active bout. Only applied in bout mode.')

        # ── Row 6: Colorway + per-paw swatches ───────────────────────────────
        _cw_lbl = ttk.Label(params_frame, text="Colorway:", anchor='e')
        _cw_lbl.grid(row=6, column=0, sticky='e', padx=(8, 4), pady=3)
        ToolTip(_cw_lbl, "Colour palette for paw stamps and skeleton:\n"
                         "  default — gold / magenta / cyan / green\n"
                         "  redblue — warm right, cool left (L/R distinction)\n"
                         "  neon    — fully saturated bright colours\n"
                         "  pastel  — soft low-saturation tints\n"
                         "  mono    — greyscale shades per paw\n"
                         "  custom  — click each paw icon to pick any colour")
        _cw_combo = ttk.Combobox(params_frame, textvariable=colorway_var,
                                 values=['default','redblue','neon','pastel','mono','custom'],
                                 state='readonly', width=9)
        _cw_combo.grid(row=6, column=1, sticky='w', pady=3)

        # Paw swatch canvases
        _swatch_frame = ttk.Frame(params_frame)
        _swatch_frame.grid(row=6, column=2, columnspan=3, sticky='w', padx=(8,0), pady=2)

        _paw_canvases = {}
        for _bp in _PAW_ORDER:
            _col_frame = ttk.Frame(_swatch_frame)
            _col_frame.pack(side='left', padx=4)
            ttk.Label(_col_frame, text=_PAW_LABEL[_bp], font=('Arial', 7)).pack()
            _c = tk.Canvas(_col_frame, width=34, height=36,
                           bg='black', highlightthickness=1, highlightbackground='#888',
                           cursor='hand2')
            _c.pack()
            _paw_canvases[_bp] = _c
            ToolTip(_c, f"{_PAW_TIPS[_bp]} — click to pick a custom colour")

        def _draw_paw(canvas, hex_color):
            canvas.delete('all')
            canvas.create_oval( 4, 18, 30, 34, fill=hex_color, outline='')  # palm
            canvas.create_oval( 2,  5, 11, 14, fill=hex_color, outline='')  # left toe
            canvas.create_oval(12,  1, 22, 11, fill=hex_color, outline='')  # middle toe
            canvas.create_oval(23,  5, 32, 14, fill=hex_color, outline='')  # right toe

        def _update_swatches(*_):
            cw = colorway_var.get()
            if cw != 'custom':
                pal = _GUI_PAW_COLORS.get(cw, _GUI_PAW_COLORS['default'])
                for bp in _PAW_ORDER:
                    paw_custom_bgr[bp] = list(pal[bp])
            for bp in _PAW_ORDER:
                b, g, r = paw_custom_bgr[bp]
                _draw_paw(_paw_canvases[bp], _bgr_to_hex(b, g, r))

        colorway_var.trace_add('write', _update_swatches)
        _update_swatches()

        def _pick_paw_color(bp):
            from tkinter import colorchooser
            b, g, r = paw_custom_bgr[bp]
            init = _bgr_to_hex(b, g, r)
            result = colorchooser.askcolor(color=init,
                                           title=f'Colour for {_PAW_TIPS[bp]}', parent=win)
            if result and result[1]:
                ri, gi, bi = (int(c) for c in result[0])
                paw_custom_bgr[bp] = [bi, gi, ri]  # store as BGR
                colorway_var.set('custom')           # switch to custom mode
                _update_swatches()

        for _bp in _PAW_ORDER:
            _paw_canvases[_bp].bind('<Button-1>',
                                    lambda e, b=_bp: _pick_paw_color(b))

        # ── Row 7: Export original ────────────────────────────────────────────
        _eo_cb = ttk.Checkbutton(params_frame,
                                 text="Also export cropped original video (for side-by-side comparison)",
                                 variable=export_orig_var)
        _eo_cb.grid(row=7, column=0, columnspan=5, sticky='w', padx=(8, 0), pady=3)
        ToolTip(_eo_cb, "Saves a second video of the original footage cropped and trimmed "
                        "to the exact same region and time window — ready for side-by-side.")

        # ── Row 8: Crop ───────────────────────────────────────────────────────
        _crop_lbl = ttk.Label(params_frame, text="Crop X1,Y1,X2,Y2:", anchor='e')
        _crop_lbl.grid(row=8, column=0, sticky='e', padx=(8, 4), pady=3)
        ToolTip(_crop_lbl, "Pixel coordinates of the crop rectangle in the source video. "
                           "Leave blank to use the full frame.")
        ttk.Entry(params_frame, textvariable=crop_var, width=30).grid(
            row=8, column=1, columnspan=3, sticky='w', pady=3)
        ttk.Label(params_frame, text="← blank = full frame", foreground='gray').grid(
            row=8, column=4, sticky='w', padx=4)

        start_var = tk.StringVar(value="")
        end_var   = tk.StringVar(value="")

        # ── Row 9: Start / End ────────────────────────────────────────────────
        _st_lbl = ttk.Label(params_frame, text="Start:", anchor='e')
        _st_lbl.grid(row=9, column=0, sticky='e', padx=(8, 4), pady=3)
        ToolTip(_st_lbl, "First frame/time to render. Enter a frame number (e.g. 2000) "
                         "or M:SS / H:MM:SS (e.g. 1:30). Leave blank for beginning.")
        ttk.Entry(params_frame, textvariable=start_var, width=14).grid(
            row=9, column=1, sticky='w', pady=3)
        _en_lbl = ttk.Label(params_frame, text="End:", anchor='e')
        _en_lbl.grid(row=9, column=2, sticky='e', padx=(8, 4), pady=3)
        ToolTip(_en_lbl, "Last frame/time to render (exclusive). Same format as Start. "
                         "Leave blank to render to end of video.")
        ttk.Entry(params_frame, textvariable=end_var, width=14).grid(
            row=9, column=3, sticky='w', pady=3)
        ttk.Label(params_frame,
                  text="← frame number or M:SS  (blank = full video)",
                  foreground='gray').grid(row=9, column=4, sticky='w', padx=4)

        # ── Buttons + progress ────────────────────────────────────────────────
        ctrl_frame = ttk.Frame(win)
        ctrl_frame.pack(fill='x', padx=10, pady=4)

        progress_var = tk.DoubleVar(value=0)
        render_btn = ttk.Button(ctrl_frame, text="▶ Render", width=14)
        cancel_btn = ttk.Button(ctrl_frame, text="✖ Cancel", width=14, state='disabled')
        render_btn.pack(side='left', padx=(0, 8))
        cancel_btn.pack(side='left')

        progress_bar = ttk.Progressbar(
            win, variable=progress_var, maximum=100, length=400)
        progress_bar.pack(fill='x', padx=10, pady=(0, 4))

        pct_label = ttk.Label(win, text="0%", anchor='center')
        pct_label.pack()

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(win, text=" Log ", padding=4)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        log_text = tk.Text(log_frame, height=10, state='normal', wrap='word',
                           font=('Courier', 9))
        log_scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
        log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side='right', fill='y')
        log_text.pack(fill='both', expand=True)

        # ── State ─────────────────────────────────────────────────────────────
        _state = {'proc': None}

        def _log(msg):
            log_text.insert(tk.END, msg + '\n')
            log_text.see(tk.END)

        def _on_done(returncode):
            render_btn.config(state='normal')
            cancel_btn.config(state='disabled')
            if returncode == 0:
                progress_var.set(100)
                pct_label.config(text="100%")
                _log("✓ Done — output saved.")
            else:
                progress_var.set(0)
                pct_label.config(text="Error")
                _log(f"✗ Error (return code {returncode}) — see log above.")

        def _run_thread(cmd):
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                _state['proc'] = proc
                for line in proc.stdout:
                    line = line.rstrip()
                    win.after(0, lambda l=line: _log(l))
                    m = re.search(r'frame\s+(\d+)/(\d+)', line)
                    if m:
                        done, total = int(m.group(1)), int(m.group(2))
                        pct = 100.0 * done / total if total else 0
                        win.after(0, lambda p=pct: (
                            progress_var.set(p),
                            pct_label.config(text=f"{p:.0f}%")
                        ))
                proc.wait()
                win.after(0, lambda: _on_done(proc.returncode))
            except Exception as exc:
                win.after(0, lambda: _log(f"✗ Exception: {exc}"))
                win.after(0, lambda: _on_done(-1))

        def _do_render():
            h5  = h5_var.get().strip()
            vid = vid_var.get().strip()
            out = out_var.get().strip()
            if not h5 or not os.path.isfile(h5):
                tk.messagebox.showerror("Missing file", "Please select a valid pose .h5 file.", parent=win)
                return
            if not vid or not os.path.isfile(vid):
                tk.messagebox.showerror("Missing file", "Please select a valid video file.", parent=win)
                return
            if not out:
                out = _default_out(h5)
                out_var.set(out)

            cmd = [sys.executable, '-u', _render_script, h5, vid,
                   '--output', out,
                   '--decay',      decay_var.get().strip(),
                   '--likelihood', lk_var.get().strip(),
                   '--threshold',  thr_var.get().strip(),
                   '--size',            sz_var.get().strip(),
                   '--hindpaw-size',    hindpaw_sz_var.get().strip(),
                   '--glow',            glow_var.get().strip(),
                   '--glow-sigma',      glow_sigma_var.get().strip(),
                   '--trail-interval',  trail_interval_var.get().strip(),
                   '--trail-decay',     trail_fade_var.get().strip(),
                   '--colorway',        colorway_var.get() if colorway_var.get() != 'custom' else 'default']
            # Per-paw colour overrides (always sent in custom mode; no-op otherwise unless changed)
            if colorway_var.get() == 'custom':
                for _bp in _PAW_ORDER:
                    b, g, r = paw_custom_bgr[_bp]
                    cmd += [f'--color-{_bp}', f'{b},{g},{r}']
            if grey_paws_var.get():
                cmd += ['--grey-paws']
            if not glow_enabled_var.get():
                cmd += ['--no-glow']
            if not trail_enabled_var.get():
                cmd += ['--no-trail']
            if not skel_enabled_var.get():
                cmd += ['--no-skeleton']
            if export_orig_var.get():
                cmd += ['--export-original']
            crop_str = crop_var.get().strip()
            if crop_str:
                cmd += ['--crop', crop_str]

            def _parse_range_arg(s):
                """Return ('frame', int) or ('time', float) or None."""
                s = s.strip()
                if not s:
                    return None
                if ':' in s:          # M:SS or H:MM:SS
                    parts = s.split(':')
                    try:
                        secs = sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
                        return ('time', secs)
                    except ValueError:
                        return None
                try:
                    return ('frame', int(s))
                except ValueError:
                    return None

            for arg_name, raw in [('start', start_var.get()), ('end', end_var.get())]:
                parsed = _parse_range_arg(raw)
                if parsed:
                    kind, val = parsed
                    if kind == 'frame':
                        cmd += [f'--{arg_name}-frame', str(val)]
                    else:
                        cmd += [f'--{arg_name}-time', f'{val:.3f}']

            pred_path = pred_file_var.get().strip()
            # Only use bout mode when no explicit start/end clip range is set.
            # If the user picked a suggestion window, Start+End are already filled
            # and the render should be a plain single clip — not all bouts.
            _has_range = bool(start_var.get().strip() or end_var.get().strip())
            if pred_path and os.path.isfile(pred_path):
                col = bout_col_var.get().strip()
                if not _has_range:
                    # True bout mode: concatenate all bouts
                    cmd += ['--bout-file', pred_path]
                    if col and col != 'auto':
                        cmd += ['--bout-column', col]
                    if label_bouts_var.get():
                        cmd += ['--label-bouts']
                elif label_bouts_var.get():
                    # Single clip from suggestion: pass bout-file for label text only
                    cmd += ['--bout-file', pred_path, '--label-bouts']
                    if col and col != 'auto':
                        cmd += ['--bout-column', col]

            render_btn.config(state='disabled')
            cancel_btn.config(state='normal')
            progress_var.set(0)
            pct_label.config(text="0%")
            log_text.delete('1.0', tk.END)
            mode = ('bout mode' if pred_path and os.path.isfile(pred_path) and not _has_range
                    else 'single clip')
            _log(f"[start] {mode} — {' '.join(cmd)}")

            threading.Thread(target=_run_thread, args=(cmd,), daemon=True).start()

        def _do_cancel():
            proc = _state.get('proc')
            if proc:
                proc.terminate()
                _log("[cancel] Render cancelled by user.")
            render_btn.config(state='normal')
            cancel_btn.config(state='disabled')

        render_btn.config(command=_do_render)
        cancel_btn.config(command=_do_cancel)

    def _create_active_learning_tab_v2(self):
        """Create Active Learning v2 tab using ActiveLearningTabV2 class."""
        al_frame = ttk.Frame(self.notebook)
        self.notebook.add(al_frame, text="🧠 Active Learning")
        self.al_frame = al_frame

        if ACTIVE_LEARNING_AVAILABLE:
            self._al_tab = ActiveLearningTabV2(al_frame, self)
        else:
            err = ttk.Frame(al_frame)
            err.pack(expand=True, fill='both', padx=20, pady=20)
            ttk.Label(err, text="⚠️ Active Learning Module Not Available",
                      font=('Arial', 16, 'bold'), foreground='red').pack(pady=20)
            ttk.Label(err, text="Please ensure active_learning_v2.py is in the same directory as PixelPaws_GUI.py",
                      font=('Arial', 12)).pack(pady=10)



    def _toggle_advanced(self):
        """Show/hide Advanced Settings (XGBoost + Training Parameters)."""
        if self._advanced_visible.get():
            self._xgb_frame.pack(fill='x', padx=15, pady=8, after=self._feature_frame)
            self._params_frame.pack(fill='x', padx=15, pady=8, after=self._xgb_frame)
        else:
            self._xgb_frame.pack_forget()
            self._params_frame.pack_forget()

    def _describe_training_profile(self):
        """Compose a short label describing which options were enabled for the run.

        Replaces the former preset-dropdown name in training_history.csv so each
        run is still identifiable by its config even after the dropdown was
        removed in favour of a single 'Thorough by default' story.
        """
        tags = []
        if self.train_use_optuna is not None and self.train_use_optuna.get():
            tags.append('optuna')
        if self.train_learning_curve is not None and self.train_learning_curve.get():
            tags.append('lcurve')
        if self.train_use_calibration is not None and self.train_use_calibration.get():
            tags.append('cal')
        if self.train_use_fold_ensemble is not None and self.train_use_fold_ensemble.get():
            tags.append('fe')
        if self.train_prune_by_gain is not None and self.train_prune_by_gain.get():
            try:
                tags.append(f'prune{int(self.train_prune_top_n.get())}')
            except Exception:
                tags.append('prune')
        return '+'.join(tags) if tags else 'none'

    def _update_training_warning(self, *args):
        """Show a yellow banner when the user has enabled a combination of
        options that implies a long training run."""
        if self._train_warning_lbl is None:
            return
        msgs = []
        if (self.train_use_optuna is not None
                and self.train_learning_curve is not None
                and self.train_use_optuna.get()
                and self.train_learning_curve.get()):
            msgs.append("⚠ Optuna + Learning Curve together: ≈30+ min expected")
        self._train_warning_lbl.config(text=("  " + "  ".join(msgs)) if msgs else "")

    def _write_training_sidecar(self, sidecar_path, classifier_data, classifier_path,
                                 run_ts, profile_label,
                                 optuna_best_hp, optuna_best_ap,
                                 oof_best_params, lovo_best_params,
                                 final_spw, behavior_name, y):
        """Write PixelPaws_<behavior>_<ts>.json — a machine-readable record of
        every hyperparameter, Optuna pick, feature flag, and metric used for
        one training run.  Diagnostic / reproducibility only — failures never
        abort training (caller wraps in try/except).
        """
        import json
        from datetime import datetime as _dt

        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return default

        def _safe_list(v):
            try:
                return list(v) if v is not None else []
            except Exception:
                return []

        model = classifier_data.get('clf_model')
        sessions_names = classifier_data.get('training_sessions', []) or []

        sidecar = {
            'timestamp':      _dt.now().isoformat(timespec='seconds'),
            'run_ts':         run_ts,
            'classifier_pkl': os.path.basename(classifier_path),
            'behavior':       behavior_name,
            'training_profile': profile_label,
            'sessions': {
                'count': len(sessions_names),
                'names': list(sessions_names),
                'total_frames':    int(len(y)) if y is not None else 0,
                'positive_frames': int(np.sum(y)) if y is not None else 0,
            },
            'metrics': {
                'cv_f1_mean': round(_safe_float(classifier_data.get('mean_cv_f1')), 4),
                'cv_f1_std':  round(_safe_float(classifier_data.get('std_cv_f1')), 4),
                'cv_f1_per_fold': [round(_safe_float(v), 4)
                                   for v in _safe_list(classifier_data.get('cv_f1_scores'))],
                'oof_best_f1': round(_safe_float(classifier_data.get('oof_best_f1')), 4),
            },
            'post_processing': {
                'oof':  {k: v for k, v in (oof_best_params or {}).items()},
                'lovo': {k: v for k, v in (lovo_best_params or {}).items()},
            },
            'xgboost_hyperparameters': {
                'n_estimators':     int(classifier_data.get('final_n_estimators', 0) or 0),
                'max_depth':        int(self.train_max_depth.get()) if self.train_max_depth else None,
                'learning_rate':    _safe_float(self.train_learning_rate.get()) if self.train_learning_rate else None,
                'subsample':        _safe_float(self.train_subsample.get()) if self.train_subsample else None,
                'colsample_bytree': _safe_float(self.train_colsample.get()) if self.train_colsample else None,
                'scale_pos_weight': round(_safe_float(final_spw), 3),
                'objective':        'binary:logistic',
                'eval_metric':      'aucpr',
            },
            'optuna': {
                'used':          bool(optuna_best_hp is not None),
                'n_trials':      int(self.train_optuna_trials.get()) if self.train_optuna_trials else None,
                'best_params':   dict(optuna_best_hp) if optuna_best_hp else None,
                'best_value_ap': round(_safe_float(optuna_best_ap), 4) if optuna_best_ap is not None else None,
            },
            'training_flags': {
                'use_calibration':     bool(self.train_use_calibration.get()) if self.train_use_calibration else False,
                'use_fold_ensemble':   bool(self.train_use_fold_ensemble.get()) if self.train_use_fold_ensemble else False,
                'use_lag_features':    bool(self.train_use_lag_features.get()) if self.train_use_lag_features else False,
                'use_egocentric':      bool(self.train_use_egocentric.get()) if self.train_use_egocentric else False,
                'use_contact_features': bool(self.train_use_contact_features.get()) if self.train_use_contact_features else False,
                'correlation_filter':  bool(self.train_correlation_filter.get()) if self.train_correlation_filter else False,
                'prune_by_gain':       bool(self.train_prune_by_gain.get()) if self.train_prune_by_gain else False,
                'prune_top_n':         int(self.train_prune_top_n.get()) if self.train_prune_top_n else None,
                'learning_curve':      bool(self.train_learning_curve.get()) if self.train_learning_curve else False,
            },
            'feature_config': {
                'bp_include_list':      _safe_list(classifier_data.get('bp_include_list')),
                'bp_pixbrt_list':       _safe_list(classifier_data.get('bp_pixbrt_list')),
                'square_size':          _safe_list(classifier_data.get('square_size')),
                'pix_threshold':        _safe_float(classifier_data.get('pix_threshold')),
                'contact_threshold':    _safe_float(classifier_data.get('contact_threshold')),
                'include_optical_flow': bool(classifier_data.get('include_optical_flow', False)),
                'bp_optflow_list':      _safe_list(classifier_data.get('bp_optflow_list')),
                'pose_feature_version':       int(classifier_data.get('pose_feature_version', 0) or 0),
                'brightness_feature_version': int(classifier_data.get('brightness_feature_version', 0) or 0),
            },
            'feature_count': (
                len(classifier_data.get('selected_feature_cols') or [])
                or (len(model.feature_names_in_)
                    if model is not None and hasattr(model, 'feature_names_in_') else 0)
            ),
        }

        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(sidecar, f, indent=2, default=str)

    def _append_training_history(self, classifier_data, preset_name, notes=""):
        """Append one row to <project>/classifiers/training_history.csv.

        Diagnostic-only — failures never abort training (caller wraps in try/except).
        """
        import csv
        from datetime import datetime as _dt
        proj = self.current_project_folder.get() if self.current_project_folder else ''
        if not proj:
            return  # no project folder — skip silently
        hist_path = os.path.join(proj, 'classifiers', 'training_history.csv')
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)

        _model = classifier_data.get('clf_model')
        _n_features = (
            len(classifier_data.get('selected_feature_cols') or [])
            or (len(_model.feature_names_in_)
                if _model is not None and hasattr(_model, 'feature_names_in_')
                else 0)
        )
        row = {
            'timestamp':         _dt.now().isoformat(timespec='seconds'),
            'behavior':          classifier_data.get('Behavior_type', ''),
            'mean_cv_f1':        round(float(classifier_data.get('mean_cv_f1', 0.0) or 0.0), 4),
            'std_cv_f1':         round(float(classifier_data.get('std_cv_f1', 0.0) or 0.0), 4),
            'oof_best_f1':       round(float(classifier_data.get('oof_best_f1', 0.0) or 0.0), 4),
            'n_sessions':        len(classifier_data.get('training_sessions', []) or []),
            'n_features':        _n_features,
            'preset':            preset_name,
            'prune_by_gain':     bool(self.train_prune_by_gain.get()) if self.train_prune_by_gain else False,
            'prune_top_n':       int(self.train_prune_top_n.get()) if self.train_prune_top_n else 0,
            'use_optuna':        bool(self.train_use_optuna.get()) if self.train_use_optuna else False,
            'use_calibration':   bool(self.train_use_calibration.get()) if self.train_use_calibration else False,
            'use_fold_ensemble': bool(self.train_use_fold_ensemble.get()) if self.train_use_fold_ensemble else False,
            'notes':             notes,
        }
        write_header = not os.path.isfile(hist_path)
        with open(hist_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    # === TRAINING TAB METHODS ===
    
    def browse_train_project(self):
        """Browse for training project folder"""
        folder = filedialog.askdirectory(title="Select Project Folder")
        if folder:
            self.train_project_folder.set(folder)
            self.current_project_folder.set(folder)
            try:
                from project_setup import _save_recent
                _save_recent(folder)
            except Exception:
                pass

    def _show_startup_wizard(self):
        """Show the project setup wizard; keep main window hidden until complete."""
        try:
            from project_setup import ProjectSetupWizard
            ProjectSetupWizard(self.root, self)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.deiconify()
            messagebox.showerror("Setup Error",
                                 f"Failed to launch project setup wizard:\n{e}")

    def _on_project_folder_changed(self, *_):
        """Called whenever current_project_folder changes — sync all tabs and load project config."""
        folder = self.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            return

        # Update status bar project name
        self._project_display_name.set(f"Project: {os.path.basename(folder)}")

        # Clear stale session selection from previous project
        self._scanned_sessions.clear()
        self._session_checked.clear()
        if self._session_tree is not None:
            for item in self._session_tree.get_children():
                self._session_tree.delete(item)

        # Auto-scan sessions for new project (silent — no popups)
        self.root.after(100, lambda: self._scan_and_populate_sessions(silent=True))

        # Sync tab-specific folder vars so users don't have to re-enter
        if self.train_project_folder is not None:
            self.train_project_folder.set(folder)
        if self.batch_folder is not None:
            self.batch_folder.set(folder)
        if hasattr(self, 'evaluation_tab') and self.evaluation_tab is not None:
            if hasattr(self.evaluation_tab, 'eval_test_folder'):
                self.evaluation_tab.eval_test_folder.set(folder)

        # Auto-load project config if one exists
        config_path = os.path.join(folder, 'PixelPaws_project.json')
        if os.path.isfile(config_path):
            self._load_project_config(config_path, silent=True)

        # Load key file (group assignments) if present
        self._load_key_file(folder)

        # Refresh classifier dropdowns
        self.refresh_pred_classifiers()
        self.refresh_pred_videos()

        # Sync analysis tab project folder and trigger background scan
        if hasattr(self, 'analysis_tab') and self.analysis_tab is not None:
            if hasattr(self.analysis_tab, 'analysis_project_var'):
                self.analysis_tab.analysis_project_var.set(folder)
                # Defer scan slightly so the tab finishes any pending layout
                self.root.after(200, lambda: self.analysis_tab.scan_project_folder(folder))

        # Sync unsupervised tab
        if UNSUPERVISED_TAB_AVAILABLE and hasattr(self, 'unsupervised_tab'):
            self.unsupervised_tab.on_project_changed()

        # Sync transitions tab
        if TRANSITIONS_TAB_AVAILABLE and hasattr(self, 'transitions_tab'):
            self.transitions_tab.on_project_changed()

        # Sync weight bearing tab
        if GAIT_LIMB_TAB_AVAILABLE and hasattr(self, 'wb_tab'):
            self.wb_tab.on_project_changed()

        # Write back (merge) so any newly set fields are persisted immediately
        self.save_project_config(folder)

    def save_project_config(self, folder=None):
        """
        Save a project-level config file to <project>/PixelPaws_project.json.

        Merges with any existing file so that fields not currently held in memory
        (e.g. last_classifier when called before training) are not overwritten
        with empty strings.
        """
        import json
        if folder is None:
            folder = self.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            return

        if PROJECT_CONFIG_AVAILABLE:
            cfg = ProjectConfig.load(folder)
            cfg.video_ext = (self.train_video_ext.get() if self.train_video_ext else '') or cfg.video_ext
            cfg.dlc_config = (self.train_dlc_config.get() if hasattr(self, 'train_dlc_config') and self.train_dlc_config else '') or cfg.dlc_config
            cfg.behavior_name = (self.train_behavior_name.get() if self.train_behavior_name else '') or cfg.behavior_name
            if hasattr(self, 'last_training_results') and self.last_training_results:
                clf = self.last_training_results.get('classifier_path', '')
                if clf:
                    cfg.last_classifier = clf
            cfg.save(folder)
            return

        # Inline fallback
        config_path = os.path.join(folder, 'PixelPaws_project.json')
        existing = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing = json.load(f)
            except Exception as _cfg_err:
                print(f"Warning: could not load project config {config_path}: {_cfg_err}")

        updates = {'project_folder': folder}
        for key, getter in [
            ('video_ext',     lambda: self.train_video_ext.get() if self.train_video_ext else ''),
            ('dlc_config',    lambda: self.train_dlc_config.get() if hasattr(self, 'train_dlc_config') and self.train_dlc_config else ''),
            ('behavior_name', lambda: self.train_behavior_name.get() if self.train_behavior_name else ''),
        ]:
            val = getter()
            if val:
                updates[key] = val

        if hasattr(self, 'last_training_results') and self.last_training_results:
            clf = self.last_training_results.get('classifier_path', '')
            if clf:
                updates['last_classifier'] = clf

        existing.update(updates)

        try:
            with open(config_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save project config: {e}")

    # ------------------------------------------------------------------
    # Key file helpers
    # ------------------------------------------------------------------

    def _load_key_file(self, folder: str):
        """Read <project>/key_file.csv → self.key_file_data {Subject: Treatment}."""
        import csv
        key_path = os.path.join(folder, 'key_file.csv')
        self.key_file_data = {}
        if os.path.isfile(key_path):
            try:
                with open(key_path, newline='') as f:
                    for row in csv.DictReader(f):
                        subj = row.get('Subject', '').strip()
                        trt  = row.get('Treatment', '').strip()
                        if subj:
                            self.key_file_data[subj] = trt
            except Exception as e:
                print(f"Warning: could not load key file: {e}")

    def _open_key_file_dialog(self):
        """Open the KeyFileGeneratorDialog for the current project.
        If no key file exists yet, this is the one place we prompt the user."""
        if not _KEY_FILE_DIALOG_AVAILABLE:
            messagebox.showerror("Unavailable",
                                 "project_setup.py not found — cannot open key file dialog.")
            return
        folder = self.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Project", "Please open a project first.")
            return
        videos_dir = os.path.join(folder, 'videos')
        import glob as _g
        _seen = {}
        for ext in ('.mp4', '.avi', '.mov', '.wmv', '.MP4', '.AVI', '.MOV', '.WMV'):
            for vf in _g.glob(os.path.join(videos_dir, f'*{ext}')):
                _seen[os.path.normcase(vf)] = vf
        basenames = [os.path.splitext(os.path.basename(v))[0]
                     for v in sorted(_seen.values())]
        if not basenames:
            messagebox.showinfo("No Videos",
                                "No video files found in videos/.\n"
                                "Add your videos first.")
            return

        def _on_save(data):
            self.key_file_data = data

        KeyFileGeneratorDialog(self.root, folder, basenames,
                               existing_groups=self.key_file_data.copy(),
                               on_save=_on_save)

    def _load_project_config(self, config_path, silent=False):
        """Load project-level config and populate tab fields."""
        try:
            folder = os.path.dirname(config_path)

            if PROJECT_CONFIG_AVAILABLE:
                cfg = ProjectConfig.load(folder)
                config = cfg.to_dict()
            else:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

            if self.train_video_ext is not None and config.get('video_ext'):
                self.train_video_ext.set(config['video_ext'])
            if hasattr(self, 'train_dlc_config') and self.train_dlc_config is not None and config.get('dlc_config'):
                self.train_dlc_config.set(config['dlc_config'])

            # Load behaviors list -> pre-fill first entry into training tab
            behaviors = config.get('behaviors') or []
            if not behaviors and config.get('behavior_name'):
                behaviors = [config['behavior_name']]
            if behaviors and self.train_behavior_name is not None:
                self.train_behavior_name.set(behaviors[0])

            # Load brightness body parts -> pre-fill training tab field
            bp = config.get('bp_pixbrt_list', [])
            if bp and self.train_bp_pixbrt is not None:
                self.train_bp_pixbrt.set(','.join(bp) if isinstance(bp, list) else bp)

            # Load optical flow settings — only pre-fill True; False keeps the default True
            if config.get('include_optical_flow'):
                self.train_include_optical_flow.set(True)
            optflow_bp = config.get('bp_optflow_list', [])
            if optflow_bp and self.train_bp_optflow is not None:
                self.train_bp_optflow.set(','.join(optflow_bp) if isinstance(optflow_bp, list) else optflow_bp)

            # Pre-fill prediction tab classifier if available
            if (self.pred_classifier_path is not None
                    and config.get('last_classifier')
                    and os.path.isfile(config['last_classifier'])):
                self.pred_classifier_path.set(config['last_classifier'])

            if not silent:
                messagebox.showinfo("Project Loaded",
                    f"Project config loaded from:\n{config_path}")
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Could not load project config:\n{e}")
    
    def browse_train_dlc_config(self):
        """Browse for DLC config.yaml for training"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=self.train_project_folder.get() if self.train_project_folder.get() else None
        )
        if filepath:
            self.train_dlc_config.set(filepath)

    # ------------------------------------------------------------------
    # ROI preview dialog — overlay current square_size boxes on a video
    # frame using DLC coordinates, so the user can verify ROI size visually
    # before training runs extraction.
    # ------------------------------------------------------------------

    def _pick_roi_preview_session(self):
        """Return (video_path, dlc_h5_path) for the ROI preview.

        Tries project/videos/ first, then falls back to a file dialog.
        """
        proj = self.train_project_folder.get() or self.current_project_folder.get() or ''
        videos_dir = os.path.join(proj, 'videos') if proj else ''
        exts = ('.mp4', '.avi', '.mov', '.mkv')

        if videos_dir and os.path.isdir(videos_dir):
            vids = sorted(f for f in os.listdir(videos_dir)
                          if f.lower().endswith(exts))
            for v in vids:
                v_path = os.path.join(videos_dir, v)
                base = os.path.splitext(v)[0]
                dlc_candidates = glob.glob(os.path.join(videos_dir, f"{base}DLC*.h5"))
                if dlc_candidates:
                    # Prefer filtered
                    filt = [f for f in dlc_candidates if 'filtered' in f.lower()]
                    return v_path, (filt or dlc_candidates)[0]

        # Fallback: ask user
        v_path = filedialog.askopenfilename(
            title="Pick a video for ROI preview",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if not v_path:
            return None, None
        base = os.path.splitext(os.path.basename(v_path))[0]
        v_dir = os.path.dirname(v_path)
        dlc_candidates = glob.glob(os.path.join(v_dir, f"{base}DLC*.h5"))
        if not dlc_candidates:
            dlc_candidates = glob.glob(os.path.join(v_dir, '*.h5'))
        if not dlc_candidates:
            messagebox.showerror(
                "No DLC file",
                f"Could not find a DLC .h5 for this video in:\n{v_dir}")
            return None, None
        filt = [f for f in dlc_candidates if 'filtered' in f.lower()]
        return v_path, (filt or dlc_candidates)[0]

    def _open_roi_preview_dialog(self):
        """Open a Toplevel previewing ROI boxes overlaid on a sample frame.

        Shows per-bodypart ROI (colored rectangle + mean brightness inside)
        so the user can visually confirm `square_size` is appropriate before
        running feature extraction.
        """
        try:
            import cv2 as _cv2
            import pandas as _pd
            from PIL import Image as _Image, ImageTk as _ImageTk
        except Exception as e:
            messagebox.showerror(
                "Missing dependency",
                f"ROI preview needs OpenCV + Pillow: {e}")
            return

        # Current bodyparts + sizes from training-tab fields
        bp_list = [s.strip() for s in self.train_bp_pixbrt.get().split(',') if s.strip()]
        if not bp_list:
            messagebox.showwarning(
                "No bodyparts",
                "Set 'Pixel Brightness Body Parts' first.")
            return
        try:
            sizes = [int(s.strip()) for s in self.train_square_sizes.get().split(',') if s.strip()]
        except ValueError:
            sizes = []
        # Pad / truncate to match bp_list
        while len(sizes) < len(bp_list):
            sizes.append(sizes[-1] if sizes else 40)
        sizes = sizes[:len(bp_list)]

        v_path, dlc_path = self._pick_roi_preview_session()
        if not v_path:
            return

        # Load DLC coords — flatten multi-index columns
        try:
            dlc = _pd.read_hdf(dlc_path)
            if isinstance(dlc.columns, _pd.MultiIndex):
                dlc.columns = ['_'.join(c).strip() for c in dlc.columns.values]
                dlc.columns = [c.replace('_likelihood', '_prob') for c in dlc.columns]
        except Exception as e:
            messagebox.showerror("DLC load failed", str(e))
            return

        # Match bodyparts to DLC columns (case-insensitive, ignore spaces/hyphens)
        def _find_bp_cols(bp):
            bp_k = bp.replace(' ', '').replace('-', '').lower()
            x_col = y_col = p_col = None
            for col in dlc.columns:
                c = col.replace(' ', '').replace('-', '').lower()
                if bp_k in c:
                    if c.endswith('_x') or c.endswith('x'):
                        x_col = col
                    elif c.endswith('_y') or c.endswith('y'):
                        y_col = col
                    elif 'prob' in c or 'likelihood' in c:
                        p_col = col
            return x_col, y_col, p_col

        bp_cols = {bp: _find_bp_cols(bp) for bp in bp_list}
        missing_bps = [bp for bp, cols in bp_cols.items() if cols[0] is None]
        if missing_bps:
            messagebox.showwarning(
                "Unmatched bodyparts",
                f"These bodyparts weren't found in the DLC file:\n{missing_bps}\n"
                f"Available columns: {list(dlc.columns[:10])}...")

        # Optional crop offset from DLC config (so boxes land on uncropped video)
        crop_x = crop_y = 0
        cfg_path = self.train_dlc_config.get() if self.train_dlc_config else ''
        if cfg_path and os.path.isfile(cfg_path):
            try:
                import yaml as _yaml
                with open(cfg_path) as fh:
                    _cfg = _yaml.safe_load(fh)
                _crop = _cfg.get('crop', '')
                if isinstance(_crop, str) and ',' in _crop:
                    parts = [int(p.strip()) for p in _crop.split(',')]
                    # Format is usually "x1, x2, y1, y2"
                    if len(parts) >= 3:
                        crop_x, crop_y = parts[0], parts[2]
            except Exception:
                pass

        cap = _cv2.VideoCapture(v_path)
        n_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        if n_frames <= 0:
            cap.release()
            messagebox.showerror("Video error", "Could not read frame count.")
            return

        # Find a well-tracked frame as the default
        def _find_good_frame():
            valid_bps = [bp for bp, c in bp_cols.items() if c[2] is not None]
            if not valid_bps:
                return 0
            probs = _pd.concat(
                [dlc[bp_cols[bp][2]].rename(bp) for bp in valid_bps], axis=1)
            good = probs.min(axis=1) > 0.8
            if good.any():
                idxs = good[good].index.tolist()
                return int(idxs[len(idxs) // 2])
            return 0

        default_frame = min(max(_find_good_frame(), 0), n_frames - 1)

        # Build the dialog
        win = tk.Toplevel(self.root)
        win.title(f"ROI Preview — {os.path.basename(v_path)}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        ww, wh = int(sw * 0.75), int(sh * 0.82)
        win.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")

        info = ttk.Label(
            win, padding=(8, 6, 8, 2),
            text=f"Video: {os.path.basename(v_path)}  |  "
                 f"{n_frames} frames @ {fps:.1f} fps  |  "
                 f"Crop offset: ({crop_x}, {crop_y})  |  "
                 f"DLC: {os.path.basename(dlc_path)}")
        info.pack(fill='x')

        # Frame slider
        slider_row = ttk.Frame(win, padding=(8, 2))
        slider_row.pack(fill='x')
        ttk.Label(slider_row, text="Frame:").pack(side='left')
        frame_var = tk.IntVar(value=default_frame)
        frame_lbl = ttk.Label(slider_row, text=str(default_frame), width=8)
        slider = ttk.Scale(
            slider_row, from_=0, to=n_frames - 1, variable=frame_var,
            orient='horizontal')
        slider.pack(side='left', fill='x', expand=True, padx=(6, 6))
        frame_lbl.pack(side='left')
        ttk.Button(
            slider_row, text="↻ Next good frame",
            command=lambda: frame_var.set(
                min(n_frames - 1, _find_good_frame()))
        ).pack(side='left', padx=4)

        # Canvas
        canvas = tk.Canvas(win, bg='black')
        canvas.pack(fill='both', expand=True, padx=8, pady=4)

        # ROI-size controls + readouts
        bottom = ttk.LabelFrame(win, text="ROI Size Adjustment", padding=8)
        bottom.pack(fill='x', padx=8, pady=(0, 4))
        size_vars = {}
        readouts = {}
        for bp, sz in zip(bp_list, sizes):
            row = ttk.Frame(bottom)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=f"{bp}:", width=14, anchor='w').pack(side='left')
            sv = tk.IntVar(value=int(sz))
            ttk.Spinbox(
                row, from_=5, to=300, textvariable=sv, width=6,
                command=lambda: _redraw()
            ).pack(side='left', padx=4)
            size_vars[bp] = sv
            ro = ttk.Label(row, text="mean=?, area=?", foreground='gray', width=28)
            ro.pack(side='left', padx=8)
            readouts[bp] = ro

        # Action buttons
        action = ttk.Frame(win, padding=8)
        action.pack(fill='x')

        def _apply():
            new_sizes = [int(size_vars[bp].get()) for bp in bp_list]
            self.train_square_sizes.set(','.join(str(s) for s in new_sizes))
            messagebox.showinfo(
                "Applied",
                f"Square sizes updated: {new_sizes}")

        def _close():
            try:
                cap.release()
            except Exception:
                pass
            win.destroy()

        ttk.Button(action, text="✓ Apply to Training",
                   command=_apply).pack(side='left', padx=4)
        ttk.Button(action, text="Close",
                   command=_close).pack(side='right', padx=4)

        # Colors for each bodypart (cycle)
        _BP_COLORS = [
            (0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
            (0, 0, 255), (255, 0, 0), (0, 255, 255), (200, 200, 200),
        ]
        bp_colors = {bp: _BP_COLORS[i % len(_BP_COLORS)]
                     for i, bp in enumerate(bp_list)}

        _tk_img_ref = {'img': None}  # hold a reference so Tk doesn't GC

        def _redraw():
            if not win.winfo_exists():
                return
            idx = max(0, min(n_frames - 1, int(frame_var.get())))
            frame_lbl.config(text=str(idx))
            cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                return
            h, w = frame.shape[:2]

            # Draw ROI per bodypart
            for bp in bp_list:
                x_col, y_col, p_col = bp_cols[bp]
                if x_col is None or y_col is None:
                    readouts[bp].config(text="(not in DLC)", foreground='crimson')
                    continue
                try:
                    rx = float(dlc[x_col].iloc[idx])
                    ry = float(dlc[y_col].iloc[idx])
                except Exception:
                    readouts[bp].config(text="(NaN pos)", foreground='crimson')
                    continue
                if _pd.isna(rx) or _pd.isna(ry):
                    readouts[bp].config(text="(NaN pos)", foreground='crimson')
                    continue
                cx, cy = int(rx) + crop_x, int(ry) + crop_y
                sz = int(size_vars[bp].get())
                x1 = max(0, cx - sz // 2)
                y1 = max(0, cy - sz // 2)
                x2 = min(w, cx + sz // 2)
                y2 = min(h, cy + sz // 2)
                color = bp_colors[bp]
                _cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                label = f"{bp} ({sz}px)"
                _cv2.putText(frame, label, (x1, max(12, y1 - 4)),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                             _cv2.LINE_AA)

                # Mean + area-above-threshold readout
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    gray = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY)
                    mean_v = float(gray.mean())
                    # Use a relative threshold (50% of global mean) for the
                    # "area" readout — purely informational
                    thr = int(gray.mean() * 1.1) if gray.mean() > 0 else 200
                    area_frac = float((gray > thr).mean())
                    prob = float(dlc[p_col].iloc[idx]) if p_col is not None else float('nan')
                    readouts[bp].config(
                        text=f"mean={mean_v:5.1f}  area>{thr}={area_frac*100:4.1f}%  prob={prob:.2f}",
                        foreground='black' if prob > 0.8 else '#b8860b')

            # Resize to canvas while keeping aspect
            cw = max(1, canvas.winfo_width())
            ch = max(1, canvas.winfo_height())
            if cw > 2 and ch > 2:
                scale = min(cw / w, ch / h)
                disp_w = max(1, int(w * scale))
                disp_h = max(1, int(h * scale))
                frame_rgb = _cv2.cvtColor(
                    _cv2.resize(frame, (disp_w, disp_h)),
                    _cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)

            pil = _Image.fromarray(frame_rgb)
            _tk_img_ref['img'] = _ImageTk.PhotoImage(pil)
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=_tk_img_ref['img'])

        # Wire events
        slider.configure(command=lambda _v: _redraw())
        canvas.bind('<Configure>', lambda e: _redraw())
        win.protocol("WM_DELETE_WINDOW", _close)

        # Initial draw after the window has a size
        win.after(100, _redraw)
    
    def scan_training_sessions(self):
        """Scan and display available training sessions"""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Folder", "Please select a project folder first.")
            return
        
        try:
            sessions = self.find_training_sessions()
            
            if not sessions:
                messagebox.showinfo("No Sessions Found", 
                    "No training sessions found in the selected folder.\n\n"
                    "Please ensure your folder structure matches:\n"
                    "Videos/ - Contains .h5 DLC files and videos\n"
                    "Targets/ - Contains .csv label files")
                return
            
            msg = f"Found {len(sessions)} session(s):\n\n"
            for s in sessions:
                msg += f"• {s['session_name']}\n"
                msg += f"  Pose: {os.path.basename(s['pose_path'])}\n"
                msg += f"  Video: {os.path.basename(s['video_path'])}\n"
                if s.get('target_path'):
                    msg += f"  Target: {os.path.basename(s['target_path'])}\n"
                else:
                    msg += f"  Target: Not found (will look for BORIS)\n"
                msg += "\n"
            
            messagebox.showinfo("Sessions Found", msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan sessions:\n{str(e)}")

    def _scan_and_populate_sessions(self, silent=False):
        """Scan sessions and populate the session selection treeview."""
        if not self.train_project_folder.get():
            if not silent:
                messagebox.showwarning("No Folder", "Please select a project folder first.")
            return
        try:
            sessions = self.find_training_sessions()
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to scan sessions:\n{str(e)}")
            return

        self._scanned_sessions = sessions
        self._session_checked.clear()

        # Clear treeview
        if self._session_tree is not None:
            for item in self._session_tree.get_children():
                self._session_tree.delete(item)

        if not sessions:
            if not silent:
                messagebox.showinfo("No Sessions", "No training sessions found.")
            self._update_session_count()
            return

        for s in sessions:
            name = s['session_name']
            has_labels_bool = bool(s.get('target_path'))
            has_labels = "Yes" if has_labels_bool else "No"
            video_name = os.path.basename(s.get('video_path', ''))
            bvar = tk.BooleanVar(value=has_labels_bool)
            self._session_checked[name] = bvar
            self._session_tree.insert("", "end", iid=name,
                                      values=("✓" if has_labels_bool else "", name, has_labels, video_name),
                                      tags=("no_labels",) if not has_labels_bool else ())

        labeled = sum(1 for s in sessions if s.get('target_path'))
        self.log_train(f"Scanned {len(sessions)} session(s) — {labeled} with labels selected by default.")
        self._update_session_count()
        # Auto-expand panel when sessions are loaded
        if not self._sess_expanded and self._scanned_sessions:
            self._toggle_session_panel()
        else:
            self._update_sess_toggle_label()   # refresh count in header even if already open

    def _on_session_tree_click(self, event):
        """Toggle checkbox when user clicks on a row in the session treeview."""
        tree = self._session_tree
        region = tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            return
        row_id = tree.identify_row(event.y)
        if not row_id or row_id not in self._session_checked:
            return
        bvar = self._session_checked[row_id]
        bvar.set(not bvar.get())
        vals = list(tree.item(row_id, "values"))
        vals[0] = "✓" if bvar.get() else ""
        tree.item(row_id, values=vals)
        self._update_session_count()
        self._update_sess_toggle_label()

    def _session_select_all(self):
        """Check all sessions in the session treeview."""
        for name, bvar in self._session_checked.items():
            bvar.set(True)
            vals = list(self._session_tree.item(name, "values"))
            vals[0] = "✓"
            self._session_tree.item(name, values=vals)
        self._update_session_count()
        self._update_sess_toggle_label()

    def _session_deselect_all(self):
        """Uncheck all sessions in the session treeview."""
        for name, bvar in self._session_checked.items():
            bvar.set(False)
            vals = list(self._session_tree.item(name, "values"))
            vals[0] = ""
            self._session_tree.item(name, values=vals)
        self._update_session_count()
        self._update_sess_toggle_label()

    def _session_select_labeled(self):
        """Check only sessions that have label files."""
        for s in self._scanned_sessions:
            name = s['session_name']
            has_labels = bool(s.get('target_path'))
            if name in self._session_checked:
                self._session_checked[name].set(has_labels)
                vals = list(self._session_tree.item(name, "values"))
                vals[0] = "✓" if has_labels else ""
                self._session_tree.item(name, values=vals)
        self._update_session_count()
        self._update_sess_toggle_label()

    def _toggle_session_panel(self):
        if self._sess_expanded:
            self._sess_inner_frame.pack_forget()
            self._sess_expanded = False
        else:
            self._sess_inner_frame.pack(fill='x', padx=5, pady=(4, 4))
            self._sess_expanded = True
        self._update_sess_toggle_label()

    def _update_sess_toggle_label(self):
        """Sync the toggle button text with current expand state and session count."""
        arrow = "▼" if self._sess_expanded else "▶"
        if self._scanned_sessions:
            total = len(self._scanned_sessions)
            selected = sum(1 for v in self._session_checked.values() if v.get())
            self._sess_toggle_btn.config(
                text=f"{arrow} Session Selection  ({selected} of {total} selected)")
        else:
            self._sess_toggle_btn.config(text=f"{arrow} Session Selection")

    def _update_session_count(self):
        """Update the session count label below the treeview."""
        if not hasattr(self, '_session_count_label') or self._session_count_label is None:
            return
        total = len(self._scanned_sessions)
        selected = sum(1 for v in self._session_checked.values() if v.get())
        labeled = sum(1 for s in self._scanned_sessions if s.get('target_path'))
        self._session_count_label.config(
            text=f"{selected} of {total} selected · {labeled} have labels"
        )

    def find_training_sessions(self) -> List[Dict]:
        """Find all training sessions in project folder using shared session discovery."""
        project_folder = self.train_project_folder.get()

        candidates = find_session_triplets(
            project_folder,
            video_ext=self.train_video_ext.get() if hasattr(self, 'train_video_ext') else '.mp4',
            prefer_filtered=True,
            require_labels=False,   # we handle the missing-labels dialog ourselves below
        )

        if not candidates:
            return []

        # ── Pass 1: separate labeled from unlabeled ──────────────────────────
        labeled_candidates   = []
        unlabeled_candidates = []
        seen_basenames = set()
        for s in candidates:
            bn = os.path.splitext(os.path.basename(s['video']))[0]
            if bn in seen_basenames:
                continue
            seen_basenames.add(bn)
            if s['labels'] is not None:
                labeled_candidates.append(s)
            else:
                unlabeled_candidates.append(s)

        # ── Pass 2: if any missing, show one checkbox dialog ─────────────────
        skipped_basenames = set()
        stopped_by_user   = False

        if unlabeled_candidates:
            # Build dialog
            dlg = tk.Toplevel(self.root)
            dlg.title("Labels Not Found")
            dlg.grab_set()
            dlg.resizable(False, False)
            _sw, _sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()

            # Header
            header_frm = ttk.Frame(dlg, padding=(18, 14, 18, 4))
            header_frm.pack(fill='x')
            ttk.Label(
                header_frm,
                text=f"{len(unlabeled_candidates)} video(s) are missing a labels file.",
                font=('Arial', 10, 'bold'),
            ).pack(anchor='w')
            ttk.Label(
                header_frm,
                text="Check the videos you want to skip and continue.\n"
                     "Uncheck any video to stop and label it first.",
                font=('Arial', 9),
                foreground='gray40',
            ).pack(anchor='w', pady=(4, 0))

            ttk.Separator(dlg, orient='horizontal').pack(fill='x', padx=18, pady=6)

            # Scrollable checkbox list
            list_frm = ttk.Frame(dlg, padding=(18, 0, 18, 0))
            list_frm.pack(fill='both', expand=True)

            canvas   = tk.Canvas(list_frm, highlightthickness=0, bd=0)
            scrollbar = ttk.Scrollbar(list_frm, orient='vertical', command=canvas.yview)
            inner    = ttk.Frame(canvas)
            inner.bind('<Configure>',
                       lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
            canvas.create_window((0, 0), window=inner, anchor='nw')
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

            check_vars = {}
            for s in unlabeled_candidates:
                bn  = os.path.splitext(os.path.basename(s['video']))[0]
                pd_ = s['project_dir']
                vd_ = s['video_dir']
                var = tk.BooleanVar(value=True)   # default: skip
                check_vars[bn] = var

                row = ttk.Frame(inner, padding=(0, 4, 0, 4))
                row.pack(fill='x', anchor='w')

                ttk.Checkbutton(row, variable=var, text=bn,
                                style='TCheckbutton').pack(anchor='w')
                searched = [
                    os.path.join(pd_, 'behavior_labels', f'{bn}_labels.csv'),
                    os.path.join(vd_,                    f'{bn}_labels.csv'),
                    os.path.join(pd_, 'labels',          f'{bn}_labels.csv'),
                ]
                for loc in searched:
                    ttk.Label(row, text=f"    • {loc}",
                              font=('Arial', 8), foreground='gray50').pack(anchor='w')

                ttk.Separator(inner, orient='horizontal').pack(fill='x', pady=2)

            # Cap height at 60 % of screen
            dlg.update_idletasks()
            max_h = int(_sh * 0.60)
            needed_h = min(inner.winfo_reqheight() + 160, max_h)
            canvas.configure(height=min(inner.winfo_reqheight(), needed_h - 160))

            ttk.Separator(dlg, orient='horizontal').pack(fill='x', padx=18, pady=6)

            # Buttons
            btn_frm = ttk.Frame(dlg, padding=(18, 0, 18, 14))
            btn_frm.pack(fill='x')

            _result = {'action': 'stop'}

            def _on_continue():
                _result['action'] = 'continue'
                dlg.destroy()

            def _on_stop():
                _result['action'] = 'stop'
                dlg.destroy()

            ttk.Button(btn_frm, text="Skip Selected & Continue",
                       command=_on_continue, style='Accent.TButton').pack(side='left', padx=(0, 8))
            ttk.Button(btn_frm, text="Cancel (Stop)",
                       command=_on_stop).pack(side='left')

            # Size and centre
            dlg.update_idletasks()
            w = max(dlg.winfo_reqwidth(), 520)
            h = dlg.winfo_reqheight()
            dlg.geometry(f"{w}x{h}+{(_sw - w) // 2}+{(_sh - h) // 2}")
            self.root.wait_window(dlg)

            if _result['action'] == 'stop':
                self.log_train("⚠️  Stopped by user — please add labels and retry.")
                return []

            # Collect which videos user chose to skip
            for bn, var in check_vars.items():
                if var.get():
                    skipped_basenames.add(bn)
                else:
                    # User unchecked → treat as stop
                    self.log_train(f"⚠️  Stopped: {bn} must be labeled first.")
                    return []

        # ── Build session list ───────────────────────────────────────────────
        sessions = []
        for s in labeled_candidates:
            sessions.append({
                'session_name': s['session_name'],
                'pose_path':    s['dlc'],
                'video_path':   s['video'],
                'target_path':  s['labels'],
            })

        for bn in skipped_basenames:
            self.log_train(f"⚠️  Skipping (no labels): {bn}")

        if skipped_basenames:
            self.log_train(f"\n📋 Scan Summary:")
            self.log_train(f"   ✓ Found {len(sessions)} videos with labels")
            self.log_train(f"   ⊗ Skipped {len(skipped_basenames)} videos without labels:")
            for bn in sorted(skipped_basenames):
                self.log_train(f"      - {bn}")
        else:
            self.log_train(f"\n✓ Found {len(sessions)} training sessions (all with labels)")

        return sessions
    
    def auto_detect_behavior_names(self):
        """Auto-detect available behavior names from target CSV files"""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Project", "Please select a project folder first.")
            return
        
        try:
            # Find training sessions
            sessions = self.find_training_sessions()
            if not sessions:
                messagebox.showwarning("No Sessions", "No training sessions found.\n\n"
                                     "Make sure your project has the structure:\n"
                                     "  project_folder/\n"
                                     "    videos/\n"
                                     "    targets/")
                return
            
            # Collect all behavior columns from all target files
            all_behaviors = set()
            for session in sessions:
                if not session['target_path'] or not os.path.isfile(session['target_path']):
                    continue
                
                try:
                    df = pd.read_csv(session['target_path'])
                    # Exclude non-behavior columns
                    excluded = {'Frame', 'frame', 'Time', 'time', 'Unnamed: 0', 'index'}
                    behaviors = [col for col in df.columns if col not in excluded]
                    all_behaviors.update(behaviors)
                except Exception as e:
                    print(f"Error reading {session['target_path']}: {e}")
                    continue
            
            if not all_behaviors:
                messagebox.showinfo("No Behaviors Found", 
                                  "No behavior columns found in target CSV files.\n\n"
                                  "Target files should have columns like:\n"
                                  "  Frame, Scratching, Grooming, etc.")
                return
            
            # Sort behaviors alphabetically
            behaviors_list = sorted(all_behaviors)
            
            # Create selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Behavior")
            dialog.geometry("450x550")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Info label
            ttk.Label(dialog, text=f"Found {len(behaviors_list)} behavior(s) in target files:",
                     font=('Arial', 10, 'bold')).pack(padx=10, pady=10)
            
            # Listbox with scrollbar
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side='right', fill='y')
            
            listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                font=('Arial', 10), height=15)
            listbox.pack(side='left', fill='both', expand=True)
            scrollbar.config(command=listbox.yview)
            
            # Add behaviors to listbox
            for behavior in behaviors_list:
                listbox.insert(tk.END, behavior)
            
            # Select first item by default
            if behaviors_list:
                listbox.selection_set(0)
            
            # Session info
            info_text = f"\nFound in {len(sessions)} session(s)"
            ttk.Label(dialog, text=info_text, foreground='gray').pack(pady=5)
            
            # Buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)
            
            def on_select():
                selection = listbox.curselection()
                if selection:
                    selected = listbox.get(selection[0])
                    self.train_behavior_name.set(selected)
                    dialog.destroy()
                    messagebox.showinfo("Selected", f"Behavior set to: {selected}")
            
            def on_cancel():
                dialog.destroy()
            
            ttk.Button(btn_frame, text="Select", command=on_select, width=15).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=15).pack(side='left', padx=5)
            
            # Double-click to select
            listbox.bind('<Double-Button-1>', lambda e: on_select())
            
            # Center dialog
            dialog.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect behaviors:\n{str(e)}")
    
    def auto_suggest_bout_params(self):
        """Automatically suggest bout parameters based on video FPS and behavior analysis"""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Project", "Please select a project folder first.")
            return
        
        try:
            # Find a video to get FPS
            sessions = self.find_training_sessions()
            if not sessions:
                messagebox.showwarning("No Sessions", "No training sessions found.")
                return
            
            # Get FPS from first video
            import cv2
            cap = cv2.VideoCapture(sessions[0]['video_path'])
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps <= 0:
                fps = 30  # Default fallback
            
            # Analyze behavior labels to get statistics
            behavior_name = self.train_behavior_name.get()
            if not behavior_name:
                messagebox.showwarning("No Behavior", "Please enter a behavior name first.")
                return
            
            all_bout_durations = []
            all_gap_durations = []
            
            for session in sessions[:3]:  # Analyze first 3 sessions for speed
                if not session['target_path'] or not os.path.isfile(session['target_path']):
                    continue
                
                try:
                    labels = pd.read_csv(session['target_path'])
                    if behavior_name not in labels.columns:
                        continue
                    
                    y = labels[behavior_name].values
                    
                    # Find bouts
                    in_bout = False
                    bout_start = 0
                    bouts = []
                    
                    for i, val in enumerate(y):
                        if val == 1 and not in_bout:
                            bout_start = i
                            in_bout = True
                        elif val == 0 and in_bout:
                            bouts.append((bout_start, i - 1))
                            in_bout = False
                    
                    if in_bout:
                        bouts.append((bout_start, len(y) - 1))
                    
                    # Calculate bout durations
                    for start, end in bouts:
                        duration = end - start + 1
                        all_bout_durations.append(duration)
                    
                    # Calculate gaps between bouts
                    for i in range(len(bouts) - 1):
                        gap = bouts[i+1][0] - bouts[i][1] - 1
                        if gap > 0:
                            all_gap_durations.append(gap)
                
                except Exception as e:
                    continue
            
            if not all_bout_durations:
                # No labeled data found, use behavior-based heuristics
                suggested_params = self._suggest_by_behavior_type(behavior_name, fps)
            else:
                # Use data-driven suggestions
                suggested_params = self._suggest_by_data_analysis(
                    all_bout_durations, all_gap_durations, fps
                )
            
            # Create suggestion dialog
            self._show_suggestion_dialog(suggested_params, fps)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not analyze data:\n{str(e)}")
    
    def _suggest_by_behavior_type(self, behavior_name, fps):
        """Suggest parameters based on common behavior patterns"""
        behavior_lower = behavior_name.lower()
        
        # Common behavior patterns (in seconds)
        behavior_patterns = {
            'scratch': {'min_duration': 0.1, 'typical_gap': 0.15},
            'groom': {'min_duration': 0.5, 'typical_gap': 0.3},
            'rear': {'min_duration': 0.3, 'typical_gap': 0.4},
            'freeze': {'min_duration': 1.0, 'typical_gap': 0.5},
            'lick': {'min_duration': 0.1, 'typical_gap': 0.2},
            'lift': {'min_duration': 0.15, 'typical_gap': 0.2},
            'shake': {'min_duration': 0.1, 'typical_gap': 0.15},
            'jump': {'min_duration': 0.2, 'typical_gap': 0.3},
            'walk': {'min_duration': 0.5, 'typical_gap': 0.3},
            'run': {'min_duration': 0.3, 'typical_gap': 0.2},
        }
        
        # Find matching pattern
        pattern = None
        for key, value in behavior_patterns.items():
            if key in behavior_lower:
                pattern = value
                break
        
        # Default if no match
        if pattern is None:
            pattern = {'min_duration': 0.2, 'typical_gap': 0.2}
        
        min_bout = max(1, int(pattern['min_duration'] * fps))
        max_gap = max(1, int(pattern['typical_gap'] * fps))
        min_after_bout = max(1, int(0.05 * fps))  # 50ms minimum
        
        return {
            'min_bout': min_bout,
            'min_after_bout': min_after_bout,
            'max_gap': max_gap,
            'method': 'behavior_heuristic',
            'fps': fps
        }
    
    def _suggest_by_data_analysis(self, bout_durations, gap_durations, fps):
        """Suggest parameters based on actual labeled data"""
        # Min bout: Use 10th percentile of bout durations
        # This ensures we don't filter out genuinely short behaviors
        min_bout = max(1, int(np.percentile(bout_durations, 10)))
        
        # Max gap: Use median of gap durations
        # This bridges typical gaps without merging separate bouts
        if gap_durations:
            max_gap = max(1, int(np.percentile(gap_durations, 50)))
        else:
            max_gap = max(1, int(0.2 * fps))  # Default 200ms
        
        # Min after bout: 5-10% of min bout duration
        min_after_bout = max(1, min_bout // 5)
        
        return {
            'min_bout': min_bout,
            'min_after_bout': min_after_bout,
            'max_gap': max_gap,
            'method': 'data_driven',
            'fps': fps,
            'stats': {
                'n_bouts': len(bout_durations),
                'mean_bout': np.mean(bout_durations),
                'median_bout': np.median(bout_durations),
                'mean_gap': np.mean(gap_durations) if gap_durations else 0,
            }
        }
    
    def _show_suggestion_dialog(self, params, fps):
        """Show dialog with suggested parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title("🤖 Auto-Suggested Bout Parameters")
        _sw = dialog.winfo_screenwidth()
        _sh = dialog.winfo_screenheight()
        _dw = min(650, int(_sw * 0.45))
        _dh = min(600, int(_sh * 0.55))
        dialog.geometry(f"{_dw}x{_dh}+{(_sw-_dw)//2}+{(_sh-_dh)//2}")
        dialog.resizable(True, True)
        
        # Title
        title = ttk.Label(dialog, text="Suggested Bout Parameters", 
                         font=('Arial', 12, 'bold'))
        title.pack(pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(dialog, text="Analysis Info", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(info_frame, text=f"Video FPS: {fps:.1f}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Method: {params['method'].replace('_', ' ').title()}").pack(anchor='w')
        
        if params['method'] == 'data_driven' and 'stats' in params:
            stats = params['stats']
            ttk.Label(info_frame, text=f"Analyzed {stats['n_bouts']} bouts").pack(anchor='w')
            ttk.Label(info_frame, text=f"Mean bout: {stats['mean_bout']:.1f} frames ({stats['mean_bout']/fps:.2f}s)").pack(anchor='w')
            ttk.Label(info_frame, text=f"Median bout: {stats['median_bout']:.1f} frames ({stats['median_bout']/fps:.2f}s)").pack(anchor='w')
            if stats['mean_gap'] > 0:
                ttk.Label(info_frame, text=f"Mean gap: {stats['mean_gap']:.1f} frames ({stats['mean_gap']/fps:.2f}s)").pack(anchor='w')
        
        # Suggestions frame
        suggest_frame = ttk.LabelFrame(dialog, text="Suggested Values", padding=10)
        suggest_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Min Bout
        row = 0
        ttk.Label(suggest_frame, text="Min Bout:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky='w', pady=5)
        ttk.Label(suggest_frame, text=f"{params['min_bout']} frames").grid(
            row=row, column=1, sticky='w', padx=10)
        ttk.Label(suggest_frame, text=f"({params['min_bout']/fps:.2f} seconds)", 
                 foreground='gray').grid(row=row, column=2, sticky='w')
        
        row += 1
        explanation = tk.Text(suggest_frame, height=2, wrap=tk.WORD, 
                            relief=tk.FLAT, bg=self.theme.colors['frame_bg'])
        explanation.insert('1.0', "Minimum consecutive frames needed to count as a valid behavior bout. "
                                 "Shorter events will be filtered out as noise.")
        explanation.config(state='disabled')
        explanation.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        # Min After Bout
        row += 1
        ttk.Label(suggest_frame, text="Min After Bout:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky='w', pady=5)
        ttk.Label(suggest_frame, text=f"{params['min_after_bout']} frames").grid(
            row=row, column=1, sticky='w', padx=10)
        ttk.Label(suggest_frame, text=f"({params['min_after_bout']/fps:.2f} seconds)", 
                 foreground='gray').grid(row=row, column=2, sticky='w')
        
        row += 1
        explanation = tk.Text(suggest_frame, height=2, wrap=tk.WORD, 
                            relief=tk.FLAT, bg=self.theme.colors['frame_bg'])
        explanation.insert('1.0', "Minimum frames required after a bout ends before another can start. "
                                 "Prevents rapid flickering between states.")
        explanation.config(state='disabled')
        explanation.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        # Max Gap
        row += 1
        ttk.Label(suggest_frame, text="Max Gap:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky='w', pady=5)
        ttk.Label(suggest_frame, text=f"{params['max_gap']} frames").grid(
            row=row, column=1, sticky='w', padx=10)
        ttk.Label(suggest_frame, text=f"({params['max_gap']/fps:.2f} seconds)", 
                 foreground='gray').grid(row=row, column=2, sticky='w')
        
        row += 1
        explanation = tk.Text(suggest_frame, height=2, wrap=tk.WORD, 
                            relief=tk.FLAT, bg=self.theme.colors['frame_bg'])
        explanation.insert('1.0', "Maximum gap that will be bridged to merge nearby bouts. "
                                 "Helps smooth over brief interruptions in continuous behavior.")
        explanation.config(state='disabled')
        explanation.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        def apply_suggestions():
            self.train_min_bout.set(params['min_bout'])
            self.train_min_after_bout.set(params['min_after_bout'])
            self.train_max_gap.set(params['max_gap'])
            dialog.destroy()
            messagebox.showinfo("Applied", "Suggested parameters have been applied!")
        
        ttk.Button(button_frame, text="✓ Apply These Values", 
                  command=apply_suggestions, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side='left', padx=5)
    
    def get_session_base(self, filename: str) -> str:
        """Extract base session name from DLC filename"""
        # Remove DLC and everything after it
        if 'DLC' in filename:
            return filename.split('DLC')[0]
        else:
            # Fallback: remove extension
            return os.path.splitext(filename)[0]
    
    def start_training(self):
        """Start the classifier training process"""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Folder", "Please select a project folder first.")
            return
        
        if not self.train_behavior_name.get().strip():
            messagebox.showwarning("No Behavior", "Please enter a behavior name.")
            return

        # Validate behavior name (used in filenames)
        _bname = self.train_behavior_name.get().strip()
        if any(c in _bname for c in r'/\:*?"<>|'):
            messagebox.showwarning("Invalid Name",
                                   "Behavior name cannot contain: / \\ : * ? \" < > |")
            return

        # Show training visualization window
        if self.train_viz_window is None or not self.train_viz_window.window.winfo_exists():
            self.train_viz_window = TrainingVisualizationWindow(self.root, self.theme)

        # Reset cancel flags, disable START button, enable Cancel
        self._training_cancel_flag.clear()
        self._feature_cancel_flag.clear()
        self._train_start_btn.config(state='disabled')
        self._train_all_btn.config(state='disabled')
        self._train_cancel_btn.config(state='normal')

        # Launch training in a background thread
        def _training_done():
            self._safe_after(lambda: self._train_start_btn.config(state='normal'))
            self._safe_after(lambda: self._train_all_btn.config(state='normal'))
            self._safe_after(lambda: self._train_cancel_btn.config(state='disabled'))

        def _run():
            try:
                self._real_training()
            finally:
                _training_done()

        threading.Thread(target=_run, daemon=True).start()

    def _cancel_training(self):
        """Signal the training thread to stop."""
        self._training_cancel_flag.set()
        self._feature_cancel_flag.set()
        self.log_train("\nCancellation requested — stopping after current step...")

    def start_training_all_behaviors(self):
        """Train one classifier per behavior column found in the selected sessions' label CSVs."""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Project", "Please select a project folder first.")
            return

        # Collect sessions the same way start_training() does
        if self._scanned_sessions:
            sessions = [s for s in self._scanned_sessions
                        if self._session_checked.get(
                            s['session_name'], tk.BooleanVar(value=False)).get()]
        else:
            sessions = self.find_training_sessions()

        if not sessions:
            messagebox.showwarning("No Sessions", "No sessions selected.")
            return

        # Discover all unique behavior columns across selected sessions
        excluded = {'Frame', 'frame', 'Time', 'time', 'Unnamed: 0', 'index'}
        all_behaviors = set()
        for s in sessions:
            if s.get('target_path') and os.path.isfile(s['target_path']):
                try:
                    df_cols = pd.read_csv(s['target_path'], nrows=0).columns.tolist()
                    for col in df_cols:
                        if col not in excluded:
                            all_behaviors.add(col)
                except Exception:
                    pass

        if not all_behaviors:
            messagebox.showwarning("No Behaviors",
                                   "No behavior columns found in the selected sessions' label CSVs.")
            return

        behaviors_list = sorted(all_behaviors)

        # Pre-flight: skip behaviors that have zero positive examples across all sessions
        skipped_no_pos = []
        filtered_behaviors = []
        for bname in behaviors_list:
            has_pos = False
            for s in sessions:
                if s.get('target_path') and os.path.isfile(s['target_path']):
                    try:
                        _col_df = pd.read_csv(s['target_path'], usecols=[bname])
                        if _col_df[bname].sum() > 0:
                            has_pos = True
                            break
                    except Exception:
                        pass
            if has_pos:
                filtered_behaviors.append(bname)
            else:
                skipped_no_pos.append(bname)

        if skipped_no_pos:
            skip_msg = (f"{len(skipped_no_pos)} behavior(s) have no positive examples and will be skipped:\n"
                        + "\n".join(f"  \u2022 {b}" for b in skipped_no_pos))
            messagebox.showwarning("Skipping Zero-Label Behaviors", skip_msg)

        if not filtered_behaviors:
            messagebox.showwarning("No Behaviors",
                                   "No behaviors have positive examples — nothing to train.")
            return

        behaviors_list = filtered_behaviors

        confirm_msg = (f"Train classifiers for {len(behaviors_list)} behavior(s)?\n\n"
                       + "\n".join(f"  \u2022 {b}" for b in behaviors_list)
                       + "\n\nThis will run the full training pipeline for each behavior in sequence.")
        if not messagebox.askyesno("Train All Behaviors", confirm_msg):
            return

        # Show training viz window
        if self.train_viz_window is None or not self.train_viz_window.window.winfo_exists():
            self.train_viz_window = TrainingVisualizationWindow(self.root, self.theme)

        self._training_cancel_flag.clear()
        self._feature_cancel_flag.clear()
        self._train_start_btn.config(state='disabled')
        self._train_all_btn.config(state='disabled')
        self._train_cancel_btn.config(state='normal')

        original_behavior = self.train_behavior_name.get()

        def _all_done():
            self._safe_after(lambda: self._train_start_btn.config(state='normal'))
            self._safe_after(lambda: self._train_all_btn.config(state='normal'))
            self._safe_after(lambda: self._train_cancel_btn.config(state='disabled'))
            self._safe_after(lambda: self.train_behavior_name.set(original_behavior))

        def _run():
            summary_rows = []
            for i, bname in enumerate(behaviors_list):
                if self._training_cancel_flag.is_set():
                    self.log_train(f"\nCancelled — skipping remaining behaviors.")
                    break

                self.log_train(f"\n{'#'*60}")
                self.log_train(f"# [{i+1}/{len(behaviors_list)}] Training: {bname}")
                self.log_train(f"{'#'*60}")

                # Set behavior name so _real_training() picks it up
                self.train_behavior_name.set(bname)
                # Reset cancel flags for each behavior (don't carry over feature errors)
                self._feature_cancel_flag.clear()

                try:
                    self._real_training()
                    r = getattr(self, 'last_training_results', {})
                    if r.get('behavior_name') == bname:
                        summary_rows.append({
                            'behavior': bname,
                            'cv_f1':    r['mean_f1'],
                            'cv_std':   r['std_f1'],
                            'oof_f1':   r.get('oof_best_f1'),
                            'status':   'OK',
                        })
                    else:
                        summary_rows.append({'behavior': bname, 'status': 'ERROR (no result)'})
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    summary_rows.append({'behavior': bname, 'status': f'ERROR: {exc}'})

            # Print summary table
            self.log_train(f"\n{'='*60}")
            self.log_train("TRAIN ALL BEHAVIORS — SUMMARY")
            self.log_train(f"{'='*60}")
            self.log_train(f"{'Behavior':<28}  {'CV F1':>12}  {'OOF F1':>8}  Status")
            self.log_train("-" * 60)
            for row in summary_rows:
                if row['status'] == 'OK':
                    cv_str  = f"{row['cv_f1']:.3f}\u00b1{row['cv_std']:.3f}"
                    oof_str = f"{row['oof_f1']:.3f}" if row['oof_f1'] is not None else "—"
                    self.log_train(f"{row['behavior']:<28}  {cv_str:>12}  {oof_str:>8}  OK")
                else:
                    self.log_train(f"{row['behavior']:<28}  {'—':>12}  {'—':>8}  {row['status']}")
            self.log_train(f"{'='*60}")
            n_ok = sum(1 for r in summary_rows if r['status'] == 'OK')
            self.log_train(f"{n_ok}/{len(summary_rows)} classifiers trained successfully.")

            _all_done()

        threading.Thread(target=_run, daemon=True).start()

    def _run_cv_loop(self, X, y, session_ids, sessions, unique_sessions, kf,
                     actual_folds, use_spw, use_early_stop, early_stop_rounds,
                     tree_method, log=True, train_fraction=1.0):
        """Run session-level K-fold CV and return OOF probabilities + fold metrics.

        Parameters
        ----------
        train_fraction : float
            Fraction of training frames to use per fold (1.0 = all).
            Used by the learning curve diagnostic to test subsets.

        Returns
        -------
        dict with keys: oof_proba, fold_f1_scores, fold_precisions,
                        fold_recalls, fold_best_iters
              or None if training was cancelled.
        """
        fold_f1_scores   = []
        fold_precisions  = []
        fold_recalls     = []
        fold_best_iters  = []
        fold_models      = []

        oof_proba = np.full(len(y), np.nan)

        for fold, (train_sess_idx, val_sess_idx) in enumerate(
                kf.split(unique_sessions), 1):

            if self._training_cancel_flag.is_set():
                if log:
                    self.log_train("\nTraining cancelled by user.")
                return None

            fold_start = time.time()
            if log:
                self.log_train(f"\n=== Fold {fold}/{actual_folds} ===")

            train_sess = unique_sessions[train_sess_idx]
            val_sess   = unique_sessions[val_sess_idx]

            if log:
                _val_names = [sessions[si].get('session_name', f'session_{si}') for si in val_sess]
                _train_names = [sessions[si].get('session_name', f'session_{si}') for si in train_sess]
                self.log_train(f"  Hold out: {', '.join(_val_names)}")
                self.log_train(f"  Train on: {', '.join(_train_names)}")

            train_mask = np.isin(session_ids, train_sess)
            val_mask   = np.isin(session_ids, val_sess)

            X_train = X[train_mask] if isinstance(X, np.ndarray) else X.loc[train_mask]
            y_train = y[train_mask]
            X_val   = X[val_mask] if isinstance(X, np.ndarray) else X.loc[val_mask]
            y_val   = y[val_mask]

            # Subsample training data for learning curve diagnostic
            if train_fraction < 1.0 and len(X_train) > 10:
                from sklearn.model_selection import train_test_split
                n_keep = max(10, int(len(X_train) * train_fraction))
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, train_size=n_keep,
                    stratify=y_train if np.sum(y_train) >= 2 else None,
                    random_state=42)

            if log:
                self.log_train(
                    f"  Train: {len(X_train)} frames, {np.sum(y_train)} positive")
                self.log_train(
                    f"  Val:   {len(X_val)} frames, {np.sum(y_val)} positive")

            # ── Class imbalance handling ───────────────────────────
            spw = 1.0
            if use_spw and np.sum(y_train) > 0:
                spw = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
                if log:
                    self.log_train(f"  scale_pos_weight = {spw:.2f}")

            # Legacy downsampling fallback (off by default)
            if (self.train_use_balancing.get()
                    and np.mean(y_train) < self.train_imbalance_thresh.get()
                    and not use_spw):
                if log:
                    self.log_train("  Applying downsampling...")
                X_train, y_train = self.balance_data(
                    X_train.values if hasattr(X_train, 'values') else X_train, y_train)
                X_train = pd.DataFrame(X_train, columns=X.columns)
                if log:
                    self.log_train(
                        f"  After downsampling: {len(X_train)} frames")

            # ── Build and fit fold model ───────────────────────────
            fold_model = xgb.XGBClassifier(
                n_estimators=self.train_n_estimators.get(),
                max_depth=self.train_max_depth.get(),
                learning_rate=self.train_learning_rate.get(),
                subsample=self.train_subsample.get(),
                colsample_bytree=self.train_colsample.get(),
                scale_pos_weight=spw,
                tree_method=tree_method,
                n_jobs=-1,
                objective='binary:logistic',
                random_state=42,
                eval_metric='aucpr',
            )

            if use_early_stop:
                fold_model.set_params(
                    early_stopping_rounds=early_stop_rounds)
                fold_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
                best_iter = getattr(fold_model, 'best_iteration',
                                    self.train_n_estimators.get() - 1)
                fold_best_iters.append(best_iter + 1)
                if log:
                    self.log_train(f"  Early stopping at tree {best_iter + 1}")
            else:
                fold_model.fit(X_train, y_train)

            # ── OOF probabilities for this fold ───────────────────
            val_proba = fold_model.predict_proba(X_val)[:, 1]
            oof_proba[val_mask] = val_proba

            val_pred = (val_proba >= 0.5).astype(int)
            f1   = f1_score(y_val, val_pred, zero_division=0)
            prec = precision_score(y_val, val_pred, zero_division=0)
            rec  = recall_score(y_val, val_pred, zero_division=0)

            fold_f1_scores.append(f1)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_models.append(fold_model)

            elapsed = time.time() - fold_start
            if log:
                self.log_train(
                    f"  F1: {f1:.3f}, Precision: {prec:.3f}, "
                    f"Recall: {rec:.3f}  ({elapsed:.1f}s)")

            if log and self.train_viz_window and \
                    self.train_viz_window.window.winfo_exists():
                self.train_viz_window.add_fold_result(
                    fold, f1, prec, rec, elapsed)

        return {
            'oof_proba': oof_proba,
            'fold_f1_scores': fold_f1_scores,
            'fold_precisions': fold_precisions,
            'fold_recalls': fold_recalls,
            'fold_best_iters': fold_best_iters,
            'fold_models': fold_models,
        }

    def _optuna_tune(self, X, y, session_ids, use_spw, tree_method, early_stop_rounds=30):
        """Run Optuna hyperparameter search using session-level CV."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        unique_sessions = np.unique(session_ids)
        n_folds = min(self.train_n_folds.get(), len(unique_sessions))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        n_trials = self.train_optuna_trials.get()

        def objective(trial):
            if self._training_cancel_flag.is_set():
                raise optuna.TrialPruned("Training cancelled by user")
            params = {
                'max_depth':        trial.suggest_int('max_depth', 3, 8),
                'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
                'subsample':        trial.suggest_float('subsample', 0.5, 0.95),
                'n_estimators':     trial.suggest_int('n_estimators', 200, 2000, step=100),
            }
            fold_f1s = []
            for train_si, val_si in kf.split(unique_sessions):
                train_sess = unique_sessions[train_si]
                val_sess = unique_sessions[val_si]
                train_mask = np.isin(session_ids, train_sess)
                val_mask = np.isin(session_ids, val_sess)
                X_tr, y_tr = X[train_mask], y[train_mask]
                X_va, y_va = X[val_mask], y[val_mask]
                spw = ((len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)) if use_spw else 1.0
                m = xgb.XGBClassifier(
                    **params, scale_pos_weight=spw, tree_method=tree_method,
                    n_jobs=-1,
                    objective='binary:logistic', random_state=42, eval_metric='aucpr',
                    early_stopping_rounds=early_stop_rounds)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                p = m.predict_proba(X_va)[:, 1]
                from sklearn.metrics import average_precision_score
                fold_f1s.append(average_precision_score(y_va, p))
            return np.mean(fold_f1s)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[lambda study, trial: self.log_train(
                           f"  Trial {trial.number+1}/{n_trials}: AP={trial.value:.4f} "
                           f"(best={study.best_value:.4f})")])
        return study.best_params, float(study.best_value)

    def _real_training(self):
        """ACTUAL classifier training implementation"""
        try:
            _pipeline_start = time.time()
            self.log_train("=" * 60)
            self.log_train("PixelPaws Classifier Training")
            self.log_train("=" * 60)
            
            # Get configuration
            project_folder = self.train_project_folder.get()
            behavior_name = self.train_behavior_name.get()
            use_spw = self.train_use_scale_pos_weight.get()
            use_early_stop = self.train_use_early_stopping.get()
            early_stop_rounds = self.train_early_stopping_rounds.get()
            
            self.log_train(f"\nProject:  {project_folder}")
            self.log_train(f"Behavior: {behavior_name}\n")
            
            # Find training sessions — honor selection if a scan was done;
            # never fall back to a full folder scan when the user has made
            # an explicit choice (that would silently include excluded sessions).
            if self._scanned_sessions:
                sessions = [s for s in self._scanned_sessions
                            if self._session_checked.get(s['session_name'], tk.BooleanVar(value=False)).get()]
                if not sessions:
                    raise ValueError(
                        "No sessions selected — check at least one session in the Session Selection panel")
            else:
                sessions = self.find_training_sessions()
                if not sessions:
                    raise ValueError("No training sessions found")
            
            self.log_train(f"Found {len(sessions)} session(s):")
            for s in sessions:
                self.log_train(f"  • {s['session_name']}")
            
            # Build feature config
            # NOTE: filter empty strings (if x.strip()) so a trailing comma in
            # the UI doesn't change the hash versus the Feature Extraction tool.
            cfg = {
                'bp_include_list': None,
                'bp_pixbrt_list': [x.strip() for x in self.train_bp_pixbrt.get().split(',') if x.strip()],
                'square_size': [int(x.strip()) for x in self.train_square_sizes.get().split(',') if x.strip()],
                'pix_threshold': self.train_pix_threshold.get(),
                'include_optical_flow': self.train_include_optical_flow.get(),
                'bp_optflow_list': [x.strip() for x in self.train_bp_optflow.get().split(',') if x.strip()]
                    if self.train_include_optical_flow.get() else [],
                'trim_to_last_positive': self.train_trim_to_last_positive.get(),
            }
            
            # Setup feature caching
            feature_cache_root = os.path.join(project_folder, 'features')
            os.makedirs(feature_cache_root, exist_ok=True)

            # ── Cache version audit ────────────────────────────────────
            # Scan each session's cache (if present) and report pose/brightness
            # version status before extraction.  Caches that are out of date
            # auto-upgrade on load via _load_features_for_prediction; this is
            # visibility, not a gating check.
            try:
                from feature_cache import FeatureCacheManager as _FCM
                _current_pose_v = _FCM.check_feature_versions.__globals__['POSE_FEATURE_VERSION']
                self.log_train("\n" + "=" * 60)
                self.log_train("CACHE VERSION AUDIT")
                self.log_train("=" * 60)
                self.log_train(
                    f"Current feature versions: pose v{_current_pose_v}, "
                    f"brightness v{_FCM.check_feature_versions.__globals__['BRIGHTNESS_FEATURE_VERSION']}")
                _up_to_date = 0
                _will_upgrade = 0
                _no_cache = 0
                for s in sessions:
                    _sname = s.get('session_name', '?')
                    # Look up the session's expected cache path via find_cache
                    _cfg_hash = self._feature_hash_key({**cfg, 'bp_include_list': None})
                    _video_dir = os.path.dirname(s.get('video_path', '')) or feature_cache_root
                    _found = _FCM.find_cache(_sname, _cfg_hash, feature_cache_root,
                                              _video_dir, project_root=feature_cache_root)
                    if not _found:
                        self.log_train(f"  {_sname}: no cache (will extract on first pass)")
                        _no_cache += 1
                        continue
                    _v = _FCM.check_feature_versions(_found)
                    _tag_src = f" [{_v['source']}]" if _v['source'] != 'sidecar' else ''
                    _pose = _v['pose_version']
                    _brt  = _v['brightness_version']
                    _pose_ok = _v['pose_up_to_date']
                    _brt_ok  = _v['brightness_up_to_date']
                    if _pose_ok and _brt_ok:
                        self.log_train(f"  {_sname}: pose v{_pose} ✓  brightness v{_brt} ✓{_tag_src}")
                        _up_to_date += 1
                    else:
                        _msg_parts = []
                        if not _pose_ok:
                            _msg_parts.append(
                                f"pose v{_pose} → v{_v['current_pose_version']} (auto-upgrade)")
                        if not _brt_ok:
                            _msg_parts.append(
                                f"brightness v{_brt} → v{_v['current_brightness_version']} (video re-read)")
                        self.log_train(f"  {_sname}: " + ', '.join(_msg_parts) + _tag_src)
                        _will_upgrade += 1
                self.log_train(
                    f"{_up_to_date}/{len(sessions)} up-to-date, "
                    f"{_will_upgrade} will auto-upgrade, "
                    f"{_no_cache} will extract from scratch.")
            except Exception as _audit_err:
                self.log_train(f"  (Cache version audit skipped: {_audit_err})")

            # ── Feature extraction ─────────────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("FEATURE EXTRACTION")
            self.log_train("=" * 60)
            
            all_X = []
            all_y = []
            session_ids = []
            session_cache_paths = {}
            included_sessions = []

            skipped = 0
            for i, session in enumerate(sessions):
                if self._training_cancel_flag.is_set():
                    self.log_train("\nTraining cancelled by user.")
                    return
                try:
                    X_s, y_s, cache_path_s = self.extract_features_for_session(
                        session, cfg, feature_cache_root, behavior_name)
                except KeyError:
                    self.log_train(
                        f"  Session {i+1}/{len(sessions)}: skipped "
                        f"('{behavior_name}' column not in labels)")
                    skipped += 1
                    continue
                all_X.append(X_s)
                all_y.append(y_s)
                session_ids.extend([i] * len(y_s))
                session_cache_paths[session['session_name']] = cache_path_s
                included_sessions.append(session)

                pos_count = np.sum(y_s)
                pos_pct = (pos_count / len(y_s)) * 100 if len(y_s) > 0 else 0
                self.log_train(
                    f"  Session {i+1}/{len(sessions)}: {len(y_s)} frames, "
                    f"{pos_count} positive ({pos_pct:.1f}%)")

            if not all_X:
                raise ValueError(
                    f"No sessions contained a '{behavior_name}' column in their label files. "
                    f"Check that the behavior name matches the column header in your label CSVs.")
            
            X = pd.concat(all_X, ignore_index=True)
            y = np.concatenate(all_y)
            session_ids = np.array(session_ids)
            
            pos_total = np.sum(y)
            neg_total = len(y) - pos_total
            self.log_train(
                f"\nTotal: {len(X)} frames, {pos_total} positive "
                f"({np.mean(y)*100:.1f}%), {neg_total} negative")
            self.log_train(f"Features:  {X.shape[1]}")

            if pos_total == 0:
                raise ValueError(
                    f"No positive examples found for behavior '{behavior_name}'. "
                    f"Check that labels are correctly encoded (1 = behavior present) in the selected sessions.")

            tree_method = 'hist'

            # ── Brightness Category B features (post-cache derived) ───
            # Always computed at training time when Pix_ columns exist —
            # gain pruning decides which survive the top-N.
            try:
                from prediction_pipeline import compute_brightness_category_b
                _n_before_b = X.shape[1]
                X = compute_brightness_category_b(X, log_fn=self.log_train)
                _added_b = X.shape[1] - _n_before_b
                if _added_b > 0:
                    self.log_train(f"  Added {_added_b} brightness Category-B features")
            except Exception as _bb_err:
                self.log_train(f"  ⚠️  Brightness Category-B augmentation failed: {_bb_err}")

            # ── Normalized pairwise distances (ARBEL parity) ──────────
            # Adds Dis_norm_* alongside existing Dis_* columns.  Gain
            # pruning picks whichever version (raw or normalized)
            # discriminates better per-behavior.
            try:
                from prediction_pipeline import compute_normalized_distances
                _n_before_n = X.shape[1]
                X = compute_normalized_distances(X, log_fn=self.log_train)
                _added_n = X.shape[1] - _n_before_n
                if _added_n > 0:
                    self.log_train(f"  Added {_added_n} normalized distance features")
            except Exception as _nd_err:
                self.log_train(f"  ⚠️  Normalized-distance augmentation failed: {_nd_err}")

            # Defensive: ensure no inf / NaN reaches XGBoost or the correlation
            # filter.  Training does its own augmentation inline (not via
            # augment_features_post_cache), so we need this duplicate of the
            # sanitize step here too.  Float32 max is ~3.4e38; clip at 1e9.
            try:
                _num_cols_t = X.select_dtypes(include=[np.number]).columns
                _inf_count_t = int(np.isinf(X[_num_cols_t].values).sum())
                if _inf_count_t > 0:
                    self.log_train(f"  ⚠️  Sanitizing {_inf_count_t} inf values before training")
                    X[_num_cols_t] = X[_num_cols_t].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    X[_num_cols_t] = X[_num_cols_t].clip(lower=-1e9, upper=1e9)
            except Exception as _sanitize_err:
                self.log_train(f"  ⚠️  Pre-train sanitization failed: {_sanitize_err}")

            # ── Lag/lead features (computed post-concat) ──────────────
            if self.train_use_lag_features.get():
                from pose_features import PoseFeatureExtractor
                ext = PoseFeatureExtractor(bodyparts=[])
                lag_df = ext.calculate_lag_features(X, lags=(-2, -1, 1, 2), top_n=10)
                if not lag_df.empty:
                    X = pd.concat([X, lag_df], axis=1)
                    self.log_train(f"  Added {len(lag_df.columns)} lag/lead features")

            # ── Redundant feature pre-filter ──────────────────────────
            # Drops near-duplicate features (|r| > 0.95) from the in-memory
            # DataFrame only — cached feature files on disk are never altered.
            if self.train_correlation_filter.get():
                self.log_train("\nCorrelation pre-filter (|r| > 0.95)...")
                n_before = X.shape[1]
                try:
                    # Compute correlation on a behavior-balanced subset so that
                    # features which diverge specifically during behavior events
                    # (tail-discriminative) are not wrongly flagged as redundant.
                    _pos_idx = np.where(y == 1)[0]
                    _neg_idx = np.where(y == 0)[0]
                    _n_pos = len(_pos_idx)
                    if _n_pos >= 30:
                        _n_neg = min(len(_neg_idx), _n_pos)
                        _rng = np.random.default_rng(42)
                        _neg_sample = _rng.choice(_neg_idx, size=_n_neg, replace=False)
                        _corr_idx = np.concatenate([_pos_idx, _neg_sample])
                        X_corr = X.iloc[_corr_idx]
                        self.log_train(
                            f"  Correlation computed on behavior-balanced subset "
                            f"({_n_pos} pos + {_n_neg} neg frames)")
                    else:
                        X_corr = X
                        self.log_train(
                            f"  ⚠️  Only {_n_pos} positive frames — "
                            f"correlation computed on all data")
                    corr_vals = X_corr.corr().abs().values
                    variances = X_corr.var().values     # survivor criterion on same subset
                    rows, cols = np.where(
                        np.triu(corr_vals > 0.95, k=1))
                    to_drop_idx = set()
                    for r, c in zip(rows, cols):
                        if r not in to_drop_idx and c not in to_drop_idx:
                            # drop whichever has lower variance
                            to_drop_idx.add(c if variances[r] >= variances[c] else r)
                    to_drop = [X.columns[i] for i in to_drop_idx]
                    if to_drop:
                        X = X.drop(columns=to_drop)
                        self.log_train(
                            f"  Dropped {len(to_drop)} redundant features "
                            f"({n_before} → {X.shape[1]})")
                        for col in sorted(to_drop)[:10]:
                            self.log_train(f"    - {col}")
                        if len(to_drop) > 10:
                            self.log_train(f"    ... and {len(to_drop) - 10} more")
                    else:
                        self.log_train("  No redundant feature pairs found")
                except Exception as _corr_err:
                    self.log_train(
                        f"  ⚠️  Correlation filter failed ({_corr_err}), "
                        f"continuing with all features")

            # ── Optuna hyperparameter tuning ──────────────────────────
            optuna_best_hp = None
            optuna_best_ap = None
            if self.train_use_optuna.get():
                self.log_train("\n" + "=" * 60)
                self.log_train("OPTUNA HYPERPARAMETER TUNING")
                self.log_train("=" * 60)
                self.log_train(
                    "Searching max_depth, learning_rate, colsample_bytree, "
                    "subsample, n_estimators via TPE Bayesian optimization.\n"
                    "Objective: mean CV Average Precision (AP) — area under the "
                    "Precision-Recall curve, threshold-independent.\n"
                    "AP=1.0 is perfect; AP=class-prevalence is a random classifier.\n")
                try:
                    optuna_best_hp, optuna_best_ap = self._optuna_tune(
                        X, y, session_ids, use_spw, tree_method, early_stop_rounds)
                    self.log_train(f"\n  Best hyperparameters (AP={optuna_best_ap:.4f}):")
                    for k, v in optuna_best_hp.items():
                        self.log_train(f"    {k}: {v}")
                    # Apply to UI vars so CV + final model use them
                    self.train_max_depth.set(optuna_best_hp['max_depth'])
                    self.train_learning_rate.set(optuna_best_hp['learning_rate'])
                    self.train_colsample.set(optuna_best_hp['colsample_bytree'])
                    self.train_subsample.set(optuna_best_hp['subsample'])
                    self.train_n_estimators.set(optuna_best_hp['n_estimators'])
                except ImportError:
                    self.log_train("  optuna not installed — pip install optuna")
                except Exception as e:
                    self.log_train(f"  Optuna tuning failed: {e}, using manual params")

            if self._training_cancel_flag.is_set():
                self.log_train("\nTraining cancelled by user.")
                return

            # ── Session-level K-Fold Cross-Validation ─────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("CROSS-VALIDATION")
            self.log_train("=" * 60)
            
            n_folds = self.train_n_folds.get()
            unique_sessions = np.unique(session_ids)
            actual_folds = min(n_folds, len(unique_sessions))
            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

            self.log_train(
                f"  {actual_folds}-fold session-level CV"
                + (f" (requested {n_folds}, clamped to {len(unique_sessions)} sessions)"
                   if actual_folds < n_folds else ""))
            self.log_train(f"  Hyperparameters:")
            self.log_train(f"    n_estimators:    {self.train_n_estimators.get()}")
            self.log_train(f"    max_depth:       {self.train_max_depth.get()}")
            self.log_train(f"    learning_rate:   {self.train_learning_rate.get()}")
            self.log_train(f"    subsample:       {self.train_subsample.get()}")
            self.log_train(f"    colsample:       {self.train_colsample.get()}")

            cv_result = self._run_cv_loop(
                X, y, session_ids, sessions, unique_sessions, kf,
                actual_folds, use_spw, use_early_stop, early_stop_rounds,
                tree_method)
            if cv_result is None:
                return  # cancelled

            fold_f1_scores  = cv_result['fold_f1_scores']
            fold_precisions = cv_result['fold_precisions']
            fold_recalls    = cv_result['fold_recalls']
            fold_best_iters = cv_result['fold_best_iters']
            oof_proba       = cv_result['oof_proba']
            fold_models     = cv_result.get('fold_models', [])

            mean_f1 = np.mean(fold_f1_scores)
            std_f1  = np.std(fold_f1_scores)
            
            self.log_train(f"\nCross-Validation Results (at threshold 0.5):")
            self.log_train(f"  Mean F1:        {mean_f1:.3f} ± {std_f1:.3f}")
            self.log_train(f"  Mean Precision: {np.mean(fold_precisions):.3f}")
            self.log_train(f"  Mean Recall:    {np.mean(fold_recalls):.3f}")

            if use_early_stop and fold_best_iters:
                self.log_train(f"  Fold iterations: {fold_best_iters}")
                _iter_mean = np.mean(fold_best_iters)
                _iter_std  = np.std(fold_best_iters)
                _iter_cv   = _iter_std / _iter_mean if _iter_mean > 0 else 0
                if _iter_cv > 0.3:
                    self.log_train(
                        f"  ⚠️  High variance in fold iterations "
                        f"(CV={_iter_cv:.2f}) — model may be sensitive "
                        f"to data splits")

            # ── Learning curve diagnostic (optional) ───────────────────
            lc_results = None  # will be set if learning curve is run
            if self.train_learning_curve.get():
                self.log_train("\n" + "=" * 60)
                self.log_train("LEARNING CURVE DIAGNOSTIC")
                self.log_train("=" * 60)
                self.log_train(
                    "Training on subsets of data to check if more labeling would help\n")

                lc_fractions = [0.25, 0.50, 0.75]
                lc_results = []
                for frac in lc_fractions:
                    if self._training_cancel_flag.is_set():
                        self.log_train("\nTraining cancelled by user.")
                        return
                    # Reuse `kf` — same splits as main CV for honest comparison
                    lc_cv = self._run_cv_loop(
                        X, y, session_ids, sessions, unique_sessions,
                        kf, actual_folds, use_spw, use_early_stop,
                        early_stop_rounds, tree_method, log=False,
                        train_fraction=frac)
                    if lc_cv is None:
                        return  # cancelled
                    lc_f1 = np.mean(lc_cv['fold_f1_scores'])
                    lc_results.append((frac, lc_f1))
                    self.log_train(
                        f"  {int(frac*100):3d}% data:  F1 = {lc_f1:.3f}")

                self.log_train(
                    f"  100% data:  F1 = {mean_f1:.3f}  <-- full training set")
                lc_results.append((1.0, mean_f1))

                # Check for plateau
                if lc_results:
                    last_delta = mean_f1 - lc_results[-1][1]
                    if last_delta < 0.02:
                        self.log_train(
                            f"\n  Model appears to be plateauing "
                            f"(+{last_delta:.3f} from 75% to 100%)")
                        self.log_train(
                            f"  More labeling may not help much — "
                            f"consider feature engineering or harder examples")
                    else:
                        self.log_train(
                            f"\n  F1 still improving (+{last_delta:.3f} "
                            f"from 75% to 100%) — more labeled data would "
                            f"likely help")

            # ── HMM transition fit (from training labels) ─────────────
            hmm_log_trans = hmm_log_prior = None
            try:
                hmm_log_trans, hmm_log_prior = fit_hmm_transitions(y)
                prevalence = float(np.mean(y))
                self.log_train(
                    f"\n  HMM transition probs "
                    f"(from training labels, prevalence={prevalence:.3f}):")
                self.log_train(
                    f"    P(stay behavior)   = {np.exp(hmm_log_trans[1, 1]):.4f}")
                self.log_train(
                    f"    P(stay non-behav)  = {np.exp(hmm_log_trans[0, 0]):.4f}")
            except Exception as _hmm_err:
                self.log_train(
                    f"  ⚠️  HMM fit failed ({_hmm_err}) — "
                    f"Viterbi smoothing will be unavailable for this classifier")

            # ── OOF post-processing sweep ──────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("OOF PARAMETER SWEEP (fold models)")
            self.log_train("=" * 60)
            self.log_train(
                "Finding best threshold, min_bout, min_after_bout, and max_gap on OOF predictions\n"
                "(these predictions were never seen during training — no data leakage)")
            
            # Any frames whose OOF proba is still NaN were never in a val fold
            # (can happen if n_sessions < n_folds).  Fill with mean to keep
            # the sweep functional.
            oof_valid_mask = ~np.isnan(oof_proba)
            if not np.all(oof_valid_mask):
                fill_val = np.nanmean(oof_proba)
                oof_proba[~oof_valid_mask] = fill_val
                self.log_train(
                    f"  ⚠️  {(~oof_valid_mask).sum()} frames had no OOF prediction "
                    f"(fewer sessions than folds) — filled with mean {fill_val:.3f}")

            # ── Optional: fit isotonic calibrator on OOF ───────────────
            prob_calibrator = None
            oof_proba_raw = oof_proba.copy()
            if self.train_use_calibration.get() and len(np.unique(y)) < 2:
                self.log_train(
                    "\n  ⚠️  Calibration skipped — only one class present in labels.")
            elif self.train_use_calibration.get():
                try:
                    from sklearn.isotonic import IsotonicRegression
                    from sklearn.metrics import brier_score_loss
                    brier_raw = brier_score_loss(y, oof_proba_raw)
                    prob_calibrator = IsotonicRegression(
                        y_min=0.0, y_max=1.0, out_of_bounds='clip')
                    prob_calibrator.fit(oof_proba_raw, y)
                    oof_proba = np.clip(
                        prob_calibrator.predict(oof_proba_raw), 0.0, 1.0)
                    brier_cal = brier_score_loss(y, oof_proba)
                    self.log_train(
                        f"\n  Probability calibration (isotonic): "
                        f"Brier {brier_raw:.4f} → {brier_cal:.4f}")
                    self.log_train(
                        "  Threshold sweep will run on calibrated probabilities.")
                except Exception as _cal_err:
                    self.log_train(
                        f"\n  ⚠️  Calibration failed ({_cal_err}); "
                        f"continuing with raw probabilities.")
                    prob_calibrator = None

            best_params = self._sweep_postprocessing(oof_proba, y)
            
            self.log_train(
                f"\n  Best OOF F1:  {best_params['f1']:.4f}")
            self.log_train(
                f"  Threshold:    {best_params['thresh']:.2f}  "
                f"(UI min_bout had {self.train_min_bout.get()})")
            self.log_train(
                f"  Min Bout:     {best_params['min_bout']} frames  "
                f"(UI: {self.train_min_bout.get()})")
            self.log_train(
                f"  Min After:    {best_params['min_after_bout']} frames  "
                f"(UI: {self.train_min_after_bout.get()})")
            self.log_train(
                f"  Max Gap:      {best_params['max_gap']} frames  "
                f"(UI: {self.train_max_gap.get()})")

            # Bout-level diagnostics at best frame-F1 params
            try:
                from evaluation_tab import _apply_bout_filtering, EvaluationTab
                y_raw_oof = (oof_proba >= best_params['thresh']).astype(int)
                y_pred_oof = _apply_bout_filtering(
                    y_raw_oof.copy(),
                    min_bout=best_params['min_bout'],
                    min_after_bout=best_params['min_after_bout'],
                    max_gap=best_params['max_gap'],
                )
                bout_m = EvaluationTab._compute_bout_metrics(y, y_pred_oof)
                self.log_train("")
                self.log_train(f"  Bout-level metrics (at best frame-F1 params):")
                self.log_train(f"    True bouts:     {bout_m['n_true_bouts']}")
                self.log_train(f"    Predicted:      {bout_m['n_pred_bouts']}")
                self.log_train(f"    Bout Precision: {bout_m['bout_precision']:.4f}")
                self.log_train(f"    Bout Recall:    {bout_m['bout_recall']:.4f}")
                self.log_train(f"    Bout F1:        {bout_m['bout_f1']:.4f}")
            except Exception as _e:
                self.log_train(f"  (Bout metrics unavailable: {_e})")

            # ── Probability calibration diagnostic ────────────────────
            try:
                from sklearn.calibration import calibration_curve
                from sklearn.metrics import brier_score_loss
                brier = brier_score_loss(y, oof_proba)
                fraction_pos, mean_pred = calibration_curve(
                    y, oof_proba, n_bins=10, strategy='uniform')
                self.log_train(
                    f"\n  Probability calibration (Brier score: {brier:.4f}):")
                self.log_train(
                    f"    {'Predicted':>10s}  {'Observed':>10s}  {'Count':>6s}")
                bin_edges = np.linspace(0, 1, 11)
                for _i in range(len(fraction_pos)):
                    _bin_mask = ((oof_proba >= bin_edges[_i])
                                 & (oof_proba < bin_edges[_i + 1]))
                    _count = int(_bin_mask.sum())
                    if _count > 0:
                        self.log_train(
                            f"    {mean_pred[_i]:10.3f}  "
                            f"{fraction_pos[_i]:10.3f}  {_count:6d}")
                if brier < 0.1:
                    self.log_train(
                        f"  Probabilities are well calibrated (Brier < 0.1)")
                elif brier > 0.2:
                    self.log_train(
                        f"  ⚠️  Probabilities are poorly calibrated "
                        f"(Brier > 0.2) — threshold sweep is compensating")
            except Exception as _cal_err:
                self.log_train(
                    f"  ⚠️  Calibration check failed: {_cal_err}")

            # ── Final model ────────────────────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("FINAL MODEL TRAINING")
            self.log_train("=" * 60)
            
            # n_estimators for final model: if early stopping was used,
            # take the mean best iteration across folds + 5% buffer.
            if use_early_stop and fold_best_iters:
                final_n_est = max(100, int(np.mean(fold_best_iters) * 1.05))
                self.log_train(
                    f"  n_estimators from early stopping: {final_n_est} "
                    f"(mean fold best: {np.mean(fold_best_iters):.0f})")
            else:
                final_n_est = self.train_n_estimators.get()
            
            # Global scale_pos_weight for final model (whole dataset)
            final_spw = 1.0
            if use_spw and pos_total > 0:
                final_spw = neg_total / pos_total
                self.log_train(f"  Final scale_pos_weight = {final_spw:.2f}")
            
            final_model = xgb.XGBClassifier(
                n_estimators=final_n_est,
                max_depth=self.train_max_depth.get(),
                learning_rate=self.train_learning_rate.get(),
                subsample=self.train_subsample.get(),
                colsample_bytree=self.train_colsample.get(),
                scale_pos_weight=final_spw,
                tree_method=tree_method,
                n_jobs=-1,
                objective='binary:logistic',
                random_state=42,
            )
            
            final_model.fit(X, y)
            self.log_train("  ✓ Final model trained on all data")

            # ── Final-model per-session F1 (in-sample sanity check) ────
            self.log_train("\nFinal model (in-sample) per-session F1:")
            try:
                from sklearn.metrics import f1_score as _f1s
                _boundaries = np.cumsum([0] + [len(_ys) for _ys in all_y])
                _thresh   = best_params['thresh']
                _min_bout = best_params['min_bout']
                _max_gap  = best_params['max_gap']
                for _i, _session in enumerate(included_sessions):
                    _X_s = X.iloc[_boundaries[_i]:_boundaries[_i + 1]]
                    _y_s = y[_boundaries[_i]:_boundaries[_i + 1]]
                    _proba_s = final_model.predict_proba(_X_s)[:, 1]
                    _y_raw   = (_proba_s >= _thresh).astype(int)
                    _y_pred_s = _apply_bout_filtering(_y_raw, _min_bout, best_params['min_after_bout'], _max_gap)
                    _f1_s = _f1s(_y_s, _y_pred_s, zero_division=0)
                    _sname = _session.get('session_name', f'session_{_i}')
                    self.log_train(f"  {_sname}: F1={_f1_s:.3f}")
            except Exception as _e:
                self.log_train(f"  ⚠️  Per-session F1 failed: {_e}")

            # ── Gain-importance prune + retrain (optional second pass) ──
            selected_feature_cols = None  # None → use all (model.feature_names_in_)
            pre_prune_model_ref = None   # holds the full-feature model when gain pruning runs

            if self.train_prune_by_gain.get():
                top_n = self.train_prune_top_n.get()
                self.log_train(f"\nFeature pruning: keeping top {top_n} features (gain importance)...")
                try:
                    importance = pd.Series(
                        final_model.feature_importances_,
                        index=final_model.feature_names_in_
                    )
                    top_n_actual = min(top_n, len(importance))
                    top_cols = importance.nlargest(top_n_actual).index.tolist()

                    self.log_train(
                        f"  Pruned: {len(importance)} → {len(top_cols)} features")

                    # Log cumulative importance guidance
                    sorted_imp = importance.sort_values(ascending=False)
                    total_imp = sorted_imp.sum()
                    if total_imp > 0:
                        self.log_train(f"\n  Gain cumulative importance:")
                        _shown_top_n = False
                        for _n in [10, 20, 30, 40, 60, len(sorted_imp)]:
                            if _n > len(sorted_imp):
                                continue
                            _cum = sorted_imp.iloc[:_n].sum() / total_imp * 100
                            _marker = ""
                            if _n == top_n_actual and not _shown_top_n:
                                _marker = "  <-- current top_n"
                                _shown_top_n = True
                            elif _n == top_n_actual:
                                _shown_top_n = True
                            self.log_train(
                                f"    Top {_n:3d}: {_cum:5.1f}%{_marker}")
                        if not _shown_top_n and top_n_actual not in [10, 20, 30, 40, 60, len(sorted_imp)]:
                            _cum = sorted_imp.iloc[:top_n_actual].sum() / total_imp * 100
                            self.log_train(
                                f"    Top {top_n_actual:3d}: {_cum:5.1f}%  <-- current top_n")

                    # Retrain on pruned feature set (same hyperparams)
                    X_pruned = X[top_cols]
                    pruned_model = xgb.XGBClassifier(**final_model.get_params())
                    pruned_model.fit(X_pruned, y)

                    pre_prune_model_ref  = final_model   # save for comparison plots
                    final_model          = pruned_model
                    selected_feature_cols = top_cols
                    self.log_train("  ✓ Prune + retrain complete.")

                    # ── Re-run CV on pruned features ──────────────────
                    # Same fold splits (same KFold random_state) → directly comparable
                    self.log_train("\n" + "=" * 60)
                    self.log_train("SHAP PRUNING — RE-EVALUATING WITH PRUNED FEATURES")
                    self.log_train("=" * 60)

                    allf_oof_f1 = best_params['f1']
                    allf_thresh = best_params['thresh']
                    allf_min_bout = best_params['min_bout']
                    allf_min_after = best_params['min_after_bout']
                    allf_max_gap = best_params['max_gap']

                    # Reuse `kf` — identical splits to the full-feature CV keeps
                    # the before/after comparison honest (not just equivalent
                    # by seed coincidence).
                    pruned_cv = self._run_cv_loop(
                        X_pruned, y, session_ids, sessions, unique_sessions,
                        kf, actual_folds, use_spw, use_early_stop,
                        early_stop_rounds, tree_method, log=False)

                    if pruned_cv is None:
                        return  # cancelled

                    oof_proba = pruned_cv['oof_proba']
                    fold_f1_scores  = pruned_cv['fold_f1_scores']
                    fold_precisions = pruned_cv['fold_precisions']
                    fold_recalls    = pruned_cv['fold_recalls']
                    fold_best_iters = pruned_cv['fold_best_iters']
                    fold_models     = pruned_cv.get('fold_models', [])

                    mean_f1 = np.mean(fold_f1_scores)
                    std_f1  = np.std(fold_f1_scores)

                    # Fill NaN OOF values (same logic as original)
                    _poof_valid = ~np.isnan(oof_proba)
                    if not np.all(_poof_valid):
                        oof_proba[~_poof_valid] = np.nanmean(oof_proba)

                    # Re-fit calibrator on pruned-model OOF when enabled
                    oof_proba_raw = oof_proba.copy()
                    if self.train_use_calibration.get():
                        try:
                            from sklearn.isotonic import IsotonicRegression
                            prob_calibrator = IsotonicRegression(
                                y_min=0.0, y_max=1.0, out_of_bounds='clip')
                            prob_calibrator.fit(oof_proba_raw, y)
                            oof_proba = np.clip(
                                prob_calibrator.predict(oof_proba_raw), 0.0, 1.0)
                            self.log_train(
                                "  Re-fit calibrator on pruned-model OOF.")
                        except Exception as _cal_err:
                            self.log_train(
                                f"  ⚠️  Pruned-model calibration failed ({_cal_err})")
                            prob_calibrator = None

                    best_params = self._sweep_postprocessing(oof_proba, y)

                    # Recompute final_n_est from pruned fold iterations
                    if use_early_stop and fold_best_iters:
                        final_n_est = max(100, int(np.mean(fold_best_iters) * 1.05))
                        final_model.set_params(n_estimators=final_n_est)
                        final_model.fit(X_pruned, y)

                    # Log before/after comparison
                    pruned_oof_f1 = best_params['f1']
                    delta = pruned_oof_f1 - allf_oof_f1
                    self.log_train(
                        f"  All-features OOF F1: {allf_oof_f1:.4f} "
                        f"(thresh={allf_thresh:.2f}, min_bout={allf_min_bout}, "
                        f"min_after={allf_min_after}, max_gap={allf_max_gap})")
                    self.log_train(
                        f"  Pruned ({len(top_cols)}) OOF F1:  {pruned_oof_f1:.4f} "
                        f"(thresh={best_params['thresh']:.2f}, "
                        f"min_bout={best_params['min_bout']}, "
                        f"min_after={best_params['min_after_bout']}, "
                        f"max_gap={best_params['max_gap']})")
                    if delta >= 0:
                        self.log_train(
                            f"  → Pruning improved OOF F1 by +{delta:.4f}")
                    else:
                        self.log_train(
                            f"  → Pruning decreased OOF F1 by {delta:.4f}")
                        self.log_train(
                            f"  ⚠️  Consider disabling gain pruning or increasing top_n")

                    self.log_train(
                        f"\n  Pruned CV F1 (@ 0.5): {mean_f1:.3f} ± {std_f1:.3f}")

                except Exception as _prune_err:
                    self.log_train(
                        f"  ⚠️  Gain pruning failed ({_prune_err}), "
                        f"using full-feature model instead.")

            # ── Final-model LOVO parameter sweep ──────────────────────
            oof_best_params = dict(best_params)  # snapshot before LOVO may overwrite
            opt_strategy = self.train_opt_strategy.get() if self.train_opt_strategy else 'auto'
            run_lovo = (opt_strategy == 'lovo') or (opt_strategy == 'auto' and len(included_sessions) >= 2)
            self.log_train(f"\nPost-processing optimization strategy: {opt_strategy.upper()}")
            if run_lovo:
                self.log_train("\n" + "=" * 60)
                self.log_train("LOVO PARAMETER SWEEP (fold models, truly held-out)")
                self.log_train("=" * 60)
                self.log_train(
                    "For each held-out session, the CV fold model that never saw it\n"
                    "is used to predict on the remaining sessions — eliminating the\n"
                    "in-sample bias of the previous final-model-based LOVO sweep.")

                _boundaries = np.cumsum([0] + [len(_ys) for _ys in all_y])
                _X_for_pred = X[selected_feature_cols] if selected_feature_cols else X

                # Build session_id → fold_model mapping using the CV splits that were
                # used for the main CV (or pruned CV, whichever is current).
                # unique_sessions is sorted; kf.split(unique_sessions) gives the index
                # slices used during training — same seed = same splits here.
                _sess_to_fold_model = {}
                if fold_models:
                    for _fi, (_tsi, _vsi) in enumerate(kf.split(unique_sessions)):
                        for _sid in unique_sessions[_vsi]:
                            _sess_to_fold_model[_sid] = fold_models[_fi]

                # LOVO sweep: for each held-out session use its fold model
                lovo_fold_results = []
                for k in range(len(included_sessions)):
                    lo, hi = _boundaries[k], _boundaries[k + 1]
                    mask = np.ones(len(y), dtype=bool)
                    mask[lo:hi] = False

                    # Identify this session's ID and its held-out fold model
                    _sess_id_k = session_ids[lo] if lo < len(session_ids) else None
                    _fm = _sess_to_fold_model.get(_sess_id_k) if _sess_id_k is not None else None

                    if _fm is not None:
                        # Truly held-out: fold model never trained on this session
                        _X_mask = _X_for_pred.iloc[mask] if hasattr(_X_for_pred, 'iloc') else _X_for_pred[mask]
                        train_proba = _fm.predict_proba(_X_mask)[:, 1]
                        _model_label = "fold model (held-out)"
                    else:
                        # Fallback for sessions not covered by a fold (rare edge case)
                        if not hasattr(self, '_lovo_fallback_warned'):
                            self.log_train(
                                "  ⚠️  No fold model found for one or more sessions — "
                                "falling back to final model predictions for those folds.")
                            self._lovo_fallback_warned = True
                        final_proba = final_model.predict_proba(_X_for_pred)[:, 1]
                        train_proba = final_proba[mask]
                        _model_label = "final model (fallback)"

                    train_y = y[mask]
                    fold_best = self._sweep_postprocessing(train_proba, train_y)
                    lovo_fold_results.append(fold_best)
                    sname = included_sessions[k].get('session_name', f'session_{k}')
                    self.log_train(
                        f"  Held-out {sname} [{_model_label}]: "
                        f"thresh={fold_best['thresh']:.2f}, "
                        f"min_bout={fold_best['min_bout']}, "
                        f"min_after={fold_best['min_after_bout']}, "
                        f"max_gap={fold_best['max_gap']}, "
                        f"F1={fold_best['f1']:.3f}")

                # Average across folds
                lovo_thresh = float(np.mean([r['thresh'] for r in lovo_fold_results]))
                lovo_mb = int(np.median([r['min_bout'] for r in lovo_fold_results]))
                lovo_ma = int(np.median([r['min_after_bout'] for r in lovo_fold_results]))
                lovo_mg = int(np.median([r['max_gap'] for r in lovo_fold_results]))

                # Evaluate averaged params on all data using final model (in-sample
                # sanity check only — not used for generalization estimate).
                _final_proba_all = final_model.predict_proba(_X_for_pred)[:, 1]
                y_raw_all = (_final_proba_all >= lovo_thresh).astype(int)
                y_filt_all = _apply_bout_filtering(
                    y_raw_all.copy(), lovo_mb, lovo_ma, lovo_mg)
                lovo_f1 = f1_score(y, y_filt_all, zero_division=0)

                self.log_train(f"\n  LOVO averaged params:")
                self.log_train(f"    Threshold:    {lovo_thresh:.2f}")
                self.log_train(f"    Min Bout:     {lovo_mb}")
                self.log_train(f"    Min After:    {lovo_ma}")
                self.log_train(f"    Max Gap:      {lovo_mg}")
                self.log_train(f"    F1 (all):     {lovo_f1:.3f}")

                # Compare with OOF params
                self.log_train(f"\n  OOF params:  thresh={best_params['thresh']:.2f}, "
                               f"F1={best_params['f1']:.3f}")
                self.log_train(f"  LOVO params: thresh={lovo_thresh:.2f}, "
                               f"F1={lovo_f1:.3f}")

                # Use LOVO params (they reflect the actual deployed model's behavior)
                best_params = {
                    'thresh': float(round(lovo_thresh, 2)),
                    'min_bout': lovo_mb,
                    'min_after_bout': lovo_ma,
                    'max_gap': lovo_mg,
                    'f1': lovo_f1,
                }
                self.log_train("  → Using LOVO params for classifier save")
                self.log_train(
                    "    LOVO F1 reflects fold-model performance on held-out sessions.\n"
                    "    OOF F1 is the session-level cross-validated generalization estimate.\n"
                    "    In-sample F1 (final model on all data) will be higher than both.")
            else:
                if opt_strategy == 'oof':
                    self.log_train("  → OOF-only strategy selected — skipping LOVO sweep")
                else:
                    self.log_train(
                        "\n  ℹ Only 1 session — skipping LOVO sweep, using OOF params")

            # ── Save classifier ────────────────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("SAVING CLASSIFIER")
            self.log_train("=" * 60)
            
            classifier_folder = os.path.join(project_folder, 'classifiers')
            plots_folder      = os.path.join(classifier_folder, 'plots')
            train_data_folder = os.path.join(classifier_folder, 'training_data')
            os.makedirs(classifier_folder,  exist_ok=True)
            os.makedirs(plots_folder,       exist_ok=True)
            os.makedirs(train_data_folder,  exist_ok=True)

            classifier_data = {
                # Core model
                'clf_model':        final_model,
                'Behavior_type':    behavior_name,
                # Gain-pruned feature subset (None if pruning was not used)
                'selected_feature_cols': selected_feature_cols,
                # Optimised post-processing params — LOVO (≥2 sessions) or OOF (1 session)
                'best_thresh':      best_params['thresh'],
                'min_bout':         best_params['min_bout'],
                'min_after_bout':   best_params['min_after_bout'],
                'max_gap':          best_params['max_gap'],
                # What the user had typed in the UI (saved for reference)
                'ui_min_bout':      self.train_min_bout.get(),
                'ui_min_after_bout':self.train_min_after_bout.get(),
                'ui_max_gap':       self.train_max_gap.get(),
                # Feature extraction config
                'bp_pixbrt_list':       cfg['bp_pixbrt_list'],
                'square_size':          cfg['square_size'],
                'pix_threshold':        cfg['pix_threshold'],
                'include_optical_flow': cfg.get('include_optical_flow', False),
                'bp_optflow_list':      cfg.get('bp_optflow_list', []),
                # Training provenance
                'training_sessions':  [s['session_name'] for s in sessions],
                'cv_f1_scores':       fold_f1_scores,
                'mean_cv_f1':         mean_f1,
                'std_cv_f1':          std_f1,
                'oof_best_f1':        best_params['f1'],
                'final_n_estimators': final_n_est,
                'scale_pos_weight':   final_spw,
                'optuna_best_params': optuna_best_hp,
                # OOF predictions (for eval tab diagnostics without re-running CV)
                'oof_proba':          oof_proba,
                'oof_proba_raw':      oof_proba_raw,
                'oof_best_params':    oof_best_params,
                # Optional isotonic calibrator fit on OOF (None when calibration disabled)
                'prob_calibrator':    prob_calibrator,
                # Optional fold-model ensemble (empty list when disabled) — averaged
                # with final_model at inference via predict_with_xgboost
                'fold_models':        (fold_models
                                       if self.train_use_fold_ensemble.get() else []),
                # HMM transition matrix for Viterbi smoothing (stored as list for
                # JSON-serialisable pkl; apply_smoothing reconverts with np.asarray)
                'hmm_log_trans': (hmm_log_trans.tolist()
                                  if hmm_log_trans is not None else None),
                'hmm_log_prior': (hmm_log_prior.tolist()
                                  if hmm_log_prior is not None else None),
                # Feature augmentation flags (used by eval tab to replay pipeline)
                'use_egocentric':     self.train_use_egocentric.get(),
                'use_lag_features':   self.train_use_lag_features.get(),
                'use_contact_features': self.train_use_contact_features.get(),
                'contact_threshold':  self.train_contact_threshold.get(),
            }
            
            # Timestamp suffix — shared across pkl + sibling pkl + sidecar JSON
            # so a single training run's outputs are grouped by timestamp.
            from datetime import datetime as _dt_now
            _run_ts = _dt_now.now().strftime('%Y%m%d_%H%M%S')

            if selected_feature_cols:
                n_feats = len(selected_feature_cols)
                clf_filename = f'PixelPaws_{behavior_name}_pruned_{n_feats}_{_run_ts}.pkl'
            else:
                clf_filename = f'PixelPaws_{behavior_name}_{_run_ts}.pkl'
            classifier_path = os.path.join(classifier_folder, clf_filename)

            _atomic_pickle_save(classifier_data, classifier_path)

            self.log_train(f"\n  ✓ Classifier saved: {classifier_path}")

            # ── JSON reproducibility sidecar ───────────────────────────────
            try:
                _preset_name = self._describe_training_profile()
                sidecar_path = os.path.splitext(classifier_path)[0] + '.json'
                self._write_training_sidecar(
                    sidecar_path, classifier_data, classifier_path,
                    run_ts=_run_ts, profile_label=_preset_name,
                    optuna_best_hp=optuna_best_hp, optuna_best_ap=optuna_best_ap,
                    oof_best_params=oof_best_params, lovo_best_params=best_params,
                    final_spw=final_spw, behavior_name=behavior_name, y=y,
                )
                self.log_train(f"  ✓ Hyperparameters + config saved: "
                               f"{os.path.basename(sidecar_path)}")
            except Exception as _json_err:
                self.log_train(f"  ⚠️  Could not write hyperparameter JSON: {_json_err}")

            try:
                self._append_training_history(classifier_data, _preset_name)
                self.log_train(
                    f"  ✓ Training history appended: "
                    f"classifiers/training_history.csv")
            except Exception as _hist_err:
                self.log_train(f"  ⚠️  Could not write training history: {_hist_err}")

            # ── Also save the full-feature (pre-prune) model when pruning was active ──
            if pre_prune_model_ref is not None:
                pre_prune_data = dict(classifier_data)   # shallow copy — same metadata
                pre_prune_data['clf_model']             = pre_prune_model_ref
                pre_prune_data['selected_feature_cols'] = None   # uses all features
                pre_prune_path = os.path.join(
                    classifier_folder,
                    f'PixelPaws_{behavior_name}_AllFeatures_{_run_ts}.pkl')
                _atomic_pickle_save(pre_prune_data, pre_prune_path)
                self.log_train(f"  ✓ Full-feature classifier saved: {pre_prune_path}")

            # Training data backup
            train_set_path = os.path.join(
                train_data_folder, f'{behavior_name}_train_set.pkl')
            _X_save = X[selected_feature_cols] if selected_feature_cols else X
            _atomic_pickle_save({'X': _X_save, 'y': y}, train_set_path)
            self.log_train(f"  ✓ Training set saved: {train_set_path}")
            
            # SHAP plots — always generated
            self.log_train("\nGenerating SHAP importance plots...")
            _shap_ok = self._generate_shap_plots(final_model, X, plots_folder, behavior_name,
                                                  pre_prune_model=pre_prune_model_ref)
            if _shap_ok:
                self.log_train(f"  Saved plot → {behavior_name}_shap_importance.png")
                self.log_train("  ✓ SHAP plots saved")

            # Full performance plots — optional
            if self.train_generate_plots.get():
                self.log_train("\nGenerating performance plots...")
                self.generate_performance_plots(
                    final_model, X, y, plots_folder, behavior_name,
                    oof_proba=oof_proba, oof_best_params=best_params,
                    pre_prune_model=pre_prune_model_ref)
                self.log_train(f"  Saved plot → {behavior_name}_performance.png")
                self._generate_raster_plots(
                    y, oof_proba, included_sessions,
                    [len(y_s) for y_s in all_y],
                    best_params, behavior_name, plots_folder)
                self.log_train(f"  Saved plot → {behavior_name}_raster.png")
                self._generate_oof_per_video_bar(
                    y, oof_proba, included_sessions,
                    [len(y_s) for y_s in all_y],
                    oof_best_params, behavior_name, plots_folder)
                self.log_train(f"  Saved plot → {behavior_name}_oof_per_video.png")
                self._generate_oof_per_video_bout_bar(
                    y, oof_proba, included_sessions,
                    [len(y_s) for y_s in all_y],
                    oof_best_params, behavior_name, plots_folder)
                self.log_train(f"  Saved plot → {behavior_name}_oof_bout_bar.png")
                self._generate_training_summary(
                    y, oof_proba, fold_f1_scores, best_params,
                    included_sessions, [len(y_s) for y_s in all_y],
                    behavior_name, plots_folder, final_model, X,
                    learning_curve_results=lc_results)
                self.log_train(f"  Saved plot → {behavior_name}_training_summary.png")
                self.log_train(f"  Plots saved to: {plots_folder}")
                self.log_train("  ✓ Performance plots saved")
            else:
                self.log_train("\n  (Performance plots skipped — 'Generate plots' is unchecked)")
            
            self.log_train("\n" + "=" * 60)
            self.log_train("✓✓✓ TRAINING COMPLETE! ✓✓✓")
            self.log_train("=" * 60)
            self.log_train(f"\nClassifier: {classifier_path}")
            self.log_train(
                f"CV F1 (@ 0.5):  {mean_f1:.3f} ± {std_f1:.3f}")
            self.log_train(
                f"OOF F1 (generalization): {oof_best_params['f1']:.3f}  "
                f"(thresh={oof_best_params['thresh']:.2f}, "
                f"min_bout={oof_best_params['min_bout']}, "
                f"min_after={oof_best_params['min_after_bout']}, "
                f"max_gap={oof_best_params['max_gap']})")
            if run_lovo:
                self.log_train(
                    f"LOVO params (saved):     "
                    f"thresh={best_params['thresh']:.2f}, "
                    f"min_bout={best_params['min_bout']}, "
                    f"min_after={best_params['min_after_bout']}, "
                    f"max_gap={best_params['max_gap']}")
            self.log_train(f"Total time: {time.time() - _pipeline_start:.1f}s")

            # Small-dataset next-step recommendation — points the user at the AL
            # tab when label quantity/quality is likely the dominant bottleneck.
            try:
                _n_pos = int(np.sum(y))
                _beh_lower = str(behavior_name).lower()
                _brief_behavior = any(kw in _beh_lower
                                       for kw in ('flinch', 'startle', 'withdraw', 'twitch'))
                if (len(included_sessions) < 10 and
                        (_brief_behavior or _n_pos < 1500)):
                    self.log_train("\n" + "=" * 60)
                    self.log_train("NEXT-STEP RECOMMENDATION")
                    self.log_train("=" * 60)
                    self.log_train(
                        f"Trained on {len(included_sessions)} session(s), {_n_pos} positive frame(s).")
                    self.log_train(
                        "Published flinch classifiers (e.g. ARBEL, 2025) use 15-24 sessions;")
                    self.log_train(
                        "their learning curves plateau around 2000-8000 positive frames.")
                    self.log_train("")
                    self.log_train(
                        "At this scale, the biggest F1 win usually comes from LABEL QUALITY,")
                    self.log_train(
                        "not model tweaks. The Active Learning tab surfaces frames in your")
                    self.log_train(
                        "existing sessions where this classifier is most uncertain — typically")
                    self.log_train(
                        "flinch boundaries and ambiguous paw motions. Relabeling those adds")
                    self.log_train(
                        "more F1 than any hyperparameter change at this sample size.")
                    self.log_train("")
            except Exception:
                pass

            # Active learning comparison if available
            if hasattr(self, 'pre_active_learning_f1'):
                pre_f1  = self.pre_active_learning_f1['mean']
                pre_std = self.pre_active_learning_f1['std']
                improvement = mean_f1 - pre_f1
                pct_improvement = (improvement / pre_f1) * 100 if pre_f1 > 0 else 0
                
                self.log_train("\n" + "=" * 60)
                self.log_train("📊 ACTIVE LEARNING F1 COMPARISON")
                self.log_train("=" * 60)
                self.log_train(f"Before Active Learning: {pre_f1:.3f} ± {pre_std:.3f}")
                self.log_train(f"After Active Learning:  {mean_f1:.3f} ± {std_f1:.3f}")
                self.log_train(f"\n{'='*60}")
                
                if improvement > 0:
                    self.log_train(
                        f"✨ IMPROVEMENT: +{improvement:.3f} (+{pct_improvement:.1f}%)")
                    self.log_train(f"{'='*60}")
                    _msg = (f"Model improved with active learning!\n\n"
                           f"Before: F1 = {pre_f1:.3f} ± {pre_std:.3f}\n"
                           f"After:  F1 = {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                           f"Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%)\n\n"
                           f"The new labels helped refine the decision boundary!")
                    self._safe_after(lambda m=_msg: messagebox.showinfo("Active Learning Success!", m))
                elif improvement < -0.01:
                    self.log_train(
                        f"⚠️  DECREASE: {improvement:.3f} ({pct_improvement:.1f}%)")
                    self.log_train(f"{'='*60}")
                    _msg = (f"F1 score decreased slightly after active learning.\n\n"
                           f"Before: {pre_f1:.3f} ± {pre_std:.3f}\n"
                           f"After:  {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                           f"Change: {improvement:.3f} ({pct_improvement:.1f}%)\n\n"
                           f"This can happen if:\n"
                           f"- New labels introduced noise\n"
                           f"- Very few frames were added\n"
                           f"- Labels were inconsistent\n\n"
                           f"Try labeling more frames or review labels for consistency.")
                    self._safe_after(lambda m=_msg: messagebox.showwarning("Performance Decreased", m))
                else:
                    self.log_train(f"No significant change: {improvement:.3f}")
                    self.log_train(f"{'='*60}")
                    _msg = (f"F1 score remained stable after active learning.\n\n"
                           f"Before: {pre_f1:.3f} ± {pre_std:.3f}\n"
                           f"After:  {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                           f"The new labels didn't significantly impact performance.\n"
                           f"This might mean the model was already well-calibrated.")
                    self._safe_after(lambda m=_msg: messagebox.showinfo("Performance Maintained", m))
                
                delattr(self, 'pre_active_learning_f1')
            
            # Store results for active learning
            self.last_training_results = {
                'classifier_path':    classifier_path,
                'sessions':           sessions,
                'mean_f1':            mean_f1,
                'std_f1':             std_f1,
                'oof_best_f1':        oof_best_params['f1'],
                'behavior_name':      behavior_name,
                'X':                  X,
                'y':                  y,
                'final_model':        final_model,
                'best_thresh':        best_params['thresh'],
                'session_cache_paths': session_cache_paths,
            }

            # Auto-save project config so other tabs can pick up the new classifier
            self.save_project_config(project_folder)
            

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log_train(
                f"\n\n{'='*60}\n✗ ERROR DURING TRAINING\n{'='*60}\n")
            self.log_train(error_msg)
            self._safe_after(
                lambda e=e: messagebox.showerror(
                    "Training Failed",
                    f"Error during training:\n\n{str(e)}\n\nSee log for details."))
    
    def run_active_learning_after_training(self):
        """Stub: redirect user to the Active Learning v2 tab."""
        messagebox.showinfo(
            "Active Learning",
            "Use the 🧠 Active Learning tab to run Active Learning v2.\n\n"
            "The new tab supports confidence histograms, learning curves,\n"
            "label propagation, and sub-behavior discovery.")
    
    def save_training_config(self):
        """Save current training configuration to file"""
        config_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Training Configuration"
        )
        
        if not config_path:
            return
        
        try:
            import json
            
            config = {
                'project_folder': self.train_project_folder.get(),
                'single_folder': self.train_single_folder.get(),
                'video_ext': self.train_video_ext.get(),
                'behavior_name': self.train_behavior_name.get(),
                'min_bout': self.train_min_bout.get(),
                'min_after_bout': self.train_min_after_bout.get(),
                'max_gap': self.train_max_gap.get(),
                'bp_pixbrt': self.train_bp_pixbrt.get(),
                'square_sizes': self.train_square_sizes.get(),
                'pix_threshold': self.train_pix_threshold.get(),
                'n_estimators': self.train_n_estimators.get(),
                'max_depth': self.train_max_depth.get(),
                'learning_rate': self.train_learning_rate.get(),
                'subsample': self.train_subsample.get(),
                'colsample': self.train_colsample.get(),
                'n_folds': self.train_n_folds.get(),
                'use_balancing': self.train_use_balancing.get(),
                'imbalance_thresh': self.train_imbalance_thresh.get(),
                'use_scale_pos_weight': self.train_use_scale_pos_weight.get(),
                'use_early_stopping': self.train_use_early_stopping.get(),
                'early_stopping_rounds': self.train_early_stopping_rounds.get(),
                'generate_plots': self.train_generate_plots.get(),
                'opt_strategy': self.train_opt_strategy.get(),
                'prune_by_gain': self.train_prune_by_gain.get(),
                'prune_top_n':   self.train_prune_top_n.get(),
                'trim_to_last_positive': self.train_trim_to_last_positive.get(),
                'use_optuna': self.train_use_optuna.get(),
                'optuna_trials': self.train_optuna_trials.get(),
                'use_lag_features': self.train_use_lag_features.get(),
                'use_egocentric': self.train_use_egocentric.get(),
                'use_contact_features': self.train_use_contact_features.get(),
                'contact_threshold': self.train_contact_threshold.get(),
                'correlation_filter': self.train_correlation_filter.get(),
                'learning_curve': self.train_learning_curve.get(),
                'use_calibration': self.train_use_calibration.get(),
                'use_fold_ensemble': self.train_use_fold_ensemble.get(),
            }

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Also update the project-level config so rig/behavior/etc are persisted
            self.save_project_config()

            messagebox.showinfo("Saved", f"Configuration saved to:\n{config_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save configuration:\n{str(e)}")
    
    def load_training_config(self):
        """Load training configuration from file"""
        config_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Training Configuration"
        )
        
        if not config_path:
            return
        
        try:
            import json
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if 'project_folder' in config:
                self.train_project_folder.set(config['project_folder'])
            if 'single_folder' in config:
                self.train_single_folder.set(config['single_folder'])
            if 'video_ext' in config:
                self.train_video_ext.set(config['video_ext'])
            if 'behavior_name' in config:
                self.train_behavior_name.set(config['behavior_name'])
            if 'min_bout' in config:
                self.train_min_bout.set(config['min_bout'])
            if 'min_after_bout' in config:
                self.train_min_after_bout.set(config['min_after_bout'])
            if 'max_gap' in config:
                self.train_max_gap.set(config['max_gap'])
            if 'bp_pixbrt' in config:
                self.train_bp_pixbrt.set(config['bp_pixbrt'])
            if 'square_sizes' in config:
                self.train_square_sizes.set(config['square_sizes'])
            if 'pix_threshold' in config:
                self.train_pix_threshold.set(config['pix_threshold'])
            if 'n_estimators' in config:
                self.train_n_estimators.set(config['n_estimators'])
            if 'max_depth' in config:
                self.train_max_depth.set(config['max_depth'])
            if 'learning_rate' in config:
                self.train_learning_rate.set(config['learning_rate'])
            if 'subsample' in config:
                self.train_subsample.set(config['subsample'])
            if 'colsample' in config:
                self.train_colsample.set(config['colsample'])
            if 'n_folds' in config:
                self.train_n_folds.set(config['n_folds'])
            if 'use_balancing' in config:
                self.train_use_balancing.set(config['use_balancing'])
            if 'imbalance_thresh' in config:
                self.train_imbalance_thresh.set(config['imbalance_thresh'])
            if 'use_scale_pos_weight' in config:
                self.train_use_scale_pos_weight.set(config['use_scale_pos_weight'])
            if 'use_early_stopping' in config:
                self.train_use_early_stopping.set(config['use_early_stopping'])
            if 'early_stopping_rounds' in config:
                self.train_early_stopping_rounds.set(config['early_stopping_rounds'])
            if 'generate_plots' in config:
                self.train_generate_plots.set(config['generate_plots'])
            if 'opt_strategy' in config:
                self.train_opt_strategy.set(config['opt_strategy'])
            # New keys preferred; fall back to legacy "shap_*" keys for back-compat.
            if 'prune_by_gain' in config:
                self.train_prune_by_gain.set(config['prune_by_gain'])
            elif 'shap_prune' in config:
                self.train_prune_by_gain.set(config['shap_prune'])
            if 'prune_top_n' in config:
                self.train_prune_top_n.set(config['prune_top_n'])
            elif 'shap_top_n' in config:
                self.train_prune_top_n.set(config['shap_top_n'])
            if 'trim_to_last_positive' in config:
                self.train_trim_to_last_positive.set(config['trim_to_last_positive'])
            if 'use_optuna' in config:
                self.train_use_optuna.set(config['use_optuna'])
            if 'optuna_trials' in config:
                self.train_optuna_trials.set(config['optuna_trials'])
            if 'use_lag_features' in config:
                self.train_use_lag_features.set(config['use_lag_features'])
            if 'use_egocentric' in config:
                self.train_use_egocentric.set(config['use_egocentric'])
            if 'use_contact_features' in config:
                self.train_use_contact_features.set(config['use_contact_features'])
            if 'contact_threshold' in config:
                self.train_contact_threshold.set(config['contact_threshold'])
            if 'correlation_filter' in config:
                self.train_correlation_filter.set(config['correlation_filter'])
            if 'learning_curve' in config:
                self.train_learning_curve.set(config['learning_curve'])
            if 'use_calibration' in config:
                self.train_use_calibration.set(config['use_calibration'])
            if 'use_fold_ensemble' in config:
                self.train_use_fold_ensemble.set(config['use_fold_ensemble'])
            # 'preset' key from older configs is ignored — the dropdown was
            # removed in favour of direct flag control.

            messagebox.showinfo("Loaded", f"Configuration loaded from:\n{config_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration:\n{str(e)}")
    
    
    @staticmethod
    def _feature_hash_key(cfg):
        """Build a stable, type-normalized hash key for feature cache files.

        Delegates to FeatureCacheManager.compute_hash when available, with
        an inline fallback to keep the app working if feature_cache.py is
        missing.
        """
        if FEATURE_CACHE_AVAILABLE:
            return FeatureCacheManager.compute_hash(cfg)
        # Inline fallback (identical logic)
        import hashlib
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

    def extract_features_for_session(self, session, cfg, cache_root, behavior_name):
        """
        Extract features for a single session with smart caching.

        Features are cached independently of behavior labels, so the same
        feature extraction can be reused for training multiple behaviors.
        """
        cfg_hash = PixelPawsGUI._feature_hash_key(cfg)

        # Feature cache file (behavior-independent)
        cache_filename = f"{session['session_name']}_features_{cfg_hash}.pkl"
        feature_cache_file = os.path.join(cache_root, cache_filename)
        self.log_train(f"  [Cache] Hash: {cfg_hash}  File: {cache_filename}")

        # Search alternative directories for cache files.
        if not os.path.isfile(feature_cache_file):
            video_dir = os.path.dirname(session.get('video_path', ''))
            if FEATURE_CACHE_AVAILABLE:
                found = FeatureCacheManager.find_cache(
                    session['session_name'], cfg_hash, cache_root, video_dir,
                    project_root=cache_root)
                if found:
                    self.log_train(f"  [Cache] Found in: {os.path.dirname(found)}")
                    feature_cache_file = found
                else:
                    # Try hash-agnostic fallback + version upgrade
                    any_match = FeatureCacheManager.find_any_cache(
                        session['session_name'], cache_root, video_dir,
                        project_root=cache_root)
                    if any_match:
                        # Try the newest path first: v4 → v5
                        upgraded = FeatureCacheManager.try_upgrade_v4_to_v5(
                            any_match, feature_cache_file, cfg,
                            session['pose_path'], log_fn=self.log_train)
                        if upgraded is None:
                            upgraded = FeatureCacheManager.try_upgrade_v2_to_v3(
                                any_match, feature_cache_file, cfg,
                                session['pose_path'], log_fn=self.log_train)
                            if upgraded is not None:
                                # Chain: v2→v3 succeeded → v3→v4 → v4→v5
                                v4 = FeatureCacheManager.try_upgrade_v3_to_v4(
                                    feature_cache_file, feature_cache_file, cfg,
                                    session['pose_path'], log_fn=self.log_train)
                                if v4 is not None:
                                    upgraded = v4
                                    v5 = FeatureCacheManager.try_upgrade_v4_to_v5(
                                        feature_cache_file, feature_cache_file, cfg,
                                        session['pose_path'], log_fn=self.log_train)
                                    if v5 is not None:
                                        upgraded = v5
                            else:
                                # Try v3→v4 directly on the old file, then chain to v5
                                upgraded = FeatureCacheManager.try_upgrade_v3_to_v4(
                                    any_match, feature_cache_file, cfg,
                                    session['pose_path'], log_fn=self.log_train)
                                if upgraded is not None:
                                    v5 = FeatureCacheManager.try_upgrade_v4_to_v5(
                                        feature_cache_file, feature_cache_file, cfg,
                                        session['pose_path'], log_fn=self.log_train)
                                    if v5 is not None:
                                        upgraded = v5
                        if upgraded is None:
                            self.log_train(
                                f"  [Cache] \u26a0 Feature file(s) found with DIFFERENT hash "
                                f"(config mismatch or stale cache):")
                            self.log_train(f"    \u2192 {any_match}")
                            self.log_train(
                                f"  [Cache] Expected hash {cfg_hash}. "
                                f"Check that Feature Extraction settings match training settings.")
                    else:
                        self.log_train(f"  [Cache] No cached features found \u2014 will extract.")
            else:
                # Inline fallback when feature_cache.py not available
                alt_dirs = [
                    video_dir,
                    os.path.join(video_dir, 'features'),
                    os.path.join(video_dir, 'FeatureCache'),
                    os.path.join(video_dir, 'PredictionCache'),
                ]
                for alt_dir in alt_dirs:
                    alt_path = os.path.join(alt_dir, cache_filename)
                    if os.path.isfile(alt_path):
                        self.log_train(f"  [Cache] Found in: {alt_dir}")
                        feature_cache_file = alt_path
                        break
                else:
                    self.log_train(f"  [Cache] No cached features found \u2014 will extract.")

        # Extract or load features (behavior-independent)
        if os.path.isfile(feature_cache_file):
            self.log_train(f"  [Cache] Loading features for {session['session_name']}")
            with open(feature_cache_file, 'rb') as f:
                X_full = pickle.load(f)
        else:
            # Extract features (only done once per video+config, reused for all behaviors)
            self.log_train(f"  [Extract] Extracting features for {session['session_name']}")
            
            # Get config path if user specified one
            config_yaml = self.train_dlc_config.get() if self.train_dlc_config.get() else None
            
            X_full = PixelPaws_ExtractFeatures(
                pose_data_file=session['pose_path'],
                video_file_path=session['video_path'],
                bp_include_list=cfg['bp_include_list'],
                bp_pixbrt_list=cfg['bp_pixbrt_list'],
                square_size=cfg['square_size'],
                pix_threshold=cfg['pix_threshold'],
                config_yaml_path=config_yaml,  # Pass config for crop detection
                include_optical_flow=cfg.get('include_optical_flow', False),
                bp_optflow_list=cfg.get('bp_optflow_list', []) or None,
                cancel_flag=self._feature_cancel_flag,
            )
            X_full = X_full.reset_index(drop=True)
            
            # Drop NaNs from features
            nan_mask = X_full.isna().any(axis=1)
            if nan_mask.any():
                self.log_train(f"    Dropping {nan_mask.sum()} NaN rows from features")
                X_full = X_full[~nan_mask].reset_index(drop=True)
            
            # Cache features (behavior-independent, reusable!)
            _atomic_pickle_save(X_full, feature_cache_file)
            self.log_train(f"    ✓ Cached features to {feature_cache_file}")
        
        # Egocentric features (computed post-cache from DLC coordinates)
        if self.train_use_egocentric.get():
            from pose_features import PoseFeatureExtractor
            _ego_ext = PoseFeatureExtractor(
                bodyparts=cfg.get('bp_include_list') or [])
            _ego_dlc = _ego_ext.load_dlc_data(session['pose_path'])
            _ego_xc, _ego_yc, _ = _ego_ext.get_bodypart_coords(_ego_dlc)
            _ego_x, _ego_y = _ego_ext.normalize_egocentric(_ego_xc, _ego_yc)
            _ego_dist = _ego_ext.calculate_distances(_ego_x, _ego_y)
            _ego_dist.columns = [f'Ego_{c}' for c in _ego_dist.columns]
            _ego_vel = _ego_ext.calculate_velocities(_ego_x, _ego_y, t=1)
            _ego_vel.columns = [f'Ego_{c}' for c in _ego_vel.columns]
            _ego_df = pd.concat([_ego_dist, _ego_vel], axis=1).fillna(0)
            # Align to X_full length (in case NaN rows were dropped during extraction)
            _ego_df = _ego_df.iloc[:len(X_full)].reset_index(drop=True)
            X_full = pd.concat([X_full.reset_index(drop=True), _ego_df], axis=1)
            self.log_train(f"    + {len(_ego_df.columns)} egocentric features")

        # Contact state features (derived post-cache from existing _Height columns)
        if self.train_use_contact_features.get():
            _height_cols = [c for c in X_full.columns if c.endswith('_Height')]
            if _height_cols and not any(c.endswith('_ContactState') for c in X_full.columns):
                from pose_features import PoseFeatureExtractor
                _ct_ext = PoseFeatureExtractor(bodyparts=[],
                    contact_threshold=self.train_contact_threshold.get())
                _ct_df = _ct_ext.calculate_contact_features(X_full)
                if not _ct_df.empty:
                    _ct_df = _ct_df.iloc[:len(X_full)].reset_index(drop=True)
                    X_full = pd.concat([X_full.reset_index(drop=True), _ct_df], axis=1)
                    self.log_train(f"    + {len(_ct_df.columns)} contact state features")

        # Load labels for THIS specific behavior
        # Validate that target_path exists
        if session['target_path'] is None:
            video_name = os.path.basename(session['video_path'])
            video_basename = os.path.splitext(video_name)[0]
            video_dir = os.path.dirname(session['video_path'])
            project_dir = os.path.dirname(video_dir)
            
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"❌ LABELS FILE NOT FOUND\n"
                f"{'='*60}\n\n"
                f"Video: {video_name}\n"
                f"Expected labels: {video_basename}_labels.csv\n\n"
                f"Searched in:\n"
                f"  1. Same folder as video:\n"
                f"     {os.path.join(video_dir, f'{video_basename}_labels.csv')}\n\n"
                f"  2. Separate labels folder:\n"
                f"     {os.path.join(project_dir, 'labels', f'{video_basename}_labels.csv')}\n\n"
                f"ACTION REQUIRED:\n"
                f"1. Go to the Label tab\n"
                f"2. Load this video\n"
                f"3. Label the behavior frames\n"
                f"4. Click 'Save Labels'\n"
                f"5. Try training again\n\n"
                f"Supported folder structures:\n"
                f"  Option 1: Labels with videos (recommended)\n"
                f"    videos/\n"
                f"      video.mp4\n"
                f"      video_labels.csv\n\n"
                f"  Option 2: Separate labels folder\n"
                f"    videos/\n"
                f"      video.mp4\n"
                f"    labels/\n"
                f"      video_labels.csv\n"
                f"{'='*60}\n"
            )
        
        if not os.path.exists(session['target_path']):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"❌ LABELS FILE MISSING\n"
                f"{'='*60}\n\n"
                f"Video: {os.path.basename(session['video_path'])}\n"
                f"Expected: {session['target_path']}\n\n"
                f"The labels file was found during session setup but no longer exists.\n"
                f"It may have been moved or deleted.\n"
                f"{'='*60}\n"
            )
        
        y_df = pd.read_csv(session['target_path'])
        if 'Frame' in y_df.columns:
            y_df = y_df.drop(columns=['Frame'])
        
        if behavior_name not in y_df.columns:
            raise KeyError(f"Behavior '{behavior_name}' not found in {session['target_path']}")
        
        y_full = y_df[behavior_name].fillna(-1).astype(int).values
        
        # Align lengths (truncate to shorter of features or labels)
        n = min(len(X_full), len(y_full))
        if len(X_full) != len(y_full):
            self.log_train(f"    Aligning: features={len(X_full)}, labels={len(y_full)}, using {n}")
        
        X = X_full.iloc[:n].copy()
        y = y_full[:n]

        # Trim trailing frames after the last positive event
        if cfg.get('trim_to_last_positive', False):
            positive_indices = np.where(y == 1)[0]
            if len(positive_indices) > 0:
                trim_at = int(positive_indices[-1]) + 1
                n_trimmed = len(y) - trim_at
                if n_trimmed > 0:
                    self.log_train(
                        f"    Trimmed {n_trimmed} trailing frame(s) after last positive "
                        f"\u2192 {trim_at} frames used")
                    X = X.iloc[:trim_at].reset_index(drop=True)
                    y = y[:trim_at]
            # (if no positives found, no trim — training fails later with a clear error)

        # Filter unlabeled frames (value = -1 means the user never reviewed this frame)
        labeled_mask = (y != -1)
        n_unlabeled = int((~labeled_mask).sum())
        if n_unlabeled > 0:
            self.log_train(
                f"    Filtered {n_unlabeled} unlabeled frame(s) (label=-1) "
                f"— only {labeled_mask.sum()} reviewed frames used")
            X = X.iloc[labeled_mask].reset_index(drop=True)
            y = y[labeled_mask]

        return X, y, feature_cache_file
    
    def _sweep_postprocessing(self, oof_proba, y):
        """
        Joint 4-D grid search over (threshold, min_bout, min_after_bout,
        max_gap) using out-of-fold probabilities.  Because these
        probabilities were produced by models that never trained on the
        corresponding frames, the chosen parameters are unbiased estimates
        of real-world post-processing performance.

        Grid aligned with evaluation_tab._grid_search_params().
        Returns a dict with keys: thresh, min_bout, min_after_bout, max_gap, f1
        """
        from evaluation_tab import _apply_bout_filtering

        # Search grids — aligned with eval tab's _grid_search_params()
        thresholds      = np.arange(0.10, 0.91, 0.05)   # 17 values
        min_bouts       = [1, 2, 3, 5, 8, 12, 15, 20]   # 8 values
        min_after_bouts = [0, 1, 3, 5]                   # 4 values
        # max_gap capped at 15 frames (~500 ms @ 30 fps).  A 20-frame cap (670 ms)
        # was observed to merge separate flinches into a single bout during OOF
        # sweep, inflating apparent F1 and producing a brittle deployment
        # threshold.  ARBEL uses max_gap=2; 15 is the upper-bound safe default.
        max_gaps        = [0, 2, 4, 5, 6, 10, 15]       # 7 values
        # Total: 17 × 8 × 4 × 7 = 3,808 combinations — runs in ~1-2 s

        best_f1    = -1.0
        best_thresh = 0.5
        best_mb     = 1
        best_ma     = 1
        best_mg     = 0

        for thresh in thresholds:
            y_raw = (oof_proba >= thresh).astype(int)
            for mb in min_bouts:
                for ma in min_after_bouts:
                    for mg in max_gaps:
                        if mb == 1 and ma == 0 and mg == 0:
                            y_filt = y_raw
                        else:
                            y_filt = _apply_bout_filtering(
                                y_raw.copy(), min_bout=mb,
                                min_after_bout=ma, max_gap=mg)
                        score = f1_score(y, y_filt, zero_division=0)
                        if score > best_f1:
                            best_f1    = score
                            best_thresh = thresh
                            best_mb     = mb
                            best_ma     = ma
                            best_mg     = mg

        return {
            'thresh':       float(round(best_thresh, 2)),
            'min_bout':     int(best_mb),
            'min_after_bout': int(best_ma),
            'max_gap':      int(best_mg),
            'f1':           float(best_f1),
        }

    def balance_data(self, X, y):
        """Balance imbalanced data using oversampling"""
        positive_idx = np.where(y == 1)[0]
        negative_idx = np.where(y == 0)[0]
        
        n_positive = len(positive_idx)
        n_negative = len(negative_idx)
        
        # Target ~20% positive
        target_pos_frac = 0.20
        target_total = int(n_positive / target_pos_frac) if n_positive > 0 else len(y)
        target_zeros = max(min(target_total - n_positive, n_negative), 0)
        
        if target_zeros > 0 and target_zeros < n_negative:
            from sklearn.utils import resample
            negative_idx_sampled = resample(negative_idx, n_samples=target_zeros, 
                                          replace=False, random_state=42)
            all_idx = np.concatenate([positive_idx, negative_idx_sampled])
        else:
            all_idx = np.concatenate([positive_idx, negative_idx])
        
        return X[all_idx], y[all_idx]

    def _generate_raster_plots(self, y_all, oof_proba_all, sessions, all_y_lengths,
                               best_params, behavior_name, output_folder):
        """Generate per-session 3-panel diagnostic figure: raster | confusion matrix | time bins."""
        if plt is None:
            return

        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None

        # Cumulative frame boundaries for slicing
        boundaries = np.cumsum([0] + list(all_y_lengths))

        thresh    = best_params['thresh']
        min_bout  = best_params['min_bout']
        max_gap   = best_params['max_gap']

        def _bouts_from_array(arr):
            padded = np.concatenate([[0], arr.astype(int), [0]])
            diff   = np.diff(padded)
            starts = np.where(diff ==  1)[0]
            ends   = np.where(diff == -1)[0]
            return list(zip(starts.tolist(), (ends - starts).tolist()))

        for i, session in enumerate(sessions):
            y_true  = y_all[boundaries[i]:boundaries[i + 1]]
            proba_s = oof_proba_all[boundaries[i]:boundaries[i + 1]]

            # Apply threshold + bout filtering
            y_raw  = (proba_s >= thresh).astype(int)
            y_pred = _apply_bout_filtering(y_raw, min_bout, 0, max_gap)

            # FPS from video
            fps = 30.0
            video_path = session.get('video_path', '')
            if video_path and os.path.isfile(video_path):
                try:
                    cap = cv2.VideoCapture(video_path)
                    try:
                        _fps = cap.get(cv2.CAP_PROP_FPS)
                        if _fps and _fps > 0:
                            fps = float(_fps)
                    finally:
                        cap.release()
                except Exception as _vid_err:
                    print(f"Warning: could not read FPS from {video_path}: {_vid_err}")

            # Per-session F1
            from sklearn.metrics import f1_score as _f1, confusion_matrix as _cm
            f1 = _f1(y_true, y_pred, zero_division=0)

            # Confusion matrix (row-normalised)
            cm_raw  = _cm(y_true, y_pred, labels=[0, 1])
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1          # avoid division by zero
            cm_norm = cm_raw / row_sums

            # Time-bin arrays (10 s bins)
            bin_frames = max(1, int(10 * fps))
            n_frames   = len(y_true)
            n_bins     = max(1, int(np.ceil(n_frames / bin_frames)))
            human_s = np.zeros(n_bins)
            model_s = np.zeros(n_bins)
            for k in range(n_bins):
                sl = slice(k * bin_frames, (k + 1) * bin_frames)
                human_s[k] = y_true[sl].sum() / fps
                model_s[k] = y_pred[sl].sum() / fps

            if pearsonr is not None and n_bins > 1:
                r_val, _ = pearsonr(human_s, model_s)
            else:
                r_val = float('nan')

            # ── Figure ──────────────────────────────────────────────────
            fig = plt.figure(figsize=(16, 6.5))
            gs  = fig.add_gridspec(2, 3, width_ratios=[5, 2, 4],
                                   height_ratios=[3, 2],
                                   wspace=0.35, hspace=0.4)
            ax_raster = fig.add_subplot(gs[0, 0])
            ax_cm     = fig.add_subplot(gs[0, 1])
            ax_bins   = fig.add_subplot(gs[0, 2])
            ax_proba  = fig.add_subplot(gs[1, :])

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
            session_name = session.get('session_name', f'session_{i}')
            ax_raster.set_title(f"{behavior_name} raster: {session_name}\n(thr={thresh:.2f})")
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color='black',    linewidth=8, label='Human'),
                Line2D([0], [0], color='#E87722',  linewidth=8, label='Model'),
            ]
            ax_raster.legend(handles=legend_handles, loc='upper left', fontsize=10)

            # Panel 2 — Confusion Matrix
            im = ax_cm.imshow(cm_norm, cmap='RdPu', vmin=0, vmax=1)
            for row in range(2):
                for col in range(2):
                    v = cm_norm[row, col]
                    text_color = 'white' if v > 0.5 else 'black'
                    ax_cm.text(col, row, f"{v:.2f}", ha='center', va='center',
                               color=text_color, fontsize=10)
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['0', '1'])
            ax_cm.set_yticklabels(['0', '1'])
            ax_cm.set_xlabel('Pred')
            ax_cm.set_ylabel('True')
            ax_cm.set_title(f"F1={f1:.2f}")

            # Panel 3 — Time Bins
            x_centers = np.arange(n_bins) * 10 + 5
            width = 0.4
            ax_bins.bar(x_centers - width / 2, human_s, width=width,
                        color='steelblue', label='Human')
            ax_bins.bar(x_centers + width / 2, model_s, width=width,
                        color='#E87722', label='Model')
            ax_bins.set_xlabel('Time (s)')
            ax_bins.set_ylabel('Seconds/bin')
            r_str = f"{r_val:.2f}" if not np.isnan(r_val) else "n/a"
            ax_bins.set_title(f"Time bins (10s);  R = {r_str}")
            ax_bins.legend(fontsize=10)
            ax_bins.spines['top'].set_visible(False)
            ax_bins.spines['right'].set_visible(False)

            # Panel 4 — Probability trace (bottom, full width)
            frames = np.arange(len(proba_s))
            ax_proba.plot(frames, proba_s, color='#2196F3', alpha=0.7,
                          linewidth=0.5, label='Probability')
            ax_proba.axhline(thresh, color='#E53935', linestyle='--',
                             linewidth=1, label=f'Threshold = {thresh:.2f}')
            # Shade true-positive regions
            true_mask = y_true.astype(bool)
            _starts = np.where(np.diff(np.concatenate([[0], true_mask.astype(int), [0]])) == 1)[0]
            _ends   = np.where(np.diff(np.concatenate([[0], true_mask.astype(int), [0]])) == -1)[0]
            for _s, _e in zip(_starts, _ends):
                ax_proba.axvspan(_s, _e, alpha=0.2, color='#4CAF50')
            ax_proba.set_xlim(0, len(proba_s))
            ax_proba.set_ylim(-0.02, 1.02)
            ax_proba.set_xlabel('Frame')
            ax_proba.set_ylabel('Probability')
            ax_proba.legend(loc='upper right', fontsize=8)
            ax_proba.spines['top'].set_visible(False)
            ax_proba.spines['right'].set_visible(False)

            out_path = os.path.join(
                output_folder,
                f"PixelPaws_{behavior_name}_Raster_{session_name}.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.log_train(f"    Raster plot: {os.path.basename(out_path)}")

    def _generate_oof_per_video_bar(self, y, oof_proba, sessions, session_lengths,
                                     best_params, behavior_name, output_folder):
        """Per-video OOF bar chart — F1, Precision, Recall."""
        if plt is None:
            return
        from sklearn.metrics import f1_score, precision_score, recall_score

        thresh = best_params['thresh']
        mb = best_params['min_bout']
        ma = best_params['min_after_bout']
        mg = best_params['max_gap']

        boundaries = np.cumsum([0] + list(session_lengths))
        names, f1s, precs, recs = [], [], [], []

        for i, session in enumerate(sessions):
            lo, hi = boundaries[i], boundaries[i + 1]
            y_s = y[lo:hi]
            p_s = oof_proba[lo:hi]
            if np.all(np.isnan(p_s)):
                continue
            y_raw = (p_s >= thresh).astype(int)
            y_pred = _apply_bout_filtering(y_raw.copy(), mb, ma, mg)
            names.append(session.get('session_name', f'session_{i}')[:20])
            f1s.append(f1_score(y_s, y_pred, zero_division=0))
            precs.append(precision_score(y_s, y_pred, zero_division=0))
            recs.append(recall_score(y_s, y_pred, zero_division=0))

        if not names:
            return

        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        ax.bar(x - w, f1s, w, label='F1', color='steelblue')
        ax.bar(x, precs, w, label='Precision', color='darkorange')
        ax.bar(x + w, recs, w, label='Recall', color='seagreen')
        ax.axhline(np.mean(f1s), ls='--', color='steelblue', alpha=0.5,
                    label=f'Mean F1 ({np.mean(f1s):.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Video OOF Performance — {behavior_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder,
                    f'PixelPaws_{behavior_name}_OOF_PerVideo.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _generate_oof_per_video_bout_bar(self, y, oof_proba, sessions, session_lengths,
                                          best_params, behavior_name, output_folder):
        """Per-video OOF BOUT-level bar chart — Bout F1, Precision, Recall."""
        if plt is None:
            return
        from evaluation_tab import _apply_bout_filtering, EvaluationTab

        thresh = best_params['thresh']
        mb, ma, mg = best_params['min_bout'], best_params['min_after_bout'], best_params['max_gap']

        boundaries = np.cumsum([0] + list(session_lengths))
        names, f1s, precs, recs = [], [], [], []

        for i, session in enumerate(sessions):
            lo, hi = boundaries[i], boundaries[i + 1]
            y_s = y[lo:hi]
            p_s = oof_proba[lo:hi]
            if np.all(np.isnan(p_s)) or len(y_s) == 0:
                continue
            y_raw = (p_s >= thresh).astype(int)
            y_pred = _apply_bout_filtering(y_raw.copy(), mb, ma, mg)
            bm = EvaluationTab._compute_bout_metrics(y_s, y_pred)
            names.append(session.get('session_name', f'session_{i}')[:20])
            f1s.append(bm['bout_f1'])
            precs.append(bm['bout_precision'])
            recs.append(bm['bout_recall'])

        if not names:
            return

        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        ax.bar(x - w, f1s,   w, label='Bout F1',        color='steelblue')
        ax.bar(x,     precs, w, label='Bout Precision', color='darkorange')
        ax.bar(x + w, recs,  w, label='Bout Recall',    color='seagreen')
        ax.axhline(np.mean(f1s), ls='--', color='steelblue', alpha=0.5,
                    label=f'Mean Bout F1 ({np.mean(f1s):.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Video OOF Bout-Level Performance — {behavior_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder,
                    f'PixelPaws_{behavior_name}_OOF_PerVideo_Bouts.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _generate_training_summary(self, y, oof_proba, fold_f1_scores, best_params,
                                    sessions, all_y_lengths, behavior_name,
                                    output_folder, final_model, X,
                                    learning_curve_results=None):
        """Generate a 2x3 Training Summary Dashboard figure."""
        if plt is None:
            return
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss

            COLORS = {
                'primary':   '#2196F3',
                'secondary': '#FF9800',
                'tertiary':  '#4CAF50',
                'neg_class': '#90CAF9',
                'pos_class': '#FF8A65',
                'threshold': '#E53935',
                'mean_line': '#7B1FA2',
                'grid':      '#E0E0E0',
            }

            thresh = best_params['thresh']
            min_bout = best_params['min_bout']
            max_gap = best_params['max_gap']
            min_after = best_params['min_after_bout']

            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)

            # OOF F1
            y_raw = (oof_proba >= thresh).astype(int)
            y_pred = _apply_bout_filtering(y_raw.copy(), min_bout, min_after, max_gap)
            oof_f1 = f1_score(y, y_pred, zero_division=0)

            n_features = X.shape[1] if hasattr(X, 'shape') else 0
            n_sessions = len(sessions)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # ── (0,0) CV F1 by Fold ──────────────────────────────────
            ax = axes[0, 0]
            n_folds = len(fold_f1_scores)
            fold_labels = [f'Fold {i+1}' for i in range(n_folds)]
            f1_arr = np.array(fold_f1_scores)
            norm = plt.Normalize(vmin=f1_arr.min() - 0.01, vmax=f1_arr.max() + 0.01)
            colors_fold = plt.cm.RdYlGn(norm(f1_arr))
            bars = ax.barh(fold_labels, f1_arr, color=colors_fold, edgecolor='white')
            ax.axvline(mean_f1, color=COLORS['mean_line'], linestyle='--',
                        linewidth=1.5, label=f'Mean = {mean_f1:.3f}')
            for bar, val in zip(bars, f1_arr):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=9)
            ax.set_xlabel('F1 Score')
            ax.set_title('CV F1 by Fold')
            ax.legend(fontsize=8)
            ax.set_xlim(0, min(1.05, f1_arr.max() + 0.08))

            # ── (0,1) OOF Probability Distributions ──────────────────
            ax = axes[0, 1]
            ax.hist(oof_proba[y == 0], bins=50, alpha=0.6, color=COLORS['neg_class'],
                    label='Negative', density=True)
            ax.hist(oof_proba[y == 1], bins=50, alpha=0.6, color=COLORS['pos_class'],
                    label='Positive', density=True)
            ax.axvline(thresh, color=COLORS['threshold'], linestyle='--',
                        linewidth=1.5, label=f'Threshold = {thresh:.2f}')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.set_title('OOF Probability Distributions')
            ax.legend(fontsize=8)

            # ── (0,2) Calibration Curve ──────────────────────────────
            ax = axes[0, 2]
            try:
                prob_true, prob_pred = calibration_curve(y, oof_proba, n_bins=10,
                                                          strategy='uniform')
                brier = brier_score_loss(y, oof_proba)
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7,
                        label='Perfectly calibrated')
                ax.plot(prob_pred, prob_true, marker='o', color=COLORS['primary'],
                        label=f'OOF (Brier={brier:.3f})')
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title('Calibration Curve')
                ax.legend(fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, 'Calibration\nnot available', ha='center',
                        va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Calibration Curve')

            # ── (1,0) Learning Curve or Top SHAP Features ────────────
            ax = axes[1, 0]
            if learning_curve_results is not None and len(learning_curve_results) > 1:
                fracs = [r[0] for r in learning_curve_results]
                f1s_lc = [r[1] for r in learning_curve_results]
                ax.plot([f * 100 for f in fracs], f1s_lc, marker='o',
                        color=COLORS['primary'], linewidth=2, markersize=8)
                for f, v in zip(fracs, f1s_lc):
                    ax.annotate(f'{v:.3f}', (f * 100, v),
                                textcoords='offset points', xytext=(0, 10),
                                ha='center', fontsize=9)
                ax.set_xlabel('% Training Data')
                ax.set_ylabel('F1 Score')
                ax.set_title('Learning Curve')
                ax.grid(True, alpha=0.3)
            else:
                # Top features by gain importance
                try:
                    imp = pd.Series(
                        final_model.feature_importances_,
                        index=final_model.feature_names_in_
                        if hasattr(final_model, 'feature_names_in_') else
                        [f'f{i}' for i in range(len(final_model.feature_importances_))]
                    )
                    top = imp.nlargest(10)
                    ax.barh(top.index.tolist(), top.values, color=COLORS['primary'])
                    ax.set_xlabel('Gain Importance')
                    ax.set_title('Top 10 Features (Gain)')
                except Exception:
                    ax.text(0.5, 0.5, 'Feature importance\nnot available', ha='center',
                            va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title('Top Features')

            # ── (1,1) Threshold Sweep (OOF) ──────────────────────────
            ax = axes[1, 1]
            thresholds = np.arange(0.05, 0.96, 0.01)
            f1s_t, precs_t, recs_t = [], [], []
            for t in thresholds:
                y_t = (oof_proba >= t).astype(int)
                f1s_t.append(f1_score(y, y_t, zero_division=0))
                precs_t.append(precision_score(y, y_t, zero_division=0))
                recs_t.append(recall_score(y, y_t, zero_division=0))
            ax.plot(thresholds, f1s_t, color=COLORS['primary'], linewidth=2, label='F1')
            ax.plot(thresholds, precs_t, color=COLORS['secondary'], linewidth=1.2, label='Precision')
            ax.plot(thresholds, recs_t, color=COLORS['tertiary'], linewidth=1.2, label='Recall')
            ax.axvline(thresh, color=COLORS['threshold'], linestyle='--',
                        linewidth=1.5, label=f'Best = {thresh:.2f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Threshold Sweep (OOF)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (1,2) Per-Video OOF F1 ───────────────────────────────
            ax = axes[1, 2]
            boundaries = np.cumsum([0] + list(all_y_lengths))
            vid_names, vid_f1s = [], []
            for i, session in enumerate(sessions):
                lo, hi = boundaries[i], boundaries[i + 1]
                y_s = y[lo:hi]
                p_s = oof_proba[lo:hi]
                if np.all(np.isnan(p_s)):
                    continue
                y_raw_s = (p_s >= thresh).astype(int)
                y_pred_s = _apply_bout_filtering(y_raw_s.copy(), min_bout, min_after, max_gap)
                vid_names.append(session.get('session_name', f'session_{i}')[:20])
                vid_f1s.append(f1_score(y_s, y_pred_s, zero_division=0))
            if vid_f1s:
                sort_idx = np.argsort(vid_f1s)[::-1]
                vid_names = [vid_names[i] for i in sort_idx]
                vid_f1s = [vid_f1s[i] for i in sort_idx]
                vid_colors = plt.cm.RdYlGn(plt.Normalize(
                    vmin=min(vid_f1s) - 0.01, vmax=max(vid_f1s) + 0.01)(vid_f1s))
                bars = ax.barh(vid_names, vid_f1s, color=vid_colors, edgecolor='white')
                for bar, val in zip(bars, vid_f1s):
                    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                            f'{val:.3f}', va='center', fontsize=9)
                vid_mean = np.mean(vid_f1s)
                ax.axvline(vid_mean, color=COLORS['mean_line'], linestyle='--',
                            linewidth=1.5, label=f'Mean = {vid_mean:.3f}')
                ax.legend(fontsize=8)
            ax.set_xlabel('F1 Score')
            ax.set_title('Per-Video OOF F1')
            ax.set_xlim(0, 1.05)

            # ── Suptitle ─────────────────────────────────────────────
            fig.suptitle(
                f'PixelPaws \u2014 {behavior_name}\n'
                f'CV F1: {mean_f1:.3f} \u00b1 {std_f1:.3f}  |  '
                f'OOF F1: {oof_f1:.3f}  |  '
                f'{n_features} features  |  {n_sessions} sessions',
                fontsize=13, fontweight='bold', y=1.02)

            fig.tight_layout()
            out_path = os.path.join(output_folder,
                                     f'PixelPaws_{behavior_name}_TrainingSummary.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.log_train(f"    Training summary: {os.path.basename(out_path)}")

        except Exception as e:
            self.log_train(f"  Warning: Training summary plot failed: {e}")

    def _generate_shap_plots(self, model, X, output_folder, behavior_name,
                              pre_prune_model=None):
        """Generate feature importance plots using XGBoost native gain importance."""
        try:
            import matplotlib.pyplot as _plt
            import pandas as _pd
            import numpy as _np

            def _plot_importance(m, title, out_path, label):
                imp = _pd.Series(
                    m.feature_importances_,
                    index=m.feature_names_in_ if hasattr(m, 'feature_names_in_')
                          else [f'f{i}' for i in range(len(m.feature_importances_))]
                ).nlargest(20).iloc[::-1]
                fig, ax = _plt.subplots(figsize=(10, max(6, len(imp) * 0.35)))
                ax.barh(imp.index.tolist(), imp.values)
                ax.set_xlabel('Gain Importance')
                ax.set_title(title, fontsize=14)
                _plt.tight_layout()
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                _plt.close(fig)

            # Pre-prune model (all features) — only when feature pruning was active
            if pre_prune_model is not None:
                n_full = (len(pre_prune_model.feature_names_in_)
                          if hasattr(pre_prune_model, 'feature_names_in_')
                          else '?')
                _plot_importance(
                    pre_prune_model,
                    f'Feature Importance (all {n_full} features) — {behavior_name}',
                    os.path.join(output_folder,
                                 f'PixelPaws_{behavior_name}_SHAP_AllFeatures.png'),
                    'all'
                )

            # Final (pruned or only) model
            bar_title = (f'Feature Importance Bar (pruned) — {behavior_name}'
                         if pre_prune_model is not None
                         else f'Feature Importance Bar — {behavior_name}')
            _plot_importance(
                model,
                bar_title,
                os.path.join(output_folder,
                             f'PixelPaws_{behavior_name}_SHAP_Bar.png'),
                'pruned'
            )

            # Importance summary (same data, alternate filename for compatibility)
            imp_title = (f'Feature Importance (pruned) — {behavior_name}'
                         if pre_prune_model is not None
                         else f'Feature Importance — {behavior_name}')
            _plot_importance(
                model,
                imp_title,
                os.path.join(output_folder,
                             f'PixelPaws_{behavior_name}_SHAP_Importance.png'),
                'summary'
            )

            return True
        except Exception as e:
            self.log_train(f"    Importance plots skipped: {e}")
            return False

    def generate_performance_plots(self, model, X, y, output_folder, behavior_name,
                                    oof_proba=None, oof_best_params=None,
                                    pre_prune_model=None):
        """Generate performance visualization plots.

        If oof_proba is provided, a second (honest) OOF threshold curve is
        drawn alongside the in-sample training curve so the two can be compared.
        The vertical marker shows the OOF-optimised threshold.
        """
        if plt is None:
            return
        
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = None
            X_array = X
        
        thresholds = np.arange(0.05, 0.96, 0.01)

        # ── In-sample (training) curve ────────────────────────────────
        y_proba_train = predict_with_xgboost(model, X)
        train_f1s, train_precs, train_recs = [], [], []
        for t in thresholds:
            yp = (y_proba_train >= t).astype(int)
            train_f1s.append(f1_score(y, yp, zero_division=0))
            train_precs.append(precision_score(y, yp, zero_division=0))
            train_recs.append(recall_score(y, yp, zero_division=0))

        # ── Pre-prune in-sample curve (only when gain pruning was active) ──
        pre_f1s = pre_precs = pre_recs = None
        if pre_prune_model is not None:
            y_proba_pre = predict_with_xgboost(pre_prune_model, X)
            pre_f1s, pre_precs, pre_recs = [], [], []
            for t in thresholds:
                yp = (y_proba_pre >= t).astype(int)
                pre_f1s.append(f1_score(y, yp, zero_division=0))
                pre_precs.append(precision_score(y, yp, zero_division=0))
                pre_recs.append(recall_score(y, yp, zero_division=0))

        # ── OOF curve (honest) ────────────────────────────────────────
        oof_f1s = None
        if oof_proba is not None:
            oof_f1s = []
            for t in thresholds:
                yp = (oof_proba >= t).astype(int)
                oof_f1s.append(f1_score(y, yp, zero_division=0))

        # ── Plot ──────────────────────────────────────────────────────
        n_pruned = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else ''
        n_full   = (len(pre_prune_model.feature_names_in_)
                    if pre_prune_model is not None and hasattr(pre_prune_model, 'feature_names_in_')
                    else '')
        post_lbl = f' ({n_pruned} feat)' if pre_prune_model is not None and n_pruned else ''

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

        # Pre-prune dotted lines (all features, lightest)
        if pre_f1s is not None:
            ax.plot(thresholds, pre_f1s,   ':', color='steelblue',  linewidth=1.2, alpha=0.45,
                    label=f'F1 pre-prune ({n_full} feat)')
            ax.plot(thresholds, pre_precs, ':', color='darkorange', linewidth=1.2, alpha=0.45,
                    label=f'Precision pre-prune ({n_full} feat)')
            ax.plot(thresholds, pre_recs,  ':', color='seagreen',   linewidth=1.2, alpha=0.45,
                    label=f'Recall pre-prune ({n_full} feat)')

        # Post-prune / only model — dashed in-sample
        ax.plot(thresholds, train_f1s,   '--', color='steelblue',
                label=f'F1 in-sample{post_lbl}',        linewidth=1.5, alpha=0.6)
        ax.plot(thresholds, train_precs, '--', color='darkorange',
                label=f'Precision in-sample{post_lbl}', linewidth=1.5, alpha=0.6)
        ax.plot(thresholds, train_recs,  '--', color='seagreen',
                label=f'Recall in-sample{post_lbl}',    linewidth=1.5, alpha=0.6)

        # OOF curves (solid — honest)
        if oof_f1s is not None:
            ax.plot(thresholds, oof_f1s, '-', color='steelblue',
                    label='F1 (out-of-fold, honest)', linewidth=2.5)

        # Mark OOF-chosen threshold
        if oof_best_params is not None:
            bt = oof_best_params['thresh']
            bf = oof_best_params['f1']
            ax.axvline(bt, color='red', linestyle=':', linewidth=1.5,
                       label=f"OOF best threshold = {bt:.2f}  (F1={bf:.3f})")
            ax.scatter([bt], [bf], color='red', zorder=5, s=60)

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score',     fontsize=12)
        prune_note = (f'\n(dotted = all {n_full} feat; dashed = top {n_pruned} feat)'
                      if pre_prune_model is not None else '')
        ax.set_title(
            f'Threshold Curve — {behavior_name}\n'
            f'(dashed = in-sample / overfit; solid = honest OOF){prune_note}',
            fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 1.02])

        plt.savefig(
            os.path.join(output_folder,
                         f'PixelPaws_{behavior_name}_PerformanceThreshold.png'),
            dpi=300, bbox_inches='tight')
        plt.close()

        # ── Calibration (reliability) diagram ─────────────────────────
        if oof_proba is not None:
            try:
                from sklearn.calibration import calibration_curve
                fig_cal, (ax_cal, ax_hist) = plt.subplots(
                    2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]},
                    constrained_layout=True)

                prob_true, prob_pred = calibration_curve(y, oof_proba, n_bins=10, strategy='uniform')
                ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfectly calibrated')
                ax_cal.plot(prob_pred, prob_true, 's-', color='steelblue', label='OOF predictions')
                ax_cal.set_xlabel('Mean predicted probability')
                ax_cal.set_ylabel('Fraction of positives')
                ax_cal.set_title(f'Calibration Curve — {behavior_name}\n(out-of-fold predictions)')
                ax_cal.legend()
                ax_cal.grid(alpha=0.3)
                ax_cal.set_xlim([0, 1])
                ax_cal.set_ylim([0, 1])

                ax_hist.hist(oof_proba[y == 0], bins=50, alpha=0.5, label='Negative', color='steelblue')
                ax_hist.hist(oof_proba[y == 1], bins=50, alpha=0.5, label='Positive', color='darkorange')
                if oof_best_params:
                    ax_hist.axvline(oof_best_params['thresh'], color='red', linestyle=':',
                                    label=f"Threshold={oof_best_params['thresh']:.2f}")
                ax_hist.set_xlabel('Predicted probability')
                ax_hist.set_ylabel('Count')
                ax_hist.legend()
                ax_hist.grid(alpha=0.3)

                plt.savefig(os.path.join(output_folder, f'PixelPaws_{behavior_name}_Calibration.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass  # calibration plot is non-critical

        # ── SHAP importance — delegated to shared helper ──────────────
        # (SHAP is now always generated unconditionally from _real_training;
        #  this call handles the case where generate_performance_plots is
        #  invoked standalone, e.g. from the Plots checkbox path)
        self._generate_shap_plots(
            model,
            X if hasattr(X, 'columns') else
            (pd.DataFrame(X_array, columns=feature_names) if feature_names else X_array),
            output_folder, behavior_name,
            pre_prune_model=pre_prune_model)
    
    def _safe_after(self, callback):
        """Schedule *callback* on the main thread, swallowing errors if the
        window has been destroyed (e.g. user closed it during a long task)."""
        try:
            self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def log_train(self, message):
        """Thread-safe: add message to training log."""
        def _do():
            if self.train_log:
                self.train_log.insert(tk.END, message + '\n')
                self.train_log.see(tk.END)
        self._safe_after(_do)

    # === ENHANCED TOOL METHODS ===
    
    def open_video_preview(self):
        """Open video preview window"""
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi"), ("All files", "*.*")]
        )
        
        if video_path:
            # For demo, use None for predictions
            try:
                preview = VideoPreviewWindow(self.root, video_path, None, None)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open video:\n{str(e)}")
    
    def open_auto_labeler(self):
        """Open auto-labeling assistant"""
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi"), ("All files", "*.*")]
        )
        
        if video_path:
            try:
                labeler = AutoLabelWindow(self.root, video_path, None, None)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open auto-labeler:\n{str(e)}")
    
    def open_quality_checker(self):
        """Open data quality checker"""
        if not self.train_project_folder.get():
            messagebox.showwarning("No Folder", "Please select a project folder first.")
            return
        
        try:
            # Find actual sessions in the project folder
            sessions = self.find_training_sessions()
            
            if not sessions:
                messagebox.showwarning("No Sessions", 
                    "No training sessions found in project folder.\n\n"
                    "Please ensure your folder contains:\n"
                    "• Videos/ subfolder with .h5 DLC files and videos\n"
                    "• Targets/ subfolder with .csv label files\n\n"
                    "Or enable 'Single Folder' mode if all files are in one place.")
                return
            
            checker = DataQualityChecker(self.root, sessions)
        except Exception as e:
            messagebox.showerror("Error", f"Could not run quality check:\n{str(e)}")
    
    def run_brightness_diagnostics(self):
        """Run brightness diagnostics with GUI integration"""
        # Import the necessary modules
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            messagebox.showerror(
                "Missing Dependencies",
                "Matplotlib and Seaborn are required.\n\n"
                "Install with:\npip install matplotlib seaborn"
            )
            return
        
        # Ask user to select features file
        file_path = filedialog.askopenfilename(
            title="Select Features File",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Create progress window in main thread
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Brightness Diagnostics")
        _sw, _sh = progress_window.winfo_screenwidth(), progress_window.winfo_screenheight()
        progress_window.geometry(f"550x220+{(_sw-550)//2}+{(_sh-220)//2}")
        progress_window.transient(self.root)
        
        ttk.Label(progress_window, 
                 text="Analyzing Brightness Features...",
                 font=('Arial', 12, 'bold')).pack(pady=20)
        
        progress_text = scrolledtext.ScrolledText(progress_window, height=6, width=60)
        progress_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Run analysis in thread
        threading.Thread(
            target=self._run_brightness_diagnostics_thread,
            args=(file_path, progress_window, progress_text),
            daemon=True
        ).start()
    
    def _run_brightness_diagnostics_thread(self, file_path, progress_window, progress_text):
        """Run brightness diagnostics analysis in background thread"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use Tkinter backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            def log(msg):
                self.root.after(0, lambda m=msg: progress_text.insert(tk.END, m + "\n"))
                self.root.after(0, lambda: progress_text.see(tk.END))
                self.root.after(0, lambda: progress_window.update())
            
            log(f"Loading: {os.path.basename(file_path)}")
            
            # Load features
            with open(file_path, 'rb') as f:
                features_df = pickle.load(f)
            
            if not isinstance(features_df, pd.DataFrame):
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Expected DataFrame, got {type(features_df)}"))
                self.root.after(0, lambda: progress_window.destroy())
                return
            
            log(f"✓ Loaded {len(features_df)} frames, {len(features_df.columns)} features")
            
            # Identify brightness features (handle multiple naming conventions)
            brightness_cols = []
            
            for col in features_df.columns:
                col_lower = col.lower()
                # Check multiple patterns:
                # 1. Standard: bodypart_pixbrt_stat
                # 2. Alternative: Pix_bodypart_stat
                # 3. Log transform: Log10(Pix_bodypart_stat)
                # 4. Derivative: |d/dt(Pix_bodypart_stat)
                if any(pattern in col_lower for pattern in ['pixbrt', 'brightness', 'pix_brt']):
                    brightness_cols.append(col)
                elif col.startswith('Pix_') or col.startswith('Log10(Pix_') or \
                     col.startswith('|d/dt(Pix_') or col.startswith('|d/dt(Log10(Pix_'):
                    brightness_cols.append(col)
            
            if not brightness_cols:
                self.root.after(0, lambda: messagebox.showerror(
                    "No Brightness Features",
                    "No brightness features found in this file.\n\n"
                    "Brightness features can be named in several ways:\n\n"
                    "Standard naming:\n"
                    "  • bodypart_pixbrt_mean\n"
                    "  • bodypart_pixbrt_std\n"
                    "  • bodypart_pixbrt_median\n\n"
                    "Alternative naming:\n"
                    "  • Pix_bodypart_stat\n"
                    "  • Log10(Pix_bodypart_stat)\n"
                    "  • |d/dt(Pix_bodypart_stat)\n\n"
                    "Make sure you extracted brightness features during\n"
                    "feature extraction by specifying body parts in the\n"
                    "'bp_pixbrt_list' parameter."))
                self.root.after(0, lambda: progress_window.destroy())
                return
            
            # Extract bodyparts (handle different naming conventions)
            bodyparts = set()
            for col in brightness_cols:
                # Remove special characters and split
                col_clean = col.replace('(', '_').replace(')', '_').replace('|', '')
                parts = col_clean.split('_')
                
                # Skip common prefixes
                skip_prefixes = ['Pix', 'Log10', 'd', 'dt', 'sum', 'centroid']
                
                for part in parts:
                    part = part.strip()
                    if part and not part.isdigit() and len(part) > 1:
                        if part not in skip_prefixes:
                            bodyparts.add(part)
            
            bodyparts = sorted(bodyparts)
            
            log(f"✓ Found {len(brightness_cols)} brightness features")
            log(f"  Body parts: {', '.join(bodyparts)}")
            
            # Calculate statistics
            log("Calculating statistics...")
            stats = []
            for col in brightness_cols:
                data = features_df[col].values
                stat = {
                    'Feature': col,
                    'Min': np.min(data),
                    'Max': np.max(data),
                    'Mean': np.mean(data),
                    'Std': np.std(data),
                    'Range': np.ptp(data),
                    'CV': np.std(data) / np.mean(data) if np.mean(data) > 0 else 0,
                }
                stats.append(stat)
            
            stats_df = pd.DataFrame(stats)
            
            # Create output directory
            output_dir = os.path.join(os.path.dirname(file_path), 'brightness_diagnostics')
            os.makedirs(output_dir, exist_ok=True)
            log(f"✓ Saving to: {output_dir}")
            
            # Close progress window (must be done in main thread)
            self.root.after(0, lambda: progress_window.destroy())
            
            # Generate plots in GUI windows
            
            # Import brightness_diagnostics functions
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, 'brightness_diagnostics.py')
            
            if os.path.isfile(script_path):
                # Import functions from script
                import importlib.util
                spec = importlib.util.spec_from_file_location("brightness_diagnostics", script_path)
                bd_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bd_module)
                
                # Generate plots using the script's functions (saves to disk)
                bd_module.plot_statistics_table(
                    stats_df,
                    os.path.join(output_dir, '1_statistics_table.png')
                )
                
                bd_module.plot_histograms(
                    features_df,
                    brightness_cols,
                    bodyparts,
                    os.path.join(output_dir, '2_brightness_histograms.png')
                )
                
                bd_module.plot_temporal_brightness(
                    features_df,
                    brightness_cols,
                    bodyparts,
                    fps=30,
                    save_path=os.path.join(output_dir, '3_temporal_brightness.png')
                )
                
                if len(brightness_cols) <= 50:
                    bd_module.plot_correlation_matrix(
                        features_df,
                        brightness_cols,
                        os.path.join(output_dir, '4_correlation_matrix.png')
                    )
                
                bd_module.generate_report(
                    features_df,
                    brightness_cols,
                    bodyparts,
                    stats_df,
                    output_dir
                )
            
            # Display results in GUI window
            self.root.after(0, lambda: self._show_brightness_results_window(
                output_dir, stats_df, bodyparts))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"Error during analysis:\n\n{str(e)}"))
            import traceback
            traceback.print_exc()
    
    def _show_brightness_results_window(self, output_dir, stats_df, bodyparts):
        """Display brightness diagnostics results in GUI window"""
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Brightness Diagnostics Results")
        sw, sh = results_window.winfo_screenwidth(), results_window.winfo_screenheight()
        w, h = int(sw * 0.65), int(sh * 0.75)
        results_window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Create notebook for different views
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Summary Statistics
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 Summary")
        
        # Summary text
        summary_text = scrolledtext.ScrolledText(summary_frame, height=15, wrap=tk.WORD)
        summary_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        summary_text.insert(tk.END, "=" * 80 + "\n")
        summary_text.insert(tk.END, "BRIGHTNESS FEATURE DIAGNOSTICS - SUMMARY\n")
        summary_text.insert(tk.END, "=" * 80 + "\n\n")
        
        summary_text.insert(tk.END, f"Total brightness features: {len(stats_df)}\n")
        summary_text.insert(tk.END, f"Body parts analyzed: {', '.join(bodyparts)}\n\n")
        
        summary_text.insert(tk.END, "STATISTICS BY BODY PART:\n")
        summary_text.insert(tk.END, "-" * 80 + "\n\n")
        
        for bodypart in bodyparts:
            bp_stats = stats_df[stats_df['Feature'].str.startswith(bodypart)]
            if len(bp_stats) == 0:
                continue
            
            mean_range = bp_stats['Range'].mean()
            mean_std = bp_stats['Std'].mean()
            
            summary_text.insert(tk.END, f"{bodypart.upper()}:\n")
            summary_text.insert(tk.END, f"  Average Range: {mean_range:.2f}\n")
            summary_text.insert(tk.END, f"  Average Std Dev: {mean_std:.2f}\n")
            
            if mean_range < 10:
                status = "❌ VERY LOW - Not informative"
            elif mean_range < 30:
                status = "⚠️  LOW - May not be informative"
            elif mean_range < 50:
                status = "✓ MODERATE - Usable"
            else:
                status = "✓✓ GOOD - Highly informative"
            
            summary_text.insert(tk.END, f"  Status: {status}\n\n")
        
        summary_text.insert(tk.END, "\n" + "=" * 80 + "\n")
        summary_text.insert(tk.END, "INTERPRETATION GUIDE\n")
        summary_text.insert(tk.END, "=" * 80 + "\n\n")
        summary_text.insert(tk.END, "Dynamic Range Categories:\n")
        summary_text.insert(tk.END, "  • < 10:  Very Low - Features likely not informative\n")
        summary_text.insert(tk.END, "  • 10-30: Low - Features may have limited utility\n")
        summary_text.insert(tk.END, "  • 30-50: Moderate - Features should be useful\n")
        summary_text.insert(tk.END, "  • > 50:  Good - Features are highly informative\n\n")
        
        summary_text.config(state='disabled')
        
        # Tab 2-5: Plot Images
        plot_files = [
            ('1_statistics_table.png', '📈 Statistics Table'),
            ('2_brightness_histograms.png', '📊 Histograms'),
            ('3_temporal_brightness.png', '⏱️ Temporal'),
            ('4_correlation_matrix.png', '🔗 Correlation'),
        ]
        
        for filename, tab_name in plot_files:
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                self._add_image_tab(notebook, file_path, tab_name)
        
        # Bottom buttons
        button_frame = ttk.Frame(results_window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="📁 Open Results Folder",
            command=lambda: self._open_folder(output_dir)
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="📄 View Full Report",
            command=lambda: self._open_text_file(os.path.join(output_dir, 'brightness_report.txt'))
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Close",
            command=results_window.destroy
        ).pack(side='right', padx=5)
        
        # Show success message
        messagebox.showinfo("Brightness Diagnostics Complete", 
                           f"Analysis complete!\n\nResults saved to:\n{output_dir}")
    
    def show_brightness_preview(self):
        """Launch brightness preview tool"""
        # Check if brightness_preview.py exists
        script_path = os.path.join(os.path.dirname(__file__), 'brightness_preview.py')
        
        if not os.path.isfile(script_path):
            messagebox.showerror("Script Not Found",
                               f"brightness_preview.py not found at:\n{script_path}\n\n"
                               "Please ensure the script is in the same directory as the GUI.")
            return
        
        # Get current session to auto-detect video and features
        video_path = None
        features_path = None
        
        if hasattr(self, 'current_video') and self.current_video:
            video_path = self.current_video
            
            # Auto-detect features file
            video_dir = os.path.dirname(video_path)
            video_base = os.path.splitext(os.path.basename(video_path))[0]
            
            # Remove DLC suffixes
            for suffix in ['DLC', '_labeled', '_filtered']:
                if suffix in video_base:
                    video_base = video_base.split(suffix)[0]
            
            features_file = os.path.join(video_dir, f"{video_base}_features.pickle")
            if os.path.isfile(features_file):
                features_path = features_file
        
        # Launch standalone script
        try:
            import subprocess
            
            if video_path and features_path:
                # Launch with auto-detected files
                subprocess.Popen([sys.executable, script_path, video_path, features_path])
            else:
                # Launch without files (user will select)
                subprocess.Popen([sys.executable, script_path])
            
            # Update status if available
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Brightness Preview tool launched")
            
        except Exception as e:
            messagebox.showerror("Launch Error",
                               f"Failed to launch brightness preview:\n{str(e)}")
    
    def correct_crop_offset_single(self):
        """Launch crop offset correction tool for single file"""
        script_path = os.path.join(os.path.dirname(__file__), 'correct_features_crop.py')
        
        if not os.path.isfile(script_path):
            messagebox.showerror("Script Not Found",
                               f"correct_features_crop.py not found at:\n{script_path}\n\n"
                               "Please ensure the script is in the same directory as the GUI.")
            return
        
        try:
            import subprocess
            subprocess.Popen([sys.executable, script_path])
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Crop correction tool launched")
        except Exception as e:
            messagebox.showerror("Launch Error",
                               f"Failed to launch crop correction:\n{str(e)}")
    
    def correct_crop_offset_batch(self):
        """Launch crop offset correction tool in batch mode"""
        script_path = os.path.join(os.path.dirname(__file__), 'correct_features_crop.py')
        
        if not os.path.isfile(script_path):
            messagebox.showerror("Script Not Found",
                               f"correct_features_crop.py not found at:\n{script_path}\n\n"
                               "Please ensure the script is in the same directory as the GUI.")
            return
        
        try:
            import subprocess
            subprocess.Popen([sys.executable, script_path, '--batch'])
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Batch crop correction tool launched")
        except Exception as e:
            messagebox.showerror("Launch Error",
                               f"Failed to launch batch correction:\n{str(e)}")

    def crop_video_for_dlc(self):
        """Launch the crop-for-DLC standalone tool."""
        script_path = os.path.join(os.path.dirname(__file__), 'crop_for_dlc.py')
        if not os.path.isfile(script_path):
            messagebox.showerror("Not Found",
                                 f"crop_for_dlc.py not found at:\n{script_path}")
            return
        try:
            import subprocess
            project = self.current_project_folder.get()
            cmd = [sys.executable, script_path]
            if project:
                cmd += ['--project', project]
            subprocess.Popen(cmd)
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Crop Video for DLC tool launched")
        except Exception as e:
            messagebox.showerror("Launch Error",
                                 f"Failed to launch crop tool:\n{str(e)}")

    def _add_image_tab(self, notebook, image_path, tab_name):
        """Add an image as a tab in the notebook"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab_name)
        
        try:
            from PIL import Image, ImageTk
            
            # Load image first to check if it exists
            if not os.path.isfile(image_path):
                ttk.Label(frame, text=f"Image file not found:\n{image_path}").pack(pady=20)
                return
            
            img = Image.open(image_path)
            photo = ImageTk.PhotoImage(img)
            
            # Create canvas
            canvas = tk.Canvas(frame, bg='white')
            
            # Create scrollbars
            h_scrollbar = ttk.Scrollbar(frame, orient='horizontal')
            v_scrollbar = ttk.Scrollbar(frame, orient='vertical')
            
            # Configure canvas scrolling
            canvas.configure(
                xscrollcommand=h_scrollbar.set,
                yscrollcommand=v_scrollbar.set
            )
            
            # Configure scrollbar commands
            h_scrollbar.configure(command=canvas.xview)
            v_scrollbar.configure(command=canvas.yview)
            
            # Grid layout (better for scrollbars)
            canvas.grid(row=0, column=0, sticky='nsew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            
            # Configure grid weights
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            
            # Add image to canvas
            canvas.create_image(0, 0, anchor='nw', image=photo)
            canvas.image = photo  # Keep a reference
            
            # Update scroll region after image is added
            frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox('all'))
            
        except ImportError:
            ttk.Label(frame, 
                     text="PIL/Pillow not installed.\n\nInstall with: pip install Pillow\n\n"
                          f"Image saved to:\n{image_path}",
                     justify='center').pack(pady=20)
        except Exception as e:
            ttk.Label(frame, text=f"Could not load image:\n{str(e)}").pack(pady=20)
    
    def inspect_features_file(self):
        """Inspect contents of a features pickle file"""
        # Ask user to select features file
        file_path = filedialog.askopenfilename(
            title="Select Features File to Inspect",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Run inspection in thread
        threading.Thread(
            target=self._inspect_features_file_thread,
            args=(file_path,),
            daemon=True
        ).start()
    
    def _inspect_features_file_thread(self, file_path):
        """Inspect features file in background thread"""
        try:
            # Load file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Analyze contents
            analysis = self._analyze_features_data(data, file_path)
            
            # Display results
            self.root.after(0, lambda: self._show_features_inspector_window(analysis, file_path))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Inspection Error",
                f"Could not inspect file:\n\n{str(e)}\n\n"
                f"Make sure this is a valid features pickle file."))
            import traceback
            traceback.print_exc()
    
    def _analyze_features_data(self, data, file_path):
        """Analyze features data structure"""
        analysis = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'data_type': str(type(data).__name__),
            'is_dataframe': isinstance(data, pd.DataFrame),
        }
        
        if isinstance(data, pd.DataFrame):
            # DataFrame analysis
            analysis['n_frames'] = len(data)
            analysis['n_features'] = len(data.columns)
            analysis['columns'] = list(data.columns)
            analysis['dtypes'] = data.dtypes.to_dict()
            analysis['memory_usage'] = data.memory_usage(deep=True).sum()
            
            # Categorize features (handle multiple naming conventions)
            pose_features = []
            brightness_features = []
            velocity_features = []
            angle_features = []
            distance_features = []
            other_features = []
            
            for col in data.columns:
                col_lower = col.lower()
                categorized = False
                
                # Brightness features (multiple patterns)
                if any(pattern in col_lower for pattern in ['pixbrt', 'brightness', 'pix_brt', '/pix', 'pix_']):
                    brightness_features.append(col)
                    categorized = True
                # Also check for Pix patterns (PixelPaws alternative naming)
                elif col.startswith('Pix_') or col.startswith('Log10(Pix_') or col.startswith('|d/dt(Pix_') or col.startswith('|d/dt(Log10(Pix_'):
                    brightness_features.append(col)
                    categorized = True
                
                # Velocity features (multiple patterns)
                if not categorized and any(pattern in col_lower for pattern in ['velocity', '_vel', 'd/dt', '|d/dt', 'vel2']):
                    velocity_features.append(col)
                    categorized = True
                
                # Angle features (multiple patterns)
                if not categorized and (col.startswith('Ang_') or 'angle' in col_lower):
                    angle_features.append(col)
                    categorized = True
                
                # Distance features (multiple patterns)
                if not categorized and (col.startswith('Dis_') or 'distance' in col_lower or 'dist_' in col_lower):
                    distance_features.append(col)
                    categorized = True
                
                # Pose features - be more flexible
                # Standard: bodypart_x, bodypart_y, bodypart_likelihood
                # Alternative: compound names like "hrpaw-flpaw" (distances/angles between bodyparts)
                if not categorized:
                    # Check if it looks like a raw pose/geometric feature
                    # - Contains bodypart names (hrpaw, hlpaw, etc.)
                    # - NOT a velocity/angle/distance (already categorized)
                    # - Likely represents spatial relationships
                    
                    if any(pattern in col_lower for pattern in ['_x', '_y', '_likelihood']):
                        pose_features.append(col)
                        categorized = True
                    # Also consider compound features with bodypart names as pose-derived
                    elif any(bp in col_lower for bp in ['paw', 'snout', 'tail', 'neck', 'centroid']):
                        # If it has bodypart names but isn't velocity/angle/distance, it's a pose feature
                        # These are typically spatial relationships between bodyparts
                        if '-' in col or 'centroid' in col_lower:
                            pose_features.append(col)
                            categorized = True
                
                if not categorized:
                    other_features.append(col)
            
            analysis['pose_features'] = pose_features
            analysis['brightness_features'] = brightness_features
            analysis['velocity_features'] = velocity_features
            analysis['angle_features'] = angle_features
            analysis['distance_features'] = distance_features
            analysis['other_features'] = other_features
            
            # Extract body parts (handle different naming conventions)
            bodyparts = set()
            base_bodyparts = ['hrpaw', 'hlpaw', 'frpaw', 'flpaw', 'snout', 'neck', 'tailbase', 'tailtip', 'centroid']
            
            for col in data.columns:
                col_lower = col.lower()
                # Check for base bodyparts in the feature name
                for bp in base_bodyparts:
                    if bp in col_lower:
                        bodyparts.add(bp)
            
            analysis['bodyparts'] = sorted(bodyparts)
            
            # Statistics
            analysis['has_nans'] = data.isnull().any().any()
            analysis['has_infs'] = np.isinf(data.select_dtypes(include=[np.number])).any().any()
            
            # Detailed feature quality analysis
            feature_quality = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = data[col]
                quality = {
                    'feature': col,
                    'n_values': len(col_data),
                    'n_nans': col_data.isnull().sum(),
                    'pct_nans': (col_data.isnull().sum() / len(col_data)) * 100,
                    'n_infs': np.isinf(col_data).sum(),
                    'n_unique': col_data.nunique(),
                    'is_constant': col_data.nunique() <= 1,
                    'variance': col_data.var() if col_data.notna().sum() > 0 else 0,
                    'zero_variance': col_data.var() < 1e-10 if col_data.notna().sum() > 0 else True,
                }
                
                # Check if feature is all NaN
                quality['all_nan'] = quality['n_nans'] == quality['n_values']
                
                # Check if feature has issues
                quality['has_issues'] = (
                    quality['all_nan'] or 
                    quality['is_constant'] or 
                    quality['zero_variance'] or 
                    quality['pct_nans'] > 50 or
                    quality['n_infs'] > 0
                )
                
                feature_quality.append(quality)
            
            analysis['feature_quality'] = pd.DataFrame(feature_quality)
            
            # Summary statistics on feature quality
            quality_df = analysis['feature_quality']
            analysis['n_all_nan_features'] = quality_df['all_nan'].sum()
            analysis['n_constant_features'] = quality_df['is_constant'].sum()
            analysis['n_zero_variance_features'] = quality_df['zero_variance'].sum()
            analysis['n_high_nan_features'] = (quality_df['pct_nans'] > 50).sum()
            analysis['n_features_with_infs'] = (quality_df['n_infs'] > 0).sum()
            analysis['n_problematic_features'] = quality_df['has_issues'].sum()
            
            # Also count features with ANY NaN (even if <50%)
            analysis['n_features_with_any_nan'] = (quality_df['n_nans'] > 0).sum()
            analysis['total_nan_cells'] = quality_df['n_nans'].sum()
            analysis['total_cells'] = quality_df['n_values'].sum()
            analysis['overall_nan_pct'] = (analysis['total_nan_cells'] / analysis['total_cells']) * 100 if analysis['total_cells'] > 0 else 0
            
            if len(data) > 0:
                if len(numeric_cols) > 0:
                    analysis['numeric_summary'] = {
                        'min': data[numeric_cols].min().min(),
                        'max': data[numeric_cols].max().max(),
                        'mean': data[numeric_cols].mean().mean(),
                    }
        
        else:
            # Non-DataFrame data
            analysis['raw_type'] = str(type(data))
            if hasattr(data, 'shape'):
                analysis['shape'] = data.shape
            if hasattr(data, '__len__'):
                analysis['length'] = len(data)
        
        return analysis
    
    def _show_features_inspector_window(self, analysis, file_path):
        """Display features inspection results"""
        # Create window
        window = tk.Toplevel(self.root)
        window.title(f"Feature File Inspector - {os.path.basename(file_path)}")
        sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.75)
        window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Title
        title_frame = ttk.Frame(window)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(title_frame, 
                 text="📋 Feature File Inspector",
                 font=('Arial', 14, 'bold')).pack(side='left')
        
        # Create notebook
        notebook = ttk.Notebook(window)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 Summary")
        
        summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, font=('Courier', 10))
        summary_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Write summary
        summary_text.insert(tk.END, "=" * 80 + "\n")
        summary_text.insert(tk.END, "FEATURE FILE INSPECTION REPORT\n")
        summary_text.insert(tk.END, "=" * 80 + "\n\n")
        
        summary_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
        summary_text.insert(tk.END, f"Location: {os.path.dirname(file_path)}\n")
        summary_text.insert(tk.END, f"Size: {analysis['file_size'] / (1024*1024):.2f} MB\n")
        summary_text.insert(tk.END, f"Type: {analysis['data_type']}\n\n")
        
        if analysis['is_dataframe']:
            summary_text.insert(tk.END, "✓ Valid DataFrame format\n\n")
            summary_text.insert(tk.END, f"Frames: {analysis['n_frames']:,}\n")
            summary_text.insert(tk.END, f"Features: {analysis['n_features']:,}\n")
            summary_text.insert(tk.END, f"Memory: {analysis['memory_usage'] / (1024*1024):.2f} MB\n\n")
            
            summary_text.insert(tk.END, "-" * 80 + "\n")
            summary_text.insert(tk.END, "FEATURE BREAKDOWN\n")
            summary_text.insert(tk.END, "-" * 80 + "\n\n")
            
            summary_text.insert(tk.END, f"Pose Features:       {len(analysis['pose_features']):4d}\n")
            summary_text.insert(tk.END, f"Brightness Features: {len(analysis['brightness_features']):4d}")
            if len(analysis['brightness_features']) == 0:
                summary_text.insert(tk.END, "  ⚠️  None found!\n")
            else:
                summary_text.insert(tk.END, "  ✓\n")
            summary_text.insert(tk.END, f"Velocity Features:   {len(analysis['velocity_features']):4d}\n")
            summary_text.insert(tk.END, f"Angle Features:      {len(analysis['angle_features']):4d}\n")
            summary_text.insert(tk.END, f"Distance Features:   {len(analysis['distance_features']):4d}\n")
            summary_text.insert(tk.END, f"Other Features:      {len(analysis['other_features']):4d}\n\n")
            
            if analysis['bodyparts']:
                summary_text.insert(tk.END, "-" * 80 + "\n")
                summary_text.insert(tk.END, "BODY PARTS\n")
                summary_text.insert(tk.END, "-" * 80 + "\n\n")
                for bp in analysis['bodyparts']:
                    summary_text.insert(tk.END, f"  • {bp}\n")
                summary_text.insert(tk.END, "\n")
            
            summary_text.insert(tk.END, "-" * 80 + "\n")
            summary_text.insert(tk.END, "DATA QUALITY\n")
            summary_text.insert(tk.END, "-" * 80 + "\n\n")
            summary_text.insert(tk.END, f"Contains NaN values: {'Yes ⚠️' if analysis['has_nans'] else 'No ✓'}\n")
            summary_text.insert(tk.END, f"Contains Inf values: {'Yes ⚠️' if analysis['has_infs'] else 'No ✓'}\n")
            
            # Show NaN statistics
            if analysis.get('n_features_with_any_nan', 0) > 0:
                summary_text.insert(tk.END, f"\nNaN Statistics:\n")
                summary_text.insert(tk.END, f"  Features with any NaN: {analysis['n_features_with_any_nan']}/{analysis['n_features']}\n")
                summary_text.insert(tk.END, f"  Overall NaN percentage: {analysis.get('overall_nan_pct', 0):.2f}%\n")
            
            summary_text.insert(tk.END, "\n")
            
            # Feature quality issues
            if analysis.get('n_problematic_features', 0) > 0:
                summary_text.insert(tk.END, "⚠️  SEVERE FEATURE QUALITY ISSUES:\n\n")
                
                if analysis.get('n_all_nan_features', 0) > 0:
                    summary_text.insert(tk.END, f"  • {analysis['n_all_nan_features']} features are ALL NaN (100% missing)\n")
                
                if analysis.get('n_constant_features', 0) > 0:
                    summary_text.insert(tk.END, f"  • {analysis['n_constant_features']} features are constant (only 1 unique value)\n")
                
                if analysis.get('n_zero_variance_features', 0) > 0:
                    summary_text.insert(tk.END, f"  • {analysis['n_zero_variance_features']} features have zero variance\n")
                
                if analysis.get('n_high_nan_features', 0) > 0:
                    summary_text.insert(tk.END, f"  • {analysis['n_high_nan_features']} features have >50% NaN values\n")
                
                if analysis.get('n_features_with_infs', 0) > 0:
                    summary_text.insert(tk.END, f"  • {analysis['n_features_with_infs']} features contain Inf values\n")
                
                summary_text.insert(tk.END, f"\n  Total severe issues: {analysis['n_problematic_features']}/{analysis['n_features']}\n")
                summary_text.insert(tk.END, "  → See 'Feature Quality' tab for details\n\n")
            elif analysis.get('n_features_with_any_nan', 0) > 0:
                summary_text.insert(tk.END, "⚠️  MINOR QUALITY ISSUES:\n\n")
                summary_text.insert(tk.END, f"  • {analysis['n_features_with_any_nan']} features have some NaN values\n")
                summary_text.insert(tk.END, f"  • Overall {analysis.get('overall_nan_pct', 0):.2f}% of data points are NaN\n")
                summary_text.insert(tk.END, "  • No features are completely NaN or constant\n")
                summary_text.insert(tk.END, "\n  → These may be acceptable depending on your use case\n")
                summary_text.insert(tk.END, "  → See 'Feature Quality' tab for details\n\n")
            else:
                summary_text.insert(tk.END, "✓ No major feature quality issues detected\n\n")
            
            if 'numeric_summary' in analysis:
                summary_text.insert(tk.END, f"Value Range:\n")
                summary_text.insert(tk.END, f"  Min:  {analysis['numeric_summary']['min']:.4f}\n")
                summary_text.insert(tk.END, f"  Max:  {analysis['numeric_summary']['max']:.4f}\n")
                summary_text.insert(tk.END, f"  Mean: {analysis['numeric_summary']['mean']:.4f}\n")
        else:
            summary_text.insert(tk.END, "⚠️  NOT a DataFrame\n\n")
            summary_text.insert(tk.END, f"This file contains: {analysis['raw_type']}\n")
            if 'shape' in analysis:
                summary_text.insert(tk.END, f"Shape: {analysis['shape']}\n")
            if 'length' in analysis:
                summary_text.insert(tk.END, f"Length: {analysis['length']}\n")
        
        summary_text.config(state='disabled')
        
        # Tab 2: Feature List (only if DataFrame)
        if analysis['is_dataframe']:
            features_frame = ttk.Frame(notebook)
            notebook.add(features_frame, text="📝 All Features")
            
            # Create treeview
            tree_frame = ttk.Frame(features_frame)
            tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
            
            tree_scroll = ttk.Scrollbar(tree_frame)
            tree_scroll.pack(side='right', fill='y')
            
            tree = ttk.Treeview(tree_frame, 
                               columns=('Feature', 'Type', 'Category'),
                               show='headings',
                               yscrollcommand=tree_scroll.set)
            tree_scroll.config(command=tree.yview)
            
            tree.heading('Feature', text='Feature Name')
            tree.heading('Type', text='Data Type')
            tree.heading('Category', text='Category')
            
            tree.column('Feature', width=400)
            tree.column('Type', width=100)
            tree.column('Category', width=150)
            
            # Add features
            for col in analysis['columns']:
                dtype = str(analysis['dtypes'][col])
                
                # Determine category
                if col in analysis['pose_features']:
                    category = 'Pose'
                elif col in analysis['brightness_features']:
                    category = 'Brightness'
                elif col in analysis['velocity_features']:
                    category = 'Velocity'
                elif col in analysis['angle_features']:
                    category = 'Angle'
                elif col in analysis['distance_features']:
                    category = 'Distance'
                else:
                    category = 'Other'
                
                tree.insert('', 'end', values=(col, dtype, category))
            
            tree.pack(fill='both', expand=True)
            
            # Search box
            search_frame = ttk.Frame(features_frame)
            search_frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
            search_var = tk.StringVar()
            search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
            search_entry.pack(side='left', padx=5)
            
            def search_features(*args):
                query = search_var.get().lower()
                # Clear current items
                for item in tree.get_children():
                    tree.delete(item)
                # Re-add matching items
                for col in analysis['columns']:
                    if query in col.lower():
                        dtype = str(analysis['dtypes'][col])
                        if col in analysis['pose_features']:
                            category = 'Pose'
                        elif col in analysis['brightness_features']:
                            category = 'Brightness'
                        elif col in analysis['velocity_features']:
                            category = 'Velocity'
                        elif col in analysis['angle_features']:
                            category = 'Angle'
                        elif col in analysis['distance_features']:
                            category = 'Distance'
                        else:
                            category = 'Other'
                        tree.insert('', 'end', values=(col, dtype, category))
            
            search_var.trace('w', search_features)
            
            ttk.Label(search_frame, 
                     text=f"Total: {len(analysis['columns'])} features",
                     foreground='gray').pack(side='right', padx=10)
            
            # Tab 3: Feature Quality (only if DataFrame with quality analysis)
            if 'feature_quality' in analysis:
                quality_frame = ttk.Frame(notebook)
                notebook.add(quality_frame, text="⚠️ Feature Quality")
                
                # Filter to features with issues OR any NaN
                quality_df = analysis['feature_quality']
                issues_to_show = quality_df[
                    (quality_df['has_issues'] == True) | (quality_df['n_nans'] > 0)
                ].copy()
                
                if len(issues_to_show) > 0:
                    # Create treeview
                    quality_tree_frame = ttk.Frame(quality_frame)
                    quality_tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
                    
                    quality_tree_scroll = ttk.Scrollbar(quality_tree_frame)
                    quality_tree_scroll.pack(side='right', fill='y')
                    
                    quality_tree = ttk.Treeview(
                        quality_tree_frame,
                        columns=('Feature', 'Issue', 'NaN %', 'Variance', 'Unique'),
                        show='headings',
                        yscrollcommand=quality_tree_scroll.set
                    )
                    quality_tree_scroll.config(command=quality_tree.yview)
                    
                    quality_tree.heading('Feature', text='Feature Name')
                    quality_tree.heading('Issue', text='Issue Type')
                    quality_tree.heading('NaN %', text='NaN %')
                    quality_tree.heading('Variance', text='Variance')
                    quality_tree.heading('Unique', text='Unique Values')
                    
                    quality_tree.column('Feature', width=300)
                    quality_tree.column('Issue', width=150)
                    quality_tree.column('NaN %', width=80)
                    quality_tree.column('Variance', width=100)
                    quality_tree.column('Unique', width=100)
                    
                    # Add features with issues
                    for _, row in issues_to_show.iterrows():
                        # Determine issue type
                        issues = []
                        if row['all_nan']:
                            issues.append('ALL NaN')
                        elif row['pct_nans'] > 50:
                            issues.append(f'{row["pct_nans"]:.1f}% NaN')
                        elif row['n_nans'] > 0:
                            issues.append(f'{row["pct_nans"]:.1f}% NaN')
                        
                        if row['is_constant']:
                            issues.append('Constant')
                        elif row['zero_variance']:
                            issues.append('Zero Var')
                        
                        if row['n_infs'] > 0:
                            issues.append(f'{row["n_infs"]} Infs')
                        
                        issue_str = ', '.join(issues) if issues else 'Minor NaN'
                        
                        quality_tree.insert('', 'end', values=(
                            row['feature'],
                            issue_str,
                            f"{row['pct_nans']:.1f}%",
                            f"{row['variance']:.6f}",
                            row['n_unique']
                        ))
                    
                    quality_tree.pack(fill='both', expand=True)
                    
                    # Summary at bottom
                    summary_frame = ttk.Frame(quality_frame)
                    summary_frame.pack(fill='x', padx=5, pady=5)
                    
                    n_severe = analysis.get('n_problematic_features', 0)
                    n_total_issues = len(issues_to_show)
                    
                    if n_severe > 0:
                        summary_label = ttk.Label(
                            summary_frame,
                            text=f"Showing {n_total_issues} features with quality issues ({n_severe} severe) out of {len(quality_df)} total features",
                            font=('Arial', 10, 'bold'),
                            foreground='red'
                        )
                    else:
                        summary_label = ttk.Label(
                            summary_frame,
                            text=f"Showing {n_total_issues} features with some NaN values out of {len(quality_df)} total features",
                            font=('Arial', 10, 'bold'),
                            foreground='orange'
                        )
                    summary_label.pack(pady=5)
                    
                    # Recommendations
                    rec_text = scrolledtext.ScrolledText(summary_frame, height=6, wrap=tk.WORD)
                    rec_text.pack(fill='x', padx=5, pady=5)
                    
                    rec_text.insert(tk.END, "RECOMMENDATIONS:\n\n")
                    
                    if analysis.get('n_all_nan_features', 0) > 0:
                        rec_text.insert(tk.END, f"• {analysis['n_all_nan_features']} features are completely NaN\n")
                        rec_text.insert(tk.END, "  → These should be REMOVED - they contain no information\n")
                        rec_text.insert(tk.END, "  → This often happens with derivative features (|d/dt) when calculation fails\n\n")
                    
                    if analysis.get('n_constant_features', 0) > 0:
                        rec_text.insert(tk.END, f"• {analysis['n_constant_features']} features are constant\n")
                        rec_text.insert(tk.END, "  → These should be REMOVED - they don't vary\n\n")
                    
                    if analysis.get('n_zero_variance_features', 0) > 0:
                        rec_text.insert(tk.END, f"• {analysis['n_zero_variance_features']} features have zero variance\n")
                        rec_text.insert(tk.END, "  → Consider removing these features\n\n")
                    
                    rec_text.config(state='disabled')
                    
                else:
                    ttk.Label(quality_frame,
                             text="✓ No problematic features detected!",
                             font=('Arial', 14, 'bold'),
                             foreground='green').pack(expand=True)
            
            # Tab 4: Feature Data Viewer (only if DataFrame)
            if analysis['is_dataframe']:
                viewer_frame = ttk.Frame(notebook)
                notebook.add(viewer_frame, text="🔍 Data Viewer")
                
                # Instructions
                ttk.Label(viewer_frame,
                         text="Select a feature to view its data values",
                         font=('Arial', 12, 'bold')).pack(pady=10)
                
                # Feature selector
                selector_frame = ttk.Frame(viewer_frame)
                selector_frame.pack(fill='x', padx=10, pady=5)
                
                ttk.Label(selector_frame, text="Feature:").pack(side='left', padx=5)
                
                feature_var = tk.StringVar()
                feature_combo = ttk.Combobox(selector_frame,
                                            textvariable=feature_var,
                                            values=analysis['columns'],
                                            width=50,
                                            state='readonly')
                feature_combo.pack(side='left', padx=5, fill='x', expand=True)
                
                def show_feature_data():
                    feature = feature_var.get()
                    if not feature:
                        return
                    
                    # Clear previous data
                    for widget in data_display_frame.winfo_children():
                        widget.destroy()
                    
                    # Get feature data
                    try:
                        # Load the actual dataframe
                        with open(file_path, 'rb') as f:
                            df = pickle.load(f)
                        
                        if feature not in df.columns:
                            ttk.Label(data_display_frame,
                                     text=f"Feature '{feature}' not found!",
                                     foreground='red').pack(pady=20)
                            return
                        
                        feature_data = df[feature]
                        
                        # Create display
                        info_frame = ttk.Frame(data_display_frame)
                        info_frame.pack(fill='x', padx=10, pady=10)
                        
                        # Statistics
                        n_total = len(feature_data)
                        n_nans = feature_data.isnull().sum()
                        n_valid = n_total - n_nans
                        pct_nan = (n_nans / n_total) * 100 if n_total > 0 else 0
                        
                        stats_text = f"""
Feature: {feature}
Total Values: {n_total:,}
Valid Values: {n_valid:,}
NaN Values: {n_nans:,} ({pct_nan:.2f}%)
"""
                        
                        if n_valid > 0:
                            stats_text += f"""
Min: {feature_data.min():.6f}
Max: {feature_data.max():.6f}
Mean: {feature_data.mean():.6f}
Std: {feature_data.std():.6f}
Median: {feature_data.median():.6f}
"""
                        
                        stats_label = ttk.Label(info_frame, text=stats_text,
                                              font=('Courier', 10),
                                              justify='left')
                        stats_label.pack(side='left', padx=20)
                        
                        # Sample data display
                        sample_frame = ttk.LabelFrame(data_display_frame,
                                                     text="Sample Data (first 100 values)",
                                                     padding=10)
                        sample_frame.pack(fill='both', expand=True, padx=10, pady=10)
                        
                        sample_text = scrolledtext.ScrolledText(sample_frame,
                                                               height=15,
                                                               width=80,
                                                               font=('Courier', 9))
                        sample_text.pack(fill='both', expand=True)
                        
                        # Show first 100 values
                        sample_text.insert(tk.END, "Index\tValue\n")
                        sample_text.insert(tk.END, "-" * 50 + "\n")
                        
                        for idx in range(min(100, len(feature_data))):
                            val = feature_data.iloc[idx]
                            if pd.isna(val):
                                sample_text.insert(tk.END, f"{idx}\tNaN\n")
                            else:
                                sample_text.insert(tk.END, f"{idx}\t{val:.6f}\n")
                        
                        if len(feature_data) > 100:
                            sample_text.insert(tk.END, f"\n... ({len(feature_data) - 100:,} more values)")
                        
                        sample_text.config(state='disabled')
                        
                    except Exception as e:
                        ttk.Label(data_display_frame,
                                 text=f"Error loading data: {str(e)}",
                                 foreground='red').pack(pady=20)
                
                ttk.Button(selector_frame,
                          text="Show Data",
                          command=show_feature_data).pack(side='left', padx=5)
                
                # Data display area
                data_display_frame = ttk.Frame(viewer_frame)
                data_display_frame.pack(fill='both', expand=True, padx=10, pady=10)
                
                ttk.Label(data_display_frame,
                         text="Select a feature and click 'Show Data'",
                         foreground='gray',
                         font=('Arial', 11)).pack(expand=True)
        
        # Bottom buttons
        button_frame = ttk.Frame(window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, 
                  text="📁 Open File Location",
                  command=lambda: self._open_folder(os.path.dirname(file_path))).pack(side='left', padx=5)
        
        if analysis['is_dataframe']:
            ttk.Button(button_frame,
                      text="💾 Export Feature List",
                      command=lambda: self._export_feature_list(analysis, file_path)).pack(side='left', padx=5)
        
        ttk.Button(button_frame,
                  text="Close",
                  command=window.destroy).pack(side='right', padx=5)
    
    def _export_feature_list(self, analysis, original_path):
        """Export feature list to text file"""
        # Ask where to save
        save_path = filedialog.asksaveasfilename(
            title="Save Feature List",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"{os.path.splitext(os.path.basename(original_path))[0]}_features.txt"
        )
        
        if not save_path:
            return
        
        try:
            with open(save_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FEATURE LIST\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Source: {os.path.basename(original_path)}\n")
                f.write(f"Total Features: {analysis['n_features']}\n")
                f.write(f"Total Frames: {analysis['n_frames']}\n\n")
                
                categories = [
                    ('Pose Features', analysis['pose_features']),
                    ('Brightness Features', analysis['brightness_features']),
                    ('Velocity Features', analysis['velocity_features']),
                    ('Angle Features', analysis['angle_features']),
                    ('Distance Features', analysis['distance_features']),
                    ('Other Features', analysis['other_features']),
                ]
                
                for cat_name, features in categories:
                    if features:
                        f.write("-" * 80 + "\n")
                        f.write(f"{cat_name} ({len(features)})\n")
                        f.write("-" * 80 + "\n")
                        for feat in features:
                            f.write(f"  {feat}\n")
                        f.write("\n")
            
            messagebox.showinfo("Export Complete", f"Feature list saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")
    
    def _open_folder(self, path):
        """Open folder in file explorer"""
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")
    
    def _open_text_file(self, path):
        """Open text file in default editor"""
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")
    
    def generate_ethogram(self):
        """Generate ethogram from predictions"""
        messagebox.showinfo("Ethogram", 
                          "Ethogram generation will create:\n"
                          "• Time budget charts\n"
                          "• Bout duration histograms\n"
                          "• Behavior raster plots\n"
                          "• Summary statistics\n\n"
                          "Select folder with prediction CSVs...")
    
    def show_training_viz(self):
        """Show training visualization window"""
        if self.train_viz_window is None or not self.train_viz_window.window.winfo_exists():
            self.train_viz_window = TrainingVisualizationWindow(self.root, self.theme)
        else:
            self.train_viz_window.window.lift()
    
    # === FEATURE EXTRACTION TOOL ===

    def open_feature_extraction(self):
        """Open the standalone Feature Extraction tool window."""
        win = tk.Toplevel(self.root)
        win.title("Feature Extraction")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.45), int(sh * 0.82)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.resizable(True, True)
        win.transient(self.root)

        # ── Header ────────────────────────────────────────────────────────
        ttk.Label(win, text="Feature Extraction",
                  font=('Arial', 14, 'bold')).pack(pady=(12, 2))
        ttk.Label(win,
                  text="Extract pose + brightness (+ optional optical flow) features.",
                  foreground='gray').pack()

        # ── Mode selector ─────────────────────────────────────────────────
        mode_frame = ttk.Frame(win)
        mode_frame.pack(fill='x', padx=12, pady=(8, 0))
        fe_mode = tk.StringVar(value='batch')
        ttk.Radiobutton(mode_frame, text="Batch / Project Folder",
                        variable=fe_mode, value='batch',
                        command=lambda: _switch_mode()).pack(side='left', padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="Single Video",
                        variable=fe_mode, value='single',
                        command=lambda: _switch_mode()).pack(side='left')

        # ── Source panel (swapped by mode) ────────────────────────────────
        source_outer = ttk.Frame(win)
        source_outer.pack(fill='x', padx=12, pady=4)

        # -- Batch panel --
        batch_panel = ttk.LabelFrame(source_outer, text="Batch Source", padding=10)
        batch_panel.columnconfigure(1, weight=1)

        ttk.Label(batch_panel, text="Project Folder:").grid(
            row=0, column=0, sticky='w', pady=3)
        fe_project = tk.StringVar(
            value=self.train_project_folder.get() if self.train_project_folder.get() else "")
        ttk.Entry(batch_panel, textvariable=fe_project, width=48).grid(
            row=0, column=1, padx=5, pady=3, sticky='ew')
        ttk.Button(batch_panel, text="Browse",
                   command=lambda: fe_project.set(
                       filedialog.askdirectory(title="Select Project Folder")
                       or fe_project.get())
                   ).grid(row=0, column=2, padx=2)

        ttk.Label(batch_panel,
                  text="Scans videos/ subfolder for DLC .h5 + matching video pairs.",
                  foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 2))

        # -- Single panel --
        single_panel = ttk.LabelFrame(source_outer, text="Single Video Source", padding=10)
        single_panel.columnconfigure(1, weight=1)

        ttk.Label(single_panel, text="DLC File (.h5/.csv):").grid(
            row=0, column=0, sticky='w', pady=3)
        fe_dlc_single = tk.StringVar()
        ttk.Entry(single_panel, textvariable=fe_dlc_single, width=48).grid(
            row=0, column=1, padx=5, pady=3, sticky='ew')
        ttk.Button(single_panel, text="Browse",
                   command=lambda: fe_dlc_single.set(
                       filedialog.askopenfilename(
                           title="Select DLC tracking file",
                           filetypes=[("DLC files", "*.h5 *.csv"), ("All files", "*.*")])
                       or fe_dlc_single.get())
                   ).grid(row=0, column=2, padx=2)

        ttk.Label(single_panel, text="Video File:").grid(
            row=1, column=0, sticky='w', pady=3)
        fe_video_single = tk.StringVar()
        ttk.Entry(single_panel, textvariable=fe_video_single, width=48).grid(
            row=1, column=1, padx=5, pady=3, sticky='ew')
        ttk.Button(single_panel, text="Browse",
                   command=lambda: fe_video_single.set(
                       filedialog.askopenfilename(
                           title="Select video file",
                           filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                                      ("All files", "*.*")])
                       or fe_video_single.get())
                   ).grid(row=1, column=2, padx=2)

        ttk.Label(single_panel, text="Output Folder:").grid(
            row=2, column=0, sticky='w', pady=3)
        fe_out_single = tk.StringVar()
        ttk.Entry(single_panel, textvariable=fe_out_single, width=48).grid(
            row=2, column=1, padx=5, pady=3, sticky='ew')
        ttk.Button(single_panel, text="Browse",
                   command=lambda: fe_out_single.set(
                       filedialog.askdirectory(title="Select output folder")
                       or fe_out_single.get())
                   ).grid(row=2, column=2, padx=2)
        ttk.Label(single_panel,
                  text="Leave blank to save alongside the video file.",
                  foreground='gray').grid(row=3, column=0, columnspan=3, sticky='w', pady=(0, 2))

        def _switch_mode():
            if fe_mode.get() == 'batch':
                single_panel.pack_forget()
                batch_panel.pack(fill='x')
            else:
                batch_panel.pack_forget()
                single_panel.pack(fill='x')

        # Start in batch mode
        batch_panel.pack(fill='x')

        # ── Feature settings ──────────────────────────────────────────────
        settings = ttk.LabelFrame(win, text="Feature Settings", padding=10)
        settings.pack(fill='x', padx=12, pady=4)
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Brightness Body Parts:").grid(
            row=0, column=0, sticky='w', pady=3)
        fe_bp_pixbrt = tk.StringVar(
            value=self.train_bp_pixbrt.get() if hasattr(self, 'train_bp_pixbrt') else "hrpaw,hlpaw,snout")
        ttk.Entry(settings, textvariable=fe_bp_pixbrt, width=40).grid(
            row=0, column=1, padx=5, pady=3, sticky='ew')
        ttk.Label(settings, text="comma-separated",
                  foreground='gray').grid(row=0, column=2, sticky='w')

        ttk.Label(settings, text="Square Sizes:").grid(
            row=1, column=0, sticky='w', pady=3)
        fe_square = tk.StringVar(
            value=self.train_square_sizes.get() if hasattr(self, 'train_square_sizes') else "40,40,40")
        ttk.Entry(settings, textvariable=fe_square, width=20).grid(
            row=1, column=1, padx=5, pady=3, sticky='w')

        ttk.Label(settings, text="Pixel Threshold:").grid(
            row=2, column=0, sticky='w', pady=3)
        fe_threshold = tk.DoubleVar(
            value=self.train_pix_threshold.get() if hasattr(self, 'train_pix_threshold') else 0.3)
        ttk.Entry(settings, textvariable=fe_threshold, width=10).grid(
            row=2, column=1, padx=5, pady=3, sticky='w')

        fe_optflow = tk.BooleanVar(
            value=self.train_include_optical_flow.get()
            if hasattr(self, 'train_include_optical_flow') else False)
        ttk.Checkbutton(settings,
                        text="Include Optical Flow  (slower — reads video frames)",
                        variable=fe_optflow).grid(row=4, column=1, sticky='w', pady=3)

        ttk.Label(settings, text="Optical Flow Parts:").grid(
            row=5, column=0, sticky='w', pady=3)
        fe_bp_optflow = tk.StringVar(
            value=self.train_bp_optflow.get() if hasattr(self, 'train_bp_optflow') else "hrpaw,hlpaw,snout")
        ttk.Entry(settings, textvariable=fe_bp_optflow, width=40).grid(
            row=5, column=1, padx=5, pady=3, sticky='ew')

        ttk.Label(settings, text="DLC Config (optional):").grid(
            row=6, column=0, sticky='w', pady=3)
        fe_dlc_config = tk.StringVar(
            value=self.train_dlc_config.get() if hasattr(self, 'train_dlc_config') else "")
        ttk.Entry(settings, textvariable=fe_dlc_config, width=40).grid(
            row=6, column=1, padx=5, pady=3, sticky='ew')
        ttk.Button(settings, text="Browse",
                   command=lambda: fe_dlc_config.set(
                       filedialog.askopenfilename(
                           title="Select DLC config.yaml",
                           filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
                       or fe_dlc_config.get())
                   ).grid(row=6, column=2, padx=2)

        # ── Progress log ──────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(win, text="Progress", padding=6)
        log_frame.pack(fill='both', expand=True, padx=12, pady=4)

        log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap='word',
                                             font=('Courier', 9))
        log_text.pack(fill='both', expand=True)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill='x', padx=12, pady=8)

        stop_event = threading.Event()

        run_btn = ttk.Button(btn_frame, text="▶  Run Extraction", style='Accent.TButton')
        run_btn.pack(side='left', padx=4)
        stop_btn = ttk.Button(btn_frame, text="■  Stop",
                              command=lambda: stop_event.set(),
                              state='disabled')
        stop_btn.pack(side='left', padx=4)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side='right', padx=4)

        def log(msg):
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            win.update_idletasks()

        def _build_cfg():
            return {
                'bp_pixbrt_list': [x.strip() for x in fe_bp_pixbrt.get().split(',') if x.strip()],
                'square_size':    [int(x.strip()) for x in fe_square.get().split(',') if x.strip()],
                'pix_threshold':  fe_threshold.get(),
                'include_optical_flow': fe_optflow.get(),
                'bp_optflow_list': [x.strip() for x in fe_bp_optflow.get().split(',') if x.strip()]
                                   if fe_optflow.get() else [],
                'dlc_config':     fe_dlc_config.get().strip() or None,
            }

        def start():
            mode = fe_mode.get()
            cfg = _build_cfg()

            if mode == 'single':
                dlc = fe_dlc_single.get().strip()
                vid = fe_video_single.get().strip()
                if not dlc or not vid:
                    messagebox.showwarning("Missing files",
                                           "Please select both a DLC file and a video file.",
                                           parent=win)
                    return
                out = fe_out_single.get().strip() or os.path.join(
                    os.path.dirname(vid), 'features')
                sessions = [{
                    'session_name': os.path.splitext(os.path.basename(vid))[0],
                    'pose_path':    dlc,
                    'video_path':   vid,
                }]
                cache_root = out
            else:
                project = fe_project.get().strip()
                if not project:
                    messagebox.showwarning("No folder",
                                           "Please select a project folder.", parent=win)
                    return
                sessions = None   # thread will scan
                cache_root = os.path.join(project, 'features')

            stop_event.clear()
            run_btn.config(state='disabled')
            stop_btn.config(state='normal')
            log_text.delete('1.0', tk.END)

            def _on_done():
                run_btn.config(state='normal')
                stop_btn.config(state='disabled')

            threading.Thread(
                target=self._run_feature_extraction_thread,
                args=(sessions, cache_root,
                      fe_project.get().strip() if mode == 'batch' else None,
                      cfg, log, stop_event, _on_done),
                daemon=True
            ).start()

        run_btn.config(command=start)

    def _run_feature_extraction_thread(self, sessions, cache_root,
                                        project_folder, cfg, log_fn, stop_event, done_fn):
        """Background worker for the Feature Extraction tool.

        sessions     : list of dicts with session_name/pose_path/video_path,
                       or None → scan project_folder for pairs.
        cache_root   : directory where .pkl files are written.
        project_folder: only used when sessions is None (batch scan).
        stop_event   : threading.Event; set by the Stop button to abort.
        """
        def log(msg):
            self.root.after(0, lambda m=msg: log_fn(m))

        try:
            # ── Discover sessions if not provided (batch mode) ─────────────
            if sessions is None:
                log(f"Project: {project_folder}")
                log("Scanning for sessions...\n")

                # Try the shared training session finder first
                found = self.find_training_sessions()
                if found:
                    sessions = found
                else:
                    # Fall back: scan videos/ subfolder for .h5 + video pairs
                    video_dir = os.path.join(project_folder, 'videos')
                    if not os.path.isdir(video_dir):
                        video_dir = project_folder
                    sessions = []
                    for h5 in glob.glob(os.path.join(video_dir, '*.h5')):
                        base = os.path.splitext(os.path.basename(h5))[0]
                        stem = base.split('DLC')[0] if 'DLC' in base else base
                        for ext in ('.mp4', '.avi', '.mov', '.mkv'):
                            vid = os.path.join(video_dir, stem + ext)
                            if os.path.isfile(vid):
                                sessions.append({
                                    'session_name': stem,
                                    'pose_path':    h5,
                                    'video_path':   vid,
                                })
                                break

                if not sessions:
                    log("No video+DLC session pairs found.")
                    self.root.after(0, done_fn)
                    return

            log(f"Found {len(sessions)} session(s):")
            for s in sessions:
                log(f"  • {s['session_name']}")
            log("")

            # ── Build cache key & hash ─────────────────────────────────────
            os.makedirs(cache_root, exist_ok=True)

            cfg_hash = PixelPawsGUI._feature_hash_key(cfg)

            # ── Extract ────────────────────────────────────────────────────
            total = len(sessions)
            skipped = 0
            errors  = 0
            for idx, session in enumerate(sessions, 1):
                if stop_event.is_set():
                    log("\nStopped by user.")
                    break

                name = session['session_name']
                log(f"[{idx}/{total}] {name}")

                cache_file = os.path.join(cache_root, f"{name}_features_{cfg_hash}.pkl")

                if os.path.isfile(cache_file):
                    log("  ✓ Already cached — skipping")
                    skipped += 1
                    continue

                try:
                    X = PixelPaws_ExtractFeatures(
                        pose_data_file=session['pose_path'],
                        video_file_path=session['video_path'],
                        bp_include_list=None,
                        bp_pixbrt_list=cfg['bp_pixbrt_list'],
                        square_size=cfg['square_size'],
                        pix_threshold=cfg['pix_threshold'],
                        config_yaml_path=cfg.get('dlc_config'),
                        include_optical_flow=cfg['include_optical_flow'],
                        bp_optflow_list=cfg['bp_optflow_list'] or None,
                    )
                    if stop_event.is_set():
                        log("\nStopped by user.")
                        break
                    X = X.reset_index(drop=True)
                    _atomic_pickle_save(X, cache_file)
                    log(f"  ✓ {X.shape[0]} frames × {X.shape[1]} features")
                    log(f"     → {cache_file}")
                except Exception as e:
                    log(f"  ✗ Error: {e}")
                    errors += 1

            if not stop_event.is_set():
                log(f"\nDone.  {total - skipped - errors} extracted, "
                    f"{skipped} skipped (cached), {errors} errors.")

        except Exception as e:
            import traceback
            log(f"\nUnexpected error: {e}")
            log(traceback.format_exc())

        self.root.after(0, done_fn)

    # === THEME METHODS ===

    def toggle_theme(self):
        """Toggle between light and dark mode"""
        new_mode = self.theme.toggle()
        ToolTip.set_theme(new_mode)
        if hasattr(self.notebook, 'update_theme'):
            self.notebook.update_theme(new_mode)
        self.apply_theme()
        self.set_status(f"Switched to {new_mode} mode")
    
    
    def _run_optimizer_direct(self, clf_path, file_sets, metric="f1"):
        """Run optimization directly without opening a window (called from Predict tab)"""
        try:
            # PixelPaws_ExtractFeatures is already defined at module level
            # No import needed - it's defined at the top of this file
            
            # predict_with_xgboost is already defined in this file at the module level
            
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Create progress window
            progress = tk.Toplevel(self.root)
            progress.title("Optimizing Parameters")
            _sw, _sh = progress.winfo_screenwidth(), progress.winfo_screenheight()
            progress.geometry(f"700x450+{(_sw-700)//2}+{(_sh-450)//2}")
            progress.transient(self.root)
            
            ttk.Label(progress, text="Parameter Optimization in Progress", 
                     font=('Arial', 12, 'bold')).pack(pady=10)
            
            status_label = ttk.Label(progress, text="Loading classifier...")
            status_label.pack(pady=5)
            
            results_text = scrolledtext.ScrolledText(progress, height=15, width=70)
            results_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            progress.update()
            
            # Load classifier
            results_text.insert(tk.END, "Loading classifier...\n")
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
            
            # Try both possible key names for compatibility
            model = clf_data.get('clf_model') or clf_data.get('model')
            if model is None:
                messagebox.showerror("Error", "Could not find model in classifier file.\nExpected 'clf_model' or 'model' key.")
                return
            
            # Clean body parts lists (remove DLC network names)
            clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
            
            # Auto-detect bp_include_list if missing
            clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=True)
            
            behavior_name = clf_data.get('Behavior_type', 'Unknown')
            results_text.insert(tk.END, f"Classifier: {behavior_name}\n\n")
            
            # Load all test data
            all_proba = []
            all_labels = []
            
            for i, file_set in enumerate(file_sets, 1):
                status_label.config(text=f"Loading test set {i}/{len(file_sets)}...")
                progress.update()
                
                results_text.insert(tk.END, f"Loading: {os.path.basename(file_set['video'])}\n")
                
                try:
                    # Load labels
                    labels_df = pd.read_csv(file_set['labels'])
                    if behavior_name not in labels_df.columns:
                        results_text.insert(tk.END, f"  ⚠ Skipping - behavior '{behavior_name}' not found in labels\n")
                        results_text.insert(tk.END, f"    Available columns: {list(labels_df.columns)}\n")
                        continue
                    
                    labels = labels_df[behavior_name].values
                    
                    if len(labels) == 0:
                        results_text.insert(tk.END, f"  ⚠ Skipping - labels file is empty\n")
                        continue
                    
                    # Use the same hash function as training (bp_include_list is always None
                    # in training, so force it here to guarantee matching cache keys).
                    cfg_hash = PixelPawsGUI._feature_hash_key(
                        {**clf_data, 'bp_include_list': None})

                    video_name = os.path.splitext(os.path.basename(file_set['video']))[0]
                    video_dir = os.path.dirname(file_set['video'])

                    # Build cache lookup list: project features/ first, then legacy video-local dirs
                    _proj_folder = self.current_project_folder.get()
                    _cache_fname = f"{video_name}_features_{cfg_hash}.pkl"
                    cache_locations = []
                    if _proj_folder and os.path.isdir(_proj_folder):
                        cache_locations.append(os.path.join(_proj_folder, 'features', _cache_fname))
                    cache_locations += [
                        os.path.join(video_dir, 'PredictionCache', _cache_fname),
                        os.path.join(video_dir, 'FeatureCache', _cache_fname),
                    ]
                    # Walk ancestor directories up to project root to handle nested video folders
                    _ancestor = video_dir
                    while True:
                        _parent = os.path.dirname(_ancestor)
                        if _parent == _ancestor:
                            break
                        _ancestor = _parent
                        cache_locations.append(os.path.join(_ancestor, 'features', _cache_fname))
                        cache_locations.append(os.path.join(_ancestor, 'FeatureCache', _cache_fname))
                        if _proj_folder and os.path.normpath(_ancestor) == os.path.normpath(_proj_folder):
                            break

                    cache_file = None
                    for loc in cache_locations:
                        if os.path.isfile(loc):
                            cache_file = loc
                            break

                    # Try to load from cache
                    if cache_file:
                        results_text.insert(tk.END, f"  ✓ Loaded from cache\n")
                        with open(cache_file, 'rb') as f:
                            X = pickle.load(f)
                    else:
                        # Extract features and save to project features/ (or video-local FeatureCache)
                        results_text.insert(tk.END, f"  Extracting features...\n")
                        progress.update()

                        if _proj_folder and os.path.isdir(_proj_folder):
                            cache_dir = os.path.join(_proj_folder, 'features')
                        else:
                            cache_dir = os.path.join(video_dir, 'FeatureCache')
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_file = os.path.join(cache_dir, _cache_fname)
                        
                        # Try to find config.yaml for crop detection
                        config_yaml = self.train_dlc_config.get() if self.train_dlc_config.get() else None
                        
                        # If not manually specified, auto-detect
                        if not config_yaml:
                            config_search_paths = [
                                os.path.join(video_dir, 'config.yaml'),
                                os.path.join(os.path.dirname(video_dir), 'config.yaml'),
                            ]
                            for cfg_path in config_search_paths:
                                if os.path.isfile(cfg_path):
                                    config_yaml = cfg_path
                                    break
                        
                        X = PixelPaws_ExtractFeatures(
                            pose_data_file=file_set['dlc'],
                            video_file_path=file_set['video'],
                            bp_include_list=clf_data.get('bp_include_list'),
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            config_yaml_path=config_yaml,  # Auto-detect crop from config
                            include_optical_flow=clf_data.get('include_optical_flow', False),
                            bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                        )

                        # Save to cache
                        _atomic_pickle_save(X, cache_file)
                        results_text.insert(tk.END, f"  ✓ Cached to {cache_file}\n")

                    X = augment_features_post_cache(X, clf_data, model, file_set['dlc'])

                    proba = predict_with_xgboost(
                        model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))

                    min_len = min(len(proba), len(labels))
                    all_proba.extend(proba[:min_len])
                    all_labels.extend(labels[:min_len])

                    results_text.insert(tk.END, f"  ✓ Loaded {min_len} frames\n")

                except Exception as e:
                    results_text.insert(tk.END, f"  ✗ Error: {str(e)}\n")
                    import traceback
                    print(f"Error processing {file_set['video']}:")
                    print(traceback.format_exc())
                    continue

            y_proba = np.array(all_proba)
            human_labels = np.array(all_labels)
            
            # Validate we have data
            if len(y_proba) == 0 or len(human_labels) == 0:
                messagebox.showerror("No Data", 
                    "No valid data was loaded from the test set.\n\n"
                    "Please check:\n"
                    "- Test set files are valid\n"
                    "- DLC and video files match\n"
                    "- Labels file contains the behavior column\n"
                    "- Feature extraction succeeded")
                progress.destroy()
                return
            
            # Check for at least some positive examples
            if np.sum(human_labels) == 0:
                messagebox.showwarning("No Positive Examples", 
                    "The test set contains no positive examples of the behavior.\n"
                    "Optimization requires at least some labeled behavior instances.")
                progress.destroy()
                return
            
            results_text.insert(tk.END, f"\nTotal: {len(y_proba)} frames\n")
            results_text.insert(tk.END, f"Positive examples: {np.sum(human_labels)} ({np.sum(human_labels)/len(human_labels)*100:.1f}%)\n\n")
            results_text.insert(tk.END, "Testing parameter combinations...\n")
            progress.update()
            
            # Grid search
            thresholds = np.arange(0.3, 0.8, 0.05)
            min_bouts = [1, 3, 5, 10, 15]
            min_after_bouts = [0, 1, 3, 5]  # Added min_after_bout
            max_gaps = [0, 5, 10, 20]
            
            best_score = 0
            best_params = {}
            
            total_tests = len(thresholds) * len(min_bouts) * len(min_after_bouts) * len(max_gaps)
            test_count = 0
            
            for threshold in thresholds:
                for min_bout in min_bouts:
                    for min_after_bout in min_after_bouts:
                        for max_gap in max_gaps:
                            test_count += 1
                            if test_count % 20 == 0:
                                status_label.config(text=f"Testing {test_count}/{total_tests}...")
                                progress.update()
                            
                            y_pred = (y_proba >= threshold).astype(int)
                            y_pred = self.apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap)
                            
                            accuracy = accuracy_score(human_labels, y_pred)
                            f1 = f1_score(human_labels, y_pred, zero_division=0)
                            
                            if metric == "f1":
                                score = f1
                            elif metric == "accuracy":
                                score = accuracy
                            else:
                                score = (f1 + accuracy) / 2
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'threshold': threshold,
                                    'min_bout': min_bout,
                                    'min_after_bout': min_after_bout,
                                    'max_gap': max_gap,
                                'min_bout': min_bout,
                                'max_gap': max_gap,
                                'accuracy': accuracy,
                                'f1': f1,
                                'precision': precision_score(human_labels, y_pred, zero_division=0),
                                'recall': recall_score(human_labels, y_pred, zero_division=0)
                            }
            
            # Display results
            results_text.insert(tk.END, "\n" + "="*60 + "\n")
            results_text.insert(tk.END, "BEST PARAMETERS\n")
            results_text.insert(tk.END, "="*60 + "\n\n")
            results_text.insert(tk.END, f"Threshold:     {best_params['threshold']:.3f}\n")
            results_text.insert(tk.END, f"Min Bout:      {best_params['min_bout']} frames\n")
            results_text.insert(tk.END, f"Threshold:     {best_params['threshold']:.3f}\n")
            results_text.insert(tk.END, f"Min Bout:      {best_params['min_bout']} frames\n")
            results_text.insert(tk.END, f"Min After:     {best_params['min_after_bout']} frames\n")
            results_text.insert(tk.END, f"Max Gap:       {best_params['max_gap']} frames\n\n")
            results_text.insert(tk.END, f"Accuracy:      {best_params['accuracy']*100:.2f}%\n")
            results_text.insert(tk.END, f"F1 Score:      {best_params['f1']*100:.2f}%\n")
            results_text.insert(tk.END, f"Precision:     {best_params['precision']*100:.2f}%\n")
            results_text.insert(tk.END, f"Recall:        {best_params['recall']*100:.2f}%\n")
            
            status_label.config(text="✓ Optimization complete!")
            
            # Store best params for preview
            self.optimized_params = best_params
            
            # Add buttons
            btn_frame = ttk.Frame(progress)
            btn_frame.pack(pady=10)
            
            def preview_with_params():
                progress.destroy()
                # Set the optimized parameters in the Predict tab
                self.optimized_threshold = best_params['threshold']
                self.optimized_min_bout = best_params['min_bout']
                self.optimized_min_after = best_params['min_after_bout']
                self.optimized_max_gap = best_params['max_gap']
                # Launch preview
                self.preview_with_predictions()
            
            ttk.Button(btn_frame, text="▶ Preview with These Parameters", 
                      command=preview_with_params, 
                      style='Accent.TButton').pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Close", 
                      command=progress.destroy).pack(side='left', padx=5)
            
            messagebox.showinfo("Optimization Complete",
                f"Best parameters found!\n\n"
                f"Threshold: {best_params['threshold']:.3f}\n"
                f"Min Bout: {best_params['min_bout']}\n"
                f"Min After: {best_params['min_after_bout']}\n"
                f"Max Gap: {best_params['max_gap']}\n\n"
                f"F1 Score: {best_params['f1']*100:.2f}%\n\n"
                f"Click 'Preview with These Parameters' to see results!")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            messagebox.showerror("Optimization Error",
                f"Failed to optimize:\n\n{str(e)}\n\n{error_detail}")
    
    def optimize_parameters(self):
        """Optimize classifier parameters against human labels"""
        
        # Check if we're being called from Predict tab with files already loaded
        auto_files = []
        if (hasattr(self, 'pred_classifier_path') and self.pred_classifier_path.get() and
            hasattr(self, 'pred_video_path') and self.pred_video_path.get() and
            hasattr(self, 'pred_dlc_path') and self.pred_dlc_path.get() and
            hasattr(self, 'pred_human_labels_path') and self.pred_human_labels_path.get()):
            
            # All files loaded in Predict tab
            auto_clf = self.pred_classifier_path.get()
            auto_video = self.pred_video_path.get()
            auto_dlc = self.pred_dlc_path.get()
            auto_labels = self.pred_human_labels_path.get()
            
            if all([os.path.isfile(f) for f in [auto_clf, auto_video, auto_dlc, auto_labels]]):
                # All files exist - run optimization directly
                self._run_optimizer_direct(auto_clf, [{
                    'video': auto_video,
                    'dlc': auto_dlc,
                    'labels': auto_labels
                }])
                return
        
        # Otherwise, open the full window for manual selection
        optimizer_window = tk.Toplevel(self.root)
        optimizer_window.title("Parameter Optimizer")
        sw, sh = optimizer_window.winfo_screenwidth(), optimizer_window.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.78)
        optimizer_window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Title
        title = ttk.Label(optimizer_window, text="Classifier Parameter Optimizer", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(optimizer_window,
            text="Optimize threshold and bout filtering parameters to maximize accuracy\n"
                 "against human-labeled data. Can test on multiple videos for better generalization.",
            justify='center')
        instructions.pack(pady=5)
        
        # Classifier selection
        clf_frame = ttk.LabelFrame(optimizer_window, text="Classifier", padding=10)
        clf_frame.pack(fill='x', padx=10, pady=10)
        
        clf_path_var = tk.StringVar()
        ttk.Label(clf_frame, text="Classifier File:").grid(row=0, column=0, sticky='w')
        ttk.Entry(clf_frame, textvariable=clf_path_var, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(clf_frame, text="📁 Browse",
                  command=lambda: clf_path_var.set(
                      filedialog.askopenfilename(
                          title="Select Classifier",
                          filetypes=[("Pickle files", "*.pkl")])
                  )).grid(row=0, column=2)
        
        # Video/Label sets
        data_frame = ttk.LabelFrame(optimizer_window, text="Test Data Sets", padding=10)
        data_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # List to store file sets
        file_sets = []
        
        # Listbox to show added files
        sets_listbox = tk.Listbox(data_frame, height=8)
        sets_listbox.pack(fill='both', expand=True, pady=5)
        
        def add_file_set():
            video = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi")])
            if not video:
                return
            
            dlc = filedialog.askopenfilename(
                title="Select DLC File for this video",
                filetypes=[("HDF5 files", "*.h5")])
            if not dlc:
                return
            
            labels = filedialog.askopenfilename(
                title="Select Human Labels CSV for this video",
                filetypes=[("CSV files", "*.csv")])
            if not labels:
                return
            
            file_sets.append({
                'video': video,
                'dlc': dlc,
                'labels': labels
            })
            
            display_name = f"{os.path.basename(video)} + {os.path.basename(labels)}"
            sets_listbox.insert(tk.END, display_name)
        
        def remove_selected():
            selection = sets_listbox.curselection()
            if selection:
                idx = selection[0]
                file_sets.pop(idx)
                sets_listbox.delete(idx)
        
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Add Video + DLC + Labels Set", 
                  command=add_file_set).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Remove Selected", 
                  command=remove_selected).pack(side='left', padx=5)
        
        ttk.Label(data_frame, text="Tip: Add multiple videos to test generalization across sessions",
                 foreground='gray').pack(pady=2)
        
        # Optimization settings
        opt_frame = ttk.LabelFrame(optimizer_window, text="Optimization Settings", padding=10)
        opt_frame.pack(fill='x', padx=10, pady=10)
        
        metric_var = tk.StringVar(value="f1")
        ttk.Label(opt_frame, text="Optimize for:").grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(opt_frame, text="F1 Score (Recommended)", 
                       variable=metric_var, value="f1").grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(opt_frame, text="Accuracy", 
                       variable=metric_var, value="accuracy").grid(row=0, column=2, sticky='w')
        ttk.Radiobutton(opt_frame, text="Balanced (F1 + Accuracy)", 
                       variable=metric_var, value="balanced").grid(row=0, column=3, sticky='w')
        
        # Results display
        results_frame = ttk.LabelFrame(optimizer_window, text="Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        results_text = scrolledtext.ScrolledText(results_frame, height=12, width=95)
        results_text.pack(fill='both', expand=True)
        
        # Status
        status_label = ttk.Label(optimizer_window, text="", foreground='blue')
        status_label.pack(pady=5)
        
        def run_optimization():
            clf_path = clf_path_var.get()
            
            if not clf_path or not os.path.isfile(clf_path):
                messagebox.showwarning("Missing Classifier", "Please select a classifier file.")
                return
            
            if not file_sets:
                messagebox.showwarning("No Test Data", "Please add at least one video + labels set.")
                return
            
            try:
                # PixelPaws_ExtractFeatures and predict_with_xgboost are already defined at module level
                # No import needed
                
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, "Starting parameter optimization...\n\n")
                status_label.config(text="Loading classifier...", foreground='blue')
                optimizer_window.update()
                
                # Load classifier
                with open(clf_path, 'rb') as f:
                    clf_data = pickle.load(f)
                
                # Try both possible key names for compatibility
                model = clf_data.get('clf_model') or clf_data.get('model')
                if model is None:
                    messagebox.showerror("Error", "Could not find model in classifier file.\nExpected 'clf_model' or 'model' key.")
                    return
                
                # Clean body parts lists (remove DLC network names)
                clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
                clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
                
                # Auto-detect bp_include_list if missing
                if not clf_data.get('bp_include_list'):
                    results_text.insert(tk.END, "⚠️  bp_include_list not found in classifier\n")
                    results_text.insert(tk.END, "  Auto-detecting from model features...\n")
                    optimizer_window.update()
                    clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=False)
                    if clf_data.get('bp_include_list'):
                        results_text.insert(tk.END, f"  ✓ Detected: {clf_data['bp_include_list']}\n")
                    optimizer_window.update()
                
                behavior_name = clf_data.get('Behavior_type', 'Unknown')
                results_text.insert(tk.END, f"Classifier: {behavior_name}\n")
                results_text.insert(tk.END, f"Test sets: {len(file_sets)}\n\n")
                
                # Load and prepare all test data
                all_proba = []
                all_labels = []
                
                for i, file_set in enumerate(file_sets, 1):
                    status_label.config(text=f"Loading test set {i}/{len(file_sets)}...")
                    optimizer_window.update()
                    
                    results_text.insert(tk.END, f"Set {i}: {os.path.basename(file_set['video'])}\n")
                    
                    # Load labels
                    labels_df = pd.read_csv(file_set['labels'])
                    if behavior_name not in labels_df.columns:
                        results_text.insert(tk.END, f"  ⚠ Skipping - behavior '{behavior_name}' not found\n")
                        continue
                    
                    labels = labels_df[behavior_name].values
                    
                    # Use the same hash function as training (bp_include_list is always None
                    # in training, so force it here to guarantee matching cache keys).
                    cfg_hash = PixelPawsGUI._feature_hash_key(
                        {**clf_data, 'bp_include_list': None})

                    video_name = os.path.splitext(os.path.basename(file_set['video']))[0]
                    video_dir = os.path.dirname(file_set['video'])

                    # Check cache locations: project/features/ first, then video-local fallbacks
                    _proj_folder = self.current_project_folder.get()
                    _cache_fname = f"{video_name}_features_{cfg_hash}.pkl"
                    cache_locations = []
                    if _proj_folder and os.path.isdir(_proj_folder):
                        cache_locations.append(os.path.join(_proj_folder, 'features', _cache_fname))
                    cache_locations += [
                        os.path.join(video_dir, 'PredictionCache', _cache_fname),
                        os.path.join(video_dir, 'FeatureCache', _cache_fname),
                    ]
                    _ancestor = video_dir
                    while True:
                        _parent = os.path.dirname(_ancestor)
                        if _parent == _ancestor:
                            break
                        _ancestor = _parent
                        cache_locations.append(os.path.join(_ancestor, 'features', _cache_fname))
                        cache_locations.append(os.path.join(_ancestor, 'FeatureCache', _cache_fname))
                        if _proj_folder and os.path.normpath(_ancestor) == os.path.normpath(_proj_folder):
                            break

                    cache_file = None
                    for loc in cache_locations:
                        if os.path.isfile(loc):
                            cache_file = loc
                            break

                    # Try to find config.yaml for crop detection
                    config_yaml = None
                    _cfg_search = [
                        os.path.join(video_dir, 'config.yaml'),
                        os.path.join(os.path.dirname(video_dir), 'config.yaml'),
                    ]
                    for _cfg_path in _cfg_search:
                        if os.path.isfile(_cfg_path):
                            config_yaml = _cfg_path
                            break

                    if not cache_file:
                        if _proj_folder and os.path.isdir(_proj_folder):
                            cache_dir = os.path.join(_proj_folder, 'features')
                        else:
                            cache_dir = os.path.join(video_dir, 'FeatureCache')
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_file = os.path.join(cache_dir, _cache_fname)

                    X = _load_features_for_prediction(
                        cache_file=cache_file,
                        model=model,
                        extract_fn=lambda: PixelPaws_ExtractFeatures(
                            pose_data_file=file_set['dlc'],
                            video_file_path=file_set['video'],
                            bp_include_list=clf_data.get('bp_include_list'),
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            config_yaml_path=config_yaml,
                            include_optical_flow=clf_data.get('include_optical_flow', False),
                            bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                        ),
                        save_path=cache_file,
                        log_fn=lambda m: results_text.insert(tk.END, m + '\n'),
                        dlc_path=file_set['dlc'],
                        clf_data=clf_data,
                    )

                    X = augment_features_post_cache(X, clf_data, model, file_set['dlc'])

                    # Get probabilities
                    proba = predict_with_xgboost(
                        model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))

                    # Match lengths
                    min_len = min(len(proba), len(labels))
                    all_proba.extend(proba[:min_len])
                    all_labels.extend(labels[:min_len])

                    results_text.insert(tk.END, f"  ✓ Loaded {min_len} frames\n")
                
                # Convert to numpy arrays
                y_proba = np.array(all_proba)
                human_labels = np.array(all_labels)
                
                results_text.insert(tk.END, f"\nTotal test frames: {len(y_proba)}\n\n")
                results_text.insert(tk.END, "="*80 + "\n")
                results_text.insert(tk.END, "TESTING PARAMETERS\n")
                results_text.insert(tk.END, "="*80 + "\n\n")
                
                # Grid search
                thresholds = np.arange(0.3, 0.8, 0.05)
                min_bouts = [1, 3, 5, 10, 15]
                min_after_bouts = [0, 1, 3, 5]  # Added min_after_bout
                max_gaps = [0, 5, 10, 20]
                
                best_score = 0
                best_params = {}
                metric_name = metric_var.get()
                
                total_tests = len(thresholds) * len(min_bouts) * len(min_after_bouts) * len(max_gaps)
                test_count = 0
                
                for threshold in thresholds:
                    for min_bout in min_bouts:
                        for min_after_bout in min_after_bouts:
                            for max_gap in max_gaps:
                                test_count += 1
                                status_label.config(
                                    text=f"Testing {test_count}/{total_tests}: "
                                         f"thresh={threshold:.2f}, min_bout={min_bout}, min_after={min_after_bout}, max_gap={max_gap}")
                                optimizer_window.update()
                                
                                # Apply threshold
                                y_pred = (y_proba >= threshold).astype(int)
                                
                                # Apply bout filtering
                                y_pred = self.apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap)
                                
                                # Calculate metrics
                                accuracy = accuracy_score(human_labels, y_pred)
                                f1 = f1_score(human_labels, y_pred, zero_division=0)
                                
                                # Score based on metric
                                if metric_name == "f1":
                                    score = f1
                                elif metric_name == "accuracy":
                                    score = accuracy
                                else:  # balanced
                                    score = (f1 + accuracy) / 2
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'threshold': threshold,
                                    'min_bout': min_bout,
                                    'min_after_bout': min_after_bout,
                                    'max_gap': max_gap,
                                    'accuracy': accuracy,
                                    'f1': f1,
                                    'precision': precision_score(human_labels, y_pred, zero_division=0),
                                    'recall': recall_score(human_labels, y_pred, zero_division=0)
                                }
                
                # Display results
                results_text.insert(tk.END, "\n" + "="*80 + "\n")
                results_text.insert(tk.END, "BEST PARAMETERS FOUND\n")
                results_text.insert(tk.END, "="*80 + "\n\n")
                
                results_text.insert(tk.END, f"Optimized for: {metric_name.upper()}\n")
                results_text.insert(tk.END, f"Tested on: {len(file_sets)} video(s), {len(y_proba)} total frames\n\n")
                
                results_text.insert(tk.END, f"Threshold:     {best_params['threshold']:.3f}\n")
                results_text.insert(tk.END, f"Min Bout:      {best_params['min_bout']} frames\n")
                results_text.insert(tk.END, f"Min After:     {best_params['min_after_bout']} frames\n")
                results_text.insert(tk.END, f"Max Gap:       {best_params['max_gap']} frames\n\n")
                
                results_text.insert(tk.END, "Performance Metrics:\n")
                results_text.insert(tk.END, f"  Accuracy:    {best_params['accuracy']*100:.2f}%\n")
                results_text.insert(tk.END, f"  F1 Score:    {best_params['f1']*100:.2f}%\n")
                results_text.insert(tk.END, f"  Precision:   {best_params['precision']*100:.2f}%\n")
                results_text.insert(tk.END, f"  Recall:      {best_params['recall']*100:.2f}%\n\n")
                
                # Compare to original
                orig_thresh = clf_data.get('best_thresh', 0.5)
                orig_min = clf_data.get('min_bout', 1)
                orig_gap = clf_data.get('max_gap', 0)
                
                results_text.insert(tk.END, "Original Classifier Parameters:\n")
                results_text.insert(tk.END, f"  Threshold:   {orig_thresh:.3f}\n")
                results_text.insert(tk.END, f"  Min Bout:    {orig_min}\n")
                results_text.insert(tk.END, f"  Max Gap:     {orig_gap}\n\n")
                
                # Calculate improvement
                y_pred_orig = (y_proba >= orig_thresh).astype(int)
                y_pred_orig = self.apply_bout_filtering(y_pred_orig, orig_min, 0, orig_gap)
                orig_f1 = f1_score(human_labels, y_pred_orig, zero_division=0)
                
                improvement = (best_params['f1'] - orig_f1) * 100
                results_text.insert(tk.END, f"F1 Improvement: {improvement:+.2f}%\n")
                
                status_label.config(text="✓ Optimization complete!", foreground='green')
                
                # Store optimized params
                self.optimized_params = best_params
                
                messagebox.showinfo("Optimization Complete",
                    f"Best parameters found!\n\n"
                    f"Tested on {len(file_sets)} video(s)\n\n"
                    f"Threshold: {best_params['threshold']:.3f}\n"
                    f"Min Bout: {best_params['min_bout']}\n"
                    f"Min After: {best_params['min_after_bout']}\n"
                    f"Max Gap: {best_params['max_gap']}\n\n"
                    f"F1 Score: {best_params['f1']*100:.2f}%\n"
                    f"Improvement: {improvement:+.2f}%\n\n"
                    f"These parameters will be auto-loaded in Preview!")
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                status_label.config(text=f"Error: {str(e)}", foreground='red')
                messagebox.showerror("Optimization Error",
                    f"Failed to optimize parameters:\n\n{str(e)}\n\n{error_detail}")
        
        ttk.Button(optimizer_window, text="Start Optimization", 
                  command=run_optimization).pack(pady=10)
    
        """Optimize classifier parameters against human labels"""
        optimizer_window = tk.Toplevel(self.root)
        optimizer_window.title("Parameter Optimizer")
        sw, sh = optimizer_window.winfo_screenwidth(), optimizer_window.winfo_screenheight()
        w, h = int(sw * 0.50), int(sh * 0.75)
        optimizer_window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Title
        title = ttk.Label(optimizer_window, text="Classifier Parameter Optimizer", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(optimizer_window,
            text="Optimize threshold and bout filtering parameters to maximize accuracy\n"
                 "against human-labeled data. Requires classifier and human labels.",
            justify='center')
        instructions.pack(pady=5)
        
        # Classifier selection
        clf_frame = ttk.LabelFrame(optimizer_window, text="Classifier", padding=10)
        clf_frame.pack(fill='x', padx=10, pady=10)
        
        clf_path_var = tk.StringVar()
        ttk.Label(clf_frame, text="Classifier File:").grid(row=0, column=0, sticky='w')
        ttk.Entry(clf_frame, textvariable=clf_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(clf_frame, text="📁 Browse",
                  command=lambda: clf_path_var.set(
                      filedialog.askopenfilename(
                          title="Select Classifier",
                          filetypes=[("Pickle files", "*.pkl")])
                  )).grid(row=0, column=2)
        
        # Video & Labels
        data_frame = ttk.LabelFrame(optimizer_window, text="Test Data", padding=10)
        data_frame.pack(fill='x', padx=10, pady=10)
        
        video_path_var = tk.StringVar()
        ttk.Label(data_frame, text="Video File:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(data_frame, textvariable=video_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="📁 Browse",
                  command=lambda: video_path_var.set(
                      filedialog.askopenfilename(
                          title="Select Video",
                          filetypes=[("Video files", "*.mp4 *.avi")])
                  )).grid(row=0, column=2)
        
        dlc_path_var = tk.StringVar()
        ttk.Label(data_frame, text="DLC File:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(data_frame, textvariable=dlc_path_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(data_frame, text="📁 Browse",
                  command=lambda: dlc_path_var.set(
                      filedialog.askopenfilename(
                          title="Select DLC File",
                          filetypes=[("HDF5 files", "*.h5")])
                  )).grid(row=1, column=2)
        
        labels_path_var = tk.StringVar()
        ttk.Label(data_frame, text="Human Labels:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(data_frame, textvariable=labels_path_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(data_frame, text="📁 Browse",
                  command=lambda: labels_path_var.set(
                      filedialog.askopenfilename(
                          title="Select Human Labels CSV",
                          filetypes=[("CSV files", "*.csv")])
                  )).grid(row=2, column=2)
        
        # Optimization settings
        opt_frame = ttk.LabelFrame(optimizer_window, text="Optimization Settings", padding=10)
        opt_frame.pack(fill='x', padx=10, pady=10)
        
        metric_var = tk.StringVar(value="f1")
        ttk.Label(opt_frame, text="Optimize for:").grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(opt_frame, text="F1 Score", variable=metric_var, value="f1").grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(opt_frame, text="Accuracy", variable=metric_var, value="accuracy").grid(row=0, column=2, sticky='w')
        ttk.Radiobutton(opt_frame, text="Balanced (F1 + Accuracy)", variable=metric_var, value="balanced").grid(row=0, column=3, sticky='w')
        
        # Results display
        results_frame = ttk.LabelFrame(optimizer_window, text="Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        results_text.pack(fill='both', expand=True)
        
        # Status
        status_label = ttk.Label(optimizer_window, text="", foreground='blue')
        status_label.pack(pady=5)
        
        def run_optimization():
            clf_path = clf_path_var.get()
            video_path = video_path_var.get()
            dlc_path = dlc_path_var.get()
            labels_path = labels_path_var.get()
            
            if not all([clf_path, video_path, dlc_path, labels_path]):
                messagebox.showwarning("Missing Files", "Please select all required files.")
                return
            
            try:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                # PixelPaws_ExtractFeatures and predict_with_xgboost are defined at module level
                
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, "Starting parameter optimization...\n\n")
                status_label.config(text="Loading classifier...", foreground='blue')
                optimizer_window.update()
                
                # Load classifier
                with open(clf_path, 'rb') as f:
                    clf_data = pickle.load(f)
                
                # Try both possible key names for compatibility
                model = clf_data.get('clf_model') or clf_data.get('model')
                if model is None:
                    messagebox.showerror("Error", "Could not find model in classifier file.\nExpected 'clf_model' or 'model' key.")
                    return
                
                behavior_name = clf_data.get('Behavior_type', 'Unknown')
                results_text.insert(tk.END, f"Classifier: {behavior_name}\n\n")
                
                # Load human labels
                labels_df = pd.read_csv(labels_path)
                if behavior_name not in labels_df.columns:
                    messagebox.showerror("Error", f"Behavior '{behavior_name}' not found in labels file.")
                    return
                
                human_labels = labels_df[behavior_name].values
                results_text.insert(tk.END, f"Human labels: {len(human_labels)} frames\n\n")
                
                # Extract features
                status_label.config(text="Extracting features...")
                optimizer_window.update()
                
                # Try to find config.yaml for crop detection
                video_dir = os.path.dirname(video_path)
                config_yaml = None
                config_search_paths = [
                    os.path.join(video_dir, 'config.yaml'),
                    os.path.join(os.path.dirname(video_dir), 'config.yaml'),
                ]
                for cfg_path in config_search_paths:
                    if os.path.isfile(cfg_path):
                        config_yaml = cfg_path
                        break
                
                X = PixelPaws_ExtractFeatures(
                    pose_data_file=dlc_path,
                    video_file_path=video_path,
                    bp_include_list=clf_data.get('bp_include_list'),
                    bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                    square_size=clf_data.get('square_size', [40]),
                    pix_threshold=clf_data.get('pix_threshold', 0.3),
                    config_yaml_path=config_yaml,  # Pass config for crop detection
                )
                
                # Add post-cache features the model may require (lag, egocentric, contact)
                X = augment_features_post_cache(X, clf_data, model, dlc_path)

                # Get probabilities
                y_proba = predict_with_xgboost(
                    model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))

                # Truncate to match labels length
                min_len = min(len(y_proba), len(human_labels))
                y_proba = y_proba[:min_len]
                human_labels = human_labels[:min_len]
                
                results_text.insert(tk.END, "="*60 + "\n")
                results_text.insert(tk.END, "TESTING PARAMETERS\n")
                results_text.insert(tk.END, "="*60 + "\n\n")
                
                # Grid search over parameters
                thresholds = np.arange(0.3, 0.8, 0.05)
                min_bouts = [1, 3, 5, 10, 15]
                max_gaps = [0, 5, 10, 20]
                
                best_score = 0
                best_params = {}
                metric_name = metric_var.get()
                
                total_tests = len(thresholds) * len(min_bouts) * len(max_gaps)
                test_count = 0
                
                for threshold in thresholds:
                    for min_bout in min_bouts:
                        for max_gap in max_gaps:
                            test_count += 1
                            status_label.config(
                                text=f"Testing {test_count}/{total_tests}: "
                                     f"thresh={threshold:.2f}, min_bout={min_bout}, max_gap={max_gap}")
                            optimizer_window.update()
                            
                            # Apply threshold
                            y_pred = (y_proba >= threshold).astype(int)
                            
                            # Apply bout filtering
                            y_pred = self.apply_bout_filtering(y_pred, min_bout, 0, max_gap)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(human_labels, y_pred)
                            f1 = f1_score(human_labels, y_pred, zero_division=0)
                            
                            # Score based on selected metric
                            if metric_name == "f1":
                                score = f1
                            elif metric_name == "accuracy":
                                score = accuracy
                            else:  # balanced
                                score = (f1 + accuracy) / 2
                            
                            # Track best
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'threshold': threshold,
                                    'min_bout': min_bout,
                                    'max_gap': max_gap,
                                    'accuracy': accuracy,
                                    'f1': f1,
                                    'precision': precision_score(human_labels, y_pred, zero_division=0),
                                    'recall': recall_score(human_labels, y_pred, zero_division=0)
                                }
                
                # Display results
                results_text.insert(tk.END, "\n" + "="*60 + "\n")
                results_text.insert(tk.END, "BEST PARAMETERS FOUND\n")
                results_text.insert(tk.END, "="*60 + "\n\n")
                
                results_text.insert(tk.END, f"Optimized for: {metric_name.upper()}\n\n")
                results_text.insert(tk.END, f"Threshold:     {best_params['threshold']:.3f}\n")
                results_text.insert(tk.END, f"Min Bout:      {best_params['min_bout']} frames\n")
                results_text.insert(tk.END, f"Max Gap:       {best_params['max_gap']} frames\n\n")
                
                results_text.insert(tk.END, "Performance Metrics:\n")
                results_text.insert(tk.END, f"  Accuracy:    {best_params['accuracy']*100:.2f}%\n")
                results_text.insert(tk.END, f"  F1 Score:    {best_params['f1']*100:.2f}%\n")
                results_text.insert(tk.END, f"  Precision:   {best_params['precision']*100:.2f}%\n")
                results_text.insert(tk.END, f"  Recall:      {best_params['recall']*100:.2f}%\n\n")
                
                # Compare to original
                orig_thresh = clf_data.get('best_thresh', 0.5)
                orig_min = clf_data.get('min_bout', 1)
                orig_gap = clf_data.get('max_gap', 0)
                
                results_text.insert(tk.END, "Original Parameters:\n")
                results_text.insert(tk.END, f"  Threshold:   {orig_thresh:.3f}\n")
                results_text.insert(tk.END, f"  Min Bout:    {orig_min}\n")
                results_text.insert(tk.END, f"  Max Gap:     {orig_gap}\n\n")
                
                # Calculate improvement
                y_pred_orig = (y_proba >= orig_thresh).astype(int)
                y_pred_orig = self.apply_bout_filtering(y_pred_orig, orig_min, 0, orig_gap)
                orig_f1 = f1_score(human_labels, y_pred_orig, zero_division=0)
                
                improvement = (best_params['f1'] - orig_f1) * 100
                results_text.insert(tk.END, f"F1 Improvement: {improvement:+.2f}%\n")
                
                status_label.config(text="✓ Optimization complete!", foreground='green')
                
                messagebox.showinfo("Optimization Complete",
                    f"Best parameters found!\n\n"
                    f"Threshold: {best_params['threshold']:.3f}\n"
                    f"Min Bout: {best_params['min_bout']}\n"
                    f"Max Gap: {best_params['max_gap']}\n\n"
                    f"F1 Score: {best_params['f1']*100:.2f}%\n"
                    f"Improvement: {improvement:+.2f}%")
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                status_label.config(text=f"Error: {str(e)}", foreground='red')
                messagebox.showerror("Optimization Error",
                    f"Failed to optimize parameters:\n\n{str(e)}\n\n{error_detail}")
        
        ttk.Button(optimizer_window, text="Start Optimization", 
                  command=run_optimization).pack(pady=10)
    
    def convert_boris_to_pixelpaws(self):
        """Convert BORIS TSV/CSV files to PixelPaws per-frame CSV format"""
        # Create converter window
        converter_window = tk.Toplevel(self.root)
        converter_window.title("BORIS to PixelPaws Converter")
        sw, sh = converter_window.winfo_screenwidth(), converter_window.winfo_screenheight()
        w, h = int(sw * 0.45), int(sh * 0.75)
        converter_window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Title
        title = ttk.Label(converter_window, text="BORIS to PixelPaws Converter", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(converter_window, 
            text="Convert BORIS event files (START/STOP or POINT) to PixelPaws per-frame format.\n"
                 "Handles Behavior type column with START/STOP events.",
            justify='center')
        instructions.pack(pady=5)
        
        # File selection
        file_frame = ttk.LabelFrame(converter_window, text="Input File", padding=10)
        file_frame.pack(fill='x', padx=10, pady=10)
        
        boris_file_var = tk.StringVar()
        ttk.Label(file_frame, text="BORIS CSV/TSV:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(file_frame, textvariable=boris_file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="📁 Browse",
                  command=lambda: boris_file_var.set(
                      filedialog.askopenfilename(
                          title="Select BORIS File",
                          filetypes=[("CSV/TSV files", "*.csv *.tsv"), ("All files", "*.*")])
                  )).grid(row=0, column=2)

        # Batch folder (optional)
        batch_frame = ttk.LabelFrame(converter_window, text="Batch Folder (Optional)", padding=10)
        batch_frame.pack(fill='x', padx=10, pady=(0, 5))

        batch_folder_var = tk.StringVar()
        ttk.Label(batch_frame, text="Folder:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(batch_frame, textvariable=batch_folder_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(batch_frame, text="📁 Browse",
                   command=lambda: batch_folder_var.set(filedialog.askdirectory(
                       title="Select folder containing BORIS CSV/TSV files"))
                   ).grid(row=0, column=2)
        ttk.Label(batch_frame,
                  text="If set, converts every .csv/.tsv in this folder.\n"
                       "Single file above is used only when no batch folder is selected.",
                  foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w')

        # Helper functions (defined early so they can be used)
        def find_column(df, candidates):
            """Find column matching one of the candidates (case-insensitive)"""
            normalized = {"".join(c.lower() for c in col if c.isalnum()): col for col in df.columns}
            for cand in candidates:
                key = "".join(c.lower() for c in cand if c.isalnum())
                for norm, real in normalized.items():
                    if key == norm:
                        return real
            return None
        
        # Parameters
        param_frame = ttk.LabelFrame(converter_window, text="Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=10)

        all_behaviors_var = tk.BooleanVar(value=False)
        def _toggle_all_behaviors():
            state = 'disabled' if all_behaviors_var.get() else 'normal'
            behavior_entry.config(state=state)
            detect_btn.config(state=state)
        ttk.Checkbutton(param_frame, text="All behaviors (one column per behavior)",
                        variable=all_behaviors_var,
                        command=_toggle_all_behaviors).grid(row=0, column=0, columnspan=3,
                                                            sticky='w', pady=(5, 2))

        ttk.Label(param_frame, text="Behavior Name:").grid(row=1, column=0, sticky='w', pady=5)
        behavior_var = tk.StringVar(value="L_licking")
        behavior_entry = ttk.Entry(param_frame, textvariable=behavior_var, width=30)
        behavior_entry.grid(row=1, column=1, sticky='w', padx=5)
        
        # Auto-detect button
        def auto_detect_behaviors():
            boris_path = boris_file_var.get().strip()
            if not boris_path or not os.path.isfile(boris_path):
                messagebox.showwarning("No File", "Please select a BORIS file first.")
                return
            
            try:
                # Load file - try CSV first, then TSV
                df = None
                try:
                    df = pd.read_csv(boris_path)
                    if len(df.columns) < 3:  # Too few columns, might be TSV
                        df = pd.read_csv(boris_path, sep='\t')
                except Exception:
                    try:
                        df = pd.read_csv(boris_path, sep='\t')
                    except Exception:
                        pass
                
                if df is None or df.empty:
                    messagebox.showerror("Error", "Could not read file as CSV or TSV.")
                    return
                
                # Find behavior column (case-insensitive)
                behavior_col = find_column(df, ["Behavior", "behaviour"])
                if not behavior_col:
                    # Show what columns we found to help user
                    cols = ", ".join(df.columns[:10])
                    messagebox.showerror("Error", 
                        f"Could not find 'Behavior' column in file.\n\n"
                        f"Found columns: {cols}...")
                    return
                
                # Get unique behaviors
                behaviors = df[behavior_col].dropna().unique().tolist()
                behaviors = sorted([str(b) for b in behaviors if str(b).strip()])
                
                if not behaviors:
                    messagebox.showinfo("No Behaviors", "No behaviors found in Behavior column.")
                    return
                
                # Show selection dialog
                dialog = tk.Toplevel(converter_window)
                dialog.title("Select Behavior")
                _sw, _sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
                dialog.geometry(f"450x500+{(_sw-450)//2}+{(_sh-500)//2}")
                dialog.transient(converter_window)
                dialog.grab_set()
                
                ttk.Label(dialog, text=f"Found {len(behaviors)} behavior(s):",
                         font=('Arial', 10, 'bold')).pack(padx=10, pady=10)
                
                # Listbox
                list_frame = ttk.Frame(dialog)
                list_frame.pack(fill='both', expand=True, padx=10, pady=5)
                
                scrollbar = ttk.Scrollbar(list_frame)
                scrollbar.pack(side='right', fill='y')
                
                listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial', 10))
                listbox.pack(side='left', fill='both', expand=True)
                scrollbar.config(command=listbox.yview)
                
                for b in behaviors:
                    listbox.insert(tk.END, b)
                
                if behaviors:
                    listbox.selection_set(0)
                
                def on_select():
                    sel = listbox.curselection()
                    if sel:
                        behavior_var.set(listbox.get(sel[0]))
                        dialog.destroy()
                
                btn_frame = ttk.Frame(dialog)
                btn_frame.pack(pady=10)
                ttk.Button(btn_frame, text="Select", command=on_select).pack(side='left', padx=5)
                ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
                
                listbox.bind('<Double-Button-1>', lambda e: on_select())
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to read BORIS file:\n{str(e)}")
        
        detect_btn = ttk.Button(param_frame, text="🔍 Auto-Detect", command=auto_detect_behaviors)
        detect_btn.grid(row=1, column=2, padx=5)

        ttk.Label(param_frame, text="Video FPS:").grid(row=2, column=0, sticky='w', pady=5)
        fps_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=fps_var, width=15).grid(row=2, column=1, sticky='w', padx=5)
        ttk.Label(param_frame, text="(leave blank to auto-detect from FPS column)").grid(row=2, column=2, sticky='w')
        
        # Output directory
        output_frame = ttk.LabelFrame(converter_window, text="Output", padding=10)
        output_frame.pack(fill='x', padx=10, pady=10)
        
        output_dir_var = tk.StringVar()
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky='w')
        ttk.Entry(output_frame, textvariable=output_dir_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="📁 Browse", 
                  command=lambda: output_dir_var.set(filedialog.askdirectory())).grid(row=0, column=2)
        
        # Status / log
        log_frame = ttk.LabelFrame(converter_window, text="Log", padding=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        log_box = tk.Text(log_frame, height=8, wrap='word', state='disabled',
                          font=('Courier', 9))
        log_scroll = ttk.Scrollbar(log_frame, command=log_box.yview)
        log_box.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side='right', fill='y')
        log_box.pack(fill='both', expand=True)

        def log_status(msg, color='blue'):
            log_box.config(state='normal')
            log_box.insert('end', msg + '\n')
            log_box.config(state='disabled')
            log_box.see('end')
            converter_window.update()
        
        def get_fps(fps_text, df):
            """Get FPS from input or DataFrame"""
            fps_text = fps_text.strip()
            if fps_text:
                try:
                    return float(fps_text)
                except (ValueError, TypeError):
                    raise ValueError("FPS must be a positive number")
            
            # Try to find FPS column
            fps_col = find_column(df, ["FPS", "FrameRate", "Frame rate"])
            if fps_col and not df[fps_col].isna().all():
                return float(df[fps_col].dropna().iloc[0])
            
            raise ValueError("Could not determine FPS. Please enter it manually or include FPS column.")
        
        def do_convert_one(boris_path, behavior_name, fps_text, output_dir, all_behaviors=False):
            """Convert a single BORIS file.

            If all_behaviors=True, ignores behavior_name and extracts every unique
            behavior into its own column in a single output CSV.
            Returns (n_frames, summary_dict, used_image_idx).
            """
            df = None
            try:
                df = pd.read_csv(boris_path)
                if len(df.columns) < 3:
                    df = pd.read_csv(boris_path, sep='\t')
            except Exception:
                df = pd.read_csv(boris_path, sep='\t')

            if df is None or df.empty:
                raise Exception("File is empty or could not be parsed")

            fps_val = get_fps(fps_text, df)

            behavior_col  = find_column(df, ["Behavior", "behaviour"])
            type_col      = find_column(df, ["Behavior type", "Type"])
            time_col      = find_column(df, ["Time", "Time (s)", "time"])
            image_idx_col = find_column(df, ["Image index", "image index"])

            if not all([behavior_col, type_col, time_col]):
                raise Exception("Could not find required columns: Behavior, Behavior type, Time")

            df_sorted = df.sort_values(time_col).reset_index(drop=True)

            # Total frame count — prefer Image index max over time × fps
            if image_idx_col and not df[image_idx_col].isna().all():
                n_frames = int(df[image_idx_col].dropna().max()) + 1
            else:
                max_time = float(df_sorted[time_col].max())
                duration_col = find_column(df, ["Media duration", "Media duration (s)"])
                if duration_col and not df[duration_col].isna().all():
                    max_time = max(max_time, float(df[duration_col].dropna().iloc[0]))
                n_frames = int(np.ceil(max_time * fps_val))

            # Determine which behaviors to extract
            if all_behaviors:
                behaviors_to_extract = sorted(
                    df[behavior_col].dropna().unique().tolist(), key=str)
            else:
                behaviors_to_extract = [behavior_name]

            # Build per-behavior label arrays
            label_arrays = {b: np.zeros(n_frames, dtype=int) for b in behaviors_to_extract}
            active_starts = {b: None for b in behaviors_to_extract}

            for _, row in df_sorted.iterrows():
                beh      = str(row[behavior_col])
                beh_type = str(row[type_col]).strip().upper()
                t        = float(row[time_col])

                if beh not in label_arrays:
                    continue

                # Frame number: use Image index if present, else time × fps
                if image_idx_col is not None and pd.notna(row.get(image_idx_col)):
                    frame_num = int(row[image_idx_col])
                else:
                    frame_num = int(round(t * fps_val))

                if beh_type == "START":
                    active_starts[beh] = frame_num
                elif beh_type == "STOP":
                    if active_starts[beh] is not None:
                        for f in range(active_starts[beh], min(frame_num, n_frames)):
                            label_arrays[beh][f] = 1
                        active_starts[beh] = None
                elif beh_type == "POINT":
                    if 0 <= frame_num < n_frames:
                        label_arrays[beh][frame_num] = 1

            output_df  = pd.DataFrame(label_arrays)
            base_name  = os.path.splitext(os.path.basename(boris_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_labels.csv")
            output_df.to_csv(output_path, index=False)

            summary = {b: int(np.sum(arr)) for b, arr in label_arrays.items()}
            return n_frames, summary, image_idx_col is not None

        def run_conversion():
            use_all = all_behaviors_var.get()
            behavior_name = behavior_var.get().strip()
            if not use_all and not behavior_name:
                messagebox.showwarning("No Behavior", "Please enter a behavior name or check 'All behaviors'.")
                return

            fps_text   = fps_var.get()
            output_dir = output_dir_var.get().strip()
            batch_dir  = batch_folder_var.get().strip()

            # Collect files to process
            if batch_dir and os.path.isdir(batch_dir):
                files = sorted(
                    glob.glob(os.path.join(batch_dir, "*.csv")) +
                    glob.glob(os.path.join(batch_dir, "*.tsv"))
                )
                if not files:
                    messagebox.showwarning("No Files", f"No .csv/.tsv files found in:\n{batch_dir}")
                    return
                use_output_dir = output_dir or batch_dir
            else:
                single = boris_file_var.get().strip()
                if not single or not os.path.isfile(single):
                    messagebox.showwarning("No File", "Please select a BORIS file or batch folder.")
                    return
                files = [single]
                use_output_dir = output_dir or os.path.dirname(single)

            os.makedirs(use_output_dir, exist_ok=True)
            ok, failed = 0, 0

            for boris_path in files:
                log_status(f"→ {os.path.basename(boris_path)}")
                try:
                    n_frames, summary, used_idx = do_convert_one(
                        boris_path, behavior_name, fps_text, use_output_dir,
                        all_behaviors=use_all)
                    src = "frame index" if used_idx else "time×fps"
                    if use_all:
                        detail = ", ".join(
                            f"{b}:{n}" for b, n in summary.items())
                        log_status(
                            f"  ✓ {n_frames} frames, {len(summary)} behaviors [{detail}], src={src}",
                            'green')
                    else:
                        n_pos = list(summary.values())[0]
                        pct = (n_pos / n_frames * 100) if n_frames else 0
                        log_status(
                            f"  ✓ {n_frames} frames, {n_pos} positive ({pct:.1f}%), src={src}",
                            'green')
                    ok += 1
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    log_status(f"  ✗ {e}", 'red')
                    failed += 1

            summary_msg = f"\nDone: {ok} succeeded, {failed} failed."
            log_status(summary_msg, 'green' if failed == 0 else 'red')
            if len(files) == 1 and ok == 1:
                messagebox.showinfo("Success", f"Converted 1 file.\nOutput: {use_output_dir}")
            elif ok > 0:
                messagebox.showinfo("Batch Complete",
                    f"{ok}/{len(files)} files converted.\nOutput: {use_output_dir}")
        
        # Convert button
        ttk.Button(converter_window, text="🔄 Convert", command=run_conversion, 
                  style='Accent.TButton').pack(pady=10)
        
        # Info
        info = ttk.Label(converter_window,
            text="Expected BORIS format: CSV/TSV with columns:\n"
                 "• Behavior (behavior name)\n"
                 "• Behavior type (START/STOP/POINT)\n"
                 "• Time (timestamp in seconds)\n"
                 "• FPS (optional - can be entered manually)",
            justify='center', foreground='gray')
        info.pack(pady=5)
    
    def apply_theme(self):
        """Apply current theme to all widgets.

        With ttkbootstrap: switches the global ttk theme (all ttk widgets
        update automatically). Only raw tk widgets (Canvas, Listbox,
        scrolledtext) need manual colour updates.

        Without ttkbootstrap: applies colours to root and train_log only
        (legacy behaviour).
        """
        if TTKBOOTSTRAP_AVAILABLE:
            theme_name = Theme._DARK_THEME if self.theme.is_dark() else Theme._LIGHT_THEME
            try:
                style = ttk.Style()
                # Neutralize primary color before switching theme
                style.colors.primary = '#888888'
                style.theme_use(theme_name)
                # Re-apply light button style (theme_use makes them solid)
                if self.theme.is_dark():
                    style.configure('TButton', background='#3a3a3a',
                                    foreground='#e0e0e0', bordercolor='#666666',
                                    lightcolor='#3a3a3a', darkcolor='#666666')
                    style.map('TButton',
                              background=[('active', '#4a4a4a'), ('pressed', '#555555')],
                              bordercolor=[('active', '#888888'), ('pressed', '#999999')],
                              lightcolor=[('active', '#4a4a4a'), ('pressed', '#555555')],
                              darkcolor=[('active', '#888888'), ('pressed', '#999999')],
                              foreground=[('active', '#f0f0f0'), ('pressed', '#f0f0f0')])
                else:
                    style.configure('TButton', background='#f8f9fa',
                                    foreground='#333333', bordercolor='#aaaaaa',
                                    lightcolor='#f8f9fa', darkcolor='#aaaaaa')
                    style.map('TButton',
                              background=[('active', '#e9ecef'), ('pressed', '#dee2e6')],
                              bordercolor=[('active', '#888888'), ('pressed', '#666666')],
                              lightcolor=[('active', '#e9ecef'), ('pressed', '#dee2e6')],
                              darkcolor=[('active', '#888888'), ('pressed', '#666666')],
                              foreground=[('active', '#222222'), ('pressed', '#222222')])
            except Exception:
                pass

        # Manual updates for raw tk widgets that don't follow ttk themes
        bg = self.theme.colors['bg']
        fg = self.theme.colors['fg']
        text_bg = self.theme.colors['text_bg']

        try:
            self.root.configure(bg=bg)
        except (tk.TclError, AttributeError):
            pass

        # Update scrolledtext log
        try:
            self.train_log.configure(bg=text_bg, fg=fg, insertbackground=fg)
        except (tk.TclError, KeyError, AttributeError):
            pass
    
    # === UTILITY METHODS ===
    
    def set_status(self, message):
        """Update status bar"""
        self.status_text.set(message)
        self.root.update_idletasks()
    
    def show_about(self):
        """Open the PixelPaws GitHub page."""
        import webbrowser
        webbrowser.open("https://github.com/rslivicki/PixelPaws")
    
    def show_docs(self):
        """Open the PixelPaws documentation on GitHub."""
        import webbrowser
        webbrowser.open("https://github.com/rslivicki/PixelPaws#readme")
    
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts = """
Keyboard Shortcuts:

F11         - Toggle fullscreen
Ctrl+D      - Toggle dark mode
Ctrl+Q      - Data quality check
Ctrl+V      - Video preview
Ctrl+L      - Auto-label assistant
Ctrl+T      - Start training

Space       - Play/Pause (in video preview)
Left/Right  - Previous/Next frame
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    # ===== PREDICTION TAB METHODS =====
    
    def refresh_pred_classifiers(self):
        """Populate the predict-tab classifier dropdown from project + global classifiers."""
        from user_config import get_global_classifiers_folder
        self.pred_classifier_options = {}

        # Local project classifiers
        clf_dir = os.path.join(self.current_project_folder.get(), 'classifiers')
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    self.pred_classifier_options[f"[Project] {f}"] = os.path.join(clf_dir, f)

        # Global classifiers library
        gcf = get_global_classifiers_folder()
        if os.path.isdir(gcf):
            for f in sorted(os.listdir(gcf)):
                if f.endswith('.pkl'):
                    self.pred_classifier_options[f"[Global] {f}"] = os.path.join(gcf, f)

        if hasattr(self, 'pred_classifier_combo'):
            self.pred_classifier_combo['values'] = list(self.pred_classifier_options.keys())

    def _on_pred_classifier_selected(self, event=None):
        """Update the full path StringVar when a dropdown item is chosen."""
        name = self.pred_classifier_combo.get()
        if name in self.pred_classifier_options:
            clf_path = self.pred_classifier_options[name]
            self.pred_classifier_path.set(clf_path)
            for _w in check_classifier_portability(clf_path):
                print(f"⚠️  [{name}] {_w}")

    def refresh_pred_videos(self):
        """Populate the predict-tab video dropdown from project videos/ folder."""
        videos_dir = os.path.join(self.current_project_folder.get(), 'videos')
        self.pred_video_options = {}
        if os.path.isdir(videos_dir):
            exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
            for f in sorted(os.listdir(videos_dir)):
                if f.lower().endswith(exts):
                    self.pred_video_options[f] = os.path.join(videos_dir, f)
        if hasattr(self, 'pred_video_combo'):
            self.pred_video_combo['values'] = list(self.pred_video_options.keys())

    def _on_pred_video_selected(self, event=None):
        """Set full video path and auto-find DLC/features when dropdown selection changes."""
        name = self.pred_video_combo.get()
        if name in self.pred_video_options:
            self.pred_video_path.set(self.pred_video_options[name])
        # Clear stale paths so auto-find re-detects for new video
        self.pred_dlc_path.set('')
        self.pred_features_path.set('')
        self._auto_find_pred_files()

    def _auto_find_pred_files(self):
        """Silently populate DLC path and features cache for the currently selected video."""
        video_path = self.pred_video_path.get()
        if not video_path or not os.path.isfile(video_path):
            return

        video_folder   = os.path.dirname(video_path)
        video_base     = os.path.splitext(os.path.basename(video_path))[0]
        project_folder = self.current_project_folder.get()

        # --- DLC pose file ---
        if not self.pred_dlc_path.get():
            dlc_files = glob.glob(os.path.join(video_folder, f"{video_base}DLC*.h5"))
            if dlc_files:
                filtered = [f for f in dlc_files if 'filtered' in f.lower()]
                self.pred_dlc_path.set((filtered or dlc_files)[0])

        # --- Features cache ---
        if not self.pred_features_path.get():
            search_locs = [
                os.path.join(project_folder, 'features'),        # canonical
                os.path.join(project_folder, 'FeatureCache'),    # legacy project-level
                os.path.join(video_folder, 'FeatureCache'),      # legacy per-video
                os.path.join(video_folder, 'PredictionCache'),   # from predict/preview
                video_folder,                                     # root video folder
            ]
            for loc in search_locs:
                matches = glob.glob(os.path.join(loc, f"{video_base}_features*.pkl"))
                if matches:
                    self.pred_features_path.set(matches[0])
                    break

    def _parse_time_to_frames(self, time_str, fps):
        """Convert a clip boundary string to a frame index.

        Accepted formats
        ----------------
        123       plain integer  → treated as a frame number (returned as-is)
        1.5       decimal        → seconds, multiplied by fps
        1:30      MM:SS          → seconds, multiplied by fps
        1:30:00   H:MM:SS        → seconds, multiplied by fps
        ''        blank          → returns None (caller uses full-video default)
        """
        s = time_str.strip()
        if not s:
            return None
        parts = s.split(':')
        try:
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                return int(seconds * fps)
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + float(parts[1])
                return int(seconds * fps)
            else:
                # No colon — distinguish frames from seconds by the presence of a decimal point
                if '.' in s:
                    return int(float(s) * fps)   # e.g. "1.5" → 1.5 s
                else:
                    return int(s)                 # e.g. "300" → frame 300
        except ValueError:
            return None

    def browse_pred_classifier(self):
        """Browse for classifier file"""
        filepath = filedialog.askopenfilename(
            title="Select Classifier File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_classifier_path.set(filepath)
    
    def view_pred_classifier_info(self):
        """View information about selected classifier"""
        clf_path = self.pred_classifier_path.get()
        if not clf_path or not os.path.isfile(clf_path):
            messagebox.showwarning("No File", "Please select a valid classifier file.")
            return
        
        try:
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
            
            info = "=== Classifier Information ===\n\n"
            info += f"File: {os.path.basename(clf_path)}\n\n"
            
            if 'Behavior_type' in clf_data:
                info += f"Behavior: {clf_data['Behavior_type']}\n"
            if 'best_thresh' in clf_data:
                info += f"Best Threshold: {clf_data['best_thresh']:.3f}\n"
            if 'min_bout' in clf_data:
                info += f"Min Bout: {clf_data['min_bout']} frames\n"
            if clf_data.get('prob_calibrator') is not None:
                info += "Calibration: isotonic (on OOF)\n"
            _fm = clf_data.get('fold_models') or []
            if _fm:
                info += f"Fold ensemble: {len(_fm)} fold models saved\n"

            warnings = check_classifier_portability(clf_path)
            if warnings:
                info += "\n⚠️  Portability warnings:\n"
                for _w in warnings:
                    info += f"  • {_w}\n"

            messagebox.showinfo("Classifier Info", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier:\n{str(e)}")
    
    def browse_pred_video(self):
        """Browse for video file"""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_video_path.set(filepath)
            # Clear stale paths so auto-find re-detects for new video
            self.pred_dlc_path.set('')
            self.pred_features_path.set('')
            self._auto_find_pred_files()

    def browse_pred_dlc(self):
        """Browse for DLC file"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Pose File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_dlc_path.set(filepath)
    
    def browse_pred_features(self):
        """Browse for pre-extracted features file"""
        filepath = filedialog.askopenfilename(
            title="Select Features File (Optional)",
            filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_features_path.set(filepath)
    
    def browse_pred_dlc_config(self):
        """Browse for DLC config.yaml file"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_dlc_config_path.set(filepath)
    
    def browse_pred_human_labels(self):
        """Browse for human labels file"""
        filepath = filedialog.askopenfilename(
            title="Select Human Labels File (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_human_labels_path.set(filepath)
    
    def browse_pred_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.pred_output_folder.set(folder)
    
    def auto_find_dlc(self):
        """Automatically find DLC file for selected video"""
        video_path = self.pred_video_path.get()
        if not video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        
        video_folder = os.path.dirname(video_path)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        
        # Look for matching DLC file
        dlc_files = glob.glob(os.path.join(video_folder, f"{video_base}DLC*.h5"))
        
        if not dlc_files:
            messagebox.showwarning("Not Found", "No matching DLC file found.")
            return
        
        # Prefer filtered if available
        filtered = [f for f in dlc_files if 'filtered' in f.lower()]
        if filtered:
            self.pred_dlc_path.set(filtered[0])
        else:
            self.pred_dlc_path.set(dlc_files[0])
        
        messagebox.showinfo("Found", f"Found DLC file:\n{os.path.basename(self.pred_dlc_path.get())}")
    
    def auto_find_dlc_config(self):
        """Automatically find DLC config.yaml file"""
        video_path = self.pred_video_path.get()
        if not video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        
        video_folder = os.path.dirname(video_path)
        
        # Search locations
        search_paths = [
            os.path.join(video_folder, 'config.yaml'),
            os.path.join(os.path.dirname(video_folder), 'config.yaml'),
        ]
        
        # Also search subdirectories
        for root, dirs, files in os.walk(video_folder):
            if 'config.yaml' in files:
                search_paths.append(os.path.join(root, 'config.yaml'))
            # Limit depth
            if root.count(os.sep) - video_folder.count(os.sep) > 2:
                break
        
        # Find first existing config
        for path in search_paths:
            if os.path.isfile(path):
                self.pred_dlc_config_path.set(path)
                messagebox.showinfo("Found", f"Found config.yaml:\n{path}")
                return
        
        messagebox.showwarning("Not Found", 
                             "No config.yaml found in video directory or parent directories.")
    
    def preview_pred_video(self):
        """Preview video"""
        video_path = self.pred_video_path.get()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("No Video", "Please select a valid video file.")
            return
        
        try:
            preview = VideoPreviewWindow(self.root, video_path, None, None)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video:\n{str(e)}")
    
    def show_parameter_dialog(self, clf_data):
        """Show dialog to adjust prediction parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Adjust Prediction Parameters")
        _sw = dialog.winfo_screenwidth()
        _sh = dialog.winfo_screenheight()
        _dw = min(600, int(_sw * 0.45))
        _dh = min(580, int(_sh * 0.55))
        dialog.geometry(f"{_dw}x{_dh}+{(_sw-_dw)//2}+{(_sh-_dh)//2}")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {}
        
        ttk.Label(dialog, text="Adjust Prediction Parameters", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Check if we have optimized parameters
        if hasattr(self, 'optimized_params'):
            # Use optimized parameters
            default_thresh = self.optimized_params['threshold']
            default_min_bout = self.optimized_params['min_bout']
            default_min_after = self.optimized_params['min_after_bout']
            default_max_gap = self.optimized_params['max_gap']
            params_source = "Optimized Parameters"
        else:
            # Use classifier defaults
            default_thresh = clf_data.get('best_thresh', 0.5)
            default_min_bout = clf_data.get('min_bout', 1)
            default_min_after = clf_data.get('min_after_bout', 1)
            default_max_gap = clf_data.get('max_gap', 0)
            params_source = "Classifier Defaults"
        
        # Show current defaults
        defaults_frame = ttk.LabelFrame(dialog, text=params_source, padding=10)
        defaults_frame.pack(fill='x', padx=15, pady=5)
        
        ttk.Label(defaults_frame, text=f"Threshold: {default_thresh:.3f}", 
                 font=('Arial', 9)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Label(defaults_frame, text=f"Min Bout: {default_min_bout} frames", 
                 font=('Arial', 9)).grid(row=0, column=1, sticky='w', pady=2, padx=20)
        ttk.Label(defaults_frame, text=f"Min After Bout: {default_min_after} frames", 
                 font=('Arial', 9)).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Label(defaults_frame, text=f"Max Gap: {default_max_gap} frames", 
                 font=('Arial', 9)).grid(row=1, column=1, sticky='w', pady=2, padx=20)
        
        # Custom parameters
        custom_frame = ttk.LabelFrame(dialog, text="Adjust Parameters", padding=10)
        custom_frame.pack(fill='x', padx=15, pady=5)
        
        use_custom_var = tk.BooleanVar(value=False)
        custom_check = ttk.Checkbutton(custom_frame, text="Use custom parameters", 
                       variable=use_custom_var,
                       command=lambda: toggle_custom())
        custom_check.grid(row=0, column=0, columnspan=2, sticky='w', pady=5)
        
        # Threshold
        ttk.Label(custom_frame, text="Threshold:").grid(row=1, column=0, sticky='w', pady=5)
        threshold_var = tk.DoubleVar(value=default_thresh)
        threshold_spin = ttk.Spinbox(custom_frame, from_=0.01, to=0.99, increment=0.01,
                                    textvariable=threshold_var, width=12, state='disabled')
        threshold_spin.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # Min Bout
        ttk.Label(custom_frame, text="Min Bout (frames):").grid(row=2, column=0, sticky='w', pady=5)
        min_bout_var = tk.IntVar(value=default_min_bout)
        min_bout_spin = ttk.Spinbox(custom_frame, from_=1, to=1000, textvariable=min_bout_var,
                   width=12, state='disabled')
        min_bout_spin.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        
        # Min After Bout
        ttk.Label(custom_frame, text="Min After Bout (frames):").grid(row=3, column=0, sticky='w', pady=5)
        min_after_var = tk.IntVar(value=default_min_after)
        min_after_spin = ttk.Spinbox(custom_frame, from_=1, to=1000, textvariable=min_after_var,
                   width=12, state='disabled')
        min_after_spin.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        
        # Max Gap
        ttk.Label(custom_frame, text="Max Gap (frames):").grid(row=4, column=0, sticky='w', pady=5)
        max_gap_var = tk.IntVar(value=default_max_gap)
        max_gap_spin = ttk.Spinbox(custom_frame, from_=0, to=1000, textvariable=max_gap_var,
                   width=12, state='disabled')
        max_gap_spin.grid(row=4, column=1, sticky='w', pady=5, padx=5)
        
        def toggle_custom():
            state = 'normal' if use_custom_var.get() else 'disabled'
            threshold_spin.config(state=state)
            min_bout_spin.config(state=state)
            min_after_spin.config(state=state)
            max_gap_spin.config(state=state)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def on_ok():
            if use_custom_var.get():
                result['threshold'] = threshold_var.get()
                result['min_bout'] = min_bout_var.get()
                result['min_after_bout'] = min_after_var.get()
                result['max_gap'] = max_gap_var.get()
            else:
                result['threshold'] = default_thresh
                result['min_bout'] = default_min_bout
                result['min_after_bout'] = default_min_after
                result['max_gap'] = default_max_gap
            dialog.destroy()
        
        def on_cancel():
            result['cancelled'] = True
            dialog.destroy()
        
        ttk.Button(button_frame, text="Generate Predictions", 
                  command=on_ok, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left', padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        # Return None if cancelled
        if result.get('cancelled'):
            return None
        
        return result
    
    def preview_with_predictions(self):
        """Preview video side-by-side with classifier predictions"""
        video_path = self.pred_video_path.get()
        dlc_path = self.pred_dlc_path.get()
        clf_path = self.pred_classifier_path.get()
        
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("No Video", "Please select a valid video file.")
            return
        if not dlc_path or not os.path.isfile(dlc_path):
            messagebox.showwarning("No DLC File", "Please select a valid DLC pose file.")
            return
        if not clf_path or not os.path.isfile(clf_path):
            messagebox.showwarning("No Classifier", "Please select a valid classifier file.")
            return
        
        try:
            # Load classifier first to get defaults
            with open(clf_path, 'rb') as f:
                clf_data_preview = pickle.load(f)
            
            # Clean body parts lists (remove DLC network names)
            clf_data_preview['bp_include_list'] = clean_bodyparts_list(clf_data_preview.get('bp_include_list', []))
            clf_data_preview['bp_pixbrt_list'] = clean_bodyparts_list(clf_data_preview.get('bp_pixbrt_list', []))
            
            # Show parameter adjustment dialog
            param_result = self.show_parameter_dialog(clf_data_preview)
            if param_result is None:  # User cancelled
                return
            
            # Run prediction in background thread, then create window on main thread
            progress = tk.Toplevel(self.root)
            progress.title("Generating Predictions...")
            _sw, _sh = progress.winfo_screenwidth(), progress.winfo_screenheight()
            progress.geometry(f"450x170+{(_sw-450)//2}+{(_sh-170)//2}")
            
            progress_label = ttk.Label(progress, text="Loading classifier...", 
                     font=('Arial', 10))
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress, mode='indeterminate', length=300)
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Store results here
            result_data = {}
            
            def run_prediction():
                try:
                    # Load classifier
                    progress_label.config(text="Loading classifier...")
                    self.root.update()
                    
                    with open(clf_path, 'rb') as f:
                        clf_data = pickle.load(f)
                    
                    # Clean body parts lists (remove DLC network names)
                    clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
                    clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
                    
                    # Auto-detect bp_include_list if missing
                    clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=True)
                    
                    model = clf_data['clf_model']
                    behavior_name = clf_data.get('Behavior_type', 'Behavior')
                    
                    # Use custom parameters from dialog
                    best_thresh = param_result['threshold']
                    min_bout_filter = param_result['min_bout']
                    min_after_filter = param_result['min_after_bout']
                    max_gap_filter = param_result['max_gap']
                    
                    # Create feature cache directory
                    video_dir = os.path.dirname(video_path)
                    cache_dir = os.path.join(video_dir, 'PredictionCache')
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Use the same hash function as training (bp_include_list is always None
                    # in training, so force it here to guarantee matching cache keys).
                    cfg_hash = PixelPawsGUI._feature_hash_key(
                        {**clf_data, 'bp_include_list': None})

                    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
                    
                    # Check for user-provided features file first. Delegate to
                    # _load_features_for_prediction which already implements the
                    # cache-validate / pose-only-upgrade / brightness-preserve
                    # ladder.  `extract_fn=None` makes the helper return None on
                    # failure rather than re-extracting — the cache-lookup block
                    # below then tries canonical cache locations.
                    features_path = self.pred_features_path.get()
                    X = None
                    features_loaded = False

                    if features_path and os.path.isfile(features_path):
                        progress_label.config(text="Loading pre-extracted features...")
                        self.root.update()
                        X = _load_features_for_prediction(
                            cache_file=features_path,
                            model=model,
                            extract_fn=None,
                            save_path=features_path,
                            dlc_path=dlc_path,
                            clf_data=clf_data,
                        )
                        features_loaded = X is not None
                    
                    # If not loaded from file, build cache lookup list
                    if not features_loaded:
                        _proj_folder = self.current_project_folder.get()
                        _cache_fname = f"{video_name_base}_features_{cfg_hash}.pkl"
                        cache_locations = []
                        if _proj_folder and os.path.isdir(_proj_folder):
                            cache_locations.append(os.path.join(_proj_folder, 'features', _cache_fname))
                        cache_locations += [
                            os.path.join(video_dir, 'PredictionCache', _cache_fname),
                            os.path.join(video_dir, 'FeatureCache', _cache_fname),
                        ]
                        # Walk ancestor directories up to project root to handle nested video folders
                        _ancestor = video_dir
                        while True:
                            _parent = os.path.dirname(_ancestor)
                            if _parent == _ancestor:
                                break
                            _ancestor = _parent
                            cache_locations.append(os.path.join(_ancestor, 'features', _cache_fname))
                            cache_locations.append(os.path.join(_ancestor, 'FeatureCache', _cache_fname))
                            if _proj_folder and os.path.normpath(_ancestor) == os.path.normpath(_proj_folder):
                                break

                        cache_file = None
                        for loc in cache_locations:
                            if os.path.isfile(loc):
                                cache_file = loc
                                break

                    # Load or extract features if not already loaded
                    if not features_loaded:
                        progress_label.config(text="Loading / extracting features...")
                        self.root.update()
                        if not cache_file:
                            # Determine save location when no cache exists yet
                            if _proj_folder and os.path.isdir(_proj_folder):
                                cache_dir = os.path.join(_proj_folder, 'features')
                            else:
                                cache_dir = os.path.join(video_dir, 'PredictionCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, _cache_fname)
                        config_yaml = self.pred_dlc_config_path.get() or None
                        X = _load_features_for_prediction(
                            cache_file=cache_file,
                            model=model,
                            extract_fn=lambda: PixelPaws_ExtractFeatures(
                                pose_data_file=dlc_path,
                                video_file_path=video_path,
                                bp_include_list=clf_data.get('bp_include_list'),
                                bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                                square_size=clf_data.get('square_size', [40]),
                                pix_threshold=clf_data.get('pix_threshold', 0.3),
                                config_yaml_path=config_yaml,
                                include_optical_flow=clf_data.get('include_optical_flow', False),
                                bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                            ),
                            save_path=cache_file,
                            dlc_path=dlc_path,
                            clf_data=clf_data,
                        )

                    X = augment_features_post_cache(X, clf_data, model, dlc_path)

                    # Predict
                    progress_label.config(text="Running classifier...")
                    self.root.update()

                    y_proba = predict_with_xgboost(
                        model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))
                    y_pred = (y_proba >= best_thresh).astype(int)
                    
                    # Apply bout filtering with custom parameters
                    progress_label.config(text="Applying bout filtering...")
                    self.root.update()
                    
                    y_pred = self.apply_bout_filtering(
                        y_pred,
                        min_bout_filter,
                        min_after_filter,
                        max_gap_filter
                    )
                    
                    # Check for human labels
                    progress_label.config(text="Checking for human labels...")
                    self.root.update()
                    
                    human_labels = None
                    
                    # First check if user manually selected a labels file
                    manual_labels_path = self.pred_human_labels_path.get() if hasattr(self, 'pred_human_labels_path') else None
                    
                    if manual_labels_path and os.path.isfile(manual_labels_path):
                        try:
                            labels_df = pd.read_csv(manual_labels_path)
                            if behavior_name in labels_df.columns:
                                human_labels = labels_df[behavior_name].values[:len(y_pred)]
                                print(f"✓ Loaded manually selected labels: {manual_labels_path}")
                            else:
                                print(f"Warning: Behavior '{behavior_name}' not found in {manual_labels_path}")
                        except Exception as e:
                            print(f"Could not load manually selected labels: {e}")
                    
                    # If no manual selection, try auto-find
                    if human_labels is None:
                        video_name_base = os.path.splitext(os.path.basename(video_path))[0]
                        
                        possible_label_paths = [
                            os.path.join(video_dir, 'behavior_labels', f'{video_name_base}.csv'),
                            os.path.join(os.path.dirname(video_dir), 'behavior_labels', f'{video_name_base}.csv'),
                            os.path.join(video_dir, f"{video_name_base}.csv"),
                            os.path.join(video_dir, "Targets", f"{video_name_base}.csv"),
                            os.path.join(os.path.dirname(video_dir), "Targets", f"{video_name_base}.csv"),
                        ]
                        
                        for label_path in possible_label_paths:
                            if os.path.isfile(label_path):
                                try:
                                    labels_df = pd.read_csv(label_path)
                                    if behavior_name in labels_df.columns:
                                        human_labels = labels_df[behavior_name].values[:len(y_pred)]
                                        print(f"✓ Auto-found human labels: {label_path}")
                                        break
                                except Exception as e:
                                    print(f"Could not load labels from {label_path}: {e}")
                    
                    # Store results
                    result_data['success'] = True
                    result_data['video_path'] = video_path
                    result_data['y_pred'] = y_pred
                    result_data['y_proba'] = y_proba
                    result_data['behavior_name'] = behavior_name
                    result_data['best_thresh'] = best_thresh
                    result_data['clf_data'] = clf_data
                    result_data['human_labels'] = human_labels
                    
                    print(f"Predictions complete!")
                    print(f"  Video: {video_path}")
                    print(f"  Predictions shape: {y_pred.shape}")
                    print(f"  Probabilities shape: {y_proba.shape}")
                    print(f"  Human labels: {'Yes' if human_labels is not None else 'No'}")
                    
                    # Close progress and open preview ON MAIN THREAD
                    def open_preview_on_main_thread():
                        try:
                            progress.destroy()
                        except tk.TclError:
                            pass
                        
                        print("[Main Thread] Creating preview window...")
                        SideBySidePreview(
                            self.root,
                            result_data['video_path'],
                            result_data['y_pred'],
                            result_data['y_proba'],
                            result_data['behavior_name'],
                            result_data['best_thresh'],
                            result_data['clf_data'],
                            result_data['human_labels'],
                            overlay_colors={
                                'behavior':    self._hex_to_bgr(self.lv_behavior_color.get()),
                                'no_behavior': self._hex_to_bgr(self.lv_nobehavior_color.get()),
                            }
                        )
                        print("[Main Thread] Preview window created successfully!")

                    progress.destroy()

                    # Call preview directly on main thread (the original working way!)
                    print(f"Opening preview window...")
                    SideBySidePreview(self.root, video_path, y_pred, y_proba,
                                    behavior_name, best_thresh, human_labels=human_labels,
                                    overlay_colors={
                                        'behavior':    self._hex_to_bgr(self.lv_behavior_color.get()),
                                        'no_behavior': self._hex_to_bgr(self.lv_nobehavior_color.get()),
                                    },
                                    dlc_path=dlc_path)
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    
                    try:
                        progress.destroy()
                    except tk.TclError:
                        pass

                    print("="*60)
                    print("ERROR in preview_with_predictions:")
                    print("="*60)
                    print(error_detail)
                    print("="*60)
                    
                    messagebox.showerror("Error", 
                        f"Failed to generate predictions:\n\n{str(e)}\n\n"
                        f"Check the console for full error details.")
            
            threading.Thread(target=run_prediction, daemon=True).start()
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print("="*60)
            print("ERROR in preview_with_predictions (outer):")
            print("="*60)
            print(error_detail)
            print("="*60)
            messagebox.showerror("Error", f"Could not start prediction:\n{str(e)}")
    
    def log_message(self, message):
        """Thread-safe: log message to prediction results if available."""
        self._pred_log(message + '\n')

    def _pred_log(self, text):
        """Thread-safe helper for prediction text widget."""
        def _do():
            if hasattr(self, 'pred_results_text') and self.pred_results_text:
                self.pred_results_text.insert(tk.END, text)
                self.pred_results_text.see(tk.END)
        self._safe_after(_do)

    def _batch_log(self, text):
        """Thread-safe helper for batch analysis log widget."""
        def _do():
            if hasattr(self, 'batch_log') and self.batch_log:
                self.batch_log.insert(tk.END, text)
                self.batch_log.see(tk.END)
        self._safe_after(_do)
    
    def apply_bout_filtering(self, y_pred, min_bout, min_after_bout, max_gap):
        """Thin wrapper — delegates to the shared _apply_bout_filtering in evaluation_tab."""
        return _apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap)
    
    def _cancel_prediction(self):
        self._predict_cancel_flag.set()
        self._pred_log("\nCancelling prediction…\n")

    def run_single_prediction(self):
        """Run prediction on single video"""
        if not self.pred_classifier_path.get():
            messagebox.showwarning("No Classifier", "Please select a classifier file.")
            return
        if not self.pred_video_path.get():
            messagebox.showwarning("No Video", "Please select a video file.")
            return
        if not self.pred_dlc_path.get():
            messagebox.showwarning("No DLC File", "Please select a DLC pose file.")
            return
        
        self._predict_cancel_flag.clear()
        self._pred_run_btn.config(state='disabled')
        self._pred_stop_btn.config(state='normal')
        threading.Thread(target=self._predict_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Overlay color helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hex_to_bgr(hex_str):
        """Convert '#RRGGBB' to OpenCV BGR tuple."""
        h = hex_str.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def _pick_overlay_color(self, var, btn):
        """Open a color chooser, update *var* and button background."""
        from tkinter.colorchooser import askcolor
        result = askcolor(color=var.get(), title="Choose overlay color")
        if result and result[1]:
            chosen = result[1]          # '#rrggbb'
            var.set(chosen)
            btn.configure(bg=chosen, activebackground=chosen)
            self.lv_color_scheme.set('Custom')
            self._refresh_overlay_preview()

    def _on_color_scheme_changed(self):
        """Apply a named color scheme preset to the behavior/no-behavior overlay colors."""
        scheme = self.lv_color_scheme.get()
        if scheme in OVERLAY_COLOR_SCHEMES:
            beh_hex, nobeh_hex = OVERLAY_COLOR_SCHEMES[scheme]
            self.lv_behavior_color.set(beh_hex)
            self.lv_nobehavior_color.set(nobeh_hex)
            self._lv_beh_color_btn.configure(bg=beh_hex, activebackground=beh_hex)
            self._lv_nobeh_color_btn.configure(bg=nobeh_hex, activebackground=nobeh_hex)
            self._refresh_overlay_preview()

    def _on_tint_toggle(self, *_):
        """Enable / disable the tint opacity scale based on the tint checkbox."""
        state = 'normal' if self.lv_frame_tint.get() else 'disabled'
        try:
            self._lv_tint_scale.configure(state=state)
        except AttributeError:
            pass   # called before widget exists
        self._refresh_overlay_preview()

    def _refresh_overlay_preview(self, *_):
        """Draw a mock behavior-active overlay onto the in-tab preview canvas."""
        try:
            import cv2, numpy as np
            from PIL import Image, ImageTk
        except ImportError:
            return

        canvas = getattr(self, 'lv_preview_canvas', None)
        if canvas is None:
            return

        CW, CH = 320, 200

        def _placeholder(msg):
            canvas.delete('all')
            canvas.create_text(CW // 2, CH // 2, text=msg,
                               fill='gray', font=('Arial', 9))

        video_path = self.pred_video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            _placeholder("Load a video to see preview")
            return

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            _placeholder("Cannot read video")
            return

        seek = min(30, max(0, total // 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            _placeholder("Cannot read frame")
            return

        color_b  = self._hex_to_bgr(self.lv_behavior_color.get())
        hud_top  = self.lv_hud_position.get() == 'top'
        do_halo  = self.lv_halo_border.get()
        do_bouts = self.lv_bout_counter.get()

        lv_h, lv_w = frame.shape[:2]

        # HUD coords
        if hud_top:
            hud_y0, hud_y1 = 0, 80
            txt_y1, txt_y2, bar_y0, bar_y1 = 22, 58, 71, 79
        else:
            hud_y0, hud_y1 = lv_h - 80, lv_h
            txt_y1 = lv_h - 80 + 22
            txt_y2 = lv_h - 80 + 58
            bar_y0 = lv_h - 80 + 71
            bar_y1 = lv_h - 80 + 79

        # HUD background
        overlay_img = frame.copy()
        cv2.rectangle(overlay_img, (0, hud_y0), (lv_w, hud_y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.5, frame, 0.5, 0, frame)

        mock_prob = 0.85
        cv2.putText(frame, f"Frame {seek}  [preview]",
                    (8, txt_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(frame, f"Behavior: YES   p = {mock_prob:.3f}",
                    (8, txt_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_b, 2)
        if do_bouts:
            cv2.putText(frame, "Bout 1 / ?",
                        (8, txt_y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_b, 1)

        bar_w = int(lv_w * mock_prob)
        cv2.rectangle(frame, (0, bar_y0), (bar_w, bar_y1), color_b, -1)
        cv2.rectangle(frame, (0, bar_y0), (lv_w - 1, bar_y1), (80, 80, 80), 1)

        if do_halo:
            cv2.rectangle(frame, (0, 0), (lv_w - 1, lv_h - 1), color_b, 18)
            inner = tuple(max(0, c - 40) for c in color_b)
            cv2.rectangle(frame, (9, 9), (lv_w - 10, lv_h - 10), inner, 6)

        # Fit into 320×200
        aspect = lv_w / lv_h
        if CW / CH > aspect:
            nh, nw = CH, int(CH * aspect)
        else:
            nw, nh = CW, int(CW / aspect)
        frame_small = cv2.resize(frame, (nw, nh))
        img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        canvas.delete('all')
        x0, y0 = (CW - nw) // 2, (CH - nh) // 2
        canvas.create_image(x0, y0, anchor='nw', image=photo)
        self.lv_preview_photo = photo   # prevent GC

    def export_labeled_video(self):
        """Start the labeled-video export thread (on-demand after prediction)."""
        if self._last_pred_y_pred is None:
            messagebox.showwarning("No Prediction", "Run a prediction first.")
            return
        import threading as _threading
        _threading.Thread(target=self._export_labeled_video_thread, daemon=True).start()

    def _export_labeled_video_thread(self):
        """Write an annotated MP4 using the results from the last prediction run."""
        try:
            import cv2
            import numpy as np
            y_pred        = self._last_pred_y_pred
            y_proba       = self._last_pred_y_proba
            fps           = self._last_pred_fps
            n_frames      = self._last_pred_n_frames
            video_path    = self._last_pred_video_path
            behavior_name = self._last_pred_behavior_name
            output_folder = self._last_pred_output_folder
            base_name     = self._last_pred_base_name

            # Read overlay checkbox / color values
            do_skeleton = self.lv_skeleton_dots.get()
            do_tint     = self.lv_frame_tint.get()
            do_timeline = self.lv_timeline_strip.get()
            do_halo     = self.lv_halo_border.get()
            do_bouts    = self.lv_bout_counter.get()
            color_b     = self._hex_to_bgr(self.lv_behavior_color.get())
            color_nb    = self._hex_to_bgr(self.lv_nobehavior_color.get())
            tint_alpha  = float(self.lv_tint_opacity.get())
            hud_top     = self.lv_hud_position.get() == 'top'

            self._pred_log("\nCreating labeled video...\n")

            clip_start = self._parse_time_to_frames(self.pred_clip_start.get(), fps)
            clip_end   = self._parse_time_to_frames(self.pred_clip_end.get(),   fps)
            clip_start = max(0, clip_start if clip_start is not None else 0)
            clip_end   = min(n_frames, clip_end if clip_end is not None else n_frames)
            clip_end   = max(clip_start + 1, clip_end)

            labeled_path = os.path.join(output_folder, f"{base_name}_labeled.mp4")
            os.makedirs(output_folder, exist_ok=True)

            cap_lv = cv2.VideoCapture(video_path)
            lv_w   = int(cap_lv.get(cv2.CAP_PROP_FRAME_WIDTH))
            lv_h   = int(cap_lv.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(labeled_path, fourcc, fps, (lv_w, lv_h))

            # Load DLC body-part coordinates for skeleton overlay
            bp_xy = {}   # bodypart -> (x_arr, y_arr, prob_arr) each shape (n_frames,)
            _crop_dx, _crop_dy = self._last_pred_crop_offset  # offset from DLC config crop
            if do_skeleton and self._last_pred_dlc_path and os.path.exists(self._last_pred_dlc_path):
                try:
                    import pandas as _pd
                    _dlc = _pd.read_hdf(self._last_pred_dlc_path)
                    # DLC H5 has 3-level MultiIndex: (scorer, bodypart, coord)
                    _dlc.columns = _pd.MultiIndex.from_tuples(
                        [(_c[1], _c[2]) for _c in _dlc.columns])
                    for _bp in _dlc.columns.get_level_values(0).unique():
                        bp_xy[_bp] = (
                            _dlc[_bp]['x'].values.astype(float) + _crop_dx,
                            _dlc[_bp]['y'].values.astype(float) + _crop_dy,
                            _dlc[_bp]['likelihood'].values.astype(float),
                        )
                except Exception:
                    bp_xy = {}   # graceful degradation — skeleton silently skipped

            # Pre-render timeline strip image (constant across frames, cursor varies)
            total_clip = clip_end - clip_start
            timeline_img = None
            if do_timeline and total_clip > 0 and lv_h >= 20:
                timeline_img = np.zeros((14, lv_w, 3), dtype=np.uint8)
                for _x in range(lv_w):
                    _idx = clip_start + int(_x * total_clip / lv_w)
                    _idx = min(_idx, clip_end - 1)
                    _tl_b  = tuple(int(c * 0.82) for c in color_b)
                    _tl_nb = tuple(int(c * 0.82) for c in color_nb)
                    timeline_img[:, _x] = _tl_b if y_pred[_idx] == 1 else _tl_nb
                # Dim slightly so it doesn't overpower
                cv2.addWeighted(np.full((14, lv_w, 3), 20, dtype=np.uint8), 0.3,
                                timeline_img, 0.7, 0, timeline_img)

            # Pre-compute bout indices (contiguous runs of pred==1) for counter
            frame_to_bout_idx = {}
            if do_bouts:
                bout_idx  = 0
                in_bout   = False
                total_bouts = 0
                # Count bouts first
                _prev = 0
                for _p in y_pred:
                    if int(_p) == 1 and _prev == 0:
                        total_bouts += 1
                    _prev = int(_p)
                # Map frames
                bout_idx = 0
                in_bout  = False
                _prev    = 0
                for _fi, _p in enumerate(y_pred):
                    if int(_p) == 1 and _prev == 0:
                        bout_idx += 1
                        in_bout = True
                    elif int(_p) == 0:
                        in_bout = False
                    if in_bout:
                        frame_to_bout_idx[_fi] = (bout_idx, total_bouts)
                    _prev = int(_p)

            # HUD geometry (recomputed once after we know lv_h)
            cap_lv.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            for fi in range(clip_start, clip_end):
                ret, frame = cap_lv.read()
                if not ret:
                    break

                prob  = float(y_proba[fi]) if fi < len(y_proba) else 0.0
                pred  = int(y_pred[fi])    if fi < len(y_pred)  else 0
                color = color_b if pred == 1 else color_nb

                # HUD coord set
                if hud_top:
                    hud_y0, hud_y1 = 0, 80
                    txt_y1, txt_y2 = 22, 58
                    bar_y0, bar_y1 = 71, 79
                else:
                    hud_y0 = lv_h - 80
                    hud_y1 = lv_h
                    txt_y1 = hud_y0 + 22
                    txt_y2 = hud_y0 + 58
                    bar_y0 = hud_y0 + 71
                    bar_y1 = hud_y0 + 79

                # 1. Frame tint (pred==1 only, applied before HUD so HUD stays crisp)
                if do_tint and pred == 1:
                    _tint_color = tuple(c // 3 for c in color_b)
                    _tint_layer = np.full_like(frame, _tint_color)
                    cv2.addWeighted(_tint_layer, tint_alpha, frame,
                                    1.0 - tint_alpha, 0, frame)

                # 2. HUD background + text + confidence bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, hud_y0), (lv_w, hud_y1), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                ts   = fi / fps
                tstr = f"{int(ts // 3600):01d}:{int((ts % 3600) // 60):02d}:{ts % 60:05.2f}"
                cv2.putText(frame, f"Frame {fi}  [{tstr}]",
                            (8, txt_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                cv2.putText(frame,
                            f"{behavior_name}: {'YES' if pred else 'NO'}   p = {prob:.3f}",
                            (8, txt_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                if do_bouts and fi in frame_to_bout_idx:
                    _bi, _bt = frame_to_bout_idx[fi]
                    cv2.putText(frame, f"Bout {_bi} / {_bt}",
                                (8, txt_y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                bar_w = int(lv_w * prob)
                cv2.rectangle(frame, (0, bar_y0), (bar_w, bar_y1), color, -1)
                cv2.rectangle(frame, (0, bar_y0), (lv_w - 1, bar_y1), (80, 80, 80), 1)

                # 3. Skeleton dots
                if bp_xy:
                    for _bp, (_xs, _ys, _ps) in bp_xy.items():
                        if fi < len(_xs):
                            _conf = float(_ps[fi])
                            if _conf > 0.3:
                                _x, _y = int(_xs[fi]), int(_ys[fi])
                                _r = max(3, int(7 * _conf))
                                cv2.circle(frame, (_x, _y), _r, color, -1)
                                cv2.circle(frame, (_x, _y), _r + 1, (255, 255, 255), 1)

                # 4. Halo border (pred==1 only, optional)
                if do_halo and pred == 1:
                    cv2.rectangle(frame, (0, 0), (lv_w - 1, lv_h - 1), color_b, 18)
                    _inner = tuple(max(0, c - 40) for c in color_b)
                    cv2.rectangle(frame, (9, 9), (lv_w - 10, lv_h - 10), _inner, 6)

                # 5. Timeline strip (drawn last — sits on top at bottom edge)
                if timeline_img is not None:
                    _cursor_x = int((fi - clip_start) * lv_w / max(total_clip - 1, 1))
                    _cursor_x = min(_cursor_x, lv_w - 1)
                    _tl = timeline_img.copy()
                    cv2.line(_tl, (_cursor_x, 0), (_cursor_x, 13), (255, 255, 255), 2)
                    frame[lv_h - 14:lv_h, :] = _tl

                writer.write(frame)

                done = fi - clip_start + 1
                if done % 500 == 0 or done == total_clip:
                    self._pred_log(f"  Writing frame {done} / {total_clip}\n")

            cap_lv.release()
            writer.release()
            self._pred_log(f"✓ Labeled video: {labeled_path}\n")
            messagebox.showinfo("Done", f"Labeled video saved:\n{labeled_path}")

        except Exception as e:
            import traceback
            self._pred_log(f"\n✗ Export failed: {traceback.format_exc()}\n")
            self._safe_after(lambda e=e: messagebox.showerror("Error", f"Export failed:\n{str(e)}"))

    def _predict_thread(self):
        """Prediction thread with feature caching and crop handling"""
        try:
            self._safe_after(lambda: self.pred_results_text.delete('1.0', tk.END))
            self._pred_log("=" * 60 + "\n")
            self._pred_log("PixelPaws Prediction\n")
            self._pred_log("=" * 60 + "\n\n")
            
            clf_path = self.pred_classifier_path.get()
            video_path = self.pred_video_path.get()
            dlc_path = self.pred_dlc_path.get()
            features_path = self.pred_features_path.get()
            dlc_config_path = self.pred_dlc_config_path.get()
            
            self._pred_log(f"Classifier: {os.path.basename(clf_path)}\n")
            self._pred_log(f"Video: {os.path.basename(video_path)}\n")
            self._pred_log(f"DLC File: {os.path.basename(dlc_path)}\n")
            if features_path:
                self._pred_log(f"Features: {os.path.basename(features_path)}\n")
            if dlc_config_path:
                self._pred_log(f"DLC Config: {os.path.basename(dlc_config_path)}\n")
            self._pred_log("\n")
            
            # Load classifier
            self._pred_log("Loading classifier...\n")
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
            
            # Clean body parts lists (remove DLC network names)
            clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
            
            # Auto-detect bp_include_list if missing
            clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=True)
            
            model = clf_data['clf_model']
            best_thresh = clf_data['best_thresh']
            behavior_name = clf_data.get('Behavior_type', 'Behavior')
            
            self._pred_log(f"  Behavior: {behavior_name}\n")
            self._pred_log(f"  Threshold: {best_thresh:.3f}\n\n")

            if self._predict_cancel_flag.is_set():
                self._pred_log("Cancelled.\n")
                return

            # Check for DLC crop parameters
            crop_x_offset = 0
            crop_y_offset = 0
            if dlc_config_path and os.path.isfile(dlc_config_path):
                self._pred_log("Checking DLC crop parameters...\n")
                try:
                    import yaml
                    with open(dlc_config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if config.get('cropping', False):
                        crop_x_offset = config.get('x1', 0)
                        crop_y_offset = config.get('y1', 0)
                        self._pred_log(
                            f"  ✓ DLC crop detected: x+{crop_x_offset}, y+{crop_y_offset}\n")
                        self._pred_log(
                            f"  Note: Features should account for crop offset\n\n")
                    else:
                        self._pred_log("  No cropping in config\n\n")
                except ImportError:
                    self._pred_log("  ⚠️  PyYAML not installed - cannot read config\n")
                    self._pred_log("     Install with: pip install pyyaml\n\n")
                except Exception as e:
                    self._pred_log(f"  ⚠️  Could not read config: {e}\n\n")
            
            # Try to load pre-extracted features first
            X = None
            features_loaded = False

            # Get video directory (needed for cache and output)
            video_dir = os.path.dirname(video_path)

            if features_path and os.path.isfile(features_path):
                self._pred_log("Loading pre-extracted features...\n")
                try:
                    with open(features_path, 'rb') as f:
                        features_data = pickle.load(f)

                    # Handle dict wrapper (e.g. {'X': array}) or bare array/DataFrame
                    if isinstance(features_data, dict):
                        if 'X' in features_data:
                            X = features_data['X']
                        else:
                            raise ValueError(
                                f"Unrecognised features dict. Keys: {list(features_data.keys())}")
                    else:
                        X = features_data

                    features_loaded = True
                    self._pred_log(
                        f"  ✓ Loaded features: {X.shape[0]} frames, {X.shape[1]} features\n")

                    if crop_x_offset != 0 or crop_y_offset != 0:
                        self._pred_log(
                            f"  ⚠️  Pre-extracted features used with crop offset detected "
                            f"(x+{crop_x_offset}, y+{crop_y_offset}).\n"
                            f"     Ensure features were extracted with crop-corrected coordinates.\n")

                except Exception as e:
                    self._pred_log(
                        f"  ✗ Could not load features file: {e}\n"
                        f"  Falling back to feature extraction...\n\n")
                    features_loaded = False
            elif features_path:
                self._pred_log(
                    f"  ⚠️  Features file not found: {features_path}\n"
                    f"  Falling back to feature extraction...\n\n")
            
            # Extract features if not loaded
            if not features_loaded:
                self._pred_log("Proceeding with feature extraction...\n")
                # Setup feature cache
                cache_dir = os.path.join(video_dir, 'PredictionCache')
                os.makedirs(cache_dir, exist_ok=True)
                
                # Create cache key
                import hashlib
                video_name = os.path.basename(video_path)
                
                cfg_key = {
                    'bp_include_list': clf_data.get('bp_include_list'),
                    'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                    'square_size': clf_data.get('square_size', [40]),
                    'pix_threshold': clf_data.get('pix_threshold', 0.3),
                    'crop_offset': (crop_x_offset, crop_y_offset),
                    'pose_feature_version': POSE_FEATURE_VERSION,
                    'include_optical_flow': clf_data.get('include_optical_flow', False),
                    'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                }
                cfg_hash = hashlib.md5(repr(cfg_key).encode()).hexdigest()[:8]

                cache_file = os.path.join(cache_dir,
                    f"{os.path.splitext(video_name)[0]}_features_{cfg_hash}.pkl")

                # Try to load cached features (with pose-only upgrade if stale)
                if crop_x_offset != 0 or crop_y_offset != 0:
                    self._pred_log(
                        f"  Applying crop offset: x+{crop_x_offset}, y+{crop_y_offset}\n")
                try:
                    X = _load_features_for_prediction(
                        cache_file=cache_file,
                        model=model,
                        extract_fn=lambda: PixelPaws_ExtractFeatures(
                            pose_data_file=dlc_path,
                            video_file_path=video_path,
                            bp_include_list=clf_data.get('bp_include_list'),
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            crop_offset_x=crop_x_offset,
                            crop_offset_y=crop_y_offset,
                            config_yaml_path=dlc_config_path if dlc_config_path else None,
                            include_optical_flow=clf_data.get('include_optical_flow', False),
                            bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                            cancel_flag=self._predict_cancel_flag,
                        ),
                        save_path=cache_file,
                        log_fn=self._pred_log,
                        dlc_path=dlc_path,
                        clf_data=clf_data,
                    )
                except InterruptedError:
                    self._pred_log("Extraction cancelled.\n")
                    return

            X = augment_features_post_cache(X, clf_data, model, dlc_path, log_fn=self._pred_log)

            # Predict
            self._pred_log("Running classifier...\n")
            y_proba = predict_with_xgboost(
                model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))

            # Apply smoothing (bout filters / HMM Viterbi / none)
            _smooth = (self.pred_smoothing_mode.get()
                       if self.pred_smoothing_mode is not None else 'bout_filters')
            self._pred_log(f"Applying smoothing ({_smooth})...\n")
            y_raw = (y_proba >= float(clf_data.get('best_thresh', 0.5))).astype(int)
            y_pred = apply_smoothing(y_proba, clf_data, _smooth)

            raw_positive = int(np.sum(y_raw))
            filtered_positive = int(np.sum(y_pred))
            self._pred_log(f"  Raw (threshold): {raw_positive} frames\n")
            self._pred_log(f"  After smoothing: {filtered_positive} frames\n\n")

            # Calculate statistics
            n_frames = len(y_pred)
            n_positive = np.sum(y_pred)
            pct_positive = (n_positive / n_frames) * 100
            
            # Get FPS
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if not fps or fps <= 0:
                fps = 30.0
                self._pred_log("Warning: video reported FPS=0, defaulting to 30\n")

            behavior_time = n_positive / fps

            # Count bouts using shared helper
            bout_stats = count_bouts(y_pred, fps)
            n_bouts = bout_stats['n_bouts']
            bouts    = bout_stats['bouts']
            
            # Results
            self._pred_log("=" * 60 + "\n")
            self._pred_log("RESULTS\n")
            self._pred_log("=" * 60 + "\n\n")
            
            self._pred_log(f"Total frames: {n_frames}\n")
            self._pred_log(f"Behavior detected: {n_positive} frames ({pct_positive:.1f}%)\n")
            self._pred_log(f"Behavior time: {behavior_time:.1f} seconds ({behavior_time/60:.1f} minutes)\n")
            self._pred_log(f"Number of bouts: {n_bouts}\n")

            if bouts:
                self._pred_log(f"Mean bout duration:   {bout_stats['mean_dur_sec']:.2f} seconds\n")
                self._pred_log(f"Median bout duration: {bout_stats['median_dur_sec']:.2f} seconds\n")
                self._pred_log(f"Min bout duration:    {bout_stats['min_dur_sec']:.2f} seconds\n")
                self._pred_log(f"Max bout duration:    {bout_stats['max_dur_sec']:.2f} seconds\n")
            
            # Save outputs
            output_folder = self.pred_output_folder.get()
            if not output_folder:
                proj = self.current_project_folder.get()
                if proj:
                    output_folder = os.path.join(proj, 'results')
                else:
                    output_folder = video_dir
            os.makedirs(output_folder, exist_ok=True)

            # Get video base name
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]
            
            self._pred_log("\n" + "=" * 60 + "\n")
            self._pred_log("SAVING OUTPUTS\n")
            self._pred_log("=" * 60 + "\n\n")
            
            if self.pred_save_csv.get():
                csv_path = os.path.join(output_folder, f"{base_name}_predictions.csv")
                df = pd.DataFrame({
                    'frame':       np.arange(len(y_pred)),
                    'probability': y_proba,
                    behavior_name: y_pred,
                })
                df.to_csv(csv_path, index=False)
                self._pred_log(f"✓ Predictions CSV: {csv_path}\n")
            
            if self.pred_save_summary.get():
                summary_path = os.path.join(output_folder, f"{base_name}_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"PixelPaws Prediction Summary\n")
                    f.write(f"{'=' * 60}\n\n")
                    f.write(f"Video: {video_name}\n")
                    f.write(f"Behavior: {behavior_name}\n")
                    f.write(f"Classifier: {os.path.basename(clf_path)}\n\n")
                    f.write(f"Total frames: {n_frames}\n")
                    f.write(f"Behavior detected: {n_positive} frames ({pct_positive:.1f}%)\n")
                    f.write(f"Behavior time: {behavior_time:.1f} seconds\n")
                    f.write(f"Number of bouts: {len(bouts)}\n")
                    if bouts:
                        f.write(f"Mean bout duration: {bout_stats['mean_dur_sec']:.2f} seconds\n")
                
                self._pred_log(f"✓ Summary: {summary_path}\n")
            
            # Stash results for the separate labeled-video export
            self._last_pred_y_pred        = y_pred
            self._last_pred_y_proba       = y_proba
            self._last_pred_fps           = fps
            self._last_pred_n_frames      = n_frames
            self._last_pred_video_path    = video_path
            self._last_pred_behavior_name = behavior_name
            self._last_pred_output_folder = output_folder
            self._last_pred_base_name     = base_name
            self._last_pred_dlc_path      = dlc_path   # for skeleton overlay in export
            self._last_pred_crop_offset   = (crop_x_offset, crop_y_offset)
            self.pred_export_video_btn.config(state='normal')

            if self.pred_generate_ethogram.get():
                self._pred_log("Ethogram plots: coming soon\n")

            self._pred_log("\n✓ Prediction complete!\n")
            
            self._safe_after(lambda: messagebox.showinfo("Complete", "Prediction completed successfully!"))

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self._pred_log(f"\n\n{'=' * 60}\n")
            self._pred_log("✗ ERROR\n")
            self._pred_log(f"{'=' * 60}\n\n")
            self._pred_log(f"{error_detail}\n")
            self._safe_after(lambda e=e: messagebox.showerror("Error", f"Prediction failed:\n\n{str(e)}"))
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(0, lambda: self._pred_run_btn.config(state='normal'))
                    self.root.after(0, lambda: self._pred_stop_btn.config(state='disabled'))
            except tk.TclError:
                pass

    # ===== BATCH TAB METHODS =====
    
    def browse_batch_folder(self):
        """Browse for batch processing folder"""
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.batch_folder.set(folder)
            self.current_project_folder.set(folder)
    
    def browse_batch_dlc_config(self):
        """Browse for DLC config.yaml for batch processing"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=self.batch_folder.get() if self.batch_folder.get() else None
        )
        if filepath:
            self.batch_dlc_config.set(filepath)
    
    def check_batch_features(self):
        """Check feature extraction status for all videos"""
        if not self.batch_folder.get():
            messagebox.showwarning("No Folder", "Please select a data folder first.")
            return
        
        if not self.batch_classifiers:
            messagebox.showwarning("No Classifiers", "Please add at least one classifier first.")
            return
        
        self.batch_log.delete('1.0', tk.END)
        self._batch_log("Checking feature extraction status...\n\n")
        self._batch_log("SMART DEFAULT SYSTEM:\n")
        self._batch_log(f"• Brightness bodyparts: {', '.join(DEFAULT_BRIGHTNESS_BODYPARTS)}\n")
        self._batch_log(f"• All pose features extracted\n")
        self._batch_log(f"• Re-extracts only if classifier needs different bodyparts\n\n")
        
        folder = self.batch_folder.get()
        ext = self.batch_video_ext.get()
        videos = glob.glob(os.path.join(folder, f"*{ext}"))
        
        if not videos:
            self._batch_log(f"✗ No videos found with extension {ext}\n")
            return
        
        self._batch_log(f"Found {len(videos)} videos\n")
        self._batch_log(f"Checking for {len(self.batch_classifiers)} classifier(s)\n\n")
        
        total_checks = len(videos) * len(self.batch_classifiers)
        ready_count = 0
        needs_extraction_count = 0
        needs_reextraction_count = 0
        
        for video_path in videos:
            video_name = os.path.basename(video_path)
            video_dir = os.path.dirname(video_path)
            video_base = os.path.splitext(video_name)[0]
            
            self._batch_log(f"📹 {video_name}:\n")
            
            for clf_path, settings in self.batch_classifiers.items():
                clf_name = os.path.basename(clf_path)
                
                try:
                    # Load classifier
                    with open(clf_path, 'rb') as f:
                        clf_data = pickle.load(f)
                    
                    clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
                    clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
                    clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=False)
                    
                    # Check what bodyparts classifier needs
                    clf_bp_pixbrt = set(clf_data.get('bp_pixbrt_list', []))
                    smart_bp_pixbrt_set = set(DEFAULT_BRIGHTNESS_BODYPARTS)
                    needs_different_bp = not clf_bp_pixbrt.issubset(smart_bp_pixbrt_set)
                    
                    # Generate smart default hash
                    import hashlib
                    smart_cfg_key = {
                        'bp_pixbrt_list': DEFAULT_BRIGHTNESS_BODYPARTS,
                        'square_size': DEFAULT_SQUARE_SIZE,
                        'pix_threshold': DEFAULT_PIX_THRESHOLD,
                    }
                    smart_hash = hashlib.md5(repr(smart_cfg_key).encode()).hexdigest()[:8]
                    
                    # Check for smart default cache
                    _proj_folder = self.current_project_folder.get()
                    _smart_fname = f"{video_base}_features_smart_{smart_hash}.pkl"
                    smart_cache_locations = []
                    if _proj_folder and os.path.isdir(_proj_folder):
                        smart_cache_locations.append(os.path.join(_proj_folder, 'features', _smart_fname))
                    smart_cache_locations += [
                        os.path.join(video_dir, 'FeatureCache', _smart_fname),
                        os.path.join(video_dir, 'featurecache', _smart_fname),
                        os.path.join(video_dir, 'features', _smart_fname),
                        os.path.join(video_dir, 'PredictionCache', _smart_fname),
                    ]

                    cache_file = None
                    cache_type = None

                    for loc in smart_cache_locations:
                        if os.path.isfile(loc):
                            cache_file = loc
                            cache_type = "smart"
                            break

                    # If smart cache not found, scan for ANY cache file and check compatibility
                    if not cache_file:
                        _scan_dir_names = ['FeatureCache', 'featurecache', 'features', 'PredictionCache']
                        _scan_paths = []
                        if _proj_folder and os.path.isdir(_proj_folder):
                            _scan_paths.append(os.path.join(_proj_folder, 'features'))
                        _scan_paths += [os.path.join(video_dir, d) for d in _scan_dir_names]
                        for _scan_path in _scan_paths:
                            if not cache_file and os.path.isdir(_scan_path):
                                pattern = os.path.join(_scan_path, f"{video_base}_features_*.pkl")
                                matches = glob.glob(pattern)
                                if matches:
                                    cache_file = matches[0]
                                    cache_type = "old"
                    
                    if cache_file and cache_type == "smart" and not needs_different_bp:
                        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                        filename = os.path.basename(cache_file)
                        self._batch_log(f"  ✓ {clf_name}: Ready\n")
                        self._batch_log(f"     {filename} ({size_mb:.1f} MB)\n")
                        ready_count += 1
                    
                    elif cache_file and cache_type == "old":
                        # Old cache found - check if compatible by loading and inspecting
                        try:
                            with open(cache_file, 'rb') as f:
                                cached_features = pickle.load(f)
                            
                            # Check if it's a DataFrame with columns
                            if hasattr(cached_features, 'columns'):
                                n_cached = len(cached_features.columns)
                                
                                # Load model to check expected features
                                model = clf_data.get('clf_model') or clf_data.get('model')
                                n_expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                                
                                if n_expected and hasattr(model, 'feature_names_in_'):
                                    # Check if all required features are present
                                    required_features = set(model.feature_names_in_)
                                    available_features = set(cached_features.columns)
                                    missing_features = required_features - available_features
                                    
                                    if not missing_features:
                                        # Compatible! Cached features are a superset
                                        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                                        filename = os.path.basename(cache_file)
                                        self._batch_log(f"  ✓ {clf_name}: Compatible cache found\n")
                                        self._batch_log(f"     {filename} ({size_mb:.1f} MB)\n")
                                        self._batch_log(f"     Cache: {n_cached} features | Model needs: {n_expected}\n")
                                        ready_count += 1
                                    else:
                                        # Missing some features
                                        self._batch_log(f"  ⚠ {clf_name}: Cache missing {len(missing_features)} features\n")
                                        self._batch_log(f"     Will re-extract with all required bodyparts\n")
                                        needs_reextraction_count += 1
                                else:
                                    # Can't verify compatibility
                                    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                                    self._batch_log(f"  ⚠ {clf_name}: Old cache found (can't verify)\n")
                                    self._batch_log(f"     {n_cached} features ({size_mb:.1f} MB)\n")
                                    self._batch_log(f"     Will attempt to use, may re-extract if incompatible\n")
                                    ready_count += 1
                            else:
                                # Not a DataFrame, can't check
                                self._batch_log(f"  ⚠ {clf_name}: Unknown cache format\n")
                                needs_extraction_count += 1
                        except Exception as e:
                            self._batch_log(f"  ⚠ {clf_name}: Error reading cache\n")
                            self._batch_log(f"     {str(e)}\n")
                            needs_extraction_count += 1
                    
                    elif cache_file and needs_different_bp:
                        extra_bp = clf_bp_pixbrt - smart_bp_pixbrt_set
                        self._batch_log(f"  ⚠ {clf_name}: Needs re-extraction\n")
                        self._batch_log(f"     Classifier needs additional bodyparts: {', '.join(extra_bp)}\n")
                        needs_reextraction_count += 1
                    else:
                        self._batch_log(f"  ✗ {clf_name}: Not cached\n")
                        if needs_different_bp:
                            extra_bp = clf_bp_pixbrt - smart_bp_pixbrt_set
                            self._batch_log(f"     Will extract with extra bodyparts: {', '.join(extra_bp)}\n")
                        else:
                            self._batch_log(f"     Will extract with smart defaults\n")
                        needs_extraction_count += 1
                
                except Exception as e:
                    import traceback
                    self._batch_log(f"  ⚠ {clf_name}: Error\n")
                    self._batch_log(f"     {str(e)}\n")
                    needs_extraction_count += 1
            
            self._batch_log("\n")
            self.root.update_idletasks()
        
        # Summary
        self._batch_log(f"\n{'='*60}\n")
        self._batch_log(f"SUMMARY:\n")
        self._batch_log(f"{'='*60}\n")
        self._batch_log(f"Total: {total_checks}\n")
        self._batch_log(f"✓ Ready: {ready_count} ({ready_count/total_checks*100:.1f}%)\n")
        self._batch_log(f"⚠ Re-extract (extra bodyparts): {needs_reextraction_count} ({needs_reextraction_count/total_checks*100:.1f}%)\n")
        self._batch_log(f"✗ Extract (first time): {needs_extraction_count} ({needs_extraction_count/total_checks*100:.1f}%)\n\n")
        
        if ready_count == total_checks:
            self._batch_log(f"🚀 All ready! Batch will be very fast.\n")
        elif ready_count > 0:
            self._batch_log(f"⚡ {ready_count} ready, {needs_reextraction_count + needs_extraction_count} will extract.\n")
        else:
            self._batch_log(f"⏱ First run - will extract and cache.\n")
            self._batch_log(f"   Subsequent runs will be much faster!\n")
        
        messagebox.showinfo("Feature Status", 
                          f"Ready: {ready_count}/{total_checks}\n"
                          f"Re-extract: {needs_reextraction_count}/{total_checks}\n"
                          f"Extract: {needs_extraction_count}/{total_checks}")
    
    def batch_add_classifier(self):
        """Add classifier to batch list"""
        filepath = filedialog.askopenfilename(
            title="Select Classifier File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath and filepath not in self.batch_classifiers:
            # Load classifier to get defaults
            try:
                with open(filepath, 'rb') as f:
                    clf_data = pickle.load(f)
                
                self.batch_classifiers[filepath] = {
                    'use_override': False,
                    'threshold': clf_data.get('best_thresh', 0.5),
                    'min_bout': clf_data.get('min_bout', 1),
                    'min_after_bout': clf_data.get('min_after_bout', 1),
                    'max_gap': clf_data.get('max_gap', 0),
                    'bin_size_sec': 60.0
                }
            except Exception:
                # Fallback if can't load
                self.batch_classifiers[filepath] = {
                    'use_override': False,
                    'threshold': 0.5,
                    'min_bout': 1,
                    'min_after_bout': 1,
                    'max_gap': 0,
                    'bin_size_sec': 60.0
                }
            
            self.batch_clf_listbox.insert(tk.END, os.path.basename(filepath))
    
    def batch_remove_classifier(self):
        """Remove selected classifier from batch list"""
        selection = self.batch_clf_listbox.curselection()
        if selection:
            idx = selection[0]
            item = self.batch_clf_listbox.get(idx)
            # Find full path
            for path in list(self.batch_classifiers.keys()):
                if os.path.basename(path) == item:
                    del self.batch_classifiers[path]
                    break
            self.batch_clf_listbox.delete(idx)
    
    def batch_edit_classifier(self):
        """Edit settings for selected classifier"""
        selection = self.batch_clf_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a classifier first.")
            return
        
        idx = selection[0]
        item = self.batch_clf_listbox.get(idx)
        
        # Find full path
        clf_path = None
        for path in self.batch_classifiers:
            if os.path.basename(path) == item:
                clf_path = path
                break
        
        if not clf_path:
            return
        
        # Load classifier to get defaults
        try:
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
        except Exception:
            clf_data = {}
        
        # Create settings dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Settings - {item}")
        _sw = dialog.winfo_screenwidth()
        _sh = dialog.winfo_screenheight()
        _dw = min(600, int(_sw * 0.45))
        _dh = min(580, int(_sh * 0.55))
        dialog.geometry(f"{_dw}x{_dh}+{(_sw-_dw)//2}+{(_sh-_dh)//2}")
        dialog.resizable(True, True)
        
        ttk.Label(dialog, text=f"Classifier: {item}", font=('Arial', 10, 'bold')).pack(pady=10)
        
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill='both', expand=True)
        
        # Override checkbox
        use_override = tk.BooleanVar(value=self.batch_classifiers[clf_path].get('use_override', False))
        ttk.Checkbutton(frame, text="Override Classifier Defaults", 
                       variable=use_override).grid(row=0, column=0, columnspan=3, sticky='w', pady=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Classifier defaults (read-only)
        ttk.Label(frame, text="Classifier Defaults:", font=('Arial', 9, 'bold')).grid(
            row=2, column=0, columnspan=3, sticky='w', pady=5)
        
        default_frame = ttk.Frame(frame)
        default_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=10)
        
        ttk.Label(default_frame, text=f"Threshold: {clf_data.get('best_thresh', 0.5):.3f}").grid(
            row=0, column=0, sticky='w')
        ttk.Label(default_frame, text=f"Min Bout: {clf_data.get('min_bout', 'N/A')} frames").grid(
            row=0, column=1, sticky='w', padx=10)
        ttk.Label(default_frame, text=f"Max Gap: {clf_data.get('max_gap', 'N/A')} frames").grid(
            row=1, column=0, sticky='w')
        ttk.Label(default_frame, text=f"Min After: {clf_data.get('min_after_bout', 'N/A')} frames").grid(
            row=1, column=1, sticky='w', padx=10)
        
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Custom parameters
        ttk.Label(frame, text="Custom Parameters:", font=('Arial', 9, 'bold')).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=5)
        
        # Threshold
        ttk.Label(frame, text="Threshold (0-1):").grid(row=6, column=0, sticky='w', pady=5, padx=5)
        threshold_var = tk.DoubleVar(value=self.batch_classifiers[clf_path].get(
            'threshold', clf_data.get('best_thresh', 0.5)))
        ttk.Entry(frame, textvariable=threshold_var, width=10).grid(row=6, column=1, pady=5, padx=5)
        
        # Min Bout
        ttk.Label(frame, text="Min Bout (frames):").grid(row=7, column=0, sticky='w', pady=5, padx=5)
        min_bout_var = tk.IntVar(value=self.batch_classifiers[clf_path].get(
            'min_bout', clf_data.get('min_bout', 1)))
        ttk.Entry(frame, textvariable=min_bout_var, width=10).grid(row=7, column=1, pady=5, padx=5)
        
        # Min After Bout
        ttk.Label(frame, text="Min After Bout (frames):").grid(row=8, column=0, sticky='w', pady=5, padx=5)
        min_after_var = tk.IntVar(value=self.batch_classifiers[clf_path].get(
            'min_after_bout', clf_data.get('min_after_bout', 1)))
        ttk.Entry(frame, textvariable=min_after_var, width=10).grid(row=8, column=1, pady=5, padx=5)
        
        # Max Gap
        ttk.Label(frame, text="Max Gap (frames):").grid(row=9, column=0, sticky='w', pady=5, padx=5)
        max_gap_var = tk.IntVar(value=self.batch_classifiers[clf_path].get(
            'max_gap', clf_data.get('max_gap', 0)))
        ttk.Entry(frame, textvariable=max_gap_var, width=10).grid(row=9, column=1, pady=5, padx=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=10, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Time binning
        ttk.Label(frame, text="Time Bin Size (seconds):").grid(row=11, column=0, sticky='w', pady=5, padx=5)
        bin_size_var = tk.DoubleVar(value=self.batch_classifiers[clf_path].get('bin_size_sec', 60.0))
        ttk.Entry(frame, textvariable=bin_size_var, width=10).grid(row=11, column=1, pady=5, padx=5)
        
        def save_settings():
            self.batch_classifiers[clf_path]['use_override'] = use_override.get()
            self.batch_classifiers[clf_path]['threshold'] = threshold_var.get()
            self.batch_classifiers[clf_path]['min_bout'] = min_bout_var.get()
            self.batch_classifiers[clf_path]['min_after_bout'] = min_after_var.get()
            self.batch_classifiers[clf_path]['max_gap'] = max_gap_var.get()
            self.batch_classifiers[clf_path]['bin_size_sec'] = bin_size_var.get()
            dialog.destroy()
            messagebox.showinfo("Saved", "Classifier settings updated!")
        
        ttk.Button(dialog, text="💾 Save Settings", command=save_settings).pack(pady=10)
    
    def batch_autodetect_classifiers(self):
        """Add all .pkl classifiers from project + global folders to the batch list."""
        from user_config import get_global_classifiers_folder
        added = 0

        # Local project classifiers
        clf_dir = os.path.join(self.current_project_folder.get(), 'classifiers')
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    path = os.path.join(clf_dir, f)
                    if path not in self.batch_classifiers:
                        self.batch_classifiers[path] = {'min_bout_sec': 0.2, 'bin_size_sec': 60}
                        self.batch_clf_listbox.insert(tk.END, f"[Project] {f}")
                        added += 1

        # Global classifiers library
        gcf = get_global_classifiers_folder()
        if os.path.isdir(gcf):
            for f in sorted(os.listdir(gcf)):
                if f.endswith('.pkl'):
                    path = os.path.join(gcf, f)
                    if path not in self.batch_classifiers:
                        self.batch_classifiers[path] = {'min_bout_sec': 0.2, 'bin_size_sec': 60}
                        self.batch_clf_listbox.insert(tk.END, f"[Global] {f}")
                        added += 1

        if added == 0:
            messagebox.showinfo("No New Classifiers",
                                "All classifiers already added, or none found.")

    def batch_preview_mapping(self):
        """Preview video-to-DLC mapping"""
        if not self.batch_folder.get():
            messagebox.showwarning("No Folder", "Please select a data folder first.")
            return
        
        try:
            folder = self.batch_folder.get()
            ext = self.batch_video_ext.get()
            prefer_filtered = self.batch_prefer_filtered.get()

            videos_dir = os.path.join(folder, 'videos')
            search_dir = videos_dir if os.path.isdir(videos_dir) else folder
            videos = glob.glob(os.path.join(search_dir, f"*{ext}"))

            if not videos:
                messagebox.showwarning("No Videos", f"No {ext} videos found in folder.")
                return

            mapping = []
            for video_path in videos:
                dlc = self.find_dlc_for_video(video_path, search_dir, prefer_filtered)
                mapping.append((video_path, dlc))
            
            # Show preview window
            preview = tk.Toplevel(self.root)
            preview.title("Video ↔ DLC Mapping Preview")
            sw, sh = preview.winfo_screenwidth(), preview.winfo_screenheight()
            w, h = int(sw * 0.55), int(sh * 0.55)
            preview.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
            
            text = scrolledtext.ScrolledText(preview, width=100, height=25, wrap=tk.WORD)
            text.pack(fill='both', expand=True, padx=5, pady=5)
            
            text.insert(tk.END, f"Found {len(videos)} video(s) in {search_dir}\n\n")
            text.insert(tk.END, "=" * 100 + "\n\n")
            
            for video, dlc in mapping:
                video_name = os.path.basename(video)
                dlc_name = os.path.basename(dlc) if dlc else "❌ NO MATCH FOUND"
                status = "✓" if dlc else "✗"
                text.insert(tk.END, f"{status} {video_name:50s} → {dlc_name}\n")
            
            # Count matches
            matched = sum(1 for _, dlc in mapping if dlc)
            text.insert(tk.END, "\n" + "=" * 100 + "\n")
            text.insert(tk.END, f"\nSummary: {matched}/{len(videos)} videos have matching DLC files\n")
            
            if matched < len(videos):
                text.insert(tk.END, f"\n⚠ Warning: {len(videos) - matched} video(s) missing DLC files!\n")
            
            btn_frame = ttk.Frame(preview)
            btn_frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Button(btn_frame, text="Close", command=preview.destroy).pack(side='right', padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview mapping:\n{str(e)}")
    
    def find_dlc_for_video(self, video_path: str, folder: str, prefer_filtered: bool = True):
        """Find matching DLC file for video"""
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        pattern = re.compile(rf"^{re.escape(video_base)}(?!\d)", re.IGNORECASE)
        
        filtered, unfiltered = [], []
        
        for f in os.listdir(folder):
            if not f.lower().endswith('.h5'):
                continue
            base = os.path.splitext(f)[0]
            if not pattern.match(base):
                continue
            full = os.path.join(folder, f)
            (filtered if "filtered" in f.lower() else unfiltered).append(full)
        
        if prefer_filtered and filtered:
            return filtered[0]
        if unfiltered:
            return unfiltered[0]
        if filtered:
            return filtered[0]
        return None
    
    def _cancel_batch_analysis(self):
        self._batch_cancel_flag.set()
        self._batch_log("\nCancelling — aborting extraction…\n")

    def run_batch_analysis(self):
        """Run batch analysis"""
        if not self.batch_folder.get():
            messagebox.showwarning("No Folder", "Please select a data folder first.")
            return
        
        if not self.batch_classifiers:
            messagebox.showwarning("No Classifiers", "Please add at least one classifier.")
            return
        
        if not messagebox.askyesno("Start Batch", 
                                   f"Start batch analysis with {len(self.batch_classifiers)} classifier(s)?\n\n"
                                   f"This may take a while depending on the number of videos."):
            return
        
        # Run in thread
        self._batch_cancel_flag.clear()
        self._batch_run_btn.config(state='disabled')
        self._batch_stop_btn.config(state='normal')
        threading.Thread(target=self._batch_analysis_thread, daemon=True).start()
    
    def _batch_analysis_thread(self):
        """Batch analysis thread - FULLY IMPLEMENTED"""
        try:
            self._safe_after(lambda: self.batch_log.delete('1.0', tk.END))
            self._batch_log("Starting batch analysis...\n\n")
            
            folder = self.batch_folder.get()
            ext = self.batch_video_ext.get()
            videos_dir = os.path.join(folder, 'videos')
            search_dir = videos_dir if os.path.isdir(videos_dir) else folder
            videos = glob.glob(os.path.join(search_dir, f"*{ext}"))

            if not videos:
                self._batch_log(f"✗ No videos found with extension {ext}\n")
                self._safe_after(lambda: messagebox.showerror("No Videos", f"No videos found with extension {ext} in folder:\n{search_dir}"))
                return

            if not self.batch_classifiers:
                self._batch_log("✗ No classifiers added\n")
                self._safe_after(lambda: messagebox.showerror("No Classifiers", "Please add at least one classifier."))
                return
            
            self._batch_log(f"Found {len(videos)} videos\n")
            self._batch_log(f"Using {len(self.batch_classifiers)} classifier(s)\n\n")
            
            # Get output options
            save_labels = self.batch_save_labels.get()
            bin_size = 60.0

            total_operations = len(videos) * len(self.batch_classifiers)
            current_operation = 0

            # Summary results for final report
            summary_results = []
            batch_timebins_files = []
            
            for video_path in videos:
                if self._batch_cancel_flag.is_set():
                    self._batch_log("\nBatch analysis cancelled by user.\n")
                    break

                video_name = os.path.basename(video_path)
                video_dir = os.path.dirname(video_path)
                video_base = os.path.splitext(video_name)[0]
                
                self._batch_log(f"\n{'='*60}\n")
                self._batch_log(f"Processing: {video_name}\n")
                self._batch_log(f"{'='*60}\n")
                self.root.update_idletasks()
                
                # Find DLC file
                dlc_path = self.find_dlc_for_video(video_path, search_dir, self.batch_prefer_filtered.get())
                
                if not dlc_path:
                    self._batch_log(f"  ✗ No DLC file found - skipping\n")
                    current_operation += len(self.batch_classifiers)
                    progress = (current_operation / total_operations) * 100
                    self.batch_progress['value'] = progress
                    self.batch_progress_label.config(
                        text=f"Processing {current_operation}/{total_operations} ({progress:.1f}%)")
                    continue
                
                self._batch_log(f"  DLC: {os.path.basename(dlc_path)}\n\n")
                
                for clf_path, settings in self.batch_classifiers.items():
                    clf_name = os.path.basename(clf_path)
                    clf_base = os.path.splitext(clf_name)[0]
                    
                    self._batch_log(f"  → Running {clf_name}...\n")
                    self.root.update_idletasks()
                    
                    try:
                        # Load classifier
                        with open(clf_path, 'rb') as f:
                            clf_data = pickle.load(f)
                        
                        # Clean body parts
                        clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
                        clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
                        
                        # Auto-detect bp_include_list if missing
                        clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=False)
                        
                        model = clf_data.get('clf_model') or clf_data.get('model')
                        behavior_name = clf_data.get('Behavior_type', 'Behavior')
                        
                        # Results subfolder named after behavior
                        _proj_folder = self.current_project_folder.get()
                        if _proj_folder and os.path.isdir(_proj_folder):
                            results_folder = os.path.join(_proj_folder, 'results', behavior_name)
                        else:
                            results_folder = os.path.join(video_dir, 'Results', behavior_name)
                        os.makedirs(results_folder, exist_ok=True)
                        
                        # Determine parameters to use
                        use_override = settings.get('use_override', False)
                        if use_override:
                            best_thresh = settings.get('threshold', clf_data.get('best_thresh', 0.5))
                            min_bout = settings.get('min_bout', clf_data.get('min_bout', 1))
                            min_after = settings.get('min_after_bout', clf_data.get('min_after_bout', 1))
                            max_gap = settings.get('max_gap', clf_data.get('max_gap', 0))
                            self._batch_log(f"     Using custom parameters\n")
                        else:
                            best_thresh = clf_data.get('best_thresh', 0.5)
                            min_bout = clf_data.get('min_bout', 1)
                            min_after = clf_data.get('min_after_bout', 1)
                            max_gap = clf_data.get('max_gap', 0)
                            self._batch_log(f"     Using classifier defaults\n")
                        
                        # Check cache for features using SMART DEFAULT strategy
                        import hashlib
                        
                        # Smart default: Use standard brightness bodyparts
                        smart_bp_pixbrt = DEFAULT_BRIGHTNESS_BODYPARTS
                        smart_square_size = DEFAULT_SQUARE_SIZE
                        smart_pix_threshold = DEFAULT_PIX_THRESHOLD
                        
                        # Create hash based on VIDEO + DEFAULT settings (not classifier-specific)
                        smart_cfg_key = {
                            'bp_pixbrt_list': smart_bp_pixbrt,
                            'square_size': smart_square_size,
                            'pix_threshold': smart_pix_threshold,
                        }
                        smart_hash = hashlib.md5(repr(smart_cfg_key).encode()).hexdigest()[:8]
                        
                        # Check if smart default cache exists
                        _proj_folder = self.current_project_folder.get()
                        cache_locations = []
                        # Project features/ folder (canonical location)
                        if _proj_folder and os.path.isdir(_proj_folder):
                            cache_locations += [
                                os.path.join(_proj_folder, 'features', f"{video_base}_features_smart_{smart_hash}.pkl"),
                            ]
                        cache_locations += [
                            os.path.join(video_dir, 'FeatureCache', f"{video_base}_features_smart_{smart_hash}.pkl"),
                            os.path.join(video_dir, 'featurecache', f"{video_base}_features_smart_{smart_hash}.pkl"),
                            os.path.join(video_dir, 'features', f"{video_base}_features_smart_{smart_hash}.pkl"),
                            os.path.join(video_dir, 'PredictionCache', f"{video_base}_features_smart_{smart_hash}.pkl"),
                        ]

                        # Also check classifier-specific cache as fallback (must match training hash).
                        clf_hash = PixelPawsGUI._feature_hash_key(
                            {**clf_data, 'bp_include_list': None})
                        if _proj_folder and os.path.isdir(_proj_folder):
                            cache_locations += [
                                os.path.join(_proj_folder, 'features', f"{video_base}_features_{clf_hash}.pkl"),
                            ]
                        cache_locations.extend([
                            os.path.join(video_dir, 'PredictionCache', f"{video_base}_features_{clf_hash}.pkl"),
                            os.path.join(video_dir, 'FeatureCache', f"{video_base}_features_{clf_hash}.pkl"),
                            os.path.join(video_dir, 'featurecache', f"{video_base}_features_{clf_hash}.pkl"),
                            os.path.join(video_dir, 'features', f"{video_base}_features_{clf_hash}.pkl"),
                        ])

                        cache_file = None
                        cache_is_compatible = False

                        for loc in cache_locations:
                            if os.path.isfile(loc):
                                cache_file = loc
                                cache_is_compatible = True  # Exact match
                                break

                        # If no exact match, scan for ANY old cache and test compatibility
                        if not cache_file:
                            _scan_dirs = ['FeatureCache', 'featurecache', 'features', 'PredictionCache']
                            # Also scan project features/ folder
                            _extra_scan = []
                            if _proj_folder and os.path.isdir(_proj_folder):
                                _extra_scan = [os.path.join(_proj_folder, 'features')]
                            for cache_dir_path in (_extra_scan + [os.path.join(video_dir, d) for d in _scan_dirs]):
                                if os.path.isdir(cache_dir_path):
                                    pattern = os.path.join(cache_dir_path, f"{video_base}_features_*.pkl")
                                    matches = glob.glob(pattern)
                                    if matches:
                                        # Found old cache - test if compatible
                                        test_cache = matches[0]
                                        try:
                                            with open(test_cache, 'rb') as f:
                                                test_X = pickle.load(f)
                                            
                                            # Check if model's required features are present
                                            if hasattr(model, 'feature_names_in_') and hasattr(test_X, 'columns'):
                                                required_features = set(model.feature_names_in_)
                                                available_features = set(test_X.columns)
                                                missing_features = required_features - available_features
                                                
                                                if not missing_features:
                                                    # Compatible!
                                                    cache_file = test_cache
                                                    cache_is_compatible = True
                                                    self._batch_log(f"     ✓ Found compatible cache from previous extraction\n")
                                                    self._batch_log(f"        Cache: {len(available_features)} features | Model needs: {len(required_features)}\n")
                                                    break
                                        except Exception:
                                            pass  # Try next cache
                        
                        # Check if classifier needs different brightness bodyparts
                        clf_bp_pixbrt = set(clf_data.get('bp_pixbrt_list', []))
                        smart_bp_pixbrt_set = set(smart_bp_pixbrt)
                        
                        # Smart compatibility check:
                        # Cached features are usable if they are a SUPERSET of what classifier needs
                        # Example: Cache has [hrpaw, hlpaw, snout, tail] → Can be used by classifier needing [hrpaw, hlpaw, snout]
                        cached_is_superset = clf_bp_pixbrt.issubset(smart_bp_pixbrt_set)
                        
                        # Load or extract features
                        config_yaml = self.batch_dlc_config.get() if self.batch_dlc_config.get() else None
                        if cache_file and cache_is_compatible:
                            try:
                                X = _load_features_for_prediction(
                                    cache_file=cache_file,
                                    model=model,
                                    extract_fn=lambda: PixelPaws_ExtractFeatures(
                                        pose_data_file=dlc_path,
                                        video_file_path=video_path,
                                        bp_include_list=clf_data.get('bp_include_list'),
                                        bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                                        square_size=clf_data.get('square_size', [40]),
                                        pix_threshold=clf_data.get('pix_threshold', 0.3),
                                        config_yaml_path=config_yaml,
                                        include_optical_flow=clf_data.get('include_optical_flow', False),
                                        bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                                        cancel_flag=self._batch_cancel_flag,
                                    ),
                                    save_path=cache_file,
                                    log_fn=self._batch_log,
                                    dlc_path=dlc_path,
                                    clf_data=clf_data,
                                )
                            except InterruptedError:
                                self._batch_log("     Extraction cancelled.\n")
                                break

                        elif not cache_is_compatible:
                            self._batch_log(f"     Extracting features (no compatible cache found)...\n")
                            self.root.update_idletasks()
                            try:
                                X = PixelPaws_ExtractFeatures(
                                    pose_data_file=dlc_path,
                                    video_file_path=video_path,
                                    bp_include_list=clf_data.get('bp_include_list'),
                                    bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                                    square_size=clf_data.get('square_size', [40]),
                                    pix_threshold=clf_data.get('pix_threshold', 0.3),
                                    config_yaml_path=config_yaml,  # Pass config for crop detection
                                    include_optical_flow=clf_data.get('include_optical_flow', False),
                                    bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                                    cancel_flag=self._batch_cancel_flag,
                                )
                            except InterruptedError:
                                self._batch_log("     Extraction cancelled.\n")
                                break

                            # Save with classifier-specific hash
                            _proj_folder = self.current_project_folder.get()
                            if _proj_folder and os.path.isdir(_proj_folder):
                                cache_dir = os.path.join(_proj_folder, 'features')
                            else:
                                cache_dir = os.path.join(video_dir, 'FeatureCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, f"{video_base}_features_{clf_hash}.pkl")
                            _atomic_pickle_save(X, cache_file)
                            self._batch_log(f"     ✓ Features cached (classifier-specific)\n")
                        else:
                            self._batch_log(f"     Extracting features (smart defaults)...\n")
                            self._batch_log(f"     Brightness bodyparts: {', '.join(smart_bp_pixbrt)}\n")
                            self.root.update_idletasks()
                            try:
                                X = PixelPaws_ExtractFeatures(
                                    pose_data_file=dlc_path,
                                    video_file_path=video_path,
                                    bp_include_list=clf_data.get('bp_include_list'),  # All pose features
                                    bp_pixbrt_list=smart_bp_pixbrt,  # Smart default brightness
                                    square_size=smart_square_size,
                                    pix_threshold=smart_pix_threshold,
                                    config_yaml_path=config_yaml,  # Pass config for crop detection
                                    cancel_flag=self._batch_cancel_flag,
                                )
                            except InterruptedError:
                                self._batch_log("     Extraction cancelled.\n")
                                break

                            # Save with smart hash
                            _proj_folder = self.current_project_folder.get()
                            if _proj_folder and os.path.isdir(_proj_folder):
                                cache_dir = os.path.join(_proj_folder, 'features')
                            else:
                                cache_dir = os.path.join(video_dir, 'FeatureCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, f"{video_base}_features_smart_{smart_hash}.pkl")
                            _atomic_pickle_save(X, cache_file)
                            self._batch_log(f"     ✓ Features cached (reusable for most classifiers)\n")

                        X = augment_features_post_cache(X, clf_data, model, dlc_path, log_fn=self._batch_log)

                        # Predict
                        self._batch_log(f"     Running prediction...\n")
                        self.root.update_idletasks()

                        y_proba = predict_with_xgboost(
                            model, X, calibrator=clf_data.get('prob_calibrator'), fold_models=clf_data.get('fold_models'))

                        # Apply smoothing (respects the predict-tab smoothing-mode choice)
                        _smooth = (self.pred_smoothing_mode.get()
                                   if self.pred_smoothing_mode is not None
                                   else 'bout_filters')
                        y_pred_filtered = apply_smoothing(y_proba, clf_data, _smooth)

                        # Calculate statistics
                        n_frames = len(y_pred_filtered)
                        n_behavior_frames = np.sum(y_pred_filtered)
                        percent_behavior = (n_behavior_frames / n_frames) * 100 if n_frames > 0 else 0

                        # Get FPS (needed for bout durations and timebins)
                        import cv2 as _cv2
                        _cap = _cv2.VideoCapture(video_path)
                        fps = _cap.get(_cv2.CAP_PROP_FPS)
                        _cap.release()
                        if not fps or fps <= 0:
                            fps = 30.0
                            self._batch_log(f"     Warning: FPS=0, defaulting to 30\n")

                        # Count bouts using shared helper
                        bout_stats = count_bouts(y_pred_filtered, fps)
                        n_bouts           = bout_stats['n_bouts']
                        bouts             = bout_stats['bouts']
                        mean_bout_duration = bout_stats['mean_dur_sec']

                        self._batch_log(f"     ✓ Found {n_bouts} bouts ({percent_behavior:.1f}% of frames)\n")

                        # Save outputs
                        if save_labels:
                            # Frame-by-frame CSV — consistent schema across all tabs
                            output_csv = os.path.join(results_folder, f"{video_base}_{clf_base}_predictions.csv")
                            results_df = pd.DataFrame({
                                'frame':       np.arange(n_frames),
                                'probability': y_proba,
                                behavior_name: y_pred_filtered,
                            })
                            results_df.to_csv(output_csv, index=False)
                            self._batch_log(f"     ✓ Saved predictions CSV\n")

                            # Bout information CSV
                            if bouts:
                                bouts_csv = os.path.join(results_folder, f"{video_base}_{clf_base}_bouts.csv")
                                bouts_df = pd.DataFrame([{
                                    'start_frame':   b['start'],
                                    'end_frame':     b['end'],
                                    'duration_frames': b['duration_frames'],
                                    'duration_sec':  b['duration_sec'],
                                } for b in bouts])
                                bouts_df.to_csv(bouts_csv, index=False)
                                self._batch_log(f"     ✓ Saved bouts CSV ({n_bouts} bouts)\n")
                        
                        frames_per_bin = int(bin_size * fps)
                        n_bins = int(np.ceil(n_frames / frames_per_bin))

                        timebin_data = []
                        for bin_idx in range(n_bins):
                            start_frame = bin_idx * frames_per_bin
                            end_frame = min((bin_idx + 1) * frames_per_bin, n_frames)
                            bin_predictions = y_pred_filtered[start_frame:end_frame]

                            timebin_data.append({
                                'bin': bin_idx,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'start_time_sec': start_frame / fps,
                                'end_time_sec': end_frame / fps,
                                'frames_with_behavior': np.sum(bin_predictions),
                                'total_time_s': np.sum(bin_predictions) / fps if fps > 0 else 0,
                                'percent_behavior': (np.sum(bin_predictions) / len(bin_predictions)) * 100
                            })

                        timebin_csv = os.path.join(results_folder, f"{video_base}_{clf_base}_timebins.csv")
                        timebin_df = pd.DataFrame(timebin_data)
                        timebin_df.to_csv(timebin_csv, index=False)
                        self._batch_log(f"     ✓ Saved time-binned summary\n")
                        batch_timebins_files.append((behavior_name, clf_base, video_path, timebin_csv))
                        
                        # Extract subject ID from video name using utility function
                        subject_id = extract_subject_id_from_filename(video_name)
                        if not subject_id:
                            subject_id = video_base  # Fallback to full name
                        
                        # Add to summary
                        summary_results.append({
                            'video': video_name,
                            'subject_id': subject_id,  # Extracted subject ID
                            'classifier': clf_name,
                            'behavior': behavior_name,
                            'n_frames': n_frames,
                            'n_bouts': n_bouts,
                            'percent_behavior': percent_behavior,
                            'mean_bout_duration': mean_bout_duration,
                            'results_folder': results_folder
                        })
                        
                        self._batch_log(f"     ✓ Complete!\n")
                        
                    except Exception as e:
                        self._batch_log(f"     ✗ Error: {str(e)}\n")
                        print(f"Error processing {video_name} with {clf_name}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    current_operation += 1
                    progress = (current_operation / total_operations) * 100
                    self.batch_progress['value'] = progress
                    self.batch_progress_label.config(
                        text=f"Processing {current_operation}/{total_operations} ({progress:.1f}%)"
                    )
                    self.root.update_idletasks()
            
            # Save summary reports — one per behavior
            if summary_results:
                summary_df = pd.DataFrame(summary_results)
                
                # Top-level Results folder for summaries
                results_root = os.path.join(folder, "Results")
                os.makedirs(results_root, exist_ok=True)
                
                # Combined summary (all behaviors, for convenience)
                combined_csv = os.path.join(results_root, "PixelPaws_Batch_Summary.csv")
                summary_df.to_csv(combined_csv, index=False)
                
                # Per-behavior summaries in their own subfolders
                per_behavior_paths = []
                for behavior, group in summary_df.groupby('behavior'):
                    behavior_dir = os.path.join(results_root, behavior)
                    os.makedirs(behavior_dir, exist_ok=True)
                    behavior_csv = os.path.join(behavior_dir, f"PixelPaws_Summary_{behavior}.csv")
                    group.to_csv(behavior_csv, index=False)
                    per_behavior_paths.append(behavior_csv)
                
                self._batch_log(f"\n\n{'='*60}\n")
                self._batch_log(f"✓ Batch analysis complete!\n")
                self._batch_log(f"{'='*60}\n")
                self._batch_log(f"\nResults folder structure:\n")
                self._batch_log(f"  Results/\n")
                for behavior in summary_df['behavior'].unique():
                    self._batch_log(f"    {behavior}/   ← per-video outputs + summary\n")
                self._batch_log(f"\nCombined summary: {combined_csv}\n")
                self._batch_log(f"\nProcessed:\n")
                self._batch_log(f"  - {len(videos)} videos\n")
                self._batch_log(f"  - {len(self.batch_classifiers)} classifiers\n")
                self._batch_log(f"  - {len(summary_results)} successful predictions\n")
            
            self.batch_progress_label.config(text="Complete!")

            if batch_timebins_files:
                self.root.after(0, self._auto_show_batch_graph_settings, batch_timebins_files)

            # Build completion message
            behaviors_run = list({r['behavior'] for r in summary_results}) if summary_results else []
            behavior_list = '\n'.join(f'    Results/{b}/' for b in sorted(behaviors_run))
            completion_msg = (
                f"Batch analysis completed!\n\n"
                f"Processed {len(videos)} video(s) with {len(self.batch_classifiers)} classifier(s).\n\n"
                f"Output folders:\n{behavior_list}\n\n"
                f"Combined summary:\n    Results/PixelPaws_Batch_Summary.csv"
            )

            self._safe_after(lambda m=completion_msg: messagebox.showinfo("Complete", m))

        except Exception as e:
            self._batch_log(f"\n\n✗ Error: {str(e)}\n")
            self._safe_after(lambda e=e: messagebox.showerror("Error", f"Batch analysis failed:\n{str(e)}"))
            import traceback
            traceback.print_exc()
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(0, lambda: self._batch_run_btn.config(state='normal'))
                    self.root.after(0, lambda: self._batch_stop_btn.config(state='disabled'))
            except tk.TclError:
                pass

    def _auto_show_batch_graph_settings(self, timebins_files):
        """Called on main thread after batch analysis. Reads timebin CSVs and opens graph settings dialog."""
        import pandas as pd
        import os

        batch_folder = self.batch_folder.get()

        def _get_treatment(video_path, batch_folder):
            video_dir = os.path.dirname(video_path)
            if os.path.normpath(video_dir) == os.path.normpath(batch_folder):
                return os.path.basename(batch_folder)
            videos_sub = os.path.join(batch_folder, 'videos')
            if os.path.normpath(video_dir) == os.path.normpath(videos_sub):
                return os.path.basename(batch_folder)
            return os.path.basename(video_dir)

        dfs = []
        for (behavior_name, clf_label, video_path, csv_path) in timebins_files:
            if not os.path.exists(csv_path):
                continue
            try:
                tdf = pd.read_csv(csv_path)
                tdf['Behavior'] = behavior_name
                tdf['Classifier'] = clf_label
                tdf['Treatment'] = _get_treatment(video_path, batch_folder)
                # subject_id from csv or derive from video filename
                if 'subject_id' in tdf.columns:
                    tdf['Subject'] = tdf['subject_id']
                else:
                    tdf['Subject'] = os.path.splitext(os.path.basename(video_path))[0]
                # Bin_Start_Min
                if 'start_time_sec' in tdf.columns:
                    tdf['Bin_Start_Min'] = tdf['start_time_sec'] / 60.0
                elif 'bin_start_sec' in tdf.columns:
                    tdf['Bin_Start_Min'] = tdf['bin_start_sec'] / 60.0
                # Total_Time_s
                if 'total_time_s' in tdf.columns:
                    tdf['Total_Time_s'] = tdf['total_time_s']
                elif 'percent_behavior' in tdf.columns and 'start_time_sec' in tdf.columns and 'end_time_sec' in tdf.columns:
                    tdf['Total_Time_s'] = tdf['percent_behavior'] / 100.0 * (tdf['end_time_sec'] - tdf['start_time_sec'])
                elif 'behavior_time_s' in tdf.columns:
                    tdf['Total_Time_s'] = tdf['behavior_time_s']
                dfs.append(tdf)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        if not dfs:
            messagebox.showinfo("Batch Graphs", "No timebin data found to plot.")
            return

        df = pd.concat(dfs, ignore_index=True)
        behaviors = sorted(df['Behavior'].unique().tolist())

        self._open_batch_graph_settings_dialog(df, behaviors)

    def _open_batch_graph_settings_dialog(self, df, behaviors):
        """Opens the Graph Settings dialog for batch timecourse generation."""
        import tkinter as tk
        from tkinter import ttk, colorchooser

        # Named colors and palette presets (from analysis_tab)
        NAMED_COLORS = [
            'blue', 'red', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'navy', 'teal', 'gold', 'coral',
            'indigo', 'lime', 'maroon'
        ]
        PALETTE_PRESETS = {
            'Colorblind-safe': ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00', '#000000'],
            'Vivid':           ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'],
            'Pastel':          ['#AEC6CF', '#FFD1DC', '#B5EAD7', '#FFDAC1', '#C7CEEA', '#F8C8D4', '#D4E6B5', '#F9E4B7'],
        }

        # Detect treatments, sort (vehicle first)
        treatments = sorted(df['Treatment'].unique().tolist())
        VEH_KEYS = ('veh', 'vehicle', 'saline', 'control', 'ctrl', 'pbs', 'water', 'baseline')

        def veh_sort_key(t):
            tl = t.lower()
            if any(k in tl for k in VEH_KEYS):
                return (0, t)
            import re
            m = re.search(r'(\d+\.?\d*)', t)
            if m:
                return (1, float(m.group(1)))
            return (2, t)
        treatments.sort(key=veh_sort_key)

        # Default colors
        default_colors = {}
        for i, t in enumerate(treatments):
            default_colors[t] = NAMED_COLORS[i % len(NAMED_COLORS)]

        max_min = max(5, int(df['Bin_Start_Min'].max()) + 5) if 'Bin_Start_Min' in df.columns and len(df) > 0 else 60

        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Graph Settings")
        _sw, _sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        dialog.geometry(f"750x880+{(_sw-750)//2}+{(_sh-880)//2}")
        dialog.resizable(True, True)
        dialog.grab_set()
        dialog.transient(self.root)

        result = {'ok': False}

        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Canvas + scrollbar for scrollable content
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        row = 0

        # 1. Time Window
        tw_frame = ttk.LabelFrame(scroll_frame, text="1. Time Window (minutes)", padding=8)
        tw_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        scroll_frame.columnconfigure(0, weight=1)
        tw_var = tk.IntVar(value=max_min)
        ttk.Label(tw_frame, text="Show up to (minutes):").grid(row=0, column=0, sticky='w')
        ttk.Spinbox(tw_frame, from_=5, to=max_min, textvariable=tw_var, width=8).grid(row=0, column=1, padx=5)
        row += 1

        # 2. Error Bar Type
        eb_frame = ttk.LabelFrame(scroll_frame, text="2. Error Bar Type", padding=8)
        eb_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        eb_var = tk.StringVar(value='SEM')
        ttk.Radiobutton(eb_frame, text="SEM (Standard Error of Mean)", variable=eb_var, value='SEM').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(eb_frame, text="SD (Standard Deviation)", variable=eb_var, value='SD').grid(row=1, column=0, sticky='w')
        row += 1

        # 3. Heatmap Color Palette
        pal_frame = ttk.LabelFrame(scroll_frame, text="3. Heatmap Color Palette", padding=8)
        pal_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        pal_var = tk.StringVar(value='YlOrRd')
        pal_combo = ttk.Combobox(pal_frame, textvariable=pal_var, values=['YlOrRd', 'viridis', 'plasma', 'Blues', 'Reds', 'Greens', 'RdYlBu_r', 'coolwarm'], state='readonly', width=15)
        pal_combo.grid(row=0, column=0, sticky='w')
        row += 1

        # 4. Groups to Include
        gi_frame = ttk.LabelFrame(scroll_frame, text="4. Groups to Include", padding=8)
        gi_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        group_vars = {}
        for i, t in enumerate(treatments):
            v = tk.BooleanVar(value=True)
            group_vars[t] = v
            ttk.Checkbutton(gi_frame, text=t, variable=v).grid(row=i, column=0, sticky='w')
        row += 1

        # 5. Treatment Order (draggable listbox)
        to_frame = ttk.LabelFrame(scroll_frame, text="5. Treatment Order (drag to reorder)", padding=8)
        to_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        order_lb = tk.Listbox(to_frame, height=min(8, len(treatments)), selectmode='single')
        for t in treatments:
            order_lb.insert('end', t)
        order_lb.pack(fill='x')

        drag_data = {'idx': None}

        def lb_button_press(e):
            drag_data['idx'] = order_lb.nearest(e.y)

        def lb_motion(e):
            new_idx = order_lb.nearest(e.y)
            old_idx = drag_data['idx']
            if new_idx != old_idx and old_idx is not None:
                item = order_lb.get(old_idx)
                order_lb.delete(old_idx)
                order_lb.insert(new_idx, item)
                order_lb.selection_set(new_idx)
                drag_data['idx'] = new_idx
        order_lb.bind('<ButtonPress-1>', lb_button_press)
        order_lb.bind('<B1-Motion>', lb_motion)
        row += 1

        # 6. Colors
        color_frame = ttk.LabelFrame(scroll_frame, text="6. Line Colors", padding=8)
        color_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)

        color_mode_var = tk.StringVar(value='individual')
        ttk.Radiobutton(color_frame, text="Individual Colors", variable=color_mode_var, value='individual').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(color_frame, text="Gradient", variable=color_mode_var, value='gradient').grid(row=0, column=1, sticky='w')

        color_rows_frame = ttk.Frame(color_frame)
        color_rows_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=4)

        color_vars = {}
        color_previews = {}

        def update_color_preview(t):
            c = color_vars[t].get()
            try:
                color_previews[t].configure(bg=c)
            except Exception:
                pass

        def pick_custom_color(t):
            from tkinter import colorchooser as cc
            color = cc.askcolor(color=color_vars[t].get(), title=f"Color for {t}")
            if color and color[1]:
                color_vars[t].set(color[1])
                update_color_preview(t)

        for i, t in enumerate(treatments):
            v = tk.StringVar(value=default_colors[t])
            color_vars[t] = v
            ttk.Label(color_rows_frame, text=t, width=20).grid(row=i, column=0, sticky='w')
            cb = ttk.Combobox(color_rows_frame, textvariable=v, values=NAMED_COLORS, width=10)
            cb.grid(row=i, column=1, padx=4)
            cb.bind('<<ComboboxSelected>>', lambda e, t=t: update_color_preview(t))
            ttk.Button(color_rows_frame, text="Pick...", command=lambda t=t: pick_custom_color(t)).grid(row=i, column=2)
            preview = tk.Label(color_rows_frame, bg=default_colors[t], width=3)
            preview.grid(row=i, column=3, padx=4)
            color_previews[t] = preview

        # Quick palette buttons
        qp_frame = ttk.Frame(color_frame)
        qp_frame.grid(row=2, column=0, columnspan=3, sticky='w', pady=4)
        ttk.Label(qp_frame, text="Quick palette:").pack(side='left')
        for name, colors in PALETTE_PRESETS.items():
            def apply_preset(c=colors):
                for j, t in enumerate(treatments):
                    if j < len(c):
                        color_vars[t].set(c[j])
                        update_color_preview(t)
            ttk.Button(qp_frame, text=name, command=apply_preset).pack(side='left', padx=3)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill='x', pady=8, padx=10)

        def on_ok():
            settings = {
                'time_window': tw_var.get(),
                'error_type': eb_var.get(),
                'palette': pal_var.get(),
                'groups': [t for t in treatments if group_vars[t].get()],
                'treatment_order': list(order_lb.get(0, 'end')),
                'colors': {t: color_vars[t].get() for t in treatments},
                'color_mode': color_mode_var.get(),
            }
            result['ok'] = True
            result['settings'] = settings
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(btn_frame, text="✓ Generate Graphs", command=on_ok).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="✗ Cancel", command=on_cancel).pack(side='left')

        dialog.wait_window()

        if result.get('ok'):
            self.batch_graph_settings = result['settings']
            self._generate_batch_timecourses(df, behaviors, result['settings'])

    def _generate_batch_timecourses(self, df, behaviors, settings):
        """Generate timecourse graphs from batch analysis timebin data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np
        import os
        import tkinter as tk
        from tkinter import ttk

        batch_folder = self.batch_folder.get()
        results_dir = os.path.join(batch_folder, 'results')
        os.makedirs(results_dir, exist_ok=True)

        time_window = settings.get('time_window', 60)
        error_type = settings.get('error_type', 'SEM')
        colors = settings.get('colors', {})
        groups = settings.get('groups', [])
        treatment_order = settings.get('treatment_order', [])

        # Filter to selected groups and time window
        df_plot = df[df['Treatment'].isin(groups)].copy()
        if 'Bin_Start_Min' in df_plot.columns:
            df_plot = df_plot[df_plot['Bin_Start_Min'] <= time_window]

        if df_plot.empty:
            from tkinter import messagebox
            messagebox.showinfo("Batch Graphs", "No data to plot for selected groups/time window.")
            return

        # Order treatments
        ordered_groups = [t for t in treatment_order if t in groups]
        for g in groups:
            if g not in ordered_groups:
                ordered_groups.append(g)

        figures = {}
        saved_paths = []

        for behavior in behaviors:
            bdf = df_plot[df_plot['Behavior'] == behavior]
            if bdf.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

            for treatment in ordered_groups:
                tdf = bdf[bdf['Treatment'] == treatment]
                if tdf.empty:
                    continue

                grouped = tdf.groupby('Bin_Start_Min')['Total_Time_s']
                means = grouped.mean()
                if error_type == 'SEM':
                    errs = grouped.sem()
                else:
                    errs = grouped.std()

                x = means.index.values
                y = means.values
                e = errs.values

                color = colors.get(treatment, None)
                ax.plot(x, y, label=treatment, color=color, linewidth=2, marker='o', markersize=4)
                ax.fill_between(x, y - e, y + e, alpha=0.2, color=color)

            err_label = 'SEM' if error_type == 'SEM' else 'SD'
            ax.set_xlabel('Time (minutes)', fontsize=12)
            ax.set_ylabel(f'Time in Behavior (s) ± {err_label}', fontsize=12)
            ax.set_title(behavior, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save PNG
            safe_name = behavior.replace(' ', '_').replace('/', '_')
            png_path = os.path.join(results_dir, f'{safe_name}_timecourse.png')
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            saved_paths.append(png_path)
            figures[behavior] = fig

            if hasattr(self, 'batch_log') and self.batch_log:
                try:
                    self.batch_log.config(state='normal')
                    self.batch_log.insert('end', f"Saved timecourse: {png_path}\n")
                    self.batch_log.config(state='disabled')
                except Exception:
                    pass

        if not figures:
            from tkinter import messagebox
            messagebox.showinfo("Batch Graphs", "No graphs were generated.")
            return

        # Display in a Toplevel with notebook
        viewer = tk.Toplevel(self.root)
        viewer.title("Batch Timecourse Graphs")
        sw, sh = viewer.winfo_screenwidth(), viewer.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.65)
        viewer.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        nb = ttk.Notebook(viewer)
        nb.pack(fill='both', expand=True, padx=5, pady=5)

        for behavior, fig in figures.items():
            tab = ttk.Frame(nb)
            nb.add(tab, text=behavior[:20])
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(canvas, fig)
            _draw_canvas_fit(canvas, fig)

        plt.close('all')


# ============================================================================
# Active Learning v2 UI Classes
# ============================================================================

# ----------------------------------------------------------------------------
# ActiveLearningTabV2 — moved to active_learning_v2.py.
# ----------------------------------------------------------------------------
if ACTIVE_LEARNING_AVAILABLE:
    from active_learning_v2 import ActiveLearningTabV2


def main():
    """Main entry point with global exception handling"""
    # Set up global exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        import traceback
        print("="*60)
        print("UNHANDLED EXCEPTION:")
        print("="*60)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("="*60)
        
        # Show error dialog
        try:
            error_msg = str(exc_value)
            messagebox.showerror("Unhandled Error",
                               f"An unexpected error occurred:\n\n{error_msg}\n\n"
                               f"Check console for full traceback.")
        except Exception:
            pass
    
    # Install exception handler
    import sys
    sys.excepthook = handle_exception
    
    # Create root window — use ttkbootstrap if available for proper theming
    if TTKBOOTSTRAP_AVAILABLE:
        root = ttk.Window(themename=Theme._LIGHT_THEME)
        # Neutralize journal's red primary → gray for all widgets
        # (checkboxes, radio buttons, combobox borders, etc.)
        s = root.style
        s.colors.primary = '#888888'
        s.theme_use(Theme._LIGHT_THEME)
        # Buttons become solid gray after theme_use; override to light style
        s.configure('TButton', background='#f8f9fa', foreground='#333333',
                     bordercolor='#aaaaaa', lightcolor='#f8f9fa',
                     darkcolor='#aaaaaa')
        s.map('TButton',
              background=[('active', '#e9ecef'), ('pressed', '#dee2e6')],
              bordercolor=[('active', '#888888'), ('pressed', '#666666')],
              lightcolor=[('active', '#e9ecef'), ('pressed', '#dee2e6')],
              darkcolor=[('active', '#888888'), ('pressed', '#666666')],
              foreground=[('active', '#222222'), ('pressed', '#222222')])
    else:
        root = tk.Tk()
    root.withdraw()  # hidden until wizard completes

    # Also handle Tkinter callback exceptions
    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        import traceback
        print("="*60)
        print("TKINTER CALLBACK EXCEPTION:")
        print("="*60)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("="*60)

        # Show error dialog
        try:
            error_msg = str(exc_value)
            messagebox.showerror("Callback Error",
                               f"An error occurred in a callback:\n\n{error_msg}\n\n"
                               f"Check console for full traceback.")
        except Exception:
            pass

    tk.Tk.report_callback_exception = report_callback_exception

    # Configure styles
    style = ttk.Style()
    if not TTKBOOTSTRAP_AVAILABLE:
        style.theme_use('clam')

    # Accent button style — bold font for primary actions
    style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

    # Bind keyboard shortcuts
    root.bind('<F11>', lambda e: root.attributes('-fullscreen',
                                                 not root.attributes('-fullscreen')))

    # Create and run application
    app = PixelPawsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
