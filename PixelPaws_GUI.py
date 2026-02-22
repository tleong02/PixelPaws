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
- classifier_training.py - XGBoost training pipeline with GPU support

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
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter.font import Font

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
    from evaluation_tab import EvaluationTab, _apply_bout_filtering, count_bouts, find_session_triplets
    EVALUATION_TAB_AVAILABLE = True
except ImportError:
    EVALUATION_TAB_AVAILABLE = False
    print("Warning: evaluation_tab.py not found. Evaluation tab will be disabled.")

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
except ImportError:
    plt = None

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
    print("[OK] PixelPaws modules loaded successfully")
except ImportError as e:
    PIXELPAWS_MODULES_AVAILABLE = False
    POSE_FEATURE_VERSION = 1  # fallback if module unavailable
    print(f"Error: Could not import PixelPaws modules: {e}")
    print("Please ensure pose_features.py, brightness_features.py, and classifier_training.py are in the same directory")

# Import Active Learning module
try:
    import active_learning
    ACTIVE_LEARNING_AVAILABLE = True
    print("[OK] Active Learning module loaded successfully")
except ImportError as e:
    ACTIVE_LEARNING_AVAILABLE = False
    print(f"Note: Active Learning module not available: {e}")



# ============================================================================
# Helper Functions
# ============================================================================

def clean_bodyparts_list(bp_list):
    """
    Clean body parts list by removing DLC network names.
    
    Example: ['DLC_resnet50_bodypart', 'paw'] -> ['bodypart', 'paw']
    """
    if bp_list is None:
        return None
    
    cleaned = []
    for bp in bp_list:
        bp_str = str(bp)
        # Remove DLC network prefixes
        for prefix in ['DLC_resnet50_', 'DLC_resnet_', 'DLC_dlcrnetms5_', 'DLC_']:
            if bp_str.startswith(prefix):
                bp_str = bp_str[len(prefix):]
                break
        # Remove trailing underscores
        bp_str = bp_str.strip('_')
        if bp_str:  # Only add non-empty strings
            cleaned.append(bp_str)
    
    return cleaned if cleaned else None


def extract_subject_id_from_filename(filename):
    """
    Extract 4-digit subject ID from filename for batch analysis.
    
    Examples:
        '260129_Formalin_2801_PixelPaws_Left_licking_predictions.csv' -> '2801'
        '260129_Formalin_3304_PixelPaws_Left_licking_bouts.csv' -> '3304'
        'Subject_2801_video.mp4' -> '2801'
    
    Args:
        filename (str): Filename to extract subject ID from
        
    Returns:
        str or None: 4-digit subject ID if found, None otherwise
    """
    import re
    
    # Remove path if present
    filename = os.path.basename(filename)
    
    # Method 1: Find 4-digit number after underscore before another underscore/dot
    match = re.search(r'_(\d{4})(?:_|\.)', filename)
    if match:
        return match.group(1)
    
    # Method 2: Find any 4-digit number that looks like a subject ID
    # Look for patterns like DATE_EXPERIMENT_SUBJECTID_...
    parts = filename.split('_')
    for part in parts:
        if len(part) == 4 and part.isdigit():
            # Make sure it's not likely a year (skip 1900-2100)
            if not (1900 <= int(part) <= 2100):
                return part
    
    # Method 3: Find any standalone 4-digit number (not embedded in a longer digit string)
    for match in re.finditer(r'(?<!\d)(\d{4})(?!\d)', filename):
        candidate = match.group(1)
        # Skip if it looks like a year
        if not (1900 <= int(candidate) <= 2100):
            return candidate
    
    return None


def clean_bodyparts_list_ORIGINAL(bp_list):
    """
    Remove DLC network names from body parts list.
    Makes PixelPaws compatible with BAREfoot classifiers.
    
    Args:
        bp_list: List of body part names (may include DLC network names)
        
    Returns:
        Cleaned list with only actual body part names, or None if list was only DLC names
    """
    if not bp_list:
        return bp_list
    
    # Filter out DLC network names (start with 'DLC_')
    cleaned = [bp for bp in bp_list if not str(bp).startswith('DLC_')]
    
    # If list had DLC names
    if len(cleaned) < len(bp_list):
        removed = [bp for bp in bp_list if str(bp).startswith('DLC_')]
        
        # If list is now empty, it means the original list ONLY had DLC network names
        # This likely means the classifier was trained with bp_include_list=None (all body parts)
        # So we should return None to maintain that behavior
        if not cleaned:
            print(f"  Note: Body parts list only contained DLC network name '{removed[0]}'")
            print(f"  This classifier was likely trained with all body parts - returning None for compatibility")
            return None
        else:
            print(f"  Note: Filtered out DLC network names: {removed[:1]}...")
    
    return cleaned


def auto_detect_bodyparts_from_model(clf_data, verbose=True):
    """
    Auto-detect bp_include_list from model features if missing.
    
    Args:
        clf_data: Classifier data dictionary
        verbose: Whether to print detection messages
    
    Returns:
        clf_data with bp_include_list populated
    """
    # If bp_include_list is already set and not empty, don't change it
    if clf_data.get('bp_include_list'):
        return clf_data
    
    # Try to infer from model features
    model = clf_data.get('clf_model') or clf_data.get('model')
    if model and hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        
        # Collect all unique body part names from ALL feature types
        bodypart_names = set()
        
        if verbose:
            print(f"  Analyzing {len(features)} model features to detect body parts...")
        
        # From velocity features (most reliable): bodypart_Vel1, bodypart_Vel2, bodypart_Vel10
        vel_count = 0
        for f in features:
            if '_Vel' in f and 'sum_' not in f and 'Pix' not in f and 'Dis_' not in f:
                bp = f.split('_Vel')[0]
                bodypart_names.add(bp)
                vel_count += 1
        if verbose and vel_count > 0:
            print(f"    Found {len(bodypart_names)} body parts from {vel_count} velocity features")
        
        # From in-frame features: bodypart_inFrame
        inframe_count = 0
        for f in features:
            if '_inFrame' in f:
                bp = f.split('_inFrame')[0]  # Remove everything after _inFrame
                bodypart_names.add(bp)
                inframe_count += 1
        if verbose and inframe_count > 0:
            print(f"    Found {len(bodypart_names)} body parts total (added {inframe_count} inFrame features)")
        
        # From distance features: Dis_bp1-bp2
        dist_count = 0
        for f in features:
            if f.startswith('Dis_') and '_Vel' not in f:
                # Extract body part names from Dis_bp1-bp2
                dist_part = f.replace('Dis_', '')
                if '-' in dist_part:
                    parts = dist_part.split('-')
                    if len(parts) == 2:
                        bodypart_names.add(parts[0])
                        bodypart_names.add(parts[1])
                        dist_count += 1
        if verbose and dist_count > 0:
            print(f"    Found {len(bodypart_names)} body parts total (added {dist_count} distance features)")
        
        # From angle features: Ang_bp1-bp2-bp3
        angle_count = 0
        for f in features:
            if f.startswith('Ang_'):
                # Extract all three body parts from Ang_bp1-bp2-bp3
                ang_part = f.replace('Ang_', '')
                if '-' in ang_part:
                    parts = ang_part.split('-')
                    if len(parts) == 3:
                        for bp in parts:
                            bodypart_names.add(bp)
                        angle_count += 1
        if verbose and angle_count > 0:
            print(f"    Found {len(bodypart_names)} body parts total (added {angle_count} angle features)")
        
        if bodypart_names:
            inferred_bodyparts = sorted(list(bodypart_names))
            
            # Check if we found a reasonable number of body parts
            if len(inferred_bodyparts) < 5:
                if verbose:
                    print(f"  ⚠️  Only detected {len(inferred_bodyparts)} body parts: {inferred_bodyparts}")
                    print(f"  Model expects more body parts. Possible causes:")
                    print(f"    1. DLC file has different body part names")
                    print(f"    2. Model was trained with different DLC network")
                    print(f"  Will attempt to use all body parts from DLC file...")
                clf_data['bp_include_list'] = None
                return clf_data
            
            clf_data['bp_include_list'] = inferred_bodyparts
            if verbose:
                print(f"  ✓ Auto-detected {len(inferred_bodyparts)} body parts from model:")
                print(f"    {inferred_bodyparts}")
            return clf_data
    
    # Could not auto-detect
    if verbose:
        print("  ⚠️  Could not auto-detect body parts - will use all from DLC file")
    clf_data['bp_include_list'] = None
    return clf_data


def PixelPaws_ExtractFeatures(pose_data_file, video_file_path, bp_pixbrt_list,
                             square_size, pix_threshold, bp_include_list=None,
                             scale_x=1, scale_y=1, dt_vel=2, min_prob=0.8, use_gpu=True,
                             crop_offset_x=0, crop_offset_y=0, config_yaml_path=None,
                             include_optical_flow=False, bp_optflow_list=None):
    """
    Extract features using new modular system (with fallback to original).
    
    This wrapper maintains backward compatibility while using the new
    pose_features.py and brightness_features.py modules when available.
    
    Args:
        pose_data_file: Path to DLC tracking file
        video_file_path: Path to video file
        bp_pixbrt_list: Body parts for brightness features
        square_size: ROI size for brightness
        pix_threshold: Brightness threshold
        bp_include_list: Body parts for pose features (None = all)
        scale_x, scale_y: Scaling factors
        dt_vel: Time delta for derivatives
        min_prob: Minimum DLC confidence
        use_gpu: Use GPU acceleration for brightness extraction (default True, auto-fallback to CPU)
        crop_offset_x: X offset for DLC crop (overrides config_yaml_path)
        crop_offset_y: Y offset for DLC crop (overrides config_yaml_path)
        config_yaml_path: Path to DLC config.yaml for auto-detecting crop (optional)
        
    Returns:
        DataFrame with all features (pose + brightness)
    """
    if not PIXELPAWS_MODULES_AVAILABLE:
        raise ImportError(
            "PixelPaws modules not found. Please ensure these files are in the same directory:\n"
            "  - pose_features.py\n"
            "  - brightness_features.py\n"
            "  - classifier_training.py"
        )
    
    print("  Extracting features with PixelPaws modules...")
    
    # Try to auto-detect crop from config.yaml if provided and offsets not explicitly set
    if config_yaml_path and crop_offset_x == 0 and crop_offset_y == 0:
        try:
            import yaml
            with open(config_yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config.get('cropping', False):
                crop_offset_x = config.get('x1', 0)
                crop_offset_y = config.get('y1', 0)
                print(f"  ✓ Detected DLC crop from config: x+{crop_offset_x}, y+{crop_offset_y}")
        except ImportError:
            print(f"  ⚠️  PyYAML not installed - cannot read config.yaml")
            print(f"     Install with: pip install pyyaml")
            print(f"     Config file: {config_yaml_path}")
        except Exception as e:
            print(f"  ⚠️  Could not read config.yaml: {e}")
    
    if crop_offset_x != 0 or crop_offset_y != 0:
        print(f"  Applying crop offset to brightness extraction: x+{crop_offset_x}, y+{crop_offset_y}")
    
    # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
    # For bp_include_list: None means "use all body parts" (valid)
    # For bp_pixbrt_list: Keep as-is even if empty (brightness needs specific body parts)
    bp_include_list_cleaned = clean_bodyparts_list(bp_include_list)
    bp_pixbrt_list_cleaned = clean_bodyparts_list(bp_pixbrt_list)
    
    # Special handling: if bp_pixbrt_list becomes None after cleaning, it's likely wrong
    # We need explicit body parts for brightness features
    if bp_pixbrt_list is not None and bp_pixbrt_list_cleaned is None:
        print("  Warning: bp_pixbrt_list became None after cleaning DLC names")
        print("  Original list:", bp_pixbrt_list)
        # Keep the cleaned version
        bp_pixbrt_list_cleaned = []
    
    # If bp_pixbrt_list is empty but original had values, something went wrong
    if not bp_pixbrt_list_cleaned and bp_pixbrt_list:
        print("  Warning: bp_pixbrt_list was not empty but cleaning resulted in empty list")
        print(f"  Original: {bp_pixbrt_list}")
        # Try to recover by removing only DLC_ prefix
        bp_pixbrt_list_cleaned = [str(bp).replace('DLC_', '').strip('_') for bp in bp_pixbrt_list 
                                  if not str(bp).startswith('DLC_') or len(str(bp)) > 10]
        if bp_pixbrt_list_cleaned:
            print(f"  Recovered: {bp_pixbrt_list_cleaned}")
    
    # 1. Extract pose features
    if bp_include_list_cleaned is None or len(bp_include_list_cleaned) == 0:
        # Load DLC file to get all body parts
        print("  Auto-detecting body parts from DLC file...")
        if pose_data_file.endswith('.h5'):
            dlc_df = pd.read_hdf(pose_data_file)
        else:
            dlc_df = pd.read_csv(pose_data_file, header=[0, 1, 2], index_col=0)
        
        # Extract body part names
        if isinstance(dlc_df.columns, pd.MultiIndex):
            # Multi-index: first level might be scorer, second level is body parts
            if dlc_df.columns.nlevels > 2:
                dlc_df.columns = dlc_df.columns.droplevel(0)  # Remove scorer
            # Get unique body part names from first level
            bp_include_list_cleaned = list(dlc_df.columns.get_level_values(0).unique())
            # Filter out scorer name if it's still there
            bp_include_list_cleaned = [bp for bp in bp_include_list_cleaned if not bp.startswith('DLC_')]
        else:
            # Flat columns: extract from column names like 'bodypart_x', 'bodypart_y'
            bp_include_list_cleaned = list(set([col.split('_')[0] for col in dlc_df.columns if '_x' in col]))
        
        print(f"  Detected {len(bp_include_list_cleaned)} body parts: {bp_include_list_cleaned}")
    
    pose_extractor = PoseFeatureExtractor(
        bodyparts=bp_include_list_cleaned,
        likelihood_threshold=min_prob,
        velocity_delta=dt_vel  # Use dt_vel parameter (default 2, matching BAREfoot)
    )
    
    # DEBUG: Print what body parts we're actually using
    print(f"  Body parts for pose features: {bp_include_list_cleaned}")
    print(f"  Number of body parts: {len(bp_include_list_cleaned) if bp_include_list_cleaned else 0}")
    
    X_pose = pose_extractor.extract_all_features(pose_data_file)
    
    # 2. Extract brightness features
    print(f"  Body parts for brightness features: {bp_pixbrt_list_cleaned}")
    if crop_offset_x != 0 or crop_offset_y != 0:
        print(f"  ✓ Applying crop offset to brightness extraction: x+{crop_offset_x}, y+{crop_offset_y}")
        print(f"     (DLC coordinates will be shifted to match full video frame)")
    
    brightness_extractor = PixelBrightnessExtractor(
        bodyparts_to_track=bp_pixbrt_list_cleaned,
        square_size=square_size if isinstance(square_size, int) else square_size[0],
        pixel_threshold=pix_threshold,
        min_prob=min_prob,
        use_gpu=use_gpu,  # Pass GPU setting through
        crop_offset_x=crop_offset_x,  # NEW: Pass crop offset
        crop_offset_y=crop_offset_y   # NEW: Pass crop offset
    )

    # Build an optical flow extractor preloaded with DLC coords if requested.
    # It will be passed into the brightness loop so both run in a single video pass.
    of_extractor = None
    if include_optical_flow and bp_optflow_list:
        try:
            from optical_flow_features import OpticalFlowExtractor
            of_extractor = OpticalFlowExtractor(
                bodyparts=bp_optflow_list,
                min_prob=min_prob,
            ).preload(pose_data_file)
            print(f"  Optical flow will be co-extracted with brightness (single pass)")
        except Exception as e:
            print(f"  ⚠ Could not prepare optical flow extractor: {e}")

    X_brightness = brightness_extractor.extract_brightness_features(
        dlc_file=pose_data_file,
        video_file=video_file_path,
        dt_vel=dt_vel,
        create_video=False,
        optical_flow_extractor=of_extractor,
    )
    
    # 3. Combine features
    X = pd.concat([X_pose, X_brightness], axis=1)

    print(f"  ✓ Extracted {X.shape[1]} features from {X.shape[0]} frames")
    return X



def predict_with_xgboost(model, X):
    """
    Predict with XGBoost model, handling GPU models and feature selection.
    
    CRITICAL: Selects only the features the model was trained on, in correct order.
    This is essential for BAREfoot compatibility and prevents feature mismatch errors.
    
    Args:
        model: Trained XGBoost model
        X: Feature DataFrame (may have more/different features than model needs)
        
    Returns:
        Array of prediction probabilities
    """
    try:
        # CRITICAL: Select only features the model was trained on
        if hasattr(model, 'feature_names_in_'):
            # Check if all required features are present
            missing_features = set(model.feature_names_in_) - set(X.columns)
            if missing_features:
                # Show first 10 missing features for debugging
                missing_list = list(missing_features)[:10]
                raise ValueError(
                    f"Model requires {len(missing_features)} features that are missing from extracted features.\n"
                    f"First 10 missing: {missing_list}\n"
                    f"This usually means:\n"
                    f"  - Model was trained with different body parts, or\n"
                    f"  - Model was trained with different velocity settings, or\n"
                    f"  - Feature extraction version mismatch.\n"
                    f"Model expects {len(model.feature_names_in_)} features total."
                )
            
            # Select features in correct order (critical for XGBoost!)
            X_model = X[model.feature_names_in_]
            print(f"  Selected {len(model.feature_names_in_)} features for prediction")
        else:
            # Older model without feature names - use all features
            print("  Warning: Model doesn't have feature_names_in_. Using all features.")
            X_model = X
        
        # Check if model has device parameter (XGBoost model with GPU)
        if hasattr(model, 'get_params') and hasattr(model, 'set_params'):
            params = model.get_params()
            current_device = params.get('device', None)
            tree_method = params.get('tree_method', None)
            
            # If model uses GPU tree_method but GPU not available, switch to CPU
            if tree_method == 'gpu_hist':
                try:
                    # Try prediction with GPU
                    y_proba = model.predict_proba(X_model)[:, 1]
                    return y_proba
                except Exception as gpu_error:
                    # GPU not available, switch to CPU tree method
                    print(f"  GPU not available ({str(gpu_error)}), switching to CPU (hist)")
                    model.set_params(tree_method='hist')
                    y_proba = model.predict_proba(X_model)[:, 1]
                    return y_proba
            
            # If model is on GPU device, temporarily switch to CPU for prediction
            if current_device and ('cuda' in str(current_device) or 'gpu' in str(current_device)):
                # Set to CPU
                model.set_params(device='cpu')
                
                # Predict
                y_proba = model.predict_proba(X_model)[:, 1]
                
                # Restore GPU device
                model.set_params(device=current_device)
                
                return y_proba
    except ValueError:
        # Re-raise feature mismatch errors with full context
        raise
    except Exception as e:
        # Log other errors but continue with fallback
        print(f"  Warning during prediction setup: {e}")
        # Use X_model if we got that far, otherwise use X
        X_model = X_model if 'X_model' in locals() else X
    
    # Default prediction (no device switching needed)
    return model.predict_proba(X_model)[:, 1]


class Theme:
    """Theme management for light/dark modes"""
    
    # Light theme colors
    LIGHT = {
        'bg': '#f0f0f0',
        'fg': '#000000',
        'select_bg': '#0078d7',
        'select_fg': '#ffffff',
        'button_bg': '#e1e1e1',
        'entry_bg': '#ffffff',
        'frame_bg': '#ffffff',
        'text_bg': '#ffffff',
        'highlight': '#0078d7',
        'border': '#cccccc',
        'plot_bg': '#ffffff',
        'plot_fg': '#000000',
    }
    
    # Dark theme colors
    DARK = {
        'bg': '#2b2b2b',
        'fg': '#e0e0e0',
        'select_bg': '#0078d7',
        'select_fg': '#ffffff',
        'button_bg': '#3c3c3c',
        'entry_bg': '#3c3c3c',
        'frame_bg': '#2b2b2b',
        'text_bg': '#1e1e1e',
        'highlight': '#0078d7',
        'border': '#3c3c3c',
        'plot_bg': '#2b2b2b',
        'plot_fg': '#e0e0e0',
    }
    
    def __init__(self, mode='light'):
        self.mode = mode
        self.colors = self.LIGHT if mode == 'light' else self.DARK
    
    def toggle(self):
        """Toggle between light and dark mode"""
        self.mode = 'dark' if self.mode == 'light' else 'light'
        self.colors = self.DARK if self.mode == 'dark' else self.LIGHT
        return self.mode
    
    def apply_to_widget(self, widget, widget_type='frame'):
        """Apply theme to a widget"""
        try:
            if widget_type in ['frame', 'labelframe']:
                widget.configure(bg=self.colors['frame_bg'])
            elif widget_type == 'label':
                widget.configure(bg=self.colors['frame_bg'], fg=self.colors['fg'])
            elif widget_type == 'button':
                widget.configure(bg=self.colors['button_bg'], fg=self.colors['fg'])
            elif widget_type == 'entry':
                widget.configure(bg=self.colors['entry_bg'], fg=self.colors['fg'],
                               insertbackground=self.colors['fg'])
            elif widget_type == 'text':
                widget.configure(bg=self.colors['text_bg'], fg=self.colors['fg'],
                               insertbackground=self.colors['fg'])
        except:
            pass


class VideoPreviewWindow:
    """Video preview window with prediction overlay"""
    
    def __init__(self, parent, video_path, dlc_path, predictions=None):
        self.window = tk.Toplevel(parent)
        self.window.title("Video Preview with Predictions")
        self.window.geometry("1200x700")
        
        self.video_path = video_path
        self.dlc_path = dlc_path
        self.predictions = predictions
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.playing = False
        
        self.setup_ui()
        self.load_frame(0)
        
    def setup_ui(self):
        """Setup preview window UI"""
        # Video display
        self.canvas = tk.Canvas(self.window, width=960, height=540, bg='black')
        self.canvas.pack(pady=10)
        
        # Controls frame
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=10, pady=5)
        
        # Playback controls
        ttk.Button(controls, text="⏮", command=self.prev_frame, width=3).pack(side='left', padx=2)
        self.play_btn = ttk.Button(controls, text="▶", command=self.toggle_play, width=3)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(controls, text="⏭", command=self.next_frame, width=3).pack(side='left', padx=2)
        
        # Frame slider
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(controls, from_=0, to=self.total_frames-1, 
                               orient='horizontal', variable=self.frame_var,
                               command=self.on_slider_change)
        self.slider.pack(side='left', fill='x', expand=True, padx=10)
        
        # Frame info
        self.frame_label = ttk.Label(controls, text="Frame: 0 / 0")
        self.frame_label.pack(side='left', padx=5)
        
        # Behavior bouts list
        if self.predictions is not None:
            bouts_frame = ttk.LabelFrame(self.window, text="Behavior Bouts", padding=5)
            bouts_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            self.bouts_listbox = tk.Listbox(bouts_frame, height=5)
            self.bouts_listbox.pack(side='left', fill='both', expand=True)
            
            scrollbar = ttk.Scrollbar(bouts_frame, orient='vertical', 
                                     command=self.bouts_listbox.yview)
            scrollbar.pack(side='right', fill='y')
            self.bouts_listbox.config(yscrollcommand=scrollbar.set)
            
            self.bouts_listbox.bind('<<ListboxSelect>>', self.on_bout_select)
            
            # Populate bouts
            self.populate_bouts()
    
    def populate_bouts(self):
        """Find and list behavior bouts"""
        if self.predictions is None:
            return
        
        preds = np.array(self.predictions).flatten()
        
        # Find bouts (consecutive 1s)
        in_bout = False
        bout_start = 0
        
        for i, val in enumerate(preds):
            if val == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif val == 0 and in_bout:
                duration = (i - bout_start) / self.fps
                self.bouts_listbox.insert(tk.END, 
                    f"Bout: frames {bout_start}-{i-1} ({duration:.2f}s)")
                in_bout = False
        
        if in_bout:
            duration = (len(preds) - bout_start) / self.fps
            self.bouts_listbox.insert(tk.END, 
                f"Bout: frames {bout_start}-{len(preds)-1} ({duration:.2f}s)")
    
    def on_bout_select(self, event):
        """Jump to selected bout"""
        selection = self.bouts_listbox.curselection()
        if selection:
            text = self.bouts_listbox.get(selection[0])
            # Extract start frame
            import re
            match = re.search(r'frames (\d+)-', text)
            if match:
                frame = int(match.group(1))
                self.load_frame(frame)
    
    def load_frame(self, frame_num):
        """Load and display a specific frame"""
        self.current_frame = max(0, min(frame_num, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Draw prediction overlay
            if self.predictions is not None and self.current_frame < len(self.predictions):
                pred = self.predictions[self.current_frame]
                color = (0, 255, 0) if pred == 1 else (255, 0, 0)
                cv2.rectangle(frame, (10, 10), (60, 40), color, -1)
                cv2.putText(frame, "BEHAVIOR" if pred == 1 else "NO", 
                           (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Resize for display
            height, width = frame.shape[:2]
            scale = min(960/width, 540/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(480, 270, image=self.photo)
        
        # Update UI
        self.frame_var.set(self.current_frame)
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames-1}")
    
    def on_slider_change(self, value):
        """Handle slider movement"""
        if not self.playing:
            self.load_frame(int(float(value)))
    
    def prev_frame(self):
        """Go to previous frame"""
        self.load_frame(self.current_frame - 1)
    
    def next_frame(self):
        """Go to next frame"""
        self.load_frame(self.current_frame + 1)
    
    def toggle_play(self):
        """Toggle playback"""
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "▶")
        
        if self.playing:
            self.play_video()
    
    def play_video(self):
        """Play video automatically"""
        if self.playing and self.current_frame < self.total_frames - 1:
            self.load_frame(self.current_frame + 1)
            delay = int(1000 / self.fps)
            self.window.after(delay, self.play_video)
        else:
            self.playing = False
            self.play_btn.config(text="▶")


class TrainingVisualizationWindow:
    """Real-time training progress visualization"""
    
    def __init__(self, parent, theme):
        self.window = tk.Toplevel(parent)
        self.window.title("Training Progress")
        self.window.geometry("900x600")
        self.theme = theme
        
        self.fold_f1_scores = []
        self.fold_times = []
        self.current_fold = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup visualization UI"""
        # Create notebook for different plots
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # F1 Score plot
        self.f1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.f1_frame, text="F1 Scores")
        
        self.f1_fig = Figure(figsize=(8, 5), facecolor=self.theme.colors['plot_bg'])
        self.f1_ax = self.f1_fig.add_subplot(111)
        self.f1_ax.set_facecolor(self.theme.colors['plot_bg'])
        self.f1_canvas = FigureCanvasTkAgg(self.f1_fig, self.f1_frame)
        self.f1_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Timing plot
        self.time_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_frame, text="Fold Times")
        
        self.time_fig = Figure(figsize=(8, 5), facecolor=self.theme.colors['plot_bg'])
        self.time_ax = self.time_fig.add_subplot(111)
        self.time_ax.set_facecolor(self.theme.colors['plot_bg'])
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, self.time_frame)
        self.time_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(self.window, height=6, wrap=tk.WORD)
        self.status_text.pack(fill='x', padx=5, pady=5)
        
        self.update_plots()
    
    def add_fold_result(self, fold_num, f1_score, precision, recall, time_elapsed):
        """Add results from a completed fold"""
        self.fold_f1_scores.append({
            'fold': fold_num,
            'f1': f1_score,
            'precision': precision,
            'recall': recall
        })
        self.fold_times.append(time_elapsed)
        self.current_fold = fold_num
        
        self.update_plots()
        self.update_status()
    
    def update_plots(self):
        """Update all plots"""
        if not self.fold_f1_scores:
            return
        
        # F1 Score plot
        self.f1_ax.clear()
        folds = [r['fold'] for r in self.fold_f1_scores]
        f1s = [r['f1'] for r in self.fold_f1_scores]
        precs = [r['precision'] for r in self.fold_f1_scores]
        recs = [r['recall'] for r in self.fold_f1_scores]
        
        self.f1_ax.bar([f - 0.2 for f in folds], f1s, 0.2, label='F1', alpha=0.8)
        self.f1_ax.bar(folds, precs, 0.2, label='Precision', alpha=0.8)
        self.f1_ax.bar([f + 0.2 for f in folds], recs, 0.2, label='Recall', alpha=0.8)
        
        self.f1_ax.set_xlabel('Fold', color=self.theme.colors['plot_fg'])
        self.f1_ax.set_ylabel('Score', color=self.theme.colors['plot_fg'])
        self.f1_ax.set_title('Cross-Validation Scores', color=self.theme.colors['plot_fg'])
        self.f1_ax.legend()
        self.f1_ax.set_ylim([0, 1])
        self.f1_ax.tick_params(colors=self.theme.colors['plot_fg'])
        self.f1_ax.spines['bottom'].set_color(self.theme.colors['plot_fg'])
        self.f1_ax.spines['left'].set_color(self.theme.colors['plot_fg'])
        self.f1_ax.spines['top'].set_visible(False)
        self.f1_ax.spines['right'].set_visible(False)
        
        self.f1_canvas.draw()
        
        # Timing plot
        self.time_ax.clear()
        self.time_ax.bar(folds, self.fold_times, alpha=0.8, color='steelblue')
        self.time_ax.set_xlabel('Fold', color=self.theme.colors['plot_fg'])
        self.time_ax.set_ylabel('Time (seconds)', color=self.theme.colors['plot_fg'])
        self.time_ax.set_title('Training Time per Fold', color=self.theme.colors['plot_fg'])
        self.time_ax.tick_params(colors=self.theme.colors['plot_fg'])
        self.time_ax.spines['bottom'].set_color(self.theme.colors['plot_fg'])
        self.time_ax.spines['left'].set_color(self.theme.colors['plot_fg'])
        self.time_ax.spines['top'].set_visible(False)
        self.time_ax.spines['right'].set_visible(False)
        
        self.time_canvas.draw()
    
    def update_status(self):
        """Update status text"""
        if self.fold_f1_scores:
            mean_f1 = np.mean([r['f1'] for r in self.fold_f1_scores])
            mean_time = np.mean(self.fold_times)
            
            status = f"\n=== Current Progress ===\n"
            status += f"Completed Folds: {len(self.fold_f1_scores)}\n"
            status += f"Mean F1 Score: {mean_f1:.3f}\n"
            status += f"Mean Time per Fold: {mean_time:.1f}s\n"
            
            if len(self.fold_times) > 1:
                est_remaining = mean_time * (5 - len(self.fold_f1_scores))  # Assuming 5 folds
                status += f"Estimated Time Remaining: {est_remaining:.0f}s\n"
            
            self.status_text.insert(tk.END, status)
            self.status_text.see(tk.END)


class AutoLabelWindow:
    """Auto-labeling suggestion tool"""
    
    def __init__(self, parent, video_path, dlc_path, classifier_path):
        self.window = tk.Toplevel(parent)
        self.window.title("Auto-Label Suggestions")
        self.window.geometry("1200x700")
        
        self.video_path = video_path
        self.dlc_path = dlc_path
        self.classifier_path = classifier_path
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.predictions = None
        self.probabilities = None
        self.labels = np.zeros(self.total_frames, dtype=int)  # User labels
        self.current_frame = 0
        self.uncertain_frames = []
        
        self.setup_ui()
        self.run_predictions()
    
    def setup_ui(self):
        """Setup auto-label UI"""
        # Top: Instructions
        inst_frame = ttk.Frame(self.window)
        inst_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(inst_frame, text="Review uncertain predictions and correct as needed. "
                                   "Focus on frames with probability 0.4-0.6.",
                 wraplength=800).pack()
        
        # Middle: Video display
        self.canvas = tk.Canvas(self.window, width=800, height=450, bg='black')
        self.canvas.pack(pady=10)
        
        # Controls
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls, text="⏮ Prev Uncertain", 
                  command=self.prev_uncertain).pack(side='left', padx=2)
        ttk.Button(controls, text="⏭ Next Uncertain", 
                  command=self.next_uncertain).pack(side='left', padx=2)
        
        # Labeling buttons
        label_frame = ttk.LabelFrame(controls, text="Label Current Frame", padding=5)
        label_frame.pack(side='left', padx=20)
        
        ttk.Button(label_frame, text="✓ Behavior (1)", 
                  command=lambda: self.label_frame(1)).pack(side='left', padx=2)
        ttk.Button(label_frame, text="✗ No Behavior (0)", 
                  command=lambda: self.label_frame(0)).pack(side='left', padx=2)
        
        # Info display
        self.info_label = ttk.Label(controls, text="")
        self.info_label.pack(side='left', padx=20)
        
        # Bottom: Statistics and export
        bottom_frame = ttk.Frame(self.window)
        bottom_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_label = ttk.Label(bottom_frame, text="")
        self.stats_label.pack(side='left')
        
        ttk.Button(bottom_frame, text="Export Labels", 
                  command=self.export_labels).pack(side='right', padx=5)
        ttk.Button(bottom_frame, text="Refresh Stats", 
                  command=self.update_stats).pack(side='right', padx=5)
    
    def run_predictions(self):
        """Run classifier and find uncertain frames"""
        messagebox.showinfo("Processing", "Running classifier to find uncertain frames...")
        
        # TODO: Actually run classifier here
        # For now, simulate with random data
        np.random.seed(42)
        self.probabilities = np.random.beta(2, 2, self.total_frames)
        self.predictions = (self.probabilities > 0.5).astype(int)
        
        # Find uncertain frames (probability between 0.4 and 0.6)
        self.uncertain_frames = np.where(
            (self.probabilities > 0.4) & (self.probabilities < 0.6)
        )[0].tolist()
        
        self.update_stats()
        
        if self.uncertain_frames:
            self.load_frame(self.uncertain_frames[0])
    
    def load_frame(self, frame_num):
        """Load and display frame with prediction info"""
        self.current_frame = frame_num
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if ret:
            # Add info overlay
            prob = self.probabilities[frame_num]
            pred = self.predictions[frame_num]
            label = self.labels[frame_num]
            
            # Color based on probability
            if prob < 0.4:
                color = (0, 0, 255)  # Red - confident no
            elif prob > 0.6:
                color = (0, 255, 0)  # Green - confident yes
            else:
                color = (255, 165, 0)  # Orange - uncertain
            
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Prediction: {'Behavior' if pred == 1 else 'No Behavior'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Probability: {prob:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if label != 0:  # Show user label if set
                cv2.putText(frame, f"Your Label: {'Behavior' if label == 1 else 'No Behavior'}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Resize and display
            height, width = frame.shape[:2]
            scale = min(800/width, 450/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            from PIL import Image, ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete("all")
            self.canvas.create_image(400, 225, image=self.photo)
        
        # Update info
        uncertain_idx = self.uncertain_frames.index(frame_num) if frame_num in self.uncertain_frames else -1
        if uncertain_idx >= 0:
            self.info_label.config(
                text=f"Uncertain frame {uncertain_idx + 1} of {len(self.uncertain_frames)}"
            )
    
    def prev_uncertain(self):
        """Go to previous uncertain frame"""
        if not self.uncertain_frames:
            return
        
        current_idx = self.uncertain_frames.index(self.current_frame) if self.current_frame in self.uncertain_frames else 0
        prev_idx = (current_idx - 1) % len(self.uncertain_frames)
        self.load_frame(self.uncertain_frames[prev_idx])
    
    def next_uncertain(self):
        """Go to next uncertain frame"""
        if not self.uncertain_frames:
            return
        
        current_idx = self.uncertain_frames.index(self.current_frame) if self.current_frame in self.uncertain_frames else -1
        next_idx = (current_idx + 1) % len(self.uncertain_frames)
        self.load_frame(self.uncertain_frames[next_idx])
    
    def label_frame(self, label):
        """Set label for current frame"""
        self.labels[self.current_frame] = label + 1  # Store as 1 or 2 (0 = unlabeled)
        self.update_stats()
        self.next_uncertain()
    
    def update_stats(self):
        """Update statistics display"""
        labeled = np.sum(self.labels > 0)
        behavior_count = np.sum(self.labels == 2)
        no_behavior_count = np.sum(self.labels == 1)
        
        stats = f"Labeled: {labeled} / {len(self.uncertain_frames)} uncertain frames | "
        stats += f"Behavior: {behavior_count} | No Behavior: {no_behavior_count}"
        
        self.stats_label.config(text=stats)
    
    def export_labels(self):
        """Export corrected labels"""
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if output_path:
            # Convert labels: 0=unlabeled, 1=no behavior, 2=behavior
            final_labels = np.where(self.labels == 0, self.predictions, self.labels - 1)
            
            df = pd.DataFrame({
                'frame': range(len(final_labels)),
                'label': final_labels,
                'probability': self.probabilities,
                'user_corrected': (self.labels > 0).astype(int)
            })
            
            df.to_csv(output_path, index=False)
            messagebox.showinfo("Exported", f"Labels saved to:\n{output_path}")



class SideBySidePreview:
    """Simple fast side-by-side preview"""
    
    def __init__(self, parent, video_path, predictions, probabilities, behavior_name, threshold, human_labels=None):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Prediction Preview - {behavior_name}")
        self.window.geometry("1400x750")
        
        # Make window more prominent
        self.window.transient()  # Independent window
        self.window.focus_force()
        self.window.lift()
        
        self.video_path = video_path
        self.predictions = predictions
        self.probabilities = probabilities
        self.behavior_name = behavior_name
        self.threshold = threshold
        self.human_labels = human_labels
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 10.0  # Default 10x speed for faster review
        
        # Graph update throttling
        self.frame_counter = 0
        self.graph_update_interval = 10  # Update marker every N frames during playback
        self.graph_redraw_counter = 0
        self.graph_redraw_interval = 30  # Redraw graph every N frames during playback
        
        # Graph update lock to prevent re-entrant calls
        self.updating_graph = False
        
        # Timeline update lock to prevent callback loop
        self.updating_timeline = False
        
        # Graph window reference (initialize before setup_ui)
        self.graph_window_obj = None
        self.graph_window_var = tk.IntVar(value=1000)
        
        self.setup_ui()
        self.update_frame()
    def setup_ui(self):
        # Info
        info = ttk.Frame(self.window)
        info.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(info, text=f"Behavior: {self.behavior_name} | Threshold: {self.threshold:.3f}",
                 font=('Arial', 10, 'bold')).pack(side='left')
        
        n_pos = np.sum(self.predictions)
        pct = (n_pos / len(self.predictions)) * 100
        
        # Add human label comparison if available
        if self.human_labels is not None and len(self.human_labels) > 0:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Handle length mismatch - use minimum length
            min_length = min(len(self.human_labels), len(self.predictions))
            
            if min_length < len(self.predictions):
                print(f"Warning: Human labels ({len(self.human_labels)} frames) shorter than predictions ({len(self.predictions)} frames)")
                print(f"Comparison will only cover first {min_length} frames")
            
            # Truncate to matching length
            human_labels_subset = self.human_labels[:min_length]
            predictions_subset = self.predictions[:min_length]
            
            # Calculate metrics
            accuracy = accuracy_score(human_labels_subset, predictions_subset) * 100
            f1 = f1_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            precision = precision_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            recall = recall_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            
            comparison_text = f"Detected: {n_pos} ({pct:.1f}%) | " \
                            f"Comparison ({min_length}/{len(self.predictions)} frames): " \
                            f"Acc: {accuracy:.1f}% | F1: {f1:.1f}% | " \
                            f"Prec: {precision:.1f}% | Rec: {recall:.1f}%"
            ttk.Label(info, text=comparison_text, font=('Arial', 9), 
                     foreground='blue').pack(side='right', padx=10)
        else:
            ttk.Label(info, text=f"Detected: {n_pos} frames ({pct:.1f}%)",
                     font=('Arial', 9)).pack(side='right')
        
        # Video panel (single, centered)
        video_frame = ttk.LabelFrame(self.window, text="Video with Predictions", padding=5)
        video_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.canvas_video = tk.Canvas(video_frame, bg='black')
        self.canvas_video.pack(fill='both', expand=True)
        
        # Controls
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=5, pady=5)
        
        self.play_btn = ttk.Button(controls, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side='left', padx=2)
        
        ttk.Button(controls, text="⏮ -100", command=lambda: self.jump(-100)).pack(side='left', padx=2)
        ttk.Button(controls, text="◀ -10", command=lambda: self.jump(-10)).pack(side='left', padx=2)
        ttk.Button(controls, text="▶ +10", command=lambda: self.jump(10)).pack(side='left', padx=2)
        ttk.Button(controls, text="⏭ +100", command=lambda: self.jump(100)).pack(side='left', padx=2)
        
        # Bout navigation
        ttk.Separator(controls, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(controls, text="⬅ Prev Bout", command=self.jump_to_prev_bout).pack(side='left', padx=2)
        ttk.Button(controls, text="Next Bout ➡", command=self.jump_to_next_bout).pack(side='left', padx=2)
        
        self.frame_label = ttk.Label(controls, text="Frame: 0 / 0")
        self.frame_label.pack(side='left', padx=20)
        
        # Playback speed controls
        ttk.Label(controls, text="Speed:").pack(side='left', padx=(20, 5))
        speed_frame = ttk.Frame(controls)
        speed_frame.pack(side='left')
        
        ttk.Button(speed_frame, text="1x", width=4, 
                  command=lambda: self.set_speed(1.0)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="5x", width=4,
                  command=lambda: self.set_speed(5.0)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="10x", width=4,
                  command=lambda: self.set_speed(10.0)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="20x", width=4,
                  command=lambda: self.set_speed(20.0)).pack(side='left', padx=1)
        
        self.speed_label = ttk.Label(controls, text=f"{self.playback_speed:.0f}x", 
                                     font=('Arial', 9, 'bold'))
        self.speed_label.pack(side='left', padx=5)
        
        # Show Graph button on the right
        ttk.Button(controls, text="📊 Show Graph", 
                  command=self.open_graph_window).pack(side='right', padx=5)
        
        # Timeline
        timeline_frame = ttk.Frame(self.window)
        timeline_frame.pack(fill='x', padx=5, pady=5)
        
        self.timeline = ttk.Scale(timeline_frame, from_=0, to=self.total_frames-1,
                                 orient='horizontal', command=self.on_timeline)
        self.timeline.pack(fill='x', expand=True)
        
        # Prediction bars
        self.pred_canvas = tk.Canvas(timeline_frame, height=30, bg='white')
        self.pred_canvas.pack(fill='x', expand=True, pady=2)
        self.pred_canvas.bind('<Button-1>', self.on_pred_click)
        self.draw_pred_timeline()
        
        # Keys
        self.window.bind('<space>', lambda e: self.toggle_play())
        self.window.bind('<Left>', lambda e: self.jump(-1))
        self.window.bind('<Right>', lambda e: self.jump(1))
    
    def draw_pred_timeline(self):
        self.pred_canvas.delete('all')
        w = self.pred_canvas.winfo_width() or 1200
        h = 30
        
        # Simple bars
        downsample = max(1, len(self.predictions) // 1000)
        for i in range(0, len(self.predictions), downsample):
            if self.predictions[i] == 1:
                x = (i / len(self.predictions)) * w
                self.pred_canvas.create_line(x, 0, x, h, fill='red', width=1)
        
        self.update_marker()
    
    def update_marker(self):
        self.pred_canvas.delete('marker')
        w = self.pred_canvas.winfo_width() or 1200
        x = (self.current_frame / self.total_frames) * w
        self.pred_canvas.create_line(x, 0, x, 30, fill='blue', width=2, tags='marker')
    
    def on_pred_click(self, event):
        """Handle click on prediction timeline"""
        w = self.pred_canvas.winfo_width() or 1200
        frame = int((event.x / w) * self.total_frames)
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update_frame()
    
    def update_frame(self):
        if self.current_frame < 0:
            self.current_frame = 0
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames - 1
            self.playing = False
            self.play_btn.config(text="▶ Play")
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Single video with overlay
            frame_display = frame.copy()
            if self.current_frame < len(self.predictions):
                pred = self.predictions[self.current_frame]
                prob = self.probabilities[self.current_frame]
                
                h, w = frame_display.shape[:2]
                if pred == 1:
                    # Behavior detected - use red text (no tint)
                    color = (0, 0, 255)  # Red in BGR
                    text = "BEHAVIOR DETECTED"
                else:
                    color = (0, 255, 0)  # Green in BGR
                    text = "No Behavior"
                
                # Draw text overlay
                cv2.putText(frame_display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame_display, f"Probability: {prob:.3f}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame_display, f"Frame: {self.current_frame}", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show human label comparison if available
                if self.human_labels is not None and self.current_frame < len(self.human_labels):
                    human_label = int(self.human_labels[self.current_frame])
                    
                    # Check if prediction matches human label
                    if pred == human_label:
                        match_text = "CORRECT"
                        match_color = (0, 255, 0)  # Green
                    else:
                        match_text = "MISMATCH"
                        match_color = (0, 165, 255)  # Orange
                    
                    human_text = f"Human: {'Behavior' if human_label == 1 else 'No Behavior'}"
                    cv2.putText(frame_display, human_text, (20, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame_display, match_text, (20, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, match_color, 2)
            
            self.show_frame(frame_display, self.canvas_video)
        
        # Update controls (protect from callback loop)
        self.updating_timeline = True
        self.timeline.set(self.current_frame)
        self.updating_timeline = False
        
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
        
        # Only update marker every N frames during playback to reduce lag
        if self.playing:
            self.frame_counter += 1
            if self.frame_counter >= self.graph_update_interval:
                self.update_marker()
                self.frame_counter = 0
        else:
            # Always update when paused
            self.update_marker()
        
        if self.playing:
            self.current_frame += 1
            # Use playback speed multiplier
            delay_ms = int(1000 / (self.fps * self.playback_speed))
            self.window.after(delay_ms, self.update_frame)
    
    def show_frame(self, frame, canvas):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cw = canvas.winfo_width() or 640
        ch = canvas.winfo_height() or 480
        
        h, w = frame_rgb.shape[:2]
        aspect = w / h
        
        if cw / ch > aspect:
            nh = ch
            nw = int(ch * aspect)
        else:
            nw = cw
            nh = int(cw / aspect)
        
        frame_resized = cv2.resize(frame_rgb, (nw, nh))
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)
        
        canvas.delete('all')
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        canvas.create_image(x, y, anchor='nw', image=photo)
        canvas.image = photo
    
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸ Pause")
            self.frame_counter = 0  # Reset counter
            self.update_frame()
        else:
            self.play_btn.config(text="▶ Play")
            # Update graph when stopped
            self.update_marker()  # Update marker immediately
            self.window.after(50, self.safe_update_graph)  # Update graph after brief delay
    
    def set_speed(self, speed):
        """Set playback speed multiplier"""
        self.playback_speed = speed
        self.speed_label.config(text=f"{speed:.0f}x")
    
    def jump(self, delta):
        self.current_frame += delta
        self.update_frame()
        # Update graph after jumping
        self.window.after(100, self.safe_update_graph)
    
    def jump_to_next_bout(self):
        """Jump to start of next behavior bout"""
        # Find all bouts (continuous sequences of predictions == 1)
        bouts = []
        in_bout = False
        bout_start = None
        
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif self.predictions[i] == 0 and in_bout:
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        if in_bout:  # Close final bout
            bouts.append((bout_start, len(self.predictions) - 1))
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected in video.")
            return
        
        # Find next bout after current frame
        for start, end in bouts:
            if start > self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.window.after(100, self.safe_update_graph)
                return
        
        # No bout found forward, wrap to first bout
        self.current_frame = bouts[0][0]
        self.update_frame()
        self.window.after(100, self.safe_update_graph)
        messagebox.showinfo("Wrapped", f"Jumped to first bout (frame {bouts[0][0]})")
    
    def jump_to_prev_bout(self):
        """Jump to start of previous behavior bout"""
        # Find all bouts
        bouts = []
        in_bout = False
        bout_start = None
        
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif self.predictions[i] == 0 and in_bout:
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        if in_bout:
            bouts.append((bout_start, len(self.predictions) - 1))
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected in video.")
            return
        
        # Find previous bout before current frame
        for start, end in reversed(bouts):
            if start < self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.window.after(100, self.safe_update_graph)
                return
        
        # No bout found backward, wrap to last bout
        self.current_frame = bouts[-1][0]
        self.update_frame()
        self.window.after(100, self.safe_update_graph)
        messagebox.showinfo("Wrapped", f"Jumped to last bout (frame {bouts[-1][0]})")
    
    def jump_to_frame_input(self):
        """Jump to frame from input box"""
        try:
            frame = int(self.jump_frame_var.get())
            if 0 <= frame < self.total_frames:
                self.current_frame = frame
                self.update_frame()
                self.jump_frame_var.set("")  # Clear input
                self.window.after(100, self.safe_update_graph)
            else:
                messagebox.showwarning("Invalid Frame", 
                    f"Frame must be between 0 and {self.total_frames - 1}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid frame number")
    
    def on_timeline(self, value):
        if self.updating_timeline:
            return  # Ignore callbacks while we're setting the value
        
        self.current_frame = int(float(value))
        if not self.playing:
            self.update_frame()
            # Update graph after scrubbing
            self.window.after(100, self.safe_update_graph)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def open_graph_window(self):
        """Open probability graph in separate window"""
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            # Window already open, just focus it
            self.graph_window_obj.lift()
            self.graph_window_obj.focus_force()
            self.update_graph_window()
            return
        
        # Create new window
        self.graph_window_obj = tk.Toplevel(self.window)
        self.graph_window_obj.title(f"Probability Graph - {self.behavior_name}")
        self.graph_window_obj.geometry("1200x400")
        
        # Controls at top - split into two rows
        controls_container = ttk.Frame(self.graph_window_obj)
        controls_container.pack(fill='x', padx=5, pady=5)
        
        # Top row - Window size and navigation
        controls_top = ttk.Frame(controls_container)
        controls_top.pack(fill='x', pady=2)
        
        ttk.Label(controls_top, text="Window Size:").pack(side='left', padx=2)
        tk.Spinbox(controls_top, from_=100, to=10000, increment=100,
                   textvariable=self.graph_window_var, width=8).pack(side='left', padx=2)
        ttk.Label(controls_top, text="frames").pack(side='left', padx=2)
        
        ttk.Button(controls_top, text="Refresh", 
                  command=self.update_graph_window).pack(side='left', padx=10)
        
        # Add bout navigation
        ttk.Separator(controls_top, orient='vertical').pack(side='left', padx=10, fill='y')
        ttk.Label(controls_top, text="Bouts:").pack(side='left', padx=2)
        ttk.Button(controls_top, text="⬅ Prev", 
                  command=self.jump_to_prev_bout).pack(side='left', padx=2)
        ttk.Button(controls_top, text="Next ➡", 
                  command=self.jump_to_next_bout).pack(side='left', padx=2)
        
        # Add mismatch navigation if human labels available
        if self.human_labels is not None and len(self.human_labels) > 0:
            ttk.Separator(controls_top, orient='vertical').pack(side='left', padx=10, fill='y')
            ttk.Label(controls_top, text="Mismatches:").pack(side='left', padx=2)
            
            ttk.Button(controls_top, text="⬅ Prev", 
                      command=self.jump_to_prev_mismatch).pack(side='left', padx=2)
            ttk.Button(controls_top, text="Next ➡", 
                      command=self.jump_to_next_mismatch).pack(side='left', padx=2)
        
        # Bottom row - Frame info and jump
        controls_bottom = ttk.Frame(controls_container)
        controls_bottom.pack(fill='x', pady=2)
        
        self.graph_current_frame_label = ttk.Label(controls_bottom, 
                                                    text=f"Current Frame: {self.current_frame}/{self.total_frames}",
                                                    font=('Arial', 9, 'bold'))
        self.graph_current_frame_label.pack(side='left', padx=10)
        
        ttk.Separator(controls_bottom, orient='vertical').pack(side='left', padx=10, fill='y')
        
        # Jump to frame input
        ttk.Label(controls_bottom, text="Jump to frame:").pack(side='left', padx=5)
        self.jump_frame_var = tk.StringVar()
        self.jump_frame_entry = ttk.Entry(controls_bottom, textvariable=self.jump_frame_var, width=10)
        self.jump_frame_entry.pack(side='left', padx=2)
        ttk.Button(controls_bottom, text="Go", command=self.jump_to_frame_input).pack(side='left', padx=2)
        self.jump_frame_entry.bind('<Return>', lambda e: self.jump_to_frame_input())
        
        # Add scrollbar at bottom BEFORE graph (important for pack order!)
        scrollbar_frame = ttk.Frame(self.graph_window_obj)
        scrollbar_frame.pack(fill='x', side='bottom', padx=5, pady=5)
        
        ttk.Label(scrollbar_frame, text="Timeline:").pack(side='left', padx=5)
        
        self.graph_scrollbar = ttk.Scale(scrollbar_frame, from_=0, to=self.total_frames-1,
                                         orient='horizontal', 
                                         command=self.on_graph_scrollbar)
        self.graph_scrollbar.pack(side='left', fill='x', expand=True, padx=5)
        self.graph_scrollbar.set(self.current_frame)
        
        # Frame indicator
        self.graph_frame_label = ttk.Label(scrollbar_frame, 
                                           text=f"{self.current_frame}/{self.total_frames}",
                                           width=15)
        self.graph_frame_label.pack(side='right', padx=5)
        
        # Graph canvas (pack AFTER scrollbar so scrollbar stays at bottom)
        graph_frame = ttk.Frame(self.graph_window_obj)
        graph_frame.pack(fill='both', expand=True, padx=5, pady=(5, 0))
        
        # Create matplotlib figure
        self.graph_fig = Figure(figsize=(12, 4), dpi=100, facecolor='white')
        self.graph_ax = self.graph_fig.add_subplot(111)
        
        # Embed in window
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bind click event to jump to frame
        self.graph_canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        # Draw initial graph
        self.update_graph_window()
    
    def on_graph_click(self, event):
        """Handle click on graph to jump to frame"""
        if event.inaxes != self.graph_ax:
            return  # Click outside plot area
        
        # Get clicked x-coordinate (frame number)
        clicked_frame = int(event.xdata)
        
        # Clamp to valid range
        clicked_frame = max(0, min(clicked_frame, self.total_frames - 1))
        
        # Jump to that frame
        self.current_frame = clicked_frame
        self.update_frame()
        self.update_graph_window()
    
    def on_graph_scrollbar(self, value):
        """Handle scrollbar movement in graph window"""
        frame = int(float(value))
        self.current_frame = frame
        self.update_frame()
        
        # Update graph and scrollbar label
        if hasattr(self, 'graph_frame_label'):
            self.graph_frame_label.config(text=f"{frame}/{self.total_frames}")
        
        # Debounce graph update - only update after user stops dragging
        if hasattr(self, '_scrollbar_update_id'):
            self.window.after_cancel(self._scrollbar_update_id)
        self._scrollbar_update_id = self.window.after(100, self.update_graph_window)
    
    def jump_to_next_mismatch(self):
        """Jump to next frame where prediction doesn't match human label"""
        if self.human_labels is None or len(self.human_labels) == 0:
            return
        
        min_len = min(len(self.predictions), len(self.human_labels))
        
        # Search forward from current frame
        for i in range(self.current_frame + 1, min_len):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # If no mismatch found forward, wrap around from beginning
        for i in range(0, self.current_frame):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # No mismatches found
        messagebox.showinfo("No Mismatches", "No mismatches found in the labeled frames!")
    
    def jump_to_prev_mismatch(self):
        """Jump to previous frame where prediction doesn't match human label"""
        if self.human_labels is None or len(self.human_labels) == 0:
            return
        
        min_len = min(len(self.predictions), len(self.human_labels))
        
        # Search backward from current frame
        for i in range(self.current_frame - 1, -1, -1):
            if i < min_len and self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # If no mismatch found backward, wrap around from end
        for i in range(min_len - 1, self.current_frame, -1):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # No mismatches found
        messagebox.showinfo("No Mismatches", "No mismatches found in the labeled frames!")
    
    def find_behavior_bouts(self):
        """Find all behavior bouts (continuous sequences of 1s in predictions)"""
        bouts = []
        in_bout = False
        bout_start = None
        
        for i, pred in enumerate(self.predictions):
            if pred == 1 and not in_bout:
                # Start of a new bout
                in_bout = True
                bout_start = i
            elif pred == 0 and in_bout:
                # End of bout
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        # Handle bout that extends to end
        if in_bout and bout_start is not None:
            bouts.append((bout_start, len(self.predictions) - 1))
        
        return bouts
    
    def jump_to_next_bout(self):
        """Jump to the start of the next behavior bout"""
        bouts = self.find_behavior_bouts()
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected!")
            return
        
        # Find next bout after current frame
        for start, end in bouts:
            if start > self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.update_graph_window()
                return
        
        # No bout found forward, wrap to first bout
        self.current_frame = bouts[0][0]
        self.update_frame()
        self.update_graph_window()
        messagebox.showinfo("Wrapped", f"Jumped to first bout (frame {bouts[0][0]})")
    
    def jump_to_prev_bout(self):
        """Jump to the start of the previous behavior bout"""
        bouts = self.find_behavior_bouts()
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected!")
            return
        
        # Find previous bout before current frame
        for start, end in reversed(bouts):
            if start < self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.update_graph_window()
                return
        
        # No bout found backward, wrap to last bout
        self.current_frame = bouts[-1][0]
        self.update_frame()
        self.update_graph_window()
        messagebox.showinfo("Wrapped", f"Jumped to last bout (frame {bouts[-1][0]})")
    
    def update_graph_window(self):
        """Update the graph window if it exists"""
        if not self.graph_window_obj or not self.graph_window_obj.winfo_exists():
            return
        
        # During playback, only update every N frames to reduce lag
        if self.playing:
            self.graph_redraw_counter += 1
            if self.graph_redraw_counter < self.graph_redraw_interval:
                return  # Skip this update
            self.graph_redraw_counter = 0  # Reset counter
        
        try:
            # Clear
            self.graph_ax.clear()
            
            # Get window
            window_size = self.graph_window_var.get()
            half_window = window_size // 2
            start_frame = max(0, self.current_frame - half_window)
            end_frame = min(len(self.probabilities), self.current_frame + half_window)
            
            frames = np.arange(start_frame, end_frame)
            probs = self.probabilities[start_frame:end_frame]
            preds = self.predictions[start_frame:end_frame]
            
            # Plot probability
            self.graph_ax.plot(frames, probs, 'b-', linewidth=2, label='Probability', zorder=3)
            
            # Threshold
            self.graph_ax.axhline(y=self.threshold, color='g', linestyle='--', 
                                 linewidth=2, label=f'Threshold ({self.threshold:.3f})', zorder=2)
            
            # Behavior bouts as shaded regions
            in_bout = False
            bout_start = None
            for i, pred in enumerate(preds):
                if pred == 1 and not in_bout:
                    bout_start = frames[i]
                    in_bout = True
                elif pred == 0 and in_bout:
                    self.graph_ax.axvspan(bout_start, frames[i], alpha=0.3, color='red', zorder=1)
                    in_bout = False
            if in_bout:  # Close final bout
                self.graph_ax.axvspan(bout_start, frames[-1], alpha=0.3, color='red', zorder=1)
            
            # Current frame marker
            self.graph_ax.axvline(x=self.current_frame, color='orange', linewidth=3, 
                                 label='Current Frame', zorder=4)
            
            # Highlight mismatches if human labels available
            if self.human_labels is not None and len(self.human_labels) > 0:
                min_len = min(len(self.predictions), len(self.human_labels))
                
                # Find mismatches in visible window
                for i, frame_idx in enumerate(frames):
                    if frame_idx < min_len:
                        if self.predictions[frame_idx] != self.human_labels[frame_idx]:
                            # Draw orange vertical line for mismatch
                            self.graph_ax.axvline(x=frame_idx, color='darkorange', 
                                                alpha=0.4, linewidth=1, zorder=2)
                
                # Update mismatch counter label
                total_mismatches = np.sum(self.predictions[:min_len] != self.human_labels[:min_len])
                if hasattr(self, 'mismatch_label'):
                    self.mismatch_label.config(
                        text=f"Mismatches: {total_mismatches} ({total_mismatches/min_len*100:.1f}%)")
            
            # Formatting
            self.graph_ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
            self.graph_ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
            self.graph_ax.set_title(f'{self.behavior_name} - Frame {self.current_frame}/{self.total_frames}', 
                                   fontsize=12, fontweight='bold')
            self.graph_ax.set_ylim(-0.05, 1.05)
            self.graph_ax.set_xlim(start_frame, end_frame)
            self.graph_ax.grid(True, alpha=0.3, zorder=0)
            self.graph_ax.legend(loc='upper right', fontsize=10)
            
            # Stats
            n_detected = np.sum(preds)
            pct = (n_detected / len(preds) * 100) if len(preds) > 0 else 0
            stats_text = f'Window: {len(preds)} frames | Detected: {n_detected} ({pct:.1f}%)'
            
            # Add mismatch info to stats if available
            if self.human_labels is not None and len(self.human_labels) > 0:
                min_len = min(len(self.predictions), len(self.human_labels))
                window_mismatches = 0
                for frame_idx in frames:
                    if frame_idx < min_len and self.predictions[frame_idx] != self.human_labels[frame_idx]:
                        window_mismatches += 1
                stats_text += f'\nMismatches in view: {window_mismatches}'
            
            self.graph_ax.text(0.02, 0.98, stats_text, transform=self.graph_ax.transAxes,
                              verticalalignment='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            self.graph_fig.tight_layout()
            self.graph_canvas.draw()
            
            # Update frame labels and scrollbar
            if hasattr(self, 'graph_scrollbar'):
                self.graph_scrollbar.set(self.current_frame)
            if hasattr(self, 'graph_frame_label'):
                self.graph_frame_label.config(text=f"{self.current_frame}/{self.total_frames}")
            if hasattr(self, 'graph_current_frame_label'):
                self.graph_current_frame_label.config(text=f"Current Frame: {self.current_frame}/{self.total_frames}")
            
        except Exception as e:
            print(f"Error updating graph: {e}")
    
    def safe_update_graph(self):
        """Wrapper to update graph window safely"""
        self.update_graph_window()
    
    def update_graph(self):
        """Legacy method - redirect to window update"""
        self.update_graph_window()
    
class DataQualityChecker:
    """Pre-training data quality validation"""
    
    def __init__(self, parent, sessions):
        self.window = tk.Toplevel(parent)
        self.window.title("Data Quality Check")
        self.window.geometry("900x600")
        
        self.sessions = sessions
        self.issues = []
        
        self.setup_ui()
        self.run_checks()
    
    def setup_ui(self):
        """Setup quality check UI"""
        # Progress
        self.progress_label = ttk.Label(self.window, text="Checking data quality...")
        self.progress_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.window, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        # Results notebook
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill='both', expand=True)
        
        # Issues tab
        issues_frame = ttk.Frame(self.notebook)
        self.notebook.add(issues_frame, text="Issues")
        
        self.issues_text = scrolledtext.ScrolledText(issues_frame, wrap=tk.WORD)
        self.issues_text.pack(fill='both', expand=True)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Details")
        
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD)
        self.details_text.pack(fill='both', expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Export Report", 
                  command=self.export_report).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Close", 
                  command=self.window.destroy).pack(side='right', padx=5)
    
    def run_checks(self):
        """Run all quality checks"""
        total_checks = len(self.sessions) * 5
        current = 0
        
        for session in self.sessions:
            session_name = session['session_name']
            
            # Check 1: File existence
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: files...")
            self.window.update()
            
            if not os.path.exists(session['pose_path']):
                self.add_issue("ERROR", session_name, f"Pose file not found: {session['pose_path']}")
            if not os.path.exists(session['video_path']):
                self.add_issue("ERROR", session_name, f"Video file not found: {session['video_path']}")
            if session.get('target_path') and not os.path.exists(session['target_path']):
                self.add_issue("WARNING", session_name, f"Target file not found: {session['target_path']}")
            
            # Check 2: DLC file quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: DLC quality...")
            self.window.update()
            
            try:
                dlc_data = pd.read_hdf(session['pose_path'])
                n_bodyparts = len(dlc_data.columns) // 3
                self.add_detail(session_name, f"DLC body parts: {n_bodyparts}")
                
                # Check for low-confidence tracking
                prob_cols = [col for col in dlc_data.columns if 'likelihood' in str(col).lower()]
                if prob_cols:
                    for col in prob_cols:
                        low_conf = (dlc_data[col] < 0.9).sum()
                        low_conf_pct = (low_conf / len(dlc_data)) * 100
                        if low_conf_pct > 20:
                            self.add_issue("WARNING", session_name, 
                                         f"{col}: {low_conf_pct:.1f}% frames with confidence < 0.9")
            except Exception as e:
                self.add_issue("ERROR", session_name, f"Could not read DLC file: {e}")
            
            # Check 3: Video quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: video...")
            self.window.update()
            
            try:
                cap = cv2.VideoCapture(session['video_path'])
                if not cap.isOpened():
                    self.add_issue("ERROR", session_name, "Could not open video file")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    self.add_detail(session_name, 
                                   f"Video: {n_frames} frames, {fps:.1f} fps, {width}x{height}")
                    
                    # Check frame count match
                    if len(dlc_data) != n_frames:
                        self.add_issue("WARNING", session_name, 
                                     f"Frame count mismatch: DLC={len(dlc_data)}, Video={n_frames}")
                    
                    cap.release()
            except Exception as e:
                self.add_issue("ERROR", session_name, f"Video error: {e}")
            
            # Check 4: Label quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: labels...")
            self.window.update()
            
            if session.get('target_path') and os.path.exists(session['target_path']):
                try:
                    labels = pd.read_csv(session['target_path'])
                    
                    # Check label distribution
                    for col in labels.columns:
                        if col.lower() != 'frame':
                            positive = labels[col].sum()
                            total = len(labels)
                            positive_pct = (positive / total) * 100
                            
                            self.add_detail(session_name, 
                                          f"Behavior '{col}': {positive} frames ({positive_pct:.1f}%)")
                            
                            if positive_pct < 1:
                                self.add_issue("WARNING", session_name, 
                                             f"Very rare behavior '{col}': only {positive_pct:.2f}%")
                            elif positive_pct > 50:
                                self.add_issue("WARNING", session_name, 
                                             f"Very common behavior '{col}': {positive_pct:.1f}%")
                    
                    # Check for label inconsistencies
                    if len(labels) != len(dlc_data):
                        self.add_issue("WARNING", session_name, 
                                     f"Label count mismatch: Labels={len(labels)}, DLC={len(dlc_data)}")
                    
                    # Bout outlier detection
                    for col in labels.columns:
                        if col.lower() == 'frame':
                            continue
                        vals = labels[col].values
                        bouts, gaps = [], []
                        in_bout, bout_start = False, 0
                        for i, v in enumerate(vals):
                            if v == 1 and not in_bout:
                                bout_start = i
                                in_bout = True
                            elif v == 0 and in_bout:
                                bouts.append(i - bout_start)
                                in_bout = False
                        if in_bout:
                            bouts.append(len(vals) - bout_start)
                        for i in range(len(bouts) - 1):
                            # gap = frames between bouts (rough)
                            pass  # gap calc needs start/end pairs; skip for now
                        
                        if bouts:
                            single_frame = sum(1 for b in bouts if b <= 2)
                            if single_frame > 0:
                                pct = 100 * single_frame / len(bouts)
                                self.add_issue(
                                    "WARNING", session_name,
                                    f"'{col}': {single_frame} bout(s) ≤2 frames "
                                    f"({pct:.0f}% of bouts) — possible accidental labels")
                            
                            # Try to get FPS for duration check
                            try:
                                cap = cv2.VideoCapture(session['video_path'])
                                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                                cap.release()
                            except Exception:
                                fps = 30
                            long_thresh_frames = fps * 30  # 30 seconds
                            long_bouts = sum(1 for b in bouts if b > long_thresh_frames)
                            if long_bouts > 0:
                                self.add_issue(
                                    "WARNING", session_name,
                                    f"'{col}': {long_bouts} bout(s) >30 s "
                                    f"— possible missed label-off click")
                
                except Exception as e:
                    self.add_issue("ERROR", session_name, f"Could not read labels: {e}")
            
            # Check 5: Duplicate detection
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: duplicates...")
            self.window.update()
            
            # Simple duplicate check based on filename
            for other in self.sessions:
                if other != session and other['session_name'] == session_name:
                    self.add_issue("ERROR", session_name, "Duplicate session name found!")
        
        self.progress_label.config(text="Quality check complete!")
        self.display_summary()
    
    def add_issue(self, level, session, message):
        """Add an issue to the list"""
        self.issues.append({
            'level': level,
            'session': session,
            'message': message
        })
        
        color_tag = 'error' if level == 'ERROR' else 'warning'
        self.issues_text.insert(tk.END, f"[{level}] {session}: {message}\n", color_tag)
    
    def add_detail(self, session, message):
        """Add detail information"""
        self.details_text.insert(tk.END, f"{session}: {message}\n")
    
    def display_summary(self):
        """Display summary of checks"""
        n_sessions = len(self.sessions)
        n_errors = len([i for i in self.issues if i['level'] == 'ERROR'])
        n_warnings = len([i for i in self.issues if i['level'] == 'WARNING'])
        
        summary = "=== Data Quality Check Summary ===\n\n"
        summary += f"Sessions checked: {n_sessions}\n"
        summary += f"Errors found: {n_errors}\n"
        summary += f"Warnings found: {n_warnings}\n\n"
        
        if n_errors == 0 and n_warnings == 0:
            summary += "✓ All checks passed! Data quality looks good.\n"
        elif n_errors == 0:
            summary += "✓ No critical errors found.\n"
            summary += f"⚠ {n_warnings} warning(s) - review recommended but not critical.\n"
        else:
            summary += f"✗ {n_errors} error(s) found - must be fixed before training!\n"
            summary += f"⚠ {n_warnings} warning(s) also found.\n"
        
        summary += "\n=== Recommendations ===\n\n"
        
        if n_errors > 0:
            summary += "1. Fix all ERROR-level issues before proceeding\n"
        if n_warnings > 0:
            summary += "2. Review WARNING-level issues - they may affect performance\n"
        
        summary += "3. Ensure all videos have matching DLC files\n"
        summary += "4. Check that behavior labels are consistent\n"
        summary += "5. Verify tracking quality is good (>80% high confidence)\n"
        
        self.summary_text.insert('1.0', summary)
        
        # Configure tags for colored text
        self.issues_text.tag_config('error', foreground='red')
        self.issues_text.tag_config('warning', foreground='orange')
    
    def export_report(self):
        """Export quality report"""
        output_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(self.summary_text.get('1.0', tk.END))
                f.write("\n\n=== ISSUES ===\n\n")
                f.write(self.issues_text.get('1.0', tk.END))
                f.write("\n\n=== DETAILS ===\n\n")
                f.write(self.details_text.get('1.0', tk.END))
            
            messagebox.showinfo("Exported", f"Report saved to:\n{output_path}")


class EthogramGenerator:
    """Generate behavior ethograms and statistics"""
    
    @staticmethod
    def generate_ethogram(predictions_dict, fps, output_folder):
        """
        Generate comprehensive ethogram analysis
        
        Args:
            predictions_dict: {behavior_name: predictions_array}
            fps: frames per second
            output_folder: where to save outputs
        """
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        for behavior, preds in predictions_dict.items():
            preds = np.array(preds).flatten()
            
            # Basic statistics
            total_frames = len(preds)
            behavior_frames = np.sum(preds)
            behavior_pct = (behavior_frames / total_frames) * 100
            behavior_time = behavior_frames / fps
            
            # Find bouts
            bouts = []
            in_bout = False
            bout_start = 0
            
            for i, val in enumerate(preds):
                if val == 1 and not in_bout:
                    bout_start = i
                    in_bout = True
                elif val == 0 and in_bout:
                    bouts.append({
                        'start_frame': bout_start,
                        'end_frame': i - 1,
                        'duration_frames': i - bout_start,
                        'duration_sec': (i - bout_start) / fps
                    })
                    in_bout = False
            
            if in_bout:
                bouts.append({
                    'start_frame': bout_start,
                    'end_frame': len(preds) - 1,
                    'duration_frames': len(preds) - bout_start,
                    'duration_sec': (len(preds) - bout_start) / fps
                })
            
            # Bout statistics
            if bouts:
                bout_durations = [b['duration_sec'] for b in bouts]
                mean_bout = np.mean(bout_durations)
                std_bout = np.std(bout_durations)
                min_bout = np.min(bout_durations)
                max_bout = np.max(bout_durations)
                
                # Inter-bout intervals
                if len(bouts) > 1:
                    intervals = []
                    for i in range(len(bouts) - 1):
                        interval = (bouts[i+1]['start_frame'] - bouts[i]['end_frame']) / fps
                        intervals.append(interval)
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                else:
                    mean_interval = np.nan
                    std_interval = np.nan
            else:
                mean_bout = std_bout = min_bout = max_bout = 0
                mean_interval = std_interval = np.nan
            
            results[behavior] = {
                'total_time_sec': behavior_time,
                'total_time_min': behavior_time / 60,
                'percentage': behavior_pct,
                'n_bouts': len(bouts),
                'mean_bout_duration': mean_bout,
                'std_bout_duration': std_bout,
                'min_bout_duration': min_bout,
                'max_bout_duration': max_bout,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'bouts': bouts
            }
        
        # Generate plots
        EthogramGenerator._plot_time_budget(results, output_folder)
        EthogramGenerator._plot_bout_distributions(results, output_folder)
        EthogramGenerator._plot_raster(predictions_dict, fps, output_folder)
        
        # Generate summary report
        EthogramGenerator._write_summary(results, output_folder)
        
        return results
    
    @staticmethod
    def _plot_time_budget(results, output_folder):
        """Plot time budget pie chart"""
        if plt is None:
            return
        
        behaviors = list(results.keys())
        times = [results[b]['total_time_min'] for b in behaviors]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(times, labels=behaviors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Time Budget')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'time_budget.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _plot_bout_distributions(results, output_folder):
        """Plot bout duration distributions"""
        if plt is None:
            return
        
        n_behaviors = len(results)
        fig, axes = plt.subplots(n_behaviors, 1, figsize=(10, 3*n_behaviors))
        
        if n_behaviors == 1:
            axes = [axes]
        
        for ax, (behavior, data) in zip(axes, results.items()):
            if data['bouts']:
                durations = [b['duration_sec'] for b in data['bouts']]
                ax.hist(durations, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(data['mean_bout_duration'], color='red', 
                          linestyle='--', label=f"Mean: {data['mean_bout_duration']:.2f}s")
                ax.set_xlabel('Bout Duration (s)')
                ax.set_ylabel('Count')
                ax.set_title(f'{behavior} - Bout Durations')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'bout_distributions.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _plot_raster(predictions_dict, fps, output_folder):
        """Plot behavior raster plot"""
        if plt is None:
            return
        
        behaviors = list(predictions_dict.keys())
        n_behaviors = len(behaviors)
        
        fig, ax = plt.subplots(figsize=(12, 2*n_behaviors))
        
        for i, behavior in enumerate(behaviors):
            preds = np.array(predictions_dict[behavior]).flatten()
            
            # Find behavior events
            events = np.where(preds == 1)[0] / fps / 60  # Convert to minutes
            
            ax.scatter(events, [i] * len(events), marker='|', s=100, alpha=0.5)
        
        ax.set_yticks(range(n_behaviors))
        ax.set_yticklabels(behaviors)
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Behavior Raster Plot')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'behavior_raster.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _write_summary(results, output_folder):
        """Write text summary"""
        summary_path = os.path.join(output_folder, 'ethogram_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=== BEHAVIOR ETHOGRAM SUMMARY ===\n\n")
            
            for behavior, data in results.items():
                f.write(f"\n{behavior.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total time: {data['total_time_min']:.2f} minutes ({data['percentage']:.1f}%)\n")
                f.write(f"Number of bouts: {data['n_bouts']}\n")
                
                if data['n_bouts'] > 0:
                    f.write(f"Mean bout duration: {data['mean_bout_duration']:.2f} ± {data['std_bout_duration']:.2f} s\n")
                    f.write(f"Bout duration range: {data['min_bout_duration']:.2f} - {data['max_bout_duration']:.2f} s\n")
                    
                    if not np.isnan(data['mean_interval']):
                        f.write(f"Mean inter-bout interval: {data['mean_interval']:.2f} ± {data['std_interval']:.2f} s\n")
                
                f.write("\n")


class PixelPawsGUI:
    """Complete integrated application class with all enhanced features"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PixelPaws - Behavioral Analysis & Recognition")
        self.root.geometry("1100x750")
        
        # Set application icon
        self.set_app_icon()
        
        # Theme
        self.theme = Theme('light')

        # Application state
        self.project_folder = tk.StringVar()
        self.classifier_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")

        # ── Shared project folder ─────────────────────────────────────────────
        # A single StringVar written by any "browse project" action; observed by
        # training, batch, and evaluation tabs so users don't re-enter it.
        self.current_project_folder = tk.StringVar()
        self.current_project_folder.trace_add('write', self._on_project_folder_changed)
        
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
        self.train_use_gpu = None
        self.train_generate_plots = None
        self.train_log = None
        
        # Initialize batch variables (will be created in create_batch_tab)
        self.batch_folder = None
        self.batch_video_ext = None
        self.batch_prefer_filtered = None
        self.batch_clf_listbox = None
        self.batch_classifiers = {}
        self.batch_save_labels = None
        self.batch_save_timebins = None
        self.batch_bin_size = None
        self.batch_generate_ethograms = None
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
        self.lv_skeleton_dots         = tk.BooleanVar(value=True)
        self.lv_frame_tint            = tk.BooleanVar(value=False)
        self.lv_timeline_strip        = tk.BooleanVar(value=True)
        
        # Setup UI
        self.setup_ui()
        self.apply_theme()

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
        
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_training_tab()
        self.create_active_learning_tab()  # Active Learning tab - right after training
        self.create_evaluation_tab()
        self.create_prediction_tab()
        self.create_batch_tab()
        
        # Analysis tab (for batch analysis and graphing)
        if ANALYSIS_TAB_AVAILABLE:
            self.analysis_tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.analysis_tab_frame, text="📊 Analysis")
            self.analysis_tab = AnalysisTab(self.analysis_tab_frame, self)
            self.analysis_tab.pack(fill='both', expand=True)
        
        self.create_tools_tab()  # New tab for enhanced tools
        
        # Status bar at bottom
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=5, pady=5)
        
        ttk.Label(status_frame, textvariable=self.status_text, 
                 relief='sunken', anchor='w').pack(fill='x')
        
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_theme)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Quality Checker", command=self.open_quality_checker)
        tools_menu.add_command(label="Auto-Label Assistant", command=self.open_auto_labeler)
        tools_menu.add_command(label="Video Preview", command=self.open_video_preview)
        tools_menu.add_separator()
        tools_menu.add_command(label="Generate Ethogram", command=self.generate_ethogram)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
    
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
        setup_frame.pack(fill='x', padx=5, pady=5)
        
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
        behavior_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(behavior_frame, text="Behavior Name:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_behavior_name = tk.StringVar(value="Scratching")
        ttk.Entry(behavior_frame, textvariable=self.train_behavior_name, width=30).grid(
            row=0, column=1, padx=5, pady=2, sticky='w')
        
        # Auto-detect button
        ttk.Button(behavior_frame, text="🔍 Auto-Detect", 
                  command=self.auto_detect_behavior_names, width=12).grid(
            row=0, column=2, padx=5, pady=2, sticky='w')
        
        tk.Label(behavior_frame, text="(Must match CSV column name)", 
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=0, column=3, sticky='w')
        
        # Filtering parameters
        ttk.Label(behavior_frame, text="Min Bout (frames):").grid(row=2, column=0, sticky='w', pady=2)
        self.train_min_bout = tk.IntVar(value=3)
        tk.Spinbox(behavior_frame, from_=1, to=100, textvariable=self.train_min_bout, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        tk.Label(behavior_frame, text="Minimum consecutive frames for valid bout", 
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=2, column=3, sticky='w')
        
        ttk.Label(behavior_frame, text="Min After Bout (frames):").grid(row=3, column=0, sticky='w', pady=2)
        self.train_min_after_bout = tk.IntVar(value=1)
        tk.Spinbox(behavior_frame, from_=1, to=100, textvariable=self.train_min_after_bout, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        tk.Label(behavior_frame, text="Minimum frames after bout ends", 
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=3, column=3, sticky='w')
        
        ttk.Label(behavior_frame, text="Max Gap (frames):").grid(row=4, column=0, sticky='w', pady=2)
        self.train_max_gap = tk.IntVar(value=5)
        tk.Spinbox(behavior_frame, from_=0, to=100, textvariable=self.train_max_gap, width=10).grid(
            row=4, column=1, sticky='w', padx=5, pady=2)
        tk.Label(behavior_frame, text="Maximum frames to bridge between bouts", 
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=4, column=3, sticky='w')
        
        # Auto-suggest button
        ttk.Button(behavior_frame, text="🤖 Auto-Suggest Bout Parameters", 
                  command=self.auto_suggest_bout_params).grid(row=5, column=0, columnspan=4, pady=10)
        
        # === FEATURE CONFIGURATION ===
        feature_frame = ttk.LabelFrame(scrollable_frame, text="Feature Configuration", padding=10)
        feature_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(feature_frame, text="Pixel Brightness Body Parts:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_bp_pixbrt = tk.StringVar(value="hrpaw,hlpaw,snout")
        ttk.Entry(feature_frame, textvariable=self.train_bp_pixbrt, width=30).grid(
            row=0, column=1, padx=5, pady=2, sticky='w')
        tk.Label(feature_frame, text="Body parts for brightness analysis (comma-separated)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=0, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="Square Sizes:").grid(row=1, column=0, sticky='w', pady=2)
        self.train_square_sizes = tk.StringVar(value="40,40,40")
        ttk.Entry(feature_frame, textvariable=self.train_square_sizes, width=30).grid(
            row=1, column=1, padx=5, pady=2, sticky='w')
        tk.Label(feature_frame, text="Window size for each body part (pixels)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=1, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="Pixel Threshold:").grid(row=2, column=0, sticky='w', pady=2)
        self.train_pix_threshold = tk.DoubleVar(value=0.3)
        ttk.Entry(feature_frame, textvariable=self.train_pix_threshold, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        tk.Label(feature_frame, text="Brightness cutoff: <1 = fraction, ≥1 = raw value", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=2, column=2, sticky='w')
        
        ttk.Label(feature_frame, text="DLC Config (optional):").grid(row=3, column=0, sticky='w', pady=2)
        self.train_dlc_config = tk.StringVar()
        ttk.Entry(feature_frame, textvariable=self.train_dlc_config, width=30).grid(
            row=3, column=1, padx=5, pady=2, sticky='w')
        ttk.Button(feature_frame, text="📁 Browse",
                  command=self.browse_train_dlc_config).grid(row=3, column=2, sticky='w', padx=2)
        tk.Label(feature_frame, text="For DLC crop offset in brightness extraction",
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=4, column=1, columnspan=2, sticky='w')

        # Optical Flow Features
        self.train_include_optical_flow = tk.BooleanVar(value=False)
        ttk.Checkbutton(feature_frame,
                        text="Include Optical Flow Features  (slower — reads video frames)",
                        variable=self.train_include_optical_flow).grid(
            row=5, column=1, columnspan=2, sticky='w', pady=2)
        ttk.Label(feature_frame, text="Optical Flow Body Parts:").grid(row=6, column=0, sticky='w', pady=2)
        self.train_bp_optflow = tk.StringVar(value="hrpaw,hlpaw,snout")
        ttk.Entry(feature_frame, textvariable=self.train_bp_optflow, width=30).grid(
            row=6, column=1, padx=5, pady=2, sticky='w')
        tk.Label(feature_frame, text="Body parts for optical flow (comma-separated)",
                 fg='gray', bg=self.theme.colors['frame_bg']).grid(row=6, column=2, sticky='w')
        
        # === XGBOOST PARAMETERS ===
        xgb_frame = ttk.LabelFrame(scrollable_frame, text="XGBoost Model Parameters", padding=10)
        xgb_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(xgb_frame, text="Number of Trees:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_n_estimators = tk.IntVar(value=1700)
        tk.Spinbox(xgb_frame, from_=100, to=5000, increment=100, 
                   textvariable=self.train_n_estimators, width=10).grid(
            row=0, column=1, sticky='w', padx=5, pady=2)
        tk.Label(xgb_frame, text="More trees = better fit but slower (1000-2000 typical)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=0, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Max Tree Depth:").grid(row=1, column=0, sticky='w', pady=2)
        self.train_max_depth = tk.IntVar(value=6)
        tk.Spinbox(xgb_frame, from_=3, to=15, textvariable=self.train_max_depth, width=10).grid(
            row=1, column=1, sticky='w', padx=5, pady=2)
        tk.Label(xgb_frame, text="Tree complexity: 4-8 typical, higher = risk overfitting", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=1, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Learning Rate:").grid(row=2, column=0, sticky='w', pady=2)
        self.train_learning_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(xgb_frame, textvariable=self.train_learning_rate, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        tk.Label(xgb_frame, text="Step size for updates: 0.01-0.1 typical, lower = more stable", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=2, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Subsample Ratio:").grid(row=3, column=0, sticky='w', pady=2)
        self.train_subsample = tk.DoubleVar(value=0.8)
        ttk.Entry(xgb_frame, textvariable=self.train_subsample, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        tk.Label(xgb_frame, text="Fraction of data per tree: 0.5-0.9, prevents overfitting", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=3, column=2, sticky='w')
        
        ttk.Label(xgb_frame, text="Feature Sampling:").grid(row=4, column=0, sticky='w', pady=2)
        self.train_colsample = tk.DoubleVar(value=0.2)
        ttk.Entry(xgb_frame, textvariable=self.train_colsample, width=10).grid(
            row=4, column=1, sticky='w', padx=5, pady=2)
        tk.Label(xgb_frame, text="Fraction of features per tree: 0.2-0.5, adds diversity", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=4, column=2, sticky='w')
        
        # === TRAINING PARAMETERS ===
        params_frame = ttk.LabelFrame(scrollable_frame, text="Training Parameters", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(params_frame, text="K-Fold Cross-Validation:").grid(row=0, column=0, sticky='w', pady=2)
        self.train_n_folds = tk.IntVar(value=5)
        tk.Spinbox(params_frame, from_=2, to=10, textvariable=self.train_n_folds, width=10).grid(
            row=0, column=1, sticky='w', padx=5, pady=2)
        tk.Label(params_frame, text="Number of validation folds: 5 typical, 10 for smaller datasets", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=0, column=2, sticky='w')
        
        # scale_pos_weight (replaces downsampling as the default)
        self.train_use_scale_pos_weight = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use scale_pos_weight for class imbalance",
                       variable=self.train_use_scale_pos_weight).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Recommended: weights positives without discarding any frames", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=1, column=2, sticky='w')

        # Fallback downsampling (kept for compatibility)
        self.train_use_balancing = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Also apply downsampling (legacy fallback)",
                       variable=self.train_use_balancing).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Downsamples negatives — use only if scale_pos_weight is off", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=2, column=2, sticky='w')

        ttk.Label(params_frame, text="Imbalance Threshold:").grid(row=3, column=0, sticky='w', pady=2)
        self.train_imbalance_thresh = tk.DoubleVar(value=0.05)
        ttk.Entry(params_frame, textvariable=self.train_imbalance_thresh, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=2)
        tk.Label(params_frame, text="Apply downsampling only if positive ratio < this value", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=3, column=2, sticky='w')

        # Early stopping
        self.train_use_early_stopping = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use early stopping (auto-selects n_estimators)",
                       variable=self.train_use_early_stopping).grid(row=4, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Stops adding trees when val F1 plateaus — prevents overfitting", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=4, column=2, sticky='w')

        ttk.Label(params_frame, text="Early Stopping Rounds:").grid(row=5, column=0, sticky='w', pady=2)
        self.train_early_stopping_rounds = tk.IntVar(value=50)
        tk.Spinbox(params_frame, from_=10, to=200, increment=10,
                   textvariable=self.train_early_stopping_rounds, width=10).grid(
            row=5, column=1, sticky='w', padx=5, pady=2)
        tk.Label(params_frame, text="Stop if no improvement for this many trees (50 typical)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=5, column=2, sticky='w')

        # Use GPU
        self.train_use_gpu = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use GPU acceleration (if available)", 
                       variable=self.train_use_gpu).grid(row=6, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Much faster training with CUDA-compatible GPU", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=6, column=2, sticky='w')
        
        # Generate plots
        self.train_generate_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Generate performance and SHAP plots",
                       variable=self.train_generate_plots).grid(row=7, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Creates threshold curve and feature importance figures", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=7, column=2, sticky='w')

        # SHAP prune + retrain
        self.train_shap_prune = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="SHAP prune + retrain (2nd pass with top features only)",
                       variable=self.train_shap_prune).grid(row=8, column=1, sticky='w', pady=2)
        tk.Label(params_frame, text="Train on all features first, then retrain on top SHAP features — reduces noise", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=8, column=2, sticky='w')

        ttk.Label(params_frame, text="Top Features (SHAP):").grid(row=9, column=0, sticky='w', pady=2)
        self.train_shap_top_n = tk.IntVar(value=40)
        tk.Spinbox(params_frame, from_=10, to=200, increment=5,
                   textvariable=self.train_shap_top_n, width=10).grid(
            row=9, column=1, sticky='w', padx=5, pady=2)
        tk.Label(params_frame, text="Number of features to keep after SHAP pruning (10–200)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=9, column=2, sticky='w')

        # === QUICK ACTIONS ===
        action_frame = ttk.LabelFrame(scrollable_frame, text="Actions", padding=10)
        action_frame.pack(fill='x', padx=5, pady=5)
        
        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="🔍 Check Data Quality", 
                  command=self.open_quality_checker, width=20).pack(side='left', padx=5, pady=5)
        ttk.Button(btn_frame, text="📋 Scan Sessions", 
                  command=self.scan_training_sessions, width=20).pack(side='left', padx=5, pady=5)
        ttk.Button(btn_frame, text="💾 Save Configuration", 
                  command=self.save_training_config, width=20).pack(side='left', padx=5, pady=5)
        ttk.Button(btn_frame, text="📂 Load Configuration", 
                  command=self.load_training_config, width=20).pack(side='left', padx=5, pady=5)
        
        # Start button (larger, prominent)
        start_frame = ttk.Frame(action_frame)
        start_frame.pack(fill='x', pady=10)
        ttk.Button(start_frame, text="▶ START TRAINING", 
                  command=self.start_training, 
                  style='Accent.TButton').pack(side='left', padx=5)
        
        # === TRAINING LOG ===
        log_frame = ttk.LabelFrame(scrollable_frame, text="Training Log", padding=5)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.train_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.train_log.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
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
        clf_frame.pack(fill='x', padx=5, pady=5)
        
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
        
        # === VIDEO SELECTION ===
        video_frame = ttk.LabelFrame(scrollable_frame, text="Video Files", padding=10)
        video_frame.pack(fill='x', padx=5, pady=5)
        
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
        output_frame.pack(fill='x', padx=5, pady=5)
        
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
        tk.Label(output_frame, text="(Leave empty to use video folder)", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=6, column=1, sticky='w')
        
        # === ACTIONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(action_frame, text="🎥 Preview Video", 
                  command=self.preview_pred_video).pack(side='left', padx=5)
        ttk.Button(action_frame, text="🔍 Preview with Predictions", 
                  command=self.preview_with_predictions).pack(side='left', padx=5)
        ttk.Button(action_frame, text="🎯 Optimize Parameters", 
                  command=self.optimize_parameters).pack(side='left', padx=5)
        ttk.Button(action_frame, text="▶ RUN PREDICTION",
                  command=self.run_single_prediction,
                  style='Accent.TButton').pack(side='left', padx=5)
        self.pred_export_video_btn = ttk.Button(
            action_frame, text="🎬 Export Labeled Video",
            command=self.export_labeled_video, state='disabled')
        self.pred_export_video_btn.pack(side='left', padx=5)

        # === EXPORT VIDEO OVERLAYS ===
        overlay_opts = ttk.LabelFrame(scrollable_frame, text="Export Video Overlays", padding=4)
        overlay_opts.pack(fill='x', padx=5, pady=(0, 4))
        ttk.Checkbutton(overlay_opts, text="Skeleton dots (DLC body parts)",
                        variable=self.lv_skeleton_dots).pack(side='left', padx=8)
        ttk.Checkbutton(overlay_opts, text="Behavior frame tint",
                        variable=self.lv_frame_tint).pack(side='left', padx=8)
        ttk.Checkbutton(overlay_opts, text="Timeline strip",
                        variable=self.lv_timeline_strip).pack(side='left', padx=8)

        # === RESULTS DISPLAY ===
        results_frame = ttk.LabelFrame(scrollable_frame, text="Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.pred_results_text = scrolledtext.ScrolledText(results_frame, height=12, wrap=tk.WORD)
        self.pred_results_text.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
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
        input_frame.pack(fill='x', padx=5, pady=5)
        
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
        output_frame.pack(fill='x', padx=5, pady=5)
        
        self.batch_save_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Save frame-by-frame labels for each video", 
                       variable=self.batch_save_labels).grid(row=0, column=0, sticky='w', pady=2)
        
        self.batch_save_timebins = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Generate time-binned summaries", 
                       variable=self.batch_save_timebins).grid(row=1, column=0, sticky='w', pady=2)
        
        ttk.Label(output_frame, text="Time Bin Size (seconds):").grid(row=2, column=0, sticky='w', pady=2)
        self.batch_bin_size = tk.DoubleVar(value=60.0)
        ttk.Entry(output_frame, textvariable=self.batch_bin_size, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)
        tk.Label(output_frame, text="For aggregating behavior across time windows", fg='gray', bg=self.theme.colors['frame_bg']).grid(row=2, column=2, sticky='w')
        
        self.batch_generate_ethograms = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Generate ethogram plots for each video", 
                       variable=self.batch_generate_ethograms).grid(row=3, column=0, columnspan=2, sticky='w', pady=2)
        
        # === ACTIONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(action_frame, text="🔍 Check Feature Status", 
                  command=self.check_batch_features, 
                  ).pack(side='left', padx=5)
        
        ttk.Button(action_frame, text="▶ RUN BATCH ANALYSIS", 
                  command=self.run_batch_analysis, 
                  style='Accent.TButton').pack(side='left', padx=5)
        
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
    
    def create_tools_tab(self):
        """Create tools tab with enhanced features"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="🛠 Tools")
        
        # Title
        title = ttk.Label(tools_frame, text="Enhanced Analysis Tools", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Tool buttons grid
        grid_frame = ttk.Frame(tools_frame)
        grid_frame.pack(expand=True)
        
        tools = [
            ("🎥 Video Preview\nwith Predictions", self.open_video_preview),
            ("🤖 Auto-Label\nAssistant", self.open_auto_labeler),
            ("🔍 Data Quality\nChecker", self.open_quality_checker),
            ("💡 Brightness\nDiagnostics", self.run_brightness_diagnostics),
            ("📋 Feature File\nInspector", self.inspect_features_file),
            ("🌟 Brightness\nPreview", self.show_brightness_preview),
            ("🔧 Correct Crop\nOffset (Single)", self.correct_crop_offset_single),
            ("🔧 Correct Crop\nOffset (Batch)", self.correct_crop_offset_batch),
            ("✂️ Crop Video\nfor DLC", self.crop_video_for_dlc),
            ("📊 Generate\nEthogram", self.generate_ethogram),
            ("📈 Training\nVisualization", self.show_training_viz),
            ("🔄 BORIS to\nPixelPaws", self.convert_boris_to_pixelpaws),
            ("🎯 Optimize\nParameters", self.optimize_parameters),
            ("⚙️ Feature\nExtraction", self.open_feature_extraction),
            ("🎨 Theme\nSwitcher", self.toggle_theme),
        ]
        
        for i, (text, command) in enumerate(tools):
            row = i // 3
            col = i % 3
            btn = ttk.Button(grid_frame, text=text, command=command, width=20)
            btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
    
    
    def create_active_learning_tab(self):
        """Create Active Learning tab"""
        al_frame = ttk.Frame(self.notebook)
        self.notebook.add(al_frame, text="🧠 Active Learning")
        
        # Check if module is available
        if not ACTIVE_LEARNING_AVAILABLE:
            error_frame = ttk.Frame(al_frame)
            error_frame.pack(expand=True, fill='both', padx=20, pady=20)
            
            ttk.Label(
                error_frame,
                text="⚠️ Active Learning Module Not Available",
                font=('Arial', 16, 'bold'),
                foreground='red'
            ).pack(pady=20)
            
            ttk.Label(
                error_frame,
                text="Please ensure active_learning.py is in the same directory as PixelPaws_GUI.py",
                font=('Arial', 12)
            ).pack(pady=10)
            
            return
        
        # Main container with scrollbar
        canvas = tk.Canvas(al_frame)
        scrollbar = ttk.Scrollbar(al_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title and description
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(
            title_frame,
            text="🧠 Active Learning",
            font=('Arial', 18, 'bold')
        ).pack()
        
        ttk.Label(
            title_frame,
            text="Intelligently suggest which frames to label for maximum classifier improvement",
            font=('Arial', 11),
            foreground='gray'
        ).pack(pady=5)
        
        # Info box
        info_frame = ttk.LabelFrame(scrollable_frame, text="How It Works", padding=15)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = (
            "Active Learning reduces labeling time by 50-70% by:\n\n"
            "1. Using your trained classifier to score every frame across ALL videos\n"
            "2. Selecting the most borderline frames globally (not per-video)\n"
            "3. Showing you those frames to label, grouped by video\n"
            "4. Automatically updating each video's labels CSV\n"
            "5. Retraining with the improved labels\n\n"
            "The global approach means your budget goes to wherever the model\n"
            "is actually struggling — not split evenly across videos regardless of difficulty."
        )
        
        ttk.Label(
            info_frame,
            text=info_text,
            justify='left',
            wraplength=700
        ).pack()
        
        # File selection section
        files_frame = ttk.LabelFrame(scrollable_frame, text="📁 Required Files", padding=15)
        files_frame.pack(fill='x', padx=20, pady=10)
        
        # Quick start button
        quick_start_frame = ttk.Frame(files_frame)
        quick_start_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(
            quick_start_frame,
            text="⚡ Quick Start - Auto-fill from Labels CSV",
            command=self.al_quick_start
        ).pack(side='left')
        
        ttk.Label(
            quick_start_frame,
            text="← Select labels CSV, then click to auto-find other files",
            font=('Arial', 9),
            foreground='gray'
        ).pack(side='left', padx=10)
        
        ttk.Separator(files_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Batch mode section
        batch_header = ttk.Frame(files_frame)
        batch_header.pack(fill='x', pady=5)
        
        ttk.Label(
            batch_header,
            text="OR - Batch Mode (Multiple Sessions):",
            font=('Arial', 11, 'bold')
        ).pack(side='left')
        
        ttk.Button(
            batch_header,
            text="🗂️ Select Project Folder",
            command=self.al_batch_mode
        ).pack(side='left', padx=10)
        
        ttk.Label(
            files_frame,
            text="Process multiple videos from a project folder (same as Classifier Training tab)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(pady=2)
        
        ttk.Separator(files_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Labels CSV
        csv_frame = ttk.Frame(files_frame)
        csv_frame.pack(fill='x', pady=5)
        
        ttk.Label(csv_frame, text="Per-Frame Labels CSV:", width=20).pack(side='left')
        self.al_labels_csv = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.al_labels_csv, width=50).pack(side='left', padx=5)
        ttk.Button(csv_frame, text="Browse", command=self.al_browse_labels).pack(side='left')
        
        # Video
        video_frame = ttk.Frame(files_frame)
        video_frame.pack(fill='x', pady=5)
        
        ttk.Label(video_frame, text="Video File:", width=20).pack(side='left')
        self.al_video_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.al_video_path, width=50).pack(side='left', padx=5)
        ttk.Button(video_frame, text="Browse", command=self.al_browse_video).pack(side='left')
        
        # DLC file
        dlc_frame = ttk.Frame(files_frame)
        dlc_frame.pack(fill='x', pady=5)
        
        ttk.Label(dlc_frame, text="DLC File (.h5):", width=20).pack(side='left')
        self.al_dlc_path = tk.StringVar()
        ttk.Entry(dlc_frame, textvariable=self.al_dlc_path, width=50).pack(side='left', padx=5)
        ttk.Button(dlc_frame, text="Browse", command=self.al_browse_dlc).pack(side='left')
        
        # Features cache
        cache_frame = ttk.Frame(files_frame)
        cache_frame.pack(fill='x', pady=5)
        
        ttk.Label(cache_frame, text="Features Cache (.pkl):", width=20).pack(side='left')
        self.al_features_cache = tk.StringVar()
        ttk.Entry(cache_frame, textvariable=self.al_features_cache, width=50).pack(side='left', padx=5)
        ttk.Button(cache_frame, text="Browse", command=self.al_browse_cache).pack(side='left')
        
        ttk.Label(
            files_frame,
            text="Tip: Features cache is auto-generated when you train or predict",
            font=('Arial', 9),
            foreground='gray'
        ).pack(pady=5)
        
        # Settings section
        settings_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Settings", padding=15)
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        # Target region selection
        target_frame = ttk.LabelFrame(settings_frame, text="Target Region", padding=10)
        target_frame.pack(fill='x', pady=5)
        
        self.al_target_mode = tk.StringVar(value='unlabeled')
        
        ttk.Radiobutton(
            target_frame,
            text="Suggest from labeled regions only (refine existing labels)",
            variable=self.al_target_mode,
            value='labeled'
        ).pack(anchor='w', pady=2)
        
        ttk.Radiobutton(
            target_frame,
            text="Extend to unlabeled regions (label new parts of video) ⭐ Recommended",
            variable=self.al_target_mode,
            value='unlabeled'
        ).pack(anchor='w', pady=2)
        
        ttk.Label(
            target_frame,
            text="Tip: Use 'unlabeled' to progressively label long videos with minimal effort!",
            font=('Arial', 9),
            foreground='gray'
        ).pack(pady=5)
        
        ttk.Separator(settings_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Number of suggestions
        suggest_frame = ttk.Frame(settings_frame)
        suggest_frame.pack(fill='x', pady=5)
        
        ttk.Label(suggest_frame, text="Frames to suggest:", width=20).pack(side='left')
        self.al_n_suggestions = tk.IntVar(value=100)
        ttk.Spinbox(
            suggest_frame,
            from_=10,
            to=500,
            textvariable=self.al_n_suggestions,
            width=10
        ).pack(side='left', padx=5)
        ttk.Label(suggest_frame, text="(total across all videos — 100 is a good starting point)", foreground='gray').pack(side='left')
        
        # Number of iterations
        iter_frame = ttk.Frame(settings_frame)
        iter_frame.pack(fill='x', pady=5)
        
        ttk.Label(iter_frame, text="Max iterations:", width=20).pack(side='left')
        self.al_n_iterations = tk.IntVar(value=1)
        ttk.Spinbox(
            iter_frame,
            from_=1,
            to=10,
            textvariable=self.al_n_iterations,
            width=10
        ).pack(side='left', padx=5)
        ttk.Label(iter_frame, text="(1 = manual, 3-5 = auto-repeat)", foreground='gray').pack(side='left')
        
        # Model path (optional)
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill='x', pady=5)
        
        ttk.Label(model_frame, text="Model (optional):", width=20).pack(side='left')
        self.al_model_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.al_model_path, width=50).pack(side='left', padx=5)
        ttk.Button(model_frame, text="Browse", command=self.al_browse_model).pack(side='left')
        
        ttk.Label(
            settings_frame,
            text="Leave model blank to train a new one automatically.\n"
                 "Note: Active learning retrains on your CURRENT labels each iteration,\n"
                 "so it adapts as you add new labels. The initial model is just a starting point.",
            font=('Arial', 9),
            foreground='gray'
        ).pack(pady=5)
        
        # Action buttons
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Button(
            action_frame,
            text="🧠 Start Active Learning",
            command=self.run_active_learning,
            style='Accent.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            action_frame,
            text="📖 View Documentation",
            command=self.show_active_learning_help
        ).pack(side='left', padx=5)
        
        # Status/log section
        log_frame = ttk.LabelFrame(scrollable_frame, text="📋 Status", padding=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.al_log = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.al_log.pack(fill='both', expand=True)
        self.al_log.insert('1.0', 'Ready to start Active Learning.\n\n'
                                  'Select your files above and click "Start Active Learning".\n')
        self.al_log.config(state='disabled')
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
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

        # Refresh classifier dropdowns
        self.refresh_pred_classifiers()
        self.refresh_pred_videos()

        # Sync analysis tab project folder and trigger background scan
        if hasattr(self, 'analysis_tab') and self.analysis_tab is not None:
            if hasattr(self.analysis_tab, 'analysis_project_var'):
                self.analysis_tab.analysis_project_var.set(folder)
                # Defer scan slightly so the tab finishes any pending layout
                self.root.after(200, lambda: self.analysis_tab.scan_project_folder(folder))

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

        config_path = os.path.join(folder, 'PixelPaws_project.json')

        # Load existing config to use as base (preserves fields we're not updating)
        existing = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Build updates — only non-empty values overwrite existing entries
        updates = {
            'project_folder': folder,
        }
        for key, getter in [
            ('video_ext',     lambda: self.train_video_ext.get() if self.train_video_ext else ''),
            ('dlc_config',    lambda: self.train_dlc_config.get() if hasattr(self, 'train_dlc_config') and self.train_dlc_config else ''),
            ('behavior_name', lambda: self.train_behavior_name.get() if self.train_behavior_name else ''),
        ]:
            val = getter()
            if val:  # only overwrite when we actually have a value
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

    def _load_project_config(self, config_path, silent=False):
        """Load project-level config and populate tab fields."""
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if self.train_video_ext is not None and config.get('video_ext'):
                self.train_video_ext.set(config['video_ext'])
            if hasattr(self, 'train_dlc_config') and self.train_dlc_config is not None and config.get('dlc_config'):
                self.train_dlc_config.set(config['dlc_config'])

            # Load behaviors list → pre-fill first entry into training tab
            behaviors = config.get('behaviors') or []
            if not behaviors and config.get('behavior_name'):
                behaviors = [config['behavior_name']]
            if behaviors and self.train_behavior_name is not None:
                self.train_behavior_name.set(behaviors[0])

            # Load brightness body parts → pre-fill training tab field
            bp = config.get('bp_pixbrt_list', [])
            if bp and self.train_bp_pixbrt is not None:
                self.train_bp_pixbrt.set(','.join(bp) if isinstance(bp, list) else bp)

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

        sessions = []
        skipped_videos = []
        stopped_by_user = False

        for s in candidates:
            base         = s['session_name']
            video_path   = s['video']
            video_dir    = s['video_dir']
            project_dir  = s['project_dir']
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            target_path  = s['labels']

            if target_path is None:
                if video_basename in skipped_videos:
                    continue

                self.log_train(f"⚠️  Warning: No labels file found for {video_basename}")
                self.log_train(f"    Tried:")
                self.log_train(f"      - {os.path.join(project_dir, 'behavior_labels', f'{video_basename}_labels.csv')}")
                self.log_train(f"      - {os.path.join(video_dir, f'{video_basename}_labels.csv')}")
                self.log_train(f"      - {os.path.join(project_dir, 'labels', f'{video_basename}_labels.csv')}")

                response = messagebox.askyesno(
                    "Labels Not Found",
                    f"No labels file found for:\n\n"
                    f"Video: {video_basename}\n\n"
                    f"Searched locations:\n"
                    f"• {os.path.join(project_dir, 'behavior_labels', f'{video_basename}_labels.csv')}\n"
                    f"• {os.path.join(video_dir, f'{video_basename}_labels.csv')}\n"
                    f"• {os.path.join(project_dir, 'labels', f'{video_basename}_labels.csv')}\n\n"
                    f"Would you like to SKIP this video and continue?\n\n"
                    f"Click 'Yes' to skip this video\n"
                    f"Click 'No' to stop and label this video first"
                )

                if response:
                    self.log_train(f"    ↳ Skipping video: {video_basename}")
                    skipped_videos.append(video_basename)
                    continue
                else:
                    self.log_train(f"    ↳ Stopped by user. Please label the video and try again.")
                    stopped_by_user = True
                    break

            sessions.append({
                'session_name': base,
                'pose_path':    s['dlc'],
                'video_path':   video_path,
                'target_path':  target_path,
            })

        if skipped_videos:
            self.log_train(f"\n📋 Scan Summary:")
            self.log_train(f"   ✓ Found {len(sessions)} videos with labels")
            self.log_train(f"   ⊗ Skipped {len(skipped_videos)} videos without labels:")
            for vid in skipped_videos:
                self.log_train(f"      - {vid}")
        elif stopped_by_user:
            self.log_train(f"\n⚠️  Scan stopped by user")
            return []
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
            dialog.geometry("400x500")
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
        dialog.geometry("550x450")
        
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
        
        if not self.train_behavior_name.get():
            messagebox.showwarning("No Behavior", "Please enter a behavior name.")
            return
        
        # Show training visualization window
        if self.train_viz_window is None or not self.train_viz_window.window.winfo_exists():
            self.train_viz_window = TrainingVisualizationWindow(self.root, self.theme)
        
        # Run REAL training
        threading.Thread(target=self._real_training, daemon=True).start()
    
    def _real_training(self):
        """ACTUAL classifier training implementation"""
        try:
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
            
            # Find training sessions
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
                'use_gpu': self.train_use_gpu.get(),
                'include_optical_flow': self.train_include_optical_flow.get(),
                'bp_optflow_list': [x.strip() for x in self.train_bp_optflow.get().split(',') if x.strip()]
                    if self.train_include_optical_flow.get() else [],
            }
            
            # Setup feature caching
            feature_cache_root = os.path.join(project_folder, 'features')
            os.makedirs(feature_cache_root, exist_ok=True)
            
            # ── Feature extraction ─────────────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("FEATURE EXTRACTION")
            self.log_train("=" * 60)
            
            all_X = []
            all_y = []
            session_ids = []
            
            for i, session in enumerate(sessions):
                X_s, y_s = self.extract_features_for_session(
                    session, cfg, feature_cache_root, behavior_name)
                all_X.append(X_s)
                all_y.append(y_s)
                session_ids.extend([i] * len(y_s))
                
                pos_count = np.sum(y_s)
                pos_pct = (pos_count / len(y_s)) * 100 if len(y_s) > 0 else 0
                self.log_train(
                    f"  Session {i+1}/{len(sessions)}: {len(y_s)} frames, "
                    f"{pos_count} positive ({pos_pct:.1f}%)")
            
            X = pd.concat(all_X, ignore_index=True)
            y = np.concatenate(all_y)
            session_ids = np.array(session_ids)
            
            pos_total = np.sum(y)
            neg_total = len(y) - pos_total
            self.log_train(
                f"\nTotal: {len(X)} frames, {pos_total} positive "
                f"({np.mean(y)*100:.1f}%), {neg_total} negative")
            
            # ── GPU detection (once, before the CV loop) ───────────────
            tree_method = 'hist'
            if self.train_use_gpu.get():
                try:
                    import xgboost as xgb_test
                    tm = xgb_test.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
                    tm.fit([[0, 0]], [0])
                    tree_method = 'gpu_hist'
                    self.log_train("\nUsing GPU acceleration (gpu_hist)")
                except Exception as e:
                    self.log_train(f"\nGPU not available, using CPU (hist): {e}")
            
            # ── Session-level K-Fold Cross-Validation ─────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("CROSS-VALIDATION")
            self.log_train("=" * 60)
            
            n_folds = self.train_n_folds.get()
            unique_sessions = np.unique(session_ids)
            actual_folds = min(n_folds, len(unique_sessions))
            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
            
            fold_f1_scores   = []
            fold_precisions  = []
            fold_recalls     = []
            fold_best_iters  = []   # for early stopping → final n_estimators
            
            # OOF containers — one slot per training frame, filled as folds run
            oof_proba  = np.full(len(y), np.nan)
            
            for fold, (train_sess_idx, val_sess_idx) in enumerate(
                    kf.split(unique_sessions), 1):
                
                fold_start = time.time()
                self.log_train(f"\n=== Fold {fold}/{actual_folds} ===")
                
                train_sess = unique_sessions[train_sess_idx]
                val_sess   = unique_sessions[val_sess_idx]
                
                train_mask = np.isin(session_ids, train_sess)
                val_mask   = np.isin(session_ids, val_sess)
                
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_val   = X[val_mask]
                y_val   = y[val_mask]
                
                self.log_train(
                    f"  Train: {len(X_train)} frames, {np.sum(y_train)} positive")
                self.log_train(
                    f"  Val:   {len(X_val)} frames, {np.sum(y_val)} positive")
                
                # ── Class imbalance handling ───────────────────────────
                # scale_pos_weight: weight the positive class without
                # throwing away any negative frames
                spw = 1.0
                if use_spw and np.sum(y_train) > 0:
                    spw = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
                    self.log_train(f"  scale_pos_weight = {spw:.2f}")
                
                # Legacy downsampling fallback (off by default)
                if (self.train_use_balancing.get()
                        and np.mean(y_train) < self.train_imbalance_thresh.get()
                        and not use_spw):
                    self.log_train("  Applying downsampling...")
                    X_train, y_train = self.balance_data(
                        X_train.values, y_train)
                    X_train = pd.DataFrame(X_train, columns=X.columns)
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
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='aucpr',   # precision-recall AUC: better than logloss for imbalanced data
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
                    fold_best_iters.append(best_iter + 1)  # +1: 0-indexed
                    self.log_train(f"  Early stopping at tree {best_iter + 1}")
                else:
                    fold_model.fit(X_train, y_train)
                
                # ── OOF probabilities for this fold ───────────────────
                val_proba = fold_model.predict_proba(X_val)[:, 1]
                oof_proba[val_mask] = val_proba
                
                # Fold metrics (raw 0.5 threshold — CV is about the model,
                # not the post-processing; sweep happens after)
                val_pred = (val_proba >= 0.5).astype(int)
                f1   = f1_score(y_val, val_pred, zero_division=0)
                prec = precision_score(y_val, val_pred, zero_division=0)
                rec  = recall_score(y_val, val_pred, zero_division=0)
                
                fold_f1_scores.append(f1)
                fold_precisions.append(prec)
                fold_recalls.append(rec)
                
                elapsed = time.time() - fold_start
                self.log_train(
                    f"  F1: {f1:.3f}, Precision: {prec:.3f}, "
                    f"Recall: {rec:.3f}  ({elapsed:.1f}s)")
                
                if self.train_viz_window and \
                        self.train_viz_window.window.winfo_exists():
                    self.train_viz_window.add_fold_result(
                        fold, f1, prec, rec, elapsed)
            
            mean_f1 = np.mean(fold_f1_scores)
            std_f1  = np.std(fold_f1_scores)
            
            self.log_train(f"\nCross-Validation Results (at threshold 0.5):")
            self.log_train(f"  Mean F1:        {mean_f1:.3f} ± {std_f1:.3f}")
            self.log_train(f"  Mean Precision: {np.mean(fold_precisions):.3f}")
            self.log_train(f"  Mean Recall:    {np.mean(fold_recalls):.3f}")
            
            # ── OOF post-processing sweep ──────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("OUT-OF-FOLD PARAMETER SWEEP")
            self.log_train("=" * 60)
            self.log_train(
                "Finding best threshold, min_bout, and max_gap on OOF predictions\n"
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
                f"  Max Gap:      {best_params['max_gap']} frames  "
                f"(UI: {self.train_max_gap.get()})")
            
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
                objective='binary:logistic',
                random_state=42,
            )
            
            final_model.fit(X, y)
            self.log_train("  ✓ Final model trained on all data")

            # ── SHAP prune + retrain (optional second pass) ────────────
            selected_feature_cols = None  # None → use all (model.feature_names_in_)
            pre_prune_model_ref = None   # holds the full-feature model when SHAP pruning runs

            if self.train_shap_prune.get():
                top_n = self.train_shap_top_n.get()
                self.log_train(f"\nSHAP pruning: keeping top {top_n} features...")
                try:
                    import shap as _shap
                    n_sample = min(3000, len(X))
                    sample_df = X.sample(n_sample, random_state=42)

                    explainer  = _shap.TreeExplainer(final_model)
                    shap_vals  = explainer.shap_values(sample_df)

                    # shap_values() returns a 2-D array for XGBoost binary
                    # (or a list for some older SHAP builds — handle both)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1]   # positive-class values
                    else:
                        sv = shap_vals

                    mean_abs = np.abs(sv).mean(axis=0)
                    importance = pd.Series(mean_abs, index=final_model.feature_names_in_)
                    top_n_actual = min(top_n, len(importance))
                    top_cols = importance.nlargest(top_n_actual).index.tolist()

                    self.log_train(
                        f"  Pruned: {len(importance)} → {len(top_cols)} features")

                    # Retrain on pruned feature set (same hyperparams)
                    X_pruned = X[top_cols]
                    pruned_model = xgb.XGBClassifier(**final_model.get_params())
                    pruned_model.fit(X_pruned, y)

                    pre_prune_model_ref  = final_model   # save for comparison plots
                    final_model          = pruned_model
                    selected_feature_cols = top_cols
                    self.log_train("  ✓ Prune + retrain complete.")

                except Exception as _shap_err:
                    self.log_train(
                        f"  ⚠️  SHAP pruning failed ({_shap_err}), "
                        f"using full-feature model instead.")

            # ── Save classifier ────────────────────────────────────────
            self.log_train("\n" + "=" * 60)
            self.log_train("SAVING CLASSIFIER")
            self.log_train("=" * 60)
            
            classifier_folder = os.path.join(project_folder, 'classifiers')
            os.makedirs(classifier_folder, exist_ok=True)
            
            classifier_data = {
                # Core model
                'clf_model':        final_model,
                'Behavior_type':    behavior_name,
                # SHAP-pruned feature subset (None if pruning was not used)
                'selected_feature_cols': selected_feature_cols,
                # OOF-optimised post-processing params (used by default)
                'best_thresh':      best_params['thresh'],
                'min_bout':         best_params['min_bout'],
                'min_after_bout':   self.train_min_after_bout.get(),
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
            }
            
            classifier_path = os.path.join(
                classifier_folder, f'PixelPaws_{behavior_name}.pkl')
            
            with open(classifier_path, 'wb') as f:
                pickle.dump(classifier_data, f)
            
            self.log_train(f"\n  ✓ Classifier saved: {classifier_path}")

            # ── Also save the full-feature (pre-prune) model when pruning was active ──
            if pre_prune_model_ref is not None:
                pre_prune_data = dict(classifier_data)   # shallow copy — same metadata
                pre_prune_data['clf_model']             = pre_prune_model_ref
                pre_prune_data['selected_feature_cols'] = None   # uses all features
                pre_prune_path = os.path.join(
                    classifier_folder, f'PixelPaws_{behavior_name}_AllFeatures.pkl')
                with open(pre_prune_path, 'wb') as f:
                    pickle.dump(pre_prune_data, f)
                self.log_train(f"  ✓ Full-feature classifier saved: {pre_prune_path}")

            # Training data backup
            train_set_path = os.path.join(
                classifier_folder, f'{behavior_name}_train_set.pkl')
            with open(train_set_path, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
            self.log_train(f"  ✓ Training set saved: {train_set_path}")
            
            # Plots
            if self.train_generate_plots.get():
                self.log_train("\nGenerating plots...")
                self.generate_performance_plots(
                    final_model, X, y, classifier_folder, behavior_name,
                    oof_proba=oof_proba, oof_best_params=best_params,
                    pre_prune_model=pre_prune_model_ref)
                self.log_train("  ✓ Plots saved")
            
            self.log_train("\n" + "=" * 60)
            self.log_train("✓✓✓ TRAINING COMPLETE! ✓✓✓")
            self.log_train("=" * 60)
            self.log_train(f"\nClassifier: {classifier_path}")
            self.log_train(
                f"CV F1 (@ 0.5):  {mean_f1:.3f} ± {std_f1:.3f}")
            self.log_train(
                f"OOF F1 (tuned): {best_params['f1']:.3f}  "
                f"(thresh={best_params['thresh']:.2f}, "
                f"min_bout={best_params['min_bout']}, "
                f"max_gap={best_params['max_gap']})")
            
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
                    messagebox.showinfo(
                        "Active Learning Success! 🎉",
                        f"Model improved with active learning!\n\n"
                        f"Before: F1 = {pre_f1:.3f} ± {pre_std:.3f}\n"
                        f"After:  F1 = {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                        f"✨ Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%)\n\n"
                        f"The new labels helped refine the decision boundary!")
                elif improvement < -0.01:
                    self.log_train(
                        f"⚠️  DECREASE: {improvement:.3f} ({pct_improvement:.1f}%)")
                    self.log_train(f"{'='*60}")
                    messagebox.showwarning(
                        "Performance Decreased",
                        f"F1 score decreased slightly after active learning.\n\n"
                        f"Before: {pre_f1:.3f} ± {pre_std:.3f}\n"
                        f"After:  {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                        f"Change: {improvement:.3f} ({pct_improvement:.1f}%)\n\n"
                        f"This can happen if:\n"
                        f"• New labels introduced noise\n"
                        f"• Very few frames were added\n"
                        f"• Labels were inconsistent\n\n"
                        f"Try labeling more frames or review labels for consistency.")
                else:
                    self.log_train(f"No significant change: {improvement:.3f}")
                    self.log_train(f"{'='*60}")
                    messagebox.showinfo(
                        "Performance Maintained",
                        f"F1 score remained stable after active learning.\n\n"
                        f"Before: {pre_f1:.3f} ± {pre_std:.3f}\n"
                        f"After:  {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                        f"The new labels didn't significantly impact performance.\n"
                        f"This might mean the model was already well-calibrated.")
                
                delattr(self, 'pre_active_learning_f1')
            
            # Store results for active learning
            self.last_training_results = {
                'classifier_path': classifier_path,
                'sessions':        sessions,
                'mean_f1':         mean_f1,
                'std_f1':          std_f1,
                'behavior_name':   behavior_name,
                'X':               X,
                'y':               y,
                'final_model':     final_model,
                'best_thresh':     best_params['thresh'],
            }

            # Auto-save project config so other tabs can pick up the new classifier
            self.save_project_config(project_folder)
            
            # Offer active learning
            response = messagebox.askyesno(
                "Training Complete! 🎉",
                f"Classifier trained successfully!\n\n"
                f"📊 Performance:\n"
                f"   CV F1 (@ 0.5):  {mean_f1:.3f} ± {std_f1:.3f}\n"
                f"   OOF F1 (tuned): {best_params['f1']:.3f}  "
                f"(thresh={best_params['thresh']:.2f}, "
                f"min_bout={best_params['min_bout']}, "
                f"max_gap={best_params['max_gap']})\n\n"
                f"💡 Would you like to run Active Learning?\n\n"
                f"Active Learning will:\n"
                f"  • Score all {len(sessions)} video(s) with the trained model\n"
                f"  • Select the most borderline frames globally\n"
                f"  • Let you label them grouped by video\n"
                f"  • Retrain with the improved labels\n\n"
                f"Run Active Learning now?",
                icon='question'
            )
            
            if response:
                self.log_train("\n" + "=" * 60)
                self.log_train("🧠 STARTING ACTIVE LEARNING")
                self.log_train("=" * 60)
                self.run_active_learning_after_training()
            else:
                messagebox.showinfo(
                    "Classifier Saved",
                    f"Classifier saved to:\n{classifier_path}\n\n"
                    f"You can run Active Learning later from the\n"
                    f"Active Learning tab to improve performance.")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log_train(
                f"\n\n{'='*60}\n✗ ERROR DURING TRAINING\n{'='*60}\n")
            self.log_train(error_msg)
            messagebox.showerror(
                "Training Failed",
                f"Error during training:\n\n{str(e)}\n\nSee log for details.")
    
    def run_active_learning_after_training(self):
        """Run cross-video active learning using the just-trained model."""
        try:
            if not hasattr(self, 'last_training_results'):
                messagebox.showerror("Error", "No recent training results found.")
                return

            results       = self.last_training_results
            sessions      = results['sessions']
            pre_f1        = results['mean_f1']
            pre_std       = results['std_f1']
            behavior_name = results['behavior_name']
            model         = results['final_model']   # use the real model, already in memory
            n_total       = self.al_n_suggestions.get() if self.al_n_suggestions else 100

            self.log_train(f"\n📊 Baseline Performance (Before Active Learning):")
            self.log_train(f"   Mean CV F1: {pre_f1:.3f} ± {pre_std:.3f}")
            self.log_train(f"\nScoring {len(sessions)} session(s) for the "
                           f"{n_total} most uncertain frames globally...\n")

            # ── Resolve feature cache paths for all sessions ──────────
            project_folder = self.train_project_folder.get()
            resolved = []
            skipped  = []

            for session in sessions:
                base_name  = session['session_name']
                video_dir  = os.path.dirname(session['video_path'])
                search_locs = [
                    video_dir,
                    os.path.dirname(video_dir),
                    os.path.join(os.path.dirname(video_dir), 'FeatureCache'),  # legacy
                    os.path.join(project_folder, 'FeatureCache'),              # legacy
                    os.path.join(project_folder, 'features'),                  # canonical
                ]
                cache_path = None
                for loc in search_locs:
                    if not os.path.exists(loc):
                        continue
                    matches = glob.glob(
                        os.path.join(loc, f"{base_name}_features*.pkl"))
                    if matches:
                        cache_path = matches[0]
                        break

                if not cache_path:
                    self.log_train(f"  ✗ Features cache not found for {base_name} — skipping")
                    skipped.append(base_name)
                    continue

                self.log_train(f"  ✓ {base_name}: {os.path.relpath(cache_path, project_folder)}")
                resolved.append({
                    'session_name':   base_name,
                    'video_path':     session['video_path'],
                    'labels_csv':     session['target_path'],
                    'features_cache': cache_path,
                    'behavior_name':  behavior_name,
                })

            if not resolved:
                messagebox.showerror(
                    "Active Learning Error",
                    "Could not find feature caches for any session.\n\n"
                    "Feature caches are created automatically during training.\n"
                    "Make sure training ran successfully before using Active Learning.")
                return

            if skipped:
                self.log_train(f"\n  ⚠️  Skipped {len(skipped)} session(s) without caches: "
                               f"{', '.join(skipped)}")

            # ── Run cross-video AL ────────────────────────────────────
            self.log_train(f"\nRunning cross-video scoring on {len(resolved)} session(s)...")
            stats = active_learning.run_cross_video_active_learning(
                sessions=resolved,
                model=model,
                n_total=n_total,
                min_frame_spacing=30,
            )

            # ── Summary and retrain offer ─────────────────────────────
            self.log_train(f"\n{'='*60}")
            self.log_train("📋 ACTIVE LEARNING SUMMARY")
            self.log_train(f"{'='*60}")
            self.log_train(f"Total new labels:     {stats['frames_labeled']}")
            self.log_train(f"Sessions updated:     {stats['sessions_updated']}/{len(resolved)}")
            for sname, count in stats['per_session'].items():
                self.log_train(f"  {sname}: {count} frames")
            self.log_train(f"\nBaseline F1: {pre_f1:.3f} ± {pre_std:.3f}")

            if stats['frames_labeled'] > 0:
                response = messagebox.askyesno(
                    "Active Learning Complete! 🎓",
                    f"Active Learning Summary:\n\n"
                    f"✓ Total new labels: {stats['frames_labeled']}\n"
                    f"✓ Sessions updated: {stats['sessions_updated']}/{len(resolved)}\n"
                    f"✓ Baseline F1: {pre_f1:.3f} ± {pre_std:.3f}\n\n"
                    f"🔄 Retrain model with new labels?\n\n"
                    f"This will re-run cross-validation with the updated labels\n"
                    f"and show the F1 improvement.\n\n"
                    f"Retrain now?",
                    icon='question'
                )
                if response:
                    self.log_train("\n🔄 Retraining with new labels...")
                    self.pre_active_learning_f1 = {'mean': pre_f1, 'std': pre_std}
                    self.root.after(2000, self.start_training)
                else:
                    self.log_train("\nℹ️  Retraining skipped — click 'Start Training' to retrain later.")
                    messagebox.showinfo(
                        "Labels Saved",
                        f"New labels ({stats['frames_labeled']}) saved to your CSV files.\n\n"
                        f"Click 'Start Training' again to retrain and see the F1 improvement.")
            else:
                messagebox.showwarning(
                    "No New Labels",
                    "No frames were labeled during Active Learning.\n\n"
                    "Check the training log for details.")

        except Exception as e:
            import traceback
            self.log_train(f"\n\n{'='*60}\n✗ ERROR IN ACTIVE LEARNING\n{'='*60}\n")
            self.log_train(traceback.format_exc())
            messagebox.showerror("Active Learning Error",
                                 f"An error occurred:\n\n{str(e)}\n\nSee log for details.")
            messagebox.showerror("Active Learning Failed", f"Error:\n\n{str(e)}")
    
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
                'use_gpu': self.train_use_gpu.get(),
                'generate_plots': self.train_generate_plots.get(),
                'shap_prune': self.train_shap_prune.get(),
                'shap_top_n': self.train_shap_top_n.get(),
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
            if 'use_gpu' in config:
                self.train_use_gpu.set(config['use_gpu'])
            if 'generate_plots' in config:
                self.train_generate_plots.set(config['generate_plots'])
            if 'shap_prune' in config:
                self.train_shap_prune.set(config['shap_prune'])
            if 'shap_top_n' in config:
                self.train_shap_top_n.set(config['shap_top_n'])
            
            messagebox.showinfo("Loaded", f"Configuration loaded from:\n{config_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration:\n{str(e)}")
    
    
    @staticmethod
    def _feature_hash_key(cfg):
        """Build a stable, type-normalized hash key for feature cache files.

        Uses explicit type coercion so the key is identical regardless of whether
        tk.BooleanVar.get() returns Python bool or int (platform-dependent on Windows).
        """
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

        # If not in the canonical location, search alternative directories.
        # This lets training pick up features pre-extracted by the Feature
        # Extraction tool (single-file or batch), the Predict tab, or any
        # other part of PixelPaws — regardless of where they were written.
        if not os.path.isfile(feature_cache_file):
            video_dir = os.path.dirname(session.get('video_path', ''))
            alt_dirs = [
                video_dir,
                os.path.join(video_dir, 'features'),
                os.path.join(video_dir, 'FeatureCache'),
                os.path.join(video_dir, 'PredictionCache'),
            ]
            # Walk ancestor directories up to the project root so deeply nested
            # video folders can still find their cached features.
            _ancestor = video_dir
            while True:
                _parent = os.path.dirname(_ancestor)
                if _parent == _ancestor:
                    break
                _ancestor = _parent
                alt_dirs.append(os.path.join(_ancestor, 'features'))
                alt_dirs.append(os.path.join(_ancestor, 'FeatureCache'))
                if os.path.normpath(_ancestor) == os.path.normpath(cache_root):
                    break
            for alt_dir in alt_dirs:
                alt_path = os.path.join(alt_dir, cache_filename)
                if os.path.isfile(alt_path):
                    self.log_train(f"  [Cache] Found in: {alt_dir}")
                    feature_cache_file = alt_path
                    break
            else:
                # Not found anywhere — glob to detect hash mismatches
                all_search_dirs = [cache_root] + alt_dirs
                mismatches = []
                for d in all_search_dirs:
                    if not os.path.isdir(d):
                        continue
                    for f in glob.glob(os.path.join(d, f"{session['session_name']}_features_*.pkl")):
                        mismatches.append(f)
                if mismatches:
                    self.log_train(
                        f"  [Cache] \u26a0 Feature file(s) found with DIFFERENT hash "
                        f"(config mismatch or stale cache):")
                    for m in mismatches:
                        self.log_train(f"    \u2192 {m}")
                    self.log_train(
                        f"  [Cache] Expected hash {cfg_hash}. "
                        f"Check that Feature Extraction settings match training settings.")
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
                use_gpu=cfg.get('use_gpu', True),  # Use GPU setting from config (default True)
                config_yaml_path=config_yaml,  # Pass config for crop detection
                include_optical_flow=cfg.get('include_optical_flow', False),
                bp_optflow_list=cfg.get('bp_optflow_list', []) or None,
            )
            X_full = X_full.reset_index(drop=True)
            
            # Drop NaNs from features
            nan_mask = X_full.isna().any(axis=1)
            if nan_mask.any():
                self.log_train(f"    Dropping {nan_mask.sum()} NaN rows from features")
                X_full = X_full[~nan_mask].reset_index(drop=True)
            
            # Cache features (behavior-independent, reusable!)
            with open(feature_cache_file, 'wb') as f:
                pickle.dump(X_full, f)
            self.log_train(f"    ✓ Cached features to {feature_cache_file}")
        
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
        
        y_full = y_df[behavior_name].astype(int).values
        
        # Align lengths (truncate to shorter of features or labels)
        n = min(len(X_full), len(y_full))
        if len(X_full) != len(y_full):
            self.log_train(f"    Aligning: features={len(X_full)}, labels={len(y_full)}, using {n}")
        
        X = X_full.iloc[:n].copy()
        y = y_full[:n]
        
        # If features had NaN rows removed, labels might be misaligned
        # This is handled by the min() above - we just use what we have
        
        return X, y
    
    def _sweep_postprocessing(self, oof_proba, y):
        """
        Joint grid search over (threshold, min_bout, max_gap) using
        out-of-fold probabilities.  Because these probabilities were
        produced by models that never trained on the corresponding frames,
        the chosen parameters are unbiased estimates of real-world
        post-processing performance.

        Returns a dict with keys: thresh, min_bout, max_gap, f1
        """
        from evaluation_tab import _apply_bout_filtering

        # Search grids — kept intentionally coarse so the sweep is fast
        thresholds  = np.arange(0.10, 0.91, 0.05)   # 17 values
        min_bouts   = [1, 2, 3, 5, 8, 12, 20]        # 7 values
        max_gaps    = [0, 2, 4, 6, 10, 15]            # 6 values
        # Total: 17 × 7 × 6 = 714 combinations — runs in <1 s on typical data

        best_f1    = -1.0
        best_thresh = 0.5
        best_mb     = 1
        best_mg     = 0

        for thresh in thresholds:
            y_raw = (oof_proba >= thresh).astype(int)
            for mb in min_bouts:
                for mg in max_gaps:
                    if mb == 1 and mg == 0:
                        y_filt = y_raw
                    else:
                        y_filt = _apply_bout_filtering(
                            y_raw.copy(), min_bout=mb,
                            min_after_bout=1, max_gap=mg)
                    score = f1_score(y, y_filt, zero_division=0)
                    if score > best_f1:
                        best_f1    = score
                        best_thresh = thresh
                        best_mb     = mb
                        best_mg     = mg

        return {
            'thresh':   float(round(best_thresh, 2)),
            'min_bout': int(best_mb),
            'max_gap':  int(best_mg),
            'f1':       float(best_f1),
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
    
    def find_optimal_threshold(self, model, X, y):
        """Find optimal classification threshold"""
        y_proba = predict_with_xgboost(model, X)
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        return best_thresh, best_f1
    
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

        # ── Pre-prune in-sample curve (only when SHAP pruning was active) ────
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

        fig, ax = plt.subplots(figsize=(11, 6))

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
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder,
                         f'PixelPaws_{behavior_name}_PerformanceThreshold.png'),
            dpi=300, bbox_inches='tight')
        plt.close()
        
        # ── SHAP importance ───────────────────────────────────────────
        try:
            import shap

            # ── Pre-prune SHAP (all features) — only when SHAP pruning was active ──
            if pre_prune_model is not None:
                X_pre_arr = X.values if hasattr(X, 'values') else X
                if len(X_pre_arr) > 5000:
                    idx_pre = np.random.choice(len(X_pre_arr), 5000, replace=False)
                    X_pre_sample = X_pre_arr[idx_pre]
                else:
                    X_pre_sample = X_pre_arr

                expl_pre = shap.TreeExplainer(pre_prune_model)
                sv_pre   = expl_pre.shap_values(X_pre_sample)

                if feature_names is not None:
                    shap.summary_plot(sv_pre,
                                      pd.DataFrame(X_pre_sample, columns=feature_names),
                                      show=False, max_display=n_pruned if n_pruned else 40)
                else:
                    shap.summary_plot(sv_pre, X_pre_sample, show=False,
                                      max_display=n_pruned if n_pruned else 40)

                plt.title(f'Feature Importance (all {n_full} features) — {behavior_name}',
                          fontsize=14)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_folder,
                                 f'PixelPaws_{behavior_name}_SHAP_AllFeatures.png'),
                    dpi=300, bbox_inches='tight')
                plt.close()

            # ── Pruned (or only) model SHAP ────────────────────────────────────
            if hasattr(model, 'feature_names_in_'):
                X_for_shap = X[model.feature_names_in_]
                shap_feat_names = list(model.feature_names_in_)
            else:
                X_for_shap = X if hasattr(X, 'columns') else pd.DataFrame(X_array)
                shap_feat_names = feature_names
            X_shap_arr = X_for_shap.values if hasattr(X_for_shap, 'values') else X_for_shap

            if len(X_shap_arr) > 5000:
                sample_idx = np.random.choice(len(X_shap_arr), 5000, replace=False)
                X_sample = X_shap_arr[sample_idx]
            else:
                X_sample = X_shap_arr

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if shap_feat_names is not None:
                X_sample_df = pd.DataFrame(X_sample, columns=shap_feat_names)
                shap.summary_plot(shap_values, X_sample_df,
                                  show=False,
                                  max_display=len(shap_feat_names) if shap_feat_names else 40)
            else:
                shap.summary_plot(shap_values, X_sample,
                                  show=False, max_display=40)

            n_shap = len(shap_feat_names) if shap_feat_names else ''
            shap_title = (f'Feature Importance (top {n_shap} pruned features) — {behavior_name}'
                          if pre_prune_model is not None
                          else f'Feature Importance — {behavior_name}')
            plt.title(shap_title, fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_folder,
                             f'PixelPaws_{behavior_name}_SHAP_Importance.png'),
                dpi=300, bbox_inches='tight')
            plt.close()

        except (ImportError, Exception) as e:
            self.log_train(f"    SHAP plot skipped: {str(e)}")
    
    def log_train(self, message):
        """Add message to training log"""
        if self.train_log:
            self.train_log.insert(tk.END, message + '\n')
            self.train_log.see(tk.END)
            self.root.update_idletasks()
    
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
        progress_window.geometry("500x200")
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
        results_window.geometry("1000x700")
        
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
        window.geometry("900x700")
        
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
        win.geometry("720x820")
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

        fe_use_gpu = tk.BooleanVar(
            value=self.train_use_gpu.get() if hasattr(self, 'train_use_gpu') else True)
        ttk.Checkbutton(settings, text="Use GPU acceleration",
                        variable=fe_use_gpu).grid(row=3, column=1, sticky='w', pady=3)

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

        run_btn = ttk.Button(btn_frame, text="▶  Run Extraction", style='Accent.TButton')
        run_btn.pack(side='left', padx=4)
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
                'use_gpu':        fe_use_gpu.get(),
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

            run_btn.config(state='disabled')
            log_text.delete('1.0', tk.END)
            threading.Thread(
                target=self._run_feature_extraction_thread,
                args=(sessions, cache_root,
                      fe_project.get().strip() if mode == 'batch' else None,
                      cfg, log, lambda: run_btn.config(state='normal')),
                daemon=True
            ).start()

        run_btn.config(command=start)

    def _run_feature_extraction_thread(self, sessions, cache_root,
                                        project_folder, cfg, log_fn, done_fn):
        """Background worker for the Feature Extraction tool.

        sessions     : list of dicts with session_name/pose_path/video_path,
                       or None → scan project_folder for pairs.
        cache_root   : directory where .pkl files are written.
        project_folder: only used when sessions is None (batch scan).
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
                        use_gpu=cfg['use_gpu'],
                        config_yaml_path=cfg.get('dlc_config'),
                        include_optical_flow=cfg['include_optical_flow'],
                        bp_optflow_list=cfg['bp_optflow_list'] or None,
                    )
                    X = X.reset_index(drop=True)
                    with open(cache_file, 'wb') as f:
                        pickle.dump(X, f)
                    log(f"  ✓ {X.shape[0]} frames × {X.shape[1]} features")
                    log(f"     → {cache_file}")
                except Exception as e:
                    log(f"  ✗ Error: {e}")
                    errors += 1

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
        self.apply_theme()
        messagebox.showinfo("Theme Changed", f"Switched to {new_mode} mode")
    
    
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
            progress.geometry("600x400")
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
            
            # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
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
                    
                    # Create cache key based on configuration (match preview hash)
                    import hashlib

                    cfg_key = {
                        'bp_include_list': clf_data.get('bp_include_list'),
                        'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                        'square_size': clf_data.get('square_size', [40]),
                        'pix_threshold': clf_data.get('pix_threshold', 0.3),
                        'pose_feature_version': POSE_FEATURE_VERSION,
                        'include_optical_flow': clf_data.get('include_optical_flow', False),
                        'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                    }
                    cfg_hash = hashlib.md5(repr(cfg_key).encode()).hexdigest()[:8]

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
                            use_gpu=True,  # GPU enabled (auto-fallback)
                            config_yaml_path=config_yaml,  # Auto-detect crop from config
                            include_optical_flow=clf_data.get('include_optical_flow', False),
                            bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                        )

                        # Save to cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump(X, f)
                        results_text.insert(tk.END, f"  ✓ Cached to {cache_file}\n")
                    
                    proba = predict_with_xgboost(model, X)
                    
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
        optimizer_window.geometry("900x750")
        
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
                
                # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
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
                    
                    # Create cache key (match preview hash)
                    import hashlib

                    cfg_key = {
                        'bp_include_list': clf_data.get('bp_include_list'),
                        'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                        'square_size': clf_data.get('square_size', [40]),
                        'pix_threshold': clf_data.get('pix_threshold', 0.3),
                        'pose_feature_version': POSE_FEATURE_VERSION,
                        'include_optical_flow': clf_data.get('include_optical_flow', False),
                        'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                    }
                    cfg_hash = hashlib.md5(repr(cfg_key).encode()).hexdigest()[:8]

                    video_name = os.path.splitext(os.path.basename(file_set['video']))[0]
                    video_dir = os.path.dirname(file_set['video'])

                    # Check both possible cache locations
                    cache_locations = [
                        os.path.join(video_dir, 'PredictionCache', f"{video_name}_features_{cfg_hash}.pkl"),  # From preview
                        os.path.join(video_dir, 'FeatureCache', f"{video_name}_features_{cfg_hash}.pkl"),     # From optimizer
                    ]

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
                        # Extract features and save to FeatureCache
                        results_text.insert(tk.END, f"  Extracting features...\n")
                        optimizer_window.update()

                        cache_dir = os.path.join(video_dir, 'FeatureCache')
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_file = os.path.join(cache_dir, f"{video_name}_features_{cfg_hash}.pkl")

                        # Try to find config.yaml for crop detection
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
                            pose_data_file=file_set['dlc'],
                            video_file_path=file_set['video'],
                            bp_include_list=clf_data.get('bp_include_list'),
                            bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                            square_size=clf_data.get('square_size', [40]),
                            pix_threshold=clf_data.get('pix_threshold', 0.3),
                            use_gpu=True,  # GPU enabled (auto-fallback)
                            config_yaml_path=config_yaml,  # Pass config for crop detection
                            include_optical_flow=clf_data.get('include_optical_flow', False),
                            bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                        )

                        # Save to cache
                        with open(cache_file, 'wb') as f:
                            pickle.dump(X, f)
                        results_text.insert(tk.END, f"  ✓ Cached for future use\n")
                    
                    # Get probabilities
                    proba = predict_with_xgboost(model, X)
                    
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
        optimizer_window.geometry("800x700")
        
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
                    use_gpu=True,  # GPU enabled (auto-fallback)
                    config_yaml_path=config_yaml,  # Pass config for crop detection
                )
                
                # Get probabilities
                y_proba = predict_with_xgboost(model, X)
                
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
        converter_window.geometry("700x550")
        
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
        
        ttk.Label(param_frame, text="Behavior Name:").grid(row=0, column=0, sticky='w', pady=5)
        behavior_var = tk.StringVar(value="L_licking")
        behavior_entry = ttk.Entry(param_frame, textvariable=behavior_var, width=30)
        behavior_entry.grid(row=0, column=1, sticky='w', padx=5)
        
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
                except:
                    try:
                        df = pd.read_csv(boris_path, sep='\t')
                    except:
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
                dialog.geometry("400x400")
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
        
        ttk.Button(param_frame, text="🔍 Auto-Detect", command=auto_detect_behaviors).grid(
            row=0, column=2, padx=5)
        
        ttk.Label(param_frame, text="Video FPS:").grid(row=1, column=0, sticky='w', pady=5)
        fps_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=fps_var, width=15).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(param_frame, text="(leave blank to auto-detect from FPS column)").grid(row=1, column=2, sticky='w')
        
        # Output directory
        output_frame = ttk.LabelFrame(converter_window, text="Output", padding=10)
        output_frame.pack(fill='x', padx=10, pady=10)
        
        output_dir_var = tk.StringVar()
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky='w')
        ttk.Entry(output_frame, textvariable=output_dir_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="📁 Browse", 
                  command=lambda: output_dir_var.set(filedialog.askdirectory())).grid(row=0, column=2)
        
        # Status
        status_label = ttk.Label(converter_window, text="", foreground='blue')
        status_label.pack(pady=5)
        
        def get_fps(fps_text, df):
            """Get FPS from input or DataFrame"""
            fps_text = fps_text.strip()
            if fps_text:
                try:
                    return float(fps_text)
                except:
                    raise ValueError("FPS must be a positive number")
            
            # Try to find FPS column
            fps_col = find_column(df, ["FPS", "FrameRate", "Frame rate"])
            if fps_col and not df[fps_col].isna().all():
                return float(df[fps_col].dropna().iloc[0])
            
            raise ValueError("Could not determine FPS. Please enter it manually or include FPS column.")
        
        def run_conversion():
            boris_path = boris_file_var.get().strip()
            if not boris_path or not os.path.isfile(boris_path):
                messagebox.showwarning("No File", "Please select a valid BORIS file.")
                return
            
            behavior_name = behavior_var.get().strip()
            if not behavior_name:
                messagebox.showwarning("No Behavior", "Please enter a behavior name.")
                return
            
            output_dir = output_dir_var.get().strip()
            if not output_dir:
                output_dir = os.path.dirname(boris_path)
            
            try:
                status_label.config(text=f"Loading {os.path.basename(boris_path)}...")
                converter_window.update()
                
                # Load file - try CSV first (more common), then TSV
                df = None
                try:
                    df = pd.read_csv(boris_path)
                    # Check if it has enough columns - if only 1-2, might be TSV
                    if len(df.columns) < 3:
                        df = pd.read_csv(boris_path, sep='\t')
                except:
                    try:
                        df = pd.read_csv(boris_path, sep='\t')
                    except Exception as e2:
                        raise Exception(f"Could not read file as CSV or TSV: {e2}")
                
                if df is None or df.empty:
                    raise Exception("File is empty or could not be parsed")
                
                # Get FPS
                fps_val = get_fps(fps_var.get(), df)
                
                # Find required columns
                behavior_col = find_column(df, ["Behavior", "behaviour"])
                type_col = find_column(df, ["Behavior type", "Type"])
                time_col = find_column(df, ["Time", "Time (s)", "time"])
                
                if not all([behavior_col, type_col, time_col]):
                    messagebox.showerror("Error", 
                        "Could not find required columns: Behavior, Behavior type, and Time.\n"
                        "Make sure your BORIS export includes these columns.")
                    return
                
                status_label.config(text="Converting events to frames...")
                converter_window.update()
                
                # Sort by time
                df_sorted = df.sort_values(time_col).reset_index(drop=True)
                
                # Get max time to determine video length
                max_time = df_sorted[time_col].max()
                
                # Try to get video duration from Media duration column
                duration_col = find_column(df, ["Media duration", "Media duration (s)"])
                if duration_col and not df[duration_col].isna().all():
                    video_duration = float(df[duration_col].dropna().iloc[0])
                    max_time = max(max_time, video_duration)
                
                # Calculate total frames
                n_frames = int(np.ceil(max_time * fps_val))
                
                # Initialize all frames as 0
                labels = np.zeros(n_frames, dtype=int)
                
                # Process START/STOP events
                active_start = None
                
                for _, row in df_sorted.iterrows():
                    beh = str(row[behavior_col])
                    beh_type = str(row[type_col]).strip().upper()
                    t = float(row[time_col])
                    
                    if beh != behavior_name:
                        continue
                    
                    if beh_type == "START":
                        active_start = t
                    
                    elif beh_type == "STOP":
                        if active_start is not None:
                            # Mark frames from START to STOP as 1
                            frame_start = int(round(active_start * fps_val))
                            frame_end = int(round(t * fps_val))
                            for f in range(frame_start, min(frame_end, n_frames)):
                                labels[f] = 1
                            active_start = None
                    
                    elif beh_type == "POINT":
                        frame = int(round(t * fps_val))
                        if 0 <= frame < n_frames:
                            labels[frame] = 1
                
                # Save per-frame CSV (PixelPaws format: just behavior column, no Frame column)
                output_df = pd.DataFrame({behavior_name: labels})
                base_name = os.path.splitext(os.path.basename(boris_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_labels.csv")
                output_df.to_csv(output_path, index=False)
                
                status_label.config(text=f"✓ Conversion successful!", foreground='green')
                
                n_positive = np.sum(labels)
                pct = (n_positive / n_frames) * 100 if n_frames > 0 else 0
                
                messagebox.showinfo("Success", 
                    f"BORIS → PixelPaws conversion complete!\n\n"
                    f"Output: {output_path}\n\n"
                    f"Total frames: {n_frames}\n"
                    f"Behavior frames: {n_positive} ({pct:.1f}%)\n"
                    f"FPS used: {fps_val}\n\n"
                    f"Format: One column '{behavior_name}' with 0/1 per frame\n"
                    f"File is named {os.path.basename(output_path)} — place it in\n"
                    f"the same folder as the matching video for PixelPaws to find it automatically."
                )
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(error_details)
                status_label.config(text="✗ Conversion failed", foreground='red')
                messagebox.showerror("Conversion Error", f"Error: {str(e)}\n\nSee console for details.")
        
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
        """Apply current theme to all widgets"""
        # This is a simplified version - full implementation would recursively apply to all widgets
        bg = self.theme.colors['bg']
        fg = self.theme.colors['fg']
        
        try:
            self.root.configure(bg=bg)
            self.train_log.configure(bg=self.theme.colors['text_bg'], 
                                   fg=fg,
                                   insertbackground=fg)
        except:
            pass
    
    # === UTILITY METHODS ===
    
    def set_status(self, message):
        """Update status bar"""
        self.status_text.set(message)
        self.root.update_idletasks()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
PixelPaws - Behavioral Analysis & Recognition

Enhanced features:
✓ Video Preview with Predictions
✓ Real-Time Training Visualization
✓ Auto-Label Suggestion Mode
✓ Behavior Ethogram Generator
✓ Data Quality Checker
✓ Dark Mode

Version: 2.0 Enhanced
"""
        messagebox.showinfo("About PixelPaws", about_text)
    
    def show_docs(self):
        """Show documentation"""
        docs_text = """
PixelPaws Documentation

ENHANCED FEATURES:

1. VIDEO PREVIEW
   - View videos with prediction overlays
   - Jump to behavior bouts
   - Frame-by-frame navigation

2. AUTO-LABEL ASSISTANT
   - AI suggests uncertain frames
   - Quick correction interface
   - Export improved labels

3. DATA QUALITY CHECKER
   - Pre-training validation
   - Identifies issues early
   - Comprehensive reports

4. ETHOGRAM GENERATOR
   - Time budget analysis
   - Bout statistics
   - Publication-ready plots

5. REAL-TIME TRAINING VIZ
   - Live F1 score tracking
   - Timing analysis
   - Early stopping guidance

For detailed documentation, see the README files.
"""
        messagebox.showinfo("Documentation", docs_text)
    
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
Ctrl+E      - Generate ethogram

Space       - Play/Pause (in video preview)
Left/Right  - Previous/Next frame
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    # ===== PREDICTION TAB METHODS =====
    
    def refresh_pred_classifiers(self):
        """Populate the predict-tab classifier dropdown from project classifiers/ folder."""
        clf_dir = os.path.join(self.current_project_folder.get(), 'classifiers')
        self.pred_classifier_options = {}
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    self.pred_classifier_options[f] = os.path.join(clf_dir, f)
        if hasattr(self, 'pred_classifier_combo'):
            self.pred_classifier_combo['values'] = list(self.pred_classifier_options.keys())

    def _on_pred_classifier_selected(self, event=None):
        """Update the full path StringVar when a dropdown item is chosen."""
        name = self.pred_classifier_combo.get()
        if name in self.pred_classifier_options:
            self.pred_classifier_path.set(self.pred_classifier_options[name])

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
        dialog.geometry("500x400")
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
        threshold_spin = tk.Spinbox(custom_frame, from_=0.01, to=0.99, increment=0.01,
                                    textvariable=threshold_var, width=12, state='disabled')
        threshold_spin.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # Min Bout
        ttk.Label(custom_frame, text="Min Bout (frames):").grid(row=2, column=0, sticky='w', pady=5)
        min_bout_var = tk.IntVar(value=default_min_bout)
        min_bout_spin = tk.Spinbox(custom_frame, from_=1, to=1000, textvariable=min_bout_var, 
                   width=12, state='disabled')
        min_bout_spin.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        
        # Min After Bout
        ttk.Label(custom_frame, text="Min After Bout (frames):").grid(row=3, column=0, sticky='w', pady=5)
        min_after_var = tk.IntVar(value=default_min_after)
        min_after_spin = tk.Spinbox(custom_frame, from_=1, to=1000, textvariable=min_after_var, 
                   width=12, state='disabled')
        min_after_spin.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        
        # Max Gap
        ttk.Label(custom_frame, text="Max Gap (frames):").grid(row=4, column=0, sticky='w', pady=5)
        max_gap_var = tk.IntVar(value=default_max_gap)
        max_gap_spin = tk.Spinbox(custom_frame, from_=0, to=1000, textvariable=max_gap_var, 
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
            
            # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
            clf_data_preview['bp_include_list'] = clean_bodyparts_list(clf_data_preview.get('bp_include_list', []))
            clf_data_preview['bp_pixbrt_list'] = clean_bodyparts_list(clf_data_preview.get('bp_pixbrt_list', []))
            
            # Show parameter adjustment dialog
            param_result = self.show_parameter_dialog(clf_data_preview)
            if param_result is None:  # User cancelled
                return
            
            # Run prediction in background thread, then create window on main thread
            progress = tk.Toplevel(self.root)
            progress.title("Generating Predictions...")
            progress.geometry("400x150")
            
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
                    
                    # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
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
                    
                    # Create cache key
                    import hashlib
                    video_name = os.path.basename(video_path)
                    
                    cfg_key = {
                        'bp_include_list': clf_data.get('bp_include_list'),
                        'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                        'square_size': clf_data.get('square_size', [40]),
                        'pix_threshold': clf_data.get('pix_threshold', 0.3),
                        'pose_feature_version': POSE_FEATURE_VERSION,
                        'include_optical_flow': clf_data.get('include_optical_flow', False),
                        'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                    }
                    cfg_hash = hashlib.md5(repr(cfg_key).encode()).hexdigest()[:8]

                    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
                    
                    # Check for user-provided features file first
                    features_path = self.pred_features_path.get()
                    X = None
                    features_loaded = False

                    if features_path and os.path.isfile(features_path):
                        progress_label.config(text="Loading pre-extracted features...")
                        self.root.update()
                        try:
                            with open(features_path, 'rb') as f:
                                features_data = pickle.load(f)
                            if isinstance(features_data, dict) and 'X' in features_data:
                                X = features_data['X']
                            else:
                                X = features_data
                            features_loaded = True
                        except Exception as e:
                            print(f"⚠️ Could not load features file: {e}")
                            features_loaded = False
                    
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
                        # Load or extract features
                        if cache_file:
                            progress_label.config(text="Loading cached features...")
                            self.root.update()
                            with open(cache_file, 'rb') as f:
                                X = pickle.load(f)
                            print(f"✓ Loaded cached features from {cache_file}")
                        else:
                            # No cache found - extract features
                            progress_label.config(text="Extracting features (this may take a while)...")
                            self.root.update()

                            # Save to project features/ folder when project is set
                            if _proj_folder and os.path.isdir(_proj_folder):
                                cache_dir = os.path.join(_proj_folder, 'features')
                            else:
                                cache_dir = os.path.join(video_dir, 'PredictionCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, _cache_fname)
                            
                            # Get config path from Predict tab (if user selected one)
                            config_yaml = self.pred_dlc_config_path.get() if self.pred_dlc_config_path.get() else None
                            
                            X = PixelPaws_ExtractFeatures(
                                pose_data_file=dlc_path,
                                video_file_path=video_path,
                                bp_include_list=clf_data.get('bp_include_list'),
                                bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                                square_size=clf_data.get('square_size', [40]),
                                pix_threshold=clf_data.get('pix_threshold', 0.3),
                                use_gpu=True,  # GPU enabled (auto-fallback)
                                config_yaml_path=config_yaml,  # Pass config for crop detection
                                include_optical_flow=clf_data.get('include_optical_flow', False),
                                bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                            )
                            
                            with open(cache_file, 'wb') as f:
                                pickle.dump(X, f)
                            
                            print(f"✓ Features extracted and cached to {cache_file}")
                    
                    # Predict
                    progress_label.config(text="Running classifier...")
                    self.root.update()
                    
                    y_proba = predict_with_xgboost(model, X)
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
                        except:
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
                            result_data['human_labels']
                        )
                        print("[Main Thread] Preview window created successfully!")
                    
                    progress.destroy()
                    
                    # Call preview directly on main thread (the original working way!)
                    print(f"Opening preview window...")
                    SideBySidePreview(self.root, video_path, y_pred, y_proba, 
                                    behavior_name, best_thresh, human_labels=human_labels)
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    
                    try:
                        progress.destroy()
                    except:
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
        """Log message to prediction results if available"""
        if hasattr(self, 'pred_results_text') and self.pred_results_text:
            try:
                self.pred_results_text.insert(tk.END, message + '\n')
                self.pred_results_text.see(tk.END)
            except:
                pass  # Fail silently if text widget not ready
    
    def apply_bout_filtering(self, y_pred, min_bout, min_after_bout, max_gap):
        """Thin wrapper — delegates to the shared _apply_bout_filtering in evaluation_tab."""
        return _apply_bout_filtering(y_pred, min_bout, min_after_bout, max_gap)
    
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
        
        threading.Thread(target=self._predict_thread, daemon=True).start()

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

            # Read overlay checkbox values
            do_skeleton = self.lv_skeleton_dots.get()
            do_tint     = self.lv_frame_tint.get()
            do_timeline = self.lv_timeline_strip.get()

            self.pred_results_text.insert(tk.END, "\nCreating labeled video...\n")

            clip_start = self._parse_time_to_frames(self.pred_clip_start.get(), fps)
            clip_end   = self._parse_time_to_frames(self.pred_clip_end.get(),   fps)
            clip_start = max(0, clip_start if clip_start is not None else 0)
            clip_end   = min(n_frames, clip_end if clip_end is not None else n_frames)
            clip_end   = max(clip_start + 1, clip_end)

            labeled_path = os.path.join(output_folder, f"{base_name}_labeled.mp4")

            cap_lv = cv2.VideoCapture(video_path)
            lv_w   = int(cap_lv.get(cv2.CAP_PROP_FRAME_WIDTH))
            lv_h   = int(cap_lv.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(labeled_path, fourcc, fps, (lv_w, lv_h))

            # Load DLC body-part coordinates for skeleton overlay
            bp_xy = {}   # bodypart -> (x_arr, y_arr, prob_arr) each shape (n_frames,)
            if do_skeleton and self._last_pred_dlc_path and os.path.exists(self._last_pred_dlc_path):
                try:
                    import pandas as _pd
                    _dlc = _pd.read_hdf(self._last_pred_dlc_path)
                    # DLC H5 has 3-level MultiIndex: (scorer, bodypart, coord)
                    _dlc.columns = _pd.MultiIndex.from_tuples(
                        [(_c[1], _c[2]) for _c in _dlc.columns])
                    for _bp in _dlc.columns.get_level_values(0).unique():
                        bp_xy[_bp] = (
                            _dlc[_bp]['x'].values.astype(float),
                            _dlc[_bp]['y'].values.astype(float),
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
                    timeline_img[:, _x] = (0, 0, 180) if y_pred[_idx] == 1 else (0, 140, 0)
                # Dim slightly so it doesn't overpower
                cv2.addWeighted(np.full((14, lv_w, 3), 20, dtype=np.uint8), 0.3,
                                timeline_img, 0.7, 0, timeline_img)

            cap_lv.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            for fi in range(clip_start, clip_end):
                ret, frame = cap_lv.read()
                if not ret:
                    break

                prob  = float(y_proba[fi]) if fi < len(y_proba) else 0.0
                pred  = int(y_pred[fi])    if fi < len(y_pred)  else 0
                color = (0, 0, 220) if pred == 1 else (0, 200, 0)

                # 1. Frame tint (pred==1 only, applied before HUD so HUD stays crisp)
                if do_tint and pred == 1:
                    _red = np.full_like(frame, (0, 0, 80))
                    cv2.addWeighted(_red, 0.22, frame, 0.78, 0, frame)

                # 2. HUD background + text + confidence bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (lv_w, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                ts   = fi / fps
                tstr = f"{int(ts // 3600):01d}:{int((ts % 3600) // 60):02d}:{ts % 60:05.2f}"
                cv2.putText(frame, f"Frame {fi}  [{tstr}]",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                cv2.putText(frame,
                            f"{behavior_name}: {'YES' if pred else 'NO'}   p = {prob:.3f}",
                            (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                bar_w = int(lv_w * prob)
                cv2.rectangle(frame, (0, 71), (bar_w, 79), color, -1)
                cv2.rectangle(frame, (0, 71), (lv_w - 1, 79), (80, 80, 80), 1)

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

                # 4. Red halo border (pred==1 only)
                if pred == 1:
                    cv2.rectangle(frame, (0, 0), (lv_w - 1, lv_h - 1), (0, 0, 220), 18)
                    cv2.rectangle(frame, (9, 9), (lv_w - 10, lv_h - 10), (40, 40, 180), 6)

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
                    self.pred_results_text.insert(
                        tk.END, f"  Writing frame {done} / {total_clip}\n")
                    self.pred_results_text.see(tk.END)

            cap_lv.release()
            writer.release()
            self.pred_results_text.insert(tk.END, f"✓ Labeled video: {labeled_path}\n")
            messagebox.showinfo("Done", f"Labeled video saved:\n{labeled_path}")

        except Exception as e:
            import traceback
            self.pred_results_text.insert(
                tk.END, f"\n✗ Export failed: {traceback.format_exc()}\n")
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    def _predict_thread(self):
        """Prediction thread with feature caching and crop handling"""
        try:
            self.pred_results_text.delete('1.0', tk.END)
            self.pred_results_text.insert(tk.END, "=" * 60 + "\n")
            self.pred_results_text.insert(tk.END, "PixelPaws Prediction\n")
            self.pred_results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            clf_path = self.pred_classifier_path.get()
            video_path = self.pred_video_path.get()
            dlc_path = self.pred_dlc_path.get()
            features_path = self.pred_features_path.get()
            dlc_config_path = self.pred_dlc_config_path.get()
            
            self.pred_results_text.insert(tk.END, f"Classifier: {os.path.basename(clf_path)}\n")
            self.pred_results_text.insert(tk.END, f"Video: {os.path.basename(video_path)}\n")
            self.pred_results_text.insert(tk.END, f"DLC File: {os.path.basename(dlc_path)}\n")
            if features_path:
                self.pred_results_text.insert(tk.END, f"Features: {os.path.basename(features_path)}\n")
            if dlc_config_path:
                self.pred_results_text.insert(tk.END, f"DLC Config: {os.path.basename(dlc_config_path)}\n")
            self.pred_results_text.insert(tk.END, "\n")
            
            # Load classifier
            self.pred_results_text.insert(tk.END, "Loading classifier...\n")
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
            
            # Clean body parts lists (remove DLC network names for BAREfoot compatibility)
            clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))
            
            # Auto-detect bp_include_list if missing
            clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=True)
            
            model = clf_data['clf_model']
            best_thresh = clf_data['best_thresh']
            behavior_name = clf_data.get('Behavior_type', 'Behavior')
            
            self.pred_results_text.insert(tk.END, f"  Behavior: {behavior_name}\n")
            self.pred_results_text.insert(tk.END, f"  Threshold: {best_thresh:.3f}\n\n")
            
            # Check for DLC crop parameters
            crop_x_offset = 0
            crop_y_offset = 0
            if dlc_config_path and os.path.isfile(dlc_config_path):
                self.pred_results_text.insert(tk.END, "Checking DLC crop parameters...\n")
                try:
                    import yaml
                    with open(dlc_config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if config.get('cropping', False):
                        crop_x_offset = config.get('x1', 0)
                        crop_y_offset = config.get('y1', 0)
                        self.pred_results_text.insert(tk.END, 
                            f"  ✓ DLC crop detected: x+{crop_x_offset}, y+{crop_y_offset}\n")
                        self.pred_results_text.insert(tk.END, 
                            f"  Note: Features should account for crop offset\n\n")
                    else:
                        self.pred_results_text.insert(tk.END, "  No cropping in config\n\n")
                except ImportError:
                    self.pred_results_text.insert(tk.END, "  ⚠️  PyYAML not installed - cannot read config\n")
                    self.pred_results_text.insert(tk.END, "     Install with: pip install pyyaml\n\n")
                except Exception as e:
                    self.pred_results_text.insert(tk.END, f"  ⚠️  Could not read config: {e}\n\n")
            
            # Try to load pre-extracted features first
            X = None
            features_loaded = False

            # Get video directory (needed for cache and output)
            video_dir = os.path.dirname(video_path)

            if features_path and os.path.isfile(features_path):
                self.pred_results_text.insert(tk.END, "Loading pre-extracted features...\n")
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
                    self.pred_results_text.insert(
                        tk.END, f"  ✓ Loaded features: {X.shape[0]} frames, {X.shape[1]} features\n")

                    if crop_x_offset != 0 or crop_y_offset != 0:
                        self.pred_results_text.insert(
                            tk.END,
                            f"  ⚠️  Pre-extracted features used with crop offset detected "
                            f"(x+{crop_x_offset}, y+{crop_y_offset}).\n"
                            f"     Ensure features were extracted with crop-corrected coordinates.\n")

                except Exception as e:
                    self.pred_results_text.insert(
                        tk.END, f"  ✗ Could not load features file: {e}\n"
                                f"  Falling back to feature extraction...\n\n")
                    features_loaded = False
            elif features_path:
                self.pred_results_text.insert(
                    tk.END, f"  ⚠️  Features file not found: {features_path}\n"
                            f"  Falling back to feature extraction...\n\n")
            
            # Extract features if not loaded
            if not features_loaded:
                self.pred_results_text.insert(tk.END, "Proceeding with feature extraction...\n")
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

                # Try to load cached features
                if os.path.isfile(cache_file):
                    self.pred_results_text.insert(tk.END, "Loading cached features...\n")
                    with open(cache_file, 'rb') as f:
                        X = pickle.load(f)
                    self.pred_results_text.insert(tk.END, f"  ✓ Loaded from cache: {cache_file}\n\n")
                else:
                    # Extract features
                    self.pred_results_text.insert(tk.END, "Extracting features...\n")
                    self.pred_results_text.insert(tk.END, "  (This may take several minutes for long videos)\n")

                    if crop_x_offset != 0 or crop_y_offset != 0:
                        self.pred_results_text.insert(tk.END,
                            f"  Applying crop offset: x+{crop_x_offset}, y+{crop_y_offset}\n")

                    X = PixelPaws_ExtractFeatures(
                        pose_data_file=dlc_path,
                        video_file_path=video_path,
                        bp_include_list=clf_data.get('bp_include_list'),
                        bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                        square_size=clf_data.get('square_size', [40]),
                        pix_threshold=clf_data.get('pix_threshold', 0.3),
                        use_gpu=True,  # GPU enabled (auto-fallback)
                        crop_offset_x=crop_x_offset,  # Pass detected crop offset
                        crop_offset_y=crop_y_offset,
                        config_yaml_path=dlc_config_path if dlc_config_path else None,  # Pass config for auto-detection
                        include_optical_flow=clf_data.get('include_optical_flow', False),
                        bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                    )
                    
                    # Save to cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(X, f)
                    
                    self.pred_results_text.insert(tk.END, f"  ✓ Features extracted and cached\n")
                    self.pred_results_text.insert(tk.END, f"  Cache: {cache_file}\n\n")
            
            # Predict
            self.pred_results_text.insert(tk.END, "Running classifier...\n")
            y_proba = predict_with_xgboost(model, X)
            y_pred = (y_proba >= best_thresh).astype(int)
            
            # Apply bout filtering
            if 'min_bout' in clf_data:
                self.pred_results_text.insert(tk.END, "Applying bout filtering...\n")
                y_pred_filtered = self.apply_bout_filtering(
                    y_pred,
                    clf_data.get('min_bout', 1),
                    clf_data.get('min_after_bout', 1),
                    clf_data.get('max_gap', 0)
                )
                
                raw_positive = np.sum(y_pred)
                filtered_positive = np.sum(y_pred_filtered)
                self.pred_results_text.insert(tk.END, f"  Raw predictions: {raw_positive} frames\n")
                self.pred_results_text.insert(tk.END, f"  After filtering: {filtered_positive} frames\n\n")
                
                y_pred = y_pred_filtered
            
            # Calculate statistics
            n_frames = len(y_pred)
            n_positive = np.sum(y_pred)
            pct_positive = (n_positive / n_frames) * 100
            
            # Get FPS
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            behavior_time = n_positive / fps

            # Count bouts using shared helper
            bout_stats = count_bouts(y_pred, fps)
            n_bouts = bout_stats['n_bouts']
            bouts    = bout_stats['bouts']
            
            # Results
            self.pred_results_text.insert(tk.END, "=" * 60 + "\n")
            self.pred_results_text.insert(tk.END, "RESULTS\n")
            self.pred_results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            self.pred_results_text.insert(tk.END, f"Total frames: {n_frames}\n")
            self.pred_results_text.insert(tk.END, f"Behavior detected: {n_positive} frames ({pct_positive:.1f}%)\n")
            self.pred_results_text.insert(tk.END, f"Behavior time: {behavior_time:.1f} seconds ({behavior_time/60:.1f} minutes)\n")
            self.pred_results_text.insert(tk.END, f"Number of bouts: {n_bouts}\n")

            if bouts:
                self.pred_results_text.insert(tk.END, f"Mean bout duration:   {bout_stats['mean_dur_sec']:.2f} seconds\n")
                self.pred_results_text.insert(tk.END, f"Median bout duration: {bout_stats['median_dur_sec']:.2f} seconds\n")
                self.pred_results_text.insert(tk.END, f"Min bout duration:    {bout_stats['min_dur_sec']:.2f} seconds\n")
                self.pred_results_text.insert(tk.END, f"Max bout duration:    {bout_stats['max_dur_sec']:.2f} seconds\n")
            
            # Save outputs
            output_folder = self.pred_output_folder.get()
            if not output_folder:
                output_folder = video_dir
            
            # Get video base name
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]
            
            self.pred_results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.pred_results_text.insert(tk.END, "SAVING OUTPUTS\n")
            self.pred_results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            if self.pred_save_csv.get():
                csv_path = os.path.join(output_folder, f"{base_name}_predictions.csv")
                df = pd.DataFrame({
                    'frame':       np.arange(len(y_pred)),
                    'probability': y_proba,
                    behavior_name: y_pred,
                })
                df.to_csv(csv_path, index=False)
                self.pred_results_text.insert(tk.END, f"✓ Predictions CSV: {csv_path}\n")
            
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
                        f.write(f"Mean bout duration: {np.mean(bout_durations):.2f} seconds\n")
                
                self.pred_results_text.insert(tk.END, f"✓ Summary: {summary_path}\n")
            
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
            self.pred_export_video_btn.config(state='normal')

            if self.pred_generate_ethogram.get():
                self.pred_results_text.insert(tk.END, "✓ Ethogram plots: (feature in development)\n")

            self.pred_results_text.insert(tk.END, "\n✓ Prediction complete!\n")
            
            messagebox.showinfo("Complete", "Prediction completed successfully!")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.pred_results_text.insert(tk.END, f"\n\n{'=' * 60}\n")
            self.pred_results_text.insert(tk.END, "✗ ERROR\n")
            self.pred_results_text.insert(tk.END, f"{'=' * 60}\n\n")
            self.pred_results_text.insert(tk.END, f"{error_detail}\n")
            messagebox.showerror("Error", f"Prediction failed:\n\n{str(e)}")
    
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
        self.batch_log.insert(tk.END, "Checking feature extraction status...\n\n")
        self.batch_log.insert(tk.END, "SMART DEFAULT SYSTEM:\n")
        self.batch_log.insert(tk.END, f"• Brightness bodyparts: {', '.join(DEFAULT_BRIGHTNESS_BODYPARTS)}\n")
        self.batch_log.insert(tk.END, f"• All pose features extracted\n")
        self.batch_log.insert(tk.END, f"• Re-extracts only if classifier needs different bodyparts\n\n")
        
        folder = self.batch_folder.get()
        ext = self.batch_video_ext.get()
        videos = glob.glob(os.path.join(folder, f"*{ext}"))
        
        if not videos:
            self.batch_log.insert(tk.END, f"✗ No videos found with extension {ext}\n")
            return
        
        self.batch_log.insert(tk.END, f"Found {len(videos)} videos\n")
        self.batch_log.insert(tk.END, f"Checking for {len(self.batch_classifiers)} classifier(s)\n\n")
        
        total_checks = len(videos) * len(self.batch_classifiers)
        ready_count = 0
        needs_extraction_count = 0
        needs_reextraction_count = 0
        
        for video_path in videos:
            video_name = os.path.basename(video_path)
            video_dir = os.path.dirname(video_path)
            video_base = os.path.splitext(video_name)[0]
            
            self.batch_log.insert(tk.END, f"📹 {video_name}:\n")
            
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
                    feature_cache_dir = os.path.join(video_dir, 'FeatureCache')
                    prediction_cache_dir = os.path.join(video_dir, 'PredictionCache')
                    
                    smart_cache_locations = [
                        os.path.join(feature_cache_dir, f"{video_base}_features_smart_{smart_hash}.pkl"),
                        os.path.join(prediction_cache_dir, f"{video_base}_features_smart_{smart_hash}.pkl"),
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
                        # Scan FeatureCache for any matching video
                        if os.path.isdir(feature_cache_dir):
                            pattern = os.path.join(feature_cache_dir, f"{video_base}_features_*.pkl")
                            matches = glob.glob(pattern)
                            if matches:
                                # Found old cache - need to check if it's compatible
                                cache_file = matches[0]
                                cache_type = "old"
                        
                        if not cache_file and os.path.isdir(prediction_cache_dir):
                            pattern = os.path.join(prediction_cache_dir, f"{video_base}_features_*.pkl")
                            matches = glob.glob(pattern)
                            if matches:
                                cache_file = matches[0]
                                cache_type = "old"
                    
                    if cache_file and cache_type == "smart" and not needs_different_bp:
                        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                        filename = os.path.basename(cache_file)
                        self.batch_log.insert(tk.END, f"  ✓ {clf_name}: Ready\n")
                        self.batch_log.insert(tk.END, f"     {filename} ({size_mb:.1f} MB)\n")
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
                                        self.batch_log.insert(tk.END, f"  ✓ {clf_name}: Compatible cache found\n")
                                        self.batch_log.insert(tk.END, f"     {filename} ({size_mb:.1f} MB)\n")
                                        self.batch_log.insert(tk.END, f"     Cache: {n_cached} features | Model needs: {n_expected}\n")
                                        ready_count += 1
                                    else:
                                        # Missing some features
                                        self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Cache missing {len(missing_features)} features\n")
                                        self.batch_log.insert(tk.END, f"     Will re-extract with all required bodyparts\n")
                                        needs_reextraction_count += 1
                                else:
                                    # Can't verify compatibility
                                    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                                    self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Old cache found (can't verify)\n")
                                    self.batch_log.insert(tk.END, f"     {n_cached} features ({size_mb:.1f} MB)\n")
                                    self.batch_log.insert(tk.END, f"     Will attempt to use, may re-extract if incompatible\n")
                                    ready_count += 1
                            else:
                                # Not a DataFrame, can't check
                                self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Unknown cache format\n")
                                needs_extraction_count += 1
                        except Exception as e:
                            self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Error reading cache\n")
                            self.batch_log.insert(tk.END, f"     {str(e)}\n")
                            needs_extraction_count += 1
                    
                    elif cache_file and needs_different_bp:
                        extra_bp = clf_bp_pixbrt - smart_bp_pixbrt_set
                        self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Needs re-extraction\n")
                        self.batch_log.insert(tk.END, f"     Classifier needs additional bodyparts: {', '.join(extra_bp)}\n")
                        needs_reextraction_count += 1
                    else:
                        self.batch_log.insert(tk.END, f"  ✗ {clf_name}: Not cached\n")
                        if needs_different_bp:
                            extra_bp = clf_bp_pixbrt - smart_bp_pixbrt_set
                            self.batch_log.insert(tk.END, f"     Will extract with extra bodyparts: {', '.join(extra_bp)}\n")
                        else:
                            self.batch_log.insert(tk.END, f"     Will extract with smart defaults\n")
                        needs_extraction_count += 1
                
                except Exception as e:
                    import traceback
                    self.batch_log.insert(tk.END, f"  ⚠ {clf_name}: Error\n")
                    self.batch_log.insert(tk.END, f"     {str(e)}\n")
                    needs_extraction_count += 1
            
            self.batch_log.insert(tk.END, "\n")
            self.root.update_idletasks()
        
        # Summary
        self.batch_log.insert(tk.END, f"\n{'='*60}\n")
        self.batch_log.insert(tk.END, f"SUMMARY:\n")
        self.batch_log.insert(tk.END, f"{'='*60}\n")
        self.batch_log.insert(tk.END, f"Total: {total_checks}\n")
        self.batch_log.insert(tk.END, f"✓ Ready: {ready_count} ({ready_count/total_checks*100:.1f}%)\n")
        self.batch_log.insert(tk.END, f"⚠ Re-extract (extra bodyparts): {needs_reextraction_count} ({needs_reextraction_count/total_checks*100:.1f}%)\n")
        self.batch_log.insert(tk.END, f"✗ Extract (first time): {needs_extraction_count} ({needs_extraction_count/total_checks*100:.1f}%)\n\n")
        
        if ready_count == total_checks:
            self.batch_log.insert(tk.END, f"🚀 All ready! Batch will be very fast.\n")
        elif ready_count > 0:
            self.batch_log.insert(tk.END, f"⚡ {ready_count} ready, {needs_reextraction_count + needs_extraction_count} will extract.\n")
        else:
            self.batch_log.insert(tk.END, f"⏱ First run - will extract and cache.\n")
            self.batch_log.insert(tk.END, f"   Subsequent runs will be much faster!\n")
        
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
            except:
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
        except:
            clf_data = {}
        
        # Create settings dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Settings - {item}")
        dialog.geometry("500x400")
        
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
        """Add all .pkl classifiers from the project classifiers/ folder to the batch list."""
        clf_dir = os.path.join(self.current_project_folder.get(), 'classifiers')
        if not os.path.isdir(clf_dir):
            messagebox.showinfo("Not Found", "No classifiers/ folder in project.")
            return
        added = 0
        for f in sorted(os.listdir(clf_dir)):
            if f.endswith('.pkl'):
                path = os.path.join(clf_dir, f)
                if path not in self.batch_classifiers:
                    self.batch_classifiers[path] = {'min_bout_sec': 0.2, 'bin_size_sec': 60}
                    self.batch_clf_listbox.insert(tk.END, f)
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
            
            videos = glob.glob(os.path.join(folder, f"*{ext}"))
            
            if not videos:
                messagebox.showwarning("No Videos", f"No {ext} videos found in folder.")
                return
            
            mapping = []
            for video_path in videos:
                dlc = self.find_dlc_for_video(video_path, folder, prefer_filtered)
                mapping.append((video_path, dlc))
            
            # Show preview window
            preview = tk.Toplevel(self.root)
            preview.title("Video ↔ DLC Mapping Preview")
            preview.geometry("900x500")
            
            text = scrolledtext.ScrolledText(preview, width=100, height=25, wrap=tk.WORD)
            text.pack(fill='both', expand=True, padx=5, pady=5)
            
            text.insert(tk.END, f"Found {len(videos)} video(s) in {folder}\n\n")
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
        threading.Thread(target=self._batch_analysis_thread, daemon=True).start()
    
    def _batch_analysis_thread(self):
        """Batch analysis thread - FULLY IMPLEMENTED"""
        try:
            self.batch_log.delete('1.0', tk.END)
            self.batch_log.insert(tk.END, "Starting batch analysis...\n\n")
            
            folder = self.batch_folder.get()
            ext = self.batch_video_ext.get()
            videos = glob.glob(os.path.join(folder, f"*{ext}"))
            
            if not videos:
                self.batch_log.insert(tk.END, f"✗ No videos found with extension {ext}\n")
                messagebox.showerror("No Videos", f"No videos found with extension {ext} in folder:\n{folder}")
                return
            
            if not self.batch_classifiers:
                self.batch_log.insert(tk.END, "✗ No classifiers added\n")
                messagebox.showerror("No Classifiers", "Please add at least one classifier.")
                return
            
            self.batch_log.insert(tk.END, f"Found {len(videos)} videos\n")
            self.batch_log.insert(tk.END, f"Using {len(self.batch_classifiers)} classifier(s)\n\n")
            
            # Get output options
            save_labels = self.batch_save_labels.get()
            save_timebins = self.batch_save_timebins.get()
            generate_ethograms = self.batch_generate_ethograms.get()
            bin_size = self.batch_bin_size.get()
            
            total_operations = len(videos) * len(self.batch_classifiers)
            current_operation = 0
            
            # Summary results for final report
            summary_results = []
            
            for video_path in videos:
                video_name = os.path.basename(video_path)
                video_dir = os.path.dirname(video_path)
                video_base = os.path.splitext(video_name)[0]
                
                self.batch_log.insert(tk.END, f"\n{'='*60}\n")
                self.batch_log.insert(tk.END, f"Processing: {video_name}\n")
                self.batch_log.insert(tk.END, f"{'='*60}\n")
                self.root.update_idletasks()
                
                # Find DLC file
                dlc_path = self.find_dlc_for_video(video_path, folder, self.batch_prefer_filtered.get())
                
                if not dlc_path:
                    self.batch_log.insert(tk.END, f"  ✗ No DLC file found - skipping\n")
                    current_operation += len(self.batch_classifiers)
                    progress = (current_operation / total_operations) * 100
                    self.batch_progress['value'] = progress
                    self.batch_progress_label.config(
                        text=f"Processing {current_operation}/{total_operations} ({progress:.1f}%)")
                    continue
                
                self.batch_log.insert(tk.END, f"  DLC: {os.path.basename(dlc_path)}\n\n")
                
                for clf_path, settings in self.batch_classifiers.items():
                    clf_name = os.path.basename(clf_path)
                    clf_base = os.path.splitext(clf_name)[0]
                    
                    self.batch_log.insert(tk.END, f"  → Running {clf_name}...\n")
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
                        results_folder = os.path.join(video_dir, "Results", behavior_name)
                        os.makedirs(results_folder, exist_ok=True)
                        
                        # Determine parameters to use
                        use_override = settings.get('use_override', False)
                        if use_override:
                            best_thresh = settings.get('threshold', clf_data.get('best_thresh', 0.5))
                            min_bout = settings.get('min_bout', clf_data.get('min_bout', 1))
                            min_after = settings.get('min_after_bout', clf_data.get('min_after_bout', 1))
                            max_gap = settings.get('max_gap', clf_data.get('max_gap', 0))
                            self.batch_log.insert(tk.END, f"     Using custom parameters\n")
                        else:
                            best_thresh = clf_data.get('best_thresh', 0.5)
                            min_bout = clf_data.get('min_bout', 1)
                            min_after = clf_data.get('min_after_bout', 1)
                            max_gap = clf_data.get('max_gap', 0)
                            self.batch_log.insert(tk.END, f"     Using classifier defaults\n")
                        
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
                        cache_locations = [
                            os.path.join(video_dir, 'FeatureCache', f"{video_base}_features_smart_{smart_hash}.pkl"),
                            os.path.join(video_dir, 'PredictionCache', f"{video_base}_features_smart_{smart_hash}.pkl"),
                        ]
                        
                        # Also check old classifier-specific cache as fallback
                        clf_cfg_key = {
                            'bp_include_list': clf_data.get('bp_include_list'),
                            'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                            'square_size': clf_data.get('square_size', [40]),
                            'pix_threshold': clf_data.get('pix_threshold', 0.3),
                            'pose_feature_version': POSE_FEATURE_VERSION,
                            'include_optical_flow': clf_data.get('include_optical_flow', False),
                            'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                        }
                        clf_hash = hashlib.md5(repr(clf_cfg_key).encode()).hexdigest()[:8]
                        cache_locations.extend([
                            os.path.join(video_dir, 'PredictionCache', f"{video_base}_features_{clf_hash}.pkl"),
                            os.path.join(video_dir, 'FeatureCache', f"{video_base}_features_{clf_hash}.pkl"),
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
                            for cache_dir_name in ['FeatureCache', 'PredictionCache']:
                                cache_dir_path = os.path.join(video_dir, cache_dir_name)
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
                                                    self.batch_log.insert(tk.END, f"     ✓ Found compatible cache from previous extraction\n")
                                                    self.batch_log.insert(tk.END, f"        Cache: {len(available_features)} features | Model needs: {len(required_features)}\n")
                                                    break
                                        except:
                                            pass  # Try next cache
                        
                        # Check if classifier needs different brightness bodyparts
                        clf_bp_pixbrt = set(clf_data.get('bp_pixbrt_list', []))
                        smart_bp_pixbrt_set = set(smart_bp_pixbrt)
                        
                        # Smart compatibility check:
                        # Cached features are usable if they are a SUPERSET of what classifier needs
                        # Example: Cache has [hrpaw, hlpaw, snout, tail] → Can be used by classifier needing [hrpaw, hlpaw, snout]
                        cached_is_superset = clf_bp_pixbrt.issubset(smart_bp_pixbrt_set)
                        
                        # Load or extract features
                        if cache_file and cache_is_compatible:
                            self.batch_log.insert(tk.END, f"     ✓ Loaded cached features\n")
                            with open(cache_file, 'rb') as f:
                                X = pickle.load(f)
                            
                            # Feature selection happens automatically in predict_with_xgboost()
                            # Model will select only the features it needs from the cache
                            model_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'
                            cache_n_features = X.shape[1] if hasattr(X, 'shape') else 'unknown'
                            
                            if model_n_features != 'unknown' and cache_n_features != 'unknown':
                                if model_n_features < cache_n_features:
                                    self.batch_log.insert(tk.END, f"     ℹ Model needs {model_n_features} features, cache has {cache_n_features}\n")
                                    self.batch_log.insert(tk.END, f"     ℹ Will auto-select required features during prediction\n")
                        
                        elif not cache_is_compatible:
                            self.batch_log.insert(tk.END, f"     Extracting features (no compatible cache found)...\n")
                            self.root.update_idletasks()
                            
                            # Get config path for crop detection
                            config_yaml = self.batch_dlc_config.get() if self.batch_dlc_config.get() else None
                            
                            X = PixelPaws_ExtractFeatures(
                                pose_data_file=dlc_path,
                                video_file_path=video_path,
                                bp_include_list=clf_data.get('bp_include_list'),
                                bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                                square_size=clf_data.get('square_size', [40]),
                                pix_threshold=clf_data.get('pix_threshold', 0.3),
                                use_gpu=True,
                                config_yaml_path=config_yaml,  # Pass config for crop detection
                                include_optical_flow=clf_data.get('include_optical_flow', False),
                                bp_optflow_list=clf_data.get('bp_optflow_list', []) or None,
                            )

                            # Save with classifier-specific hash
                            cache_dir = os.path.join(video_dir, 'FeatureCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, f"{video_base}_features_{clf_hash}.pkl")
                            with open(cache_file, 'wb') as f:
                                pickle.dump(X, f)
                            self.batch_log.insert(tk.END, f"     ✓ Features cached (classifier-specific)\n")
                        else:
                            self.batch_log.insert(tk.END, f"     Extracting features (smart defaults)...\n")
                            self.batch_log.insert(tk.END, f"     Brightness bodyparts: {', '.join(smart_bp_pixbrt)}\n")
                            self.root.update_idletasks()
                            
                            # Get config path for crop detection
                            config_yaml = self.batch_dlc_config.get() if self.batch_dlc_config.get() else None
                            
                            X = PixelPaws_ExtractFeatures(
                                pose_data_file=dlc_path,
                                video_file_path=video_path,
                                bp_include_list=clf_data.get('bp_include_list'),  # All pose features
                                bp_pixbrt_list=smart_bp_pixbrt,  # Smart default brightness
                                square_size=smart_square_size,
                                pix_threshold=smart_pix_threshold,
                                use_gpu=True,
                                config_yaml_path=config_yaml,  # Pass config for crop detection
                            )
                            
                            # Save with smart hash
                            cache_dir = os.path.join(video_dir, 'FeatureCache')
                            os.makedirs(cache_dir, exist_ok=True)
                            cache_file = os.path.join(cache_dir, f"{video_base}_features_smart_{smart_hash}.pkl")
                            with open(cache_file, 'wb') as f:
                                pickle.dump(X, f)
                            self.batch_log.insert(tk.END, f"     ✓ Features cached (reusable for most classifiers)\n")
                        
                        # Predict
                        self.batch_log.insert(tk.END, f"     Running prediction...\n")
                        self.root.update_idletasks()
                        
                        y_proba = predict_with_xgboost(model, X)
                        y_pred = (y_proba >= best_thresh).astype(int)

                        # Apply filtering
                        y_pred_filtered = self.apply_bout_filtering(
                            y_pred, min_bout, min_after, max_gap
                        )

                        # Calculate statistics
                        n_frames = len(y_pred_filtered)
                        n_behavior_frames = np.sum(y_pred_filtered)
                        percent_behavior = (n_behavior_frames / n_frames) * 100 if n_frames > 0 else 0

                        # Get FPS (needed for bout durations and timebins)
                        import cv2 as _cv2
                        _cap = _cv2.VideoCapture(video_path)
                        fps = _cap.get(_cv2.CAP_PROP_FPS)
                        _cap.release()

                        # Count bouts using shared helper
                        bout_stats = count_bouts(y_pred_filtered, fps)
                        n_bouts           = bout_stats['n_bouts']
                        bouts             = bout_stats['bouts']
                        mean_bout_duration = bout_stats['mean_dur_sec']

                        self.batch_log.insert(tk.END, f"     ✓ Found {n_bouts} bouts ({percent_behavior:.1f}% of frames)\n")

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
                            self.batch_log.insert(tk.END, f"     ✓ Saved predictions CSV\n")

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
                                self.batch_log.insert(tk.END, f"     ✓ Saved bouts CSV ({n_bouts} bouts)\n")
                        
                        if save_timebins:
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
                                    'percent_behavior': (np.sum(bin_predictions) / len(bin_predictions)) * 100
                                })
                            
                            timebin_csv = os.path.join(results_folder, f"{video_base}_{clf_base}_timebins.csv")
                            timebin_df = pd.DataFrame(timebin_data)
                            timebin_df.to_csv(timebin_csv, index=False)
                            self.batch_log.insert(tk.END, f"     ✓ Saved time-binned summary\n")
                        
                        if generate_ethograms:
                            try:
                                import matplotlib.pyplot as plt
                                
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                                
                                # Ethogram
                                ax1.fill_between(np.arange(n_frames), 0, y_pred_filtered, 
                                                color='red', alpha=0.6, label=behavior_name)
                                ax1.set_ylabel('Behavior')
                                ax1.set_ylim(-0.1, 1.1)
                                ax1.set_title(f'Ethogram: {video_name} - {behavior_name}')
                                ax1.legend()
                                ax1.grid(True, alpha=0.3)
                                
                                # Probability
                                ax2.plot(np.arange(n_frames), y_proba, color='blue', alpha=0.7, label='Probability')
                                ax2.axhline(y=best_thresh, color='red', linestyle='--', label=f'Threshold ({best_thresh:.2f})')
                                ax2.set_xlabel('Frame')
                                ax2.set_ylabel('Probability')
                                ax2.set_ylim(-0.05, 1.05)
                                ax2.legend()
                                ax2.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                
                                ethogram_png = os.path.join(results_folder, f"{video_base}_{clf_base}_ethogram.png")
                                plt.savefig(ethogram_png, dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                self.batch_log.insert(tk.END, f"     ✓ Generated ethogram plot\n")
                            except Exception as e:
                                self.batch_log.insert(tk.END, f"     ⚠ Ethogram failed: {e}\n")
                        
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
                        
                        self.batch_log.insert(tk.END, f"     ✓ Complete!\n")
                        
                    except Exception as e:
                        self.batch_log.insert(tk.END, f"     ✗ Error: {str(e)}\n")
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
                
                self.batch_log.insert(tk.END, f"\n\n{'='*60}\n")
                self.batch_log.insert(tk.END, f"✓ Batch analysis complete!\n")
                self.batch_log.insert(tk.END, f"{'='*60}\n")
                self.batch_log.insert(tk.END, f"\nResults folder structure:\n")
                self.batch_log.insert(tk.END, f"  Results/\n")
                for behavior in summary_df['behavior'].unique():
                    self.batch_log.insert(tk.END, f"    {behavior}/   ← per-video outputs + summary\n")
                self.batch_log.insert(tk.END, f"\nCombined summary: {combined_csv}\n")
                self.batch_log.insert(tk.END, f"\nProcessed:\n")
                self.batch_log.insert(tk.END, f"  - {len(videos)} videos\n")
                self.batch_log.insert(tk.END, f"  - {len(self.batch_classifiers)} classifiers\n")
                self.batch_log.insert(tk.END, f"  - {len(summary_results)} successful predictions\n")
            
            self.batch_progress_label.config(text="Complete!")
            
            # Build completion message
            behaviors_run = list({r['behavior'] for r in summary_results}) if summary_results else []
            behavior_list = '\n'.join(f'    Results/{b}/' for b in sorted(behaviors_run))
            completion_msg = (
                f"Batch analysis completed!\n\n"
                f"Processed {len(videos)} video(s) with {len(self.batch_classifiers)} classifier(s).\n\n"
                f"Output folders:\n{behavior_list}\n\n"
                f"Combined summary:\n    Results/PixelPaws_Batch_Summary.csv"
            )
            
            messagebox.showinfo("Complete", completion_msg)
            
        except Exception as e:
            self.batch_log.insert(tk.END, f"\n\n✗ Error: {str(e)}\n")
            messagebox.showerror("Error", f"Batch analysis failed:\n{str(e)}")
            import traceback
            traceback.print_exc()



    # === ACTIVE LEARNING TAB METHODS ===
    
    def al_batch_mode(self):
        """Run Active Learning in batch mode on multiple sessions"""
        folder = filedialog.askdirectory(title="Select Project Folder for Batch Active Learning")
        if not folder:
            return
        
        self.al_log_message("="*60)
        self.al_log_message("🗂️ BATCH MODE - Active Learning")
        self.al_log_message("="*60)
        self.al_log_message(f"Project folder: {folder}\n")
        
        try:
            # Determine folder structure
            has_subfolders = os.path.isdir(os.path.join(folder, 'Videos'))
            if has_subfolders:
                video_folder = os.path.join(folder, 'Videos')
                target_folder = os.path.join(folder, 'Targets')
            else:
                video_folder = target_folder = folder
            feature_cache_folder = os.path.join(folder, 'features')
            
            # Find DLC files
            dlc_files = glob.glob(os.path.join(video_folder, '*.h5'))
            if not dlc_files:
                messagebox.showerror("Error", "No DLC files (.h5) found")
                return
            
            self.al_log_message(f"Found {len(dlc_files)} session(s)\n")
            
            # Get settings
            settings = self._get_batch_al_settings(len(dlc_files))
            if not settings:
                return
            
            n_suggestions, n_iterations, target_mode = settings
            
            # PHASE 1: Scan all sessions and prepare
            self.al_log_message("="*60)
            self.al_log_message("PHASE 1: SCANNING SESSIONS")
            self.al_log_message("="*60)
            
            sessions_to_process = []
            
            for dlc_path in dlc_files:
                base_name = self.get_session_base(os.path.basename(dlc_path))
                
                self.al_log_message(f"\nScanning: {base_name}")
                
                # Find files
                video_path = self._find_video_file(video_folder, base_name)
                if not video_path:
                    self.al_log_message("  ✗ Video not found, skipping")
                    continue
                
                labels_csv = self._find_labels_file(target_folder, base_name)
                if not labels_csv:
                    self.al_log_message("  ✗ Labels CSV not found, skipping")
                    continue
                
                features_cache = self._find_features_file(feature_cache_folder, base_name)
                
                sessions_to_process.append({
                    'base_name': base_name,
                    'video_path': video_path,
                    'dlc_path': dlc_path,
                    'labels_csv': labels_csv,
                    'features_cache': features_cache
                })
                
                if features_cache:
                    self.al_log_message("  ✓ Features found")
                else:
                    self.al_log_message("  ⚠ Features need extraction")
            
            if not sessions_to_process:
                messagebox.showerror("Error", "No valid sessions found")
                return
            
            self.al_log_message(f"\n✓ Found {len(sessions_to_process)} valid sessions")
            
            # PHASE 2: Extract features for all sessions that need it
            sessions_needing_features = [s for s in sessions_to_process if not s['features_cache']]
            
            if sessions_needing_features:
                self.al_log_message("\n" + "="*60)
                self.al_log_message("PHASE 2: FEATURE EXTRACTION")
                self.al_log_message("="*60)
                self.al_log_message(f"Extracting features for {len(sessions_needing_features)} session(s)\n")
                
                for i, session in enumerate(sessions_needing_features, 1):
                    self.al_log_message(f"[{i}/{len(sessions_needing_features)}] {session['base_name']}")
                    
                    features_cache = self._extract_features_for_session(
                        session['base_name'],
                        session['video_path'],
                        session['dlc_path'],
                        feature_cache_folder
                    )
                    
                    if features_cache:
                        session['features_cache'] = features_cache
                    else:
                        self.al_log_message("  ✗ Failed - will skip this session")
                
                # Remove sessions that failed feature extraction
                sessions_to_process = [s for s in sessions_to_process if s['features_cache']]
                self.al_log_message(f"\n✓ {len(sessions_to_process)} sessions ready for Active Learning")
            
            # PHASE 3: Run Active Learning on all sessions
            self.al_log_message("\n" + "="*60)
            self.al_log_message("PHASE 3: ACTIVE LEARNING")
            self.al_log_message("="*60)
            
            successful = 0
            failed = 0
            
            for session in sessions_to_process:
                self.al_log_message(f"\n{'='*60}")
                self.al_log_message(f"Labeling: {session['base_name']}")
                self.al_log_message(f"{'='*60}")
                
                try:
                    stats = active_learning.run_active_learning(
                        labels_csv=session['labels_csv'],
                        video_path=session['video_path'],
                        dlc_path=session['dlc_path'],
                        features_cache=session['features_cache'],
                        model_path=None,
                        n_suggestions=n_suggestions,
                        n_iterations=n_iterations,
                        extend_to_unlabeled=(target_mode == 'unlabeled')
                    )
                    
                    self.al_log_message(f"✓ Complete: {stats['frames_labeled']} frames labeled")
                    successful += 1
                    
                except Exception as e:
                    self.al_log_message(f"✗ Error: {str(e)}")
                    failed += 1
            
            # PHASE 4: Retrain classifier with updated labels
            if successful > 0:
                self.al_log_message("\n" + "="*60)
                self.al_log_message("PHASE 4: RETRAINING CLASSIFIER")
                self.al_log_message("="*60)
                
                response = messagebox.askyesno(
                    "Retrain Classifier?",
                    f"Active Learning complete!\n\n"
                    f"Successfully labeled {successful} session(s).\n\n"
                    f"Would you like to retrain the classifier now with the updated labels?\n\n"
                    f"This will improve accuracy for future predictions."
                )
                
                if response:
                    self._retrain_classifier_after_al(folder, sessions_to_process)
                else:
                    self.al_log_message("\nℹ Skipping retraining")
                    self.al_log_message("  You can retrain manually in the 'Train Classifier' tab")
            
            # Summary
            self.al_log_message(f"\n{'='*60}")
            self.al_log_message("BATCH COMPLETE")
            self.al_log_message(f"{'='*60}")
            self.al_log_message(f"Successful: {successful}")
            self.al_log_message(f"Failed: {failed}")
            
            messagebox.showinfo(
                "Batch Complete",
                f"Batch Active Learning Complete!\n\n"
                f"Successful: {successful}\n"
                f"Failed: {failed}\n\n"
                f"Check the log for details."
            )
            
        except Exception as e:
            self.al_log_message(f"\n✗ Batch error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Batch Error", str(e))
    
    def _retrain_classifier_after_al(self, folder, sessions):
        """Retrain classifier after Active Learning with updated labels"""
        try:
            import pickle
            import pandas as pd
            import numpy as np
            
            self.al_log_message("\nRetraining classifier with updated labels...")
            
            # Determine behavior from first session's labels
            first_labels = sessions[0]['labels_csv']
            behavior_name = self._extract_behavior_name(first_labels)
            
            if not behavior_name:
                behavior_name = "behavior"
            
            self.al_log_message(f"  Behavior: {behavior_name}")
            
            # Collect all label files
            label_files = [s['labels_csv'] for s in sessions]
            self.al_log_message(f"  Training sessions: {len(label_files)}")
            
            # Load features and labels for all sessions
            X_all = []
            y_all = []
            
            for session in sessions:
                with open(session['features_cache'], 'rb') as f:
                    features = pickle.load(f)
                    if isinstance(features, pd.DataFrame):
                        features = features.values
                
                labels_df = pd.read_csv(session['labels_csv'])
                labels = labels_df.iloc[:, 0].values
                
                # Ensure same length
                min_len = min(len(features), len(labels))
                X_all.append(features[:min_len])
                y_all.append(labels[:min_len])
            
            X_train = np.vstack(X_all)
            y_train = np.concatenate(y_all)
            
            self.al_log_message(f"  Total samples: {len(y_train):,}")
            self.al_log_message(f"  Positive: {(y_train == 1).sum():,}")
            self.al_log_message(f"  Negative: {(y_train == 0).sum():,}")
            
            # Train classifier
            from xgboost import XGBClassifier
            from sklearn.model_selection import cross_val_score
            
            self.al_log_message("\n  Training XGBoost classifier...")
            
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            
            clf.fit(X_train, y_train)
            
            # Cross-validation
            self.al_log_message("  Evaluating with cross-validation...")
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
            mean_f1 = cv_scores.mean()
            std_f1 = cv_scores.std()
            
            self.al_log_message(f"  ✓ Mean F1 Score: {mean_f1:.3f} ± {std_f1:.3f}")
            
            # Save classifier
            import pickle
            classifier_folder = os.path.join(folder, 'classifiers')
            os.makedirs(classifier_folder, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            classifier_path = os.path.join(
                classifier_folder,
                f"PixelPaws_{behavior_name}_AL_{timestamp}.pkl"
            )
            
            # Save with config
            classifier_data = {
                'clf_model': clf,
                'best_thresh': 0.5,
                'Behavior_type': behavior_name,
                'training_sessions': [s['base_name'] for s in sessions],
                'mean_cv_f1': float(mean_f1),
                'std_cv_f1': float(std_f1),
                'cv_f1_scores': cv_scores.tolist(),
                'trained_after_active_learning': True
            }
            
            # Add feature config if available
            if hasattr(self, 'al_model_path') and self.al_model_path.get():
                try:
                    with open(self.al_model_path.get(), 'rb') as f:
                        old_clf = pickle.load(f)
                    if isinstance(old_clf, dict):
                        for key in ['bp_pixbrt_list', 'square_size', 'pix_threshold', 'bp_include_list']:
                            if key in old_clf:
                                classifier_data[key] = old_clf[key]
                except:
                    pass
            
            with open(classifier_path, 'wb') as f:
                pickle.dump(classifier_data, f)
            
            self.al_log_message(f"\n✓ Classifier saved:")
            self.al_log_message(f"  {os.path.basename(classifier_path)}")
            self.al_log_message(f"  F1 Score: {mean_f1:.3f}")
            
            messagebox.showinfo(
                "Retraining Complete",
                f"Classifier retrained successfully!\n\n"
                f"F1 Score: {mean_f1:.3f} ± {std_f1:.3f}\n\n"
                f"Saved to:\n{os.path.basename(classifier_path)}"
            )
            
        except Exception as e:
            self.al_log_message(f"\n✗ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Retraining Error", f"Failed to retrain classifier:\n\n{str(e)}")
    
    def _extract_behavior_name(self, labels_csv):
        """Extract behavior name from labels CSV filename"""
        filename = os.path.basename(labels_csv)
        # Try to extract behavior name
        for behavior in ['Licking', 'Left_licking', 'Right_licking', 'Grooming', 'Scratching', 'Flinching']:
            if behavior in filename:
                return behavior
        return None
    
    def _get_batch_al_settings(self, n_sessions):
        """Show settings dialog for batch Active Learning"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Active Learning Settings")
        dialog.geometry("500x400")
        
        ttk.Label(dialog, text="Batch Active Learning", font=('Arial', 14, 'bold')).pack(pady=10)
        ttk.Label(dialog, text=f"Processing {n_sessions} session(s)", font=('Arial', 10)).pack(pady=5)
        
        frame = ttk.LabelFrame(dialog, text="Settings", padding=15)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Classifier selection (for feature extraction config)
        ttk.Label(frame, text="Classifier (optional):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        ttk.Label(frame, text="Select to use its feature config for extraction", 
                 font=('Arial', 9), foreground='gray').pack(anchor='w')
        
        clf_frame = ttk.Frame(frame)
        clf_frame.pack(fill='x', pady=5)
        classifier_var = tk.StringVar()
        ttk.Entry(clf_frame, textvariable=classifier_var, width=35).pack(side='left', padx=5)
        ttk.Button(clf_frame, text="Browse", 
                  command=lambda: classifier_var.set(
                      filedialog.askopenfilename(title="Select Classifier", 
                                                filetypes=[("Pickle", "*.pkl")])
                  )).pack(side='left')
        
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(frame, text="Target:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        target_var = tk.StringVar(value='unlabeled')
        ttk.Radiobutton(frame, text="Extend to unlabeled regions", variable=target_var, value='unlabeled').pack(anchor='w')
        ttk.Radiobutton(frame, text="Refine labeled regions", variable=target_var, value='labeled').pack(anchor='w', pady=(0, 10))
        
        f1 = ttk.Frame(frame)
        f1.pack(fill='x', pady=5)
        ttk.Label(f1, text="Frames per session:").pack(side='left')
        frames_var = tk.IntVar(value=20)
        ttk.Spinbox(f1, from_=5, to=100, textvariable=frames_var, width=10).pack(side='left', padx=5)
        
        f2 = ttk.Frame(frame)
        f2.pack(fill='x', pady=5)
        ttk.Label(f2, text="Iterations:").pack(side='left')
        iter_var = tk.IntVar(value=1)
        ttk.Spinbox(f2, from_=1, to=5, textvariable=iter_var, width=10).pack(side='left', padx=5)
        
        result = {'proceed': False}
        
        def on_start():
            result['proceed'] = True
            result['settings'] = (frames_var.get(), iter_var.get(), target_var.get())
            result['classifier'] = classifier_var.get() if classifier_var.get() else None
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Start", command=on_start).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left')
        
        dialog.wait_window()
        
        if result['proceed']:
            # Set the classifier path so extraction can use it
            if result.get('classifier'):
                if not hasattr(self, 'al_model_path'):
                    self.al_model_path = tk.StringVar()
                self.al_model_path.set(result['classifier'])
            return result['settings']
        return None
    
    def _find_video_file(self, folder, base_name):
        """Find video file for session"""
        for ext in ['.mp4', '.avi', '.MP4', '.AVI']:
            path = os.path.join(folder, base_name + ext)
            if os.path.exists(path):
                return path
        return None
    
    def _find_labels_file(self, folder, base_name):
        """Find labels CSV for session"""
        # Try exact match first
        path = os.path.join(folder, f"{base_name}.csv")
        if os.path.exists(path):
            return path
        
        # Try with behavior suffixes
        for behavior in ['Licking', 'Left_licking', 'Right_licking', 'Grooming', 'Scratching']:
            for suffix in ['_comma_perframe.csv', '_perframe.csv', '.csv']:
                path = os.path.join(folder, f"{base_name}_{behavior}{suffix}")
                if os.path.exists(path):
                    return path
        return None
    
    def _find_features_file(self, folder, base_name):
        """Find features cache for session"""
        if not os.path.exists(folder):
            return None
        for file in os.listdir(folder):
            if file.startswith(base_name) and 'features' in file and file.endswith('.pkl'):
                return os.path.join(folder, file)
        return None
    
    def _extract_features_for_session(self, base_name, video_path, dlc_path, output_folder):
        """Extract features for a single session"""
        try:
            import hashlib
            import pickle
            
            # Try to load feature config from model if available
            feature_config = None
            model_path = self.al_model_path.get().strip() if hasattr(self, 'al_model_path') and self.al_model_path.get() else None
            
            if model_path and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        clf_data = pickle.load(f)
                    
                    if isinstance(clf_data, dict):
                        feature_config = {
                            'bp_include_list': clf_data.get('bp_include_list'),
                            'bp_pixbrt_list': clf_data.get('bp_pixbrt_list', []),
                            'square_size': clf_data.get('square_size', [40]),
                            'pix_threshold': clf_data.get('pix_threshold', 0.3),
                            'pose_feature_version': POSE_FEATURE_VERSION,
                            'include_optical_flow': clf_data.get('include_optical_flow', False),
                            'bp_optflow_list': clf_data.get('bp_optflow_list', []),
                        }
                        self.al_log_message(f"  Using feature config from classifier:")
                        self.al_log_message(f"    bp_pixbrt_list: {feature_config['bp_pixbrt_list']}")
                        self.al_log_message(f"    square_size: {feature_config['square_size']}")
                        self.al_log_message(f"    pix_threshold: {feature_config['pix_threshold']}")
                except Exception as e:
                    self.al_log_message(f"  Could not load classifier config: {e}")
            
            if not feature_config or not feature_config.get('bp_pixbrt_list'):
                # Default: Extract brightness for common paw body parts
                # This ensures features are always extracted with brightness
                feature_config = {
                    'bp_include_list': None,
                    'bp_pixbrt_list': ['hrpaw', 'hlpaw', 'snout'],  # Common body parts
                    'square_size': [40, 40, 40],
                    'pix_threshold': 0.3,
                    'pose_feature_version': POSE_FEATURE_VERSION,
                    'include_optical_flow': False,
                    'bp_optflow_list': [],
                }
                self.al_log_message(f"  Using default brightness config:")
                self.al_log_message(f"    bp_pixbrt_list: {feature_config['bp_pixbrt_list']}")
                self.al_log_message(f"    ✓ Will extract brightness features")
            
            self.root.update_idletasks()
            
            # Extract features
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
            
            features = PixelPaws_ExtractFeatures(
                pose_data_file=dlc_path,
                video_file_path=video_path,
                bp_include_list=feature_config['bp_include_list'],
                bp_pixbrt_list=feature_config['bp_pixbrt_list'],
                square_size=feature_config['square_size'],
                pix_threshold=feature_config['pix_threshold'],
                use_gpu=True,
                config_yaml_path=config_yaml,  # Pass config for crop detection
                include_optical_flow=feature_config.get('include_optical_flow', False),
                bp_optflow_list=feature_config.get('bp_optflow_list', []) or None,
            )
            
            # Save to cache
            cfg_hash = hashlib.md5(repr(feature_config).encode()).hexdigest()[:8]
            os.makedirs(output_folder, exist_ok=True)
            features_cache = os.path.join(output_folder, f"{base_name}_features_{cfg_hash}.pkl")
            
            with open(features_cache, 'wb') as f:
                pickle.dump(features, f)
            
            self.al_log_message(f"  ✓ Features extracted: {features.shape}")
            return features_cache
            
        except Exception as e:
            self.al_log_message(f"  ✗ Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def al_quick_start(self):
        """Quick start - browse for labels CSV and auto-find everything else"""
        filepath = filedialog.askopenfilename(
            title="Select Per-Frame Labels CSV for Quick Start",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.al_labels_csv.set(filepath)
            self.al_log_message(f"Selected labels: {os.path.basename(filepath)}")
            self.al_log_message("")
            
            # Auto-search for related files
            self.al_auto_search_files(filepath)
            
            # Show summary
            found_count = sum([
                bool(self.al_video_path.get()),
                bool(self.al_dlc_path.get()),
                bool(self.al_features_cache.get())
            ])
            
            if found_count == 3:
                self.al_log_message("✓ All files found! Ready to start.")
                messagebox.showinfo(
                    "Quick Start Complete",
                    "All required files found automatically!\n\n"
                    "You can now click 'Start Active Learning'."
                )
            elif found_count > 0:
                self.al_log_message(f"Found {found_count}/3 files. Please browse for the missing ones.")
                messagebox.showwarning(
                    "Partial Success",
                    f"Found {found_count} out of 3 files automatically.\n\n"
                    f"Please use the Browse buttons to select the missing files."
                )
            else:
                self.al_log_message("Could not auto-find files. Please browse manually.")
                messagebox.showwarning(
                    "Auto-Search Failed",
                    "Could not automatically find the required files.\n\n"
                    "Please use the Browse buttons to select each file manually."
                )
    
    def al_browse_labels(self):
        """Browse for per-frame labels CSV"""
        filepath = filedialog.askopenfilename(
            title="Select Per-Frame Labels CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.al_labels_csv.set(filepath)
            self.al_log_message(f"Selected labels: {os.path.basename(filepath)}")
            
            # Auto-search for related files
            self.al_auto_search_files(filepath)
    
    def al_auto_search_files(self, labels_csv):
        """
        Automatically search for DLC file, video, and features cache
        based on the labels CSV filename.
        """
        try:
            # Get base name from labels CSV
            labels_dir = os.path.dirname(labels_csv)
            labels_filename = os.path.basename(labels_csv)
            
            self.al_log_message(f"\n🔍 Auto-Search Debug:")
            self.al_log_message(f"  Labels CSV: {labels_filename}")
            self.al_log_message(f"  Directory: {labels_dir}")
            
            # Parse base name
            # "251114_Formalin_S4_Licking_comma_perframe.csv" → "251114_Formalin_S4"
            base_name = labels_filename
            
            # Remove .csv extension
            if base_name.endswith('.csv'):
                base_name = base_name[:-4]
            
            self.al_log_message(f"  After removing .csv: {base_name}")
            
            # Remove _perframe
            if '_perframe' in base_name:
                base_name = base_name.replace('_perframe', '')
            
            self.al_log_message(f"  After removing _perframe: {base_name}")
            
            # Remove _comma
            if '_comma' in base_name:
                base_name = base_name.replace('_comma', '')
            
            self.al_log_message(f"  After removing _comma: {base_name}")
            
            # Remove _Labels/_labels suffix (common issue)
            labels_suffixes = ['_Labels', '_labels', '_LABELS']
            for suffix in labels_suffixes:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    self.al_log_message(f"  After removing {suffix}: {base_name}")
                    break
            
            # Remove behavior suffixes (keep checking until none found)
            suffixes = ['_Licking', '_Grooming', '_Scratching', '_Flinching', 
                       '_licking', '_grooming', '_scratching', '_flinching',
                       '_Left_licking', '_Right_licking', '_Left', '_Right']
            
            for suffix in suffixes:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    self.al_log_message(f"  After removing {suffix}: {base_name}")
                    break
            
            self.al_log_message(f"\n  Final base name: {base_name}")
            self.al_log_message("")
            
            # Search for video file
            if not self.al_video_path.get():
                self.al_log_message("  Searching for video...")
                video_found = False
                
                # Build list of directories to search
                parent_dir = os.path.dirname(labels_dir)
                search_dirs = [
                    labels_dir,  # Same folder as labels
                    parent_dir,  # Parent folder
                ]
                
                # Add sibling folders with common names
                if parent_dir:
                    for video_folder_name in ['videos', 'Videos', 'video', 'VIDEOS', 'Video']:
                        sibling_path = os.path.join(parent_dir, video_folder_name)
                        if os.path.isdir(sibling_path):
                            search_dirs.append(sibling_path)
                
                # List what's in each directory
                for search_dir in search_dirs:
                    self.al_log_message(f"  Looking in: {search_dir}")
                    try:
                        files_in_dir = os.listdir(search_dir)
                        video_files = [f for f in files_in_dir if f.endswith(('.mp4', '.avi', '.MP4', '.AVI', '.mov', '.MOV'))]
                        if video_files:
                            self.al_log_message(f"    Video files found: {video_files[:3]}...")  # Show first 3
                    except Exception as e:
                        self.al_log_message(f"    Error listing directory: {e}")
                        continue
                    
                    # Try exact match in this directory
                    for ext in ['.mp4', '.avi', '.MP4', '.AVI', '.mov', '.MOV']:
                        video_path = os.path.join(search_dir, base_name + ext)
                        if os.path.exists(video_path):
                            self.al_video_path.set(video_path)
                            self.al_log_message(f"  ✓ Found: {os.path.relpath(video_path, parent_dir)}")
                            video_found = True
                            break
                    
                    if video_found:
                        break
                
                if not video_found:
                    self.al_log_message(f"  ✗ Not found: {base_name}.mp4")
            
            # Search for DLC file
            if not self.al_dlc_path.get():
                self.al_log_message("\n  Searching for DLC file...")
                dlc_found = False
                
                # Build list of directories to search
                parent_dir = os.path.dirname(labels_dir)
                search_dirs = [
                    labels_dir,  # Same folder as labels
                    parent_dir,  # Parent folder
                ]
                
                # Add sibling folders with common names
                if parent_dir:
                    for dlc_folder_name in ['DLC', 'dlc', 'DLC_files', 'dlc_files', 'h5', 'h5_files', 'videos', 'Videos']:
                        sibling_path = os.path.join(parent_dir, dlc_folder_name)
                        if os.path.isdir(sibling_path):
                            search_dirs.append(sibling_path)
                
                # Search each directory
                for search_dir in search_dirs:
                    try:
                        h5_files = [f for f in os.listdir(search_dir) if f.endswith('.h5')]
                        if h5_files:
                            self.al_log_message(f"  Looking in: {search_dir}")
                            self.al_log_message(f"    .h5 files found: {len(h5_files)}")
                    except Exception as e:
                        continue
                    
                    # Look for .h5 files starting with base name
                    for file in h5_files:
                        # Check if starts with base name and contains DLC
                        if file.startswith(base_name) and 'DLC' in file:
                            dlc_path = os.path.join(search_dir, file)
                            self.al_dlc_path.set(dlc_path)
                            self.al_log_message(f"  ✓ Found: {os.path.relpath(dlc_path, parent_dir)}")
                            dlc_found = True
                            break
                    
                    if dlc_found:
                        break
                
                if not dlc_found:
                    self.al_log_message(f"  ✗ Not found: {base_name}DLC*.h5")
            
            # Search for features cache
            if not self.al_features_cache.get():
                self.al_log_message("\n  Searching for features cache...")
                cache_found = False
                
                # Build list of directories to search
                parent_dir = os.path.dirname(labels_dir)
                video_dir = os.path.dirname(self.al_video_path.get()) if self.al_video_path.get() else None
                
                search_dirs = []
                
                # Priority: video folder (if found), then labels folder, then parent
                if video_dir and os.path.isdir(video_dir):
                    search_dirs.append((video_dir, "video directory"))
                search_dirs.append((labels_dir, "labels directory"))
                if parent_dir and parent_dir != labels_dir:
                    search_dirs.append((parent_dir, "parent directory"))
                
                # Add common cache subdirectories
                for base_dir, _ in search_dirs[:]:
                    for cache_subdir in ['FeatureCache', 'PredictionCache', 'cache', 'Cache']:
                        cache_path = os.path.join(base_dir, cache_subdir)
                        if os.path.isdir(cache_path):
                            search_dirs.append((cache_path, f"{cache_subdir}/"))
                
                for cache_dir, desc in search_dirs:
                    if not os.path.exists(cache_dir):
                        self.al_log_message(f"  {desc}: does not exist")
                        continue
                    
                    self.al_log_message(f"  Checking {desc}...")
                    
                    try:
                        pkl_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                        features_pkls = [f for f in pkl_files if 'features' in f.lower()]
                        matching = [f for f in features_pkls if f.startswith(base_name)]
                        
                        self.al_log_message(f"    .pkl files: {len(pkl_files)}")
                        self.al_log_message(f"    with 'features': {len(features_pkls)}")
                        self.al_log_message(f"    starting with '{base_name}': {len(matching)}")
                        
                        if matching:
                            self.al_log_message(f"    Matches: {matching}")
                    except Exception as e:
                        self.al_log_message(f"    Error: {e}")
                    
                    # Look for matching files
                    for file in os.listdir(cache_dir):
                        if file.endswith('.pkl') and file.startswith(base_name) and 'features' in file.lower():
                            cache_path = os.path.join(cache_dir, file)
                            self.al_features_cache.set(cache_path)
                            self.al_log_message(f"  ✓ Found: {file} in {desc}")
                            cache_found = True
                            break
                    
                    if cache_found:
                        break
                
                if not cache_found:
                    self.al_log_message(f"  ✗ Not found: {base_name}_features_*.pkl")
            
            self.al_log_message("")
            
        except Exception as e:
            self.al_log_message(f"\n  Error during auto-search: {e}")
            import traceback
            traceback.print_exc()
    
    def al_browse_video(self):
        """Browse for video file"""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.MP4 *.AVI"), ("All files", "*.*")]
        )
        if filepath:
            self.al_video_path.set(filepath)
            self.al_log_message(f"Selected video: {os.path.basename(filepath)}")
    
    def al_browse_dlc(self):
        """Browse for DLC file"""
        filepath = filedialog.askopenfilename(
            title="Select DLC File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if filepath:
            self.al_dlc_path.set(filepath)
            self.al_log_message(f"Selected DLC: {os.path.basename(filepath)}")
    
    def al_browse_cache(self):
        """Browse for features cache"""
        filepath = filedialog.askopenfilename(
            title="Select Features Cache",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.al_features_cache.set(filepath)
            self.al_log_message(f"Selected cache: {os.path.basename(filepath)}")
    
    def al_browse_model(self):
        """Browse for model file"""
        filepath = filedialog.askopenfilename(
            title="Select Model File (Optional)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.al_model_path.set(filepath)
            self.al_log_message(f"Selected model: {os.path.basename(filepath)}")
    
    def al_log_message(self, message):
        """Add message to Active Learning log"""
        self.al_log.config(state='normal')
        self.al_log.insert(tk.END, message + '\n')
        self.al_log.see(tk.END)
        self.al_log.config(state='disabled')
        self.root.update_idletasks()
    
    def run_active_learning(self):
        """Run Active Learning session"""
        try:
            # Validate inputs
            labels_csv = self.al_labels_csv.get().strip()
            video_path = self.al_video_path.get().strip()
            dlc_path = self.al_dlc_path.get().strip()
            features_cache = self.al_features_cache.get().strip()
            
            if not labels_csv:
                messagebox.showerror("Error", "Please select a per-frame labels CSV file")
                return
            
            if not video_path:
                messagebox.showerror("Error", "Please select a video file")
                return
            
            if not dlc_path:
                messagebox.showerror("Error", "Please select a DLC file")
                return
            
            if not features_cache:
                # Offer to extract features automatically
                response = messagebox.askyesno(
                    "Features Not Found",
                    "Features cache not specified or not found.\n\n"
                    "Would you like to extract features now?\n"
                    "(This may take a few minutes)"
                )
                
                if response:
                    self.al_log_message("Extracting features...")
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    _pf = self.current_project_folder.get() if hasattr(self, 'current_project_folder') else ''
                    if _pf and os.path.isdir(_pf):
                        cache_folder = os.path.join(_pf, 'features')
                    else:
                        cache_folder = os.path.join(os.path.dirname(labels_csv), 'features')
                    
                    features_cache = self._extract_features_for_session(
                        base_name, video_path, dlc_path, cache_folder
                    )
                    
                    if not features_cache:
                        messagebox.showerror("Error", "Feature extraction failed")
                        return
                    
                    # Update GUI
                    self.al_features_cache.set(features_cache)
                    self.al_log_message(f"✓ Features saved to: {os.path.basename(features_cache)}\n")
                else:
                    messagebox.showerror("Error", "Features cache required to continue")
                    return
            
            # Check files exist
            for filepath, name in [(labels_csv, "Labels CSV"), 
                                  (video_path, "Video"),
                                  (dlc_path, "DLC file"),
                                  (features_cache, "Features cache")]:
                if not os.path.exists(filepath):
                    messagebox.showerror("Error", f"{name} not found:\n{filepath}")
                    return
            
            # Get settings
            n_suggestions = self.al_n_suggestions.get()
            n_iterations = self.al_n_iterations.get()
            extend_to_unlabeled = (self.al_target_mode.get() == 'unlabeled')
            model_path = self.al_model_path.get().strip() if self.al_model_path.get().strip() else None
            
            if model_path and not os.path.exists(model_path):
                messagebox.showwarning("Warning", 
                    f"Model file not found:\n{model_path}\n\n"
                    f"A new model will be trained automatically.")
                model_path = None
            
            # Confirm start
            response = messagebox.askyesno(
                "Start Active Learning?",
                f"Active Learning will:\n\n"
                f"1. Find {n_suggestions} most uncertain frames\n"
                f"2. Show you each frame to label\n"
                f"3. Update your labels CSV automatically\n"
                f"4. You can then retrain with new labels\n\n"
                f"This saves 50-70% labeling time!\n\n"
                f"Continue?"
            )
            
            if not response:
                return
            
            # Clear log
            self.al_log.config(state='normal')
            self.al_log.delete('1.0', tk.END)
            self.al_log.config(state='disabled')
            
            self.al_log_message("="*60)
            self.al_log_message("🧠 ACTIVE LEARNING SESSION")
            self.al_log_message("="*60)
            self.al_log_message("")
            self.al_log_message(f"Labels CSV: {os.path.basename(labels_csv)}")
            self.al_log_message(f"Video: {os.path.basename(video_path)}")
            self.al_log_message(f"DLC: {os.path.basename(dlc_path)}")
            self.al_log_message(f"Features: {os.path.basename(features_cache)}")
            self.al_log_message(f"Suggestions: {n_suggestions}")
            self.al_log_message("")
            self.al_log_message("Starting active learning...")
            
            # Run active learning directly (NOT in thread, to avoid Tkinter issues)
            try:
                stats = active_learning.run_active_learning(
                    labels_csv=labels_csv,
                    video_path=video_path,
                    dlc_path=dlc_path,
                    features_cache=features_cache,
                    model_path=model_path,
                    n_suggestions=n_suggestions,
                    n_iterations=n_iterations,
                    extend_to_unlabeled=extend_to_unlabeled
                )
                
                # Show results
                self.al_show_results(stats, labels_csv)
                
            except Exception as error:
                # Show error
                error_msg = str(error)
                self.al_show_error(error_msg)
            
        except Exception as e:
            print(f"Error in run_active_learning: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Active Learning Error",
                f"An error occurred:\n\n{str(e)}\n\n"
                f"Check console for details."
            )
    
    def al_show_results(self, stats, labels_csv):
        """Show Active Learning results"""
        self.al_log_message("")
        self.al_log_message("="*60)
        self.al_log_message("📊 SESSION COMPLETE")
        self.al_log_message("="*60)
        self.al_log_message(f"Frames labeled: {stats.get('frames_labeled', 0)}")
        self.al_log_message(f"Iterations: {stats.get('iterations', 0)}")
        self.al_log_message(f"Positive labels: {stats.get('initial_positive', 0)} → {stats.get('final_positive', 0)}")
        self.al_log_message(f"Updated CSV: {os.path.basename(labels_csv)}")
        self.al_log_message("")
        self.al_log_message("✓ Active Learning complete!")
        self.al_log_message("Next step: Retrain your classifier with the updated labels.")
        
        messagebox.showinfo(
            "Active Learning Complete!",
            f"Session Statistics:\n\n"
            f"Frames labeled: {stats.get('frames_labeled', 0)}\n"
            f"Iterations: {stats.get('iterations', 0)}\n"
            f"Total positive labels: {stats.get('initial_positive', 0)} → {stats.get('final_positive', 0)}\n\n"
            f"Updated CSV: {os.path.basename(labels_csv)}\n\n"
            f"Next step: Retrain your classifier with\n"
            f"the updated labels to see improvement!"
        )
    
    def al_show_error(self, error):
        """Show Active Learning error"""
        self.al_log_message("")
        self.al_log_message("✗ ERROR")
        self.al_log_message(str(error))
        
        messagebox.showerror(
            "Active Learning Error",
            f"An error occurred:\n\n{str(error)}\n\n"
            f"Check the log for details."
        )
    
    def show_active_learning_help(self):
        """Show Active Learning documentation"""
        help_text = """
ACTIVE LEARNING HELP

Active Learning reduces labeling time by 50-70% by intelligently 
suggesting which frames to label for maximum improvement.

HOW IT WORKS:
1. Train an initial classifier on your existing labels
2. System finds frames where model is most uncertain
3. You label those frames (YES/NO/SKIP)
4. Labels are automatically saved to your CSV
5. Retrain with improved labels
6. Repeat until satisfied

REQUIRED FILES:
• Per-Frame Labels CSV: Output from BORIS converter
  Format: One column with behavior name, 0/1 per frame
  
• Video File: The video being analyzed (.mp4 or .avi)

• DLC File: DeepLabCut tracking data (.h5)

• Features Cache: Auto-generated during training/prediction
  Location: Usually in PredictionCache/ or FeatureCache/

SETTINGS:
• Frames to suggest: 20-50 recommended
  - Fewer (5-10): Quick iterations
  - More (50-100): Thorough coverage
  
• Model (optional): Pre-trained classifier
  - Leave blank to train automatically

TIPS:
• Start with 30-50 manual labels first
• Use Space to play context (±2 seconds)
• Label most suggestions (don't skip too many)
• Run 2-3 iterations for best results
• Check accuracy improvement after each round

KEYBOARD SHORTCUTS:
Y = Yes (behavior present)
N = No (not present)
S = Skip this frame
Space = Play context

TIME SAVINGS:
Without AL: 100 labels = 100 minutes
With AL:    40 labels = 40 minutes (60% faster!)

For more information, see the documentation.
"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Active Learning Help")
        help_window.geometry("700x600")
        
        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 10))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', help_text)
        text.config(state='disabled')
        
        ttk.Button(
            help_window,
            text="Close",
            command=help_window.destroy
        ).pack(pady=10)


# ============================================================================
# Main Application Entry Point
# ============================================================================

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
        except:
            pass
    
    # Install exception handler
    import sys
    sys.excepthook = handle_exception
    
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
        except:
            pass
    
    tk.Tk.report_callback_exception = report_callback_exception
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create accent button style
    style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    # Bind keyboard shortcuts
    root.bind('<F11>', lambda e: root.attributes('-fullscreen', 
                                                 not root.attributes('-fullscreen')))
    
    # Create and run application
    app = PixelPawsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
