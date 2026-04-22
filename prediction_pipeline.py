"""
prediction_pipeline.py — Feature extraction and prediction logic
=================================================================
Pure functions for:
  - Body-part list cleaning and subject-id parsing
  - Auto-detecting body parts from a trained model's feature names
  - Running the PixelPaws feature pipeline (pose + brightness + optional flow)
  - Loading cached features with incremental upgrade paths
  - Post-cache feature augmentation (egocentric / contact / lag)
  - XGBoost prediction with strict feature-column alignment

Extracted from PixelPaws_GUI.py so that callers (evaluation_tab, predict_tab,
transitions_tab, project_setup, unsupervised_tab, gait_limb_tab, analysis_tab)
don't have to import the GUI module to run predictions.

``PixelPaws_GUI.py`` re-exports every public name from this module so existing
``from PixelPaws_GUI import X`` imports keep working.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd

try:
    from pose_features import PoseFeatureExtractor
    from brightness_features import PixelBrightnessExtractor
    PIXELPAWS_MODULES_AVAILABLE = True
except ImportError:
    PoseFeatureExtractor = None
    PixelBrightnessExtractor = None
    PIXELPAWS_MODULES_AVAILABLE = False

try:
    from feature_cache import FeatureCacheManager
    _FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FeatureCacheManager = None
    _FEATURE_CACHE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Classifier portability check
# ---------------------------------------------------------------------------

def check_classifier_portability(clf_path, project_bp_list=None):
    """Inspect a classifier pkl and return warnings about compatibility with
    the current PixelPaws feature versions and (optionally) the current
    project's body-part set.

    Args:
        clf_path: Path to the classifier .pkl
        project_bp_list: Optional list of body-part names from the active
                         project config. When provided, mismatches against
                         the model's bp_include_list are flagged.

    Returns:
        list[str] — human-readable warnings (empty when the classifier is fully
        compatible). Does not raise; pickle / IO errors are returned as
        warnings too so the caller can show them to the user.
    """
    warnings = []
    try:
        try:
            from pose_features import POSE_FEATURE_VERSION as _POSE_V
        except Exception:
            _POSE_V = None
        try:
            from brightness_features import BRIGHTNESS_FEATURE_VERSION as _BRT_V
        except Exception:
            _BRT_V = None

        try:
            with open(clf_path, 'rb') as _f:
                cd = pickle.load(_f)
        except Exception as _open_err:
            return [f"Could not open classifier ({_open_err})"]

        model = cd.get('clf_model')
        if model is None:
            warnings.append("Pkl has no 'clf_model' key — file may be corrupt.")

        model_bp = cd.get('bp_include_list') or []
        if project_bp_list and model_bp:
            missing = set(project_bp_list) - set(model_bp)
            extra   = set(model_bp) - set(project_bp_list)
            if missing:
                warnings.append(
                    f"Classifier trained without {len(missing)} current-project "
                    f"bodypart(s): {sorted(missing)[:5]}")
            if extra:
                warnings.append(
                    f"Classifier expects {len(extra)} bodypart(s) not in the "
                    f"current project: {sorted(extra)[:5]}")

        pkl_pose_v = cd.get('pose_feature_version')
        if _POSE_V is not None and pkl_pose_v is not None and pkl_pose_v != _POSE_V:
            warnings.append(
                f"Pose feature version {pkl_pose_v} ≠ current {_POSE_V}. "
                "Inference may need pose-only re-extraction.")
        elif _POSE_V is not None and pkl_pose_v is None:
            warnings.append(
                "Classifier predates pose feature versioning — upgrade may be silent.")

        pkl_brt_v = cd.get('brightness_feature_version')
        if _BRT_V is not None and pkl_brt_v is not None and pkl_brt_v != _BRT_V:
            warnings.append(
                f"Brightness feature version {pkl_brt_v} ≠ current {_BRT_V}. "
                "Video re-read required for brightness features.")

        if model is not None and hasattr(model, 'feature_names_in_'):
            n = len(model.feature_names_in_)
            if cd.get('selected_feature_cols'):
                pruned_n = len(cd['selected_feature_cols'])
                if pruned_n != n:
                    warnings.append(
                        f"SHAP-pruned feature count mismatch: stored {pruned_n}, "
                        f"model {n} — pkl may be inconsistent.")

    except Exception as _e:
        warnings.append(f"Portability check failed: {_e}")

    return warnings


# ---------------------------------------------------------------------------
# Post-processing smoothing — switchable between bout filters and HMM Viterbi
# ---------------------------------------------------------------------------

def apply_smoothing(y_proba, clf_data, mode='bout_filters'):
    """Apply post-processing smoothing to raw classifier probabilities.

    Parameters
    ----------
    y_proba  : np.ndarray — raw per-frame P(behavior) from predict_with_xgboost
    clf_data : dict       — loaded classifier pkl
    mode     : str        — 'bout_filters' | 'hmm_viterbi' | 'none'

    Returns
    -------
    np.ndarray of int (0 or 1), length == len(y_proba)

    Backward compatibility
    ----------------------
    Old classifiers without 'hmm_log_trans'/'hmm_log_prior' keys fall back to
    bout filters when mode=='hmm_viterbi', with a console warning.
    """
    thresh = float(clf_data.get('best_thresh', 0.5))

    if mode == 'hmm_viterbi':
        log_trans = clf_data.get('hmm_log_trans')
        log_prior = clf_data.get('hmm_log_prior')
        if log_trans is not None and log_prior is not None:
            try:
                from evaluation_tab import viterbi_smooth
                return viterbi_smooth(y_proba,
                                      np.asarray(log_trans),
                                      np.asarray(log_prior))
            except Exception as _ve:
                print(f"  ⚠ Viterbi smoothing failed ({_ve}) — falling back to bout filters")
        else:
            print("  ⚠ HMM params not in this classifier — falling back to bout filters")
        # Fall through to bout_filters

    y_pred = (y_proba >= thresh).astype(int)
    if mode == 'none':
        return y_pred

    # 'bout_filters' (default) or fallback from hmm_viterbi
    try:
        from evaluation_tab import _apply_bout_filtering
        return _apply_bout_filtering(
            y_pred,
            int(clf_data.get('min_bout', 1)),
            int(clf_data.get('min_after_bout', 0)),
            int(clf_data.get('max_gap', 0)),
        )
    except Exception as _fe:
        print(f"  ⚠ Bout filtering failed ({_fe}) — returning raw threshold")
        return y_pred


# ---------------------------------------------------------------------------
# Body-part list cleaning
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Auto-detect body parts from a trained model's feature names
# ---------------------------------------------------------------------------

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

        def _strip_ego(bp):
            """Strip egocentric prefix — Ego_flpaw → flpaw."""
            return bp[4:] if bp.startswith('Ego_') else bp

        # From velocity features (most reliable): bodypart_Vel1, bodypart_Vel2, bodypart_Vel10
        vel_count = 0
        for f in features:
            if '_Vel' in f and 'sum_' not in f and 'Pix' not in f and 'Dis_' not in f:
                bp = _strip_ego(f.split('_Vel')[0])
                bodypart_names.add(bp)
                vel_count += 1
        if verbose and vel_count > 0:
            print(f"    Found {len(bodypart_names)} body parts from {vel_count} velocity features")

        # From in-frame features: bodypart_inFrame
        inframe_count = 0
        for f in features:
            if '_inFrame' in f:
                bp = _strip_ego(f.split('_inFrame')[0])
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
                # Strip Ego_ prefix (e.g. Ego_Dis_bp1-bp2 → Ego_bp1-bp2 after Dis_ removal)
                if dist_part.startswith('Ego_'):
                    dist_part = dist_part[4:]
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
                if ang_part.startswith('Ego_'):
                    ang_part = ang_part[4:]
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


# ---------------------------------------------------------------------------
# Full feature extraction (pose + brightness + optional optical flow)
# ---------------------------------------------------------------------------

def PixelPaws_ExtractFeatures(pose_data_file, video_file_path, bp_pixbrt_list,
                              square_size, pix_threshold, bp_include_list=None,
                              scale_x=1, scale_y=1, dt_vel=2, min_prob=0.8,
                              crop_offset_x=0, crop_offset_y=0, config_yaml_path=None,
                              include_optical_flow=False, bp_optflow_list=None,
                              cancel_flag=None,
                              ):
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

    # Clean body parts lists (remove DLC network names)
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
        velocity_delta=dt_vel
    )

    # DEBUG: Print what body parts we're actually using
    print(f"  Body parts for pose features: {bp_include_list_cleaned}")
    print(f"  Number of body parts: {len(bp_include_list_cleaned) if bp_include_list_cleaned else 0}")

    X_pose = pose_extractor.extract_all_features(
        pose_data_file)

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
        crop_offset_x=crop_offset_x,
        crop_offset_y=crop_offset_y
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
        cancel_flag=cancel_flag,
    )

    # 3. Combine features
    X = pd.concat([X_pose, X_brightness], axis=1)

    print(f"  ✓ Extracted {X.shape[1]} features from {X.shape[0]} frames")
    return X


# ---------------------------------------------------------------------------
# XGBoost prediction (strict feature alignment)
# ---------------------------------------------------------------------------

def predict_with_xgboost(model, X, calibrator=None, fold_models=None):
    """
    Predict with XGBoost model, handling GPU models and feature selection.

    CRITICAL: Selects only the features the model was trained on, in correct order.
    This is essential for BAREfoot compatibility and prevents feature mismatch errors.

    Args:
        model: Trained XGBoost model (the final model trained on all data)
        X: Feature DataFrame (may have more/different features than model needs)
        calibrator: Optional fitted sklearn calibrator (e.g., IsotonicRegression)
                    with a .predict(p) method returning calibrated probabilities.
        fold_models: Optional list of additional XGBoost models (one per CV fold).
                     When provided, their predict_proba outputs are averaged with
                     the primary model's before calibration is applied.

    Returns:
        Array of prediction probabilities (calibrated if calibrator was provided)
    """
    try:
        # CRITICAL: Select only features the model was trained on
        if hasattr(model, 'feature_names_in_'):
            # Check if all required features are present
            missing_features = set(model.feature_names_in_) - set(X.columns)
            if missing_features:
                # Show first 10 missing features for debugging
                missing_list = [str(f) for f in list(missing_features)[:10]]
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

    except ValueError:
        # Re-raise feature mismatch errors with full context
        raise
    except Exception as e:
        # Log other errors but continue with fallback
        print(f"  Warning during prediction setup: {e}")
        # Use X_model if we got that far, otherwise use X
        X_model = X_model if 'X_model' in locals() else X

    y_proba = model.predict_proba(X_model)[:, 1]

    if fold_models:
        fold_probas = [y_proba]
        for _fm in fold_models:
            try:
                _fm_X = X[_fm.feature_names_in_] if hasattr(_fm, 'feature_names_in_') else X_model
                fold_probas.append(_fm.predict_proba(_fm_X)[:, 1])
            except Exception as _fm_err:
                print(f"  Warning: fold model predict failed ({_fm_err}); skipping")
        if len(fold_probas) > 1:
            y_proba = np.mean(np.stack(fold_probas, axis=0), axis=0)
            print(f"  Averaged {len(fold_probas)} models (final + {len(fold_probas) - 1} fold)")

    if calibrator is not None:
        try:
            y_proba = np.clip(calibrator.predict(y_proba), 0.0, 1.0)
            print("  Applied probability calibration")
        except Exception as _cal_err:
            print(f"  Warning: calibrator failed ({_cal_err}); using raw probabilities")

    return y_proba


# ---------------------------------------------------------------------------
# Post-cache feature augmentation (egocentric / contact / lag)
# ---------------------------------------------------------------------------

def augment_features_post_cache(X, clf_data, model, dlc_path, log_fn=None):
    """
    Add egocentric and lag features to X if the model requires them.
    Matches the post-cache augmentation done during training.
    Returns the (possibly augmented) DataFrame.
    """
    # --- Egocentric features ---
    try:
        _need_ego = clf_data.get('use_egocentric', False)
        if not _need_ego and hasattr(model, 'feature_names_in_'):
            _need_ego = any(f.startswith('Ego_') for f in model.feature_names_in_)
        if _need_ego:
            _ego_ext = PoseFeatureExtractor(bodyparts=[])
            _ego_dlc = _ego_ext.load_dlc_data(dlc_path)
            _ego_xc, _ego_yc, _ = _ego_ext.get_bodypart_coords(_ego_dlc)
            _ego_x, _ego_y = _ego_ext.normalize_egocentric(_ego_xc, _ego_yc)
            _ego_dist = _ego_ext.calculate_distances(_ego_x, _ego_y)
            _ego_dist.columns = [f'Ego_{c}' for c in _ego_dist.columns]
            _ego_vel = _ego_ext.calculate_velocities(_ego_x, _ego_y, t=1)
            _ego_vel.columns = [f'Ego_{c}' for c in _ego_vel.columns]
            _ego_df = pd.concat([_ego_dist, _ego_vel], axis=1).fillna(0)
            _ego_df = _ego_df.iloc[:len(X)].reset_index(drop=True)
            X = pd.concat([X.reset_index(drop=True), _ego_df], axis=1)
            if log_fn:
                log_fn(f'  + {len(_ego_df.columns)} egocentric features')
    except Exception as e:
        if log_fn:
            log_fn(f'  ⚠️  Egocentric augmentation failed: {e}')

    # --- Contact state features ---
    try:
        _need_contact = clf_data.get('use_contact_features', False)
        if not _need_contact and hasattr(model, 'feature_names_in_'):
            _need_contact = any(f.endswith('_ContactState') for f in model.feature_names_in_)
        if _need_contact:
            _height_cols = [c for c in X.columns if c.endswith('_Height')]
            if _height_cols and not any(c.endswith('_ContactState') for c in X.columns):
                _ct_thresh = clf_data.get('contact_threshold', 15.0)
                _ct_ext = PoseFeatureExtractor(bodyparts=[], contact_threshold=_ct_thresh)
                _ct_df = _ct_ext.calculate_contact_features(X)
                if not _ct_df.empty:
                    _ct_df = _ct_df.iloc[:len(X)].reset_index(drop=True)
                    X = pd.concat([X.reset_index(drop=True), _ct_df], axis=1)
                    if log_fn:
                        log_fn(f'  + {len(_ct_df.columns)} contact state features')
    except Exception as e:
        if log_fn:
            log_fn(f'  ⚠️  Contact augmentation failed: {e}')

    # --- Lag/lead features ---
    try:
        _need_lag = clf_data.get('use_lag_features', False)
        _lag_feat_names = []
        if hasattr(model, 'feature_names_in_'):
            _lag_feat_names = [f for f in model.feature_names_in_ if '_lag' in f]
        if not _need_lag and _lag_feat_names:
            _need_lag = True
        if _need_lag and _lag_feat_names:
            _lag_bases = set()
            for _lf in _lag_feat_names:
                _m = re.match(r'^(.+)_lag[mp]\d+$', _lf)
                if _m:
                    _lag_bases.add(_m.group(1))
            _lag_cols = [c for c in _lag_bases if c in X.columns]
            if _lag_cols:
                _lag_dfs = []
                for _lag in (-2, -1, 1, 2):
                    _shifted = X[_lag_cols].shift(_lag).fillna(0)
                    _sign = f"m{abs(_lag)}" if _lag < 0 else f"p{_lag}"
                    _shifted.columns = [f"{c}_lag{_sign}" for c in _lag_cols]
                    _lag_dfs.append(_shifted)
                _lag_df = pd.concat(_lag_dfs, axis=1)
                X = pd.concat([X, _lag_df], axis=1)
                if log_fn:
                    log_fn(f'  + {len(_lag_df.columns)} lag/lead features')
        elif _need_lag and not _lag_feat_names:
            _lag_ext = PoseFeatureExtractor(bodyparts=[])
            _lag_df = _lag_ext.calculate_lag_features(X, lags=(-2, -1, 1, 2), top_n=10)
            if not _lag_df.empty:
                X = pd.concat([X, _lag_df], axis=1)
                if log_fn:
                    log_fn(f'  + {len(_lag_df.columns)} lag/lead features (variance fallback)')
    except Exception as e:
        if log_fn:
            log_fn(f'  ⚠️  Lag augmentation failed: {e}')

    # --- Brightness Category B (post-cache derived) ------------------------
    # All 7 are deterministic transforms of existing cache columns.  Always
    # computed when Pix_ columns exist — gain pruning at training time
    # decides which survive.  At predict time XGBoost selects only the
    # columns the model was trained on via feature_names_in_.
    try:
        X = compute_brightness_category_b(X, log_fn=log_fn)
    except Exception as e:
        if log_fn:
            log_fn(f'  ⚠️  Brightness Category-B augmentation failed: {e}')

    # --- Normalized pairwise distances (ARBEL parity) ----------------------
    # Same design as Category B: always compute when Dis_ columns exist,
    # let gain pruning pick between raw / normalized versions per-behavior.
    try:
        X = compute_normalized_distances(X, log_fn=log_fn)
    except Exception as e:
        if log_fn:
            log_fn(f'  ⚠️  Normalized-distance augmentation failed: {e}')

    return X


def compute_normalized_distances(X, log_fn=None):
    """Add ARBEL-parity max-matrix normalized distances to X.

    For each `Dis_bp1-bp2` column already in X, adds a `Dis_norm_bp1-bp2`
    column equal to `Dis_bp1-bp2 / max_over_all_Dis_columns`.  Gives the
    classifier a scale-invariant version of each pairwise distance that
    generalizes across camera zooms and animal sizes.

    Idempotent — skips if `Dis_norm_*` already present.
    """
    _dis_cols = [c for c in X.columns
                 if c.startswith('Dis_') and not c.startswith('Dis_norm_')]
    _norm_existing = [c for c in X.columns if c.startswith('Dis_norm_')]
    if not _dis_cols or _norm_existing:
        return X
    try:
        _dmax = float(X[_dis_cols].abs().max().max())
    except Exception:
        _dmax = 0.0
    if _dmax <= 0:
        return X
    _norm_df = X[_dis_cols] / _dmax
    _norm_df.columns = [c.replace('Dis_', 'Dis_norm_', 1) for c in _dis_cols]
    X = pd.concat([X.reset_index(drop=True),
                   _norm_df.reset_index(drop=True)], axis=1)
    if log_fn:
        log_fn(f'  + {len(_norm_df.columns)} normalized distance features')
    return X


def compute_brightness_category_b(X, log_fn=None):
    """Add seven derived brightness features to X in place and return it.

    Features (one per bodypart unless noted, 15-20 new columns for a typical
    3-paw project):
      Pix_baseline_sub_[bp]   — session-median-subtracted brightness
      Pix_std_temporal_[bp]   — rolling std of Pix_[bp] over 10 frames
      Pix_jerk_[bp]           — 3rd derivative of Pix_[bp]
      Pix_onset_[bp]          — peak / time-to-peak over an 11-frame window
      Pix_prequi_[bp]         — reciprocal rolling variance shifted back 5 frames
      Pix_corr_[lbp-rbp]      — rolling Pearson correlation between L/R paw pairs
      Pix_velprod_[bp]        — Pix_[bp] * |Vel1_[bp]|

    All deterministic transforms of cache columns — no new overfit sources.
    Caller wraps in try/except; this function is best-effort.
    """
    _pix_cols = [c for c in X.columns if c.startswith('Pix_')
                 and not c.startswith('Pix_baseline_sub_')
                 and not c.startswith('Pix_std_temporal_')
                 and not c.startswith('Pix_jerk_')
                 and not c.startswith('Pix_onset_')
                 and not c.startswith('Pix_prequi_')
                 and not c.startswith('Pix_corr_')
                 and not c.startswith('Pix_velprod_')
                 and '/' not in c]
    if not _pix_cols:
        return X

    _aug_dfs = []

    # B.1 baseline-subtracted brightness
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _median = float(X[_col].median())
        _aug_dfs.append(
            (X[_col] - _median).rename(f'Pix_baseline_sub_{_bp}').to_frame())

    # B.2 rolling temporal std (trembling indicator)
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _aug_dfs.append(
            X[_col].rolling(10, min_periods=1).std()
                   .fillna(0).rename(f'Pix_std_temporal_{_bp}').to_frame())

    # B.3 brightness jerk (3rd derivative, absolute)
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _j = X[_col].diff().diff().diff().abs().fillna(0)
        _aug_dfs.append(_j.rename(f'Pix_jerk_{_bp}').to_frame())

    # B.4 brightness onset sharpness (peak / time-to-peak)
    def _onset(arr):
        if arr.size < 2 or arr.max() <= 0:
            return 0.0
        return float(arr.max()) / (abs(int(np.argmax(arr)) - (arr.size // 2)) + 1.0)
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _o = X[_col].rolling(11, center=True, min_periods=1).apply(_onset, raw=True).fillna(0)
        _aug_dfs.append(_o.rename(f'Pix_onset_{_bp}').to_frame())

    # B.5 pre-event quiescence on brightness (stillness-before-spike)
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _var = X[_col].rolling(20, min_periods=3).var().shift(5)
        _pq = (1.0 / (_var + 1e-3)).fillna(0)
        _aug_dfs.append(_pq.rename(f'Pix_prequi_{_bp}').to_frame())

    # B.6 bilateral Pix correlation (L/R pairs only)
    _LR_PATTERNS = [('left', 'right'), ('Left', 'Right'),
                    ('hl', 'hr'), ('fl', 'fr'), ('l', 'r')]
    _bps = [c[len('Pix_'):] for c in _pix_cols]
    _bp_set = set(_bps)
    _pairs = []
    _seen = set()
    for _bp in _bps:
        if _bp in _seen:
            continue
        for _lp, _rp in _LR_PATTERNS:
            if _bp.startswith(_lp):
                _cand = _rp + _bp[len(_lp):]
                if _cand in _bp_set and _cand not in _seen:
                    _pairs.append((_bp, _cand))
                    _seen.update({_bp, _cand})
                    break
            elif _bp.startswith(_rp):
                _cand = _lp + _bp[len(_rp):]
                if _cand in _bp_set and _cand not in _seen:
                    _pairs.append((_cand, _bp))
                    _seen.update({_bp, _cand})
                    break
    for _lbp, _rbp in _pairs:
        _corr = X[f'Pix_{_lbp}'].rolling(15, min_periods=3).corr(X[f'Pix_{_rbp}']).fillna(0)
        _aug_dfs.append(_corr.rename(f'Pix_corr_{_lbp}-{_rbp}').to_frame())

    # B.7 brightness × velocity product (pressure-change-during-motion)
    for _col in _pix_cols:
        _bp = _col[len('Pix_'):]
        _vel_col = None
        for _cand_v in (f'{_bp}_Vel1', f'{_bp}_Vel2'):
            if _cand_v in X.columns:
                _vel_col = _cand_v
                break
        if _vel_col is None:
            continue
        _vp = (X[_col] * X[_vel_col].abs()).fillna(0)
        _aug_dfs.append(_vp.rename(f'Pix_velprod_{_bp}').to_frame())

    if _aug_dfs:
        _brightness_b = pd.concat(_aug_dfs, axis=1)
        X = pd.concat([X.reset_index(drop=True),
                       _brightness_b.reset_index(drop=True)], axis=1)
        if log_fn:
            log_fn(f'  + {len(_brightness_b.columns)} brightness Category-B features')
    return X


# ---------------------------------------------------------------------------
# Feature cache loading with incremental upgrade
# ---------------------------------------------------------------------------

# Features added post-cache by augment_features_post_cache() — missing these
# columns from a cache is NOT a reason to re-extract.
_POST_CACHE_RE = re.compile(
    r'^Ego_|_ContactState$|_ContactTransition$|_DutyCycle$|^N_InContact$|_lag[mp]\d+$'
)


def _load_features_for_prediction(cache_file, model, extract_fn=None,
                                  save_path=None, log_fn=None,
                                  dlc_path=None, clf_data=None):
    """Load feature cache if base features are compatible with model; re-extract if not.

    Parameters
    ----------
    cache_file : str or None
        Path to an existing feature pkl to try first.
    model : XGBoost model
        Used to check feature_names_in_ compatibility.
    extract_fn : callable() -> pd.DataFrame, or None
        Called when cache is absent or stale. Must return the full feature DataFrame.
        When None, the function returns None instead of extracting — useful when the
        caller wants to try a user-provided features file first and fall through to
        a different extraction strategy on failure.
    save_path : str or None
        Where to write freshly extracted features (None = don't save).
    log_fn : callable(str) or None
        Optional logging callback.
    dlc_path : str or None
        Path to the DLC .h5 file; enables pose-only upgrade without re-reading video.
    clf_data : dict or None
        Classifier metadata dict; required for pose-only upgrade.

    Returns
    -------
    pd.DataFrame  (base features only — caller must still call augment_features_post_cache)
    or None when extract_fn is None and the cache could not be used.
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    def _save(df, target):
        """Atomic write — delegates to FeatureCacheManager when available."""
        if _FEATURE_CACHE_AVAILABLE:
            FeatureCacheManager.save_cache(df, target)
        else:
            import tempfile
            _dir = os.path.dirname(target) or '.'
            os.makedirs(_dir, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=_dir, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    pickle.dump(df, f)
                os.replace(tmp_path, target)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    # --- try cache ---
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file, 'rb') as _f:
                X = pickle.load(_f)
            # Unwrap training-data-backup pickles of the form {'X': df, 'y': ...}
            if isinstance(X, dict) and 'X' in X:
                X = X['X']
            if hasattr(model, 'feature_names_in_'):
                _base_needed = {str(f) for f in model.feature_names_in_
                                if not _POST_CACHE_RE.search(str(f))}
                _missing = _base_needed - set(X.columns)
                if not _missing:
                    _log(f"✓ Loaded cached features from {cache_file}")
                    return X
                _log(f"⚠ Cache stale — {len(_missing)} base feature(s) missing. "
                     f"Trying pose-only upgrade (no video re-read)...")

                # --- try incremental pose-only upgrade first ---
                _cache_version = lambda cols: (
                    'v5' if any('_PreQuiescence' in c for c in cols) else
                    'v4' if any('_VelAsymmetry' in c for c in cols) else
                    'v3' if any('_Jerk1'        in c for c in cols) else
                    'v2' if any('_Vel1'         in c for c in cols) else
                    'v1 (or unknown)')
                if dlc_path and clf_data:
                    try:
                        _tgt = save_path or cache_file
                        # Try the newest upgrade path first: v4 → v5
                        _upgraded = FeatureCacheManager.try_upgrade_v4_to_v5(
                            cache_file, _tgt, clf_data, dlc_path, log_fn=_log)
                        if _upgraded is None:
                            _upgraded = FeatureCacheManager.try_upgrade_v3_to_v4(
                                cache_file, _tgt, clf_data, dlc_path, log_fn=_log)
                            if _upgraded is not None:
                                # Chain: v3→v4 succeeded; now try v4→v5 on the saved file
                                _v5 = FeatureCacheManager.try_upgrade_v4_to_v5(
                                    _tgt, _tgt, clf_data, dlc_path, log_fn=_log)
                                if _v5 is not None:
                                    _upgraded = _v5
                        if _upgraded is None:
                            _upgraded = FeatureCacheManager.try_upgrade_v2_to_v3(
                                cache_file, _tgt, clf_data, dlc_path, log_fn=_log)
                            if _upgraded is not None:
                                # Chain: v2→v3 → v3→v4 → v4→v5
                                _v4 = FeatureCacheManager.try_upgrade_v3_to_v4(
                                    _tgt, _tgt, clf_data, dlc_path, log_fn=_log)
                                if _v4 is not None:
                                    _upgraded = _v4
                                    _v5 = FeatureCacheManager.try_upgrade_v4_to_v5(
                                        _tgt, _tgt, clf_data, dlc_path, log_fn=_log)
                                    if _v5 is not None:
                                        _upgraded = _v5
                        if _upgraded is not None:
                            _still = {str(f) for f in model.feature_names_in_
                                      if not _POST_CACHE_RE.search(str(f))} - set(_upgraded.columns)
                            if not _still:
                                _log("✓ Cache upgraded from DLC (no video re-read)")
                                return _upgraded
                            _still_brt  = [f for f in sorted(_still) if 'brt' in f.lower() or 'pix' in f.lower()]
                            _still_pose = [f for f in sorted(_still) if f not in _still_brt]
                            _log(f"⚠ After upgrade, {len(_still)} feature(s) still missing:")
                            if _still_brt:
                                _log(f"  Brightness (need video): {_still_brt[:5]}")
                            if _still_pose:
                                _log(f"  Pose (unexpected): {_still_pose[:5]}")
                            _log("  Falling back to full re-extraction.")
                        else:
                            _ver = _cache_version(X.columns)
                            _log(f"  Cache appears to be {_ver}. "
                                 f"Upgrade returned None — "
                                 f"{'already at v4' if _ver == 'v4' else 'too old to upgrade incrementally'}.")
                            _missing_list = sorted(_missing)[:8]
                            _brt = [f for f in _missing_list if 'brt' in f.lower() or 'pix' in f.lower()]
                            _pose = [f for f in _missing_list if f not in _brt]
                            if _brt:
                                _log(f"  Missing brightness feature(s): {_brt} — requires video re-read.")
                            if _pose:
                                _log(f"  Missing pose feature(s): {_pose}")
                        # --- brightness-preserve: if only pose features missing, skip video re-read ---
                        _candidate = _upgraded if _upgraded is not None else X
                        _cand_still = _base_needed - set(_candidate.columns)
                        _brt_still  = [f for f in _cand_still
                                        if 'brt' in f.lower() or 'pix' in f.lower()]
                        _pose_still = [f for f in _cand_still if f not in _brt_still]
                        if not _brt_still and _pose_still:
                            _log("  Brightness features present in cache — re-extracting pose from DLC (no video re-read)...")
                            try:
                                _pe = PoseFeatureExtractor(
                                    bodyparts=clf_data.get('bp_include_list') or [],
                                    likelihood_threshold=clf_data.get('pix_threshold', 0.8),
                                    velocity_delta=clf_data.get('dt_vel', 2),
                                )
                                _pose_fresh = _pe.extract_all_features(dlc_path)
                                if len(_pose_fresh) > len(_candidate):
                                    _pose_fresh = _pose_fresh.iloc[:len(_candidate)].reset_index(drop=True)
                                if len(_pose_fresh) == len(_candidate):
                                    _brt_cols = [c for c in _candidate.columns
                                                 if 'brt' in c.lower() or 'pix' in c.lower()]
                                    _merged = pd.concat([
                                        _pose_fresh.reset_index(drop=True),
                                        _candidate[_brt_cols].reset_index(drop=True)
                                    ], axis=1)
                                    if not (_base_needed - set(_merged.columns)):
                                        _log("✓ Pose re-extracted from DLC; brightness preserved from cache (no video re-read)")
                                        if save_path:
                                            _save(_merged, save_path)
                                        return _merged
                                    _log(f"  Merged still missing {len(_base_needed - set(_merged.columns))} feature(s) — falling back")
                                else:
                                    _log(f"  Row count mismatch ({len(_candidate)} vs {len(_pose_fresh)}) — falling back")
                            except Exception as _brt_err:
                                _log(f"  Brightness-preserve attempt failed ({_brt_err}) — falling back")
                    except Exception as _ue:
                        _log(f"⚠ Pose-only upgrade failed ({_ue}) — falling back to full re-extraction")

            else:
                _log(f"✓ Loaded cached features from {cache_file}")
                return X
        except Exception as _e:
            _log(f"⚠ Could not load cache: {_e}")

    # --- extract fresh (or signal failure to caller) ---
    if extract_fn is None:
        return None
    _log("Extracting features...")
    X = extract_fn()
    if save_path:
        _save(X, save_path)
        _log(f"✓ Features extracted and cached to {save_path}")
    return X
