# CLAUDE.md — PixelPaws

## Project Overview

PixelPaws is a tkinter desktop GUI for automated rodent behavioral analysis from DeepLabCut pose estimation data. It trains XGBoost classifiers on pose/brightness features, provides SHAP explainability, active learning, batch prediction, and unsupervised clustering (UMAP + HDBSCAN).

- **Python 3.11+**
- **Run:** `python PixelPaws_GUI.py`
- **Install:** `pip install -r requirements.txt`
- Additional optional deps: `pip install umap-learn hdbscan` (for unsupervised tab)

## Architecture

- **Monolithic main GUI** — `PixelPaws_GUI.py` (~12K lines) contains the core app, all training logic, prediction, and feature extraction
- **External tab modules** — separate files handle specific tabs (evaluation, analysis, predict, active learning, unsupervised, gait & limb use)
- **Shared state** — `current_project_folder` (tkinter `StringVar`) is the central project path; a trace callback `_on_project_folder_changed` refreshes all tabs when it changes
- **Session discovery** — `find_session_triplets()` in `evaluation_tab.py` scans for video + pose + label file triples; used by training, evaluation, and other tabs
- **Startup sequence** — `root.withdraw()` on init → `ProjectSetupWizard` (3-step modal in `project_setup.py`) → `root.deiconify()` on finish or `root.destroy()` on close

## File Map

### Core
| File | Description |
|------|-------------|
| `PixelPaws_GUI.py` | Main GUI application — training, prediction, feature extraction, all UI |
| `project_setup.py` | Startup wizard (3-step project creation/selection) |

### Tab Modules
| File | Description |
|------|-------------|
| `evaluation_tab.py` | Classifier evaluation, SHAP analysis, `find_session_triplets()` |
| `analysis_tab.py` | Batch analysis and statistics |
| `predict_tab.py` | Batch prediction on unlabeled videos |
| `active_learning_v2.py` | Active learning workflow |
| `unsupervised_tab.py` | UMAP + HDBSCAN unsupervised clustering |
| `gait_limb_tab.py` | Gait & Limb Use analysis |

### Feature Extraction
| File | Description |
|------|-------------|
| `pose_features.py` | Pose-based feature extractor (velocities, angles, distances, kinematics) |
| `brightness_features.py` | Pixel brightness features around body parts |
| `optical_flow_features.py` | Optical flow feature extraction |
| `brightness_preview.py` | Visual preview of brightness ROIs |
| `brightness_diagnostics.py` | Brightness feature diagnostics |

### Utilities
| File | Description |
|------|-------------|
| `classifier_training.py` | XGBoost classifier training helpers |
| `behavior_presets.py` | Predefined behavior configurations |
| `label_manager.py` | Label file management |
| `analysis_utils.py` | Shared analysis helpers |
| `analyze_batch_results.py` | Batch result analysis |
| `render_skeleton_video.py` | Skeleton overlay video rendering |
| `crop_for_dlc.py` | Video cropping for DeepLabCut |
| `correct_features_crop.py` | Feature correction after cropping |
| `check_classifier.py` | Classifier inspection utility |

## Project Folder Layout

```
<project>/
  videos/            — videos + DeepLabCut .h5 pose files
  behavior_labels/   — label CSVs (canonical location, searched first)
  classifiers/       — trained .pkl classifiers
  evaluations/       — evaluation reports + SHAP outputs
  features/          — cached feature files (canonical cache location)
  results/           — prediction outputs
  analysis/          — batch analysis outputs
  unsupervised/      — UMAP/HDBSCAN run outputs
  PixelPaws_project.json
```

## Feature Caching

This is the most complex subsystem in the codebase. Understand it fully before modifying any feature extraction or loading code.

### Cache file naming
```
{session_name}_features_{cfg_hash}.pkl
```
The hash is an MD5 of a normalized config dict (see `_feature_hash_key` at GUI.py:5514):
- `bp_include_list`, `bp_pixbrt_list`, `square_size`, `pix_threshold`
- `POSE_FEATURE_VERSION` (integer)
- `include_optical_flow`, `bp_optflow_list`

### Canonical location
```
<project>/features/{session}_features_{hash}.pkl
```

### Fallback search order (GUI.py:5550-5575)
When a cache file is not found in the canonical location, the code searches:
1. `<project>/features/` (canonical)
2. `<video_dir>/` (same directory as video)
3. `<video_dir>/features/`
4. `<video_dir>/FeatureCache/` (legacy)
5. `<video_dir>/PredictionCache/`
6. Ancestor directory walk up to project root: `<ancestor>/features/` + `<ancestor>/FeatureCache/` at each level

This same fallback pattern is replicated in:
- Training: `extract_features_for_session()` (GUI.py:5532)
- Prediction tab: (GUI.py:~7939, ~8374)
- Evaluation tab: (evaluation_tab.py:~742)
- Feature deletion/management: (GUI.py:~9343)
- Unsupervised tab: (unsupervised_tab.py:~1032)

### Version upgrade (v2 → v3)
When a hash-mismatched cache is found, the code checks if it's an older version missing kinematics columns (`_Jerk1`). If so, it extracts only the missing columns and concatenates them, avoiding a full video re-read (GUI.py:5586-5635).

### Smart defaults variant
```
{session}_features_smart_{hash}.pkl
```

## Coding Conventions

- **Optional imports** — new dependencies use `try/except` with `*_AVAILABLE` boolean flags for graceful degradation
- **No automated test suite** — all testing is manual via the GUI
- **Body part names** — lowercase: `hrpaw`, `hlpaw`, `snout`, `flpaw`, `frpaw`, `tailbase`
- **Classifier storage** — `.pkl` files go directly to `<project>/classifiers/` (no `rig_name` subdirectory)
- **Label search priority** — `behavior_labels/` is first in `label_candidates` in `find_session_triplets()`

## Claude Behavioral Guidance

- **Never add Co-Authored-By lines to commits**
- **Feature cache awareness** — when modifying feature extraction or loading code, always check ALL cache search locations (see Feature Caching section above). The fallback hierarchy spans 7+ directories including ancestor walks. The same pattern is duplicated in at least 5 places. Missing a location causes "features not found" bugs.
- **Read before edit** — always read the target file/function before modifying. The codebase is large (~12K lines in the main file) and heavily interconnected.
- **Test manually** — no automated tests exist. Verify changes by running the GUI.
- **Preserve the optional-import pattern** — new dependencies must degrade gracefully with `try/except` and an `*_AVAILABLE` flag.
- **Don't duplicate existing helpers** — check for existing functions before writing new ones. Key helpers: `_feature_hash_key`, `find_session_triplets`, `_treatment_groups`, `refresh_pred_classifiers`, `refresh_classifiers`.
- **Project folder convention** — use `<project>/features/` as canonical write location, but always respect the full fallback chain when reading/loading.
- **`behavior_labels/` is canonical** — it is searched first for label files. Don't change this priority.
