# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PixelPaws is a desktop GUI application for automated animal behavior analysis using machine learning. It processes DeepLabCut pose estimation data from videos to train XGBoost classifiers that detect animal behaviors (flinching, licking, grooming, etc.). The project targets behavioral neuroscience researchers.

## Running the Application

```bash
# Launch main GUI
python PixelPaws_GUI.py

# Install dependencies (GPU-accelerated, recommended)
pip install -r requirements_gpu.txt
python enable_pytorch_gpu.py

# Install CPU-only
pip install -r requirements.txt
```

There is no build step, test suite, or linting configuration.

## Architecture

The application is a single-directory flat Python project (no package structure). The main entry point is `PixelPaws_GUI.py`.

### Layers

**1. GUI Layer — `PixelPaws_GUI.py`**
A large monolithic file containing all tkinter GUI classes. Key classes: `PixelPawsGUI` (main window with tabbed interface), `VideoPreviewWindow`, `TrainingVisualizationWindow`, `AutoLabelWindow`, `EthogramGenerator`. The GUI has five tabs: Train, Predict, Analyze, Evaluate, and (optional) Active Learning.

**2. Feature Extraction Layer**
- `pose_features.py` — `PoseFeatureExtractor`: loads DeepLabCut H5/CSV files and computes kinematic features (pairwise distances, joint angles, velocities at multiple timescales, in-frame probabilities). Matches the BAREfoot algorithm exactly.
- `brightness_features.py` — `PixelBrightnessExtractor`: extracts pixel brightness from square ROIs around body parts in video frames. Supports optional PyTorch/CUDA acceleration for significant speedup.
- `pixelpaws_easy.py` — convenience wrappers (`extract_all_features_auto`, `extract_all_features`, `extract_pose_only`, `extract_brightness_only`) for scripted use outside the GUI.

**3. Classification Layer — `classifier_training.py`**
`BehaviorClassifier` wraps XGBoost with the full training pipeline: data balancing, k-fold cross-validation, threshold optimization, SHAP feature importance. Default hyperparameters (n_estimators=1700, max_depth=6, lr=0.01, colsample_bytree=0.2) match the BAREfoot paper.

**4. Pipeline Tabs** (imported into the GUI)
- `analysis_tab.py` — batch analysis with time binning and group statistics
- `evaluation_tab.py` — model evaluation against labeled test data
- `predict_tab.py` — single-video prediction interface
- `active_learning.py` — intelligent frame selection to minimize labeling effort

**5. Utilities**
- `label_manager.py` — `SmartLabelManager`: handles dense regions (from BORIS) and sparse frames (from active learning) without conflating them during training.
- `behavior_presets.py` — predefined configs for 10+ behaviors with recommended body parts, bout parameters, and ROI sizes.

### Data Flow

```
Video + DeepLabCut H5/CSV
    → Feature Extraction (pose + brightness, cached by video hash)
    → XGBoost Training (balanced, cross-validated, threshold-optimized)
    → Saved Model (pickle)
    → Prediction (frame-by-frame → bout filtering → CSV/video/ethogram)
    → Batch Analysis & Statistics
```

**Feature caching** is keyed to the video file (not the classifier), so multiple classifiers with different feature subsets can all use the same cached extraction.

## Multiple GUI Versions

The root directory contains several versions of the main file: `PixelPaws_GUI.py` (current), `PixelPaws_GUI_v2.py`, `PixelPaws_GUI_FIXED.py`. When making changes, always target `PixelPaws_GUI.py` unless directed otherwise.

## Key Dependencies

- **XGBoost ≥ 3.0** — classifier backend
- **h5py / tables** — reading DeepLabCut HDF5 output
- **OpenCV** — video frame extraction
- **PyTorch (optional)** — GPU-accelerated brightness extraction
- **tkinter** — GUI (standard library)
- **SHAP** — feature importance in evaluation output
