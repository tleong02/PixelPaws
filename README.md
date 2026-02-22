# PixelPaws — User Guide

Desktop GUI for automated animal behavior classification using DeepLabCut pose data and XGBoost. PixelPaws takes the `.h5` output from DeepLabCut, extracts kinematic and pixel-brightness features, trains an XGBoost classifier to detect a target behavior frame-by-frame, and then runs batch predictions and group statistics — all through a point-and-click interface with no coding required.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Setup Wizard](#project-setup-wizard)
3. [Preparing Your Data](#preparing-your-data)
4. [Crop for DLC Tool](#crop-for-dlc-tool)
5. [Tab-by-Tab Guide](#tab-by-tab-guide)
   - [Train](#train-tab)
   - [Predict](#predict-tab)
   - [Evaluate](#evaluate-tab)
   - [Analyze](#analyze-tab)
   - [Active Learning](#active-learning-tab)
6. [Project Folder Layout](#project-folder-layout)
7. [Requirements](#requirements)
8. [Attribution & License](#attribution--license)

---

## Installation

**GPU (recommended — faster brightness extraction):**
```bash
git clone https://github.com/rslivicki/PixelPaws.git
cd PixelPaws
pip install -r requirements_gpu.txt
python enable_pytorch_gpu.py
```

**CPU only:**
```bash
pip install -r requirements.txt
```

**Launch:**
```bash
python PixelPaws_GUI.py
```

---

## Project Setup Wizard

The Project Setup Wizard opens automatically on first launch. It walks through three steps:

**Step 1 — Project folder.** Choose an existing folder or create a new one. PixelPaws creates the standard subfolder structure inside it (`videos/`, `behavior_labels/`, `classifiers/`, etc.) and writes a `PixelPaws_project.json` config file.

**Step 2 — Behaviors.** Enter the names of the behaviors you plan to train classifiers for (e.g., `Flinch`, `Lick`, `Groom`). You can add or remove behaviors later from the Train tab. Built-in presets are available for common pain-assay behaviors.

**Step 3 — Body parts & features.** Choose which DeepLabCut body parts to use for feature extraction and whether to include pixel-brightness features. These choices are saved to the project config and can be changed later per classifier.

After finishing the wizard the main window appears with all five tabs ready to use.

---

## Preparing Your Data

### DeepLabCut output

PixelPaws reads the `.h5` file that DeepLabCut produces after analyzing a video. Place the video file and its corresponding `.h5` file in the `videos/` subfolder of your project. The files must share the same base name (e.g., `session01.mp4` and `session01DLC_resnet50_...h5`). PixelPaws matches them automatically by scanning for H5 files in the same folder as the video.

DLC CSV output (`.csv`) is also supported as a fallback.

### Label CSVs

Behavior labels are one row per video frame, one column per behavior (0 = absent, 1 = present):

```
frame,Flinch,Lick
0,0,0
1,1,0
2,1,0
3,0,1
```

Save label files in the `behavior_labels/` subfolder. The filename must contain the same session identifier as the video (e.g., `session01_labels.csv` for `session01.mp4`). PixelPaws discovers labels automatically during training and evaluation.

Label files can be created with any tool that produces this format — BORIS exported in the right mode, a custom script, or the built-in Active Learning loop.

### Feature caching

The first time PixelPaws processes a video it extracts all pose and brightness features and saves them to `features/` as a `.pkl` file keyed to the video. Subsequent runs skip extraction and load from cache, which makes retraining with different hyperparameters fast. Delete a cache file to force re-extraction (e.g., after changing body-part or ROI settings).

---

## Crop for DLC Tool

Before running DeepLabCut you may want to spatially crop your videos so that DLC focuses on the relevant region of the frame (e.g., the behavioral arena rather than the full camera view). The Crop for DLC tool in PixelPaws automates this.

**What it does:** Encodes a new video (or a batch of videos) containing only the pixels inside a rectangular region of interest, using FFmpeg under the hood. The crop offsets are saved to the project config so that all downstream coordinate math stays consistent.

**How to use it:**

1. Open **Tools → Crop for DLC** from the menu bar.
2. Select a video (single file) or a folder (batch mode).
3. A preview frame opens. Click and drag to draw the crop rectangle, or enter pixel values directly for X offset, Y offset, Width, and Height. Default values (X=286, Y=0, W=761, H=720) are pre-filled for a common rig setup and can be changed.
4. Click **Preview** to see the cropped region on the frame.
5. In batch mode, the tool first shows a confirmation dialog listing all videos it will process. Review the list and click **Proceed** to start encoding.
6. FFmpeg quality and codec settings are configurable in the dialog.
7. The crop offsets are written to `PixelPaws_project.json` so that brightness-feature extraction later maps pixel coordinates correctly.

Cropped videos are saved alongside the originals with a `_cropped` suffix. Run DeepLabCut on the cropped videos, then place the resulting H5 files in `videos/` as usual.

---

## Tab-by-Tab Guide

### Train Tab

The Train tab is where you build a classifier for a single behavior.

**Session discovery.** When you set a project folder, PixelPaws scans `videos/` and `behavior_labels/` to find matching triplets: video + DLC H5 + label CSV. Matched sessions appear in the session list. Check the sessions you want to include in training.

**Behavior name.** Type the exact column name from your label CSV (case-sensitive). A dropdown suggests behavior names found in the discovered label files.

**Feature settings.**
- *Pose features* — kinematic features computed from body-part coordinates: pairwise distances, joint angles, velocities at multiple timescales, and in-frame probability (confidence) scores.
- *Brightness features* — pixel brightness in square ROIs around selected body parts, extracted directly from video frames. Requires the video file. PyTorch/CUDA accelerates this significantly on GPU.
- Choose which body parts to include. Fewer body parts = faster extraction; more = richer features.

**Bout parameters.** Set the minimum bout duration (frames) and minimum inter-bout interval to merge adjacent detections. These are applied at prediction time, not during training.

**Start Training.** Click to begin. A training visualization window opens showing:
- Cross-validation F1 scores across folds
- Precision/recall curve
- Threshold optimization plot (the threshold that maximizes F1 on the validation set)
- Feature importance (SHAP values, computed on the final model)

The trained classifier is saved as a `.pkl` file to `classifiers/`. The filename encodes the behavior name, threshold, and a timestamp.

### Predict Tab

The Predict tab runs a trained classifier on a single video.

1. Select the **video file** (the original, not the cropped version — PixelPaws uses the path stored in the project config to find the H5 automatically, or you can specify it manually).
2. Select the **classifier** from the dropdown (populated from `classifiers/`).
3. Choose output options:
   - *Prediction CSV* — one row per frame with the predicted probability and binary label.
   - *Annotated video* — video with behavior label overlaid on each frame.
   - *Ethogram* — image showing behavior presence as a color raster across time.
   - *Statistics summary* — total time, bout count, mean bout duration, etc.
4. Click **Run Prediction**.

Output files are saved to `results/` with the session name as a prefix.

### Evaluate Tab

The Evaluate tab measures how well a classifier performs against hand-labeled ground truth.

1. Select the **classifier** to evaluate.
2. PixelPaws discovers available labeled test sessions automatically (same triplet logic as training). You can also point it to a specific session manually.
3. Click **Run Evaluation**.

Results include:
- Confusion matrix
- Precision, Recall, and F1 score at the trained threshold
- Precision-recall curve
- SHAP summary plot showing which features most influence predictions

All outputs are saved to `evaluations/` as a text report and image files.

### Analyze Tab

The Analyze tab performs cohort-level batch analysis: it loads prediction CSVs for multiple animals, bins behavior time into user-defined windows, and generates grouped statistics plots.

**Setup:**
1. Load a **key file** — a CSV or XLSX with at minimum `Subject` and `Treatment` columns (Subject values must match the prediction file names).
2. Select the **predictions folder** containing the per-animal prediction CSVs from the Predict tab (or from batch prediction).
3. Set the **time bin size** (e.g., 5 minutes) and which **metrics** to calculate (total time, bout count, mean bout duration, AUC, percent time, bout frequency).

**Optional graph types (Settings panel):**
- *Show individual animal traces on time course* — overlays faint per-animal lines behind the mean ± error, letting you see the spread in the raw data.
- *Show cumulative time plot* — running total of behavior time over the session, mean ± error per treatment.
- *Show latency to first bout* — for each animal, the time of the first bin with any detected behavior; displayed as a bar + scatter plot per treatment. Animals with no bouts are excluded and noted on the graph.
- *Formalin phase analysis* — splits the session into Acute (default 0–10 min) and Phase II (default 10–60 min) windows.
- *Statistical testing* — adds significance markers using two-way ANOVA (time × treatment) with Tukey HSD post-hoc for the time course, and one-way ANOVA or t-test for the bar graphs.

**Graph Settings dialog.** Click **Generate Graphs** to open the dialog before plotting:
1. *Time window* — maximum minutes to display.
2. *Error bars* — SEM or SD.
3. *Heatmap palette* — colormap for the time-bin heatmap tab.
4. *Groups to Include* — checkboxes for each treatment group. Unchecking a group immediately grays out its color swatch in the preview. Gradient mode redistributes colors across the remaining included groups.
5. *Treatment order* — drag items in the list to reorder left-to-right display.
6. *Colors* — choose individual colors per group (with a custom color picker) or use gradient mode to assign a colormap gradient across dose levels, with vehicle/control automatically rendered white with a black outline.

Each graph opens in a tabbed window with **Save Figure** (PNG/PDF/SVG at 300 dpi) and **Export Data** (CSV) buttons.

### Active Learning Tab

Active Learning minimizes the number of frames you need to hand-label by having the model identify the frames where it is most uncertain.

**Workflow:**
1. Train an initial classifier on a small set of labeled frames (or use a starter classifier).
2. On the Active Learning tab, point PixelPaws at an unlabeled video and run **Select Frames**. The model scores every frame and selects the most informative ones (highest prediction entropy or proximity to the 0.5 decision boundary).
3. Open the selected frames in the built-in labeling interface. Label each frame (1 = behavior present, 0 = absent).
4. The new labels are merged with the existing label set and the classifier is retrained.
5. Repeat until performance plateaus — typically 3–5 iterations reduces labeling effort by 50–80% compared to labeling a random sample.

The active learning label files are saved in `behavior_labels/` and are compatible with the standard training pipeline.

---

## Project Folder Layout

```
my_project/
├── videos/            # Videos (.mp4, .avi, etc.) + DLC .h5 files
├── behavior_labels/   # Label CSVs (frame × behavior columns)
├── classifiers/       # Trained .pkl classifiers
├── features/          # Cached feature files (safe to delete)
├── results/           # Prediction outputs (per-video CSVs, videos, ethograms)
├── analysis/          # Batch analysis outputs
├── evaluations/       # Evaluation reports + SHAP plots
└── PixelPaws_project.json
```

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| numpy | ≥ 2.0 | |
| pandas | ≥ 2.0 | |
| xgboost | ≥ 3.0 | Classifier backend |
| scikit-learn | ≥ 1.3 | Cross-validation, metrics |
| shap | ≥ 0.43 | Feature importance plots |
| statsmodels | ≥ 0.14 | Two-way ANOVA in Analyze tab |
| scipy | ≥ 1.10 | Post-hoc tests |
| opencv-python | ≥ 4.8 | Video frame extraction |
| h5py / tables | ≥ 3.8 | Reading DLC HDF5 output |
| matplotlib | ≥ 3.7 | Graphs |
| seaborn | ≥ 0.12 | Heatmaps |
| torch *(optional)* | ≥ 2.0 | GPU-accelerated brightness extraction |

---

## Attribution & License

Feature extraction is based on the BAREfoot algorithm:

> Barkai O, Zhang B, et al. *BAREfoot: Behavior with Automatic Recognition and Evaluation.* Cell Reports Methods, 2025. https://github.com/OmerBarkai/BAREfoot

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.

© 2026 rslivicki
