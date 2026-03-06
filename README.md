# PixelPaws — User Guide

Desktop GUI for automated animal behavior classification using DeepLabCut pose data and XGBoost. PixelPaws takes the `.h5` output from DeepLabCut, extracts kinematic and pixel-brightness features, trains an XGBoost classifier to detect a target behavior frame-by-frame, and then runs batch predictions and group statistics.

## Demo

Example of automated behavior scoring using a scratching classifier. Behavior detections are overlaid frame-by-frame on the video.

<video src="media/260219_Rim_S1_cropped_labeled.mp4" controls width="720"></video>

---

> 📷 **Building the filming enclosure?** See [hardware.md](hardware.md) for the full bill of materials, 3D-printed enclosure files, wiring notes, and camera/lens specs.

## Table of Contents

1. [Installation](#installation)
2. [Project Setup Wizard](#project-setup-wizard)
3. [Preparing Your Data](#preparing-your-data)
4. [Labeling with BORIS](#labeling-with-boris)
5. [Crop for DLC Tool](#crop-for-dlc-tool)
6. [Tab-by-Tab Guide](#tab-by-tab-guide)
   - [Train](#train-tab)
   - [Predict](#predict-tab)
   - [Evaluate](#evaluate-tab)
   - [Analyze](#analyze-tab)
   - [Active Learning](#active-learning-tab)
   - [Tools](#tools-tab)
7. [Project Folder Layout](#project-folder-layout)
8. [Requirements](#requirements)
9. [Attribution & License](#attribution--license)

---

## Installation

```bash
git clone https://github.com/rslivicki/PixelPaws.git
cd PixelPaws
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

## Labeling with BORIS

[BORIS (Behavioral Observation Research Interactive Software)](https://www.boris.unito.it/) is a free, widely-used tool for scoring animal behavior from video. PixelPaws accepts BORIS exports directly through its built-in converter.

### Labeling workflow in BORIS

1. Open your video in BORIS and create an ethogram with the behavior(s) you want to train (e.g., `Flinch`, `Lick`).
2. Score each session using **START/STOP** events (for continuous behaviors) or **POINT** events (for instantaneous ones). Both are supported.
3. When finished, export the observation via **File → Export events → Save as CSV** (or TSV). Make sure the export includes at minimum the following columns:
   - **Behavior** — the behavior name
   - **Behavior type** — `START`, `STOP`, or `POINT`
   - **Time** — timestamp in seconds
   - **FPS** *(optional)* — if included, the converter reads it automatically; otherwise you enter it manually

### Converting BORIS labels to PixelPaws format

PixelPaws needs labels as a per-frame CSV (one row per video frame, one column per behavior, values 0 or 1). The BORIS converter handles this translation.

**Via the Tools tab or Tools menu → BORIS to PixelPaws:**

1. Click **Browse** and select your BORIS export file (CSV or TSV).
2. Click **🔍 Auto-Detect** to scan the file and pick the behavior you want to convert from a list, or type the behavior name directly.
3. Enter the video **FPS** (frames per second). If your BORIS export has an FPS column, leave the field blank and it will be read automatically.
4. Choose an **output directory** (defaults to the same folder as the BORIS file).
5. Click **🔄 Convert**.

The converter produces a file named `<boris_filename>_labels.csv` with a single column named after the behavior:

```
Flinch
0
0
1
1
1
0
```

Place this file in `behavior_labels/` inside your project folder (or in the same folder as the video — PixelPaws checks both). The session will then appear automatically in the Train and Evaluate session lists.

### Tips

- **One behavior per conversion run.** Run the converter once per behavior name. If you scored multiple behaviors in the same BORIS file, run it once for each and then merge the output CSVs column-by-column before training a multi-behavior session.
- **Frame alignment.** The converter multiplies each timestamp by FPS and rounds to the nearest frame. Using the exact FPS your camera recorded at (e.g., 60.0, not 59.94) avoids drift over long recordings.
- **Dense vs. sparse labels.** BORIS exports cover the full video duration with 0s between scored bouts — this is ideal for training. If you only have labels for a subset of frames (e.g., from Active Learning), the `SmartLabelManager` handles mixing the two automatically during training.

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
- *Brightness features* — pixel brightness in square ROIs around selected body parts, extracted directly from video frames. Requires the video file.
- Choose which body parts to include. Fewer body parts = faster extraction; more = richer features.

**Bout parameters.** Set the minimum bout duration (frames) and minimum inter-bout interval to merge adjacent detections. These are applied at prediction time, not during training.

**Start Training.** Click to begin. A training visualization window opens showing:
- Cross-validation F1 scores across folds
- Precision/recall curve
- Threshold optimization plot (the threshold that maximizes F1 on the validation set)
- Feature importance (SHAP values, computed on the final model)

The trained classifier is saved as a `.pkl` file to `classifiers/`. The filename encodes the behavior name, threshold, and a timestamp.

### Predict Tab

The Predict tab runs a trained classifier on a single video and reports behavior statistics for that session.

#### Inputs

| Field | Required | Notes |
|---|---|---|
| Classifier | Yes | Select a `.pkl` file from `classifiers/`. Click **View Classifier Info** to confirm the behavior name, threshold, and feature settings stored inside. |
| Video file | Yes | The original (uncropped) video. |
| DLC pose file | Yes | The `.h5` file produced by DeepLabCut for this video. Click **🔍 Auto-Find DLC File** to locate it automatically based on the video filename. |
| Features file | No | A pre-extracted `.pkl` cache from `features/`. If provided, feature extraction is skipped entirely, saving several minutes. |
| DLC config | No | The `config.yaml` from your DeepLabCut project. If provided and cropping was enabled in DLC, the crop offsets are read and applied to brightness feature extraction so pixel coordinates stay correct. Click **🔍 Auto-Find Config** to search for it automatically. |
| Human labels | No | A per-frame label CSV in PixelPaws format. Providing it records the path alongside the prediction for your own reference; for a full quantitative comparison (precision, recall, F1, SHAP) use the **Evaluate tab** instead. |
| Output folder | No | Defaults to the video's folder if left blank. |

#### What it does

1. Loads the classifier and reads its stored behavior name, threshold, body-part lists, bout-filtering parameters, and feature settings.
2. Extracts pose + brightness features if no cached file is provided (this is the slow step — expect 2–10 minutes depending on video length).
3. Runs the XGBoost model on every frame to produce a per-frame probability score.
4. Applies the trained threshold to produce binary frame labels (0 = absent, 1 = present).
5. Applies **bout filtering**: removes detections shorter than the minimum bout duration and fills gaps shorter than the maximum inter-bout interval. These parameters are stored in the classifier file, not set manually here.
6. Displays a summary in the results panel: total frames, frames with behavior detected, total behavior time in seconds and minutes.

#### Output options

| Option | Output file | Contents |
|---|---|---|
| Save frame-by-frame predictions (CSV) | `<video>_predictions.csv` | One row per frame: `frame`, `prediction` (0/1), `probability` (raw score). This file is what the **Analyze tab** consumes for batch statistics. |
| Create labeled video | `<video>_labeled.mp4` | Video with the behavior label overlaid on each frame. Slower to generate. |
| Save behavior summary statistics | `<video>_summary.txt` | Total time, bout count, mean/max bout duration, percentage of session. |
| Generate ethogram plots | `<video>_ethogram.png` | Color raster showing behavior presence across the full session timeline. |

#### Comparing against human labels

The Predict tab includes an optional **Human Labels** field where you can point to a ground-truth label CSV for the session. This path is stored with the prediction for reference but the Predict tab itself does not compute agreement metrics. For a full evaluation — confusion matrix, precision, recall, F1 at the trained threshold, precision-recall curve, and SHAP feature importance — use the **Evaluate tab**, which is designed specifically for that purpose.

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

### Tools Tab

The Tools tab provides quick access to a set of utilities that complement the main pipeline:

| Tool | Description |
|---|---|
| **Video Preview** | Play a video alongside its prediction CSV — predictions are overlaid on each frame so you can visually verify classifier output. |
| **Auto-Label Assistant** | Steps through frames at a configurable interval and prompts you to label each one; outputs a label CSV in the PixelPaws format. |
| **Data Quality Checker** | Scans all label CSVs in the project for common issues: class imbalance, missing frames, duplicate rows, and sessions with very few positive examples. |
| **Brightness Diagnostics** | Plots mean brightness for each body-part ROI over time for a selected video; useful for detecting lighting artefacts or ROI misalignment. |
| **Feature File Inspector** | Opens a cached feature `.pkl` and shows column names, shapes, and summary statistics — helpful for debugging feature extraction. |
| **Brightness Preview** | Shows a single video frame with the brightness ROI rectangles drawn around each selected body part so you can confirm they are positioned correctly. |
| **Correct Crop Offset (Single / Batch)** | If videos were cropped before DLC and the crop offsets changed between sessions, these tools remap the stored offsets in prediction CSVs so that coordinates stay consistent. |
| **Crop Video for DLC** | Spatially crops a video (or batch of videos) to a user-defined rectangle and saves the result for DLC analysis. Crop offsets are written to the project config. See [Crop for DLC Tool](#crop-for-dlc-tool) above. |
| **Generate Ethogram** | Creates an ethogram image (color raster of behavior presence over time) from any prediction CSV; can be saved as PNG. |
| **Training Visualization** | Re-opens the training visualization window for the most recently trained classifier (cross-validation scores, precision-recall curve, SHAP summary). |
| **BORIS to PixelPaws** | Converts a BORIS event-log export (CSV) into the frame-indexed label CSV format that PixelPaws expects, using the video frame rate to map timestamps to frame numbers. |
| **Optimize Parameters** | Grid-searches bout-filtering parameters (minimum bout duration, minimum inter-bout interval) to maximize agreement with hand labels on a selected session. |
| **Feature Extraction** | Runs feature extraction manually on a selected video + H5 pair and saves the result to `features/`; useful for pre-caching before a training run. |
| **Theme Switcher** | Toggles between light and dark UI themes. |

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

---

## Attribution & License

Feature extraction is based on the BAREfoot algorithm:

> Barkai O, Zhang B, et al. *BAREfoot: Behavior with Automatic Recognition and Evaluation.* Cell Reports Methods, 2025. https://github.com/OmerBarkai/BAREfoot

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.

© 2026 rslivicki
