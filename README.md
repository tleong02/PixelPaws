# PixelPaws 🐾

> Automated animal behavior classification for behavioral neuroscience research.

PixelPaws is a desktop GUI that takes [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) pose estimation output and trains XGBoost classifiers to automatically detect and quantify animal behaviors — flinching, licking, grooming, and more — frame by frame, with no coding required.

![Main window overview](docs/images/main_window.png)

---

## Table of Contents

- [Why PixelPaws?](#why-pixelpaws)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tabs & Workflow](#tabs--workflow)
  - [Project Setup Wizard](#project-setup-wizard)
  - [Train](#train-tab)
  - [Predict](#predict-tab)
  - [Evaluate](#evaluate-tab)
  - [Analyze](#analyze-tab)
  - [Active Learning](#active-learning-tab)
- [Project Folder Layout](#project-folder-layout)
- [Scripted / Headless Use](#scripted--headless-use)
- [Requirements](#requirements)

---

## Why PixelPaws?

Manual scoring of animal behavior videos is slow, subjective, and doesn't scale. PixelPaws automates this by combining two complementary feature types:

- **Pose kinematics** — pairwise distances, joint angles, and velocities computed from DeepLabCut keypoints across multiple timescales
- **Pixel brightness** — raw pixel intensity in small ROIs around key body parts (e.g. paw, snout) directly from the video frames

Together these give the classifier both *where the animal is* and *what the pixels look like*, making it robust to behaviors that involve subtle local changes (licking, flinching) as well as large postural shifts (rearing, grooming).

---

## Features

| Feature | Details |
|---|---|
| Project wizard | Guided setup — define behaviors, body parts, video format |
| Feature caching | Extracted features cached per video; reused across all classifiers |
| XGBoost training | Balanced, k-fold cross-validated, threshold-optimized |
| SHAP explanations | Per-feature importance plots for every trained classifier |
| Prediction | Frame-by-frame output → bout filtering → CSV / annotated video / ethogram |
| Batch analysis | Process entire cohorts with time-binning and group statistics |
| Model evaluation | Compare predictions against hand-labeled ground truth |
| Active learning | Smart frame selection to minimize labeling effort |
| GPU acceleration | Optional PyTorch/CUDA support for ~4.5× faster brightness extraction |

---

## Installation

### Prerequisites

- Python 3.7+ (3.11 recommended)
- DeepLabCut H5 output files from your videos
- NVIDIA GPU optional but recommended

### GPU version (recommended)

```bash
git clone https://github.com/rslivicki/PixelPaws.git
cd PixelPaws
pip install -r requirements_gpu.txt
python enable_pytorch_gpu.py
```

### CPU only

```bash
git clone https://github.com/rslivicki/PixelPaws.git
cd PixelPaws
pip install -r requirements.txt
```

### Verify GPU (optional)

```bash
python enable_pytorch_gpu.py --check
```

Expected output:
```
PyTorch Status:
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
✓ PyTorch with CUDA is ready!
```

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for CUDA version troubleshooting.

---

## Quick Start

```bash
python PixelPaws_GUI.py
```

The **Project Setup Wizard** launches automatically. Follow the 3 steps, then use the tabs to train, predict, and analyze.

---

## Tabs & Workflow

### Project Setup Wizard

Runs on first launch. Takes ~1 minute to configure.

![Project setup wizard step 1](docs/images/wizard_step1.png)

**Step 1 — Choose Project**
Create a new project folder or open an existing one. PixelPaws creates the standard subfolder layout automatically.

![Project setup wizard step 2](docs/images/wizard_step2.png)

**Step 2 — Configure**
- Set your video format (`.mp4`, `.avi`, etc.)
- Define your **behaviors** — add as many as you like (e.g. *Lick*, *Flinch*, *Groom*). Each behavior will get its own classifier.
- Choose **brightness body parts** — the body parts whose pixel ROIs will be extracted. Defaults to `hrpaw`, `hlpaw`, `snout`. Keeping this to 3–5 parts is recommended as brightness extraction is the slowest step.
- Set the square ROI size in pixels (default 20 px).

> **Tip:** Click 🔄 to auto-detect body part names from your DLC `.h5` files.

![Project setup wizard step 3](docs/images/wizard_step3.png)

**Step 3 — Extract Features**
Optionally extract pose + brightness features for all videos in `videos/` right away, or skip and do it later from the Train tab.

---

### Train Tab

![Train tab](docs/images/train_tab.png)

Train a binary XGBoost classifier for a single behavior.

1. Select your **project folder** — sessions are auto-discovered (video + DLC H5 + label CSV triplets)
2. Set the **behavior name** (must match a column in your label CSVs)
3. Configure **feature settings** — pose only, brightness only, or both
4. Adjust **bout parameters** — minimum bout length, gap-fill duration
5. Click **Start Training**

A live training visualization window shows cross-validation progress, ROC curves, and threshold optimization in real time.

![Training visualization window](docs/images/training_viz.png)

**Output:** `classifiers/<behavior_name>.pkl`

#### Hyperparameter defaults

| Parameter | Default | Notes |
|---|---|---|
| n_estimators | 1700 |
| max_depth | 6 | |
| learning_rate | 0.01 | |
| colsample_bytree | 0.2 | |
| k-folds | 5 | Stratified |

---

### Predict Tab

![Predict tab](docs/images/predict_tab.png)

Run a trained classifier on a new video.

1. Select a **video file** and its corresponding **DLC H5 file**
2. Choose a **classifier** from the dropdown (auto-populated from `classifiers/`)
3. Select outputs: CSV predictions, annotated video, summary stats, ethogram
4. Click **Run Prediction**

![Ethogram output](docs/images/ethogram.png)

---

### Evaluate Tab

![Evaluate tab](docs/images/evaluate_tab.png)

Measure classifier performance against hand-labeled test data.

- Reports precision, recall, F1, and accuracy
- Generates a SHAP feature importance plot
- Optionally saves per-frame predictions alongside ground truth

![SHAP feature importance](docs/images/shap_plot.png)

---

### Analyze Tab

![Analyze tab](docs/images/analyze_tab.png)

Batch-process an entire cohort of videos.

- Runs one or more classifiers across all videos in a folder
- Bins results into time windows (e.g. 60-second epochs)
- Exports per-animal and group-level summary CSVs
- Generates ethograms for the whole cohort

---

### Active Learning Tab

![Active learning tab](docs/images/active_learning_tab.png)

Minimize labeling effort by letting the model tell you which frames are most informative to label next.

1. Train an initial classifier on a small labeled set
2. Run active learning to score all unlabeled frames by uncertainty
3. Label only the suggested frames in BORIS or your annotation tool
4. Retrain — repeat until performance plateaus

---

## Project Folder Layout

```
my_project/
├── videos/               # Raw videos + DLC .h5 files
│   ├── session1.mp4
│   └── session1DLC_resnet50_...h5
├── behavior_labels/      # Label CSVs (one column per behavior, one row per frame)
│   └── session1_labels.csv
├── classifiers/          # Trained classifiers
│   └── Lick.pkl
├── features/             # Cached feature files (auto-generated, safe to delete)
├── results/              # Prediction CSVs and annotated videos
├── analysis/             # Batch analysis outputs
├── evaluations/          # Evaluation reports and SHAP plots
└── PixelPaws_project.json
```

### Label CSV format

```
frame,Lick,Flinch,Groom
0,0,0,0
1,0,1,0
2,1,0,0
...
```

One row per video frame. Column names must match the behavior names used during training.

---

## Scripted / Headless Use

Feature extraction and prediction can be run from Python without the GUI:

```python
from pixelpaws_easy import extract_all_features_auto

# Extract pose + brightness features for one session
extract_all_features_auto(
    video_path="videos/session1.mp4",
    dlc_path="videos/session1DLC_resnet50.h5",
    bp_pixbrt_list=["hrpaw", "hlpaw", "snout"],
    square_size=20,
)
```

```python
from classifier_training import BehaviorClassifier
import pandas as pd

clf = BehaviorClassifier()
clf.load("classifiers/Lick.pkl")

features = pd.read_parquet("features/session1_features.parquet")
predictions, probabilities = clf.predict(features)
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 2.0 | Numerical computing |
| `pandas` | ≥ 2.0 | Data handling |
| `xgboost` | ≥ 3.0 | Classifier |
| `scikit-learn` | ≥ 1.3 | Cross-validation, metrics |
| `shap` | ≥ 0.43 | Feature importance |
| `opencv-python` | ≥ 4.8 | Video frame extraction |
| `h5py` / `tables` | ≥ 3.8 | DLC HDF5 files |
| `matplotlib` / `seaborn` | ≥ 3.7 / 0.12 | Plotting |
| `torch` *(optional)* | ≥ 2.0 | GPU brightness extraction |

---

## Adding Screenshots

Screenshots belong in `docs/images/`. To add them:

```
docs/
└── images/
    ├── main_window.png
    ├── wizard_step1.png
    ├── wizard_step2.png
    ├── wizard_step3.png
    ├── train_tab.png
    ├── training_viz.png
    ├── predict_tab.png
    ├── ethogram.png
    ├── evaluate_tab.png
    ├── shap_plot.png
    ├── analyze_tab.png
    └── active_learning_tab.png
```

Once the files are in place the images will render automatically on GitHub.

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use; commercial use requires permission.

© 2026 rslivicki
