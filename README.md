# PixelPaws

**Automated animal behavior classification for behavioral neuroscience research.**

PixelPaws is a desktop GUI application that takes [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) pose estimation data and trains XGBoost classifiers to automatically detect and quantify animal behaviors — flinching, licking, grooming, and more — frame by frame.

---

## Features

- **Project wizard** — guided setup to configure behaviors, body parts, and video settings
- **Feature extraction** — pose kinematics (distances, angles, velocities) + pixel brightness around body parts, cached per video
- **XGBoost training** — balanced, k-fold cross-validated, threshold-optimized; matches the BAREfoot algorithm
- **SHAP feature importance** — understand what drives each classifier's decisions
- **Prediction** — run any trained classifier on new videos; output CSV, annotated video, or ethogram
- **Batch analysis** — process entire cohorts with time-binning and group statistics
- **Evaluation** — compare classifier predictions against hand-labeled ground truth
- **Active learning** — intelligently select frames to label, minimizing annotation effort

---

## Requirements

- Python 3.7+ (3.11 recommended)
- Windows / macOS / Linux
- NVIDIA GPU optional but recommended for brightness extraction (~4.5× speedup)

---

## Installation

### GPU version (recommended)
```bash
pip install -r requirements_gpu.txt
python enable_pytorch_gpu.py
```

### CPU only
```bash
pip install -r requirements.txt
```

### Launch
```bash
python PixelPaws_GUI.py
```

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for troubleshooting and version details.

---

## Workflow

```
Video + DeepLabCut H5/CSV
    → Feature extraction  (pose kinematics + pixel brightness, cached)
    → XGBoost training    (balanced, cross-validated, threshold-optimized)
    → Saved classifier    (.pkl)
    → Prediction          (frame-by-frame → bout filtering → CSV / video / ethogram)
    → Batch analysis      (time bins, group statistics)
```

### Project folder layout
```
my_project/
  videos/           — videos + DLC .h5 files
  behavior_labels/  — label CSVs
  classifiers/      — trained .pkl classifiers
  features/         — cached feature files (auto-generated)
  results/          — prediction outputs
  analysis/         — batch analysis outputs
  PixelPaws_project.json
```

---

## Usage

1. **Launch** `python PixelPaws_GUI.py`
2. **Project wizard** — create or open a project folder, define your behaviors and brightness body parts
3. **Train tab** — point to your labeled sessions and train a classifier
4. **Predict tab** — run the classifier on new videos
5. **Evaluate tab** — check performance against held-out labeled data
6. **Analyze tab** — batch-process a cohort and export statistics

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `xgboost ≥ 3.0` | Classifier backend |
| `h5py` / `tables` | Reading DeepLabCut HDF5 output |
| `opencv-python` | Video frame extraction |
| `shap` | Feature importance |
| `torch` *(optional)* | GPU-accelerated brightness extraction |
| `tkinter` | GUI (Python standard library) |

---

## Scripted / headless use

```python
from pixelpaws_easy import extract_all_features_auto

extract_all_features_auto(
    video_path="session.mp4",
    dlc_path="sessionDLC_resnet50.h5",
    bp_pixbrt_list=["hrpaw", "hlpaw", "snout"],
    square_size=20,
)
```

---

## License

MIT
