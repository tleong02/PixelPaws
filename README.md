# PixelPaws

Desktop GUI for automated animal behavior classification using DeepLabCut pose data and XGBoost.

---

## What it does

Takes DeepLabCut `.h5` output and trains classifiers to detect behaviors (flinching, licking, grooming, etc.) frame by frame — no coding required. Features combine pose kinematics (joint distances, angles, velocities) with pixel brightness around key body parts.

---

## Installation

**GPU (recommended):**
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

---

## Quick start

```bash
python PixelPaws_GUI.py
```

The Project Setup Wizard launches automatically. Follow the 3 steps, then use the tabs to train, predict, and analyze.

---

## Workflow

1. **Train** — select labeled sessions, pick a behavior, click Start Training
2. **Predict** — run a trained classifier on a new video; outputs CSV, annotated video, and ethogram
3. **Evaluate** — measure performance against hand-labeled ground truth; generates SHAP plots
4. **Analyze** — batch-process a cohort with time-binning and group statistics
5. **Active Learning** — let the model select which frames to label next

---

## Project folder layout

```
my_project/
├── videos/            # Videos + DLC .h5 files
├── behavior_labels/   # Label CSVs (frame × behavior columns)
├── classifiers/       # Trained .pkl classifiers
├── features/          # Cached features (safe to delete)
├── results/           # Prediction outputs
├── analysis/          # Batch analysis outputs
├── evaluations/       # Eval reports + SHAP plots
└── PixelPaws_project.json
```

Label CSV format — one row per frame, one column per behavior:
```
frame,Lick,Flinch
0,0,0
1,1,0
```

---

## Requirements

| Package | Version |
|---|---|
| numpy | ≥ 2.0 |
| pandas | ≥ 2.0 |
| xgboost | ≥ 3.0 |
| scikit-learn | ≥ 1.3 |
| shap | ≥ 0.43 |
| statsmodels | ≥ 0.14 |
| opencv-python | ≥ 4.8 |
| h5py / tables | ≥ 3.8 |
| matplotlib / seaborn | ≥ 3.7 / 0.12 |
| torch *(optional)* | ≥ 2.0 |

---

## Attribution

Feature extraction based on:
> Barkai O, Zhang B, et al. *BAREfoot: Behavior with Automatic Recognition and Evaluation.* Cell Reports Methods, 2025. https://github.com/OmerBarkai/BAREfoot

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.

© 2026 rslivicki
