"""
active_learning_v2.py — PixelPaws Active Learning v2
=====================================================
Inspired by A-SOiD (Tillmann et al., Nature Methods 2024).

New features vs v1:
- Confidence histogram inspection with adjustable threshold
- Learning curve tracking (train F1 + CV F1) with JSON persistence
- Auto-convergence detection
- Temporal label propagation (cosine similarity)
- Post-AL sub-behavior discovery (UMAP + HDBSCAN)
"""

import os
import json
import pickle
import threading
import traceback
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2

try:
    import ttkbootstrap as ttk
    _TTKBOOTSTRAP = True
except ImportError:
    from tkinter import ttk
    _TTKBOOTSTRAP = False

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _align_features(model, features: np.ndarray, feature_cols) -> np.ndarray:
    """Reindex numpy feature array to match model.feature_names_in_.
    - If model has no feature_names_in_ or feature_cols is None: return as-is.
    - Already-aligned: return as-is (fast path).
    - Missing columns filled with 0.0; extra columns dropped.
    """
    if not hasattr(model, 'feature_names_in_') or feature_cols is None:
        return features
    model_cols = list(model.feature_names_in_)
    if list(feature_cols) == model_cols:
        return features
    df = pd.DataFrame(features, columns=feature_cols)
    return df.reindex(columns=model_cols, fill_value=0.0).values.astype(np.float32)


from ui_utils import ToolTip, _bind_tight_layout_on_resize
from dialogs import ConfidenceHistogramDialog


# Optional UMAP + HDBSCAN
try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    UMAP_HDBSCAN_AVAILABLE = True
except ImportError:
    UMAP_HDBSCAN_AVAILABLE = False

# Optional sklearn metrics
try:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def _make_bout_groups(frame_indices, min_bout=1):
    """
    Assign group IDs to labeled frame indices so that a new group
    starts whenever consecutive frames are more than min_bout apart.
    Returns int array same length as frame_indices.
    """
    if len(frame_indices) == 0:
        return np.array([], dtype=int)
    groups = np.zeros(len(frame_indices), dtype=int)
    gid = 0
    for i in range(1, len(frame_indices)):
        if frame_indices[i] - frame_indices[i - 1] > min_bout:
            gid += 1
        groups[i] = gid
    return groups


# Session discovery
try:
    from evaluation_tab import find_session_triplets
    _FIND_SESSIONS_AVAILABLE = True
except ImportError:
    find_session_triplets = None
    _FIND_SESSIONS_AVAILABLE = False


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class ALIterationRecord:
    iteration: int
    n_labeled_total: int
    n_positive: int
    train_f1: float
    oof_f1: Optional[float]
    n_below_threshold: int
    timestamp: str
    oof_precision: Optional[float] = None
    oof_recall:    Optional[float] = None


@dataclass
class BoutCandidate:
    """A contiguous run of uncertain frames presented as a single labeling unit."""
    start_frame: int        # core uncertain region start
    end_frame: int          # core uncertain region end (inclusive)
    clip_start: int         # clip start shown to user (with context padding)
    clip_end: int           # clip end shown to user (with context padding)
    mean_proba: float
    mean_uncertainty: float
    duration_frames: int    # length of core bout
    session_idx: int = 0
    video_path: str = ""


# ============================================================================
# UncertaintyEngineV2
# ============================================================================

class UncertaintyEngineV2:
    def __init__(self, min_frame_spacing=30):
        self.min_frame_spacing = min_frame_spacing
        self._last_probas = None

    def score_all_frames(self, model, features) -> np.ndarray:
        probas = model.predict_proba(features)[:, 1]
        self._last_probas = probas
        return probas

    def find_uncertain_frames(self, probas, current_labels, n_suggestions,
                               confidence_threshold=0.30,
                               avoid_labeled=True) -> Tuple[np.ndarray, np.ndarray]:
        # Eligible: within threshold of 0.5
        uncertainty = 1.0 - np.abs(probas - 0.5) * 2
        eligible_mask = np.abs(probas - 0.5) * 2 < confidence_threshold

        scores = uncertainty.copy()

        if avoid_labeled:
            labeled_mask = current_labels >= 0
            scores[labeled_mask] *= 0.1

        # Only consider eligible frames
        scores[~eligible_mask] = 0.0

        selected = []
        suppressed = np.zeros(len(probas), dtype=bool)

        for _ in range(n_suggestions):
            if scores.max() <= 0:
                break
            idx = int(np.argmax(scores))
            selected.append(idx)
            # Suppress neighbors
            lo = max(0, idx - self.min_frame_spacing)
            hi = min(len(scores), idx + self.min_frame_spacing + 1)
            scores[lo:hi] = 0.0

        selected = np.array(selected, dtype=int)
        if len(selected) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        return selected, probas[selected]

    def find_uncertain_bouts(self, probas, current_labels, n_bouts=10,
                              confidence_threshold=0.30,
                              min_bout_frames=5, context_frames=30,
                              max_bout_frames=300,
                              class_balanced: bool = True,
                              diversity_radius: int = 0,
                              _stats=None) -> List[BoutCandidate]:
        """Find contiguous unlabeled segments, ranked by classifier uncertainty.

        Long runs are subdivided into windows of at most max_bout_frames so the
        user is never shown a clip that spans the entire video.
        """
        # If labels extend beyond features cache, fill with 0.5 (max uncertainty)
        n_labels = len(current_labels)
        if n_labels > len(probas):
            probas = np.concatenate([probas, np.full(n_labels - len(probas), 0.5)])

        # Base: unlabeled frames (labels == -1)
        base_mask = current_labels < 0
        n = len(probas)

        # Group consecutive unlabeled frames into runs
        runs = []
        in_run = False
        run_start = 0
        for i in range(n):
            if base_mask[i] and not in_run:
                run_start = i
                in_run = True
            elif not base_mask[i] and in_run:
                runs.append((run_start, i - 1))
                in_run = False
        if in_run:
            runs.append((run_start, n - 1))

        bouts = []
        n_too_short = 0
        for (run_start, run_end) in runs:
            dur = run_end - run_start + 1
            if dur < min_bout_frames:
                n_too_short += 1
                continue

            if dur <= max_bout_frames:
                # Short enough: use as a single bout
                windows = [(run_start, run_end)]
            else:
                # Subdivide into non-overlapping windows of max_bout_frames
                windows = []
                pos = run_start
                while pos <= run_end:
                    w_end = min(pos + max_bout_frames - 1, run_end)
                    if w_end - pos + 1 >= min_bout_frames:
                        windows.append((pos, w_end))
                    pos += max_bout_frames

            for (start, end) in windows:
                clip_start = max(0, start - context_frames)
                clip_end = min(n - 1, end + context_frames)
                seg_probas = probas[start:end + 1]
                mean_proba = float(np.mean(seg_probas))
                mean_uncertainty = float(np.mean(1.0 - np.abs(seg_probas - 0.5) * 2))
                bouts.append(BoutCandidate(
                    start_frame=start, end_frame=end,
                    clip_start=clip_start, clip_end=clip_end,
                    mean_proba=mean_proba, mean_uncertainty=mean_uncertainty,
                    duration_frames=end - start + 1,
                ))

        if _stats is not None:
            _stats['n_runs'] = len(runs)
            _stats['n_too_short'] = n_too_short

        bouts.sort(key=lambda b: -b.mean_uncertainty)

        def _apply_diversity(bucket, radius):
            if radius <= 0:
                return bucket
            kept = []
            for b in bucket:  # already sorted by descending uncertainty
                if not any(abs(b.start_frame - s.start_frame) < radius for s in kept):
                    kept.append(b)
            return kept

        if not class_balanced:
            return bouts[:n_bouts]

        # Round-robin: alternate positive-predicted and negative-predicted bouts
        pos_bouts = _apply_diversity([b for b in bouts if b.mean_proba >= 0.5], diversity_radius)
        neg_bouts = _apply_diversity([b for b in bouts if b.mean_proba < 0.5],  diversity_radius)
        selected, i, j = [], 0, 0
        while len(selected) < n_bouts and (i < len(pos_bouts) or j < len(neg_bouts)):
            if i < len(pos_bouts):
                selected.append(pos_bouts[i]); i += 1
            if j < len(neg_bouts) and len(selected) < n_bouts:
                selected.append(neg_bouts[j]); j += 1
        return selected


# ============================================================================
# LabelPropagator
# ============================================================================

class LabelPropagator:
    def __init__(self, n_neighbors=5, max_frame_spread=30, similarity_threshold=0.92):
        self.n_neighbors = n_neighbors
        self.max_frame_spread = max_frame_spread
        self.similarity_threshold = similarity_threshold

    def propagate(self, labeled_frame: int, label: int,
                  features: np.ndarray, current_labels: np.ndarray) -> dict:
        result = {}
        v = features[labeled_frame].astype(float)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return result

        lo = max(0, labeled_frame - self.max_frame_spread)
        hi = min(len(features), labeled_frame + self.max_frame_spread + 1)

        candidates = []
        for f in range(lo, hi):
            if f == labeled_frame:
                continue
            if current_labels[f] != -1:
                continue
            u = features[f].astype(float)
            u_norm = np.linalg.norm(u)
            if u_norm == 0:
                continue
            sim = np.dot(v, u) / (v_norm * u_norm)
            if sim >= self.similarity_threshold:
                candidates.append((sim, f))

        # Sort descending by similarity, take top n_neighbors
        candidates.sort(key=lambda x: -x[0])
        for _, f in candidates[:self.n_neighbors]:
            result[f] = label

        return result


# ============================================================================
# LearningCurveTracker
# ============================================================================

class LearningCurveTracker:
    def __init__(self):
        self.records: List[ALIterationRecord] = []

    def record(self, model, X_train, y_train, n_below_threshold,
               labels_array=None, min_bout=1) -> ALIterationRecord:
        n_labeled = len(y_train)
        n_positive = int((y_train == 1).sum())

        # Train F1
        if SKLEARN_AVAILABLE and n_labeled > 0:
            preds = model.predict(X_train)
            try:
                train_f1 = float(f1_score(y_train, preds, zero_division=0))
            except Exception:
                train_f1 = 0.0
        else:
            train_f1 = 0.0

        # OOF F1
        oof_f1 = None
        oof_precision = None
        oof_recall = None
        if SKLEARN_AVAILABLE and n_labeled >= 30:
            n_neg = n_labeled - n_positive
            if min(n_positive, n_neg) >= 3:
                try:
                    from sklearn.model_selection import GroupKFold
                    from xgboost import XGBClassifier
                    _cv_mode = 'frame'
                    if labels_array is not None:
                        _lab_idx = np.where(labels_array >= 0)[0]
                        _grps = _make_bout_groups(_lab_idx, min_bout)
                        _n_grps = int(_grps.max()) + 1 if len(_grps) > 0 else 0
                    else:
                        _n_grps = 0

                    spw = float(n_neg / max(n_positive, 1))
                    oof_true, oof_proba = [], []
                    if _n_grps >= 3:
                        # Bout-aware: no subsampling (keep groups intact)
                        _gkf = GroupKFold(n_splits=3)
                        for tr_i, val_i in _gkf.split(X_train, y_train, groups=_grps):
                            clf = XGBClassifier(n_estimators=100, max_depth=6,
                                                learning_rate=0.1, scale_pos_weight=spw,
                                                random_state=42, verbosity=0)
                            clf.fit(X_train[tr_i], y_train[tr_i])
                            oof_proba.extend(clf.predict_proba(X_train[val_i])[:, 1].tolist())
                            oof_true.extend(y_train[val_i].tolist())
                        _cv_mode = 'bout'
                    else:
                        # Fallback: random frame subsample + StratifiedKFold
                        if n_labeled > 500:
                            rng = np.random.RandomState(42)
                            idx = rng.choice(n_labeled, 500, replace=False)
                            Xs, ys = X_train[idx], y_train[idx]
                        else:
                            Xs, ys = X_train, y_train
                        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        for tr_i, val_i in skf.split(Xs, ys):
                            clf = XGBClassifier(n_estimators=100, max_depth=6,
                                                learning_rate=0.1, scale_pos_weight=spw,
                                                random_state=42, verbosity=0)
                            clf.fit(Xs[tr_i], ys[tr_i])
                            oof_proba.extend(clf.predict_proba(Xs[val_i])[:, 1].tolist())
                            oof_true.extend(ys[val_i].tolist())
                    oof_proba_arr = np.array(oof_proba)
                    oof_true_arr  = np.array(oof_true)
                    best_f1 = 0.0
                    best_thresh = 0.5
                    for thresh in np.arange(0.05, 0.96, 0.05):
                        preds = (oof_proba_arr >= thresh).astype(int)
                        score = float(f1_score(oof_true_arr, preds, zero_division=0))
                        if score > best_f1:
                            best_f1 = score
                            best_thresh = float(thresh)
                    oof_f1 = best_f1
                    from sklearn.metrics import precision_score, recall_score
                    best_preds = (oof_proba_arr >= best_thresh).astype(int)
                    oof_precision = float(precision_score(oof_true_arr, best_preds, zero_division=0))
                    oof_recall    = float(recall_score(oof_true_arr,    best_preds, zero_division=0))
                except Exception:
                    oof_f1 = None
                    oof_precision = None
                    oof_recall = None

        rec = ALIterationRecord(
            iteration=len(self.records),
            n_labeled_total=n_labeled,
            n_positive=n_positive,
            train_f1=train_f1,
            oof_f1=oof_f1,
            n_below_threshold=n_below_threshold,
            timestamp=datetime.now().isoformat(),
            oof_precision=oof_precision if oof_f1 is not None else None,
            oof_recall=oof_recall    if oof_f1 is not None else None,
        )
        self.records.append(rec)
        return rec

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.records = []
        for d in data:
            if 'cv_f1' in d and 'oof_f1' not in d:
                d['oof_f1'] = d.pop('cv_f1')
            d.setdefault('oof_precision', None)
            d.setdefault('oof_recall', None)
            self.records.append(ALIterationRecord(**d))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                'iteration', 'n_labeled_total', 'n_positive',
                'train_f1', 'oof_f1', 'n_below_threshold', 'timestamp'
            ])
        return pd.DataFrame([asdict(r) for r in self.records])


# ============================================================================
# ALSessionV2
# ============================================================================

class ALSessionV2:
    def __init__(self, labels_csv: str, video_path: str, features_cache: str,
                 min_frame_spacing: int = 30):
        self.labels_csv = labels_csv
        self.video_path = video_path
        self.features_cache = features_cache

        # Load labels from CSV
        df = pd.read_csv(labels_csv)
        self.behavior_name = df.columns[0]
        raw = df[self.behavior_name].values
        # Treat NaN as -1 (unlabeled)
        self._labels = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))

        # Load features
        with open(features_cache, 'rb') as f:
            feats = pickle.load(f)
        if isinstance(feats, pd.DataFrame):
            self._feature_cols = list(feats.columns)
            feats = feats.values
        else:
            self._feature_cols = None
        self._features = feats.astype(np.float32)

        # Sync length: extend labels with -1 if features cover more frames than CSV
        n_feat = len(self._features)
        n_csv  = len(self._labels)
        if n_feat > n_csv:
            # Features cover more frames than CSV: extend labels with -1 (unlabeled)
            self._labels = np.concatenate([
                self._labels,
                np.full(n_feat - n_csv, -1, dtype=int)
            ])
            self._n_csv_rows = n_csv   # remember original CSV length for save logic
            self._truncation_warning = None
        elif n_feat < n_csv:
            # Features shorter than CSV: can only score up to features length
            truncated_tail = self._labels[n_feat:]
            n_truncated_labeled = int(np.sum(truncated_tail >= 0))
            self._labels = self._labels[:n_feat]
            self._n_csv_rows = n_feat
            self._truncation_warning = (n_csv - n_feat, n_truncated_labeled)
        else:
            self._n_csv_rows = n_csv
            self._truncation_warning = None
        # Now len(_labels) == len(_features) always

        self._engine = UncertaintyEngineV2(min_frame_spacing)
        self._propagator = LabelPropagator()
        self.tracker = LearningCurveTracker()
        self._iteration = 0
        self._seen_bouts: set = set()   # (start_frame, end_frame) of shown bouts

    def get_labels_and_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        labeled_mask = self._labels >= 0
        X_labeled = self._features[labeled_mask]
        y_labeled = self._labels[labeled_mask]
        return labeled_mask, X_labeled, y_labeled, self._features

    def train_model(self):
        labeled_mask, X_labeled, y_labeled, _ = self.get_labels_and_features()
        if len(X_labeled) == 0:
            raise ValueError("No labeled frames available for training.")
        n_positive = int((y_labeled == 1).sum())
        if n_positive == 0:
            raise ValueError("No positive-labeled frames. Label at least one positive example.")
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, verbosity=0)
        if self._feature_cols:
            model.fit(pd.DataFrame(X_labeled, columns=self._feature_cols), y_labeled)
        else:
            model.fit(X_labeled, y_labeled)
        return model

    def get_full_probas(self, model) -> np.ndarray:
        feats = _align_features(model, self._features, self._feature_cols)
        return self._engine.score_all_frames(model, feats)

    def _adaptive_max_bout_frames(self, default: int = 300) -> int:
        """90th-percentile duration of labeled positive runs, or default if too few samples."""
        positive = (self._labels == 1)
        if not positive.any():
            return default
        durations = []
        run_len = 0
        for v in positive:
            if v:
                run_len += 1
            elif run_len:
                durations.append(run_len)
                run_len = 0
        if run_len:
            durations.append(run_len)
        if len(durations) < 3:
            return default
        p90 = int(np.percentile(durations, 90))
        return max(30, min(p90, 1000))

    def retrain_and_snapshot(self, confidence_threshold: float = 0.3,
                              snapshot_dir: str = None) -> dict:
        """Retrain from current labels, record to tracker, optionally save snapshot pkl."""
        import pickle, os, time

        prev_cv = self.tracker.records[-1].cv_f1 if self.tracker.records else None

        model = self.train_model()
        probas = self.get_full_probas(model)
        n_below = int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
        labeled_mask, X_labeled, y_labeled, _ = self.get_labels_and_features()
        record = self.tracker.record(model, X_labeled, y_labeled, n_below)

        snapshot_path = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            fname = f"al_iter{record.iteration}_{ts}.pkl"
            snapshot_path = os.path.join(snapshot_dir, fname)
            with open(snapshot_path, 'wb') as f:
                pickle.dump({
                    'clf_model': model,
                    'iteration': record.iteration,
                    'train_f1': record.train_f1,
                    'cv_f1': record.cv_f1,
                    'n_labeled_total': record.n_labeled_total,
                }, f)

        delta_cv = None
        if prev_cv is not None and record.cv_f1 is not None:
            delta_cv = record.cv_f1 - prev_cv

        return {
            'model': model, 'probas': probas, 'record': record,
            'snapshot_path': snapshot_path, 'delta_cv': delta_cv,
        }

    def run_one_iteration(self, n_bouts: int, confidence_threshold: float,
                          min_bout_frames: int = 5, context_frames: int = 30,
                          max_bout_frames=None,          # None = adaptive
                          class_balanced: bool = True, diversity_radius: int = 0,
                          progress_callback=None, model=None) -> dict:
        if model is None:
            model = self.train_model()
        probas = self.get_full_probas(model)
        max_bout = max_bout_frames if max_bout_frames else self._adaptive_max_bout_frames()
        _candidates = self._engine.find_uncertain_bouts(
            probas, self._labels, n_bouts * 3, confidence_threshold,
            min_bout_frames, context_frames, max_bout_frames=max_bout,
            class_balanced=class_balanced, diversity_radius=diversity_radius)
        _unseen = [b for b in _candidates if (b.start_frame, b.end_frame) not in self._seen_bouts]
        _seen_c = [b for b in _candidates if (b.start_frame, b.end_frame) in self._seen_bouts]
        bouts = (_unseen + _seen_c)[:n_bouts]
        for b in bouts:
            self._seen_bouts.add((b.start_frame, b.end_frame))
        for b in bouts:
            b.session_idx = 0
            b.video_path = self.video_path
        n_eligible = int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
        return {
            'model': model,
            'probas': probas,
            'bouts': bouts,
            'n_eligible': n_eligible,
        }

    def apply_labels(self, new_labels: dict, confidence_threshold: float,
                     propagate: bool = False,
                     probas: np.ndarray = None) -> dict:
        n_propagated = 0

        for key, label in new_labels.items():
            if isinstance(key, tuple):
                # Bout-keyed: expand to per-frame
                # Support both 2-tuple (start, end) and 3-tuple (session_idx, start, end)
                if len(key) == 3:
                    _, start, end = key
                else:
                    start, end = key
                for f in range(start, min(end + 1, len(self._labels))):
                    self._labels[f] = label
            else:
                # Frame-keyed (legacy)
                self._labels[key] = label
                if propagate:
                    propagated = self._propagator.propagate(
                        key, label, self._features, self._labels
                    )
                    for pf, pl in propagated.items():
                        self._labels[pf] = pl
                    n_propagated += len(propagated)

        # Save CSV
        self.save_labels_csv()

        # Compute stats for tracker
        labeled_mask, X_labeled, y_labeled, _ = self.get_labels_and_features()
        n_labeled_total = int(labeled_mask.sum())
        n_positive = int((y_labeled == 1).sum())

        try:
            model = self.train_model()
            n_below = int(np.sum(
                np.abs(self._engine.score_all_frames(model, self._features) - 0.5) * 2
                < confidence_threshold
            ))
            record = self.tracker.record(model, X_labeled, y_labeled, n_below)
        except Exception:
            record = None

        self._iteration += 1

        return {
            'propagated_count': n_propagated,
            'n_labeled_total': n_labeled_total,
            'n_positive': n_positive,
            'iteration_record': record,
        }

    def count_eligible(self, probas: np.ndarray, threshold: float,
                       min_bout_frames: int = 5) -> tuple:
        """Returns (n_frames, n_bouts, stats) where n_frames = unlabeled & uncertain."""
        # Extend probas with 0.5 (max uncertainty) for frames beyond features cache
        n_lab = len(self._labels)
        if n_lab > len(probas):
            ext_probas = np.concatenate([probas, np.full(n_lab - len(probas), 0.5)])
        else:
            ext_probas = probas[:n_lab]
        uncertain = np.abs(ext_probas - 0.5) * 2 < threshold
        unlabeled = self._labels < 0
        n_frames = int(np.sum(uncertain & unlabeled))
        stats = {}
        bouts = self._engine.find_uncertain_bouts(
            probas, self._labels, n_bouts=9999,
            confidence_threshold=threshold, min_bout_frames=min_bout_frames,
            _stats=stats)
        return n_frames, len(bouts), stats

    def count_positive(self) -> int:
        return int((self._labels == 1).sum())

    def is_converged(self, probas: np.ndarray, confidence_threshold: float) -> bool:
        return np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold) == 0

    def save_labels_csv(self):
        """Read existing CSV, update first column with self._labels, grow if needed."""
        try:
            df = pd.read_csv(self.labels_csv)
            col = df.columns[0]
            n_df  = len(df)
            n_lbl = len(self._labels)

            # Build output array: -1 → NaN
            out = self._labels.astype(float)
            out[out == -1] = np.nan

            if n_lbl > n_df:
                # Check whether any frames beyond the original CSV have been labeled
                extra = out[n_df:]
                last_new = -1
                for i in range(len(extra) - 1, -1, -1):
                    if not np.isnan(extra[i]):
                        last_new = i
                        break
                if last_new >= 0:
                    # Extend df with NaN rows up to and including the furthest new label
                    new_rows = pd.DataFrame({col: extra[:last_new + 1]})
                    df = pd.concat([df, new_rows], ignore_index=True)

            # Update all rows within current df
            n = min(len(df), n_lbl)
            df.loc[:n - 1, col] = out[:n]
            df.to_csv(self.labels_csv, index=False)
        except Exception:
            traceback.print_exc()


# ============================================================================
# MultiSessionAL
# ============================================================================

class MultiSessionAL:
    def __init__(self, sessions: list, min_frame_spacing: int = 5):
        """
        sessions: list of {'labels_csv': str, 'video_path': str, 'features_cache': str}
        """
        import pickle
        self._subs = []
        self.behavior_name = None
        for i, s in enumerate(sessions):
            df = pd.read_csv(s['labels_csv'])
            bname = df.columns[0]
            if self.behavior_name is None:
                self.behavior_name = bname
            raw = df[bname].values
            labels = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))
            with open(s['features_cache'], 'rb') as f:
                feats = pickle.load(f)
            feat_cols = list(feats.columns) if isinstance(feats, pd.DataFrame) else None
            if isinstance(feats, pd.DataFrame):
                feats = feats.values
            feats = feats.astype(np.float32)
            n_feat = len(feats)
            n_csv = len(labels)
            _truncation_warning = None
            if n_feat > n_csv:
                # Features cover more frames than CSV — extend labels with -1 (unlabeled)
                labels = np.concatenate([labels, np.full(n_feat - n_csv, -1, dtype=int)])
            elif n_feat < n_csv:
                truncated_tail = labels[n_feat:]
                n_truncated_labeled = int(np.sum(truncated_tail >= 0))
                labels = labels[:n_feat]
                _truncation_warning = (n_csv - n_feat, n_truncated_labeled)
            self._subs.append({
                'labels': labels, 'features': feats, 'feature_cols': feat_cols,
                'video_path': s['video_path'], 'labels_csv': s['labels_csv'],
                '_truncation_warning': _truncation_warning,
            })
        self._engine = UncertaintyEngineV2(min_frame_spacing)
        self._propagator = LabelPropagator()
        self.tracker = LearningCurveTracker()
        self._iteration = 0
        self._last_model = None
        self._seen_bouts: set = set()   # (session_idx, start_frame, end_frame) of shown bouts

    def train_model(self):
        """Pool labeled frames from all sessions and train XGBoost."""
        from xgboost import XGBClassifier
        dfs, ys = [], []
        for sub in self._subs:
            mask = sub['labels'] >= 0
            if not mask.any():
                continue
            X_sub = sub['features'][mask]
            if sub.get('feature_cols'):
                dfs.append(pd.DataFrame(X_sub, columns=sub['feature_cols']))
            else:
                dfs.append(pd.DataFrame(X_sub))
            ys.append(sub['labels'][mask])
        if not dfs:
            raise ValueError("No labeled frames across any session.")
        X = pd.concat(dfs, ignore_index=True).fillna(0.0)
        y = np.concatenate(ys)
        if (y == 1).sum() == 0:
            raise ValueError("No positive-labeled frames.")
        model = XGBClassifier(n_estimators=100, max_depth=6,
                              learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X, y)
        return model

    def run_one_iteration(self, n_bouts, confidence_threshold,
                          min_bout_frames=5, context_frames=30,
                          max_bout_frames=None,          # None = adaptive (300)
                          class_balanced: bool = True, diversity_radius: int = 0,
                          progress_callback=None, model=None) -> dict:
        if model is None:
            model = self.train_model()
        self._last_model = model
        all_bouts = []
        n_eligible = 0
        max_bout = max_bout_frames if max_bout_frames else 300
        for i, sub in enumerate(self._subs):
            probas = model.predict_proba(
                _align_features(model, sub['features'], sub.get('feature_cols')))[:, 1]
            n_eligible += int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
            bouts = self._engine.find_uncertain_bouts(
                probas, sub['labels'], n_bouts=n_bouts * 3,
                confidence_threshold=confidence_threshold,
                min_bout_frames=min_bout_frames, context_frames=context_frames,
                max_bout_frames=max_bout,
                class_balanced=class_balanced, diversity_radius=diversity_radius)
            for b in bouts:
                b.session_idx = i
                b.video_path = sub['video_path']
            all_bouts.extend(bouts)
        # Round-robin interleave by session so every video is represented
        from collections import deque as _deque
        _by_sess = {}
        for _b in all_bouts:
            _by_sess.setdefault(_b.session_idx, []).append(_b)
        # Each per-session list is already ordered by descending uncertainty
        # (find_uncertain_bouts returns top n_bouts sorted that way)
        _queues = [_deque(_bouts) for _bouts in _by_sess.values()]
        _selected = []
        while len(_selected) < n_bouts and _queues:
            _next_queues = []
            for _q in _queues:
                if _q:
                    _selected.append(_q.popleft())
                    if len(_selected) >= n_bouts:
                        break
                if _q:
                    _next_queues.append(_q)
            _queues = _next_queues
        # Seen-bout deduplication: prefer bouts not shown in prior iterations
        _unseen = [b for b in _selected
                   if (b.session_idx, b.start_frame, b.end_frame) not in self._seen_bouts]
        _seen_c = [b for b in _selected
                   if (b.session_idx, b.start_frame, b.end_frame) in self._seen_bouts]
        _selected = (_unseen + _seen_c)[:n_bouts]
        for b in _selected:
            self._seen_bouts.add((b.session_idx, b.start_frame, b.end_frame))
        all_probas = np.concatenate([
            model.predict_proba(
                _align_features(model, sub['features'], sub.get('feature_cols')))[:, 1]
            for sub in self._subs])
        return {'model': model, 'probas': all_probas,
                'bouts': _selected, 'n_eligible': n_eligible}

    def apply_labels(self, new_labels: dict, confidence_threshold: float,
                     propagate: bool = False,
                     probas: np.ndarray = None) -> dict:
        """new_labels keys: (session_idx, start, end) tuples.
        probas, if given, is the concatenated array across all sub-sessions in order.
        """
        # Build per-session offsets into concatenated probas array
        offsets = []
        pos = 0
        for sub in self._subs:
            offsets.append(pos)
            pos += len(sub['features'])

        for key, label in new_labels.items():
            if len(key) == 3:
                sess_idx, start, end = key
            else:
                sess_idx, (start, end) = 0, key
            sub = self._subs[sess_idx]
            offset = offsets[sess_idx]
            for f in range(start, min(end + 1, len(sub['labels']))):
                if label == 1 and probas is not None:
                    global_f = offset + f
                    if global_f < len(probas):
                        sub['labels'][f] = 1 if probas[global_f] > 0.5 else 0
                    else:
                        sub['labels'][f] = label
                else:
                    sub['labels'][f] = label

        for sub in self._subs:
            self._save_labels_csv(sub)

        Xs, ys = [], []
        for sub in self._subs:
            mask = sub['labels'] >= 0
            if mask.any():
                Xs.append(sub['features'][mask])
                ys.append(sub['labels'][mask])
        n_labeled = sum(len(y) for y in ys)
        n_positive = sum((y == 1).sum() for y in ys)

        record = None
        try:
            model = self.train_model()
            X = np.concatenate(Xs)
            y_all = np.concatenate(ys)
            # Compute actual n_below from fresh predictions
            all_p = np.concatenate([
                model.predict_proba(sub['features'])[:, 1] for sub in self._subs])
            n_below = int(np.sum(np.abs(all_p - 0.5) * 2 < confidence_threshold))
            record = self.tracker.record(model, X, y_all, n_below)
        except Exception:
            import traceback
            traceback.print_exc()

        self._iteration += 1
        return {'propagated_count': 0, 'n_labeled_total': n_labeled,
                'n_positive': int(n_positive), 'iteration_record': record}

    def is_converged(self, probas, confidence_threshold) -> bool:
        return np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold) == 0

    def count_eligible(self, probas, threshold, min_bout_frames=5) -> tuple:
        n_frames = 0
        n_bouts = 0
        pos = 0
        for sub in self._subs:
            n = len(sub['features'])
            sub_probas = probas[pos:pos + n]
            pos += n
            uncertain = np.abs(sub_probas - 0.5) * 2 < threshold
            unlabeled = sub['labels'] < 0
            n_frames += int(np.sum(uncertain & unlabeled))
            bouts = self._engine.find_uncertain_bouts(
                sub_probas, sub['labels'], n_bouts=9999,
                confidence_threshold=threshold, min_bout_frames=min_bout_frames)
            n_bouts += len(bouts)
        return n_frames, n_bouts, {}

    def count_positive(self) -> int:
        return sum(int((s['labels'] == 1).sum()) for s in self._subs)

    def retrain_and_snapshot(self, confidence_threshold: float = 0.3,
                              snapshot_dir: str = None) -> dict:
        import pickle, os, time

        prev_cv = self.tracker.records[-1].cv_f1 if self.tracker.records else None

        model = self.train_model()
        all_p = np.concatenate([
            model.predict_proba(sub['features'])[:, 1] for sub in self._subs])
        n_below = int(np.sum(np.abs(all_p - 0.5) * 2 < confidence_threshold))

        Xs, ys = [], []
        for sub in self._subs:
            mask = sub['labels'] >= 0
            if mask.any():
                Xs.append(sub['features'][mask])
                ys.append(sub['labels'][mask])
        X = np.concatenate(Xs)
        y_all = np.concatenate(ys)
        record = self.tracker.record(model, X, y_all, n_below)

        snapshot_path = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            fname = f"al_iter{record.iteration}_{ts}.pkl"
            snapshot_path = os.path.join(snapshot_dir, fname)
            with open(snapshot_path, 'wb') as f:
                pickle.dump({
                    'clf_model': model,
                    'iteration': record.iteration,
                    'train_f1': record.train_f1,
                    'cv_f1': record.cv_f1,
                    'n_labeled_total': record.n_labeled_total,
                }, f)

        delta_cv = None
        if prev_cv is not None and record.cv_f1 is not None:
            delta_cv = record.cv_f1 - prev_cv

        return {
            'model': model, 'probas': all_p, 'record': record,
            'snapshot_path': snapshot_path, 'delta_cv': delta_cv,
        }

    @staticmethod
    def _save_labels_csv(sub: dict):
        try:
            df = pd.read_csv(sub['labels_csv'])
            col = df.columns[0]
            n = min(len(df), len(sub['labels']))
            df.loc[:n-1, col] = sub['labels'][:n].astype(float)
            df.loc[df[col] == -1, col] = np.nan
            df.to_csv(sub['labels_csv'], index=False)
        except Exception:
            import traceback
            traceback.print_exc()


# ============================================================================
# run_directed_discovery
# ============================================================================

def run_directed_discovery(project_folder: str, labels_csv: str, features_cache: str,
                            behavior_name: str, run_name: str = 'al_discovery',
                            min_cluster_size: int = 50) -> Optional[str]:
    """
    Run UMAP + HDBSCAN on positive-labeled frames to find sub-behaviors.
    Saves per-cluster CSVs to <project>/unsupervised/<run_name>/.
    Returns output path or None on failure.
    Does NOT import from unsupervised_tab.py.
    """
    if not UMAP_HDBSCAN_AVAILABLE:
        print("run_directed_discovery: umap-learn/hdbscan not installed. Skipping.")
        return None
    try:
        # Load labels
        df_labels = pd.read_csv(labels_csv)
        bname = df_labels.columns[0]
        raw = df_labels[bname].values
        labels_arr = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))

        # Load features
        with open(features_cache, 'rb') as f:
            feats = pickle.load(f)
        if isinstance(feats, pd.DataFrame):
            feats = feats.values
        feats = feats.astype(np.float32)

        min_len = min(len(labels_arr), len(feats))
        labels_arr = labels_arr[:min_len]
        feats = feats[:min_len]

        pos_mask = labels_arr == 1
        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) < 50:
            print(f"run_directed_discovery: only {len(pos_indices)} positive frames, need >=50.")
            return None

        X = feats[pos_indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embedding)

        # Save outputs
        out_dir = os.path.join(project_folder, 'unsupervised', run_name)
        os.makedirs(out_dir, exist_ok=True)

        unique_clusters = sorted(set(cluster_labels))
        for c in unique_clusters:
            if c == -1:
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{c:02d}'

            mask = cluster_labels == c
            frame_indices = pos_indices[mask]

            # Build output DataFrame — one binary column per cluster
            total_frames = len(labels_arr)
            col = np.zeros(total_frames, dtype=int)
            col[frame_indices] = 1

            out_df = pd.DataFrame({cluster_name: col})
            out_path = os.path.join(out_dir, f'{cluster_name}.csv')
            out_df.to_csv(out_path, index=False)

        # Save summary
        summary = {
            'behavior': bname,
            'run_name': run_name,
            'n_positive_frames': int(len(pos_indices)),
            'n_clusters': len([c for c in unique_clusters if c != -1]),
            'n_noise': int(np.sum(cluster_labels == -1)),
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(out_dir, 'discovery_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"run_directed_discovery: saved {len(unique_clusters)} cluster CSVs to {out_dir}")
        return out_dir
    except Exception as e:
        traceback.print_exc()
        return None


# ============================================================================
# LabelingInterface (migrated verbatim from active_learning.py)
# ============================================================================

class LabelingInterface:
    """
    GUI for labeling suggested frames.
    """

    def __init__(self,
                 video_path: str,
                 suggested_frames: np.ndarray,
                 confidences: np.ndarray,
                 behavior_name: str):
        """
        Args:
            video_path: Path to video file
            suggested_frames: Frame indices to label
            confidences: Model confidence for each frame
            behavior_name: Name of behavior being labeled
        """
        self.video_path = video_path
        self.suggested_frames = suggested_frames
        self.confidences = confidences
        self.behavior_name = behavior_name

        # Results
        self.labels = {}  # frame_idx -> 0 or 1
        self.current_idx = 0

        # Video
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # UI
        self.window = None
        self.video_label = None
        self.context_start = -60  # Show 1 sec before
        self.context_end = 60     # Show 1 sec after

    def run(self) -> dict:
        """
        Show labeling interface and return labeled frames.

        Returns:
            Dictionary mapping frame_idx -> label (0 or 1)
        """
        self.create_ui()
        self.show_current_frame()

        # If using Toplevel, use wait_window instead of mainloop
        if isinstance(self.window, tk.Toplevel):
            self.window.wait_window()
        else:
            self.window.mainloop()

        self.cap.release()
        return self.labels

    def create_ui(self):
        """Create the labeling interface window"""
        # Use Toplevel instead of Tk() since we're being called from PixelPaws
        # which already has a Tk() root
        import tkinter as tk
        from tkinter import ttk

        # Try to get existing root, otherwise create new one
        try:
            root = tk._default_root
            if root:
                self.window = tk.Toplevel(root)
            else:
                self.window = tk.Tk()
        except Exception:
            self.window = tk.Tk()

        self.window.title(f"Active Learning - {self.behavior_name}")
        _sw = self.window.winfo_screenwidth()
        _sh = self.window.winfo_screenheight()
        _w = int(_sw * 0.75)
        _h = int(_sh * 0.75)
        self.window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        self.window.resizable(True, True)

        # Title
        title_frame = ttk.Frame(self.window)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            title_frame,
            text=f"🧠 Active Learning - {self.behavior_name}",
            font=("Arial", 16, "bold")
        ).pack()

        # Progress
        progress_frame = ttk.Frame(self.window)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(
            progress_frame,
            text=f"Frame 1 of {len(self.suggested_frames)}",
            font=("Arial", 12)
        )
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=800,
            mode='determinate',
            maximum=len(self.suggested_frames)
        )
        self.progress_bar.pack(pady=5)

        # Video display
        video_frame = ttk.Frame(self.window)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()

        # Frame info
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(
            info_frame,
            text="",
            font=("Arial", 10)
        )
        self.info_label.pack()

        # Question
        question_frame = ttk.Frame(self.window)
        question_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            question_frame,
            text=f"Is this {self.behavior_name} behavior?",
            font=("Arial", 14, "bold")
        ).pack()

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        _bs = {'bootstyle': 'success'} if _TTKBOOTSTRAP else {}
        btn_yes = ttk.Button(
            button_frame, text="✓ YES (Y)", width=15,
            command=lambda: self.label_frame(1), **_bs)
        btn_yes.pack(side=tk.LEFT, padx=5, expand=True)

        _bd = {'bootstyle': 'danger'} if _TTKBOOTSTRAP else {}
        btn_no = ttk.Button(
            button_frame, text="✗ NO (N)", width=15,
            command=lambda: self.label_frame(0), **_bd)
        btn_no.pack(side=tk.LEFT, padx=5, expand=True)

        _bsec = {'bootstyle': 'secondary'} if _TTKBOOTSTRAP else {}
        btn_skip = ttk.Button(
            button_frame, text="? SKIP (S)", width=15,
            command=self.skip_frame, **_bsec)
        btn_skip.pack(side=tk.LEFT, padx=5, expand=True)

        # Context playback button
        context_frame = ttk.Frame(self.window)
        context_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            context_frame,
            text="▶ Play Context (±2 sec)",
            command=self.play_context
        ).pack()

        # Keyboard shortcuts
        self.window.bind('y', lambda e: self.label_frame(1))
        self.window.bind('Y', lambda e: self.label_frame(1))
        self.window.bind('n', lambda e: self.label_frame(0))
        self.window.bind('N', lambda e: self.label_frame(0))
        self.window.bind('s', lambda e: self.skip_frame())
        self.window.bind('S', lambda e: self.skip_frame())
        self.window.bind('<space>', lambda e: self.play_context())

        # Shortcuts label
        ttk.Label(
            self.window,
            text="Shortcuts: Y=Yes | N=No | S=Skip | Space=Play Context",
            font=("Arial", 9),
            foreground="gray"
        ).pack(pady=5)

    def show_current_frame(self):
        """Display the current frame to label"""
        if self.current_idx >= len(self.suggested_frames):
            self.finish_labeling()
            return

        frame_idx = self.suggested_frames[self.current_idx]
        confidence = self.confidences[self.current_idx]

        # Update progress
        self.progress_label.config(
            text=f"Frame {self.current_idx + 1} of {len(self.suggested_frames)}"
        )
        self.progress_bar['value'] = self.current_idx

        # Update info
        timestamp = frame_idx / self.fps
        self.info_label.config(
            text=f"Frame: {frame_idx} / {self.total_frames}  |  "
                 f"Time: {timestamp:.2f}s  |  "
                 f"Confidence: {confidence:.1%} (Uncertain!)"
        )

        # Load and display frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            # Resize for display
            height, width = frame.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)

            # CRITICAL: Keep reference to prevent garbage collection
            self.current_photo = photo
            self.video_label.config(image=photo)

    def label_frame(self, label: int):
        """Label current frame and move to next"""
        frame_idx = self.suggested_frames[self.current_idx]

        # Store label in dict (no bounds check needed - dict can store any frame index)
        self.labels[frame_idx] = label

        print(f"  Frame {frame_idx}: {'YES' if label == 1 else 'NO'}")

        self.current_idx += 1

        # Check if we're done
        if self.current_idx >= len(self.suggested_frames):
            self.close_interface()
        else:
            self.show_current_frame()

    def skip_frame(self):
        """Skip current frame without labeling"""
        frame_idx = self.suggested_frames[self.current_idx]
        print(f"  Frame {frame_idx}: SKIPPED")

        self.current_idx += 1

        # Check if we're done
        if self.current_idx >= len(self.suggested_frames):
            self.close_interface()
        else:
            self.show_current_frame()

    def play_context(self):
        """Play video context around current frame"""
        frame_idx = self.suggested_frames[self.current_idx]

        # Calculate context window
        start_frame = max(0, frame_idx + self.context_start)
        end_frame = min(self.total_frames, frame_idx + self.context_end)

        # Create playback window
        play_window = tk.Toplevel(self.window)
        play_window.title("Context Playback")
        _sw = play_window.winfo_screenwidth()
        _sh = play_window.winfo_screenheight()
        _w = int(_sw * 0.65)
        _h = int(_sh * 0.65)
        play_window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        play_window.resizable(True, True)

        play_label = tk.Label(play_window, bg='black')
        play_label.pack(fill=tk.BOTH, expand=True)

        # Play frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Highlight target frame
            if i == frame_idx:
                cv2.rectangle(frame, (10, 10),
                              (frame.shape[1]-10, frame.shape[0]-10),
                              (0, 255, 0), 5)
                cv2.putText(frame, "TARGET FRAME", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize and display
            height, width = frame.shape[:2]
            max_width = 750
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(frame, (max_width, int(height * scale)))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)

            play_label.config(image=photo)
            play_label.photo = photo  # Keep reference
            play_window.update()

            # Delay to match FPS
            time.sleep(1.0 / self.fps)

            if not play_window.winfo_exists():
                break

        if play_window.winfo_exists():
            play_window.destroy()

    def finish_labeling(self):
        """Show completion message and close"""
        n_labeled = len(self.labels)
        n_total = len(self.suggested_frames)

        messagebox.showinfo(
            "Active Learning Complete!",
            f"Labeled {n_labeled} out of {n_total} suggested frames.\n\n"
            f"Labels will be saved to your per-frame CSV file.\n"
            f"The model will now be retrained with these new labels."
        )

        self.window.destroy()

    def close_interface(self):
        """Alias for finish_labeling"""
        self.finish_labeling()


# ============================================================================
# BoutLabelingInterface — bout-level labeling (A-SOiD style)
# ============================================================================

class BoutLabelingInterface:
    """
    GUI for bout-level labeling.  Shows a looping video clip for each uncertain
    bout; user clicks YES / NO / SKIP for the entire clip.

    Returns {(start_frame, end_frame): 0_or_1} — consumed by ALSessionV2.apply_labels().
    """

    MAX_CLIP_FRAMES = 600   # cap to avoid memory issues on long clips

    def __init__(self, video_path: str, bouts: List[BoutCandidate],
                 probas: np.ndarray, behavior_name: str, fps: float):
        self.video_path = video_path
        self.bouts = bouts
        self.probas = probas
        self.behavior_name = behavior_name

        self.cap = cv2.VideoCapture(video_path)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = max(fps if fps > 0 else _cap_fps, 1.0)

        self._result_labels: Dict[Tuple[int, int], int] = {}
        self._current_idx = 0
        self._loop_frames: list = []   # list of (PhotoImage, x_off, y_off)
        self._loop_pos = 0
        self._paused = False
        self._after_id = None
        self._window_open = False

        # Tk widgets (set in _build_window)
        self._window = None
        self._video_canvas = None
        self._progress_label = None
        self._info_var = None
        self._trace_ax = None
        self._trace_canvas_wgt = None
        self._cursor_line = None
        self._mark_in: Optional[int] = None    # absolute frame, or None = use bout.start_frame
        self._mark_out: Optional[int] = None   # absolute frame, or None = use bout.end_frame
        self._sel_var = None                   # tk.StringVar created in _build_window

    # ------------------------------------------------------------------
    def run(self) -> dict:
        if not self.bouts:
            return {}
        self._build_window()
        if self._window_open:
            self._load_bout(0)
            self._window.wait_window()
        self.cap.release()
        return self._result_labels

    # ------------------------------------------------------------------
    def _build_window(self):
        try:
            root = tk._default_root
            self._window = tk.Toplevel(root)
        except Exception:
            self._window = tk.Tk()

        self._window_open = True
        self._window.title(f"Active Learning — {self.behavior_name}")
        _sw = self._window.winfo_screenwidth()
        _sh = self._window.winfo_screenheight()
        _w = int(_sw * 0.90)
        _h = int(_sh * 0.90)
        self._window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        self._window.resizable(True, True)
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Header
        hdr = ttk.Frame(self._window)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        self._header_label = ttk.Label(hdr, text=f"Active Learning — {self.behavior_name}",
                                       font=('Arial', 13, 'bold'))
        self._header_label.pack(side='left')
        self._progress_label = ttk.Label(hdr, text="", font=('Arial', 10))
        self._progress_label.pack(side='right')

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Video canvas — fills available width, fixed height for pre-rendered frames
        self._video_canvas = tk.Canvas(self._window, bg='black', width=900, height=520)
        self._video_canvas.pack(padx=10, pady=4)

        # Info bar
        self._info_var = tk.StringVar(value="")
        ttk.Label(self._window, textvariable=self._info_var,
                  font=('Arial', 9), foreground='navy').pack()

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Probability trace
        if MATPLOTLIB_AVAILABLE:
            try:
                _fig, self._trace_ax = plt.subplots(figsize=(6.4, 0.9), dpi=90, constrained_layout=True)
                self._trace_canvas_wgt = FigureCanvasTkAgg(_fig, master=self._window)
                self._trace_canvas_wgt.get_tk_widget().pack(fill='x', padx=10, pady=2)
                _bind_tight_layout_on_resize(self._trace_canvas_wgt, _fig)
                self._trace_fig = _fig
            except Exception:
                self._trace_ax = None
                self._trace_canvas_wgt = None

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Label buttons
        btn_frame = ttk.Frame(self._window)
        btn_frame.pack(pady=4)
        _bs = {'bootstyle': 'success'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="✓ YES (Y)", width=12,
                  command=lambda: self._label_bout(1), **_bs).pack(side='left', padx=8)
        _bd = {'bootstyle': 'danger'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="✗ NO (N)", width=12,
                  command=lambda: self._label_bout(0), **_bd).pack(side='left', padx=8)
        _bsec = {'bootstyle': 'secondary'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="? SKIP / Next [S/↵]", width=18,
                  command=self._skip_bout, **_bsec).pack(side='left', padx=8)

        # Step controls
        step_frame = ttk.Frame(self._window)
        step_frame.pack(pady=2)
        ttk.Button(step_frame, text="◀ Step (←)", width=12,
                   command=lambda: self._step_frame(-1)).pack(side='left', padx=6)
        ttk.Button(step_frame, text="Step (→) ▶", width=12,
                   command=lambda: self._step_frame(1)).pack(side='left', padx=6)

        # Mark In / Mark Out row
        mark_frame = ttk.Frame(self._window)
        mark_frame.pack(pady=2)
        ttk.Button(mark_frame, text="Mark In [I]", width=12,
                   command=self._set_mark_in).pack(side='left', padx=6)
        ttk.Button(mark_frame, text="Mark Out [O]", width=12,
                   command=self._set_mark_out).pack(side='left', padx=6)
        ttk.Button(mark_frame, text="Clear [C]", width=10,
                   command=self._clear_marks).pack(side='left', padx=6)

        # Selection status label
        self._sel_var = tk.StringVar(value="Selection: full bout")
        ttk.Label(self._window, textvariable=self._sel_var,
                  font=('Arial', 9), foreground='darkorange').pack(pady=1)

        ttk.Label(self._window,
                  text="Shortcuts: Y=Yes  N=No  S/Enter=Next Bout  Space=Pause/Resume  ←/→=Step  I=Mark In  O=Mark Out  C=Clear",
                  font=('Arial', 9), foreground='gray').pack(pady=2)

        self._window.bind('y', lambda e: self._label_bout(1))
        self._window.bind('Y', lambda e: self._label_bout(1))
        self._window.bind('n', lambda e: self._label_bout(0))
        self._window.bind('N', lambda e: self._label_bout(0))
        self._window.bind('s', lambda e: self._skip_bout())
        self._window.bind('S', lambda e: self._skip_bout())
        self._window.bind('<space>', lambda e: self._toggle_pause())
        self._window.bind('<Left>',  lambda e: self._step_frame(-1))
        self._window.bind('<Right>', lambda e: self._step_frame(1))
        self._window.bind('i', lambda e: self._set_mark_in())
        self._window.bind('I', lambda e: self._set_mark_in())
        self._window.bind('o', lambda e: self._set_mark_out())
        self._window.bind('O', lambda e: self._set_mark_out())
        self._window.bind('c', lambda e: self._clear_marks())
        self._window.bind('C', lambda e: self._clear_marks())
        self._window.bind('<Return>', lambda e: self._skip_bout())

    # ------------------------------------------------------------------
    def _load_bout(self, idx: int):
        if idx >= len(self.bouts):
            self._finish()
            return

        self._mark_in = None
        self._mark_out = None
        if self._sel_var:
            self._sel_var.set("Selection: full bout")

        self._current_idx = idx
        bout = self.bouts[idx]

        # Reopen cap if this bout's video differs from current
        if bout.video_path and bout.video_path != getattr(self, '_current_video_path', None):
            self.cap.release()
            self.cap = cv2.VideoCapture(bout.video_path)
            _fps = self.cap.get(cv2.CAP_PROP_FPS)
            if _fps > 0:
                self.fps = _fps
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._current_video_path = bout.video_path

        self._progress_label.config(text=f"Bout {idx + 1} of {len(self.bouts)}")
        if bout.video_path:
            vname = os.path.basename(bout.video_path)
            self._header_label.config(
                text=f"Active Learning — {self.behavior_name}  [{vname}]")
        dur_sec = bout.duration_frames / self.fps
        _vname = os.path.basename(bout.video_path) if bout.video_path else ""
        _vpart = f"{_vname}  |  " if _vname else ""
        self._info_base = (_vpart +
                           f"Frames {bout.start_frame}–{bout.end_frame} | "
                           f"{dur_sec:.1f} sec | Mean P(1)={bout.mean_proba:.2f}")
        self._info_var.set(self._info_base + " | \u25B6 LOOPING")

        # Clamp clip to MAX_CLIP_FRAMES
        clip_start = bout.clip_start
        clip_end = min(bout.clip_end, clip_start + self.MAX_CLIP_FRAMES - 1)
        if bout.clip_end > clip_end:
            self._info_base += (f"  | TRIMMED (showing {clip_end - clip_start + 1}"
                                f" of {bout.clip_end - bout.clip_start + 1} frames)")

        # Read frame dimensions first
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        ret, first_frame = self.cap.read()
        if not ret:
            self._advance_bout()
            return

        display_w, display_h = 900, 520
        h, w = first_frame.shape[:2]
        scale = min(display_w / w, display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        x_off = (display_w - new_w) // 2
        y_off = (display_h - new_h) // 2

        # Pre-read all clip frames
        from PIL import Image, ImageTk
        clip_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for fi in range(clip_start, clip_end + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (new_w, new_h))
            # Green border for core uncertain region
            if bout.start_frame <= fi <= bout.end_frame:
                cv2.rectangle(frame_resized, (2, 2), (new_w - 3, new_h - 3), (0, 200, 0), 3)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            clip_frames.append((photo, x_off, y_off))

        if not clip_frames:
            self._advance_bout()
            return

        self._loop_frames = clip_frames
        self._loop_pos = 0
        self._paused = False

        # Draw probability trace
        self._draw_trace(bout)

        # Cancel any pending loop callback, then start fresh
        if self._after_id is not None:
            try:
                self._window.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        self._schedule_next_frame()

    def _schedule_next_frame(self):
        delay_ms = max(1, int(1000.0 / self.fps))
        self._after_id = self._window.after(delay_ms, self._next_frame)

    def _next_frame(self):
        if not self._window_open or not self._loop_frames:
            return
        if self._paused:
            self._after_id = self._window.after(50, self._next_frame)
            return

        photo, x_off, y_off = self._loop_frames[self._loop_pos]
        self._video_canvas.delete('all')
        self._video_canvas.create_image(x_off, y_off, anchor='nw', image=photo)
        self._video_canvas.image = photo  # keep reference

        # Update probability trace cursor
        if (self._trace_ax is not None and self._cursor_line is not None
                and self._trace_canvas_wgt is not None):
            bout = self.bouts[self._current_idx]
            abs_frame = bout.clip_start + self._loop_pos
            self._cursor_line.set_xdata([abs_frame, abs_frame])
            self._trace_canvas_wgt.draw_idle()

        self._loop_pos = (self._loop_pos + 1) % len(self._loop_frames)
        self._schedule_next_frame()

    def _draw_trace(self, bout: BoutCandidate):
        if self._trace_ax is None or self._trace_canvas_wgt is None:
            return
        try:
            self._trace_ax.clear()
            clip_probas = self.probas[bout.clip_start:bout.clip_end + 1]
            x = np.arange(bout.clip_start, bout.clip_start + len(clip_probas))
            self._trace_ax.plot(x, clip_probas, color='steelblue', linewidth=1.0)
            self._trace_ax.axhspan(0.35, 0.65, color='orange', alpha=0.2)
            self._trace_ax.axvspan(bout.start_frame, bout.end_frame,
                                   color='green', alpha=0.15)
            # User-selected sub-range (when marks are set)
            if self._mark_in is not None or self._mark_out is not None:
                sel_s = self._mark_in if self._mark_in is not None else bout.start_frame
                sel_e = self._mark_out if self._mark_out is not None else bout.end_frame
                self._trace_ax.axvspan(sel_s, sel_e, color='gold', alpha=0.40, zorder=3)
            self._trace_ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
            self._trace_ax.set_ylim(0, 1)
            self._trace_ax.set_ylabel("P(1)", fontsize=7)
            self._trace_ax.tick_params(labelsize=6)
            self._cursor_line = self._trace_ax.axvline(bout.clip_start,
                                                        color='red', linewidth=1.0)
            self._trace_canvas_wgt.draw()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            bout = self.bouts[self._current_idx]
            abs_frame = bout.clip_start + self._loop_pos
            self._info_var.set(self._info_base + f" | \u23F8 Frame {abs_frame}")
        else:
            self._info_var.set(self._info_base + " | \u25B6 LOOPING")

    def _step_frame(self, delta: int):
        """Step one frame forward (delta=1) or backward (delta=-1). Auto-pauses."""
        if not self._window_open or not self._loop_frames:
            return
        if not self._paused:
            self._paused = True
        self._loop_pos = max(0, min(len(self._loop_frames) - 1, self._loop_pos + delta))
        photo, x_off, y_off = self._loop_frames[self._loop_pos]
        self._video_canvas.delete('all')
        self._video_canvas.create_image(x_off, y_off, anchor='nw', image=photo)
        self._video_canvas.image = photo
        bout = self.bouts[self._current_idx]
        abs_frame = bout.clip_start + self._loop_pos
        if (self._trace_ax is not None and self._cursor_line is not None
                and self._trace_canvas_wgt is not None):
            self._cursor_line.set_xdata([abs_frame, abs_frame])
            self._trace_canvas_wgt.draw_idle()
        self._info_var.set(self._info_base + f" | \u23F8 Frame {abs_frame}")

    def _set_mark_in(self):
        if not self._window_open or not self._loop_frames:
            return
        bout = self.bouts[self._current_idx]
        self._mark_in = bout.clip_start + self._loop_pos
        self._update_mark_display(bout)

    def _set_mark_out(self):
        if not self._window_open or not self._loop_frames:
            return
        bout = self.bouts[self._current_idx]
        self._mark_out = bout.clip_start + self._loop_pos
        self._update_mark_display(bout)

    def _clear_marks(self):
        self._mark_in = None
        self._mark_out = None
        if self._current_idx < len(self.bouts):
            self._update_mark_display(self.bouts[self._current_idx])

    def _update_mark_display(self, bout: BoutCandidate):
        eff_start = self._mark_in if self._mark_in is not None else bout.start_frame
        eff_end   = self._mark_out if self._mark_out is not None else bout.end_frame
        if self._mark_in is None and self._mark_out is None:
            if self._sel_var:
                self._sel_var.set("Selection: full bout")
        else:
            n = max(0, eff_end - eff_start + 1)
            if self._sel_var:
                self._sel_var.set(f"Selection: frames {eff_start}–{eff_end}  ({n} frames)")
        self._draw_trace(bout)

    def _label_bout(self, label: int):
        if not self._window_open or self._current_idx >= len(self.bouts):
            return
        bout = self.bouts[self._current_idx]
        eff_start = self._mark_in if self._mark_in is not None else bout.start_frame
        eff_end   = self._mark_out if self._mark_out is not None else bout.end_frame
        key = (bout.session_idx, eff_start, eff_end)
        self._result_labels[key] = label
        if self._mark_in is not None or self._mark_out is not None:
            # Partial section labeled — stay on this bout so more sections can be marked
            self._clear_marks()
        else:
            # Full-bout label — advance immediately
            self._advance_bout()

    def _skip_bout(self):
        if self._window_open:
            self._advance_bout()

    def _advance_bout(self):
        next_idx = self._current_idx + 1
        if next_idx >= len(self.bouts):
            self._finish()
        else:
            self._load_bout(next_idx)

    def _finish(self):
        if self._after_id is not None:
            try:
                self._window.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._window_open:
            self._window_open = False
            try:
                self._window.destroy()
            except Exception:
                pass

    def _on_close(self):
        self._finish()

# ===========================================================================
# ActiveLearningTabV2 — main tab class (moved from PixelPaws_GUI.py)
# ===========================================================================

class ActiveLearningTabV2(ttk.Frame):
    """
    Active Learning v2 tab.
    Layout: horizontal PanedWindow — left=controls, right=plot+log.
    """
    def __init__(self, parent, parent_app):
        super().__init__(parent)
        self.app = parent_app
        self.pack(fill='both', expand=True)

        # State
        self._session = None
        self._last_probas = None
        self._last_model = None
        self._last_frames = None
        self._n_labeled_at_load = 0
        self._sessions_list = []
        self._clf_options = {}
        self._base_clf_f1 = None   # CV F1 of pre-loaded classifier (baseline for plot)

        # SharedVars
        self._threshold_var = tk.DoubleVar(value=0.30)
        self._n_suggestions_var = tk.IntVar(value=10)
        self._min_spacing_var = tk.IntVar(value=5)
        self._context_frames_var = tk.IntVar(value=10)
        self._budget_var = tk.IntVar(value=2000)
        self._max_bout_var = tk.IntVar(value=0)   # 0 = adaptive
        self._eligible_count_var = tk.StringVar(value="— not scored yet —")
        self._bout_aware_cv_var = tk.BooleanVar(value=True)
        self._propagate_var = tk.BooleanVar(value=False)
        self._class_balanced_var = tk.BooleanVar(value=True)
        self._diversity_radius_var = tk.IntVar(value=0)
        self._auto_iter_var = tk.IntVar(value=3)
        self._auto_remaining = 0
        self._stop_auto_var = tk.BooleanVar(value=False)
        self._btn_next_iter = None  # reference set in _build_left

        self._build_ui()

        # React to project changes
        self.app.current_project_folder.trace_add('write', lambda *_: self._on_project_changed())

    def _build_ui(self):
        # Header
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(hdr, text="🧠 Active Learning v2",
                  font=('Arial', 14, 'bold')).pack(side='left')

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned, width=500)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # Sessions
        sf = ttk.LabelFrame(parent, text="Sessions", padding=5)
        sf.pack(fill='both', expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(sf)
        btn_row.pack(fill='x', pady=(0, 4))
        ttk.Button(btn_row, text="🔄 Scan", width=8,
                   command=self._scan_sessions).pack(side='left', padx=(0, 4))

        lb_frame = ttk.Frame(sf)
        lb_frame.pack(fill='both', expand=True)
        self._session_lb = tk.Listbox(lb_frame, selectmode='extended', height=5)
        lb_sb = ttk.Scrollbar(lb_frame, command=self._session_lb.yview)
        self._session_lb.configure(yscrollcommand=lb_sb.set)
        self._session_lb.pack(side='left', fill='both', expand=True)
        lb_sb.pack(side='right', fill='y')

        # Classifier
        cf = ttk.LabelFrame(parent, text="Classifier (for scoring)", padding=5)
        cf.pack(fill='x', padx=4, pady=4)
        clf_row = ttk.Frame(cf)
        clf_row.pack(fill='x')
        self._clf_combo = ttk.Combobox(clf_row, state='readonly', width=28)
        self._clf_combo.pack(side='left', padx=(0, 4))
        ttk.Button(clf_row, text="↺", width=3,
                   command=self._refresh_classifiers).pack(side='left')
        ttk.Button(clf_row, text="📁", width=3,
                   command=self._browse_classifier).pack(side='left', padx=(2, 0))
        ToolTip(self._clf_combo,
                "Optional: select a pre-trained classifier (.pkl) to score frames. "
                "If left blank (or no classifier selected), the tab trains a fresh model "
                "from your current labels.")

        # Parameters
        pf = ttk.LabelFrame(parent, text="Parameters", padding=5)
        pf.pack(fill='x', padx=4, pady=4)

        def _row(parent, label, var, from_, to, width=6, tooltip=None):
            r = ttk.Frame(parent)
            r.pack(fill='x', pady=1)
            lbl = ttk.Label(r, text=label, width=22)
            lbl.pack(side='left')
            spx = ttk.Spinbox(r, from_=from_, to=to, textvariable=var, width=width)
            spx.pack(side='left')
            if tooltip:
                ToolTip(lbl, tooltip)
                ToolTip(spx, tooltip)
            return r

        _row(pf, "Bouts / iteration:", self._n_suggestions_var, 1, 200,
             tooltip="Number of uncertain video clips to present per labeling round. Lower = shorter sessions; higher = more frames labeled per click.")
        _row(pf, "Min bout frames:", self._min_spacing_var, 1, 100,
             tooltip="Minimum number of consecutive uncertain frames required to form a bout. Shorter runs are ignored.")
        _row(pf, "Context frames:", self._context_frames_var, 0, 300,
             tooltip="Extra frames shown before and after the uncertain region so you can see the behavior in context. Does not affect which frames get labeled.")
        _row(pf, "Label budget (new):", self._budget_var, 10, 5000,
             tooltip="Maximum number of new frames to annotate this session before active learning stops automatically. Frames already labeled when the session was loaded do not count.")
        _row(pf, "Max bout frames (0=auto):", self._max_bout_var, 0, 2000,
             tooltip="Cap the maximum clip length. 0 = auto (uses 90th-percentile of positive bout lengths). Increase if bouts are being cut short; decrease to avoid very long clips.")

        _cb_ba = ttk.Checkbutton(pf, text="Bout-aware CV", variable=self._bout_aware_cv_var)
        _cb_ba.pack(anchor='w', pady=(4, 0))
        ToolTip(_cb_ba, "Use bout-grouped cross-validation (GroupKFold) instead of frame-level "
                        "StratifiedKFold. Recommended — prevents data leakage across bouts. "
                        "Auto-falls back to frame-level when too few bout groups exist.")

        _cb_cb = ttk.Checkbutton(pf, text="Class-balanced queries",
                                  variable=self._class_balanced_var)
        _cb_cb.pack(anchor='w', pady=(2, 0))
        ToolTip(_cb_cb, "Alternate positive-predicted and negative-predicted bouts in each "
                        "query round, so labels stay balanced even when one class dominates.")

        _div_row = ttk.Frame(pf)
        _div_row.pack(fill='x', pady=(2, 0))
        _div_lbl = ttk.Label(_div_row, text="Diversity radius (0=off):", width=22)
        _div_lbl.pack(side='left')
        _div_spx = ttk.Spinbox(_div_row, from_=0, to=500,
                                textvariable=self._diversity_radius_var, width=6)
        _div_spx.pack(side='left')
        ToolTip(_div_lbl, "Minimum frame gap between bouts in a query. "
                          "0 = disabled. Increase (e.g. 150) to spread queries across the video.")
        ToolTip(_div_spx, "Minimum frame gap between bouts in a query. "
                          "0 = disabled. Increase (e.g. 150) to spread queries across the video.")

        _cb_prop = ttk.Checkbutton(pf, text="Label propagation (cosine sim)",
                                    variable=self._propagate_var)
        _cb_prop.pack(anchor='w', pady=(2, 0))
        ToolTip(_cb_prop, "After labeling a frame, auto-label nearby frames with cosine "
                          "similarity ≥ 0.92 in feature space. Speeds up labeling "
                          "when behavior is temporally clustered.")

        _btn_auto = ttk.Button(pf, text="🔍 Auto-detect from labels",
                               command=self._auto_detect_bout_lengths)
        _btn_auto.pack(fill='x', pady=(4, 0))
        ToolTip(_btn_auto, "Scan positive labels in the selected session(s) to detect actual bout lengths "
                           "and set Min/Max bout frames automatically.")

        # Threshold
        tf = ttk.LabelFrame(parent, text="Uncertainty Threshold", padding=5)
        tf.pack(fill='x', padx=4, pady=4)
        thresh_row = ttk.Frame(tf)
        thresh_row.pack(fill='x')
        _thresh_lbl = ttk.Label(thresh_row, text="Threshold:")
        _thresh_lbl.pack(side='left')
        _thresh_scale = ttk.Scale(thresh_row, from_=0.05, to=1.0, variable=self._threshold_var,
                                  orient='horizontal', length=150,
                                  command=lambda _: self._update_eligible_count())
        _thresh_scale.pack(side='left', padx=4)
        ToolTip(_thresh_lbl, "Frames whose model confidence is within this distance of P=0.5 are considered uncertain and eligible for labeling.")
        ToolTip(_thresh_scale, "Frames whose model confidence is within this distance of P=0.5 are considered uncertain and eligible for labeling.")
        ttk.Label(thresh_row, textvariable=tk.StringVar()).pack(side='left')  # placeholder
        ttk.Label(tf, textvariable=self._eligible_count_var,
                  font=('Arial', 9), foreground='navy').pack(anchor='w', pady=2)

        # Buttons
        btn_f = ttk.LabelFrame(parent, text="Actions", padding=5)
        btn_f.pack(fill='x', padx=4, pady=4)
        _btn_score = ttk.Button(btn_f, text="1. Score + Histogram",
                                command=self._score_and_histogram)
        _btn_score.pack(fill='x', pady=2)
        ToolTip(_btn_score, "Train a model on current labels, score every frame, and open the confidence distribution chart.")
        _btn_label = ttk.Button(btn_f, text="2. Start Labeling",
                                command=self._start_labeling)
        _btn_label.pack(fill='x', pady=2)
        ToolTip(_btn_label, "Find the most uncertain video clips and open the bout-labeling interface.")
        _btn_retrain = ttk.Button(btn_f, text="Retrain & Save Snapshot",
                                   command=self._retrain_and_compare)
        _btn_retrain.pack(fill='x', pady=2)
        ToolTip(_btn_retrain, "Retrain on all current labels, save a snapshot pkl to classifiers/, and update the learning curve.")
        self._btn_next_iter = ttk.Button(btn_f, text="Next Iteration →",
                                         command=self._start_labeling,
                                         state='disabled')
        self._btn_next_iter.pack(fill='x', pady=2)
        ToolTip(self._btn_next_iter, "Score frames with the latest model and open another "
                                     "labeling round. Enabled after first scoring or retrain.")

        auto_row = ttk.Frame(btn_f)
        auto_row.pack(fill='x', pady=(4, 2))
        ttk.Button(auto_row, text="🔁 Auto-iterate",
                   command=self._auto_iterate).pack(side='left', fill='x', expand=True)
        ttk.Spinbox(auto_row, from_=1, to=20, textvariable=self._auto_iter_var,
                    width=4).pack(side='left', padx=(4, 0))
        ttk.Label(auto_row, text="iters").pack(side='left', padx=(2, 0))
        ttk.Button(btn_f, text="⏹ Stop Auto",
                   command=lambda: self._stop_auto_var.set(True)).pack(fill='x', pady=(0, 2))

        _btn_disc = ttk.Button(btn_f, text="3. Run Discovery",
                               command=self._run_discovery)
        _btn_disc.pack(fill='x', pady=2)
        ToolTip(_btn_disc, "Use UMAP + HDBSCAN to find sub-behaviors within your positive-labeled frames.")

    def _build_right(self, parent):
        # Learning curve plot
        plot_lf = ttk.LabelFrame(parent, text="Learning Curve", padding=4)
        plot_lf.pack(fill='x', padx=4, pady=4)

        if MATPLOTLIB_AVAILABLE:
            self._lc_fig, self._lc_ax = plt.subplots(figsize=(5, 2.5), dpi=90,
                                                      constrained_layout=True)
            self._lc_canvas = FigureCanvasTkAgg(self._lc_fig, master=plot_lf)
            self._lc_canvas.get_tk_widget().pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(self._lc_canvas, self._lc_fig)
            self._draw_empty_curve()
        else:
            ttk.Label(plot_lf, text="(install matplotlib to see learning curve)").pack()

        # Log
        log_lf = ttk.LabelFrame(parent, text="Log", padding=4)
        log_lf.pack(fill='both', expand=True, padx=4, pady=4)
        from tkinter import scrolledtext
        self._log = scrolledtext.ScrolledText(log_lf, height=14, wrap='word',
                                              font=('Consolas', 9))
        self._log.pack(fill='both', expand=True)

    # ------------------------------------------------------------------
    # Project / session / classifier helpers
    # ------------------------------------------------------------------

    def _on_project_changed(self):
        self._scan_sessions()
        self._refresh_classifiers()

    def _scan_sessions(self):
        folder = self.app.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            return
        if not _FIND_SESSIONS_AVAILABLE:
            self._log_msg("Session discovery unavailable (evaluation_tab not found)")
            return
        try:
            sessions = find_session_triplets(folder, prefer_filtered=True, require_labels=True)
            self._sessions_list = sessions
            self._session_lb.delete(0, 'end')
            n_missing_cache = 0
            for idx, s in enumerate(sessions):
                cache = self._get_features_cache(s)
                if cache:
                    self._session_lb.insert('end', s['session_name'])
                else:
                    self._session_lb.insert('end', f"{s['session_name']}  [no features]")
                    self._session_lb.itemconfig(idx, foreground='red')
                    n_missing_cache += 1
            if sessions:
                self._session_lb.selection_set(0)
            self._log_msg(f"Scanned: {len(sessions)} session(s) found.")
            if n_missing_cache > 0:
                self._log_msg(f"\u26a0 {n_missing_cache} session(s) missing feature cache — extract features first (Train tab).")
        except Exception as e:
            self._log_msg(f"Scan error: {e}")

    def _refresh_classifiers(self):
        folder = self.app.current_project_folder.get()
        clf_dir = os.path.join(folder, 'classifiers')
        self._clf_options = {}
        if os.path.isdir(clf_dir):
            for f in sorted(os.listdir(clf_dir)):
                if f.endswith('.pkl'):
                    self._clf_options[f] = os.path.join(clf_dir, f)
        self._clf_combo['values'] = list(self._clf_options.keys())
        if not self._clf_options:
            return
        # Auto-select by best F1 (oof_best_f1 preferred, fallback mean_cv_f1)
        best_name = None
        best_f1 = -1.0
        for name, path in self._clf_options.items():
            try:
                import pickle as _pk
                with open(path, 'rb') as _f:
                    data = _pk.load(_f)
                if not isinstance(data, dict) or 'clf_model' not in data:
                    continue
                f1 = data.get('oof_best_f1') or data.get('mean_cv_f1')
                if f1 is not None and float(f1) > best_f1:
                    best_f1 = float(f1)
                    best_name = name
            except Exception:
                continue
        if best_name:
            self._clf_combo.set(best_name)
        else:
            self._clf_combo.current(0)   # fallback: no readable F1 found

    def _browse_classifier(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Classifier (.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        name = os.path.basename(path)
        self._clf_options[name] = path
        self._clf_combo['values'] = list(self._clf_options.keys())
        self._clf_combo.set(name)

    def _get_selected_session(self):
        sel = self._session_lb.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx < len(self._sessions_list):
            return self._sessions_list[idx]
        return None

    def _get_selected_sessions(self) -> list:
        indices = self._session_lb.curselection()
        return [self._sessions_list[i] for i in indices if i < len(self._sessions_list)]

    def _get_features_cache(self, session):
        """Find features cache for session."""
        import glob as _glob
        folder = self.app.current_project_folder.get()
        base = session.get('session_name', '')
        search_dirs = [
            os.path.join(folder, 'features'),
            os.path.dirname(session.get('video_path', '')),
        ]
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            matches = _glob.glob(os.path.join(d, f"{base}_features*.pkl"))
            if matches:
                return matches[0]
        return None

    def _load_selected_classifier(self):
        """Return the selected pre-trained classifier, or None to train from labels."""
        name = self._clf_combo.get()
        if name and name in self._clf_options:
            import pickle
            try:
                with open(self._clf_options[name], 'rb') as f:
                    data = pickle.load(f)
                # PixelPaws .pkl files are dicts with 'clf_model' key
                if isinstance(data, dict):
                    return data['clf_model']
                return data
            except Exception as e:
                self._log_msg(f"Warning: could not load classifier '{name}': {e}")
        return None

    def _load_selected_classifier_data(self):
        """Return the full pkl dict for the selected classifier, or None."""
        name = self._clf_combo.get()
        if name and name in self._clf_options:
            import pickle
            try:
                with open(self._clf_options[name], 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'clf_model' in data:
                    return data
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Scoring + histogram
    # ------------------------------------------------------------------

    def _score_and_histogram(self):
        selected = self._get_selected_sessions()
        if not selected:
            messagebox.showwarning("No session", "Please select a session first.")
            return

        # Validate all selected sessions
        for s in selected:
            lcsv = s.get('labels_path') or s.get('target_path')
            fc = self._get_features_cache(s)
            if not lcsv or not os.path.isfile(lcsv):
                messagebox.showerror("Missing file", f"Labels CSV not found:\n{lcsv}")
                return
            if not fc or not os.path.isfile(fc):
                messagebox.showerror("Missing file",
                                     f"Features cache not found for session '{s['session_name']}'.\n"
                                     "Please run feature extraction first (Train tab).")
                return

        self._log_msg("Initializing session and scoring frames...")

        def _run():
            try:
                selected_snap = self._get_selected_sessions()
                import shutil as _shutil, datetime as _dt
                _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                _preAL_backups = []
                for _s in selected_snap:
                    _lcsv = _s.get('labels_path') or _s.get('target_path')
                    if _lcsv and os.path.isfile(_lcsv):
                        _bdir = os.path.join(os.path.dirname(_lcsv), 'label_backups')
                        os.makedirs(_bdir, exist_ok=True)
                        _stem = os.path.splitext(os.path.basename(_lcsv))[0]
                        _dst = os.path.join(_bdir, f'{_stem}_preAL_{_ts}.csv')
                        _shutil.copy2(_lcsv, _dst)
                        _preAL_backups.append(_dst)
                if _preAL_backups:
                    _bmsgs = [os.path.basename(p) for p in _preAL_backups]
                    self.app.root.after(0, lambda _bmsgs=_bmsgs: self._log_msg(
                        "Label backup(s): " + ", ".join(_bmsgs)))
                if len(selected_snap) == 1:
                    s = selected_snap[0]
                    labels_csv = s.get('labels_path') or s.get('target_path')
                    video_path = s.get('video_path', '')
                    features_cache = self._get_features_cache(s)
                    sess = ALSessionV2(
                        labels_csv=labels_csv,
                        video_path=video_path,
                        features_cache=features_cache,
                        min_frame_spacing=self._min_spacing_var.get()
                    )
                else:
                    from active_learning_v2 import MultiSessionAL
                    sess = MultiSessionAL([{
                        'labels_csv': s.get('labels_path') or s.get('target_path'),
                        'video_path': s.get('video_path', ''),
                        'features_cache': self._get_features_cache(s),
                    } for s in selected_snap], min_frame_spacing=self._min_spacing_var.get())
                    labels_csv = (selected_snap[0].get('labels_path') or
                                  selected_snap[0].get('target_path'))
                self._session = sess

                # Check for feature-label truncation warnings
                if hasattr(sess, '_truncation_warning'):
                    warn = getattr(sess, '_truncation_warning', None)
                    if warn and warn[1] > 0:
                        _tw = warn
                        self.app.root.after(0, lambda _tw=_tw: messagebox.showwarning(
                            "Label Truncation",
                            f"Feature cache is shorter than labels CSV by {_tw[0]} rows.\n"
                            f"{_tw[1]} labeled frames beyond the feature range will be ignored.\n\n"
                            "Re-extract features to include all frames."))
                elif hasattr(sess, '_subs'):
                    for _sub in sess._subs:
                        _tw = _sub.get('_truncation_warning')
                        if _tw and _tw[1] > 0:
                            _sname = os.path.basename(_sub.get('video_path', ''))
                            self.app.root.after(0, lambda _tw=_tw, _sn=_sname: messagebox.showwarning(
                                "Label Truncation",
                                f"Session '{_sn}': feature cache is shorter than labels CSV by {_tw[0]} rows.\n"
                                f"{_tw[1]} labeled frames beyond the feature range will be ignored.\n\n"
                                "Re-extract features to include all frames."))

                # Capture how many frames were already labeled at load time
                # so the budget only counts newly annotated frames this session
                if hasattr(sess, '_labels'):
                    self._n_labeled_at_load = int(np.sum(sess._labels >= 0))
                elif hasattr(sess, '_subs'):
                    self._n_labeled_at_load = sum(int(np.sum(sub['labels'] >= 0))
                                                   for sub in sess._subs)
                else:
                    self._n_labeled_at_load = 0

                # Diagnostic: show label breakdown after loading
                if hasattr(sess, '_labels'):
                    import numpy as _np2
                    _lbl = sess._labels
                    _n_pos = int(_np2.sum(_lbl == 1))
                    _n_neg = int(_np2.sum(_lbl == 0))
                    _n_unl = int(_np2.sum(_lbl < 0))
                    _n_feat = len(sess._features)
                    _msg = (f"Labels loaded: {_n_pos} positive, {_n_neg} negative, "
                            f"{_n_unl} unlabeled (total {len(_lbl)})")
                    if _n_feat < len(_lbl):
                        _msg += (f" — features cover only {_n_feat} frames; "
                                 f"{len(_lbl) - _n_feat} unlabeled tail frames "
                                 f"scored at max uncertainty")
                    self.app.root.after(0, lambda: self._log_msg(_msg))

                # Load curve from previous session if exists
                curve_path = self._get_curve_path()
                if os.path.isfile(curve_path):
                    try:
                        sess.tracker.load(curve_path)
                        self.app.root.after(0, lambda: self._log_msg(
                            f"Loaded {len(sess.tracker.records)} previous iteration(s) from curve."))
                        self.app.root.after(0, self._refresh_plot)
                    except Exception:
                        pass

                clf_override = self._load_selected_classifier()
                if clf_override is not None:
                    model = clf_override
                    self.app.root.after(0, lambda: self._log_msg(
                        f"Using pre-trained classifier: {self._clf_combo.get()}"))
                    # Reindex features to match SHAP-pruned classifier's column list
                    clf_data_full = self._load_selected_classifier_data()
                    sel_cols = clf_data_full.get('selected_feature_cols') if clf_data_full else None
                    base_f1 = clf_data_full.get('mean_cv_f1') if clf_data_full else None
                    self.app.root.after(0, lambda v=base_f1: setattr(self, '_base_clf_f1', v))
                    if sel_cols and hasattr(sess, '_feature_cols') and sess._feature_cols:
                        import pandas as _pd
                        feat_df = _pd.DataFrame(sess._features, columns=sess._feature_cols)
                        # Compute post-cache features (lag, contact) that model may need
                        feat_df = augment_features_post_cache(
                            feat_df, clf_data_full, model,
                            getattr(sess, '_dlc_path', '') or '')
                        missing = [c for c in sel_cols if c not in feat_df.columns]
                        if missing:
                            self.app.root.after(0, lambda: self._log_msg(
                                f"⚠ Classifier expects {len(missing)} feature(s) not in cache — predictions may be unreliable."))
                        feat_df = feat_df.reindex(columns=sel_cols, fill_value=0.0)
                        sess._features = feat_df.values
                        sess._feature_cols = sel_cols
                else:
                    model = sess.train_model()
                    self._base_clf_f1 = None
                if hasattr(sess, 'get_full_probas'):
                    probas = sess.get_full_probas(model)
                else:
                    # MultiSessionAL: concatenate probas from all sub-sessions
                    import numpy as _np
                    from active_learning_v2 import _align_features as _al_align
                    probas = _np.concatenate([
                        model.predict_proba(
                            _al_align(model, sub['features'], sub.get('feature_cols')))[:, 1]
                        for sub in sess._subs])
                self._last_probas = probas
                self._last_model = model
                self.app.root.after(0, lambda: self._btn_next_iter.configure(state='normal'))

                threshold = self._threshold_var.get()
                n_frames, n_bouts_eligible, bout_stats = sess.count_eligible(
                    probas, threshold, self._min_spacing_var.get())
                n_eligible = n_frames
                n_pos = sess.count_positive()
                _msg = f"{n_pos:,} pos labeled | {n_eligible:,} uncertain in {n_bouts_eligible} bouts"
                self.app.root.after(0, lambda: self._eligible_count_var.set(_msg))
                self.app.root.after(0, lambda: self._log_msg(
                    f"Scored {len(probas):,} frames. "
                    f"{n_pos:,} pos labeled. "
                    f"{n_eligible:,} eligible unlabeled frames in {n_bouts_eligible} bouts "
                    f"at threshold={threshold:.2f}"))
                # --- Convergence hint (A-SOID-style) ---
                _n_sugg = self._n_suggestions_var.get()
                if 0 < n_bouts_eligible < _n_sugg:
                    self.app.root.after(0, lambda nb=n_bouts_eligible, ns=_n_sugg: self._log_msg(
                        f"⚑ Only {nb} uncertain bout(s) remain (< {ns} requested) — "
                        f"model may be converging. Consider stopping or lowering the "
                        f"confidence threshold to find more candidates."))
                elif n_eligible > 0 and (n_eligible / max(len(probas), 1)) < 0.02:
                    self.app.root.after(0, lambda ne=n_eligible, nt=len(probas): self._log_msg(
                        f"⚑ Only {ne:,} / {nt:,} frames ({ne/nt*100:.1f}%) remain uncertain — "
                        f"model is converging."))
                if n_bouts_eligible == 0:
                    _n_runs = bout_stats.get('n_runs', 0)
                    _n_short = bout_stats.get('n_too_short', 0)
                    _min_bf = self._min_spacing_var.get()
                    if _n_runs == 0:
                        self.app.root.after(0, lambda: self._log_msg(
                            "  → No unlabeled frame runs found. Session may be fully labeled."))
                    else:
                        self.app.root.after(0, lambda _n_runs=_n_runs, _n_short=_n_short, _min_bf=_min_bf: self._log_msg(
                            f"  → {_n_runs} unlabeled run(s) found but all filtered: "
                            f"{_n_short} too short (<{_min_bf} frames). "
                            f"Try lowering 'Min bout frames'."))
                self.app.root.after(0, self._show_histogram)
            except Exception as e:
                _e = e
                self.app.root.after(0, lambda _e=_e: messagebox.showerror("Error", str(_e)))
                self.app.root.after(0, lambda _e=_e: self._log_msg(f"Error: {_e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_histogram(self):
        if self._last_probas is None:
            return
        ConfidenceHistogramDialog(
            parent_root=self.app.root,
            probas=self._last_probas,
            threshold_var=self._threshold_var,
            on_proceed=None,
            on_cancel=None,
        )

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    def _start_labeling(self):
        if self._session is None or self._last_probas is None:
            messagebox.showwarning("Not scored", "Run 'Score + Histogram' first.")
            return

        selected = self._get_selected_sessions()
        if not selected:
            return

        # For single session validate video path upfront
        if len(selected) == 1:
            video_path = selected[0].get('video_path', '')
            if not os.path.isfile(video_path):
                messagebox.showerror("Missing file", f"Video not found:\n{video_path}")
                return
        else:
            video_path = selected[0].get('video_path', '')

        # Check budget (count only frames newly labeled this session)
        if hasattr(self._session, '_labels'):
            n_labeled = int(np.sum(self._session._labels >= 0))
        else:
            n_labeled = sum(int(np.sum(sub['labels'] >= 0))
                            for sub in self._session._subs)
        n_new = n_labeled - getattr(self, '_n_labeled_at_load', 0)
        if n_new >= self._budget_var.get():
            if not messagebox.askyesno("Budget reached",
                                       f"Label budget ({self._budget_var.get()} new frames) reached.\nContinue anyway?"):
                return

        threshold = self._threshold_var.get()
        n_bouts = self._n_suggestions_var.get()
        min_bout_frames = self._min_spacing_var.get()
        context_frames = self._context_frames_var.get()
        class_balanced = self._class_balanced_var.get()
        diversity_radius = self._diversity_radius_var.get()

        def _run():
            try:
                clf_override = self._load_selected_classifier()
                result = self._session.run_one_iteration(
                    n_bouts=n_bouts,
                    confidence_threshold=threshold,
                    min_bout_frames=min_bout_frames,
                    context_frames=context_frames,
                    max_bout_frames=self._max_bout_var.get() or None,  # None → adaptive
                    class_balanced=class_balanced,
                    diversity_radius=diversity_radius,
                    model=clf_override,   # None = retrain from labels
                )
                bouts = result['bouts']
                self._last_probas = result['probas']
                self.app.root.after(0, lambda: self._btn_next_iter.configure(state='normal'))

                if len(bouts) == 0:
                    self.app.root.after(0, lambda: messagebox.showinfo(
                        "Converged", "No uncertain bouts remain. Model has converged!"))
                    return

                self.app.root.after(0, lambda: self._run_bout_labeling_ui(
                    bouts, result['probas'], video_path))
            except Exception as e:
                import traceback
                traceback.print_exc()
                _e = e
                self.app.root.after(0, lambda _e=_e: messagebox.showerror("Error", str(_e)))

        threading.Thread(target=_run, daemon=True).start()

    def _retrain_and_compare(self, _auto_continue=False):
        if self._session is None:
            messagebox.showwarning("No session", "Run '1. Score + Histogram' first.")
            return

        # Pre-flight: need positive + negative labels
        if hasattr(self._session, '_labels'):
            lbl = self._session._labels
            n_pos = int((lbl == 1).sum()); n_neg = int((lbl == 0).sum())
        else:
            n_pos = sum(int((s['labels'] == 1).sum()) for s in self._session._subs)
            n_neg = sum(int((s['labels'] == 0).sum()) for s in self._session._subs)
        if n_pos == 0:
            messagebox.showwarning("No positive labels",
                                   "Label at least one YES bout before retraining.")
            return
        if n_neg == 0:
            messagebox.showwarning("No negative labels",
                                   "Label at least one NO bout before retraining.")
            return

        # Read UI vars on main thread
        threshold_var_val = self._threshold_var.get()
        pf = self.app.current_project_folder.get()
        snap_dir = os.path.join(pf, 'classifiers') if pf else None
        if not snap_dir:
            self._log_msg("⚠ No project folder — classifier will not be saved.")
        behavior_name = getattr(self._session, 'behavior_name', 'behavior')
        base_clf_data = self._load_selected_classifier_data()  # full dict or None
        _al_min_bout = int(base_clf_data.get('min_bout', 1)) if base_clf_data else 1

        self._log_msg("Retraining (full pipeline)…")

        def _run():
            try:
                from xgboost import XGBClassifier
                from sklearn.model_selection import StratifiedKFold
                from sklearn.metrics import f1_score as _f1

                # ── Backup label CSVs before this retrain iteration ──────────────
                import shutil as _shutil, datetime as _dt
                _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                _selected_snap = self._get_selected_sessions() if hasattr(self, '_get_selected_sessions') else []
                _backup_paths = []
                for _s in _selected_snap:
                    _lcsv = _s.get('labels_path') or _s.get('target_path')
                    if _lcsv and os.path.isfile(_lcsv):
                        _bdir = os.path.join(os.path.dirname(_lcsv), 'label_backups')
                        os.makedirs(_bdir, exist_ok=True)
                        _stem = os.path.splitext(os.path.basename(_lcsv))[0]
                        _bdst = os.path.join(_bdir, f'{_stem}_backup_{_ts}.csv')
                        _shutil.copy2(_lcsv, _bdst)
                        _backup_paths.append(_bdst)

                # --- Gather labeled data ---
                if hasattr(self._session, '_labels'):
                    mask = self._session._labels >= 0
                    X = self._session._features[mask]
                    y = self._session._labels[mask]
                    _lab_indices = np.where(self._session._labels >= 0)[0]
                else:
                    import pandas as _pd_fin
                    _sub_dfs_lbl = []
                    ys = []
                    _lab_idx_parts = []
                    _offset = 0
                    for sub in self._session._subs:
                        m = sub['labels'] >= 0
                        if m.any():
                            _sub_dfs_lbl.append(_pd_fin.DataFrame(
                                sub['features'][m],
                                columns=sub.get('feature_cols') or range(sub['features'].shape[1])))
                            ys.append(sub['labels'][m])
                        _idx = np.where(sub['labels'] >= 0)[0] + _offset
                        _lab_idx_parts.append(_idx)
                        _offset += len(sub['labels'])
                    X_df = _pd_fin.concat(_sub_dfs_lbl, ignore_index=True).fillna(0.0)
                    feature_cols = list(X_df.columns)
                    X = X_df.values.astype(np.float32)
                    y = np.concatenate(ys)
                    _lab_indices = np.concatenate(_lab_idx_parts) if _lab_idx_parts else np.array([], dtype=int)

                n_labeled = len(y)
                if not hasattr(self._session, '_labels'):
                    pass  # feature_cols already set above
                else:
                    feature_cols = getattr(self._session, '_feature_cols', None)

                # --- Class imbalance weight (mirrors full training pipeline) ---
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

                # --- 3-fold CV for OOF predictions (bout-aware if possible) ---
                from sklearn.model_selection import GroupKFold
                from active_learning_v2 import _make_bout_groups
                n_splits = min(3, int((y == 1).sum()), int((y == 0).sum()))
                n_splits = max(n_splits, 2)
                _groups = _make_bout_groups(_lab_indices, _al_min_bout)
                _n_groups = int(_groups.max()) + 1 if len(_groups) > 0 else 0
                _cv_mode = 'frame-level'
                oof_proba = np.full(n_labeled, 0.5)
                fold_f1s = []
                if self._bout_aware_cv_var.get() and _n_groups >= n_splits:
                    _gkf = GroupKFold(n_splits=n_splits)
                    for tr_idx, val_idx in _gkf.split(X, y, groups=_groups):
                        fold_clf = XGBClassifier(n_estimators=200, max_depth=6,
                                                 learning_rate=0.1, scale_pos_weight=spw,
                                                 random_state=42, verbosity=0)
                        fold_clf.fit(X[tr_idx], y[tr_idx])
                        oof_proba[val_idx] = fold_clf.predict_proba(X[val_idx])[:, 1]
                        preds = (oof_proba[val_idx] >= 0.5).astype(int)
                        fold_f1s.append(float(_f1(y[val_idx], preds, zero_division=0)))
                    _cv_mode = f'bout-aware ({_n_groups} groups)'
                else:
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    for tr_idx, val_idx in skf.split(X, y):
                        fold_clf = XGBClassifier(n_estimators=200, max_depth=6,
                                                 learning_rate=0.1, scale_pos_weight=spw,
                                                 random_state=42, verbosity=0)
                        fold_clf.fit(X[tr_idx], y[tr_idx])
                        oof_proba[val_idx] = fold_clf.predict_proba(X[val_idx])[:, 1]
                        preds = (oof_proba[val_idx] >= 0.5).astype(int)
                        fold_f1s.append(float(_f1(y[val_idx], preds, zero_division=0)))
                    if not self._bout_aware_cv_var.get():
                        _cv_mode = 'frame-level (manual)'

                mean_cv_f1 = float(np.mean(fold_f1s))
                std_cv_f1  = float(np.std(fold_f1s))

                # --- OOF parameter sweep ---
                best_params = self.app._sweep_postprocessing(oof_proba, y)

                # --- Final model on all labeled data ---
                final_clf = XGBClassifier(n_estimators=300, max_depth=6,
                                           learning_rate=0.1, scale_pos_weight=spw,
                                           random_state=42, verbosity=0)
                if feature_cols:
                    import pandas as _pd_fin
                    final_clf.fit(_pd_fin.DataFrame(X, columns=feature_cols), y)
                else:
                    final_clf.fit(X, y)
                from active_learning_v2 import _align_features as _al_align
                if hasattr(self._session, '_features'):
                    probas_all = final_clf.predict_proba(
                        _al_align(final_clf, self._session._features,
                                  getattr(self._session, '_feature_cols', None)))[:, 1]
                else:
                    import pandas as _pd_fin
                    _all_dfs = []
                    for _s in self._session._subs:
                        _all_dfs.append(_pd_fin.DataFrame(
                            _s['features'],
                            columns=_s.get('feature_cols') or range(_s['features'].shape[1])))
                    _X_all = _pd_fin.concat(_all_dfs, ignore_index=True).fillna(0.0)
                    if hasattr(final_clf, 'feature_names_in_'):
                        _X_all = _X_all.reindex(columns=final_clf.feature_names_in_, fill_value=0.0)
                    probas_all = final_clf.predict_proba(_X_all)[:, 1]

                # --- Build full classifier_data ---
                def _get(key, default=None):
                    return base_clf_data.get(key, default) if base_clf_data else default

                clf_data = {
                    'clf_model':             final_clf,
                    'Behavior_type':         behavior_name,
                    'selected_feature_cols': feature_cols,
                    'best_thresh':           best_params['thresh'],
                    'min_bout':              best_params['min_bout'],
                    'min_after_bout':        best_params['min_after_bout'],
                    'max_gap':               best_params['max_gap'],
                    'ui_min_bout':           best_params['min_bout'],
                    'ui_min_after_bout':     _get('ui_min_after_bout', 1),
                    'ui_max_gap':            best_params['max_gap'],
                    'bp_include_list':       _get('bp_include_list'),
                    'bp_pixbrt_list':        _get('bp_pixbrt_list', []),
                    'square_size':           _get('square_size', [40]),
                    'pix_threshold':         _get('pix_threshold', 0.3),
                    'include_optical_flow':  _get('include_optical_flow', True),
                    'bp_optflow_list':       _get('bp_optflow_list', []),
                    # Provenance
                    'training_source':       'active_learning',
                    'n_labeled_total':       n_labeled,
                    'n_positive':            int((y == 1).sum()),
                    'cv_f1_scores':          fold_f1s,
                    'mean_cv_f1':            mean_cv_f1,
                    'std_cv_f1':             std_cv_f1,
                    'oof_best_f1':           best_params['f1'],
                }

                # --- Save ---
                saved_path = None
                if snap_dir:
                    os.makedirs(snap_dir, exist_ok=True)
                    fname = f"PixelPaws_{behavior_name}_AL.pkl"
                    saved_path = os.path.join(snap_dir, fname)
                    _atomic_pickle_save(clf_data, saved_path)

                # --- Learning curve record (for plot) ---
                n_below = int(np.sum(np.abs(probas_all - 0.5) * 2 < threshold_var_val))
                record = self._session.tracker.record(
                    final_clf, X, y, n_below,
                    labels_array=getattr(self._session, '_labels', None),
                    min_bout=_al_min_bout)
                self._session.tracker.save(self._get_curve_path())
                self._last_probas = probas_all
                self._last_model  = final_clf

                # --- Log ---
                def _post_log():
                    self._log_msg("=" * 52)
                    self._log_msg(f"  RETRAIN COMPLETE — iteration {self._session._iteration}")
                    self._log_msg("=" * 52)
                    self._log_msg(f"  Labeled frames : {n_labeled}  "
                                  f"(+{int((y==1).sum())}  /  -{int((y==0).sum())})")
                    self._log_msg(f"  Class balance  : 1:{spw:.1f}  (neg/pos weight)")
                    self._log_msg(f"  CV mode        : {_cv_mode}")
                    self._log_msg(f"  CV F1 @ 0.5    : {mean_cv_f1:.3f} ± {std_cv_f1:.3f}  "
                                  f"[{', '.join(f'{v:.3f}' for v in fold_f1s)}]")
                    self._log_msg(f"  OOF F1 (tuned) : {best_params['f1']:.3f}")
                    self._log_msg(f"  thresh         : {best_params['thresh']:.2f}  "
                                  f"min_bout={best_params['min_bout']}  "
                                  f"max_gap={best_params['max_gap']}")
                    if saved_path:
                        self._log_msg(f"  Saved → {os.path.basename(saved_path)}")
                    else:
                        self._log_msg("  ⚠ No project folder — classifier not saved.")
                    if _backup_paths:
                        self._log_msg("  Label backups → " +
                                      ", ".join(os.path.basename(p) for p in _backup_paths))
                    if self._btn_next_iter:
                        self._btn_next_iter.configure(state='normal')

                self.app.root.after(0, _post_log)
                self.app.root.after(0, self._refresh_plot)
                self.app.root.after(0, self._refresh_classifiers)
                if _auto_continue:
                    self.app.root.after(0, self._do_auto_next)

            except Exception as e:
                import traceback; traceback.print_exc()
                err = str(e)
                self.app.root.after(0, lambda: self._log_msg(f"✗ Retrain failed: {err}"))
                self.app.root.after(0, lambda: messagebox.showerror("Retrain error", err))

        threading.Thread(target=_run, daemon=True).start()

    def _auto_iterate(self):
        if self._session is None:
            messagebox.showwarning("No session", "Load a session first (Score + Histogram).")
            return
        self._auto_remaining = self._auto_iter_var.get()
        self._stop_auto_var.set(False)
        self._log_msg(f"Auto-iterate: {self._auto_remaining} iteration(s) queued.")
        self._start_labeling()

    def _do_auto_next(self):
        """Called after each auto-mode retrain; chains next iteration if remaining."""
        if self._auto_remaining <= 0 or self._stop_auto_var.get():
            self._log_msg("Auto-iterate: done.")
            return
        if (self._last_probas is not None and self._session is not None and
                self._session.is_converged(self._last_probas, self._threshold_var.get())):
            self._log_msg("Auto-iterate: model converged — stopping early.")
            return
        self._log_msg(f"Auto-iterate: {self._auto_remaining} iteration(s) remaining → next round...")
        self._start_labeling()

    def _run_bout_labeling_ui(self, bouts, probas, video_path):
        """Launch BoutLabelingInterface on the main thread, then apply labels."""
        bname = self._session.behavior_name if self._session else "behavior"

        # Read fps from video
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
        _cap.release()

        interface = BoutLabelingInterface(
            video_path=video_path,
            bouts=bouts,
            probas=probas,
            behavior_name=bname,
            fps=fps,
        )
        new_labels = interface.run()  # {(start, end): 0 or 1}

        if not new_labels:
            self._log_msg("No bouts labeled.")
            return

        try:
            stats = self._session.apply_labels(
                new_labels=new_labels,
                confidence_threshold=self._threshold_var.get(),
                propagate=self._propagate_var.get(),
                probas=self._last_probas,
            )
            # Save curve
            self._session.tracker.save(self._get_curve_path())

            n_bouts_labeled = len(new_labels)
            msg = (f"Labeled {n_bouts_labeled} bout(s). "
                   f"Total labeled frames: {stats.get('n_labeled_total', 0)}.")
            self._log_msg(msg)
            self._refresh_plot()
            self._check_convergence(stats)
            # Auto-retrain: rebuild classifier if both label classes present
            if hasattr(self._session, '_subs'):
                _n_pos = sum(int((s['labels'] == 1).sum()) for s in self._session._subs)
                _n_neg = sum(int((s['labels'] == 0).sum()) for s in self._session._subs)
            else:
                _n_pos = int((self._session._labels == 1).sum())
                _n_neg = int((self._session._labels == 0).sum())
            if _n_pos > 0 and _n_neg > 0:
                if self._auto_remaining > 0:
                    # Auto mode: skip dialog, retrain and chain
                    self._auto_remaining -= 1
                    self._log_msg("Auto mode: retraining...")
                    self._retrain_and_compare(_auto_continue=True)
                else:
                    do_retrain = messagebox.askyesno(
                        "Retrain?",
                        f"Labels applied: {len(new_labels)} bout(s).\n"
                        f"Total: {_n_pos} positive, {_n_neg} negative frames.\n\n"
                        "Retrain classifier now?\n"
                        "(Choose 'No' to review labels first — retrain manually later.)")
                    if do_retrain:
                        self._log_msg("Retraining classifier...")
                        self._retrain_and_compare()
                    else:
                        self._log_msg("Retrain deferred — click 'Retrain & Save Snapshot' when ready.")
            else:
                self._log_msg("Auto-retrain skipped — need at least one YES and one NO bout.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error applying labels", str(e))

    def _auto_detect_bout_lengths(self):
        """Scan label data to set Min/Max bout frames from actual positive bouts."""
        import numpy as _np

        # --- Gather label sources: (labels_array, identifier, video_path_or_None) ---
        sources = []
        if self._session is not None:
            # Session already loaded (post-scoring): read in-memory arrays
            if hasattr(self._session, '_subs'):
                for sub in self._session._subs:
                    sources.append((sub['labels'],
                                    os.path.basename(sub['video_path']),
                                    sub['video_path']))
            else:
                sources.append((self._session._labels, "(loaded session)", None))
        else:
            # Pre-scoring fallback: read label CSVs directly from selected sessions
            selected = self._get_selected_sessions()
            if not selected:
                messagebox.showwarning("No session selected",
                    "Select a session in the list first.")
                return
            for s in selected:
                lcsv = s.get('labels') or s.get('target_path')
                if not lcsv or not os.path.isfile(lcsv):
                    self._log_msg(f"  ⚠ Labels CSV not found: {lcsv}")
                    continue
                try:
                    import pandas as _pd
                    df = _pd.read_csv(lcsv)
                    raw = df[df.columns[0]].values
                    labels = _np.where(_np.isnan(raw.astype(float)), -1, raw.astype(int))
                    vpath = s.get('video') or s.get('video_path')
                    sources.append((labels, s['session_name'], vpath))
                except Exception as e:
                    self._log_msg(f"  ⚠ Could not read {lcsv}: {e}")

        if not sources:
            messagebox.showinfo("No labels", "No readable label files found.")
            return

        # --- FPS helper ---
        def _fps_for(vpath):
            if not vpath:
                return None
            try:
                import cv2 as _cv2
                cap = _cv2.VideoCapture(vpath)
                fps = cap.get(_cv2.CAP_PROP_FPS)
                cap.release()
                return float(fps) if fps and fps > 0 else None
            except Exception:
                return None

        def _fmt_bout(length, start, ident, vpath):
            fps = _fps_for(vpath)
            loc = f"frame {start}"
            if fps:
                loc += f" / {start / fps:.1f} s"
            return f"{length} frames, starts {loc} — \"{ident}\""

        # --- Find positive bout records: (length, start_frame, identifier, video_path) ---
        bout_records = []
        for labels, ident, vpath in sources:
            pos = (labels == 1).astype(int)
            if pos.sum() == 0:
                continue
            padded = _np.concatenate([[0], pos, [0]])
            starts = _np.where(_np.diff(padded) == 1)[0]
            ends   = _np.where(_np.diff(padded) == -1)[0]
            for s, e in zip(starts, ends):
                bout_records.append((int(e - s), int(s), ident, vpath))

        if not bout_records:
            messagebox.showinfo("No labels",
                "No positive-labeled frames found in the selected session(s).")
            return

        arr = _np.array([r[0] for r in bout_records])
        min_len   = int(arr.min())
        pct90_len = int(_np.percentile(arr, 90))
        max_len   = int(arr.max())
        median    = int(_np.median(arr))

        min_rec = next(r for r in bout_records if r[0] == min_len)
        max_rec = next(r for r in bout_records if r[0] == max_len)

        self._min_spacing_var.set(max(1, min_len))
        self._max_bout_var.set(pct90_len)

        self._log_msg(
            f"Auto-detected {len(arr)} positive bouts — "
            f"min={min_len}  median={median}  90th-pct={pct90_len}  max={max_len} frames\n"
            f"  min bout: {_fmt_bout(*min_rec)}\n"
            f"  max bout: {_fmt_bout(*max_rec)}\n"
            f"  → Min bout frames set to {max(1, min_len)}, "
            f"Max bout frames set to {pct90_len}"
        )

    def _check_convergence(self, stats):
        """Check auto-convergence and plateau after labeling."""
        if self._last_probas is not None:
            if self._session.is_converged(self._last_probas, self._threshold_var.get()):
                if messagebox.askyesno("Converged",
                        "No uncertain frames remain.\n\nSave final classifier now?"):
                    self._retrain_and_compare()
                return

        records = self._session.tracker.records
        if len(records) >= 3:
            last_3_cv = [r.oof_f1 for r in records[-3:] if r.oof_f1 is not None]
            if len(last_3_cv) == 3 and (max(last_3_cv) - min(last_3_cv)) < 0.01:
                if messagebox.askyesno("Plateau Detected",
                        f"OOF F1 stable at ~{last_3_cv[-1]:.3f} for 3 iterations.\n\n"
                        "Save final classifier and stop?"):
                    self._retrain_and_compare()
                    self._log_msg("Convergence — final classifier saved.")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _run_discovery(self):
        if self._session is None:
            messagebox.showwarning("No session", "Score a session first.")
            return

        project_folder = self.app.current_project_folder.get()
        if hasattr(self._session, 'labels_csv'):
            labels_csv = self._session.labels_csv
            features_cache = self._session.features_cache
            behavior_name = self._session.behavior_name
        else:
            # MultiSessionAL — use first sub-session
            first = self._session._subs[0]
            labels_csv = first['labels_csv']
            features_cache = first['features_cache']
            behavior_name = self._session.behavior_name

        self._log_msg("Starting directed discovery (UMAP + HDBSCAN on positive frames)...")

        def _run():
            out = run_directed_discovery(
                project_folder=project_folder,
                labels_csv=labels_csv,
                features_cache=features_cache,
                behavior_name=behavior_name,
                run_name='al_discovery',
            )
            if out:
                self.app.root.after(0, lambda: self._log_msg(f"Discovery complete: {out}"))
                self.app.root.after(0, lambda: messagebox.showinfo(
                    "Discovery complete",
                    f"Sub-behavior clusters saved to:\n{out}\n\n"
                    "Open the Discover tab to visualize clusters."))
            else:
                self.app.root.after(0, lambda: self._log_msg(
                    "Discovery failed or insufficient positive frames (need >=50). "
                    "Ensure umap-learn and hdbscan are installed."))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Threshold + eligible count
    # ------------------------------------------------------------------

    def _update_eligible_count(self):
        if self._last_probas is None or self._session is None:
            return
        t = self._threshold_var.get()
        n_frames, n_bouts, _ = self._session.count_eligible(
            self._last_probas, t, self._min_spacing_var.get())
        n_pos = self._session.count_positive()
        self._eligible_count_var.set(f"{n_pos:,} pos labeled | {n_frames:,} uncertain in {n_bouts} bouts")

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_empty_curve(self):
        ax = self._lc_ax
        ax.clear()
        ax.text(0.5, 0.5, "No iterations yet\nRun an iteration to build the curve",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='#888888', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        self._lc_canvas.draw()

    def _refresh_plot(self):
        if not MATPLOTLIB_AVAILABLE or self._session is None:
            return
        tracker = self._session.tracker
        if not tracker.records:
            self._draw_empty_curve()
            return

        import numpy as _np
        df = tracker.to_dataframe()
        ax = self._lc_ax
        ax.clear()

        # --- Style ---
        ax.grid(True, alpha=0.3, zorder=0)

        # --- X axis = iteration number (avoids the giant frame-count range) ---
        iters = df['iteration'].values.astype(int)
        train_f1 = df['train_f1'].values

        cv_rows = df.dropna(subset=['oof_f1'])
        cv_iters = cv_rows['iteration'].values.astype(int)
        cv_f1 = cv_rows['oof_f1'].values

        # Filled confidence band between train and CV (like A-SOiD)
        if len(cv_rows) > 0:
            # Build aligned arrays for fill
            _cv_dict = dict(zip(cv_iters, cv_f1))
            _aligned_cv = _np.array([_cv_dict.get(i, _np.nan) for i in iters])
            _valid = ~_np.isnan(_aligned_cv)
            if _valid.sum() > 0:
                _ix = iters[_valid]
                _tr = train_f1[_valid]
                _cv = _aligned_cv[_valid]
                _max_gap = (_tr - _cv).max()
                _fill_color = ('#d62728' if _max_gap >= 0.2
                               else ('#ff7f0e' if _max_gap >= 0.1 else '#2ca02c'))
                ax.fill_between(_ix, _cv, _tr, alpha=0.18, color=_fill_color,
                                zorder=1, label=None)

        # Training fit line
        ax.plot(iters, train_f1, color='#1f77b4', linewidth=1.8,
                marker='o', markersize=5, zorder=4, label='Train F1')

        # CV F1 line
        if len(cv_rows) > 0:
            ax.plot(cv_iters, cv_f1, color='#2ca02c', linewidth=1.8,
                    marker='s', markersize=5, zorder=4, label='OOF F1')

        # Precision & Recall lines
        if 'oof_precision' in df.columns and 'oof_recall' in df.columns:
            _pr_rows = df.dropna(subset=['oof_precision', 'oof_recall'])
            if not _pr_rows.empty:
                _pr_iters = _pr_rows['iteration'].values.astype(int)
                ax.plot(_pr_iters, _pr_rows['oof_precision'].values, 's--',
                        color='steelblue', linewidth=1.2, alpha=0.7,
                        markersize=4, zorder=3, label='OOF Precision')
                ax.plot(_pr_iters, _pr_rows['oof_recall'].values, '^--',
                        color='darkorange', linewidth=1.2, alpha=0.7,
                        markersize=4, zorder=3, label='OOF Recall')

        # Per-point n_labeled annotation (small, above each marker)
        for _, row in df.iterrows():
            n = int(row['n_labeled_total'])
            label_str = f"{n//1000}k" if n >= 1000 else str(n)
            ax.annotate(label_str,
                        (int(row['iteration']), row['train_f1']),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=6, color='#555555', ha='center', zorder=5)

        # Overfitting badge (top-left corner, only when gap is real)
        _gap_rows = df.dropna(subset=['oof_f1'])
        if not _gap_rows.empty:
            _max_gap = (_gap_rows['train_f1'] - _gap_rows['oof_f1']).max()
            if _max_gap >= 0.2:
                ax.text(0.02, 0.97, f"overfit gap {_max_gap:.2f}",
                        transform=ax.transAxes, fontsize=6.5,
                        color='#d62728', va='top', style='italic',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#fff0f0', ec='#d62728',
                                  alpha=0.8))

        # Baseline classifier reference line
        if self._base_clf_f1 is not None:
            ax.axhline(self._base_clf_f1, color='#e6a817', linestyle=':',
                       linewidth=1.4, zorder=2,
                       label=f'Baseline F1={self._base_clf_f1:.3f}')

        # --- Axes ---
        ax.set_xlim(iters.min() - 0.5, iters.max() + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.set_title("Learning Curve", fontsize=9, pad=4)
        # Integer x-ticks only
        ax.set_xticks(iters)
        ax.legend(fontsize=7, framealpha=0.7, loc='lower right')
        self._lc_canvas.draw()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}\n"
        try:
            self._log.insert('end', full)
            self._log.see('end')
        except Exception:
            pass

    @staticmethod
    def _curve_path(labels_csv: str) -> str:
        base = os.path.splitext(labels_csv)[0]
        return base + '_al_curve.json'

    def _get_curve_path(self):
        """Return curve JSON path for current session (single or multi)."""
        if hasattr(self._session, 'labels_csv'):
            return self._curve_path(self._session.labels_csv)
        # Multi-session: derive from project folder + behavior name
        folder = self.app.current_project_folder.get()
        bname = getattr(self._session, 'behavior_name', 'behavior')
        return os.path.join(folder, 'features', f'{bname}_multi_al_curve.json')


# ============================================================================
# Main Application Entry Point
# ============================================================================
