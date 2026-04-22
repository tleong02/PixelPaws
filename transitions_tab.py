"""
transitions_tab.py — PixelPaws Behavioral State Transition Analysis
====================================================================
Computes transition probability matrices between behavioral states,
visualizes ethograms, heatmaps, directed network graphs, and group
comparisons.  Includes latent behavioral state discovery via k-means
clustering of windowed transition matrices, state occupancy analysis,
and PCA-based continuous indices (inspired by LUPE, Nature 2026).

State sources:
  1. Unsupervised clusters from the Discover tab (primary)
  2. Supervised classifier predictions from results/ (optional)
"""

import os
import glob
import hashlib
import json
import pickle
import threading
import traceback

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ---------------------------------------------------------------------------
# Optional: matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional: networkx for directed graph
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from evaluation_tab import find_session_triplets
    _FIND_SESSION_TRIPLETS_AVAILABLE = True
except ImportError:
    find_session_triplets = None
    _FIND_SESSION_TRIPLETS_AVAILABLE = False

try:
    from feature_cache import FeatureCacheManager as _TransFeatureCacheManager
    _TRANS_FEATURE_CACHE_AVAILABLE = True
except ImportError:
    _TransFeatureCacheManager = None
    _TRANS_FEATURE_CACHE_AVAILABLE = False


from ui_utils import _bind_tight_layout_on_resize


# ═══════════════════════════════════════════════════════════════════════════
# Transition computation (pure functions, no GUI dependency)
# ═══════════════════════════════════════════════════════════════════════════

def compute_transition_matrix(state_seq, states=None, normalize=True,
                              zero_diagonal=False):
    """Compute transition matrix from a 1-D integer state sequence.

    Parameters
    ----------
    state_seq : array-like of int
    states : list of int, optional – ordered state IDs (rows/cols).
        If None, derived from unique values in *state_seq*.
    normalize : bool – row-normalize to probabilities.
    zero_diagonal : bool – zero self-transition diagonal.

    Returns
    -------
    matrix : np.ndarray (n_states, n_states)
    states : list of int – ordered state IDs matching rows/cols
    """
    seq = np.asarray(state_seq, dtype=int)
    if states is None:
        states = sorted(set(seq))
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in state_to_idx and b in state_to_idx:
            mat[state_to_idx[a], state_to_idx[b]] += 1
    if zero_diagonal:
        np.fill_diagonal(mat, 0)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    return mat, states


def compute_windowed_transitions(state_seq, fps, window_sec, step_sec,
                                 states=None, zero_diagonal=False, mode='frame'):
    """Slide a window across *state_seq* and compute a transition matrix
    for each window position.

    Returns list of (time_center_sec, matrix) tuples.
    """
    seq = np.asarray(state_seq, dtype=int)
    if states is None:
        states = sorted(set(seq))
    win_frames = max(1, int(round(window_sec * fps)))
    step_frames = max(1, int(round(step_sec * fps)))
    results = []
    start = 0
    while start + win_frames <= len(seq):
        chunk = seq[start:start + win_frames]
        if mode == 'bout':
            mat, _ = compute_bout_transition_matrix(
                chunk, states=states, normalize=True)
        else:
            mat, _ = compute_transition_matrix(chunk, states=states,
                                               normalize=True,
                                               zero_diagonal=zero_diagonal)
        center = (start + win_frames / 2) / fps
        results.append((center, mat))
        start += step_frames
    return results, states


def smooth_state_sequence(seq, min_frames):
    """Remove short bouts (< min_frames) by replacing them with the
    surrounding state."""
    if min_frames <= 1:
        return seq
    out = seq.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        bout_len = j - i
        if bout_len < min_frames:
            # Replace with previous state if possible, else next
            replacement = out[i - 1] if i > 0 else (out[j] if j < n else out[i])
            out[i:j] = replacement
        i = j
    return out


def extract_bouts(state_seq):
    """Convert per-frame state sequence to ordered list of bout dicts."""
    if len(state_seq) == 0:
        return []
    bouts, current, start = [], state_seq[0], 0
    for i in range(1, len(state_seq)):
        if state_seq[i] != current:
            bouts.append({'state': current, 'start': start, 'end': i - 1})
            current, start = state_seq[i], i
    bouts.append({'state': current, 'start': start, 'end': len(state_seq) - 1})
    return bouts


def compute_bout_transition_matrix(state_seq, states=None, normalize=True):
    """Transition matrix over bout-to-bout switches. Diagonal is always 0."""
    bouts = extract_bouts(state_seq)
    if states is None:
        states = sorted(set(state_seq))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n))
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a['state'] in idx and b['state'] in idx:
            counts[idx[a['state']], idx[b['state']]] += 1
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        counts = counts / row_sums
    return counts, states


def compute_temporal_probabilities(state_seq, fps, bin_sec, states):
    """Bin *state_seq* into time bins and compute fraction per state.

    Returns
    -------
    time_centers : np.ndarray of float (n_bins,)
    prob_matrix  : np.ndarray (n_bins, n_states)
    """
    seq = np.asarray(state_seq, dtype=int)
    bin_frames = max(1, int(round(bin_sec * fps)))
    n_bins = max(1, len(seq) // bin_frames)
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    prob = np.zeros((n_bins, n_states))
    centers = np.zeros(n_bins)
    for b in range(n_bins):
        start = b * bin_frames
        end = min(start + bin_frames, len(seq))
        chunk = seq[start:end]
        centers[b] = (start + end) / 2.0 / fps
        for frame_val in chunk:
            idx = state_to_idx.get(frame_val)
            if idx is not None:
                prob[b, idx] += 1
        total = prob[b].sum()
        if total > 0:
            prob[b] /= total
    return centers, prob


def cluster_transition_matrices(windowed_dict, k, states, n_init=100):
    """K-means on flattened windowed transition matrices (LUPE method).

    Parameters
    ----------
    windowed_dict : dict  {session: [(time_center, matrix), ...]}
    k : int               number of latent states
    states : list         ordered state IDs
    n_init : int          KMeans n_init

    Returns
    -------
    centroids : np.ndarray (k, n_states, n_states) — centroid matrices
    session_latent_map : dict {session: list of int} — latent state ID per window
    """
    from sklearn.cluster import KMeans

    n_states = len(states)
    flat_rows = []
    session_indices = []  # (session_name, window_idx)
    for session, wresults in windowed_dict.items():
        for wi, (t, mat) in enumerate(wresults):
            flat_rows.append(mat.ravel())
            session_indices.append((session, wi))

    X = np.vstack(flat_rows)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    labels = km.fit_predict(X)

    # Build per-session map
    session_latent_map = {s: [] for s in windowed_dict}
    for (session, wi), lbl in zip(session_indices, labels):
        session_latent_map[session].append(int(lbl))

    # Reshape centroids
    centroids = km.cluster_centers_.reshape(k, n_states, n_states)
    return centroids, session_latent_map


def compute_state_occupancy(session_latent_map, n_latent_states):
    """Fractional occupancy of each latent state per session.

    Returns
    -------
    dict {session: np.ndarray of shape (n_latent_states,)}
    """
    occupancy = {}
    for session, latent_ids in session_latent_map.items():
        arr = np.array(latent_ids, dtype=int)
        counts = np.bincount(arr, minlength=n_latent_states).astype(float)
        total = counts.sum()
        if total > 0:
            counts /= total
        occupancy[session] = counts
    return occupancy


def pca_on_occupancy(occupancy_dict):
    """PCA on stacked occupancy vectors.

    Returns
    -------
    pca_model : sklearn PCA
    scores_dict : dict {session: (pc1, pc2)}
    loadings : np.ndarray (n_components, n_latent_states)
    """
    from sklearn.decomposition import PCA

    sessions = sorted(occupancy_dict.keys())
    X = np.vstack([occupancy_dict[s] for s in sessions])
    n_comp = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    scores_dict = {}
    for i, s in enumerate(sessions):
        pc1 = float(scores[i, 0]) if n_comp >= 1 else 0.0
        pc2 = float(scores[i, 1]) if n_comp >= 2 else 0.0
        scores_dict[s] = (pc1, pc2)
    return pca, scores_dict, pca.components_


def reduce_clusters(processed_seqs, states, target_n):
    """Merge fine-grained clusters into target_n meta-clusters using
    agglomerative clustering on transition count profiles.

    Parameters
    ----------
    processed_seqs : dict {session_name: np.array of int}
    states : list of int — ordered state IDs
    target_n : int — desired number of meta-clusters

    Returns
    -------
    mapping : dict {old_id: new_id}
    new_states : list of int — sorted new state IDs (0..target_n-1)
    merge_info : dict {new_id: list of old_ids}
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    # Pool all sequences and compute raw transition count matrix
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=float)
    for seq in processed_seqs.values():
        for a, b in zip(seq[:-1], seq[1:]):
            if a in state_to_idx and b in state_to_idx:
                counts[state_to_idx[a], state_to_idx[b]] += 1

    # Each cluster's profile = its row in the count matrix
    profiles = counts.copy()

    # Agglomerative clustering on profiles
    if n <= target_n:
        # Nothing to reduce
        mapping = {s: i for i, s in enumerate(states)}
        new_states = list(range(len(states)))
        merge_info = {i: [s] for i, s in enumerate(states)}
        return mapping, new_states, merge_info

    Z = linkage(profiles, method='ward')
    labels = fcluster(Z, t=target_n, criterion='maxclust')
    # labels is 1-indexed; convert to 0-indexed
    labels = labels - 1

    # Build mapping: old cluster ID -> new meta-cluster ID
    mapping = {}
    merge_info = {}
    for idx, old_id in enumerate(states):
        new_id = int(labels[idx])
        mapping[old_id] = new_id
        if new_id not in merge_info:
            merge_info[new_id] = []
        merge_info[new_id].append(old_id)

    new_states = sorted(merge_info.keys())
    return mapping, new_states, merge_info


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# TransitionVideoPreview
# ═══════════════════════════════════════════════════════════════════════════

class TransitionVideoPreview:
    """Multi-state video preview with per-frame probability graph.
    Adapted from SideBySidePreview for n-class transitions output."""

    # seaborn/tab10-style colors for states (index 0 = Other = grey)
    _STATE_COLORS_BGR = [
        (128, 128, 128),  # 0 = Other
        (214,  39,  40),  # 1
        ( 31, 119, 180),  # 2
        ( 44, 160,  44),  # 3
        (148, 103, 189),  # 4
        (140,  86,  75),  # 5
        (227, 119, 194),  # 6
        (188, 189,  34),  # 7
        ( 23, 190, 207),  # 8
        (255, 127,  14),  # 9
    ]
    _STATE_COLORS_HEX = [
        '#808080', '#d62728', '#1f77b4', '#2ca02c',
        '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#ff7f0e',
    ]

    def __init__(self, parent, video_path, state_seq, prob_matrix,
                 state_names, session_name):
        self.parent = parent
        self.video_path = video_path
        self.state_seq = np.asarray(state_seq, dtype=int)
        self.prob_matrix = prob_matrix  # (n_frames × n_classifiers) or None
        self.state_names = state_names  # index 0 = "Other"
        self.session_name = session_name

        try:
            import cv2 as _cv2
            self._cv2 = _cv2
        except ImportError:
            messagebox.showerror("Missing dependency",
                "OpenCV (cv2) is required for video preview.\n"
                "Install with: pip install opencv-python", parent=parent)
            return

        self.cap = self._cv2.VideoCapture(video_path)
        self.fps = self.cap.get(self._cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self._last_read_frame = -1
        self._canvas_image_id = None
        self.graph_window_obj = None
        self.graph_window_var = tk.IntVar(value=500)
        self.graph_redraw_counter = 0
        self.graph_redraw_interval = 5

        self._build_ui()
        self.window.after(100, self.update_frame)

    # ── UI build ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Video Preview \u2014 {self.session_name}")
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        w, h = int(sw * 0.72), int(sh * 0.72)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Controls row
        ctrl = ttk.Frame(self.window)
        ctrl.pack(fill='x', padx=6, pady=4)
        self.play_btn = ttk.Button(ctrl, text="\u25b6 Play", command=self.toggle_play)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u23ee -100", command=lambda: self.jump(-100)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u25c4 -10",  command=lambda: self.jump(-10)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u25ba +10",  command=lambda: self.jump(10)).pack(side='left', padx=2)
        ttk.Button(ctrl, text="\u23ed +100", command=lambda: self.jump(100)).pack(side='left', padx=2)
        for lbl, spd in [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("5x", 5.0)]:
            ttk.Button(ctrl, text=lbl, width=5,
                       command=lambda s=spd: self.set_speed(s)).pack(side='left', padx=1)
        if self.prob_matrix is not None:
            ttk.Button(ctrl, text="\U0001f4c8 Prob Graph",
                       command=self.open_graph_window).pack(side='right', padx=6)

        # Frame label
        self.frame_label = ttk.Label(ctrl, text="0 / 0")
        self.frame_label.pack(side='right', padx=8)

        # Video canvas
        self.canvas_video = tk.Canvas(self.window, bg='black')
        self.canvas_video.pack(fill='both', expand=True, padx=4, pady=2)

        # Scrub slider
        slider_row = ttk.Frame(self.window)
        slider_row.pack(fill='x', padx=6, pady=(0, 4))
        self.slider = ttk.Scale(slider_row, from_=0, to=max(0, self.total_frames - 1),
                                orient='horizontal', command=self._on_slider)
        self.slider.pack(fill='x', expand=True)

        # State color timeline bar
        self.timeline_canvas = tk.Canvas(self.window, height=16, bg='#eee')
        self.timeline_canvas.pack(fill='x', padx=6, pady=(0, 4))
        self.timeline_canvas.bind('<Button-1>', self._on_timeline_click)
        self.window.after(200, self._draw_timeline)

    # ── Frame display ─────────────────────────────────────────────────
    def update_frame(self):
        cv2 = self._cv2
        if self.current_frame != self._last_read_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self._last_read_frame = self.current_frame

        if ret:
            # State lookup uses state_seq length (may be downsampled)
            n_state = len(self.state_seq)
            if n_state > 0 and self.total_frames > 0:
                fi_state = min(int(self.current_frame * n_state / self.total_frames), n_state - 1)
            else:
                fi_state = 0
            # Prob lookup uses prob_matrix length (always full-length)
            n_prob = len(self.prob_matrix) if self.prob_matrix is not None else 0
            if n_prob > 0 and self.total_frames > 0:
                fi_prob = min(int(self.current_frame * n_prob / self.total_frames), n_prob - 1)
            else:
                fi_prob = 0

            state_id = int(self.state_seq[fi_state])
            state_name = (self.state_names[state_id]
                          if state_id < len(self.state_names) else f"State {state_id}")
            color_bgr = self._get_color_bgr(state_id)

            # Overlay: state name
            cv2.putText(frame, state_name,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_bgr, 3)
            if self.prob_matrix is not None and fi_prob < n_prob:
                probs = self.prob_matrix[fi_prob]  # (n_classifiers,)
                argmax_ci = int(np.argmax(probs))   # 0-based index of highest-prob classifier
                for ci, p in enumerate(probs):
                    sname = (self.state_names[ci + 1]
                             if ci + 1 < len(self.state_names) else f"State {ci+1}")
                    is_assigned = (state_id == ci + 1)
                    is_best     = (ci == argmax_ci)
                    if is_assigned and is_best:
                        marker = '>'    # assigned AND highest prob (normal)
                    elif is_assigned:
                        marker = '<'    # assigned but NOT highest prob  (priority/gap-fill override)
                    elif is_best:
                        marker = '*'    # highest prob but NOT assigned
                    else:
                        marker = ' '
                    cv2.putText(frame, f"{marker} {sname}: {p:.2f}",
                                (20, 100 + ci * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                self._get_color_bgr(ci + 1), 2)
                # Legend — only shown when assigned ≠ argmax
                if state_id != 0 and state_id != argmax_ci + 1:
                    legend_y = 100 + len(probs) * 35 + 8
                    cv2.putText(frame, "< assigned   * highest prob",
                                (20, legend_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (200, 200, 200), 1)
            cv2.putText(frame, f"Frame {self.current_frame}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._show_frame(frame)

        self.slider.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame} / {self.total_frames}")
        self._update_timeline_marker()
        self._maybe_update_graph()

    def _show_frame(self, frame):
        from PIL import Image, ImageTk
        cv2 = self._cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cw = self.canvas_video.winfo_width() or 640
        ch = self.canvas_video.winfo_height() or 480
        h, w = frame_rgb.shape[:2]
        aspect = w / h
        if cw / ch > aspect:
            nh, nw = ch, int(ch * aspect)
        else:
            nw, nh = cw, int(cw / aspect)
        frame_resized = cv2.resize(frame_rgb, (nw, nh))
        photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        x, y = (cw - nw) // 2, (ch - nh) // 2
        if self._canvas_image_id is None:
            self.canvas_video.delete('all')
            self._canvas_image_id = self.canvas_video.create_image(
                x, y, anchor='nw', image=photo)
        else:
            self.canvas_video.coords(self._canvas_image_id, x, y)
            self.canvas_video.itemconfig(self._canvas_image_id, image=photo)
        self.canvas_video.image = photo  # prevent GC

    # ── Timeline bar ──────────────────────────────────────────────────
    def _draw_timeline(self):
        self.timeline_canvas.delete('all')
        w = self.timeline_canvas.winfo_width() or 800
        n = len(self.state_seq)
        if n == 0 or w < 2:
            return
        ds = max(1, n // w)
        for i in range(0, n, ds):
            sid = int(self.state_seq[i])
            color = (self._STATE_COLORS_HEX[sid % len(self._STATE_COLORS_HEX)])
            x = int(i / n * w)
            self.timeline_canvas.create_line(x, 0, x, 16, fill=color, width=max(1, ds))
        self._update_timeline_marker()

    def _update_timeline_marker(self):
        self.timeline_canvas.delete('marker')
        w = self.timeline_canvas.winfo_width() or 800
        x = int(self.current_frame / self.total_frames * w) if self.total_frames > 0 else 0
        self.timeline_canvas.create_line(x, 0, x, 16, fill='white', width=2, tags='marker')

    def _on_timeline_click(self, event):
        w = self.timeline_canvas.winfo_width() or 800
        frame = int(event.x / w * self.total_frames)
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update_frame()

    # ── Playback ──────────────────────────────────────────────────────
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="\u23f8 Pause" if self.playing else "\u25b6 Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self):
        if not self.playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="\u25b6 Play")
            return
        self.current_frame += 1
        self.update_frame()
        delay = max(1, int(1000 / (self.fps * self.playback_speed)))
        self.window.after(delay, self._play_loop)

    def set_speed(self, spd):
        self.playback_speed = spd

    def jump(self, delta):
        self.current_frame = max(0, min(self.current_frame + delta, self.total_frames - 1))
        self.update_frame()

    def _on_slider(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()

    # ── Probability graph window ──────────────────────────────────────
    def open_graph_window(self):
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            self.graph_window_obj.lift()
            self._update_graph()
            return
        self.graph_window_obj = tk.Toplevel(self.window)
        self.graph_window_obj.title(f"Probability Graph \u2014 {self.session_name}")
        sw = self.graph_window_obj.winfo_screenwidth()
        sh = self.graph_window_obj.winfo_screenheight()
        w, h = int(sw * 0.75), int(sh * 0.55)
        self.graph_window_obj.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        ctrl = ttk.Frame(self.graph_window_obj)
        ctrl.pack(fill='x', padx=5, pady=4)
        ttk.Label(ctrl, text="Window:").pack(side='left', padx=2)
        ttk.Spinbox(ctrl, from_=100, to=10000, increment=100,
                    textvariable=self.graph_window_var, width=7).pack(side='left')
        ttk.Label(ctrl, text="frames").pack(side='left', padx=2)
        ttk.Button(ctrl, text="Refresh", command=self._update_graph).pack(side='left', padx=8)

        self.graph_frame_lbl = ttk.Label(ctrl, text="", width=18)
        self.graph_frame_lbl.pack(side='right', padx=6)
        self.graph_scrollbar = ttk.Scale(
            ctrl, from_=0, to=self.total_frames - 1,
            orient='horizontal', command=self._on_graph_scroll)
        self.graph_scrollbar.pack(side='right', fill='x', expand=True, padx=5)

        graph_frame = ttk.Frame(self.graph_window_obj)
        graph_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.graph_fig = Figure(figsize=(12, 3.5), dpi=100, facecolor='white',
                                constrained_layout=True)
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_canvas_widget = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas_widget.get_tk_widget().pack(fill='both', expand=True)
        self.graph_canvas_widget.mpl_connect('button_press_event', self._on_graph_click)
        _bind_tight_layout_on_resize(self.graph_canvas_widget, self.graph_fig)
        self._update_graph()

    def _update_graph(self):
        if not self.graph_window_obj or not self.graph_window_obj.winfo_exists():
            return
        if self.playing:
            self.graph_redraw_counter += 1
            if self.graph_redraw_counter < self.graph_redraw_interval:
                return
            self.graph_redraw_counter = 0

        try:
            self.graph_ax.clear()
            # Compute feature-frame index the same way as update_frame (proportional mapping)
            n_feat = len(self.prob_matrix)
            if n_feat > 0 and self.total_frames > 0:
                fi = min(int(self.current_frame * n_feat / self.total_frames), n_feat - 1)
            else:
                fi = 0

            half = self.graph_window_var.get() // 2
            feat_start = max(0, fi - half)
            feat_end   = min(n_feat, fi + half)
            feat_indices = np.arange(feat_start, feat_end)
            # Convert feature-frame indices → video-frame numbers for x-axis
            video_frames = (feat_indices * self.total_frames / n_feat).astype(int) if self.total_frames > 0 else feat_indices

            current_state = int(self.state_seq[fi])

            n_clf = self.prob_matrix.shape[1]
            for ci in range(n_clf):
                sname = (self.state_names[ci + 1]
                         if ci + 1 < len(self.state_names) else f"State {ci+1}")
                color = (self._STATE_COLORS_HEX[(ci + 1) % len(self._STATE_COLORS_HEX)])
                lw = 2.5 if current_state == ci + 1 else 1.0
                alpha = 1.0 if current_state == ci + 1 else 0.5
                self.graph_ax.plot(video_frames, self.prob_matrix[feat_start:feat_end, ci],
                                   color=color, linewidth=lw, alpha=alpha,
                                   label=sname, zorder=3)

            self.graph_ax.axvline(x=self.current_frame, color='black',
                                  linewidth=2, linestyle='--',
                                  label='Current frame', zorder=4)
            self.graph_ax.set_xlabel('Frame (video)')
            self.graph_ax.set_ylabel('Probability')
            self.graph_ax.set_ylim(-0.05, 1.05)
            x_left  = int(video_frames[0])  if len(video_frames) else 0
            x_right = int(video_frames[-1]) if len(video_frames) else self.total_frames
            self.graph_ax.set_xlim(x_left, x_right)
            state_lbl = (self.state_names[current_state]
                         if current_state < len(self.state_names)
                         else f"State {current_state}")
            self.graph_ax.set_title(
                f"{self.session_name}  \u2014  Frame {self.current_frame}  \u2014  "
                f"Current state: {state_lbl}", fontsize=11)
            self.graph_ax.grid(True, alpha=0.3)
            self.graph_ax.legend(loc='upper right', fontsize=9, ncol=2)
            self.graph_canvas_widget.draw()

            if hasattr(self, 'graph_scrollbar'):
                self.graph_scrollbar.set(self.current_frame)
            if hasattr(self, 'graph_frame_lbl'):
                self.graph_frame_lbl.config(
                    text=f"{self.current_frame} / {self.total_frames}")
        except Exception as e:
            print(f"Graph update error: {e}")

    def _maybe_update_graph(self):
        if (self.graph_window_obj and self.graph_window_obj.winfo_exists()
                and self.prob_matrix is not None):
            self._update_graph()

    def _on_graph_click(self, event):
        if event.inaxes != self.graph_ax:
            return
        self.current_frame = max(0, min(int(event.xdata), self.total_frames - 1))
        self.update_frame()

    def _on_graph_scroll(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()

    # ── Helpers ───────────────────────────────────────────────────────
    def _get_color_bgr(self, state_id):
        return self._STATE_COLORS_BGR[state_id % len(self._STATE_COLORS_BGR)]

    def _on_close(self):
        self.playing = False
        self.cap.release()
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            self.graph_window_obj.destroy()
        self.window.destroy()


# TransitionsTab  (ttk.Frame)
# ═══════════════════════════════════════════════════════════════════════════

class TransitionsTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui

        # Internal state
        self._state_seqs = {}       # {session_name: np.array of int}
        self._states = []           # ordered state IDs
        self._state_labels = {}     # {state_id: user label}  e.g. {0: "Still"}
        self._matrices = {}         # {session_name: (matrix, states)}
        self._windowed = {}         # {session_name: [(t, mat), ...]}
        self._group_matrices = {}   # {group: mean_matrix}
        self._group_sem = {}        # {group: sem_matrix}
        self._group_subject_matrices = {}  # {group: [matrix, ...]} — individual subjects
        self._frame_probs = {}  # {session_name: np.ndarray (n_frames × n_classifiers)}
        self._key_df = None
        self._session_subjects = {} # {session_name: subject}
        self._merge_info = None     # {new_id: [old_ids]} from cluster reduction
        self._model_bundle = None   # full model.pkl contents for summary view
        self._worker_thread = None
        self._stop_event = threading.Event()

        # Latent state discovery (LUPE method)
        self._latent_centroids = None    # (k, n_states, n_states)
        self._session_latent_map = {}    # {session: [latent_state_id per window]}
        self._n_latent = 0
        self._occupancy = {}             # {session: array(k,)} fractional occupancy
        self._pca_model = None
        self._pca_scores = {}            # {session: (pc1, pc2)}
        self._pca_loadings = None
        self._temporal_probs = {}        # {session: (time_centers, prob_matrix)}
        self._group_occupancy = {}       # {treatment: mean_array}
        self._group_occupancy_sem = {}   # {treatment: sem_array}

        # Supervised prediction state
        self._loaded_classifiers = []   # list of clf_data dicts
        self._priority_order = []       # indices into _loaded_classifiers, in priority order
        self._trans_sessions = []       # list of session dicts from find_session_triplets
        self._trans_session_checked = {}  # {session_name: BooleanVar}
        self._pending_session_selection = None  # set by _load_config, consumed by _scan_trans_sessions
        self._effective_fps = 25.0
        self._assign_mode = tk.StringVar(value='priority')
        self._palette_var  = tk.StringVar(value='deep')
        self._bot_palette_var = tk.StringVar(value='tab10')
        self._show_annot_var = tk.BooleanVar(value=True)
        self._show_sig_var = tk.BooleanVar(value=False)
        self._transition_mode = tk.StringVar(value='bout')

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Scrollable container
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        sf = self._scroll_frame

        # Title
        ttk.Label(sf, text="Behavioral State Transitions",
                  font=('Arial', 14, 'bold')).pack(anchor='w', padx=20, pady=(15, 5))
        ttk.Label(sf, text="Compute transition probabilities between behavioral "
                  "states and compare across treatment groups.",
                  wraplength=700).pack(anchor='w', padx=20, pady=(0, 10))

        # ── State Source ──────────────────────────────────────────────
        src_frame = ttk.LabelFrame(sf, text="State Source", padding=10)
        src_frame.pack(fill='x', padx=20, pady=5)

        self._source_var = tk.StringVar(value='unsupervised')
        ttk.Radiobutton(src_frame, text="Unsupervised clusters (Discover tab)",
                        variable=self._source_var, value='unsupervised',
                        command=self._toggle_source).grid(row=0, column=0,
                                                          sticky='w', columnspan=3)
        ttk.Radiobutton(src_frame, text="Classifier predictions (Results)",
                        variable=self._source_var, value='supervised',
                        command=self._toggle_source).grid(row=1, column=0,
                                                          sticky='w', columnspan=3)

        # Unsupervised sub-panel
        self._unsup_frame = ttk.Frame(src_frame)
        self._unsup_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        ttk.Label(self._unsup_frame, text="Run:").pack(side='left', padx=(20, 5))
        self._run_combo = ttk.Combobox(self._unsup_frame, state='readonly', width=30)
        self._run_combo.pack(side='left', padx=5)
        ttk.Button(self._unsup_frame, text="Refresh",
                   command=self._scan_runs).pack(side='left', padx=5)

        # Supervised sub-panel (hidden initially)
        self._sup_frame = ttk.Frame(src_frame)

        # -- Classifiers sub-section --
        clf_section = ttk.LabelFrame(self._sup_frame, text="Classifiers", padding=5)
        clf_section.pack(fill='x', pady=(5, 3))

        clf_list_frame = ttk.Frame(clf_section)
        clf_list_frame.pack(fill='x')
        self._clf_listbox = tk.Listbox(clf_list_frame, height=4, selectmode='single',
                                       font=('Courier', 8), width=80)
        _clf_scroll = ttk.Scrollbar(clf_list_frame, command=self._clf_listbox.yview)
        _clf_hscroll = ttk.Scrollbar(clf_list_frame, orient='horizontal',
                                     command=self._clf_listbox.xview)
        self._clf_listbox.configure(yscrollcommand=_clf_scroll.set,
                                    xscrollcommand=_clf_hscroll.set)
        self._clf_listbox.pack(side='left', fill='both', expand=True)
        _clf_scroll.pack(side='right', fill='y')
        _clf_hscroll.pack(side='bottom', fill='x')
        self._clf_listbox.bind('<Double-ButtonRelease-1>',
                               lambda e: self._edit_clf_settings())

        clf_btn_row = ttk.Frame(clf_section)
        clf_btn_row.pack(fill='x', pady=(3, 0))
        ttk.Button(clf_btn_row, text="Add Classifier",
                   command=self._add_classifier).pack(side='left', padx=(0, 5))
        ttk.Button(clf_btn_row, text="Remove",
                   command=self._remove_classifier).pack(side='left')
        ttk.Button(clf_btn_row, text="Edit Settings",
                   command=self._edit_clf_settings).pack(side='left', padx=5)

        # -- Sessions sub-section --
        sess_section = ttk.LabelFrame(self._sup_frame, text="Sessions", padding=5)
        sess_section.pack(fill='x', pady=(3, 3))

        sess_btn_row = ttk.Frame(sess_section)
        sess_btn_row.pack(fill='x', pady=(0, 3))
        ttk.Button(sess_btn_row, text="Refresh Sessions",
                   command=self._scan_trans_sessions).pack(side='left', padx=(0, 5))
        ttk.Button(sess_btn_row, text="Select All",
                   command=self._trans_select_all).pack(side='left', padx=(0, 5))
        ttk.Button(sess_btn_row, text="Deselect All",
                   command=self._trans_deselect_all).pack(side='left')

        trans_tree_frame = ttk.Frame(sess_section)
        trans_tree_frame.pack(fill='x', pady=(0, 3))
        self._trans_tree = ttk.Treeview(
            trans_tree_frame,
            columns=("check", "session", "video"),
            show="headings",
            selectmode="none",
            height=6,
        )
        self._trans_tree.heading("check", text="✓")
        self._trans_tree.heading("session", text="Session Name")
        self._trans_tree.heading("video", text="Video")
        self._trans_tree.column("check", width=30, anchor="center", stretch=False)
        self._trans_tree.column("session", width=250, anchor="w")
        self._trans_tree.column("video", width=250, anchor="w")
        _trans_tree_sb = ttk.Scrollbar(trans_tree_frame, orient='vertical',
                                       command=self._trans_tree.yview)
        self._trans_tree.configure(yscrollcommand=_trans_tree_sb.set)
        self._trans_tree.pack(side='left', fill='x', expand=True)
        _trans_tree_sb.pack(side='right', fill='y')
        self._trans_tree.bind("<ButtonRelease-1>", self._on_trans_tree_click)

        # -- State assignment mode --
        assign_section = ttk.LabelFrame(self._sup_frame, text="State Assignment",
                                        padding=5)
        assign_section.pack(fill='x', pady=(3, 3))

        assign_row = ttk.Frame(assign_section)
        assign_row.pack(fill='x')
        ttk.Radiobutton(assign_row, text="Priority ranking",
                        variable=self._assign_mode, value='priority',
                        command=self._toggle_assign_mode).pack(side='left',
                                                               padx=(0, 15))
        ttk.Radiobutton(assign_row, text="Best wins (argmax)",
                        variable=self._assign_mode, value='argmax',
                        command=self._toggle_assign_mode).pack(side='left')

        self._priority_frame = ttk.Frame(assign_section)
        self._priority_frame.pack(fill='x', pady=(3, 0))
        ttk.Label(self._priority_frame,
                  text="Highest priority wins when multiple classifiers fire",
                  foreground='gray').pack(anchor='w', padx=(20, 0))
        prio_list_row = ttk.Frame(self._priority_frame)
        prio_list_row.pack(fill='x', padx=(20, 0), pady=(3, 0))
        self._priority_listbox = tk.Listbox(prio_list_row, height=4,
                                            selectmode='single',
                                            font=('Courier', 8))
        self._priority_listbox.pack(side='left', fill='x', expand=True)
        prio_btn_col = ttk.Frame(prio_list_row)
        prio_btn_col.pack(side='left', padx=(5, 0))
        ttk.Button(prio_btn_col, text="Up",
                   command=self._move_priority_up).pack(fill='x', pady=(0, 3))
        ttk.Button(prio_btn_col, text="Down",
                   command=self._move_priority_down).pack(fill='x')

        # -- Transition mode --
        tmode_section = ttk.LabelFrame(self._sup_frame, text="Transition Mode",
                                       padding=5)
        tmode_section.pack(fill='x', pady=(3, 5))
        tmode_row = ttk.Frame(tmode_section)
        tmode_row.pack(fill='x')
        ttk.Radiobutton(tmode_row, text="Per-bout",
                        variable=self._transition_mode, value='bout').pack(
                            side='left', padx=(0, 15))
        ttk.Radiobutton(tmode_row, text="Per-frame (LUPE style)",
                        variable=self._transition_mode, value='frame').pack(
                            side='left')

        # Legacy supervised vars (for fallback _load_supervised_states path)
        self._results_var = tk.StringVar()
        self._behavior_list_frame = ttk.Frame(src_frame)
        self._behavior_vars = {}  # {name: BooleanVar}

        # ── Cluster Labels (optional rename table) ────────────────────
        lbl_frame = ttk.LabelFrame(sf, text="Cluster Labels (optional)", padding=10)
        lbl_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(lbl_frame,
                  text="Rename clusters for readability. Leave blank to keep default names.",
                  wraplength=600).pack(anchor='w')
        self._label_table_frame = ttk.Frame(lbl_frame)
        self._label_table_frame.pack(fill='x', pady=5)
        self._label_entries = {}  # {state_id: Entry widget}

        # ── Settings ──────────────────────────────────────────────────
        set_frame = ttk.LabelFrame(sf, text="Settings", padding=10)
        set_frame.pack(fill='x', padx=20, pady=5)

        row = 0
        ttk.Label(set_frame, text="FPS:").grid(row=row, column=0, sticky='w')
        self._fps_var = tk.IntVar(value=60)
        ttk.Spinbox(set_frame, from_=1, to=240, textvariable=self._fps_var,
                     width=6).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Button(set_frame, text="Auto-detect",
                   command=self._auto_detect_fps).grid(row=row, column=2, sticky='w', padx=5)

        row += 1
        self._downsample_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(set_frame, text="Downsample to 20 Hz — mode of every N frames (LUPE)",
                        variable=self._downsample_var).grid(
            row=row, column=0, columnspan=3, sticky='w', pady=(2, 0))

        row += 1
        ttk.Label(set_frame, text="Min state duration (ms):").grid(row=row, column=0,
                                                                     sticky='w')
        self._smooth_ms_var = tk.IntVar(value=100)
        ttk.Spinbox(set_frame, from_=0, to=5000, increment=50,
                     textvariable=self._smooth_ms_var,
                     width=6).grid(row=row, column=1, sticky='w', padx=5)

        row += 1
        self._exclude_noise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(set_frame, text="Exclude noise cluster (-1)",
                        variable=self._exclude_noise_var).grid(
                            row=row, column=0, columnspan=3, sticky='w')

        row += 1
        self._exclude_other_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(set_frame, text="Exclude 'Other' state (state 0) from analysis",
                        variable=self._exclude_other_var).grid(
                            row=row, column=0, columnspan=3, sticky='w')

        row += 1
        self._zero_diag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(set_frame, text="Zero self-transitions on diagonal",
                        variable=self._zero_diag_var).grid(
                            row=row, column=0, columnspan=3, sticky='w')

        # Cluster reduction
        row += 1
        self._reduce_clusters_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(set_frame, text="Reduce clusters (merge similar)",
                        variable=self._reduce_clusters_var,
                        command=self._toggle_reduce).grid(
                            row=row, column=0, columnspan=2, sticky='w')

        row += 1
        self._reduce_sub_frame = ttk.Frame(set_frame)
        self._reduce_sub_frame.grid(row=row, column=0, columnspan=3,
                                     sticky='w', padx=(20, 0))
        ttk.Label(self._reduce_sub_frame, text="Target clusters:").pack(
            side='left')
        self._target_clusters_var = tk.IntVar(value=10)
        ttk.Spinbox(self._reduce_sub_frame, from_=2, to=50,
                     textvariable=self._target_clusters_var,
                     width=4).pack(side='left', padx=5)
        self._reduce_status_label = ttk.Label(self._reduce_sub_frame, text="")
        self._reduce_status_label.pack(side='left', padx=10)
        self._reduce_sub_frame.grid_remove()  # hidden until checkbox enabled

        row += 1
        ttk.Label(set_frame, text="Temporal prob bin (s):").grid(
            row=row, column=0, sticky='w')
        self._prob_bin_var = tk.DoubleVar(value=30)
        ttk.Spinbox(set_frame, from_=5, to=600, increment=5,
                     textvariable=self._prob_bin_var,
                     width=6).grid(row=row, column=1, sticky='w', padx=5)

        # ── Time Windows ──────────────────────────────────────────────
        tw_frame = ttk.LabelFrame(sf, text="Time Windows", padding=10)
        tw_frame.pack(fill='x', padx=20, pady=5)

        self._time_mode_var = tk.StringVar(value='sliding')
        ttk.Radiobutton(tw_frame, text="Full session",
                        variable=self._time_mode_var, value='full',
                        command=self._toggle_time).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(tw_frame, text="Time range",
                        variable=self._time_mode_var, value='range',
                        command=self._toggle_time).grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(tw_frame, text="Sliding windows",
                        variable=self._time_mode_var, value='sliding',
                        command=self._toggle_time).grid(row=2, column=0, sticky='w')

        # Range sub-frame
        self._range_frame = ttk.Frame(tw_frame)
        ttk.Label(self._range_frame, text="Start (s):").pack(side='left', padx=(20, 2))
        self._range_start_var = tk.DoubleVar(value=0)
        ttk.Spinbox(self._range_frame, from_=0, to=99999, increment=30,
                     textvariable=self._range_start_var,
                     width=8).pack(side='left', padx=2)
        ttk.Label(self._range_frame, text="End (s):").pack(side='left', padx=(10, 2))
        self._range_end_var = tk.DoubleVar(value=1800)
        ttk.Spinbox(self._range_frame, from_=0, to=99999, increment=30,
                     textvariable=self._range_end_var,
                     width=8).pack(side='left', padx=2)

        # Sliding sub-frame
        self._sliding_frame = ttk.Frame(tw_frame)
        ttk.Label(self._sliding_frame, text="Window (s):").pack(side='left', padx=(20, 2))
        self._win_sec_var = tk.DoubleVar(value=30)
        ttk.Spinbox(self._sliding_frame, from_=1, to=600, increment=10,
                     textvariable=self._win_sec_var,
                     width=8).pack(side='left', padx=2)
        ttk.Label(self._sliding_frame, text="Step (s):").pack(side='left', padx=(10, 2))
        self._step_sec_var = tk.DoubleVar(value=10)
        ttk.Spinbox(self._sliding_frame, from_=1, to=600, increment=10,
                     textvariable=self._step_sec_var,
                     width=8).pack(side='left', padx=2)

        # Row 3: optional duration cap
        self._dur_limit_var = tk.BooleanVar(value=False)
        self._dur_limit_min = tk.DoubleVar(value=30.0)
        dur_row = ttk.Frame(tw_frame)
        dur_row.grid(row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))
        ttk.Checkbutton(dur_row, text="Crop to first",
                        variable=self._dur_limit_var).pack(side='left')
        ttk.Spinbox(dur_row, from_=1, to=9999, increment=5,
                    textvariable=self._dur_limit_min,
                    width=6).pack(side='left', padx=2)
        ttk.Label(dur_row, text="min").pack(side='left')

        # Row 4: Behavior Over Time palette
        bot_pal_row = ttk.Frame(tw_frame)
        bot_pal_row.grid(row=4, column=0, columnspan=3, sticky='w', pady=(4, 0))
        ttk.Label(bot_pal_row, text="Behavior Over Time palette:").pack(side='left')
        self._bot_palette_combo = ttk.Combobox(
            bot_pal_row, textvariable=self._bot_palette_var, state='readonly', width=12,
            values=['tab10', 'tab20', 'deep', 'colorblind', 'muted', 'bright',
                    'Set1', 'Set2', 'Dark2', 'Paired'])
        self._bot_palette_combo.pack(side='left', padx=(5, 0))

        # ── Latent State Discovery ────────────────────────────────────
        lat_frame = ttk.LabelFrame(sf, text="Latent State Discovery (LUPE)", padding=10)
        lat_frame.pack(fill='x', padx=20, pady=5)

        self._discover_latent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lat_frame,
                        text="Discover latent behavioral states "
                             "(k-means on windowed transition matrices)",
                        variable=self._discover_latent_var).pack(anchor='w')

        lat_sub = ttk.Frame(lat_frame)
        lat_sub.pack(fill='x', pady=(5, 0))
        ttk.Label(lat_sub, text="Number of latent states (k):").pack(
            side='left', padx=(20, 5))
        self._n_latent_var = tk.IntVar(value=6)
        ttk.Spinbox(lat_sub, from_=2, to=20, textvariable=self._n_latent_var,
                     width=4).pack(side='left', padx=5)

        ttk.Label(lat_frame,
                  text="Requires sliding windows mode. Clusters windowed "
                       "transition matrices to find latent behavioral states "
                       "(LUPE method).",
                  wraplength=600, foreground='gray').pack(
                      anchor='w', padx=20, pady=(5, 0))

        # ── Key File ─────────────────────────────────────────────────
        kf_frame = ttk.LabelFrame(sf, text="Key File (for group comparison)", padding=10)
        kf_frame.pack(fill='x', padx=20, pady=5)

        kf_row = ttk.Frame(kf_frame)
        kf_row.pack(fill='x')
        self._key_file_var = tk.StringVar()
        ttk.Entry(kf_row, textvariable=self._key_file_var,
                  width=50).pack(side='left', padx=(0, 5))
        ttk.Button(kf_row, text="Browse",
                   command=self._browse_key_file).pack(side='left', padx=5)
        ttk.Button(kf_row, text="Load",
                   command=self._load_key_file).pack(side='left', padx=5)
        self._key_status = ttk.Label(kf_frame, text="No key file loaded")
        self._key_status.pack(anchor='w', pady=(5, 0))

        # ── Run ──────────────────────────────────────────────────────
        run_frame = ttk.LabelFrame(sf, text="Run", padding=10)
        run_frame.pack(fill='x', padx=20, pady=5)

        btn_row = ttk.Frame(run_frame)
        btn_row.pack(fill='x')
        ttk.Button(btn_row, text="Compute Transitions",
                   command=self._start_compute).pack(side='left', padx=5)
        self._stop_btn = ttk.Button(btn_row, text="\u25a0  Stop",
                                    command=self._stop_compute, state='disabled')
        self._stop_btn.pack(side='left', padx=2)
        ttk.Separator(btn_row, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(btn_row, text="Save Config",
                   command=self._save_config).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Load Config",
                   command=self._load_config).pack(side='left', padx=2)
        self._progress = ttk.Progressbar(btn_row, mode='indeterminate', length=200)
        self._progress.pack(side='left', padx=10)

        self._log = scrolledtext.ScrolledText(run_frame, height=6, state='disabled',
                                              wrap='word')
        self._log.pack(fill='x', pady=(5, 0))

        # ── Results ──────────────────────────────────────────────────
        res_frame = ttk.LabelFrame(sf, text="Results", padding=10)
        res_frame.pack(fill='both', padx=20, pady=5, expand=True)

        # View selector
        view_row = ttk.Frame(res_frame)
        view_row.pack(fill='x', pady=(0, 5))
        ttk.Label(view_row, text="View:").pack(side='left', padx=(0, 5))
        self._view_var = tk.StringVar(value='Ethogram')
        self._view_combo = ttk.Combobox(
            view_row, textvariable=self._view_var, state='readonly', width=20,
            values=['Ethogram', 'Temporal Probability', 'Heatmap', 'Network',
                   'Group Comparison', 'Timeline', 'Behavior Over Time',
                   'Latent States', 'State Occupancy', 'PCA', 'Meta-cluster Summary'])
        self._view_combo.pack(side='left', padx=5)
        self._view_combo.bind('<<ComboboxSelected>>', self._on_view_changed)

        ttk.Label(view_row, text="Palette:").pack(side='left', padx=(10, 2))
        self._palette_combo = ttk.Combobox(
            view_row, textvariable=self._palette_var, state='readonly', width=12,
            values=['deep', 'colorblind', 'muted', 'bright', 'dark',
                    'tab10', 'tab20', 'Set1', 'Set2', 'Dark2', 'Paired',
                    'magma', 'plasma', 'viridis', 'inferno', 'cividis', 'turbo'])
        self._palette_combo.pack(side='left', padx=2)
        self._palette_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        ttk.Checkbutton(view_row, text="Cell values",
                        variable=self._show_annot_var,
                        command=self._refresh_plot).pack(side='left', padx=(8, 2))
        self._sig_cb = ttk.Checkbutton(view_row, text="Sig. markers",
                                       variable=self._show_sig_var,
                                       command=self._refresh_plot)
        self._sig_cb.pack(side='left', padx=(8, 2))

        # Video preview controls
        self._preview_session_var = tk.StringVar()
        self._preview_session_combo = ttk.Combobox(
            view_row, textvariable=self._preview_session_var,
            state='readonly', width=20)
        self._preview_session_combo.pack(side='left', padx=(16, 2))
        ttk.Button(view_row, text="\u25b6 Preview",
                   command=self._open_video_preview).pack(side='left', padx=(2, 4))

        # State pair selector (for timeline view)
        self._pair_frame = ttk.Frame(view_row)
        ttk.Label(self._pair_frame, text="Transition:").pack(side='left', padx=(10, 2))
        self._pair_combo = ttk.Combobox(self._pair_frame, state='readonly', width=25)
        self._pair_combo.pack(side='left', padx=2)
        self._pair_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_plot())

        # Matplotlib canvas
        self._fig = plt.figure(figsize=(9, 5), constrained_layout=True) if MATPLOTLIB_AVAILABLE else None
        self._canvas = None
        if MATPLOTLIB_AVAILABLE:
            self._canvas = FigureCanvasTkAgg(self._fig, master=res_frame)
            self._canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(self._canvas, res_frame)
            toolbar.update()
            _bind_tight_layout_on_resize(self._canvas, self._fig)

        # ── Export ───────────────────────────────────────────────────
        exp_frame = ttk.LabelFrame(sf, text="Export", padding=10)
        exp_frame.pack(fill='x', padx=20, pady=(5, 20))

        ttk.Button(exp_frame, text="Export Matrices (CSV)",
                   command=self._export_matrices).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export State Sequences (CSV)",
                   command=self._export_sequences).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export Figure (PNG)",
                   command=lambda: self._export_figure('png')).pack(side='left', padx=5)
        ttk.Button(exp_frame, text="Export Figure (PDF)",
                   command=lambda: self._export_figure('pdf')).pack(side='left', padx=5)

        self._toggle_time()

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------

    def _toggle_source(self):
        if self._source_var.get() == 'unsupervised':
            self._unsup_frame.grid(row=2, column=0, columnspan=3, sticky='ew',
                                   pady=(5, 0))
            self._sup_frame.grid_forget()
        else:
            self._unsup_frame.grid_forget()
            self._sup_frame.grid(row=2, column=0, columnspan=3, sticky='ew',
                                 pady=(5, 0))

    def _toggle_reduce(self):
        if self._reduce_clusters_var.get():
            self._reduce_sub_frame.grid()
        else:
            self._reduce_sub_frame.grid_remove()

    def _toggle_assign_mode(self):
        if self._assign_mode.get() == 'priority':
            self._priority_frame.pack(fill='x', pady=(3, 0))
        else:
            self._priority_frame.pack_forget()

    # ------------------------------------------------------------------
    # Supervised classifier helpers
    # ------------------------------------------------------------------

    def _add_classifier(self):
        folder = self.app.current_project_folder.get()
        clf_dir = os.path.join(folder, 'classifiers') if folder else None
        paths = filedialog.askopenfilenames(
            title="Select Classifier .pkl file(s)",
            initialdir=clf_dir if clf_dir and os.path.isdir(clf_dir) else folder,
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        for path in paths:
            try:
                with open(path, 'rb') as fh:
                    clf_data = pickle.load(fh)
                if 'clf_model' not in clf_data:
                    self._log_msg(f"Skipping {os.path.basename(path)}: "
                                  f"no clf_model key")
                    continue
                clf_data['_path'] = path
                self._loaded_classifiers.append(clf_data)
                self._priority_order.append(len(self._loaded_classifiers) - 1)
                self._log_msg(f"Loaded: {clf_data.get('Behavior_type', '?')} "
                              f"from {os.path.basename(path)}")
            except Exception as e:
                self._log_msg(f"Error loading {os.path.basename(path)}: {e}")
        self._update_clf_listbox()
        self._update_priority_listbox()

    def _remove_classifier(self):
        sel = self._clf_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        removed_clf_idx = idx
        del self._loaded_classifiers[removed_clf_idx]
        # Rebuild priority order: remove reference to removed index,
        # and decrement any index > removed
        self._priority_order = [
            (i - 1 if i > removed_clf_idx else i)
            for i in self._priority_order
            if i != removed_clf_idx
        ]
        self._update_clf_listbox()
        self._update_priority_listbox()

    def _edit_clf_settings(self):
        sel = self._clf_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select classifier",
                                "Click a classifier first, then Edit Settings.")
            return
        idx = sel[0]
        cd = self._loaded_classifiers[idx]
        self._open_clf_edit_dialog(idx, cd)

    def _open_clf_edit_dialog(self, idx, cd):
        dlg = tk.Toplevel(self)
        dlg.title("Edit Classifier Settings")
        dlg.resizable(False, False)
        dlg.grab_set()

        bname = cd.get('Behavior_type', f'Classifier {idx+1}')
        ttk.Label(dlg, text=f"Editing: {bname}",
                  font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(12, 8), sticky='w')

        fields = [
            ("Threshold (0–1):",         'best_thresh',    0.5,
             dict(from_=0.0, to=1.0, increment=0.01, format='%.3f')),
            ("Min bout (frames):",       'min_bout',       0,
             dict(from_=0, to=9999, increment=1)),
            ("Min after bout (frames):", 'min_after_bout', 0,
             dict(from_=0, to=9999, increment=1)),
            ("Max gap (frames):",        'max_gap',        0,
             dict(from_=0, to=9999, increment=1)),
        ]
        vars_ = {}
        for row, (label, key, default, spin_kw) in enumerate(fields, start=1):
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky='w',
                                            padx=(15, 5), pady=3)
            val = cd.get(key, default)
            if key == 'best_thresh':
                v = tk.DoubleVar(value=round(float(val), 3))
            else:
                v = tk.IntVar(value=int(val))
            vars_[key] = v
            ttk.Spinbox(dlg, textvariable=v, width=9, **spin_kw).grid(
                row=row, column=1, sticky='w', padx=(0, 15), pady=3)

        def _reset():
            path = cd.get('_path', '')
            if not path or not os.path.isfile(path):
                messagebox.showwarning("No file",
                    "Original .pkl path not found — cannot reset.", parent=dlg)
                return
            try:
                with open(path, 'rb') as fh:
                    orig = pickle.load(fh)
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=dlg)
                return
            for _, key, default, _ in fields:
                val = orig.get(key, default)
                if key == 'best_thresh':
                    vars_[key].set(round(float(val), 3))
                else:
                    vars_[key].set(int(val))

        def _ok():
            for _, key, _, _ in fields:
                try:
                    self._loaded_classifiers[idx][key] = vars_[key].get()
                except Exception:
                    pass
            self._update_clf_listbox()
            self._update_priority_listbox()
            dlg.destroy()

        btn_row = ttk.Frame(dlg)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2,
                     pady=(8, 12), padx=15, sticky='e')
        ttk.Button(btn_row, text="Reset to Defaults",
                   command=_reset).pack(side='left', padx=(0, 20))
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side='left', padx=5)
        ttk.Button(btn_row, text="OK",
                   command=_ok).pack(side='left', padx=5)

        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _update_clf_listbox(self):
        self._clf_listbox.delete(0, 'end')
        for cd in self._loaded_classifiers:
            bname  = cd.get('Behavior_type', '?')
            thresh = cd.get('best_thresh', 0.5)
            mb     = cd.get('min_bout', 0)
            mab    = cd.get('min_after_bout', 0)
            mg     = cd.get('max_gap', 0)
            self._clf_listbox.insert('end',
                f"{bname:<20} thresh={thresh:.3f}  "
                f"min_bout={mb}  min_after={mab}  max_gap={mg}")

    def _scan_trans_sessions(self, silent=False):
        """Scan project folder for sessions using find_session_triplets."""
        proj = self.app.current_project_folder.get()
        if not proj:
            if not silent:
                messagebox.showwarning("No project",
                                       "Select a project folder first.")
            return
        if not _FIND_SESSION_TRIPLETS_AVAILABLE:
            if not silent:
                messagebox.showerror("Error",
                                     "Cannot import find_session_triplets.")
            return

        sessions = find_session_triplets(proj, require_labels=False,
                                         recursive=True)
        self._trans_sessions = sessions

        # Rebuild treeview
        self._trans_tree.delete(*self._trans_tree.get_children())
        self._trans_session_checked.clear()

        for s in sessions:
            name = s['session_name']
            video_name = os.path.basename(s.get('video_path', '') or '')
            var = tk.BooleanVar(value=True)
            self._trans_session_checked[name] = var
            self._trans_tree.insert('', 'end', iid=name,
                                    values=("✓", name, video_name))

        self._apply_pending_session_selection()

        if not silent:
            self._log_msg(f"Found {len(sessions)} session(s) in project.")
        elif sessions:
            self._log_msg(f"Found {len(sessions)} session(s) in project.")

    def _on_trans_tree_click(self, event):
        """Toggle checkbox when user clicks a row in the session treeview."""
        tree = self._trans_tree
        region = tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            return
        row_id = tree.identify_row(event.y)
        if not row_id or row_id not in self._trans_session_checked:
            return
        bvar = self._trans_session_checked[row_id]
        bvar.set(not bvar.get())
        vals = list(tree.item(row_id, "values"))
        vals[0] = "✓" if bvar.get() else ""
        tree.item(row_id, values=vals)

    def _trans_select_all(self):
        for name, bvar in self._trans_session_checked.items():
            bvar.set(True)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = "✓"
            self._trans_tree.item(name, values=vals)

    def _trans_deselect_all(self):
        for name, bvar in self._trans_session_checked.items():
            bvar.set(False)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = ""
            self._trans_tree.item(name, values=vals)

    def _apply_pending_session_selection(self):
        """Apply _pending_session_selection to the treeview if sessions are loaded."""
        if self._pending_session_selection is None:
            return
        if not self._trans_session_checked:
            return  # sessions not yet scanned — will be applied after _scan_trans_sessions
        for name, bvar in self._trans_session_checked.items():
            checked = name in self._pending_session_selection
            bvar.set(checked)
            vals = list(self._trans_tree.item(name, "values"))
            vals[0] = "✓" if checked else ""
            self._trans_tree.item(name, values=vals)
        self._pending_session_selection = None  # consumed

    def _update_priority_listbox(self):
        self._priority_listbox.delete(0, 'end')
        for idx in self._priority_order:
            cd = self._loaded_classifiers[idx]
            bname = cd.get('Behavior_type', '?')
            thresh = cd.get('best_thresh', 0.5)
            self._priority_listbox.insert('end',
                                          f"{bname}  (thresh={thresh:.3f})")

    def _move_priority_up(self):
        sel = self._priority_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self._priority_order[i], self._priority_order[i - 1] = (
            self._priority_order[i - 1], self._priority_order[i])
        self._update_priority_listbox()
        self._priority_listbox.selection_set(i - 1)

    def _move_priority_down(self):
        sel = self._priority_listbox.curselection()
        if not sel or sel[0] >= len(self._priority_order) - 1:
            return
        i = sel[0]
        self._priority_order[i], self._priority_order[i + 1] = (
            self._priority_order[i + 1], self._priority_order[i])
        self._update_priority_listbox()
        self._priority_listbox.selection_set(i + 1)

    def _toggle_time(self):
        self._range_frame.grid_forget()
        self._sliding_frame.grid_forget()
        self._pair_frame.pack_forget()
        mode = self._time_mode_var.get()
        if mode == 'range':
            self._range_frame.grid(row=1, column=1, columnspan=2, sticky='w', padx=10)
        elif mode == 'sliding':
            self._sliding_frame.grid(row=2, column=1, columnspan=2, sticky='w', padx=10)

    # ------------------------------------------------------------------
    # Scan unsupervised runs
    # ------------------------------------------------------------------

    def _scan_runs(self):
        folder = self.app.current_project_folder.get()
        if not folder:
            messagebox.showwarning("No project", "Select a project folder first.")
            return
        unsup_dir = os.path.join(folder, 'unsupervised')
        if not os.path.isdir(unsup_dir):
            self._run_combo['values'] = []
            self._run_combo.set('')
            self._log_msg("No unsupervised/ directory found in project.")
            return
        runs = sorted(d for d in os.listdir(unsup_dir)
                      if os.path.isdir(os.path.join(unsup_dir, d)))
        self._run_combo['values'] = runs
        if runs:
            self._run_combo.set(runs[0])
        self._log_msg(f"Found {len(runs)} Discover run(s): {', '.join(runs)}")

    # ------------------------------------------------------------------
    # Supervised source helpers
    # ------------------------------------------------------------------

    def _browse_results(self):
        folder = self.app.current_project_folder.get()
        init_dir = os.path.join(folder, 'results') if folder else None
        chosen = filedialog.askdirectory(title="Select Results Folder",
                                         initialdir=init_dir)
        if chosen:
            self._results_var.set(chosen)
            self._scan_supervised()

    def _scan_supervised(self):
        folder = self._results_var.get()
        if not folder or not os.path.isdir(folder):
            return
        behaviors = set()
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if f.endswith('.csv') and 'prediction' in f.lower():
                    bname = self._extract_behavior_name(f)
                    if bname:
                        behaviors.add(bname)
        # Clear old checkboxes
        for w in self._behavior_list_frame.winfo_children():
            w.destroy()
        self._behavior_vars.clear()
        if behaviors:
            self._behavior_list_frame.grid(row=3, column=0, columnspan=3,
                                            sticky='ew', pady=(5, 0))
            ttk.Label(self._behavior_list_frame,
                      text="Select behaviors:").pack(anchor='w', padx=20)
            for b in sorted(behaviors):
                var = tk.BooleanVar(value=True)
                self._behavior_vars[b] = var
                ttk.Checkbutton(self._behavior_list_frame, text=b,
                                variable=var).pack(anchor='w', padx=30)
            self._log_msg(f"Found {len(behaviors)} behavior(s): "
                          f"{', '.join(sorted(behaviors))}")
        else:
            self._log_msg("No prediction CSVs found.")

    @staticmethod
    def _extract_behavior_name(filename):
        """Extract behavior name from prediction filename."""
        name = filename.replace('.csv', '')
        for suffix in ['_predictions', '_prediction', '_pred']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        parts = name.split('_')
        if 'PixelPaws' in parts:
            idx = parts.index('PixelPaws')
            behavior_parts = parts[idx + 1:]
            if behavior_parts:
                return '_'.join(behavior_parts)
        return None

    # ------------------------------------------------------------------
    # Key file
    # ------------------------------------------------------------------

    def _browse_key_file(self):
        folder = self.app.current_project_folder.get() or ''
        path = filedialog.askopenfilename(
            title="Select Key File",
            initialdir=folder,
            filetypes=[("CSV/Excel", "*.csv *.xlsx"), ("All", "*.*")])
        if path:
            self._key_file_var.set(path)
            self._load_key_file()

    def _load_key_file(self):
        path = self._key_file_var.get()
        if not path or not os.path.isfile(path):
            return
        try:
            if path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            if 'Subject' not in df.columns or 'Treatment' not in df.columns:
                messagebox.showerror("Invalid",
                                     "Key file must have Subject and Treatment columns.")
                return
            df['Subject'] = df['Subject'].astype(str)
            self._key_df = df
            treatments = df['Treatment'].unique()
            self._key_status.config(
                text=f"Loaded: {len(df)} subjects, "
                     f"{len(treatments)} group(s): {', '.join(map(str, treatments))}",
                foreground='green')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load key file:\n{e}")

    # ------------------------------------------------------------------
    # FPS auto-detect
    # ------------------------------------------------------------------

    def _auto_detect_fps(self):
        folder = self.app.current_project_folder.get()
        vid_dir = os.path.join(folder, 'videos') if folder else ''
        path = filedialog.askopenfilename(
            title="Select a video to detect FPS",
            initialdir=vid_dir if os.path.isdir(vid_dir) else folder,
            filetypes=[("Video", "*.mp4 *.avi *.mov"), ("All", "*.*")])
        if not path:
            return
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self._fps_var.set(int(round(fps)))
                self._log_msg(f"Detected FPS: {fps:.2f} -> set to {int(round(fps))}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not detect FPS:\n{e}")

    # ------------------------------------------------------------------
    # Cluster label editing
    # ------------------------------------------------------------------

    def _populate_label_table(self, states):
        """Build or rebuild the editable rename table for discovered states."""
        for w in self._label_table_frame.winfo_children():
            w.destroy()
        self._label_entries.clear()

        ttk.Label(self._label_table_frame, text="ID", width=6,
                  font=('Arial', 9, 'bold')).grid(row=0, column=0)
        ttk.Label(self._label_table_frame, text="Label", width=20,
                  font=('Arial', 9, 'bold')).grid(row=0, column=1)

        for i, sid in enumerate(states):
            ttk.Label(self._label_table_frame,
                      text=str(sid), width=6).grid(row=i + 1, column=0)
            entry = ttk.Entry(self._label_table_frame, width=20)
            entry.grid(row=i + 1, column=1, padx=5, pady=1)
            # Pre-fill with existing label or merge info
            if sid in self._state_labels:
                entry.insert(0, self._state_labels[sid])
            elif (self._merge_info is not None and sid in self._merge_info
                  and len(self._merge_info[sid]) > 1):
                old_ids = self._merge_info[sid]
                n = len(old_ids)
                if n <= 4:
                    detail = ','.join(map(str, old_ids))
                else:
                    detail = f"{n} clusters"
                entry.insert(0, f"Meta {sid} ({detail})")
            self._label_entries[sid] = entry

    def _read_label_entries(self):
        """Harvest the label table into self._state_labels."""
        self._state_labels = {}
        for sid, entry in self._label_entries.items():
            txt = entry.get().strip()
            if txt:
                self._state_labels[sid] = txt

    def _state_name(self, sid):
        """Return user label or default cluster name."""
        return self._state_labels.get(sid, f"Cluster {sid}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_msg(self, msg):
        def _append():
            self._log.config(state='normal')
            self._log.insert('end', msg + '\n')
            self._log.see('end')
            self._log.config(state='disabled')
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.app.root.after(0, _append)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_unsupervised_states(self, run_name):
        """Load per-session cluster IDs from a Discover run."""
        folder = self.app.current_project_folder.get()
        run_dir = os.path.join(folder, 'unsupervised', run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Try loading from per-session CSV files
        csv_files = glob.glob(os.path.join(run_dir, '*_cluster_ids.csv'))
        seqs = {}
        if csv_files:
            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                session = fname.replace('_cluster_ids.csv', '')
                df = pd.read_csv(csv_path)
                seqs[session] = df['cluster_id'].values
            self._log_msg(f"Loaded {len(seqs)} session(s) from CSVs")
            # Best-effort: also load model.pkl for embedding data
            model_path = os.path.join(run_dir, 'model.pkl')
            if os.path.isfile(model_path):
                try:
                    with open(model_path, 'rb') as fh:
                        self._model_bundle = pickle.load(fh)
                    self._log_msg("Loaded model bundle (embedding available)")
                except Exception:
                    self._model_bundle = None
            else:
                self._model_bundle = None
        else:
            # Fall back to model bundle
            model_path = os.path.join(run_dir, 'model.pkl')
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"No cluster_ids CSVs or model.pkl in {run_dir}")
            with open(model_path, 'rb') as fh:
                bundle = pickle.load(fh)
            self._model_bundle = bundle
            labels = bundle.get('cluster_labels')
            row_map = bundle.get('sessions')
            if labels is None or row_map is None:
                raise ValueError("Model bundle missing cluster_labels or sessions")
            for session, (start, end) in row_map.items():
                seqs[session] = labels[start:end].copy()
            self._log_msg(f"Loaded {len(seqs)} session(s) from model bundle")

        return seqs

    def _load_supervised_states(self):
        """Combine binary prediction CSVs into multi-class state sequences."""
        folder = self._results_var.get()
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError("Results folder not set or not found")

        selected = [b for b, v in self._behavior_vars.items() if v.get()]
        if not selected:
            raise ValueError("No behaviors selected")

        # Collect prediction files per behavior
        behavior_files = {b: [] for b in selected}
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if not f.endswith('.csv') or 'prediction' not in f.lower():
                    continue
                bname = self._extract_behavior_name(f)
                if bname in behavior_files:
                    behavior_files[bname].append(os.path.join(dirpath, f))

        # Group by session (subject)
        session_preds = {}  # {session: {behavior: array}}
        for bname, files in behavior_files.items():
            for fpath in files:
                fname = os.path.basename(fpath).replace('.csv', '')
                # Remove behavior suffix to get session name
                for suffix in ['_predictions', '_prediction', '_pred']:
                    if fname.endswith(suffix):
                        fname = fname[:-len(suffix)]
                        break
                # Remove behavior name from end to get base session
                if fname.endswith('_' + bname):
                    session = fname[:-len(bname) - 1]
                elif 'PixelPaws' in fname:
                    parts = fname.split('_')
                    idx = parts.index('PixelPaws')
                    session = '_'.join(parts[:idx])
                else:
                    session = fname

                df = pd.read_csv(fpath)
                # Look for probability or binary prediction column
                pred_col = None
                for c in df.columns:
                    cl = c.lower()
                    if 'probability' in cl or 'prob' in cl:
                        pred_col = c
                        break
                if pred_col is None:
                    for c in df.columns:
                        cl = c.lower()
                        if 'prediction' in cl or 'pred' in cl:
                            pred_col = c
                            break
                if pred_col is None:
                    continue

                if session not in session_preds:
                    session_preds[session] = {}
                session_preds[session][bname] = df[pred_col].values

        # Combine into multi-class: highest probability wins, "Idle" otherwise
        seqs = {}
        all_behaviors = sorted(selected)
        state_map = {b: i + 1 for i, b in enumerate(all_behaviors)}
        state_map_inv = {v: k for k, v in state_map.items()}
        idle_id = 0

        for session, bdict in session_preds.items():
            n_frames = max(len(v) for v in bdict.values())
            prob_matrix = np.zeros((n_frames, len(all_behaviors)))
            for bi, bname in enumerate(all_behaviors):
                if bname in bdict:
                    arr = bdict[bname]
                    prob_matrix[:len(arr), bi] = arr

            states = np.full(n_frames, idle_id, dtype=int)
            max_probs = prob_matrix.max(axis=1)
            active = max_probs > 0.5
            states[active] = prob_matrix[active].argmax(axis=1) + 1
            seqs[session] = states

        # Set up state labels
        self._state_labels = {idle_id: "Idle"}
        for bname, sid in state_map.items():
            self._state_labels[sid] = bname

        self._log_msg(f"Combined {len(all_behaviors)} behaviors into "
                      f"{len(seqs)} session(s)")
        return seqs

    def _predict_and_assign_states(self):
        """Run classifiers on checked sessions; assign per-frame state."""
        if not self._loaded_classifiers:
            raise ValueError("No classifiers loaded. Use 'Add Classifier'.")

        if not self._trans_sessions:
            raise ValueError("No sessions found. Click 'Refresh Sessions' first.")
        checked_sessions = [s for s in self._trans_sessions
                            if self._trans_session_checked.get(
                                s['session_name'], tk.BooleanVar()).get()]
        if not checked_sessions:
            raise ValueError("No sessions selected.")

        # Import utilities from parent modules
        try:
            from PixelPaws_GUI import PixelPaws_ExtractFeatures, \
                augment_features_post_cache, predict_with_xgboost, \
                _load_features_for_prediction
        except ImportError:
            raise ImportError(
                "Cannot import prediction utilities from PixelPaws_GUI.")
        try:
            from evaluation_tab import _apply_bout_filtering
        except ImportError:
            raise ImportError(
                "Cannot import _apply_bout_filtering from evaluation_tab.")

        assign_mode = self._assign_mode.get()
        seqs = {}

        # State labels: 0 = Other, 1..N = behaviors in load order
        self._state_labels = {0: 'Other'}
        for bi, cd in enumerate(self._loaded_classifiers):
            self._state_labels[bi + 1] = cd.get('Behavior_type',
                                                  f'Behavior_{bi + 1}')

        # Build feature config from first classifier (all share same features)
        cd0 = self._loaded_classifiers[0]
        cfg = {
            'bp_include_list':      cd0.get('bp_include_list', None),
            'bp_pixbrt_list':       cd0.get('bp_pixbrt_list', []),
            'square_size':          cd0.get('square_size', [20]),
            'pix_threshold':        cd0.get('pix_threshold', 0.3),
            'include_optical_flow': cd0.get('include_optical_flow', False),
            'bp_optflow_list':      cd0.get('bp_optflow_list', []),
        }
        if _TRANS_FEATURE_CACHE_AVAILABLE:
            cfg_hash = _TransFeatureCacheManager.compute_hash(cfg)
        else:
            key_dict = {
                'bp_include_list':  cfg['bp_include_list'],
                'bp_pixbrt_list':   list(cfg['bp_pixbrt_list']),
                'square_size':      [int(x) for x in cfg['square_size']]
                                    if hasattr(cfg['square_size'], '__iter__')
                                    else [int(cfg['square_size'])],
                'pix_threshold':    round(float(cfg['pix_threshold']), 6),
                'include_optical_flow': bool(cfg['include_optical_flow']),
                'bp_optflow_list':  list(cfg['bp_optflow_list']),
            }
            cfg_hash = hashlib.md5(
                repr(key_dict).encode('utf-8')).hexdigest()[:8]

        proj = self.app.current_project_folder.get()
        cache_root = os.path.join(proj, 'features') if proj else None

        for session in checked_sessions:
            if self._stop_event.is_set():
                self._log_msg("  Stopped.")
                return {}
            session_name = session['session_name']
            dlc_path = session['pose_path']
            video_path = session.get('video_path', '') or ''

            if not dlc_path or not os.path.isfile(dlc_path):
                self._log_msg(f"  {session_name}: DLC file not found, skipping")
                continue

            self._log_msg(f"  Processing: {session_name}")

            try:
                # Find any existing cache for this session (any version/hash)
                existing_cache = None
                if _TRANS_FEATURE_CACHE_AVAILABLE and cache_root:
                    os.makedirs(cache_root, exist_ok=True)
                    existing_cache = _TransFeatureCacheManager.find_any_cache(
                        session_name, cache_root,
                        os.path.dirname(video_path), proj)

                save_path = (os.path.join(
                    cache_root,
                    f"{session_name}_features_{cfg_hash}.pkl")
                             if cache_root else None)

                def _extract():
                    return PixelPaws_ExtractFeatures(
                        pose_data_file=dlc_path,
                        video_file_path=video_path
                                        if os.path.isfile(video_path) else '',
                        bp_pixbrt_list=cd0.get('bp_pixbrt_list', []),
                        square_size=cd0.get('square_size', 20),
                        pix_threshold=cd0.get('pix_threshold', 50),
                        bp_include_list=cd0.get('bp_include_list', None),
                    )

                X = _load_features_for_prediction(
                    cache_file=existing_cache,
                    model=cd0.get('clf_model'),
                    extract_fn=_extract,
                    save_path=save_path,
                    log_fn=self._log_msg,
                    dlc_path=dlc_path,
                    clf_data=cd0,
                )
            except Exception as e:
                self._log_msg(f"  {session_name}: feature extraction failed: "
                              f"{e}")
                continue

            n_frames = len(X)
            prob_matrix = np.zeros((n_frames, len(self._loaded_classifiers)))
            binary_matrix = np.zeros(
                (n_frames, len(self._loaded_classifiers)), dtype=int)

            for bi, cd in enumerate(self._loaded_classifiers):
                try:
                    model = cd['clf_model']
                    X_aug = augment_features_post_cache(
                        X.copy(), cd, model, dlc_path,
                        log_fn=self._log_msg)
                    proba = predict_with_xgboost(
                        model, X_aug,
                        calibrator=cd.get('prob_calibrator'),
                        fold_models=cd.get('fold_models'))
                    prob_matrix[:len(proba), bi] = proba

                    thresh = cd.get('best_thresh', 0.5)
                    binary_raw = (proba >= thresh).astype(int)
                    min_bout = cd.get('min_bout', 0)
                    min_after = cd.get('min_after_bout', 0)
                    max_gap = cd.get('max_gap', 0)
                    binary_filt = _apply_bout_filtering(
                        binary_raw, min_bout, min_after, max_gap)
                    binary_matrix[:len(binary_filt), bi] = binary_filt
                except Exception as e:
                    self._log_msg(f"  {session_name} / "
                                  f"{cd.get('Behavior_type','?')}: {e}")

            # Assign states
            states_arr = np.zeros(n_frames, dtype=int)
            if assign_mode == 'priority':
                for frame_i in range(n_frames):
                    assigned = False
                    for rank_pos, clf_idx in enumerate(self._priority_order):
                        if binary_matrix[frame_i, clf_idx] == 1:
                            states_arr[frame_i] = clf_idx + 1
                            assigned = True
                            break
                    if not assigned:
                        states_arr[frame_i] = 0
            else:  # argmax
                for frame_i in range(n_frames):
                    best = int(np.argmax(prob_matrix[frame_i]))
                    best_clf = self._loaded_classifiers[best]
                    thresh = best_clf.get('best_thresh', 0.5)
                    if prob_matrix[frame_i, best] >= thresh:
                        states_arr[frame_i] = best + 1
                    else:
                        states_arr[frame_i] = 0

            seqs[session_name] = states_arr
            self._frame_probs[session_name] = prob_matrix.copy()
            self._log_msg(
                f"  {session_name}: {n_frames} frames assigned")

        if not seqs:
            raise ValueError("No sessions produced valid state sequences.")
        return seqs

    # ------------------------------------------------------------------
    # Subject resolution
    # ------------------------------------------------------------------

    def _resolve_subject(self, session_name):
        """Match session name to a subject in the key file."""
        if self._key_df is None:
            return session_name
        stem = session_name
        for suffix in ['_predictions', '_prediction', '_pred', '_clusters']:
            stem = stem.replace(suffix, '')
        tokens = stem.split('_')
        for subj in self._key_df['Subject']:
            subj_str = str(subj).strip()
            if subj_str in tokens:
                return subj_str
            if f'_{subj_str}_' in f'_{stem}_':
                return subj_str
        return session_name

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def _notify_cache_status(self):
        """Scan checked sessions for stale caches; show one-time info dialog."""
        try:
            cd0 = self._loaded_classifiers[0]
            cfg = {
                'bp_include_list':      cd0.get('bp_include_list', None),
                'bp_pixbrt_list':       cd0.get('bp_pixbrt_list', []),
                'square_size':          cd0.get('square_size', [20]),
                'pix_threshold':        cd0.get('pix_threshold', 0.3),
                'include_optical_flow': cd0.get('include_optical_flow', False),
                'bp_optflow_list':      cd0.get('bp_optflow_list', []),
            }
            cfg_hash = _TransFeatureCacheManager.compute_hash(cfg)
            proj = self.app.current_project_folder.get()
            cache_root = os.path.join(proj, 'features') if proj else None
            if not cache_root:
                return

            checked_sessions = [s for s in self._trans_sessions
                                 if self._trans_session_checked.get(
                                     s['session_name'],
                                     tk.BooleanVar(value=False)).get()]

            stale, missing = [], []
            for session in checked_sessions:
                sname = session['session_name']
                vpath = session.get('video_path', '') or ''
                found = _TransFeatureCacheManager.find_any_cache(
                    sname, cache_root, os.path.dirname(vpath), proj)
                if found is None:
                    missing.append(sname)
                elif cfg_hash not in os.path.basename(found):
                    stale.append(sname)

            parts = []
            if stale:
                parts.append(
                    f"{len(stale)} session(s) have older feature caches \u2014 "
                    f"PixelPaws will attempt to upgrade them without re-reading video.")
            if missing:
                parts.append(
                    f"{len(missing)} session(s) have no cache \u2014 "
                    f"full feature extraction required.")
            if not parts:
                return   # everything is up-to-date, no dialog needed

            messagebox.showinfo(
                "Feature Cache Status",
                "\n\n".join(parts),
                parent=self)
        except Exception:
            pass   # never block compute on a scan failure

    def _stop_compute(self):
        self._stop_event.set()
        self._log_msg("Stop requested \u2014 aborting after current step...")

    def _start_compute(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("Busy", "Computation already running.")
            return
        self._stop_event.clear()
        self._stop_btn.config(state='normal')
        self._progress.start()
        self._read_label_entries()

        # Pre-flight: detect stale feature caches and inform user
        if (self._source_var.get() == 'supervised'
                and self._loaded_classifiers
                and _TRANS_FEATURE_CACHE_AVAILABLE):
            self._notify_cache_status()

        self._worker_thread = threading.Thread(target=self._compute_thread,
                                               daemon=True)
        self._worker_thread.start()

    def _compute_thread(self):
        try:
            self._log_msg("--- Starting transition computation ---")
            self._frame_probs = {}

            # 1. Load state sequences
            if self._source_var.get() == 'unsupervised':
                run = self._run_combo.get()
                if not run:
                    raise ValueError("Select a Discover run first.")
                seqs = self._load_unsupervised_states(run)
            else:
                if self._loaded_classifiers:
                    seqs = self._predict_and_assign_states()
                else:
                    seqs = self._load_supervised_states()

            if not seqs:
                raise ValueError("No state sequences loaded.")

            # 2. Pre-process: smoothing + noise exclusion
            fps = self._fps_var.get()
            smooth_ms = self._smooth_ms_var.get()
            exclude_noise = self._exclude_noise_var.get()
            min_frames = max(1, round(smooth_ms * fps / 1000)) if smooth_ms > 0 else 0

            processed = {}
            for name, seq in seqs.items():
                if self._stop_event.is_set():
                    self._log_msg("Stopped.")
                    self.app.root.after(0, self._on_compute_done)
                    return
                s = seq.copy()
                if exclude_noise:
                    # Replace -1 (noise) with nearest valid state
                    valid_mask = s >= 0
                    if valid_mask.any() and not valid_mask.all():
                        # Forward fill then backward fill
                        valid_idx = np.where(valid_mask)[0]
                        for i in range(len(s)):
                            if s[i] < 0:
                                # Find nearest valid
                                dists = np.abs(valid_idx - i)
                                s[i] = s[valid_idx[dists.argmin()]]
                if min_frames > 0:
                    s = smooth_state_sequence(s, min_frames)
                # Downsample to ~20 Hz (LUPE style): take mode of every N frames
                if self._downsample_var.get() and fps > 20:
                    n = max(1, round(fps / 20))
                    trim = len(s) - len(s) % n
                    if trim > 0:
                        from scipy.stats import mode as scipy_mode
                        chunks = s[:trim].reshape(-1, n)
                        s = scipy_mode(chunks, axis=1).mode.flatten()
                    else:
                        s = s[:0]  # edge case: sequence shorter than one block
                processed[name] = s

            # Apply effective fps after downsampling
            if self._downsample_var.get() and fps > 20:
                n = max(1, round(fps / 20))
                fps = fps / n
            self._effective_fps = fps   # store for _plot_behavior_over_time

            # Duration cap (applies to all time modes)
            if self._dur_limit_var.get():
                cap_frames = int(self._dur_limit_min.get() * 60 * fps)
                processed = {name: s[:cap_frames] for name, s in processed.items()}
                self._log_msg(f"Duration cap: first {self._dur_limit_min.get():.0f} min "
                              f"({cap_frames} frames)")

            # Determine global state set
            all_states_set = set()
            for s in processed.values():
                all_states_set.update(s)
            states = sorted(all_states_set)

            # 2b. Cluster reduction (optional)
            self._merge_info = None
            reduce = self._reduce_clusters_var.get()
            target_n = self._target_clusters_var.get()

            if reduce and len(states) > target_n:
                self._log_msg(f"Reducing {len(states)} clusters -> "
                              f"{target_n} meta-clusters...")
                mapping, states, merge_info = reduce_clusters(
                    processed, states, target_n)
                # Remap all sequences
                for name in processed:
                    processed[name] = np.array(
                        [mapping[v] for v in processed[name]])
                self._merge_info = merge_info
                # Log the merge mapping
                for new_id, old_ids in sorted(merge_info.items()):
                    self._log_msg(f"  Meta-cluster {new_id}: "
                                  f"merged from {old_ids}")
                self._log_msg(f"Reduction complete: {len(states)} "
                              f"meta-clusters")
                # Update status label on main thread
                orig_n = len(all_states_set)
                self.app.root.after(0, lambda: self._reduce_status_label.config(
                    text=f"{orig_n} -> {len(states)} clusters"))
            elif reduce and len(states) <= target_n:
                self._log_msg(f"Already have {len(states)} clusters "
                              f"(<= target {target_n}), skipping reduction.")
                self.app.root.after(0, lambda: self._reduce_status_label.config(
                    text=f"{len(states)} clusters (no reduction needed)"))

            # Exclude 'Other' (state 0) if requested
            if self._exclude_other_var.get() and 0 in states:
                self._log_msg("Excluding 'Other' (state 0) from analysis...")
                for name in processed:
                    seq = processed[name]
                    processed[name] = np.where(seq == 0, -1, seq)
                states = [s for s in states if s != 0]
                self._state_seqs = processed   # update before matrix step
                self._states = states

            self._state_seqs = processed
            self._states = states
            self._session_subjects = {n: self._resolve_subject(n)
                                      for n in processed}

            # Populate label table on main thread
            self.app.root.after(0, lambda: self._populate_label_table(states))

            # 3. Time slicing
            time_mode = self._time_mode_var.get()
            sliced = {}
            if time_mode == 'range':
                start_f = int(round(self._range_start_var.get() * fps))
                end_f = int(round(self._range_end_var.get() * fps))
                for name, s in processed.items():
                    sliced[name] = s[start_f:end_f]
                self._log_msg(f"Time range: {self._range_start_var.get():.0f}s "
                              f"– {self._range_end_var.get():.0f}s")
            else:
                sliced = processed

            # 4. Compute matrices
            zero_diag = self._zero_diag_var.get()
            t_mode = self._transition_mode.get()
            self._matrices = {}
            for name, s in sliced.items():
                if t_mode == 'bout':
                    mat, _ = compute_bout_transition_matrix(
                        s, states=states, normalize=True)
                else:
                    mat, _ = compute_transition_matrix(
                        s, states=states, normalize=True,
                        zero_diagonal=zero_diag)
                self._matrices[name] = mat
                self._log_msg(f"  {name}: {len(s)} frames, "
                              f"{len(states)} states")

            # 5. Windowed transitions (if sliding mode)
            self._windowed = {}
            if time_mode == 'sliding':
                win_s = self._win_sec_var.get()
                step_s = self._step_sec_var.get()
                for name, s in processed.items():
                    wresults, _ = compute_windowed_transitions(
                        s, fps, win_s, step_s, states=states,
                        zero_diagonal=zero_diag, mode=t_mode)
                    self._windowed[name] = wresults
                self._log_msg(f"Sliding windows: {win_s:.0f}s window, "
                              f"{step_s:.0f}s step")

            # 5b. Latent state discovery (if enabled and sliding mode)
            discover_latent = self._discover_latent_var.get()
            n_latent = self._n_latent_var.get()
            self._latent_centroids = None
            self._session_latent_map = {}
            self._n_latent = 0
            self._occupancy = {}
            self._pca_model = None
            self._pca_scores = {}
            self._pca_loadings = None
            self._group_occupancy = {}
            self._group_occupancy_sem = {}

            if discover_latent and time_mode == 'sliding' and self._windowed:
                self._log_msg(f"Running k-means clustering (k={n_latent})...")
                centroids, session_latent_map = cluster_transition_matrices(
                    self._windowed, k=n_latent, states=states)
                self._latent_centroids = centroids
                self._session_latent_map = session_latent_map
                self._n_latent = n_latent

                # State occupancy fractions
                self._occupancy = compute_state_occupancy(
                    session_latent_map, n_latent)
                self._log_msg(f"Computed state occupancy for "
                              f"{len(self._occupancy)} sessions")

                # PCA on occupancy
                if len(self._occupancy) >= 2:
                    self._pca_model, self._pca_scores, self._pca_loadings = \
                        pca_on_occupancy(self._occupancy)
                    self._log_msg("PCA on occupancy complete")

                # Group aggregation of occupancy
                if self._key_df is not None:
                    grp_occ = {}  # {treatment: [array, ...]}
                    for name, occ in self._occupancy.items():
                        subj = self._session_subjects.get(name, name)
                        row = self._key_df[self._key_df['Subject'] == subj]
                        if not row.empty:
                            treatment = str(row.iloc[0]['Treatment'])
                            if treatment not in grp_occ:
                                grp_occ[treatment] = []
                            grp_occ[treatment].append(occ)
                    for grp, occs in grp_occ.items():
                        stack = np.stack(occs)
                        self._group_occupancy[grp] = stack.mean(axis=0)
                        self._group_occupancy_sem[grp] = (
                            stack.std(axis=0, ddof=1) / np.sqrt(len(occs))
                            if len(occs) > 1 else np.zeros(n_latent))
            elif discover_latent and time_mode != 'sliding':
                self._log_msg("NOTE: Latent state discovery requires "
                              "sliding windows mode.")

            # 5c. Temporal probability (always computed)
            prob_bin_sec = self._prob_bin_var.get()
            self._temporal_probs = {}
            for name, s in processed.items():
                self._temporal_probs[name] = compute_temporal_probabilities(
                    s, fps, prob_bin_sec, states)
            self._log_msg(f"Temporal probabilities computed "
                          f"(bin={prob_bin_sec:.0f}s)")

            # 6. Group aggregation
            self._group_matrices = {}
            self._group_sem = {}
            self._group_subject_matrices = {}
            if self._key_df is not None:
                groups = {}  # {treatment: [matrix, ...]}
                for name, mat in self._matrices.items():
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                        if treatment not in groups:
                            groups[treatment] = []
                        groups[treatment].append(mat)

                for grp, mats in groups.items():
                    stack = np.stack(mats)
                    self._group_matrices[grp] = stack.mean(axis=0)
                    self._group_sem[grp] = (stack.std(axis=0, ddof=1) /
                                            np.sqrt(len(mats))
                                            if len(mats) > 1
                                            else np.zeros_like(mats[0]))
                    self._group_subject_matrices[grp] = list(mats)
                    self._log_msg(f"  Group '{grp}': {len(mats)} subjects")

            self._log_msg("--- Computation complete ---")
            self.app.root.after(0, self._on_compute_done)

        except Exception as e:
            self._log_msg(f"ERROR: {e}")
            traceback.print_exc()
            self.app.root.after(0, self._on_compute_done)

    def _on_compute_done(self):
        self._progress.stop()
        self._stop_btn.config(state='disabled')
        self._read_label_entries()
        # Populate pair selector for timeline
        if self._states:
            pairs = []
            for si in self._states:
                for sj in self._states:
                    if si != sj:
                        pairs.append(f"{self._state_name(si)} -> "
                                     f"{self._state_name(sj)}")
            self._pair_combo['values'] = pairs
            if pairs:
                self._pair_combo.set(pairs[0])
        sessions_with_video = [
            s['session_name'] for s in self._trans_sessions
            if s.get('video') or s.get('video_path')
        ]
        self._preview_session_combo['values'] = sessions_with_video
        if sessions_with_video and not self._preview_session_var.get():
            self._preview_session_var.set(sessions_with_video[0])
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Config Save / Load
    # ------------------------------------------------------------------

    def _save_config(self):
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'transitions') if proj else '/'
        os.makedirs(init_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Transitions Config",
            initialdir=init_dir,
            defaultextension='.json',
            filetypes=[('JSON config', '*.json'), ('All files', '*.*')])
        if not path:
            return
        cfg = {
            'source':           self._source_var.get(),
            'assign_mode':      self._assign_mode.get(),
            'transition_mode':  self._transition_mode.get(),
            'fps':              self._fps_var.get(),
            'smooth_ms':        self._smooth_ms_var.get(),
            'exclude_noise':    self._exclude_noise_var.get(),
            'zero_diag':        self._zero_diag_var.get(),
            'reduce_clusters':  self._reduce_clusters_var.get(),
            'target_clusters':  self._target_clusters_var.get(),
            'prob_bin':         self._prob_bin_var.get(),
            'time_mode':        self._time_mode_var.get(),
            'range_start':      self._range_start_var.get(),
            'range_end':        self._range_end_var.get(),
            'win_sec':          self._win_sec_var.get(),
            'step_sec':         self._step_sec_var.get(),
            'downsample':       self._downsample_var.get(),
            'discover_latent':  self._discover_latent_var.get(),
            'n_latent':         self._n_latent_var.get(),
            'key_file':         self._key_file_var.get(),
            'view':             self._view_var.get(),
            'palette':          self._palette_var.get(),
            'show_annot':       self._show_annot_var.get(),
            'classifiers': [
                {
                    'path':           cd.get('_path', ''),
                    'best_thresh':    cd.get('best_thresh'),
                    'min_bout':       cd.get('min_bout'),
                    'min_after_bout': cd.get('min_after_bout'),
                    'max_gap':        cd.get('max_gap'),
                }
                for cd in self._loaded_classifiers
            ],
            'priority_order': list(self._priority_order),
            'session_selection': [
                name for name, var in self._trans_session_checked.items() if var.get()
            ],
        }
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        self._log_msg(f"Config saved to {path}")

    def _load_config(self):
        proj = self.app.current_project_folder.get()
        init_dir = os.path.join(proj, 'transitions') if proj else '/'
        path = filedialog.askopenfilename(
            parent=self,
            title="Load Transitions Config",
            initialdir=init_dir,
            filetypes=[('JSON config', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Config", f"Could not read config: {e}", parent=self)
            return

        # Apply scalar settings
        _sv = lambda key, var: var.set(cfg[key]) if key in cfg else None
        _sv('source',          self._source_var)
        _sv('assign_mode',     self._assign_mode)
        _sv('transition_mode', self._transition_mode)
        _sv('fps',             self._fps_var)
        _sv('smooth_ms',       self._smooth_ms_var)
        _sv('exclude_noise',   self._exclude_noise_var)
        _sv('zero_diag',       self._zero_diag_var)
        _sv('reduce_clusters', self._reduce_clusters_var)
        _sv('target_clusters', self._target_clusters_var)
        _sv('prob_bin',        self._prob_bin_var)
        _sv('time_mode',       self._time_mode_var)
        _sv('range_start',     self._range_start_var)
        _sv('range_end',       self._range_end_var)
        _sv('win_sec',         self._win_sec_var)
        _sv('step_sec',        self._step_sec_var)
        _sv('downsample',      self._downsample_var)
        _sv('discover_latent', self._discover_latent_var)
        _sv('n_latent',        self._n_latent_var)
        _sv('key_file',        self._key_file_var)
        _sv('view',            self._view_var)
        _sv('palette',         self._palette_var)
        _sv('show_annot',      self._show_annot_var)

        # Restore session selection
        saved_sel = cfg.get('session_selection')
        if saved_sel is not None:
            self._pending_session_selection = set(saved_sel)
            self._apply_pending_session_selection()

        # Re-load classifiers
        clf_entries = cfg.get('classifiers', [])
        if clf_entries:
            self._loaded_classifiers.clear()
            self._priority_order.clear()
            missing = []
            for i, entry in enumerate(clf_entries):
                fpath = entry.get('path', '')
                if not fpath or not os.path.isfile(fpath):
                    missing.append(fpath or f'(entry {i})')
                    continue
                try:
                    with open(fpath, 'rb') as fh:
                        cd = pickle.load(fh)
                    if 'clf_model' not in cd:
                        missing.append(fpath)
                        continue
                    cd['_path'] = fpath
                    for key in ('best_thresh', 'min_bout', 'min_after_bout', 'max_gap'):
                        if entry.get(key) is not None:
                            cd[key] = entry[key]
                    self._loaded_classifiers.append(cd)
                    self._priority_order.append(len(self._loaded_classifiers) - 1)
                except Exception as e:
                    missing.append(f"{fpath} ({e})")

            if missing:
                messagebox.showwarning(
                    "Load Config",
                    "Could not load classifier(s):\n" + "\n".join(missing),
                    parent=self)

            saved_order = cfg.get('priority_order', [])
            if len(saved_order) == len(self._loaded_classifiers):
                self._priority_order[:] = saved_order

            self._update_clf_listbox()

        self._log_msg(f"Config loaded from {path}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _on_view_changed(self, event=None):
        view = self._view_var.get()
        if view == 'Timeline':
            self._pair_frame.pack(side='left', padx=10)
        else:
            self._pair_frame.pack_forget()
        self._refresh_plot()

    def _refresh_plot(self):
        if not MATPLOTLIB_AVAILABLE or self._fig is None:
            return
        self._fig.clear()
        view = self._view_var.get()
        try:
            if view == 'Ethogram':
                self._plot_ethogram()
            elif view == 'Temporal Probability':
                self._plot_temporal_probability()
            elif view == 'Heatmap':
                self._plot_heatmap()
            elif view == 'Network':
                self._plot_network()
            elif view == 'Group Comparison':
                self._plot_group_comparison()
            elif view == 'Timeline':
                self._plot_timeline()
            elif view == 'Behavior Over Time':
                self._plot_behavior_over_time()
            elif view == 'Latent States':
                self._plot_latent_states()
            elif view == 'State Occupancy':
                self._plot_state_occupancy()
            elif view == 'PCA':
                self._plot_pca()
            elif view == 'Meta-cluster Summary':
                self._plot_meta_summary()
        except Exception as e:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center',
                    transform=ax.transAxes)
        self._canvas.draw_idle()

    def _get_cmap(self):
        """Return a colormap for states."""
        import matplotlib.colors as mcolors
        n = max(len(self._states), 1)
        name = self._palette_var.get()
        _SNS = {'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'}
        if name in _SNS:
            return mcolors.ListedColormap(sns.color_palette(name, n_colors=n))
        return plt.cm.get_cmap(name, n)

    def _plot_ethogram(self):
        """Color-coded timeline bars per subject, grouped by treatment."""
        if not self._state_seqs:
            return
        fps = self._fps_var.get()
        cmap = self._get_cmap()
        state_to_idx = {s: i for i, s in enumerate(self._states)}

        # Order sessions by group if key file loaded
        ordered = []
        if self._key_df is not None and self._group_matrices:
            for grp in sorted(self._group_matrices.keys()):
                for name in sorted(self._state_seqs.keys()):
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty and str(row.iloc[0]['Treatment']) == grp:
                        ordered.append((grp, name))
        if not ordered:
            ordered = [('', name) for name in sorted(self._state_seqs.keys())]

        ax = self._fig.add_subplot(111)
        n_sessions = len(ordered)
        yticks, ylabels = [], []

        # Downsample to ≤3000 columns for display performance
        max_frames = max((len(self._state_seqs[n]) for _, n in ordered), default=1)
        n_cols = min(max_frames, 3000)

        # Build RGBA image: white background = no data / excluded
        img = np.ones((n_sessions, n_cols, 4), dtype=float)

        for yi, (grp, name) in enumerate(ordered):
            seq = np.asarray(self._state_seqs[name])
            n_frames = len(seq)
            # Sample at evenly-spaced column centres
            col_indices = np.linspace(0, n_frames - 1, n_cols).astype(int)
            sampled = seq[col_indices]
            for xi, state_val in enumerate(sampled):
                idx = state_to_idx.get(int(state_val), -1)
                if idx >= 0:
                    img[yi, xi] = cmap(idx)
            subj = self._session_subjects.get(name, name)
            yticks.append(yi)
            ylabels.append(f"{subj} ({grp})" if grp else subj)

        t_end = max_frames / fps
        ax.imshow(img, aspect='auto',
                  extent=[0, t_end, n_sessions - 0.5, -0.5],
                  interpolation='nearest')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title("Ethogram")

        # Legend
        patches = [mpatches.Patch(color=cmap(i), label=self._state_name(s))
                   for i, s in enumerate(self._states)]
        ax.legend(handles=patches, loc='upper right', fontsize=7,
                  ncol=max(1, len(patches) // 6))

    def _plot_heatmap(self):
        """Transition probability heatmap (mean across all sessions)."""
        if not self._matrices:
            return
        # Average all session matrices
        all_mats = list(self._matrices.values())
        mean_mat = np.mean(np.stack(all_mats), axis=0)

        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]
        if n_states > 15:
            labels = [str(s) for s in self._states]
        show_annot = n_states <= 20 and self._show_annot_var.get()
        annot_size = max(5, 10 - n_states // 3) if show_annot else 7
        tick_size = max(5, 9 - n_states // 5)

        _shrink = min(0.85, max(0.35, n_states / (n_states + 6)))
        _SEQUENTIAL = {'magma', 'plasma', 'viridis', 'inferno', 'cividis', 'turbo'}
        _heatmap_cmap = self._palette_var.get() if self._palette_var.get() in _SEQUENTIAL else 'YlOrRd'
        ax = self._fig.add_subplot(111)
        sns.heatmap(mean_mat, annot=show_annot,
                    fmt='.1f' if n_states > 10 else '.2f',
                    annot_kws={'size': annot_size} if show_annot else {},
                    cmap=_heatmap_cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, vmin=0, vmax=1, square=n_states <= 20,
                    cbar_kws={'label': 'P(transition)', 'shrink': _shrink})
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_title("Transition Probability Matrix (all subjects)")
        ax.tick_params(labelsize=tick_size)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    def _plot_network(self):
        """Directed graph: node size ~ time-in-state, edge width ~ P(transition)."""
        if not NETWORKX_AVAILABLE:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "networkx not installed.\npip install networkx",
                    ha='center', va='center', transform=ax.transAxes)
            return
        if not self._matrices:
            return

        all_mats = list(self._matrices.values())
        mean_mat = np.mean(np.stack(all_mats), axis=0)

        # Time-in-state fractions
        all_seqs = np.concatenate(list(self._state_seqs.values()))
        state_fracs = {}
        for i, s in enumerate(self._states):
            state_fracs[s] = (all_seqs == s).mean()

        G = nx.DiGraph()
        cmap = self._get_cmap()
        for i, si in enumerate(self._states):
            G.add_node(self._state_name(si), size=state_fracs.get(si, 0.01),
                       color=cmap(i))
        threshold = 0.02
        for i, si in enumerate(self._states):
            for j, sj in enumerate(self._states):
                if i != j and mean_mat[i, j] > threshold:
                    G.add_edge(self._state_name(si), self._state_name(sj),
                               weight=mean_mat[i, j])

        ax = self._fig.add_subplot(111)
        pos = nx.spring_layout(G, seed=42, k=2.0)
        node_sizes = [G.nodes[n]['size'] * 3000 + 200 for n in G.nodes]
        node_colors = [G.nodes[n]['color'] for n in G.nodes]
        edge_widths = [G.edges[e]['weight'] * 5 for e in G.edges]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               alpha=0.6, edge_color='gray',
                               arrows=True, arrowsize=15,
                               connectionstyle='arc3,rad=0.15')
        # Edge labels
        edge_labels = {e: f"{G.edges[e]['weight']:.2f}" for e in G.edges
                       if G.edges[e]['weight'] > 0.05}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6)
        ax.set_title("State Transition Network")
        ax.axis('off')

    def _compute_sig_matrix(self, mats_a, mats_b):
        """Return Bonferroni-corrected p-value matrix (Mann-Whitney U, two-sided)."""
        from scipy.stats import mannwhitneyu
        n = mats_a[0].shape[0]
        pvals = np.ones((n, n))
        if len(mats_a) < 2 or len(mats_b) < 2:
            return pvals  # not enough subjects
        for i in range(n):
            for j in range(n):
                a = [m[i, j] for m in mats_a]
                b = [m[i, j] for m in mats_b]
                try:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                    pvals[i, j] = p
                except Exception:
                    pass
        pvals = np.minimum(pvals * n * n, 1.0)  # Bonferroni
        return pvals

    def _plot_group_comparison(self):
        """Side-by-side heatmaps + difference matrix."""
        if not self._group_matrices:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Load a key file and compute\nto see group comparison.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        groups = sorted(self._group_matrices.keys())
        n_groups = len(groups)
        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]
        if n_states > 15:
            labels = [str(s) for s in self._states]
        show_annot = n_states <= 15 and self._show_annot_var.get()
        annot_size = max(5, 9 - n_states // 3) if show_annot else 7
        tick_size = max(4, 8 - n_states // 5)
        use_square = n_states <= 15
        _shrink = min(0.85, max(0.35, n_states / (n_states + 6)))
        _diff_shrink = min(0.75, max(0.28, n_states / (n_states + 9)))
        _SEQUENTIAL = {'magma', 'plasma', 'viridis', 'inferno', 'cividis', 'turbo'}
        _heatmap_cmap = self._palette_var.get() if self._palette_var.get() in _SEQUENTIAL else 'YlOrRd'
        # n_groups heatmaps + difference if exactly 2 groups
        n_plots = n_groups + (1 if n_groups == 2 else 0)
        axes = self._fig.subplots(1, n_plots, squeeze=False)[0]

        for gi, grp in enumerate(groups):
            mat = self._group_matrices[grp]
            # Only show colorbar on the first group heatmap — all share 0-1 scale
            show_cbar = (gi == 0)
            sns.heatmap(mat, annot=show_annot,
                        fmt='.1f' if n_states > 10 else '.2f',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=_heatmap_cmap,
                        xticklabels=labels, yticklabels=labels,
                        ax=axes[gi], vmin=0, vmax=1, square=use_square,
                        cbar=show_cbar,
                        cbar_kws={'shrink': _shrink} if show_cbar else {})
            axes[gi].set_title(grp, fontsize=10)
            axes[gi].set_xlabel("To")
            # Only label y-axis on the leftmost subplot
            axes[gi].set_ylabel("From" if gi == 0 else "")
            axes[gi].tick_params(labelsize=tick_size)
            axes[gi].set_xticklabels(axes[gi].get_xticklabels(),
                                     rotation=45, ha='right')
            axes[gi].set_yticklabels(
                axes[gi].get_yticklabels() if gi == 0 else [], rotation=0)

        if n_groups == 2:
            diff = (self._group_matrices[groups[1]] -
                    self._group_matrices[groups[0]])
            vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
            sns.heatmap(diff, annot=show_annot,
                        fmt='.1f' if n_states > 10 else '.2f',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=_heatmap_cmap,
                        xticklabels=labels, yticklabels=[],
                        ax=axes[-1], vmin=-vmax, vmax=vmax, square=use_square,
                        cbar_kws={'shrink': _diff_shrink})
            axes[-1].set_title(f"{groups[1]} - {groups[0]}", fontsize=10)
            axes[-1].set_xlabel("To")
            axes[-1].set_ylabel("")
            axes[-1].tick_params(labelsize=tick_size)
            axes[-1].set_xticklabels(axes[-1].get_xticklabels(),
                                     rotation=45, ha='right')

            if (self._show_sig_var.get()
                    and all(g in self._group_subject_matrices for g in groups)):
                mats_a = self._group_subject_matrices[groups[0]]
                mats_b = self._group_subject_matrices[groups[1]]
                pvals = self._compute_sig_matrix(mats_a, mats_b)
                for i in range(n_states):
                    for j in range(n_states):
                        p = pvals[i, j]
                        if p < 0.001:
                            marker = '***'
                        elif p < 0.01:
                            marker = '**'
                        elif p < 0.05:
                            marker = '*'
                        else:
                            continue
                        axes[-1].text(j + 0.5, i + 0.5, marker,
                                      ha='center', va='center',
                                      fontsize=max(6, annot_size),
                                      color='black', fontweight='bold')

    def _plot_timeline(self):
        """Line plot of P(state_i -> state_j) over time for sliding windows."""
        if not self._windowed:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Use 'Sliding windows' mode\nto see timeline.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Parse selected pair
        pair_text = self._pair_combo.get()
        if ' -> ' not in pair_text:
            return
        from_name, to_name = pair_text.split(' -> ', 1)
        # Find indices
        name_to_idx = {}
        for i, s in enumerate(self._states):
            name_to_idx[self._state_name(s)] = i
        if from_name not in name_to_idx or to_name not in name_to_idx:
            return
        fi, ti = name_to_idx[from_name], name_to_idx[to_name]

        ax = self._fig.add_subplot(111)

        # Group by treatment if key file loaded
        if self._key_df is not None and self._group_matrices:
            groups = {}  # {treatment: [(times, probs), ...]}
            for name, wresults in self._windowed.items():
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                if not row.empty:
                    grp = str(row.iloc[0]['Treatment'])
                else:
                    grp = 'Unknown'
                if grp not in groups:
                    groups[grp] = []
                times = [t for t, _ in wresults]
                probs = [m[fi, ti] for _, m in wresults]
                groups[grp].append((times, probs))

            colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
            for ci, (grp, traces) in enumerate(sorted(groups.items())):
                # Align to common time grid
                all_times = sorted(set(t for ts, _ in traces for t in ts))
                aligned = np.full((len(traces), len(all_times)), np.nan)
                for ri, (ts, ps) in enumerate(traces):
                    for t, p in zip(ts, ps):
                        idx = all_times.index(t)
                        aligned[ri, idx] = p
                mean = np.nanmean(aligned, axis=0)
                sem = (np.nanstd(aligned, axis=0, ddof=1) /
                       np.sqrt(np.sum(~np.isnan(aligned), axis=0)))
                sem = np.nan_to_num(sem)
                ax.plot(all_times, mean, label=grp, color=colors[ci], linewidth=2)
                ax.fill_between(all_times, mean - sem, mean + sem,
                                alpha=0.2, color=colors[ci])
        else:
            # Individual traces
            for name, wresults in self._windowed.items():
                times = [t for t, _ in wresults]
                probs = [m[fi, ti] for _, m in wresults]
                ax.plot(times, probs, label=self._session_subjects.get(name, name),
                        alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(transition)")
        ax.set_title(f"Transition: {pair_text}")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    def _plot_behavior_over_time(self):
        """Mean ± SEM behavior occupancy over time, one subplot per group.
        Requires sliding-windows mode so _windowed is populated."""
        if not self._windowed:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Use 'Sliding windows' mode and re-compute\n"
                    "to see the Behavior Over Time plot.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        fps = self._effective_fps
        win_s  = self._win_sec_var.get()
        step_s = self._step_sec_var.get()
        win_frames  = max(1, int(round(win_s  * fps)))
        step_frames = max(1, int(round(step_s * fps)))

        # Compute per-session occupancy time-courses
        session_occ = {}
        for name, seq in self._state_seqs.items():
            s = np.asarray(seq, dtype=int)
            occ_series = []
            for start in range(0, len(s) - win_frames + 1, step_frames):
                chunk = s[start:start + win_frames]
                center_min = (start + win_frames / 2) / fps / 60.0
                fracs = {sid: float(np.sum(chunk == sid)) / len(chunk)
                         for sid in self._states}
                occ_series.append((center_min, fracs))
            session_occ[name] = occ_series

        # Determine groups
        if self._key_df is not None and self._group_matrices:
            groups = sorted(self._group_matrices.keys())
            def _get_group(name):
                subj = self._session_subjects.get(name, name)
                row  = self._key_df[self._key_df['Subject'] == subj]
                return str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
        else:
            groups = ['All']
            def _get_group(name): return 'All'

        n_groups = len(groups)
        axes = self._fig.subplots(1, n_groups, squeeze=False, sharey=True)[0]

        # Get colors from selected palette
        n_states = len(self._states)
        pal_name = self._bot_palette_var.get()
        _SNS = {'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'}
        if pal_name in _SNS:
            raw_colors = sns.color_palette(pal_name, n_colors=n_states)
        else:
            cmap = plt.cm.get_cmap(pal_name, n_states)
            raw_colors = [cmap(i) for i in range(n_states)]

        for gi, grp in enumerate(groups):
            ax = axes[gi]
            grp_sessions = [n for n in session_occ if _get_group(n) == grp]
            if not grp_sessions:
                ax.set_title(grp)
                continue

            # Build common time grid
            all_times = sorted({t for name in grp_sessions
                                for t, _ in session_occ[name]})
            if not all_times:
                continue
            time_arr = np.array(all_times)

            for si, sid in enumerate(self._states):
                color = raw_colors[si % len(raw_colors)]
                sname = self._state_name(sid)

                # Align each session to the common time grid
                mat = np.full((len(grp_sessions), len(all_times)), np.nan)
                for ri, name in enumerate(grp_sessions):
                    t2f = {t: f.get(sid, 0.0) for t, f in session_occ[name]}
                    for ci_t, t in enumerate(all_times):
                        if t in t2f:
                            mat[ri, ci_t] = t2f[t]

                n_valid = np.sum(~np.isnan(mat), axis=0)
                mean    = np.nanmean(mat, axis=0)
                sem     = np.where(n_valid > 1,
                                   np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_valid),
                                   0.0)

                ax.plot(time_arr, mean, color=color, linewidth=2, label=sname)
                ax.fill_between(time_arr, mean - sem, mean + sem,
                                alpha=0.2, color=color)

            ax.set_title(grp, fontsize=10)
            ax.set_xlabel("Time (min)")
            if gi == 0:
                ax.set_ylabel("Behaviour probability")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)

        # Shared legend on last axis
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, fontsize=8, loc='upper right')

    # ------------------------------------------------------------------
    # New LUPE-inspired visualizations
    # ------------------------------------------------------------------

    def _plot_temporal_probability(self):
        """Stacked area chart of behavior fractions over time (LUPE panel e)."""
        if not self._temporal_probs:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Run Compute Transitions first.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        cmap = self._get_cmap()
        state_labels = [self._state_name(s) for s in self._states]
        colors = [cmap(i) for i in range(len(self._states))]

        # Group sessions by treatment if key file loaded
        if self._key_df is not None and self._group_matrices:
            groups = {}
            for name in self._temporal_probs:
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                grp = str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
                if grp not in groups:
                    groups[grp] = []
                groups[grp].append(name)

            sorted_groups = sorted(groups.keys())
            n_groups = len(sorted_groups)
            axes = self._fig.subplots(n_groups, 1, squeeze=False)[:, 0]

            for gi, grp in enumerate(sorted_groups):
                ax = axes[gi]
                sessions = groups[grp]
                # Average temporal probs across sessions in group
                # Align to common time grid via interpolation
                all_centers = []
                all_probs = []
                for name in sessions:
                    centers, prob = self._temporal_probs[name]
                    all_centers.append(centers)
                    all_probs.append(prob)

                if len(sessions) == 1:
                    t = all_centers[0]
                    mean_prob = all_probs[0]
                else:
                    # Use the shortest common time range
                    min_len = min(len(c) for c in all_centers)
                    t = all_centers[0][:min_len]
                    stacked = np.stack([p[:min_len] for p in all_probs])
                    mean_prob = stacked.mean(axis=0)

                ax.stackplot(t, mean_prob.T, labels=state_labels,
                             colors=colors, alpha=0.85)
                ax.set_title(grp, fontsize=10)
                ax.set_ylabel("Fraction")
                ax.set_ylim(0, 1)
                if gi == n_groups - 1:
                    ax.set_xlabel("Time (s)")
                if gi == 0:
                    ax.legend(loc='upper right', fontsize=7,
                              ncol=max(1, len(state_labels) // 4))
        else:
            # One subplot per session (max 6, then average)
            sessions = sorted(self._temporal_probs.keys())
            if len(sessions) <= 6:
                axes = self._fig.subplots(len(sessions), 1,
                                          squeeze=False)[:, 0]
                for si, name in enumerate(sessions):
                    ax = axes[si]
                    centers, prob = self._temporal_probs[name]
                    ax.stackplot(centers, prob.T, labels=state_labels,
                                 colors=colors, alpha=0.85)
                    subj = self._session_subjects.get(name, name)
                    ax.set_title(subj, fontsize=9)
                    ax.set_ylim(0, 1)
                    if si == len(sessions) - 1:
                        ax.set_xlabel("Time (s)")
                    if si == 0:
                        ax.legend(loc='upper right', fontsize=7,
                                  ncol=max(1, len(state_labels) // 4))
            else:
                # Average all
                ax = self._fig.add_subplot(111)
                min_len = min(len(self._temporal_probs[n][0])
                              for n in sessions)
                t = self._temporal_probs[sessions[0]][0][:min_len]
                stacked = np.stack(
                    [self._temporal_probs[n][1][:min_len] for n in sessions])
                mean_prob = stacked.mean(axis=0)
                ax.stackplot(t, mean_prob.T, labels=state_labels,
                             colors=colors, alpha=0.85)
                ax.set_title(f"Mean across {len(sessions)} sessions")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Fraction")
                ax.set_ylim(0, 1)
                ax.legend(loc='upper right', fontsize=7,
                          ncol=max(1, len(state_labels) // 4))

    def _plot_latent_states(self):
        """Grid of centroid transition matrix heatmaps (LUPE panel h left)."""
        if self._latent_centroids is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Enable 'Discover latent behavioral states'\n"
                    "with sliding windows mode and re-compute.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        k = self._latent_centroids.shape[0]
        n_states = len(self._states)
        labels = [self._state_name(s) for s in self._states]

        # Adapt layout to state count
        show_annot = self._show_annot_var.get() and n_states <= 12
        use_square = n_states <= 15
        annot_size = max(5, 9 - n_states // 4) if show_annot else 7
        tick_size = max(4, 8 - n_states // 5)
        # Abbreviate labels if too many states
        if n_states > 10:
            labels = [str(s) for s in self._states]

        n_cols = 3 if k > 2 else k
        n_rows = int(np.ceil(k / n_cols))

        cmap = self._get_cmap()

        axes = self._fig.subplots(n_rows, n_cols, squeeze=False)
        for i in range(k):
            r, c = divmod(i, n_cols)
            ax = axes[r][c]
            mat = self._latent_centroids[i]
            sns.heatmap(mat, annot=show_annot,
                        fmt='.1f' if n_states > 8 else '.2f',
                        annot_kws={'size': annot_size} if show_annot else {},
                        cmap=cmap,
                        xticklabels=labels, yticklabels=labels,
                        ax=ax, vmin=0, vmax=1, square=use_square,
                        cbar=False)
            ax.set_title(f"Latent State {i + 1}", fontsize=9)
            ax.tick_params(labelsize=tick_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Hide unused axes
        for i in range(k, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes[r][c].set_visible(False)

    def _plot_state_occupancy(self):
        """Grouped bar chart of latent state occupancy (LUPE panel h right)."""
        if not self._occupancy:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Enable 'Discover latent behavioral states'\n"
                    "with sliding windows mode and re-compute.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        ax = self._fig.add_subplot(111)
        k = self._n_latent
        x = np.arange(k)
        state_labels = [f"LS {i + 1}" for i in range(k)]

        if self._group_occupancy:
            groups = sorted(self._group_occupancy.keys())
            n_g = len(groups)
            width = 0.7 / n_g
            grp_colors = plt.cm.tab10(np.linspace(0, 1, max(n_g, 1)))

            for gi, grp in enumerate(groups):
                offset = (gi - n_g / 2 + 0.5) * width
                means = self._group_occupancy[grp]
                sems = self._group_occupancy_sem[grp]
                ax.bar(x + offset, means, width, yerr=sems,
                       label=grp, color=grp_colors[gi], alpha=0.8,
                       capsize=3)

                # Overlay individual animal dots
                for name, occ in self._occupancy.items():
                    subj = self._session_subjects.get(name, name)
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty and str(row.iloc[0]['Treatment']) == grp:
                        jitter = np.random.uniform(-width * 0.3,
                                                   width * 0.3, k)
                        ax.scatter(x + offset + jitter, occ,
                                   color=grp_colors[gi], edgecolors='black',
                                   linewidths=0.5, s=20, zorder=5, alpha=0.7)

            ax.legend(fontsize=8)
        else:
            # No groups — show per-session bars
            sessions = sorted(self._occupancy.keys())
            n_s = len(sessions)
            width = 0.7 / max(n_s, 1)
            colors = plt.cm.tab20(np.linspace(0, 1, max(n_s, 1)))
            for si, name in enumerate(sessions):
                offset = (si - n_s / 2 + 0.5) * width
                occ = self._occupancy[name]
                subj = self._session_subjects.get(name, name)
                ax.bar(x + offset, occ, width, label=subj,
                       color=colors[si], alpha=0.8)
            if n_s <= 10:
                ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Latent State")
        ax.set_ylabel("Fraction of Time")
        ax.set_title("Latent State Occupancy")
        ax.set_ylim(bottom=0)

    def _plot_pca(self):
        """PCA scatter + loadings (LUPE panels i-l)."""
        if not self._pca_scores or self._pca_model is None:
            ax = self._fig.add_subplot(111)
            msg = ("Enable 'Discover latent behavioral states'\n"
                   "with sliding windows mode and re-compute.\n"
                   "(Requires >= 2 sessions)")
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    transform=ax.transAxes)
            return

        axes = self._fig.subplots(1, 2, squeeze=False)[0]
        ax_scatter = axes[0]
        ax_load = axes[1]

        # --- Scatter plot ---
        if self._key_df is not None:
            groups = {}
            for name, (pc1, pc2) in self._pca_scores.items():
                subj = self._session_subjects.get(name, name)
                row = self._key_df[self._key_df['Subject'] == subj]
                grp = str(row.iloc[0]['Treatment']) if not row.empty else 'Unknown'
                if grp not in groups:
                    groups[grp] = ([], [])
                groups[grp][0].append(pc1)
                groups[grp][1].append(pc2)

            sorted_groups = sorted(groups.keys())
            grp_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_groups), 1)))
            for gi, grp in enumerate(sorted_groups):
                xs, ys = groups[grp]
                ax_scatter.scatter(xs, ys, label=grp, color=grp_colors[gi],
                                   s=60, edgecolors='black', linewidths=0.5,
                                   alpha=0.8)
                # 95% confidence ellipse
                if len(xs) >= 3:
                    from matplotlib.patches import Ellipse
                    mean_x, mean_y = np.mean(xs), np.mean(ys)
                    cov = np.cov(xs, ys)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    angle = np.degrees(np.arctan2(eigvecs[1, 0],
                                                   eigvecs[0, 0]))
                    # 95% CI: chi2 with 2 dof at 0.05 = 5.991
                    scale = np.sqrt(5.991)
                    w = 2 * scale * np.sqrt(max(eigvals[0], 0))
                    h = 2 * scale * np.sqrt(max(eigvals[1], 0))
                    ell = Ellipse(xy=(mean_x, mean_y), width=w, height=h,
                                  angle=angle, facecolor=grp_colors[gi],
                                  alpha=0.15, edgecolor=grp_colors[gi],
                                  linewidth=1.5)
                    ax_scatter.add_patch(ell)

            ax_scatter.legend(fontsize=8)
        else:
            xs = [v[0] for v in self._pca_scores.values()]
            ys = [v[1] for v in self._pca_scores.values()]
            labels = [self._session_subjects.get(n, n)
                      for n in self._pca_scores]
            ax_scatter.scatter(xs, ys, s=60, edgecolors='black',
                               linewidths=0.5)
            for lbl, x, y in zip(labels, xs, ys):
                ax_scatter.annotate(lbl, (x, y), fontsize=7,
                                    textcoords='offset points',
                                    xytext=(5, 5))

        var_explained = self._pca_model.explained_variance_ratio_
        ax_scatter.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
        if len(var_explained) > 1:
            ax_scatter.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
        else:
            ax_scatter.set_ylabel("PC2")
        ax_scatter.set_title("PCA on Latent State Occupancy")
        ax_scatter.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_scatter.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # --- Loadings bar chart ---
        k = self._pca_loadings.shape[1]
        x = np.arange(k)
        ls_labels = [f"LS {i + 1}" for i in range(k)]
        width = 0.35
        ax_load.bar(x - width / 2, self._pca_loadings[0], width,
                     label=f"PC1 ({var_explained[0]*100:.1f}%)",
                     color='steelblue')
        if self._pca_loadings.shape[0] >= 2:
            ax_load.bar(x + width / 2, self._pca_loadings[1], width,
                         label=f"PC2 ({var_explained[1]*100:.1f}%)",
                         color='coral')
        ax_load.set_xticks(x)
        ax_load.set_xticklabels(ls_labels, fontsize=8)
        ax_load.set_xlabel("Latent State")
        ax_load.set_ylabel("Loading")
        ax_load.set_title("PCA Loadings")
        ax_load.legend(fontsize=8)
        ax_load.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # ------------------------------------------------------------------
    # Meta-cluster Summary
    # ------------------------------------------------------------------

    def _build_meta_mapping(self):
        """Return dict {original_cluster_id: meta_cluster_id}, or None."""
        if self._merge_info is None:
            return None
        mapping = {}
        for new_id, old_ids in self._merge_info.items():
            for old_id in old_ids:
                mapping[old_id] = new_id
        return mapping

    def _plot_meta_summary(self):
        """UMAP scatter colored by meta-cluster + composition table."""
        has_embedding = (self._model_bundle is not None
                         and 'embedding' in self._model_bundle
                         and 'cluster_labels' in self._model_bundle)

        if not self._states:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data loaded.\nRun Compute Transitions first.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            return

        meta_mapping = self._build_meta_mapping()

        if has_embedding:
            axes = self._fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
            ax_scatter, ax_table = axes
            self._draw_meta_umap(ax_scatter, meta_mapping)
        else:
            ax_table = self._fig.add_subplot(111)
            if self._model_bundle is None:
                ax_table.set_title(
                    "No embedding available (requires model.pkl from Discover run)",
                    fontsize=10, fontstyle='italic')

        self._draw_composition_table(ax_table, meta_mapping)

    def _draw_meta_umap(self, ax, meta_mapping):
        """Draw UMAP scatter colored by meta-clusters (or original clusters)."""
        embedding = self._model_bundle['embedding']
        orig_labels = self._model_bundle['cluster_labels']

        # Remap labels to meta-clusters if reduction is active
        if meta_mapping is not None:
            plot_labels = np.array([meta_mapping.get(l, -1) for l in orig_labels])
        else:
            plot_labels = orig_labels.copy()

        # Subsample for performance (cap at 150k points)
        n_pts = len(embedding)
        max_pts = 150_000
        if n_pts > max_pts:
            idx = np.random.default_rng(42).choice(n_pts, max_pts, replace=False)
            embedding = embedding[idx]
            plot_labels = plot_labels[idx]

        cmap = self._get_cmap()
        unique_labels = sorted(set(plot_labels))

        for label in unique_labels:
            mask = plot_labels == label
            if label == -1:
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c='lightgrey', s=1, alpha=0.3, label='Noise',
                           rasterized=True)
            else:
                state_idx = self._states.index(label) if label in self._states else label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[cmap(state_idx)], s=1, alpha=0.4,
                           label=self._state_name(label), rasterized=True)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        n_clusters = len([l for l in unique_labels if l != -1])
        if meta_mapping is not None:
            ax.set_title(f"UMAP colored by meta-clusters ({n_clusters})")
        else:
            ax.set_title(f"UMAP ({n_clusters} clusters)")
        ax.legend(loc='best', fontsize=6, markerscale=5, ncol=max(1, n_clusters // 10))

    def _draw_composition_table(self, ax, meta_mapping):
        """Draw composition table showing cluster details."""
        ax.axis('off')

        # Compute frame counts per state across all sessions
        total_frames = sum(len(s) for s in self._state_seqs.values())

        rows = []
        for sid in self._states:
            label = self._state_name(sid)
            if meta_mapping is not None and self._merge_info is not None:
                merged_from = self._merge_info.get(sid, [sid])
                merged_str = ', '.join(str(x) for x in merged_from)
            else:
                merged_from = [sid]
                merged_str = str(sid)

            # Count frames for this state across all sessions
            frame_count = sum(
                np.sum(seq == sid) for seq in self._state_seqs.values())
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            rows.append([str(sid), label, merged_str,
                         f"{frame_count:,}", f"{pct:.1f}%"])

        if not rows:
            ax.text(0.5, 0.5, "No states to display.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        col_labels = ['ID', 'Label', 'Merged From', 'Frames', '%']
        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc='center', cellLoc='center')

        # Adaptive font size
        n_rows = len(rows)
        font_size = max(7, min(10, 14 - n_rows // 3))
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.3)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

        # Color-code the ID column to match cluster colors
        cmap = self._get_cmap()
        for i, sid in enumerate(self._states):
            cell = table[i + 1, 0]
            state_idx = self._states.index(sid)
            rgba = cmap(state_idx)
            cell.set_facecolor((*rgba[:3], 0.3))

        if meta_mapping is not None:
            ax.set_title("Meta-cluster Composition", fontsize=11,
                         fontweight='bold', pad=10)
        else:
            ax.set_title("Cluster Composition", fontsize=11,
                         fontweight='bold', pad=10)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _output_dir(self):
        folder = self.app.current_project_folder.get()
        out = os.path.join(folder, 'analysis', 'transitions')
        os.makedirs(out, exist_ok=True)
        return out

    def _open_video_preview(self):
        session_name = self._preview_session_var.get()
        if not session_name:
            messagebox.showwarning("No session", "Select a session to preview.",
                                   parent=self)
            return
        if session_name not in self._state_seqs:
            messagebox.showwarning("No data",
                "Run computation first so state sequences are available.",
                parent=self)
            return

        # Resolve video path
        video_path = ''
        for s in self._trans_sessions:
            if s.get('session_name') == session_name:
                video_path = s.get('video') or s.get('video_path', '')
                break
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("Video not found",
                f"Cannot locate video file for '{session_name}'.", parent=self)
            return

        state_seq  = self._state_seqs[session_name]
        prob_mat   = self._frame_probs.get(session_name)   # may be None (unsupervised)
        state_names = ['Other'] + [
            cd.get('Behavior_type', f'State {i+1}')
            for i, cd in enumerate(self._loaded_classifiers)
        ]

        TransitionVideoPreview(
            parent=self,
            video_path=video_path,
            state_seq=state_seq,
            prob_matrix=prob_mat,
            state_names=state_names,
            session_name=session_name,
        )

    def _export_matrices(self):
        if not self._matrices:
            messagebox.showwarning("No data", "Run Compute Transitions first.")
            return
        out_dir = self._output_dir()
        labels = [self._state_name(s) for s in self._states]

        # Per-subject
        for name, mat in self._matrices.items():
            subj = self._session_subjects.get(name, name)
            df = pd.DataFrame(mat, index=labels, columns=labels)
            df.to_csv(os.path.join(out_dir, f"transition_matrix_{subj}.csv"))

        # Group means
        for grp, mat in self._group_matrices.items():
            df = pd.DataFrame(mat, index=labels, columns=labels)
            df.to_csv(os.path.join(out_dir, f"group_mean_matrix_{grp}.csv"))

        # Windowed
        for name, wresults in self._windowed.items():
            subj = self._session_subjects.get(name, name)
            rows = []
            for t, mat in wresults:
                row = {'time_center_s': t}
                for i, si in enumerate(self._states):
                    for j, sj in enumerate(self._states):
                        row[f"{self._state_name(si)}->{self._state_name(sj)}"] = mat[i, j]
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, f"windowed_transitions_{subj}.csv"),
                index=False)

        # Latent state exports
        if self._latent_centroids is not None:
            labels_ls = [self._state_name(s) for s in self._states]
            for i in range(self._latent_centroids.shape[0]):
                df = pd.DataFrame(self._latent_centroids[i],
                                  index=labels_ls, columns=labels_ls)
                df.to_csv(os.path.join(out_dir,
                                       f"latent_centroid_{i + 1}.csv"))

        if self._occupancy:
            rows = []
            for name, occ in self._occupancy.items():
                subj = self._session_subjects.get(name, name)
                treatment = ''
                if self._key_df is not None:
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                r = {'session': name, 'subject': subj,
                     'treatment': treatment}
                for si in range(len(occ)):
                    r[f'state_{si + 1}_frac'] = occ[si]
                rows.append(r)
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, 'state_occupancy.csv'), index=False)

        if self._pca_scores:
            rows = []
            for name, (pc1, pc2) in self._pca_scores.items():
                subj = self._session_subjects.get(name, name)
                treatment = ''
                if self._key_df is not None:
                    row = self._key_df[self._key_df['Subject'] == subj]
                    if not row.empty:
                        treatment = str(row.iloc[0]['Treatment'])
                rows.append({'session': name, 'subject': subj,
                             'treatment': treatment,
                             'PC1': pc1, 'PC2': pc2})
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, 'pca_scores.csv'), index=False)

        self._log_msg(f"Matrices exported to {out_dir}")
        messagebox.showinfo("Export", f"Matrices saved to:\n{out_dir}")

    def _export_sequences(self):
        if not self._state_seqs:
            messagebox.showwarning("No data", "Run Compute Transitions first.")
            return
        out_dir = self._output_dir()
        fps = self._fps_var.get()
        for name, seq in self._state_seqs.items():
            subj = self._session_subjects.get(name, name)
            df = pd.DataFrame({
                'frame': range(len(seq)),
                'time_s': np.arange(len(seq)) / fps,
                'state_id': seq,
                'state_label': [self._state_name(s) for s in seq],
            })
            df.to_csv(os.path.join(out_dir, f"state_sequence_{subj}.csv"),
                      index=False)
        self._log_msg(f"Sequences exported to {out_dir}")
        messagebox.showinfo("Export", f"Sequences saved to:\n{out_dir}")

    def _export_figure(self, fmt='png'):
        if self._fig is None:
            return
        out_dir = self._output_dir()
        view = self._view_var.get().lower().replace(' ', '_')
        path = os.path.join(out_dir, f"{view}.{fmt}")
        self._fig.savefig(path, dpi=200, bbox_inches='tight')
        self._log_msg(f"Figure saved: {path}")
        messagebox.showinfo("Export", f"Figure saved to:\n{path}")

    # ------------------------------------------------------------------
    # Project change hook
    # ------------------------------------------------------------------

    def on_project_changed(self):
        """Called when the project folder changes."""
        self._scan_runs()
        # Auto-fill key file if present
        folder = self.app.current_project_folder.get()
        if folder:
            for pattern in ['*.csv', '*.xlsx']:
                for f in glob.glob(os.path.join(folder, pattern)):
                    base = os.path.basename(f).lower()
                    if 'key' in base:
                        self._key_file_var.set(f)
                        break
            # Auto-fill results folder (legacy supervised mode)
            results_dir = os.path.join(folder, 'results')
            if os.path.isdir(results_dir) and hasattr(self, '_results_var'):
                self._results_var.set(results_dir)
        # Refresh supervised session list
        self._scan_trans_sessions(silent=True)
