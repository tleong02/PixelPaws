"""
Optical Flow Feature Extraction Module

Extracts sparse Lucas-Kanade optical flow features at DeepLabCut body-part
locations.  This is faster than dense Farneback flow because only a handful
of seed points are tracked per frame.

Output columns per body part:
    bpname_FlowMag  — magnitude of the optical-flow displacement vector
    bpname_FlowX    — horizontal component (positive = rightward)
    bpname_FlowY    — vertical component (positive = downward)

Frames where the DLC confidence is below min_prob, or where tracking failed,
receive zero-filled values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


class OpticalFlowExtractor:
    """
    Extract sparse optical-flow features at DLC body-part locations.

    Parameters
    ----------
    bodyparts : list[str]
        Body-part names to track (substring matching, same convention as
        PoseFeatureExtractor).
    min_prob : float
        Minimum DLC confidence to consider a point valid (default 0.6).
    lk_params : dict | None
        Parameters forwarded to ``cv2.calcOpticalFlowPyrLK``.  If None, a
        sensible default is used.
    """

    _DEFAULT_LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(3, 10, 0.03),  # cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    )

    def __init__(
        self,
        bodyparts: List[str],
        min_prob: float = 0.6,
        lk_params: Optional[dict] = None,
    ):
        self.bodyparts = bodyparts
        self.min_prob = min_prob
        self.lk_params = lk_params if lk_params is not None else self._DEFAULT_LK_PARAMS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preload(self, dlc_file: str) -> 'OpticalFlowExtractor':
        """
        Load DLC coordinates ahead of time so this extractor can be used
        inside another extractor's frame loop (single-pass extraction).

        Call this before passing the extractor to
        ``PixelBrightnessExtractor.extract_brightness_features()``.
        """
        self._coords, self._probs, self._bp_names = self._load_dlc_coords(dlc_file)
        return self

    def compute_flow_for_frame(self, frame_idx: int,
                               prev_gray: np.ndarray,
                               gray: np.ndarray) -> dict:
        """
        Compute sparse optical-flow for one frame transition.

        Both arrays must be uint8 grayscale (as returned by
        ``cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)``).

        Returns
        -------
        dict mapping body-part name → {'mag': float, 'x': float, 'y': float}.
        All tracked body parts are always present; zero-filled when confidence
        is low or tracking fails.  Returns ``{}`` if ``preload()`` was not called.
        """
        try:
            import cv2
        except ImportError:
            return {}

        if frame_idx == 0 or not hasattr(self, '_coords') or not self._bp_names:
            return {}

        prev_idx = frame_idx - 1
        result = {}
        for bp in self._bp_names:
            p = float(self._probs[bp][prev_idx]) if prev_idx < len(self._probs[bp]) else 0.0
            if p < self.min_prob:
                result[bp] = {'mag': 0.0, 'x': 0.0, 'y': 0.0}
                continue
            if prev_idx >= len(self._coords[bp]['x']) or prev_idx >= len(self._coords[bp]['y']):
                result[bp] = {'mag': 0.0, 'x': 0.0, 'y': 0.0}
                continue
            x0 = float(self._coords[bp]['x'][prev_idx])
            y0 = float(self._coords[bp]['y'][prev_idx])
            if np.isnan(x0) or np.isnan(y0):
                result[bp] = {'mag': 0.0, 'x': 0.0, 'y': 0.0}
                continue
            pt = np.array([[[x0, y0]]], dtype=np.float32)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pt, None, **self.lk_params)
            if status is not None and status[0, 0] == 1:
                dx = float(next_pts[0, 0, 0]) - x0
                dy = float(next_pts[0, 0, 1]) - y0
                result[bp] = {
                    'mag': float(np.sqrt(dx * dx + dy * dy)),
                    'x': dx,
                    'y': dy,
                }
            else:
                result[bp] = {'mag': 0.0, 'x': 0.0, 'y': 0.0}
        return result

    def extract_features(self, dlc_file: str, video_file: str) -> pd.DataFrame:
        """
        Compute per-frame optical-flow features.

        Parameters
        ----------
        dlc_file : str
            Path to DLC tracking file (.h5 or .csv).
        video_file : str
            Path to the corresponding video file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``bpname_FlowMag``, ``bpname_FlowX``,
            ``bpname_FlowY`` for every requested body part, one row per frame.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for optical flow extraction.  "
                "Install with:  pip install opencv-python"
            ) from exc

        # ── Load DLC coordinates ────────────────────────────────────────────
        coords, probs, bp_names = self._load_dlc_coords(dlc_file)
        n_frames_dlc = len(coords[bp_names[0]]['x']) if bp_names else 0

        # ── Open video ──────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = min(n_frames_dlc, n_frames_video) if n_frames_dlc > 0 else n_frames_video

        # ── Pre-allocate output arrays ──────────────────────────────────────
        flow_mag  = {bp: np.zeros(n_frames) for bp in bp_names}
        flow_x    = {bp: np.zeros(n_frames) for bp in bp_names}
        flow_y    = {bp: np.zeros(n_frames) for bp in bp_names}

        # ── Iterate frames ──────────────────────────────────────────────────
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        while frame_idx < n_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and frame_idx > 0:
                prev_idx = frame_idx - 1
                for bp in bp_names:
                    p = probs[bp][prev_idx]
                    if p < self.min_prob:
                        # Low confidence — leave zeros
                        pass
                    else:
                        x0 = float(coords[bp]['x'][prev_idx])
                        y0 = float(coords[bp]['y'][prev_idx])
                        if np.isnan(x0) or np.isnan(y0):
                            pass
                        else:
                            pt = np.array([[[x0, y0]]], dtype=np.float32)
                            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                                prev_gray, gray, pt, None, **self.lk_params
                            )
                            if status is not None and status[0, 0] == 1:
                                dx = float(next_pts[0, 0, 0]) - x0
                                dy = float(next_pts[0, 0, 1]) - y0
                                flow_x[bp][frame_idx]   = dx
                                flow_y[bp][frame_idx]   = dy
                                flow_mag[bp][frame_idx] = np.sqrt(dx * dx + dy * dy)

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # ── Assemble DataFrame ──────────────────────────────────────────────
        cols = {}
        for bp in bp_names:
            cols[f'{bp}_FlowMag'] = flow_mag[bp][:frame_idx]
            cols[f'{bp}_FlowX']   = flow_x[bp][:frame_idx]
            cols[f'{bp}_FlowY']   = flow_y[bp][:frame_idx]

        return pd.DataFrame(cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dlc_coords(self, dlc_file: str):
        """
        Load x, y coordinates and probabilities from a DLC .h5 or .csv file.

        Returns
        -------
        coords : dict[str, dict]   — {'bodypart': {'x': array, 'y': array}}
        probs  : dict[str, array]  — {'bodypart': array}
        bp_names : list[str]       — body-part names that matched self.bodyparts
        """
        ext = dlc_file.rsplit('.', 1)[-1].lower()
        if ext == 'h5':
            df = pd.read_hdf(dlc_file)
            # Flatten multi-index: (scorer, bodypart, coord) → bodypart_coord
            new_cols = []
            for col in df.columns:
                parts = col[1:]          # drop scorer level
                new_cols.append('_'.join(parts))
            df.columns = new_cols
            df.columns = df.columns.str.replace('likelihood', 'prob')
        elif ext == 'csv':
            df = pd.read_csv(dlc_file)
            new_headers = df.iloc[0, 1:] + '_' + df.iloc[1, 1:]
            new_headers = new_headers.str.replace('likelihood', 'prob', regex=True)
            df = df.iloc[2:, 1:].reset_index(drop=True)
            df.columns = new_headers
            df = df.astype(float)
        else:
            raise ValueError(f"Unsupported DLC file format: {dlc_file}")

        # Find body parts matching self.bodyparts (substring matching)
        all_bp_raw = list(dict.fromkeys(
            col.rsplit('_', 1)[0]
            for col in df.columns
            if col.endswith('_x') or col.endswith('_y') or col.endswith('_prob')
        ))

        matched_bps = [
            bp for bp in all_bp_raw
            if any(substr in bp for substr in self.bodyparts)
        ]

        coords: dict = {}
        probs: dict = {}
        for bp in matched_bps:
            x_col = f'{bp}_x'
            y_col = f'{bp}_y'
            p_col = f'{bp}_prob'
            if x_col in df.columns and y_col in df.columns:
                coords[bp] = {
                    'x': df[x_col].values.astype(float),
                    'y': df[y_col].values.astype(float),
                }
                if p_col in df.columns:
                    probs[bp] = df[p_col].values.astype(float)
                else:
                    probs[bp] = np.ones(len(df))

        return coords, probs, matched_bps
