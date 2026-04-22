"""
Pose Feature Extraction Module

Extracts kinematic and spatial features from DeepLabCut pose estimation data.

Features extracted:
- Pairwise distances between body parts
- Joint angles (3-point law-of-cosines)
- Velocities and accelerations at multiple timescales
- Distance velocities (rate of change)
- Body part visibility (in-frame probability)
- Paw height, jerk, convex hull, body elongation, bilateral asymmetry
- Rolling velocity statistics (max, std over short windows)
- Angular velocity (rate of change of joint angles)
- Signed velocity components (x and y directions separately)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import itertools

# Increment this when the feature set changes so cached files are invalidated
POSE_FEATURE_VERSION = 5


class PoseFeatureExtractor:
    """
    Extracts kinematic and spatial features from pose estimation data.
    """
    
    def __init__(self,
                 bodyparts: List[str],
                 likelihood_threshold: float = 0.8,
                 velocity_delta: int = 2,
                 contact_threshold: float = 15.0):
        """
        Initialize pose feature extractor.

        Args:
            bodyparts: List of body part names from DLC
            likelihood_threshold: Minimum confidence for including data points (default 0.8)
            velocity_delta: Time steps for middle velocity calculation (default 2)
                           Note: Velocities are always calculated for dt=1, dt=velocity_delta, and dt=10
            contact_threshold: Height (px) below which a body part is considered in contact (default 15.0)
        """
        self.bodyparts = bodyparts
        self.likelihood_threshold = likelihood_threshold
        self.velocity_delta = velocity_delta
        self.contact_threshold = contact_threshold
        
    def load_dlc_data(self, filepath: str) -> pd.DataFrame:
        """
        Load DeepLabCut H5 or CSV file.
"""
        file_extension = filepath.split('.')[-1]
        
        if file_extension == 'csv':
            # Load CSV with first 3 rows as headers
            df = pd.read_csv(filepath)
            # Combine rows 0 and 1 to create column names: bodypart_coordinate
            new_headers = df.iloc[0, 1:] + '_' + df.iloc[1, 1:]
            # Replace 'likelihood' with 'prob' for consistency
            new_headers = new_headers.str.replace('likelihood', 'prob', regex=True)
            # Drop first 2 header rows and first column (index)
            df = df.iloc[2:, 1:]
            df.columns = new_headers
            df = df.astype(float)
            
        elif file_extension == 'h5':
            # Load H5 file
            df = pd.read_hdf(filepath)
            # Flatten MultiIndex columns
            new_headers = []
            for column_name in df.columns:
                # Drop first level (scorer), join remaining with underscore
                new_header = column_name[1:]  # Skip first level
                new_header = '_'.join(new_header)  # Join bodypart_coordinate
                new_headers.append(new_header)
            df.columns = new_headers
            # Replace 'likelihood' with 'prob'
            df.columns = df.columns.str.replace('likelihood', 'prob')
            
        else:
            raise ValueError(f"Unsupported file format: {filepath}. Only .h5 and .csv are supported.")
        
        return df
    
    def get_bodypart_coords(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract x, y coordinates and probabilities for all body parts.

        Expects flattened columns: bodypart_x, bodypart_y, bodypart_prob
        
        Returns:
            Tuple of (x_coords, y_coords, probabilities)
        """
        # Extract every 3rd column: x (0, 3, 6...), y (1, 4, 7...), prob (2, 5, 8...)
        bp_xcord = df.iloc[:, ::3].reset_index(drop=True)
        bp_ycord = df.iloc[:, 1::3].reset_index(drop=True)
        bp_prob = df.iloc[:, 2::3].reset_index(drop=True)
        
        # Filter to requested bodyparts using substring matching
        if self.bodyparts:
            included_columns_x = [col for col in bp_xcord.columns if any(substr in col for substr in self.bodyparts)]
            included_columns_y = [col for col in bp_ycord.columns if any(substr in col for substr in self.bodyparts)]
            included_columns_p = [col for col in bp_prob.columns if any(substr in col for substr in self.bodyparts)]
            
            bp_xcord = bp_xcord[included_columns_x]
            bp_ycord = bp_ycord[included_columns_y]
            bp_prob = bp_prob[included_columns_p]
        
        # Validate that we got some body parts
        if bp_xcord.empty or bp_ycord.empty:
            available_bodyparts = list(set([c.split('_x')[0] for c in df.columns if '_x' in c]))
            raise ValueError(
                f"No matching body parts found in DLC file.\n"
                f"Requested: {self.bodyparts}\n"
                f"Available in file: {available_bodyparts}"
            )
        
        return bp_xcord, bp_ycord, bp_prob
    
    def calculate_distances(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Euclidean distances between all pairs of body parts.

        Naming convention: Dis_bp1-bp2
        """
        new_columns = []
        
        for i in range(len(bp_xcord.columns)):
            for j in range(i + 1, len(bp_xcord.columns)):
                distances = np.sqrt(
                    (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, j]) ** 2 + 
                    (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, j]) ** 2
                )
                
                bp1name = bp_xcord.columns[i].replace("_x", "")
                bp2name = bp_xcord.columns[j].replace("_x", "")
                column_name = f"Dis_{bp1name}-{bp2name}"
                
                new_columns.append(pd.DataFrame({column_name: distances}))
        
        if not new_columns:
            # No distances to calculate (need at least 2 body parts)
            return pd.DataFrame()
        
        BP_distances = pd.concat(new_columns, axis=1)
        return BP_distances
    
    def calculate_angles(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate angles at joints using law of cosines.

        For three points (i, k, j), calculates angle at point k.
        Naming convention: Ang_bp1-bp2-bp3 (angle at bp2)
        """
        if len(bp_xcord.columns) < 3:
            # Need at least 3 body parts to calculate angles
            return pd.DataFrame()
        
        angle_cols = {}
        bp_columns = bp_xcord.columns

        # Get unique permutations (avoiding duplicates)
        permutations = list(itertools.permutations(range(len(bp_columns)), 2))
        unique_permutations = []
        for perm in permutations:
            reverse_perm = perm[::-1]
            if perm not in unique_permutations and reverse_perm not in unique_permutations:
                unique_permutations.append(perm)

        # Calculate angles for all combinations
        for i_p in range(len(unique_permutations)):
            for i_bp in range(len(bp_columns)):
                i, j = unique_permutations[i_p]
                k = i_bp

                if (i != k) and (j != k):  # Don't check angle from a point to itself
                    bp1name = bp_columns[i].replace("_x", "")
                    bp2name = bp_columns[j].replace("_x", "")
                    bp3name = bp_columns[k].replace("_x", "")

                    # Law of cosines: angle at point k
                    AC = (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, k])**2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, k])**2
                    BC = (bp_xcord.iloc[:, j] - bp_xcord.iloc[:, k])**2 + (bp_ycord.iloc[:, j] - bp_ycord.iloc[:, k])**2
                    AB = (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, j])**2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, j])**2

                    AC = np.sqrt(AC)
                    BC = np.sqrt(BC)
                    AB = np.sqrt(AB)

                    # Angle in degrees (safe: handle zero denominators and clamp for arccos)
                    denom = 2 * AC * BC
                    denom = denom.replace(0, np.nan)
                    cos_val = ((BC**2 + AC**2 - AB**2) / denom).clip(-1, 1)
                    AngleC = np.rad2deg(np.arccos(cos_val))

                    angle_column = f'Ang_{bp1name}-{bp3name}-{bp2name}'
                    angle_cols[angle_column] = AngleC.values

        return pd.DataFrame(angle_cols, index=bp_xcord.index) if angle_cols else pd.DataFrame()
    
    def calculate_velocities(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """
        Calculate velocity of each body part.

        Args:
            bp_xcord: X coordinates
            bp_ycord: Y coordinates
            t: Time steps for velocity calculation
            
        Returns:
            DataFrame with velocity features
        """
        vel_cols = {}

        for i in range(len(bp_xcord.columns)):
            diff_distances_x = bp_xcord.iloc[:, i].diff(periods=t)
            diff_distances_y = bp_ycord.iloc[:, i].diff(periods=t)
            distance = diff_distances_x ** 2 + diff_distances_y ** 2
            velocity = np.sqrt(distance) / np.abs(t)
            col_name = bp_xcord.columns[i].replace("_x", "") + f'_Vel{t}'
            vel_cols[col_name] = velocity.values

        BP_velocity = pd.DataFrame(vel_cols, index=bp_xcord.index)

        # Fill missing values and set velocities for first/last time points to zero
        BP_velocity.fillna(0, inplace=True)
        if t > 0:
            BP_velocity.iloc[:t, :] = 0
        elif t < 0:
            BP_velocity.iloc[t:, :] = 0

        return BP_velocity
    
    def calculate_distance_velocities(self, bp_dist: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """
        Calculate rate of change of distances.
"""
        bp_dist_vel = bp_dist.diff(periods=t).fillna(0)
        bp_dist_vel.columns = [col + "_Vel" + str(t) for col in bp_dist_vel.columns]
        return bp_dist_vel
    
    def calculate_in_frame_probability(self, bp_prob: pd.DataFrame, prob_thresh: float = 0.8) -> pd.DataFrame:
        """
        Calculate binary features indicating if body part is visible in frame.

        Args:
            bp_prob: Probability dataframe
            prob_thresh: Threshold for considering body part visible
            
        Returns:
            Binary dataframe (1 = visible, 0 = not visible)
        """
        bp_inFrame = (bp_prob >= prob_thresh).astype(int)
        bp_inFrame.columns = [col.replace('_prob', '_inFrame_p' + str(prob_thresh)) 
                             for col in bp_prob.columns]
        return bp_inFrame
    
    def calculate_paw_height(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                             window: int = 500) -> pd.DataFrame:
        """
        Estimate height of each body part above the floor.

        Image y-axis increases downward, so the floor corresponds to the
        rolling maximum of y.  Height > 0 when a paw is lifted.

        Naming convention: bpname_Height
        """
        result_cols = []
        for col in bp_ycord.columns:
            bp_name = col.replace('_y', '')
            floor = bp_ycord[col].rolling(window, min_periods=1).max()
            height = (floor - bp_ycord[col]).clip(lower=0)
            height.name = f'{bp_name}_Height'
            result_cols.append(height)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_acceleration(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                               t: int = 1) -> pd.DataFrame:
        """
        Calculate magnitude of acceleration for each body part.

        Acceleration = second difference of position / t.
        Naming convention: bpname_Accel{t}
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(t)
            vy = bp_ycord.iloc[:, i].diff(t)
            ax = vx.diff(t)
            ay = vy.diff(t)
            accel = np.sqrt(ax ** 2 + ay ** 2) / t
            accel.name = f'{bp_name}_Accel{t}'
            result_cols.append(accel)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_convex_hull_area(self, bp_xcord: pd.DataFrame,
                                   bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Per-frame area of the convex hull formed by all tracked body parts.

        Returns a single column 'hull_area'.  Frames with fewer than 3 valid
        points receive area = 0.
        """
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            return pd.DataFrame()

        n_frames = len(bp_xcord)
        areas = np.zeros(n_frames)
        for i in range(n_frames):
            xs = bp_xcord.iloc[i].values.astype(float)
            ys = bp_ycord.iloc[i].values.astype(float)
            mask = ~(np.isnan(xs) | np.isnan(ys))
            pts = np.column_stack([xs[mask], ys[mask]])
            if pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    areas[i] = hull.volume  # In 2-D, .volume is the area
                except Exception:
                    areas[i] = 0
        return pd.DataFrame({'hull_area': areas})

    def calculate_body_elongation(self, bp_xcord: pd.DataFrame,
                                  bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Per-frame aspect ratio (width / height) of the bounding box of all
        tracked body parts.

        Returns a single column 'body_elongation'.  Degenerate frames (zero
        height) get elongation = 1.
        """
        w = bp_xcord.max(axis=1) - bp_xcord.min(axis=1)
        h = bp_ycord.max(axis=1) - bp_ycord.min(axis=1)
        elongation = (w / h.replace(0, np.nan)).fillna(0)
        elongation.name = 'body_elongation'
        return elongation.to_frame()

    def calculate_bilateral_asymmetry(self, bp_xcord: pd.DataFrame,
                                      bp_ycord: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Per-frame distance asymmetry between paired left/right body parts.

        Auto-detects pairs by common prefix conventions:
          hl/hr, fl/fr, left/right, l/r

        Returns two columns per pair: {left}-{right}_AsymX and _AsymY.
        Returns None if no pairs are detected.
        """
        bp_names = [col.replace('_x', '') for col in bp_xcord.columns]
        bp_name_set = set(bp_names)

        # Ordered from most-specific to least-specific to avoid false matches
        LEFT_RIGHT_PATTERNS = [
            ('left', 'right'),
            ('Left', 'Right'),
            ('hl', 'hr'),
            ('fl', 'fr'),
            ('l', 'r'),
        ]

        pairs = []
        seen: set = set()
        for bp in bp_names:
            if bp in seen:
                continue
            for left_pat, right_pat in LEFT_RIGHT_PATTERNS:
                if bp.startswith(left_pat):
                    candidate = right_pat + bp[len(left_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((bp, candidate))
                        seen.add(bp)
                        seen.add(candidate)
                        break
                elif bp.startswith(right_pat):
                    candidate = left_pat + bp[len(right_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((candidate, bp))  # store as (left, right)
                        seen.add(bp)
                        seen.add(candidate)
                        break

        if not pairs:
            return None

        result_dfs = []
        for left_bp, right_bp in pairs:
            left_x = bp_xcord[f'{left_bp}_x'].values
            left_y = bp_ycord[f'{left_bp}_y'].values
            right_x = bp_xcord[f'{right_bp}_x'].values
            right_y = bp_ycord[f'{right_bp}_y'].values
            col_prefix = f'{left_bp}-{right_bp}'
            result_dfs.append(pd.DataFrame({
                f'{col_prefix}_AsymX': np.abs(left_x - right_x),
                f'{col_prefix}_AsymY': np.abs(left_y - right_y),
            }))

        return pd.concat(result_dfs, axis=1).reset_index(drop=True)

    def calculate_jerk(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                       t: int = 1) -> pd.DataFrame:
        """
        Calculate jerk (third derivative of position) for each body part.

        Captures the explosive onset of brief high-speed behaviors like flinches.
        Naming convention: bpname_Jerk{t}
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(t)
            vy = bp_ycord.iloc[:, i].diff(t)
            jx = vx.diff(t).diff(t)
            jy = vy.diff(t).diff(t)
            jerk = np.sqrt(jx ** 2 + jy ** 2) / t
            jerk.name = f'{bp_name}_Jerk{t}'
            result_cols.append(jerk)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_rolling_velocity_stats(self, bp_xcord: pd.DataFrame,
                                         bp_ycord: pd.DataFrame,
                                         windows: tuple = (5, 10)) -> pd.DataFrame:
        """
        Rolling max and std of velocity over short windows.

        Captures the peak velocity even when the classifier frame is slightly
        before/after the motion peak. Critical for brief bouts.
        Naming convention: bpname_Vel1_VelMaxW{w}, bpname_Vel1_VelStdW{w}
        """
        vel = self.calculate_velocities(bp_xcord, bp_ycord, t=1)
        result_cols = []
        for w in windows:
            roll_max = vel.rolling(w, center=True, min_periods=1).max()
            roll_std = vel.rolling(w, center=True, min_periods=1).std().fillna(0)
            roll_max.columns = [f'{c}_VelMaxW{w}' for c in vel.columns]
            roll_std.columns = [f'{c}_VelStdW{w}' for c in vel.columns]
            result_cols.extend([roll_max, roll_std])
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_velocity_components(self, bp_xcord: pd.DataFrame,
                                      bp_ycord: pd.DataFrame,
                                      t: int = 1) -> pd.DataFrame:
        """
        Signed x and y velocity components for each body part.

        Directional withdrawal has a specific sign pattern not captured by
        speed magnitude alone.
        Naming convention: bpname_Vx{t}, bpname_Vy{t}
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(t).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(t).fillna(0)
            vx.name = f'{bp_name}_Vx{t}'
            vy.name = f'{bp_name}_Vy{t}'
            result_cols.extend([vx, vy])
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    # ------------------------------------------------------------------
    # Flinch-discriminative features (v4)
    # ------------------------------------------------------------------

    def calculate_velocity_asymmetry(self, bp_xcord: pd.DataFrame,
                                     bp_ycord: pd.DataFrame,
                                     window: int = 30) -> pd.DataFrame:
        """Ratio of peak upward velocity to mean downward velocity.

        Flinches have fast-up / slow-down; stepping is symmetric.

        Returns per body part:
          {bp}_VelAsymmetry  — directional (Vy positive vs negative)
          {bp}_RiseFallRatio — direction-agnostic speed peak/mean ratio
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            speed = np.sqrt(bp_xcord.iloc[:, i].diff(1).fillna(0) ** 2 + vy ** 2)

            # Directional asymmetry: max(positive Vy) / mean(|negative Vy|)
            pos_vy = vy.clip(lower=0)
            neg_vy_abs = (-vy).clip(lower=0)
            roll_max_pos = pos_vy.rolling(window, min_periods=1).max()
            roll_mean_neg = neg_vy_abs.rolling(window, min_periods=1).mean()
            asym = roll_max_pos / (roll_mean_neg + 1e-6)
            asym.name = f'{bp_name}_VelAsymmetry'
            result_cols.append(asym)

            # Direction-agnostic: rolling max / rolling mean of speed
            roll_max_spd = speed.rolling(window, min_periods=1).max()
            roll_mean_spd = speed.rolling(window, min_periods=1).mean()
            rise_fall = roll_max_spd / (roll_mean_spd + 1e-6)
            rise_fall.name = f'{bp_name}_RiseFallRatio'
            result_cols.append(rise_fall)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_peak_jerk(self, bp_xcord: pd.DataFrame,
                            bp_ycord: pd.DataFrame,
                            peak_window: int = 5,
                            baseline_window: int = 30) -> pd.DataFrame:
        """Peak jerk magnitude and peak-to-baseline ratio.

        Flinch onset has much higher peak-to-baseline jerk than voluntary
        movements.
        """
        jerk_df = self.calculate_jerk(bp_xcord, bp_ycord, t=1)
        result_cols = []
        for col in jerk_df.columns:
            bp_name = col.replace('_Jerk1', '')
            jerk_series = jerk_df[col]
            peak = jerk_series.rolling(peak_window, center=True, min_periods=1).max()
            peak.name = f'{bp_name}_JerkPeakW{peak_window}'
            result_cols.append(peak)

            baseline = jerk_series.rolling(baseline_window, min_periods=1).median()
            ratio = peak / (baseline + 1e-6)
            ratio.name = f'{bp_name}_JerkPeakRatio'
            result_cols.append(ratio)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_height_velocity(self, bp_ycord: pd.DataFrame,
                                  window: int = 500) -> pd.DataFrame:
        """First derivative of paw height at dt=1 and dt=2.

        Captures the *rate* of height change — flinches produce sharp
        height-velocity spikes that static paw height misses.
        """
        height_df = self.calculate_paw_height(
            pd.DataFrame(),  # bp_xcord not used by paw_height
            bp_ycord, window=window)
        if height_df.empty:
            return pd.DataFrame()

        result_cols = []
        for col in height_df.columns:
            bp_name = col.replace('_Height', '')
            for dt in (1, 2):
                hv = height_df[col].diff(dt).fillna(0)
                hv.name = f'{bp_name}_HeightVel{dt}'
                result_cols.append(hv)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_vy_dominance(self, bp_xcord: pd.DataFrame,
                               bp_ycord: pd.DataFrame,
                               t: int = 1) -> pd.DataFrame:
        """Fraction of motion that is vertical: |Vy| / (|Vx| + |Vy| + eps).

        Flinches are primarily vertical; grooming/locomotion have horizontal
        components.
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(t).fillna(0).abs()
            vy = bp_ycord.iloc[:, i].diff(t).fillna(0).abs()
            dom = vy / (vx + vy + 1e-6)
            dom.name = f'{bp_name}_VyDominance'
            result_cols.append(dom)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_acceleration_std(self, bp_xcord: pd.DataFrame,
                                   bp_ycord: pd.DataFrame,
                                   t: int = 1,
                                   window: int = 5) -> pd.DataFrame:
        """Rolling std of acceleration over a short window.

        Flinches produce sharp acceleration spikes = high local std.
        """
        accel_df = self.calculate_acceleration(bp_xcord, bp_ycord, t=t)
        if accel_df.empty:
            return pd.DataFrame()

        result_cols = []
        for col in accel_df.columns:
            bp_name = col.replace(f'_Accel{t}', '')
            astd = accel_df[col].rolling(window, center=True, min_periods=1).std().fillna(0)
            astd.name = f'{bp_name}_AccelStdW{window}'
            result_cols.append(astd)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_contralateral_velocity_corr(self, bp_xcord: pd.DataFrame,
                                              bp_ycord: pd.DataFrame,
                                              window: int = 15) -> pd.DataFrame:
        """Rolling Pearson correlation of left vs right paw velocity.

        Low correlation = unilateral movement (flinch); high = bilateral
        (walking).  Reuses the LEFT_RIGHT_PATTERNS from bilateral asymmetry.
        """
        bp_names = [col.replace('_x', '') for col in bp_xcord.columns]
        bp_name_set = set(bp_names)

        LEFT_RIGHT_PATTERNS = [
            ('left', 'right'), ('Left', 'Right'),
            ('hl', 'hr'), ('fl', 'fr'), ('l', 'r'),
        ]

        pairs = []
        seen: set = set()
        for bp in bp_names:
            if bp in seen:
                continue
            for left_pat, right_pat in LEFT_RIGHT_PATTERNS:
                if bp.startswith(left_pat):
                    candidate = right_pat + bp[len(left_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((bp, candidate))
                        seen.update({bp, candidate})
                        break
                elif bp.startswith(right_pat):
                    candidate = left_pat + bp[len(right_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((candidate, bp))
                        seen.update({bp, candidate})
                        break

        if not pairs:
            return pd.DataFrame()

        # Compute velocity magnitude for each body part
        vel_map = {}
        for i, col in enumerate(bp_xcord.columns):
            bp_name = col.replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(1).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            vel_map[bp_name] = np.sqrt(vx ** 2 + vy ** 2)

        result_cols = []
        for left_bp, right_bp in pairs:
            left_vel = vel_map.get(left_bp)
            right_vel = vel_map.get(right_bp)
            if left_vel is None or right_vel is None:
                continue
            corr = left_vel.rolling(window, min_periods=3).corr(right_vel).fillna(0)
            corr.name = f'{left_bp}-{right_bp}_VelCorr'
            result_cols.append(corr)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    # ------------------------------------------------------------------
    # Temporal-context features (v5) — general-purpose
    # ------------------------------------------------------------------
    # These plug gaps in v4: "what was the paw doing just *before* the spike",
    # "is motion upward or downward", "how sharp is the onset", and
    # "is there high-frequency wiggle energy".  Despite being motivated by
    # flinch detection, they're useful for any brief/directional event.

    def calculate_pre_event_quiescence(self, bp_xcord: pd.DataFrame,
                                       bp_ycord: pd.DataFrame,
                                       lookback: int = 20,
                                       spike_window: int = 5) -> pd.DataFrame:
        """Inverse rolling variance of position in the window *preceding* each frame.

        Captures "explosive from stillness" — the defining flinch signature
        (nothing in v4 represents this).  High values = paw was still in the
        `lookback` frames leading up to `spike_window` frames before now.

        Returns one column per body part: `{bp}_PreQuiescence`.
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            pos_var = (bp_xcord.iloc[:, i].rolling(lookback, min_periods=3).var()
                       + bp_ycord.iloc[:, i].rolling(lookback, min_periods=3).var())
            # Shift so variance is measured over [t - (lookback+spike_window), t - spike_window]
            pre_var = pos_var.shift(spike_window)
            # Reciprocal so stillness → HIGH value (better for tree splits targeting flinches)
            pre_q = 1.0 / (pre_var + 1e-3)
            pre_q.name = f'{bp_name}_PreQuiescence'
            result_cols.append(pre_q)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_signed_jerk_y(self, bp_xcord: pd.DataFrame,
                                bp_ycord: pd.DataFrame,
                                t: int = 1) -> pd.DataFrame:
        """Signed vertical jerk — sign preserved (unlike `calculate_jerk` magnitude).

        Image y grows downward, so we negate so that POSITIVE = upward motion
        (the flinch direction).  Tree splits can discriminate directional
        events that the magnitude jerk blurs together.

        Returns one column per body part: `{bp}_Jy_signed`.
        """
        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vy = bp_ycord.iloc[:, i].diff(t)
            ay = vy.diff(t)
            jy = -ay.diff(t)   # negate: up (decreasing y) → positive
            jy.name = f'{bp_name}_Jy_signed'
            result_cols.append(jy)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_onset_sharpness(self, bp_xcord: pd.DataFrame,
                                  bp_ycord: pd.DataFrame,
                                  peak_window: int = 5) -> pd.DataFrame:
        """Rolling ratio: peak speed / (distance of peak from window center + 1).

        Flinches peak sharply at the window center (short time-to-peak);
        locomotion has smeared peaks.  Captures the time-axis of onset that
        v4's JerkPeakRatio misses.

        Returns one column per body part: `{bp}_OnsetSharpness`.
        """
        width = 2 * peak_window + 1

        def _onset(arr):
            if arr.size < 2:
                return 0.0
            peak = arr.max()
            if peak <= 0:
                return 0.0
            offset = abs(int(np.argmax(arr)) - (arr.size // 2))
            return peak / (offset + 1.0)

        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(1).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            speed = np.sqrt(vx ** 2 + vy ** 2)
            onset = speed.rolling(width, center=True, min_periods=1).apply(_onset, raw=True)
            onset.name = f'{bp_name}_OnsetSharpness'
            result_cols.append(onset)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_hf_energy(self, bp_xcord: pd.DataFrame,
                            bp_ycord: pd.DataFrame,
                            window: int = 10) -> pd.DataFrame:
        """High-frequency energy proxy per body part.

        Uses a 2nd-order Butterworth high-pass at normalized frequency 0.5
        (~10 Hz at 20 fps, higher at higher fps) then rolling RMS over
        `window` frames.  Falls back to d²-position (second difference) when
        scipy.signal is not available.

        Captures the spectral "wiggle" signature that low-order velocity /
        accel / jerk miss.  Useful for flinches (broadband HF) and grooming
        (sustained HF) — let gain pruning decide per-behavior.

        Returns one column per body part: `{bp}_HFEnergy`.
        """
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 0.5, btype='highpass')
            _use_butter = True
        except Exception:
            _use_butter = False

        result_cols = []
        for i in range(len(bp_xcord.columns)):
            bp_name = bp_xcord.columns[i].replace('_x', '')
            x = bp_xcord.iloc[:, i].ffill().bfill().fillna(0).values
            y = bp_ycord.iloc[:, i].ffill().bfill().fillna(0).values
            if _use_butter and len(x) > 20:
                try:
                    x_hf = filtfilt(b, a, x)
                    y_hf = filtfilt(b, a, y)
                except Exception:
                    x_hf = np.diff(np.diff(x, prepend=x[0]), prepend=0)
                    y_hf = np.diff(np.diff(y, prepend=y[0]), prepend=0)
            else:
                x_hf = np.diff(np.diff(x, prepend=x[0]), prepend=0)
                y_hf = np.diff(np.diff(y, prepend=y[0]), prepend=0)
            energy = pd.Series(np.sqrt(x_hf ** 2 + y_hf ** 2))
            energy = energy.rolling(window, min_periods=1).mean()
            energy.name = f'{bp_name}_HFEnergy'
            result_cols.append(energy)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_contact_features(self, height_df: pd.DataFrame,
                                    contact_threshold: float = None,
                                    window: int = 30) -> pd.DataFrame:
        """Derive binary contact state from Height columns.

        Returns per body part:
          {bp}_ContactState      — 1 if Height < threshold, else 0
          {bp}_ContactTransition — diff of ContactState (+1 = foot strike, -1 = toe off)
          {bp}_DutyCycle         — rolling mean of ContactState over *window* frames
        Plus one global column:
          N_InContact            — sum of ContactState across all body parts
        """
        if contact_threshold is None:
            contact_threshold = self.contact_threshold
        height_cols = [c for c in height_df.columns if c.endswith('_Height')]
        if not height_cols:
            return pd.DataFrame()

        result_cols = []
        contact_states = []
        for col in height_cols:
            bp_name = col.replace('_Height', '')
            state = (height_df[col] < contact_threshold).astype(int)
            state.name = f'{bp_name}_ContactState'
            contact_states.append(state)
            result_cols.append(state)

            transition = state.diff().fillna(0).astype(int)
            transition.name = f'{bp_name}_ContactTransition'
            result_cols.append(transition)

            duty = state.rolling(window, min_periods=1).mean()
            duty.name = f'{bp_name}_DutyCycle'
            result_cols.append(duty)

        n_in_contact = pd.concat(contact_states, axis=1).sum(axis=1)
        n_in_contact.name = 'N_InContact'
        result_cols.append(n_in_contact)

        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_lag_features(self, feature_df: pd.DataFrame,
                               lags: tuple = (-2, -1, 1, 2),
                               top_n: int = 10,
                               shap_importance: pd.Series = None) -> pd.DataFrame:
        """Add time-shifted copies of top features.

        If shap_importance is provided, selects top_n features by importance.
        Otherwise, selects the top_n highest-variance features.
        Lags are in frames: negative = past, positive = future.
        """
        if shap_importance is not None:
            cols = shap_importance.nlargest(top_n).index.tolist()
            cols = [c for c in cols if c in feature_df.columns]
        else:
            variances = feature_df.var().nlargest(top_n)
            cols = variances.index.tolist()

        lag_dfs = []
        for lag in lags:
            shifted = feature_df[cols].shift(lag).fillna(0)
            sign = f"m{abs(lag)}" if lag < 0 else f"p{lag}"
            shifted.columns = [f"{c}_lag{sign}" for c in cols]
            lag_dfs.append(shifted)
        if not lag_dfs:
            return pd.DataFrame()
        return pd.concat(lag_dfs, axis=1)

    def normalize_egocentric(self, bp_xcord: pd.DataFrame,
                             bp_ycord: pd.DataFrame,
                             reference_bp: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Subtract a reference body part's position from all others.

        If reference_bp is None, uses the centroid of all body parts.
        Returns normalized (bp_xcord, bp_ycord) DataFrames.
        """
        if reference_bp:
            ref_x_col = f'{reference_bp}_x'
            ref_y_col = f'{reference_bp}_y'
            if ref_x_col in bp_xcord.columns and ref_y_col in bp_ycord.columns:
                ref_x = bp_xcord[ref_x_col]
                ref_y = bp_ycord[ref_y_col]
            else:
                ref_x = bp_xcord.mean(axis=1)
                ref_y = bp_ycord.mean(axis=1)
        else:
            ref_x = bp_xcord.mean(axis=1)
            ref_y = bp_ycord.mean(axis=1)

        norm_x = bp_xcord.subtract(ref_x, axis=0)
        norm_y = bp_ycord.subtract(ref_y, axis=0)
        return norm_x, norm_y

    def extract_v4_features_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v4-new flinch features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_velocity_asymmetry,
                   self.calculate_peak_jerk,
                   self.calculate_vy_dominance,
                   self.calculate_acceleration_std]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        # Height velocity
        height_vel = self.calculate_height_velocity(bp_ycord)
        if height_vel is not None and not height_vel.empty:
            feature_dfs.append(height_vel)
        # Contralateral correlation
        contra = self.calculate_contralateral_velocity_corr(bp_xcord, bp_ycord)
        if contra is not None and not contra.empty:
            feature_dfs.append(contra)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_v5_features_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v5-new temporal-context features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        Returns the four v5 columns per body part: PreQuiescence, Jy_signed,
        OnsetSharpness, HFEnergy.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_pre_event_quiescence,
                   self.calculate_signed_jerk_y,
                   self.calculate_onset_sharpness,
                   self.calculate_hf_energy]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_new_kinematics_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v3-new kinematic features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_jerk, self.calculate_velocity_components]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        rolling_stats = self.calculate_rolling_velocity_stats(bp_xcord, bp_ycord, windows=(5, 10))
        if not rolling_stats.empty:
            feature_dfs.append(rolling_stats)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_all_features(self,
                           dlc_file: str,
                           include_angles: bool = True,
                           include_velocities: bool = True,
                           include_distance_velocities: bool = True,
                           include_in_frame: bool = True,
                           include_new_pose: bool = True,
                           include_new_kinematics: bool = True,
                           include_flinch_features: bool = True,
                           include_temporal_context_features: bool = True,
                           include_lag_features: bool = False) -> pd.DataFrame:
        """
        Extract all pose features from DLC file.

        Args:
            dlc_file: Path to DLC H5 or CSV file
            include_angles: Include angle features
            include_velocities: Include velocity features
            include_distance_velocities: Include distance velocity features
            include_in_frame: Include in-frame probability features
            include_new_pose: Include the 5 new coordinate-based features
                (paw height, acceleration, convex hull area, body elongation,
                bilateral asymmetry).  Default True.
            include_new_kinematics: Include v3 kinematic features (jerk, rolling
                velocity stats, signed velocity components).  Default True.

        Returns:
            DataFrame with all pose features
        """
        # Load DLC data
        dlc_df = self.load_dlc_data(dlc_file)
        
        # Get coordinates and probabilities
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        
        # Calculate distances
        bp_distances = self.calculate_distances(bp_xcord, bp_ycord)

        # Initialize feature list - only add distances if not empty
        feature_dfs = []
        if not bp_distances.empty:
            feature_dfs.append(bp_distances)
        
        # Add velocities with multiple time deltas
        if include_velocities:
            # Velocity with dt=1
            bp_vel_1 = self.calculate_velocities(bp_xcord, bp_ycord, t=1)
            if not bp_vel_1.empty:
                feature_dfs.append(bp_vel_1)
                # Sum of all velocities at dt=1
                sum_vel_1 = bp_vel_1.sum(axis=1).to_frame(name='sum_Vel1')
                feature_dfs.append(sum_vel_1)
            
            # Velocity with dt=velocity_delta (usually 2)
            if self.velocity_delta != 1:
                bp_vel_delta = self.calculate_velocities(bp_xcord, bp_ycord, t=self.velocity_delta)
                if not bp_vel_delta.empty:
                    feature_dfs.append(bp_vel_delta)
                    # Sum of all velocities at dt=velocity_delta
                    sum_vel_delta = bp_vel_delta.sum(axis=1).to_frame(name=f'sum_Vel{self.velocity_delta}')
                    feature_dfs.append(sum_vel_delta)
            
            # Velocity with dt=10
            if self.velocity_delta != 10:
                bp_vel_10 = self.calculate_velocities(bp_xcord, bp_ycord, t=10)
                if not bp_vel_10.empty:
                    feature_dfs.append(bp_vel_10)
                    # Sum of all velocities at dt=10
                    sum_vel_10 = bp_vel_10.sum(axis=1).to_frame(name='sum_Vel10')
                    feature_dfs.append(sum_vel_10)
        
        # Add angles
        if include_angles:
            bp_angles = self.calculate_angles(bp_xcord, bp_ycord)
            if not bp_angles.empty:
                feature_dfs.append(bp_angles)
        
        # Add distance velocities
        if include_distance_velocities and not bp_distances.empty:
            bp_dist_vel = self.calculate_distance_velocities(bp_distances, t=self.velocity_delta)
            if not bp_dist_vel.empty:
                feature_dfs.append(bp_dist_vel)
        
        # Add in-frame features
        if include_in_frame:
            bp_in_frame = self.calculate_in_frame_probability(bp_prob, self.likelihood_threshold)
            if not bp_in_frame.empty:
                feature_dfs.append(bp_in_frame)

        # Add new coordinate-based pose features (v2)
        if include_new_pose:
            for fn in [
                self.calculate_paw_height,
                self.calculate_acceleration,
                self.calculate_convex_hull_area,
                self.calculate_body_elongation,
                self.calculate_bilateral_asymmetry,
            ]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)

        # New kinematic features (v3): jerk, rolling velocity stats, signed velocity components
        if include_new_kinematics:
            for fn in [self.calculate_jerk, self.calculate_velocity_components]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)
            rolling_stats = self.calculate_rolling_velocity_stats(bp_xcord, bp_ycord, windows=(5, 10))
            if not rolling_stats.empty:
                feature_dfs.append(rolling_stats)

        # Flinch-discriminative features (v4)
        if include_flinch_features:
            for fn in [self.calculate_velocity_asymmetry,
                       self.calculate_peak_jerk,
                       self.calculate_vy_dominance,
                       self.calculate_acceleration_std]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)
            # Height velocity (depends on paw height)
            height_vel = self.calculate_height_velocity(bp_ycord)
            if height_vel is not None and not height_vel.empty:
                feature_dfs.append(height_vel)
            # Contralateral correlation
            contra = self.calculate_contralateral_velocity_corr(bp_xcord, bp_ycord)
            if contra is not None and not contra.empty:
                feature_dfs.append(contra)

        # v5 temporal-context features — general-purpose (not flinch-only)
        if include_temporal_context_features:
            for fn in [self.calculate_pre_event_quiescence,
                       self.calculate_signed_jerk_y,
                       self.calculate_onset_sharpness,
                       self.calculate_hf_energy]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)

        # Check if we have any features
        if not feature_dfs:
            raise ValueError("No features could be extracted. Check that your DLC file has valid body part data.")

        # Combine all features
        X = pd.concat(feature_dfs, axis=1)

        # Lag/lead features (computed post-concat so variance ranking is global)
        if include_lag_features:
            lag_df = self.calculate_lag_features(X, lags=(-2, -1, 1, 2), top_n=10)
            if not lag_df.empty:
                X = pd.concat([X, lag_df], axis=1)

        # Fill NaN values
        X = X.fillna(0)

        return X


def moving_window_filter(df: pd.DataFrame, window: int, std_threshold: float) -> pd.DataFrame:
    """
    Apply moving window filter to smooth data.

    Args:
        df: DataFrame to filter
        window: Window size
        std_threshold: Standard deviation threshold
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for col in df.columns:
        rolling_mean = df[col].rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, center=True, min_periods=1).std()
        
        # Replace values that deviate significantly
        mask = np.abs(df[col] - rolling_mean) > (std_threshold * rolling_std)
        filtered_df.loc[mask, col] = rolling_mean[mask]
    
    return filtered_df


# Convenience function
def extract_pose_features(dlc_file: str,
                         bodyparts: List[str],
                         likelihood_threshold: float = 0.8,
                         velocity_delta: int = 1) -> pd.DataFrame:
    """
    Quick function to extract all pose features from a DLC file.
    
    Args:
        dlc_file: Path to DLC file
        bodyparts: List of body part names
        likelihood_threshold: Minimum confidence threshold
        velocity_delta: Time steps for velocity
        
    Returns:
        DataFrame with all pose features
    """
    extractor = PoseFeatureExtractor(bodyparts, likelihood_threshold, velocity_delta)
    return extractor.extract_all_features(dlc_file)


if __name__ == "__main__":
    print("Pose Feature Extraction Module")
    print("=" * 50)
    print("Extracts kinematic and spatial features from DLC pose data")
    print("\nFeatures:")
    print("  - Distances between body part pairs")
    print("  - Angles at joints (3-point angles)")
    print("  - Velocities of body parts")
    print("  - Distance velocities")
    print("  - Body part visibility")
