"""
Pose Feature Extraction Module
Inspired by BAREfoot's AniML_utils_PoseFeatureExtraction.py

Extracts kinematic and spatial features from DeepLabCut pose estimation data.
This implementation matches BAREfoot's approach while using original code.

Features extracted:
- Distances between body part pairs
- Angles at joints (3-point angles)
- Velocities of body parts
- Distance velocities (rate of change)
- Body part visibility (in-frame probability)

Citation: Inspired by Barkai et al. (2025), Cell Rep Methods
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import itertools

# Increment this when the feature set changes so cached files are invalidated
POSE_FEATURE_VERSION = 2


class PoseFeatureExtractor:
    """
    Extract kinematic and spatial features from pose estimation data.
    Matches BAREfoot's feature extraction approach.
    """
    
    def __init__(self, 
                 bodyparts: List[str],
                 likelihood_threshold: float = 0.8,
                 velocity_delta: int = 2):
        """
        Initialize pose feature extractor.
        
        Args:
            bodyparts: List of body part names from DLC
            likelihood_threshold: Minimum confidence for including data points (default 0.8 from BAREfoot)
            velocity_delta: Time steps for middle velocity calculation (default 2, matching BAREfoot's dt_vel)
                           Note: Velocities are always calculated for dt=1, dt=velocity_delta, and dt=10
        """
        self.bodyparts = bodyparts
        self.likelihood_threshold = likelihood_threshold
        self.velocity_delta = velocity_delta
        
    def load_dlc_data(self, filepath: str) -> pd.DataFrame:
        """
        Load DeepLabCut H5 or CSV file.
        Matches BAREfoot's open_file_as_dataframe function exactly.
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
        Matches BAREfoot's getBPcords function EXACTLY.
        
        Expects flattened columns: bodypart_x, bodypart_y, bodypart_prob
        
        Returns:
            Tuple of (x_coords, y_coords, probabilities)
        """
        # Extract every 3rd column: x (0, 3, 6...), y (1, 4, 7...), prob (2, 5, 8...)
        # This matches BAREfoot's approach exactly
        bp_xcord = df.iloc[:, ::3].reset_index(drop=True)
        bp_ycord = df.iloc[:, 1::3].reset_index(drop=True)
        bp_prob = df.iloc[:, 2::3].reset_index(drop=True)
        
        # Filter to requested bodyparts using substring matching (like BAREfoot)
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
        Matches BAREfoot's bp_distances function.
        
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
        Matches BAREfoot's angle function.
        
        For three points (i, k, j), calculates angle at point k.
        Naming convention: Ang_bp1-bp2-bp3 (angle at bp2)
        """
        if len(bp_xcord.columns) < 3:
            # Need at least 3 body parts to calculate angles
            return pd.DataFrame()
        
        BP_angles = pd.DataFrame()
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
                    
                    # Angle in degrees
                    AngleC = np.arccos((BC**2 + AC**2 - AB**2) / (2 * AC * BC))
                    AngleC = np.rad2deg(AngleC)
                    
                    angle_column = f'Ang_{bp1name}-{bp3name}-{bp2name}'
                    BP_angles = pd.concat([BP_angles, pd.DataFrame(AngleC, columns=[angle_column])], axis=1)
        
        return BP_angles
    
    def calculate_velocities(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """
        Calculate velocity of each body part.
        Matches BAREfoot's bp_velocity function.
        
        Args:
            bp_xcord: X coordinates
            bp_ycord: Y coordinates
            t: Time steps for velocity calculation
            
        Returns:
            DataFrame with velocity features
        """
        BP_velocity = pd.DataFrame()
        
        for i in range(len(bp_xcord.columns)):
            diff_distances_x = bp_xcord.iloc[:, i].diff(periods=t)
            diff_distances_y = bp_ycord.iloc[:, i].diff(periods=t)
            distance = diff_distances_x ** 2 + diff_distances_y ** 2
            velocity = np.sqrt(distance) / np.abs(t)
            velocity.name = bp_xcord.columns[i].replace("_x", "") + f'_Vel{t}'
            BP_velocity = pd.concat([BP_velocity, velocity], axis=1)
        
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
        Matches BAREfoot's distances_velocity function.
        """
        bp_dist_vel = bp_dist.diff(periods=t)
        bp_dist_vel.columns = [col + "_Vel" + str(t) for col in bp_dist_vel.columns]
        return bp_dist_vel
    
    def calculate_in_frame_probability(self, bp_prob: pd.DataFrame, prob_thresh: float = 0.8) -> pd.DataFrame:
        """
        Calculate binary features indicating if body part is visible in frame.
        Matches BAREfoot's bp_inFrame function.
        
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
        elongation = (w / h.replace(0, np.nan)).fillna(1)
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

    def extract_all_features(self,
                           dlc_file: str,
                           include_angles: bool = True,
                           include_velocities: bool = True,
                           include_distance_velocities: bool = True,
                           include_in_frame: bool = True,
                           include_new_pose: bool = True) -> pd.DataFrame:
        """
        Extract all pose features from DLC file.
        Combines all feature types like BAREfoot's feature extraction pipeline.

        Args:
            dlc_file: Path to DLC H5 or CSV file
            include_angles: Include angle features
            include_velocities: Include velocity features
            include_distance_velocities: Include distance velocity features
            include_in_frame: Include in-frame probability features
            include_new_pose: Include the 5 new coordinate-based features
                (paw height, acceleration, convex hull area, body elongation,
                bilateral asymmetry).  Default True.

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
        
        # Add velocities with multiple time deltas (matching BAREfoot exactly)
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
            
            # Velocity with dt=10 (BAREfoot standard)
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
        
        # Add distance velocities (only if we have distances) - NOT used in standard BAREfoot
        # Keeping for backward compatibility with PixelPaws-trained models
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

        # Check if we have any features
        if not feature_dfs:
            raise ValueError("No features could be extracted. Check that your DLC file has valid body part data.")

        
        # Combine all features
        X = pd.concat(feature_dfs, axis=1)
        
        # Fill NaN values
        X = X.fillna(0)
        
        return X


def moving_window_filter(df: pd.DataFrame, window: int, std_threshold: float) -> pd.DataFrame:
    """
    Apply moving window filter to smooth data.
    Matches BAREfoot's filtering approach.
    
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
