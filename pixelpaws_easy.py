"""
PixelPaws Convenience Functions
Easy-to-use wrappers for feature extraction with smart defaults
"""

import pandas as pd
from typing import List, Optional, Union
from pose_features import PoseFeatureExtractor
from brightness_features import PixelBrightnessExtractor


def extract_all_features_auto(dlc_file: str,
                              video_file: str,
                              include_brightness: bool = True,
                              **kwargs) -> pd.DataFrame:
    """
    Extract ALL features automatically - no body parts needed!
    
    This function reads the DLC file, detects all body parts,
    and extracts both pose and brightness features automatically.
    
    Perfect for beginners or quick analysis.
    
    Args:
        dlc_file: Path to DLC tracking file (.h5 or .csv)
        video_file: Path to video file (.mp4, .avi, etc.)
        include_brightness: Include brightness features (slower but better)
        **kwargs: Optional parameters (square_size, likelihood_threshold, etc.)
        
    Returns:
        DataFrame with all features
        
    Example:
        >>> # One line to extract everything!
        >>> features = extract_all_features_auto("video_DLC.h5", "video.mp4")
        >>> print(f"Extracted {features.shape[1]} features!")
    """
    # Auto-detect body parts from DLC file
    if dlc_file.endswith('.h5'):
        dlc_df = pd.read_hdf(dlc_file)
    else:
        dlc_df = pd.read_csv(dlc_file, header=[0, 1, 2], index_col=0)
    
    # Extract body part names
    if isinstance(dlc_df.columns, pd.MultiIndex):
        # Multi-index: ('bodypart', 'x'/'y'/'likelihood')
        bodyparts = list(set([col[0] for col in dlc_df.columns if col[0] != 'bodyparts']))
    else:
        # Flat columns: 'bodypart_x', 'bodypart_y', 'bodypart_likelihood'
        bodyparts = list(set([col.split('_')[0] for col in dlc_df.columns if '_x' in col]))
    
    print(f"Auto-detected body parts: {bodyparts}")
    
    # Call the full function
    return extract_all_features(
        dlc_file=dlc_file,
        video_file=video_file,
        bodyparts=bodyparts,
        include_brightness=include_brightness,
        **kwargs
    )


def extract_all_features(dlc_file: str,
                        video_file: str,
                        bodyparts: List[str],
                        bodyparts_brightness: Optional[List[str]] = None,
                        include_pose: bool = True,
                        include_brightness: bool = True,
                        likelihood_threshold: float = 0.8,
                        velocity_delta: int = 1,
                        square_size: Union[int, List[int]] = 50,
                        pixel_threshold: Optional[float] = None,
                        dt_vel: int = 2,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Extract pose and brightness features with smart defaults.
    
    This is the RECOMMENDED function for most users. Specify body parts
    and let PixelPaws handle the rest with proven defaults.
    
    Args:
        dlc_file: Path to DLC tracking file
        video_file: Path to video file
        bodyparts: List of body parts to use for pose features
        bodyparts_brightness: Body parts for brightness (None = use all bodyparts)
        include_pose: Extract pose features (distances, angles, velocities)
        include_brightness: Extract brightness features (light-based depth)
        likelihood_threshold: Min DLC confidence (0.8 = BAREfoot default)
        velocity_delta: Time steps for velocity (1 = BAREfoot default)
        square_size: ROI size in pixels (50 = BAREfoot default)
        pixel_threshold: Brightness threshold (None = auto-compute)
        dt_vel: Time steps for brightness derivatives (2 = BAREfoot default)
        verbose: Print progress messages
        
    Returns:
        DataFrame with selected features
        
    Example:
        >>> # Standard usage - specify body parts
        >>> features = extract_all_features(
        ...     dlc_file="video_DLC.h5",
        ...     video_file="video.mp4",
        ...     bodyparts=['hlpaw', 'hrpaw', 'snout', 'tailbase']
        ... )
        
        >>> # Advanced - separate brightness body parts
        >>> features = extract_all_features(
        ...     dlc_file="video_DLC.h5",
        ...     video_file="video.mp4",
        ...     bodyparts=['hlpaw', 'hrpaw', 'snout', 'tailbase'],
        ...     bodyparts_brightness=['hlpaw', 'hrpaw']  # Only paws for brightness
        ... )
        
        >>> # Pose only (faster, no video needed)
        >>> features = extract_all_features(
        ...     dlc_file="video_DLC.h5",
        ...     video_file=None,  # Not needed for pose only
        ...     bodyparts=['hlpaw', 'hrpaw', 'snout'],
        ...     include_brightness=False
        ... )
    """
    features_list = []
    
    # 1. Extract pose features
    if include_pose:
        if verbose:
            print("Extracting pose features...")
        
        pose_extractor = PoseFeatureExtractor(
            bodyparts=bodyparts,
            likelihood_threshold=likelihood_threshold,
            velocity_delta=velocity_delta
        )
        pose_features = pose_extractor.extract_all_features(dlc_file)
        features_list.append(pose_features)
        
        if verbose:
            print(f"  ✓ Extracted {pose_features.shape[1]} pose features")
    
    # 2. Extract brightness features
    if include_brightness:
        if video_file is None:
            raise ValueError("video_file required when include_brightness=True")
        
        if verbose:
            print("Extracting brightness features (this may take a few minutes)...")
        
        # Use subset of body parts if specified
        bp_brightness = bodyparts_brightness if bodyparts_brightness is not None else bodyparts
        
        brightness_extractor = PixelBrightnessExtractor(
            bodyparts_to_track=bp_brightness,
            square_size=square_size,
            pixel_threshold=pixel_threshold,
            min_prob=likelihood_threshold
        )
        brightness_features = brightness_extractor.extract_brightness_features(
            dlc_file=dlc_file,
            video_file=video_file,
            dt_vel=dt_vel,
            create_video=False
        )
        features_list.append(brightness_features)
        
        if verbose:
            print(f"  ✓ Extracted {brightness_features.shape[1]} brightness features")
    
    # 3. Combine all features
    if not features_list:
        raise ValueError("No features extracted. Enable include_pose or include_brightness")
    
    all_features = pd.concat(features_list, axis=1)
    
    if verbose:
        print(f"\n✓ Total: {all_features.shape[1]} features from {all_features.shape[0]} frames")
    
    return all_features


def extract_pose_only(dlc_file: str,
                     bodyparts: List[str],
                     **kwargs) -> pd.DataFrame:
    """
    Extract ONLY pose features (fast, no video needed).
    
    Use this when:
    - You don't have the video file
    - You want fast extraction
    - Brightness features aren't needed for your behavior
    
    Args:
        dlc_file: Path to DLC tracking file
        bodyparts: List of body parts
        **kwargs: Optional parameters (likelihood_threshold, velocity_delta)
        
    Returns:
        DataFrame with pose features only
        
    Example:
        >>> features = extract_pose_only(
        ...     dlc_file="video_DLC.h5",
        ...     bodyparts=['hlpaw', 'hrpaw', 'snout']
        ... )
    """
    return extract_all_features(
        dlc_file=dlc_file,
        video_file=None,
        bodyparts=bodyparts,
        include_brightness=False,
        **kwargs
    )


def extract_brightness_only(dlc_file: str,
                           video_file: str,
                           bodyparts: List[str],
                           **kwargs) -> pd.DataFrame:
    """
    Extract ONLY brightness features.
    
    Use this when:
    - You only need brightness/depth information
    - You already have pose features from another source
    
    Args:
        dlc_file: Path to DLC tracking file
        video_file: Path to video file
        bodyparts: List of body parts
        **kwargs: Optional parameters (square_size, pixel_threshold, etc.)
        
    Returns:
        DataFrame with brightness features only
        
    Example:
        >>> features = extract_brightness_only(
        ...     dlc_file="video_DLC.h5",
        ...     video_file="video.mp4",
        ...     bodyparts=['hlpaw', 'hrpaw']
        ... )
    """
    return extract_all_features(
        dlc_file=dlc_file,
        video_file=video_file,
        bodyparts=bodyparts,
        include_pose=False,
        include_brightness=True,
        **kwargs
    )


# Quick reference
RECOMMENDED_BODY_PARTS = {
    'mouse_pain': ['hlpaw', 'hrpaw', 'snout'],  # Typical for pain behaviors
    'mouse_full': ['hlpaw', 'hrpaw', 'flpaw', 'frpaw', 'snout', 'tailbase'],  # Comprehensive
    'rat_pain': ['hlpaw', 'hrpaw', 'snout'],
    'rat_full': ['hlpaw', 'hrpaw', 'flpaw', 'frpaw', 'snout', 'tailbase', 'neck'],
}


def get_recommended_bodyparts(species: str = 'mouse', 
                             scope: str = 'pain') -> List[str]:
    """
    Get recommended body parts for common scenarios.
    
    Args:
        species: 'mouse' or 'rat'
        scope: 'pain' (minimal) or 'full' (comprehensive)
        
    Returns:
        List of recommended body parts
        
    Example:
        >>> bodyparts = get_recommended_bodyparts('mouse', 'pain')
        >>> features = extract_all_features(
        ...     dlc_file="video_DLC.h5",
        ...     video_file="video.mp4",
        ...     bodyparts=bodyparts
        ... )
    """
    key = f"{species}_{scope}"
    if key not in RECOMMENDED_BODY_PARTS:
        available = list(RECOMMENDED_BODY_PARTS.keys())
        raise ValueError(f"Unknown combination. Available: {available}")
    
    return RECOMMENDED_BODY_PARTS[key]


if __name__ == "__main__":
    print("PixelPaws Convenience Functions")
    print("=" * 50)
    print("\nThree levels of usage:\n")
    
    print("Level 1 - Automatic (Easiest):")
    print("  features = extract_all_features_auto('video_DLC.h5', 'video.mp4')")
    print()
    
    print("Level 2 - Standard (Recommended):")
    print("  features = extract_all_features(")
    print("      dlc_file='video_DLC.h5',")
    print("      video_file='video.mp4',")
    print("      bodyparts=['hlpaw', 'hrpaw', 'snout']")
    print("  )")
    print()
    
    print("Level 3 - Advanced (Full Control):")
    print("  features = extract_all_features(")
    print("      dlc_file='video_DLC.h5',")
    print("      video_file='video.mp4',")
    print("      bodyparts=['hlpaw', 'hrpaw', 'snout'],")
    print("      bodyparts_brightness=['hlpaw', 'hrpaw'],  # Subset")
    print("      square_size=60,")
    print("      pixel_threshold=0.25")
    print("  )")
