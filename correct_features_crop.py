#!/usr/bin/env python3
"""
Feature Crop Correction Tool
Applies DLC crop offsets to features file and re-extracts brightness
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import pickle
import glob
from tkinter import Tk, filedialog, messagebox


def find_config_yaml(video_path):
    """Find DLC config.yaml file"""
    vdir = os.path.dirname(video_path)
    
    # Search locations
    search_paths = [
        os.path.join(vdir, 'config.yaml'),
        os.path.join(os.path.dirname(vdir), 'config.yaml'),
    ]
    
    # Check subdirectories
    for item in os.listdir(vdir):
        item_path = os.path.join(vdir, item)
        if os.path.isdir(item_path):
            search_paths.append(os.path.join(item_path, 'config.yaml'))
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    return None


def read_crop_params(config_path):
    """Read crop parameters from config.yaml"""
    x1 = y1 = x2 = y2 = None
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('x1:'):
                x1 = int(line.split(':')[1].strip())
            elif line.startswith('x2:'):
                x2 = int(line.split(':')[1].strip())
            elif line.startswith('y1:'):
                y1 = int(line.split(':')[1].strip())
            elif line.startswith('y2:'):
                y2 = int(line.split(':')[1].strip())
    
    return x1, y1, x2, y2


def apply_crop_offset(features_df, x_offset, y_offset):
    """Apply crop offset to all pose coordinates"""
    print(f"\n📍 Applying crop offset: x+={x_offset}, y+={y_offset}")
    print(f"\nAvailable columns in features file:")
    print(f"Total columns: {len(features_df.columns)}")
    
    # Show sample of column names
    all_cols = list(features_df.columns)
    print(f"\nFirst 20 columns:")
    for i, col in enumerate(all_cols[:20]):
        print(f"  {i+1}. {col}")
    
    if len(all_cols) > 20:
        print(f"  ... and {len(all_cols)-20} more")
    
    # Look for coordinate columns
    coord_cols = [c for c in all_cols if '_x' in c.lower() or '_y' in c.lower() or c.lower().endswith('x') or c.lower().endswith('y')]
    
    if coord_cols:
        print(f"\nFound {len(coord_cols)} coordinate columns:")
        for col in coord_cols[:10]:
            print(f"  - {col}")
        if len(coord_cols) > 10:
            print(f"  ... and {len(coord_cols)-10} more")
    else:
        print(f"\n⚠️  WARNING: No coordinate columns found!")
        print(f"    Features file appears to only contain calculated features,")
        print(f"    not raw pose coordinates.")
        print(f"\n    This means:")
        print(f"    1. Coordinate offset cannot be applied (no coordinates to correct)")
        print(f"    2. Brightness cannot be re-extracted (no coordinates available)")
        print(f"\n    You need to re-extract features from the DLC file with:")
        print(f"    - Corrected DLC coordinates (apply offset to .h5 file)")
        print(f"    OR")
        print(f"    - Extract features using full-frame coordinates from start")
    
    corrected_cols = []
    
    for col in features_df.columns:
        col_lower = col.lower()
        
        # Check if it's a pose coordinate
        if '_x' in col_lower or col_lower.endswith('x'):
            # Apply x offset
            features_df[col] = features_df[col] + x_offset
            corrected_cols.append(col)
            print(f"  ✓ Corrected {col} (x + {x_offset})")
            
        elif '_y' in col_lower or col_lower.endswith('y'):
            # Apply y offset
            features_df[col] = features_df[col] + y_offset
            corrected_cols.append(col)
            print(f"  ✓ Corrected {col} (y + {y_offset})")
    
    print(f"\n✅ Corrected {len(corrected_cols)} coordinate columns")
    
    return features_df


def recalculate_brightness(features_df, video_path, x_offset=0, y_offset=0, radius=25):
    """Re-extract brightness features with corrected coordinates - FAST single-pass method
    
    Args:
        features_df: Features dataframe
        video_path: Path to video file
        x_offset: Crop offset to add to x coordinates
        y_offset: Crop offset to add to y coordinates
        radius: Extraction radius in pixels
    """
    print(f"\n💡 Re-calculating brightness features (radius={radius}px, offset=+{x_offset},+{y_offset})...")
    print("    Using FAST single-pass extraction (like original feature extraction)")
    
    # Import tkinter for progress bar
    from tkinter import Tk, Toplevel, ttk, Label
    import time
    
    # Find brightness features
    brightness_cols = [c for c in features_df.columns 
                      if any(p in c.lower() for p in ['pix', 'pixbrt', 'brightness'])
                      and '|d/dt' not in c
                      and 'log10' not in c.lower()]  # Skip ratios, will recalculate
    
    if not brightness_cols:
        print("⚠️  No brightness features found - skipping recalculation")
        return features_df
    
    print(f"    Found {len(brightness_cols)} brightness features to recalculate")
    
    # Find bodyparts from feature names
    bodyparts_to_extract = []
    for feat in brightness_cols:
        for bp in ['hrpaw', 'hlpaw', 'frpaw', 'flpaw', 'snout', 'neck', 'tailbase', 'tailtip', 'centroid']:
            if bp in feat.lower():
                if bp not in bodyparts_to_extract:
                    bodyparts_to_extract.append(bp)
                break
    
    print(f"    Bodyparts: {bodyparts_to_extract}")
    
    # Load DLC file to get coordinates
    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    # Remove DLC suffixes
    for sfx in ['DLC', '_labeled', '_filtered']:
        if sfx in video_base:
            video_base = video_base.split(sfx)[0]
            break
    
    # Look for DLC .h5 file
    h5_patterns = [
        os.path.join(video_dir, f"{video_base}DLC*.h5"),
        os.path.join(video_dir, f"{video_base}_filtered.h5"),
        os.path.join(video_dir, f"{video_base}*.h5")
    ]
    
    dlc_file = None
    for pattern in h5_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if '_features' not in match.lower():
                dlc_file = match
                break
        if dlc_file:
            break
    
    if not dlc_file:
        print("❌ Could not find DLC .h5 file - cannot recalculate brightness")
        return features_df
    
    print(f"    DLC file: {os.path.basename(dlc_file)}")
    
    # Load DLC data
    dlc_df = pd.read_hdf(dlc_file)
    scorer = dlc_df.columns.get_level_values(0)[0]
    available_bodyparts = dlc_df[scorer].columns.get_level_values(0).unique().tolist()
    
    # Pre-extract ALL coordinates for ALL bodyparts (vectorized!)
    print(f"    Pre-loading coordinates for {len(bodyparts_to_extract)} bodyparts...")
    coords = {}
    for bp in bodyparts_to_extract:
        if bp in available_bodyparts:
            x_raw = dlc_df[scorer][bp]['x'].values.astype(np.int32)
            y_raw = dlc_df[scorer][bp]['y'].values.astype(np.int32)
            
            # Apply crop offset ONCE to entire arrays (vectorized!)
            coords[bp] = {
                'x': x_raw + x_offset,
                'y': y_raw + y_offset
            }
            print(f"      ✓ {bp}: offset applied (+{x_offset}, +{y_offset})")
        else:
            print(f"      ✗ {bp}: not in DLC file")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video")
        return features_df
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    Video: {total_frames} frames, {vid_w}x{vid_h}")
    
    # Create progress window
    progress_root = Tk()
    progress_root.withdraw()
    progress_win = Toplevel(progress_root)
    progress_win.title("Re-extracting Brightness Features")
    _sw, _sh = progress_win.winfo_screenwidth(), progress_win.winfo_screenheight()
    progress_win.geometry(f"550x140+{(_sw-550)//2}+{(_sh-140)//2}")
    
    Label(progress_win, text="Re-extracting brightness (single-pass)...", 
          font=('Arial', 11, 'bold')).pack(pady=10)
    
    progress_label = Label(progress_win, text="Starting...", font=('Arial', 9))
    progress_label.pack(pady=5)
    
    progress_bar = ttk.Progressbar(progress_win, length=400, mode='determinate')
    progress_bar.pack(pady=10)
    
    time_label = Label(progress_win, text="", font=('Arial', 9), foreground='gray')
    time_label.pack(pady=5)
    
    start_time = time.time()
    
    # Pre-allocate arrays for ALL brightness features
    brightness_data = {bp: np.ones(total_frames, dtype=np.float32) for bp in bodyparts_to_extract}
    
    # SINGLE PASS through video - extract ALL bodyparts per frame
    print(f"    Processing frames (single pass)...")
    
    for frame_idx in range(total_frames):
        # Update progress every 100 frames
        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / total_frames
            elapsed = time.time() - start_time
            
            if frame_idx > 0:
                fps = frame_idx / elapsed
                eta_seconds = (total_frames - frame_idx) / fps
                eta_min = int(eta_seconds / 60)
                eta_sec = int(eta_seconds % 60)
                time_label.config(text=f"Speed: {fps:.0f} fps | Elapsed: {int(elapsed/60)}m {int(elapsed%60)}s | ETA: {eta_min}m {eta_sec}s")
            
            progress_label.config(text=f"Frame {frame_idx:,}/{total_frames:,} ({pct:.1f}%)")
            progress_bar['value'] = pct
            progress_win.update()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale once
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Extract brightness for ALL bodyparts from this frame
        for bp in bodyparts_to_extract:
            if bp not in coords:
                continue
            
            x = coords[bp]['x'][frame_idx]
            y = coords[bp]['y'][frame_idx]
            
            # Extract region
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(vid_w, x + radius)
            y2 = min(vid_h, y + radius)
            
            roi = frame_gray[y1:y2, x1:x2]
            
            if roi.size > 0:
                brightness_data[bp][frame_idx] = np.mean(roi)
    
    cap.release()
    progress_win.destroy()
    progress_root.destroy()
    
    # Update feature columns
    print(f"\n    Updating feature columns...")
    for feat in brightness_cols:
        for bp in bodyparts_to_extract:
            if bp in feat.lower():
                features_df[feat] = brightness_data[bp]
                n_valid = np.sum(brightness_data[bp] > 1.0)
                print(f"      ✓ {feat}: {n_valid:,}/{total_frames:,} valid ({100*n_valid/total_frames:.1f}%)")
                break
    
    # Recalculate ratio features
    ratio_cols = [c for c in features_df.columns if 'log10' in c.lower() and 'pix' in c.lower() and '|d/dt' not in c]
    if ratio_cols:
        print(f"\n    Recalculating {len(ratio_cols)} ratio features...")
        for ratio_col in ratio_cols:
            # Parse ratio formula: Log10(Pix_hrpaw/Pix_hlpaw)
            if '/' in ratio_col:
                parts = ratio_col.replace('Log10(', '').replace(')', '').split('/')
                if len(parts) == 2:
                    feat1 = parts[0].strip()
                    feat2 = parts[1].strip()
                    
                    if feat1 in features_df.columns and feat2 in features_df.columns:
                        ratio = features_df[feat1] / features_df[feat2].replace(0, 1e-10)
                        features_df[ratio_col] = np.log10(ratio)
                        n_valid = np.sum(~np.isnan(features_df[ratio_col]))
                        print(f"      ✓ {ratio_col}: {n_valid:,} valid")
    
    # Recalculate derivatives
    derivative_cols = [c for c in features_df.columns if '|d/dt' in c and 'pix' in c.lower()]
    if derivative_cols:
        print(f"\n    Recalculating {len(derivative_cols)} derivative features...")
        for deriv_col in derivative_cols:
            # Find base feature name
            base_feat = deriv_col.replace('|d/dt(', '').replace(')|', '')
            
            if base_feat in features_df.columns:
                base_vals = features_df[base_feat].values
                deriv_vals = np.abs(np.diff(base_vals, prepend=np.nan))
                features_df[deriv_col] = deriv_vals
                n_valid = np.sum(~np.isnan(deriv_vals))
                print(f"      ✓ {deriv_col}: {n_valid:,} valid")
    
    total_time = time.time() - start_time
    print(f"\n✅ Brightness recalculation complete! ({int(total_time/60)}m {int(total_time%60)}s)")
    
    return features_df


def main():
    """Main function"""
    print("=" * 70)
    print("Feature Crop Correction Tool")
    print("=" * 70)
    
    # Check for batch mode
    batch_mode = '--batch' in sys.argv
    
    if batch_mode:
        run_batch_mode()
    else:
        run_single_mode()


def run_single_mode():
    """Run single file correction mode"""
    # Hide Tk root window
    root = Tk()
    root.withdraw()
    
    # Select features file
    print("\n📂 Select features file to correct...")
    features_path = filedialog.askopenfilename(
        title="Select Features File",
        filetypes=[("Pickle files", "*.pickle *.pkl"), ("All files", "*.*")]
    )
    
    if not features_path:
        print("❌ No file selected, exiting")
        return
    
    print(f"✓ Selected: {os.path.basename(features_path)}")
    
    # Process single file
    process_features_file(features_path)


def run_batch_mode():
    """Run batch correction mode for multiple files"""
    root = Tk()
    root.withdraw()
    
    print("\n" + "=" * 70)
    print("BATCH MODE - Process Multiple Features Files")
    print("=" * 70)
    
    # Select directory
    print("\n📂 Select directory containing features files...")
    directory = filedialog.askdirectory(title="Select Directory with Features Files")
    
    if not directory:
        print("❌ No directory selected, exiting")
        return
    
    # Find all features files
    features_files = []
    
    # Check main directory
    for file in os.listdir(directory):
        if file.endswith(('.pickle', '.pkl')) and '_features' in file and '_corrected' not in file:
            features_files.append(os.path.join(directory, file))
    
    # Check cache subdirectories (canonical first, then legacy)
    for cache_name in ('features', 'FeatureCache'):
        cache_subdir = os.path.join(directory, cache_name)
        if os.path.isdir(cache_subdir):
            for file in os.listdir(cache_subdir):
                if file.endswith(('.pickle', '.pkl')) and '_features' in file and '_corrected' not in file:
                    features_files.append(os.path.join(cache_subdir, file))
    
    if not features_files:
        messagebox.showwarning("No Files Found",
                             f"No features files found in:\n{directory}\n\n"
                             "Looking for files matching: *_features*.pickle or *.pkl")
        print("❌ No features files found")
        return
    
    print(f"\n✓ Found {len(features_files)} features files")
    for f in features_files:
        print(f"  - {os.path.basename(f)}")
    
    # Ask for crop parameters once
    response = messagebox.askyesno(
        "Batch Processing",
        f"Found {len(features_files)} features files.\n\n"
        "Will attempt to auto-detect crop parameters from config.yaml.\n"
        "If not found, you'll be prompted once for all files.\n\n"
        "Continue?"
    )
    
    if not response:
        print("❌ Batch processing cancelled")
        return
    
    # Try to find config.yaml
    config_path = None
    for features_file in features_files:
        config = find_config_yaml(os.path.dirname(features_file))
        if config:
            config_path = config
            break
    
    # Get crop parameters
    if config_path:
        print(f"\n✓ Found config: {config_path}")
        x1, y1, x2, y2 = read_crop_params(config_path)
        print(f"  Crop parameters: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    else:
        print("\n⚠️  No config.yaml found")
        from tkinter import simpledialog
        x1 = simpledialog.askinteger("X Offset", "Enter x1 (left crop) for ALL files:", initialvalue=0)
        y1 = simpledialog.askinteger("Y Offset", "Enter y1 (top crop) for ALL files:", initialvalue=0)
        
        if x1 is None or y1 is None:
            print("❌ Invalid input, exiting")
            return
    
    # Ask about brightness recalculation once
    recalc = messagebox.askyesno(
        "Recalculate Brightness?",
        f"Crop offset: x+{x1}, y+{y1}\n\n"
        f"Recalculate brightness for all {len(features_files)} files?\n"
        "(Recommended if brightness features are affected)\n\n"
        "This may take 5-10 minutes per file."
    )
    
    # Process each file
    print("\n" + "=" * 70)
    print("PROCESSING FILES")
    print("=" * 70)
    
    results = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for i, features_path in enumerate(features_files, 1):
        print(f"\n[{i}/{len(features_files)}] Processing: {os.path.basename(features_path)}")
        print("-" * 70)
        
        try:
            # Check if already corrected
            if '_corrected' in features_path:
                print("⏭️  Already corrected, skipping")
                results['skipped'] += 1
                continue
            
            # Process file
            success = process_features_file(
                features_path, 
                x_offset=x1, 
                y_offset=y1,
                recalc_brightness=recalc,
                batch_mode=True
            )
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results['failed'] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏭️  Skipped: {results['skipped']}")
    print(f"Total: {len(features_files)}")
    print("=" * 70)
    
    messagebox.showinfo(
        "Batch Complete!",
        f"Processed {len(features_files)} files\n\n"
        f"✓ Success: {results['success']}\n"
        f"❌ Failed: {results['failed']}\n"
        f"⏭️  Skipped: {results['skipped']}\n\n"
        "All corrected files saved with '_corrected' suffix."
    )


def process_features_file(features_path, x_offset=None, y_offset=None, recalc_brightness=None, batch_mode=False):
    """Process a single features file
    
    Args:
        features_path: Path to features file
        x_offset: X crop offset (if None, will auto-detect or prompt)
        y_offset: Y crop offset (if None, will auto-detect or prompt)
        recalc_brightness: Whether to recalculate brightness (if None, will prompt)
        batch_mode: If True, suppress some dialogs
    
    Returns:
        bool: True if successful
    """
    try:
        # Find corresponding video
        feat_dir = os.path.dirname(features_path)
        feat_base = os.path.basename(features_path)
        
        # Remove _features suffix and hash
        video_base = feat_base.split('_features')[0]
        
        # Look for video
        video_patterns = [
            os.path.join(feat_dir, f"{video_base}.mp4"),
            os.path.join(feat_dir, f"{video_base}.avi"),
            os.path.join(os.path.dirname(feat_dir), f"{video_base}.mp4"),
            os.path.join(os.path.dirname(feat_dir), f"{video_base}.avi"),
        ]
        
        video_path = None
        for pattern in video_patterns:
            if os.path.isfile(pattern):
                video_path = pattern
                break
        
        if not video_path and not batch_mode:
            root = Tk()
            root.withdraw()
            print("\n⚠️  Could not auto-detect video")
            print("📂 Select video file...")
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Videos", "*.mp4 *.avi"), ("All files", "*.*")]
            )
            
            if not video_path:
                print("❌ No video selected")
                return False
        
        if video_path:
            print(f"✓ Video: {os.path.basename(video_path)}")
        
        # Get crop parameters if not provided
        if x_offset is None or y_offset is None:
            # Find config.yaml
            print("\n🔍 Looking for DLC config.yaml...")
            config_path = find_config_yaml(video_path if video_path else feat_dir)
            
            if config_path:
                print(f"✓ Found: {config_path}")
                x1, y1, x2, y2 = read_crop_params(config_path)
                print(f"  Crop parameters: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                x_offset = x1
                y_offset = y1
            else:
                print("❌ Could not find config.yaml")
                if not batch_mode:
                    root = Tk()
                    root.withdraw()
                    from tkinter import simpledialog
                    x_offset = simpledialog.askinteger("X Offset", "Enter x1 (left crop):", initialvalue=0)
                    y_offset = simpledialog.askinteger("Y Offset", "Enter y1 (top crop):", initialvalue=0)
                else:
                    print("Using default offset (0, 0)")
                    x_offset = 0
                    y_offset = 0
        
        if x_offset == 0 and y_offset == 0 and not batch_mode:
            root = Tk()
            root.withdraw()
            response = messagebox.askyesno(
                "No Crop Offset",
                "Crop offset is 0,0 - features may already be corrected.\n\n"
                "Continue anyway?"
            )
            if not response:
                print("⏭️  Skipped")
                return False
        
        # Ask about brightness if not specified
        if recalc_brightness is None and not batch_mode:
            root = Tk()
            root.withdraw()
            recalc_brightness = messagebox.askyesno(
                "Recalculate Brightness?",
                f"Crop offset: x+{x_offset}, y+{y_offset}\n\n"
                "Do you want to recalculate brightness features?\n"
                "(This will re-extract from video - may take 5-10 minutes)\n\n"
                "Select 'No' to only correct coordinate columns."
            )
        
        if recalc_brightness is None:
            recalc_brightness = False
        
        # Load features
        print(f"\n📥 Loading features...")
        with open(features_path, 'rb') as f:
            features_df = pickle.load(f)
        
        print(f"✓ Loaded: {features_df.shape[0]} frames, {features_df.shape[1]} features")
        
        # Apply crop offset to coordinates
        features_df = apply_crop_offset(features_df, x_offset, y_offset)
        
        # Recalculate brightness if requested
        if recalc_brightness and video_path:
            features_df = recalculate_brightness(features_df, video_path, 
                                                x_offset=x_offset, y_offset=y_offset, 
                                                radius=25)
        else:
            print("\n⏭️  Skipping brightness recalculation")
        
        # Save corrected features
        output_path = features_path.replace('.pickle', '_corrected.pickle').replace('.pkl', '_corrected.pkl')
        if output_path == features_path:
            output_path = features_path.replace('.pickle', '') + '_corrected.pickle'
        
        print(f"\n💾 Saving corrected features...")
        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)
        
        print(f"✓ Saved: {os.path.basename(output_path)}")
        
        if not batch_mode:
            # Summary
            print("\n" + "="*70)
            print("✅ CORRECTION COMPLETE!")
            print("="*70)
            print(f"Input:  {os.path.basename(features_path)}")
            print(f"Output: {os.path.basename(output_path)}")
            print(f"Offset: x+{x_offset}, y+{y_offset}")
            print(f"Brightness recalculated: {'Yes' if recalc_brightness else 'No'}")
            print("\n⚠️  IMPORTANT: Use the _corrected.pickle file for training!")
            print("="*70)
            
            root = Tk()
            root.withdraw()
            messagebox.showinfo(
                "Complete!",
                f"Features corrected and saved to:\n{os.path.basename(output_path)}\n\n"
                f"Offset applied: x+{x_offset}, y+{y_offset}\n"
                f"Brightness recalculated: {'Yes' if recalc_brightness else 'No'}\n\n"
                "Use the _corrected file for training!"
            )
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()
