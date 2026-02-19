"""
Batch Results Analysis Script
Analyzes PixelPaws batch output with subject key file
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path

def extract_subject_id(filename):
    """
    Extract 4-digit subject ID from filename.
    
    Examples:
        '260129_Formalin_2801_PixelPaws_Left_licking_predictions.csv' -> '2801'
        '260129_Formalin_3304_PixelPaws_Left_licking_bouts.csv' -> '3304'
    """
    # Method 1: Find 4-digit number after underscore before another underscore/dot
    match = re.search(r'_(\d{4})(?:_|\.)', filename)
    if match:
        return match.group(1)
    
    # Method 2: Find any 4-digit number in specific position
    # Pattern: DATE_EXPERIMENT_SUBJECTID_...
    parts = filename.split('_')
    for part in parts:
        if len(part) == 4 and part.isdigit():
            return part
    
    return None


def load_key_file(key_path):
    """Load subject key file (Excel)"""
    df = pd.read_excel(key_path)
    # Ensure Subject column is string for matching
    df['Subject'] = df['Subject'].astype(str)
    return df


def analyze_predictions(results_folder, key_file_path, fps=60):
    """
    Analyze all prediction CSV files in Results folder
    
    Args:
        results_folder: Path to Results/ folder
        key_file_path: Path to Excel key file
        fps: Video frame rate (default 60)
    """
    print("="*80)
    print("BATCH RESULTS ANALYSIS")
    print("="*80)
    
    # Load key file
    print(f"\n📁 Loading key file: {key_file_path}")
    key_df = load_key_file(key_file_path)
    print(f"   Found {len(key_df)} subjects in key file")
    print(f"   Subjects: {sorted(key_df['Subject'].tolist())}")
    
    # Find all prediction files
    pred_pattern = os.path.join(results_folder, "*_predictions.csv")
    pred_files = glob.glob(pred_pattern)
    
    print(f"\n📊 Found {len(pred_files)} prediction files")
    
    if not pred_files:
        print(f"   ⚠️  No files matching pattern: {pred_pattern}")
        return None
    
    # Process each file
    results = []
    
    for pred_file in sorted(pred_files):
        filename = os.path.basename(pred_file)
        
        # Extract subject ID
        subject_id = extract_subject_id(filename)
        
        if not subject_id:
            print(f"   ⚠️  Could not extract subject ID from: {filename}")
            continue
        
        # Check if subject in key file
        if subject_id not in key_df['Subject'].values:
            print(f"   ⚠️  Subject {subject_id} not in key file (from {filename})")
            continue
        
        # Get treatment info
        subject_info = key_df[key_df['Subject'] == subject_id].iloc[0]
        treatment = subject_info['Treatment']
        
        # Load predictions
        try:
            pred_df = pd.read_csv(pred_file)
            
            # Extract behavior name (last column that isn't standard columns)
            standard_cols = ['frame', 'probability', 'prediction_raw', 'prediction_filtered']
            behavior_cols = [col for col in pred_df.columns if col not in standard_cols]
            
            if behavior_cols:
                behavior_name = behavior_cols[0]
                predictions = pred_df[behavior_name].values
            else:
                predictions = pred_df['prediction_filtered'].values
                behavior_name = 'behavior'
            
            # Calculate metrics
            n_frames = len(predictions)
            n_positive = np.sum(predictions)
            percent_behavior = (n_positive / n_frames) * 100 if n_frames > 0 else 0
            duration_sec = n_frames / fps
            behavior_sec = n_positive / fps
            
            # Try to load bouts file
            bouts_file = pred_file.replace('_predictions.csv', '_bouts.csv')
            n_bouts = 0
            mean_bout_duration_sec = 0
            
            if os.path.exists(bouts_file):
                try:
                    bouts_df = pd.read_csv(bouts_file)
                    n_bouts = len(bouts_df)
                    if n_bouts > 0:
                        mean_bout_duration_sec = bouts_df['duration'].mean() / fps
                except:
                    pass
            
            # Store results
            results.append({
                'subject_id': subject_id,
                'treatment': treatment,
                'filename': filename,
                'behavior': behavior_name,
                'total_frames': n_frames,
                'total_duration_sec': duration_sec,
                'behavior_frames': n_positive,
                'behavior_duration_sec': behavior_sec,
                'percent_behavior': percent_behavior,
                'n_bouts': n_bouts,
                'mean_bout_duration_sec': mean_bout_duration_sec
            })
            
            print(f"   ✓ {subject_id} ({treatment}): {percent_behavior:.1f}% behavior, {n_bouts} bouts")
            
        except Exception as e:
            print(f"   ✗ Error processing {filename}: {e}")
            continue
    
    # Create results DataFrame
    if not results:
        print("\n⚠️  No valid results found!")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Summary by treatment
    print("\n" + "="*80)
    print("SUMMARY BY TREATMENT")
    print("="*80)
    
    for treatment in results_df['treatment'].unique():
        subset = results_df[results_df['treatment'] == treatment]
        
        print(f"\n{treatment}:")
        print(f"  N: {len(subset)}")
        print(f"  Behavior %: {subset['percent_behavior'].mean():.2f} ± {subset['percent_behavior'].std():.2f}")
        print(f"  Bouts: {subset['n_bouts'].mean():.1f} ± {subset['n_bouts'].std():.1f}")
        print(f"  Mean bout duration: {subset['mean_bout_duration_sec'].mean():.2f}s ± {subset['mean_bout_duration_sec'].std():.2f}s")
    
    return results_df


def main():
    """
    Main analysis function
    
    Update these paths to match your setup:
    """
    # CONFIGURE THESE PATHS:
    results_folder = "Results"  # Or full path: "E:/RSVIDS/Blackbox/2601_JDR_videos/Results"
    key_file = "Formalin_test-Mouse_Key.xlsx"
    fps = 60  # Your video frame rate
    
    # Run analysis
    results_df = analyze_predictions(results_folder, key_file, fps)
    
    if results_df is not None:
        # Save detailed results
        output_file = os.path.join(results_folder, "Analysis_Summary.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved detailed results to: {output_file}")
        
        # Save treatment summary
        summary = results_df.groupby('treatment').agg({
            'subject_id': 'count',
            'percent_behavior': ['mean', 'std'],
            'n_bouts': ['mean', 'std'],
            'mean_bout_duration_sec': ['mean', 'std']
        }).round(2)
        
        summary_file = os.path.join(results_folder, "Treatment_Summary.csv")
        summary.to_csv(summary_file)
        print(f"✓ Saved treatment summary to: {summary_file}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)


if __name__ == "__main__":
    main()
