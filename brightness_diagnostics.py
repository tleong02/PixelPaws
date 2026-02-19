"""
Brightness Feature Diagnostics Tool
Analyzes the dynamic range and temporal variation of brightness features
to determine if they're informative for behavior classification.

Usage:
    python brightness_diagnostics.py

The script will prompt you to select a features file (extracted .pkl)
and generate comprehensive diagnostic plots.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def select_features_file():
    """Open file dialog to select features pickle file"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Features File",
        filetypes=[
            ("Pickle files", "*.pkl"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def load_features(file_path):
    """Load features from pickle file"""
    print(f"Loading features from: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            features = pickle.load(f)
        
        if isinstance(features, pd.DataFrame):
            print(f"✓ Loaded {len(features)} frames, {len(features.columns)} features")
            return features
        else:
            print(f"✗ Error: Expected DataFrame, got {type(features)}")
            return None
            
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def identify_brightness_features(features_df):
    """Identify which columns are brightness features"""
    brightness_cols = []
    bodyparts = set()
    
    for col in features_df.columns:
        col_lower = col.lower()
        is_brightness = False
        
        # Check multiple patterns for brightness features:
        # 1. Standard: bodypart_pixbrt_stat
        if 'pixbrt' in col_lower or 'brightness' in col_lower or 'pix_brt' in col_lower:
            is_brightness = True
        
        # 2. Alternative: Pix_bodypart_stat, /Pix patterns
        elif col.startswith('Pix_') or col.startswith('Log10(Pix_') or \
             col.startswith('|d/dt(Pix_') or col.startswith('|d/dt(Log10(Pix_') or \
             '/Pix' in col or 'Pix_' in col:
            is_brightness = True
        
        if is_brightness:
            brightness_cols.append(col)
            
            # Extract bodypart names (handle different formats)
            # Remove special characters and split
            col_clean = col.replace('(', '_').replace(')', '_').replace('|', '').replace('/', '_')
            parts = col_clean.split('_')
            
            # Skip common prefixes/suffixes and derivative markers
            skip_words = ['Pix', 'Log10', 'd', 'dt', 'sum', 'centroid', 'mean', 'std', 'median', 
                         'min', 'max', 'range', 'var', 'q25', 'q75']
            
            # Base bodypart names to look for
            base_bodyparts = ['hrpaw', 'hlpaw', 'frpaw', 'flpaw', 'snout', 'neck', 'tailbase', 'tailtip']
            
            # Check each part for bodypart names
            for part in parts:
                part = part.strip().lower()
                if part and len(part) > 1 and part not in skip_words:
                    # Check if it's a known bodypart
                    if any(bp in part for bp in base_bodyparts):
                        bodyparts.add(part)
                    # Or if it contains paw/snout/tail/neck (compound bodyparts)
                    elif any(keyword in part for keyword in ['paw', 'snout', 'tail', 'neck']):
                        bodyparts.add(part)
    
    bodyparts = sorted(bodyparts)
    
    if brightness_cols:
        print(f"\n📊 Found {len(brightness_cols)} brightness features")
        print(f"   Body parts: {', '.join(bodyparts) if bodyparts else 'Could not extract'}")
    
    return brightness_cols, bodyparts


def calculate_statistics(features_df, brightness_cols):
    """Calculate comprehensive statistics for brightness features"""
    stats = []
    
    for col in brightness_cols:
        data = features_df[col].values
        
        # Count NaNs
        n_nans = np.isnan(data).sum()
        n_total = len(data)
        pct_nans = (n_nans / n_total) * 100 if n_total > 0 else 0
        
        # Use nan-aware functions
        stat = {
            'Feature': col,
            'Min': np.nanmin(data) if n_nans < n_total else np.nan,
            'Max': np.nanmax(data) if n_nans < n_total else np.nan,
            'Mean': np.nanmean(data) if n_nans < n_total else np.nan,
            'Median': np.nanmedian(data) if n_nans < n_total else np.nan,
            'Std': np.nanstd(data) if n_nans < n_total else np.nan,
            'Range': np.nanmax(data) - np.nanmin(data) if n_nans < n_total else np.nan,
            'CV': np.nanstd(data) / np.nanmean(data) if (n_nans < n_total and np.nanmean(data) > 0) else 0,
            'Q25': np.nanpercentile(data, 25) if n_nans < n_total else np.nan,
            'Q75': np.nanpercentile(data, 75) if n_nans < n_total else np.nan,
            'N_NaNs': n_nans,
            'Pct_NaNs': pct_nans,
        }
        
        # Dynamic range in dB (if values are positive)
        min_val = np.nanmin(data) if n_nans < n_total else 0
        max_val = np.nanmax(data) if n_nans < n_total else 0
        
        if min_val > 0 and max_val > 0:
            stat['Dynamic_Range_dB'] = 20 * np.log10(max_val / min_val)
        else:
            stat['Dynamic_Range_dB'] = np.nan
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def plot_statistics_table(stats_df, save_path):
    """Create a visual table of statistics"""
    fig, ax = plt.subplots(figsize=(16, max(8, len(stats_df) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key columns including NaN percentage
    display_cols = ['Feature', 'Min', 'Max', 'Mean', 'Std', 'Range', 'CV', 'Pct_NaNs']
    display_df = stats_df[display_cols].copy()
    
    # Round numeric columns
    for col in display_cols[1:]:
        if col == 'Pct_NaNs':
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '0%')
        else:
            display_df[col] = display_df[col].round(2)
    
    # Color code rows based on range
    cell_colors = []
    for _, row in display_df.iterrows():
        range_val = row['Range']
        
        # Handle NaN range
        if pd.isna(range_val) or range_val < 0.001:
            color = '#ffcccc'  # Red - very low/NaN range
        elif range_val < 10:
            color = '#ffcccc'  # Red - very low range
        elif range_val < 30:
            color = '#fff3cd'  # Yellow - low range
        elif range_val < 50:
            color = '#d1ecf1'  # Blue - moderate range
        else:
            color = '#d4edda'  # Green - good range
        
        cell_colors.append([color] * len(display_cols))
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['#e9ecef'] * len(display_cols)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Make header bold
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#495057')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Brightness Feature Statistics\n' +
              'Color code: Red=Very Low Range, Yellow=Low, Blue=Moderate, Green=Good',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved statistics table: {save_path}")
    plt.close()


def plot_histograms(features_df, brightness_cols, bodyparts, save_path):
    """Create histogram for each brightness feature, grouped by bodypart"""
    # Group features by bodypart (handle different naming conventions)
    bodypart_features = {bp: [] for bp in bodyparts}
    
    for col in brightness_cols:
        col_lower = col.lower()
        # Match bodypart anywhere in feature name
        for bp in bodyparts:
            if bp.lower() in col_lower:
                bodypart_features[bp].append(col)
                break
    
    # Remove bodyparts with no features
    bodyparts_with_features = [bp for bp in bodyparts if bodypart_features[bp]]
    
    if not bodyparts_with_features:
        print("⚠️  Could not match brightness features to bodyparts")
        return
    
    # Detect if data is normalized (0-1) or raw (0-255)
    sample_data = features_df[brightness_cols].values.flatten()
    sample_data = sample_data[~np.isnan(sample_data)]  # Remove NaNs
    max_val = np.max(sample_data) if len(sample_data) > 0 else 255
    is_normalized = max_val <= 1.5  # If max <= 1.5, assume normalized
    
    x_max = 1.0 if is_normalized else 255
    x_label = "Brightness Value (normalized 0-1)" if is_normalized else "Brightness Value (0-255)"
    low_range_threshold = 0.1 if is_normalized else 30
    
    # Create figure
    n_bodyparts = len(bodyparts_with_features)
    fig, axes = plt.subplots(n_bodyparts, 1, figsize=(12, 3 * n_bodyparts))
    
    if n_bodyparts == 1:
        axes = [axes]
    
    for ax, bodypart in zip(axes, bodyparts_with_features):
        features = bodypart_features[bodypart]
        
        if not features:
            continue
        
        # Plot histogram for each feature of this bodypart
        for feature in features:
            data = features_df[feature].values
            data = data[~np.isnan(data)]  # Remove NaNs
            if len(data) > 0:
                # Extract label from feature name
                label = feature.replace('Pix_', '').replace('Log10(', '').replace('|d/dt(', '').replace(')', '')
                ax.hist(data, bins=50, alpha=0.5, label=label[:30], edgecolor='black', linewidth=0.5)
        
        # Calculate overall stats for this bodypart
        all_data = features_df[features].values.flatten()
        all_data = all_data[~np.isnan(all_data)]
        range_val = np.ptp(all_data) if len(all_data) > 0 else 0
        mean_val = np.mean(all_data) if len(all_data) > 0 else 0
        std_val = np.std(all_data) if len(all_data) > 0 else 0
        
        ax.set_title(f'{bodypart.upper()} - Range: {range_val:.4f}, Mean: {mean_val:.4f}, Std: {std_val:.4f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlim([0, x_max])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
        # Add warning for low range
        if range_val < low_range_threshold:
            ax.axvspan(0, x_max, alpha=0.1, color='red')
            ax.text(0.5, 0.95, '⚠️ LOW DYNAMIC RANGE',
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    plt.suptitle('Brightness Distribution by Body Part', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histograms: {save_path}")
    plt.close()


def plot_temporal_brightness(features_df, brightness_cols, bodyparts, fps=30, save_path=None):
    """Plot brightness over time for each bodypart"""
    # Group features by bodypart (handle different naming conventions)
    bodypart_features = {bp: [] for bp in bodyparts}
    
    for col in brightness_cols:
        col_lower = col.lower()
        # Match bodypart anywhere in feature name
        for bp in bodyparts:
            if bp.lower() in col_lower:
                bodypart_features[bp].append(col)
                break
    
    # Remove bodyparts with no features
    bodyparts_with_features = [bp for bp in bodyparts if bodypart_features[bp]]
    
    if not bodyparts_with_features:
        print("⚠️  Could not match brightness features to bodyparts")
        return
    
    # Create figure
    n_bodyparts = len(bodyparts_with_features)
    fig, axes = plt.subplots(n_bodyparts, 1, figsize=(14, 2.5 * n_bodyparts), sharex=True)
    
    if n_bodyparts == 1:
        axes = [axes]
    
    time = np.arange(len(features_df)) / fps / 60  # Convert to minutes
    
    # Detect if data is normalized (0-1) or raw (0-255)
    sample_data = features_df[brightness_cols].values.flatten()
    sample_data = sample_data[~np.isnan(sample_data)]  # Remove NaNs
    max_val = np.max(sample_data) if len(sample_data) > 0 else 255
    is_normalized = max_val <= 1.5  # If max <= 1.5, assume normalized
    
    y_max = 1.0 if is_normalized else 255
    y_label_suffix = "(normalized 0-1)" if is_normalized else "(0-255)"
    
    for ax, bodypart in zip(axes, bodyparts_with_features):
        features = bodypart_features[bodypart]
        
        if not features:
            continue
        
        # Calculate mean brightness across all features for this bodypart
        bodypart_data = features_df[features].mean(axis=1).values
        std_val = np.std(bodypart_data)
        range_val = np.ptp(bodypart_data)
        
        # Plot
        ax.plot(time, bodypart_data, linewidth=1, alpha=0.7, color='steelblue')
        ax.fill_between(time, 
                        bodypart_data - features_df[features].std(axis=1),
                        bodypart_data + features_df[features].std(axis=1),
                        alpha=0.2, color='steelblue')
        
        ax.set_ylabel(f'{bodypart.upper()}\nBrightness {y_label_suffix}', fontsize=10, fontweight='bold')
        ax.set_ylim([0, y_max])
        ax.grid(alpha=0.3)
        
        # Add statistics text
        ax.text(0.02, 0.97, f'Std: {std_val:.4f}\nRange: {range_val:.4f}',
               transform=ax.transAxes,
               va='top', ha='left',
               fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Highlight low variance (adjust threshold for normalized data)
        threshold = 0.02 if is_normalized else 5
        if std_val < threshold:
            ax.set_facecolor('#fff3cd')
            ax.text(0.98, 0.97, '⚠️ LOW VARIANCE',
                   transform=ax.transAxes,
                   va='top', ha='right',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    axes[-1].set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    plt.suptitle('Brightness Over Time (Mean ± Std across features)',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved temporal plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(features_df, brightness_cols, save_path):
    """Plot correlation matrix between brightness features"""
    # Limit to first 50 features if too many
    if len(brightness_cols) > 50:
        print(f"⚠️  Too many features ({len(brightness_cols)}), showing first 50")
        brightness_cols = brightness_cols[:50]
    
    # Calculate correlation
    corr_matrix = features_df[brightness_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax)
    
    ax.set_title('Brightness Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved correlation matrix: {save_path}")
    plt.close()


def generate_report(features_df, brightness_cols, bodyparts, stats_df, output_dir):
    """Generate text report with recommendations"""
    report_path = os.path.join(output_dir, 'brightness_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BRIGHTNESS FEATURE DIAGNOSTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Video: {len(features_df)} frames\n")
        f.write(f"Brightness Features: {len(brightness_cols)}\n")
        
        # Check for ALL NaN features FIRST (before bodyparts analysis)
        all_nan_features = []
        partially_nan_features = []
        
        for col in brightness_cols:
            if col in features_df.columns:
                col_data = features_df[col]
                n_nans = col_data.isnull().sum()
                n_total = len(col_data)
                
                if n_nans == n_total:  # ALL NaN
                    all_nan_features.append(col)
                elif n_nans > 0:  # Some NaN
                    pct_nan = (n_nans / n_total) * 100
                    partially_nan_features.append((col, pct_nan))
        
        # Clean bodyparts list (remove derivative markers and ratio notation)
        clean_bodyparts = []
        for bp in bodyparts:
            # Skip if it looks like derivative notation or ratio
            if '/' in bp or 'd/dt' in bp.lower() or bp.lower() in ['d', 'dt']:
                continue
            clean_bodyparts.append(bp)
        
        f.write(f"Body Parts: {', '.join(clean_bodyparts) if clean_bodyparts else 'N/A'}\n\n")
        
        # Report NaN features prominently
        if all_nan_features:
            f.write("=" * 80 + "\n")
            f.write("WARNING - CRITICAL ISSUES DETECTED\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Found {len(all_nan_features)} features that are COMPLETELY NaN:\n\n")
            for feat in all_nan_features:
                f.write(f"  - {feat}\n")
            f.write("\n[!] These features contain NO information and should be REMOVED!\n")
            f.write("\nCommon causes:\n")
            f.write("  - Derivative calculation failed (|d/dt features)\n")
            f.write("  - Division by zero in ratio features\n")
            f.write("  - Log of zero or negative values\n\n")
        
        if partially_nan_features:
            f.write("=" * 80 + "\n")
            f.write("MINOR ISSUES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Found {len(partially_nan_features)} features with some NaN values:\n\n")
            for feat, pct in partially_nan_features:
                f.write(f"  - {feat}: {pct:.1f}% NaN\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("SUMMARY BY BODY PART\n")
        f.write("-" * 80 + "\n\n")
        
        # Analyze each bodypart (using flexible matching)
        warnings = []
        recommendations = []
        
        for bodypart in bodyparts:
            # Match bodypart anywhere in feature name (flexible)
            bp_features = [col for col in brightness_cols if bodypart.lower() in col.lower()]
            bp_stats = stats_df[stats_df['Feature'].str.contains(bodypart, case=False, na=False)]
            
            if len(bp_stats) == 0:
                continue
            
            # Filter out NaN features for analysis
            bp_stats_valid = bp_stats[~bp_stats['Mean'].isna()]
            
            if len(bp_stats_valid) == 0:
                f.write(f"{bodypart.upper()}:\n")
                f.write(f"  Features: {len(bp_features)}\n")
                f.write(f"  [!] ALL features are NaN - no valid data\n\n")
                continue
            
            mean_range = bp_stats_valid['Range'].mean()
            mean_std = bp_stats_valid['Std'].mean()
            mean_cv = bp_stats_valid['CV'].mean()
            n_nan_features = len(bp_stats) - len(bp_stats_valid)
            
            f.write(f"{bodypart.upper()}:\n")
            f.write(f"  Features: {len(bp_features)} ({n_nan_features} are NaN)\n")
            f.write(f"  Average Range: {mean_range:.2f}\n")
            f.write(f"  Average Std Dev: {mean_std:.2f}\n")
            f.write(f"  Coefficient of Variation: {mean_cv:.3f}\n")
            
            # Assess quality
            if mean_range < 10:
                status = "[X] VERY LOW - Not informative"
                warnings.append(f"{bodypart}: Very low dynamic range ({mean_range:.1f})")
                recommendations.append(f"Consider REMOVING {bodypart} brightness features")
            elif mean_range < 30:
                status = "[!] LOW - May not be informative"
                warnings.append(f"{bodypart}: Low dynamic range ({mean_range:.1f})")
                recommendations.append(f"Review necessity of {bodypart} brightness features")
            elif mean_range < 50:
                status = "[OK] MODERATE - Usable"
            else:
                status = "[GOOD] HIGHLY INFORMATIVE"
            
            f.write(f"  Status: {status}\n\n")
        
        # Overall assessment
        f.write("-" * 80 + "\n")
        f.write("OVERALL ASSESSMENT\n")
        f.write("-" * 80 + "\n\n")
        
        overall_range = stats_df['Range'].mean()
        overall_std = stats_df['Std'].mean()
        
        f.write(f"Average Dynamic Range: {overall_range:.2f}\n")
        f.write(f"Average Standard Deviation: {overall_std:.2f}\n\n")
        
        if warnings:
            f.write("WARNINGS:\n")
            for warning in warnings:
                f.write(f"  - {warning}\n")
            f.write("\n")
        
        if recommendations:
            f.write("RECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"  - {rec}\n")
            f.write("\n")
        
        # Good practices
        f.write("-" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("-" * 80 + "\n\n")
        f.write("Dynamic Range Categories:\n")
        f.write("  - < 10:  Very Low - Features likely not informative\n")
        f.write("  - 10-30: Low - Features may have limited utility\n")
        f.write("  - 30-50: Moderate - Features should be useful\n")
        f.write("  - > 50:  Good - Features are highly informative\n\n")
        f.write("Coefficient of Variation (CV = Std/Mean):\n")
        f.write("  - < 0.1: Low variability\n")
        f.write("  - 0.1-0.3: Moderate variability\n")
        f.write("  - > 0.3: High variability (good for classification)\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved text report: {report_path}")


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("BRIGHTNESS FEATURE DIAGNOSTICS")
    print("=" * 80 + "\n")
    
    # Select features file
    features_path = select_features_file()
    
    if not features_path:
        print("✗ No file selected. Exiting.")
        return
    
    # Load features
    features_df = load_features(features_path)
    
    if features_df is None:
        return
    
    # Identify brightness features
    brightness_cols, bodyparts = identify_brightness_features(features_df)
    
    if not brightness_cols:
        print("\n✗ No brightness features found in this file.")
        print("   Brightness features should contain 'pixbrt' in their name.")
        return
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats_df = calculate_statistics(features_df, brightness_cols)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(features_path), 'brightness_diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Saving results to: {output_dir}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    
    plot_statistics_table(
        stats_df,
        os.path.join(output_dir, '1_statistics_table.png')
    )
    
    plot_histograms(
        features_df,
        brightness_cols,
        bodyparts,
        os.path.join(output_dir, '2_brightness_histograms.png')
    )
    
    plot_temporal_brightness(
        features_df,
        brightness_cols,
        bodyparts,
        fps=30,
        save_path=os.path.join(output_dir, '3_temporal_brightness.png')
    )
    
    if len(brightness_cols) <= 50:
        plot_correlation_matrix(
            features_df,
            brightness_cols,
            os.path.join(output_dir, '4_correlation_matrix.png')
        )
    
    # Generate text report
    print("\n📝 Generating report...")
    generate_report(features_df, brightness_cols, bodyparts, stats_df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Analysis complete!")
    print(f"✓ Generated {4 if len(brightness_cols) <= 50 else 3} plots")
    print(f"✓ Results saved to: {output_dir}")
    print(f"\n📊 Quick Stats:")
    print(f"   Average Range: {stats_df['Range'].mean():.1f}")
    print(f"   Average Std: {stats_df['Std'].mean():.1f}")
    
    # Warnings
    low_range = stats_df[stats_df['Range'] < 30]
    if len(low_range) > 0:
        print(f"\n⚠️  {len(low_range)} features have low dynamic range (<30)")
        print("   Review the report for recommendations")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
