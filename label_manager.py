"""
Smart Label Manager for PixelPaws Active Learning

Handles both dense regions (BORIS manual labeling) and sparse labels (Active Learning)
with clear semantics to avoid training confusion.

Key Concepts:
- Dense regions: Contiguous labeled sections (e.g., first 10 min from BORIS)
  - Explicitly labeled frames: Use their labels (0 or 1)
  - Unlabeled frames within region: Assume negative (0)
  
- Sparse labels: Individual frames labeled via Active Learning
  - Only use explicitly labeled frames
  - Don't make assumptions about surrounding frames

Author: PixelPaws Team
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional


class SmartLabelManager:
    """
    Manages labels with explicit handling of dense and sparse regions.
    Prevents training confusion when mixing BORIS and Active Learning labels.
    """
    
    def __init__(self, video_name: str, total_frames: int, behavior_name: str = None):
        """
        Initialize label manager.
        
        Args:
            video_name: Base name of video (without extension)
            total_frames: Total number of frames in video
            behavior_name: Name of behavior being labeled (optional)
        """
        self.video_name = video_name
        self.total_frames = total_frames
        self.behavior_name = behavior_name
        
        # Sparse database: Only stores explicitly labeled frames
        self.sparse_db = pd.DataFrame(columns=[
            'frame_index', 'label', 'source', 'timestamp', 'confidence'
        ])
        
        # Dense regions: List of (start, end) tuples
        self.dense_regions = []
        
        # Metadata
        self.metadata = {
            'video_name': video_name,
            'total_frames': total_frames,
            'behavior_name': behavior_name,
            'created': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'label_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'dense_region_count': 0,
            'active_learning_iterations': 0
        }
    
    def add_dense_region(self, start: int, end: int, labels: np.ndarray, 
                        source: str = 'boris_manual'):
        """
        Add a densely labeled region (e.g., from BORIS).
        
        Within this region:
        - Explicitly labeled frames (label in [0,1]): Use their labels
        - Unlabeled frames: Assume negative (0)
        
        Args:
            start: Start frame index (inclusive)
            end: End frame index (inclusive)
            labels: Array of labels for frames [start:end+1]
                   Values: 0 (negative), 1 (positive), -1 (unlabeled in source)
            source: Source of labels (e.g., 'boris_manual')
        """
        print(f"Adding dense region: frames {start:,} to {end:,}")
        print(f"  Region length: {end - start + 1:,} frames")
        
        # Validate
        expected_length = end - start + 1
        if len(labels) != expected_length:
            raise ValueError(
                f"Labels length ({len(labels)}) doesn't match region "
                f"({expected_length} frames)"
            )
        
        # Register dense region
        self.dense_regions.append((start, end))
        
        # Add explicitly labeled frames to sparse database
        new_labels = []
        for i, label in enumerate(labels):
            frame_idx = start + i
            
            # Store explicitly labeled frames
            if label in [0, 1]:
                new_labels.append({
                    'frame_index': frame_idx,
                    'label': int(label),
                    'source': source,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 1.0
                })
        
        # Add to database
        if new_labels:
            new_df = pd.DataFrame(new_labels)
            self.sparse_db = pd.concat([self.sparse_db, new_df], ignore_index=True)
        
        # Update metadata
        self.metadata['dense_region_count'] = len(self.dense_regions)
        self._update_counts()
        
        print(f"  ✓ Added {len(new_labels):,} explicitly labeled frames")
        print(f"  Note: {expected_length - len(new_labels):,} unlabeled frames "
              f"will be assumed negative within region")
    
    def add_sparse_labels(self, frames: List[int], labels: List[int], 
                         source: str = 'active_learning',
                         confidences: Optional[List[float]] = None):
        """
        Add sparse labels (e.g., from Active Learning).
        
        Does NOT make assumptions about unlabeled frames.
        Only the explicitly labeled frames are stored.
        
        Args:
            frames: List of frame indices
            labels: List of labels (0 or 1)
            source: Source identifier (e.g., 'active_learning_iter1')
            confidences: Optional list of model confidences when labeled
        """
        print(f"\nAdding sparse labels from {source}")
        print(f"  Frames to add: {len(frames)}")
        
        if confidences is None:
            confidences = [1.0] * len(frames)
        
        # Validate
        if len(frames) != len(labels):
            raise ValueError("Frames and labels must have same length")
        
        # Create new entries
        new_labels = []
        for frame, label, conf in zip(frames, labels, confidences):
            new_labels.append({
                'frame_index': int(frame),
                'label': int(label),
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'confidence': float(conf)
            })
        
        # Add to database
        new_df = pd.DataFrame(new_labels)
        self.sparse_db = pd.concat([self.sparse_db, new_df], ignore_index=True)
        
        # Update metadata
        if 'active_learning' in source:
            self.metadata['active_learning_iterations'] += 1
        self._update_counts()
        
        positive = sum(labels)
        negative = len(labels) - positive
        print(f"  ✓ Added {len(frames)} labels ({positive} positive, {negative} negative)")
    
    def get_training_set(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training set with smart handling of dense and sparse regions.
        
        Rules:
        1. Dense regions: Use explicit labels + assume unlabeled = negative
        2. Sparse regions: Only use explicit labels
        3. Never make assumptions about unlabeled sparse regions
        
        Args:
            features: Feature matrix (n_frames, n_features) - numpy array
            
        Returns:
            X_train: Training features
            y_train: Training labels
            train_indices: Indices of frames used for training
        """
        print(f"\n📊 Generating training set...")
        
        # Convert DataFrame to numpy if needed (backup safety check)
        if isinstance(features, pd.DataFrame):
            print(f"  Converting DataFrame to numpy array...")
            features = features.values
        
        # Start with all frames unlabeled
        labels = np.full(self.total_frames, -1, dtype=int)
        
        # 1. Fill dense regions
        for start, end in self.dense_regions:
            print(f"  Processing dense region: {start:,}-{end:,}")
            
            # Initially assume all negative in dense region
            labels[start:end+1] = 0
            
            # Override with explicit labels
            region_db = self.sparse_db[
                (self.sparse_db['frame_index'] >= start) &
                (self.sparse_db['frame_index'] <= end)
            ]
            
            for _, row in region_db.iterrows():
                labels[int(row['frame_index'])] = int(row['label'])
            
            explicit_count = len(region_db)
            assumed_count = (end - start + 1) - explicit_count
            print(f"    Explicit labels: {explicit_count:,}")
            print(f"    Assumed negative: {assumed_count:,}")
        
        # 2. Add sparse labels (outside dense regions)
        sparse_labels = self.sparse_db[
            ~self.sparse_db['frame_index'].apply(self._in_dense_region)
        ]
        
        if len(sparse_labels) > 0:
            print(f"  Processing sparse labels: {len(sparse_labels):,}")
            for _, row in sparse_labels.iterrows():
                labels[int(row['frame_index'])] = int(row['label'])
        
        # 3. Extract training set (only labeled frames)
        train_mask = (labels != -1)
        train_indices = np.where(train_mask)[0]
        
        X_train = features[train_indices]
        y_train = labels[train_indices]
        
        # Summary
        print(f"\n  Training set summary:")
        print(f"    Total frames in video: {self.total_frames:,}")
        print(f"    Frames used for training: {len(train_indices):,} ({len(train_indices)/self.total_frames*100:.1f}%)")
        print(f"    Positive samples: {(y_train == 1).sum():,}")
        print(f"    Negative samples: {(y_train == 0).sum():,}")
        print(f"    Unlabeled (excluded): {(~train_mask).sum():,}")
        
        return X_train, y_train, train_indices
    
    def get_unlabeled_frames(self, exclude_dense_regions: bool = True) -> List[int]:
        """
        Get list of frames that haven't been explicitly labeled.
        
        Args:
            exclude_dense_regions: If True, also exclude entire dense regions
                                  (useful for Active Learning on new sections)
            
        Returns:
            List of unlabeled frame indices
        """
        # Get explicitly labeled frames
        labeled_frames = set(self.sparse_db['frame_index'].values)
        
        # Get all frames
        all_frames = set(range(self.total_frames))
        
        # Remove labeled
        unlabeled = all_frames - labeled_frames
        
        # Optionally remove dense regions
        if exclude_dense_regions:
            for start, end in self.dense_regions:
                dense_frames = set(range(start, end + 1))
                unlabeled = unlabeled - dense_frames
        
        return sorted(list(unlabeled))
    
    def get_label_coverage_map(self) -> np.ndarray:
        """
        Get coverage map showing where labels exist.
        
        Returns:
            Array of shape (total_frames,) with values:
            - 0: Unlabeled
            - 1: Explicitly labeled
            - 2: Within dense region (assumed negative)
        """
        coverage = np.zeros(self.total_frames, dtype=int)
        
        # Mark dense regions
        for start, end in self.dense_regions:
            coverage[start:end+1] = 2
        
        # Mark explicit labels
        for frame in self.sparse_db['frame_index']:
            coverage[int(frame)] = 1
        
        return coverage
    
    def save(self, output_dir: str):
        """
        Save label database and metadata to disk.
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sparse database
        db_path = os.path.join(output_dir, f"{self.video_name}_labels.db.csv")
        self.sparse_db.to_csv(db_path, index=False)
        print(f"✓ Saved label database: {db_path}")
        
        # Save metadata with dense regions
        meta_path = os.path.join(output_dir, f"{self.video_name}_metadata.json")
        meta_with_regions = self.metadata.copy()
        meta_with_regions['dense_regions'] = [
            {'start': int(s), 'end': int(e)} for s, e in self.dense_regions
        ]
        
        with open(meta_path, 'w') as f:
            json.dump(meta_with_regions, f, indent=2)
        print(f"✓ Saved metadata: {meta_path}")
    
    def load(self, input_dir: str):
        """
        Load label database and metadata from disk.
        
        Args:
            input_dir: Directory containing saved files
        """
        # Load sparse database
        db_path = os.path.join(input_dir, f"{self.video_name}_labels.db.csv")
        if os.path.exists(db_path):
            self.sparse_db = pd.read_csv(db_path)
            print(f"✓ Loaded label database: {db_path}")
        else:
            print(f"⚠ No existing database found at {db_path}")
        
        # Load metadata
        meta_path = os.path.join(input_dir, f"{self.video_name}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                loaded_meta = json.load(f)
            
            # Restore dense regions
            if 'dense_regions' in loaded_meta:
                self.dense_regions = [
                    (r['start'], r['end']) for r in loaded_meta['dense_regions']
                ]
                del loaded_meta['dense_regions']
            
            self.metadata.update(loaded_meta)
            print(f"✓ Loaded metadata: {meta_path}")
        else:
            print(f"⚠ No existing metadata found at {meta_path}")
    
    def export_to_perframe_csv(self, output_path: str, default_label: int = 0):
        """
        Export to dense per-frame CSV format (for compatibility).
        
        Args:
            output_path: Path to save CSV file
            default_label: Default label for unlabeled frames (0 or -1)
        """
        print(f"\nExporting to per-frame CSV: {output_path}")
        
        # Generate dense labels
        labels = np.full(self.total_frames, default_label, dtype=int)
        
        # Fill dense regions
        for start, end in self.dense_regions:
            labels[start:end+1] = 0  # Assume negative
        
        # Apply explicit labels
        for _, row in self.sparse_db.iterrows():
            labels[int(row['frame_index'])] = int(row['label'])
        
        # Create DataFrame
        behavior_name = self.behavior_name or 'behavior'
        df = pd.DataFrame({behavior_name: labels})
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"  ✓ Exported {self.total_frames:,} frames")
        print(f"  ✓ {(labels == 1).sum():,} positive labels")
        print(f"  ✓ {(labels == 0).sum():,} negative labels")
        if default_label == -1:
            print(f"  ✓ {(labels == -1).sum():,} unlabeled")
    
    def _in_dense_region(self, frame: int) -> bool:
        """Check if frame is within any dense region"""
        for start, end in self.dense_regions:
            if start <= frame <= end:
                return True
        return False
    
    def _update_counts(self):
        """Update metadata counts"""
        self.metadata['last_modified'] = datetime.now().isoformat()
        self.metadata['label_count'] = len(self.sparse_db)
        self.metadata['positive_count'] = int((self.sparse_db['label'] == 1).sum())
        self.metadata['negative_count'] = int((self.sparse_db['label'] == 0).sum())
    
    def print_summary(self):
        """Print summary of current label state"""
        print("\n" + "="*60)
        print(f"📊 LABEL SUMMARY: {self.video_name}")
        print("="*60)
        print(f"Behavior: {self.behavior_name or 'Not specified'}")
        print(f"Total frames: {self.total_frames:,}")
        print(f"\nDense regions: {len(self.dense_regions)}")
        for i, (start, end) in enumerate(self.dense_regions, 1):
            print(f"  {i}. Frames {start:,} to {end:,} ({end-start+1:,} frames)")
        
        print(f"\nExplicit labels: {len(self.sparse_db):,}")
        print(f"  Positive: {self.metadata['positive_count']:,}")
        print(f"  Negative: {self.metadata['negative_count']:,}")
        
        if len(self.dense_regions) > 0:
            dense_frames = sum(end - start + 1 for start, end in self.dense_regions)
            explicit_in_dense = len(self.sparse_db[
                self.sparse_db['frame_index'].apply(self._in_dense_region)
            ])
            assumed_negative = dense_frames - explicit_in_dense
            print(f"\nDense region details:")
            print(f"  Total frames: {dense_frames:,}")
            print(f"  Explicitly labeled: {explicit_in_dense:,}")
            print(f"  Assumed negative: {assumed_negative:,}")
        
        sparse_labels = self.sparse_db[
            ~self.sparse_db['frame_index'].apply(self._in_dense_region)
        ]
        if len(sparse_labels) > 0:
            print(f"\nSparse labels (outside dense regions): {len(sparse_labels):,}")
        
        unlabeled = self.get_unlabeled_frames(exclude_dense_regions=True)
        print(f"\nUnlabeled frames (available for AL): {len(unlabeled):,}")
        
        print(f"\nActive Learning iterations: {self.metadata['active_learning_iterations']}")
        print("="*60)


def convert_boris_to_label_manager(boris_csv: str, video_name: str, 
                                   total_frames: int, behavior_name: str,
                                   fps: float = None) -> SmartLabelManager:
    """
    Convert BORIS CSV to SmartLabelManager format.
    
    Args:
        boris_csv: Path to BORIS export CSV
        video_name: Base name of video
        total_frames: Total frames in video
        behavior_name: Behavior to extract
        fps: Frames per second (if not in CSV)
        
    Returns:
        SmartLabelManager with labels loaded
    """
    print(f"\n🔄 Converting BORIS to Label Manager...")
    print(f"  BORIS file: {boris_csv}")
    print(f"  Behavior: {behavior_name}")
    
    # Load BORIS CSV (implementation depends on your BORIS converter)
    # This is a placeholder - use your existing BORIS converter
    from boris_converter import convert_boris_to_perframe  # Your existing function
    
    perframe_labels = convert_boris_to_perframe(boris_csv, behavior_name, fps)
    
    # Find the labeled range
    labeled_indices = np.where(perframe_labels != -1)[0]
    if len(labeled_indices) == 0:
        raise ValueError(f"No labels found for behavior '{behavior_name}' in BORIS file")
    
    start_frame = labeled_indices[0]
    end_frame = labeled_indices[-1]
    
    # Create manager
    manager = SmartLabelManager(video_name, total_frames, behavior_name)
    
    # Add as dense region
    manager.add_dense_region(
        start=int(start_frame),
        end=int(end_frame),
        labels=perframe_labels[start_frame:end_frame+1],
        source='boris_manual'
    )
    
    print(f"  ✓ Loaded frames {start_frame:,} to {end_frame:,}")
    
    return manager


if __name__ == "__main__":
    # Example usage
    print("SmartLabelManager Example")
    print("="*60)
    
    # Create manager for a video
    manager = SmartLabelManager(
        video_name="251114_Formalin_S4",
        total_frames=300000,
        behavior_name="Left_licking"
    )
    
    # Add dense region from BORIS (first 10 minutes = 60,000 frames @ 100 fps)
    boris_labels = np.random.randint(0, 2, 60000)  # Placeholder
    manager.add_dense_region(
        start=0,
        end=59999,
        labels=boris_labels,
        source='boris_manual'
    )
    
    # Simulate Active Learning iterations
    for i in range(3):
        # Find 50 uncertain frames
        unlabeled = manager.get_unlabeled_frames(exclude_dense_regions=True)
        suggested = np.random.choice(unlabeled, 50, replace=False)
        
        # Simulate user labeling
        al_labels = np.random.randint(0, 2, 50)
        
        # Add to manager
        manager.add_sparse_labels(
            frames=suggested.tolist(),
            labels=al_labels.tolist(),
            source=f'active_learning_iter{i+1}'
        )
    
    # Print summary
    manager.print_summary()
    
    # Generate training set (would need actual features)
    # X_train, y_train, indices = manager.get_training_set(features)
