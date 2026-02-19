"""
Active Learning Module for PixelPaws

Intelligently suggests which frames to label for maximum classifier improvement.
Reduces labeling time by 50-70% while maintaining accuracy.

Usage:
    from active_learning import ActiveLearningSession
    
    session = ActiveLearningSession(
        labels_csv="path/to/labels_perframe.csv",
        video_path="path/to/video.mp4",
        dlc_path="path/to/dlc.h5",
        features_cache="path/to/features_cache.pkl"
    )
    
    session.run()

Author: PixelPaws Team
"""

import numpy as np
import pandas as pd
import pickle
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import os
from typing import List, Tuple, Optional
import time
import glob

try:
    from evaluation_tab import find_session_triplets as _find_session_triplets
    _HAS_EVAL_TAB = True
except ImportError:
    _HAS_EVAL_TAB = False


def scan_folders_for_sessions(root_folder: str, search_parents: bool = True) -> List[dict]:
    """
    Scan for matching video, DLC, and label files.
    Delegates to find_session_triplets() from evaluation_tab for consistent
    discovery across all PixelPaws contexts.

    The ``search_parents`` flag is kept for backward compatibility; when True,
    the parent directory is also tried (mirroring the original behaviour).
    """
    if _HAS_EVAL_TAB:
        sessions = _find_session_triplets(root_folder, prefer_filtered=True, require_labels=True)
        if not sessions and search_parents:
            parent = os.path.dirname(root_folder)
            if parent and parent != root_folder:
                sessions = _find_session_triplets(parent, prefer_filtered=True, require_labels=True)
                # Filter to only sessions whose video lives under root_folder or parent
    else:
        # Fallback: minimal inline implementation if evaluation_tab is unavailable
        sessions = _scan_folders_fallback(root_folder, search_parents)

    print(f"🔍 Scan complete — {len(sessions)} session(s) found in {root_folder}")
    for s in sessions:
        print(f"  ✓ {s['session_name']}")
    return sessions


def _scan_folders_fallback(root_folder: str, search_parents: bool) -> List[dict]:
    """Minimal fallback used when evaluation_tab is not importable."""
    search_dirs = [root_folder]
    if search_parents:
        parent = os.path.dirname(root_folder)
        if parent and parent != root_folder:
            search_dirs.append(parent)

    all_dlc = []
    for d in search_dirs:
        all_dlc.extend(glob.glob(os.path.join(d, '**', '*.h5'), recursive=True))
    all_dlc = list(set(all_dlc))

    sessions = []
    seen = set()
    for dlc_path in all_dlc:
        dlc_name = os.path.basename(dlc_path)
        base = dlc_name.split('DLC')[0] if 'DLC' in dlc_name else os.path.splitext(dlc_name)[0]
        for sfx in ('_Labels', '_labels', '_perframe'):
            if base.endswith(sfx):
                base = base[:-len(sfx)]
                break
        if base in seen:
            continue

        dlc_dir = os.path.dirname(dlc_path)
        video_path = None
        for ext in ('.mp4', '.avi', '.MP4', '.AVI'):
            c = os.path.join(dlc_dir, base + ext)
            if os.path.isfile(c):
                video_path = c
                break
        if not video_path:
            continue

        video_dir  = os.path.dirname(video_path)
        parent_dir = os.path.dirname(video_dir)
        label_candidates = [
            os.path.join(video_dir,  f'{base}_labels.csv'),
            os.path.join(video_dir,  f'{base}_perframe.csv'),
            os.path.join(parent_dir, 'labels', f'{base}_labels.csv'),
            os.path.join(parent_dir, 'Labels', f'{base}_labels.csv'),
            os.path.join(parent_dir, 'Targets', f'{base}.csv'),
        ]
        labels_path = next((p for p in label_candidates if os.path.isfile(p)), None)
        if not labels_path:
            continue

        seen.add(base)
        features_path = os.path.join(video_dir, f'{base}_features_cache.pkl')
        sessions.append({
            'session_name': base,
            'base_name':    base,
            'video':        video_path,
            'video_path':   video_path,
            'dlc':          dlc_path,
            'dlc_path':     dlc_path,
            'labels':       labels_path,
            'labels_path':  labels_path,
            'target_path':  labels_path,
            'video_dir':    video_dir,
            'project_dir':  parent_dir,
            'features_path': features_path,
            'has_features': os.path.exists(features_path),
        })
    return sessions


class ActiveLearningEngine:
    """
    Core active learning algorithm.
    Finds frames where the model is most uncertain.
    """
    
    def __init__(self, min_frame_spacing: int = 30):
        """
        Args:
            min_frame_spacing: Minimum frames between suggestions (avoid clustering)
        """
        self.min_frame_spacing = min_frame_spacing
    
    def find_uncertain_frames(self, 
                             model,
                             features: np.ndarray,
                             current_labels: np.ndarray,
                             n_suggestions: int = 20,
                             avoid_labeled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find frames where model is most uncertain.
        
        Strategies:
        1. Find frames near decision boundary (confidence ≈ 0.5)
        2. Prioritize unlabeled regions
        3. Ensure spatial diversity (spread across video)
        
        Args:
            model: Trained classifier with predict_proba method
            features: Feature matrix (n_frames, n_features)
            current_labels: Current label array (n_frames,)
            n_suggestions: Number of frames to suggest
            avoid_labeled: Whether to avoid already-labeled positive frames
            
        Returns:
            uncertain_indices: Frame indices to label
            confidences: Model confidence for each suggested frame
        """
        print(f"  Finding {n_suggestions} most informative frames...")
        
        # Get prediction probabilities
        probas = model.predict_proba(features)[:, 1]  # P(behavior=1)
        
        # Calculate uncertainty (distance from decision boundary)
        # Uncertainty is highest when proba = 0.5
        uncertainty = 1 - np.abs(probas - 0.5) * 2
        
        # Identify explicitly labeled frames (positive labels)
        if avoid_labeled:
            labeled_mask = current_labels == 1
            uncertainty[labeled_mask] *= 0.1  # Strongly deprioritize
        
        # Find top uncertain frames with spatial diversity
        selected_indices = []
        uncertainty_copy = uncertainty.copy()
        
        while len(selected_indices) < n_suggestions:
            # Find most uncertain remaining frame
            max_idx = np.argmax(uncertainty_copy)
            
            if uncertainty_copy[max_idx] <= 0:
                # No more uncertain frames
                break
            
            selected_indices.append(max_idx)
            
            # Suppress nearby frames to ensure diversity
            start = max(0, max_idx - self.min_frame_spacing)
            end = min(len(uncertainty_copy), max_idx + self.min_frame_spacing)
            uncertainty_copy[start:end] = 0
        
        selected_indices = np.array(selected_indices)
        confidences = probas[selected_indices]
        
        print(f"  ✓ Selected {len(selected_indices)} frames")
        print(f"    Confidence range: {confidences.min():.2f} - {confidences.max():.2f}")
        print(f"    Mean uncertainty: {uncertainty[selected_indices].mean():.3f}")
        
        return selected_indices, confidences


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
        except:
            self.window = tk.Tk()
        
        self.window.title(f"Active Learning - {self.behavior_name}")
        self.window.geometry("900x700")
        
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
        
        btn_yes = tk.Button(
            button_frame,
            text="✓ YES (Y)",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            width=15,
            height=2,
            command=lambda: self.label_frame(1)
        )
        btn_yes.pack(side=tk.LEFT, padx=5, expand=True)
        
        btn_no = tk.Button(
            button_frame,
            text="✗ NO (N)",
            font=("Arial", 14, "bold"),
            bg="#f44336",
            fg="white",
            width=15,
            height=2,
            command=lambda: self.label_frame(0)
        )
        btn_no.pack(side=tk.LEFT, padx=5, expand=True)
        
        btn_skip = tk.Button(
            button_frame,
            text="? SKIP (S)",
            font=("Arial", 14),
            bg="#9E9E9E",
            fg="white",
            width=15,
            height=2,
            command=self.skip_frame
        )
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
        play_window.geometry("800x600")
        
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


class ActiveLearningSession:
    """
    Complete active learning session manager with smart label handling.
    """
    
    def __init__(self,
                 labels_csv: str,
                 video_path: str,
                 dlc_path: str,
                 features_cache: str,
                 model_path: Optional[str] = None,
                 use_smart_labels: bool = True):
        """
        Args:
            labels_csv: Path to per-frame labels CSV
            video_path: Path to video file
            dlc_path: Path to DLC tracking file
            features_cache: Path to cached features (or where to save)
            model_path: Path to trained model (optional, will train if not provided)
            use_smart_labels: Use SmartLabelManager (recommended)
        """
        self.labels_csv = labels_csv
        self.video_path = video_path
        self.dlc_path = dlc_path
        self.features_cache = features_cache
        self.model_path = model_path
        self.use_smart_labels = use_smart_labels
        
        # Get video info
        import cv2
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Extract video base name
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        if use_smart_labels:
            # Use SmartLabelManager
            from label_manager import SmartLabelManager
            
            # Load or create label manager
            labels_dir = os.path.dirname(labels_csv)
            manager_dir = os.path.join(labels_dir, 'LabelManager')
            
            self.label_manager = SmartLabelManager(
                video_name=video_basename,
                total_frames=self.total_frames,
                behavior_name=None  # Will be detected from CSV
            )
            
            # Try to load existing manager
            if os.path.exists(manager_dir):
                try:
                    self.label_manager.load(manager_dir)
                    print("✓ Loaded existing label manager")
                except:
                    pass
            
            # If no existing manager, initialize from per-frame CSV
            if len(self.label_manager.sparse_db) == 0:
                print("Initializing label manager from per-frame CSV...")
                self._initialize_from_perframe_csv()
            
            # Ensure behavior_name is set on session object (for CSV export)
            self.behavior_name = self.label_manager.behavior_name
            if not self.behavior_name:
                # Fallback: read from CSV
                df = pd.read_csv(labels_csv)
                self.behavior_name = df.columns[0]
                self.label_manager.behavior_name = self.behavior_name
        
        else:
            # Legacy: Load per-frame CSV directly
            self.label_manager = None
            self.labels_df = pd.read_csv(labels_csv)
            self.behavior_name = self.labels_df.columns[0]
            self.labels = self.labels_df[self.behavior_name].values.astype(int)
            
            print(f"Loaded labels: {self.behavior_name}")
            print(f"  Total frames: {len(self.labels)}")
            print(f"  Positive labels: {self.labels.sum()}")
        
        # Engine
        self.engine = ActiveLearningEngine()
    
    def _initialize_from_perframe_csv(self):
        """Initialize SmartLabelManager from existing per-frame CSV"""
        # Load CSV
        df = pd.read_csv(self.labels_csv)
        self.behavior_name = df.columns[0]
        self.label_manager.behavior_name = self.behavior_name
        
        labels = df[self.behavior_name].values
        
        # Find labeled range (assume it's a contiguous dense region)
        labeled_indices = np.where(labels != -1)[0] if -1 in labels else np.arange(len(labels))
        
        if len(labeled_indices) > 0:
            start_frame = labeled_indices[0]
            end_frame = labeled_indices[-1]
            
            # Add as dense region
            self.label_manager.add_dense_region(
                start=int(start_frame),
                end=int(end_frame),
                labels=labels[start_frame:end_frame+1],
                source='existing_perframe_csv'
            )
            
            print(f"  ✓ Initialized from frames {start_frame:,} to {end_frame:,}")
            
            # Save the manager
            labels_dir = os.path.dirname(self.labels_csv)
            manager_dir = os.path.join(labels_dir, 'LabelManager')
            self.label_manager.save(manager_dir)
        
    def run(self, n_suggestions: int = 20, n_iterations: int = 1,
            extend_to_unlabeled: bool = True) -> dict:
        """
        Run active learning session.
        
        Args:
            n_suggestions: Number of frames to suggest per iteration
            n_iterations: Number of iterations to run
            extend_to_unlabeled: If True, suggest frames from unlabeled regions
                               If False, suggest frames from already labeled regions
            
        Returns:
            Statistics about the session
        """
        print("\n" + "="*60)
        print("🧠 ACTIVE LEARNING SESSION")
        print("="*60)
        
        # Get initial stats
        if self.use_smart_labels:
            self.label_manager.print_summary()
            initial_positive = self.label_manager.metadata['positive_count']
        else:
            initial_positive = int(self.labels.sum())
        
        stats = {
            'frames_labeled': 0,
            'iterations': 0,
            'initial_positive': initial_positive,
            'final_positive': initial_positive
        }
        
        for iteration in range(n_iterations):
            print(f"\n📍 Iteration {iteration + 1} / {n_iterations}")
            print("-" * 60)
            
            try:
                # Load features
                print("Loading features...")
                if os.path.exists(self.features_cache):
                    with open(self.features_cache, 'rb') as f:
                        features = pickle.load(f)
                    print(f"  ✓ Loaded cached features: {features.shape}")
                    
                    # Convert DataFrame to numpy immediately (before any indexing operations)
                    if isinstance(features, pd.DataFrame):
                        print(f"  Converting DataFrame to numpy array...")
                        features = features.values
                        print(f"  ✓ Converted to numpy: {features.shape}")
                    
                else:
                    print(f"  ✗ Features not found!")
                    print(f"    Expected: {self.features_cache}")
                    raise FileNotFoundError(f"Features cache not found: {self.features_cache}")
                
                # Get training set
                if self.use_smart_labels:
                    X_train, y_train, train_indices = self.label_manager.get_training_set(features)
                else:
                    # Legacy mode: use all labels, trim features if needed
                    if len(features) != len(self.labels):
                        if len(features) > len(self.labels):
                            print(f"  Note: Features cover full video ({len(features)} frames)")
                            print(f"        Labels cover subset ({len(self.labels)} frames)")
                            print(f"  Using only the labeled portion for active learning")
                            features = features[:len(self.labels)]
                            print(f"  ✓ Using first {len(features)} frames")
                        else:
                            raise ValueError(
                                f"Feature count mismatch!\n"
                                f"Features: {len(features)} frames\n"
                                f"Labels: {len(self.labels)} frames\n"
                                f"Features should be >= labels."
                            )
                    X_train = features
                    y_train = self.labels
                    train_indices = np.arange(len(y_train))
                
                # Train or load model
                print("Loading/training model...")
                model = self._load_or_train_model(X_train, y_train)
                
                # Find uncertain frames
                if self.use_smart_labels:
                    # Get unlabeled frames
                    if extend_to_unlabeled:
                        candidate_frames = self.label_manager.get_unlabeled_frames(
                            exclude_dense_regions=True
                        )
                        print(f"\nSearching {len(candidate_frames):,} unlabeled frames...")
                    else:
                        candidate_frames = train_indices.tolist()
                        print(f"\nSearching within {len(candidate_frames):,} labeled frames...")
                    
                    if len(candidate_frames) == 0:
                        print("  ✗ No candidate frames available!")
                        break
                    
                    # Ensure candidate_frames is numpy array for filtering
                    candidate_frames = np.array(candidate_frames)
                    
                    # Filter candidate frames to valid indices BEFORE indexing
                    valid_candidate_mask = candidate_frames < len(features)
                    if not np.all(valid_candidate_mask):
                        n_invalid = (~valid_candidate_mask).sum()
                        print(f"  ⚠️  Filtering {n_invalid} out-of-bounds candidate frames")
                        print(f"     (Frames >= {len(features)} are beyond features array)")
                        candidate_frames = candidate_frames[valid_candidate_mask]
                    
                    if len(candidate_frames) == 0:
                        print(f"  ✗ No valid candidate frames after filtering!")
                        break
                    
                    # Get predictions for candidates
                    candidate_features = features[candidate_frames]
                    uncertain_indices, confidences = self.engine.find_uncertain_frames(
                        model, candidate_features, np.zeros(len(candidate_features)),
                        n_suggestions=min(n_suggestions, len(candidate_frames))
                    )
                    
                    # Map back to video frame indices
                    uncertain_frames = np.array([candidate_frames[i] for i in uncertain_indices])
                else:
                    # Legacy mode
                    uncertain_frames, confidences = self.engine.find_uncertain_frames(
                        model, features, self.labels, n_suggestions=n_suggestions
                    )
                
                # Filter out any frames that are out of bounds for the labels array
                labels_size = len(self.labels) if not self.use_smart_labels else len(features)
                valid_mask = uncertain_frames < labels_size
                if not np.all(valid_mask):
                    n_invalid = (~valid_mask).sum()
                    print(f"  ⚠️  Filtering {n_invalid} out-of-bounds frames")
                    print(f"     (Frames >= {labels_size} are beyond labels array)")
                    uncertain_frames = uncertain_frames[valid_mask]
                    confidences = confidences[valid_mask]
                    
                if len(uncertain_frames) == 0:
                    print(f"  ✗ No valid frames to label after filtering!")
                    break
                
                # Show labeling interface
                print("\nLaunching labeling interface...")
                interface = LabelingInterface(
                    self.video_path,
                    uncertain_frames,
                    confidences,
                    self.behavior_name if hasattr(self, 'behavior_name') else 
                    self.label_manager.behavior_name
                )
                
                new_labels = interface.run()
                
                # Update labels
                if new_labels:
                    print(f"\nUpdating labels with {len(new_labels)} new labels...")
                    
                    if self.use_smart_labels:
                        # Add to label manager
                        frames = list(new_labels.keys())
                        labels = list(new_labels.values())
                        
                        self.label_manager.add_sparse_labels(
                            frames=frames,
                            labels=labels,
                            source=f'active_learning_iter{iteration+1}',
                            confidences=confidences[list(range(len(frames)))].tolist()
                        )
                        
                        # Save manager
                        labels_dir = os.path.dirname(self.labels_csv)
                        manager_dir = os.path.join(labels_dir, 'LabelManager')
                        self.label_manager.save(manager_dir)
                        
                        # Update the per-frame CSV with ORIGINAL column name preserved
                        print(f"  Updating {os.path.basename(self.labels_csv)} with behavior name: {self.behavior_name}")
                        
                        # Read existing CSV
                        df_existing = pd.read_csv(self.labels_csv)
                        
                        # Update with new labels (sparse labels from this iteration)
                        for frame_idx, label in new_labels.items():
                            if frame_idx < len(df_existing):
                                df_existing.iloc[frame_idx, 0] = label
                        
                        # Save with original column name preserved
                        df_existing.to_csv(self.labels_csv, index=False)
                        
                        stats['final_positive'] = self.label_manager.metadata['positive_count']
                    else:
                        # Legacy mode: update per-frame CSV
                        for frame_idx, label in new_labels.items():
                            self.labels[frame_idx] = label
                            self.labels_df.iloc[frame_idx, 0] = label
                        
                        self.labels_df.to_csv(self.labels_csv, index=False)
                        stats['final_positive'] = int(self.labels.sum())
                    
                    print(f"  ✓ Saved to {self.labels_csv}")
                    
                    # Update stats
                    stats['frames_labeled'] += len(new_labels)
                    stats['iterations'] += 1
                else:
                    print("\n  No frames labeled, stopping.")
                    break
                    
            except Exception as e:
                print(f"\n✗ Error in iteration {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Final statistics
        print("\n" + "="*60)
        print("📊 SESSION COMPLETE")
        print("="*60)
        print(f"Total frames labeled: {stats['frames_labeled']}")
        print(f"Iterations completed: {stats['iterations']}")
        print(f"Positive labels: {stats['initial_positive']} → {stats['final_positive']}")
        print(f"Updated CSV: {self.labels_csv}")
        print("\nNext step: Retrain your classifier with the updated labels!")
        
        if self.use_smart_labels:
            self.label_manager.print_summary()
        
        return stats
    
    def _load_or_train_model(self, X_train, y_train):
        """Load existing model or train new one"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                print(f"  Loaded pickle file, type: {type(loaded_data)}")
                
                # Check if it's a PixelPaws classifier (dict) or raw model
                if isinstance(loaded_data, dict):
                    print(f"  Keys in dict: {list(loaded_data.keys())}")
                    
                    # Try different possible keys
                    model = None
                    for key in ['clf_model', 'model', 'classifier', 'clf', 'xgb_model']:
                        if key in loaded_data:
                            model = loaded_data[key]
                            print(f"  ✓ Extracted model from key: '{key}'")
                            break
                    
                    if model is None:
                        print("  ✗ Could not find model in dict keys")
                        print("  Training new model...")
                else:
                    # Raw XGBoost model
                    model = loaded_data
                    print("  ✓ Loaded raw XGBoost model")
                
                # Verify it has predict_proba
                if model is not None and not hasattr(model, 'predict_proba'):
                    print("  ✗ Loaded object doesn't have predict_proba")
                    print("  Training new model...")
                    model = None
                
            except Exception as e:
                print(f"  ✗ Error loading model: {e}")
                print("  Training new model instead...")
                model = None
        else:
            model = None
        
        # Train new model if needed
        if model is None:
            print("  Training new model...")
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            
            # Check if we have positive samples
            n_positive = (y_train == 1).sum()
            n_negative = (y_train == 0).sum()
            print(f"  Training data: {n_positive} positive, {n_negative} negative")
            
            if n_positive == 0:
                raise ValueError(
                    "No positive labels found!\n"
                    "Active learning needs at least some positive examples.\n"
                    "Please manually label some frames first."
                )
            
            if n_positive < 5:
                print(f"  ⚠ Warning: Only {n_positive} positive labels. Results may be poor.")
            
            model.fit(X_train, y_train)
            print("  ✓ Model trained")
        
        return model

        """
        Run active learning session.
        
        Args:
            n_suggestions: Number of frames to suggest per iteration
            n_iterations: Number of iterations to run
            
        Returns:
            Statistics about the session
        """
        print("\n" + "="*60)
        print("🧠 ACTIVE LEARNING SESSION")
        print("="*60)
        
        stats = {
            'frames_labeled': 0,
            'iterations': 0,
            'initial_positive': int(self.labels.sum()),
            'final_positive': int(self.labels.sum())
        }
        
        for iteration in range(n_iterations):
            print(f"\n📍 Iteration {iteration + 1} / {n_iterations}")
            print("-" * 60)
            
            try:
                # Load features
                print("Loading features...")
                if os.path.exists(self.features_cache):
                    with open(self.features_cache, 'rb') as f:
                        features = pickle.load(f)
                    print(f"  ✓ Loaded cached features: {features.shape}")
                else:
                    print(f"  ✗ Features not found!")
                    print(f"    Expected: {self.features_cache}")
                    raise FileNotFoundError(f"Features cache not found: {self.features_cache}")
                
                # Check features match labels
                if len(features) != len(self.labels):
                    # Features cover full video, labels cover subset
                    if len(features) > len(self.labels):
                        print(f"  Note: Features cover full video ({len(features)} frames)")
                        print(f"        Labels cover subset ({len(self.labels)} frames)")
                        print(f"  Using only the labeled portion for active learning")
                        
                        # Trim features to match labeled portion
                        features = features[:len(self.labels)]
                        print(f"  ✓ Using first {len(features)} frames")
                    else:
                        raise ValueError(
                            f"Feature count mismatch!\n"
                            f"Features: {len(features)} frames\n"
                            f"Labels: {len(self.labels)} frames\n"
                            f"Features should be >= labels.\n"
                            f"Make sure the features were extracted from the same video."
                        )
                
                # Train or load model
                print("Loading/training model...")
                if self.model_path and os.path.exists(self.model_path):
                    try:
                        with open(self.model_path, 'rb') as f:
                            loaded_data = pickle.load(f)
                        
                        print(f"  Loaded pickle file, type: {type(loaded_data)}")
                        
                        # Check if it's a PixelPaws classifier (dict) or raw model
                        if isinstance(loaded_data, dict):
                            print(f"  Keys in dict: {list(loaded_data.keys())}")
                            
                            # Try different possible keys
                            model = None
                            for key in ['clf_model', 'model', 'classifier', 'clf', 'xgb_model']:
                                if key in loaded_data:
                                    model = loaded_data[key]
                                    print(f"  ✓ Extracted model from key: '{key}'")
                                    break
                            
                            if model is None:
                                print("  ✗ Could not find model in dict keys")
                                print("  Training new model...")
                        else:
                            # Raw XGBoost model
                            model = loaded_data
                            print("  ✓ Loaded raw XGBoost model")
                        
                        # Verify it has predict_proba
                        if model is not None and not hasattr(model, 'predict_proba'):
                            print("  ✗ Loaded object doesn't have predict_proba")
                            print("  Training new model...")
                            model = None
                        
                    except Exception as e:
                        print(f"  ✗ Error loading model: {e}")
                        print("  Training new model instead...")
                        model = None
                else:
                    model = None
                
                # Train new model if needed
                if model is None:
                    print("  Training new model...")
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0
                    )
                    
                    # Check if we have positive samples
                    n_positive = self.labels.sum()
                    n_negative = (self.labels == 0).sum()
                    print(f"  Training data: {n_positive} positive, {n_negative} negative")
                    
                    if n_positive == 0:
                        raise ValueError(
                            "No positive labels found in CSV!\n"
                            "Active learning needs at least some positive examples.\n"
                            "Please manually label some frames first."
                        )
                    
                    if n_positive < 5:
                        print(f"  ⚠ Warning: Only {n_positive} positive labels. Results may be poor.")
                    
                    model.fit(features, self.labels)
                    print("  ✓ Model trained")
                
                # Find uncertain frames
                print(f"\nFinding {n_suggestions} uncertain frames...")
                uncertain_frames, confidences = self.engine.find_uncertain_frames(
                    model, features, self.labels, n_suggestions=n_suggestions
                )
                
                if len(uncertain_frames) == 0:
                    print("  ✗ No uncertain frames found!")
                    raise ValueError("Could not find any uncertain frames to suggest.")
                
                # Show labeling interface
                print("\nLaunching labeling interface...")
                interface = LabelingInterface(
                    self.video_path,
                    uncertain_frames,
                    confidences,
                    self.behavior_name
                )
                
                new_labels = interface.run()
                
                # Update labels CSV
                if new_labels:
                    print(f"\nUpdating labels CSV with {len(new_labels)} new labels...")
                    for frame_idx, label in new_labels.items():
                        self.labels[frame_idx] = label
                        self.labels_df.iloc[frame_idx, 0] = label
                    
                    # Save updated labels
                    self.labels_df.to_csv(self.labels_csv, index=False)
                    print(f"  ✓ Saved to {self.labels_csv}")
                    
                    # Update stats
                    stats['frames_labeled'] += len(new_labels)
                    stats['iterations'] += 1
                    stats['final_positive'] = int(self.labels.sum())
                else:
                    print("\n  No frames labeled, stopping.")
                    break
                    
            except Exception as e:
                print(f"\n✗ Error in iteration {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to show in GUI
        
        # Final statistics
        print("\n" + "="*60)
        print("📊 SESSION COMPLETE")
        print("="*60)
        print(f"Total frames labeled: {stats['frames_labeled']}")
        print(f"Iterations completed: {stats['iterations']}")
        print(f"Positive labels: {stats['initial_positive']} → {stats['final_positive']}")
        print(f"Updated CSV: {self.labels_csv}")
        print("\nNext step: Retrain your classifier with the updated labels!")
        
        return stats


def run_active_learning(labels_csv: str,
                       video_path: str,
                       dlc_path: str,
                       features_cache: str,
                       model_path: Optional[str] = None,
                       n_suggestions: int = 20,
                       n_iterations: int = 1,
                       extend_to_unlabeled: bool = True) -> dict:
    """
    Convenience function to run active learning.
    
    Args:
        labels_csv: Path to per-frame labels CSV
        video_path: Path to video file  
        dlc_path: Path to DLC tracking file
        features_cache: Path to cached features
        model_path: Path to trained model (optional)
        n_suggestions: Number of frames to suggest per iteration
        n_iterations: Number of iterations to run
        extend_to_unlabeled: If True, suggest from unlabeled regions
        
    Returns:
        Statistics dictionary
    """
    session = ActiveLearningSession(
        labels_csv=labels_csv,
        video_path=video_path,
        dlc_path=dlc_path,
        features_cache=features_cache,
        model_path=model_path,
        use_smart_labels=True  # Always use smart labels
    )
    
    return session.run(
        n_suggestions=n_suggestions,
        n_iterations=n_iterations,
        extend_to_unlabeled=extend_to_unlabeled
    )


def run_cross_video_active_learning(sessions: list,
                                    model,
                                    n_total: int = 100,
                                    min_frame_spacing: int = 30) -> dict:
    """
    Find the n_total most uncertain frames across ALL sessions using one
    shared model, then label them grouped by video.

    This is better than the per-video approach because:
    - Uses the real trained model (not a re-trained mini-model per video)
    - Budget is allocated globally — videos with more uncertainty get more frames
    - You see all borderline frames in one pass rather than N separate sessions

    Args:
        sessions: list of dicts, each with:
                    session_name   (str)
                    video_path     (str)
                    labels_csv     (str)  path to _labels.csv
                    features_cache (str)  path to resolved .pkl cache
                    behavior_name  (str)  column name in labels CSV
        model:   trained classifier with predict_proba()
        n_total: total frames to suggest across ALL videos (default 100)
        min_frame_spacing: min frames between suggestions within one video
                           (prevents clustering — keeps suggestions spread out)

    Returns:
        dict: frames_labeled, sessions_updated, per_session (name -> count)
    """
    print("\n" + "="*60)
    print("🧠 CROSS-VIDEO ACTIVE LEARNING")
    print("="*60)
    print(f"Sessions:  {len(sessions)}")
    print(f"Budget:    {n_total} frames total across all videos")
    print(f"Spacing:   ≥{min_frame_spacing} frames between picks within each video")

    # ── 1. Load features and score uncertainty for every session ──────
    all_uncertainty  = []
    all_proba        = []
    session_offsets  = []   # (flat_start, flat_end, session_dict)
    offset = 0

    for session in sessions:
        cache_path = session.get('features_cache', '')
        if not cache_path or not os.path.exists(cache_path):
            print(f"  ✗ Features cache not found for {session['session_name']}, skipping")
            continue

        print(f"\nLoading features: {session['session_name']}")
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        if isinstance(features, pd.DataFrame):
            features = features.values

        proba       = model.predict_proba(features)[:, 1]
        uncertainty = 1.0 - np.abs(proba - 0.5) * 2   # 1 at boundary, 0 at extremes

        n = len(proba)
        all_uncertainty.append(uncertainty)
        all_proba.append(proba)
        session_offsets.append((offset, offset + n, session))
        offset += n
        print(f"  {n:,} frames scored  |  mean uncertainty: {uncertainty.mean():.3f}  "
              f"|  frames near boundary (>0.8): {(uncertainty > 0.8).sum():,}")

    if not session_offsets:
        raise ValueError("No sessions with valid feature caches found.")

    flat_uncertainty = np.concatenate(all_uncertainty)
    flat_proba       = np.concatenate(all_proba)
    print(f"\nTotal frames scored: {len(flat_uncertainty):,}")

    # ── 2. Greedy global selection with per-video spacing ─────────────
    # Work on a copy; suppress within the owning video only after each pick.
    unc_work = flat_uncertainty.copy()
    selected = []   # global flat indices

    while len(selected) < n_total:
        best = int(np.argmax(unc_work))
        if unc_work[best] <= 0:
            break
        selected.append(best)

        # Suppress ±min_frame_spacing within this video only
        for start, end, _ in session_offsets:
            if start <= best < end:
                suppress_start = max(start,   best - min_frame_spacing)
                suppress_end   = min(end,     best + min_frame_spacing + 1)
                unc_work[suppress_start:suppress_end] = 0.0
                break

    print(f"\n✓ Selected {len(selected)} frames globally")

    # ── 3. Group selected frames back by session ──────────────────────
    session_frames = {}   # session_name -> {indices, proba, session}
    for flat_idx in selected:
        for start, end, session in session_offsets:
            if start <= flat_idx < end:
                local_idx = flat_idx - start
                name = session['session_name']
                if name not in session_frames:
                    session_frames[name] = {'indices': [], 'proba': [], 'session': session}
                session_frames[name]['indices'].append(local_idx)
                session_frames[name]['proba'].append(float(flat_proba[flat_idx]))
                break

    print("\nFrames per video:")
    for name, info in session_frames.items():
        print(f"  {name}: {len(info['indices'])} frames")

    # ── 4. Label each video ───────────────────────────────────────────
    stats = {
        'frames_labeled': 0,
        'sessions_updated': 0,
        'per_session': {}
    }

    for name, info in session_frames.items():
        session   = info['session']
        indices   = np.array(info['indices'])
        proba_arr = np.array(info['proba'])

        # Sort most-borderline first so the user sees the hardest frames early
        unc_arr    = 1.0 - np.abs(proba_arr - 0.5) * 2
        sort_order = np.argsort(-unc_arr)
        indices    = indices[sort_order]
        proba_arr  = proba_arr[sort_order]

        print(f"\n{'='*60}")
        print(f"Labeling: {name}  ({len(indices)} frames)")
        print(f"{'='*60}")

        behavior_name = session.get('behavior_name', 'Behavior')
        interface = LabelingInterface(
            video_path=session['video_path'],
            suggested_frames=indices,
            confidences=proba_arr,
            behavior_name=behavior_name
        )
        new_labels = interface.run()

        if new_labels:
            labels_csv = session['labels_csv']
            df = pd.read_csv(labels_csv)
            for frame_idx, label in new_labels.items():
                if frame_idx < len(df):
                    df.iloc[frame_idx, 0] = label
            df.to_csv(labels_csv, index=False)
            print(f"  ✓ Saved {len(new_labels)} labels → {os.path.basename(labels_csv)}")
            stats['frames_labeled']   += len(new_labels)
            stats['sessions_updated'] += 1
            stats['per_session'][name] = len(new_labels)
        else:
            print(f"  No frames labeled for {name}")
            stats['per_session'][name] = 0

    print(f"\n{'='*60}")
    print(f"📊 CROSS-VIDEO AL COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames labeled: {stats['frames_labeled']}")
    print(f"Sessions updated:     {stats['sessions_updated']}/{len(session_frames)}")

    return stats


def run_batch_active_learning(root_folder: str,
                              n_suggestions: int = 20,
                              n_iterations: int = 1,
                              extend_to_unlabeled: bool = True,
                              search_parents: bool = True) -> dict:
    """
    Run active learning on all sessions found in a folder structure.
    
    Automatically scans subfolders AND parent/sibling folders to find matching 
    video/DLC/label files.
    
    Args:
        root_folder: Starting directory to scan
        n_suggestions: Number of frames to suggest per iteration per session
        n_iterations: Number of iterations per session
        extend_to_unlabeled: If True, suggest from unlabeled regions
        search_parents: If True, also search parent directory and siblings
        
    Returns:
        Dictionary with results for each session
    """
    print("="*60)
    print("🚀 BATCH ACTIVE LEARNING")
    print("="*60)
    print(f"Root folder: {root_folder}")
    print(f"Search parents: {'Yes' if search_parents else 'No'}\n")
    
    # Scan for sessions (including parent/sibling folders)
    sessions = scan_folders_for_sessions(root_folder, search_parents=search_parents)
    
    if not sessions:
        print("❌ No sessions found!")
        print("\nMake sure your folder contains:")
        print("  - Video files (.mp4, .avi)")
        print("  - DLC files (.h5)")
        print("  - Labels files (_labels.csv or .csv)")
        return {}
    
    # Ask user which sessions to process
    print(f"\nFound {len(sessions)} session(s).")
    response = messagebox.askyesno(
        "Batch Active Learning",
        f"Found {len(sessions)} session(s) in:\n{root_folder}\n\n"
        f"Process all sessions?\n\n"
        f"Settings:\n"
        f"  • {n_suggestions} frames per iteration\n"
        f"  • {n_iterations} iteration(s) per session"
    )
    
    if not response:
        print("❌ Cancelled by user")
        return {}
    
    # Process each session
    results = {}
    
    for i, session_info in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"📍 Session {i}/{len(sessions)}: {session_info['base_name']}")
        print(f"{'='*60}")
        
        try:
            stats = run_active_learning(
                labels_csv=session_info['labels_path'],
                video_path=session_info['video_path'],
                dlc_path=session_info['dlc_path'],
                features_cache=session_info['features_path'],
                n_suggestions=n_suggestions,
                n_iterations=n_iterations,
                extend_to_unlabeled=extend_to_unlabeled
            )
            
            results[session_info['base_name']] = {
                'success': True,
                'stats': stats
            }
            
        except Exception as e:
            print(f"\n❌ Error processing {session_info['base_name']}: {e}")
            results[session_info['base_name']] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 BATCH SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    print(f"✓ Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(results)}")
        print("\nFailed sessions:")
        for name, result in results.items():
            if not result['success']:
                print(f"  • {name}: {result['error']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Active Learning Module for PixelPaws")
    print("Import this module and call run_active_learning()")
