"""
PixelPaws Predict Tab Module
Handles single video prediction interface and logic
"""

import os
import sys
import pickle
import glob
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml


class PredictTab:
    """Predict tab for single video behavioral prediction"""
    
    def __init__(self, parent_notebook, parent_app):
        """
        Initialize Predict tab
        
        Args:
            parent_notebook: ttk.Notebook to add tab to
            parent_app: Main GUI application instance (for shared resources)
        """
        self.notebook = parent_notebook
        self.app = parent_app
        
        # Create tab
        self.frame = ttk.Frame(self.notebook)
        self.notebook.add(self.frame, text="🎬 Predict")
        
        # Initialize variables
        self.pred_classifier_path = tk.StringVar()
        self.pred_video_path = tk.StringVar()
        self.pred_dlc_path = tk.StringVar()
        self.pred_features_path = tk.StringVar()
        self.pred_dlc_config_path = tk.StringVar()
        self.pred_human_labels_path = tk.StringVar()
        self.pred_output_folder = tk.StringVar()
        
        self.pred_save_csv = tk.BooleanVar(value=True)
        self.pred_save_video = tk.BooleanVar(value=False)
        self.pred_save_summary = tk.BooleanVar(value=True)
        self.pred_generate_ethogram = tk.BooleanVar(value=False)
        
        # Build UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the Predict tab UI"""
        # Create scrollable canvas
        canvas = tk.Canvas(self.frame)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === CLASSIFIER SELECTION ===
        clf_frame = ttk.LabelFrame(scrollable_frame, text="Classifier", padding=10)
        clf_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(clf_frame, text="Classifier File:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(clf_frame, textvariable=self.pred_classifier_path, width=50).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(clf_frame, text="📁 Browse", 
                  command=self.browse_classifier).grid(row=0, column=2, pady=2)
        
        ttk.Button(clf_frame, text="📋 View Classifier Info", 
                  command=self.view_classifier_info).grid(row=1, column=1, sticky='w', pady=5)
        
        # === VIDEO SELECTION ===
        video_frame = ttk.LabelFrame(scrollable_frame, text="Video Files", padding=10)
        video_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(video_frame, text="Video File:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(video_frame, textvariable=self.pred_video_path, width=50).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_video).grid(row=0, column=2, pady=2)
        
        ttk.Label(video_frame, text="DLC Pose File:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(video_frame, textvariable=self.pred_dlc_path, width=50).grid(
            row=1, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_dlc).grid(row=1, column=2, pady=2)
        
        ttk.Button(video_frame, text="🔍 Auto-Find DLC File", 
                  command=self.auto_find_dlc).grid(row=2, column=1, sticky='w', pady=5)
        
        # Features file (optional)
        ttk.Label(video_frame, text="Features File (optional):").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(video_frame, textvariable=self.pred_features_path, width=50).grid(
            row=3, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_features).grid(row=3, column=2, pady=2)
        
        ttk.Label(video_frame, text="Skip feature extraction if file provided", 
                 font=('Arial', 8), foreground='gray').grid(row=3, column=1, sticky='e', padx=5)
        
        # DLC Config for crop parameters
        ttk.Label(video_frame, text="DLC Config (for crop):").grid(row=4, column=0, sticky='w', pady=2)
        ttk.Entry(video_frame, textvariable=self.pred_dlc_config_path, width=50).grid(
            row=4, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_dlc_config).grid(row=4, column=2, pady=2)
        
        ttk.Button(video_frame, text="🔍 Auto-Find Config", 
                  command=self.auto_find_dlc_config).grid(row=5, column=1, sticky='w', pady=5)
        
        # Human labels (optional)
        ttk.Label(video_frame, text="Human Labels (optional):").grid(row=6, column=0, sticky='w', pady=2)
        ttk.Entry(video_frame, textvariable=self.pred_human_labels_path, width=50).grid(
            row=6, column=1, padx=5, pady=2)
        ttk.Button(video_frame, text="📁 Browse", 
                  command=self.browse_human_labels).grid(row=6, column=2, pady=2)
        
        # === OUTPUT OPTIONS ===
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Options", padding=10)
        output_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(output_frame, text="Save frame-by-frame predictions (CSV)", 
                       variable=self.pred_save_csv).grid(row=0, column=0, sticky='w', pady=2)
        
        ttk.Checkbutton(output_frame, text="Create labeled video (slower)", 
                       variable=self.pred_save_video).grid(row=1, column=0, sticky='w', pady=2)
        
        ttk.Checkbutton(output_frame, text="Save behavior summary statistics", 
                       variable=self.pred_save_summary).grid(row=2, column=0, sticky='w', pady=2)
        
        ttk.Checkbutton(output_frame, text="Generate ethogram plots", 
                       variable=self.pred_generate_ethogram).grid(row=3, column=0, sticky='w', pady=2)
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=4, column=0, sticky='w', pady=5)
        ttk.Entry(output_frame, textvariable=self.pred_output_folder, width=40).grid(
            row=4, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="📁 Browse", 
                  command=self.browse_output_folder).grid(row=4, column=2, pady=5)
        
        ttk.Label(output_frame, text="(Leave empty to use video folder)", 
                 font=('Arial', 8), foreground='gray').grid(row=5, column=1, sticky='w')
        
        # === ACTION BUTTONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(action_frame, text="▶ RUN PREDICTION", 
                  command=self.run_prediction, 
                  style='Accent.TButton').pack(side='left', padx=5)
        
        # === RESULTS DISPLAY ===
        results_frame = ttk.LabelFrame(scrollable_frame, text="Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    # === BROWSE METHODS ===
    
    def browse_classifier(self):
        """Browse for classifier file"""
        filepath = filedialog.askopenfilename(
            title="Select Classifier File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_classifier_path.set(filepath)
    
    def browse_video(self):
        """Browse for video file"""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_video_path.set(filepath)
    
    def browse_dlc(self):
        """Browse for DLC file"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Pose File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_dlc_path.set(filepath)
    
    def browse_features(self):
        """Browse for pre-extracted features file"""
        filepath = filedialog.askopenfilename(
            title="Select Features File (Optional)",
            filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_features_path.set(filepath)
    
    def browse_dlc_config(self):
        """Browse for DLC config.yaml file"""
        filepath = filedialog.askopenfilename(
            title="Select DLC Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_dlc_config_path.set(filepath)
    
    def browse_human_labels(self):
        """Browse for human labels file"""
        filepath = filedialog.askopenfilename(
            title="Select Human Labels File (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_human_labels_path.set(filepath)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.pred_output_folder.set(folder)
    
    # === AUTO-FIND METHODS ===
    
    def auto_find_dlc(self):
        """Automatically find DLC file for selected video"""
        video_path = self.pred_video_path.get()
        if not video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        
        video_folder = os.path.dirname(video_path)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        
        # Look for matching DLC file
        dlc_files = glob.glob(os.path.join(video_folder, f"{video_base}DLC*.h5"))
        
        if not dlc_files:
            messagebox.showwarning("Not Found", "No matching DLC file found.")
            return
        
        # Prefer filtered if available
        filtered = [f for f in dlc_files if 'filtered' in f.lower()]
        if filtered:
            self.pred_dlc_path.set(filtered[0])
        else:
            self.pred_dlc_path.set(dlc_files[0])
        
        messagebox.showinfo("Found", f"Found DLC file:\n{os.path.basename(self.pred_dlc_path.get())}")
    
    def auto_find_dlc_config(self):
        """Automatically find DLC config.yaml file"""
        video_path = self.pred_video_path.get()
        if not video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        
        video_folder = os.path.dirname(video_path)
        
        # Search locations
        search_paths = [
            os.path.join(video_folder, 'config.yaml'),
            os.path.join(os.path.dirname(video_folder), 'config.yaml'),
        ]
        
        # Also search subdirectories
        for root, dirs, files in os.walk(video_folder):
            if 'config.yaml' in files:
                search_paths.append(os.path.join(root, 'config.yaml'))
            # Limit depth
            if root.count(os.sep) - video_folder.count(os.sep) > 2:
                break
        
        # Find first existing config
        for path in search_paths:
            if os.path.isfile(path):
                self.pred_dlc_config_path.set(path)
                messagebox.showinfo("Found", f"Found config.yaml:\n{path}")
                return
        
        messagebox.showwarning("Not Found", 
                             "No config.yaml found in video directory or parent directories.")
    
    # === INFO METHODS ===
    
    def view_classifier_info(self):
        """Display classifier information"""
        clf_path = self.pred_classifier_path.get()
        if not clf_path or not os.path.isfile(clf_path):
            messagebox.showwarning("No Classifier", "Please select a valid classifier file.")
            return
        
        try:
            with open(clf_path, 'rb') as f:
                clf_data = pickle.load(f)
            
            # Build info string
            info = "Classifier Information:\n"
            info += "="*60 + "\n\n"
            
            if 'Behavior_type' in clf_data:
                info += f"Behavior: {clf_data['Behavior_type']}\n"
            if 'best_thresh' in clf_data:
                info += f"Threshold: {clf_data['best_thresh']:.3f}\n"
            if 'min_bout' in clf_data:
                info += f"Min Bout: {clf_data['min_bout']} frames\n"
            if 'bp_include_list' in clf_data:
                info += f"\nBody Parts: {', '.join(clf_data['bp_include_list'])}\n"
            if 'bp_pixbrt_list' in clf_data:
                info += f"Brightness Body Parts: {', '.join(clf_data['bp_pixbrt_list'])}\n"
            
            # Model info
            if 'clf_model' in clf_data:
                model = clf_data['clf_model']
                if hasattr(model, 'n_features_in_'):
                    info += f"\nModel Features: {model.n_features_in_}\n"
            
            messagebox.showinfo("Classifier Info", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier:\n{e}")
    
    # === PREDICTION METHOD ===
    
    def run_prediction(self):
        """Run prediction on single video"""
        # Validate inputs
        if not self.pred_classifier_path.get():
            messagebox.showwarning("No Classifier", "Please select a classifier file.")
            return
        if not self.pred_video_path.get():
            messagebox.showwarning("No Video", "Please select a video file.")
            return
        if not self.pred_dlc_path.get():
            messagebox.showwarning("No DLC File", "Please select a DLC pose file.")
            return
        
        # Run in thread
        threading.Thread(target=self._predict_thread, daemon=True).start()
    
    def _log(self, text):
        """Thread-safe logging to results_text widget."""
        try:
            self.app.root.after(0, lambda t=text: self.results_text.insert(tk.END, t))
        except Exception:
            pass

    def _log_clear(self):
        """Thread-safe clear of results_text widget."""
        try:
            self.app.root.after(0, lambda: self.results_text.delete('1.0', tk.END))
        except Exception:
            pass

    def _predict_thread(self):
        """Prediction thread - imports main GUI functions"""
        # Import here to avoid circular imports
        try:
            from PixelPaws_GUI import (
                PixelPaws_ExtractFeatures,
                predict_with_xgboost,
                clean_bodyparts_list,
                auto_detect_bodyparts_from_model,
            )
        except ImportError:
            self.app.root.after(0, lambda: messagebox.showerror("Import Error",
                               "Could not import required functions from main GUI.\n"
                               "Make sure PixelPaws_GUI.py is in the same directory."))
            return

        try:
            self._log_clear()
            self._log("=" * 60 + "\n")
            self._log("PixelPaws Prediction\n")
            self._log("=" * 60 + "\n\n")

            # Get paths
            clf_path = self.pred_classifier_path.get()
            video_path = self.pred_video_path.get()
            dlc_path = self.pred_dlc_path.get()
            features_path = self.pred_features_path.get()
            dlc_config_path = self.pred_dlc_config_path.get()

            self._log(f"Classifier: {os.path.basename(clf_path)}\n")
            self._log(f"Video: {os.path.basename(video_path)}\n")
            self._log(f"DLC File: {os.path.basename(dlc_path)}\n")
            if features_path:
                self._log(f"Features: {os.path.basename(features_path)}\n")
            if dlc_config_path:
                self._log(f"DLC Config: {os.path.basename(dlc_config_path)}\n")
            self._log("\n")

            # Load classifier
            self._log("Loading classifier...\n")
            try:
                with open(clf_path, 'rb') as f:
                    clf_data = pickle.load(f)
            except Exception as e:
                self._log(f"✗ Failed to load classifier: {e}\n")
                return

            # Clean body parts
            clf_data['bp_include_list'] = clean_bodyparts_list(clf_data.get('bp_include_list', []))
            clf_data['bp_pixbrt_list'] = clean_bodyparts_list(clf_data.get('bp_pixbrt_list', []))

            # Auto-detect bodyparts
            clf_data = auto_detect_bodyparts_from_model(clf_data, verbose=True)

            model = clf_data['clf_model']
            best_thresh = clf_data['best_thresh']
            behavior_name = clf_data.get('Behavior_type', 'Behavior')

            self._log(f"  Behavior: {behavior_name}\n")
            self._log(f"  Threshold: {best_thresh:.3f}\n\n")

            # Check for DLC crop
            crop_x_offset = 0
            crop_y_offset = 0
            if dlc_config_path and os.path.isfile(dlc_config_path):
                self._log("Checking DLC crop parameters...\n")
                try:
                    with open(dlc_config_path, 'r') as f:
                        config = yaml.safe_load(f)

                    if config.get('cropping', False):
                        crop_x_offset = config.get('x1', 0)
                        crop_y_offset = config.get('y1', 0)
                        self._log(
                            f"  ✓ DLC crop detected: x+{crop_x_offset}, y+{crop_y_offset}\n\n")
                    else:
                        self._log("  No cropping in config\n\n")
                except Exception as e:
                    self._log(f"  ⚠️  Could not read config: {e}\n\n")

            # Try to load features
            X = None
            features_loaded = False
            video_dir = os.path.dirname(video_path)

            if features_path and os.path.isfile(features_path):
                self._log("Loading pre-extracted features...\n")
                try:
                    with open(features_path, 'rb') as f:
                        features_data = pickle.load(f)

                    if isinstance(features_data, dict) and 'X' in features_data:
                        X = features_data['X']
                    else:
                        X = features_data

                    features_loaded = True
                    self._log(
                        f"  ✓ Loaded: {X.shape[0]} frames, {X.shape[1]} features\n\n")

                    if crop_x_offset != 0 or crop_y_offset != 0:
                        self._log(
                            f"  ⚠️  Using pre-extracted features with detected crop\n")
                        self._log(
                            f"     Make sure features were extracted with correct coordinates!\n\n")

                except Exception as e:
                    self._log(f"  ✗ Error loading: {e}\n")
                    self._log("  Falling back to extraction...\n\n")
                    features_loaded = False

            # Extract if needed
            if not features_loaded:
                self._log("Extracting features...\n")
                self._log("  (This may take several minutes)\n")

                X = PixelPaws_ExtractFeatures(
                    pose_data_file=dlc_path,
                    video_file_path=video_path,
                    bp_include_list=clf_data.get('bp_include_list'),
                    bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
                    square_size=clf_data.get('square_size', [40]),
                    pix_threshold=clf_data.get('pix_threshold', 0.3),
                )

                self._log(f"  ✓ Features extracted\n\n")

            # Predict
            self._log("Running classifier...\n")
            y_proba = predict_with_xgboost(model, X)
            y_pred = (y_proba >= best_thresh).astype(int)

            # Apply bout filtering
            if 'min_bout' in clf_data:
                self._log("Applying bout filtering...\n")
                y_pred = self._apply_bout_filtering(
                    y_pred,
                    clf_data.get('min_bout', 1),
                    clf_data.get('min_after_bout', 1),
                    clf_data.get('max_gap', 0)
                )

            # Calculate statistics
            n_frames = len(y_pred)
            n_positive = np.sum(y_pred)

            # Get FPS
            import cv2
            cap = cv2.VideoCapture(video_path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
            if not fps or fps <= 0:
                fps = 30.0
                self._log("Warning: video reported FPS=0, defaulting to 30\n")

            behavior_time = n_positive / fps

            # Display results
            self._log("\n" + "="*60 + "\n")
            self._log("RESULTS\n")
            self._log("="*60 + "\n")
            self._log(f"Total frames: {n_frames}\n")
            self._log(f"Behavior detected: {n_positive} frames ({100*n_positive/n_frames:.1f}%)\n")
            self._log(f"Behavior time: {behavior_time:.1f} seconds ({behavior_time/60:.1f} minutes)\n")
            self._log("="*60 + "\n")

            # Save outputs
            self._save_outputs(video_path, video_dir, behavior_name, y_pred, y_proba, fps)

        except Exception as e:
            import traceback
            self._log("\n" + "="*60 + "\n")
            self._log("✗ ERROR\n")
            self._log("="*60 + "\n")
            self._log(f"{traceback.format_exc()}\n")
    
    def _apply_bout_filtering(self, y_pred, min_bout, min_after_bout, max_gap):
        """Apply bout filtering to predictions"""
        y_filtered = y_pred.copy()
        
        # Remove short bouts
        if min_bout > 1:
            i = 0
            while i < len(y_filtered):
                if y_filtered[i] == 1:
                    bout_start = i
                    while i < len(y_filtered) and y_filtered[i] == 1:
                        i += 1
                    bout_length = i - bout_start
                    if bout_length < min_bout:
                        y_filtered[bout_start:i] = 0
                else:
                    i += 1
        
        # Fill short gaps
        if max_gap > 0:
            i = 0
            while i < len(y_filtered):
                if y_filtered[i] == 0:
                    gap_start = i
                    while i < len(y_filtered) and y_filtered[i] == 0:
                        i += 1
                    
                    gap_length = i - gap_start
                    if gap_length > 0 and gap_length <= max_gap and i < len(y_filtered):
                        if y_filtered[i] == 1:
                            y_filtered[gap_start:i] = 1
                else:
                    i += 1
        
        return y_filtered
    
    def _save_outputs(self, video_path, video_dir, behavior_name, y_pred, y_proba, fps):
        """Save prediction outputs"""
        output_dir = self.pred_output_folder.get() or video_dir
        os.makedirs(output_dir, exist_ok=True)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        
        self._log(f"\nSaving outputs to: {output_dir}\n")

        # Save CSV
        if self.pred_save_csv.get():
            csv_path = os.path.join(output_dir, f"{video_base}_predictions.csv")
            df = pd.DataFrame({
                'frame': range(len(y_pred)),
                'prediction': y_pred,
                'probability': y_proba
            })
            df.to_csv(csv_path, index=False)
            self._log(f"  ✓ Saved CSV: {os.path.basename(csv_path)}\n")

        self._log("\n✅ Prediction complete!\n")
