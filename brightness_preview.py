#!/usr/bin/env python3
"""
Brightness Preview Tool
Shows extraction area and brightness values from video frames
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class BrightnessPreview:
    def __init__(self, root, video_path=None, features_path=None):
        self.root = root
        self.root.title("Brightness Preview")
        self.root.geometry("500x650")
        
        self.video_path = video_path
        self.features_path = features_path
        self.dlc_path = None
        self.dlc_df = None
        self.crop_offset_x = 0  # DLC crop offset
        self.crop_offset_y = 0
        
        self.cap = None
        self.features_df = None
        self.brightness_features = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        self.graph_win = None
        self.video_win = None
        
        self.setup_ui()
        
        if video_path and not features_path:
            self.auto_detect_features()
        
        if self.video_path and self.features_path:
            self.load_data()
    
    def setup_ui(self):
        """Create UI elements"""
        # Header
        ttk.Label(self.root, text="Brightness Preview", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Files section
        files_frame = ttk.LabelFrame(self.root, text="Files", padding=5)
        files_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(files_frame, text="Select Video", 
                  command=self.select_video).grid(row=0, column=0, padx=2, sticky='w')
        self.video_lbl = ttk.Label(files_frame, text="No video", foreground='gray')
        self.video_lbl.grid(row=0, column=1, padx=5, sticky='w')
        
        ttk.Button(files_frame, text="Select Features",
                  command=self.select_features).grid(row=1, column=0, padx=2, sticky='w')
        self.feat_lbl = ttk.Label(files_frame, text="Auto-detect", foreground='gray')
        self.feat_lbl.grid(row=1, column=1, padx=5, sticky='w')
        
        ttk.Button(files_frame, text="Select DLC File (optional)",
                  command=self.select_dlc).grid(row=2, column=0, padx=2, sticky='w')
        self.dlc_lbl = ttk.Label(files_frame, text="For pose overlay", foreground='gray')
        self.dlc_lbl.grid(row=2, column=1, padx=5, sticky='w')
        
        ttk.Button(files_frame, text="Select Config (optional)",
                  command=self.select_config).grid(row=3, column=0, padx=2, sticky='w')
        self.config_lbl = ttk.Label(files_frame, text="For DLC crop offset", foreground='gray')
        self.config_lbl.grid(row=3, column=1, padx=5, sticky='w')
        
        # Store config path
        self.config_path = None
        
        # Info section
        info_frame = ttk.LabelFrame(self.root, text="Feature Info", padding=5)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.info_txt = tk.Text(info_frame, height=5, wrap='word', 
                               font=('Courier', 8), state='disabled')
        self.info_txt.pack(fill='x')
        
        # Controls section
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls", padding=5)
        ctrl_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(ctrl_frame, text="Feature:").grid(row=0, column=0, sticky='w', padx=2)
        self.feature_var = tk.StringVar()
        self.feature_cb = ttk.Combobox(ctrl_frame, textvariable=self.feature_var, 
                                      state='readonly', width=30)
        self.feature_cb.grid(row=0, column=1, columnspan=2, sticky='ew', padx=2)
        self.feature_cb.bind('<<ComboboxSelected>>', lambda e: self.on_feature_change())
        
        ttk.Label(ctrl_frame, text="Radius:").grid(row=1, column=0, sticky='w', padx=2)
        self.radius_var = tk.IntVar(value=25)
        self.radius_scale = ttk.Scale(ctrl_frame, from_=10, to=100, 
                                     variable=self.radius_var, orient='horizontal',
                                     command=lambda x: self.update_radius_lbl())
        self.radius_scale.grid(row=1, column=1, sticky='ew', padx=2)
        self.radius_lbl = ttk.Label(ctrl_frame, text="25 px", width=6)
        self.radius_lbl.grid(row=1, column=2, sticky='w', padx=2)
        
        ttk.Label(ctrl_frame, text="Frame:").grid(row=2, column=0, sticky='w', padx=2)
        self.frame_var = tk.IntVar(value=0)
        self.frame_spin = ttk.Spinbox(ctrl_frame, from_=0, to=0, 
                                     textvariable=self.frame_var, width=10,
                                     command=self.update_display)
        self.frame_spin.grid(row=2, column=1, sticky='w', padx=2)
        self.frame_info_lbl = ttk.Label(ctrl_frame, text="")
        self.frame_info_lbl.grid(row=2, column=2, sticky='w', padx=2)
        
        # Crop offset controls
        ttk.Label(ctrl_frame, text="X Offset:").grid(row=3, column=0, sticky='w', padx=2)
        self.offset_x_var = tk.IntVar(value=0)
        self.offset_x_spin = ttk.Spinbox(ctrl_frame, from_=-500, to=500,
                                        textvariable=self.offset_x_var, width=10,
                                        command=self.update_offset)
        self.offset_x_spin.grid(row=3, column=1, sticky='w', padx=2)
        ttk.Label(ctrl_frame, text="(DLC crop)").grid(row=3, column=2, sticky='w', padx=2)
        
        ttk.Label(ctrl_frame, text="Y Offset:").grid(row=4, column=0, sticky='w', padx=2)
        self.offset_y_var = tk.IntVar(value=0)
        self.offset_y_spin = ttk.Spinbox(ctrl_frame, from_=-500, to=500,
                                        textvariable=self.offset_y_var, width=10,
                                        command=self.update_offset)
        self.offset_y_spin.grid(row=4, column=1, sticky='w', padx=2)
        ttk.Label(ctrl_frame, text="(DLC crop)").grid(row=4, column=2, sticky='w', padx=2)
        
        ctrl_frame.columnconfigure(1, weight=1)
        
        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Show Graph", 
                  command=self.show_graph).pack(side='left', padx=2, expand=True, fill='x')
        ttk.Button(btn_frame, text="Show Video",
                  command=self.show_video).pack(side='left', padx=2, expand=True, fill='x')
        ttk.Button(btn_frame, text="Update",
                  command=self.update_display).pack(side='left', padx=2, expand=True, fill='x')
        
        # Status
        self.status_lbl = ttk.Label(self.root, text="Select video to begin", 
                                   relief='sunken', anchor='w')
        self.status_lbl.pack(fill='x', side='bottom')
        
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
    
    def update_radius_lbl(self):
        self.radius_lbl.config(text=f"{self.radius_var.get()} px")
        if self.features_df is not None:
            self.update_display()
    
    def update_offset(self):
        self.crop_offset_x = self.offset_x_var.get()
        self.crop_offset_y = self.offset_y_var.get()
        if self.features_df is not None:
            self.update_display()
    
    def detect_dlc_crop(self):
        """Try to detect DLC crop parameters from config.yaml or .h5 metadata"""
        if not self.video_path:
            return
        
        # Priority 1: User-selected config.yaml
        if self.config_path and os.path.isfile(self.config_path):
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config.get('cropping', False):
                    x1 = config.get('x1', 0)
                    y1 = config.get('y1', 0)
                    if x1 != 0 or y1 != 0:
                        self.crop_offset_x = x1
                        self.crop_offset_y = y1
                        self.offset_x_var.set(x1)
                        self.offset_y_var.set(y1)
                        print(f"[CROP] ✓ Loaded from user config: x1={x1}, y1={y1}")
                        self.status_lbl.config(text=f"Crop from config: x1={x1}, y1={y1}")
                        return
            except ImportError:
                print(f"[CROP] PyYAML not installed - install with: pip install pyyaml")
            except Exception as e:
                print(f"[CROP] Could not read config: {e}")
        
        # Priority 2: Try to read from DLC .h5 file metadata
        if self.dlc_path:
            try:
                import tables
                with tables.open_file(self.dlc_path, 'r') as h5:
                    # Check for metadata
                    if hasattr(h5.root, '_v_attrs'):
                        attrs = h5.root._v_attrs
                        if hasattr(attrs, 'cropping'):
                            print(f"[CROP] DLC file has cropping metadata: {attrs.cropping}")
                        if hasattr(attrs, 'x1'):
                            x1 = attrs.x1
                            self.crop_offset_x = x1
                            self.offset_x_var.set(x1)
                            print(f"[CROP] Found x1={x1} in DLC metadata")
                        if hasattr(attrs, 'y1'):
                            y1 = attrs.y1
                            self.crop_offset_y = y1
                            self.offset_y_var.set(y1)
                            print(f"[CROP] Found y1={y1} in DLC metadata")
                        
                        if self.crop_offset_x != 0 or self.crop_offset_y != 0:
                            self.status_lbl.config(text=f"Crop from DLC: x1={self.crop_offset_x}, y1={self.crop_offset_y}")
                            return
            except Exception as e:
                print(f"[CROP] Could not read DLC metadata: {e}")
        
        # Priority 3: Fall back to auto-detect config.yaml
        vdir = os.path.dirname(self.video_path)
        
        # Common DLC config locations
        config_paths = [
            os.path.join(vdir, 'config.yaml'),
            os.path.join(os.path.dirname(vdir), 'config.yaml'),
        ]
        
        # Also check for project folders
        for item in os.listdir(vdir):
            item_path = os.path.join(vdir, item)
            if os.path.isdir(item_path):
                config_paths.append(os.path.join(item_path, 'config.yaml'))
        
        for config_path in config_paths:
            if os.path.isfile(config_path):
                try:
                    with open(config_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith('x1:'):
                                x1 = int(line.split(':')[1].strip())
                                self.crop_offset_x = x1
                                self.offset_x_var.set(x1)
                                print(f"[CROP] Detected x1={x1} from {config_path}")
                            elif line.strip().startswith('y1:'):
                                y1 = int(line.split(':')[1].strip())
                                self.crop_offset_y = y1
                                self.offset_y_var.set(y1)
                                print(f"[CROP] Detected y1={y1} from {config_path}")
                    
                    if self.crop_offset_x != 0 or self.crop_offset_y != 0:
                        self.status_lbl.config(text=f"Crop detected: x1={self.crop_offset_x}, y1={self.crop_offset_y}")
                        return
                except:
                    pass
    
    def check_if_features_are_corrected(self):
        """Check if features file already has corrected (full-frame) coordinates"""
        if not self.features_df is not None:
            return
        
        # Look for pose columns
        pose_cols = [c for c in self.features_df.columns if '_x' in c.lower() or '_y' in c.lower()]
        
        if not pose_cols:
            print("[FEATURES] No pose coordinates in features file")
            return
        
        # Get video dimensions
        if self.cap:
            vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Check a sample of coordinates
            sample_size = min(100, len(self.features_df))
            for col in pose_cols[:4]:  # Check first few columns
                sample_vals = self.features_df[col].dropna().head(sample_size)
                if len(sample_vals) > 0:
                    max_val = sample_vals.max()
                    
                    if '_x' in col.lower():
                        if max_val > vid_w:
                            print(f"[FEATURES] ⚠️ {col} max={max_val:.1f} > video width {vid_w}")
                            print(f"[FEATURES] Features may have UNCORRECTED crop coordinates!")
                        elif max_val < (vid_w - self.crop_offset_x):
                            print(f"[FEATURES] ✓ {col} max={max_val:.1f} < cropped width {vid_w - self.crop_offset_x}")
                            print(f"[FEATURES] Features likely have CROPPED coordinates (need offset)")
                        else:
                            print(f"[FEATURES] {col} max={max_val:.1f}, video width={vid_w}")
                            print(f"[FEATURES] Features may already be CORRECTED (full-frame)")
                    
                    elif '_y' in col.lower():
                        if max_val > vid_h:
                            print(f"[FEATURES] ⚠️ {col} max={max_val:.1f} > video height {vid_h}")
    
    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Videos", "*.mp4 *.avi"), ("All", "*.*")])
        if path:
            self.video_path = path
            self.video_lbl.config(text=os.path.basename(path), foreground='black')
            if self.auto_detect_features():
                self.load_data()
    
    def select_features(self):
        path = filedialog.askopenfilename(
            title="Select Features",
            filetypes=[("Pickle", "*.pickle *.pkl"), ("All", "*.*")],
            initialdir=os.path.dirname(self.video_path) if self.video_path else None)
        if path:
            self.features_path = path
            self.feat_lbl.config(text=os.path.basename(path), foreground='black')
            if self.video_path:
                self.load_data()
    
    def select_dlc(self):
        """Select DLC .h5 file for pose coordinates"""
        path = filedialog.askopenfilename(
            title="Select DLC Pose File",
            filetypes=[("HDF5", "*.h5"), ("All", "*.*")],
            initialdir=os.path.dirname(self.video_path) if self.video_path else None)
        if path:
            self.dlc_path = path
            self.dlc_lbl.config(text=os.path.basename(path), foreground='black')
            self.load_dlc_file()
            # Try to detect crop from DLC metadata
            self.detect_dlc_crop()
            # Reload display if already loaded
            if self.features_df is not None:
                self.update_display()
            self.status_lbl.config(text=f"✓ DLC file loaded: {os.path.basename(path)}")
    
    def select_config(self):
        """Select DLC config.yaml file for crop parameters"""
        path = filedialog.askopenfilename(
            title="Select DLC Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All", "*.*")],
            initialdir=os.path.dirname(self.video_path) if self.video_path else None)
        if path:
            self.config_path = path
            self.config_lbl.config(text=os.path.basename(path), foreground='black')
            # Try to load crop from config
            self.detect_dlc_crop()
            # Reload display if already loaded
            if self.features_df is not None:
                self.update_display()
            self.status_lbl.config(text=f"✓ Config loaded: {os.path.basename(path)}")
    
    def auto_detect_features(self):
        if not self.video_path:
            return False
        
        vdir = os.path.dirname(self.video_path)
        vbase = os.path.splitext(os.path.basename(self.video_path))[0]
        
        # Remove DLC suffixes
        for sfx in ['DLC', '_labeled', '_filtered', 'DLC_resnet', 'DLC_dlcrnetms5']:
            if sfx in vbase:
                vbase = vbase.split(sfx)[0]
                break
        
        # Check cache subdirectories (canonical first, then legacy names)
        patterns = []
        for cache_name in ('features', 'FeatureCache', 'PredictionCache'):
            cdir = os.path.join(vdir, cache_name)
            if os.path.isdir(cdir):
                patterns.extend([
                    os.path.join(cdir, f"{vbase}_features*.pickle"),
                    os.path.join(cdir, f"{vbase}_features*.pkl"),
                ])
        
        # Then same dir
        patterns.extend([
            os.path.join(vdir, f"{vbase}_features*.pickle"),
            os.path.join(vdir, f"{vbase}_features*.pkl")
        ])
        
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                self.features_path = matches[0]
                self.feat_lbl.config(text=f"✓ {os.path.basename(matches[0])}", 
                                   foreground='green')
                self.auto_detect_dlc()
                return True
        
        self.feat_lbl.config(text="Not found", foreground='orange')
        return False
    
    def auto_detect_dlc(self):
        if not self.video_path:
            return False
        
        vdir = os.path.dirname(self.video_path)
        vbase = os.path.splitext(os.path.basename(self.video_path))[0]
        
        for sfx in ['DLC', '_labeled', '_filtered']:
            if sfx in vbase:
                vbase = vbase.split(sfx)[0]
                break
        
        patterns = [
            os.path.join(vdir, f"{vbase}DLC*.h5"),
            os.path.join(vdir, f"{vbase}_filtered.h5"),
            os.path.join(vdir, f"{vbase}*.h5")
        ]
        
        for pattern in patterns:
            h5s = glob.glob(pattern)
            for h5 in h5s:
                if '_features' not in h5.lower():
                    self.dlc_path = h5
                    return True
        
        return False
    
    def load_dlc_file(self):
        """Load DLC .h5 file for pose coordinates"""
        if not self.dlc_path or not os.path.isfile(self.dlc_path):
            print("[DLC] No DLC file to load")
            return False
        
        try:
            print(f"[DLC] Loading: {self.dlc_path}")
            self.dlc_df = pd.read_hdf(self.dlc_path)
            
            # Get scorer name (first level of multi-index columns)
            if isinstance(self.dlc_df.columns, pd.MultiIndex):
                scorer = self.dlc_df.columns.get_level_values(0)[0]
                print(f"[DLC] Scorer: {scorer}")
                print(f"[DLC] Bodyparts: {self.dlc_df[scorer].columns.get_level_values(0).unique().tolist()}")
            else:
                print("[DLC] Warning: Non-standard DLC format")
            
            print(f"[DLC] ✓ Loaded {len(self.dlc_df)} frames")
            return True
            
        except Exception as e:
            print(f"[DLC] ✗ Failed to load: {e}")
            self.dlc_df = None
            return False
    
    def load_data(self):
        try:
            # Load video
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception("Cannot open video")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS)))
            self.frame_spin.config(to=self.total_frames-1)
            
            # Try to auto-detect and load DLC file if not already set
            if not self.dlc_path:
                if self.auto_detect_dlc():
                    self.load_dlc_file()
            elif self.dlc_df is None:
                # DLC path is set but not loaded yet
                self.load_dlc_file()
            
            # Try to detect DLC crop parameters
            self.detect_dlc_crop()
            
            # Load features
            with open(self.features_path, 'rb') as f:
                self.features_df = pickle.load(f)
            
            # Check if features already have corrected coordinates
            self.check_if_features_are_corrected()
            
            # Find brightness features
            self.brightness_features = [c for c in self.features_df.columns 
                                       if any(p in c.lower() for p in ['pix', 'pixbrt', 'brightness'])]
            self.brightness_features.sort()
            
            if not self.brightness_features:
                messagebox.showwarning("No Features", 
                                      "No brightness features found")
                return
            
            self.feature_cb['values'] = self.brightness_features
            self.feature_cb.current(0)
            
            self.update_info()
            self.current_frame = 0
            self.frame_var.set(0)
            
            self.status_lbl.config(text=f"Loaded: {self.total_frames} frames, "
                                      f"{len(self.brightness_features)} features")
            
            self.show_graph()
            self.show_video()
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def update_info(self):
        feat = self.feature_var.get()
        if not feat or feat not in self.features_df.columns:
            return
        
        data = self.features_df[feat].values
        n_valid = np.sum(~np.isnan(data))
        n_total = len(data)
        
        bp = self.get_bodypart(feat)
        
        info = f"Feature: {feat}\nBodypart: {bp or 'Unknown'}\n\n"
        info += f"Valid: {n_valid:,} ({100*n_valid/n_total:.1f}%)\n"
        
        if n_valid > 0:
            info += f"Range: {np.nanmin(data):.2f} - {np.nanmax(data):.2f}\n"
            info += f"Mean: {np.nanmean(data):.2f}"
        
        self.info_txt.config(state='normal')
        self.info_txt.delete('1.0', tk.END)
        self.info_txt.insert('1.0', info)
        self.info_txt.config(state='disabled')
    
    def get_bodypart(self, feat):
        bps = ['hrpaw', 'hlpaw', 'frpaw', 'flpaw', 'snout', 'neck', 'tailbase', 'tailtip', 'centroid']
        for bp in bps:
            if bp in feat.lower():
                return bp
        return None
    
    def get_position(self, bp, frame_idx):
        if not bp:
            return None, None
        
        print(f"\n[DEBUG] Getting position for {bp} at frame {frame_idx}")
        
        # Try features first
        x_col = y_col = None
        for col in self.features_df.columns:
            if bp in col.lower():
                if '_x' in col.lower() or col.lower().endswith('x'):
                    x_col = col
                elif '_y' in col.lower() or col.lower().endswith('y'):
                    y_col = col
        
        if x_col and y_col:
            x = self.features_df[x_col].iloc[frame_idx]
            y = self.features_df[y_col].iloc[frame_idx]
            print(f"[DEBUG] From features: x={x}, y={y}")
            if pd.notna(x) and pd.notna(y):
                return int(x), int(y)
        
        # Try DLC
        if self.dlc_path:
            try:
                if self.dlc_df is None:
                    print(f"[DEBUG] Loading DLC file: {os.path.basename(self.dlc_path)}")
                    self.dlc_df = pd.read_hdf(self.dlc_path)
                    print(f"[DEBUG] DLC loaded, shape: {self.dlc_df.shape}")
                
                scorer = self.dlc_df.columns.get_level_values(0)[0]
                bps_in_dlc = self.dlc_df[scorer].columns.get_level_values(0).unique().tolist()
                print(f"[DEBUG] DLC bodyparts: {bps_in_dlc}")
                
                if bp in bps_in_dlc:
                    x = self.dlc_df[scorer][bp]['x'].iloc[frame_idx]
                    y = self.dlc_df[scorer][bp]['y'].iloc[frame_idx]
                    
                    # Get video resolution for comparison
                    if self.cap:
                        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[DEBUG] Video resolution: {vid_w}x{vid_h}")
                        print(f"[DEBUG] DLC raw coordinates: x={x:.2f}, y={y:.2f}")
                        print(f"[DEBUG] Crop offset: x={self.crop_offset_x}, y={self.crop_offset_y}")
                        
                        # Apply crop offset
                        x_corrected = x + self.crop_offset_x
                        y_corrected = y + self.crop_offset_y
                        
                        print(f"[DEBUG] Corrected coordinates: x={x_corrected:.2f}, y={y_corrected:.2f}")
                        print(f"[DEBUG] Position as % of frame: x={100*x_corrected/vid_w:.1f}%, y={100*y_corrected/vid_h:.1f}%")
                    
                    if pd.notna(x) and pd.notna(y):
                        x_int = int(round(x + self.crop_offset_x))
                        y_int = int(round(y + self.crop_offset_y))
                        print(f"[DEBUG] Returning corrected position: ({x_int}, {y_int})")
                        return x_int, y_int
            except Exception as e:
                print(f"[DEBUG] Error reading DLC: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[DEBUG] No position found")
        return None, None
    
    def on_feature_change(self):
        self.update_info()
        if self.graph_win and self.graph_win.winfo_exists():
            self.plot_graph()
        self.update_display()
    
    def show_graph(self):
        if not self.features_df is not None:
            messagebox.showwarning("No Data", "Load data first")
            return
        
        if self.graph_win and self.graph_win.winfo_exists():
            self.graph_win.lift()
            return
        
        self.graph_win = tk.Toplevel(self.root)
        self.graph_win.title("Timeline")
        self.graph_win.geometry("800x400")
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_win)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.canvas.mpl_connect('button_press_event', self.on_graph_click)
        self.plot_graph()
    
    def plot_graph(self):
        if not hasattr(self, 'ax'):
            return
        
        feat = self.feature_var.get()
        if not feat or feat not in self.features_df.columns:
            return
        
        data = self.features_df[feat].values
        time_min = np.arange(len(data)) / (self.fps * 60)
        
        self.ax.clear()
        self.ax.plot(time_min, data, 'b-', linewidth=1)
        self.ax.set_xlabel('Time (min)')
        self.ax.set_ylabel('Brightness')
        self.ax.set_title(f'{feat}\nClick to select frame')
        self.ax.grid(alpha=0.3)
        
        if 0 <= self.current_frame < len(data):
            t = self.current_frame / (self.fps * 60)
            self.ax.axvline(t, color='r', linestyle='--', linewidth=2)
            self.ax.plot(t, data[self.current_frame], 'ro', markersize=10, zorder=5)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_graph_click(self, event):
        if event.inaxes != self.ax:
            return
        
        frame = int(event.xdata * self.fps * 60)
        frame = max(0, min(frame, self.total_frames-1))
        
        self.current_frame = frame
        self.frame_var.set(frame)
        self.update_display()
    
    def show_video(self):
        if not self.features_df is not None:
            messagebox.showwarning("No Data", "Load data first")
            return
        
        if self.video_win and self.video_win.winfo_exists():
            self.video_win.lift()
            return
        
        self.video_win = tk.Toplevel(self.root)
        self.video_win.title("Video")
        self.video_win.geometry("900x700")
        
        self.video_canvas = tk.Canvas(self.video_win, bg='black')
        self.video_canvas.pack(fill='both', expand=True)
        
        self.update_display()
    
    def update_display(self):
        if not self.cap or self.features_df is None:
            return
        
        if not self.video_win or not self.video_win.winfo_exists():
            return
        
        feat = self.feature_var.get()
        if not feat:
            return
        
        self.current_frame = self.frame_var.get()
        feat_val = self.features_df[feat].iloc[self.current_frame]
        
        bp = self.get_bodypart(feat)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        x, y = self.get_position(bp, self.current_frame)
        
        if x is not None and y is not None:
            vis = self.draw_extraction(frame, x, y, feat, feat_val)
        else:
            vis = frame.copy()
            cv2.putText(vis, f"{feat}: {feat_val:.4f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, "Position unknown", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        self.display_frame(vis)
        
        if self.graph_win and self.graph_win.winfo_exists():
            self.plot_graph()
        
        self.frame_info_lbl.config(text=f"Val: {feat_val:.4f}")
    
    def draw_extraction(self, frame, x, y, feat, feat_val):
        radius = self.radius_var.get()
        h, w = frame.shape[:2]
        
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        region = frame[y1:y2, x1:x2]
        if region.shape[0] == 0 or region.shape[1] == 0:
            return frame
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_bright = gray.mean()
        
        vis = frame.copy()
        
        # Draw extraction area
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.circle(vis, (x, y), radius, (0, 255, 255), 3)
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        
        # Corner markers
        m = 20
        cv2.line(vis, (x1, y1), (x1+m, y1), (0, 255, 0), 4)
        cv2.line(vis, (x1, y1), (x1, y1+m), (0, 255, 0), 4)
        cv2.line(vis, (x2, y1), (x2-m, y1), (0, 255, 0), 4)
        cv2.line(vis, (x2, y1), (x2, y1+m), (0, 255, 0), 4)
        cv2.line(vis, (x1, y2), (x1+m, y2), (0, 255, 0), 4)
        cv2.line(vis, (x1, y2), (x1, y2-m), (0, 255, 0), 4)
        cv2.line(vis, (x2, y2), (x2-m, y2), (0, 255, 0), 4)
        cv2.line(vis, (x2, y2), (x2, y2-m), (0, 255, 0), 4)
        
        # Text with background
        def put_text_bg(img, txt, pos, scale=0.7, thick=2):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
            cv2.rectangle(img, (pos[0]-5, pos[1]-th-5), (pos[0]+tw+5, pos[1]+5), (0, 0, 0), -1)
            cv2.putText(img, txt, pos, font, scale, (255, 255, 255), thick)
        
        put_text_bg(vis, f"Frame: {self.current_frame}", (10, 30), 0.8)
        put_text_bg(vis, f"Position: ({x}, {y})", (10, 60), 0.7)
        put_text_bg(vis, f"Area: {radius*2}x{radius*2}px", (10, 90), 0.7)
        put_text_bg(vis, f"Mean: {mean_bright:.2f}", (10, 120), 0.7)
        put_text_bg(vis, f"Feature: {feat_val:.4f}", (10, 150), 0.7)
        
        if mean_bright < 5:
            put_text_bg(vis, "LOW - Try larger radius!", (10, 180), 0.7)
        
        # Inset
        inset_sz = 250
        scale = min(inset_sz / region.shape[1], inset_sz / region.shape[0])
        new_w = int(region.shape[1] * scale)
        new_h = int(region.shape[0] * scale)
        
        if new_w > 0 and new_h > 0:
            region_sc = cv2.resize(region, (new_w, new_h))
            gray_sc = cv2.resize(gray, (new_w, new_h))
            gray_bgr = cv2.cvtColor(gray_sc, cv2.COLOR_GRAY2BGR)
            
            ix = w - inset_sz - 30
            iy = 30
            
            bg_h = new_h * 2 + 80
            cv2.rectangle(vis, (ix-10, iy-30), (ix+inset_sz+10, iy+bg_h), (0, 0, 0), -1)
            cv2.rectangle(vis, (ix-10, iy-30), (ix+inset_sz+10, iy+bg_h), (255, 255, 255), 2)
            
            cv2.putText(vis, "EXTRACTED", (ix, iy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            vis[iy:iy+new_h, ix:ix+new_w] = region_sc
            cv2.rectangle(vis, (ix, iy), (ix+new_w, iy+new_h), (0, 255, 0), 2)
            cv2.putText(vis, "Color", (ix+5, iy+new_h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            gy = iy + new_h + 40
            vis[gy:gy+new_h, ix:ix+new_w] = gray_bgr
            cv2.rectangle(vis, (ix, gy), (ix+new_w, gy+new_h), (0, 255, 255), 2)
            cv2.putText(vis, "Grayscale", (ix+5, gy+new_h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return vis
    
    def display_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        
        if cw > 1 and ch > 1:
            scale = min(cw / pil_img.width, ch / pil_img.height, 1.0)
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        self.video_canvas.delete('all')
        self.video_canvas.create_image(cw//2, ch//2, image=photo, anchor='center')
        self.video_canvas.image = photo
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.graph_win and self.graph_win.winfo_exists():
            self.graph_win.destroy()
        if self.video_win and self.video_win.winfo_exists():
            self.video_win.destroy()
        self.root.destroy()


def main(video_path=None, features_path=None):
    root = tk.Tk()
    app = BrightnessPreview(root, video_path, features_path)
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1], None)
    else:
        main()
