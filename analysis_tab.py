"""
Analysis Tab for PixelPaws
Batch analysis with time binning, treatment groups, and publication-ready graphs
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime


class AnalysisTab(ttk.Frame):
    """
    Analysis tab for batch processing results
    - Load key file with subject-treatment mapping
    - Configure time bins
    - Calculate metrics (time, bouts, AUC)
    - Generate graphs
    - Export results
    """
    
    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.main_gui = main_gui
        
        # Data storage
        self.key_file = None
        self.key_df = None
        self.prediction_files = []
        self.results_df = None
        
        # Multi-behavior support
        self.available_behaviors = {}  # {behavior_name: [file_info1, file_info2, ...]}
        self.selected_behaviors = []   # List of selected behavior names
        self.analyze_mode = tk.StringVar(value='separate')  # 'separate' or 'combined'
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the Analysis tab UI"""
        # Main container with scrollbar
        canvas = tk.Canvas(self, bg='white')
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(
            title_frame,
            text="📊 Batch Analysis & Graphing",
            font=('Arial', 16, 'bold')
        ).pack(side='left')
        
        ttk.Button(
            title_frame,
            text="❓ Help",
            command=self.show_help
        ).pack(side='right')
        
        # Data Input Section
        self.create_data_input_section(scrollable_frame)
        
        # Settings Section
        self.create_settings_section(scrollable_frame)
        
        # Analysis Section
        self.create_analysis_section(scrollable_frame)
        
        # Results Section
        self.create_results_section(scrollable_frame)
    
    def create_data_input_section(self, parent):
        """Create data input section"""
        frame = ttk.LabelFrame(parent, text="📁 Data Input", padding=15)
        frame.pack(fill='x', padx=20, pady=10)
        
        # Key file
        key_frame = ttk.Frame(frame)
        key_frame.pack(fill='x', pady=5)
        
        ttk.Label(key_frame, text="Key File:", width=15).pack(side='left')
        self.key_file_var = tk.StringVar()
        ttk.Entry(key_frame, textvariable=self.key_file_var, width=50).pack(side='left', padx=5)
        ttk.Button(key_frame, text="Browse", command=self.browse_key_file).pack(side='left')
        
        ttk.Label(
            frame,
            text="Key file should have columns: Subject, Treatment (CSV or XLSX)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # Prediction files
        pred_frame = ttk.Frame(frame)
        pred_frame.pack(fill='x', pady=5)
        
        ttk.Label(pred_frame, text="Predictions:", width=15).pack(side='left')
        self.pred_folder_var = tk.StringVar()
        ttk.Entry(pred_frame, textvariable=self.pred_folder_var, width=50).pack(side='left', padx=5)
        ttk.Button(pred_frame, text="Browse Folder", command=self.browse_predictions).pack(side='left')
        
        ttk.Label(
            frame,
            text="Folder containing prediction CSV files from batch processing",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # Status
        self.data_status_label = ttk.Label(frame, text="", foreground='gray')
        self.data_status_label.pack(anchor='w', pady=5)
        
        # Behavior selection (shown after scanning predictions)
        self.behavior_selection_frame = ttk.LabelFrame(frame, text="🎯 Select Behaviors to Analyze", padding=10)
        # Will be packed when behaviors are detected
    
    def create_settings_section(self, parent):
        """Create settings section"""
        frame = ttk.LabelFrame(parent, text="⚙️ Analysis Settings", padding=15)
        frame.pack(fill='x', padx=20, pady=10)
        
        # Time binning
        bin_frame = ttk.Frame(frame)
        bin_frame.pack(fill='x', pady=5)
        
        ttk.Label(bin_frame, text="Time Bin Size:", width=15).pack(side='left')
        self.bin_size_var = tk.IntVar(value=5)
        ttk.Spinbox(
            bin_frame,
            from_=1,
            to=60,
            textvariable=self.bin_size_var,
            width=10
        ).pack(side='left', padx=5)
        ttk.Label(bin_frame, text="minutes").pack(side='left')
        
        ttk.Label(
            frame,
            text="Video will be divided into bins of this duration",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # FPS setting
        fps_frame = ttk.Frame(frame)
        fps_frame.pack(fill='x', pady=5)
        
        ttk.Label(fps_frame, text="Video FPS:", width=15).pack(side='left')
        self.fps_var = tk.IntVar(value=60)
        ttk.Spinbox(
            fps_frame,
            from_=1,
            to=120,
            textvariable=self.fps_var,
            width=10
        ).pack(side='left', padx=5)
        ttk.Label(fps_frame, text="frames/second").pack(side='left', padx=5)
        ttk.Button(
            fps_frame,
            text="Auto-Detect",
            command=self.auto_detect_fps
        ).pack(side='left')
        
        ttk.Label(
            frame,
            text="Frame rate of videos (used for time calculations)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # Phase Analysis (for formalin test, pain studies, etc.)
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(
            frame,
            text="Phase Analysis (Optional)",
            font=('Arial', 11, 'bold')
        ).pack(anchor='w', pady=5)
        
        self.enable_phases_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Enable phase analysis (e.g., Acute Phase and Phase II)",
            variable=self.enable_phases_var,
            command=self.toggle_phase_settings
        ).pack(anchor='w', pady=5)
        
        # Phase settings (initially hidden)
        self.phase_settings_frame = ttk.Frame(frame)
        
        # Acute Phase
        acute_frame = ttk.Frame(self.phase_settings_frame)
        acute_frame.pack(fill='x', pady=5)
        ttk.Label(acute_frame, text="Acute Phase:", width=15).pack(side='left')
        self.acute_start_var = tk.IntVar(value=0)
        self.acute_end_var = tk.IntVar(value=10)
        ttk.Spinbox(acute_frame, from_=0, to=120, textvariable=self.acute_start_var, width=8).pack(side='left', padx=2)
        ttk.Label(acute_frame, text="to").pack(side='left', padx=2)
        ttk.Spinbox(acute_frame, from_=0, to=120, textvariable=self.acute_end_var, width=8).pack(side='left', padx=2)
        ttk.Label(acute_frame, text="minutes").pack(side='left')
        
        # Phase II
        phase2_frame = ttk.Frame(self.phase_settings_frame)
        phase2_frame.pack(fill='x', pady=5)
        ttk.Label(phase2_frame, text="Phase II:", width=15).pack(side='left')
        self.phase2_start_var = tk.IntVar(value=10)
        self.phase2_end_var = tk.IntVar(value=60)
        ttk.Spinbox(phase2_frame, from_=0, to=120, textvariable=self.phase2_start_var, width=8).pack(side='left', padx=2)
        ttk.Label(phase2_frame, text="to").pack(side='left', padx=2)
        ttk.Spinbox(phase2_frame, from_=0, to=120, textvariable=self.phase2_end_var, width=8).pack(side='left', padx=2)
        ttk.Label(phase2_frame, text="minutes").pack(side='left')
        
        ttk.Label(
            self.phase_settings_frame,
            text="Typical formalin test: Acute = 0-10 min, Phase II = 10-60 min",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # Statistical Testing
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(
            frame,
            text="Statistical Testing (Optional)",
            font=('Arial', 11, 'bold')
        ).pack(anchor='w', pady=5)
        
        self.enable_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Add significance markers to graphs",
            variable=self.enable_stats_var,
            command=self.toggle_stats_settings
        ).pack(anchor='w', pady=5)
        
        # Stats settings (initially hidden)
        self.stats_settings_frame = ttk.Frame(frame)
        
        # Test type selection
        test_frame = ttk.Frame(self.stats_settings_frame)
        test_frame.pack(fill='x', pady=5)
        ttk.Label(test_frame, text="Test Type:", width=15).pack(side='left')
        self.stats_test_var = tk.StringVar(value='auto')
        ttk.Radiobutton(test_frame, text="Auto (2 groups: t-test, >2 groups: ANOVA)", 
                       variable=self.stats_test_var, value='auto').pack(anchor='w', padx=20)
        ttk.Radiobutton(test_frame, text="T-test (unpaired)", 
                       variable=self.stats_test_var, value='ttest').pack(anchor='w', padx=20)
        ttk.Radiobutton(test_frame, text="ANOVA with post-hoc", 
                       variable=self.stats_test_var, value='anova').pack(anchor='w', padx=20)
        
        # Significance level
        alpha_frame = ttk.Frame(self.stats_settings_frame)
        alpha_frame.pack(fill='x', pady=5)
        ttk.Label(alpha_frame, text="Significance level:", width=15).pack(side='left')
        self.stats_alpha_var = tk.DoubleVar(value=0.05)
        ttk.Radiobutton(alpha_frame, text="p < 0.05", variable=self.stats_alpha_var, value=0.05).pack(side='left', padx=5)
        ttk.Radiobutton(alpha_frame, text="p < 0.01", variable=self.stats_alpha_var, value=0.01).pack(side='left', padx=5)
        ttk.Radiobutton(alpha_frame, text="p < 0.001", variable=self.stats_alpha_var, value=0.001).pack(side='left', padx=5)
        
        # Time course specific option
        ttk.Separator(self.stats_settings_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(self.stats_settings_frame, text="Time Course Options:", font=('Arial', 9, 'bold')).pack(anchor='w')
        
        self.timecourse_posthoc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.stats_settings_frame,
            text="Show pairwise post-hoc at each timepoint (adds detail to Statistics tab)",
            variable=self.timecourse_posthoc_var
        ).pack(anchor='w', padx=20, pady=2)
        
        ttk.Label(
            self.stats_settings_frame,
            text="Note: Graph markers show overall ANOVA result. Post-hoc details in Statistics tab.",
            font=('Arial', 8),
            foreground='gray'
        ).pack(anchor='w', padx=20, pady=2)
        
        ttk.Label(
            self.stats_settings_frame,
            text="* p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        ttk.Label(
            frame,
            text="Frame rate of your videos (30 fps is common)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', pady=2)
        
        # Metrics selection
        ttk.Label(frame, text="Metrics to Calculate:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        
        metrics_frame = ttk.Frame(frame)
        metrics_frame.pack(fill='x', pady=5)
        
        self.metric_time = tk.BooleanVar(value=True)
        self.metric_bouts = tk.BooleanVar(value=True)
        self.metric_mean_bout = tk.BooleanVar(value=True)
        self.metric_auc = tk.BooleanVar(value=True)
        self.metric_percent = tk.BooleanVar(value=True)
        self.metric_frequency = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(metrics_frame, text="Total Time (s)", variable=self.metric_time).pack(anchor='w')
        ttk.Checkbutton(metrics_frame, text="Number of Bouts", variable=self.metric_bouts).pack(anchor='w')
        ttk.Checkbutton(metrics_frame, text="Mean Bout Duration", variable=self.metric_mean_bout).pack(anchor='w')
        ttk.Checkbutton(metrics_frame, text="AUC (Cumulative)", variable=self.metric_auc).pack(anchor='w')
        ttk.Checkbutton(metrics_frame, text="Percentage of Time", variable=self.metric_percent).pack(anchor='w')
        ttk.Checkbutton(metrics_frame, text="Bout Frequency (bouts/min)", variable=self.metric_frequency).pack(anchor='w')
        
        # Formalin Phase Analysis
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)
        ttk.Label(frame, text="🧪 Formalin Phase Analysis (Optional):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(5, 5))
        
        self.enable_phase_analysis = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, 
            text="Enable phase-specific analysis (Acute & Phase II)",
            variable=self.enable_phase_analysis,
            command=self.toggle_phase_settings
        ).pack(anchor='w', pady=5)
        
        # Phase settings container
        self.phase_settings_frame = ttk.Frame(frame)
        
        # Acute Phase
        acute_frame = ttk.LabelFrame(self.phase_settings_frame, text="Acute Phase", padding=10)
        acute_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Label(acute_frame, text="Start Time:").grid(row=0, column=0, sticky='w', padx=2)
        self.acute_start_var = tk.IntVar(value=0)
        ttk.Spinbox(acute_frame, from_=0, to=60, textvariable=self.acute_start_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(acute_frame, text="min").grid(row=0, column=2, sticky='w', padx=2)
        
        ttk.Label(acute_frame, text="End Time:").grid(row=1, column=0, sticky='w', padx=2, pady=5)
        self.acute_end_var = tk.IntVar(value=10)
        ttk.Spinbox(acute_frame, from_=0, to=60, textvariable=self.acute_end_var, width=8).grid(row=1, column=1, padx=2, pady=5)
        ttk.Label(acute_frame, text="min").grid(row=1, column=2, sticky='w', padx=2)
        
        # Phase II
        phase2_frame = ttk.LabelFrame(self.phase_settings_frame, text="Phase II", padding=10)
        phase2_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Label(phase2_frame, text="Start Time:").grid(row=0, column=0, sticky='w', padx=2)
        self.phase2_start_var = tk.IntVar(value=10)
        ttk.Spinbox(phase2_frame, from_=0, to=120, textvariable=self.phase2_start_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(phase2_frame, text="min").grid(row=0, column=2, sticky='w', padx=2)
        
        ttk.Label(phase2_frame, text="End Time:").grid(row=1, column=0, sticky='w', padx=2, pady=5)
        self.phase2_end_var = tk.IntVar(value=60)
        ttk.Spinbox(phase2_frame, from_=0, to=120, textvariable=self.phase2_end_var, width=8).grid(row=1, column=1, padx=2, pady=5)
        ttk.Label(phase2_frame, text="min").grid(row=1, column=2, sticky='w', padx=2)
        
        ttk.Label(
            self.phase_settings_frame,
            text="Phase analysis calculates total time in each phase period",
            font=('Arial', 9),
            foreground='gray'
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        # Initially hidden
        self.toggle_phase_settings()
    
    def toggle_phase_settings(self):
        """Show/hide phase analysis settings"""
        if self.enable_phase_analysis.get():
            self.phase_settings_frame.pack(fill='x', pady=10)
        else:
            self.phase_settings_frame.pack_forget()
    
    def create_analysis_section(self, parent):
        """Create analysis section"""
        frame = ttk.LabelFrame(parent, text="📊 Analysis", padding=15)
        frame.pack(fill='x', padx=20, pady=10)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=5)
        
        self.run_btn = ttk.Button(
            btn_frame,
            text="🚀 Run Analysis",
            command=self.run_analysis,
            style='Accent.TButton'
        )
        self.run_btn.pack(side='left', padx=5)
        
        self.export_btn = ttk.Button(
            btn_frame,
            text="💾 Export Results",
            command=self.export_results,
            state='disabled'
        )
        self.export_btn.pack(side='left', padx=5)
        
        self.graph_btn = ttk.Button(
            btn_frame,
            text="📈 Generate Graphs",
            command=self.generate_graphs,
            state='disabled'
        )
        self.graph_btn.pack(side='left', padx=5)
        
        # Progress
        self.progress_label = ttk.Label(frame, text="")
        self.progress_label.pack(anchor='w', pady=5)
    
    def create_results_section(self, parent):
        """Create results display section"""
        frame = ttk.LabelFrame(parent, text="📋 Results", padding=15)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create treeview for results
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        self.results_tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            show='headings'
        )
        
        vsb.config(command=self.results_tree.yview)
        hsb.config(command=self.results_tree.xview)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.results_tree.pack(fill='both', expand=True)
    
    def browse_key_file(self):
        """Browse for key file"""
        filepath = filedialog.askopenfilename(
            title="Select Key File",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.key_file_var.set(filepath)
            self.load_key_file(filepath)
    
    def load_key_file(self, filepath):
        """Load and validate key file"""
        try:
            # Load file
            if filepath.endswith('.xlsx'):
                self.key_df = pd.read_excel(filepath)
            else:
                self.key_df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['Subject', 'Treatment']
            missing_cols = [col for col in required_cols if col not in self.key_df.columns]
            
            if missing_cols:
                messagebox.showerror(
                    "Invalid Key File",
                    f"Key file is missing required columns: {', '.join(missing_cols)}\n\n"
                    f"Found columns: {', '.join(self.key_df.columns)}\n\n"
                    f"Required columns: Subject, Treatment"
                )
                self.key_df = None
                return
            
            # CRITICAL: Convert Subject column to string for matching
            # Excel stores as int (2801), but filenames extract as string ("2801")
            self.key_df['Subject'] = self.key_df['Subject'].astype(str)
            
            # Show summary
            n_subjects = len(self.key_df)
            treatments = self.key_df['Treatment'].unique()
            
            self.data_status_label.config(
                text=f"✓ Key file loaded: {n_subjects} subjects, {len(treatments)} treatment(s): {', '.join(map(str, treatments))}",
                foreground='green'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load key file:\n{e}")
            self.key_df = None
    
    def browse_predictions(self):
        """Browse for predictions folder"""
        folder = filedialog.askdirectory(title="Select Predictions Folder")
        
        if folder:
            self.pred_folder_var.set(folder)
            self.scan_predictions(folder)
    
    
    def toggle_phase_settings(self):
        """Show/hide phase analysis settings"""
        if self.enable_phases_var.get():
            self.phase_settings_frame.pack(fill='x', pady=5)
        else:
            self.phase_settings_frame.pack_forget()
    
    def toggle_stats_settings(self):
        """Show/hide statistical testing settings"""
        if self.enable_stats_var.get():
            self.stats_settings_frame.pack(fill='x', pady=5)
        else:
            self.stats_settings_frame.pack_forget()
    
    def auto_detect_fps(self):
        """Auto-detect FPS from video files"""
        # Ask user to select a video file from their project
        video_file = filedialog.askopenfilename(
            title="Select a Video File to Detect FPS",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
                ("All files", "*.*")
            ]
        )
        
        if not video_file:
            return
        
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                self.fps_var.set(int(round(fps)))
                messagebox.showinfo(
                    "FPS Detected",
                    f"Detected FPS: {fps:.2f}\n\nSet to: {int(round(fps))} fps"
                )
            else:
                messagebox.showerror(
                    "Detection Failed",
                    "Could not detect FPS from video file.\n\nPlease set manually."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video:\n{e}")
    
    def scan_predictions(self, folder):
        """Scan folder for prediction files"""
        try:
            prediction_files = []
            behaviors = {}  # {behavior_name: [file_info, ...]}
            
            # Check if folder contains result subfolders (PixelPaws batch output format)
            items = os.listdir(folder)
            result_folders = [item for item in items if os.path.isdir(os.path.join(folder, item)) 
                            and ('Results' in item or 'PixelPaws_Results' in item)]
            
            # Check if this IS a Results folder (new single-folder format)
            is_results_folder = os.path.basename(folder) in ['Results', 'PixelPaws_Results']
            
            if result_folders:
                # OLD FORMAT: Each subject has a results folder
                self.al_log_message(f"Found {len(result_folders)} result folder(s)")
                
                for result_folder in result_folders:
                    folder_path = os.path.join(folder, result_folder)
                    
                    # Look for ALL prediction CSV files (may be multiple behaviors)
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv') and 'prediction' in file.lower():
                            pred_file = os.path.join(folder_path, file)
                            
                            # Extract behavior name from filename
                            # Format: SubjectName_BehaviorName_predictions.csv
                            filename = os.path.basename(file)
                            behavior_name = self.extract_behavior_name(filename, result_folder)
                            
                            file_info = {
                                'path': pred_file,
                                'folder': result_folder,
                                'filename': filename,
                                'behavior': behavior_name
                            }
                            
                            prediction_files.append(file_info)
                            
                            # Group by behavior
                            if behavior_name not in behaviors:
                                behaviors[behavior_name] = []
                            behaviors[behavior_name].append(file_info)
            
            elif is_results_folder or not result_folders:
                # NEW FORMAT: All files in single Results folder OR direct CSV files
                self.al_log_message(f"Scanning for prediction files in {os.path.basename(folder)}")
                
                for file in os.listdir(folder):
                    # Skip non-CSV files and summary files
                    if not file.endswith('.csv'):
                        continue
                    if any(skip in file for skip in ['Summary', 'Treatment', 'Analysis']):
                        continue
                    
                    # Only process prediction files
                    if 'prediction' in file.lower():
                        full_path = os.path.join(folder, file)
                        behavior_name = self.extract_behavior_name(file, None)
                        
                        file_info = {
                            'path': full_path,
                            'folder': None,  # No individual subject folders
                            'filename': file,
                            'behavior': behavior_name
                        }
                        
                        prediction_files.append(file_info)
                        
                        if behavior_name not in behaviors:
                            behaviors[behavior_name] = []
                        behaviors[behavior_name].append(file_info)
            
            self.prediction_files = prediction_files
            self.available_behaviors = behaviors
            
            if prediction_files:
                self.data_status_label.config(
                    text=f"✓ Found {len(prediction_files)} prediction file(s) | {len(behaviors)} behavior(s)",
                    foreground='green'
                )
                
                # Print detected behaviors to console for debugging
                print(f"\n📊 Detected Behaviors:")
                for behavior_name, files in behaviors.items():
                    print(f"  • {behavior_name}: {len(files)} subject(s)")
                print()
                
                # Show behavior selection UI
                self.setup_behavior_selection()
            else:
                self.data_status_label.config(
                    text="⚠ No CSV files found in folder or subfolders",
                    foreground='orange'
                )
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan predictions folder:\n{e}")
            import traceback
            traceback.print_exc()
    
    def extract_behavior_name(self, filename, folder_name):
        """Extract behavior name from filename or folder"""
        # Remove extensions
        name = filename.replace('.csv', '')
        
        # For new format: 260129_Formalin_2801_PixelPaws_Left_licking_predictions.csv
        # We want to extract: "Left_licking"
        
        # Remove common suffixes first
        for suffix in ['_predictions', '_prediction', '_timebins', '_timebin', '_bouts', '_bout']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # Split into parts
        parts = name.split('_')
        
        # Find "PixelPaws" marker - behavior comes after it
        if 'PixelPaws' in parts:
            pixelpaws_idx = parts.index('PixelPaws')
            # Everything after PixelPaws is the behavior name
            behavior_parts = parts[pixelpaws_idx + 1:]
            if behavior_parts:
                return '_'.join(behavior_parts)
        
        # Fallback: Try to extract after classifier name pattern
        # Look for patterns like "Left_licking", "Right_licking", "Scratching"
        # These are typically at the end after date/subject/experiment info
        
        # Remove date-like parts (6 digits), short IDs (4 digits), experiment names
        filtered_parts = []
        for part in parts:
            # Skip if it's a date (6 digits) or subject ID (4 digits)
            if part.isdigit() and len(part) in [4, 6]:
                continue
            # Skip common experiment/metadata words
            if part.lower() in ['pixelpaws', 'results', 'formalin', 'formoxy']:
                continue
            filtered_parts.append(part)
        
        # Return remaining parts as behavior name
        if filtered_parts:
            return '_'.join(filtered_parts)
        else:
            return 'Unknown'
    
    def setup_behavior_selection(self):
        """Setup UI for selecting which behaviors to analyze"""
        # Clear previous widgets
        for widget in self.behavior_selection_frame.winfo_children():
            widget.destroy()
        
        # Pack the frame
        self.behavior_selection_frame.pack(fill='x', pady=10)
        
        # Info label
        ttk.Label(
            self.behavior_selection_frame,
            text=f"Found {len(self.available_behaviors)} behavior(s). Select which to analyze:",
            font=('Arial', 10, 'bold')
        ).pack(anchor='w', pady=5)
        
        # Checkbox for each behavior
        self.behavior_vars = {}
        for behavior_name in sorted(self.available_behaviors.keys()):
            n_files = len(self.available_behaviors[behavior_name])
            var = tk.BooleanVar(value=True)  # Default: all selected
            self.behavior_vars[behavior_name] = var
            
            cb = ttk.Checkbutton(
                self.behavior_selection_frame,
                text=f"{behavior_name} ({n_files} subjects)",
                variable=var
            )
            cb.pack(anchor='w', padx=20, pady=2)
        
        # Analysis mode selection
        ttk.Separator(self.behavior_selection_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(
            self.behavior_selection_frame,
            text="Analysis Mode:",
            font=('Arial', 10, 'bold')
        ).pack(anchor='w', pady=5)
        
        ttk.Radiobutton(
            self.behavior_selection_frame,
            text="Analyze each behavior separately (separate graphs)",
            variable=self.analyze_mode,
            value='separate'
        ).pack(anchor='w', padx=20, pady=2)
        
        ttk.Radiobutton(
            self.behavior_selection_frame,
            text="Analyze combined (sum all behaviors)",
            variable=self.analyze_mode,
            value='combined'
        ).pack(anchor='w', padx=20, pady=2)
        
        ttk.Radiobutton(
            self.behavior_selection_frame,
            text="Analyze both separately AND combined",
            variable=self.analyze_mode,
            value='both'
        ).pack(anchor='w', padx=20, pady=2)
        
        ttk.Label(
            self.behavior_selection_frame,
            text="Combined mode: All selected behaviors summed together (e.g., total nociceptive behaviors)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', padx=20, pady=2)
    
    def get_selected_behaviors(self):
        """Get list of selected behavior names"""
        return [name for name, var in self.behavior_vars.items() if var.get()]
    
    def al_log_message(self, message):
        """Log message (helper method for compatibility)"""
        print(message)
    
    def run_analysis(self):
        """Run the batch analysis"""
        # Validate inputs
        if self.key_df is None:
            messagebox.showerror("Error", "Please load a key file first")
            return
        
        if not self.prediction_files:
            messagebox.showerror("Error", "Please select a predictions folder")
            return
        
        # Get selected behaviors
        selected_behaviors = self.get_selected_behaviors()
        if not selected_behaviors:
            messagebox.showerror("Error", "Please select at least one behavior to analyze")
            return
        
        try:
            self.progress_label.config(text="🔄 Running analysis...")
            self.run_btn.config(state='disabled')
            self.update_idletasks()
            
            # Get settings
            bin_size_min = self.bin_size_var.get()
            fps = self.fps_var.get()
            analysis_mode = self.analyze_mode.get()
            
            # Filter prediction files to only selected behaviors
            filtered_files = [f for f in self.prediction_files if f['behavior'] in selected_behaviors]
            
            # Process based on analysis mode
            all_results = []
            
            if analysis_mode in ['separate', 'both']:
                # Analyze each behavior separately
                for behavior in selected_behaviors:
                    behavior_files = [f for f in filtered_files if f['behavior'] == behavior]
                    
                    self.progress_label.config(text=f"🔄 Analyzing {behavior}...")
                    self.update_idletasks()
                    
                    results = []
                    for i, pred_file_info in enumerate(behavior_files, 1):
                        result = self.analyze_single_file(pred_file_info, bin_size_min, fps, behavior)
                        if result is not None:
                            results.extend(result)
                    
                    all_results.extend(results)
            
            if analysis_mode in ['combined', 'both']:
                # Analyze combined (sum all behaviors)
                self.progress_label.config(text="🔄 Analyzing combined behaviors...")
                self.update_idletasks()
                
                combined_results = self.analyze_combined_behaviors(
                    filtered_files, selected_behaviors, bin_size_min, fps
                )
                all_results.extend(combined_results)
            
            if not all_results:
                messagebox.showerror("Error", "No valid data processed")
                return
            
            # Create results dataframe
            self.results_df = pd.DataFrame(all_results)
            
            # Display results
            self.display_results()
            
            # Enable buttons
            self.export_btn.config(state='normal')
            self.graph_btn.config(state='normal')
            
            behavior_text = f"{len(selected_behaviors)} behavior(s)" if len(selected_behaviors) > 1 else selected_behaviors[0]
            mode_text = {
                'separate': 'separately',
                'combined': 'combined',
                'both': 'separately + combined'
            }[analysis_mode]
            
            self.progress_label.config(
                text=f"✓ Analysis complete! {behavior_text} analyzed {mode_text}",
                foreground='green'
            )
            
            messagebox.showinfo(
                "Analysis Complete",
                f"Successfully analyzed:\n"
                f"• {len(selected_behaviors)} behavior(s)\n"
                f"• {len(filtered_files)} file(s)\n"
                f"• Mode: {mode_text}\n\n"
                f"Generated {len(all_results)} data points"
            )
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred:\n\n{e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.run_btn.config(state='normal')
    
    def analyze_single_file(self, pred_file_info, bin_size_min, fps, behavior_name=None):
        """Analyze a single prediction file"""
        try:
            # Extract info
            pred_file = pred_file_info['path']
            result_folder = pred_file_info.get('folder')
            if behavior_name is None:
                behavior_name = pred_file_info.get('behavior', 'Unknown')
            
            # Extract subject name from filename (works for both old and new format)
            filename = os.path.basename(pred_file)
            subject = self.extract_subject_name(filename)
            
            # Find subject in key file
            subject_row = self.key_df[self.key_df['Subject'] == subject]
            
            if subject_row.empty:
                print(f"Warning: Subject '{subject}' not found in key file, skipping")
                return None
            
            treatment = subject_row.iloc[0]['Treatment']
            
            # Load prediction data
            pred_df = pd.read_csv(pred_file)
            
            # Determine which column has predictions
            # Priority: prediction_filtered > prediction_raw > last column > first column with 0/1 values
            if 'prediction_filtered' in pred_df.columns:
                predictions = pred_df['prediction_filtered'].values
                print(f"  Using 'prediction_filtered' column")
            elif 'prediction_raw' in pred_df.columns:
                predictions = pred_df['prediction_raw'].values
                print(f"  Using 'prediction_raw' column")
            elif 'Left_licking' in pred_df.columns:
                predictions = pred_df['Left_licking'].values
                print(f"  Using 'Left_licking' column")
            elif 'Right_licking' in pred_df.columns:
                predictions = pred_df['Right_licking'].values
                print(f"  Using 'Right_licking' column")
            else:
                # Use last column (usually the prediction column)
                predictions = pred_df.iloc[:, -1].values
                print(f"  Using last column: {pred_df.columns[-1]}")
            
            # Verify predictions are binary (0 or 1)
            unique_vals = np.unique(predictions)
            if not all(val in [0, 1] for val in unique_vals):
                print(f"Warning: Predictions contain non-binary values: {unique_vals}")
                # Convert to binary if needed
                predictions = (predictions > 0.5).astype(int)
            
            # Calculate time bins
            total_frames = len(predictions)
            total_seconds = total_frames / fps
            total_minutes = total_seconds / 60
            
            print(f"  Video: {total_frames} frames @ {fps} fps = {total_minutes:.1f} minutes")
            
            bin_size_sec = bin_size_min * 60
            bin_size_frames = int(bin_size_sec * fps)
            
            n_bins = int(np.ceil(total_frames / bin_size_frames))
            
            # Analyze each bin
            results = []
            
            for bin_idx in range(n_bins):
                start_frame = bin_idx * bin_size_frames
                end_frame = min(start_frame + bin_size_frames, total_frames)
                
                bin_preds = predictions[start_frame:end_frame]
                
                # Calculate actual bin duration (might be shorter for last bin)
                actual_bin_duration_sec = len(bin_preds) / fps
                
                # Calculate metrics
                metrics = self.calculate_metrics(bin_preds, fps, actual_bin_duration_sec, bin_idx)
                
                # Debug logging for first bin
                if bin_idx == 0:
                    print(f"  First bin ({bin_idx * bin_size_min}-{(bin_idx + 1) * bin_size_min} min):")
                    print(f"    Frames: {start_frame}-{end_frame}")
                    print(f"    Predictions sum: {np.sum(bin_preds == 1)}/{len(bin_preds)}")
                    print(f"    N_Bouts detected: {metrics.get('N_Bouts', 0)}")
                
                # Add metadata
                metrics['Subject'] = subject
                metrics['Treatment'] = treatment
                metrics['Behavior'] = behavior_name
                metrics['Bin'] = f"{bin_idx * bin_size_min}-{(bin_idx + 1) * bin_size_min}"
                metrics['Bin_Index'] = bin_idx
                metrics['Bin_Start_Min'] = bin_idx * bin_size_min
                metrics['Bin_End_Min'] = (bin_idx + 1) * bin_size_min
                
                results.append(metrics)
            
            # Add phase-specific analysis if enabled
            if self.enable_phase_analysis.get():
                phase_results = self.calculate_phase_metrics(
                    predictions, fps, subject, treatment, behavior_name
                )
                results.extend(phase_results)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {pred_file_info.get('path', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_phase_metrics(self, predictions, fps, subject, treatment, behavior_name):
        """Calculate metrics for specific phases (e.g., Acute and Phase II for formalin)"""
        results = []
        
        # Get phase settings
        acute_start_min = self.acute_start_var.get()
        acute_end_min = self.acute_end_var.get()
        phase2_start_min = self.phase2_start_var.get()
        phase2_end_min = self.phase2_end_var.get()
        
        # Convert to frames
        acute_start_frame = int(acute_start_min * 60 * fps)
        acute_end_frame = int(acute_end_min * 60 * fps)
        phase2_start_frame = int(phase2_start_min * 60 * fps)
        phase2_end_frame = int(phase2_end_min * 60 * fps)
        
        total_frames = len(predictions)
        
        # Acute Phase
        if acute_end_frame <= total_frames:
            acute_preds = predictions[acute_start_frame:acute_end_frame]
            acute_duration_sec = len(acute_preds) / fps
            
            # Calculate metrics for acute phase
            acute_metrics = self.calculate_metrics(acute_preds, fps, acute_duration_sec, bin_idx=None)
            acute_metrics.update({
                'Subject': subject,
                'Treatment': treatment,
                'Behavior': behavior_name,
                'Phase': 'Acute',
                'Phase_Start_Min': acute_start_min,
                'Phase_End_Min': acute_end_min,
                'Bin': f'Acute ({acute_start_min}-{acute_end_min} min)',
                'Bin_Index': -1,  # Use -1 to indicate phase analysis
            })
            results.append(acute_metrics)
            
            print(f"  Acute Phase ({acute_start_min}-{acute_end_min} min): {np.sum(acute_preds == 1)} positive frames")
        
        # Phase II
        if phase2_end_frame <= total_frames:
            phase2_preds = predictions[phase2_start_frame:phase2_end_frame]
            phase2_duration_sec = len(phase2_preds) / fps
            
            # Calculate metrics for Phase II
            phase2_metrics = self.calculate_metrics(phase2_preds, fps, phase2_duration_sec, bin_idx=None)
            phase2_metrics.update({
                'Subject': subject,
                'Treatment': treatment,
                'Behavior': behavior_name,
                'Phase': 'Phase_II',
                'Phase_Start_Min': phase2_start_min,
                'Phase_End_Min': phase2_end_min,
                'Bin': f'Phase II ({phase2_start_min}-{phase2_end_min} min)',
                'Bin_Index': -2,  # Use -2 for Phase II
            })
            results.append(phase2_metrics)
            
            print(f"  Phase II ({phase2_start_min}-{phase2_end_min} min): {np.sum(phase2_preds == 1)} positive frames")
        else:
            print(f"  Warning: Video too short for Phase II analysis (need {phase2_end_min} min, have {total_frames/fps/60:.1f} min)")
        
        return results
    
    def extract_subject_name(self, filename):
        """Extract subject name from filename"""
        # Import the utility function from main GUI
        try:
            from PixelPaws_GUI import extract_subject_id_from_filename
            
            # Use the smart extraction function
            subject_id = extract_subject_id_from_filename(filename)
            if subject_id:
                return subject_id
        except ImportError:
            pass
        
        # Fallback to old method if import fails
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Remove common suffixes
        for suffix in ['_predictions', '_prediction', '_pred', '_Predictions', '_bouts']:
            if suffix in name:
                name = name.replace(suffix, '')
        
        return name
    
    def analyze_combined_behaviors(self, filtered_files, selected_behaviors, bin_size_min, fps):
        """Analyze combined behaviors (sum predictions across all behaviors)"""
        try:
            # Group files by subject
            subject_files = {}
            for file_info in filtered_files:
                pred_file = file_info['path']
                result_folder = file_info.get('folder')
                filename = os.path.basename(pred_file)
                
                # Extract subject using the new method
                subject = self.extract_subject_name(filename)
                
                if subject not in subject_files:
                    subject_files[subject] = []
                subject_files[subject].append(file_info)
            
            # Process each subject
            results = []
            
            for subject, files in subject_files.items():
                # Find treatment
                treatment_match = self.key_df[self.key_df['Subject'] == subject]
                if treatment_match.empty:
                    print(f"  Warning: {subject} not found in key file")
                    continue
                treatment = treatment_match.iloc[0]['Treatment']
                
                # Load all prediction files for this subject
                all_predictions = []
                min_frames = float('inf')
                
                for file_info in files:
                    pred_df = pd.read_csv(file_info['path'])
                    
                    # Get predictions
                    if 'prediction_filtered' in pred_df.columns:
                        preds = pred_df['prediction_filtered'].values
                    elif 'prediction_raw' in pred_df.columns:
                        preds = pred_df['prediction_raw'].values
                    else:
                        preds = pred_df.iloc[:, -1].values
                    
                    all_predictions.append(preds)
                    min_frames = min(min_frames, len(preds))
                
                # Truncate all to same length and sum
                truncated_preds = [p[:min_frames] for p in all_predictions]
                combined_predictions = np.sum(truncated_preds, axis=0)
                
                # Binarize: any behavior present = 1
                combined_predictions = (combined_predictions > 0).astype(int)
                
                # Calculate time bins
                total_frames = len(combined_predictions)
                bin_size_sec = bin_size_min * 60
                bin_size_frames = int(bin_size_sec * fps)
                n_bins = int(np.ceil(total_frames / bin_size_frames))
                
                # Analyze each bin
                for bin_idx in range(n_bins):
                    start_frame = bin_idx * bin_size_frames
                    end_frame = min(start_frame + bin_size_frames, total_frames)
                    
                    bin_preds = combined_predictions[start_frame:end_frame]
                    actual_bin_duration_sec = len(bin_preds) / fps
                    
                    metrics = self.calculate_metrics(bin_preds, fps, actual_bin_duration_sec, bin_idx)
                    
                    # Add metadata
                    metrics['Subject'] = subject
                    metrics['Treatment'] = treatment
                    metrics['Behavior'] = 'Combined_' + '+'.join(sorted(selected_behaviors))
                    metrics['Bin'] = f"{bin_idx * bin_size_min}-{(bin_idx + 1) * bin_size_min}"
                    metrics['Bin_Index'] = bin_idx
                    metrics['Bin_Start_Min'] = bin_idx * bin_size_min
                    metrics['Bin_End_Min'] = (bin_idx + 1) * bin_size_min
                    
                    results.append(metrics)
            
            # Calculate phase-specific metrics if enabled
            if self.enable_phase_analysis.get():
                phase_results = self.calculate_phase_metrics(
                    predictions, fps, subject, treatment, behavior_name
                )
                results.extend(phase_results)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing combined behaviors: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_metrics(self, predictions, fps, bin_duration_sec, bin_idx):
        """Calculate behavior metrics for a time bin"""
        metrics = {}
        
        # Total time in behavior
        if self.metric_time.get():
            frames_in_behavior = np.sum(predictions == 1)
            metrics['Total_Time_s'] = frames_in_behavior / fps
        
        # Number of bouts
        if self.metric_bouts.get() or self.metric_mean_bout.get() or self.metric_frequency.get():
            bouts = self.detect_bouts(predictions)
            metrics['N_Bouts'] = len(bouts)
            
            # Mean bout duration
            if self.metric_mean_bout.get() and len(bouts) > 0:
                bout_durations = [(end - start + 1) / fps for start, end in bouts]
                metrics['Mean_Bout_Duration_s'] = np.mean(bout_durations)
            else:
                metrics['Mean_Bout_Duration_s'] = 0
            
            # Bout frequency
            if self.metric_frequency.get():
                metrics['Bout_Frequency_per_min'] = (len(bouts) / bin_duration_sec) * 60
        
        # Percentage of time
        if self.metric_percent.get():
            metrics['Percent_Time'] = (np.sum(predictions == 1) / len(predictions)) * 100
        
        # AUC (cumulative time up to this bin)
        if self.metric_auc.get() and bin_idx is not None and bin_idx >= 0:
            # AUC is just total time for this implementation
            # (true cumulative AUC will be calculated later)
            frames_in_behavior = np.sum(predictions == 1)
            metrics['AUC'] = frames_in_behavior / fps
        
        return metrics
    
    def detect_bouts(self, predictions):
        """Detect behavioral bouts (continuous periods of behavior)"""
        bouts = []
        in_bout = False
        bout_start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1:
                if not in_bout:
                    in_bout = True
                    bout_start = i
            else:
                if in_bout:
                    bouts.append((bout_start, i - 1))
                    in_bout = False
        
        # Close final bout if needed
        if in_bout:
            bouts.append((bout_start, len(predictions) - 1))
        
        return bouts
    
    
    def display_results(self):
        """Display results in treeview"""
        # Clear existing
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.results_df is None:
            return
        
        # Configure columns
        columns = list(self.results_df.columns)
        self.results_tree['columns'] = columns
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100, anchor='center')
        
        # Insert data
        for _, row in self.results_df.iterrows():
            values = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
            self.results_tree.insert('', 'end', values=values)
    
    def export_results(self):
        """Export results to CSV"""
        if self.results_df is None:
            messagebox.showerror("Error", "No results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.results_df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Results exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export:\n{e}")
    
    def generate_graphs(self):
        """Generate publication-ready graphs"""
        if self.results_df is None:
            messagebox.showerror("Error", "No results to graph")
            return
        
        # Check if multiple behaviors
        behaviors = sorted(self.results_df['Behavior'].unique())
        
        if len(behaviors) > 1:
            # Ask user which behaviors to graph
            behavior_dialog = tk.Toplevel(self)
            behavior_dialog.title("Select Behaviors to Graph")
            behavior_dialog.geometry("450x400")
            behavior_dialog.grab_set()
            
            ttk.Label(
                behavior_dialog, 
                text="Select behaviors to graph:", 
                font=('Arial', 12, 'bold')
            ).pack(pady=10)
            
            ttk.Label(
                behavior_dialog,
                text="All selected behaviors will appear in the same window",
                font=('Arial', 9),
                foreground='gray'
            ).pack(pady=5)
            
            # Checkboxes for each behavior
            behavior_vars = {}
            for behavior in behaviors:
                var = tk.BooleanVar(value=True)  # All selected by default
                behavior_vars[behavior] = var
                
                n_subjects = len(self.results_df[self.results_df['Behavior'] == behavior]['Subject'].unique())
                cb = ttk.Checkbutton(
                    behavior_dialog,
                    text=f"{behavior} ({n_subjects} subjects)",
                    variable=var
                )
                cb.pack(anchor='w', padx=30, pady=5)
            
            # Select All / Deselect All buttons
            btn_frame = ttk.Frame(behavior_dialog)
            btn_frame.pack(pady=10)
            
            def select_all():
                for var in behavior_vars.values():
                    var.set(True)
            
            def deselect_all():
                for var in behavior_vars.values():
                    var.set(False)
            
            ttk.Button(btn_frame, text="Select All", command=select_all).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Deselect All", command=deselect_all).pack(side='left', padx=5)
            
            def on_ok():
                selected = [b for b, v in behavior_vars.items() if v.get()]
                if not selected:
                    messagebox.showwarning("No Selection", "Please select at least one behavior")
                    return
                behavior_dialog.result = selected
                behavior_dialog.destroy()
            
            def on_cancel():
                behavior_dialog.result = None
                behavior_dialog.destroy()
            
            ttk.Button(behavior_dialog, text="Generate Graphs", command=on_ok, 
                      style='Accent.TButton' if hasattr(ttk, 'Accent') else None).pack(pady=10)
            ttk.Button(behavior_dialog, text="Cancel", command=on_cancel).pack()
            
            behavior_dialog.wait_window()
            
            if not hasattr(behavior_dialog, 'result') or behavior_dialog.result is None:
                return
            
            selected_behaviors = behavior_dialog.result
        else:
            selected_behaviors = [behaviors[0]]
        
        # Generate graphs for all selected behaviors in one window
        self.generate_multi_behavior_graphs(selected_behaviors)
    
    def generate_multi_behavior_graphs(self, selected_behaviors):
        """Generate graphs for multiple behaviors in a single window"""
        # Filter to selected behaviors
        filtered_df = self.results_df[self.results_df['Behavior'].isin(selected_behaviors)].copy()
        
        if filtered_df.empty:
            messagebox.showerror("Error", "No data found for selected behaviors")
            return
        
        # Get unique treatments (across all behaviors)
        treatments = sorted(filtered_df['Treatment'].unique())
        
        # Get max time available
        max_time = filtered_df['Bin_End_Min'].max()
        
        # Ask user for treatment order, colors, AND time window (ONE TIME for all behaviors)
        order_dialog = tk.Toplevel(self)
        order_dialog.title(f"Graph Settings - {len(selected_behaviors)} Behavior(s)")
        order_dialog.geometry("550x700")  # Increased height
        order_dialog.grab_set()
        
        behavior_list = ", ".join(selected_behaviors) if len(selected_behaviors) <= 3 else f"{len(selected_behaviors)} behaviors"
        ttk.Label(
            order_dialog, 
            text=f"Graph Settings: {behavior_list}", 
            font=('Arial', 14, 'bold')
        ).pack(pady=10)
        
        # Time window section
        ttk.Label(order_dialog, text="1. Time Window", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        time_frame = ttk.Frame(order_dialog)
        time_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(time_frame, text="Show data up to:").pack(side='left')
        time_var = tk.IntVar(value=int(max_time))
        ttk.Spinbox(time_frame, from_=5, to=int(max_time), textvariable=time_var, width=10).pack(side='left', padx=5)
        ttk.Label(time_frame, text=f"minutes (max: {int(max_time)} min)").pack(side='left')
        
        # Error bar type section
        ttk.Label(order_dialog, text="2. Error Bar Type", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        error_frame = ttk.Frame(order_dialog)
        error_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(error_frame, text="Display:").pack(side='left')
        error_var = tk.StringVar(value='SEM')
        ttk.Radiobutton(error_frame, text="SEM (Standard Error)", variable=error_var, value='SEM').pack(side='left', padx=5)
        ttk.Radiobutton(error_frame, text="SD (Standard Deviation)", variable=error_var, value='SD').pack(side='left', padx=5)
        ttk.Label(
            order_dialog,
            text="SEM = SD / √n  (shows precision of the mean)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(anchor='w', padx=20, pady=2)
        
        # Heatmap color palette section
        ttk.Label(order_dialog, text="3. Heatmap Color Palette", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        palette_frame = ttk.Frame(order_dialog)
        palette_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(palette_frame, text="Palette:").pack(side='left')
        palette_var = tk.StringVar(value='YlOrRd')
        palette_options = ['YlOrRd', 'viridis', 'plasma', 'Blues', 'Reds', 'Greens', 'RdYlBu_r', 'coolwarm']
        ttk.Combobox(palette_frame, textvariable=palette_var, values=palette_options, 
                    state='readonly', width=15).pack(side='left', padx=5)
        ttk.Label(palette_frame, text="(for heatmap only)", font=('Arial', 9), foreground='gray').pack(side='left')
        
        # Treatment order section
        ttk.Label(order_dialog, text="4. Treatment Order", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        ttk.Label(order_dialog, text="(Left to right, or top to bottom)", font=('Arial', 9), foreground='gray').pack(anchor='w', padx=20)
        
        # Create listbox with treatments
        listbox_frame = ttk.Frame(order_dialog)
        listbox_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame, font=('Arial', 11), height=len(treatments))
        listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=listbox.yview)
        scrollbar.pack(side='right', fill='y')
        listbox.config(yscrollcommand=scrollbar.set)
        
        for treatment in treatments:
            listbox.insert('end', treatment)
        
        # Color selection section
        ttk.Label(order_dialog, text="5. Colors (for graphs, not heatmap)", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        
        color_frame = ttk.Frame(order_dialog)
        color_frame.pack(fill='x', padx=20, pady=5)
        
        # Color options
        color_options = {
            'Teal': '#66c2a5',
            'Orange': '#fc8d62',
            'Purple': '#8da0cb',
            'Pink': '#e78ac3',
            'Green': '#a6d854',
            'Yellow': '#ffd92f',
            'Brown': '#e5c494',
            'Gray': '#b3b3b3',
            'White (black outline)': 'white_black'  # Special case
        }
        
        color_vars = {}
        for treatment in treatments:
            tf = ttk.Frame(color_frame)
            tf.pack(fill='x', pady=3)
            ttk.Label(tf, text=f"{treatment}:", width=15).pack(side='left')
            color_var = tk.StringVar(value=list(color_options.keys())[treatments.index(treatment) % len(color_options)])
            color_vars[treatment] = color_var
            ttk.Combobox(tf, textvariable=color_var, values=list(color_options.keys()), 
                        state='readonly', width=15).pack(side='left', padx=5)
        
        # Instructions
        inst_frame = ttk.Frame(order_dialog)
        inst_frame.pack(fill='x', padx=20, pady=10)
        ttk.Label(inst_frame, text="💡 Tip: Click and drag items in the list to reorder", 
                 font=('Arial', 9), foreground='blue').pack(anchor='w')
        
        # Variables to store results
        result = {'order': None, 'colors': None, 'time_window': None, 'heatmap_palette': None, 'error_type': None, 'cancelled': False}
        
        # Drag and drop functionality
        def on_drag_start(event):
            widget = event.widget
            index = widget.nearest(event.y)
            widget.selection_clear(0, 'end')
            widget.selection_set(index)
            widget.activate(index)
            widget.drag_data = {'index': index, 'item': widget.get(index)}
        
        def on_drag_motion(event):
            widget = event.widget
            index = widget.nearest(event.y)
            if hasattr(widget, 'drag_data') and index != widget.drag_data['index']:
                # Remove from old position
                widget.delete(widget.drag_data['index'])
                # Insert at new position
                widget.insert(index, widget.drag_data['item'])
                widget.drag_data['index'] = index
                widget.selection_clear(0, 'end')
                widget.selection_set(index)
        
        listbox.bind('<Button-1>', on_drag_start)
        listbox.bind('<B1-Motion>', on_drag_motion)
        
        def on_ok():
            result['order'] = [listbox.get(i) for i in range(listbox.size())]
            result['colors'] = {t: color_options[color_vars[t].get()] for t in treatments}
            result['time_window'] = time_var.get()
            result['heatmap_palette'] = palette_var.get()
            result['error_type'] = error_var.get()
            order_dialog.destroy()
        
        def on_cancel():
            result['cancelled'] = True
            order_dialog.destroy()
        
        btn_frame = ttk.Frame(order_dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="✓ Generate Graphs", command=on_ok, width=20).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="✗ Cancel", command=on_cancel, width=15).pack(side='left', padx=5)
        
        order_dialog.wait_window()
        
        if result['cancelled'] or not result['order']:
            return
        
        # Store treatment order, colors, time window, heatmap palette, and error type for use in all graph functions
        self.treatment_order = result['order']
        self.treatment_colors = result['colors']
        self.time_window = result['time_window']
        self.heatmap_palette = result['heatmap_palette']
        self.error_type = result['error_type']
        
        # Filter data to time window
        self.filtered_results_df = filtered_df[filtered_df['Bin_Start_Min'] < self.time_window].copy()
        
        # Create new window for graphs
        graph_window = tk.Toplevel(self)
        behaviors_text = " + ".join(selected_behaviors) if len(selected_behaviors) <= 3 else f"{len(selected_behaviors)} Behaviors"
        graph_window.title(f"Analysis Graphs - {behaviors_text}")
        graph_window.geometry("1200x800")
        
        # Create main notebook for behaviors
        main_notebook = ttk.Notebook(graph_window)
        main_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create a tab for each behavior
        for behavior in selected_behaviors:
            # Filter to this behavior
            behavior_df = self.filtered_results_df[self.filtered_results_df['Behavior'] == behavior].copy()
            
            # Create frame for this behavior
            behavior_frame = ttk.Frame(main_notebook)
            main_notebook.add(behavior_frame, text=behavior)
            
            # Create sub-notebook for graph types
            graph_notebook = ttk.Notebook(behavior_frame)
            graph_notebook.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Temporarily set filtered_results_df to this behavior's data
            original_df = self.filtered_results_df
            self.filtered_results_df = behavior_df
            
            # Verify we're only showing data for this behavior
            if 'Behavior' in behavior_df.columns:
                unique_behaviors = behavior_df['Behavior'].unique()
                if len(unique_behaviors) != 1 or unique_behaviors[0] != behavior:
                    print(f"WARNING: Behavior filter may have failed. Expected {behavior}, got {unique_behaviors}")
            
            # Generate different graph types for this behavior
            self.create_time_course_graph(graph_notebook, behavior)
            self.create_total_time_graph(graph_notebook, behavior)
            self.create_bout_analysis_graph(graph_notebook, behavior)
            
            # Add phase analysis if enabled
            if hasattr(self, 'enable_phases_var') and self.enable_phases_var.get():
                self.create_phase_analysis_graph(graph_notebook, behavior)
            
            self.create_heatmap_graph(graph_notebook, behavior)
            
            # Add statistics tab if enabled
            if hasattr(self, 'enable_stats_var') and self.enable_stats_var.get():
                self.create_statistics_tab(graph_notebook, behavior, behavior_df)
            
            # Restore original dataframe
            self.filtered_results_df = original_df
    
    def create_time_course_graph(self, notebook, behavior_name=""):
        """Create time course line plot with SEM error bars"""
        frame = ttk.Frame(notebook)
        tab_title = f"Time Course - {behavior_name}" if behavior_name else "Time Course"
        notebook.add(frame, text="Time Course")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use filtered data
        df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
        
        # Group by treatment and bin
        if 'Total_Time_s' in df.columns:
            # Calculate error based on user selection
            error_type = self.error_type if hasattr(self, 'error_type') else 'SEM'
            
            if error_type == 'SD':
                grouped = df.groupby(['Treatment', 'Bin_Start_Min'])['Total_Time_s'].agg(['mean', 'std']).reset_index()
                grouped.rename(columns={'std': 'error'}, inplace=True)
            else:  # SEM
                grouped = df.groupby(['Treatment', 'Bin_Start_Min'])['Total_Time_s'].agg(['mean', 'sem']).reset_index()
                grouped.rename(columns={'sem': 'error'}, inplace=True)
            
            # Use stored treatment order and colors
            treatments = self.treatment_order if hasattr(self, 'treatment_order') else grouped['Treatment'].unique()
            colors = self.treatment_colors if hasattr(self, 'treatment_colors') else {}
            
            for treatment in treatments:
                data = grouped[grouped['Treatment'] == treatment]
                color = colors.get(treatment, None)
                
                # Handle white with black outline
                if color == 'white_black':
                    ax.errorbar(
                        data['Bin_Start_Min'], 
                        data['mean'],
                        yerr=data['error'],
                        marker='o', 
                        label=treatment, 
                        linewidth=2.5,
                        capsize=5,
                        capthick=2,
                        color='black',
                        markerfacecolor='white',
                        markeredgecolor='black',
                        markeredgewidth=2,
                        markersize=8
                    )
                else:
                    ax.errorbar(
                        data['Bin_Start_Min'], 
                        data['mean'],
                        yerr=data['error'],
                        marker='o', 
                        label=treatment, 
                        linewidth=2,
                        capsize=5,
                        capthick=2,
                        color=color
                    )
            
            ax.set_xlabel('Time (minutes)', fontsize=12)
            ylabel = f'Time in Behavior (seconds) ± {error_type}'
            ax.set_ylabel(ylabel, fontsize=12)
            title = f'{behavior_name} - Behavior Over Time by Treatment' if behavior_name else 'Behavior Over Time by Treatment'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add statistical testing - Two-way ANOVA (time × treatment) with post-hoc
            if self.enable_stats_var.get():
                from scipy import stats
                import warnings
                warnings.filterwarnings('ignore')
                
                # Get unique time bins
                time_bins = sorted(df['Bin_Start_Min'].unique())
                alpha = self.stats_alpha_var.get()
                
                # Store which time bins are significant for post-hoc
                significant_bins = []
                
                # Perform two-way ANOVA using statsmodels
                try:
                    import statsmodels.api as sm
                    from statsmodels.formula.api import ols
                    
                    # Prepare data for two-way ANOVA
                    anova_df = df[['Subject', 'Treatment', 'Bin_Start_Min', 'Total_Time_s']].copy()
                    anova_df['Time'] = anova_df['Bin_Start_Min'].astype('category')
                    anova_df['Treatment'] = anova_df['Treatment'].astype('category')
                    
                    # Two-way repeated measures ANOVA
                    model = ols('Total_Time_s ~ C(Treatment) + C(Time) + C(Treatment):C(Time)', data=anova_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    # Check for interaction effect
                    interaction_p = anova_table.loc['C(Treatment):C(Time)', 'PR(>F)']
                    treatment_p = anova_table.loc['C(Treatment)', 'PR(>F)']
                    time_p = anova_table.loc['C(Time)', 'PR(>F)']
                    
                    # Store two-way ANOVA results for stats tab
                    self.timecourse_anova_results = {
                        'anova_table': anova_table,
                        'treatment_p': treatment_p,
                        'time_p': time_p,
                        'interaction_p': interaction_p
                    }
                    
                    # If interaction is significant OR treatment effect is significant, do post-hoc at each timepoint
                    if interaction_p < alpha or treatment_p < alpha:
                        for bin_start in time_bins:
                            bin_df = df[df['Bin_Start_Min'] == bin_start]
                            
                            # Get data for each treatment at this time bin
                            groups = []
                            group_means = []
                            for treatment in treatments:
                                treat_data = bin_df[bin_df['Treatment'] == treatment]['Total_Time_s'].values
                                if len(treat_data) > 0:
                                    groups.append(treat_data)
                                    group_means.append(np.mean(treat_data))
                            
                            # Perform post-hoc test if we have enough groups
                            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                                if len(groups) == 2:
                                    # t-test for 2 groups (no multiple comparison issue)
                                    _, p_val = stats.ttest_ind(groups[0], groups[1])
                                    test_type = 't-test'
                                else:
                                    # For 3+ groups: First do ANOVA, then Tukey HSD if significant
                                    f_stat, anova_p = stats.f_oneway(*groups)
                                    
                                    print(f"Timepoint {bin_start}: {len(groups)} groups, ANOVA p={anova_p:.4f}")
                                    
                                    # If ANOVA is significant, use Tukey HSD for pairwise comparisons
                                    if anova_p < alpha:
                                        try:
                                            from scipy.stats import tukey_hsd
                                            # Tukey HSD test for all pairwise comparisons
                                            res = tukey_hsd(*groups)
                                            # Get the minimum p-value from all pairwise comparisons
                                            p_val = res.pvalue.min()
                                            test_type = 'Tukey HSD'
                                            print(f"  → Using Tukey HSD, min p={p_val:.4f}")
                                        except (ImportError, AttributeError) as e:
                                            print(f"  → Tukey HSD not available ({e}), using Bonferroni")
                                            # Fallback: Use Bonferroni correction
                                            # Number of pairwise comparisons
                                            n_comparisons = len(groups) * (len(groups) - 1) / 2
                                            bonferroni_alpha = alpha / n_comparisons
                                            
                                            # Find minimum p-value from pairwise t-tests
                                            min_p = 1.0
                                            for i in range(len(groups)):
                                                for j in range(i+1, len(groups)):
                                                    _, p = stats.ttest_ind(groups[i], groups[j])
                                                    min_p = min(min_p, p)
                                            
                                            p_val = min_p
                                            test_type = 'Bonferroni'
                                            print(f"  → Using Bonferroni, min p={p_val:.4f}")
                                    else:
                                        # ANOVA not significant, skip this timepoint
                                        print(f"  → ANOVA not significant, skipping")
                                        continue
                                
                                # Determine significance marker
                                if p_val < 0.001:
                                    marker = '***'
                                    significant = True
                                elif p_val < 0.01:
                                    marker = '**'
                                    significant = True
                                elif p_val < alpha:
                                    marker = '*'
                                    significant = True
                                else:
                                    marker = None
                                    significant = False
                                
                                if significant:
                                    # Find the maximum mean value at this timepoint
                                    max_mean = max(group_means)
                                    max_idx = group_means.index(max_mean)
                                    max_treatment = treatments[max_idx]
                                    
                                    # Get error for this treatment at this timepoint
                                    grouped_at_bin = grouped[grouped['Bin_Start_Min'] == bin_start]
                                    error_row = grouped_at_bin[grouped_at_bin['Treatment'] == max_treatment]
                                    
                                    if not error_row.empty:
                                        error_val = error_row['error'].iloc[0]
                                        # Position marker well above the error bar (15% of max mean above error bar)
                                        marker_y = max_mean + error_val + (max_mean * 0.15)
                                    else:
                                        marker_y = max_mean * 1.20
                                    
                                    significant_bins.append((bin_start, marker, marker_y, p_val))
                    
                    # Add significance markers to graph (BLACK color)
                    for bin_start, marker, marker_y, p_val in significant_bins:
                        ax.text(bin_start, marker_y, marker, 
                               ha='center', va='bottom', 
                               fontsize=11, fontweight='bold',
                               color='black')
                    
                    # Add legend with two-way ANOVA results
                    if interaction_p < 0.001:
                        int_text = 'p<0.001***'
                    elif interaction_p < 0.01:
                        int_text = f'p={interaction_p:.3f}**'
                    elif interaction_p < alpha:
                        int_text = f'p={interaction_p:.3f}*'
                    else:
                        int_text = f'p={interaction_p:.3f} ns'
                    
                    legend_text = f"Two-way ANOVA Time×Treatment: {int_text}\n"
                    legend_text += "Post-hoc: Tukey HSD (* p<0.05, ** p<0.01, *** p<0.001)"
                    
                    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='left',
                           fontsize=8, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
                    
                except ImportError:
                    # Fallback if statsmodels not available - just do post-hoc at each timepoint
                    print("Warning: statsmodels not installed. Performing post-hoc tests only.")
                    self.timecourse_anova_results = None
                    
                    for bin_start in time_bins:
                        bin_df = df[df['Bin_Start_Min'] == bin_start]
                        groups = []
                        group_means = []
                        
                        for treatment in treatments:
                            treat_data = bin_df[bin_df['Treatment'] == treatment]['Total_Time_s'].values
                            if len(treat_data) > 0:
                                groups.append(treat_data)
                                group_means.append(np.mean(treat_data))
                        
                        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                            if len(groups) == 2:
                                _, p_val = stats.ttest_ind(groups[0], groups[1])
                            else:
                                # ANOVA first, then Bonferroni-corrected pairwise tests
                                f_stat, anova_p = stats.f_oneway(*groups)
                                if anova_p >= alpha:
                                    continue  # ANOVA not significant, skip this timepoint
                                
                                # Bonferroni correction for multiple pairwise comparisons
                                n_comparisons = len(groups) * (len(groups) - 1) / 2
                                min_p = 1.0
                                for i in range(len(groups)):
                                    for j in range(i+1, len(groups)):
                                        _, p = stats.ttest_ind(groups[i], groups[j])
                                        min_p = min(min_p, p)
                                p_val = min_p
                            
                            if p_val < 0.001:
                                marker = '***'
                            elif p_val < 0.01:
                                marker = '**'
                            elif p_val < alpha:
                                marker = '*'
                            else:
                                continue
                            
                            max_mean = max(group_means)
                            marker_y = max_mean * 1.20
                            significant_bins.append((bin_start, marker, marker_y, p_val))
                            ax.text(bin_start, marker_y, marker, ha='center', va='bottom', 
                                   fontsize=11, fontweight='bold', color='black')
                    
                    if significant_bins:
                        ax.text(0.02, 0.98, "Post-hoc: Bonferroni (* p<0.05, ** p<0.01, *** p<0.001)", 
                               transform=ax.transAxes, verticalalignment='top',
                               horizontalalignment='left', fontsize=8, color='black',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Set x-axis limit if time window specified
            if hasattr(self, 'time_window'):
                ax.set_xlim(-2, self.time_window + 2)
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="💾 Save Figure", 
                  command=lambda: self.save_figure(fig)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="📊 Export Data", 
                  command=lambda: self.export_timecourse_data(behavior_name)).pack(side='left', padx=5)
    
    def create_total_time_graph(self, notebook, behavior_name=""):
        """Create total time bar plot with individual subjects and SEM error bars"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Total Time")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use filtered data (specific to this behavior)
        df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
        
        if 'Total_Time_s' in df.columns:
            # Sum total time per subject
            total_per_subject = df.groupby(['Subject', 'Treatment'])['Total_Time_s'].sum().reset_index()
            
            # Use stored treatment order and colors
            treatments = self.treatment_order if hasattr(self, 'treatment_order') else total_per_subject['Treatment'].unique()
            treatment_colors = self.treatment_colors if hasattr(self, 'treatment_colors') else {}
            
            # Calculate positions for bars
            treatment_positions = {t: i for i, t in enumerate(treatments)}
            
            # Plot individual subjects as scatter points
            for treatment in treatments:
                data = total_per_subject[total_per_subject['Treatment'] == treatment]
                x_pos = treatment_positions[treatment]
                color = treatment_colors.get(treatment, 'gray')
                
                # Handle white with black outline
                if color == 'white_black':
                    # Individual subjects as points with jitter
                    np.random.seed(42)
                    jitter = np.random.normal(0, 0.05, len(data))
                    ax.scatter(
                        [x_pos] * len(data) + jitter,
                        data['Total_Time_s'],
                        alpha=0.8,
                        s=80,
                        facecolors='white',
                        edgecolors='black',
                        linewidth=2,
                        zorder=3
                    )
                else:
                    # Individual subjects as points with jitter
                    np.random.seed(42)
                    jitter = np.random.normal(0, 0.05, len(data))
                    ax.scatter(
                        [x_pos] * len(data) + jitter,
                        data['Total_Time_s'],
                        alpha=0.6,
                        s=80,
                        color=color,
                        edgecolors='black',
                        linewidth=1,
                        zorder=3
                    )
            
            # Add bar plot with mean and error (SEM or SD)
            error_type = self.error_type if hasattr(self, 'error_type') else 'SEM'
            
            x = np.arange(len(treatments))
            means = [total_per_subject[total_per_subject['Treatment'] == t]['Total_Time_s'].mean() 
                    for t in treatments]
            
            if error_type == 'SD':
                errors = [total_per_subject[total_per_subject['Treatment'] == t]['Total_Time_s'].std() 
                         for t in treatments]
            else:  # SEM
                errors = [total_per_subject[total_per_subject['Treatment'] == t]['Total_Time_s'].sem() 
                         for t in treatments]
            
            # Handle white_black for bars
            bar_colors = []
            bar_edgecolors = []
            for t in treatments:
                c = treatment_colors.get(t, 'gray')
                if c == 'white_black':
                    bar_colors.append('white')
                    bar_edgecolors.append('black')
                else:
                    bar_colors.append(c)
                    bar_edgecolors.append('black')
            
            bars = ax.bar(
                x, 
                means, 
                yerr=errors, 
                capsize=8,
                alpha=0.7,
                color=bar_colors,
                edgecolor=bar_edgecolors,
                linewidth=2,
                zorder=2
            )
            
            ax.set_xticks(x)
            ax.set_xticklabels(treatments, fontsize=11)
            ylabel = f'Total Time in Behavior (seconds) ± {error_type}'
            ax.set_ylabel(ylabel, fontsize=12)
            title = f'{behavior_name} - Total Behavior Time by Treatment' if behavior_name else 'Total Behavior Time by Treatment'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, zorder=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add statistical testing
            if self.enable_stats_var.get():
                data_by_treatment = {t: total_per_subject[total_per_subject['Treatment'] == t]['Total_Time_s'].values
                                    for t in treatments}
                stats_results = self.perform_statistical_test(data_by_treatment, treatments)
                if stats_results:
                    y_max = max(means) + max(errors) if errors else max(means)
                    self.add_significance_markers(ax, x, stats_results, treatments, y_max)
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="💾 Save Figure", 
                  command=lambda: self.save_figure(fig)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="📊 Export Data", 
                  command=lambda: self.export_total_time_data(behavior_name)).pack(side='left', padx=5)
    
    def create_bout_analysis_graph(self, notebook, behavior_name=""):
        """Create bout analysis boxplots with individual subjects and mean line"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Bout Analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use filtered data (specific to this behavior)
        df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
        
        if 'N_Bouts' in df.columns:
            # Total bouts per subject
            total_bouts = df.groupby(['Subject', 'Treatment'])['N_Bouts'].sum().reset_index()
            treatments = self.treatment_order if hasattr(self, 'treatment_order') else total_bouts['Treatment'].unique()
            treatment_colors = self.treatment_colors if hasattr(self, 'treatment_colors') else {}
            
            # Prepare data for boxplot
            bout_data = [total_bouts[total_bouts['Treatment'] == t]['N_Bouts'].values 
                        for t in treatments]
            
            # Calculate means for each treatment
            means = [np.mean(data) for data in bout_data]
            
            # Create boxplot (without showing median, we'll add mean line)
            bp1 = ax1.boxplot(
                bout_data,
                labels=treatments,
                patch_artist=True,
                showmeans=False,
                showfliers=False,  # We'll show outliers as individual points
                boxprops=dict(facecolor='lightblue', alpha=0.5),
                medianprops=dict(linewidth=0),  # Hide median line
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                widths=0.6
            )
            
            # Color boxes and add MEAN lines
            for i, (patch, treatment) in enumerate(zip(bp1['boxes'], treatments)):
                color = treatment_colors.get(treatment, 'gray')
                if color == 'white_black':
                    patch.set_facecolor('white')
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)
                else:
                    patch.set_facecolor(color)
                patch.set_alpha(0.5)
                
                # Add mean line (thick)
                line_color = 'black' if color == 'white_black' else color
                ax1.hlines(means[i], i + 0.7, i + 1.3, colors=line_color, linewidth=3, zorder=4, label='Mean' if i == 0 else '')
            
            # Add individual subject points
            for i, treatment in enumerate(treatments):
                data = total_bouts[total_bouts['Treatment'] == treatment]['N_Bouts'].values
                color = treatment_colors.get(treatment, 'gray')
                # Add jitter
                np.random.seed(42)
                x_jitter = np.random.normal(i + 1, 0.04, size=len(data))
                
                if color == 'white_black':
                    ax1.scatter(
                        x_jitter,
                        data,
                        alpha=0.8,
                        s=100,
                        facecolors='white',
                        edgecolors='black',
                        linewidth=2,
                        zorder=3
                    )
                else:
                    ax1.scatter(
                        x_jitter,
                        data,
                        alpha=0.7,
                        s=100,
                        color=color,
                        edgecolors='black',
                        linewidth=1.5,
                        zorder=3
                    )
            
            ax1.set_ylabel('Total Number of Bouts', fontsize=12)
            title = f'{behavior_name} - Bout Count by Treatment' if behavior_name else 'Bout Count by Treatment'
            ax1.set_title(title, fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
        
        if 'Mean_Bout_Duration_s' in df.columns:
            # Mean bout duration per subject (averaged across bins)
            bout_dur = df.groupby(['Subject', 'Treatment'])['Mean_Bout_Duration_s'].mean().reset_index()
            treatments = self.treatment_order if hasattr(self, 'treatment_order') else bout_dur['Treatment'].unique()
            treatment_colors = self.treatment_colors if hasattr(self, 'treatment_colors') else {}
            
            # Prepare data for boxplot
            dur_data = [bout_dur[bout_dur['Treatment'] == t]['Mean_Bout_Duration_s'].values 
                       for t in treatments]
            
            # Calculate means for each treatment
            means = [np.mean(data) for data in dur_data]
            
            # Create boxplot (without showing median, we'll add mean line)
            bp2 = ax2.boxplot(
                dur_data,
                labels=treatments,
                patch_artist=True,
                showmeans=False,
                showfliers=False,
                boxprops=dict(facecolor='lightgreen', alpha=0.5),
                medianprops=dict(linewidth=0),  # Hide median line
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                widths=0.6
            )
            
            # Color boxes and add MEAN lines
            for i, (patch, treatment) in enumerate(zip(bp2['boxes'], treatments)):
                color = treatment_colors.get(treatment, 'gray')
                if color == 'white_black':
                    patch.set_facecolor('white')
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)
                else:
                    patch.set_facecolor(color)
                patch.set_alpha(0.5)
                
                # Add mean line (thick)
                line_color = 'black' if color == 'white_black' else color
                ax2.hlines(means[i], i + 0.7, i + 1.3, colors=line_color, linewidth=3, zorder=4)
            
            # Add individual subject points
            for i, treatment in enumerate(treatments):
                data = bout_dur[bout_dur['Treatment'] == treatment]['Mean_Bout_Duration_s'].values
                color = treatment_colors.get(treatment, 'gray')
                # Add jitter
                np.random.seed(42)
                x_jitter = np.random.normal(i + 1, 0.04, size=len(data))
                
                if color == 'white_black':
                    ax2.scatter(
                        x_jitter,
                        data,
                        alpha=0.8,
                        s=100,
                        facecolors='white',
                        edgecolors='black',
                        linewidth=2,
                        zorder=3
                    )
                else:
                    ax2.scatter(
                        x_jitter,
                        data,
                        alpha=0.7,
                        s=100,
                        color=color,
                        edgecolors='black',
                        linewidth=1.5,
                        zorder=3
                    )
            
            ax2.set_ylabel('Mean Bout Duration (seconds)', fontsize=12)
            title = f'{behavior_name} - Bout Duration by Treatment' if behavior_name else 'Bout Duration by Treatment'
            ax2.set_title(title, fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Add statistical testing for duration
            if self.enable_stats_var.get():
                data_by_treatment = {t: bout_dur[bout_dur['Treatment'] == t]['Mean_Bout_Duration_s'].values
                                    for t in treatments}
                stats_results = self.perform_statistical_test(data_by_treatment, treatments)
                if stats_results:
                    y_max = max([d.max() for d in dur_data if len(d) > 0])
                    # x positions for boxplot are 1, 2, 3, ...
                    x_positions = np.arange(1, len(treatments) + 1)
                    self.add_significance_markers(ax2, x_positions, stats_results, treatments, y_max)
        
        # Add stats to bout count (ax1)
        if 'N_Bouts' in df.columns and self.enable_stats_var.get():
            total_bouts = df.groupby(['Subject', 'Treatment'])['N_Bouts'].sum().reset_index()
            treatments = self.treatment_order if hasattr(self, 'treatment_order') else total_bouts['Treatment'].unique()
            data_by_treatment = {t: total_bouts[total_bouts['Treatment'] == t]['N_Bouts'].values
                                for t in treatments}
            stats_results = self.perform_statistical_test(data_by_treatment, treatments)
            if stats_results:
                y_max = max([d.max() for d in bout_data if len(d) > 0])
                x_positions = np.arange(1, len(treatments) + 1)
                self.add_significance_markers(ax1, x_positions, stats_results, treatments, y_max)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="💾 Save Figure", 
                  command=lambda: self.save_figure(fig)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="📊 Export Data", 
                  command=lambda: self.export_bout_data(behavior_name)).pack(side='left', padx=5)
    
    
    def create_phase_analysis_graph(self, notebook, behavior_name=""):
        """Create phase-specific analysis graphs (Acute Phase and Phase II on separate graphs)"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Phase Analysis")
        
        # Create 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use filtered data
        df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
        
        # Get phase boundaries
        acute_start = self.acute_start_var.get()
        acute_end = self.acute_end_var.get()
        phase2_start = self.phase2_start_var.get()
        phase2_end = self.phase2_end_var.get()
        
        # Calculate total time per subject in each phase
        results_list = []
        
        for subject in df['Subject'].unique():
            subject_df = df[df['Subject'] == subject]
            treatment = subject_df['Treatment'].iloc[0]
            
            # Acute phase
            acute_df = subject_df[
                (subject_df['Bin_Start_Min'] >= acute_start) & 
                (subject_df['Bin_End_Min'] <= acute_end)
            ]
            acute_time = acute_df['Total_Time_s'].sum() if 'Total_Time_s' in acute_df.columns else 0
            
            # Phase II
            phase2_df = subject_df[
                (subject_df['Bin_Start_Min'] >= phase2_start) & 
                (subject_df['Bin_End_Min'] <= phase2_end)
            ]
            phase2_time = phase2_df['Total_Time_s'].sum() if 'Total_Time_s' in phase2_df.columns else 0
            
            results_list.append({
                'Subject': subject,
                'Treatment': treatment,
                'Acute_Time': acute_time,
                'Phase2_Time': phase2_time
            })
        
        phase_df = pd.DataFrame(results_list)
        
        # Use stored treatment order and colors
        treatments = self.treatment_order if hasattr(self, 'treatment_order') else phase_df['Treatment'].unique()
        treatment_colors = self.treatment_colors if hasattr(self, 'treatment_colors') else {}
        error_type = self.error_type if hasattr(self, 'error_type') else 'SEM'
        
        # === ACUTE PHASE GRAPH (Left) ===
        x = np.arange(len(treatments))
        
        # Calculate means and errors for acute phase
        acute_means = []
        acute_errors = []
        
        for treatment in treatments:
            treat_df = phase_df[phase_df['Treatment'] == treatment]
            acute_means.append(treat_df['Acute_Time'].mean())
            
            if error_type == 'SD':
                acute_errors.append(treat_df['Acute_Time'].std())
            else:  # SEM
                acute_errors.append(treat_df['Acute_Time'].sem())
        
        # Handle bar colors
        bar_colors = []
        bar_edgecolors = []
        for t in treatments:
            c = treatment_colors.get(t, 'gray')
            if c == 'white_black':
                bar_colors.append('white')
                bar_edgecolors.append('black')
            else:
                bar_colors.append(c)
                bar_edgecolors.append('black')
        
        # Plot acute phase bars
        bars1 = ax1.bar(x, acute_means, yerr=acute_errors, capsize=8, 
                       alpha=0.7, color=bar_colors, edgecolor=bar_edgecolors, 
                       linewidth=2, zorder=2)
        
        # Add individual subject points for acute
        for i, treatment in enumerate(treatments):
            treat_df = phase_df[phase_df['Treatment'] == treatment]
            color = treatment_colors.get(treatment, 'gray')
            
            np.random.seed(42)
            jitter = np.random.normal(0, 0.05, len(treat_df))
            
            if color == 'white_black':
                ax1.scatter(
                    [i] * len(treat_df) + jitter,
                    treat_df['Acute_Time'],
                    alpha=0.8, s=80, facecolors='white', edgecolors='black',
                    linewidth=2, zorder=3
                )
            else:
                ax1.scatter(
                    [i] * len(treat_df) + jitter,
                    treat_df['Acute_Time'],
                    alpha=0.6, s=80, color=color, edgecolors='black',
                    linewidth=1, zorder=3
                )
        
        # Format acute phase graph
        ax1.set_xlabel('Treatment', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Total Time (seconds) ± {error_type}', fontsize=12, fontweight='bold')
        ax1.set_title(f'Acute Phase ({acute_start}-{acute_end} min)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(treatments, fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # === PHASE II GRAPH (Right) ===
        
        # Calculate means and errors for Phase II
        phase2_means = []
        phase2_errors = []
        
        for treatment in treatments:
            treat_df = phase_df[phase_df['Treatment'] == treatment]
            phase2_means.append(treat_df['Phase2_Time'].mean())
            
            if error_type == 'SD':
                phase2_errors.append(treat_df['Phase2_Time'].std())
            else:  # SEM
                phase2_errors.append(treat_df['Phase2_Time'].sem())
        
        # Plot Phase II bars
        bars2 = ax2.bar(x, phase2_means, yerr=phase2_errors, capsize=8,
                       alpha=0.7, color=bar_colors, edgecolor=bar_edgecolors,
                       linewidth=2, zorder=2)
        
        # Add individual subject points for Phase II
        for i, treatment in enumerate(treatments):
            treat_df = phase_df[phase_df['Treatment'] == treatment]
            color = treatment_colors.get(treatment, 'gray')
            
            np.random.seed(42)
            jitter = np.random.normal(0, 0.05, len(treat_df))
            
            if color == 'white_black':
                ax2.scatter(
                    [i] * len(treat_df) + jitter,
                    treat_df['Phase2_Time'],
                    alpha=0.8, s=80, facecolors='white', edgecolors='black',
                    linewidth=2, zorder=3
                )
            else:
                ax2.scatter(
                    [i] * len(treat_df) + jitter,
                    treat_df['Phase2_Time'],
                    alpha=0.6, s=80, color=color, edgecolors='black',
                    linewidth=1, zorder=3
                )
        
        # Format Phase II graph
        ax2.set_xlabel('Treatment', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'Total Time (seconds) ± {error_type}', fontsize=12, fontweight='bold')
        ax2.set_title(f'Phase II ({phase2_start}-{phase2_end} min)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(treatments, fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add statistical testing for both phases
        if self.enable_stats_var.get():
            # Acute phase stats
            acute_data_by_treatment = {t: phase_df[phase_df['Treatment'] == t]['Acute_Time'].values
                                      for t in treatments}
            acute_stats = self.perform_statistical_test(acute_data_by_treatment, treatments)
            if acute_stats:
                y_max_acute = max(acute_means) + max(acute_errors) if acute_errors else max(acute_means)
                self.add_significance_markers(ax1, x, acute_stats, treatments, y_max_acute)
            
            # Phase II stats
            phase2_data_by_treatment = {t: phase_df[phase_df['Treatment'] == t]['Phase2_Time'].values
                                       for t in treatments}
            phase2_stats = self.perform_statistical_test(phase2_data_by_treatment, treatments)
            if phase2_stats:
                y_max_phase2 = max(phase2_means) + max(phase2_errors) if phase2_errors else max(phase2_means)
                self.add_significance_markers(ax2, x, phase2_stats, treatments, y_max_phase2)
        
        # Add overall title
        if behavior_name:
            fig.suptitle(f'{behavior_name} - Phase Analysis', fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="💾 Save Figure", 
                  command=lambda: self.save_figure(fig)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="📊 Export Data", 
                  command=lambda: self.export_phase_data(behavior_name)).pack(side='left', padx=5)
    
    def create_heatmap_graph(self, notebook, behavior_name=""):
        """Create heatmap of behavior across time and subjects, grouped by treatment"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Heatmap")
        
        # Use filtered data
        df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
        
        if 'Total_Time_s' not in df.columns:
            return
        
        # Use stored treatment order and palette
        treatment_order = self.treatment_order if hasattr(self, 'treatment_order') else sorted(df['Treatment'].unique())
        cmap = self.heatmap_palette if hasattr(self, 'heatmap_palette') else 'YlOrRd'
        
        # Get subject-treatment mapping
        subject_treatment = df[['Subject', 'Treatment']].drop_duplicates().set_index('Subject')['Treatment'].to_dict()
        
        # Sort subjects by treatment
        sorted_subjects = []
        treatment_info = []  # Store (treatment, start_idx, n_subjects)
        
        for treatment in treatment_order:
            treatment_subjects = [subj for subj, trt in subject_treatment.items() if trt == treatment]
            treatment_subjects.sort()
            if treatment_subjects:
                start_idx = len(sorted_subjects)
                n_subjects = len(treatment_subjects)
                treatment_info.append((treatment, start_idx, n_subjects))
                sorted_subjects.extend(treatment_subjects)
        
        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            values='Total_Time_s',
            index='Subject',
            columns='Bin_Start_Min',
            aggfunc='mean'
        )
        
        # Reorder to match sorted subjects
        pivot_data = pivot_data.reindex(sorted_subjects)
        
        # Dynamic sizing
        n_subjects = len(sorted_subjects)
        n_bins = len(pivot_data.columns)
        fig_height = max(7, n_subjects * 0.4)  # Slightly larger for better spacing
        fig_width = max(11, n_bins * 0.35)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # X-axis (time bins)
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns, fontsize=9)
        ax.set_xlabel('Time Bin (minutes)', fontsize=11, fontweight='bold')
        
        # Y-axis (subjects)
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index, fontsize=8)
        ax.set_ylabel('Subject', fontsize=11, fontweight='bold')
        
        # Add horizontal white lines and labels
        for i, (treatment, start_idx, n_subjects) in enumerate(treatment_info):
            # Calculate positions
            end_idx = start_idx + n_subjects
            mid_pos = start_idx + (n_subjects - 1) / 2  # Center of this group
            
            # Draw separator line between groups (except after last)
            if i < len(treatment_info) - 1:
                separator_y = end_idx - 0.5
                ax.axhline(y=separator_y, color='white', linewidth=5, linestyle='-', zorder=10)
            
            # Add treatment label on the right side, perfectly centered
            label_x = len(pivot_data.columns) + 1.2  # Further right to avoid overlap
            ax.text(
                label_x,
                mid_pos,
                f'{treatment}\n(n={n_subjects})',
                va='center',
                ha='left',
                fontsize=9,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.95
                )
            )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Time in Behavior (s)', rotation=270, labelpad=18, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # Title
        title = f'{behavior_name} - Behavior Heatmap Across Time and Subjects\n(Grouped by Treatment)' if behavior_name else 'Behavior Heatmap Across Time and Subjects\n(Grouped by Treatment)'
        ax.set_title(title,
                    fontsize=12, fontweight='bold', pad=12)
        
        # Adjust layout - more room for labels on right
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        ttk.Button(frame, text="💾 Save Figure",
                  command=lambda: self.save_figure(fig)).pack(pady=5)
    
    
    
    def create_statistics_tab(self, notebook, behavior_name, behavior_df):
        """Create statistics summary tab with tables of all statistical comparisons"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Statistics")
        
        # Create scrollable canvas
        canvas = tk.Canvas(frame, bg='white')
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title = f'Statistical Analysis Summary - {behavior_name}' if behavior_name else 'Statistical Analysis Summary'
        ttk.Label(scrollable_frame, text=title, font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Get data
        df = behavior_df
        treatments = self.treatment_order if hasattr(self, 'treatment_order') else df['Treatment'].unique()
        alpha = self.stats_alpha_var.get()
        
        # === OVERALL BEHAVIOR (TOTAL TIME) ===
        self.add_stats_section(scrollable_frame, "1. Total Behavior Time", df, treatments, 'Total_Time_s', 'sum')
        
        # === BOUT COUNT ===
        self.add_stats_section(scrollable_frame, "2. Number of Bouts", df, treatments, 'N_Bouts', 'sum')
        
        # === BOUT DURATION ===
        self.add_stats_section(scrollable_frame, "3. Mean Bout Duration", df, treatments, 'Mean_Bout_Duration_s', 'mean')
        
        # === TIME COURSE (per timepoint ANOVA) ===
        self.add_timecourse_stats_section(scrollable_frame, "4. Time Course Analysis (Per Timepoint)", df, treatments)
        
        # === PHASE ANALYSIS (if enabled) ===
        section_num = 5
        if hasattr(self, 'enable_phases_var') and self.enable_phases_var.get():
            acute_start = self.acute_start_var.get()
            acute_end = self.acute_end_var.get()
            phase2_start = self.phase2_start_var.get()
            phase2_end = self.phase2_end_var.get()
            
            # Calculate phase data
            phase_results = []
            for subject in df['Subject'].unique():
                subject_df = df[df['Subject'] == subject]
                treatment = subject_df['Treatment'].iloc[0]
                
                # Acute phase
                acute_df = subject_df[
                    (subject_df['Bin_Start_Min'] >= acute_start) & 
                    (subject_df['Bin_End_Min'] <= acute_end)
                ]
                acute_time = acute_df['Total_Time_s'].sum() if 'Total_Time_s' in acute_df.columns else 0
                
                # Phase II
                phase2_df = subject_df[
                    (subject_df['Bin_Start_Min'] >= phase2_start) & 
                    (subject_df['Bin_End_Min'] <= phase2_end)
                ]
                phase2_time = phase2_df['Total_Time_s'].sum() if 'Total_Time_s' in phase2_df.columns else 0
                
                phase_results.append({
                    'Subject': subject,
                    'Treatment': treatment,
                    'Acute_Time': acute_time,
                    'Phase2_Time': phase2_time
                })
            
            phase_df = pd.DataFrame(phase_results)
            
            self.add_stats_section(scrollable_frame, f"{section_num}. Acute Phase ({acute_start}-{acute_end} min)", 
                                 phase_df, treatments, 'Acute_Time', 'value')
            self.add_stats_section(scrollable_frame, f"{section_num+1}. Phase II ({phase2_start}-{phase2_end} min)", 
                                 phase_df, treatments, 'Phase2_Time', 'value')
        
        # Pack canvas
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Add export button
        export_btn_frame = ttk.Frame(frame)
        export_btn_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        ttk.Button(export_btn_frame, text="📊 Export Statistics to CSV", 
                  command=lambda: self.export_statistics(behavior_name)).pack()
    
    
    def add_timecourse_stats_section(self, parent, title, df, treatments):
        """Add time course statistics showing two-way ANOVA and post-hoc results"""
        from scipy import stats
        
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill='x', padx=10, pady=10)
        
        # Show two-way ANOVA if available
        if hasattr(self, 'timecourse_anova_results') and self.timecourse_anova_results:
            ttk.Label(section_frame, text="═══ Two-Way ANOVA (Time × Treatment) ═══", 
                     font=('Arial', 10, 'bold'), foreground='darkblue').pack(anchor='w', pady=5)
            
            results = self.timecourse_anova_results
            anova_table = results['anova_table']
            alpha = self.stats_alpha_var.get()
            
            # Create main effects table
            main_effects_frame = ttk.Frame(section_frame)
            main_effects_frame.pack(fill='x', pady=5, padx=20)
            
            # Headers
            headers = ['Source', 'df', 'Sum Sq', 'F-value', 'p-value', 'Significance']
            for i, header in enumerate(headers):
                ttk.Label(main_effects_frame, text=header, font=('Arial', 9, 'bold'), 
                         relief='solid', borderwidth=1, width=13).grid(row=0, column=i, sticky='ew', padx=1, pady=1)
            
            # Main effects and interaction
            sources = [
                ('Treatment', 'C(Treatment)'),
                ('Time', 'C(Time)'),
                ('Time×Treatment', 'C(Treatment):C(Time)')
            ]
            
            for row_idx, (source_name, source_key) in enumerate(sources, start=1):
                if source_key in anova_table.index:
                    row_data = anova_table.loc[source_key]
                    df_val = int(row_data['df'])
                    sum_sq = row_data['sum_sq']
                    f_val = row_data['F']
                    p_val = row_data['PR(>F)']
                    
                    if p_val < 0.001:
                        p_text = 'p < 0.001'
                        sig = '***'
                        fg = 'darkgreen'
                    elif p_val < 0.01:
                        p_text = f'p = {p_val:.4f}'
                        sig = '**'
                        fg = 'green'
                    elif p_val < alpha:
                        p_text = f'p = {p_val:.4f}'
                        sig = '*'
                        fg = 'orange'
                    else:
                        p_text = f'p = {p_val:.4f}'
                        sig = 'ns'
                        fg = 'gray'
                    
                    ttk.Label(main_effects_frame, text=source_name, relief='solid', borderwidth=1, width=13).grid(
                        row=row_idx, column=0, sticky='ew', padx=1, pady=1)
                    ttk.Label(main_effects_frame, text=str(df_val), relief='solid', borderwidth=1, width=13).grid(
                        row=row_idx, column=1, sticky='ew', padx=1, pady=1)
                    ttk.Label(main_effects_frame, text=f"{sum_sq:.2f}", relief='solid', borderwidth=1, width=13).grid(
                        row=row_idx, column=2, sticky='ew', padx=1, pady=1)
                    ttk.Label(main_effects_frame, text=f"{f_val:.3f}", relief='solid', borderwidth=1, width=13).grid(
                        row=row_idx, column=3, sticky='ew', padx=1, pady=1)
                    ttk.Label(main_effects_frame, text=p_text, relief='solid', borderwidth=1, width=13).grid(
                        row=row_idx, column=4, sticky='ew', padx=1, pady=1)
                    ttk.Label(main_effects_frame, text=sig, relief='solid', borderwidth=1, width=13,
                             foreground=fg, font=('Arial', 9, 'bold')).grid(
                        row=row_idx, column=5, sticky='ew', padx=1, pady=1)
            
            # Interpretation
            interp_text = "Interpretation: "
            if results['treatment_p'] < alpha:
                interp_text += "Treatment groups differ overall. "
            if results['time_p'] < alpha:
                interp_text += "Behavior changes over time. "
            if results['interaction_p'] < alpha:
                interp_text += "Groups show different time patterns (interaction significant)."
            else:
                interp_text += "Groups show similar time patterns (no interaction)."
            
            ttk.Label(section_frame, text=interp_text, font=('Arial', 9, 'italic'), 
                     foreground='darkblue', wraplength=700).pack(anchor='w', padx=20, pady=5)
            
            ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(section_frame, text="═══ Post-hoc Tests (Per Timepoint) ═══", 
                 font=('Arial', 10, 'bold'), foreground='darkblue').pack(anchor='w', pady=5)
        ttk.Label(section_frame, text="Tests if treatments differ at each individual timepoint. Only significant results shown.", 
                 font=('Arial', 9, 'italic'), foreground='gray').pack(anchor='w', pady=(0,5))
        
        # Get unique time bins
        time_bins = sorted(df['Bin_Start_Min'].unique())
        alpha = self.stats_alpha_var.get()
        
        # Create table for significant time bins
        table_frame = ttk.Frame(section_frame)
        table_frame.pack(fill='x', pady=5)
        
        # Headers
        ttk.Label(table_frame, text="Time Bin (min)", font=('Arial', 9, 'bold'), 
                 relief='solid', borderwidth=1, width=15).grid(row=0, column=0, sticky='ew', padx=1, pady=1)
        ttk.Label(table_frame, text="Test", font=('Arial', 9, 'bold'), 
                 relief='solid', borderwidth=1, width=18).grid(row=0, column=1, sticky='ew', padx=1, pady=1)
        ttk.Label(table_frame, text="Statistic", font=('Arial', 9, 'bold'), 
                 relief='solid', borderwidth=1, width=15).grid(row=0, column=2, sticky='ew', padx=1, pady=1)
        ttk.Label(table_frame, text="p-value", font=('Arial', 9, 'bold'), 
                 relief='solid', borderwidth=1, width=15).grid(row=0, column=3, sticky='ew', padx=1, pady=1)
        ttk.Label(table_frame, text="Significance", font=('Arial', 9, 'bold'), 
                 relief='solid', borderwidth=1, width=12).grid(row=0, column=4, sticky='ew', padx=1, pady=1)
        
        row_idx = 1
        n_significant = 0
        
        for bin_start in time_bins:
            bin_df = df[df['Bin_Start_Min'] == bin_start]
            
            # Get data for each treatment at this time bin
            groups = []
            group_means = []
            for treatment in treatments:
                treat_data = bin_df[bin_df['Treatment'] == treatment]['Total_Time_s'].values
                if len(treat_data) > 0:
                    groups.append(treat_data)
                    group_means.append(np.mean(treat_data))
            
            # Perform test if we have enough groups
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                if len(groups) == 2:
                    # t-test for 2 groups (no multiple comparison correction needed)
                    test_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                    test_name = "t-test (2 groups)"
                    stat_display = f"t={test_stat:.3f}"
                else:
                    # For 3+ groups: First ANOVA, then Tukey HSD if significant
                    f_stat, anova_p = stats.f_oneway(*groups)
                    
                    if anova_p < alpha:
                        # ANOVA significant, perform Tukey HSD
                        try:
                            from scipy.stats import tukey_hsd
                            res = tukey_hsd(*groups)
                            # Get minimum p-value (most significant comparison)
                            p_val = res.pvalue.min()
                            test_name = f"Tukey HSD ({len(groups)} groups)"
                            stat_display = f"q(min)={res.statistic.min():.3f}"
                        except (ImportError, AttributeError) as e:
                            # Fallback to Bonferroni correction
                            print(f"Warning: Tukey HSD not available ({e}), using Bonferroni")
                            n_comparisons = len(groups) * (len(groups) - 1) / 2
                            min_p = 1.0
                            for i in range(len(groups)):
                                for j in range(i+1, len(groups)):
                                    _, p = stats.ttest_ind(groups[i], groups[j])
                                    min_p = min(min_p, p)
                            p_val = min_p
                            test_name = f"Bonferroni ({len(groups)} groups)"
                            stat_display = f"p(min)={min_p:.4f}"
                    else:
                        # ANOVA not significant, skip this timepoint
                        continue
                
                # Only show if significant
                if p_val < alpha:
                    n_significant += 1
                    
                    # Determine significance marker
                    if p_val < 0.001:
                        p_display = 'p < 0.001'
                        sig = '***'
                        fg = 'darkgreen'
                    elif p_val < 0.01:
                        p_display = f'p = {p_val:.4f}'
                        sig = '**'
                        fg = 'green'
                    elif p_val < alpha:
                        p_display = f'p = {p_val:.4f}'
                        sig = '*'
                        fg = 'orange'
                    else:
                        continue
                    
                    # Find bin end for display
                    bin_end = bin_df['Bin_End_Min'].iloc[0] if 'Bin_End_Min' in bin_df.columns else bin_start + 5
                    
                    ttk.Label(table_frame, text=f"{bin_start}-{bin_end}", 
                             relief='solid', borderwidth=1, width=15).grid(
                        row=row_idx, column=0, sticky='ew', padx=1, pady=1)
                    ttk.Label(table_frame, text=test_name, 
                             relief='solid', borderwidth=1, width=18).grid(
                        row=row_idx, column=1, sticky='ew', padx=1, pady=1)
                    ttk.Label(table_frame, text=stat_display if 'stat_display' in locals() else f"{test_stat:.3f}", 
                             relief='solid', borderwidth=1, width=15).grid(
                        row=row_idx, column=2, sticky='ew', padx=1, pady=1)
                    ttk.Label(table_frame, text=p_display, 
                             relief='solid', borderwidth=1, width=15).grid(
                        row=row_idx, column=3, sticky='ew', padx=1, pady=1)
                    ttk.Label(table_frame, text=sig, 
                             relief='solid', borderwidth=1, width=12,
                             foreground=fg, font=('Arial', 9, 'bold')).grid(
                        row=row_idx, column=4, sticky='ew', padx=1, pady=1)
                    
                    row_idx += 1
        
        if n_significant == 0:
            ttk.Label(section_frame, text="No significant differences found at any timepoint", 
                     foreground='gray', font=('Arial', 9, 'italic')).pack(anchor='w', pady=5)
        else:
            summary_text = f"Found {n_significant} significant time bin(s) out of {len(time_bins)} total bins"
            ttk.Label(section_frame, text=summary_text, 
                     foreground='darkblue', font=('Arial', 9, 'bold')).pack(anchor='w', pady=5)
    
    def add_stats_section(self, parent, title, df, treatments, column, agg_method):
        """Add a statistics section with descriptive stats and test results"""
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill='x', padx=10, pady=10)
        
        # Calculate data per subject
        if agg_method == 'sum':
            per_subject = df.groupby(['Subject', 'Treatment'])[column].sum().reset_index()
        elif agg_method == 'mean':
            per_subject = df.groupby(['Subject', 'Treatment'])[column].mean().reset_index()
        else:  # 'value' - data is already per subject
            per_subject = df[['Subject', 'Treatment', column]].copy()
        
        # Descriptive statistics table
        desc_frame = ttk.Frame(section_frame)
        desc_frame.pack(fill='x', pady=5)
        
        ttk.Label(desc_frame, text="Descriptive Statistics:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Create table
        desc_table = ttk.Frame(desc_frame)
        desc_table.pack(fill='x', pady=5)
        
        headers = ['Treatment', 'N', 'Mean', 'SD', 'SEM', 'Min', 'Max']
        for i, header in enumerate(headers):
            ttk.Label(desc_table, text=header, font=('Arial', 9, 'bold'), 
                     relief='solid', borderwidth=1, width=12).grid(row=0, column=i, sticky='ew', padx=1, pady=1)
        
        for row_idx, treatment in enumerate(treatments, start=1):
            data = per_subject[per_subject['Treatment'] == treatment][column].values
            if len(data) == 0:
                continue
            
            ttk.Label(desc_table, text=treatment, relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=0, sticky='ew', padx=1, pady=1)
            ttk.Label(desc_table, text=str(len(data)), relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=1, sticky='ew', padx=1, pady=1)
            ttk.Label(desc_table, text=f"{np.mean(data):.2f}", relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=2, sticky='ew', padx=1, pady=1)
            ttk.Label(desc_table, text=f"{np.std(data, ddof=1):.2f}", relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=3, sticky='ew', padx=1, pady=1)
            
            sem = np.std(data, ddof=1) / np.sqrt(len(data))
            ttk.Label(desc_table, text=f"{sem:.2f}", relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=4, sticky='ew', padx=1, pady=1)
            ttk.Label(desc_table, text=f"{np.min(data):.2f}", relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=5, sticky='ew', padx=1, pady=1)
            ttk.Label(desc_table, text=f"{np.max(data):.2f}", relief='solid', borderwidth=1, width=12).grid(
                row=row_idx, column=6, sticky='ew', padx=1, pady=1)
        
        # Statistical test results
        ttk.Label(section_frame, text="Statistical Test:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        
        data_by_treatment = {t: per_subject[per_subject['Treatment'] == t][column].values
                            for t in treatments}
        stats_results = self.perform_statistical_test(data_by_treatment, treatments)
        
        if stats_results:
            test_frame = ttk.Frame(section_frame)
            test_frame.pack(fill='x', pady=5)
            
            # Overall test result
            test_type = stats_results['test_type']
            p_val = stats_results['p_value']
            
            if p_val < 0.001:
                p_text = 'p < 0.001'
                sig = '***'
            elif p_val < 0.01:
                p_text = f'p = {p_val:.4f}'
                sig = '**'
            elif p_val < stats_results['alpha']:
                p_text = f'p = {p_val:.4f}'
                sig = '*'
            else:
                p_text = f'p = {p_val:.4f}'
                sig = 'ns'
            
            result_text = f"{test_type}: {p_text} {sig}"
            ttk.Label(test_frame, text=result_text, font=('Arial', 10), foreground='darkblue').pack(anchor='w')
            
            # Pairwise comparisons (if ANOVA and significant)
            if 'pairwise' in stats_results and stats_results['pairwise']:
                ttk.Label(section_frame, text="Pairwise Comparisons:", 
                         font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
                
                pairwise_table = ttk.Frame(section_frame)
                pairwise_table.pack(fill='x', pady=5)
                
                # Headers
                ttk.Label(pairwise_table, text="Comparison", font=('Arial', 9, 'bold'), 
                         relief='solid', borderwidth=1, width=30).grid(row=0, column=0, sticky='ew', padx=1, pady=1)
                ttk.Label(pairwise_table, text="p-value", font=('Arial', 9, 'bold'), 
                         relief='solid', borderwidth=1, width=15).grid(row=0, column=1, sticky='ew', padx=1, pady=1)
                ttk.Label(pairwise_table, text="Significance", font=('Arial', 9, 'bold'), 
                         relief='solid', borderwidth=1, width=15).grid(row=0, column=2, sticky='ew', padx=1, pady=1)
                
                for row_idx, (comparison, result) in enumerate(stats_results['pairwise'].items(), start=1):
                    comp_text = comparison.replace('_vs_', ' vs ')
                    p = result['p_value']
                    
                    if p < 0.001:
                        p_display = 'p < 0.001'
                        sig = '***'
                    elif p < 0.01:
                        p_display = f'p = {p:.4f}'
                        sig = '**'
                    elif p < stats_results['alpha']:
                        p_display = f'p = {p:.4f}'
                        sig = '*'
                    else:
                        p_display = f'p = {p:.4f}'
                        sig = 'ns'
                    
                    ttk.Label(pairwise_table, text=comp_text, relief='solid', borderwidth=1, width=30).grid(
                        row=row_idx, column=0, sticky='ew', padx=1, pady=1)
                    ttk.Label(pairwise_table, text=p_display, relief='solid', borderwidth=1, width=15).grid(
                        row=row_idx, column=1, sticky='ew', padx=1, pady=1)
                    
                    # Color code significance
                    if sig == '***':
                        fg = 'darkgreen'
                    elif sig == '**':
                        fg = 'green'
                    elif sig == '*':
                        fg = 'orange'
                    else:
                        fg = 'gray'
                    
                    ttk.Label(pairwise_table, text=sig, relief='solid', borderwidth=1, width=15, 
                             foreground=fg, font=('Arial', 9, 'bold')).grid(
                        row=row_idx, column=2, sticky='ew', padx=1, pady=1)
        else:
            ttk.Label(section_frame, text="No statistical test performed", 
                     foreground='gray', font=('Arial', 9, 'italic')).pack(anchor='w')
    
    def export_statistics(self, behavior_name):
        """Export statistics to CSV file"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{behavior_name}_statistics.csv" if behavior_name else "statistics.csv"
        )
        
        if filename:
            # This would need to be implemented to gather all stats into a CSV
            # For now, show a message
            messagebox.showinfo("Export", "Statistics export feature coming soon!")
    
    def perform_statistical_test(self, data_by_treatment, treatments):
        """
        Perform statistical test and return results
        
        Args:
            data_by_treatment: dict {treatment: [values]}
            treatments: list of treatment names
            
        Returns:
            dict with 'test_type', 'p_value', 'significant', 'pairwise' (if ANOVA)
        """
        from scipy import stats
        
        if not self.enable_stats_var.get():
            return None
        
        # Get data as lists
        groups = [data_by_treatment[t] for t in treatments]
        
        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return None
        
        alpha = self.stats_alpha_var.get()
        test_type = self.stats_test_var.get()
        
        # Auto-select test
        if test_type == 'auto':
            test_type = 'ttest' if len(groups) == 2 else 'anova'
        
        results = {'alpha': alpha}
        
        if test_type == 'ttest' and len(groups) == 2:
            # Unpaired t-test
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            results['test_type'] = 't-test'
            results['p_value'] = p_val
            results['significant'] = p_val < alpha
            results['comparison'] = f"{treatments[0]} vs {treatments[1]}"
            
        elif test_type == 'anova' or len(groups) > 2:
            # One-way ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            results['test_type'] = 'ANOVA'
            results['p_value'] = p_val
            results['significant'] = p_val < alpha
            
            # If significant, do pairwise comparisons
            if p_val < alpha:
                pairwise = {}
                for i in range(len(treatments)):
                    for j in range(i+1, len(treatments)):
                        t_stat, p_pairwise = stats.ttest_ind(groups[i], groups[j])
                        pairwise[f"{treatments[i]}_vs_{treatments[j]}"] = {
                            'p_value': p_pairwise,
                            'significant': p_pairwise < alpha
                        }
                results['pairwise'] = pairwise
        
        return results
    
    def add_significance_markers(self, ax, x_positions, stats_results, treatments, y_max):
        """
        Add significance markers (*, **, ***) to a graph
        
        Args:
            ax: matplotlib axis
            x_positions: list or array of x positions for each treatment
            stats_results: dict from perform_statistical_test
            treatments: list of treatment names
            y_max: maximum y value for positioning markers
        """
        if not stats_results or not stats_results.get('significant'):
            return
        
        alpha = stats_results['alpha']
        
        # Determine marker symbol
        p_val = stats_results['p_value']
        if p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < alpha:
            marker = '*'
        else:
            marker = 'ns'
        
        # Position for significance marker
        y_pos = y_max * 1.1
        
        if stats_results['test_type'] == 't-test':
            # Simple line between two groups
            x1, x2 = x_positions[0], x_positions[1]
            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.text((x1 + x2) / 2, y_pos, marker, ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        elif stats_results['test_type'] == 'ANOVA' and 'pairwise' in stats_results:
            # Add pairwise comparison brackets
            pairwise = stats_results['pairwise']
            bracket_height = y_max * 0.05
            current_y = y_pos
            
            for comparison, result in pairwise.items():
                if result['significant']:
                    # Parse comparison
                    parts = comparison.split('_vs_')
                    if len(parts) == 2:
                        idx1 = treatments.index(parts[0]) if parts[0] in treatments else None
                        idx2 = treatments.index(parts[1]) if parts[1] in treatments else None
                        
                        if idx1 is not None and idx2 is not None:
                            x1, x2 = x_positions[idx1], x_positions[idx2]
                            
                            # Determine marker
                            p = result['p_value']
                            if p < 0.001:
                                mark = '***'
                            elif p < 0.01:
                                mark = '**'
                            elif p < alpha:
                                mark = '*'
                            else:
                                continue
                            
                            # Draw bracket
                            ax.plot([x1, x1, x2, x2], 
                                   [current_y, current_y + bracket_height, current_y + bracket_height, current_y],
                                   'k-', linewidth=1)
                            ax.text((x1 + x2) / 2, current_y + bracket_height, mark, 
                                   ha='center', va='bottom', fontsize=10, fontweight='bold')
                            
                            current_y += bracket_height * 2.5
    
    
    def export_timecourse_data(self, behavior_name=""):
        """Export time course data to CSV"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{behavior_name}_timecourse_data.csv" if behavior_name else "timecourse_data.csv"
        )
        
        if filename:
            df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
            
            if 'Total_Time_s' in df.columns:
                # Export with all relevant columns
                export_df = df[['Subject', 'Treatment', 'Bin_Start_Min', 'Bin_End_Min', 'Total_Time_s']].copy()
                export_df = export_df.sort_values(['Treatment', 'Subject', 'Bin_Start_Min'])
                export_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Time course data exported to:\n{filename}")
    
    def export_total_time_data(self, behavior_name=""):
        """Export total time data to CSV"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{behavior_name}_total_time_data.csv" if behavior_name else "total_time_data.csv"
        )
        
        if filename:
            df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
            
            if 'Total_Time_s' in df.columns:
                # Sum total time per subject
                export_df = df.groupby(['Subject', 'Treatment'])['Total_Time_s'].sum().reset_index()
                export_df.columns = ['Subject', 'Treatment', 'Total_Time_Seconds']
                export_df = export_df.sort_values(['Treatment', 'Subject'])
                export_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Total time data exported to:\n{filename}")
    
    def export_bout_data(self, behavior_name=""):
        """Export bout analysis data to CSV"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{behavior_name}_bout_data.csv" if behavior_name else "bout_data.csv"
        )
        
        if filename:
            df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
            
            if 'N_Bouts' in df.columns:
                # Total bouts per subject
                bout_count = df.groupby(['Subject', 'Treatment'])['N_Bouts'].sum().reset_index()
                bout_count.columns = ['Subject', 'Treatment', 'Total_Bouts']
                
                # Mean bout duration per subject
                if 'Mean_Bout_Duration_s' in df.columns:
                    bout_duration = df.groupby(['Subject', 'Treatment'])['Mean_Bout_Duration_s'].mean().reset_index()
                    bout_duration.columns = ['Subject', 'Treatment', 'Mean_Bout_Duration_Seconds']
                    
                    # Merge
                    export_df = pd.merge(bout_count, bout_duration, on=['Subject', 'Treatment'])
                else:
                    export_df = bout_count
                
                export_df = export_df.sort_values(['Treatment', 'Subject'])
                export_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Bout data exported to:\n{filename}")
    
    def export_phase_data(self, behavior_name=""):
        """Export phase analysis data to CSV"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{behavior_name}_phase_data.csv" if behavior_name else "phase_data.csv"
        )
        
        if filename:
            df = self.filtered_results_df if hasattr(self, 'filtered_results_df') else self.results_df
            
            # Get phase boundaries
            acute_start = self.acute_start_var.get()
            acute_end = self.acute_end_var.get()
            phase2_start = self.phase2_start_var.get()
            phase2_end = self.phase2_end_var.get()
            
            # Calculate phase data for each subject
            phase_results = []
            for subject in df['Subject'].unique():
                subject_df = df[df['Subject'] == subject]
                treatment = subject_df['Treatment'].iloc[0]
                
                # Acute phase
                acute_df = subject_df[
                    (subject_df['Bin_Start_Min'] >= acute_start) & 
                    (subject_df['Bin_End_Min'] <= acute_end)
                ]
                acute_time = acute_df['Total_Time_s'].sum() if 'Total_Time_s' in acute_df.columns else 0
                
                # Phase II
                phase2_df = subject_df[
                    (subject_df['Bin_Start_Min'] >= phase2_start) & 
                    (subject_df['Bin_End_Min'] <= phase2_end)
                ]
                phase2_time = phase2_df['Total_Time_s'].sum() if 'Total_Time_s' in phase2_df.columns else 0
                
                phase_results.append({
                    'Subject': subject,
                    'Treatment': treatment,
                    f'Acute_Phase_{acute_start}-{acute_end}min_Seconds': acute_time,
                    f'Phase_II_{phase2_start}-{phase2_end}min_Seconds': phase2_time
                })
            
            export_df = pd.DataFrame(phase_results)
            export_df = export_df.sort_values(['Treatment', 'Subject'])
            export_df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Phase data exported to:\n{filename}")
    
    def save_figure(self, fig):
        """Save matplotlib figure"""
        filepath = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save figure:\n{e}")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
📊 BATCH ANALYSIS & GRAPHING

PURPOSE:
Analyze behavioral predictions across multiple subjects with treatment groups.

WORKFLOW:
1. Load Key File
   - CSV or XLSX with columns: Subject, Treatment
   - Example: Subject=251114_Formalin_S1, Treatment=Formalin

2. Select Predictions Folder
   - Folder containing prediction CSV files from batch processing
   - Files should be named with subject IDs

3. Configure Settings
   - Set time bin size (e.g., 5 minutes)
   - Select metrics to calculate

4. Run Analysis
   - Calculates metrics for each subject/bin
   - Groups by treatment

5. View Results
   - Results table shows all data points
   - Export to CSV for further analysis

6. Generate Graphs
   - Time course: Behavior over time by treatment
   - Total time: Bar plot with error bars
   - Bout analysis: Count and duration
   - Heatmap: Behavior across subjects and time

METRICS:
• Total Time: Seconds spent in behavior
• Number of Bouts: Count of behavior episodes
• Mean Bout Duration: Average bout length
• AUC: Cumulative time (area under curve)
• Percentage: % of time in behavior
• Bout Frequency: Bouts per minute

TIPS:
• Ensure subject names in key file match prediction filenames
• Use consistent bin sizes for comparison
• Export high-resolution figures (PNG/PDF) for publication
        """
        
        help_window = tk.Toplevel(self)
        help_window.title("Analysis Help")
        help_window.geometry("600x700")
        
        text = tk.Text(help_window, wrap='word', padx=10, pady=10)
        text.pack(fill='both', expand=True)
        text.insert('1.0', help_text)
        text.config(state='disabled')
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
