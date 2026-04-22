"""
dialogs.py — Standalone UI windows and dialogs for PixelPaws
=============================================================
Self-contained window classes extracted from PixelPaws_GUI.py:
  - Theme                        theme state + plot colours
  - VideoPreviewWindow           video+prediction overlay viewer
  - TrainingVisualizationWindow  live CV progress plots
  - AutoLabelWindow              uncertain-frame labelling assistant
  - SideBySidePreview            side-by-side prediction review tool
  - DataQualityChecker           pre-training sanity checks
  - EthogramGenerator            behaviour ethogram + summary plots
  - ConfidenceHistogramDialog    threshold picker with live count

None of these classes reach back into the main GUI; they take the
values they need as constructor arguments.
"""

import os
import time
import re
import bisect
import threading
from bisect import bisect_left

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError:
    from tkinter import ttk

import numpy as np
import pandas as pd
import cv2

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    plt = None

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
except ImportError:
    pass

try:
    from scipy.stats import beta
except ImportError:
    beta = None

from ui_utils import _bind_tight_layout_on_resize


class Theme:
    """Theme management — delegates to ttkbootstrap when available, falls back to manual."""

    # Light themes map to ttkbootstrap theme names
    _LIGHT_THEME = 'journal'
    _DARK_THEME = 'darkly'

    # Plot colors (matplotlib doesn't use ttk styles)
    _PLOT_COLORS = {
        'light': {'plot_bg': '#ffffff', 'plot_fg': '#000000'},
        'dark':  {'plot_bg': '#2b2b2b', 'plot_fg': '#e0e0e0'},
    }

    # Full fallback dicts for non-ttkbootstrap environments
    LIGHT = {
        'bg': '#f0f0f0', 'fg': '#000000',
        'select_bg': '#0078d7', 'select_fg': '#ffffff',
        'button_bg': '#e1e1e1', 'entry_bg': '#ffffff',
        'frame_bg': '#ffffff', 'text_bg': '#ffffff',
        'highlight': '#0078d7', 'border': '#cccccc',
        'plot_bg': '#ffffff', 'plot_fg': '#000000',
    }
    DARK = {
        'bg': '#2b2b2b', 'fg': '#e0e0e0',
        'select_bg': '#0078d7', 'select_fg': '#ffffff',
        'button_bg': '#3c3c3c', 'entry_bg': '#3c3c3c',
        'frame_bg': '#2b2b2b', 'text_bg': '#1e1e1e',
        'highlight': '#0078d7', 'border': '#3c3c3c',
        'plot_bg': '#2b2b2b', 'plot_fg': '#e0e0e0',
    }

    def __init__(self, mode='light'):
        self.mode = mode
        self.colors = self.LIGHT.copy() if mode == 'light' else self.DARK.copy()

    def is_dark(self):
        return self.mode == 'dark'

    def toggle(self):
        """Toggle between light and dark mode. Returns new mode name."""
        self.mode = 'dark' if self.mode == 'light' else 'light'
        self.colors = self.DARK.copy() if self.is_dark() else self.LIGHT.copy()
        return self.mode

    @property
    def plot_colors(self):
        """Return matplotlib-safe color dict for current mode."""
        return self._PLOT_COLORS[self.mode]

    def apply_to_widget(self, widget, widget_type='frame'):
        """Apply theme to a raw tk widget (not needed for ttk widgets with ttkbootstrap)."""
        try:
            if widget_type in ['frame', 'labelframe']:
                widget.configure(bg=self.colors['frame_bg'])
            elif widget_type == 'label':
                widget.configure(bg=self.colors['frame_bg'], fg=self.colors['fg'])
            elif widget_type == 'button':
                widget.configure(bg=self.colors['button_bg'], fg=self.colors['fg'])
            elif widget_type == 'entry':
                widget.configure(bg=self.colors['entry_bg'], fg=self.colors['fg'],
                               insertbackground=self.colors['fg'])
            elif widget_type == 'text':
                widget.configure(bg=self.colors['text_bg'], fg=self.colors['fg'],
                               insertbackground=self.colors['fg'])
        except (tk.TclError, KeyError, AttributeError):
            pass


class VideoPreviewWindow:
    """Video preview window with prediction overlay"""
    
    def __init__(self, parent, video_path, dlc_path, predictions=None):
        self.window = tk.Toplevel(parent)
        self.window.title("Video Preview with Predictions")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        w, h = int(sw * 0.75), int(sh * 0.75)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        self.video_path = video_path
        self.dlc_path = dlc_path
        self.predictions = predictions
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.playing = False
        
        self.setup_ui()
        self.load_frame(0)
        
    def setup_ui(self):
        """Setup preview window UI"""
        # Video display
        self.canvas = tk.Canvas(self.window, width=960, height=540, bg='black')
        self.canvas.pack(pady=10)
        
        # Controls frame
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=10, pady=5)
        
        # Playback controls
        ttk.Button(controls, text="⏮", command=self.prev_frame, width=3).pack(side='left', padx=2)
        self.play_btn = ttk.Button(controls, text="▶", command=self.toggle_play, width=3)
        self.play_btn.pack(side='left', padx=2)
        ttk.Button(controls, text="⏭", command=self.next_frame, width=3).pack(side='left', padx=2)
        
        # Frame slider
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(controls, from_=0, to=self.total_frames-1, 
                               orient='horizontal', variable=self.frame_var,
                               command=self.on_slider_change)
        self.slider.pack(side='left', fill='x', expand=True, padx=10)
        
        # Frame info
        self.frame_label = ttk.Label(controls, text="Frame: 0 / 0")
        self.frame_label.pack(side='left', padx=5)
        
        # Behavior bouts list
        if self.predictions is not None:
            bouts_frame = ttk.LabelFrame(self.window, text="Behavior Bouts", padding=5)
            bouts_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            self.bouts_listbox = tk.Listbox(bouts_frame, height=5)
            self.bouts_listbox.pack(side='left', fill='both', expand=True)
            
            scrollbar = ttk.Scrollbar(bouts_frame, orient='vertical', 
                                     command=self.bouts_listbox.yview)
            scrollbar.pack(side='right', fill='y')
            self.bouts_listbox.config(yscrollcommand=scrollbar.set)
            
            self.bouts_listbox.bind('<<ListboxSelect>>', self.on_bout_select)
            
            # Populate bouts
            self.populate_bouts()
    
    def populate_bouts(self):
        """Find and list behavior bouts"""
        if self.predictions is None:
            return
        
        preds = np.array(self.predictions).flatten()
        
        # Find bouts (consecutive 1s)
        in_bout = False
        bout_start = 0
        
        for i, val in enumerate(preds):
            if val == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif val == 0 and in_bout:
                duration = (i - bout_start) / self.fps
                self.bouts_listbox.insert(tk.END, 
                    f"Bout: frames {bout_start}-{i-1} ({duration:.2f}s)")
                in_bout = False
        
        if in_bout:
            duration = (len(preds) - bout_start) / self.fps
            self.bouts_listbox.insert(tk.END, 
                f"Bout: frames {bout_start}-{len(preds)-1} ({duration:.2f}s)")
    
    def on_bout_select(self, event):
        """Jump to selected bout"""
        selection = self.bouts_listbox.curselection()
        if selection:
            text = self.bouts_listbox.get(selection[0])
            # Extract start frame
            import re
            match = re.search(r'frames (\d+)-', text)
            if match:
                frame = int(match.group(1))
                self.load_frame(frame)
    
    def load_frame(self, frame_num):
        """Load and display a specific frame"""
        self.current_frame = max(0, min(frame_num, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Draw prediction overlay
            if self.predictions is not None and self.current_frame < len(self.predictions):
                pred = self.predictions[self.current_frame]
                color = (0, 255, 0) if pred == 1 else (255, 0, 0)
                cv2.rectangle(frame, (10, 10), (60, 40), color, -1)
                cv2.putText(frame, "BEHAVIOR" if pred == 1 else "NO", 
                           (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Resize for display
            height, width = frame.shape[:2]
            scale = min(960/width, 540/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(480, 270, image=self.photo)
        
        # Update UI
        self.frame_var.set(self.current_frame)
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames-1}")
    
    def on_slider_change(self, value):
        """Handle slider movement"""
        if not self.playing:
            self.load_frame(int(float(value)))
    
    def prev_frame(self):
        """Go to previous frame"""
        self.load_frame(self.current_frame - 1)
    
    def next_frame(self):
        """Go to next frame"""
        self.load_frame(self.current_frame + 1)
    
    def toggle_play(self):
        """Toggle playback"""
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "▶")
        
        if self.playing:
            self.play_video()
    
    def play_video(self):
        """Play video automatically"""
        if self.playing and self.current_frame < self.total_frames - 1:
            self.load_frame(self.current_frame + 1)
            delay = int(1000 / self.fps)
            self.window.after(delay, self.play_video)
        else:
            self.playing = False
            self.play_btn.config(text="▶")


class TrainingVisualizationWindow:
    """Real-time training progress visualization"""
    
    def __init__(self, parent, theme):
        self.window = tk.Toplevel(parent)
        self.window.title("Training Progress")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.65)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.theme = theme
        
        self.fold_f1_scores = []
        self.fold_times = []
        self.current_fold = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup visualization UI"""
        # Create notebook for different plots
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # F1 Score plot
        self.f1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.f1_frame, text="F1 Scores")
        
        self.f1_fig = Figure(figsize=(8, 5), facecolor=self.theme.colors['plot_bg'], constrained_layout=True)
        self.f1_ax = self.f1_fig.add_subplot(111)
        self.f1_ax.set_facecolor(self.theme.colors['plot_bg'])
        self.f1_canvas = FigureCanvasTkAgg(self.f1_fig, self.f1_frame)
        self.f1_canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(self.f1_canvas, self.f1_fig)
        
        # Timing plot
        self.time_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_frame, text="Fold Times")
        
        self.time_fig = Figure(figsize=(8, 5), facecolor=self.theme.colors['plot_bg'], constrained_layout=True)
        self.time_ax = self.time_fig.add_subplot(111)
        self.time_ax.set_facecolor(self.theme.colors['plot_bg'])
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, self.time_frame)
        self.time_canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(self.time_canvas, self.time_fig)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(self.window, height=6, wrap=tk.WORD)
        self.status_text.pack(fill='x', padx=5, pady=5)
        
        self.update_plots()
    
    def add_fold_result(self, fold_num, f1_score, precision, recall, time_elapsed):
        """Add results from a completed fold"""
        self.fold_f1_scores.append({
            'fold': fold_num,
            'f1': f1_score,
            'precision': precision,
            'recall': recall
        })
        self.fold_times.append(time_elapsed)
        self.current_fold = fold_num
        
        self.update_plots()
        self.update_status()
    
    def update_plots(self):
        """Update all plots"""
        if not self.fold_f1_scores:
            return
        
        # F1 Score plot
        self.f1_ax.clear()
        folds = [r['fold'] for r in self.fold_f1_scores]
        f1s = [r['f1'] for r in self.fold_f1_scores]
        precs = [r['precision'] for r in self.fold_f1_scores]
        recs = [r['recall'] for r in self.fold_f1_scores]
        
        self.f1_ax.bar([f - 0.2 for f in folds], f1s, 0.2, label='F1', alpha=0.8)
        self.f1_ax.bar(folds, precs, 0.2, label='Precision', alpha=0.8)
        self.f1_ax.bar([f + 0.2 for f in folds], recs, 0.2, label='Recall', alpha=0.8)
        
        self.f1_ax.set_xlabel('Fold', color=self.theme.colors['plot_fg'])
        self.f1_ax.set_ylabel('Score', color=self.theme.colors['plot_fg'])
        self.f1_ax.set_title('Cross-Validation Scores', color=self.theme.colors['plot_fg'])
        self.f1_ax.legend()
        self.f1_ax.set_ylim([0, 1])
        self.f1_ax.tick_params(colors=self.theme.colors['plot_fg'])
        self.f1_ax.spines['bottom'].set_color(self.theme.colors['plot_fg'])
        self.f1_ax.spines['left'].set_color(self.theme.colors['plot_fg'])
        self.f1_ax.spines['top'].set_visible(False)
        self.f1_ax.spines['right'].set_visible(False)

        self.f1_canvas.draw()
        
        # Timing plot
        self.time_ax.clear()
        self.time_ax.bar(folds, self.fold_times, alpha=0.8, color='steelblue')
        self.time_ax.set_xlabel('Fold', color=self.theme.colors['plot_fg'])
        self.time_ax.set_ylabel('Time (seconds)', color=self.theme.colors['plot_fg'])
        self.time_ax.set_title('Training Time per Fold', color=self.theme.colors['plot_fg'])
        self.time_ax.tick_params(colors=self.theme.colors['plot_fg'])
        self.time_ax.spines['bottom'].set_color(self.theme.colors['plot_fg'])
        self.time_ax.spines['left'].set_color(self.theme.colors['plot_fg'])
        self.time_ax.spines['top'].set_visible(False)
        self.time_ax.spines['right'].set_visible(False)

        self.time_canvas.draw()
    
    def update_status(self):
        """Update status text"""
        if self.fold_f1_scores:
            mean_f1 = np.mean([r['f1'] for r in self.fold_f1_scores])
            mean_time = np.mean(self.fold_times)
            
            status = f"\n=== Current Progress ===\n"
            status += f"Completed Folds: {len(self.fold_f1_scores)}\n"
            status += f"Mean F1 Score: {mean_f1:.3f}\n"
            status += f"Mean Time per Fold: {mean_time:.1f}s\n"
            
            if len(self.fold_times) > 1:
                est_remaining = mean_time * (5 - len(self.fold_f1_scores))  # Assuming 5 folds
                status += f"Estimated Time Remaining: {est_remaining:.0f}s\n"
            
            self.status_text.insert(tk.END, status)
            self.status_text.see(tk.END)


class AutoLabelWindow:
    """Auto-labeling suggestion tool"""
    
    def __init__(self, parent, video_path, dlc_path, classifier_path):
        self.window = tk.Toplevel(parent)
        self.window.title("Auto-Label Suggestions")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        w, h = int(sw * 0.75), int(sh * 0.75)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        self.video_path = video_path
        self.dlc_path = dlc_path
        self.classifier_path = classifier_path
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.predictions = None
        self.probabilities = None
        self.labels = np.zeros(self.total_frames, dtype=int)  # User labels
        self.current_frame = 0
        self.uncertain_frames = []
        
        self.setup_ui()
        self.run_predictions()
    
    def setup_ui(self):
        """Setup auto-label UI"""
        # Top: Instructions
        inst_frame = ttk.Frame(self.window)
        inst_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(inst_frame, text="Review uncertain predictions and correct as needed. "
                                   "Focus on frames with probability 0.4-0.6.",
                 wraplength=800).pack()
        
        # Middle: Video display
        self.canvas = tk.Canvas(self.window, width=800, height=450, bg='black')
        self.canvas.pack(pady=10)
        
        # Controls
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls, text="⏮ Prev Uncertain", 
                  command=self.prev_uncertain).pack(side='left', padx=2)
        ttk.Button(controls, text="⏭ Next Uncertain", 
                  command=self.next_uncertain).pack(side='left', padx=2)
        
        # Labeling buttons
        label_frame = ttk.LabelFrame(controls, text="Label Current Frame", padding=5)
        label_frame.pack(side='left', padx=20)
        
        ttk.Button(label_frame, text="✓ Behavior (1)", 
                  command=lambda: self.label_frame(1)).pack(side='left', padx=2)
        ttk.Button(label_frame, text="✗ No Behavior (0)", 
                  command=lambda: self.label_frame(0)).pack(side='left', padx=2)
        
        # Info display
        self.info_label = ttk.Label(controls, text="")
        self.info_label.pack(side='left', padx=20)
        
        # Bottom: Statistics and export
        bottom_frame = ttk.Frame(self.window)
        bottom_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_label = ttk.Label(bottom_frame, text="")
        self.stats_label.pack(side='left')
        
        ttk.Button(bottom_frame, text="Export Labels", 
                  command=self.export_labels).pack(side='right', padx=5)
        ttk.Button(bottom_frame, text="Refresh Stats", 
                  command=self.update_stats).pack(side='right', padx=5)
    
    def run_predictions(self):
        """Run classifier and find uncertain frames"""
        messagebox.showinfo("Processing", "Running classifier to find uncertain frames...")
        
        # TODO: Actually run classifier here
        # For now, simulate with random data
        np.random.seed(42)
        self.probabilities = np.random.beta(2, 2, self.total_frames)
        self.predictions = (self.probabilities > 0.5).astype(int)
        
        # Find uncertain frames (probability between 0.4 and 0.6)
        self.uncertain_frames = np.where(
            (self.probabilities > 0.4) & (self.probabilities < 0.6)
        )[0].tolist()
        
        self.update_stats()
        
        if self.uncertain_frames:
            self.load_frame(self.uncertain_frames[0])
    
    def load_frame(self, frame_num):
        """Load and display frame with prediction info"""
        self.current_frame = frame_num
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if ret:
            # Add info overlay
            prob = self.probabilities[frame_num]
            pred = self.predictions[frame_num]
            label = self.labels[frame_num]
            
            # Color based on probability
            if prob < 0.4:
                color = (0, 0, 255)  # Red - confident no
            elif prob > 0.6:
                color = (0, 255, 0)  # Green - confident yes
            else:
                color = (255, 165, 0)  # Orange - uncertain
            
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Prediction: {'Behavior' if pred == 1 else 'No Behavior'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Probability: {prob:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if label != 0:  # Show user label if set
                cv2.putText(frame, f"Your Label: {'Behavior' if label == 1 else 'No Behavior'}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Resize and display
            height, width = frame.shape[:2]
            scale = min(800/width, 450/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            from PIL import Image, ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete("all")
            self.canvas.create_image(400, 225, image=self.photo)
        
        # Update info
        uncertain_idx = self.uncertain_frames.index(frame_num) if frame_num in self.uncertain_frames else -1
        if uncertain_idx >= 0:
            self.info_label.config(
                text=f"Uncertain frame {uncertain_idx + 1} of {len(self.uncertain_frames)}"
            )
    
    def _find_nearest_uncertain_idx(self):
        """Find the index of the nearest uncertain frame to current_frame."""
        if self.current_frame in self.uncertain_frames:
            return self.uncertain_frames.index(self.current_frame)
        # Binary search for nearest uncertain frame
        import bisect
        pos = bisect.bisect_left(self.uncertain_frames, self.current_frame)
        if pos == 0:
            return 0
        if pos >= len(self.uncertain_frames):
            return len(self.uncertain_frames) - 1
        # Return whichever is closer
        before = self.uncertain_frames[pos - 1]
        after = self.uncertain_frames[pos]
        if self.current_frame - before <= after - self.current_frame:
            return pos - 1
        return pos

    def prev_uncertain(self):
        """Go to previous uncertain frame"""
        if not self.uncertain_frames:
            return

        current_idx = self._find_nearest_uncertain_idx()
        prev_idx = (current_idx - 1) % len(self.uncertain_frames)
        self.load_frame(self.uncertain_frames[prev_idx])

    def next_uncertain(self):
        """Go to next uncertain frame"""
        if not self.uncertain_frames:
            return

        current_idx = self._find_nearest_uncertain_idx()
        next_idx = (current_idx + 1) % len(self.uncertain_frames)
        self.load_frame(self.uncertain_frames[next_idx])
    
    def label_frame(self, label):
        """Set label for current frame"""
        self.labels[self.current_frame] = label + 1  # Store as 1 or 2 (0 = unlabeled)
        self.update_stats()
        self.next_uncertain()
    
    def update_stats(self):
        """Update statistics display"""
        labeled = np.sum(self.labels > 0)
        behavior_count = np.sum(self.labels == 2)
        no_behavior_count = np.sum(self.labels == 1)
        
        stats = f"Labeled: {labeled} / {len(self.uncertain_frames)} uncertain frames | "
        stats += f"Behavior: {behavior_count} | No Behavior: {no_behavior_count}"
        
        self.stats_label.config(text=stats)
    
    def export_labels(self):
        """Export corrected labels"""
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if output_path:
            # Convert labels: 0=unlabeled, 1=no behavior, 2=behavior
            final_labels = np.where(self.labels == 0, self.predictions, self.labels - 1)
            
            df = pd.DataFrame({
                'frame': range(len(final_labels)),
                'label': final_labels,
                'probability': self.probabilities,
                'user_corrected': (self.labels > 0).astype(int)
            })
            
            df.to_csv(output_path, index=False)
            messagebox.showinfo("Exported", f"Labels saved to:\n{output_path}")



class SideBySidePreview:
    """Simple fast side-by-side preview"""
    
    def __init__(self, parent, video_path, predictions, probabilities, behavior_name, threshold, human_labels=None, overlay_colors=None, dlc_path=None):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Prediction Preview - {behavior_name}")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        w, h = int(sw * 0.78), int(sh * 0.78)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        # Make window more prominent
        self.window.transient()  # Independent window
        self.window.focus_force()
        self.window.lift()

        self.video_path = video_path
        self.predictions = predictions
        self.probabilities = probabilities
        self.behavior_name = behavior_name
        self.threshold = threshold
        self.human_labels = human_labels
        _oc = overlay_colors or {}
        self.color_behavior   = _oc.get('behavior',    (0, 0, 255))
        self.color_nobehavior = _oc.get('no_behavior', (0, 255, 0))
        self.dlc_path = dlc_path

        # Load DLC body part coordinates
        self.bp_xy = {}
        if self.dlc_path and os.path.isfile(self.dlc_path):
            try:
                _dlc = pd.read_hdf(self.dlc_path)
                _dlc.columns = pd.MultiIndex.from_tuples(
                    [(_c[1], _c[2]) for _c in _dlc.columns])
                for _bp in _dlc.columns.get_level_values(0).unique():
                    self.bp_xy[_bp] = (
                        _dlc[_bp]['x'].values.astype(float),
                        _dlc[_bp]['y'].values.astype(float),
                        _dlc[_bp]['likelihood'].values.astype(float),
                    )
            except Exception:
                self.bp_xy = {}

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0  # Default 1x speed
        self._last_read_frame = -1  # Track sequential reads to avoid costly seeks
        self._canvas_image_id = None  # Reuse canvas image object

        # Graph update throttling
        self.frame_counter = 0
        self.graph_update_interval = 10  # Update marker every N frames during playback
        self.graph_redraw_counter = 0
        self.graph_redraw_interval = 30  # Redraw graph every N frames during playback
        
        # Graph update lock to prevent re-entrant calls
        self.updating_graph = False
        
        # Timeline update lock to prevent callback loop
        self.updating_timeline = False
        
        # Graph window reference (initialize before setup_ui)
        self.graph_window_obj = None
        self.graph_window_var = tk.IntVar(value=1000)
        
        self.setup_ui()
        self.update_frame()
    def setup_ui(self):
        # Info
        info = ttk.Frame(self.window)
        info.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(info, text=f"Behavior: {self.behavior_name} | Threshold: {self.threshold:.3f}",
                 font=('Arial', 10, 'bold')).pack(side='left')
        
        n_pos = np.sum(self.predictions)
        pct = (n_pos / len(self.predictions)) * 100
        
        # Add human label comparison if available
        if self.human_labels is not None and len(self.human_labels) > 0:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Handle length mismatch - use minimum length
            min_length = min(len(self.human_labels), len(self.predictions))
            
            if min_length < len(self.predictions):
                print(f"Warning: Human labels ({len(self.human_labels)} frames) shorter than predictions ({len(self.predictions)} frames)")
                print(f"Comparison will only cover first {min_length} frames")
            
            # Truncate to matching length
            human_labels_subset = self.human_labels[:min_length]
            predictions_subset = self.predictions[:min_length]
            
            # Calculate metrics
            accuracy = accuracy_score(human_labels_subset, predictions_subset) * 100
            f1 = f1_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            precision = precision_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            recall = recall_score(human_labels_subset, predictions_subset, zero_division=0) * 100
            
            comparison_text = f"Detected: {n_pos} ({pct:.1f}%) | " \
                            f"Comparison ({min_length}/{len(self.predictions)} frames): " \
                            f"Acc: {accuracy:.1f}% | F1: {f1:.1f}% | " \
                            f"Prec: {precision:.1f}% | Rec: {recall:.1f}%"
            ttk.Label(info, text=comparison_text, font=('Arial', 9), 
                     foreground='blue').pack(side='right', padx=10)
        else:
            ttk.Label(info, text=f"Detected: {n_pos} frames ({pct:.1f}%)",
                     font=('Arial', 9)).pack(side='right')
        
        # Video panel (single, centered)
        video_frame = ttk.LabelFrame(self.window, text="Video with Predictions", padding=5)
        video_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.canvas_video = tk.Canvas(video_frame, bg='black')
        self.canvas_video.pack(fill='both', expand=True)
        
        # Controls
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=5, pady=5)
        
        self.play_btn = ttk.Button(controls, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side='left', padx=2)
        
        ttk.Button(controls, text="⏮ -100", command=lambda: self.jump(-100)).pack(side='left', padx=2)
        ttk.Button(controls, text="◀ -10", command=lambda: self.jump(-10)).pack(side='left', padx=2)
        ttk.Button(controls, text="▶ +10", command=lambda: self.jump(10)).pack(side='left', padx=2)
        ttk.Button(controls, text="⏭ +100", command=lambda: self.jump(100)).pack(side='left', padx=2)
        
        # Bout navigation
        ttk.Separator(controls, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(controls, text="⬅ Prev Bout", command=self.jump_to_prev_bout).pack(side='left', padx=2)
        ttk.Button(controls, text="Next Bout ➡", command=self.jump_to_next_bout).pack(side='left', padx=2)
        
        self.frame_label = ttk.Label(controls, text="Frame: 0 / 0")
        self.frame_label.pack(side='left', padx=20)
        
        # Playback speed controls
        ttk.Label(controls, text="Speed:").pack(side='left', padx=(20, 5))
        speed_frame = ttk.Frame(controls)
        speed_frame.pack(side='left')
        
        ttk.Button(speed_frame, text="0.25x", width=5,
                  command=lambda: self.set_speed(0.25)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="0.5x", width=4,
                  command=lambda: self.set_speed(0.5)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="1x", width=4,
                  command=lambda: self.set_speed(1.0)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="2x", width=4,
                  command=lambda: self.set_speed(2.0)).pack(side='left', padx=1)
        ttk.Button(speed_frame, text="5x", width=4,
                  command=lambda: self.set_speed(5.0)).pack(side='left', padx=1)
        
        self.speed_label = ttk.Label(controls, text=f"{self.playback_speed:.0f}x", 
                                     font=('Arial', 9, 'bold'))
        self.speed_label.pack(side='left', padx=5)
        
        # Show Graph button on the right
        ttk.Button(controls, text="📊 Show Graph",
                  command=self.open_graph_window).pack(side='right', padx=5)

        # Show DLC Points toggle
        self.show_dlc_var = tk.BooleanVar(value=False)
        self._dlc_cb = ttk.Checkbutton(controls, text="Show DLC Points",
                                        variable=self.show_dlc_var,
                                        command=self._on_dlc_toggle)
        self._dlc_cb.pack(side='right', padx=5)
        if not self.bp_xy:
            self._dlc_cb.configure(state='disabled')
        
        # Timeline
        timeline_frame = ttk.Frame(self.window)
        timeline_frame.pack(fill='x', padx=5, pady=5)
        
        self.timeline = ttk.Scale(timeline_frame, from_=0, to=self.total_frames-1,
                                 orient='horizontal', command=self.on_timeline)
        self.timeline.pack(fill='x', expand=True)
        
        # Prediction bars
        self.pred_canvas = tk.Canvas(timeline_frame, height=30, bg='white')
        self.pred_canvas.pack(fill='x', expand=True, pady=2)
        self.pred_canvas.bind('<Button-1>', self.on_pred_click)
        self.draw_pred_timeline()
        
        # Keys
        self.window.bind('<space>', lambda e: self.toggle_play())
        self.window.bind('<Left>', lambda e: self.jump(-1))
        self.window.bind('<Right>', lambda e: self.jump(1))
    
    def draw_pred_timeline(self):
        self.pred_canvas.delete('all')
        w = self.pred_canvas.winfo_width() or 1200
        h = 30
        
        # Simple bars
        downsample = max(1, len(self.predictions) // 1000)
        for i in range(0, len(self.predictions), downsample):
            if self.predictions[i] == 1:
                x = (i / len(self.predictions)) * w
                self.pred_canvas.create_line(x, 0, x, h, fill='red', width=1)
        
        self.update_marker()
    
    def update_marker(self):
        self.pred_canvas.delete('marker')
        w = self.pred_canvas.winfo_width() or 1200
        x = (self.current_frame / self.total_frames) * w
        self.pred_canvas.create_line(x, 0, x, 30, fill='blue', width=2, tags='marker')
    
    def on_pred_click(self, event):
        """Handle click on prediction timeline"""
        w = self.pred_canvas.winfo_width() or 1200
        frame = int((event.x / w) * self.total_frames)
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update_frame()
    
    def update_frame(self):
        if self.current_frame < 0:
            self.current_frame = 0
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames - 1
            self.playing = False
            self.play_btn.config(text="▶ Play")
            return
        
        # Only seek if non-sequential (seeking is expensive — decodes from keyframe)
        if self.current_frame != self._last_read_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self._last_read_frame = self.current_frame

        if ret:
            # Draw overlays directly on frame (not reused)
            frame_display = frame
            if self.current_frame < len(self.predictions):
                pred = self.predictions[self.current_frame]
                prob = self.probabilities[self.current_frame]
                
                h, w = frame_display.shape[:2]
                if pred == 1:
                    color = self.color_behavior
                    text = "BEHAVIOR DETECTED"
                else:
                    color = self.color_nobehavior
                    text = "No Behavior"
                
                # Draw text overlay
                cv2.putText(frame_display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame_display, f"Probability: {prob:.3f}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame_display, f"Frame: {self.current_frame}", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show human label comparison if available
                if self.human_labels is not None and self.current_frame < len(self.human_labels):
                    human_label = int(self.human_labels[self.current_frame])
                    
                    # Check if prediction matches human label
                    if pred == human_label:
                        match_text = "CORRECT"
                        match_color = (0, 255, 0)  # Green
                    else:
                        match_text = "MISMATCH"
                        match_color = (0, 165, 255)  # Orange
                    
                    human_text = f"Human: {'Behavior' if human_label == 1 else 'No Behavior'}"
                    cv2.putText(frame_display, human_text, (20, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame_display, match_text, (20, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, match_color, 2)
            
            # DLC body part dots
            if self.show_dlc_var.get() and self.bp_xy:
                fi = self.current_frame
                for _bp, (_xs, _ys, _ps) in self.bp_xy.items():
                    if fi < len(_xs):
                        _conf = float(_ps[fi])
                        if _conf > 0.3:
                            _x, _y = int(_xs[fi]), int(_ys[fi])
                            _r = max(3, int(7 * _conf))
                            cv2.circle(frame_display, (_x, _y), _r, (0, 255, 255), -1)
                            cv2.circle(frame_display, (_x, _y), _r + 1, (255, 255, 255), 1)

            self.show_frame(frame_display, self.canvas_video)
        
        # Update controls (protect from callback loop)
        self.updating_timeline = True
        self.timeline.set(self.current_frame)
        self.updating_timeline = False
        
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
        
        # Only update marker every N frames during playback to reduce lag
        if self.playing:
            self.frame_counter += 1
            if self.frame_counter >= self.graph_update_interval:
                self.update_marker()
                self.frame_counter = 0
        else:
            # Always update when paused
            self.update_marker()
        
        if self.playing:
            # Calculate frame step — skip frames when speed outpaces render rate
            target_delay = 1000 / (self.fps * self.playback_speed)
            MIN_DELAY = 15  # ms floor to keep UI responsive
            if target_delay >= MIN_DELAY:
                frame_step = 1
                delay_ms = int(target_delay)
            else:
                frame_step = max(1, round(MIN_DELAY / target_delay))
                delay_ms = MIN_DELAY
            self.current_frame += frame_step
            self.window.after(delay_ms, self.update_frame)
    
    def show_frame(self, frame, canvas):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cw = canvas.winfo_width() or 640
        ch = canvas.winfo_height() or 480
        
        h, w = frame_rgb.shape[:2]
        aspect = w / h
        
        if cw / ch > aspect:
            nh = ch
            nw = int(ch * aspect)
        else:
            nw = cw
            nh = int(cw / aspect)
        
        frame_resized = cv2.resize(frame_rgb, (nw, nh))
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)
        
        x = (cw - nw) // 2
        y = (ch - nh) // 2

        # Reuse existing canvas image item if possible
        if not hasattr(self, '_canvas_image_id') or self._canvas_image_id is None:
            canvas.delete('all')
            self._canvas_image_id = canvas.create_image(x, y, anchor='nw', image=photo)
        else:
            canvas.coords(self._canvas_image_id, x, y)
            canvas.itemconfig(self._canvas_image_id, image=photo)
        canvas.image = photo  # prevent GC
    
    def _on_dlc_toggle(self):
        """Redraw current frame when DLC point visibility changes."""
        if not self.playing:
            self.update_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸ Pause")
            self.frame_counter = 0  # Reset counter
            self.update_frame()
        else:
            self.play_btn.config(text="▶ Play")
            # Update graph when stopped
            self.update_marker()  # Update marker immediately
            self.window.after(50, self.safe_update_graph)  # Update graph after brief delay

    def set_speed(self, speed):
        """Set playback speed multiplier"""
        self.playback_speed = speed
        if speed < 1:
            self.speed_label.config(text=f"{speed:g}x")
        else:
            self.speed_label.config(text=f"{speed:.0f}x")
    
    def jump(self, delta):
        self.current_frame += delta
        self.update_frame()
        # Update graph after jumping
        self.window.after(100, self.safe_update_graph)
    
    def jump_to_next_bout(self):
        """Jump to start of next behavior bout"""
        # Find all bouts (continuous sequences of predictions == 1)
        bouts = []
        in_bout = False
        bout_start = None
        
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif self.predictions[i] == 0 and in_bout:
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        if in_bout:  # Close final bout
            bouts.append((bout_start, len(self.predictions) - 1))
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected in video.")
            return
        
        # Find next bout after current frame
        for start, end in bouts:
            if start > self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.window.after(100, self.safe_update_graph)
                return
        
        # No bout found forward, wrap to first bout
        self.current_frame = bouts[0][0]
        self.update_frame()
        self.window.after(100, self.safe_update_graph)
        messagebox.showinfo("Wrapped", f"Jumped to first bout (frame {bouts[0][0]})")
    
    def jump_to_prev_bout(self):
        """Jump to start of previous behavior bout"""
        # Find all bouts
        bouts = []
        in_bout = False
        bout_start = None
        
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and not in_bout:
                bout_start = i
                in_bout = True
            elif self.predictions[i] == 0 and in_bout:
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        if in_bout:
            bouts.append((bout_start, len(self.predictions) - 1))
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected in video.")
            return
        
        # Find previous bout before current frame
        for start, end in reversed(bouts):
            if start < self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.window.after(100, self.safe_update_graph)
                return
        
        # No bout found backward, wrap to last bout
        self.current_frame = bouts[-1][0]
        self.update_frame()
        self.window.after(100, self.safe_update_graph)
        messagebox.showinfo("Wrapped", f"Jumped to last bout (frame {bouts[-1][0]})")
    
    def jump_to_frame_input(self):
        """Jump to frame from input box"""
        try:
            frame = int(self.jump_frame_var.get())
            if 0 <= frame < self.total_frames:
                self.current_frame = frame
                self.update_frame()
                self.jump_frame_var.set("")  # Clear input
                self.window.after(100, self.safe_update_graph)
            else:
                messagebox.showwarning("Invalid Frame", 
                    f"Frame must be between 0 and {self.total_frames - 1}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid frame number")
    
    def on_timeline(self, value):
        if self.updating_timeline:
            return  # Ignore callbacks while we're setting the value
        
        self.current_frame = int(float(value))
        if not self.playing:
            self.update_frame()
            # Update graph after scrubbing
            self.window.after(100, self.safe_update_graph)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def open_graph_window(self):
        """Open probability graph in separate window"""
        if self.graph_window_obj and self.graph_window_obj.winfo_exists():
            # Window already open, just focus it
            self.graph_window_obj.lift()
            self.graph_window_obj.focus_force()
            self.update_graph_window()
            return
        
        # Create new window
        self.graph_window_obj = tk.Toplevel(self.window)
        self.graph_window_obj.title(f"Probability Graph - {self.behavior_name}")
        sw, sh = self.graph_window_obj.winfo_screenwidth(), self.graph_window_obj.winfo_screenheight()
        w, h = int(sw * 0.75), int(sh * 0.45)
        self.graph_window_obj.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        # Controls at top - split into two rows
        controls_container = ttk.Frame(self.graph_window_obj)
        controls_container.pack(fill='x', padx=5, pady=5)
        
        # Top row - Window size and navigation
        controls_top = ttk.Frame(controls_container)
        controls_top.pack(fill='x', pady=2)
        
        ttk.Label(controls_top, text="Window Size:").pack(side='left', padx=2)
        ttk.Spinbox(controls_top, from_=100, to=10000, increment=100,
                   textvariable=self.graph_window_var, width=8).pack(side='left', padx=2)
        ttk.Label(controls_top, text="frames").pack(side='left', padx=2)
        
        ttk.Button(controls_top, text="Refresh", 
                  command=self.update_graph_window).pack(side='left', padx=10)
        
        # Add bout navigation
        ttk.Separator(controls_top, orient='vertical').pack(side='left', padx=10, fill='y')
        ttk.Label(controls_top, text="Bouts:").pack(side='left', padx=2)
        ttk.Button(controls_top, text="⬅ Prev", 
                  command=self.jump_to_prev_bout).pack(side='left', padx=2)
        ttk.Button(controls_top, text="Next ➡", 
                  command=self.jump_to_next_bout).pack(side='left', padx=2)
        
        # Add mismatch navigation if human labels available
        if self.human_labels is not None and len(self.human_labels) > 0:
            ttk.Separator(controls_top, orient='vertical').pack(side='left', padx=10, fill='y')
            ttk.Label(controls_top, text="Mismatches:").pack(side='left', padx=2)
            
            ttk.Button(controls_top, text="⬅ Prev", 
                      command=self.jump_to_prev_mismatch).pack(side='left', padx=2)
            ttk.Button(controls_top, text="Next ➡", 
                      command=self.jump_to_next_mismatch).pack(side='left', padx=2)
        
        # Bottom row - Frame info and jump
        controls_bottom = ttk.Frame(controls_container)
        controls_bottom.pack(fill='x', pady=2)
        
        self.graph_current_frame_label = ttk.Label(controls_bottom, 
                                                    text=f"Current Frame: {self.current_frame}/{self.total_frames}",
                                                    font=('Arial', 9, 'bold'))
        self.graph_current_frame_label.pack(side='left', padx=10)
        
        ttk.Separator(controls_bottom, orient='vertical').pack(side='left', padx=10, fill='y')
        
        # Jump to frame input
        ttk.Label(controls_bottom, text="Jump to frame:").pack(side='left', padx=5)
        self.jump_frame_var = tk.StringVar()
        self.jump_frame_entry = ttk.Entry(controls_bottom, textvariable=self.jump_frame_var, width=10)
        self.jump_frame_entry.pack(side='left', padx=2)
        ttk.Button(controls_bottom, text="Go", command=self.jump_to_frame_input).pack(side='left', padx=2)
        self.jump_frame_entry.bind('<Return>', lambda e: self.jump_to_frame_input())
        
        # Add scrollbar at bottom BEFORE graph (important for pack order!)
        scrollbar_frame = ttk.Frame(self.graph_window_obj)
        scrollbar_frame.pack(fill='x', side='bottom', padx=5, pady=5)
        
        ttk.Label(scrollbar_frame, text="Timeline:").pack(side='left', padx=5)
        
        self.graph_scrollbar = ttk.Scale(scrollbar_frame, from_=0, to=self.total_frames-1,
                                         orient='horizontal', 
                                         command=self.on_graph_scrollbar)
        self.graph_scrollbar.pack(side='left', fill='x', expand=True, padx=5)
        self.graph_scrollbar.set(self.current_frame)
        
        # Frame indicator
        self.graph_frame_label = ttk.Label(scrollbar_frame, 
                                           text=f"{self.current_frame}/{self.total_frames}",
                                           width=15)
        self.graph_frame_label.pack(side='right', padx=5)
        
        # Graph canvas (pack AFTER scrollbar so scrollbar stays at bottom)
        graph_frame = ttk.Frame(self.graph_window_obj)
        graph_frame.pack(fill='both', expand=True, padx=5, pady=(5, 0))
        
        # Create matplotlib figure
        self.graph_fig = Figure(figsize=(12, 4), dpi=100, facecolor='white', constrained_layout=True)
        self.graph_ax = self.graph_fig.add_subplot(111)
        
        # Embed in window
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill='both', expand=True)
        _bind_tight_layout_on_resize(self.graph_canvas, self.graph_fig)
        
        # Bind click event to jump to frame
        self.graph_canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        # Draw initial graph
        self.update_graph_window()
    
    def on_graph_click(self, event):
        """Handle click on graph to jump to frame"""
        if event.inaxes != self.graph_ax:
            return  # Click outside plot area
        
        # Get clicked x-coordinate (frame number)
        clicked_frame = int(event.xdata)
        
        # Clamp to valid range
        clicked_frame = max(0, min(clicked_frame, self.total_frames - 1))
        
        # Jump to that frame
        self.current_frame = clicked_frame
        self.update_frame()
        self.update_graph_window()
    
    def on_graph_scrollbar(self, value):
        """Handle scrollbar movement in graph window"""
        frame = int(float(value))
        self.current_frame = frame
        self.update_frame()
        
        # Update graph and scrollbar label
        if hasattr(self, 'graph_frame_label'):
            self.graph_frame_label.config(text=f"{frame}/{self.total_frames}")
        
        # Debounce graph update - only update after user stops dragging
        if hasattr(self, '_scrollbar_update_id'):
            self.window.after_cancel(self._scrollbar_update_id)
        self._scrollbar_update_id = self.window.after(100, self.update_graph_window)
    
    def jump_to_next_mismatch(self):
        """Jump to next frame where prediction doesn't match human label"""
        if self.human_labels is None or len(self.human_labels) == 0:
            return
        
        min_len = min(len(self.predictions), len(self.human_labels))
        
        # Search forward from current frame
        for i in range(self.current_frame + 1, min_len):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # If no mismatch found forward, wrap around from beginning
        for i in range(0, self.current_frame):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # No mismatches found
        messagebox.showinfo("No Mismatches", "No mismatches found in the labeled frames!")
    
    def jump_to_prev_mismatch(self):
        """Jump to previous frame where prediction doesn't match human label"""
        if self.human_labels is None or len(self.human_labels) == 0:
            return
        
        min_len = min(len(self.predictions), len(self.human_labels))
        
        # Search backward from current frame
        for i in range(self.current_frame - 1, -1, -1):
            if i < min_len and self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # If no mismatch found backward, wrap around from end
        for i in range(min_len - 1, self.current_frame, -1):
            if self.predictions[i] != self.human_labels[i]:
                self.current_frame = i
                self.update_frame()
                self.update_graph_window()
                return
        
        # No mismatches found
        messagebox.showinfo("No Mismatches", "No mismatches found in the labeled frames!")
    
    def find_behavior_bouts(self):
        """Find all behavior bouts (continuous sequences of 1s in predictions)"""
        bouts = []
        in_bout = False
        bout_start = None
        
        for i, pred in enumerate(self.predictions):
            if pred == 1 and not in_bout:
                # Start of a new bout
                in_bout = True
                bout_start = i
            elif pred == 0 and in_bout:
                # End of bout
                bouts.append((bout_start, i - 1))
                in_bout = False
        
        # Handle bout that extends to end
        if in_bout and bout_start is not None:
            bouts.append((bout_start, len(self.predictions) - 1))
        
        return bouts
    
    def jump_to_next_bout(self):
        """Jump to the start of the next behavior bout"""
        bouts = self.find_behavior_bouts()
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected!")
            return
        
        # Find next bout after current frame
        for start, end in bouts:
            if start > self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.update_graph_window()
                return
        
        # No bout found forward, wrap to first bout
        self.current_frame = bouts[0][0]
        self.update_frame()
        self.update_graph_window()
        messagebox.showinfo("Wrapped", f"Jumped to first bout (frame {bouts[0][0]})")
    
    def jump_to_prev_bout(self):
        """Jump to the start of the previous behavior bout"""
        bouts = self.find_behavior_bouts()
        
        if not bouts:
            messagebox.showinfo("No Bouts", "No behavior bouts detected!")
            return
        
        # Find previous bout before current frame
        for start, end in reversed(bouts):
            if start < self.current_frame:
                self.current_frame = start
                self.update_frame()
                self.update_graph_window()
                return
        
        # No bout found backward, wrap to last bout
        self.current_frame = bouts[-1][0]
        self.update_frame()
        self.update_graph_window()
        messagebox.showinfo("Wrapped", f"Jumped to last bout (frame {bouts[-1][0]})")
    
    def update_graph_window(self):
        """Update the graph window if it exists"""
        if not self.graph_window_obj or not self.graph_window_obj.winfo_exists():
            return
        
        # During playback, only update every N frames to reduce lag
        if self.playing:
            self.graph_redraw_counter += 1
            if self.graph_redraw_counter < self.graph_redraw_interval:
                return  # Skip this update
            self.graph_redraw_counter = 0  # Reset counter
        
        try:
            # Clear
            self.graph_ax.clear()
            
            # Get window
            window_size = self.graph_window_var.get()
            half_window = window_size // 2
            start_frame = max(0, self.current_frame - half_window)
            end_frame = min(len(self.probabilities), self.current_frame + half_window)
            
            frames = np.arange(start_frame, end_frame)
            probs = self.probabilities[start_frame:end_frame]
            preds = self.predictions[start_frame:end_frame]
            
            # Plot probability
            self.graph_ax.plot(frames, probs, 'b-', linewidth=2, label='Probability', zorder=3)
            
            # Threshold
            self.graph_ax.axhline(y=self.threshold, color='g', linestyle='--', 
                                 linewidth=2, label=f'Threshold ({self.threshold:.3f})', zorder=2)
            
            # Behavior bouts as shaded regions
            in_bout = False
            bout_start = None
            for i, pred in enumerate(preds):
                if pred == 1 and not in_bout:
                    bout_start = frames[i]
                    in_bout = True
                elif pred == 0 and in_bout:
                    self.graph_ax.axvspan(bout_start, frames[i], alpha=0.3, color='red', zorder=1)
                    in_bout = False
            if in_bout:  # Close final bout
                self.graph_ax.axvspan(bout_start, frames[-1], alpha=0.3, color='red', zorder=1)
            
            # Current frame marker
            self.graph_ax.axvline(x=self.current_frame, color='orange', linewidth=3, 
                                 label='Current Frame', zorder=4)
            
            # Highlight mismatches if human labels available
            if self.human_labels is not None and len(self.human_labels) > 0:
                min_len = min(len(self.predictions), len(self.human_labels))
                
                # Find mismatches in visible window
                for i, frame_idx in enumerate(frames):
                    if frame_idx < min_len:
                        if self.predictions[frame_idx] != self.human_labels[frame_idx]:
                            # Draw orange vertical line for mismatch
                            self.graph_ax.axvline(x=frame_idx, color='darkorange', 
                                                alpha=0.4, linewidth=1, zorder=2)
                
                # Update mismatch counter label
                total_mismatches = np.sum(self.predictions[:min_len] != self.human_labels[:min_len])
                if hasattr(self, 'mismatch_label'):
                    self.mismatch_label.config(
                        text=f"Mismatches: {total_mismatches} ({total_mismatches/min_len*100:.1f}%)")
            
            # Formatting
            self.graph_ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
            self.graph_ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
            self.graph_ax.set_title(f'{self.behavior_name} - Frame {self.current_frame}/{self.total_frames}', 
                                   fontsize=12, fontweight='bold')
            self.graph_ax.set_ylim(-0.05, 1.05)
            self.graph_ax.set_xlim(start_frame, end_frame)
            self.graph_ax.grid(True, alpha=0.3, zorder=0)
            self.graph_ax.legend(loc='upper right', fontsize=10)
            
            # Stats
            n_detected = np.sum(preds)
            pct = (n_detected / len(preds) * 100) if len(preds) > 0 else 0
            stats_text = f'Window: {len(preds)} frames | Detected: {n_detected} ({pct:.1f}%)'
            
            # Add mismatch info to stats if available
            if self.human_labels is not None and len(self.human_labels) > 0:
                min_len = min(len(self.predictions), len(self.human_labels))
                window_mismatches = 0
                for frame_idx in frames:
                    if frame_idx < min_len and self.predictions[frame_idx] != self.human_labels[frame_idx]:
                        window_mismatches += 1
                stats_text += f'\nMismatches in view: {window_mismatches}'
            
            self.graph_ax.text(0.02, 0.98, stats_text, transform=self.graph_ax.transAxes,
                              verticalalignment='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            self.graph_canvas.draw()
            
            # Update frame labels and scrollbar
            if hasattr(self, 'graph_scrollbar'):
                self.graph_scrollbar.set(self.current_frame)
            if hasattr(self, 'graph_frame_label'):
                self.graph_frame_label.config(text=f"{self.current_frame}/{self.total_frames}")
            if hasattr(self, 'graph_current_frame_label'):
                self.graph_current_frame_label.config(text=f"Current Frame: {self.current_frame}/{self.total_frames}")
            
        except Exception as e:
            print(f"Error updating graph: {e}")
    
    def safe_update_graph(self):
        """Wrapper to update graph window safely"""
        self.update_graph_window()
    
    def update_graph(self):
        """Legacy method - redirect to window update"""
        self.update_graph_window()
    
class DataQualityChecker:
    """Pre-training data quality validation"""
    
    def __init__(self, parent, sessions):
        self.window = tk.Toplevel(parent)
        self.window.title("Data Quality Check")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.65)
        self.window.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        
        self.sessions = sessions
        self.issues = []
        
        self.setup_ui()
        self.run_checks()
    
    def setup_ui(self):
        """Setup quality check UI"""
        # Progress
        self.progress_label = ttk.Label(self.window, text="Checking data quality...")
        self.progress_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.window, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        # Results notebook
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill='both', expand=True)
        
        # Issues tab
        issues_frame = ttk.Frame(self.notebook)
        self.notebook.add(issues_frame, text="Issues")
        
        self.issues_text = scrolledtext.ScrolledText(issues_frame, wrap=tk.WORD)
        self.issues_text.pack(fill='both', expand=True)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Details")
        
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD)
        self.details_text.pack(fill='both', expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Export Report", 
                  command=self.export_report).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Close", 
                  command=self.window.destroy).pack(side='right', padx=5)
    
    def run_checks(self):
        """Run all quality checks"""
        total_checks = len(self.sessions) * 5
        current = 0
        
        for session in self.sessions:
            session_name = session['session_name']
            
            # Check 1: File existence
            current += 1
            self.progress['value'] = (current / total_checks) * 1001
            
            if not os.path.exists(session['pose_path']):
                self.add_issue("ERROR", session_name, f"Pose file not found: {session['pose_path']}")
            if not os.path.exists(session['video_path']):
                self.add_issue("ERROR", session_name, f"Video file not found: {session['video_path']}")
            if session.get('target_path') and not os.path.exists(session['target_path']):
                self.add_issue("WARNING", session_name, f"Target file not found: {session['target_path']}")
            
            # Check 2: DLC file quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: DLC quality...")
            self.window.update()
            
            try:
                dlc_data = pd.read_hdf(session['pose_path'])
                n_bodyparts = len(dlc_data.columns) // 3
                self.add_detail(session_name, f"DLC body parts: {n_bodyparts}")
                
                # Check for low-confidence tracking
                prob_cols = [col for col in dlc_data.columns if 'likelihood' in str(col).lower()]
                if prob_cols:
                    for col in prob_cols:
                        low_conf = (dlc_data[col] < 0.9).sum()
                        low_conf_pct = (low_conf / len(dlc_data)) * 100
                        if low_conf_pct > 20:
                            self.add_issue("WARNING", session_name, 
                                         f"{col}: {low_conf_pct:.1f}% frames with confidence < 0.9")
            except Exception as e:
                self.add_issue("ERROR", session_name, f"Could not read DLC file: {e}")
            
            # Check 3: Video quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: video...")
            self.window.update()
            
            try:
                cap = cv2.VideoCapture(session['video_path'])
                try:
                    if not cap.isOpened():
                        self.add_issue("ERROR", session_name, "Could not open video file")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        self.add_detail(session_name,
                                       f"Video: {n_frames} frames, {fps:.1f} fps, {width}x{height}")

                        # Check frame count match
                        if len(dlc_data) != n_frames:
                            self.add_issue("WARNING", session_name,
                                         f"Frame count mismatch: DLC={len(dlc_data)}, Video={n_frames}")
                finally:
                    cap.release()
            except Exception as e:
                self.add_issue("ERROR", session_name, f"Video error: {e}")
            
            # Check 4: Label quality
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: labels...")
            self.window.update()
            
            if session.get('target_path') and os.path.exists(session['target_path']):
                try:
                    labels = pd.read_csv(session['target_path'])
                    4
                    # Check label distribution
                    for col in labels.columns:
                        if col.lower() != 'frame':
                            positive = labels[col].sum()
                            total = len(labels)
                            positive_pct = (positive / total) * 100
                            
                            self.add_detail(session_name, 
                                          f"Behavior '{col}': {positive} frames ({positive_pct:.1f}%)")
                            
                            if positive_pct < 1:
                                self.add_issue("WARNING", session_name, 
                                             f"Very rare behavior '{col}': only {positive_pct:.2f}%")
                            elif positive_pct > 50:
                                self.add_issue("WARNING", session_name, 
                                             f"Very common behavior '{col}': {positive_pct:.1f}%")
                    
                    # Check for label inconsistencies
                    if len(labels) != len(dlc_data):
                        self.add_issue("WARNING", session_name, 
                                     f"Label count mismatch: Labels={len(labels)}, DLC={len(dlc_data)}")
                    
                    # Bout outlier detection
                    for col in labels.columns:
                        if col.lower() == 'frame':
                            continue
                        vals = labels[col].values
                        bouts, gaps = [], []
                        in_bout, bout_start = False, 0
                        for i, v in enumerate(vals):
                            if v == 1 and not in_bout:
                                bout_start = i
                                in_bout = True
                            elif v == 0 and in_bout:
                                bouts.append(i - bout_start)
                                in_bout = False
                        if in_bout:
                            bouts.append(len(vals) - bout_start)
                        for i in range(len(bouts) - 1):
                            # gap = frames between bouts (rough)
                            pass  # gap calc needs start/end pairs; skip for now
                        
                        if bouts:
                            single_frame = sum(1 for b in bouts if b <= 2)
                            if single_frame > 0:
                                pct = 100 * single_frame / len(bouts)
                                self.add_issue(
                                    "WARNING", session_name,
                                    f"'{col}': {single_frame} bout(s) ≤2 frames "
                                    f"({pct:.0f}% of bouts) — possible accidental labels")
                            
                            # Try to get FPS for duration check
                            try:
                                cap = cv2.VideoCapture(session['video_path'])
                                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                                cap.release()
                            except Exception:
                                fps = 30
                            long_thresh_frames = fps * 30  # 30 seconds
                            long_bouts = sum(1 for b in bouts if b > long_thresh_frames)
                            if long_bouts > 0:
                                self.add_issue(
                                    "WARNING", session_name,
                                    f"'{col}': {long_bouts} bout(s) >30 s "
                                    f"— possible missed label-off click")
                
                except Exception as e:
                    self.add_issue("ERROR", session_name, f"Could not read labels: {e}")
            
            # Check 5: Duplicate detection
            current += 1
            self.progress['value'] = (current / total_checks) * 100
            self.progress_label.config(text=f"Checking {session_name}: duplicates...")
            self.window.update()
            
            # Simple duplicate check based on filename
            for other in self.sessions:
                if other != session and other['session_name'] == session_name:
                    self.add_issue("ERROR", session_name, "Duplicate session name found!")
        
        self.progress_label.config(text="Quality check complete!")
        self.display_summary()
    
    def add_issue(self, level, session, message):
        """Add an issue to the list"""
        self.issues.append({
            'level': level,
            'session': session,
            'message': message
        })
        
        color_tag = 'error' if level == 'ERROR' else 'warning'
        self.issues_text.insert(tk.END, f"[{level}] {session}: {message}\n", color_tag)
    
    def add_detail(self, session, message):
        """Add detail information"""
        self.details_text.insert(tk.END, f"{session}: {message}\n")
    
    def display_summary(self):
        """Display summary of checks"""
        n_sessions = len(self.sessions)
        n_errors = len([i for i in self.issues if i['level'] == 'ERROR'])
        n_warnings = len([i for i in self.issues if i['level'] == 'WARNING'])
        
        summary = "=== Data Quality Check Summary ===\n\n"
        summary += f"Sessions checked: {n_sessions}\n"
        summary += f"Errors found: {n_errors}\n"
        summary += f"Warnings found: {n_warnings}\n\n"
        
        if n_errors == 0 and n_warnings == 0:
            summary += "✓ All checks passed! Data quality looks good.\n"
        elif n_errors == 0:
            summary += "✓ No critical errors found.\n"
            summary += f"⚠ {n_warnings} warning(s) - review recommended but not critical.\n"
        else:
            summary += f"✗ {n_errors} error(s) found - must be fixed before training!\n"
            summary += f"⚠ {n_warnings} warning(s) also found.\n"
        
        summary += "\n=== Recommendations ===\n\n"
        
        if n_errors > 0:
            summary += "1. Fix all ERROR-level issues before proceeding\n"
        if n_warnings > 0:
            summary += "2. Review WARNING-level issues - they may affect performance\n"
        
        summary += "3. Ensure all videos have matching DLC files\n"
        summary += "4. Check that behavior labels are consistent\n"
        summary += "5. Verify tracking quality is good (>80% high confidence)\n"
        
        self.summary_text.insert('1.0', summary)
        
        # Configure tags for colored text
        self.issues_text.tag_config('error', foreground='red')
        self.issues_text.tag_config('warning', foreground='orange')
    
    def export_report(self):
        """Export quality report"""
        output_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(self.summary_text.get('1.0', tk.END))
                f.write("\n\n=== ISSUES ===\n\n")
                f.write(self.issues_text.get('1.0', tk.END))
                f.write("\n\n=== DETAILS ===\n\n")
                f.write(self.details_text.get('1.0', tk.END))
            
            messagebox.showinfo("Exported", f"Report saved to:\n{output_path}")


class EthogramGenerator:
    """Generate behavior ethograms and statistics"""
    
    @staticmethod
    def generate_ethogram(predictions_dict, fps, output_folder):
        """
        Generate comprehensive ethogram analysis
        
        Args:
            predictions_dict: {behavior_name: predictions_array}
            fps: frames per second
            output_folder: where to save outputs
        """
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        for behavior, preds in predictions_dict.items():
            preds = np.array(preds).flatten()
            
            # Basic statistics
            total_frames = len(preds)
            behavior_frames = np.sum(preds)
            behavior_pct = (behavior_frames / total_frames) * 100
            behavior_time = behavior_frames / fps
            
            # Find bouts
            bouts = []
            in_bout = False
            bout_start = 0
            
            for i, val in enumerate(preds):
                if val == 1 and not in_bout:
                    bout_start = i
                    in_bout = True
                elif val == 0 and in_bout:
                    bouts.append({
                        'start_frame': bout_start,
                        'end_frame': i - 1,
                        'duration_frames': i - bout_start,
                        'duration_sec': (i - bout_start) / fps
                    })
                    in_bout = False
            
            if in_bout:
                bouts.append({
                    'start_frame': bout_start,
                    'end_frame': len(preds) - 1,
                    'duration_frames': len(preds) - bout_start,
                    'duration_sec': (len(preds) - bout_start) / fps
                })
            
            # Bout statistics
            if bouts:
                bout_durations = [b['duration_sec'] for b in bouts]
                mean_bout = np.mean(bout_durations)
                std_bout = np.std(bout_durations)
                min_bout = np.min(bout_durations)
                max_bout = np.max(bout_durations)
                
                # Inter-bout intervals
                if len(bouts) > 1:
                    intervals = []
                    for i in range(len(bouts) - 1):
                        interval = (bouts[i+1]['start_frame'] - bouts[i]['end_frame']) / fps
                        intervals.append(interval)
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                else:
                    mean_interval = np.nan
                    std_interval = np.nan
            else:
                mean_bout = std_bout = min_bout = max_bout = 0
                mean_interval = std_interval = np.nan
            
            results[behavior] = {
                'total_time_sec': behavior_time,
                'total_time_min': behavior_time / 60,
                'percentage': behavior_pct,
                'n_bouts': len(bouts),
                'mean_bout_duration': mean_bout,
                'std_bout_duration': std_bout,
                'min_bout_duration': min_bout,
                'max_bout_duration': max_bout,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'bouts': bouts
            }
        
        # Generate plots
        EthogramGenerator._plot_time_budget(results, output_folder)
        EthogramGenerator._plot_bout_distributions(results, output_folder)
        EthogramGenerator._plot_raster(predictions_dict, fps, output_folder)
        
        # Generate summary report
        EthogramGenerator._write_summary(results, output_folder)
        
        return results
    
    @staticmethod
    def _plot_time_budget(results, output_folder):
        """Plot time budget pie chart"""
        if plt is None:
            return
        
        behaviors = list(results.keys())
        times = [results[b]['total_time_min'] for b in behaviors]
        
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.pie(times, labels=behaviors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Time Budget')
        
        plt.savefig(os.path.join(output_folder, 'time_budget.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _plot_bout_distributions(results, output_folder):
        """Plot bout duration distributions"""
        if plt is None:
            return
        
        n_behaviors = len(results)
        fig, axes = plt.subplots(n_behaviors, 1, figsize=(10, 3*n_behaviors), constrained_layout=True)
        
        if n_behaviors == 1:
            axes = [axes]
        
        for ax, (behavior, data) in zip(axes, results.items()):
            if data['bouts']:
                durations = [b['duration_sec'] for b in data['bouts']]
                ax.hist(durations, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(data['mean_bout_duration'], color='red', 
                          linestyle='--', label=f"Mean: {data['mean_bout_duration']:.2f}s")
                ax.set_xlabel('Bout Duration (s)')
                ax.set_ylabel('Count')
                ax.set_title(f'{behavior} - Bout Durations')
                ax.legend()
        
        plt.savefig(os.path.join(output_folder, 'bout_distributions.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _plot_raster(predictions_dict, fps, output_folder):
        """Plot behavior raster plot"""
        if plt is None:
            return
        
        behaviors = list(predictions_dict.keys())
        n_behaviors = len(behaviors)
        
        fig, ax = plt.subplots(figsize=(12, 2*n_behaviors), constrained_layout=True)
        
        for i, behavior in enumerate(behaviors):
            preds = np.array(predictions_dict[behavior]).flatten()
            
            # Find behavior events
            events = np.where(preds == 1)[0] / fps / 60  # Convert to minutes
            
            ax.scatter(events, [i] * len(events), marker='|', s=100, alpha=0.5)
        
        ax.set_yticks(range(n_behaviors))
        ax.set_yticklabels(behaviors)
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Behavior Raster Plot')
        ax.grid(axis='x', alpha=0.3)
        
        plt.savefig(os.path.join(output_folder, 'behavior_raster.png'), dpi=300)
        plt.close()
    
    @staticmethod
    def _write_summary(results, output_folder):
        """Write text summary"""
        summary_path = os.path.join(output_folder, 'ethogram_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=== BEHAVIOR ETHOGRAM SUMMARY ===\n\n")
            
            for behavior, data in results.items():
                f.write(f"\n{behavior.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total time: {data['total_time_min']:.2f} minutes ({data['percentage']:.1f}%)\n")
                f.write(f"Number of bouts: {data['n_bouts']}\n")
                
                if data['n_bouts'] > 0:
                    f.write(f"Mean bout duration: {data['mean_bout_duration']:.2f} ± {data['std_bout_duration']:.2f} s\n")
                    f.write(f"Bout duration range: {data['min_bout_duration']:.2f} - {data['max_bout_duration']:.2f} s\n")
                    
                    if not np.isnan(data['mean_interval']):
                        f.write(f"Mean inter-bout interval: {data['mean_interval']:.2f} ± {data['std_interval']:.2f} s\n")
                
                f.write("\n")


OVERLAY_COLOR_SCHEMES = {
    'Red / Gray':          ('#E00000', '#707070'),
    'Orange / Sky Blue':   ('#FF8C00', '#00BFFF'),
    'Classic Red / Green': ('#DC0000', '#00C800'),
    'Magenta / Cyan':      ('#FF00BB', '#00DDCC'),
    'Yellow / Blue':       ('#FFD700', '#0077FF'),
    'White / Gray':        ('#FFFFFF', '#888888'),
    'Purple / Lime':       ('#AA00CC', '#99CC00'),
}


class ConfidenceHistogramDialog:
    """
    Modal Toplevel showing a 50-bin histogram of P(1) for all frames.
    Red near 0.5, blue at extremes. Two vertical dashed lines at
    0.5 +/- threshold/2. Threshold Scale widget + live eligible-count label.
    """
    def __init__(self, parent_root, probas: np.ndarray, threshold_var: tk.DoubleVar,
                 on_proceed, on_cancel):
        self.probas = probas
        self.threshold_var = threshold_var
        self.on_proceed = on_proceed
        self.on_cancel = on_cancel
        self._result = None
        self.root = parent_root

        self.win = tk.Toplevel(parent_root)
        self.win.title("Confidence Histogram — Select Threshold")
        _sw, _sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f"750x600+{(_sw-750)//2}+{(_sh-600)//2}")
        self.win.grab_set()

        self._build_ui()
        self._draw_histogram()
        # Trace threshold changes
        self._trace_id = self.threshold_var.trace_add(
            'write', lambda *_: self.root.after(0, self._on_threshold_changed))
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build_ui(self):
        # Canvas for histogram
        if MATPLOTLIB_AVAILABLE:
            self._fig, self._ax = plt.subplots(figsize=(6, 3), dpi=90, constrained_layout=True)
            self._canvas = FigureCanvasTkAgg(self._fig, master=self.win)
            self._canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=(8, 4))
            _bind_tight_layout_on_resize(self._canvas, self._fig)
        else:
            ttk.Label(self.win, text="(matplotlib not available — install it to see histogram)").pack(pady=20)

        # Threshold slider
        ctrl = ttk.Frame(self.win)
        ctrl.pack(fill='x', padx=10, pady=4)
        ttk.Label(ctrl, text="Uncertainty threshold:").pack(side='left')
        self._thresh_scale = ttk.Scale(ctrl, from_=0.05, to=1.0,
                                       variable=self.threshold_var, orient='horizontal',
                                       length=280)
        self._thresh_scale.pack(side='left', padx=6)
        self._thresh_label = ttk.Label(ctrl, text="0.30")
        self._thresh_label.pack(side='left')

        # Eligible count
        self._count_label = ttk.Label(self.win, text="", font=('Arial', 10))
        self._count_label.pack(pady=2)

        # Buttons
        btn_row = ttk.Frame(self.win)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="Proceed to Labeling",
                   command=self._proceed).pack(side='left', padx=6)
        ttk.Button(btn_row, text="Cancel",
                   command=self._cancel).pack(side='left', padx=6)

    def _draw_histogram(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        self._ax.clear()
        n, bins, patches = self._ax.hist(self.probas, bins=50, range=(0, 1), edgecolor='none')
        # Color: red near 0.5, blue at extremes
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for patch, center in zip(patches, bin_centers):
            dist = abs(center - 0.5) * 2  # 0 at boundary, 1 at extremes
            patch.set_facecolor((1.0 - dist, 0.2, dist))  # red -> blue
        # Threshold lines
        t = self.threshold_var.get()
        lo = 0.5 - t / 2
        hi = 0.5 + t / 2
        self._ax.axvline(lo, color='gold', linestyle='--', linewidth=1.5)
        self._ax.axvline(hi, color='gold', linestyle='--', linewidth=1.5)
        self._ax.set_xlabel("P(behavior=1)")
        self._ax.set_ylabel("Frame count")
        self._ax.set_title("Frame Confidence Distribution")
        self._canvas.draw()
        self._update_count_label()

    def _update_count_label(self):
        t = self.threshold_var.get()
        n_eligible = int(np.sum(np.abs(self.probas - 0.5) * 2 < t))
        self._thresh_label.config(text=f"{t:.2f}")
        self._count_label.config(text=f"{n_eligible:,} frames eligible (within uncertainty zone)")

    def _on_threshold_changed(self):
        self._draw_histogram()

    def _cleanup_trace(self):
        try:
            self.threshold_var.trace_remove('write', self._trace_id)
        except Exception:
            pass

    def _proceed(self):
        self._cleanup_trace()
        self._result = 'proceed'
        if self.on_proceed:
            self.on_proceed()
        self.win.destroy()

    def _cancel(self):
        self._cleanup_trace()
        self._result = 'cancel'
        if self.on_cancel:
            self.on_cancel()
        self.win.destroy()


