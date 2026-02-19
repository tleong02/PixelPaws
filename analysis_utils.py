# PixelPaws - Additional Analysis Utilities
# SHAP analysis, learning curves, and video export functions

import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPAnalyzer:
    """
    Perform SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    Explains which features contribute most to predictions.
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained XGBoost model
            feature_names: Names of features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def explain_model(self, X: np.ndarray, max_samples: int = 2000):
        """
        Create SHAP explainer and calculate SHAP values.
        
        Args:
            X: Feature matrix to explain
            max_samples: Maximum number of samples to use
        """
        # Sample data if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)
        
        return self.shap_values
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance using SHAP values.
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_model() first")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Get feature importance
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(mean_shap))],
                'importance': mean_shap
            })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_n} Feature Importances (SHAP)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return importance_df
    
    def plot_summary(self, save_path: str = None):
        """
        Create SHAP summary plot showing feature impacts.
        
        Args:
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("Must run explain_model() first")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            features=None,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class LearningCurveAnalyzer:
    """
    Analyze learning curves to determine if more training data is needed.
    """
    
    @staticmethod
    def calculate_learning_curve(classifier_class,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 train_sizes: List[int],
                                 n_splits: int = 5,
                                 **classifier_kwargs) -> Tuple[List[float], List[float]]:
        """
        Calculate learning curve showing F1 score vs training set size.
        
        Args:
            classifier_class: Classifier class (e.g., BehaviorClassifier)
            X: Full feature matrix
            y: Full labels
            train_sizes: List of training set sizes to test
            n_splits: Number of CV splits
            classifier_kwargs: Arguments for classifier initialization
            
        Returns:
            Tuple of (train_scores, val_scores)
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import f1_score
        
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            fold_train_scores = []
            fold_val_scores = []
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                # Subsample training data
                if size < len(train_idx):
                    train_idx = np.random.choice(train_idx, size, replace=False)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train classifier
                clf = classifier_class(**classifier_kwargs)
                clf.train(X_train, y_train, validation_split=0)
                
                # Evaluate
                y_train_pred = clf.predict(X_train)
                y_val_pred = clf.predict(X_val)
                
                fold_train_scores.append(f1_score(y_train, y_train_pred))
                fold_val_scores.append(f1_score(y_val, y_val_pred))
            
            train_scores.append(np.mean(fold_train_scores))
            val_scores.append(np.mean(fold_val_scores))
        
        return train_scores, val_scores
    
    @staticmethod
    def plot_learning_curve(train_sizes: List[int],
                           train_scores: List[float],
                           val_scores: List[float],
                           save_path: str = None):
        """
        Plot learning curve.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training F1 scores
            val_scores: Validation F1 scores
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores, 'o-', label='Training F1', linewidth=2)
        plt.plot(train_sizes, val_scores, 'o-', label='Validation F1', linewidth=2)
        
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def interpret_learning_curve(train_scores: List[float],
                                 val_scores: List[float],
                                 threshold_plateau: float = 0.02) -> str:
        """
        Interpret learning curve and provide recommendations.
        
        Args:
            train_scores: Training F1 scores
            val_scores: Validation F1 scores
            threshold_plateau: Threshold for detecting plateau
            
        Returns:
            Recommendation string
        """
        # Check if validation score is still improving
        if len(val_scores) >= 3:
            recent_improvement = val_scores[-1] - val_scores[-3]
            
            if recent_improvement > threshold_plateau:
                return ("✓ Model is still improving. Consider adding more training data "
                       "to further improve performance.")
            elif val_scores[-1] < 0.7:
                return ("⚠ Model has plateaued but performance is low. Consider: "
                       "1) Adding more diverse training data, "
                       "2) Tuning hyperparameters, "
                       "3) Engineering new features.")
            else:
                return ("✓ Model has reached good performance plateau. "
                       "Ready for use on new data.")
        
        return "Need more data points to assess learning curve."


class VideoExporter:
    """
    Export videos with behavior predictions overlaid as visual labels.
    """
    
    @staticmethod
    def create_labeled_video(video_path: str,
                           predictions: np.ndarray,
                           output_path: str,
                           behavior_name: str = "Behavior",
                           fps: float = None,
                           color: Tuple[int, int, int] = (0, 0, 255),
                           show_frame_numbers: bool = True):
        """
        Create video with behavior predictions overlaid.
        
        Args:
            video_path: Input video path
            predictions: Binary predictions array
            output_path: Output video path
            behavior_name: Name of behavior to display
            fps: Output FPS (None = same as input)
            color: RGB color for positive frames (default red)
            show_frame_numbers: Show frame numbers
        """
        # Open input video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps is None:
            fps = input_fps
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(predictions):
                # Add overlay if behavior detected
                if predictions[frame_idx] == 1:
                    # Add colored border
                    border_thickness = 10
                    cv2.rectangle(
                        frame,
                        (0, 0),
                        (width - 1, height - 1),
                        color,
                        border_thickness
                    )
                    
                    # Add text label
                    label = f"{behavior_name} DETECTED"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Add background for text
                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (10 + text_width + 20, 10 + text_height + 20),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Add text
                    cv2.putText(
                        frame,
                        label,
                        (20, 10 + text_height + 10),
                        font,
                        font_scale,
                        color,
                        thickness
                    )
                
                # Add frame number
                if show_frame_numbers:
                    frame_text = f"Frame: {frame_idx}"
                    cv2.putText(
                        frame,
                        frame_text,
                        (width - 200, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
    
    @staticmethod
    def create_comparison_video(video_path: str,
                              predictions: np.ndarray,
                              ground_truth: np.ndarray,
                              output_path: str,
                              behavior_name: str = "Behavior",
                              fps: float = None):
        """
        Create side-by-side comparison video of predictions vs ground truth.
        
        Args:
            video_path: Input video path
            predictions: Model predictions
            ground_truth: True labels
            output_path: Output video path
            behavior_name: Behavior name
            fps: Output FPS
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps is None:
            fps = input_fps
        
        # Double width for side-by-side
        output_width = width * 2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(predictions):
                # Create two copies
                frame_pred = frame.copy()
                frame_truth = frame.copy()
                
                # Add prediction overlay
                if predictions[frame_idx] == 1:
                    cv2.rectangle(frame_pred, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                    cv2.putText(frame_pred, "PREDICTED", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Add ground truth overlay
                if ground_truth[frame_idx] == 1:
                    cv2.rectangle(frame_truth, (0, 0), (width-1, height-1), (0, 255, 0), 10)
                    cv2.putText(frame_truth, "ACTUAL", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Combine side by side
                combined = np.hstack([frame_pred, frame_truth])
                
                # Add labels
                cv2.putText(combined, "Model Prediction", (50, height - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Ground Truth", (width + 50, height - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(combined)
            
            frame_idx += 1
        
        cap.release()
        out.release()


# Standalone functions for easy import

def analyze_feature_importance(model, X, feature_names, output_path=None):
    """
    Quick function to analyze and plot feature importance using SHAP.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        output_path: Where to save plot
    
    Returns:
        DataFrame with feature importances
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Install with: pip install shap")
        return None
    
    analyzer = SHAPAnalyzer(model, feature_names)
    analyzer.explain_model(X)
    importance_df = analyzer.plot_feature_importance(save_path=output_path)
    
    return importance_df


def generate_learning_curve(classifier_class, X, y, output_path=None, **kwargs):
    """
    Quick function to generate and plot learning curve.
    
    Args:
        classifier_class: Classifier class
        X: Feature matrix
        y: Labels
        output_path: Where to save plot
        **kwargs: Classifier arguments
    
    Returns:
        Tuple of (train_scores, val_scores, recommendation)
    """
    # Define training sizes
    max_samples = len(y)
    train_sizes = [
        min(50, max_samples),
        min(100, max_samples),
        min(500, max_samples),
        min(1000, max_samples),
        min(2000, max_samples),
        min(5000, max_samples),
        max_samples
    ]
    train_sizes = sorted(list(set([s for s in train_sizes if s > 0])))
    
    # Calculate curve
    print("Calculating learning curve...")
    train_scores, val_scores = LearningCurveAnalyzer.calculate_learning_curve(
        classifier_class, X, y, train_sizes, **kwargs
    )
    
    # Plot
    LearningCurveAnalyzer.plot_learning_curve(
        train_sizes, train_scores, val_scores, output_path
    )
    
    # Interpret
    recommendation = LearningCurveAnalyzer.interpret_learning_curve(
        train_scores, val_scores
    )
    
    print(f"\nLearning Curve Analysis:")
    print(f"Training sizes tested: {train_sizes}")
    print(f"Validation F1 scores: {[f'{s:.3f}' for s in val_scores]}")
    print(f"\nRecommendation: {recommendation}")
    
    return train_scores, val_scores, recommendation


def export_labeled_video(video_path, predictions, output_path, behavior_name="Behavior"):
    """
    Quick function to export video with behavior labels.
    
    Args:
        video_path: Input video
        predictions: Binary predictions
        output_path: Output video path
        behavior_name: Name of behavior
    """
    VideoExporter.create_labeled_video(
        video_path, predictions, output_path, behavior_name
    )
    print(f"Labeled video saved to: {output_path}")


if __name__ == "__main__":
    print("PixelPaws Analysis Utilities")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - SHAPAnalyzer: Explain model predictions")
    print("  - LearningCurveAnalyzer: Assess training data needs")
    print("  - VideoExporter: Create labeled videos")
    print("\nImport these classes or use the standalone functions:")
    print("  - analyze_feature_importance()")
    print("  - generate_learning_curve()")
    print("  - export_labeled_video()")
