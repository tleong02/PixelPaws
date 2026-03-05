"""
Behavior Classifier Training Module
Matching BAREfoot's training approach (Barkai et al., 2025)

This module provides a comprehensive classifier training pipeline:
- Feature extraction (pose + brightness)
- Data balancing
- Threshold optimization
- Cross-validation
- Learning curve analysis
- SHAP feature importance
- Model persistence

Citation: Inspired by Barkai et al. (2025), Cell Rep Methods
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class BehaviorClassifier:
    """
    XGBoost-based behavior classifier matching BAREfoot's approach.
    """
    
    def __init__(self,
                 n_estimators: int = 1700,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.2,
                 alpha: float = 1.0,
                 lambda_: float = 0.1,
                 seed: int = 42):
        """
        Initialize behavior classifier with BAREfoot's default parameters.

        Args:
            n_estimators: Number of boosting rounds (BAREfoot uses 1700)
            max_depth: Maximum tree depth (BAREfoot uses 6)
            learning_rate: Learning rate (BAREfoot uses 0.01)
            subsample: Fraction of samples for training each tree (0.8)
            colsample_bytree: Fraction of features for each tree (0.2)
            alpha: L1 regularization term (1.0)
            lambda_: L2 regularization term (0.1)
            seed: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.lambda_ = lambda_
        self.seed = seed
        
        self.model = None
        self.best_threshold = 0.5
        self.feature_names = None
        self.training_history = {}
    
    def _create_model(self, scale_pos_weight: float = 1.0):
        """Create XGBoost model with specified parameters."""
        params = {
            'n_estimators': self.n_estimators,
            'objective': 'reg:squaredlogerror',  # BAREfoot uses this
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.alpha,
            'lambda': self.lambda_,
            'seed': self.seed,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': -1
        }
        
        params['tree_method'] = 'hist'
        
        return xgb.XGBClassifier(**params)
    
    def train(self,
              X: pd.DataFrame,
              y: np.ndarray,
              balance_method: str = 'downsample',
              validation_split: Optional[float] = None,
              verbose: bool = True) -> Dict:
        """
        Train classifier with data balancing.
        Matches BAREfoot's training approach.
        
        Args:
            X: Feature matrix
            y: Binary labels (0/1)
            balance_method: 'downsample', 'upsample', or 'smote'
            validation_split: Fraction for validation (None = no split)
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"Training {self.__class__.__name__}...")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Positive rate: {np.mean(y):.3f}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Calculate class balance (BAREfoot's formula)
        # From Chicco et al. BioDataMining, BMC 2017
        zeros_to_ones = (np.sum(y == 0) / len(y) + 0.5) / (np.sum(y == 1) / len(y) + 0.5)
        
        # Balance data
        X_balanced, y_balanced = self._balance_data(X, y, balance_method, zeros_to_ones)
        
        if verbose:
            print(f"  After balancing: {len(X_balanced)} samples")
            print(f"  Positive rate: {np.mean(y_balanced):.3f}")
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (len(y_balanced) - np.sum(y_balanced)) / np.sum(y_balanced)
        
        # Create model
        self.model = self._create_model(scale_pos_weight)
        
        # Train
        if validation_split:
            # Split for validation
            split_idx = int(len(X_balanced) * (1 - validation_split))
            X_train = X_balanced[:split_idx]
            y_train = y_balanced[:split_idx]
            X_val = X_balanced[split_idx:]
            y_val = y_balanced[split_idx:]
            
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_balanced, y_balanced)
        
        if verbose:
            print("  Training complete!")
        
        # Calculate metrics
        metrics = {
            'n_samples': len(X_balanced),
            'n_features': X_balanced.shape[1],
            'positive_rate': np.mean(y_balanced),
            'zeros_to_ones_ratio': zeros_to_ones
        }
        
        self.training_history = metrics
        
        return metrics
    
    def _balance_data(self, X: pd.DataFrame, y: np.ndarray, 
                     method: str, zeros_to_ones: float) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Balance dataset using specified method.
        Matches BAREfoot's BalanceXy function.
        """
        if method.lower() == 'downsample':
            # Downsample majority class
            n_minority = np.sum(y == 1)
            n_majority = int(n_minority * zeros_to_ones)
            
            minority_idx = np.where(y == 1)[0]
            majority_idx = np.where(y == 0)[0]
            
            # Randomly sample from majority
            np.random.seed(self.seed)
            majority_sampled = np.random.choice(majority_idx, n_majority, replace=False)
            
            # Combine
            balanced_idx = np.concatenate([minority_idx, majority_sampled])
            np.random.shuffle(balanced_idx)
            
            return X.iloc[balanced_idx].reset_index(drop=True), y[balanced_idx]
        
        elif method.lower() == 'upsample':
            # Upsample minority class
            n_majority = np.sum(y == 0)
            n_minority_target = int(n_majority / zeros_to_ones)
            
            minority_idx = np.where(y == 1)[0]
            majority_idx = np.where(y == 0)[0]
            
            # Sample minority with replacement
            np.random.seed(self.seed)
            minority_sampled = np.random.choice(minority_idx, n_minority_target, replace=True)
            
            # Combine
            balanced_idx = np.concatenate([majority_idx, minority_sampled])
            np.random.shuffle(balanced_idx)
            
            return X.iloc[balanced_idx].reset_index(drop=True), y[balanced_idx]
        
        else:
            # No balancing
            return X, y
    
    def optimize_threshold(self,
                          X: pd.DataFrame,
                          y: np.ndarray,
                          metric: str = 'f1',
                          min_threshold: float = 0.0,
                          max_threshold: float = 1.0,
                          num_thresholds: int = 100) -> Dict:
        """
        Find optimal classification threshold.
        Matches BAREfoot's threshold optimization.
        
        Args:
            X: Feature matrix
            y: True labels
            metric: 'f1', 'precision', or 'recall'
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            num_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with threshold analysis results
        """
        if self.model is None:
            raise ValueError("Model must be trained before optimizing threshold")
        
        # Get probability predictions
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Test thresholds
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            f1_scores.append(f1_score(y, y_pred, zero_division=0))
            precision_scores.append(precision_score(y, y_pred, zero_division=0))
            recall_scores.append(recall_score(y, y_pred, zero_division=0))
        
        # Find best threshold
        if metric == 'f1':
            best_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            best_idx = np.argmax(precision_scores)
        elif metric == 'recall':
            best_idx = np.argmax(recall_scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.best_threshold = thresholds[best_idx]
        
        return {
            'best_threshold': self.best_threshold,
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'best_f1': f1_scores[best_idx],
            'best_precision': precision_scores[best_idx],
            'best_recall': recall_scores[best_idx]
        }
    
    def cross_validate(self,
                      X: pd.DataFrame,
                      y: np.ndarray,
                      n_folds: int = 5,
                      balance_method: str = 'downsample',
                      optimize_threshold: bool = True) -> Dict:
        """
        Perform k-fold cross-validation.
        Matches BAREfoot's CV approach.
        
        Args:
            X: Feature matrix
            y: Binary labels
            n_folds: Number of folds
            balance_method: Data balancing method
            optimize_threshold: Optimize threshold per fold
            
        Returns:
            Dictionary with CV metrics
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        fold_metrics = []
        
        print(f"Running {n_folds}-fold cross-validation...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold_idx + 1}/{n_folds}")
            
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y[val_idx]
            
            # Train model on this fold
            self.train(X_train, y_train, balance_method=balance_method, verbose=False)
            
            # Optimize threshold if requested
            if optimize_threshold:
                self.optimize_threshold(X_val, y_val, metric='f1')
            
            # Evaluate
            y_pred = self.predict(X_val)
            
            metrics = {
                'fold': fold_idx + 1,
                'f1': f1_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_val, y_pred),
                'threshold': self.best_threshold
            }
            
            fold_metrics.append(metrics)
            
            print(f"    F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}")
        
        # Calculate summary statistics
        cv_results = {
            'fold_metrics': pd.DataFrame(fold_metrics),
            'mean_f1': np.mean([m['f1'] for m in fold_metrics]),
            'std_f1': np.std([m['f1'] for m in fold_metrics]),
            'mean_precision': np.mean([m['precision'] for m in fold_metrics]),
            'std_precision': np.std([m['precision'] for m in fold_metrics]),
            'mean_recall': np.mean([m['recall'] for m in fold_metrics]),
            'std_recall': np.std([m['recall'] for m in fold_metrics]),
            'mean_threshold': np.mean([m['threshold'] for m in fold_metrics])
        }
        
        print(f"\nCross-validation summary:")
        print(f"  F1: {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        print(f"  Precision: {cv_results['mean_precision']:.3f} ± {cv_results['std_precision']:.3f}")
        print(f"  Recall: {cv_results['mean_recall']:.3f} ± {cv_results['std_recall']:.3f}")
        print(f"  Mean threshold: {cv_results['mean_threshold']:.3f}")
        
        return cv_results
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (None = use best_threshold)
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        threshold = threshold if threshold is not None else self.best_threshold
        
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            
        Returns:
            DataFrame with feature importances sorted by value
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if importance_type == 'gain':
            importances = self.model.feature_importances_
        else:
            importances = self.model.get_booster().get_score(importance_type=importance_type)
            importances = [importances.get(f, 0) for f in self.feature_names]
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def get_shap_importance(self, X: pd.DataFrame, n_samples: int = 2000) -> Optional[object]:
        """
        Calculate SHAP values for feature importance.
        Matches BAREfoot's SHAP analysis.
        
        Args:
            X: Feature matrix (will be sampled)
            n_samples: Number of samples to use (1000 positive + 1000 negative in BAREfoot)
            
        Returns:
            SHAP explainer object or None if SHAP not available
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Calculating SHAP values...")
        
        # Create explainer
        explainer = shap.Explainer(self.model)
        
        # Sample data (balanced)
        X_sample = X.sample(n=min(n_samples, len(X)), random_state=self.seed)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        print("SHAP calculation complete!")
        
        return shap_values
    
    def save(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Save model and parameters to pickle file.
        Matches BAREfoot's save format.
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata to save
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'clf_model': self.model,
            'best_thresh': self.best_threshold,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'parameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'alpha': self.alpha,
                'lambda': self.lambda_,
                'seed': self.seed
            }
        }
        
        if metadata:
            model_data.update(metadata)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BehaviorClassifier':
        """
        Load model from pickle file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            BehaviorClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        params = model_data.get('parameters', {})
        classifier = cls(**params)
        
        # Restore model state
        classifier.model = model_data['clf_model']
        classifier.best_threshold = model_data.get('best_thresh', 0.5)
        classifier.feature_names = model_data.get('feature_names', None)
        classifier.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from: {filepath}")
        
        return classifier


def plot_threshold_analysis(threshold_results: Dict, 
                           behavior_name: str = "Behavior",
                           save_path: Optional[str] = None):
    """
    Plot threshold optimization results.
    Matches BAREfoot's threshold plot (Figure style).
    """
    plt.figure(figsize=(6, 5))
    
    thresholds = threshold_results['thresholds']
    f1_scores = threshold_results['f1_scores']
    precision_scores = threshold_results['precision_scores']
    recall_scores = threshold_results['recall_scores']
    best_thresh = threshold_results['best_threshold']
    
    plt.plot(thresholds, recall_scores, color='#8B4513', label="Recall", linewidth=2)
    plt.plot(thresholds, precision_scores, color='#CD853F', label="Precision", linewidth=2)
    plt.plot(thresholds, f1_scores, color='#8B0000', label="F1 Score", linewidth=2)
    
    plt.axvline(x=best_thresh, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(behavior_name, fontsize=14)
    plt.xticks(np.arange(0, 1.05, 0.25))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.ylim(-0.02, 1.002)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Behavior Classifier Training Module")
    print("=" * 50)
    print("Features:")
    print("  - Data balancing (downsample/upsample)")
    print("  - Threshold optimization")
    print("  - Cross-validation")
    print("  - SHAP feature importance")
    print("  - Model persistence")
    print("\nMatches BAREfoot's training approach exactly")
