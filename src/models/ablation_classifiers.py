"""Ablation ML classifiers for feature-based fault classification.

This module provides traditional ML classifiers (Random Forest, XGBoost)
that use features extracted from SparkNet for comparison studies.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class AblationClassifier:
    """Base class for ablation classifiers using SparkNet features."""

    FAULT_CLASSES = [
        "Clean",
        "Dusty",
        "Bird-drop",
        "Electrical-damage",
        "Physical-damage",
        "Snow-Covered",
    ]

    def __init__(self, name: str = "base"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Train the classifier on extracted features.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            validation_data: Optional tuple of (val_features, val_labels)

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels for features.

        Args:
            features: Feature array of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for features.

        Args:
            features: Feature array of shape (n_samples, n_features)

        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)

    def evaluate(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """Evaluate classifier performance.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: True labels of shape (n_samples,)

        Returns:
            Dictionary containing accuracy, precision, recall, f1, confusion matrix
        """
        predictions = self.predict(features)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
            "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions),
            "classification_report": classification_report(
                labels, predictions, target_names=self.FAULT_CLASSES, zero_division=0
            ),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save classifier to disk.

        Args:
            path: Path to save the classifier
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load classifier from disk.

        Args:
            path: Path to load the classifier from
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.name = state["name"]
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]


class RandomForestAblation(AblationClassifier):
    """Random Forest classifier for SparkNet feature comparison."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        """Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed for reproducibility
        """
        super().__init__(name="RandomForest")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestAblation")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Train Random Forest on extracted features."""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit model
        self.model.fit(features_scaled, labels)
        self.is_fitted = True

        # Calculate training metrics
        train_predictions = self.model.predict(features_scaled)
        metrics = {
            "train_accuracy": accuracy_score(labels, train_predictions),
            "train_f1_macro": f1_score(labels, train_predictions, average="macro", zero_division=0),
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, features_scaled, labels, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

        # Validation metrics if provided
        if validation_data is not None:
            val_features, val_labels = validation_data
            val_features_scaled = self.scaler.transform(val_features)
            val_predictions = self.model.predict(val_features_scaled)
            metrics["val_accuracy"] = accuracy_score(val_labels, val_predictions)
            metrics["val_f1_macro"] = f1_score(val_labels, val_predictions, average="macro", zero_division=0)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from the trained model.

        Returns:
            Array of feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self.model.feature_importances_


class XGBoostAblation(AblationClassifier):
    """XGBoost classifier for SparkNet feature comparison."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            random_state: Random seed for reproducibility
        """
        super().__init__(name="XGBoost")

        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostAblation")

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Train XGBoost on extracted features."""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Prepare eval set if validation data provided
        eval_set = None
        if validation_data is not None:
            val_features, val_labels = validation_data
            val_features_scaled = self.scaler.transform(val_features)
            eval_set = [(val_features_scaled, val_labels)]

        # Fit model
        self.model.fit(
            features_scaled,
            labels,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True

        # Calculate training metrics
        train_predictions = self.model.predict(features_scaled)
        metrics = {
            "train_accuracy": accuracy_score(labels, train_predictions),
            "train_f1_macro": f1_score(labels, train_predictions, average="macro", zero_division=0),
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, features_scaled, labels, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

        # Validation metrics if provided
        if validation_data is not None:
            val_features, val_labels = validation_data
            val_features_scaled = self.scaler.transform(val_features)
            val_predictions = self.model.predict(val_features_scaled)
            metrics["val_accuracy"] = accuracy_score(val_labels, val_predictions)
            metrics["val_f1_macro"] = f1_score(val_labels, val_predictions, average="macro", zero_division=0)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from the trained model.

        Returns:
            Array of feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self.model.feature_importances_


class AblationStudy:
    """Run ablation study comparing SparkNet with traditional ML classifiers."""

    def __init__(
        self,
        feature_extractor=None,
        classifiers: Optional[List[str]] = None,
    ):
        """Initialize ablation study.

        Args:
            feature_extractor: SparkNet model for feature extraction (optional)
            classifiers: List of classifier names to use ["rf", "xgb"]
        """
        self.feature_extractor = feature_extractor
        self.classifiers: Dict[str, AblationClassifier] = {}

        classifier_names = classifiers or ["rf", "xgb"]

        for name in classifier_names:
            if name == "rf" and SKLEARN_AVAILABLE:
                self.classifiers["RandomForest"] = RandomForestAblation()
            elif name == "xgb" and XGBOOST_AVAILABLE:
                self.classifiers["XGBoost"] = XGBoostAblation()

    def extract_features(
        self,
        images: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract features from images using SparkNet.

        Args:
            images: Array of images (N, H, W, C) or (N, C, H, W)
            batch_size: Batch size for feature extraction

        Returns:
            Feature array of shape (N, feature_dim)
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not set")

        import torch

        self.feature_extractor.eval()
        features_list = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                # Convert to tensor if needed
                if isinstance(batch, np.ndarray):
                    # Assume (N, H, W, C) format, convert to (N, C, H, W)
                    if batch.ndim == 4 and batch.shape[-1] == 3:
                        batch = np.transpose(batch, (0, 3, 1, 2))
                    batch = torch.from_numpy(batch).float() / 255.0

                # Extract features
                batch_features = self.feature_extractor.get_features(batch)
                features_list.append(batch_features.cpu().numpy())

        return np.concatenate(features_list, axis=0)

    def run_study(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray, str]]]:
        """Run ablation study with all classifiers.

        Args:
            train_features: Training feature array
            train_labels: Training labels
            test_features: Test feature array
            test_labels: Test labels
            validation_data: Optional validation data tuple

        Returns:
            Dictionary mapping classifier names to their evaluation results
        """
        results = {}

        for name, classifier in self.classifiers.items():
            print(f"\nTraining {name}...")

            # Train
            train_metrics = classifier.fit(
                train_features,
                train_labels,
                validation_data=validation_data,
            )

            # Evaluate on test set
            eval_metrics = classifier.evaluate(test_features, test_labels)

            # Combine metrics
            results[name] = {
                **train_metrics,
                **eval_metrics,
            }

            print(f"{name} Test Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"{name} Test F1 (macro): {eval_metrics['f1_macro']:.4f}")

        return results

    def compare_results(
        self,
        results: Dict[str, Dict],
        sparknet_accuracy: Optional[float] = None,
        sparknet_f1: Optional[float] = None,
    ) -> str:
        """Generate comparison report.

        Args:
            results: Results dictionary from run_study
            sparknet_accuracy: SparkNet test accuracy for comparison
            sparknet_f1: SparkNet test F1 score for comparison

        Returns:
            Formatted comparison report string
        """
        report = ["=" * 60]
        report.append("ABLATION STUDY RESULTS")
        report.append("=" * 60)

        # Header
        report.append(f"\n{'Classifier':<20} {'Accuracy':>12} {'F1 (macro)':>12} {'F1 (weighted)':>14}")
        report.append("-" * 60)

        # SparkNet baseline if provided
        if sparknet_accuracy is not None:
            f1_str = f"{sparknet_f1:.4f}" if sparknet_f1 else "N/A"
            report.append(f"{'SparkNet (baseline)':<20} {sparknet_accuracy:>12.4f} {f1_str:>12}")

        # Other classifiers
        for name, metrics in results.items():
            report.append(
                f"{name:<20} {metrics['accuracy']:>12.4f} "
                f"{metrics['f1_macro']:>12.4f} {metrics['f1_weighted']:>14.4f}"
            )

        report.append("-" * 60)

        # Feature importance section
        report.append("\nFEATURE IMPORTANCE (Top 10):")
        report.append("-" * 40)

        for name, classifier in self.classifiers.items():
            if hasattr(classifier, "get_feature_importance") and classifier.is_fitted:
                importance = classifier.get_feature_importance()
                top_indices = np.argsort(importance)[-10:][::-1]
                report.append(f"\n{name}:")
                for idx in top_indices:
                    report.append(f"  Feature {idx}: {importance[idx]:.4f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_results(
        self,
        results: Dict[str, Dict],
        output_path: Union[str, Path],
    ) -> None:
        """Save study results to disk.

        Args:
            results: Results dictionary from run_study
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, metrics in results.items():
            serializable_results[name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[name][key] = value.tolist()
                else:
                    serializable_results[name][key] = value

        import json
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
