"""Tests for ablation ML classifiers."""

import numpy as np
import pytest

# Skip all tests if sklearn not available
pytest.importorskip("sklearn")

from src.models.ablation_classifiers import (
    AblationClassifier,
    RandomForestAblation,
    AblationStudy,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
)


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 256

    # Generate features
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, 6, n_samples)

    X_val = np.random.randn(n_samples // 5, n_features).astype(np.float32)
    y_val = np.random.randint(0, 6, n_samples // 5)

    X_test = np.random.randn(n_samples // 5, n_features).astype(np.float32)
    y_test = np.random.randint(0, 6, n_samples // 5)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


class TestRandomForestAblation:
    """Tests for Random Forest classifier."""

    def test_rf_initialization(self):
        """Test Random Forest initialization."""
        clf = RandomForestAblation()
        assert clf.name == "RandomForest"
        assert clf.is_fitted is False
        assert clf.model is not None

    def test_rf_fit(self, sample_data):
        """Test Random Forest training."""
        clf = RandomForestAblation(n_estimators=10)
        metrics = clf.fit(
            sample_data["X_train"],
            sample_data["y_train"],
            validation_data=(sample_data["X_val"], sample_data["y_val"]),
        )

        assert clf.is_fitted is True
        assert "train_accuracy" in metrics
        assert "cv_accuracy_mean" in metrics
        assert "val_accuracy" in metrics
        assert 0 <= metrics["train_accuracy"] <= 1

    def test_rf_predict(self, sample_data):
        """Test Random Forest prediction."""
        clf = RandomForestAblation(n_estimators=10)
        clf.fit(sample_data["X_train"], sample_data["y_train"])

        predictions = clf.predict(sample_data["X_test"])

        assert predictions.shape == sample_data["y_test"].shape
        assert all(0 <= p < 6 for p in predictions)

    def test_rf_predict_proba(self, sample_data):
        """Test Random Forest probability prediction."""
        clf = RandomForestAblation(n_estimators=10)
        clf.fit(sample_data["X_train"], sample_data["y_train"])

        proba = clf.predict_proba(sample_data["X_test"])

        assert proba.shape[0] == sample_data["X_test"].shape[0]
        assert proba.shape[1] == 6
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_rf_evaluate(self, sample_data):
        """Test Random Forest evaluation."""
        clf = RandomForestAblation(n_estimators=10)
        clf.fit(sample_data["X_train"], sample_data["y_train"])

        metrics = clf.evaluate(sample_data["X_test"], sample_data["y_test"])

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

    def test_rf_feature_importance(self, sample_data):
        """Test Random Forest feature importance."""
        clf = RandomForestAblation(n_estimators=10)
        clf.fit(sample_data["X_train"], sample_data["y_train"])

        importance = clf.get_feature_importance()

        assert importance.shape[0] == sample_data["X_train"].shape[1]
        assert all(i >= 0 for i in importance)

    def test_rf_save_load(self, sample_data, tmp_path):
        """Test Random Forest save and load."""
        clf = RandomForestAblation(n_estimators=10)
        clf.fit(sample_data["X_train"], sample_data["y_train"])

        original_predictions = clf.predict(sample_data["X_test"])

        # Save
        path = tmp_path / "rf_classifier.pkl"
        clf.save(path)

        # Load
        clf2 = RandomForestAblation()
        clf2.load(path)

        loaded_predictions = clf2.predict(sample_data["X_test"])

        assert np.array_equal(original_predictions, loaded_predictions)

    def test_rf_not_fitted_error(self, sample_data):
        """Test error when predicting without fitting."""
        clf = RandomForestAblation()

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(sample_data["X_test"])


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostAblation:
    """Tests for XGBoost classifier."""

    def test_xgb_initialization(self):
        """Test XGBoost initialization."""
        from src.models.ablation_classifiers import XGBoostAblation

        clf = XGBoostAblation()
        assert clf.name == "XGBoost"
        assert clf.is_fitted is False

    def test_xgb_fit_and_predict(self, sample_data):
        """Test XGBoost training and prediction."""
        from src.models.ablation_classifiers import XGBoostAblation

        clf = XGBoostAblation(n_estimators=10)
        metrics = clf.fit(
            sample_data["X_train"],
            sample_data["y_train"],
            validation_data=(sample_data["X_val"], sample_data["y_val"]),
        )

        assert clf.is_fitted is True
        assert "train_accuracy" in metrics

        predictions = clf.predict(sample_data["X_test"])
        assert predictions.shape == sample_data["y_test"].shape


class TestAblationStudy:
    """Tests for ablation study runner."""

    def test_study_initialization(self):
        """Test ablation study initialization."""
        study = AblationStudy(classifiers=["rf"])

        assert "RandomForest" in study.classifiers
        assert len(study.classifiers) >= 1

    def test_study_run(self, sample_data):
        """Test running ablation study."""
        study = AblationStudy(classifiers=["rf"])

        results = study.run_study(
            train_features=sample_data["X_train"],
            train_labels=sample_data["y_train"],
            test_features=sample_data["X_test"],
            test_labels=sample_data["y_test"],
            validation_data=(sample_data["X_val"], sample_data["y_val"]),
        )

        assert "RandomForest" in results
        assert "accuracy" in results["RandomForest"]

    def test_study_compare_results(self, sample_data):
        """Test comparison report generation."""
        study = AblationStudy(classifiers=["rf"])

        results = study.run_study(
            train_features=sample_data["X_train"],
            train_labels=sample_data["y_train"],
            test_features=sample_data["X_test"],
            test_labels=sample_data["y_test"],
        )

        report = study.compare_results(results, sparknet_accuracy=0.95)

        assert "ABLATION STUDY RESULTS" in report
        assert "RandomForest" in report
        assert "SparkNet (baseline)" in report

    def test_study_save_results(self, sample_data, tmp_path):
        """Test saving study results."""
        study = AblationStudy(classifiers=["rf"])

        results = study.run_study(
            train_features=sample_data["X_train"],
            train_labels=sample_data["y_train"],
            test_features=sample_data["X_test"],
            test_labels=sample_data["y_test"],
        )

        output_path = tmp_path / "results.json"
        study.save_results(results, output_path)

        assert output_path.exists()

        import json
        with open(output_path) as f:
            loaded = json.load(f)

        assert "RandomForest" in loaded


class TestAblationWithSparkNet:
    """Tests for ablation with SparkNet feature extraction."""

    def test_feature_extraction_dimensions(self):
        """Test that SparkNet features have expected dimensions."""
        import torch
        from src.models.sparknet import SparkNet

        model = SparkNet(num_classes=6)
        model.eval()

        x = torch.randn(4, 3, 227, 227)
        with torch.no_grad():
            features = model.get_features(x)

        assert len(features.shape) == 2
        assert features.shape[0] == 4
        # Features should be from GAP layer
        assert features.shape[1] > 100  # Reasonable feature dimension
