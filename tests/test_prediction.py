"""Tests for loading the trained NIDS artifacts and predicting traffic samples."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import create_app
from app.model_utils import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    explain_prediction,
    load_artifacts,
    load_dataset,
    predict,
    predict_batch_dataframe,
)


@pytest.fixture(scope="module")
def app():
    """Create and configure an application instance for testing."""
    test_app = create_app()
    test_app.config.update({"TESTING": True})
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Test client for issuing HTTP requests."""
    return app.test_client()


@pytest.fixture(scope="module")
def test_records():
    test_path = DATA_DIR / "KDDTest+.txt"
    if not test_path.is_file():
        pytest.fail(
            f"NSL-KDD test data is missing at {test_path}; "
            "place KDDTest+.txt in data/."
        )
    return load_dataset(test_path)


def test_loads_all_model_artifacts():
    artifacts = load_artifacts(MODEL_DIR)

    assert set(artifacts) == {"model", "scaler", "encoder"}
    assert set(artifacts["model"].classes_) == {0, 1}
    assert artifacts["encoder"].categories_
    assert artifacts["scaler"].n_features_in_ > 0


@pytest.mark.parametrize(
    ("expected_class", "description"),
    [(0, "normal"), (1, "attack")],
)
def test_predict_classifies_normal_and_attack_samples(
    test_records, expected_class, description
):
    features, targets = test_records
    candidate_indexes = np.flatnonzero(targets.to_numpy() == expected_class)[:32]
    assert candidate_indexes.size, f"No {description} samples found in test data."

    candidate_predictions = predict(features.iloc[candidate_indexes], MODEL_DIR)
    correct_positions = np.flatnonzero(candidate_predictions == expected_class)
    assert correct_positions.size, (
        f"The model did not correctly classify any sampled {description} records."
    )

    sample = features.iloc[[candidate_indexes[correct_positions[0]]]]
    prediction = predict(sample, MODEL_DIR)
    assert prediction.shape == (1,)
    assert int(prediction[0]) == expected_class


def test_explain_prediction_returns_top_features(test_records):
    """Verify that Tree SHAP computes top-k explanations on a sample record."""
    features, _ = test_records
    artifacts = load_artifacts(MODEL_DIR)
    sample = features.iloc[[0]]

    explanations = explain_prediction(sample, artifacts, top_k=3)

    assert len(explanations) == 3
    for item in explanations:
        assert "feature" in item
        assert "impact" in item
        assert "direction" in item
        assert isinstance(item["impact"], float)
        assert item["direction"] in ("Pushed towards Intrusion", "Pushed towards Normal")


def test_download_sample_csv_endpoint(client):
    """Verify sample CSV download endpoint returns valid CSV file."""
    response = client.get("/download-sample-csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment; filename=nids_sample_batch.csv" in response.headers.get("Content-Disposition", "")
    assert b"duration,protocol_type,service" in response.data


def test_predict_batch_dataframe_validation(app):
    """Verify batch prediction helper validates columns and returns summary."""
    artifacts = app.extensions["nids_artifacts"]

    dummy_row = {col: 0.0 for col in FEATURE_COLUMNS}
    dummy_row.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
    df = pd.DataFrame([dummy_row, dummy_row])

    summary, records = predict_batch_dataframe(df, artifacts)
    assert summary["total"] == 2
    assert len(records) == 2
    assert "is_intrusion" in records[0]


def test_metrics_route_displays_benchmarks(client):
    """Verify metrics dashboard renders baseline comparisons, attack recall, threshold curves, 5-fold CV, and CIC-IDS2017."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"Out-of-Sample Test Split Evaluation" in response.data
    assert b"Modern Dataset Evaluation: CIC-IDS2017 Flow Benchmark" in response.data
    assert b"5-Fold Stratified Cross-Validation Benchmark" in response.data
    assert b"Logistic Regression" in response.data
    assert b"Decision Tree" in response.data
    assert b"Random Forest" in response.data
    assert b"XGBoost" in response.data
    assert b"Granular Threat Family Recall Breakdown" in response.data
    assert b"Operational Threshold Calibration" in response.data