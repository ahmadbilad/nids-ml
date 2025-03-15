"""Tests for loading the trained NIDS artifacts and predicting traffic samples."""

import numpy as np
import pytest

from app.model_utils import (
    DATA_DIR,
    MODEL_DIR,
    load_artifacts,
    load_dataset,
    predict,
)


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
