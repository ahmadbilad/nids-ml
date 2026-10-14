"""Dataset preparation, model training, and inference utilities."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]
TARGET_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"
DATASET_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN, DIFFICULTY_COLUMN]
CLASS_NAMES = {0: "normal", 1: "intrusion"}


def load_dataset(path):
    """Load an NSL-KDD file and return its 41 features and binary target."""
    dataset = pd.read_csv(
        path,
        header=None,
        names=DATASET_COLUMNS,
        skipinitialspace=True,
        low_memory=False,
    )
    if dataset.shape[1] != len(DATASET_COLUMNS):
        raise ValueError(
            f"{path} must contain {len(DATASET_COLUMNS)} columns; "
            f"found {dataset.shape[1]}."
        )
    if dataset.empty:
        raise ValueError(f"{path} contains no dataset rows.")

    labels = dataset[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")
    if labels.eq("").any():
        raise ValueError(f"{path} contains empty target labels.")

    features = dataset[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype(str).str.strip()

    targets = (labels != "normal").astype(np.int8)
    return features, targets


def transform_features(features, scaler, encoder):
    """Encode categorical features and scale numeric features consistently."""
    missing = set(FEATURE_COLUMNS).difference(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    categorical = features[CATEGORICAL_COLUMNS].astype(str)
    numeric = features[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    encoded = encoder.transform(categorical)
    scaled = scaler.transform(numeric)
    return hstack((encoded, scaled), format="csr", dtype=np.float64)


def _classification_metrics(expected, predicted):
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, zero_division=0)),
        "recall": float(recall_score(expected, predicted, zero_division=0)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def load_artifacts(model_dir=MODEL_DIR):
    """Load the trained estimator and preprocessing artifacts from disk."""
    model_dir = Path(model_dir)
    artifacts = {}
    for name in ("model", "scaler", "encoder"):
        path = model_dir / f"{name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                "Train the model with `python train_model.py`."
            )
        artifacts[name] = joblib.load(path)

    if not callable(getattr(artifacts["model"], "predict_proba", None)):
        raise TypeError("The saved classifier must support probability predictions.")
    return artifacts


def predict_with_artifacts(features, artifacts):
    """Predict binary classes and per-row confidence from loaded artifacts."""
    transformed = transform_features(
        features, artifacts["scaler"], artifacts["encoder"]
    )
    model = artifacts["model"]
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    class_indexes = {
        class_label: index for index, class_label in enumerate(model.classes_)
    }
    confidence = np.array(
        [
            probabilities[row_index, class_indexes[prediction]]
            for row_index, prediction in enumerate(predictions)
        ]
    )
    return predictions, confidence


def explain_prediction(features, artifacts, top_k=3):
    """Compute SHAP feature importance for a single prediction row."""
    transformed = transform_features(
        features, artifacts["scaler"], artifacts["encoder"]
    )
    model = artifacts["model"]
    encoder = artifacts["encoder"]

    # Match exact feature column ordering: encoded categoricals followed by numeric features
    encoded_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    all_feature_names = encoded_feature_names + NUMERIC_COLUMNS

    explainer = shap.TreeExplainer(model)
    # Convert sparse matrix to dense array for TreeExplainer compatibility
    shap_values = explainer.shap_values(transformed.toarray())

    # Extract Class 1 (intrusion) contributions
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        instance_shap = shap_values[0, :, 1]
    else:
        instance_shap = shap_values[0]

    # Rank features by contribution magnitude
    top_indices = np.argsort(np.abs(instance_shap))[::-1][:top_k]

    explanations = []
    for idx in top_indices:
        impact_val = float(instance_shap[idx])
        explanations.append({
            "feature": all_feature_names[idx].replace("_", " ").title(),
            "impact": round(impact_val, 4),
            "direction": (
                "Pushed towards Intrusion"
                if impact_val > 0
                else "Pushed towards Normal"
            ),
        })

    return explanations


def evaluate_saved_model(test_path=DATA_DIR / "KDDTest+.txt", model_dir=MODEL_DIR):
    """Evaluate the saved classifier and return metrics for the supplied split."""
    features, targets = load_dataset(test_path)
    artifacts = load_artifacts(model_dir)
    predictions, _ = predict_with_artifacts(features, artifacts)
    return _classification_metrics(targets, predictions)


def train_models(
    train_path=DATA_DIR / "KDDTrain+.txt",
    test_path=DATA_DIR / "KDDTest+.txt",
    artifact_dir=MODEL_DIR,
):
    """Train baseline classifiers and Random Forest, then save the RF artifacts."""
    train_features, train_targets = load_dataset(train_path)
    test_features, test_targets = load_dataset(test_path)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    scaler = StandardScaler()
    encoder.fit(train_features[CATEGORICAL_COLUMNS])
    scaler.fit(train_features[NUMERIC_COLUMNS])

    encoded_train = transform_features(train_features, scaler, encoder)
    encoded_test = transform_features(test_features, scaler, encoder)

    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }
    metrics = {}
    for name, classifier in classifiers.items():
        classifier.fit(encoded_train, train_targets)
        metrics[name] = _classification_metrics(
            test_targets, classifier.predict(encoded_test)
        )
        print(f"{name}: {metrics[name]}")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    random_forest_metrics = metrics["random_forest"]
    joblib.dump(classifiers["random_forest"], artifact_dir / "model.pkl")
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(encoder, artifact_dir / "encoder.pkl")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(random_forest_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def predict(features, model_dir=MODEL_DIR):
    """Return the saved model's binary predictions for feature rows."""
    artifacts = load_artifacts(model_dir)
    predictions, _ = predict_with_artifacts(features, artifacts)
    return predictions