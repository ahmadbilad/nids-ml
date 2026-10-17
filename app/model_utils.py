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
from xgboost import XGBClassifier

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

ATTACK_MAPPING = {
    # DoS
    "neptune": "DoS",
    "back": "DoS",
    "land": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "httptunnel": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def load_dataset(path, return_attack_types=False):
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

    raw_labels = dataset[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".").str.lower()
    if raw_labels.eq("").any():
        raise ValueError(f"{path} contains empty target labels.")

    features = dataset[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype(str).str.strip()

    targets = (raw_labels != "normal").astype(np.int8)

    if return_attack_types:
        attack_types = raw_labels.map(lambda x: ATTACK_MAPPING.get(x, "Normal" if x == "normal" else "Other"))
        return features, targets, attack_types

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
    tn, fp, fn, tp = int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0]), int(matrix[1][1])
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, zero_division=0)),
        "recall": float(recall_score(expected, predicted, zero_division=0)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def compute_per_attack_recall(targets, predictions, attack_types):
    """Calculate recall metrics partitioned by attack category."""
    classes = ["DoS", "Probe", "R2L", "U2R"]
    breakdown = {}
    targets_np = np.asarray(targets)
    predictions_np = np.asarray(predictions)
    attack_series = pd.Series(attack_types).reset_index(drop=True)

    for cat in classes:
        indices = np.where(attack_series == cat)[0]
        total = int(len(indices))
        if total > 0:
            detected = int(np.sum(predictions_np[indices] == 1))
            recall = float(detected / total)
        else:
            detected = 0
            recall = 0.0
        breakdown[cat] = {
            "total": total,
            "detected": detected,
            "missed": total - detected,
            "recall": recall,
        }
    return breakdown


def compute_threshold_curve(targets, probas, step=0.05):
    """Evaluate precision, recall, and false negative counts across threshold cuts."""
    targets_np = np.asarray(targets)
    thresholds = np.arange(0.1, 0.95, step)
    curve_data = []

    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets_np, preds, labels=[0, 1]).ravel()
        p = precision_score(targets_np, preds, zero_division=0)
        r = recall_score(targets_np, preds, zero_division=0)
        f1 = f1_score(targets_np, preds, zero_division=0)
        curve_data.append({
            "threshold": round(float(t), 2),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        })
    return curve_data


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


def predict_with_artifacts(features, artifacts, threshold=0.5):
    """Predict binary classes and per-row confidence from loaded artifacts with adjustable threshold."""
    transformed = transform_features(
        features, artifacts["scaler"], artifacts["encoder"]
    )
    model = artifacts["model"]
    probabilities = model.predict_proba(transformed)
    class_indexes = {
        class_label: index for index, class_label in enumerate(model.classes_)
    }
    intrusion_index = class_indexes[1]
    intrusion_probs = probabilities[:, intrusion_index]
    predictions = (intrusion_probs >= threshold).astype(int)

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

    encoded_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    all_feature_names = encoded_feature_names + NUMERIC_COLUMNS

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed.toarray())

    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        instance_shap = shap_values[0, :, 1]
    else:
        instance_shap = shap_values[0]

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


def predict_batch_dataframe(df, artifacts, max_rows=500, threshold=0.5):
    """Validate, preprocess, and predict a batch DataFrame."""
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    missing_cols = set(FEATURE_COLUMNS).difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {', '.join(sorted(missing_cols))}"
        )

    features = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    for col in CATEGORICAL_COLUMNS:
        features[col] = features[col].astype(str).str.strip()

    predictions, confidences = predict_with_artifacts(features, artifacts, threshold=threshold)

    total_samples = len(predictions)
    intrusions = int(np.sum(predictions == 1))
    normal = int(total_samples - intrusions)

    results = []
    for i in range(total_samples):
        results.append({
            "index": i + 1,
            "prediction": CLASS_NAMES[int(predictions[i])],
            "is_intrusion": bool(predictions[i] == 1),
            "confidence": round(float(confidences[i]) * 100, 1),
            "protocol": str(features.iloc[i]["protocol_type"]),
            "service": str(features.iloc[i]["service"]),
            "flag": str(features.iloc[i]["flag"]),
            "src_bytes": int(features.iloc[i]["src_bytes"]),
            "dst_bytes": int(features.iloc[i]["dst_bytes"]),
        })

    summary = {
        "total": total_samples,
        "intrusions": intrusions,
        "normal": normal,
        "intrusion_rate": (
            round((intrusions / total_samples) * 100, 1)
            if total_samples > 0
            else 0.0
        ),
    }

    return summary, results


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
    """Train baseline classifiers, XGBoost, compute attack-category recall, and save RF artifacts."""
    train_features, train_targets = load_dataset(train_path)
    test_features, test_targets, test_attack_types = load_dataset(
        test_path, return_attack_types=True
    )

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
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics = {}
    for name, classifier in classifiers.items():
        classifier.fit(encoded_train, train_targets)
        metrics[name] = _classification_metrics(
            test_targets, classifier.predict(encoded_test)
        )
        print(f"{name}: {metrics[name]}")

    rf_model = classifiers["random_forest"]
    rf_probas = rf_model.predict_proba(encoded_test)[:, 1]
    rf_preds_default = rf_model.predict(encoded_test)

    per_attack_recall = compute_per_attack_recall(
        test_targets, rf_preds_default, test_attack_types
    )
    threshold_curve = compute_threshold_curve(test_targets, rf_probas, step=0.05)

    tuned_threshold = 0.35
    rf_preds_tuned = (rf_probas >= tuned_threshold).astype(int)
    tuned_metrics = _classification_metrics(test_targets, rf_preds_tuned)
    tuned_attack_recall = compute_per_attack_recall(
        test_targets, rf_preds_tuned, test_attack_types
    )

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    evaluation_payload = {
        "primary_model": "random_forest",
        "models": metrics,
        "accuracy": metrics["random_forest"]["accuracy"],
        "precision": metrics["random_forest"]["precision"],
        "recall": metrics["random_forest"]["recall"],
        "f1": metrics["random_forest"]["f1"],
        "confusion_matrix": metrics["random_forest"]["confusion_matrix"],
        "false_negatives": metrics["random_forest"]["false_negatives"],
        "false_positives": metrics["random_forest"]["false_positives"],
        "per_attack_recall": per_attack_recall,
        "threshold_curve": threshold_curve,
        "tuned_evaluation": {
            "threshold": tuned_threshold,
            "metrics": tuned_metrics,
            "per_attack_recall": tuned_attack_recall,
        },
    }

    joblib.dump(classifiers["random_forest"], artifact_dir / "model.pkl")
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(encoder, artifact_dir / "encoder.pkl")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(evaluation_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def predict(features, model_dir=MODEL_DIR):
    """Return the saved model's binary predictions for feature rows."""
    artifacts = load_artifacts(model_dir)
    predictions, _ = predict_with_artifacts(features, artifacts)
    return predictions