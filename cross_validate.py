"""Run 5-Fold Stratified Cross-Validation on NSL-KDD with fixed seed."""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.model_utils import (
    CATEGORICAL_COLUMNS,
    DATA_DIR,
    MODEL_DIR,
    NUMERIC_COLUMNS,
    load_dataset,
    transform_features,
)


def main():
    print("=" * 65)
    print("1. Loading KDDTrain+ benchmark dataset...")
    train_features, train_targets = load_dataset(DATA_DIR / "KDDTrain+.txt")

    print("2. Fitting preprocessors (OneHotEncoder + StandardScaler)...")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    scaler = StandardScaler()
    encoder.fit(train_features[CATEGORICAL_COLUMNS])
    scaler.fit(train_features[NUMERIC_COLUMNS])

    encoded_train = transform_features(train_features, scaler, encoder)
    print(f"   Transformed feature matrix shape: {encoded_train.shape}")

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    print("\n3. Executing 5-Fold Stratified Cross-Validation (Seed=42)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]

    cv_summary = {}

    print("-" * 65)
    for name, clf in classifiers.items():
        print(f"--> Running CV on {name}...")
        scores = cross_validate(clf, encoded_train, train_targets, cv=cv, scoring=scoring, n_jobs=-1)
        
        acc_m, acc_s = float(np.mean(scores["test_accuracy"])), float(np.std(scores["test_accuracy"]))
        prec_m, prec_s = float(np.mean(scores["test_precision"])), float(np.std(scores["test_precision"]))
        rec_m, rec_s = float(np.mean(scores["test_recall"])), float(np.std(scores["test_recall"]))
        f1_m, f1_s = float(np.mean(scores["test_f1"])), float(np.std(scores["test_f1"]))

        cv_summary[name] = {
            "accuracy": f"{acc_m * 100:.2f}% ± {acc_s * 100:.2f}%",
            "precision": f"{prec_m * 100:.2f}% ± {prec_s * 100:.2f}%",
            "recall": f"{rec_m * 100:.2f}% ± {rec_s * 100:.2f}%",
            "f1": f"{f1_m:.4f} ± {f1_s:.4f}",
            "raw": {
                "acc_mean": acc_m, "acc_std": acc_s,
                "prec_mean": prec_m, "prec_std": prec_s,
                "rec_mean": rec_m, "rec_std": rec_s,
                "f1_mean": f1_m, "f1_std": f1_s,
            }
        }

        print(f"    Accuracy:  {cv_summary[name]['accuracy']}")
        print(f"    Precision: {cv_summary[name]['precision']}")
        print(f"    Recall:    {cv_summary[name]['recall']}")
        print(f"    F1-Score:  {cv_summary[name]['f1']}")
        print("-" * 65)

    # Save to model/cv_metrics.json
    cv_output_path = MODEL_DIR / "cv_metrics.json"
    cv_output_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    print(f"\nSaved cross-validation results to {cv_output_path}")


if __name__ == "__main__":
    main()