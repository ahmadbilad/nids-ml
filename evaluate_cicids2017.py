"""Evaluate all 4 architectures on the modern CIC-IDS2017 benchmark dataset."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
CIC_PATH = DATA_DIR / "cicids2017_sample.csv"


def evaluate_cicids2017():
    if not CIC_PATH.is_file():
        raise FileNotFoundError(f"Missing CIC-IDS2017 dataset at {CIC_PATH}. Run download_cicids2017.py first.")

    print(f"Loading CIC-IDS2017 dataset from {CIC_PATH}...")
    df = pd.read_csv(CIC_PATH)
    
    # Target: 0 = BENIGN, 1 = INTRUSION
    raw_labels = df["Label"].astype(str)
    targets = (raw_labels != "BENIGN").astype(np.int8)

    feature_cols = [c for c in df.columns if c != "Label"]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 80/20 Stratified Split
    x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(
        features, targets, raw_labels, test_size=0.20, random_state=42, stratify=targets
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1),
    }

    results = {}
    print("\nBenchmarking classifiers on modern CIC-IDS2017 flows:")
    print("-" * 65)

    for name, clf in classifiers.items():
        clf.fit(x_train_scaled, y_train)
        preds = clf.predict(x_test_scaled)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
        fn = int(cm[1][0])
        fp = int(cm[0][1])

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_negatives": fn,
            "false_positives": fp,
            "confusion_matrix": cm,
        }

        print(f"[{name}] Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f} | FN: {fn}")

    # Breakdown per modern attack type for Random Forest
    rf = classifiers["Random Forest"]
    rf_preds = rf.predict(x_test_scaled)
    test_labels_series = labels_test.reset_index(drop=True)
    
    attack_breakdown = {}
    for attack in ["DDoS", "PortScan", "DoS Hulk", "FTP-Patator"]:
        idxs = np.where(test_labels_series == attack)[0]
        tot = len(idxs)
        det = int(np.sum(rf_preds[idxs] == 1)) if tot > 0 else 0
        attack_breakdown[attack] = {
            "total": tot,
            "detected": det,
            "recall": float(det / tot) if tot > 0 else 0.0,
        }

    output_payload = {
        "dataset_name": "CIC-IDS2017",
        "total_test_samples": len(y_test),
        "models": results,
        "attack_breakdown": attack_breakdown,
    }

    out_file = MODEL_DIR / "cic_ids2017_metrics.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print("-" * 65)
    print(f"Saved modern CIC-IDS2017 evaluation metrics to {out_file}")


if __name__ == "__main__":
    evaluate_cicids2017()