"""Faithfulness evaluation (Deletion/Insertion curves) comparing SHAP and LIME on NSL-KDD."""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
import shap
from lime.lime_tabular import LimeTabularExplainer

from app.model_utils import (
    CATEGORICAL_COLUMNS,
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    NUMERIC_COLUMNS,
    load_artifacts,
    load_dataset,
    transform_features,
)

warnings.filterwarnings("ignore")

# Compatible trapezoid integration across NumPy 1.x and 2.x
trapezoid_func = getattr(np, "trapezoid", getattr(np, "trapz", None))


def run_xai_benchmark(n_samples=25, steps=10):
    """Benchmark Tree SHAP against LIME Tabular using Deletion and Insertion curves."""
    print("=" * 70)
    print("1. Loading artifacts and test data...")
    artifacts = load_artifacts(MODEL_DIR)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]

    features, targets = load_dataset(DATA_DIR / "KDDTest+.txt")

    # Sample attack instances that the model confidently identifies
    transformed_all = transform_features(features, scaler, encoder)
    probas = model.predict_proba(transformed_all)[:, 1]

    # Select confident attack instances for explanation evaluation
    attack_indices = np.where((targets.to_numpy() == 1) & (probas >= 0.70))[0]
    selected_indices = attack_indices[:n_samples]

    sample_features = features.iloc[selected_indices].copy().reset_index(drop=True)
    sample_encoded = transform_features(sample_features, scaler, encoder).toarray()

    # Precompute feature names and baseline (median values in transformed feature space)
    encoded_cat_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    all_feature_names = encoded_cat_names + NUMERIC_COLUMNS
    baseline_values = np.median(sample_encoded, axis=0)
    total_features = sample_encoded.shape[1]

    print(f"2. Initializing explainers across {total_features} encoded features...")
    # Tree SHAP Explainer
    tree_explainer = shap.TreeExplainer(model)

    # LIME Tabular Explainer
    lime_explainer = LimeTabularExplainer(
        training_data=sample_encoded,
        feature_names=all_feature_names,
        class_names=["Normal", "Intrusion"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    print("3. Generating explanations and computing Deletion/Insertion curves...")
    shap_deletion_curves = []
    shap_insertion_curves = []
    lime_deletion_curves = []
    lime_insertion_curves = []

    k_steps = np.linspace(1, min(20, total_features), steps, dtype=int)

    for i in range(len(sample_features)):
        instance_transformed = sample_encoded[i : i + 1]
        orig_prob = model.predict_proba(instance_transformed)[0, 1]

        # --- A. Tree SHAP Rankings ---
        shap_vals = tree_explainer.shap_values(instance_transformed)
        if isinstance(shap_vals, list):
            instance_shap = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            instance_shap = shap_vals[0, :, 1]
        else:
            instance_shap = shap_vals[0]

        shap_ranked_indices = np.argsort(-instance_shap)

        # --- B. LIME Rankings ---
        def predict_fn(x):
            return model.predict_proba(x)

        lime_exp = lime_explainer.explain_instance(
            instance_transformed[0],
            predict_fn,
            num_features=total_features,
            labels=[1],
        )
        lime_weights = np.zeros(total_features)
        for feat_idx, weight in lime_exp.local_exp[1]:
            lime_weights[feat_idx] = weight
        lime_ranked_indices = np.argsort(-lime_weights)

        # --- Deletion Curves ---
        shap_del = [orig_prob]
        lime_del = [orig_prob]

        for k in k_steps:
            x_del_shap = instance_transformed.copy()
            x_del_shap[0, shap_ranked_indices[:k]] = baseline_values[shap_ranked_indices[:k]]
            shap_del.append(float(model.predict_proba(x_del_shap)[0, 1]))

            x_del_lime = instance_transformed.copy()
            x_del_lime[0, lime_ranked_indices[:k]] = baseline_values[lime_ranked_indices[:k]]
            lime_del.append(float(model.predict_proba(x_del_lime)[0, 1]))

        shap_deletion_curves.append(shap_del)
        lime_deletion_curves.append(lime_del)

        # --- Insertion Curves ---
        base_prob = float(model.predict_proba(baseline_values.reshape(1, -1))[0, 1])
        shap_ins = [base_prob]
        lime_ins = [base_prob]

        for k in k_steps:
            x_ins_shap = baseline_values.copy().reshape(1, -1)
            x_ins_shap[0, shap_ranked_indices[:k]] = instance_transformed[0, shap_ranked_indices[:k]]
            shap_ins.append(float(model.predict_proba(x_ins_shap)[0, 1]))

            x_ins_lime = baseline_values.copy().reshape(1, -1)
            x_ins_lime[0, lime_ranked_indices[:k]] = instance_transformed[0, lime_ranked_indices[:k]]
            lime_ins.append(float(model.predict_proba(x_ins_lime)[0, 1]))

        shap_insertion_curves.append(shap_ins)
        lime_insertion_curves.append(lime_ins)

    steps_labels = [0] + list(k_steps)
    mean_shap_del = np.mean(shap_deletion_curves, axis=0)
    mean_lime_del = np.mean(lime_deletion_curves, axis=0)
    mean_shap_ins = np.mean(shap_insertion_curves, axis=0)
    mean_lime_ins = np.mean(lime_insertion_curves, axis=0)

    # Normalized AUC
    auc_shap_del = float(trapezoid_func(mean_shap_del) / len(steps_labels))
    auc_lime_del = float(trapezoid_func(mean_lime_del) / len(steps_labels))
    auc_shap_ins = float(trapezoid_func(mean_shap_ins) / len(steps_labels))
    auc_lime_ins = float(trapezoid_func(mean_lime_ins) / len(steps_labels))

    print("\n" + "=" * 70)
    print("XAI FAITHFULNESS BENCHMARK RESULTS:")
    print("-" * 70)
    print("Deletion AUC (Lower = Better Faithfulness):")
    print(f"  Tree SHAP: {auc_shap_del:.4f}")
    print(f"  LIME:      {auc_lime_del:.4f}")
    print("Insertion AUC (Higher = Faster Signal Recovery):")
    print(f"  Tree SHAP: {auc_shap_ins:.4f}")
    print(f"  LIME:      {auc_lime_ins:.4f}")
    print("-" * 70)

    payload = {
        "steps": [int(s) for s in steps_labels],
        "shap": {
            "deletion_curve": [round(float(v), 4) for v in mean_shap_del],
            "insertion_curve": [round(float(v), 4) for v in mean_shap_ins],
            "deletion_auc": round(auc_shap_del, 4),
            "insertion_auc": round(auc_shap_ins, 4),
        },
        "lime": {
            "deletion_curve": [round(float(v), 4) for v in mean_lime_del],
            "insertion_curve": [round(float(v), 4) for v in mean_lime_ins],
            "deletion_auc": round(auc_lime_del, 4),
            "insertion_auc": round(auc_lime_ins, 4),
        },
        "verdict": (
            "Tree SHAP demonstrates higher faithfulness with lower Deletion AUC"
            if auc_shap_del <= auc_lime_del
            else "LIME demonstrates higher sensitivity on this sample split."
        ),
    }

    out_file = MODEL_DIR / "xai_metrics.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved explainability benchmark metrics to: {out_file}")


if __name__ == "__main__":
    run_xai_benchmark()