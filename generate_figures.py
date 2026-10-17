"""
Generate publication-quality figures for research reporting and repository archival.
Outputs all charts to the results/ directory at 300 DPI.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
})


def plot_confusion_matrix(metrics_data):
    """Plot normalized and absolute confusion matrix for Random Forest."""
    cm = np.array(metrics_data["confusion_matrix"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Intrusion"])
    ax.set_yticklabels(["Normal", "Intrusion"])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
    ax.set_title("NSL-KDD Test Split Confusion Matrix (RF, $\\tau=0.50$)", pad=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_confusion_matrix.png")
    plt.close(fig)
    print(" -> Saved results/fig1_confusion_matrix.png")


def plot_threshold_calibration(metrics_data):
    """Plot Precision-Recall and False Negative curves across decision thresholds."""
    curve = metrics_data.get("threshold_curve", [])
    if not curve:
        return

    thresholds = [x["threshold"] for x in curve]
    precisions = [x["precision"] * 100 for x in curve]
    recalls = [x["recall"] * 100 for x in curve]
    fn_counts = [x["false_negatives"] for x in curve]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))

    ax1.plot(thresholds, precisions, "g-o", markersize=4, label="Precision (%)", linewidth=1.5)
    ax1.plot(thresholds, recalls, "b-s", markersize=4, label="Recall (%)", linewidth=1.5)
    ax1.axvline(x=0.35, color="red", linestyle="--", linewidth=1.5, label="Calibrated Cutoff ($\\tau=0.35$)")
    ax1.axvline(x=0.50, color="gray", linestyle=":", linewidth=1.2, label="Default Cutoff ($\\tau=0.50$)")

    ax1.set_xlabel("Operational Decision Threshold ($\\tau$)", fontweight="bold")
    ax1.set_ylabel("Metric Score (%)", fontweight="bold")
    ax1.set_ylim(45, 102)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Twin axis for Missed Threat Count (False Negatives)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, fn_counts, "r--^", markersize=3, alpha=0.5, label="False Negatives (Count)")
    ax2.set_ylabel("False Negatives (Missed Attacks)", color="darkred", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="darkred")

    # Unified Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)

    plt.title("Security SOC Threshold Calibration Trade-off", fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_threshold_calibration.png")
    plt.close(fig)
    print(" -> Saved results/fig2_threshold_calibration.png")


def plot_xai_faithfulness(xai_data):
    """Plot SHAP vs. LIME Deletion and Insertion Curves."""
    if not xai_data:
        return

    steps = xai_data["steps"]
    shap_del = xai_data["shap"]["deletion_curve"]
    lime_del = xai_data["lime"]["deletion_curve"]
    shap_ins = xai_data["shap"]["insertion_curve"]
    lime_ins = xai_data["lime"]["insertion_curve"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # Deletion Curve (Lower is more faithful)
    ax1.plot(steps, shap_del, "b-o", markersize=4, label=f"Tree SHAP (AUC={xai_data['shap']['deletion_auc']:.3f})", linewidth=1.8)
    ax1.plot(steps, lime_del, "r--s", markersize=4, label=f"LIME (AUC={xai_data['lime']['deletion_auc']:.3f})", linewidth=1.8)
    ax1.set_title("(a) Deletion Curve (Signal Decay)", fontweight="bold")
    ax1.set_xlabel("Top-k Salient Features Perturbed", fontweight="bold")
    ax1.set_ylabel("Intrusion Prediction Probability", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Insertion Curve (Higher is faster signal recovery)
    ax2.plot(steps, shap_ins, "b-o", markersize=4, label=f"Tree SHAP (AUC={xai_data['shap']['insertion_auc']:.3f})", linewidth=1.8)
    ax2.plot(steps, lime_ins, "r--s", markersize=4, label=f"LIME (AUC={xai_data['lime']['insertion_auc']:.3f})", linewidth=1.8)
    ax2.set_title("(b) Insertion Curve (Signal Recovery)", fontweight="bold")
    ax2.set_xlabel("Top-k Salient Features Restored", fontweight="bold")
    ax2.set_ylabel("Intrusion Prediction Probability", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.suptitle("Explainability Faithfulness: Tree SHAP vs. LIME Tabular", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_xai_faithfulness.png")
    plt.close(fig)
    print(" -> Saved results/fig3_xai_faithfulness.png")


def plot_threat_family_breakdown(metrics_data):
    """Plot detection recall across distinct attack classes."""
    breakdown = metrics_data.get("per_attack_recall", {})
    if not breakdown:
        return

    families = list(breakdown.keys())
    recalls = [breakdown[f]["recall"] * 100 for f in families]
    totals = [breakdown[f]["total"] for f in families]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(families, recalls, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], width=0.55, edgecolor="black")

    for bar, total, rec in zip(bars, totals, recalls):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{rec:.1f}%\n(N={total:,})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Detection Recall Rate (%)", fontweight="bold")
    ax.set_xlabel("Attack Vector Taxonomy", fontweight="bold")
    ax.set_title("Granular Attack Family Detection Asymmetry (NSL-KDD Test)", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_attack_recall_asymmetry.png")
    plt.close(fig)
    print(" -> Saved results/fig4_attack_recall_asymmetry.png")


def main():
    print("Generating academic figures from serialized artifacts...")

    metrics_file = MODEL_DIR / "metrics.json"
    if metrics_file.is_file():
        with open(metrics_file, encoding="utf-8") as f:
            metrics_data = json.load(f)
        plot_confusion_matrix(metrics_data)
        plot_threshold_calibration(metrics_data)
        plot_threat_family_breakdown(metrics_data)

    xai_file = MODEL_DIR / "xai_metrics.json"
    if xai_file.is_file():
        with open(xai_file, encoding="utf-8") as f:
            xai_data = json.load(f)
        plot_xai_faithfulness(xai_data)

    print(f"All figures generated successfully in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()