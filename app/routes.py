"""HTTP routes for the NIDS-ML application."""

import json
import math
from pathlib import Path

import pandas as pd
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app.model_utils import (
    CATEGORICAL_COLUMNS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    evaluate_saved_model,
    explain_prediction,
    predict_with_artifacts,
)

bp = Blueprint("main", __name__)
FEATURE_LABELS = {
    column: column.replace("_", " ").title() for column in FEATURE_COLUMNS
}


@bp.get("/")
def index():
    """Render the landing page and network traffic prediction form."""
    encoder = current_app.extensions["nids_artifacts"]["encoder"]
    categories = {
        column: encoder.categories_[index].tolist()
        for index, column in enumerate(CATEGORICAL_COLUMNS)
    }
    return render_template(
        "index.html",
        features=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        categories=categories,
        labels=FEATURE_LABELS,
    )


@bp.post("/predict")
def submit_prediction():
    """Validate traffic features and redirect using POST/Redirect/GET."""
    values = {}
    errors = []
    for column in FEATURE_COLUMNS:
        raw_value = request.form.get(column, "").strip()
        if not raw_value:
            errors.append(f"{FEATURE_LABELS[column]} is required.")
            continue
        if column in NUMERIC_COLUMNS:
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"{FEATURE_LABELS[column]} must be a number.")
                continue
            if not math.isfinite(value):
                errors.append(f"{FEATURE_LABELS[column]} must be a finite number.")
                continue
            values[column] = value
        else:
            values[column] = raw_value

    if errors:
        for error in errors:
            flash(error, "danger")
        return redirect(url_for("main.index"))

    features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    artifacts = current_app.extensions["nids_artifacts"]

    predictions, confidences = predict_with_artifacts(features, artifacts)
    explanations = explain_prediction(features, artifacts)

    prediction = int(predictions[0])
    session["prediction_result"] = {
        "class_id": prediction,
        "label": CLASS_NAMES[prediction],
        "confidence": float(confidences[0]),
        "explanations": explanations,
    }
    return redirect(url_for("main.result"))


@bp.get("/result")
def result():
    """Display the latest prediction after a form submission."""
    prediction_result = session.pop("prediction_result", None)
    if prediction_result is None:
        flash("Submit a traffic record to see its prediction.", "info")
        return redirect(url_for("main.index"))
    return render_template("result.html", result=prediction_result)


@bp.get("/metrics")
def metrics():
    """Render test-split accuracy, F1, and the confusion matrix."""
    metrics_path = Path(current_app.config["MODEL_DIR"]) / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        test_path = Path(current_app.config["DATA_DIR"]) / "KDDTest+.txt"
        if not test_path.is_file():
            return render_template(
                "metrics.html",
                report=None,
                chart_data=None,
                error="Evaluation metrics are unavailable. Provide the NSL-KDD "
                "test file or regenerate model/metrics.json during training.",
            ), 503
        report = evaluate_saved_model(test_path, current_app.config["MODEL_DIR"])

    matrix = report.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        return render_template(
            "metrics.html",
            report=None,
            chart_data=None,
            error="The saved confusion matrix has an invalid format.",
        ), 500
    chart_data = {
        "labels": ["Normal", "Intrusion"],
        "matrix": matrix,
    }
    return render_template(
        "metrics.html",
        report=report,
        chart_data=json.dumps(chart_data),
        error=None,
    )


@bp.get("/about")
def about():
    """Describe the project, dataset, and model architecture."""
    return render_template("about.html")