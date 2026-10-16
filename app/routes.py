"""HTTP routes for the NIDS-ML application."""

import io
import json
import math
from pathlib import Path

import pandas as pd
from flask import (
    Blueprint,
    Response,
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
    predict_batch_dataframe,
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


@bp.post("/predict-batch")
def submit_batch():
    """Handle batch CSV file upload for multi-connection classification."""
    if "file" not in request.files:
        flash("Please upload a CSV file.", "danger")
        return redirect(url_for("main.index"))

    file = request.files["file"]
    if not file.filename or not file.filename.endswith(".csv"):
        flash("Invalid file format. Please upload a .csv file.", "danger")
        return redirect(url_for("main.index"))

    try:
        content = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        artifacts = current_app.extensions["nids_artifacts"]
        summary, records = predict_batch_dataframe(df, artifacts)
    except Exception as exc:
        flash(f"Error processing CSV: {str(exc)}", "danger")
        return redirect(url_for("main.index"))

    session["batch_result"] = {
        "summary": summary,
        "records": records,
    }
    return redirect(url_for("main.batch_result"))


@bp.get("/batch-result")
def batch_result():
    """Render the results of the batch CSV analysis."""
    batch_data = session.pop("batch_result", None)
    if not batch_data:
        flash("Upload a CSV file to inspect batch results.", "info")
        return redirect(url_for("main.index"))
    return render_template(
        "batch_result.html",
        summary=batch_data["summary"],
        records=batch_data["records"],
    )


@bp.get("/download-sample-csv")
def download_sample_csv():
    """Generate and return a sample CSV for reviewers and users to test batch mode."""
    # Creates 3 test connection flows (Normal HTTP, DoS SYN Flood, Normal SMTP)
    sample_data = [
        {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 215, "dst_bytes": 4507, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 30,
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.03,
            "dst_host_srv_diff_host_rate": 0.04, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        },
        {
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 250, "srv_count": 15, "serror_rate": 1.0, "srv_serror_rate": 1.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 0.06,
            "diff_srv_rate": 0.07, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
            "dst_host_srv_count": 15, "dst_host_same_srv_rate": 0.06,
            "dst_host_diff_srv_rate": 0.07, "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
            "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        },
        {
            "duration": 0, "protocol_type": "tcp", "service": "smtp", "flag": "SF",
            "src_bytes": 744, "dst_bytes": 328, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
            "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
            "count": 2, "srv_count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 12,
            "dst_host_srv_count": 210, "dst_host_same_srv_rate": 0.92,
            "dst_host_diff_srv_rate": 0.08, "dst_host_same_src_port_rate": 0.08,
            "dst_host_srv_diff_host_rate": 0.03, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        },
    ]
    df = pd.DataFrame(sample_data, columns=FEATURE_COLUMNS)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=nids_sample_batch.csv"},
    )


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
    """Render test-split accuracy, F1, baseline table, and confusion matrix."""
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
        chart_data=chart_data,
        error=None,
    )


@bp.get("/about")
def about():
    """Describe the project, dataset, and model architecture."""
    return render_template("about.html")