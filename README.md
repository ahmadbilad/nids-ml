# NIDS-ML

A machine-learning network intrusion detection system built with Python and
Flask, using the NSL-KDD dataset.

## Project structure

```text
app/          Flask application, templates, and static assets
data/         Local NSL-KDD dataset files (not committed)
model/        Trained model artifacts (generated locally, not committed)
notebooks/    Exploratory analysis and experiments
tests/        Automated tests
```

## Local setup

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/`. Dataset files and generated
model artifacts are excluded from version control.

## Train the models

After installing the requirements, train and evaluate Logistic Regression,
Decision Tree, and Random Forest classifiers against the official NSL-KDD test
set:

```powershell
python train_model.py
```

The target is binary (`normal` or `intrusion`); the dataset difficulty score is
not used as a feature. Training saves the Random Forest classifier and its
numeric scaler and categorical encoder as `model/model.pkl`,
`model/scaler.pkl`, and `model/encoder.pkl`. Test-split evaluation metrics are
saved to `model/metrics.json` so the metrics page does not require dataset files
at runtime. Unknown categorical feature values in test or prediction data are
handled by the encoder.

## Run

The Flask application loads all three model artifacts during startup and fails
with an actionable error if an artifact is missing. Set `SECRET_KEY` to a
stable, private value when running multiple workers or deploying:

```powershell
$env:SECRET_KEY = "replace-with-a-private-value"
flask --app "app:create_app()" run
```

The application provides `/` for traffic submission, `/predict` for validated
POST/Redirect/GET inference, `/result` for the prediction and confidence, and
`/metrics` and `/about` for model evaluation and project information.

## Deployment

The included `render.yaml` describes a Render web service. It installs
`requirements.txt` and starts the Flask application with Gunicorn.
