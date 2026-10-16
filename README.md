# Explainable Network Intrusion Detection System (X-NIDS)

![CI - Pytest](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)
![Framework](https://img.shields.io/badge/framework-Flask-black.svg)
![Model](https://img.shields.io/badge/model-Random%20Forest%20(Ensemble)-orange.svg)
![Explainability](https://img.shields.io/badge/XAI-Tree%20SHAP-purple.svg)
[![Deployment](https://img.shields.io/badge/deployment-Railway%20(Production)-success.svg)](https://nids-ml-production.up.railway.app/)

An end-to-end Machine Learning-powered Network Intrusion Detection System (NIDS) trained on the benchmark **NSL-KDD** dataset. The system combines ensemble classification with **Explainable AI (Tree SHAP)** to eliminate black-box opacity, providing local feature attribution alongside scalable batch log ingestion.

**Live Production Deployment:** [https://nids-ml-production.up.railway.app/](https://nids-ml-production.up.railway.app/)

---

## 1. System Architecture

The pipeline processes network connection telemetry through ingestion, serialization-safe encoding, model inference, and local explainability:

```text
[ Ingestion Interface ]
       │
       ├── Single Flow Record Form (POST /predict)
       └── Multi-Flow CSV Batch File (POST /predict-batch)
               │
               ▼
[ Preprocessing & Transformation Pipeline ]
       │
       ├── One-Hot Encoding (Categorical: protocol_type, service, flag)
       ├── StandardScaler (Continuous/Rate Flow Features)
       └── Feature Matrix Alignment (41 NSL-KDD Dimensions)
               │
               ▼
[ Inference & Explainability Engine ]
       │
       ├── Random Forest Classifier (200 Estimators, Balanced Class Weights)
       │       └── Output: Class Verdict + Prediction Confidence (%)
       │
       └── Tree SHAP (Shapley Additive exPlanations)
               └── Output: Top-3 Local Drivers (Intrusion vs. Normal Directionality)
               │
               ▼
[ Presentation & Diagnostic Layer ]
       ├── Single-Record XAI Rationale Modal
       ├── Multi-Flow Ingestion Summary Table & Threat Metrics
       └── Empirical Baseline Benchmark Dashboard (/metrics)
```

---

## 2. Dataset & Feature Engineering

The system evaluates connection records across the 41 features defined in the **NSL-KDD** benchmark dataset:

- **Categorical Dimensions (3):** `protocol_type` (TCP, UDP, ICMP), `service` (HTTP, SMTP, Private, etc.), and connection status `flag` (SF, S0, REJ, etc.).
- **Numeric Traffic Features (38):**
  - *Basic Connection Features:* `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`.
  - *Content / Host Features:* `num_failed_logins`, `logged_in`, `root_shell`, `su_attempted`, `num_compromised`.
  - *Time-based Traffic Features:* `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`.
  - *Host-based Traffic Features:* Destination host connection density, SYN/REJ error rates, and port distribution ratios (`dst_host_count`, `dst_host_srv_count`, etc.).

All numerical features are standardized via zero-mean unit-variance scaling (`StandardScaler`), while categorical dimensions are one-hot encoded (`handle_unknown='ignore'`).

---

## 3. Empirical Model Evaluation & Benchmarks

To address class imbalance and minimize operational disruption in network monitoring environments, the ensemble architecture was empirically benchmarked against linear and single-tree baselines on the out-of-sample **KDDTest+** split:

| Model Architecture            | Type                 | Accuracy | Precision (Attack) | Recall | F1-Score | Operational Characteristics                         |
| :----------------------------- | :------------------- | :------: | :-----------------: | :----: | :------: | :--------------------------------------------------- |
| **Logistic Regression**        | Linear Parametric    |  75.5%   |        91.7%         | 62.6%  |  0.744   | High throughput, underfits non-linear boundary        |
| **Decision Tree**               | Non-Linear (Single)  |  80.8%   |        95.9%         | 69.2%  |  0.804   | Susceptible to variance & localized overfitting       |
| **Random Forest (Deployed)**   | Bagged Ensemble      | **77.5%** |      **96.9%**      | **62.5%** | **0.760** | **Suppresses false positives (96.9% precision)**      |

### Production Selection Rationale

In enterprise security operations centers (SOC), **false positives** lead to alert fatigue and unintentional denial of service for benign users. Random Forest achieves **96.9% precision**, yielding only 261 false positives out of 9,711 normal connections on the test benchmark, making it the superior architecture for automated production filtering.

---

## 4. Explainable AI (XAI) Implementation

Traditional ensemble methods operate as opaque black boxes. This system integrates **Tree SHAP** (`shap.TreeExplainer`) to compute exact Shapley feature attributions for every prediction:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

- **Interpretability:** Identifies the top network parameters driving an anomaly (e.g., elevated `dst_host_serror_rate`, zero `dst_bytes`, unusual `service`).
- **Directionality:** Explicitly indicates whether a feature pushed the decision boundary toward an **Intrusion** (positive SHAP attribution) or **Normal** behavior (negative SHAP attribution).

---

## 5. Key Features

- **Single-Record Threat Diagnosis:** Interactive form providing immediate verdict, confidence score, and top-3 SHAP attributions.
- **Automated Batch Flow Ingestion:** Upload multi-row CSV logs (up to 500 connections) for bulk screening, returning overall threat concentration rates and row-by-row diagnostic records.
- **One-Click Benchmark Testing:** Built-in endpoint (`/download-sample-csv`) providing instant test flow generation for external reviewers.
- **Empirical Analytics Dashboard:** Visual confusion matrix powered by Chart.js and multi-model comparative performance tables.

---

## 6. Project Structure

```text
nids-ml/
├── app/
│   ├── __init__.py           # Application factory & artifact injection
│   ├── model_utils.py        # Pipeline: preprocessing, SHAP XAI, batch logic
│   ├── routes.py             # UI controllers, endpoints, and file handlers
│   └── templates/            # Jinja2 views (index, result, batch_result, metrics)
├── data/                     # Benchmark dataset splits (KDDTrain+, KDDTest+)
├── model/                    # Serialized production artifacts (model, scaler, encoder)
├── notebooks/                # Exploratory analysis & experiments
├── tests/                    # Pytest automated test suite (7 passing tests)
├── Procfile                  # Railway deployment process command
├── requirements.txt          # Production dependencies
└── train_model.py            # Offline training & benchmarking script
```

---

## 7. Local Installation & Development

### Prerequisites

- Python 3.10+ (tested through Python 3.14)
- Git

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahmadbilad/nids-ml.git
   cd nids-ml
   ```

2. **Initialize and activate a virtual environment:**

   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install project dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the automated test suite:**

   ```bash
   python -m pytest
   ```

5. **Launch the development server:**

   ```bash
   python -m flask --app "app:create_app()" run --reload
   ```

   Then navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 8. Production Deployment & CI/CD

- **Hosting Platform:** Railway
- **Build Engine:** Railway Railpack (Python 3.13 runtime)
- **WSGI Application Server:** Gunicorn 23.0.0 (`app:create_app()`)
- **Continuous Deployment:** Automated CI/CD pipeline triggered on pushes to the `main` branch
