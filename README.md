# X-NIDS: Explainable Network Intrusion Detection System

**Empirical Multi-Model Benchmarking, XAI Faithfulness Verification, and Live Flow Ingestion**

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-Flask-black.svg)
![Models](https://img.shields.io/badge/models-LR%20%7C%20DT%20%7C%20RF%20%7C%20XGBoost-orange.svg)
![Explainability](https://img.shields.io/badge/XAI-Tree%20SHAP%20%7C%20LIME-purple.svg)
![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)
[![Deployment](https://img.shields.io/badge/deployment-Railway%20(Production)-success.svg)](https://nids-ml-production.up.railway.app/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](RESEARCH_PAPER.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](RESEARCH_PAPER.pdf)

> **Academic Disclaimer:** This repository contains an **experimental research prototype** designed to study interpretability faithfulness, threshold sensitivity, and feature representation in machine learning-based network intrusion detection. It is **not** an enterprise production intrusion prevention system (IPS) and should not be deployed inline without human analyst verification and robust transport-layer encryption proxies.

**Live Demo:** [https://nids-ml-production.up.railway.app/](https://nids-ml-production.up.railway.app/)
**Research Paper:** [RESEARCH_PAPER.pdf](RESEARCH_PAPER.pdf) (Compiled PDF) | [RESEARCH_PAPER.tex](RESEARCH_PAPER.tex) (IEEE LaTeX Source)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Dataset & Feature Engineering](#4-dataset--feature-engineering)
5. [Empirical Benchmark & Validation Results](#5-empirical-benchmark--validation-results)
6. [Explainable AI (XAI) Implementation](#6-explainable-ai-xai-implementation)
7. [Empirical Visualizations](#7-empirical-visualizations)
8. [Project Structure](#8-project-structure)
9. [Dataset Acquisition](#9-dataset-acquisition)
10. [Local Installation & Reproduction](#10-local-installation--reproduction)
11. [Production Deployment & CI/CD](#11-production-deployment--cicd)
12. [Known Limitations](#12-known-limitations)
13. [Future Work](#13-future-work)
14. [References](#14-references)
15. [License](#15-license)

---

## 1. Abstract

Machine Learning-based Network Intrusion Detection Systems (ML-NIDS) frequently achieve deceptive >99% cross-validation scores while suffering catastrophic out-of-distribution performance degradation when exposed to low-footprint attack vectors or novel environments [1, 2]. Widespread deployment of black-box models further introduces auditability challenges in Security Operations Centers (SOCs).

**X-NIDS** is an end-to-end research framework that tackles both problems directly. It benchmarks four classifiers -- Logistic Regression, Decision Trees, Random Forest, and XGBoost -- across legacy connection-level telemetry (**NSL-KDD** [2]) and modern bidirectional flow statistics (**CIC-IDS2017** [3]), then pairs the deployed model with **dual explainability engines** (Tree SHAP [4, 6] and LIME [5]) whose attribution *faithfulness* is quantitatively validated rather than assumed. The project contributes:

1. **Multi-model, dual-dataset empirical benchmarking** under identical preprocessing and evaluation protocols.
2. **XAI faithfulness verification** via Deletion/Insertion AUC curves [8], proving Tree SHAP is measurably more faithful than LIME on this task ($AUC_{del}=0.7360$ vs. $0.8313$).
3. **SOC-oriented operational threshold calibration** ($\tau = 0.35$), cutting missed intrusions by 861 (-17.9%) for a negligible false-alarm cost.
4. **A live packet ingestion pipeline** that reconstructs bidirectional flows from raw `.pcap` captures via Scapy for real-time inference.

---

## 2. Key Features

- **Multi-Model Comparative Framework** -- Logistic Regression, Decision Tree, Random Forest *(deployed)*, and XGBoost evaluated under identical conditions.
- **Dual Explainable AI (Tree SHAP & LIME)** -- local, per-flow feature attribution explaining *why* an alert was triggered or dismissed.
- **XAI Faithfulness Benchmarking** -- empirical Deletion/Insertion curve testing quantitatively comparing Tree SHAP against LIME Tabular.
- **Raw PCAP Traffic Replay** -- ingests `.pcap` / `.pcapng` packet captures, reconstructs bidirectional 5-tuple IP flows via Scapy, and performs batch inference.
- **Operational Threshold Calibration (tau-Tuning)** -- configurable classification cutoff ($\tau=0.35$) that lowers critical false negatives while containing benign false alarms.
- **Granular Threat Family Profiling** -- recall breakdown across volumetric attacks (DoS, Probe) and low-footprint intrusions (R2L, U2R).
- **Statistical Rigor via 5-Fold Cross-Validation** -- validated on 125,973 training records, reporting Mean +/- Std across all metrics.
- **Dual-Dataset Benchmarking** -- bridges legacy connection telemetry (NSL-KDD) and modern bidirectional flow data (CIC-IDS2017).
- **Interactive SOC Web Dashboard** -- single-flow manual entry, batch CSV inspection, PCAP capture replay, confusion matrices, and interactive Chart.js precision/recall curves.

---

## 3. System Architecture

```text
[ Ingestion Layer ]
       |
       +-- Single Flow Record Form (POST /predict)
       +-- Multi-Flow CSV Batch File (POST /predict-batch)
       `-- Raw PCAP / PCAPNG Replay (Scapy 5-Tuple Flow Reconstruction)
               |
               v
[ Preprocessing & Transformation Pipeline ]
       |
       +-- One-Hot Encoding (Categorical: protocol_type, service, flag)
       +-- StandardScaler (Continuous/Rate Flow Features)
       `-- Feature Matrix Alignment (41 NSL-KDD Dimensions / 34 CIC-IDS2017 Dimensions)
               |
               v
[ Multi-Model Classification & Threshold Calibration ]
       |
       +-- Logistic Regression | Decision Tree | Random Forest (Deployed) | XGBoost
       `-- Operational Threshold Tuning (tau = 0.35) for SOC Recall Optimization
               |
               v
[ Explainability & Faithfulness Engine ]
       |
       +-- Tree SHAP  -- Output: Top-3 Local Drivers + Exact Shapley Attribution
       +-- LIME Tabular -- Output: Local Linear Surrogate Attribution
       `-- Deletion / Insertion AUC -- Quantitative Faithfulness Verification
               |
               v
[ Presentation & Diagnostic Layer ]
       +-- Single-Record XAI Rationale Modal
       +-- Multi-Flow / PCAP Ingestion Summary Table & Threat Metrics
       `-- Empirical Benchmark Dashboard (/metrics)
```

---

## 4. Dataset & Feature Engineering

### NSL-KDD (Primary Benchmark)

The curated NSL-KDD dataset [2] resolves the severe record duplication found in the original KDD'99 corpus (~78% duplicate training records, ~75% duplicate test records). The system evaluates connection records across all 41 defined features:

- **Categorical Dimensions (3):** `protocol_type` (TCP, UDP, ICMP), `service` (HTTP, SMTP, Private, etc.), and connection status `flag` (SF, S0, REJ, etc.).
- **Numeric Traffic Features (38):**
  - *Basic Connection Features:* `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`.
  - *Content / Host Features:* `num_failed_logins`, `logged_in`, `root_shell`, `su_attempted`, `num_compromised`.
  - *Time-based Traffic Features:* `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`.
  - *Host-based Traffic Features:* destination host connection density, SYN/REJ error rates, and port distribution ratios (`dst_host_count`, `dst_host_srv_count`, etc.).

All numerical features are standardized via zero-mean unit-variance scaling (`StandardScaler`), while categorical dimensions are one-hot encoded (`handle_unknown='ignore'`), yielding a 122-dimensional sparse feature space.

### CIC-IDS2017 (Modern Benchmark)

A modern complement built on bidirectional flow statistics captured with CICFlowMeter [3] (34 features: `Flow Bytes/s`, `Flow Packets/s`, inter-arrival timings, etc.), covering contemporary attack profiles such as DDoS (LOIC), PortScan, DoS Hulk, and FTP-Patator brute force.

---

## 5. Empirical Benchmark & Validation Results

### 5.1 5-Fold Stratified Cross-Validation (`KDDTrain+`, N = 125,973, seed = 42)

| Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 97.39% +/- 0.13% | 97.66% +/- 0.14% | 96.71% +/- 0.22% | 0.9718 +/- 0.0015 |
| Decision Tree | 99.84% +/- 0.02% | 99.82% +/- 0.05% | 99.84% +/- 0.02% | 0.9983 +/- 0.0002 |
| **Random Forest** *(Deployed)* | **99.89% +/- 0.02%** | **99.95% +/- 0.02%** | **99.82% +/- 0.02%** | **0.9988 +/- 0.0002** |
| **XGBoost** | **99.90% +/- 0.02%** | **99.92% +/- 0.03%** | **99.87% +/- 0.04%** | **0.9989 +/- 0.0002** |

### 5.2 Out-of-Sample Test Evaluation (`KDDTest+`, N = 22,544)

| Architecture | Accuracy | Precision | Recall (tau=0.50) | F1-Score | False Negatives | Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Logistic Regression | 75.5% | 91.7% | 62.6% | 0.744 | 4,805 | Linear baseline; rapid inference |
| Decision Tree | 80.8% | 95.9% | 69.2% | 0.804 | 3,952 | Non-linear; higher variance |
| XGBoost | 79.3% | 96.8% | 65.9% | 0.784 | 4,380 | Gradient-boosted baseline |
| **Random Forest** *(Deployed)* | **77.5%** | **96.9%** | **62.5%** | **0.760** | 4,807 | Highest precision; minimizes false alarms |

**Production selection rationale:** in enterprise SOCs, false positives drive alert fatigue and unintentional denial of service for benign users. Random Forest's 96.9% precision yields only 261 false positives out of 9,711 normal connections, making it the preferred architecture for automated production filtering.

### 5.3 Granular Threat Family Recall Breakdown (NSL-KDD, Random Forest, tau=0.50)

| Threat Category | Attack Vectors Represented | Test Count | Detected | Missed (FN) | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DoS** | Neptune, Smurf, Apache2, Back, Teardrop | 7,458 | 6,065 | 1,393 | **81.3%** |
| **Probe** | Portsweep, Nmap, Satan, Ipsweep | 2,421 | 1,792 | 629 | **74.0%** |
| **R2L** | Guess_passwd, Warezclient, Sendmail, Imap | 2,754 | 152 | 2,602 | **5.5%** |
| **U2R** | Buffer_overflow, Rootkit, Loadmodule, Perl | 200 | 17 | 183 | **8.5%** |

Volumetric attacks (DoS, Probe) are detected reliably because they manifest clear deviations in packet rates and error rates. Stealthy, low-footprint intrusions (R2L, U2R) closely mimic benign sessions and remain largely invisible to tabular classifiers without payload inspection -- see [Known Limitations](#12-known-limitations).

### 5.4 Operational Threshold Tuning (tau-Calibration on Random Forest)

Lowering the decision boundary from tau=0.50 to tau=0.35 optimizes threat interception for high-security environments:

| Operational Metric | Default (tau=0.50) | Tuned (tau=0.35) | Impact |
| :--- | :---: | :---: | :---: |
| Attack Recall | 62.5% | **69.3%** | **+6.8%** |
| False Negatives (Missed Threats) | 4,807 | **3,946** | **-861 missed attacks** |
| Precision | 96.9% | 96.8% | -0.1% |
| False Positives (Alarms) | 261 | 293 | +32 benign alerts |
| Harmonic F1-Score | 0.760 | **0.807** | **+0.047** |

### 5.5 Modern Benchmark: CIC-IDS2017 Flow Telemetry

Evaluated on 10,000 bidirectional flow records (34 features) over an 80/20 stratified split:

| Architecture | Accuracy | Precision | Recall | F1-Score | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 100.0% | 100.0% | 100.0% | 1.0000 | 0 |
| Decision Tree | 100.0% | 100.0% | 100.0% | 1.0000 | 0 |
| Random Forest | 100.0% | 100.0% | 100.0% | 1.0000 | 0 |
| XGBoost | 100.0% | 100.0% | 100.0% | 1.0000 | 0 |

All four architectures achieve perfect separation on this benchmark, reflecting the high linear separability of modern statistical flow metrics on volumetric threats -- and highlighting the risk of **benchmark saturation** when synthetic flow generators produce artificially distinct statistical footprints.

### 5.6 Explainability Faithfulness: Tree SHAP vs. LIME Tabular

Evaluated via iterative **Deletion** (removing top features to degrade confidence) and **Insertion** (restoring top features from a neutral baseline to recover confidence) curves [8]:

| Explainer | Deletion AUC (lower = more faithful) | Insertion AUC (higher = faster recovery) | Verdict |
| :--- | :---: | :---: | :--- |
| **Tree SHAP** *(Deployed)* | **0.7360** | **0.8739** | Superior faithfulness -- features directly drive tree splits |
| LIME Tabular | 0.8313 | 0.8186 | Local linear surrogate; slower signal degradation |

---

## 6. Explainable AI (XAI) Implementation

Traditional ensemble methods operate as opaque black boxes. This system integrates **Tree SHAP** (`shap.TreeExplainer`) [4, 6] to compute exact Shapley feature attributions for every prediction, satisfying additive efficiency:

$$\hat{f}(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$

where $\phi_0$ is the model's expected output over the training corpus and $\phi_i$ is the credit attributed to feature $i$. As a secondary, model-agnostic cross-check, the system also integrates **LIME** [5], fitting a local linear surrogate around each perturbed instance.

- **Interpretability:** identifies the top network parameters driving an anomaly (e.g., elevated `dst_host_serror_rate`, zero `dst_bytes`, unusual `service`).
- **Directionality:** indicates whether a feature pushed the decision boundary toward **Intrusion** (positive SHAP attribution) or **Normal** behavior (negative SHAP attribution).
- **Faithfulness-verified:** attribution quality is not assumed -- it is empirically measured via Deletion/Insertion AUC (Section 5.6), the same methodology used in [8].

Full derivations for the Deletion/Insertion faithfulness metrics and the Shapley value formulation are provided in the accompanying [research paper](docs/RESEARCH_PAPER.tex).

---

## 7. Empirical Visualizations

All figures are dynamically inspectable inside the web dashboard (`/metrics`) or available as publication-ready vector assets in `results/`:

| Figure | Description | File Path |
| :--- | :--- | :--- |
| Figure 1 | Out-of-sample confusion matrix on `KDDTest+` | `results/fig1_confusion_matrix.png` |
| Figure 2 | Precision-recall / false-negative trade-off across tau in [0.10, 0.90] | `results/fig2_threshold_calibration.png` |
| Figure 3 | XAI faithfulness Deletion/Insertion curves (SHAP vs. LIME) | `results/fig3_xai_faithfulness.png` |
| Figure 4 | Attack-family detection recall asymmetry (DoS, Probe, R2L, U2R) | `results/fig4_attack_recall_asymmetry.png` |

---

## 8. Project Structure

```text
nids-ml/
+-- app/
|   +-- __init__.py             # Application factory & artifact injection
|   +-- model_utils.py          # Preprocessing, inference & SHAP/LIME explainers
|   +-- pcap_utils.py           # Scapy PCAP reader & bidirectional flow builder
|   +-- routes.py               # Route handlers & reporting endpoints
|   +-- static/                 # CSS & UI assets
|   `-- templates/
|       +-- base.html           # Shell layout & navigation
|       +-- index.html          # Single prediction, CSV batch & PCAP upload
|       +-- result.html         # SHAP/LIME explanation card & prediction view
|       +-- batch_result.html   # Batch CSV & PCAP classification dashboard
|       `-- metrics.html        # Empirical benchmarks, XAI curves & diagnostics
+-- data/
|   +-- KDDTrain+.txt           # NSL-KDD training corpus
|   +-- KDDTest+.txt            # NSL-KDD out-of-sample test split
|   +-- cicids2017_sample.csv   # Modern CIC-IDS2017 flow benchmark sample
|   `-- sample_traffic.pcap     # Synthesized HTTP & SYN flood packet capture
+-- model/
|   +-- model.pkl               # Trained Random Forest artifact
|   +-- scaler.pkl              # Fitted StandardScaler
|   +-- encoder.pkl             # Fitted OneHotEncoder
|   +-- metrics.json            # Primary evaluation payload & threshold curves
|   +-- cv_metrics.json         # 5-fold stratified cross-validation results
|   +-- cic_ids2017_metrics.json# Modern flow benchmark results
|   `-- xai_metrics.json        # SHAP vs. LIME Deletion/Insertion curves
+-- docs/
|   `-- RESEARCH_PAPER.tex      # Full IEEE-format academic research paper
+-- results/                    # Generated publication-ready figures (PNG)
+-- notebooks/                  # Exploratory analysis & experiments
+-- tests/
|   +-- conftest.py
|   +-- test_prediction.py      # Core inference & metrics suite
|   `-- test_phase2.py          # PCAP ingestion & XAI faithfulness tests
+-- train_model.py              # Primary model training pipeline
+-- cross_validate.py           # 5-fold CV runner with fixed random seeds
+-- download_cicids2017.py      # CIC-IDS2017 benchmark generation script
+-- evaluate_cicids2017.py      # Modern flow evaluation pipeline
+-- evaluate_xai.py             # Faithfulness benchmarking (SHAP vs. LIME)
+-- generate_sample_pcap.py     # Synthetic PCAP test traffic generator
+-- generate_figures.py         # Publication figure generator
+-- run.py                      # Local application entry point
+-- Procfile                    # Railway deployment process command
+-- requirements.txt            # Production dependencies
+-- LICENSE                     # MIT License
`-- README.md
```

---

## 9. Dataset Acquisition

### NSL-KDD (Primary Benchmark)

- **Source:** Official University of New Brunswick (UNB) repository -- `https://www.unb.ca/cic/datasets/nsl.html`
- **Required files:**
  - `KDDTrain+.txt` -> place at `data/KDDTrain+.txt` (125,973 records)
  - `KDDTest+.txt` -> place at `data/KDDTest+.txt` (22,544 records)

### CIC-IDS2017 (Modern Benchmark Sample)

To avoid a prohibitive multi-day, 50 GB raw PCAP download, use the calibrated generation script:

```bash
python download_cicids2017.py
```

This creates `data/cicids2017_sample.csv` -- 10,000 bidirectional flow records with 34 CICFlowMeter-derived features.

---

## 10. Local Installation & Reproduction

### Prerequisites

- Python 3.10+ (tested through Python 3.14)
- Git

### 10.1 Setup

```bash
git clone https://github.com/ahmadbilad/nids-ml.git
cd nids-ml

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 10.2 Reproduce All Models, Benchmarks & Figures

```bash
# 1. Train NSL-KDD models & generate out-of-sample test split metrics
python train_model.py

# 2. Run 5-fold stratified cross-validation (fixed seed = 42)
python cross_validate.py

# 3. Generate & evaluate the modern CIC-IDS2017 benchmark
python download_cicids2017.py
python evaluate_cicids2017.py

# 4. Synthesize a sample PCAP and evaluate XAI faithfulness (SHAP vs. LIME)
python generate_sample_pcap.py
python evaluate_xai.py

# 5. Generate publication-ready figures into results/
python generate_figures.py

# 6. Run the automated regression & integration test suite
python -m pytest
```

### 10.3 Launch the Web Application

```bash
python run.py
```

- Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to test manual input, batch CSV upload, or replay `data/sample_traffic.pcap`.
- Visit [http://127.0.0.1:5000/metrics](http://127.0.0.1:5000/metrics) to inspect model benchmarks, threshold tuning, and SHAP-vs-LIME faithfulness curves.

---

## 11. Production Deployment & CI/CD

- **Hosting Platform:** Railway
- **Build Engine:** Railway Railpack (Python 3.13 runtime)
- **WSGI Application Server:** Gunicorn 23.0.0 (`app:create_app()`)
- **Continuous Deployment:** automated CI/CD pipeline triggered on pushes to the `main` branch
- **Live scope:** the production deployment serves the full feature set described above -- multi-model comparison, SHAP/LIME explanations, threshold-tuned Random Forest inference, batch CSV ingestion, and the `/metrics` benchmark dashboard

---

## 12. Known Limitations

1. **R2L / U2R zero-day vulnerability:** as shown in the recall breakdown (5.5% on R2L, 8.5% on U2R), tabular connection models trained on legacy features cannot reliably detect subtle payload privilege escalations or stealthy credential stuffing without Deep Packet Inspection (DPI).
2. **Synthetic PCAP flow heuristics:** the embedded Scapy replay utility extracts standard 5-tuple bidirectional metrics but uses heuristic port mapping for protocol abstraction. High-bandwidth enterprise environments require dedicated native collectors such as Zeek or CICFlowMeter.
3. **Base-rate fallacy in real-world SOC operations:** the model demonstrates 96.9% precision on balanced test splits, but in live enterprise networks where attacks represent <0.01% of total traffic, an uncalibrated model will generate unsustainable false-alarm volumes. Threshold tuning to tau=0.35 addresses false negatives but requires upstream signature filtering to mitigate analyst fatigue.
4. **Adversarial robustness:** the tree ensembles are not adversarially trained and remain susceptible to evasion attacks that manipulate packet inter-arrival times or insert dummy padding bytes without altering attack semantics.

---

## 13. Future Work

- Hybrid architectures combining statistical flow features with deep sequence representations (Transformers) for encrypted payload analysis.
- Adversarial training to harden decision trees against evasion perturbations.
- Native flow collector integration (Zeek / CICFlowMeter) to replace heuristic PCAP port-mapping.
- Upstream signature-based pre-filtering to mitigate base-rate false-alarm volume in live deployments.

---

## 14. References

1. R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in *IEEE Symposium on Security and Privacy (SP)*, 2010, pp. 305-316.
2. M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in *IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*, 2009, pp. 1-6.
3. I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)*, 2018, pp. 108-116.
4. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
5. M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 1135-1144.
6. S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56-67, 2020.
7. M. Wang, K. Zheng, Y. Yang, and X. Wang, "An explainable machine learning framework for network intrusion detection systems," *IEEE Access*, vol. 8, pp. 73127-73141, 2020.
8. V. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized input sampling for explanation of black-box models," in *Proc. British Machine Vision Conference (BMVC)*, 2018.

The full methodology, derivations, and discussion are available in the companion paper: [`docs/RESEARCH_PAPER.tex`](docs/RESEARCH_PAPER.tex) (IEEE conference format, ready to compile with `IEEEtran`).

---

## 15. License

This project is licensed under the **MIT License** -- see [`LICENSE`](LICENSE) for details.

---

**Author:** [ahmadbilad](https://github.com/ahmadbilad)
**Repository:** [github.com/ahmadbilad/nids-ml](https://github.com/ahmadbilad/nids-ml)
