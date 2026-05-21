# 🏦 LoanSense — Loan Default Prediction & Automated Risk Reporting

> ML-powered loan risk scoring system built with **7 classical ML models from scratch using NumPy only**, served via a Flask REST API, automated through an n8n pipeline, and visualized in a Streamlit dashboard.

<!-- BANNER: Take a screenshot of the Streamlit dashboard Home page -->
<!-- Save as assets/banner.png and uncomment below -->
<!-- ![LoanSense Banner](assets/banner.png) -->

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey?style=flat-square)
![n8n](https://img.shields.io/badge/n8n-automation-ea4b71?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b?style=flat-square)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [System Architecture](#-system-architecture)
- [Models Built From Scratch](#-models-built-from-scratch)
- [Results](#-results)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [n8n Automation Pipeline](#-n8n-automation-pipeline)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [API Reference](#-api-reference)
- [Sample Output](#-sample-output)
- [Key Learnings](#-key-learnings)

---

## 🎯 Problem Statement

Small lending institutions and NBFCs lack affordable, explainable loan risk scoring systems. Enterprise solutions are expensive and operate as black boxes. Loan officers need a fast, transparent tool that tells them **not just whether to approve a loan, but why**.

---

## 💡 Solution

LoanSense scores loan applicants using **7 classical ML models built entirely from scratch in NumPy** — no sklearn, no black boxes. Every prediction comes with:

- A **default probability** with calibrated threshold
- A **risk level** — HIGH / MEDIUM / LOW
- **Top 3 risk factors** explaining the decision in plain English
- A **recommendation** — Approve / Review / Reject
- A **saved HTML report** per applicant
- A full **audit log** of every prediction
- A **Streamlit dashboard** for live scoring and analytics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Dashboard                │
│         localhost:8501 — 4 pages                │
└──────────────────┬──────────────────────────────┘
                   │ POST /predict
┌──────────────────▼──────────────────────────────┐
│              Flask REST API                     │
│              localhost:5000                     │
│         6 endpoints — validated input           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         Decision Tree Model                     │
│    NumPy from scratch | AUC: 0.83 | F1: 42%     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           n8n Automation Pipeline               │
│              localhost:5678                     │
│  Validate → ML API → HTML Report → Log → Alert  │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
  reports/                  logs/
  LoanSense_*.html     all_predictions.json
                        high_risk_alerts.json
```

---

## 🧠 Models Built From Scratch

All models implemented using **NumPy only** — no sklearn, no PyTorch, no external ML libraries.

| # | Model | File | Key Concept Used |
|---|-------|------|-----------------|
| 1 | **Logistic Regression** | `models/logistic_regression.py` | MLE + Gradient Descent |
| 2 | **LDA** | `models/lda.py` | Class-conditional Gaussians + Eigendecomposition |
| 3 | **Naive Bayes** | `models/naive_bayes.py` | Bayes theorem + Gaussian PDF |
| 4 | **Bayesian Classifier** | `models/bayesian_classifier.py` | MAP estimation |
| 5 | **Decision Tree** | `models/decision_tree.py` | Gini impurity + recursive splitting |
| 6 | **Random Forest** | `models/random_forest.py` | Bootstrap aggregation + majority voting |
| 7 | **Neural Network** | `models/neural_network.py` | Backprop + mini-batch SGD + softmax |

Every model follows the same interface:
```python
model.fit(X_train, y_train)      # train
model.predict(X_test)            # binary prediction
model.predict_proba(X_test)      # probability scores
model.save(path)                 # persist to disk
model.load(path)                 # load from disk
```

---

## 📊 Results

Present in assests

### Best Model Performance (After Hypertuning)

| Metric | Score |
|--------|-------|
| **Model** | Decision Tree |
| **AUC-ROC** | 0.8293 |
| **F1 Score** | 42.16% |
| **Recall** | 44.86% |
| **Precision** | 39.77% |
| **Accuracy** | 91.47% |
| **Threshold** | 0.20 |

> F1 improved from **22% → 42%** after grid search over 60 hyperparameter combinations (max_depth × min_samples_split × threshold).

### Confusion Matrix Breakdown

```
TP =   933  → correctly flagged as risky     ✅
FP = 1,413  → flagged but actually safe      (conservative — by design)
FN = 1,147  → missed actual defaulters       ⚠️
TN = 26,507 → correctly approved             ✅
```

---

## 🖥️ Streamlit Dashboard

4-page interactive dashboard built with Streamlit.


| Page | What it shows |
|------|--------------|
| 🏠 **Home** | Live metrics — total predictions, high risk count, approval rate, recent predictions table |
| 🔍 **Predict** | Full input form → instant risk score → risk badge → top 3 risk factors → n8n pipeline button |
| 📊 **Analytics** | All 5 evaluation charts + live log analysis + risk distribution + CSV download |
| ℹ️ **About** | Architecture, all 7 models, dataset info, key achievements, tech stack |

---

## ⚙️ n8n Automation Pipeline

The ML model is connected to a full **local automation pipeline** built in n8n — no cloud, no Google, runs entirely on your machine.

```
[POST Webhook]
      ↓
[Validate & Normalize Input]     ← checks all 10 fields + ranges
      ↓
[Call Flask ML API]              ← POST localhost:5000/predict
      ↓
[Parse & Build Report]           ← merges ML output + input features
      ↓
[Generate HTML Report]           ← styled, color-coded, branded
      ↓
      ├──> [Save HTML to /reports/]         ← per-applicant file
      ├──> [Log to all_predictions.json]    ← complete audit trail
      └──> [If HIGH RISK → log to alerts]   ← separate alert file
      ↓
[Return JSON Response]
```

**Result:** Raw applicant data → full risk report in **under 2 seconds**.

---

## 📁 Project Structure

```
loansense/
│
├── models/                          ← All 7 ML models from scratch
│   ├── logistic_regression.py
│   ├── lda.py
│   ├── naive_bayes.py
│   ├── bayesian_classifier.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── neural_network.py
│
├── notebooks/
│   ├── eda.ipynb                    ← Exploratory data analysis (10 cells)
│   ├── train.py                     ← Train all 7 models + compare metrics
│   ├── hypertune.py                 ← Grid search — 60 combinations
│   ├── retrain_best.py              ← Retrain with best hyperparameters
│   └── evaluate.py                  ← Generate all 5 evaluation charts
│
├── api/
│   ├── app.py                       ← Flask REST API (6 endpoints)
│   └── test.py                      ← Quick API test script
│
├── n8n/
│   └── loansense_workflow_local.json ← Full automation workflow (import this)
│
├── assets/                          ← All evaluation charts (PNG)
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── precision_recall.png
│   └── ...
│
├── dashboard.py                     ← Streamlit dashboard (4 pages)
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `data/`, `models/saved/`, `logs/`, and `reports/` are gitignored. Download the dataset and train locally.

---

## 🚀 How to Run

### 1. Clone & Install

```bash
git clone https://github.com/harry37725/loansense.git
cd loansense

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Download Dataset

Download `cs-training.csv` from [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it at:
```
data/cs-training.csv
```

### 3. Run EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

### 4. Train All 7 Models

```bash
python notebooks/train.py
```

### 5. Hypertune Best Model

```bash
python notebooks/hypertune.py
python notebooks/retrain_best.py
```

### 6. Generate Evaluation Charts

```bash
python notebooks/evaluate.py
```

### 7. Start Flask API

```bash
python api/app.py
# API live at http://localhost:5000
```

### 8. Start Streamlit Dashboard

```bash
streamlit run dashboard.py
# Dashboard at http://localhost:8501
```

### 9. Import n8n Workflow (optional)

```bash
npx n8n
# Open http://localhost:5678
# New Workflow → ⋮ → Import from file
# Select n8n/loansense_workflow_local.json
# Click "Listen for test event" on Webhook node
```

---

## 🔌 API Reference

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Check API status |
| `POST` | `/predict` | Score a loan applicant |
| `GET`  | `/model-info` | Model metadata & metrics |
| `POST` | `/save-report` | Save HTML report to disk |
| `POST` | `/log-prediction` | Log prediction to JSON |
| `POST` | `/log-high-risk` | Log high risk alert |

### POST /predict — Request Body

```json
{
  "applicant_id"     : "APP-001",
  "revolving_util"   : 0.85,
  "age"              : 28,
  "late_30_59"       : 3,
  "debt_ratio"       : 0.7,
  "monthly_income"   : 3500,
  "open_credit_lines": 6,
  "late_90"          : 2,
  "real_estate_loans": 0,
  "late_60_89"       : 1,
  "dependents"       : 1
}
```

---

## 📬 Sample Output

### API Response

```json
{
  "applicant_id"       : "APP-001",
  "default_probability": 0.623,
  "risk_level"         : "HIGH",
  "prediction"         : "WILL DEFAULT",
  "recommendation"     : "REJECT — High default risk",
  "top_risk_factors"   : [
    "High credit utilization (85%)",
    "2 serious late payment(s) over 90 days",
    "3 late payment(s) in 30-59 day range"
  ],
  "model"              : "Decision Tree",
  "threshold_used"     : 0.2
}
```

### Generated HTML Report

<!-- REPORT SCREENSHOT: Open any file from reports/ folder in browser, screenshot it -->
<!-- Save as assets/sample_report.png and uncomment below -->
<!-- ![Sample Report](assets/sample_report.png) -->

Each prediction generates a styled HTML report saved to `reports/` with:
- Color-coded risk badge (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)
- Default probability percentage
- Top 3 risk factors
- Full applicant details grid
- Recommendation banner

---

## 📚 Key Learnings

- Implemented **MLE, MAP estimation, Bayes theorem, Gini impurity, and backpropagation** from mathematical first principles — no ML libraries
- Handled **severe class imbalance** (6.7% default rate) through threshold tuning — F1 improved from 22% to 42%
- Built a **production-ready REST API** with 6 endpoints, input validation, and fully explainable outputs
- Designed a **full agentic automation pipeline** in n8n — from raw webhook input to saved HTML report in under 2 seconds
- Implemented **mini-batch SGD** in the Neural Network to handle 120k rows efficiently
- Fixed **Decision Tree bottleneck** using percentile-based threshold sampling instead of exhaustive unique value search

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | NumPy (from scratch) |
| Data Processing | Pandas, Matplotlib, Seaborn |
| REST API | Flask |
| Automation | n8n |
| Dashboard | Streamlit |
| Language | Python 3.13 |

---

## 📄 Dataset

**Give Me Some Credit** — [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data)

- 150,000 real loan applicants
- 10 features: credit utilization, age, debt ratio, late payments, income, dependents
- Binary target: serious delinquency in next 2 years (yes/no)
- Class imbalance: 93.3% non-default vs 6.7% default

---
