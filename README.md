# Student Performance & At-Risk Prediction

An educational, leak-free Machine Learning internship project that flags students who may benefit from **optional early academic support** before their next assessment (~2–3 weeks ahead).

> **Advisory Only:** Model outputs exist solely to help teachers and mentors offer timely tutoring, check-ins, or study resources. This tool must **never** trigger automated punishment, grading penalties, or admission decisions. Protected/sensitive attributes (such as `gender`) are strictly excluded from model features and kept only for post-hoc fairness auditing.

---

## Problem Summary

- **Prediction unit:** One student right after assessment checkpoint `k` (`k = 3..7`)
- **Target (`low_next_score`):** `1` if assessment `k + 1` score is `< 50` (out of `100`), else `0`
- **Primary priority:** **Recall** (missing a struggling student is costlier than offering optional help), tracked alongside **Precision**, **F1**, **F2**, **Confusion Matrix**, and **PR-AUC**
- **Reproducibility:** `SEED = 42` stored in [`src/config.py`](src/config.py)
- **Anti-leakage design:** Features at checkpoint `k` use only assessments `<= k`, splits are chronological (`train: k=3,4,5`, `val: k=6`, `test: k=7`), preprocessing (`SimpleImputer`, `StandardScaler`) is fitted inside `scikit-learn` `Pipeline`s on training data only, and `test.csv` is evaluated once.

See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and [`docs/problem_definition.md`](docs/problem_definition.md) for the full project rules and problem specification.

---

## Project Structure

```text
student-risk-prediction/
├── app/
│   └── streamlit_app.py       # Advisory Streamlit app (Single Student Check & Cohort Overview)
├── data/
│   ├── raw/                   # Raw synthetic CSV generated with seed 42
│   └── processed/             # Checkpoint feature dataset & train/val/test CSV splits
├── docs/
│   └── problem_definition.md  # Formal problem definition, metrics, and ethical limits
├── models/
│   ├── model.joblib           # Saved winning scikit-learn Pipeline artifact
│   └── metadata.json          # Feature schema, F2 threshold, risk-band cutoffs, validation metrics
├── notebooks/
│   └── 01_eda.ipynb           # Executed EDA notebook with leakage thinking & charts
├── reports/
│   ├── figures/               # EDA, evaluation, and feature-importance PNG charts
│   ├── bias_audit.csv         # Post-hoc fairness audit across gender and grade_level
│   ├── feature_importance.csv # Validation permutation importance table
│   ├── metrics.json           # Held-out test evaluation, breakdowns, and error analysis
│   └── metrics_table.csv      # Held-out test model comparison table
├── src/
│   ├── config.py              # Central paths, SEED=42, and problem constants
│   ├── data_generator.py      # Reproducible synthetic longitudinal dataset generator
│   ├── data_quality.py        # Audit & cleaning (no imputation here)
│   ├── features.py            # Shared leak-free feature builder for training & app
│   ├── split.py               # Chronological split by checkpoint (train/val/test)
│   ├── train.py               # Baselines, hyperparameter search on val, F2 threshold & bands
│   ├── evaluate.py            # One-time held-out test evaluation, plots, error analysis
│   ├── explain.py             # Global permutation importance & local median-perturbation explanations
│   ├── audit.py               # Post-hoc subgroup fairness audit on test.csv
│   └── predict.py             # Live inference helper loading models/model.joblib
├── tests/                     # Automated pytest suite
├── .env.example               # Placeholder environment variables only (no secrets)
├── .gitignore
├── PROJECT_CONTEXT.md
└── requirements.txt
```

---

## Setup Instructions

1. **Create and activate a Python 3.11 virtual environment:**

   ```powershell
   # Using uv (recommended if installed):
   uv venv --python 3.11 .venv

   # Or using Python 3.11 directly:
   python -m venv .venv

   # Activate on Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

---

## Run the End-to-End Pipeline

Run each step from the repository root (`d:\student-risk-prediction`):

```powershell
# 1. Generate synthetic raw data (1,500 students x 8 assessments, seed=42)
python -m src.data_generator

# 2. Check data quality before & after cleaning (no imputation yet)
python -m src.data_quality

# 3. Build leak-free checkpoint features & chronological train/val/test splits
python -m src.split

# 4. Train baselines + candidate Pipelines, tune on val.csv, save models/model.joblib
python -m src.train

# 5. Run one-time held-out evaluation on test.csv, save metrics & figures
python -m src.evaluate

# 6. Compute global permutation importance (val.csv) & test local explanation
python -m src.explain

# 7. Run post-hoc fairness audit on test.csv (gender & grade_level)
python -m src.audit

# 8. Optional CLI inference demo
python -m src.predict
```

---

## Launch the Advisory Streamlit App

```powershell
streamlit run app/streamlit_app.py
```

The app provides two tabs:
- **Single Student Check:** Enter the last 3 assessments (`k-2`, `k-1`, `k`) and grade level to see predicted risk probability, risk band (`Low` / `Medium` / `High`), top 3 risk-increasing and top 3 risk-reducing factors, and non-punitive support suggestions.
- **Cohort & Model Overview:** Inspect the saved test metrics table, confusion matrix, precision-recall curve, model comparison chart, global feature importance, and subgroup bias audit.

---

## Held-Out Test Results Summary (`test.csv`, `k = 7`, `1,498` students)

- **Selected Model:** `HistGradientBoostingClassifier` (`class_weight="balanced"`, validation F2-tuned threshold `= 0.37`)
- **Test Performance:** Recall **`0.8958`** (`275` of `307` struggling students caught), Precision **`0.4630`**, F1 **`0.6104`**, F2 **`0.7547`**, PR-AUC **`0.7195`** (`TN=872, FP=319, FN=32, TP=275`)
- **Baselines on Test:** `DummyClassifier(strategy="prior")` (`Recall=0.0000`, `PR-AUC=0.2049`); simple `last_score < 50` rule (`Recall=0.6287`, `Precision=0.6328`, `PR-AUC=0.4739`)
- **Post-hoc Gender Parity (`gender` excluded from features):** Female (`Recall=0.8931`, `FPR=0.2789`) vs. Male (`Recall=0.8929`, `FPR=0.2712`)

