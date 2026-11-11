# Student Performance & At-Risk Prediction

An end-to-end machine learning early-warning system that estimates whether a student is likely to score below **50/100** on their next assessment (`low_next_score`), built strictly as an **advisory decision-support tool** for teachers and school counselors.

> **Important Scope & Ethics Notice**  
> This project is for **proactive academic support only** (such as check-ins, tutoring, or counselor outreach). It must **never** be used for automated grading, disciplinary action, or admission/placement decisions. Every flagged result requires human review by an educator or counselor.

---

## Live Demo

- **Streamlit App (Placeholder):** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(replace with your deployed Streamlit Community Cloud URL once published)*

To preview locally, run `streamlit run app/streamlit_app.py` from the repository root after following the setup steps below.

---

## Project Summary

- **Goal:** At assessment checkpoint `k` (`k = 3..7`), predict whether a student will score `< 50` on assessment `k + 1` using only observable assessment, attendance, assignment completion, and late-submission records up to checkpoint `k`.
- **Leakage Prevention:** Features use strictly historical data (`1..k`), data splits are chronological (`Train: k = 3,4,5`, `Validation: k = 6`, `Test: k = 7`), imputation (`SimpleImputer(strategy="median")`) and scaling fit only inside the `scikit-learn` `Pipeline` on training data, and `test.csv` is evaluated once after threshold and model selection freeze.
- **Fairness & Privacy Guardrails:** Synthetic `gender` is excluded from model features (`FEATURE_COLUMNS`) and used exclusively in post-hoc bias auditing (`src/audit.py`).
- **Model Selection & Thresholding:** Compares a prior-class dummy baseline and a simple heuristic (`last_score < 50`) against `LogisticRegression`, `RandomForestClassifier`, and `HistGradientBoostingClassifier`. The final selected model (`HistGradientBoostingClassifier`) uses a validation-tuned threshold (`0.37`) maximizing recall-weighted **F2**, paired with **Low / Medium / High** risk bands (`< 0.37`, `0.37–0.76`, `>= 0.76`).
- **Explainability & Interactive UI:** Provides global permutation importance (`src/explain.py`), per-student median-substitution risk drivers, and a two-tab Streamlit web app (`app/streamlit_app.py`) using identical feature logic (`build_features_from_inputs`).

---

## Folder Structure

```text
student-risk-prediction/
├── .env.example                       # Placeholder environment variables template
├── .gitignore                         # Ignores .venv, caches, checkpoints, .env (keeps data/ & models/)
├── PROJECT_CONTEXT.md                 # Project engineering & ethics guardrails
├── README.md                          # Project overview, setup, commands, and limitations
├── requirements.txt                   # Python dependencies
├── app/
│   └── streamlit_app.py               # Two-tab advisory Streamlit application
├── data/
│   ├── raw/
│   │   └── students_raw.csv           # Synthetic longitudinal records (seed=42)
│   └── processed/
│       ├── checkpoint_dataset.csv     # Engineered student-checkpoint dataset (k=3..7)
│       ├── train.csv                  # Chronological training split (k=3,4,5)
│       ├── val.csv                    # Chronological validation split (k=6)
│       └── test.csv                   # Chronological held-out test split (k=7)
├── docs/
│   └── problem_definition.md          # Business objective, metrics, and out-of-scope boundaries
├── models/
│   ├── model.joblib                   # Saved final scikit-learn Pipeline
│   └── metadata.json                  # Features, threshold, risk bands, seed, and validation metrics
├── notebooks/
│   └── 01_eda.ipynb                   # Exploratory data analysis, quality checks, and leakage notes
├── reports/
│   ├── TECHNICAL_REPORT.md            # Detailed technical report with [WRITE IN YOUR OWN WORDS] prompts
│   ├── metrics.json                   # Held-out test metrics, error analysis, and subgroup breakdowns
│   ├── metrics_table.csv              # Model comparison summary table on the test set
│   ├── feature_importance.csv         # Validation permutation importance scores
│   ├── bias_audit.csv                 # Test-set subgroup parity audit by gender and grade_level
│   └── figures/                       # Saved EDA, evaluation, and explainability PNG charts
├── src/
│   ├── __init__.py
│   ├── config.py                      # Central constants (SEED=42, paths, thresholds)
│   ├── data_generator.py              # Reproducible synthetic student data generator
│   ├── data_quality.py                # Quality diagnostics and cleaning (dedupe + invalid -> NaN)
│   ├── features.py                    # Leakage-safe checkpoint feature engineering + app input helper
│   ├── split.py                       # Chronological train/val/test split builder
│   ├── train.py                       # Pipeline training, validation tuning, and model serialization
│   ├── evaluate.py                    # Single-pass test evaluation, error analysis, and charts
│   ├── explain.py                     # Global permutation importance & local median-baseline explanation
│   ├── audit.py                       # Post-hoc fairness audit across synthetic gender and grade_level
│   └── predict.py                     # Cached model loader & end-to-end prediction helper
└── tests/
    ├── __init__.py
    ├── test_data.py                   # Reproducibility & cleaning tests
    ├── test_features.py               # Anti-leakage, column exclusion, and target-shift tests
    ├── test_split.py                  # Strict chronological split ordering test
    └── test_predict.py                # Artifact loading, missing values, and band-boundary tests
```

---

## Setup

- **Python Version:** Python `3.11` (developed and tested on Python `3.11.15`)

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Optional)* Inspect `.env.example` if configuring environment variables locally; no external API keys or secrets are required to run the pipeline or app.

---

## Reproduction Commands

Run the following commands **in exact order** from the repository root (`student-risk-prediction/`):

```bash
python -m src.data_generator
python -m src.split
python -m src.train
python -m src.evaluate
python -m src.audit
pytest
streamlit run app/streamlit_app.py
```

*(Optional)* To regenerate the global permutation-importance artifact and chart (`reports/feature_importance.csv` and `reports/figures/feature_importance.png`) or execute the EDA notebook, run `python -m src.explain` or open `notebooks/01_eda.ipynb`.

---

## Dataset Note (100% Synthetic Data)

- **No Real Student Data:** Every record in `data/raw/students_raw.csv` is **100% synthetic** (`N_STUDENTS = 1500`, `N_ASSESSMENTS = 8`, `SEED = 42`). No real students, schools, or personally identifiable information are used.
- **How It Is Generated (`src/data_generator.py`):**
  - Each student is assigned `grade_level` (`8`, `9`, or `10`), a synthetic `gender` label (`Female`, `Male`, or `Non-binary`, reserved strictly for post-hoc bias auditing), and latent student attributes (**ability** and **effort**) plus slow longitudinal improvement/decline trends and random per-assessment noise.
  - Latent effort and ability drive correlated trajectories across `score` (`0–100`), `attendance_pct`, `assignment_completion_pct`, and `late_submission_pct`, calibrated so roughly `20%` of assessments fall below `50` (class imbalance).
  - Realistic data-quality defects are intentionally injected: ~`3%` missing `attendance_pct`, ~`2%` missing `assignment_completion_pct`, a small number of out-of-range anomalies (negative scores and attendance `> 100`), and ~`0.5%` duplicate rows (`60` duplicates added to `12,000` base rows = `12,060` raw rows).

---

## Known Limitations

1. **Synthetic Data Simplifications:** Real classroom dynamics involve unobserved shocks (illness, family emergencies, curriculum shifts, teacher grading variance) that a latent ability/effort simulator cannot fully replicate. Metrics measured here (`Recall = 89.58%`, `PR-AUC = 0.7195` on the synthetic test set) will not transfer automatically to a real school without retraining and recalibration on local historical records.
2. **Recall-First False Positive Trade-Off:** Tuning the threshold (`0.37`) to optimize **F2** catches most struggling students (`275 / 307` in test) but flags `319` false positives (`Precision = 46.30%`), meaning supportive resources must remain low-stakes and counselor-verified so students are not mislabeled or overwhelmed.
3. **Short Trajectory Cold-Start & Blind Spots:** Predictions start at checkpoint `k = 3` (requiring prior history) and rely heavily on recent scores (`mean_score_last3` permutation importance = `0.4658`), so sudden unannounced drops among previously high-scoring students (`STU_0442`-style false negatives) remain hard to foresee from quantitative logs alone.
4. **Small Subgroup Statistical Variance:** In the synthetic bias audit (`src/audit.py`), the `Non-binary` subgroup contains only `n = 69` test rows, making subgroup parity rates (`flagged = 23.19%`, `recall = 100.0%`, `FPR = 13.11%`) noisy and sensitive to small counts.
5. **Advisory Boundary Only:** Predictions reflect statistical risk correlations, not causal diagnoses of *why* a student is struggling; educators must combine model signals with qualitative context.
