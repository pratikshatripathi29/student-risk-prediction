# Project Context & Permanent Rules

**Project:** Student Performance & At-Risk Prediction (ML Internship Project)

These rules apply throughout the entire project and must never be broken.

---

## 1. Tech Stack

Use only these core tools and libraries unless explicitly approved:

- **Python:** 3.11
- **Data & Math:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Visualization:** `matplotlib`
- **Model Persistence:** `joblib`
- **Web App / UI:** `streamlit`
- **Testing:** `pytest`

---

## 2. Reproducibility

- **Random seed = `42` everywhere.**
- Store the shared seed constant in `src/config.py` and import it wherever randomness is involved (data generation, train/validation/test splits, model initialization, cross-validation, etc.).

---

## 3. Code Style & Mentorship Standard

- Write **simple, readable code** with clear variable names and helpful comments explaining *why* a step exists (especially for tricky ML steps).
- Avoid over-engineering or overly clever shortcuts.
- **After creating any new file, explain it in plain language** so the purpose and logic remain easy to understand.

---

## 4. Machine Learning Integrity & Anti-Leakage Rules

- **Fit preprocessing on training data only.** Always use `scikit-learn` `Pipeline` (and `ColumnTransformer` when needed) so scalers, encoders, and imputers never peek at validation or test data.
- **Test data is used only once, at final evaluation.** Do not tune hyperparameters, select models, or make feature decisions using the test set. Use a validation set or cross-validation on the training data for development decisions.
- **Never hard-code predictions.** Both evaluations and the Streamlit app must load and run the saved model artifact (`.joblib`).

---

## 5. Fairness, Privacy & Security

- **No protected or sensitive attributes as model features.** Never pass attributes such as gender, religion, caste, disability, family income, ethnicity, or similar demographic/socioeconomic fields into the model.
- **No secrets or real personal data in the repository.** Never commit API keys, passwords, tokens, student names, emails, or real institutional records. Only commit `.env.example` with placeholder values if environment variables are needed.

---

## 6. Ethical Use Boundary

- **Advisory only.** Predictions are meant solely to help educators and mentors offer timely, supportive academic help (such as tutoring, check-ins, or study resources).
- This tool must **never** trigger automated punishment, grading penalties, academic probation, or admission/expulsion decisions.

