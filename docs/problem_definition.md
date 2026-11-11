# Problem Definition: Student Performance & At-Risk Prediction

## 1. Business Decision

**What decision does this model support?**
Identifying which students may benefit from extra academic support (such as tutoring, study groups, office-hour check-ins, or targeted learning resources) *before* they fall too far behind.

---

## 2. Prediction Unit

**What is one row (one prediction) in our dataset?**
One student at one checkpoint — specifically, right after **assessment `k`**, where `k` ranges from `3` to `7` (`k = 3, 4, 5, 6, 7`).

- Each `(student_id, checkpoint_k)` pair is a single prediction instance.
- Starting at `k = 3` ensures we have enough prior assessment history for each student before making a prediction.

---

## 3. Target Variable

- **Target name:** `low_next_score`
- **Definition:**
  - `1` (Positive / At-Risk) if the **next** assessment (`k + 1`) score is **below 50 out of 100** (`score_{k+1} < 50`).
  - `0` (Negative / Not At-Risk) if the **next** assessment (`k + 1`) score is **50 or above** (`score_{k+1} >= 50`).

---

## 4. Prediction Horizon & Prediction Time

- **Prediction Horizon:** The next assessment (`k + 1`), which occurs roughly **2–3 weeks** after assessment `k`.
- **Prediction Time:** Immediately after assessment `k` results are recorded.
- **Information Cutoff (No Future Leakage):** Only information available **up to and including assessment `k`** may be used to predict `low_next_score` for assessment `k + 1`. Any scores, attendance, or activity recorded after assessment `k` are strictly forbidden as features.

---

## 5. Metric Choice & Why Recall Matters Most

### Primary Priority: Recall (Sensitivity)
In early academic intervention, errors have unequal costs:
- **False Negative (missed struggling student):** A student who needed help gets overlooked, struggles on the next assessment, and may fall further behind. **High cost.**
- **False Positive (extra help offered unnecessarily):** A student who would have scored >= 50 is invited to optional tutoring or a check-in. **Low cost** (and often still mildly helpful).

Because missing an at-risk student is much more harmful than offering unnecessary support, **recall** is prioritized over precision.

### Balanced Tracking
At the same time, if the model flags *everyone* as at-risk, support staff will be overwhelmed (alert fatigue). Therefore, alongside **recall**, we track **precision** and **F1** to ensure alerts stay practical and actionable.

### Success Metrics to Report
1. **Recall** (how many truly at-risk students we catch)
2. **Precision** (of the students we flag, how many truly needed help)
3. **F1 Score** (balance between precision and recall)
4. **Confusion Matrix** (exact counts of True Positives, False Positives, True Negatives, False Negatives)
5. **PR-AUC (Precision-Recall Area Under the Curve)** (overall ranking quality across thresholds, especially informative when at-risk cases are the minority class)

---

## 6. Out of Scope & Ethical Limits

### Out of Scope
- Predicting final course grades, graduation outcomes, or multi-year retention.
- Making predictions before assessment 3 (`k < 3`) when there is not yet enough baseline history.
- Using protected or sensitive personal attributes (such as gender, religion, caste, disability, or family income) as inputs.
- Real-time LMS integration, automated emailing, or production authentication systems (for this internship scope, the focus is an accurate, reproducible pipeline and a clean advisory Streamlit app).

### Ethical Limits
- **Human-in-the-loop only:** Model outputs are advisory signals for educators, never final judgments.
- **No punitive automation:** Predictions must never be used to penalize grades, restrict course enrollment, trigger disciplinary action, or make admission/scholarship decisions.
- **Transparency & Humility:** Scores reflect patterns in past academic engagement and assessment performance, not a student's fixed potential. Every intervention should be supportive and respectful of student privacy.

