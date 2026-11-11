"""Advisory Student Support Tool built with Streamlit.

Deployment & rules enforced from PROJECT_CONTEXT.md:
- Adds the repository root to `sys.path` at the very top so the app runs cleanly
  both locally (`streamlit run app/streamlit_app.py`) and on Streamlit Community Cloud.
- Loads the saved scikit-learn Pipeline (`models/model.joblib`) and metadata
  (`models/metadata.json`) via `@st.cache_resource` wrapping `src.predict.load_model()`.
- Real predictions only: uses `src.predict.predict_risk()` (which calls
  `build_features_from_inputs()` from `src/features.py` and `explain_prediction()`
  from `src/explain.py`). Nothing is hard-coded.
- Strictly advisory: displays a clear human-review banner, allows blank inputs to
  test missing-data imputation, and validates 0-100 ranges.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import streamlit as st

# Add repo root to sys.path so `from src...` works on Streamlit Community Cloud and locally.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, METRICS_PATH, MODEL_METADATA_PATH, MODEL_PATH
from src.predict import load_model, predict_risk


@st.cache_resource
def get_cached_model() -> tuple[Pipeline, dict[str, Any]]:
    """Load and cache the model artifact once across Streamlit reruns."""
    return load_model(model_path=MODEL_PATH, metadata_path=MODEL_METADATA_PATH)


def parse_optional_percentage(raw_value: str, field_label: str) -> tuple[float, str | None]:
    """Parse an editable text input into a 0-100 float or NaN when left blank.

    Returns (parsed_float_or_nan, error_message_or_none).
    """
    text = raw_value.strip()
    if text == "":
        return float("nan"), None

    try:
        numeric = float(text)
    except ValueError:
        return float("nan"), f"**{field_label}** must be a number between 0 and 100 (or left blank)."

    if np.isnan(numeric) or np.isinf(numeric) or not (0.0 <= numeric <= 100.0):
        return float("nan"), f"**{field_label}** must be between 0 and 100 (got `{raw_value}`)."

    return numeric, None


def render_risk_band_badge(risk_band: str) -> None:
    """Display the Low / Medium / High risk band with clear color coding."""
    if risk_band == "High":
        st.error("**Risk Band: HIGH** — Priority optional support recommended.")
    elif risk_band == "Medium":
        st.warning("**Risk Band: MEDIUM** — Check-in and optional support recommended.")
    else:
        st.success("**Risk Band: LOW** — Continue routine classroom support.")


def render_prediction_tab(model_bundle: tuple[Pipeline, dict[str, Any]]) -> None:
    """Render student inputs (allowing blanks & 0-100 validation) and advisory output."""
    st.subheader("Check a Student After Assessment k (Predicting Next Assessment k+1)")
    st.caption(
        "Enter the student's last 3 assessments in chronological order (`k-2` oldest, `k-1` previous, `k` most recent). "
        "Leave any box **blank** to test how the Pipeline's training-median imputer handles missing data."
    )

    grade_level = st.selectbox("Grade Level", options=[8, 9, 10], index=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Assessment `k-2` (Oldest of 3)**")
        s1_raw = st.text_input("Score (k-2)", value="64", placeholder="0-100 or blank", key="s1")
        a1_raw = st.text_input("Attendance % (k-2)", value="86", placeholder="0-100 or blank", key="a1")
        c1_raw = st.text_input("Assignment Completion % (k-2)", value="80", placeholder="0-100 or blank", key="c1")
        l1_raw = st.text_input("Late Submission % (k-2)", value="12", placeholder="0-100 or blank", key="l1")

    with col2:
        st.markdown("**Assessment `k-1` (Previous)**")
        s2_raw = st.text_input("Score (k-1)", value="55", placeholder="0-100 or blank", key="s2")
        a2_raw = st.text_input("Attendance % (k-1)", value="79", placeholder="0-100 or blank", key="a2")
        c2_raw = st.text_input("Assignment Completion % (k-1)", value="71", placeholder="0-100 or blank", key="c2")
        l2_raw = st.text_input("Late Submission % (k-1)", value="20", placeholder="0-100 or blank", key="l2")

    with col3:
        st.markdown("**Assessment `k` (Most Recent)**")
        s3_raw = st.text_input("Score (k)", value="46", placeholder="0-100 or blank", key="s3")
        a3_raw = st.text_input("Attendance % (k)", value="72", placeholder="0-100 or blank", key="a3")
        c3_raw = st.text_input("Assignment Completion % (k)", value="62", placeholder="0-100 or blank", key="c3")
        l3_raw = st.text_input("Late Submission % (k)", value="28", placeholder="0-100 or blank", key="l3")

    # Parse and validate all 12 inputs (allowing blanks as NaN).
    raw_fields = [
        ("Score (k-2)", s1_raw),
        ("Score (k-1)", s2_raw),
        ("Score (k)", s3_raw),
        ("Attendance % (k-2)", a1_raw),
        ("Attendance % (k-1)", a2_raw),
        ("Attendance % (k)", a3_raw),
        ("Assignment Completion % (k-2)", c1_raw),
        ("Assignment Completion % (k-1)", c2_raw),
        ("Assignment Completion % (k)", c3_raw),
        ("Late Submission % (k-2)", l1_raw),
        ("Late Submission % (k-1)", l2_raw),
        ("Late Submission % (k)", l3_raw),
    ]

    parsed_values: list[float] = []
    validation_errors: list[str] = []
    for label, raw_val in raw_fields:
        num, err = parse_optional_percentage(raw_val, label)
        if err is not None:
            validation_errors.append(err)
        parsed_values.append(num)

    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    inputs = {
        "grade_level": int(grade_level),
        "scores_last3": parsed_values[0:3],
        "attendance_last3": parsed_values[3:6],
        "completion_last3": parsed_values[6:9],
        "late_submission_last3": parsed_values[9:12],
    }

    result = predict_risk(inputs, model_bundle=model_bundle)

    if result["missing_input_count"] > 0:
        st.info(
            f"Detected **{result['missing_input_count']} blank input(s)**. Missing entries are "
            "handled automatically inside the scikit-learn Pipeline using training-set medians."
        )

    st.divider()
    st.subheader("Advisory Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Probability (Next Score < 50)", f"{result['probability'] * 100:.1f}%")
    m2.metric("Risk Band", result["risk_band"])
    m3.metric(
        "Support Flag Decision",
        "Flagged for Optional Support" if result["flagged"] else "Not Flagged",
        delta=f"Cutoff: {result['decision_threshold'] * 100:.0f}%",
        delta_color="off",
    )

    render_risk_band_badge(result["risk_band"])

    cutoffs = result["risk_band_cutoffs"]
    st.caption(
        f"Risk bands calibrated on validation data: "
        f"Low (< {cutoffs['low_max'] * 100:.0f}%), "
        f"Medium ({cutoffs['low_max'] * 100:.0f}% to < {cutoffs['medium_max'] * 100:.0f}%), "
        f"High (>= {cutoffs['medium_max'] * 100:.0f}%)."
    )

    sig_col1, sig_col2 = st.columns(2)
    with sig_col1:
        st.markdown("#### Top Signals Increasing Risk")
        if result["increasing_risk"]:
            for item in result["increasing_risk"]:
                st.markdown(f"- {item['explanation']}")
        else:
            st.write("No signals are pushing predicted risk above typical median levels.")

    with sig_col2:
        st.markdown("#### Top Signals Reducing Risk")
        if result["reducing_risk"]:
            for item in result["reducing_risk"]:
                st.markdown(f"- {item['explanation']}")
        else:
            st.write("No protective signals are pulling predicted risk below typical median levels.")

    st.markdown("#### Human Review Note")
    st.info(
        "**Before reaching out:** A teacher or counselor should review context the model cannot see "
        "(recent illness, family events, partial/ungraded make-up work, or classroom engagement) and "
        "choose a supportive, non-punitive check-in or tutoring invitation."
    )
    st.text_area(
        "Optional counselor/teacher review notes (kept in browser session only; not saved):",
        placeholder="Example: Check in after class about missed assignment deadlines and invite to Thursday study group.",
        key="human_review_note",
    )

    with st.expander("Limitations & uncertainty"):
        st.markdown(
            "- **Probabilistic estimate, not a certainty:** A `70%` risk score means roughly `7` out of `10` "
            "similar historical checkpoints scored below `50`, not that a specific student will definitely fail.\n"
            "- **False alarms vs. missed students:** Because the threshold (`0.37`) prioritizes catching struggling "
            "students (**Recall ~90%** on test data), about half of flagged students (**Precision ~46%**) still "
            "score `>= 50`—which is why flags must only trigger low-cost, optional help.\n"
            "- **Sudden drops can be missed:** Students with steady past scores who encounter an unexpected difficulty "
            "before the next exam may not be flagged early.\n"
            "- **Missing inputs increase uncertainty:** When fields are left blank, the model falls back to "
            "training medians rather than the student's true recent behavior.\n"
            "- **Synthetic internship scope:** Trained on a simulated 1,500-student cohort; real school deployment "
            "would require local calibration, governance review, and ongoing fairness monitoring."
        )

    with st.expander("Computed features sent to model"):
        feat_df = pd.DataFrame(
            {"feature": list(result["computed_features"].keys()), "value": list(result["computed_features"].values())}
        )
        st.dataframe(feat_df, width="stretch", hide_index=True)


def render_performance_tab() -> None:
    """Read reports/metrics.json and display test metrics table and confusion matrix."""
    st.subheader("Model Performance (One-Time Held-Out Test Evaluation)")

    if not METRICS_PATH.exists():
        st.error(f"`{METRICS_PATH}` not found. Run `python -m src.evaluate` first.")
        return

    metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    st.markdown(
        f"**Final Model:** `{metrics_data.get('final_model')}` | "
        f"**Frozen Threshold (from Validation):** `{metrics_data.get('frozen_threshold_from_val')}` | "
        f"**Test Rows (`k = 7`):** `{metrics_data.get('test_rows')}` | "
        f"**At-Risk Rate:** `{metrics_data.get('test_at_risk_rate', 0) * 100:.2f}%`"
    )

    st.markdown("#### Test Set Comparison Table (Baselines & Candidate Models)")
    models_table = pd.DataFrame(metrics_data.get("models_test_metrics", []))
    st.dataframe(models_table, width="stretch", hide_index=True)

    # Resolve image path relative to repo FIGURES_DIR so it works on both Windows and Linux cloud.
    cm_fig_path = FIGURES_DIR / "test_confusion_matrix.png"
    pr_fig_path = FIGURES_DIR / "test_precision_recall_curve.png"
    comp_fig_path = FIGURES_DIR / "test_model_comparison.png"

    col_cm, col_pr = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion Matrix (Final Model on `test.csv`)")
        if cm_fig_path.exists():
            st.image(str(cm_fig_path), caption="Held-out test confusion matrix at threshold 0.37")
        else:
            st.info("Confusion matrix figure not found. Run `python -m src.evaluate`.")

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        if pr_fig_path.exists():
            st.image(str(pr_fig_path), caption="Precision-Recall curves with chosen validation threshold marked")
        else:
            st.info("Precision-Recall curve figure not found. Run `python -m src.evaluate`.")

    if comp_fig_path.exists():
        st.markdown("#### Model Comparison Chart")
        st.image(str(comp_fig_path), caption="Recall, Precision, F1, and PR-AUC across baselines and tuned models")

    if "grade_level_breakdown" in metrics_data:
        st.markdown("#### Performance Breakdown by Grade Level")
        st.dataframe(pd.DataFrame(metrics_data["grade_level_breakdown"]), width="stretch", hide_index=True)


def main() -> None:
    """Set page layout, show required advisory banner, and mount app tabs."""
    st.set_page_config(page_title="Student Support Advisory Tool", layout="wide")

    st.title("Student Performance & Early Support Advisory Tool")
    st.warning(
        "**Advisory tool only. Not for grading, punishment or admission decisions. "
        "A teacher/counselor must review every result.**"
    )

    try:
        model_bundle = get_cached_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab_predict, tab_perf = st.tabs(["Predict Student Risk", "Model performance"])

    with tab_predict:
        render_prediction_tab(model_bundle)

    with tab_perf:
        render_performance_tab()


if __name__ == "__main__":
    main()
