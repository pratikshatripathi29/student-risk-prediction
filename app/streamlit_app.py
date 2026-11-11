"""Advisory Student Support Tool built with Streamlit.

Rules enforced from PROJECT_CONTEXT.md:
- Loads the saved scikit-learn Pipeline (`models/model.joblib`) and metadata
  (`models/metadata.json`) via `@st.cache_resource`—never hard-codes predictions.
- Uses `build_features_from_inputs` from `src/features.py` so feature logic is
  identical to training.
- Uses `explain_prediction` from `src/explain.py` to show plain-English risk
  and protective drivers.
- Strictly advisory only: never displays sensitive/protected attributes (no
  gender input) and never automates punishment, grading, or admissions.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
import streamlit as st

# Ensure project root is importable regardless of where `streamlit run` is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BIAS_AUDIT_PATH,
    FIGURES_DIR,
    METRICS_TABLE_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)
from src.explain import explain_prediction
from src.features import build_features_from_inputs
from src.predict import assign_risk_band, load_artifacts


@st.cache_resource
def get_model_and_metadata() -> tuple[Pipeline, dict[str, Any]]:
    """Load and cache the saved model Pipeline and metadata once per app session."""
    return load_artifacts(model_path=MODEL_PATH, metadata_path=MODEL_METADATA_PATH)


def suggest_supportive_actions(
    risk_band: str,
    increasing_drivers: list[dict[str, Any]],
) -> list[str]:
    """Generate constructive, non-punitive academic support ideas for educators."""
    actions: list[str] = []

    if risk_band == "High":
        actions.append(
            "Schedule a supportive 1-on-1 check-in within the next week to understand "
            "any academic or personal roadblocks."
        )
        actions.append(
            "Offer targeted tutoring, guided office hours, or small-group review before the next assessment."
        )
    elif risk_band == "Medium":
        actions.append(
            "Send an encouraging check-in note and invite the student to optional review sessions or study hours."
        )
        actions.append(
            "Check progress midway through the next 2-3 week unit rather than waiting until the next exam."
        )
    else:
        actions.append(
            "Continue routine classroom encouragement and standard check-ins—current indicators look steady."
        )

    driver_features = {d["feature"] for d in increasing_drivers}
    if driver_features & {"completion_last", "completion_mean_last3", "completion_change"}:
        actions.append(
            "Assignment completion is lagging: help the student break upcoming coursework into smaller milestones."
        )
    if driver_features & {"attendance_last", "attendance_mean_last3", "attendance_change"}:
        actions.append(
            "Recent attendance dipped: share missed class notes/recordings and check whether support is needed."
        )
    if driver_features & {"late_submission_last", "late_submission_mean_last3"}:
        actions.append(
            "Late submissions are elevated: review time-management strategies and clarify assignment deadlines early."
        )
    if driver_features & {"last_score", "mean_score_last3", "min_score_last3", "score_slope_last3"}:
        actions.append(
            "Recent scores suggest concept gaps: review questions missed on the last assessment together."
        )

    return actions


def render_single_student_tab(model: Pipeline, metadata: dict[str, Any]) -> None:
    """Render interactive last-3 assessment inputs, live risk estimate, and explanations."""
    st.subheader("Single Student Check (Checkpoint k -> Next Assessment k+1)")
    st.write(
        "Enter the student's **last 3 chronological assessments** (`k-2`, `k-1`, `k` [most recent]) "
        "and grade level to estimate the advisory probability of scoring below `50/100` on the next assessment."
    )

    grade_level = st.selectbox("Grade Level", options=[8, 9, 10], index=1)

    col_old, col_prev, col_last = st.columns(3)

    with col_old:
        st.markdown("**Assessment `k-2` (Oldest of 3)**")
        score_1 = st.slider("Score (k-2)", 0.0, 100.0, 66.0, 0.5, key="score_1")
        att_1 = st.slider("Attendance % (k-2)", 0.0, 100.0, 86.0, 0.5, key="att_1")
        comp_1 = st.slider("Assignment Completion % (k-2)", 0.0, 100.0, 82.0, 0.5, key="comp_1")
        late_1 = st.slider("Late Submissions % (k-2)", 0.0, 100.0, 12.0, 0.5, key="late_1")

    with col_prev:
        st.markdown("**Assessment `k-1` (Previous)**")
        score_2 = st.slider("Score (k-1)", 0.0, 100.0, 58.0, 0.5, key="score_2")
        att_2 = st.slider("Attendance % (k-1)", 0.0, 100.0, 82.0, 0.5, key="att_2")
        comp_2 = st.slider("Assignment Completion % (k-1)", 0.0, 100.0, 75.0, 0.5, key="comp_2")
        late_2 = st.slider("Late Submissions % (k-1)", 0.0, 100.0, 18.0, 0.5, key="late_2")

    with col_last:
        st.markdown("**Assessment `k` (Most Recent)**")
        score_3 = st.slider("Score (k)", 0.0, 100.0, 48.0, 0.5, key="score_3")
        att_3 = st.slider("Attendance % (k)", 0.0, 100.0, 76.0, 0.5, key="att_3")
        comp_3 = st.slider("Assignment Completion % (k)", 0.0, 100.0, 68.0, 0.5, key="comp_3")
        late_3 = st.slider("Late Submissions % (k)", 0.0, 100.0, 24.0, 0.5, key="late_3")

    # Build features using the exact same helper used in training/splitting.
    features_df = build_features_from_inputs(
        scores_last3=[score_1, score_2, score_3],
        attendance_last3=[att_1, att_2, att_3],
        completion_last3=[comp_1, comp_2, comp_3],
        late_submission_last3=[late_1, late_2, late_3],
        grade_level=int(grade_level),
    )

    feature_cols = list(metadata["feature_columns"])
    threshold = float(metadata["decision_threshold"])
    band_cutoffs = dict(metadata["risk_band_cutoffs"])

    risk_proba = float(model.predict_proba(features_df[feature_cols])[:, 1][0])
    risk_band = assign_risk_band(risk_proba, band_cutoffs)
    flagged = risk_proba >= threshold

    explanation = explain_prediction(model, features_df, top_n=3)

    st.divider()
    st.subheader("Advisory Risk Assessment")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Risk (Score < 50)", f"{risk_proba * 100:.1f}%")
    m2.metric("Risk Band", risk_band)
    m3.metric(
        "Optional Support Alert",
        "Recommended" if flagged else "Not Flagged",
        delta=f"Threshold: {threshold * 100:.0f}%",
        delta_color="off",
    )

    st.caption(
        f"Band cutoffs calibrated on validation data: "
        f"Low (< {band_cutoffs['low_max']:.2f}), "
        f"Medium ({band_cutoffs['low_max']:.2f} to < {band_cutoffs['medium_max']:.2f}), "
        f"High (>= {band_cutoffs['medium_max']:.2f})."
    )

    exp_col_1, exp_col_2 = st.columns(2)
    with exp_col_1:
        st.markdown("#### Top Factors Increasing Risk")
        if explanation["increasing_risk"]:
            for item in explanation["increasing_risk"]:
                st.markdown(f"- {item['explanation']}")
        else:
            st.write("No features are currently pushing risk above typical median levels.")

    with exp_col_2:
        st.markdown("#### Top Factors Reducing Risk")
        if explanation["reducing_risk"]:
            for item in explanation["reducing_risk"]:
                st.markdown(f"- {item['explanation']}")
        else:
            st.write("No protective features are pulling risk below typical median levels.")

    st.markdown("#### Suggested Supportive Actions (Optional & Non-Punitive)")
    for action in suggest_supportive_actions(risk_band, explanation["increasing_risk"]):
        st.markdown(f"- {action}")

    with st.expander("Inspect Engineered Features Passed to Model"):
        st.dataframe(features_df.T.rename(columns={0: "value"}), width="stretch")


def render_cohort_overview_tab(metadata: dict[str, Any]) -> None:
    """Render saved test evaluation tables, figures, and global feature importance."""
    st.subheader("Cohort Evaluation & Model Overview")
    st.markdown(
        f"**Saved Model:** `{metadata.get('model_family')}` | "
        f"**Decision Threshold (Val F2-tuned):** `{metadata.get('decision_threshold')}` | "
        f"**Random Seed:** `{metadata.get('seed')}` | "
        f"**scikit-learn:** `{metadata.get('sklearn_version')}`"
    )

    if METRICS_TABLE_PATH.exists():
        st.markdown("#### Held-Out Test Set Metrics Comparison (`test.csv`, Checkpoint `k = 7`)")
        metrics_df = pd.read_csv(METRICS_TABLE_PATH)
        st.dataframe(metrics_df, width="stretch", hide_index=True)
    else:
        st.info("Run `python -m src.evaluate` to generate `reports/metrics_table.csv`.")

    fig_col_1, fig_col_2 = st.columns(2)
    cm_path = FIGURES_DIR / "test_confusion_matrix.png"
    pr_path = FIGURES_DIR / "test_precision_recall_curve.png"
    comp_path = FIGURES_DIR / "test_model_comparison.png"
    fi_path = FIGURES_DIR / "feature_importance.png"

    with fig_col_1:
        if cm_path.exists():
            st.image(str(cm_path), caption="Final Model Confusion Matrix (Held-Out Test Set)")
        else:
            st.info("Run `python -m src.evaluate` to generate the confusion matrix figure.")

    with fig_col_2:
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curves & Chosen Threshold")
        else:
            st.info("Run `python -m src.evaluate` to generate the precision-recall curve figure.")

    fig_col_3, fig_col_4 = st.columns(2)
    with fig_col_3:
        if comp_path.exists():
            st.image(str(comp_path), caption="Baselines vs. Models Comparison on Test Set")
        else:
            st.info("Run `python -m src.evaluate` to generate the model comparison chart.")

    with fig_col_4:
        if fi_path.exists():
            st.image(str(fi_path), caption="Global Permutation Importance (Validation PR-AUC Drop)")
        else:
            st.info("Run `python -m src.explain` to generate the feature importance figure.")

    if BIAS_AUDIT_PATH.exists():
        st.markdown("#### Post-Hoc Subgroup Audit (`gender` Never Used as a Model Feature)")
        audit_df = pd.read_csv(BIAS_AUDIT_PATH)
        st.dataframe(audit_df, width="stretch", hide_index=True)


def main() -> None:
    """Configure page layout, render advisory notice, and mount the two app tabs."""
    st.set_page_config(
        page_title="Student Support Advisory Tool",
        layout="wide",
    )

    st.title("Student Performance & Early Support Advisory Tool")
    st.warning(
        "**Advisory Only:** Predictions estimate whether a student may benefit from optional "
        "academic support (such as tutoring, check-ins, or study resources) on the next assessment. "
        "This tool must **NEVER** be used for automated punishment, grading penalties, academic "
        "probation, or admission decisions. Protected attributes (such as gender) are never used as inputs."
    )

    try:
        model, metadata = get_model_and_metadata()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()

    tab_single, tab_overview = st.tabs(["Single Student Check", "Cohort & Model Overview"])

    with tab_single:
        render_single_student_tab(model, metadata)

    with tab_overview:
        render_cohort_overview_tab(metadata)


if __name__ == "__main__":
    main()
