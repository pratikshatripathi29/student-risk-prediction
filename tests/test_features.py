"""Tests for leak-free checkpoint features, excluded columns, and target alignment."""

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, TRAIN_DATA_PATH
from src.data_quality import clean_data
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    build_features_from_inputs,
)


def _make_student_history(scores: list[float]) -> pd.DataFrame:
    """Create a valid 8-assessment student DataFrame for controlled feature tests."""
    return pd.DataFrame(
        [
            {
                "student_id": "STU_TEST",
                "grade_level": 9,
                "gender": "Male",
                "assessment_no": idx + 1,
                "assessment_date": f"2025-09-{idx + 1:02d}",
                "score": score,
                "attendance_pct": 85.0 - idx,
                "assignment_completion_pct": 80.0 - idx,
                "late_submission_pct": 10.0 + idx,
            }
            for idx, score in enumerate(scores)
        ]
    )


def test_checkpoint_features_do_not_leak_future_rows() -> None:
    """Altering assessments after k (e.g., 4..8) must never change features at checkpoint k=3."""
    base_df = _make_student_history([72.0, 66.0, 60.0, 75.0, 80.0, 85.0, 90.0, 92.0])
    altered_df = base_df.copy()
    altered_df.loc[altered_df["assessment_no"] > 3, "score"] = 5.0
    altered_df.loc[altered_df["assessment_no"] > 3, "attendance_pct"] = 0.0
    altered_df.loc[altered_df["assessment_no"] > 3, "assignment_completion_pct"] = 0.0
    altered_df.loc[altered_df["assessment_no"] > 3, "late_submission_pct"] = 100.0

    feat_base = build_features(base_df)
    feat_altered = build_features(altered_df)

    row_base = feat_base.loc[feat_base["checkpoint_no"] == 3, FEATURE_COLUMNS].iloc[0]
    row_altered = feat_altered.loc[feat_altered["checkpoint_no"] == 3, FEATURE_COLUMNS].iloc[0]
    pd.testing.assert_series_equal(row_base, row_altered)


def test_gender_and_checkpoint_no_not_in_feature_columns() -> None:
    """Protected attributes, IDs, checkpoint index, and target must never be model features."""
    assert "gender" not in FEATURE_COLUMNS
    assert "checkpoint_no" not in FEATURE_COLUMNS
    assert "student_id" not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_target_uses_next_assessment_k_plus_1() -> None:
    """Target low_next_score at checkpoint k must reflect whether assessment k+1 score < 50."""
    # Assessments: 1=70, 2=65, 3=60, 4=49.9 (<50 -> 1), 5=50.0 (>=50 -> 0), 6=45.0 (<50 -> 1), 7=80, 8=NaN
    df = _make_student_history([70.0, 65.0, 60.0, 49.9, 50.0, 45.0, 80.0, np.nan])
    features_df = build_features(df).set_index("checkpoint_no")

    assert features_df.loc[3, TARGET_COLUMN] == 1  # assessment 4 is 49.9 (< 50)
    assert features_df.loc[4, TARGET_COLUMN] == 0  # assessment 5 is 50.0 (>= 50, boundary!)
    assert features_df.loc[5, TARGET_COLUMN] == 1  # assessment 6 is 45.0 (< 50)
    assert features_df.loc[6, TARGET_COLUMN] == 0  # assessment 7 is 80.0 (>= 50)
    assert 7 not in features_df.index  # assessment 8 is NaN -> checkpoint 7 skipped cleanly


def test_training_serving_feature_parity() -> None:
    """build_features_from_inputs must match build_features output on train.csv."""
    raw_clean = clean_data(pd.read_csv(RAW_DATA_PATH))
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    stu_k3 = raw_clean[
        (raw_clean["student_id"] == "STU_0001") & (raw_clean["assessment_no"] <= 3)
    ].sort_values("assessment_no")

    app_df = build_features_from_inputs(
        scores_last3=stu_k3["score"].tolist(),
        attendance_last3=stu_k3["attendance_pct"].tolist(),
        completion_last3=stu_k3["assignment_completion_pct"].tolist(),
        late_submission_last3=stu_k3["late_submission_pct"].tolist(),
        grade_level=int(stu_k3["grade_level"].iloc[0]),
    )
    train_row = train_df[
        (train_df["student_id"] == "STU_0001") & (train_df["checkpoint_no"] == 3)
    ][FEATURE_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(app_df, train_row, check_exact=False, atol=1e-9)

