"""Tests for reproducible synthetic data generation and non-leaking cleaning."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEED
from src.data_generator import generate_dataset
from src.data_quality import clean_data, report_data_quality


def test_generator_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    """Running generate_dataset twice with the same seed must produce identical DataFrames."""
    path_1 = tmp_path / "run_1.csv"
    path_2 = tmp_path / "run_2.csv"

    df_1 = generate_dataset(output_path=path_1, seed=SEED)
    df_2 = generate_dataset(output_path=path_2, seed=SEED)

    pd.testing.assert_frame_equal(df_1, df_2)
    assert path_1.read_text(encoding="utf-8") == path_2.read_text(encoding="utf-8")


def test_cleaning_converts_impossible_values_to_nan_without_imputing() -> None:
    """clean_data must drop exact duplicates, convert out-of-range values to NaN, and not impute."""
    raw = pd.DataFrame(
        [
            {
                "student_id": "STU_0001",
                "grade_level": 8,
                "gender": "Female",
                "assessment_no": 1,
                "assessment_date": "2025-09-01",
                "score": -12.0,  # impossible (< 0) -> NaN
                "attendance_pct": 112.0,  # impossible (> 100) -> NaN
                "assignment_completion_pct": np.nan,  # pre-existing missing -> stays NaN
                "late_submission_pct": 105.0,  # impossible (> 100) -> NaN
            },
            {
                "student_id": "STU_0001",
                "grade_level": 8,
                "gender": "Female",
                "assessment_no": 1,
                "assessment_date": "2025-09-01",
                "score": -12.0,
                "attendance_pct": 112.0,
                "assignment_completion_pct": np.nan,
                "late_submission_pct": 105.0,
            },
            {
                "student_id": "STU_0001",
                "grade_level": 8,
                "gender": "Female",
                "assessment_no": 2,
                "assessment_date": "2025-09-18",
                "score": 68.5,
                "attendance_pct": 90.0,
                "assignment_completion_pct": 85.0,
                "late_submission_pct": 5.0,
            },
        ]
    )

    cleaned = clean_data(raw)
    report = report_data_quality(cleaned)

    assert len(cleaned) == 2
    assert report["duplicates"]["exact_duplicate_rows"] == 0
    assert report["out_of_range"]["score"]["total_out_of_range"] == 0
    assert report["out_of_range"]["attendance_pct"]["total_out_of_range"] == 0
    assert report["out_of_range"]["late_submission_pct"]["total_out_of_range"] == 0

    row_1 = cleaned.loc[cleaned["assessment_no"] == 1].iloc[0]
    assert pd.isna(row_1["score"])
    assert pd.isna(row_1["attendance_pct"])
    assert pd.isna(row_1["assignment_completion_pct"])
    assert pd.isna(row_1["late_submission_pct"])

