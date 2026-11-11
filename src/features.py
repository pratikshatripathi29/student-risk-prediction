"""Leak-free checkpoint feature engineering for training and inference.

Design principle (training-serving consistency):
- Both offline dataset creation (`build_features`) and real-time single-student
  prediction in the Streamlit app (`build_features_from_inputs`) call the same
  helper `_compute_checkpoint_features`.
- Every feature at checkpoint `k` (k = 3..7) is computed strictly from
  assessments <= k. No future assessments (> k) are ever read into features.
- `gender` is never included in `FEATURE_COLUMNS` (audit-only per PROJECT_CONTEXT.md).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import LOW_SCORE_THRESHOLD, MAX_CHECKPOINT, MIN_CHECKPOINT
from src.data_quality import clean_data

# Exact ordered list of model input features.
# Keep student_id, checkpoint_no, gender, and low_next_score OUT of this list.
FEATURE_COLUMNS: list[str] = [
    "last_score",
    "mean_score_last3",
    "score_std_last3",
    "min_score_last3",
    "score_slope_last3",
    "attendance_last",
    "attendance_mean_last3",
    "attendance_change",
    "completion_last",
    "completion_mean_last3",
    "completion_change",
    "late_submission_last",
    "late_submission_mean_last3",
    "grade_level",
]

TARGET_COLUMN: str = "low_next_score"
ID_AND_AUDIT_COLUMNS: list[str] = ["student_id", "checkpoint_no", "gender"]


def _safe_mean(values: Sequence[float]) -> float:
    """Return arithmetic mean ignoring NaNs, or NaN if all entries are missing."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def _safe_std(values: Sequence[float]) -> float:
    """Return sample standard deviation (ddof=1) when >= 2 valid values exist."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return float("nan")
    return float(np.std(valid, ddof=1))


def _safe_min(values: Sequence[float]) -> float:
    """Return minimum ignoring NaNs, or NaN if all entries are missing."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan")
    return float(np.min(valid))


def _safe_linear_slope(values: Sequence[float]) -> float:
    """Compute linear trend (points per assessment) over the last 3 checkpoints.

    Uses ordinary least squares on x = [0, 1, ..., n-1] for valid points so
    missing entries do not fail and spacing stays accurate. Returns NaN if
    fewer than 2 valid scores exist.
    """
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    valid_mask = ~np.isnan(y)

    if valid_mask.sum() < 2:
        return float("nan")

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    x_centered = x_valid - np.mean(x_valid)
    y_centered = y_valid - np.mean(y_valid)
    denominator = float(np.sum(x_centered**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def _safe_change(values: Sequence[float]) -> float:
    """Compute latest value minus immediately preceding value (k minus k-1)."""
    if len(values) < 2:
        return float("nan")
    prev_val = float(values[-2])
    last_val = float(values[-1])
    if np.isnan(prev_val) or np.isnan(last_val):
        return float("nan")
    return last_val - prev_val


def _compute_checkpoint_features(
    scores_last3: Sequence[float],
    attendance_last3: Sequence[float],
    completion_last3: Sequence[float],
    late_submission_last3: Sequence[float],
    grade_level: int,
) -> dict[str, float | int]:
    """Build the single feature dictionary shared by training and app inference.

    All sequences must be ordered chronologically: [k-2, k-1, k].
    """
    return {
        "last_score": float(scores_last3[-1]),
        "mean_score_last3": _safe_mean(scores_last3),
        "score_std_last3": _safe_std(scores_last3),
        "min_score_last3": _safe_min(scores_last3),
        "score_slope_last3": _safe_linear_slope(scores_last3),
        "attendance_last": float(attendance_last3[-1]),
        "attendance_mean_last3": _safe_mean(attendance_last3),
        "attendance_change": _safe_change(attendance_last3),
        "completion_last": float(completion_last3[-1]),
        "completion_mean_last3": _safe_mean(completion_last3),
        "completion_change": _safe_change(completion_last3),
        "late_submission_last": float(late_submission_last3[-1]),
        "late_submission_mean_last3": _safe_mean(late_submission_last3),
        "grade_level": int(grade_level),
    }


def build_features_from_inputs(
    scores_last3: Sequence[float],
    attendance_last3: Sequence[float],
    completion_last3: Sequence[float],
    late_submission_last3: Sequence[float],
    grade_level: int,
) -> pd.DataFrame:
    """Build a 1-row feature DataFrame from interactive app inputs.

    Expects 3 chronological values ([k-2, k-1, k]) for each metric so the
    Streamlit app executes the exact same feature logic used in training.
    """
    for name, seq in [
        ("scores_last3", scores_last3),
        ("attendance_last3", attendance_last3),
        ("completion_last3", completion_last3),
        ("late_submission_last3", late_submission_last3),
    ]:
        if len(seq) != 3:
            raise ValueError(f"{name} must contain exactly 3 values ([k-2, k-1, k]), got {len(seq)}.")

    row = _compute_checkpoint_features(
        scores_last3=scores_last3,
        attendance_last3=attendance_last3,
        completion_last3=completion_last3,
        late_submission_last3=late_submission_last3,
        grade_level=grade_level,
    )
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform cleaned long-format assessment data into checkpoint rows (k = 3..7).

    For each student and each checkpoint k in [MIN_CHECKPOINT, MAX_CHECKPOINT]:
    - Uses ONLY assessments in {k-2, k-1, k} (all <= k) to build features.
    - Reads assessment k+1's score solely to create the label `low_next_score`.
    - Skips checkpoints where assessment k+1's score is missing (NaN), because
      a ground-truth target cannot be assigned without fabricating labels.
    """
    clean_df = clean_data(df)
    has_gender = "gender" in clean_df.columns

    rows: list[dict[str, object]] = []

    for student_id, student_df in clean_df.groupby("student_id", sort=True):
        # Index by assessment_no so lookups strictly enforce <= k and k+1 boundaries.
        indexed = student_df.set_index("assessment_no")
        grade_level = int(student_df["grade_level"].iloc[0])
        gender = student_df["gender"].iloc[0] if has_gender else None

        for k in range(MIN_CHECKPOINT, MAX_CHECKPOINT + 1):
            next_assessment = k + 1
            if next_assessment not in indexed.index:
                continue

            next_score = float(indexed.loc[next_assessment, "score"])
            if np.isnan(next_score):
                # Cannot determine true binary label when assessment k+1 score was missing/invalid.
                continue

            # Only look at assessments up to k (specifically the last 3: k-2, k-1, k).
            window_nos = [k - 2, k - 1, k]
            if not all(no in indexed.index for no in window_nos):
                continue

            window = indexed.loc[window_nos]
            feature_dict = _compute_checkpoint_features(
                scores_last3=window["score"].to_numpy(dtype=float),
                attendance_last3=window["attendance_pct"].to_numpy(dtype=float),
                completion_last3=window["assignment_completion_pct"].to_numpy(dtype=float),
                late_submission_last3=window["late_submission_pct"].to_numpy(dtype=float),
                grade_level=grade_level,
            )

            record: dict[str, object] = {
                "student_id": student_id,
                "checkpoint_no": k,
            }
            if has_gender:
                record["gender"] = gender
            record.update(feature_dict)
            record[TARGET_COLUMN] = int(next_score < LOW_SCORE_THRESHOLD)
            rows.append(record)

    output_cols = ["student_id", "checkpoint_no"]
    if has_gender:
        output_cols.append("gender")
    output_cols.extend(FEATURE_COLUMNS)
    output_cols.append(TARGET_COLUMN)

    return pd.DataFrame(rows, columns=output_cols)
