"""Data quality auditing and non-leaking cleaning utilities.

Why keep cleaning separate from imputation?
- Removing exact duplicate rows and flagging physically impossible values
  (such as negative test scores or percentages over 100%) are deterministic,
  row-level domain rules that do not inspect dataset-wide statistics.
- Filling missing values (imputation) DOES compute dataset statistics
  (like median or mean), so imputation must happen later inside an sklearn
  Pipeline fitted on training data only to prevent data leakage.

Run a quick audit-and-clean check on the raw dataset:
    python -m src.data_quality
"""

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH

# Numeric columns whose valid values must lie within [0, 100].
BOUNDED_PERCENT_COLUMNS: tuple[str, ...] = (
    "score",
    "attendance_pct",
    "assignment_completion_pct",
    "late_submission_pct",
)


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return the count of missing (NaN) values for every column."""
    return {col: int(df[col].isna().sum()) for col in df.columns}


def check_duplicates(df: pd.DataFrame) -> dict[str, int]:
    """Return counts of exact duplicate rows and duplicate (student, assessment) keys."""
    exact_dups = int(df.duplicated().sum())
    key_cols = ["student_id", "assessment_no"]
    if all(col in df.columns for col in key_cols):
        key_dups = int(df.duplicated(subset=key_cols).sum())
    else:
        key_dups = 0

    return {
        "exact_duplicate_rows": exact_dups,
        "duplicate_student_assessment_keys": key_dups,
    }


def check_out_of_range_values(
    df: pd.DataFrame,
    columns: tuple[str, ...] = BOUNDED_PERCENT_COLUMNS,
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> dict[str, dict[str, int]]:
    """Count values outside [min_val, max_val] for each bounded numeric column."""
    results: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        below = int((series < min_val).sum())
        above = int((series > max_val).sum())
        results[col] = {
            "below_min": below,
            "above_max": above,
            "total_out_of_range": below + above,
        }
    return results


def report_data_quality(df: pd.DataFrame) -> dict[str, object]:
    """Build a structured quality report covering shape, missingness, duplicates, and ranges."""
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_students": int(df["student_id"].nunique()) if "student_id" in df.columns else 0,
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df),
        "out_of_range": check_out_of_range_values(df),
    }


def print_quality_report(
    report: dict[str, object],
    title: str = "Data Quality Report",
) -> None:
    """Print a beginner-friendly summary of the dictionary returned by report_data_quality."""
    n_rows = int(report["n_rows"])
    print(f"=== {title} ===")
    print(f"Rows: {n_rows} | Columns: {report['n_columns']} | Unique students: {report['n_students']}")

    print("\nMissing values:")
    missing_values: dict[str, int] = report["missing_values"]  # type: ignore[assignment]
    for col, count in missing_values.items():
        if count > 0:
            pct = (count / n_rows * 100) if n_rows else 0.0
            print(f"  - {col}: {count} ({pct:.2f}%)")
    if all(count == 0 for count in missing_values.values()):
        print("  - None")

    duplicates: dict[str, int] = report["duplicates"]  # type: ignore[assignment]
    print("\nDuplicates:")
    print(f"  - Exact duplicate rows: {duplicates['exact_duplicate_rows']}")
    print(f"  - Duplicate (student_id, assessment_no) keys: {duplicates['duplicate_student_assessment_keys']}")

    out_of_range: dict[str, dict[str, int]] = report["out_of_range"]  # type: ignore[assignment]
    print("\nOut-of-range values (valid range: 0 to 100):")
    any_oob = False
    for col, stats in out_of_range.items():
        if stats["total_out_of_range"] > 0:
            any_oob = True
            print(
                f"  - {col}: {stats['total_out_of_range']} total "
                f"(< 0: {stats['below_min']}, > 100: {stats['above_max']})"
            )
    if not any_oob:
        print("  - None")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and convert impossible values to NaN.

    Important: This function intentionally does NOT impute missing values.
    Imputation must happen inside the scikit-learn Pipeline on training data only.
    """
    cleaned = df.copy()

    # 1. Remove exact duplicate rows.
    cleaned = cleaned.drop_duplicates()

    # 2. Convert impossible values outside [0, 100] to NaN so they are not treated as real numbers.
    for col in BOUNDED_PERCENT_COLUMNS:
        if col in cleaned.columns:
            invalid_mask = (cleaned[col] < 0.0) | (cleaned[col] > 100.0)
            cleaned.loc[invalid_mask, col] = np.nan

    # 3. Keep rows in predictable student-and-assessment order.
    sort_cols = [c for c in ("student_id", "assessment_no") if c in cleaned.columns]
    if sort_cols:
        cleaned = cleaned.sort_values(sort_cols, kind="mergesort")

    return cleaned.reset_index(drop=True)


def main() -> None:
    """Run quality reporting before and after cleaning on the raw CSV."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    before_report = report_data_quality(raw_df)
    print_quality_report(before_report, title="Before Cleaning (Raw Data)")

    print()
    cleaned_df = clean_data(raw_df)
    after_report = report_data_quality(cleaned_df)
    print_quality_report(after_report, title="After Cleaning (No Imputation Yet)")


if __name__ == "__main__":
    main()
