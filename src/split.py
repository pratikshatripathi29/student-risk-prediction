"""Chronological train / validation / test split by assessment checkpoint.

Why chronological splitting instead of random splitting?
1. Prevents future-to-past temporal leakage:
   In real school use, we train on earlier checkpoints (assessments 3, 4, 5)
   to predict future checkpoints (assessment 6 in validation, assessment 7 in
   final test). A random split would mix checkpoint 7 rows into training and
   checkpoint 3 rows into test—letting the model learn future term shifts and
   exam difficulties before predicting the past.
2. Reflects realistic deployment:
   Every split evaluates forward in time on unseen future outcomes
   (train targets = assessments 4-6; val target = assessment 7;
   test target = assessment 8).
3. Keeps test data isolated:
   Per PROJECT_CONTEXT.md, test.csv is touched only once at final evaluation.

Run from the project root:
    python -m src.split
"""

from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from src.features import TARGET_COLUMN, build_features

TRAIN_CHECKPOINTS: tuple[int, ...] = (3, 4, 5)
VAL_CHECKPOINTS: tuple[int, ...] = (6,)
TEST_CHECKPOINTS: tuple[int, ...] = (7,)


def split_by_checkpoint(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split checkpoint features chronologically into train (3,4,5), val (6), and test (7)."""
    train_df = (
        features_df[features_df["checkpoint_no"].isin(TRAIN_CHECKPOINTS)]
        .copy()
        .reset_index(drop=True)
    )
    val_df = (
        features_df[features_df["checkpoint_no"].isin(VAL_CHECKPOINTS)]
        .copy()
        .reset_index(drop=True)
    )
    test_df = (
        features_df[features_df["checkpoint_no"].isin(TEST_CHECKPOINTS)]
        .copy()
        .reset_index(drop=True)
    )
    return train_df, val_df, test_df


def _print_split_summary(name: str, df: pd.DataFrame, checkpoints: tuple[int, ...]) -> None:
    """Print row count, student count, and target class balance for one split."""
    n_rows = len(df)
    n_pos = int(df[TARGET_COLUMN].sum()) if n_rows else 0
    n_neg = n_rows - n_pos
    pos_pct = (n_pos / n_rows * 100) if n_rows else 0.0

    print(
        f"{name:<11} | k={list(checkpoints)} | rows={n_rows:5d} | "
        f"students={df['student_id'].nunique():4d} | "
        f"at-risk (1)={n_pos:4d} ({pos_pct:5.2f}%) | "
        f"not at-risk (0)={n_neg:4d} ({100 - pos_pct:5.2f}%)"
    )


def run_split(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    train_path: Path = TRAIN_DATA_PATH,
    val_path: Path = VAL_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data, build leak-free checkpoint features, split, and save CSVs."""
    raw_df = pd.read_csv(raw_path)
    features_df = build_features(raw_df)

    # Save the full checkpoint dataset as well as the chronological splits.
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_path, index=False)

    train_df, val_df, test_df = split_by_checkpoint(features_df)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=== Chronological Checkpoint Split Summary ===")
    print(f"Total checkpoint rows built: {len(features_df)} (from {raw_path})")
    print()
    _print_split_summary("Train", train_df, TRAIN_CHECKPOINTS)
    _print_split_summary("Validation", val_df, VAL_CHECKPOINTS)
    _print_split_summary("Test", test_df, TEST_CHECKPOINTS)
    print("\nSaved files:")
    print(f"  - Full features: {processed_path}")
    print(f"  - Train split:   {train_path}")
    print(f"  - Val split:     {val_path}")
    print(f"  - Test split:    {test_path}")

    return train_df, val_df, test_df


def main() -> None:
    """Entry point for `python -m src.split`."""
    run_split()


if __name__ == "__main__":
    main()
