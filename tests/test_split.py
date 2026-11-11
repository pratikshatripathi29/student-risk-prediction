"""Tests for chronological train / validation / test splitting."""

import pandas as pd

from src.config import TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH
from src.split import split_by_checkpoint


def test_chronological_split_ordering_on_synthetic_data() -> None:
    """Max checkpoint in train must be strictly less than validation, which is < test."""
    df = pd.DataFrame(
        [
            {"student_id": "STU_0001", "checkpoint_no": k, "low_next_score": 0}
            for k in range(3, 8)
        ]
    )
    train_df, val_df, test_df = split_by_checkpoint(df)

    assert train_df["checkpoint_no"].max() < val_df["checkpoint_no"].min()
    assert val_df["checkpoint_no"].max() < test_df["checkpoint_no"].min()


def test_saved_splits_preserve_strict_chronological_order() -> None:
    """Saved train.csv, val.csv, and test.csv must enforce train < val < test checkpoints."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert int(train_df["checkpoint_no"].max()) == 5
    assert int(val_df["checkpoint_no"].min()) == 6
    assert int(val_df["checkpoint_no"].max()) == 6
    assert int(test_df["checkpoint_no"].min()) == 7
    assert int(test_df["checkpoint_no"].max()) == 7

    assert train_df["checkpoint_no"].max() < val_df["checkpoint_no"].min()
    assert val_df["checkpoint_no"].max() < test_df["checkpoint_no"].min()

