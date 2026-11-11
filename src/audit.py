"""Post-hoc fairness & bias audit across synthetic gender and grade_level on test.csv.

Ethical rule from PROJECT_CONTEXT.md:
- `gender` is NEVER included in `FEATURE_COLUMNS` or used during training or
  prediction. It is loaded here SOLELY to audit whether the trained model
  produces fair alert rates, recall, and false-positive rates across groups.

What subgroup differences would be concerning in a school setting?
1. Large Recall gap (e.g., > 5-10 percentage points lower for one subgroup):
   Means struggling students in that group are systematically MISSED and do not
   receive early academic support (violates Equal Opportunity).
2. Large False-Positive Rate (FPR) or Flagged-Rate gap (e.g., > 5-10 points
   higher for a subgroup with similar actual at-risk rates):
   Means students in that group are over-flagged despite doing fine, which can
   create stigma, bias expectations, or misallocate support staff time.
3. Small-sample caveat:
   For smaller subgroups (e.g., `Non-binary` with ~69 students and ~8 positives),
   a single student shifts Recall by ~12.5 percentage points. Always inspect
   subgroup sample sizes (`n_students`, `actual_positives`) alongside rates.

Run from the project root:
    python -m src.audit
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import BIAS_AUDIT_PATH, MODEL_METADATA_PATH, MODEL_PATH, TEST_DATA_PATH
from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def compute_group_fairness_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str,
) -> list[dict[str, Any]]:
    """Compute flagged rate, recall, and false-positive rate for each subgroup."""
    rows: list[dict[str, Any]] = []

    for group_value, idx in df.groupby(group_col, sort=True).groups.items():
        y_true_sub = df.loc[idx, TARGET_COLUMN].to_numpy()
        y_pred_sub = y_pred[list(idx)]

        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1]).ravel()
        n_total = int(len(y_true_sub))
        actual_positives = int(tp + fn)
        actual_negatives = int(tn + fp)

        flagged_rate = float(y_pred_sub.mean()) if n_total > 0 else 0.0
        recall = float(tp / actual_positives) if actual_positives > 0 else 0.0
        false_positive_rate = float(fp / actual_negatives) if actual_negatives > 0 else 0.0
        actual_at_risk_rate = float(actual_positives / n_total) if n_total > 0 else 0.0

        rows.append(
            {
                "group_type": group_col,
                "group_value": str(group_value),
                "n_students": n_total,
                "actual_positives": actual_positives,
                "actual_at_risk_rate": round(actual_at_risk_rate, 4),
                "flagged_rate": round(flagged_rate, 4),
                "recall": round(recall, 4),
                "false_positive_rate": round(false_positive_rate, 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return rows


def run_bias_audit(
    test_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    output_path: Path = BIAS_AUDIT_PATH,
) -> pd.DataFrame:
    """Audit the saved model on test.csv across synthetic gender and grade_level."""
    test_df = pd.read_csv(test_path)
    model: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Safety check: verify `gender` was never included in model features.
    model_features = list(metadata.get("feature_columns", FEATURE_COLUMNS))
    if "gender" in model_features or "gender" in FEATURE_COLUMNS:
        raise RuntimeError("Policy violation: `gender` must NEVER be in model features.")

    threshold = float(metadata["decision_threshold"])
    y_proba = model.predict_proba(test_df[model_features])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    audit_rows: list[dict[str, Any]] = []
    for col in ("gender", "grade_level"):
        if col in test_df.columns:
            audit_rows.extend(compute_group_fairness_metrics(test_df, y_pred, group_col=col))

    audit_df = pd.DataFrame(audit_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)

    print("=== Post-Hoc Bias Audit on test.csv ===")
    print(f"Model: {metadata.get('model_family')} | Threshold: {threshold:.2f}")
    print("Note: `gender` is used here ONLY for auditing; it is never a model feature.\n")
    print(audit_df.to_string(index=False))

    # Print quick max-gap check for the larger subgroups (n >= 100) to highlight real gaps.
    for col in ("gender", "grade_level"):
        sub = audit_df[(audit_df["group_type"] == col) & (audit_df["n_students"] >= 100)]
        if len(sub) >= 2:
            recall_gap = float(sub["recall"].max() - sub["recall"].min())
            fpr_gap = float(sub["false_positive_rate"].max() - sub["false_positive_rate"].min())
            flag_gap = float(sub["flagged_rate"].max() - sub["flagged_rate"].min())
            print(
                f"\n{col} gaps (subgroups n >= 100): "
                f"flagged_rate gap={flag_gap:.4f}, recall gap={recall_gap:.4f}, "
                f"false_positive_rate gap={fpr_gap:.4f}"
            )

    print(f"\nSaved audit report to: {output_path}")
    return audit_df


def main() -> None:
    """Entry point for `python -m src.audit`."""
    run_bias_audit()


if __name__ == "__main__":
    main()

