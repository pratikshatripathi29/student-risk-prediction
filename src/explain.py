"""Global and single-student model explainability utilities.

What this module provides:
1. `global_importance()`: computes model-agnostic permutation importance on the
   validation set (`val.csv`, scored by PR-AUC drop) and saves a figure to
   `reports/figures/feature_importance.png`.
2. `explain_prediction(model, row)`: compares a single student's features against
   the training-set medians learned by the pipeline's `SimpleImputer`, returning
   the top 3 risk-increasing factors and top 3 risk-reducing factors in plain English.

Run from the project root:
    python -m src.explain
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    MODEL_PATH,
    SEED,
    VAL_DATA_PATH,
)
from src.features import FEATURE_COLUMNS, TARGET_COLUMN

# Human-readable labels for plots and plain-English explanations.
FEATURE_LABELS: dict[str, str] = {
    "last_score": "Most recent score",
    "mean_score_last3": "Average score (last 3 assessments)",
    "score_std_last3": "Score variability (last 3)",
    "min_score_last3": "Lowest score (last 3 assessments)",
    "score_slope_last3": "Score trend (last 3 assessments)",
    "attendance_last": "Most recent attendance (%)",
    "attendance_mean_last3": "Average attendance (last 3)",
    "attendance_change": "Attendance change (last vs. previous)",
    "completion_last": "Most recent assignment completion (%)",
    "completion_mean_last3": "Average assignment completion (last 3)",
    "completion_change": "Completion change (last vs. previous)",
    "late_submission_last": "Most recent late submission (%)",
    "late_submission_mean_last3": "Average late submission (last 3)",
    "grade_level": "Grade level",
}


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a horizontal bar chart of validation permutation importance."""
    ordered = importance_df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4C78A8" if val > 0 else "#BAB0AC" for val in ordered["importance_mean"]]
    ax.barh(
        ordered["label"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=colors,
        ecolor="#333333",
        capsize=3,
    )
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Global Feature Importance (Validation PR-AUC Drop When Shuffled)")
    ax.set_xlabel("Mean Decrease in Validation PR-AUC")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def global_importance(
    model: Pipeline | None = None,
    val_df: pd.DataFrame | None = None,
    model_path: Path = MODEL_PATH,
    val_path: Path = VAL_DATA_PATH,
    importance_csv_path: Path = FEATURE_IMPORTANCE_PATH,
    figure_path: Path = FIGURES_DIR / "feature_importance.png",
    seed: int = SEED,
    n_repeats: int = 15,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Compute permutation importance on val.csv and save the feature importance chart."""
    if model is None:
        model = joblib.load(model_path)
    if val_df is None:
        val_df = pd.read_csv(val_path)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "label": [FEATURE_LABELS.get(col, col) for col in FEATURE_COLUMNS],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    if save_outputs:
        importance_csv_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_csv_path, index=False)
        plot_feature_importance(importance_df, figure_path)

    return importance_df


def _to_single_row_df(row: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normalize a 1-row DataFrame or Series into a 1-row feature DataFrame."""
    if isinstance(row, pd.Series):
        return row[FEATURE_COLUMNS].to_frame().T.astype(float)
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError(f"Expected exactly 1 row for explanation, got {len(row)}.")
        return row[FEATURE_COLUMNS].copy().reset_index(drop=True)
    raise TypeError(f"row must be a pandas Series or 1-row DataFrame, got {type(row).__name__}.")


def explain_prediction(
    model: Pipeline,
    row: pd.DataFrame | pd.Series,
    top_n: int = 3,
) -> dict[str, Any]:
    """Explain one prediction by replacing each feature with its training median.

    For each feature:
        impact = p(student) - p(student with that feature set to training median)
    - Positive impact -> feature pushes predicted risk UP (risk factor).
    - Negative impact -> feature pulls predicted risk DOWN (protective factor).

    Returns the top `top_n` factors increasing risk and top `top_n` reducing it,
    each with a plain-English sentence.
    """
    row_df = _to_single_row_df(row)
    base_proba = float(model.predict_proba(row_df)[:, 1][0])

    imputer = model.named_steps.get("imputer")
    if imputer is None or not hasattr(imputer, "statistics_"):
        raise ValueError("Model Pipeline must include a fitted 'imputer' step with statistics_.")

    train_medians = imputer.statistics_

    # Create a 14-row batch where row j replaces feature j with its training median.
    perturbed = pd.concat([row_df] * len(FEATURE_COLUMNS), ignore_index=True)
    for idx, col in enumerate(FEATURE_COLUMNS):
        perturbed.loc[idx, col] = float(train_medians[idx])

    perturbed_probas = model.predict_proba(perturbed)[:, 1]

    increasing: list[dict[str, Any]] = []
    reducing: list[dict[str, Any]] = []

    for idx, col in enumerate(FEATURE_COLUMNS):
        raw_val = row_df.iloc[0][col]
        median_val = float(train_medians[idx])
        label = FEATURE_LABELS.get(col, col)
        impact = base_proba - float(perturbed_probas[idx])
        pct_shift = abs(impact) * 100.0

        # Ignore tiny numerical noise (~0) or missing inputs (which already equal the median).
        if pd.isna(raw_val) or abs(impact) < 1e-4:
            continue

        student_val = round(float(raw_val), 2)
        median_rounded = round(median_val, 2)

        if impact > 0:
            text = (
                f"{label} is {student_val} (vs. typical median {median_rounded}), "
                f"increasing predicted risk by +{pct_shift:.1f} percentage points."
            )
            increasing.append(
                {
                    "feature": col,
                    "label": label,
                    "student_value": student_val,
                    "train_median": median_rounded,
                    "risk_impact": round(impact, 4),
                    "explanation": text,
                }
            )
        else:
            text = (
                f"{label} is {student_val} (vs. typical median {median_rounded}), "
                f"reducing predicted risk by {pct_shift:.1f} percentage points."
            )
            reducing.append(
                {
                    "feature": col,
                    "label": label,
                    "student_value": student_val,
                    "train_median": median_rounded,
                    "risk_impact": round(impact, 4),
                    "explanation": text,
                }
            )

    increasing.sort(key=lambda item: float(item["risk_impact"]), reverse=True)
    reducing.sort(key=lambda item: float(item["risk_impact"]))

    return {
        "predicted_probability": round(base_proba, 4),
        "increasing_risk": increasing[:top_n],
        "reducing_risk": reducing[:top_n],
    }


def main() -> None:
    """Run global permutation importance and print a sample local explanation."""
    model: Pipeline = joblib.load(MODEL_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    importance_df = global_importance(model=model, val_df=val_df, save_outputs=True)
    print("=== Global Permutation Importance (Validation Set, PR-AUC Drop) ===")
    print(importance_df.round(4).to_string(index=False))

    # Show one local explanation for a borderline/at-risk student on validation.
    val_probas = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    sample_idx = int(np.argsort(np.abs(val_probas - 0.65))[0])
    sample_row = val_df.iloc[sample_idx]
    local_exp = explain_prediction(model, sample_row, top_n=3)

    print(
        f"\n=== Sample Local Explanation (Student {sample_row['student_id']}, "
        f"Predicted Risk = {local_exp['predicted_probability']:.1%}) ==="
    )
    print("Top factors increasing risk:")
    for item in local_exp["increasing_risk"] or [{"explanation": "None above median risk."}]:
        print(f"  + {item['explanation']}")
    print("Top factors reducing risk:")
    for item in local_exp["reducing_risk"] or [{"explanation": "None below median risk."}]:
        print(f"  - {item['explanation']}")

    print(f"\nSaved table:  {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved figure: {FIGURES_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
