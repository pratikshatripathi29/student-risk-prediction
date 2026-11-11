"""One-time final evaluation on the held-out test split (test.csv).

Rules enforced from PROJECT_CONTEXT.md and docs/problem_definition.md:
- Evaluate on `test.csv` ONCE using the model and thresholds already selected
  on `val.csv` (never retune hyperparameters or thresholds on `test.csv`).
- Save quantitative results to `reports/metrics.json` and `reports/metrics_table.csv`.
- Save diagnostic charts to `reports/figures/`:
    1. `test_confusion_matrix.png`
    2. `test_precision_recall_curve.png`
    3. `test_model_comparison.png`
- Perform error analysis (FP/FN examples, FN vs. TP feature means) and
  break down performance by `grade_level` (plus post-hoc `gender` fairness audit).

Run from the project root:
    python -m src.evaluate
"""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR,
    LOW_SCORE_THRESHOLD,
    METRICS_PATH,
    METRICS_TABLE_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SEED,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.train import compute_metrics, make_pipeline


def _flatten_metrics_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics + 2x2 confusion matrix counts into one clean table row."""
    cm = metrics["confusion_matrix"]
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    return {
        "model": name,
        "threshold": float(metrics["threshold"]),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "f1": round(float(metrics["f1"]), 4),
        "f2": round(float(metrics["f2"]), 4),
        "pr_auc": round(float(metrics["pr_auc"]), 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_all_models_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    final_pipeline: Pipeline,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate baselines, validation-tuned candidates, and the saved final model on test.csv."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    rows: list[dict[str, Any]] = []
    test_probas: dict[str, np.ndarray] = {}

    # 1. Baseline A: DummyClassifier(strategy="prior") fitted on train.csv.
    dummy = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = compute_metrics(y_test, dummy_proba, threshold=0.50)
    rows.append(_flatten_metrics_row("Baseline A: Dummy (prior)", dummy_metrics))
    test_probas["Baseline A: Dummy (prior)"] = dummy_proba

    # 2. Baseline B: Simple rule (last_score < 50).
    rule_pred = (X_test["last_score"].fillna(LOW_SCORE_THRESHOLD) < LOW_SCORE_THRESHOLD).astype(float)
    rule_proba = rule_pred.to_numpy()
    rule_metrics = compute_metrics(y_test, rule_proba, threshold=0.50)
    rows.append(_flatten_metrics_row("Baseline B: Rule (last_score < 50)", rule_metrics))
    test_probas["Baseline B: Rule (last_score < 50)"] = rule_proba

    # 3. Candidate families using exact validation-selected params & thresholds (never retuned).
    winner_family = str(metadata["model_family"])
    seed = int(metadata.get("seed", SEED))
    val_entries = metadata["validation_comparison_tuned_f2"]

    for entry in val_entries:
        family = entry.get("family")
        if not family:
            continue
        frozen_threshold = float(entry["threshold"])

        if family == winner_family:
            # Evaluate the saved Pipeline artifact loaded from models/model.joblib.
            proba = final_pipeline.predict_proba(X_test)[:, 1]
            label = f"{family} (final saved)"
        else:
            # Refit competing family on train.csv with its validation-chosen hyperparameters.
            pipe = make_pipeline(family, entry["params"], seed=seed)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            label = f"{family} (val-tuned)"

        metrics = compute_metrics(y_test, proba, threshold=frozen_threshold)
        rows.append(_flatten_metrics_row(label, metrics))
        test_probas[label] = proba

    return rows, test_probas


def plot_confusion_matrix_figure(
    cm_counts: list[list[int]],
    model_name: str,
    threshold: float,
    output_path: Path,
) -> None:
    """Save an annotated 2x2 confusion matrix heatmap for the final model on test.csv."""
    cm = np.array(cm_counts)
    total = cm.sum()
    cell_names = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
    annotations = np.empty_like(cm, dtype=object)
    for r in range(2):
        for c in range(2):
            pct = cm[r, c] / total * 100 if total else 0.0
            annotations[r, c] = f"{cell_names[r][c]}\n{cm[r, c]:,} ({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: Not At-Risk (0)", "Predicted: At-Risk (1)"],
        yticklabels=["Actual: Not At-Risk (0)", "Actual: At-Risk (1)"],
        ax=ax,
    )
    ax.set_title(f"Test Confusion Matrix — {model_name}\n(Threshold = {threshold:.2f})")
    ax.set_xlabel("Model Prediction")
    ax.set_ylabel("Actual Next Assessment Outcome")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_figure(
    y_test: pd.Series,
    test_probas: dict[str, np.ndarray],
    final_label: str,
    final_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Save Precision-Recall curves comparing models on test.csv with operating point marked."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    prevalence = float(y_test.mean())
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"No-Skill Prevalence ({prevalence:.3f})",
    )

    palette = {
        "LogisticRegression": "#4C78A8",
        "RandomForestClassifier": "#F58518",
        "HistGradientBoostingClassifier": "#E45756",
    }

    for label, proba in test_probas.items():
        if label.startswith("Baseline"):
            continue
        prec, rec, _ = precision_recall_curve(y_test, proba)
        color = next((c for key, c in palette.items() if key in label), None)
        lw = 2.6 if label == final_label else 1.7
        alpha = 1.0 if label == final_label else 0.8
        ax.plot(rec, prec, label=label, linewidth=lw, alpha=alpha, color=color)

    # Mark the final model's chosen validation threshold operating point.
    ax.scatter(
        [final_metrics["recall"]],
        [final_metrics["precision"]],
        color="#E45756",
        edgecolor="black",
        s=90,
        zorder=5,
        label=(
            f"Chosen Threshold ({final_metrics['threshold']:.2f}): "
            f"R={final_metrics['recall']:.2f}, P={final_metrics['precision']:.2f}"
        ),
    )

    ax.set_title("Test Precision-Recall Curves (Held-Out Checkpoint k=7)")
    ax.set_xlabel("Recall (Share of Struggling Students Caught)")
    ax.set_ylabel("Precision (Share of Alerts That Truly Struggle)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_figure(
    metrics_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing Recall, Precision, F1, and PR-AUC on test.csv."""
    df_plot = pd.DataFrame(metrics_rows)
    short_names = [
        name.replace("Baseline A: ", "")
        .replace("Baseline B: ", "")
        .replace(" (final saved)", "\n(final)")
        .replace(" (val-tuned)", "\n(tuned)")
        for name in df_plot["model"]
    ]

    metrics_to_plot = ["recall", "precision", "f1", "pr_auc"]
    metric_labels = ["Recall", "Precision", "F1", "PR-AUC"]
    colors = ["#E45756", "#4C78A8", "#72B7B2", "#54A24B"]

    x = np.arange(len(short_names))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for idx, (col, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, df_plot[col], width=width, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("Test Set Comparison Across Baselines and Candidate Models")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_ylim(0.0, 1.12)
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_subgroup_breakdown(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
    group_col: str,
) -> list[dict[str, Any]]:
    """Compute test performance broken down by a subgroup column (e.g., grade_level or gender)."""
    rows: list[dict[str, Any]] = []
    for group_val, idx in test_df.groupby(group_col, sort=True).groups.items():
        y_sub = test_df.loc[idx, TARGET_COLUMN]
        proba_sub = y_proba[list(idx)]
        m = compute_metrics(y_sub, proba_sub, threshold=threshold)
        cm = m["confusion_matrix"]
        rows.append(
            {
                group_col: group_val if not isinstance(group_val, np.integer) else int(group_val),
                "n_students": int(len(y_sub)),
                "at_risk_count": int(y_sub.sum()),
                "at_risk_pct": round(float(y_sub.mean()) * 100, 2),
                "precision": round(float(m["precision"]), 4),
                "recall": round(float(m["recall"]), 4),
                "f1": round(float(m["f1"]), 4),
                "pr_auc": round(float(m["pr_auc"]), 4),
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            }
        )
    return rows


def perform_error_analysis(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
    n_examples: int = 5,
) -> dict[str, Any]:
    """Inspect False Positives, False Negatives, and FN-vs-TP average feature profiles."""
    analysis_df = test_df.copy()
    analysis_df["pred_proba"] = np.round(y_proba, 4)
    analysis_df["pred_label"] = (y_proba >= threshold).astype(int)

    tp_df = analysis_df[(analysis_df[TARGET_COLUMN] == 1) & (analysis_df["pred_label"] == 1)]
    fn_df = analysis_df[(analysis_df[TARGET_COLUMN] == 1) & (analysis_df["pred_label"] == 0)]
    fp_df = analysis_df[(analysis_df[TARGET_COLUMN] == 0) & (analysis_df["pred_label"] == 1)]

    key_cols = [
        "student_id",
        "grade_level",
        "last_score",
        "mean_score_last3",
        "score_slope_last3",
        "attendance_last",
        "completion_last",
        "late_submission_last",
        "pred_proba",
        TARGET_COLUMN,
    ]

    # Pick representative examples: highest-confidence FPs and lowest-probability FNs.
    fp_examples = (
        fp_df.sort_values("pred_proba", ascending=False)[key_cols]
        .head(n_examples)
        .round(2)
        .to_dict(orient="records")
    )
    fn_examples = (
        fn_df.sort_values("pred_proba", ascending=True)[key_cols]
        .head(n_examples)
        .round(2)
        .to_dict(orient="records")
    )

    compare_cols = [
        "last_score",
        "mean_score_last3",
        "min_score_last3",
        "score_slope_last3",
        "attendance_last",
        "attendance_mean_last3",
        "completion_last",
        "completion_mean_last3",
        "late_submission_last",
        "pred_proba",
    ]
    fn_vs_tp_means: list[dict[str, Any]] = []
    for col in compare_cols:
        fn_mean = float(fn_df[col].mean()) if len(fn_df) > 0 else float("nan")
        tp_mean = float(tp_df[col].mean()) if len(tp_df) > 0 else float("nan")
        fn_vs_tp_means.append(
            {
                "feature": col,
                "fn_mean": round(fn_mean, 2),
                "tp_mean": round(tp_mean, 2),
                "diff_fn_minus_tp": round(fn_mean - tp_mean, 2),
            }
        )

    school_interpretation = {
        "false_positive": (
            "A student who would actually score >= 50 on the next assessment is flagged "
            "as needing support. For a school, this means offering optional extra help "
            "(such as a quick check-in or study resource) to a student who did not strictly need it—low cost."
        ),
        "false_negative": (
            "A student who ends up scoring < 50 on the next assessment is NOT flagged. "
            "For a school, a struggling student is overlooked and misses early intervention—high cost."
        ),
    }

    return {
        "counts": {
            "tp": int(len(tp_df)),
            "fn": int(len(fn_df)),
            "fp": int(len(fp_df)),
        },
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
        "fn_vs_tp_feature_means": fn_vs_tp_means,
        "school_interpretation": school_interpretation,
    }


def run_evaluation(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    metrics_json_path: Path = METRICS_PATH,
    metrics_csv_path: Path = METRICS_TABLE_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    """Run one-time held-out evaluation on test.csv and save tables, figures, and insights."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    final_pipeline: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))

    winner_family = str(metadata["model_family"])
    frozen_threshold = float(metadata["decision_threshold"])
    final_label = f"{winner_family} (final saved)"

    metrics_rows, test_probas = evaluate_all_models_on_test(
        train_df=train_df,
        test_df=test_df,
        final_pipeline=final_pipeline,
        metadata=metadata,
    )

    final_row = next(r for r in metrics_rows if r["model"] == final_label)
    final_proba = test_probas[final_label]
    final_cm = [[final_row["tn"], final_row["fp"]], [final_row["fn"], final_row["tp"]]]

    # Save tabular CSV and JSON metrics.
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save figures to reports/figures/.
    cm_fig_path = figures_dir / "test_confusion_matrix.png"
    pr_fig_path = figures_dir / "test_precision_recall_curve.png"
    comp_fig_path = figures_dir / "test_model_comparison.png"

    plot_confusion_matrix_figure(final_cm, winner_family, frozen_threshold, cm_fig_path)
    plot_precision_recall_figure(test_df[TARGET_COLUMN], test_probas, final_label, final_row, pr_fig_path)
    plot_model_comparison_figure(metrics_rows, comp_fig_path)

    # Subgroup breakdowns + error analysis.
    grade_breakdown = compute_subgroup_breakdown(
        test_df=test_df,
        y_proba=final_proba,
        threshold=frozen_threshold,
        group_col="grade_level",
    )
    gender_audit = (
        compute_subgroup_breakdown(
            test_df=test_df,
            y_proba=final_proba,
            threshold=frozen_threshold,
            group_col="gender",
        )
        if "gender" in test_df.columns
        else []
    )
    error_analysis = perform_error_analysis(
        test_df=test_df,
        y_proba=final_proba,
        threshold=frozen_threshold,
    )

    report_payload: dict[str, Any] = {
        "test_rows": int(len(test_df)),
        "test_at_risk_rate": round(float(test_df[TARGET_COLUMN].mean()), 4),
        "final_model": winner_family,
        "frozen_threshold_from_val": frozen_threshold,
        "models_test_metrics": metrics_rows,
        "grade_level_breakdown": grade_breakdown,
        "gender_bias_audit_posthoc": gender_audit,
        "error_analysis": error_analysis,
        "saved_figures": {
            "confusion_matrix": str(cm_fig_path),
            "precision_recall_curve": str(pr_fig_path),
            "model_comparison": str(comp_fig_path),
        },
    }
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    # Print clear console summary.
    print("=== Final One-Time Test Evaluation (test.csv, k=7 -> Assessment 8) ===")
    print(f"Test rows: {len(test_df)} | At-risk rate: {test_df[TARGET_COLUMN].mean() * 100:.2f}%")
    print(f"Final model: {winner_family} | Frozen threshold from validation: {frozen_threshold:.2f}\n")
    print(metrics_df.to_string(index=False))

    print("\n=== Performance by Grade Level ===")
    print(pd.DataFrame(grade_breakdown).to_string(index=False))

    if gender_audit:
        print("\n=== Post-Hoc Fairness Audit by Gender (Never Used as Model Feature) ===")
        print(pd.DataFrame(gender_audit).to_string(index=False))

    print("\n=== Error Analysis: Average Features of Missed (FN) vs. Caught (TP) Struggling Students ===")
    print(pd.DataFrame(error_analysis["fn_vs_tp_feature_means"]).to_string(index=False))

    print("\n--- Example False Negatives (Struggling Students Missed by Model) ---")
    print(pd.DataFrame(error_analysis["fn_examples"]).to_string(index=False))

    print("\n--- Example False Positives (Students Flagged Who Scored >= 50) ---")
    print(pd.DataFrame(error_analysis["fp_examples"]).to_string(index=False))

    print("\n--- What Errors Mean for a School ---")
    print(f"False Positive (FP): {error_analysis['school_interpretation']['false_positive']}")
    print(f"False Negative (FN): {error_analysis['school_interpretation']['false_negative']}")

    print("\nSaved reports & figures:")
    print(f"  - {metrics_json_path}")
    print(f"  - {metrics_csv_path}")
    print(f"  - {cm_fig_path}")
    print(f"  - {pr_fig_path}")
    print(f"  - {comp_fig_path}")

    return report_payload


def main() -> None:
    """Entry point for `python -m src.evaluate`."""
    run_evaluation()


if __name__ == "__main__":
    main()
