"""Train baseline and candidate models using leak-free scikit-learn Pipelines.

What this module does:
1. Loads ONLY `train.csv` and `val.csv` (`test.csv` is strictly untouched here).
2. Evaluates two simple baselines on validation:
   - Baseline A: `DummyClassifier(strategy="prior")` (predicts no at-risk alerts,
     giving probabilities equal to the training positive rate).
   - Baseline B: Simple rule (`last_score < 50`).
3. Runs a small hyperparameter search for three model families, each wrapped in
   an `sklearn.pipeline.Pipeline` with `SimpleImputer(strategy="median")`:
   - `LogisticRegression(class_weight="balanced")` + `StandardScaler`
   - `RandomForestClassifier(class_weight="balanced")`
   - `HistGradientBoostingClassifier`
4. Scores every candidate on the VALIDATION set only (`val.csv`).
5. Selects the best model by validation performance (PR-AUC, breaking ties by F2),
   tunes its probability decision threshold on `val.csv` to maximize F2
   (recall-weighted), and derives Low / Medium / High risk-band cutoffs from
   validation probabilities.
6. Saves the winning full Pipeline to `models/model.joblib` and metadata
   (feature list, threshold, band cutoffs, seed, sklearn version, training date,
   validation metrics) to `models/metadata.json`.

Run from the project root:
    python -m src.train
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOW_SCORE_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SEED,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute imbalanced classification metrics at a probability threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": round(float(threshold), 4),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> list[dict[str, Any]]:
    """Evaluate Baseline A (Dummy prior) and Baseline B (last_score < 50 rule)."""
    rows: list[dict[str, Any]] = []

    # Baseline A: DummyClassifier predicting class priors from training data.
    dummy = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_val)[:, 1]
    dummy_metrics = compute_metrics(y_val, dummy_proba, threshold=0.50)
    rows.append(
        {
            "name": "Baseline A: Dummy (prior)",
            "params": {"strategy": "prior"},
            **dummy_metrics,
        }
    )

    # Baseline B: Simple rule — flag student if their latest score < 50.
    # If last_score is missing (NaN), the simple rule cannot flag and predicts 0.
    rule_pred = (X_val["last_score"].fillna(LOW_SCORE_THRESHOLD) < LOW_SCORE_THRESHOLD).astype(float)
    rule_metrics = compute_metrics(y_val, rule_pred.to_numpy(), threshold=0.50)
    rows.append(
        {
            "name": "Baseline B: Rule (last_score < 50)",
            "params": {"cutoff": LOW_SCORE_THRESHOLD},
            **rule_metrics,
        }
    )

    return rows


def get_candidate_search_space(seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    """Define a compact, readable hyperparameter search grid per model family."""
    return {
        "LogisticRegression": [
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
        "RandomForestClassifier": [
            {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10},
            {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
            {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20},
            {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 15},
        ],
        "HistGradientBoostingClassifier": [
            {"learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 20, "class_weight": "balanced"},
            {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "class_weight": "balanced"},
            {"learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 30, "class_weight": "balanced"},
            {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "class_weight": None},
        ],
    }


def make_pipeline(model_family: str, params: dict[str, Any], seed: int = SEED) -> Pipeline:
    """Wrap a model inside an sklearn Pipeline with median imputation (and scaling if needed)."""
    if model_family == "LogisticRegression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                        **params,
                    ),
                ),
            ]
        )

    if model_family == "RandomForestClassifier":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        **params,
                    ),
                ),
            ]
        )

    if model_family == "HistGradientBoostingClassifier":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        random_state=seed,
                        **params,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model_family: {model_family}")


def tune_threshold_for_f2(
    y_val: pd.Series | np.ndarray,
    val_proba: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Find the probability threshold on validation that maximizes F2 (recall-weighted)."""
    best_threshold = 0.50
    best_metrics = compute_metrics(y_val, val_proba, threshold=0.50)
    best_f2 = float(best_metrics["f2"])

    for thr in np.round(np.arange(0.15, 0.81, 0.02), 2):
        metrics = compute_metrics(y_val, val_proba, threshold=float(thr))
        if float(metrics["f2"]) > best_f2:
            best_f2 = float(metrics["f2"])
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def compute_risk_band_cutoffs(
    y_val: pd.Series | np.ndarray,
    val_proba: np.ndarray,
    decision_threshold: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Derive Low / Medium / High risk-band cutoffs from validation probabilities.

    Design:
    - Low Risk:    p < low_max   (aligned with `decision_threshold`, so unflagged students are Low)
    - Medium Risk: low_max <= p < medium_max (flagged students with moderate risk probability)
    - High Risk:   p >= medium_max (top half / highest-confidence flagged students)
    """
    low_max = round(float(decision_threshold), 2)

    flagged_probs = val_proba[val_proba >= low_max]
    if len(flagged_probs) >= 4:
        # Place the Medium/High boundary at the median probability among flagged students
        # (ensuring a sensible gap above low_max).
        medium_max = round(float(np.quantile(flagged_probs, 0.55)), 2)
        if medium_max <= low_max + 0.05:
            medium_max = round(min(0.95, low_max + 0.15), 2)
    else:
        medium_max = round(min(0.95, low_max + 0.20), 2)

    cutoffs = {
        "low_max": low_max,
        "medium_max": medium_max,
    }

    y_arr = np.asarray(y_val)
    bands = {
        "Low": val_proba < low_max,
        "Medium": (val_proba >= low_max) & (val_proba < medium_max),
        "High": val_proba >= medium_max,
    }
    band_summary: dict[str, dict[str, Any]] = {}
    for band_name, mask in bands.items():
        count = int(mask.sum())
        actual_rate = float(y_arr[mask].mean()) if count > 0 else 0.0
        band_summary[band_name] = {
            "count": count,
            "share_pct": round(count / len(y_arr) * 100, 2),
            "observed_at_risk_rate": round(actual_rate, 4),
        }

    return cutoffs, band_summary


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    """Print a clean validation comparison table for baselines and candidate models."""
    header = (
        f"{'Model':<38} | {'Thr':>5} | {'Precision':>9} | "
        f"{'Recall':>7} | {'F1':>7} | {'F2':>7} | {'PR-AUC':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<38} | {row['threshold']:>5.2f} | "
            f"{row['precision']:>9.4f} | {row['recall']:>7.4f} | "
            f"{row['f1']:>7.4f} | {row['f2']:>7.4f} | {row['pr_auc']:>7.4f}"
        )


def train_and_select(
    train_path: Path = TRAIN_DATA_PATH,
    val_path: Path = VAL_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    seed: int = SEED,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train baselines and tuned candidates on train.csv; select and save via val.csv."""
    # Load ONLY train and validation splits (test.csv is never touched here).
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    print("=== Training & Validation Model Selection ===")
    print(
        f"Train rows: {len(X_train)} (at-risk: {y_train.mean() * 100:.2f}%) | "
        f"Val rows: {len(X_val)} (at-risk: {y_val.mean() * 100:.2f}%) | "
        f"Features: {len(FEATURE_COLUMNS)}\n"
    )

    baseline_rows = evaluate_baselines(X_train, y_train, X_val, y_val)

    # Search small hyperparameter grids per candidate family on validation ONLY.
    search_space = get_candidate_search_space(seed=seed)
    best_per_family: list[dict[str, Any]] = []
    best_pipelines: dict[str, Pipeline] = {}
    best_val_probas: dict[str, np.ndarray] = {}

    for family, param_list in search_space.items():
        best_score = -1.0
        best_entry: dict[str, Any] = {}
        best_pipe: Pipeline | None = None
        best_proba: np.ndarray | None = None

        for params in param_list:
            pipe = make_pipeline(family, params, seed=seed)
            pipe.fit(X_train, y_train)
            val_proba = pipe.predict_proba(X_val)[:, 1]

            # Tune F2 threshold on validation for each candidate so fair F2/recall comparison
            # is also visible, while using threshold-free PR-AUC as the primary ranking key.
            tuned_thr, tuned_metrics = tune_threshold_for_f2(y_val, val_proba)
            default_metrics = compute_metrics(y_val, val_proba, threshold=0.50)

            # Primary score: validation PR-AUC (with small tie-breaker on tuned F2).
            selection_score = float(default_metrics["pr_auc"]) + 1e-4 * float(tuned_metrics["f2"])
            if selection_score > best_score:
                best_score = selection_score
                best_pipe = pipe
                best_proba = val_proba
                best_entry = {
                    "family": family,
                    "name": f"{family} (tuned)",
                    "params": params,
                    "default_0_50": default_metrics,
                    "tuned_f2_threshold": tuned_thr,
                    **tuned_metrics,
                }

        assert best_pipe is not None and best_proba is not None
        best_per_family.append(best_entry)
        best_pipelines[family] = best_pipe
        best_val_probas[family] = best_proba

    # Print comparison at standard threshold 0.50 first, then with tuned F2 thresholds.
    print("--- Validation Comparison (Default Threshold = 0.50 / Native Rule) ---")
    default_table = baseline_rows + [
        {"name": f"{e['family']} (best)", "params": e["params"], **e["default_0_50"]}
        for e in best_per_family
    ]
    print_comparison_table(default_table)

    print("\n--- Validation Comparison (Threshold Tuned on Validation to Maximize F2) ---")
    tuned_table = baseline_rows + best_per_family
    print_comparison_table(tuned_table)

    # Select overall winning model by highest validation PR-AUC (breaking ties by F2).
    winner_entry = max(
        best_per_family,
        key=lambda e: (round(float(e["pr_auc"]), 6), float(e["f2"])),
    )
    winner_family = str(winner_entry["family"])
    winner_pipeline = best_pipelines[winner_family]
    winner_proba = best_val_probas[winner_family]
    winner_threshold = float(winner_entry["threshold"])

    # Compute Low / Medium / High risk-band cutoffs from validation probabilities.
    band_cutoffs, band_summary = compute_risk_band_cutoffs(
        y_val=y_val,
        val_proba=winner_proba,
        decision_threshold=winner_threshold,
    )

    print("\n=== Winning Model & Risk Bands (Validation Set) ===")
    print(f"Selected Model:       {winner_family}")
    print(f"Best Hyperparameters: {winner_entry['params']}")
    print(f"F2-Optimal Threshold: {winner_threshold:.2f}")
    print(
        f"Risk-Band Cutoffs:    Low (< {band_cutoffs['low_max']:.2f}), "
        f"Medium ({band_cutoffs['low_max']:.2f} to < {band_cutoffs['medium_max']:.2f}), "
        f"High (>= {band_cutoffs['medium_max']:.2f})"
    )
    print("\nValidation Risk-Band Calibration:")
    for band_name, stats in band_summary.items():
        print(
            f"  - {band_name:<6}: {stats['count']:4d} students ({stats['share_pct']:5.1f}%) | "
            f"actual at-risk rate = {stats['observed_at_risk_rate'] * 100:5.1f}%"
        )

    # Save the fitted Pipeline and metadata.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, model_path)

    metadata: dict[str, Any] = {
        "model_family": winner_family,
        "best_params": winner_entry["params"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "decision_threshold": winner_threshold,
        "risk_band_cutoffs": band_cutoffs,
        "risk_band_validation_summary": band_summary,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "selection_criterion": "highest validation PR-AUC; threshold tuned on validation for max F2",
        "validation_metrics_winner_tuned": {
            "precision": winner_entry["precision"],
            "recall": winner_entry["recall"],
            "f1": winner_entry["f1"],
            "f2": winner_entry["f2"],
            "pr_auc": winner_entry["pr_auc"],
            "confusion_matrix": winner_entry["confusion_matrix"],
        },
        "validation_comparison_default_threshold": default_table,
        "validation_comparison_tuned_f2": [
            {k: v for k, v in entry.items() if k != "default_0_50"}
            for entry in tuned_table
        ],
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved full pipeline to: {model_path}")
    print(f"Saved metadata to:      {metadata_path}")

    return winner_pipeline, metadata


def main() -> None:
    """Entry point for `python -m src.train`."""
    train_and_select()


if __name__ == "__main__":
    main()
