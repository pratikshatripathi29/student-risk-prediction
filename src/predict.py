"""Inference module loading the saved model artifact for advisory predictions.

Rules enforced from PROJECT_CONTEXT.md:
- Never hard-code predictions: always load the trained Pipeline from
  `models/model.joblib` and metadata from `models/metadata.json` via `load_model()`.
- Training-serving consistency: `predict_risk(inputs)` calls
  `build_features_from_inputs` from `src/features.py` so app predictions compute
  the exact same features used in training.
- Advisory only: outputs support signals and plain-English explanations from
  `src/explain.py`, never automated punishment, grading, or admission decisions.

Run a CLI demo from the project root:
    python -m src.predict
"""

from collections.abc import Sequence
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_METADATA_PATH, MODEL_PATH
from src.explain import explain_prediction
from src.features import FEATURE_COLUMNS, build_features_from_inputs

ADVISORY_DISCLAIMER: str = (
    "Advisory tool only. Not for grading, punishment or admission decisions. "
    "A teacher/counselor must review every result."
)


@lru_cache(maxsize=4)
def load_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load and cache the saved scikit-learn Pipeline and metadata JSON."""
    model_file = Path(model_path)
    meta_file = Path(metadata_path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_file}. Run `python -m src.train` first."
        )
    if not meta_file.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {meta_file}. Run `python -m src.train` first."
        )

    pipeline: Pipeline = joblib.load(model_file)
    metadata: dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))
    return pipeline, metadata


def assign_risk_band(probability: float, band_cutoffs: dict[str, float]) -> str:
    """Map predicted risk probability to Low, Medium, or High risk band."""
    low_max = float(band_cutoffs.get("low_max", 0.37))
    medium_max = float(band_cutoffs.get("medium_max", 0.76))

    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def _validate_and_normalize_sequence(values: Sequence[Any], name: str) -> list[float]:
    """Ensure a 3-element sequence has values in [0, 100] (allowing None/NaN as missing)."""
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values ([k-2, k-1, k]), got {len(values)}.")

    cleaned: list[float] = []
    for idx, val in enumerate(values):
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            cleaned.append(float("nan"))
            continue

        num = float(val)
        if not (0.0 <= num <= 100.0):
            raise ValueError(f"{name}[{idx}] must be between 0 and 100, got {num}.")
        cleaned.append(num)

    return cleaned


def predict_risk(
    inputs: dict[str, Any],
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    model_bundle: tuple[Pipeline, dict[str, Any]] | None = None,
    top_n_factors: int = 3,
) -> dict[str, Any]:
    """Predict next-assessment risk from raw student inputs using the saved model.

    Expected keys in `inputs`:
    - `grade_level`: int (8, 9, or 10)
    - `scores_last3`: 3 chronological scores in [0, 100] (None/NaN allowed)
    - `attendance_last3`: 3 attendance percentages in [0, 100] (None/NaN allowed)
    - `completion_last3`: 3 completion percentages in [0, 100] (None/NaN allowed)
    - `late_submission_last3`: 3 late-submission percentages in [0, 100] (None/NaN allowed)
    """
    pipeline, metadata = model_bundle if model_bundle is not None else load_model(model_path, metadata_path)

    scores = _validate_and_normalize_sequence(inputs["scores_last3"], "scores_last3")
    attendance = _validate_and_normalize_sequence(inputs["attendance_last3"], "attendance_last3")
    completion = _validate_and_normalize_sequence(inputs["completion_last3"], "completion_last3")
    late_sub = _validate_and_normalize_sequence(inputs["late_submission_last3"], "late_submission_last3")
    grade_level = int(inputs["grade_level"])

    features_df = build_features_from_inputs(
        scores_last3=scores,
        attendance_last3=attendance,
        completion_last3=completion,
        late_submission_last3=late_sub,
        grade_level=grade_level,
    )

    expected_cols = list(metadata.get("feature_columns", FEATURE_COLUMNS))
    X = features_df[expected_cols]
    risk_proba = float(pipeline.predict_proba(X)[:, 1][0])

    threshold = float(metadata.get("decision_threshold", 0.50))
    band_cutoffs = dict(metadata.get("risk_band_cutoffs", {"low_max": threshold, "medium_max": 0.75}))

    flagged = risk_proba >= threshold
    risk_band = assign_risk_band(risk_proba, band_cutoffs)
    explanation = explain_prediction(pipeline, X, top_n=top_n_factors)

    missing_input_count = int(
        sum(np.isnan(val) for seq in (scores, attendance, completion, late_sub) for val in seq)
    )

    return {
        "probability": round(risk_proba, 4),
        "risk_band": risk_band,
        "flagged": bool(flagged),
        "decision_threshold": threshold,
        "risk_band_cutoffs": band_cutoffs,
        "model_family": metadata.get("model_family", type(pipeline.named_steps["model"]).__name__),
        "increasing_risk": explanation["increasing_risk"],
        "reducing_risk": explanation["reducing_risk"],
        "computed_features": features_df.iloc[0].round(2).to_dict(),
        "missing_input_count": missing_input_count,
        "advisory_note": ADVISORY_DISCLAIMER,
    }


def main() -> None:
    """Demonstrate live inference on two example trajectories (including missing data)."""
    pipeline, metadata = load_model()
    print(f"Loaded model: {metadata.get('model_family')} | Threshold: {metadata.get('decision_threshold')}")

    examples = [
        (
            "Example A (Declining scores & engagement)",
            {
                "grade_level": 9,
                "scores_last3": [64.0, 55.0, 46.0],
                "attendance_last3": [86.0, 79.0, 72.0],
                "completion_last3": [80.0, 71.0, 62.0],
                "late_submission_last3": [12.0, 20.0, 28.0],
            },
        ),
        (
            "Example B (Consistent performance with 1 missing attendance entry)",
            {
                "grade_level": 9,
                "scores_last3": [72.0, 75.0, 78.0],
                "attendance_last3": [92.0, None, 95.0],
                "completion_last3": [88.0, 90.0, 93.0],
                "late_submission_last3": [6.0, 5.0, 4.0],
            },
        ),
    ]

    for title, inputs in examples:
        pred = predict_risk(inputs, model_bundle=(pipeline, metadata))
        print(f"\n--- {title} ---")
        print(
            f"Risk Probability: {pred['probability']:.1%} | "
            f"Risk Band: {pred['risk_band']} | "
            f"Flagged: {pred['flagged']} | "
            f"Missing inputs: {pred['missing_input_count']}"
        )
        print("Increasing risk:")
        for item in pred["increasing_risk"] or [{"explanation": "None."}]:
            print(f"  + {item['explanation']}")
        print("Reducing risk:")
        for item in pred["reducing_risk"] or [{"explanation": "None."}]:
            print(f"  - {item['explanation']}")

    print(f"\nNote: {ADVISORY_DISCLAIMER}")


if __name__ == "__main__":
    main()
