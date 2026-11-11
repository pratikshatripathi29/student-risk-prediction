"""Inference module loading the saved model artifact for advisory predictions.

Rules enforced from PROJECT_CONTEXT.md:
- Never hard-code predictions: always load the trained Pipeline from
  `models/model.joblib` and decision thresholds from `models/metadata.json`.
- Training-serving consistency: uses `build_features_from_inputs` from
  `src/features.py` so app predictions compute features identically to training.
- Advisory only: outputs support signals and explanations for educators, never
  automated grading, disciplinary, or admission decisions.

Run a CLI demo from the project root:
    python -m src.predict
"""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_METADATA_PATH, MODEL_PATH
from src.explain import explain_prediction
from src.features import FEATURE_COLUMNS, build_features_from_inputs

ADVISORY_DISCLAIMER: str = (
    "Advisory support signal only: use alongside teacher judgment to offer timely "
    "tutoring or check-ins—never for grading penalties, punishment, or admissions."
)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved scikit-learn Pipeline and metadata JSON from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run `python -m src.train` first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}. Run `python -m src.train` first.")

    pipeline: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
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


def predict_from_features(
    features_df: pd.DataFrame,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    include_explanation: bool = True,
    top_n_drivers: int = 3,
) -> dict[str, Any]:
    """Run prediction on a 1-row feature DataFrame using the saved model artifact."""
    expected_cols = list(metadata.get("feature_columns", FEATURE_COLUMNS))
    missing_cols = [col for col in expected_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    if len(features_df) != 1:
        raise ValueError(f"predict_from_features expects 1 row, got {len(features_df)}.")

    X = features_df[expected_cols]
    risk_proba = float(pipeline.predict_proba(X)[:, 1][0])

    threshold = float(metadata.get("decision_threshold", 0.50))
    band_cutoffs = dict(metadata.get("risk_band_cutoffs", {"low_max": threshold, "medium_max": 0.75}))

    flagged = risk_proba >= threshold
    risk_band = assign_risk_band(risk_proba, band_cutoffs)

    explanation: dict[str, Any] = {"increasing_risk": [], "reducing_risk": []}
    if include_explanation:
        explanation = explain_prediction(pipeline, X, top_n=top_n_drivers)

    return {
        "risk_probability": round(risk_proba, 4),
        "risk_band": risk_band,
        "flagged_for_support": bool(flagged),
        "decision_threshold": threshold,
        "risk_band_cutoffs": band_cutoffs,
        "model_family": metadata.get("model_family", type(pipeline.named_steps["model"]).__name__),
        "increasing_risk": explanation["increasing_risk"],
        "reducing_risk": explanation["reducing_risk"],
        "advisory_note": ADVISORY_DISCLAIMER,
    }


def predict_from_student_history(
    scores_last3: Sequence[float],
    attendance_last3: Sequence[float],
    completion_last3: Sequence[float],
    late_submission_last3: Sequence[float],
    grade_level: int,
    pipeline: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
    include_explanation: bool = True,
) -> dict[str, Any]:
    """Build features from raw last-3 inputs and return advisory risk prediction."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_artifacts()

    features_df = build_features_from_inputs(
        scores_last3=scores_last3,
        attendance_last3=attendance_last3,
        completion_last3=completion_last3,
        late_submission_last3=late_submission_last3,
        grade_level=grade_level,
    )
    result = predict_from_features(
        features_df=features_df,
        pipeline=pipeline,
        metadata=metadata,
        include_explanation=include_explanation,
    )
    result["computed_features"] = features_df.iloc[0].round(2).to_dict()
    return result


def main() -> None:
    """Demonstrate live inference on two example student trajectories."""
    pipeline, metadata = load_artifacts()
    print(f"Loaded model: {metadata.get('model_family')} | Threshold: {metadata.get('decision_threshold')}")

    examples = [
        (
            "Example A (Declining engagement & scores)",
            dict(
                scores_last3=[64.0, 55.0, 46.0],
                attendance_last3=[86.0, 79.0, 72.0],
                completion_last3=[80.0, 71.0, 62.0],
                late_submission_last3=[12.0, 20.0, 28.0],
                grade_level=9,
            ),
        ),
        (
            "Example B (Consistent performance)",
            dict(
                scores_last3=[72.0, 75.0, 78.0],
                attendance_last3=[92.0, 94.0, 95.0],
                completion_last3=[88.0, 90.0, 93.0],
                late_submission_last3=[6.0, 5.0, 4.0],
                grade_level=9,
            ),
        ),
    ]

    for title, inputs in examples:
        pred = predict_from_student_history(pipeline=pipeline, metadata=metadata, **inputs)
        print(f"\n--- {title} ---")
        print(
            f"Risk Probability: {pred['risk_probability']:.1%} | "
            f"Band: {pred['risk_band']} | "
            f"Flagged for Support: {pred['flagged_for_support']}"
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
