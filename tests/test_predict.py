"""Tests for saved artifact loading, inference, missing values, and risk-band boundaries."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.features import FEATURE_COLUMNS
from src.predict import assign_risk_band, load_model, predict_risk


def test_saved_artifact_loads_pipeline_and_metadata() -> None:
    """load_model must return a fitted sklearn Pipeline and valid metadata without gender."""
    pipeline, metadata = load_model()

    assert isinstance(pipeline, Pipeline)
    assert list(metadata["feature_columns"]) == FEATURE_COLUMNS
    assert "gender" not in metadata["feature_columns"]
    assert "gender" not in list(pipeline.feature_names_in_)


def test_predict_risk_returns_valid_probability_and_handles_missing_values() -> None:
    """predict_risk must output probabilities in [0, 1] and impute None/NaN values cleanly."""
    complete = predict_risk(
        {
            "grade_level": 9,
            "scores_last3": [64.0, 55.0, 46.0],
            "attendance_last3": [86.0, 79.0, 72.0],
            "completion_last3": [80.0, 71.0, 62.0],
            "late_submission_last3": [12.0, 20.0, 28.0],
        }
    )
    assert 0.0 <= complete["probability"] <= 1.0
    assert complete["risk_band"] in {"Low", "Medium", "High"}
    assert complete["flagged"] == (complete["probability"] >= complete["decision_threshold"])

    with_missing = predict_risk(
        {
            "grade_level": 8,
            "scores_last3": [None, 60.0, np.nan],
            "attendance_last3": [85.0, None, 80.0],
            "completion_last3": [None, None, None],
            "late_submission_last3": [10.0, np.nan, 15.0],
        }
    )
    assert 0.0 <= with_missing["probability"] <= 1.0
    assert with_missing["missing_input_count"] == 7


def test_risk_band_cutoffs_at_exact_boundaries() -> None:
    """Risk bands must map cleanly at exact low_max and medium_max thresholds."""
    _, metadata = load_model()
    cutoffs = metadata["risk_band_cutoffs"]
    low_max = float(cutoffs["low_max"])
    medium_max = float(cutoffs["medium_max"])

    assert assign_risk_band(0.0, cutoffs) == "Low"
    assert assign_risk_band(low_max - 1e-6, cutoffs) == "Low"
    assert assign_risk_band(low_max, cutoffs) == "Medium"
    assert assign_risk_band(medium_max - 1e-6, cutoffs) == "Medium"
    assert assign_risk_band(medium_max, cutoffs) == "High"
    assert assign_risk_band(1.0, cutoffs) == "High"


def test_predict_risk_rejects_out_of_range_inputs() -> None:
    """Inputs outside [0, 100] should raise a ValueError."""
    with pytest.raises(ValueError, match="between 0 and 100"):
        predict_risk(
            {
                "grade_level": 10,
                "scores_last3": [70.0, 110.0, 65.0],
                "attendance_last3": [85.0, 85.0, 85.0],
                "completion_last3": [80.0, 80.0, 80.0],
                "late_submission_last3": [10.0, 10.0, 10.0],
            }
        )


def test_streamlit_app_smoke_handles_blanks_and_validation() -> None:
    """Streamlit app loads cleanly, handles blank inputs, and flags out-of-range inputs."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py")).run(timeout=30)
    assert len(at.exception) == 0
    assert [t.label for t in at.tabs] == ["Predict Student Risk", "Model performance"]

    at.text_input(key="a2").set_value("").run(timeout=30)
    assert len(at.exception) == 0
    assert any("1 blank input(s)" in info.value for info in at.info)

    at.text_input(key="s3").set_value("125").run(timeout=30)
    assert any("between 0 and 100" in err.value for err in at.error)

