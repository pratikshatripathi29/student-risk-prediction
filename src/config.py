"""Central configuration for the Student Performance & At-Risk Prediction project.

Keeping shared constants and file paths in one file avoids magic numbers
and ensures every script uses the exact same random seed and folder layout.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Permanent random seed used across data generation, splitting, and modeling.
SEED: int = 42

# ---------------------------------------------------------------------------
# Project Directory Paths
# ---------------------------------------------------------------------------
# Resolve the repository root relative to this file (src/config.py -> repo root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Standard File Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH: Path = RAW_DATA_DIR / "students_raw.csv"
PROCESSED_DATA_PATH: Path = PROCESSED_DATA_DIR / "checkpoint_dataset.csv"

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH: Path = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "metadata.json"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
METRICS_TABLE_PATH: Path = REPORTS_DIR / "metrics_table.csv"
FEATURE_IMPORTANCE_PATH: Path = REPORTS_DIR / "feature_importance.csv"
BIAS_AUDIT_PATH: Path = REPORTS_DIR / "bias_audit.csv"

# ---------------------------------------------------------------------------
# Dataset & Problem Definition Constants
# ---------------------------------------------------------------------------
# Number of synthetic students to simulate.
N_STUDENTS: int = 1500

# Total assessments per student (assessments 1 through 8).
N_ASSESSMENTS: int = 8

# Checkpoints k = 3..7 predict whether assessment k + 1 (4..8) falls below 50.
MIN_CHECKPOINT: int = 3
MAX_CHECKPOINT: int = N_ASSESSMENTS - 1

# Target threshold: low_next_score = 1 if next assessment score < 50, else 0.
LOW_SCORE_THRESHOLD: int = 50

