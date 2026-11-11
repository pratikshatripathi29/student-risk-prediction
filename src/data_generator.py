"""Generate a realistic synthetic student assessment dataset for the project.

Why synthetic data?
- Real student records are sensitive and protected by privacy rules.
- A controlled synthetic generator lets us test our ML pipeline safely,
  guarantee reproducibility with SEED = 42, and simulate real-world data issues
  (missing entries, impossible values, duplicates, and class imbalance).

How the simulation works:
1. For each of N_STUDENTS students, create observable demographics
   (`grade_level`, and synthetic `gender` strictly for post-hoc bias audits)
   plus hidden latent traits (`ability`, `effort`, and a slow `trend_slope`).
   `gender` is generated independently of `ability` and `effort` and must
   NEVER be used as a model feature.
2. Simulate N_ASSESSMENTS sequential assessments (~2-3 weeks apart) per student
   in long format (one row per student per assessment). Latent traits plus a
   slowly drifting recent-engagement factor drive `score`, `attendance_pct`,
   `assignment_completion_pct`, and `late_submission_pct` so all features move
   together realistically.
3. Tune the score baseline so roughly 15-25% of scores fall below
   LOW_SCORE_THRESHOLD (50), creating realistic class imbalance.
4. Intentionally inject realistic raw-data defects (~3% missing attendance,
   ~2% missing assignment completion, a few impossible values such as
   attendance > 100 or negative scores, and ~0.5% duplicate rows) so downstream
   data-quality checks have real issues to catch.
5. Save the raw dataset to `data/raw/students_raw.csv` and print a summary.

Run from the project root:
    python -m src.data_generator
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LOW_SCORE_THRESHOLD,
    N_ASSESSMENTS,
    N_STUDENTS,
    RAW_DATA_PATH,
    SEED,
)


def generate_student_profiles(
    rng: np.random.Generator,
    n_students: int,
) -> pd.DataFrame:
    """Create one row per student with observable IDs/metadata and hidden traits."""
    student_ids = [f"STU_{i:04d}" for i in range(1, n_students + 1)]
    grade_levels = rng.choice([8, 9, 10], size=n_students, replace=True)

    # Synthetic gender is ONLY for downstream fairness/bias audits.
    # It is generated independently of ability and effort, and never used as a feature.
    genders = rng.choice(
        ["Female", "Male", "Non-binary"],
        size=n_students,
        p=[0.48, 0.48, 0.04],
    )

    # Hidden academic ability (standardized normal distribution).
    ability = rng.normal(loc=0.0, scale=1.0, size=n_students)

    # Hidden effort has a mild positive correlation (~0.30) with ability,
    # reflecting that preparation and consistency often reinforce each other.
    effort = 0.30 * ability + np.sqrt(1.0 - 0.30**2) * rng.normal(
        loc=0.0, scale=1.0, size=n_students
    )

    # Slow individual trajectory across the term (points per assessment):
    # positive = steady improvement, negative = gradual burnout/decline.
    trend_slope = rng.normal(loc=0.0, scale=1.5, size=n_students)

    return pd.DataFrame(
        {
            "student_id": student_ids,
            "grade_level": grade_levels,
            "gender": genders,
            "ability": ability,
            "effort": effort,
            "trend_slope": trend_slope,
        }
    )


def build_assessment_dates(n_assessments: int) -> list[str]:
    """Build assessment dates spaced ~2-3 weeks (17-18 days) apart."""
    start_date = date(2025, 9, 1)
    dates: list[str] = []
    current_day_offset = 0

    for idx in range(n_assessments):
        dates.append((start_date + timedelta(days=current_day_offset)).isoformat())
        # Alternate 17 and 18 days (~2.5 weeks) to match a realistic term calendar.
        current_day_offset += 17 if idx % 2 == 0 else 18

    return dates


def generate_assessment_records(
    rng: np.random.Generator,
    profiles: pd.DataFrame,
    n_assessments: int,
) -> pd.DataFrame:
    """Simulate longitudinal assessment rows for every student across the term."""
    n_students = len(profiles)
    assessment_dates = build_assessment_dates(n_assessments)

    # Small exam-to-exam difficulty variation shared across the whole cohort.
    exam_difficulty = rng.normal(loc=0.0, scale=2.0, size=n_assessments)

    # Time-varying engagement drift (AR(1) process): captures multi-week
    # good streaks or rough patches so recent behavior carries predictive signal.
    engagement_drift = np.zeros((n_students, n_assessments), dtype=float)
    engagement_drift[:, 0] = rng.normal(loc=0.0, scale=0.6, size=n_students)
    for t in range(1, n_assessments):
        engagement_drift[:, t] = 0.65 * engagement_drift[:, t - 1] + rng.normal(
            loc=0.0, scale=0.55, size=n_students
        )

    ability = profiles["ability"].to_numpy()
    effort = profiles["effort"].to_numpy()
    trend_slope = profiles["trend_slope"].to_numpy()

    records: list[dict[str, object]] = []

    for t in range(n_assessments):
        assessment_no = t + 1
        drift_t = engagement_drift[:, t]

        # Attendance (%) is strongly tied to effort and recent engagement.
        raw_attendance = (
            85.0
            + 7.5 * effort
            + 4.5 * drift_t
            + 0.35 * trend_slope * t
            + rng.normal(loc=0.0, scale=4.5, size=n_students)
        )
        attendance_pct = np.round(np.clip(raw_attendance, 0.0, 100.0), 1)

        # Assignment completion (%) reflects effort, engagement, and a touch of ability.
        raw_completion = (
            80.0
            + 9.5 * effort
            + 2.5 * ability
            + 5.5 * drift_t
            + 0.45 * trend_slope * t
            + rng.normal(loc=0.0, scale=6.0, size=n_students)
        )
        assignment_completion_pct = np.round(np.clip(raw_completion, 0.0, 100.0), 1)

        # Late submission (%) moves opposite to effort and engagement.
        raw_late = (
            14.0
            - 6.5 * effort
            - 3.5 * drift_t
            - 0.25 * trend_slope * t
            + rng.normal(loc=0.0, scale=5.0, size=n_students)
        )
        late_submission_pct = np.round(np.clip(raw_late, 0.0, 100.0), 1)

        # Assessment score (0-100): tuned so ~15-25% of scores fall below 50.
        raw_score = (
            64.5
            + 10.5 * ability
            + 6.5 * effort
            + 4.5 * drift_t
            + trend_slope * t
            - exam_difficulty[t]
            + rng.normal(loc=0.0, scale=8.5, size=n_students)
        )
        score = np.round(np.clip(raw_score, 0.0, 100.0), 1)

        for i in range(n_students):
            records.append(
                {
                    "student_id": profiles.loc[i, "student_id"],
                    "grade_level": int(profiles.loc[i, "grade_level"]),
                    "gender": profiles.loc[i, "gender"],
                    "assessment_no": assessment_no,
                    "assessment_date": assessment_dates[t],
                    "score": float(score[i]),
                    "attendance_pct": float(attendance_pct[i]),
                    "assignment_completion_pct": float(assignment_completion_pct[i]),
                    "late_submission_pct": float(late_submission_pct[i]),
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["student_id", "assessment_no"]).reset_index(drop=True)


def inject_data_issues(
    rng: np.random.Generator,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inject realistic missingness, impossible values, and duplicate rows."""
    dirty_df = df.copy()
    n_rows = len(dirty_df)

    # 1. Inject ~3% missing attendance values.
    n_missing_attendance = int(round(0.03 * n_rows))
    missing_att_idx = rng.choice(n_rows, size=n_missing_attendance, replace=False)
    dirty_df.loc[missing_att_idx, "attendance_pct"] = np.nan

    # 2. Inject ~2% missing assignment completion values.
    n_missing_completion = int(round(0.02 * n_rows))
    missing_comp_idx = rng.choice(n_rows, size=n_missing_completion, replace=False)
    dirty_df.loc[missing_comp_idx, "assignment_completion_pct"] = np.nan

    # 3. Inject a few impossible values (typos/data-entry bugs):
    #    - attendance > 100 on non-missing attendance rows
    #    - negative scores on separate rows
    valid_att_indices = dirty_df.index[dirty_df["attendance_pct"].notna()].to_numpy()
    n_impossible_att = 12
    impossible_att_idx = rng.choice(
        valid_att_indices, size=n_impossible_att, replace=False
    )
    dirty_df.loc[impossible_att_idx, "attendance_pct"] = np.round(
        rng.uniform(101.0, 115.0, size=n_impossible_att), 1
    )

    n_impossible_score = 12
    impossible_score_idx = rng.choice(n_rows, size=n_impossible_score, replace=False)
    dirty_df.loc[impossible_score_idx, "score"] = np.round(
        rng.uniform(-15.0, -1.0, size=n_impossible_score), 1
    )

    # 4. Inject ~0.5% exact duplicate rows.
    n_duplicates = int(round(0.005 * n_rows))
    dup_idx = rng.choice(n_rows, size=n_duplicates, replace=False)
    duplicate_rows = dirty_df.iloc[dup_idx].copy()

    dirty_df = (
        pd.concat([dirty_df, duplicate_rows], ignore_index=True)
        .sort_values(["student_id", "assessment_no"], kind="mergesort")
        .reset_index(drop=True)
    )

    issue_counts = {
        "missing_attendance": int(dirty_df["attendance_pct"].isna().sum()),
        "missing_completion": int(dirty_df["assignment_completion_pct"].isna().sum()),
        "impossible_attendance_gt_100": int((dirty_df["attendance_pct"] > 100).sum()),
        "impossible_negative_scores": int((dirty_df["score"] < 0).sum()),
        "duplicate_rows": int(dirty_df.duplicated().sum()),
    }
    return dirty_df, issue_counts


def print_summary(
    clean_rows_count: int,
    df: pd.DataFrame,
    issue_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Print an easy-to-read audit summary of the generated raw dataset."""
    valid_scores = df[df["score"] >= 0]
    pct_below_50_all = (valid_scores["score"] < LOW_SCORE_THRESHOLD).mean() * 100
    pct_below_50_targets = (
        valid_scores.loc[valid_scores["assessment_no"] >= 4, "score"]
        < LOW_SCORE_THRESHOLD
    ).mean() * 100

    print("=== Synthetic Student Dataset Summary ===")
    print(f"Saved to:                  {output_path}")
    print(f"Students:                  {df['student_id'].nunique()}")
    print(f"Clean base rows:           {clean_rows_count}")
    print(f"Final rows (with dups):    {len(df)}")
    print(f"Columns ({len(df.columns)}):               {list(df.columns)}")
    print(
        f"Scores < {LOW_SCORE_THRESHOLD} (all valid):    {pct_below_50_all:.2f}%"
    )
    print(
        f"Scores < {LOW_SCORE_THRESHOLD} (assess 4..8):  {pct_below_50_targets:.2f}%"
    )
    print("\n--- Injected Data Quality Issues ---")
    print(
        f"Missing attendance_pct:    {issue_counts['missing_attendance']} "
        f"({issue_counts['missing_attendance'] / len(df) * 100:.2f}%)"
    )
    print(
        f"Missing completion_pct:    {issue_counts['missing_completion']} "
        f"({issue_counts['missing_completion'] / len(df) * 100:.2f}%)"
    )
    print(
        f"Impossible attendance>100: {issue_counts['impossible_attendance_gt_100']}"
    )
    print(
        f"Impossible negative score: {issue_counts['impossible_negative_scores']}"
    )
    print(
        f"Exact duplicate rows:      {issue_counts['duplicate_rows']} "
        f"({issue_counts['duplicate_rows'] / len(df) * 100:.2f}%)"
    )
    print("\n--- Grade & Audit-Only Gender Breakdown (Unique Students) ---")
    unique_students = df.drop_duplicates(subset=["student_id"])
    print("Grade levels:", unique_students["grade_level"].value_counts().sort_index().to_dict())
    print("Gender (audit only):", unique_students["gender"].value_counts().to_dict())


def generate_dataset(
    output_path: Path = RAW_DATA_PATH,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the full synthetic generation pipeline and write the CSV to disk."""
    rng = np.random.default_rng(seed)

    profiles = generate_student_profiles(rng=rng, n_students=N_STUDENTS)
    clean_df = generate_assessment_records(
        rng=rng, profiles=profiles, n_assessments=N_ASSESSMENTS
    )
    dirty_df, issue_counts = inject_data_issues(rng=rng, df=clean_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dirty_df.to_csv(output_path, index=False)

    print_summary(
        clean_rows_count=len(clean_df),
        df=dirty_df,
        issue_counts=issue_counts,
        output_path=output_path,
    )
    return dirty_df


def main() -> None:
    """Entry point for `python -m src.data_generator`."""
    generate_dataset()


if __name__ == "__main__":
    main()
