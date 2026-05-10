from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
TEAM_PROCESSING_DIR = REPO_ROOT / "Matteo"
DATA_DIR = BASE_DIR / "data" / "ML project data"
SUBMISSION_DIR = BASE_DIR / "submissions"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "submission.csv"

BASE_CATEGORICAL_COLS = [
    "Agency",
    "Incident Zip",
    "Police Precinct",
    "Borough",
    "Open Data Channel Type",
    "Problem_Grouped",
    "Location_Grouped",
]

RAW_DETAIL_CATEGORICAL_COLS = [
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Additional Details",
    "Location Type",
    "Address Type",
    "City",
    "Community Board",
    "Council District",
    "Facility Type",
]

GEO_CATEGORICAL_COLS = [
    "Street Name",
    "BBL",
]

CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + RAW_DETAIL_CATEGORICAL_COLS + GEO_CATEGORICAL_COLS

BASE_NUMERICAL_COLS = [
    "Created_Hour",
    "Created_DayOfWeek",
    "Created_Month",
    "Is_Weekend",
    "Is_Landmark",
    "Is_Taxi",
]

DATE_NUMERICAL_COLS = [
    "Created_Day",
    "Created_Minute",
    "Created_Is_Night",
    "Created_Is_Business_Hours",
]

GEO_NUMERICAL_COLS = [
    "Latitude",
    "Longitude",
    "X Coordinate (State Plane)",
    "Y Coordinate (State Plane)",
]

NUMERICAL_COLS = BASE_NUMERICAL_COLS + DATE_NUMERICAL_COLS + GEO_NUMERICAL_COLS

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

sys.path.insert(0, str(TEAM_PROCESSING_DIR))
from DS_processing import Process_test_DS, Process_train_DS


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files without letting the index column become a feature."""
    required_paths = [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing local data files. Unzip/copy the project CSV files into "
            "`Riccardo/data/ML project data/` before running this model:\n"
            f"{missing}"
        )

    train_raw = pd.read_csv(TRAIN_PATH, index_col=0)
    test_raw = pd.read_csv(TEST_PATH, index_col=0)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    return train_raw, test_raw, sample_submission


def add_raw_features(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add safe raw columns that are available in both train and test."""
    features = processed_df.drop(columns=["Y"], errors="ignore").copy()

    for col in RAW_DETAIL_CATEGORICAL_COLS + GEO_CATEGORICAL_COLS + GEO_NUMERICAL_COLS:
        if col in raw_df.columns:
            features[col] = raw_df.loc[features.index, col]

    if "Created Date" in raw_df.columns:
        created = pd.to_datetime(
            raw_df.loc[features.index, "Created Date"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        features["Created_Day"] = created.dt.day
        features["Created_Minute"] = created.dt.minute
        features["Created_Is_Night"] = created.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
        features["Created_Is_Business_Hours"] = (
            (created.dt.hour >= 9)
            & (created.dt.hour <= 17)
            & (~created.dt.dayofweek.isin([5, 6]))
        ).astype(int)

    return features


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make CatBoost inputs explicit and stable across train, validation, and test."""
    prepared = df.copy()

    for col in CATEGORICAL_COLS:
        if col not in prepared.columns:
            prepared[col] = "Unknown"
        prepared[col] = (
            prepared[col]
            .fillna("Unknown")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

    for col in NUMERICAL_COLS:
        if col not in prepared.columns:
            prepared[col] = np.nan
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
        if prepared[col].isna().any():
            median = prepared[col].median()
            prepared[col] = prepared[col].fillna(0 if pd.isna(median) else median)

    return prepared[FEATURE_COLS]


def build_feature_frame(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    features = add_raw_features(processed_df, raw_df)
    return prepare_features(features)


def print_dataset_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Matteo's requested quick checks, plus target balance."""
    print("\nProcessed train info:")
    train_df.info()

    print("\nProcessed test info:")
    test_df.info()

    print("\nNull values in processed train:")
    print(train_df.isna().sum().sort_values(ascending=False))

    print("\nNull values in processed test:")
    print(test_df.isna().sum().sort_values(ascending=False))

    print("\nTarget distribution:")
    print(train_df["Y"].value_counts(normalize=True).round(4))


def build_model(iterations: int, depth: int, learning_rate: float, l2_leaf_reg: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=150,
        allow_writing_files=False,
        verbose=100,
    )


def best_accuracy_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.35, 0.65, 301)
    scores = [accuracy_score(y_true, (probabilities >= threshold).astype(int)) for threshold in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def save_feature_importance(model: CatBoostClassifier, train_pool: Pool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.get_feature_importance(train_pool),
        }
    ).sort_values("importance", ascending=False)

    out_path = OUTPUT_DIR / "catboost_feature_importance.csv"
    importance.to_csv(out_path, index=False)
    print(f"\nSaved feature importance to: {out_path}")
    print("\nTop feature importances:")
    print(importance.head(12).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost for the NYC 311 ML project.")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--skip-final", action="store_true", help="Only run validation, without final test submission.")
    args = parser.parse_args()

    train_raw, test_raw, sample_submission = load_raw_data()

    train_df = Process_train_DS(train_raw.copy())
    test_df = Process_test_DS(test_raw.copy())
    print_dataset_checks(train_df, test_df)

    X = build_feature_frame(train_df, train_raw)
    y = train_df["Y"].astype(int)
    X_test = build_feature_frame(test_df, test_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Train Y=1 proportion: {y_train.mean():.4f}")
    print(f"Validation Y=1 proportion: {y_val.mean():.4f}")

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_COLS)

    model = build_model(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_probabilities = model.predict_proba(val_pool)[:, 1]
    default_predictions = (val_probabilities >= 0.5).astype(int)
    default_accuracy = accuracy_score(y_val, default_predictions)
    tuned_threshold, tuned_accuracy = best_accuracy_threshold(y_val, val_probabilities)
    tuned_predictions = (val_probabilities >= tuned_threshold).astype(int)

    print("\nValidation results:")
    print(f"Default threshold accuracy: {default_accuracy:.5f}")
    print(f"Best validation threshold: {tuned_threshold:.3f}")
    print(f"Tuned threshold accuracy:  {tuned_accuracy:.5f}")
    print(f"Best iteration: {model.get_best_iteration()}")

    print("\nClassification report at tuned threshold:")
    print(classification_report(y_val, tuned_predictions, target_names=["Not closed in 24h", "Closed in 24h"]))

    print("Confusion matrix at tuned threshold:")
    print(confusion_matrix(y_val, tuned_predictions))

    save_feature_importance(model, train_pool)

    if args.skip_final:
        return

    best_iteration = model.get_best_iteration()
    final_iterations = int(best_iteration) + 1 if best_iteration is not None and best_iteration > 0 else args.iterations
    final_model = build_model(
        iterations=final_iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
    )

    full_train_pool = Pool(X, y, cat_features=CATEGORICAL_COLS)
    test_pool = Pool(X_test, cat_features=CATEGORICAL_COLS)
    final_model.fit(full_train_pool)

    test_probabilities = final_model.predict_proba(test_pool)[:, 1]
    test_predictions = (test_probabilities >= tuned_threshold).astype(int)

    if len(test_predictions) != len(sample_submission):
        raise ValueError(
            f"Prediction length mismatch: got {len(test_predictions)}, "
            f"sample submission has {len(sample_submission)} rows."
        )

    submission = sample_submission.copy()
    submission["prediction"] = test_predictions
    submission = submission[sample_submission.columns]

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / "catboost_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSaved final CatBoost submission to: {submission_path}")
    print("\nSubmission preview:")
    print(submission.head())
    print("\nPrediction distribution:")
    print(submission["prediction"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
