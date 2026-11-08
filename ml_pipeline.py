#!/usr/bin/env python3

"""
ML Pipeline — goal_after_87 Prediction
======================================

Comparative Random Forest experiment for predicting whether a goal
will be scored after the 87th minute.

TRAINING DISTRIBUTION EXPERIMENT:
    1. 73/27 = Real training distribution
    2. 50/50 = Balanced training set
    3. 67/33 = Intermediate training set

IMPORTANT:
- Features must be available before or at minute 87.
- No post-87 / full-time information must be used as input.
- Missing-value imputation is fitted ONLY on each model's training data.
- The test set is created ONCE and is identical for all experiments.
- The test set is NEVER resampled.
- The test set keeps the real-world class distribution.
- Only the training set is resampled for the 50/50 and 67/33 experiments.
- Random Forest class_weight is intentionally set to None so that the
  experiment isolates the effect of training-set class distribution.
- All experiments use the same model hyperparameters.
- The trained pipelines can be saved and reused for future predictions.

Usage:

    python3 ml_pipeline.py --input ML_Dataset_PreMatch_Live87.csv

Recommended baseline:

    python3 ml_pipeline.py \
        --input ML_Dataset_PreMatch_Live87.csv \
        --n-estimators 100 \
        --max-depth 10

Optional chronological evaluation:

    python3 ml_pipeline.py \
        --input ML_Dataset_PreMatch_Live87.csv \
        --n-estimators 100 \
        --max-depth 10 \
        --split chronological

Outputs:

    outputs/
        comparison_metrics.csv
        comparison_metrics.json

        73_27/
            goal_after_87_random_forest.joblib
            metrics.json
            confusion_matrix.csv
            feature_importance.csv
            predictions.csv
            classification_report.txt

        50_50/
            goal_after_87_random_forest.joblib
            metrics.json
            confusion_matrix.csv
            feature_importance.csv
            predictions.csv
            classification_report.txt

        67_33/
            goal_after_87_random_forest.joblib
            metrics.json
            confusion_matrix.csv
            feature_importance.csv
            predictions.csv
            classification_report.txt

Designed to be swappable with:
    - XGBoost
    - LightGBM
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# =====================================================================
# FEATURE DEFINITIONS
# =====================================================================

FEATURE_COLS = [
    # ---------------------------------------------------------------
    # Pre-Match: standings
    # ---------------------------------------------------------------
    "home_team_rank",
    "away_team_rank",
    "home_team_points_standing",
    "away_team_points_standing",
    "home_team_matches_played",
    "away_team_matches_played",
    "home_team_wins",
    "away_team_wins",
    "home_team_draws",
    "away_team_draws",
    "home_team_losses",
    "away_team_losses",
    "home_team_goals_for",
    "away_team_goals_for",
    "home_team_goals_against",
    "away_team_goals_against",
    "home_team_goal_difference",
    "away_team_goal_difference",

    # ---------------------------------------------------------------
    # Pre-Match: position & points
    # ---------------------------------------------------------------
    "home_position",
    "away_position",
    "home_points",
    "away_points",

    # ---------------------------------------------------------------
    # Pre-Match: form
    # ---------------------------------------------------------------
    "home_form5",
    "away_form5",
    "home_form10",
    "away_form10",
    "home_scored_freq",
    "away_scored_freq",
    "home_wins_freq",
    "away_wins_freq",

    # ---------------------------------------------------------------
    # Pre-Match: averages
    # ---------------------------------------------------------------
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",
    "home_avg_xg",
    "away_avg_xg",

    # ---------------------------------------------------------------
    # Pre-Match: frequencies
    # ---------------------------------------------------------------
    "home_btts_freq",
    "away_btts_freq",
    "home_over25_freq",
    "away_over25_freq",

    # ---------------------------------------------------------------
    # Pre-Match: H2H
    # ---------------------------------------------------------------
    "h2h5_over25_freq",
    "h2h5_btts_freq",

    # ---------------------------------------------------------------
    # Pre-Match: league rates
    # ---------------------------------------------------------------
    "league_home_over25_freq",
    "league_away_over25_freq",
    "league_home_over05_freq",
    "league_away_over05_freq",
    "league_home_btts_freq",
    "league_away_btts_freq",
    "league_2nd_half_over15_freq",

    # ---------------------------------------------------------------
    # Pre-Match: odds
    # ---------------------------------------------------------------
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",

    # ---------------------------------------------------------------
    # Live at 87'
    # ---------------------------------------------------------------
    "score_87_home",
    "score_87_away",
    "leading_team_87",
    "goal_diff_87",
    "is_draw_87",

    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",

    "home_corners_87",
    "away_corners_87",
    "home_fouls_87",
    "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87",
    "away_red_cards_87",
    "home_offsides_87",
    "away_offsides_87",

    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",

    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",

    "minutes_since_last_goal",
    "minutes_since_last_shot",
    "minutes_since_last_shot_on_target",
    "minutes_since_last_corner",
]


CATEGORICAL_COLS = [
    "leading_team_87",
    "is_draw_87",
]

TARGET = "goal_after_87"

MATCH_ID_COL = "match_id"

RANDOM_STATE = 42


# =====================================================================
# TRAINING DISTRIBUTIONS
# =====================================================================

EXPERIMENTS = {
    "73_27": {
        "label": "73/27 — Real Distribution",
        "description": (
            "Uses the complete training set without resampling."
        ),
    },

    "50_50": {
        "label": "50/50 — Balanced",
        "description": (
            "Keeps all positive samples and randomly samples "
            "the same number of negative samples."
        ),
    },

    "67_33": {
        "label": "67/33 — Intermediate",
        "description": (
            "Keeps all positive samples and randomly samples "
            "two negative samples per positive sample."
        ),
    },
}


# =====================================================================
# DATA LOADING
# =====================================================================

def load_dataset(filepath):
    """
    Load CSV or Excel dataset.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".xlsx":
        df = pd.read_excel(filepath)

    elif extension == ".csv":
        df = pd.read_csv(filepath)

    else:
        raise ValueError(
            "Unsupported file format. Use .csv or .xlsx"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


# =====================================================================
# DATA VALIDATION
# =====================================================================

def validate_dataset(df):
    """
    Verify that all required columns exist.
    """

    required_cols = set(
        FEATURE_COLS + [
            TARGET,
            MATCH_ID_COL,
        ]
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        print("\nERROR: Missing required columns:")

        for col in missing_cols:
            print(f"  - {col}")

        raise ValueError(
            f"{len(missing_cols)} required column(s) missing."
        )

    # ---------------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------------

    target_numeric = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if target_numeric.isna().any():

        raise ValueError(
            f"Target '{TARGET}' contains missing or invalid values."
        )

    invalid_target_values = set(
        target_numeric.unique()
    ) - {0, 1}

    if invalid_target_values:

        raise ValueError(
            f"Target '{TARGET}' must contain only 0/1. "
            f"Invalid values: {invalid_target_values}"
        )

    df[TARGET] = target_numeric.astype(int)

    # ---------------------------------------------------------------
    # Verify both classes exist
    # ---------------------------------------------------------------

    if df[TARGET].nunique() < 2:

        raise ValueError(
            "The target contains only one class. "
            "Both 0 and 1 are required for training."
        )

    return df


# =====================================================================
# FEATURE PREPROCESSING
# =====================================================================

def preprocess_raw_features(df):
    """
    Prepare categorical columns.

    IMPORTANT:
    Missing numeric values are NOT imputed here.

    Imputation is deliberately performed inside the sklearn Pipeline
    after the train/test split to avoid data leakage.
    """

    df = df.copy()

    # ---------------------------------------------------------------
    # leading_team_87
    # ---------------------------------------------------------------

    leading_map = {
        "Home": 0,
        "Away": 1,
        "Draw": 2,
    }

    if df["leading_team_87"].dtype == object:

        df["leading_team_87"] = (
            df["leading_team_87"]
            .astype(str)
            .str.strip()
            .map(leading_map)
        )

    else:

        df["leading_team_87"] = pd.to_numeric(
            df["leading_team_87"],
            errors="coerce",
        )

    df.loc[
        ~df["leading_team_87"].isin([0, 1, 2]),
        "leading_team_87",
    ] = np.nan

    # ---------------------------------------------------------------
    # is_draw_87
    # ---------------------------------------------------------------

    if df["is_draw_87"].dtype == object:

        draw_map = {
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "Yes": 1,
            "No": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
        }

        df["is_draw_87"] = (
            df["is_draw_87"]
            .astype(str)
            .str.strip()
            .map(draw_map)
        )

    else:

        df["is_draw_87"] = pd.to_numeric(
            df["is_draw_87"],
            errors="coerce",
        )

    df.loc[
        ~df["is_draw_87"].isin([0, 1]),
        "is_draw_87",
    ] = np.nan

    # ---------------------------------------------------------------
    # Numeric features
    # ---------------------------------------------------------------

    for col in FEATURE_COLS:

        if col in CATEGORICAL_COLS:
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# =====================================================================
# MODEL CREATION
# =====================================================================

def create_model(
    n_estimators=100,
    max_depth=10,
):
    """
    Create a leakage-safe Random Forest pipeline.

    class_weight is intentionally None.

    The purpose of this experiment is to isolate the effect of
    training-set class distribution:

        73/27 vs 50/50 vs 67/33

    Using class_weight='balanced' would introduce a second form
    of class reweighting and make the comparison less clean.

    SimpleImputer is inside the pipeline, meaning:
    - fit() calculates medians only from the training data;
    - test data is transformed using training medians.
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )

    return model


# =====================================================================
# TRAIN / TEST SPLIT
# =====================================================================

def split_dataset(
    df,
    test_size=0.20,
    split_method="random",
):
    """
    Split dataset ONCE.

    This function creates the common test set that will be used
    by all three experiments.

    random:
        Stratified random split.

    chronological:
        Chronological split based on a date column.

    Chronological split requires one of:
        date
        match_date
        game_date
        fixture_date
        kickoff
        kickoff_time
    """

    X = df[FEATURE_COLS]
    y = df[TARGET]
    match_ids = df[MATCH_ID_COL]

    # ---------------------------------------------------------------
    # RANDOM SPLIT
    # ---------------------------------------------------------------

    if split_method == "random":

        (
            X_train,
            X_test,
            y_train,
            y_test,
            ids_train,
            ids_test,
        ) = train_test_split(
            X,
            y,
            match_ids,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            ids_train,
            ids_test,
        )

    # ---------------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ---------------------------------------------------------------

    date_candidates = [
        "date",
        "match_date",
        "game_date",
        "fixture_date",
        "kickoff",
        "kickoff_time",
    ]

    date_col = None

    for candidate in date_candidates:

        if candidate in df.columns:

            date_col = candidate
            break

    if date_col is None:

        raise ValueError(
            "Chronological split requested, but no date column "
            "was found.\n"
            f"Expected one of: {', '.join(date_candidates)}"
        )

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    if dates.isna().all():

        raise ValueError(
            f"Date column '{date_col}' could not be parsed."
        )

    temp = df.copy()
    temp["_parsed_date"] = dates

    # Put valid dates first and fail if dates are missing.
    if temp["_parsed_date"].isna().any():

        missing_dates = int(
            temp["_parsed_date"].isna().sum()
        )

        raise ValueError(
            f"Chronological split cannot be performed because "
            f"{missing_dates:,} row(s) have an invalid/missing date "
            f"in '{date_col}'."
        )

    temp = temp.sort_values(
        "_parsed_date"
    ).reset_index(drop=True)

    split_index = int(
        len(temp) * (1 - test_size)
    )

    if split_index <= 0 or split_index >= len(temp):

        raise ValueError(
            "Invalid chronological split. "
            "Adjust --test-size."
        )

    train_df = temp.iloc[:split_index]
    test_df = temp.iloc[split_index:]

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    ids_train = train_df[MATCH_ID_COL]
    ids_test = test_df[MATCH_ID_COL]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
    )


# =====================================================================
# TRAINING DATA RESAMPLING
# =====================================================================

def prepare_training_distribution(
    X_train,
    y_train,
    ids_train,
    experiment_name,
):
    """
    Create the training dataset for one experiment.

    IMPORTANT:
    - The test set is never touched here.
    - All positive training examples are retained.
    - Only negative training examples are downsampled.
    - Sampling is deterministic using RANDOM_STATE.

    73/27:
        Use all training data.

    50/50:
        Keep all positives and sample exactly the same number
        of negatives.

    67/33:
        Keep all positives and sample two negatives per positive.
    """

    train_df = X_train.copy()

    train_df["_target"] = y_train.values
    train_df["_match_id"] = ids_train.values

    positive_df = train_df[
        train_df["_target"] == 1
    ].copy()

    negative_df = train_df[
        train_df["_target"] == 0
    ].copy()

    positive_count = len(positive_df)
    negative_count = len(negative_df)

    if positive_count == 0 or negative_count == 0:

        raise ValueError(
            "Training data must contain both positive and negative "
            "classes."
        )

    # ---------------------------------------------------------------
    # 73/27 — Real distribution
    # ---------------------------------------------------------------

    if experiment_name == "73_27":

        sampled_df = train_df.copy()

    # ---------------------------------------------------------------
    # 50/50 — Balanced
    # ---------------------------------------------------------------

    elif experiment_name == "50_50":

        required_negatives = positive_count

        if negative_count < required_negatives:

            raise ValueError(
                "Not enough negative samples to create a 50/50 "
                "training dataset."
            )

        sampled_negative_df = negative_df.sample(
            n=required_negatives,
            random_state=RANDOM_STATE,
        )

        sampled_df = pd.concat(
            [
                positive_df,
                sampled_negative_df,
            ],
            axis=0,
        )

    # ---------------------------------------------------------------
    # 67/33 — 2 negatives per positive
    # ---------------------------------------------------------------

    elif experiment_name == "67_33":

        required_negatives = positive_count * 2

        if negative_count < required_negatives:

            raise ValueError(
                "Not enough negative samples to create the "
                "67/33 training dataset."
            )

        sampled_negative_df = negative_df.sample(
            n=required_negatives,
            random_state=RANDOM_STATE,
        )

        sampled_df = pd.concat(
            [
                positive_df,
                sampled_negative_df,
            ],
            axis=0,
        )

    else:

        raise ValueError(
            f"Unknown experiment: {experiment_name}"
        )

    # ---------------------------------------------------------------
    # Shuffle training data
    # ---------------------------------------------------------------

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    X_sampled = sampled_df[
        FEATURE_COLS
    ]

    y_sampled = sampled_df[
        "_target"
    ].astype(int)

    ids_sampled = sampled_df[
        "_match_id"
    ]

    return (
        X_sampled,
        y_sampled,
        ids_sampled,
    )


# =====================================================================
# EVALUATION
# =====================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    ids_test,
    threshold=0.50,
):
    """
    Evaluate the trained model on the SAME untouched test set.
    """

    # ---------------------------------------------------------------
    # Probability of class 1
    # ---------------------------------------------------------------

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------------
    # Custom threshold
    # ---------------------------------------------------------------

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                y_prob,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_test,
                y_prob,
            )
        ),

        "threshold": float(
            threshold
        ),
    }

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No goal after 87",
            "Goal after 87",
        ],
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "match_id": ids_test.values,
            "probability": y_prob,
            "prediction": y_pred,
            "actual": y_test.values,
        }
    )

    return (
        metrics,
        cm,
        report,
        predictions,
    )


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def get_feature_importance(model):
    """
    Extract Random Forest feature importances.
    """

    classifier = model.named_steps[
        "classifier"
    ]

    importances = (
        classifier.feature_importances_
    )

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": importances,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance.insert(
        0,
        "rank",
        range(
            1,
            len(feature_importance) + 1,
        ),
    )

    return feature_importance


# =====================================================================
# SAVE OUTPUTS
# =====================================================================

def save_outputs(
    model,
    metrics,
    cm,
    report,
    predictions,
    feature_importance,
    output_dir,
):
    """
    Save model and evaluation results for one experiment.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model_path = os.path.join(
        output_dir,
        "goal_after_87_random_forest.joblib",
    )

    joblib.dump(
        model,
        model_path,
    )

    # ---------------------------------------------------------------
    # Metrics JSON
    # ---------------------------------------------------------------

    metrics_path = os.path.join(
        output_dir,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual_0",
            "Actual_1",
        ],
        columns=[
            "Predicted_0",
            "Predicted_1",
        ],
    )

    cm_path = os.path.join(
        output_dir,
        "confusion_matrix.csv",
    )

    cm_df.to_csv(
        cm_path
    )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    feature_path = os.path.join(
        output_dir,
        "feature_importance.csv",
    )

    feature_importance.to_csv(
        feature_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    predictions_path = os.path.join(
        output_dir,
        "predictions.csv",
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report_path = os.path.join(
        output_dir,
        "classification_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    return {
        "model": model_path,
        "metrics": metrics_path,
        "confusion_matrix": cm_path,
        "feature_importance": feature_path,
        "predictions": predictions_path,
        "classification_report": report_path,
    }


# =====================================================================
# SAVE COMPARISON
# =====================================================================

def save_comparison(
    comparison_results,
    output_dir,
):
    """
    Save a single comparison table containing all experiments.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------------

    rows = []

    for experiment_name, result in comparison_results.items():

        metrics = result["metrics"]

        row = {
            "experiment": experiment_name,
            "label": result["label"],

            "train_samples": result[
                "train_samples"
            ],

            "train_class_0": result[
                "train_class_0"
            ],

            "train_class_1": result[
                "train_class_1"
            ],

            "train_positive_rate": result[
                "train_positive_rate"
            ],

            "test_samples": result[
                "test_samples"
            ],

            "test_class_0": result[
                "test_class_0"
            ],

            "test_class_1": result[
                "test_class_1"
            ],

            "test_positive_rate": result[
                "test_positive_rate"
            ],

            "accuracy": metrics[
                "accuracy"
            ],

            "precision": metrics[
                "precision"
            ],

            "recall": metrics[
                "recall"
            ],

            "f1": metrics[
                "f1"
            ],

            "roc_auc": metrics[
                "roc_auc"
            ],

            "pr_auc": metrics[
                "pr_auc"
            ],

            "threshold": metrics[
                "threshold"
            ],
        }

        rows.append(row)

    comparison_df = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    csv_path = os.path.join(
        output_dir,
        "comparison_metrics.csv",
    )

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # JSON
    # ---------------------------------------------------------------

    json_data = {}

    for experiment_name, result in comparison_results.items():

        json_data[experiment_name] = {
            "label": result["label"],
            "description": result["description"],
            "train_samples": result[
                "train_samples"
            ],
            "train_class_0": result[
                "train_class_0"
            ],
            "train_class_1": result[
                "train_class_1"
            ],
            "train_positive_rate": result[
                "train_positive_rate"
            ],
            "test_samples": result[
                "test_samples"
            ],
            "test_class_0": result[
                "test_class_0"
            ],
            "test_class_1": result[
                "test_class_1"
            ],
            "test_positive_rate": result[
                "test_positive_rate"
            ],
            "metrics": result["metrics"],
        }

    json_path = os.path.join(
        output_dir,
        "comparison_metrics.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=4,
        )

    return {
        "csv": csv_path,
        "json": json_path,
        "dataframe": comparison_df,
    }


# =====================================================================
# PRINT EXPERIMENT SUMMARY
# =====================================================================

def print_experiment_result(
    experiment_name,
    result,
):
    """
    Print one experiment's results.
    """

    metrics = result["metrics"]

    print(
        "\n" + "=" * 70
    )

    print(
        f"RESULT — {result['label']}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTraining samples: "
        f"{result['train_samples']:,}"
    )

    print(
        f"Training class 0: "
        f"{result['train_class_0']:,}"
    )

    print(
        f"Training class 1: "
        f"{result['train_class_1']:,}"
    )

    print(
        f"Training positive rate: "
        f"{result['train_positive_rate']:.2%}"
    )

    print(
        "\nTest set:"
    )

    print(
        f"  Samples:       "
        f"{result['test_samples']:,}"
    )

    print(
        f"  Class 0:       "
        f"{result['test_class_0']:,}"
    )

    print(
        f"  Class 1:       "
        f"{result['test_class_1']:,}"
    )

    print(
        f"  Positive rate: "
        f"{result['test_positive_rate']:.2%}"
    )

    print(
        "\nMetrics:"
    )

    print(
        f"  Accuracy:   "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision:  "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall:     "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1-score:   "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  ROC-AUC:    "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:     "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"  Threshold:  "
        f"{metrics['threshold']:.2f}"
    )


# =====================================================================
# PRINT COMPARISON TABLE
# =====================================================================

def print_comparison_table(
    comparison_df
):
    """
    Print a compact comparison table.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 70
    )

    display_df = comparison_df[
        [
            "experiment",
            "train_samples",
            "train_positive_rate",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].copy()

    display_df[
        "train_positive_rate"
    ] = (
        display_df[
            "train_positive_rate"
        ] * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

    for col in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:

        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}"
        )

    print()

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ML Pipeline — comparative "
            "goal_after_87 prediction"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV or Excel dataset",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of Random Forest trees",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test set proportion",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Probability threshold for prediction",
    )

    parser.add_argument(
        "--split",
        choices=[
            "random",
            "chronological",
        ],
        default="random",
        help=(
            "Train/test split method. "
            "Use chronological for future-like evaluation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Directory for comparison and "
            "model outputs"
        ),
    )

    args = parser.parse_args()

    # =================================================================
    # VALIDATE ARGUMENTS
    # =================================================================

    if not 0 < args.test_size < 1:

        raise ValueError(
            "--test-size must be between 0 and 1."
        )

    if not 0 < args.threshold < 1:

        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if args.n_estimators <= 0:

        raise ValueError(
            "--n-estimators must be greater than 0."
        )

    if args.max_depth <= 0:

        raise ValueError(
            "--max-depth must be greater than 0."
        )

    # =================================================================
    # HEADER
    # =================================================================

    print("=" * 70)
    print(
        "ML PIPELINE — GOAL_AFTER_87"
    )
    print(
        "COMPARATIVE TRAINING DISTRIBUTION EXPERIMENT"
    )
    print("=" * 70)

    print("\nExperiments:")

    print(
        "  1. 73/27 — Real training distribution"
    )

    print(
        "  2. 50/50 — Balanced training distribution"
    )

    print(
        "  3. 67/33 — Intermediate training distribution"
    )

    print("\nModel:")
    print(
        "  Random Forest Classifier"
    )

    print("\nConfiguration:")

    print(
        f"  Trees:                 "
        f"{args.n_estimators}"
    )

    print(
        f"  Max depth:             "
        f"{args.max_depth}"
    )

    print(
        f"  Test size:             "
        f"{args.test_size:.0%}"
    )

    print(
        f"  Split:                 "
        f"{args.split}"
    )

    print(
        f"  Prediction threshold:  "
        f"{args.threshold:.2f}"
    )

    print(
        f"  Random state:          "
        f"{RANDOM_STATE}"
    )

    print(
        "\nclass_weight: None"
    )

    print(
        "Reason: isolate the effect of training-set "
        "class distribution."
    )

    # =================================================================
    # LOAD DATA
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "1. LOADING DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"Dataset: {args.input}"
    )

    df = load_dataset(
        args.input
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # =================================================================
    # VALIDATE
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "2. VALIDATING DATASET"
    )

    print(
        "-" * 70
    )

    df = validate_dataset(
        df
    )

    print(
        f"Required features: "
        f"{len(FEATURE_COLS)}"
    )

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Target rate: "
        f"{df[TARGET].mean():.2%}"
    )

    target_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    print(
        f"Class 0: "
        f"{target_counts.get(0, 0):,}"
    )

    print(
        f"Class 1: "
        f"{target_counts.get(1, 0):,}"
    )

    # =================================================================
    # PREPROCESS RAW FEATURES
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "3. PREPROCESSING"
    )

    print(
        "-" * 70
    )

    df = preprocess_raw_features(
        df
    )

    print(
        "Categorical variables encoded."
    )

    print(
        "Numeric missing values will be imputed "
        "inside each training pipeline."
    )

    print(
        "✓ No dataset-wide median imputation."
    )

    # =================================================================
    # COMMON TRAIN / TEST SPLIT
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "4. COMMON TRAIN / TEST SPLIT"
    )

    print(
        "-" * 70
    )

    print(
        "Creating ONE test set shared by all experiments."
    )

    print(
        "The test set will NOT be resampled."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
    ) = split_dataset(
        df,
        test_size=args.test_size,
        split_method=args.split,
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples:  "
        f"{len(X_test):,}"
    )

    print(
        f"Train target rate: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Test target rate:  "
        f"{y_test.mean():.2%}"
    )

    train_class_counts = (
        y_train
        .value_counts()
        .sort_index()
    )

    test_class_counts = (
        y_test
        .value_counts()
        .sort_index()
    )

    print(
        "\nCommon test set:"
    )

    print(
        f"  Class 0: "
        f"{test_class_counts.get(0, 0):,}"
    )

    print(
        f"  Class 1: "
        f"{test_class_counts.get(1, 0):,}"
    )

    print(
        f"  Positive rate: "
        f"{y_test.mean():.2%}"
    )

    # =================================================================
    # RUN EXPERIMENTS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "5. RUNNING COMPARATIVE EXPERIMENTS"
    )

    print(
        "-" * 70
    )

    comparison_results = {}

    for experiment_name in EXPERIMENTS:

        experiment_info = (
            EXPERIMENTS[
                experiment_name
            ]
        )

        print(
            "\n" + "#" * 70
        )

        print(
            f"EXPERIMENT: "
            f"{experiment_info['label']}"
        )

        print(
            "#" * 70
        )

        print(
            f"\n{experiment_info['description']}"
        )

        # -------------------------------------------------------------
        # Prepare training distribution
        # -------------------------------------------------------------

        (
            X_experiment_train,
            y_experiment_train,
            ids_experiment_train,
        ) = prepare_training_distribution(
            X_train=X_train,
            y_train=y_train,
            ids_train=ids_train,
            experiment_name=experiment_name,
        )

        train_class_0 = int(
            (y_experiment_train == 0).sum()
        )

        train_class_1 = int(
            (y_experiment_train == 1).sum()
        )

        train_positive_rate = float(
            y_experiment_train.mean()
        )

        print(
            "\nTraining distribution:"
        )

        print(
            f"  Total:      "
            f"{len(y_experiment_train):,}"
        )

        print(
            f"  Class 0:    "
            f"{train_class_0:,}"
        )

        print(
            f"  Class 1:    "
            f"{train_class_1:,}"
        )

        print(
            f"  Positive:   "
            f"{train_positive_rate:.2%}"
        )

        # -------------------------------------------------------------
        # Create model
        # -------------------------------------------------------------

        print(
            "\nCreating Random Forest..."
        )

        model = create_model(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        print(
            "Training..."
        )

        model.fit(
            X_experiment_train,
            y_experiment_train,
        )

        print(
            "✓ Training complete."
        )

        # -------------------------------------------------------------
        # Evaluate on SAME test set
        # -------------------------------------------------------------

        print(
            "Evaluating on common untouched test set..."
        )

        (
            metrics,
            cm,
            report,
            predictions,
        ) = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ids_test=ids_test,
            threshold=args.threshold,
        )

        # -------------------------------------------------------------
        # Feature importance
        # -------------------------------------------------------------

        feature_importance = (
            get_feature_importance(
                model
            )
        )

        # -------------------------------------------------------------
        # Output directory
        # -------------------------------------------------------------

        experiment_output_dir = os.path.join(
            args.output_dir,
            experiment_name,
        )

        output_files = save_outputs(
            model=model,
            metrics=metrics,
            cm=cm,
            report=report,
            predictions=predictions,
            feature_importance=feature_importance,
            output_dir=experiment_output_dir,
        )

        # -------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------

        comparison_results[
            experiment_name
        ] = {
            "label": experiment_info[
                "label"
            ],

            "description": experiment_info[
                "description"
            ],

            "train_samples": len(
                y_experiment_train
            ),

            "train_class_0": train_class_0,

            "train_class_1": train_class_1,

            "train_positive_rate": (
                train_positive_rate
            ),

            "test_samples": len(
                y_test
            ),

            "test_class_0": int(
                (y_test == 0).sum()
            ),

            "test_class_1": int(
                (y_test == 1).sum()
            ),

            "test_positive_rate": float(
                y_test.mean()
            ),

            "metrics": metrics,

            "outputs": output_files,
        }

        # -------------------------------------------------------------
        # Print result
        # -------------------------------------------------------------

        print_experiment_result(
            experiment_name,
            comparison_results[
                experiment_name
            ],
        )

        print(
            "\nSaved:"
        )

        for name, path in output_files.items():

            print(
                f"  ✓ {name:<22} "
                f"{path}"
            )

    # =================================================================
    # SAVE COMPARISON
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "6. SAVING COMPARISON"
    )

    print(
        "-" * 70
    )

    comparison_output = save_comparison(
        comparison_results=comparison_results,
        output_dir=args.output_dir,
    )

    print(
        f"✓ Comparison CSV:  "
        f"{comparison_output['csv']}"
    )

    print(
        f"✓ Comparison JSON: "
        f"{comparison_output['json']}"
    )

    # =================================================================
    # PRINT FINAL COMPARISON
    # =================================================================

    print_comparison_table(
        comparison_output[
            "dataframe"
        ]
    )

    # =================================================================
    # FINAL INTERPRETATION
    # =================================================================

    comparison_df = (
        comparison_output[
            "dataframe"
        ]
    )

    best_pr_auc_row = comparison_df.loc[
        comparison_df["pr_auc"].idxmax()
    ]

    best_roc_auc_row = comparison_df.loc[
        comparison_df["roc_auc"].idxmax()
    ]

    best_f1_row = comparison_df.loc[
        comparison_df["f1"].idxmax()
    ]

    best_recall_row = comparison_df.loc[
        comparison_df["recall"].idxmax()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest PR-AUC:"
    )

    print(
        f"  {best_pr_auc_row['label']}"
        f" → {best_pr_auc_row['pr_auc']:.4f}"
    )

    print(
        "\nBest ROC-AUC:"
    )

    print(
        f"  {best_roc_auc_row['label']}"
        f" → {best_roc_auc_row['roc_auc']:.4f}"
    )

    print(
        "\nBest F1:"
    )

    print(
        f"  {best_f1_row['label']}"
        f" → {best_f1_row['f1']:.4f}"
    )

    print(
        "\nBest Recall:"
    )

    print(
        f"  {best_recall_row['label']}"
        f" → {best_recall_row['recall']:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The ranking above is descriptive only."
    )

    print(
        "The most important metrics for this project are "
        "PR-AUC, precision, recall, F1 and eventually ROI."
    )

    print(
        "Accuracy alone must NOT be used to select the model."
    )

    print(
        "\nThe test set was kept identical across all three "
        "experiments."
    )

    print(
        "Therefore differences between the models can be "
        "attributed primarily to the training distribution."
    )

    print(
        "\nThe next stage should be:"
    )

    print(
        "  1. Analyze probability distributions."
    )

    print(
        "  2. Test multiple probability thresholds."
    )

    print(
        "  3. Run chronological validation."
    )

    print(
        "  4. Select the most promising configuration."
    )

    print(
        "  5. Perform the ROI / betting backtest."
    )

    print(
        "\nDo NOT use the current test results as evidence "
        "of betting profitability."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPARATIVE PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nProcess interrupted by user."
        )

        sys.exit(1)

    except Exception as e:

        print(
            f"\n\nERROR: {e}"
        )

        sys.exit(1)