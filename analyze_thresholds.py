#!/usr/bin/env python3

"""
analyze_thresholds.py
=====================

Analyse des probabilités et des seuils pour les modèles
goal_after_87.

Ce script analyse les fichiers :

    outputs/73_27/predictions.csv
    outputs/50_50/predictions.csv
    outputs/67_33/predictions.csv

Il ne réentraîne aucun modèle.

Il analyse :

    - Precision
    - Recall
    - F1
    - Actual goal rate
    - Lift
    - Nombre de matchs sélectionnés
    - Taux de sélection
    - Probabilité moyenne
    - Performance par tranche de probabilité
    - Meilleur seuil F1
    - Seuils atteignant différentes précisions

Il crée :

    outputs/threshold_analysis/
        threshold_metrics.csv
        probability_bins.csv
        best_thresholds.csv
        precision_targets.csv
        model_comparison.csv
        analysis_summary.json
        analysis_report.txt

        plots/
            precision_vs_threshold.png
            recall_vs_threshold.png
            f1_vs_threshold.png
            actual_goal_rate_vs_threshold.png
            lift_vs_threshold.png
            probability_bins_goal_rate.png

Usage :

    python3 analyze_thresholds.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_OUTPUT_DIR = "outputs"

ANALYSIS_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "threshold_analysis",
)

PLOTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "plots",
)

MODELS = {
    "73_27": {
        "label": "73/27 — Real Distribution",
        "path": os.path.join(
            BASE_OUTPUT_DIR,
            "73_27",
            "predictions.csv",
        ),
    },

    "50_50": {
        "label": "50/50 — Balanced",
        "path": os.path.join(
            BASE_OUTPUT_DIR,
            "50_50",
            "predictions.csv",
        ),
    },

    "67_33": {
        "label": "67/33 — Intermediate",
        "path": os.path.join(
            BASE_OUTPUT_DIR,
            "67_33",
            "predictions.csv",
        ),
    },
}


THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


PROBABILITY_BINS = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]


PRECISION_TARGETS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


REQUIRED_COLUMNS = [
    "match_id",
    "probability",
    "prediction",
    "actual",
]


# =====================================================================
# LOAD PREDICTIONS
# =====================================================================

def load_predictions(model_name, model_info):
    """
    Load one predictions.csv file.
    """

    path = model_info["path"]

    print(
        f"\nLoading {model_name}:"
    )

    print(
        f"  {path}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Predictions file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:

        raise ValueError(
            f"Predictions file is empty:\n{path}"
        )

    print(
        f"  Rows: {len(df):,}"
    )

    print(
        f"  Columns: {len(df.columns):,}"
    )

    return df


# =====================================================================
# VALIDATION
# =====================================================================

def validate_predictions(df, model_name):
    """
    Validate prediction data.
    """

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{model_name}: missing columns: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------------
    # Probability
    # ---------------------------------------------------------------

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    if df["probability"].isna().any():

        raise ValueError(
            f"{model_name}: probability contains "
            "missing or invalid values."
        )

    if (
        (df["probability"] < 0).any()
        or
        (df["probability"] > 1).any()
    ):

        raise ValueError(
            f"{model_name}: probability must be "
            "between 0 and 1."
        )

    # ---------------------------------------------------------------
    # Actual
    # ---------------------------------------------------------------

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    if df["actual"].isna().any():

        raise ValueError(
            f"{model_name}: actual contains "
            "missing or invalid values."
        )

    invalid_actual = set(
        df["actual"].unique()
    ) - {0, 1}

    if invalid_actual:

        raise ValueError(
            f"{model_name}: actual must contain only "
            f"0/1. Invalid values: {invalid_actual}"
        )

    df["actual"] = df["actual"].astype(int)

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    if df["prediction"].isna().any():

        raise ValueError(
            f"{model_name}: prediction contains "
            "missing or invalid values."
        )

    invalid_prediction = set(
        df["prediction"].unique()
    ) - {0, 1}

    if invalid_prediction:

        raise ValueError(
            f"{model_name}: prediction must contain "
            f"only 0/1. Invalid values: "
            f"{invalid_prediction}"
        )

    df["prediction"] = df["prediction"].astype(int)

    return df


# =====================================================================
# THRESHOLD ANALYSIS
# =====================================================================

def analyze_thresholds(
    df,
    model_name,
):
    """
    Analyze model performance at different probability thresholds.
    """

    baseline_rate = float(
        df["actual"].mean()
    )

    rows = []

    y_true = df["actual"].values

    for threshold in THRESHOLDS:

        y_pred = (
            df["probability"] >= threshold
        ).astype(int)

        selected_mask = (
            df["probability"] >= threshold
        )

        selected_matches = int(
            selected_mask.sum()
        )

        total_matches = len(df)

        selection_rate = (
            selected_matches / total_matches
            if total_matches > 0
            else 0.0
        )

        # -----------------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        true_negative = int(cm[0, 0])
        false_positive = int(cm[0, 1])
        false_negative = int(cm[1, 0])
        true_positive = int(cm[1, 1])

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # -----------------------------------------------------------
        # Actual goal rate among selected matches
        # -----------------------------------------------------------

        if selected_matches > 0:

            selected_actual = df.loc[
                selected_mask,
                "actual",
            ]

            actual_goal_rate = float(
                selected_actual.mean()
            )

            average_probability = float(
                df.loc[
                    selected_mask,
                    "probability",
                ].mean()
            )

        else:

            actual_goal_rate = 0.0
            average_probability = 0.0

        # -----------------------------------------------------------
        # Lift
        # -----------------------------------------------------------

        if baseline_rate > 0:

            lift = (
                actual_goal_rate
                / baseline_rate
            )

        else:

            lift = 0.0

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,

                "selected_matches":
                    selected_matches,

                "selection_rate":
                    selection_rate,

                "true_positives":
                    true_positive,

                "false_positives":
                    false_positive,

                "true_negatives":
                    true_negative,

                "false_negatives":
                    false_negative,

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "f1":
                    float(f1),

                "actual_goal_rate":
                    actual_goal_rate,

                "average_probability":
                    average_probability,

                "baseline_rate":
                    baseline_rate,

                "lift_vs_baseline":
                    float(lift),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# PROBABILITY BIN ANALYSIS
# =====================================================================

def analyze_probability_bins(
    df,
    model_name,
):
    """
    Analyze actual outcomes inside probability bins.
    """

    baseline_rate = float(
        df["actual"].mean()
    )

    rows = []

    for index, (
        lower,
        upper,
    ) in enumerate(PROBABILITY_BINS):

        # -----------------------------------------------------------
        # Last bin includes 1.00
        # -----------------------------------------------------------

        if upper == 1.00:

            mask = (
                (df["probability"] >= lower)
                &
                (df["probability"] <= upper)
            )

        else:

            mask = (
                (df["probability"] >= lower)
                &
                (df["probability"] < upper)
            )

        bin_df = df.loc[mask]

        matches = len(bin_df)

        if matches > 0:

            actual_goals = int(
                bin_df["actual"].sum()
            )

            actual_goal_rate = float(
                bin_df["actual"].mean()
            )

            average_probability = float(
                bin_df["probability"].mean()
            )

            lift = (
                actual_goal_rate
                / baseline_rate
                if baseline_rate > 0
                else 0.0
            )

        else:

            actual_goals = 0
            actual_goal_rate = 0.0
            average_probability = 0.0
            lift = 0.0

        rows.append(
            {
                "model": model_name,

                "bin_index": index + 1,

                "probability_bin":
                    f"{lower:.2f}-{upper:.2f}",

                "lower_bound":
                    lower,

                "upper_bound":
                    upper,

                "matches":
                    matches,

                "actual_goals_after_87":
                    actual_goals,

                "actual_goal_rate":
                    actual_goal_rate,

                "average_probability":
                    average_probability,

                "baseline_rate":
                    baseline_rate,

                "lift_vs_baseline":
                    float(lift),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# BEST THRESHOLDS
# =====================================================================

def find_best_thresholds(
    threshold_df,
    model_name,
):
    """
    Find best thresholds for F1, precision and recall.
    """

    model_df = threshold_df[
        threshold_df["model"] == model_name
    ].copy()

    if model_df.empty:

        raise ValueError(
            f"No threshold data for {model_name}"
        )

    # ---------------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------------

    best_f1 = model_df.loc[
        model_df["f1"].idxmax()
    ]

    # ---------------------------------------------------------------
    # Best precision
    # ---------------------------------------------------------------

    best_precision = model_df.loc[
        model_df["precision"].idxmax()
    ]

    # ---------------------------------------------------------------
    # Best recall
    # ---------------------------------------------------------------

    best_recall = model_df.loc[
        model_df["recall"].idxmax()
    ]

    return {
        "model": model_name,

        "best_f1_threshold":
            float(best_f1["threshold"]),

        "best_f1":
            float(best_f1["f1"]),

        "best_f1_precision":
            float(best_f1["precision"]),

        "best_f1_recall":
            float(best_f1["recall"]),

        "best_f1_actual_goal_rate":
            float(best_f1["actual_goal_rate"]),

        "best_f1_lift":
            float(best_f1["lift_vs_baseline"]),

        "best_f1_selected_matches":
            int(best_f1["selected_matches"]),

        "best_precision_threshold":
            float(best_precision["threshold"]),

        "best_precision":
            float(best_precision["precision"]),

        "best_precision_recall":
            float(best_precision["recall"]),

        "best_precision_f1":
            float(best_precision["f1"]),

        "best_recall_threshold":
            float(best_recall["threshold"]),

        "best_recall":
            float(best_recall["recall"]),

        "best_recall_precision":
            float(best_recall["precision"]),

        "best_recall_f1":
            float(best_recall["f1"]),
    }


# =====================================================================
# PRECISION TARGETS
# =====================================================================

def find_precision_targets(
    threshold_df,
    model_name,
):
    """
    Find the first threshold where each precision target is reached.
    """

    model_df = threshold_df[
        threshold_df["model"] == model_name
    ].copy()

    rows = []

    for target in PRECISION_TARGETS:

        eligible = model_df[
            model_df["precision"] >= target
        ].copy()

        if eligible.empty:

            rows.append(
                {
                    "model": model_name,
                    "precision_target": target,
                    "threshold": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "selected_matches": np.nan,
                    "actual_goal_rate": np.nan,
                    "lift_vs_baseline": np.nan,
                }
            )

            continue

        # -----------------------------------------------------------
        # Choose the LOWEST threshold that satisfies the target.
        # This maximizes coverage while respecting the precision goal.
        # -----------------------------------------------------------

        selected = eligible.sort_values(
            "threshold"
        ).iloc[0]

        rows.append(
            {
                "model": model_name,
                "precision_target": target,

                "threshold":
                    float(selected["threshold"]),

                "precision":
                    float(selected["precision"]),

                "recall":
                    float(selected["recall"]),

                "f1":
                    float(selected["f1"]),

                "selected_matches":
                    int(selected["selected_matches"]),

                "actual_goal_rate":
                    float(selected["actual_goal_rate"]),

                "lift_vs_baseline":
                    float(
                        selected[
                            "lift_vs_baseline"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# MODEL COMPARISON
# =====================================================================

def create_model_comparison(
    threshold_df,
    bins_df,
    best_threshold_list,
):
    """
    Create a high-level comparison of the models.
    """

    rows = []

    for best in best_threshold_list:

        model_name = best["model"]

        model_thresholds = threshold_df[
            threshold_df["model"] == model_name
        ]

        model_bins = bins_df[
            bins_df["model"] == model_name
        ]

        baseline_rate = (
            float(
                model_thresholds[
                    "baseline_rate"
                ].iloc[0]
            )
            if not model_thresholds.empty
            else 0.0
        )

        # -----------------------------------------------------------
        # Probability >= 0.50
        # -----------------------------------------------------------

        threshold_050 = model_thresholds[
            model_thresholds["threshold"] == 0.50
        ]

        if not threshold_050.empty:

            row_050 = threshold_050.iloc[0]

            rate_050 = float(
                row_050["actual_goal_rate"]
            )

            lift_050 = float(
                row_050["lift_vs_baseline"]
            )

        else:

            rate_050 = np.nan
            lift_050 = np.nan

        # -----------------------------------------------------------
        # Probability >= 0.70
        # -----------------------------------------------------------

        threshold_070 = model_thresholds[
            model_thresholds["threshold"] == 0.70
        ]

        if not threshold_070.empty:

            row_070 = threshold_070.iloc[0]

            rate_070 = float(
                row_070["actual_goal_rate"]
            )

            lift_070 = float(
                row_070["lift_vs_baseline"]
            )

        else:

            rate_070 = np.nan
            lift_070 = np.nan

        # -----------------------------------------------------------
        # Probability >= 0.80
        # -----------------------------------------------------------

        threshold_080 = model_thresholds[
            model_thresholds["threshold"] == 0.80
        ]

        if not threshold_080.empty:

            row_080 = threshold_080.iloc[0]

            rate_080 = float(
                row_080["actual_goal_rate"]
            )

            lift_080 = float(
                row_080["lift_vs_baseline"]
            )

        else:

            rate_080 = np.nan
            lift_080 = np.nan

        # -----------------------------------------------------------
        # Probability >= 0.90
        # -----------------------------------------------------------

        threshold_090 = model_thresholds[
            model_thresholds["threshold"] == 0.90
        ]

        if not threshold_090.empty:

            row_090 = threshold_090.iloc[0]

            rate_090 = float(
                row_090["actual_goal_rate"]
            )

            lift_090 = float(
                row_090["lift_vs_baseline"]
            )

        else:

            rate_090 = np.nan
            lift_090 = np.nan

        # -----------------------------------------------------------
        # Probability ranking signal
        # -----------------------------------------------------------

        low_bin = model_bins[
            model_bins["probability_bin"] == "0.50-0.60"
        ]

        high_bin = model_bins[
            model_bins["probability_bin"] == "0.80-0.90"
        ]

        low_rate = (
            float(low_bin["actual_goal_rate"].iloc[0])
            if not low_bin.empty
            else np.nan
        )

        high_rate = (
            float(high_bin["actual_goal_rate"].iloc[0])
            if not high_bin.empty
            else np.nan
        )

        if (
            not np.isnan(low_rate)
            and not np.isnan(high_rate)
            and high_rate > low_rate
        ):

            probability_signal = (
                "Positive ranking signal detected."
            )

        else:

            probability_signal = (
                "Weak probability ranking signal."
            )

        rows.append(
            {
                "model": model_name,

                "baseline_rate":
                    baseline_rate,

                "best_f1_threshold":
                    best["best_f1_threshold"],

                "best_f1":
                    best["best_f1"],

                "best_f1_precision":
                    best["best_f1_precision"],

                "best_f1_recall":
                    best["best_f1_recall"],

                "best_f1_actual_goal_rate":
                    best["best_f1_actual_goal_rate"],

                "best_f1_lift":
                    best["best_f1_lift"],

                "rate_probability_50_plus":
                    rate_050,

                "lift_probability_50_plus":
                    lift_050,

                "rate_probability_70_plus":
                    rate_070,

                "lift_probability_70_plus":
                    lift_070,

                "rate_probability_80_plus":
                    rate_080,

                "lift_probability_80_plus":
                    lift_080,

                "rate_probability_90_plus":
                    rate_090,

                "lift_probability_90_plus":
                    lift_090,

                "probability_signal":
                    probability_signal,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# CREATE PLOTS
# =====================================================================

def create_plots(
    threshold_df,
    bins_df,
):
    """
    Create all requested plots.

    One chart per figure.
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    # =================================================================
    # PRECISION VS THRESHOLD
    # =================================================================

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in MODELS:

        model_df = threshold_df[
            threshold_df["model"] == model_name
        ]

        plt.plot(
            model_df["threshold"],
            model_df["precision"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision vs Probability Threshold"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "precision_vs_threshold.png",
        ),
        dpi=150,
    )

    plt.close()

    # =================================================================
    # RECALL VS THRESHOLD
    # =================================================================

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in MODELS:

        model_df = threshold_df[
            threshold_df["model"] == model_name
        ]

        plt.plot(
            model_df["threshold"],
            model_df["recall"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Recall vs Probability Threshold"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "recall_vs_threshold.png",
        ),
        dpi=150,
    )

    plt.close()

    # =================================================================
    # F1 VS THRESHOLD
    # =================================================================

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in MODELS:

        model_df = threshold_df[
            threshold_df["model"] == model_name
        ]

        plt.plot(
            model_df["threshold"],
            model_df["f1"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "F1-score"
    )

    plt.title(
        "F1-score vs Probability Threshold"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "f1_vs_threshold.png",
        ),
        dpi=150,
    )

    plt.close()

    # =================================================================
    # ACTUAL GOAL RATE VS THRESHOLD
    # =================================================================

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in MODELS:

        model_df = threshold_df[
            threshold_df["model"] == model_name
        ]

        plt.plot(
            model_df["threshold"],
            model_df["actual_goal_rate"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Actual Goal After 87 Rate"
    )

    plt.title(
        "Actual Goal Rate vs Probability Threshold"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "actual_goal_rate_vs_threshold.png",
        ),
        dpi=150,
    )

    plt.close()

    # =================================================================
    # LIFT VS THRESHOLD
    # =================================================================

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in MODELS:

        model_df = threshold_df[
            threshold_df["model"] == model_name
        ]

        plt.plot(
            model_df["threshold"],
            model_df["lift_vs_baseline"],
            marker="o",
            label=model_name,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Lift vs Baseline"
    )

    plt.title(
        "Lift vs Probability Threshold"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "lift_vs_threshold.png",
        ),
        dpi=150,
    )

    plt.close()

    # =================================================================
    # ACTUAL GOAL RATE BY PROBABILITY BIN
    # =================================================================

    plt.figure(
        figsize=(12, 7)
    )

    for model_name in MODELS:

        model_df = bins_df[
            bins_df["model"] == model_name
        ]

        plt.plot(
            model_df["probability_bin"],
            model_df["actual_goal_rate"],
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Predicted Probability Bin"
    )

    plt.ylabel(
        "Actual Goal After 87 Rate"
    )

    plt.title(
        "Actual Goal Rate by Probability Bin"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "probability_bins_goal_rate.png",
        ),
        dpi=150,
    )

    plt.close()

    print(
        f"\n✓ Plots saved to: {PLOTS_DIR}"
    )


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(
    threshold_df,
    bins_df,
    best_df,
    precision_targets_df,
    comparison_df,
    summary,
):
    """
    Save all analysis files.
    """

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Threshold metrics
    # ---------------------------------------------------------------

    threshold_path = os.path.join(
        ANALYSIS_DIR,
        "threshold_metrics.csv",
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Probability bins
    # ---------------------------------------------------------------

    bins_path = os.path.join(
        ANALYSIS_DIR,
        "probability_bins.csv",
    )

    bins_df.to_csv(
        bins_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Best thresholds
    # ---------------------------------------------------------------

    best_path = os.path.join(
        ANALYSIS_DIR,
        "best_thresholds.csv",
    )

    best_df.to_csv(
        best_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Precision targets
    # ---------------------------------------------------------------

    precision_path = os.path.join(
        ANALYSIS_DIR,
        "precision_targets.csv",
    )

    precision_targets_df.to_csv(
        precision_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------------

    comparison_path = os.path.join(
        ANALYSIS_DIR,
        "model_comparison.csv",
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # JSON
    # ---------------------------------------------------------------

    json_path = os.path.join(
        ANALYSIS_DIR,
        "analysis_summary.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        f"  ✓ {threshold_path}"
    )

    print(
        f"  ✓ {bins_path}"
    )

    print(
        f"  ✓ {best_path}"
    )

    print(
        f"  ✓ {precision_path}"
    )

    print(
        f"  ✓ {comparison_path}"
    )

    print(
        f"  ✓ {json_path}"
    )


# =====================================================================
# REPORT
# =====================================================================

def generate_report(
    best_df,
    precision_targets_df,
    comparison_df,
    bins_df,
):
    """
    Generate a human-readable analysis report.
    """

    report_path = os.path.join(
        ANALYSIS_DIR,
        "analysis_report.txt",
    )

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "GOAL_AFTER_87 — THRESHOLD ANALYSIS"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "This report analyzes existing model predictions."
    )

    lines.append(
        "No model was retrained."
    )

    lines.append(
        "No dataset was modified."
    )

    lines.append("")

    # =================================================================
    # BEST THRESHOLDS
    # =================================================================

    lines.append(
        "-" * 70
    )

    lines.append(
        "BEST THRESHOLDS"
    )

    lines.append(
        "-" * 70
    )

    for _, row in best_df.iterrows():

        lines.append("")

        lines.append(
            f"Model: {row['model']}"
        )

        lines.append(
            f"Best F1 threshold: "
            f"{row['best_f1_threshold']:.2f}"
        )

        lines.append(
            f"Precision: "
            f"{row['best_f1_precision']:.4f}"
        )

        lines.append(
            f"Recall: "
            f"{row['best_f1_recall']:.4f}"
        )

        lines.append(
            f"F1: "
            f"{row['best_f1']:.4f}"
        )

        lines.append(
            f"Actual goal rate: "
            f"{row['best_f1_actual_goal_rate']:.2%}"
        )

        lines.append(
            f"Lift: "
            f"{row['best_f1_lift']:.2f}x"
        )

        lines.append(
            f"Selected matches: "
            f"{int(row['best_f1_selected_matches']):,}"
        )

    # =================================================================
    # PRECISION TARGETS
    # =================================================================

    lines.append("")

    lines.append(
        "-" * 70
    )

    lines.append(
        "PRECISION TARGETS"
    )

    lines.append(
        "-" * 70
    )

    for model_name in MODELS:

        lines.append("")

        lines.append(
            f"Model: {model_name}"
        )

        model_targets = precision_targets_df[
            precision_targets_df["model"]
            == model_name
        ]

        for _, row in model_targets.iterrows():

            target = row[
                "precision_target"
            ]

            threshold = row[
                "threshold"
            ]

            if pd.isna(threshold):

                lines.append(
                    f"  Precision >= {target:.0%}: N/A"
                )

            else:

                lines.append(
                    f"  Precision >= {target:.0%}: "
                    f"threshold {threshold:.2f}, "
                    f"precision "
                    f"{row['precision']:.2%}, "
                    f"recall "
                    f"{row['recall']:.2%}, "
                    f"F1 "
                    f"{row['f1']:.4f}, "
                    f"selected "
                    f"{int(row['selected_matches']):,}, "
                    f"lift "
                    f"{row['lift_vs_baseline']:.2f}x"
                )

    # =================================================================
    # PROBABILITY QUALITY
    # =================================================================

    lines.append("")

    lines.append(
        "-" * 70
    )

    lines.append(
        "PROBABILITY QUALITY"
    )

    lines.append(
        "-" * 70
    )

    for model_name in MODELS:

        lines.append("")

        lines.append(
            f"Model: {model_name}"
        )

        model_bins = bins_df[
            bins_df["model"]
            == model_name
        ]

        for _, row in model_bins.iterrows():

            lines.append(
                f"  {row['probability_bin']} "
                f"| matches="
                f"{int(row['matches']):,} "
                f"| actual="
                f"{row['actual_goal_rate']:.2%} "
                f"| avg_prob="
                f"{row['average_probability']:.3f} "
                f"| lift="
                f"{row['lift_vs_baseline']:.2f}x"
            )

    # =================================================================
    # INTERPRETATION
    # =================================================================

    lines.append("")

    lines.append(
        "-" * 70
    )

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "-" * 70
    )

    for _, row in comparison_df.iterrows():

        lines.append("")

        lines.append(
            f"{row['model']}:"
        )

        lines.append(
            f"  Baseline goal rate: "
            f"{row['baseline_rate']:.2%}"
        )

        lines.append(
            f"  Probability >= 0.50: "
            f"{row['rate_probability_50_plus']:.2%} "
            f"({row['lift_probability_50_plus']:.2f}x lift)"
        )

        lines.append(
            f"  Probability >= 0.70: "
            f"{row['rate_probability_70_plus']:.2%} "
            f"({row['lift_probability_70_plus']:.2f}x lift)"
        )

        lines.append(
            f"  Probability >= 0.80: "
            f"{row['rate_probability_80_plus']:.2%} "
            f"({row['lift_probability_80_plus']:.2f}x lift)"
        )

        lines.append(
            f"  Probability >= 0.90: "
            f"{row['rate_probability_90_plus']:.2%} "
            f"({row['lift_probability_90_plus']:.2f}x lift)"
        )

        lines.append(
            f"  Signal assessment: "
            f"{row['probability_signal']}"
        )

    # =================================================================
    # IMPORTANT LIMITATIONS
    # =================================================================

    lines.append("")

    lines.append(
        "-" * 70
    )

    lines.append(
        "IMPORTANT LIMITATIONS"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "1. This analysis uses the existing random test split."
    )

    lines.append(
        "2. It does not prove future performance."
    )

    lines.append(
        "3. It does not prove betting profitability."
    )

    lines.append(
        "4. Threshold optimization on the test set can overfit."
    )

    lines.append(
        "5. Chronological validation is required before final model selection."
    )

    lines.append(
        "6. Probability calibration should be considered because "
        "50/50 and 67/33 training distributions alter the class prior."
    )

    lines.append(
        "7. Final model selection must ultimately be based on "
        "out-of-sample performance and ROI."
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "END OF REPORT"
    )

    lines.append(
        "=" * 70
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"\n✓ Report saved: {report_path}"
    )


# =====================================================================
# PRINT CONSOLE SUMMARY
# =====================================================================

def print_console_summary(
    best_df,
    comparison_df,
):
    """
    Print important results in terminal.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS SUMMARY"
    )

    print(
        "=" * 70
    )

    for _, row in best_df.iterrows():

        print("")

        print(
            f"{row['model']}"
        )

        print(
            f"  Best F1 threshold: "
            f"{row['best_f1_threshold']:.2f}"
        )

        print(
            f"  Precision:          "
            f"{row['best_f1_precision']:.4f}"
        )

        print(
            f"  Recall:             "
            f"{row['best_f1_recall']:.4f}"
        )

        print(
            f"  F1:                 "
            f"{row['best_f1']:.4f}"
        )

        print(
            f"  Actual goal rate:   "
            f"{row['best_f1_actual_goal_rate']:.2%}"
        )

        print(
            f"  Lift:               "
            f"{row['best_f1_lift']:.2f}x"
        )

        print(
            f"  Selected matches:   "
            f"{int(row['best_f1_selected_matches']):,}"
        )

    print(
        "\n" + "-" * 70
    )

    print(
        "PROBABILITY QUALITY"
    )

    print(
        "-" * 70
    )

    for _, row in comparison_df.iterrows():

        print("")

        print(
            row["model"]
        )

        print(
            f"  Baseline: "
            f"{row['baseline_rate']:.2%}"
        )

        print(
            f"  >= 0.50: "
            f"{row['rate_probability_50_plus']:.2%} "
            f"| Lift {row['lift_probability_50_plus']:.2f}x"
        )

        print(
            f"  >= 0.70: "
            f"{row['rate_probability_70_plus']:.2%} "
            f"| Lift {row['lift_probability_70_plus']:.2f}x"
        )

        print(
            f"  >= 0.80: "
            f"{row['rate_probability_80_plus']:.2%} "
            f"| Lift {row['lift_probability_80_plus']:.2f}x"
        )

        print(
            f"  >= 0.90: "
            f"{row['rate_probability_90_plus']:.2%} "
            f"| Lift {row['lift_probability_90_plus']:.2f}x"
        )

        print(
            f"  Assessment: "
            f"{row['probability_signal']}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults directory:"
    )

    print(
        f"  {ANALYSIS_DIR}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GOAL_AFTER_87 — THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nThis script analyzes existing predictions."
    )

    print(
        "No model will be retrained."
    )

    print(
        "No dataset will be modified."
    )

    # =================================================================
    # CREATE DIRECTORIES
    # =================================================================

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    # =================================================================
    # LOAD ALL MODELS
    # =================================================================

    predictions = {}

    for model_name, model_info in MODELS.items():

        df = load_predictions(
            model_name,
            model_info,
        )

        df = validate_predictions(
            df,
            model_name,
        )

        predictions[
            model_name
        ] = df

    # =================================================================
    # ANALYZE THRESHOLDS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "1. THRESHOLD ANALYSIS"
    )

    print(
        "-" * 70
    )

    threshold_results = []

    for model_name, df in predictions.items():

        print(
            f"\nAnalyzing {model_name}..."
        )

        result = analyze_thresholds(
            df,
            model_name,
        )

        threshold_results.append(
            result
        )

    threshold_df = pd.concat(
        threshold_results,
        ignore_index=True,
    )

    # =================================================================
    # ANALYZE PROBABILITY BINS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "2. PROBABILITY BIN ANALYSIS"
    )

    print(
        "-" * 70
    )

    bin_results = []

    for model_name, df in predictions.items():

        print(
            f"\nAnalyzing probability bins: "
            f"{model_name}..."
        )

        result = analyze_probability_bins(
            df,
            model_name,
        )

        bin_results.append(
            result
        )

    bins_df = pd.concat(
        bin_results,
        ignore_index=True,
    )

    # =================================================================
    # BEST THRESHOLDS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "3. FINDING BEST THRESHOLDS"
    )

    print(
        "-" * 70
    )

    best_threshold_list = []

    for model_name in MODELS:

        result = find_best_thresholds(
            threshold_df,
            model_name,
        )

        best_threshold_list.append(
            result
        )

    best_df = pd.DataFrame(
        best_threshold_list
    )

    # =================================================================
    # PRECISION TARGETS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "4. PRECISION TARGET ANALYSIS"
    )

    print(
        "-" * 70
    )

    precision_target_results = []

    for model_name in MODELS:

        result = find_precision_targets(
            threshold_df,
            model_name,
        )

        precision_target_results.append(
            result
        )

    precision_targets_df = pd.concat(
        precision_target_results,
        ignore_index=True,
    )

    # =================================================================
    # MODEL COMPARISON
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "5. MODEL COMPARISON"
    )

    print(
        "-" * 70
    )

    comparison_df = create_model_comparison(
        threshold_df=threshold_df,
        bins_df=bins_df,
        best_threshold_list=best_threshold_list,
    )

    # =================================================================
    # CREATE PLOTS
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "6. CREATING PLOTS"
    )

    print(
        "-" * 70
    )

    create_plots(
        threshold_df,
        bins_df,
    )

    # =================================================================
    # CREATE SUMMARY JSON
    # =================================================================

    summary = {
        "models": {},

        "important_note": (
            "This analysis evaluates existing test predictions. "
            "It does not establish future performance or betting profitability."
        ),
    }

    for model_name in MODELS:

        best_row = best_df[
            best_df["model"] == model_name
        ].iloc[0]

        comparison_row = comparison_df[
            comparison_df["model"] == model_name
        ].iloc[0]

        model_bins = bins_df[
            bins_df["model"] == model_name
        ]

        model_thresholds = threshold_df[
            threshold_df["model"] == model_name
        ]

        summary["models"][
            model_name
        ] = {
            "best_f1": {
                "threshold":
                    float(
                        best_row[
                            "best_f1_threshold"
                        ]
                    ),

                "precision":
                    float(
                        best_row[
                            "best_f1_precision"
                        ]
                    ),

                "recall":
                    float(
                        best_row[
                            "best_f1_recall"
                        ]
                    ),

                "f1":
                    float(
                        best_row[
                            "best_f1"
                        ]
                    ),

                "actual_goal_rate":
                    float(
                        best_row[
                            "best_f1_actual_goal_rate"
                        ]
                    ),

                "lift":
                    float(
                        best_row[
                            "best_f1_lift"
                        ]
                    ),

                "selected_matches":
                    int(
                        best_row[
                            "best_f1_selected_matches"
                        ]
                    ),
            },

            "baseline_rate":
                float(
                    comparison_row[
                        "baseline_rate"
                    ]
                ),

            "probability_quality":
                comparison_row[
                    "probability_signal"
                ],

            "thresholds": (
                model_thresholds
                .replace(
                    {
                        np.nan: None
                    }
                )
                .to_dict(
                    orient="records"
                )
            ),

            "probability_bins": (
                model_bins
                .replace(
                    {
                        np.nan: None
                    }
                )
                .to_dict(
                    orient="records"
                )
            ),
        }

    # =================================================================
    # SAVE
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "7. SAVING RESULTS"
    )

    print(
        "-" * 70
    )

    save_results(
        threshold_df=threshold_df,
        bins_df=bins_df,
        best_df=best_df,
        precision_targets_df=precision_targets_df,
        comparison_df=comparison_df,
        summary=summary,
    )

    # =================================================================
    # REPORT
    # =================================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "8. GENERATING REPORT"
    )

    print(
        "-" * 70
    )

    generate_report(
        best_df=best_df,
        precision_targets_df=precision_targets_df,
        comparison_df=comparison_df,
        bins_df=bins_df,
    )

    # =================================================================
    # CONSOLE SUMMARY
    # =================================================================

    print_console_summary(
        best_df=best_df,
        comparison_df=comparison_df,
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

        raise SystemExit(1)

    except Exception as e:

        print(
            f"\n\nERROR: {e}"
        )

        raise SystemExit(1)