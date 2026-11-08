"""
GOAL_AFTER_87 — PROBABILITY SIGNAL ANALYSIS

Diagnostic complet des probabilités produites par les modèles.

Ce script :
- ne réentraîne aucun modèle
- ne modifie aucun dataset
- analyse les predictions.csv existants
- analyse les tranches de probabilité
- calcule taux réel de goal_after_87
- calcule lift
- mesure la stabilité statistique
- mesure la discrimination
- produit une conclusion automatique

Inputs:
    outputs/73_27/predictions.csv
    outputs/50_50/predictions.csv
    outputs/67_33/predictions.csv

Outputs:
    outputs/probability_signal_analysis/
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = {
    "73_27": Path("outputs/73_27/predictions.csv"),
    "50_50": Path("outputs/50_50/predictions.csv"),
    "67_33": Path("outputs/67_33/predictions.csv"),
}

OUTPUT_DIR = Path("outputs/probability_signal_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BINS = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01),
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):
    """Convert numpy values to native Python float."""
    if pd.isna(value):
        return None
    return float(value)


def wilson_interval(successes, total, z=1.96):
    """
    Wilson 95% confidence interval for a proportion.
    More reliable than normal approximation for small samples.
    """
    if total == 0:
        return None, None

    p = successes / total

    denominator = 1 + (z ** 2) / total
    center = (
        p
        + (z ** 2) / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p) / total
            )
            + (z ** 2) / (4 * total ** 2))
        / denominator
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def get_stability_label(n):
    """Classify statistical reliability based on sample size."""
    if n >= 500:
        return "Very strong"
    if n >= 200:
        return "Strong"
    if n >= 100:
        return "Moderate"
    if n >= 50:
        return "Weak"
    return "Very weak"


def monotonicity_score(rates):
    """
    Measure whether observed goal rates generally increase
    as predicted probability increases.

    Returns percentage of valid adjacent comparisons
    where the next bin is >= previous bin.
    """
    clean = [
        rate
        for rate in rates
        if rate is not None
    ]

    if len(clean) < 2:
        return None

    comparisons = len(clean) - 1
    increasing = sum(
        clean[i + 1] >= clean[i]
        for i in range(comparisons)
    )

    return increasing / comparisons


def get_signal_assessment(
    roc_auc,
    pr_auc,
    monotonicity,
    high_lift,
    high_n,
    spearman,
):
    """
    Conservative automatic assessment.

    This is a diagnostic, not a betting recommendation.
    """

    evidence = 0

    if roc_auc is not None and roc_auc >= 0.60:
        evidence += 1

    if pr_auc is not None and pr_auc >= 0.35:
        evidence += 1

    if monotonicity is not None and monotonicity >= 0.70:
        evidence += 1

    if high_lift is not None and high_lift >= 1.25:
        evidence += 1

    if high_n >= 100:
        evidence += 1

    if spearman is not None and spearman >= 0.10:
        evidence += 1

    if evidence >= 5:
        return "STRONG SIGNAL"

    if evidence >= 3:
        return "POTENTIAL SIGNAL"

    if evidence >= 1:
        return "WEAK SIGNAL"

    return "NO EXPLOITABLE SIGNAL"


# ============================================================================
# LOAD DATA
# ============================================================================

def load_predictions(name, path):
    print()
    print("=" * 70)
    print(f"LOADING {name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required = {
        "match_id",
        "probability",
        "prediction",
        "actual",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns: {sorted(missing)}"
        )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "probability",
            "actual",
            "prediction",
        ]
    ).copy()

    removed = before - len(df)

    if removed > 0:
        print(f"Removed invalid rows: {removed}")

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"{name}: probabilities outside [0, 1]."
        )

    if not df["actual"].isin([0, 1]).all():
        raise ValueError(
            f"{name}: actual must contain only 0/1."
        )

    print(
        f"Actual positive rate: "
        f"{df['actual'].mean() * 100:.2f}%"
    )

    print(
        f"Probability range: "
        f"{df['probability'].min():.4f} → "
        f"{df['probability'].max():.4f}"
    )

    print(
        f"Probability mean: "
        f"{df['probability'].mean():.4f}"
    )

    print(
        f"Probability median: "
        f"{df['probability'].median():.4f}"
    )

    return df


# ============================================================================
# PROBABILITY BINS
# ============================================================================

def analyze_bins(df, model_name):
    baseline = df["actual"].mean()

    rows = []

    for lower, upper in BINS:

        if upper <= 1.00:
            mask = (
                (df["probability"] >= lower)
                & (df["probability"] < upper)
            )
        else:
            mask = (
                (df["probability"] >= lower)
                & (df["probability"] <= 1.00)
            )

        subset = df.loc[mask]

        n = len(subset)

        if n == 0:
            rows.append({
                "model": model_name,
                "bin": f"{lower:.2f}-{min(upper, 1.00):.2f}",
                "lower": lower,
                "upper": min(upper, 1.00),
                "matches": 0,
                "goals": 0,
                "goal_rate": None,
                "goal_rate_pct": None,
                "ci95_lower": None,
                "ci95_upper": None,
                "ci95_lower_pct": None,
                "ci95_upper_pct": None,
                "lift": None,
                "average_probability": None,
                "median_probability": None,
                "stability": "No data",
            })
            continue

        goals = int(subset["actual"].sum())

        goal_rate = goals / n

        ci_low, ci_high = wilson_interval(
            goals,
            n,
        )

        lift = (
            goal_rate / baseline
            if baseline > 0
            else None
        )

        rows.append({
            "model": model_name,
            "bin": f"{lower:.2f}-{min(upper, 1.00):.2f}",
            "lower": lower,
            "upper": min(upper, 1.00),
            "matches": n,
            "goals": goals,
            "goal_rate": goal_rate,
            "goal_rate_pct": goal_rate * 100,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "ci95_lower_pct": ci_low * 100,
            "ci95_upper_pct": ci_high * 100,
            "lift": lift,
            "average_probability": subset["probability"].mean(),
            "median_probability": subset["probability"].median(),
            "stability": get_stability_label(n),
        })

    return pd.DataFrame(rows)


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def analyze_thresholds(df, model_name):
    baseline = df["actual"].mean()

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    rows = []

    for threshold in thresholds:

        subset = df[
            df["probability"] >= threshold
        ]

        n = len(subset)

        if n == 0:
            rows.append({
                "model": model_name,
                "threshold": threshold,
                "matches": 0,
                "selection_rate": 0,
                "goals": 0,
                "goal_rate": None,
                "lift": None,
                "average_probability": None,
                "ci95_lower": None,
                "ci95_upper": None,
                "stability": "No data",
            })
            continue

        goals = int(subset["actual"].sum())

        goal_rate = goals / n

        ci_low, ci_high = wilson_interval(
            goals,
            n,
        )

        lift = (
            goal_rate / baseline
            if baseline > 0
            else None
        )

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "matches": n,
            "selection_rate": n / len(df),
            "goals": goals,
            "goal_rate": goal_rate,
            "goal_rate_pct": goal_rate * 100,
            "lift": lift,
            "average_probability": subset["probability"].mean(),
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "ci95_lower_pct": ci_low * 100,
            "ci95_upper_pct": ci_high * 100,
            "stability": get_stability_label(n),
        })

    return pd.DataFrame(rows)


# ============================================================================
# DISCRIMINATION ANALYSIS
# ============================================================================

def analyze_discrimination(df, model_name):
    y = df["actual"]
    probability = df["probability"]

    baseline = y.mean()

    roc_auc = roc_auc_score(
        y,
        probability,
    )

    pr_auc = average_precision_score(
        y,
        probability,
    )

    brier = brier_score_loss(
        y,
        probability,
    )

    logloss = log_loss(
        y,
        probability,
    )

    pearson_corr, pearson_p = pearsonr(
        probability,
        y,
    )

    spearman_corr, spearman_p = spearmanr(
        probability,
        y,
    )

    bins_df = analyze_bins(
        df,
        model_name,
    )

    valid_rates = bins_df[
        bins_df["matches"] > 0
    ]["goal_rate"].tolist()

    monotonicity = monotonicity_score(
        valid_rates
    )

    # ------------------------------------------------------------
    # HIGH PROBABILITY GROUP
    # ------------------------------------------------------------

    high = df[
        df["probability"] >= 0.50
    ]

    high_n = len(high)

    if high_n > 0:
        high_rate = high["actual"].mean()
        high_lift = high_rate / baseline
    else:
        high_rate = None
        high_lift = None

    # ------------------------------------------------------------
    # VERY HIGH PROBABILITY GROUP
    # ------------------------------------------------------------

    very_high = df[
        df["probability"] >= 0.70
    ]

    very_high_n = len(very_high)

    if very_high_n > 0:
        very_high_rate = very_high["actual"].mean()
        very_high_lift = (
            very_high_rate / baseline
        )
    else:
        very_high_rate = None
        very_high_lift = None

    # ------------------------------------------------------------
    # LOW PROBABILITY GROUP
    # ------------------------------------------------------------

    low = df[
        df["probability"] < 0.30
    ]

    low_n = len(low)

    if low_n > 0:
        low_rate = low["actual"].mean()
        low_lift = low_rate / baseline
    else:
        low_rate = None
        low_lift = None

    # ------------------------------------------------------------
    # TOP VS BOTTOM
    # ------------------------------------------------------------

    sorted_df = df.sort_values(
        "probability"
    ).reset_index(drop=True)

    n = len(sorted_df)

    quintile_size = max(
        1,
        int(n * 0.20)
    )

    bottom20 = sorted_df.iloc[:quintile_size]
    top20 = sorted_df.iloc[-quintile_size:]

    bottom20_rate = bottom20["actual"].mean()
    top20_rate = top20["actual"].mean()

    top20_lift = (
        top20_rate / baseline
        if baseline > 0
        else None
    )

    bottom20_lift = (
        bottom20_rate / baseline
        if baseline > 0
        else None
    )

    top_bottom_ratio = (
        top20_rate / bottom20_rate
        if bottom20_rate > 0
        else None
    )

    assessment = get_signal_assessment(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        monotonicity=monotonicity,
        high_lift=high_lift,
        high_n=high_n,
        spearman=spearman_corr,
    )

    return {
        "model": model_name,
        "matches": len(df),
        "baseline_rate": baseline,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "brier_score": brier,
        "log_loss": logloss,

        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,

        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,

        "monotonicity_score": monotonicity,

        "high_50_matches": high_n,
        "high_50_goal_rate": high_rate,
        "high_50_lift": high_lift,

        "very_high_70_matches": very_high_n,
        "very_high_70_goal_rate": very_high_rate,
        "very_high_70_lift": very_high_lift,

        "low_30_matches": low_n,
        "low_30_goal_rate": low_rate,
        "low_30_lift": low_lift,

        "bottom20_matches": len(bottom20),
        "bottom20_goal_rate": bottom20_rate,
        "bottom20_lift": bottom20_lift,

        "top20_matches": len(top20),
        "top20_goal_rate": top20_rate,
        "top20_lift": top20_lift,

        "top_bottom_rate_ratio": top_bottom_ratio,

        "assessment": assessment,
    }


# ============================================================================
# PRINT MODEL REPORT
# ============================================================================

def print_model_report(
    model_name,
    df,
    bins_df,
    thresholds_df,
    metrics,
):

    print()
    print()
    print("#" * 80)
    print(f"MODEL: {model_name}")
    print("#" * 80)

    print()
    print("1. DATA")
    print("-" * 80)

    print(f"Matches:       {len(df):,}")
    print(
        f"Baseline:      "
        f"{metrics['baseline_rate'] * 100:.2f}%"
    )

    print(
        f"Probability:   "
        f"{df['probability'].min():.4f} → "
        f"{df['probability'].max():.4f}"
    )

    print(
        f"Mean prob.:    "
        f"{df['probability'].mean():.4f}"
    )

    print(
        f"Median prob.:  "
        f"{df['probability'].median():.4f}"
    )

    print()
    print("2. PROBABILITY BINS")
    print("-" * 80)

    print(
        f"{'BIN':<12}"
        f"{'MATCHES':>10}"
        f"{'GOALS':>10}"
        f"{'RATE':>12}"
        f"{'LIFT':>10}"
        f"{'95% CI':>25}"
        f"{'STABILITY':>15}"
    )

    for _, row in bins_df.iterrows():

        if row["matches"] == 0:
            print(
                f"{row['bin']:<12}"
                f"{0:>10}"
                f"{0:>10}"
                f"{'N/A':>12}"
                f"{'N/A':>10}"
                f"{'N/A':>25}"
                f"{'No data':>15}"
            )
            continue

        ci = (
            f"{row['ci95_lower_pct']:.1f}% - "
            f"{row['ci95_upper_pct']:.1f}%"
        )

        print(
            f"{row['bin']:<12}"
            f"{row['matches']:>10,}"
            f"{row['goals']:>10,}"
            f"{row['goal_rate_pct']:>11.2f}%"
            f"{row['lift']:>9.2f}x"
            f"{ci:>25}"
            f"{row['stability']:>15}"
        )

    print()
    print("3. THRESHOLD ANALYSIS")
    print("-" * 80)

    print(
        f"{'THRESHOLD':<12}"
        f"{'MATCHES':>10}"
        f"{'GOAL RATE':>15}"
        f"{'LIFT':>12}"
        f"{'95% CI':>25}"
        f"{'STABILITY':>15}"
    )

    for _, row in thresholds_df.iterrows():

        if row["matches"] == 0:
            continue

        ci = (
            f"{row['ci95_lower_pct']:.1f}% - "
            f"{row['ci95_upper_pct']:.1f}%"
        )

        print(
            f">= {row['threshold']:<8.2f}"
            f"{row['matches']:>10,}"
            f"{row['goal_rate_pct']:>14.2f}%"
            f"{row['lift']:>11.2f}x"
            f"{ci:>25}"
            f"{row['stability']:>15}"
        )

    print()
    print("4. DISCRIMINATION")
    print("-" * 80)

    print(
        f"ROC-AUC:             "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:              "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:         "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Log loss:            "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Pearson correlation: "
        f"{metrics['pearson_correlation']:.4f}"
    )

    print(
        f"Spearman correlation:"
        f" {metrics['spearman_correlation']:.4f}"
    )

    print(
        f"Monotonicity:        "
        f"{metrics['monotonicity_score'] * 100:.1f}%"
    )

    print()
    print("5. HIGH PROBABILITY GROUPS")
    print("-" * 80)

    print(
        f">= 0.50: "
        f"{metrics['high_50_matches']:,} matches | "
        f"{metrics['high_50_goal_rate'] * 100:.2f}% | "
        f"Lift {metrics['high_50_lift']:.2f}x"
        if metrics["high_50_matches"] > 0
        else ">= 0.50: no matches"
    )

    print(
        f">= 0.70: "
        f"{metrics['very_high_70_matches']:,} matches | "
        f"{metrics['very_high_70_goal_rate'] * 100:.2f}% | "
        f"Lift {metrics['very_high_70_lift']:.2f}x"
        if metrics["very_high_70_matches"] > 0
        else ">= 0.70: no matches"
    )

    print()
    print("6. TOP 20% VS BOTTOM 20%")
    print("-" * 80)

    print(
        f"Bottom 20%: "
        f"{metrics['bottom20_goal_rate'] * 100:.2f}% "
        f"({metrics['bottom20_lift']:.2f}x)"
    )

    print(
        f"Top 20%:    "
        f"{metrics['top20_goal_rate'] * 100:.2f}% "
        f"({metrics['top20_lift']:.2f}x)"
    )

    if metrics["top_bottom_rate_ratio"] is not None:
        print(
            f"Top/Bottom ratio: "
            f"{metrics['top_bottom_rate_ratio']:.2f}x"
        )

    print()
    print("7. FINAL ASSESSMENT")
    print("-" * 80)

    print(
        f">>> {metrics['assessment']}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("GOAL_AFTER_87 — PROBABILITY SIGNAL ANALYSIS")
    print("=" * 80)

    print()
    print("This script:")
    print("  - does NOT retrain models")
    print("  - does NOT modify datasets")
    print("  - analyzes existing predictions")
    print("  - measures probability discrimination")
    print("  - measures statistical stability")
    print("  - evaluates whether a useful signal exists")

    all_bins = []
    all_thresholds = []
    all_metrics = []

    loaded_data = {}

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    for model_name, path in MODELS.items():

        df = load_predictions(
            model_name,
            path,
        )

        loaded_data[model_name] = df

    # ------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------

    for model_name, df in loaded_data.items():

        print()
        print("=" * 80)
        print(f"ANALYZING {model_name}")
        print("=" * 80)

        bins_df = analyze_bins(
            df,
            model_name,
        )

        thresholds_df = analyze_thresholds(
            df,
            model_name,
        )

        metrics = analyze_discrimination(
            df,
            model_name,
        )

        all_bins.append(
            bins_df
        )

        all_thresholds.append(
            thresholds_df
        )

        all_metrics.append(
            metrics
        )

        print_model_report(
            model_name,
            df,
            bins_df,
            thresholds_df,
            metrics,
        )

    # ------------------------------------------------------------
    # COMBINE
    # ------------------------------------------------------------

    bins_final = pd.concat(
        all_bins,
        ignore_index=True,
    )

    thresholds_final = pd.concat(
        all_thresholds,
        ignore_index=True,
    )

    metrics_final = pd.DataFrame(
        all_metrics
    )

    # ------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------

    bins_path = (
        OUTPUT_DIR
        / "probability_bins_detailed.csv"
    )

    thresholds_path = (
        OUTPUT_DIR
        / "threshold_analysis_detailed.csv"
    )

    metrics_path = (
        OUTPUT_DIR
        / "model_signal_metrics.csv"
    )

    bins_final.to_csv(
        bins_path,
        index=False,
    )

    thresholds_final.to_csv(
        thresholds_path,
        index=False,
    )

    metrics_final.to_csv(
        metrics_path,
        index=False,
    )

    # ------------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------------

    ranking = metrics_final.copy()

    ranking["signal_score"] = (
        ranking["roc_auc"].fillna(0)
        + ranking["pr_auc"].fillna(0)
        + ranking["monotonicity_score"].fillna(0)
    )

    ranking = ranking.sort_values(
        "signal_score",
        ascending=False,
    )

    best_model = ranking.iloc[0]

    # ------------------------------------------------------------
    # JSON SUMMARY
    # ------------------------------------------------------------

    summary = {
        "models": {}
    }

    for _, row in metrics_final.iterrows():

        data = {}

        for key, value in row.to_dict().items():

            if isinstance(value, (np.integer,)):
                data[key] = int(value)

            elif isinstance(value, (np.floating,)):
                data[key] = float(value)

            elif pd.isna(value):
                data[key] = None

            else:
                data[key] = value

        summary["models"][row["model"]] = data

    summary["best_model_by_diagnostic_score"] = (
        best_model["model"]
    )

    summary["interpretation"] = (
        "Diagnostic only. "
        "A statistical signal does not imply betting profitability. "
        "Chronological validation and ROI backtesting are required."
    )

    json_path = (
        OUTPUT_DIR
        / "signal_analysis_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # TEXT REPORT
    # ------------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "signal_analysis_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "GOAL_AFTER_87 — PROBABILITY SIGNAL ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "This is a diagnostic analysis only.\n"
        )

        f.write(
            "No model was retrained.\n"
        )

        f.write(
            "No dataset was modified.\n\n"
        )

        for _, row in metrics_final.iterrows():

            f.write(
                f"MODEL: {row['model']}\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )

            f.write(
                f"Matches: {int(row['matches']):,}\n"
            )

            f.write(
                f"Baseline: "
                f"{row['baseline_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"ROC-AUC: "
                f"{row['roc_auc']:.4f}\n"
            )

            f.write(
                f"PR-AUC: "
                f"{row['pr_auc']:.4f}\n"
            )

            f.write(
                f"Brier score: "
                f"{row['brier_score']:.4f}\n"
            )

            f.write(
                f"Log loss: "
                f"{row['log_loss']:.4f}\n"
            )

            f.write(
                f"Pearson: "
                f"{row['pearson_correlation']:.4f}\n"
            )

            f.write(
                f"Spearman: "
                f"{row['spearman_correlation']:.4f}\n"
            )

            f.write(
                f"Monotonicity: "
                f"{row['monotonicity_score'] * 100:.1f}%\n"
            )

            f.write(
                f">=0.50 matches: "
                f"{int(row['high_50_matches']):,}\n"
            )

            f.write(
                f">=0.50 goal rate: "
                f"{row['high_50_goal_rate'] * 100:.2f}%\n"
                if row["high_50_goal_rate"] is not None
                else ">=0.50 goal rate: N/A\n"
            )

            f.write(
                f">=0.50 lift: "
                f"{row['high_50_lift']:.2f}x\n"
                if row["high_50_lift"] is not None
                else ">=0.50 lift: N/A\n"
            )

            f.write(
                f">=0.70 matches: "
                f"{int(row['very_high_70_matches']):,}\n"
            )

            f.write(
                f">=0.70 goal rate: "
                f"{row['very_high_70_goal_rate'] * 100:.2f}%\n"
                if row["very_high_70_goal_rate"] is not None
                else ">=0.70 goal rate: N/A\n"
            )

            f.write(
                f">=0.70 lift: "
                f"{row['very_high_70_lift']:.2f}x\n"
                if row["very_high_70_lift"] is not None
                else ">=0.70 lift: N/A\n"
            )

            f.write(
                f"Bottom 20% goal rate: "
                f"{row['bottom20_goal_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"Top 20% goal rate: "
                f"{row['top20_goal_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"Top/Bottom ratio: "
                f"{row['top_bottom_rate_ratio']:.2f}x\n"
                if row["top_bottom_rate_ratio"] is not None
                else "Top/Bottom ratio: N/A\n"
            )

            f.write(
                f"\nASSESSMENT: "
                f"{row['assessment']}\n\n"
            )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "A predictive signal is not equivalent to betting profitability.\n"
        )

        f.write(
            "The next rigorous step is chronological validation "
            "followed by odds-aware ROI backtesting.\n"
        )

    # ------------------------------------------------------------
    # FINAL TERMINAL SUMMARY
    # ------------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for _, row in metrics_final.iterrows():

        print()
        print(row["model"])

        print(
            f"  ROC-AUC:       "
            f"{row['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC:        "
            f"{row['pr_auc']:.4f}"
        )

        print(
            f"  Spearman:      "
            f"{row['spearman_correlation']:.4f}"
        )

        print(
            f"  Monotonicity:  "
            f"{row['monotonicity_score'] * 100:.1f}%"
        )

        print(
            f"  Top 20% rate:  "
            f"{row['top20_goal_rate'] * 100:.2f}%"
        )

        print(
            f"  Bottom 20%:    "
            f"{row['bottom20_goal_rate'] * 100:.2f}%"
        )

        print(
            f"  Top/Bottom:    "
            f"{row['top_bottom_rate_ratio']:.2f}x"
            if row["top_bottom_rate_ratio"] is not None
            else "  Top/Bottom:    N/A"
        )

        print(
            f"  Assessment:    "
            f"{row['assessment']}"
        )

    print()
    print("=" * 80)

    print(
        f"Best diagnostic model: "
        f"{best_model['model']}"
    )

    print()
    print("Saved:")
    print(
        f"  ✓ {bins_path}"
    )
    print(
        f"  ✓ {thresholds_path}"
    )
    print(
        f"  ✓ {metrics_path}"
    )
    print(
        f"  ✓ {json_path}"
    )
    print(
        f"  ✓ {report_path}"
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()