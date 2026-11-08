from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "model_optimization"
    / "model_optimization_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "calibration_threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "model",
    "distribution",
    "fold",
    "dataset_index",
    "probability",
    "actual",
]

TOP_PERCENTAGES = [5, 10, 15, 20, 25, 30]

THRESHOLDS = [
    round(x, 2)
    for x in np.arange(0.20, 0.81, 0.05)
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
    (0.90, 1.01),
]


# ============================================================
# UTILITAIRES
# ============================================================

def wilson_interval(successes, n, z=1.96):
    """
    Wilson confidence interval for a proportion.
    """
    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1 + (z ** 2 / n)

    center = (
        p
        + (z ** 2 / (2 * n))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p) / n
                + z ** 2 / (4 * n ** 2)
            )
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def safe_roc_auc(y_true, scores):
    """
    ROC-AUC sécurisé lorsque la cible ne contient qu'une seule classe.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, scores)


def safe_pr_auc(y_true, scores):
    """
    PR-AUC sécurisé.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, scores)


def safe_brier(y_true, scores):
    """
    Brier Score.
    """
    return brier_score_loss(y_true, scores)


def format_pct(value):
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def format_float(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("CALIBRATION & THRESHOLD ANALYSIS")
print("=" * 70)

print("\n" + "=" * 70)
print("CHARGEMENT DES PREDICTIONS")
print("=" * 70)

print(f"Fichier : {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFichier introuvable :\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Lignes : {len(df):,}")
print(f"Colonnes : {len(df.columns)}")

print("\nColonnes détectées :")
for column in df.columns:
    print(f" - {column}")


# ============================================================
# VALIDATION DU FORMAT
# ============================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nColonnes manquantes : "
        + str(missing_columns)
        + "\n\nColonnes attendues : "
        + str(REQUIRED_COLUMNS)
        + "\n\nColonnes présentes : "
        + str(list(df.columns))
    )


# ============================================================
# NETTOYAGE
# ============================================================

df = df.copy()

df["probability"] = pd.to_numeric(
    df["probability"],
    errors="coerce",
)

df["actual"] = pd.to_numeric(
    df["actual"],
    errors="coerce",
)

df["fold"] = pd.to_numeric(
    df["fold"],
    errors="coerce",
)

before = len(df)

df = df.dropna(
    subset=[
        "model",
        "distribution",
        "fold",
        "probability",
        "actual",
    ]
)

after = len(df)

if before != after:
    print(
        f"\nLignes supprimées pour valeurs invalides : "
        f"{before - after:,}"
    )

# Sécurisation des probabilités
df["probability"] = df["probability"].clip(0.0, 1.0)

df["actual"] = df["actual"].astype(int)
df["fold"] = df["fold"].astype(int)


# ============================================================
# INFORMATIONS DATASET
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURE DU DATASET")
print("=" * 70)

print(f"Nombre de modèles : {df['model'].nunique()}")
print(f"Nombre de distributions : {df['distribution'].nunique()}")
print(f"Nombre de folds : {df['fold'].nunique()}")

print(
    f"Taux global de buts : "
    f"{df['actual'].mean() * 100:.2f}%"
)

print("\nModèles :")

for model in sorted(df["model"].unique()):
    print(f" - {model}")

print("\nDistributions :")

for distribution in sorted(df["distribution"].unique()):
    print(f" - {distribution}")


# ============================================================
# 1. SCORE QUALITY
# ============================================================

print("\n" + "=" * 70)
print("1. SCORE QUALITY")
print("=" * 70)

score_quality_rows = []

groups = df.groupby(
    ["model", "distribution"],
    sort=True,
)

for (model, distribution), group in groups:

    y_true = group["actual"].values
    scores = group["probability"].values

    roc_auc = safe_roc_auc(
        y_true,
        scores,
    )

    pr_auc = safe_pr_auc(
        y_true,
        scores,
    )

    brier = safe_brier(
        y_true,
        scores,
    )

    baseline = y_true.mean()

    score_quality_rows.append(
        {
            "model": model,
            "distribution": distribution,
            "n": len(group),
            "positive_count": int(y_true.sum()),
            "negative_count": int((1 - y_true).sum()),
            "baseline_rate": baseline,
            "mean_score": scores.mean(),
            "median_score": np.median(scores),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
        }
    )

score_quality = pd.DataFrame(
    score_quality_rows
)

score_quality.to_csv(
    OUTPUT_DIR / "score_quality.csv",
    index=False,
)

print(
    score_quality[
        [
            "model",
            "distribution",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "baseline_rate",
        ]
    ]
    .sort_values(
        "roc_auc",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 2. THRESHOLDS BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("2. THRESHOLD ANALYSIS")
print("=" * 70)

threshold_rows = []

group_columns = [
    "model",
    "distribution",
    "fold",
]

for (
    model,
    distribution,
    fold,
), group in df.groupby(
    group_columns,
    sort=True,
):

    y_true = group["actual"].values
    scores = group["probability"].values

    baseline_rate = y_true.mean()

    for threshold in THRESHOLDS:

        selected = scores >= threshold

        n_selected = int(selected.sum())

        if n_selected == 0:
            goals = 0
            goal_rate = np.nan
            precision = np.nan
            recall = np.nan
            f1 = np.nan
            lift = np.nan
            ci_low = np.nan
            ci_high = np.nan

        else:

            selected_actual = y_true[selected]

            goals = int(selected_actual.sum())

            goal_rate = goals / n_selected

            precision = goal_rate

            total_goals = int(y_true.sum())

            if total_goals > 0:
                recall = goals / total_goals
            else:
                recall = np.nan

            if (
                not pd.isna(precision)
                and not pd.isna(recall)
                and (precision + recall) > 0
            ):
                f1 = (
                    2
                    * precision
                    * recall
                    / (precision + recall)
                )
            else:
                f1 = np.nan

            if baseline_rate > 0:
                lift = goal_rate / baseline_rate
            else:
                lift = np.nan

            ci_low, ci_high = wilson_interval(
                goals,
                n_selected,
            )

        threshold_rows.append(
            {
                "model": model,
                "distribution": distribution,
                "fold": fold,
                "threshold": threshold,
                "n_total": len(group),
                "n_selected": n_selected,
                "selection_rate": (
                    n_selected / len(group)
                ),
                "goals": goals,
                "goal_rate": goal_rate,
                "baseline_rate": baseline_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "lift": lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "above_baseline": (
                    goal_rate > baseline_rate
                    if not pd.isna(goal_rate)
                    else False
                ),
            }
        )

thresholds_by_fold = pd.DataFrame(
    threshold_rows
)

thresholds_by_fold.to_csv(
    OUTPUT_DIR / "thresholds_by_fold.csv",
    index=False,
)


# ============================================================
# 3. THRESHOLD SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("3. THRESHOLD SUMMARY")
print("=" * 70)

threshold_summary_rows = []

for (
    model,
    distribution,
    threshold,
), group in thresholds_by_fold.groupby(
    [
        "model",
        "distribution",
        "threshold",
    ],
    sort=True,
):

    valid = group[
        group["n_selected"] > 0
    ]

    if len(valid) == 0:
        continue

    threshold_summary_rows.append(
        {
            "model": model,
            "distribution": distribution,
            "threshold": threshold,
            "folds": len(valid),
            "mean_n_selected": valid[
                "n_selected"
            ].mean(),
            "mean_selection_rate": valid[
                "selection_rate"
            ].mean(),
            "mean_goal_rate": valid[
                "goal_rate"
            ].mean(),
            "std_goal_rate": valid[
                "goal_rate"
            ].std(),
            "min_goal_rate": valid[
                "goal_rate"
            ].min(),
            "max_goal_rate": valid[
                "goal_rate"
            ].max(),
            "mean_baseline_rate": valid[
                "baseline_rate"
            ].mean(),
            "mean_precision": valid[
                "precision"
            ].mean(),
            "mean_recall": valid[
                "recall"
            ].mean(),
            "mean_f1": valid[
                "f1"
            ].mean(),
            "mean_lift": valid[
                "lift"
            ].mean(),
            "min_lift": valid[
                "lift"
            ].min(),
            "max_lift": valid[
                "lift"
            ].max(),
            "folds_above_baseline": int(
                valid["above_baseline"].sum()
            ),
            "stability": (
                valid["above_baseline"].mean()
            ),
        }
    )

threshold_summary = pd.DataFrame(
    threshold_summary_rows
)

threshold_summary.to_csv(
    OUTPUT_DIR / "threshold_summary.csv",
    index=False,
)


# ============================================================
# 4. TOP PERCENTAGES BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("4. TOP PERCENTAGE ANALYSIS")
print("=" * 70)

top_rows = []

for (
    model,
    distribution,
    fold,
), group in df.groupby(
    group_columns,
    sort=True,
):

    group = group.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    y_true = group["actual"].values

    baseline_rate = y_true.mean()

    n_total = len(group)

    for top_percentage in TOP_PERCENTAGES:

        n_selected = max(
            1,
            int(
                np.ceil(
                    n_total
                    * top_percentage
                    / 100
                )
            ),
        )

        selected_actual = y_true[
            :n_selected
        ]

        goals = int(
            selected_actual.sum()
        )

        goal_rate = (
            goals / n_selected
        )

        lift = (
            goal_rate / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        ci_low, ci_high = (
            wilson_interval(
                goals,
                n_selected,
            )
        )

        top_rows.append(
            {
                "model": model,
                "distribution": distribution,
                "fold": fold,
                "top_percentage": top_percentage,
                "n_total": n_total,
                "n_selected": n_selected,
                "goals": goals,
                "goal_rate": goal_rate,
                "baseline_rate": baseline_rate,
                "lift": lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "above_baseline": (
                    goal_rate > baseline_rate
                ),
            }
        )

top_percentages_by_fold = pd.DataFrame(
    top_rows
)

top_percentages_by_fold.to_csv(
    OUTPUT_DIR / "top_percentages_by_fold.csv",
    index=False,
)


# ============================================================
# 5. TOP PERCENTAGE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5. TOP PERCENTAGE SUMMARY")
print("=" * 70)

top_summary_rows = []

for (
    model,
    distribution,
    top_percentage,
), group in top_percentages_by_fold.groupby(
    [
        "model",
        "distribution",
        "top_percentage",
    ],
    sort=True,
):

    top_summary_rows.append(
        {
            "model": model,
            "distribution": distribution,
            "top_percentage": top_percentage,
            "folds": len(group),
            "mean_goal_rate": group[
                "goal_rate"
            ].mean(),
            "std_goal_rate": group[
                "goal_rate"
            ].std(),
            "min_goal_rate": group[
                "goal_rate"
            ].min(),
            "max_goal_rate": group[
                "goal_rate"
            ].max(),
            "mean_baseline_rate": group[
                "baseline_rate"
            ].mean(),
            "mean_lift": group[
                "lift"
            ].mean(),
            "min_lift": group[
                "lift"
            ].min(),
            "max_lift": group[
                "lift"
            ].max(),
            "folds_above_baseline": int(
                group[
                    "above_baseline"
                ].sum()
            ),
            "stability": group[
                "above_baseline"
            ].mean(),
            "mean_selected": group[
                "n_selected"
            ].mean(),
        }
    )

top_percentages_summary = pd.DataFrame(
    top_summary_rows
)

top_percentages_summary.to_csv(
    OUTPUT_DIR / "top_percentages_summary.csv",
    index=False,
)


# ============================================================
# 6. RANKING DES TOP ZONES
# ============================================================

print("\n" + "=" * 70)
print("6. RANKING DES ZONES")
print("=" * 70)

ranking = top_percentages_summary.copy()

# Score de robustesse simple :
# priorité à la stabilité, au lift et au volume.
ranking["robustness_score"] = (
    ranking["mean_lift"]
    * ranking["stability"]
    * np.log1p(
        ranking["mean_selected"]
    )
)

ranking = ranking.sort_values(
    [
        "stability",
        "mean_lift",
        "mean_goal_rate",
    ],
    ascending=False,
)

ranking.to_csv(
    OUTPUT_DIR / "ranking_top_percentages.csv",
    index=False,
)

print(
    ranking[
        [
            "model",
            "distribution",
            "top_percentage",
            "mean_goal_rate",
            "mean_lift",
            "stability",
            "min_lift",
            "mean_selected",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 7. PROBABILITY BINS
# ============================================================

print("\n" + "=" * 70)
print("7. PROBABILITY BINS")
print("=" * 70)

calibration_rows = []

for (
    model,
    distribution,
    fold,
), group in df.groupby(
    group_columns,
    sort=True,
):

    y_true = group["actual"].values
    scores = group["probability"].values

    baseline_rate = y_true.mean()

    for (
        bin_low,
        bin_high,
    ) in PROBABILITY_BINS:

        if bin_high <= 1.0:

            mask = (
                (scores >= bin_low)
                & (scores < bin_high)
            )

        else:

            mask = (
                (scores >= bin_low)
                & (scores <= 1.0)
            )

        n = int(mask.sum())

        if n == 0:
            goals = 0
            goal_rate = np.nan
            lift = np.nan
            ci_low = np.nan
            ci_high = np.nan
            mean_score = np.nan

        else:

            bin_actual = y_true[mask]
            bin_scores = scores[mask]

            goals = int(
                bin_actual.sum()
            )

            goal_rate = (
                goals / n
            )

            lift = (
                goal_rate / baseline_rate
                if baseline_rate > 0
                else np.nan
            )

            ci_low, ci_high = (
                wilson_interval(
                    goals,
                    n,
                )
            )

            mean_score = (
                bin_scores.mean()
            )

        calibration_rows.append(
            {
                "model": model,
                "distribution": distribution,
                "fold": fold,
                "bin_low": bin_low,
                "bin_high": bin_high,
                "bin": (
                    f"{bin_low:.2f}"
                    f"-"
                    f"{min(bin_high, 1.0):.2f}"
                ),
                "n": n,
                "goals": goals,
                "goal_rate": goal_rate,
                "baseline_rate": baseline_rate,
                "lift": lift,
                "mean_score": mean_score,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

calibration_bins = pd.DataFrame(
    calibration_rows
)

calibration_bins.to_csv(
    OUTPUT_DIR / "calibration_bins.csv",
    index=False,
)


# ============================================================
# 8. CALIBRATION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("8. CALIBRATION COMPARISON")
print("=" * 70)

"""
IMPORTANT :

Les valeurs 'probability' sont produites par différents modèles
et différentes distributions d'entraînement.

On calcule donc ici :

1. Mean predicted score
2. Observed positive rate
3. Calibration gap

Ce n'est PAS une calibration statistique finale.

Le but est de diagnostiquer si les scores ressemblent
approximativement aux fréquences observées.

Une vraie calibration finale devra être faite avec une méthode
out-of-fold / temporelle, sans utiliser les mêmes données pour
apprendre puis évaluer la calibration.
"""

calibration_comparison_rows = []

for (
    model,
    distribution,
), group in df.groupby(
    [
        "model",
        "distribution",
    ],
    sort=True,
):

    scores = group["probability"].values
    actual = group["actual"].values

    mean_score = scores.mean()
    observed_rate = actual.mean()

    calibration_gap = (
        observed_rate
        - mean_score
    )

    absolute_gap = abs(
        calibration_gap
    )

    calibration_comparison_rows.append(
        {
            "model": model,
            "distribution": distribution,
            "n": len(group),
            "mean_predicted_score": mean_score,
            "observed_goal_rate": observed_rate,
            "calibration_gap": calibration_gap,
            "absolute_calibration_gap": absolute_gap,
        }
    )

calibration_comparison = pd.DataFrame(
    calibration_comparison_rows
)

calibration_comparison = (
    calibration_comparison
    .sort_values(
        "absolute_calibration_gap"
    )
)

calibration_comparison.to_csv(
    OUTPUT_DIR / "calibration_comparison.csv",
    index=False,
)


# ============================================================
# 9. RAPPORT FINAL
# ============================================================

print("\n" + "=" * 70)
print("9. GENERATION DU RAPPORT")
print("=" * 70)

report_lines = []

report_lines.append(
    "=" * 70
)

report_lines.append(
    "CALIBRATION & THRESHOLD ANALYSIS REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    f"Dataset : {INPUT_FILE}"
)

report_lines.append(
    f"Nombre de lignes : {len(df):,}"
)

report_lines.append(
    f"Nombre de modèles : {df['model'].nunique()}"
)

report_lines.append(
    f"Nombre de distributions : "
    f"{df['distribution'].nunique()}"
)

report_lines.append(
    f"Nombre de folds : {df['fold'].nunique()}"
)

report_lines.append("")

report_lines.append(
    f"Taux global de goal_after_87 : "
    f"{df['actual'].mean() * 100:.2f}%"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT :"
)

report_lines.append(
    "Les valeurs 'probability' sont utilisées "
    "principalement comme scores de ranking."
)

report_lines.append(
    "Elles ne doivent pas encore être interprétées "
    "comme des probabilités parfaitement calibrées."
)

report_lines.append(
    "La calibration finale devra être réalisée "
    "avec une procédure temporelle out-of-fold."
)

report_lines.append("")


# ------------------------------------------------------------
# TOP SCORE QUALITY
# ------------------------------------------------------------

report_lines.append(
    "=" * 70
)

report_lines.append(
    "MEILLEURS SCORES ROC-AUC"
)

report_lines.append(
    "=" * 70
)

best_quality = (
    score_quality
    .sort_values(
        [
            "roc_auc",
            "pr_auc",
        ],
        ascending=False,
    )
    .head(10)
)

for _, row in best_quality.iterrows():

    report_lines.append(
        f"{row['model']} | "
        f"{row['distribution']} | "
        f"ROC-AUC={row['roc_auc']:.4f} | "
        f"PR-AUC={row['pr_auc']:.4f} | "
        f"Brier={row['brier_score']:.4f}"
    )

report_lines.append("")


# ------------------------------------------------------------
# TOP ZONES
# ------------------------------------------------------------

report_lines.append(
    "=" * 70
)

report_lines.append(
    "MEILLEURES ZONES TOP %"
)

report_lines.append(
    "=" * 70
)

best_zones = (
    ranking[
        ranking["stability"] >= 0.80
    ]
    .sort_values(
        [
            "mean_lift",
            "mean_goal_rate",
        ],
        ascending=False,
    )
    .head(15)
)

for _, row in best_zones.iterrows():

    report_lines.append(
        f"{row['model']} | "
        f"{row['distribution']} | "
        f"Top {int(row['top_percentage'])}% | "
        f"Goal rate={row['mean_goal_rate'] * 100:.2f}% | "
        f"Lift={row['mean_lift']:.2f}x | "
        f"Stability={row['stability'] * 100:.0f}%"
    )

report_lines.append("")


# ------------------------------------------------------------
# BEST THRESHOLDS
# ------------------------------------------------------------

report_lines.append(
    "=" * 70
)

report_lines.append(
    "MEILLEURS SEUILS"
)

report_lines.append(
    "=" * 70
)

if len(threshold_summary) > 0:

    threshold_candidates = (
        threshold_summary[
            (
                threshold_summary[
                    "stability"
                ] >= 0.80
            )
            &
            (
                threshold_summary[
                    "mean_n_selected"
                ] >= 20
            )
        ]
        .sort_values(
            [
                "mean_lift",
                "mean_f1",
            ],
            ascending=False,
        )
        .head(15)
    )

    for _, row in threshold_candidates.iterrows():

        report_lines.append(
            f"{row['model']} | "
            f"{row['distribution']} | "
            f"Threshold={row['threshold']:.2f} | "
            f"Goal rate={row['mean_goal_rate'] * 100:.2f}% | "
            f"Lift={row['mean_lift']:.2f}x | "
            f"F1={row['mean_f1']:.3f} | "
            f"Stability={row['stability'] * 100:.0f}% | "
            f"N={row['mean_n_selected']:.0f}"
        )

else:

    report_lines.append(
        "Aucun seuil exploitable."
    )

report_lines.append("")


# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------

report_lines.append(
    "=" * 70
)

report_lines.append(
    "INTERPRETATION"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    "1. Les scores doivent être évalués principalement "
    "comme des scores de ranking."
)

report_lines.append(
    "2. Les Top 5% sont très sélectifs et peuvent avoir "
    "une variance élevée."
)

report_lines.append(
    "3. Les Top 10-20% sont généralement plus utiles "
    "pour rechercher un compromis entre signal et volume."
)

report_lines.append(
    "4. Un lift supérieur à 1 indique que la sélection "
    "contient davantage de goals_after_87 que la moyenne."
)

report_lines.append(
    "5. Une stabilité élevée signifie que le phénomène "
    "apparaît sur plusieurs folds temporels."
)

report_lines.append(
    "6. Aucun résultat de cette analyse ne constitue "
    "encore une preuve de rentabilité."
)

report_lines.append(
    "7. La prochaine étape importante est de comparer "
    "le score du modèle aux cotes disponibles à la 87e minute."
)

report_lines.append(
    "8. Une vraie décision de pari nécessitera ensuite "
    "une analyse d'edge et de ROI avec les cotes historiques."
)

report_lines.append("")

report_lines.append(
    "FIN DU RAPPORT"
)

report_lines.append(
    "=" * 70
)

REPORT_FILE = (
    OUTPUT_DIR
    / "calibration_threshold_report.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# RÉSUMÉ CONSOLE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)

print(
    f"\nRésultats enregistrés dans :"
)

print(
    OUTPUT_DIR
)

print("\nFichiers générés :")

output_files = [
    "score_quality.csv",
    "thresholds_by_fold.csv",
    "threshold_summary.csv",
    "top_percentages_by_fold.csv",
    "top_percentages_summary.csv",
    "ranking_top_percentages.csv",
    "calibration_bins.csv",
    "calibration_comparison.csv",
    "calibration_threshold_report.txt",
]

for filename in output_files:

    print(
        f" - {filename}"
    )


# ============================================================
# RÉSUMÉ DES MEILLEURS RÉSULTATS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 ZONES LES PLUS INTÉRESSANTES")
print("=" * 70)

display_columns = [
    "model",
    "distribution",
    "top_percentage",
    "mean_goal_rate",
    "mean_lift",
    "stability",
]

print(
    ranking[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("FIN")
print("=" * 70)