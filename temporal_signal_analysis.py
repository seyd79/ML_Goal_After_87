"""
Temporal Signal Analysis
========================

Analyse les prédictions issues de la validation chronologique.

Entrée :
    outputs/chronological_validation/chronological_predictions.csv

Colonnes attendues :
    model
    fold
    dataset_index
    probability
    prediction
    actual

Analyses :
    - Top 10 / 15 / 20 / 25 / 30 %
    - Taux réel de goal_after_87
    - Lift
    - Ratio top/bottom
    - Intervalles de confiance Wilson
    - Analyse par seuil de probabilité
    - Analyse par bins de probabilité
    - ROC-AUC
    - PR-AUC
    - Brier score
    - Log loss
    - Stabilité temporelle par fold
    - Comparaison des modèles

Sorties :
    outputs/temporal_signal_analysis/
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "outputs/chronological_validation/chronological_predictions.csv"
)

OUTPUT_DIR = Path("outputs/temporal_signal_analysis")

MODELS = ["73_27", "67_33", "50_50"]

TOP_PERCENTAGES = [10, 15, 20, 25, 30]

THRESHOLDS = [
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

def wilson_ci(successes, total, confidence=0.95):
    """
    Wilson confidence interval for a binomial proportion.
    """

    if total == 0:
        return np.nan, np.nan

    z = 1.959963984540054

    p = successes / total

    denominator = 1 + (z ** 2 / total)

    centre = (
        p
        + (z ** 2 / (2 * total))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p) / total
            )
            + (z ** 2 / (4 * total ** 2))
        )
        / denominator
    )

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return lower, upper


def stability_label(n):
    """
    Interprétation simple basée sur la taille de l'échantillon.
    """

    if n < 10:
        return "VERY_WEAK"

    if n < 30:
        return "WEAK"

    if n < 100:
        return "MODERATE"

    return "STRONG"


def safe_auc(y_true, probability):
    """
    ROC-AUC robuste si une seule classe est présente.
    """

    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return roc_auc_score(y_true, probability)

    except Exception:
        return np.nan


def safe_pr_auc(y_true, probability):
    """
    PR-AUC robuste.
    """

    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return average_precision_score(y_true, probability)

    except Exception:
        return np.nan


def safe_brier(y_true, probability):
    try:
        return brier_score_loss(y_true, probability)
    except Exception:
        return np.nan


def safe_log_loss(y_true, probability):
    try:
        probability = np.clip(probability, 1e-15, 1 - 1e-15)
        return log_loss(
            y_true,
            probability,
            labels=[0, 1],
        )
    except Exception:
        return np.nan


def get_model_label(model):
    """
    Normalise le nom des modèles.
    """

    value = str(model).strip()

    if value in {"73_27", "73/27", "73-27"}:
        return "73_27"

    if value in {"67_33", "67/33", "67-33"}:
        return "67_33"

    if value in {"50_50", "50/50", "50-50"}:
        return "50_50"

    return value


# ============================================================
# CHARGEMENT
# ============================================================

def load_predictions():
    print("=" * 70)
    print("TEMPORAL SIGNAL ANALYSIS")
    print("=" * 70)

    print()
    print("CHARGEMENT DES PRÉDICTIONS CHRONOLOGIQUES")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Fichier : {INPUT_FILE}")
    print(f"Lignes  : {len(df):,}")
    print(f"Colonnes: {len(df.columns)}")

    print()
    print("Colonnes détectées :")
    print(list(df.columns))

    required_columns = [
        "model",
        "fold",
        "dataset_index",
        "probability",
        "prediction",
        "actual",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df = df.copy()

    df["model"] = df["model"].apply(get_model_label)

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce",
    )

    df["dataset_index"] = pd.to_numeric(
        df["dataset_index"],
        errors="coerce",
    )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    # Supprimer les lignes invalides
    before = len(df)

    df = df.dropna(
        subset=[
            "model",
            "fold",
            "probability",
            "actual",
        ]
    ).copy()

    removed = before - len(df)

    if removed > 0:
        print(
            f"\nLignes supprimées pour données invalides : {removed}"
        )

    # Normaliser la cible
    df["actual"] = df["actual"].astype(int)

    # Vérification
    invalid_target = ~df["actual"].isin([0, 1])

    if invalid_target.any():
        raise ValueError(
            "La colonne 'actual' contient des valeurs autres que 0/1."
        )

    # Probabilités bornées
    if (
        (df["probability"] < 0).any()
        or (df["probability"] > 1).any()
    ):
        raise ValueError(
            "La colonne 'probability' contient des valeurs hors [0, 1]."
        )

    print()
    print("Modèles détectés :")
    print(df["model"].value_counts().to_string())

    print()
    print("Folds détectés :")
    print(sorted(df["fold"].dropna().unique()))

    return df


# ============================================================
# ANALYSE PAR MODÈLE / FOLD
# ============================================================

def analyze_fold_metrics(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[df["model"] == model].copy()

        for fold in sorted(model_df["fold"].unique()):

            fold_df = model_df[
                model_df["fold"] == fold
            ].copy()

            y_true = fold_df["actual"].values
            probability = fold_df["probability"].values

            baseline = y_true.mean()

            auc = safe_auc(
                y_true,
                probability,
            )

            pr_auc = safe_pr_auc(
                y_true,
                probability,
            )

            brier = safe_brier(
                y_true,
                probability,
            )

            ll = safe_log_loss(
                y_true,
                probability,
            )

            rows.append({
                "model": model,
                "fold": int(fold),
                "n": len(fold_df),
                "goals": int(y_true.sum()),
                "goal_rate": baseline,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "brier": brier,
                "log_loss": ll,
            })

    return pd.DataFrame(rows)


# ============================================================
# TOP X %
# ============================================================

def analyze_top_percentages(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[
            df["model"] == model
        ].copy()

        for fold in sorted(model_df["fold"].unique()):

            fold_df = model_df[
                model_df["fold"] == fold
            ].copy()

            if fold_df.empty:
                continue

            baseline = fold_df["actual"].mean()

            ordered = fold_df.sort_values(
                "probability",
                ascending=False,
            ).reset_index(drop=True)

            n = len(ordered)

            for percentage in TOP_PERCENTAGES:

                k = max(
                    1,
                    int(math.ceil(
                        n * percentage / 100
                    ))
                )

                selected = ordered.iloc[:k]

                goals = int(
                    selected["actual"].sum()
                )

                rate = (
                    goals / len(selected)
                    if len(selected) > 0
                    else np.nan
                )

                lift = (
                    rate / baseline
                    if baseline > 0
                    else np.nan
                )

                ci_low, ci_high = wilson_ci(
                    goals,
                    len(selected),
                )

                rows.append({
                    "model": model,
                    "fold": int(fold),
                    "top_percentage": percentage,
                    "n_selected": len(selected),
                    "goals": goals,
                    "goal_rate": rate,
                    "baseline_rate": baseline,
                    "lift": lift,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "stability": stability_label(
                        len(selected)
                    ),
                })

    return pd.DataFrame(rows)


# ============================================================
# TOP / BOTTOM
# ============================================================

def analyze_top_bottom(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[
            df["model"] == model
        ].copy()

        for fold in sorted(model_df["fold"].unique()):

            fold_df = model_df[
                model_df["fold"] == fold
            ].copy()

            if fold_df.empty:
                continue

            ordered = fold_df.sort_values(
                "probability",
                ascending=False,
            ).reset_index(drop=True)

            n = len(ordered)

            k = max(
                1,
                int(math.ceil(n * 0.20))
            )

            top = ordered.iloc[:k]
            bottom = ordered.iloc[-k:]

            top_rate = top["actual"].mean()
            bottom_rate = bottom["actual"].mean()

            ratio = (
                top_rate / bottom_rate
                if bottom_rate > 0
                else np.nan
            )

            difference = (
                top_rate - bottom_rate
            )

            baseline = ordered["actual"].mean()

            top_lift = (
                top_rate / baseline
                if baseline > 0
                else np.nan
            )

            bottom_lift = (
                bottom_rate / baseline
                if baseline > 0
                else np.nan
            )

            rows.append({
                "model": model,
                "fold": int(fold),
                "n_total": n,
                "top20_n": len(top),
                "bottom20_n": len(bottom),
                "top20_goal_rate": top_rate,
                "bottom20_goal_rate": bottom_rate,
                "top20_lift": top_lift,
                "bottom20_lift": bottom_lift,
                "top_bottom_ratio": ratio,
                "top_minus_bottom": difference,
                "top20_better": bool(
                    top_rate > bottom_rate
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# SEUILS
# ============================================================

def analyze_thresholds(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[
            df["model"] == model
        ].copy()

        for fold in sorted(model_df["fold"].unique()):

            fold_df = model_df[
                model_df["fold"] == fold
            ].copy()

            if fold_df.empty:
                continue

            baseline = fold_df["actual"].mean()

            y_true = fold_df["actual"].values
            probability = fold_df["probability"].values

            for threshold in THRESHOLDS:

                selected = fold_df[
                    fold_df["probability"] >= threshold
                ]

                n_selected = len(selected)

                if n_selected > 0:

                    goals = int(
                        selected["actual"].sum()
                    )

                    rate = goals / n_selected

                    precision = rate

                    recall = (
                        goals / y_true.sum()
                        if y_true.sum() > 0
                        else np.nan
                    )

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if (
                            precision + recall
                        ) > 0
                        else 0
                    )

                    lift = (
                        rate / baseline
                        if baseline > 0
                        else np.nan
                    )

                    ci_low, ci_high = wilson_ci(
                        goals,
                        n_selected,
                    )

                else:

                    goals = 0
                    rate = np.nan
                    precision = np.nan
                    recall = 0
                    f1 = 0
                    lift = np.nan
                    ci_low = np.nan
                    ci_high = np.nan

                rows.append({
                    "model": model,
                    "fold": int(fold),
                    "threshold": threshold,
                    "n_selected": n_selected,
                    "goals": goals,
                    "goal_rate": rate,
                    "baseline_rate": baseline,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "lift": lift,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "stability": stability_label(
                        n_selected
                    ),
                })

    return pd.DataFrame(rows)


# ============================================================
# PROBABILITY BINS
# ============================================================

def analyze_probability_bins(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[
            df["model"] == model
        ].copy()

        for fold in sorted(model_df["fold"].unique()):

            fold_df = model_df[
                model_df["fold"] == fold
            ].copy()

            baseline = fold_df["actual"].mean()

            for low, high in PROBABILITY_BINS:

                selected = fold_df[
                    (
                        fold_df["probability"] >= low
                    )
                    &
                    (
                        fold_df["probability"] < high
                    )
                ]

                n = len(selected)

                if n > 0:

                    goals = int(
                        selected["actual"].sum()
                    )

                    rate = goals / n

                    lift = (
                        rate / baseline
                        if baseline > 0
                        else np.nan
                    )

                    ci_low, ci_high = wilson_ci(
                        goals,
                        n,
                    )

                else:

                    goals = 0
                    rate = np.nan
                    lift = np.nan
                    ci_low = np.nan
                    ci_high = np.nan

                rows.append({
                    "model": model,
                    "fold": int(fold),
                    "bin_low": low,
                    "bin_high": high,
                    "bin": f"{low:.2f}-{high:.2f}",
                    "n": n,
                    "goals": goals,
                    "goal_rate": rate,
                    "baseline_rate": baseline,
                    "lift": lift,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "stability": stability_label(n),
                })

    return pd.DataFrame(rows)


# ============================================================
# STABILITÉ DES RANKINGS
# ============================================================

def analyze_rank_stability(df):
    rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[
            df["model"] == model
        ].copy()

        for percentage in TOP_PERCENTAGES:

            fold_rates = []

            for fold in sorted(
                model_df["fold"].unique()
            ):

                fold_df = model_df[
                    model_df["fold"] == fold
                ].copy()

                if fold_df.empty:
                    continue

                ordered = fold_df.sort_values(
                    "probability",
                    ascending=False,
                )

                n = len(ordered)

                k = max(
                    1,
                    int(math.ceil(
                        n * percentage / 100
                    ))
                )

                selected = ordered.iloc[:k]

                rate = selected["actual"].mean()

                fold_rates.append(rate)

            if not fold_rates:
                continue

            fold_rates = np.array(
                fold_rates,
                dtype=float,
            )

            rows.append({
                "model": model,
                "top_percentage": percentage,
                "folds": len(fold_rates),
                "mean_goal_rate": fold_rates.mean(),
                "std_goal_rate": fold_rates.std(
                    ddof=0
                ),
                "min_goal_rate": fold_rates.min(),
                "max_goal_rate": fold_rates.max(),
                "all_folds_above_baseline": None,
            })

    result = pd.DataFrame(rows)

    # Baseline global par modèle
    for model in result["model"].unique():

        baseline = df[
            df["model"] == model
        ]["actual"].mean()

        mask = result["model"] == model

        result.loc[
            mask,
            "all_folds_above_baseline"
        ] = (
            result.loc[
                mask,
                "mean_goal_rate"
            ] > baseline
        )

    return result


# ============================================================
# COMPARAISON DES MODÈLES
# ============================================================

def build_model_comparison(
    fold_metrics,
    top_bottom,
):
    rows = []

    for model in sorted(
        fold_metrics["model"].unique()
    ):

        metrics = fold_metrics[
            fold_metrics["model"] == model
        ]

        tb = top_bottom[
            top_bottom["model"] == model
        ]

        rows.append({
            "model": model,
            "folds": len(metrics),
            "roc_auc_mean": metrics[
                "roc_auc"
            ].mean(),
            "roc_auc_std": metrics[
                "roc_auc"
            ].std(ddof=0),
            "pr_auc_mean": metrics[
                "pr_auc"
            ].mean(),
            "pr_auc_std": metrics[
                "pr_auc"
            ].std(ddof=0),
            "brier_mean": metrics[
                "brier"
            ].mean(),
            "log_loss_mean": metrics[
                "log_loss"
            ].mean(),
            "top20_mean": tb[
                "top20_goal_rate"
            ].mean(),
            "bottom20_mean": tb[
                "bottom20_goal_rate"
            ].mean(),
            "top20_lift_mean": tb[
                "top20_lift"
            ].mean(),
            "top_bottom_ratio_mean": tb[
                "top_bottom_ratio"
            ].mean(),
            "top20_better_folds": int(
                tb["top20_better"].sum()
            ),
            "top20_better_rate": tb[
                "top20_better"
            ].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# ÉVALUATION AUTOMATIQUE
# ============================================================

def assess_signal(
    comparison,
):
    assessments = []

    for _, row in comparison.iterrows():

        auc = row["roc_auc_mean"]
        ratio = row["top_bottom_ratio_mean"]
        better_rate = row["top20_better_rate"]

        if (
            auc >= 0.60
            and ratio >= 1.50
            and better_rate >= 0.80
        ):
            assessment = "STRONG_SIGNAL"

        elif (
            auc >= 0.55
            and ratio >= 1.25
            and better_rate >= 0.80
        ):
            assessment = "MODERATE_SIGNAL"

        elif (
            auc >= 0.52
            and ratio >= 1.10
            and better_rate >= 0.60
        ):
            assessment = "WEAK_SIGNAL"

        else:
            assessment = "VERY_WEAK_OR_NONE"

        assessments.append({
            "model": row["model"],
            "assessment": assessment,
            "roc_auc_mean": auc,
            "top_bottom_ratio_mean": ratio,
            "top20_better_rate": better_rate,
        })

    return pd.DataFrame(assessments)


# ============================================================
# RAPPORT TEXTE
# ============================================================

def create_text_report(
    df,
    fold_metrics,
    top_bottom,
    thresholds,
    bins,
    comparison,
    assessment,
):
    lines = []

    lines.append("=" * 70)
    lines.append("TEMPORAL SIGNAL ANALYSIS REPORT")
    lines.append("=" * 70)

    lines.append("")
    lines.append("DATASET DE PRÉDICTIONS")
    lines.append("-" * 70)

    lines.append(
        f"Nombre total de prédictions : {len(df):,}"
    )

    lines.append(
        f"Nombre de modèles : {df['model'].nunique()}"
    )

    lines.append(
        f"Nombre de folds : {df['fold'].nunique()}"
    )

    global_rate = df["actual"].mean()

    lines.append(
        f"Taux global de goal_after_87 : "
        f"{global_rate:.2%}"
    )

    lines.append("")
    lines.append("COMPARAISON DES MODÈLES")
    lines.append("-" * 70)

    for _, row in comparison.iterrows():

        lines.append("")
        lines.append(
            f"Modèle {row['model']}"
        )

        lines.append(
            f"  ROC-AUC moyen      : "
            f"{row['roc_auc_mean']:.4f}"
        )

        lines.append(
            f"  ROC-AUC std        : "
            f"{row['roc_auc_std']:.4f}"
        )

        lines.append(
            f"  PR-AUC moyen       : "
            f"{row['pr_auc_mean']:.4f}"
        )

        lines.append(
            f"  Brier moyen        : "
            f"{row['brier_mean']:.4f}"
        )

        lines.append(
            f"  Log loss moyen     : "
            f"{row['log_loss_mean']:.4f}"
        )

        lines.append(
            f"  Top 20%            : "
            f"{row['top20_mean']:.2%}"
        )

        lines.append(
            f"  Bottom 20%         : "
            f"{row['bottom20_mean']:.2%}"
        )

        lines.append(
            f"  Top 20% lift       : "
            f"{row['top20_lift_mean']:.2f}x"
        )

        lines.append(
            f"  Top/Bottom ratio   : "
            f"{row['top_bottom_ratio_mean']:.2f}x"
        )

        lines.append(
            f"  Top 20% meilleur   : "
            f"{int(row['top20_better_folds'])}/"
            f"{int(row['folds'])} folds"
        )

    lines.append("")
    lines.append("ÉVALUATION DU SIGNAL")
    lines.append("-" * 70)

    for _, row in assessment.iterrows():

        lines.append(
            f"{row['model']} : {row['assessment']}"
        )

    lines.append("")
    lines.append("INTERPRÉTATION")
    lines.append("-" * 70)

    lines.append(
        "Cette analyse cherche à déterminer si les probabilités "
        "du modèle permettent de classer les matchs selon un "
        "risque croissant de goal_after_87."
    )

    lines.append(
        "Le ROC-AUC mesure la capacité générale de classement."
    )

    lines.append(
        "Le Top 20% vs Bottom 20% permet de vérifier si les "
        "matchs ayant les scores les plus élevés ont réellement "
        "davantage de buts après la 87e minute."
    )

    lines.append(
        "Un signal stable dans plusieurs folds est plus intéressant "
        "qu'un excellent résultat obtenu sur un seul fold."
    )

    lines.append("")
    lines.append(
        "IMPORTANT : cette analyse ne démontre pas encore "
        "la rentabilité d'une stratégie de pari."
    )

    lines.append(
        "La prochaine étape doit être une analyse de seuils "
        "et de rendement potentiel uniquement après confirmation "
        "que l'ordre chronologique du dataset est valide."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Chargement
    # --------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------
    # 2. Métriques par fold
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE DES MÉTRIQUES PAR FOLD")
    print("=" * 70)

    fold_metrics = analyze_fold_metrics(df)

    print(
        fold_metrics.to_string(index=False)
    )

    # --------------------------------------------------------
    # 3. Top %
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE TOP 10 / 15 / 20 / 25 / 30 %")
    print("=" * 70)

    top_percentages = analyze_top_percentages(
        df
    )

    # --------------------------------------------------------
    # 4. Top vs Bottom
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE TOP 20 % VS BOTTOM 20 %")
    print("=" * 70)

    top_bottom = analyze_top_bottom(
        df
    )

    print(
        top_bottom.to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Seuils
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE DES SEUILS")
    print("=" * 70)

    thresholds = analyze_thresholds(
        df
    )

    # --------------------------------------------------------
    # 6. Probability bins
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE DES BINS DE PROBABILITÉ")
    print("=" * 70)

    bins = analyze_probability_bins(
        df
    )

    # --------------------------------------------------------
    # 7. Stabilité ranking
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STABILITÉ DU SIGNAL")
    print("=" * 70)

    rank_stability = analyze_rank_stability(
        df
    )

    # --------------------------------------------------------
    # 8. Comparaison modèles
    # --------------------------------------------------------

    comparison = build_model_comparison(
        fold_metrics,
        top_bottom,
    )

    # --------------------------------------------------------
    # 9. Assessment
    # --------------------------------------------------------

    assessment = assess_signal(
        comparison
    )

    # --------------------------------------------------------
    # 10. Sauvegarde CSV
    # --------------------------------------------------------

    fold_metrics.to_csv(
        OUTPUT_DIR / "temporal_fold_metrics.csv",
        index=False,
    )

    top_percentages.to_csv(
        OUTPUT_DIR / "temporal_top_percentages.csv",
        index=False,
    )

    top_bottom.to_csv(
        OUTPUT_DIR / "temporal_top_bottom_20.csv",
        index=False,
    )

    thresholds.to_csv(
        OUTPUT_DIR / "temporal_thresholds.csv",
        index=False,
    )

    bins.to_csv(
        OUTPUT_DIR / "temporal_probability_bins.csv",
        index=False,
    )

    rank_stability.to_csv(
        OUTPUT_DIR / "temporal_rank_stability.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "temporal_model_comparison.csv",
        index=False,
    )

    assessment.to_csv(
        OUTPUT_DIR / "temporal_signal_assessment.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 11. JSON résumé
    # --------------------------------------------------------

    summary = {
        "total_predictions": int(len(df)),
        "models": sorted(
            df["model"].unique().tolist()
        ),
        "folds": sorted(
            [
                int(x)
                for x in df["fold"].unique()
            ]
        ),
        "global_goal_rate": float(
            df["actual"].mean()
        ),
        "model_comparison": comparison.to_dict(
            orient="records"
        ),
        "assessment": assessment.to_dict(
            orient="records"
        ),
    }

    with open(
        OUTPUT_DIR / "temporal_signal_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 12. Rapport texte
    # --------------------------------------------------------

    report = create_text_report(
        df,
        fold_metrics,
        top_bottom,
        thresholds,
        bins,
        comparison,
        assessment,
    )

    with open(
        OUTPUT_DIR / "temporal_signal_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # 13. Résumé console
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)

    print()
    print(
        f"Prédictions analysées : {len(df):,}"
    )

    print(
        f"Taux global goal_after_87 : "
        f"{df['actual'].mean():.2%}"
    )

    print()

    print(
        comparison[
            [
                "model",
                "roc_auc_mean",
                "pr_auc_mean",
                "top20_mean",
                "bottom20_mean",
                "top_bottom_ratio_mean",
                "top20_better_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("ÉVALUATION :")

    print(
        assessment.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("FICHIERS CRÉÉS")
    print("=" * 70)

    print(
        f"Répertoire : {OUTPUT_DIR}"
    )

    print(
        "  - temporal_fold_metrics.csv"
    )

    print(
        "  - temporal_top_percentages.csv"
    )

    print(
        "  - temporal_top_bottom_20.csv"
    )

    print(
        "  - temporal_thresholds.csv"
    )

    print(
        "  - temporal_probability_bins.csv"
    )

    print(
        "  - temporal_rank_stability.csv"
    )

    print(
        "  - temporal_model_comparison.csv"
    )

    print(
        "  - temporal_signal_assessment.csv"
    )

    print(
        "  - temporal_signal_summary.json"
    )

    print(
        "  - temporal_signal_report.txt"
    )

    print()
    print("=" * 70)
    print("ANALYSE TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()