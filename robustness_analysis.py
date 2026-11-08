"""
robustness_analysis.py

Analyse de robustesse du signal ML sur les prédictions temporelles.

Objectif :
- Vérifier si le signal observé sur les Top X% est réellement stable
  sur les 5 folds temporels.
- Comparer les zones Top 5 / 10 / 15 / 20 / 25 / 30 %.
- Mesurer :
    * taux de buts après 87'
    * baseline
    * lift
    * minimum par fold
    * maximum par fold
    * moyenne
    * écart-type
    * pire lift
    * nombre moyen de matchs sélectionnés
    * stabilité au-dessus du baseline
    * intervalle de confiance approximatif
- Produire un classement de robustesse.

IMPORTANT :
Les probabilités du modèle sont utilisées comme SCORES DE RANKING.
Elles ne sont pas considérées comme des probabilités calibrées.

Entrée attendue :
outputs/temporal_signal_analysis/temporal_top_percentages.csv

Sorties :
outputs/robustness_analysis/
    robustness_by_fold.csv
    robustness_summary.csv
    robustness_ranking.csv
    robustness_report.txt
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs" / "temporal_signal_analysis"
INPUT_FILE = INPUT_DIR / "temporal_top_percentages.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "robustness_analysis"

TOP_PERCENTAGES = [5, 10, 15, 20, 25, 30]

# Modèle prioritaire identifié lors de l'analyse précédente.
PRIMARY_MODEL = "67_33"

# Tolérance pour considérer qu'un fold est au-dessus du baseline.
# On utilise > baseline strictement.
EPSILON = 1e-12


# ============================================================
# UTILITAIRES
# ============================================================

def print_header(title: str):
    """Affiche un titre lisible."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """
    Intervalle de confiance de Wilson approximatif pour une proportion.
    """

    if n <= 0:
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
            )
            + (z ** 2 / (4 * n ** 2))
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def safe_mean(series):
    """Moyenne robuste."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def safe_std(series):
    """Écart-type robuste."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) <= 1:
        return 0.0

    return values.std(ddof=1)


# ============================================================
# CHARGEMENT
# ============================================================

def load_data():
    """Charge le fichier temporal_top_percentages.csv."""

    print_header("CHARGEMENT DES DONNÉES")

    if not INPUT_FILE.exists():
        print(f"ERREUR : fichier introuvable :")
        print(INPUT_FILE)
        print()
        print("Vérifie que le script est placé à la racine du projet")
        print("et que outputs/temporal_signal_analysis existe.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    print(f"Fichier : {INPUT_FILE}")
    print(f"Lignes : {len(df):,}")
    print(f"Colonnes : {len(df.columns)}")

    required_columns = [
        "model",
        "fold",
        "top_percentage",
        "n_selected",
        "goals",
        "goal_rate",
        "baseline_rate",
        "lift",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print()
        print("ERREUR : colonnes manquantes :")
        for col in missing:
            print(f"  - {col}")

        print()
        print("Colonnes disponibles :")
        print(list(df.columns))

        sys.exit(1)

    return df


# ============================================================
# NETTOYAGE
# ============================================================

def clean_data(df):
    """Nettoie et normalise les types."""

    print_header("NETTOYAGE")

    numeric_columns = [
        "top_percentage",
        "n_selected",
        "goals",
        "goal_rate",
        "baseline_rate",
        "lift",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "model",
            "fold",
            "top_percentage",
            "n_selected",
            "goals",
            "goal_rate",
            "baseline_rate",
        ]
    ).copy()

    removed = before - len(df)

    if removed:
        print(f"Lignes supprimées : {removed}")

    df = df[
        df["top_percentage"].isin(TOP_PERCENTAGES)
    ].copy()

    print(
        f"Top percentages analysés : "
        f"{TOP_PERCENTAGES}"
    )

    print(
        f"Lignes après filtrage : {len(df):,}"
    )

    return df


# ============================================================
# ANALYSE PAR FOLD
# ============================================================

def build_fold_analysis(df):
    """
    Construit une table détaillée par modèle / fold / top%.
    """

    print_header("ANALYSE PAR FOLD")

    rows = []

    for (model, top_percentage), group in df.groupby(
        ["model", "top_percentage"]
    ):

        group = group.sort_values("fold")

        for _, row in group.iterrows():

            goal_rate = float(row["goal_rate"])
            baseline = float(row["baseline_rate"])

            if baseline > 0:
                calculated_lift = goal_rate / baseline
            else:
                calculated_lift = np.nan

            above_baseline = (
                goal_rate > baseline + EPSILON
            )

            # Recalcul de l'IC à partir des matchs sélectionnés.
            ci_low, ci_high = wilson_interval(
                int(row["goals"]),
                int(row["n_selected"])
            )

            rows.append({
                "model": model,
                "fold": int(row["fold"]),
                "top_percentage": int(top_percentage),

                "n_selected": int(row["n_selected"]),
                "goals": int(row["goals"]),

                "goal_rate": goal_rate,
                "baseline_rate": baseline,

                "lift_original": (
                    float(row["lift"])
                    if pd.notna(row["lift"])
                    else np.nan
                ),

                "lift_calculated": calculated_lift,

                "above_baseline": above_baseline,

                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "model",
            "top_percentage",
            "fold",
        ]
    ).reset_index(drop=True)

    return result


# ============================================================
# RÉSUMÉ PAR ZONE
# ============================================================

def build_summary(fold_df):
    """
    Résume chaque modèle / Top%.
    """

    print_header("RÉSUMÉ DES ZONES")

    rows = []

    for (model, top_percentage), group in fold_df.groupby(
        ["model", "top_percentage"]
    ):

        goal_rates = group["goal_rate"].to_numpy()
        baselines = group["baseline_rate"].to_numpy()
        lifts = group["lift_calculated"].to_numpy()

        above_baseline_count = int(
            group["above_baseline"].sum()
        )

        total_folds = len(group)

        stability_rate = (
            above_baseline_count / total_folds
            if total_folds > 0
            else np.nan
        )

        mean_goal_rate = np.mean(goal_rates)
        std_goal_rate = (
            np.std(goal_rates, ddof=1)
            if len(goal_rates) > 1
            else 0.0
        )

        min_goal_rate = np.min(goal_rates)
        max_goal_rate = np.max(goal_rates)

        mean_baseline = np.mean(baselines)

        mean_lift = (
            np.mean(lifts)
            if len(lifts)
            else np.nan
        )

        min_lift = (
            np.min(lifts)
            if len(lifts)
            else np.nan
        )

        max_lift = (
            np.max(lifts)
            if len(lifts)
            else np.nan
        )

        mean_n_selected = group["n_selected"].mean()

        total_selected = group["n_selected"].sum()
        total_goals = group["goals"].sum()

        pooled_goal_rate = (
            total_goals / total_selected
            if total_selected > 0
            else np.nan
        )

        pooled_baseline = (
            np.average(
                group["baseline_rate"],
                weights=group["n_selected"]
            )
            if total_selected > 0
            else np.nan
        )

        pooled_lift = (
            pooled_goal_rate / pooled_baseline
            if pooled_baseline > 0
            else np.nan
        )

        # Pire différence au baseline.
        fold_differences = (
            group["goal_rate"]
            - group["baseline_rate"]
        )

        worst_margin = np.min(
            fold_differences
        )

        best_margin = np.max(
            fold_differences
        )

        # Score de robustesse :
        #
        # 1. priorité à la stabilité
        # 2. priorité au pire fold
        # 3. priorité au lift moyen
        # 4. légère pénalisation de la volatilité
        #
        # Ce score n'est PAS un score de performance
        # de trading. Il sert uniquement à classer
        # la robustesse statistique des zones.
        #
        robustness_score = (
            stability_rate * 50
            + max(0.0, worst_margin) * 100
            + max(0.0, mean_lift - 1.0) * 20
            - std_goal_rate * 20
        )

        rows.append({
            "model": model,
            "top_percentage": int(top_percentage),

            "folds": total_folds,

            "mean_goal_rate": mean_goal_rate,
            "std_goal_rate": std_goal_rate,

            "min_goal_rate": min_goal_rate,
            "max_goal_rate": max_goal_rate,

            "mean_baseline_rate": mean_baseline,

            "mean_lift": mean_lift,
            "min_lift": min_lift,
            "max_lift": max_lift,

            "worst_margin_vs_baseline": worst_margin,
            "best_margin_vs_baseline": best_margin,

            "folds_above_baseline": above_baseline_count,
            "stability_rate": stability_rate,

            "mean_n_selected": mean_n_selected,

            "total_selected": total_selected,
            "total_goals": total_goals,

            "pooled_goal_rate": pooled_goal_rate,
            "pooled_baseline_rate": pooled_baseline,
            "pooled_lift": pooled_lift,

            "robustness_score": robustness_score,
        })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# CLASSEMENT
# ============================================================

def build_ranking(summary_df):
    """
    Construit le classement des zones les plus robustes.
    """

    print_header("CLASSEMENT DE ROBUSTESSE")

    ranking = summary_df.copy()

    # Priorité :
    # 1. stabilité
    # 2. pire marge
    # 3. moyenne du taux
    # 4. faible volatilité
    ranking = ranking.sort_values(
        [
            "stability_rate",
            "worst_margin_vs_baseline",
            "mean_goal_rate",
            "std_goal_rate",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ]
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1)
    )

    return ranking


# ============================================================
# RAPPORT TEXTE
# ============================================================

def build_report(
    fold_df,
    summary_df,
    ranking_df
):
    """Crée un rapport texte lisible."""

    lines = []

    lines.append(
        "RAPPORT DE ROBUSTESSE ML - GOAL AFTER 87"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        "Objectif : vérifier la stabilité du signal "
        "dans les folds temporels."
    )

    lines.append(
        "Les probabilités du modèle sont utilisées "
        "comme scores de ranking."
    )

    lines.append(
        "Aucune conclusion de rentabilité betting n'est "
        "tirée à cette étape."
    )

    lines.append("")

    # --------------------------------------------------------
    # Modèle primaire
    # --------------------------------------------------------

    primary = summary_df[
        summary_df["model"] == PRIMARY_MODEL
    ].copy()

    lines.append(
        f"MODÈLE PRINCIPAL : {PRIMARY_MODEL}"
    )
    lines.append("-" * 80)

    if primary.empty:
        lines.append(
            f"Aucune donnée trouvée pour {PRIMARY_MODEL}."
        )
    else:

        primary = primary.sort_values(
            "top_percentage"
        )

        for _, row in primary.iterrows():

            top = int(row["top_percentage"])

            mean_rate = row["mean_goal_rate"] * 100
            std_rate = row["std_goal_rate"] * 100

            baseline = (
                row["mean_baseline_rate"] * 100
            )

            min_rate = row["min_goal_rate"] * 100
            max_rate = row["max_goal_rate"] * 100

            mean_lift = row["mean_lift"]

            stability = (
                row["stability_rate"] * 100
            )

            worst_margin = (
                row["worst_margin_vs_baseline"] * 100
            )

            lines.append(
                f"Top {top:>2}% | "
                f"moyenne={mean_rate:5.2f}% | "
                f"baseline={baseline:5.2f}% | "
                f"lift={mean_lift:4.2f}x | "
                f"std={std_rate:4.2f}% | "
                f"min={min_rate:5.2f}% | "
                f"max={max_rate:5.2f}% | "
                f"stabilité={stability:5.1f}% | "
                f"pire marge={worst_margin:+5.2f} pts"
            )

    lines.append("")

    # --------------------------------------------------------
    # Meilleures zones
    # --------------------------------------------------------

    lines.append(
        "MEILLEURES ZONES DE ROBUSTESSE"
    )
    lines.append("-" * 80)

    for _, row in ranking_df.head(10).iterrows():

        lines.append(
            f"#{int(row['rank'])} "
            f"{row['model']} "
            f"Top {int(row['top_percentage'])}% | "
            f"score={row['robustness_score']:.2f} | "
            f"mean={row['mean_goal_rate'] * 100:.2f}% | "
            f"lift={row['mean_lift']:.2f}x | "
            f"stabilité={row['stability_rate'] * 100:.0f}% | "
            f"worst="
            f"{row['worst_margin_vs_baseline'] * 100:+.2f} pts"
        )

    lines.append("")

    # --------------------------------------------------------
    # Conclusion automatique
    # --------------------------------------------------------

    lines.append(
        "INTERPRÉTATION"
    )
    lines.append("-" * 80)

    primary_candidates = primary[
        primary["stability_rate"] >= 0.8
    ].copy()

    if not primary_candidates.empty:

        strongest = primary_candidates.sort_values(
            [
                "worst_margin_vs_baseline",
                "mean_goal_rate",
            ],
            ascending=False
        ).iloc[0]

        most_stable = primary_candidates.sort_values(
            [
                "std_goal_rate",
                "mean_goal_rate",
            ],
            ascending=[
                True,
                False,
            ]
        ).iloc[0]

        lines.append(
            f"- Zone la plus forte selon le pire fold : "
            f"Top {int(strongest['top_percentage'])}%."
        )

        lines.append(
            f"- Zone la plus stable selon l'écart-type : "
            f"Top {int(most_stable['top_percentage'])}%."
        )

        lines.append(
            "- Une zone est considérée intéressante uniquement "
            "si elle reste au-dessus du baseline dans la majorité "
            "des folds."
        )

        lines.append(
            "- Cette analyse ne prouve pas encore la rentabilité "
            "d'un pari."
        )

    else:

        lines.append(
            "- Aucune zone du modèle primaire n'atteint "
            "80% de stabilité."
        )

        lines.append(
            "- Le signal doit être considéré comme fragile "
            "avant de passer au backtest betting."
        )

    lines.append("")

    lines.append(
        "PROCHAINE ÉTAPE"
    )
    lines.append("-" * 80)

    lines.append(
        "1. Sélectionner une ou plusieurs zones robustes."
    )

    lines.append(
        "2. Introduire les cotes disponibles avant match."
    )

    lines.append(
        "3. Calculer la probabilité implicite des cotes."
    )

    lines.append(
        "4. Définir une règle de sélection ML + odds."
    )

    lines.append(
        "5. Effectuer le backtest ROI hors échantillon."
    )

    lines.append(
        "6. Mesurer ROI, drawdown, nombre de paris, "
        "profit factor et séries de pertes."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# AFFICHAGE
# ============================================================

def display_primary_results(summary_df):
    """Affiche les résultats du modèle principal."""

    print_header(
        f"RÉSULTATS DU MODÈLE {PRIMARY_MODEL}"
    )

    primary = summary_df[
        summary_df["model"] == PRIMARY_MODEL
    ].copy()

    if primary.empty:
        print(
            f"Aucun résultat pour {PRIMARY_MODEL}."
        )
        return

    primary = primary.sort_values(
        "top_percentage"
    )

    display_columns = [
        "top_percentage",
        "mean_goal_rate",
        "mean_baseline_rate",
        "mean_lift",
        "std_goal_rate",
        "min_goal_rate",
        "max_goal_rate",
        "folds_above_baseline",
        "stability_rate",
        "worst_margin_vs_baseline",
        "mean_n_selected",
    ]

    display_df = primary[
        display_columns
    ].copy()

    display_df["mean_goal_rate"] *= 100
    display_df["mean_baseline_rate"] *= 100
    display_df["std_goal_rate"] *= 100
    display_df["min_goal_rate"] *= 100
    display_df["max_goal_rate"] *= 100
    display_df["stability_rate"] *= 100
    display_df["worst_margin_vs_baseline"] *= 100

    display_df = display_df.rename(
        columns={
            "top_percentage": "Top %",
            "mean_goal_rate": "Goal rate moyen %",
            "mean_baseline_rate": "Baseline %",
            "mean_lift": "Lift moyen",
            "std_goal_rate": "Std %",
            "min_goal_rate": "Min fold %",
            "max_goal_rate": "Max fold %",
            "folds_above_baseline": "Folds > baseline",
            "stability_rate": "Stabilité %",
            "worst_margin_vs_baseline": "Pire marge pts",
            "mean_n_selected": "Matchs moyens",
        }
    )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )


def display_ranking(ranking_df):
    """Affiche le classement."""

    print_header("TOP DES ZONES LES PLUS ROBUSTES")

    columns = [
        "rank",
        "model",
        "top_percentage",
        "mean_goal_rate",
        "mean_lift",
        "stability_rate",
        "worst_margin_vs_baseline",
        "std_goal_rate",
        "robustness_score",
    ]

    display_df = ranking_df[
        columns
    ].head(15).copy()

    display_df["mean_goal_rate"] *= 100
    display_df["stability_rate"] *= 100
    display_df["worst_margin_vs_baseline"] *= 100
    display_df["std_goal_rate"] *= 100

    display_df = display_df.rename(
        columns={
            "rank": "#",
            "model": "Modèle",
            "top_percentage": "Top %",
            "mean_goal_rate": "Goal rate %",
            "mean_lift": "Lift",
            "stability_rate": "Stabilité %",
            "worst_margin_vs_baseline": "Pire marge pts",
            "std_goal_rate": "Std %",
            "robustness_score": "Score",
        }
    )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ROBUSTNESS ANALYSIS - GOAL AFTER 87"
    )

    print(
        "Analyse des zones Top 5/10/15/20/25/30 %"
    )

    print(
        f"Modèle principal : {PRIMARY_MODEL}"
    )

    print(
        f"Entrée : {INPUT_FILE}"
    )

    # --------------------------------------------------------
    # Création du dossier de sortie
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df = clean_data(df)

    if df.empty:
        print()
        print(
            "ERREUR : aucune donnée exploitable."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Analyse par fold
    # --------------------------------------------------------

    fold_df = build_fold_analysis(df)

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    summary_df = build_summary(
        fold_df
    )

    # --------------------------------------------------------
    # Classement
    # --------------------------------------------------------

    ranking_df = build_ranking(
        summary_df
    )

    # --------------------------------------------------------
    # Sauvegarde fold
    # --------------------------------------------------------

    fold_file = (
        OUTPUT_DIR
        / "robustness_by_fold.csv"
    )

    fold_df.to_csv(
        fold_file,
        index=False
    )

    # --------------------------------------------------------
    # Sauvegarde résumé
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / "robustness_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Sauvegarde classement
    # --------------------------------------------------------

    ranking_file = (
        OUTPUT_DIR
        / "robustness_ranking.csv"
    )

    ranking_df.to_csv(
        ranking_file,
        index=False
    )

    # --------------------------------------------------------
    # Rapport texte
    # --------------------------------------------------------

    report = build_report(
        fold_df,
        summary_df,
        ranking_df
    )

    report_file = (
        OUTPUT_DIR
        / "robustness_report.txt"
    )

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    display_primary_results(
        summary_df
    )

    display_ranking(
        ranking_df
    )

    # --------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------

    print_header("FICHIERS GÉNÉRÉS")

    print(f"✓ {fold_file}")
    print(f"✓ {summary_file}")
    print(f"✓ {ranking_file}")
    print(f"✓ {report_file}")

    print()

    # --------------------------------------------------------
    # Recommandation automatique
    # --------------------------------------------------------

    primary = summary_df[
        summary_df["model"] == PRIMARY_MODEL
    ].copy()

    if not primary.empty:

        robust = primary[
            primary["stability_rate"] >= 0.8
        ].copy()

        if not robust.empty:

            robust = robust.sort_values(
                [
                    "worst_margin_vs_baseline",
                    "mean_goal_rate",
                ],
                ascending=False
            )

            best = robust.iloc[0]

            print(
                f"ZONE PRIORITAIRE À TESTER ENSUITE : "
                f"{PRIMARY_MODEL} Top "
                f"{int(best['top_percentage'])}%"
            )

            print(
                f"Goal rate moyen : "
                f"{best['mean_goal_rate'] * 100:.2f}%"
            )

            print(
                f"Baseline moyen : "
                f"{best['mean_baseline_rate'] * 100:.2f}%"
            )

            print(
                f"Lift moyen : "
                f"{best['mean_lift']:.2f}x"
            )

            print(
                f"Stabilité : "
                f"{best['stability_rate'] * 100:.0f}%"
            )

            print(
                f"Pire marge vs baseline : "
                f"{best['worst_margin_vs_baseline'] * 100:+.2f} points"
            )

            print()

            print(
                "IMPORTANT : cette zone n'est PAS encore "
                "une stratégie de pari."
            )

            print(
                "La prochaine étape sera le backtest avec "
                "les cotes et le ROI."
            )

        else:

            print(
                "ATTENTION : aucune zone du modèle primaire "
                "n'atteint 80% de stabilité."
            )

    print()
    print(
        "Analyse terminée."
    )


if __name__ == "__main__":
    main()