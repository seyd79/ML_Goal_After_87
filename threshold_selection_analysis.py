from pathlib import Path
import pandas as pd
import numpy as np

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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "threshold_selection_analysis"

# Cote hypothétique utilisée pour la simulation
ODDS = 3.20

# Mise par pari
STAKE = 1.0

# Pourcentage de sélection à analyser
TOP_PERCENTAGES = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30]

# Modèles / distributions à analyser
# Si None = tous
SELECTED_MODELS = None
SELECTED_DISTRIBUTIONS = None


# ============================================================
# UTILITAIRES
# ============================================================

def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def calculate_metrics(df):
    """
    Calcule les métriques de betting pour un ensemble de paris.
    """

    bets = len(df)

    if bets == 0:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit": 0.0,
            "roi": 0.0,
        }

    wins = int(df["actual"].sum())
    losses = bets - wins

    win_rate = wins / bets

    profit = (
        wins * (ODDS - 1) * STAKE
        - losses * STAKE
    )

    total_stake = bets * STAKE

    roi = safe_div(profit, total_stake)

    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit": profit,
        "roi": roi,
    }


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("THRESHOLD / TOP SELECTION ANALYSIS")
print("=" * 70)

print(f"\nFichier : {INPUT_FILE}")
print(f"Cote simulée : {ODDS}")
print(f"Mise : {STAKE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFichier introuvable : {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nNombre de lignes chargées : {len(df):,}")

required_columns = [
    "model",
    "distribution",
    "fold",
    "dataset_index",
    "probability",
    "actual",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )


# ============================================================
# NETTOYAGE
# ============================================================

df["probability"] = pd.to_numeric(
    df["probability"],
    errors="coerce"
)

df["actual"] = pd.to_numeric(
    df["actual"],
    errors="coerce"
)

df["fold"] = pd.to_numeric(
    df["fold"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "model",
        "distribution",
        "fold",
        "probability",
        "actual",
    ]
).copy()

df["actual"] = df["actual"].astype(int)
df["fold"] = df["fold"].astype(int)

print(f"Lignes après nettoyage : {len(df):,}")

print(
    f"Taux global de buts après 87' : "
    f"{df['actual'].mean() * 100:.2f}%"
)


# ============================================================
# FILTRES OPTIONNELS
# ============================================================

if SELECTED_MODELS is not None:
    df = df[
        df["model"].isin(SELECTED_MODELS)
    ].copy()

if SELECTED_DISTRIBUTIONS is not None:
    df = df[
        df["distribution"].isin(SELECTED_DISTRIBUTIONS)
    ].copy()


# ============================================================
# DOSSIER OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ANALYSE TOP X%
# ============================================================

summary_rows = []
fold_rows = []
selected_rows = []

group_columns = [
    "model",
    "distribution",
]

groups = df.groupby(group_columns)

total_groups = len(groups)

print(
    f"\nGroupes à analyser : {total_groups}"
)

for group_number, (
    (model, distribution),
    group_df
) in enumerate(groups, start=1):

    print(
        f"\n[{group_number}/{total_groups}] "
        f"{model} | {distribution}"
    )

    group_df = group_df.copy()

    # --------------------------------------------------------
    # Classement par fold
    # --------------------------------------------------------

    for top_pct in TOP_PERCENTAGES:

        selected_parts = []

        # Chaque fold est classé séparément.
        # Cela reproduit une utilisation réelle :
        # on sélectionne les meilleurs scores du fold.
        for fold, fold_df in group_df.groupby("fold"):

            fold_df = fold_df.copy()

            fold_df = fold_df.sort_values(
                "probability",
                ascending=False
            )

            n_matches = len(fold_df)

            n_select = max(
                1,
                int(np.ceil(
                    n_matches * top_pct / 100
                ))
            )

            selected = fold_df.head(
                n_select
            ).copy()

            selected["selection_pct"] = top_pct

            selected_parts.append(
                selected
            )

            # ------------------------------------------------
            # Métriques fold
            # ------------------------------------------------

            metrics = calculate_metrics(
                selected
            )

            baseline_rate = fold_df["actual"].mean()

            lift = safe_div(
                metrics["win_rate"],
                baseline_rate
            )

            fold_rows.append({
                "model": model,
                "distribution": distribution,
                "fold": fold,
                "top_percentage": top_pct,
                "matches_tested": len(fold_df),
                "bets": metrics["bets"],
                "wins": metrics["wins"],
                "losses": metrics["losses"],
                "win_rate": metrics["win_rate"],
                "baseline_rate": baseline_rate,
                "lift": lift,
                "profit": metrics["profit"],
                "roi": metrics["roi"],
                "average_probability": selected[
                    "probability"
                ].mean(),
                "minimum_probability": selected[
                    "probability"
                ].min(),
                "maximum_probability": selected[
                    "probability"
                ].max(),
            })

        # ----------------------------------------------------
        # Résultat global
        # ----------------------------------------------------

        selected_all = pd.concat(
            selected_parts,
            ignore_index=True
        )

        selected_all = selected_all.drop_duplicates(
            subset=[
                "model",
                "distribution",
                "dataset_index",
            ]
        )

        metrics = calculate_metrics(
            selected_all
        )

        baseline_rate = group_df["actual"].mean()

        lift = safe_div(
            metrics["win_rate"],
            baseline_rate
        )

        fair_odds = safe_div(
            1,
            metrics["win_rate"]
        )

        # ----------------------------------------------------
        # Résultats par fold
        # ----------------------------------------------------

        fold_results = [
            row
            for row in fold_rows
            if (
                row["model"] == model
                and row["distribution"] == distribution
                and row["top_percentage"] == top_pct
            )
        ]

        fold_rois = [
            row["roi"]
            for row in fold_results
        ]

        positive_folds = sum(
            roi > 0
            for roi in fold_rois
        )

        profitable_fold_rate = safe_div(
            positive_folds,
            len(fold_rois)
        )

        worst_fold_roi = (
            min(fold_rois)
            if fold_rois
            else 0
        )

        best_fold_roi = (
            max(fold_rois)
            if fold_rois
            else 0
        )

        # ----------------------------------------------------
        # Losing streak
        # ----------------------------------------------------

        ordered = selected_all.sort_values(
            "dataset_index"
        )

        max_losing_streak = 0
        current_losing_streak = 0

        for result in ordered["actual"]:

            if result == 0:

                current_losing_streak += 1

                max_losing_streak = max(
                    max_losing_streak,
                    current_losing_streak
                )

            else:

                current_losing_streak = 0

        # ----------------------------------------------------
        # Résultat
        # ----------------------------------------------------

        summary_rows.append({
            "model": model,
            "distribution": distribution,
            "top_percentage": top_pct,
            "matches_tested": len(group_df),
            "bets": metrics["bets"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "win_rate": metrics["win_rate"],
            "baseline_rate": baseline_rate,
            "lift": lift,
            "odds": ODDS,
            "break_even_rate": 1 / ODDS,
            "fair_odds_from_observed_win_rate": fair_odds,
            "average_probability": selected_all[
                "probability"
            ].mean(),
            "minimum_probability": selected_all[
                "probability"
            ].min(),
            "maximum_probability": selected_all[
                "probability"
            ].max(),
            "total_stake": metrics["bets"] * STAKE,
            "profit": metrics["profit"],
            "roi": metrics["roi"],
            "profitable_folds": positive_folds,
            "total_folds": len(fold_rois),
            "profitable_fold_rate": profitable_fold_rate,
            "worst_fold_roi": worst_fold_roi,
            "best_fold_roi": best_fold_roi,
            "max_losing_streak": max_losing_streak,
        })

        # ----------------------------------------------------
        # Garder les matchs sélectionnés
        # ----------------------------------------------------

        selected_temp = selected_all.copy()

        selected_temp["top_percentage"] = top_pct

        selected_rows.append(
            selected_temp
        )


# ============================================================
# DATAFRAMES FINAUX
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

fold_df = pd.DataFrame(
    fold_rows
)

selected_df = pd.concat(
    selected_rows,
    ignore_index=True
)


# ============================================================
# CLASSEMENT
# ============================================================

# Classement principal :
# ROI + stabilité + volume
#
# Le ROI reste le critère principal,
# mais on conserve les autres métriques
# pour éviter de regarder uniquement le ROI.

summary_df = summary_df.sort_values(
    [
        "roi",
        "profitable_fold_rate",
        "bets",
    ],
    ascending=[
        False,
        False,
        False,
    ]
).reset_index(drop=True)

summary_df.insert(
    0,
    "rank",
    range(1, len(summary_df) + 1)
)


# ============================================================
# CLASSEMENT ROBUSTE
# ============================================================

robust_df = summary_df.copy()

# Score de robustesse simple :
#
# ROI normalisé
# + stabilité des folds
# + volume
#
# Ce score n'est PAS une métrique scientifique.
# Il sert uniquement à classer les stratégies.

roi_component = (
    robust_df["roi"]
    .clip(lower=-1, upper=2)
)

stability_component = (
    robust_df["profitable_fold_rate"]
)

volume_component = (
    np.log1p(
        robust_df["bets"]
    )
    / np.log1p(
        robust_df["matches_tested"]
    )
)

robust_df["robust_score"] = (
    0.50 * roi_component
    + 0.35 * stability_component
    + 0.15 * volume_component
)

robust_df = robust_df.sort_values(
    "robust_score",
    ascending=False
).reset_index(drop=True)

robust_df.insert(
    0,
    "robust_rank",
    range(1, len(robust_df) + 1)
)


# ============================================================
# MEILLEURES STRATÉGIES PAR TOP %
# ============================================================

best_by_percentage = (
    summary_df
    .sort_values(
        "roi",
        ascending=False
    )
    .groupby(
        "top_percentage",
        as_index=False
    )
    .first()
)

best_by_percentage = best_by_percentage.sort_values(
    "top_percentage"
)


# ============================================================
# MEILLEURES STRATÉGIES STABLES
# ============================================================

stable_strategies = summary_df[
    (
        summary_df["profitable_fold_rate"] >= 0.80
    )
    &
    (
        summary_df["bets"] >= 100
    )
].copy()

stable_strategies = stable_strategies.sort_values(
    "roi",
    ascending=False
)


# ============================================================
# SAUVEGARDE
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "threshold_selection_summary.csv"
)

fold_file = (
    OUTPUT_DIR
    / "threshold_selection_by_fold.csv"
)

robust_file = (
    OUTPUT_DIR
    / "threshold_selection_robust_ranking.csv"
)

percentage_file = (
    OUTPUT_DIR
    / "best_strategy_by_percentage.csv"
)

stable_file = (
    OUTPUT_DIR
    / "stable_strategies.csv"
)

selected_file = (
    OUTPUT_DIR
    / "selected_matches.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

fold_df.to_csv(
    fold_file,
    index=False
)

robust_df.to_csv(
    robust_file,
    index=False
)

best_by_percentage.to_csv(
    percentage_file,
    index=False
)

stable_strategies.to_csv(
    stable_file,
    index=False
)

selected_df.to_csv(
    selected_file,
    index=False
)


# ============================================================
# RAPPORT TEXTE
# ============================================================

report_file = (
    OUTPUT_DIR
    / "threshold_selection_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "THRESHOLD / TOP SELECTION ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Cote simulée : {ODDS}\n"
    )

    f.write(
        f"Mise : {STAKE}\n"
    )

    f.write(
        f"Seuil de rentabilité : "
        f"{(1 / ODDS) * 100:.2f}%\n\n"
    )

    f.write(
        f"Nombre total de lignes : "
        f"{len(df):,}\n"
    )

    f.write(
        f"Taux global goal_after_87 : "
        f"{df['actual'].mean() * 100:.2f}%\n\n"
    )

    f.write(
        "TOP 20 STRATEGIES PAR ROI\n"
    )

    f.write(
        "-" * 70 + "\n\n"
    )

    for _, row in summary_df.head(20).iterrows():

        f.write(
            f"{row['model']} | "
            f"{row['distribution']} | "
            f"Top {row['top_percentage']}% | "
            f"Bets={int(row['bets'])} | "
            f"Wins={int(row['wins'])} | "
            f"Losses={int(row['losses'])} | "
            f"WinRate={row['win_rate'] * 100:.2f}% | "
            f"ROI={row['roi'] * 100:.2f}% | "
            f"ProfitableFolds="
            f"{int(row['profitable_folds'])}/"
            f"{int(row['total_folds'])} | "
            f"WorstFold="
            f"{row['worst_fold_roi'] * 100:.2f}% | "
            f"MaxLosingStreak="
            f"{int(row['max_losing_streak'])}\n"
        )

    f.write(
        "\n\n"
        "TOP 20 STRATEGIES PAR ROBUSTESSE\n"
    )

    f.write(
        "-" * 70 + "\n\n"
    )

    for _, row in robust_df.head(20).iterrows():

        f.write(
            f"{row['model']} | "
            f"{row['distribution']} | "
            f"Top {row['top_percentage']}% | "
            f"RobustScore="
            f"{row['robust_score']:.4f} | "
            f"ROI="
            f"{row['roi'] * 100:.2f}% | "
            f"Bets="
            f"{int(row['bets'])} | "
            f"WinRate="
            f"{row['win_rate'] * 100:.2f}% | "
            f"ProfitableFolds="
            f"{int(row['profitable_folds'])}/"
            f"{int(row['total_folds'])}\n"
        )

    f.write(
        "\n\n"
        "STRATEGIES STABLES\n"
    )

    f.write(
        "-" * 70 + "\n\n"
    )

    for _, row in stable_strategies.head(20).iterrows():

        f.write(
            f"{row['model']} | "
            f"{row['distribution']} | "
            f"Top {row['top_percentage']}% | "
            f"Bets={int(row['bets'])} | "
            f"WinRate="
            f"{row['win_rate'] * 100:.2f}% | "
            f"ROI="
            f"{row['roi'] * 100:.2f}% | "
            f"ProfitableFolds="
            f"{int(row['profitable_folds'])}/"
            f"{int(row['total_folds'])} | "
            f"WorstFold="
            f"{row['worst_fold_roi'] * 100:.2f}%\n"
        )

    f.write(
        "\n\n"
        "IMPORTANT\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Les cotes utilisées ici sont hypothétiques.\n"
        f"Chaque pari est simulé avec une cote fixe de {ODDS}.\n"
        "Ce résultat ne représente donc PAS encore une "
        "rentabilité réelle.\n\n"
    )

    f.write(
        "La prochaine étape consiste à remplacer cette cote "
        "hypothétique par les véritables cotes disponibles "
        "à la 87e minute.\n"
    )


# ============================================================
# AFFICHAGE CONSOLE
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 STRATÉGIES PAR ROI")
print("=" * 70)

display_columns = [
    "rank",
    "model",
    "distribution",
    "top_percentage",
    "bets",
    "wins",
    "losses",
    "win_rate",
    "roi",
    "profitable_fold_rate",
    "worst_fold_roi",
    "max_losing_streak",
]

display_df = summary_df[
    display_columns
].head(20).copy()

display_df["win_rate"] = (
    display_df["win_rate"] * 100
).round(2)

display_df["roi"] = (
    display_df["roi"] * 100
).round(2)

display_df["profitable_fold_rate"] = (
    display_df["profitable_fold_rate"] * 100
).round(1)

display_df["worst_fold_roi"] = (
    display_df["worst_fold_roi"] * 100
).round(2)

print(
    display_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("TOP 20 STRATÉGIES PAR ROBUSTESSE")
print("=" * 70)

robust_display_columns = [
    "robust_rank",
    "model",
    "distribution",
    "top_percentage",
    "bets",
    "win_rate",
    "roi",
    "profitable_fold_rate",
    "worst_fold_roi",
    "robust_score",
]

robust_display = robust_df[
    robust_display_columns
].head(20).copy()

robust_display["win_rate"] = (
    robust_display["win_rate"] * 100
).round(2)

robust_display["roi"] = (
    robust_display["roi"] * 100
).round(2)

robust_display["profitable_fold_rate"] = (
    robust_display["profitable_fold_rate"] * 100
).round(1)

robust_display["worst_fold_roi"] = (
    robust_display["worst_fold_roi"] * 100
).round(2)

robust_display["robust_score"] = (
    robust_display["robust_score"]
    .round(4)
)

print(
    robust_display.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("MEILLEURE STRATÉGIE POUR CHAQUE TOP %")
print("=" * 70)

percentage_display = best_by_percentage[
    [
        "top_percentage",
        "model",
        "distribution",
        "bets",
        "wins",
        "losses",
        "win_rate",
        "roi",
        "profitable_fold_rate",
    ]
].copy()

percentage_display["win_rate"] = (
    percentage_display["win_rate"] * 100
).round(2)

percentage_display["roi"] = (
    percentage_display["roi"] * 100
).round(2)

percentage_display["profitable_fold_rate"] = (
    percentage_display["profitable_fold_rate"] * 100
).round(1)

print(
    percentage_display.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("FICHIERS CRÉÉS")
print("=" * 70)

print(f"\n{summary_file}")
print(f"{fold_file}")
print(f"{robust_file}")
print(f"{percentage_file}")
print(f"{stable_file}")
print(f"{selected_file}")
print(f"{report_file}")

print("\nAnalyse terminée.")