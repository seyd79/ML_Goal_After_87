#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TOP ZONES ANALYSIS
==================

Analyse les zones exclusives du classement ML pour déterminer
où se concentre le signal "goal_after_87".

Cote fixe : 3.20

Important :
- Les cotes sont HYPOTHÉTIQUES et fixes.
- Cette analyse ne représente PAS encore un ROI réel bookmaker.
- Les matchs sont classés séparément à l'intérieur de chaque fold.
- Les zones sont exclusives :
    Top 0-1 %
    Top 1-2 %
    Top 2-3 %
    Top 3-5 %
    Top 5-7.5 %
    Top 7.5-10 %
    Top 10-15 %
    Top 15-20 %
    Top 20-30 %
    Top 30-50 %

Entrée :
outputs/model_optimization/model_optimization_predictions.csv

Sorties :
outputs/top_zones_analysis/
    top_zones_summary.csv
    top_zones_by_fold.csv
    top_zones_by_model.csv
    top_zones_ranking.csv
    top_zones_report.txt
    selected_matches.csv
"""

from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "model_optimization"
    / "model_optimization_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "top_zones_analysis"
)

ODDS = 3.20
STAKE = 1.0

# Zones exclusives.
# Exemple :
# Top 0-1 % = les 1 % meilleurs scores
# Top 1-2 % = entre le 1er et le 2e percentile de classement
ZONES = [
    ("Top_0_1", 0.0, 1.0),
    ("Top_1_2", 1.0, 2.0),
    ("Top_2_3", 2.0, 3.0),
    ("Top_3_5", 3.0, 5.0),
    ("Top_5_7_5", 5.0, 7.5),
    ("Top_7_5_10", 7.5, 10.0),
    ("Top_10_15", 10.0, 15.0),
    ("Top_15_20", 15.0, 20.0),
    ("Top_20_30", 20.0, 30.0),
    ("Top_30_50", 30.0, 50.0),
]

# Nombre minimum de paris pour qu'une stratégie soit considérée
# comme suffisamment volumique dans le classement secondaire.
MIN_BETS_FOR_ROBUST_RANKING = 20


# ============================================================
# UTILITAIRES
# ============================================================

def safe_pct(value, decimals=2):
    """Convertit une valeur numérique en pourcentage formaté."""
    if pd.isna(value):
        return ""
    return f"{value * 100:.{decimals}f}%"


def safe_float(value, decimals=4):
    """Format numérique simple."""
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def calculate_profit(wins, losses, odds=ODDS, stake=STAKE):
    """
    Profit d'un back bet à cote décimale.

    Victoire :
        profit net = stake * (odds - 1)

    Défaite :
        profit net = -stake
    """
    return (
        wins * stake * (odds - 1.0)
        - losses * stake
    )


def calculate_roi(wins, losses, odds=ODDS, stake=STAKE):
    """ROI = profit / montant total misé."""
    bets = wins + losses

    if bets == 0:
        return np.nan

    profit = calculate_profit(
        wins=wins,
        losses=losses,
        odds=odds,
        stake=stake,
    )

    total_stake = bets * stake

    if total_stake == 0:
        return np.nan

    return profit / total_stake


def calculate_max_losing_streak(results):
    """
    Calcule la plus longue série de pertes.

    results :
        liste/array de 1 = win
        0 = loss
    """
    max_streak = 0
    current = 0

    for result in results:
        if result == 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak


def calculate_max_drawdown(results, odds=ODDS, stake=STAKE):
    """
    Calcule le max drawdown en unités.

    Résultats :
        1 = victoire
        0 = défaite
    """
    if len(results) == 0:
        return np.nan

    balance = 0.0
    peak = 0.0
    max_dd = 0.0

    win_profit = stake * (odds - 1.0)
    loss_profit = -stake

    for result in results:
        if result == 1:
            balance += win_profit
        else:
            balance += loss_profit

        peak = max(peak, balance)

        drawdown = peak - balance
        max_dd = max(max_dd, drawdown)

    return max_dd


def assign_zone(rank_pct):
    """
    Retourne la zone exclusive correspondant au percentile de classement.

    rank_pct :
        0 = meilleur score
        100 = pire score
    """

    for zone_name, lower, upper in ZONES:

        # Première zone : [0, 1]
        if lower == 0.0:
            if rank_pct >= lower and rank_pct <= upper:
                return zone_name

        # Zones suivantes :
        # (lower, upper]
        else:
            if rank_pct > lower and rank_pct <= upper:
                return zone_name

    return None


# ============================================================
# CHARGEMENT
# ============================================================

def load_predictions():
    print("=" * 80)
    print("TOP ZONES ANALYSIS")
    print("=" * 80)

    print(f"\nFichier d'entrée :")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        print("\nERREUR : fichier introuvable.")
        print("Vérifie que le fichier existe :")
        print(INPUT_FILE)
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows chargées : {len(df):,}")
    print(f"Colonnes : {list(df.columns)}")

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
        print("\nERREUR : colonnes manquantes :")
        for col in missing:
            print(f"  - {col}")

        print("\nColonnes attendues :")
        for col in required_columns:
            print(f"  - {col}")

        sys.exit(1)

    # Nettoyage
    df = df.dropna(
        subset=[
            "model",
            "distribution",
            "fold",
            "dataset_index",
            "probability",
            "actual",
        ]
    ).copy()

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
    ).astype(int)

    df["dataset_index"] = pd.to_numeric(
        df["dataset_index"],
        errors="coerce",
    ).astype(int)

    df = df.dropna(
        subset=[
            "probability",
            "actual",
        ]
    ).copy()

    # Sécurité
    df["actual"] = df["actual"].astype(int)

    # Vérification des probabilités
    invalid_probability = (
        (df["probability"] < 0)
        | (df["probability"] > 1)
    )

    if invalid_probability.any():
        print(
            f"\nAVERTISSEMENT : "
            f"{invalid_probability.sum():,} probabilités invalides."
        )

        df = df.loc[~invalid_probability].copy()

    print(f"Rows après nettoyage : {len(df):,}")

    global_rate = df["actual"].mean()

    print(
        f"Taux global goal_after_87 : "
        f"{global_rate * 100:.4f}%"
    )

    print(f"\nCote fixe utilisée : {ODDS:.2f}")

    break_even = 1.0 / ODDS

    print(
        f"Break-even à {ODDS:.2f} : "
        f"{break_even * 100:.2f}%"
    )

    return df


# ============================================================
# CALCUL DES ZONES
# ============================================================

def prepare_zone_data(df):
    """
    Pour chaque modèle/distribution/fold :

    1. Classe les matchs par probabilité décroissante.
    2. Calcule leur percentile de classement.
    3. Attribue une zone exclusive.
    """

    print("\n" + "=" * 80)
    print("CONSTRUCTION DES ZONES")
    print("=" * 80)

    all_parts = []

    group_columns = [
        "model",
        "distribution",
        "fold",
    ]

    grouped = df.groupby(
        group_columns,
        sort=True,
    )

    total_groups = len(grouped)

    print(f"\nGroupes à traiter : {total_groups}")

    for group_number, (
        (model, distribution, fold),
        group,
    ) in enumerate(grouped, start=1):

        group = group.copy()

        # Classement :
        # probabilité la plus élevée = meilleur rang
        group = group.sort_values(
            by=[
                "probability",
                "dataset_index",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        n = len(group)

        if n == 0:
            continue

        # Percentile de classement.
        #
        # Premier match = 0 %
        # Dernier match ≈ 100 %
        #
        # Avec 1,380 matchs :
        # Top 1 % ≈ 14 matchs
        if n == 1:
            group["rank_pct"] = 0.0
        else:
            group["rank_pct"] = (
                group.index / (n - 1)
            ) * 100.0

        group["zone"] = group["rank_pct"].apply(
            assign_zone
        )

        # Matchs dans une zone
        group = group[
            group["zone"].notna()
        ].copy()

        all_parts.append(group)

        if group_number == 1 or group_number % 10 == 0:
            print(
                f"  Groupe {group_number}/{total_groups}"
            )

    if not all_parts:
        print("\nERREUR : aucune zone créée.")
        sys.exit(1)

    result = pd.concat(
        all_parts,
        ignore_index=True,
    )

    print(
        f"\nRows avec zone : "
        f"{len(result):,}"
    )

    print("\nDistribution globale des zones :")

    zone_counts = (
        result["zone"]
        .value_counts()
        .reindex(
            [z[0] for z in ZONES],
            fill_value=0,
        )
    )

    for zone, count in zone_counts.items():
        print(
            f"  {zone:<15} "
            f"{count:>8,}"
        )

    return result


# ============================================================
# AGRÉGATION PRINCIPALE
# ============================================================

def calculate_zone_metrics(
    zone_df,
    group_columns,
):
    """
    Calcule toutes les métriques pour une combinaison
    de modèle/distribution/zone ou pour une autre granularité.
    """

    records = []

    grouped = zone_df.groupby(
        group_columns,
        sort=False,
    )

    for keys, group in grouped:

        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(
            zip(group_columns, keys)
        )

        actuals = group["actual"].astype(int).tolist()

        bets = len(actuals)

        if bets == 0:
            continue

        wins = int(sum(actuals))
        losses = bets - wins

        win_rate = wins / bets

        baseline = (
            group["actual"].mean()
        )

        lift = (
            win_rate / baseline
            if baseline > 0
            else np.nan
        )

        profit = calculate_profit(
            wins=wins,
            losses=losses,
        )

        roi = calculate_roi(
            wins=wins,
            losses=losses,
        )

        fair_odds = (
            1.0 / win_rate
            if win_rate > 0
            else np.nan
        )

        avg_probability = (
            group["probability"].mean()
        )

        min_probability = (
            group["probability"].min()
        )

        max_probability = (
            group["probability"].max()
        )

        implied_probability = (
            1.0 / ODDS
        )

        edge_points = (
            win_rate - implied_probability
        )

        max_losing_streak = (
            calculate_max_losing_streak(actuals)
        )

        max_drawdown = (
            calculate_max_drawdown(actuals)
        )

        records.append(
            {
                **key_dict,

                "bets": bets,
                "wins": wins,
                "losses": losses,

                "win_rate": win_rate,
                "baseline": baseline,
                "lift": lift,

                "odds": ODDS,
                "break_even_rate": implied_probability,

                "edge_points": edge_points,

                "fair_odds": fair_odds,

                "avg_probability": avg_probability,
                "min_probability": min_probability,
                "max_probability": max_probability,

                "total_stake": bets * STAKE,
                "profit": profit,
                "roi": roi,

                "bets_per_100_matches": (
                    bets / group["dataset_index"].nunique()
                    * 100.0
                    if group["dataset_index"].nunique() > 0
                    else np.nan
                ),

                "max_losing_streak": max_losing_streak,
                "max_drawdown": max_drawdown,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# ANALYSE GLOBALE PAR ZONE
# ============================================================

def build_summary(zone_df):
    print("\n" + "=" * 80)
    print("ANALYSE GLOBALE PAR ZONE")
    print("=" * 80)

    # Une ligne par model + distribution + zone
    summary = calculate_zone_metrics(
        zone_df,
        [
            "model",
            "distribution",
            "zone",
        ],
    )

    # Ajouter l'ordre des zones
    zone_order = {
        zone_name: index
        for index, (
            zone_name,
            _,
            _,
        ) in enumerate(ZONES)
    }

    summary["zone_order"] = (
        summary["zone"]
        .map(zone_order)
    )

    summary = summary.sort_values(
        by=[
            "roi",
            "win_rate",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return summary


# ============================================================
# ANALYSE PAR FOLD
# ============================================================

def build_by_fold(zone_df):
    print("\n" + "=" * 80)
    print("ANALYSE PAR FOLD")
    print("=" * 80)

    by_fold = calculate_zone_metrics(
        zone_df,
        [
            "model",
            "distribution",
            "fold",
            "zone",
        ],
    )

    zone_order = {
        zone_name: index
        for index, (
            zone_name,
            _,
            _,
        ) in enumerate(ZONES)
    }

    by_fold["zone_order"] = (
        by_fold["zone"]
        .map(zone_order)
    )

    by_fold = by_fold.sort_values(
        by=[
            "model",
            "distribution",
            "fold",
            "zone_order",
        ]
    ).reset_index(drop=True)

    return by_fold


# ============================================================
# AGRÉGATION PAR MODÈLE
# ============================================================

def build_by_model(zone_df):
    print("\n" + "=" * 80)
    print("ANALYSE PAR MODÈLE")
    print("=" * 80)

    by_model = calculate_zone_metrics(
        zone_df,
        [
            "model",
            "zone",
        ],
    )

    zone_order = {
        zone_name: index
        for index, (
            zone_name,
            _,
            _,
        ) in enumerate(ZONES)
    }

    by_model["zone_order"] = (
        by_model["zone"]
        .map(zone_order)
    )

    by_model = by_model.sort_values(
        by=[
            "model",
            "zone_order",
        ]
    ).reset_index(drop=True)

    return by_model


# ============================================================
# ROBUSTESSE
# ============================================================

def build_robust_ranking(
    summary,
    by_fold,
):
    """
    Classe les stratégies en tenant compte de :

    - ROI
    - win rate
    - lift
    - nombre de paris
    - nombre de folds positifs
    - pire ROI d'un fold
    - max losing streak
    - max drawdown

    Important :
    ce classement est descriptif.
    Il ne doit pas être utilisé seul pour déclarer
    une stratégie réellement rentable.
    """

    print("\n" + "=" * 80)
    print("CLASSEMENT DE ROBUSTESSE")
    print("=" * 80)

    fold_groups = by_fold.groupby(
        [
            "model",
            "distribution",
            "zone",
        ],
        sort=False,
    )

    records = []

    for keys, group in fold_groups:

        model, distribution, zone = keys

        base_rows = summary[
            (summary["model"] == model)
            & (summary["distribution"] == distribution)
            & (summary["zone"] == zone)
        ]

        if base_rows.empty:
            continue

        base = base_rows.iloc[0]

        fold_rois = (
            group["roi"]
            .dropna()
            .values
        )

        fold_win_rates = (
            group["win_rate"]
            .dropna()
            .values
        )

        fold_lifts = (
            group["lift"]
            .dropna()
            .values
        )

        profitable_folds = int(
            np.sum(fold_rois > 0)
        )

        total_folds = len(fold_rois)

        profitable_fold_rate = (
            profitable_folds / total_folds
            if total_folds > 0
            else np.nan
        )

        worst_fold_roi = (
            np.min(fold_rois)
            if len(fold_rois)
            else np.nan
        )

        best_fold_roi = (
            np.max(fold_rois)
            if len(fold_rois)
            else np.nan
        )

        mean_fold_roi = (
            np.mean(fold_rois)
            if len(fold_rois)
            else np.nan
        )

        std_fold_roi = (
            np.std(fold_rois, ddof=0)
            if len(fold_rois)
            else np.nan
        )

        mean_fold_win_rate = (
            np.mean(fold_win_rates)
            if len(fold_win_rates)
            else np.nan
        )

        mean_fold_lift = (
            np.mean(fold_lifts)
            if len(fold_lifts)
            else np.nan
        )

        # Score de robustesse descriptif.
        #
        # On favorise :
        # - profitable fold rate
        # - ROI
        # - lift
        # - volume
        #
        # Le score n'est PAS une probabilité.
        volume_factor = min(
            1.0,
            base["bets"] / 100.0
        )

        robustness_score = (
            0.35 * (
                profitable_fold_rate
                if not pd.isna(profitable_fold_rate)
                else 0
            )
            + 0.25 * (
                max(
                    0.0,
                    min(
                        1.0,
                        base["roi"] / 1.0
                    )
                )
            )
            + 0.20 * (
                max(
                    0.0,
                    min(
                        2.0,
                        base["lift"] / 2.0
                    )
                )
            )
            + 0.20 * volume_factor
        )

        records.append(
            {
                "model": model,
                "distribution": distribution,
                "zone": zone,

                "bets": base["bets"],
                "wins": base["wins"],
                "losses": base["losses"],

                "win_rate": base["win_rate"],
                "baseline": base["baseline"],
                "lift": base["lift"],

                "odds": base["odds"],
                "break_even_rate": base["break_even_rate"],
                "edge_points": base["edge_points"],
                "fair_odds": base["fair_odds"],

                "profit": base["profit"],
                "roi": base["roi"],

                "profitable_folds": profitable_folds,
                "total_folds": total_folds,
                "profitable_fold_rate": profitable_fold_rate,

                "mean_fold_roi": mean_fold_roi,
                "std_fold_roi": std_fold_roi,
                "worst_fold_roi": worst_fold_roi,
                "best_fold_roi": best_fold_roi,

                "mean_fold_win_rate": mean_fold_win_rate,
                "mean_fold_lift": mean_fold_lift,

                "max_losing_streak": base[
                    "max_losing_streak"
                ],

                "max_drawdown": base[
                    "max_drawdown"
                ],

                "bets_per_100_matches": base[
                    "bets_per_100_matches"
                ],

                "robustness_score": robustness_score,
            }
        )

    ranking = pd.DataFrame(records)

    if ranking.empty:
        return ranking

    ranking["sufficient_volume"] = (
        ranking["bets"]
        >= MIN_BETS_FOR_ROBUST_RANKING
    )

    # Priorité :
    # 1. volume suffisant
    # 2. robustesse
    # 3. ROI
    # 4. lift
    ranking = ranking.sort_values(
        by=[
            "sufficient_volume",
            "robustness_score",
            "roi",
            "lift",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    ranking["rank"] = (
        np.arange(len(ranking)) + 1
    )

    return ranking


# ============================================================
# MEILLEURE ZONE PAR ZONE
# ============================================================

def build_zone_overview(summary):
    """
    Pour chaque zone, identifie les meilleures combinaisons
    modèle/distribution.

    Cela permet de répondre à :

    "Pour Top 1 %, quel modèle fonctionne le mieux ?"
    "Pour Top 5 %, quel modèle fonctionne le mieux ?"
    etc.
    """

    records = []

    for zone_name, _, _ in ZONES:

        zone_data = summary[
            summary["zone"] == zone_name
        ].copy()

        if zone_data.empty:
            continue

        zone_data = zone_data.sort_values(
            by=[
                "roi",
                "win_rate",
                "bets",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        best = zone_data.iloc[0]

        records.append(
            {
                "zone": zone_name,
                "best_model": best["model"],
                "best_distribution": best[
                    "distribution"
                ],

                "bets": best["bets"],
                "wins": best["wins"],
                "losses": best["losses"],

                "win_rate": best["win_rate"],
                "baseline": best["baseline"],
                "lift": best["lift"],

                "odds": best["odds"],
                "break_even_rate": best[
                    "break_even_rate"
                ],

                "edge_points": best[
                    "edge_points"
                ],

                "fair_odds": best[
                    "fair_odds"
                ],

                "profit": best["profit"],
                "roi": best["roi"],

                "max_losing_streak": best[
                    "max_losing_streak"
                ],

                "max_drawdown": best[
                    "max_drawdown"
                ],
            }
        )

    return pd.DataFrame(records)


# ============================================================
# MATCHES SÉLECTIONNÉS
# ============================================================

def build_selected_matches(zone_df):
    """
    Exporte les matchs appartenant aux zones étudiées.

    Chaque ligne garde :
    - modèle
    - distribution
    - fold
    - index du match
    - probabilité
    - résultat
    - rang
    - zone
    """

    selected = zone_df.copy()

    selected = selected[
        [
            "model",
            "distribution",
            "fold",
            "dataset_index",
            "probability",
            "actual",
            "rank_pct",
            "zone",
        ]
    ].copy()

    selected = selected.sort_values(
        by=[
            "model",
            "distribution",
            "fold",
            "rank_pct",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    return selected


# ============================================================
# RAPPORT TEXTE
# ============================================================

def generate_report(
    df,
    summary,
    by_fold,
    ranking,
    zone_overview,
):
    report_path = (
        OUTPUT_DIR
        / "top_zones_report.txt"
    )

    global_rate = df["actual"].mean()
    break_even = 1.0 / ODDS

    lines = []

    lines.append(
        "=" * 80
    )
    lines.append(
        "TOP ZONES ANALYSIS - RAPPORT"
    )
    lines.append(
        "=" * 80
    )

    lines.append("")
    lines.append(
        f"Dataset predictions : {INPUT_FILE}"
    )

    lines.append(
        f"Rows analysées : {len(df):,}"
    )

    lines.append(
        f"Taux global goal_after_87 : "
        f"{global_rate * 100:.4f}%"
    )

    lines.append(
        f"Cote fixe : {ODDS:.2f}"
    )

    lines.append(
        f"Break-even : {break_even * 100:.2f}%"
    )

    lines.append("")
    lines.append(
        "IMPORTANT :"
    )
    lines.append(
        "- Les cotes 3.20 sont hypothétiques."
    )
    lines.append(
        "- Les zones sont exclusives."
    )
    lines.append(
        "- Le classement est calculé séparément "
        "dans chaque fold."
    )
    lines.append(
        "- Un même match ne doit appartenir "
        "qu'à une seule zone pour une combinaison "
        "modèle/distribution/fold."
    )
    lines.append(
        "- Un ROI positif ici ne constitue pas "
        "encore une preuve de rentabilité réelle."
    )

    # --------------------------------------------------------
    # Vue par zone
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 80
    )
    lines.append(
        "MEILLEURE STRATÉGIE POUR CHAQUE ZONE"
    )
    lines.append(
        "=" * 80
    )

    if zone_overview.empty:
        lines.append(
            "Aucune donnée."
        )
    else:
        for _, row in zone_overview.iterrows():

            lines.append("")
            lines.append(
                f"{row['zone']}"
            )

            lines.append(
                f"  Modèle        : "
                f"{row['best_model']}"
            )

            lines.append(
                f"  Distribution  : "
                f"{row['best_distribution']}"
            )

            lines.append(
                f"  Paris         : "
                f"{int(row['bets'])}"
            )

            lines.append(
                f"  Wins/Losses   : "
                f"{int(row['wins'])}/"
                f"{int(row['losses'])}"
            )

            lines.append(
                f"  Win rate      : "
                f"{row['win_rate'] * 100:.2f}%"
            )

            lines.append(
                f"  Baseline      : "
                f"{row['baseline'] * 100:.2f}%"
            )

            lines.append(
                f"  Lift          : "
                f"{row['lift']:.2f}x"
            )

            lines.append(
                f"  Edge          : "
                f"{row['edge_points'] * 100:.2f} pts"
            )

            lines.append(
                f"  Fair odds     : "
                f"{row['fair_odds']:.3f}"
                if not pd.isna(row["fair_odds"])
                else "  Fair odds     : N/A"
            )

            lines.append(
                f"  Profit simulé : "
                f"{row['profit']:.2f}"
            )

            lines.append(
                f"  ROI simulé    : "
                f"{row['roi'] * 100:.2f}%"
            )

            lines.append(
                f"  Losing streak : "
                f"{int(row['max_losing_streak'])}"
            )

            lines.append(
                f"  Max drawdown  : "
                f"{row['max_drawdown']:.2f}"
            )

    # --------------------------------------------------------
    # Top stratégies
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 80
    )
    lines.append(
        "TOP 20 STRATÉGIES PAR ROI"
    )
    lines.append(
        "=" * 80
    )

    top_roi = summary.sort_values(
        by=[
            "roi",
            "bets",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(20)

    for i, (_, row) in enumerate(
        top_roi.iterrows(),
        start=1,
    ):

        lines.append(
            f"{i:>2}. "
            f"{row['model']} | "
            f"{row['distribution']} | "
            f"{row['zone']} | "
            f"bets={int(row['bets'])} | "
            f"WR={row['win_rate'] * 100:.2f}% | "
            f"lift={row['lift']:.2f}x | "
            f"ROI={row['roi'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Robustesse
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 80
    )
    lines.append(
        "TOP 20 PAR ROBUSTESSE"
    )
    lines.append(
        "=" * 80
    )

    if ranking.empty:
        lines.append(
            "Aucune donnée."
        )
    else:

        for i, (_, row) in enumerate(
            ranking.head(20).iterrows(),
            start=1,
        ):

            lines.append(
                f"{i:>2}. "
                f"{row['model']} | "
                f"{row['distribution']} | "
                f"{row['zone']} | "
                f"bets={int(row['bets'])} | "
                f"WR={row['win_rate'] * 100:.2f}% | "
                f"ROI={row['roi'] * 100:.2f}% | "
                f"folds+={int(row['profitable_folds'])}/"
                f"{int(row['total_folds'])} | "
                f"worst_fold="
                f"{row['worst_fold_roi'] * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 80
    )
    lines.append(
        "DIAGNOSTIC"
    )
    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "1. Le but de cette analyse est de déterminer "
        "si le signal ML est concentré dans certaines "
        "zones du classement."
    )

    lines.append(
        "2. Les zones ne doivent pas être considérées "
        "comme des modèles différents : elles correspondent "
        "à différents niveaux de sélectivité du même score ML."
    )

    lines.append(
        "3. Une zone très performante mais contenant "
        "très peu de paris peut être fortement sensible "
        "à la variance."
    )

    lines.append(
        "4. Une zone légèrement moins rentable mais "
        "beaucoup plus volumique peut être plus utile "
        "pour le Live Scanner."
    )

    lines.append(
        "5. Le ROI simulé dépend entièrement de la cote "
        f"fixe de {ODDS:.2f}. Il faudra remplacer cette "
        "hypothèse par les vraies cotes de marché à 87' "
        "avant toute conclusion de rentabilité."
    )

    lines.append("")
    lines.append(
        "PROCHAINE ÉTAPE APRÈS CETTE ANALYSE :"
    )

    lines.append(
        "Identifier les zones qui combinent : "
        "signal + volume + stabilité."
    )

    lines.append(
        "Ces zones pourront ensuite devenir les niveaux "
        "de signal du Live Scanner."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# SAUVEGARDE
# ============================================================

def save_outputs(
    summary,
    by_fold,
    by_model,
    ranking,
    zone_overview,
    selected_matches,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "top_zones_summary.csv"
    )

    by_fold_path = (
        OUTPUT_DIR
        / "top_zones_by_fold.csv"
    )

    by_model_path = (
        OUTPUT_DIR
        / "top_zones_by_model.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "top_zones_ranking.csv"
    )

    zone_overview_path = (
        OUTPUT_DIR
        / "best_strategy_by_zone.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "selected_matches.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    by_fold.to_csv(
        by_fold_path,
        index=False,
    )

    by_model.to_csv(
        by_model_path,
        index=False,
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    zone_overview.to_csv(
        zone_overview_path,
        index=False,
    )

    selected_matches.to_csv(
        selected_path,
        index=False,
    )

    return {
        "summary": summary_path,
        "by_fold": by_fold_path,
        "by_model": by_model_path,
        "ranking": ranking_path,
        "zone_overview": zone_overview_path,
        "selected_matches": selected_path,
    }


# ============================================================
# AFFICHAGE
# ============================================================

def print_main_results(
    summary,
    ranking,
    zone_overview,
):
    print("\n" + "=" * 80)
    print("RÉSULTATS PAR ZONE")
    print("=" * 80)

    if zone_overview.empty:
        print("\nAucun résultat.")
        return

    display_columns = [
        "zone",
        "best_model",
        "best_distribution",
        "bets",
        "wins",
        "losses",
        "win_rate",
        "baseline",
        "lift",
        "profit",
        "roi",
    ]

    display = zone_overview[
        display_columns
    ].copy()

    display["win_rate"] = (
        display["win_rate"] * 100
    ).round(2)

    display["baseline"] = (
        display["baseline"] * 100
    ).round(2)

    display["lift"] = (
        display["lift"]
    ).round(2)

    display["profit"] = (
        display["profit"]
    ).round(2)

    display["roi"] = (
        display["roi"] * 100
    ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("TOP 20 STRATÉGIES PAR ROI")
    print("=" * 80)

    top_roi = summary.sort_values(
        by=[
            "roi",
            "bets",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(20)

    cols = [
        "model",
        "distribution",
        "zone",
        "bets",
        "wins",
        "losses",
        "win_rate",
        "lift",
        "roi",
        "max_losing_streak",
        "max_drawdown",
    ]

    display = top_roi[cols].copy()

    display["win_rate"] = (
        display["win_rate"] * 100
    ).round(2)

    display["lift"] = (
        display["lift"]
    ).round(2)

    display["roi"] = (
        display["roi"] * 100
    ).round(2)

    display["max_drawdown"] = (
        display["max_drawdown"]
    ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("TOP 20 PAR ROBUSTESSE")
    print("=" * 80)

    if ranking.empty:
        print("Aucun résultat.")
        return

    cols = [
        "rank",
        "model",
        "distribution",
        "zone",
        "bets",
        "win_rate",
        "lift",
        "roi",
        "profitable_fold_rate",
        "worst_fold_roi",
        "max_losing_streak",
        "max_drawdown",
    ]

    display = ranking.head(20)[
        cols
    ].copy()

    display["win_rate"] = (
        display["win_rate"] * 100
    ).round(2)

    display["lift"] = (
        display["lift"]
    ).round(2)

    display["roi"] = (
        display["roi"] * 100
    ).round(2)

    display["profitable_fold_rate"] = (
        display["profitable_fold_rate"] * 100
    ).round(2)

    display["worst_fold_roi"] = (
        display["worst_fold_roi"] * 100
    ).round(2)

    display["max_drawdown"] = (
        display["max_drawdown"]
    ).round(2)

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Chargement
    # --------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------
    # 2. Construction des zones
    # --------------------------------------------------------

    zone_df = prepare_zone_data(df)

    # --------------------------------------------------------
    # 3. Résumé modèle + distribution + zone
    # --------------------------------------------------------

    summary = build_summary(
        zone_df
    )

    # --------------------------------------------------------
    # 4. Analyse par fold
    # --------------------------------------------------------

    by_fold = build_by_fold(
        zone_df
    )

    # --------------------------------------------------------
    # 5. Analyse par modèle
    # --------------------------------------------------------

    by_model = build_by_model(
        zone_df
    )

    # --------------------------------------------------------
    # 6. Classement robustesse
    # --------------------------------------------------------

    ranking = build_robust_ranking(
        summary,
        by_fold,
    )

    # --------------------------------------------------------
    # 7. Meilleure stratégie par zone
    # --------------------------------------------------------

    zone_overview = build_zone_overview(
        summary
    )

    # --------------------------------------------------------
    # 8. Matchs sélectionnés
    # --------------------------------------------------------

    selected_matches = build_selected_matches(
        zone_df
    )

    # --------------------------------------------------------
    # 9. Sauvegarde
    # --------------------------------------------------------

    paths = save_outputs(
        summary=summary,
        by_fold=by_fold,
        by_model=by_model,
        ranking=ranking,
        zone_overview=zone_overview,
        selected_matches=selected_matches,
    )

    # --------------------------------------------------------
    # 10. Rapport texte
    # --------------------------------------------------------

    report_path = generate_report(
        df=df,
        summary=summary,
        by_fold=by_fold,
        ranking=ranking,
        zone_overview=zone_overview,
    )

    # --------------------------------------------------------
    # 11. Affichage
    # --------------------------------------------------------

    print_main_results(
        summary=summary,
        ranking=ranking,
        zone_overview=zone_overview,
    )

    # --------------------------------------------------------
    # 12. Fichiers créés
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FICHIERS CRÉÉS")
    print("=" * 80)

    for name, path in paths.items():
        print(
            f"\n{name:<20} : {path}"
        )

    print(
        f"\n{'report':<20} : {report_path}"
    )

    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)

    print(
        "\nCote utilisée : "
        f"{ODDS:.2f}"
    )

    print(
        "Break-even : "
        f"{1 / ODDS * 100:.2f}%"
    )

    print(
        "\nIMPORTANT : "
        "ne sélectionne pas encore une zone uniquement "
        "sur son ROI. Regarde également le volume, "
        "le lift, la stabilité par fold, le drawdown "
        "et la série maximale de pertes."
    )


if __name__ == "__main__":
    main()