"""
FINAL OUT-OF-SAMPLE VALIDATION
==============================

Objectif
--------
Effectuer une validation finale anti-overfitting du système ML.

Principe
--------
Les résultats précédents ont permis d'identifier des stratégies candidates.
Cette étape réserve une partie finale du dataset comme période totalement
hors-échantillon (OOS = Out Of Sample).

IMPORTANT :
- La partie OOS ne sert PAS à choisir le modèle.
- La partie OOS ne sert PAS à choisir le Top %.
- La partie OOS sert uniquement à mesurer la performance finale.
- Les odds 3.20 sont encore HYPOTHETIQUES.
- Les vraies odds de marché à 87' devront remplacer 3.20 dans une future étape.

Pipeline
--------
1. Charger les prédictions issues de model_optimization.py
2. Séparer chronologiquement :
      - développement = 80%
      - OOS final      = 20%
3. Sur la partie développement :
      - analyser les stratégies candidates
      - sélectionner les stratégies les plus robustes
4. Fixer les stratégies AVANT de regarder les résultats OOS.
5. Évaluer ces stratégies uniquement sur la période OOS.
6. Calculer :
      - nombre de paris
      - wins / losses
      - win rate
      - baseline
      - lift
      - profit
      - ROI
      - ROI par fold
      - profitable fold rate
      - max losing streak
      - max drawdown
      - bets / 100 matchs
      - sensibilité aux odds
7. Générer les fichiers CSV + rapport TXT.

ATTENTION
---------
Les prédictions utilisées ici sont déjà produites par une validation
chronologique précédente. Elles permettent donc de tester les performances
sur une période finale qui n'a pas servi à choisir la stratégie.

"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "model_optimization",
    "model_optimization_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "final_oos_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# OOS CONFIGURATION
# ------------------------------------------------------------

# Derniers 20% = validation finale complètement hors-échantillon.
OOS_FRACTION = 0.20

# Odds hypothétiques utilisées pour la simulation.
# NE PAS considérer comme les vraies odds historiques à 87'.
ODDS = 3.20

STAKE = 1.0

# Top % testés.
TOP_PERCENTAGES = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0
]

# Odds supplémentaires pour tester la sensibilité.
SENSITIVITY_ODDS = [
    2.50,
    2.75,
    3.00,
    3.20,
    3.40,
    3.50,
    4.00
]


# ------------------------------------------------------------
# CANDIDATE STRATEGIES
# ------------------------------------------------------------
#
# Ces stratégies viennent de l'analyse précédente.
#
# IMPORTANT :
# Elles sont volontairement limitées.
# On ne teste pas toutes les 48 combinaisons possibles sur OOS
# puis on choisit la meilleure, car cela réintroduirait une forme
# de sélection sur le test final.
#
# Les candidates sont donc fixées AVANT l'évaluation OOS.
#

CANDIDATE_STRATEGIES = [
    {
        "model": "ET_500_d15",
        "distribution": "73_27",
        "top_pct": 5.0,
        "name": "ET_500_d15_73_27_Top5"
    },
    {
        "model": "ET_500_d15",
        "distribution": "73_27",
        "top_pct": 2.0,
        "name": "ET_500_d15_73_27_Top2"
    },
    {
        "model": "ET_500_d15",
        "distribution": "73_27",
        "top_pct": 1.0,
        "name": "ET_500_d15_73_27_Top1"
    },
    {
        "model": "RF_400_d12",
        "distribution": "73_27",
        "top_pct": 3.0,
        "name": "RF_400_d12_73_27_Top3"
    },
    {
        "model": "ET_300_d10",
        "distribution": "73_27",
        "top_pct": 7.5,
        "name": "ET_300_d10_73_27_Top7_5"
    },
    {
        "model": "GB_200_d3",
        "distribution": "73_27",
        "top_pct": 10.0,
        "name": "GB_200_d3_73_27_Top10"
    }
]


# ============================================================
# UTILITAIRES
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def calculate_profit(actual, odds, stake=1.0):
    """
    Calcul du profit bookmaker classique :
      win  = +(odds - 1) * stake
      loss = -stake
    """
    actual = np.asarray(actual)

    wins = actual == 1
    losses = actual == 0

    profit = (
        wins.sum() * (odds - 1.0) * stake
        - losses.sum() * stake
    )

    return float(profit)


def calculate_roi(actual, odds, stake=1.0):
    actual = np.asarray(actual)

    if len(actual) == 0:
        return np.nan

    profit = calculate_profit(actual, odds, stake)
    total_stake = len(actual) * stake

    if total_stake == 0:
        return np.nan

    return profit / total_stake


def calculate_max_losing_streak(actual):
    """
    Plus longue série consécutive de pertes.
    """
    max_streak = 0
    current = 0

    for value in actual:
        if int(value) == 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return int(max_streak)


def calculate_max_drawdown(actual, odds, stake=1.0):
    """
    Maximum drawdown en unités.

    Exemple :
      bankroll : 0 -> 2.2 -> 1.2 -> -1.0

    Le drawdown mesure la chute depuis le dernier sommet.
    """

    bankroll = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for value in actual:

        if int(value) == 1:
            bankroll += (odds - 1.0) * stake
        else:
            bankroll -= stake

        peak = max(peak, bankroll)

        drawdown = peak - bankroll

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

    return float(max_drawdown)


def fair_odds(probability):
    if probability <= 0:
        return np.inf

    return 1.0 / probability


def break_even_probability(odds):
    if odds <= 0:
        return np.nan

    return 1.0 / odds


def get_top_selection(df, top_pct):
    """
    Sélectionne le top X% à l'intérieur du dataframe.

    Le classement est fait uniquement avec probability.
    """

    if df.empty:
        return df.copy()

    n = len(df)

    count = max(
        1,
        int(np.ceil(n * top_pct / 100.0))
    )

    ranked = df.sort_values(
        "probability",
        ascending=False
    )

    return ranked.head(count).copy()


# ============================================================
# CHARGEMENT
# ============================================================

print_header("FINAL OUT-OF-SAMPLE VALIDATION")

print(f"Dataset predictions : {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):

    print()
    print("ERREUR : fichier introuvable.")
    print(INPUT_FILE)
    print()
    print("Lance d'abord :")
    print("python3 model_optimization.py")
    sys.exit(1)


df = pd.read_csv(INPUT_FILE)

print()
print(f"Rows chargées : {len(df):,}")
print(f"Colonnes       : {len(df.columns)}")


# ============================================================
# VALIDATION DES COLONNES
# ============================================================

required_columns = [
    "model",
    "distribution",
    "fold",
    "dataset_index",
    "probability",
    "actual"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    print()
    print("ERREUR : colonnes manquantes :")
    print(missing)
    sys.exit(1)


# ============================================================
# NETTOYAGE
# ============================================================

df = df.copy()

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

df["dataset_index"] = pd.to_numeric(
    df["dataset_index"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "model",
        "distribution",
        "fold",
        "dataset_index",
        "probability",
        "actual"
    ]
)

df["actual"] = df["actual"].astype(int)
df["fold"] = df["fold"].astype(int)
df["dataset_index"] = df["dataset_index"].astype(int)

df = df[
    df["actual"].isin([0, 1])
].copy()

df = df.sort_values(
    "dataset_index"
).reset_index(drop=True)


print()
print(f"Rows après nettoyage : {len(df):,}")

global_rate = df["actual"].mean()

print(
    f"Taux global de goal_after_87 : "
    f"{global_rate:.4%}"
)


# ============================================================
# SPLIT CHRONOLOGIQUE FINAL
# ============================================================

print_header("SPLIT CHRONOLOGIQUE")

unique_indices = np.sort(
    df["dataset_index"].unique()
)

n_indices = len(unique_indices)

if n_indices < 100:

    print(
        "ERREUR : pas assez d'indices temporels "
        "pour effectuer une validation OOS."
    )

    sys.exit(1)


split_position = int(
    np.floor(
        n_indices * (1.0 - OOS_FRACTION)
    )
)

split_position = max(
    1,
    min(split_position, n_indices - 1)
)

development_indices = unique_indices[
    :split_position
]

oos_indices = unique_indices[
    split_position:
]

development = df[
    df["dataset_index"].isin(
        development_indices
    )
].copy()

oos = df[
    df["dataset_index"].isin(
        oos_indices
    )
].copy()


print(
    f"Développement : "
    f"{len(development):,} rows"
)

print(
    f"OOS final     : "
    f"{len(oos):,} rows"
)

print(
    f"Part OOS       : "
    f"{len(oos) / len(df):.2%}"
)

print()
print(
    f"Index développement : "
    f"{development['dataset_index'].min():,}"
    f" -> "
    f"{development['dataset_index'].max():,}"
)

print(
    f"Index OOS           : "
    f"{oos['dataset_index'].min():,}"
    f" -> "
    f"{oos['dataset_index'].max():,}"
)


development_rate = development["actual"].mean()
oos_rate = oos["actual"].mean()

print()
print(
    f"Baseline développement : "
    f"{development_rate:.4%}"
)

print(
    f"Baseline OOS           : "
    f"{oos_rate:.4%}"
)


# ============================================================
# IDENTIFICATION DES FOLDS OOS
# ============================================================

print()
print("Folds présents dans OOS :")

for fold in sorted(oos["fold"].unique()):

    fold_df = oos[
        oos["fold"] == fold
    ]

    print(
        f"  Fold {fold}: "
        f"{len(fold_df):,} rows | "
        f"goal rate={fold_df['actual'].mean():.2%}"
    )


# ============================================================
# ANALYSE DES STRATEGIES SUR DEVELOPPEMENT
# ============================================================

print_header(
    "ÉVALUATION DES STRATÉGIES SUR DÉVELOPPEMENT"
)

development_results = []


for strategy in CANDIDATE_STRATEGIES:

    model = strategy["model"]
    distribution = strategy["distribution"]
    top_pct = strategy["top_pct"]

    subset = development[
        (development["model"] == model)
        &
        (development["distribution"] == distribution)
    ].copy()

    if subset.empty:
        print(
            f"[SKIP] {strategy['name']} "
            f"-> aucune donnée"
        )
        continue

    selected = get_top_selection(
        subset,
        top_pct
    )

    actual = selected["actual"].to_numpy()

    bets = len(actual)
    wins = int(actual.sum())
    losses = bets - wins

    win_rate = (
        wins / bets
        if bets > 0
        else np.nan
    )

    baseline = subset["actual"].mean()

    lift = (
        win_rate / baseline
        if baseline > 0
        else np.nan
    )

    profit = calculate_profit(
        actual,
        ODDS,
        STAKE
    )

    roi = calculate_roi(
        actual,
        ODDS,
        STAKE
    )

    development_results.append({
        "strategy": strategy["name"],
        "model": model,
        "distribution": distribution,
        "top_pct": top_pct,
        "matches_available": len(subset),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "baseline": baseline,
        "lift": lift,
        "odds": ODDS,
        "profit": profit,
        "roi": roi
    })


development_results_df = pd.DataFrame(
    development_results
)

if not development_results_df.empty:

    development_results_df = development_results_df.sort_values(
        ["roi", "win_rate"],
        ascending=False
    )

    print()

    print(
        development_results_df[
            [
                "strategy",
                "top_pct",
                "bets",
                "win_rate",
                "lift",
                "profit",
                "roi"
            ]
        ].to_string(
            index=False,
            formatters={
                "win_rate": "{:.2%}".format,
                "baseline": "{:.2%}".format,
                "lift": "{:.2f}x".format,
                "roi": "{:+.2%}".format
            }
        )
    )


development_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "development_strategy_results.csv"
    ),
    index=False
)


# ============================================================
# EVALUATION OOS
# ============================================================

print_header(
    "VALIDATION FINALE OUT-OF-SAMPLE"
)

print(
    "IMPORTANT : les résultats ci-dessous sont calculés "
    "uniquement sur la période OOS."
)

print(
    "Aucune sélection supplémentaire de stratégie "
    "n'est effectuée sur OOS."
)


oos_results = []


for strategy in CANDIDATE_STRATEGIES:

    model = strategy["model"]
    distribution = strategy["distribution"]
    top_pct = strategy["top_pct"]

    subset = oos[
        (oos["model"] == model)
        &
        (oos["distribution"] == distribution)
    ].copy()

    if subset.empty:

        print(
            f"[SKIP] {strategy['name']} "
            f"-> aucune donnée OOS"
        )

        continue

    selected = get_top_selection(
        subset,
        top_pct
    )

    selected = selected.sort_values(
        "dataset_index"
    ).copy()

    actual = selected["actual"].to_numpy()

    bets = len(actual)

    wins = int(actual.sum())

    losses = bets - wins

    win_rate = (
        wins / bets
        if bets > 0
        else np.nan
    )

    baseline = subset["actual"].mean()

    lift = (
        win_rate / baseline
        if baseline > 0
        else np.nan
    )

    profit = calculate_profit(
        actual,
        ODDS,
        STAKE
    )

    roi = calculate_roi(
        actual,
        ODDS,
        STAKE
    )

    avg_probability = (
        selected["probability"].mean()
        if bets > 0
        else np.nan
    )

    min_probability = (
        selected["probability"].min()
        if bets > 0
        else np.nan
    )

    max_probability = (
        selected["probability"].max()
        if bets > 0
        else np.nan
    )

    fair = fair_odds(
        avg_probability
    )

    max_streak = calculate_max_losing_streak(
        actual
    )

    max_dd = calculate_max_drawdown(
        actual,
        ODDS,
        STAKE
    )

    profitable_folds = 0
    total_folds = 0

    fold_rois = []

    for fold in sorted(
        selected["fold"].unique()
    ):

        fold_selected = selected[
            selected["fold"] == fold
        ]

        fold_actual = fold_selected[
            "actual"
        ].to_numpy()

        if len(fold_actual) == 0:
            continue

        fold_roi = calculate_roi(
            fold_actual,
            ODDS,
            STAKE
        )

        fold_rois.append(
            {
                "fold": fold,
                "roi": fold_roi,
                "bets": len(fold_actual),
                "wins": int(
                    fold_actual.sum()
                )
            }
        )

        total_folds += 1

        if fold_roi > 0:
            profitable_folds += 1

    profitable_fold_rate = (
        profitable_folds / total_folds
        if total_folds > 0
        else np.nan
    )

    bets_per_100 = (
        bets / len(subset) * 100
        if len(subset) > 0
        else np.nan
    )

    oos_results.append({
        "strategy": strategy["name"],
        "model": model,
        "distribution": distribution,
        "top_pct": top_pct,

        "oos_matches": len(subset),

        "bets": bets,
        "wins": wins,
        "losses": losses,

        "win_rate": win_rate,
        "baseline": baseline,
        "lift": lift,

        "odds": ODDS,
        "break_even": break_even_probability(
            ODDS
        ),

        "avg_probability": avg_probability,
        "min_probability": min_probability,
        "max_probability": max_probability,

        "fair_odds_from_avg_probability": fair,

        "stake": STAKE,
        "total_stake": bets * STAKE,
        "profit": profit,
        "roi": roi,

        "profitable_folds": profitable_folds,
        "total_folds": total_folds,
        "profitable_fold_rate": profitable_fold_rate,

        "bets_per_100_matches": bets_per_100,

        "max_losing_streak": max_streak,
        "max_drawdown": max_dd
    })


oos_results_df = pd.DataFrame(
    oos_results
)


# ============================================================
# AFFICHAGE OOS
# ============================================================

if not oos_results_df.empty:

    oos_results_df = oos_results_df.sort_values(
        [
            "roi",
            "profitable_fold_rate",
            "win_rate"
        ],
        ascending=False
    ).reset_index(drop=True)

    print()

    display_columns = [
        "strategy",
        "top_pct",
        "bets",
        "wins",
        "losses",
        "win_rate",
        "baseline",
        "lift",
        "profit",
        "roi",
        "profitable_fold_rate",
        "bets_per_100_matches",
        "max_losing_streak",
        "max_drawdown"
    ]

    print(
        oos_results_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "win_rate": "{:.2%}".format,
                "baseline": "{:.2%}".format,
                "lift": "{:.2f}x".format,
                "profit": "{:+.2f}".format,
                "roi": "{:+.2%}".format,
                "profitable_fold_rate": "{:.2%}".format,
                "bets_per_100_matches": "{:.2f}".format,
                "max_drawdown": "{:.2f}".format
            }
        )
    )


oos_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_oos_results.csv"
    ),
    index=False
)


# ============================================================
# OOS PAR FOLD
# ============================================================

print_header(
    "PERFORMANCE OOS PAR FOLD"
)

oos_fold_results = []


for strategy in CANDIDATE_STRATEGIES:

    model = strategy["model"]
    distribution = strategy["distribution"]
    top_pct = strategy["top_pct"]

    subset = oos[
        (oos["model"] == model)
        &
        (oos["distribution"] == distribution)
    ].copy()

    if subset.empty:
        continue

    selected = get_top_selection(
        subset,
        top_pct
    )

    selected = selected.sort_values(
        "dataset_index"
    )

    for fold in sorted(
        selected["fold"].unique()
    ):

        fold_df = selected[
            selected["fold"] == fold
        ].copy()

        actual = fold_df[
            "actual"
        ].to_numpy()

        bets = len(actual)

        wins = int(actual.sum())

        losses = bets - wins

        win_rate = (
            wins / bets
            if bets > 0
            else np.nan
        )

        baseline = fold_df["actual"].mean()

        profit = calculate_profit(
            actual,
            ODDS,
            STAKE
        )

        roi = calculate_roi(
            actual,
            ODDS,
            STAKE
        )

        oos_fold_results.append({
            "strategy": strategy["name"],
            "model": model,
            "distribution": distribution,
            "top_pct": top_pct,
            "fold": fold,
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "baseline": baseline,
            "lift": (
                win_rate / baseline
                if baseline > 0
                else np.nan
            ),
            "profit": profit,
            "roi": roi,
            "max_losing_streak": calculate_max_losing_streak(
                actual
            ),
            "max_drawdown": calculate_max_drawdown(
                actual,
                ODDS,
                STAKE
            )
        })


oos_fold_results_df = pd.DataFrame(
    oos_fold_results
)


if not oos_fold_results_df.empty:

    print()

    print(
        oos_fold_results_df.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2%}".format,
                "baseline": "{:.2%}".format,
                "lift": "{:.2f}x".format,
                "profit": "{:+.2f}".format,
                "roi": "{:+.2%}".format
            }
        )
    )


oos_fold_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_oos_by_fold.csv"
    ),
    index=False
)


# ============================================================
# SENSIBILITÉ AUX ODDS
# ============================================================

print_header(
    "SENSIBILITÉ AUX ODDS"

)

odds_results = []


for strategy in CANDIDATE_STRATEGIES:

    model = strategy["model"]
    distribution = strategy["distribution"]
    top_pct = strategy["top_pct"]

    subset = oos[
        (oos["model"] == model)
        &
        (oos["distribution"] == distribution)
    ].copy()

    if subset.empty:
        continue

    selected = get_top_selection(
        subset,
        top_pct
    )

    actual = selected[
        "actual"
    ].to_numpy()

    for odds in SENSITIVITY_ODDS:

        profit = calculate_profit(
            actual,
            odds,
            STAKE
        )

        roi = calculate_roi(
            actual,
            odds,
            STAKE
        )

        odds_results.append({
            "strategy": strategy["name"],
            "model": model,
            "distribution": distribution,
            "top_pct": top_pct,
            "odds": odds,
            "break_even_probability": 1.0 / odds,
            "bets": len(actual),
            "wins": int(actual.sum()),
            "losses": len(actual) - int(actual.sum()),
            "win_rate": (
                actual.mean()
                if len(actual) > 0
                else np.nan
            ),
            "profit": profit,
            "roi": roi
        })


odds_results_df = pd.DataFrame(
    odds_results
)


odds_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_oos_odds_sensitivity.csv"
    ),
    index=False
)


# ============================================================
# MATCHES SÉLECTIONNÉS
# ============================================================

print_header(
    "EXPORT DES MATCHES OOS SÉLECTIONNÉS"
)

selected_matches = []


for strategy in CANDIDATE_STRATEGIES:

    model = strategy["model"]
    distribution = strategy["distribution"]
    top_pct = strategy["top_pct"]

    subset = oos[
        (oos["model"] == model)
        &
        (oos["distribution"] == distribution)
    ].copy()

    if subset.empty:
        continue

    selected = get_top_selection(
        subset,
        top_pct
    ).copy()

    selected["strategy"] = strategy["name"]

    selected_matches.append(
        selected[
            [
                "strategy",
                "model",
                "distribution",
                "fold",
                "dataset_index",
                "probability",
                "actual"
            ]
        ]
    )


if selected_matches:

    selected_matches_df = pd.concat(
        selected_matches,
        ignore_index=True
    )

else:

    selected_matches_df = pd.DataFrame(
        columns=[
            "strategy",
            "model",
            "distribution",
            "fold",
            "dataset_index",
            "probability",
            "actual"
        ]
    )


selected_matches_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_oos_selected_matches.csv"
    ),
    index=False
)


# ============================================================
# RANKING FINAL
# ============================================================

print_header(
    "RANKING FINAL OOS"
)

if not oos_results_df.empty:

    ranking = oos_results_df.copy()

    # Score de robustesse simple.
    #
    # Ce score ne remplace PAS le ROI.
    # Il favorise :
    # - ROI positif
    # - plusieurs folds rentables
    # - bon win rate
    # - volume raisonnable
    #
    ranking["robustness_score"] = (
        ranking["roi"].fillna(-1.0) * 100
        +
        ranking["profitable_fold_rate"].fillna(0.0) * 20
        +
        ranking["lift"].fillna(0.0) * 5
        +
        np.log1p(
            ranking["bets"].fillna(0)
        ) * 0.5
    )

    ranking = ranking.sort_values(
        "robustness_score",
        ascending=False
    ).reset_index(drop=True)

else:

    ranking = pd.DataFrame()


ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_oos_ranking.csv"
    ),
    index=False
)


# ============================================================
# DIAGNOSTIC FINAL
# ============================================================

print_header(
    "DIAGNOSTIC FINAL"
)

report_lines = []

report_lines.append(
    "FINAL OUT-OF-SAMPLE VALIDATION REPORT"
)

report_lines.append(
    "=" * 80
)

report_lines.append("")

report_lines.append(
    "OBJECTIF"
)

report_lines.append(
    "Validation finale anti-overfitting d'un système "
    "de ranking ML pour goal_after_87."
)

report_lines.append("")

report_lines.append(
    "CONFIGURATION"
)

report_lines.append(
    f"Odds simulées : {ODDS:.2f}"
)

report_lines.append(
    f"Stake : {STAKE:.2f}"
)

report_lines.append(
    f"OOS fraction : {OOS_FRACTION:.2%}"
)

report_lines.append(
    f"Total rows : {len(df):,}"
)

report_lines.append(
    f"Development rows : {len(development):,}"
)

report_lines.append(
    f"OOS rows : {len(oos):,}"
)

report_lines.append("")

report_lines.append(
    "BASELINES"
)

report_lines.append(
    f"Global : {global_rate:.2%}"
)

report_lines.append(
    f"Development : {development_rate:.2%}"
)

report_lines.append(
    f"OOS : {oos_rate:.2%}"
)

report_lines.append("")

report_lines.append(
    "INTERPRÉTATION"
)

report_lines.append(
    "Les résultats OOS doivent être considérés comme "
    "la première vraie mesure hors-échantillon du système."
)

report_lines.append(
    "Ils ne doivent pas être utilisés pour sélectionner "
    "à nouveau une stratégie avant une prochaine validation."
)

report_lines.append(
    "Les odds 3.20 sont hypothétiques et ne représentent "
    "pas les vraies odds disponibles à la 87e minute."
)

report_lines.append("")

if not oos_results_df.empty:

    best = oos_results_df.iloc[0]

    report_lines.append(
        "MEILLEURE STRATÉGIE SELON LE ROI OOS"
    )

    report_lines.append(
        f"Strategy : {best['strategy']}"
    )

    report_lines.append(
        f"Bets : {int(best['bets'])}"
    )

    report_lines.append(
        f"Wins : {int(best['wins'])}"
    )

    report_lines.append(
        f"Losses : {int(best['losses'])}"
    )

    report_lines.append(
        f"Win rate : {best['win_rate']:.2%}"
    )

    report_lines.append(
        f"Baseline : {best['baseline']:.2%}"
    )

    report_lines.append(
        f"Lift : {best['lift']:.2f}x"
    )

    report_lines.append(
        f"Profit simulé : {best['profit']:+.2f}"
    )

    report_lines.append(
        f"ROI simulé : {best['roi']:+.2%}"
    )

    report_lines.append(
        f"Profitable folds : "
        f"{int(best['profitable_folds'])}/"
        f"{int(best['total_folds'])}"
    )

    report_lines.append(
        f"Max losing streak : "
        f"{int(best['max_losing_streak'])}"
    )

    report_lines.append(
        f"Max drawdown : "
        f"{best['max_drawdown']:.2f}"
    )

    report_lines.append("")

    # Diagnostic de profitabilité.
    if best["roi"] > 0:

        report_lines.append(
            "DIAGNOSTIC : ROI OOS POSITIF."
        )

        report_lines.append(
            "Cela indique que la stratégie conserve "
            "un signal sur la période finale."
        )

    else:

        report_lines.append(
            "DIAGNOSTIC : ROI OOS NÉGATIF."
        )

        report_lines.append(
            "Le signal observé précédemment ne se "
            "confirme pas suffisamment sur la période finale."
        )

    report_lines.append("")

    if best["win_rate"] > break_even_probability(ODDS):

        report_lines.append(
            "Le win rate OOS est supérieur au seuil "
            "de break-even correspondant aux odds simulées."
        )

    else:

        report_lines.append(
            "Le win rate OOS est inférieur au seuil "
            "de break-even correspondant aux odds simulées."
        )

    report_lines.append("")

report_lines.append(
    "PROCHAINE ÉTAPE"
)

report_lines.append(
    "Remplacer les odds fixes 3.20 par les vraies odds "
    "du marché à la 87e minute."
)

report_lines.append(
    "Ensuite effectuer un backtest avec les odds historiques "
    "réelles afin de mesurer le véritable edge économique."
)

report_lines.append(
    "Avant toute utilisation en production, une nouvelle "
    "période OOS totalement vierge doit être conservée."
)

report_lines.append("")

report_lines.append(
    "FICHIERS PRODUITS"
)

report_lines.append(
    "- development_strategy_results.csv"
)

report_lines.append(
    "- final_oos_results.csv"
)

report_lines.append(
    "- final_oos_by_fold.csv"
)

report_lines.append(
    "- final_oos_odds_sensitivity.csv"
)

report_lines.append(
    "- final_oos_selected_matches.csv"
)

report_lines.append(
    "- final_oos_ranking.csv"
)

report_lines.append(
    "- final_oos_report.txt"
)


report_text = "\n".join(
    report_lines
)

report_path = os.path.join(
    OUTPUT_DIR,
    "final_oos_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report_text)


# ============================================================
# RÉSUMÉ TERMINAL
# ============================================================

print_header(
    "VALIDATION TERMINÉE"
)

print(
    f"Output directory : {OUTPUT_DIR}"
)

print()

if not oos_results_df.empty:

    print(
        "TOP STRATÉGIES OOS :"
    )

    top_display = oos_results_df.head(
        min(10, len(oos_results_df))
    )

    for rank, (_, row) in enumerate(
        top_display.iterrows(),
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{row['strategy']} | "
            f"Bets={int(row['bets'])} | "
            f"Win={row['win_rate']:.2%} | "
            f"ROI={row['roi']:+.2%} | "
            f"Folds={int(row['profitable_folds'])}/"
            f"{int(row['total_folds'])}"
        )

print()

print(
    "IMPORTANT :"
)

print(
    "Un ROI positif ici ne signifie PAS encore "
    "que le système est rentable en conditions réelles."
)

print(
    "Les odds 3.20 restent hypothétiques."
)

print()

print(
    "Prochaine étape après cette validation :"
)

print(
    "1. récupérer les vraies odds à la 87e minute"
)

print(
    "2. refaire le backtest avec ces odds"
)

print(
    "3. mesurer l'edge réel"
)

print(
    "4. construire ensuite le scanner live"
)

print()

print(
    "Rapport :"
)

print(report_path)

print()
print("=" * 80)
print("FIN")
print("=" * 80)