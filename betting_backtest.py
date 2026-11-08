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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "betting_backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cote moyenne supposée à la 87e minute
DEFAULT_ODDS = 3.20

# Mise par pari
STAKE = 1.0

# Zones de sélection
TOP_PERCENTAGES = [1, 2, 3, 5, 10, 15, 20, 25, 30]

# Cotes à tester pour la sensibilité
ODDS_TO_TEST = [2.50, 2.75, 3.00, 3.20, 3.40, 3.50, 4.00]

# Modèles/distributions à analyser
# None = analyser toutes les combinaisons
MODELS_TO_TEST = None
DISTRIBUTIONS_TO_TEST = None


# ============================================================
# OUTILS
# ============================================================

def safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def calculate_roi(profit, total_staked):
    return safe_div(profit, total_staked)


def calculate_breakeven_winrate(odds):
    return 1 / odds


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 80)
print("BETTING BACKTEST - GOAL AFTER 87'")
print("=" * 80)

print(f"\n📂 Lecture du fichier :")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\n❌ Fichier introuvable :\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\n✅ Dataset chargé")
print(f"   Lignes : {len(df):,}")
print(f"   Colonnes : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "model",
    "distribution",
    "fold",
    "dataset_index",
    "probability",
    "actual",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"\n❌ Colonnes manquantes : {missing}"
    )

df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
df["fold"] = pd.to_numeric(df["fold"], errors="coerce")

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

# Vérification de la cible
invalid_actual = ~df["actual"].isin([0, 1])

if invalid_actual.any():
    raise ValueError(
        "❌ La colonne 'actual' contient des valeurs autres que 0/1."
    )


# ============================================================
# INFORMATIONS DATASET
# ============================================================

print("\n" + "=" * 80)
print("INFORMATIONS GÉNÉRALES")
print("=" * 80)

print(f"\nMatchs uniques approximatifs dans les prédictions :")
print(f"   {df['dataset_index'].nunique():,}")

print(f"\nNombre de modèles : {df['model'].nunique()}")
print(f"Nombre de distributions : {df['distribution'].nunique()}")
print(f"Nombre de folds : {df['fold'].nunique()}")

print("\nModèles disponibles :")
for model in sorted(df["model"].unique()):
    print(f"   - {model}")

print("\nDistributions disponibles :")
for distribution in sorted(df["distribution"].unique()):
    print(f"   - {distribution}")


# ============================================================
# FILTRES OPTIONNELS
# ============================================================

if MODELS_TO_TEST is not None:
    df = df[df["model"].isin(MODELS_TO_TEST)].copy()

if DISTRIBUTIONS_TO_TEST is not None:
    df = df[df["distribution"].isin(DISTRIBUTIONS_TO_TEST)].copy()


# ============================================================
# FONCTION BACKTEST
# ============================================================

def run_backtest(group, top_percentage, odds):
    """
    Sélectionne les matchs ayant les probabilités les plus élevées
    dans chaque fold.

    Important :
    Le classement est fait séparément pour chaque fold afin d'éviter
    de comparer les probabilités entre périodes différentes.
    """

    selected_parts = []

    for fold, fold_df in group.groupby("fold"):

        fold_df = fold_df.sort_values(
            "probability",
            ascending=False
        ).copy()

        n_matches = len(fold_df)

        n_selected = max(
            1,
            int(np.ceil(n_matches * top_percentage / 100))
        )

        selected = fold_df.head(n_selected).copy()

        selected["selected"] = True
        selected["odds"] = odds
        selected["stake"] = STAKE

        # Résultat du pari
        selected["win"] = selected["actual"] == 1

        # Profit :
        # gagnant = odds - 1
        # perdant = -1
        selected["profit"] = np.where(
            selected["win"],
            STAKE * (odds - 1),
            -STAKE
        )

        selected_parts.append(selected)

    if not selected_parts:
        return pd.DataFrame()

    return pd.concat(selected_parts, ignore_index=True)


# ============================================================
# BACKTEST GLOBAL
# ============================================================

print("\n" + "=" * 80)
print("BACKTEST PRINCIPAL")
print("=" * 80)

print(f"\n💰 Cote utilisée : {DEFAULT_ODDS:.2f}")
print(f"💵 Mise par pari : {STAKE:.2f}")
print(
    f"🎯 Seuil de rentabilité : "
    f"{calculate_breakeven_winrate(DEFAULT_ODDS) * 100:.2f}%"
)


results = []
selected_details = []

group_columns = ["model", "distribution"]

for (model, distribution), group in df.groupby(group_columns):

    total_tested = len(group)

    for top_percentage in TOP_PERCENTAGES:

        selected = run_backtest(
            group,
            top_percentage,
            DEFAULT_ODDS
        )

        if selected.empty:
            continue

        total_detected = len(selected)
        wins = int(selected["win"].sum())
        losses = total_detected - wins

        win_rate = safe_div(wins, total_detected)

        total_staked = total_detected * STAKE
        profit = selected["profit"].sum()

        roi = calculate_roi(
            profit,
            total_staked
        )

        average_probability = selected["probability"].mean()
        minimum_probability = selected["probability"].min()
        maximum_probability = selected["probability"].max()

        fair_odds = (
            safe_div(1, average_probability)
            if average_probability > 0
            else np.nan
        )

        lift = safe_div(
            win_rate,
            group["actual"].mean()
        )

        # Stabilité par fold
        fold_results = []

        for fold, fold_selected in selected.groupby("fold"):

            fold_wins = int(fold_selected["win"].sum())
            fold_detected = len(fold_selected)

            fold_win_rate = safe_div(
                fold_wins,
                fold_detected
            )

            fold_profit = fold_selected["profit"].sum()

            fold_staked = fold_detected * STAKE

            fold_roi = calculate_roi(
                fold_profit,
                fold_staked
            )

            fold_results.append({
                "fold": fold,
                "detected": fold_detected,
                "wins": fold_wins,
                "losses": fold_detected - fold_wins,
                "win_rate": fold_win_rate,
                "profit": fold_profit,
                "roi": fold_roi,
            })

        fold_df = pd.DataFrame(fold_results)

        positive_folds = int(
            (fold_df["profit"] > 0).sum()
        )

        profitable_fold_rate = safe_div(
            positive_folds,
            len(fold_df)
        )

        worst_fold_roi = fold_df["roi"].min()
        best_fold_roi = fold_df["roi"].max()

        results.append({
            "model": model,
            "distribution": distribution,
            "top_percentage": top_percentage,
            "total_matches_tested": total_tested,
            "matches_detected": total_detected,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "baseline_rate": group["actual"].mean(),
            "lift": lift,
            "odds": DEFAULT_ODDS,
            "breakeven_win_rate": calculate_breakeven_winrate(
                DEFAULT_ODDS
            ),
            "average_probability": average_probability,
            "minimum_probability": minimum_probability,
            "maximum_probability": maximum_probability,
            "fair_odds_from_average_probability": fair_odds,
            "total_staked": total_staked,
            "profit": profit,
            "roi": roi,
            "positive_folds": positive_folds,
            "total_folds": len(fold_df),
            "profitable_fold_rate": profitable_fold_rate,
            "worst_fold_roi": worst_fold_roi,
            "best_fold_roi": best_fold_roi,
        })

        selected["model"] = model
        selected["distribution"] = distribution
        selected["top_percentage"] = top_percentage

        selected_details.append(selected)


# ============================================================
# RESULTATS
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError(
        "❌ Aucun résultat de backtest n'a été généré."
    )

results_df = results_df.sort_values(
    [
        "roi",
        "win_rate",
        "matches_detected"
    ],
    ascending=False
).reset_index(drop=True)


# ============================================================
# AFFICHAGE PRINCIPAL
# ============================================================

print("\n" + "=" * 80)
print("🏆 MEILLEURES STRATÉGIES À COTE 3.20")
print("=" * 80)

display_columns = [
    "model",
    "distribution",
    "top_percentage",
    "total_matches_tested",
    "matches_detected",
    "wins",
    "losses",
    "win_rate",
    "lift",
    "profit",
    "roi",
    "profitable_fold_rate",
]

display_df = results_df[display_columns].head(30).copy()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

print(
    display_df.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: f"{x * 100:.2f}%",
            "lift": lambda x: f"{x:.2f}x",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x * 100:+.2f}%",
            "profitable_fold_rate": lambda x: f"{x * 100:.0f}%",
        }
    )
)


# ============================================================
# MEILLEUR TOP 5
# ============================================================

print("\n" + "=" * 80)
print("🔥 ANALYSE DU TOP 5")
print("=" * 80)

top5_df = results_df[
    results_df["top_percentage"] == 5
].copy()

top5_df = top5_df.sort_values(
    "roi",
    ascending=False
)

print(
    top5_df[display_columns].head(20).to_string(
        index=False,
        formatters={
            "win_rate": lambda x: f"{x * 100:.2f}%",
            "lift": lambda x: f"{x:.2f}x",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x * 100:+.2f}%",
            "profitable_fold_rate": lambda x: f"{x * 100:.0f}%",
        }
    )
)


# ============================================================
# MEILLEUR TOP 10
# ============================================================

print("\n" + "=" * 80)
print("🔥 ANALYSE DU TOP 10")
print("=" * 80)

top10_df = results_df[
    results_df["top_percentage"] == 10
].copy()

top10_df = top10_df.sort_values(
    "roi",
    ascending=False
)

print(
    top10_df[display_columns].head(20).to_string(
        index=False,
        formatters={
            "win_rate": lambda x: f"{x * 100:.2f}%",
            "lift": lambda x: f"{x:.2f}x",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x * 100:+.2f}%",
            "profitable_fold_rate": lambda x: f"{x * 100:.0f}%",
        }
    )
)


# ============================================================
# ANALYSE PAR FOLD
# ============================================================

print("\n" + "=" * 80)
print("📅 ANALYSE PAR FOLD")
print("=" * 80)

fold_results_all = []

if selected_details:

    all_selected = pd.concat(
        selected_details,
        ignore_index=True
    )

    for (
        model,
        distribution,
        top_percentage,
        fold
    ), fold_selected in all_selected.groupby(
        [
            "model",
            "distribution",
            "top_percentage",
            "fold"
        ]
    ):

        detected = len(fold_selected)
        wins = int(fold_selected["win"].sum())
        losses = detected - wins

        win_rate = safe_div(
            wins,
            detected
        )

        profit = fold_selected["profit"].sum()

        staked = detected * STAKE

        roi = calculate_roi(
            profit,
            staked
        )

        fold_results_all.append({
            "model": model,
            "distribution": distribution,
            "top_percentage": top_percentage,
            "fold": fold,
            "matches_detected": detected,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit": profit,
            "roi": roi,
        })

fold_results_df = pd.DataFrame(
    fold_results_all
)

fold_results_df = fold_results_df.sort_values(
    [
        "model",
        "distribution",
        "top_percentage",
        "fold"
    ]
)

print(
    fold_results_df.head(100).to_string(
        index=False,
        formatters={
            "win_rate": lambda x: f"{x * 100:.2f}%",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x * 100:+.2f}%",
        }
    )
)


# ============================================================
# TEST DE DIFFERENTES COTES
# ============================================================

print("\n" + "=" * 80)
print("💰 SENSIBILITÉ AUX COTES")
print("=" * 80)

odds_results = []

# On se concentre sur les stratégies principales
# Top 5, Top 10 et Top 15

for (model, distribution), group in df.groupby(
    ["model", "distribution"]
):

    for top_percentage in [5, 10, 15]:

        for odds in ODDS_TO_TEST:

            selected = run_backtest(
                group,
                top_percentage,
                odds
            )

            if selected.empty:
                continue

            detected = len(selected)
            wins = int(selected["win"].sum())
            losses = detected - wins

            profit = selected["profit"].sum()

            total_staked = detected * STAKE

            roi = calculate_roi(
                profit,
                total_staked
            )

            odds_results.append({
                "model": model,
                "distribution": distribution,
                "top_percentage": top_percentage,
                "odds": odds,
                "matches_detected": detected,
                "wins": wins,
                "losses": losses,
                "win_rate": safe_div(wins, detected),
                "breakeven_win_rate": calculate_breakeven_winrate(
                    odds
                ),
                "profit": profit,
                "roi": roi,
            })

odds_results_df = pd.DataFrame(
    odds_results
)

odds_results_df = odds_results_df.sort_values(
    "roi",
    ascending=False
)

print(
    odds_results_df.head(50).to_string(
        index=False,
        formatters={
            "win_rate": lambda x: f"{x * 100:.2f}%",
            "breakeven_win_rate": lambda x: f"{x * 100:.2f}%",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x * 100:+.2f}%",
        }
    )
)


# ============================================================
# MEILLEUR MODELE
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 80)
print("🏆 MEILLEURE STRATÉGIE OBSERVÉE")
print("=" * 80)

print(f"""
Modèle                  : {best['model']}
Distribution            : {best['distribution']}
Zone                    : Top {best['top_percentage']} %

Matchs testés            : {int(best['total_matches_tested']):,}
Matchs détectés          : {int(best['matches_detected']):,}

Gagnants                 : {int(best['wins']):,}
Perdants                 : {int(best['losses']):,}

Taux de réussite         : {best['win_rate'] * 100:.2f} %
Taux de base             : {best['baseline_rate'] * 100:.2f} %
Lift                     : {best['lift']:.2f}x

Cote simulée             : {best['odds']:.2f}
Seuil rentabilité        : {best['breakeven_win_rate'] * 100:.2f} %

Probabilité moyenne     : {best['average_probability'] * 100:.2f} %
Cote juste moyenne       : {best['fair_odds_from_average_probability']:.2f}

Mise totale              : {best['total_staked']:.2f}
Profit                   : {best['profit']:+.2f}
ROI                      : {best['roi'] * 100:+.2f} %

Folds rentables          : {int(best['positive_folds'])}/{int(best['total_folds'])}
Stabilité profitable     : {best['profitable_fold_rate'] * 100:.0f} %

Pire fold ROI            : {best['worst_fold_roi'] * 100:+.2f} %
Meilleur fold ROI        : {best['best_fold_roi'] * 100:+.2f} %
""")


# ============================================================
# SAUVEGARDE
# ============================================================

results_file = (
    OUTPUT_DIR
    / "betting_backtest_summary.csv"
)

fold_file = (
    OUTPUT_DIR
    / "betting_backtest_by_fold.csv"
)

odds_file = (
    OUTPUT_DIR
    / "odds_sensitivity.csv"
)

selected_file = (
    OUTPUT_DIR
    / "selected_matches.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

fold_results_df.to_csv(
    fold_file,
    index=False
)

odds_results_df.to_csv(
    odds_file,
    index=False
)

if selected_details:
    all_selected = pd.concat(
        selected_details,
        ignore_index=True
    )

    all_selected.to_csv(
        selected_file,
        index=False
    )


# ============================================================
# RAPPORT TEXTE
# ============================================================

report_file = (
    OUTPUT_DIR
    / "betting_backtest_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("BETTING BACKTEST - GOAL AFTER 87'\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        f"Cote principale : {DEFAULT_ODDS:.2f}\n"
    )

    f.write(
        f"Mise par pari : {STAKE:.2f}\n"
    )

    f.write(
        f"Seuil rentabilité : "
        f"{calculate_breakeven_winrate(DEFAULT_ODDS) * 100:.2f}%\n\n"
    )

    f.write("MEILLEURE STRATEGIE\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"Modèle : {best['model']}\n"
    )

    f.write(
        f"Distribution : {best['distribution']}\n"
    )

    f.write(
        f"Top : {best['top_percentage']}%\n"
    )

    f.write(
        f"Matchs testés : "
        f"{int(best['total_matches_tested'])}\n"
    )

    f.write(
        f"Matchs détectés : "
        f"{int(best['matches_detected'])}\n"
    )

    f.write(
        f"Gagnants : {int(best['wins'])}\n"
    )

    f.write(
        f"Perdants : {int(best['losses'])}\n"
    )

    f.write(
        f"Taux réussite : "
        f"{best['win_rate'] * 100:.2f}%\n"
    )

    f.write(
        f"Lift : {best['lift']:.2f}x\n"
    )

    f.write(
        f"Profit : {best['profit']:+.2f}\n"
    )

    f.write(
        f"ROI : {best['roi'] * 100:+.2f}%\n"
    )

    f.write(
        f"Folds rentables : "
        f"{int(best['positive_folds'])}/"
        f"{int(best['total_folds'])}\n"
    )

    f.write(
        f"Pire fold ROI : "
        f"{best['worst_fold_roi'] * 100:+.2f}%\n"
    )

    f.write(
        f"Meilleur fold ROI : "
        f"{best['best_fold_roi'] * 100:+.2f}%\n"
    )

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("ATTENTION\n")
    f.write("=" * 80 + "\n")

    f.write(
        "Les cotes sont simulées avec une cote fixe. "
        "Ce backtest ne constitue donc pas encore une preuve "
        "de rentabilité réelle.\n"
    )

    f.write(
        "La prochaine étape consiste à comparer ces résultats "
        "avec des cotes historiques réelles à la 87e minute.\n"
    )


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 80)
print("✅ BACKTEST TERMINÉ")
print("=" * 80)

print("\nFichiers créés :")

print(f"📄 {results_file}")
print(f"📄 {fold_file}")
print(f"📄 {odds_file}")
print(f"📄 {selected_file}")
print(f"📄 {report_file}")

print("\n🎯 La prochaine chose à regarder est :")
print("   1. Matchs détectés")
print("   2. Win rate")
print("   3. Profit")
print("   4. ROI")
print("   5. ROI de chaque fold")
print("   6. Sensibilité aux différentes cotes")

print("\n" + "=" * 80)