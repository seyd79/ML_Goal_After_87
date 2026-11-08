from pathlib import Path
from math import sqrt
import pandas as pd

SOURCE = Path(
    "outputs/v4_research/V4_TRAIN278_B29_RESEARCH_V1.csv"
)

print("=" * 72)
print("ÉTAPE 222 — AUDIT DU CONTEXTE DES MATCHS V4.1")
print("=" * 72)

df = pd.read_csv(SOURCE, dtype={"match_id": str})

required = [
    "match_id", "goal_after_87",
    "score_87_home", "score_87_away",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "home_red_cards_87", "away_red_cards_87",
]

if not set(required).issubset(df.columns):
    raise SystemExit("Colonnes nécessaires manquantes")

if len(df) != 278 or df.match_id.duplicated().any():
    raise SystemExit("TRAIN278 incorrect")

for col in required[1:]:
    df[col] = pd.to_numeric(df[col], errors="raise")
    if df[col].isna().any():
        raise SystemExit(f"Valeurs manquantes : {col}")

y = df.goal_after_87.astype(int)

if not y.isin([0, 1]).all() or int(y.sum()) != 72:
    raise SystemExit("Labels inattendus")

home = df.score_87_home
away = df.score_87_away

draw = home == away
one_goal = (home - away).abs() == 1
large_gap = (home - away).abs() >= 2

trailing_shots = pd.Series(0, index=df.index)
trailing_sot = pd.Series(0, index=df.index)

trailing_shots.loc[home < away] = (
    df.loc[home < away, "shots_home_last5"]
)
trailing_shots.loc[away < home] = (
    df.loc[away < home, "shots_away_last5"]
)

trailing_sot.loc[home < away] = (
    df.loc[home < away, "sot_home_last5"]
)
trailing_sot.loc[away < home] = (
    df.loc[away < home, "sot_away_last5"]
)

any_red = (
    df.home_red_cards_87 + df.away_red_cards_87
) >= 1

red_imbalance = (
    df.home_red_cards_87 != df.away_red_cards_87
)

groups = {
    "Match nul à 87'": draw,
    "Écart de 1 but": one_goal,
    "Écart de 2+ buts": large_gap,
    "Au moins 1 carton rouge": any_red,
    "Déséquilibre cartons rouges": red_imbalance,
    "Équipe menée : 2+ tirs récents":
        (~draw) & (trailing_shots >= 2),
    "Équipe menée : 1+ tir cadré récent":
        (~draw) & (trailing_sot >= 1),
    "Match nul : 3+ tirs récents":
        draw & (
            df.shots_home_last5 +
            df.shots_away_last5 >= 3
        ),
}


def wilson(wins, n):
    if n == 0:
        return None

    z = 1.96
    p = wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    margin = (
        z * sqrt(
            p*(1-p)/n + z*z/(4*n*n)
        ) / den
    )
    return center - margin, center + margin


base = y.mean()

print("\n=== POPULATION ===")
print("Matchs :", len(df))
print("Positifs :", int(y.sum()))
print(f"Taux de base : {base:.1%}")

print("\n=== FRÉQUENCES PAR CONTEXTE ===")

for name, mask in groups.items():
    n = int(mask.sum())
    wins = int(y[mask].sum())

    if n == 0:
        print(f"{name}: aucun match")
        continue

    rate = wins / n
    low, high = wilson(wins, n)

    print(f"\n{name}")
    print(f"  Effectif       : {n}")
    print(f"  But après 87'  : {wins}")
    print(f"  Réussite       : {rate:.1%}")
    print(f"  Écart à la base: {(rate-base)*100:+.1f} points")
    print(
        f"  IC Wilson 95% : "
        f"[{low:.1%}, {high:.1%}]"
    )

    if n < 20:
        print("  ATTENTION : très petit effectif")

print("\n=== LIMITES ===")
print("Analyse descriptive sur TRAIN uniquement.")
print("Groupes potentiellement chevauchants.")
print("Aucun filtre validé pour les paris.")
print("Aucun seuil optimisé.")
print("Aucun modèle entraîné.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 222 TERMINÉE.")
