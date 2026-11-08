from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

PATH = Path(
    "outputs/v4_research/V4_SIMPLE_CHRONO_OOF_V1.csv"
)

df = pd.read_csv(PATH, dtype={"match_id": str})

MODELS = ["B29_C1", "SIMPLE6_C01", "SIMPLE8_C01"]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60]

required = {
    "match_id", "source", "fold", "y", *MODELS
}

if not required.issubset(df.columns):
    raise SystemExit("Colonnes OOF manquantes")

if len(df) != 121 or df.match_id.duplicated().any():
    raise SystemExit("Population OOF inattendue")

if not df.y.isin([0, 1]).all():
    raise SystemExit("Labels invalides")

if not df[MODELS].apply(
    lambda x: x.between(0, 1).all()
).all():
    raise SystemExit("Probabilités invalides")

def wilson(wins, n):
    if n == 0:
        return None

    z = 1.96
    p = wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(
        p*(1-p)/n + z*z/(4*n*n)
    ) / den

    return max(0, center-half), min(1, center+half)

print("=" * 72)
print("ÉTAPE 187 — SÉLECTIONS CHRONOLOGIQUES V4")
print("=" * 72)

base_rate = float(df.y.mean())
baseline_brier = base_rate * (1-base_rate)

print("\n=== RÉFÉRENCE DESCRIPTIVE ===")
print("Matchs :", len(df))
print("Positifs :", int(df.y.sum()))
print("Taux observé :", f"{base_rate:.2%}")
print(
    "Brier d'une constante égale au taux observé :",
    f"{baseline_brier:.4f}"
)
print("Référence rétrospective, non prospective.")

print("\n=== PERFORMANCES GLOBALES ===")

for model in MODELS:
    p = df[model]

    print(
        f"{model:14s} | "
        f"AUC={roc_auc_score(df.y, p):.4f} | "
        f"AP={average_precision_score(df.y, p):.4f} | "
        f"Brier={brier_score_loss(df.y, p):.4f}"
    )

print("\n=== SÉLECTIONS À SEUIL FIXE ===")

for model in MODELS:
    print("\nMODÈLE :", model)

    for t in THRESHOLDS:
        g = df[df[model] >= t]
        n = len(g)
        wins = int(g.y.sum())

        if n == 0:
            print(f"Seuil {t:.2f} | aucune sélection")
            continue

        low, high = wilson(wins, n)

        print(
            f"Seuil {t:.2f} | "
            f"sélections={n:3d} | "
            f"réussites={wins:3d} | "
            f"taux={wins/n:.1%} | "
            f"couverture={n/len(df):.1%} | "
            f"IC95≈[{low:.1%}; {high:.1%}]"
        )

print("\n=== TOP 10 PAR BLOC CHRONOLOGIQUE ===")

for fold, g in df.groupby("fold", sort=True):
    print(
        f"\nBLOC {fold} | n={len(g)} | "
        f"positifs={int(g.y.sum())} | "
        f"taux={g.y.mean():.1%}"
    )

    for model in MODELS:
        top = g.nlargest(10, model)
        wins = int(top.y.sum())

        print(
            f"  {model:14s} | "
            f"TOP10={wins}/10 ({wins/10:.0%}) | "
            f"proba_max={g[model].max():.3f}"
        )

print("\n=== SIMPLE6 : SÉLECTIONS PAR SOURCE ET BLOC ===")

for t in (0.40, 0.50):
    print(f"\nSEUIL SIMPLE6 >= {t:.2f}")

    for (fold, source), g in df.groupby(
        ["fold", "source"], sort=True
    ):
        selected = g[g["SIMPLE6_C01"] >= t]

        if selected.empty:
            print(
                f"BLOC {fold} | {source:11s} | "
                "0 sélection"
            )
            continue

        n = len(selected)
        wins = int(selected.y.sum())

        print(
            f"BLOC {fold} | {source:11s} | "
            f"{wins}/{n} = {wins/n:.1%}"
        )

print("\n=== DISTRIBUTION DES PROBABILITÉS ===")

for model in MODELS:
    p = df[model]

    print(
        f"{model:14s} | "
        f"min={p.min():.3f} | "
        f"médiane={p.median():.3f} | "
        f"max={p.max():.3f} | "
        f">=0.80={int((p>=0.80).sum())}"
    )

print("\n=== BILAN ===")
print("Diagnostic exploratoire sur 121 matchs OOF.")
print("Aucun seuil optimisé automatiquement.")
print("Aucun réentraînement.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Scanner LIVE inchangé.")
