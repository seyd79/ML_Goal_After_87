from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PATH = Path(
    "outputs/v4_research/"
    "V4_BASELINE_VALIDATION_PREDICTIONS_V1.csv"
)

df = pd.read_csv(PATH, dtype={"match_id": str})

A = "V4-A23-LR"
B = "V4-B29-LR"

required = [
    "match_id", "kickoff_utc", "goal_after_87", A, B
]

missing = set(required) - set(df.columns)
if missing:
    raise SystemExit(f"Colonnes absentes : {sorted(missing)}")

if len(df) != 154 or df.match_id.duplicated().any():
    raise SystemExit("Population validation inattendue")

df["date"] = pd.to_datetime(
    df.kickoff_utc, utc=True, errors="raise"
)
df = df.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

y = pd.to_numeric(
    df.goal_after_87, errors="raise"
).to_numpy()

pa = pd.to_numeric(df[A], errors="raise").to_numpy()
pb = pd.to_numeric(df[B], errors="raise").to_numpy()

if not set(np.unique(y)).issubset({0, 1}):
    raise SystemExit("Labels non binaires")

if not np.isfinite(pa).all() or not np.isfinite(pb).all():
    raise SystemExit("Probabilités non finies")

if not (
    ((pa >= 0) & (pa <= 1)).all()
    and ((pb >= 0) & (pb <= 1)).all()
):
    raise SystemExit("Probabilités hors de [0,1]")

y = y.astype(int)

def top_idx(p, k):
    return np.argsort(-p, kind="stable")[:k]

def precision_at(y, p, k):
    return float(y[top_idx(p, k)].mean())

def wilson(wins, n, z=1.96):
    p = wins / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    radius = (
        z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    )
    return center-radius, center+radius

print("=" * 70)
print("ÉTAPE 168 — AUDIT DE DISCRIMINATION V4")
print("=" * 70)

print("\n=== PERFORMANCE GLOBALE ===")
print("Matchs :", len(df))
print("Positifs :", int(y.sum()))
print("Fréquence :", f"{y.mean():.1%}")

for name, p in [(A, pa), (B, pb)]:
    print(
        name,
        "| AUC =", f"{roc_auc_score(y, p):.4f}",
        "| AP =", f"{average_precision_score(y, p):.4f}"
    )

print("\n=== COMPARAISON DES SÉLECTIONS ===")

for k in (10, 20, 30, 50):
    ia = set(top_idx(pa, k))
    ib = set(top_idx(pb, k))

    wins_a = int(y[list(ia)].sum())
    wins_b = int(y[list(ib)].sum())

    lo, hi = wilson(wins_b, k)

    print(f"\nTOP {k}")
    print(f"  A23 : {wins_a}/{k}")
    print(f"  B29 : {wins_b}/{k}")
    print(f"  Matchs communs : {len(ia & ib)}/{k}")
    print(f"  Nouveaux matchs B29 : {len(ib - ia)}")
    print(f"  Gagnants entrants B29 : {int(y[list(ib-ia)].sum())}")
    print(f"  Gagnants sortants A23 : {int(y[list(ia-ib)].sum())}")
    print(
        "  IC Wilson 95 % B29 :",
        f"[{lo:.1%} ; {hi:.1%}]"
    )

print("\n=== STABILITÉ CHRONOLOGIQUE ===")

mid = len(df) // 2

for label, subset in [
    ("PREMIÈRE MOITIÉ", df.iloc[:mid]),
    ("DEUXIÈME MOITIÉ", df.iloc[mid:]),
]:
    yy = subset.goal_after_87.astype(int).to_numpy()
    pp = subset[B].astype(float).to_numpy()

    print(f"\n{label}")
    print("  Dates :", subset.date.min(), "à", subset.date.max())
    print("  Matchs :", len(yy))
    print("  Fréquence positive :", f"{yy.mean():.1%}")

    if len(np.unique(yy)) == 2:
        print("  AUC :", f"{roc_auc_score(yy, pp):.4f}")
        print(
            "  AP :",
            f"{average_precision_score(yy, pp):.4f}"
        )

    for k in (10, 20):
        wins = int(yy[top_idx(pp, k)].sum())
        print(f"  TOP {k} : {wins}/{k} ({wins/k:.1%})")

print("\n=== INCERTITUDE PAR BOOTSTRAP APPARIÉ ===")

rng = np.random.default_rng(42)
auc_delta = []
ap_delta = []
top20_b = []

for _ in range(3000):
    idx = rng.integers(0, len(y), len(y))

    yy = y[idx]
    aa = pa[idx]
    bb = pb[idx]

    if len(np.unique(yy)) < 2:
        continue

    auc_delta.append(
        roc_auc_score(yy, bb) - roc_auc_score(yy, aa)
    )

    ap_delta.append(
        average_precision_score(yy, bb)
        - average_precision_score(yy, aa)
    )

    top20_b.append(precision_at(yy, bb, 20))

def report(label, values):
    low, high = np.percentile(values, [2.5, 97.5])
    print(
        label,
        "| IC bootstrap 95 % :",
        f"[{low:+.4f} ; {high:+.4f}]"
    )

report("Gain AUC B29-A23", auc_delta)
report("Gain AP B29-A23", ap_delta)
report("TOP20 B29", top20_b)

print("\n=== FIN AUDIT ===")
print("Validation uniquement.")
print("Bootstrap exploratoire, non prospectif.")
print("Aucun modèle réentraîné.")
print("TEST_FUTUR non consulté.")
print("V3-C LIVE inchangé.")
