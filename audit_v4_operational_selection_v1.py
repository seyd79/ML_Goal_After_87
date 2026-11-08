from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)

PRED = Path(
    "outputs/v4_research/"
    "V4_BASELINE_VALIDATION_PREDICTIONS_V1.csv"
)

A = "V4-A23-LR"
B = "V4-B29-LR"

df = pd.read_csv(PRED, dtype={"match_id": str})

if len(df) != 154 or df.match_id.duplicated().any():
    raise SystemExit("Population VALIDATION incorrecte")

df["date"] = pd.to_datetime(
    df["kickoff_utc"], utc=True, errors="raise"
)
df["y"] = pd.to_numeric(
    df["goal_after_87"], errors="raise"
)
df["p"] = pd.to_numeric(df[B], errors="raise")

if not df["y"].isin([0, 1]).all():
    raise SystemExit("Labels invalides")

if not df["p"].between(0, 1).all():
    raise SystemExit("Probabilités invalides")

if not df[A].between(0, 1).all():
    raise SystemExit("Probabilités A23 invalides")

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    usecols=["match_id", "split"],
    dtype={"match_id": str},
)

original = set(
    plan.loc[
        plan["split"] == "VALIDATION", "match_id"
    ]
)

batch_ids = set()

for path in sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv")):
    part = pd.read_csv(
        path,
        usecols=["match_id", "research_split"],
        dtype={"match_id": str},
    )
    batch_ids.update(
        part.loc[
            part["research_split"] == "VALIDATION",
            "match_id",
        ]
    )

def source(match_id):
    if match_id in original:
        return "ALL_320"
    if match_id in batch_ids:
        return "BATCH02_11"
    return "INCONNU"

df["source"] = df["match_id"].map(source)

if (df["source"] == "INCONNU").any():
    raise SystemExit("Sources non identifiées")

df = df.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

df["period"] = np.where(
    np.arange(len(df)) < len(df) // 2,
    "PREMIERE_MOITIE",
    "DEUXIEME_MOITIE",
)

def wilson(wins, n):
    if n == 0:
        return "N/A"
    z = 1.96
    p = wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    radius = z * np.sqrt(
        p*(1-p)/n + z*z/(4*n*n)
    ) / den
    return f"{center-radius:.1%} à {center+radius:.1%}"

def report(group, threshold):
    selected = group.loc[group["p"] >= threshold]
    n = len(selected)
    wins = int(selected["y"].sum())

    rate = f"{wins/n:.1%}" if n else "N/A"
    coverage = n / len(group) if len(group) else 0

    return (
        f"{n:3d} sélections | "
        f"{wins:3d} buts | "
        f"réussite={rate:>6s} | "
        f"couverture={coverage:5.1%}"
    )

print("=" * 75)
print("ÉTAPE 169 — SÉLECTION OPÉRATIONNELLE V4-B29-LR")
print("=" * 75)

print("\n=== POPULATION VALIDATION ===")
print("Matchs :", len(df))
print("Buts tardifs :", int(df["y"].sum()))
print("Taux de base :", f"{df['y'].mean():.1%}")

print("\n=== DISTRIBUTION DES PROBABILITÉS ===")
print(
    df["p"].describe(
        percentiles=[.10, .25, .50, .75, .90, .95]
    ).round(4).to_string()
)

thresholds = [.30, .35, .40, .45, .50, .55, .60]

print("\n=== SEUILS FIXES — VALIDATION COMPLÈTE ===")
for threshold in thresholds:
    print(
        f"p >= {threshold:.2f} | "
        + report(df, threshold)
    )

print("\n=== SEUILS FIXES — STABILITÉ TEMPORELLE ===")
for period, group in df.groupby("period", sort=False):
    print(
        f"\n{period} | "
        f"n={len(group)} | "
        f"taux de base={group['y'].mean():.1%}"
    )
    for threshold in thresholds:
        print(
            f"p >= {threshold:.2f} | "
            + report(group, threshold)
        )

print("\n=== SEUILS FIXES — PROVENANCE ===")
for name, group in df.groupby("source"):
    print(
        f"\n{name} | n={len(group)} | "
        f"taux de base={group['y'].mean():.1%}"
    )
    for threshold in thresholds:
        print(
            f"p >= {threshold:.2f} | "
            + report(group, threshold)
        )

print("\n=== INTERVALLES DE CONFIANCE EXPLORATOIRES ===")
for threshold in thresholds:
    selected = df.loc[df["p"] >= threshold]
    n = len(selected)
    wins = int(selected["y"].sum())
    if n:
        print(
            f"p >= {threshold:.2f} | "
            f"{wins}/{n} | IC Wilson : {wilson(wins, n)}"
        )

print("\n=== CALIBRATION PAR TRANCHES ===")
bins = [0, .20, .30, .40, .50, .60, .70, .80, 1.000001]

df["prob_band"] = pd.cut(
    df["p"], bins=bins,
    right=False,
    include_lowest=True,
)

for band, group in df.groupby(
    "prob_band", observed=True
):
    print(
        f"{band} | n={len(group):3d} | "
        f"proba moyenne={group['p'].mean():.1%} | "
        f"réussite réelle={group['y'].mean():.1%}"
    )

print("\n=== CONTRÔLES ===")
print("Sources :", df["source"].value_counts().to_dict())
print("TEST_FUTUR non chargé.")
print("Aucun seuil optimisé automatiquement.")
print("Aucun réentraînement.")
print("Scanner V3-C inchangé.")
print("RÉSULTATS EXPLORATOIRES UNIQUEMENT.")
