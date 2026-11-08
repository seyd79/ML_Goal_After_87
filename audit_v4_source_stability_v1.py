from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("outputs/pressure_historical/pressure_v2_dataset_join")
PRED = Path(
    "outputs/v4_research/"
    "V4_BASELINE_VALIDATION_PREDICTIONS_V1.csv"
)

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

base = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
    usecols=["match_id", "goal_after_87"] + B6,
)

base = base.merge(
    plan, on="match_id", validate="one_to_one"
)
base["source"] = "ALL_320"

parts = []
for path in sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv")):
    p = pd.read_csv(
        path,
        dtype={"match_id": str},
        usecols=[
            "match_id", "goal_after_87",
            "research_split",
        ] + B6,
    )
    p = p.rename(columns={"research_split": "split"})
    p["source"] = "BATCH02_11"
    parts.append(p)

if len(parts) != 10:
    raise SystemExit("10 lots B attendus")

data = pd.concat(
    [base] + parts, ignore_index=True
)

data["split"] = data["split"].astype(str).str.upper()

if len(data) != 520 or data.match_id.duplicated().any():
    raise SystemExit("Population ou IDs incorrects")

if data.split.value_counts().to_dict() != {
    "TRAIN": 183,
    "VALIDATION": 154,
    "TEST_FUTUR": 183,
}:
    raise SystemExit("Partitions incorrectes")

data = data[
    data.split.isin(["TRAIN", "VALIDATION"])
].copy()

data["y"] = pd.to_numeric(
    data["goal_after_87"], errors="raise"
)

if not data.y.isin([0, 1]).all():
    raise SystemExit("Labels non binaires")

for col in B6:
    data[col] = pd.to_numeric(data[col], errors="coerce")

if data[B6].isna().any().any():
    raise SystemExit("Variables B manquantes")

data["total_shots_last5"] = (
    data.shots_home_last5 + data.shots_away_last5
)
data["total_sot_last5"] = (
    data.sot_home_last5 + data.sot_away_last5
)
data["total_corners_last5"] = (
    data.corners_home_last5 + data.corners_away_last5
)

FEATURES = B6 + [
    "total_shots_last5",
    "total_sot_last5",
    "total_corners_last5",
]

print("=" * 72)
print("ÉTAPE 174 — STABILITÉ DES CARACTÉRISTIQUES B29")
print("=" * 72)

for split in ("TRAIN", "VALIDATION"):
    print(f"\n=== {split} ===")

    for source in ("ALL_320", "BATCH02_11"):
        g = data[
            (data.split == split)
            & (data.source == source)
        ]

        positives = g[g.y == 1]
        negatives = g[g.y == 0]

        print(
            f"\nSOURCE : {source} | "
            f"n={len(g)} | "
            f"positifs={len(positives)} | "
            f"taux={g.y.mean():.1%}"
        )

        print(
            f"{'VARIABLE':27s} "
            f"{'SANS BUT':>9s} "
            f"{'AVEC BUT':>9s} "
            f"{'ÉCART':>9s}"
        )

        for col in FEATURES:
            neg = negatives[col].mean()
            pos = positives[col].mean()
            delta = pos - neg

            print(
                f"{col:27s} "
                f"{neg:9.3f} "
                f"{pos:9.3f} "
                f"{delta:+9.3f}"
            )

print("\n=== VALEUR AJOUTÉE B29 PAR SOURCE — VALIDATION ===")

pred = pd.read_csv(PRED, dtype={"match_id": str})

if len(pred) != 154 or pred.match_id.duplicated().any():
    raise SystemExit("Prédictions de validation incorrectes")

valid_sources = data[
    data.split == "VALIDATION"
][["match_id", "source"]]

pred = pred.merge(
    valid_sources,
    on="match_id",
    validate="one_to_one"
)

if pred.source.isna().any():
    raise SystemExit("Origine manquante")

for source, g in pred.groupby("source"):
    y = pd.to_numeric(
        g["goal_after_87"], errors="raise"
    )

    print(f"\n{source} | n={len(g)}")

    for model in ("V4-A23-LR", "V4-B29-LR"):
        p = pd.to_numeric(g[model], errors="raise")

        if y.nunique() < 2:
            print(model, "| AUC non calculable")
            continue

        print(
            model,
            "| AUC =", round(roc_auc_score(y, p), 4),
            "| AP =", round(average_precision_score(y, p), 4),
        )

print("\n=== BILAN ===")
print("TRAIN et VALIDATION uniquement.")
print("TEST_FUTUR exclu de l'analyse.")
print("Aucun réentraînement.")
print("Aucune modification des données.")
print("Scanner V3-C inchangé.")
