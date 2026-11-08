from pathlib import Path
import csv

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

JOIN = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)
OUT = Path("outputs/v4_research")

TRAIN = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
DEST = OUT / "V4_VALIDATION154_COMPARISON_V1.csv"

print("=" * 72)
print("ÉTAPE 205 — VALIDATION CHRONOLOGIQUE V4")
print("=" * 72)

if DEST.exists():
    raise SystemExit("Sortie existante : remplacement refusé")

train = pd.read_csv(TRAIN, dtype={"match_id": str})

old = train[
    ~train.research_origin.isin(
        ["V4_CACHED12", "V4_FETCH83"]
    )
].copy()

if len(train) != 278 or len(old) != 183:
    raise SystemExit("TRAIN inattendu")

if train.match_id.duplicated().any():
    raise SystemExit("Doublons TRAIN")

if int(train.goal_after_87.sum()) != 72:
    raise SystemExit("Labels TRAIN inattendus")

split = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype=str,
    usecols=["match_id", "split"],
)

val_ids = set(
    split.loc[split["split"] == "VALIDATION", "match_id"]
)

if len(val_ids) != 44:
    raise SystemExit("44 VALIDATION ALL_320 attendus")

rows = []

def append_selected(path, allowed_ids=None):
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if allowed_ids is not None:
                if row["match_id"] not in allowed_ids:
                    continue
            elif row.get("research_split") != "VALIDATION":
                continue

            rows.append(row)

append_selected(
    JOIN / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    allowed_ids=val_ids,
)

for n in range(2, 12):
    append_selected(
        JOIN / f"BATCH{n:02d}_B_RESEARCH_V1.csv"
    )

valid = pd.DataFrame(rows)

if len(valid) != 154 or valid.match_id.duplicated().any():
    raise SystemExit("154 matchs VALIDATION uniques attendus")

if int(pd.to_numeric(valid.goal_after_87).sum()) != 39:
    raise SystemExit("39 positifs VALIDATION attendus")

if set(valid.match_id) & set(train.match_id):
    raise SystemExit("Chevauchement TRAIN / VALIDATION")

train["kickoff"] = pd.to_datetime(
    train.kickoff_utc, utc=True, errors="raise"
)
valid["kickoff"] = pd.to_datetime(
    valid.kickoff_utc, utc=True, errors="raise"
)

if train.kickoff.max() >= valid.kickoff.min():
    raise SystemExit("Chevauchement temporel")

valid = valid.sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

def features(df):
    def n(col):
        return pd.to_numeric(df[col], errors="raise")

    return pd.DataFrame({
        "score_total_87":
            n("score_87_home") + n("score_87_away"),
        "score_diff_abs_87":
            (n("score_87_home") - n("score_87_away")).abs(),
        "is_draw_87": n("is_draw_87"),
        "shots_total_last5":
            n("shots_home_last5") + n("shots_away_last5"),
        "sot_total_last5":
            n("sot_home_last5") + n("sot_away_last5"),
        "corners_total_last5":
            n("corners_home_last5") + n("corners_away_last5"),
    })

def predict(training):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        ),
    )

    model.fit(
        features(training),
        training.goal_after_87.astype(int),
    )
    return model.predict_proba(features(valid))[:, 1]

valid["p_old183"] = predict(old)
valid["p_new278"] = predict(train)

y = valid.goal_after_87.astype(int)

print("\n=== POPULATIONS ===")
print("TRAIN ancien :", len(old))
print("TRAIN enrichi :", len(train))
print("VALIDATION :", len(valid))
print("Positifs VALIDATION :", int(y.sum()))

print("\n=== PERFORMANCES GLOBALES ===")

for col in ["p_old183", "p_new278"]:
    p = valid[col]

    print(
        f"{col:12s} | "
        f"AUC={roc_auc_score(y, p):.4f} | "
        f"AP={average_precision_score(y, p):.4f} | "
        f"Brier={brier_score_loss(y, p):.4f}"
    )

print("\n=== MEILLEURES PRÉDICTIONS ===")

for k in [10, 20, 30]:
    print(f"\nTOP {k}")

    for col in ["p_old183", "p_new278"]:
        selected = valid.nlargest(k, col)
        wins = int(selected.goal_after_87.astype(int).sum())

        print(
            f"  {col:12s} | "
            f"{wins}/{k} = {wins/k:.1%}"
        )

print("\n=== SEUIL FIXÉ À 0.30 ===")

for col in ["p_old183", "p_new278"]:
    selected = valid[valid[col] >= 0.30]
    n = len(selected)
    wins = int(selected.goal_after_87.astype(int).sum())

    if n:
        print(
            f"{col:12s} | "
            f"{wins}/{n} = {wins/n:.1%} "
            f"| couverture={n/len(valid):.1%}"
        )
    else:
        print(col, "| aucune sélection")

print("\n=== DEUX MOITIÉS CHRONOLOGIQUES ===")

for name, part in [
    ("PREMIÈRE", valid.iloc[:77]),
    ("SECONDE", valid.iloc[77:]),
]:
    yy = part.goal_after_87.astype(int)

    print(
        f"\n{name} | matchs={len(part)} "
        f"| positifs={int(yy.sum())}"
    )

    for col in ["p_old183", "p_new278"]:
        top = part.nlargest(10, col)
        wins = int(top.goal_after_87.astype(int).sum())

        print(
            f"  {col:12s} | TOP10={wins}/10 "
            f"| AUC={roc_auc_score(yy, part[col]):.4f}"
        )

valid[
    [
        "match_id", "kickoff_utc",
        "goal_after_87", "p_old183", "p_new278"
    ]
].to_csv(DEST, index=False)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("Même validation pour les deux modèles.")
print("Même architecture SIMPLE6, C=0.1.")
print("Aucun réglage de paramètres.")
print("TEST_FUTUR non évalué.")
print("Scanner V3-C inchangé.")
print("VALIDATION exploratoire, non indépendante.")
