from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("outputs/pressure_historical/pressure_v2_dataset_join")
OOF = Path("outputs/v4_research/V4_CHRONO_OOF_PREDICTIONS_V1.csv")

print("=" * 72)
print("ÉTAPE 173 — AUDIT QUALITÉ DES CARACTÉRISTIQUES V4")
print("=" * 72)

oof = pd.read_csv(OOF, dtype={"match_id": str})

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

train_ids = set(plan.loc[plan.split == "TRAIN", "match_id"])

base = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)
base = base[base.match_id.isin(train_ids)].copy()
base["source"] = "ALL_320"

parts = []

for path in sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv")):
    p = pd.read_csv(path, dtype={"match_id": str})
    p = p[p.research_split == "TRAIN"].copy()
    p["source"] = "BATCH02_11"
    parts.append(p)

batches = pd.concat(parts, ignore_index=True)
all_train = pd.concat([base, batches], ignore_index=True)

if len(all_train) != 183 or all_train.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

columns = [
    "match_id", "source",
    "goal_after_87",
    "score_87_home", "score_87_away",
    "goal_diff_87", "goal_diff_signed_87",
    "is_draw_87",
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87", "away_shots_on_target_87",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

df = oof.merge(
    all_train[columns],
    on="match_id",
    validate="one_to_one",
    how="left",
)

if len(df) != 121 or df.source.isna().any():
    raise SystemExit("Jointure OOF incorrecte")

numeric_cols = [c for c in columns if c not in ("match_id", "source")]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if not (
    df["y"].eq(df["goal_after_87"]).all()
):
    raise SystemExit("Divergences de labels OOF")

print("\n=== POPULATION ===")
print("Matchs OOF :", len(df))
print(df.source.value_counts().to_string())

print("\n=== CONTRÔLES DE COHÉRENCE ===")

checks = {
    "Score négatif": (
        (df.score_87_home < 0) |
        (df.score_87_away < 0)
    ),
    "Différence signée incorrecte": (
        df.goal_diff_signed_87 !=
        df.score_87_home - df.score_87_away
    ),
    "Égalité incohérente": (
        df.is_draw_87 !=
        (df.score_87_home == df.score_87_away).astype(int)
    ),
    "Différence absolue incohérente": (
        df.goal_diff_87 !=
        (df.score_87_home - df.score_87_away).abs()
    ),
    "Tirs cadrés domicile > tirs": (
        df.home_shots_on_target_87 > df.home_shots_87
    ),
    "Tirs cadrés extérieur > tirs": (
        df.away_shots_on_target_87 > df.away_shots_87
    ),
    "Pression tirs domicile > total": (
        df.shots_home_last5 > df.home_shots_87
    ),
    "Pression tirs extérieur > total": (
        df.shots_away_last5 > df.away_shots_87
    ),
    "Pression cadrés domicile > tirs récents": (
        df.sot_home_last5 > df.shots_home_last5
    ),
    "Pression cadrés extérieur > tirs récents": (
        df.sot_away_last5 > df.shots_away_last5
    ),
}

for name, mask in checks.items():
    print(f"\n{name} : {int(mask.sum())}")
    for source, group in df[mask].groupby("source"):
        print(f"  {source}: {len(group)}")

print("\n=== MOYENNES DES CARACTÉRISTIQUES PAR SOURCE ===")

features = [
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87", "away_shots_on_target_87",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

print(
    df.groupby("source")[features]
    .mean()
    .round(2)
    .T
    .to_string()
)

print("\n=== PRESSION SELON LE LABEL ===")

for source, group in df.groupby("source"):
    print(f"\nSOURCE : {source}")
    print(
        group.groupby("y")[
            [
                "shots_home_last5",
                "shots_away_last5",
                "sot_home_last5",
                "sot_away_last5",
                "corners_home_last5",
                "corners_away_last5",
            ]
        ].mean().round(2).to_string()
    )

print("\n=== SEUIL OOF EXACT ===")

p = pd.to_numeric(oof["B29_C1"], errors="raise")
threshold = float(np.quantile(p, 0.85))

for label, t in [
    ("Seuil exact", threshold),
    ("Seuil arrondi", 0.5849),
]:
    selected = p >= t
    wins = int(oof.loc[selected, "y"].sum())

    print(
        label,
        f"valeur={t:.12f}",
        f"sélections={int(selected.sum())}",
        f"réussites={wins}"
    )

print("\n=== BILAN ===")
print("Audit descriptif, sans correction automatique.")
print("Une incohérence doit être vérifiée selon les conventions des données.")
print("Aucun réentraînement.")
print("TEST_FUTUR non évalué.")
print("Scanner LIVE inchangé.")
