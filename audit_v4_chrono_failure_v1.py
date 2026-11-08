from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("outputs/pressure_historical/pressure_v2_dataset_join")
PRED = Path(
    "outputs/v4_research/"
    "V4_PRESSURE_ABLATION_TRAIN_OOF_V1.csv"
)

print("=" * 72)
print("ÉTAPE 185 — DIAGNOSTIC DES ÉCHECS CHRONOLOGIQUES V4")
print("=" * 72)

oof = pd.read_csv(PRED, dtype={"match_id": str})

required = {
    "match_id", "source", "y", "fold",
    "A23", "A23_AWAY3", "A23_TOTAL3", "B29",
}
if not required.issubset(oof.columns):
    raise SystemExit(
        f"Colonnes manquantes : {sorted(required-set(oof.columns))}"
    )

if len(oof) != 121 or oof.match_id.duplicated().any():
    raise SystemExit("Population OOF inattendue")

all_data = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)

batch_files = sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv"))
if len(batch_files) != 10:
    raise SystemExit("10 fichiers B attendus")

batches = pd.concat(
    [
        pd.read_csv(path, dtype={"match_id": str})
        for path in batch_files
    ],
    ignore_index=True,
)

features = [
    "match_id", "goal_after_87",
    "score_87_home", "score_87_away",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

raw = pd.concat(
    [
        all_data[features],
        batches[features],
    ],
    ignore_index=True,
)

if raw.match_id.duplicated().any():
    raise SystemExit("IDs dupliqués dans les sources")

df = oof.merge(
    raw,
    on="match_id",
    validate="one_to_one"
)

if len(df) != 121:
    raise SystemExit("Jointure incorrecte")

df["y"] = pd.to_numeric(df["y"], errors="raise")
df["goal_after_87"] = pd.to_numeric(
    df["goal_after_87"], errors="raise"
)

if not (df["y"] == df["goal_after_87"]).all():
    raise SystemExit("Labels différents entre OOF et dataset")

df["score_87"] = (
    df["score_87_home"].astype(int).astype(str)
    + "-"
    + df["score_87_away"].astype(int).astype(str)
)

df["score_total"] = (
    df["score_87_home"] + df["score_87_away"]
)

df["score_situation"] = "AUTRE"
df.loc[
    df["score_87_home"] == df["score_87_away"],
    "score_situation",
] = "EGALITE"
df.loc[
    df["score_87_home"] > df["score_87_away"],
    "score_situation",
] = "DOMICILE_MENE"
df.loc[
    df["score_87_home"] < df["score_87_away"],
    "score_situation",
] = "EXTERIEUR_MENE"

print("\n=== PERFORMANCES CHRONOLOGIQUES ===")

for fold, g in df.groupby("fold"):
    positives = int(g.y.sum())
    auc = roc_auc_score(g.y, g.B29)
    ap = average_precision_score(g.y, g.B29)

    print(
        f"BLOC {fold} | n={len(g)} | "
        f"positifs={positives} | "
        f"AUC={auc:.4f} | AP={ap:.4f}"
    )

print("\n=== DÉCOMPOSITION PAR SOURCE ===")

for fold, fold_data in df.groupby("fold"):
    print(f"\nBLOC {fold}")

    for source, g in fold_data.groupby("source"):
        print(
            source,
            f"n={len(g)}",
            f"positifs={int(g.y.sum())}",
            f"prob_moyenne={g.B29.mean():.3f}",
        )

        if g.y.nunique() == 2:
            print(
                "  AUC =",
                round(roc_auc_score(g.y, g.B29), 4)
            )

print("\n=== BLOC 2 : SCORES À 87' ===")

block = df[df.fold == 2].copy()

print(
    block.groupby(
        ["score_situation", "y"]
    ).size().unstack(fill_value=0).to_string()
)

print("\n=== BLOC 2 : PROBABILITÉS SELON LABEL ===")

print(
    block.groupby("y")[
        ["A23", "A23_AWAY3", "A23_TOTAL3", "B29"]
    ].agg(["count", "mean", "median"]).round(3).to_string()
)

print("\n=== BLOC 2 : 25 MATCHS CLASSÉS PAR B29 ===")

columns = [
    "match_id", "source", "score_87",
    "y", "B29",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
]

print(
    block.sort_values("B29", ascending=False)[columns]
    .to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

print("\n=== BLOC 2 : TOP 5 ET TOP 10 ===")

ranked = block.sort_values("B29", ascending=False)

for k in (5, 10):
    selected = ranked.head(k)
    wins = int(selected.y.sum())

    print(
        f"TOP {k} : {wins}/{k} = {wins/k:.1%}"
    )

print("\n=== BILAN ===")
print("121 prédictions OOF historiques uniquement.")
print("Aucun réentraînement.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Aucune modification des données.")
print("Scanner V3-C inchangé.")
