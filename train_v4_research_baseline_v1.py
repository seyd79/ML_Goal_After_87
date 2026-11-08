from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(
    "outputs/pressure_historical/pressure_v2_dataset_join"
)

OUT = Path("outputs/v4_research")

A23 = [
    "score_87_home",
    "score_87_away",
    "goal_diff_87",
    "is_draw_87",
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_shots_off_target_87",
    "away_shots_off_target_87",
    "home_shots_blocked_87",
    "away_shots_blocked_87",
    "home_corners_87",
    "away_corners_87",
    "home_fouls_87",
    "away_fouls_87",
    "home_yellow_cards_87",
    "away_yellow_cards_87",
    "home_red_cards_87",
    "away_red_cards_87",
    "home_offsides_87",
    "away_offsides_87",
    "goal_diff_signed_87",
]

B6 = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

B29 = A23 + B6

assert len(A23) == 23
assert len(B29) == 29
assert len(set(B29)) == 29

print("=" * 75)
print("ÉTAPE 167 — PREMIER ENTRAÎNEMENT EXPÉRIMENTAL V4")
print("=" * 75)

consolidated = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str}
)

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"]
)

if plan.match_id.duplicated().any():
    raise SystemExit("ERREUR : doublons dans le plan")

consolidated = consolidated.merge(
    plan,
    on="match_id",
    how="left",
    validate="one_to_one"
)

paths = sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(paths) != 10:
    raise SystemExit("ERREUR : 10 lots B attendus")

parts = []

for path in paths:
    part = pd.read_csv(
        path,
        dtype={"match_id": str}
    )
    part["split"] = part["research_split"]
    parts.append(part)

batches = pd.concat(parts, ignore_index=True)

# Garder uniquement les colonnes autorisées
# pour la recherche et l'identification.
columns = [
    "match_id",
    "kickoff_utc",
    "goal_after_87",
    "split",
] + B29

for name, frame in [
    ("consolidated", consolidated),
    ("batches", batches)
]:
    missing = set(columns) - set(frame.columns)

    if missing:
        raise SystemExit(
            f"Colonnes absentes de {name}: {sorted(missing)}"
        )

data = pd.concat(
    [
        consolidated[columns],
        batches[columns]
    ],
    ignore_index=True
)

data["split"] = (
    data["split"]
    .astype(str)
    .str.strip()
    .str.upper()
)

if data.match_id.isna().any():
    raise SystemExit("ERREUR : match_id manquant")

if data.match_id.duplicated().any():
    raise SystemExit("ERREUR : match_id dupliqué")

if len(data) != 520:
    raise SystemExit(
        f"ERREUR : 520 matchs attendus, {len(data)} trouvés"
    )

expected_splits = {
    "TRAIN": 183,
    "VALIDATION": 154,
    "TEST_FUTUR": 183,
}

actual_splits = data.split.value_counts().to_dict()

if actual_splits != expected_splits:
    raise SystemExit(
        f"Partitions inattendues : {actual_splits}"
    )

# Le jeu TEST_FUTUR est retiré avant
# toute analyse des labels et caractéristiques.
data = data[
    data["split"].isin(["TRAIN", "VALIDATION"])
].copy()

train = data[data.split == "TRAIN"].copy()
valid = data[data.split == "VALIDATION"].copy()

for frame in (train, valid):
    frame["date"] = pd.to_datetime(
        frame["kickoff_utc"],
        utc=True,
        errors="coerce"
    )

    if frame["date"].isna().any():
        raise SystemExit("Dates invalides")

    frame["goal_after_87"] = pd.to_numeric(
        frame["goal_after_87"],
        errors="coerce"
    )

    if not frame["goal_after_87"].isin([0, 1]).all():
        raise SystemExit("Labels non binaires")

if not train.date.max() < valid.date.min():
    raise SystemExit(
        "ERREUR : fuite entre TRAIN et VALIDATION"
    )

for col in B29:
    train[col] = pd.to_numeric(
        train[col], errors="coerce"
    )
    valid[col] = pd.to_numeric(
        valid[col], errors="coerce"
    )

    if train[col].notna().sum() == 0:
        raise SystemExit(
            f"Variable entièrement manquante dans TRAIN: {col}"
        )

print("\n=== POPULATION ===")
print("TRAIN :", len(train))
print("VALIDATION :", len(valid))
print("Positifs TRAIN :", int(train.goal_after_87.sum()))
print("Positifs VALIDATION :", int(valid.goal_after_87.sum()))

baseline = valid.goal_after_87.mean()

print("Fréquence positive VALIDATION :",
      f"{baseline:.2%}")

print("\n=== DONNÉES MANQUANTES ===")

for col in B29:
    miss_train = train[col].isna().mean()
    miss_valid = valid[col].isna().mean()

    if miss_train > 0 or miss_valid > 0:
        print(
            col,
            f"TRAIN={miss_train:.1%}",
            f"VALIDATION={miss_valid:.1%}"
        )

def model_factory(algorithm):
    if algorithm == "LR":
        return Pipeline([
            ("imputer", SimpleImputer(
                strategy="median"
            )),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=3000,
                random_state=42
            )),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        )),
    ])

def topk_metrics(y, p, k):
    order = np.argsort(-p, kind="stable")
    selected = y[order[:k]]
    return float(selected.mean()), int(selected.sum())

results = []
predictions = valid[
    ["match_id", "kickoff_utc", "goal_after_87"]
].copy()

configs = [
    ("V4-A23-LR", "LR", A23),
    ("V4-B29-LR", "LR", B29),
    ("V4-A23-RF", "RF", A23),
    ("V4-B29-RF", "RF", B29),
]

y_train = train.goal_after_87.astype(int).to_numpy()
y_valid = valid.goal_after_87.astype(int).to_numpy()

print("\n=== COMPARAISON DES MODÈLES ===")

for name, algorithm, features in configs:
    model = model_factory(algorithm)

    model.fit(
        train[features],
        y_train
    )

    proba = model.predict_proba(
        valid[features]
    )[:, 1]

    if not np.isfinite(proba).all():
        raise SystemExit(
            f"Probabilités invalides : {name}"
        )

    result = {
        "model": name,
        "features": len(features),
        "train_n": len(train),
        "validation_n": len(valid),
        "baseline_rate": baseline,
        "roc_auc": roc_auc_score(
            y_valid, proba
        ),
        "average_precision": average_precision_score(
            y_valid, proba
        ),
        "brier": brier_score_loss(
            y_valid, proba
        ),
    }

    for k in (10, 20, 30, 50):
        precision, wins = topk_metrics(
            y_valid, proba, k
        )
        result[f"precision_at_{k}"] = precision
        result[f"wins_at_{k}"] = wins

    results.append(result)
    predictions[name] = proba

    print(f"\n{name}")
    print(f"  ROC-AUC           : {result['roc_auc']:.4f}")
    print(f"  Average Precision : {result['average_precision']:.4f}")
    print(f"  Brier             : {result['brier']:.4f}")

    for k in (10, 20, 30, 50):
        print(
            f"  TOP {k:2d} : "
            f"{result[f'wins_at_{k}']:2d}/{k} "
            f"({result[f'precision_at_{k}']:.1%})"
        )

table = pd.DataFrame(results)

print("\n=== RÉSUMÉ COMPARATIF ===")

display_cols = [
    "model",
    "roc_auc",
    "average_precision",
    "precision_at_10",
    "precision_at_20",
    "precision_at_30",
    "precision_at_50",
]

print(
    table[display_cols]
    .round(4)
    .to_string(index=False)
)

print("\n=== GAIN B29 SUR A23 ===")

for algorithm in ("LR", "RF"):
    ra = next(
        r for r in results
        if r["model"] == f"V4-A23-{algorithm}"
    )
    rb = next(
        r for r in results
        if r["model"] == f"V4-B29-{algorithm}"
    )

    print(
        algorithm,
        "| Delta AUC :",
        f"{rb['roc_auc'] - ra['roc_auc']:+.4f}",
        "| Delta AP :",
        f"{rb['average_precision'] - ra['average_precision']:+.4f}"
    )

OUT.mkdir(parents=True, exist_ok=True)

metrics_path = OUT / "V4_BASELINE_VALIDATION_METRICS_V1.csv"
predictions_path = OUT / "V4_BASELINE_VALIDATION_PREDICTIONS_V1.csv"

table.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False)

print("\n=== FICHIERS CRÉÉS ===")
print(metrics_path)
print(predictions_path)

print("\n=== BILAN ===")
print("Entraînement réalisé uniquement sur TRAIN.")
print("Évaluation uniquement sur VALIDATION.")
print("TEST_FUTUR exclu.")
print("Aucun modèle de production modifié.")
print("Statut : RESEARCH_ONLY.")
print("ÉTAPE 167 TERMINÉE.")
