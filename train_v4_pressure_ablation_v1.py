from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("outputs/pressure_historical/pressure_v2_dataset_join")
OUT = Path("outputs/v4_research")

A23 = [
    "score_87_home", "score_87_away",
    "goal_diff_87", "is_draw_87",
    "home_shots_87", "away_shots_87",
    "home_shots_on_target_87", "away_shots_on_target_87",
    "home_shots_off_target_87", "away_shots_off_target_87",
    "home_shots_blocked_87", "away_shots_blocked_87",
    "home_corners_87", "away_corners_87",
    "home_fouls_87", "away_fouls_87",
    "home_yellow_cards_87", "away_yellow_cards_87",
    "home_red_cards_87", "away_red_cards_87",
    "home_offsides_87", "away_offsides_87",
    "goal_diff_signed_87",
]

B6 = [
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
]

AWAY3 = [
    "shots_away_last5",
    "sot_away_last5",
    "corners_away_last5",
]

TOTAL3 = [
    "total_shots_last5",
    "total_sot_last5",
    "total_corners_last5",
]

print("=== ÉTAPE 175 : ABLATION DES VARIABLES DE PRESSION ===")

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

if plan.match_id.duplicated().any():
    raise SystemExit("Doublons dans le plan")

base = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)

base = base.merge(plan, on="match_id", validate="one_to_one")
base = base[base["split"] == "TRAIN"].copy()
base["source"] = "ALL_320"

parts = []
paths = sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(paths) != 10:
    raise SystemExit("10 lots B attendus")

for path in paths:
    part = pd.read_csv(path, dtype={"match_id": str})
    part = part[part["research_split"] == "TRAIN"].copy()
    part["source"] = "BATCH02_11"
    parts.append(part)

columns = [
    "match_id", "kickoff_utc", "goal_after_87",
    "source"
] + A23 + B6

train = pd.concat(
    [base[columns]] + [part[columns] for part in parts],
    ignore_index=True,
)

if len(train) != 183 or train.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

train["kickoff"] = pd.to_datetime(
    train["kickoff_utc"], utc=True, errors="raise"
)
train["y"] = pd.to_numeric(
    train["goal_after_87"], errors="raise"
)

if not train["y"].isin([0, 1]).all():
    raise SystemExit("Labels invalides")

for col in A23 + B6:
    train[col] = pd.to_numeric(train[col], errors="coerce")

train["total_shots_last5"] = (
    train["shots_home_last5"] + train["shots_away_last5"]
)
train["total_sot_last5"] = (
    train["sot_home_last5"] + train["sot_away_last5"]
)
train["total_corners_last5"] = (
    train["corners_home_last5"] + train["corners_away_last5"]
)

train = train.sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

configs = {
    "A23": A23,
    "A23_AWAY3": A23 + AWAY3,
    "A23_TOTAL3": A23 + TOTAL3,
    "B29": A23 + B6,
}

def model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, max_iter=3000, random_state=42
        )),
    ])

n = len(train)
fractions = (0.35, 0.50, 0.65, 0.80)

boundaries = sorted(set(
    train.kickoff.iloc[min(int(n*f), n-1)]
    for f in fractions
))

if len(boundaries) != 4:
    raise SystemExit("Frontières temporelles insuffisantes")

oof = {name: np.full(n, np.nan) for name in configs}
fold_ids = np.zeros(n, dtype=int)

print("\n=== VALIDATION CHRONOLOGIQUE ===")

for i, start in enumerate(boundaries):
    end = boundaries[i+1] if i+1 < len(boundaries) else None

    mask_test = train.kickoff >= start
    if end is not None:
        mask_test &= train.kickoff < end

    cutoff = start - pd.Timedelta(hours=4)
    mask_fit = train.kickoff < cutoff

    y_fit = train.loc[mask_fit, "y"].astype(int)

    if (
        mask_fit.sum() < 30
        or mask_test.sum() < 5
        or y_fit.nunique() != 2
        or y_fit.sum() < 5
        or (y_fit == 0).sum() < 5
    ):
        raise SystemExit(f"Bloc {i+1} non admissible")

    fold_ids[mask_test.to_numpy()] = i+1

    print(
        f"BLOC {i+1} | fit={mask_fit.sum()} "
        f"| test={mask_test.sum()} "
        f"| début={start}"
    )

    for name, features in configs.items():
        m = model()
        m.fit(train.loc[mask_fit, features], y_fit)

        oof[name][mask_test.to_numpy()] = m.predict_proba(
            train.loc[mask_test, features]
        )[:, 1]

common = fold_ids > 0

if common.sum() != 121:
    raise SystemExit(
        f"Nombre OOF inattendu : {common.sum()}"
    )

y = train.loc[common, "y"].astype(int).to_numpy()

print("\n=== PERFORMANCE OOF GLOBALE ===")
print("Matchs :", len(y))
print("Positifs :", int(y.sum()))
print("Taux de base :", f"{y.mean():.1%}")

for name, values in oof.items():
    p = values[common]

    if not np.isfinite(p).all():
        raise SystemExit(f"Prédictions invalides : {name}")

    print(
        f"{name:12s} | "
        f"AUC={roc_auc_score(y, p):.4f} | "
        f"AP={average_precision_score(y, p):.4f}"
    )

print("\n=== PERFORMANCE PAR BLOC ===")

for i in range(1, 5):
    mask = fold_ids == i
    yy = train.loc[mask, "y"].astype(int).to_numpy()

    print(
        f"\nBLOC {i} | n={len(yy)} "
        f"| taux positif={yy.mean():.1%}"
    )

    for name, values in oof.items():
        pp = values[mask]

        if len(np.unique(yy)) == 2:
            auc = roc_auc_score(yy, pp)
            ap = average_precision_score(yy, pp)
            print(
                f"  {name:12s} | "
                f"AUC={auc:.4f} | AP={ap:.4f}"
            )

print("\n=== PERFORMANCE PAR SOURCE ===")

for source in ("ALL_320", "BATCH02_11"):
    mask = common & (train.source == source).to_numpy()
    yy = train.loc[mask, "y"].astype(int).to_numpy()

    print(
        f"\n{source} | n={len(yy)} "
        f"| taux positif={yy.mean():.1%}"
    )

    if len(np.unique(yy)) < 2:
        continue

    for name, values in oof.items():
        pp = values[mask]
        print(
            f"  {name:12s} | "
            f"AUC={roc_auc_score(yy, pp):.4f} | "
            f"AP={average_precision_score(yy, pp):.4f}"
        )

OUT.mkdir(parents=True, exist_ok=True)

result = train.loc[
    common, [
        "match_id", "kickoff_utc", "source", "y"
    ]
].copy()

result["fold"] = fold_ids[common]

for name, values in oof.items():
    result[name] = values[common]

destination = OUT / "V4_PRESSURE_ABLATION_TRAIN_OOF_V1.csv"
result.to_csv(destination, index=False)

print("\n=== BILAN ===")
print("Fichier créé :", destination)
print("Comparaison réalisée sur TRAIN uniquement.")
print("Même algorithme et même régularisation.")
print("Embargo chronologique conservé : 4 heures.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Aucun changement du scanner LIVE.")
print("Statut : RESEARCH_ONLY")
print("ÉTAPE 175 TERMINÉE")
