from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

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

plan = pd.read_csv(
    ROOT / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

base = pd.read_csv(
    ROOT / "ALL_320_B_SCORE_CORRECTED_V3_RESEARCH.csv",
    dtype={"match_id": str},
)

base = base.merge(
    plan, on="match_id", validate="one_to_one"
)
base = base[base["split"] == "TRAIN"].copy()
base["source"] = "ALL_320"

parts = []
paths = sorted(ROOT.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(paths) != 10:
    raise SystemExit("10 lots B attendus")

for path in paths:
    p = pd.read_csv(path, dtype={"match_id": str})
    p = p[p["research_split"] == "TRAIN"].copy()
    p["source"] = "BATCH02_11"
    parts.append(p)

columns = [
    "match_id", "kickoff_utc", "goal_after_87", "source"
] + A23 + B6

train = pd.concat(
    [base[columns]] + [p[columns] for p in parts],
    ignore_index=True,
)

if len(train) != 183 or train.match_id.duplicated().any():
    raise SystemExit("Population TRAIN incorrecte")

train["kickoff"] = pd.to_datetime(
    train["kickoff_utc"], utc=True, errors="raise"
)

train["y"] = pd.to_numeric(
    train["goal_after_87"], errors="raise"
).astype(int)

if not train.y.isin([0, 1]).all():
    raise SystemExit("Labels incorrects")

for col in A23 + B6:
    train[col] = pd.to_numeric(train[col], errors="coerce")

train["score_total_87"] = (
    train.score_87_home + train.score_87_away
)
train["score_diff_abs_87"] = (
    train.score_87_home - train.score_87_away
).abs()

train["shots_total_last5"] = (
    train.shots_home_last5 + train.shots_away_last5
)
train["sot_total_last5"] = (
    train.sot_home_last5 + train.sot_away_last5
)
train["corners_total_last5"] = (
    train.corners_home_last5 + train.corners_away_last5
)

SIMPLE6 = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]

SIMPLE8 = [
    "score_87_home",
    "score_87_away",
    "is_draw_87",
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_total_last5",
]

CONFIGS = {
    "B29_C1": (A23 + B6, 1.0),
    "SIMPLE6_C01": (SIMPLE6, 0.1),
    "SIMPLE8_C01": (SIMPLE8, 0.1),
}

train = train.sort_values(
    ["kickoff", "match_id"]
).reset_index(drop=True)

n = len(train)

fractions = (0.35, 0.50, 0.65, 0.80)
boundaries = sorted(set(
    train.kickoff.iloc[min(int(n*f), n-1)]
    for f in fractions
))

if len(boundaries) != 4:
    raise SystemExit("Frontières temporelles inattendues")

def make_model(c):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=c, max_iter=3000, random_state=42
        )),
    ])

predictions = {
    name: np.full(n, np.nan)
    for name in CONFIGS
}
folds = np.zeros(n, dtype=int)

print("=" * 72)
print("ÉTAPE 186 — ARCHITECTURES V4 SIMPLIFIÉES")
print("=" * 72)

for i, start in enumerate(boundaries):
    end = boundaries[i+1] if i+1 < 4 else None

    mask_test = train.kickoff >= start
    if end is not None:
        mask_test &= train.kickoff < end

    mask_fit = train.kickoff < (
        start - pd.Timedelta(hours=4)
    )

    y_fit = train.loc[mask_fit, "y"]
    y_test = train.loc[mask_test, "y"]

    if (
        mask_fit.sum() < 30
        or mask_test.sum() < 5
        or y_fit.nunique() != 2
    ):
        raise SystemExit(f"Bloc {i+1} invalide")

    folds[mask_test.to_numpy()] = i + 1

    print(
        f"\nBLOC {i+1} "
        f"| entraînement={mask_fit.sum()} "
        f"| évaluation={mask_test.sum()} "
        f"| positifs={int(y_test.sum())}"
    )

    for name, (features, c) in CONFIGS.items():
        model = make_model(c)
        model.fit(train.loc[mask_fit, features], y_fit)

        p_test = model.predict_proba(
            train.loc[mask_test, features]
        )[:, 1]

        predictions[name][mask_test.to_numpy()] = p_test

        if y_test.nunique() == 2:
            auc = roc_auc_score(y_test, p_test)
            ap = average_precision_score(y_test, p_test)
        else:
            auc, ap = float("nan"), float("nan")

        if i == 1:
            p_fit = model.predict_proba(
                train.loc[mask_fit, features]
            )[:, 1]

            fit_auc = roc_auc_score(y_fit, p_fit)

            print(
                f"  {name:14s} "
                f"TRAIN_AUC={fit_auc:.4f} "
                f"OOF_AUC={auc:.4f} "
                f"OOF_AP={ap:.4f}"
            )
        else:
            print(
                f"  {name:14s} "
                f"OOF_AUC={auc:.4f} "
                f"OOF_AP={ap:.4f}"
            )

common = folds > 0

if int(common.sum()) != 121:
    raise SystemExit("121 prédictions OOF attendues")

y = train.loc[common, "y"].to_numpy()

print("\n=== COMPARAISON OOF GLOBALE ===")
print("Matchs :", len(y))
print("Positifs :", int(y.sum()))
print("Taux de base :", f"{y.mean():.2%}")

for name, p_all in predictions.items():
    p = p_all[common]

    if not np.isfinite(p).all():
        raise SystemExit(f"Prédictions manquantes : {name}")

    print(
        f"{name:14s} "
        f"AUC={roc_auc_score(y, p):.4f} "
        f"AP={average_precision_score(y, p):.4f} "
        f"Brier={brier_score_loss(y, p):.4f}"
    )

print("\n=== RÉUSSITE TOP 10 DANS LE BLOC 2 ===")

mask2 = folds == 2
block_y = train.loc[mask2, "y"].to_numpy()

for name, p_all in predictions.items():
    block_p = p_all[mask2]
    selected = np.argsort(-block_p)[:10]
    wins = int(block_y[selected].sum())

    print(f"{name:14s} {wins}/10")

print("\n=== ROC-AUC PAR SOURCE ===")

for source in ("ALL_320", "BATCH02_11"):
    mask = common & (train.source == source).to_numpy()
    yy = train.loc[mask, "y"].to_numpy()

    print(f"\n{source} | n={len(yy)}")

    for name, p_all in predictions.items():
        p = p_all[mask]
        print(
            f"  {name:14s} "
            f"AUC={roc_auc_score(yy, p):.4f} "
            f"AP={average_precision_score(yy, p):.4f}"
        )

OUT.mkdir(parents=True, exist_ok=True)

result = train.loc[
    common,
    ["match_id", "kickoff_utc", "source", "y"]
].copy()

result["fold"] = folds[common]

for name, p_all in predictions.items():
    result[name] = p_all[common]

destination = OUT / "V4_SIMPLE_CHRONO_OOF_V1.csv"
result.to_csv(destination, index=False)

print("\n=== BILAN ===")
print("Fichier :", destination)
print("183 TRAIN uniquement.")
print("121 prédictions OOF.")
print("Embargo chronologique : 4 heures.")
print("VALIDATION et TEST_FUTUR non évalués.")
print("Scanner V3-C inchangé.")
print("Statut : RESEARCH_ONLY")
