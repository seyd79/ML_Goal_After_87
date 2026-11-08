#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
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

ROOT = Path("outputs/v4_research")
TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
OOF = ROOT / "V4_SIMPLE_CHRONO_OOF_V1.csv"
DEST = ROOT / "V41_REDCARD_CHRONO_OOF_V1.csv"

EMBARGO = pd.Timedelta(hours=4)

if DEST.exists():
    raise SystemExit("Sortie déjà existante : aucun remplacement")

print("=" * 72)
print("ÉTAPE 223 — SIMPLE6 VS SIMPLE7 CARTONS ROUGES")
print("=" * 72)

train = pd.read_csv(TRAIN, dtype={"match_id": str})
oof = pd.read_csv(OOF, dtype={"match_id": str})

if len(train) != 278 or len(oof) != 121:
    raise SystemExit("Effectifs inattendus")

if train.match_id.duplicated().any():
    raise SystemExit("Doublons TRAIN")

if oof.match_id.duplicated().any():
    raise SystemExit("Doublons OOF")

required = [
    "match_id", "kickoff_utc", "goal_after_87",
    "score_87_home", "score_87_away",
    "is_draw_87",
    "shots_home_last5", "shots_away_last5",
    "sot_home_last5", "sot_away_last5",
    "corners_home_last5", "corners_away_last5",
    "home_red_cards_87", "away_red_cards_87",
]

if not set(required).issubset(train.columns):
    raise SystemExit("Colonnes TRAIN manquantes")

if not set([
    "match_id", "kickoff_utc", "fold", "y"
]).issubset(oof.columns):
    raise SystemExit("Colonnes OOF manquantes")

train["kickoff"] = pd.to_datetime(
    train.kickoff_utc, utc=True, errors="raise"
)
oof["kickoff"] = pd.to_datetime(
    oof.kickoff_utc, utc=True, errors="raise"
)

for col in required[2:]:
    train[col] = pd.to_numeric(train[col], errors="raise")

    if not np.isfinite(train[col]).all():
        raise SystemExit(f"Valeur non finie : {col}")

if not train.goal_after_87.isin([0, 1]).all():
    raise SystemExit("Labels invalides")

if int(train.goal_after_87.sum()) != 72:
    raise SystemExit("Nombre de positifs TRAIN inattendu")

if not (train.kickoff < pd.Timestamp(
    "2026-08-25T00:00:00Z"
)).all():
    raise SystemExit("Match après la limite TRAIN")

for col in ("home_red_cards_87", "away_red_cards_87"):
    if (train[col] < 0).any():
        raise SystemExit(f"Cartons rouges négatifs : {col}")

    if (train[col] % 1 != 0).any():
        raise SystemExit(f"Cartons rouges non entiers : {col}")

train["red_card_imbalance_87"] = (
    train.home_red_cards_87 != train.away_red_cards_87
).astype(int)

lookup = train.set_index("match_id")

if not set(oof.match_id).issubset(set(train.match_id)):
    raise SystemExit("Match OOF absent du TRAIN")

expected_labels = lookup.loc[
    oof.match_id, "goal_after_87"
].to_numpy(dtype=int)

if not np.array_equal(
    expected_labels, oof.y.to_numpy(dtype=int)
):
    raise SystemExit("Labels OOF incompatibles")

if oof.fold.nunique() != 4:
    raise SystemExit("Quatre folds attendus")

if int(oof.y.sum()) != 31:
    raise SystemExit("31 positifs OOF attendus")


def features(df, with_red):
    home = df.score_87_home
    away = df.score_87_away

    X = pd.DataFrame(index=df.index)

    X["score_total_87"] = home + away
    X["score_diff_abs_87"] = (home-away).abs()
    X["is_draw_87"] = df.is_draw_87

    X["shots_total_last5"] = (
        df.shots_home_last5 + df.shots_away_last5
    )
    X["sot_total_last5"] = (
        df.sot_home_last5 + df.sot_away_last5
    )
    X["corners_total_last5"] = (
        df.corners_home_last5 + df.corners_away_last5
    )

    if with_red:
        X["red_card_imbalance_87"] = (
            df.red_card_imbalance_87
        )

    return X


def predict(fit, test, with_red):
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
        features(fit, with_red),
        fit.goal_after_87.astype(int),
    )

    return model.predict_proba(
        features(test, with_red)
    )[:, 1]


expected_sizes = [58, 108, 188, 209]
parts = []

print("\n=== BLOCS CHRONOLOGIQUES ===")

for i, (fold, block) in enumerate(
    oof.groupby("fold", sort=True)
):
    block = block.copy()

    cutoff = block.kickoff.min() - EMBARGO

    fit = train[train.kickoff < cutoff].copy()

    if len(fit) != expected_sizes[i]:
        raise SystemExit(
            f"Fold {i+1} : effectif TRAIN inattendu"
        )

    if set(block.match_id) & set(fit.match_id):
        raise SystemExit("Fuite temporelle : ID commun")

    if fit.kickoff.max() >= cutoff:
        raise SystemExit("Fuite chronologique")

    if fit.goal_after_87.nunique() != 2:
        raise SystemExit("TRAIN sans deux classes")

    test = lookup.loc[block.match_id].reset_index()

    block["SIMPLE6"] = predict(
        fit, test, with_red=False
    )
    block["SIMPLE7_RED"] = predict(
        fit, test, with_red=True
    )
    block["red_card_imbalance_87"] = (
        test.red_card_imbalance_87.to_numpy()
    )

    parts.append(block)

    print(
        f"Bloc {i+1} | OOF={len(block)} "
        f"| TRAIN={len(fit)} "
        f"| positifs={int(block.y.sum())}"
    )

result = pd.concat(parts, ignore_index=True)

print("\n=== PERFORMANCES GLOBALES ===")

for name in ("SIMPLE6", "SIMPLE7_RED"):
    y = result.y.astype(int)
    p = result[name].astype(float)

    print(
        f"{name:12s} | "
        f"AUC={roc_auc_score(y,p):.4f} | "
        f"AP={average_precision_score(y,p):.4f} | "
        f"Brier={brier_score_loss(y,p):.4f}"
    )

print("\n=== TOP 10 PAR BLOC ===")

for i, (fold, block) in enumerate(
    result.groupby("fold", sort=True),
    start=1,
):
    print(f"\nBLOC {i}")

    for name in ("SIMPLE6", "SIMPLE7_RED"):
        top = block.nlargest(10, name)
        wins = int(top.y.sum())

        print(f"  {name:12s} : {wins}/10")

print("\n=== TOP 40 : 10 PAR BLOC ===")

for name in ("SIMPLE6", "SIMPLE7_RED"):
    top = pd.concat([
        block.nlargest(10, name)
        for _, block in result.groupby(
            "fold", sort=True
        )
    ])

    wins = int(top.y.sum())

    print(
        f"{name:12s} | {wins}/40 "
        f"= {wins/40:.1%}"
    )

print("\n=== SEUIL FIXE 0.30 ===")

for name in ("SIMPLE6", "SIMPLE7_RED"):
    selected = result[result[name] >= 0.30]
    n = len(selected)
    wins = int(selected.y.sum())

    if n:
        print(
            f"{name:12s} | {wins}/{n} "
            f"= {wins/n:.1%} "
            f"| couverture={n/len(result):.1%}"
        )
    else:
        print(f"{name:12s} | aucune sélection")

print("\n=== CARTONS ROUGES DANS LES 121 OOF ===")

for value in (0, 1):
    sub = result[
        result.red_card_imbalance_87 == value
    ]

    n = len(sub)
    wins = int(sub.y.sum())

    print(
        f"Déséquilibre={value} | "
        f"matchs={n} | positifs={wins} | "
        f"taux={wins/n:.1%}" if n
        else f"Déséquilibre={value} | aucun match"
    )

columns = [
    "match_id", "kickoff_utc", "fold", "y",
    "red_card_imbalance_87",
    "SIMPLE6", "SIMPLE7_RED",
]

result[columns].to_csv(
    DEST, index=False, mode="x"
)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("Aucun seuil optimisé.")
print("Même modèle logistique C=0.1.")
print("Même population OOF.")
print("Embargo chronologique de quatre heures.")
print("VALIDATION non consultée.")
print("TEST_FUTUR non consulté.")
print("Modèles V4 figés inchangés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 223 TERMINÉE.")
