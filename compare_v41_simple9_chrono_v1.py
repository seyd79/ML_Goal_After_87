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

TRAIN_FILE = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
PRESSURE_FILE = ROOT / "V41_PRESSURE10_AUDIT_V1.csv"
OOF_FILE = ROOT / "V4_SIMPLE_CHRONO_OOF_V1.csv"

DEST = ROOT / "V41_SIMPLE9_CHRONO_COMPARISON_V1.csv"

EMBARGO = pd.Timedelta(hours=4)

if DEST.exists():
    raise SystemExit(
        "Fichier de sortie déjà existant. "
        "Remplacement interdit."
    )

print("=" * 72)
print("ÉTAPE 221 — V4.1 SIMPLE6 VS SIMPLE9")
print("=" * 72)

train = pd.read_csv(
    TRAIN_FILE, dtype={"match_id": str}
)
pressure = pd.read_csv(
    PRESSURE_FILE, dtype={"match_id": str}
)
oof = pd.read_csv(
    OOF_FILE, dtype={"match_id": str}
)

if (
    len(train) != 278
    or len(pressure) != 278
    or len(oof) != 121
):
    raise SystemExit("Effectifs inattendus")

for name, df in [
    ("TRAIN", train),
    ("PRESSURE", pressure),
    ("OOF", oof),
]:
    if df.match_id.duplicated().any():
        raise SystemExit(f"Doublons dans {name}")

if set(train.match_id) != set(pressure.match_id):
    raise SystemExit(
        "Identifiants TRAIN/PRESSURE différents"
    )

if not (
    pressure.status == "RECONSTRUCTED_RESEARCH_ONLY"
).all():
    raise SystemExit(
        "Des reconstructions sont invalides"
    )

NEEDED = [
    "shots_acceleration",
    "sot_acceleration",
    "corners_acceleration",
]

train = train.merge(
    pressure[["match_id"] + NEEDED],
    on="match_id",
    validate="one_to_one",
)

if len(train) != 278:
    raise SystemExit("Fusion incorrecte")

train["kickoff"] = pd.to_datetime(
    train.kickoff_utc,
    utc=True,
    errors="raise",
)

oof["kickoff"] = pd.to_datetime(
    oof.kickoff_utc,
    utc=True,
    errors="raise",
)

if not (
    train.kickoff < pd.Timestamp(
        "2026-08-25T00:00:00Z"
    )
).all():
    raise SystemExit("Match hors période TRAIN")

for name in NEEDED:
    train[name] = pd.to_numeric(
        train[name], errors="raise"
    )

    if not np.isfinite(train[name]).all():
        raise SystemExit(
            f"Variable invalide : {name}"
        )

old = train[
    ~train.research_origin.isin([
        "V4_CACHED12",
        "V4_FETCH83",
    ])
].copy()

if len(old) != 183:
    raise SystemExit("TRAIN historique incorrect")

if not set(oof.match_id).issubset(
    set(old.match_id)
):
    raise SystemExit(
        "OOF contient des matchs hors TRAIN ancien"
    )

lookup = train.set_index("match_id")

labels = lookup.loc[
    oof.match_id, "goal_after_87"
].to_numpy(dtype=int)

if not np.array_equal(
    labels,
    oof.y.to_numpy(dtype=int)
):
    raise SystemExit("Labels OOF incompatibles")

if (
    oof.fold.nunique() != 4
    or int(oof.y.sum()) != 31
):
    raise SystemExit("Folds OOF inattendus")


SIMPLE6 = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]

SIMPLE9 = SIMPLE6 + NEEDED


def make_features(df, columns):
    def n(col):
        return pd.to_numeric(
            df[col], errors="raise"
        )

    home = n("score_87_home")
    away = n("score_87_away")

    x = pd.DataFrame(index=df.index)

    x["score_total_87"] = home + away
    x["score_diff_abs_87"] = (home - away).abs()
    x["is_draw_87"] = n("is_draw_87")

    x["shots_total_last5"] = (
        n("shots_home_last5")
        + n("shots_away_last5")
    )

    x["sot_total_last5"] = (
        n("sot_home_last5")
        + n("sot_away_last5")
    )

    x["corners_total_last5"] = (
        n("corners_home_last5")
        + n("corners_away_last5")
    )

    for name in NEEDED:
        if name in columns:
            x[name] = n(name)

    return x[columns]


def predict(fit, test, columns):
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
        make_features(fit, columns),
        fit.goal_after_87.astype(int),
    )

    return model.predict_proba(
        make_features(test, columns)
    )[:, 1]


CONFIGS = {
    "S6_OLD183": ("old", SIMPLE6),
    "S9_OLD183": ("old", SIMPLE9),
    "S6_NEW278": ("new", SIMPLE6),
    "S9_NEW278": ("new", SIMPLE9),
}

expected_old = [50, 76, 114, 128]
expected_new = [58, 108, 188, 209]

parts = []

print("\n=== VALIDATION CHRONOLOGIQUE ===")

for i, (fold, block) in enumerate(
    oof.groupby("fold", sort=True)
):
    block = block.copy()

    cutoff = block.kickoff.min() - EMBARGO

    fit_old = old[
        old.kickoff < cutoff
    ].copy()

    fit_new = train[
        train.kickoff < cutoff
    ].copy()

    if (
        len(fit_old) != expected_old[i]
        or len(fit_new) != expected_new[i]
    ):
        raise SystemExit(
            f"Effectifs du fold {fold} inattendus"
        )

    if set(block.match_id) & set(fit_new.match_id):
        raise SystemExit(
            "Fuite temporelle : match OOF dans TRAIN"
        )

    test = lookup.loc[
        block.match_id
    ].reset_index()

    if not (
        fit_new.kickoff.max() < cutoff
    ):
        raise SystemExit(
            "Fuite chronologique détectée"
        )

    print(
        f"Bloc {i+1} : "
        f"OOF={len(block)}, "
        f"TRAIN183={len(fit_old)}, "
        f"TRAIN278={len(fit_new)}"
    )

    for name, (source, features) in CONFIGS.items():
        fit = (
            fit_old if source == "old"
            else fit_new
        )

        block[name] = predict(
            fit, test, features
        )

    parts.append(block)

result = pd.concat(
    parts, ignore_index=True
)

print("\n=== PERFORMANCES GLOBALES ===")

for name in CONFIGS:
    y = result.y.astype(int)
    p = result[name]

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
    print(
        f"\nBloc {i} | "
        f"positifs={int(block.y.sum())}"
    )

    for name in CONFIGS:
        top = block.nlargest(10, name)

        wins = int(top.y.sum())

        print(
            f"  {name:12s} : "
            f"{wins}/10 = {wins/10:.0%}"
        )

print("\n=== RÉUSSITE TOP 40 (10 PAR BLOC) ===")

for name in CONFIGS:
    selected = pd.concat([
        block.nlargest(10, name)
        for _, block in result.groupby(
            "fold", sort=True
        )
    ])

    wins = int(selected.y.sum())

    print(
        f"{name:12s} : "
        f"{wins}/40 = {wins/40:.1%}"
    )

print("\n=== SEUIL FIXE 0.30 ===")

for name in CONFIGS:
    selected = result[
        result[name] >= 0.30
    ]

    count = len(selected)
    wins = int(selected.y.sum())

    if count:
        print(
            f"{name:12s} : "
            f"{wins}/{count} = {wins/count:.1%}"
        )
    else:
        print(
            f"{name:12s} : aucune sélection"
        )

columns = (
    ["match_id", "kickoff_utc", "fold", "y"]
    + list(CONFIGS)
)

result[columns].to_csv(
    DEST,
    index=False,
    mode="x",
)

print("\n=== FICHIER CRÉÉ ===")
print(DEST)

print("\n=== GARANTIES ===")
print("Même population OOF : 121 matchs.")
print("Embargo : 4 heures.")
print("Aucun nouveau seuil optimisé.")
print("Aucune donnée VALIDATION consultée.")
print("Aucune donnée TEST_FUTUR consultée.")
print("Modèles figés V4 inchangés.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 221 TERMINÉE.")
