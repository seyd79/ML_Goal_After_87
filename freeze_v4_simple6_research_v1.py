from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT = Path("outputs/v4_research")
SOURCE = OUT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
DEST = OUT / "frozen_simple6_v1"

FILES = [
    DEST / "SIMPLE6_OLD183.joblib",
    DEST / "SIMPLE6_NEW278.joblib",
    DEST / "V4_FROZEN_PROTOCOL.json",
]

print("=" * 72)
print("ÉTAPE 206 — GEL DES CANDIDATS V4 RESEARCH")
print("=" * 72)

if any(p.exists() for p in FILES):
    raise SystemExit(
        "Sortie déjà existante : aucun remplacement autorisé"
    )

df = pd.read_csv(SOURCE, dtype={"match_id": str})

if len(df) != 278 or df.match_id.duplicated().any():
    raise SystemExit("TRAIN278 invalide")

old = df[
    ~df.research_origin.isin(
        ["V4_CACHED12", "V4_FETCH83"]
    )
].copy()

if (
    len(old) != 183
    or int(old.goal_after_87.sum()) != 49
    or int(df.goal_after_87.sum()) != 72
):
    raise SystemExit("Effectifs historiques incorrects")

dates = pd.to_datetime(
    df.kickoff_utc, utc=True, errors="raise"
)

if not (
    dates < pd.Timestamp("2026-08-25T00:00:00Z")
).all():
    raise SystemExit("Date TRAIN non autorisée")

FEATURES = [
    "score_total_87",
    "score_diff_abs_87",
    "is_draw_87",
    "shots_total_last5",
    "sot_total_last5",
    "corners_total_last5",
]

def make_features(frame):
    def n(name):
        return pd.to_numeric(
            frame[name], errors="raise"
        )

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
    })[FEATURES]

def train(frame):
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
        make_features(frame),
        frame.goal_after_87.astype(int)
    )
    return model

models = {
    "SIMPLE6_OLD183": train(old),
    "SIMPLE6_NEW278": train(df),
}

digest = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()

protocol = {
    "version": "V4_SIMPLE6_FROZEN_RESEARCH_V1",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "RESEARCH_ONLY",
    "primary_model": "SIMPLE6_OLD183",
    "challenger_model": "SIMPLE6_NEW278",
    "features": FEATURES,
    "model_type": "LogisticRegression",
    "C": 0.1,
    "selection_rule": "probability >= 0.30",
    "threshold": 0.30,
    "selection_rule_status": "exploratory_not_certified",
    "target": "goal_after_87",
    "train_counts": {
        "SIMPLE6_OLD183": 183,
        "SIMPLE6_NEW278": 278,
    },
    "train_positives": {
        "SIMPLE6_OLD183": 49,
        "SIMPLE6_NEW278": 72,
    },
    "source_sha256": digest,
    "evaluation_plan": {
        "same_future_matches_for_both_models": True,
        "no_parameter_tuning_during_evaluation": True,
        "compare_precision_and_coverage": True,
        "compare_brier_and_roc_auc": True,
        "report_uncertainty_intervals": True,
        "target_precision": 0.60,
        "independent_future_data_required": True,
    },
    "live_certified": False,
    "training_certified": False,
}

DEST.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    path = DEST / f"{name}.joblib"
    joblib.dump(model, path)
    print("MODÈLE FIGÉ :", path)

with (DEST / "V4_FROZEN_PROTOCOL.json").open(
    "x", encoding="utf-8"
) as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

print("\n=== BILAN ===")
print("Référence : SIMPLE6_OLD183")
print("Concurrent : SIMPLE6_NEW278")
print("Seuil expérimental : 0.30")
print("Objectif de précision : 60 %")
print("Validation non consultée.")
print("TEST_FUTUR non consulté.")
print("Aucun scanner LIVE modifié.")
print("Modèles non certifiés.")
print("ÉTAPE 206 TERMINÉE.")
