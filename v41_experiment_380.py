import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = Path("outputs/v4_research")

DATA = BASE / "v41_candidate_a_375/V41_A_TRAIN112_CORE43_PROVISIONAL.csv"
SOURCE_MANIFEST = BASE / "v41_candidate_a_375/V41_A_CORE43_MANIFEST.json"

FOLDS = BASE / "v41_protocol_376/V41_A_TRAIN_ONLY_FOLDS.csv"
PROTOCOL = BASE / "v41_protocol_376/V41_A_PROTOCOL_FREEZE.json"

OUT = BASE / "v41_experiment_380"
PRED = OUT / "V41_A2_OOF60_PREDICTIONS.csv"
REPORT = OUT / "V41_A2_EXPERIMENT_REPORT.json"

FEATURES = [
    "score_87_home",
    "score_87_away",
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "minutes_since_last_shot_on_target",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
]

C = 0.1
THRESHOLD = 0.60

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def matrix(rows):
    result = []
    for r in rows:
        values = []
        for feature in FEATURES:
            v = str(r[feature]).strip()
            if v.lower() in ("", "nan", "none", "null", "na", "n/a"):
                values.append(np.nan)
            else:
                values.append(float(v))
        result.append(values)
    return np.asarray(result, dtype=float)

print("=" * 70)
print("ETAPE 380 - V4.1-A2, 10 FEATURES")
print("=" * 70)

if OUT.exists():
    raise RuntimeError("STOP : dossier 380 deja existant")

source_manifest = read_json(SOURCE_MANIFEST)
protocol = read_json(PROTOCOL)

assert sha(DATA) == source_manifest["output_sha256"]
assert sha(DATA) == protocol["source_sha256"]
assert sha(FOLDS) == protocol["fold_file_sha256"]

assert source_manifest["training_authorized"] is False
assert protocol["training_authorized"] is False

assert len(FEATURES) == 10
assert len(set(FEATURES)) == 10
assert set(FEATURES).issubset(source_manifest["features"])

rows = read_csv(DATA)
assignments = read_csv(FOLDS)

by_id = {r["match_id"]: r for r in rows}

assert len(rows) == len(by_id) == 112
assert sorted(int(r["goal_after_87"]) for r in rows).count(1) == 38

predictions = []
seen_oof = set()

for fold in (1, 2, 3):
    fit_ids = [
        r["match_id"] for r in assignments
        if int(r["fold"]) == fold and r["role"] == "FIT"
    ]

    oof_ids = [
        r["match_id"] for r in assignments
        if int(r["fold"]) == fold and r["role"] == "OUT_OF_FOLD"
    ]

    assert fit_ids and oof_ids
    assert len(fit_ids) == len(set(fit_ids))
    assert len(oof_ids) == len(set(oof_ids))
    assert set(fit_ids).isdisjoint(oof_ids)
    assert set(fit_ids + oof_ids).issubset(by_id)
    assert not seen_oof.intersection(oof_ids)

    fit_rows = [by_id[mid] for mid in fit_ids]
    oof_rows = [by_id[mid] for mid in oof_ids]

    assert max(r["kickoff_utc"] for r in fit_rows) < min(
        r["kickoff_utc"] for r in oof_rows
    )

    X_fit = matrix(fit_rows)
    X_oof = matrix(oof_rows)

    assert np.isfinite(X_fit[~np.isnan(X_fit)]).all()
    assert np.isfinite(X_oof[~np.isnan(X_oof)]).all()
    assert not np.isnan(X_fit).all(axis=0).any()

    y_fit = np.asarray(
        [int(r["goal_after_87"]) for r in fit_rows]
    )

    assert set(y_fit) == {0, 1}

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight=None,
            max_iter=2000,
            random_state=41,
        )),
    ])

    model.fit(X_fit, y_fit)

    probabilities = model.predict_proba(X_oof)[:, 1]

    wins = 0
    alerts = 0

    for row, p in zip(oof_rows, probabilities):
        y = int(row["goal_after_87"])
        alert = int(p >= THRESHOLD)

        alerts += alert
        wins += alert * y

        predictions.append({
            "fold": fold,
            "match_id": row["match_id"],
            "kickoff_utc": row["kickoff_utc"],
            "label": y,
            "probability": float(p),
            "alert": alert,
        })

    seen_oof.update(oof_ids)

    print(
        f"FOLD {fold} | FIT={len(fit_ids)}"
        f" | OOF={len(oof_ids)}"
        f" | alertes={alerts}"
        f" | gagnantes={wins}"
    )

assert len(predictions) == len(seen_oof) == 60

predictions.sort(key=lambda r: (
    r["kickoff_utc"], r["match_id"]
))

y = np.asarray([r["label"] for r in predictions])
p = np.asarray([r["probability"] for r in predictions])
selected = p >= THRESHOLD

alerts = int(selected.sum())
wins = int(y[selected].sum())
precision = wins / alerts if alerts else None

auc = float(roc_auc_score(y, p))
brier = float(brier_score_loss(y, p))
recall = wins / int(y.sum())

OUT.mkdir(parents=True, exist_ok=False)

with PRED.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold", "match_id", "kickoff_utc",
            "label", "probability", "alert"
        ]
    )
    writer.writeheader()
    writer.writerows(predictions)

report = {
    "status": "EXPLORATORY_REUSED_OOF_NOT_INDEPENDENT",
    "candidate": "V4.1-A2",
    "feature_count": len(FEATURES),
    "features": FEATURES,
    "regularization_C": C,
    "threshold": THRESHOLD,
    "dataset_sha256": sha(DATA),
    "folds_sha256": sha(FOLDS),
    "predictions_sha256": sha(PRED),
    "matches": 60,
    "positives": int(y.sum()),
    "alerts": alerts,
    "wins": wins,
    "precision": precision,
    "recall": recall,
    "roc_auc": auc,
    "brier": brier,
    "oof_previously_examined": True,
    "independent_validation": False,
    "official_training_authorized": False,
    "production_authorized": False,
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("RESULTATS V4.1-A2")
print("=" * 70)
print("Matchs OOF :", len(y))
print("Features :", len(FEATURES))
print("Alertes :", alerts)
print("Gagnantes :", wins)
print("Perdantes :", alerts - wins)

if precision is None:
    print("Precision : non definie (0 alerte)")
else:
    print(f"Precision : {precision * 100:.2f}%")

print(f"Rappel : {recall * 100:.2f}%")
print(f"ROC AUC : {auc:.4f}")
print(f"Brier : {brier:.5f}")
print("Predictions :", PRED)
print("Rapport :", REPORT)
print("Resultats exploratoires, OOF deja examines.")
print("Validation officielle / HOLDOUT : NON CONSULTES")
print("V3-C : INCHANGE")
print("ETAPE 380 TERMINEE")
