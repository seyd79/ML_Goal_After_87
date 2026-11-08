import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("outputs/v4_research")

MODEL = BASE / "v41_fitted_candidate_383/V41_A2_RESEARCH_TRAIN112.joblib"
FREEZE = BASE / "v41_candidate_freeze_382/V41_A2_EXPLORATORY_FREEZE.json"
MATRIX = BASE / "v41_mass350_final_features_404W/V41_MASS350_FINAL_FEATURES_BLIND.csv"
SOURCE_MANIFEST = BASE / "v41_mass350_final_features_404W/V41_MASS350_FINAL_FEATURES_MANIFEST.json"

OUT = BASE / "v41_mass350_predictions_404Y"
PRED_FILE = OUT / "V41_A2_MASS344_PREDICTIONS_BLIND.csv"
REPORT_FILE = OUT / "V41_A2_MASS344_PREDICTIONS_FREEZE.json"

EXPECTED_MODEL_SHA = "cb9e21b7a2f2ae9517bb9de9a7978486c1ba05ebc19a7cb4e712a7cd56a79d99"
EXPECTED_MATRIX_SHA = "e47592c677aa2f0932c3d104ab7291bea18035736baabfa58c2fa2b5ca51728c"

EXPECTED_FEATURES = [
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

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stop(message):
    raise SystemExit("STOP : " + message)

print("=" * 76)
print("404Y — PREDICTIONS AVEUGLES V4.1-A2 / MASS350")
print("=" * 76)

if OUT.exists():
    stop("dossier 404Y deja existant")

if sha(MODEL) != EXPECTED_MODEL_SHA:
    stop("modele modifie")

if sha(MATRIX) != EXPECTED_MATRIX_SHA:
    stop("matrice 404W modifiee")

freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
source_manifest = json.loads(
    SOURCE_MANIFEST.read_text(encoding="utf-8")
)

if source_manifest.get("output_sha256") != EXPECTED_MATRIX_SHA:
    stop("manifeste 404W incoherent")

if freeze.get("candidate") != "V4.1-A2":
    stop("candidat incorrect")

features = freeze.get("features")
if features != EXPECTED_FEATURES:
    stop("ordre des 10 variables incorrect")

threshold = float(freeze.get("threshold", -1))
if threshold != 0.60:
    stop("seuil incorrect")

# Lire exclusivement les metadonnees et les 10 features.
# Ne jamais charger les labels, meme indirectement.
wanted = ["match_id", "kickoff_utc"] + features

with MATRIX.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    if reader.fieldnames != wanted:
        stop("schema de la matrice inattendu")

    rows = [{k: row[k] for k in wanted} for row in reader]

if len(rows) != 350:
    stop("350 matchs attendus")

ids = [r["match_id"] for r in rows]
if len(set(ids)) != 350:
    stop("identifiants dupliques")

complete = []
excluded = []

for row in rows:
    values = []

    for feature in features:
        raw = str(row[feature]).strip()
        try:
            value = float(raw) if raw else None
        except ValueError:
            stop("valeur non numerique : " + row["match_id"])

        if value is not None and not math.isfinite(value):
            stop("valeur non finie : " + row["match_id"])

        values.append(value)

    if all(value is not None for value in values):
        complete.append((row, values))
    else:
        excluded.append(row["match_id"])

if len(complete) != 344 or len(excluded) != 6:
    stop("couverture inattendue")

model = joblib.load(MODEL)

if not hasattr(model, "predict_proba"):
    stop("modele sans predict_proba")

classes = list(model.classes_)
if classes != [0, 1]:
    stop("ordre des classes inattendu")

if model.n_features_in_ != 10:
    stop("modele incompatible avec les features")

X = pd.DataFrame(
    [values for _, values in complete],
    columns=features,
    dtype=float,
)

proba = model.predict_proba(X)

if proba.shape != (344, 2):
    stop("forme des probabilites incorrecte")

if not np.isfinite(proba).all():
    stop("probabilite non finie")

if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-10):
    stop("probabilites non normalisees")

positive_index = classes.index(1)
positive_proba = proba[:, positive_index]

output_rows = []

for (row, _), probability in zip(complete, positive_proba):
    p = float(probability)

    if not 0.0 <= p <= 1.0:
        stop("probabilite hors intervalle")

    output_rows.append({
        "match_id": row["match_id"],
        "kickoff_utc": row["kickoff_utc"],
        "probability_goal_after_87": format(p, ".17g"),
        "threshold": "0.60",
        "decision": "ALERT" if p >= threshold else "NO_ALERT",
    })

output_rows.sort(key=lambda r: (r["kickoff_utc"], r["match_id"]))

alerts = sum(r["decision"] == "ALERT" for r in output_rows)

OUT.mkdir(parents=True, exist_ok=False)

with PRED_FILE.open("x", encoding="utf-8", newline="") as f:
    fieldnames = [
        "match_id",
        "kickoff_utc",
        "probability_goal_after_87",
        "threshold",
        "decision",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

report = {
    "status": "EXPLORATORY_BLIND_PREDICTIONS_FROZEN",
    "candidate": "V4.1-A2",
    "model_sha256": sha(MODEL),
    "freeze_sha256": sha(FREEZE),
    "input_features_sha256": sha(MATRIX),
    "features_in_order": features,
    "threshold": threshold,
    "matches_total": len(rows),
    "matches_scored": len(output_rows),
    "matches_excluded": len(excluded),
    "excluded_match_ids": sorted(excluded),
    "alerts": alerts,
    "no_alerts": len(output_rows) - alerts,
    "predictions_sha256": sha(PRED_FILE),
    "labels_accessed": False,
    "model_retrained": False,
    "fully_independent_holdout_certified": False,
    "prematch_temporal_certified": False,
    "live_equivalence_certified": False,
}

with REPORT_FILE.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("Modele :", report["candidate"])
print("Matchs totaux :", report["matches_total"])
print("Matchs avec prediction :", report["matches_scored"])
print("Matchs exclus :", report["matches_excluded"])
print("Seuil :", threshold)

print("\n=== DECISIONS ===")
print("ALERT :", alerts)
print("NO_ALERT :", len(output_rows) - alerts)

print("\n=== GEL DES PREDICTIONS ===")
print("SHA256 :", report["predictions_sha256"])
print("Fichier :", PRED_FILE)
print("Manifeste :", REPORT_FILE)

print("\n404Y TERMINEE")
print("Aucun label consulte.")
print("Aucun reentrainement.")
print("Aucun resultat reel compare.")
