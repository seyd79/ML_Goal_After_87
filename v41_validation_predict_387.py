import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

BASE = Path("outputs/v4_research")

DATA = (
    BASE / "v41_dataset_preparation_343"
    / "V41_CORE49_197_QUARANTINE_NO_LABEL.csv"
)
DATA_FREEZE = (
    BASE / "v41_dataset_freeze_344"
    / "V41_CORE49_197_QUARANTINE_FREEZE.json"
)

SPLIT = (
    BASE / "v41_chronological_split_345"
    / "V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
)
SPLIT_FREEZE = (
    BASE / "v41_split_freeze_346"
    / "V41_SPLIT_FREEZE_PROVISIONAL.json"
)

FREEZE = (
    BASE / "v41_candidate_freeze_382"
    / "V41_A2_EXPLORATORY_FREEZE.json"
)

MODEL_DIR = BASE / "v41_fitted_candidate_383"
MODEL_FILE = MODEL_DIR / "V41_A2_RESEARCH_TRAIN112.joblib"
MODEL_REPORT = MODEL_DIR / "V41_A2_FITTED_RESEARCH_MANIFEST.json"

VALID_DIR = BASE / "v41_validation_own_goal_385"
VALID_FILE = VALID_DIR / "V41_A2_VALIDATION46_FEATURES_BLIND.csv"
VALID_REPORT = VALID_DIR / "V41_A2_VALIDATION_FEATURES_MANIFEST.json"

OUT = BASE / "v41_validation_predictions_387"
PRED_FILE = OUT / "V41_A2_VALIDATION46_PREDICTIONS_BLIND.csv"
REPORT_FILE = OUT / "V41_A2_VALIDATION46_PREDICTIONS_FREEZE.json"

META = [
    "match_id",
    "competition_id",
    "season_id",
    "kickoff_utc",
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    if not columns or len(set(columns)) != len(columns):
        raise RuntimeError("STOP : colonnes CSV invalides")

    return columns, rows

def check(condition, message):
    if not condition:
        raise RuntimeError("STOP : " + message)

print("=" * 72)
print("ETAPE 387 - PREDICTIONS AVEUGLES V4.1-A2")
print("=" * 72)

check(not OUT.exists(), "dossier 387 deja existant")

data_freeze = read_json(DATA_FREEZE)
split_freeze = read_json(SPLIT_FREEZE)
candidate = read_json(FREEZE)
model_report = read_json(MODEL_REPORT)
valid_report = read_json(VALID_REPORT)

# Integrite des sources figees.
check(sha(DATA) == data_freeze["sha256"],
      "source 343 modifiee")
check(sha(SPLIT) == split_freeze["split_sha256"],
      "split 345 modifie")
check(sha(FREEZE) == model_report["freeze_382_sha256"],
      "gel 382 modifie")
check(sha(MODEL_FILE) == model_report["model_sha256"],
      "modele 383 modifie")
check(sha(VALID_FILE) == valid_report["output_sha256"],
      "features validation 385 modifiees")

check(valid_report["source_sha256"] == sha(DATA),
      "source validation incoherente")
check(valid_report["split_sha256"] == sha(SPLIT),
      "split validation incoherent")
check(valid_report["candidate_freeze_sha256"] == sha(FREEZE),
      "candidat validation incoherent")

check(candidate["candidate"] == "V4.1-A2",
      "candidat inattendu")
check(candidate["status"] == "EXPLORATORY_CANDIDATE_FROZEN",
      "configuration non gelee")

check(model_report["status"] == "FITTED_EXPLORATORY_ONLY",
      "modele non conforme")

check(model_report["training_matches"] == 112,
      "taille TRAIN incorrecte")

check(model_report["features"] == candidate["features"],
      "features modele incorrectes")

check(valid_report["features"] == candidate["features"],
      "features validation incorrectes")

check(candidate["regularization_C"] == 0.1,
      "regularisation modifiee")
check(candidate["threshold"] == 0.60,
      "seuil modifie")

check(model_report["production_authorized"] is False,
      "statut production inattendu")
check(valid_report["labels_calculated"] is False,
      "statut labels inattendu")
check(valid_report["model_predictions_calculated"] is False,
      "statut predictions source inattendu")

features = candidate["features"]
threshold = candidate["threshold"]

check(len(features) == len(set(features)) == 10,
      "schema 10 features invalide")

# Lire le split pour confirmer l'appartenance exacte.
_, split_rows = read_csv(SPLIT)

split_ids = {
    "TRAIN_PROVISIONAL": set(),
    "VALIDATION_PROVISIONAL": set(),
    "HOLDOUT_V41_PROVISIONAL": set(),
}

all_ids = []

for row in split_rows:
    mid = row["match_id"]
    role = row["split"]

    check(role in split_ids,
          "role split inconnu")
    split_ids[role].add(mid)
    all_ids.append(mid)

check(len(all_ids) == len(set(all_ids)) == 197,
      "IDs split dupliques")

train_ids = split_ids["TRAIN_PROVISIONAL"]
validation_ids = split_ids["VALIDATION_PROVISIONAL"]
holdout_ids = split_ids["HOLDOUT_V41_PROVISIONAL"]

check(len(train_ids) == 112, "TRAIN incorrect")
check(len(validation_ids) == 46, "validation incorrecte")
check(len(holdout_ids) == 39, "holdout incorrect")

check(not (train_ids & validation_ids),
      "chevauchement TRAIN/VALIDATION")
check(not (holdout_ids & validation_ids),
      "chevauchement HOLDOUT/VALIDATION")

columns, rows = read_csv(VALID_FILE)

check(columns == META + features,
      "colonnes validation inattendues")

check(len(rows) == 46, "46 rencontres attendues")

ids = [r["match_id"] for r in rows]

check(len(ids) == len(set(ids)) == 46,
      "IDs validation dupliques")

check(set(ids) == validation_ids,
      "rencontres hors validation")

# Bloquer explicitement toute cible.
for forbidden in (
    "goal_after_87",
    "target",
    "label",
    "target_status",
    "final_score",
    "score_ft",
):
    check(forbidden not in columns,
          "colonne cible interdite : " + forbidden)

# Matrice dans l'ordre exact du candidat gele.
X = np.asarray([
    [float(row[f]) for f in features]
    for row in rows
], dtype=float)

check(X.shape == (46, 10), "matrice invalide")
check(np.isfinite(X).all(),
      "valeurs manquantes ou infinies")

# Charger seulement l'artefact cree localement a l'etape 383.
# Ne jamais utiliser joblib.load sur un fichier non fiable.
model = joblib.load(MODEL_FILE)

check(list(model.named_steps["classifier"].classes_) == [0, 1],
      "classes inattendues")

classifier = model.named_steps["classifier"]

check(classifier.C == 0.1,
      "regularisation du modele differente")
check(classifier.penalty == "l2",
      "penalite differente")
check(classifier.solver == "liblinear",
      "solveur different")
check(classifier.class_weight is None,
      "class_weight different")

check(model.named_steps["imputer"].strategy == "median",
      "imputation differente")

check(model.named_steps["scaler"].with_mean is True,
      "standardisation differente")

check(int(model.n_features_in_) == 10,
      "nombre de features modele incorrect")

# Aucun fit / refit : predictions uniquement.
probabilities = model.predict_proba(X)[:, 1]

check(len(probabilities) == 46,
      "nombre de predictions incorrect")

check(np.isfinite(probabilities).all(),
      "probabilites non finies")

check(np.all((probabilities >= 0) & (probabilities <= 1)),
      "probabilites hors bornes")

output_rows = []

for row, probability in zip(rows, probabilities):
    p = float(probability)

    output_rows.append({
        "match_id": row["match_id"],
        "kickoff_utc": row["kickoff_utc"],
        "probability": format(p, ".17g"),
        "alert": int(p >= threshold),
    })

output_rows.sort(
    key=lambda r: (r["kickoff_utc"], r["match_id"])
)

OUT.mkdir(parents=True, exist_ok=False)

with PRED_FILE.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "match_id",
            "kickoff_utc",
            "probability",
            "alert",
        ],
    )
    writer.writeheader()
    writer.writerows(output_rows)

alerts = sum(int(r["alert"]) for r in output_rows)

report = {
    "candidate": "V4.1-A2",
    "status": "VALIDATION_PREDICTIONS_BLIND_FROZEN",
    "model_sha256": sha(MODEL_FILE),
    "candidate_freeze_sha256": sha(FREEZE),
    "validation_features_sha256": sha(VALID_FILE),
    "split_sha256": sha(SPLIT),
    "predictions_sha256": sha(PRED_FILE),
    "script_sha256": sha(Path(__file__)),
    "features": features,
    "regularization_C": 0.1,
    "threshold": threshold,
    "prediction_count": 46,
    "alert_count": alerts,
    "labels_accessed": False,
    "outcomes_evaluated": False,
    "independent_validation_certified": False,
    "protected_sets_fully_certified": False,
    "temporal_availability_fully_certified": False,
    "holdout_accessed": False,
    "production_authorized": False,
}

with REPORT_FILE.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 72)
print("BILAN ETAPE 387")
print("=" * 72)

print("Modele V4.1-A2 : INCHANGE")
print("Matchs de validation :", len(output_rows))
print("Features :", len(features))
print("Seuil :", threshold)
print("Alertes generees :", alerts)
print("Non-alertes :", len(output_rows) - alerts)

print("\nPredictions :", PRED_FILE)
print("SHA256 predictions :", sha(PRED_FILE))
print("Manifeste :", REPORT_FILE)

print("\nLabels consultes : NON")
print("Precision calculee : NON")
print("Nouvel entrainement : NON")
print("Holdout consulte : NON")
print("V3-C modifie : NON")
print("Certifications restantes : EN ATTENTE")
print("ETAPE 387 TERMINEE")
