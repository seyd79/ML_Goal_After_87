import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BASE = Path("outputs/v4_research")

DATA = BASE / "v41_dataset_preparation_343/V41_CORE49_197_QUARANTINE_NO_LABEL.csv"
FREEZE_DATA = BASE / "v41_dataset_freeze_344/V41_CORE49_197_QUARANTINE_FREEZE.json"

SPLIT = BASE / "v41_chronological_split_345/V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
FREEZE_SPLIT = BASE / "v41_split_freeze_346/V41_SPLIT_FREEZE_PROVISIONAL.json"

CANDIDATE = BASE / "v41_candidate_freeze_382/V41_A2_EXPLORATORY_FREEZE.json"

MODEL = BASE / "v41_fitted_candidate_383/V41_A2_RESEARCH_TRAIN112.joblib"
MODEL_REPORT = BASE / "v41_fitted_candidate_383/V41_A2_FITTED_RESEARCH_MANIFEST.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def missing(value):
    return str(value).strip().lower() in {
        "", "nan", "none", "null", "na", "n/a"
    }

print("=" * 72)
print("ETAPE 384 - PREPARATION VALIDATION V4.1-A2")
print("=" * 72)

data_freeze = load_json(FREEZE_DATA)
split_freeze = load_json(FREEZE_SPLIT)
candidate = load_json(CANDIDATE)
model_report = load_json(MODEL_REPORT)

assert sha(DATA) == data_freeze["sha256"]
assert sha(SPLIT) == split_freeze["split_sha256"]
assert sha(MODEL) == model_report["model_sha256"]
assert sha(CANDIDATE) == model_report["freeze_382_sha256"]

assert candidate["candidate"] == "V4.1-A2"
assert model_report["candidate"] == "V4.1-A2"
assert model_report["training_matches"] == 112
assert model_report["features"] == candidate["features"]
assert model_report["threshold"] == candidate["threshold"] == 0.60

assert model_report["validation_accessed"] is False
assert model_report["holdout_accessed"] is False
assert model_report["production_authorized"] is False

data = load_csv(DATA)
splits = load_csv(SPLIT)

assert len(data) == len(splits) == 197

by_id = {r["match_id"]: r for r in data}
split_by_id = {r["match_id"]: r for r in splits}

assert len(by_id) == len(split_by_id) == 197
assert set(by_id) == set(split_by_id)

train_ids = [
    mid for mid, row in split_by_id.items()
    if row["split"] == "TRAIN_PROVISIONAL"
]
validation_ids = [
    mid for mid, row in split_by_id.items()
    if row["split"] == "VALIDATION_PROVISIONAL"
]
holdout_ids = [
    mid for mid, row in split_by_id.items()
    if row["split"] == "HOLDOUT_V41_PROVISIONAL"
]

assert len(train_ids) == 112
assert len(validation_ids) == 46
assert len(holdout_ids) == 39

assert not (
    set(train_ids) & set(validation_ids)
    or set(train_ids) & set(holdout_ids)
    or set(validation_ids) & set(holdout_ids)
)

features = candidate["features"]
assert len(features) == len(set(features)) == 10
assert all(f in data[0] for f in features)

# Aucune cible historique ne doit se trouver dans la matrice technique.
for forbidden in (
    "goal_after_87",
    "target_status",
    "target_confidence",
):
    assert forbidden not in data[0], (
        "STOP : colonne cible presente : " + forbidden
    )

train = [by_id[mid] for mid in train_ids]
validation = [by_id[mid] for mid in validation_ids]
holdout = [by_id[mid] for mid in holdout_ids]

assert max(r["kickoff_utc"] for r in train) < min(
    r["kickoff_utc"] for r in validation
)
assert max(r["kickoff_utc"] for r in validation) < min(
    r["kickoff_utc"] for r in holdout
)

missing_counts = Counter()

for row in validation:
    for feature in features:
        value = row[feature]

        if missing(value):
            missing_counts[feature] += 1
            continue

        number = float(value)

        if not math.isfinite(number):
            raise RuntimeError(
                "STOP : valeur non finie "
                + row["match_id"] + " / " + feature
            )

print("\n=== INTEGRITE ===")
print("Dataset technique : OK")
print("Partitions chronologiques : OK")
print("Modele V4.1-A2 : OK")
print("Configuration gelee : OK")
print("Labels validation non charges : OK")

print("\n=== REPARTITION ===")
print("TRAIN :", len(train))
print("VALIDATION :", len(validation))
print("HOLDOUT :", len(holdout))

print("\n=== CARACTERISTIQUES VALIDATION ===")
print("Features attendues :", len(features))

for feature in features:
    print(
        f"{feature:38s} "
        f"manquantes={missing_counts[feature]:2d}/46"
    )

print("\n=== LIMITES ===")
print("Correction own_goal validation : NON VERIFIEE")
print("Labels validation : NON RECONSTRUITS")
print("Exclusions protegees completes : NON CERTIFIEES")
print("Predictions validation : NON EFFECTUEES")
print("HOLDOUT : NON UTILISE POUR EVALUATION")
print("Aucun fichier modifie")
print("Aucun appel API")
print("Aucun modele entraine")
print("ETAPE 384 TERMINEE")
