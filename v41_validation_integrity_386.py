import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path("outputs/v4_research")

SOURCE = BASE / "v41_dataset_preparation_343/V41_CORE49_197_QUARANTINE_NO_LABEL.csv"
SOURCE_FREEZE = BASE / "v41_dataset_freeze_344/V41_CORE49_197_QUARANTINE_FREEZE.json"

SPLIT = BASE / "v41_chronological_split_345/V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
SPLIT_FREEZE = BASE / "v41_split_freeze_346/V41_SPLIT_FREEZE_PROVISIONAL.json"

CANDIDATE = BASE / "v41_candidate_freeze_382/V41_A2_EXPLORATORY_FREEZE.json"

MODEL = BASE / "v41_fitted_candidate_383/V41_A2_RESEARCH_TRAIN112.joblib"
MODEL_REPORT = BASE / "v41_fitted_candidate_383/V41_A2_FITTED_RESEARCH_MANIFEST.json"

DIR385 = BASE / "v41_validation_own_goal_385"
VALIDATION = DIR385 / "V41_A2_VALIDATION46_FEATURES_BLIND.csv"
MANIFEST385 = DIR385 / "V41_A2_VALIDATION_FEATURES_MANIFEST.json"
CHECKPOINTS = DIR385 / "checkpoints"

CORRECTOR = Path("v41_live87_own_goal_364.py")

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
    if columns is None or len(columns) != len(set(columns)):
        raise RuntimeError("STOP : en-tetes CSV invalides")
    return columns, rows

def num(value):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("STOP : nombre non fini")
    return number

def check(condition, message):
    if not condition:
        raise RuntimeError("STOP : " + message)

print("=" * 72)
print("ETAPE 386 - INTEGRITE VALIDATION V4.1-A2")
print("=" * 72)

source_freeze = read_json(SOURCE_FREEZE)
split_freeze = read_json(SPLIT_FREEZE)
candidate = read_json(CANDIDATE)
model_report = read_json(MODEL_REPORT)
manifest = read_json(MANIFEST385)

check(sha(SOURCE) == source_freeze["sha256"],
      "source 343 modifiee")
check(sha(SPLIT) == split_freeze["split_sha256"],
      "split 345 modifie")
check(sha(MODEL) == model_report["model_sha256"],
      "modele 383 modifie")
check(sha(CANDIDATE) == model_report["freeze_382_sha256"],
      "gel 382 modifie")
check(sha(VALIDATION) == manifest["output_sha256"],
      "validation 385 modifiee")

check(manifest["source_sha256"] == sha(SOURCE),
      "source 385 incoherente")
check(manifest["split_sha256"] == sha(SPLIT),
      "split 385 incoherent")
check(manifest["candidate_freeze_sha256"] == sha(CANDIDATE),
      "gel 385 incoherent")
check(manifest["corrector_sha256"] == sha(CORRECTOR),
      "correctif 364 modifie")

check(candidate["candidate"] == "V4.1-A2",
      "candidat incorrect")
check(candidate["threshold"] == 0.60,
      "seuil incorrect")
check(candidate["regularization_C"] == 0.1,
      "regularisation incorrecte")
check(model_report["features"] == candidate["features"],
      "features modele differentes")
check(manifest["features"] == candidate["features"],
      "features validation differentes")

check(manifest["labels_calculated"] is False,
      "statut labels inattendu")
check(manifest["model_predictions_calculated"] is False,
      "statut predictions inattendu")
check(manifest["production_authorized"] is False,
      "statut production inattendu")

source_columns, source = read_csv(SOURCE)
split_columns, splits = read_csv(SPLIT)
validation_columns, validation = read_csv(VALIDATION)

source_by_id = {r["match_id"]: r for r in source}
validation_by_id = {r["match_id"]: r for r in validation}

check(len(source) == len(source_by_id) == 197,
      "source 197 incorrecte")
check(len(splits) == 197,
      "split 197 incorrect")
check(len(validation) == len(validation_by_id) == 46,
      "validation 46 incorrecte")

validation_ids = {
    r["match_id"]
    for r in splits
    if r["split"] == "VALIDATION_PROVISIONAL"
}

train_ids = {
    r["match_id"]
    for r in splits
    if r["split"] == "TRAIN_PROVISIONAL"
}

holdout_ids = {
    r["match_id"]
    for r in splits
    if r["split"] == "HOLDOUT_V41_PROVISIONAL"
}

check(len(validation_ids) == 46,
      "partition validation incorrecte")
check(len(train_ids) == 112 and len(holdout_ids) == 39,
      "tailles des partitions incorrectes")
check(set(validation_by_id) == validation_ids,
      "IDs validation differents")
check(not validation_ids.intersection(train_ids),
      "chevauchement TRAIN")
check(not validation_ids.intersection(holdout_ids),
      "chevauchement HOLDOUT")

features = candidate["features"]
meta = ["match_id", "competition_id", "season_id", "kickoff_utc"]
expected_columns = meta + features

check(validation_columns == expected_columns,
      "schema validation incorrect")

for forbidden in (
    "goal_after_87",
    "target_status",
    "target_confidence",
    "final_score",
    "score_ft",
):
    check(forbidden not in validation_columns,
          "colonne cible detectee : " + forbidden)

changed_ids = []
missing_values = 0

for mid in sorted(validation_ids):
    frozen = source_by_id[mid]
    row = validation_by_id[mid]

    checkpoint_path = CHECKPOINTS / (mid + ".json")
    check(checkpoint_path.is_file(),
          "checkpoint absent : " + mid)

    cp = read_json(checkpoint_path)

    check(cp["match_id"] == mid,
          "ID checkpoint incorrect : " + mid)
    check(cp["source_sha256"] == sha(SOURCE),
          "source checkpoint incorrecte : " + mid)
    check(cp["corrector_sha256"] == sha(CORRECTOR),
          "correctif checkpoint incorrect : " + mid)

    for field in meta:
        check(row[field] == frozen[field],
              f"metadonnee modifiee : {mid} / {field}")

    for field in features:
        value = row[field].strip()
        check(value.lower() not in {
            "", "nan", "none", "null", "na", "n/a"
        }, f"valeur manquante : {mid} / {field}")
        num(value)

        if field not in ("score_87_home", "score_87_away"):
            check(row[field] == frozen[field],
                  f"feature inattendue modifiee : {mid} / {field}")

    before = cp["score_before"]
    after = cp["score_after"]

    check(len(before) == len(after) == 2,
          "checkpoint score invalide : " + mid)

    check(
        num(frozen["score_87_home"]) == num(before[0])
        and num(frozen["score_87_away"]) == num(before[1]),
        "score initial incoherent : " + mid
    )

    check(
        num(row["score_87_home"]) == num(after[0])
        and num(row["score_87_away"]) == num(after[1]),
        "score corrige incoherent : " + mid
    )

    check(
        num(before[0]) + num(before[1])
        == num(after[0]) + num(after[1]),
        "total de buts altere : " + mid
    )

    changed = (
        num(before[0]) != num(after[0])
        or num(before[1]) != num(after[1])
    )

    check(changed == cp["score_changed"],
          "statut correction incoherent : " + mid)

    if changed:
        check(cp["own_goals_le87"] >= 1,
              "correction sans own_goal : " + mid)
        changed_ids.append(mid)

check(
    set(changed_ids) == set(manifest["corrected_match_ids"]),
    "liste des corrections differente du manifeste"
)

check(len(changed_ids) == 4,
      "nombre de corrections different du journal 385")

print("\n=== INTEGRITE DES DONNEES ===")
print("Empreinte source 343 : OK")
print("Empreinte split 345 : OK")
print("Empreinte modele 383 : OK")
print("Empreinte validation 385 : OK")
print("Configuration gelee : OK")

print("\n=== VALIDATION ===")
print("Matchs :", len(validation))
print("Features :", len(features))
print("Valeurs manquantes :", missing_values)
print("Corrections own_goal :", len(changed_ids))
print("Autres caracteristiques modifiees : 0")
print("Chevauchement TRAIN : 0")
print("Chevauchement HOLDOUT : 0")

print("\n=== MATCHS CORRIGES ===")
for mid in changed_ids:
    print(mid)

print("\n=== STATUT ===")
print("Labels : NON CONSULTES")
print("Predictions : NON CALCULEES")
print("Exclusions de tous les jeux proteges : NON CERTIFIEES")
print("Disponibilite temporelle complete : NON CERTIFIEE")
print("Production : NON AUTORISEE")
print("Aucun fichier modifie")
print("ETAPE 386 TERMINEE")
