import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path("outputs/v4_research")

SOURCE_DIR = BASE / "v41_train_dataset_358"
SOURCE = SOURCE_DIR / "V41_TRAIN112_LABELED_PROVISIONAL.csv"
SOURCE_REPORT = SOURCE_DIR / "V41_TRAIN112_PROVENANCE.json"

AUDITS = BASE / "v41_own_goal_audit_365/checkpoints"
CORRECTOR = Path("v41_live87_own_goal_364.py")

DEST = BASE / "v41_train_dataset_corrected_366"
OUTPUT = DEST / "V41_TRAIN112_OWN_GOAL_CORRECTED.csv"
MANIFEST = DEST / "V41_OWN_GOAL_CORRECTION_MANIFEST.json"

FIELDS = {
    "score_87_home",
    "score_87_away",
    "goal_diff_87",
    "is_draw_87",
    "leading_team_87",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def equivalent(a, b):
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)

if DEST.exists():
    raise RuntimeError("STOP : dossier 366 deja existant")

with SOURCE_REPORT.open(encoding="utf-8") as f:
    source_report = json.load(f)

if sha(SOURCE) != source_report["train_dataset_sha256"]:
    raise RuntimeError("STOP : dataset source modifie")

if not CORRECTOR.is_file():
    raise RuntimeError("STOP : correctif 364 absent")

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)

if not columns or not FIELDS.issubset(columns):
    raise RuntimeError("STOP : colonnes de score manquantes")

if len(rows) != 112:
    raise RuntimeError("STOP : 112 matchs attendus")

ids = [r["match_id"] for r in rows]

if len(set(ids)) != 112:
    raise RuntimeError("STOP : IDs dupliques")

if Counter(r["goal_after_87"] for r in rows) != {"0": 74, "1": 38}:
    raise RuntimeError("STOP : labels inattendus")

audit_files = list(AUDITS.glob("mt_*.json"))

if len(audit_files) != 112 or {p.stem for p in audit_files} != set(ids):
    raise RuntimeError("STOP : audit incomplet ou supplementaire")

changes = []
checkpoint_hashes = {}

for row in rows:
    mid = row["match_id"]
    path = AUDITS / (mid + ".json")

    with path.open(encoding="utf-8") as f:
        audit = json.load(f)

    if (
        audit.get("match_id") != mid
        or audit.get("split") != "TRAIN_PROVISIONAL"
        or audit.get("status") != "AUDIT_ONLY_NOT_CERTIFIED"
        or audit.get("label_original") != int(row["goal_after_87"])
        or audit.get("frozen_vs_original_mismatch") != []
        or audit.get("score_monotone") is not True
        or audit.get("label_score_agrees") is not True
    ):
        raise RuntimeError("STOP : audit invalide : " + mid)

    modifications = audit.get("changed_fields")

    if not isinstance(modifications, dict):
        raise RuntimeError("STOP : format corrections invalide")

    if not set(modifications).issubset(FIELDS):
        raise RuntimeError("STOP : correction hors caracteristiques autorisees")

    if modifications and not audit.get("own_goals_le87"):
        raise RuntimeError("STOP : correction sans own_goal : " + mid)

    modified_fields = []

    for field, detail in modifications.items():
        old = detail.get("original")
        new = detail.get("corrected")

        if not equivalent(row[field], old):
            raise RuntimeError(
                f"STOP : divergence source {mid} / {field}"
            )

        if new is None:
            raise RuntimeError(
                f"STOP : valeur corrigee absente {mid} / {field}"
            )

        if equivalent(old, new):
            raise RuntimeError(
                f"STOP : fausse modification {mid} / {field}"
            )

        row[field] = str(new)
        modified_fields.append(field)

    score = audit.get("corrected_score_87")

    if not isinstance(score, list) or len(score) != 2:
        raise RuntimeError("STOP : score corrige absent : " + mid)

    if not (
        equivalent(row["score_87_home"], score[0])
        and equivalent(row["score_87_away"], score[1])
    ):
        raise RuntimeError("STOP : score corrige incoherent : " + mid)

    h = float(row["score_87_home"])
    a = float(row["score_87_away"])

    expected = {
        "goal_diff_87": abs(h - a),
        "is_draw_87": int(h == a),
        "leading_team_87": 0 if h > a else (1 if a > h else 2),
    }

    for field, value in expected.items():
        if not equivalent(row[field], value):
            raise RuntimeError(
                f"STOP : incoherence derivee {mid} / {field}"
            )

    if modified_fields:
        changes.append({
            "match_id": mid,
            "fields": sorted(modified_fields),
        })

    checkpoint_hashes[mid] = sha(path)

if len(changes) != 10:
    raise RuntimeError(
        f"STOP : 10 matchs corriges attendus, {len(changes)} trouves"
    )

DEST.mkdir(parents=True, exist_ok=False)

with OUTPUT.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "status": "PROVISIONAL_RESEARCH_QUARANTINE",
    "source_file_sha256": sha(SOURCE),
    "corrector_364_sha256": sha(CORRECTOR),
    "checkpoint_sha256_by_match": checkpoint_hashes,
    "corrected_file_sha256": sha(OUTPUT),
    "rows": 112,
    "corrected_matches": len(changes),
    "corrections": changes,
    "positive_labels": 38,
    "negative_labels": 74,
    "labels_modified": False,
    "validation_accessed": False,
    "holdout_accessed": False,
    "independent_label_certification": False,
    "protected_sets_fully_certified": False,
    "training_authorized": False,
}

with MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("ETAPE 366 - TRAIN V4.1 CORRIGE")
print("=" * 70)
print("Matchs :", len(rows))
print("Matchs corriges :", len(changes))
print("Matchs inchanges :", len(rows) - len(changes))
print("Labels positifs : 38")
print("Labels negatifs : 74")
print("Labels modifies : NON")
print("SHA256 :", sha(OUTPUT))
print("Dataset :", OUTPUT)
print("Manifest :", MANIFEST)
print("Source originale : INCHANGEE")
print("Validation / HOLDOUT : NON CONSULTES")
print("Entrainement : NON AUTORISE")
print("Aucun appel API.")
print("ETAPE 366 TERMINEE.")
