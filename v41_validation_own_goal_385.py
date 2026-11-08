import csv
import hashlib
import json
import os
from pathlib import Path

import collect_historical_dataset_v2 as collector
import rebuild_clean_prematch_v2 as prematch
import v41_live87_own_goal_364 as corrected
from v41_targeted_collector_328 import load_live87

BASE = Path("outputs/v4_research")
DATA = BASE / "v41_dataset_preparation_343/V41_CORE49_197_QUARANTINE_NO_LABEL.csv"
DATA_FREEZE = BASE / "v41_dataset_freeze_344/V41_CORE49_197_QUARANTINE_FREEZE.json"
SPLIT = BASE / "v41_chronological_split_345/V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
SPLIT_FREEZE = BASE / "v41_split_freeze_346/V41_SPLIT_FREEZE_PROVISIONAL.json"
CANDIDATE = BASE / "v41_candidate_freeze_382/V41_A2_EXPLORATORY_FREEZE.json"
CORRECTOR = Path("v41_live87_own_goal_364.py")

OUT = BASE / "v41_validation_own_goal_385"
CHECKPOINTS = OUT / "checkpoints"
CSV_OUT = OUT / "V41_A2_VALIDATION46_FEATURES_BLIND.csv"
REPORT = OUT / "V41_A2_VALIDATION_FEATURES_MANIFEST.json"

META = ["match_id", "competition_id", "season_id", "kickoff_utc"]
SCORE = ["score_87_home", "score_87_away"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def number(value):
    return float(value)

print("=" * 72)
print("ETAPE 385 - CORRECTION AVEUGLE VALIDATION V4.1-A2")
print("=" * 72)

if os.getenv("V41_AUTORISE_AUDIT_VALIDATION") != "OUI":
    raise RuntimeError("STOP : autorisation audit manquante")

assert sha(DATA) == load_json(DATA_FREEZE)["sha256"]
assert sha(SPLIT) == load_json(SPLIT_FREEZE)["split_sha256"]

candidate = load_json(CANDIDATE)
assert candidate["candidate"] == "V4.1-A2"
assert candidate["threshold"] == 0.60
assert candidate["regularization_C"] == 0.1

features = candidate["features"]
assert len(features) == len(set(features)) == 10
assert set(SCORE).issubset(features)

data = read_csv(DATA)
splits = read_csv(SPLIT)

assert len(data) == len(splits) == 197

by_id = {r["match_id"]: r for r in data}
assert len(by_id) == 197
assert "goal_after_87" not in data[0]

validation_ids = [
    r["match_id"] for r in splits
    if r["split"] == "VALIDATION_PROVISIONAL"
]
assert len(validation_ids) == len(set(validation_ids)) == 46
assert set(validation_ids).issubset(by_id)

data_hash = sha(DATA)
split_hash = sha(SPLIT)
corrector_hash = sha(CORRECTOR)

if CSV_OUT.exists() or REPORT.exists():
    raise RuntimeError("STOP : sortie 385 deja finalisee")

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

lab = prematch.load_lab()
client = prematch.build_rate_limited_client(lab)
original = load_live87()

for i, mid in enumerate(validation_ids, 1):
    checkpoint = CHECKPOINTS / (mid + ".json")

    if checkpoint.exists():
        record = load_json(checkpoint)
        assert record["match_id"] == mid
        assert record["source_sha256"] == data_hash
        assert record["corrector_sha256"] == corrector_hash
        print(f"[{i}/46] DEJA VERIFIE : {mid}")
        continue

    detail = prematch.get_match_detail(lab, client, mid)
    if not detail or str(lab.extract_match_id(detail)) != mid:
        raise RuntimeError("STOP : detail invalide " + mid)

    timeline, shots = collector.fetch_timeline_and_shotmap(
        original, client, mid
    )

    before, _ = collector.reconstruct_live87(
        original, detail, timeline, shots
    )
    after, _ = collector.reconstruct_live87(
        corrected, detail, timeline, shots
    )

    frozen = by_id[mid]

    for field in SCORE:
        if before.get(field) is None or after.get(field) is None:
            raise RuntimeError(
                "STOP : score incomplet " + mid + " / " + field
            )
        if number(frozen[field]) != number(before[field]):
            raise RuntimeError(
                "STOP : reconstruction differente du gel "
                + mid + " / " + field
            )

    old_total = sum(number(before[f]) for f in SCORE)
    new_total = sum(number(after[f]) for f in SCORE)

    if old_total != new_total:
        raise RuntimeError("STOP : nombre de buts modifie " + mid)

    own_goals = sum(
        1 for event in timeline
        if isinstance(event, dict)
        and str(event.get("type", "")).strip().lower() == "own_goal"
        and corrected.extract_event_minute(event) is not None
        and corrected.extract_event_minute(event) <= 87
    )

    changed = any(
        number(before[f]) != number(after[f])
        for f in SCORE
    )

    if changed and own_goals == 0:
        raise RuntimeError(
            "STOP : correction sans own_goal explicite " + mid
        )

    record = {
        "match_id": mid,
        "source_sha256": data_hash,
        "corrector_sha256": corrector_hash,
        "score_before": [before[f] for f in SCORE],
        "score_after": [after[f] for f in SCORE],
        "own_goals_le87": own_goals,
        "score_changed": changed,
        "status": "FEATURE_AUDIT_ONLY",
    }

    with checkpoint.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(
        f"[{i}/46] AUDIT : {mid} | "
        f"own_goals={own_goals} | "
        f"correction={'OUI' if changed else 'NON'}"
    )

# Assembler uniquement après les 46 audits.
output_rows = []
changed_ids = []

for mid in validation_ids:
    record = load_json(CHECKPOINTS / (mid + ".json"))

    assert record["source_sha256"] == data_hash
    assert record["corrector_sha256"] == corrector_hash

    row = by_id[mid]
    result = {k: row[k] for k in META + features}

    for index, field in enumerate(SCORE):
        result[field] = record["score_after"][index]

    h = number(result["score_87_home"])
    a = number(result["score_87_away"])

    if h < 0 or a < 0:
        raise RuntimeError("STOP : score negatif " + mid)

    if record["score_changed"]:
        changed_ids.append(mid)

    output_rows.append(result)

assert len(output_rows) == 46
assert len({r["match_id"] for r in output_rows}) == 46

OUT.mkdir(parents=True, exist_ok=True)

with CSV_OUT.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=META + features)
    writer.writeheader()
    writer.writerows(output_rows)

manifest = {
    "candidate": "V4.1-A2",
    "status": "VALIDATION_FEATURES_PROVISIONAL_BLIND",
    "source_sha256": data_hash,
    "split_sha256": split_hash,
    "candidate_freeze_sha256": sha(CANDIDATE),
    "corrector_sha256": corrector_hash,
    "output_sha256": sha(CSV_OUT),
    "validation_matches": 46,
    "feature_count": 10,
    "features": features,
    "corrected_match_ids": changed_ids,
    "labels_calculated": False,
    "model_predictions_calculated": False,
    "independent_validation_completed": False,
    "protected_sets_fully_certified": False,
    "production_authorized": False,
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 72)
print("BILAN ETAPE 385")
print("=" * 72)
print("Matchs validation prepares :", len(output_rows))
print("Matchs avec correction own_goal :", len(changed_ids))
print("Features :", len(features))
print("Labels calcules : NON")
print("Predictions calculees : NON")
print("Holdout utilise : NON")
print("Fichier :", CSV_OUT)
print("Manifeste :", REPORT)
print("ETAPE 385 TERMINEE")
