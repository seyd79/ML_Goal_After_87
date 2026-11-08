import csv
import hashlib
import json
from pathlib import Path

BASE = Path("outputs/v4_research")
SRC = BASE / "v41_validation_predictions_387"
PRED = SRC / "V41_A2_VALIDATION46_PREDICTIONS_BLIND.csv"
MANIFEST = SRC / "V41_A2_VALIDATION46_PREDICTIONS_FREEZE.json"

OUT = BASE / "v41_threshold_challenger_389b"
FILE = OUT / "V41_A2_THRESHOLD050_BLIND.csv"
REPORT = OUT / "V41_A2_THRESHOLD050_FREEZE.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if OUT.exists():
    raise RuntimeError("STOP : scenario deja fige")

with MANIFEST.open(encoding="utf-8") as f:
    source = json.load(f)

assert sha(PRED) == source["predictions_sha256"]
assert source["candidate"] == "V4.1-A2"
assert source["threshold"] == 0.60
assert source["labels_accessed"] is False
assert source["outcomes_evaluated"] is False

with PRED.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 46
assert len({r["match_id"] for r in rows}) == 46

result = []

for row in rows:
    p = float(row["probability"])
    assert 0 <= p <= 1
    assert int(row["alert"]) == int(p >= 0.60)

    result.append({
        "match_id": row["match_id"],
        "kickoff_utc": row["kickoff_utc"],
        "probability": row["probability"],
        "alert_strict_060": int(p >= 0.60),
        "alert_elargi_050": int(p >= 0.50),
    })

strict = sum(r["alert_strict_060"] for r in result)
elargi = sum(r["alert_elargi_050"] for r in result)

assert strict == 1
assert elargi == 10

OUT.mkdir(parents=True, exist_ok=False)

with FILE.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)

freeze = {
    "candidate": "V4.1-A2",
    "status": "EXPLORATORY_THRESHOLD_CHALLENGER_FROZEN",
    "source_predictions_sha256": sha(PRED),
    "challenger_predictions_sha256": sha(FILE),
    "original_threshold": 0.60,
    "challenger_threshold": 0.50,
    "matches": 46,
    "strict_alerts": strict,
    "challenger_alerts": elargi,
    "chosen_after_unlabeled_validation_distribution_review": True,
    "labels_accessed": False,
    "outcomes_evaluated": False,
    "independent_performance_certified": False,
    "production_authorized": False,
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(freeze, f, indent=2)

print("=" * 60)
print("V4.1-A2 - COMPARAISON DES SEUILS FIGEE")
print("=" * 60)
print("Matchs :", len(result))
print("Alertes seuil 60% :", strict)
print("Alertes seuil 50% :", elargi)
print("Alertes supplementaires :", elargi - strict)
print("Labels consultes : NON")
print("Resultats consultes : NON")
print("Modele modifie : NON")
print("V3-C modifie : NON")
print("SHA256 :", sha(FILE))
print("ETAPE 389B TERMINEE")
