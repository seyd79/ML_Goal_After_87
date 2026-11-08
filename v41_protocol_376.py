import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path("outputs/v4_research/v41_candidate_a_375")
DATA = BASE / "V41_A_TRAIN112_CORE43_PROVISIONAL.csv"
SOURCE_MANIFEST = BASE / "V41_A_CORE43_MANIFEST.json"

OUT_DIR = Path("outputs/v4_research/v41_protocol_376")
OUT_FILE = OUT_DIR / "V41_A_TRAIN_ONLY_FOLDS.csv"
OUT_MANIFEST = OUT_DIR / "V41_A_PROTOCOL_FREEZE.json"

MONTHS = ["2025-10", "2025-11", "2025-12"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if OUT_DIR.exists():
    raise RuntimeError("STOP : protocole 376 deja existant")

with SOURCE_MANIFEST.open(encoding="utf-8") as f:
    manifest = json.load(f)

if sha(DATA) != manifest["output_sha256"]:
    raise RuntimeError("STOP : dataset 375 modifie")

if manifest.get("training_authorized") is not False:
    raise RuntimeError("STOP : statut d'autorisation inattendu")

with DATA.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)

assert len(rows) == 112
assert len({r["match_id"] for r in rows}) == 112
assert len(manifest["features"]) == 43
assert all(c in columns for c in manifest["features"])
assert Counter(r["goal_after_87"] for r in rows) == {
    "0": 74,
    "1": 38,
}

rows.sort(key=lambda r: (r["kickoff_utc"], r["match_id"]))

assignments = []
fold_reports = []

print("=" * 72)
print("ETAPE 376 - PROTOCOLE CHRONOLOGIQUE TRAIN UNIQUEMENT")
print("=" * 72)

for number, month in enumerate(MONTHS, 1):
    train = [
        r for r in rows
        if r["kickoff_utc"][:7] < month
    ]

    test = [
        r for r in rows
        if r["kickoff_utc"][:7] == month
    ]

    if not train or not test:
        raise RuntimeError(f"STOP : fold incomplet {month}")

    if max(r["kickoff_utc"] for r in train) >= min(
        r["kickoff_utc"] for r in test
    ):
        raise RuntimeError("STOP : ordre chronologique invalide")

    train_counts = Counter(r["goal_after_87"] for r in train)
    test_counts = Counter(r["goal_after_87"] for r in test)

    if not all(train_counts[str(i)] > 0 for i in (0, 1)):
        raise RuntimeError("STOP : une classe manque dans TRAIN")

    for role, subset in [
        ("FIT", train),
        ("OUT_OF_FOLD", test),
    ]:
        for row in subset:
            assignments.append({
                "fold": number,
                "role": role,
                "match_id": row["match_id"],
                "kickoff_utc": row["kickoff_utc"],
            })

    fold_reports.append({
        "fold": number,
        "evaluation_month": month,
        "fit_count": len(train),
        "fit_positive": train_counts["1"],
        "oof_count": len(test),
        "oof_positive": test_counts["1"],
    })

    print(
        f"FOLD {number} | mois={month} | "
        f"FIT={len(train)} ({train_counts['1']} positifs) | "
        f"OOF={len(test)} ({test_counts['1']} positifs)"
    )

oof_ids = [
    r["match_id"]
    for r in assignments
    if r["role"] == "OUT_OF_FOLD"
]

if len(oof_ids) != len(set(oof_ids)):
    raise RuntimeError("STOP : matchs evalues plusieurs fois")

OUT_DIR.mkdir(parents=True, exist_ok=False)

with OUT_FILE.open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["fold", "role", "match_id", "kickoff_utc"],
    )
    writer.writeheader()
    writer.writerows(assignments)

protocol = {
    "candidate": "V4.1-A",
    "status": "DESIGN_ONLY_NOT_TRAINED",
    "source_sha256": sha(DATA),
    "fold_file_sha256": sha(OUT_FILE),
    "features": manifest["features"],
    "folds": fold_reports,
    "total_train_matches": 112,
    "unique_oof_matches": len(oof_ids),
    "imputation_rule": "fit_median_inside_each_fold_only",
    "holdout_accessed": False,
    "official_validation_accessed": False,
    "training_authorized": False,
}

with OUT_MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

print("\nMatchs evalues hors apprentissage interne :", len(oof_ids))
print("Nombre de folds :", len(fold_reports))
print("Fichier :", OUT_FILE)
print("Manifest :", OUT_MANIFEST)
print("Aucun modele entraine.")
print("Validation officielle / HOLDOUT : NON CONSULTES")
print("ETAPE 376 TERMINEE.")
