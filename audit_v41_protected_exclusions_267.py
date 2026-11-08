import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("outputs/v4_research/v41_context_pool_266")
SOURCE = ROOT / "CONTEXT_POOL1131_PROVISIONAL.csv"

PROTECTED = {
    "V3I_BATCH1_279":
        Path("outputs/v3i_unseen_holdout_500/"
             "batch1_frozen_279/V3I_BATCH1_279_IDS.csv"),

    "V3I_BATCH2_221":
        Path("outputs/v3i_unseen_holdout_500/"
             "batch2_frozen_221/V3I_BATCH2_221_READY_BLIND.csv"),

    "V3I_HOLDOUT500":
        Path("outputs/v3i_unseen_holdout_500/"
             "V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv"),

    "V3I_RESERVE168":
        Path("outputs/v3i_unseen_holdout_500/"
             "reserve168_final_v3c_only/"
             "reserve168_predictions_v3c_final.csv"),

    "V3C_EXTERNAL_TEST":
        Path("outputs/v3c_external_test/"
             "v3c_scored_snapshots_full.csv"),

    "V4_VALIDATION154":
        Path("outputs/v4_research/"
             "V4_CHRONO_VALIDATION_V1.csv"),

    "FIRST_HALF_EXTERNAL_HOLDOUT":
        Path("outputs/first_half_37/"
             "v3_draw_with_goals_historical_external_holdout/"
             "first_half_37_v3_historical_external_listing.csv"),
}

OLD_SNAPSHOTS = Path(
    "outputs/history/live_feature_snapshots_all703_labeled_v3.csv"
)

def ids_only(path):
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        id_column = next(
            (c for c in fields
             if c.strip().lower() in
             ("match_id", "fixture_id", "id", "matchid")),
            None
        )

        if id_column is None:
            raise ValueError(
                "Colonne ID absente : " + str(path)
                + " | colonnes : " + str(fields[:15])
            )

        # Lecture de la colonne ID uniquement.
        return {
            str(row[id_column]).strip()
            for row in reader
            if row.get(id_column)
            and str(row[id_column]).strip()
        }

def load_pool(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if "match_id" not in fields:
        raise ValueError("Pool sans colonne match_id")

    return fields, rows

print("=" * 76)
print("ETAPE 267 - EXCLUSION DES ANCIENS ENSEMBLES PROTEGES")
print("=" * 76)

fields, rows = load_pool(SOURCE)
pool_ids = {r["match_id"] for r in rows}

if len(rows) != 1131 or len(pool_ids) != 1131:
    raise SystemExit("Population 1131 ou unicite incorrecte")

protected_ids = set()
source_stats = {}

print("\n=== SOURCES PROTEGEES ===")

for name, path in PROTECTED.items():
    ids = ids_only(path)
    overlap = ids & pool_ids

    protected_ids.update(ids)
    source_stats[name] = {
        "ids_uniques": len(ids),
        "chevauchement_pool": len(overlap),
    }

    print(
        f"{name:29s} "
        f"IDs={len(ids):5d} "
        f"dans_pool={len(overlap):5d}"
    )

standard = [
    r for r in rows
    if r["match_id"] not in protected_ids
]

print("\n=== ANCIENS SNAPSHOTS (OPTION STRICTE) ===")

old_ids = ids_only(OLD_SNAPSHOTS)
old_overlap = old_ids & pool_ids

strict_excluded = protected_ids | old_ids

strict = [
    r for r in rows
    if r["match_id"] not in strict_excluded
]

print("IDs anciens snapshots :", len(old_ids))
print("Chevauchement pool :", len(old_overlap))

def positive_count(items):
    return sum(
        int(float(r["goal_after_87"]))
        for r in items
    )

def summary(name, items):
    counts = Counter(
        r.get("source_research", "")
        for r in items
    )
    print(
        f"{name:20s} "
        f"matchs={len(items):4d} "
        f"positifs={positive_count(items):4d} "
        f"origines={dict(counts)}"
    )

print("\n=== POPULATIONS ===")
summary("POOL INITIAL", rows)
summary("POOL STANDARD", standard)
summary("POOL STRICT", strict)

def save(items, filename, status):
    path = ROOT / filename
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()

        for original in items:
            row = dict(original)
            row["admission_status"] = status
            writer.writerow(row)

    return path

standard_path = save(
    standard,
    "CONTEXT_POOL_STANDARD_PROVISIONAL_V1.csv",
    "PROVISIONAL_KNOWN_PROTECTED_EXCLUDED"
)

strict_path = save(
    strict,
    "CONTEXT_POOL_STRICT_PROVISIONAL_V1.csv",
    "PROVISIONAL_STRICT_OLD_SNAPSHOTS_EXCLUDED"
)

report = {
    "source_pool": str(SOURCE),
    "original_n": len(rows),
    "known_protected_sources": source_stats,
    "known_protected_union_overlap":
        len(pool_ids & protected_ids),
    "old_snapshots_unique_ids": len(old_ids),
    "old_snapshots_overlap": len(old_overlap),
    "standard_n": len(standard),
    "standard_positive": positive_count(standard),
    "strict_n": len(strict),
    "strict_positive": positive_count(strict),
    "test_futur_accessed": False,
    "labels_from_protected_files_read": False,
    "certified": False,
}

report_path = ROOT / "PROTECTED_EXCLUSIONS_AUDIT_V1.json"

with report_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n=== FICHIERS ===")
print("STANDARD :", standard_path)
print("STRICT   :", strict_path)
print("AUDIT    :", report_path)

print("\nATTENTION : exclusions limitees aux fichiers references.")
print("TEST_FUTUR et autres partitions non cartographiees")
print("ne sont pas certifies absents.")
print("Aucun label protege consulte.")
print("Aucun entrainement. Aucun appel API.")
print("V3-C inchange.")
print("ETAPE 267 TERMINEE.")
