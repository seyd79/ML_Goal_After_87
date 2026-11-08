import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(".")
BASE = ROOT / "outputs/v4_research"

SPLIT = (
    BASE / "v41_chronological_split_345"
    / "V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
)

SPLIT_FREEZE = (
    BASE / "v41_split_freeze_346"
    / "V41_SPLIT_FREEZE_PROVISIONAL.json"
)

SEARCH_DIRS = [
    ROOT / "outputs/v3i_unseen_holdout_500",
    ROOT / "outputs/v3d_pressure_experiment",
    ROOT / "outputs/first_half_37/final_holdout_v1",
    ROOT / "outputs/first_half_37/v2_final_holdout",
    ROOT / "outputs/pressure_historical/pressure_v2_dataset_join",
]

ID_HEADERS = {
    "matchid", "fixtureid", "eventid",
    "gameid", "matchkey", "fixturekey",
    "idmatch", "idfixture"
}

def canonical(value):
    match = re.fullmatch(
        r"(?:mt[_-]?)?([0-9]{5,15})(?:\.0)?",
        str(value).strip(),
        flags=re.IGNORECASE
    )
    return match.group(1) if match else None

def norm(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_ids(path):
    with path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        if not headers:
            return None, "EMPTY", 0, 0

        normalized = [norm(h) for h in headers]

        positions = [
            i for i, header in enumerate(normalized)
            if header in ID_HEADERS
        ]

        if len(positions) != 1:
            return None, "AMBIGUOUS_OR_MISSING_ID", 0, 0

        pos = positions[0]
        ids = set()
        invalid = 0
        count = 0

        for row in reader:
            count += 1
            mid = (
                canonical(row[pos])
                if pos < len(row)
                else None
            )
            if mid is None:
                invalid += 1
            else:
                ids.add(mid)

        return ids, headers[pos], invalid, count

print("=" * 72)
print("ETAPE 391 - AUDIT COMPLEMENTAIRE DES IDS PROTEGES")
print("=" * 72)

import json

with SPLIT_FREEZE.open(encoding="utf-8") as f:
    frozen = json.load(f)

assert sha(SPLIT) == frozen["split_sha256"]

partitions = {
    "TRAIN_PROVISIONAL": set(),
    "VALIDATION_PROVISIONAL": set(),
    "HOLDOUT_V41_PROVISIONAL": set(),
}

with SPLIT.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        role = row["split"]
        if role not in partitions:
            raise RuntimeError("STOP : partition inconnue")

        mid = canonical(row["match_id"])
        if mid is None:
            raise RuntimeError("STOP : ID V4.1 invalide")

        if any(mid in ids for ids in partitions.values()):
            raise RuntimeError("STOP : doublon V4.1")

        partitions[role].add(mid)

assert len(partitions["TRAIN_PROVISIONAL"]) == 112
assert len(partitions["VALIDATION_PROVISIONAL"]) == 46
assert len(partitions["HOLDOUT_V41_PROVISIONAL"]) == 39

# Identifie les fichiers potentiellement proteges.
paths = set()

for directory in SEARCH_DIRS:
    if not directory.is_dir():
        print("REPERTOIRE ABSENT :", directory)
        continue

    for path in directory.rglob("*.csv"):
        name = path.name.lower()

        if any(term in name for term in (
            "holdout",
            "test_futur",
            "targets",
        )):
            paths.add(path)

# Ajouter explicitement quelques sources importantes.
explicit = [
    ROOT / "outputs/v3i_unseen_holdout_500"
    / "V3I_UNSEEN_HOLDOUT_500_TARGETS.csv",

    ROOT / "outputs/first_half_37/v2_final_holdout"
    / "evaluation/first_half_37_v2_final_holdout_predictions.csv",

    ROOT / "outputs/first_half_37/v2_final_holdout"
    / "evaluation/first_half_37_v2_final_holdout_predictions.csv",

    ROOT / "outputs/first_half_37/v2_final_holdout"
    / "first_half_37_v2_final_holdout_collection_audit.csv",
]

for path in explicit:
    if path.is_file():
        paths.add(path)

checked = 0
incomplete = []
overlaps = []
total_identifiers = set()

print("\n=== FICHIERS EXAMINES ===")

for path in sorted(paths, key=str):
    relative = path.as_posix()

    try:
        ids, column, invalid, count = read_ids(path)
    except (OSError, csv.Error, UnicodeError) as exc:
        print("INACCESSIBLE :", relative, type(exc).__name__)
        incomplete.append(relative)
        continue

    if ids is None or invalid > 0 or count == 0:
        print(
            "INCOMPLET :", relative,
            "| colonne :", column,
            "| lignes :", count,
            "| invalides :", invalid
        )
        incomplete.append(relative)
        continue

    checked += 1
    total_identifiers.update(ids)

    print(
        "OK :", relative,
        "| colonne :", column,
        "| IDs :", len(ids)
    )

    for role, group in partitions.items():
        common = sorted(ids.intersection(group))

        if common:
            overlaps.append((relative, role, common))
            print(
                "  CHEVAUCHEMENT",
                role,
                ":",
                len(common)
            )
            print(
                "  MATCHS :",
                ", ".join(
                    "mt_" + mid for mid in common[:20]
                )
            )

print("\n" + "=" * 72)
print("BILAN ETAPE 391")
print("=" * 72)

print("Fichiers candidats :", len(paths))
print("Fichiers controles :", checked)
print("Fichiers a approfondir :", len(incomplete))
print("IDs distincts couverts :", len(total_identifiers))
print("Source/partition avec chevauchement :", len(overlaps))

if overlaps:
    print("\nALERTE : CHEVAUCHEMENTS DETECTES")
else:
    print("\nAucun chevauchement dans les fichiers exploitables.")

if incomplete:
    print("\nFICHIERS NON CERTIFIES :")
    for path in incomplete:
        print(" -", path)

print("\nAudit toujours non exhaustif.")
print("JSON et autres formats non certifies.")
print("Labels de validation : NON CONSULTES")
print("Predictions V4.1 : INCHANGEES")
print("Seuil 0.50 : INCHANGE")
print("Seuil 0.60 : INCHANGE")
print("Aucun fichier ecrit.")
print("Aucun entrainement.")
print("ETAPE 391 TERMINEE")
