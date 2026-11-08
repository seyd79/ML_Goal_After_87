import csv
import hashlib
import re
from pathlib import Path

BASE = Path("outputs/v4_research")

SPLIT = (
    BASE / "v41_chronological_split_345"
    / "V41_CHRONO_SPLIT_197_PROVISIONAL.csv"
)

SOURCES = {
    "V3C_545_A_TEST_FUTUR":
        "outputs/pressure_historical/pressure_v2_dataset_join/"
        "V3C_545_A_TEST_FUTUR_RESEARCH_V1.csv",

    "V3C_545_B_TEST_FUTUR":
        "outputs/pressure_historical/pressure_v2_dataset_join/"
        "V3C_545_B_TEST_FUTUR_RESEARCH_V1.csv",

    "V3I_500_BLIND":
        "outputs/v3i_unseen_holdout_500/"
        "V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv",

    "V3I_500_READY":
        "outputs/v3i_unseen_holdout_500/"
        "V3I_UNSEEN_HOLDOUT_500_V3C_READY.csv",

    "FIRST_HALF37_FINAL":
        "outputs/first_half_37/final_holdout_v1/"
        "first_half_37_final_holdout_predictions.csv",

    "FIRST_HALF37_V2_LISTED":
        "outputs/first_half_37/v2_final_holdout/"
        "first_half_37_v2_final_holdout_listed_matches.csv",

    "FIRST_HALF37_V2_FINAL":
        "outputs/first_half_37/v2_final_holdout/"
        "First_Half_37_V2_FINAL_HOLDOUT.csv",

    "V3D_HOLDOUT":
        "outputs/v3d_pressure_experiment/"
        "holdout_predictions.csv",
}

PREFERRED_COLUMNS = [
    "matchid",
    "idmatch",
    "fixtureid",
    "idfixture",
    "eventid",
    "gameid",
    "matchkey",
    "fixturekey",
]

def canonical(value):
    value = str(value).strip()

    match = re.fullmatch(
        r"(?:mt[_-]?)?([0-9]{5,15})(?:\.0)?",
        value,
        flags=re.IGNORECASE,
    )

    return match.group(1) if match else None

def normalized_header(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())

def read_ids(path):
    with path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        if not headers:
            return set(), "EMPTY", 0

        normalized = [
            normalized_header(h) for h in headers
        ]

        selected = None
        for name in PREFERRED_COLUMNS:
            if name in normalized:
                selected = normalized.index(name)
                break

        if selected is None:
            if "id" in normalized:
                selected = normalized.index("id")
            else:
                return set(), "NO_ID_COLUMN", 0

        ids = set()
        invalid = 0

        for row in reader:
            if selected >= len(row):
                invalid += 1
                continue

            mid = canonical(row[selected])

            if mid is None:
                invalid += 1
            else:
                ids.add(mid)

        return ids, headers[selected], invalid

print("=" * 72)
print("ETAPE 390 - AUDIT IDs PROTEGES V4.1")
print("=" * 72)

if not SPLIT.is_file():
    raise RuntimeError("STOP : split 345 introuvable")

partitions = {
    "TRAIN": set(),
    "VALIDATION": set(),
    "HOLDOUT": set(),
}

roles = {
    "TRAIN_PROVISIONAL": "TRAIN",
    "VALIDATION_PROVISIONAL": "VALIDATION",
    "HOLDOUT_V41_PROVISIONAL": "HOLDOUT",
}

with SPLIT.open(
    encoding="utf-8-sig",
    newline=""
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        role = roles.get(row["split"])

        if role is None:
            raise RuntimeError("STOP : role inconnu")

        mid = canonical(row["match_id"])

        if mid is None:
            raise RuntimeError("STOP : ID V4.1 invalide")

        if any(mid in group for group in partitions.values()):
            raise RuntimeError("STOP : doublon partitions V4.1")

        partitions[role].add(mid)

assert len(partitions["TRAIN"]) == 112
assert len(partitions["VALIDATION"]) == 46
assert len(partitions["HOLDOUT"]) == 39

print("\n=== ENSEMBLES V4.1 ===")
for name, ids in partitions.items():
    print(f"{name:12s} : {len(ids)} matchs")

print("\n=== COMPARAISON DES IDENTIFIANTS ===")

incomplete = []
overlaps = []

for name, filename in SOURCES.items():
    path = Path(filename)

    print("\nSOURCE :", name)

    if not path.is_file():
        print("STATUT : FICHIER ABSENT")
        incomplete.append(name)
        continue

    try:
        ids, column, invalid = read_ids(path)
    except (OSError, csv.Error) as e:
        print("STATUT : ERREUR DE LECTURE", type(e).__name__)
        incomplete.append(name)
        continue

    print("Colonne ID :", column)
    print("IDs exploitables :", len(ids))
    print("Lignes ID invalides :", invalid)

    if (
        column in ("EMPTY", "NO_ID_COLUMN")
        or len(ids) == 0
        or invalid > 0
    ):
        print("STATUT : CONTROLE INCOMPLET")
        incomplete.append(name)
        continue

    for group_name, group_ids in partitions.items():
        intersection = sorted(ids & group_ids)

        print(
            f"Chevauchement {group_name:10s} : "
            f"{len(intersection)}"
        )

        if intersection:
            overlaps.append((name, group_name, intersection))
            print(
                "IDs communs :",
                ", ".join("mt_" + x for x in intersection[:20])
            )

    if column.lower() == "id":
        print(
            "ATTENTION : colonne generique 'id', "
            "semantique a confirmer"
        )
        incomplete.append(name)

print("\n" + "=" * 72)
print("BILAN ETAPE 390")
print("=" * 72)

print("Sources examinees :", len(SOURCES))
print("Sources a approfondir :", len(set(incomplete)))
print("Couples source/partition avec chevauchement :", len(overlaps))

if overlaps:
    print("\nCHEVAUCHEMENTS DETECTES")
    for name, role, ids in overlaps:
        print(f"{name} / {role} : {len(ids)}")
else:
    print("\nAucun chevauchement detecte dans les IDs exploitables.")

if incomplete:
    print("\nCONTROLES INCOMPLETS")
    for name in sorted(set(incomplete)):
        print(name)

print("\nIMPORTANT : audit partiel, non exhaustif.")
print("Les autres jeux proteges restent a verifier.")
print("Aucun label exploite.")
print("Aucune prediction modifiee.")
print("Aucun fichier ecrit.")
print("Aucun modele entraine.")
print("ETAPE 390 TERMINEE")
