import csv
import os
from pathlib import Path

ROOT = Path("outputs")

PILOT = ROOT / (
    "v4_research/v41_pilot_plan_298/"
    "PILOT12_METADATA_ONLY.csv"
)
PLAN = ROOT / (
    "v4_research/v41_chronological_plan_297/"
    "MATCHES_SELECTED_CHRONOLOGICALLY.csv"
)

PROTECTED = {
    "V3I_BATCH1_279": ROOT / (
        "v3i_unseen_holdout_500/batch1_frozen_279/"
        "V3I_BATCH1_279_IDS.csv"
    ),
    "V3I_BATCH2_221": ROOT / (
        "v3i_unseen_holdout_500/batch2_frozen_221/"
        "V3I_BATCH2_221_READY_BLIND.csv"
    ),
    "V3I_HOLDOUT500": ROOT / (
        "v3i_unseen_holdout_500/"
        "V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv"
    ),
    "V3I_RESERVE168": ROOT / (
        "v3i_unseen_holdout_500/reserve168_final_v3c_only/"
        "reserve168_predictions_v3c_final.csv"
    ),
    "V3C_EXTERNAL_TEST": ROOT / (
        "v3c_external_test/"
        "v3c_scored_snapshots_full.csv"
    ),
    "V4_VALIDATION154": ROOT / (
        "v4_research/V4_CHRONO_VALIDATION_V1.csv"
    ),
    "FIRST_HALF_EXTERNAL_HOLDOUT": ROOT / (
        "first_half_37/"
        "v3_draw_with_goals_historical_external_holdout/"
        "first_half_37_v3_historical_external_listing.csv"
    ),
}

def read_ids(path):
    if not path.is_file():
        raise RuntimeError(f"Source absente : {path}")

    if "test_futur" in str(path).lower():
        raise RuntimeError("Acces TEST_FUTUR interdit")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        normalized = [x.strip().lower() for x in header]

        positions = [
            normalized.index(name)
            for name in ("match_id", "fixture_id", "matchid", "id")
            if name in normalized
        ]

        if not positions:
            raise RuntimeError(
                f"Colonne ID introuvable : {path}"
            )

        pos = positions[0]
        ids = set()

        for row in reader:
            if pos < len(row):
                value = row[pos].strip()
                if value:
                    ids.add(value)

        return ids

print("=" * 72)
print("ETAPE 322 - CONTROLE ID-ONLY DES EXCLUSIONS V4.1")
print("=" * 72)

pilot = read_ids(PILOT)
plan = read_ids(PLAN)

assert len(pilot) == 12
assert len(plan) == 227
assert pilot <= plan

print("Pilote :", len(pilot))
print("Plan :", len(plan))

if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
    raise SystemExit(
        "\nSTOP : lecture des fichiers proteges NON autorisee.\n"
        "Pour autoriser uniquement le controle des identifiants :\n"
        "V41_AUTHORISE_ID_ONLY=OUI python3 "
        "v41_protected_id_gate_322.py"
    )

overlap_pilot = set()
overlap_plan = set()

print("\n=== SOURCES PROTEGEES ===")

for name, path in PROTECTED.items():
    ids = read_ids(path)

    collisions_pilot = ids & pilot
    collisions_plan = ids & plan

    overlap_pilot.update(collisions_pilot)
    overlap_plan.update(collisions_plan)

    print(
        name,
        "| IDs :", len(ids),
        "| collisions pilote :", len(collisions_pilot),
        "| collisions plan :", len(collisions_plan),
    )

print("\n=== DECISION ===")
print("Collisions uniques pilote :", len(overlap_pilot))
print("Collisions uniques plan :", len(overlap_plan))

if not overlap_pilot and not overlap_plan:
    print("Controle ID-only des sept sources : PASS")
else:
    print("Controle ID-only : FAIL")
    print("Les candidats en collision restent exclus.")

print("\nAucun ID protege affiche ou exporte.")
print("Aucun label utilise.")
print("Aucun modele entraine.")
print("Aucune collecte.")
print("TEST_FUTUR non consulte.")
print("V3-C inchange.")
print(
    "Ce controle ne certifie pas les partitions "
    "protegees inconnues ou absentes."
)
print("ETAPE 322 TERMINEE.")
