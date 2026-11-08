import csv
import os
from collections import Counter
from pathlib import Path

ROOT = Path("outputs")

PLAN = ROOT / (
    "v4_research/v41_chronological_plan_297/"
    "MATCHES_SELECTED_CHRONOLOGICALLY.csv"
)
PILOT = ROOT / (
    "v4_research/v41_pilot_plan_298/"
    "PILOT12_METADATA_ONLY.csv"
)
DEST = ROOT / "v4_research/v41_clearance_323"

PROTECTED = [
    ROOT / "v3i_unseen_holdout_500/batch1_frozen_279/V3I_BATCH1_279_IDS.csv",
    ROOT / "v3i_unseen_holdout_500/batch2_frozen_221/V3I_BATCH2_221_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv",
    ROOT / "v3i_unseen_holdout_500/reserve168_final_v3c_only/reserve168_predictions_v3c_final.csv",
    ROOT / "v3c_external_test/v3c_scored_snapshots_full.csv",
    ROOT / "v4_research/V4_CHRONO_VALIDATION_V1.csv",
    ROOT / "first_half_37/v3_draw_with_goals_historical_external_holdout/first_half_37_v3_historical_external_listing.csv",
]

def metadata(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        if "match_id" not in fields:
            raise RuntimeError(f"match_id absent : {path}")
        rows = list(r)
    return fields, rows

def ids_only(path):
    if "test_futur" in str(path).lower():
        raise RuntimeError("TEST_FUTUR interdit")

    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        headers = [x.strip().lower() for x in next(r)]
        keys = ("match_id", "fixture_id", "matchid", "id")
        positions = [headers.index(k) for k in keys if k in headers]

        if not positions:
            raise RuntimeError(f"Identifiant absent : {path}")

        pos = positions[0]
        return {
            row[pos].strip()
            for row in r
            if len(row) > pos and row[pos].strip()
        }

def write(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

print("=" * 72)
print("ETAPE 323B - PLAN 210 ET PILOTE 10")
print("=" * 72)

if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
    raise SystemExit("STOP : autorisation ID-only requise")

plan_fields, plan = metadata(PLAN)
pilot_fields, pilot = metadata(PILOT)

plan_ids = [r["match_id"].strip() for r in plan]
pilot_ids = [r["match_id"].strip() for r in pilot]

assert len(plan) == 227 and len(set(plan_ids)) == 227
assert len(pilot) == 12 and len(set(pilot_ids)) == 12
assert set(pilot_ids) <= set(plan_ids)

blocked = set()

for path in PROTECTED:
    blocked.update(ids_only(path) & set(plan_ids))

clean_plan = [
    r for r in plan
    if r["match_id"].strip() not in blocked
]
clean_pilot = [
    r for r in pilot
    if r["match_id"].strip() not in blocked
]

if not (
    len(blocked) == 17
    and len(clean_plan) == 210
    and len(clean_pilot) == 10
):
    raise RuntimeError(
        "STOP : population differente de l'etape 322"
    )

def mark(rows):
    out = []
    for row in rows:
        r = dict(row)
        r["protected_clearance"] = "SEVEN_SOURCES_ID_ONLY_PASS"
        r["collection_authorized"] = "NO"
        out.append(r)
    return out

plan_out = mark(clean_plan)
pilot_out = mark(clean_pilot)

extra = ["protected_clearance", "collection_authorized"]
plan_columns = list(dict.fromkeys(plan_fields + extra))
pilot_columns = list(dict.fromkeys(pilot_fields + extra))

out_plan = DEST / "PLAN210_ID_CHECKED_NOT_AUTHORIZED.csv"
out_pilot = DEST / "PILOT10_ID_CHECKED_NOT_AUTHORIZED.csv"

if out_plan.exists() or out_pilot.exists():
    raise RuntimeError("STOP : sortie existante, aucun ecrasement")

DEST.mkdir(parents=True, exist_ok=True)

write(out_plan, plan_columns, plan_out)
write(out_pilot, pilot_columns, pilot_out)

print("\n=== POPULATIONS ===")
print("Candidats exclus :", len(blocked))
print("Plan conserve :", len(plan_out))
print("Pilote conserve :", len(pilot_out))

print("\n=== DIVERSITE DU PILOTE ===")
months = Counter(
    (
        r.get("competition_id", ""),
        r.get("kickoff_utc", "")[:7]
    )
    for r in pilot_out
)

for (competition, month), count in sorted(months.items()):
    print(competition, month, ":", count)

print("\n=== MANIFESTES CREES ===")
print(out_plan)
print(out_pilot)

print("\n=== LIMITES ===")
print("7 sources protegees verifiees par ID uniquement.")
print("Autres partitions non certifiees.")
print("Tous les candidats restent NON autorises a collecter.")
print("Aucun label ou score final consulte.")
print("Aucune collecte effectuee.")
print("TEST_FUTUR non consulte.")
print("V3-C inchange.")
print("ETAPE 323B TERMINEE.")
