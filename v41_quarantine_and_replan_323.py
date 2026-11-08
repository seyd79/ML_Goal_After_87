import csv
import os
from collections import Counter
from datetime import date
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

PROTECTED = {
    "V3I_BATCH1_279":
        ROOT / "v3i_unseen_holdout_500/batch1_frozen_279/V3I_BATCH1_279_IDS.csv",
    "V3I_BATCH2_221":
        ROOT / "v3i_unseen_holdout_500/batch2_frozen_221/V3I_BATCH2_221_READY_BLIND.csv",
    "V3I_HOLDOUT500":
        ROOT / "v3i_unseen_holdout_500/V3I_UNSEEN_HOLDOUT_500_V3C_READY_BLIND.csv",
    "V3I_RESERVE168":
        ROOT / "v3i_unseen_holdout_500/reserve168_final_v3c_only/reserve168_predictions_v3c_final.csv",
    "V3C_EXTERNAL_TEST":
        ROOT / "v3c_external_test/v3c_scored_snapshots_full.csv",
    "V4_VALIDATION154":
        ROOT / "v4_research/V4_CHRONO_VALIDATION_V1.csv",
    "FIRST_HALF_EXTERNAL_HOLDOUT":
        ROOT / "first_half_37/v3_draw_with_goals_historical_external_holdout/first_half_37_v3_historical_external_listing.csv",
}

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        return fields, list(reader)

def read_ids_only(path):
    if "test_futur" in str(path).lower():
        raise RuntimeError("TEST_FUTUR interdit")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = [x.strip().lower() for x in next(reader)]

        candidates = ("match_id", "fixture_id", "matchid", "id")
        positions = [headers.index(x) for x in candidates if x in headers]

        if not positions:
            raise RuntimeError("Colonne ID absente : " + str(path))

        pos = positions[0]
        return {
            row[pos].strip()
            for row in reader
            if len(row) > pos and row[pos].strip()
        }

def month(row):
    kickoff = row.get("kickoff_utc", "")
    value = (kickoff or row.get("month", "")).strip()
    return value[:7]

def comp(row):
    return str(row.get("competition_id", "")).strip()

def day(row):
    return str(row.get("kickoff_utc", "")).strip()[:10]

def week(row):
    return date.fromisoformat(day(row)).isocalendar()[:2]

def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

print("=" * 76)
print("ETAPE 323 - QUARANTAINE ET REPLANIFICATION V4.1")
print("=" * 76)

if os.environ.get("V41_AUTHORISE_ID_ONLY") != "OUI":
    raise SystemExit(
        "STOP : autorisation ID-only requise."
    )

plan_fields, plan_rows = read_csv(PLAN)
pilot_fields, pilot_rows = read_csv(PILOT)

plan_ids = [r["match_id"].strip() for r in plan_rows]
pilot_ids = [r["match_id"].strip() for r in pilot_rows]

assert len(plan_rows) == 227
assert len(pilot_rows) == 12
assert len(set(plan_ids)) == 227
assert len(set(pilot_ids)) == 12
assert set(pilot_ids) <= set(plan_ids)

by_id = {r["match_id"].strip(): r for r in plan_rows}

blocked = set()

print("\n=== EXCLUSIONS ===")

for name, path in PROTECTED.items():
    ids = read_ids_only(path)
    collisions = ids & set(plan_ids)
    blocked.update(collisions)

    print(name, "| collisions :", len(collisions))

blocked_pilot = set(pilot_ids) & blocked

assert len(blocked) == 17, "Nombre de collisions modifie"
assert len(blocked_pilot) == 2, "Collisions pilote modifiees"

clean_plan = [
    r for r in plan_rows
    if r["match_id"].strip() not in blocked
]

kept_pilot = [
    by_id[mid]
    for mid in pilot_ids
    if mid not in blocked
]

print("\n=== APRES QUARANTAINE ===")
print("Plan initial :", len(plan_rows))
print("Exclusions uniques :", len(blocked))
print("Plan restant :", len(clean_plan))
print("Pilote restant :", len(kept_pilot))

assert len(clean_plan) == 210
assert len(kept_pilot) == 10

# Remplacer chaque match exclu par un candidat
# de la meme competition et du meme mois.
selected = list(kept_pilot)
selected_ids = {r["match_id"] for r in selected}
replacement_ids = set()

for old_id in pilot_ids:
    if old_id not in blocked:
        continue

    old = by_id[old_id]
    target_comp = comp(old)
    target_month = month(old)

    candidates = [
        r for r in clean_plan
        if r["match_id"] not in selected_ids
        and comp(r) == target_comp
        and month(r) == target_month
    ]

    if not candidates:
        raise RuntimeError(
            "Aucun remplacant dans la meme competition "
            f"et le meme mois : {target_comp} / {target_month}. "
            "Aucun manifeste cree."
        )

    used_weeks = {
        (comp(r), week(r))
        for r in selected
    }
    used_days = {
        (comp(r), day(r))
        for r in selected
    }

    candidates.sort(
        key=lambda r: (
            (comp(r), week(r)) in used_weeks,
            (comp(r), day(r)) in used_days,
            day(r),
            r["match_id"],
        )
    )

    replacement = candidates[0]
    selected.append(replacement)
    selected_ids.add(replacement["match_id"])
    replacement_ids.add(replacement["match_id"])

    print(
        "REMPLACEMENT :",
        target_comp,
        target_month,
        "| nouveau candidat :",
        replacement["match_id"]
    )

assert len(selected) == 12
assert len(selected_ids) == 12
assert selected_ids <= {
    r["match_id"] for r in clean_plan
}
assert not (selected_ids & blocked)

def mark(row, role):
    result = dict(row)
    result["protected_clearance"] = "SEVEN_ID_SOURCES_CHECKED"
    result["collection_authorized"] = "NO"
    result["v41_research_role"] = role
    return result

new_plan = [
    mark(r, "CANDIDATE")
    for r in clean_plan
]

new_pilot = [
    mark(
        r,
        "REPLACEMENT" if r["match_id"] in replacement_ids
        else "RETAINED"
    )
    for r in selected
]

output_fields = list(dict.fromkeys(
    plan_fields
    + [
        "protected_clearance",
        "collection_authorized",
        "v41_research_role",
    ]
))

out_plan = DEST / "PLAN210_ID_CHECKED_NOT_AUTHORIZED.csv"
out_pilot = DEST / "PILOT12_REVISED_NOT_AUTHORIZED.csv"

if out_plan.exists() or out_pilot.exists():
    raise RuntimeError(
        "Fichiers de sortie deja existants : "
        "aucun ecrasement autorise"
    )

DEST.mkdir(parents=True, exist_ok=True)

write_csv(out_plan, new_plan, output_fields)
write_csv(out_pilot, new_pilot, output_fields)

print("\n=== BILAN ===")
print("Plan nettoye :", len(new_plan))
print("Pilote reconstitue :", len(new_pilot))
print("Nouveaux remplacants :", len(replacement_ids))
print(
    "Couples competition-mois du pilote :",
    len({(comp(r), month(r)) for r in new_pilot})
)
print("Fichier plan :", out_plan)
print("Fichier pilote :", out_pilot)

print("\n=== SECURITE ===")
print("Collisions protegees exclues des nouveaux manifestes.")
print("Aucun identifiant protege exporte.")
print("Collection_authorized = NO.")
print("Controle limite a 7 sources connues.")
print("Autres partitions non certifiees.")
print("Aucun label ni score final exploite.")
print("Aucune collecte.")
print("TEST_FUTUR non consulte.")
print("V3-C inchange.")
print("ETAPE 323 TERMINEE.")
