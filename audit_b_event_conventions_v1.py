import csv
import json
import math
from pathlib import Path
from collections import Counter

from build_historical_ab_research_v3 import SHOTS, SOT

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

dataset = JOIN / "BATCH11_B_RESEARCH_V1.csv"

with dataset.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 20:
    raise ValueError("20 matchs attendus")

comparisons = Counter()
event_types = Counter()
goal_contributions = Counter()

print("=== ETAPE 128 : CONVENTIONS HISTORIQUE / LIVE ===")
print("Matchs :", len(rows))
print("Fenêtre : 82 <= minute < 87")
print("Aucun fichier TEST utilisé")

for row in rows:
    mid = row["match_id"]

    with (CACHE / f"{mid}.json").open(
        encoding="utf-8"
    ) as f:
        cache = json.load(f)

    data = cache.get("data", {})

    if (
        data.get("match_id") != mid
        or data.get("coverage") != "full"
        or not isinstance(data.get("events"), list)
    ):
        raise ValueError(f"Cache invalide : {mid}")

    historical = Counter()
    live = Counter()

    for event in data["events"]:
        if not isinstance(event, dict):
            continue

        try:
            minute = float(event.get("minute"))
        except (ValueError, TypeError):
            continue

        if not math.isfinite(minute):
            continue

        if (
            event.get("period") != "second_half"
            or not (82 <= minute < 87)
        ):
            continue

        typ = str(event.get("type") or "").lower()
        event_types[typ] += 1

        # Règles du constructeur historique.
        if typ in SHOTS:
            historical["shots"] += 1

        if typ in SOT:
            historical["sot"] += 1

        if typ == "corner_kick":
            historical["corners"] += 1

        # Règles de classification LIVE observées.
        is_shot = (
            typ in {"shot", "shots", "attempt", "total_shot"}
            or "shot" in typ
        )

        is_sot = (
            "shot_on_target" in typ
            or "on_target" in typ
            or "saved" in typ
        )

        if is_shot:
            live["shots"] += 1

        if is_sot:
            live["sot"] += 1

        if "corner" in typ:
            live["corners"] += 1

        if typ in {"goal", "own_goal", "penalty_scored"}:
            goal_contributions[
                (typ, typ in SHOTS, typ in SOT)
            ] += 1

    expected = {
        "shots": (
            int(row["shots_home_last5"])
            + int(row["shots_away_last5"])
        ),
        "sot": (
            int(row["sot_home_last5"])
            + int(row["sot_away_last5"])
        ),
        "corners": (
            int(row["corners_home_last5"])
            + int(row["corners_away_last5"])
        ),
    }

    differences = []

    for feature in ("shots", "sot", "corners"):
        if historical[feature] != expected[feature]:
            comparisons[f"historical_csv_{feature}"] += 1
            differences.append(
                f"{feature}: CSV={expected[feature]}, "
                f"recalcul={historical[feature]}"
            )

        if live[feature] != historical[feature]:
            comparisons[f"live_historical_{feature}"] += 1
            differences.append(
                f"{feature}: hist={historical[feature]}, "
                f"live={live[feature]}"
            )

    if differences:
        print(f"\n{mid}")
        for difference in differences:
            print(" ", difference)

print("\n=== TYPES EVENEMENTS ===")
for typ, count in event_types.most_common():
    print(f"{typ}: {count}")

print("\n=== CONTRIBUTION DES BUTS ===")
for (typ, shot, sot), count in goal_contributions.items():
    print(
        f"{typ}: {count} | "
        f"historical_shot={shot} | historical_sot={sot}"
    )

print("\n=== BILAN DIVERGENCES ===")
for key in (
    "historical_csv_shots",
    "historical_csv_sot",
    "historical_csv_corners",
    "live_historical_shots",
    "live_historical_sot",
    "live_historical_corners",
):
    print(f"{key}: {comparisons[key]} matchs")

print("\nIMPORTANT : comparaison des totaux uniquement.")
print("Attribution domicile/exterieur non certifiee.")
print("Aucun appel API.")
print("Aucun fichier modifie.")
print("Scanner LIVE inchange.")
