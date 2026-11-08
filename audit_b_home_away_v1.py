import csv
import json
from pathlib import Path
from collections import Counter

from build_historical_ab_research_v3 import (
    SHOTS, SOT, match_side, event_minute
)

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

MANIFEST = JOIN / "HISTORICAL_WAVE1_BATCH11_MANIFEST_V1.csv"
DATASET = JOIN / "BATCH11_B_RESEARCH_V1.csv"

FEATURES = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ids = [row["match_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Doublons : {path}")
    return {row["match_id"]: row for row in rows}

manifest = read_csv(MANIFEST)
dataset = read_csv(DATASET)

if len(dataset) != 20 or len(manifest) != 20:
    raise ValueError("20 matchs attendus")

if set(manifest) != set(dataset):
    raise ValueError("IDs manifest / dataset différents")

required = {"home_team", "away_team"}
missing = required - set(next(iter(manifest.values())))
if missing:
    raise ValueError(
        f"Colonnes équipes absentes : {sorted(missing)}. "
        f"Colonnes disponibles : "
        f"{list(next(iter(manifest.values())))}"
    )

mismatches = Counter()
corner_matches = 0
inverted_matches = 0
unknown_ownership = 0

print("=== ÉTAPE 129 : DOMICILE / EXTÉRIEUR ===")

for mid, row in dataset.items():
    path = CACHE / f"{mid}.json"

    with path.open(encoding="utf-8") as f:
        cache = json.load(f)

    payload = cache.get("data", {})

    if (
        payload.get("match_id") != mid
        or payload.get("coverage") != "full"
        or not isinstance(payload.get("events"), list)
    ):
        raise ValueError(f"Cache invalide : {mid}")

    info = manifest[mid]
    historical = Counter()
    direct = Counter()
    n_corners = 0

    for event in payload["events"]:
        if not isinstance(event, dict):
            continue

        try:
            minute = event_minute(event)
        except (ValueError, TypeError):
            continue

        if (
            event.get("period") != "second_half"
            or not (82 <= minute < 87)
        ):
            continue

        typ = str(event.get("type") or "").strip().lower()

        if typ not in SHOTS and typ not in SOT and typ != "corner_kick":
            continue

        side = match_side(event, info)

        if side not in ("home", "away"):
            unknown_ownership += 1
            print(f"ÉQUIPE INCONNUE : {mid} | {typ}")
            continue

        if typ in SHOTS:
            historical[f"shots_{side}_last5"] += 1

        if typ in SOT:
            historical[f"sot_{side}_last5"] += 1

        if typ == "corner_kick":
            n_corners += 1

            opposite = (
                "away" if side == "home" else "home"
            )

            # Convention historique du builder B.
            historical[f"corners_{opposite}_last5"] += 1

            # Convention directe du scanner LIVE.
            direct[f"corners_{side}_last5"] += 1

    if n_corners:
        corner_matches += 1

    different_corners = any(
        historical[f"corners_{side}_last5"]
        != direct[f"corners_{side}_last5"]
        for side in ("home", "away")
    )

    if different_corners:
        inverted_matches += 1

    errors = []

    for feature in FEATURES:
        expected = int(row[feature])
        actual = historical[feature]

        if expected != actual:
            mismatches[feature] += 1
            errors.append(
                f"{feature}: CSV={expected}, recalcul={actual}"
            )

    if n_corners or errors:
        print(f"\n{mid}")
        print(
            "Équipes :",
            info["home_team"],
            "contre",
            info["away_team"]
        )
        print("Corners observés :", n_corners)

        print(
            "Corners historique B :",
            historical["corners_home_last5"],
            historical["corners_away_last5"]
        )

        print(
            "Corners attribution directe :",
            direct["corners_home_last5"],
            direct["corners_away_last5"]
        )

        print(
            "Répartition différente :",
            different_corners
        )

        for error in errors:
            print("DIVERGENCE CSV :", error)

print("\n=== BILAN ===")
print("Matchs analysés :", len(dataset))
print("Matchs avec corners :", corner_matches)
print(
    "Matchs avec répartition corners différente :",
    inverted_matches
)
print("Attributions inconnues :", unknown_ownership)

print("\n=== ÉCARTS AVEC LES SIX VARIABLES CSV ===")
for feature in FEATURES:
    print(feature, ":", mismatches[feature])

print("\nAucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Scanner LIVE inchangé.")
print("TEST FUTUR non utilisé.")
