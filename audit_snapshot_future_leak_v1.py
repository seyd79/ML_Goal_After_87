import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path("outputs/history")

FILES = [
    "live_feature_snapshots_v3c.csv",
    "live_feature_snapshots_all703_labeled_v3.csv",
]

LAST_EVENTS = [
    "last_goal_minute",
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",
]

CUMULATIVE = [
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_corners_87",
    "away_corners_87",
]


def numeric(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        return dt if dt.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


print("=== ÉTAPE 136 : AUDIT FUITE TEMPORELLE ===")

for name in FILES:
    path = ROOT / name

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    required = (
        {"match_id", "minute", "snapshot_timestamp_utc"}
        | set(LAST_EVENTS)
        | set(CUMULATIVE)
    )

    missing = required - set(columns)
    if missing:
        raise ValueError(
            f"{name}: colonnes absentes {sorted(missing)}"
        )

    print("\n" + "=" * 65)
    print("FICHIER :", name)
    print("Lignes :", len(rows))

    future_by_feature = Counter()
    affected_rows = set()
    affected_matches = set()
    examples = []

    groups = defaultdict(lambda: defaultdict(list))
    invalid_minutes = 0
    invalid_timestamps = 0

    for index, row in enumerate(rows, 1):
        mid = str(row.get("match_id") or "").strip()
        minute = numeric(row.get("minute"))
        captured = timestamp(
            row.get("snapshot_timestamp_utc")
        )

        if minute is None:
            invalid_minutes += 1
            continue

        if captured is None:
            invalid_timestamps += 1
            continue

        if mid and minute in (85, 86, 87):
            groups[mid][int(minute)].append(
                (captured, row)
            )

        for feature in LAST_EVENTS:
            last = numeric(row.get(feature))

            # Strictement supérieur à la minute
            # annoncée par le snapshot.
            if last is not None and last > minute:
                future_by_feature[feature] += 1
                affected_rows.add(index)
                if mid:
                    affected_matches.add(mid)

                if len(examples) < 12:
                    examples.append((
                        mid, minute, feature, last,
                        captured.isoformat()
                    ))

    print("\n=== ÉVÉNEMENTS FUTURS ===")

    for feature in LAST_EVENTS:
        print(
            f"{feature}: "
            f"{future_by_feature[feature]} lignes"
        )

    print(
        "Lignes concernées (sans doublons) :",
        len(affected_rows)
    )
    print(
        "Matchs concernés :",
        len(affected_matches)
    )

    print("\n=== EXEMPLES ===")
    for mid, minute, feature, last, captured in examples:
        print(
            f"{mid} | capture={minute}' | "
            f"{feature}={last}' | "
            f"horodatage={captured}"
        )

    triplets = {
        mid: minutes
        for mid, minutes in groups.items()
        if all(m in minutes for m in (85, 86, 87))
    }

    print("\n=== COUVERTURE 85 / 86 / 87 ===")
    print("Matchs uniques :", len(groups))
    print(
        "Matchs avec les 3 minutes :",
        len(triplets)
    )
    print(
        "Matchs avec 85 et 87 :",
        sum(
            85 in m and 87 in m
            for m in groups.values()
        )
    )

    decreases = Counter()
    chronology_errors = 0
    examples_decreases = []

    for mid, minutes in triplets.items():
        # Première capture de chaque minute.
        snapshots = {
            m: sorted(minutes[m], key=lambda x: x[0])[0]
            for m in (85, 86, 87)
        }

        times = [
            snapshots[m][0]
            for m in (85, 86, 87)
        ]

        if not (times[0] < times[1] < times[2]):
            chronology_errors += 1

        for feature in CUMULATIVE:
            values = [
                numeric(snapshots[m][1].get(feature))
                for m in (85, 86, 87)
            ]

            if any(v is None for v in values):
                continue

            if values[1] < values[0] or values[2] < values[1]:
                decreases[feature] += 1

                if len(examples_decreases) < 10:
                    examples_decreases.append(
                        (mid, feature, values)
                    )

    print("\n=== COHÉRENCE DES COMPTEURS ===")
    print(
        "Ordre des horodatages incorrect :",
        chronology_errors
    )

    for feature in CUMULATIVE:
        print(
            f"{feature}: "
            f"{decreases[feature]} matchs avec baisse"
        )

    for mid, feature, values in examples_decreases:
        print(
            f"EXEMPLE {mid}: {feature} = {values}"
        )

    print("\nMinutes invalides :", invalid_minutes)
    print("Horodatages invalides :", invalid_timestamps)

print("\n=== FIN ===")
print("Aucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Aucun modèle déployé.")
print("TEST FUTUR historique B non ouvert.")
