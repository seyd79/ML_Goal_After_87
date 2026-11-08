import csv
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path("outputs/history")

FILES = [
    "live_feature_snapshots_all703_labeled_v3.csv",
    "live_feature_snapshots_v3c.csv",
]

STATS = [
    "home_shots_87",
    "away_shots_87",
    "home_shots_on_target_87",
    "away_shots_on_target_87",
    "home_corners_87",
    "away_corners_87",
]

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)

def minute_value(value):
    try:
        v = float(str(value).strip())
        return v if 0 <= v <= 130 else None
    except (TypeError, ValueError):
        return None

def timestamp(value):
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None

print("=== ÉTAPE 135 : COUVERTURE TEMPORELLE LIVE ===")

for filename in FILES:
    path = ROOT / filename
    columns, rows = read(path)

    print("\n" + "=" * 65)
    print("FICHIER :", filename)
    print("Lignes :", len(rows))

    minute_columns = [
        c for c in columns
        if (
            "minute" in c.lower()
            or c.lower() in ("elapsed", "match_time")
        )
    ]

    print("\nCOLONNES MINUTES :", minute_columns)

    for col in minute_columns:
        values = [
            minute_value(r.get(col))
            for r in rows
        ]
        valid = [v for v in values if v is not None]

        print(
            f"{col}: {len(valid)}/{len(rows)} "
            "valeurs numériques plausibles"
        )

        if valid:
            distribution = Counter(
                int(v) for v in valid
                if 80 <= v <= 90
            )
            print(
                "  Distribution 80'–90' :",
                dict(sorted(distribution.items()))
            )

    print("\nSTATISTIQUES PRÉSENTES :")

    for col in STATS:
        if col not in columns:
            print(col, ": COLONNE ABSENTE")
            continue

        filled = sum(
            str(r.get(col) or "").strip().lower()
            not in ("", "nan", "none", "null")
            for r in rows
        )

        print(col, ":", filled, "/", len(rows))

    groups = defaultdict(list)

    for row in rows:
        mid = str(row.get("match_id") or "").strip()
        t = timestamp(row.get("snapshot_timestamp_utc"))

        if mid and t:
            groups[mid].append((t, row))

    print("\nMATCHS HORODATÉS :", len(groups))
    print(
        "Avec 2+ captures :",
        sum(len(v) >= 2 for v in groups.values())
    )
    print(
        "Avec 3+ captures :",
        sum(len(v) >= 3 for v in groups.values())
    )

    candidate_minute_columns = [
        c for c in minute_columns
        if c.lower() in (
            "minute", "match_minute",
            "current_minute", "match_elapsed_minute"
        )
    ]

    print("\nCHRONOLOGIE DES CINQ PREMIERS MATCHS :")

    for mid in sorted(groups)[:5]:
        snapshots = sorted(
            groups[mid],
            key=lambda x: x[0]
        )

        print("\n", mid)

        for t, row in snapshots[:8]:
            minutes = {
                c: row.get(c)
                for c in candidate_minute_columns
            }
            print(
                " ", t.isoformat(),
                "minutes=", minutes,
                "shots_home=",
                row.get("home_shots_87"),
                "corners_home=",
                row.get("home_corners_87")
            )

    print("\nAUTRES COLONNES TEMPORELLES :")
    print([
        c for c in columns
        if any(
            x in c.lower()
            for x in ("period", "clock", "half", "phase")
        )
    ])

print("\n=== FIN ===")
print("Aucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Aucune donnée TEST FUTUR historique B ouverte.")
print("Scanner LIVE inchangé.")
