import csv
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PATH = Path(
    "outputs/history/live_feature_snapshots_v3c.csv"
)

WATCH = (
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",
)


def num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def true_value(value):
    return str(value).strip().lower() in (
        "true", "1", "yes", "oui"
    )


def parse_time(value):
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None


with PATH.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = list(reader)

required = {
    "match_id", "minute", "alert",
    "snapshot_timestamp_utc"
} | set(WATCH)

missing = required - set(columns)

if missing:
    raise ValueError(
        f"Colonnes absentes : {sorted(missing)}"
    )

groups = defaultdict(list)

for row in rows:
    mid = str(row.get("match_id") or "").strip()
    minute = num(row.get("minute"))
    captured = parse_time(
        row.get("snapshot_timestamp_utc")
    )

    if not mid or minute is None or captured is None:
        continue

    bad = [
        feature
        for feature in WATCH
        if (
            num(row.get(feature)) is not None
            and num(row.get(feature)) > minute
        )
    ]

    groups[mid].append({
        "minute": int(minute),
        "time": captured,
        "alert": true_value(row.get("alert")),
        "exposed": bool(bad),
        "bad": bad,
        "probability": num(
            row.get("model_probability")
        ),
        "goal_after_87": row.get(
            "goal_after_87", ""
        ),
        "target_status": row.get(
            "target_status", ""
        ),
    })


affected = []

for mid, snapshots in groups.items():
    snapshots.sort(
        key=lambda r: (r["time"], r["minute"])
    )

    alerts = [
        s for s in snapshots if s["alert"]
    ]

    exposed_alerts = [
        s for s in alerts if s["exposed"]
    ]

    if not exposed_alerts:
        continue

    first = alerts[0]

    affected.append({
        "match_id": mid,
        "first_alert": first,
        "alert_count": len(alerts),
        "exposed_alert_count": len(exposed_alerts),
        "exposed_minutes": sorted({
            s["minute"] for s in exposed_alerts
        }),
        "snapshots": snapshots,
    })


affected.sort(key=lambda x: x["match_id"])

print("=== ETAPE 140 : MATCHS ALERTES EXPOSES ===")
print("Matchs :", len(affected))

first_exposed = 0
first_clean = 0
pending = 0
conflicting_labels = 0

for entry in affected:
    mid = entry["match_id"]
    first = entry["first_alert"]

    if first["exposed"]:
        first_exposed += 1
    else:
        first_clean += 1

    labels = {
        str(s["goal_after_87"]).strip()
        for s in entry["snapshots"]
        if str(s["goal_after_87"]).strip().lower()
        not in ("", "nan", "none", "null")
    }

    statuses = sorted({
        str(s["target_status"]).strip()
        for s in entry["snapshots"]
        if str(s["target_status"]).strip()
    })

    if not labels:
        pending += 1

    if len(labels) > 1:
        conflicting_labels += 1

    print("\n" + "-" * 62)
    print("Match :", mid)
    print(
        "Premiere alerte :",
        f"{first['minute']}'",
        "| probabilite =", first["probability"]
    )
    print(
        "Premiere alerte exposee :",
        first["exposed"]
    )
    print(
        "Alertes snapshots :",
        entry["alert_count"]
    )
    print(
        "Alertes snapshots exposees :",
        entry["exposed_alert_count"]
    )
    print(
        "Minutes des alertes exposees :",
        entry["exposed_minutes"]
    )
    print(
        "Labels enregistres :",
        sorted(labels) if labels else "ABSENTS"
    )
    print(
        "Statuts cibles :",
        statuses if statuses else "ABSENTS"
    )

print("\n=== BILAN ===")
print("Matchs avec alerte exposee :", len(affected))
print(
    "Premiere alerte elle-meme exposee :",
    first_exposed
)
print(
    "Premiere alerte non exposee au controle :",
    first_clean
)
print(
    "Matchs sans label renseigne :",
    pending
)
print(
    "Matchs avec labels contradictoires :",
    conflicting_labels
)

if len(affected) != 13:
    print(
        "ATTENTION : effectif different de l'audit 139."
    )

print("\n=== LIMITES ===")
print(
    "Une premiere alerte non exposee a ce controle "
    "n'est pas necessairement exempte d'autres anomalies."
)
print(
    "Les labels ne sont pas consideres comme verifies "
    "sans examen de leur statut et de leur source."
)
print("Aucun appel API.")
print("Aucun entrainement.")
print("Aucun fichier modifie.")
print("Scanner LIVE inchange.")
