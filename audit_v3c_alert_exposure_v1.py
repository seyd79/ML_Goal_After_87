import csv
import math
from pathlib import Path
from collections import Counter, defaultdict

PATH = Path(
    "outputs/history/live_feature_snapshots_v3c.csv"
)

ACTIVE_TEMPORAL = {
    "last_shot_minute",
    "last_shot_on_target_minute",
    "last_corner_minute",
}

def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def is_alert(value):
    return str(value).strip().lower() in {
        "true", "1", "yes", "oui"
    }

with PATH.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = list(reader)

required = {
    "match_id",
    "minute",
    "alert",
    "model_probability",
} | ACTIVE_TEMPORAL

missing = required - set(columns)

if missing:
    raise ValueError(
        f"Colonnes manquantes : {sorted(missing)}"
    )

groups = defaultdict(list)
minutes = Counter()
flags = Counter()
decision_counts = Counter()
examples = []

affected_matches = set()
affected_alert_matches = set()

for row in rows:
    minute = number(row.get("minute"))
    match_id = str(row.get("match_id") or "").strip()

    if minute is None:
        continue

    anomalous = [
        feature
        for feature in ACTIVE_TEMPORAL
        if (
            number(row.get(feature)) is not None
            and number(row.get(feature)) > minute
        )
    ]

    exposed = bool(anomalous)
    alert = is_alert(row.get("alert"))
    probability = number(
        row.get("model_probability")
    )

    group = "EXPOSE" if exposed else "NON_EXPOSE"

    groups[group].append(probability)
    minutes[(group, int(minute))] += 1
    flags[(group, str(row.get("alert")))] += 1

    decision_counts[
        (group, str(row.get("decision", "")))
    ] += 1

    if exposed:
        affected_matches.add(match_id)

        if alert:
            affected_alert_matches.add(match_id)

        if len(examples) < 20:
            examples.append({
                "match_id": match_id,
                "minute": int(minute),
                "features": anomalous,
                "probability": probability,
                "alert": row.get("alert"),
                "decision": row.get("decision"),
            })

def avg(values):
    valid = [
        v for v in values
        if v is not None
    ]

    if not valid:
        return "INDISPONIBLE"

    return round(sum(valid) / len(valid), 5)

print("=== ETAPE 139 : EXPOSITION ALERTES V3-C ===")
print("Snapshots total :", len(rows))

for group in ("EXPOSE", "NON_EXPOSE"):
    values = groups[group]

    print(f"\n=== {group} ===")
    print("Snapshots :", len(values))
    print("Probabilite moyenne :", avg(values))

    print("Repartition par minute :")
    for minute in (85, 86, 87):
        print(
            f"  {minute}' :",
            minutes[(group, minute)]
        )

    print("Valeurs de la colonne alert :")
    for (g, value), count in flags.items():
        if g == group:
            print(" ", repr(value), ":", count)

    print("Valeurs de la colonne decision :")
    for (g, value), count in decision_counts.items():
        if g == group:
            print(" ", repr(value), ":", count)

print("\n=== MATCHS EXPOSES ===")
print("Matchs avec anomalie :", len(affected_matches))
print(
    "Matchs avec au moins une alerte exposee :",
    len(affected_alert_matches)
)

print("\n=== EXEMPLES ===")
for example in examples:
    print(example)

print("\n=== LIMITES ===")
print(
    "Audit descriptif : aucune preuve "
    "de modification des predictions."
)
print(
    "Les alertes multiples d'un meme match "
    "ne sont pas des paris independants."
)
print("Aucun appel API.")
print("Aucun entrainement.")
print("Aucun fichier modifie.")
print("Scanner LIVE inchange.")
