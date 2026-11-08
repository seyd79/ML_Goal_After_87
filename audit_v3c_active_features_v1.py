import csv
import json
import math
from pathlib import Path
from collections import Counter

SCHEMA = Path(
    "outputs/model_v3_experiments/schemas/"
    "V3_C_AGGRESSIVE_schema.json"
)

SNAPSHOTS = Path(
    "outputs/history/live_feature_snapshots_v3c.csv"
)

LAST_EVENTS = {
    "last_goal_minute": "minutes_since_last_goal",
    "last_shot_minute": "minutes_since_last_shot",
    "last_shot_on_target_minute":
        "minutes_since_last_shot_on_target",
    "last_corner_minute": "minutes_since_last_corner",
}

PERIOD_FEATURES = [
    "shots_80_87",
    "sot_80_87",
    "corners_80_87",
    "fouls_80_87",
    "cards_80_87",
    "subs_80_87",
]

def numeric(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None

with SCHEMA.open(encoding="utf-8") as f:
    schema = json.load(f)

if isinstance(schema, dict):
    # Même priorité que le chargeur V3-C.
    features = schema.get(
        "features_used",
        schema.get(
            "core_features",
            schema.get(
                "features",
                schema.get("feature_names", [])
            )
        )
    )
elif isinstance(schema, list):
    features = schema
else:
    raise ValueError("Format de schéma invalide")

if not isinstance(features, list):
    raise ValueError("Liste de features invalide")

features = [str(x).strip() for x in features]

if len(features) != 38 or len(set(features)) != 38:
    raise ValueError(
        f"38 variables uniques attendues, trouvé {len(features)}"
    )

print("=== ÉTAPE 138 : SCHÉMA ACTIF V3-C ===")
print("Variables du modèle :", len(features))

for i, feature in enumerate(features, 1):
    print(f"{i:02d}. {feature}")

direct_temporal = [
    f for f in LAST_EVENTS
    if f in features
]

derived_temporal = [
    f for f in LAST_EVENTS.values()
    if f in features
]

period_used = [
    f for f in PERIOD_FEATURES
    if f in features
]

print("\n=== VARIABLES TEMPORELLES UTILISÉES ===")
print("Minutes du dernier événement :", direct_temporal)
print("Minutes écoulées depuis événement :", derived_temporal)
print("Compteurs de période :", period_used)

with SNAPSHOTS.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)
    columns = set(reader.fieldnames or [])
    rows = list(reader)

print("\n=== AUDIT DES SNAPSHOTS ===")
print("Lignes :", len(rows))

missing = [f for f in features if f not in columns]
print("Variables du modèle absentes du CSV :", missing)

anomalies = Counter()
rows_with_future = set()
rows_with_active_future = set()
matches_with_active_future = set()
examples = []

for index, row in enumerate(rows, 1):
    minute = numeric(row.get("minute"))
    mid = str(row.get("match_id") or "")

    if minute is None:
        continue

    for last_feature, derived_feature in LAST_EVENTS.items():
        last = numeric(row.get(last_feature))

        if last is None or last <= minute:
            continue

        rows_with_future.add(index)

        anomalies[last_feature] += 1

        affects_active_temporal = (
            last_feature in features
            or derived_feature in features
        )

        if affects_active_temporal:
            rows_with_active_future.add(index)

            if mid:
                matches_with_active_future.add(mid)

            if len(examples) < 15:
                examples.append({
                    "match_id": mid,
                    "minute": minute,
                    "feature": last_feature,
                    "last_event": last,
                    "active_direct": last_feature in features,
                    "active_derived": derived_feature in features,
                    "derived_value": row.get(derived_feature)
                })

print("\n=== BILAN ===")
print(
    "Lignes avec anomalie temporelle observée :",
    len(rows_with_future)
)
print(
    "Lignes où la variable directe ou dérivée "
    "est dans le schéma actif :",
    len(rows_with_active_future)
)
print(
    "Matchs concernés par ces variables actives :",
    len(matches_with_active_future)
)

print("\n=== ANOMALIES PAR TYPE ===")
for feature in LAST_EVENTS:
    print(feature, ":", anomalies[feature])

print("\n=== EXEMPLES EXPOSITION MODÈLE ===")
for example in examples:
    print(example)

print("\n=== CONCLUSION TECHNIQUE ===")

if rows_with_active_future:
    print(
        "EXPOSITION DIRECTE OU DÉRIVÉE DÉTECTÉE : "
        "audit de provenance et d'impact requis."
    )
else:
    print(
        "AUCUNE EXPOSITION DIRECTE OU DÉRIVÉE "
        "détectée dans les quatre familles analysées."
    )
    print(
        "Cela ne certifie pas les autres variables "
        "ni les compteurs de période."
    )

print("\nAucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Aucun modèle déployé.")
