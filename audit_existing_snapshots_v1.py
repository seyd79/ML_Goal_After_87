import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path("outputs")

FILES = [
    "history/live_feature_snapshots_all703_labeled_v3.csv",
    "history/live_feature_snapshots_labeled.csv",
    "history/live_feature_snapshots.csv",
    "history/live_feature_snapshots_labeled_v3.csv",
    "history/live_feature_snapshots_labeled_v2.csv",
    "history/live_feature_snapshots_v3c.csv",
    "history/v3c_127_alertes_snapshots.csv",
    "v3c_external_test/v3c_scored_snapshots_full.csv",
]

SUMMARY = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "V3C_545_CONSOLIDATION_SUMMARY_V1.json"
)

with SUMMARY.open(encoding="utf-8") as f:
    features_b = json.load(f)["features_b"]

TIME_HINTS = (
    "timestamp", "captured", "observed",
    "fetched", "collected", "created",
    "snapshot_time", "datetime"
)

ID_HINTS = (
    "match_id", "fixture_id", "event_id"
)

print("=== ÉTAPE 134 : AUDIT SNAPSHOTS EXISTANTS ===")

for relative in FILES:
    path = ROOT / relative

    print("\n" + "=" * 72)
    print("FICHIER :", path)

    if not path.is_file():
        print("ABSENT")
        continue

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        columns = reader.fieldnames or []
        rows = list(reader)

    print("Lignes :", len(rows))
    print("Colonnes :", len(columns))

    print("\nIDENTIFIANTS :")
    ids = [
        c for c in columns
        if c.lower() in ID_HINTS
    ]
    print(ids or "Aucun identifiant standard")

    print("\nHORODATAGES POTENTIELS :")
    time_columns = [
        c for c in columns
        if any(x in c.lower() for x in TIME_HINTS)
    ]
    print(time_columns or "Aucun")

    for col in time_columns[:5]:
        values = [
            r.get(col, "").strip()
            for r in rows
            if r.get(col, "").strip()
        ]

        print(
            f"  {col}: renseignés={len(values)}, "
            f"exemple={values[0] if values else 'ABSENT'}"
        )

    if "match_id" in columns:
        match_ids = [
            r["match_id"] for r in rows
            if r.get("match_id")
        ]
        counts = Counter(match_ids)

        print("\nMATCHS UNIQUES :", len(counts))
        print(
            "Matchs avec plusieurs lignes :",
            sum(n > 1 for n in counts.values())
        )

    present = [
        c for c in features_b
        if c in columns
    ]

    missing = [
        c for c in features_b
        if c not in columns
    ]

    print("\nVARIABLES B PRÉSENTES :", len(present), "/29")
    print("Variables présentes :", present)
    print("Variables absentes :", missing)

    event_columns = [
        c for c in columns
        if any(
            x in c.lower()
            for x in ("timeline", "events", "event_json", "raw")
        )
    ]

    print("\nCOLONNES ÉVÉNEMENTS BRUTS :", event_columns)

    label_columns = [
        c for c in columns
        if any(
            x in c.lower()
            for x in ("label", "target", "goal_after_87")
        )
    ]

    print("COLONNES LABEL :", label_columns)

print("\n=== STRUCTURE SNAPSHOTS JSON FIRST HALF ===")

json_path = (
    ROOT /
    "first_half_37/shadow_live/"
    "FIRST_HALF37_LIVE_STATS_SNAPSHOTS.json"
)

if json_path.exists():
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    print("Type racine :", type(data).__name__)

    if isinstance(data, dict):
        print("Clés racine :", list(data)[:30])

    elif isinstance(data, list):
        print("Nombre d'entrées :", len(data))
        if data and isinstance(data[0], dict):
            print("Clés première entrée :", list(data[0])[:30])

print("\n=== FIN ===")
print("Aucun appel API")
print("Aucun entraînement")
print("Aucun fichier modifié")
print("TEST FUTUR historique B non utilisé")
