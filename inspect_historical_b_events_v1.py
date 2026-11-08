import csv
import json
from pathlib import Path

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

DATASET = JOIN / "BATCH11_B_RESEARCH_V1.csv"

EXTRA = [
    "shots_home_last5",
    "shots_away_last5",
    "sot_home_last5",
    "sot_away_last5",
    "corners_home_last5",
    "corners_away_last5",
]

with DATASET.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 20:
    raise ValueError("BATCH11 doit contenir 20 matchs")

print("\n=== ÉTAPE 127 : AUDIT ÉVÉNEMENTS RÉELS ===")
print("Matchs disponibles :", len(rows))

missing = [col for col in EXTRA if col not in rows[0]]
if missing:
    raise ValueError(f"Variables manquantes : {missing}")

selected = []

for row in rows:
    mid = row["match_id"]
    path = CACHE / f"{mid}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    payload = data.get("data")

    if not isinstance(payload, dict):
        raise ValueError(f"Payload absent : {mid}")

    if payload.get("match_id") != mid:
        raise ValueError(f"Match ID incohérent : {mid}")

    if payload.get("coverage") != "full":
        raise ValueError(f"Timeline non FULL : {mid}")

    events = payload.get("events")

    if not isinstance(events, list):
        raise ValueError(f"Événements invalides : {mid}")

    window = []

    for event in events:
        if not isinstance(event, dict):
            continue

        try:
            minute = float(event.get("minute"))
        except (TypeError, ValueError):
            continue

        if 82 <= minute < 87:
            window.append(event)

    selected.append((row, window))

# Priorité aux matchs ayant le plus d'événements
# dans la fenêtre de calcul de B.
selected.sort(key=lambda item: len(item[1]), reverse=True)

for row, window in selected[:5]:
    mid = row["match_id"]

    print("\n" + "=" * 65)
    print("MATCH :", mid)
    print("LABEL :", row["goal_after_87"])
    print("Événements 82'–87' :", len(window))

    print("\nVARIABLES HISTORIQUES B :")
    for name in EXTRA:
        print(f"  {name}: {row[name]}")

    print("\nÉVÉNEMENTS BRUTS :")

    for event in window[:15]:
        print({
            "minute": event.get("minute"),
            "period": event.get("period"),
            "type": event.get("type"),
            "team": event.get("team"),
            "team_id": event.get("team_id"),
        })

    if len(window) > 15:
        print(f"... {len(window) - 15} événements supplémentaires")

print("\n=== FIN DE L'AUDIT ===")
print("Aucun appel API.")
print("Aucun entraînement.")
print("Aucun fichier modifié.")
print("Scanner LIVE inchangé.")
print("TEST FUTUR non utilisé.")
