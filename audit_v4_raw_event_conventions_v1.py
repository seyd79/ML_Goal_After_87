from pathlib import Path
from collections import Counter, defaultdict
import json
import pandas as pd

ROOT = Path("outputs/pressure_historical")
JOIN = ROOT / "pressure_v2_dataset_join"
CACHE = ROOT / "timeline_cache"

print("=" * 72)
print("ÉTAPE 177 — CONVENTIONS DES ÉVÉNEMENTS V4")
print("=" * 72)

plan = pd.read_csv(
    JOIN / "ALL_320_CHRONO_SPLIT_PLAN_V1.csv",
    dtype={"match_id": str},
    usecols=["match_id", "split"],
)

ids_all = set(
    plan.loc[plan["split"] == "TRAIN", "match_id"]
)

ids_batch = set()
paths = sorted(JOIN.glob("BATCH*_B_RESEARCH_V1.csv"))

if len(paths) != 10:
    raise SystemExit("10 lots B attendus")

for path in paths:
    part = pd.read_csv(
        path,
        dtype={"match_id": str},
        usecols=["match_id", "research_split"],
    )
    ids_batch.update(
        part.loc[
            part["research_split"] == "TRAIN",
            "match_id",
        ]
    )

if len(ids_all) != 93 or len(ids_batch) != 90:
    raise SystemExit("Population TRAIN inattendue")

if ids_all & ids_batch:
    raise SystemExit("Chevauchement des sources")

sources = {
    "ALL_320": ids_all,
    "BATCH02_11": ids_batch,
}

type_counts = defaultdict(Counter)
window_counts = defaultdict(Counter)
team_counts = defaultdict(Counter)
period_counts = defaultdict(Counter)
examples = defaultdict(dict)
quality = defaultdict(Counter)

def normalize(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)

def minute_value(value):
    try:
        n = float(value)
        if n != n:
            return None
        return n
    except (TypeError, ValueError):
        return None

for source, match_ids in sources.items():
    for mid in sorted(match_ids):
        candidate_paths = [
            CACHE / f"{mid}.json",
            CACHE / f"mt_{mid}.json",
        ]

        path = next(
            (p for p in candidate_paths if p.exists()),
            None
        )

        if path is None:
            raise SystemExit(f"Cache manquant : {mid}")

        obj = json.loads(path.read_text(encoding="utf-8"))
        data = obj.get("data", obj)
        events = data.get("events", [])

        if not isinstance(events, list):
            raise SystemExit(f"Events invalides : {mid}")

        quality[source]["matchs"] += 1
        quality[source]["events"] += len(events)

        sequences = []

        for event in events:
            if not isinstance(event, dict):
                quality[source]["evenement_non_dict"] += 1
                continue

            typ = normalize(event.get("type"))
            team = normalize(event.get("team"))
            period = normalize(event.get("period"))
            minute = minute_value(event.get("minute"))

            type_counts[source][typ] += 1
            team_counts[source][team] += 1
            period_counts[source][period] += 1

            if minute is None:
                quality[source]["minute_invalide"] += 1
                continue

            if 82 <= minute < 87:
                window_counts[source][f"AVANT87:{typ}"] += 1
            elif minute == 87:
                window_counts[source][f"A87:{typ}"] += 1
            elif minute > 87:
                window_counts[source][f"APRES87:{typ}"] += 1

            seq = event.get("sequence")
            if seq is not None:
                sequences.append(normalize(seq))

            keyword = typ.lower()

            if any(s in keyword for s in (
                "shot", "corner", "goal", "penalty"
            )):
                if typ not in examples[source]:
                    examples[source][typ] = {
                        "match_id": mid,
                        "event": event,
                    }

        quality[source]["sequences_dupliquees"] += (
            len(sequences) - len(set(sequences))
        )

for source in sources:
    print(f"\n{'=' * 60}")
    print("SOURCE :", source)
    print("=" * 60)

    print("\n=== QUALITÉ ===")
    print(dict(quality[source]))

    print("\n=== TYPES D'ÉVÉNEMENTS ===")
    for typ, count in type_counts[source].most_common(35):
        print(f"{typ:30s} {count:5d}")

    print("\n=== VALEURS DE TEAM ===")
    for team, count in team_counts[source].most_common(12):
        print(f"{team[:45]:45s} {count:5d}")

    print("\n=== PÉRIODES ===")
    for period, count in period_counts[source].most_common(12):
        print(f"{period[:35]:35s} {count:5d}")

    print("\n=== ÉVÉNEMENTS 82' À 86' ===")
    before = [
        (k.replace("AVANT87:", ""), n)
        for k, n in window_counts[source].items()
        if k.startswith("AVANT87:")
    ]
    for typ, count in sorted(before, key=lambda x: -x[1])[:25]:
        print(f"{typ:30s} {count:5d}")

    print("\n=== ÉVÉNEMENTS EXACTEMENT À 87' ===")
    at87 = [
        (k.replace("A87:", ""), n)
        for k, n in window_counts[source].items()
        if k.startswith("A87:")
    ]
    for typ, count in sorted(at87, key=lambda x: -x[1])[:25]:
        print(f"{typ:30s} {count:5d}")

    print("\n=== EXEMPLES BRUTS (TIRS / CORNERS / BUTS) ===")
    for typ, example in list(examples[source].items())[:12]:
        print(
            typ,
            json.dumps(example, ensure_ascii=False, default=str)
        )

print("\n=== FIN DE L'AUDIT 177 ===")
print("Uniquement les 183 matchs TRAIN.")
print("Aucun label évalué.")
print("VALIDATION et TEST_FUTUR non consultés.")
print("Aucun fichier existant modifié.")
print("Scanner V3-C inchangé.")
