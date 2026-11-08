#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉTAPE 230 — PILOTE TIMELINES HISTORIQUES V4.1

- Python 3.9
- Maximum 3 appels API
- Aucun label consulté
- Aucun modèle entraîné
- Aucun fichier existant écrasé
- Cache V3-C inchangé
"""

import csv
import importlib.util
import json
import re
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import (
    get_events,
    event_minute,
)

ROOT = Path("outputs/v4_research")

MANIFEST = ROOT / "V41_JULY_PILOT_MANIFEST_V1.csv"
DEST = ROOT / "v41_july_pilot" / "raw_timelines"

SPLIT = Path(
    "outputs/pressure_historical/"
    "pressure_v2_dataset_join/"
    "ALL_320_CHRONO_SPLIT_PLAN_V1.csv"
)

TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"

MAX_REQUESTS = 3
DELAY = 2.0


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


print("=" * 72)
print("ÉTAPE 230 — PILOTE TIMELINES V4.1")
print("=" * 72)

for path in (MANIFEST, SPLIT, TRAIN):
    if not path.is_file():
        raise SystemExit(
            f"Fichier obligatoire absent : {path}"
        )

rows = read_csv(MANIFEST)

if len(rows) != 200:
    raise SystemExit(
        f"200 candidats attendus : {len(rows)}"
    )

ids = [r["match_id"] for r in rows]

if len(set(ids)) != 200:
    raise SystemExit("Doublons dans le manifeste")

split = read_csv(SPLIT)
train = read_csv(TRAIN)

if (
    len(split) != 320
    or len(train) != 278
):
    raise SystemExit(
        "Population historique inattendue"
    )

reserved = {
    r["match_id"] for r in split
} | {
    r["match_id"] for r in train
}

if set(ids) & reserved:
    raise SystemExit(
        "Chevauchement avec les populations protégées"
    )

for row in rows:
    mid = row["match_id"]

    if not re.fullmatch(r"mt_[0-9]+", mid):
        raise SystemExit(
            f"Identifiant invalide : {mid}"
        )

    if row.get("research_status") != "DISCOVERED_ONLY":
        raise SystemExit(
            f"Statut du manifeste inattendu : {mid}"
        )

    if row.get("label_used_for_selection") != "0":
        raise SystemExit(
            f"Sélection non conforme : {mid}"
        )


module_path = Path("diagnostic_live_api.py")

if not module_path.is_file():
    raise SystemExit(
        "diagnostic_live_api.py introuvable"
    )

spec = importlib.util.spec_from_file_location(
    "v41_timeline_api",
    module_path,
)

api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

if not callable(getattr(api, "api_get", None)):
    raise SystemExit(
        "Fonction api_get indisponible"
    )


def analyze(payload):
    events = get_events(payload)

    if events is None:
        return {
            "events_recognized": False,
            "events_total": None,
            "event_types": {},
            "events_77_86": None,
            "events_after_87": None,
            "relevant_events_unknown_minute": None,
        }

    if not isinstance(events, list):
        raise ValueError(
            "Liste d'événements non reconnue"
        )

    types = Counter()
    recent = 0
    after = 0
    unknown = 0

    relevant_types = {
        "goal",
        "shot_on_target",
        "shot_off_target",
        "shot_blocked",
        "corner_kick",
    }

    for event in events:
        if not isinstance(event, dict):
            continue

        typ = str(
            event.get("type") or ""
        ).strip().lower()

        types[typ] += 1

        period = str(
            event.get("period") or ""
        ).strip().lower()

        if period != "second_half":
            continue

        minute = event_minute(event)

        if minute is None:
            if typ in relevant_types:
                unknown += 1
            continue

        if 77 <= minute < 87:
            recent += 1

        if minute > 87:
            after += 1

    return {
        "events_recognized": True,
        "events_total": len(events),
        "event_types": dict(types),
        "events_77_86": recent,
        "events_after_87": after,
        "relevant_events_unknown_minute": unknown,
    }


DEST.mkdir(parents=True, exist_ok=True)

print("\n=== PARAMÈTRES ===")
print("Candidats dans le manifeste :", len(rows))
print("Pilote :", MAX_REQUESTS, "premiers matchs")
print("Délai entre requêtes :", DELAY, "secondes")
print("Destination :", DEST)
print("Aucun label utilisé")

requests_made = 0
saved = 0
errors = 0

for index, row in enumerate(
    rows[:MAX_REQUESTS],
    start=1,
):
    mid = row["match_id"]
    destination = DEST / f"{mid}.json"

    print("\n" + "=" * 65)
    print(f"MATCH {index} : {mid}")
    print(
        row["home_team"],
        "vs",
        row["away_team"],
    )
    print("=" * 65)

    if destination.exists():
        print("SKIP : fichier déjà présent")
        continue

    if requests_made:
        time.sleep(DELAY)

    requests_made += 1

    try:
        payload = api.api_get(
            f"/football/matches/{mid}/timeline"
        )

        if payload is None:
            raise ValueError(
                "Réponse API vide"
            )

        if not isinstance(payload, (dict, list)):
            raise ValueError(
                "Format JSON inattendu"
            )

        # Un ID explicite contradictoire est bloquant.
        if isinstance(payload, dict):
            inner = payload.get("data")

            source = (
                inner
                if isinstance(inner, dict)
                else payload
            )

            returned_id = (
                source.get("match_id")
                or source.get("id")
            )

            if (
                returned_id is not None
                and str(returned_id) != mid
            ):
                raise ValueError(
                    "Identifiant de réponse différent "
                    f"du match demandé : {returned_id}"
                )

        diagnostic = analyze(payload)

        record = {
            "match_id": mid,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_id": row["home_id"],
            "away_id": row["away_id"],
            "kickoff_utc": row["kickoff_utc"],
            "retrieved_at_utc": utc_now(),
            "source": "HISTORICAL_API",
            "research_status": (
                "RAW_TIMELINE_AUDIT_REQUIRED"
            ),
            "training_certified": False,
            "live_certified": False,
            "label_used_for_selection": False,
            "diagnostic": diagnostic,
            "timeline_response": payload,
        }

        # Création exclusive : jamais d'écrasement.
        with destination.open(
            "x", encoding="utf-8"
        ) as f:
            json.dump(
                record,
                f,
                ensure_ascii=False,
            )

        saved += 1

        print("Fichier enregistré :", destination)
        print(
            "Événements :",
            diagnostic["events_total"],
        )
        print(
            "Événements entre 77' et 86' :",
            diagnostic["events_77_86"],
        )
        print(
            "Événements après 87' :",
            diagnostic["events_after_87"],
        )
        print(
            "Types d'événements :",
            diagnostic["event_types"],
        )

        if not diagnostic["events_recognized"]:
            print(
                "ATTENTION : structure des événements "
                "non reconnue"
            )

    except Exception as exc:
        errors += 1

        print(
            "ERREUR :",
            type(exc).__name__,
            str(exc),
        )

        # Une limitation de l'API ne doit pas
        # déclencher de nouvelles requêtes.
        message = str(exc).lower()

        if (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        ):
            print(
                "ARRÊT : limitation API possible"
            )
            break


print("\n" + "=" * 72)
print("BILAN — ÉTAPE 230")
print("=" * 72)

print("Requêtes effectuées :", requests_made)
print("Timelines enregistrées :", saved)
print("Erreurs :", errors)
print("Répertoire :", DEST)

print("\n=== LIMITES ===")
print("Les fichiers sont des réponses brutes.")
print("Les événements après 87' sont normaux dans")
print("une timeline historique, mais seront exclus")
print("des futures caractéristiques prédictives.")
print("Aucune complétude certifiée.")
print("Aucun label reconstruit.")
print("Aucune nouvelle donnée d'entraînement créée.")

print("\n=== PROTECTIONS ===")
print("Aucun fichier existant écrasé.")
print("Aucun modèle modifié.")
print("Aucun cache historique V3-C modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 230 TERMINÉE.")
