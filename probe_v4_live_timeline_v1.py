#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from v3c_pressure_features import (
    get_events,
    event_minute,
)

print("=" * 72)
print("ÉTAPE 216 — DIAGNOSTIC TIMELINE LIVE V4")
print("=" * 72)

SOURCE = Path("diagnostic_live_api.py")

if not SOURCE.is_file():
    raise SystemExit("Client API introuvable")

spec = importlib.util.spec_from_file_location(
    "v4_live_probe_client", SOURCE
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def matches_from(payload):
    if isinstance(payload, list):
        if all(isinstance(x, Mapping) for x in payload):
            return payload
        return None

    if isinstance(payload, Mapping):
        for key in (
            "matches", "data", "results",
            "response", "items", "result"
        ):
            if key in payload:
                found = matches_from(payload[key])
                if found is not None:
                    return found

    return None


def utc_timestamp(value):
    if not isinstance(value, str):
        return None

    try:
        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def valid_minute(value):
    if isinstance(value, bool):
        return None

    try:
        n = int(value)
        if str(value).strip() != str(n):
            return None
        return n
    except (ValueError, TypeError):
        return None


def valid_score(value):
    n = valid_minute(value)
    return n if n is not None and n >= 0 else None


def team_id(team):
    if not isinstance(team, Mapping):
        return None
    value = team.get("id")
    return str(value) if value else None


print("\n=== REQUÊTE 1 : MATCHS LIVE ===")

data = api.api_get(
    "/football/matches",
    params={
        "status": "live",
        "page": 1,
        "per_page": 100,
    },
)

matches = matches_from(data)

if matches is None:
    raise SystemExit("Liste LIVE non reconnue")

print("Matchs retournés :", len(matches))

now = datetime.now(timezone.utc)
eligible = []
rejections = Counter()

for match in matches:
    if str(match.get("status", "")).lower() != "live":
        rejections["STATUS_NON_LIVE"] += 1
        continue

    live = match.get("live")

    if not isinstance(live, Mapping):
        rejections["LIVE_ABSENT"] += 1
        continue

    period = live.get("period")
    minute = valid_minute(
        live.get("elapsed_minutes")
    )

    if period != "second_half":
        rejections["AUTRE_PERIODE"] += 1
        continue

    if minute is None or not 46 <= minute <= 87:
        rejections["MINUTE_HORS_PLAGE"] += 1
        continue

    updated = utc_timestamp(live.get("updated_at"))

    if updated is None:
        rejections["HORODATAGE_ABSENT"] += 1
        continue

    age = (now - updated).total_seconds()

    if not -10 <= age <= 120:
        rejections["MISE_A_JOUR_TROP_ANCIENNE"] += 1
        continue

    score = match.get("score")

    if not isinstance(score, Mapping):
        rejections["SCORE_ABSENT"] += 1
        continue

    home_score = valid_score(score.get("home"))
    away_score = valid_score(score.get("away"))

    if home_score is None or away_score is None:
        rejections["SCORE_INVALIDE"] += 1
        continue

    home = match.get("home_team")
    away = match.get("away_team")

    if not team_id(home) or not team_id(away):
        rejections["EQUIPES_NON_IDENTIFIEES"] += 1
        continue

    mid = match.get("id")

    if not mid:
        rejections["MATCH_ID_ABSENT"] += 1
        continue

    eligible.append({
        "match_id": str(mid),
        "minute": minute,
        "home_score": home_score,
        "away_score": away_score,
        "home_id": team_id(home),
        "away_id": team_id(away),
        "home_name": home.get("name"),
        "away_name": away.get("name"),
        "updated_at": updated.isoformat(),
        "age_seconds": round(age, 1),
    })

print("\n=== MATCHS ADMISSIBLES AU DIAGNOSTIC ===")
print("Nombre :", len(eligible))
print("Motifs d'exclusion :", dict(rejections))

for item in sorted(
    eligible, key=lambda x: -x["minute"]
)[:10]:
    print(
        f"{item['match_id']} | "
        f"{item['minute']}' | "
        f"{item['home_name']} "
        f"{item['home_score']}-"
        f"{item['away_score']} "
        f"{item['away_name']} | "
        f"fraîcheur={item['age_seconds']}s"
    )

if not eligible:
    print("\nAucun match admissible actuellement.")
    print("Aucune requête timeline effectuée.")
    print("Relancer ultérieurement si nécessaire.")
    raise SystemExit(0)

# Préférer une rencontre proche de la 87e minute.
selected = min(
    eligible,
    key=lambda x: abs(x["minute"] - 87)
)

mid = selected["match_id"]

print("\n=== MATCH SÉLECTIONNÉ ===")
print("ID :", mid)
print("Minute annoncée :", selected["minute"])
print(
    "Score :",
    f"{selected['home_score']}-{selected['away_score']}"
)
print("Mise à jour :", selected["updated_at"])

print("\n=== REQUÊTE 2 : TIMELINE LIVE ===")

request_start = datetime.now(timezone.utc)

timeline = api.api_get(
    f"/football/matches/{mid}/timeline"
)

received_at = datetime.now(timezone.utc)

events = get_events(timeline)

print("Requête lancée à :", request_start.isoformat())
print("Réponse reçue à :", received_at.isoformat())

if events is None:
    print("Timeline non reconnue.")
    raise SystemExit(0)

print("Nombre total d'événements :", len(events))

types = Counter()
periods = Counter()
recent_types = Counter()
future = 0
invalid_minute = 0

for event in events:
    typ = str(event.get("type", "INCONNU"))
    period = str(event.get("period", "INCONNUE"))

    types[typ] += 1
    periods[period] += 1

    minute = event_minute(event)

    if minute is None:
        invalid_minute += 1
        continue

    if minute >= selected["minute"]:
        future += 1

    if (
        period == "second_half"
        and 82 <= minute < 87
    ):
        recent_types[typ] += 1

print("\n=== TYPES D'ÉVÉNEMENTS ===")

for name, count in types.most_common(25):
    print(f"{name:30s} : {count}")

print("\n=== PÉRIODES ===")
print(dict(periods))

print("\n=== ÉVÉNEMENTS 82' À 86' ===")

if recent_types:
    for name, count in recent_types.most_common():
        print(f"{name:30s} : {count}")
else:
    print("Aucun événement observé dans cette fenêtre.")

print("\n=== CONTRÔLES ===")
print("Minutes d'événements invalides :", invalid_minute)
print(
    "Événements à une minute >= minute annoncée :",
    future
)

print("\n=== INTERPRÉTATION ===")

if future:
    print(
        "ATTENTION : la timeline contient des événements "
        "qui ne précèdent pas la minute annoncée."
    )

if not any(
    t in types
    for t in (
        "shot_on_target",
        "shot_off_target",
        "shot_blocked",
        "corner_kick",
    )
):
    print(
        "ATTENTION : aucun événement explicite "
        "de tir/corner identifié dans cette timeline."
    )

print(
    "La couverture des tirs et corners n'est PAS "
    "certifiée par ce diagnostic."
)

print("\n=== GARANTIES ===")
print("Au maximum deux requêtes API prévues.")
print("Aucun fichier de données enregistré.")
print("Aucune prédiction.")
print("Aucun label.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 216 TERMINÉE.")
